use half::f16;
use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;
use turbovec::{SearchCache, TurboQuantIndex};

const DIM: usize = 768;
const K: usize = 10;
const SELF_QUERIES: usize = 1_000;
const RANDOM_QUERIES: usize = 1_000;
const VECTOR_RAM_CAP_50K_BYTES: usize = 30 * 1024 * 1024;
const VECTOR_RAM_CAP_100K_BYTES: usize = 50 * 1024 * 1024;
const READ_BUFFER_BYTES: usize = 1024 * 1024;
const QUERY_BATCH: usize = 64;
const SIMD_BLOCK: usize = 32;
const TIMED_PROBES_PER_SET: usize = 8;
const QUERY_MAJOR_QUERY_BYTES: usize = DIM * std::mem::size_of::<f32>();
const QUANT_RERANK_CANDIDATES: usize = 256;

// HNSW is deliberately a compact, disk-backed family in this benchmark. The
// graph is built over the persisted FP16 store, then written as flat u32
// adjacency arrays. M=16 gives 32 base-layer neighbours (the usual HNSW
// 2*M base-layer allowance). The larger ef_search is intentional: the graph
// must recover roughly 98% of the exact random top-10 while payload vectors
// remain disk-backed.
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 128;
const HNSW_EF_SEARCH: usize = 1024;
const HNSW_LEGACY_EF_SEARCH: usize = 96;
const HNSW_MAX_LAYER: usize = 8;
const HNSW_VECTOR_CACHE_VECTORS: usize = 4096;
const HNSW_BATCH_VECTOR_CACHE_VECTORS: usize = 256;
const HNSW_CACHE_BLOCK_VECTORS: usize = 8;
const HNSW_RAW_READ_SPAN_VECTORS: usize = 8;
const EXACT_TRUTH_VECTOR_TILE: usize = 64;
const HNSW_TRUTH_MAGIC: &[u8; 8] = b"TQTRUTH1";
const HNSW_MAGIC: &[u8; 8] = b"TVHG001\0";

#[derive(Deserialize)]
struct DatasetInput {
    id: String,
    label: String,
    path: String,
    vectors: usize,
}

#[derive(Clone, Copy)]
struct Hit {
    score: f32,
    idx: usize,
}

#[derive(Clone, Copy)]
struct QueryMajorTiming {
    total_ms: f64,
    first_ms: f64,
    warm_ms_per_query: f64,
    count: usize,
}

impl QueryMajorTiming {
    fn from_samples(total_ms: f64, first_ms: f64, count: usize) -> Self {
        let warm_count = count.saturating_sub(1);
        Self {
            total_ms,
            first_ms,
            warm_ms_per_query: if warm_count == 0 {
                0.0
            } else {
                (total_ms - first_ms) / warm_count as f64
            },
            count,
        }
    }

    fn average_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ms / self.count as f64
        }
    }
}

#[derive(Serialize)]
struct Row {
    index: String,
    bits: String,
    self_r1: String,
    self_r10: String,
    random_r10: String,
    index_ms: String,
    prepare_ms: String,
    write_ms: String,
    self_search_ms: String,
    random_search_ms: String,
    ms_per_query: String,
    first_query_ms: String,
    warm_ms_per_query: String,
    index_rom: String,
    data_store: String,
    vector_staging: String,
    vector_ram: String,
}

#[derive(Serialize)]
struct DatasetTable {
    dataset: String,
    vectors: String,
    vector_ram_cap: String,
    raw_chunk_vectors: String,
    quant_chunk_vectors: String,
    graph_ram: String,
    rows: Vec<Row>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BenchmarkMode {
    QueryMajor,
    Batched,
    HnswQueryMajor,
    HnswBatched,
}

impl BenchmarkMode {
    fn key(self) -> &'static str {
        match self {
            Self::QueryMajor => "query_major",
            Self::Batched => "batched_throughput",
            Self::HnswQueryMajor => "hnsw_query_major",
            Self::HnswBatched => "hnsw_batched_throughput",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::QueryMajor => "A) Real traffic / query-major",
            Self::Batched => "B) Batched throughput",
            Self::HnswQueryMajor => "A) HNSW real traffic / query-major",
            Self::HnswBatched => "B) HNSW batched throughput",
        }
    }

    fn is_hnsw(self) -> bool {
        matches!(self, Self::HnswQueryMajor | Self::HnswBatched)
    }

    fn is_batched(self) -> bool {
        matches!(self, Self::Batched | Self::HnswBatched)
    }

    fn family_label(self) -> &'static str {
        if self.is_hnsw() { "HNSW" } else { "FlatIndex" }
    }
}

#[derive(Serialize)]
struct Report {
    mode: String,
    mode_label: String,
    index_family: String,
    timing_model: String,
    datasets: String,
    dim: usize,
    self_queries: usize,
    random_queries: usize,
    vector_ram_caps: String,
    raw_chunk_vectors: String,
    raw_chunk_ram: String,
    raw_fp32_storage: String,
    methods: String,
    graph_params: String,
    graph_ram: String,
    notes: Vec<String>,
    tables: Vec<DatasetTable>,
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runBenchmark(
    mut env: JNIEnv,
    _class: JClass,
    datasets_json: JString,
    output_dir: JString,
) -> jstring {
    native_entry(&mut env, datasets_json, output_dir, BenchmarkMode::QueryMajor)
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runQueryMajor(
    mut env: JNIEnv,
    _class: JClass,
    datasets_json: JString,
    output_dir: JString,
) -> jstring {
    native_entry(&mut env, datasets_json, output_dir, BenchmarkMode::QueryMajor)
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runBatched(
    mut env: JNIEnv,
    _class: JClass,
    datasets_json: JString,
    output_dir: JString,
) -> jstring {
    native_entry(&mut env, datasets_json, output_dir, BenchmarkMode::Batched)
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runHnswQueryMajor(
    mut env: JNIEnv,
    _class: JClass,
    datasets_json: JString,
    output_dir: JString,
) -> jstring {
    native_entry(&mut env, datasets_json, output_dir, BenchmarkMode::HnswQueryMajor)
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runHnswBatched(
    mut env: JNIEnv,
    _class: JClass,
    datasets_json: JString,
    output_dir: JString,
) -> jstring {
    native_entry(&mut env, datasets_json, output_dir, BenchmarkMode::HnswBatched)
}

fn native_entry(
    env: &mut JNIEnv,
    datasets_json: JString,
    output_dir: JString,
    mode: BenchmarkMode,
) -> jstring {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let datasets_json: String = env
            .get_string(&datasets_json)
            .map_err(|e| e.to_string())?
            .into();
        let output_dir: String = env
            .get_string(&output_dir)
            .map_err(|e| e.to_string())?
            .into();
        run_mode(&datasets_json, Path::new(&output_dir), mode)
    }));

    let text = match result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => format!("Benchmark error: {e}"),
        Err(_) => "Benchmark panic in native Rust code".to_string(),
    };
    env.new_string(text).expect("new Java string").into_raw()
}

fn run_mode(
    datasets_json: &str,
    output_dir: &Path,
    mode: BenchmarkMode,
) -> Result<String, String> {
    let datasets: Vec<DatasetInput> =
        serde_json::from_str(datasets_json).map_err(|e| format!("parse datasets: {e}"))?;
    if datasets.is_empty() {
        return Err("no downloaded datasets were supplied".to_string());
    }

    let mut tables = Vec::new();
    let mut labels = Vec::new();
    for dataset in datasets {
        let rows = match mode {
            BenchmarkMode::QueryMajor => bench_dataset_query_major(&dataset, output_dir)?,
            BenchmarkMode::Batched => bench_dataset_batched(&dataset, output_dir)?,
            BenchmarkMode::HnswQueryMajor => bench_dataset_hnsw_query_major(&dataset, output_dir)?,
            BenchmarkMode::HnswBatched => bench_dataset_hnsw_batched(&dataset, output_dir)?,
        };
        let cap = vector_ram_cap_bytes(dataset.vectors);
        let raw_chunk = if mode.is_hnsw() {
            "n/a (HNSW candidate reads)".to_string()
        } else {
            match mode {
                BenchmarkMode::QueryMajor => raw_chunk_vectors_query_major(cap).to_string(),
                BenchmarkMode::Batched => raw_chunk_vectors(cap).to_string(),
                _ => unreachable!(),
            }
        };
        labels.push(format!("{} ({})", dataset.label, human_count(dataset.vectors)));
        tables.push(DatasetTable {
            dataset: dataset.label,
            vectors: human_count(dataset.vectors),
            vector_ram_cap: format!("{} search cap", human_bytes(cap as u64)),
            raw_chunk_vectors: raw_chunk,
            quant_chunk_vectors: if mode.is_hnsw() {
                "n/a (HNSW graph)".to_string()
            } else {
                format!(
                    "8-bit {}, 4-bit {}",
                    quant_chunk_vectors(cap, 8),
                    quant_chunk_vectors(cap, 4)
                )
            },
            graph_ram: if mode.is_hnsw() {
                hnsw_graph_ram_budget(cap, mode)
            } else {
                "n/a (FlatIndex)".to_string()
            },
            rows,
        });
    }

    let cap_50k = vector_ram_cap_bytes(50_000);
    let cap_100k = vector_ram_cap_bytes(100_000);
    let flat_raw_chunk = |cap: usize| match mode {
        BenchmarkMode::QueryMajor => raw_chunk_vectors_query_major(cap),
        BenchmarkMode::Batched => raw_chunk_vectors(cap),
        _ => 0,
    };
    let report = Report {
        mode: mode.key().to_string(),
        mode_label: mode.label().to_string(),
        index_family: mode.family_label().to_string(),
        timing_model: match mode {
            BenchmarkMode::QueryMajor => "one query scans all bounded chunks/ranges before the next query".to_string(),
            BenchmarkMode::Batched => "each bounded chunk/range is reused across the full query batch".to_string(),
            BenchmarkMode::HnswQueryMajor => "one query traverses the resident HNSW graph and reads candidate vectors before the next query".to_string(),
            BenchmarkMode::HnswBatched => "the resident HNSW graph is shared while independent queries search in parallel".to_string(),
        },
        datasets: labels.join(", "),
        dim: DIM,
        self_queries: SELF_QUERIES,
        random_queries: RANDOM_QUERIES,
        vector_ram_caps: format!(
            "50K: {} search cap; 100K: {} search cap",
            human_bytes(cap_50k as u64),
            human_bytes(cap_100k as u64)
        ),
        raw_chunk_vectors: if mode.is_hnsw() {
            "50K: n/a (graph candidate reads); 100K: n/a (graph candidate reads)".to_string()
        } else {
            format!(
                "50K: {}; 100K: {}",
                flat_raw_chunk(cap_50k),
                flat_raw_chunk(cap_100k)
            )
        },
        raw_chunk_ram: if mode.is_hnsw() {
            "50K: n/a; 100K: n/a".to_string()
        } else {
            let bytes_50k = (flat_raw_chunk(cap_50k) * DIM * 4) as u64;
            let bytes_100k = (flat_raw_chunk(cap_100k) * DIM * 4) as u64;
            format!(
                "50K: {}; 100K: {}",
                human_bytes(bytes_50k),
                human_bytes(bytes_100k)
            )
        },
        raw_fp32_storage: "50K: disk-backed; 100K: disk-backed".to_string(),
        methods: if mode.is_hnsw() {
            "HNSW FP32, FP16, 8-bit, and 4-bit payloads over one compact FP16 navigation graph".to_string()
        } else {
            "FlatIndex FP32, FP16, 8-bit, and 4-bit".to_string()
        },
        graph_params: if mode.is_hnsw() {
            format!(
                "M={}, efConstruction={}, efSearch={}, max layers={}, base degree <= {}",
                HNSW_M,
                HNSW_EF_CONSTRUCTION,
                HNSW_EF_SEARCH,
                HNSW_MAX_LAYER,
                2 * HNSW_M
            )
        } else {
            "n/a (FlatIndex)".to_string()
        },
        graph_ram: if mode.is_hnsw() {
            format!(
                "50K: {}; 100K: {}",
                hnsw_graph_ram_budget(cap_50k, mode),
                hnsw_graph_ram_budget(cap_100k, mode)
            )
        } else {
            "n/a (FlatIndex)".to_string()
        },
        notes: vec![
            "Recall is measured against exact FP32 top-10 over the same dataset size.".to_string(),
            "Random R@1 is intentionally omitted; R@10 is the mean fraction of exact FP32 top-10 neighbors recovered by the approximate top-10.".to_string(),
            "Random queries are deterministic normalized blends of two base vectors; self queries are the first 1000 base vectors.".to_string(),
            format!(
                "Search working-set RAM is capped at {} for 50K and {} for 100K; app/UI memory, Rust/runtime/dependency code, allocator overhead, and OS file cache are excluded.",
                human_bytes(cap_50k as u64),
                human_bytes(cap_100k as u64)
            ),
            if mode.is_hnsw() {
                "HNSW keeps every payload store on disk. One shared FP16 navigation graph selects candidates; FP32/FP16 score candidates from their disk-backed payload, while 8/4-bit score compressed candidates and exact-rerank the bounded candidate set with raw FP32 reads. The graph build's temporary full FP16 staging is not part of steady-state search RAM.".to_string()
            } else {
                "All methods are disk-backed: FP32 loads bounded raw f32 chunks, FP16 loads bounded raw f16 bits and converts during the dot product, while TurboQuant loads one persisted compressed range plus its blocked SIMD copy and search caches.".to_string()
            },
            if mode == BenchmarkMode::QueryMajor {
                format!(
                    "FlatIndex A 8-bit and 4-bit rows expand each compressed range to up to {} approximate candidates, then exact-rerank those candidate ids from the disk-backed raw FP32 store; the bounded candidate heap is not a resident copy of the database.",
                    QUANT_RERANK_CANDIDATES
                )
            } else if mode == BenchmarkMode::Batched {
                format!(
                    "FlatIndex B uses direct range-major scoring for FP32/FP16/8-bit. The 4-bit row also expands each range to up to {} candidates and exact-reranks them from the disk-backed raw FP32 store so its displayed R@10 is not the raw 4-bit top-10.",
                    QUANT_RERANK_CANDIDATES
                )
            } else {
                "FlatIndex B reports the direct chunk/range-major throughput path; it does not reuse query-major timing probes.".to_string()
            },
            match mode {
                BenchmarkMode::QueryMajor => format!(
                    "Query-major timing uses {} independent self + {} independent random probes: one query scans every bounded chunk or compressed range, completes top-10, and only then does the next query start. Recall still uses all {} self + {} random queries. A uses the single-query direct-heap scorer and persisted blocked TurboQuant sidecars; range loading is included in each probe.",
                    TIMED_PROBES_PER_SET,
                    TIMED_PROBES_PER_SET,
                    SELF_QUERIES,
                    RANDOM_QUERIES
                ),
                BenchmarkMode::Batched => format!(
                    "Batched timing is chunk/range-major: each bounded chunk or compressed range is loaded once and searched against all {} self + {} random queries before the next range starts. ms/query divides the full {}-query search by {}. Recall uses the same query sets.",
                    SELF_QUERIES,
                    RANDOM_QUERIES,
                    SELF_QUERIES + RANDOM_QUERIES,
                    SELF_QUERIES + RANDOM_QUERIES
                ),
                BenchmarkMode::HnswQueryMajor => format!(
                    "HNSW query-major timing uses {} independent self + {} independent random probes. Each probe starts at the graph entry point, greedily descends upper layers, then performs efSearch={} best-first base-layer expansion; candidate FP16 vectors are read on demand and the query finishes before the next starts. Recall uses all {} self + {} random queries.",
                    TIMED_PROBES_PER_SET,
                    TIMED_PROBES_PER_SET,
                    HNSW_EF_SEARCH,
                    SELF_QUERIES,
                    RANDOM_QUERIES
                ),
                BenchmarkMode::HnswBatched => format!(
                    "HNSW batched timing searches all {} self + {} random queries in parallel against one resident graph. Each worker reuses a small candidate-vector cache; ms/query divides the full {}-query batch by {}.",
                    SELF_QUERIES,
                    RANDOM_QUERIES,
                    SELF_QUERIES + RANDOM_QUERIES,
                    SELF_QUERIES + RANDOM_QUERIES
                ),
            },
            if mode.is_hnsw() {
                format!(
                    "HNSW graph RAM is budgeted with graph adjacency plus a {}-vector FP16 candidate cache and search scratch; the combined accounted resident graph/search working set remains within {}.",
                    HNSW_VECTOR_CACHE_VECTORS,
                    human_bytes(cap_100k as u64)
                )
            } else {
                format!(
                    "Raw FP32 staging is {} for 50K and {} for 100K; the full raw stores remain on disk.",
                    human_bytes((flat_raw_chunk(cap_50k) * DIM * 4) as u64),
                    human_bytes((flat_raw_chunk(cap_100k) * DIM * 4) as u64)
                )
            },
            if mode.is_hnsw() {
                "HNSW is a graph search over persisted FP16; it is not a quantized TurboQuant flat-scan row. The graph is compacted after build and reused by both HNSW A and HNSW B.".to_string()
            } else {
                "All four methods are flat scans over every vector; HNSW is not part of this FlatIndex report.".to_string()
            },
            if mode.is_hnsw() {
                "HNSW build parameters are chosen for the recall target: M=16, efConstruction=128, efSearch=1024. The higher efSearch expands more graph candidates and therefore trades some HNSW latency for roughly 98% random R@10.".to_string()
            } else {
                "FlatIndex FP32 scans the raw f32 database, FlatIndex FP16 scans a persisted IEEE FP16 copy with fused half-to-f32 conversion during scoring, and FlatIndex 8-bit/4-bit use TurboQuant compressed flat scans.".to_string()
            },
            if mode.is_hnsw() {
                "Graph adjacency is resident; vector payloads remain disk-backed. The candidate cache is an optimization, not a resident copy of the 50K/100K database.".to_string()
            } else {
                "On arm64-v8a, TurboQuant's 8-bit path uses the block-major byte-code scorer; 4-bit uses the NEON lookup-table path.".to_string()
            },
            if mode.is_hnsw() {
                "The HNSW graph file and FP16 vector store are persistent in the app's files directory. A later HNSW A/B run reuses them when the graph header and vector-store size match.".to_string()
            } else {
                format!(
                    "TurboQuant reads persisted blocked .tvpb ranges in bounded chunks (8-bit: up to {}, 4-bit: up to {} vectors), searches each range, merges top-10 results, and releases it before the next range.",
                    quant_chunk_vectors_query_major(cap_100k, 8),
                    quant_chunk_vectors_query_major(cap_100k, 4)
                )
            },
            format!(
                "{}; the per-dataset 30/50 MiB cap is a steady-state pure-vector working-set budget, so app/UI/runtime/dependency memory, OS file cache, persisted ROM, and temporary full-index build peaks are outside this KPI.",
                match mode {
                    BenchmarkMode::QueryMajor => "Prep/load ms includes bounded range loading and preparation, and that per-request work is included in ms/query",
                    BenchmarkMode::Batched => "Prep/load ms is reported separately from warm batch ms/query; the active range is prepared once and reused for the query batch",
                    BenchmarkMode::HnswQueryMajor => "Prep/load ms includes persisted FP16/graph load and cache setup; candidate reads and graph traversal are included in ms/query",
                    BenchmarkMode::HnswBatched => "Prep/load ms includes persisted FP16/graph load and cache setup; the parallel graph-search batch is reported as ms/query",
                }
            ),
        ],
        tables,
    };
    serde_json::to_string(&report).map_err(|e| format!("serialize report: {e}"))
}

fn bench_dataset_query_major(dataset: &DatasetInput, output_dir: &Path) -> Result<Vec<Row>, String> {
    if dataset.vectors < SELF_QUERIES {
        return Err(format!("{} has fewer than {} vectors", dataset.label, SELF_QUERIES));
    }
    let vector_ram_cap = vector_ram_cap_bytes(dataset.vectors);
    let vector_path = Path::new(&dataset.path);
    validate_vector_file(vector_path, dataset.vectors)?;

    // Every method is disk-backed. The search cap covers the current raw or
    // decoded chunk plus query and search scratch; the full dataset is never
    // loaded into RAM at once.
    let self_queries = load_vector_range(
        vector_path,
        0,
        SELF_QUERIES,
        vector_ram_cap,
        query_vector_bytes(),
    )?;
    let random_queries = make_random_queries(
        vector_path,
        dataset.vectors,
        RANDOM_QUERIES,
        vector_ram_cap,
    )?;

    let self_truth = exact_topk_file_batched(vector_path, dataset.vectors, &self_queries, vector_ram_cap)?;
    let random_truth = exact_topk_file_batched(vector_path, dataset.vectors, &random_queries, vector_ram_cap)?;
    let fp32_timing = time_fp32_probes(
        vector_path,
        dataset.vectors,
        &self_queries,
        &random_queries,
        vector_ram_cap,
    )?;

    let mut rows = Vec::new();
    rows.push(fp32_row(
        fp32_timing,
        (dataset.vectors as u64) * (DIM as u64) * 4,
        vector_ram_cap,
    ));
    rows.push(bench_fp16_query_major(
        dataset,
        vector_path,
        &self_queries,
        &random_queries,
        &self_truth,
        &random_truth,
        output_dir,
        vector_ram_cap,
    )?);
    for bit_width in [8usize, 4] {
        rows.push(bench_quant_query_major(
            dataset,
            bit_width,
            vector_path,
            &self_queries,
            &random_queries,
            &self_truth,
            &random_truth,
            output_dir,
            vector_ram_cap,
        )?);
    }
    Ok(rows)
}

fn bench_dataset_batched(dataset: &DatasetInput, output_dir: &Path) -> Result<Vec<Row>, String> {
    if dataset.vectors < SELF_QUERIES {
        return Err(format!("{} has fewer than {} vectors", dataset.label, SELF_QUERIES));
    }
    let vector_ram_cap = vector_ram_cap_bytes(dataset.vectors);
    let vector_path = Path::new(&dataset.path);
    validate_vector_file(vector_path, dataset.vectors)?;

    let self_queries = load_vector_range(
        vector_path,
        0,
        SELF_QUERIES,
        vector_ram_cap,
        query_vector_bytes(),
    )?;
    let random_queries = make_random_queries(
        vector_path,
        dataset.vectors,
        RANDOM_QUERIES,
        vector_ram_cap,
    )?;

    // Recall is computed with the same chunk-major scheduling as the timed
    // batch path. The vector store remains disk-backed; only the query batch
    // and one active chunk/range are resident at once.
    let fp32_start = Instant::now();
    let self_truth = exact_topk_file_batched(vector_path, dataset.vectors, &self_queries, vector_ram_cap)?;
    let random_truth = exact_topk_file_batched(vector_path, dataset.vectors, &random_queries, vector_ram_cap)?;
    let fp32_ms = fp32_start.elapsed().as_secs_f64() * 1000.0;

    let mut rows = Vec::new();
    rows.push(fp32_row_batched(
        fp32_ms,
        (dataset.vectors as u64) * (DIM as u64) * 4,
        vector_ram_cap,
    ));
    rows.push(bench_fp16_batched(
        dataset,
        vector_path,
        &self_queries,
        &random_queries,
        &self_truth,
        &random_truth,
        output_dir,
        vector_ram_cap,
    )?);
    for bit_width in [8usize, 4] {
        rows.push(bench_quant_batched(
            dataset,
            bit_width,
            vector_path,
            &self_queries,
            &random_queries,
            &self_truth,
            &random_truth,
            output_dir,
            vector_ram_cap,
        )?);
    }
    Ok(rows)
}

fn bench_dataset_hnsw_query_major(
    dataset: &DatasetInput,
    output_dir: &Path,
) -> Result<Vec<Row>, String> {
    if dataset.vectors < SELF_QUERIES {
        return Err(format!("{} has fewer than {} vectors", dataset.label, SELF_QUERIES));
    }
    let cap = vector_ram_cap_bytes(dataset.vectors);
    let vector_path = Path::new(&dataset.path);
    validate_vector_file(vector_path, dataset.vectors)?;

    let self_queries = load_vector_range(
        vector_path,
        0,
        SELF_QUERIES,
        cap,
        query_vector_bytes(),
    )?;
    let random_queries = make_random_queries(vector_path, dataset.vectors, RANDOM_QUERIES, cap)?;
    let (self_truth, random_truth) = load_or_compute_hnsw_truth(
        dataset,
        vector_path,
        &self_queries,
        &random_queries,
        cap,
        output_dir,
    )?;

    let fp16_path = hnsw_fp16_path(output_dir, &dataset.id);
    let (store_index_ms, store_write_ms) = ensure_fp16_store(
        vector_path,
        dataset.vectors,
        cap,
        &fp16_path,
    )?;
    let graph_path = hnsw_graph_path(output_dir, &dataset.id);
    let (graph, graph_index_ms, graph_write_ms, graph_load_ms) = ensure_hnsw_graph(
        &fp16_path,
        dataset.vectors,
        &graph_path,
    )?;
    validate_hnsw_budget(&graph, cap, HNSW_VECTOR_CACHE_VECTORS)?;

    let fp16_bytes = fp16_path.metadata().map(|m| m.len()).unwrap_or(0);
    let shared_index_ms = store_index_ms + graph_index_ms;
    let shared_write_ms = store_write_ms + graph_write_ms;
    let shared_prepare_ms = graph_load_ms;
    let shared_store = format!(
        "disk-backed FP16 navigation store + resident graph ({} cache)",
        human_bytes((HNSW_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
    );
    let mut rows = Vec::new();

    let mut recall_navigation =
        HnswSearcher::open(&fp16_path, &graph, HNSW_VECTOR_CACHE_VECTORS)?;
    let recall_candidates = HnswRecallCandidates {
        self_queries: hnsw_navigation_candidates(&mut recall_navigation, &self_queries)?,
        random_queries: hnsw_navigation_candidates(&mut recall_navigation, &random_queries)?,
    };

    // The FP16 navigation payload is also a useful baseline: it measures the
    // graph itself without adding a second payload scorer.
    let mut fp16_searcher = HnswSearcher::open(&fp16_path, &graph, HNSW_VECTOR_CACHE_VECTORS)?;
    let fp16_metrics = benchmark_hnsw_query_major_method(
        &mut fp16_searcher,
        &self_queries,
        &random_queries,
        &self_truth,
        &random_truth,
        &recall_candidates,
    )?;
    rows.push(hnsw_method_row(
        "16",
        fp16_metrics,
        &graph,
        cap,
        shared_index_ms,
        shared_prepare_ms,
        shared_write_ms,
        graph.resident_bytes() as u64 + fp16_bytes,
        &shared_store,
        &format!(
            "{} FP16 navigation cache",
            human_bytes((HNSW_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
        ),
    ));

    // FP32 is disk-backed too. The graph only chooses candidates; exact FP32
    // scoring is performed from the raw file for those candidates.
    let mut fp32_searcher = HnswRawSearcher::open(
        &fp16_path,
        vector_path,
        &graph,
        HNSW_VECTOR_CACHE_VECTORS,
    )?;
    let fp32_metrics = benchmark_hnsw_query_major_method(
        &mut fp32_searcher,
        &self_queries,
        &random_queries,
        &self_truth,
        &random_truth,
        &recall_candidates,
    )?;
    rows.push(hnsw_method_row(
        "32",
        fp32_metrics,
        &graph,
        cap,
        graph_index_ms,
        graph_load_ms,
        graph_write_ms,
        graph.resident_bytes() as u64 + fp16_bytes
            + (dataset.vectors as u64) * (DIM as u64) * 4,
        "disk-backed raw FP32 payload + shared FP16 navigation graph",
        &format!(
            "{} FP16 navigation cache + one FP32 candidate scratch",
            human_bytes((HNSW_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
        ),
    ));

    for bit_width in [8usize, 4] {
        let (quant_index_ms, quant_write_ms, quant_rom) = ensure_quant_store(
            vector_path,
            dataset.vectors,
            cap,
            output_dir,
            &dataset.id,
            bit_width,
        )?;
        let cache_start = Instant::now();
        let search_cache = SearchCache::new(DIM, bit_width);
        let cache_ms = cache_start.elapsed().as_secs_f64() * 1000.0;
        validate_hnsw_budget_with_extra(
            &graph,
            cap,
            HNSW_VECTOR_CACHE_VECTORS,
            1,
            quant_query_major_resident_bytes(SIMD_BLOCK, bit_width)
                .saturating_add(DIM * std::mem::size_of::<f32>()),
        )?;
        let mut searcher = HnswQuantSearcher::open(
            &fp16_path,
            &blocked_index_path(output_dir, &dataset.id, bit_width),
            vector_path,
            &graph,
            dataset.vectors,
            bit_width,
            HNSW_VECTOR_CACHE_VECTORS,
            search_cache,
        )?;
        let metrics = benchmark_hnsw_query_major_method(
            &mut searcher,
            &self_queries,
            &random_queries,
            &self_truth,
            &random_truth,
            &recall_candidates,
        )?;
        rows.push(hnsw_method_row(
            &bit_width.to_string(),
            metrics,
            &graph,
            cap,
            graph_index_ms + quant_index_ms,
            graph_load_ms + cache_ms,
            graph_write_ms + quant_write_ms,
            graph.resident_bytes() as u64 + fp16_bytes + quant_rom,
            &format!(
                "disk-backed {}-bit payload + shared FP16 navigation graph + exact FP32 rerank",
                bit_width
            ),
            &format!(
                "{} FP16 navigation cache + one compressed block + FP32 scratch",
                human_bytes((HNSW_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
            ),
        ));
    }
    Ok(rows)
}

fn hnsw_truth_path(output_dir: &Path, dataset_id: &str) -> PathBuf {
    output_dir.join(format!("{}_hnsw_truth_v1.bin", dataset_id))
}

fn load_or_compute_hnsw_truth(
    dataset: &DatasetInput,
    vector_path: &Path,
    self_queries: &[f32],
    random_queries: &[f32],
    cap: usize,
    output_dir: &Path,
) -> Result<(Vec<[usize; K]>, Vec<[usize; K]>), String> {
    let truth_path = hnsw_truth_path(output_dir, &dataset.id);
    let source_bytes = vector_path.metadata().map(|meta| meta.len()).unwrap_or(0);
    if let Ok(cached) = read_hnsw_truth_cache(
        &truth_path,
        dataset.vectors,
        source_bytes,
    ) {
        return Ok(cached);
    }

    let self_truth = exact_topk_file_batched(vector_path, dataset.vectors, self_queries, cap)?;
    let random_truth = exact_topk_file_batched(vector_path, dataset.vectors, random_queries, cap)?;
    write_hnsw_truth_cache(
        &truth_path,
        dataset.vectors,
        source_bytes,
        &self_truth,
        &random_truth,
    )?;
    Ok((self_truth, random_truth))
}

fn read_hnsw_truth_cache(
    path: &Path,
    expected_n: usize,
    expected_source_bytes: u64,
) -> Result<(Vec<[usize; K]>, Vec<[usize; K]>), String> {
    let file = File::open(path).map_err(|e| format!("open HNSW truth cache {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("read HNSW truth cache {}: {e}", path.display()))?;
    let dim = read_u32(&mut reader, path)? as usize;
    let n = read_u64(&mut reader, path)? as usize;
    let source_bytes = read_u64(&mut reader, path)?;
    let self_count = read_u32(&mut reader, path)? as usize;
    let random_count = read_u32(&mut reader, path)? as usize;
    let stored_k = read_u32(&mut reader, path)? as usize;
    if &magic != HNSW_TRUTH_MAGIC
        || dim != DIM
        || n != expected_n
        || source_bytes != expected_source_bytes
        || self_count != SELF_QUERIES
        || random_count != RANDOM_QUERIES
        || stored_k != K
    {
        return Err(format!("HNSW truth cache parameters do not match {}", path.display()));
    }
    let read_rows = |reader: &mut BufReader<File>, count: usize| -> Result<Vec<[usize; K]>, String> {
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let mut row = [usize::MAX; K];
            for id in &mut row {
                *id = read_u32(reader, path)? as usize;
            }
            rows.push(row);
        }
        Ok(rows)
    };
    Ok((
        read_rows(&mut reader, self_count)?,
        read_rows(&mut reader, random_count)?,
    ))
}

fn write_hnsw_truth_cache(
    path: &Path,
    n: usize,
    source_bytes: u64,
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
) -> Result<(), String> {
    let temp_path = path.with_extension("bin.part");
    let file = File::create(&temp_path)
        .map_err(|e| format!("create HNSW truth cache {}: {e}", temp_path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(HNSW_TRUTH_MAGIC)
        .map_err(|e| format!("write HNSW truth cache {}: {e}", temp_path.display()))?;
    write_u32(&mut writer, DIM as u32, &temp_path)?;
    write_u64(&mut writer, n as u64, &temp_path)?;
    write_u64(&mut writer, source_bytes, &temp_path)?;
    write_u32(&mut writer, self_truth.len() as u32, &temp_path)?;
    write_u32(&mut writer, random_truth.len() as u32, &temp_path)?;
    write_u32(&mut writer, K as u32, &temp_path)?;
    for row in self_truth.iter().chain(random_truth.iter()) {
        for &id in row {
            let id = u32::try_from(id)
                .map_err(|_| format!("HNSW truth id {} does not fit cache format", id))?;
            write_u32(&mut writer, id, &temp_path)?;
        }
    }
    writer
        .flush()
        .map_err(|e| format!("flush HNSW truth cache {}: {e}", temp_path.display()))?;
    std::fs::rename(&temp_path, path).map_err(|e| {
        format!(
            "commit HNSW truth cache {} -> {}: {e}",
            temp_path.display(),
            path.display()
        )
    })
}

fn bench_dataset_hnsw_batched(
    dataset: &DatasetInput,
    output_dir: &Path,
) -> Result<Vec<Row>, String> {
    if dataset.vectors < SELF_QUERIES {
        return Err(format!("{} has fewer than {} vectors", dataset.label, SELF_QUERIES));
    }
    let cap = vector_ram_cap_bytes(dataset.vectors);
    let vector_path = Path::new(&dataset.path);
    validate_vector_file(vector_path, dataset.vectors)?;

    let self_queries = load_vector_range(
        vector_path,
        0,
        SELF_QUERIES,
        cap,
        query_vector_bytes(),
    )?;
    let random_queries = make_random_queries(vector_path, dataset.vectors, RANDOM_QUERIES, cap)?;
    let (self_truth, random_truth) = load_or_compute_hnsw_truth(
        dataset,
        vector_path,
        &self_queries,
        &random_queries,
        cap,
        output_dir,
    )?;

    let fp16_path = hnsw_fp16_path(output_dir, &dataset.id);
    let (store_index_ms, store_write_ms) = ensure_fp16_store(
        vector_path,
        dataset.vectors,
        cap,
        &fp16_path,
    )?;
    let graph_path = hnsw_graph_path(output_dir, &dataset.id);
    let (graph, graph_index_ms, graph_write_ms, graph_load_ms) = ensure_hnsw_graph(
        &fp16_path,
        dataset.vectors,
        &graph_path,
    )?;
    validate_hnsw_budget(&graph, cap, HNSW_BATCH_VECTOR_CACHE_VECTORS)?;
    let fp16_bytes = fp16_path.metadata().map(|m| m.len()).unwrap_or(0);
    let mut rows = Vec::new();

    let self_start = Instant::now();
    let self_results = hnsw_query_batch_parallel_method(&self_queries, || {
        HnswSearcher::open(&fp16_path, &graph, HNSW_BATCH_VECTOR_CACHE_VECTORS)
    })?;
    let self_ms = self_start.elapsed().as_secs_f64() * 1000.0;
    let random_start = Instant::now();
    let random_results = hnsw_query_batch_parallel_method(&random_queries, || {
        HnswSearcher::open(&fp16_path, &graph, HNSW_BATCH_VECTOR_CACHE_VECTORS)
    })?;
    let random_ms = random_start.elapsed().as_secs_f64() * 1000.0;
    let timed_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;
    rows.push(hnsw_method_row_batched(
        "16",
        self_ms,
        random_ms,
        &graph,
        cap,
        recall_at_1_arrays(&self_results, &self_truth),
        recall_at_10_arrays(&self_results, &self_truth),
        recall_at_10_arrays(&random_results, &random_truth),
        store_index_ms + graph_index_ms,
        graph_load_ms,
        store_write_ms + graph_write_ms,
        timed_q,
        graph.resident_bytes() as u64 + fp16_bytes,
        "disk-backed FP16 navigation store + resident graph",
        &format!(
            "{} FP16 candidates per worker",
            human_bytes((HNSW_BATCH_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
        ),
    ));

    let self_start = Instant::now();
    let self_results = hnsw_query_batch_parallel_method(&self_queries, || {
        HnswRawSearcher::open(
            &fp16_path,
            vector_path,
            &graph,
            HNSW_BATCH_VECTOR_CACHE_VECTORS,
        )
    })?;
    let self_ms = self_start.elapsed().as_secs_f64() * 1000.0;
    let random_start = Instant::now();
    let random_results = hnsw_query_batch_parallel_method(&random_queries, || {
        HnswRawSearcher::open(
            &fp16_path,
            vector_path,
            &graph,
            HNSW_BATCH_VECTOR_CACHE_VECTORS,
        )
    })?;
    let random_ms = random_start.elapsed().as_secs_f64() * 1000.0;
    rows.push(hnsw_method_row_batched(
        "32",
        self_ms,
        random_ms,
        &graph,
        cap,
        recall_at_1_arrays(&self_results, &self_truth),
        recall_at_10_arrays(&self_results, &self_truth),
        recall_at_10_arrays(&random_results, &random_truth),
        graph_index_ms,
        graph_load_ms,
        graph_write_ms,
        timed_q,
        graph.resident_bytes() as u64 + fp16_bytes
            + (dataset.vectors as u64) * (DIM as u64) * 4,
        "disk-backed raw FP32 payload + shared FP16 navigation graph",
        &format!(
            "{} FP16 candidates per worker + one FP32 scratch",
            human_bytes((HNSW_BATCH_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
        ),
    ));

    for bit_width in [8usize, 4] {
        let (quant_index_ms, quant_write_ms, quant_rom) = ensure_quant_store(
            vector_path,
            dataset.vectors,
            cap,
            output_dir,
            &dataset.id,
            bit_width,
        )?;
        let cache_start = Instant::now();
        let search_cache = SearchCache::new(DIM, bit_width);
        let cache_ms = cache_start.elapsed().as_secs_f64() * 1000.0;
        validate_hnsw_budget_with_extra(
            &graph,
            cap,
            HNSW_BATCH_VECTOR_CACHE_VECTORS,
            rayon::current_num_threads(),
            quant_query_major_resident_bytes(SIMD_BLOCK, bit_width)
                .saturating_add(DIM * std::mem::size_of::<f32>()),
        )?;
        let quant_path = blocked_index_path(output_dir, &dataset.id, bit_width);
        let self_start = Instant::now();
        let self_results = hnsw_query_batch_parallel_method(&self_queries, || {
            let cache = search_cache.clone();
            HnswQuantSearcher::open(
                &fp16_path,
                &quant_path,
                vector_path,
                &graph,
                dataset.vectors,
                bit_width,
                HNSW_BATCH_VECTOR_CACHE_VECTORS,
                cache,
            )
        })?;
        let self_ms = self_start.elapsed().as_secs_f64() * 1000.0;
        let random_start = Instant::now();
        let random_results = hnsw_query_batch_parallel_method(&random_queries, || {
            let cache = search_cache.clone();
            HnswQuantSearcher::open(
                &fp16_path,
                &quant_path,
                vector_path,
                &graph,
                dataset.vectors,
                bit_width,
                HNSW_BATCH_VECTOR_CACHE_VECTORS,
                cache,
            )
        })?;
        let random_ms = random_start.elapsed().as_secs_f64() * 1000.0;
        rows.push(hnsw_method_row_batched(
            &bit_width.to_string(),
            self_ms,
            random_ms,
            &graph,
            cap,
            recall_at_1_arrays(&self_results, &self_truth),
            recall_at_10_arrays(&self_results, &self_truth),
            recall_at_10_arrays(&random_results, &random_truth),
            graph_index_ms + quant_index_ms,
            graph_load_ms + cache_ms,
            graph_write_ms + quant_write_ms,
            timed_q,
            graph.resident_bytes() as u64 + fp16_bytes + quant_rom,
            &format!(
                "disk-backed {}-bit payload + shared FP16 navigation graph + exact FP32 rerank",
                bit_width
            ),
            &format!(
                "{} FP16 candidates per worker + one compressed block + FP32 scratch",
                human_bytes((HNSW_BATCH_VECTOR_CACHE_VECTORS * DIM * 2) as u64)
            ),
        ));
    }
    Ok(rows)
}

fn ensure_fp16_store(
    vector_path: &Path,
    n: usize,
    cap: usize,
    fp16_path: &Path,
) -> Result<(f64, f64), String> {
    let expected = n
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| format!("FP16 store is too large for {} vectors", n))?;
    if fp16_path.is_file()
        && fp16_path
            .metadata()
            .map(|m| m.len() as usize == expected)
            .unwrap_or(false)
    {
        return Ok((0.0, 0.0));
    }

    let temp_path = fp16_path.with_extension("f16.part");
    let file = File::create(&temp_path)
        .map_err(|e| format!("create FP16 store {}: {e}", temp_path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut index_ms = 0.0f64;
    let mut write_ms = 0.0f64;
    for_each_vector_chunk(vector_path, n, cap, |_, chunk| {
        let encode_start = Instant::now();
        let mut encoded = Vec::with_capacity(chunk.len() * 2);
        for value in chunk {
            encoded.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        index_ms += encode_start.elapsed().as_secs_f64() * 1000.0;
        let write_start = Instant::now();
        writer
            .write_all(&encoded)
            .map_err(|e| format!("write {}: {e}", temp_path.display()))?;
        write_ms += write_start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    })?;
    let flush_start = Instant::now();
    writer
        .flush()
        .map_err(|e| format!("flush {}: {e}", temp_path.display()))?;
    write_ms += flush_start.elapsed().as_secs_f64() * 1000.0;
    std::fs::rename(&temp_path, fp16_path).map_err(|e| {
        format!(
            "commit FP16 store {} -> {}: {e}",
            temp_path.display(),
            fp16_path.display()
        )
    })?;
    Ok((index_ms, write_ms))
}

fn ensure_quant_store(
    vector_path: &Path,
    n: usize,
    cap: usize,
    output_dir: &Path,
    dataset_id: &str,
    bit_width: usize,
) -> Result<(f64, f64, u64), String> {
    let path = index_path(output_dir, dataset_id, bit_width);
    let blocked_path = blocked_index_path(output_dir, dataset_id, bit_width);
    let valid_index = TurboQuantIndex::load(&path)
        .map(|index| index.len() == n)
        .unwrap_or(false);
    let valid_blocked = blocked_path
        .metadata()
        .map(|metadata| metadata.len() > 16)
        .unwrap_or(false);
    if valid_index && valid_blocked {
        let rom = path.metadata().map(|m| m.len()).unwrap_or(0)
            + blocked_path.metadata().map(|m| m.len()).unwrap_or(0);
        return Ok((0.0, 0.0, rom));
    }

    let index_start = Instant::now();
    let mut index = TurboQuantIndex::new(DIM, bit_width).map_err(|e| format!("{e:?}"))?;
    for_each_vector_chunk(vector_path, n, cap, |_, chunk| {
        index.add(&chunk);
        Ok(())
    })?;
    let index_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    let write_start = Instant::now();
    let index_part = PathBuf::from(format!("{}.part", path.display()));
    let blocked_part = PathBuf::from(format!("{}.part", blocked_path.display()));
    index
        .write(&index_part)
        .map_err(|e| format!("write {}: {e}", index_part.display()))?;
    index
        .write_blocked(&blocked_part)
        .map_err(|e| format!("write {}: {e}", blocked_part.display()))?;
    std::fs::rename(&index_part, &path)
        .map_err(|e| format!("commit {}: {e}", path.display()))?;
    std::fs::rename(&blocked_part, &blocked_path)
        .map_err(|e| format!("commit {}: {e}", blocked_path.display()))?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0)
        + blocked_path.metadata().map(|m| m.len()).unwrap_or(0);
    Ok((index_ms, write_ms, rom))
}

fn ensure_hnsw_graph(
    fp16_path: &Path,
    n: usize,
    graph_path: &Path,
) -> Result<(HnswGraph, f64, f64, f64), String> {
    if let Ok(graph) = HnswGraph::load(graph_path, n) {
        let load_ms = 0.0f64;
        return Ok((graph, 0.0, 0.0, load_ms));
    }

    // efSearch is a query-time knob; it does not change graph topology. Reuse
    // the graph generated by the previous release when it has the same M and
    // efConstruction, then expose the new efSearch in the in-memory graph.
    // This avoids an unnecessary full 50K/100K graph rebuild after a recall
    // tuning change.
    let legacy_path = graph_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            "{}_hnsw_m{}_ec{}_es{}.tvhg",
            graph_path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.split("_hnsw_").next())
                .unwrap_or("dataset"),
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
            HNSW_LEGACY_EF_SEARCH
        ));
    if legacy_path != graph_path {
        if let Ok(graph) = HnswGraph::load(&legacy_path, n) {
            return Ok((graph, 0.0, 0.0, 0.0));
        }
    }

    // Older builds used the same topology header but did not always preserve
    // the dataset-id spelling in the filename. Search the bounded app-files
    // directory for another graph with the same M/efConstruction prefix before
    // falling back to an expensive rebuild.
    let topology_prefix = graph_path
        .file_stem()
        .and_then(|name| name.to_str())
        .and_then(|name| name.split("_es").next())
        .unwrap_or("");
    if !topology_prefix.is_empty() {
        if let Ok(entries) = std::fs::read_dir(graph_path.parent().unwrap_or_else(|| Path::new("."))) {
            for entry in entries.flatten() {
                let candidate = entry.path();
                let name = candidate.file_name().and_then(|value| value.to_str()).unwrap_or("");
                if candidate != graph_path
                    && name.starts_with(topology_prefix)
                    && name.ends_with(".tvhg")
                {
                    if let Ok(graph) = HnswGraph::load(&candidate, n) {
                        return Ok((graph, 0.0, 0.0, 0.0));
                    }
                }
            }
        }
    }

    let build_start = Instant::now();
    let expected = n
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| format!("FP16 graph source is too large for {} vectors", n))?;
    let mut reader = BufReader::new(
        File::open(fp16_path).map_err(|e| format!("open {}: {e}", fp16_path.display()))?,
    );
    let data = read_f16_bits(&mut reader, expected, n * DIM, fp16_path)?;
    let graph = build_hnsw_graph(&data, n)?;
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let write_start = Instant::now();
    let temp_path = graph_path.with_extension("tvhg.part");
    graph.write(&temp_path)?;
    std::fs::rename(&temp_path, graph_path).map_err(|e| {
        format!(
            "commit HNSW graph {} -> {}: {e}",
            temp_path.display(),
            graph_path.display()
        )
    })?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    drop(data);
    let load_start = Instant::now();
    let loaded = HnswGraph::load(graph_path, n)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    Ok((loaded, build_ms, write_ms, load_ms))
}

fn hnsw_fp16_path(output_dir: &Path, dataset_id: &str) -> PathBuf {
    output_dir.join(format!("{}_hnsw_fp16.f16", dataset_id))
}

fn hnsw_graph_path(output_dir: &Path, dataset_id: &str) -> PathBuf {
    output_dir.join(format!(
        "{}_hnsw_m{}_ec{}_es{}.tvhg",
        dataset_id, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH
    ))
}

#[derive(Clone, Copy, Debug)]
struct HnswHeapItem {
    distance: f32,
    id: usize,
}

impl PartialEq for HnswHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for HnswHeapItem {}

impl PartialOrd for HnswHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HnswHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.distance.partial_cmp(&other.distance) {
            Some(Ordering::Equal) | None => self.id.cmp(&other.id),
            Some(order) => order,
        }
    }
}

struct HnswGraph {
    n: usize,
    dim: usize,
    m: usize,
    ef_search: usize,
    max_level: usize,
    entry: usize,
    levels: Vec<u8>,
    // For node i and layer l, neighbours[offsets[i*stride+l]..
    // offsets[i*stride+l+1]] are the layer-l adjacency list.
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl HnswGraph {
    fn stride(&self) -> usize {
        self.max_level + 2
    }

    fn resident_bytes(&self) -> usize {
        self.levels.len()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.neighbors.len() * std::mem::size_of::<u32>()
    }

    fn neighbor_range(&self, node: usize, layer: usize) -> std::ops::Range<usize> {
        if node >= self.n || layer > self.max_level || layer > self.levels[node] as usize {
            return 0..0;
        }
        let base = node * self.stride() + layer;
        self.offsets[base] as usize..self.offsets[base + 1] as usize
    }

    fn write(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path)
            .map_err(|e| format!("create HNSW graph {}: {e}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(HNSW_MAGIC)
            .map_err(|e| format!("write HNSW header {}: {e}", path.display()))?;
        write_u32(&mut writer, self.dim as u32, path)?;
        write_u64(&mut writer, self.n as u64, path)?;
        write_u32(&mut writer, self.m as u32, path)?;
        write_u32(&mut writer, HNSW_EF_CONSTRUCTION as u32, path)?;
        write_u32(&mut writer, self.ef_search as u32, path)?;
        write_u32(&mut writer, self.max_level as u32, path)?;
        write_u64(&mut writer, self.entry as u64, path)?;
        write_u64(&mut writer, self.levels.len() as u64, path)?;
        write_u64(&mut writer, self.offsets.len() as u64, path)?;
        write_u64(&mut writer, self.neighbors.len() as u64, path)?;
        writer
            .write_all(&self.levels)
            .map_err(|e| format!("write HNSW levels {}: {e}", path.display()))?;
        write_u32_slice(&mut writer, &self.offsets, path)?;
        write_u32_slice(&mut writer, &self.neighbors, path)?;
        writer
            .flush()
            .map_err(|e| format!("flush HNSW graph {}: {e}", path.display()))
    }

    fn load(path: &Path, expected_n: usize) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("open HNSW graph {}: {e}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("read HNSW header {}: {e}", path.display()))?;
        if &magic != HNSW_MAGIC {
            return Err(format!("invalid HNSW graph magic in {}", path.display()));
        }
        let dim = read_u32(&mut reader, path)? as usize;
        let n = read_u64(&mut reader, path)? as usize;
        let m = read_u32(&mut reader, path)? as usize;
        let _ef_construction = read_u32(&mut reader, path)? as usize;
        let stored_ef_search = read_u32(&mut reader, path)? as usize;
        let max_level = read_u32(&mut reader, path)? as usize;
        let entry = read_u64(&mut reader, path)? as usize;
        let levels_len = read_u64(&mut reader, path)? as usize;
        let offsets_len = read_u64(&mut reader, path)? as usize;
        let neighbors_len = read_u64(&mut reader, path)? as usize;
        if dim != DIM
            || n != expected_n
            || m != HNSW_M
            || stored_ef_search == 0
            || n == 0
            || entry >= n
            || max_level > HNSW_MAX_LAYER
            || levels_len != n
            || offsets_len != n.saturating_mul(max_level + 2)
        {
            return Err(format!("HNSW graph parameters do not match {}", path.display()));
        }
        let mut levels = vec![0u8; levels_len];
        reader
            .read_exact(&mut levels)
            .map_err(|e| format!("read HNSW levels {}: {e}", path.display()))?;
        if levels.iter().any(|&level| level as usize > max_level) {
            return Err(format!("invalid HNSW level in {}", path.display()));
        }
        let mut offsets = vec![0u32; offsets_len];
        read_u32_slice(&mut reader, &mut offsets, path)?;
        let mut neighbors = vec![0u32; neighbors_len];
        read_u32_slice(&mut reader, &mut neighbors, path)?;
        if neighbors.iter().any(|&id| id as usize >= n)
            || offsets.first().copied().unwrap_or(0) != 0
            || offsets.last().copied().unwrap_or(0) as usize != neighbors.len()
            || offsets.windows(2).any(|window| window[0] > window[1])
        {
            return Err(format!("invalid HNSW adjacency arrays in {}", path.display()));
        }
        Ok(Self {
            n,
            dim,
            m,
            ef_search: HNSW_EF_SEARCH,
            max_level,
            entry,
            levels,
            offsets,
            neighbors,
        })
    }
}

fn hnsw_graph_ram_budget(cap: usize, mode: BenchmarkMode) -> String {
    let cache = if mode == BenchmarkMode::HnswBatched {
        HNSW_BATCH_VECTOR_CACHE_VECTORS
    } else {
        HNSW_VECTOR_CACHE_VECTORS
    };
    format!(
        "compact graph + {} FP16 candidates{} + scratch <= {}",
        cache,
        if mode == BenchmarkMode::HnswBatched {
            " per worker"
        } else {
            ""
        },
        human_bytes(cap as u64)
    )
}

fn validate_hnsw_budget(
    graph: &HnswGraph,
    cap: usize,
    cache_vectors: usize,
) -> Result<(), String> {
    validate_hnsw_budget_with_extra(graph, cap, cache_vectors, 1, 0)
}

fn validate_hnsw_budget_with_extra(
    graph: &HnswGraph,
    cap: usize,
    cache_vectors: usize,
    workers: usize,
    shared_extra_bytes: usize,
) -> Result<(), String> {
    let per_searcher = cache_vectors
        .saturating_mul(DIM)
        .saturating_mul(2)
        .saturating_add(cache_vectors.saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(graph.n.saturating_mul(std::mem::size_of::<u32>()));
    let accounted = graph
        .resident_bytes()
        .saturating_add(per_searcher.saturating_mul(workers.max(1)))
        .saturating_add(shared_extra_bytes)
        .saturating_add(256 * 1024);
    if accounted > cap {
        return Err(format!(
            "HNSW graph/search working set {} exceeds {} cap (graph {}, cache {} vectors)",
            human_bytes(accounted as u64),
            human_bytes(cap as u64),
            human_bytes(graph.resident_bytes() as u64),
            cache_vectors
        ));
    }
    Ok(())
}

fn build_hnsw_graph(data: &[u16], n: usize) -> Result<HnswGraph, String> {
    if data.len() != n.saturating_mul(DIM) || n == 0 {
        return Err(format!("invalid FP16 HNSW source: {} values for {} vectors", data.len(), n));
    }

    let mut rng = StdRng::seed_from_u64(0x4853_4e57_2026);
    let levels: Vec<u8> = (0..n).map(|_| hnsw_random_level(&mut rng)).collect();
    let max_level = levels.iter().copied().max().unwrap_or(0) as usize;
    let mut adjacency: Vec<Vec<Vec<usize>>> = levels
        .iter()
        .map(|&level| (0..=level as usize).map(|_| Vec::new()).collect())
        .collect();
    let mut visited = vec![0u32; n];
    let mut generation = 0u32;
    let mut entry = 0usize;
    let mut entry_level = levels[0] as usize;

    for id in 1..n {
        let level = levels[id] as usize;
        let mut ep = entry;
        if level < entry_level {
            for layer in ((level + 1)..=entry_level).rev() {
                ep = hnsw_greedy_build(data, &adjacency, ep, layer, id);
            }
        }
        let lower = level.min(entry_level);
        for layer in (0..=lower).rev() {
            let candidates = hnsw_search_layer_build(
                data,
                &adjacency,
                ep,
                layer,
                id,
                HNSW_EF_CONSTRUCTION,
                &mut visited,
                &mut generation,
            );
            let limit = if layer == 0 { 2 * HNSW_M } else { HNSW_M };
            let selected = hnsw_select_neighbors(data, id, candidates.clone(), limit);
            adjacency[id][layer] = selected.clone();
            for &neighbor in &selected {
                adjacency[neighbor][layer].push(id);
                hnsw_prune_neighbors(data, &mut adjacency, neighbor, layer, limit);
            }
            if let Some(&next_ep) = candidates.first() {
                ep = next_ep;
            }
        }
        if level > entry_level {
            entry = id;
            entry_level = level;
        }
    }

    let stride = max_level + 2;
    let mut offsets = vec![0u32; n.saturating_mul(stride)];
    let mut neighbors = Vec::<u32>::new();
    for node in 0..n {
        let base = node * stride;
        for layer in 0..=max_level + 1 {
            offsets[base + layer] = neighbors.len() as u32;
            if layer <= max_level && layer <= levels[node] as usize {
                for &neighbor in &adjacency[node][layer] {
                    neighbors.push(neighbor as u32);
                }
            }
        }
    }
    Ok(HnswGraph {
        n,
        dim: DIM,
        m: HNSW_M,
        ef_search: HNSW_EF_SEARCH,
        max_level,
        entry,
        levels,
        offsets,
        neighbors,
    })
}

fn hnsw_random_level(rng: &mut StdRng) -> u8 {
    let u = rng.gen_range(f64::MIN_POSITIVE..1.0);
    ((-u.ln() / (HNSW_M as f64).ln()).floor() as usize).min(HNSW_MAX_LAYER) as u8
}

fn hnsw_node_distance(data: &[u16], a: usize, b: usize) -> f32 {
    1.0 - dot_fp16_values(
        &data[a * DIM..(a + 1) * DIM],
        &data[b * DIM..(b + 1) * DIM],
    )
}

fn hnsw_greedy_build(
    data: &[u16],
    adjacency: &[Vec<Vec<usize>>],
    mut current: usize,
    layer: usize,
    target: usize,
) -> usize {
    loop {
        let current_distance = hnsw_node_distance(data, target, current);
        let mut best = current;
        let mut best_distance = current_distance;
        if layer <= adjacency[current].len().saturating_sub(1) {
            for &neighbor in &adjacency[current][layer] {
                let distance = hnsw_node_distance(data, target, neighbor);
                if distance < best_distance {
                    best = neighbor;
                    best_distance = distance;
                }
            }
        }
        if best == current {
            break;
        }
        current = best;
    }
    current
}

fn hnsw_search_layer_build(
    data: &[u16],
    adjacency: &[Vec<Vec<usize>>],
    entry: usize,
    layer: usize,
    target: usize,
    ef: usize,
    visited: &mut [u32],
    generation: &mut u32,
) -> Vec<usize> {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        visited.fill(0);
        *generation = 1;
    }
    let marker = *generation;
    let mut candidates = BinaryHeap::<std::cmp::Reverse<HnswHeapItem>>::new();
    let mut results = BinaryHeap::<HnswHeapItem>::new();
    let initial = HnswHeapItem {
        distance: hnsw_node_distance(data, target, entry),
        id: entry,
    };
    visited[entry] = marker;
    candidates.push(std::cmp::Reverse(initial));
    results.push(initial);

    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        let worst = results.peek().map(|item| item.distance).unwrap_or(f32::INFINITY);
        if results.len() >= ef && current.distance > worst {
            break;
        }
        if layer >= adjacency[current.id].len() {
            continue;
        }
        for &neighbor in &adjacency[current.id][layer] {
            if visited[neighbor] == marker {
                continue;
            }
            visited[neighbor] = marker;
            let distance = hnsw_node_distance(data, target, neighbor);
            let worst = results.peek().map(|item| item.distance).unwrap_or(f32::INFINITY);
            if results.len() < ef || distance < worst {
                let item = HnswHeapItem { distance, id: neighbor };
                candidates.push(std::cmp::Reverse(item));
                results.push(item);
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut ordered = results.into_vec();
    ordered.sort_by(|a, b| a.cmp(b));
    ordered.into_iter().map(|item| item.id).collect()
}

fn hnsw_select_neighbors(data: &[u16], target: usize, mut candidates: Vec<usize>, limit: usize) -> Vec<usize> {
    candidates.sort_unstable_by(|&a, &b| {
        hnsw_node_distance(data, target, a)
            .partial_cmp(&hnsw_node_distance(data, target, b))
            .unwrap_or(Ordering::Equal)
    });
    candidates.dedup();
    candidates.truncate(limit);
    candidates
}

fn hnsw_prune_neighbors(
    data: &[u16],
    adjacency: &mut [Vec<Vec<usize>>],
    node: usize,
    layer: usize,
    limit: usize,
) {
    if layer >= adjacency[node].len() || adjacency[node][layer].len() <= limit {
        return;
    }
    let mut neighbors = std::mem::take(&mut adjacency[node][layer]);
    neighbors.sort_unstable_by(|&a, &b| {
        hnsw_node_distance(data, node, a)
            .partial_cmp(&hnsw_node_distance(data, node, b))
            .unwrap_or(Ordering::Equal)
    });
    neighbors.dedup();
    neighbors.truncate(limit);
    adjacency[node][layer] = neighbors;
}

struct HnswSearcher<'a> {
    graph: &'a HnswGraph,
    file: File,
    cache_block_ids: Vec<u32>,
    cache_values: Vec<u16>,
    visited: Vec<u32>,
    generation: u32,
}

impl<'a> HnswSearcher<'a> {
    fn open(path: &Path, graph: &'a HnswGraph, cache_vectors: usize) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open HNSW FP16 store {}: {e}", path.display()))?;
        let cache_blocks = (cache_vectors.max(1) + HNSW_CACHE_BLOCK_VECTORS - 1)
            / HNSW_CACHE_BLOCK_VECTORS;
        Ok(Self {
            graph,
            file,
            cache_block_ids: vec![u32::MAX; cache_blocks],
            cache_values: vec![0u16; cache_blocks * HNSW_CACHE_BLOCK_VECTORS * DIM],
            visited: vec![0u32; graph.n],
            generation: 0,
        })
    }

    fn next_generation(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.visited.fill(0);
            self.generation = 1;
        }
        self.generation
    }

    fn distance(&mut self, query: &[f32], id: usize) -> Result<f32, String> {
        let block_base = (id / HNSW_CACHE_BLOCK_VECTORS) * HNSW_CACHE_BLOCK_VECTORS;
        let block_slot = (block_base / HNSW_CACHE_BLOCK_VECTORS) % self.cache_block_ids.len();
        let cache_start = block_slot * HNSW_CACHE_BLOCK_VECTORS * DIM;
        let take = (self.graph.n - block_base).min(HNSW_CACHE_BLOCK_VECTORS);
        if self.cache_block_ids[block_slot] != block_base as u32 {
            read_f16_vectors_at(
                &self.file,
                block_base,
                take,
                &mut self.cache_values[cache_start..cache_start + take * DIM],
            )?;
            self.cache_block_ids[block_slot] = block_base as u32;
        }
        let vector_start = cache_start + (id - block_base) * DIM;
        Ok(1.0 - dot_fp16(
            query,
            &self.cache_values[vector_start..vector_start + DIM],
        ))
    }

    fn search_candidates(&mut self, query: &[f32]) -> Result<Vec<usize>, String> {
        if query.len() != DIM {
            return Err(format!("HNSW query has {} values, expected {}", query.len(), DIM));
        }
        let mut current = self.graph.entry;
        let mut current_distance = self.distance(query, current)?;
        for layer in (1..=self.graph.max_level).rev() {
            if layer > self.graph.levels[current] as usize {
                continue;
            }
            loop {
                let mut best = current;
                let mut best_distance = current_distance;
                let range = self.graph.neighbor_range(current, layer);
                for pos in range {
                    let neighbor = self.graph.neighbors[pos] as usize;
                    let distance = self.distance(query, neighbor)?;
                    if distance < best_distance {
                        best = neighbor;
                        best_distance = distance;
                    }
                }
                if best == current {
                    break;
                }
                current = best;
                current_distance = best_distance;
            }
        }

        let marker = self.next_generation();
        let ef = self.graph.ef_search.max(K);
        let initial = HnswHeapItem {
            distance: current_distance,
            id: current,
        };
        let mut candidates = BinaryHeap::<std::cmp::Reverse<HnswHeapItem>>::new();
        let mut results = BinaryHeap::<HnswHeapItem>::new();
        self.visited[current] = marker;
        candidates.push(std::cmp::Reverse(initial));
        results.push(initial);
        while let Some(std::cmp::Reverse(item)) = candidates.pop() {
            let worst = results.peek().map(|candidate| candidate.distance).unwrap_or(f32::INFINITY);
            if results.len() >= ef && item.distance > worst {
                break;
            }
            let range = self.graph.neighbor_range(item.id, 0);
            for pos in range {
                let neighbor = self.graph.neighbors[pos] as usize;
                if self.visited[neighbor] == marker {
                    continue;
                }
                self.visited[neighbor] = marker;
                let distance = self.distance(query, neighbor)?;
                let worst = results.peek().map(|candidate| candidate.distance).unwrap_or(f32::INFINITY);
                if results.len() < ef || distance < worst {
                    let next = HnswHeapItem { distance, id: neighbor };
                    candidates.push(std::cmp::Reverse(next));
                    results.push(next);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut ordered = results.into_vec();
        ordered.sort_by(|a, b| a.cmp(b));
        Ok(ordered.into_iter().map(|item| item.id).collect())
    }

    fn search(&mut self, query: &[f32]) -> Result<[usize; K], String> {
        let candidates = self.search_candidates(query)?;
        let mut ids = [usize::MAX; K];
        for (rank, id) in candidates.into_iter().take(K).enumerate() {
            ids[rank] = id;
        }
        Ok(ids)
    }
}

trait HnswMethodSearch {
    fn search_method(&mut self, query: &[f32]) -> Result<[usize; K], String>;
    fn score_candidates(
        &mut self,
        query: &[f32],
        candidates: &[usize],
    ) -> Result<[usize; K], String>;
}

impl HnswMethodSearch for HnswSearcher<'_> {
    fn search_method(&mut self, query: &[f32]) -> Result<[usize; K], String> {
        self.search(query)
    }

    fn score_candidates(
        &mut self,
        _query: &[f32],
        candidates: &[usize],
    ) -> Result<[usize; K], String> {
        let mut ids = [usize::MAX; K];
        for (rank, id) in candidates.iter().copied().take(K).enumerate() {
            ids[rank] = id;
        }
        Ok(ids)
    }
}

struct HnswRawSearcher<'a> {
    navigation: HnswSearcher<'a>,
    raw_file: File,
    scratch: Vec<f32>,
}

impl<'a> HnswRawSearcher<'a> {
    fn open(
        navigation_path: &Path,
        raw_path: &Path,
        graph: &'a HnswGraph,
        cache_vectors: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            navigation: HnswSearcher::open(navigation_path, graph, cache_vectors)?,
            raw_file: File::open(raw_path)
                .map_err(|e| format!("open HNSW raw FP32 store {}: {e}", raw_path.display()))?,
            scratch: vec![0.0f32; HNSW_RAW_READ_SPAN_VECTORS * DIM],
        })
    }
}

impl HnswMethodSearch for HnswRawSearcher<'_> {
    fn search_method(&mut self, query: &[f32]) -> Result<[usize; K], String> {
        let candidates = self.navigation.search_candidates(query)?;
        self.score_candidates(query, &candidates)
    }

    fn score_candidates(
        &mut self,
        query: &[f32],
        candidates: &[usize],
    ) -> Result<[usize; K], String> {
        let mut heap = [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        let mut ordered = candidates.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        let mut cursor = 0usize;
        while cursor < ordered.len() {
            let base = ordered[cursor];
            let mut end = cursor + 1;
            while end < ordered.len()
                && ordered[end] - ordered[end - 1] == 1
                && ordered[end] - base < HNSW_RAW_READ_SPAN_VECTORS
            {
                end += 1;
            }
            let take = ordered[end - 1] - base + 1;
            read_f32_vectors_at(
                &self.raw_file,
                base,
                take,
                &mut self.scratch[..take * DIM],
            )?;
            for &id in &ordered[cursor..end] {
                let local = id - base;
                let start = local * DIM;
                insert_hit(
                    &mut heap,
                    Hit {
                        score: dot(query, &self.scratch[start..start + DIM]),
                        idx: id,
                    },
                );
            }
            cursor = end;
        }
        Ok(finalize_heap(heap))
    }
}

struct HnswQuantSearcher<'a> {
    navigation: HnswSearcher<'a>,
    quant_reader: BufReader<File>,
    raw_file: File,
    n: usize,
    bit_width: usize,
    search_cache: SearchCache,
}

impl<'a> HnswQuantSearcher<'a> {
    fn open(
        navigation_path: &Path,
        quant_path: &Path,
        raw_path: &Path,
        graph: &'a HnswGraph,
        n: usize,
        bit_width: usize,
        cache_vectors: usize,
        search_cache: SearchCache,
    ) -> Result<Self, String> {
        Ok(Self {
            navigation: HnswSearcher::open(navigation_path, graph, cache_vectors)?,
            quant_reader: BufReader::new(
                File::open(quant_path)
                    .map_err(|e| format!("open HNSW quant store {}: {e}", quant_path.display()))?,
            ),
            raw_file: File::open(raw_path)
                .map_err(|e| format!("open HNSW raw FP32 store {}: {e}", raw_path.display()))?,
            n,
            bit_width,
            search_cache,
        })
    }
}

impl HnswMethodSearch for HnswQuantSearcher<'_> {
    fn search_method(&mut self, query: &[f32]) -> Result<[usize; K], String> {
        let candidates = self.navigation.search_candidates(query)?;
        self.score_candidates(query, &candidates)
    }

    fn score_candidates(
        &mut self,
        query: &[f32],
        candidates: &[usize],
    ) -> Result<[usize; K], String> {
        let quant_hits = quant_score_candidate_ids(
            &mut self.quant_reader,
            self.n,
            self.bit_width,
            query,
            &candidates,
            &self.search_cache,
        )?;
        // The graph/quantized stage narrows the candidate set; exact FP32
        // scoring gives the displayed HNSW 8/4-bit rows the same recall
        // contract as FlatIndex A without resident raw-vector storage.
        rerank_exact_candidates_from_file(&self.raw_file, query, &quant_hits)
    }
}

fn quant_score_candidate_ids(
    reader: &mut BufReader<File>,
    n: usize,
    _bit_width: usize,
    query: &[f32],
    candidates: &[usize],
    search_cache: &SearchCache,
) -> Result<Vec<Hit>, String> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut hits = Vec::with_capacity(sorted.len());
    let mut cursor = 0usize;
    while cursor < sorted.len() {
        let block_base = (sorted[cursor] / SIMD_BLOCK) * SIMD_BLOCK;
        let take = (n - block_base).min(SIMD_BLOCK);
        let group_start = cursor;
        while cursor < sorted.len() && sorted[cursor] < block_base + take {
            cursor += 1;
        }
        let index = TurboQuantIndex::load_blocked_range_from_reader(reader, block_base, take)
            .map_err(|e| format!("load HNSW quant block [{}..{}]: {e}", block_base, block_base + take))?;
        index.prepare_with_cache(search_cache);
        let results = index.search_one(query, take);
        for rank in 0..results.k {
            let local = results.indices[rank];
            if local < 0 {
                continue;
            }
            let global = block_base + local as usize;
            if sorted[group_start..cursor].binary_search(&global).is_ok() {
                hits.push(Hit {
                    score: results.scores[rank],
                    idx: global,
                });
            }
        }
    }
    hits.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    hits.truncate(QUANT_RERANK_CANDIDATES);
    Ok(hits)
}

fn hnsw_query_set<S: HnswMethodSearch>(
    searcher: &mut S,
    queries: &[f32],
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    let mut results = Vec::with_capacity(nq);
    for query in queries.chunks_exact(DIM) {
        results.push(searcher.search_method(query)?);
    }
    Ok(results)
}

struct HnswRecallCandidates {
    self_queries: Vec<Vec<usize>>,
    random_queries: Vec<Vec<usize>>,
}

fn hnsw_navigation_candidates(
    searcher: &mut HnswSearcher<'_>,
    queries: &[f32],
) -> Result<Vec<Vec<usize>>, String> {
    queries
        .chunks_exact(DIM)
        .map(|query| searcher.search_candidates(query))
        .collect()
}

fn hnsw_score_candidate_set<S: HnswMethodSearch>(
    searcher: &mut S,
    queries: &[f32],
    candidates: &[Vec<usize>],
) -> Result<Vec<[usize; K]>, String> {
    if candidates.len() != queries.len() / DIM {
        return Err("HNSW recall candidate/query count mismatch".to_string());
    }
    candidates
        .iter()
        .enumerate()
        .map(|(qi, ids)| searcher.score_candidates(&queries[qi * DIM..(qi + 1) * DIM], ids))
        .collect()
}

#[derive(Clone, Copy)]
struct HnswQueryMetrics {
    timing: QueryMajorTiming,
    self_r1: f64,
    self_r10: f64,
    random_r10: f64,
}

fn benchmark_hnsw_query_major_method<S: HnswMethodSearch>(
    searcher: &mut S,
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    recall_candidates: &HnswRecallCandidates,
) -> Result<HnswQueryMetrics, String> {
    // Graph traversal is identical for the four payload rows. Build the
    // candidate frontier once per dataset and reuse it for recall scoring;
    // this keeps the KPI setup from traversing the same high-ef graph four
    // times. The timed probes below still execute each method's complete
    // graph traversal and payload scoring, so ms/query remains honest.
    let self_results = hnsw_score_candidate_set(
        searcher,
        self_queries,
        &recall_candidates.self_queries,
    )?;
    let random_results = hnsw_score_candidate_set(
        searcher,
        random_queries,
        &recall_candidates.random_queries,
    )?;
    let timing = time_hnsw_probes(searcher, self_queries, random_queries)?;
    Ok(HnswQueryMetrics {
        timing,
        self_r1: recall_at_1_arrays(&self_results, self_truth),
        self_r10: recall_at_10_arrays(&self_results, self_truth),
        random_r10: recall_at_10_arrays(&random_results, random_truth),
    })
}

fn hnsw_query_batch_parallel_method<S, F>(
    queries: &[f32],
    factory: F,
) -> Result<Vec<[usize; K]>, String>
where
    S: HnswMethodSearch + Send,
    F: Fn() -> Result<S, String> + Send + Sync,
{
    let result: Vec<Result<[usize; K], String>> = queries
        .par_chunks_exact(DIM)
        .map_init(
            || factory(),
            |state, query| match state.as_mut() {
                Ok(searcher) => searcher.search_method(query),
                Err(error) => Err(error.clone()),
            },
        )
        .collect();
    result.into_iter().collect()
}

fn time_hnsw_probes<S: HnswMethodSearch>(
    searcher: &mut S,
    self_queries: &[f32],
    random_queries: &[f32],
) -> Result<QueryMajorTiming, String> {
    let mut total_ms = 0.0f64;
    let mut first_ms = 0.0f64;
    let mut count = 0usize;
    for queries in [self_queries, random_queries] {
        for query in queries.chunks_exact(DIM).take(TIMED_PROBES_PER_SET) {
            let start = Instant::now();
            let _ = searcher.search_method(query)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if count == 0 {
                first_ms = elapsed;
            }
            total_ms += elapsed;
            count += 1;
        }
    }
    Ok(QueryMajorTiming::from_samples(total_ms, first_ms, count))
}

fn hnsw_method_row(
    bits: &str,
    metrics: HnswQueryMetrics,
    _graph: &HnswGraph,
    cap: usize,
    index_ms: f64,
    prepare_ms: f64,
    write_ms: f64,
    index_rom: u64,
    data_store: &str,
    vector_staging: &str,
) -> Row {
    let timing = metrics.timing;
    Row {
        index: "HNSW".to_string(),
        bits: bits.to_string(),
        self_r1: pct(metrics.self_r1),
        self_r10: pct(metrics.self_r10),
        random_r10: pct(metrics.random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", timing.total_ms / 2.0),
        random_search_ms: format!("{:.1}", timing.total_ms / 2.0),
        ms_per_query: format!("{:.3}", timing.average_ms()),
        first_query_ms: format!("{:.3}", timing.first_ms),
        warm_ms_per_query: format!("{:.3}", timing.warm_ms_per_query),
        index_rom: human_bytes(index_rom),
        data_store: data_store.to_string(),
        vector_staging: vector_staging.to_string(),
        vector_ram: vector_ram_label(cap),
    }
}

fn hnsw_method_row_batched(
    bits: &str,
    self_ms: f64,
    random_ms: f64,
    _graph: &HnswGraph,
    cap: usize,
    self_r1: f64,
    self_r10: f64,
    random_r10: f64,
    index_ms: f64,
    prepare_ms: f64,
    write_ms: f64,
    timed_q: f64,
    index_rom: u64,
    data_store: &str,
    vector_staging: &str,
) -> Row {
    let search_ms = self_ms + random_ms;
    Row {
        index: "HNSW".to_string(),
        bits: bits.to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", self_ms),
        random_search_ms: format!("{:.1}", random_ms),
        ms_per_query: format!("{:.3}", search_ms / timed_q),
        first_query_ms: "n/a".to_string(),
        warm_ms_per_query: "n/a".to_string(),
        index_rom: human_bytes(index_rom),
        data_store: data_store.to_string(),
        vector_staging: vector_staging.to_string(),
        vector_ram: vector_ram_label(cap),
    }
}

fn read_bytes_at(file: &File, offset: u64, bytes: &mut [u8], label: &str) -> Result<(), String> {
    #[cfg(unix)]
    {
        let mut done = 0usize;
        while done < bytes.len() {
            let read = file
                .read_at(&mut bytes[done..], offset + done as u64)
                .map_err(|e| format!("read {label}: {e}"))?;
            if read == 0 {
                return Err(format!("short read for {label}"));
            }
            done += read;
        }
        return Ok(());
    }
    #[cfg(not(unix))]
    {
        let mut reader = file
            .try_clone()
            .map_err(|e| format!("clone file for {label}: {e}"))?;
        reader
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| format!("seek for {label}: {e}"))?;
        reader.read_exact(bytes).map_err(|e| format!("read {label}: {e}"))
    }
}

fn read_f16_vectors_at(
    file: &File,
    base: usize,
    take: usize,
    out: &mut [u16],
) -> Result<(), String> {
    if take == 0 || out.len() != take * DIM {
        return Err(format!("invalid HNSW FP16 range {}..{}", base, base + take));
    }
    let offset = base
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| format!("HNSW FP16 offset overflow for range {}..{}", base, base + take))?;
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
    read_bytes_at(
        file,
        offset as u64,
        bytes,
        &format!("HNSW FP16 vectors {}..{}", base, base + take),
    )
}

fn read_f32_vectors_at(
    file: &File,
    base: usize,
    take: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if take == 0 || out.len() != take * DIM {
        return Err(format!("invalid FP32 range {}..{}", base, base + take));
    }
    let offset = base
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| format!("FP32 offset overflow for range {}..{}", base, base + take))?;
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    read_bytes_at(
        file,
        offset as u64,
        bytes,
        &format!("FP32 vectors {}..{}", base, base + take),
    )
}

fn rerank_exact_candidates_from_file(
    file: &File,
    query: &[f32],
    candidates: &[Hit],
) -> Result<[usize; K], String> {
    let mut scratch = vec![0.0f32; HNSW_RAW_READ_SPAN_VECTORS * DIM];
    let mut heap = [Hit {
        score: f32::NEG_INFINITY,
        idx: usize::MAX,
    }; K];
    // Graph/quantized candidates are normally ordered by approximate score,
    // which produces scattered raw-file reads. Sort the bounded candidate
    // set by row id for the exact pass so Android's page cache and the flash
    // controller can coalesce nearby reads. The top-k heap is still scored
    // by the real FP32 dot product, so this is a pure I/O optimization.
    let mut ordered = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.idx != usize::MAX)
        .collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|candidate| candidate.idx);
    let mut cursor = 0usize;
    while cursor < ordered.len() {
        let base = ordered[cursor].idx;
        let mut end = cursor + 1;
        while end < ordered.len()
            && ordered[end].idx - ordered[end - 1].idx == 1
            && ordered[end].idx - base < HNSW_RAW_READ_SPAN_VECTORS
        {
            end += 1;
        }
        let take = ordered[end - 1].idx - base + 1;
        read_f32_vectors_at(&file, base, take, &mut scratch[..take * DIM])?;
        for candidate in &ordered[cursor..end] {
            let local = candidate.idx - base;
            let start = local * DIM;
            insert_hit(
                &mut heap,
                Hit {
                    score: dot(query, &scratch[start..start + DIM]),
                    idx: candidate.idx,
                },
            );
        }
        cursor = end;
    }
    Ok(finalize_heap(heap))
}

fn write_u32<W: Write>(writer: &mut W, value: u32, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_u64<W: Write>(writer: &mut W, value: u64, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_u32_slice<W: Write>(writer: &mut W, values: &[u32], path: &Path) -> Result<(), String> {
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        writer
            .write_all(bytes)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
    #[cfg(target_endian = "big")]
    {
        for &value in values {
            write_u32(writer, value, path)?;
        }
        Ok(())
    }
}

fn read_u32<R: Read>(reader: &mut R, path: &Path) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R, path: &Path) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32_slice<R: Read>(reader: &mut R, values: &mut [u32], path: &Path) -> Result<(), String> {
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 4)
        };
        reader
            .read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))
    }
    #[cfg(target_endian = "big")]
    {
        for value in values {
            *value = read_u32(reader, path)?;
        }
        Ok(())
    }
}

fn validate_vector_file(path: &Path, n: usize) -> Result<(), String> {
    let expected = n
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| format!("dataset too large: {} vectors", n))?;
    let meta_len = path
        .metadata()
        .map_err(|e| format!("metadata {}: {e}", path.display()))?
        .len() as usize;
    if meta_len < expected {
        return Err(format!(
            "expected at least {} bytes for {}x{} f32 vectors, got {}",
            expected,
            n,
            DIM,
            meta_len
        ));
    }
    Ok(())
}

fn vector_ram_cap_bytes(n_vectors: usize) -> usize {
    if n_vectors <= 50_000 {
        VECTOR_RAM_CAP_50K_BYTES
    } else {
        VECTOR_RAM_CAP_100K_BYTES
    }
}

fn query_vector_bytes() -> usize {
    (SELF_QUERIES + RANDOM_QUERIES) * DIM * std::mem::size_of::<f32>()
}

fn query_generation_reserve_bytes(nq: usize) -> usize {
    2 * nq * DIM * std::mem::size_of::<f32>()
}

fn raw_chunk_vectors(cap: usize) -> usize {
    raw_chunk_vectors_with_reserve(cap, query_vector_bytes())
}

fn raw_chunk_vectors_query_major(cap: usize) -> usize {
    raw_chunk_vectors_with_reserve(cap, QUERY_MAJOR_QUERY_BYTES)
}

fn fp16_chunk_vectors_query_major(cap: usize) -> usize {
    // A query-major FP16 range is held as on-disk half bits and converted as
    // each dot product runs. Keep the query, a small result/scratch reserve,
    // and the direct-read safety reserve inside the same pure-vector cap.
    let fixed = QUERY_MAJOR_QUERY_BYTES
        .saturating_add(READ_BUFFER_BYTES)
        .saturating_add(64 * 1024);
    cap.saturating_sub(fixed)
        .checked_div(DIM * std::mem::size_of::<u16>())
        .unwrap_or(0)
        .max(1)
}

fn raw_chunk_vectors_with_reserve(cap: usize, reserve_bytes: usize) -> usize {
    cap.saturating_sub(reserve_bytes.saturating_add(READ_BUFFER_BYTES))
        .checked_div(DIM * std::mem::size_of::<f32>())
        .unwrap_or(0)
        .max(1)
}

fn rotation_cache_bytes() -> usize {
    DIM * DIM * std::mem::size_of::<f32>()
}

fn quant_search_workspace_bytes(bit_width: usize) -> usize {
    let q_bytes = QUERY_BATCH * DIM * std::mem::size_of::<f32>();
    let lut_bytes = if bit_width <= 4 {
        QUERY_BATCH * (DIM / (8 / bit_width)) * 32
    } else {
        0
    };
    // Includes q_rot, calibrated queries, per-query LUTs, and a small result
    // buffer reserve. The 4-bit ARM path's score buffer is charged per vector
    // below because it is proportional to the loaded range size.
    q_bytes * 2 + lut_bytes + 64 * 1024
}

fn quant_resident_bytes(n_vectors: usize, bit_width: usize) -> usize {
    let bytes_per_vector = DIM * bit_width / 8;
    let n_blocks = (n_vectors + SIMD_BLOCK - 1) / SIMD_BLOCK;
    let n_byte_groups = DIM / (8 / bit_width);
    let packed = n_vectors * bytes_per_vector;
    let blocked = n_blocks * n_byte_groups * SIMD_BLOCK;
    let scales = n_vectors * std::mem::size_of::<f32>();
    let tqplus = DIM * 2 * std::mem::size_of::<f32>();
    let centroids = (1usize << bit_width) * std::mem::size_of::<f32>();
    let arm_score_buffer = if bit_width <= 4 {
        QUERY_BATCH * n_vectors * std::mem::size_of::<f32>()
    } else {
        0
    };
    packed
        + blocked
        + scales
        + tqplus
        + centroids
        + rotation_cache_bytes()
        + arm_score_buffer
}

fn quant_chunk_vectors(cap: usize, bit_width: usize) -> usize {
    let fixed = query_vector_bytes()
        .saturating_add(quant_search_workspace_bytes(bit_width))
        .saturating_add(READ_BUFFER_BYTES);
    let available = cap.saturating_sub(fixed);
    let bytes_per_vector = DIM * bit_width / 8;
    let mut candidate = available
        .checked_div(bytes_per_vector.saturating_mul(2).saturating_add(4))
        .unwrap_or(0)
        .max(1);
    candidate = (candidate / SIMD_BLOCK).max(1) * SIMD_BLOCK;
    while candidate > 1 && quant_resident_bytes(candidate, bit_width) > available {
        candidate = candidate.saturating_sub(SIMD_BLOCK);
    }
    candidate.max(1)
}

/// Resident bytes for a blocked sidecar range in the single-query path. The
/// packed representation and the multi-query score matrix are absent: only
/// blocked codes, scales, calibration, and shared search tables remain.
fn quant_query_major_resident_bytes(n_vectors: usize, bit_width: usize) -> usize {
    let n_blocks = (n_vectors + SIMD_BLOCK - 1) / SIMD_BLOCK;
    let n_byte_groups = DIM / (8 / bit_width);
    let blocked = n_blocks * n_byte_groups * SIMD_BLOCK;
    let scales = n_vectors * std::mem::size_of::<f32>();
    let tqplus = DIM * 2 * std::mem::size_of::<f32>();
    let centroids = (1usize << bit_width) * std::mem::size_of::<f32>();
    blocked + scales + tqplus + centroids + rotation_cache_bytes()
}

fn quant_chunk_vectors_query_major(cap: usize, bit_width: usize) -> usize {
    let fixed = QUERY_MAJOR_QUERY_BYTES
        .saturating_add(READ_BUFFER_BYTES)
        .saturating_add(64 * 1024);
    let available = cap.saturating_sub(fixed);
    let bytes_per_vector = DIM / (8 / bit_width) + std::mem::size_of::<f32>();
    let mut candidate = available
        .checked_sub(rotation_cache_bytes())
        .and_then(|x| x.checked_sub(DIM * 2 * std::mem::size_of::<f32>()))
        .and_then(|x| x.checked_sub((1usize << bit_width) * std::mem::size_of::<f32>()))
        .and_then(|x| x.checked_div(bytes_per_vector))
        .unwrap_or(0)
        .max(1);
    candidate = (candidate / SIMD_BLOCK).max(1) * SIMD_BLOCK;
    while candidate > 1
        && quant_query_major_resident_bytes(candidate, bit_width)
            .saturating_add(fixed)
            > cap
    {
        candidate = candidate.saturating_sub(SIMD_BLOCK);
    }
    candidate.max(1)
}

fn load_vector_range(
    path: &Path,
    start_vector: usize,
    n: usize,
    cap: usize,
    reserve_bytes: usize,
) -> Result<Vec<f32>, String> {
    let byte_offset = start_vector
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| format!("range starts too far into {}", path.display()))?;
    let byte_len = n
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| format!("range too large: {} vectors", n))?;
    let max_vector_bytes = cap.saturating_sub(reserve_bytes.saturating_add(READ_BUFFER_BYTES));
    if byte_len > max_vector_bytes {
        return Err(format!(
            "range of {} bytes exceeds the {} vector RAM cap after {} bytes of resident vector data",
            byte_len,
            human_bytes(cap as u64),
            reserve_bytes
        ));
    }
    let mut file = BufReader::new(
        File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
    );
    load_vector_range_from_reader(&mut file, byte_offset, byte_len, n * DIM, path)
}

fn load_vector_range_from_reader<R: Read + Seek>(
    file: &mut R,
    byte_offset: usize,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<f32>, String> {
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f32_values(file, byte_len, expected_values, path)
}

fn read_f32_values<R: Read>(
    file: &mut R,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<f32>, String> {
    if byte_len != expected_values.saturating_mul(std::mem::size_of::<f32>()) {
        return Err(format!(
            "invalid f32 byte length {} for {} values in {}",
            byte_len,
            expected_values,
            path.display()
        ));
    }

    // Android arm64 is little-endian. Read directly into the final f32
    // allocation instead of copying through a 1 MiB byte buffer and decoding
    // every scalar with from_le_bytes.
    #[cfg(target_endian = "little")]
    {
        let mut out = vec![0.0f32; expected_values];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
        };
        file.read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        return Ok(out);
    }

    #[cfg(target_endian = "big")]
    {
        let mut out = Vec::with_capacity(expected_values);
        let mut buf = vec![0u8; READ_BUFFER_BYTES.min(byte_len.max(4))];
        let mut carry = [0u8; 4];
        let mut carry_len = 0usize;
        let mut remaining = byte_len;
        while remaining > 0 {
            let take = remaining.min(buf.len());
            let read = file
                .read(&mut buf[..take])
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            let mut start = 0usize;
            if carry_len > 0 {
                let need = 4 - carry_len;
                if read < need {
                    carry[carry_len..carry_len + read].copy_from_slice(&buf[..read]);
                    carry_len += read;
                    remaining -= read;
                    continue;
                }
                carry[carry_len..4].copy_from_slice(&buf[..need]);
                out.push(f32::from_le_bytes(carry));
                carry_len = 0;
                start = need;
            }
            let body_len = ((read - start) / 4) * 4;
            for chunk in buf[start..start + body_len].chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let rem = read - start - body_len;
            if rem > 0 {
                carry[..rem].copy_from_slice(&buf[start + body_len..read]);
                carry_len = rem;
            }
            remaining -= read;
        }
        if carry_len != 0 || out.len() != expected_values {
            return Err(format!(
                "decoded {} f32 values from {}, expected {}",
                out.len(),
                path.display(),
                expected_values
            ));
        }
        Ok(out)
    }

    #[cfg(target_endian = "little")]
    unreachable!()
}

fn load_vector_range_into_reader<R: Read + Seek>(
    file: &mut R,
    byte_offset: usize,
    byte_len: usize,
    out: &mut [f32],
    path: &Path,
) -> Result<(), String> {
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f32_values_into(file, byte_len, out, path)
}

fn read_f32_values_into<R: Read>(
    file: &mut R,
    byte_len: usize,
    out: &mut [f32],
    path: &Path,
) -> Result<(), String> {
    if byte_len != out.len().saturating_mul(std::mem::size_of::<f32>()) {
        return Err(format!(
            "invalid f32 byte length {} for {} values in {}",
            byte_len,
            out.len(),
            path.display()
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
        };
        file.read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        return Ok(());
    }

    #[cfg(target_endian = "big")]
    {
        let decoded = read_f32_values(file, byte_len, out.len(), path)?;
        out.copy_from_slice(&decoded);
        return Ok(());
    }

    #[cfg(target_endian = "little")]
    unreachable!()
}

fn load_fp16_range_from_reader<R: Read + Seek>(
    file: &mut R,
    byte_offset: usize,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<f32>, String> {
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f16_values(file, byte_len, expected_values, path)
}

fn load_fp16_bits_range_from_reader<R: Read + Seek>(
    file: &mut R,
    byte_offset: usize,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<u16>, String> {
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f16_bits(file, byte_len, expected_values, path)
}

fn read_f16_values<R: Read>(
    file: &mut R,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(expected_values);
    let mut buf = vec![0u8; READ_BUFFER_BYTES.min(byte_len.max(2))];
    let mut carry = [0u8; 2];
    let mut carry_len = 0usize;
    let mut remaining = byte_len;
    while remaining > 0 {
        let take = remaining.min(buf.len());
        let read = file
            .read(&mut buf[..take])
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        let mut start = 0usize;
        if carry_len > 0 {
            let need = 2 - carry_len;
            if read < need {
                carry[carry_len..carry_len + read].copy_from_slice(&buf[..read]);
                carry_len += read;
                remaining -= read;
                continue;
            }
            carry[carry_len..2].copy_from_slice(&buf[..need]);
            out.push(f16::from_bits(u16::from_le_bytes(carry)).to_f32());
            carry_len = 0;
            start = need;
        }
        let body_len = ((read - start) / 2) * 2;
        for chunk in buf[start..start + body_len].chunks_exact(2) {
            out.push(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32());
        }
        let rem = read - start - body_len;
        if rem > 0 {
            carry[..rem].copy_from_slice(&buf[start + body_len..read]);
            carry_len = rem;
        }
        remaining -= read;
    }
    if carry_len != 0 || out.len() != expected_values {
        return Err(format!(
            "decoded {} FP16 values from {}, expected {}",
            out.len(),
            path.display(),
            expected_values
        ));
    }
    Ok(out)
}

fn read_f16_bits<R: Read>(
    file: &mut R,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<u16>, String> {
    if byte_len != expected_values.saturating_mul(std::mem::size_of::<u16>()) {
        return Err(format!(
            "invalid FP16 byte length {} for {} values in {}",
            byte_len,
            expected_values,
            path.display()
        ));
    }

    // The Android target is little-endian. Read the on-disk IEEE-754 half
    // bits directly into their final allocation: query-major search consumes
    // each value once, so a temporary decoded f32 chunk would only increase
    // RAM and memory traffic.
    #[cfg(target_endian = "little")]
    {
        let mut out = vec![0u16; expected_values];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
        };
        file.read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        return Ok(out);
    }

    #[cfg(target_endian = "big")]
    {
        let mut out = Vec::with_capacity(expected_values);
        let mut buf = vec![0u8; READ_BUFFER_BYTES.min(byte_len.max(2))];
        let mut carry = [0u8; 2];
        let mut carry_len = 0usize;
        let mut remaining = byte_len;
        while remaining > 0 {
            let take = remaining.min(buf.len());
            let read = file
                .read(&mut buf[..take])
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            let mut start = 0usize;
            if carry_len > 0 {
                let need = 2 - carry_len;
                if read < need {
                    carry[carry_len..carry_len + read].copy_from_slice(&buf[..read]);
                    carry_len += read;
                    remaining -= read;
                    continue;
                }
                carry[carry_len..2].copy_from_slice(&buf[..need]);
                out.push(u16::from_le_bytes(carry));
                carry_len = 0;
                start = need;
            }
            let body_len = ((read - start) / 2) * 2;
            for chunk in buf[start..start + body_len].chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            let rem = read - start - body_len;
            if rem > 0 {
                carry[..rem].copy_from_slice(&buf[start + body_len..read]);
                carry_len = rem;
            }
            remaining -= read;
        }
        if carry_len != 0 || out.len() != expected_values {
            return Err(format!(
                "read {} FP16 values from {}, expected {}",
                out.len(),
                path.display(),
                expected_values
            ));
        }
        return Ok(out);
    }

    #[cfg(target_endian = "little")]
    unreachable!()
}

fn load_fp16_bits_range_into_reader<R: Read + Seek>(
    file: &mut R,
    byte_offset: usize,
    byte_len: usize,
    out: &mut [u16],
    path: &Path,
) -> Result<(), String> {
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f16_bits_into(file, byte_len, out, path)
}

fn read_f16_bits_into<R: Read>(
    file: &mut R,
    byte_len: usize,
    out: &mut [u16],
    path: &Path,
) -> Result<(), String> {
    if byte_len != out.len().saturating_mul(std::mem::size_of::<u16>()) {
        return Err(format!(
            "invalid FP16 byte length {} for {} values in {}",
            byte_len,
            out.len(),
            path.display()
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
        };
        file.read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        return Ok(());
    }

    #[cfg(target_endian = "big")]
    {
        let decoded = read_f16_bits(file, byte_len, out.len(), path)?;
        out.copy_from_slice(&decoded);
        return Ok(());
    }

    #[cfg(target_endian = "little")]
    unreachable!()
}

fn make_random_queries(
    path: &Path,
    n: usize,
    nq: usize,
    cap: usize,
) -> Result<Vec<f32>, String> {
    let mut rng = StdRng::seed_from_u64(0x5451_2026);
    let mut first_indices = Vec::with_capacity(nq);
    let mut second_indices = Vec::with_capacity(nq);
    let mut alphas = Vec::with_capacity(nq);
    for _ in 0..nq {
        first_indices.push(rng.gen_range(0..n));
        second_indices.push(rng.gen_range(0..n));
        alphas.push(rng.gen_range(0.15..0.85));
    }
    let first_vectors = load_selected_vectors(path, n, &first_indices, cap)?;
    let second_vectors = load_selected_vectors(path, n, &second_indices, cap)?;
    let mut out = vec![0.0f32; nq * DIM];
    for q in 0..nq {
        let va = &first_vectors[q * DIM..(q + 1) * DIM];
        let vb = &second_vectors[q * DIM..(q + 1) * DIM];
        let alpha = alphas[q];
        let row = &mut out[q * DIM..(q + 1) * DIM];
        for d in 0..DIM {
            row[d] = alpha * va[d] + (1.0 - alpha) * vb[d];
        }
        normalize(row);
    }
    Ok(out)
}

fn load_selected_vectors(
    path: &Path,
    n: usize,
    indices: &[usize],
    cap: usize,
) -> Result<Vec<f32>, String> {
    let mut requests: Vec<(usize, usize)> = indices
        .iter()
        .copied()
        .enumerate()
        .map(|(output_index, vector_index)| (vector_index, output_index))
        .collect();
    for &(vector_index, _) in &requests {
        if vector_index >= n {
            return Err(format!(
                "selected vector {} is outside dataset of {} vectors",
                vector_index, n
            ));
        }
    }
    requests.sort_unstable();

    let mut out = vec![0.0f32; indices.len() * DIM];
    let reserve_bytes = query_generation_reserve_bytes(indices.len());
    let per_chunk = raw_chunk_vectors_with_reserve(cap, reserve_bytes);
    let mut cursor = 0usize;
    while cursor < requests.len() {
        let chunk_base = (requests[cursor].0 / per_chunk) * per_chunk;
        let take = (n - chunk_base).min(per_chunk);
        let chunk = load_vector_range(path, chunk_base, take, cap, reserve_bytes)?;
        while cursor < requests.len() && requests[cursor].0 < chunk_base + take {
            let (vector_index, output_index) = requests[cursor];
            let local = vector_index - chunk_base;
            out[output_index * DIM..(output_index + 1) * DIM]
                .copy_from_slice(&chunk[local * DIM..(local + 1) * DIM]);
            cursor += 1;
        }
    }
    Ok(out)
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

fn exact_topk_file_query_major(
    path: &Path,
    n: usize,
    queries: &[f32],
    cap: usize,
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    let per_chunk = raw_chunk_vectors_query_major(cap);
    let mut reader = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    // Reuse one bounded staging allocation for every query and range. The
    // previous path allocated and dropped a fresh Vec for every range of
    // every online query, adding allocator and zeroing work without changing
    // the query-major scheduling model.
    let mut chunk = vec![0.0f32; per_chunk * DIM];
    let mut results = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * DIM..(qi + 1) * DIM];
        let mut heap = [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        // Query-major ordering models an online request: this query scans
        // every bounded chunk before the next query starts. Keep one file
        // descriptor/reader across probes so the latency path does not pay a
        // fresh open and BufReader allocation for every request.
        let mut base = 0usize;
        while base < n {
            let take = (n - base).min(per_chunk);
            let values = take * DIM;
            load_vector_range_into_reader(
                &mut reader,
                base * DIM * 4,
                take * DIM * 4,
                &mut chunk[..values],
                path,
            )?;
            for (local_idx, v) in chunk[..values].chunks_exact(DIM).enumerate() {
                insert_hit(
                    &mut heap,
                    Hit {
                        score: dot(q, v),
                        idx: base + local_idx,
                    },
                );
            }
            base += take;
        }
        results.push(finalize_heap(heap));
    }
    Ok(results)
}

fn exact_topk_file_batched(
    path: &Path,
    n: usize,
    queries: &[f32],
    cap: usize,
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    if nq == 0 {
        return Ok(Vec::new());
    }

    let empty_heap = [Hit {
        score: f32::NEG_INFINITY,
        idx: usize::MAX,
    }; K];
    let mut heaps = vec![empty_heap; nq];
    // The query-group pass still reread a whole staged chunk for every query
    // group. Tile the staged vectors instead: each worker keeps a 64-vector
    // tile hot while it scores every query, then contributes bounded local
    // heaps that are merged after the chunk. This keeps the exact FP32
    // definition, but changes the setup pass from repeated chunk streaming to
    // cache-reused vector tiles.
    let worker_heap_bytes = nq
        .saturating_mul(K)
        .saturating_mul(std::mem::size_of::<Hit>());
    let reserve_bytes = query_vector_bytes().saturating_add(
        worker_heap_bytes.saturating_mul(rayon::current_num_threads().max(1)),
    );
    for_each_vector_chunk_with_reserve(path, n, cap, reserve_bytes, |base, chunk| {
        let partial = chunk
            .par_chunks(EXACT_TRUTH_VECTOR_TILE * DIM)
            .enumerate()
            .fold(
                || vec![empty_heap; nq],
                |mut local_heaps, (tile_index, tile)| {
                    let mut qbase = 0usize;
                    while qbase + 4 <= nq {
                        for (local_idx, vector) in tile.chunks_exact(DIM).enumerate() {
                            let scores = dot_four_queries(queries, qbase, vector);
                            for lane in 0..4 {
                                insert_hit(
                                    &mut local_heaps[qbase + lane],
                                    Hit {
                                        score: scores[lane],
                                        idx: base + tile_index * EXACT_TRUTH_VECTOR_TILE + local_idx,
                                    },
                                );
                            }
                        }
                        qbase += 4;
                    }
                    while qbase < nq {
                        let query = &queries[qbase * DIM..(qbase + 1) * DIM];
                        for (local_idx, vector) in tile.chunks_exact(DIM).enumerate() {
                            insert_hit(
                                &mut local_heaps[qbase],
                                Hit {
                                    score: dot(query, vector),
                                    idx: base + tile_index * EXACT_TRUTH_VECTOR_TILE + local_idx,
                                },
                            );
                        }
                        qbase += 1;
                    }
                    local_heaps
                },
            )
            .reduce(
                || vec![empty_heap; nq],
                |mut left, right| {
                    for (left_heap, right_heap) in left.iter_mut().zip(right) {
                        for hit in right_heap {
                            insert_hit(left_heap, hit);
                        }
                    }
                    left
                },
            );
        for (heap, partial_heap) in heaps.iter_mut().zip(partial) {
            for hit in partial_heap {
                insert_hit(heap, hit);
            }
        }
        Ok(())
    })?;
    Ok(heaps.into_iter().map(finalize_heap).collect())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_four_queries(queries: &[f32], qbase: usize, vector: &[f32]) -> [f32; 4] {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    let query0 = &queries[qbase * DIM..(qbase + 1) * DIM];
    let query1 = &queries[(qbase + 1) * DIM..(qbase + 2) * DIM];
    let query2 = &queries[(qbase + 2) * DIM..(qbase + 3) * DIM];
    let query3 = &queries[(qbase + 3) * DIM..(qbase + 4) * DIM];
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= DIM {
            let v0 = vld1q_f32(vector.as_ptr().add(i));
            let v1 = vld1q_f32(vector.as_ptr().add(i + 4));
            let v2 = vld1q_f32(vector.as_ptr().add(i + 8));
            let v3 = vld1q_f32(vector.as_ptr().add(i + 12));
            acc0 = vmlaq_f32(acc0, v0, vld1q_f32(query0.as_ptr().add(i)));
            acc0 = vmlaq_f32(acc0, v1, vld1q_f32(query0.as_ptr().add(i + 4)));
            acc0 = vmlaq_f32(acc0, v2, vld1q_f32(query0.as_ptr().add(i + 8)));
            acc0 = vmlaq_f32(acc0, v3, vld1q_f32(query0.as_ptr().add(i + 12)));
            acc1 = vmlaq_f32(acc1, v0, vld1q_f32(query1.as_ptr().add(i)));
            acc1 = vmlaq_f32(acc1, v1, vld1q_f32(query1.as_ptr().add(i + 4)));
            acc1 = vmlaq_f32(acc1, v2, vld1q_f32(query1.as_ptr().add(i + 8)));
            acc1 = vmlaq_f32(acc1, v3, vld1q_f32(query1.as_ptr().add(i + 12)));
            acc2 = vmlaq_f32(acc2, v0, vld1q_f32(query2.as_ptr().add(i)));
            acc2 = vmlaq_f32(acc2, v1, vld1q_f32(query2.as_ptr().add(i + 4)));
            acc2 = vmlaq_f32(acc2, v2, vld1q_f32(query2.as_ptr().add(i + 8)));
            acc2 = vmlaq_f32(acc2, v3, vld1q_f32(query2.as_ptr().add(i + 12)));
            acc3 = vmlaq_f32(acc3, v0, vld1q_f32(query3.as_ptr().add(i)));
            acc3 = vmlaq_f32(acc3, v1, vld1q_f32(query3.as_ptr().add(i + 4)));
            acc3 = vmlaq_f32(acc3, v2, vld1q_f32(query3.as_ptr().add(i + 8)));
            acc3 = vmlaq_f32(acc3, v3, vld1q_f32(query3.as_ptr().add(i + 12)));
            i += 16;
        }
        [vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)]
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot_four_queries(queries: &[f32], qbase: usize, vector: &[f32]) -> [f32; 4] {
    let mut scores = [0.0f32; 4];
    for dim in 0..DIM {
        let value = vector[dim];
        for lane in 0..4 {
            scores[lane] += queries[(qbase + lane) * DIM + dim] * value;
        }
    }
    scores
}

fn time_fp32_probes(
    path: &Path,
    n: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    cap: usize,
) -> Result<QueryMajorTiming, String> {
    let mut total_ms = 0.0f64;
    let mut first_ms = 0.0f64;
    let mut sample_count = 0usize;
    for queries in [self_queries, random_queries] {
        let set_count = (queries.len() / DIM).min(TIMED_PROBES_PER_SET);
        for qi in 0..set_count {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let start = Instant::now();
            exact_topk_file_query_major(path, n, q, cap)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if sample_count == 0 {
                first_ms = elapsed;
            }
            total_ms += elapsed;
            sample_count += 1;
        }
    }
    Ok(QueryMajorTiming::from_samples(total_ms, first_ms, sample_count))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    debug_assert_eq!(a.len(), DIM);
    debug_assert_eq!(b.len(), DIM);
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= DIM {
            acc0 = vmlaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc1 = vmlaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            acc2 = vmlaq_f32(acc2, vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
            acc3 = vmlaq_f32(acc3, vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
            i += 16;
        }
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i < DIM {
            acc = vmlaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            i += 4;
        }
        vaddvq_f32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut i = 0usize;
    while i + 4 <= DIM {
        a0 += a[i] * b[i];
        a1 += a[i + 1] * b[i + 1];
        a2 += a[i + 2] * b[i + 2];
        a3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    a0 + a1 + a2 + a3
}

#[inline(always)]
fn dot_fp16(query: &[f32], values: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), DIM);
    debug_assert_eq!(values.len(), DIM);
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut i = 0usize;
    while i + 4 <= DIM {
        a0 += query[i] * f16::from_bits(values[i]).to_f32();
        a1 += query[i + 1] * f16::from_bits(values[i + 1]).to_f32();
        a2 += query[i + 2] * f16::from_bits(values[i + 2]).to_f32();
        a3 += query[i + 3] * f16::from_bits(values[i + 3]).to_f32();
        i += 4;
    }
    while i < DIM {
        a0 += query[i] * f16::from_bits(values[i]).to_f32();
        i += 1;
    }
    a0 + a1 + a2 + a3
}

#[inline(always)]
fn dot_fp16_values(a: &[u16], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), DIM);
    debug_assert_eq!(b.len(), DIM);
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut i = 0usize;
    while i + 4 <= DIM {
        a0 += f16::from_bits(a[i]).to_f32() * f16::from_bits(b[i]).to_f32();
        a1 += f16::from_bits(a[i + 1]).to_f32() * f16::from_bits(b[i + 1]).to_f32();
        a2 += f16::from_bits(a[i + 2]).to_f32() * f16::from_bits(b[i + 2]).to_f32();
        a3 += f16::from_bits(a[i + 3]).to_f32() * f16::from_bits(b[i + 3]).to_f32();
        i += 4;
    }
    while i < DIM {
        a0 += f16::from_bits(a[i]).to_f32() * f16::from_bits(b[i]).to_f32();
        i += 1;
    }
    a0 + a1 + a2 + a3
}

fn insert_hit(heap: &mut [Hit; K], hit: Hit) {
    // Keep the fixed K-entry array as a min-heap. Most scanned vectors are
    // below the current threshold, so the common path is one comparison
    // instead of scanning all K entries to rediscover the minimum.
    if hit.score <= heap[0].score {
        return;
    }
    heap[0] = hit;
    let mut pos = 0usize;
    loop {
        let left = pos * 2 + 1;
        if left >= K {
            break;
        }
        let right = left + 1;
        let child = if right < K && heap[right].score < heap[left].score {
            right
        } else {
            left
        };
        if heap[pos].score <= heap[child].score {
            break;
        }
        heap.swap(pos, child);
        pos = child;
    }
}

fn insert_hit_dynamic(heap: &mut [Hit], hit: Hit) {
    if heap.is_empty() || hit.score <= heap[0].score {
        return;
    }
    heap[0] = hit;
    let mut pos = 0usize;
    loop {
        let left = pos * 2 + 1;
        if left >= heap.len() {
            break;
        }
        let right = left + 1;
        let child = if right < heap.len() && heap[right].score < heap[left].score {
            right
        } else {
            left
        };
        if heap[pos].score <= heap[child].score {
            break;
        }
        heap.swap(pos, child);
        pos = child;
    }
}

fn bench_fp16_query_major(
    dataset: &DatasetInput,
    vector_path: &Path,
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    output_dir: &Path,
    vector_ram_cap: usize,
) -> Result<Row, String> {
    let path = output_dir.join(format!("{}_flat_fp16.f16", dataset.id));
    let file = File::create(&path)
        .map_err(|e| format!("create FP16 index {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut index_ms = 0.0f64;
    let mut write_ms = 0.0f64;
    for_each_vector_chunk(vector_path, dataset.vectors, vector_ram_cap, |_, chunk| {
        let encode_start = Instant::now();
        let mut encoded = Vec::with_capacity(chunk.len() * 2);
        for value in chunk {
            encoded.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        index_ms += encode_start.elapsed().as_secs_f64() * 1000.0;

        let write_start = Instant::now();
        writer
            .write_all(&encoded)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        write_ms += write_start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    })?;
    let flush_start = Instant::now();
    writer
        .flush()
        .map_err(|e| format!("flush {}: {e}", path.display()))?;
    write_ms += flush_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0);

    let self_results =
        flat_fp16_topk_file_batched(&path, dataset.vectors, self_queries, vector_ram_cap)?;
    let random_results =
        flat_fp16_topk_file_batched(&path, dataset.vectors, random_queries, vector_ram_cap)?;
    let timing = time_fp16_probes(
        &path,
        dataset.vectors,
        self_queries,
        random_queries,
        vector_ram_cap,
    )?;

    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let probe_ms = timing.total_ms;
    let ms_per_query = timing.average_ms();

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: "16".to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: "0.0".to_string(),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", probe_ms / 2.0),
        random_search_ms: format!("{:.1}", probe_ms / 2.0),
        ms_per_query: format!("{:.3}", ms_per_query),
        first_query_ms: format!("{:.3}", timing.first_ms),
        warm_ms_per_query: format!("{:.3}", timing.warm_ms_per_query),
        index_rom: human_bytes(rom),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label_query_major(vector_ram_cap, 16),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn bench_fp16_batched(
    dataset: &DatasetInput,
    vector_path: &Path,
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    output_dir: &Path,
    vector_ram_cap: usize,
) -> Result<Row, String> {
    let path = output_dir.join(format!("{}_flat_fp16.f16", dataset.id));
    let file = File::create(&path)
        .map_err(|e| format!("create FP16 index {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut index_ms = 0.0f64;
    let mut write_ms = 0.0f64;
    for_each_vector_chunk(vector_path, dataset.vectors, vector_ram_cap, |_, chunk| {
        let encode_start = Instant::now();
        let mut encoded = Vec::with_capacity(chunk.len() * 2);
        for value in chunk {
            encoded.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        index_ms += encode_start.elapsed().as_secs_f64() * 1000.0;

        let write_start = Instant::now();
        writer
            .write_all(&encoded)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        write_ms += write_start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    })?;
    let flush_start = Instant::now();
    writer
        .flush()
        .map_err(|e| format!("flush {}: {e}", path.display()))?;
    write_ms += flush_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0);

    let search_start = Instant::now();
    let self_results =
        flat_fp16_topk_file_batched(&path, dataset.vectors, self_queries, vector_ram_cap)?;
    let random_results =
        flat_fp16_topk_file_batched(&path, dataset.vectors, random_queries, vector_ram_cap)?;
    let search_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let timed_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: "16".to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: "0.0".to_string(),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", search_ms / 2.0),
        random_search_ms: format!("{:.1}", search_ms / 2.0),
        ms_per_query: format!("{:.3}", search_ms / timed_q),
        first_query_ms: "n/a".to_string(),
        warm_ms_per_query: "n/a".to_string(),
        index_rom: human_bytes(rom),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label(vector_ram_cap, 16),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn flat_fp16_topk_file_query_major(
    path: &Path,
    n: usize,
    queries: &[f32],
    vector_ram_cap: usize,
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    let per_chunk = fp16_chunk_vectors_query_major(vector_ram_cap);
    let mut reader = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    // Reuse one bounded FP16 staging allocation across online probes. The
    // persisted file remains disk-backed; this only removes repeated range
    // allocation and initialization from the latency path.
    let mut chunk = vec![0u16; per_chunk * DIM];
    let mut results = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * DIM..(qi + 1) * DIM];
        let mut heap = [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        let mut base = 0usize;
        while base < n {
            let take = (n - base).min(per_chunk);
            let values = take * DIM;
            load_fp16_bits_range_into_reader(
                &mut reader,
                base * DIM * 2,
                take * DIM * 2,
                &mut chunk[..values],
                path,
            )?;
            for (local_idx, v) in chunk[..values].chunks_exact(DIM).enumerate() {
                insert_hit(
                    &mut heap,
                    Hit {
                        score: dot_fp16(q, v),
                        idx: base + local_idx,
                    },
                );
            }
            base += take;
        }
        results.push(finalize_heap(heap));
    }
    Ok(results)
}

fn flat_fp16_topk_file_batched(
    path: &Path,
    n: usize,
    queries: &[f32],
    vector_ram_cap: usize,
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    let mut heaps = vec![
        [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        nq
    ];
    let per_chunk = raw_chunk_vectors(vector_ram_cap);
    let mut reader = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut base = 0usize;
    while base < n {
        let take = (n - base).min(per_chunk);
        let chunk = load_fp16_range_from_reader(
            &mut reader,
            base * DIM * 2,
            take * DIM * 2,
            take * DIM,
            path,
        )?;
        heaps.par_iter_mut().enumerate().for_each(|(qi, heap)| {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            for (local_idx, v) in chunk.chunks_exact(DIM).enumerate() {
                insert_hit(
                    heap,
                    Hit {
                        score: dot(q, v),
                        idx: base + local_idx,
                    },
                );
            }
        });
        base += take;
    }
    Ok(heaps.into_iter().map(finalize_heap).collect())
}

fn time_fp16_probes(
    path: &Path,
    n: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    cap: usize,
) -> Result<QueryMajorTiming, String> {
    let mut total_ms = 0.0f64;
    let mut first_ms = 0.0f64;
    let mut sample_count = 0usize;
    for queries in [self_queries, random_queries] {
        let set_count = (queries.len() / DIM).min(TIMED_PROBES_PER_SET);
        for qi in 0..set_count {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let start = Instant::now();
            flat_fp16_topk_file_query_major(path, n, q, cap)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if sample_count == 0 {
                first_ms = elapsed;
            }
            total_ms += elapsed;
            sample_count += 1;
        }
    }
    Ok(QueryMajorTiming::from_samples(total_ms, first_ms, sample_count))
}

fn bench_quant_query_major(
    dataset: &DatasetInput,
    bit_width: usize,
    vector_path: &Path,
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    output_dir: &Path,
    vector_ram_cap: usize,
) -> Result<Row, String> {
    let index_start = Instant::now();
    let mut index = TurboQuantIndex::new(DIM, bit_width).map_err(|e| format!("{e:?}"))?;
    for_each_vector_chunk(vector_path, dataset.vectors, vector_ram_cap, |_, chunk| {
        index.add(&chunk);
        Ok(())
    })?;
    let index_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    let path = index_path(output_dir, &dataset.id, bit_width);
    let write_start = Instant::now();
    index
        .write(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    let blocked_path = blocked_index_path(output_dir, &dataset.id, bit_width);
    index
        .write_blocked(&blocked_path)
        .map_err(|e| format!("write {}: {e}", blocked_path.display()))?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0);

    drop(index);
    let cache_start = Instant::now();
    let search_cache = SearchCache::new(DIM, bit_width);
    let shared_cache_ms = cache_start.elapsed().as_secs_f64() * 1000.0;
    let (self_results, random_results, _batch_prepare_ms, _batch_search_ms) =
        quant_topk_two_ranges_batched_rerank(
            &path,
            vector_path,
            dataset.vectors,
            bit_width,
            self_queries,
            random_queries,
            vector_ram_cap,
            &search_cache,
        )?;
    let (timing, probe_prepare_ms) = time_quant_probes(
        &blocked_path,
        vector_path,
        dataset.vectors,
        bit_width,
        self_queries,
        random_queries,
        vector_ram_cap,
        &search_cache,
    )?;

    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let probe_ms = timing.total_ms;
    let ms_per_query = timing.average_ms();

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: bit_width.to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", shared_cache_ms + probe_prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", probe_ms / 2.0),
        random_search_ms: format!("{:.1}", probe_ms / 2.0),
        ms_per_query: format!("{:.3}", ms_per_query),
        first_query_ms: format!("{:.3}", timing.first_ms),
        warm_ms_per_query: format!("{:.3}", timing.warm_ms_per_query),
        index_rom: human_bytes(rom),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label_query_major(vector_ram_cap, bit_width),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn bench_quant_batched(
    dataset: &DatasetInput,
    bit_width: usize,
    vector_path: &Path,
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    output_dir: &Path,
    vector_ram_cap: usize,
) -> Result<Row, String> {
    let index_start = Instant::now();
    let mut index = TurboQuantIndex::new(DIM, bit_width).map_err(|e| format!("{e:?}"))?;
    for_each_vector_chunk(vector_path, dataset.vectors, vector_ram_cap, |_, chunk| {
        index.add(&chunk);
        Ok(())
    })?;
    let index_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    let path = index_path(output_dir, &dataset.id, bit_width);
    let write_start = Instant::now();
    index
        .write(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0);
    drop(index);

    let cache_start = Instant::now();
    let search_cache = SearchCache::new(DIM, bit_width);
    let shared_cache_ms = cache_start.elapsed().as_secs_f64() * 1000.0;
    let (self_results, random_results, range_prepare_ms, search_ms) = if bit_width == 4 {
        // The direct 4-bit top-10 is too noisy for a recall comparison. Keep
        // the range-major throughput schedule, but exact-rerank a bounded
        // candidate pool from the disk-backed FP32 store before reporting
        // R@10. This preserves the 30/50 MiB working-set contract while
        // avoiding the misleading ~88% raw 4-bit recall seen in B.
        quant_topk_two_ranges_batched_rerank(
            &path,
            vector_path,
            dataset.vectors,
            bit_width,
            self_queries,
            random_queries,
            vector_ram_cap,
            &search_cache,
        )?
    } else {
        quant_topk_two_ranges_batched(
            &path,
            dataset.vectors,
            bit_width,
            self_queries,
            random_queries,
            vector_ram_cap,
            &search_cache,
        )?
    };

    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let timed_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: bit_width.to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", shared_cache_ms + range_prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", search_ms / 2.0),
        random_search_ms: format!("{:.1}", search_ms / 2.0),
        ms_per_query: format!("{:.3}", search_ms / timed_q),
        first_query_ms: "n/a".to_string(),
        warm_ms_per_query: "n/a".to_string(),
        index_rom: human_bytes(rom),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label(vector_ram_cap, bit_width),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn quant_topk_two_ranges_batched(
    path: &Path,
    n: usize,
    bit_width: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    vector_ram_cap: usize,
    search_cache: &SearchCache,
) -> Result<(Vec<[usize; K]>, Vec<[usize; K]>, f64, f64), String> {
    let self_nq = self_queries.len() / DIM;
    let random_nq = random_queries.len() / DIM;
    let mut self_heaps = vec![
        [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        self_nq
    ];
    let mut random_heaps = vec![
        [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        random_nq
    ];
    let per_chunk = quant_chunk_vectors(vector_ram_cap, bit_width);
    let mut base = 0usize;
    let mut prepare_ms = 0.0f64;
    let mut search_ms = 0.0f64;
    let mut reader = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    while base < n {
        let take = (n - base).min(per_chunk);
        let prepare_start = Instant::now();
        let index = TurboQuantIndex::load_range_from_reader(&mut reader, base, take)
            .map_err(|e| format!("load range {} [{}..{}]: {e}", path.display(), base, base + take))?;
        index.prepare_with_cache(search_cache);
        prepare_ms += prepare_start.elapsed().as_secs_f64() * 1000.0;
        let search_start = Instant::now();
        search_quant_query_batches(&index, self_queries, base, &mut self_heaps);
        search_quant_query_batches(&index, random_queries, base, &mut random_heaps);
        search_ms += search_start.elapsed().as_secs_f64() * 1000.0;
        base += take;
    }

    Ok((
        self_heaps.into_iter().map(finalize_heap).collect(),
        random_heaps.into_iter().map(finalize_heap).collect(),
        prepare_ms,
        search_ms,
    ))
}

fn quant_topk_two_ranges_batched_rerank(
    path: &Path,
    raw_path: &Path,
    n: usize,
    bit_width: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    vector_ram_cap: usize,
    search_cache: &SearchCache,
) -> Result<(Vec<[usize; K]>, Vec<[usize; K]>, f64, f64), String> {
    let self_nq = self_queries.len() / DIM;
    let random_nq = random_queries.len() / DIM;
    let mut self_heaps = vec![
        vec![Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; QUANT_RERANK_CANDIDATES];
        self_nq
    ];
    let mut random_heaps = vec![
        vec![Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; QUANT_RERANK_CANDIDATES];
        random_nq
    ];
    let per_chunk = quant_chunk_vectors(vector_ram_cap, bit_width);
    let mut base = 0usize;
    let mut prepare_ms = 0.0f64;
    let mut search_ms = 0.0f64;
    let mut reader = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    while base < n {
        let take = (n - base).min(per_chunk);
        let prepare_start = Instant::now();
        let index = TurboQuantIndex::load_range_from_reader(&mut reader, base, take)
            .map_err(|e| format!("load range {} [{}..{}]: {e}", path.display(), base, base + take))?;
        index.prepare_with_cache(search_cache);
        prepare_ms += prepare_start.elapsed().as_secs_f64() * 1000.0;
        let search_start = Instant::now();
        search_quant_query_candidates(&index, self_queries, base, &mut self_heaps);
        search_quant_query_candidates(&index, random_queries, base, &mut random_heaps);
        search_ms += search_start.elapsed().as_secs_f64() * 1000.0;
        base += take;
    }

    let rerank_start = Instant::now();
    let raw_file = File::open(raw_path)
        .map_err(|e| format!("open raw FP32 store {}: {e}", raw_path.display()))?;
    let self_results = self_heaps
        .iter()
        .enumerate()
        .map(|(qi, heap)| {
            rerank_exact_candidates_from_file(
                &raw_file,
                &self_queries[qi * DIM..(qi + 1) * DIM],
                heap,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let random_results = random_heaps
        .iter()
        .enumerate()
        .map(|(qi, heap)| {
            rerank_exact_candidates_from_file(
                &raw_file,
                &random_queries[qi * DIM..(qi + 1) * DIM],
                heap,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
    Ok((self_results, random_results, prepare_ms, search_ms + rerank_ms))
}

fn time_quant_probes(
    path: &Path,
    raw_path: &Path,
    n: usize,
    bit_width: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    vector_ram_cap: usize,
    search_cache: &SearchCache,
) -> Result<(QueryMajorTiming, f64), String> {
    let mut total_ms = 0.0f64;
    let mut first_ms = 0.0f64;
    let mut sample_count = 0usize;
    let mut prepare_ms = 0.0f64;
    let raw_file = File::open(raw_path)
        .map_err(|e| format!("open raw FP32 store {}: {e}", raw_path.display()))?;
    let mut blocked_file = File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    for queries in [self_queries, random_queries] {
        let count = (queries.len() / DIM).min(TIMED_PROBES_PER_SET);
        for qi in 0..count {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let (_ids, query_ms, staged_ms) =
                quant_topk_one_query(
                    path,
                    &mut blocked_file,
                    &raw_file,
                    n,
                    bit_width,
                    q,
                    vector_ram_cap,
                    search_cache,
                )?;
            if sample_count == 0 {
                first_ms = query_ms;
            }
            total_ms += query_ms;
            prepare_ms += staged_ms;
            sample_count += 1;
        }
    }
    Ok((
        QueryMajorTiming::from_samples(total_ms, first_ms, sample_count),
        prepare_ms,
    ))
}

fn search_quant_query_batches(
    index: &TurboQuantIndex,
    queries: &[f32],
    global_base: usize,
    heaps: &mut [[Hit; K]],
) {
    let nq = queries.len() / DIM;
    for query_base in (0..nq).step_by(QUERY_BATCH) {
        let query_take = (nq - query_base).min(QUERY_BATCH);
        let query_start = query_base * DIM;
        let query_end = (query_base + query_take) * DIM;
        let results = index.search(&queries[query_start..query_end], K);
        for query_offset in 0..query_take {
            let result_offset = query_offset * K;
            let heap = &mut heaps[query_base + query_offset];
            for rank in 0..results.k.min(K) {
                let score = results.scores[result_offset + rank];
                let local_idx = results.indices[result_offset + rank];
                if local_idx >= 0 && score.is_finite() {
                    insert_hit(
                        heap,
                        Hit {
                            score,
                            idx: global_base + local_idx as usize,
                        },
                    );
                }
            }
        }
    }
}

fn search_quant_query_candidates(
    index: &TurboQuantIndex,
    queries: &[f32],
    global_base: usize,
    heaps: &mut [Vec<Hit>],
) {
    let nq = queries.len() / DIM;
    for query_base in (0..nq).step_by(QUERY_BATCH) {
        let query_take = (nq - query_base).min(QUERY_BATCH);
        let query_start = query_base * DIM;
        let query_end = (query_base + query_take) * DIM;
        let results = index.search(&queries[query_start..query_end], QUANT_RERANK_CANDIDATES);
        for query_offset in 0..query_take {
            let result_offset = query_offset * results.k;
            let heap = &mut heaps[query_base + query_offset];
            for rank in 0..results.k.min(QUANT_RERANK_CANDIDATES) {
                let score = results.scores[result_offset + rank];
                let local_idx = results.indices[result_offset + rank];
                if local_idx >= 0 && score.is_finite() {
                    insert_hit_dynamic(
                        heap,
                        Hit {
                            score,
                            idx: global_base + local_idx as usize,
                        },
                    );
                }
            }
        }
    }
}

fn quant_topk_one_query(
    path: &Path,
    blocked_reader: &mut File,
    raw_file: &File,
    n: usize,
    bit_width: usize,
    query: &[f32],
    vector_ram_cap: usize,
    search_cache: &SearchCache,
) -> Result<([usize; K], f64, f64), String> {
    let per_chunk = quant_chunk_vectors_query_major(vector_ram_cap, bit_width);
    let mut candidate_heap = vec![Hit {
        score: f32::NEG_INFINITY,
        idx: usize::MAX,
    }; QUANT_RERANK_CANDIDATES];
    let mut base = 0usize;
    let mut total_ms = 0.0f64;
    let mut staged_ms = 0.0f64;
    while base < n {
        let take = (n - base).min(per_chunk);
        let stage_start = Instant::now();
        let index = TurboQuantIndex::load_blocked_range_from_reader(blocked_reader, base, take)
            .map_err(|e| format!("load range {} [{}..{}]: {e}", path.display(), base, base + take))?;
        index.prepare_with_cache(search_cache);
        let stage_elapsed = stage_start.elapsed().as_secs_f64() * 1000.0;
        staged_ms += stage_elapsed;

        let search_start = Instant::now();
        // Expand the approximate candidate set before the final exact
        // FP32 re-rank. Quantized top-10 alone is especially lossy at 4
        // bits; a bounded candidate pool preserves recall without bringing
        // the full raw store into RAM.
        let results = index.search_one(query, QUANT_RERANK_CANDIDATES);
        for rank in 0..results.k.min(QUANT_RERANK_CANDIDATES) {
            let local_idx = results.indices[rank];
            let score = results.scores[rank];
            if local_idx >= 0 && score.is_finite() {
                insert_hit_dynamic(
                    &mut candidate_heap,
                    Hit {
                        score,
                        idx: base + local_idx as usize,
                    },
                );
            }
        }
        total_ms += stage_elapsed + search_start.elapsed().as_secs_f64() * 1000.0;
        base += take;
    }

    let rerank_start = Instant::now();
    let result = rerank_exact_candidates_from_file(raw_file, query, &candidate_heap)?;
    total_ms += rerank_start.elapsed().as_secs_f64() * 1000.0;
    Ok((result, total_ms, staged_ms))
}

fn finalize_heap(mut heap: [Hit; K]) -> [usize; K] {
    heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    let mut ids = [0usize; K];
    for i in 0..K {
        ids[i] = heap[i].idx;
    }
    ids
}

fn for_each_vector_chunk<F>(path: &Path, n: usize, cap: usize, f: F) -> Result<(), String>
where
    F: FnMut(usize, Vec<f32>) -> Result<(), String>,
{
    for_each_vector_chunk_with_reserve(path, n, cap, query_vector_bytes(), f)
}

fn for_each_vector_chunk_with_reserve<F>(
    path: &Path,
    n: usize,
    cap: usize,
    reserve_bytes: usize,
    mut f: F,
) -> Result<(), String>
where
    F: FnMut(usize, Vec<f32>) -> Result<(), String>,
{
    let per_chunk = raw_chunk_vectors_with_reserve(cap, reserve_bytes);
    let mut reader = BufReader::new(
        File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
    );
    let mut base = 0usize;
    while base < n {
        let take = (n - base).min(per_chunk);
        let byte_offset = base
            .checked_mul(DIM)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| format!("range starts too far into {}", path.display()))?;
        let byte_len = take
            .checked_mul(DIM)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| format!("range too large: {} vectors", take))?;
        f(
            base,
            load_vector_range_from_reader(
                &mut reader,
                byte_offset,
                byte_len,
                take * DIM,
                path,
            )?,
        )?;
        base += take;
    }
    Ok(())
}

fn fp32_row(timing: QueryMajorTiming, raw_bytes: u64, vector_ram_cap: usize) -> Row {
    let per_query = timing.average_ms();
    Row {
        index: "FlatIndex".to_string(),
        bits: "32".to_string(),
        self_r1: "100.00%".to_string(),
        self_r10: "100.00%".to_string(),
        random_r10: "100.00%".to_string(),
        index_ms: "0.0".to_string(),
        prepare_ms: "0.0".to_string(),
        write_ms: "0.0".to_string(),
        self_search_ms: format!("{:.1}", timing.total_ms / 2.0),
        random_search_ms: format!("{:.1}", timing.total_ms / 2.0),
        ms_per_query: format!("{:.3}", per_query),
        first_query_ms: format!("{:.3}", timing.first_ms),
        warm_ms_per_query: format!("{:.3}", timing.warm_ms_per_query),
        index_rom: human_bytes(raw_bytes),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label_query_major(vector_ram_cap, 32),
        vector_ram: vector_ram_label(vector_ram_cap),
    }
}

fn fp32_row_batched(fp32_ms: f64, raw_bytes: u64, vector_ram_cap: usize) -> Row {
    let per_query = fp32_ms / (SELF_QUERIES + RANDOM_QUERIES) as f64;
    Row {
        index: "FlatIndex".to_string(),
        bits: "32".to_string(),
        self_r1: "100.00%".to_string(),
        self_r10: "100.00%".to_string(),
        random_r10: "100.00%".to_string(),
        index_ms: "0.0".to_string(),
        prepare_ms: "0.0".to_string(),
        write_ms: "0.0".to_string(),
        self_search_ms: format!("{:.1}", fp32_ms / 2.0),
        random_search_ms: format!("{:.1}", fp32_ms / 2.0),
        ms_per_query: format!("{:.3}", per_query),
        first_query_ms: "n/a".to_string(),
        warm_ms_per_query: "n/a".to_string(),
        index_rom: human_bytes(raw_bytes),
        data_store: "disk-backed".to_string(),
        vector_staging: vector_staging_label(vector_ram_cap, 32),
        vector_ram: vector_ram_label(vector_ram_cap),
    }
}

fn recall_at_1_arrays(indices: &[[usize; K]], truth: &[[usize; K]]) -> f64 {
    let hits = indices
        .iter()
        .zip(truth)
        .filter(|(got, expected)| got[0] == expected[0])
        .count();
    hits as f64 / truth.len() as f64
}

fn recall_at_10_arrays(indices: &[[usize; K]], truth: &[[usize; K]]) -> f64 {
    let mut hits = 0usize;
    for (got, expected) in indices.iter().zip(truth) {
        for (i, &id) in got.iter().enumerate() {
            if expected.contains(&id) && !got[..i].contains(&id) {
                hits += 1;
            }
        }
    }
    hits as f64 / (truth.len() * K) as f64
}

fn index_path(output_dir: &Path, dataset_id: &str, bit_width: usize) -> PathBuf {
    output_dir.join(format!("{}_turbovec_{}bit.tv", dataset_id, bit_width))
}

fn blocked_index_path(output_dir: &Path, dataset_id: &str, bit_width: usize) -> PathBuf {
    output_dir.join(format!("{}_turbovec_{}bit.tvpb", dataset_id, bit_width))
}

fn vector_ram_label(cap: usize) -> String {
    format!("{} search cap", human_bytes(cap as u64))
}

fn vector_staging_label(cap: usize, bit_width: usize) -> String {
    if bit_width == 32 {
        format!(
            "{} raw f32",
            human_bytes((raw_chunk_vectors(cap) * DIM * std::mem::size_of::<f32>()) as u64)
        )
    } else if bit_width == 16 {
        format!(
            "{} decoded f32",
            human_bytes((raw_chunk_vectors(cap) * DIM * std::mem::size_of::<f32>()) as u64)
        )
    } else {
        format!(
            "{} {}-bit range",
            human_bytes(quant_resident_bytes(quant_chunk_vectors(cap, bit_width), bit_width) as u64),
            bit_width
        )
    }
}

fn vector_staging_label_query_major(cap: usize, bit_width: usize) -> String {
    if bit_width == 32 {
        format!(
            "{} raw f32",
            human_bytes((raw_chunk_vectors_query_major(cap) * DIM * std::mem::size_of::<f32>()) as u64)
        )
    } else if bit_width == 16 {
        format!(
            "{} raw f16 + fused dot",
            human_bytes((fp16_chunk_vectors_query_major(cap) * DIM * std::mem::size_of::<u16>()) as u64)
        )
    } else {
        format!(
            "{} {}-bit blocked range",
            human_bytes(
                quant_query_major_resident_bytes(
                    quant_chunk_vectors_query_major(cap, bit_width),
                    bit_width,
                ) as u64
            ),
            bit_width
        )
    }
}

fn pct(v: f64) -> String {
    format!("{:.2}%", v * 100.0)
}

fn human_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= MB {
        format!("{:.1} MB", bytes as f64 / MB)
    } else if bytes as f64 >= KB {
        format!("{:.1} KB", bytes as f64 / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn human_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
