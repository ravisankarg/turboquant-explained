use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use turbovec::TurboQuantIndex;

const DIM: usize = 768;
const N: usize = 50_000;
const K: usize = 10;
const SELF_QUERIES: usize = 1_000;
const RANDOM_QUERIES: usize = 1_000;

#[derive(Clone, Copy)]
struct Hit {
    score: f32,
    idx: usize,
}

#[derive(Serialize)]
struct Row {
    index: String,
    bits: String,
    self_r1: String,
    self_r10: String,
    random_r1: String,
    random_r10: String,
    index_ms: String,
    prepare_ms: String,
    write_ms: String,
    self_search_ms: String,
    random_search_ms: String,
    us_per_query: String,
    index_rom: String,
    ram_delta: String,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    dim: usize,
    base_vectors: usize,
    self_queries: usize,
    random_queries: usize,
    fp32_exact_ms: String,
    notes: Vec<String>,
    table: Vec<Row>,
}

#[no_mangle]
pub extern "system" fn Java_com_turboquant_benchmark_NativeBench_runBenchmark(
    mut env: JNIEnv,
    _class: JClass,
    vector_path: JString,
    output_dir: JString,
) -> jstring {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let vector_path: String = env
            .get_string(&vector_path)
            .map_err(|e| e.to_string())?
            .into();
        let output_dir: String = env
            .get_string(&output_dir)
            .map_err(|e| e.to_string())?
            .into();
        run(Path::new(&vector_path), Path::new(&output_dir))
    }));

    let text = match result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => format!("Benchmark error: {e}"),
        Err(_) => "Benchmark panic in native Rust code".to_string(),
    };
    env.new_string(text).expect("new Java string").into_raw()
}

fn run(vector_path: &Path, output_dir: &Path) -> Result<String, String> {
    let load_start = Instant::now();
    let vectors = load_vectors(vector_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let self_queries = vectors[..SELF_QUERIES * DIM].to_vec();
    let random_queries = make_random_queries(&vectors, RANDOM_QUERIES);

    let fp32_start = Instant::now();
    let self_truth = exact_topk(&vectors, &self_queries);
    let random_truth = exact_topk(&vectors, &random_queries);
    let fp32_ms = fp32_start.elapsed().as_secs_f64() * 1000.0;

    let mut rows = Vec::new();
    rows.push(fp32_row(fp32_ms, vector_path.metadata().map(|m| m.len()).unwrap_or(0)));
    for bit_width in [8usize, 4, 3, 2] {
        rows.push(bench_quant(
            bit_width,
            &vectors,
            &self_queries,
            &random_queries,
            &self_truth,
            &random_truth,
            output_dir,
        )?);
    }

    let report = Report {
        dataset: format!("Cohere train first 50K raw f32, loaded in {:.1} ms", load_ms),
        dim: DIM,
        base_vectors: N,
        self_queries: SELF_QUERIES,
        random_queries: RANDOM_QUERIES,
        fp32_exact_ms: format!("{:.1}", fp32_ms),
        notes: vec![
            "Recall is measured against exact FP32 top-10 over the same 50K vectors.".to_string(),
            "Random queries are deterministic normalized random vectors; self queries are the first 1000 base vectors.".to_string(),
            "The cloned turbovec crate was extended here to support 8-bit indexes in addition to 2, 3, and 4 bit.".to_string(),
            "On arm64-v8a, turbovec's aarch64 NEON path is used for 2/3/4-bit search; 8-bit uses an exact block-major byte-code scorer with NEON-built query LUTs.".to_string(),
        ],
        table: rows,
    };
    serde_json::to_string(&report).map_err(|e| format!("serialize report: {e}"))
}

fn load_vectors(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let expected = N * DIM * 4;
    if bytes.len() != expected {
        return Err(format!(
            "expected {} bytes for {}x{} f32 vectors, got {}",
            expected,
            N,
            DIM,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(N * DIM);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn make_random_queries(vectors: &[f32], nq: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0x5451_2026);
    let mut out = vec![0.0f32; nq * DIM];
    for q in 0..nq {
        let a = rng.gen_range(0..N);
        let b = rng.gen_range(0..N);
        let alpha: f32 = rng.gen_range(0.15..0.85);
        let row = &mut out[q * DIM..(q + 1) * DIM];
        for d in 0..DIM {
            row[d] = alpha * vectors[a * DIM + d] + (1.0 - alpha) * vectors[b * DIM + d];
        }
        normalize(row);
    }
    out
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

fn exact_topk(vectors: &[f32], queries: &[f32]) -> Vec<[usize; K]> {
    queries
        .par_chunks_exact(DIM)
        .map(|q| {
            let mut heap = [Hit {
                score: f32::NEG_INFINITY,
                idx: usize::MAX,
            }; K];
            for i in 0..N {
                let score = dot(q, &vectors[i * DIM..(i + 1) * DIM]);
                insert_hit(&mut heap, Hit { score, idx: i });
            }
            heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            let mut ids = [0usize; K];
            for i in 0..K {
                ids[i] = heap[i].idx;
            }
            ids
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..DIM {
        sum += a[i] * b[i];
    }
    sum
}

fn insert_hit(heap: &mut [Hit; K], hit: Hit) {
    let mut min_pos = 0;
    let mut min_score = heap[0].score;
    for i in 1..K {
        if heap[i].score < min_score {
            min_score = heap[i].score;
            min_pos = i;
        }
    }
    if hit.score > min_score {
        heap[min_pos] = hit;
    }
}

fn bench_quant(
    bit_width: usize,
    vectors: &[f32],
    self_queries: &[f32],
    random_queries: &[f32],
    self_truth: &[[usize; K]],
    random_truth: &[[usize; K]],
    output_dir: &Path,
) -> Result<Row, String> {
    let rss_before = rss_kb();
    let index_start = Instant::now();
    let mut index = TurboQuantIndex::new(DIM, bit_width).map_err(|e| format!("{e:?}"))?;
    index.add(vectors);
    let index_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    let prepare_start = Instant::now();
    index.prepare();
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

    let path = index_path(output_dir, bit_width);
    let write_start = Instant::now();
    index
        .write(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let rom = path.metadata().map(|m| m.len()).unwrap_or(0);

    let self_start = Instant::now();
    let self_results = index.search(self_queries, K);
    let self_ms = self_start.elapsed().as_secs_f64() * 1000.0;

    let random_start = Instant::now();
    let random_results = index.search(random_queries, K);
    let random_ms = random_start.elapsed().as_secs_f64() * 1000.0;
    let rss_after = rss_kb();

    let (self_r1, self_r10) = recall(&self_results.indices, self_truth);
    let (random_r1, random_r10) = recall(&random_results.indices, random_truth);
    let total_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;
    let us_per_query = ((self_ms + random_ms) * 1000.0) / total_q;

    Ok(Row {
        index: "turbovec".to_string(),
        bits: bit_width.to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r1: pct(random_r1),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", self_ms),
        random_search_ms: format!("{:.1}", random_ms),
        us_per_query: format!("{:.1}", us_per_query),
        index_rom: human_bytes(rom),
        ram_delta: human_kb(rss_after.saturating_sub(rss_before)),
    })
}

fn fp32_row(fp32_ms: f64, raw_bytes: u64) -> Row {
    let per_query = (fp32_ms * 1000.0) / ((SELF_QUERIES + RANDOM_QUERIES) as f64);
    Row {
        index: "exact fp32".to_string(),
        bits: "32".to_string(),
        self_r1: "100.00%".to_string(),
        self_r10: "100.00%".to_string(),
        random_r1: "100.00%".to_string(),
        random_r10: "100.00%".to_string(),
        index_ms: "0.0".to_string(),
        prepare_ms: "0.0".to_string(),
        write_ms: "0.0".to_string(),
        self_search_ms: format!("{:.1}", fp32_ms / 2.0),
        random_search_ms: format!("{:.1}", fp32_ms / 2.0),
        us_per_query: format!("{:.1}", per_query),
        index_rom: human_bytes(raw_bytes),
        ram_delta: human_bytes(raw_bytes),
    }
}

fn recall(indices: &[i64], truth: &[[usize; K]]) -> (f64, f64) {
    let mut r1 = 0usize;
    let mut r10 = 0usize;
    for (q, expected) in truth.iter().enumerate() {
        let got = &indices[q * K..(q + 1) * K];
        if got[0] >= 0 && got[0] as usize == expected[0] {
            r1 += 1;
        }
        if got.iter().any(|&x| x >= 0 && expected.contains(&(x as usize))) {
            r10 += 1;
        }
    }
    let n = truth.len() as f64;
    (r1 as f64 / n, r10 as f64 / n)
}

fn index_path(output_dir: &Path, bit_width: usize) -> PathBuf {
    output_dir.join(format!("cohere_50k_turbovec_{}bit.tv", bit_width))
}

fn rss_kb() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn pct(v: f64) -> String {
    format!("{:.2}%", v * 100.0)
}

fn human_kb(kb: u64) -> String {
    human_bytes(kb * 1024)
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
