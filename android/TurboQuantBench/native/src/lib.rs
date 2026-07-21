use half::f16;
use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use turbovec::TurboQuantIndex;

const DIM: usize = 768;
const K: usize = 10;
const SELF_QUERIES: usize = 1_000;
const RANDOM_QUERIES: usize = 1_000;
const VECTOR_RAM_50K_BYTES: usize = 30 * 1024 * 1024;
const VECTOR_RAM_100K_BYTES: usize = 50 * 1024 * 1024;
const READ_BUFFER_BYTES: usize = 64 * 1024;
const QUERY_BATCH: usize = 64;
const SIMD_BLOCK: usize = 32;

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
    index_rom: String,
    vector_ram: String,
}

#[derive(Serialize)]
struct DatasetTable {
    dataset: String,
    vectors: String,
    vector_ram_cap: String,
    raw_chunk_vectors: String,
    quant_chunk_vectors: String,
    rows: Vec<Row>,
}

#[derive(Serialize)]
struct Report {
    datasets: String,
    dim: usize,
    self_queries: usize,
    random_queries: usize,
    vector_ram_caps: String,
    raw_chunk_vectors: String,
    methods: String,
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
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let datasets_json: String = env
            .get_string(&datasets_json)
            .map_err(|e| e.to_string())?
            .into();
        let output_dir: String = env
            .get_string(&output_dir)
            .map_err(|e| e.to_string())?
            .into();
        run(&datasets_json, Path::new(&output_dir))
    }));

    let text = match result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => format!("Benchmark error: {e}"),
        Err(_) => "Benchmark panic in native Rust code".to_string(),
    };
    env.new_string(text).expect("new Java string").into_raw()
}

fn run(datasets_json: &str, output_dir: &Path) -> Result<String, String> {
    let datasets: Vec<DatasetInput> =
        serde_json::from_str(datasets_json).map_err(|e| format!("parse datasets: {e}"))?;
    if datasets.is_empty() {
        return Err("no downloaded datasets were supplied".to_string());
    }

    let mut tables = Vec::new();
    let mut labels = Vec::new();
    for dataset in datasets {
        let rows = bench_dataset(&dataset, output_dir)?;
        let cap = vector_ram_cap_bytes(dataset.vectors);
        labels.push(format!("{} ({})", dataset.label, human_count(dataset.vectors)));
        tables.push(DatasetTable {
            dataset: dataset.label,
            vectors: human_count(dataset.vectors),
            vector_ram_cap: format!("{} cap", human_bytes(cap as u64)),
            raw_chunk_vectors: raw_chunk_vectors(cap).to_string(),
            quant_chunk_vectors: format!(
                "8-bit {}, 4-bit {}",
                quant_chunk_vectors(cap, 8),
                quant_chunk_vectors(cap, 4)
            ),
            rows,
        });
    }

    let report = Report {
        datasets: labels.join(", "),
        dim: DIM,
        self_queries: SELF_QUERIES,
        random_queries: RANDOM_QUERIES,
        vector_ram_caps: format!(
            "50K: {} cap; 100K: {} cap",
            human_bytes(VECTOR_RAM_50K_BYTES as u64),
            human_bytes(VECTOR_RAM_100K_BYTES as u64)
        ),
        raw_chunk_vectors: format!(
            "50K: {}; 100K: {}",
            raw_chunk_vectors(VECTOR_RAM_50K_BYTES),
            raw_chunk_vectors(VECTOR_RAM_100K_BYTES)
        ),
        methods: "FlatIndex FP32, FP16, 8-bit, and 4-bit".to_string(),
        notes: vec![
            "Recall is measured against exact FP32 top-10 over the same dataset size.".to_string(),
            "Random R@1 is intentionally omitted; R@10 is the mean fraction of exact FP32 top-10 neighbors recovered by the approximate top-10.".to_string(),
            "Random queries are deterministic normalized blends of two base vectors; self queries are the first 1000 base vectors.".to_string(),
            "Steady-state pure vector RAM is capped at 30 MiB for 50K and 50 MiB for 100K; app/UI memory, Rust/runtime/dependency code, allocator overhead, and OS file cache are excluded.".to_string(),
            "FP32/FP16 keep the query vectors plus one bounded raw/decoded chunk; TurboQuant keeps one persisted compressed range plus its blocked SIMD copy and search caches.".to_string(),
            "All four methods are flat scans over every vector; HNSW is disabled for this run.".to_string(),
            "FlatIndex FP32 scans the raw f32 database, FlatIndex FP16 scans a persisted IEEE FP16 copy, and FlatIndex 8-bit/4-bit use TurboQuant compressed flat scans.".to_string(),
            "On arm64-v8a, TurboQuant's 8-bit path uses the block-major byte-code scorer; 4-bit uses the NEON lookup-table path.".to_string(),
            format!(
                "TurboQuant reads persisted .tv ranges in bounded chunks (8-bit: up to {}, 4-bit: up to {} vectors), searches each range, merges top-10 results, and releases it before the next range.",
                quant_chunk_vectors(VECTOR_RAM_100K_BYTES, 8),
                quant_chunk_vectors(VECTOR_RAM_100K_BYTES, 4)
            ),
            "Prep/load ms includes loading and preparing the bounded TurboQuant ranges; the RAM cap is a steady-state search budget, so temporary full-index build/preparation peaks are not part of this KPI.".to_string(),
        ],
        tables,
    };
    serde_json::to_string(&report).map_err(|e| format!("serialize report: {e}"))
}

fn bench_dataset(dataset: &DatasetInput, output_dir: &Path) -> Result<Vec<Row>, String> {
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

    let fp32_start = Instant::now();
    let self_truth = exact_topk_file(vector_path, dataset.vectors, &self_queries, vector_ram_cap)?;
    let random_truth = exact_topk_file(vector_path, dataset.vectors, &random_queries, vector_ram_cap)?;
    let fp32_ms = fp32_start.elapsed().as_secs_f64() * 1000.0;

    let mut rows = Vec::new();
    rows.push(fp32_row(
        fp32_ms,
        (dataset.vectors as u64) * (DIM as u64) * 4,
        vector_ram_cap,
    ));
    rows.push(bench_fp16(
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
        rows.push(bench_quant(
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
        VECTOR_RAM_50K_BYTES
    } else {
        VECTOR_RAM_100K_BYTES
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
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f32_values(&mut file, byte_len, n * DIM, path)
}

fn read_f32_values(
    file: &mut File,
    byte_len: usize,
    expected_values: usize,
    path: &Path,
) -> Result<Vec<f32>, String> {
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

fn load_fp16_range(
    path: &Path,
    start_vector: usize,
    n: usize,
    cap: usize,
    reserve_bytes: usize,
) -> Result<Vec<f32>, String> {
    let byte_offset = start_vector
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| format!("range starts too far into {}", path.display()))?;
    let value_count = n
        .checked_mul(DIM)
        .ok_or_else(|| format!("range too large: {} vectors", n))?;
    let decoded_bytes = value_count
        .checked_mul(4)
        .ok_or_else(|| format!("decoded range too large: {} vectors", n))?;
    let byte_len = value_count
        .checked_mul(2)
        .ok_or_else(|| format!("range too large: {} vectors", n))?;
    if decoded_bytes > cap.saturating_sub(reserve_bytes.saturating_add(READ_BUFFER_BYTES)) {
        return Err(format!(
            "decoded range of {} bytes exceeds the {} vector RAM cap after {} bytes of resident vector data",
            decoded_bytes,
            human_bytes(cap as u64),
            reserve_bytes
        ));
    }
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(byte_offset as u64))
        .map_err(|e| format!("seek {}: {e}", path.display()))?;
    read_f16_values(&mut file, byte_len, value_count, path)
}

fn read_f16_values(
    file: &mut File,
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

fn exact_topk_file(
    path: &Path,
    n: usize,
    queries: &[f32],
    cap: usize,
) -> Result<Vec<[usize; K]>, String> {
    let nq = queries.len() / DIM;
    let mut heaps = vec![
        [Hit {
            score: f32::NEG_INFINITY,
            idx: usize::MAX,
        }; K];
        nq
    ];
    for_each_vector_chunk(path, n, cap, |base, chunk| {
        heaps.par_iter_mut().enumerate().for_each(|(qi, heap)| {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            for (local_idx, v) in chunk.chunks_exact(DIM).enumerate() {
                let score = dot(q, v);
                insert_hit(
                    heap,
                    Hit {
                        score,
                        idx: base + local_idx,
                    },
                );
            }
        });
        Ok(())
    })?;
    Ok(heaps
        .into_iter()
        .map(|mut heap| {
            heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            let mut ids = [0usize; K];
            for i in 0..K {
                ids[i] = heap[i].idx;
            }
            ids
        })
        .collect())
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

fn bench_fp16(
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

    let self_start = Instant::now();
    let self_results = flat_fp16_topk_file(&path, dataset.vectors, self_queries, vector_ram_cap)?;
    let self_ms = self_start.elapsed().as_secs_f64() * 1000.0;

    let random_start = Instant::now();
    let random_results = flat_fp16_topk_file(&path, dataset.vectors, random_queries, vector_ram_cap)?;
    let random_ms = random_start.elapsed().as_secs_f64() * 1000.0;

    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let total_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;
    let ms_per_query = (self_ms + random_ms) / total_q;

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: "16".to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: "0.0".to_string(),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", self_ms),
        random_search_ms: format!("{:.1}", random_ms),
        ms_per_query: format!("{:.3}", ms_per_query),
        index_rom: human_bytes(rom),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn flat_fp16_topk_file(
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
    let mut base = 0usize;
    while base < n {
        let take = (n - base).min(per_chunk);
        let chunk = load_fp16_range(
            path,
            base,
            take,
            vector_ram_cap,
            query_vector_bytes(),
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
    Ok(heaps
        .into_iter()
        .map(|mut heap| {
            heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            let mut ids = [0usize; K];
            for i in 0..K {
                ids[i] = heap[i].idx;
            }
            ids
        })
        .collect())
}

fn bench_quant(
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
    let (self_results, random_results, prepare_ms, self_ms, random_ms) =
        quant_topk_two_ranges(
            &path,
            dataset.vectors,
            bit_width,
            self_queries,
            random_queries,
            vector_ram_cap,
        )?;

    let self_r1 = recall_at_1_arrays(&self_results, self_truth);
    let self_r10 = recall_at_10_arrays(&self_results, self_truth);
    let random_r10 = recall_at_10_arrays(&random_results, random_truth);
    let total_q = (SELF_QUERIES + RANDOM_QUERIES) as f64;
    let ms_per_query = (self_ms + random_ms) / total_q;

    Ok(Row {
        index: "FlatIndex".to_string(),
        bits: bit_width.to_string(),
        self_r1: pct(self_r1),
        self_r10: pct(self_r10),
        random_r10: pct(random_r10),
        index_ms: format!("{:.1}", index_ms),
        prepare_ms: format!("{:.1}", prepare_ms),
        write_ms: format!("{:.1}", write_ms),
        self_search_ms: format!("{:.1}", self_ms),
        random_search_ms: format!("{:.1}", random_ms),
        ms_per_query: format!("{:.3}", ms_per_query),
        index_rom: human_bytes(rom),
        vector_ram: vector_ram_label(vector_ram_cap),
    })
}

fn quant_topk_two_ranges(
    path: &Path,
    n: usize,
    bit_width: usize,
    self_queries: &[f32],
    random_queries: &[f32],
    vector_ram_cap: usize,
) -> Result<(Vec<[usize; K]>, Vec<[usize; K]>, f64, f64, f64), String> {
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
    let mut self_ms = 0.0f64;
    let mut random_ms = 0.0f64;
    while base < n {
        let take = (n - base).min(per_chunk);
        let prepare_start = Instant::now();
        let index = TurboQuantIndex::load_range(path, base, take)
            .map_err(|e| format!("load range {} [{}..{}]: {e}", path.display(), base, base + take))?;
        index.prepare();
        prepare_ms += prepare_start.elapsed().as_secs_f64() * 1000.0;

        let self_start = Instant::now();
        search_quant_query_batches(&index, self_queries, base, &mut self_heaps);
        self_ms += self_start.elapsed().as_secs_f64() * 1000.0;

        let random_start = Instant::now();
        search_quant_query_batches(&index, random_queries, base, &mut random_heaps);
        random_ms += random_start.elapsed().as_secs_f64() * 1000.0;

        base += take;
    }

    Ok((
        finalize_topk_heaps(self_heaps),
        finalize_topk_heaps(random_heaps),
        prepare_ms,
        self_ms,
        random_ms,
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

fn finalize_topk_heaps(heaps: Vec<[Hit; K]>) -> Vec<[usize; K]> {
    heaps
        .into_iter()
        .map(|mut heap| {
            heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            let mut ids = [0usize; K];
            for i in 0..K {
                ids[i] = heap[i].idx;
            }
            ids
        })
        .collect()
}

fn for_each_vector_chunk<F>(path: &Path, n: usize, cap: usize, mut f: F) -> Result<(), String>
where
    F: FnMut(usize, Vec<f32>) -> Result<(), String>,
{
    let per_chunk = raw_chunk_vectors(cap);
    let mut base = 0usize;
    while base < n {
        let take = (n - base).min(per_chunk);
        f(
            base,
            load_vector_range(path, base, take, cap, query_vector_bytes())?,
        )?;
        base += take;
    }
    Ok(())
}

fn fp32_row(fp32_ms: f64, raw_bytes: u64, vector_ram_cap: usize) -> Row {
    let per_query = fp32_ms / ((SELF_QUERIES + RANDOM_QUERIES) as f64);
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
        index_rom: human_bytes(raw_bytes),
        vector_ram: vector_ram_label(vector_ram_cap),
    }
}

fn recall_at_1(indices: &[i64], truth: &[[usize; K]]) -> f64 {
    let mut hits = 0usize;
    for (q, expected) in truth.iter().enumerate() {
        let got = &indices[q * K..(q + 1) * K];
        if got[0] >= 0 && got[0] as usize == expected[0] {
            hits += 1;
        }
    }
    hits as f64 / truth.len() as f64
}

fn recall_at_10(indices: &[i64], truth: &[[usize; K]]) -> f64 {
    let mut hits = 0usize;
    for (q, expected) in truth.iter().enumerate() {
        let got = &indices[q * K..(q + 1) * K];
        for (i, &x) in got.iter().enumerate() {
            if x >= 0 {
                let id = x as usize;
                if expected.contains(&id) && !got[..i].contains(&x) {
                    hits += 1;
                }
            }
        }
    }
    hits as f64 / (truth.len() * K) as f64
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

fn vector_ram_label(cap: usize) -> String {
    format!("{} cap", human_bytes(cap as u64))
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
