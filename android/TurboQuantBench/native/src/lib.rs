use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use turbovec::TurboQuantIndex;

const DIM: usize = 768;
const K: usize = 10;
const SELF_QUERIES: usize = 1_000;
const RANDOM_QUERIES: usize = 1_000;

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
    dataset: String,
    vectors: String,
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
    datasets: String,
    dim: usize,
    self_queries: usize,
    random_queries: usize,
    notes: Vec<String>,
    table: Vec<Row>,
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

    let mut table = Vec::new();
    let mut labels = Vec::new();
    for dataset in datasets {
        let mut rows = bench_dataset(&dataset, output_dir)?;
        labels.push(format!("{} ({})", dataset.label, human_count(dataset.vectors)));
        table.append(&mut rows);
    }

    let report = Report {
        datasets: labels.join(", "),
        dim: DIM,
        self_queries: SELF_QUERIES,
        random_queries: RANDOM_QUERIES,
        notes: vec![
            "Recall is measured against exact FP32 top-10 over the same dataset size.".to_string(),
            "Random queries are deterministic normalized blends of two base vectors; self queries are the first 1000 base vectors.".to_string(),
            "The cloned turbovec crate was extended here to support 8-bit indexes in addition to 2, 3, and 4 bit.".to_string(),
            "On arm64-v8a, turbovec's aarch64 NEON path is used for 2/3/4-bit search; 8-bit uses an exact block-major byte-code scorer with NEON-built query LUTs.".to_string(),
        ],
        table,
    };
    serde_json::to_string(&report).map_err(|e| format!("serialize report: {e}"))
}

fn bench_dataset(dataset: &DatasetInput, output_dir: &Path) -> Result<Vec<Row>, String> {
    if dataset.vectors < SELF_QUERIES {
        return Err(format!("{} has fewer than {} vectors", dataset.label, SELF_QUERIES));
    }
    let vector_path = Path::new(&dataset.path);
    let load_start = Instant::now();
    let vectors = load_vectors(vector_path, dataset.vectors)?;
    let _load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let self_queries = vectors[..SELF_QUERIES * DIM].to_vec();
    let random_queries = make_random_queries(&vectors, dataset.vectors, RANDOM_QUERIES);

    let fp32_start = Instant::now();
    let self_truth = exact_topk(&vectors, dataset.vectors, &self_queries);
    let random_truth = exact_topk(&vectors, dataset.vectors, &random_queries);
    let fp32_ms = fp32_start.elapsed().as_secs_f64() * 1000.0;

    let mut rows = Vec::new();
    rows.push(fp32_row(
        dataset,
        fp32_ms,
        vector_path.metadata().map(|m| m.len()).unwrap_or(0),
    ));
    for bit_width in [8usize, 4, 3, 2] {
        rows.push(bench_quant(
            dataset,
            bit_width,
            &vectors,
            &self_queries,
            &random_queries,
            &self_truth,
            &random_truth,
            output_dir,
        )?);
    }
    Ok(rows)
}

fn load_vectors(path: &Path, n: usize) -> Result<Vec<f32>, String> {
    let expected = n
        .checked_mul(DIM)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| format!("dataset too large: {} vectors", n))?;
    let meta_len = path
        .metadata()
        .map_err(|e| format!("metadata {}: {e}", path.display()))?
        .len() as usize;
    if meta_len != expected {
        return Err(format!(
            "expected {} bytes for {}x{} f32 vectors, got {}",
            expected,
            n,
            DIM,
            meta_len
        ));
    }
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut out = Vec::with_capacity(n * DIM);
    let mut buf = vec![0u8; 1024 * 1024];
    let mut carry = [0u8; 4];
    let mut carry_len = 0usize;
    loop {
        let read = file
            .read(&mut buf)
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
    }
    if carry_len != 0 || out.len() != n * DIM {
        return Err(format!(
            "decoded {} f32 values from {}, expected {}",
            out.len(),
            path.display(),
            n * DIM
        ));
    }
    Ok(out)
}

fn make_random_queries(vectors: &[f32], n: usize, nq: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0x5451_2026);
    let mut out = vec![0.0f32; nq * DIM];
    for q in 0..nq {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
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

fn exact_topk(vectors: &[f32], n: usize, queries: &[f32]) -> Vec<[usize; K]> {
    queries
        .par_chunks_exact(DIM)
        .map(|q| {
            let mut heap = [Hit {
                score: f32::NEG_INFINITY,
                idx: usize::MAX,
            }; K];
            for i in 0..n {
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
    dataset: &DatasetInput,
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

    let path = index_path(output_dir, &dataset.id, bit_width);
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
        dataset: dataset.label.clone(),
        vectors: human_count(dataset.vectors),
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

fn fp32_row(dataset: &DatasetInput, fp32_ms: f64, raw_bytes: u64) -> Row {
    let per_query = (fp32_ms * 1000.0) / ((SELF_QUERIES + RANDOM_QUERIES) as f64);
    Row {
        dataset: dataset.label.clone(),
        vectors: human_count(dataset.vectors),
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

fn index_path(output_dir: &Path, dataset_id: &str, bit_width: usize) -> PathBuf {
    output_dir.join(format!("{}_turbovec_{}bit.tv", dataset_id, bit_width))
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

fn human_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
