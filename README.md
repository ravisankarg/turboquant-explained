# TurboQuant Explained

Static GitHub Pages explainer plus an Android benchmark app for disk-backed vector search.

Live site: <https://ravisankarg.github.io/turboquant-explained/>

## Repository contents

- `index.html`, `styles.css`, `script.js`: the GitHub Pages explainer and current S25 benchmark dashboard.
- `turbovec/`: the Rust TurboVec/TurboQuant implementation and ARM search kernels.
- `android/TurboQuantBench/`: the Android app and JNI benchmark driver.
- `releases/android/vecdb-release-v0.5.apk`: the tested v0.5 release APK.
- `docs/ANDROID_APP.md`: build, install, and direct-report extraction notes.
- `docs/BENCHMARK_RESULTS.md`: the complete benchmark tables and definitions.

## Android benchmark flow

The app downloads one Cohere 100K, 768-dimensional FP32 slice, then benchmarks separate 50K and 100K tables from that file. The counts are not added together. Every payload is disk-backed, including FP32: the app reads bounded chunks or compressed ranges, searches them, merges top-10 results, and releases them before reading the next one.

The app has two timing paths for each index family:

- **A — real traffic / query-major:** one query traverses or scans the complete selected index, finishes its top-10, and only then does the next query start. This is the production-style latency KPI.
- **B — batched throughput:** bounded chunks/ranges or graph workers are reused across the 2,000-query batch. This measures amortized throughput, not one independent request latency.

Each path tests FlatIndex and HNSW with FP32, FP16, 8-bit, and 4-bit payloads. Recall uses 1,000 self queries and 1,000 deterministic random-mixture queries against exact FP32 top-10 ground truth.

Install the tested APK:

```bash
adb install -r releases/android/vecdb-release-v0.5.apk
adb shell mkdir -p /sdcard/test_apks
adb push releases/android/vecdb-release-v0.5.apk /sdcard/test_apks/vecdb-release-v0.5.apk
```

Release artifact: `releases/android/vecdb-release-v0.5.apk`, versionName `0.5`, versionCode `6`, SHA-256 `98d3a42165d4ce7fa596a23a399b3d01319d5dedbc1602ce46d64a46022b201c`.

## S25 Ultra results

Device: Samsung Galaxy S25 Ultra class device (`SM_S938B`, arm64-v8a). The numbers below are pulled directly from the APK's cached JSON report provider, not copied from UI labels. `ms/query` is milliseconds per query, never microseconds. Each dataset has its own pure-vector search cap: 50K = 30.0 MB; 100K = 50.0 MB.

The tables intentionally omit index/build, prep, and write columns. Index construction is a one-time operation and is not part of `ms/query`; the relevant FlatIndex build times are listed once in each table heading. HNSW rows reuse the persisted graph/payload cache in this run, so graph construction is also excluded from the timed search KPI.

### FlatIndex — A) real traffic / query-major

Threads: 1 timed query worker; 8 Rayon workers are used only for recall/truth and preparation. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 80.4, 8-bit 1,922.6, 4-bit 943.4 ms; 100K — FP32 0.0, FP16 175.2, 8-bit 2,361.3, 4-bit 2,486.6 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 35.061 | 146.5 MB | 29.0 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.89% | 99.64% | 136.253 | 73.2 MB | 28.9 MB raw FP16 + fused dot | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.99% | 100.00% | 46.551 | 36.8 MB | 28.9 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 99.99% | 100.00% | 19.153 | 18.5 MB | 28.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 73.689 | 293.0 MB | 49.0 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 161.029 | 146.5 MB | 48.9 MB raw FP16 + fused dot | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 100.00% | 100.00% | 119.751 | 73.6 MB | 48.9 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 100.00% | 100.00% | 54.799 | 37.0 MB | 48.9 MB blocked 4-bit range | 50.0 MB |

### FlatIndex — B) batched throughput

Threads: 8 Rayon workers in the timed range-major batch. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 90.8, 8-bit 1,253.9, 4-bit 1,031.2 ms; 100K — FP32 0.0, FP16 213.5, 8-bit 2,802.8, 4-bit 3,653.1 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 1.318 | 146.5 MB | 23.1 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.89% | 99.64% | 1.324 | 73.2 MB | 23.1 MB decoded FP32 chunk | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.14% | 98.65% | 4.705 | 36.8 MB | 22.7 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 99.99% | 100.00% | 0.651 | 18.5 MB | 21.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 3.343 | 293.0 MB | 43.1 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 3.461 | 146.5 MB | 43.1 MB decoded FP32 chunk | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.16% | 98.63% | 12.820 | 73.6 MB | 42.7 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 100.00% | 100.00% | 1.369 | 37.0 MB | 41.9 MB blocked 4-bit range | 50.0 MB |

### HNSW — A) real traffic / query-major

Threads: 1 timed query worker; 8 Rayon workers are used for recall/truth and preparation. Graph parameters: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. The compact graph is resident; payload vectors remain disk-backed. Graph build is one-time and was a persistent-cache hit in this run.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 70.392 | 227.4 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 67.577 | 80.9 MB | 6.0 MB FP16 navigation cache | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 99.75% | 98.99% | 292.741 | 154.6 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 99.75% | 98.99% | 265.853 | 117.9 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 80.161 | 454.8 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 73.792 | 161.9 MB | 6.0 MB FP16 navigation cache | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 99.50% | 98.42% | 285.574 | 309.1 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 99.50% | 98.42% | 303.364 | 235.9 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤50.0 MB |

### HNSW — B) batched throughput

Threads: 8 Rayon workers in the timed graph-search batch. The same graph parameters are used: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. Each worker has a bounded 256-vector FP16 candidate cache; graph build is one-time and excluded.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 16.848 | 227.4 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 6.948 | 80.9 MB | 384 KB FP16 candidates/worker | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 99.75% | 98.99% | 72.618 | 154.6 MB | 384 KB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 99.75% | 98.99% | 64.260 | 117.9 MB | 384 KB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 20.054 | 454.8 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 18.967 | 161.9 MB | 384 KB FP16 candidates/worker | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 99.50% | 98.42% | 83.506 | 309.1 MB | 384 KB cache + compressed block + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 99.50% | 98.42% | 75.411 | 235.9 MB | 384 KB cache + compressed block + FP32 scratch | ≤50.0 MB |

## Legend and benchmark interpretation

- `Self R@1` is the percentage of self queries whose original vector is ranked first. Random R@1 is intentionally omitted.
- `Self R@10` and `Random R@10` are mean `|approximate top-10 ∩ exact FP32 top-10| / 10`. A 9/10 overlap is 90%, not 100%.
- `ms/query` is milliseconds per query. A query-major row completes one independent query across every bounded chunk/range or graph candidate frontier before the next begins. A batched row divides one 2,000-query batch by 2,000 after reusing each chunk/range or graph worker cache; it is a throughput result, not production single-request latency.
- `Vector staging (RAM only)` is the active vector payload working set, not total Android app RAM. It excludes UI objects, Rust/runtime/dependency libraries, allocator overhead, OS page cache, persisted ROM, and temporary full-index build peaks. The full raw FP32 stores are on disk.
- The FlatIndex query-major raw FP32 windows contain 9,897 vectors at 50K and 16,724 at 100K. The batched windows contain 7,898 and 14,725. FP16 is also disk-backed; its active data is decoded only for the current bounded chunk.
- The pure-vector search cap is 30 MB for 50K and 50 MB for 100K. HNSW reports graph/search RAM separately because compact adjacency is resident; payload vectors are still disk-backed.
- FlatIndex A low-bit search exact-reranks up to 256 compressed candidates from the raw FP32 file. FlatIndex B 4-bit uses the same exact-candidate rerank, which is why its displayed recall is not raw 4-bit recall.

## Why the measured optimizations behave this way

- Flat A uses direct bounded file reads, persistent blocked TurboQuant ranges, a direct top-k heap, bounded raw/decoded chunks, and exact candidate reranking for low-bit rows. The 4-bit ARM NEON nibble-LUT kernel gives the best Flat A latency in this run while keeping random R@10 at 100%.
- Flat B is range-major: each bounded range is loaded once and searched against all 2,000 queries. Eight Rayon workers make it excellent for batch throughput, but it does not represent independent one-query traffic.
- HNSW uses `M=16`, `efConstruction=128`, and `efSearch=1024` to keep random R@10 above 98%. A direct block cache reads candidate FP16 vectors with bounded `read_at` spans; B uses a smaller per-worker cache. Quantized payloads exact-rerank the bounded candidate set from disk.
- ARM NEON is used for FP32/four-query dot products and the 4-bit nibble-LUT path. No GPU result is claimed: this workload is dominated by one-query disk I/O and candidate reads, and the APK has no verified Vulkan/NNAPI search kernel. Dispatch and transfer overhead would not be a justified optimization without a measured device kernel.
- 8-bit can be slower than FP16 even though its ROM is smaller: the ARM byte-code path pays range preparation, query rotation/calibration, and scalar byte-indexed centroid lookups. FP16 has a simpler dense conversion-and-dot loop. Four-bit lookup compute is very fast, but its end-to-end time still includes disk staging and exact reranking.

The global minimum random R@10 in these four current tables is 98.22%, so every published method/dataset row meets the requested 98% floor.

## Direct KPI extraction over adb

Release builds are non-debuggable, so `run-as` is not used. The APK exposes only cached reports through a read-only provider restricted to the adb shell UID:

```bash
adb exec-out content read \
  --uri 'content://com.turboquant.benchmark.reports/report?mode=hnsw_query_major'
```

Valid modes are `query_major`, `batched_throughput`, `hnsw_query_major`, and `hnsw_batched_throughput`.

## Native implementation notes

- Rust/JNI runs the benchmark in a foreground service so it survives backgrounding and screen-off periods.
- Rayon is used for exact truth/recall preparation and the timed batch path.
- FlatIndex uses bounded disk reads; HNSW keeps compact adjacency resident while all payload stores remain disk-backed.
- The 4-bit scorer uses ARM NEON table lookup accumulation; the 8-bit byte-code scorer is intentionally documented as scalar on the main byte-indexed loop.
- The Android release is arm64-v8a only, matching the S25 benchmark target.
