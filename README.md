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

## High-level execution flows

The benchmark keeps the representation being measured honest. FP16, 8-bit, and 4-bit searches read and score their own persisted representation; they do not consult the raw FP32 store for a better answer or a final rerank.

- **FlatIndex A — query-major:** open the selected payload store, then for each independent query read every bounded chunk or compressed range, score every vector in that representation, merge the top-10, release the range, and start the next query. This is the production-style latency path.
- **FlatIndex B — batched throughput:** open one bounded chunk or compressed range, score the complete 2,000-query batch against it, update one top-k heap per query, release the range, and continue to the next range. It minimizes repeated I/O and measures amortized throughput rather than isolated request latency.
- **HNSW A — query-major:** traverse the resident compact graph from its entry point with `efSearch=1024`, read candidate payload spans from disk, score the candidates with the selected payload representation, finalize top-10, and only then begin the next query.
- **HNSW B — batched throughput:** share the resident graph across eight Rayon workers, give each worker a bounded candidate cache, and process the 2,000 queries in parallel. The graph is reused across the batch, so this number is not a one-request latency prediction.

The representation-specific inner loops are:

- **FP32:** bounded raw `.f32` reads and exact 768-dimensional dot products. The full raw database is never resident.
- **FP16:** persisted IEEE half bits, bounded reads, and arm64 FP16-to-FP32 conversion fused with NEON dot products. The active half-bit chunk is released before the next chunk is read.
- **8-bit TurboQuant:** rotate and calibrate the query, build byte-code lookup tables, scan the block-major byte payload, and merge same-bit scores. The ARM query-major path uses a coarse high-nibble candidate pass followed by exact 8-bit scoring of that bounded candidate pool; it never reads FP32 vectors.
- **4-bit TurboQuant:** rotate and calibrate the query, build nibble lookup tables, scan 32-vector blocked two-nibble bytes with the ARM NEON LUT kernel, and merge same-bit scores. Four-bit distortion is therefore visible in recall instead of being hidden by an FP32 rerank.

The graph and payload are separate in HNSW: the graph is built over a persisted FP16 navigation store and compact adjacency is resident for traversal, while FP32/FP16/8-bit/4-bit payload stores remain disk-backed. Quantized HNSW rows finalize candidates from their own bit-width only.

Install the tested APK:

```bash
adb install -r releases/android/vecdb-release-v0.5.apk
adb shell mkdir -p /sdcard/test_apks
adb push releases/android/vecdb-release-v0.5.apk /sdcard/test_apks/vecdb-release-v0.5.apk
```

Release artifact: `releases/android/vecdb-release-v0.5.apk`, versionName `0.5`, versionCode `6`, SHA-256 `2a785a84ded4679e3b75dd30848b1b595506683755ac8bf4df3749fd7e52cf79`.

## S25 Ultra results

Device: Samsung Galaxy S25 Ultra class device (`SM_S938B`, arm64-v8a). The numbers below are pulled from the APK's cached JSON report provider, not copied from UI labels. `ms/query` is milliseconds per query, never microseconds. Each dataset has its own pure-vector search cap: 50K = 30.0 MB; 100K = 50.0 MB.

The tables intentionally omit index/build, prep, and write columns. Index construction is a one-time operation and is not part of `ms/query`; the relevant FlatIndex build times are listed once in each table heading. HNSW rows reuse the persisted graph/payload cache in this run, so graph construction is also excluded from the timed search KPI.

### FlatIndex — A) real traffic / query-major

Threads: 1 timed query worker; 8 Rayon workers are used only for recall/truth and preparation. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 78.0, 8-bit 1,237.2, 4-bit 1,154.4 ms; 100K — FP32 0.0, FP16 170.9, 8-bit 2,461.6, 4-bit 3,112.6 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 33.604 | 146.5 MB | 29.0 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.88% | 99.64% | 21.381 | 73.2 MB | 28.9 MB raw FP16 + fused dot | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.10% | 98.54% | 30.123 | 36.8 MB | 28.9 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 93.04% | 90.00% | 16.210 | 18.5 MB | 28.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 68.712 | 293.0 MB | 49.0 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 45.118 | 146.5 MB | 48.9 MB raw FP16 + fused dot | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.31% | 98.77% | 85.440 | 73.6 MB | 48.9 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 93.46% | 90.32% | 70.775 | 37.0 MB | 48.9 MB blocked 4-bit range | 50.0 MB |

### FlatIndex — B) batched throughput

Threads: 8 Rayon workers in the timed range-major batch. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 87.1, 8-bit 1,300.7, 4-bit 1,183.4 ms; 100K — FP32 0.0, FP16 191.8, 8-bit 2,857.8, 4-bit 4,512.1 ms.

The B table is the last device report pulled before the final raw-half FP16 and ARM coarse-to-exact 8-bit B-path source edits were packaged into v0.5. Its recall and memory accounting are valid for the same-bit contract; reconnect the S25 and rerun B to measure those final B-only latency changes.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 0.587 | 146.5 MB | 23.1 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.88% | 99.64% | 1.330 | 73.2 MB | 23.1 MB raw FP16 chunk | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.10% | 98.54% | 4.727 | 36.8 MB | 22.7 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 93.04% | 90.00% | 0.381 | 18.5 MB | 21.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 1.294 | 293.0 MB | 43.1 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 3.363 | 146.5 MB | 43.1 MB raw FP16 chunk | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.31% | 98.77% | 11.400 | 73.6 MB | 42.7 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 93.46% | 90.32% | 1.383 | 37.0 MB | 41.9 MB blocked 4-bit range | 50.0 MB |

### HNSW — A) real traffic / query-major

Threads: 1 timed query worker; 8 Rayon workers are used for recall/truth and preparation. Graph parameters: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. The compact graph is resident; payload vectors remain disk-backed. Graph build is one-time and was a persistent-cache hit in this run. The quantized rows are finalized from their own bit-width payload; there is no raw-FP32 cross-bit rerank.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 41.544 | 227.4 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 39.506 | 80.9 MB | 6.0 MB FP16 navigation cache | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 98.89% | 97.67% | 56.208 | 154.6 MB | 6.0 MB cache + compressed block | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 92.94% | 89.50% | 53.324 | 117.9 MB | 6.0 MB cache + compressed block | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 49.873 | 454.8 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 48.026 | 161.9 MB | 6.0 MB FP16 navigation cache | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 98.85% | 97.39% | 67.199 | 309.1 MB | 6.0 MB cache + compressed block | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 93.21% | 89.51% | 63.597 | 235.9 MB | 6.0 MB cache + compressed block | ≤50.0 MB |

### HNSW — B) batched throughput

Threads: 8 Rayon workers in the timed graph-search batch. The same graph parameters are used: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. Each worker has a bounded 256-vector FP16 candidate cache; graph build is one-time and excluded.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 6.054 | 227.4 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 5.411 | 80.9 MB | 384 KB FP16 candidates/worker | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 98.89% | 97.67% | 8.849 | 154.6 MB | 384 KB cache + compressed block | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 92.94% | 89.50% | 12.448 | 117.9 MB | 384 KB cache + compressed block | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 11.969 | 454.8 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 8.446 | 161.9 MB | 384 KB FP16 candidates/worker | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 98.85% | 97.39% | 16.042 | 309.1 MB | 384 KB cache + compressed block | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 93.21% | 89.51% | 15.200 | 235.9 MB | 384 KB cache + compressed block | ≤50.0 MB |

## Legend and benchmark interpretation

- `Self R@1` is the percentage of self queries whose original vector is ranked first. Random R@1 is intentionally omitted.
- `Self R@10` and `Random R@10` are mean `|approximate top-10 ∩ exact FP32 top-10| / 10`. A 9/10 overlap is 90%, not 100%.
- `ms/query` is milliseconds per query. A query-major row completes one independent query across every bounded chunk/range or graph candidate frontier before the next begins. A batched row divides one 2,000-query batch by 2,000 after reusing each chunk/range or graph worker cache; it is a throughput result, not production single-request latency.
- `Vector staging (RAM only)` is the active vector payload working set, not total Android app RAM. It excludes UI objects, Rust/runtime/dependency libraries, allocator overhead, OS page cache, persisted ROM, and temporary full-index build peaks. The full raw FP32 stores are on disk.
- The FlatIndex query-major raw FP32 windows contain 9,897 vectors at 50K and 16,724 at 100K. The batched windows contain 7,898 and 14,725. FP16 is also disk-backed; the current arm64 path reads half bits and converts during the dot product instead of creating a full resident FP32 copy.
- The pure-vector search cap is 30 MB for 50K and 50 MB for 100K. HNSW reports graph/search RAM separately because compact adjacency is resident; payload vectors are still disk-backed.
- FlatIndex low-bit search may narrow a range to a bounded candidate pool for speed, but final ranking uses only the persisted bit-width being measured. No raw FP32 vector is read by FlatIndex 8-bit or 4-bit search, so the displayed low-bit recall is the actual same-bit result rather than an FP32-rerank result.

## Why the measured optimizations behave this way

- Flat A uses direct bounded file reads, persistent blocked TurboQuant ranges, a direct top-k heap, bounded raw/half-bit chunks, and same-bit candidate finalization. The 4-bit ARM NEON nibble-LUT kernel gives the best Flat A latency in this snapshot, but its random R@10 is 90.00% at 50K and 90.32% at 100K because the contract disallows an FP32 rerank.
- Flat B is range-major: each bounded range is loaded once and searched against all 2,000 queries. Eight Rayon workers make it excellent for batch throughput, but it does not represent independent one-query traffic.
- HNSW uses `M=16`, `efConstruction=128`, and `efSearch=1024` to widen the graph frontier. FP32/FP16 score their own payloads; 8-bit/4-bit score only their own persisted compressed payloads. The high frontier improves graph recall, but same-bit quantization still lowers HNSW 8/4 recall and sparse candidate reads make those rows slower.
- ARM NEON is used for FP32/four-query dot products and the 4-bit nibble-LUT path. No GPU result is claimed: this workload is dominated by one-query disk I/O and candidate reads, and the APK has no verified Vulkan/NNAPI search kernel. Dispatch and transfer overhead would not be a justified optimization without a measured device kernel.
- FlatIndex A FP16 uses AArch64 FP16-to-FP32 conversion plus NEON dot products. It beats FP32 in this snapshot: 21.381 vs 33.604 ms/query at 50K and 45.118 vs 68.712 ms/query at 100K, with the same recall contract. The persisted half store remains disk-backed and the staging cap is unchanged.
- 8-bit can still be slower than FP16 even though its ROM is smaller: the ARM byte-code path pays range preparation, query rotation/calibration, and byte-indexed centroid lookups. FP16 is a dense sequential dot loop that NEON handles efficiently. Four-bit lookup arithmetic is very fast, but recall is more sensitive because four bits preserve less information and no FP32 rerank is allowed.
- HNSW 8/4 latency is not automatically equal to HNSW FP32/FP16: graph traversal is shared, but each payload scorer has different lookup/conversion work, candidate read patterns, and same-bit score quality. In this run HNSW 8/4 are slower and fall below the 98% random-recall target.

The current snapshot does not meet the 98% random R@10 floor for FlatIndex 4-bit or HNSW 8/4. Those rows are published as an honest diagnosis of the pure same-bit contract and remain the next recall/latency tuning targets; FP32/FP16 and FlatIndex 8-bit are at or above 98% in the measured rows.

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
