# Android Benchmark App

Path: `android/TurboQuantBench`

Release artifact: `releases/android/vecdb-release-v0.5.apk` (versionName `0.5`, versionCode `6`). `releases/android/TurboQuantBench-release.apk` is kept as a compatibility copy.

## How To Use

1. Install the APK:

   ```bash
   adb install -r releases/android/vecdb-release-v0.5.apk
   ```

2. Open `VectorDB 3.0`.
3. Tap `Download Cohere 100K vectors` to download the single raw slice used by the benchmarks.
4. Wait for download progress to finish.
5. Tap `Benchmark available datasets`.
6. The app benchmarks 50K and 100K as separate tests from the same 100K file; it does not add 50K on top of 100K.
7. Read the native UI tables:
   - `Summary`
   - `KPI Tables` with separate 50K and 100K sections; each dataset title lists thread usage and one-time index/build time once, outside `ms/query`
   - `Benchmark legend` (including cap, staging, graph, and same-bit scoring definitions)

Downloads run in a foreground data-sync service, so they continue when the app is backgrounded or the screen is off. Partial downloads are kept as `.part` files in private app storage. If a transfer stops, tap the same dataset button again and the app resumes from the saved byte offset using an HTTP `Range` request.

Benchmarks also run in a foreground service with a partial wake lock, so the native benchmark continues when the app is backgrounded or the screen turns off.

The app stores downloaded files in private app storage, so reinstalling with `adb install -r` keeps them as long as the app is not uninstalled or data-cleared.

## Build

```bash
cd android/TurboQuantBench
cp local.properties.example local.properties
# edit sdk.dir if your Android SDK differs
JAVA_HOME=/home/ravi/AG/Android_SDK/jdk \
PATH=/home/ravi/AG/Android_SDK/jdk/bin:/home/ravi/AG/Android_SDK/gradle/bin:$HOME/.cargo/bin:$PATH \
gradle assembleRelease
```

The app builds `arm64-v8a` only because the benchmark target is modern Android phones such as the S25 Ultra. If `app/release.keystore` exists, it is used for release signing. Otherwise Gradle falls back to debug signing so public checkouts can build.

## App Architecture

- `MainActivity.java`: single 100K dataset download, separate 50K/100K benchmark slice launch, foreground-service launch, download progress polling, table rendering.
- `DownloadService.java`: background foreground-service downloader with notification progress and resumable `.part` files.
- `BenchmarkService.java`: foreground benchmark runner with a partial wake lock and last-report persistence.
- `NativeBench.java`: JNI bridge.
- `native/src/lib.rs`: benchmark driver, 30 MiB/50 MiB bounded steady-state vector budgets for 50K/100K, FP32 exact baseline, persisted FP16 disk scan, TurboQuant 8/4-bit flat scans, compact disk-backed HNSW payload variants, and JSON report output.
- `turbovec/turbovec/src/search.rs`: low-bit NEON path and ARM 8-bit byte scorer, whose main byte-indexed lookup loop is scalar.
- `turbovec/turbovec/src/encode.rs`: quantization and scale correction.

## Methods In The App

### FlatIndex FP32 exact

Flat brute-force exact scan over all vectors in the selected dataset. It reads the raw file in bounded chunks, computes dot products over 768 floats, and keeps top-10. This is ground truth for self R@1/self R@10 and random R@10.

For the current report, `R@10` is the standard mean overlap: `|approximate top-10 ∩ exact top-10| / 10`, averaged over queries. A query with only one matching neighbor is therefore 10% R@10, not 100%.

### FlatIndex FP16

The app converts the raw vectors to a persisted IEEE FP16 file, then scans that file in bounded chunks. On the arm64 S25 path, FlatIndex A reads the half bits and uses native FP16-to-FP32 conversion plus NEON dot products four dimensions at a time; it does not expand the complete chunk to resident FP32. Each chunk is consumed by the top-k scan and released before the next chunk is read. Query vectors and the active half-bit chunk are charged to the 30 MiB/50 MiB steady-state vector budgets for the 50K/100K tables. This removes the old scalar `f16::to_f32()` conversion bottleneck while preserving FP16 recall.

FAISS CPU is not currently bundled in the APK because this checkout does not vendor an Android NDK build of the FAISS C++ library. FAISS GPU is not an Android app target here because FAISS GPU is CUDA/NVIDIA-oriented.

### FlatIndex 8-bit TurboQuant

Uses the extended 8-bit Rust code path:

- Lloyd-Max 256-level codebook.
- TQ+ calibration.
- bit-plane packed storage.
- blocked 32-vector search layout.
- Android ARM block-major byte-code scorer.
- The main ARM per-vector byte-indexed lookup loop is scalar because ARM has no efficient FP32 gather-by-byte instruction for this operation.

This is why 8-bit can be slower than FP16 in the current APK: FP16 decodes a bounded chunk and runs a simple dense dot-product scan, while 8-bit also rotates/calibrates queries and repeats scalar centroid lookups for every dimension. The query transform is repeated for each bounded `.tv` range and one-query latency probe. The smaller 8-bit ROM therefore does not imply lower end-to-end latency.

### 4-bit TurboQuant

Uses the original low-bit TurboVec search design:

- bit-plane packed codes.
- blocked 32-vector layout.
- per-query nibble LUTs.
- ARM NEON table lookups and vector accumulators.

The 4-bit path is intentionally scored only from the persisted 4-bit payload. It does not read the raw FP32 file for a final rerank. That keeps the recall number honest for a pure 4-bit store, but it also exposes the expected quantization loss: the current device snapshot is about 90% random R@10 for FlatIndex A and about 89.5% for HNSW.

### HNSW payload variants

HNSW uses one compact resident FP16 navigation graph with `M=16`, `efConstruction=128`, `efSearch=1024`, max layers 8, and base degree at most 32. A traverses that graph one query at a time; B searches the graph in parallel with a bounded candidate cache per worker. The graph adjacency is resident, but FP32, FP16, 8-bit, and 4-bit payload stores remain disk-backed. FP32/FP16 score their own payloads, and HNSW 8/4 score only their own compressed payloads; there is no raw-FP32 cross-bit rerank.

The larger `efSearch=1024` frontier is intentional: it widens graph navigation and improves candidate recall, but it cannot restore precision lost by an 8-bit or 4-bit payload. It increases candidate expansion and disk reads, so it is a recall/latency tradeoff rather than a claim of lowest possible HNSW latency. The current snapshot is below the 98% random-recall floor for HNSW 8/4, which is the next tuning target.

### A/B scheduling paths

- **FlatIndex A, real traffic:** one query reads and scans every bounded chunk/range, merges top-10, releases it, and then the next query starts. The timing includes the per-query disk/range work.
- **FlatIndex B, batched throughput:** one chunk/range is loaded once and scanned against all 2,000 queries with eight Rayon workers. This amortizes I/O and is not independent request latency.
- **HNSW A, real traffic:** one query traverses the resident graph, reads candidate payload spans, scores them with the selected representation, and completes before the next query.
- **HNSW B, batched throughput:** eight workers share the resident graph and reuse bounded candidate caches while processing the 2,000-query batch. Its `ms/query` is an amortized throughput number.

For all four paths, the 30 MiB (50K) and 50 MiB (100K) cap applies to pure vector staging/search working memory. It does not include total Android RSS, Rust/runtime/dependency libraries, allocator overhead, OS page cache, persisted ROM, or temporary full-index build peaks.

## What `prepare()` Does

`Prep/load ms` is the persisted-index loading and cache-preparation work measured during the query-major probes. It builds or warms:

- rotation matrix cache;
- centroid/codebook cache;
- blocked SIMD/search layout from compact bit-plane codes.

The report's `Search RAM` is a total accounted steady-state pure-vector budget: 30 MiB for 50K and 50 MiB for 100K. It includes query vectors plus one active raw/half-bit chunk or one persisted TurboQuant range, blocked SIMD copy, and search caches. App/UI memory, Rust/runtime/dependency code, allocator overhead, OS file cache, and persisted ROM are excluded. Temporary full-index build/preparation peaks are also excluded from this steady-state KPI. All payloads, including FP32 and HNSW variants, use disk-backed bounded reads; TurboQuant reads bounded `.tv`/`.tvpb` ranges, searches each range, merges top-10 results, and releases the range before the next one. HNSW keeps one compact graph resident while payload vectors remain disk-backed. Quantized rows use same-bit finalization only.

`Raw FP32 chunk vectors` is the number of uncompressed 768-d FP32 vectors staged from disk at once: 9,897 for the 50K/30 MiB table and 16,724 for the 100K/50 MiB table under the query-major reserve. It is not the total dataset size or total Android process RAM. FP16 remains disk-backed and uses raw half-bit staging with conversion during scoring; the persisted file is 16-bit.

`ms/query` is the one-query-at-a-time latency KPI. It averages 16 independent probes (8 self + 8 random); every probe scans every bounded chunk or range before the next probe starts, so unrelated requests do not share an active chunk. FP32 and FP16 include bounded file reads and decoding; TurboQuant includes loading and preparing each compressed range. The 1000 + 1000 query workload is still used for recall, where chunk reuse is a throughput optimization that does not change the answer.

## Direct KPI extraction over adb

Release builds are non-debuggable, so `adb shell run-as com.turboquant.benchmark` is expected to be denied. The cached JSON is exposed through a read-only provider restricted to the adb shell UID:

```bash
adb exec-out content read \
  --uri 'content://com.turboquant.benchmark.reports/report?mode=batched_throughput'
```

Use `query_major`, `batched_throughput`, `hnsw_query_major`, or `hnsw_batched_throughput` for `mode`. This reads the report directly from the app's private cache and avoids UI scrolling and logcat parsing.

For a separately preloaded index, a cold first query can add initial preparation:

```text
first-query latency = prep ms + search latency
```

For the current one-query disk-backed benchmark:

```text
latency = ms/query
```
