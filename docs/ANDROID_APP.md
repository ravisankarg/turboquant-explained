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
   - `Benchmark legend` (including cap, staging, graph, and rerank definitions)

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

The app converts the raw vectors to a persisted IEEE FP16 file, then scans that file in bounded chunks. Each chunk is decoded into FP32 staging memory, consumed by the top-k scan, and released before the next chunk is read. Query vectors and the active decoded chunk are charged to the 30 MiB/50 MiB steady-state vector budgets for the 50K/100K tables. The FP16 disk representation is 16-bit; `Vector staging` says `decoded f32` because the active chunk is expanded to f32 for the current dot-product loop.

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

### HNSW payload variants

HNSW uses one compact resident FP16 navigation graph with `M=16`, `efConstruction=128`, `efSearch=1024`, max layers 8, and base degree at most 32. A traverses that graph one query at a time; B searches the graph in parallel with a bounded 256-vector FP16 candidate cache per worker. The graph adjacency is resident, but FP32, FP16, 8-bit, and 4-bit payload stores remain disk-backed. The low-bit HNSW rows exact-rerank their bounded candidate pool from raw FP32 disk reads to keep random R@10 above 98%.

The larger `efSearch=1024` frontier is intentional: the earlier 512 setting missed the 98% random-recall floor on this dataset. It increases candidate expansion and disk reads, so it is a recall/latency tradeoff rather than a claim of lowest possible HNSW latency.

## What `prepare()` Does

`Prep/load ms` is the persisted-index loading and cache-preparation work measured during the query-major probes. It builds or warms:

- rotation matrix cache;
- centroid/codebook cache;
- blocked SIMD/search layout from compact bit-plane codes.

The report's `Search RAM` is a total accounted steady-state pure-vector budget: 30 MiB for 50K and 50 MiB for 100K. It includes query vectors plus one active raw/decoded chunk or one persisted TurboQuant range, blocked SIMD copy, and search caches. App/UI memory, Rust/runtime/dependency code, allocator overhead, OS file cache, and persisted ROM are excluded. Temporary full-index build/preparation peaks are also excluded from this steady-state KPI. All payloads, including FP32 and HNSW variants, use disk-backed bounded reads; TurboQuant reads bounded `.tv`/`.tvpb` ranges, searches each range, merges top-10 results, and releases the range before the next one. HNSW keeps one compact graph resident while payload vectors remain disk-backed. FlatIndex A low-bit rows exact-rerank up to 256 compressed candidates from raw FP32 disk reads.

`Raw FP32 chunk vectors` is the number of uncompressed 768-d FP32 vectors staged from disk at once: 9,897 for the 50K/30 MiB table and 16,724 for the 100K/50 MiB table under the query-major reserve. It is not the total dataset size or total Android process RAM. FP16 uses the same decoded f32 staging size even though its persisted file is 16-bit.

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
