# Android Benchmark App

Path: `android/TurboQuantBench`

Release artifact: `releases/android/vecdb-release-v0.3.apk` (versionName `0.3`, versionCode `4`). `releases/android/TurboQuantBench-release.apk` is kept as a compatibility copy.

## How To Use

1. Install the APK:

   ```bash
   adb install -r releases/android/vecdb-release-v0.3.apk
   ```

2. Open `VectorDB 3.0`.
3. Tap `Download Cohere 100K vectors` to download the single raw slice used by the benchmarks.
4. Wait for download progress to finish.
5. Tap `Benchmark available datasets`.
6. The app benchmarks 50K and 100K as separate tests from the same 100K file; it does not add 50K on top of 100K.
7. Read the native UI tables:
   - `Summary`
   - `KPI Tables` with separate 50K and 100K sections
   - `Benchmark legend`

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
- `native/src/lib.rs`: benchmark driver, 30 MiB bounded steady-state vector budget for both 50K/100K, FP32 exact baseline, persisted FP16 disk scan, TurboQuant 8/4-bit flat scans, and JSON report output.
- `turbovec/turbovec/src/search.rs`: low-bit NEON path and ARM 8-bit byte scorer, whose main byte-indexed lookup loop is scalar.
- `turbovec/turbovec/src/encode.rs`: quantization and scale correction.

## Methods In The App

### FlatIndex FP32 exact

Flat brute-force exact scan over all vectors in the selected dataset. It reads the raw file in bounded chunks, computes dot products over 768 floats, and keeps top-10. This is ground truth for self R@1/self R@10 and random R@10.

For the current report, `R@10` is the standard mean overlap: `|approximate top-10 ∩ exact top-10| / 10`, averaged over queries. A query with only one matching neighbor is therefore 10% R@10, not 100%.

### FlatIndex FP16

The app converts the raw vectors to a persisted IEEE FP16 file, then scans that file in bounded chunks. Each chunk is decoded into FP32 staging memory, consumed by the top-k scan, and released before the next chunk is read. Query vectors and the active decoded chunk are charged to the 30 MiB steady-state vector budget for both tables. The FP16 disk representation is 16-bit; `Vector staging` says `decoded f32` because the active chunk is expanded to f32 for the current dot-product loop.

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

## What `prepare()` Does

`Prep/load ms` is the persisted-index loading and cache-preparation work measured during the query-major probes. It builds or warms:

- rotation matrix cache;
- centroid/codebook cache;
- blocked SIMD/search layout from compact bit-plane codes.

The report's `Search RAM` is a total accounted steady-state pure-vector budget: 30 MiB for both 50K and 100K. It includes query vectors plus one active raw/decoded chunk or one persisted TurboQuant range, blocked SIMD copy, and search caches. App/UI memory, Rust/runtime/dependency code, allocator overhead, OS file cache, and persisted ROM are excluded. Temporary full-index build/preparation peaks are also excluded from this steady-state KPI. All methods, including FP32, use disk-backed bounded chunks or ranges; TurboQuant reads bounded `.tv` ranges, searches each range, merges top-10 results, and releases it before the next range. HNSW is disabled for this experiment.

`Raw FP32 chunk vectors` is the number of uncompressed 768-d FP32 vectors staged from disk at once: 8,218 for both 50K and 100K. The raw chunk alone is approximately 24.1 MiB; query and working buffers use the remainder of the cap. It is not the total dataset size or total Android process RAM. FP16 uses the same decoded f32 staging size even though its persisted file is 16-bit.

`ms/query` is the one-query-at-a-time latency KPI. It averages 16 independent probes (8 self + 8 random); every probe scans every bounded chunk or range before the next probe starts, so unrelated requests do not share an active chunk. FP32 and FP16 include bounded file reads and decoding; TurboQuant includes loading and preparing each compressed range. The 1000 + 1000 query workload is still used for recall, where chunk reuse is a throughput optimization that does not change the answer.

For a separately preloaded index, a cold first query can add initial preparation:

```text
first-query latency = prep ms + search latency
```

For the current one-query disk-backed benchmark:

```text
latency = ms/query
```
