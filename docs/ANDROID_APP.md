# Android Benchmark App

Path: `android/TurboQuantBench`

Release artifact: `releases/android/TurboQuantBench-release.apk`

## How To Use

1. Install the APK:

   ```bash
   adb install -r releases/android/TurboQuantBench-release.apk
   ```

2. Open `TurboQuant Bench`.
3. Tap either `Download Cohere 50K vectors (146.5 MB)` for the quick dataset or `Download Cohere 1M vectors (2.86 GB)` for the full raw slice.
4. Wait for download progress to finish.
5. Tap `Benchmark available datasets`.
6. The app checks which datasets are present and benchmarks all of them in one run.
7. Read the native UI table:
   - `Summary`
   - `KPI Table`
   - `Latency Notes`

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

- `MainActivity.java`: 50K/1M dataset choices, foreground-service launch, download progress polling, table rendering.
- `DownloadService.java`: background foreground-service downloader with notification progress and resumable `.part` files.
- `BenchmarkService.java`: foreground benchmark runner with a partial wake lock and last-report persistence.
- `NativeBench.java`: JNI bridge.
- `native/src/lib.rs`: benchmark driver, FP32 exact baseline, bundled HNSW baseline, TurboQuant index build/search, JSON report output.
- `turbovec/turbovec/src/search.rs`: low-bit NEON path and optimized 8-bit byte path.
- `turbovec/turbovec/src/encode.rs`: quantization and scale correction.

## Methods In The App

### FP32 exact

Flat brute-force exact scan over all vectors in the selected dataset. It computes dot products over 768 floats and keeps top-10. This is ground truth for R@1/R@10.

### HNSW

Uses the Rust `hnsw_rs` implementation bundled into the app's native library. It builds an HNSW graph with dot distance over normalized vectors and reports the same recall and latency fields as TurboQuant. No phone-side package install is required.

FAISS CPU is not currently bundled in the APK because this checkout does not vendor an Android NDK build of the FAISS C++ library. FAISS GPU is not an Android app target here because FAISS GPU is CUDA/NVIDIA-oriented.

### 8-bit TurboQuant

Uses the extended 8-bit Rust code path:

- Lloyd-Max 256-level codebook.
- TQ+ calibration.
- bit-plane packed storage.
- blocked 32-vector search layout.
- Android ARM block-major byte-code scorer.
- NEON precomputes `query[d] * centroid[code]` lookup rows.

### 4/3/2-bit TurboQuant

Uses the original low-bit TurboVec search design:

- bit-plane packed codes.
- blocked 32-vector layout.
- per-query nibble LUTs.
- ARM NEON table lookups and vector accumulators.

## What `prepare()` Does

`prepare()` is a one-time cache step. It builds or warms:

- rotation matrix cache;
- centroid/codebook cache;
- blocked SIMD/search layout from compact bit-plane codes.

For a cold first query:

```text
first-query latency = prep ms + search latency
```

For production warm queries after `prepare()`:

```text
latency = us/query
```
