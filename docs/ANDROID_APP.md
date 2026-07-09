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

- `MainActivity.java`: 50K/1M dataset choices, download progress, table rendering.
- `NativeBench.java`: JNI bridge.
- `native/src/lib.rs`: benchmark driver, FP32 exact baseline, TurboQuant index build/search, JSON report output.
- `turbovec/turbovec/src/search.rs`: low-bit NEON path and optimized 8-bit byte path.
- `turbovec/turbovec/src/encode.rs`: quantization and scale correction.

## Methods In The App

### FP32 exact

Flat brute-force exact scan over all vectors in the selected dataset. It computes dot products over 768 floats and keeps top-10. This is ground truth for R@1/R@10.

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
