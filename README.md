# TurboQuant Explained

Static GitHub Pages explainer plus an Android benchmark app for TurboQuant vector search.

Live site after Pages deploy:

https://ravisankarg.github.io/turboquant-explained/

## What Is In This Repo

- `index.html`, `styles.css`, `script.js`: the GitHub Pages explainer with visualizations, benchmark tables, and Android app documentation.
- `turbovec/`: cloned and extended Rust TurboVec/TurboQuant implementation.
- `android/TurboQuantBench/`: Android app that calls the Rust search/index code through JNI.
- `releases/android/TurboQuantBench-release.apk`: release APK tested on Samsung Galaxy S25 Ultra.
- `docs/ANDROID_APP.md`: app usage, build, and implementation notes.
- `docs/BENCHMARK_RESULTS.md`: S25 Ultra benchmark results and KPI definitions.

## Android Benchmark Flow

The app does this on-device:

1. Download first 50,000 Cohere 768-d vectors from Hugging Face raw `.f32` data.
2. Build an exact FP32 baseline for recall ground truth.
3. Build TurboQuant indexes at 8, 4, 3, and 2 bits.
4. Run 1000 self queries and 1000 deterministic random mixture queries.
5. Report R@1, R@10, index time, prepare time, search latency, ROM, and RAM.

Install the tested APK:

```bash
adb install -r releases/android/TurboQuantBench-release.apk
```

Build locally:

```bash
cd android/TurboQuantBench
cp local.properties.example local.properties
# edit sdk.dir if needed
JAVA_HOME=/home/ravi/AG/Android_SDK/jdk \
PATH=/home/ravi/AG/Android_SDK/jdk/bin:/home/ravi/AG/Android_SDK/gradle/bin:$HOME/.cargo/bin:$PATH \
gradle assembleRelease
```

If `app/release.keystore` is absent, the Gradle build falls back to debug signing for local reproducibility.

## S25 Ultra Results

Dataset: first 50K vectors from `YoKONCy/Cohere-1M-wikipedia-768d`, 768 dimensions.

| Method | Bits | Self R@1 | Self R@10 | Random R@1 | Random R@10 | Index ms | Prep ms | Write ms | Self ms | Random ms | us/query | ROM | RAM delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact fp32 | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 4155.7 | 4155.7 | 4155.7 | 146.5 MB | 146.5 MB |
| turbovec | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 1293.7 | 259.4 | 17.2 | 4886.6 | 4890.9 | 4888.8 | 36.8 MB | 124.4 MB |
| turbovec | 4 | 100.00% | 100.00% | 99.20% | 100.00% | 994.9 | 147.4 | 8.1 | 143.6 | 149.8 | 146.7 | 18.5 MB | 78.5 MB |
| turbovec | 3 | 100.00% | 100.00% | 99.20% | 100.00% | 977.0 | 74.0 | 6.0 | 148.0 | 153.0 | 150.5 | 13.9 MB | 34.4 MB |
| turbovec | 2 | 99.60% | 100.00% | 89.70% | 99.80% | 927.1 | 79.5 | 4.3 | 83.2 | 80.4 | 81.8 | 9.4 MB | 37.6 MB |

`us/query` is the warm query latency after `prepare()` has already built search caches.

## Rust Changes

The cloned TurboVec code was extended to support 8-bit indexes:

- `turbovec/turbovec/src/lib.rs`: constructors accept `bit_width=8`.
- `turbovec/turbovec/src/io.rs`: persisted `.tv` / `.tvim` load validation accepts 8-bit.
- `turbovec/turbovec/src/encode.rs`: 8-bit quantization uses binary search over Lloyd-Max boundaries.
- `turbovec/turbovec/src/search.rs`: Android ARM path has a block-major 8-bit byte scorer with NEON-built query-centroid LUTs.

Low-bit 2/3/4 search keeps the original optimized NEON nibble-LUT path.

## Why Native Is Used

The fast paths are in Rust/JNI, not Java:

- One JNI call enters native code for the full benchmark.
- Rayon parallelism is used for FP32 baseline and exact truth generation.
- TurboQuant search scans a blocked 32-vector layout prepared once by `prepare()`.
- 2/3/4-bit search uses ARM NEON lookup-table kernels.
- 8-bit search uses exact byte-code centroid scoring with a NEON-precomputed query LUT.

Java handles UI, download progress, and table rendering only.
