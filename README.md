# TurboQuant Explained

Static GitHub Pages explainer plus an Android benchmark app for TurboQuant vector search.

Live site after Pages deploy:

https://ravisankarg.github.io/turboquant-explained/

## What Is In This Repo

- `index.html`, `styles.css`, `script.js`: the GitHub Pages explainer with visualizations, benchmark tables, and Android app documentation.
- `turbovec/`: cloned and extended Rust TurboVec/TurboQuant implementation.
- `android/TurboQuantBench/`: Android app that calls the Rust search/index code through JNI.
- `releases/android/vecdb-release-v0.3.apk`: named v0.3 release APK tested on Samsung Galaxy S25 Ultra.
- `releases/android/TurboQuantBench-release.apk`: compatibility copy of the tested release APK.
- `docs/ANDROID_APP.md`: app usage, build, and implementation notes.
- `docs/BENCHMARK_RESULTS.md`: S25 Ultra benchmark results and KPI definitions.

## Android Benchmark Flow

The app does this on-device:

1. Download one 100,000-vector Cohere 768-d raw `.f32` slice from Hugging Face.
2. Downloads continue in a foreground data-sync service if the app is backgrounded or the screen turns off. Interrupted transfers resume from the saved `.part` byte offset.
3. Tap the benchmark button; it runs separate 50K and 100K benchmark tests from that same 100K file, so the counts are not added together.
4. For each benchmark test, build an exact FP32 baseline for recall ground truth.
5. Build four FlatIndex methods: raw FP32, persisted IEEE FP16, TurboQuant 8-bit, and TurboQuant 4-bit. HNSW is disabled for the current experiment.
6. Run 1000 self queries and 1000 deterministic random mixture queries per dataset.
7. Report separate 50K and 100K KPI tables with self R@1/R@10, random R@10, index time, cache-load time, `ms/query`, ROM, data-store mode, vector staging, and search RAM. Both tables use a 30 MiB pure-vector search budget; app/runtime/dependency overhead is excluded and every method consumes persisted data in bounded chunks.

Install the tested APK:

```bash
adb install -r releases/android/vecdb-release-v0.3.apk
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

Latest tested artifact: `releases/android/vecdb-release-v0.3.apk`, versionName `0.3`, versionCode `4`, SHA-256 `d1df34c225409e0af7a438e65f862cdbd03d552bdf618fa3aa328810336adae0`. The same file was installed on the attached S25 Ultra and copied to `/sdcard/test_apks/vecdb-release-v0.3.apk`.

## S25 Ultra Results

Measured dataset: the current APK downloads one 100K Cohere 768-dimensional FP32 slice and benchmarks separate 50K and 100K tables from it. HNSW is disabled. Each table uses 1000 self queries, 1000 deterministic normalized random-mixture queries, and exact FP32 top-10 ground truth.

| Method | Dataset | Vectors | Bits | Self R@1 | Self R@10 | Random R@10 | Index ms | Prep/load ms | Write ms | ms/query | ROM | Data store | Vector staging | Search RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FlatIndex | Cohere 50K | 50K | 32 | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 119.741 | 146.5 MB | disk-backed | 24.1 MB raw f32 | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 16 | 100.00% | 99.89% | 99.65% | 86.1 | 0.0 | 23.4 | 164.716 | 73.2 MB | disk-backed | 24.1 MB decoded f32 | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 8 | 100.00% | 98.97% | 98.61% | 1231.0 | 17906.1 | 14.3 | 1144.697 | 36.8 MB | disk-backed | 23.6 MB 8-bit range | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 4 | 100.00% | 91.51% | 88.21% | 1200.3 | 7972.0 | 12.2 | 499.583 | 18.5 MB | disk-backed | 22.9 MB 4-bit range | 30.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 32 | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 363.430 | 293.0 MB | disk-backed | 24.1 MB raw f32 | 30.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 16 | 100.00% | 99.74% | 99.73% | 347.5 | 0.0 | 91.2 | 539.497 | 146.5 MB | disk-backed | 24.1 MB decoded f32 | 30.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 8 | 100.00% | 99.15% | 98.66% | 4616.2 | 42517.6 | 50.4 | 2723.837 | 73.6 MB | disk-backed | 23.6 MB 8-bit range | 30.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 4 | 100.00% | 92.01% | 87.95% | 2202.1 | 16321.0 | 19.3 | 1023.149 | 37.0 MB | disk-backed | 22.9 MB 4-bit range | 30.0 MiB cap |

### Legend and memory-cap definitions

- `Self R@1`: the percentage of self queries whose original vector is ranked first. Random R@1 is intentionally omitted.
- `Self R@10`: the standard mean top-10 overlap against exact FP32 top-10 for self queries. A 9/10 overlap is 90%, not 100%.
- `Random R@10`: the same mean `|approximate top-10 ∩ exact FP32 top-10| / 10`, averaged over deterministic normalized random-mixture queries.
- `Index ms`: method-specific index construction time. It is zero for the raw FP32 reference, covers FP16 conversion for the persisted FP16 file, and covers add/quantize/in-memory storage for TurboQuant.
- `Prep/load ms`: total persisted range loading and search-layout preparation observed across the query-major run. For TurboQuant this work is also included in end-to-end `ms/query` because ranges are not resident between requests.
- `Write ms`: time to write the persisted FP16 or `.tv` index file.
- `ms/query`: total query-major latency-probe time divided by 16 probes, in milliseconds rather than microseconds. The probes are 8 self + 8 random queries; each probe scans every bounded chunk/range before the next starts, so chunks/ranges are not reused across unrelated requests. TurboQuant range load and preparation are included. Recall still uses 1000 self + 1000 random queries.
- `ROM`: persisted index file size on disk, not RAM usage.
- `Data store`: all four methods are disk-backed; the full 50K/100K stores are not resident in RAM.
- `Vector staging`: method-specific vector payload held for the active disk chunk/range. FP32 uses a 24.1 MiB raw f32 chunk; FP16 uses a 24.1 MiB decoded f32 chunk; the 8-bit and 4-bit active ranges are about 23.6 MiB and 22.9 MiB.
- `Search RAM`: the total accounted steady-state pure-vector budget: query vectors plus one active chunk/range, blocked search layout, and search caches. It is 30 MiB for both tables, not a cap on total Android process RSS. App/UI objects, Rust/runtime/dependency libraries, allocator overhead, OS file cache, persisted ROM, and temporary full-index build/preparation peaks are outside this KPI.
- `Raw FP32 chunk vectors` means 8,218 uncompressed 768-d FP32 vectors staged from disk at one time for both tables. Each vector is 768 x 4 bytes, so the raw chunk is about 24.1 MiB; query and working buffers use the rest of the cap. It is a chunk size, not the total vector count.
- FP16 reads its persisted file in bounded decoded chunks. TurboQuant reads bounded `.tv` ranges, searches each range, merges top-10 results, and releases the range before loading the next one.
- Latency uses query-major probes to model one-query-at-a-time use: FP32 and FP16 load every bounded chunk for each probe, and TurboQuant loads/prepares every compressed range for each probe, before moving to the next probe. The 30 MiB cap applies to each active query working set; the full 1000+1000 query workload remains for recall.

### Why the low-bit rows are not faster end-to-end here

The lower ROM size does not automatically mean lower end-to-end latency. On the arm64 S25 Ultra path, FP16 decodes a bounded chunk and runs a simple dense dot-product scan. TurboQuant 8-bit must load and prepare each compressed range, rotate/calibrate each query, and perform scalar byte-code/centroid lookups for every dimension; ARM has no efficient FP32 gather-by-byte instruction for this operation. The 4-bit scorer itself is optimized with NEON nibble LUTs, but the query-major disk-backed KPI also includes loading and preparing every range for each independent request. In this run, 4-bit is therefore faster than 8-bit but still slower end-to-end than FP32: 499.583 vs 119.741 ms/query at 50K, and 1023.149 vs 363.430 ms/query at 100K. The LUT kernel is fast; range staging/preparation is now the dominant cost.

## Rust Changes

The cloned TurboVec code was extended to support 8-bit indexes:

- `turbovec/turbovec/src/lib.rs`: constructors accept `bit_width=8`.
- `turbovec/turbovec/src/io.rs`: persisted `.tv` / `.tvim` load validation accepts 8-bit.
- `turbovec/turbovec/src/encode.rs`: 8-bit quantization uses binary search over Lloyd-Max boundaries.
- `turbovec/turbovec/src/search.rs`: Android ARM path has a block-major 8-bit byte scorer whose main per-vector lookup loop is scalar, while 4-bit uses the NEON nibble-LUT kernel.

Low-bit 2/3/4 search keeps the original optimized NEON nibble-LUT path; the current Android experiment runs the 4-bit configuration.

## Why Native Is Used

The fast paths are in Rust/JNI, not Java:

- One foreground service enters native code for the full benchmark, so the run survives backgrounding and screen-off periods.
- Rayon parallelism is used for FP32 baseline and exact truth generation.
- HNSW is currently disabled so the comparison isolates flat scans under the fixed staging cap.
- TurboQuant search scans bounded persisted `.tv` ranges using a blocked 32-vector layout prepared for each active range.
- 4-bit search uses ARM NEON lookup-table kernels; 8-bit search uses the byte-code scorer.
- 8-bit search uses exact byte-code centroid scoring, but the ARM byte-indexed lookup loop is scalar; this is the reason its warm latency can exceed FP16.

Java handles UI, download progress, and table rendering only.
