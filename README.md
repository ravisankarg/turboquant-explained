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
7. Report separate 50K and 100K KPI tables with self R@1/R@10, random R@10, index time, cache-load time, search latency, ROM, and total steady-state vector RAM. The pure vector budget is 30 MiB for 50K and 50 MiB for 100K, with app/runtime/dependency overhead excluded; methods consume persisted data in bounded chunks.

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

Latest tested artifact: `releases/android/vecdb-release-v0.3.apk`, versionName `0.3`, versionCode `4`, SHA-256 `49e2e8b46b67442c894bbcb08322ae1cc9e5abe76c015acaaf961c1563d3ffbb`. The same file was installed on the attached S25 Ultra and copied to `/sdcard/test_apks/vecdb-release-v0.3.apk`.

## S25 Ultra Results

Measured dataset: the current APK downloads one 100K Cohere 768-dimensional FP32 slice and benchmarks separate 50K and 100K tables from it. HNSW is disabled. Each table uses 1000 self queries, 1000 deterministic normalized random-mixture queries, and exact FP32 top-10 ground truth.

| Method | Dataset | Vectors | Bits | Self R@1 | Self R@10 | Random R@10 | Index ms | Prep/load ms | Write ms | Self ms | Random ms | ms/query | ROM | Vector RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FlatIndex | Cohere 50K | 50K | 32 | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 4750.4 | 4750.4 | 4.750 | 146.5 MB | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 16 | 100.00% | 99.89% | 99.65% | 94.2 | 0.0 | 24.2 | 6152.2 | 8177.2 | 7.165 | 73.2 MB | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 8 | 100.00% | 98.97% | 98.61% | 2756.1 | 2935.1 | 22.1 | 17139.7 | 17679.8 | 17.410 | 36.8 MB | 30.0 MiB cap |
| FlatIndex | Cohere 50K | 50K | 4 | 100.00% | 91.51% | 88.21% | 3193.2 | 1489.4 | 14.1 | 878.0 | 855.2 | 0.867 | 18.5 MB | 30.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 32 | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 40749.4 | 40749.4 | 40.749 | 293.0 MB | 50.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 16 | 100.00% | 99.74% | 99.73% | 328.1 | 0.0 | 81.8 | 22634.9 | 10169.0 | 16.402 | 146.5 MB | 50.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 8 | 100.00% | 99.19% | 98.65% | 4424.8 | 2450.6 | 58.8 | 21897.5 | 18842.3 | 20.370 | 73.6 MB | 50.0 MiB cap |
| FlatIndex | Cohere 100K | 100K | 4 | 100.00% | 92.06% | 88.02% | 2422.4 | 781.7 | 17.7 | 496.9 | 469.6 | 0.483 | 37.0 MB | 50.0 MiB cap |

### Legend and memory-cap definitions

- `Self R@1`: the percentage of self queries whose original vector is ranked first. Random R@1 is intentionally omitted.
- `Self R@10`: the standard mean top-10 overlap against exact FP32 top-10 for self queries. A 9/10 overlap is 90%, not 100%.
- `Random R@10`: the same mean `|approximate top-10 ∩ exact FP32 top-10| / 10`, averaged over deterministic normalized random-mixture queries.
- `Index ms`: method-specific index construction time. It is zero for the raw FP32 reference, covers FP16 conversion for the persisted FP16 file, and covers add/quantize/in-memory storage for TurboQuant.
- `Prep/load ms`: persisted index/range loading plus one-time search-layout preparation. It is not included in warm `ms/query`.
- `Write ms`: time to write the persisted FP16 or `.tv` index file.
- `Self ms` and `Random ms`: total search time for 1000 queries at `k=10`.
- `ms/query`: `(Self ms + Random ms) / 2000`; this is the warm search-latency KPI, in milliseconds rather than microseconds.
- `ROM`: persisted index file size on disk, not RAM usage.
- `Vector RAM`: the total accounted steady-state pure-vector budget: query vectors plus one active raw/decoded chunk or one compressed TurboQuant range, blocked search layout, and search caches.
- The cap is 30 MiB for the 50K table and 50 MiB for the 100K table. It is not a cap on total Android process RSS. App/UI objects, Rust/runtime/dependency libraries, allocator overhead, OS file cache, and persisted ROM are outside this KPI. Temporary full-index build/preparation peaks are also excluded from this steady-state search budget.
- `Raw FP32 chunk vectors` means the number of uncompressed 768-d FP32 vectors staged from disk at one time: 8,218 for 50K and 15,045 for 100K. Each vector is 768 x 4 bytes, so the raw chunk itself is about 24.1 MiB or 44.1 MiB; query and working buffers use the rest of the cap. It is a chunk size, not the total vector count.
- FP16 reads its persisted file in bounded decoded chunks. TurboQuant reads bounded `.tv` ranges, searches each range, merges top-10 results, and releases the range before loading the next one.

### Why 8-bit is slower than FP16 here

The lower ROM size of 8-bit does not automatically mean lower latency. On the arm64 S25 Ultra path, FP16 decodes a bounded chunk and runs a simple dense dot-product scan. TurboQuant 8-bit must rotate and calibrate each query, then its current ARM scorer performs scalar byte-code/centroid lookups for every dimension; ARM has no efficient FP32 gather-by-byte instruction for this operation. The same query transform is repeated for each bounded range and query batch. By contrast, 4-bit uses the optimized NEON nibble-LUT kernel, which is why 4-bit is much faster than the current 8-bit path despite having lower precision.

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
