# Latest S25 Ultra Benchmark Results

Device: Samsung Galaxy S25 Ultra class device reported by ADB as `SM_S938B`.

Measured dataset: one downloaded 100K Cohere 768-dimensional FP32 slice. The APK benchmarks 50K and 100K as separate tables from that file; the counts are not added together. HNSW is disabled for this experiment.

Queries:

- 1000 self queries: first 1000 base vectors.
- 1000 random queries: deterministic normalized random mixtures.
- Top-k: 10.
- Recall baseline: exact FP32 top-10 over the same table dataset size.

## KPI Tables

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

## Legend and Header Definitions

- `Dataset`: downloaded vector slice; the current APK uses Cohere 50K and Cohere 100K tables.
- `Vectors`: number of base vectors in that row.
- `Method`: search implementation.
- `Bits`: bits per vector coordinate.
- `Self R@1`: percent of self queries where the original vector is ranked first.
- `Self R@10`: mean `|approximate top-10 ∩ exact FP32 top-10| / 10` over self queries. A 9/10 overlap is 90%, not 100%.
- `Random R@10`: the same mean top-10 overlap over deterministic normalized random-mixture queries. Random R@1 is intentionally not reported.
- `Index ms`: method-specific construction time: zero for the raw FP32 reference, FP16 conversion time for the persisted FP16 file, and add/quantize/in-memory storage for TurboQuant.
- `Prep/load ms`: total persisted range loading and search-layout preparation observed across the query-major run. For TurboQuant this work is also included in end-to-end `ms/query` because ranges are not resident between requests.
- `Write ms`: persisted `.tv` index write time.
- `ms/query`: total query-major latency-probe time divided by 16 probes, in milliseconds. The probes are 8 self + 8 random queries; each probe scans every bounded chunk/range before the next starts, so chunks/ranges are not reused across unrelated requests. TurboQuant range load and preparation are included. Recall still uses 1000 self + 1000 random queries.
- `ROM`: persisted index file size.
- `Data store`: all four methods are disk-backed; the full 50K/100K stores are not resident in RAM.
- `Vector staging`: method-specific vector payload held for the active disk chunk/range. FP32 uses raw f32; FP16 uses decoded f32; the low-bit rows use their compressed active ranges.
- `Search RAM`: total accounted steady-state pure-vector budget, not overall Android process RSS.

## Current experiment conditions

The current APK uses a total accounted steady-state search RAM budget of 30 MiB for both 50K and 100K. `Search RAM` includes query vectors plus one active raw/decoded chunk or one persisted TurboQuant range, its blocked SIMD copy, and search caches. App/UI memory, Rust/runtime/dependency code, allocator overhead, OS file cache, and persisted ROM are excluded. Temporary full-index build/preparation peaks are also outside this steady-state search KPI.

`Raw FP32 chunk vectors` means the number of uncompressed 768-d FP32 vectors staged from disk at once: 8,218 for both 50K and 100K. Each vector is 3,072 bytes, so the raw chunk itself is approximately 24.1 MiB; query and working buffers consume the remaining cap. This is a bounded chunk size, not the total number of vectors. FP16 has the same decoded f32 staging size even though its persisted disk representation is 16-bit.

FP16 reads its persisted cache in bounded decoded chunks. TurboQuant reads bounded `.tv` ranges, searches each range, merges top-10 results, and releases the range before loading the next one.

Latency uses query-major probes to model one-query-at-a-time use: FP32 and FP16 load every bounded chunk for each probe, and TurboQuant loads/prepares every compressed range for each probe, before moving to the next probe. The 30 MiB cap applies to each active query working set; the full 1000+1000 query workload remains for recall.

On the arm64 S25 Ultra path, 8-bit is slower than FP16 despite its smaller ROM because FP16 uses a simple dense dot-product scan after decoding, while 8-bit also pays per-range load/preparation plus query rotation/calibration and scalar byte-code/centroid lookups for every dimension. The 4-bit path uses the optimized NEON nibble-LUT kernel, so its loaded-range compute is efficient, but the query-major disk-backed KPI includes per-request range staging/preparation. It is faster than 8-bit here, but not faster than FP32 end-to-end: 499.583 vs 119.741 ms/query at 50K and 1023.149 vs 363.430 ms/query at 100K.

## Reading The Results

For one-query-at-a-time end-to-end search, use `ms/query`. It averages 16 independent latency probes, includes bounded disk staging, and includes per-range load/preparation for TurboQuant.

For cold first query after loading/building an index without calling `prepare()`:

```text
cold latency = prep ms + one search latency
```

For warm production use:

```text
call prepare() once
then use ms/query as search latency
```
