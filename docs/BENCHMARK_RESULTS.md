# Latest S25 Ultra Benchmark Results

Device: Samsung Galaxy S25 Ultra class device reported by ADB as `SM_S938B`.

Measured dataset: one downloaded 100K Cohere 768-dimensional FP32 slice. The APK benchmarks 50K and 100K as separate tables from that file; the counts are not added together. HNSW is disabled for this experiment.

Queries:

- 1000 self queries: first 1000 base vectors.
- 1000 random queries: deterministic normalized random mixtures.
- Top-k: 10.
- Recall baseline: exact FP32 top-10 over the same table dataset size.

## KPI Tables

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

## Legend and Header Definitions

- `Dataset`: downloaded vector slice; the current APK uses Cohere 50K and Cohere 100K tables.
- `Vectors`: number of base vectors in that row.
- `Method`: search implementation.
- `Bits`: bits per vector coordinate.
- `Self R@1`: percent of self queries where the original vector is ranked first.
- `Self R@10`: mean `|approximate top-10 ∩ exact FP32 top-10| / 10` over self queries. A 9/10 overlap is 90%, not 100%.
- `Random R@10`: the same mean top-10 overlap over deterministic normalized random-mixture queries. Random R@1 is intentionally not reported.
- `Index ms`: method-specific construction time: zero for the raw FP32 reference, FP16 conversion time for the persisted FP16 file, and add/quantize/in-memory storage for TurboQuant.
- `Prep/load ms`: persisted-index load plus one-time search-cache preparation; not paid per warm query.
- `Write ms`: persisted `.tv` index write time.
- `Self ms`: total time for 1000 self-query searches.
- `Random ms`: total time for 1000 random-query searches.
- `ms/query`: warm average search latency per query in milliseconds.
- `ROM`: persisted index file size.
- `Vector RAM`: total accounted steady-state pure-vector budget, not overall Android process RSS.

## Current experiment conditions

The current APK uses a total accounted steady-state vector RAM budget of 30 MiB for 50K and 50 MiB for 100K. `Vector RAM` includes query vectors plus one active raw/decoded chunk or one persisted TurboQuant range, its blocked SIMD copy, and search caches. App/UI memory, Rust/runtime/dependency code, allocator overhead, OS file cache, and persisted ROM are excluded. Temporary full-index build/preparation peaks are also outside this steady-state search KPI.

`Raw FP32 chunk vectors` means the number of uncompressed 768-d FP32 vectors staged from disk at once: 8,218 for 50K and 15,045 for 100K. Each vector is 3,072 bytes, so the raw chunk itself is approximately 24.1 MiB or 44.1 MiB; query and working buffers consume the remaining cap. This is a bounded chunk size, not the total number of vectors.

FP16 reads its persisted cache in bounded decoded chunks. TurboQuant reads bounded `.tv` ranges, searches each range, merges top-10 results, and releases the range before loading the next one.

On the arm64 S25 Ultra path, 8-bit is slower than FP16 despite its smaller ROM because FP16 uses a simple dense dot-product scan after decoding, while 8-bit performs query rotation/calibration and scalar byte-code/centroid lookups for every dimension. ARM has no efficient FP32 gather-by-byte instruction for this operation, and the query transform is repeated for each bounded range and query batch. The 4-bit path uses the optimized NEON nibble-LUT kernel, so it is much faster.

## Reading The Results

For steady-state search latency, use `ms/query`.

For cold first query after loading/building an index without calling `prepare()`:

```text
cold latency = prep ms + one search latency
```

For warm production use:

```text
call prepare() once
then use ms/query as search latency
```
