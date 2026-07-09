# S25 Ultra Benchmark Results

Device: Samsung Galaxy S25 Ultra class device reported by ADB as `SM_S938B`.

Measured dataset: first 50,000 vectors from `YoKONCy/Cohere-1M-wikipedia-768d`.

The app can also download the first 1,000,000 vectors and will benchmark both 50K and 1M in one combined KPI table when both files are present. The table below is the measured S25 Ultra 50K run.

Vector shape: `50,000 x 768` FP32.

Queries:

- 1000 self queries: first 1000 base vectors.
- 1000 random queries: deterministic normalized random mixtures.
- Top-k: 10.
- Recall baseline: exact FP32 top-10 over the same 50K vectors.

## KPI Table

| Method | Bits | Self R@1 | Self R@10 | Random R@1 | Random R@10 | Index ms | Prep ms | Write ms | Self ms | Random ms | us/query | ROM | RAM delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact fp32 | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 4155.7 | 4155.7 | 4155.7 | 146.5 MB | 146.5 MB |
| turbovec | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 1293.7 | 259.4 | 17.2 | 4886.6 | 4890.9 | 4888.8 | 36.8 MB | 124.4 MB |
| turbovec | 4 | 100.00% | 100.00% | 99.20% | 100.00% | 994.9 | 147.4 | 8.1 | 143.6 | 149.8 | 146.7 | 18.5 MB | 78.5 MB |
| turbovec | 3 | 100.00% | 100.00% | 99.20% | 100.00% | 977.0 | 74.0 | 6.0 | 148.0 | 153.0 | 150.5 | 13.9 MB | 34.4 MB |
| turbovec | 2 | 99.60% | 100.00% | 89.70% | 99.80% | 927.1 | 79.5 | 4.3 | 83.2 | 80.4 | 81.8 | 9.4 MB | 37.6 MB |

## Header Definitions

- `Dataset`: downloaded vector slice, for example Cohere 50K or Cohere 1M.
- `Vectors`: number of base vectors in that row.
- `Method`: search implementation.
- `Bits`: bits per vector coordinate.
- `Self R@1`: percent of self queries where the original vector is ranked first.
- `Self R@10`: percent of self queries where the original vector appears in the top 10.
- `Random R@1`: percent of random mixture queries where approximate top-1 equals exact FP32 top-1.
- `Random R@10`: percent of random mixture queries where approximate top-10 intersects exact FP32 top-10.
- `Index ms`: add, rotate, calibrate, quantize, and store the in-memory index.
- `Prep ms`: one-time search-cache preparation; not paid per warm query.
- `Write ms`: persisted `.tv` index write time.
- `Self ms`: total time for 1000 self-query searches.
- `Random ms`: total time for 1000 random-query searches.
- `us/query`: warm average search latency per query.
- `ROM`: persisted index file size.
- `RAM delta`: process RSS increase while building/preparing that index.

## Reading The Results

For steady-state search latency, use `us/query`.

For cold first query after loading/building an index without calling `prepare()`:

```text
cold latency = prep ms + one search latency
```

For warm production use:

```text
call prepare() once
then use us/query as search latency
```
