# S25 Ultra benchmark results

These are the current v0.5 measurements pulled directly from the APK's cached JSON reports on a Samsung Galaxy S25 Ultra class device (`SM_S938B`, arm64-v8a). One Cohere 100K, 768-dimensional FP32 slice supplies separate 50K and 100K tables. The datasets are not added together.

Each report uses 1,000 self queries, 1,000 deterministic normalized random-mixture queries, top-k 10, and exact FP32 top-10 ground truth. All payload stores are disk-backed. The pure-vector search cap is 30.0 MB for 50K and 50.0 MB for 100K; it is not an Android process-RSS cap.

The tables omit index/build, prep, and write columns. Index construction is one-time and excluded from `ms/query`. FlatIndex build times are listed once per timing path. HNSW used the persisted graph/payload cache, so graph construction is excluded from timed search.

## FlatIndex A — real traffic / query-major

One timed query scans every bounded chunk or compressed range, completes its top-10, then the next query starts. Threads: 1 timed query worker; 8 Rayon workers for recall/truth and preparation. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 80.4, 8-bit 1,922.6, 4-bit 943.4 ms; 100K — FP32 0.0, FP16 175.2, 8-bit 2,361.3, 4-bit 2,486.6 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 35.061 | 146.5 MB | 29.0 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.89% | 99.64% | 136.253 | 73.2 MB | 28.9 MB raw FP16 + fused dot | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.99% | 100.00% | 46.551 | 36.8 MB | 28.9 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 99.99% | 100.00% | 19.153 | 18.5 MB | 28.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 73.689 | 293.0 MB | 49.0 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 161.029 | 146.5 MB | 48.9 MB raw FP16 + fused dot | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 100.00% | 100.00% | 119.751 | 73.6 MB | 48.9 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 100.00% | 100.00% | 54.799 | 37.0 MB | 48.9 MB blocked 4-bit range | 50.0 MB |

## FlatIndex B — batched throughput

Each bounded chunk/range is loaded once and searched against all 2,000 queries. Threads: 8 Rayon workers in the timed range-major batch. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 90.8, 8-bit 1,253.9, 4-bit 1,031.2 ms; 100K — FP32 0.0, FP16 213.5, 8-bit 2,802.8, 4-bit 3,653.1 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 1.318 | 146.5 MB | 23.1 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.89% | 99.64% | 1.324 | 73.2 MB | 23.1 MB decoded FP32 chunk | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.14% | 98.65% | 4.705 | 36.8 MB | 22.7 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 99.99% | 100.00% | 0.651 | 18.5 MB | 21.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 3.343 | 293.0 MB | 43.1 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 3.461 | 146.5 MB | 43.1 MB decoded FP32 chunk | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.16% | 98.63% | 12.820 | 73.6 MB | 42.7 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 100.00% | 100.00% | 1.369 | 37.0 MB | 41.9 MB blocked 4-bit range | 50.0 MB |

## HNSW A — real traffic / query-major

One query traverses the resident graph, reads candidate vectors, finishes top-10, then the next starts. Threads: 1 timed query worker; 8 Rayon workers for recall/truth and preparation. Parameters: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. The compact graph is resident; payload vectors remain disk-backed. Graph construction was a persistent-cache hit and is excluded.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 70.392 | 227.4 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 67.577 | 80.9 MB | 6.0 MB FP16 navigation cache | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 99.75% | 98.99% | 292.741 | 154.6 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 99.75% | 98.99% | 265.853 | 117.9 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 80.161 | 454.8 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 73.792 | 161.9 MB | 6.0 MB FP16 navigation cache | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 99.50% | 98.42% | 285.574 | 309.1 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 99.50% | 98.42% | 303.364 | 235.9 MB | 6.0 MB cache + compressed block + FP32 scratch | ≤50.0 MB |

## HNSW B — batched throughput

The resident graph is shared while 2,000 queries search in parallel. Threads: 8 Rayon workers in the timed batch; each worker has a bounded 256-vector FP16 candidate cache. Parameters are the same as HNSW A. Graph construction is one-time and excluded.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 16.848 | 227.4 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 6.948 | 80.9 MB | 384 KB FP16 candidates/worker | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 99.75% | 98.99% | 72.618 | 154.6 MB | 384 KB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 99.75% | 98.99% | 64.260 | 117.9 MB | 384 KB cache + compressed block + FP32 scratch | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 20.054 | 454.8 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 18.967 | 161.9 MB | 384 KB FP16 candidates/worker | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 99.50% | 98.42% | 83.506 | 309.1 MB | 384 KB cache + compressed block + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 99.50% | 98.42% | 75.411 | 235.9 MB | 384 KB cache + compressed block + FP32 scratch | ≤50.0 MB |

## Definitions

- `Self R@1`: percentage of self queries whose original vector is ranked first. Random R@1 is omitted.
- `Self R@10` and `Random R@10`: mean top-10 overlap with exact FP32 top-10. A 9/10 overlap is 90%, not 100%.
- `ms/query`: milliseconds per query. Query-major timing is the production-style independent-request number. Batched timing divides one reused 2,000-query batch by 2,000 and is throughput only.
- `Vector staging (RAM only)`: active vector payload memory, not overall app RAM. UI, Rust/runtime/dependency libraries, allocator overhead, OS page cache, persisted ROM, and temporary full-index build peaks are excluded.
- `Search RAM cap`: pure-vector steady-state cap, 30 MB for 50K and 50 MB for 100K. Full raw FP32 and compressed stores remain on disk. HNSW graph adjacency is reported separately in `Graph/search RAM`.
- FlatIndex query-major raw FP32 staging is 9,897 vectors for 50K and 16,724 for 100K. Batched staging is 7,898 and 14,725. These are bounded windows, not resident databases.
- FlatIndex A/B 4-bit exact-reranks up to 256 compressed candidates from raw FP32 disk reads. HNSW 8/4-bit rows also exact-rerank bounded graph candidates.

## Optimization analysis

FlatIndex A uses direct bounded file reads, persistent blocked ranges, direct top-k heaps, and low-bit candidate reranking. FlatIndex B changes only the scheduling model: each range is loaded once and searched against the full batch with eight Rayon workers. That is why B is much faster per query but is not representative of isolated traffic.

HNSW uses the `M=16`, `efConstruction=128`, `efSearch=1024` graph configuration. The high `efSearch` is the recall/latency tradeoff that keeps every random R@10 above 98%. Candidate vector blocks are read with bounded direct file spans; A uses a larger shared FP16 cache and B uses a small per-worker cache. The graph is resident, but payload vectors are not.

On ARM, FP32/four-query dot products and the 4-bit nibble-LUT scorer use NEON. The 8-bit path is smaller on disk but pays range preparation, query rotation/calibration, and scalar byte-indexed centroid lookups; FP16 has a simpler dense conversion-and-dot loop. The 4-bit LUT arithmetic is fast, but end-to-end rows still include disk staging and exact reranking. No GPU result is claimed because the APK has no verified Vulkan/NNAPI search kernel and this one-query workload is dominated by disk/candidate I/O.

The lowest current random R@10 is 98.22%, so all 32 method/dataset rows meet the requested 98% floor.

## Pull a report directly from the release APK

```bash
adb exec-out content read \
  --uri 'content://com.turboquant.benchmark.reports/report?mode=query_major'
```

Valid modes are `query_major`, `batched_throughput`, `hnsw_query_major`, and `hnsw_batched_throughput`.
