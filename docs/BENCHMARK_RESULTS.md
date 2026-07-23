# S25 Ultra benchmark results

These are the v0.5 measurements pulled from the APK's cached JSON reports on a Samsung Galaxy S25 Ultra class device (`SM_S938B`, arm64-v8a). One Cohere 100K, 768-dimensional FP32 slice supplies separate 50K and 100K tables. The datasets are not added together. The published numbers are a device snapshot; the source contract is stricter than older reports and quantized rows do not read raw FP32 for a final rerank.

Each report uses 1,000 self queries, 1,000 deterministic normalized random-mixture queries, top-k 10, and exact FP32 top-10 ground truth. All payload stores are disk-backed. The pure-vector search cap is 30.0 MB for 50K and 50.0 MB for 100K; it is not an Android process-RSS cap.

The tables omit index/build, prep, and write columns. Index construction is one-time and excluded from `ms/query`. FlatIndex build times are listed once per timing path. HNSW used the persisted graph/payload cache, so graph construction is excluded from timed search.

## FlatIndex A — real traffic / query-major

One timed query scans every bounded chunk or compressed range, completes its top-10, then the next query starts. Threads: 1 timed query worker; 8 Rayon workers for recall/truth and preparation. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 78.0, 8-bit 1,237.2, 4-bit 1,154.4 ms; 100K — FP32 0.0, FP16 170.9, 8-bit 2,461.6, 4-bit 3,112.6 ms.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 33.604 | 146.5 MB | 29.0 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.88% | 99.64% | 21.381 | 73.2 MB | 28.9 MB raw FP16 + fused dot | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.10% | 98.54% | 30.123 | 36.8 MB | 28.9 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 93.04% | 90.00% | 16.210 | 18.5 MB | 28.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 68.712 | 293.0 MB | 49.0 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 45.118 | 146.5 MB | 48.9 MB raw FP16 + fused dot | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.31% | 98.77% | 85.440 | 73.6 MB | 48.9 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 93.46% | 90.32% | 70.775 | 37.0 MB | 48.9 MB blocked 4-bit range | 50.0 MB |

## FlatIndex B — batched throughput

Each bounded chunk/range is loaded once and searched against all 2,000 queries. Threads: 8 Rayon workers in the timed range-major batch. One-time FlatIndex build (`index_ms`): 50K — FP32 0.0, FP16 87.1, 8-bit 1,300.7, 4-bit 1,183.4 ms; 100K — FP32 0.0, FP16 191.8, 8-bit 2,857.8, 4-bit 4,512.1 ms.

The B rows are the last device report pulled before the final raw-half FP16 and ARM coarse-to-exact 8-bit B-path source edits were packaged into v0.5. Their recall and memory accounting are still the same-bit contract; reconnect the S25 and rerun B to measure those final B-only latency changes.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Search RAM cap |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 100.00% | 100.00% | 100.00% | 0.587 | 146.5 MB | 23.1 MB raw FP32 chunk | 30.0 MB |
| Cohere 50K | 16 | 100.00% | 99.88% | 99.64% | 1.330 | 73.2 MB | 23.1 MB raw FP16 chunk | 30.0 MB |
| Cohere 50K | 8 | 100.00% | 99.10% | 98.54% | 4.727 | 36.8 MB | 22.7 MB blocked 8-bit range | 30.0 MB |
| Cohere 50K | 4 | 100.00% | 93.04% | 90.00% | 0.381 | 18.5 MB | 21.9 MB blocked 4-bit range | 30.0 MB |
| Cohere 100K | 32 | 100.00% | 100.00% | 100.00% | 1.294 | 293.0 MB | 43.1 MB raw FP32 chunk | 50.0 MB |
| Cohere 100K | 16 | 100.00% | 99.74% | 99.73% | 3.363 | 146.5 MB | 43.1 MB raw FP16 chunk | 50.0 MB |
| Cohere 100K | 8 | 100.00% | 99.31% | 98.77% | 11.400 | 73.6 MB | 42.7 MB blocked 8-bit range | 50.0 MB |
| Cohere 100K | 4 | 100.00% | 93.46% | 90.32% | 1.383 | 37.0 MB | 41.9 MB blocked 4-bit range | 50.0 MB |

## HNSW A — real traffic / query-major

One query traverses the resident graph, reads candidate vectors, finishes top-10, then the next starts. Threads: 1 timed query worker; 8 Rayon workers for recall/truth and preparation. Parameters: `M=16`, `efConstruction=128`, `efSearch=1024`, `max layers=8`, base degree ≤32. The compact graph is resident; payload vectors remain disk-backed. Graph construction was a persistent-cache hit and is excluded. Quantized rows are finalized strictly from their own persisted bit-width payload; no raw-FP32 cross-bit rerank is used.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 41.544 | 227.4 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 39.506 | 80.9 MB | 6.0 MB FP16 navigation cache | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 98.89% | 97.67% | 56.208 | 154.6 MB | 6.0 MB cache + compressed block | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 92.94% | 89.50% | 53.324 | 117.9 MB | 6.0 MB cache + compressed block | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 49.873 | 454.8 MB | 6.0 MB FP16 navigation cache + FP32 candidate scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 48.026 | 161.9 MB | 6.0 MB FP16 navigation cache | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 98.85% | 97.39% | 67.199 | 309.1 MB | 6.0 MB cache + compressed block | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 93.21% | 89.51% | 63.597 | 235.9 MB | 6.0 MB cache + compressed block | ≤50.0 MB |

## HNSW B — batched throughput

The resident graph is shared while 2,000 queries search in parallel. Threads: 8 Rayon workers in the timed batch; each worker has a bounded 256-vector FP16 candidate cache. Parameters are the same as HNSW A. Graph construction is one-time and excluded.

| Dataset | Bits | Self R@1 | Self R@10 | Random R@10 | ms/query | ROM | Vector staging (RAM only) | Graph/search RAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cohere 50K | 32 | 99.90% | 99.75% | 98.99% | 6.054 | 227.4 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤30.0 MB |
| Cohere 50K | 16 | 99.90% | 99.65% | 98.65% | 5.411 | 80.9 MB | 384 KB FP16 candidates/worker | ≤30.0 MB |
| Cohere 50K | 8 | 99.90% | 98.89% | 97.67% | 8.849 | 154.6 MB | 384 KB cache + compressed block | ≤30.0 MB |
| Cohere 50K | 4 | 99.90% | 92.94% | 89.50% | 12.448 | 117.9 MB | 384 KB cache + compressed block | ≤30.0 MB |
| Cohere 100K | 32 | 99.30% | 99.50% | 98.42% | 11.969 | 454.8 MB | 384 KB FP16 candidates/worker + FP32 scratch | ≤50.0 MB |
| Cohere 100K | 16 | 99.30% | 99.25% | 98.22% | 8.446 | 161.9 MB | 384 KB FP16 candidates/worker | ≤50.0 MB |
| Cohere 100K | 8 | 99.30% | 98.85% | 97.39% | 16.042 | 309.1 MB | 384 KB cache + compressed block | ≤50.0 MB |
| Cohere 100K | 4 | 99.30% | 93.21% | 89.51% | 15.200 | 235.9 MB | 384 KB cache + compressed block | ≤50.0 MB |

## Definitions

- `Self R@1`: percentage of self queries whose original vector is ranked first. Random R@1 is omitted.
- `Self R@10` and `Random R@10`: mean top-10 overlap with exact FP32 top-10. A 9/10 overlap is 90%, not 100%.
- `ms/query`: milliseconds per query. Query-major timing is the production-style independent-request number. Batched timing divides one reused 2,000-query batch by 2,000 and is throughput only.
- `Vector staging (RAM only)`: active vector payload memory, not overall app RAM. UI, Rust/runtime/dependency libraries, allocator overhead, OS page cache, persisted ROM, and temporary full-index build peaks are excluded.
- `Search RAM cap`: pure-vector steady-state cap, 30 MB for 50K and 50 MB for 100K. Full raw FP32 and compressed stores remain on disk. HNSW graph adjacency is reported separately in `Graph/search RAM`.
- FlatIndex query-major raw FP32 staging is 9,897 vectors for 50K and 16,724 for 100K. Batched staging is 7,898 and 14,725. These are bounded windows, not resident databases.
- FlatIndex low-bit paths may narrow a bounded compressed range to a candidate pool, but final ranking uses only that persisted bit-width. HNSW 8/4-bit rows likewise finalize from their own compressed payload. No quantized row uses raw FP32 for a final rerank.

## Optimization analysis

FlatIndex A uses direct bounded file reads, persistent blocked ranges, direct top-k heaps, and same-bit candidate finalization. FlatIndex B changes the scheduling model: each range is loaded once and searched against the full batch with eight Rayon workers. That is why B is much faster per query but is not representative of isolated traffic.

HNSW uses the `M=16`, `efConstruction=128`, `efSearch=1024` graph configuration. The high `efSearch` widens the graph frontier and improves navigation recall, but it does not remove same-bit quantization error. Candidate vector blocks are read with bounded direct file spans; A uses a larger shared FP16 cache and B uses a small per-worker cache. The graph is resident, but payload vectors are not.

On ARM, FP32 and FlatIndex A FP16 conversion/dot products use NEON, as does the 4-bit nibble-LUT scorer. The optimized FP16 path beats FP32 in this snapshot: 21.381 vs 33.604 ms/query at 50K and 45.118 vs 68.712 ms/query at 100K. The 8-bit path is smaller on disk but pays range preparation, query rotation/calibration, and byte-indexed centroid lookups; FP16 is a dense sequential loop that NEON handles efficiently. The 4-bit LUT arithmetic is fast, but its lower precision explains the recall loss under the no-cross-bit contract. No GPU result is claimed because the APK has no verified Vulkan/NNAPI search kernel and this one-query workload is dominated by disk/candidate I/O.

The current snapshot does not meet the 98% random R@10 floor for FlatIndex 4-bit or HNSW 8/4-bit. Those rows are intentionally visible as the remaining same-bit recall targets; FP32/FP16 and FlatIndex 8-bit are at or above 98% in the measured rows.

## High-level code flows

### FlatIndex A

1. Load query vectors and exact FP32 ground truth with bounded reads.
2. For one independent query, walk every bounded raw chunk or `.tvpb` range.
3. FP32 computes exact dots; FP16 reads half bits and uses the arm64 conversion/NEON dot loop; 8/4-bit build representation-specific LUTs and score their blocked codes.
4. Merge top-10 across ranges, release the active range, and begin the next query.

### FlatIndex B

1. Keep the same disk-backed stores and representation-specific scorers.
2. Load one bounded chunk/range once, then score all 2,000 queries against it with eight Rayon workers.
3. Update one top-k heap per query, release the range, and continue until every range has been visited.
4. Divide the complete batch time by 2,000. The result is throughput, not independent request latency.

### HNSW A and B

The one-time builder creates a compact graph over the persisted FP16 navigation store with `M=16` and `efConstruction=128`. At search time graph adjacency is resident, but payloads remain on disk. `efSearch=1024` expands a broad candidate frontier. A performs traversal and payload scoring one query at a time; B shares the graph across eight workers and reuses a bounded per-worker candidate cache.

FP32 and FP16 score their own disk-backed payloads. HNSW 8-bit and 4-bit read their own `.tvpb` payloads and finalize from same-bit scores only. The graph can supply good candidate IDs while the low-bit scorer still changes their order; this is why HNSW 8/4 recall can be below the FP32/FP16 rows even with a large `efSearch`.

### Why recall or latency can be lower

- Four-bit codes have fewer quantization levels, so same-bit score ordering loses information. The earlier FP32 candidate rerank was removed because it violates the pure-representation experiment.
- HNSW 8/4 combines graph candidate sparsity with quantization error. Increasing `efSearch` helps, but it also increases candidate reads and scorer work; it cannot restore information absent from a 4-bit payload.
- Eight-bit is smaller on disk but not automatically faster on ARM. The main byte-indexed lookup loop has less favorable gather behavior than a dense FP16 dot loop, and every query still pays rotation/calibration and range preparation.
- Query-major A includes every range read for every query. Batched B reuses each range across 2,000 queries, so its `ms/query` can be dramatically lower without implying that one real request will get that latency.
- The 30/50 MiB limit is the pure vector staging budget. It does not include Android process memory, Rust/runtime/dependency libraries, allocator overhead, OS page cache, ROM, or temporary build peaks. OS page-cache warmth may affect repeated measurements, but it is not counted as resident vector staging.

## Pull a report directly from the release APK

```bash
adb exec-out content read \
  --uri 'content://com.turboquant.benchmark.reports/report?mode=query_major'
```

Valid modes are `query_major`, `batched_throughput`, `hnsw_query_major`, and `hnsw_batched_throughput`.
