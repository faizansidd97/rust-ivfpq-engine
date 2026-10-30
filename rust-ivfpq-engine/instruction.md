# In-Memory Vector Search Acceleration with IVF-PQ

You are provided with a baseline vector search engine in `/workspace` implemented in Rust. The current implementation performs an unindexed brute-force linear scan over 128-dimensional floating point vectors (`f32`). Under target production queries, this flat scan fails latency and throughput limits.

Your objective is to implement an **Inverted File with Product Quantization (IVF-PQ)** index in `src/lib.rs` and optimize query execution.

## Requirements

1. **Dimensionality & Subspaces:**
   - Vector dimension: $D = 128$.
   - Subspace decomposition: $M = 16$ orthogonal chunks ($D^* = 8$ dimensions per chunk).
   - Centroids per codebook: $K = 256$ (each chunk is represented as a single byte `u8` code; 16 bytes per vector).
2. **Coarse Quantizer (Inverted Lists):**
   - Cluster vectors across $N_{\text{clusters}} = 64$ coarse centroids.
   - For each cluster, maintain an inverted list storing the vector ID (`u64`) and its 16-byte product-quantized residual code (`[u8; 16]`).
3. **Core API Implementation (`src/lib.rs`):**
   - `VectorIndex::new() -> Self`
   - `VectorIndex::train(&mut self, data: &[f32])`: Train the 64 coarse centroids using k-means, compute residuals for training vectors, and train the $M=16$ codebooks.
   - `VectorIndex::add(&mut self, id: u64, vector: &[f32])`: Assign the vector to its nearest coarse cluster, compute its residual, quantize each subvector against the codebook, and append `(id, [u8; 16])` to that cluster's inverted list.
   - `VectorIndex::search(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<SearchResult>`:
     - Select the closest `nprobe` coarse centroids to the query vector.
     - For each probed cluster, construct an Asymmetric Distance Computation (ADC) lookup table ($M \times 256$) comparing the query residual to the subspace centroids.
     - Scan the inverted list using lookup table additions, and return the top-$k$ nearest candidates sorted in ascending order of squared L2 distance.

## Performance Criteria
The hidden evaluation harness tests your engine with 30,000 indexed vectors and 250 test queries:
- **Recall@10:** Must achieve $\ge 0.82$ against exact Euclidean nearest neighbors.
- **Latency:** Mean search latency must be $< 2.0\text{ ms}$ on 4 vCPUs.
- **Stability:** The binary must build without warnings as errors and run without panics.
