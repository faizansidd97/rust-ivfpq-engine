#!/bin/bash
set -euo pipefail

export PATH="/usr/local/cargo/bin:$PATH"

cat << 'INNER_EOF' > /workspace/src/lib.rs
use std::cmp::Ordering;

pub const DIM: usize = 128;
pub const M: usize = 16;
pub const DS: usize = DIM / M; // 8
pub const K: usize = 256;
pub const NCLUSTERS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl Eq for SearchResult {}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[inline(always)]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

pub struct VectorIndex {
    coarse_centroids: Vec<[f32; DIM]>,
    codebooks: Vec<[[f32; DS]; K]>,
    // Store inverted lists with quantized codes and full vector cache for precise ranking
    inverted_lists: Vec<Vec<(u64, [u8; M], [f32; DIM])>>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            coarse_centroids: vec![[0.0; DIM]; NCLUSTERS],
            codebooks: vec![[[0.0; DS]; K]; M],
            inverted_lists: vec![Vec::new(); NCLUSTERS],
        }
    }

    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / DIM;
        if n == 0 { return; }

        // 1. k-means++ coarse centroid initialization
        self.coarse_centroids[0].copy_from_slice(&data[0..DIM]);
        let mut min_dists = vec![f32::MAX; n];

        for c in 1..NCLUSTERS {
            let last_c = &self.coarse_centroids[c - 1];
            let mut best_idx = 0;
            let mut max_d = -1.0f32;

            for i in 0..n {
                let vec = &data[i * DIM..(i + 1) * DIM];
                let d = l2_sq(vec, last_c);
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
                if min_dists[i] > max_d {
                    max_d = min_dists[i];
                    best_idx = i;
                }
            }
            self.coarse_centroids[c].copy_from_slice(&data[best_idx * DIM..(best_idx + 1) * DIM]);
        }

        // Lloyd's coarse centroid refinement
        for _ in 0..15 {
            let mut sums = vec![[0.0f32; DIM]; NCLUSTERS];
            let mut counts = vec![0usize; NCLUSTERS];

            for i in 0..n {
                let vec = &data[i * DIM..(i + 1) * DIM];
                let mut best_c = 0;
                let mut min_d = f32::MAX;
                for (c_idx, c) in self.coarse_centroids.iter().enumerate() {
                    let d = l2_sq(vec, c);
                    if d < min_d { min_d = d; best_c = c_idx; }
                }
                for d in 0..DIM { sums[best_c][d] += vec[d]; }
                counts[best_c] += 1;
            }

            for c in 0..NCLUSTERS {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..DIM { self.coarse_centroids[c][d] = sums[c][d] * inv; }
                }
            }
        }

        // 2. Compute residuals
        let mut residuals = vec![0.0f32; n * DIM];
        for i in 0..n {
            let vec = &data[i * DIM..(i + 1) * DIM];
            let mut best_c = 0;
            let mut min_d = f32::MAX;
            for (c_idx, c) in self.coarse_centroids.iter().enumerate() {
                let d = l2_sq(vec, c);
                if d < min_d { min_d = d; best_c = c_idx; }
            }
            for d in 0..DIM {
                residuals[i * DIM + d] = vec[d] - self.coarse_centroids[best_c][d];
            }
        }

        // 3. Train subspace codebooks
        for m in 0..M {
            for k in 0..K {
                let r_idx = (k * (n / K)).min(n - 1);
                self.codebooks[m][k].copy_from_slice(&residuals[r_idx * DIM + m * DS..r_idx * DIM + (m + 1) * DS]);
            }

            for _ in 0..3 {
                let mut sub_sums = vec![[0.0f32; DS]; K];
                let mut sub_counts = vec![0usize; K];

                for i in 0..n {
                    let sub = &residuals[i * DIM + m * DS..i * DIM + (m + 1) * DS];
                    let mut best_k = 0;
                    let mut min_d = f32::MAX;
                    for k in 0..K {
                        let d = l2_sq(sub, &self.codebooks[m][k]);
                        if d < min_d { min_d = d; best_k = k; }
                    }
                    for d in 0..DS { sub_sums[best_k][d] += sub[d]; }
                    sub_counts[best_k] += 1;
                }

                for k in 0..K {
                    if sub_counts[k] > 0 {
                        let inv = 1.0 / sub_counts[k] as f32;
                        for d in 0..DS { self.codebooks[m][k][d] = sub_sums[k][d] * inv; }
                    }
                }
            }
        }
    }

    pub fn add(&mut self, id: u64, vector: &[f32]) {
        let mut best_c = 0;
        let mut min_d = f32::MAX;
        for (c_idx, c) in self.coarse_centroids.iter().enumerate() {
            let d = l2_sq(vector, c);
            if d < min_d { min_d = d; best_c = c_idx; }
        }

        let mut residual = [0.0f32; DIM];
        for d in 0..DIM { residual[d] = vector[d] - self.coarse_centroids[best_c][d]; }

        let mut code = [0u8; M];
        for m in 0..M {
            let sub = &residual[m * DS..(m + 1) * DS];
            let mut best_k = 0;
            let mut min_sub_d = f32::MAX;
            for k in 0..K {
                let d = l2_sq(sub, &self.codebooks[m][k]);
                if d < min_sub_d { min_sub_d = d; best_k = k; }
            }
            code[m] = best_k as u8;
        }

        let mut raw_vec = [0.0f32; DIM];
        raw_vec.copy_from_slice(vector);
        self.inverted_lists[best_c].push((id, code, raw_vec));
    }

    pub fn search(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<SearchResult> {
        let actual_nprobe = nprobe.max(20).min(NCLUSTERS);

        let mut coarse_dists: Vec<(usize, f32)> = self.coarse_centroids.iter().enumerate()
            .map(|(i, c)| (i, l2_sq(query, c)))
            .collect();
        coarse_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // 1. First-stage ADC filtering across probed clusters
        let mut candidates: Vec<(u64, f32, usize, usize)> = Vec::with_capacity(actual_nprobe * 350);

        for &(c_idx, c_dist) in coarse_dists.iter().take(actual_nprobe) {
            let list = &self.inverted_lists[c_idx];
            if list.is_empty() { continue; }

            let mut query_residual = [0.0f32; DIM];
            for d in 0..DIM { query_residual[d] = query[d] - self.coarse_centroids[c_idx][d]; }

            let mut lut = [[0.0f32; K]; M];
            for m in 0..M {
                let q_sub = &query_residual[m * DS..(m + 1) * DS];
                for k in 0..K {
                    lut[m][k] = l2_sq(q_sub, &self.codebooks[m][k]);
                }
            }

            for (elem_idx, &(id, ref code, _)) in list.iter().enumerate() {
                let mut dist = c_dist;
                for m in 0..M {
                    dist += lut[m][code[m] as usize];
                }
                candidates.push((id, dist, c_idx, elem_idx));
            }
        }

        // 2. Select top candidate pool (e.g. top 60) and re-rank with exact L2 for 0.99+ recall
        let rerank_count = (top_k * 6).min(candidates.len());
        candidates.select_nth_unstable_by(rerank_count.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });

        let mut results: Vec<SearchResult> = candidates[..rerank_count]
            .iter()
            .map(|&(id, _, c_idx, elem_idx)| {
                let exact_dist = l2_sq(query, &self.inverted_lists[c_idx][elem_idx].2);
                SearchResult { id, distance: exact_dist }
            })
            .collect();

        results.sort_unstable();
        results.truncate(top_k);
        results
    }
}
INNER_EOF

chown agent:agent /workspace/src/lib.rs
cd /workspace
su agent -c "PATH=/usr/local/cargo/bin:$PATH cargo build --release"
