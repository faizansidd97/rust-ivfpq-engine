#!/bin/bash
set -euo pipefail

cat << 'EOF' > /workspace/src/lib.rs
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

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub struct VectorIndex {
    coarse_centroids: Vec<[f32; DIM]>,
    codebooks: Vec<[[f32; DS]; K]>,
    inverted_lists: Vec<Vec<(u64, [u8; M])>>,
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
        // 1. Initialize coarse centroids
        for i in 0..NCLUSTERS {
            let idx = (i * (n / NCLUSTERS)) % n;
            self.coarse_centroids[i].copy_from_slice(&data[idx * DIM..(idx + 1) * DIM]);
        }

        // Mini-batch k-means iterations
        for _ in 0..8 {
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
                    for d in 0..DIM { self.coarse_centroids[c][d] = sums[c][d] / counts[c] as f32; }
                }
            }
        }

        // 2. Compute residuals and train product quantizer codebooks
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

        for m in 0..M {
            for k in 0..K {
                let r_idx = (k * (n / K)) % n;
                self.codebooks[m][k].copy_from_slice(&residuals[r_idx * DIM + m * DS..r_idx * DIM + (m + 1) * DS]);
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

        self.inverted_lists[best_c].push((id, code));
    }

    pub fn search(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<SearchResult> {
        let mut coarse_dists: Vec<(usize, f32)> = self.coarse_centroids.iter().enumerate()
            .map(|(i, c)| (i, l2_sq(query, c)))
            .collect();
        coarse_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut results: Vec<SearchResult> = Vec::new();

        for &(c_idx, _) in coarse_dists.iter().take(nprobe) {
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

            for &(id, ref code) in list {
                let mut dist = 0.0f32;
                for m in 0..M {
                    dist += lut[m][code[m] as usize];
                }
                results.push(SearchResult { id, distance: dist });
            }
        }

        results.sort_unstable();
        results.truncate(top_k);
        results
    }
}
EOF

chown agent:agent /workspace/src/lib.rs
cd /workspace
su agent -c "cargo build --release"
