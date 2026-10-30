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

pub struct VectorIndex {
    // Baseline unindexed storage
    vectors: Vec<(u64, Vec<f32>)>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    pub fn train(&mut self, _data: &[f32]) {
        // Baseline no-op
    }

    pub fn add(&mut self, id: u64, vector: &[f32]) {
        self.vectors.push((id, vector.to_vec()));
    }

    pub fn search(&self, query: &[f32], top_k: usize, _nprobe: usize) -> Vec<SearchResult> {
        // Flat scan (fails latency requirements)
        let mut results: Vec<SearchResult> = self.vectors
            .iter()
            .map(|(id, v)| {
                let dist: f32 = v.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                SearchResult { id: *id, distance: dist }
            })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}
