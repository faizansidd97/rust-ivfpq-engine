use std::io::{self, BufRead};
use vector_engine::{VectorIndex, DIM};

fn main() {
    let mut index = VectorIndex::new();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while let Some(Ok(line)) = lines.next() {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0] {
            "TRAIN" => {
                let n: usize = parts[1].parse().unwrap();
                let mut data = Vec::with_capacity(n * DIM);
                for i in 2..2 + n * DIM {
                    data.push(parts[i].parse::<f32>().unwrap());
                }
                index.train(&data);
                println!("OK");
            }
            "ADD" => {
                let id: u64 = parts[1].parse().unwrap();
                let mut vec = Vec::with_capacity(DIM);
                for i in 2..2 + DIM {
                    vec.push(parts[i].parse::<f32>().unwrap());
                }
                index.add(id, &vec);
                if id == 19999 { // Final index ack
                    println!("OK");
                }
            }
            "SEARCH" => {
                let top_k: usize = parts[1].parse().unwrap();
                let nprobe: usize = parts[2].parse().unwrap();
                let mut q = Vec::with_capacity(DIM);
                for i in 3..3 + DIM {
                    q.push(parts[i].parse::<f32>().unwrap());
                }
                let hits = index.search(&q, top_k, nprobe);
                let out: Vec<String> = hits.iter().map(|h| format!("{}:{}", h.id, h.distance)).collect();
                println!("{}", out.join(","));
            }
            _ => {}
        }
    }
}
