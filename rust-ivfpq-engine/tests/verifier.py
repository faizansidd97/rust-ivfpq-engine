import subprocess
import numpy as np
import time
import sys

DIM = 128
N_TRAIN = 3000
N_VEC = 20000
N_QUERY = 200
TOP_K = 10
NPROBE = 8

np.random.seed(1337)

# Generate clustered synthetic vectors
cluster_centers = np.random.randn(64, DIM).astype(np.float32)
data = []
for _ in range(N_VEC):
    c = cluster_centers[np.random.choice(64)]
    data.append(c + np.random.randn(DIM).astype(np.float32) * 0.2)
data = np.vstack(data)

queries = []
for _ in range(N_QUERY):
    c = cluster_centers[np.random.choice(64)]
    queries.append(c + np.random.randn(DIM).astype(np.float32) * 0.2)
queries = np.vstack(queries)

# Compute ground truth via exact Euclidean distance
ground_truth = []
for q in queries:
    dists = np.sum((data - q) ** 2, axis=1)
    ground_truth.append(set(np.argsort(dists)[:TOP_K]))

proc = subprocess.Popen(
    ["/workspace/target/release/vector-engine"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd="/workspace"
)

try:
    # 1. Train
    train_data = data[:N_TRAIN].flatten()
    proc.stdin.write(f"TRAIN {N_TRAIN} " + " ".join(f"{x:.4f}" for x in train_data) + "\n")
    proc.stdin.flush()
    res = proc.stdout.readline().strip()
    if res != "OK":
        sys.exit(1)

    # 2. Add
    for i in range(N_VEC):
        proc.stdin.write(f"ADD {i} " + " ".join(f"{x:.4f}" for x in data[i]) + "\n")
        if i % 1000 == 0:
            proc.stdin.flush()
    proc.stdin.flush()
    
    # Wait for completion acknowledgment
    res = proc.stdout.readline().strip()
    if res != "OK":
        sys.exit(1)

    # 3. Search and profile latency
    hits = 0
    latencies = []
    for q_idx, q in enumerate(queries):
        start = time.perf_counter()
        proc.stdin.write(f"SEARCH {TOP_K} {NPROBE} " + " ".join(f"{x:.4f}" for x in q) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        latencies.append(time.perf_counter() - start)

        if line:
            returned_ids = [int(item.split(":")[0]) for item in line.split(",") if ":" in item]
            hits += len(set(returned_ids).intersection(ground_truth[q_idx]))

    recall = hits / (N_QUERY * TOP_K)
    mean_latency_ms = float(np.mean(latencies)) * 1000.0

    # Verification SLA: Recall >= 0.82 and Latency < 2.0 ms
    if recall >= 0.82 and mean_latency_ms < 2.0:
        score = 1.000
    else:
        score = 0.000

    sys.stderr.write(f"Recall: {recall:.4f}, Latency: {mean_latency_ms:.2f}ms => Score: {score:.3f}\n")
    print(f"{score:.3f}")

except Exception as e:
    sys.stderr.write(f"Verifier error: {e}\n")
    print("0.000")
finally:
    if proc.poll() is None:
        proc.kill()
