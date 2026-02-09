"""Insight Engine Performance Benchmarks."""
import time
import statistics
import random
import json
from pathlib import Path

random.seed(42)

def percentile(data, p):
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])

def benchmark_csv_profiling():
    """Auto-profile CSV column type detection."""
    columns = [
        [random.gauss(100, 15) for _ in range(1000)],
        [random.choice(["A", "B", "C", "D"]) for _ in range(1000)],
        [f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(1000)],
    ]
    times = []
    for _ in range(500):
        start = time.perf_counter()
        for col in columns:
            sample = col[:100]
            numeric = sum(1 for v in sample if isinstance(v, (int, float))) / len(sample)
            unique_ratio = len(set(str(v) for v in sample)) / len(sample)
            if numeric > 0.8: dtype = "numeric"
            elif unique_ratio < 0.05: dtype = "categorical"
            else: dtype = "text"
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return {"op": "CSV Auto-Profiling", "n": 500, "p50": round(percentile(times, 50), 4), "p95": round(percentile(times, 95), 4), "p99": round(percentile(times, 99), 4), "ops_sec": round(500 / (sum(times) / 1000), 1)}

def benchmark_clustering():
    """K-Means clustering computation."""
    import math
    data = [(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(500)]
    times = []
    for _ in range(200):
        start = time.perf_counter()
        centroids = [data[i] for i in range(3)]
        for _ in range(10):
            clusters = [[] for _ in range(3)]
            for point in data:
                dists = [math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in centroids]
                clusters[dists.index(min(dists))].append(point)
            centroids = [(sum(p[0] for p in c)/max(len(c),1), sum(p[1] for p in c)/max(len(c),1)) for c in clusters]
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return {"op": "K-Means Clustering (500pts, 3 clusters)", "n": 200, "p50": round(percentile(times, 50), 4), "p95": round(percentile(times, 95), 4), "p99": round(percentile(times, 99), 4), "ops_sec": round(200 / (sum(times) / 1000), 1)}

def benchmark_anomaly_detection():
    """Z-score anomaly detection."""
    data = [random.gauss(50, 10) for _ in range(1000)]
    data[500] = 150  # inject anomaly
    times = []
    for _ in range(500):
        start = time.perf_counter()
        mean = sum(data) / len(data)
        std = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5
        anomalies = [x for x in data if abs(x - mean) / max(std, 1e-10) > 3]
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return {"op": "Z-Score Anomaly Detection (1K points)", "n": 500, "p50": round(percentile(times, 50), 4), "p95": round(percentile(times, 95), 4), "p99": round(percentile(times, 99), 4), "ops_sec": round(500 / (sum(times) / 1000), 1)}

def benchmark_statistical_test():
    """T-test computation."""
    a = [random.gauss(50, 10) for _ in range(100)]
    b = [random.gauss(55, 10) for _ in range(100)]
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        mean_a, mean_b = sum(a)/len(a), sum(b)/len(b)
        var_a = sum((x - mean_a)**2 for x in a) / (len(a) - 1)
        var_b = sum((x - mean_b)**2 for x in b) / (len(b) - 1)
        se = (var_a/len(a) + var_b/len(b)) ** 0.5
        t_stat = (mean_a - mean_b) / max(se, 1e-10)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return {"op": "T-Test (2 x 100 samples)", "n": 1000, "p50": round(percentile(times, 50), 4), "p95": round(percentile(times, 95), 4), "p99": round(percentile(times, 99), 4), "ops_sec": round(1000 / (sum(times) / 1000), 1)}

def main():
    results = []
    for bench in [benchmark_csv_profiling, benchmark_clustering, benchmark_anomaly_detection, benchmark_statistical_test]:
        print(f"Running {bench.__doc__.strip()}...")
        r = bench()
        results.append(r)
        print(f"  P50: {r['p50']}ms | P95: {r['p95']}ms | P99: {r['p99']}ms | {r['ops_sec']} ops/sec")

    out = Path(__file__).parent / "RESULTS.md"
    with open(out, "w") as f:
        f.write("# Insight Engine Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Operation | Iterations | P50 (ms) | P95 (ms) | P99 (ms) | Throughput |\n")
        f.write("|-----------|-----------|----------|----------|----------|------------|\n")
        for r in results:
            f.write(f"| {r['op']} | {r['n']:,} | {r['p50']} | {r['p95']} | {r['p99']} | {r['ops_sec']:,.0f} ops/sec |\n")
        f.write("\n> All benchmarks use synthetic data. No external services required.\n")
    print(f"\nResults: {out}")

if __name__ == "__main__":
    main()
