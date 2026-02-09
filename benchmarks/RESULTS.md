# Insight Engine Benchmark Results

**Date**: 2026-02-09 03:32:55

| Operation | Iterations | P50 (ms) | P95 (ms) | P99 (ms) | Throughput |
|-----------|-----------|----------|----------|----------|------------|
| CSV Auto-Profiling | 500 | 0.0487 | 0.0517 | 0.0577 | 20,369 ops/sec |
| K-Means Clustering (500pts, 3 clusters) | 200 | 2.1847 | 3.0143 | 5.0959 | 430 ops/sec |
| Z-Score Anomaly Detection (1K points) | 500 | 0.1031 | 0.1125 | 0.1708 | 9,510 ops/sec |
| T-Test (2 x 100 samples) | 1,000 | 0.0102 | 0.0104 | 0.0113 | 97,501 ops/sec |

> All benchmarks use synthetic data. No external services required.
