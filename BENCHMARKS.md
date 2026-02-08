# Insight Engine -- Benchmarks

Generated: 2026-02-08

## Test Suite Summary

134 tests across 10 modules. All tests run without network access or external API keys.

| Module | Test File | Tests | Description |
|--------|-----------|-------|-------------|
| Profiler | `test_profiler.py` | ~14 | Column type detection, distributions, correlations |
| Dashboard Generator | `test_dashboard_generator.py` | ~12 | Chart type selection, layout, Plotly output |
| Data Cleaner | `test_cleaner.py` | ~14 | Dedup, standardization, imputation |
| Predictor | `test_predictor.py` | ~12 | Classification/regression, SHAP, feature importance |
| Attribution | `test_attribution.py` | ~14 | 4 attribution models, channel allocation |
| Report Generator | `test_report_generator.py` | ~12 | Markdown output, findings, metrics |
| Anomaly Detector | `test_anomaly_detector.py` | ~14 | Z-score, IQR, multi-column detection |
| Clustering | `test_clustering.py` | ~14 | K-means, DBSCAN, silhouette, comparison |
| Feature Lab | `test_feature_lab.py` | ~14 | Scaling, encoding, polynomials, interactions |
| Forecaster | `test_forecaster.py` | ~14 | Moving avg, exp smoothing, trend, ensemble |
| **Total** | **10 files** | **134** | |

## Tech Stack Versions

| Dependency | Version |
|-----------|---------|
| Python | 3.11, 3.12 |
| scikit-learn | >=1.3 |
| XGBoost | >=1.7 |
| Pandas | >=2.0 |
| NumPy | >=1.24 |

## How to Reproduce

```bash
git clone https://github.com/ChunkyTortoise/insight-engine.git
cd insight-engine
pip install -r requirements.txt
make test
# or: python -m pytest tests/ -v
```

## Notes

- All ML tests use deterministic seeds for reproducibility
- Anomaly detector tests use stdlib only (no numpy)
- Forecaster tests use pure-Python math for portability
- No external API calls or network access required
