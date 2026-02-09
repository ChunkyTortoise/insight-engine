[![Sponsor](https://img.shields.io/badge/Sponsor-💖-pink.svg)](https://github.com/sponsors/ChunkyTortoise)

# Insight Engine

**Marketing teams waste 8+ hours/week building reports from spreadsheets.** Upload a CSV or Excel file and get instant dashboards, predictive models, marketing attribution, and downloadable reports.

![CI](https://github.com/ChunkyTortoise/insight-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-134%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B.svg?logo=streamlit&logoColor=white)](https://ct-insight-engine.streamlit.app)

**[Live Demo](https://ct-insight-engine.streamlit.app)** -- try it without installing anything.

## What This Solves

- **Manual reporting burns time** -- Auto-profiler detects column types, distributions, outliers, and correlations in seconds
- **No visibility into which channels drive conversions** -- Four attribution models show exactly where marketing budget should go
- **Predictive modeling requires ML expertise** -- Upload labeled data, pick a target column, get a trained model with SHAP explanations
- **No way to segment customers** -- K-means and DBSCAN clustering with silhouette scoring for automatic customer segmentation
- **Forecasting requires specialized tools** -- Moving average, exponential smoothing, and ensemble forecasts from any time series column

## Architecture

```
CSV/Excel Upload
      |
      v
+--------------+    +------------------+    +-----------------+
|  Profiler    |--->|  Dashboard Gen   |--->|  Report Gen     |
|  (auto-type  |    |  (Plotly charts, |    |  (Markdown/PDF  |
|   detection) |    |   auto-layout)   |    |   with charts)  |
+--------------+    +------------------+    +-----------------+
       |
       +---> Cleaner (dedup, standardize, impute)
       +---> Predictor (auto-ML, SHAP)
       +---> Attribution (4 models)
       +---> Anomaly Detector (Z-score + IQR)
       +---> Clustering (K-means, DBSCAN)
       +---> Feature Lab (scaling, encoding, polynomials)
       +---> Forecaster (moving avg, exp smoothing, ensemble)
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| **Profiler** | `profiler.py` | Auto-detect column types, distributions, outliers, and correlations |
| **Dashboard Generator** | `dashboard_generator.py` | Plotly histograms, pie charts, heatmaps, scatter matrices |
| **Data Cleaner** | `cleaner.py` | Dedup (exact + fuzzy), column standardization, smart imputation |
| **Predictor** | `predictor.py` | Auto-detect classification/regression, gradient boosting, SHAP |
| **Attribution** | `attribution.py` | First-touch, last-touch, linear, time-decay marketing attribution |
| **Report Generator** | `report_generator.py` | Markdown reports with findings, metrics, chart placeholders |
| **Anomaly Detector** | `anomaly_detector.py` | Z-score and IQR outlier detection (stdlib only, no numpy required) |
| **Clustering** | `clustering.py` | K-means and DBSCAN with silhouette scoring and cluster comparison |
| **Feature Lab** | `feature_lab.py` | Feature scaling, encoding, polynomial features, interaction terms |
| **Forecaster** | `forecaster.py` | Moving average, exponential smoothing, linear trend, ensemble forecasts |

## Quick Start

```bash
git clone https://github.com/ChunkyTortoise/insight-engine.git
cd insight-engine
pip install -r requirements.txt
make test
make demo
```

## Demo Datasets

| Dataset | Rows | Use Case |
|---------|------|----------|
| E-commerce Transactions | 1,000 | Revenue analysis, category distributions, return rates |
| Marketing Touchpoints | ~800 | Attribution modeling across 6 channels |
| HR Attrition | 500 | Predictive modeling (who will leave?) |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit, Plotly |
| Data | Pandas, NumPy, openpyxl |
| ML | scikit-learn, XGBoost, SHAP |
| Testing | pytest (134 tests) |
| CI | GitHub Actions (Python 3.11, 3.12) |
| Linting | Ruff |

## Project Structure

```
insight-engine/
├── app.py                          # Streamlit application
├── insight_engine/
│   ├── profiler.py                 # Auto-profiling + column type detection
│   ├── dashboard_generator.py      # Chart generation + layout
│   ├── attribution.py              # 4 marketing attribution models
│   ├── predictor.py                # Auto-ML + SHAP explanations
│   ├── cleaner.py                  # Dedup, standardize, impute
│   ├── report_generator.py         # Markdown/PDF report generation
│   ├── anomaly_detector.py         # Z-score + IQR outlier detection
│   ├── clustering.py               # K-means, DBSCAN, silhouette scores
│   ├── feature_lab.py              # Feature scaling, encoding, polynomials
│   └── forecaster.py               # Time series forecasting (4 methods)
├── demo_data/                      # 3 sample datasets
├── tests/                          # 10 test files, one per module
├── .github/workflows/ci.yml        # CI pipeline
├── Makefile                        # demo, test, lint, setup
└── requirements.txt
```

## Testing

```bash
make test                           # Full suite (134 tests)
python -m pytest tests/ -v          # Verbose output
python -m pytest tests/test_profiler.py  # Single module
```

## Related Projects

- [EnterpriseHub](https://github.com/ChunkyTortoise/EnterpriseHub) -- Real estate AI platform with BI dashboards and CRM integration
- [docqa-engine](https://github.com/ChunkyTortoise/docqa-engine) -- RAG document Q&A with hybrid retrieval and prompt engineering lab
- [ai-orchestrator](https://github.com/ChunkyTortoise/ai-orchestrator) -- AgentForge: unified async LLM interface (Claude, Gemini, OpenAI, Perplexity)
- [scrape-and-serve](https://github.com/ChunkyTortoise/scrape-and-serve) -- Web scraping, price monitoring, Excel-to-web apps, and SEO tools
- [prompt-engineering-lab](https://github.com/ChunkyTortoise/prompt-engineering-lab) -- 8 prompt patterns, A/B testing, TF-IDF evaluation
- [llm-integration-starter](https://github.com/ChunkyTortoise/llm-integration-starter) -- Production LLM patterns: completion, streaming, function calling, RAG, hardening
- [Portfolio](https://chunkytortoise.github.io) -- Project showcase and services

## Deploy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/chunkytortoise/insight-engine/main/app.py)

## License

MIT -- see [LICENSE](LICENSE) for details.
