# Insight Engine

**Marketing teams waste 8+ hours/week building reports from spreadsheets.** Upload a CSV or Excel file and get instant dashboards, predictive models, marketing attribution, and downloadable reports.

[![CI](https://img.shields.io/github/actions/workflow/status/ChunkyTortoise/insight-engine/ci.yml?label=CI)](https://github.com/ChunkyTortoise/insight-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F1C40F.svg)](LICENSE)

## What This Solves

- **Manual reporting burns time** -- Auto-profiler detects column types, distributions, outliers, and correlations in seconds. Dashboard generator picks the right chart types automatically.
- **No visibility into which channels drive conversions** -- Four attribution models (first-touch, last-touch, linear, time-decay) show exactly where marketing budget should go.
- **Predictive modeling requires ML expertise** -- Upload labeled data, pick a target column, get a trained model with accuracy metrics, feature importances, and SHAP explanations.

## Architecture

```
CSV/Excel Upload
      |
      v
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Profiler    │───>│  Dashboard Gen   │───>│  Report Gen     │
│  (auto-type  │    │  (Plotly charts,  │    │  (Markdown/PDF   │
│   detection) │    │   auto-layout)   │    │   with charts)  │
└──────┬───────┘    └──────────────────┘    └─────────────────┘
       │
       ├───> Cleaner (dedup, standardize, impute)
       ├───> Predictor (auto-ML, SHAP)
       └───> Attribution (4 models)
```

## Quick Start

```bash
git clone https://github.com/ChunkyTortoise/insight-engine.git
cd insight-engine
pip install -r requirements.txt

# Demo mode -- 3 sample datasets, no config needed
make demo
```

### What You Get

1. **Auto-Profiler** -- Column types, null rates, outlier counts, correlations
2. **Dashboard Generator** -- Histograms, pie charts, heatmaps, scatter matrices (Plotly)
3. **Data Cleaner** -- Dedup (exact + fuzzy), column standardization, smart imputation
4. **Predictive Modeling** -- Auto-detects classification/regression, trains gradient boosting, SHAP explanations
5. **Marketing Attribution** -- First-touch, last-touch, linear, time-decay across any channel data
6. **Report Generator** -- Markdown reports with findings, metrics, and chart placeholders

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
| Testing | pytest |
| CI | GitHub Actions (Python 3.11, 3.12) |

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
│   └── report_generator.py         # Markdown/PDF report generation
├── demo_data/
│   ├── generate_demo_data.py       # Reproducible sample data generator
│   ├── ecommerce.csv               # 1,000 transactions
│   ├── marketing_touchpoints.csv   # ~800 touchpoints
│   └── hr_attrition.csv            # 500 employees
├── tests/                          # One test file per module
├── .github/workflows/ci.yml        # CI pipeline
├── Makefile                        # demo, test, lint, setup
└── requirements.txt
```

## Testing

```bash
make test                           # Full suite
python -m pytest tests/ -v          # Verbose output
python -m pytest tests/test_profiler.py  # Single module
```

## Related Projects

- [EnterpriseHub](https://github.com/ChunkyTortoise/EnterpriseHub) -- Real estate AI platform with BI dashboards and CRM integration
- [jorge_real_estate_bots](https://github.com/ChunkyTortoise/jorge_real_estate_bots) -- Three-bot lead qualification system (Lead, Buyer, Seller)
- [ai-orchestrator](https://github.com/ChunkyTortoise/ai-orchestrator) -- AgentForge: unified async LLM interface (Claude, Gemini, OpenAI, Perplexity)
- [Revenue-Sprint](https://github.com/ChunkyTortoise/Revenue-Sprint) -- AI-powered freelance pipeline: job scanning, proposal generation, prompt injection testing

## License

MIT -- see [LICENSE](LICENSE) for details.
