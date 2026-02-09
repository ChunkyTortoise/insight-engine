# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-02-09

### Added
- Data quality scoring module (completeness, validity, consistency checks)
- Regression diagnostics (residual analysis, VIF, heteroscedasticity testing)
- Advanced anomaly detection (isolation forest, LOF, multi-method ensemble)
- Dockerfile and docker-compose.yml for containerized deployment
- Architecture Decision Records (docs/adr/)
- Performance benchmarks (benchmarks/)
- SECURITY.md and CODE_OF_CONDUCT.md

### Changed
- README architecture diagram upgraded from ASCII to Mermaid
- Test badge updated to 521+ tests

## [0.2.0] - 2026-01-28

### Added
- Statistical testing framework: t-test, chi-square, ANOVA, Mann-Whitney, Kruskal-Wallis, Shapiro-Wilk
- KPI framework with custom metric definitions and threshold alerting
- Dimensionality reduction (PCA, t-SNE) with visualization
- 92 new tests covering statistical testing, KPI framework, and dimensionality reduction

## [0.1.0] - 2026-01-15

### Added
- Auto-profiler with column type detection (numeric, categorical, datetime, text)
- Dashboard generator with Plotly charts and auto-layout
- Data cleaner (dedup, standardize, impute)
- Predictor with auto-ML and SHAP explanations
- Four marketing attribution models (first-touch, last-touch, linear, time-decay)
- Report generator (Markdown with chart placeholders)
- Anomaly detector (Z-score, IQR)
- Clustering (K-means, DBSCAN with silhouette scoring)
- Feature lab (scaling, encoding, polynomial features)
- Forecaster (moving average, exponential smoothing, linear trend, ensemble)
- Model observatory (SHAP explainability, feature importance)
- Hyperparameter tuner with cross-validation
- Streamlit dashboard application
- 3 demo datasets (e-commerce, marketing touchpoints, HR attrition)
- CI pipeline with GitHub Actions (Python 3.11, 3.12)
- 313 automated tests
