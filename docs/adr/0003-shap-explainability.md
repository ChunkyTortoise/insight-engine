# ADR 0003: SHAP Explainability

## Status
Accepted

## Context
Tree-based feature importance (e.g., Gini importance) is biased toward high-cardinality features, producing misleading explanations. Users need trustworthy, theoretically grounded explanations of model predictions to make informed decisions and build confidence in the system's recommendations.

## Decision
Use SHAP (SHapley Additive exPlanations) values for all prediction models. Generate waterfall plots for individual prediction explanations and beeswarm plots for global feature importance. Use the TreeSHAP approximation for tree-based models to mitigate computational cost.

## Consequences
- **Positive**: SHAP values are theoretically grounded in cooperative game theory, providing consistent and locally accurate feature attributions. Waterfall and beeswarm visualizations are intuitive for non-technical users. TreeSHAP makes computation tractable for tree-based models.
- **Negative**: SHAP computation is expensive for non-tree models (exact SHAP is exponential). Even with TreeSHAP, large datasets with many features can be slow. Users may over-interpret small SHAP differences as meaningful.
