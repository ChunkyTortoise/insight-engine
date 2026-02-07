"""Predictive Modeling: auto-train classifier/regressor with SHAP explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class PredictionResult:
    task_type: str  # "classification" or "regression"
    target_column: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]
    shap_values: np.ndarray | None = None
    predictions: np.ndarray | None = None
    model: Any = None
    feature_names: list[str] = field(default_factory=list)


def detect_task_type(series: pd.Series) -> str:
    """Detect whether target is classification or regression."""
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(len(series), 1)
        if series.nunique() <= 20 or unique_ratio < 0.05:
            return "classification"
        return "regression"
    return "classification"


def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str], dict[str, LabelEncoder]]:
    """Prepare features: encode categoricals, drop target, handle NaN."""
    feature_df = df.drop(columns=[target]).copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in feature_df.columns:
        if feature_df[col].dtype == "object" or feature_df[col].dtype.name == "category":
            le = LabelEncoder()
            feature_df[col] = feature_df[col].fillna("__missing__")
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            encoders[col] = le
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    feature_names = list(feature_df.columns)
    return feature_df, feature_names, encoders


def train_model(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PredictionResult:
    """Auto-detect task type, train model, compute metrics and feature importances."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    task_type = detect_task_type(df[target])
    feature_df, feature_names, encoders = _prepare_features(df, target)

    y = df[target].copy()
    target_encoder = None
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.fillna("__missing__").astype(str)))
    else:
        y = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else 0)

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=test_size, random_state=random_state)

    if task_type == "classification":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_weighted": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
        }
    else:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "r2": round(r2_score(y_test, preds), 4),
            "mae": round(mean_absolute_error(y_test, preds), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        }

    # Feature importances
    importances = dict(zip(feature_names, [round(float(v), 4) for v in model.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # SHAP values (optional — try import)
    shap_values = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except ImportError:
        pass

    return PredictionResult(
        task_type=task_type,
        target_column=target,
        metrics=metrics,
        feature_importances=importances,
        shap_values=shap_values,
        predictions=preds,
        model=model,
        feature_names=feature_names,
    )
