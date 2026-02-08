"""Train and compare multiple model types, auto-select best, feature importances."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@dataclass
class ModelResult:
    """Result of training a single model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importances: dict[str, float]
    training_time_ms: float


@dataclass
class ObservatoryReport:
    """Comparison report across multiple models."""

    results: list[ModelResult]
    best_model: str
    best_score: float
    metric_used: str
    feature_rankings: list[tuple[str, float]]


_METRIC_FUNCTIONS: dict[str, Any] = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
    "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
    "f1": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
}


def _get_feature_importances(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from a fitted model.

    Uses feature_importances_ for tree-based models and coef_ for linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
    else:
        return {name: 0.0 for name in feature_names}

    return {name: round(float(imp), 6) for name, imp in zip(feature_names, importances)}


class ModelObservatory:
    """Train and compare multiple model types."""

    SUPPORTED_MODELS: dict[str, type] = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "ridge": RidgeClassifier,
        "svm": SVC,
    }

    def train_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_names: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs: Any,
    ) -> ModelResult:
        """Train a single model and return its performance metrics."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(self.SUPPORTED_MODELS.keys())}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Build model with sensible defaults
        model_cls = self.SUPPORTED_MODELS[model_name]
        model_kwargs: dict[str, Any] = {}

        if model_name == "svm":
            model_kwargs["probability"] = True
        if model_name == "logistic_regression":
            model_kwargs["max_iter"] = 200
        if model_name in ("random_forest", "gradient_boosting"):
            model_kwargs["random_state"] = random_state
        if model_name == "svm":
            model_kwargs["random_state"] = random_state

        model_kwargs.update(kwargs)
        model = model_cls(**model_kwargs)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed_ms = (time.perf_counter() - start) * 1000

        y_pred = model.predict(X_test)

        importances = _get_feature_importances(model, feature_names)

        return ModelResult(
            name=model_name,
            accuracy=round(float(accuracy_score(y_test, y_pred)), 6),
            precision=round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
            recall=round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
            f1=round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
            feature_importances=importances,
            training_time_ms=round(elapsed_ms, 3),
        )

    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[str] | None = None,
        metric: str = "f1",
        test_size: float = 0.2,
        feature_names: list[str] | None = None,
        random_state: int = 42,
    ) -> ObservatoryReport:
        """Train multiple models and compare by the specified metric."""
        if metric not in _METRIC_FUNCTIONS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(_METRIC_FUNCTIONS.keys())}")

        if models is None:
            models = list(self.SUPPORTED_MODELS.keys())

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        results: list[ModelResult] = []
        for model_name in models:
            result = self.train_single(
                X,
                y,
                model_name,
                feature_names=feature_names,
                test_size=test_size,
                random_state=random_state,
            )
            results.append(result)

        # Select best by metric
        best_result = max(results, key=lambda r: getattr(r, metric))
        best_model = best_result.name
        best_score = getattr(best_result, metric)

        # Aggregate feature importances across all models
        feature_totals: dict[str, list[float]] = {name: [] for name in feature_names}
        for result in results:
            for name, imp in result.feature_importances.items():
                if name in feature_totals:
                    feature_totals[name].append(imp)

        feature_rankings = [
            (name, round(sum(vals) / len(vals), 6) if vals else 0.0) for name, vals in feature_totals.items()
        ]
        feature_rankings.sort(key=lambda x: x[1], reverse=True)

        return ObservatoryReport(
            results=results,
            best_model=best_model,
            best_score=best_score,
            metric_used=metric,
            feature_rankings=feature_rankings,
        )

    def get_feature_importances(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importances from a fitted model."""
        return _get_feature_importances(model, feature_names)
