"""Hyperparameter tuning: grid search and random search with default param grids."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC


@dataclass
class TuningResult:
    """Result of a hyperparameter tuning run."""

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict]
    search_method: str
    total_combinations: int
    elapsed_ms: float


_MODEL_CLASSES: dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "ridge": RidgeClassifier,
    "svm": SVC,
}

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
    "gradient_boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [200]},
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
}


def _build_model(model_name: str) -> Any:
    """Create a model instance with sensible defaults."""
    if model_name not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODEL_CLASSES.keys())}")

    model_cls = _MODEL_CLASSES[model_name]

    if model_name == "svm":
        return model_cls(probability=True)
    if model_name == "logistic_regression":
        return model_cls(max_iter=200)
    return model_cls()


def _extract_cv_results(cv_results: dict) -> list[dict]:
    """Extract per-candidate results from sklearn CV results dict."""
    all_results: list[dict] = []
    n_candidates = len(cv_results["mean_test_score"])
    for i in range(n_candidates):
        entry: dict[str, Any] = {
            "params": cv_results["params"][i],
            "mean_score": round(float(cv_results["mean_test_score"][i]), 6),
            "std_score": round(float(cv_results["std_test_score"][i]), 6),
            "rank": int(cv_results["rank_test_score"][i]),
        }
        all_results.append(entry)
    return all_results


class HyperTuner:
    """Hyperparameter tuning via grid search and random search."""

    def grid_search(
        self,
        X: Any,
        y: Any,
        model_name: str,
        param_grid: dict[str, list] | None = None,
        cv: int = 3,
        metric: str = "f1",
        random_state: int = 42,
    ) -> TuningResult:
        """Exhaustive grid search over parameter combinations."""
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRIDS.get(model_name, {})

        model = _build_model(model_name)

        scoring = f"{metric}_weighted" if metric in ("precision", "recall", "f1") else metric

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score=0.0,
        )

        start = time.perf_counter()
        search.fit(X, y)
        elapsed_ms = (time.perf_counter() - start) * 1000

        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        return TuningResult(
            model_name=model_name,
            best_params=dict(search.best_params_),
            best_score=round(float(search.best_score_), 6),
            all_results=_extract_cv_results(search.cv_results_),
            search_method="grid",
            total_combinations=total_combinations,
            elapsed_ms=round(elapsed_ms, 3),
        )

    def random_search(
        self,
        X: Any,
        y: Any,
        model_name: str,
        param_distributions: dict[str, list] | None = None,
        n_iter: int = 10,
        cv: int = 3,
        metric: str = "f1",
        random_state: int = 42,
    ) -> TuningResult:
        """Randomized search over parameter distributions."""
        if param_distributions is None:
            param_distributions = DEFAULT_PARAM_GRIDS.get(model_name, {})

        model = _build_model(model_name)

        scoring = f"{metric}_weighted" if metric in ("precision", "recall", "f1") else metric

        # Cap n_iter to total possible combinations
        total_combinations = 1
        for values in param_distributions.values():
            total_combinations *= len(values)
        effective_n_iter = min(n_iter, total_combinations)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=effective_n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=1,
            error_score=0.0,
        )

        start = time.perf_counter()
        search.fit(X, y)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return TuningResult(
            model_name=model_name,
            best_params=dict(search.best_params_),
            best_score=round(float(search.best_score_), 6),
            all_results=_extract_cv_results(search.cv_results_),
            search_method="random",
            total_combinations=total_combinations,
            elapsed_ms=round(elapsed_ms, 3),
        )

    def auto_tune(
        self,
        X: Any,
        y: Any,
        models: list[str] | None = None,
        method: str = "random",
        n_iter: int = 10,
        cv: int = 3,
        metric: str = "f1",
        random_state: int = 42,
    ) -> dict[str, TuningResult]:
        """Tune multiple models automatically, returning best params for each."""
        if models is None:
            models = list(_MODEL_CLASSES.keys())

        results: dict[str, TuningResult] = {}
        for model_name in models:
            if method == "grid":
                results[model_name] = self.grid_search(
                    X,
                    y,
                    model_name,
                    cv=cv,
                    metric=metric,
                    random_state=random_state,
                )
            else:
                results[model_name] = self.random_search(
                    X,
                    y,
                    model_name,
                    n_iter=n_iter,
                    cv=cv,
                    metric=metric,
                    random_state=random_state,
                )

        return results
