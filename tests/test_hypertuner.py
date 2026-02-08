"""Tests for the hyperparameter tuning module."""

import pytest
from sklearn.datasets import make_classification

from insight_engine.hypertuner import DEFAULT_PARAM_GRIDS, HyperTuner, TuningResult


def _make_dataset(n_samples: int = 150, n_features: int = 5, random_state: int = 42):
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        random_state=random_state,
    )
    return X, y


class TestHyperTuner:
    def test_grid_search(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        result = tuner.grid_search(X, y, "ridge")
        assert isinstance(result, TuningResult)
        assert result.search_method == "grid"
        assert result.best_score > 0

    def test_random_search(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        result = tuner.random_search(X, y, "ridge", n_iter=3)
        assert isinstance(result, TuningResult)
        assert result.search_method == "random"
        assert result.best_score > 0

    def test_default_params(self):
        assert "random_forest" in DEFAULT_PARAM_GRIDS
        assert "gradient_boosting" in DEFAULT_PARAM_GRIDS
        assert "logistic_regression" in DEFAULT_PARAM_GRIDS
        assert "ridge" in DEFAULT_PARAM_GRIDS
        assert "svm" in DEFAULT_PARAM_GRIDS

    def test_custom_params(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        custom_grid = {"alpha": [0.5, 1.0, 5.0]}
        result = tuner.grid_search(X, y, "ridge", param_grid=custom_grid)
        assert result.best_params["alpha"] in [0.5, 1.0, 5.0]
        assert result.total_combinations == 3

    def test_auto_tune(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        results = tuner.auto_tune(X, y, models=["ridge", "logistic_regression"], method="random", n_iter=3)
        assert "ridge" in results
        assert "logistic_regression" in results
        assert isinstance(results["ridge"], TuningResult)
        assert isinstance(results["logistic_regression"], TuningResult)

    def test_cv_respected(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        # Use grid search with small grid so cv folds are clear
        result = tuner.grid_search(X, y, "ridge", param_grid={"alpha": [1.0]}, cv=5)
        assert isinstance(result, TuningResult)
        assert result.best_score > 0

    def test_unknown_model_raises(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        with pytest.raises(ValueError, match="Unknown model"):
            tuner.grid_search(X, y, "nonexistent_model")


class TestTuningResult:
    def test_fields(self):
        result = TuningResult(
            model_name="ridge",
            best_params={"alpha": 1.0},
            best_score=0.85,
            all_results=[{"params": {"alpha": 1.0}, "mean_score": 0.85}],
            search_method="grid",
            total_combinations=3,
            elapsed_ms=100.0,
        )
        assert result.model_name == "ridge"
        assert result.elapsed_ms == 100.0

    def test_best_params_present(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        result = tuner.grid_search(X, y, "ridge")
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0

    def test_all_results_populated(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        result = tuner.grid_search(X, y, "ridge")
        assert isinstance(result.all_results, list)
        assert len(result.all_results) > 0
        first = result.all_results[0]
        assert "params" in first
        assert "mean_score" in first

    def test_search_method_recorded(self):
        X, y = _make_dataset()
        tuner = HyperTuner()
        grid_result = tuner.grid_search(X, y, "ridge")
        assert grid_result.search_method == "grid"
        random_result = tuner.random_search(X, y, "ridge", n_iter=2)
        assert random_result.search_method == "random"
