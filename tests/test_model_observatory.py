"""Tests for the model observatory module."""

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from insight_engine.model_observatory import ModelObservatory, ModelResult, ObservatoryReport


def _make_dataset(n_samples: int = 200, n_features: int = 5, random_state: int = 42):
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        random_state=random_state,
    )
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


class TestModelResult:
    def test_fields(self):
        result = ModelResult(
            name="test_model",
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1=0.86,
            feature_importances={"a": 0.5, "b": 0.3},
            training_time_ms=42.0,
        )
        assert result.name == "test_model"
        assert result.accuracy == 0.9
        assert result.precision == 0.85
        assert result.recall == 0.88
        assert result.f1 == 0.86
        assert result.training_time_ms == 42.0

    def test_feature_importances(self):
        result = ModelResult(
            name="rf",
            accuracy=0.8,
            precision=0.8,
            recall=0.8,
            f1=0.8,
            feature_importances={"x": 0.7, "y": 0.2, "z": 0.1},
            training_time_ms=10.0,
        )
        assert len(result.feature_importances) == 3
        assert result.feature_importances["x"] == 0.7


class TestModelObservatory:
    def test_train_single_rf(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        result = obs.train_single(X, y, "random_forest", feature_names=names)
        assert isinstance(result, ModelResult)
        assert result.name == "random_forest"
        assert 0.0 <= result.accuracy <= 1.0
        assert result.training_time_ms > 0

    def test_train_single_lr(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        result = obs.train_single(X, y, "logistic_regression", feature_names=names)
        assert isinstance(result, ModelResult)
        assert result.name == "logistic_regression"
        assert 0.0 <= result.f1 <= 1.0

    def test_compare_models(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, models=["random_forest", "logistic_regression"], feature_names=names)
        assert isinstance(report, ObservatoryReport)
        assert len(report.results) == 2
        assert report.best_model in ("random_forest", "logistic_regression")

    def test_best_model_selected(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, models=["random_forest", "ridge"], metric="accuracy", feature_names=names)
        # Best score should match the best model's metric
        best = next(r for r in report.results if r.name == report.best_model)
        assert report.best_score == best.accuracy

    def test_feature_importances_tree(self):
        X, y, names = _make_dataset()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        obs = ModelObservatory()
        imps = obs.get_feature_importances(model, names)
        assert len(imps) == len(names)
        assert all(v >= 0 for v in imps.values())

    def test_feature_importances_linear(self):
        X, y, names = _make_dataset()
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        obs = ModelObservatory()
        imps = obs.get_feature_importances(model, names)
        assert len(imps) == len(names)

    def test_supported_models(self):
        obs = ModelObservatory()
        assert "random_forest" in obs.SUPPORTED_MODELS
        assert "gradient_boosting" in obs.SUPPORTED_MODELS
        assert "logistic_regression" in obs.SUPPORTED_MODELS
        assert "ridge" in obs.SUPPORTED_MODELS
        assert "svm" in obs.SUPPORTED_MODELS
        assert len(obs.SUPPORTED_MODELS) >= 5

    def test_custom_metric(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(
            X,
            y,
            models=["random_forest", "logistic_regression"],
            metric="accuracy",
            feature_names=names,
        )
        assert report.metric_used == "accuracy"

    def test_unknown_model(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        with pytest.raises(ValueError, match="Unknown model"):
            obs.train_single(X, y, "xgboost_magic", feature_names=names)


class TestObservatoryReport:
    def test_report_fields(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, models=["random_forest", "logistic_regression"], feature_names=names)
        assert isinstance(report.results, list)
        assert isinstance(report.best_model, str)
        assert isinstance(report.best_score, float)
        assert isinstance(report.metric_used, str)
        assert isinstance(report.feature_rankings, list)

    def test_feature_rankings_sorted(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, models=["random_forest", "logistic_regression"], feature_names=names)
        rankings = report.feature_rankings
        # Should be sorted descending by importance
        for i in range(len(rankings) - 1):
            assert rankings[i][1] >= rankings[i + 1][1]

    def test_results_count(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, models=["random_forest", "ridge", "svm"], feature_names=names)
        assert len(report.results) == 3

    def test_best_score_range(self):
        X, y, names = _make_dataset()
        obs = ModelObservatory()
        report = obs.compare_models(X, y, feature_names=names)
        assert 0.0 <= report.best_score <= 1.0
