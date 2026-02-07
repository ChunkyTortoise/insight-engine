"""Tests for the predictive modeling module."""

import numpy as np
import pandas as pd
import pytest

from insight_engine.predictor import PredictionResult, detect_task_type, train_model


@pytest.fixture
def classification_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "feature_1": np.random.uniform(0, 10, n),
            "feature_2": np.random.uniform(0, 10, n),
            "category": np.random.choice(["A", "B"], n),
            "target": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def regression_df():
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 10, n)
    return pd.DataFrame(
        {
            "feature_1": x1,
            "feature_2": x2,
            "noise": np.random.normal(0, 1, n),
            "target": 3 * x1 + 2 * x2 + np.random.normal(0, 2, n),
        }
    )


class TestDetectTaskType:
    def test_binary_classification(self):
        s = pd.Series([0, 1, 0, 1, 0])
        assert detect_task_type(s) == "classification"

    def test_multi_class(self):
        s = pd.Series(["A", "B", "C", "A", "B"])
        assert detect_task_type(s) == "classification"

    def test_regression(self):
        s = pd.Series(np.random.uniform(0, 100, 100))
        assert detect_task_type(s) == "regression"

    def test_few_unique_numeric(self):
        s = pd.Series(np.random.choice([1, 2, 3, 4, 5], 1000))
        assert detect_task_type(s) == "classification"


class TestTrainModel:
    def test_classification(self, classification_df):
        result = train_model(classification_df, "target")
        assert isinstance(result, PredictionResult)
        assert result.task_type == "classification"
        assert "accuracy" in result.metrics
        assert "f1_weighted" in result.metrics
        assert len(result.feature_importances) > 0

    def test_regression(self, regression_df):
        result = train_model(regression_df, "target")
        assert result.task_type == "regression"
        assert "r2" in result.metrics
        assert "mae" in result.metrics
        assert "rmse" in result.metrics

    def test_string_target(self):
        df = pd.DataFrame(
            {
                "x": np.random.uniform(0, 10, 100),
                "label": np.random.choice(["good", "bad"], 100),
            }
        )
        result = train_model(df, "label")
        assert result.task_type == "classification"
        assert result.metrics["accuracy"] >= 0

    def test_missing_target_column(self, classification_df):
        with pytest.raises(ValueError, match="not found"):
            train_model(classification_df, "nonexistent")

    def test_handles_missing_values(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [None, "a", "b", "a", "b", "a", "b", "a", "b", "a"] * 10,
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
            }
        )
        result = train_model(df, "target")
        assert result.task_type == "classification"

    def test_feature_names(self, classification_df):
        result = train_model(classification_df, "target")
        assert len(result.feature_names) == 3
        assert "target" not in result.feature_names
