"""Tests for the auto-profiler module."""

import numpy as np
import pandas as pd
import pytest

from insight_engine.profiler import (
    DataProfile,
    detect_column_type,
    profile_column,
    profile_dataframe,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(100),
        "name": [f"Item {i}" for i in range(100)],
        "category": np.random.choice(["A", "B", "C"], 100),
        "price": np.random.uniform(10, 100, 100).round(2),
        "quantity": np.random.randint(1, 50, 100),
        "rating": np.random.choice([1, 2, 3, 4, 5], 100),
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


class TestDetectColumnType:
    def test_integer(self):
        s = pd.Series(range(100))
        assert detect_column_type(s) == "integer"

    def test_float(self):
        s = pd.Series(np.random.uniform(0, 1, 100))
        assert detect_column_type(s) == "float"

    def test_categorical_string(self):
        s = pd.Series(np.random.choice(["A", "B", "C"], 1000))
        assert detect_column_type(s) == "categorical"

    def test_datetime(self):
        s = pd.Series(pd.date_range("2024-01-01", periods=10))
        assert detect_column_type(s) == "datetime"

    def test_boolean(self):
        s = pd.Series([True, False, True, False])
        assert detect_column_type(s) == "boolean"

    def test_categorical_numeric(self):
        s = pd.Series(np.random.choice([1, 2, 3], 1000))
        assert detect_column_type(s) == "categorical_numeric"


class TestProfileColumn:
    def test_numeric_column(self):
        s = pd.Series([1, 2, 3, 4, 5, None], name="test")
        profile = profile_column(s)
        assert profile.name == "test"
        assert profile.non_null_count == 5
        assert profile.null_count == 1
        assert profile.mean is not None
        assert profile.median is not None

    def test_string_column(self):
        s = pd.Series(["a", "b", "a", "c", None], name="cat")
        profile = profile_column(s)
        assert profile.null_count == 1
        assert profile.unique_count == 3

    def test_all_null(self):
        s = pd.Series([None, None, None], name="empty")
        profile = profile_column(s)
        assert profile.null_pct == 100.0


class TestProfileDataframe:
    def test_basic(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert isinstance(profile, DataProfile)
        assert profile.row_count == 100
        assert profile.column_count == 7
        assert len(profile.columns) == 7

    def test_correlation_matrix(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert profile.correlation_matrix is not None
        assert profile.correlation_matrix.shape[0] > 0

    def test_memory_usage(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert profile.memory_usage_mb > 0

    def test_duplicate_detection(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = profile_dataframe(df)
        assert profile.duplicate_rows == 1

    def test_outlier_detection(self):
        data = list(range(100)) + [1000]
        df = pd.DataFrame({"val": data})
        profile = profile_dataframe(df)
        assert profile.columns[0].outlier_count > 0
