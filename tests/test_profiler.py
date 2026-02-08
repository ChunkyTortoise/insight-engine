"""Tests for the auto-profiler module."""

import numpy as np
import pandas as pd
import pytest

from insight_engine.profiler import (
    ColumnQuality,
    DataProfile,
    DataQualityScore,
    detect_column_type,
    profile_column,
    profile_dataframe,
    score_column_quality,
    score_data_quality,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": range(100),
            "name": [f"Item {i}" for i in range(100)],
            "category": np.random.choice(["A", "B", "C"], 100),
            "price": np.random.uniform(10, 100, 100).round(2),
            "quantity": np.random.randint(1, 50, 100),
            "rating": np.random.choice([1, 2, 3, 4, 5], 100),
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
    )


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

    def test_quality_included_in_profile(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert profile.quality is not None
        assert isinstance(profile.quality, DataQualityScore)
        assert 0 <= profile.quality.overall_score <= 100


class TestScoreColumnQuality:
    def test_complete_column(self):
        s = pd.Series([1, 2, 3, 4, 5], name="nums")
        q = score_column_quality(s, "integer")
        assert q.completeness == 100.0
        assert q.name == "nums"

    def test_column_with_nulls(self):
        s = pd.Series([1, 2, None, 4, None], name="partial")
        q = score_column_quality(s, "integer")
        assert q.completeness == 60.0

    def test_all_unique(self):
        s = pd.Series(["a", "b", "c", "d"], name="uniq")
        q = score_column_quality(s, "string")
        assert q.uniqueness == 100.0

    def test_low_uniqueness(self):
        s = pd.Series(["a", "a", "a", "a"], name="dups")
        q = score_column_quality(s, "string")
        assert q.uniqueness == 25.0

    def test_validity_numeric(self):
        s = pd.Series([1, 2, 3], name="valid_nums")
        q = score_column_quality(s, "integer")
        assert q.validity == 100.0

    def test_all_null_column(self):
        s = pd.Series([None, None], name="empty")
        q = score_column_quality(s, "string")
        assert q.completeness == 0.0
        assert q.validity == 100.0  # no non-nulls to be invalid


class TestScoreDataQuality:
    def test_perfect_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        score = score_data_quality(df)
        assert score.completeness == 100.0
        assert score.validity == 100.0
        assert score.overall_score > 90

    def test_data_with_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        score = score_data_quality(df)
        assert score.completeness < 100.0
        assert score.overall_score < 100.0

    def test_column_scores_populated(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        score = score_data_quality(df)
        assert len(score.column_scores) == 2
        assert all(isinstance(c, ColumnQuality) for c in score.column_scores)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        score = score_data_quality(df)
        assert score.overall_score == 0.0

    def test_weighted_scoring(self):
        """Overall score uses 50% completeness + 30% validity + 20% uniqueness."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        score = score_data_quality(df)
        expected = round(
            score.completeness * 0.5 + score.validity * 0.3 + score.uniqueness * 0.2,
            2,
        )
        assert score.overall_score == expected
