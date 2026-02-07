"""Tests for the dashboard generator module."""

import numpy as np
import pandas as pd
import pytest

from insight_engine.dashboard_generator import (
    generate_correlation_heatmap,
    generate_dashboard,
    generate_distribution_chart,
    generate_scatter_matrix,
)
from insight_engine.profiler import profile_dataframe


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C", "D"], 200),
        "revenue": np.random.uniform(100, 10000, 200).round(2),
        "quantity": np.random.randint(1, 50, 200),
        "date": pd.date_range("2024-01-01", periods=200, freq="D"),
        "is_returned": np.random.choice([True, False], 200),
    })


class TestGenerateDistributionChart:
    def test_categorical(self, sample_df):
        chart = generate_distribution_chart(sample_df, "category", "categorical")
        assert chart.chart_type == "categorical"
        assert chart.fig is not None

    def test_numeric(self, sample_df):
        chart = generate_distribution_chart(sample_df, "revenue", "float")
        assert chart.chart_type == "histogram"
        assert chart.fig is not None

    def test_boolean(self, sample_df):
        chart = generate_distribution_chart(sample_df, "is_returned", "boolean")
        assert chart.chart_type == "categorical"

    def test_datetime(self, sample_df):
        chart = generate_distribution_chart(sample_df, "date", "datetime")
        assert chart.chart_type == "line"


class TestGenerateCorrelationHeatmap:
    def test_with_numeric_cols(self, sample_df):
        profile = profile_dataframe(sample_df)
        chart = generate_correlation_heatmap(profile)
        assert chart is not None
        assert chart.chart_type == "heatmap"

    def test_no_numeric_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["m", "n"]})
        profile = profile_dataframe(df)
        chart = generate_correlation_heatmap(profile)
        assert chart is None


class TestGenerateScatterMatrix:
    def test_enough_cols(self, sample_df):
        chart = generate_scatter_matrix(sample_df, ["revenue", "quantity"])
        assert chart is not None
        assert chart.chart_type == "scatter_matrix"

    def test_single_col(self, sample_df):
        chart = generate_scatter_matrix(sample_df, ["revenue"])
        assert chart is None


class TestGenerateDashboard:
    def test_full_dashboard(self, sample_df):
        profile = profile_dataframe(sample_df)
        charts = generate_dashboard(sample_df, profile)
        assert len(charts) > 0
        assert all(c.fig is not None for c in charts)

    def test_empty_df(self):
        df = pd.DataFrame({"a": [], "b": []})
        profile = profile_dataframe(df)
        charts = generate_dashboard(df, profile)
        assert isinstance(charts, list)
