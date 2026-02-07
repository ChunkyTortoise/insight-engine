"""Tests for the data cleaning module."""

import pandas as pd
import pytest

from insight_engine.cleaner import (
    clean_dataframe,
    impute_missing,
    remove_duplicates,
    standardize_columns,
)


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "First Name": ["  Alice  ", "Bob", "Charlie", "Alice", "Bob", None],
        "Last-Name": ["Smith", "Jones", "Brown", "Smith", "Jones", "Wilson"],
        "Age": [25, 30, None, 25, 30, 45],
        "Salary": [50000, 60000, 70000, 50000, 60000, None],
    })


class TestRemoveDuplicates:
    def test_exact_duplicates(self, messy_df):
        cleaned, count = remove_duplicates(messy_df)
        assert count == 1  # Alice Smith + Bob Jones duplicated
        assert len(cleaned) == 5

    def test_subset_duplicates(self, messy_df):
        cleaned, count = remove_duplicates(messy_df, subset=["First Name"])
        # "  Alice  " != "Alice" (whitespace), but "Bob" appears twice → 1 removed
        assert count == 1

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned, count = remove_duplicates(df)
        assert count == 0
        assert len(cleaned) == 3

    def test_fuzzy_dedup(self):
        df = pd.DataFrame({
            "name": ["John Smith", "john  smith", "Jane Doe"],
        })
        cleaned, count = remove_duplicates(df, subset=["name"], fuzzy=True)
        assert count == 1
        assert len(cleaned) == 2


class TestStandardizeColumns:
    def test_column_names(self, messy_df):
        result, ops = standardize_columns(messy_df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns
        assert len(ops) > 0

    def test_whitespace_stripping(self, messy_df):
        result, _ = standardize_columns(messy_df)
        assert result["first_name"].iloc[0] == "Alice"


class TestImputeMissing:
    def test_smart_imputation(self, messy_df):
        result, count = impute_missing(messy_df, strategy="smart")
        assert count > 0
        assert result["Age"].isna().sum() == 0
        assert result["Salary"].isna().sum() == 0

    def test_mean_imputation(self):
        df = pd.DataFrame({"x": [1, 2, None, 4]})
        result, count = impute_missing(df, strategy="mean")
        assert count == 1
        assert abs(result["x"].iloc[2] - 2.333) < 0.01

    def test_drop_strategy(self, messy_df):
        result, count = impute_missing(messy_df, strategy="drop")
        assert result.isna().sum().sum() == 0
        assert count > 0

    def test_no_missing(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result, count = impute_missing(df)
        assert count == 0


class TestCleanDataframe:
    def test_full_pipeline(self, messy_df):
        cleaned, report = clean_dataframe(messy_df)
        assert report.original_rows == 6
        assert report.duplicates_removed >= 1
        assert report.nulls_imputed > 0
        assert len(report.operations) > 0

    def test_skip_all(self, messy_df):
        cleaned, report = clean_dataframe(
            messy_df, dedup=False, standardize=False, impute=False
        )
        assert len(cleaned) == len(messy_df)

    def test_report_fields(self, messy_df):
        _, report = clean_dataframe(messy_df)
        assert report.cleaned_rows <= report.original_rows
        assert isinstance(report.columns_standardized, list)
