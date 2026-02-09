"""Tests for data quality scoring."""

from __future__ import annotations

import pandas as pd

from insight_engine.data_quality import ColumnQuality, DataQualityReport, DataQualityScorer


class TestCompleteness:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_perfect(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert self.scorer._completeness(s) == 1.0

    def test_with_nulls(self):
        s = pd.Series([1, None, 3, None, 5])
        score = self.scorer._completeness(s)
        assert score == 0.6

    def test_all_null(self):
        s = pd.Series([None, None, None])
        assert self.scorer._completeness(s) == 0.0

    def test_empty(self):
        s = pd.Series([], dtype=float)
        assert self.scorer._completeness(s) == 1.0


class TestUniqueness:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_all_unique(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert self.scorer._uniqueness(s) == 1.0

    def test_all_duplicates(self):
        s = pd.Series([1, 1, 1, 1])
        assert self.scorer._uniqueness(s) == 0.25

    def test_mixed(self):
        s = pd.Series([1, 2, 2, 3])
        score = self.scorer._uniqueness(s)
        assert score == 0.75

    def test_with_nulls_ignored(self):
        s = pd.Series([1, 2, None, 2])
        score = self.scorer._uniqueness(s)
        # 2 unique out of 3 non-null
        assert abs(score - 2 / 3) < 0.01


class TestConsistency:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_all_same_type(self):
        s = pd.Series([1, 2, 3, 4])
        assert self.scorer._consistency(s) == 1.0

    def test_mixed_types(self):
        s = pd.Series([1, "two", 3, "four"])
        score = self.scorer._consistency(s)
        assert score == 0.5

    def test_all_null(self):
        s = pd.Series([None, None])
        assert self.scorer._consistency(s) == 1.0


class TestValidity:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_no_rules(self):
        s = pd.Series([1, 2, 3])
        assert self.scorer._validity(s) == 1.0

    def test_with_rules(self):
        s = pd.Series([10, 20, -5, 30])
        rules = [lambda x: x > 0]
        score = self.scorer._validity(s, rules)
        assert score == 0.75

    def test_all_pass(self):
        s = pd.Series([2, 4, 6])
        rules = [lambda x: x % 2 == 0]
        assert self.scorer._validity(s, rules) == 1.0


class TestScoreColumn:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_perfect_data(self):
        s = pd.Series([1, 2, 3, 4, 5], name="test_col")
        cq = self.scorer.score_column(s)
        assert isinstance(cq, ColumnQuality)
        assert cq.completeness == 1.0
        assert cq.uniqueness == 1.0
        assert cq.consistency == 1.0
        assert cq.overall == 1.0

    def test_imperfect_data(self):
        s = pd.Series([1, None, 1, None, 1], name="imperfect")
        cq = self.scorer.score_column(s)
        assert cq.completeness < 1.0
        assert cq.uniqueness < 1.0
        assert cq.overall < 1.0

    def test_column_name(self):
        s = pd.Series([1], name="my_col")
        assert self.scorer.score_column(s).name == "my_col"


class TestScoreDataframe:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_perfect_df(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = self.scorer.score_dataframe(df)
        assert isinstance(report, DataQualityReport)
        assert report.overall_score == 1.0
        assert report.row_count == 3
        assert len(report.columns) == 2

    def test_df_with_issues(self):
        df = pd.DataFrame(
            {
                "a": [1, None, None, None, None],
                "b": [1, 1, 1, 1, 1],
            }
        )
        report = self.scorer.score_dataframe(df)
        assert report.overall_score < 1.0
        assert len(report.issues) > 0

    def test_empty_df(self):
        df = pd.DataFrame()
        report = self.scorer.score_dataframe(df)
        assert report.row_count == 0
        assert "empty" in report.issues[0].lower()

    def test_single_column(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        report = self.scorer.score_dataframe(df)
        assert len(report.columns) == 1

    def test_with_rules(self):
        df = pd.DataFrame({"age": [25, -1, 30, 200]})
        rules = {"age": [lambda x: 0 < x < 150]}
        report = self.scorer.score_dataframe(df, rules=rules)
        age_col = report.columns[0]
        assert age_col.validity < 1.0

    def test_overall_weighted(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        report = self.scorer.score_dataframe(df)
        # Overall should be average of column overalls
        expected = sum(c.overall for c in report.columns) / len(report.columns)
        assert abs(report.overall_score - expected) < 0.01
