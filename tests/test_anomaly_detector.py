"""Tests for the anomaly detector module."""

import math

from insight_engine.anomaly_detector import Anomaly, AnomalyDetector


class TestZscoreDetection:
    def test_finds_known_outlier(self):
        """A value far from the mean should be flagged."""
        data = [10, 11, 10, 12, 10, 11, 10, 100]
        detector = AnomalyDetector(z_threshold=2.0)
        anomalies = detector.detect_zscore(data, column="price")
        assert len(anomalies) >= 1
        outlier_indices = {a.row_index for a in anomalies}
        assert 7 in outlier_indices  # 100 is the outlier
        assert anomalies[0].method == "zscore"
        assert anomalies[0].direction == "high"
        assert anomalies[0].column == "price"

    def test_no_anomalies_in_tight_data(self):
        """Normally distributed data within threshold should produce no anomalies."""
        data = [50, 51, 49, 50, 52, 48, 50, 51]
        detector = AnomalyDetector(z_threshold=3.0)
        anomalies = detector.detect_zscore(data)
        assert len(anomalies) == 0

    def test_low_direction_outlier(self):
        """A value far below the mean should be flagged as 'low'."""
        data = [100, 101, 99, 100, 102, 100, 0]
        detector = AnomalyDetector(z_threshold=2.0)
        anomalies = detector.detect_zscore(data)
        low_hits = [a for a in anomalies if a.direction == "low"]
        assert len(low_hits) >= 1
        assert low_hits[0].value == 0


class TestIqrDetection:
    def test_finds_known_outlier(self):
        """A value far outside the IQR fences should be flagged."""
        data = list(range(1, 21)) + [200]
        detector = AnomalyDetector(iqr_multiplier=1.5)
        anomalies = detector.detect_iqr(data, column="sales")
        assert len(anomalies) >= 1
        outlier_values = {a.value for a in anomalies}
        assert 200 in outlier_values
        assert anomalies[-1].method == "iqr"
        assert anomalies[-1].direction == "high"

    def test_low_outlier_flagged(self):
        """A value far below Q1 should be flagged as 'low'."""
        data = list(range(50, 101)) + [-100]
        detector = AnomalyDetector(iqr_multiplier=1.5)
        anomalies = detector.detect_iqr(data)
        low_hits = [a for a in anomalies if a.direction == "low"]
        assert len(low_hits) >= 1
        assert low_hits[0].value == -100


class TestEdgeCases:
    def test_empty_data_returns_empty(self):
        detector = AnomalyDetector()
        assert detector.detect_zscore([]) == []
        assert detector.detect_iqr([]) == []

    def test_single_value_returns_empty(self):
        detector = AnomalyDetector()
        assert detector.detect_zscore([42.0]) == []
        assert detector.detect_iqr([42.0]) == []

    def test_constant_values_returns_empty(self):
        """When all values are identical, std=0 and IQR=0 -- no anomalies possible."""
        detector = AnomalyDetector()
        data = [5.0] * 100
        assert detector.detect_zscore(data) == []
        assert detector.detect_iqr(data) == []

    def test_nan_values_skipped(self):
        """NaN values should be silently skipped, not cause errors."""
        detector = AnomalyDetector(z_threshold=2.0)
        data = [10, 11, float("nan"), 10, 12, float("inf"), 10, 100]
        anomalies = detector.detect_zscore(data)
        # Should not crash, and NaN/inf should not appear as anomalies
        for a in anomalies:
            assert math.isfinite(a.value)


class TestDetectAll:
    def test_multiple_columns(self):
        """detect_all should process each column independently."""
        detector = AnomalyDetector(z_threshold=2.0, iqr_multiplier=1.5)
        data = {
            "revenue": [10, 11, 10, 12, 10, 11, 10, 200],
            "cost": [5, 5, 5, 5, 5, 5, 5, 5],
        }
        results = detector.detect_all(data)
        assert "revenue" in results
        assert "cost" in results
        assert len(results["revenue"]) >= 1  # 200 is an outlier
        assert len(results["cost"]) == 0  # all identical, no outliers

    def test_deduplicates_by_row_index(self):
        """If both methods flag the same row, only one entry (highest score) kept."""
        detector = AnomalyDetector(z_threshold=2.0, iqr_multiplier=1.5)
        data = {"val": list(range(1, 51)) + [500]}
        results = detector.detect_all(data)
        row_indices = [a.row_index for a in results["val"]]
        # No duplicate row indices
        assert len(row_indices) == len(set(row_indices))


class TestSummary:
    def test_summary_statistics(self):
        detector = AnomalyDetector(z_threshold=2.0, iqr_multiplier=1.5)
        data = {
            "revenue": [10, 11, 10, 12, 10, 11, 10, 200],
            "cost": [5, 5, 5, 5, 5, 5, 5, 5],
        }
        results = detector.detect_all(data)
        s = detector.summary(results)
        assert s["total_anomalies"] >= 1
        assert s["columns_affected"] >= 1
        assert "revenue" in s["per_column"]
        assert "cost" not in s["per_column"]  # no anomalies in constant data
        assert s["most_anomalous"] is not None
        assert isinstance(s["most_anomalous"], Anomaly)

    def test_empty_summary(self):
        detector = AnomalyDetector()
        s = detector.summary({})
        assert s["total_anomalies"] == 0
        assert s["columns_affected"] == 0
        assert s["most_anomalous"] is None
