"""Anomaly Detection: Z-score and IQR-based outlier detection for numeric data."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Anomaly:
    """A single detected anomaly."""

    column: str
    row_index: int
    value: float
    method: str  # "iqr" or "zscore"
    score: float  # how anomalous (z-score magnitude or IQR distance ratio)
    direction: str  # "high" or "low"


class AnomalyDetector:
    """Detect outliers in numeric data using Z-score and IQR methods.

    Args:
        z_threshold: Number of standard deviations for Z-score flagging (default 3.0).
        iqr_multiplier: IQR fence multiplier (default 1.5).
    """

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_zscore(self, data: list[float], column: str = "value") -> list[Anomaly]:
        """Detect outliers using Z-score method.

        Values beyond z_threshold standard deviations from the mean are flagged.
        Returns empty list for empty data, single-element lists, or constant values.
        NaN/inf values are silently skipped.
        """
        clean = [x for x in data if _is_finite(x)]
        if len(clean) < 2:
            return []

        mean = sum(clean) / len(clean)
        variance = sum((x - mean) ** 2 for x in clean) / len(clean)
        std = math.sqrt(variance)

        if std == 0:
            return []

        anomalies: list[Anomaly] = []
        for i, raw in enumerate(data):
            if not _is_finite(raw):
                continue
            z = (raw - mean) / std
            if abs(z) > self.z_threshold:
                anomalies.append(
                    Anomaly(
                        column=column,
                        row_index=i,
                        value=raw,
                        method="zscore",
                        score=round(abs(z), 4),
                        direction="high" if z > 0 else "low",
                    )
                )
        return anomalies

    def detect_iqr(self, data: list[float], column: str = "value") -> list[Anomaly]:
        """Detect outliers using IQR method.

        Values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR are flagged.
        Returns empty list for empty data, single-element lists, or zero-IQR data.
        NaN/inf values are silently skipped.
        """
        clean = [x for x in data if _is_finite(x)]
        if len(clean) < 2:
            return []

        sorted_data = sorted(clean)
        q1 = _percentile(sorted_data, 25)
        q3 = _percentile(sorted_data, 75)
        iqr = q3 - q1

        if iqr == 0:
            return []

        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr

        anomalies: list[Anomaly] = []
        for i, raw in enumerate(data):
            if not _is_finite(raw):
                continue
            if raw < lower_fence:
                distance = (lower_fence - raw) / iqr
                anomalies.append(
                    Anomaly(
                        column=column,
                        row_index=i,
                        value=raw,
                        method="iqr",
                        score=round(distance, 4),
                        direction="low",
                    )
                )
            elif raw > upper_fence:
                distance = (raw - upper_fence) / iqr
                anomalies.append(
                    Anomaly(
                        column=column,
                        row_index=i,
                        value=raw,
                        method="iqr",
                        score=round(distance, 4),
                        direction="high",
                    )
                )
        return anomalies

    def detect_all(self, data: dict[str, list[float]]) -> dict[str, list[Anomaly]]:
        """Run both Z-score and IQR methods on all columns.

        Returns a dict mapping column names to combined anomaly lists (deduplicated
        by row index -- if both methods flag the same row, the higher score wins).
        """
        results: dict[str, list[Anomaly]] = {}
        for col, values in data.items():
            zscore_hits = self.detect_zscore(values, column=col)
            iqr_hits = self.detect_iqr(values, column=col)

            # Merge: keep highest score per row index
            best: dict[int, Anomaly] = {}
            for a in zscore_hits + iqr_hits:
                existing = best.get(a.row_index)
                if existing is None or a.score > existing.score:
                    best[a.row_index] = a

            results[col] = sorted(best.values(), key=lambda a: a.row_index)
        return results

    def summary(self, anomalies: dict[str, list[Anomaly]]) -> dict:
        """Return summary statistics for detected anomalies.

        Returns:
            dict with keys: total_anomalies, columns_affected, per_column (dict of counts),
            most_anomalous (the single highest-scored Anomaly or None).
        """
        total = 0
        per_column: dict[str, int] = {}
        most_anomalous: Anomaly | None = None

        for col, items in anomalies.items():
            count = len(items)
            total += count
            if count > 0:
                per_column[col] = count
                for a in items:
                    if most_anomalous is None or a.score > most_anomalous.score:
                        most_anomalous = a

        return {
            "total_anomalies": total,
            "columns_affected": len(per_column),
            "per_column": per_column,
            "most_anomalous": most_anomalous,
        }


def _is_finite(x: float) -> bool:
    """Check if a value is a finite number (not NaN, inf, or non-numeric)."""
    try:
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Calculate percentile using linear interpolation on pre-sorted data."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]

    k = (pct / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_data[int(k)]

    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
