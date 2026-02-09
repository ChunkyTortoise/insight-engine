"""Data quality scoring: completeness, uniqueness, consistency, validity."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnQuality:
    """Quality metrics for a single column."""

    name: str
    completeness: float
    uniqueness: float
    consistency: float
    validity: float
    overall: float


@dataclass
class DataQualityReport:
    """Quality report for an entire DataFrame."""

    columns: list[ColumnQuality] = field(default_factory=list)
    overall_score: float = 0.0
    row_count: int = 0
    issues: list[str] = field(default_factory=list)


class DataQualityScorer:
    """Score data quality across completeness, uniqueness, consistency, and validity.

    Provides per-column and overall DataFrame quality metrics on a 0-1 scale.
    """

    def _completeness(self, series: pd.Series) -> float:
        """Fraction of non-null values: 1 - (null_count / total)."""
        if len(series) == 0:
            return 1.0
        null_count = int(series.isna().sum())
        return round(1.0 - (null_count / len(series)), 6)

    def _uniqueness(self, series: pd.Series) -> float:
        """Fraction of unique values: unique_count / total."""
        if len(series) == 0:
            return 1.0
        non_null = series.dropna()
        if len(non_null) == 0:
            return 1.0
        return round(non_null.nunique() / len(non_null), 6)

    def _consistency(self, series: pd.Series) -> float:
        """Dominant type ratio: fraction of values matching the most common type."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return 1.0
        type_counts: dict[str, int] = {}
        for val in non_null:
            type_name = type(val).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        dominant = max(type_counts.values())
        return round(dominant / len(non_null), 6)

    def _validity(self, series: pd.Series, rules: list | None = None) -> float:
        """Fraction of values matching validation rules.

        If no rules provided, returns 1.0 (all valid by default).
        Each rule is a callable: value -> bool.
        """
        if rules is None or len(rules) == 0:
            return 1.0
        non_null = series.dropna()
        if len(non_null) == 0:
            return 1.0
        valid_count = 0
        for val in non_null:
            if all(rule(val) for rule in rules):
                valid_count += 1
        return round(valid_count / len(non_null), 6)

    def score_column(
        self,
        series: pd.Series,
        rules: list | None = None,
        weights: tuple[float, float, float, float] = (0.3, 0.2, 0.3, 0.2),
    ) -> ColumnQuality:
        """Score a single column on all quality dimensions.

        Args:
            series: The column data.
            rules: Optional validation rules (list of callables).
            weights: Weights for (completeness, uniqueness, consistency, validity).

        Returns:
            ColumnQuality with individual and overall scores.
        """
        completeness = self._completeness(series)
        uniqueness = self._uniqueness(series)
        consistency = self._consistency(series)
        validity = self._validity(series, rules)

        w_c, w_u, w_cn, w_v = weights
        total_weight = w_c + w_u + w_cn + w_v
        overall = round(
            (completeness * w_c + uniqueness * w_u + consistency * w_cn + validity * w_v) / total_weight,
            6,
        )

        return ColumnQuality(
            name=str(series.name) if series.name is not None else "unnamed",
            completeness=completeness,
            uniqueness=uniqueness,
            consistency=consistency,
            validity=validity,
            overall=overall,
        )

    def score_dataframe(
        self,
        df: pd.DataFrame,
        rules: dict[str, list] | None = None,
    ) -> DataQualityReport:
        """Score an entire DataFrame.

        Args:
            df: The DataFrame to score.
            rules: Optional dict mapping column names to validation rules.

        Returns:
            DataQualityReport with per-column breakdown and overall score.
        """
        if df.empty:
            return DataQualityReport(
                columns=[],
                overall_score=1.0,
                row_count=0,
                issues=["DataFrame is empty"],
            )

        rules = rules or {}
        column_scores: list[ColumnQuality] = []
        issues: list[str] = []

        for col in df.columns:
            col_rules = rules.get(str(col))
            cq = self.score_column(df[col], rules=col_rules)
            column_scores.append(cq)

            if cq.completeness < 0.9:
                issues.append(f"Column '{cq.name}' has low completeness: {cq.completeness:.2%}")
            if cq.uniqueness < 0.5:
                issues.append(f"Column '{cq.name}' has low uniqueness: {cq.uniqueness:.2%}")
            if cq.consistency < 0.9:
                issues.append(f"Column '{cq.name}' has low consistency: {cq.consistency:.2%}")

        overall = round(sum(cq.overall for cq in column_scores) / len(column_scores), 6)

        return DataQualityReport(
            columns=column_scores,
            overall_score=overall,
            row_count=len(df),
            issues=issues,
        )
