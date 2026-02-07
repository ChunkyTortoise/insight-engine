"""Auto-Profiler: column-type detection, distributions, outliers, correlations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_val: Any = None
    max_val: Any = None
    q25: float | None = None
    q75: float | None = None
    skewness: float | None = None
    outlier_count: int = 0


@dataclass
class DataProfile:
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    correlation_matrix: pd.DataFrame | None = None
    duplicate_rows: int = 0
    memory_usage_mb: float = 0.0


def detect_column_type(series: pd.Series) -> str:
    """Detect semantic column type beyond pandas dtype."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if series.dropna().apply(lambda x: x == int(x) if pd.notna(x) else True).all():
            unique_ratio = series.nunique() / max(len(series), 1)
            if unique_ratio < 0.05 and series.nunique() < 20:
                return "categorical_numeric"
            return "integer"
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Try parsing as datetime
    sample = series.dropna().head(100)
    if len(sample) > 0:
        try:
            pd.to_datetime(sample, format="mixed")
            return "datetime_string"
        except (ValueError, TypeError):
            pass

    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio < 0.05 and series.nunique() < 50:
        return "categorical"
    if series.str.len().mean() > 100 if hasattr(series, "str") else False:
        return "text"
    return "string"


def _count_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def profile_column(series: pd.Series) -> ColumnProfile:
    """Profile a single column."""
    col_type = detect_column_type(series)
    non_null = series.notna().sum()
    null_count = series.isna().sum()
    total = len(series)

    profile = ColumnProfile(
        name=series.name,
        dtype=col_type,
        non_null_count=int(non_null),
        null_count=int(null_count),
        null_pct=round(null_count / max(total, 1) * 100, 2),
        unique_count=int(series.nunique()),
    )

    # Top values for categorical/string columns
    if col_type in ("categorical", "categorical_numeric", "string", "boolean"):
        vc = series.value_counts().head(10)
        profile.top_values = list(zip(vc.index.tolist(), vc.values.tolist()))

    # Numeric statistics (exclude boolean — numpy can't compute quantile on bools)
    if col_type in ("integer", "float", "categorical_numeric"):
        numeric = series.dropna()
        if len(numeric) > 0:
            profile.mean = round(float(numeric.mean()), 4)
            profile.median = round(float(numeric.median()), 4)
            profile.std = round(float(numeric.std()), 4)
            profile.min_val = float(numeric.min())
            profile.max_val = float(numeric.max())
            profile.q25 = round(float(numeric.quantile(0.25)), 4)
            profile.q75 = round(float(numeric.quantile(0.75)), 4)
            profile.skewness = round(float(numeric.skew()), 4)
            profile.outlier_count = _count_outliers_iqr(numeric)

    return profile


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Profile an entire DataFrame."""
    columns = [profile_column(df[col]) for col in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr() if len(numeric_cols.columns) > 1 else None

    return DataProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        correlation_matrix=corr,
        duplicate_rows=int(df.duplicated().sum()),
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    )
