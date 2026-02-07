"""Dashboard Generator: auto-selects chart types based on column profiles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insight_engine.profiler import DataProfile


@dataclass
class ChartSpec:
    title: str
    chart_type: str
    fig: go.Figure


def _select_chart_type(col_type: str, unique_count: int, row_count: int) -> str:
    """Select appropriate chart type based on column characteristics."""
    if col_type in ("categorical", "categorical_numeric", "boolean"):
        if unique_count <= 8:
            return "pie"
        return "bar"
    if col_type in ("integer", "float"):
        return "histogram"
    if col_type in ("datetime", "datetime_string"):
        return "line"
    return "bar"


def generate_distribution_chart(df: pd.DataFrame, column: str, col_type: str) -> ChartSpec:
    """Generate a distribution chart for a single column."""
    series = df[column].dropna()

    if col_type in ("categorical", "categorical_numeric", "boolean"):
        counts = series.value_counts().head(15)
        if len(counts) <= 8:
            fig = px.pie(values=counts.values, names=counts.index.astype(str), title=f"{column} Distribution")
        else:
            fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"{column} Distribution")
            fig.update_layout(xaxis_title=column, yaxis_title="Count")
        return ChartSpec(title=f"{column} Distribution", chart_type="categorical", fig=fig)

    if col_type in ("integer", "float"):
        fig = px.histogram(df, x=column, title=f"{column} Distribution", nbins=min(50, series.nunique()))
        fig.update_layout(xaxis_title=column, yaxis_title="Frequency")
        return ChartSpec(title=f"{column} Distribution", chart_type="histogram", fig=fig)

    if col_type in ("datetime", "datetime_string"):
        try:
            dt_series = pd.to_datetime(series, format="mixed")
            counts = dt_series.dt.date.value_counts().sort_index()
            fig = px.line(x=counts.index, y=counts.values, title=f"{column} Over Time")
            fig.update_layout(xaxis_title="Date", yaxis_title="Count")
            return ChartSpec(title=f"{column} Over Time", chart_type="line", fig=fig)
        except (ValueError, TypeError):
            pass

    # Fallback: bar chart of top values
    counts = series.value_counts().head(15)
    fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"{column} Top Values")
    return ChartSpec(title=f"{column} Top Values", chart_type="bar", fig=fig)


def generate_correlation_heatmap(profile: DataProfile) -> ChartSpec | None:
    """Generate correlation heatmap from profile."""
    if profile.correlation_matrix is None:
        return None

    corr = profile.correlation_matrix
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlations",
    )
    fig.update_layout(width=600, height=500)
    return ChartSpec(title="Feature Correlations", chart_type="heatmap", fig=fig)


def generate_scatter_matrix(df: pd.DataFrame, numeric_cols: list[str], max_cols: int = 5) -> ChartSpec | None:
    """Generate scatter matrix for numeric columns."""
    if len(numeric_cols) < 2:
        return None

    cols = numeric_cols[:max_cols]
    fig = px.scatter_matrix(df[cols], title="Numeric Feature Relationships")
    fig.update_layout(width=800, height=700)
    return ChartSpec(title="Scatter Matrix", chart_type="scatter_matrix", fig=fig)


def generate_dashboard(df: pd.DataFrame, profile: DataProfile) -> list[ChartSpec]:
    """Generate a complete dashboard from a DataFrame and its profile."""
    charts: list[ChartSpec] = []

    # Distribution charts for each column (skip high-cardinality strings)
    for col_profile in profile.columns:
        if col_profile.dtype in ("text", "string") and col_profile.unique_count > 50:
            continue
        chart = generate_distribution_chart(df, col_profile.name, col_profile.dtype)
        charts.append(chart)

    # Correlation heatmap
    corr_chart = generate_correlation_heatmap(profile)
    if corr_chart:
        charts.append(corr_chart)

    # Scatter matrix for numeric columns
    numeric_cols = [c.name for c in profile.columns if c.dtype in ("integer", "float")]
    scatter = generate_scatter_matrix(df, numeric_cols)
    if scatter:
        charts.append(scatter)

    return charts
