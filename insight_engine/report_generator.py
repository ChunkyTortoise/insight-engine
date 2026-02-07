"""PDF Report Generator: branded reports with charts and insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go

from insight_engine.profiler import DataProfile


@dataclass
class ReportSection:
    title: str
    content: str
    chart: go.Figure | None = None


@dataclass
class Report:
    title: str
    generated_at: str
    sections: list[ReportSection] = field(default_factory=list)
    profile: DataProfile | None = None


def _profile_to_markdown(profile: DataProfile) -> str:
    """Convert DataProfile to markdown summary."""
    lines = [
        f"**Rows**: {profile.row_count:,} | **Columns**: {profile.column_count} | "
        f"**Duplicates**: {profile.duplicate_rows:,} | **Memory**: {profile.memory_usage_mb:.1f} MB",
        "",
        "| Column | Type | Non-Null | Null% | Unique | Outliers |",
        "|--------|------|----------|-------|--------|----------|",
    ]
    for col in profile.columns:
        lines.append(
            f"| {col.name} | {col.dtype} | {col.non_null_count:,} | {col.null_pct}% | "
            f"{col.unique_count:,} | {col.outlier_count} |"
        )

    # Numeric summary
    numeric_cols = [c for c in profile.columns if c.mean is not None]
    if numeric_cols:
        lines.extend(
            [
                "",
                "### Numeric Summary",
                "| Column | Mean | Median | Std | Min | Max | Skewness |",
                "|--------|------|--------|-----|-----|-----|----------|",
            ]
        )
        for col in numeric_cols:
            lines.append(
                f"| {col.name} | {col.mean} | {col.median} | {col.std} | "
                f"{col.min_val} | {col.max_val} | {col.skewness} |"
            )

    return "\n".join(lines)


def generate_report(
    profile: DataProfile,
    charts: list[go.Figure] | None = None,
    title: str = "Data Analysis Report",
    additional_sections: list[ReportSection] | None = None,
) -> Report:
    """Generate a structured report from profile and charts."""
    report = Report(
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        profile=profile,
    )

    # Data overview section
    report.sections.append(
        ReportSection(
            title="Data Overview",
            content=_profile_to_markdown(profile),
        )
    )

    # Key findings
    findings = []
    high_null_cols = [c for c in profile.columns if c.null_pct > 20]
    if high_null_cols:
        findings.append(f"**High missing values**: {', '.join(c.name for c in high_null_cols)} (>{20}% null)")

    high_outlier_cols = [c for c in profile.columns if c.outlier_count > 0]
    if high_outlier_cols:
        total_outliers = sum(c.outlier_count for c in high_outlier_cols)
        findings.append(f"**Outliers detected**: {total_outliers} outliers across {len(high_outlier_cols)} columns")

    if profile.duplicate_rows > 0:
        findings.append(f"**Duplicate rows**: {profile.duplicate_rows:,} duplicate rows found")

    skewed_cols = [c for c in profile.columns if c.skewness is not None and abs(c.skewness) > 1]
    if skewed_cols:
        findings.append(f"**Skewed distributions**: {', '.join(c.name for c in skewed_cols)} (|skewness| > 1)")

    if findings:
        report.sections.append(ReportSection(title="Key Findings", content="\n".join(f"- {f}" for f in findings)))

    # Charts
    if charts:
        for i, chart in enumerate(charts):
            report.sections.append(
                ReportSection(
                    title=chart.layout.title.text if chart.layout.title else f"Chart {i + 1}",
                    content="",
                    chart=chart,
                )
            )

    if additional_sections:
        report.sections.extend(additional_sections)

    return report


def report_to_markdown(report: Report) -> str:
    """Convert report to markdown string."""
    lines = [
        f"# {report.title}",
        f"*Generated: {report.generated_at}*",
        "",
    ]

    for section in report.sections:
        lines.append(f"## {section.title}")
        lines.append(section.content)
        if section.chart:
            lines.append("*[Chart embedded in PDF/HTML version]*")
        lines.append("")

    return "\n".join(lines)


def export_charts_as_images(charts: list[go.Figure], output_dir: Path) -> list[Path]:
    """Export Plotly charts as PNG images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, chart in enumerate(charts):
        path = output_dir / f"chart_{i + 1}.png"
        try:
            chart.write_image(str(path))
            paths.append(path)
        except Exception:
            # kaleido not installed — skip image export
            pass
    return paths
