"""Tests for the report generator module."""

import numpy as np
import pandas as pd
import pytest

from insight_engine.profiler import profile_dataframe
from insight_engine.report_generator import (
    ReportSection,
    generate_report,
    report_to_markdown,
)


@pytest.fixture
def sample_profile():
    df = pd.DataFrame({
        "revenue": np.random.uniform(100, 10000, 100).round(2),
        "quantity": np.random.randint(1, 50, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "notes": [None] * 30 + ["note"] * 70,
    })
    return profile_dataframe(df)


class TestGenerateReport:
    def test_basic(self, sample_profile):
        report = generate_report(sample_profile)
        assert report.title == "Data Analysis Report"
        assert len(report.sections) >= 1
        assert report.generated_at is not None

    def test_custom_title(self, sample_profile):
        report = generate_report(sample_profile, title="Custom Report")
        assert report.title == "Custom Report"

    def test_key_findings(self, sample_profile):
        report = generate_report(sample_profile)
        section_titles = [s.title for s in report.sections]
        assert "Data Overview" in section_titles

    def test_additional_sections(self, sample_profile):
        extra = [ReportSection(title="Custom", content="Custom content")]
        report = generate_report(sample_profile, additional_sections=extra)
        section_titles = [s.title for s in report.sections]
        assert "Custom" in section_titles


class TestReportToMarkdown:
    def test_contains_title(self, sample_profile):
        report = generate_report(sample_profile, title="Test Report")
        md = report_to_markdown(report)
        assert "# Test Report" in md

    def test_contains_sections(self, sample_profile):
        report = generate_report(sample_profile)
        md = report_to_markdown(report)
        assert "## Data Overview" in md
        assert "Column" in md
        assert "Type" in md

    def test_markdown_format(self, sample_profile):
        report = generate_report(sample_profile)
        md = report_to_markdown(report)
        assert md.startswith("#")
        assert "Generated:" in md
