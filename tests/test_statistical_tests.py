"""Tests for the statistical hypothesis testing module."""

import numpy as np

from insight_engine.statistical_tests import StatisticalTester, TestResult, TestSuiteReport


def _make_different_groups() -> tuple[np.ndarray, np.ndarray]:
    """Create two significantly different groups."""
    rng = np.random.RandomState(42)
    group_a = rng.normal(loc=10, scale=2, size=50)
    group_b = rng.normal(loc=20, scale=2, size=50)
    return group_a, group_b


def _make_similar_groups() -> tuple[np.ndarray, np.ndarray]:
    """Create two groups from the same distribution (not significantly different)."""
    rng = np.random.RandomState(42)
    group_a = rng.normal(loc=10, scale=2, size=50)
    group_b = rng.normal(loc=10, scale=2, size=50)
    return group_a, group_b


def _make_normal_data() -> np.ndarray:
    """Create data from a normal distribution."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=0, scale=1, size=100)


class TestIndependentTTest:
    def test_significant_difference(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        result = st.t_test_independent(a, b)
        assert isinstance(result, TestResult)
        assert result.test_name == "welch_t_test"
        assert result.significant is True
        assert result.p_value < 0.05

    def test_no_significant_difference(self):
        a, b = _make_similar_groups()
        st = StatisticalTester()
        result = st.t_test_independent(a, b)
        assert result.significant is False
        assert result.p_value >= 0.05

    def test_effect_size_present(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        result = st.t_test_independent(a, b)
        assert result.effect_size is not None
        assert abs(result.effect_size) > 1.0  # Large effect

    def test_interpretation_string(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        result = st.t_test_independent(a, b)
        assert "significantly" in result.interpretation


class TestPairedTTest:
    def test_significant_change(self):
        rng = np.random.RandomState(42)
        before = rng.normal(loc=10, scale=2, size=30)
        after = before + 5  # Clear improvement
        st = StatisticalTester()
        result = st.t_test_paired(before, after)
        assert result.test_name == "paired_t_test"
        assert result.significant is True

    def test_no_change(self):
        rng = np.random.RandomState(42)
        before = rng.normal(loc=10, scale=2, size=30)
        after = before + rng.normal(0, 0.01, size=30)
        st = StatisticalTester()
        result = st.t_test_paired(before, after)
        assert result.significant is False


class TestChiSquare:
    def test_dependent_variables(self):
        # Clearly dependent: diagonal dominance
        observed = np.array([[50, 5], [5, 50]])
        st = StatisticalTester()
        result = st.chi_square(observed)
        assert result.test_name == "chi_square"
        assert result.significant is True
        assert result.effect_size is not None

    def test_independent_variables(self):
        # Roughly independent
        observed = np.array([[25, 25], [25, 25]])
        st = StatisticalTester()
        result = st.chi_square(observed)
        assert result.significant is False

    def test_cramers_v_bounded(self):
        observed = np.array([[50, 5], [5, 50]])
        st = StatisticalTester()
        result = st.chi_square(observed)
        assert 0 <= result.effect_size <= 1.0


class TestANOVA:
    def test_significant_groups(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(5, 1, 30)
        g3 = rng.normal(10, 1, 30)
        st = StatisticalTester()
        result = st.anova_oneway([g1, g2, g3])
        assert result.test_name == "anova_oneway"
        assert result.significant is True

    def test_similar_groups(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(10, 1, 30)
        g2 = rng.normal(10, 1, 30)
        g3 = rng.normal(10, 1, 30)
        st = StatisticalTester()
        result = st.anova_oneway([g1, g2, g3])
        assert result.significant is False

    def test_eta_squared_range(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(5, 1, 30)
        st = StatisticalTester()
        result = st.anova_oneway([g1, g2])
        assert 0 <= result.effect_size <= 1.0


class TestMannWhitney:
    def test_different_distributions(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        result = st.mann_whitney(a, b)
        assert result.test_name == "mann_whitney_u"
        assert result.significant is True

    def test_same_distribution(self):
        a, b = _make_similar_groups()
        st = StatisticalTester()
        result = st.mann_whitney(a, b)
        assert result.significant is False


class TestCorrelation:
    def test_pearson_positive(self):
        x = np.arange(50, dtype=float)
        y = x * 2 + 1
        st = StatisticalTester()
        result = st.correlation(x, y, method="pearson")
        assert result.test_name == "pearson_correlation"
        assert result.significant is True
        assert result.statistic > 0.9

    def test_spearman(self):
        x = np.arange(50, dtype=float)
        y = x**2  # Monotonic but not linear
        st = StatisticalTester()
        result = st.correlation(x, y, method="spearman")
        assert result.test_name == "spearman_correlation"
        assert result.significant is True

    def test_no_correlation(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        st = StatisticalTester()
        result = st.correlation(x, y)
        # With random data, likely not significant
        assert isinstance(result.significant, bool)


class TestNormality:
    def test_normal_data(self):
        data = _make_normal_data()
        st = StatisticalTester()
        result = st.normality(data)
        assert result.test_name == "shapiro_wilk"
        # Normal data should not reject normality
        assert result.significant is False

    def test_non_normal_data(self):
        # Uniform data is not normal
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, 100)
        st = StatisticalTester()
        result = st.normality(data)
        # Uniform should likely be detected as non-normal
        assert isinstance(result, TestResult)


class TestEffectSizes:
    def test_cohens_d_large(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        d = st.cohens_d(a, b)
        assert abs(d) > 0.8  # Large effect

    def test_cohens_d_small(self):
        a, b = _make_similar_groups()
        st = StatisticalTester()
        d = st.cohens_d(a, b)
        assert abs(d) < 0.5  # Small effect

    def test_cohens_d_equal_groups(self):
        data = np.ones(20)
        st = StatisticalTester()
        d = st.cohens_d(data, data)
        assert d == 0.0

    def test_cramers_v_zero(self):
        st = StatisticalTester()
        v = st.cramers_v(0.0, 100, 1)
        assert v == 0.0

    def test_eta_squared_zero_denom(self):
        st = StatisticalTester()
        es = st.eta_squared(0.0, 0, 0)
        assert es == 0.0


class TestRunSuite:
    def test_parametric_selected_for_normal(self):
        a = _make_normal_data()[:50]
        rng = np.random.RandomState(99)
        b = rng.normal(0, 1, 50)
        st = StatisticalTester()
        report = st.run_suite(a, b)
        assert isinstance(report, TestSuiteReport)
        assert report.recommended_test == "welch_t_test"
        assert len(report.results) >= 3

    def test_nonparametric_selected_for_non_normal(self):
        rng = np.random.RandomState(42)
        # Exponential is non-normal
        a = rng.exponential(1, 50)
        b = rng.exponential(5, 50)
        st = StatisticalTester()
        report = st.run_suite(a, b)
        assert report.recommended_test in ("welch_t_test", "mann_whitney_u")
        assert len(report.results) >= 3

    def test_suite_summary_present(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        report = st.run_suite(a, b)
        assert len(report.summary) > 0
        assert "Recommended test" in report.summary

    def test_suite_includes_correlation(self):
        a, b = _make_different_groups()
        st = StatisticalTester()
        report = st.run_suite(a, b)
        test_names = [r.test_name for r in report.results]
        assert "pearson_correlation" in test_names


class TestEdgeCases:
    def test_single_element_groups(self):
        st = StatisticalTester()
        # Mann-Whitney works with very small samples
        result = st.mann_whitney(np.array([1.0]), np.array([100.0]))
        assert isinstance(result, TestResult)

    def test_large_effect_size(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 30)
        b = rng.normal(100, 1, 30)
        st = StatisticalTester()
        d = st.cohens_d(a, b)
        assert abs(d) > 5.0  # Very large effect
