"""Statistical hypothesis testing: t-tests, chi-square, ANOVA, correlation, normality."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a single statistical test."""

    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    effect_size: float | None = None
    interpretation: str = ""


@dataclass
class TestSuiteReport:
    """Result of running an automated test suite."""

    results: list[TestResult] = field(default_factory=list)
    summary: str = ""
    recommended_test: str = ""


class StatisticalTester:
    """Statistical hypothesis testing toolkit."""

    def t_test_independent(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        """Welch's independent samples t-test."""
        group_a = np.asarray(group_a, dtype=float)
        group_b = np.asarray(group_b, dtype=float)
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
        es = self.cohens_d(group_a, group_b)
        sig = bool(p < alpha)
        interp = (
            f"Groups are {'significantly' if sig else 'not significantly'} different "
            f"(t={round(float(stat), 6)}, p={round(float(p), 6)}, d={round(es, 6)})"
        )
        return TestResult(
            test_name="welch_t_test",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=round(es, 6),
            interpretation=interp,
        )

    def t_test_paired(
        self,
        before: np.ndarray,
        after: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        """Paired samples t-test."""
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        stat, p = stats.ttest_rel(before, after)
        diff = after - before
        es = round(float(np.mean(diff) / np.std(diff, ddof=1)), 6) if np.std(diff, ddof=1) > 0 else 0.0
        sig = bool(p < alpha)
        interp = (
            f"Paired difference is {'significant' if sig else 'not significant'} "
            f"(t={round(float(stat), 6)}, p={round(float(p), 6)})"
        )
        return TestResult(
            test_name="paired_t_test",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=es,
            interpretation=interp,
        )

    def chi_square(
        self,
        observed: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        """Chi-square test of independence on a contingency table."""
        observed = np.asarray(observed, dtype=float)
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        n = float(observed.sum())
        min_dim = min(observed.shape) - 1
        es = self.cramers_v(float(chi2), int(n), min_dim)
        sig = bool(p < alpha)
        interp = (
            f"Variables are {'dependent' if sig else 'independent'} "
            f"(chi2={round(float(chi2), 6)}, p={round(float(p), 6)}, V={round(es, 6)})"
        )
        return TestResult(
            test_name="chi_square",
            statistic=round(float(chi2), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=round(es, 6),
            interpretation=interp,
        )

    def anova_oneway(
        self,
        groups: list[np.ndarray],
        alpha: float = 0.05,
    ) -> TestResult:
        """One-way ANOVA across multiple groups."""
        groups = [np.asarray(g, dtype=float) for g in groups]
        stat, p = stats.f_oneway(*groups)
        n_total = sum(len(g) for g in groups)
        df_between = len(groups) - 1
        df_within = n_total - len(groups)
        es = self.eta_squared(float(stat), df_between, df_within)
        sig = bool(p < alpha)
        interp = (
            f"Group means are {'significantly' if sig else 'not significantly'} different "
            f"(F={round(float(stat), 6)}, p={round(float(p), 6)}, eta2={round(es, 6)})"
        )
        return TestResult(
            test_name="anova_oneway",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=round(es, 6),
            interpretation=interp,
        )

    def mann_whitney(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric alternative to independent t-test)."""
        group_a = np.asarray(group_a, dtype=float)
        group_b = np.asarray(group_b, dtype=float)
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        # Rank-biserial correlation as effect size
        n1, n2 = len(group_a), len(group_b)
        es = round(1.0 - (2.0 * float(stat)) / (n1 * n2), 6)
        sig = bool(p < alpha)
        interp = (
            f"Distributions are {'significantly' if sig else 'not significantly'} different "
            f"(U={round(float(stat), 6)}, p={round(float(p), 6)})"
        )
        return TestResult(
            test_name="mann_whitney_u",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=es,
            interpretation=interp,
        )

    def correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
        alpha: float = 0.05,
    ) -> TestResult:
        """Correlation test (Pearson or Spearman)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if method == "pearson":
            stat, p = stats.pearsonr(x, y)
        elif method == "spearman":
            stat, p = stats.spearmanr(x, y)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: pearson, spearman")
        sig = bool(p < alpha)
        interp = (
            f"{method.title()} correlation is {'significant' if sig else 'not significant'} "
            f"(r={round(float(stat), 6)}, p={round(float(p), 6)})"
        )
        return TestResult(
            test_name=f"{method}_correlation",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=round(abs(float(stat)), 6),
            interpretation=interp,
        )

    def normality(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        """Shapiro-Wilk normality test."""
        data = np.asarray(data, dtype=float)
        stat, p = stats.shapiro(data)
        sig = bool(p < alpha)
        interp = (
            f"Data is {'not ' if sig else ''}normally distributed (W={round(float(stat), 6)}, p={round(float(p), 6)})"
        )
        return TestResult(
            test_name="shapiro_wilk",
            statistic=round(float(stat), 6),
            p_value=round(float(p), 6),
            alpha=alpha,
            significant=sig,
            effect_size=None,
            interpretation=interp,
        )

    def cohens_d(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Cohen's d effect size for two independent groups."""
        group_a = np.asarray(group_a, dtype=float)
        group_b = np.asarray(group_b, dtype=float)
        n1, n2 = len(group_a), len(group_b)
        var1, var2 = float(np.var(group_a, ddof=1)), float(np.var(group_b, ddof=1))
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return round(float((np.mean(group_a) - np.mean(group_b)) / pooled_std), 6)

    def cramers_v(self, chi2: float, n: int, min_dim: int) -> float:
        """Cramer's V effect size for chi-square test."""
        if n == 0 or min_dim == 0:
            return 0.0
        return round(float(np.sqrt(chi2 / (n * min_dim))), 6)

    def eta_squared(self, f_stat: float, df_between: int, df_within: int) -> float:
        """Eta-squared effect size for ANOVA."""
        denom = f_stat * df_between + df_within
        if denom == 0:
            return 0.0
        return round(float((f_stat * df_between) / denom), 6)

    def run_suite(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: float = 0.05,
    ) -> TestSuiteReport:
        """Auto-select parametric vs non-parametric tests based on normality."""
        group_a = np.asarray(group_a, dtype=float)
        group_b = np.asarray(group_b, dtype=float)

        results: list[TestResult] = []

        # Normality tests
        norm_a = self.normality(group_a, alpha=alpha)
        norm_b = self.normality(group_b, alpha=alpha)
        results.append(norm_a)
        results.append(norm_b)

        both_normal = not norm_a.significant and not norm_b.significant

        if both_normal:
            # Parametric: Welch's t-test
            t_result = self.t_test_independent(group_a, group_b, alpha=alpha)
            results.append(t_result)
            recommended = "welch_t_test"
        else:
            # Non-parametric: Mann-Whitney U
            mw_result = self.mann_whitney(group_a, group_b, alpha=alpha)
            results.append(mw_result)
            recommended = "mann_whitney_u"

        # Always include correlation
        corr_result = self.correlation(group_a, group_b, method="pearson", alpha=alpha)
        results.append(corr_result)

        summary = (
            f"Normality: group_a={'normal' if not norm_a.significant else 'non-normal'}, "
            f"group_b={'normal' if not norm_b.significant else 'non-normal'}. "
            f"Recommended test: {recommended}."
        )

        return TestSuiteReport(
            results=results,
            summary=summary,
            recommended_test=recommended,
        )
