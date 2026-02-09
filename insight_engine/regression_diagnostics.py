"""Regression diagnostics: residuals, VIF, Cook's distance, heteroscedasticity, Durbin-Watson."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Legacy dataclasses (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------


@dataclass
class ResidualStats:
    """Statistics for regression residuals."""

    mean: float
    std: float
    is_normal: bool
    p_value: float


@dataclass
class VIFResultLegacy:
    """Variance Inflation Factor for a single feature (legacy)."""

    feature: str
    vif_value: float
    is_problematic: bool  # True if VIF > 10


# Backward-compat alias
VIFResult = VIFResultLegacy


@dataclass
class DiagnosticsReport:
    """Complete regression diagnostics report (legacy)."""

    residuals: ResidualStats
    vif_results: list[VIFResultLegacy] = field(default_factory=list)
    cooks_distances: list[float] = field(default_factory=list)
    dw_statistic: float = 0.0
    heteroscedasticity_pvalue: float = 0.0


class RegressionDiagnostics:
    """OLS regression diagnostics toolkit.

    Provides residual analysis, multicollinearity detection (VIF),
    influential observation detection (Cook's distance), autocorrelation
    (Durbin-Watson), and heteroscedasticity testing.
    """

    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> ResidualStats:
        """Compute residual statistics and normality test (Shapiro-Wilk)."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred

        mean_val = round(float(np.mean(residuals)), 6)
        std_val = round(float(np.std(residuals, ddof=1)), 6) if len(residuals) > 1 else 0.0

        if len(residuals) >= 3:
            _, p_val = stats.shapiro(residuals)
            p_val = round(float(p_val), 6)
            is_normal = bool(p_val >= 0.05)
        else:
            p_val = 1.0
            is_normal = True

        return ResidualStats(
            mean=mean_val,
            std=std_val,
            is_normal=is_normal,
            p_value=p_val,
        )

    def vif(self, X: np.ndarray, feature_names: list[str] | None = None) -> list[VIFResultLegacy]:
        """Compute Variance Inflation Factor per feature."""
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results: list[VIFResultLegacy] = []
        for i in range(n_features):
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            if X_others.shape[1] == 0:
                results.append(VIFResultLegacy(feature=feature_names[i], vif_value=1.0, is_problematic=False))
                continue
            model = LinearRegression()
            model.fit(X_others, y_i)
            y_pred = model.predict(X_others)
            ss_res = float(np.sum((y_i - y_pred) ** 2))
            ss_tot = float(np.sum((y_i - np.mean(y_i)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            vif_val = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
            results.append(
                VIFResultLegacy(
                    feature=feature_names[i],
                    vif_value=round(vif_val, 4),
                    is_problematic=bool(vif_val > 10),
                )
            )
        return results

    def cooks_distance(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Compute Cook's distance for each observation."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = float(np.sum(residuals**2)) / max(n - p, 1)

        try:
            hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
            h = np.diag(hat_matrix)
        except np.linalg.LinAlgError:
            return [0.0] * n

        cooks = []
        for i in range(n):
            denom = p * mse * (1 - h[i]) ** 2
            if denom > 0:
                d = float(residuals[i] ** 2 * h[i]) / denom
            else:
                d = 0.0
            cooks.append(round(d, 6))

        return cooks

    def heteroscedasticity_test(self, residuals: np.ndarray, X: np.ndarray) -> float:
        """Breusch-Pagan style heteroscedasticity test."""
        residuals = np.asarray(residuals, dtype=float)
        X = np.asarray(X, dtype=float)

        squared_resid = residuals**2
        model = LinearRegression()
        model.fit(X, squared_resid)
        pred = model.predict(X)

        ss_reg = float(np.sum((pred - np.mean(squared_resid)) ** 2))
        ss_res = float(np.sum((squared_resid - pred) ** 2))

        n, p = X.shape
        df_reg = p
        df_res = max(n - p - 1, 1)

        ms_reg = ss_reg / max(df_reg, 1)
        ms_res = ss_res / df_res

        f_stat = ms_reg / ms_res if ms_res > 0 else 0.0
        p_val = 1.0 - float(stats.f.cdf(f_stat, df_reg, df_res))

        return round(p_val, 6)

    def durbin_watson(self, residuals: np.ndarray) -> float:
        """Durbin-Watson statistic for autocorrelation."""
        residuals = np.asarray(residuals, dtype=float)
        if len(residuals) < 2:
            return 2.0
        diff = np.diff(residuals)
        ss_diff = float(np.sum(diff**2))
        ss_resid = float(np.sum(residuals**2))
        if ss_resid == 0:
            return 2.0
        return round(ss_diff / ss_resid, 6)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DiagnosticsReport:
        """Fit OLS regression and compute all diagnostics."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        resid_stats = self.residual_analysis(y, y_pred)
        vif_results = self.vif(X, feature_names)
        cooks = self.cooks_distance(X, y)
        dw = self.durbin_watson(residuals)
        hetero_p = self.heteroscedasticity_test(residuals, X)

        return DiagnosticsReport(
            residuals=resid_stats,
            vif_results=vif_results,
            cooks_distances=cooks,
            dw_statistic=dw,
            heteroscedasticity_pvalue=hetero_p,
        )


# ---------------------------------------------------------------------------
# New Tier-2 dataclasses and classes
# ---------------------------------------------------------------------------


@dataclass
class ResidualAnalysis:
    """Results from residual analysis (Tier 2)."""

    residuals: np.ndarray
    mean: float
    std: float
    skewness: float
    kurtosis: float
    durbin_watson: float
    normality_pvalue: float


@dataclass
class AdvancedVIFResult:
    """Variance Inflation Factor for a single feature (Tier 2)."""

    feature_name: str
    vif_value: float
    is_multicollinear: bool


@dataclass
class InfluenceResult:
    """Influence diagnostics for regression."""

    cooks_distance: np.ndarray
    leverage: np.ndarray
    influential_indices: list[int]


@dataclass
class DiagnosticReport:
    """Comprehensive regression diagnostic report (Tier 2)."""

    residuals: ResidualAnalysis
    vif: list[AdvancedVIFResult]
    influence: InfluenceResult
    overall_health: str
    warnings: list[str] = field(default_factory=list)


class ResidualAnalyzer:
    """Analyze regression residuals (Tier 2)."""

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> ResidualAnalysis:
        """Compute residual statistics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            ResidualAnalysis with stats including Durbin-Watson and normality test.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred

        mean = round(float(np.mean(residuals)), 6)
        std = round(float(np.std(residuals, ddof=1)), 6) if len(residuals) > 1 else 0.0

        skewness = round(float(stats.skew(residuals)), 6) if len(residuals) > 2 else 0.0
        kurtosis = round(float(stats.kurtosis(residuals)), 6) if len(residuals) > 3 else 0.0

        # Durbin-Watson: sum of squared differences / sum of squared residuals
        if len(residuals) > 1:
            diff = np.diff(residuals)
            ss_diff = float(np.sum(diff**2))
            ss_resid = float(np.sum(residuals**2))
            dw = round(ss_diff / ss_resid, 6) if ss_resid > 0 else 0.0
        else:
            dw = 0.0

        # Shapiro-Wilk normality test on residuals
        if len(residuals) >= 3:
            _, p_norm = stats.shapiro(residuals)
            normality_pvalue = round(float(p_norm), 6)
        else:
            normality_pvalue = 1.0

        return ResidualAnalysis(
            residuals=residuals,
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            durbin_watson=dw,
            normality_pvalue=normality_pvalue,
        )


class MulticollinearityDetector:
    """Detect multicollinearity via Variance Inflation Factors (Tier 2)."""

    def compute_vif(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        threshold: float = 5.0,
    ) -> list[AdvancedVIFResult]:
        """Compute VIF for each feature.

        VIF_j = 1 / (1 - R_j^2) where R_j^2 is R-squared from regressing
        feature j on all other features.

        Args:
            X: Feature matrix (n_samples, n_features).
            feature_names: Optional feature names.
            threshold: VIF threshold for multicollinearity flag (default 5.0).

        Returns:
            List of AdvancedVIFResult, one per feature.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if n_features == 1:
            return [
                AdvancedVIFResult(
                    feature_name=feature_names[0],
                    vif_value=1.0,
                    is_multicollinear=False,
                )
            ]

        results: list[AdvancedVIFResult] = []
        for j in range(n_features):
            y_j = X[:, j]
            X_others = np.delete(X, j, axis=1)

            # Check for constant column
            if np.std(y_j) == 0:
                results.append(
                    AdvancedVIFResult(
                        feature_name=feature_names[j],
                        vif_value=float("inf"),
                        is_multicollinear=True,
                    )
                )
                continue

            # Add intercept
            X_aug = np.column_stack([np.ones(n_samples), X_others])

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_j, rcond=None)
                y_pred = X_aug @ coeffs
                ss_res = float(np.sum((y_j - y_pred) ** 2))
                ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                r_squared = max(0.0, min(1.0, r_squared))

                if r_squared >= 1.0:
                    vif = float("inf")
                else:
                    vif = round(1.0 / (1.0 - r_squared), 6)
            except np.linalg.LinAlgError:
                vif = float("inf")

            results.append(
                AdvancedVIFResult(
                    feature_name=feature_names[j],
                    vif_value=vif,
                    is_multicollinear=vif > threshold,
                )
            )
        return results


class InfluenceAnalyzer:
    """Analyze influence of individual observations on regression (Tier 2)."""

    def analyze(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> InfluenceResult:
        """Compute Cook's distance and leverage scores.

        Cook's D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
        Leverage from hat matrix: H = X(X'X)^(-1)X'
        Influential points: Cook's D > 4/n

        Args:
            X: Feature matrix (n_samples, n_features).
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            InfluenceResult with Cook's distance, leverage, and influential indices.
        """
        X = np.asarray(X, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        residuals = y_true - y_pred

        # Add intercept for hat matrix calculation
        X_aug = np.column_stack([np.ones(n), X])

        # Hat matrix: H = X(X'X)^(-1)X'
        try:
            XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)

        hat_matrix = X_aug @ XtX_inv @ X_aug.T
        leverage = np.diag(hat_matrix)

        # MSE
        mse = float(np.sum(residuals**2)) / max(n - p - 1, 1)

        # Cook's distance
        p_full = p + 1  # includes intercept
        cooks_d = np.zeros(n)
        for i in range(n):
            h_ii = leverage[i]
            e_i = residuals[i]
            denom = p_full * mse * (1 - h_ii) ** 2
            if denom > 0:
                cooks_d[i] = (e_i**2 / denom) * h_ii
            else:
                cooks_d[i] = 0.0

        # Flag influential: Cook's D > 4/n
        threshold = 4.0 / n if n > 0 else 0.0
        influential = [int(i) for i in range(n) if cooks_d[i] > threshold]

        return InfluenceResult(
            cooks_distance=np.round(cooks_d, 6),
            leverage=np.round(leverage, 6),
            influential_indices=influential,
        )


class RegressionDiagnosticRunner:
    """Run all regression diagnostics and generate a report (Tier 2)."""

    def __init__(self) -> None:
        self._residual_analyzer = ResidualAnalyzer()
        self._multicollinearity_detector = MulticollinearityDetector()
        self._influence_analyzer = InfluenceAnalyzer()

    def run_all(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DiagnosticReport:
        """Run all diagnostics and aggregate into a report.

        Args:
            X: Feature matrix.
            y_true: Actual values.
            y_pred: Predicted values.
            feature_names: Optional feature names for VIF.

        Returns:
            DiagnosticReport with residuals, VIF, influence, health assessment, and warnings.
        """
        X = np.asarray(X, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        residual_analysis = self._residual_analyzer.analyze(y_true, y_pred)
        vif_results = self._multicollinearity_detector.compute_vif(X, feature_names)
        influence_result = self._influence_analyzer.analyze(X, y_true, y_pred)

        warnings_list: list[str] = []
        issues = 0

        # Check residual normality
        if residual_analysis.normality_pvalue < 0.05:
            warnings_list.append("Residuals are not normally distributed (Shapiro-Wilk p < 0.05)")
            issues += 1

        # Check autocorrelation (Durbin-Watson far from 2)
        dw = residual_analysis.durbin_watson
        if dw < 1.5 or dw > 2.5:
            warnings_list.append(f"Possible autocorrelation in residuals (Durbin-Watson = {dw})")
            issues += 1

        # Check multicollinearity
        high_vif = [v for v in vif_results if v.is_multicollinear]
        if high_vif:
            names = [v.feature_name for v in high_vif]
            warnings_list.append(f"Multicollinearity detected in features: {', '.join(names)}")
            issues += 1

        # Check influential points
        n = len(y_true)
        n_influential = len(influence_result.influential_indices)
        if n_influential > 0:
            pct = round(100 * n_influential / n, 1) if n > 0 else 0
            warnings_list.append(f"{n_influential} influential point(s) detected ({pct}% of data)")
            issues += 1

        # Check skewness
        if abs(residual_analysis.skewness) > 1.0:
            warnings_list.append(f"Residuals are skewed (skewness = {residual_analysis.skewness})")
            issues += 1

        # Overall health
        if issues == 0:
            overall_health = "good"
        elif issues <= 2:
            overall_health = "moderate"
        else:
            overall_health = "poor"

        return DiagnosticReport(
            residuals=residual_analysis,
            vif=vif_results,
            influence=influence_result,
            overall_health=overall_health,
            warnings=warnings_list,
        )
