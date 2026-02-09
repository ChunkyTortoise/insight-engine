"""insight-engine: Upload CSV/Excel, get instant dashboards, predictive models, and PDF reports."""

__version__ = "0.1.0"

from insight_engine.advanced_anomaly import (
    AdvancedAnomaly,
    AnomalyEnsemble,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    MahalanobisDetector,
)
from insight_engine.anomaly_detector import Anomaly, AnomalyDetector
from insight_engine.clustering import (
    ClusterComparison,
    Clusterer,
    ClusterEvaluation,
    ClusterResult,
    ElbowResult,
)
from insight_engine.data_quality import ColumnQuality, DataQualityReport, DataQualityScorer
from insight_engine.dimensionality import DimensionalityReducer, ReductionResult, ScreePlotData
from insight_engine.feature_lab import FeatureLab, FeatureResult, SelectionResult
from insight_engine.forecaster import (
    ConfidenceForecast,
    CrossValResult,
    DecompositionResult,
    ForecastComparison,
    Forecaster,
    ForecastResult,
)
from insight_engine.hypertuner import HyperTuner, TuningResult
from insight_engine.kpi_framework import KPIDashboard, KPIDefinition, KPIEngine, KPIResult
from insight_engine.model_observatory import ModelObservatory, ModelResult, ObservatoryReport
from insight_engine.regression_diagnostics import (
    AdvancedVIFResult,
    DiagnosticReport,
    DiagnosticsReport,
    InfluenceAnalyzer,
    InfluenceResult,
    MulticollinearityDetector,
    RegressionDiagnosticRunner,
    RegressionDiagnostics,
    ResidualAnalysis,
    ResidualAnalyzer,
    ResidualStats,
    VIFResult,
)
from insight_engine.statistical_tests import StatisticalTester, TestResult, TestSuiteReport

__all__ = [
    "AdvancedAnomaly",
    "AdvancedVIFResult",
    "Anomaly",
    "AnomalyDetector",
    "AnomalyEnsemble",
    "ClusterComparison",
    "ClusterEvaluation",
    "ClusterResult",
    "Clusterer",
    "ColumnQuality",
    "ConfidenceForecast",
    "CrossValResult",
    "DataQualityReport",
    "DataQualityScorer",
    "DecompositionResult",
    "DiagnosticReport",
    "DiagnosticsReport",
    "DimensionalityReducer",
    "ElbowResult",
    "FeatureLab",
    "FeatureResult",
    "ForecastComparison",
    "ForecastResult",
    "Forecaster",
    "HyperTuner",
    "InfluenceAnalyzer",
    "InfluenceResult",
    "IsolationForestDetector",
    "KPIDashboard",
    "KPIDefinition",
    "KPIEngine",
    "KPIResult",
    "LocalOutlierFactorDetector",
    "MahalanobisDetector",
    "ModelObservatory",
    "ModelResult",
    "MulticollinearityDetector",
    "ObservatoryReport",
    "ReductionResult",
    "RegressionDiagnosticRunner",
    "RegressionDiagnostics",
    "ResidualAnalysis",
    "ResidualAnalyzer",
    "ResidualStats",
    "ScreePlotData",
    "SelectionResult",
    "StatisticalTester",
    "TestResult",
    "TestSuiteReport",
    "TuningResult",
    "VIFResult",
]
