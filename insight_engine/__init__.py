"""insight-engine: Upload CSV/Excel, get instant dashboards, predictive models, and PDF reports."""

__version__ = "0.1.0"

from insight_engine.anomaly_detector import Anomaly, AnomalyDetector
from insight_engine.clustering import ClusterComparison, Clusterer, ClusterResult
from insight_engine.feature_lab import FeatureLab, FeatureResult
from insight_engine.forecaster import ForecastComparison, Forecaster, ForecastResult

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "ClusterComparison",
    "ClusterResult",
    "Clusterer",
    "FeatureLab",
    "FeatureResult",
    "ForecastComparison",
    "ForecastResult",
    "Forecaster",
]
