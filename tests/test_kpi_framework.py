"""Tests for the KPI framework module."""

from insight_engine.kpi_framework import KPIDashboard, KPIEngine


class TestFormulaEvaluation:
    def test_simple_division(self):
        engine = KPIEngine()
        engine.define("cpc", "cost / clicks", {"on_track": 1.0, "warning": 2.0}, "lower_is_better", "$")
        result = engine.compute("cpc", {"cost": 100.0, "clicks": 50.0})
        assert result.value == 2.0

    def test_addition(self):
        engine = KPIEngine()
        engine.define("total", "a + b", {"on_track": 100, "warning": 50})
        result = engine.compute("total", {"a": 40.0, "b": 60.0})
        assert result.value == 100.0

    def test_subtraction(self):
        engine = KPIEngine()
        engine.define("profit", "revenue - cost", {"on_track": 500, "warning": 200})
        result = engine.compute("profit", {"revenue": 1000.0, "cost": 300.0})
        assert result.value == 700.0

    def test_multiplication(self):
        engine = KPIEngine()
        engine.define("total_rev", "price * units", {"on_track": 1000, "warning": 500})
        result = engine.compute("total_rev", {"price": 25.0, "units": 40.0})
        assert result.value == 1000.0

    def test_complex_formula(self):
        engine = KPIEngine()
        engine.define("roi", "(revenue - cost) / cost", {"on_track": 0.5, "warning": 0.2})
        result = engine.compute("roi", {"revenue": 150.0, "cost": 100.0})
        assert result.value == 0.5

    def test_parentheses(self):
        engine = KPIEngine()
        engine.define("metric", "(a + b) * c", {"on_track": 100, "warning": 50})
        result = engine.compute("metric", {"a": 3.0, "b": 7.0, "c": 5.0})
        assert result.value == 50.0

    def test_division_by_zero(self):
        engine = KPIEngine()
        engine.define("rate", "a / b", {"on_track": 1, "warning": 0.5})
        result = engine.compute("rate", {"a": 10.0, "b": 0.0})
        assert result.value == 0.0  # Safe: no exception

    def test_invalid_formula(self):
        engine = KPIEngine()
        engine.define("bad", "import os", {"on_track": 1, "warning": 0.5})
        try:
            engine.compute("bad", {})
            assert False, "Should have raised ValueError"
        except (ValueError, KeyError):
            pass

    def test_unknown_variable(self):
        engine = KPIEngine()
        engine.define("metric", "unknown_var / 1", {"on_track": 1, "warning": 0.5})
        try:
            engine.compute("metric", {"x": 1.0})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_undefined_kpi(self):
        engine = KPIEngine()
        try:
            engine.compute("nonexistent", {"x": 1.0})
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestStatusAssessment:
    def test_higher_is_better_on_track(self):
        engine = KPIEngine()
        status = engine.assess_status(100, {"on_track": 80, "warning": 50}, "higher_is_better")
        assert status == "on_track"

    def test_higher_is_better_warning(self):
        engine = KPIEngine()
        status = engine.assess_status(60, {"on_track": 80, "warning": 50}, "higher_is_better")
        assert status == "warning"

    def test_higher_is_better_critical(self):
        engine = KPIEngine()
        status = engine.assess_status(30, {"on_track": 80, "warning": 50}, "higher_is_better")
        assert status == "critical"

    def test_lower_is_better_on_track(self):
        engine = KPIEngine()
        status = engine.assess_status(5, {"on_track": 10, "warning": 20}, "lower_is_better")
        assert status == "on_track"

    def test_lower_is_better_warning(self):
        engine = KPIEngine()
        status = engine.assess_status(15, {"on_track": 10, "warning": 20}, "lower_is_better")
        assert status == "warning"

    def test_lower_is_better_critical(self):
        engine = KPIEngine()
        status = engine.assess_status(25, {"on_track": 10, "warning": 20}, "lower_is_better")
        assert status == "critical"

    def test_boundary_on_track(self):
        engine = KPIEngine()
        status = engine.assess_status(80, {"on_track": 80, "warning": 50}, "higher_is_better")
        assert status == "on_track"

    def test_boundary_warning(self):
        engine = KPIEngine()
        status = engine.assess_status(50, {"on_track": 80, "warning": 50}, "higher_is_better")
        assert status == "warning"


class TestTrendDetection:
    def test_improving_trend(self):
        engine = KPIEngine()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        trend = engine.detect_trend(values)
        assert trend == "improving"

    def test_declining_trend(self):
        engine = KPIEngine()
        values = [50.0, 40.0, 30.0, 20.0, 10.0]
        trend = engine.detect_trend(values)
        assert trend == "declining"

    def test_stable_trend(self):
        engine = KPIEngine()
        values = [10.0, 10.0, 10.0, 10.0]
        trend = engine.detect_trend(values)
        assert trend == "stable"

    def test_single_value_stable(self):
        engine = KPIEngine()
        trend = engine.detect_trend([42.0])
        assert trend == "stable"

    def test_empty_values_stable(self):
        engine = KPIEngine()
        trend = engine.detect_trend([])
        assert trend == "stable"


class TestDashboard:
    def test_compute_all(self):
        engine = KPIEngine()
        engine.define("rev", "revenue / 1", {"on_track": 1000, "warning": 500})
        engine.define("cost", "expenses / 1", {"on_track": 200, "warning": 500}, "lower_is_better")
        dashboard = engine.compute_all({"revenue": 1500.0, "expenses": 100.0})
        assert isinstance(dashboard, KPIDashboard)
        assert len(dashboard.results) == 2

    def test_health_score_all_on_track(self):
        engine = KPIEngine()
        engine.define("a", "x / 1", {"on_track": 50, "warning": 30})
        engine.define("b", "y / 1", {"on_track": 50, "warning": 30})
        dashboard = engine.compute_all({"x": 100.0, "y": 100.0})
        assert dashboard.health_score == 1.0

    def test_health_score_none_on_track(self):
        engine = KPIEngine()
        engine.define("a", "x / 1", {"on_track": 200, "warning": 100})
        engine.define("b", "y / 1", {"on_track": 200, "warning": 100})
        dashboard = engine.compute_all({"x": 10.0, "y": 10.0})
        assert dashboard.health_score == 0.0

    def test_health_score_partial(self):
        engine = KPIEngine()
        engine.define("good", "x / 1", {"on_track": 50, "warning": 30})
        engine.define("bad", "y / 1", {"on_track": 200, "warning": 100})
        dashboard = engine.compute_all({"x": 100.0, "y": 10.0})
        assert dashboard.health_score == 0.5

    def test_timestamp_present(self):
        engine = KPIEngine()
        engine.define("rev", "x / 1", {"on_track": 50, "warning": 30})
        dashboard = engine.compute_all({"x": 100.0})
        assert len(dashboard.timestamp) > 0

    def test_empty_dashboard(self):
        engine = KPIEngine()
        dashboard = engine.compute_all({})
        assert dashboard.health_score == 0.0
        assert len(dashboard.results) == 0


class TestSuggestKPIs:
    def test_revenue_suggestion(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["revenue", "users"])
        names = [s.name for s in suggestions]
        assert "revenue_growth" in names

    def test_cost_suggestion(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["cost_per_unit"])
        names = [s.name for s in suggestions]
        assert "cost_reduction" in names

    def test_conversion_suggestion(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["conversion_rate"])
        names = [s.name for s in suggestions]
        assert "conversion_rate" in names

    def test_churn_suggestion(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["monthly_churn"])
        names = [s.name for s in suggestions]
        assert "churn_rate" in names

    def test_no_match(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["foo", "bar"])
        assert len(suggestions) == 0

    def test_multiple_matches(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["revenue", "cost", "conversion"])
        assert len(suggestions) == 3

    def test_suggestion_direction(self):
        engine = KPIEngine()
        suggestions = engine.suggest_kpis(["cost"])
        cost_kpi = [s for s in suggestions if s.name == "cost_reduction"][0]
        assert cost_kpi.direction == "lower_is_better"
