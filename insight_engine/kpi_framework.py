"""KPI framework: define, compute, assess, and auto-suggest business KPIs."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class KPIDefinition:
    """Definition of a Key Performance Indicator."""

    name: str
    formula: str
    thresholds: dict[str, float]
    direction: str = "higher_is_better"
    unit: str = ""


@dataclass
class KPIResult:
    """Result of computing a KPI."""

    name: str
    value: float
    status: str
    direction: str
    unit: str
    trend: str | None = None


@dataclass
class KPIDashboard:
    """Dashboard of all computed KPI results."""

    results: list[KPIResult] = field(default_factory=list)
    health_score: float = 0.0
    timestamp: str = ""


# Safe expression evaluator using AST
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str, variables: dict[str, float]) -> float:
    """Evaluate a simple math expression with named variables.

    Supports: +, -, *, /, parentheses, and variable names from the data dict.
    No function calls, attribute access, or other constructs allowed.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {expr}") from e

    def _eval_node(node: ast.expr) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable '{node.id}' in formula")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                return 0.0
            return float(op_fn(left, right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return float(op_fn(_eval_node(node.operand)))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval_node(tree)


class KPIEngine:
    """KPI definition, computation, and assessment engine."""

    def __init__(self) -> None:
        self._kpis: dict[str, KPIDefinition] = {}

    def define(
        self,
        name: str,
        formula: str,
        thresholds: dict[str, float],
        direction: str = "higher_is_better",
        unit: str = "",
    ) -> KPIDefinition:
        """Register a KPI definition."""
        kpi = KPIDefinition(
            name=name,
            formula=formula,
            thresholds=thresholds,
            direction=direction,
            unit=unit,
        )
        self._kpis[name] = kpi
        return kpi

    def compute(self, name: str, data: dict[str, float]) -> KPIResult:
        """Evaluate a KPI formula with provided data and assess status."""
        if name not in self._kpis:
            raise KeyError(f"KPI '{name}' not defined")

        kpi = self._kpis[name]
        value = round(_safe_eval(kpi.formula, data), 6)
        status = self.assess_status(value, kpi.thresholds, kpi.direction)

        return KPIResult(
            name=kpi.name,
            value=value,
            status=status,
            direction=kpi.direction,
            unit=kpi.unit,
        )

    def compute_all(self, data: dict[str, float]) -> KPIDashboard:
        """Compute all defined KPIs and return a dashboard."""
        results: list[KPIResult] = []
        for name in self._kpis:
            result = self.compute(name, data)
            results.append(result)

        on_track_count = sum(1 for r in results if r.status == "on_track")
        health = round(on_track_count / len(results), 6) if results else 0.0

        return KPIDashboard(
            results=results,
            health_score=health,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def assess_status(
        self,
        value: float,
        thresholds: dict[str, float],
        direction: str,
    ) -> str:
        """Assess KPI status against thresholds.

        Returns 'on_track', 'warning', or 'critical'.
        """
        on_track = thresholds.get("on_track", 0.0)
        warning = thresholds.get("warning", 0.0)

        if direction == "higher_is_better":
            if value >= on_track:
                return "on_track"
            if value >= warning:
                return "warning"
            return "critical"
        else:
            # lower_is_better
            if value <= on_track:
                return "on_track"
            if value <= warning:
                return "warning"
            return "critical"

    def detect_trend(self, values: list[float]) -> str:
        """Detect trend using simple linear regression slope.

        Returns 'improving', 'stable', or 'declining'.
        """
        if len(values) < 2:
            return "stable"

        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        # Linear regression: slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            return "stable"

        slope = float(ss_xy / ss_xx)

        # Normalize slope relative to mean to determine significance
        y_range = float(np.ptp(y))
        if y_range == 0:
            return "stable"

        relative_slope = slope / y_range

        if relative_slope > 0.1:
            return "improving"
        if relative_slope < -0.1:
            return "declining"
        return "stable"

    def suggest_kpis(self, column_names: list[str]) -> list[KPIDefinition]:
        """Auto-suggest KPIs based on column name patterns."""
        suggestions: list[KPIDefinition] = []
        patterns: dict[str, tuple[str, str, dict[str, float], str]] = {
            r"revenue": ("revenue_growth", "revenue / 1", {"on_track": 10000, "warning": 5000}, "higher_is_better"),
            r"cost": ("cost_reduction", "cost / 1", {"on_track": 1000, "warning": 5000}, "lower_is_better"),
            r"conversion": (
                "conversion_rate",
                "conversion / 1",
                {"on_track": 0.1, "warning": 0.05},
                "higher_is_better",
            ),
            r"churn": ("churn_rate", "churn / 1", {"on_track": 0.05, "warning": 0.1}, "lower_is_better"),
            r"satisfaction|nps": (
                "satisfaction_score",
                "satisfaction / 1",
                {"on_track": 8.0, "warning": 6.0},
                "higher_is_better",
            ),
            r"response.?time": (
                "response_time",
                "response_time / 1",
                {"on_track": 200, "warning": 500},
                "lower_is_better",
            ),
            r"profit": ("profit_margin", "profit / 1", {"on_track": 0.2, "warning": 0.1}, "higher_is_better"),
        }

        matched_names: set[str] = set()
        for col in column_names:
            col_lower = col.lower()
            for pattern, (name, formula, thresholds, direction) in patterns.items():
                if re.search(pattern, col_lower) and name not in matched_names:
                    matched_names.add(name)
                    suggestions.append(
                        KPIDefinition(
                            name=name,
                            formula=formula,
                            thresholds=thresholds,
                            direction=direction,
                        )
                    )

        return suggestions
