"""Tolerance stackup analysis — worst-case accumulation along assembly chains.

Given a chain of dimensions (each with nominal ± tolerance), computes the
worst-case accumulated error at the end of the chain.  This is the
*worst-case method* (also called extreme variation method), which assumes
all dimensions simultaneously deviate in the worst direction.

Typical usage::

    stack = ToleranceStackup()
    stack.add_dimension("plate_thickness", 6.0, upper=0.1, lower=-0.1)
    stack.add_dimension("washer_thickness", 0.5, upper=0.05, lower=-0.05)
    stack.add_dimension("bracket_height", 47.0, upper=0.2, lower=-0.2)
    result = stack.compute_stackup()
    # result.nominal = 53.5
    # result.upper_dev = 0.35
    # result.lower_dev = -0.35

    ok = stack.check_acceptable(allowed_upper=0.5, allowed_lower=-0.5)

Pure-function module: no FreeCAD imports, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..knowledge.tolerance import (
    compute_fit,
    it_tolerance,
    recommend_fit,
)


# ============================================================================
# Data model
# ============================================================================

@dataclass
class ToleranceDimension:
    """A single dimension with bilateral tolerance."""

    name: str
    nominal: float          # mm
    upper_dev: float        # mm, positive
    lower_dev: float        # mm, negative (or zero)
    direction: str = "+"    # "+" adds to chain, "-" subtracts

    @property
    def max_value(self) -> float:
        return self.nominal + self.upper_dev

    @property
    def min_value(self) -> float:
        return self.nominal + self.lower_dev


@dataclass
class StackupResult:
    """Result from tolerance stackup computation."""

    method: str = "worst_case"    # "worst_case" | "rss"
    nominal: float = 0.0        # Sum of nominals
    upper_dev: float = 0.0      # Worst-case positive deviation
    lower_dev: float = 0.0      # Worst-case negative deviation
    max_value: float = 0.0      # nominal + upper_dev
    min_value: float = 0.0      # nominal + lower_dev
    total_tolerance: float = 0.0  # upper_dev - lower_dev
    dimensions: list[ToleranceDimension] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "nominal": round(self.nominal, 4),
            "upper_dev": round(self.upper_dev, 4),
            "lower_dev": round(self.lower_dev, 4),
            "max_value": round(self.max_value, 4),
            "min_value": round(self.min_value, 4),
            "total_tolerance": round(self.total_tolerance, 4),
        }


# ============================================================================
# ToleranceStackup
# ============================================================================

class ToleranceStackup:
    """Accumulate tolerance along a dimensional chain (worst-case method).

    Dimensions are added with ``add_dimension()``.  The stackup direction
    is controlled by the ``direction`` parameter: "+" means the dimension
    contributes positively to the chain total, "-" means it subtracts.

    The worst-case method assumes all dimensions deviate simultaneously
    in the direction that maximizes (or minimizes) the total.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._dimensions: list[ToleranceDimension] = []

    def add_dimension(
        self,
        name: str,
        nominal: float,
        upper: float = 0.0,
        lower: float = 0.0,
        direction: str = "+",
        it_grade: str = "",
        nominal_d: float = 0.0,
    ) -> None:
        """Add a dimension to the tolerance chain.

        Args:
            name: Descriptive name for this dimension.
            nominal: Nominal value in mm.
            upper: Upper deviation in mm (positive).
            lower: Lower deviation in mm (negative or zero).
            direction: "+" or "-" — how this dimension contributes to the chain.
            it_grade: Optional IT grade string.  If provided, tolerance is
                      looked up from the ISO table using ``nominal_d``.
            nominal_d: Nominal diameter for IT grade lookup (if different from nominal).
        """
        if it_grade:
            d = nominal_d if nominal_d > 0 else nominal
            it = it_tolerance(d, it_grade)
            upper = it / 2
            lower = -it / 2

        self._dimensions.append(ToleranceDimension(
            name=name,
            nominal=nominal,
            upper_dev=upper,
            lower_dev=lower,
            direction=direction,
        ))

    def add_fit_dimension(
        self,
        name: str,
        nominal_d: float,
        hole_grade: str,
        shaft_grade: str,
        hole_deviation: str = "H",
        shaft_deviation: str = "h",
        side: str = "hole",
        direction: str = "+",
    ) -> None:
        """Add a dimension derived from a fit calculation.

        Args:
            name: Descriptive name.
            nominal_d: Basic diameter in mm.
            hole_grade: IT grade for hole.
            shaft_grade: IT grade for shaft.
            hole_deviation: Hole deviation letter.
            shaft_deviation: Shaft deviation letter.
            side: "hole" or "shaft" — which side's tolerance to add.
            direction: "+" or "-" — chain contribution direction.
        """
        from ..knowledge.tolerance import hole_deviations, shaft_deviations

        fit = compute_fit(nominal_d, hole_grade, shaft_grade,
                          hole_deviation, shaft_deviation)
        if side == "hole":
            upper = fit.hole_es
            lower = fit.hole_ei
        else:
            upper = fit.shaft_es
            lower = fit.shaft_ei

        self._dimensions.append(ToleranceDimension(
            name=name,
            nominal=nominal_d,
            upper_dev=upper,
            lower_dev=lower,
            direction=direction,
        ))

    def compute_stackup(self) -> StackupResult:
        """Compute worst-case tolerance stackup.

        For each dimension:
        - Positive direction: worst-case max = nominal + upper_dev, min = nominal + lower_dev
        - Negative direction: worst-case max = nominal + lower_dev, min = nominal + upper_dev
          (reversed because subtracting)

        The chain total:
        - nominal = sum of all nominals (with direction sign)
        - upper_dev = worst-case positive deviation of the total
        - lower_dev = worst-case negative deviation of the total
        """
        total_nominal = 0.0
        total_upper = 0.0  # worst-case positive deviation
        total_lower = 0.0  # worst-case negative deviation

        for dim in self._dimensions:
            if dim.direction == "+":
                total_nominal += dim.nominal
                total_upper += dim.upper_dev
                total_lower += dim.lower_dev
            else:
                total_nominal -= dim.nominal
                # When subtracting, upper deviation of the subtracted dim
                # pushes the total DOWN, and lower deviation pushes it UP
                total_upper += -dim.lower_dev
                total_lower += -dim.upper_dev

        return StackupResult(
            method="worst_case",
            nominal=total_nominal,
            upper_dev=total_upper,
            lower_dev=total_lower,
            max_value=total_nominal + total_upper,
            min_value=total_nominal + total_lower,
            total_tolerance=total_upper - total_lower,
            dimensions=list(self._dimensions),
        )

    def compute_rss(self) -> StackupResult:
        """Root Sum Square statistical tolerance analysis.

        Assumes normal distribution with ±3σ tolerance bounds.
        RSS deviation = sqrt(sum of squared deviations).
        """
        total_nominal = 0.0
        rss_upper = 0.0
        rss_lower = 0.0

        for dim in self._dimensions:
            if dim.direction == "+":
                total_nominal += dim.nominal
                rss_upper += dim.upper_dev ** 2
                rss_lower += dim.lower_dev ** 2
            else:
                total_nominal -= dim.nominal
                rss_upper += (-dim.lower_dev) ** 2
                rss_lower += (-dim.upper_dev) ** 2

        rss_upper = rss_upper ** 0.5
        rss_lower = rss_lower ** 0.5

        return StackupResult(
            method="rss",
            nominal=total_nominal,
            upper_dev=rss_upper,
            lower_dev=-rss_lower,
            max_value=total_nominal + rss_upper,
            min_value=total_nominal - rss_lower,
            total_tolerance=rss_upper + rss_lower,
            dimensions=list(self._dimensions),
        )

    def check_acceptable(
        self,
        allowed_upper: float = 0.0,
        allowed_lower: float = 0.0,
        allowed_total: float = 0.0,
        method: str = "worst_case",
    ) -> bool:
        """Check if the stackup is within acceptable limits.

        Args:
            allowed_upper: Maximum acceptable upper deviation (mm).
            allowed_lower: Maximum acceptable lower deviation magnitude (mm, positive).
            allowed_total: Maximum acceptable total tolerance band (mm).
            method: "worst_case" or "rss" — which analysis method to use.

        Returns:
            True if all specified limits are satisfied.
        """
        if method == "rss":
            result = self.compute_rss()
        else:
            result = self.compute_stackup()
        if allowed_upper > 0 and result.upper_dev > allowed_upper:
            return False
        if allowed_lower > 0 and abs(result.lower_dev) > allowed_lower:
            return False
        if allowed_total > 0 and result.total_tolerance > allowed_total:
            return False
        return True

    def clear(self) -> None:
        """Remove all dimensions from the chain."""
        self._dimensions.clear()

    @property
    def dimension_count(self) -> int:
        return len(self._dimensions)


# ============================================================================
# Assembly chain analysis helper
# ============================================================================

def analyze_assembly_chain(
    chain: list[dict],
    allowed_total: float = 0.0,
) -> StackupResult:
    """Analyze a tolerance chain from a list of dimension dicts.

    Each dict should have: name, nominal, upper (or it_grade), lower, direction.

    Args:
        chain: List of dimension dictionaries.
        allowed_total: If > 0, mark result as unacceptable if exceeded.

    Returns:
        StackupResult with computed values.
    """
    stack = ToleranceStackup()
    for d in chain:
        stack.add_dimension(
            name=d.get("name", ""),
            nominal=d.get("nominal", 0.0),
            upper=d.get("upper", 0.0),
            lower=d.get("lower", 0.0),
            direction=d.get("direction", "+"),
            it_grade=d.get("it_grade", ""),
            nominal_d=d.get("nominal_d", 0.0),
        )
    return stack.compute_stackup()


# ============================================================================
# Tool integration
# ============================================================================

def tolerance_analysis_tool_factory() -> tuple[Any, Any]:
    """Create the tolerance_analysis tool."""
    from ..models.base import ToolDefinition

    definition = ToolDefinition(
        name="tolerance_analysis",
        description="Analyze tolerance stackup along an assembly dimensional chain (worst-case method)",
        parameters={
            "dimensions": {
                "type": "array",
                "description": "List of {name, nominal, upper, lower, direction} dicts",
            },
            "allowed_total": {
                "type": "number",
                "description": "Maximum acceptable total tolerance band (mm), 0 = no check",
            },
        },
    )

    class _ToleranceAnalysisTool:
        def execute(self, *, dimensions: list | None = None,
                    allowed_total: float = 0.0, **kwargs) -> str:
            dims = dimensions or []
            result = analyze_assembly_chain(dims, allowed_total)
            lines = [
                f"Tolerance Stackup: {result.nominal:.3f} mm nominal",
                f"  Upper: +{result.upper_dev:.4f} mm → max {result.max_value:.3f} mm",
                f"  Lower: {result.lower_dev:.4f} mm → min {result.min_value:.3f} mm",
                f"  Total band: {result.total_tolerance:.4f} mm",
                f"  Dimensions: {len(result.dimensions)}",
            ]
            if allowed_total > 0:
                ok = result.total_tolerance <= allowed_total
                lines.append(f"  Acceptable ({allowed_total:.3f} mm): {'YES' if ok else 'NO'}")
            for d in result.dimensions:
                lines.append(f"    - {d.name}: {d.nominal:.3f} ({d.lower_dev:+.4f} / {d.upper_dev:+.4f}) [{d.direction}]")
            return "\n".join(lines)

        def get_definition(self) -> ToolDefinition:
            return definition

    return definition, _ToleranceAnalysisTool
