"""ISO 286 tolerance and fit system — standard tolerance grades, fundamental deviations, and fit calculations.

Provides:
- ISO standard tolerance grades (IT5–IT12) for basic sizes 1–500 mm
- Shaft fundamental deviations (f, g, h, js, k, p) — most common in mechanical design
- Hole fundamental deviations (F, G, H, Js, K, P) — H-basis system
- Fit computation: clearance, transition, interference
- Fit recommendations by application (bearing seat, sliding, locating, press)
- Tolerance-aware dimension calculations for connection features

All dimensions in millimeters.  Tolerance values from ISO 286-1:2010.

Pure-function module: no FreeCAD imports, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Basic size ranges (ISO 286 Table 1)
# ============================================================================

# Each tuple: (lower, upper) in mm.  Upper bound is exclusive.
_BASIC_SIZE_RANGES: list[tuple[float, float]] = [
    (1, 3),
    (3, 6),
    (6, 10),
    (10, 18),
    (18, 30),
    (30, 50),
    (50, 80),
    (80, 120),
    (120, 180),
    (180, 250),
    (250, 315),
    (315, 400),
    (400, 500),
]


def _size_range_index(nominal_d: float) -> int:
    """Return the index into _BASIC_SIZE_RANGES for a given nominal diameter.

    ISO 286-1:2010 ranges have **exclusive** lower and **inclusive** upper
    bounds, i.e. (lo, hi].  The first range (1-3) additionally includes
    the lower bound 1.0 itself.
    Clamps to the nearest valid range for values outside 1-500 mm.
    """
    for i, (lo, hi) in enumerate(_BASIC_SIZE_RANGES):
        if lo < nominal_d <= hi:
            return i
        # First range also includes the exact lower bound (1.0)
        if i == 0 and nominal_d == lo:
            return 0
    # Outside table: clamp
    if nominal_d <= _BASIC_SIZE_RANGES[0][0]:
        return 0
    return len(_BASIC_SIZE_RANGES) - 1


# ============================================================================
# Standard tolerance values (IT grades) — ISO 286-1 Table 2
# ============================================================================
# Values in micrometers (µm).  Convert to mm by dividing by 1000.
# Rows = basic size ranges (same order as _BASIC_SIZE_RANGES)
# Columns = IT5, IT6, IT7, IT8, IT9, IT10, IT11, IT12

_IT_GRADES: list[list[int]] = [
    #  IT5  IT6  IT7  IT8  IT9  IT10 IT11 IT12
    [   4,   6,  10,  14,  25,  40,  60, 100],  # 1-3
    [   5,   8,  12,  18,  30,  48,  75, 120],  # 3-6
    [   6,   9,  15,  22,  36,  58,  90, 150],  # 6-10
    [   8,  11,  18,  27,  43,  70, 110, 180],  # 10-18
    [   9,  13,  21,  33,  52,  84, 130, 210],  # 18-30
    [  11,  16,  25,  39,  62, 100, 160, 250],  # 30-50
    [  13,  19,  30,  46,  74, 120, 190, 300],  # 50-80
    [  15,  22,  35,  54,  87, 140, 220, 350],  # 80-120
    [  18,  25,  40,  63, 100, 160, 250, 400],  # 120-180
    [  20,  29,  46,  72, 115, 185, 290, 460],  # 180-250
    [  23,  32,  52,  81, 130, 210, 320, 520],  # 250-315
    [  25,  36,  57,  89, 140, 230, 360, 570],  # 315-400
    [  27,  40,  63,  97, 155, 250, 400, 630],  # 400-500
]

_GRADE_INDEX: dict[str, int] = {
    "IT5": 0, "IT6": 1, "IT7": 2, "IT8": 3,
    "IT9": 4, "IT10": 5, "IT11": 6, "IT12": 7,
}


def it_tolerance(nominal_d: float, grade: str) -> float:
    """Return the standard IT tolerance in mm for a given nominal diameter and grade.

    Args:
        nominal_d: Basic (nominal) dimension in mm.
        grade: IT grade string, e.g. "IT7".

    Returns:
        Tolerance value in mm (always positive).
    """
    idx = _size_range_index(nominal_d)
    col = _GRADE_INDEX.get(grade)
    if col is None:
        raise ValueError(f"Unknown IT grade '{grade}'. Valid: {list(_GRADE_INDEX)}")
    return _IT_GRADES[idx][col] / 1000.0


# ============================================================================
# Shaft fundamental deviations (µm) — ISO 286
# ============================================================================
# Positive values mean the shaft is LARGER than nominal.
# For shafts a-h: es = -|fundamental deviation| (shaft is smaller than nominal)
# For shafts k-zc: ei = +|fundamental deviation| (shaft is larger than nominal)

# es (upper deviation) for shaft letters f, g, h — in µm
_SHAFT_ES: dict[str, list[int]] = {
    #         1-3  3-6  6-10 10-18 18-30 30-50 50-80 80-120 120-180 180-250 250-315 315-400 400-500
    "f":  [   -6, -10,  -13,  -16,  -20,  -25,  -30,  -36,   -43,   -50,   -56,   -62,   -68],
    "g":  [   -2,   -4,   -5,   -6,   -7,   -9,  -10,  -12,   -14,   -15,   -17,   -18,   -20],
    "h":  [    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,     0,     0],
}

# ei (lower deviation) for shaft letters k, p — in µm
# For k: varies by IT grade. Values below are for IT grades ≤ 8.
#         For IT grades > 8, ei = 0.
_SHAFT_EI: dict[str, list[int]] = {
    #         1-3  3-6  6-10 10-18 18-30 30-50 50-80 80-120 120-180 180-250 250-315 315-400 400-500
    "k":  [    0,   +1,   +1,   +1,   +2,   +2,   +2,   +3,    +3,    +4,    +4,    +5,    +5],
    "p":  [   +6,  +12,  +15,  +18,  +22,  +26,  +32,  +37,   +43,   +50,   +56,   +62,   +68],
}

# js: symmetric, es = +IT/2, ei = -IT/2 (no lookup table needed)


def shaft_deviations(
    nominal_d: float,
    letter: str,
    grade: str,
) -> tuple[float, float]:
    """Compute shaft upper and lower deviations in mm.

    Args:
        nominal_d: Basic size in mm.
        letter: Shaft deviation letter (f, g, h, js, k, p).
        grade: IT grade string, e.g. "IT6".

    Returns:
        (es, ei) — upper and lower deviations in mm.
    """
    idx = _size_range_index(nominal_d)
    it = it_tolerance(nominal_d, grade)
    letter = letter.lower()

    if letter == "js":
        return (it / 2, -it / 2)
    elif letter in _SHAFT_ES:
        es = _SHAFT_ES[letter][idx] / 1000.0
        ei = es - it
        return (es, ei)
    elif letter in _SHAFT_EI:
        ei_val = _SHAFT_EI[letter][idx] / 1000.0
        # For 'k': ei = 0 when IT grade > 8
        if letter == "k" and _GRADE_INDEX.get(grade, 0) > 3:
            ei_val = 0.0
        es = ei_val + it
        return (es, ei_val)
    else:
        raise ValueError(
            f"Unknown shaft deviation letter '{letter}'. "
            f"Supported: f, g, h, js, k, p"
        )


# ============================================================================
# Hole fundamental deviations — H-basis system
# ============================================================================
# For holes A-H: EI (lower deviation) is positive or zero.
# For holes K-ZC: ES (upper deviation) is negative or zero.
# Holes use the "same letter, opposite sign" rule from shafts,
# with a correction Δ for transition/interference fits.

def hole_deviations(
    nominal_d: float,
    letter: str,
    grade: str,
) -> tuple[float, float]:
    """Compute hole upper and lower deviations in mm.

    Uses the general rule: hole deviation = -(shaft deviation) + Δ,
    where Δ accounts for the difference in IT grades between hole and shaft.
    For same-grade fits (most common), Δ = 0.

    Args:
        nominal_d: Basic size in mm.
        letter: Hole deviation letter (F, G, H, Js, K, P).
        grade: IT grade string, e.g. "IT7".

    Returns:
        (ES, EI) — upper and lower deviations in mm.
    """
    idx = _size_range_index(nominal_d)
    it = it_tolerance(nominal_d, grade)
    letter_upper = letter.upper()

    if letter_upper == "JS":
        return (it / 2, -it / 2)
    elif letter_upper == "H":
        return (it, 0.0)
    elif letter_upper == "G":
        # EI = -es(g), ES = EI + IT
        es_shaft = _SHAFT_ES["g"][idx] / 1000.0
        ei = -es_shaft  # positive
        es = ei + it
        return (es, ei)
    elif letter_upper == "F":
        es_shaft = _SHAFT_ES["f"][idx] / 1000.0
        ei = -es_shaft  # positive
        es = ei + it
        return (es, ei)
    elif letter_upper == "K":
        # For K: ES = -ei(k) for grades ≤ 8
        ei_shaft = _SHAFT_EI["k"][idx] / 1000.0
        if _GRADE_INDEX.get(grade, 0) > 3:
            ei_shaft = 0.0
        es = -ei_shaft
        ei = es - it
        return (es, ei)
    elif letter_upper == "P":
        ei_shaft = _SHAFT_EI["p"][idx] / 1000.0
        es = -ei_shaft
        ei = es - it
        return (es, ei)
    else:
        raise ValueError(
            f"Unknown hole deviation letter '{letter}'. "
            f"Supported: F, G, H, Js, K, P"
        )


# ============================================================================
# Fit calculation
# ============================================================================

@dataclass
class FitResult:
    """Result of a fit calculation between hole and shaft."""

    fit_type: str          # "clearance" | "transition" | "interference"
    max_clearance: float   # mm, positive = gap
    min_clearance: float   # mm, negative = interference
    max_interference: float  # mm, positive = overlap
    min_interference: float  # mm, positive = overlap, 0 for clearance fits

    # Detailed deviation values (mm)
    hole_es: float = 0.0
    hole_ei: float = 0.0
    shaft_es: float = 0.0
    shaft_ei: float = 0.0


def compute_fit(
    nominal_d: float,
    hole_grade: str,
    shaft_grade: str,
    hole_deviation: str = "H",
    shaft_deviation: str = "h",
) -> FitResult:
    """Compute the fit between a hole and shaft.

    Args:
        nominal_d: Basic (nominal) diameter in mm.
        hole_grade: IT grade for hole, e.g. "IT7".
        shaft_grade: IT grade for shaft, e.g. "IT6".
        hole_deviation: Hole deviation letter (H, G, F, Js, K, P).
        shaft_deviation: Shaft deviation letter (h, g, f, js, k, p).

    Returns:
        FitResult with fit type, clearances, and interferences.
    """
    hole_es, hole_ei = hole_deviations(nominal_d, hole_deviation, hole_grade)
    shaft_es, shaft_ei = shaft_deviations(nominal_d, shaft_deviation, shaft_grade)

    # Max clearance = hole max - shaft min
    max_clearance = hole_es - shaft_ei
    # Min clearance = hole min - shaft max
    min_clearance = hole_ei - shaft_es

    # Determine fit type
    if min_clearance >= 0:
        fit_type = "clearance"
        max_interference = 0.0
        min_interference = 0.0
    elif max_clearance <= 0:
        fit_type = "interference"
        max_interference = abs(min_clearance)
        min_interference = abs(max_clearance)
    else:
        fit_type = "transition"
        max_interference = abs(min_clearance)
        min_interference = 0.0

    return FitResult(
        fit_type=fit_type,
        max_clearance=max_clearance,
        min_clearance=min_clearance,
        max_interference=max_interference,
        min_interference=min_interference,
        hole_es=hole_es,
        hole_ei=hole_ei,
        shaft_es=shaft_es,
        shaft_ei=shaft_ei,
    )


# ============================================================================
# Fit recommendations
# ============================================================================

@dataclass
class FitRecommendation:
    """A recommended fit for a specific application."""

    code: str              # e.g. "H7/g6"
    fit_type: str          # "clearance" | "transition" | "interference"
    application: str       # e.g. "bearing_seat"
    description: str       # e.g. "Locational clearance fit for bearing outer race"
    hole_grade: str        # e.g. "IT7"
    shaft_grade: str       # e.g. "IT6"
    hole_deviation: str    # e.g. "H"
    shaft_deviation: str   # e.g. "g"


FIT_RECOMMENDATIONS: dict[str, FitRecommendation] = {
    "bearing_seat": FitRecommendation(
        code="H7/js6",
        fit_type="transition",
        application="bearing_seat",
        description="Transition fit for bearing outer race — locates bearing precisely, allows press-in assembly",
        hole_grade="IT7", shaft_grade="IT6",
        hole_deviation="H", shaft_deviation="js",
    ),
    "sliding": FitRecommendation(
        code="H8/f7",
        fit_type="clearance",
        application="sliding",
        description="Running/sliding fit — good lubrication clearance for moving parts",
        hole_grade="IT8", shaft_grade="IT7",
        hole_deviation="H", shaft_deviation="f",
    ),
    "locating": FitRecommendation(
        code="H7/g6",
        fit_type="clearance",
        application="locating",
        description="Locational clearance fit — snug fit for accurate location, free assembly",
        hole_grade="IT7", shaft_grade="IT6",
        hole_deviation="H", shaft_deviation="g",
    ),
    "snug": FitRecommendation(
        code="H7/h6",
        fit_type="clearance",
        application="snug",
        description="Locational fit — very close clearance, hand-assembly possible",
        hole_grade="IT7", shaft_grade="IT6",
        hole_deviation="H", shaft_deviation="h",
    ),
    "transition": FitRecommendation(
        code="H7/k6",
        fit_type="transition",
        application="transition",
        description="Transition fit for keyed or pinned parts — may need light press",
        hole_grade="IT7", shaft_grade="IT6",
        hole_deviation="H", shaft_deviation="k",
    ),
    "press": FitRecommendation(
        code="H7/p6",
        fit_type="interference",
        application="press",
        description="Press fit — permanent assembly, requires arbor press or thermal assembly",
        hole_grade="IT7", shaft_grade="IT6",
        hole_deviation="H", shaft_deviation="p",
    ),
}


def recommend_fit(application: str) -> FitRecommendation | None:
    """Get the recommended fit for an application.

    Args:
        application: One of "bearing_seat", "sliding", "locating",
                     "snug", "transition", "press".

    Returns:
        FitRecommendation or None if not found.
    """
    return FIT_RECOMMENDATIONS.get(application)


# ============================================================================
# Tolerance-aware dimension helpers
# ============================================================================

def tolerance_hole_diameter(
    nominal_d: float,
    application: str = "locating",
) -> tuple[float, float, float]:
    """Compute tolerance-aware hole diameter for a given application.

    Args:
        nominal_d: Basic (nominal) diameter in mm.
        application: Fit application type.

    Returns:
        (min_d, nominal_d, max_d) in mm — hole diameter range.
    """
    rec = recommend_fit(application)
    if rec is None:
        rec = FIT_RECOMMENDATIONS["locating"]

    es, ei = hole_deviations(nominal_d, rec.hole_deviation, rec.hole_grade)
    return (nominal_d + ei, nominal_d, nominal_d + es)


def tolerance_shaft_diameter(
    nominal_d: float,
    application: str = "locating",
) -> tuple[float, float, float]:
    """Compute tolerance-aware shaft diameter for a given application.

    Args:
        nominal_d: Basic (nominal) diameter in mm.
        application: Fit application type.

    Returns:
        (min_d, nominal_d, max_d) in mm — shaft diameter range.
        Note: for clearance fits (f, g, h), max_d may be less than nominal.
    """
    rec = recommend_fit(application)
    if rec is None:
        rec = FIT_RECOMMENDATIONS["locating"]

    es, ei = shaft_deviations(nominal_d, rec.shaft_deviation, rec.shaft_grade)
    return (nominal_d + ei, nominal_d, nominal_d + es)


def press_fit_bore_diameter(
    outer_d: float,
    interference: float = 0.0,
) -> tuple[float, float, float]:
    """Compute press-fit bore diameter with H7 tolerance.

    Args:
        outer_d: Outer part diameter (e.g. bearing OD) in mm.
        interference: Desired interference in mm (0 = use H7/p6 recommendation).

    Returns:
        (min_bore, nominal_bore, max_bore) in mm.
    """
    if interference > 0:
        # Custom interference: bore nominal = OD - interference
        nominal = outer_d - interference
        hole_es, hole_ei = hole_deviations(outer_d, "H", "IT7")
        return (nominal + hole_ei, nominal, nominal + hole_es)
    else:
        # Use H7/p6 recommendation
        rec = FIT_RECOMMENDATIONS["press"]
        hole_es, hole_ei = hole_deviations(outer_d, rec.hole_deviation, rec.hole_grade)
        nominal = outer_d
        return (nominal + hole_ei, nominal, nominal + hole_es)


def bearing_seat_diameter(
    bearing_od: float,
) -> tuple[float, float, float]:
    """Compute bearing seat bore diameter with H7/js6 tolerance.

    Args:
        bearing_od: Bearing outer diameter in mm.

    Returns:
        (min_bore, nominal_bore, max_bore) in mm.
    """
    return tolerance_hole_diameter(bearing_od, "bearing_seat")
