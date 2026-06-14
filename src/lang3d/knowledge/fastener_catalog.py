"""ISO/DIN standard fastener catalog — authoritative dimension database.

Consolidates all standard fastener specifications in one place:
- Socket head cap screws (DIN 912 / ISO 4762)
- Hex nuts (DIN 934 / ISO 4032)
- Flat washers (DIN 125 / ISO 7089)
- Spring washers (DIN 127)
- Heat-set thread inserts (brass, for 3D printing)
- Set screws / grub screws (DIN 916)

All dimensions in millimeters.  Data sourced from ISO/DIN standards and
verified against manufacturer datasheets (Bossard, Würth, McMaster-Carr).

This module is the single source of truth — other modules (connection_features,
part_library, assembly_generator) should import from here.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class BoltSpec:
    """ISO metric bolt dimensions."""

    size: str                # e.g. "M3"
    thread_diameter: float   # Nominal thread diameter (mm)
    pitch: float             # Thread pitch (mm), coarse series
    head_diameter: float     # Socket head cap screw head Ø (mm)
    head_height: float       # Head height (mm)
    socket_width: float      # Hex socket width across flats (mm)
    socket_depth: float      # Hex socket depth (mm)
    # DIN 912 / ISO 4762 values:
    # head_diameter ≈ 1.5×thread_d
    # head_height ≈ thread_d
    # socket_width ≈ 0.8×thread_d (standard hex key size)

    @property
    def thread_radius(self) -> float:
        return self.thread_diameter / 2

    @property
    def head_radius(self) -> float:
        return self.head_diameter / 2


@dataclass(frozen=True)
class NutSpec:
    """ISO metric hex nut dimensions."""

    size: str                # e.g. "M3"
    thread_diameter: float   # Nominal thread diameter (mm)
    pitch: float             # Thread pitch (mm)
    width_across_flats: float  # Wrench size / hex width (mm)
    height: float            # Nut height (mm)
    width_across_corners: float  # Point-to-point hex width (mm)

    @property
    def outer_radius(self) -> float:
        """Circumscribed circle radius (for cylindrical approximation)."""
        return self.width_across_corners / 2


@dataclass(frozen=True)
class WasherSpec:
    """ISO metric flat washer dimensions."""

    size: str                # e.g. "M3"
    bolt_diameter: float     # Nominal bolt diameter (mm)
    inner_diameter: float    # Hole diameter (mm)
    outer_diameter: float    # Outer diameter (mm)
    thickness: float         # Thickness (mm)

    @property
    def outer_radius(self) -> float:
        return self.outer_diameter / 2

    @property
    def inner_radius(self) -> float:
        return self.inner_diameter / 2


@dataclass(frozen=True)
class SpringWasherSpec:
    """DIN 127 spring (split lock) washer dimensions."""

    size: str
    bolt_diameter: float
    inner_diameter: float
    outer_diameter: float
    thickness: float         # Material thickness (mm)
    free_height: float       # Uncompressed height (mm)


@dataclass(frozen=True)
class ThreadInsertSpec:
    """Heat-set brass thread insert for 3D printing.

    Commonly used brands: Ruthex, Gridfinity, CNC Kitchen.
    Installed by heating with soldering iron → melts into plastic.
    """

    size: str                # e.g. "M3"
    thread_diameter: float   # Internal thread diameter (mm)
    outer_diameter: float    # Knurled outer diameter (mm)
    length: float            # Insert length (mm)
    install_hole_diameter: float  # Hole to drill in plastic (mm)
    min_wall_thickness: float     # Minimum surrounding plastic (mm)


@dataclass(frozen=True)
class SetScrewSpec:
    """DIN 916 hex socket set screw (grub screw) dimensions."""

    size: str
    thread_diameter: float
    pitch: float
    socket_width: float      # Hex key size (mm)
    socket_depth: float      # Usable hex depth (mm)


@dataclass(frozen=True)
class DowelPinSpec:
    """ISO 8734 parallel dowel pin dimensions.

    Dowel pins provide precise alignment between two parts.
    Slip fit (H7) in one part, press fit in the other.
    """

    size: str              # e.g. "D5x20"
    diameter: float        # Nominal pin diameter (mm)
    length: float          # Pin length (mm)
    tolerance: str = "m6"  # Pin tolerance class (ISO 8734 = m6)

    @property
    def hole_diameter_slip(self) -> float:
        """H7 slip-fit hole diameter: nominal + 0.01mm clearance."""
        return self.diameter + 0.01

    @property
    def hole_diameter_press(self) -> float:
        """Press-fit hole diameter: nominal - 0.01mm interference."""
        return self.diameter - 0.01


@dataclass(frozen=True)
class ClearanceHoleSpec:
    """ISO 273 clearance hole dimensions for a bolt size."""

    size: str
    normal: float            # Normal fit clearance diameter (mm)
    close: float             # Close fit
    loose: float             # Loose fit


@dataclass(frozen=True)
class TapHoleSpec:
    """Tap drill hole diameter for cutting internal threads."""

    size: str
    tap_diameter: float      # Drill bit diameter for tapping (mm)
    thread_diameter: float   # Resulting thread diameter (mm)


# ============================================================================
# Standard dimension tables
# ============================================================================


# Socket head cap screws — DIN 912 / ISO 4762
BOLT_SPECS: dict[str, BoltSpec] = {
    "M1.6": BoltSpec("M1.6", 1.6, 0.35, 3.00, 1.60, 1.25, 0.85),
    "M2":   BoltSpec("M2",   2.0, 0.40, 3.80, 2.00, 1.5, 1.0),
    "M2.5": BoltSpec("M2.5", 2.5, 0.45, 4.50, 2.50, 2.0, 1.2),
    "M3":   BoltSpec("M3",   3.0, 0.50, 5.50, 3.00, 2.5, 1.3),
    "M4":   BoltSpec("M4",   4.0, 0.70, 7.00, 4.00, 3.0, 2.0),
    "M5":   BoltSpec("M5",   5.0, 0.80, 8.50, 5.00, 4.0, 2.5),
    "M6":   BoltSpec("M6",   6.0, 1.00, 10.00, 6.00, 5.0, 3.0),
    "M8":   BoltSpec("M8",   8.0, 1.25, 13.00, 8.00, 6.0, 4.0),
    "M10":  BoltSpec("M10", 10.0, 1.50, 16.00, 10.00, 8.0, 5.0),
    "M12":  BoltSpec("M12", 12.0, 1.75, 18.00, 12.00, 10.0, 6.0),
}

# Standard bolt lengths (mm) — preferred series per ISO 888
STANDARD_BOLT_LENGTHS: list[float] = [
    4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 80, 90, 100,
]

# Hex nuts — DIN 934 / ISO 4032
NUT_SPECS: dict[str, NutSpec] = {
    "M2":   NutSpec("M2",   2.0, 0.40, 4.0,  1.6, 4.6),
    "M2.5": NutSpec("M2.5", 2.5, 0.45, 5.0,  2.0, 5.8),
    "M3":   NutSpec("M3",   3.0, 0.50, 5.5,  2.4, 6.4),
    "M4":   NutSpec("M4",   4.0, 0.70, 7.0,  3.2, 8.1),
    "M5":   NutSpec("M5",   5.0, 0.80, 8.0,  4.7, 9.2),
    "M6":   NutSpec("M6",   6.0, 1.00, 10.0, 5.2, 11.5),
    "M8":   NutSpec("M8",   8.0, 1.25, 13.0, 6.8, 15.0),
    "M10":  NutSpec("M10", 10.0, 1.50, 17.0, 8.4, 19.6),
    "M12":  NutSpec("M12", 12.0, 1.75, 19.0, 10.8, 21.9),
}

# Flat washers — DIN 125 / ISO 7089 (normal series)
WASHER_SPECS: dict[str, WasherSpec] = {
    "M2":   WasherSpec("M2",   2.0, 2.2, 5.0,  0.3),
    "M2.5": WasherSpec("M2.5", 2.5, 2.7, 6.0,  0.5),
    "M3":   WasherSpec("M3",   3.0, 3.2, 7.0,  0.5),
    "M4":   WasherSpec("M4",   4.0, 4.3, 9.0,  0.8),
    "M5":   WasherSpec("M5",   5.0, 5.3, 10.0, 1.0),
    "M6":   WasherSpec("M6",   6.0, 6.4, 12.0, 1.6),
    "M8":   WasherSpec("M8",   8.0, 8.4, 16.0, 1.6),
    "M10":  WasherSpec("M10", 10.0, 10.5, 20.0, 2.0),
    "M12":  WasherSpec("M12", 12.0, 13.0, 24.0, 2.5),
}

# Spring washers (split lock washers) — DIN 127
SPRING_WASHER_SPECS: dict[str, SpringWasherSpec] = {
    "M2":   SpringWasherSpec("M2",   2.0, 2.1, 4.2, 0.5, 1.0),
    "M2.5": SpringWasherSpec("M2.5", 2.5, 2.6, 5.2, 0.6, 1.2),
    "M3":   SpringWasherSpec("M3",   3.0, 3.1, 6.2, 0.8, 1.6),
    "M4":   SpringWasherSpec("M4",   4.0, 4.1, 8.2, 1.0, 2.0),
    "M5":   SpringWasherSpec("M5",   5.0, 5.1, 10.2, 1.2, 2.4),
    "M6":   SpringWasherSpec("M6",   6.0, 6.2, 12.2, 1.6, 3.2),
    "M8":   SpringWasherSpec("M8",   8.0, 8.2, 16.3, 2.0, 4.0),
    "M10":  SpringWasherSpec("M10", 10.0, 10.2, 20.5, 2.5, 5.0),
    "M12":  SpringWasherSpec("M12", 12.0, 12.3, 24.5, 3.0, 6.0),
}

# Heat-set brass thread inserts for 3D printing
THREAD_INSERT_SPECS: dict[str, ThreadInsertSpec] = {
    "M2":   ThreadInsertSpec("M2",   2.0, 3.5,  4.0, 3.6, 1.5),
    "M2.5": ThreadInsertSpec("M2.5", 2.5, 4.2,  5.0, 4.3, 1.8),
    "M3":   ThreadInsertSpec("M3",   3.0, 4.6,  5.6, 4.7, 2.0),
    "M4":   ThreadInsertSpec("M4",   4.0, 6.0,  7.0, 6.1, 2.5),
    "M5":   ThreadInsertSpec("M5",   5.0, 7.1,  8.0, 7.1, 3.0),
    "M6":   ThreadInsertSpec("M6",   6.0, 8.3, 10.0, 8.4, 3.5),
}

# Set screws / grub screws — DIN 916
SET_SCREW_SPECS: dict[str, SetScrewSpec] = {
    "M2":   SetScrewSpec("M2",   2.0, 0.40, 1.5, 1.0),
    "M2.5": SetScrewSpec("M2.5", 2.5, 0.45, 2.0, 1.2),
    "M3":   SetScrewSpec("M3",   3.0, 0.50, 2.5, 1.3),
    "M4":   SetScrewSpec("M4",   4.0, 0.70, 3.0, 2.0),
    "M5":   SetScrewSpec("M5",   5.0, 0.80, 4.0, 2.5),
    "M6":   SetScrewSpec("M6",   6.0, 1.00, 5.0, 3.0),
    "M8":   SetScrewSpec("M8",   8.0, 1.25, 6.0, 4.0),
}

# Dowel pins — ISO 8734 (parallel pins, hardened steel, m6 tolerance)
DOWEL_PIN_SPECS: dict[str, DowelPinSpec] = {
    "D3x10": DowelPinSpec("D3x10", 3.0, 10.0),
    "D3x16": DowelPinSpec("D3x16", 3.0, 16.0),
    "D3x20": DowelPinSpec("D3x20", 3.0, 20.0),
    "D4x16": DowelPinSpec("D4x16", 4.0, 16.0),
    "D4x20": DowelPinSpec("D4x20", 4.0, 20.0),
    "D4x25": DowelPinSpec("D4x25", 4.0, 25.0),
    "D5x20": DowelPinSpec("D5x20", 5.0, 20.0),
    "D5x25": DowelPinSpec("D5x25", 5.0, 25.0),
    "D5x30": DowelPinSpec("D5x30", 5.0, 30.0),
    "D6x25": DowelPinSpec("D6x25", 6.0, 25.0),
    "D6x30": DowelPinSpec("D6x30", 6.0, 30.0),
    "D6x40": DowelPinSpec("D6x40", 6.0, 40.0),
    "D8x30": DowelPinSpec("D8x30", 8.0, 30.0),
    "D8x40": DowelPinSpec("D8x40", 8.0, 40.0),
    "D10x40": DowelPinSpec("D10x40", 10.0, 40.0),
}

# Spring pins (roll pins) — ISO 8752
SPRING_PIN_SPECS: dict[str, DowelPinSpec] = {
    "SP3x12": DowelPinSpec("SP3x12", 3.0, 12.0, tolerance="spring"),
    "SP3x18": DowelPinSpec("SP3x18", 3.0, 18.0, tolerance="spring"),
    "SP4x16": DowelPinSpec("SP4x16", 4.0, 16.0, tolerance="spring"),
    "SP5x20": DowelPinSpec("SP5x20", 5.0, 20.0, tolerance="spring"),
    "SP6x25": DowelPinSpec("SP6x25", 6.0, 25.0, tolerance="spring"),
}

# Clearance holes — ISO 273
CLEARANCE_HOLE_SPECS: dict[str, ClearanceHoleSpec] = {
    "M1.6": ClearanceHoleSpec("M1.6", 2.0, 1.85, 2.2),
    "M2":   ClearanceHoleSpec("M2",   2.4, 2.2, 2.8),
    "M2.5": ClearanceHoleSpec("M2.5", 2.9, 2.7, 3.3),
    "M3":   ClearanceHoleSpec("M3",   3.4, 3.2, 3.9),
    "M4":   ClearanceHoleSpec("M4",   4.5, 4.3, 5.0),
    "M5":   ClearanceHoleSpec("M5",   5.5, 5.3, 6.0),
    "M6":   ClearanceHoleSpec("M6",   6.6, 6.4, 7.0),
    "M8":   ClearanceHoleSpec("M8",   9.0, 8.4, 10.0),
    "M10":  ClearanceHoleSpec("M10", 11.0, 10.5, 12.0),
    "M12":  ClearanceHoleSpec("M12", 13.5, 13.0, 14.5),
}

# Tap drill holes for threading into plastic
TAP_HOLE_SPECS: dict[str, TapHoleSpec] = {
    "M2":   TapHoleSpec("M2",   1.6, 2.0),
    "M2.5": TapHoleSpec("M2.5", 2.05, 2.5),
    "M3":   TapHoleSpec("M3",   2.5, 3.0),
    "M4":   TapHoleSpec("M4",   3.3, 4.0),
    "M5":   TapHoleSpec("M5",   4.2, 5.0),
    "M6":   TapHoleSpec("M6",   5.0, 6.0),
    "M8":   TapHoleSpec("M8",   6.8, 8.0),
    "M10":  TapHoleSpec("M10",  8.5, 10.0),
    "M12":  TapHoleSpec("M12", 10.2, 12.0),
}

# Recommended torque (N·m) — steel bolts into PLA/ABS (3D printed parts)
TORQUE_PLA: dict[str, float] = {
    "M2": 0.10, "M2.5": 0.20, "M3": 0.30, "M4": 0.70,
    "M5": 1.40, "M6": 2.30, "M8": 5.00, "M10": 10.0, "M12": 15.0,
}

# Recommended torque (N·m) — steel bolts into steel/aluminum
TORQUE_STEEL: dict[str, float] = {
    "M2": 0.25, "M2.5": 0.50, "M3": 1.00, "M4": 2.50,
    "M5": 5.00, "M6": 8.50, "M8": 20.0, "M10": 40.0, "M12": 65.0,
}


# ============================================================================
# Query functions
# ============================================================================


def get_bolt_spec(size: str) -> BoltSpec | None:
    """Get bolt specification by size (e.g. 'M3')."""
    return BOLT_SPECS.get(size)


def get_nut_spec(size: str) -> NutSpec | None:
    """Get nut specification by size."""
    return NUT_SPECS.get(size)


def get_washer_spec(size: str) -> WasherSpec | None:
    """Get flat washer specification by size."""
    return WASHER_SPECS.get(size)


def get_spring_washer_spec(size: str) -> SpringWasherSpec | None:
    """Get spring washer specification by size."""
    return SPRING_WASHER_SPECS.get(size)


def get_thread_insert_spec(size: str) -> ThreadInsertSpec | None:
    """Get heat-set thread insert specification by size."""
    return THREAD_INSERT_SPECS.get(size)


def get_set_screw_spec(size: str) -> SetScrewSpec | None:
    """Get set screw specification by size."""
    return SET_SCREW_SPECS.get(size)


def get_dowel_pin_spec(size: str) -> DowelPinSpec | None:
    """Get dowel pin specification by size string (e.g. 'D5x20')."""
    return DOWEL_PIN_SPECS.get(size)


def get_spring_pin_spec(size: str) -> DowelPinSpec | None:
    """Get spring pin specification by size string (e.g. 'SP5x20')."""
    return SPRING_PIN_SPECS.get(size)


def recommend_dowel_pin(
    alignment_accuracy: str = "normal",
    plate_thickness: float = 10.0,
) -> DowelPinSpec | None:
    """Recommend a dowel pin based on alignment accuracy and plate thickness.

    Args:
        alignment_accuracy: "coarse", "normal", or "fine"
        plate_thickness: Minimum engagement depth in mm

    Returns:
        Recommended DowelPinSpec, or None.
    """
    if alignment_accuracy == "fine":
        preferred = ["D6x30", "D6x25", "D5x25", "D5x20", "D4x20"]
    elif alignment_accuracy == "coarse":
        preferred = ["D8x30", "D6x25", "D5x20", "D4x16", "D3x16"]
    else:  # normal
        preferred = ["D5x20", "D5x25", "D6x25", "D4x20", "D3x20"]

    for size in preferred:
        spec = DOWEL_PIN_SPECS.get(size)
        if spec and spec.length >= plate_thickness:
            return spec
    # Fallback: first pin that fits
    for spec in DOWEL_PIN_SPECS.values():
        if spec.length >= plate_thickness:
            return spec
    return None


def get_clearance_hole(size: str, fit: str = "normal") -> float:
    """Get clearance hole diameter for a bolt size.

    Args:
        size: Bolt size, e.g. 'M3'
        fit: 'close', 'normal', or 'loose'

    Returns:
        Hole diameter in mm, or 0 if unknown.
    """
    spec = CLEARANCE_HOLE_SPECS.get(size)
    if spec is None:
        try:
            d = float(size.replace("M", ""))
            return d + 0.4
        except ValueError:
            return 0.0
    return {"close": spec.close, "normal": spec.normal, "loose": spec.loose}[fit]


def get_tap_hole(size: str) -> float:
    """Get tap drill diameter for cutting internal threads."""
    spec = TAP_HOLE_SPECS.get(size)
    return spec.tap_diameter if spec else 0.0


def get_torque(size: str, material: str = "PLA") -> float:
    """Get recommended tightening torque in N·m."""
    mat = material.lower()
    if mat in ("steel", "stainlesssteel", "aluminum", "aluminium"):
        return TORQUE_STEEL.get(size, 1.0)
    return TORQUE_PLA.get(size, 0.3)


def recommend_bolt_length(grip_mm: float) -> float:
    """Select the shortest standard bolt length >= grip.

    Args:
        grip_mm: Total material thickness the bolt must pass through
                 (plate_a + plate_b + washer + nut engagement).

    Returns:
        Standard bolt length in mm.
    """
    for length in STANDARD_BOLT_LENGTHS:
        if length >= grip_mm:
            return float(length)
    # Beyond standard range: round up to 10mm increments
    return float(int(grip_mm / 10 + 1) * 10)


def recommend_fastener_set(
    bolt_size: str,
    grip_mm: float,
    *,
    with_washer: bool = True,
    with_spring_washer: bool = False,
    use_thread_insert: bool = False,
) -> dict | None:
    """Recommend a complete fastener set for a given bolt size and grip.

    Returns:
        dict with bolt_length, nut_size, washer specs, thread_insert specs,
        and total count of parts, or None if bolt_size is unknown.
    """
    bolt = get_bolt_spec(bolt_size)
    if bolt is None:
        return None

    washer = get_washer_spec(bolt_size) if with_washer else None
    spring = get_spring_washer_spec(bolt_size) if with_spring_washer else None
    insert = get_thread_insert_spec(bolt_size) if use_thread_insert else None

    # Calculate bolt length: grip + washer(s) + nut engagement
    extra = 0.0
    if washer:
        extra += washer.thickness
    if spring:
        extra += spring.thickness
    if insert:
        extra += insert.length * 0.3  # partial engagement
    else:
        nut = get_nut_spec(bolt_size)
        if nut:
            extra += nut.height

    bolt_length = recommend_bolt_length(grip_mm + extra)

    result: dict = {
        "bolt_size": bolt_size,
        "bolt_length_mm": bolt_length,
        "bolt_spec": {
            "thread_diameter": bolt.thread_diameter,
            "head_diameter": bolt.head_diameter,
            "head_height": bolt.head_height,
        },
        "torque_nm": get_torque(bolt_size, "PLA"),
        "clearance_hole_mm": get_clearance_hole(bolt_size),
        "tap_hole_mm": get_tap_hole(bolt_size),
        "parts_count": 2,  # bolt + nut
    }

    if washer:
        result["washer"] = {
            "inner_diameter": washer.inner_diameter,
            "outer_diameter": washer.outer_diameter,
            "thickness": washer.thickness,
        }
        result["parts_count"] += 1

    if spring:
        result["spring_washer"] = {
            "inner_diameter": spring.inner_diameter,
            "outer_diameter": spring.outer_diameter,
            "thickness": spring.thickness,
        }
        result["parts_count"] += 1

    if insert:
        result["thread_insert"] = {
            "outer_diameter": insert.outer_diameter,
            "length": insert.length,
            "install_hole_diameter": insert.install_hole_diameter,
        }
        result["parts_count"] += 1

    return result


def list_available_sizes() -> list[str]:
    """List all available metric bolt sizes."""
    return list(BOLT_SPECS.keys())


def validate_size(size: str) -> bool:
    """Check if a metric bolt size is in the catalog."""
    return size in BOLT_SPECS
