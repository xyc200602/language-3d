"""Material properties database for FEA simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Material:
    """Engineering material with mechanical properties."""

    name: str
    category: str
    youngs_modulus: float      # MPa (E)
    poissons_ratio: float      # dimensionless (nu)
    density: float             # kg/m^3 (rho)
    yield_strength: float      # MPa (sigma_y)
    ultimate_strength: float   # MPa (sigma_u)
    thermal_expansion: float   # 1e-6 /K (alpha)
    description: str = ""


# ============================================================================
# Standard Material Presets
# ============================================================================

MATERIAL_PRESETS: dict[str, Material] = {
    "steel": Material(
        name="Steel (AISI 1045)",
        category="metal",
        youngs_modulus=200_000,
        poissons_ratio=0.29,
        density=7850,
        yield_strength=350,
        ultimate_strength=565,
        thermal_expansion=12.0,
        description="Medium carbon steel, general purpose",
    ),
    "aluminum": Material(
        name="Aluminum (6061-T6)",
        category="metal",
        youngs_modulus=68_900,
        poissons_ratio=0.33,
        density=2700,
        yield_strength=276,
        ultimate_strength=310,
        thermal_expansion=23.6,
        description="Aluminum alloy, excellent machinability and weldability",
    ),
    "pla": Material(
        name="PLA (Polylactic Acid)",
        category="polymer",
        youngs_modulus=3_500,
        poissons_ratio=0.36,
        density=1240,
        yield_strength=45,
        ultimate_strength=55,
        thermal_expansion=68.0,
        description="PLA 3D printing filament (FDM-conservative), biodegradable",
    ),
    "abs": Material(
        name="ABS (Acrylonitrile Butadiene Styrene)",
        category="polymer",
        youngs_modulus=2_300,
        poissons_ratio=0.35,
        density=1040,
        yield_strength=30,
        ultimate_strength=38,
        thermal_expansion=90.0,
        description="ABS 3D printing filament (FDM-conservative), impact resistant",
    ),
    "titanium": Material(
        name="Titanium (Ti-6Al-4V)",
        category="metal",
        youngs_modulus=113_800,
        poissons_ratio=0.34,
        density=4430,
        yield_strength=880,
        ultimate_strength=950,
        thermal_expansion=8.6,
        description="Titanium alloy, high strength-to-weight ratio",
    ),
    "copper": Material(
        name="Copper (C11000)",
        category="metal",
        youngs_modulus=117_000,
        poissons_ratio=0.34,
        density=8960,
        yield_strength=70,
        ultimate_strength=220,
        thermal_expansion=16.5,
        description="Pure copper, excellent thermal/electrical conductivity",
    ),
}

# ============================================================================
# Safety Factors by Application
# ============================================================================

SAFETY_FACTORS: dict[str, float] = {
    "static": 1.5,
    "dynamic": 2.0,
    "impact": 3.0,
    "3d_printing": 2.5,
}


def get_material(name: str) -> Material | None:
    """Get a material by name (case-insensitive lookup).

    Checks exact match first, then prefix match.
    """
    key = name.lower().strip()
    if key in MATERIAL_PRESETS:
        return MATERIAL_PRESETS[key]
    # Alias lookups
    aliases = {
        "aluminium": "aluminum",
        "al": "aluminum",
        "ti": "titanium",
        "ti6al4v": "titanium",
    }
    if key in aliases:
        return MATERIAL_PRESETS.get(aliases[key])
    return None


def compute_safety_factor(material: Material, max_stress: float) -> float:
    """Compute the safety factor given a material and maximum von Mises stress.

    Returns the ratio yield_strength / max_stress.
    A value < 1.0 indicates yielding (unsafe).
    """
    if max_stress <= 0:
        return float("inf")
    return material.yield_strength / max_stress
