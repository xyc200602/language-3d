"""Parametric wheeled mobile-base topology generator.

The wheeled-base counterpart to ``arm_topology.build_arm_example``. Instead
of feeding the LLM the hand-written ``EXAMPLE_4W_ROBOT`` constant (whose
hard-coded wheel/motor dimensions never adapt to payload, and which the
LLM "creatively" mutates — breaking the axis/anchor/no_distribute
conventions that keep wheels flat on the ground), this module synthesises a
structurally-correct chassis from high-level parameters.

Why this matters (root cause of the dual-arm failures):
  * ``assembly_solver`` is correct — a static ``EXAMPLE_4W_ROBOT`` solve
    places all 4 wheels at Z=-15 with 0.0mm variation.
  * But the LLM, given the example as *advisory text*, changes the wheel
    joint axis to "z" (wheel stands vertical), drops ``no_distribute`` (the
    sibling-distribution algorithm flings wheels 600mm from centre), or
    re-parents wheels onto the arm chain. Every failure mode traces back to
    the LLM not preserving the conventions.
  * A parametric generator (like ``arm_topology`` already proved for arms)
    makes those conventions *structural* — the LLM adjusts dimensions, not
    topology.

Convention pins (must match ``EXAMPLE_4W_ROBOT`` exactly, validated by the
solver producing flat, symmetric wheels):
  * wheel joint: ``revolute, axis="y", parent_anchor="left"/"right",
    child_anchor="center", no_distribute=True`` — axis=y makes the cylinder
    lie horizontally (its spin axis is Y, so it rolls forward/back).
  * motors: ``fixed, parent=base_plate, parent_anchor="bottom",
    distribution_group="motors"`` — the solver's 2×2 grid distribution puts
    them at the four corners of the base underside.
  * standoffs: ``fixed, parent=base_plate, parent_anchor="top",
    distribution_group="standoffs"`` — same grid, on top, holding the
    top_plate up.
  * top_plate: the arm-mounting deck. Its height = standoff height + base
    plate thickness, so arms sit clear of the wheels.

Sizes come from ``WheelBaseCalculator`` (payload → wheel diameter, track
width, ground clearance) and ``DC_MOTOR_CATALOG`` (real motor dimensions),
so a 20kg-payload robot gets bigger wheels than a 2kg one — not the flat
65mm of the hard-coded example.
"""

from __future__ import annotations

import json
from typing import Any

from .mobile_base import WheelBaseCalculator, DC_MOTOR_CATALOG


# ---------------------------------------------------------------------------
# Corner naming convention
# ---------------------------------------------------------------------------
#
# Four-corner layout, viewed from above (X right, Y forward):
#
#        rl (−X, +Y)        rr (+X, +Y)
#               ┌─────────────┐
#               │  base_plate │
#               └─────────────┘
#        fl (−X, −Y)        fr (+X, −Y)
#
# fl/fr/rl/rr = front-left / front-right / rear-left / rear-right.
# The solver's 2x2 distribution assigns corners by joint *order*
# (assembly_solver._compute_distribution_offset: row=idx//2, col=idx%2),
# so the order we emit joints in determines their physical placement.
# We emit in the order fl, fr, rl, rr which the solver maps to a clockwise
# corner sweep — matching the real robot layout.
_CORNERS = ["fl", "fr", "rl", "rr"]


# ---------------------------------------------------------------------------
# Dimension derivation (delegates to WheelBaseCalculator)
# ---------------------------------------------------------------------------


def _chassis_dims(payload_kg: float, drive_type: str) -> dict[str, float]:
    """Derive chassis + wheel dimensions from payload via WheelBaseCalculator."""
    calc = WheelBaseCalculator(payload_kg=payload_kg, drive_type=drive_type)
    wheel_d = calc.wheel_diameter_mm()
    track = calc.track_width_mm()        # left↔right (X)
    wheelbase = calc.wheelbase_mm()      # front↔rear (Y)
    clearance = calc.ground_clearance_mm()
    # base plate sits above the wheels by (clearance + half wheel + motor height).
    # We keep EXAMPLE_4W_ROBOT's layered structure: base_plate (thin) on top of
    # motors, standoffs lift the top_plate above base_plate.
    base_thickness = max(5.0, payload_kg * 0.4)  # heavier payload → thicker deck
    standoff_height = max(40.0, wheel_d * 0.8)   # arms must clear the wheels
    return {
        "wheel_diameter": wheel_d,
        "wheel_width": max(20.0, wheel_d * 0.35),
        "track_width": track,
        "wheelbase": wheelbase,
        "base_length": wheelbase,   # solver Y extent (front↔rear)
        "base_width": track,        # solver X extent (left↔right)
        "base_thickness": base_thickness,
        "standoff_height": standoff_height,
        "ground_clearance": clearance,
    }


def _motor_dims(payload_kg: float) -> dict[str, float]:
    """Pick a real DC motor from the catalog based on payload."""
    # Choose by payload tier — heavier robot needs more torque.
    candidates = list(DC_MOTOR_CATALOG.values())
    # Sort by stall torque descending, pick first that handles the load.
    # Fallback to a generic 40×30×25 if catalog lookup fails.
    chosen = None
    for m in sorted(candidates, key=lambda m: getattr(m, "stall_torque_nm", 0), reverse=True):
        if getattr(m, "stall_torque_nm", 0) >= payload_kg * 0.3:
            chosen = m
            break
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is not None:
        return {
            "length": getattr(chosen, "body_length_mm", 40.0),
            "width": getattr(chosen, "body_width_mm", 30.0),
            "height": getattr(chosen, "body_height_mm", 25.0),
        }
    return {"length": 40.0, "width": 30.0, "height": 25.0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_wheeled_base(
    wheel_count: int = 4,
    drive_type: str = "differential",
    payload_kg: float = 5.0,
    arm_mount_points: list[str] | None = None,
) -> str:
    """Build a complete wheeled-base assembly-JSON string.

    Parameters mirror how a real chassis is specified: how many wheels,
    what drive kinematics, and the payload it must carry. Dimensions are
    derived from ``WheelBaseCalculator`` so a heavier robot gets larger
    wheels and a wider stance — the key gap in the hard-coded example.

    Returns JSON text matching the ``EXAMPLE_4W_ROBOT`` schema (same keys,
    same joint conventions) so the downstream solver / cad / export need
    **zero** changes. Wheels are guaranteed flat and symmetric because the
    axis/anchor/no_distribute conventions are baked into the generator, not
    left to the LLM.

    ``arm_mount_points`` names the sides of the top_plate where arms will
    attach (consumed later by ``assembly_compose.compose_dual_arm_assembly``).
    """
    wheel_count = max(2, min(int(wheel_count), 4))
    if arm_mount_points is None:
        arm_mount_points = ["left", "right"]

    dims = _chassis_dims(payload_kg, drive_type)
    motor = _motor_dims(payload_kg)
    corners = _CORNERS[:wheel_count]

    parts: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []

    # --- base_plate (the main deck) ---
    parts.append({
        "name": "base_plate", "category": "structural",
        "description": "主底盘板", "material": "Aluminum",
        "dimensions": {
            "length": dims["base_length"], "width": dims["base_width"],
            "height": dims["base_thickness"],
        },
    })

    # --- motors (under the base, one per wheel) ---
    # distribution_group="motors" → solver 2x2 grid puts them at corners.
    # parent_anchor="bottom" so they hang beneath the deck.
    for corner in corners:
        parts.append({
            "name": f"motor_{corner}", "category": "actuator",
            "description": f"{corner}驱动电机", "material": "Steel",
            "dimensions": dict(motor),
        })
        joints.append({
            "type": "fixed", "parent": "base_plate", "child": f"motor_{corner}",
            "parent_anchor": "bottom", "child_anchor": "top",
            "distribution_group": "motors",
        })

    # --- wheels (the part that was always broken) ---
    # THE critical convention: axis="y" (cylinder lies horizontal, rolls
    # forward), child_anchor="center" (wheel centre on motor side face),
    # no_distribute=True (do NOT let the sibling-grid move it — its position
    # is fixed relative to its parent motor). Left-side motors get
    # parent_anchor="left", right-side "right", so wheels poke outward.
    wheel_dims = {"diameter": dims["wheel_diameter"], "height": dims["wheel_width"]}
    for corner in corners:
        side = "left" if corner in ("fl", "rl") else "right"
        parts.append({
            "name": f"wheel_{corner}", "category": "mechanical",
            "description": f"{corner}轮", "material": "Rubber",
            "dimensions": dict(wheel_dims),
        })
        joints.append({
            "type": "revolute", "parent": f"motor_{corner}", "child": f"wheel_{corner}",
            "axis": "y", "range_deg": [-360, 360],
            "parent_anchor": side, "child_anchor": "center",
            "no_distribute": True,
        })

    # --- standoffs + top_plate (arm-mounting deck) ---
    # Standoffs lift the top_plate above the base so arms clear the wheels.
    standoff_dims = {"diameter": 6, "height": dims["standoff_height"], "length": 8}
    for corner in corners:
        parts.append({
            "name": f"standoff_{corner}", "category": "structural",
            "description": f"{corner}铜柱", "material": "Steel",
            "dimensions": dict(standoff_dims),
        })
        joints.append({
            "type": "fixed", "parent": "base_plate", "child": f"standoff_{corner}",
            "parent_anchor": "top", "child_anchor": "bottom",
            "distribution_group": "standoffs",
        })
    # top_plate sits ABOVE the standoffs, centred on the base. Mounting it on
    # a single corner standoff (the old EXAMPLE_4W_ROBOT convention) offset
    # the deck to one corner, which made dual arms sit asymmetrically. Mount
    # on base_plate centre with a Z offset = standoff height so the deck
    # clears the wheels and stays centred (arms then mirror symmetrically).
    parts.append({
        "name": "top_plate", "category": "structural",
        "description": "上盖板（机械臂安装面）", "material": "Aluminum",
        "dimensions": {
            "length": dims["base_length"] * 0.9, "width": dims["base_width"] * 0.9,
            "height": 3.0,
        },
    })
    joints.append({
        "type": "fixed", "parent": "base_plate", "child": "top_plate",
        "parent_anchor": "top", "child_anchor": "bottom",
        "offset": [0.0, 0.0, dims["standoff_height"]],
        "no_distribute": True,
    })

    # --- battery_box (representative payload, mounted on base top) ---
    parts.append({
        "name": "battery_box", "category": "electronics",
        "description": "电池盒", "material": "PLA",
        "dimensions": {
            "length": dims["base_length"] * 0.5, "width": dims["base_width"] * 0.3,
            "height": max(25.0, dims["base_thickness"] * 3),
        },
    })
    joints.append({
        "type": "fixed", "parent": "base_plate", "child": "battery_box",
        "parent_anchor": "top", "child_anchor": "bottom", "no_distribute": True,
    })

    assembly = {
        "name": f"{wheel_count}w_{drive_type}_base",
        "description": f"{wheel_count}轮{drive_type}移动底盘",
        "default_angles": {},
        "parts": parts,
        "joints": joints,
        # Metadata for the composer: where arms should attach.
        "_arm_mount": {
            "deck": "top_plate",
            "deck_height_mm": dims["standoff_height"] + dims["base_thickness"],
            "sides": arm_mount_points,
            "offset_mm": dims["base_width"] * 0.35,  # left/right spread
        },
    }
    return json.dumps(assembly, ensure_ascii=False, indent=2)


def parse_drive_type(description: str) -> str:
    """Infer the drive type from a natural-language description.

    Returns 'mecanum' / 'omnidirectional' / 'differential'. Defaults to
    'differential' (the most common and simplest).
    """
    if not description:
        return "differential"
    t = description.lower()
    if "麦克纳姆" in description or "mecanum" in t or "全向" in description:
        return "mecanum"
    if "omni" in t:
        return "mecanum"  # treat omni as mecanum kinematically
    return "differential"
