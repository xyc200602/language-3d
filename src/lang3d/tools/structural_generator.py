"""Structural part auto-generator — generate custom parts from interface requirements.

Given a set of functional parts (motors, bearings, servos) that need to be mounted,
this module automatically generates a structural part template with:
- Correct bounding box geometry
- Matching mounting holes/features on the appropriate faces
- Parametric FreeCAD script
- Composite MountingInterface

Typical usage::

    bracket = generate_structural_part(
        name="custom_motor_bracket",
        interfaces=[
            InterfaceRequirement("nema17_stepper", anchor="front"),
            InterfaceRequirement("bearing_608", anchor="back"),
        ],
        part_type="bracket",
    )
    # bracket.fc_script_template → complete FreeCAD script
    # bracket.mounting_interface → all holes from both sides
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.parts_catalog import (
    BoltHole,
    MountingInterface,
    ParamDef,
    PartTemplate,
    get_mounting_interface,
    PART_CATALOG,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InterfaceRequirement:
    """Specification for one functional part that needs to be mounted."""

    functional_part_id: str    # e.g. "nema17_stepper"
    anchor: str                # "front" / "back" / "top" / "bottom" / "left" / "right"
    offset: tuple[float, float, float] = (0, 0, 0)
    rotation_deg: float = 0.0


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------


def _infer_shape(anchors: list[str]) -> str:
    """Infer the structural shape from the set of anchors.

    - 1 face → plate
    - 2 opposite faces → housing (box)
    - 2 adjacent faces → bracket (L-shape)
    - 3+ faces → housing
    """
    if len(anchors) <= 1:
        return "plate"

    opposite_pairs = {
        frozenset({"front", "back"}),
        frozenset({"top", "bottom"}),
        frozenset({"left", "right"}),
    }
    anchor_set = set(anchors)

    for pair in opposite_pairs:
        if pair <= anchor_set:
            return "housing"

    if len(anchors) == 2:
        return "bracket"

    return "housing"


def _face_dimensions(
    anchor: str,
    length: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Return (face_length, face_width) for a given anchor."""
    mapping = {
        "front": (length, height),
        "back": (length, height),
        "left": (width, height),
        "right": (width, height),
        "top": (length, width),
        "bottom": (length, width),
    }
    return mapping.get(anchor, (length, width))


def _face_center(
    anchor: str,
    length: float,
    width: float,
    height: float,
) -> tuple[float, float, float]:
    """Return center point of a face on a bounding box [0,L]×[0,W]×[0,H]."""
    mapping = {
        "front": (length / 2, 0, height / 2),
        "back": (length / 2, width, height / 2),
        "left": (0, width / 2, height / 2),
        "right": (length, width / 2, height / 2),
        "top": (length / 2, width / 2, height),
        "bottom": (length / 2, width / 2, 0),
    }
    return mapping.get(anchor, (length / 2, width / 2, height / 2))


# ---------------------------------------------------------------------------
# FreeCAD script builder
# ---------------------------------------------------------------------------


def _build_fc_script(
    name: str,
    shape: str,
    length: float,
    width: float,
    height: float,
    interface_ops: list[str],
) -> str:
    """Build a complete FreeCAD Python script.

    Args:
        name: Part name
        shape: "plate" / "bracket" / "housing"
        length, width, height: Bounding box dimensions
        interface_ops: List of Python code strings for hole/cut features

    Returns:
        Complete FreeCAD script as a string.
    """
    lines = [
        'import FreeCAD, Part, math',
        f'doc = FreeCAD.newDocument("{name}")',
    ]

    if shape == "plate":
        lines.append(
            f'body = Part.makeBox({length}, {width}, {height})'
        )
    elif shape == "bracket":
        lines.append(f'# Vertical plate')
        lines.append(f'vert = Part.makeBox({length}, {width}, {height})')
        lines.append(f'# Horizontal shelf')
        shelf_h = min(height * 0.4, 15.0)
        lines.append(
            f'shelf = Part.makeBox({length}, {width}, {shelf_h})'
        )
        lines.append('body = vert.fuse(shelf)')
    else:  # housing
        lines.append(
            f'body = Part.makeBox({length}, {width}, {height})'
        )

    # Add interface feature operations
    for op in interface_ops:
        lines.append(op)

    lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
    lines.append('obj.Shape = body')
    lines.append('doc.recompute()')

    return '\n'.join(lines) + '\n'


def _generate_interface_ops(
    iface: MountingInterface,
    anchor: str,
    length: float,
    width: float,
    height: float,
    thickness: float,
    offset: tuple[float, float, float] = (0, 0, 0),
) -> tuple[list[str], list[BoltHole]]:
    """Generate FreeCAD Python operations for one interface.

    Returns:
        (script_lines, holes) — Python code strings and BoltHole definitions.
    """
    ops: list[str] = []
    holes: list[BoltHole] = []

    cx, cy, cz = _face_center(anchor, length, width, height)
    cx += offset[0]
    cy += offset[1]
    cz += offset[2]

    # Determine the hole direction based on anchor face
    if anchor in ("front", "back"):
        hole_depth = width + 4
    elif anchor in ("left", "right"):
        hole_depth = length + 4
    else:  # top, bottom
        hole_depth = height + 4

    # Through holes
    for i, hole in enumerate(iface.holes):
        hole_r = hole.diameter / 2
        hx = cx + hole.x - (length / 2 if anchor not in ("left", "right") else 0)
        hy = cy + hole.y - (width / 2 if anchor not in ("front", "back") else 0)
        hz = cz

        # Clamp to body bounds
        hx = max(hole_r + 1, min(hx, length - hole_r - 1))
        hy = max(hole_r + 1, min(hy, width - hole_r - 1))

        ops.append(
            f'hole_{anchor}_{i} = Part.makeCylinder({hole_r}, {hole_depth})'
        )
        ops.append(
            f'hole_{anchor}_{i}.translate(FreeCAD.Vector({hx}, {hy}, {hz}))'
        )
        ops.append(f'body = body.cut(hole_{anchor}_{i})')

        holes.append(BoltHole(x=hx, y=hy, diameter=hole.diameter))

    # Bore (for motor shaft, bearing seat, etc.)
    if iface.bore_diameter > 0:
        bore_r = iface.bore_diameter / 2
        ops.append(
            f'bore_{anchor} = Part.makeCylinder({bore_r}, {hole_depth})'
        )
        ops.append(
            f'bore_{anchor}.translate(FreeCAD.Vector({cx}, {cy}, {cz}))'
        )
        ops.append(f'body = body.cut(bore_{anchor})')

    return ops, holes


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------


def _generate_params(
    length: float,
    width: float,
    height: float,
) -> list[ParamDef]:
    """Generate parametric definitions from bounding box dimensions."""
    return [
        ParamDef("length", "长度", "mm", length, length * 0.5, length * 1.5, 1),
        ParamDef("width", "宽度", "mm", width, width * 0.5, width * 1.5, 1),
        ParamDef("height", "高度/厚度", "mm", height, height * 0.5, height * 1.5, 0.5),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_structural_part(
    name: str,
    interfaces: list[InterfaceRequirement],
    *,
    part_type: str = "bracket",
    material: str = "PLA",
    min_wall_thickness: float = 3.0,
) -> PartTemplate:
    """Generate a structural part template from interface requirements.

    Algorithm:
    1. Collect MountingInterface data for each functional part
    2. Compute bounding box
    3. Infer shape (plate / bracket / housing)
    4. Generate FreeCAD script with holes/features
    5. Build composite MountingInterface
    6. Return complete PartTemplate

    Args:
        name: Unique template ID and part name
        interfaces: List of functional part mounting requirements
        part_type: Shape hint ("bracket" / "plate" / "housing" / "link")
        material: Default material
        min_wall_thickness: Minimum wall thickness around features (mm)

    Returns:
        A complete PartTemplate ready for use or registration.
    """
    warnings: list[str] = []
    collected: list[tuple[MountingInterface, InterfaceRequirement]] = []

    # Step 1: Collect interface data
    for req in interfaces:
        mi = get_mounting_interface(req.functional_part_id)
        if mi is None:
            warnings.append(
                f"No MountingInterface found for '{req.functional_part_id}'. "
                f"Skipping this interface."
            )
            continue
        collected.append((mi, req))

    if not collected:
        # Fallback: create a minimal plate
        warnings.append("No valid interfaces found. Generating minimal plate.")
        return PartTemplate(
            id=name,
            name_en=name.replace("_", " ").title(),
            name_cn=name,
            category="structural",
            subcategory="plate",
            description=f"Auto-generated structural part: {name}",
            tags=["auto-generated", "structural"],
            material_default=material,
            parameters=[
                ParamDef("length", "长度", "mm", 50, 20, 200, 1),
                ParamDef("width", "宽度", "mm", 50, 20, 200, 1),
                ParamDef("height", "高度", "mm", 5, 2, 20, 0.5),
            ],
            fc_script_template=(
                'import FreeCAD, Part\n'
                f'doc = FreeCAD.newDocument("{name}")\n'
                'body = Part.makeBox({length}, {width}, {height})\n'
                f'obj = doc.addObject("Part::Feature", "{name}")\n'
                'obj.Shape = body\ndoc.recompute()\n'
            ),
        )

    # Step 2: Compute bounding box
    anchors = [req.anchor for _, req in collected]
    shape = _infer_shape(anchors)

    # Calculate dimensions from interface features
    max_face_dims: dict[str, tuple[float, float]] = {}
    for mi, req in collected:
        fl, fw = _face_dimensions(req.anchor, 100, 100, 100)
        # Estimate feature span from holes
        if mi.holes:
            max_x = max(h.x for h in mi.holes)
            max_y = max(h.y for h in mi.holes)
            min_x = min(h.x for h in mi.holes)
            min_y = min(h.y for h in mi.holes)
            span_x = max_x - min_x + max(h.diameter for h in mi.holes)
            span_y = max_y - min_y + max(h.diameter for h in mi.holes)
        else:
            span_x = mi.bore_diameter + 10 if mi.bore_diameter > 0 else 20
            span_y = span_x

        needed_l = span_x + 2 * min_wall_thickness
        needed_w = span_y + 2 * min_wall_thickness

        current = max_face_dims.get(req.anchor, (0, 0))
        max_face_dims[req.anchor] = (
            max(current[0], needed_l),
            max(current[1], needed_w),
        )

    # Derive overall bounding box
    # Default: infer from the largest face dimensions
    length = 50.0
    width = 50.0
    height = min_wall_thickness * 2  # minimum thickness

    for anchor, (fl, fw) in max_face_dims.items():
        if anchor in ("front", "back"):
            length = max(length, fl)
            height = max(height, fw)
        elif anchor in ("left", "right"):
            width = max(width, fl)
            height = max(height, fw)
        elif anchor in ("top", "bottom"):
            length = max(length, fl)
            width = max(width, fw)

    # Ensure minimum thickness
    height = max(height, min_wall_thickness)

    # Step 3: Generate FreeCAD script
    all_ops: list[str] = []
    all_holes: list[BoltHole] = []

    for mi, req in collected:
        ops, holes = _generate_interface_ops(
            mi, req.anchor, length, width, height,
            min_wall_thickness, req.offset,
        )
        all_ops.extend(ops)
        all_holes.extend(holes)

    script = _build_fc_script(name, shape, length, width, height, all_ops)

    # Step 4: Build composite MountingInterface
    composite_iface = MountingInterface(
        interface_type="through_hole" if all_holes else "press_fit",
        contact_face="top",
        holes=all_holes,
        bore_diameter=max(
            (mi.bore_diameter for mi, _ in collected),
            default=0,
        ),
    )

    # Step 5: Build parameters
    params = _generate_params(length, width, height)

    # Step 6: Assemble template
    template = PartTemplate(
        id=name,
        name_en=name.replace("_", " ").title(),
        name_cn=name,
        category="structural",
        subcategory=part_type,
        description=f"Auto-generated {part_type}: {name}. "
                    f"Mounts: {', '.join(r.functional_part_id for r in interfaces)}",
        tags=["auto-generated", "structural", part_type],
        material_default=material,
        parameters=params,
        fc_script_template=script,
        notes=f"Auto-generated from interfaces: "
              f"{', '.join(r.functional_part_id for r in interfaces)}. "
              + ("; ".join(warnings) if warnings else ""),
    )
    template.mounting_interface = composite_iface

    return template


def register_generated_template(template: PartTemplate) -> bool:
    """Register a generated template in the global PART_CATALOG.

    Args:
        template: The PartTemplate to register.

    Returns:
        True if registered successfully, False if ID already exists.
    """
    if template.id in PART_CATALOG:
        return False
    PART_CATALOG[template.id] = template
    return True
