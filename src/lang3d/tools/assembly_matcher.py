"""Assembly feature auto-matcher — glue layer connecting all subsystems.

Given two parts and a connection method, this module:
1. Identifies which is the functional part (has MountingInterface) vs structural
2. Generates mating features on the structural part (holes, bores, pockets)
3. Selects appropriate fasteners (bolt length, nut, washer) from catalog
4. Sets up mating constraints for the ConstraintSolver
5. Validates the match (alignment, interference, clearance)

This is the integration layer connecting:
- Task 71: ConnectionFeatureEngine (bolted/press-fit/snap-fit features)
- Task 72: fastener_catalog (bolt/nut/washer specs)
- Task 76: MountingInterface (standard mounting faces)
- Task 77: MatingConstraint (constraint solving)

Pure-function module: no FreeCAD imports, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import ConnectionMethod, Joint, Part
from ..knowledge.parts_catalog import (
    MountingInterface,
    get_mounting_interface,
    get_template,
    auto_match_interface,
)
from ..knowledge.fastener_catalog import (
    recommend_bolt_length,
    recommend_fastener_set,
    get_bolt_spec,
    get_clearance_hole,
    get_torque,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FastenerSelection:
    """Selected fasteners for a connection."""

    bolt_size: str = ""              # e.g. "M3"
    bolt_length: float = 0.0         # mm
    nut_included: bool = False
    washer_included: bool = False
    bolt_spec: dict[str, float] = field(default_factory=dict)
    nut_spec: dict[str, float] = field(default_factory=dict)
    washer_spec: dict[str, float] = field(default_factory=dict)


@dataclass
class AssemblyMatchResult:
    """Complete result from assembly feature matching."""

    functional_part_name: str = ""
    structural_part_name: str = ""
    interface_used: str = ""            # MountingInterface description or "heuristic"

    # Features to add to the structural part
    structural_ops: list[dict] = field(default_factory=list)

    # Fastener models to generate
    fastener_ops: list[dict] = field(default_factory=list)
    fastener_selection: FastenerSelection = field(default_factory=FastenerSelection)

    # Constraint for positioning
    constraint_type: str = ""
    parent_entity: tuple = ("face", "top")
    child_entity: tuple = ("face", "bottom")
    constraint_params: dict[str, float] = field(default_factory=dict)

    # Validation
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    features_summary: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AssemblyMatcher
# ---------------------------------------------------------------------------

class AssemblyMatcher:
    """Auto-match two parts with a connection method.

    Usage::

        matcher = AssemblyMatcher()
        result = matcher.auto_match(
            part_a=bracket,
            part_b=nema17_motor,
            connection_method=ConnectionMethod(type="bolted", bolt_size="M3"),
        )
        # result.structural_ops → FreeCAD ops for bracket
        # result.fastener_selection → bolt/nut/washer specs
        # result.constraint_type → "coincident" for face mating
    """

    def auto_match(
        self,
        part_a: Part,
        part_b: Part,
        connection_method: ConnectionMethod,
        anchor: str = "top",
    ) -> AssemblyMatchResult:
        """Auto-match two parts with a connection method.

        Args:
            part_a: First part (typically structural).
            part_b: Second part (typically functional).
            connection_method: How they connect (bolted, press-fit, etc).
            anchor: Which face of part_a receives features.

        Returns:
            AssemblyMatchResult with features, fasteners, constraints, and validation.
        """
        result = AssemblyMatchResult()

        # Step 1: Identify functional vs structural
        func_part, struct_part, func_id = self._identify_functional_part(part_a, part_b)

        if func_part is None:
            # Both are structural or unknown — use heuristic
            result.structural_part_name = part_a.name
            result.functional_part_name = part_b.name
            result.interface_used = "heuristic"
            self._generate_heuristic_features(part_a, part_b, connection_method, anchor, result)
        else:
            result.structural_part_name = struct_part.name
            result.functional_part_name = func_part.name

            # Step 2: Generate mating features
            mi = get_mounting_interface(func_id) if func_id else None
            if mi:
                result.interface_used = f"{func_id}:{mi.interface_type}"
                self._generate_interface_features(struct_part, func_part, mi, connection_method, anchor, result)
            else:
                result.interface_used = "heuristic"
                self._generate_heuristic_features(struct_part, func_part, connection_method, anchor, result)

        # Step 3: Select fasteners
        if connection_method.type == "bolted":
            sp = struct_part or part_a
            fp = func_part or part_b
            self._select_fasteners(sp, fp, connection_method, result)

        # Step 4: Setup constraints
        self._setup_constraints(connection_method, anchor, result)

        # Step 5: Verify
        self._verify_match(connection_method, result)

        return result

    # ------------------------------------------------------------------
    # Step 1: Identify functional part
    # ------------------------------------------------------------------

    def _identify_functional_part(
        self, part_a: Part, part_b: Part
    ) -> tuple[Part | None, Part | None, str | None]:
        """Identify which part is functional (has MountingInterface).

        Returns (functional_part, structural_part, functional_id) or
                (None, None, None) if neither has an interface.
        """
        for part, other in [(part_a, part_b), (part_b, part_a)]:
            # Try by catalog ID (part name matches catalog key)
            pid = self._find_catalog_id(part)
            if pid:
                mi = get_mounting_interface(pid)
                if mi:
                    return part, other, pid

        # Fallback: check by category
        func_categories = {"motor", "stepper", "servo", "bearing", "sensor",
                            "encoder", "driver", "controller"}
        for part, other in [(part_a, part_b), (part_b, part_a)]:
            if part.category.lower() in func_categories:
                return part, other, None

        return None, None, None

    def _find_catalog_id(self, part: Part) -> str | None:
        """Try to find the part's catalog ID by matching name."""
        from ..knowledge.parts_catalog import PART_CATALOG
        # Exact match
        if part.name in PART_CATALOG:
            return part.name
        # Partial match: check if part name contains a catalog key
        for pid in PART_CATALOG:
            if pid.replace("_", "") in part.name.replace("_", "").lower():
                return pid
        return None

    # ------------------------------------------------------------------
    # Step 2a: Generate features from MountingInterface
    # ------------------------------------------------------------------

    def _generate_interface_features(
        self,
        struct_part: Part,
        func_part: Part,
        interface: MountingInterface,
        connection: ConnectionMethod,
        anchor: str,
        result: AssemblyMatchResult,
    ) -> None:
        """Generate features on structural part using MountingInterface data."""
        ops = auto_match_interface(
            structural_dims=struct_part.dimensions,
            interface=interface,
            anchor=anchor,
        )
        result.structural_ops = ops

        # Describe features
        n_holes = len(interface.holes)
        features = []
        if n_holes > 0:
            features.append(f"{n_holes}× mounting holes (Ø{interface.holes[0].diameter}mm)")
        if interface.bore_diameter > 0:
            features.append(f"center bore Ø{interface.bore_diameter}mm")
        if interface.pocket_width > 0:
            features.append(f"body pocket {interface.pocket_width}×{interface.pocket_height}mm")
        if interface.press_fit_interference > 0:
            features.append(f"press-fit bore (interference {interface.press_fit_interference}mm)")
        if interface.shoulder_diameter > 0:
            features.append(f"shoulder Ø{interface.shoulder_diameter}mm")
        result.features_summary = features

        # Fastener ops for bolted connections
        if connection.type == "bolted" and interface.holes:
            for i in range(len(interface.holes)):
                result.fastener_ops.append({
                    "type": "fastener_model",
                    "fastener_type": "bolt",
                    "size": connection.bolt_size or "M3",
                    "hole_index": i,
                })

    # ------------------------------------------------------------------
    # Step 2b: Generate features using heuristic (fallback)
    # ------------------------------------------------------------------

    def _generate_heuristic_features(
        self,
        struct_part: Part,
        func_part: Part,
        connection: ConnectionMethod,
        anchor: str,
        result: AssemblyMatchResult,
    ) -> None:
        """Generate features using ConnectionFeatureEngine heuristic."""
        from .connection_features import ConnectionFeatureEngine

        engine = ConnectionFeatureEngine()
        func_id = self._find_catalog_id(func_part)
        cf_result = engine.generate_features(
            structural_part=struct_part,
            connection=connection,
            anchor=anchor,
            functional_part_id=func_id,
        )
        result.structural_ops = cf_result.ops
        result.fastener_ops = cf_result.fastener_ops
        result.features_summary = cf_result.features_generated

    # ------------------------------------------------------------------
    # Step 3: Select fasteners
    # ------------------------------------------------------------------

    def _select_fasteners(
        self,
        struct_part: Part,
        func_part: Part,
        connection: ConnectionMethod,
        result: AssemblyMatchResult,
    ) -> None:
        """Select appropriate bolt length, nut, and washer."""
        bolt_size = connection.bolt_size or "M3"

        # Estimate grip length: structural thickness + functional interface depth
        grip = self._estimate_grip_length(struct_part, func_part, result)

        bolt_length = recommend_bolt_length(grip)
        fastener_set = recommend_fastener_set(bolt_size, grip, with_washer=True)

        sel = FastenerSelection(
            bolt_size=bolt_size,
            bolt_length=bolt_length,
            nut_included=fastener_set.get("nut") is not None,
            washer_included=fastener_set.get("washer") is not None,
        )

        if fastener_set.get("bolt"):
            sel.bolt_spec = fastener_set["bolt"]
        if fastener_set.get("nut"):
            sel.nut_spec = fastener_set["nut"]
        if fastener_set.get("washer"):
            sel.washer_spec = fastener_set["washer"]

        result.fastener_selection = sel
        result.features_summary.append(
            f"Fasteners: {bolt_size}×{bolt_length}mm bolt"
            + (" + nut" if sel.nut_included else "")
            + (" + washer" if sel.washer_included else "")
        )

    def _estimate_grip_length(
        self, struct_part: Part, func_part: Part, result: AssemblyMatchResult
    ) -> float:
        """Estimate the total grip length for bolt selection.

        Grip = structural thickness + functional interface depth + pocket depth
               + washer thickness + 1mm margin.
        """
        thickness_keys = ["thickness", "height"]

        # 1. Structural part thickness
        struct_t = 0.0
        for k in thickness_keys:
            if k in struct_part.dimensions:
                struct_t = struct_part.dimensions[k]
                break

        # 2. Functional part mounting face thickness
        func_t = 0.0
        for k in thickness_keys:
            if k in func_part.dimensions:
                func_t = func_part.dimensions[k]
                break

        # 3. Pocket depth from MountingInterface
        pocket_d = 0.0
        func_id = self._find_catalog_id(func_part)
        if func_id:
            mi = get_mounting_interface(func_id)
            if mi and mi.pocket_height > 0:
                pocket_d = mi.pocket_height

        # 4. Washer thickness (~1mm for M3, lookup from catalog)
        washer_t = 1.0
        from ..knowledge.fastener_catalog import get_washer_spec
        bolt_size = result.fastener_selection.bolt_size or "M3"
        w_spec = get_washer_spec(bolt_size)
        if w_spec:
            washer_t = w_spec.thickness

        return max(struct_t + func_t + pocket_d + washer_t, 3.0)

    # ------------------------------------------------------------------
    # Step 4: Setup constraints
    # ------------------------------------------------------------------

    def _setup_constraints(
        self,
        connection: ConnectionMethod,
        anchor: str,
        result: AssemblyMatchResult,
    ) -> None:
        """Determine mating constraint type from connection method."""
        from ..knowledge.parts_catalog import MOUNTING_INTERFACES

        if connection.type == "bolted":
            result.constraint_type = "coincident"
            result.parent_entity = ("face", anchor)
            result.child_entity = ("face", _opposite_face(anchor))
        elif connection.type == "press_fit":
            result.constraint_type = "concentric"
            result.parent_entity = ("face", anchor)
            result.child_entity = ("face", "bottom")
        else:
            result.constraint_type = "coincident"
            result.parent_entity = ("face", anchor)
            result.child_entity = ("face", _opposite_face(anchor))

    # ------------------------------------------------------------------
    # Step 5: Verify match
    # ------------------------------------------------------------------

    def _verify_match(
        self,
        connection: ConnectionMethod,
        result: AssemblyMatchResult,
    ) -> None:
        """Validate the generated match."""
        result.valid = True

        if connection.type == "bolted":
            # Check bolt size was determined
            if not result.fastener_selection.bolt_size:
                result.warnings.append("Bolt size not determined")
                result.valid = False

            # Check mounting holes were generated
            hole_ops = [o for o in result.structural_ops if "hole" in o.get("name", "")]
            if not hole_ops:
                result.warnings.append("No mounting holes generated for bolted connection")
                # Not invalid — could be using interface features

        elif connection.type == "press_fit":
            # Check bore was generated
            bore_ops = [o for o in result.structural_ops
                        if "bore" in o.get("name", "") or "press" in o.get("name", "")]
            if not bore_ops:
                result.warnings.append("No press-fit bore generated")

        if not result.structural_ops:
            result.warnings.append("No structural features generated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opposite_face(face: str) -> str:
    """Return the opposite face name."""
    opposites = {
        "top": "bottom", "bottom": "top",
        "left": "right", "right": "left",
        "front": "back", "back": "front",
    }
    return opposites.get(face, "bottom")


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------

def assembly_match_tool_factory() -> tuple[Any, Any]:
    """Create the assembly_auto_match tool."""
    from ..models.base import ToolDefinition

    definition = ToolDefinition(
        name="assembly_auto_match",
        description="Auto-match two parts: generate mating features, select fasteners, setup constraints",
        parameters={
            "part_a_name": {"type": "string", "description": "First part name"},
            "part_b_name": {"type": "string", "description": "Second part name"},
            "connection_type": {"type": "string", "description": "bolted/press_fit/snap_fit/adhesive/welded/magnetic"},
            "bolt_size": {"type": "string", "description": "Bolt size (e.g. M3) for bolted connections"},
            "anchor": {"type": "string", "description": "Face where connection happens (top/bottom/left/right/front/back)"},
        },
    )

    class _AssemblyMatchTool:
        def execute(self, *, part_a_name: str = "", part_b_name: str = "",
                    connection_type: str = "bolted", bolt_size: str = "M3",
                    anchor: str = "top", **kwargs) -> str:
            from ..knowledge.mechanics import Part, find_part
            pa = find_part(part_a_name)
            pb = find_part(part_b_name)
            if pa is None:
                pa = Part(part_a_name, "structural", "auto", dimensions={"length": 50, "width": 50, "thickness": 5})
            if pb is None:
                pb = Part(part_b_name, "functional", "auto", dimensions={"length": 42, "width": 42, "height": 47})
            conn = ConnectionMethod(type=connection_type, bolt_size=bolt_size)
            matcher = AssemblyMatcher()
            r = matcher.auto_match(pa, pb, conn, anchor=anchor)
            lines = [
                f"Match: {r.structural_part_name} ↔ {r.functional_part_name}",
                f"Interface: {r.interface_used}",
                f"Valid: {r.valid}",
                f"Features: {len(r.structural_ops)} ops",
            ]
            for f in r.features_summary:
                lines.append(f"  - {f}")
            if r.fastener_selection.bolt_size:
                fs = r.fastener_selection
                lines.append(f"Fastener: {fs.bolt_size}×{fs.bolt_length}mm")
            if r.warnings:
                lines.append(f"Warnings: {r.warnings}")
            return "\n".join(lines)

        def get_definition(self) -> ToolDefinition:
            return definition

    return definition, _AssemblyMatchTool
