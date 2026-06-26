"""Iterative design optimization tools.

Provides:
  - Requirement change triggering redesign
  - Change impact analysis (which parts are affected)
  - Incremental updates (only regenerate affected artifacts)
  - Design comparison between iterations

Usage:
  from lang3d.tools.iteration import iterate_design, analyze_impact
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequirementChange:
    """A change to design requirements."""
    change_type: str          # "payload", "reach", "material", "actuator", "controller"
    old_value: Any
    new_value: Any
    description: str = ""


@dataclass
class ImpactAnalysis:
    """Analysis of what a requirement change affects."""
    change: RequirementChange
    affected_parts: list[str] = field(default_factory=list)
    affected_joints: list[str] = field(default_factory=list)
    affected_artifacts: list[str] = field(default_factory=list)
    severity: str = "minor"          # "minor", "moderate", "major"
    auto_fixable: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class DesignSnapshot:
    """Captures the current state of a design."""
    assembly: Assembly
    actuator_ids: list[str]
    sensor_ids: list[str]
    controller: str = "esp32"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached generated artifacts
    bom: dict[str, Any] | None = None
    firmware: dict[str, str] | None = None
    wiring: str | None = None
    print_config: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None

    def copy(self) -> DesignSnapshot:
        """Deep copy of the snapshot."""
        return copy.deepcopy(self)


@dataclass
class ChangeDiff:
    """Difference between two design snapshots."""
    part_changes: list[dict[str, Any]] = field(default_factory=list)
    joint_changes: list[dict[str, Any]] = field(default_factory=list)
    actuator_changes: list[dict[str, Any]] = field(default_factory=list)
    cost_change_cny: float = 0.0
    weight_change_g: float = 0.0
    power_change_w: float = 0.0
    artifacts_regenerated: list[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Impact analysis
# ---------------------------------------------------------------------------

def analyze_impact(
    snapshot: DesignSnapshot,
    change: RequirementChange,
) -> ImpactAnalysis:
    """Analyze what a requirement change affects.

    Returns an ImpactAnalysis describing which parts, joints, and artifacts
    need to be updated.
    """
    assembly = snapshot.assembly
    result = ImpactAnalysis(change=change)

    if change.change_type == "payload":
        _impact_payload(snapshot, change, result)
    elif change.change_type == "reach":
        _impact_reach(snapshot, change, result)
    elif change.change_type == "material":
        _impact_material(snapshot, change, result)
    elif change.change_type == "actuator":
        _impact_actuator(snapshot, change, result)
    elif change.change_type == "controller":
        _impact_controller(snapshot, change, result)
    else:
        result.severity = "unknown"
        result.auto_fixable = False
        result.notes.append(f"Unknown change type: {change.change_type}")

    return result


def _impact_payload(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    result: ImpactAnalysis,
) -> None:
    """Payload change → affects torque analysis, actuator selection, firmware."""
    old_payload = change.old_value
    new_payload = change.new_value

    # All joints that carry payload are affected
    revolute = [j for j in snapshot.assembly.joints if j.type == "revolute"]
    result.affected_joints = [j.child for j in revolute]

    # Parts connected to those joints are affected
    result.affected_parts = list({
        j.parent for j in revolute
    } | {
        j.child for j in revolute
    })

    delta = abs(new_payload - old_payload)
    if delta > old_payload * 0.5:
        result.severity = "major"
    elif delta > old_payload * 0.2:
        result.severity = "moderate"
    else:
        result.severity = "minor"

    result.affected_artifacts = ["actuator_selection", "firmware", "bom", "quality"]
    result.notes.append(
        f"Payload {old_payload}g → {new_payload}g (Δ{delta:.0f}g), "
        f"torque requirements change for all joints"
    )


def _impact_reach(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    result: ImpactAnalysis,
) -> None:
    """Reach change → affects link dimensions, IK workspace."""
    old_reach = change.old_value
    new_reach = change.new_value

    # Pitch joints (shoulder, elbow) are most affected
    revolute = [j for j in snapshot.assembly.joints if j.type == "revolute"]
    pitch_joints = [j for j in revolute if j.axis in ("y", "auto")]

    result.affected_joints = [j.child for j in pitch_joints]
    result.affected_parts = [j.child for j in pitch_joints]

    ratio = new_reach / max(old_reach, 1)
    if ratio > 1.5 or ratio < 0.67:
        result.severity = "major"
    elif ratio > 1.2 or ratio < 0.83:
        result.severity = "moderate"
    else:
        result.severity = "minor"

    result.affected_artifacts = [
        "assembly", "ik", "firmware", "print", "bom", "quality",
    ]
    result.notes.append(
        f"Reach {old_reach}mm → {new_reach}mm, "
        f"link dimensions and IK workspace change"
    )


def _impact_material(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    result: ImpactAnalysis,
) -> None:
    """Material change → affects print params, weight, cost."""
    result.affected_parts = [p.name for p in snapshot.assembly.parts]
    result.affected_joints = []  # Joints don't change
    result.severity = "minor"
    result.affected_artifacts = ["print", "bom", "quality"]
    result.notes.append(
        f"Material {change.old_value} → {change.new_value}, "
        f"print parameters and cost change"
    )


def _impact_actuator(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    result: ImpactAnalysis,
) -> None:
    """Actuator change → affects firmware, wiring, BOM, power budget."""
    revolute = [j for j in snapshot.assembly.joints if j.type == "revolute"]
    result.affected_joints = [j.child for j in revolute]
    result.affected_parts = [j.child for j in revolute]

    old_id = change.old_value
    new_id = change.new_value

    # Check if torque class changes significantly
    from ..knowledge.actuators import ACTUATORS
    old_act = ACTUATORS.get(old_id)
    new_act = ACTUATORS.get(new_id)
    old_torque = old_act.torque_kgcm if old_act else 0
    new_torque = new_act.torque_kgcm if new_act else 0

    if abs(new_torque - old_torque) > old_torque * 0.5:
        result.severity = "moderate"
    else:
        result.severity = "minor"

    result.affected_artifacts = ["firmware", "wiring", "bom", "quality"]
    result.notes.append(
        f"Actuator {old_id} (τ={old_torque} kg·cm) → "
        f"{new_id} (τ={new_torque} kg·cm)"
    )


def _impact_controller(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    result: ImpactAnalysis,
) -> None:
    """Controller change → affects firmware, wiring."""
    result.affected_parts = []
    result.affected_joints = []
    result.severity = "minor"
    result.affected_artifacts = ["firmware", "wiring", "bom", "assembly_guide"]
    result.notes.append(
        f"Controller {change.old_value} → {change.new_value}"
    )


# ---------------------------------------------------------------------------
# Change application
# ---------------------------------------------------------------------------

def apply_change(
    snapshot: DesignSnapshot,
    change: RequirementChange,
) -> tuple[DesignSnapshot, ChangeDiff]:
    """Apply a requirement change to a design snapshot.

    Returns the updated snapshot and a diff describing what changed.
    Only regenerates artifacts that are affected by the change.
    """
    new_snapshot = snapshot.copy()
    diff = ChangeDiff()
    impact = analyze_impact(snapshot, change)

    if change.change_type == "payload":
        _apply_payload(new_snapshot, change, diff)
    elif change.change_type == "reach":
        _apply_reach(new_snapshot, change, diff)
    elif change.change_type == "material":
        _apply_material(new_snapshot, change, diff)
    elif change.change_type == "actuator":
        _apply_actuator(new_snapshot, change, diff)
    elif change.change_type == "controller":
        _apply_controller(new_snapshot, change, diff)

    # Regenerate only affected artifacts
    _regenerate_affected(new_snapshot, impact, diff)

    return new_snapshot, diff


def _apply_payload(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    diff: ChangeDiff,
) -> None:
    """Apply payload change — may re-select actuators if torque insufficient."""
    new_payload = change.new_value
    snapshot.metadata["payload_g"] = new_payload

    # Re-analyze torques with new payload
    from .actuator_tools import analyze_assembly_torques, select_actuators
    torques = analyze_assembly_torques(
        snapshot.assembly,
        safety_factor=2.0,
        payload_g=new_payload,
    )

    # Check if current actuators still sufficient
    from ..knowledge.actuators import ACTUATORS
    new_actuator_ids = []
    changed = False
    for i, analysis in enumerate(torques):
        required = analysis.get("required_torque_kgcm", 0)
        if i < len(snapshot.actuator_ids):
            current_id = snapshot.actuator_ids[i]
            cur_act = ACTUATORS.get(current_id)
            current_torque = cur_act.torque_kgcm if cur_act else 0
            if current_torque >= required:
                new_actuator_ids.append(current_id)
            else:
                # Need stronger actuator
                recs = select_actuators(min_torque_kgcm=required, count=1)
                if recs:
                    new_id = recs[0]["id"]
                    new_actuator_ids.append(new_id)
                    diff.actuator_changes.append({
                        "joint": analysis.get("joint", f"joint_{i}"),
                        "old": current_id,
                        "new": new_id,
                        "reason": f"τ_req={required:.1f} > τ_avail={current_torque:.1f}",
                    })
                    changed = True
                else:
                    new_actuator_ids.append(current_id)
        else:
            recs = select_actuators(min_torque_kgcm=required, count=1)
            if recs:
                new_actuator_ids.append(recs[0]["model"])

    if changed:
        snapshot.actuator_ids = new_actuator_ids

    diff.summary = (
        f"Payload updated: {change.old_value}g → {change.new_value}g"
        + (f", {len(diff.actuator_changes)} actuator(s) re-selected" if changed else "")
    )


def _is_functional_part(part: "Part") -> bool:
    """True if *part* is a real COTS functional component that must not be
    rescaled (AGENTS.md §1.2). Mirrors ``agent.modifier._is_functional_part``
    (kept local to avoid a tools→agent import cycle). Note: "mechanical" is
    NOT treated as functional — arm_topology uses it for the designable
    gripper base/fingers. Only real actuators (servos/motors) are protected."""
    cat = (part.category or "").lower()
    if cat in ("actuator", "bearing", "gear", "fastener"):
        return True
    desc = (part.description or "").lower()
    functional_markers = (
        "servo", "mg996", "sg90", "ds3218", "dynamixel", "nema",
        "motor", "电机", "舵机", "马达", "bearing", "轴承",
    )
    return any(m in desc for m in functional_markers)


def _apply_reach(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    diff: ChangeDiff,
) -> None:
    """Apply reach change — scale link dimensions proportionally.

    Functional parts (servos/motors) are SKIPPED: a reach change must not
    rescale a real COTS actuator. Only the structural links resize.
    """
    old_reach = max(float(change.old_value), 1.0)
    new_reach = float(change.new_value)
    scale = new_reach / old_reach

    snapshot.metadata["reach_mm"] = new_reach

    # Scale pitch-link parts
    revolute = [j for j in snapshot.assembly.joints if j.type == "revolute"]
    pitch_joints = [j for j in revolute if j.axis in ("y", "auto")]
    skipped_functional = 0

    for j in pitch_joints:
        part = next((p for p in snapshot.assembly.parts if p.name == j.child), None)
        if part is None:
            continue
        # Functional parts (servos driving the joint) keep their real dimensions.
        if _is_functional_part(part):
            skipped_functional += 1
            continue

        old_dims = dict(part.dimensions)
        new_dims = {}
        for key, val in part.dimensions.items():
            if key in ("diameter", "height", "length", "width", "thickness",
                        "outer_diameter", "inner_diameter", "wall_thickness"):
                new_dims[key] = round(val * scale, 1)
            else:
                new_dims[key] = val
        part.dimensions = new_dims

        diff.part_changes.append({
            "part": part.name,
            "old_dimensions": old_dims,
            "new_dimensions": new_dims,
        })

    diff.summary = (
        f"Reach updated: {old_reach}mm → {new_reach}mm "
        f"(scale={scale:.2f}x), {len(diff.part_changes)} part(s) resized"
        + (f", {skipped_functional} functional part(s) kept (not rescaled)" if skipped_functional else "")
    )


def _apply_material(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    diff: ChangeDiff,
) -> None:
    """Apply material change to all parts."""
    new_material = change.new_value

    for part in snapshot.assembly.parts:
        old = part.material
        part.material = new_material
        diff.part_changes.append({
            "part": part.name,
            "old_material": old,
            "new_material": new_material,
        })

    diff.summary = f"Material updated: {change.old_value} → {new_material}"


def _apply_actuator(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    diff: ChangeDiff,
) -> None:
    """Replace an actuator in the design."""
    old_id = change.old_value
    new_id = change.new_value

    new_ids = []
    replaced = False
    for aid in snapshot.actuator_ids:
        if aid == old_id and not replaced:
            new_ids.append(new_id)
            replaced = True
        else:
            new_ids.append(aid)

    # If not found, append
    if not replaced:
        new_ids.append(new_id)

    snapshot.actuator_ids = new_ids

    diff.actuator_changes.append({
        "old": old_id,
        "new": new_id,
    })
    diff.summary = f"Actuator replaced: {old_id} → {new_id}"


def _apply_controller(
    snapshot: DesignSnapshot,
    change: RequirementChange,
    diff: ChangeDiff,
) -> None:
    """Change the controller."""
    snapshot.controller = change.new_value
    diff.summary = f"Controller updated: {change.old_value} → {change.new_value}"


# ---------------------------------------------------------------------------
# Artifact regeneration
# ---------------------------------------------------------------------------

def _regenerate_affected(
    snapshot: DesignSnapshot,
    impact: ImpactAnalysis,
    diff: ChangeDiff,
) -> None:
    """Regenerate only the artifacts affected by the change."""
    artifacts = set(impact.affected_artifacts)

    if "bom" in artifacts:
        from .bom_gen import generate_bom
        snapshot.bom = generate_bom(
            snapshot.assembly,
            snapshot.actuator_ids,
            snapshot.sensor_ids,
            snapshot.controller,
        )
        diff.artifacts_regenerated.append("bom")

    if "firmware" in artifacts:
        from .code_gen import generate_firmware
        snapshot.firmware = generate_firmware(
            snapshot.assembly,
            snapshot.actuator_ids,
            snapshot.controller,
            sensor_ids=snapshot.sensor_ids,
        )
        diff.artifacts_regenerated.append("firmware")

    if "wiring" in artifacts:
        from .code_gen import generate_wiring
        snapshot.wiring = generate_wiring(
            snapshot.actuator_ids,
            snapshot.controller,
        )
        diff.artifacts_regenerated.append("wiring")

    if "print" in artifacts:
        from .print_optimize import optimize_assembly_print
        material = "PLA"
        if snapshot.assembly.parts:
            material = snapshot.assembly.parts[0].material
        snapshot.print_config = optimize_assembly_print(
            snapshot.assembly, quality="standard", material=material,
        )
        diff.artifacts_regenerated.append("print_config")

    if "quality" in artifacts:
        from .quality import generate_quality_report
        snapshot.quality_report = generate_quality_report(snapshot.assembly)
        diff.artifacts_regenerated.append("quality_report")

    if "assembly" in artifacts:
        from .assembly_solver import AssemblySolver
        solver = AssemblySolver(snapshot.assembly)
        snapshot.metadata["placements"] = solver.solve()
        diff.artifacts_regenerated.append("assembly")

    if "ik" in artifacts:
        from .ik_solver import solve_ik, _extract_chain
        links, base_height = _extract_chain(snapshot.assembly)
        max_reach = sum(l.length for l in links if l.axis in ("y", "auto"))
        if max_reach < 1:
            max_reach = 100
        # Test a representative target
        target = (max_reach * 0.5, 0, base_height + max_reach * 0.5)
        result = solve_ik(snapshot.assembly, target, approach="ccd", tolerance_mm=2.0)
        snapshot.metadata["ik_test"] = {
            "target": list(target),
            "error_mm": result.error_mm,
            "reachable": result.reachable,
        }
        diff.artifacts_regenerated.append("ik")

    if "actuator_selection" in artifacts:
        from .actuator_tools import analyze_assembly_torques, power_budget
        payload = snapshot.metadata.get("payload_g", 0)
        torques = analyze_assembly_torques(snapshot.assembly, payload_g=payload)
        budget = power_budget(
            snapshot.actuator_ids,
            sensor_ids=snapshot.sensor_ids,
        )
        snapshot.metadata["torque_analysis"] = torques
        snapshot.metadata["power_budget"] = budget
        diff.artifacts_regenerated.append("actuator_analysis")

    if "assembly_guide" in artifacts:
        from .assembly_doc import generate_assembly_guide
        guide = generate_assembly_guide(
            snapshot.assembly,
            snapshot.actuator_ids,
            snapshot.sensor_ids,
            snapshot.controller,
        )
        snapshot.metadata["assembly_guide"] = guide
        diff.artifacts_regenerated.append("assembly_guide")


# ---------------------------------------------------------------------------
# Design comparison
# ---------------------------------------------------------------------------

def compare_designs(before: DesignSnapshot, after: DesignSnapshot) -> ChangeDiff:
    """Compare two design snapshots and return the differences."""
    diff = ChangeDiff()

    # Compare parts
    before_parts = {p.name: p for p in before.assembly.parts}
    after_parts = {p.name: p for p in after.assembly.parts}

    for name in before_parts:
        if name in after_parts:
            bp = before_parts[name]
            ap = after_parts[name]
            changes = {}
            if bp.material != ap.material:
                changes["material"] = {"old": bp.material, "new": ap.material}
            if bp.dimensions != ap.dimensions:
                changes["dimensions"] = {"old": bp.dimensions, "new": ap.dimensions}
            if changes:
                diff.part_changes.append({"part": name, **changes})

    # Compare actuators
    for i, (old_id, new_id) in enumerate(
        zip(before.actuator_ids, after.actuator_ids)
    ):
        if old_id != new_id:
            diff.actuator_changes.append({
                "index": i,
                "old": old_id,
                "new": new_id,
            })

    # Compare costs
    if before.bom and after.bom:
        old_cost = before.bom.get("cost_summary", {}).get("total_cost_cny", 0)
        new_cost = after.bom.get("cost_summary", {}).get("total_cost_cny", 0)
        diff.cost_change_cny = round(new_cost - old_cost, 2)

    # Compare power
    old_budget = before.metadata.get("power_budget", {})
    new_budget = after.metadata.get("power_budget", {})
    old_power = old_budget.get("total_power_w", 0)
    new_power = new_budget.get("total_power_w", 0)
    diff.power_change_w = round(new_power - old_power, 2)

    # Summary
    parts = len(diff.part_changes)
    actuators = len(diff.actuator_changes)
    items = []
    if parts:
        items.append(f"{parts} part(s)")
    if actuators:
        items.append(f"{actuators} actuator(s)")
    diff.summary = f"Changes: {', '.join(items)}" if items else "No changes"

    return diff


# ---------------------------------------------------------------------------
# Full iteration
# ---------------------------------------------------------------------------

def iterate_design(
    snapshot: DesignSnapshot,
    change: RequirementChange,
) -> tuple[DesignSnapshot, ImpactAnalysis, ChangeDiff]:
    """Perform a full design iteration.

    1. Analyze impact of the change
    2. Apply the change
    3. Regenerate affected artifacts
    4. Return updated snapshot + analysis + diff
    """
    impact = analyze_impact(snapshot, change)
    new_snapshot, diff = apply_change(snapshot, change)
    return new_snapshot, impact, diff


def create_snapshot(
    assembly: Assembly,
    actuator_ids: list[str],
    sensor_ids: list[str],
    controller: str = "esp32",
    generate_all: bool = False,
) -> DesignSnapshot:
    """Create a design snapshot, optionally generating all artifacts."""
    snapshot = DesignSnapshot(
        assembly=copy.deepcopy(assembly),
        actuator_ids=list(actuator_ids),
        sensor_ids=list(sensor_ids),
        controller=controller,
    )

    if generate_all:
        # Generate all artifacts upfront
        from .actuator_tools import analyze_assembly_torques, power_budget
        from .assembly_solver import AssemblySolver
        from .bom_gen import generate_bom
        from .code_gen import generate_firmware, generate_wiring
        from .ik_solver import solve_ik, _extract_chain
        from .print_optimize import optimize_assembly_print
        from .quality import generate_quality_report

        snapshot.bom = generate_bom(assembly, actuator_ids, sensor_ids, controller)
        snapshot.firmware = generate_firmware(
            assembly, actuator_ids, controller, sensor_ids=sensor_ids,
        )
        snapshot.wiring = generate_wiring(actuator_ids, controller)

        material = "PLA"
        if assembly.parts:
            material = assembly.parts[0].material
        snapshot.print_config = optimize_assembly_print(assembly, material=material)
        snapshot.quality_report = generate_quality_report(assembly)

        # Assembly + IK + torque analysis
        solver = AssemblySolver(assembly)
        snapshot.metadata["placements"] = solver.solve()

        links, base_height = _extract_chain(assembly)
        max_reach = sum(l.length for l in links if l.axis in ("y", "auto"))
        if max_reach < 1:
            max_reach = 100
        target = (max_reach * 0.5, 0, base_height + max_reach * 0.5)
        ik_result = solve_ik(assembly, target, approach="ccd", tolerance_mm=2.0)
        snapshot.metadata["ik_test"] = {
            "target": list(target),
            "error_mm": ik_result.error_mm,
            "reachable": ik_result.reachable,
        }

        torques = analyze_assembly_torques(assembly)
        budget = power_budget(actuator_ids, sensor_ids=sensor_ids)
        snapshot.metadata["torque_analysis"] = torques
        snapshot.metadata["power_budget"] = budget

    return snapshot


# ---------------------------------------------------------------------------
# Tool: iteration_design
# ---------------------------------------------------------------------------

class IterationTool(Tool):
    """Tool for iterative design optimization."""

    name = "iteration_design"
    description = (
        "迭代设计优化：分析需求变更的影响，自动重新设计受影响的部分。"
        "支持负载、工作半径、材料、执行器、控制器等变更类型。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "装配体名称（如 'robotic_arm'）",
                    },
                    "change_type": {
                        "type": "string",
                        "enum": ["payload", "reach", "material", "actuator", "controller"],
                        "description": "变更类型",
                    },
                    "old_value": {
                        "description": "旧值",
                    },
                    "new_value": {
                        "description": "新值",
                    },
                },
                "required": ["change_type", "old_value", "new_value"],
            },
        )

    def execute(
        self,
        *,
        change_type: str = "payload",
        old_value: Any = None,
        new_value: Any = None,
        assembly_name: str = "robotic_arm",
        **kwargs: Any,
    ) -> str:
        if old_value is None or new_value is None:
            return "错误：必须指定 old_value 和 new_value"

        # Resolve assembly
        from .assembly_solver import _resolve_assembly
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        # Default actuators/sensors
        actuator_ids = kwargs.get("actuator_ids", ["MG996R", "MG996R", "DS3218", "SG90"])
        sensor_ids = kwargs.get("sensor_ids", ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"])
        controller = kwargs.get("controller", "esp32")

        # Create snapshot
        snapshot = create_snapshot(assembly, actuator_ids, sensor_ids, controller)

        # Type conversion for numeric values
        if change_type in ("payload", "reach"):
            old_value = float(old_value)
            new_value = float(new_value)

        change = RequirementChange(
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
        )

        new_snapshot, impact, diff = iterate_design(snapshot, change)

        # Build report
        lines = [
            f"[迭代设计] {assembly.name}",
            f"变更: {change_type}: {old_value} → {new_value}",
            f"严重程度: {impact.severity}",
            f"影响零件: {', '.join(impact.affected_parts) or '无'}",
            f"影响关节: {', '.join(impact.affected_joints) or '无'}",
            f"需重新生成: {', '.join(impact.affected_artifacts) or '无'}",
            f"已重新生成: {', '.join(diff.artifacts_regenerated) or '无'}",
            "",
            f"摘要: {diff.summary}",
        ]

        if diff.part_changes:
            lines.append("")
            lines.append("--- 零件变更 ---")
            for pc in diff.part_changes:
                name = pc.get("part", "?")
                lines.append(f"  {name}:")
                if "old_dimensions" in pc:
                    lines.append(f"    尺寸: {pc['old_dimensions']} → {pc['new_dimensions']}")
                if "old_material" in pc:
                    lines.append(f"    材料: {pc['old_material']} → {pc['new_material']}")

        if diff.actuator_changes:
            lines.append("")
            lines.append("--- 执行器变更 ---")
            for ac in diff.actuator_changes:
                lines.append(f"  {ac.get('old', '?')} → {ac.get('new', '?')}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_iteration_tools(registry: Any) -> None:
    """Register iteration design tools."""
    registry.register(IterationTool())
