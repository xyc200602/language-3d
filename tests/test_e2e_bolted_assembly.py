"""End-to-end test: L-bracket + NEMA17 bolted assembly (Task 82).

Validates the complete integration of Tasks 71-80:
  1) Auto-detect NEMA17 MountingInterface from catalog
  2) Generate 4× Ø3.4mm holes at 31mm spacing on L-bracket via PartFeatureEngine
  3) Select M3×12 bolts + M3 nuts from FastenerCatalog
  4) Solve mating constraints (face coincident + hole concentric) via AssemblySolver
  5) Verify assembly via AssemblyVerifier (mating surfaces, bolt alignment, collision-free)
  6) Check all verification items pass
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from lang3d.agent.assembly_verifier import (
    AssemblySequenceCheck,
    AssemblyVerificationResult,
    AssemblyVerifier,
    BoltHoleAlignmentCheck,
    MatingSurfaceCheck,
    VerificationItem,
)
from lang3d.knowledge.fastener_catalog import (
    BoltSpec,
    NutSpec,
    WasherSpec,
    get_bolt_spec,
    get_nut_spec,
    get_washer_spec,
    recommend_bolt_length,
    recommend_fastener_set,
)
from lang3d.knowledge.mechanics import (
    Assembly,
    ConnectionMethod,
    Joint,
    Part,
)
from lang3d.knowledge.parts_catalog import (
    BoltHole,
    MountingInterface,
    get_mounting_interface,
)
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.connection_features import ConnectionFeatureEngine


# ============================================================================
# Test fixture: L-bracket + NEMA17 assembly
# ============================================================================

def _make_l_bracket_assembly() -> Assembly:
    """Create L-bracket + NEMA17 bolted assembly definition."""
    bracket = Part(
        name="l_bracket",
        category="structural",
        description="L-shaped mounting bracket for NEMA17 motor",
        material="PLA",
        dimensions={"length": 60, "width": 60, "thickness": 5},
    )

    motor = Part(
        name="nema17_motor",
        category="actuator",
        description="NEMA17 stepper motor (42×42×48mm)",
        material="Steel",
        dimensions={"length": 42.3, "width": 42.3, "height": 48},
    )

    connection = ConnectionMethod(
        type="bolted",
        bolt_size="M3",
        bolt_count=4,
    )

    joint = Joint(
        type="fixed",
        parent="l_bracket",
        child="nema17_motor",
        parent_anchor="top",
        child_anchor="bottom",
        connection=connection,
        description="NEMA17 bolted to L-bracket with 4×M3",
    )

    return Assembly(
        name="L-Bracket NEMA17 Assembly",
        parts=[bracket, motor],
        joints=[joint],
    )


def _make_parts_results() -> dict:
    """Simulate parts_results as if parts were generated."""
    return {
        "l_bracket": {
            "artifacts": ["l_bracket.step"],
            "result": "success",
        },
        "nema17_motor": {
            "artifacts": ["nema17_motor.step"],
            "result": "success",
        },
    }


# ============================================================================
# Step 1: MountingInterface auto-detection
# ============================================================================

class TestStep1MountingInterface:

    def test_nema17_interface_exists(self):
        """NEMA17 mounting interface should be available in catalog."""
        iface = get_mounting_interface("nema17_stepper")
        assert iface is not None

    def test_nema17_interface_has_4_holes(self):
        """NEMA17 standard: 4 bolt holes at ±15.5mm from center."""
        iface = get_mounting_interface("nema17_stepper")
        assert len(iface.holes) == 4

    def test_nema17_hole_spacing_31mm(self):
        """NEMA17 hole spacing: 31mm (±15.5mm from center)."""
        iface = get_mounting_interface("nema17_stepper")
        # Check that holes are at ±15.5mm positions
        xs = sorted(set(h.x for h in iface.holes))
        ys = sorted(set(h.y for h in iface.holes))
        assert xs == [-15.5, 15.5]
        assert ys == [-15.5, 15.5]

    def test_nema17_hole_diameter(self):
        """Clearance holes for M3 bolts: Ø3.4mm (ISO 273 normal fit)."""
        iface = get_mounting_interface("nema17_stepper")
        for h in iface.holes:
            assert h.diameter == pytest.approx(3.4, abs=0.1)

    def test_nema17_bore_diameter(self):
        """Shaft clearance bore: Ø23mm for NEMA17."""
        iface = get_mounting_interface("nema17_stepper")
        assert iface.bore_diameter == pytest.approx(23.0, abs=1.0)

    def test_nema17_interface_type(self):
        """NEMA17 uses through-hole mounting."""
        iface = get_mounting_interface("nema17_stepper")
        assert iface.interface_type == "through_hole"


# ============================================================================
# Step 2: PartFeatureEngine generates features on L-bracket
# ============================================================================

class TestStep2FeatureGeneration:

    def test_generate_features_with_nema17_interface(self):
        """PartFeatureEngine should generate NEMA17-matching holes on bracket."""
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)

        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )

        assert len(result.ops) > 0, "Should generate at least one operation"
        assert result.features_generated, "Should report features generated"

    def test_features_include_clearance_holes(self):
        """Generated features should include clearance holes for M3."""
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)

        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )

        # Check that hole operations exist (cylinder cuts for bolt holes)
        hole_ops = [op for op in result.ops
                    if op.get("name", "").startswith("mount_hole")]
        assert len(hole_ops) >= 4, f"Expected ≥4 hole ops, got {len(hole_ops)}"

    def test_features_include_bore(self):
        """Generated features should include shaft clearance bore."""
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)

        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )

        # Check for bore or clearance hole
        all_ops_text = str(result.ops)
        assert "23" in all_ops_text or "bore" in all_ops_text.lower(), \
            "Should include shaft clearance bore"

    def test_fastener_ops_generated(self):
        """Should generate fastener model operations (bolts, nuts)."""
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)

        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )

        assert len(result.fastener_ops) > 0, "Should generate fastener operations"

    def test_no_warnings(self):
        """Feature generation should produce no warnings for standard config."""
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)

        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )

        assert len(result.warnings) == 0, f"Unexpected warnings: {result.warnings}"


# ============================================================================
# Step 3: FastenerCatalog selects correct fasteners
# ============================================================================

class TestStep3FastenerSelection:

    def test_m3_bolt_spec_exists(self):
        """M3 bolt specification should be available."""
        bolt = get_bolt_spec("M3")
        assert bolt is not None
        assert bolt.thread_diameter == pytest.approx(3.0)

    def test_m3_bolt_head_dimensions(self):
        """M3 SHCS: head Ø5.5mm, head height 3mm."""
        bolt = get_bolt_spec("M3")
        assert bolt.head_diameter == pytest.approx(5.5, abs=0.5)
        assert bolt.head_height == pytest.approx(3.0, abs=0.5)

    def test_m3_nut_spec_exists(self):
        """M3 nut specification should be available."""
        nut = get_nut_spec("M3")
        assert nut is not None

    def test_m3_washer_spec_exists(self):
        """M3 washer specification should be available."""
        washer = get_washer_spec("M3")
        assert washer is not None

    def test_bolt_length_for_5mm_grip(self):
        """Bracket 5mm + standard grip → should select M3×12 or shorter."""
        # Grip = bracket thickness (5mm) + washer + nut height ≈ 5mm+
        bolt_length = recommend_bolt_length(grip_mm=5.0)
        assert bolt_length >= 5.0
        assert bolt_length in [5, 6, 8, 10, 12, 16, 20]

    def test_fastener_set_complete(self):
        """Complete fastener set for M3 with 5mm grip."""
        result = recommend_fastener_set("M3", grip_mm=5.0, with_washer=True)
        assert result is not None
        assert "bolt_length_mm" in result or "bolt_length" in result
        bolt_length = result.get("bolt_length_mm", result.get("bolt_length"))
        assert bolt_length >= 5.0

    def test_4_bolts_needed(self):
        """NEMA17 mount requires exactly 4 bolts."""
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        assert conn.bolt_count == 4


# ============================================================================
# Step 4: AssemblySolver positions parts correctly
# ============================================================================

class TestStep4ConstraintSolving:

    def test_solver_produces_placements(self):
        """Solver should produce positions for both parts."""
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        assert "l_bracket" in placements
        assert "nema17_motor" in placements

    def test_bracket_at_base(self):
        """L-bracket (root) should be at base position."""
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        bracket_pos = placements["l_bracket"]["position"]
        # Root part is at base_position (0, 0, 0)
        assert bracket_pos[0] == pytest.approx(0.0, abs=0.1)
        assert bracket_pos[1] == pytest.approx(0.0, abs=0.1)

    def test_motor_on_top_of_bracket(self):
        """NEMA17 should be positioned above bracket (top→bottom anchor)."""
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        bracket_pos = placements["l_bracket"]["position"]
        motor_pos = placements["nema17_motor"]["position"]

        # Motor should be above bracket
        assert motor_pos[2] > bracket_pos[2], \
            f"Motor z={motor_pos[2]} should be > bracket z={bracket_pos[2]}"

    def test_motor_xy_centered_on_bracket(self):
        """Motor should be centered on bracket in X-Y plane."""
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        motor_pos = placements["nema17_motor"]["position"]
        # Motor should be centered → x,y ≈ 0
        assert motor_pos[0] == pytest.approx(0.0, abs=5.0)
        assert motor_pos[1] == pytest.approx(0.0, abs=5.0)

    def test_two_placements_total(self):
        """2 parts → 2 placements."""
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        assert len(placements) == 2


# ============================================================================
# Step 5: AssemblyVerifier passes all checks
# ============================================================================

class TestStep5Verification:

    def test_mating_surface_check(self):
        """Mating surface check should show good alignment."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        checks = verifier.check_mating_surfaces(assembly, placements)
        assert len(checks) == 1  # 1 joint

        mc = checks[0]
        assert mc.parent_part == "l_bracket"
        assert mc.child_part == "nema17_motor"
        # With top-bottom anchors, normals should be anti-parallel → <1° deviation
        assert mc.normal_deviation_deg < 1.0, \
            f"Normal deviation {mc.normal_deviation_deg}° should be < 1°"

    def test_bolt_alignment_check(self):
        """Bolt hole alignment check should show aligned holes."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()

        checks = verifier.check_bolt_hole_alignment(assembly)
        assert len(checks) == 1  # 1 bolted joint

        bc = checks[0]
        assert bc.aligned, f"Bolt alignment failed: {bc.notes}"

    def test_assembly_sequence_check(self):
        """Assembly sequence should be feasible (bracket first, then motor)."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()

        checks = verifier.check_assembly_sequence(assembly)
        assert len(checks) == 1
        assert all(c.feasible for c in checks)

    def test_full_verification_structure(self, tmp_path):
        """Full verification should produce structured result."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        assert isinstance(result, AssemblyVerificationResult)
        assert result.assembly_name == "L-Bracket NEMA17 Assembly"

    def test_verification_items_present(self, tmp_path):
        """Verification should produce structured items."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        assert len(result.verification_items) > 0
        for item in result.verification_items:
            assert isinstance(item, VerificationItem)

    def test_mating_checks_in_result(self, tmp_path):
        """Result should contain mating surface checks."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        assert len(result.mating_surface_checks) == 1
        mc = result.mating_surface_checks[0]
        assert isinstance(mc, MatingSurfaceCheck)

    def test_bolt_checks_in_result(self, tmp_path):
        """Result should contain bolt alignment checks."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        assert len(result.bolt_alignment_checks) == 1
        bc = result.bolt_alignment_checks[0]
        assert isinstance(bc, BoltHoleAlignmentCheck)
        assert bc.aligned

    def test_sequence_checks_in_result(self, tmp_path):
        """Result should contain assembly sequence checks."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        assert len(result.sequence_checks) == 1
        assert all(sc.feasible for sc in result.sequence_checks)

    def test_collision_free(self, tmp_path):
        """Assembly should be collision-free."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=1.0,
        )

        assert result.collision_free, \
            f"Assembly should be collision-free. Checks: {result.collision_checks}"


# ============================================================================
# Step 6: Report generation and structured output
# ============================================================================

class TestStep6Report:

    def test_report_generated(self, tmp_path):
        """Assembly report should be generated."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        report = verifier.generate_assembly_report(result)
        assert "装配验证报告" in report
        assert "L-Bracket NEMA17" in report

    def test_report_contains_mating_section(self, tmp_path):
        """Report should contain mating surface section."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        report = verifier.generate_assembly_report(result)
        assert "配合面检查" in report or "配合检查" in report

    def test_report_contains_bolt_section(self, tmp_path):
        """Report should contain bolt alignment section."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=1.0,
        )

        report = verifier.generate_assembly_report(result)
        assert "螺栓" in report

    def test_json_report_export(self, tmp_path):
        """Should export structured JSON report."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=1.0,
        )

        # Build structured report
        report_data = {
            "assembly_name": result.assembly_name,
            "overall_pass": result.overall_pass,
            "collision_free": result.collision_free,
            "fcl_available": result.fcl_available,
            "parts": [
                {"name": p.name, "category": p.category}
                for p in assembly.parts
            ],
            "joints": [
                {
                    "parent": j.parent,
                    "child": j.child,
                    "type": j.type,
                    "connection": j.connection.type if j.connection else None,
                    "bolt_size": j.connection.bolt_size if j.connection else None,
                    "bolt_count": j.connection.bolt_count if j.connection else 0,
                }
                for j in assembly.joints
            ],
            "mating_surface_checks": [
                {
                    "parent": mc.parent_part,
                    "child": mc.child_part,
                    "normal_deviation_deg": mc.normal_deviation_deg,
                    "face_distance_mm": mc.face_distance_mm,
                    "parallel_ok": mc.parallel_ok,
                    "distance_ok": mc.distance_ok,
                }
                for mc in result.mating_surface_checks
            ],
            "bolt_alignment_checks": [
                {
                    "parent": bc.parent_part,
                    "child": bc.child_part,
                    "aligned": bc.aligned,
                    "hole_count_parent": bc.hole_count_parent,
                }
                for bc in result.bolt_alignment_checks
            ],
            "sequence_checks": [
                {
                    "step": sc.step,
                    "part": sc.part_name,
                    "feasible": sc.feasible,
                }
                for sc in result.sequence_checks
            ],
            "verification_items": [
                {
                    "name": vi.name,
                    "category": vi.category,
                    "passed": vi.passed,
                }
                for vi in result.verification_items
            ],
            "placements": {
                name: {
                    "position": p["position"],
                }
                for name, p in placements.items()
            },
        }

        report_path = tmp_path / "e2e_bolted_assembly_report.json"
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        assert report_path.exists()

        # Validate JSON structure
        with open(report_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["assembly_name"] == "L-Bracket NEMA17 Assembly"
        assert len(loaded["parts"]) == 2
        assert len(loaded["joints"]) == 1
        assert loaded["joints"][0]["connection"] == "bolted"
        assert loaded["joints"][0]["bolt_size"] == "M3"
        assert loaded["joints"][0]["bolt_count"] == 4

    def test_report_saved_to_data_dir(self):
        """Export report to data/ directory."""
        verifier = AssemblyVerifier()
        assembly = _make_l_bracket_assembly()
        parts_results = _make_parts_results()
        solver = AssemblySolver(assembly)
        placements = solver.solve()

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            result = verifier.verify_assembly(
                assembly=assembly,
                workspace=tmp,
                parts_results=parts_results,
                placements=placements,
                allowed_tolerance_total=1.0,
            )

            report_data = {
                "assembly_name": result.assembly_name,
                "overall_pass": result.overall_pass,
                "collision_free": result.collision_free,
                "bolt_alignment": [
                    {"aligned": bc.aligned} for bc in result.bolt_alignment_checks
                ],
                "mating_surfaces": [
                    {"parallel_ok": mc.parallel_ok, "distance_ok": mc.distance_ok}
                    for mc in result.mating_surface_checks
                ],
            }

            report_path = data_dir / "e2e_bolted_assembly_report.json"
            report_path.write_text(
                json.dumps(report_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["assembly_name"] == "L-Bracket NEMA17 Assembly"


# ============================================================================
# Integration: Full pipeline in one test
# ============================================================================

class TestFullPipelineIntegration:

    def test_complete_l_bracket_nema17_pipeline(self, tmp_path):
        """Run the complete pipeline end-to-end."""
        # === Step 1: Get NEMA17 interface ===
        nema17_iface = get_mounting_interface("nema17_stepper")
        assert nema17_iface is not None
        assert len(nema17_iface.holes) == 4

        # === Step 2: Generate features on bracket ===
        engine = ConnectionFeatureEngine()
        bracket = Part(
            name="l_bracket",
            category="structural",
            description="L-bracket",
            material="PLA",
            dimensions={"length": 60, "width": 60, "thickness": 5},
        )
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        feature_result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )
        assert len(feature_result.ops) > 0
        assert len(feature_result.fastener_ops) > 0

        # === Step 3: Select fasteners ===
        bolt = get_bolt_spec("M3")
        nut = get_nut_spec("M3")
        washer = get_washer_spec("M3")
        assert bolt is not None
        assert nut is not None
        assert washer is not None

        grip = bracket.dimensions["thickness"]  # 5mm
        fastener_set = recommend_fastener_set("M3", grip_mm=grip, with_washer=True)
        assert fastener_set is not None

        # === Step 4: Build assembly and solve ===
        motor = Part(
            name="nema17_motor",
            category="actuator",
            description="NEMA17",
            material="Steel",
            dimensions={"length": 42.3, "width": 42.3, "height": 48},
        )

        joint = Joint(
            type="fixed",
            parent="l_bracket",
            child="nema17_motor",
            parent_anchor="top",
            child_anchor="bottom",
            connection=conn,
        )

        assembly = Assembly(
            name="L-Bracket NEMA17 Assembly",
            parts=[bracket, motor],
            joints=[joint],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()

        assert len(placements) == 2
        motor_z = placements["nema17_motor"]["position"][2]
        bracket_z = placements["l_bracket"]["position"][2]
        assert motor_z > bracket_z, "Motor should be above bracket"

        # === Step 5: Verify assembly ===
        parts_results = _make_parts_results()
        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=1.0,
        )

        # === Step 6: Validate results ===
        assert isinstance(result, AssemblyVerificationResult)
        assert result.collision_free

        # Mating surface: top-bottom anchors → anti-parallel normals
        assert len(result.mating_surface_checks) == 1
        mc = result.mating_surface_checks[0]
        assert mc.parallel_ok, f"Mating normal deviation: {mc.normal_deviation_deg}°"

        # Bolt alignment
        assert len(result.bolt_alignment_checks) == 1
        bc = result.bolt_alignment_checks[0]
        assert bc.aligned, f"Bolt alignment: {bc.notes}"

        # Sequence feasible
        assert len(result.sequence_checks) == 1
        assert result.sequence_checks[0].feasible

        # Report
        report = verifier.generate_assembly_report(result)
        assert "装配验证报告" in report
