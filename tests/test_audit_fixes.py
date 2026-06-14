"""Tests for audit-report P0/P1 fixes (G1-G6).

Covers:
- G1: COMPLEX_ROBOT_KEYWORDS includes arm / DOF keywords
- G2: firmware defines servo_raw_write + SERVO_OBJECTS
- G3: BOM mass uses material-aware density
- G5: build_assembly_script imports real STL with primitive fallback
- G6: _VLM_FIX_PROMPT formats without error
- C1: revolute joints default to press_fit, prismatic stays null
"""

from __future__ import annotations

import json
import math

import pytest

from lang3d.agent.planner import COMPLEX_ROBOT_KEYWORDS, Planner
from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY, Assembly, Part
from lang3d.tools.assembly_generator import _normalize_gripper_fingers, _parse_assembly_json
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.bom_gen import _part_to_bom_entry, generate_bom
from lang3d.tools.code_gen import generate_firmware
from lang3d.tools.freecad import build_assembly_script
from lang3d.tools.urdf_export import AssemblyToURDF


# ============================================================================
# G1: COMPLEX_ROBOT_KEYWORDS
# ============================================================================

class TestComplexRobotKeywords:
    """G1: DoF-qualified arms are recognized as complex robots."""

    def test_complex_keywords(self):
        # DoF-specific qualifiers are the unambiguous complex-robot signal.
        assert "4dof" in COMPLEX_ROBOT_KEYWORDS
        assert "4-dof" in COMPLEX_ROBOT_KEYWORDS
        assert "4自由度" in COMPLEX_ROBOT_KEYWORDS
        assert "6dof" in COMPLEX_ROBOT_KEYWORDS
        assert "6-dof" in COMPLEX_ROBOT_KEYWORDS
        assert "6自由度" in COMPLEX_ROBOT_KEYWORDS

    def test_dof_arm_routes_to_complex_robot(self):
        # A 4DOF/6DOF arm should use the complex_robot pipeline.
        assert Planner._detect_task_type("设计一个4自由度机械臂") == "complex_robot"
        assert Planner._detect_task_type("build a 6dof robotic arm") == "complex_robot"

    def test_plain_arm_still_assembly(self):
        # A plain arm without DoF qualifier must stay on the assembly path.
        assert Planner._detect_task_type("装配一个3自由度机械臂") == "assembly"
        assert Planner._detect_task_type("Create an assembly of robotic arm") == "assembly"


# ============================================================================
# G2: firmware servo_raw_write
# ============================================================================

class TestServoRawWriteDefined:
    """G2: firmware defines servo_raw_write + SERVO_OBJECTS array."""

    @pytest.fixture
    def firmware(self):
        return generate_firmware(ROBOTIC_ARM_ASSEMBLY, ["MG996R", "MG996R", "DS3218", "SG90"])

    def test_servo_raw_write_defined(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "void servo_raw_write" in ino, "servo_raw_write must be defined in main .ino"

    def test_servo_objects_array_defined(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "SERVO_OBJECTS" in ino, "SERVO_OBJECTS pointer array must be declared"

    def test_servo_raw_write_uses_writeMicroseconds(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "writeMicroseconds" in ino


# ============================================================================
# G3: BOM material-aware density
# ============================================================================

class TestBomMaterialDensity:
    """G3: BOM mass reflects the part material density, not hardcoded PLA."""

    def test_bom_material_density(self):
        # 100 x 100 x 10 mm aluminum plate -> 100000 mm³
        part = Part(
            name="aluminum_plate",
            category="plate",
            description="Aluminum test plate",
            material="Aluminum",
            dimensions={"length": 100, "width": 100, "height": 10},
        )
        entry = _part_to_bom_entry(part)
        # Aluminum density 2700 kg/m³ -> 0.0027 g/mm³ -> 270 g
        assert entry["estimated_weight_g"] == pytest.approx(270.0, abs=0.5)

    def test_bom_not_pla_default_for_aluminum(self):
        part = Part(
            name="alu_block",
            category="block",
            description="Aluminum block",
            material="Aluminum",
            dimensions={"length": 100, "width": 100, "height": 10},
        )
        entry = _part_to_bom_entry(part)
        # PLA would give 125 g; Aluminum must be significantly higher
        assert entry["estimated_weight_g"] > 200.0


# ============================================================================
# G5: build_assembly_script STL import with fallback
# ============================================================================

class TestAssemblyScriptStlImport:
    """G5: build_assembly_script uses Mesh.read when stl_path present."""

    def test_assembly_script_stl_import(self):
        parts = [
            {
                "name": "base_link",
                "shape_type": "box",
                "dimensions": {"length": 80, "width": 80, "height": 10},
                "subsystem": "frame",
                "stl_path": "/tmp/fake/base_link.stl",
            }
        ]
        positions = {"base_link": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]}}
        script = build_assembly_script(assembly_parts=parts, positions=positions)
        assert "Mesh.read" in script, "script must attempt STL import via Mesh.read"
        assert "Part.Shape" in script, "script must convert mesh topology to Part.Shape"
        assert "makeBox" in script, "script must keep primitive fallback in else branch"
        assert "_os.path.exists" in script, "script must guard STL path with os.path.exists"

    def test_assembly_script_no_stl_uses_primitive(self):
        parts = [
            {
                "name": "base_link",
                "shape_type": "box",
                "dimensions": {"length": 80, "width": 80, "height": 10},
                "subsystem": "frame",
            }
        ]
        positions = {"base_link": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]}}
        script = build_assembly_script(assembly_parts=parts, positions=positions)
        assert "Mesh.read" not in script, "no stl_path -> no Mesh.read"
        assert "makeBox" in script


# ============================================================================
# G6: _VLM_FIX_PROMPT format
# ============================================================================

class TestVlmFixPromptFormat:
    """G6: _VLM_FIX_PROMPT formats with problems + description placeholders."""

    def test_vlm_fix_prompt_format(self):
        from lang3d.tools.assembly_generator import _VLM_FIX_PROMPT

        text = _VLM_FIX_PROMPT.format(
            problems="- Wheels off the ground\n- Missing gripper",
            description="4-DOF robotic arm with gripper",
        )
        assert "4-DOF robotic arm with gripper" in text
        assert "Wheels off the ground" in text
        # Classified fix guidance should be present
        assert "joint" in text.lower() or "offset" in text.lower()


# ============================================================================
# C1: connection_method defaults for revolute / fixed / prismatic
# ============================================================================

def _minimal_assembly_json(joints):
    """Build a minimal assembly JSON with the given joint dicts.

    parts are synthesized to satisfy parent/child references in the joints.
    """
    part_names = set()
    for j in joints:
        part_names.add(j["parent"])
        part_names.add(j["child"])
    parts = [
        {
            "name": n,
            "category": "structural",
            "description": f"{n} part",
            "material": "PLA",
            "dimensions": {"length": 40, "width": 20, "height": 10},
        }
        for n in sorted(part_names)
    ]
    return json.dumps({"name": "test_asm", "parts": parts, "joints": joints})


class TestConnectionMethodDefaults:
    """C1: revolute→press_fit, fixed→bolted, prismatic→null (intentional)."""

    def test_revolute_defaults_to_press_fit(self):
        asm_json = _minimal_assembly_json([
            {"type": "revolute", "parent": "base", "child": "link1"},
        ])
        asm = _parse_assembly_json(asm_json)
        joint = asm.joints[0]
        assert joint.connection is not None
        assert joint.connection.type == "press_fit"
        # interference should be a small positive value (typical bearing fit)
        assert joint.connection.interference_mm > 0

    def test_fixed_still_bolted(self):
        asm_json = _minimal_assembly_json([
            {"type": "fixed", "parent": "base", "child": "bracket"},
        ])
        asm = _parse_assembly_json(asm_json)
        joint = asm.joints[0]
        assert joint.connection is not None
        assert joint.connection.type == "bolted"
        assert joint.connection.bolt_size == "M3"
        assert joint.connection.bolt_count == 4

    def test_prismatic_stays_null(self):
        asm_json = _minimal_assembly_json([
            {"type": "prismatic", "parent": "rail", "child": "finger"},
        ])
        asm = _parse_assembly_json(asm_json)
        joint = asm.joints[0]
        # Sliding interface is not a fastening; null is intentional.
        assert joint.connection is None

    def test_explicit_connection_not_overwritten(self):
        # If the LLM already set bolted on a revolute joint, the fallback
        # must NOT overwrite it with press_fit.
        asm_json = _minimal_assembly_json([{
            "type": "revolute",
            "parent": "base",
            "child": "link1",
            "connection_method": "bolted",
            "connection_detail": {"bolt_size": "M4", "bolt_count": 2},
        }])
        asm = _parse_assembly_json(asm_json)
        joint = asm.joints[0]
        assert joint.connection is not None
        assert joint.connection.type == "bolted"
        assert joint.connection.bolt_size == "M4"
        assert joint.connection.bolt_count == 2


class TestBomRevoluteNoDefaultFasteners:
    """C1 downstream: revolute→press_fit must not pull in default M3×10."""

    def test_bom_revolute_no_default_fasteners(self):
        asm_json = _minimal_assembly_json([
            {"type": "revolute", "parent": "base", "child": "link1"},
            {"type": "revolute", "parent": "link1", "child": "link2"},
        ])
        asm = _parse_assembly_json(asm_json)
        bom = generate_bom(asm)
        std_names = " ".join(p.get("name", "") for p in bom["standard_parts"])
        # The fallback _add_default_fasteners emits "M3×10 螺丝";
        # press_fit revolute joints must NOT trigger that path.
        assert "M3×10 螺丝" not in std_names, (
            f"revolute press_fit joints must not add default M3×10 fasteners; "
            f"got standard_parts: {bom['standard_parts']}"
        )
        # Bearings should still be counted for revolute joints.
        assert "轴承" in std_names


# ============================================================================
# Audit P0: gripper finger anchors + URDF / kinematics validation
# ============================================================================


def _gripper_finger_assembly_json():
    """Build a minimal gripper assembly with inconsistent finger anchors.

    The LLM-emitted joints deliberately use ``front``/``back`` face anchors so
    that, before the C1 fix, the sanitizer would copy them through and let
    anchor displacement compound with the lateral offset.
    """
    return json.dumps({
        "name": "gripper_test",
        "parts": [
            {
                "name": "gripper_base",
                "category": "mechanical",
                "description": "gripper base",
                "material": "PLA",
                "dimensions": {"length": 40, "width": 30, "height": 20},
            },
            {
                "name": "finger_left",
                "category": "mechanical",
                "description": "left finger",
                "material": "PLA",
                "dimensions": {"length": 8, "width": 5, "height": 30},
            },
            {
                "name": "finger_right",
                "category": "mechanical",
                "description": "right finger",
                "material": "PLA",
                "dimensions": {"length": 8, "width": 5, "height": 30},
            },
        ],
        "joints": [
            {
                "type": "prismatic",
                "parent": "gripper_base",
                "child": "finger_left",
                "parent_anchor": "front",
                "child_anchor": "back",
            },
            {
                "type": "prismatic",
                "parent": "gripper_base",
                "child": "finger_right",
                "parent_anchor": "back",
                "child_anchor": "front",
            },
        ],
    })


class TestNormalizeGripperFingersCenterAnchors:
    """C1: finger joints are forced to ``center`` anchors."""

    def test_normalize_gripper_fingers_uses_center_anchors(self):
        asm = _parse_assembly_json(_gripper_finger_assembly_json())
        # Sanity: the parsed JSON really did carry non-center anchors, so the
        # sanitizer has something to fix.
        left_before = next(j for j in asm.joints if j.child == "finger_left")
        assert left_before.parent_anchor != "center"

        asm = _normalize_gripper_fingers(asm)

        left_joint = next(j for j in asm.joints if j.child == "finger_left")
        right_joint = next(j for j in asm.joints if j.child == "finger_right")
        for j in (left_joint, right_joint):
            assert j.parent_anchor == "center", (
                f"parent_anchor must be 'center', got {j.parent_anchor!r}"
            )
            assert j.child_anchor == "center", (
                f"child_anchor must be 'center', got {j.child_anchor!r}"
            )

    def test_normalize_gripper_fingers_gap_on_x_axis(self):
        """Gap must be on X (lateral), fingers extend forward in Y.

        Solver convention: front=(0,-1,0), back=(0,1,0) → arm extends in Y.
        FreeCAD finger length (X, 60 mm) maps to solver Y (forward) via the
        renderer's swap_xy.  Placing the ±gap on X keeps the two 60 mm bars
        parallel and side-by-side (a proper parallel-jaw gripper).  Putting
        the gap on Y would stack the fingers along the arm's reach axis,
        making them appear "sideways" (perpendicular to the arm) in renders.
        """
        asm = _parse_assembly_json(_gripper_finger_assembly_json())
        asm = _normalize_gripper_fingers(asm)

        left_joint = next(j for j in asm.joints if j.child == "finger_left")
        right_joint = next(j for j in asm.joints if j.child == "finger_right")

        # Prismatic axis is X so URDF kinematics slide fingers laterally.
        assert left_joint.axis == "x", (
            f"left finger axis must be 'x', got {left_joint.axis!r}"
        )
        assert right_joint.axis == "x", (
            f"right finger axis must be 'x', got {right_joint.axis!r}"
        )

        # Offsets must be ±X (lateral) so finger bars extend forward in Y.
        lx, ly, lz = left_joint.offset
        rx, ry, rz = right_joint.offset
        assert ly == 0.0 and ry == 0.0, (
            f"finger Y offset must be 0; got left={ly}, right={ry}"
        )
        assert lx < 0 and rx > 0, (
            f"left X offset must be negative, right positive; "
            f"got left_x={lx}, right_x={rx}"
        )
        assert lx + rx == 0.0, (
            f"X offsets must be symmetric; got left_x={lx}, right_x={rx}"
        )

    def test_normalize_gripper_fingers_urdf_origin_sane(self):
        import xml.etree.ElementTree as ET

        asm = _parse_assembly_json(_gripper_finger_assembly_json())
        asm = _normalize_gripper_fingers(asm)

        solver = AssemblySolver(asm)
        positions = solver.solve()
        xml_str = AssemblyToURDF(asm, positions=positions).convert()

        root = ET.fromstring(xml_str)
        finger_origins = []
        for je in root.findall(".//joint"):
            child_el = je.find("child")
            child_link = child_el.get("link", "") if child_el is not None else ""
            if "finger" not in child_link.lower():
                continue
            oe = je.find("origin")
            if oe is None:
                continue
            xyz = oe.get("xyz", "0 0 0").split()
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            finger_origins.append(math.sqrt(x * x + y * y + z * z))

        assert finger_origins, "expected at least one finger joint in URDF"
        # Before the C1 fix the left finger origin was 0.322 m.  Center anchors
        # must keep both finger origins well under 0.1 m (far below the C2
        # 0.2 m threshold).
        worst = max(finger_origins)
        assert worst < 0.1, (
            f"finger joint origin magnitude {worst:.3f} m exceeds 0.1 m; "
            f"all={finger_origins}"
        )


class TestPhase5UrdfOriginCheckCatchesAbsurd:
    """C2: Phase 5 flags movable joints placed far from their parent."""

    def test_phase5_urdf_origin_check_catches_absurd(self, tmp_path):
        # Synthetic URDF: a revolute joint with a 0.5 m origin — far beyond
        # the max(0.2 m, 2 x parent_dim) threshold.
        absurd_urdf = (
            "<robot name=\"absurd\">"
            "<link name=\"base\"/>"
            "<link name=\"arm\"/>"
            "<joint name=\"j1\" type=\"revolute\">"
            "<parent link=\"base\"/>"
            "<child link=\"arm\"/>"
            "<origin xyz=\"0.5 0 0\"/>"
            "</joint>"
            "</robot>"
        )
        (tmp_path / "urdf.xml").write_text(absurd_urdf, encoding="utf-8")

        assembly = Assembly(
            name="absurd",
            parts=[
                Part(
                    name="base",
                    category="structural",
                    description="base",
                    material="PLA",
                    dimensions={"length": 40, "width": 40, "height": 10},
                ),
                Part(
                    name="arm",
                    category="structural",
                    description="arm",
                    material="PLA",
                    dimensions={"length": 40, "width": 20, "height": 10},
                ),
            ],
            joints=[],
        )

        # Lazy import: test_e2e_production lives in the tests dir, which
        # pytest places on sys.path via its prepend import mode.
        from test_e2e_production import _phase5_content_validation

        checks: list[dict] = []
        _phase5_content_validation(checks, assembly, str(tmp_path))

        origin_check = next(
            (c for c in checks if c["step"] == "urdf_origins_sane"), None,
        )
        assert origin_check is not None, (
            f"urdf_origins_sane check not produced; got {[c['step'] for c in checks]}"
        )
        assert origin_check["status"] == "FAIL", (
            f"expected FAIL for 0.5 m origin, got {origin_check}"
        )
        assert origin_check.get("critical") is True


class TestPhase6WorkspaceNontrivialPasses4dof:
    """C4: a real arm reaches a non-trivial workspace volume."""

    def test_phase6_workspace_nontrivial_passes_4dof(self):
        from test_e2e_production import _phase6_physical_sanity

        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        positions = solver.solve()

        checks: list[dict] = []
        _phase6_physical_sanity(checks, ROBOTIC_ARM_ASSEMBLY, positions)

        ws_check = next(
            (c for c in checks if c["step"] == "workspace_nontrivial"), None,
        )
        assert ws_check is not None, (
            f"workspace_nontrivial check not produced; "
            f"got {[c['step'] for c in checks]}"
        )
        assert ws_check["status"] == "PASS", (
            f"expected workspace_nontrivial PASS for ROBOTIC_ARM_ASSEMBLY, "
            f"got {ws_check}"
        )
