"""End-to-end assembly flow tests — 3 test cases.

Tests the full pipeline without LLM dependency:
  Assembly definition → AssemblySolver → positions → IK verification

Cases:
  1. Flange connection: 2 parts + 1 assembly (simple)
  2. 3-DOF Robotic arm: 8 parts + assembly + IK (full chain)
  3. 4-wheel differential chassis: 10 parts (scaling)

Usage:
  python tests/test_e2e_assembly.py          # Run all, print report
  pytest tests/test_e2e_assembly.py -v       # Via pytest

Report saved to: data/e2e_assembly_report.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.ik_solver import solve_ik, _fk_verify


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class AssemblyCaseResult:
    case_id: str
    case_name: str
    part_count: int
    joint_count: int
    success: bool = False
    positions_computed: bool = False
    chain_valid: bool = False          # All parts have valid positions
    ik_targets_tested: int = 0
    ik_reachable: int = 0
    ik_avg_error_mm: float = 0.0
    solver_time_ms: float = 0.0
    ik_time_ms: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Case 1: Flange connection (2 parts + 1 assembly)
# ---------------------------------------------------------------------------

def _build_flange_assembly() -> Assembly:
    """Two pipes connected by a flange."""
    return Assembly(
        name="Flange Connection",
        description="两段管道用法兰连接",
        parts=[
            Part(
                name="pipe_lower", category="structural",
                description="下段管道",
                material="Steel",
                dimensions={"outer_diameter": 50, "height": 100, "wall_thickness": 5},
            ),
            Part(
                name="flange_plate", category="connector",
                description="法兰盘",
                material="Steel",
                dimensions={"outer_diameter": 80, "height": 12},
            ),
        ],
        joints=[
            Joint("fixed", "pipe_lower", "flange_plate",
                  description="法兰焊接在管道顶部",
                  parent_anchor="top", child_anchor="bottom"),
        ],
    )


# ---------------------------------------------------------------------------
# Case 2: 3-DOF Robotic arm (8 parts, built-in)
# ---------------------------------------------------------------------------

def _get_robotic_arm_assembly() -> Assembly:
    from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
    return ROBOTIC_ARM_ASSEMBLY


# ---------------------------------------------------------------------------
# Case 3: 4-wheel differential chassis (10 parts)
# ---------------------------------------------------------------------------

def _build_chassis_assembly() -> Assembly:
    """4-wheel differential drive mobile chassis.

    Parts: base_plate + 4*motor_mount + 4*wheel + battery_box = 10
    Layout:
      base_plate (bottom)
        ├── motor_front_left  (top → bottom, fixed)
        ├── motor_front_right (top → bottom, fixed)
        ├── motor_rear_left   (top → bottom, fixed)
        ├── motor_rear_right  (top → bottom, fixed)
        └── battery_box       (top → bottom, fixed)

      Each motor_mount has a wheel child (top → bottom, revolute).
    """
    base_w, base_l, base_h = 200, 250, 8
    motor_w, motor_h = 30, 40
    wheel_dia, wheel_h = 60, 20
    bat_w, bat_l, bat_h = 80, 120, 40

    parts = [
        Part(name="base_plate", category="structural",
             description="底盘底板", material="PLA",
             dimensions={"width": base_w, "length": base_l, "height": base_h}),
        # 4 motor mounts at corners
        Part(name="motor_front_left", category="actuator",
             description="左前电机座", material="PLA",
             dimensions={"width": motor_w, "length": motor_w, "height": motor_h}),
        Part(name="motor_front_right", category="actuator",
             description="右前电机座", material="PLA",
             dimensions={"width": motor_w, "length": motor_w, "height": motor_h}),
        Part(name="motor_rear_left", category="actuator",
             description="左后电机座", material="PLA",
             dimensions={"width": motor_w, "length": motor_w, "height": motor_h}),
        Part(name="motor_rear_right", category="actuator",
             description="右后电机座", material="PLA",
             dimensions={"width": motor_w, "length": motor_w, "height": motor_h}),
        # 4 wheels
        Part(name="wheel_front_left", category="wheel",
             description="左前轮", material="TPU",
             dimensions={"outer_diameter": wheel_dia, "height": wheel_h}),
        Part(name="wheel_front_right", category="wheel",
             description="右前轮", material="TPU",
             dimensions={"outer_diameter": wheel_dia, "height": wheel_h}),
        Part(name="wheel_rear_left", category="wheel",
             description="左后轮", material="TPU",
             dimensions={"outer_diameter": wheel_dia, "height": wheel_h}),
        Part(name="wheel_rear_right", category="wheel",
             description="右后轮", material="TPU",
             dimensions={"outer_diameter": wheel_dia, "height": wheel_h}),
        # Battery box
        Part(name="battery_box", category="power",
             description="电池盒", material="PLA",
             dimensions={"width": bat_w, "length": bat_l, "height": bat_h}),
    ]

    joints = [
        # Motor mounts on base plate (with offsets to corners)
        Joint("fixed", "base_plate", "motor_front_left",
              description="左前电机座",
              parent_anchor="top", child_anchor="bottom",
              offset=(-70, -90, 0)),
        Joint("fixed", "base_plate", "motor_front_right",
              description="右前电机座",
              parent_anchor="top", child_anchor="bottom",
              offset=(-70, 90, 0)),
        Joint("fixed", "base_plate", "motor_rear_left",
              description="左后电机座",
              parent_anchor="top", child_anchor="bottom",
              offset=(70, -90, 0)),
        Joint("fixed", "base_plate", "motor_rear_right",
              description="右后电机座",
              parent_anchor="top", child_anchor="bottom",
              offset=(70, 90, 0)),
        # Wheels on motor mounts (revolute)
        Joint("revolute", "motor_front_left", "wheel_front_left",
              (-360, 360), "左前轮旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "motor_front_right", "wheel_front_right",
              (-360, 360), "右前轮旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "motor_rear_left", "wheel_rear_left",
              (-360, 360), "左后轮旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "motor_rear_right", "wheel_rear_right",
              (-360, 360), "右后轮旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        # Battery box on base plate center
        Joint("fixed", "base_plate", "battery_box",
              description="电池盒在底板中央",
              parent_anchor="top", child_anchor="bottom"),
    ]

    return Assembly(
        name="4-Wheel Differential Chassis",
        description="4轮差速移动底盘",
        parts=parts,
        joints=joints,
    )


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def _run_assembly_test(
    case_id: str,
    case_name: str,
    assembly: Assembly,
    ik_targets: list[tuple[float, float, float]] | None = None,
) -> AssemblyCaseResult:
    """Run a complete assembly test: solve → validate → optional IK."""
    result = AssemblyCaseResult(
        case_id=case_id,
        case_name=case_name,
        part_count=len(assembly.parts),
        joint_count=len(assembly.joints),
    )

    try:
        # Step 1: Solve assembly positions
        t0 = time.perf_counter()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        result.solver_time_ms = (time.perf_counter() - t0) * 1000

        # Validate: all parts have positions
        result.positions_computed = len(placements) == len(assembly.parts)
        if not result.positions_computed:
            missing = {p.name for p in assembly.parts} - set(placements.keys())
            result.error = f"Missing placements: {missing}"
            return result

        # Validate: positions are finite numbers
        for pname, p in placements.items():
            for v in p["position"]:
                if not math.isfinite(v):
                    result.error = f"Non-finite position for {pname}: {p['position']}"
                    return result

        result.chain_valid = True

        # Step 2: IK tests (if targets provided)
        if ik_targets:
            errors = []
            t0 = time.perf_counter()
            for target in ik_targets:
                ik_result = solve_ik(
                    assembly, target=target,
                    approach="auto", tolerance_mm=5.0,
                    max_iterations=500,
                )
                result.ik_targets_tested += 1
                if ik_result.reachable:
                    result.ik_reachable += 1
                errors.append(ik_result.error_mm)
            result.ik_time_ms = (time.perf_counter() - t0) * 1000
            result.ik_avg_error_mm = sum(errors) / len(errors) if errors else 0.0

        result.success = result.positions_computed and result.chain_valid

    except Exception as e:
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Test cases as pytest tests
# ---------------------------------------------------------------------------

class TestFlangeConnection:
    """Case 1: Two pipes with flange — simplest assembly."""

    def test_assembly_solve(self):
        asm = _build_flange_assembly()
        result = _run_assembly_test("flange", "法兰连接", asm)
        assert result.success, f"Assembly solve failed: {result.error}"

    def test_two_parts_placed(self):
        asm = _build_flange_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert len(placements) == 2

    def test_flange_above_pipe(self):
        asm = _build_flange_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        pipe_z = placements["pipe_lower"]["position"][2]
        flange_z = placements["flange_plate"]["position"][2]
        assert flange_z > pipe_z, "Flange should be above pipe"

    def test_solver_time_reasonable(self):
        asm = _build_flange_assembly()
        result = _run_assembly_test("flange", "法兰连接", asm)
        assert result.solver_time_ms < 100, f"Solver too slow: {result.solver_time_ms:.1f}ms"


class TestRoboticArm:
    """Case 2: 3-DOF robotic arm — full chain with IK."""

    def test_assembly_solve(self):
        asm = _get_robotic_arm_assembly()
        result = _run_assembly_test("robotic_arm", "3-DOF 机械臂", asm)
        assert result.success, f"Assembly solve failed: {result.error}"

    def test_all_8_parts_placed(self):
        asm = _get_robotic_arm_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert len(placements) == len(asm.parts)

    def test_chain_ascending(self):
        asm = _get_robotic_arm_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        chain = ["base_plate", "base_joint_housing", "shoulder_link",
                 "elbow_joint", "forearm_link", "wrist_joint",
                 "end_effector_mount"]
        prev_z = -1e9
        for name in chain:
            z = placements[name]["position"][2]
            assert z >= prev_z, f"{name} z={z} < prev {prev_z}"
            prev_z = z

    def test_ik_home_position(self):
        """IK should find zero-error solution for home position."""
        asm = _get_robotic_arm_assembly()
        solver = AssemblySolver(asm)
        home = solver.solve()
        ee = home["end_effector_mount"]["position"]

        result = solve_ik(asm, target=tuple(ee), approach="ccd",
                          tolerance_mm=1.0, max_iterations=100)
        assert result.reachable, f"Home IK failed: error={result.error_mm:.2f}mm"

    def test_ik_reachable_target(self):
        """IK should reach a known-reachable target."""
        asm = _get_robotic_arm_assembly()
        # (50, 50, 100) was verified reachable
        result = solve_ik(asm, target=(50, 50, 100), approach="ccd",
                          tolerance_mm=5.0, max_iterations=500)
        assert result.error_mm < 10, f"IK error too large: {result.error_mm:.2f}mm"

    def test_ik_fk_roundtrip(self):
        """IK solution verified by FK should match target."""
        asm = _get_robotic_arm_assembly()
        target = (50, 50, 100)
        result = solve_ik(asm, target=target, approach="ccd",
                          tolerance_mm=5.0, max_iterations=500)

        # FK verify
        ee_actual = _fk_verify(asm, result.joint_angles, "end_effector_mount")
        error = math.sqrt(
            (target[0] - ee_actual[0]) ** 2 +
            (target[1] - ee_actual[1]) ** 2 +
            (target[2] - ee_actual[2]) ** 2
        )
        assert error < 10, f"FK roundtrip error: {error:.2f}mm"

    def test_full_pipeline(self):
        """Full pipeline: solve + IK on multiple targets."""
        asm = _get_robotic_arm_assembly()
        targets = [
            (0, 0, 166.5),    # Home
            (50, 50, 100),    # Diagonal
            (80, 0, 80),      # Forward
        ]
        result = _run_assembly_test("robotic_arm", "3-DOF 机械臂", asm, ik_targets=targets)
        assert result.success
        assert result.ik_targets_tested == 3
        # At least home should be reachable
        assert result.ik_reachable >= 1

    def test_solver_performance(self):
        asm = _get_robotic_arm_assembly()
        result = _run_assembly_test("robotic_arm", "3-DOF 机械臂", asm)
        assert result.solver_time_ms < 100, f"Solver too slow: {result.solver_time_ms:.1f}ms"


class TestDifferentialChassis:
    """Case 3: 4-wheel differential chassis — 10 parts."""

    def test_assembly_solve(self):
        asm = _build_chassis_assembly()
        result = _run_assembly_test("chassis", "4轮差速底盘", asm)
        assert result.success, f"Assembly solve failed: {result.error}"

    def test_all_10_parts_placed(self):
        asm = _build_chassis_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert len(placements) == 10

    def test_wheels_above_motors(self):
        asm = _build_chassis_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()

        for corner in ["front_left", "front_right", "rear_left", "rear_right"]:
            motor_z = placements[f"motor_{corner}"]["position"][2]
            wheel_z = placements[f"wheel_{corner}"]["position"][2]
            assert wheel_z > motor_z, f"Wheel {corner} z={wheel_z} should be > motor z={motor_z}"

    def test_motors_above_base(self):
        asm = _build_chassis_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        base_z = placements["base_plate"]["position"][2]

        for corner in ["front_left", "front_right", "rear_left", "rear_right"]:
            motor_z = placements[f"motor_{corner}"]["position"][2]
            assert motor_z > base_z, f"Motor {corner} should be above base"

    def test_battery_box_above_base(self):
        asm = _build_chassis_assembly()
        solver = AssemblySolver(asm)
        placements = solver.solve()
        base_z = placements["base_plate"]["position"][2]
        bat_z = placements["battery_box"]["position"][2]
        assert bat_z > base_z

    def test_wheel_rotation_does_not_change_z(self):
        """Rotating wheels around Z should not change Z position."""
        asm = _build_chassis_assembly()
        solver = AssemblySolver(asm)

        p0 = solver.solve(joint_angles={"wheel_front_left": 0})
        p90 = solver.solve(joint_angles={"wheel_front_left": 90})

        z0 = p0["wheel_front_left"]["position"][2]
        z90 = p90["wheel_front_left"]["position"][2]
        assert z0 == pytest.approx(z90, abs=0.01)

    def test_solver_performance(self):
        asm = _build_chassis_assembly()
        result = _run_assembly_test("chassis", "4轮差速底盘", asm)
        assert result.solver_time_ms < 100, f"Solver too slow: {result.solver_time_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _run_all_cases() -> list[AssemblyCaseResult]:
    """Run all 3 test cases and return results."""
    cases = [
        ("flange", "法兰连接", _build_flange_assembly(), None),
        ("robotic_arm", "3-DOF 机械臂", _get_robotic_arm_assembly(),
         [(0, 0, 166.5), (50, 50, 100), (80, 0, 80)]),
        ("chassis", "4轮差速底盘", _build_chassis_assembly(), None),
    ]

    results = []
    for case_id, name, asm, targets in cases:
        r = _run_assembly_test(case_id, name, asm, ik_targets=targets)
        results.append(r)
    return results


def print_report(results: list[AssemblyCaseResult]) -> None:
    """Print a structured test report."""
    print("\n" + "=" * 70)
    print("  端到端装配流程测试报告")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.success)

    print(f"\n总览: {passed}/{total} 通过\n")
    print(f"{'案例':<20} {'零件':>4} {'关节':>4} {'求解':>6} {'耗时':>8} {'IK':>8}")
    print("-" * 70)

    for r in results:
        solve_str = "OK" if r.positions_computed else "FAIL"
        ik_str = f"{r.ik_reachable}/{r.ik_targets_tested}" if r.ik_targets_tested > 0 else "N/A"
        print(f"{r.case_name:<20} {r.part_count:>4} {r.joint_count:>4} "
              f"{solve_str:>6} {r.solver_time_ms:>6.1f}ms {ik_str:>8}")

        if r.error:
            print(f"  ERROR: {r.error}")

    # Summary
    print("\n" + "-" * 70)
    total_parts = sum(r.part_count for r in results)
    total_joints = sum(r.joint_count for r in results)
    total_ik = sum(r.ik_targets_tested for r in results)
    total_ik_ok = sum(r.ik_reachable for r in results)
    avg_solver = sum(r.solver_time_ms for r in results) / max(total, 1)

    print(f"总零件: {total_parts}  总关节: {total_joints}  "
          f"IK: {total_ik_ok}/{total_ik}  平均求解: {avg_solver:.1f}ms")

    all_passed = all(r.success for r in results)
    print(f"\n最终结果: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70 + "\n")


def save_report(results: list[AssemblyCaseResult]) -> None:
    """Save results as JSON."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "e2e_assembly_report.json"

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.success),
            "total_parts": sum(r.part_count for r in results),
            "total_joints": sum(r.joint_count for r in results),
            "ik_tested": sum(r.ik_targets_tested for r in results),
            "ik_reachable": sum(r.ik_reachable for r in results),
            "avg_solver_ms": round(sum(r.solver_time_ms for r in results) / max(len(results), 1), 2),
        },
        "cases": [asdict(r) for r in results],
    }

    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved to: {path}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = _run_all_cases()
    print_report(results)
    save_report(results)

    all_passed = all(r.success for r in results)
    exit(0 if all_passed else 1)
