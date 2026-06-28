"""End-to-end design test — "Design a 3-DOF desktop robotic arm".

Simulates the full production pipeline from requirements to deliverables:
  1. Assembly definition (parts + joints)
  2. Assembly constraint solving (auto-positioning)
  3. Inverse kinematics (reachability verification)
  4. Actuator selection (torque analysis + power budget)
  5. Firmware generation (IK + servo + sensors + serial)
  6. Sensor integration (encoders + limit switches + IMU)
  7. BOM generation (parts + standard + electronics + cost)
  8. Assembly guide (step-by-step instructions)
  9. Print optimization (orientation + tolerance + packing)
 10. Quality control (inspection + test + maintenance)

Usage:
  python tests/test_e2e_design.py          # Run all, print report
  pytest tests/test_e2e_design.py -v       # Via pytest

Report saved to: data/e2e_design_report.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.actuator_tools import (
    analyze_assembly_torques,
    power_budget,
    select_actuators,
)
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.bom_gen import generate_bom
from lang3d.tools.assembly_doc import generate_assembly_guide
from lang3d.tools.code_gen import generate_firmware, generate_wiring, generate_test_sequence
from lang3d.tools.ik_solver import solve_ik
from lang3d.tools.print_optimize import optimize_assembly_print
from lang3d.tools.quality import generate_quality_report


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    name: str
    success: bool = False
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DesignReport:
    requirement: str
    stages: list[PipelineStage] = field(default_factory=list)
    total_time_ms: float = 0.0
    all_passed: bool = False
    deliverables: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> PipelineStage:
        s = PipelineStage(name=name)
        self.stages.append(s)
        return s


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> DesignReport:
    """Run the complete design pipeline for a 3-DOF robotic arm."""
    report = DesignReport(
        requirement="设计一个 3 自由度桌面机械臂，工作半径 300mm，负载 200g，用 SG90 舵机",
    )

    assembly = ROBOTIC_ARM_ASSEMBLY
    actuator_ids = ["MG996R", "MG996R", "DS3218", "SG90"]
    sensor_ids = ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]
    controller = "esp32"
    t0 = time.perf_counter()

    # --- Stage 1: Assembly definition ---
    stage = report.stage("Assembly Definition")
    t1 = time.perf_counter()
    try:
        assert len(assembly.parts) >= 8
        assert len(assembly.joints) >= 6
        revolute = [j for j in assembly.joints if j.type == "revolute"]
        assert len(revolute) >= 4
        stage.details = {
            "parts": len(assembly.parts),
            "joints": len(assembly.joints),
            "revolute_joints": len(revolute),
        }
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 2: Assembly solving ---
    stage = report.stage("Assembly Solving")
    t1 = time.perf_counter()
    try:
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        assert len(placements) >= len(assembly.parts)
        # Verify all positions are finite
        for name, data in placements.items():
            pos = data.get("position", [0, 0, 0])
            assert all(math.isfinite(p) for p in pos), f"Non-finite position for {name}"
        stage.details = {"parts_positioned": len(placements)}
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 3: IK verification ---
    stage = report.stage("Inverse Kinematics")
    t1 = time.perf_counter()
    try:
        # Test multiple reachable targets
        # Full chain: base_h=24 + shoulder=35 + elbow=32.5 + wrist=55 + ee=20 = 166.5mm
        # CCD solver with tolerance 2mm for pipeline validation
        targets = [
            (80, 0, 80),
            (80, 30, 80),
            (90, 20, 90),
            (90, 0, 90),
        ]
        reachable = 0
        errors = []
        for target in targets:
            result = solve_ik(assembly, target, approach="auto", tolerance_mm=2.0, max_iterations=500)
            errors.append(result.error_mm)
            if result.reachable:
                reachable += 1
        stage.details = {
            "targets_tested": len(targets),
            "reachable": reachable,
            "avg_error_mm": round(sum(errors) / len(errors), 2) if errors else 0,
        }
        stage.success = reachable >= 1  # At least 1 target reachable
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 4: Actuator selection ---
    stage = report.stage("Actuator Selection")
    t1 = time.perf_counter()
    try:
        torque_analysis = analyze_assembly_torques(assembly, safety_factor=2.0)
        budget = power_budget(actuator_ids, duty_cycle=0.3, sensor_ids=sensor_ids)
        recs = select_actuators(min_torque_kgcm=5.0, count=3)
        assert len(torque_analysis) > 0
        assert budget["total_power_w"] > 0
        assert len(recs) > 0
        stage.details = {
            "joints_analyzed": len(torque_analysis),
            "total_power_w": budget["total_power_w"],
            "supply_power_w": budget["supply_power_w"],
            "sensor_count": len(budget.get("sensors", [])),
        }
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 5: Firmware generation ---
    stage = report.stage("Firmware Generation")
    t1 = time.perf_counter()
    try:
        firmware = generate_firmware(assembly, actuator_ids, controller, sensor_ids=sensor_ids)
        assert "robot_arm.ino" in firmware
        assert "ik_solver.h" in firmware
        assert "ik_solver.cpp" in firmware
        assert "servo_driver.h" in firmware
        assert "servo_driver.cpp" in firmware
        assert "sensor_driver.h" in firmware
        assert "sensor_driver.cpp" in firmware
        # Verify content
        ino = firmware["robot_arm.ino"]
        assert "void setup()" in ino
        assert "void loop()" in ino
        assert "Serial.begin" in ino
        stage.details = {"files_generated": len(firmware), "files": list(firmware.keys())}
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 6: Wiring diagram ---
    stage = report.stage("Wiring Diagram")
    t1 = time.perf_counter()
    try:
        wiring = generate_wiring(actuator_ids, controller)
        assert "MG996R" in wiring
        assert "GPIO" in wiring
        assert "GND" in wiring
        stage.details = {"length_chars": len(wiring)}
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 7: Test sequence ---
    stage = report.stage("Test Sequence")
    t1 = time.perf_counter()
    try:
        test_seq = generate_test_sequence(assembly, steps=8)
        assert "Phase 1" in test_seq
        assert "Phase 2" in test_seq
        assert "Phase 3" in test_seq
        assert "[ ]" in test_seq
        stage.details = {"length_chars": len(test_seq)}
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 8: BOM ---
    stage = report.stage("BOM Generation")
    t1 = time.perf_counter()
    try:
        bom = generate_bom(assembly, actuator_ids, sensor_ids, controller)
        assert len(bom["custom_parts"]) >= 8
        assert len(bom["standard_parts"]) > 0
        assert len(bom["electronics"]) > 0
        assert bom["cost_summary"]["total_cost_cny"] > 0
        stage.details = {
            "custom_parts": len(bom["custom_parts"]),
            "standard_parts": len(bom["standard_parts"]),
            "electronics": len(bom["electronics"]),
            "total_cost_cny": bom["cost_summary"]["total_cost_cny"],
        }
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 9: Assembly guide ---
    stage = report.stage("Assembly Guide")
    t1 = time.perf_counter()
    try:
        guide = generate_assembly_guide(assembly, actuator_ids, sensor_ids, controller)
        assert "装配指导书" in guide
        assert "零件清单" in guide
        assert "装配步骤" in guide
        assert "接线说明" in guide
        assert "校准" in guide
        assert "常见问题" in guide
        stage.details = {"length_chars": len(guide)}
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 10: Print optimization ---
    stage = report.stage("Print Optimization")
    t1 = time.perf_counter()
    try:
        print_result = optimize_assembly_print(assembly, quality="standard", material="PLA")
        assert len(print_result["parts"]) >= 8
        assert print_result["packing"]["plates_needed"] >= 1
        assert print_result["summary"]["total_print_time_min"] >= 0
        stage.details = {
            "parts_optimized": len(print_result["parts"]),
            "plates_needed": print_result["summary"]["plates_needed"],
            "plate_utilization_pct": print_result["summary"]["plate_utilization_pct"],
            "total_print_time_min": print_result["summary"]["total_print_time_min"],
        }
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Stage 11: Quality control ---
    stage = report.stage("Quality Control")
    t1 = time.perf_counter()
    try:
        quality = generate_quality_report(assembly)
        assert "inspection" in quality
        assert "test_procedure" in quality
        assert "maintenance" in quality
        insp = quality["inspection"]
        assert insp["summary"]["total_dimensions"] > 0
        test = quality["test_procedure"]
        assert len(test["phases"]) >= 4
        maint = quality["maintenance"]
        assert len(maint["schedules"]) >= 3
        stage.details = {
            "dimensions_to_inspect": insp["summary"]["total_dimensions"],
            "critical_dimensions": insp["summary"]["critical_dimensions"],
            "test_phases": len(test["phases"]),
            "maintenance_schedules": len(maint["schedules"]),
        }
        stage.success = True
    except AssertionError as e:
        stage.error = str(e)
    stage.duration_ms = (time.perf_counter() - t1) * 1000

    # --- Finalize ---
    report.total_time_ms = (time.perf_counter() - t0) * 1000
    report.all_passed = all(s.success for s in report.stages)

    # Collect deliverables summary
    report.deliverables = {
        "firmware_files": list(firmware.keys()) if 'firmware' in dir() else [],
        "bom_cost_cny": bom["cost_summary"]["total_cost_cny"] if 'bom' in dir() else 0,
        "stages_passed": sum(1 for s in report.stages if s.success),
        "stages_total": len(report.stages),
    }

    return report


def save_report(report: DesignReport, path: str = "data/e2e_design_report.json") -> None:
    """Save report to JSON."""
    data = {
        "requirement": report.requirement,
        "all_passed": report.all_passed,
        "total_time_ms": round(report.total_time_ms, 1),
        "stages": [
            {
                "name": s.name,
                "success": s.success,
                "duration_ms": round(s.duration_ms, 1),
                "details": s.details,
                "error": s.error,
            }
            for s in report.stages
        ],
        "deliverables": report.deliverables,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(report: DesignReport) -> None:
    """Print human-readable report."""
    print(f"\n{'='*60}")
    print(f"E2E Design Report: {report.requirement}")
    print(f"{'='*60}")
    for s in report.stages:
        status = "PASS" if s.success else "FAIL"
        print(f"  [{status}] {s.name} ({s.duration_ms:.0f}ms)")
        if s.error:
            print(f"         Error: {s.error}")
        elif s.details:
            for k, v in s.details.items():
                print(f"         {k}: {v}")
    print(f"{'='*60}")
    print(f"Result: {'ALL PASSED' if report.all_passed else 'SOME FAILED'}")
    print(f"Total time: {report.total_time_ms:.0f}ms")
    print(f"Deliverables: {report.deliverables.get('stages_passed', '?')}/{report.deliverables.get('stages_total', '?')} stages")
    print(f"{'='*60}\n")


# ============================================================================
# Pytest tests
# ============================================================================

class TestE2EDesign:
    """End-to-end design pipeline tests."""

    @pytest.fixture(scope="class")
    def report(self):
        r = run_full_pipeline()
        save_report(r)
        print_report(r)
        return r

    def test_all_stages_passed(self, report):
        assert report.all_passed, f"Failed stages: {[s.name for s in report.stages if not s.success]}"

    def test_has_11_stages(self, report):
        assert len(report.stages) == 11

    def test_assembly_definition(self, report):
        s = report.stages[0]
        assert s.success
        assert s.details["parts"] >= 8
        assert s.details["revolute_joints"] >= 4

    def test_assembly_solving(self, report):
        s = report.stages[1]
        assert s.success
        assert s.details["parts_positioned"] >= 8

    def test_ik_verification(self, report):
        s = report.stages[2]
        assert s.success
        assert s.details["targets_tested"] >= 4

    def test_actuator_selection(self, report):
        s = report.stages[3]
        assert s.success
        assert s.details["total_power_w"] > 0

    def test_firmware_generation(self, report):
        s = report.stages[4]
        assert s.success
        assert s.details["files_generated"] >= 7

    def test_wiring_diagram(self, report):
        s = report.stages[5]
        assert s.success

    def test_test_sequence(self, report):
        s = report.stages[6]
        assert s.success

    def test_bom_generation(self, report):
        s = report.stages[7]
        assert s.success
        assert s.details["total_cost_cny"] > 0

    def test_assembly_guide(self, report):
        s = report.stages[8]
        assert s.success

    def test_print_optimization(self, report):
        s = report.stages[9]
        assert s.success
        assert s.details["plates_needed"] >= 1

    def test_quality_control(self, report):
        s = report.stages[10]
        assert s.success
        assert s.details["dimensions_to_inspect"] > 0
        assert s.details["test_phases"] >= 4

    def test_report_saved(self, report):
        p = Path("data/e2e_design_report.json")
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["requirement"] == report.requirement
        assert len(data["stages"]) == 11

    def test_total_time_reasonable(self, report):
        # Should complete in under 30 seconds
        assert report.total_time_ms < 30000

    def test_deliverables(self, report):
        d = report.deliverables
        assert d["stages_passed"] == d["stages_total"]
        assert len(d["firmware_files"]) >= 7
        assert d["bom_cost_cny"] > 0


if __name__ == "__main__":
    report = run_full_pipeline()
    print_report(report)
    save_report(report)
    print(f"Report saved to data/e2e_design_report.json")
