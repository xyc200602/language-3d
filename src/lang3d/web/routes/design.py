"""Design analysis routes: hierarchy, assembly tree, stability, power.

Extracted from web/app.py (P1-1 God Module split, AGENTS.md §2.1).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared paths from app.py
# Helpers resolved dynamically from app.py each call (keeps @patch live
# in tests; avoids the caching bug that broke test_async_convert).
def _get_data_root():
    from ..app import DATA_ROOT
    return DATA_ROOT


# Module-level cache for design calculations (avoids repeated solve).
# MOVED here from slicing.py on 2026-07-03: this helper + cache were
# stranded in slicing.py during the P1-1 God-Module split, but ONLY
# design.py calls _get_default_robot_data (3 call sites). The mis-
# placement caused a NameError on every /api/design/* endpoint
# (test_web_complex_robot, 8 failures). Slicing.py never used it.
_design_cache: dict[str, Any] = {}


def _get_default_robot_data() -> dict[str, Any]:
    """Build and cache the default complex_robot assembly data."""
    if "robot" in _design_cache:
        return _design_cache["robot"]

    from ...tools.export_package import build_complex_robot, _build_subsystems
    from ...tools.assembly_solver import AssemblySolver
    from ...knowledge.mechanics import compute_assembly_mass

    assembly = build_complex_robot()
    solver = AssemblySolver(assembly)
    positions = solver.solve()
    mass_result = compute_assembly_mass(assembly)
    subsystems = _build_subsystems(assembly, positions)

    data = {
        "assembly": assembly,
        "positions": positions,
        "mass_result": mass_result,
        "subsystems": subsystems,
    }
    _design_cache["robot"] = data
    return data


@router.get("/api/design/hierarchy")
async def api_design_hierarchy() -> JSONResponse:
    """Return subsystem decomposition for the complex robot design.

    NOTE: uses a demo robot configuration (not the active run's assembly).
    The hierarchy/tree/stability/power endpoints are design-exploration aids
    with representative parameters, not computed from a specific generated
    assembly."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]
        subsystems = data["subsystems"]

        ss_list = []
        for name, parts in subsystems.items():
            ss_list.append({
                "name": name,
                "parts": parts,
                "part_count": len(parts),
                "status": "completed",
            })

        return JSONResponse({
            "assembly_name": assembly.name,
            "total_parts": len(assembly.parts),
            "total_joints": len(assembly.joints),
            "subsystems": ss_list,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/design/assembly-tree")
async def api_design_assembly_tree() -> JSONResponse:
    """Return the assembly hierarchy tree (parent -> children) from joints."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]

        # Build adjacency: parent -> list of (child, joint_type)
        children_map: dict[str, list[dict[str, Any]]] = {}
        all_children: set[str] = set()
        for j in assembly.joints:
            children_map.setdefault(j.parent, []).append({
                "name": j.child,
                "joint_type": j.type,
            })
            all_children.add(j.child)

        # Root = part that is never a child
        roots = [p.name for p in assembly.parts if p.name not in all_children]
        root_name = roots[0] if roots else assembly.parts[0].name

        def _build_tree(name: str) -> dict[str, Any]:
            node: dict[str, Any] = {"name": name}
            kids = children_map.get(name, [])
            if kids:
                node["children"] = [
                    {**_build_tree(k["name"]), "joint_type": k["joint_type"]}
                    for k in kids
                ]
            return node

        tree = _build_tree(root_name)
        return JSONResponse({"tree": tree})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/design/stability")
async def api_design_stability() -> JSONResponse:
    """Return stability analysis for the complex robot design."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]
        positions = data["positions"]
        mass_result = data["mass_result"]

        from ...tools.stability import (
            compute_support_polygon,
            compute_static_stability,
            check_tip_over_risk,
        )

        contacts = []
        for s in ["fl", "fr", "rl", "rr"]:
            pos = positions.get(f"wheel_{s}", {}).get("position", [0, 0, 0])
            contacts.append([pos[0], pos[1], pos[2]])

        com = list(mass_result["center_of_mass_mm"])
        polygon = compute_support_polygon(contacts)
        poly_2d = [[p[0], p[1]] for p in (polygon if len(polygon) >= 3 else contacts)]
        static_stab = compute_static_stability(com, poly_2d)
        tip_risk = check_tip_over_risk(
            com=com, contact_points=contacts,
            mass_kg=mass_result["total_mass_kg"],
        )

        return JSONResponse({
            "total_mass_kg": round(mass_result["total_mass_kg"], 3),
            "center_of_mass_mm": com,
            "support_polygon": {
                "vertices": len(polygon),
                "area_mm2": round(static_stab.get("polygon_area_mm2", 0), 1),
            },
            "static_stability": {
                "stable": static_stab.get("stable", False),
                "margin_mm": round(static_stab.get("margin_mm", 0), 1),
            },
            "tip_over_risk": {
                "risk_level": tip_risk.get("risk_level", "unknown"),
                "min_stability_margin_mm": round(
                    tip_risk.get("min_stability_margin_mm", 0), 1
                ),
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/design/power-budget")
async def api_design_power_budget() -> JSONResponse:
    """Return power budget analysis for the complex robot design."""
    try:
        from ...tools.power_budget import PowerBudgetCalculator

        calc = PowerBudgetCalculator("MobileRobotDualArm")
        calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=4)
        calc.add_servo("Arm Servos", "MG996R", duty_cycle=0.3, quantity=6)
        calc.add_controller("IPC", tdp_w=15.0)
        calc.add_sensor_load("Sensors", power_w=2.0, quantity=3)

        consumers = [
            {
                "name": c.name,
                "category": c.category,
                "peak_power_w": c.peak_power_w,
                "avg_power_w": c.avg_power_w,
                "duty_cycle": c.duty_cycle,
                "quantity": c.quantity,
            }
            for c in calc.consumers
        ]

        batt_recs = calc.recommend_battery(runtime_target_h=0.5)
        battery_recommendations = [
            {
                "name": r.get("battery").name if hasattr(r.get("battery"), "name") else str(r.get("battery")),
                "runtime_h": r.get("runtime_h"),
                "margin_pct": r.get("margin_pct"),
            }
            for r in batt_recs[:3]
        ]

        return JSONResponse({
            "consumers": consumers,
            "peak_power_w": round(calc.compute_total_peak(), 1),
            "avg_power_w": round(calc.compute_total_avg(), 1),
            "battery_recommendations": battery_recommendations,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
