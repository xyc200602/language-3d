"""Iterative editing session — Claude-Code-style assembly modification.

The session loads an existing run folder (containing ``assembly.json`` and
optionally ``engineering_package/``), accepts free-text edit requests,
applies targeted modifications via :mod:`lang3d.agent.modifier`, runs
verification (solver + VTK render + collision + optional MuJoCo), and
saves back to the same folder — *in-place* by default.

Designed to be driven from the CLI ``/iter`` command, but the
``IterativeSession`` class is also reusable from scripts or the web
backend.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent.modifier import (
    ModificationRequest,
    apply_modification,
    classify_modification,
    modifications_diff,
)
from .knowledge.mechanics import Assembly, Joint, Part

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assembly JSON (de)serialisation — kept here so the interactive path does
# not need to import the LLM-heavy assembly_generator module.
# ---------------------------------------------------------------------------


def assembly_to_dict(assembly: Assembly) -> dict[str, Any]:
    """Serialise an Assembly to a plain dict (matches generator output)."""
    return {
        "name": assembly.name,
        "description": assembly.description,
        "default_angles": dict(assembly.default_angles),
        "parts": [
            {
                "name": p.name,
                "category": p.category,
                "description": p.description,
                "material": p.material,
                "dimensions": dict(p.dimensions),
                "notes": p.notes,
            }
            for p in assembly.parts
        ],
        "joints": [
            {
                "type": j.type,
                "parent": j.parent,
                "child": j.child,
                "range_deg": list(j.range_deg),
                "description": j.description,
                "parent_anchor": j.parent_anchor,
                "child_anchor": j.child_anchor,
                "offset": list(j.offset or (0.0, 0.0, 0.0)),
                "axis": j.axis,
                "no_distribute": j.no_distribute,
                "distribution_group": j.distribution_group,
                "connection_method": _safe_connection_str(j),
                "mimic_joint": getattr(j, "mimic_joint", ""),
            }
            for j in assembly.joints
        ],
    }


def _safe_connection_str(j: Joint) -> str:
    """Best-effort string for the joint's physical connection method."""
    conn = getattr(j, "connection", None)
    if conn is None:
        return ""
    # ConnectionMethod can be a dataclass or enum
    name = getattr(conn, "name", None) or getattr(conn, "method", None)
    if name:
        return str(name)
    return str(conn)


def assembly_from_dict(data: dict[str, Any]) -> Assembly:
    """Reverse of :func:`assembly_to_dict`."""
    parts = [
        Part(
            name=p["name"],
            category=p.get("category", ""),
            description=p.get("description", ""),
            material=p.get("material", "PLA"),
            dimensions=dict(p.get("dimensions", {})),
            notes=p.get("notes", ""),
        )
        for p in data.get("parts", [])
    ]
    joints = []
    for j in data.get("joints", []):
        kwargs: dict[str, Any] = {
            "type": j.get("type", "fixed"),
            "parent": j["parent"],
            "child": j["child"],
            "description": j.get("description", ""),
            "parent_anchor": j.get("parent_anchor", "top"),
            "child_anchor": j.get("child_anchor", "bottom"),
            "offset": tuple(j.get("offset", [0.0, 0.0, 0.0])),
            "axis": j.get("axis", "auto"),
            "no_distribute": j.get("no_distribute", False),
            "distribution_group": j.get("distribution_group", ""),
        }
        if "range_deg" in j:
            kwargs["range_deg"] = tuple(j["range_deg"])
        if j.get("mimic_joint"):
            kwargs["mimic_joint"] = j["mimic_joint"]
        joints.append(Joint(**kwargs))
    return Assembly(
        name=data.get("name", "assembly"),
        parts=parts,
        joints=joints,
        description=data.get("description", ""),
        default_angles=dict(data.get("default_angles", {})),
    )


# ---------------------------------------------------------------------------
# Session history entry
# ---------------------------------------------------------------------------


@dataclass
class HistoryEntry:
    """A single in-place edit."""

    request_text: str
    request: ModificationRequest
    diff: dict[str, Any]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    # Snapshot of assembly BEFORE the edit, for /undo
    assembly_before: Assembly | None = None


# ---------------------------------------------------------------------------
# IterativeSession
# ---------------------------------------------------------------------------


class IterativeSession:
    """Load → edit → verify → save loop for an existing run folder.

    Folder layout (canonical, post Phase 2)::

        <folder>/
        ├── assembly.json
        ├── engineering_package/
        │   ├── stl_parts/
        │   ├── urdf.xml
        │   └── ...
        └── vlm_renders/round_N/  (created/extended on each /render)

    Saving is *in-place* by default: ``assembly.json`` is overwritten and
    only the modified STLs are regenerated.  Each call to :meth:`apply`
    pushes a HistoryEntry so :meth:`undo` can revert.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Run folder not found: {self.folder}")
        assembly_path = self.folder / "assembly.json"
        if not assembly_path.exists():
            raise FileNotFoundError(f"assembly.json not found in {self.folder}")
        self.assembly_path = assembly_path
        self.history: list[HistoryEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        data = json.loads(self.assembly_path.read_text(encoding="utf-8"))
        self.assembly = assembly_from_dict(data)
        logger.info(
            "Loaded assembly %r (%d parts, %d joints) from %s",
            self.assembly.name, len(self.assembly.parts),
            len(self.assembly.joints), self.folder,
        )

    def save(self, folder: str | Path | None = None) -> Path:
        """Save assembly.json (and regenerate changed STLs).

        When *folder* is None (default), saves in-place.  Otherwise saves
        to the given folder (used by ``/save-as``).

        After writing assembly.json, regenerates STLs for parts whose
        dimensions changed (so the 3D viewer reflects edits immediately).
        Uses FreeCAD when available; falls back to trimesh box previews.
        """
        target = Path(folder).resolve() if folder else self.folder
        target.mkdir(parents=True, exist_ok=True)
        out = target / "assembly.json"
        out.write_text(
            json.dumps(assembly_to_dict(self.assembly), indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )

        # Regenerate STLs for changed parts so the 3D viewer updates.
        self._regen_changed_stls(target)

        if folder:
            # /save-as: copy the rest of the engineering_package
            src_pkg = self.folder / "engineering_package"
            if src_pkg.is_dir():
                dst_pkg = target / "engineering_package"
                if dst_pkg.is_dir():
                    shutil.rmtree(dst_pkg)
                shutil.copytree(src_pkg, dst_pkg)
            self.folder = target
            self.assembly_path = out
        return out

    def _regen_changed_stls(self, target: Path) -> None:
        """Regenerate STLs for parts whose dimensions changed.

        Compares the current assembly against the last saved version (from
        history). For each changed part, regenerates its STL using FreeCAD
        (if available) or a trimesh box preview (fallback). This makes the
        3D viewer reflect edits immediately — previously only assembly.json
        was updated and the viewer showed stale geometry.
        """
        if not self.history:
            return  # no edits — nothing changed

        last = self.history[-1]
        before_dims: dict[str, dict] = {}
        for p in last.assembly_before.parts:
            before_dims[p.name] = dict(p.dimensions)

        changed = [
            p for p in self.assembly.parts
            if p.name in before_dims and before_dims[p.name] != dict(p.dimensions)
        ]
        if not changed:
            return  # no dimension changes

        stl_dir = target / "engineering_package" / "stl_parts"
        if not stl_dir.is_dir():
            stl_dir = target / "stl_parts"
        stl_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ..tools.export_package import generate_part_stls
            from ..knowledge.mechanics import Assembly as Asm
            # Build a mini-assembly with just the changed parts.
            mini = Asm(name="regen", parts=changed, joints=[],
                        description="partial regen")
            generate_part_stls(mini, str(stl_dir), timeout=30)
        except Exception:
            # FreeCAD not available — fall back to trimesh box previews.
            try:
                import trimesh
                for p in changed:
                    d = p.dimensions
                    l = d.get("length", d.get("diameter", 20))
                    w = d.get("width", d.get("diameter", l))
                    h = d.get("height", d.get("thickness", 20))
                    mesh = trimesh.creation.box(extents=[l, w, h])
                    mesh.export(str(stl_dir / f"{p.name}.stl"))
            except ImportError:
                pass  # no trimesh either — skip STL regen silently

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def apply(self, request_text: str) -> dict[str, Any]:
        """Classify + apply the edit request.  Returns a UI-friendly dict.

        Does NOT auto-save; caller should call :meth:`save` afterwards
        (the CLI does this automatically on each /iter edit).
        """
        req = classify_modification(request_text, self.assembly)
        before = copy.deepcopy(self.assembly)
        new_asm = apply_modification(
            self.assembly, req,
            description=self.assembly.description or request_text,
        )
        diff = modifications_diff(before, new_asm)

        self.history.append(HistoryEntry(
            request_text=request_text,
            request=req,
            diff=diff,
            assembly_before=before,
        ))
        self.assembly = new_asm
        return {
            "scope": req.scope,
            "intent": req.intent,
            "target": req.target,
            "params": req.params,
            "diff": diff,
            "applied": (
                bool(diff["parts_changed"])
                or bool(diff["joints_changed"])
                or bool(diff["parts_added"])
                or bool(diff["parts_removed"])
                or req.scope == "whole"  # whole-regen always "applies"
            ),
        }

    def undo(self) -> bool:
        """Revert the last edit.  Returns True if there was something to undo."""
        if not self.history:
            return False
        entry = self.history.pop()
        if entry.assembly_before is not None:
            self.assembly = entry.assembly_before
        return True

    # ------------------------------------------------------------------
    # Verify (optional, called on demand or after each edit)
    # ------------------------------------------------------------------

    def verify(self, *, with_collision: bool = True) -> dict[str, Any]:
        """Run solver + collision check.  Returns a structured report.

        This is a lightweight, LLM-free verification — full VLM verify
        still requires the GLM backend and is dispatched through the
        main assembly_generator pipeline.
        """
        report: dict[str, Any] = {"ok": True, "checks": []}

        # Solver
        try:
            from .tools.pipeline_context import AssemblyContext
            ctx = AssemblyContext(assembly=self.assembly)
            positions = ctx.ensure_positions()
            n_pos = len(positions)
            n_nan = sum(
                1 for p in positions.values()
                if any(_is_nan(c) for c in p.get("position", [0, 0, 0]))
            )
            report["checks"].append({
                "name": "solver",
                "ok": n_nan == 0,
                "detail": f"{n_pos} positions, {n_nan} NaN",
            })
            if n_nan:
                report["ok"] = False
        except Exception as e:
            report["checks"].append({"name": "solver", "ok": False,
                                     "detail": f"error: {e}"})
            report["ok"] = False
            positions = {}

        # Collision (trimesh + python-fcl)
        if with_collision and positions:
            try:
                from .tools.mesh_collision import MeshCollisionChecker
                checker = MeshCollisionChecker()
                result = checker.check_assembly_collisions(
                    self.assembly, positions, skip_adjacent=True,
                )
                severe = [p for p in result.pairs
                          if p.is_collision and p.penetration_depth_mm > 1.0]
                report["checks"].append({
                    "name": "collisions",
                    "ok": len(severe) == 0,
                    "detail": f"{len(severe)} severe pairs of "
                              f"{result.pairs_checked} checked",
                })
                if severe:
                    report["ok"] = False
            except ImportError:
                report["checks"].append({
                    "name": "collisions", "ok": True,
                    "detail": "skipped (trimesh/fcl not installed)",
                })
            except Exception as e:
                report["checks"].append({
                    "name": "collisions", "ok": True,
                    "detail": f"skipped ({e})",
                })

        return report

    # ------------------------------------------------------------------
    # Rendering (delegates to VTK renderer if available)
    # ------------------------------------------------------------------

    def render(self) -> Path | None:
        """Render current assembly to PNGs in ``vlm_renders/manual_<ts>/``."""
        try:
            from .tools.pipeline_context import AssemblyContext
            from .tools.vtk_renderer import render_assembly_from_positions
        except ImportError as e:
            logger.warning("Renderer unavailable: %s", e)
            return None

        ctx = AssemblyContext(assembly=self.assembly)
        positions = ctx.ensure_positions()
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.folder / "vlm_renders" / f"manual_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        parts_dicts = [
            {"name": p.name, "category": p.category, "dimensions": p.dimensions}
            for p in self.assembly.parts
        ]
        try:
            render_assembly_from_positions(
                parts=parts_dicts,
                positions=positions,
                output_dir=str(out_dir),
                joints=list(self.assembly.joints),
                gripper_closeup=True,
            )
            return out_dir
        except Exception as e:
            logger.warning("Render failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"Assembly: {self.assembly.name}",
            f"  Parts:   {len(self.assembly.parts)}",
            f"  Joints:  {len(self.assembly.joints)}",
            f"  Folder:  {self.folder}",
        ]
        if self.history:
            lines.append(f"  History: {len(self.history)} edits")
            for i, entry in enumerate(self.history[-3:], 1):
                lines.append(
                    f"    [{i}] {entry.request.scope}/{entry.request.intent}"
                    f" -> {entry.request_text[:50]}"
                )
        return "\n".join(lines)


def _is_nan(v: Any) -> bool:
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Convenience entry point for the CLI
# ---------------------------------------------------------------------------


def start_iter_session(folder: str) -> IterativeSession:
    """Factory used by the CLI's ``/iter`` command."""
    return IterativeSession(folder)
