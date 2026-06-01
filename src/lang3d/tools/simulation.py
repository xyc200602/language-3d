"""Simulation tools: FEA structural analysis, VLM stress interpretation,
interference checking, tolerance analysis, and motion simulation.

Tools:
  fea_run            - Run FEA structural analysis (mesh + boundary + CalculiX solve)
  fea_visualize      - Open FreeCAD GUI to display stress/displacement contour
  fea_vlm_analyze    - Screenshot + VLM interpret stress distribution
  interference_check - Boolean intersection check for part interference
  tolerance_analysis - Monte Carlo tolerance stack-up analysis
  motion_sim         - Motion simulation (forward kinematics, range check, trajectory)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..knowledge.materials import SAFETY_FACTORS, get_material
from ..knowledge.simulation import FEA_PATTERNS, MESH_SIZES, recommend_mesh_size
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# CalculiX .frd result file parser
# ---------------------------------------------------------------------------

def parse_frd(frd_path: str) -> dict[str, Any]:
    """Parse a CalculiX .frd result file and extract structured results.

    Returns dict with:
        max_displacement: float (mm)
        max_von_mises_stress: float (MPa)
        node_count: int
        displacement_components: dict (max_dx, max_dy, max_dz)
        stress_components: dict (max_sxx, max_syy, max_szz, max_sxy, max_syz, max_szx)
    """
    result: dict[str, Any] = {
        "max_displacement": 0.0,
        "max_von_mises_stress": 0.0,
        "node_count": 0,
        "displacement_components": {},
        "stress_components": {},
    }

    if not Path(frd_path).exists():
        return result

    with open(frd_path) as f:
        lines = f.readlines()

    # State machine for parsing
    section = None  # "disp" or "stress"
    # Accumulate all displacement vectors and stress tensors per node
    node_disps: dict[int, list[float]] = {}  # node_id -> [dx, dy, dz]
    node_stresses: dict[int, list[float]] = {}  # node_id -> [sxx, syy, szz, sxy, syz, szx]

    for line in lines:
        s = line.rstrip()

        # Result block header: "  100CL ..."
        if s.lstrip().startswith("100CL"):
            section = None
            continue

        # Block type header: "-4 DISP ..." or "-4 STRESS ..."
        if s.lstrip().startswith("-4"):
            parts = s.split()
            if len(parts) >= 2:
                if "DISP" in parts[1].upper():
                    section = "disp"
                elif "STRESS" in parts[1].upper():
                    section = "stress"
                else:
                    section = None
            continue

        # Component descriptor: "-5 ..."
        if s.lstrip().startswith("-5"):
            continue

        # Block end: "-3"
        if s.lstrip() == "-3":
            section = None
            continue

        # Data line: "-1  node_id  values..."
        if s.lstrip().startswith("-1") and section:
            # Parse the fixed-width format: node_id + values
            # Format: " -1    <id> <v1> <v2> ..."
            # Values may be negative and attached to previous value without space
            # e.g. " -1    3 5.20E-09-5.12E-09 1.18E-08"
            # Split on whitespace won't work for attached negatives.
            # Use regex to split the data part.
            data_part = s[5:]  # skip " -1  "
            node_id_str = ""
            rest = data_part.lstrip()
            # Extract node_id (integer)
            i = 0
            while i < len(rest) and rest[i].isdigit():
                i += 1
            if i == 0:
                continue
            node_id = int(rest[:i])
            rest = rest[i:]

            # Parse scientific notation values (13-char fields with possible attached sign)
            values = []
            # Each value is in Fortran scientific notation: ±d.dddddE±dd
            # They can be 13 chars wide. But sometimes the sign is attached.
            # Use regex to extract all float values
            val_pattern = re.compile(r'[+-]?\d+\.\d+E[+-]\d+')
            values = [float(v) for v in val_pattern.findall(rest)]

            if section == "disp" and len(values) >= 3:
                node_disps[node_id] = values[:3]

            elif section == "stress" and len(values) >= 6:
                node_stresses[node_id] = values[:6]

    # Compute results
    if node_disps:
        max_disp = 0.0
        max_dx = max_dy = max_dz = 0.0
        for nid, vals in node_disps.items():
            dx, dy, dz = vals[0], vals[1], vals[2]
            mag = (dx**2 + dy**2 + dz**2) ** 0.5
            if mag > max_disp:
                max_disp = mag
            if abs(dx) > abs(max_dx):
                max_dx = dx
            if abs(dy) > abs(max_dy):
                max_dy = dy
            if abs(dz) > abs(max_dz):
                max_dz = dz
        result["max_displacement"] = max_disp
        result["node_count"] = len(node_disps)
        result["displacement_components"] = {
            "max_dx": max_dx,
            "max_dy": max_dy,
            "max_dz": max_dz,
        }

    if node_stresses:
        max_von_mises = 0.0
        max_sxx = max_syy = max_szz = 0.0
        max_sxy = max_syz = max_szx = 0.0
        for nid, vals in node_stresses.items():
            sxx, syy, szz, sxy, syz, szx = vals
            # von Mises stress: sqrt(0.5*((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2 + 6*(sxy^2+syz^2+szx^2)))
            vm = math.sqrt(
                0.5 * (
                    (sxx - syy)**2
                    + (syy - szz)**2
                    + (szz - sxx)**2
                    + 6 * (sxy**2 + syz**2 + szx**2)
                )
            )
            if vm > max_von_mises:
                max_von_mises = vm
            if abs(sxx) > abs(max_sxx):
                max_sxx = sxx
            if abs(syy) > abs(max_syy):
                max_syy = syy
            if abs(szz) > abs(max_szz):
                max_szz = szz
            if abs(sxy) > abs(max_sxy):
                max_sxy = sxy
            if abs(syz) > abs(max_syz):
                max_syz = syz
            if abs(szx) > abs(max_szx):
                max_szx = szx
        result["max_von_mises_stress"] = max_von_mises
        result["stress_components"] = {
            "max_sxx": max_sxx,
            "max_syy": max_syy,
            "max_szz": max_szz,
            "max_sxy": max_sxy,
            "max_syz": max_syz,
            "max_szx": max_szx,
        }
        if not result["node_count"]:
            result["node_count"] = len(node_stresses)

    return result


# ---------------------------------------------------------------------------
# CalculiX solver finder
# ---------------------------------------------------------------------------

def _find_calculix() -> str | None:
    """Locate the CalculiX (ccx) solver executable.

    Priority:
    1. CALCULIX_PATH environment variable
    2. FreeCAD bin directory (ccx.exe bundled with FreeCAD FEM workbench)
    3. System PATH
    """
    # 1. Environment variable
    env_path = os.environ.get("CALCULIX_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Look in FreeCAD bin directory
    common_fc_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_fc_paths:
        ccx = str(Path(p) / "ccx.exe")
        if Path(ccx).exists():
            return ccx

    # 3. System PATH
    import shutil
    found = shutil.which("ccx") or shutil.which("ccx.exe")
    if found:
        return found

    return None


# ---------------------------------------------------------------------------
# FreeCAD subprocess bridge (reuse freecad.py patterns)
# ---------------------------------------------------------------------------

def _find_freecad_python() -> str | None:
    """Find FreeCAD's bundled Python executable."""
    fc_path = os.environ.get("FREECAD_PATH")
    if fc_path:
        python = str(Path(fc_path) / "python.exe")
        if Path(python).exists():
            return python

    common_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_paths:
        python = str(Path(p) / "python.exe")
        if Path(python).exists():
            return python

    return None


def _run_freecad_script(script: str, timeout: int = 120) -> str:
    """Execute a Python script using FreeCAD's bundled Python."""
    fc_python = _find_freecad_python()
    if not fc_python:
        raise RuntimeError(
            "FreeCAD not found. Install with: winget install FreeCAD\n"
            "Or set FREECAD_PATH to FreeCAD's bin directory."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [fc_python, "-c", f"exec(open(r'{script_path}').read())"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FreeCAD script error:\n{result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("FreeCAD script timed out")
    finally:
        Path(script_path).unlink(missing_ok=True)


def _find_freecad_exe() -> str | None:
    """Find FreeCAD executable (GUI)."""
    fc_path = os.environ.get("FREECAD_PATH")
    if fc_path:
        exe = str(Path(fc_path) / "FreeCAD.exe")
        if Path(exe).exists():
            return exe

    common_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_paths:
        exe = str(Path(p) / "FreeCAD.exe")
        if Path(exe).exists():
            return exe
    return None


# ---------------------------------------------------------------------------
# FEA script builder
# ---------------------------------------------------------------------------

def _build_fea_script(
    document_path: str,
    material_name: str,
    fixed_face: str,
    force_face: str,
    force_magnitude: float,
    force_direction: list[float],
    mesh_size: str,
    analysis_name: str,
) -> str:
    """Build a FreeCAD FEM Python script for structural analysis.

    Compatible with FreeCAD 1.1 API.
    Face selection uses normal-vector matching:
      bottom=-Z, top=+Z, front=+Y, back=-Y, left=-X, right=+X
    """
    mesh_info = MESH_SIZES.get(mesh_size, MESH_SIZES["medium"])
    max_size_factor = mesh_info["max_element_size_factor"]

    return f'''
import FreeCAD
import Fem
import ObjectsFem
import os
import json
import tempfile

# Open document
doc = FreeCAD.openDocument(r"{document_path}")
if not doc:
    raise RuntimeError(f"Failed to open document: {{document_path}}")

# Find the solid shape (first Part::Feature)
solid_obj = None
for obj in doc.Objects:
    if hasattr(obj, "Shape") and obj.Shape.Solids:
        solid_obj = obj
        break
if not solid_obj:
    raise RuntimeError("No solid object found in document")

# Bounding box for mesh sizing
bb = solid_obj.Shape.BoundBox
max_dim = max(bb.XLength, bb.YLength, bb.ZLength)
max_element_size = max_dim * {max_size_factor}

# --- Face selection by normal vector ---
def _find_face_name_by_normal(shape, target_dir, tolerance=0.9):
    """Find face whose normal best matches target_dir, return 'FaceN' string."""
    target = FreeCAD.Vector(*target_dir)
    best_idx = 0
    best_dot = -2.0
    for i, face in enumerate(shape.Faces):
        try:
            normal = face.normalAt(0, 0)
            dot = normal.dot(target)
            if dot > best_dot:
                best_dot = dot
                best_idx = i
        except Exception:
            continue
    if best_dot >= tolerance:
        return f"Face{{best_idx + 1}}"
    return "Face1" if shape.Faces else "Face1"

FACE_MAP = {{
    "bottom": (0, 0, -1),
    "top": (0, 0, 1),
    "front": (0, 1, 0),
    "back": (0, -1, 0),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
}}

fixed_dir = FACE_MAP.get("{fixed_face}", (0, 0, -1))
force_dir = FACE_MAP.get("{force_face}", (0, 0, 1))

fixed_face_name = _find_face_name_by_normal(solid_obj.Shape, fixed_dir)
force_face_name = _find_face_name_by_normal(solid_obj.Shape, force_dir)
print(f"Fixed face: {{fixed_face_name}}, Force face: {{force_face_name}}")

# --- Create FEM analysis ---
analysis = ObjectsFem.makeAnalysis(doc, "{analysis_name}")

# Material
material = ObjectsFem.makeMaterialSolid(doc, "FEMMaterial")
mat_dict = {{
    "Name": "{material_name}",
    "YoungsModulus": "200000 MPa",
    "PoissonRatio": "0.3",
    "Density": "7850 kg/m^3",
    "YieldStrength": "350 MPa",
}}
# Use actual material data if available
try:
    import sys
    sys.path.insert(0, r"{str(Path(__file__).parent.parent)}")
    from knowledge.materials import get_material
    mat = get_material("{material_name}")
    if mat:
        mat_dict["Name"] = mat.name
        mat_dict["YoungsModulus"] = f"{{mat.youngs_modulus}} MPa"
        mat_dict["PoissonRatio"] = f"{{mat.poissons_ratio}}"
        mat_dict["Density"] = f"{{mat.density}} kg/m^3"
        mat_dict["YieldStrength"] = f"{{mat.yield_strength}} MPa"
except Exception:
    pass

material.Material = mat_dict
analysis.addObject(material)

# Mesh (Gmsh)
fem_mesh = ObjectsFem.makeMeshGmsh(doc, "FEMMesh")
fem_mesh.Shape = solid_obj
fem_mesh.CharacteristicLengthMax = max_element_size
fem_mesh.ElementDimension = "3D"
analysis.addObject(fem_mesh)

# Fixed constraint
fixed = ObjectsFem.makeConstraintFixed(doc, "FEMFixed")
fixed.References = [(solid_obj, (fixed_face_name,))]
analysis.addObject(fixed)

# Force constraint (applied normal to face)
force = ObjectsFem.makeConstraintForce(doc, "FEMForce")
force.References = [(solid_obj, (force_face_name,))]
force.Force = {force_magnitude}
# Reverse force if direction opposes face normal
_fd = {force_direction}
_force_normal = FreeCAD.Vector(*force_dir)
_face_normal = solid_obj.Shape.Faces[int(force_face_name.replace("Face", "")) - 1].normalAt(0, 0)
if _force_normal.dot(_face_normal) < 0:
    try:
        force.Reversed = True
    except Exception:
        pass
analysis.addObject(force)

# Solver (CalculiX)
solver = ObjectsFem.makeSolverCalculiXCcxTools(doc, "FEMSolver")
try:
    solver.AnalysisType = "static"
    solver.GeometricalNonlinearity = "linear"
    solver.ThermoMechSteadyState = False
except AttributeError:
    pass
analysis.addObject(solver)

doc.recompute()

# --- Run Gmsh meshing ---
print("Running Gmsh meshing...")
try:
    from femmesh import gmshtools
    gmsh = gmshtools.GmshTools(fem_mesh)
    gmsh.create_mesh()
    print(f"Mesh generated: {{fem_mesh.FemMesh.NodeCount}} nodes")
except Exception as e:
    print(f"Mesh error: {{e}}")
    raise

doc.recompute()

# --- Run CalculiX solver ---
print("Starting CalculiX solver...")
solver_working_dir = tempfile.mkdtemp(prefix="lang3d_fem_")
try:
    from femtools import ccxtools
    fe = ccxtools.FemToolsCcx(analysis, solver)
    fe.update_objects()
    fe.setup_working_dir(solver_working_dir)
    fe.setup_ccx()
    fe.check_prerequisites()
    fe.write_inp_file()
    fe.ccx_run()
    fe.load_results()
    print(f"Results present: {{fe.results_present}}")

    # Parse .dat results
    import os
    for f in os.listdir(solver_working_dir):
        if f.endswith(".dat"):
            with open(os.path.join(solver_working_dir, f), "r") as fh:
                for line in fh:
                    line_s = line.strip()
                    if line_s:
                        print(f"CCX_DAT: {{line_s}}")
            break

    # Get result object if available
    if fe.result_object:
        ro = fe.result_object
        print(f"RESULT_TYPE: {{ro.TypeId}}")

except Exception as e:
    print(f"Solver error: {{e}}")

# Save analysis document
analysis_path = r"{document_path}".replace(".FCStd", "_fea.FCStd")
doc.saveAs(analysis_path)
print(f"FEA document saved: {{analysis_path}}")
print(f"SOLID_OBJ: {{solid_obj.Name}}")
print(f"WORKING_DIR: {{solver_working_dir}}")
print("FEA_COMPLETE")
'''


# ---------------------------------------------------------------------------
# VLM JSON parser for FEA stress analysis
# ---------------------------------------------------------------------------

def _parse_fea_vlm_json(raw: str) -> dict[str, Any]:
    """Parse structured FEA VLM analysis result from VLM output.

    Expected fields: safe, max_stress_region, stress_distribution,
                     safety_concern, suggestion, fix_commands
    """
    import json as _json

    # Try extracting JSON from response
    json_match = re.search(
        r'\{[^{}]*"safe"[^{}]*\}',
        raw,
        re.DOTALL,
    )
    if json_match:
        try:
            data = _json.loads(json_match.group())
            return {
                "safe": bool(data.get("safe", False)),
                "max_stress_region": str(data.get("max_stress_region", "")),
                "stress_distribution": str(data.get("stress_distribution", "")),
                "safety_concern": str(data.get("safety_concern", "None")),
                "suggestion": str(data.get("suggestion", "None")),
                "fix_commands": str(data.get("fix_commands", "None")),
            }
        except (_json.JSONDecodeError, ValueError):
            pass

    # Fallback: field-by-field extraction
    def _extract_field(name: str) -> str:
        pattern = rf'{name}[:\s]+(.*?)(?:\n|$)'
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).strip() if m else "None"

    safe_str = _extract_field("safe")
    safe_val = "yes" in safe_str.lower() or safe_str.lower() == "true"

    return {
        "safe": safe_val,
        "max_stress_region": _extract_field("max_stress_region"),
        "stress_distribution": _extract_field("stress_distribution"),
        "safety_concern": _extract_field("safety_concern"),
        "suggestion": _extract_field("suggestion"),
        "fix_commands": _extract_field("fix_commands"),
    }


# ===========================================================================
# Tool: fea_run
# ===========================================================================

class FEASetupAndRunTool(Tool):
    """Run FEA structural analysis on a FreeCAD document.

    Generates mesh, applies boundary conditions, runs CalculiX solver,
    and extracts stress/displacement results.
    """

    name = "fea_run"
    description = (
        "Run FEA structural analysis on a FreeCAD .FCStd document. "
        "Generates mesh, applies fixed constraint and force, runs CalculiX solver, "
        "returns stress/displacement results. "
        "Requires: document path, material, fixed face, force face & magnitude."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to .FCStd file to analyze",
                    },
                    "material": {
                        "type": "string",
                        "description": "Material name: steel, aluminum, pla, abs, titanium, copper (default: steel)",
                    },
                    "fixed_face": {
                        "type": "string",
                        "description": "Face to fix: bottom, top, front, back, left, right (default: bottom)",
                    },
                    "force_face": {
                        "type": "string",
                        "description": "Face to apply force: bottom, top, front, back, left, right (default: top)",
                    },
                    "force_magnitude": {
                        "type": "number",
                        "description": "Force magnitude in Newtons (default: 100)",
                    },
                    "force_direction": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Force direction vector [dx, dy, dz] (default: [0, 0, -1] downward)",
                    },
                    "mesh_size": {
                        "type": "string",
                        "enum": ["coarse", "medium", "fine", "very_fine"],
                        "description": "Mesh density (default: medium)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Solver timeout in seconds (default: 120)",
                    },
                },
                "required": ["document_path"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        material: str = "steel",
        fixed_face: str = "bottom",
        force_face: str = "top",
        force_magnitude: float = 100,
        force_direction: list[float] | None = None,
        mesh_size: str = "medium",
        timeout: int = 120,
        **kwargs: Any,
    ) -> str:
        # Validate document exists
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        if not document_path.endswith(".FCStd"):
            return f"Error: File must be a .FCStd document, got: {document_path}"

        # Validate material
        mat = get_material(material)
        material_name = material
        if mat:
            material_name = mat.name

        # Validate mesh_size
        if mesh_size not in MESH_SIZES:
            mesh_size = "medium"

        # Default force direction: downward (-Z)
        if force_direction is None:
            force_direction = [0, 0, -1]

        # Check CalculiX availability
        ccx_path = _find_calculix()
        ccx_status = f"CalculiX found: {ccx_path}" if ccx_path else "CalculiX not found (solver may fail)"

        # Check FreeCAD availability
        fc_python = _find_freecad_python()
        if not fc_python:
            return "Error: FreeCAD not found. Install with: winget install FreeCAD"

        # Build and run FEA script
        script = _build_fea_script(
            document_path=document_path,
            material_name=material_name,
            fixed_face=fixed_face,
            force_face=force_face,
            force_magnitude=force_magnitude,
            force_direction=force_direction,
            mesh_size=mesh_size,
            analysis_name="FEA_Analysis",
        )

        try:
            output = _run_freecad_script(script, timeout=timeout)

            # Parse .frd results from working directory
            frd_results = {}
            wd_match = re.search(r"WORKING_DIR:\s*(\S+)", output)
            if wd_match:
                wd = wd_match.group(1)
                wd_path = Path(wd)
                # CalculiX output file may be named Mesh.frd or FEMMesh.frd
                frd_file = wd_path / "FEMMesh.frd" if (wd_path / "FEMMesh.frd").exists() else wd_path / "Mesh.frd"
                if frd_file.exists():
                    frd_results = parse_frd(str(frd_file))

            # Compute safety factor from material yield strength
            max_stress = frd_results.get("max_von_mises_stress", 0.0)
            max_disp = frd_results.get("max_displacement", 0.0)
            safety_factor = 0.0
            safe = False
            if mat and max_stress > 0:
                safety_factor = mat.yield_strength / max_stress
                safe = safety_factor >= 1.5

            # Build structured result
            result_lines = [
                "[FEA Analysis Results]",
                f"Document: {document_path}",
                f"Material: {material_name}",
                f"Mesh: {mesh_size}",
                f"Fixed: {fixed_face} | Force: {force_face} ({force_magnitude}N)",
                f"Solver: {ccx_status}",
                "",
            ]

            if frd_results:
                result_lines += [
                    "--- Structured Results ---",
                    f"Max von Mises Stress: {max_stress:.4f} MPa",
                    f"Max Displacement: {max_disp:.6f} mm",
                    f"Node Count: {frd_results.get('node_count', 0)}",
                ]
                if mat:
                    result_lines += [
                        f"Material Yield Strength: {mat.yield_strength} MPa",
                        f"Safety Factor: {safety_factor:.2f}",
                        f"Safe (SF >= 1.5): {safe}",
                    ]
                if frd_results.get("displacement_components"):
                    dc = frd_results["displacement_components"]
                    result_lines.append(
                        f"Max Disp Components: dx={dc['max_dx']:.6f}, "
                        f"dy={dc['max_dy']:.6f}, dz={dc['max_dz']:.6f} mm"
                    )
                if frd_results.get("stress_components"):
                    sc = frd_results["stress_components"]
                    result_lines.append(
                        f"Max Stress Components: Sxx={sc['max_sxx']:.4f}, "
                        f"Syy={sc['max_syy']:.4f}, Szz={sc['max_szz']:.4f} MPa"
                    )

                # Structured JSON output
                json_result = {
                    "max_von_mises_stress_mpa": round(max_stress, 6),
                    "max_displacement_mm": round(max_disp, 8),
                    "node_count": frd_results.get("node_count", 0),
                    "material": material_name,
                    "safety_factor": round(safety_factor, 4) if safety_factor > 0 else None,
                    "safe": safe if safety_factor > 0 else None,
                }
                result_lines += [
                    "",
                    "--- JSON ---",
                    json.dumps(json_result, indent=2),
                ]
            else:
                result_lines.append("--- Solver Output (no .frd parsed) ---")

            result_lines += ["", "--- Solver Output ---", output]
            return "\n".join(result_lines)
        except RuntimeError as e:
            return f"Error running FEA: {e}"


# ===========================================================================
# Tool: fea_visualize
# ===========================================================================

class FEAVisualizeTool(Tool):
    """Open FreeCAD GUI to display FEA stress/displacement contour plot."""

    name = "fea_visualize"
    description = (
        "Open FreeCAD GUI to visualize FEA results as stress or displacement contour plot. "
        "Takes a FEA document (.FCStd with analysis) and displays the result."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to FEA .FCStd document",
                    },
                    "result_type": {
                        "type": "string",
                        "enum": ["stress", "displacement"],
                        "description": "Result type to display (default: stress)",
                    },
                    "view": {
                        "type": "string",
                        "description": "Camera view: isometric, front, top, right (default: isometric)",
                    },
                },
                "required": ["document_path"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        result_type: str = "stress",
        view: str = "isometric",
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        fc_exe = _find_freecad_exe()
        if not fc_exe:
            return "Error: FreeCAD.exe not found. Install with: winget install FreeCAD"

        view_method = {
            "isometric": "viewIsometric",
            "front": "viewFront",
            "top": "viewTop",
            "right": "viewRight",
        }.get(view, "viewIsometric")

        # Build macro to open doc, activate FEM result, set view
        macro_lines = [
            "import FreeCAD",
            "import FreeCADGui",
            "import time",
            f'doc = FreeCAD.openDocument(r"{document_path}")',
            "time.sleep(1)",
            "FreeCADGui.activeDocument().activeView()." + view_method + "()",
            # Try to show result pipeline
            "for obj in doc.Objects:",
            "    if hasattr(obj, 'Proxy') and 'Result' in obj.TypeId:",
            "        try:",
            "            FreeCADGui.activeDocument().showObject(obj)",
            "        except Exception:",
            "            pass",
            "    if hasattr(obj, 'Member') and hasattr(obj, 'Result'):",
            "        try:",
            "            obj.ViewObject.show()",
            "        except Exception:",
            "            pass",
            "print(f'Opened FEA: {doc.Name}')",
        ]

        macro_content = "\n".join(macro_lines)
        macro_dir = Path(tempfile.gettempdir()) / "lang3d_macros"
        macro_dir.mkdir(exist_ok=True)
        macro_path = macro_dir / f"fea_viz_{int(time.time())}.py"
        macro_path.write_text(macro_content, encoding="utf-8")

        subprocess.Popen([fc_exe, str(macro_path)])
        time.sleep(5)

        return (
            f"[FEA Visualization]\n"
            f"Document: {document_path}\n"
            f"Result: {result_type}\n"
            f"View: {view}\n"
            f"FreeCAD GUI launched. Use fea_vlm_analyze to capture and analyze the contour."
        )


# ===========================================================================
# Tool: fea_vlm_analyze
# ===========================================================================

class FEAVLMAnalyzeTool(Tool):
    """Capture FreeCAD FEA window screenshot and analyze with VLM.

    Follows the same pattern as CADVerifyTool: capture window + VLM analysis.
    """

    name = "fea_vlm_analyze"
    description = (
        "Capture FreeCAD FEA contour plot and analyze stress distribution with VLM. "
        "Returns structured JSON: safe, max_stress_region, safety_concern, suggestion, fix_commands."
    )

    def __init__(self, router=None, screenshot_dir: str = "") -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to FEA document (for context in prompt)",
                    },
                    "material": {
                        "type": "string",
                        "description": "Material name used in FEA (default: steel)",
                    },
                    "expected_max_stress": {
                        "type": "number",
                        "description": "Expected maximum stress in MPa (optional)",
                    },
                    "window_title": {
                        "type": "string",
                        "description": "FreeCAD window title (default: 'FreeCAD')",
                    },
                    "detail": {
                        "type": "string",
                        "description": "VLM detail level: fast, standard, detailed, maximum (default: detailed)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        document_path: str = "",
        material: str = "steel",
        expected_max_stress: float | None = None,
        window_title: str = "FreeCAD",
        detail: str = "detailed",
        **kwargs: Any,
    ) -> str:
        if not self.router:
            return "Error: VLM router not configured. Cannot analyze screenshot."

        try:
            import ctypes
            import ctypes.wintypes
            from PIL import ImageGrab
            from .screen import _find_windows_by_title

            matches = _find_windows_by_title(window_title)
            if not matches:
                return f"Error: No window found matching '{window_title}'. Open FreeCAD with FEA results first."

            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)

            for hwnd, full_title in matches:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                left = max(0, rect.left)
                top = max(0, rect.top)
                right = min(screen_w, rect.right)
                bottom = min(screen_h, rect.bottom)
                if right - left < 10 or bottom - top < 10:
                    continue

                user32.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                save_dir = Path(self.screenshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                filepath = save_dir / f"fea_vlm_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                # Build VLM prompt for FEA analysis
                mat_info = get_material(material)
                mat_str = f"{mat_info.name} (yield={mat_info.yield_strength} MPa)" if mat_info else material

                analyze_prompt = (
                    "You are an FEA stress analysis expert. "
                    "Analyze this FreeCAD FEM contour plot screenshot.\n\n"
                    f"Material: {mat_str}\n"
                )
                if expected_max_stress:
                    analyze_prompt += f"Expected max stress: {expected_max_stress} MPa\n"
                analyze_prompt += (
                    f"Document: {document_path}\n\n"
                    "You MUST respond with EXACTLY this JSON format (no markdown, no backticks, raw JSON only):\n"
                    '{"safe": true/false, "max_stress_region": "where max stress occurs", '
                    '"stress_distribution": "description of stress pattern", '
                    '"safety_concern": "any safety issues, or None", '
                    '"suggestion": "design improvement suggestion, or None", '
                    '"fix_commands": "suggested fc_batch operations to fix, or None"}\n\n'
                    "Consider:\n"
                    "- Is the maximum stress below the yield strength?\n"
                    "- Are there stress concentrations that need fillets?\n"
                    "- Is the stress distribution uniform or localized?\n"
                    "- Safety factor = yield_strength / max_stress (target > 1.5)"
                )

                from ..models.router import VisionDetail

                try:
                    vd = VisionDetail(detail)
                except ValueError:
                    vd = None

                result = self.router.vision(str(filepath), analyze_prompt, detail=vd)

                # Parse structured result
                parsed = _parse_fea_vlm_json(result)

                return (
                    f"[FEA VLM Analysis - Window: '{full_title}']\n"
                    f"SAFE: {parsed['safe']}\n"
                    f"MAX_STRESS_REGION: {parsed['max_stress_region']}\n"
                    f"STRESS_DISTRIBUTION: {parsed['stress_distribution']}\n"
                    f"SAFETY_CONCERN: {parsed['safety_concern']}\n"
                    f"SUGGESTION: {parsed['suggestion']}\n"
                    f"FIX_COMMANDS: {parsed['fix_commands']}\n"
                    f"\n--- Raw VLM output ---\n{result}"
                )

            return f"Error: All matching windows for '{window_title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


# ===========================================================================
# Tool: interference_check
# ===========================================================================

class InterferenceCheckTool(Tool):
    """Check for interference between parts using boolean intersection."""

    name = "interference_check"
    description = (
        "Check for interference (overlap) between parts in a FreeCAD document. "
        "Uses boolean intersection to detect overlapping volumes. "
        "Returns list of interfering pairs with overlap volumes."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to .FCStd file",
                    },
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": "Pairs of object names to check, e.g. [['obj1','obj2']]. Checks all pairs if omitted.",
                    },
                },
                "required": ["document_path"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        pairs: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        fc_python = _find_freecad_python()
        if not fc_python:
            return "Error: FreeCAD not found."

        pairs_json = json.dumps(pairs) if pairs else "None"

        script = f'''
import FreeCAD
import Part
import json

doc = FreeCAD.openDocument(r"{document_path}")
if not doc:
    raise RuntimeError("Failed to open document")

# Collect solid objects
objects = {{}}
for obj in doc.Objects:
    if hasattr(obj, "Shape") and obj.Shape.Solids:
        objects[obj.Name] = obj

names = list(objects.keys())
print(f"Found {{len(names)}} solid objects: {{names}}")

pairs_to_check = {pairs_json}
if not pairs_to_check:
    # Check all pairs
    pairs_to_check = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs_to_check.append([names[i], names[j]])

interferences = []
for pair in pairs_to_check:
    n1, n2 = pair[0], pair[1]
    if n1 not in objects or n2 not in objects:
        print(f"Warning: object {{n1}} or {{n2}} not found")
        continue
    o1 = objects[n1]
    o2 = objects[n2]
    try:
        common = o1.Shape.common(o2.Shape)
        vol = common.Volume
        if vol > 0.001:  # threshold 0.001 mm^3
            interferences.append({{"pair": [n1, n2], "volume": round(vol, 3)}})
            print(f"INTERFERENCE: {{n1}} <-> {{n2}}, volume = {{vol:.3f}} mm3")
        else:
            print(f"OK: {{n1}} <-> {{n2}} (no interference)")
    except Exception as e:
        print(f"Error checking {{n1}} <-> {{n2}}: {{e}}")

if interferences:
    print(f"\\nTOTAL INTERFERENCES: {{len(interferences)}}")
else:
    print("\\nNo interferences found.")
'''

        try:
            output = _run_freecad_script(script)
            return f"[Interference Check]\nDocument: {document_path}\n\n{output}"
        except RuntimeError as e:
            return f"Error checking interference: {e}"


# ===========================================================================
# Tool: tolerance_analysis
# ===========================================================================

class ToleranceAnalysisTool(Tool):
    """Monte Carlo tolerance stack-up analysis.

    Pure Python - does not require FreeCAD.
    Supports normal and uniform distributions.
    """

    name = "tolerance_analysis"
    description = (
        "Monte Carlo tolerance stack-up analysis. "
        "Analyzes dimensional tolerance chains using statistical simulation. "
        "Returns mean, stdev, min, max, and Cpk."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Dimension name"},
                                "nominal": {"type": "number", "description": "Nominal value (mm)"},
                                "tolerance": {"type": "number", "description": "Plus/minus tolerance (mm)"},
                                "distribution": {
                                    "type": "string",
                                    "enum": ["normal", "uniform"],
                                    "description": "Distribution type (default: normal)",
                                },
                            },
                            "required": ["name", "nominal", "tolerance"],
                        },
                        "description": "List of dimensions in the tolerance chain",
                    },
                    "target_dimension": {
                        "type": "string",
                        "description": "Name of the dimension to analyze (sum of all dimensions if omitted)",
                    },
                    "samples": {
                        "type": "integer",
                        "description": "Number of Monte Carlo samples (default: 10000)",
                    },
                    "spec_limit_upper": {
                        "type": "number",
                        "description": "Upper specification limit (optional)",
                    },
                    "spec_limit_lower": {
                        "type": "number",
                        "description": "Lower specification limit (optional)",
                    },
                },
                "required": ["dimensions"],
            },
        )

    def execute(
        self,
        *,
        dimensions: list[dict],
        target_dimension: str = "",
        samples: int = 10000,
        spec_limit_upper: float | None = None,
        spec_limit_lower: float | None = None,
        **kwargs: Any,
    ) -> str:
        if not dimensions:
            return "Error: No dimensions provided"

        # Validate dimensions
        for dim in dimensions:
            if "name" not in dim or "nominal" not in dim or "tolerance" not in dim:
                return f"Error: Each dimension needs name, nominal, and tolerance. Got: {dim}"

        # Run Monte Carlo simulation
        random.seed(42)  # Reproducible
        results = []
        for _ in range(samples):
            total = 0.0
            for dim in dimensions:
                nominal = dim["nominal"]
                tol = dim["tolerance"]
                dist = dim.get("distribution", "normal")
                if dist == "uniform":
                    value = random.uniform(nominal - tol, nominal + tol)
                else:  # normal (default)
                    sigma = tol / 3.0  # 3-sigma process
                    value = random.gauss(nominal, sigma)
                total += value
            results.append(total)

        # Statistics
        mean = sum(results) / len(results)
        variance = sum((x - mean) ** 2 for x in results) / len(results)
        stdev = math.sqrt(variance)
        min_val = min(results)
        max_val = max(results)

        # Cpk calculation
        cpk = None
        if spec_limit_upper is not None and spec_limit_lower is not None and stdev > 0:
            cpu = (spec_limit_upper - mean) / (3 * stdev)
            cpl = (mean - spec_limit_lower) / (3 * stdev)
            cpk = min(cpu, cpl)
        elif spec_limit_upper is not None and stdev > 0:
            cpk = (spec_limit_upper - mean) / (3 * stdev)
        elif spec_limit_lower is not None and stdev > 0:
            cpk = (mean - spec_limit_lower) / (3 * stdev)

        # Build output
        nominal_sum = sum(d["nominal"] for d in dimensions)
        tol_sum = sum(d["tolerance"] for d in dimensions)

        lines = [
            "[Tolerance Analysis - Monte Carlo]",
            f"Dimensions: {len(dimensions)}",
            f"Samples: {samples}",
            f"Nominal sum: {nominal_sum:.4f} mm",
            f"Worst-case tolerance: +/-{tol_sum:.4f} mm",
            "",
            "--- Results ---",
            f"Mean:   {mean:.4f} mm",
            f"StdDev: {stdev:.4f} mm",
            f"Min:    {min_val:.4f} mm",
            f"Max:    {max_val:.4f} mm",
        ]

        if cpk is not None:
            lines.append(f"Cpk:    {cpk:.4f}")
            if cpk >= 1.33:
                lines.append("Assessment: GOOD (Cpk >= 1.33)")
            elif cpk >= 1.0:
                lines.append("Assessment: MARGINAL (1.0 <= Cpk < 1.33)")
            else:
                lines.append("Assessment: POOR (Cpk < 1.0) - high defect risk")

        lines.append("")
        lines.append("--- Dimension Details ---")
        for dim in dimensions:
            dist = dim.get("distribution", "normal")
            lines.append(f"  {dim['name']}: {dim['nominal']} +/- {dim['tolerance']} mm ({dist})")

        return "\n".join(lines)


# ===========================================================================
# Tool: motion_sim (delegates to motion tools)
# ===========================================================================

class MotionSimTool(Tool):
    """Motion simulation tool - performs kinematic analysis via FreeCAD.

    Supports forward kinematics, range checking, and trajectory planning.
    Delegates to motion_range, motion_trajectory tools internally.
    """

    name = "motion_sim"
    description = (
        "Run motion simulation on a FreeCAD assembly. "
        "Performs forward kinematics, joint range analysis, or trajectory planning. "
        "Requires: document path, joint angles or target position."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to .FCStd file with assembly",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "Joint angles for forward kinematics: {object_name: angle_degrees}",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["forward_kinematics", "range_check", "trajectory"],
                        "description": "Analysis type (default: forward_kinematics)",
                    },
                    "target_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target position [x, y, z] for trajectory planning",
                    },
                    "trajectory_steps": {
                        "type": "integer",
                        "description": "Number of trajectory interpolation steps (default: 10)",
                    },
                    "joints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "axis": {"type": "string"},
                                "range": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            },
                        },
                        "description": "Joint definitions for simulation (legacy)",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["kinematic", "dynamic"],
                        "description": "Simulation type (default: kinematic)",
                    },
                },
                "required": ["document_path"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        joint_angles: dict[str, float] | None = None,
        analysis_type: str = "forward_kinematics",
        target_position: list[float] | None = None,
        trajectory_steps: int = 10,
        joints: list[dict] | None = None,
        type: str = "kinematic",
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: Document not found: {document_path}"

        if not document_path.endswith(".FCStd"):
            return f"Error: File must be a .FCStd document, got: {document_path}"

        # Check FreeCAD availability
        fc_python = _find_freecad_python()
        if not fc_python:
            return "Error: FreeCAD not found. Install with: winget install FreeCAD"

        if analysis_type == "forward_kinematics":
            if not joint_angles:
                return "Error: joint_angles required for forward_kinematics analysis"
            return self._run_forward_kinematics(document_path, joint_angles)

        elif analysis_type == "range_check":
            if not joints:
                return "Error: joints list required for range_check analysis"
            return self._run_range_check(document_path, joints)

        elif analysis_type == "trajectory":
            if not joint_angles:
                return "Error: joint_angles (start) required for trajectory analysis"
            if not target_position or len(target_position) != 3:
                return "Error: target_position [x,y,z] required for trajectory analysis"
            return self._run_trajectory(document_path, joint_angles, target_position, trajectory_steps)

        else:
            return f"Error: Unknown analysis_type '{analysis_type}'"

    def _run_forward_kinematics(self, document_path: str, joint_angles: dict[str, float]) -> str:
        from .motion import _build_forward_kinematics_script, _run_freecad_script as _motion_run

        script = _build_forward_kinematics_script(document_path, joint_angles)
        try:
            output = _motion_run(script, timeout=120)
            return (
                f"[Motion Simulation - Forward Kinematics]\n"
                f"Document: {document_path}\n"
                f"Joint angles: {joint_angles}\n"
                f"\n--- Results ---\n{output}"
            )
        except RuntimeError as e:
            return f"Error: {e}"

    def _run_range_check(self, document_path: str, joints: list[dict]) -> str:
        from .motion import _build_range_check_script, _run_freecad_script as _motion_run

        results = []
        for joint in joints:
            jname = joint.get("name", "")
            jtype = joint.get("type", "revolute")
            jrange = joint.get("range", [-180, 180])
            if not jname:
                continue
            script = _build_range_check_script(document_path, jname, jtype, jrange, steps=36)
            try:
                output = _motion_run(script, timeout=120)
                results.append(f"Joint '{jname}' ({jtype}):\n{output}")
            except RuntimeError as e:
                results.append(f"Joint '{jname}': Error - {e}")

        return (
            f"[Motion Simulation - Range Check]\n"
            f"Document: {document_path}\n"
            f"Joints checked: {len(results)}\n"
            f"\n--- Results ---\n" + "\n\n".join(results)
        )

    def _run_trajectory(
        self, document_path: str, start_angles: dict[str, float],
        target_position: list[float], steps: int,
    ) -> str:
        from .motion import _build_trajectory_script, _run_freecad_script as _motion_run

        # Use target position offset as end angles (approximation)
        end_angles = {k: v + 30.0 for k, v in start_angles.items()}
        script = _build_trajectory_script(document_path, start_angles, end_angles, steps)
        try:
            output = _motion_run(script, timeout=120)
            return (
                f"[Motion Simulation - Trajectory]\n"
                f"Document: {document_path}\n"
                f"Start angles: {start_angles}\n"
                f"Target position: {target_position}\n"
                f"Steps: {steps}\n"
                f"\n--- Results ---\n{output}"
            )
        except RuntimeError as e:
            return f"Error: {e}"


# ===========================================================================
# Registration
# ===========================================================================

def register_simulation_tools(
    registry: Any,
    router: Any = None,
    screenshot_dir: str = "",
) -> None:
    """Register all simulation tools.

    Args:
        registry: ToolRegistry instance.
        router: ModelRouter instance (required for fea_vlm_analyze).
        screenshot_dir: Directory for screenshot storage.
    """
    registry.register(FEASetupAndRunTool())
    registry.register(FEAVisualizeTool())
    registry.register(FEAVLMAnalyzeTool(router=router, screenshot_dir=screenshot_dir))
    registry.register(InterferenceCheckTool())
    registry.register(ToleranceAnalysisTool())
    registry.register(MotionSimTool())

    # Register motion tools
    try:
        from .motion import register_motion_tools
        register_motion_tools(registry, router=router, screenshot_dir=screenshot_dir)
    except Exception:
        pass
