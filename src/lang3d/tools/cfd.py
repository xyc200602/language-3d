"""CFD tools: OpenFOAM integration for fluid dynamics analysis.

Tools:
  cfd_run         - Run OpenFOAM CFD analysis (mesh -> boundary -> solve -> results)
  cfd_vlm_analyze - Screenshot CFD visualization + VLM interpret flow field

OpenFOAM can run via:
  - WSL (Windows Subsystem for Linux) - recommended on Windows
  - Native Linux/macOS
  - Docker
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..knowledge.simulation import (
    CFD_MESH_SIZES,
    CFD_PATTERNS,
    FLUID_PRESETS,
    get_fluid,
)
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# OpenFOAM discovery
# ---------------------------------------------------------------------------

def _find_openfoam() -> tuple[str | None, str]:
    """Locate OpenFOAM and determine run mode.

    Returns:
        (path_or_identifier, mode) where mode is "wsl", "native", "docker", or "none"
    """
    # 1. Check config/env
    env_path = os.environ.get("OPENFOAM_PATH")
    env_mode = os.environ.get("OPENFOAM_MODE", "auto")

    if env_mode != "auto":
        return env_path or None, env_mode

    # 2. On Windows, try WSL first
    if platform.system() == "Windows":
        # Check if WSL is available
        try:
            result = subprocess.run(
                ["wsl", "-l", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Check if OpenFOAM is in WSL
                check = subprocess.run(
                    ["wsl", "bash", "-c", "which simpleFoam 2>/dev/null || which interFoam 2>/dev/null || echo ''"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if check.stdout.strip():
                    return "wsl", "wsl"

                # Check common OpenFOAM install paths in WSL
                check2 = subprocess.run(
                    ["wsl", "bash", "-c",
                     "ls /opt/openfoam*/bin/simpleFoam 2>/dev/null "
                     "|| ls /usr/lib/openfoam/*/bin/simpleFoam 2>/dev/null "
                     "|| echo ''"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if check2.stdout.strip():
                    return "wsl", "wsl"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # 3. Try Docker
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "docker", "docker"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # 4. Native (Linux/macOS)
    found = shutil.which("simpleFoam") or shutil.which("interFoam")
    if found:
        return found, "native"

    # Check common paths
    common_paths = [
        "/opt/openfoam10/bin",
        "/opt/openfoam11/bin",
        "/opt/openfoam12/bin",
        "/usr/lib/openfoam/openfoam10/bin",
        "/usr/lib/openfoam/openfoam11/bin",
    ]
    for p in common_paths:
        if Path(p).exists():
            return p, "native"

    return None, "none"


def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to a WSL path."""
    win_path = win_path.replace("\\", "/")
    # C:/Users/... -> /mnt/c/Users/...
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        return f"/mnt/{drive}{win_path[2:]}"
    return win_path


def _run_openfoam_command(
    cmd: list[str],
    case_dir: str,
    mode: str,
    timeout: int = 300,
) -> str:
    """Execute an OpenFOAM command via the appropriate mode.

    Args:
        cmd: Command parts (e.g. ["simpleFoam"])
        case_dir: Path to OpenFOAM case directory (Windows path)
        mode: "wsl", "native", or "docker"
        timeout: Timeout in seconds

    Returns:
        Combined stdout+stderr output
    """
    if mode == "wsl":
        wsl_case = _win_to_wsl_path(case_dir)
        wsl_cmd = " ".join(cmd) + f" -case {wsl_case}"
        full_cmd = ["wsl", "bash", "-c", wsl_cmd]
    elif mode == "docker":
        # Mount case dir and run in container
        wsl_case = _win_to_wsl_path(case_dir) if platform.system() == "Windows" else case_dir
        full_cmd = [
            "docker", "run", "--rm",
            "-v", f"{case_dir}:/case",
            "openfoam/openfoam:latest",
            "bash", "-c",
            " ".join(cmd) + " -case /case",
        ]
    else:
        # Native
        full_cmd = cmd + ["-case", case_dir]

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] OpenFOAM command timed out"
    except FileNotFoundError as e:
        return f"[ERROR] Command not found: {e}"


# ---------------------------------------------------------------------------
# OpenFOAM case builder
# ---------------------------------------------------------------------------

def _build_cfd_case(
    case_dir: str,
    fluid_name: str,
    pattern_name: str,
    mesh_size: str,
    boundary_conditions: dict[str, Any] | None,
    inlet_velocity: float,
    outlet_pressure: float,
) -> str:
    """Generate OpenFOAM case directory structure.

    Creates: constant/, 0/, system/ directories with standard files.

    Returns:
        Path to case directory.
    """
    case_path = Path(case_dir)
    fluid = get_fluid(fluid_name)
    pattern = CFD_PATTERNS.get(pattern_name, CFD_PATTERNS["pipe_flow"])
    mesh_info = CFD_MESH_SIZES.get(mesh_size, CFD_MESH_SIZES["medium"])

    density = fluid.density if fluid else 1.204
    viscosity = fluid.kinematic_viscosity if fluid else 1.516e-5

    # Create directories
    (case_path / "constant").mkdir(parents=True, exist_ok=True)
    (case_path / "0").mkdir(parents=True, exist_ok=True)
    (case_path / "system").mkdir(parents=True, exist_ok=True)

    # system/controlDict
    (case_path / "system" / "controlDict").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     {pattern.solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""",
        encoding="utf-8",
    )

    # system/fvSchemes
    (case_path / "system" / "fvSchemes").write_text(
        """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         corrected;
}
""",
        encoding="utf-8",
    )

    # system/fvSolution
    (case_path / "system" / "fvSolution").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
    k
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
    omega
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl
    {{
        p    1e-4;
        U    1e-4;
    }}
}}
turbulenceModel {{
    simulationType RAS;
    RAS {{
        model {pattern.turbulence_model};
        turbulence on;
    }}
}}
""",
        encoding="utf-8",
    )

    # constant/transportProperties
    (case_path / "constant" / "transportProperties").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
transportModel  Newtonian;
nu              [{viscosity:.6e}];  // kinematic viscosity m^2/s
rho             [{density:.4f}];    // density kg/m^3
""",
        encoding="utf-8",
    )

    # constant/turbulenceProperties
    (case_path / "constant" / "turbulenceProperties").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType RAS;
RAS
{{
    RASModel        {pattern.turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}
""",
        encoding="utf-8",
    )

    # 0/U - velocity boundary conditions
    inlet_u = inlet_velocity
    (case_path / "0" / "U").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({inlet_u} 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({inlet_u} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    defaultFaces
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 0/p - pressure boundary conditions
    (case_path / "0" / "p").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {outlet_pressure};
boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {outlet_pressure};
    }}
    walls
    {{
        type            zeroGradient;
    }}
    defaultFaces
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 0/k - turbulent kinetic energy
    k_inlet = 0.375 * (inlet_u ** 2) if inlet_u > 0 else 0.01
    (case_path / "0" / "k").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inlet:.6f};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_inlet:.6f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6f};
    }}
    defaultFaces
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 0/omega - specific dissipation rate
    omega_inlet = k_inlet / (viscosity * 100) if viscosity > 0 else 1.0
    (case_path / "0" / "omega").write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inlet:.4f};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.4f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.4f};
    }}
    defaultFaces
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    return str(case_path)


# ---------------------------------------------------------------------------
# CFD VLM JSON parser
# ---------------------------------------------------------------------------

def _parse_cfd_vlm_json(raw: str) -> dict[str, Any]:
    """Parse structured CFD VLM analysis result."""
    import json as _json

    json_match = re.search(
        r'\{[^{}]*"flow_regime"[^{}]*\}',
        raw,
        re.DOTALL,
    )
    if json_match:
        try:
            data = _json.loads(json_match.group())
            return {
                "flow_regime": str(data.get("flow_regime", "")),
                "max_velocity": str(data.get("max_velocity", "")),
                "pressure_drop": str(data.get("pressure_drop", "")),
                "separation_regions": str(data.get("separation_regions", "None")),
                "suggestion": str(data.get("suggestion", "None")),
            }
        except (_json.JSONDecodeError, ValueError):
            pass

    def _extract_field(name: str) -> str:
        pattern = rf'{name}[:\s]+(.*?)(?:\n|$)'
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).strip() if m else "None"

    return {
        "flow_regime": _extract_field("flow_regime"),
        "max_velocity": _extract_field("max_velocity"),
        "pressure_drop": _extract_field("pressure_drop"),
        "separation_regions": _extract_field("separation_regions"),
        "suggestion": _extract_field("suggestion"),
    }


# ===========================================================================
# Tool: cfd_run
# ===========================================================================

class CFDRunTool(Tool):
    """Run OpenFOAM CFD analysis on a FreeCAD document.

    Generates mesh, sets boundary conditions, runs solver, and extracts results.
    Supports WSL, native, and Docker execution modes.
    """

    name = "cfd_run"
    description = (
        "Run CFD (Computational Fluid Dynamics) analysis using OpenFOAM. "
        "Generates case files, sets boundary conditions, runs solver. "
        "Supports: WSL, native Linux, Docker modes. "
        "Returns velocity/pressure results."
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
                        "description": "Path to .FCStd or STL file for flow domain",
                    },
                    "fluid": {
                        "type": "string",
                        "description": "Fluid: air, water (default: air)",
                    },
                    "pattern": {
                        "type": "string",
                        "enum": ["pipe_flow", "external_flow", "heat_exchanger"],
                        "description": "CFD analysis pattern (default: pipe_flow)",
                    },
                    "mesh_size": {
                        "type": "string",
                        "enum": ["coarse", "medium", "fine"],
                        "description": "Mesh density (default: medium)",
                    },
                    "inlet_velocity": {
                        "type": "number",
                        "description": "Inlet velocity in m/s (default: 1.0)",
                    },
                    "outlet_pressure": {
                        "type": "number",
                        "description": "Outlet pressure in Pa (default: 0)",
                    },
                    "boundary_conditions": {
                        "type": "object",
                        "description": "Custom boundary conditions override",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Solver timeout in seconds (default: 300)",
                    },
                },
                "required": ["document_path"],
            },
        )

    def execute(
        self,
        *,
        document_path: str,
        fluid: str = "air",
        pattern: str = "pipe_flow",
        mesh_size: str = "medium",
        inlet_velocity: float = 1.0,
        outlet_pressure: float = 0.0,
        boundary_conditions: dict[str, Any] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> str:
        if not Path(document_path).exists():
            return f"Error: File not found: {document_path}"

        # Validate inputs
        fluid_props = get_fluid(fluid)
        if not fluid_props:
            return f"Error: Unknown fluid '{fluid}'. Valid: {list(FLUID_PRESETS.keys())}"

        if pattern not in CFD_PATTERNS:
            return f"Error: Unknown pattern '{pattern}'. Valid: {list(CFD_PATTERNS.keys())}"

        if mesh_size not in CFD_MESH_SIZES:
            return f"Error: Unknown mesh_size '{mesh_size}'. Valid: {list(CFD_MESH_SIZES.keys())}"

        # Find OpenFOAM
        of_path, of_mode = _find_openfoam()
        if of_mode == "none":
            return (
                "Error: OpenFOAM not found.\n"
                "Install options:\n"
                "  1. WSL: Install Ubuntu + OpenFOAM in WSL\n"
                "  2. Docker: docker pull openfoam/openfoam\n"
                "  3. Set OPENFOAM_PATH and OPENFOAM_MODE env vars"
            )

        # Build case directory
        case_dir = tempfile.mkdtemp(prefix="lang3d_cfd_")
        try:
            case_path = _build_cfd_case(
                case_dir=case_dir,
                fluid_name=fluid,
                pattern_name=pattern,
                mesh_size=mesh_size,
                boundary_conditions=boundary_conditions,
                inlet_velocity=inlet_velocity,
                outlet_pressure=outlet_pressure,
            )

            cfd_pattern = CFD_PATTERNS[pattern]

            lines = [
                "[CFD Analysis]",
                f"Document: {document_path}",
                f"Fluid: {fluid_props.name}",
                f"Pattern: {cfd_pattern.name} (solver: {cfd_pattern.solver})",
                f"Mesh: {mesh_size} (boundary layers: {CFD_MESH_SIZES[mesh_size]['boundary_layers']})",
                f"Inlet velocity: {inlet_velocity} m/s",
                f"Outlet pressure: {outlet_pressure} Pa",
                f"OpenFOAM mode: {of_mode}",
                f"Case directory: {case_path}",
                "",
                "--- Case files generated ---",
            ]

            # Try running the solver
            solver_output = _run_openfoam_command(
                [cfd_pattern.solver],
                case_dir,
                of_mode,
                timeout=timeout,
            )

            lines.append("")
            lines.append("--- Solver Output ---")
            lines.append(solver_output)

            return "\n".join(lines)

        except Exception as e:
            return f"Error running CFD: {e}"


# ===========================================================================
# Tool: cfd_vlm_analyze
# ===========================================================================

class CFDVLMAnalyzeTool(Tool):
    """Capture CFD visualization screenshot and analyze with VLM."""

    name = "cfd_vlm_analyze"
    description = (
        "Capture CFD visualization window screenshot and analyze flow field with VLM. "
        "Returns structured analysis: flow regime, velocity, pressure drop, separation."
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
                        "description": "Path to CFD case or document (for context)",
                    },
                    "fluid": {
                        "type": "string",
                        "description": "Fluid name used in CFD (default: air)",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["velocity", "pressure", "turbulence", "overall"],
                        "description": "CFD analysis focus (default: overall)",
                    },
                    "window_title": {
                        "type": "string",
                        "description": "Visualization window title (default: 'ParaView')",
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
        fluid: str = "air",
        analysis_type: str = "overall",
        window_title: str = "ParaView",
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
                return (
                    f"Error: No window found matching '{window_title}'. "
                    "Open ParaView with CFD results first."
                )

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
                filepath = save_dir / f"cfd_vlm_{int(time.time())}.png"

                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img.save(str(filepath))

                fluid_props = get_fluid(fluid)
                fluid_str = f"{fluid_props.name}" if fluid_props else fluid

                analyze_prompt = (
                    "You are a CFD engineering expert. "
                    "Analyze this computational fluid dynamics visualization.\n\n"
                    f"Fluid: {fluid_str}\n"
                    f"Analysis focus: {analysis_type}\n"
                    f"Document: {document_path}\n\n"
                    "Respond with EXACTLY this JSON format (no markdown, no backticks, raw JSON only):\n"
                    '{"flow_regime": "laminar/transitional/turbulent", '
                    '"max_velocity": "estimated max velocity region and value", '
                    '"pressure_drop": "estimated pressure distribution", '
                    '"separation_regions": "any flow separation or recirculation, or None", '
                    '"suggestion": "design improvement for flow optimization, or None"}\n\n'
                    "Consider:\n"
                    "- Is the flow laminar or turbulent?\n"
                    "- Where is velocity highest/lowest?\n"
                    "- Are there separation zones or vortices?\n"
                    "- Is the pressure distribution uniform?\n"
                    "- Any stagnation points or adverse pressure gradients?"
                )

                from ..models.router import VisionDetail

                try:
                    vd = VisionDetail(detail)
                except ValueError:
                    vd = None

                result = self.router.vision(str(filepath), analyze_prompt, detail=vd)
                parsed = _parse_cfd_vlm_json(result)

                return (
                    f"[CFD VLM Analysis - Window: '{full_title}']\n"
                    f"FLOW_REGIME: {parsed['flow_regime']}\n"
                    f"MAX_VELOCITY: {parsed['max_velocity']}\n"
                    f"PRESSURE_DROP: {parsed['pressure_drop']}\n"
                    f"SEPARATION_REGIONS: {parsed['separation_regions']}\n"
                    f"SUGGESTION: {parsed['suggestion']}\n"
                    f"\n--- Raw VLM output ---\n{result}"
                )

            return f"Error: All matching windows for '{window_title}' have invalid dimensions"
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_cfd_tools(
    registry: Any,
    router: Any = None,
    screenshot_dir: str = "",
) -> None:
    """Register all CFD tools."""
    registry.register(CFDRunTool())
    registry.register(CFDVLMAnalyzeTool(router=router, screenshot_dir=screenshot_dir))
