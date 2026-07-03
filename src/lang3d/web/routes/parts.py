"""Part library routes: catalog, template, generate, analyze, assemble.

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
    from ...app import DATA_ROOT
    return DATA_ROOT

def _workspace_root():
    from ...app import _workspace_root as _wr
    return _wr()

def _resolve_safe(path, ws):
    from ...app import _resolve_safe as _rs
    return _rs(path, ws)

def _find_freecad():
    from ...app import _find_freecad as _ff
    return _ff()

@router.get("/api/parts/catalog")
async def api_parts_catalog(
    query: str = Query("", description="Search keyword"),
    category: str = Query("", description="Filter by category"),
) -> JSONResponse:
    """List/search part templates in the catalog."""
    try:
        from ...knowledge.parts_catalog import (
            CATEGORY_TREE,
            get_all_templates,
            search_parts,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    if query or category:
        results = search_parts(
            query=query,
            category=category if category else None,
        )
    else:
        results = get_all_templates()

    templates = []
    for t in results:
        templates.append({
            "id": t.id,
            "name_en": t.name_en,
            "name_cn": t.name_cn,
            "category": t.category,
            "subcategory": t.subcategory,
            "description": t.description,
            "tags": t.tags,
            "material_default": t.material_default,
            "parameters": [
                {
                    "name": p.name,
                    "display_name_cn": p.display_name_cn,
                    "unit": p.unit,
                    "default": p.default,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "step": p.step,
                    "fixed": p.fixed,
                    "param_type": p.param_type,
                    "choices": p.choices,
                }
                for p in t.parameters
            ],
            "standard_sizes": t.standard_sizes,
            "notes": t.notes,
            "quality_levels": t.quality_levels,
        })

    return JSONResponse({
        "templates": templates,
        "total": len(templates),
        "categories": CATEGORY_TREE,
    })


@router.get("/api/parts/template/{part_id}")
async def api_parts_template(part_id: str) -> JSONResponse:
    """Get detailed info for a single part template."""
    try:
        from ...knowledge.parts_catalog import get_template
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    template = get_template(part_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Part template '{part_id}' not found")

    return JSONResponse({
        "id": template.id,
        "name_en": template.name_en,
        "name_cn": template.name_cn,
        "category": template.category,
        "subcategory": template.subcategory,
        "description": template.description,
        "tags": template.tags,
        "material_default": template.material_default,
        "parameters": [
            {
                "name": p.name,
                "display_name_cn": p.display_name_cn,
                "unit": p.unit,
                "default": p.default,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "step": p.step,
                "fixed": p.fixed,
                "param_type": p.param_type,
                "choices": p.choices,
            }
            for p in template.parameters
        ],
        "standard_sizes": template.standard_sizes,
        "notes": template.notes,
        "quality_levels": template.quality_levels,
    })


@router.post("/api/parts/generate")
async def api_parts_generate(payload: dict[str, Any]) -> JSONResponse:
    """Generate a parametric part on the server."""
    part_id = payload.get("part_id", "")
    parameters = payload.get("parameters")
    variant_index = payload.get("variant_index")

    if not part_id:
        raise HTTPException(status_code=400, detail="Missing 'part_id'")

    try:
        from ...knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    template = get_template(part_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Part template '{part_id}' not found")

    # Resolve parameters
    try:
        if variant_index is not None:
            idx = int(variant_index)
            if idx < 0 or idx >= len(template.standard_sizes):
                raise HTTPException(status_code=400, detail="variant_index out of range")
            parameters = template.standard_sizes[idx]
        resolved = resolve_parameters(template, parameters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate script
    try:
        model_script = format_fc_script(template, resolved)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Script template error: {e}")

    # Build output paths
    ws = _workspace_root()
    parts_dir = ws / "parts_library"
    parts_dir.mkdir(parents=True, exist_ok=True)
    param_desc = "_".join(f"{k}{int(v) if v == int(v) else v}" for k, v in resolved.items())
    safe_name = f"{part_id}_{param_desc}"
    fcstd_path = parts_dir / f"{safe_name}.FCStd"
    stl_path = parts_dir / f"{safe_name}.stl"

    full_script = model_script + f"""
import os
import subprocess
import tempfile
import threading
import time
os.makedirs(r'{parts_dir}', exist_ok=True)
doc.saveAs(r'{fcstd_path}')
print(f"Saved: {fcstd_path}")
import Mesh
_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]
if _export_list:
    Mesh.export(_export_list, r'{stl_path}')
    print(f"STL: {stl_path} ({os.path.getsize(r'{stl_path}'):,} bytes)")
"""

    # Try to run FreeCAD
    freecad = _find_freecad()
    fc_python = None
    if freecad:
        # Find FreeCAD Python from same directory
        fc_python = str(Path(freecad).parent / "python.exe")
        if not Path(fc_python).exists():
            fc_python = None

    # Also try the FreeCAD tool's finder
    if not fc_python:
        try:
            from ...tools.freecad import _find_freecad_python
            fc_python = _find_freecad_python()
        except Exception as e:  # FreeCAD not installed — optional dep
            logger.debug("FreeCAD python not found: %s", e)

    if not fc_python:
        raise HTTPException(status_code=503, detail="FreeCAD not available for part generation")

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(full_script)
        script_path = tf.name

    try:
        proc = subprocess.run(
            [fc_python, script_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FreeCAD error: {(proc.stderr or proc.stdout)[:500]}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="FreeCAD generation timed out")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    result: dict[str, Any] = {
        "part_id": part_id,
        "name": template.name_cn,
        "parameters": resolved,
        "fcstd_path": str(fcstd_path),
    }
    if stl_path.exists():
        result["stl_path"] = str(stl_path)
        result["stl_size_kb"] = stl_path.stat().st_size // 1024
        # Make relative to workspace for web serving
        try:
            result["stl_rel"] = str(stl_path.relative_to(ws)).replace("\\", "/")
        except ValueError:
            pass

    return JSONResponse(result)


@router.get("/api/parts/generated")
async def api_parts_generated() -> JSONResponse:
    """List all generated/imported parts with file existence checks."""
    try:
        from ...tools.part_library import _get_parts_store
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    store = _get_parts_store()
    parts = []
    for p in store.list_all():
        entry = p.to_dict()
        entry["fcstd_exists"] = Path(p.fcstd_path).exists() if p.fcstd_path else False
        entry["stl_exists"] = Path(p.stl_path).exists() if p.stl_path else False
        parts.append(entry)

    return JSONResponse({"parts": parts, "total": len(parts)})


@router.delete("/api/parts/generated/{name}")
async def api_parts_generated_delete(name: str) -> JSONResponse:
    """Delete a generated part record by name."""
    try:
        from ...tools.part_library import _get_parts_store
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    store = _get_parts_store()
    removed = store.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Part '{name}' not found in store")
    return JSONResponse({"status": "deleted", "name": name})


@router.post("/api/parts/analyze")
async def api_parts_analyze(payload: dict[str, Any]) -> JSONResponse:
    """Run 3D print feasibility analysis on an STL/FCStd file."""
    stl_path = payload.get("stl_path", "")
    orientation = payload.get("orientation", "auto")

    if not stl_path:
        raise HTTPException(status_code=400, detail="Missing 'stl_path'")

    # Validate file exists and is in workspace
    ws = _workspace_root()
    resolved = _resolve_safe(stl_path, ws)
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found or access denied")

    try:
        from ...tools.part_library import _run_print_analysis
        result_str = _run_print_analysis(str(resolved), orientation)
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Try to extract JSON from the result
    analysis_data = {"raw": result_str}
    for line in result_str.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                analysis_data = json.loads(line)
            except json.JSONDecodeError:
                pass
            break

    return JSONResponse({"analysis": analysis_data, "stl_path": str(resolved)})


@router.post("/api/parts/assemble")
async def api_parts_assemble(payload: dict[str, Any]) -> JSONResponse:
    """Assemble multiple parts into a single FreeCAD document."""
    assembly_name = payload.get("assembly_name", "")
    parts = payload.get("parts", [])
    output_format = payload.get("output_format", "fcstd")

    if not assembly_name:
        raise HTTPException(status_code=400, detail="Missing 'assembly_name'")
    if not parts:
        raise HTTPException(status_code=400, detail="Missing 'parts' list")

    # Validate all part files exist
    ws = _workspace_root()
    validated_parts = []
    for p in parts:
        fpath = p.get("file", "")
        if not fpath:
            raise HTTPException(status_code=400, detail="Each part must have a 'file' field")
        resolved = _resolve_safe(fpath, ws)
        if resolved is None:
            resolved = _resolve_safe(fpath, _get_data_root())
        if resolved is None or not resolved.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {fpath}")
        validated_parts.append({
            "file": str(resolved),
            "name": p.get("name", Path(fpath).stem),
            "position": p.get("position", [0, 0, 0]),
            "rotation": p.get("rotation", [0, 0, 1, 0]),
        })

    try:
        from ...tools.part_library import PartAssembleTool
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    tool = PartAssembleTool()
    result_str = tool.execute(
        assembly_name=assembly_name,
        parts=validated_parts,
        output_format=output_format,
    )

    if result_str.startswith("错误"):
        raise HTTPException(status_code=500, detail=result_str)

    # Find generated files
    parts_dir = ws / "parts_library"
    fcstd_file = parts_dir / f"{assembly_name}.FCStd"
    stl_file = parts_dir / f"{assembly_name}.stl"

    result: dict[str, Any] = {
        "assembly_name": assembly_name,
        "part_count": len(validated_parts),
        "output_format": output_format,
        "result": result_str,
    }
    if fcstd_file.exists():
        result["fcstd_path"] = str(fcstd_file)
        try:
            result["fcstd_rel"] = str(fcstd_file.relative_to(ws)).replace("\\", "/")
        except ValueError:
            pass
    if stl_file.exists():
        result["stl_path"] = str(stl_file)
        result["stl_size_kb"] = stl_file.stat().st_size // 1024

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Routes: slicing
# ---------------------------------------------------------------------------
