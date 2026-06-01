"""CAD utility tools - generic operations not tied to a specific CAD package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


class CADCheckTool(Tool):
    """Check if a CAD file is valid."""

    name = "cad_check"
    description = "Check if a CAD/3D model file exists and get its properties"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CAD file (STL, STEP, etc.)",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, **kwargs: Any) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        suffix = p.suffix.lower()
        size = p.stat().st_size
        info = [f"File: {p}", f"Size: {size:,} bytes ({size / 1024:.1f} KB)", f"Type: {suffix}"]

        if suffix == ".stl":
            try:
                # Quick STL validation: check header and count triangles
                text = p.read_bytes()[:80]
                is_binary = not text.startswith(b"solid") or b"\x00" in text[10:]

                if is_binary:
                    data = p.read_bytes()
                    num_triangles = int.from_bytes(data[80:84], "little")
                    info.append(f"Format: Binary STL")
                    info.append(f"Triangles: {num_triangles:,}")
                else:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    num_triangles = content.count("facet normal")
                    info.append(f"Format: ASCII STL")
                    info.append(f"Triangles: {num_triangles:,}")
            except Exception as e:
                info.append(f"STL parse error: {e}")

        return "\n".join(info)


class MeshStatsTool(Tool):
    """Get statistics about an STL mesh using trimesh or basic parsing."""

    name = "mesh_stats"
    description = "Analyze an STL mesh and return statistics (dimensions, volume, surface area)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the STL file",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, *, path: str, **kwargs: Any) -> str:
        try:
            import trimesh

            mesh = trimesh.load(path)
            bounds = mesh.bounds
            return (
                f"Mesh: {path}\n"
                f"Vertices: {len(mesh.vertices):,}\n"
                f"Faces: {len(mesh.faces):,}\n"
                f"Bounding box: {bounds[0]} to {bounds[1]}\n"
                f"Dimensions: {mesh.extents}\n"
                f"Volume: {mesh.volume:.6f}\n"
                f"Surface area: {mesh.area:.6f}\n"
                f"Is watertight: {mesh.is_watertight}"
            )
        except ImportError:
            # Fallback: just basic info
            p = Path(path)
            if not p.exists():
                return f"Error: File not found: {path}"
            size = p.stat().st_size
            return f"File: {p} ({size:,} bytes)\n(Install trimesh for detailed mesh analysis: pip install trimesh)"
        except Exception as e:
            return f"Error: {e}"


def register_cad_utils(registry: Any) -> None:
    """Register CAD utility tools."""
    registry.register(CADCheckTool())
    registry.register(MeshStatsTool())
