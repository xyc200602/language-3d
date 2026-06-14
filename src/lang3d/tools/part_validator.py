"""Batch part validation — FreeCAD execution + auto-retry + VLM verification.

For each part, the validator:
1. Generates FreeCAD ops via ``part_feature_engine.generate_ops``
2. Builds the script via ``freecad._build_script``
3. Runs it through FreeCAD's bundled Python
4. Checks STL output (existence + reasonable size)
5. On failure, progressively simplifies features and retries
6. Optionally renders the STL via VTK and sends to VLM for verification

This automates what was previously manual debugging (e.g. battery_box
shell failures).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Part
from .part_feature_engine import FeatureConfig, generate_ops, infer_features

logger = logging.getLogger(__name__)


# ============================================================================
# Data model
# ============================================================================


@dataclass
class PartValidationResult:
    """Result of validating a single part."""

    part_name: str
    passed: bool = False
    stl_path: str | None = None
    stl_size_bytes: int = 0
    freecad_stdout: str = ""
    freecad_error: str | None = None
    # How many simplification levels were needed (0 = full features passed)
    simplification_level: int = 0
    # Description of what was simplified
    simplification_note: str = ""
    # VLM results (optional)
    vlm_verified: bool | None = None
    vlm_observed: str = ""
    vlm_match: bool | None = None
    vlm_error: str = ""
    # trimesh geometric analysis (watertight, winding, body_count, ...).
    # Populated only when trimesh was able to load the STL; empty dict means
    # the geometric check was skipped (e.g. trimesh import failed or the
    # file could not be parsed).  Never blocks the pipeline on its own —
    # only ``watertight=False`` flips the part to FAIL.
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def stl_size_kb(self) -> float:
        return self.stl_size_bytes / 1024


@dataclass
class BatchValidationReport:
    """Summary of validating all parts."""

    results: list[PartValidationResult] = field(default_factory=list)
    total_parts: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0  # FreeCAD not available

    @property
    def pass_rate(self) -> float:
        if self.total_parts == 0:
            return 0.0
        return self.passed / self.total_parts

    @property
    def failed_parts(self) -> list[str]:
        return [r.part_name for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parts": self.total_parts,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": round(self.pass_rate, 3),
            "failed_parts": self.failed_parts,
            "results": [
                {
                    "part": r.part_name,
                    "passed": r.passed,
                    "stl_size_kb": round(r.stl_size_kb, 1),
                    "simplification_level": r.simplification_level,
                    "simplification_note": r.simplification_note,
                    "vlm_verified": r.vlm_verified,
                    "vlm_match": r.vlm_match,
                    "vlm_error": r.vlm_error,
                    "quality": r.quality,
                    "error": r.freecad_error,
                }
                for r in self.results
            ],
        }


# ============================================================================
# Feature simplification
# ============================================================================

# Each level removes one category of features.  Level 0 = full features.
_SIMPLIFICATION_STEPS: list[tuple[str, str]] = [
    # (attribute_name_on_FeatureConfig, human_note)
    ("fillets", "removed fillets"),
    ("chamfers", "removed chamfers"),
    ("cable_channels", "removed cable channels"),
    ("shell", "removed shell"),
    ("bearing_seats", "removed bearing seats"),
    ("mounting_holes", "removed mounting holes (kept bore)"),
    ("bore", "removed bore/keyway"),
]


def _simplify_config(config: FeatureConfig, level: int) -> FeatureConfig:
    """Return a copy of *config* with features removed up to *level*.

    Level 0: full features.
    Level 1: no fillets.
    Level 2: no fillets, no chamfers.
    ...
    Level N (beyond list): empty config = primitive fallback.
    """
    import copy
    cfg = copy.deepcopy(config)

    for i in range(min(level, len(_SIMPLIFICATION_STEPS))):
        attr, _note = _SIMPLIFICATION_STEPS[i]
        if hasattr(cfg, attr):
            default: list | None = [] if isinstance(getattr(cfg, attr), list) else None
            setattr(cfg, attr, default)

    return cfg


def _simplification_note(level: int) -> str:
    if level == 0:
        return "full features"
    parts = []
    for i in range(min(level, len(_SIMPLIFICATION_STEPS))):
        _attr, note = _SIMPLIFICATION_STEPS[i]
        parts.append(note)
    if level > len(_SIMPLIFICATION_STEPS):
        parts.append("primitive fallback")
    return "; ".join(parts)


# ============================================================================
# STL validation helpers
# ============================================================================

_MIN_STL_BYTES = 100  # anything smaller is likely garbage
_MAX_STL_BYTES = 100 * 1024 * 1024  # 100 MB sanity cap


def _validate_stl(stl_path: str) -> tuple[bool, int, str, dict[str, Any]]:
    """Check STL file exists, has reasonable size, and run geometric checks.

    After the size gate passes, loads the STL via trimesh and inspects:

    - ``mesh.is_watertight`` — closed surface, no holes (FAIL if False)
    - ``mesh.is_winding_consistent`` — face normals agree (warning only)
    - ``mesh.split()`` length — multiple disjoint bodies (warning only)

    The geometric results are returned in the ``quality`` dict so they can
    be attached to the ``PartValidationResult`` and serialized into the
    validation report JSON.

    trimesh failures (import error, parse error) are logged as warnings
    and degrade gracefully — they never block the pipeline.  Only a
    successfully-loaded non-watertight mesh flips ``ok`` to False.

    Returns ``(ok, size_bytes, error_message, quality)``.
    """
    quality: dict[str, Any] = {}
    if not os.path.isfile(stl_path):
        return False, 0, f"STL file not found: {stl_path}", quality
    size = os.path.getsize(stl_path)
    if size < _MIN_STL_BYTES:
        return False, size, f"STL too small ({size} bytes, minimum {_MIN_STL_BYTES})", quality
    if size > _MAX_STL_BYTES:
        return False, size, f"STL suspiciously large ({size} bytes)", quality

    # ---- trimesh geometric analysis (graceful degradation) ----
    try:
        import trimesh
        # process=True is required for is_watertight / split() to work —
        # STL stores each triangle with its own vertices, so the loader
        # must merge duplicates to recover the true topology.
        mesh = trimesh.load(stl_path, force="mesh", process=True)
        # trimesh may return a Scene or list for multi-body files; coerce
        # to a single Trimesh by concatenating.
        if not isinstance(mesh, trimesh.Trimesh):
            try:
                mesh = trimesh.util.concatenate(
                    [m for m in (mesh.geometry.values()
                                 if hasattr(mesh, "geometry")
                                 else [mesh])
                     if isinstance(m, trimesh.Trimesh)]
                )
            except Exception:
                mesh = None
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            quality["checked"] = False
            quality["error"] = "trimesh returned empty mesh"
            logger.warning(
                "trimesh loaded no geometry for %s — skipping geometric checks",
                stl_path,
            )
            return True, size, "", quality

        bodies = mesh.split() if hasattr(mesh, "split") else [mesh]
        body_count = len(bodies) if bodies is not None else 1

        quality["checked"] = True
        quality["watertight"] = bool(mesh.is_watertight)
        quality["winding_consistent"] = bool(mesh.is_winding_consistent)
        quality["body_count"] = int(body_count)
        quality["vertex_count"] = int(len(mesh.vertices))
        quality["triangle_count"] = int(len(mesh.faces))

        if not quality["watertight"]:
            return False, size, (
                f"mesh not watertight (bodies={body_count}, "
                f"tris={quality['triangle_count']}, "
                f"winding={quality['winding_consistent']})"
            ), quality

        if body_count > 1:
            logger.warning(
                "STL %s contains %d disjoint bodies (winding_consistent=%s)",
                os.path.basename(stl_path), body_count, quality["winding_consistent"],
            )
        if not quality["winding_consistent"]:
            logger.warning(
                "STL %s has inconsistent face winding", os.path.basename(stl_path),
            )
    except Exception as e:
        quality["checked"] = False
        quality["error"] = f"{type(e).__name__}: {e}"
        logger.warning(
            "trimesh geometry check skipped for %s: %s",
            os.path.basename(stl_path), e,
        )

    return True, size, "", quality


# ============================================================================
# FreeCAD availability check
# ============================================================================


def _freecad_available() -> bool:
    """Check if FreeCAD's bundled Python is available."""
    try:
        from .freecad import _find_freecad_python
        return _find_freecad_python() is not None
    except Exception:
        return False


# ============================================================================
# Single-part validation with auto-retry
# ============================================================================


def validate_part(
    part: Part,
    workspace: str,
    max_simplification: int = len(_SIMPLIFICATION_STEPS),
    timeout: int = 120,
    *,
    vlm_router: Any = None,
    vlm_detail: str = "detailed",
    joints: list | None = None,
) -> PartValidationResult:
    """Validate a single part through FreeCAD execution.

    Ties together: ops generation → script build → FreeCAD run → STL check.
    On failure, progressively simplifies features and retries.

    Args:
        part: The Part to validate.
        workspace: Directory for STL output.
        max_simplification: Max simplification level to try (default: all).
        timeout: FreeCAD subprocess timeout in seconds.
        vlm_router: Optional ModelRouter for VLM visual verification.
        vlm_detail: VLM detail level string.
        joints: Optional list of Joint objects involving this part.
                When provided, connection features are generated.

    Returns:
        PartValidationResult with pass/fail and diagnostics.
    """
    result = PartValidationResult(part_name=part.name)

    os.makedirs(workspace, exist_ok=True)
    config = infer_features(part)

    for level in range(max_simplification + 1):
        cfg = _simplify_config(config, level)
        note = _simplification_note(level)

        # Generate ops and script
        try:
            ops = generate_ops(part, config=cfg, joints=joints)
        except Exception as e:
            result.freecad_error = f"generate_ops failed at level {level}: {e}"
            result.simplification_level = level
            result.simplification_note = note
            continue

        # Build and run FreeCAD script
        stl_path = os.path.join(workspace, f"{part.name}.stl")
        ok, stdout, stderr = _run_and_check(ops, stl_path, workspace, timeout)

        if ok:
            stl_ok, stl_size, stl_err, quality = _validate_stl(stl_path)
            if stl_ok:
                result.passed = True
                result.stl_path = stl_path
                result.stl_size_bytes = stl_size
                result.freecad_stdout = stdout
                result.simplification_level = level
                result.simplification_note = note
                result.quality = quality
                logger.info(
                    "Part '%s' PASSED at simplification level %d (%s, %d KB)",
                    part.name, level, note, stl_size // 1024,
                )
                break
            else:
                result.freecad_error = stl_err
                result.freecad_stdout = stdout
                result.simplification_level = level
                result.simplification_note = note
        else:
            result.freecad_error = stderr
            result.freecad_stdout = stdout
            result.simplification_level = level
            result.simplification_note = note
            logger.debug(
                "Part '%s' failed at level %d (%s): %s",
                part.name, level, note, (stderr or "")[:200],
            )

    # Optional VLM verification
    if result.passed and result.stl_path and vlm_router is not None:
        _vlm_verify(result, part, vlm_router, vlm_detail)

    return result


def _run_and_check(
    ops: list[dict],
    expected_stl: str,
    workspace: str,
    timeout: int,
) -> tuple[bool, str, str | None]:
    """Build script, run via FreeCAD, return (success, stdout, error).

    Also removes stale STL from previous attempts before running.
    """
    from .freecad import _build_script, _run_freecad_script

    # Clean up old STL to avoid false positives
    if os.path.isfile(expected_stl):
        os.remove(expected_stl)

    script = _build_script(ops)
    # Replace workspace placeholder
    script = script.replace("{WORKSPACE}", workspace.replace("\\", "/"))

    try:
        stdout = _run_freecad_script(script, timeout=timeout)
        return True, stdout, None
    except RuntimeError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", str(e)


# ============================================================================
# VLM visual verification
# ============================================================================


def _vlm_verify(
    result: PartValidationResult,
    part: Part,
    router: Any,
    detail: str,
) -> None:
    """Render STL via VTK and send to VLM for structural verification."""
    logger.info("VLM verifying part '%s' (stl=%s)", part.name, result.stl_path)
    try:
        from .vtk_renderer import render_stl_multi_angle

        with tempfile.TemporaryDirectory(prefix="vlm_validate_") as tmpdir:
            pngs = render_stl_multi_angle(result.stl_path, tmpdir)
            if not pngs:
                result.vlm_verified = False
                return

            # Build prompt
            prompt = _build_part_verify_prompt(part)

            # Send first (isometric) view to VLM
            vlm_response = router.vision(pngs[0], prompt)
            result.vlm_verified = True
            result.vlm_observed = vlm_response[:500]

            # Parse match
            from .vlm import _parse_verification_json
            parsed = _parse_verification_json(vlm_response)
            result.vlm_match = parsed.get("match", False)

    except ImportError as e:
        result.vlm_verified = False
        result.vlm_error = f"VTK import error: {e}"
    except Exception as e:
        result.vlm_verified = False
        result.vlm_error = f"{type(e).__name__}: {e}"


def _build_part_verify_prompt(part: Part) -> str:
    """Build a VLM verification prompt for a single part."""
    family = part.name.split("_")[0] if "_" in part.name else part.name
    dims = part.dimensions
    dim_str = ", ".join(f"{k}={v}" for k, v in dims.items())

    return (
        "You are a 3D CAD model verification expert.\n\n"
        f"Verify this rendered 3D model of '{part.name}' ({part.category}).\n"
        f"Dimensions: {dim_str}\n\n"
        "Step 1: Describe what you see (shape, features like holes, slots, walls).\n"
        "Step 2: Check if the model has appropriate engineering features.\n"
        "Step 3: Give your conclusion.\n\n"
        'Respond with JSON: {"match": true/false, "observed": "description", '
        '"differences": "null", "suggestion": "null", '
        '"confidence": "high/medium/low"}\n\n'
        "MATCHING RULES:\n"
        "- Be GENEROUS. If the basic shape matches, report match=true.\n"
        "- Missing small features (tiny chamfers, thin walls) should still match.\n"
        "- Focus on: correct overall shape, any visible holes/slots/bore.\n"
    )


# ============================================================================
# Batch validation
# ============================================================================


def validate_all_parts(
    parts: list[Part],
    workspace: str,
    max_simplification: int = len(_SIMPLIFICATION_STEPS),
    timeout: int = 120,
    *,
    vlm_router: Any = None,
    vlm_detail: str = "detailed",
    vlm_sample: int = 0,
    joints_by_part: dict[str, list] | None = None,
) -> BatchValidationReport:
    """Validate all parts through FreeCAD execution.

    Args:
        parts: List of Part objects to validate.
        workspace: Directory for STL output.
        max_simplification: Max simplification level per part.
        timeout: FreeCAD subprocess timeout in seconds.
        vlm_router: Optional ModelRouter for VLM verification.
        vlm_detail: VLM detail level.
        vlm_sample: If > 0, only run VLM verification on this many parts
                    (sampled from passed parts).
        joints_by_part: Optional mapping of part_name → list[Joint].
                       When provided, connection features are generated
                       for each part during validation.

    Returns:
        BatchValidationReport with per-part results and summary.
    """
    report = BatchValidationReport(total_parts=len(parts))

    if not _freecad_available():
        logger.error("FreeCAD not available — skipping validation")
        report.skipped = len(parts)
        for p in parts:
            report.results.append(PartValidationResult(
                part_name=p.name,
                freecad_error="FreeCAD not available",
            ))
        return report

    # Decide which parts get VLM verification
    vlm_indices: set[int] = set()
    if vlm_router is not None and vlm_sample > 0:
        import random
        vlm_indices = set(random.sample(range(len(parts)), min(vlm_sample, len(parts))))

    for i, part in enumerate(parts):
        logger.info("Validating %d/%d: %s", i + 1, len(parts), part.name)
        use_vlm = vlm_router if (i in vlm_indices) else None
        part_joints = joints_by_part.get(part.name, []) if joints_by_part else None
        result = validate_part(
            part, workspace,
            max_simplification=max_simplification,
            timeout=timeout,
            vlm_router=use_vlm,
            vlm_detail=vlm_detail,
            joints=part_joints,
        )
        report.results.append(result)
        if result.passed:
            report.passed += 1
        else:
            report.failed += 1

    logger.info(
        "Batch validation complete: %d/%d passed (%.1f%%)",
        report.passed, report.total_parts, report.pass_rate * 100,
    )
    if report.failed_parts:
        logger.warning("Failed parts: %s", ", ".join(report.failed_parts))

    return report
