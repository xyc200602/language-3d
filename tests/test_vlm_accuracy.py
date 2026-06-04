"""VLM verification accuracy test — measure cad_verify correctness.

Uses existing FCStd files from data/projects/engineering/ to test
cad_verify with different configurations (standard, detailed, multi-angle).
No modeling needed — only tests the verification pipeline.

Usage:
  # Run all VLM accuracy tests
  python tests/test_vlm_accuracy.py

  # Run via pytest
  pytest tests/test_vlm_accuracy.py -v -m e2e
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    return bool(os.environ.get("GLM_API_KEY"))


def _has_freecad() -> bool:
    return any(
        (Path(p) / "python.exe").exists()
        for p in [
            os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
            r"C:\Program Files\FreeCAD 1.1\bin",
            r"C:\Program Files\FreeCAD\bin",
        ]
    )


skip_reason = "需要 GLM API + FreeCAD"


# ---------------------------------------------------------------------------
# Test cases — existing models + expected descriptions
# ---------------------------------------------------------------------------

VLM_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "cantilever_bracket",
        "file": "data/projects/engineering/cantilever_bracket.FCStd",
        "description": (
            "悬臂支架：底板120x80x10mm，竖板10x80x60mm垂直相连，"
            "底板4个直径8mm安装孔（四角距边15mm），竖板1个直径15mm轴承孔"
        ),
    },
    {
        "id": "stepped_shaft",
        "file": "data/projects/engineering/stepped_shaft.FCStd",
        "description": (
            "阶梯轴：第一段直径20mm长30mm，第二段直径15mm长25mm，"
            "第三段直径10mm长20mm，同轴连接。第二段有键槽4x2x12mm"
        ),
    },
    {
        "id": "l_bracket",
        "file": "data/projects/engineering/l_bracket.FCStd",
        "description": (
            "L型支架：底板80x60x8mm，竖板8x60x40mm垂直相连，"
            "底板有2个直径6mm安装孔距两端15mm"
        ),
    },
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    case_id: str
    config: str               # "standard" / "detailed" / "multi_angle"
    match: bool | None = None  # True=MATCH, False=MISMATCH, None=error
    elapsed_seconds: float = 0.0
    error: str | None = None
    raw_output: str = ""


def _parse_match_from_output(output: str) -> bool | None:
    """Parse the match result from cad_verify output."""
    lower = output.lower()
    # Look for structured MATCH field
    if "match: true" in lower or '"match": true' in lower or "'match': true" in lower:
        return True
    if "match: false" in lower or '"match": false' in lower or "'match': false" in lower:
        return False
    return None


def _safe_print(text: str, max_len: int = 300) -> None:
    try:
        print(text[:max_len])
    except UnicodeEncodeError:
        safe = text[:max_len].encode("gbk", errors="replace").decode("gbk")
        print(safe)


# ---------------------------------------------------------------------------
# Direct tool invocation
# ---------------------------------------------------------------------------

def run_verify_test(
    case: dict[str, Any],
    config: str = "standard",
    *,
    project_root: Path | None = None,
) -> VerifyResult:
    """Run a single cad_verify test against an existing FCStd file.

    Steps:
      1. fc_open_gui — open the FCStd file in FreeCAD
      2. Wait for GUI to stabilize
      3. cad_verify — capture and verify with specified config
      4. fc_close_gui — cleanup
    """
    from lang3d.config import load_config
    from lang3d.models.router import ModelRouter
    from lang3d.tools.freecad import FCOpenGUITool, FCCloseGUITool
    from lang3d.tools.vlm import CADVerifyTool

    if project_root is None:
        project_root = Path(__file__).parent.parent

    fcstd_path = project_root / case["file"]
    if not fcstd_path.exists():
        return VerifyResult(
            case_id=case["id"],
            config=config,
            match=None,
            error=f"File not found: {fcstd_path}",
        )

    config_obj = load_config()
    screenshot_dir = config_obj.agent.screenshot_dir
    router = ModelRouter(config_obj)

    open_tool = FCOpenGUITool()
    close_tool = FCCloseGUITool()
    verify_tool = CADVerifyTool(router, screenshot_dir=screenshot_dir)

    result = VerifyResult(case_id=case["id"], config=config)

    try:
        # Step 1: Open FreeCAD GUI with the file
        _safe_print(f"    Opening {fcstd_path.name}...")
        open_result = open_tool.execute(file_path=str(fcstd_path), view="isometric", wait_seconds=8)
        _safe_print(f"    Open result: {open_result[:150]}")

        # Step 2: Wait for window to stabilize
        time.sleep(3)

        # Step 3: Run verification
        start = time.time()

        if config == "multi_angle":
            verify_output = verify_tool.execute(
                expected=case["description"],
                detail="detailed",
                angles="isometric,front,top",
            )
        elif config == "detailed":
            verify_output = verify_tool.execute(
                expected=case["description"],
                detail="detailed",
            )
        else:  # standard
            verify_output = verify_tool.execute(
                expected=case["description"],
                detail="standard",
            )

        elapsed = time.time() - start
        result.elapsed_seconds = elapsed
        result.raw_output = verify_output[:1000]

        # Parse result
        result.match = _parse_match_from_output(verify_output)
        _safe_print(f"    Verify ({config}): match={result.match} | {elapsed:.1f}s")
        _safe_print(f"    Output: {verify_output[:200]}")

    except Exception as e:
        result.error = str(e)[:500]
        _safe_print(f"    ERROR: {e}")
    finally:
        # Step 4: Close FreeCAD
        try:
            close_tool.execute()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_vlm_report(results: list[VerifyResult]) -> None:
    """Print VLM accuracy report."""
    print()
    print("=" * 70)
    print("  VLM Verification Accuracy Report")
    print("  Model: GLM-4V-Flash / GLM-4V-Plus")
    print("=" * 70)
    print(f"  {'Case':<22} {'Config':<14} {'Match':>8} {'Time':>8}")
    print("  " + "-" * 54)

    # Group by config
    by_config: dict[str, list[VerifyResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)

    for r in results:
        match_str = {True: "MATCH", False: "MISMATCH", None: "ERROR"}[r.match]
        time_str = f"{r.elapsed_seconds:.1f}s"
        print(f"  {r.case_id:<22} {r.config:<14} {match_str:>8} {time_str:>8}")

    print("  " + "-" * 54)

    # Summary by config
    for cfg, cfg_results in by_config.items():
        matches = sum(1 for r in cfg_results if r.match is True)
        total = len(cfg_results)
        pct = 100 * matches // max(total, 1)
        print(f"  {cfg}: {matches}/{total} ({pct}%) correct")

    # Overall
    all_matches = sum(1 for r in results if r.match is True)
    all_total = len(results)
    all_pct = 100 * all_matches // max(all_total, 1)
    print(f"\n  Overall: {all_matches}/{all_total} ({all_pct}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.skipif(not _has_api_key() or not _has_freecad(), reason=skip_reason)
class TestVLMAccuracy:
    """Test cad_verify accuracy against existing models."""

    def _get_root(self) -> Path:
        return Path(__file__).parent.parent

    # --- cantilever_bracket ---

    def test_verify_cantilever_standard(self):
        """Single-angle standard verification — cantilever bracket."""
        result = run_verify_test(VLM_TEST_CASES[0], "standard", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_cantilever_detailed(self):
        """Single-angle detailed verification — cantilever bracket."""
        result = run_verify_test(VLM_TEST_CASES[0], "detailed", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_cantilever_multi_angle(self):
        """Multi-angle verification — cantilever bracket."""
        result = run_verify_test(VLM_TEST_CASES[0], "multi_angle", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    # --- stepped_shaft ---

    def test_verify_shaft_standard(self):
        """Single-angle standard verification — stepped shaft."""
        result = run_verify_test(VLM_TEST_CASES[1], "standard", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_shaft_detailed(self):
        """Single-angle detailed verification — stepped shaft."""
        result = run_verify_test(VLM_TEST_CASES[1], "detailed", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_shaft_multi_angle(self):
        """Multi-angle verification — stepped shaft."""
        result = run_verify_test(VLM_TEST_CASES[1], "multi_angle", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    # --- l_bracket ---

    def test_verify_l_bracket_standard(self):
        """Single-angle standard verification — L bracket."""
        result = run_verify_test(VLM_TEST_CASES[2], "standard", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_l_bracket_detailed(self):
        """Single-angle detailed verification — L bracket."""
        result = run_verify_test(VLM_TEST_CASES[2], "detailed", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"

    def test_verify_l_bracket_multi_angle(self):
        """Multi-angle verification — L bracket."""
        result = run_verify_test(VLM_TEST_CASES[2], "multi_angle", project_root=self._get_root())
        assert result.match is not None, f"Verify failed: {result.error}"


# ---------------------------------------------------------------------------
# Standalone script entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  VLM Verification Accuracy Test")
    print("  Testing cad_verify with 3 configs x 3 models")
    print("=" * 70)

    if not _has_api_key():
        print("ERROR: GLM_API_KEY not configured.")
        sys.exit(1)
    if not _has_freecad():
        print("ERROR: FreeCAD not found.")
        sys.exit(1)

    root = Path(__file__).parent.parent

    # Check all test files exist
    for case in VLM_TEST_CASES:
        fpath = root / case["file"]
        if not fpath.exists():
            print(f"ERROR: {fpath} not found. Run engineering cases first.")
            sys.exit(1)

    configs = ["standard", "detailed", "multi_angle"]
    all_results: list[VerifyResult] = []

    for case in VLM_TEST_CASES:
        print(f"\n--- {case['id']} ---")
        for cfg in configs:
            print(f"\n  Config: {cfg}")
            result = run_verify_test(case, cfg, project_root=root)
            all_results.append(result)

            status = {True: "MATCH", False: "MISMATCH", None: "ERROR"}[result.match]
            print(f"  => {status} ({result.elapsed_seconds:.1f}s)")

            # Brief pause between tests
            time.sleep(2)

    print_vlm_report(all_results)

    # Save JSON
    import json
    report_path = root / "data" / "vlm_accuracy_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {report_path}")
