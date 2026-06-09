"""End-to-end reliability test suite — 8 gradient cases.

Measures the real success rate of the main pipeline:
  natural language → Agent → fc_batch → FCStd file → (optional) cad_verify

Usage:
  # Run all 8 cases and print structured report
  python tests/test_e2e_reliability.py

  # Run via pytest (each case individually)
  pytest tests/test_e2e_reliability.py -v -m e2e

  # Run only Level 1-2
  pytest tests/test_e2e_reliability.py -v -m "e2e and (level1 or level2)"
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    """Check if GLM API key is available."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    return bool(os.environ.get("GLM_API_KEY"))


def _has_freecad() -> bool:
    """Check if FreeCAD is installed."""
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
# Case definitions — 8 gradient cases
# ---------------------------------------------------------------------------

E2E_CASES: list[dict[str, Any]] = [
    # Level 1: 基础体素 (不需要布尔运算)
    {
        "id": "cube",
        "name": "正方体",
        "task": "创建一个 30x30x30mm 的正方体，保存为 cube.FCStd",
        "difficulty": 1,
        "expected_files": ["cube.FCStd"],
        "expect_verify": False,
        "marker": "level1",
    },
    {
        "id": "cylinder",
        "name": "圆柱体",
        "task": "创建一个半径10mm、高25mm的圆柱体，保存为 cylinder.FCStd 并导出 STL",
        "difficulty": 1,
        "expected_files": ["cylinder.FCStd", "cylinder.stl"],
        "expect_verify": False,
        "marker": "level1",
    },

    # Level 2: 布尔运算
    {
        "id": "plate_with_hole",
        "name": "带孔平板",
        "task": "创建一个 50x50x10mm 的方形板，中心打一个直径10mm的通孔，保存为 plate_hole.FCStd",
        "difficulty": 2,
        "expected_files": ["plate_hole.FCStd"],
        "expect_verify": True,
        "marker": "level2",
    },
    {
        "id": "l_bracket",
        "name": "L型支架",
        "task": (
            "创建一个 L 型支架：\n"
            "1. 底板：80x60x8mm\n"
            "2. 竖板：8x60x40mm，与底板一端垂直相连\n"
            "3. 底板上有2个直径6mm的安装孔，距两端15mm\n"
            "保存为 l_bracket.FCStd，建模后用 cad_verify 验证"
        ),
        "difficulty": 2,
        "expected_files": ["l_bracket.FCStd"],
        "expect_verify": True,
        "marker": "level2",
    },

    # Level 3: 复杂特征
    {
        "id": "cantilever_bracket",
        "name": "悬臂支架",
        "task": (
            "创建悬臂支架：底板120x80x10mm，竖板10x80x60mm垂直相连，"
            "底板4个直径8mm安装孔（四角距边15mm），竖板1个直径15mm轴承孔。"
            "保存为 cantilever_bracket.FCStd，用 cad_verify 验证"
        ),
        "difficulty": 3,
        "expected_files": ["cantilever_bracket.FCStd"],
        "expect_verify": True,
        "marker": "level3",
    },
    {
        "id": "stepped_shaft",
        "name": "阶梯轴",
        "task": (
            "创建阶梯轴：第一段直径20mm长30mm，第二段直径15mm长25mm，"
            "第三段直径10mm长20mm，同轴连接。第二段有键槽4x2x12mm。"
            "保存为 stepped_shaft.FCStd，用 cad_verify 验证"
        ),
        "difficulty": 3,
        "expected_files": ["stepped_shaft.FCStd"],
        "expect_verify": True,
        "marker": "level3",
    },

    # Level 4: 多特征组合
    {
        "id": "flange_coupling",
        "name": "法兰联轴器",
        "task": (
            "创建法兰联轴器：中心圆柱直径25mm高30mm，"
            "两端各一个法兰盘外径60mm厚8mm，"
            "每个法兰盘4个直径8mm螺栓孔均匀分布在直径45mm圆周上，"
            "中心有直径12mm通孔。"
            "保存为 flange.FCStd，用 cad_verify 验证"
        ),
        "difficulty": 4,
        "expected_files": ["flange.FCStd"],
        "expect_verify": True,
        "marker": "level4",
    },
    {
        "id": "bearing_block",
        "name": "轴承座",
        "task": (
            "创建轴承座：底座100x60x15mm，中间有U型座孔直径40mm深35mm，"
            "底座两侧各2个直径10mm安装孔（距边15mm），"
            "座孔两侧有M8螺栓孔用于轴承盖。"
            "保存为 bearing_block.FCStd，用 cad_verify 验证"
        ),
        "difficulty": 4,
        "expected_files": ["bearing_block.FCStd"],
        "expect_verify": True,
        "marker": "level4",
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    case_name: str
    difficulty: int
    success: bool = False                 # Final: files created (+ verify if expected)
    file_created: bool = False            # All expected files exist
    verify_result: str | None = None      # "MATCH" / "MISMATCH" / "ERROR" / None
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args_summary}]
    fc_batch_calls: int = 0
    verify_calls: int = 0
    fix_attempts: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None
    fcstd_size: int = 0
    extra_info: str = ""


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _safe_print(text: str, max_len: int = 300) -> None:
    """Print text safely on Windows GBK terminals."""
    try:
        print(text[:max_len])
    except UnicodeEncodeError:
        safe = text[:max_len].encode("gbk", errors="replace").decode("gbk")
        print(safe)


def run_single_case(case: dict[str, Any], tmp_path: str | Path) -> CaseResult:
    """Run a single e2e case, returning structured result."""
    from lang3d.agent.core import Agent
    from lang3d.config import load_config

    result = CaseResult(
        case_id=case["id"],
        case_name=case["name"],
        difficulty=case["difficulty"],
    )

    config = load_config()
    config.agent.workspace = str(tmp_path)
    config.agent.max_turns = 15

    agent = Agent(config)
    agent.state.workspace = Path(tmp_path)

    tool_log: list[dict] = []

    def on_tool_call(name: str, args: dict) -> None:
        # Summarize args to avoid huge logs
        args_summary = {k: (str(v)[:80] + "..." if len(str(v)) > 80 else str(v))
                        for k, v in args.items()}
        tool_log.append({"name": name, "args": args_summary})
        _safe_print(f"    [CALL] {name}({list(args.keys())})")

    def on_tool_result(name: str, tool_result: str) -> None:
        short = tool_result[:150] + "..." if len(tool_result) > 150 else tool_result
        _safe_print(f"    [RESULT] {name}: {short}")

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)

    start = time.time()
    try:
        agent_result = agent.run_task(case["task"], use_planning=False)
        result.extra_info = agent_result[:500] if agent_result else ""
    except Exception as e:
        result.error = str(e)[:500]
        _safe_print(f"    [ERROR] {e}")
    elapsed = time.time() - start
    result.elapsed_seconds = elapsed

    # Record tool calls
    result.tool_calls = tool_log
    result.fc_batch_calls = sum(1 for t in tool_log if t["name"] == "fc_batch")
    result.verify_calls = sum(1 for t in tool_log if t["name"] == "cad_verify")

    # Count fix attempts: fc_batch calls after the first cad_verify
    first_verify_idx = next(
        (i for i, t in enumerate(tool_log) if t["name"] == "cad_verify"),
        len(tool_log),
    )
    result.fix_attempts = sum(
        1 for t in tool_log[first_verify_idx + 1:] if t["name"] == "fc_batch"
    )

    # Check file creation
    tmp = Path(tmp_path)
    all_files_exist = True
    for fname in case["expected_files"]:
        fpath = tmp / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            all_files_exist = False
            _safe_print(f"    [MISSING] {fname}")
        else:
            _safe_print(f"    [OK] {fname} ({fpath.stat().st_size:,} bytes)")
            if fname.endswith(".FCStd"):
                result.fcstd_size = fpath.stat().st_size
    result.file_created = all_files_exist

    # Check verification result from tool results
    for t in tool_log:
        if t["name"] == "cad_verify":
            args_str = str(t.get("args", {}))
            # Will be set from on_tool_result; check extra_info
            break

    # Parse verify result from agent output
    if result.verify_calls > 0:
        output_lower = result.extra_info.lower()
        if '"match": true' in output_lower or "'match': true" in output_lower or "match=true" in output_lower:
            result.verify_result = "MATCH"
        elif '"match": false' in output_lower or "'match': false" in output_lower or "match=false" in output_lower:
            result.verify_result = "MISMATCH"
        else:
            # If verify was called but we can't parse, check error
            result.verify_result = "UNKNOWN"

    # Overall success: file created (+ verify pass if expected)
    if result.file_created:
        if case["expect_verify"]:
            result.success = result.verify_result == "MATCH"
        else:
            result.success = True

    if result.error and not result.file_created:
        result.success = False

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: list[CaseResult]) -> None:
    """Print a structured reliability report."""
    print()
    print("=" * 76)
    print("  Language-3D E2E Reliability Report")
    print("  Model: GLM-5.1 + GLM-4V-Flash")
    print("  FreeCAD: 1.1.1")
    print("=" * 76)
    print(f"  {'Case':<22} {'Level':>5} {'Success':>8} {'Verify':>10} {'Tools':>6} {'Fixes':>6} {'Time':>8}")
    print("  " + "-" * 68)

    success_count = 0
    verify_total = 0
    verify_pass = 0
    total_time = 0.0

    for r in results:
        success_str = "YES" if r.success else "NO"
        verify_str = r.verify_result or "-"
        if r.verify_result == "UNKNOWN":
            verify_str = "UNKNOWN (!)"
        if r.error and not r.success:
            verify_str = f"{verify_str} (ERR)"
        time_str = f"{r.elapsed_seconds:.1f}s"

        print(
            f"  {r.case_id:<22} {r.difficulty:>5} {success_str:>8} "
            f"{verify_str:>10} {len(r.tool_calls):>6} "
            f"{r.fix_attempts:>6} {time_str:>8}"
        )

        if r.success:
            success_count += 1
        if r.verify_result:
            verify_total += 1
            if r.verify_result == "MATCH":
                verify_pass += 1
        total_time += r.elapsed_seconds

    print("  " + "-" * 68)
    overall_pct = f"{success_count}/{len(results)} ({100 * success_count // max(len(results), 1)}%)"
    verify_pct = (
        f"{verify_pass}/{verify_total} ({100 * verify_pass // max(verify_total, 1)}%)"
        if verify_total > 0 else "-"
    )
    avg_time = f"{total_time / max(len(results), 1):.1f}s"
    print(f"  Overall: {overall_pct} success, {verify_pct} verify pass")
    print(f"  Avg time: {avg_time}")
    print("=" * 76)


def save_results_json(results: list[CaseResult], path: str | Path) -> None:
    """Save results as JSON for later analysis."""
    data = []
    for r in results:
        d = asdict(r)
        data.append(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {path}")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_cases(
    cases: list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
) -> list[CaseResult]:
    """Run all e2e cases and return results."""
    if cases is None:
        cases = E2E_CASES
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "e2e_runs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for i, case in enumerate(cases):
        case_dir = output_dir / f"{i:02d}_{case['id']}"
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i + 1}/{len(cases)}] {case['name']} (Level {case['difficulty']})")
        _safe_print(f"  Task: {case['task'][:200]}")
        print(f"  Output: {case_dir}")

        result = run_single_case(case, case_dir)
        results.append(result)

        status = "PASS" if result.success else "FAIL"
        print(f"  => {status} | files={result.file_created} | verify={result.verify_result} | "
              f"time={result.elapsed_seconds:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------

def _make_case_id(case: dict[str, Any]) -> str:
    return f"L{case['difficulty']}_{case['id']}"


@pytest.mark.e2e
@pytest.mark.skipif(not _has_api_key() or not _has_freecad(), reason=skip_reason)
class TestE2EReliability:
    """Run each e2e case as an individual pytest test."""

    def test_cube(self, tmp_path):
        """Level 1: 正方体"""
        result = run_single_case(E2E_CASES[0], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[0]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_cylinder(self, tmp_path):
        """Level 1: 圆柱体"""
        result = run_single_case(E2E_CASES[1], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[1]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_plate_with_hole(self, tmp_path):
        """Level 2: 带孔平板"""
        result = run_single_case(E2E_CASES[2], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[2]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_l_bracket(self, tmp_path):
        """Level 2: L型支架"""
        result = run_single_case(E2E_CASES[3], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[3]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_cantilever_bracket(self, tmp_path):
        """Level 3: 悬臂支架"""
        result = run_single_case(E2E_CASES[4], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[4]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_stepped_shaft(self, tmp_path):
        """Level 3: 阶梯轴"""
        result = run_single_case(E2E_CASES[5], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[5]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_flange_coupling(self, tmp_path):
        """Level 4: 法兰联轴器"""
        result = run_single_case(E2E_CASES[6], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[6]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )

    def test_bearing_block(self, tmp_path):
        """Level 4: 轴承座"""
        result = run_single_case(E2E_CASES[7], tmp_path)
        assert result.file_created, f"Files not created: {result.error}"
        if E2E_CASES[7]["expect_verify"]:
            assert result.verify_result == "MATCH", (
                f"VLM verify failed: {result.verify_result} | extra: {result.extra_info[:300]}"
            )


# ---------------------------------------------------------------------------
# Standalone script entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("  Language-3D E2E Reliability Suite")
    print("  Running 8 gradient cases (Level 1 → Level 4)")
    print("=" * 76)

    if not _has_api_key():
        print("ERROR: GLM_API_KEY not configured. Set it in .env or environment.")
        sys.exit(1)
    if not _has_freecad():
        print("ERROR: FreeCAD not found. Install FreeCAD 1.1+ first.")
        sys.exit(1)

    # Optionally filter by level from command line
    levels = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--level"):
            levels.add(int(arg.split("=")[1]) if "=" in arg else int(arg))

    cases = E2E_CASES
    if levels:
        cases = [c for c in E2E_CASES if c["difficulty"] in levels]
        print(f"  Filtered to Level(s): {sorted(levels)}")

    results = run_all_cases(cases)
    print_report(results)

    # Save JSON report
    report_path = Path(__file__).parent.parent / "data" / "e2e_report.json"
    save_results_json(results, report_path)

    # Exit code based on success rate
    passed = sum(1 for r in results if r.success)
    total = len(results)
    if passed == total:
        print(f"\n  ALL {total} CASES PASSED")
        sys.exit(0)
    else:
        print(f"\n  {passed}/{total} CASES PASSED ({total - passed} FAILED)")
        sys.exit(1)
