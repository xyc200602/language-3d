"""Shared test fixtures for Language-3D tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Marker registry
#
# The project's tests span four distinct runtime profiles.  Each profile
# is exposed as a pytest marker so callers can run a fast sanity subset
# without dragging in heavy / unavailable external deps:
#
#   pytest -m unit                       # fast pure-Python tests (~30s)
#   pytest -m "unit or integration"      # everything runnable locally
#   pytest -m "not e2e and not api"      # skip anything that needs LLM
#   pytest -m e2e                        # only the full NL→assembly pipeline
#
# Auto-classification lives in ``pytest_collection_modifyitems`` below —
# no per-file edits are required.  A test may carry multiple markers
# (e.g. e2e + api + integration); ``-m`` filters are logical expressions.
# ---------------------------------------------------------------------------
_MARKER_DESCRIPTIONS = {
    "unit": (
        "Pure-Python tests with no external deps (default). "
        "Runs in seconds; safe to execute on any machine."
    ),
    "integration": (
        "Tests that load real geometry / spawn FreeCAD / use trimesh+fcl / "
        "use MuJoCo.  Requires the corresponding package installed."
    ),
    "e2e": (
        "End-to-end pipeline tests (NL → assembly → export).  Typically "
        "slow; often requires GLM API key as well."
    ),
    "api": (
        "Tests that call out to a remote LLM (GLM API).  Skipped "
        "automatically when GLM_API_KEY is not set."
    ),
    "gui": (
        "Tests that require a desktop / display (PyAutoGUI, FreeCAD GUI, "
        "VLM screenshot capture).  Skip on headless CI."
    ),
    "freecad": (
        "Alias kept for backward compatibility — test_part_validator.py:404 "
        "uses @pytest.mark.freecad directly.  Treated as 'integration'."
    ),
}


def pytest_configure(config: pytest.Config) -> None:
    """Register all custom markers so pytest doesn't warn about them."""
    for name, desc in _MARKER_DESCRIPTIONS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


# ---------------------------------------------------------------------------
# Auto-classification
# ---------------------------------------------------------------------------


# Keyword → marker mapping.  The check is substring-based against the
# test file's source text, so it survives renames better than import
# inspection.  Tuned from the audit run on 2026-06-18.
_E2E_KEYWORDS = (
    "generate_assembly_with_vlm_loop",
    "export_engineering_package",
    "run_e2e_case",
)
_API_KEYWORDS = (
    "GLM_API_KEY",
    "GLMBackend",
    "generate_assembly_from_nl",
    "os.environ.get(\"GLM",
    "os.environ.get('GLM",
)
_GUI_KEYWORDS = (
    "pyautogui",
    "gui_action",
    "fc_menu",
    "fc_open_gui",
    "vlm_locate",
)
_INTEGRATION_KEYWORDS = (
    "import freecad",  # case-sensitive (the actual import)
    "from freecad",
    "FreeCAD",  # any reference
    "python-fcl",
    "from fcl",
    "import trimesh",
    "from trimesh",
    "import mujoco",
    "from mujoco",
    "SimMujocoTool",
    "fc_batch",
    "freecad-script",
    "MeshCollisionChecker",
)


def _classify_test_file(path: Path) -> set[str]:
    """Return the set of markers a test file should have.

    Reads the source text once per file and applies keyword heuristics.
    The result is cached in a module-level dict so collection of 3.9k
    tests doesn't re-read the same 134 files.
    """
    cache = _classify_test_file._cache  # type: ignore[attr-defined]
    key = str(path)
    if key in cache:
        return cache[key]

    markers: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = ""

    name_lower = path.name.lower()
    # Filename-based e2e detection (covers test_*_e2e.py and test_e2e_*.py)
    if "_e2e" in name_lower or name_lower.startswith("e2e_"):
        markers.add("e2e")

    # Source-based detection
    if any(k in text for k in _E2E_KEYWORDS):
        markers.add("e2e")
    if any(k in text for k in _API_KEYWORDS):
        markers.add("api")
    if any(k in text for k in _GUI_KEYWORDS):
        markers.add("gui")
    if any(k in text for k in _INTEGRATION_KEYWORDS):
        markers.add("integration")

    # 'unit' is the fallback — applied ONLY if nothing else matched,
    # so `pytest -m unit` doesn't accidentally include heavy tests.
    if not markers:
        markers.add("unit")

    cache[key] = markers
    return markers


# Module-level cache (initialised on first call)
_classify_test_file._cache = {}  # type: ignore[attr-defined]


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark each collected test based on its file's content.

    Runs after collection, before execution.  Cheap: 134 files read once,
    results cached.  Adds 1+ markers to every test item.
    """
    for item in items:
        path = Path(str(item.fspath))
        markers = _classify_test_file(path)
        for marker_name in markers:
            item.add_marker(getattr(pytest.mark, marker_name))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace():
    """Provide a temporary workspace directory."""
    with tempfile.TemporaryDirectory(prefix="lang3d_test_") as d:
        yield Path(d)


@pytest.fixture
def mock_router():
    """Provide a mock ModelRouter."""
    router = MagicMock()
    router.chat.return_value = MagicMock(
        content="mock response",
        tool_calls=[],
        usage={"input_tokens": 0, "output_tokens": 0},
    )
    return router


@pytest.fixture
def mock_tools():
    """Provide a mock ToolRegistry."""
    registry = MagicMock()
    registry.get_relevant_definitions.return_value = []
    registry.execute.return_value = "mock tool result"
    return registry
