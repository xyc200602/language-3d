"""Unit tests for :mod:`lang3d.experience.store`.

Mirrors the structure of ``test_assembly_templates.py`` — pure-Python,
no LLM/FreeCAD/MuJoCo, runs in milliseconds.  Covers the four behaviours the
external audit (H2) cares about:

1. **Store-after** — a verified-good case is persisted and survives reload.
2. **Retrieve-before** — a similar query retrieves it; dissimilar doesn't.
3. **Self-evolving** — repeated retrieval promotes a case (retrieval_hits),
   and pruning keeps the most-retrieved cases.
4. **Honesty** — failure modes (corrupt JSON, empty store) don't crash and
   don't pretend to have results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lang3d.experience.store import (
    MAX_ENTRIES_PER_CATEGORY,
    CaseRecord,
    ExperienceStore,
    WEIGHT_CATEGORY,
    WEIGHT_DOF_EXACT,
    WEIGHT_KEYWORD,
    _extract_query_dof,
    reset_store_for_tests,
    score_case,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ExperienceStore:
    """Fresh isolated store per test — no leakage between tests."""
    return ExperienceStore(tmp_path / "exp")


def _arm_case(
    description: str = "4自由度机械臂",
    *,
    dof: int = 4,
    category: str = "fixed_arm",
    hits: int = 0,
) -> CaseRecord:
    """Minimal but realistic verified-good arm case."""
    return CaseRecord(
        description=description,
        robot_category=category,
        dof=dof,
        assembly_name="4dof_robotic_arm",
        part_count=12,
        joint_types={"revolute": 4, "fixed": 8},
        default_angles={"joint_1": 0.0, "joint_2": -45.0},
        rounds_taken=1,
        run_dir="data/runs/4dof_arm/20260702_120000",
        retrieval_hits=hits,
    )


def _wheeled_case(hits: int = 0) -> CaseRecord:
    return CaseRecord(
        description="四轮差速移动底盘带双臂机器人",
        robot_category="wheeled_arm",
        dof=8,
        assembly_name="4wheel_dual_arm_robot",
        part_count=33,
        joint_types={"revolute": 8, "fixed": 25},
        default_angles={},
        rounds_taken=2,
        run_dir="data/runs/4wheel_dual_arm/20260702_140000",
        retrieval_hits=hits,
    )


def _wheeled_case_variant(hits: int = 0) -> CaseRecord:
    """A second, distinct wheeled-arm prompt — used to test multi-case stats
    without tripping the description-based dedup."""
    return CaseRecord(
        description="四轮移动机器人带单机械臂",
        robot_category="wheeled_arm",
        dof=5,
        assembly_name="4wheel_single_arm_robot",
        part_count=24,
        joint_types={"revolute": 5, "fixed": 19},
        default_angles={},
        rounds_taken=1,
        run_dir="data/runs/4wheel_arm/20260702_150000",
        retrieval_hits=hits,
    )


# ---------------------------------------------------------------------------
# DOF extraction (pure helper)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, expected",
    [
        ("4 dof arm", 4),
        ("4dof arm", 4),
        ("4自由度机械臂", 4),
        ("4 自由度", 4),
        ("7dof_arm", 7),
        ("a robot arm", None),
        ("", None),
        ("dof arm", None),  # no number prefix
    ],
)
def test_extract_query_dof(query: str, expected: int | None) -> None:
    assert _extract_query_dof(query) == expected


# ---------------------------------------------------------------------------
# score_case (pure scoring — no I/O)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_score_case_exact_description_match() -> None:
    """Substring match on the description earns the keyword bonus."""
    case = _arm_case("4自由度机械臂")
    s = score_case(case, "4自由度机械臂", "fixed_arm", query_dof=4)
    # keyword(3) + category(5) + dof-exact(2) + tokens(<=4*1.5)
    assert s >= WEIGHT_KEYWORD + WEIGHT_CATEGORY + WEIGHT_DOF_EXACT


@pytest.mark.unit
def test_score_case_same_category_boosts() -> None:
    """Same-category match outscores cross-category for an ambiguous query."""
    case = _arm_case()
    same = score_case(case, "arm", "fixed_arm", query_dof=None)
    cross = score_case(case, "arm", "wheeled", query_dof=None)
    assert same > cross
    assert same - cross == pytest.approx(WEIGHT_CATEGORY)


@pytest.mark.unit
def test_score_case_dof_proximity() -> None:
    """±1 DOF earns partial credit; >1 DOF off earns nothing on the DOF axis."""
    case = _arm_case(dof=4)
    exact = score_case(case, "arm", "", query_dof=4)
    near = score_case(case, "arm", "", query_dof=5)
    far = score_case(case, "arm", "", query_dof=7)
    assert exact > near > far
    assert far < exact  # far still gets keyword tokens but no dof bonus


@pytest.mark.unit
def test_score_case_zero_when_nothing_matches() -> None:
    """A totally unrelated query scores 0 (no keyword, no category, no dof)."""
    case = _arm_case("4自由度机械臂")
    # Query tokens ("banana", "smoothie") share nothing with the case.
    s = score_case(case, "banana smoothie", "wheeled", query_dof=99)
    assert s == 0.0


# ---------------------------------------------------------------------------
# Store — record + retrieve round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_then_retrieve_finds_similar(store: ExperienceStore) -> None:
    """The canonical store-after → retrieve-before round trip."""
    store.record(_arm_case("4自由度机械臂"))

    hits = store.retrieve("4自由度机械臂", "fixed_arm")
    assert len(hits) == 1
    assert hits[0].description == "4自由度机械臂"
    assert hits[0].dof == 4


@pytest.mark.unit
def test_retrieve_returns_empty_for_unrelated_query(
    store: ExperienceStore,
) -> None:
    """An empty store returns [].  A populated store with no match returns []."""
    assert store.retrieve("anything", "fixed_arm") == []

    store.record(_arm_case("4自由度机械臂"))
    # Totally unrelated — should not surface.
    hits = store.retrieve("banana smoothie", "wheeled")
    assert hits == []


@pytest.mark.unit
def test_record_dedups_on_description(store: ExperienceStore) -> None:
    """Re-recording the same prompt updates rather than stacks."""
    store.record(_arm_case("4自由度机械臂"))
    store.record(_arm_case("4自由度机械臂", dof=5))  # updated DOF

    all_cases = store.all_cases()
    assert len(all_cases) == 1
    assert all_cases[0].dof == 5


@pytest.mark.unit
def test_record_preserves_retrieval_hits_across_dedup(
    store: ExperienceStore,
) -> None:
    """When a case is re-recorded, cumulative popularity is preserved."""
    store.record(_arm_case(hits=0))
    # Simulate two retrievals bumping the counter.
    store.retrieve("4自由度机械臂", "fixed_arm")
    store.retrieve("4自由度机械臂", "fixed_arm")

    cases = store.all_cases()
    assert cases[0].retrieval_hits >= 2

    # Now re-record with hits=0 in the new payload — old hits must survive.
    store.record(_arm_case(hits=0))
    cases = store.all_cases()
    assert cases[0].retrieval_hits >= 2


# ---------------------------------------------------------------------------
# Self-evolving — retrieval popularity promotes survival
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retrieval_bumps_hit_counter(store: ExperienceStore) -> None:
    """Each retrieve() that returns a case increments its popularity."""
    store.record(_arm_case())
    assert store.all_cases()[0].retrieval_hits == 0

    store.retrieve("4自由度机械臂", "fixed_arm")
    assert store.all_cases()[0].retrieval_hits == 1

    store.retrieve("4自由度机械臂", "fixed_arm")
    assert store.all_cases()[0].retrieval_hits == 2


@pytest.mark.unit
def test_pruning_keeps_most_retrieved(store: ExperienceStore) -> None:
    """Past MAX_ENTRIES_PER_CATEGORY, least-popular cases are evicted."""
    # Fill the category to the cap.
    for i in range(MAX_ENTRIES_PER_CATEGORY):
        store.record(_arm_case(f"arm case {i}"))

    assert len(store.all_cases()) == MAX_ENTRIES_PER_CATEGORY

    # Retrieve one specific case many times to make it popular.
    popular_desc = "arm case 5"
    for _ in range(5):
        store.retrieve(popular_desc, "fixed_arm")

    # Add one more — should trigger pruning.
    store.record(_arm_case("overflow arm case"))

    cases = store.all_cases()
    assert len(cases) == MAX_ENTRIES_PER_CATEGORY  # cap maintained
    # The popular case must have survived.
    assert any(c.description == popular_desc for c in cases)


# ---------------------------------------------------------------------------
# Robustness — corrupt JSON, empty queries, multi-category
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_corrupt_json_does_not_crash(store: ExperienceStore, tmp_path: Path) -> None:
    """A corrupted category file is treated as empty, not fatal (AGENTS §1.1)."""
    cat_path = store.store_dir / "fixed_arm.json"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat_path.write_text("{not valid json", encoding="utf-8")

    # Should not raise — should return [].
    assert store.retrieve("4自由度机械臂", "fixed_arm") == []
    # And recording should still work (overwrite the corrupt file).
    store.record(_arm_case())
    assert len(store.all_cases()) == 1


@pytest.mark.unit
def test_multi_category_isolation(store: ExperienceStore) -> None:
    """A wheeled query shouldn't surface arm cases (and vice versa)."""
    store.record(_arm_case())
    store.record(_wheeled_case())

    arm_hits = store.retrieve("机械臂", "fixed_arm")
    wheeled_hits = store.retrieve("底盘", "wheeled_arm")

    assert all(c.robot_category == "fixed_arm" for c in arm_hits)
    assert all(c.robot_category == "wheeled_arm" for c in wheeled_hits)


@pytest.mark.unit
def test_retrieve_top_k_limits_results(store: ExperienceStore) -> None:
    """k=2 caps the result list even when more cases match."""
    for i in range(5):
        store.record(_arm_case(f"4自由度机械臂 variant {i}"))

    hits = store.retrieve("4自由度机械臂", "fixed_arm", k=2)
    assert len(hits) <= 2


@pytest.mark.unit
def test_stats_reports_per_category_counts(store: ExperienceStore) -> None:
    store.record(_arm_case())
    store.record(_wheeled_case())
    # Second wheeled case with a different description — dedup keys on the
    # prompt, so a distinct prompt is treated as a distinct case.
    store.record(_wheeled_case_variant())

    stats = store.stats()
    assert stats.get("fixed_arm") == 1
    assert stats.get("wheeled_arm") == 2


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_store_singleton(tmp_path: Path) -> None:
    """get_store returns the same instance across calls (singleton)."""
    from lang3d.experience.store import get_store

    s1 = reset_store_for_tests(tmp_path / "g1")
    s2 = get_store()
    assert s1 is s2


@pytest.mark.unit
def test_persistence_across_instances(tmp_path: Path) -> None:
    """A new ExperienceStore at the same dir sees previously-written cases."""
    s1 = ExperienceStore(tmp_path / "persist")
    s1.record(_arm_case())

    s2 = ExperienceStore(tmp_path / "persist")
    hits = s2.retrieve("4自由度机械臂", "fixed_arm")
    assert len(hits) == 1
    assert hits[0].description == "4自由度机械臂"
