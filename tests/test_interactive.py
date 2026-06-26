"""Tests for the IterativeSession class (src/lang3d/interactive.py).

These cover the Claude-Code-style edit loop:
- Load assembly from a run folder
- Apply NL edits with deterministic scope classification
- Save in-place (assembly.json overwritten)
- Undo stack
- modifications_diff reporting
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))

from lang3d.interactive import (  # noqa: E402
    IterativeSession,
    assembly_from_dict,
    assembly_to_dict,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_arm() -> Assembly:
    return Assembly(
        name="test_arm",
        description="test arm",
        parts=[
            Part(name="base_plate", category="structural",
                 description="base", dimensions={"length": 100, "width": 100, "height": 5}),
            Part(name="shoulder_link", category="link",
                 description="shoulder", dimensions={"length": 120, "width": 20, "height": 25}),
            Part(name="gripper_finger_left", category="effector",
                 description="l", dimensions={"length": 30, "width": 4, "height": 15}),
            Part(name="gripper_finger_right", category="effector",
                 description="r", dimensions={"length": 30, "width": 4, "height": 15}),
        ],
        joints=[
            Joint(type="fixed", parent="base_plate", child="shoulder_link"),
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_left", offset=(0.0, -2.0, 0.0)),
            Joint(type="prismatic", parent="shoulder_link",
                  child="gripper_finger_right", offset=(0.0, 2.0, 0.0)),
        ],
    )


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    """Create a fake run folder with assembly.json."""
    asm = _make_arm()
    asm_path = tmp_path / "assembly.json"
    asm_path.write_text(
        json.dumps(assembly_to_dict(asm), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# (De)serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_round_trip_preserves_parts_and_joints(self):
        asm = _make_arm()
        d = assembly_to_dict(asm)
        asm2 = assembly_from_dict(d)
        assert len(asm2.parts) == len(asm.parts)
        assert len(asm2.joints) == len(asm.joints)
        # Spot-check a part
        finger = next(p for p in asm2.parts if p.name == "gripper_finger_left")
        assert finger.dimensions["length"] == 30
        # Spot-check a joint
        j = next(j for j in asm2.joints if j.child == "gripper_finger_left")
        assert j.parent == "shoulder_link"
        assert j.offset == (0.0, -2.0, 0.0)

    def test_round_trip_preserves_default_angles(self):
        asm = _make_arm()
        asm.default_angles = {"shoulder_link": 30.0}
        asm2 = assembly_from_dict(assembly_to_dict(asm))
        assert asm2.default_angles == {"shoulder_link": 30.0}


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLoad:
    def test_load_existing_folder(self, run_folder: Path):
        session = IterativeSession(run_folder)
        assert session.assembly.name == "test_arm"
        assert len(session.assembly.parts) == 4

    def test_load_missing_folder_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            IterativeSession(tmp_path / "nonexistent")

    def test_load_missing_assembly_raises(self, tmp_path: Path):
        # tmp_path already exists as an empty dir — IterativeSession
        # should refuse because there's no assembly.json in it.
        with pytest.raises(FileNotFoundError):
            IterativeSession(tmp_path)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_part_scale(self, run_folder: Path):
        session = IterativeSession(run_folder)
        result = session.apply("enlarge gripper_finger_left 2x")
        assert result["scope"] == "part"
        assert result["target"] == "gripper_finger_left"
        assert result["applied"]
        # Verify the actual assembly was modified
        finger = next(p for p in session.assembly.parts
                      if p.name == "gripper_finger_left")
        assert finger.dimensions["length"] == 60.0
        # Verify history was recorded
        assert len(session.history) == 1

    def test_apply_subsystem_scale(self, run_folder: Path):
        session = IterativeSession(run_folder)
        result = session.apply("把夹爪加长50%")
        assert result["scope"] == "subsystem"
        assert result["target"] == "gripper"
        # Both fingers should be in the diff
        changed_names = [c["name"] for c in result["diff"]["parts_changed"]]
        assert "gripper_finger_left" in changed_names
        assert "gripper_finger_right" in changed_names

    def test_apply_with_unknown_intent_falls_back_to_whole(
        self, run_folder: Path,
    ):
        # "paint it red" doesn't match any subsystem/part keyword and isn't
        # "redo" — classifier falls back to scope=whole.  The whole-scope
        # path would normally call the LLM (api_key required); in unit
        # tests we expect that to raise.  Verify the classifier got there.
        from lang3d.agent.modifier import classify_modification
        session = IterativeSession(run_folder)
        req = classify_modification("paint it red", session.assembly)
        assert req.scope == "whole"
        # The actual apply() call would raise RuntimeError for missing
        # api_key; that's the expected behaviour and not what we test here.

    def test_history_grows(self, run_folder: Path):
        session = IterativeSession(run_folder)
        session.apply("enlarge shoulder_link 1.5x")
        session.apply("enlarge base_plate 1.2x")
        assert len(session.history) == 2


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


class TestUndo:
    def test_undo_reverts_last_edit(self, run_folder: Path):
        session = IterativeSession(run_folder)
        original = next(p for p in session.assembly.parts
                        if p.name == "shoulder_link")
        original_length = original.dimensions["length"]
        session.apply("enlarge shoulder_link 2x")
        # Verify it was actually scaled
        scaled = next(p for p in session.assembly.parts
                      if p.name == "shoulder_link")
        assert scaled.dimensions["length"] == original_length * 2
        # Undo
        assert session.undo()
        reverted = next(p for p in session.assembly.parts
                        if p.name == "shoulder_link")
        assert reverted.dimensions["length"] == original_length

    def test_undo_empty_history_returns_false(self, run_folder: Path):
        session = IterativeSession(run_folder)
        assert session.undo() is False

    def test_undo_multiple_times(self, run_folder: Path):
        session = IterativeSession(run_folder)
        session.apply("enlarge shoulder_link 2x")
        session.apply("enlarge shoulder_link 2x")
        assert len(session.history) == 2
        assert session.undo()
        assert len(session.history) == 1
        assert session.undo()
        assert len(session.history) == 0
        assert session.undo() is False  # nothing left


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_in_place_overwrites(self, run_folder: Path):
        session = IterativeSession(run_folder)
        session.apply("enlarge shoulder_link 2x")
        session.save()
        # Re-read from disk
        asm_path = run_folder / "assembly.json"
        saved = json.loads(asm_path.read_text(encoding="utf-8"))
        shoulder = next(p for p in saved["parts"]
                        if p["name"] == "shoulder_link")
        assert shoulder["dimensions"]["length"] == 240.0

    def test_save_as_creates_new_folder(self, run_folder: Path, tmp_path: Path):
        session = IterativeSession(run_folder)
        session.apply("enlarge shoulder_link 2x")
        new_folder = tmp_path / "new_run"
        session.save(new_folder)
        assert (new_folder / "assembly.json").exists()
        # Original folder should be untouched on disk (we only changed
        # session.folder after the save)
        # Reload from original folder to confirm
        original_asm = json.loads(
            (run_folder / "assembly.json").read_text(encoding="utf-8"))
        original_shoulder = next(p for p in original_asm["parts"]
                                 if p["name"] == "shoulder_link")
        assert original_shoulder["dimensions"]["length"] == 120.0
        # New folder has the modified assembly
        new_asm = json.loads(
            (new_folder / "assembly.json").read_text(encoding="utf-8"))
        new_shoulder = next(p for p in new_asm["parts"]
                            if p["name"] == "shoulder_link")
        assert new_shoulder["dimensions"]["length"] == 240.0


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_includes_part_joint_counts(self, run_folder: Path):
        session = IterativeSession(run_folder)
        text = session.describe()
        assert "Parts:" in text
        assert "Joints:" in text

    def test_describe_includes_recent_history(self, run_folder: Path):
        session = IterativeSession(run_folder)
        session.apply("enlarge shoulder_link 2x")
        text = session.describe()
        assert "History:" in text
        assert "1 edits" in text


# ---------------------------------------------------------------------------
# Verify (solver + collision check, no LLM)
# ---------------------------------------------------------------------------


class TestVerify:
    def test_verify_returns_structured_report(self, run_folder: Path):
        session = IterativeSession(run_folder)
        # Disable collision check (trimesh may not be installed)
        report = session.verify(with_collision=False)
        assert "ok" in report
        assert "checks" in report
        check_names = [c["name"] for c in report["checks"]]
        assert "solver" in check_names
