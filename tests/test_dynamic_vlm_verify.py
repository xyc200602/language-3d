"""Tests for dynamic (motion-based) VLM verification.

Covers the extract_motion_key_frames + verify_motion pipeline that feeds
MuJoCo simulation frames to GLM-4.6V for motion-behaviour judgement.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Shared fixtures from test_sim_mujoco — reuse the example URDF discovery.
from test_sim_mujoco import _EXAMPLE_URDF, _mujoco_available


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mujoco_available() or _EXAMPLE_URDF is None or not _EXAMPLE_URDF.exists(),
    reason="MuJoCo not installed or no example URDF available",
)
class TestExtractMotionKeyFrames:
    def test_returns_three_labeled_frames(self) -> None:
        """extract_motion_key_frames produces initial/mid_sweep/extreme frames."""
        from lang3d.tools.sim_mujoco import extract_motion_key_frames
        frames = extract_motion_key_frames(str(_EXAMPLE_URDF))
        assert len(frames) == 3
        labels = [f["label"] for f in frames]
        assert labels == ["initial", "mid_sweep", "extreme"]

    def test_frames_are_base64_png(self) -> None:
        """Each frame's image is a valid data:image/png;base64 URI."""
        from lang3d.tools.sim_mujoco import extract_motion_key_frames
        import base64
        import io
        from PIL import Image
        frames = extract_motion_key_frames(str(_EXAMPLE_URDF))
        for f in frames:
            assert f["image"].startswith("data:image/png;base64,")
            b64 = f["image"].split(",", 1)[1]
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            assert img.size[0] > 0 and img.size[1] > 0

    def test_nonexistent_urdf_returns_empty(self) -> None:
        """A missing URDF returns an empty list, not a crash."""
        from lang3d.tools.sim_mujoco import extract_motion_key_frames
        frames = extract_motion_key_frames("/nonexistent/robot.urdf")
        assert frames == []


# ---------------------------------------------------------------------------
# VLM verification (mocked — real API calls are in the 'api' marker suite)
# ---------------------------------------------------------------------------

class TestVerifyMotion:
    """Mocked tests for the GLM-4.6V call — no real API key needed."""

    _SAMPLE_FRAMES = [
        {"label": "initial", "description": "rest pose",
         "image": "data:image/png;base64,iVBORw0KGgo="},
        {"label": "mid_sweep", "description": "mid sweep",
         "image": "data:image/png;base64,iVBORw0KGgo="},
        {"label": "extreme", "description": "extreme",
         "image": "data:image/png;base64,iVBORw0KGgo="},
    ]

    def test_empty_frames_returns_pass(self) -> None:
        """No frames → passed=True (nothing to check)."""
        from lang3d.tools.assembly_gen.dynamic_vlm_verify import verify_motion
        result = verify_motion([], api_key="fake")
        assert result["passed"] is True
        assert result["problems"] == []

    @patch("openai.OpenAI")
    def test_pass_response_parsed(self, mock_openai_cls) -> None:
        """A clean 'passed' JSON response is parsed correctly."""
        from lang3d.tools.assembly_gen.dynamic_vlm_verify import verify_motion
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"passed": true, "problems": [], "fix_hints": []}'
        mock_client.chat.completions.create.return_value = mock_resp

        result = verify_motion(self._SAMPLE_FRAMES, api_key="fake")
        assert result["passed"] is True
        assert result["problems"] == []

    @patch("openai.OpenAI")
    def test_problems_parsed_from_json_fence(self, mock_openai_cls) -> None:
        """JSON wrapped in ```json fences is correctly extracted."""
        from lang3d.tools.assembly_gen.dynamic_vlm_verify import verify_motion
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            '```json\n{"passed": false, "problems": ["shoulder collides at 60°"], '
            '"fix_hints": ["clamp shoulder_pitch upper to 45"]}\n```'
        )
        mock_client.chat.completions.create.return_value = mock_resp

        result = verify_motion(self._SAMPLE_FRAMES, api_key="fake")
        assert result["passed"] is False
        assert len(result["problems"]) == 1
        assert "60°" in result["problems"][0]
        assert len(result["fix_hints"]) == 1

    @patch("openai.OpenAI")
    def test_api_failure_returns_fail(self, mock_openai_cls) -> None:
        """An API exception returns passed=False with the error in problems."""
        from lang3d.tools.assembly_gen.dynamic_vlm_verify import verify_motion
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("timeout")

        result = verify_motion(self._SAMPLE_FRAMES, api_key="fake")
        assert result["passed"] is False
        assert any("timeout" in p for p in result["problems"])
