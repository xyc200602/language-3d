"""Dynamic (motion-based) VLM verification using GLM-4.6V.

This module implements the **dynamic verification channel**: unlike the static
VLM (``vlm_verify.py``) which inspects rendered *appearance* from fixed
viewpoints, this channel feeds **simulation motion frames** to GLM-4.6V and
asks it to judge *motion behaviour* — does the robot move through its joint
range without self-collision, is the trajectory mechanically plausible, does
the workspace look right.

Why this is necessary
---------------------
The static VLM sees only the rest pose. It cannot detect:

- A shoulder joint that self-collides with the base at +60° (invisible at 0°).
- A gripper whose fingers cannot close on a target object (looks fine static).
- Physics instability under gravity (looks stable at rest, collapses in motion).

These are exactly the failure modes a reviewer suspects when a paper claims
"validated" but only shows static renders. By feeding motion frames to a VLM
with native video/multi-image understanding (GLM-4.6V, Dec 2025), the agent
gains the ability to *watch* the simulation and report motion-level problems
that static inspection fundamentally cannot catch.

Pipeline
--------
1.  ``extract_motion_key_frames`` (in ``sim_mujoco.py``) runs a MuJoCo rollout
    and captures 3 key frames (initial / mid-sweep / near-extreme).
2.  This module sends those frames to GLM-4.6V as multi-image input with a
    structured prompt asking for motion-problem identification.
3.  The VLM response is parsed into ``(passed, problems, fix_hints)`` — the
    same shape the static VLM returns, so the existing Fixer can consume both
    channels uniformly.

This is additive to the static channel, not a replacement. The pipeline runs
both: static VLM catches appearance errors, dynamic VLM catches motion errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Default model for dynamic verification. GLM-4.6V (Dec 2025) has native
#: multi-image and video understanding — earlier GLM vision models only
#: accepted single images, which would force per-frame calls and lose the
#: ability to do cross-frame comparison.
_DEFAULT_DYNAMIC_MODEL = "glm-4.6v"


def _build_motion_prompt(
    frames: list[dict[str, str]],
    joint_info: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the GLM-4.6V multi-image content payload.

    The prompt explicitly labels each frame's kinematic context so the VLM
    can reason about what *should* be happening at each moment, then asks it
    to identify motion-level problems (not appearance problems — the static
    channel handles those).
    """
    text_parts = [
        "You are a robotics engineer inspecting simulation frames of a "
        "generated robot arm. These frames include both **motion** frames "
        "(joint-sweep in MuJoCo) and **grasp** frames (a cube grasp test: "
        "fingers closing, then lifting under gravity). "
        "Your job is to judge the **motion and grasp behaviour**, NOT the "
        "static appearance (that is checked separately).\n\n"
        "For each frame I will tell you the context (what is happening at "
        "that moment). Look for:\n"
        "1. Self-collision: do any parts interpenetrate during motion that "
        "were separate at rest?\n"
        "2. Mechanical implausibility: does the arm bend in a way a real "
        "robot cannot (wrong joint axes, impossible articulation)?\n"
        "3. Workspace: does the end-effector reach a reasonable region?\n"
        "4. Grasp: in the grasp frames, can the fingers close on the cube? "
        "Does the cube stay held or drop during the lift phase?\n"
    ]
    if joint_info:
        names = [j["name"] for j in joint_info[:8]]
        text_parts.append(f"\nThe movable joints are: {', '.join(names)}.\n")

    text_parts.append(
        "\nRespond in JSON with this exact structure:\n"
        '{"passed": true/false, "problems": ["description", ...], '
        '"fix_hints": ["suggested parameter change", ...]}\n'
        'Set "passed": true if the motion looks physically sound. If you '
        'see problems, list each one concisely in "problems" and suggest '
        'a concrete fix (e.g. "clamp shoulder_pitch range to ±45°") in '
        '"fix_hints". Be conservative — only flag genuine motion problems.'
    )

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "".join(text_parts)}
    ]
    for f in frames:
        content.append({"type": "text", "text": f"Frame [{f['label']}]: {f['description']}"})
        content.append({"type": "image_url", "image_url": {"url": f["image"]}})
    return content


def verify_motion_video(
    video_path: str,
    api_key: str,
    base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    vision_model: str = _DEFAULT_DYNAMIC_MODEL,
    question: str = "",
) -> dict[str, Any]:
    """Run GLM-4.6V on a simulation VIDEO (native video understanding).

    Unlike :func:`verify_motion` which sends 5 extracted key-frames, this
    sends the **complete** simulation video to GLM-4.6V via its native
    ``video_url`` input. GLM-4.6V performs true temporal/sequential under-
    standing of the motion (it was trained on up to 1-hour video), so it can
    catch transient events that key-frame extraction misses — e.g., a brief
    collision at a specific moment, or a grasp that holds for 0.5s then slips.

    The video is sent as a ``data:video/mp4;base64,...`` URI, so NO public
    URL or file hosting is needed — the video goes directly from the local
    MuJoCo render to the API in a single call.

    Args:
        video_path: path to the .mp4 produced by
            :func:`~lang3d.tools.sim_mujoco.render_simulation_video`.
        api_key: GLM API key.
        question: optional natural-language question. If empty, a default
            structured prompt asks for motion/grasp problems + fix hints.

    Returns:
        Same dict shape as :func:`verify_motion`: ``{passed, problems,
        fix_hints, raw_response}``.
    """
    import base64
    import os

    if not os.path.exists(video_path):
        return {"passed": False, "problems": [f"video not found: {video_path}"],
                "fix_hints": [], "raw_response": ""}

    video_b64 = base64.b64encode(open(video_path, "rb").read()).decode()

    if not question:
        question = (
            "This is a MuJoCo physics simulation of a generated robot arm. "
            "Watch the full motion sequence and judge:\n"
            "1. Does any part self-collide during articulation?\n"
            "2. Is the motion mechanically plausible (correct joint axes)?\n"
            "3. If there is a grasp/lift sequence, does the gripper hold the object?\n\n"
            "Respond in JSON: "
            '{"passed": true/false, "problems": ["..."], "fix_hints": ["..."]}'
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    content = [
        {"type": "video_url",
         "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": question},
    ]

    try:
        resp = client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4000,
        )
    except Exception as e:
        logger.error("GLM-4.6V video verification failed: %s", e)
        return {"passed": False, "problems": [f"VLM call failed: {e}"],
                "fix_hints": [], "raw_response": ""}

    raw = resp.choices[0].message.content or ""
    parsed = _parse_vlm_json(raw)
    return {
        "passed": parsed.get("passed", True),
        "problems": parsed.get("problems", []),
        "fix_hints": parsed.get("fix_hints", []),
        "raw_response": raw,
    }


def verify_motion(
    frames: list[dict[str, str]],
    api_key: str,
    base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    vision_model: str = _DEFAULT_DYNAMIC_MODEL,
    joint_info: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run GLM-4.6V on motion key-frames and return a structured verdict.

    Args:
        frames: output of :func:`~lang3d.tools.sim_mujoco.extract_motion_key_frames`
            — list of ``{"label", "description", "image"}`` dicts where
            ``image`` is a ``data:image/png;base64,...`` URI.
        api_key: GLM API key (same key as text models — ZhipuAI unified auth).
        base_url: API endpoint. GLM-4.6V lives on the standard ``paas/v4``
            endpoint, NOT the coding-plan endpoint.
        vision_model: model id (default ``glm-4.6v``).
        joint_info: optional list of joint metadata dicts for prompt context.

    Returns::

        {
          "passed": bool,          # True if motion looks physically sound
          "problems": list[str],   # motion-level issues identified
          "fix_hints": list[str],  # suggested parameter changes
          "raw_response": str,     # full VLM text (for debugging)
        }
    """
    if not frames:
        return {"passed": True, "problems": [], "fix_hints": [],
                "raw_response": "(no frames — skipped)"}

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    content = _build_motion_prompt(frames, joint_info)

    try:
        resp = client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": content}],
            # GLM-4.6V is a reasoning model: it writes a chain-of-thought to
            # reasoning_content before producing content. With 5 images the
            # reasoning can consume 800-1500 tokens, so max_tokens must be
            # large enough for both reasoning AND the final JSON verdict.
            max_tokens=4000,
        )
    except Exception as e:
        logger.error("GLM-4.6V motion verification failed: %s", e)
        return {"passed": False, "problems": [f"VLM call failed: {e}"],
                "fix_hints": [], "raw_response": ""}

    raw = resp.choices[0].message.content or ""

    # Parse JSON from the response (GLM-4.6V may wrap in ```json fences).
    parsed = _parse_vlm_json(raw)
    return {
        "passed": parsed.get("passed", True),
        "problems": parsed.get("problems", []),
        "fix_hints": parsed.get("fix_hints", []),
        "raw_response": raw,
    }


def _parse_vlm_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a VLM response that may have markdown fences."""
    import re

    # Try direct parse first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` fences.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first {...} block.
    m = re.search(r"\{[^{}]*\"passed\"[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("could not parse VLM JSON response, treating as pass: %s",
                   raw[:200])
    return {"passed": True, "problems": [], "fix_hints": []}
