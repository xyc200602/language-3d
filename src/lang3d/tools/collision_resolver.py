"""Collision feedback loop: detect → diagnose → fix → re-solve.

This module closes the gap between :mod:`mesh_collision` (detection) and
:mod:`motion_collision.collision_fix_suggester` (diagnosis) by actually
*applying* structured fixes to the assembly and re-solving.

Round-robin strategy
--------------------
1. **Round 1-N (direct fixes)**: ``CollisionFixSuggester.apply_fixes()``
   mutates a deep copy of the assembly — adjusting ``joint.offset``,
   ``part.dimensions``, ``joint.range_deg``, ``default_angles`` — then
   the solver re-runs and the checker re-verifies.  Milliseconds per
   round, no LLM cost.

2. **Final round (LLM fallback)**: if direct fixes stall, the collision
   problems are formatted as a redesign prompt and the LLM regenerates
   the assembly JSON from scratch.  Expensive but creative.

The resolver never mutates the caller's assembly — every round produces
a fresh deep copy.  Termination is guaranteed by ``max_rounds`` and a
monotonic-improvement check (if collision count rises two rounds in a
row, bail out early with the best attempt).

Public API
----------
- :class:`CollisionResolver` — main entry point
- :class:`CollisionResolution` — result dataclass
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly
from .assembly_solver import AssemblySolver

logger = logging.getLogger(__name__)

# Collision detection is optional — degrade gracefully if trimesh/FCL
# are not installed so the resolver can still be imported and tested
# without the heavy native deps.
try:
    from .mesh_collision import MeshCollisionChecker, CollisionResult
    from .motion_collision import (
        CollisionFixReport,
        CollisionFixSuggester,
        CollisionFixSuggestion,
        MotionCollisionChecker,
        MotionCollisionResult,
    )
    _HAS_COLLISION_LIBS = True
except ImportError:  # pragma: no cover — depends on optional deps
    _HAS_COLLISION_LIBS = False
    MeshCollisionChecker = None  # type: ignore[assignment,misc]
    MotionCollisionChecker = None  # type: ignore[assignment,misc]
    CollisionFixSuggester = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CollisionResolution:
    """Outcome of one :meth:`CollisionResolver.resolve` call.

    Attributes
    ----------
    resolved:
        ``True`` only when the final round produced a collision-free
        configuration.  ``False`` means some collisions remain — check
        ``remaining_count`` and use ``modified_assembly`` anyway (it is
        the best attempt, not the original).
    remaining_count:
        Number of colliding pairs still present after the last round.
        ``0`` when ``resolved`` is ``True``.
    rounds_used:
        How many detect→fix→re-solve rounds actually executed.
    fixes_applied:
        Flat list of every :class:`CollisionFixSuggestion` that was
        applied across all rounds (for audit / logging).
    collision_history:
        Per-round collision counts.  ``collision_history[i]`` is the
        pair count seen *before* applying fixes in round ``i+1``.
        Monotonically decreasing means the loop is converging.
    modified_assembly:
        Deep-copied assembly with adjusted parameters.  Untouched if
        ``resolved`` is ``False`` *and* no round improved the count.
    modified_positions:
        Re-solved placements dict matching ``modified_assembly``.
    """

    resolved: bool = False
    remaining_count: int = 0
    rounds_used: int = 0
    fixes_applied: list[Any] = field(default_factory=list)
    collision_history: list[int] = field(default_factory=list)
    modified_assembly: Assembly | None = None
    modified_positions: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class CollisionResolver:
    """Closed-loop collision fixer built on top of the existing detector.

    Parameters
    ----------
    max_rounds:
        Hard upper bound on detect→fix→re-solve iterations.  Rounds
        ``1..max_rounds-1`` use direct parameter fixes; the final round
        may invoke the LLM fallback (when ``llm_fallback`` is True).
    llm_fallback:
        Enable the expensive LLM regeneration round.  Defaults to
        ``False`` so the resolver is free to call repeatedly from inside
        hot paths like ``AssemblySolver.solve``.
    min_confidence:
        Suggestions below this threshold are silently skipped during
        direct-fix application.
    """

    def __init__(
        self,
        max_rounds: int = 2,
        llm_fallback: bool = False,
        min_confidence: float = 0.6,
    ) -> None:
        if not _HAS_COLLISION_LIBS:
            raise RuntimeError(
                "CollisionResolver requires trimesh + python-fcl. "
                "Install with: pip install trimesh python-fcl",
            )
        self._max_rounds = max(1, max_rounds)
        self._llm_fallback = llm_fallback
        self._min_confidence = min_confidence
        self._checker = MeshCollisionChecker()
        self._motion_checker = MotionCollisionChecker()
        self._fixer = CollisionFixSuggester()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        assembly: Assembly,
        positions: dict[str, Any],
        joint_angles: dict[str, float] | None = None,
    ) -> CollisionResolution:
        """Run the detect→fix→re-solve loop.

        The caller's ``assembly`` and ``positions`` are never mutated;
        every fix round works on a deep copy.

        Args:
            assembly: The assembly to improve.
            positions: Solved placements for ``assembly`` (from
                :meth:`AssemblySolver.solve`).
            joint_angles: Optional home-state angles for motion sweep.

        Returns:
            A :class:`CollisionResolution` describing the outcome.  When
            ``resolved`` is ``False``, ``modified_assembly`` still holds
            the best attempt and should be preferred over the original.
        """
        best_assembly = assembly
        best_positions = positions
        best_count = self._count_collisions(assembly, positions, joint_angles)
        history: list[int] = [best_count]
        all_fixes: list[Any] = []

        if best_count == 0:
            return CollisionResolution(
                resolved=True,
                rounds_used=0,
                modified_assembly=assembly,
                modified_positions=positions,
                collision_history=history,
            )

        last_count = best_count
        for round_num in range(1, self._max_rounds + 1):
            use_llm = (
                self._llm_fallback
                and round_num == self._max_rounds
                and round_num > 1
            )

            # Step 1: Diagnose (motion sweep → structured suggestions)
            motion_result = self._safe_motion_sweep(best_assembly, joint_angles)
            fix_report = self._fixer.suggest_fixes(motion_result, best_assembly)

            if not fix_report.suggestions:
                logger.info(
                    "Round %d: no suggestions produced, stopping", round_num,
                )
                break

            # Step 2: Apply fixes
            if use_llm:
                new_assembly = self._apply_llm_fix(best_assembly, fix_report)
            else:
                new_assembly = self._fixer.apply_fixes(
                    best_assembly,
                    fix_report,
                    min_confidence=self._min_confidence,
                )
                all_fixes.extend(
                    s for s in fix_report.suggestions
                    if s.confidence >= self._min_confidence
                )

            # Step 3: Re-solve
            try:
                solver = AssemblySolver(new_assembly)
                new_positions = solver.solve(joint_angles or {})
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "Round %d: re-solve failed (%s), keeping previous state",
                    round_num, exc,
                )
                break

            # Step 4: Re-check
            new_count = self._count_collisions(
                new_assembly, new_positions, joint_angles,
            )
            history.append(new_count)
            logger.info(
                "Round %d (%s): collisions %d → %d",
                round_num, "llm" if use_llm else "direct",
                last_count, new_count,
            )

            if new_count < best_count:
                best_assembly = new_assembly
                best_positions = new_positions
                best_count = new_count

            if best_count == 0:
                break

            # Bail out if we are making things worse two rounds in a row.
            if new_count >= last_count and round_num >= 2:
                logger.info(
                    "Round %d: no improvement (%d ≥ %d), stopping early",
                    round_num, new_count, last_count,
                )
                break
            last_count = new_count

        return CollisionResolution(
            resolved=(best_count == 0),
            remaining_count=best_count,
            rounds_used=len(history) - 1,
            fixes_applied=all_fixes,
            collision_history=history,
            modified_assembly=best_assembly,
            modified_positions=best_positions,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_collisions(
        self,
        assembly: Assembly,
        positions: dict[str, Any],
        joint_angles: dict[str, float] | None,
    ) -> int:
        """Count collision pairs (static + motion sweep).  Lower = better."""
        try:
            static = self._checker.check_assembly_collisions(
                assembly, positions, skip_adjacent=True,
            )
            n_static = sum(
                1 for p in static.pairs if p.is_collision and p.penetration_depth_mm > 1.0
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Static collision check failed: %s", exc)
            n_static = 0

        try:
            motion = self._safe_motion_sweep(assembly, joint_angles)
            n_motion = sum(
                1 for jr in motion.joint_results if jr.has_collision
            )
        except Exception:  # pragma: no cover — defensive
            n_motion = 0

        return n_static + n_motion

    def _safe_motion_sweep(
        self,
        assembly: Assembly,
        joint_angles: dict[str, float] | None,
    ) -> Any:
        """Run the motion sweep, swallowing FCL-only errors.

        The sweep needs FCL and assembled meshes; if either is missing
        it returns an empty result so the resolver can still proceed on
        static collisions alone.
        """
        try:
            return self._motion_checker.check_motion_collisions(
                assembly, base_angles=joint_angles, skip_adjacent=True,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Motion sweep failed: %s", exc)
            return _EmptyMotionResult()

    def _apply_llm_fix(
        self,
        assembly: Assembly,
        fixes: CollisionFixReport,
    ) -> Assembly:
        """Fallback: regenerate the assembly via LLM with collision feedback.

        Mirrors the pattern in :func:`assembly_generator.generate_assembly_with_vlm_loop`
        — formats the collision problems as a fix prompt and asks the
        LLM to redesign the offending parts.
        """
        from ..models.base import Message
        from ..models.glm import GLMBackend
        from .assembly_generator import (
            ASSEMBLY_GEN_SYSTEM_PROMPT,
            _assembly_to_json,
            _parse_assembly_json,
        )
        import os

        api_key = os.environ.get("GLM_API_KEY", "")
        if not api_key:
            logger.warning("LLM fallback skipped: GLM_API_KEY not set")
            return assembly

        problems = "\n".join(
            f"- [{s.fix_type}] {s.description} (置信度 {s.confidence:.0%})"
            for s in fixes.suggestions
        )

        prompt = (
            "以下机械臂装配体在仿真中检测到碰撞，请重新设计以消除碰撞：\n\n"
            f"碰撞问题：\n{problems}\n\n"
            f"原装配体 JSON（参考）：\n{_assembly_to_json(assembly)}\n\n"
            "要求：\n"
            "1. 返回纯 JSON，不要包裹在 ```json``` 中\n"
            "2. 保留原有零件名称和拓扑结构\n"
            "3. 调整引发碰撞的零件尺寸/位置/关节范围\n"
            "4. 所有关节形成连通树（base_plate 为根）\n"
        )

        try:
            backend = GLMBackend(api_key=api_key)
            resp = backend.chat(
                messages=[Message(role="user", content=prompt)],
                system=ASSEMBLY_GEN_SYSTEM_PROMPT,
                temperature=0.5,
                max_tokens=16384,
            )
            return _parse_assembly_json(resp.content)
        except Exception as exc:
            logger.warning("LLM fallback failed (%s), keeping previous assembly", exc)
            return assembly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _EmptyMotionResult:
    """Stand-in returned when the motion sweep cannot run.

    Mimics the ``MotionCollisionResult`` attributes used by the
    fix suggester so it can be fed in without crashing.
    """

    collision_free: bool = True
    joints_checked: int = 0
    joint_results: list = field(default_factory=list)
    summary: str = "motion sweep skipped"


__all__ = [
    "CollisionResolver",
    "CollisionResolution",
]
