"""Self-evolving experience store — lexical retrieval over verified-good cases.

See package docstring (:mod:`lang3d.experience`) for the design rationale and
the external-audit finding (H2) this addresses.

Public API
----------
* :class:`CaseRecord` — serialisable summary of one verified-good assembly.
* :class:`ExperienceStore` — append + retrieve + persist.
* :func:`get_store` — process-global accessor (mirrors
  :func:`lang3d.models.cache.get_cache`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default store location.  Resolved relative to project root so it lives next
#: to ``data/runs/`` rather than in the user's home directory — these are
#: project artefacts, not user preferences.
DEFAULT_STORE_DIR = Path("data/experience")

#: Cap per robot category to keep the JSON file bounded and retrieval fast.
#: Past this, lowest-scored entries (oldest, least-retrieved) are pruned.
MAX_ENTRIES_PER_CATEGORY = 50

#: Minimum similarity score for a case to be returned by :meth:`retrieve`.
#: Cases below this are considered irrelevant noise.
MIN_RETRIEVAL_SCORE = 1.0

#: Weight constants — kept in sync with
#: :func:`lang3d.knowledge.assembly_templates.search_assembly_templates`
#: (keyword 3x, robot-type 5x, DOF proximity 2x) so retrieval behaves the
#: same way users already see in template search.
WEIGHT_KEYWORD = 3.0
WEIGHT_CATEGORY = 5.0
WEIGHT_DOF_EXACT = 2.0
WEIGHT_DOF_NEAR = 1.0
WEIGHT_TOKEN_PARTIAL = 1.5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    """One verified-good assembly case, distilled for storage.

    Only fields needed for retrieval and for re-priming generation are kept —
    the full STL/URDF artefacts stay on disk under ``data/runs/<case>/`` and
    are referenced by ``run_dir``.  This keeps the store lightweight (a few
    hundred bytes per case) even when the underlying assembly has megabytes
    of geometry.
    """

    description: str
    """The original natural-language prompt (e.g. "4自由度机械臂")."""

    robot_category: str
    """Output of :func:`~lang3d.tools.assembly_gen.vlm_verify._classify_robot`
    — one of ``fixed_arm`` / ``wheeled`` / ``wheeled_arm`` / ``assembly``."""

    dof: int
    """Degrees of freedom — count of revolute (articulated rotation) joints.

    This matches the user's mental model of "an N-DOF arm": the rotation axes
    a person counts when they say "4自由度机械臂". Prismatic joints (gripper
    finger slides) are excluded because they are end-effector mechanism, not
    arm articulation — counting them inflated DOF and broke DOF-proximity
    retrieval (every stored arm was off by its finger count). Prismatic and
    fixed joint counts are still available in :attr:`joint_types`.  0 for
    pure assemblies with no revolute joints.
    """

    assembly_name: str
    """The ``Assembly.name`` produced (e.g. ``"4dof_robotic_arm"``)."""

    part_count: int
    """Number of parts in the assembly."""

    joint_types: dict[str, int]
    """Histogram of joint types, e.g. ``{"revolute": 4, "fixed": 8}``."""

    default_angles: dict[str, float]
    """The assembly's ``default_angles`` map — the most reuse-critical field,
    since these are what the sanitiser spent rounds converging on."""

    rounds_taken: int
    """How many VLM-fix rounds the case needed.  Lower is better precedent."""

    run_dir: str
    """Relative path to the full run artefacts (``data/runs/<case>/<ts>``)."""

    retrieval_hits: int = 0
    """Incremented each time this case is returned by :meth:`retrieve`.
    Used as the pruning tiebreaker — popular cases survive longer."""

    stored_at: str = ""
    """ISO-8601 timestamp of when the case was written.  Set by
    :meth:`ExperienceStore.record`."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaseRecord":
        # Tolerate older records missing newer fields (forward-compat).
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Scoring — pure function, testable without the store
# ---------------------------------------------------------------------------


_DOF_RE = re.compile(r"(\d+)\s*(?:dof|自由度)")


def _extract_query_dof(query: str) -> int | None:
    """Pull a DOF number out of a free-text query.

    Matches ``"4 dof"``, ``"4dof"``, ``"4自由度"``, ``"4 自由度"``.  Returns
    ``None`` when no DOF is mentioned.
    """
    m = _DOF_RE.search(query.lower())
    return int(m.group(1)) if m else None


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase retrieval tokens.

    Strips punctuation, normalises separators, drops empty tokens.  Used for
    partial-match scoring between query and stored description/keywords.
    """
    return [
        t for t in re.split(r"[\s\-_/]+", text.lower().strip())
        if t
    ]


def score_case(
    case: CaseRecord,
    query: str,
    robot_category: str = "",
    query_dof: int | None = None,
) -> float:
    """Lexical similarity score between a query and a stored case.

    Mirrors :func:`lang3d.knowledge.assembly_templates.search_assembly_templates`
    weighting (keyword 3x / category 5x / DOF 2x) so the experience store's
    ranking is consistent with the existing template-search UX.

    Returns 0.0 when nothing matches.  Pure function — no I/O, safe to call
    from tests directly.
    """
    score = 0.0
    q_lower = (query or "").lower().strip()
    if query_dof is None:
        query_dof = _extract_query_dof(q_lower)
    q_tokens = set(_tokenise(q_lower))
    case_tokens = set(_tokenise(case.description)) | set(
        _tokenise(case.assembly_name)
    )

    # --- Keyword full-substring match (3x) ---
    if q_lower and case.description:
        if q_lower in case.description.lower() or case.description.lower() in q_lower:
            score += WEIGHT_KEYWORD

    # --- Token-overlap partial match (1.5x per matched token) ---
    if q_tokens and case_tokens:
        overlap = q_tokens & case_tokens
        # Cap the partial-match bonus so a 20-token query can't dominate.
        score += min(len(overlap), 4) * WEIGHT_TOKEN_PARTIAL

    # --- Robot-category match (5x — the strongest single signal) ---
    if robot_category and case.robot_category == robot_category:
        score += WEIGHT_CATEGORY

    # --- DOF proximity (exact 2x, ±1 1x) ---
    if query_dof is not None and case.dof > 0:
        if case.dof == query_dof:
            score += WEIGHT_DOF_EXACT
        elif abs(case.dof - query_dof) <= 1:
            score += WEIGHT_DOF_NEAR

    return score


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class _StoreFile:
    """On-disk JSON layout.  One file per category for parallel-safe writes."""

    cases: list[dict[str, Any]] = field(default_factory=list)


class ExperienceStore:
    """Append-only store of verified-good assembly cases with lexical retrieval.

    Thread-safe (one lock per store instance).  Persistence is JSON, one file
    per robot category under ``<store_dir>/<category>.json``.  Writes are
    atomic via ``tempfile + os.replace`` — the same pattern as
    :meth:`lang3d.agent.state.AgentState.checkpoint`.

    Nota bene: this is *not* a hot loop.  The store is read once at the start
    of each pipeline run (retrieve-before) and written once at the end
    (store-after).  A JSON file per category is plenty fast at the
    :data:`MAX_ENTRIES_PER_CATEGORY` = 50 cap.
    """

    def __init__(self, store_dir: str | os.PathLike[str] = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self._lock = threading.Lock()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("experience store at %s", self.store_dir)

    # -- Path helpers ------------------------------------------------------

    def _category_path(self, category: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", category or "assembly")
        return self.store_dir / f"{safe}.json"

    def _load_category(self, category: str) -> list[CaseRecord]:
        path = self._category_path(category)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            # Don't silently swallow — log loudly and treat as empty.  A
            # corrupted store must not crash the pipeline (AGENTS.md §1.1).
            logger.warning(
                "experience store: corrupt %s (%s) — treating as empty",
                path, e,
            )
            return []
        return [CaseRecord.from_dict(d) for d in raw.get("cases", [])]

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        """Write ``payload`` to ``path`` atomically.

        Mirrors :meth:`lang3d.agent.state.AgentState._atomic_write`: temp file
        in the same directory, then ``os.replace``.  Crash mid-write leaves
        either the old or the new file, never a truncated half-write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            # Best-effort cleanup of the temp file on failure; the exception
            # then propagates per AGENTS.md §1.1 (don't swallow silently).
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- Public API --------------------------------------------------------

    def record(self, case: CaseRecord) -> None:
        """Append a verified-good case to the store.

        Deduplicates on ``description`` (case-insensitive) — re-running the
        same prompt updates the existing record rather than stacking copies.
        Prunes to :data:`MAX_ENTRIES_PER_CATEGORY` keeping the
        highest-``retrieval_hits`` cases.
        """
        from datetime import datetime
        if not case.stored_at:
            case.stored_at = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            cases = self._load_category(case.robot_category)

            # Dedup: replace if same description (case-insensitive).
            deduped = [
                c for c in cases
                if c.description.lower() != case.description.lower()
            ]
            # Preserve cumulative retrieval_hits across re-records.
            for old in cases:
                if old.description.lower() == case.description.lower():
                    case.retrieval_hits = max(
                        case.retrieval_hits, old.retrieval_hits,
                    )
                    break
            deduped.append(case)

            # Prune: keep top-N by retrieval_hits, then recency as tiebreaker.
            if len(deduped) > MAX_ENTRIES_PER_CATEGORY:
                deduped.sort(
                    key=lambda c: (c.retrieval_hits, c.stored_at),
                    reverse=True,
                )
                deduped = deduped[:MAX_ENTRIES_PER_CATEGORY]

            payload = {"cases": [c.to_dict() for c in deduped]}
            self._atomic_write(self._category_path(case.robot_category), payload)
            logger.info(
                "experience store: recorded '%s' (cat=%s, dof=%d) → %d cases",
                case.description, case.robot_category, case.dof, len(deduped),
            )

    def retrieve(
        self,
        query: str,
        robot_category: str = "",
        k: int = 3,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ) -> list[CaseRecord]:
        """Return the ``k`` most-similar verified-good cases for ``query``.

        Searches *all* categories (same-category gets a +5 boost) so that a
        slightly-misclassified case is still retrievable.  Increments
        ``retrieval_hits`` on returned cases — this is the "self-evolving"
        signal: cases that get retrieved often survive pruning.
        """
        q_dof = _extract_query_dof(query)
        scored: list[tuple[float, CaseRecord, str]] = []

        with self._lock:
            for cat_path in self.store_dir.glob("*.json"):
                category = cat_path.stem
                for case in self._load_category(category):
                    s = score_case(case, query, robot_category, q_dof)
                    if s >= min_score:
                        scored.append((s, case, category))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        # Bump retrieval_hits and persist (best-effort — failure to update
        # the popularity counter is non-fatal).
        if top:
            try:
                self._bump_hits([c for _, c, cat in top])
            except OSError as e:
                logger.warning("experience store: hit-bump failed (%s)", e)

        return [c for _, c, _ in top]

    def _bump_hits(self, cases: list[CaseRecord]) -> None:
        """Increment ``retrieval_hits`` for ``cases`` in their files."""
        # Group by category so we do one read-modify-write per file.
        by_cat: dict[str, list[CaseRecord]] = {}
        for c in cases:
            by_cat.setdefault(c.robot_category, []).append(c)
        now_iso = cases[0].stored_at if cases else ""
        for cat, hits in by_cat.items():
            existing = self._load_category(cat)
            # Match by description (case-insensitive) — identity is the prompt.
            hit_descs = {h.description.lower() for h in hits}
            for c in existing:
                if c.description.lower() in hit_descs:
                    c.retrieval_hits += 1
                    if now_iso:
                        c.stored_at = now_iso
            payload = {"cases": [c.to_dict() for c in existing]}
            self._atomic_write(self._category_path(cat), payload)

    # -- Introspection (used by tests + the experience-store CLI/report) ----

    def stats(self) -> dict[str, int]:
        """Return ``{category: case_count}`` for every category file present."""
        out: dict[str, int] = {}
        with self._lock:
            for cat_path in self.store_dir.glob("*.json"):
                out[cat_path.stem] = len(self._load_category(cat_path.stem))
        return out

    def all_cases(self) -> list[CaseRecord]:
        """Flat list of every stored case across all categories."""
        out: list[CaseRecord] = []
        with self._lock:
            for cat_path in self.store_dir.glob("*.json"):
                out.extend(self._load_category(cat_path.stem))
        return out


# ---------------------------------------------------------------------------
# Process-global accessor (mirrors models.cache.get_cache)
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
_GLOBAL_STORE: ExperienceStore | None = None


def get_store(store_dir: str | os.PathLike[str] | None = None) -> ExperienceStore:
    """Return the process-global :class:`ExperienceStore`.

    First caller wins on ``store_dir`` — subsequent calls reuse the existing
    instance regardless of the ``store_dir`` argument, matching the
    :func:`lang3d.models.cache.get_cache` singleton semantics.  Pass an
    explicit ``store_dir`` on the very first call (e.g. from tests) to point
    the store at a temp directory.
    """
    global _GLOBAL_STORE
    with _STORE_LOCK:
        if _GLOBAL_STORE is None:
            _GLOBAL_STORE = ExperienceStore(store_dir or DEFAULT_STORE_DIR)
        return _GLOBAL_STORE


def reset_store_for_tests(store_dir: str | os.PathLike[str]) -> ExperienceStore:
    """Replace the global store with a fresh one rooted at ``store_dir``.

    Test-only escape hatch: tests need a clean store per test, but
    :func:`get_store` is a singleton.  Calling this discards the global and
    builds a new one at the given path.  Never call from production code.
    """
    global _GLOBAL_STORE
    with _STORE_LOCK:
        _GLOBAL_STORE = ExperienceStore(store_dir)
        return _GLOBAL_STORE
