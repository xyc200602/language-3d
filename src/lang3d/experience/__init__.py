"""Self-evolving experience store for the Language-3D assembly pipeline.

The store accumulates *verified-good* assembly plans produced by the pipeline
and retrieves the most similar past cases when a new natural-language
description arrives, so that the Architect stage can be primed with successful
precedent instead of generating from scratch every run.

Design choices (intentional, see ``docs/paper/experience_store_design.md``):

* **Lexical retrieval, not embeddings.**  The project's only LLM backend
  (:class:`~lang3d.models.glm.GLMBackend`) exposes ``.chat()`` but no embedding
  endpoint.  Rather than introduce an unvalidated embedding-API dependency,
  we mirror the proven keyword-weighted scoring of
  :func:`lang3d.knowledge.assembly_templates.search_assembly_templates`
  (keyword 3x, robot-category 5x, DOF proximity 2x).  This keeps the store
  testable without API keys and honest about its capabilities.

* **Category bucketing via :func:`lang3d.tools.assembly_gen.vlm_verify._classify_robot`.**
  Cases are stored with their robot category (``fixed_arm`` / ``wheeled`` /
  ``wheeled_arm`` / ``assembly``) and retrieval boosts same-category matches,
  matching how the VLM verifier already reasons about the assembly.

* **Asymmetric write/read.**  Only *verified-good* cases (``passed == True``
  at end of pipeline) are stored, so the store is a curated successes-only
  memory — not a log of every attempt.  This avoids poisoning retrieval with
  failure modes.

* **Atomic persistence** modelled on
  :class:`~lang3d.agent.state.AgentState.checkpoint`: write to a temp file,
  then ``os.replace`` for crash-safety.  Automatic pruning past
  :data:`MAX_ENTRIES_PER_CATEGORY` keeps the store bounded.

This directly addresses external-audit finding H2: ArtiCAD ships a
self-evolving experience store; Language-3D previously did not.  The
implementation here is intentionally simpler (lexical vs FAISS) but follows
the same retrieve-before-generate / store-after-verify shape.
"""

from __future__ import annotations

from .store import (
    CaseRecord,
    ExperienceStore,
    get_store,
    reset_store_for_tests,
)

__all__ = [
    "CaseRecord",
    "ExperienceStore",
    "get_store",
    "reset_store_for_tests",
]
