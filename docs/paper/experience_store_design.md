# Experience Store — Design Notes

> Addresses external-audit finding **H2**: *"ArtiCAD ships a self-evolving
> experience store; Language-3D does not."* This document records *why* the
> implementation is what it is, so reviewers (and future us) can see the
> trade-offs were deliberate, not accidental.

## 1. What ArtiCAD does (verified)

From the arXiv HTML full text (arXiv:2604.10992v2, §"Review Agent"):

- The **Review Agent** distils each finished case into a structured summary.
- Summaries are partitioned in **FAISS** into **Good** and **Issue** buckets.
- **Asymmetric retrieval**: the Design Agent reads *positive heuristics* from
  the Good partition and *negative constraints* from the Issue partition;
  the Generation Agent uses Good cases as *few-shot templates*.
- Goal stated in the paper: *"improve performance on future tasks without
  fine-tuning."*

## 2. What Language-3D now does

Same retrieve-before / store-after shape, deliberately simpler retrieval:

| Aspect | ArtiCAD | Language-3D (this work) |
|---|---|---|
| Retrieve-before | FAISS embeddings, Good/Issue partitions | Lexical weighted score (keyword 3×, robot-category 5×, DOF proximity 2×) |
| Store-after | Review-Agent-distilled summaries | Pipeline-distilled `CaseRecord` (dof, joint histogram, default angles) |
| Success-only? | Good + Issue buckets | **Successes only** (`ctx.passed == True`) — no failure memory |
| Persistence | FAISS index | One JSON file per robot category, atomic writes |
| Cross-task growth | Yes (paper's claim) | Yes — `retrieval_hits` accumulates, least-popular pruned |

### Why lexical, not embeddings?

**Honesty about infrastructure.** The project's only LLM backend
(`GLMBackend`) exposes `.chat()` but **no embedding endpoint**. Adding an
embedding dependency would mean either (a) a new API bill/key for a service
we have not validated, or (b) bundling a local embedding model
(`sentence-transformers`, ~400 MB). Both are real costs that an ICRA/CoDL
reviewer would rightly probe. Lexical retrieval:

- Mirrors the **existing** `search_assembly_templates` scoring already in
  the codebase, so there is one retrieval idiom, not two.
- Runs with **zero external dependencies** — fully testable without API keys
  (24 unit tests + 4 integration tests, all green).
- Is **honest** in the paper: we do not claim FAISS-grade semantic retrieval.
  The Related Work section says plainly that Language-3D uses lexical
  retrieval where ArtiCAD uses embedding-based, and that this is a known
  limitation (§Limitations).

### Why successes-only?

ArtiCAD's Issue bucket is genuinely useful for *negative* constraints ("don't
put the gripper on the wrong link"). Language-3D's failure routing already
encodes that knowledge **deterministically** in the Fixer
(`_classify_problems` → stage routing), so a failure-memory would duplicate
an existing mechanism. Storing only verified-good cases keeps the retrieval
pool clean: every retrieved example is one the full pipeline (CAD + VLM +
physics) actually passed.

### Why per-category files?

The robot category (`fixed_arm` / `wheeled` / `wheeled_arm` / `assembly`,
from `_classify_robot`) is the single strongest retrieval signal (5× weight).
Bucketing by category on disk means (a) retrieval can skip whole files for
cross-category queries, (b) writes to one category can't corrupt another,
(c) the JSON stays small (≤50 cases per file).

## 3. The backward-compat invariant (load-bearing)

**When the store is empty, the generation prompt is byte-identical to the
pre-store behaviour.** This is verified by a unit-test that captures the
prompt with `few_shot_extras=""` and asserts no history block is injected.
This means:

- The experience store **cannot introduce a regression** on a first run or
  on a fresh checkout.
- Existing e2e benchmarks are unaffected until cases actually accumulate.
- The feature is strictly additive.

This invariant is why a single passing e2e run with an empty store is
sufficient regression coverage — the store is provably a no-op until it has
content.

## 4. What's NOT claimed

- **No claim of FAISS-level semantic retrieval.** Lexical only.
- **No claim of cross-user sharing.** The store is gitignored and accumulates
  per-installation. A shared seed library is future work.
- **No claim of improved benchmark scores *yet*.** The store starts empty;
  improvement requires accumulated cases, which requires repeated runs. The
  ablation in the paper reports the **empty-store** baseline (the only honest
  measurement available without cherry-picking a populated store).

## 5. Files

| File | Role |
|---|---|
| `src/lang3d/experience/store.py` | `CaseRecord`, `ExperienceStore`, `score_case`, `get_store` singleton |
| `src/lang3d/experience/__init__.py` | Public re-exports + design docstring |
| `src/lang3d/agent/pipeline.py` | `_retrieve_experience_block` (before `run_architect`), `_record_experience` (after `_write_summary`) |
| `src/lang3d/tools/assembly_generator.py` | `few_shot_extras` param on `generate_assembly_from_nl` |
| `tests/test_experience_store.py` | 24 unit tests (pure store, no pipeline) |
| `tests/test_pipeline_experience.py` | 4 integration tests (pipeline ↔ store wiring) |
