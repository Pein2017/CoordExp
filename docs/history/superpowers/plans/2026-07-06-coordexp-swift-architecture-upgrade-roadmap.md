# CoordExp-Swift Architecture Upgrade Roadmap

> **Status:** planning note, not implementation approval.
> **Use with:** `grill-me`, `improve-codebase-architecture`, then OpenSpec only
> for compatibility-sensitive contracts.

## Decision

Adopt the architecture review's main diagnosis: CoordExp-Swift V1 has a real
local training/inference backbone, and the packed-Qwen/FA2 path is not currently
the primary blocker. The next upgrades should target hidden correctness and
operator-knowledge risks around cache identity, production readiness,
checkpoint-to-inference handoff, training assembly locality, loss-plan
authority, and docs authority.

Do not start one broad architecture refactor. Split the work into five upgrade
waves:

1. **Pack Cache Semantic Identity**
   - First upgrade.
   - Conservative patch first: include Qwen forward-side semantic producers in
     the packing-cache determinant.
   - Deeper cache-shape refactor is optional and should be discussed after the
     conservative invalidation patch.

2. **Artifact Readiness And Checkpoint Handoff**
   - Design as one artifact-driven production gate.
   - Read-only validator should consume run/checkpoint artifacts and return
     promote/hold with actionable failure evidence.
   - Production inference should prefer `checkpoint_handoff.json` as canonical
     identity for base model, adapter payload, special-token embedding delta,
     tokenizer/processor/template, and intended inference family.

3. **Training Assembly Locality**
   - Refactor `run_training_pipeline` only around phase-level contracts and
     receipts, not just to shorten the function.
   - Preserve `src/train.py` and public config behavior.
   - Each extracted phase should increase locality, artifact visibility, or test
     leverage.

4. **Loss Plan Authority**
   - Make `LossRunner` the single owner of the resolved executable loss plan and
     the artifact plan emitted by training.
   - Resolve zero-weight semantics before adding the next serious auxiliary
     loss.

5. **Authority Docs Sweep**
   - Coupled cleanup after code/spec changes.
   - Make CoordExp-Swift V1 local modules the unambiguous current authority for
     this worktree.
   - Mark MS-Swift/mainline routes as reference, historical, or future-work
     context where applicable.

Confirmed grill decisions:

- Wave A starts with the conservative determinant patch, not a cache-payload
  redesign.
- Wave B becomes an OpenSpec change because production readiness and checkpoint
  handoff are compatibility-sensitive contracts.
- Production inference should use `checkpoint_handoff.json` as canonical in
  production mode. Manual path composition remains allowed only as marked
  research/dev evidence.
- Cache and artifact gates come before `run_training_pipeline` refactoring.
- `weight: 0` loss terms are invalid unless the term explicitly declares a
  diagnostic or validator mode.
- Docs are updated with each contract change, followed by one final residue
  sweep.

Global constraint: standardize the infrastructure pipelines and lose weight.
Every upgrade should reduce duplicated authority, implicit operator knowledge,
or ad hoc handoff surfaces. Do not add a new abstraction, config knob, report,
or workflow unless it protects correctness, makes the pipeline easier to audit,
or replaces an existing heavier path.

## Rationale

The highest-risk failure modes are silent or operator-memory-driven:

- stale pack caches can reuse old packed-forward semantics;
- production launch validity is currently distributed across receipts, specs,
  tests, and human checklists;
- checkpoint-to-inference parity can be broken by manually pairing a correct
  adapter with the wrong embedding delta or base/template identity;
- training assembly is hard to inspect because many correctness-sensitive
  phases live inside one orchestration function;
- loss semantics are split between config, runner construction, and pipeline
  artifact emission;
- docs still contain mixed authority between Swift V1, mainline training, and
  MS-Swift reference routes.

The packed-Qwen and FA2 implementation should not be reopened as a broad
redesign unless new evidence shows a concrete bug. Current review evidence
points instead to surrounding lifecycle and artifact seams.

## Consequence

### Wave A: Pack Cache Semantic Identity

Recommended immediate behavior:

- Add Qwen forward-side producer identities to
  `PACKING_CACHE_CODE_IDENTITY_FILES`, at minimum:
  - `src/qwen/positions.py`
  - `src/qwen/fa2.py`
  - `src/qwen/forward.py`
- Add tests proving cache fingerprints change when these identities change.
- Keep worker count as provenance only, not a semantic determinant.

Open design fork:

- **A1:** continue caching complete `SupervisedMicroStep` objects, with broader
  semantic invalidation.
- **A2:** cache only stable packed/supervision data and rebuild Qwen
  position/FA2/forward decorations after cache load.

Recommendation: **A1 now, A2 later only if cache churn or correctness reviews
keep recurring.**

### Wave B: Artifact Readiness And Checkpoint Handoff

Create one read-only readiness validator over run/checkpoint artifacts. It
should verify at least:

- resolved config and config fingerprint;
- rank/world-size status;
- pack cache identity and pack-plan receipt;
- FA2/MRoPE proof receipt;
- scheduler/runtime receipts;
- final checkpoint presence and checkpoint metadata;
- `checkpoint_handoff.json`;
- adapter and special-token embedding delta identity;
- base model, tokenizer, processor, template, and intended inference family;
- eval-forward or val200 artifacts when the requested gate requires them.

Production inference should have a canonical handoff path:

- **production mode:** consume `checkpoint_handoff.json` and reject mismatches;
- **research/dev mode:** allow explicit manual paths, but record manual
  composition as noncanonical evidence.

This wave likely deserves OpenSpec because it changes production-readiness and
inference-handoff contracts.

### Wave C: Training Assembly Locality

Do not split `run_training_pipeline` into shallow helpers. Split by phase only
when the extracted module owns a meaningful interface and receipt:

- run/config identity;
- Qwen/model/trainable-surface assembly;
- pack-cache and pack-plan assembly;
- optimizer/scheduler/runtime assembly;
- eval/cache/checkpoint handler assembly;
- trainer execution and finalization.

Each phase must have a narrow parity check against existing artifacts.

### Wave D: Loss Plan Authority

`LossRunner.from_config(...)` should produce the executable runner and a
resolved loss-plan artifact for training to write directly.

Zero-weight terms must not remain implicit. Candidate semantics:

- **Z1:** `weight: 0` disables the term entirely.
- **Z2:** `weight: 0` means diagnostic-only.
- **Z3:** `weight: 0` is invalid unless the term explicitly declares
  diagnostic/validator mode.

Recommendation: **Z3**, because it is least ambiguous for future research
losses.

### Wave E: Authority Docs Sweep

After Waves A-D, run a docs residue pass for:

- stale `src.sft` route references;
- stale `src/trainers` or mainline-only training authority wording;
- unqualified MS-Swift-as-boundary wording;
- `sorted` vs `geo_sorted` ambiguity;
- missing or stale pointers from `docs/COORDEXP_SWIFT.md`,
  `docs/training/README.md`, `docs/ARTIFACTS.md`, and
  `docs/standards/UPSTREAM.md`.

## Evidence

- Scope: `partial`
- Review source:
  `/data/CoordExp/.codex/attachments/f3cf5424-d726-4cc9-af52-26b782e7bb22/pasted-text.txt`
- Existing reviewed worktree:
  `/data/CoordExp/.worktrees/CoordExp-swift`
- Review-reported validation:
  `conda run -n ms python -m pytest tests/training/test_pack_cache.py tests/qwen/test_fa2.py tests/qwen/test_positions.py tests/losses/test_runner.py -q`
  with `49 passed`
- Current routing docs consulted:
  `docs/AGENT_INDEX.md`, `docs/catalog.yaml`,
  `docs/architecture/README.md`
- CodeGraph spot-check:
  `src/training/pack_cache.py`, `src/training/pipeline.py`,
  `src/losses/runner.py`, training runtime relationships

## Grill Questions

1. Should Wave A be implemented as the conservative determinant patch first?
   - **A. Yes, add forward-side producer identities now.**
   - B. Pause and redesign cache payload shape first.
   - Recommendation: **A**.

2. Should Wave B become an OpenSpec change?
   - **A. Yes, production readiness and checkpoint handoff are stable
     compatibility-sensitive contracts.**
   - B. Keep it as internal tooling without spec.
   - Recommendation: **A**.

3. Should production inference require checkpoint handoff identity by default?
   - **A. Yes, use `checkpoint_handoff.json` as canonical in production mode;
     allow manual paths only in marked research/dev mode.**
   - B. Keep manual adapter and embedding-delta paths equally canonical.
   - Recommendation: **A**.

4. Should `run_training_pipeline` be refactored before Wave B?
   - A. Yes, assembly locality first.
   - **B. No, harden cache and artifact gates first.**
   - Recommendation: **B**.

5. What should zero-weight loss terms mean?
   - A. Disabled.
   - B. Diagnostic-only.
   - **C. Invalid unless the term explicitly declares diagnostic/validator
     mode.**
   - Recommendation: **C**.

6. Should docs cleanup be a separate first-class wave?
   - A. Yes, do broad docs cleanup immediately.
   - **B. No, update docs with each contract change and do one final residue
     sweep.**
   - Recommendation: **B**.
