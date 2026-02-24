## Why

Stage-2 **Two-Channel Teacher Forcing** (Expectation/Rollout) is currently implemented as
`custom.trainer_variant: stage2_two_channel`, and currently couples most of its learning objective and
training-time telemetry to a monolithic `compute_loss()` implementation (including Channel-A soft self-context
iterations, CE masking/weighting, bbox regression losses, coord distribution regularizers, and step-level logging
aggregation).

Rollout-aligned teacher forcing is currently exposed as `custom.trainer_variant: stage2_rollout_aligned` and has the
same core problem:
- It implements a separate monolithic teacher-forcing objective (`RolloutMatchingSFTTrainer.compute_loss()`) with its
  own masking + coord distribution supervision, and
- Stage-2 two-channel inherits from `RolloutMatchingSFTTrainer`, so “Stage-2 refactors” that do not also refactor
  rollout-matching tend to accumulate duplication and subtle semantic drift.

This makes research iteration expensive and error-prone:
- Adding, removing, or reweighting an auxiliary loss or diagnostic typically requires editing a large function with
  multiple intertwined invariants (packing/meta alignment, Channel-A vs Channel-B differences, DDP-safe logging).
- Running clean ablations is harder than it should be: the “objective surface” is not explicitly declared in YAML, so
  comparing runs requires code inspection rather than config diffing.
- Stage-1 is now **static-only packing** (per `stage1-static-packing`), and Stage-2 Two-Channel uses **post-rollout packing**
  internally. Both regimes need to share teacher-forcing utilities without leaking assumptions across packing modes.

Separately, `progress/full_idea.md` introduces two architectural expectations that are hard to retrofit if we only
refactor `compute_loss()`:
1) A **Unified Loss Registry (Deduped)** that is the *single source of truth* for loss component names + mask rules
   across Stage-1, Stage-2 Channel-A, and Stage-2 Channel-B.
2) A **Straight-Through (ST) bridge** (hard forward / soft backward) for:
   - Channel-A coord-slot self-context embeddings, and
   - geometry coord decode (optional) so geometry can be evaluated on “inference boxes” while retaining smooth grads.

`progress/full_idea.md` also makes Channel-B “Unified one-pass” semantics first-class:
- **FP-neutral** (do not directly penalize unmatched predicted objects),
- **FN-focused** (append FN records *inside* the same top-level `objects[]` container),
- **closure supervised** (top-level `}` and `<|im_end|>` remain supervised),
- and **matched-prefix structure supervision** (matched objects: `CE_struct=1`, `CE_desc=0`).

Today, we already see semantic drift between “Stage-2 Channel-B” and “Stage-2 rollout-aligned” implementations
(e.g., whether appended FN `desc` tokens are supervised), which makes `full_idea` hard to land without a shared,
strict contract.

We want a unified, config-declared teacher-forcing objective system that:
- makes Stage-2 **Two-Channel Teacher Forcing** **config-declared** (objective + diagnostics composition in YAML),
- makes rollout-only Stage-2 **config-declared** (same pipeline contract; different forward/batch-prep strategy),
- establishes the **Unified Loss Registry** as a *code-level internal contract* shared by Stage-1 and Stage-2,
- exposes **ST toggles** so we can enable/ablate ST without rewriting trainer internals,
- and reduces redundancy while keeping packing correctness and DDP-safe logging.

And we want to **implement the `full_idea` teacher-forcing objective semantics** in a way that:
- is representable and auditable via the module pipeline + registry, and
- can be adopted incrementally without breaking the overall workflow (YAML-first, packing-first).

## What Changes

### 1) Unified Loss Registry (internal contract; shared across stages)
- Define canonical loss component names and masking semantics (token-type × context × object-subset) as a shared
  *internal* registry contract consumed by:
  - Stage-1 GT teacher forcing (`context=gt`),
  - Stage-2 Channel-A (`context=gt` for CE anchor + `context=self_context` for geo),
  - Stage-2 Channel-B (`context=rollout`; FP-neutral + EOS-enforced).

### 2) ST bridge is config-controlled (enable to see the difference)
- Add config knobs to select:
  - `coord_ctx_embed_mode: soft|st|hard` (Channel-A coord-slot context embeddings),
  - `coord_decode_mode: exp|st` (geometry coord decode),
  without requiring upstream model edits.

### 3) Config-declared teacher-forcing pipeline (objective + diagnostics)
- Introduce a YAML-declared module pipeline with an explicit ordered list of:
  - **Objective modules** (loss-contributing, fail-fast when enabled),
  - **Diagnostics modules** (metrics-only / logging-only, best-effort).
- Refactor Stage-2 Two-Channel and Stage-2 Rollout-Aligned `compute_loss()` to:
  - run a channel-owned forward strategy (Channel-A N× forward; Channel-B one-pass teacher forcing),
  - build a `TeacherForcingContext` (including registry masks + relevant logits per context),
  - execute configured modules to produce loss + metrics,
  - preserve packing invariants and existing DDP-safe log aggregation semantics.

### 4) Stage-1 stays static-packing compatible
- Stage-1 remains static-only packing; the shared registry + modules are designed to be packing-safe (segment-aware).
- The initial implementation may keep Stage-1’s current mixin-based trainer composition, but Stage-1 loss/mask logic
  must share the same registry implementation (no duplicated “loss math”).

### 5) `full_idea` semantics become implementable (and testable) as a first-class pipeline
- Implement Channel-B FP-neutral + FN-focused mask semantics as registry-backed module behavior:
  - matched prefix objects: `CE_struct=1`, `CE_desc=0`, `CE_coord=0`,
  - FP objects (incl. dropped-invalid spans): `0` for CE and geometry,
  - FN injected objects: `CE_struct=1`, `CE_desc=1` (by default), `CE_coord=0`,
  - closure/end tokens: supervised (EOS-enforced).
- Provide explicit config knobs/presets so we can run coherent supervision variants side-by-side when needed (e.g.,
  FN `desc` supervised on/off) via YAML diffs.

## Capabilities

### Added Capabilities
- `teacher-forcing-unified-loss-registry`: Define the canonical internal loss registry (contexts, token types, object
  subsets, component naming) used across Stage-1 and Stage-2.
- `teacher-forcing-objective-pipeline`: Define a YAML-declared module pipeline contract for teacher-forcing trainers
  (ordering, enable/disable, per-channel applicability, fail-fast vs best-effort, metric key expectations).

### Modified Capabilities
- `stage2-ab-training`: Extend the Stage-2 two-channel training surface to allow an explicit objective/diagnostics module
  pipeline declaration, add ST config knobs, and require that defaults preserve the current objective semantics when the
  pipeline keys are not provided (via an explicit Default Pipeline Manifest in the delta spec).
- `rollout-matching-sft`: Extend Stage-2 rollout-aligned teacher forcing (`custom.trainer_variant: stage2_rollout_aligned`)
  to support the same config-declared objective/diagnostics module pipeline and unified loss registry contract,
  preserving default behavior when not configured.

### Naming / Deprecation (Public Clarity)

To avoid confusing public readers with internal “AB” terminology and to keep the codebase single-mode, this change
standardizes trainer naming with no backward-compat layer:
- `custom.trainer_variant: stage2_two_channel` (two-channel Expectation/Rollout)
- `custom.trainer_variant: stage2_rollout_aligned` (rollout-only)

The older strings (`stage2_ab_training`, `rollout_matching_sft`) are removed and MUST fail fast with actionable
guidance.

## Impact

- Trainer implementation:
  - `src/trainers/stage2_two_channel.py` (refactor compute_loss orchestration; ST bridge; preserve DDP-safe logging)
  - `src/trainers/stage2_rollout_aligned.py` (refactor compute_loss orchestration to use pipeline + registry)
  - Shared registry + module pipeline runtime (added under `src/trainers/`).
  - Stage-1 mixins and Stage-2 Two-Channel modules reuse the same shared loss/mask utilities (no duplicated definitions).
- Config schema + validation:
  - `src/config/schema.py` (typed config for pipeline + ST knobs; unknown-key fail-fast)
  - `src/config/rollout_matching_schema.py` (if pipeline config is declared under `rollout_matching.*`, it must be
    typed and strict here as well)
  - `configs/stage2_two_channel/*` leaf examples (smoke/prod) to enable ST or declare pipelines explicitly.
- Metrics/docs:
  - Emit canonical `loss/<component>` keys derived from the unified loss registry and sync docs/scripts to the canonical
    contract (no metric aliases).
  - Add documentation for pipeline identity keys (module list + checksum) and ST-related diagnostics.
  - Make COCO detection `mAP` a standard eval-step metric (like `rollout/f1`) when `rollout_matching.eval_detection.enabled=true`,
    logged under `rollout/mAP`.
- Verification:
  - Unit tests asserting the default (implicit) pipeline matches current behavior for fixed teacher-forced batches
    (Channel-A + Channel-B), including packed-mode meta handling.
  - Tests asserting a minimal pipeline override (enable/disable one module) changes only the expected loss terms/keys.
  - Tests for ST modes to ensure forward/backward semantics match the ST definitions (hard forward + soft grad).
  - Tests for `full_idea` Channel-B semantics (FP-neutral + FN-focused + closure supervised), including:
    - matched-prefix `desc` tokens are not supervised,
    - FN-injected `desc` tokens are supervised by default,
    - FP spans are fully masked out,
    - and geometry losses include matched+FN but exclude FP.

## Finalized Delta Notes (implementation sync)

- YAML key names remain unchanged from the draft (`stage2_ab.pipeline.*`, `rollout_matching.pipeline.*`).
- Rollout-rich meta contract is now explicit in both Stage-2 trainers and includes:
  - `prefix_struct_pos`,
  - `tail_desc_pos`,
  - `tail_closure_pos`,
  - `bbox_groups_prefix`,
  - `bbox_groups_fn`,
  - plus existing `prompt_len/prefix_len/train_len` and coord-supervision fields.
- Canonical registry scalar naming uses `loss/coord_token_ce` for coord-bin CE; the trainer-specific ambiguous
  `loss/coord_ce` label is superseded by the canonical key in pipeline-emitted payloads.
