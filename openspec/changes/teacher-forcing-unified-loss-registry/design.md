## Context

Stage-2 **Two-Channel Teacher Forcing** (Expectation/Rollout) is implemented as a specialized trainer
(`Stage2TwoChannelTrainer` in `src/trainers/stage2_two_channel.py`) that:

Data → transforms/packing → training → artifacts (current):
- **Data (raw samples):** the Stage-2 variants use ms-swift’s identity collator so the trainer receives a list of raw
  samples (must include `messages` and `assistant_payload`).
- **Channel routing:** `training_step()` chooses Channel-A vs Channel-B deterministically by schedule.
- **Teacher-forcing batch construction:**
  - Channel-A builds a teacher-forced assistant payload directly from GT (`assistant_payload`) and encodes it via the
    Qwen3-VL chat template.
  - Channel-B runs rollout → parse/match → FN-injection and then encodes a teacher-forced completion.
  - When `training.packing=true`, Stage-2 two-channel uses trainer-internal post-rollout packing (carry-only) and attaches a
    packed segment list as `_rollout_matching_meta` (segment boundaries via per-segment `encoded_len`).
- **Forward:** Channel-A may execute multiple soft self-context iterations (embedding replacement at coord slots) and
  anchors CE to the A1 logits; Channel-B is a standard teacher-forced forward.
- **Loss + logging:** `compute_loss()` currently computes weighted CE, bbox regression losses, optional coord
  distribution regularizers, and buffers metrics into a per-step accumulator (`_PendingStage2Log`) that is merged into
  the optimizer-step log line with DDP-safe reduction in `log()`.

Stage-1 SFT is now **static-only packing** (per `stage1-static-packing`). Stage-1 already supports composition of
auxiliary losses/diagnostics via dynamic trainer mixins in `src/sft.py`, but the loss definitions and masking rules
are not shared with Stage-2 two-channel.

Additionally, `progress/full_idea.md` introduces two architecture-level requirements that should become stable
contracts rather than “spread across code paths”:
- **Unified Loss Registry (Deduped):** one canonical definition for loss component names + mask rules across Stage-1,
  Stage-2 Channel-A, and Stage-2 Channel-B (contexts: `gt`, `self_context`, `rollout`).
- **Straight-Through (ST) bridge:** hard-forward / soft-backward behavior for coord-slot self-context embeddings and
  (optionally) geometry coord decode.

This change designs a single, cohesive contract that can be implemented incrementally while preserving current
behavior by default.

Rollout-aligned teacher forcing is implemented by `Stage2RolloutAlignedTrainer` in `src/trainers/stage2_rollout_aligned.py`
and is configured via `custom.trainer_variant: stage2_rollout_aligned`. Stage-2 two-channel inherits from
`Stage2RolloutAlignedTrainer`, so:
- any shared loss/mask utilities MUST live outside the concrete trainer classes, and
- pipeline/registry refactors should treat rollout-matching SFT as a first-class consumer (not an optional follow-on),
  otherwise duplication and contract drift will accumulate in the base class.

Stage-2 two-channel MUST preserve:
- Qwen3-VL chat-template compatibility (including padding-free packing `position_ids` handling),
- packing as the primary efficiency lever,
- geometry invariants (never drop/reorder coords),
- DDP-safe optimizer-step metric aggregation (no rank-conditional collectives),
- “fail-fast when objective changes” vs “best-effort diagnostics” separation.

## Current Drift / Inconsistencies (What’s confusing today)

This refactor is explicitly motivated by *semantic drift* across teacher-forcing code paths that should share the
same tensor-flow semantics:

- **Rollout-matching SFT vs Stage-2 Channel-B desc supervision**
  - `RolloutMatchingSFTTrainer._prepare_batch_inputs()` currently ignores appended-tail `desc` value tokens for CE by
    setting `tail_ignore_pos = _find_desc_value_token_positions(...)` on the append fragment
    (`src/trainers/stage2_rollout_aligned.py`).
  - Stage-2 Channel-B instead computes `tail_desc_pos` and applies `desc_ce_weight` (default 1.0) to those tokens
    (`src/trainers/stage2_two_channel.py`).
  - `progress/full_idea.md` expects FN-injected objects to receive `CE_desc=1` by default.

- **Rollout-matching SFT lacks matched-prefix structure CE**
  - Rollout-matching `_build_labels_and_coord_targets_for_sample()` only assigns CE labels in the tail span; the
    prefix is CE-masked entirely (`src/trainers/stage2_rollout_aligned.py`).
  - Stage-2 Channel-B computes `prefix_struct_pos` via `_matched_prefix_structure_positions(...)` and supervises only
    structure tokens for matched prefix objects (`src/trainers/stage2_two_channel.py`), matching `full_idea`.

- **Meta schema fragmentation**
  - Stage-2 Channel-B meta includes `prefix_struct_pos`, `tail_desc_pos`, `bbox_groups_prefix`, `bbox_groups_fn`
    in addition to `prompt_len/prefix_len/train_len` and coord-supervision lists.
  - Rollout-matching SFT meta contains only the coord-target lists + `tail_ignore_pos`.
  - Stage-1 uses mixin-local conventions. This increases the chance of silent “same name, different meaning” bugs.

- **Docs↔code mismatch on Stage-2 geometry policy**
  - The Stage-2 runbook states that bbox geometry loss uses SmoothL1+CIoU (`docs/training/STAGE2_RUNBOOK.md:65`).
  - Stage-2 two-channel implements SmoothL1+CIoU in `src/trainers/stage2_two_channel.py` (see `_bbox_smoothl1_ciou_loss`).
  - Rollout-matching SFT currently does not include SmoothL1+CIoU in its objective
    (`src/trainers/stage2_rollout_aligned.py`), so running `custom.trainer_variant: stage2_rollout_aligned` can silently
    produce a different Stage-2 objective than the runbook implies.

- **Scheduler description mismatch (idea doc vs implementation)**
  - `progress/full_idea.md` describes the realized Channel-B frequency as “determined by queue availability and routing
    decisions” (`progress/full_idea.md:716`).
  - The implemented Stage-2 two-channel router is a deterministic Bresenham-style schedule keyed on `global_step`
    (`src/trainers/stage2_two_channel/scheduler.py:_stage2_channel_for_step`), and the runbook documents the deterministic
    policy (`docs/training/STAGE2_RUNBOOK.md:77-81`).

The unified registry + pipeline is designed to make these differences explicit, versionable, and testable.

## Goals / Non-Goals

**Goals:**
- Make Stage-2 two-channel’s learning objective and diagnostics **config-declared** via YAML (no additional CLI flags).
- Make rollout-aligned Stage-2’s learning objective and diagnostics **config-declared** via the same module pipeline
  contract (while preserving rollout-specific batch construction and meta formats).
- Establish the **Unified Loss Registry** as a *code-level internal contract* shared by Stage-1 + Stage-2 two-channel:
  - shared component naming (`struct_ce`, `desc_ce`, `coord_reg`, `geo`, …),
  - shared token-type masking rules across contexts (`gt`, `self_context`, `rollout`),
  - shared FP-neutral + EOS-enforced semantics for Channel-B rollout context.
- Expose **ST toggles** to enable/ablate ST behavior without rewriting trainer internals:
  - `coord_ctx_embed_mode: soft|st|hard` (Channel-A),
  - `coord_decode_mode: exp|st` (Channel-A + Channel-B geometry decode).
- Reduce redundancy by extracting shared building blocks so Stage-1 and Stage-2 Two-Channel reuse the same implementation of
  mask building, coord distribution utilities, and bbox loss math.
- Make COCO detection `mAP` a standard eval-step metric (like `rollout/f1`) by enabling
  `rollout_matching.eval_detection.enabled` by default and reusing the offline detection evaluator implementation
  (`openspec/specs/detection-evaluator`) for eval-step scoring.
- Implement the **`progress/full_idea.md` teacher-forcing objective semantics** as a first-class pipeline:
  - Channel-A: CE anchored to GT logits + geometry on self-context logits,
  - Channel-B: FP-neutral + FN-focused + closure supervised + matched-prefix struct-only CE,
  - geometry computed from coord logits via CoordExp decode with canonicalization + CIoU stability.
- Preserve the overall workflow (YAML-first experiments, packing as the primary efficiency lever) while allowing
  **explicit objective variants** (e.g., FN `desc` supervised on/off) so ablations can be compared via config diffs.

**Non-Goals:**
- Change Stage-2 Two-Channel schedule semantics or Channel-A soft self-context algorithmic intent (N× forward remains; only
  modularization + optional ST bridge are introduced).
- Redesign Hungarian matching itself (cost definition, gating policy) unless required by `full_idea` correctness fixes.
- Change Stage-1 static packing or Stage-2 trainer-internal post-rollout packing behavior.
- Introduce additional training CLI flags.
- Make large architecture forks (e.g., separate trainer framework, custom RL loops).

## Code Organization (Anchor: `docs/standards/CODE_STYLE.md`)

Implementation SHOULD follow `docs/standards/CODE_STYLE.md` (Transformers-inspired “Option A”), specifically:
- Separate **contracts** (stable) from **algorithms** (stable-ish) and **trainer orchestration** (change-friendly).
- Prefer explicit `@dataclass` outputs over ad-hoc tuples for internal contracts introduced by this change (e.g., `TeacherForcingContext`,
  module outputs).
- Keep contracts dependency-light (do not import optional heavy backends at import time).

Recommended placement (non-normative, but preferred for maintainability):
- Shared teacher-forcing package (internal; reused across Stage-1 + Stage-2):
  - Contracts (dataclasses/enums): `src/trainers/teacher_forcing/contracts.py`
  - Token-type partitioning (`struct/desc/coord/eos`): `src/trainers/teacher_forcing/token_types.py`
  - Rollout-context masks (FP-neutral + EOS-enforced): `src/trainers/teacher_forcing/rollout_masks.py`
  - Forward strategies (Expectation / Rollout):
    - `src/trainers/teacher_forcing/forwards/proxy_self_context.py` (`ProxySelfContextForward`)
    - `src/trainers/teacher_forcing/forwards/rollout_aligned.py` (`RolloutAlignedForward`)
  - Pipeline parsing + executor + checksum: `src/trainers/teacher_forcing/objective_pipeline.py`
  - Objective/diagnostics modules (atomic; composition-first):
    - `src/trainers/teacher_forcing/modules/token_ce.py`
    - `src/trainers/teacher_forcing/modules/bbox_geo.py`
    - `src/trainers/teacher_forcing/modules/coord_reg.py`
    - `src/trainers/teacher_forcing/modules/coord_diag.py`
- Pure loss math utilities that are not trainer-specific: colocate with existing loss modules (`src/trainers/losses/`)
  or coord-vocab utilities (`src/coord_tokens/`) as appropriate.

Recommended trainer file naming (public-facing clarity; no backward-compat layer):
- Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout):
  - `src/trainers/stage2_two_channel.py`
- Stage-2 Rollout-Aligned Teacher Forcing (rollout-only):
  - `src/trainers/stage2_rollout_aligned.py`

Normative rename policy for this change (fail-fast, single naming):
- The old trainer module paths (`src/trainers/stage2_ab_training.py`, `src/trainers/rollout_matching_sft.py`) SHOULD be
  removed (not kept as shims).
- The old `custom.trainer_variant` strings (`stage2_ab_training`, `rollout_matching_sft`) MUST be removed and MUST fail
  fast with actionable guidance.

Recommended subpackage naming:
- `src/trainers/stage2_ab/` → `src/trainers/stage2_two_channel/`

## Decisions

0) **Public naming: Expectation/Rollout (avoid AB terminology in docs/papers)**

- Paper and docs terminology SHOULD refer to the two Stage-2 channels as:
  - **Expectation** (proxy self-context; no rollout)
  - **Rollout** (rollout-aligned teacher forcing; strict parse/match + FN injection)
- Internal code MAY continue using `"A"`/`"B"` identifiers for short internal names, but documentation
  SHOULD use Expectation/Rollout terminology for reader clarity.

0B) **Trainer variant naming is stable and single-mode**

- Prefer the following `custom.trainer_variant` strings in docs and example configs:
  - `stage2_two_channel`
  - `stage2_rollout_aligned`
- The older strings (`stage2_ab_training`, `rollout_matching_sft`) are removed and MUST fail fast with actionable
  guidance.

1) **Unified Loss Registry is a code-level internal contract (shared across stages)**

- Define a shared internal registry contract that is the *single source of truth* for:
  - token-type partitioning (`struct`, `desc`, `coord`, `eos`),
  - context types (`gt`, `self_context`, `rollout`),
  - (Channel-B only) object subsets (`matched`, `fp`, `fn`) and FP-neutral + EOS-enforced masking rules,
  - loss component names and expected metric key prefixes.
- Stage-1 (static packing) and Stage-2 Two-Channel MUST call into the same registry implementation for mask building and loss
  math. Stage-1 may continue to use mixin composition initially, but it MUST NOT redefine registry semantics in
  separate code paths.

Rationale:
- `progress/full_idea.md` treats this registry as the canonical definition. If we only refactor Stage-2 Two-Channel into modules
  without a shared registry contract, Stage-1 and Stage-2 will continue to drift.

2) **ST bridge is explicit and config-controlled (default preserves current behavior)**

- Add two explicit config knobs to Stage-2 Two-Channel (typed under `stage2_ab.*`):
  - `coord_ctx_embed_mode`: `soft|st|hard` (applies to Channel-A self-context iterations; `soft` is the default).
  - `coord_decode_mode`: `exp|st` (applies to bbox geometry decode; `exp` is the default).
- ST semantics MUST match the `full_idea.md` definition:
  - ST-embedding: hard forward (`argmax` embed) + soft gradient (expected-embed path).
  - ST-decode: hard forward (`argmax` bin value) + soft gradient (expectation decode path).

2B) **Rollout-context semantics are coherent by default; ablations are explicit**

- Treat `progress/full_idea.md` as the canonical intent for teacher-forcing objective semantics, especially for
  Channel-B rollout context (FP-neutral + FN-focused + closure supervised).
- Stage-2 Channel-B and rollout-matching SFT MUST converge on the same rollout-context masking semantics by default:
  - matched prefix: struct-only CE,
  - FP spans: fully masked (FP-neutral),
  - FN injected: struct+desc CE (FN `desc` supervised by default),
  - closure/EOS: supervised (EOS-enforced).
- If reproducing prior rollout-matching behavior is desired for comparisons, it MUST be expressible as explicit,
  typed ablation weights (e.g., FN `desc` weight = 0, matched-prefix struct weight = 0), without forking trainer code.
- Implementation mechanism:
  - Prefer expressing these as **module config** inside the declared pipeline (e.g., `token_ce` module config),
    rather than as ad-hoc trainer-specific conditionals.
- Reproducibility requirement:
  - The resolved module list + module configs (and any “preset” name, if used) MUST be captured in the pipeline
    identity checksum so ablations are auditable.

3) **Config surface: `stage2_ab.pipeline` (ordered, typed, strict)**

- Add a typed config block under the canonical Stage-2 Two-Channel namespace (`stage2_ab`) to declare an ordered pipeline:
  - `stage2_ab.pipeline.objective`: list of objective modules (loss-changing).
  - `stage2_ab.pipeline.diagnostics`: list of diagnostics modules (metrics-only).
- Unknown keys in `stage2_ab.pipeline` MUST fail fast (consistent with `stage2-ab-training` strict config policy).
- Defaulting / preserve-current semantics:
  - If `stage2_ab.pipeline` is absent, Stage-2 Two-Channel constructs a default pipeline from existing Stage-2 Two-Channel knobs
    (e.g., CE weights, bbox weights, coord regularizer weights) so behavior matches today.
  - If `stage2_ab.pipeline` is present, it becomes the source of truth for objective/diagnostics composition.
  - Single-mode guardrail: when `stage2_ab.pipeline` is present, reject objective-affecting flat knobs that would
    otherwise contribute to the default objective (fail fast; require moving values into module configs).

Alternatives considered:
- Reuse Stage-1 style “dynamic Python mixins” for Stage-2 Two-Channel. Rejected for Stage-2 Two-Channel because the forward strategy and
  loss composition are tightly coupled (Channel-A A1 CE anchoring, packed-segment meta) and an explicit pipeline is
  easier to reason about and test than MRO ordering.

4) **Config surface: rollout-matching SFT pipeline declaration**

- For `custom.trainer_variant: stage2_rollout_aligned`, add a typed and strict pipeline declaration under the rollout
  matching namespace:
  - Proposed canonical location: `rollout_matching.pipeline` (same module-spec shape as `stage2_ab.pipeline`).
- Guardrails:
  - If `custom.trainer_variant: stage2_two_channel`, the presence of `rollout_matching.pipeline` MUST fail fast with
    guidance to use `stage2_ab.pipeline` (avoid “config declared but ignored” ambiguity).
  - If `custom.trainer_variant: stage2_rollout_aligned`, the presence of `stage2_ab.pipeline` MUST fail fast (avoid the
    symmetric ambiguity).

5) **Module contracts: objective (fail-fast) vs diagnostics (best-effort)**

- Define a unified internal `TeacherForcingContext` structure that modules consume. At minimum it carries:
  - `channel` (Stage-2 Two-Channel: `"A"`/`"B"`) and per-segment meta (`_rollout_matching_meta` list),
  - `input_ids` and segment boundary info (works for both packed and unpacked batches),
  - **logits per registry context**:
    - Stage-1: `gt`,
    - Stage-2 Channel-A: `gt` (A1) + `self_context` (final iteration),
    - Stage-2 Channel-B: `rollout`,
  - standardized *derived* views (preferred over raw meta interpretation):
    - token-type masks per context (`struct/desc/coord/eos`),
    - coord supervision groups (positions + GT bins) and/or bbox groups for `geo`,
    - rollout object-subset masks (`matched/fp/fn`) for `context=rollout`,
    so objective modules can be implemented once and reused across trainers.
  - packing indicators (packed-mode vs unpacked-mode) and segment boundaries,
  - coord-vocab metadata (coord token ids / id→bin mapping when needed).
- Objective modules MUST be fail-fast when enabled: if prerequisites are missing or alignment fails, training MUST raise
  rather than silently changing the objective.
- Diagnostics modules MUST be best-effort: unexpected exceptions are warned once and the diagnostic is skipped/disabled
  without blocking training.

6) **Forward strategies remain channel-owned; pipeline consumes outputs**

- Keep Channel-A’s iterative soft self-context forward as a distinct forward strategy (it is not “just a loss term”).
- The pipeline consumes a unified `TeacherForcingContext` regardless of channel; CE anchoring is expressed by selecting
  `logits` for `context=gt` in Channel-A.
- Channel-B continues to use standard teacher forcing.

7) **Logging aggregation is centralized and DDP-safe**

- Preserve existing optimizer-step log aggregation behavior:
  - Stage-2 Two-Channel continues to buffer per-micro metrics into `_PendingStage2Log` and merges into `Trainer.log()` on every
    rank with DDP-safe collectives.
  - The base rollout-aligned log buffer remains intact so `rollout/*`, `time/*`, and `packing/*` telemetry remains
    available (key naming may be normalized per the metrics reduction/naming contract in this change).
- Modules MUST NOT call distributed collectives directly. They emit per-micro metrics (numbers/tensors) to the central
  aggregator which owns any reduction policy.

8) **Reproducibility: pipeline identity is part of the run record**

- Log the resolved module list (names + enabled flags + per-module config hash) at trainer init.
- Emit a stable pipeline checksum (e.g., canonical JSON digest) so runs are auditable from logs alone.

## Risks / Trade-offs

- **[Risk] Objective drift during refactor →** Mitigation: add golden tests that compare current monolith behavior vs
  pipeline default loss
  components for fixed teacher-forced batches (Channel-A and Channel-B), including packed-mode meta paths.
- **[Risk] ST breaks parity when enabled →** Mitigation: keep ST off by default; add focused tests verifying ST forward
  values differ while gradients still flow via expectation path; add a smoke config enabling ST explicitly.
- **[Risk] Module ordering ambiguity →** Mitigation: declare that list order is execution order; provide a canonical
  default order and fail fast on duplicates/unknown names.
- **[Risk] DDP deadlocks from conditional reductions →** Mitigation: keep the existing `log()` reduction path rank
  unconditional; modules never gate collectives; aggregator validates “all ranks participate” invariants.
- **[Risk] Config surface complexity →** Mitigation: keep the initial module set minimal (cover today’s components);
  allow module-specific config maps but keep typed parsing and strict unknown-key behavior.
- **[Trade-off] Some duplication remains short-term →** Mitigation: extract only the highest-leverage shared helpers
  first (coord distribution loss/diag + forward preamble), then iterate.

## Migration Plan

1. Land the shared **Unified Loss Registry** utilities (mask builder + shared loss math) and use it as the canonical
   implementation for `full_idea` teacher-forcing semantics.
2. Land pipeline machinery + strict module registry, and implement **`full_idea`-aligned default modules**:
   - `token_ce` (FP-neutral + FN-focused + closure supervised; matched-prefix struct-only CE),
   - `bbox_geo` (CoordExp decode + canonicalization + SmoothL1 + CIoU),
   - `coord_reg` (optional coord distribution regularizers; explicit per-term weights),
   - `coord_diag` (best-effort diagnostics).
3. Update Stage-2 Rollout-Aligned batch-prep/meta to emit the richer meta needed by the registry (e.g., `prefix_struct_pos`
   and `tail_desc_pos`), so rollout-aligned and Stage-2 Channel-B can share the same `token_ce` implementation.
4. Refactor both Stage-2 Two-Channel and Stage-2 Rollout-Aligned `compute_loss()` to call the shared pipeline executor.
   - Provide explicit, logged ablation knobs/presets so experiments can compare objective variants via YAML diffs.
5. Add ST config knobs and implement ST embedding + ST decode; provide smoke configs that enable ST explicitly so “ST can
   see the difference” without changing defaults.
6. (Optional follow-on) Replace Stage-1 mixin composition with an explicit pipeline once the shared registry + executor
   are stable, or keep mixins as thin wrappers over the same module implementations.

## Open Questions

- Module configs live under `stage2_ab.pipeline.*`. Objective-affecting flat keys remain supported only for the
  “pipeline omitted” default-manifest path. When a pipeline is authored explicitly, duplicated flat objective knobs are
  rejected (single-mode fail-fast).
- For rollout-context CE: do we want to expose ablations as (a) explicit numeric weights only, or (b) allow a small set
  of named presets with semantics-based names (e.g., `fn_desc_off`)? (Proposed: numeric weights in `token_ce` module
  config; optionally add presets later if they reduce authoring friction.)
- Do we want Stage-1 to eventually expose a pipeline config surface as well, or treat Stage-1 as “mixin-composed but
  registry-backed” indefinitely?
- Do we want to converge the Stage-2 Two-Channel and Stage-2 Rollout-Aligned pipeline config paths into a single shared namespace
  long-term (e.g., `teacher_forcing.pipeline`), or keep the per-trainer config surface permanently?
