## Why

Current Stage-2 Channel-B supervision is still framed around edited anchor
targets, false-negative insertion, sorted/tail object sequencing, and one merged
teacher-forced forward. That contract is poorly aligned with autoregressive
decoding failures: a rollout error happens at a specific self-prefix boundary or
transition, while the current target builder often supervises a later edited
sequence.

This change introduces a decoding-aligned Stage-2 correction contract: use the
remaining labeled/UL object set at grammar-valid self-prefixes, anchor training
exactly before the first actionable error, and compile the result into the
shared teacher-forcing IR with strict next-token logits alignment.

## What Changes

- Add a residual-set self-prefix correction contract for Stage-2 rollout
  training.
- Add explicit Stage-2 objective module name `residual_set_correction` with
  `application.preset: rollout_self_prefix`.
- Replace the canonical "edited anchor target / one merged forward" Stage-2
  target semantics with independent correction samples from K valid rollouts.
- Introduce a reusable residual-state machine that emits token-level
  `ValidAction` records with transition-validated `next_state`.
- Require desc-gated one-to-one geometry matching before completed emitted
  objects can enter the emitted set `E_t`.
- Anchor correction samples at the earliest actionable token or object boundary;
  repeated-object boundaries use positive-only residual-set correction, not
  duplicate-specific unlikelihood.
- Support rollout-local UL-promoted positives mined from strict K-valid
  consensus clusters; keep UL provenance, weight, metrics, and artifacts
  separate from labeled GT.
- Define corrected teacher-forced sequence semantics:
  `logit_position = target_position - 1`, raw bad tokens are provenance only,
  and corrected roll-in uses deterministic seeded valid-branch sampling.
- Define coordinate repair as a local bbox-tail span from the anchor, while
  schema, boundary, and description corrections remain one-token spans.
- Keep STOP/EOS as a token-level `ValidAction`; core STOP and continuation
  actions are mutually exclusive, with continuation margin disabled by default.
- **BREAKING** for new residual-set Stage-2 configs: final target construction
  no longer uses the legacy edited-anchor clean-prefix sequence, sorted/tail FN
  insertion, or one merged teacher-forced forward as the canonical target
  semantics.
- Legacy baseline configs remain valid through their existing objective modules;
  the breaking target-semantics change applies only when
  `residual_set_correction` is explicitly selected.
- **Non-goal**: do not restore `loss_duplicate_burst_unlikelihood`,
  bbox/geometry auxiliaries, coord regularizers, or hidden pseudo-GT dataset
  mutation.

## Capabilities

### New Capabilities

- `stage2-residual-set-correction`: Defines grammar-valid self-prefix
  eligibility, residual-state-machine actions, correction-event anchoring,
  rollout-local UL promotion, corrected roll-in, coordinate spans, STOP handling,
  diagnostics, artifacts, and event-to-IR compilation.

### Modified Capabilities

- `stage2-ab-training`: Supersedes the Channel-B edited-anchor / merged-forward
  target contract for the new residual-set objective path; adds config,
  artifact, metric, and validation requirements for residual-set correction and
  UL mining.
- `teacher-forcing-unified-loss-registry`: Adds the residual-set Stage-2
  objective naming/weighting contract and provenance-separated UL metrics while
  preserving hard-SFT and current ET-RMP/stage2-trie baselines as ablations.
- `teacher-forcing-objective-pipeline`: Requires Stage-2 residual builders to
  compile into `TeacherForcingTargetIR` / `SupervisionAtom` and keeps loss
  modules ignorant of FN/FP/duplicate/UL mining semantics.

## Impact

- Affected training surfaces:
  - `configs/stage2_two_channel/**`
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/**`
  - `src/trainers/rollout_matching/**`
  - `src/training/teacher_forcing/**`
  - `src/metrics/**`
- Affected stable docs/specs:
  - `docs/training/STAGE2_DESIGN.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/IMPLEMENTATION_MAP.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
  - `openspec/specs/teacher-forcing-objective-pipeline/spec.md`
- Affected artifacts and diagnostics:
  - Stage-2 monitor dumps and batch metrics gain residual-set, correction-event,
    valid-action, logits-alignment, UL consensus, UL artifact, and
    STOP/continuation diagnostics.
  - `ul_clusters.jsonl` is materialized under the Stage-2 artifact root only
    when monitor/debug/smoke artifact dumping is enabled.
- Verification impact:
  - Unit tests must cover state-machine action consistency, next-token logits
    alignment, desc-gated matching, corrected roll-in, coordinate bbox-tail
    spans, UL consensus/admission, STOP exclusivity, and removal of
    duplicate-specific live losses.
  - Smoke runs must distinguish runnable contract validation from model-quality
    claims; performance comparison remains an experiment report, not an
    OpenSpec success gate.
