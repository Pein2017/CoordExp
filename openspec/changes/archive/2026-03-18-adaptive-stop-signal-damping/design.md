## Context

This change adds an optional teacher-forcing experiment that weakens semantic
stop supervision when the model locally prefers continuing with another object.
Recent repo exploration established that the real dense stop boundary is not a
generic EOS push: for the current CoordJSON + Qwen3-VL path, the decisive
branch is the tokenizer-aware `']},'` vs `']}'` competition at the first
terminal object boundary, while the later top-level `']}'` and `<|im_end|>`
remain closure-tail signals.

The implementation is cross-cutting but stays inside existing training
surfaces:

- data -> transforms/packing:
  - assistant targets are built from JSONL into dense `{"objects":[...]}` text
    and then tokenized through the Qwen3-VL chat template
- training:
  - semantic stop-branch metadata must become available in packing-safe
    teacher-forcing context views
  - `token_ce` must optionally compute a damped stop objective on
    `context=gt`
- artifacts:
  - registry names and metrics must stay canonical across Stage-1 single-forward
    and Stage-2 Channel-A `A1`
  - downstream rollout/eval interpretation continues to use the existing
    `rollout/*`, `eval/parsing/*`, and `eval/detection/*` families

Compatibility constraints remain fixed:

- keep Qwen3-VL chat-template compatibility
- keep `do_resize=false` training semantics
- preserve geometry and coord ordering invariants
- avoid upstream HF model edits
- stay YAML-first with no new CLI flags

## Goals / Non-Goals

**Goals:**

- Add one canonical semantic stop-branch locator that works from the encoded
  target sequence rather than raw-brace heuristics.
- Implement stop-signal damping as an opt-in `token_ce` feature with strict
  config validation and reproducible defaults.
- Share the semantic stop objective definition across Stage-1 GT teacher
  forcing and Stage-2 Channel-A `A1` GT-anchor teacher forcing.
- Keep `struct_ce`, EOS supervision, Channel-B duplicate-ul, geometry, and
  coord-reg routing coherent and auditable.
- Validate the feature with a concrete multi-GPU smoke run using a Stage-2
  config derived from
  `configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml`.

**Non-Goals:**

- No tokenizer-vocab change, no new special tokens, and no serialization-format
  change.
- No change to Channel-B rollout suppression semantics in this change.
- No new model heads or non-logit stop classifier.
- No broad trainer refactor beyond what is needed to expose canonical
  stop-branch metadata and metrics.

## Decisions

### 1. Semantic stop-branch discovery lives in packing-safe teacher-forcing metadata

The canonical stop branch should be derived once from the encoded assistant
target and then exposed through teacher-forcing context metadata. The metadata
should identify:

- the semantic stop token position: first terminal `']}'`
- the closure-tail positions that follow
- the tokenizer-specific continue-vs-stop pair used by the loss

Why:

- it keeps Stage-1 and Stage-2 aligned on the same stop definition
- it avoids re-implementing fragile brace counting in multiple trainers
- it preserves packing safety because branch indices stay segment-local

Alternative considered:

- compute stop positions ad hoc inside `token_ce` from decoded strings
  - rejected because it duplicates logic, is harder to validate under packing,
    and makes fail-fast behavior harder to centralize

### 2. `token_ce` is the host for stop-signal damping

Stop-signal damping should remain a `token_ce` concern rather than a brand-new
top-level pipeline module. The authored surface is:

- `token_ce.config.stop_signal_damping.enabled`
- `token_ce.config.stop_signal_damping.min_weight`
- `token_ce.config.stop_signal_damping.max_weight`
- `token_ce.config.stop_signal_damping.branch_temperature`
- `token_ce.config.stop_signal_damping.curve_gamma`
- `token_ce.config.stop_signal_damping.detach_gate`

Locked defaults:

- `enabled: false`
- `min_weight: 0.2`
- `max_weight: 1.0`
- `branch_temperature: 1.0`
- `curve_gamma: 2.0`
- `detach_gate: true`

Why:

- the feature only changes how a subset of text CE is weighted
- this keeps the experiment config-first and close to the current CE host
- it avoids duplicating text-loss routing or inventing a second partial CE
  module

Alternatives considered:

- a separate `stop_signal_ce` pipeline objective module
  - rejected for v1 because the real change is a structured specialization of
    existing token CE rather than a fully independent forward/input contract

### 3. Registry and metrics treat semantic stop supervision as a separate canonical atom

When enabled, semantic stop positions move out of `struct_ce` and into a
separate canonical component `stop_signal_ce`. Emission stays stage-aware:

- Stage-1 objective atom:
  - `loss/gt_text/stop_signal_ce`
- Stage-2 Channel-A objective atom:
  - `loss/A1_text/stop_signal_ce`

Diagnostics also stay provenance-aware:

- Stage-1:
  - `stop_signal/gt/*`
- Stage-2 Channel-A:
  - `stop_signal/A1/*`

Metric semantics are locked:

- `p_stop_mean` / `p_cont_mean` are pair-normalized probabilities after
  `branch_temperature`
- `margin_mean` is the pair-local logit margin
  `(stop_logit - continue_logit) / branch_temperature`
- `weight_mean` is the post-bounding, post-`curve_gamma` effective weight

Why:

- it makes the experiment auditable instead of hiding it inside `struct_ce`
- it keeps Stage-1 and Stage-2 comparable without forcing identical provenance
  naming

Alternative considered:

- keep semantic stop supervision folded inside `struct_ce`
  - rejected because the experiment would become hard to measure and easy to
    misinterpret

### 4. Downstream evaluation remains on existing rollout/eval families

The new feature only adds branch-local diagnostics. It does not create a second
rollout or eval namespace. Downstream impact is still judged through existing
metrics such as parsing validity, detection recall, and FP drift.

Why:

- this keeps downstream comparison aligned with the repo’s current evaluation
  contract
- it avoids fragmenting monitoring into feature-specific eval keys

Alternative considered:

- add `stop_signal_eval/*` metrics
  - rejected because it would duplicate existing rollout/eval surfaces without
    adding new measurement capability

### 5. Validation includes a concrete two-GPU Stage-2 smoke path

Implementation should add a tiny smoke config that extends
`configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml`, enables the
new stop-signal feature, reduces the run budget, and writes to smoke-specific
output directories. The smoke command should use the documented wrapper:

```bash
config=<smoke-yaml> gpus=0,1 bash scripts/train.sh
```

Why:

- it validates the new feature on the exact Stage-2 A-only path most relevant
  to the proposal
- it checks config resolution, torchrun launch, DDP-safe metric aggregation,
  and the feature’s interaction with the real 1024-pixel data path

Alternative considered:

- limit validation to unit tests and single-GPU dry runs
  - rejected because the request explicitly wants a CUDA 0,1 smoke run and the
    feature adds new multi-step metrics that are worth checking under
    two-process aggregation

## Risks / Trade-offs

- [Tokenizer coupling] -> The semantic stop branch depends on the current
  tokenizer/chat-template segmentation of `']},'` and `']}'`. Mitigation:
  derive positions from encoded targets, keep fail-fast behavior when the branch
  is ambiguous, and document the dependence explicitly.
- [Loss-accounting drift] -> Moving semantic stop positions out of `struct_ce`
  can create double-counting or accidental omission. Mitigation: keep the
  canonical `stop_signal_ce` rule in the unified-loss registry and emit a
  distinct objective atom.
- [Stage-1 / Stage-2 naming drift] -> Stage-1 is single-forward while Stage-2
  uses `A1` provenance. Mitigation: lock Stage-1 `loss/gt_text/*` and
  `stop_signal/gt/*` names explicitly in the metrics spec.
- [Smoke instability] -> A new two-GPU smoke run can fail for launcher or
  output-path reasons unrelated to the objective math. Mitigation: derive the
  smoke config from an existing Stage-2 ablation config, keep the run tiny, and
  use smoke-specific output directories.

## Migration Plan

1. Add the semantic stop-branch metadata path and `token_ce` stop-signal
   weighting support behind the disabled-by-default config.
2. Add canonical registry/metric emission and update training docs.
3. Add a dedicated smoke config extending
   `configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml`.
4. Run targeted tests first, then run the requested `gpus=0,1` smoke via
   `scripts/train.sh`.
5. If the smoke fails because of the new feature, disable
   `token_ce.config.stop_signal_damping.enabled` to recover baseline behavior
   without reverting unrelated changes.

## Open Questions

- Whether the first implementation should log any additional internal debug-only
  counters beyond the canonical `stop_signal/*` family.
- Whether a future follow-up should extend the same stop-signal damping idea to
  rollout-aligned `context=gt` anchor surfaces once the Stage-1 / Stage-2 A-only
  path is validated.
