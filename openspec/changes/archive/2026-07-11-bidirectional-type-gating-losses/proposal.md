## Why

Stage-1 compact teacher-forcing needs a stable type-family objective that is
stronger than plain full-vocabulary CE at positions where the model must first
choose among schema, coordinate, description, and stop families. The current
`main` checkout has partial `objective.terms.token_type_mass` config plumbing
and older compact type-gate helpers, but `hard_sft` currently rejects the term
and the probability math still uses an allowed-union mass rather than the
promoted bidirectional exclusive family-mass objective.

This change promotes the type-gating loss as a stable `main` contract, decoupled
from the experimental coverage-ledger worktree. It lets the ledger branch reuse
the same loss later without owning or entangling the stable loss semantics.

## What Changes

- Define `objective.terms.token_type_mass` as the stable bidirectional
  type-family mass term for `objective.id: research_teacher_forcing`.
- This change uses the current implementation/public Stage-1 route
  (`pipeline.id: stage1_research_teacher_forcing`,
  `objective.id: research_teacher_forcing`) but does not migrate or rename
  older stable OpenSpec wording for legacy `sft` / `teacher_forcing`
  objectives.
- Replace the old allowed-type-mass/type-gate semantics with an exclusive
  four-family target:
  - `schema`: compact structural/schema tokens such as
    `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
    `<|box_end|>`, plus compact row separators when present.
  - `coord`: the 1000 coordinate tokens `<|coord_0|>` through
    `<|coord_999|>`.
  - `desc`: free-description tokens after excluding schema, coord, stop,
    padding, and control specials.
  - `stop`: the Qwen chat stop marker `<|im_end|>`.
- Keep `<|endoftext|>` and `<|end_of_text|>` out of semantic stop
  supervision; they remain padding/text terminator/control tokens, not training
  EOS targets for this surface.
- Make `objective.terms.token_type_mass` support only `enabled` and `weight`.
  There is no authored `mode` knob; bidirectional exclusive family mass is the
  only current behavior.
- Allow `hard_sft` to use `token_type_mass` while still rejecting
  `conditional_valid_set_likelihood`, `within_valid_coverage`, and
  `continuation_margin` under `hard_sft`.
- Emit unsuffixed raw and contribution metrics. Metric keys MUST NOT introduce
  `_weighted` aliases.

## Non-Goals

- Do not implement the coverage ledger, ledger head, or ledger inference engine.
- Do not implement continuation-margin training, geometry-tail losses, or hard
  bbox-geometry penalties in this main split.
- Do not launch GPU smoke runs from this change. The 128-sample smoke configs
  are prepared later in the ledger worktree after this stable loss is available.
- The main implementation plan stops before ledger worktree edits; any later
  ledger smoke-YAML preparation is a separate controller step, and any GPU
  launch requires separate explicit approval.
- Do not change inference decoding semantics or adapter checkpoint loading.

## Capabilities

### Modified Capabilities

- `stage1-detection-objectives`: Authoring and profile contract for mandatory
  Stage-1 research teacher-forcing type-family mass.
- `teacher-forcing-unified-loss-registry`: Canonical family partition and
  exclusive family-mass probability semantics.
- `trainer-metrics-components`: Metric names, raw/contribution meaning, and
  aggregation scope for the promoted term.

## Impact

- Config/schema impact:
  - `src/config/schema.py`
  - `configs/stage1/detection_teacher_forcing/*`
  - config contract tests for `hard_sft` term validation.
- Loss/runtime impact:
  - `src/detection/token_types.py`
  - `src/training/teacher_forcing/probabilities.py`
  - `src/training/objectives/teacher_forcing.py`
  - `src/trainers/metrics/teacher_forcing.py`
- Metrics impact:
  - `src/training/teacher_forcing/metrics.py`
  - trainer/objective runner tests that assert emitted metric keys.
- Reproducibility impact:
  - resolved runtime payloads and config artifacts must record
    `token_type_mass.enabled` and `token_type_mass.weight`.
  - pure-CE comparator configs may keep the term disabled explicitly.
