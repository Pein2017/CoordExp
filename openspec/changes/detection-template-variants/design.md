## Context

The compact detection surface is currently implemented as `compact_full` across
training, inference, parsing, prompt examples, and token-row validation. The
actual serialized assistant rows are a stable part of the model contract, but
the public surface is split across several lower-level knobs such as format
names, row separators, parse modes, serialization policies, and token-row
groups.

This change introduces compact wrapper variants for ablation and anchoring
experiments while keeping compact wrapper authoring centralized under
`detection_template.id`. The follow-on
`compact-template-field-order-ablation` change amends field-order authoring:
`custom.object_field_order` remains the authored source for desc-first versus
geometry-first compact row layout. The implementation must keep Qwen
chat-template behavior intact, keep `do_resize` false for detection training,
preserve geometry order and coordinates, and avoid editing upstream HF model
internals.

Current high-impact code owners include:

- `src/detection/template.py` for first-class template rendering/parsing.
- `src/detection/teacher_forcing/compact_full_policy.py` for compact parsing and
  rendering helpers.
- `src/common/detection_sequence.py` for compatibility/diagnostic parsing
  facades that should not silently become the strict training/eval contract.
- `src/detection/teacher_forcing/target_builder.py` and related description
  token helpers for token-level teacher-forcing targets.
- `src/config/schema.py` and `src/detection/runtime.py` for config validation and
  runtime support.
- `src/config/prompts.py` and `src/infer/runtime.py` for dense prompt and
  inference message construction.
- `src/infer/runtime.py` and the shared inference runtime contract for parser
  provenance and backend prompt parity.
- `src/infer/checkpoints.py` and token-row validation for token_embeddings_adapter row
  contracts.
- `src/detection/evaluation.py` and infer artifact metadata for post-hoc mAP
  preflight and scoring.

Data flow for this change:

```text
training/inference YAML
-> typed config resolves detection_template.id
-> template registry derives render/parser/prompt/token-row contract
-> dataset and teacher-forcing builders render canonical assistant text
-> token_embeddings_adapter trains exactly the required token rows
-> inference renders matching prompts and parses generated text by template id
-> artifacts persist detection_template.id
-> post-hoc mAP scores the existing normalized pixel geometry schema
```

## Goals / Non-Goals

**Goals:**

- Make `detection_template.id` the authored source of truth for compact template
  family, closure tokens, row separator, parser family, structural token rows,
  and template provenance; `compact-template-field-order-ablation` adds
  `custom.object_field_order` as the companion authored source for desc-first
  versus geometry-first row layout.
- Rename the current compact surface from `compact_full` to `compact` in active
  configs/docs without keeping `compact_full` as a schema alias.
- Support semantic compact template ids:
  - `compact`
  - `compact_box_closed`
  - `compact_object_box_closed`
  - `compact_object_box_closed_lines`
- Keep `stage1_json_pretty` supported as a separate non-compact template.
- Derive parser, renderer, prompt row pattern, row separator, required special
  tokens, trainable token_embeddings_adapter rows, artifact metadata, and post-hoc mAP
  preflight from the selected template id.
- Validate that compact token-row adaptation supports the selected 1002, 1003,
  or 1004 row contract.
- Make inference/eval artifacts sufficient for post-hoc mAP to accept each
  variant without output-shape guessing.

**Non-Goals:**

- No implementation before the OpenSpec docs and implementation roadmap validate
  after review convergence.
- No permanent compatibility alias for `compact_full`.
- No arbitrary user-authored separator, closure-token, parser-mode, or
  serialization-policy knobs.
- No tokenizer migration. `<|object_ref_end|>` and `<|box_end|>` are already in
  the tokenizer vocab.
- No rollout behavioral test gate in this change. Rollout code should remain
  import-safe, but there is no rollout checkpoint for these variants yet.
- No benchmark or production metric claim.
- No new raw-generation artifact contract. Standard post-hoc mAP consumes the
  normalized `gt_vs_pred.pred` object arrays produced by inference
  materialization.

## Decisions

### Decision: Use semantic template ids, not versioned or parameterized ids

The public ids are:

```text
stage1_json_pretty
compact
compact_box_closed
compact_object_box_closed
compact_object_box_closed_lines
```

`compact` is the semantic replacement for the current `compact_full` bytes:

```text
<|object_ref_start|>{desc}<|box_start|>{coords}
```

The other compact ids name the structural difference directly. The line variant
canonical renderer includes exactly one newline after every object row, including
the final row.

Alternative considered: keep `compact_full` plus a `v1` or policy suffix. That
would preserve old naming but would keep implementation history in the research
surface and make ablation names harder to scan.

Alternative considered: expose booleans such as `include_box_end` and
`row_separator`. That would make ad hoc variants easier but would expand the
schema into a serialization framework. The agreed surface is a small registry.

### Decision: Centralize template contracts in one registry/resolver

Each template id resolves a compact contract containing:

- canonical row renderer,
- strict parser,
- prompt example row,
- row separator,
- required structural tokens,
- required trainable token rows,
- artifact metadata value,
- post-hoc parsing policy.

Lower-level names such as parser ids or serialization policies may exist as
internal code details, but they are not independently authored config choices.
Existing metadata carriers such as `detection_sequence_format`, `row_separator`,
`compact_full_parse_mode`, `parser_id`, or `parsing.compact_full.mode` must be
removed as authored sources of truth or reduced to derived metadata under the
semantic template id.

The canonical YAML authoring path is `detection_template.id` for both training
and inference configs. Inference runtime code may materialize the resolved value
into internal dataclasses or provenance records, but it should not introduce an
inference-only alias such as `infer.detection_template.id`; old inference fields
such as `infer.detection_sequence_format`, `infer.row_separator`,
`infer.compact_full_parse_mode`, and `infer.parsing.compact_full` are
rejection-only migration diagnostics.

### Decision: Keep strict and compatibility parser boundaries separate

Strict training, inference-materialization, and metric-bearing parsing should be
selected by the template resolver. Low-level row helpers may be reused, but the
compatibility facade in `src/common/detection_sequence.py` remains diagnostic or
legacy-facing unless it is deliberately migrated. This avoids collapsing strict
parser failures into salvage behavior that is useful for inspection but not for
metric-bearing artifacts.

### Decision: Make compact parsers strict

Training targets, inference parsing, and post-hoc mAP use the selected template
id as the parse contract. A closed variant must require its closure tokens. The
line variant must require the canonical newline placement. The `compact` parser
must not silently accept closure-token output.

Alternative considered: infer the variant from generated text. This would make
old artifacts easier to inspect but would weaken ablation interpretation and
would hide config/artifact mismatches.

### Decision: Derive token_embeddings_adapter rows from the template id

The token_embeddings_adapter remains the mechanism for trainable token rows, but the row
set is no longer coord-only on compact detection runs. The selected compact
template derives the required row count:

- `compact`: 1002 rows.
- `compact_box_closed`: 1003 rows.
- `compact_object_box_closed`: 1004 rows.
- `compact_object_box_closed_lines`: 1004 rows.

The persisted module name is `token_embeddings_adapter`; docs/specs should
describe this surface as token-row adaptation when compact detection structural
rows are included.

Adapter checkpoint validation for compact templates applies when an adapter
checkpoint carries the token_embeddings_adapter module. It must validate the exact
template-derived row id set, reject missing/extra/duplicate ids, and ensure saved
offset tensor row counts agree with the resolved ids. Full or merged checkpoints
that do not carry a token_embeddings_adapter module still need resolved template metadata,
but adapter tensor validation is not applicable to them.

### Decision: Persist template metadata in artifacts for mAP

Inference artifacts must persist the resolved `detection_template.id` in
resolved config, run metadata, and `gt_vs_pred.jsonl` rows. Generated compact
text is parsed during inference materialization using that id, before mAP
consumes normalized objects. Standard post-hoc mAP then scores the existing
pixel-ready `gt`/`pred` arrays; it does not reparse raw compact text or infer a
variant from raw output. If a new post-change artifact lacks the template id, the
post-hoc evaluator fails preflight with an actionable metadata error.

### Decision: Use a breaking migration

Active configs/docs should be updated from `compact_full` to `compact`, and old
checkpoint metadata may be manually edited by the operator only for the
`compact_full` to `compact` case where serialized rows and the 1002-row adapter
contract already match. Closed variants require regenerated artifacts and
checkpoints trained or validated with the 1003 or 1004 row contract. The schema
should reject `compact_full` after this change.

## Risks / Trade-offs

- Old artifacts fail post-hoc parsing without manual metadata edits.
  Mitigation: document the breaking change and keep the error actionable.
- Multiple variant ids touch many surfaces.
  Mitigation: derive every downstream contract from a single registry and add
  focused roundtrip/config/artifact tests.
- Token-row validation can silently under-train closure-token variants if it is
  left coord-only.
  Mitigation: make required trainable row ids template-derived and fail fast on
  mismatch.
- Prompt examples can drift from renderers.
  Mitigation: resolve prompt row patterns from the same template contract used
  by training and inference renderers.
- Eval can become variant-dependent if mAP sees raw text.
  Mitigation: parse generated compact text during inference materialization and
  keep standard post-hoc mAP on the existing normalized object/geometry schema.
- Rollout surfaces may still import compact helpers.
  Mitigation: keep import compatibility and defer rollout behavioral tests until
  a rollout checkpoint exists for these template variants.

## Migration Plan

1. Update OpenSpec artifacts and record review convergence.
2. Rename active config/doc references from `compact_full` to `compact`.
3. Add the semantic template registry and make schema validation accept only the
   new ids.
4. Update render/parse, teacher-forcing target construction, prompt generation,
   token-row validation, inference artifact metadata, and evaluator preflight to
   resolve through `detection_template.id`.
5. Update tests for roundtrip rendering/parsing, config rejection of old fields,
   token-row counts, artifact metadata, and post-hoc mAP parsing across compact
   variants.
6. Sync the completed main change into the prefix-denoising worktree/branch only
   after the main implementation is approved and verified.

Rollback is a normal git revert of this change before training new checkpoints.
After checkpoints are trained with new semantic ids, rollback requires manual
metadata handling and is not treated as automatic compatibility support.

## Open Questions

None blocking. Implementation is authorized after the reviewed docs validate.
