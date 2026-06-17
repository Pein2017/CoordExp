## Why

Detection training, inference, and evaluation currently treat the compact detection
serialization as a mostly implicit `compact_full` contract. We need several
compact wrapper variants for ablation and anchoring experiments, but the public
surface should stay small: semantic template metadata must drive rendering,
parsing, prompt text, token-row adaptation, artifacts, and post-hoc metrics.

Supersession note: `compact-template-field-order-ablation` amends this change
for bbox-first compact rows. In that follow-on contract, `detection_template.id`
owns closure/separator/template-family behavior, while `custom.object_field_order`
owns desc-first versus geometry-first row layout.

## What Changes

- Add semantic detection template ids:
  - `compact`
  - `compact_box_closed`
  - `compact_object_box_closed`
  - `compact_object_box_closed_lines`
- Keep `stage1_json_pretty` supported as a separate non-compact template id.
- Treat `compact_full` as the old name for `compact` during active config/doc
  cleanup, but do not keep it as an accepted schema alias after the change.
- **BREAKING, AMENDED**: make `detection_template.id` the authored source of
  truth for compact template family, closure tokens, row separator, parser
  family, and structural token rows; reject independent authored parse modes,
  serialization policies, row separators, or compact-format aliases that
  duplicate the selected template. The follow-on
  `compact-template-field-order-ablation` change keeps `custom.object_field_order`
  as the authored source for desc-first versus geometry-first row layout.
- Derive renderer, parser, prompt row pattern, row separator, required structural
  tokens, trainable token rows, artifact metadata, inference materialization,
  and post-hoc mAP preflight from the selected semantic template id.
- Require template-derived token-row adaptation to support the selected compact
  template:
  - `compact`: 1002 rows, including coord tokens plus
    `<|object_ref_start|>` and `<|box_start|>`
  - `compact_box_closed`: 1003 rows, adding `<|box_end|>`
  - `compact_object_box_closed` and `compact_object_box_closed_lines`: 1004 rows,
    adding `<|object_ref_end|>` and `<|box_end|>`
- Require inference artifacts to persist the resolved `detection_template.id` so
  generated compact text is materialized by contract and post-hoc mAP accepts
  the normalized predictions without heuristics or CLI overrides.
- Do not add rollout behavioral tests in this change; rollout parsing/rendering
  should remain import-safe, but no rollout checkpoint exists yet for these
  variants.

## Capabilities

### New Capabilities

- `detection-template-variants`: Defines semantic detection template ids and the
  derived render/parse/prompt/token-row/artifact contract for compact detection
  wrapper variants.

### Modified Capabilities

- `stage1-detection-objectives`: Update the compact recursive detection contract
  from `compact_full` to `compact`, and make compact structural token rows
  template-derived for the new variants.
- `token_embeddings_adapter`: Extend the adapter contract from coord-only rows
  to the selected template's required structural rows when compact detection
  token-row adaptation is enabled.
- `dataset-prompt-variants`: Include the selected detection template id in dense
  prompt resolution so prompt examples and row separators match the renderer.
- `inference-engine`: Persist resolved detection template metadata in inference
  artifacts and parse generated compact output using the artifact/config
  template id.
- `shared-inference-runtime`: Make the shared parser/provenance seam carry the
  resolved detection template id as derived parser policy metadata.
- `detection-evaluator`: Make post-hoc mAP require resolved template metadata
  for post-change artifacts while scoring the existing normalized
  object/geometry evaluation schema.

## Impact

- Affected schema and config surfaces include detection training configs,
  inference configs, active compact detection examples, token-row validation, and
  resolved artifact metadata.
- Affected runtime surfaces include detection template rendering/parsing,
  teacher-forcing target construction, dense prompt construction, offset-adapter
  row selection, inference artifact materialization, and evaluator post-hoc mAP
  parsing.
- Existing checkpoints or old artifacts that still say `compact_full` or omit
  `detection_template.id` must be manually edited or regenerated; this change
  does not provide backward-compatible aliases or artifact guessing. Metadata-only
  edits are valid only for legacy `compact_full` to `compact` artifacts whose
  serialized bytes and 1002-row adapter contract already match `compact`.
- Tokenizer migration is not required because `<|object_ref_end|>` and
  `<|box_end|>` already exist in the Qwen tokenizer vocab, but trainable row
  validation must require their rows for variants that use them.
