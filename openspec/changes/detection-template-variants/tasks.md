## 1. Template Registry and Config Surface

- [x] 1.1 Add semantic detection template ids for `compact`,
  `compact_box_closed`, `compact_object_box_closed`,
  `compact_object_box_closed_lines`, and existing `stage1_json_pretty`.
- [x] 1.2 Add a single resolver that derives renderer, parser, prompt row
  pattern, row separator, required structural tokens, required trainable rows,
  and artifact metadata from `detection_template.id`.
- [x] 1.3 Update typed training and inference config validation so compact
  serialization is authored only through `detection_template.id`.
- [x] 1.4 Reject `compact_full` and independent compact parse/serialization/row
  separator knobs with actionable config errors.
- [x] 1.5 Rename active compact configs/docs from `compact_full` to `compact`
  without adding a lasting schema alias.
- [x] 1.6 Use the canonical YAML authoring shape `detection_template.id` for
  both training and inference configs; old inference keys such as
  `infer.detection_sequence_format`, `infer.row_separator`,
  `infer.compact_full_parse_mode`, and `infer.parsing.compact_full` are
  rejection-only migration diagnostics, not aliases.

## 2. Rendering, Parsing, and Teacher-Forcing Targets

- [x] 2.1 Implement canonical render/parse support for all compact variants,
  including final-newline behavior for `compact_object_box_closed_lines`.
- [x] 2.2 Make strict compact parsers reject missing closure tokens, unexpected
  closure tokens, and newline usage that contradicts the selected template id.
- [x] 2.3 Update template-specific rendered entries, separators, structural span
  projection, and render events for all compact variants.
- [x] 2.4 Update description and structural tokenization contexts so
  `<|object_ref_end|>`, `<|box_end|>`, and line separators are represented only
  for the templates that require them.
- [x] 2.5 Update teacher-forcing branch role sequences, target-IR atom positions,
  and trie/branch construction to match the selected template.
- [x] 2.6 Verify final-newline handling for `compact_object_box_closed_lines`
  remains separate from the Qwen `<|im_end|>` chat-stop contract.
- [x] 2.7 Add render/parse roundtrip tests for `stage1_json_pretty` and all four
  compact template ids.
- [x] 2.8 Add teacher-forcing token-id, structural-role, terminal-span, and
  target-IR alignment tests for each compact template id.
- [x] 2.9 Propagate the resolved template id or contract from dataset/runtime
  call sites into teacher-forcing target construction so rendered assistant
  bytes, target IR atoms, span masks, and stop atoms come from one contract.

## 3. Token Rows and Offset Adapter

- [x] 3.1 Validate that each structural token required by the selected template
  resolves to a single tokenizer id.
- [x] 3.2 Derive compact token-row requirements from `detection_template.id`:
  1002 rows for `compact`, 1003 for `compact_box_closed`, and 1004 for the
  object-box-closed variants; `stage1_json_pretty` must not require compact
  token-row adaptation.
- [x] 3.2a Add native structural token-id constants for `<|object_ref_end|>` and
  `<|box_end|>` alongside existing Qwen native constants, and expose a
  template-derived row-id resolver for exact trainable/adapted rows.
- [x] 3.3 Update token-row schema/runtime validation to fail fast when the
  selected compact template's exact structural rows are missing, extra,
  duplicated, or not trainable.
- [x] 3.4 Update token_embeddings_adapter row selection/checkpoint validation so adapter
  checkpoints require the exact template-derived row set with no missing, extra,
  or duplicate ids.
- [x] 3.5 Validate `coord_ids`, embedding offset rows, lm-head offset rows when
  present, and `modules_to_save` against the selected template id.
- [x] 3.6 Add targeted tests for the 1002, 1003, and 1004 token-row contracts,
  including missing closure rows, extra rows, duplicates, tensor shape mismatch,
  and the renamed `compact` path not bypassing validation.
- [x] 3.7 Include `detection_template.id` in encoded-sample cache fingerprints
  and the training resolved-config/provenance carrier so changing only the
  template id cannot reuse stale rendered training examples.

## 4. Prompts, Inference, Artifacts, and Evaluation

- [x] 4.1 Update dense prompt resolution, shared inference runtime prompt policy,
  and prompt hashing to include `detection_template.id` and to render
  template-specific compact examples.
- [x] 4.2 Update inference config/runtime so generated compact text is parsed by
  the strict parser derived from `detection_template.id`.
- [x] 4.3 Persist the resolved `detection_template.id` in `resolved_config.json`,
  `summary.json`, shared runtime parser/provenance metadata, backend request
  provenance, and prompt fingerprints; persist `detection_template_id` in
  `gt_vs_pred.jsonl` records consumed by post-hoc mAP.
- [x] 4.4 Update post-hoc mAP preflight and artifact discovery to require
  template metadata for post-change artifacts, fail if it is missing, ignore old
  `compact_full_parse_mode` or `parsing.compact_full.mode` as source-of-truth,
  and score the existing normalized object/geometry schema without reparsing raw
  compact text.
- [x] 4.5 Add synthetic post-hoc mAP/evaluator tests covering each compact
  variant with equivalent normalized predictions; place metadata-preflight
  rejection tests in dependency-light files so they do not disappear when
  pycocotools is unavailable.
- [x] 4.6 Add backend prompt parity tests showing HF, local vLLM, and
  server-backed vLLM use byte-equivalent final system/user prompt payloads,
  prompt hash, and resolved template id for the same config, including
  closure-token and final-newline expectations.

## 5. Stage-2 and Rollout Boundaries

- [x] 5.1 Keep Stage-2 and rollout imports/config resolution from breaking when
  compact template helpers are renamed or moved.
- [x] 5.2 Do not add rollout behavioral tests for these template variants in this
  change because there is no rollout checkpoint yet.
- [x] 5.3 Leave rollout-specific parser/appender behavior as a future checkpointed
  validation surface unless implementation discovers an immediate import or
  schema break.
- [x] 5.4 Parse
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` and
  assert the comparator contract: `objective.id: teacher_forcing`,
  `objective.variant: random_permutation_et_rmp_ce`,
  `detection_template.id: compact`, `data.object_ordering:
  random_permutation`, `objective.state_weighting: uniform_permutation`,
  `objective.normalization: semantic_image_bucket_balanced`, and top-level
  `objective.trie_*` support/balance weights.
- [ ] 5.5 Add prefix-rollin config/schema tests that accept all compact semantic
  template ids, reject `stage1_json_pretty`, and reject obsolete flat trie
  aliases with errors pointing to nested `objective.target` /
  `objective.boundary`; do not add checkpointed rollout behavior tests.

## 6. Verification and Handoff

- [x] 6.1 Run targeted unit tests for config validation, template roundtrips,
  teacher-forcing token alignment, token-row contracts, inference artifact
  metadata, post-hoc mAP preflight/scoring, and Stage-2 rollout import/config
  boundaries.
- [x] 6.2 Verify metadata-only migration is allowed only for legacy
  `compact_full` to `compact` artifacts/checkpoints with matching 1002-row
  contracts; closed variants require regenerated output or validated 1003/1004
  rows.
- [x] 6.3 Run `openspec validate detection-template-variants --type change
  --strict`.
- [x] 6.4 Run `git diff --check`.
- [x] 6.5 Document any skipped broad or hardware-heavy checks with the reason.
- [x] 6.6 After main implementation is approved and verified, sync the completed
  change only into the prefix-denoising worktree/branch requested by the user.
