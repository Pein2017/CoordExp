## 1. Contract and parser/rendering

- [x] 1.1 Add `compact_object_closed` to the semantic compact template registry
  with exact structural tokens, token-row ids, row count, prompt pattern, and
  strict parser metadata.
- [x] 1.2 Refactor compact row rendering so it accepts the resolved
  `object_field_order` and renders either desc-first or geometry-first segment
  order without adding free-form separator or closure booleans.
- [x] 1.3 Update strict compact parsing to require the configured field order
  and reject the opposite order for metric-bearing training/inference/eval
  paths.
- [x] 1.4 Update render spans, structural spans, trie-eligible spans, and
  render events so desc spans, bbox spans, closure-token spans, separator spans,
  and control spans remain correct for both field orders.
- [x] 1.5 Add render/parse roundtrip tests over each compact template id x
  `desc_first` / `geometry_first`, including the requested rich bbox-first row:
  `<|box_start|>coords<|box_end|><|object_ref_start|>{desc}<|object_ref_end|>`.

## 2. Config, prompts, and Stage-1 SFT data path

- [x] 2.1 Keep `custom.object_field_order` as the standard authored field-order
  source and route it into compact template rendering for standard Stage-1 SFT.
- [x] 2.2 Keep `detection_template.id` or `custom.detection_template_id` as the
  active semantic template source according to the existing Stage-1 SFT surface,
  and do not add a new CLI flag.
- [x] 2.3 Update dense prompt examples and prompt hashes to compose compact
  template id with object field order.
- [x] 2.4 Update `JSONLinesBuilder`, dense-caption helper surfaces, and standard
  SFT conversation construction so assistant target bytes match the resolved
  pair.
- [x] 2.5 Add tests proving standard Stage-1 SFT with
  `custom.object_field_order: geometry_first` renders bbox-first compact rows
  without using `TeacherForcingRollin`.

## 3. Cache, packing, and provenance

- [x] 3.1 Update encoded-sample cache fingerprints so changing only
  `object_field_order` changes cache identity for compact targets.
- [x] 3.2 Update static packing fingerprints so changing only
  `object_field_order`, changing only compact template id, or changing only
  `global_max_length` changes cache identity.
- [x] 3.3 Verify compact token-row requirements remain template-derived and
  independent of field order, including the new `compact_object_closed` 1003-row
  case.
- [x] 3.4 Update resolved training metadata and any packing metadata carriers to
  record both detection template id and object field order.

## 4. Inference, artifacts, evaluator, and docs

- [x] 4.1 Update inference prompt/parser policy to carry both compact template id
  and object field order.
- [x] 4.2 Persist both axes in `resolved_config.json`, `summary.json`, and
  `gt_vs_pred.jsonl` compact artifact records.
- [x] 4.3 Update post-hoc mAP preflight to require both axes for post-change
  compact artifacts while continuing to score normalized `gt`/`pred` objects.
- [x] 4.4 Add dependency-light artifact-preflight tests for missing
  `object_field_order` and missing `detection_template_id`.
- [x] 4.5 Add or update docs in `docs/data/PACKING.md` and
  `docs/training/STAGE1_OBJECTIVE.md` for the two-knob compact contract and the
  approval-gated final ablation.

## 5. Ablation configs and smoke launch

- [x] 5.1 Add desc-first and bbox-first production Stage-1 SFT config leaves for
  the final ablation using sorted ordering, packing length `12000`, LLM-only
  trainability, and
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- [x] 5.2 Add desc-first and bbox-first tiny smoke config overlays that preserve
  semantic identity while reducing runtime scale.
- [x] 5.3 Add config-load tests comparing the paired production leaves and smoke
  leaves, allowing only intentional run/output/logging/cache identity
  differences.
- [x] 5.4 After user approval, run targeted unit tests and then launch two smoke
  runs, one desc-first and one bbox-first.
- [x] 5.5 Record smoke artifact roots, config paths, parse/drop counters, cache
  identity, and any skipped expensive checks before production launch approval.

## 6. Review and implementation gates

- [x] 6.1 Run OpenSpec validation for this change after implementation edits:
  `openspec validate compact-template-field-order-ablation --type change --strict`.
- [x] 6.2 Run targeted tests for render/parse, prompts, cache fingerprints,
  packing fingerprints, Stage-1 config loading, inference provenance, and
  evaluator preflight.
- [x] 6.3 Run `git diff --check`.
- [x] 6.4 Do not launch production training until the user approves the
  implementation and smoke evidence.

## Implementation Evidence

- OpenSpec validation:
  `openspec validate compact-template-field-order-ablation --type change --strict`
  passed after implementation edits.
- Targeted archive-readiness tests:
  `python -m pytest tests/test_detection_compact_full_template.py tests/test_detection_template_variants.py tests/test_detection_template_registry.py tests/test_detection_template_parsing_eval.py tests/test_prompt_variants.py tests/test_encoded_sample_cache_runtime_config.py tests/test_stage1_static_packing_runtime_config.py tests/test_detection_training_config_contract.py tests/test_infer_compact_full_policy_contract.py tests/test_unified_infer_pipeline.py tests/test_latest_detection_view_metadata.py tests/test_decode_provenance_contract.py tests/tokens/test_token_roles.py -q`
  passed with `301 passed, 0 failed, 96 skipped`.
- Config/cache parity tests:
  `python -m pytest tests/test_stage1_static_packing_runtime_config.py tests/test_encoded_sample_cache_runtime_config.py tests/test_detection_training_config_contract.py -q`
  passed with `85 passed, 0 failed, 96 skipped`.
- Smoke artifact roots used as evidence:
  - desc-first:
    `temp/stage1_smoke_outputs/stage1/smoke/coco_bbox_max60_1024-coco80-desc_first-compact_object_box_closed-sorted-packed12k-natural_adjacent-pure_ce/smoke_2steps-pure_ce-coco80-desc_first-1024-compact_object_box_closed-sorted-packed12k-natural_adjacent-llm_only/v2-20260617-130249`
  - geometry-first:
    `temp/stage1_smoke_outputs/stage1/smoke/coco_bbox_max60_1024-coco80-geometry_first-compact_object_box_closed-sorted-packed12k-natural_adjacent-pure_ce/smoke_2steps-pure_ce-coco80-geometry_first-1024-compact_object_box_closed-sorted-packed12k-natural_adjacent-llm_only/v1-20260617-130711`
- Smoke cache identity roots used as evidence:
  - desc-first:
    `temp/static_packing_smoke/stage1_coco80_desc_first_1024_compact_object_box_closed_sorted_packed12k_natural_adjacent/global_max_length_12000`
  - geometry-first:
    `temp/static_packing_smoke/stage1_coco80_geometry_first_1024_compact_object_box_closed_sorted_packed12k_natural_adjacent/global_max_length_12000`
- Training smoke runs do not produce inference parse/drop counters; parse/drop
  compatibility is covered by strict parser, inference provenance, and evaluator
  preflight tests in the targeted archive-readiness suite above.
- Production launch approval was provided in chat before the geometry-first
  production training launch. The launch used `scripts/train.sh`, and the live
  log confirmed `object_ordering=sorted`,
  `object_field_order=geometry_first`,
  `detection_template_id=compact_object_box_closed`, global packing length
  `12000`, effective batch `64`, and token-adapter rows `1004`.
