## 1. Config and contract plumbing

- [x] 1.1 Add `custom.object_field_order` to typed config with default `desc_first` and allowed values `{desc_first, geometry_first}`.
- [x] 1.2 Ensure loader rejects invalid values with clear actionable errors.
- [x] 1.3 Thread resolved value through stage-1 dataset builders and stage-2 trainers via existing config plumbing (no new CLI args).

## 2. Stage-1 serialization

- [x] 2.1 Update `JSONLinesBuilder` object payload construction to honor `custom.object_field_order`.
- [x] 2.1a Ensure `geometry_first` is implemented as "`bbox_2d`/`poly` key before `desc`" (no new `geometry` key).
- [x] 2.2 Ensure `assistant_payload` and rendered assistant text use the same order.
- [x] 2.2a Ensure rendered assistant text ordering is validated at the chat-template boundary (not only builder dict-level output).
- [x] 2.3 Confirm object sequence remains governed by `custom.object_ordering` only.

## 3. Stage-2 serialization parity

- [x] 3.1 Update Channel-A teacher-forced payload construction to honor `custom.object_field_order`.
- [x] 3.2 Update Channel-B FN append serializer (`serialize_append_fragment`) to honor `custom.object_field_order`.
- [x] 3.2a Keep wording/implementation explicit that `geometry_first` means existing geometry key first (`bbox_2d` today; `poly` when present).
- [x] 3.3 Keep object key numbering and appearance-order matching semantics unchanged.

## 4. Prompt alignment

- [x] 4.1 Add geometry-first dense prompt variants that explicitly request geometry before desc.
- [x] 4.2 Select prompt variants using `custom.object_field_order` while preserving existing object-ordering wording (`sorted` / `random`).
- [x] 4.3 Keep default prompt behavior unchanged for `desc_first`.

## 5. Tests

- [x] 5.1 Add/extend config tests for `custom.object_field_order` parsing and fail-fast behavior.
- [x] 5.2 Add/extend stage-1 builder tests asserting `desc_first` vs `geometry_first` object field order.
- [x] 5.3 Add/extend stage-2 tests asserting field order in:
  - Channel-A payload serialization,
  - Channel-B FN append fragment serialization.
- [x] 5.4 Add/extend prompt-selection tests for geometry-first wording.
- [x] 5.5 Add/extend tests ensuring assistant outputs for `poly` objects do not emit `poly_points`.

## 6. Validation commands

- [x] 6.1 Run targeted config tests:
  - `conda run -n ms python -m pytest -q tests/test_custom_extra_merge.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
- [x] 6.2 Run serialization-focused tests:
  - `conda run -n ms python -m pytest -q tests/test_dataset_runtime_contracts.py`
  - `conda run -n ms python -m pytest -q tests/test_rollout_matching_sft.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- [x] 6.3 Optional smoke check using conversation dumps:
  - run one short `scripts/train.sh` config with `custom.dump_conversation_text: true` and `custom.object_field_order: geometry_first`,
  - verify dumped assistant payload field order and unchanged object instance ordering.
- [x] 6.4 Run chat-template rendering audit on a representative sample:
  - `PYTHONPATH=. conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path/to/data.jsonl> --index 0`
  - verify rendered assistant JSON text reflects `custom.object_field_order` and matches structured payload ordering.

## 7. Reproducibility and docs

- [x] 7.1 Ensure run/config logs preserve `custom.object_field_order` for ablation traceability.
- [x] 7.2 Update relevant docs/spec notes to state this change is infrastructure-only (no claimed quality win).
