## 1. CoordJSON Transpiler (CoordJSON → strict JSON)

- [ ] 1.1 Add `src/utils/coordjson_transpiler.py` with strict+salvage conversion APIs and CoordTok validation (`<|coord_k|>` → int k).
- [ ] 1.2 Implement strict-mode validation (fail-fast) for cooked SFT/GT: closed record schema, exactly one geometry key, geometry arity checks, non-empty `desc`, and `custom.object_field_order` key-order enforcement.
- [ ] 1.3 Implement salvage-mode conversion for rollout preds:
  - extract only the first valid top-level `{"objects": [...]}` container (ignore leading/trailing junk; ignore later containers),
  - make container-boundary scanning string-aware so braces/brackets inside `desc` strings do not affect depth tracking,
  - drop invalid records and drop an incomplete final record on truncation,
  - on sample-level parse-fail (no valid container), report a parse-fail flag/counter and return safe empty strict JSON `{"objects": []}`,
  - preserve the valid prefix without token-inserting repair (no adding braces/quotes/commas).
- [ ] 1.4 Add unit tests for strict mode (fail-fast) and salvage mode covering:
  - drop-invalid records + truncated-tail,
  - leading/trailing junk around the container + multiple containers (first-only selection),
  - string-aware boundary detection when `desc` contains braces/brackets,
  - sample-level parse-fail returning `{"objects": []}`,
  - bbox and poly (`tests/test_coordjson_transpiler.py`).

## 2. Cooked Target Serialization (Stage-1 / SFT cooking)

- [ ] 2.1 Update `src/datasets/builders/jsonlines.py` to render assistant targets as CoordJSON `{"objects": [...]}` (replace `"object_{n}"` keys).
- [ ] 2.2 Emit bare coord tokens in geometry arrays (no quotes) by converting raw JSONL geometry strings like `"<|coord_12|>"` into CoordTok literals `<|coord_12|>` in the serialized assistant text.
- [ ] 2.3 Enforce closed record schema in cooked targets (no extra keys; only `{bbox_2d|poly, desc}`) and validate cooked outputs via transpiler strict mode during build.
- [ ] 2.4 Preserve object instance ordering behavior (`custom.object_ordering`) while changing only the container shape and per-record key order.
- [ ] 2.5 Update prompt resolution in `src/config/prompts.py` to instruct CoordJSON + `{"objects": [...]}` output and to align wording with `custom.object_field_order`.
- [ ] 2.6 Ensure optional dataset-side metadata (e.g., `poly_points`, `line_points`, per-object `metadata`) is NEVER emitted in assistant records or rendered assistant CoordJSON text; assistant records are closed schema `{bbox_2d|poly, desc}` only. Add regression coverage.
- [ ] 2.7 Add a tiny shared assistant-JSON dump helper (e.g., under `src/utils/`) that centralizes the canonical ms-swift-compatible style (`ensure_ascii=False`, `indent=None`, `separators=(", ", ": ")`).
- [ ] 2.8 Route assistant-output-like deterministic serialization through the shared helper (at minimum: `src/datasets/builders/jsonlines.py`, stage-2 teacher-forced payload construction, and FN-append serialization paths) to prevent spacing drift.

## 3. Rollout Parsing + Stage-2 Teacher-Forced Construction

- [ ] 3.1 Update `src/trainers/rollout_matching/parsing.py` to parse the top-level `{"objects": [...]}` container and enumerate predicted records in array appearance order (no re-serialization).
- [ ] 3.2 Replace object-key-centric logic in `src/trainers/rollout_matching/parsing.py` / `src/trainers/rollout_matching/contracts.py` (e.g., `object_<N>` regex, max-object-index extraction) with `objects[]` index semantics and append-ready prefix cuts inside the `"objects"` array.
- [ ] 3.3 Update `src/trainers/rollout_matching_sft.py` and `src/trainers/stage2_ab_training.py` to build `Y_train` by suffix-trimming a rollout prefix to an append-ready `{"objects": [...` prefix and FN-appending unmatched GT records as additional array elements, then closing with `]}`.
- [ ] 3.4 Ensure key-order ablation is enforced: cooked targets fail-fast; rollout records that violate the configured `custom.object_field_order` are treated as invalid and dropped.
- [ ] 3.5 Update any downstream list consumers (e.g., `src/trainers/rollout_matching/matching.py`) to rely on list order and indices rather than `"object_{n}"` keys.
- [ ] 3.6 Implement deterministic invalid-rollout handling for training: when a rollout cannot be parsed into an append-ready `{"objects": [` prefix, treat it as having zero predicted objects, fall back to prefix `{"objects": [`, and increment an `invalid_rollout` / parse-fail metric (no crash, no silent skip).
- [ ] 3.7 Add regression coverage for token-internal suffix cuts on fused closing tokens (e.g., token text contains `]}` / `}]}`) to ensure suffix-only trimming can still produce an append-ready prefix.
- [ ] 3.8 Implement and wire Stage-2 Channel-B metrics as required by spec: `stage2_ab/channel_b/invalid_rollout`, `stage2_ab/channel_b/strict_drop/N_valid_pred`, `stage2_ab/channel_b/strict_drop/N_drop_invalid`, and deterministic `stage2_ab/channel_b/strict_drop/reason/<bucket>` counting with fixed precedence.

## 4. Remove `repeat_terminate` (code/config/docs/tests)

- [ ] 4.1 Delete the repeat-terminate implementation in `src/common/repeat_terminate.py` and remove all call sites (e.g., `src/trainers/rollout_matching/preflight.py`, `src/trainers/rollout_matching_sft.py`, `src/trainers/stage2_ab_training.py`).
- [ ] 4.2 Remove vLLM repeat-terminate server plugin wiring (`scripts/vllm_repeat_terminate_plugin.py`) and any server-mode dependency on `coordexp.repeat_terminate_triggered`.
- [ ] 4.3 Remove `rollout_matching.repeat_terminate` from YAML configs (e.g., `configs/stage2_ab/base.yaml`, `configs/dlora/stage2_rollout_matching_ckpt3106.yaml`) and update config schema/validation accordingly.
- [ ] 4.4 Update or remove repeat-terminate tests and metrics expectations (`tests/test_repeat_terminate.py`, `tests/test_vllm_repeat_terminate_plugin.py`, `tests/test_rollout_matching_sft.py`, `tests/test_stage2_ab_training.py`, `tests/test_stage2_ab_vllm_server_mode_smoke.py`).
- [ ] 4.5 Update docs referencing repeat-terminate (`docs/training/METRICS_LOSSES.md`, `docs/training/STAGE2_RUNBOOK.md`, `progress/stage_2_symptom.md`).
- [ ] 4.6 Update `scripts/train_stage2.sh` to remove hard dependencies on repeat-terminate (plugin file existence checks and `/infer/` activation probes that expect `coordexp.repeat_terminate_triggered`).

## 5. Tooling + Eval/Viz Adapters

- [ ] 5.1 Update `scripts/tools/inspect_chat_template.py` to render and inspect the new CoordJSON assistant output (`{"objects": [...]}`) and to optionally show the strict-JSON-transpiled view for debugging.
- [ ] 5.2 Update eval/viz code paths under `src/eval/` / `src/infer/` (and any JSONL artifact writers) that assume `"object_{n}"` keys to iterate over `objects[]` by index.
- [ ] 5.3 Update analysis scripts that parse `"object_{n}"` (e.g., `scripts/analysis/report_rollout_stability.py`) to support `objects[]`.
- [ ] 5.4 Standardize readability of JSON dump artifacts:
  - JSONL dumps MUST remain one-JSON-object-per-line (no indentation/newlines).
  - When dumping assistant-output-like content, dumps SHOULD include both:
    - the rendered assistant CoordJSON text (exactly as rendered for chat-template ingestion), and
    - the transpiled strict JSON object view (as a structured object, not as a nested JSON string), to reduce escape-noise.
  - JSON dumps intended for inspection MUST use `ensure_ascii=False` and the repo’s canonical separators `(", ", ": ")`, and MUST NOT use `sort_keys=True` for assistant-output-like structures (preserve configured per-record key order for ablation readability).
- [ ] 5.5 Make CoordJSON→strict-JSON transpile the explicit parsing boundary for assistant-output-like strings in eval/infer/tooling paths: remove direct `json.loads` on raw CoordJSON assistant text and route through the transpiler first.

## 6. Documentation + Progress Alignment

- [ ] 6.1 Update `progress/full_idea.md` to replace `"object_{n}"` invariants with `{"objects": [...]}`, update examples, and update FN-append pseudocode to append array elements and close with `]}`.
- [ ] 6.2 Update dataset contract docs (`docs/data/JSONL_CONTRACT.md`, `docs/data/README.md`) to clarify: raw JSONL remains standard JSON with quoted coord-token strings; model-facing assistant outputs are CoordJSON; internal parsing uses strict JSON with integer geometry bins.
- [ ] 6.3 Update other progress docs that show assistant output examples (e.g., `progress/pretrain/first_stage.md`) to use the new container and CoordJSON coord literals.
- [ ] 6.4 Update training runbook docs that describe FN injection and parsing (e.g., `docs/training/STAGE2_RUNBOOK.md`) to use the `{"objects": [...]}` array container (no `object_N` keys).
- [ ] 6.5 Update repo documentation examples that still show the old container (e.g., `src/README.md`) to use CoordJSON + `{"objects": [...]}`.
- [ ] 6.6 Update configs/prompts that describe poly as vertex pairs to the flat poly list contract (e.g., `configs/fusion/lvis_bbox_only_vs_poly_prefer_1to1.yaml`).

## 7. Regression Tests + Smoke Checks

- [ ] 7.1 Update dataset/runtime contract tests that assume `"object_{n}"` (`tests/test_dataset_runtime_contracts.py`) to assert `{"objects": [...]}` and per-record key order under `custom.object_field_order`.
- [ ] 7.2 Update rollout-matching tests (`tests/test_rollout_matching_sft.py`) to cover: `objects[]` parsing, prefix trimming inside `"objects"` array, FN append into array, and order-violation dropping for rollouts.
- [ ] 7.3 Update stage-2 AB tests (`tests/test_stage2_ab_training.py`) to cover teacher-forced payload building and matching under the new container, and to remove repeat-terminate metrics.
- [ ] 7.4 Add a minimal chat-template inspection regression via `conda run -n ms python scripts/tools/inspect_chat_template.py ...` on a known JSONL row to confirm the emitted assistant text is CoordJSON and the transpiled strict JSON parses.
- [ ] 7.5 Add golden-string tests for canonical CoordJSON serialization (bbox + poly; geometry_first + desc_first) to prevent whitespace/separator drift from silently changing tokenization.
- [ ] 7.5.1 Add helper-level regression tests asserting byte-exact serialization style (single-line, separators `", "`/`": "`, no pretty indentation) so Qwen3/Qwen3-VL whitespace-sensitive tokenization remains stable.
- [ ] 7.6 Add a grep-driven inventory gate: require `rg -n '\"object_'` across `src/ tests/ scripts/ configs/ docs/ progress/` to be empty or explicitly justified (legacy fixtures only), and update currently-known failing files (e.g., `tests/test_dense_caption_prompt_override.py`, `tests/test_token_types_coord_quotes.py`, `tests/test_tokenizer_decode_frame_canary.py`, `tests/test_coord_utils.py`, `tests/test_stage2_ab_packing_mask_gradients.py`).
- [ ] 7.7 Add Stage-2 metric assertions for Channel-B diagnostics: verify presence of `stage2_ab/channel_b/invalid_rollout`, strict-drop `N_valid_pred`, strict-drop `N_drop_invalid`, and deterministic `reason/<bucket>` behavior for multi-violation dropped records.
- [ ] 7.8 Add an inventory gate for assistant-output parsing boundaries: grep for direct `json.loads(` on raw assistant CoordJSON text in `src/ scripts/ tests/` and require justification or transpiler usage.
