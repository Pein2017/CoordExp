# Wave 1 Source Study And Gate Resolution

## Scope

This note resolves the source-study and pre-implementation gates for
`build-coordexp-swift-inference-infra`. It records evidence for the first
implementation wave without creating `src/infer.py`, `src/inference/`, or any
new source implementation.

Source implementation remains blocked until the user explicitly approves the
implementation kickoff after reviewing this note and the active OpenSpec change.

## Status

- P0 blockers: none found.
- P1 gates resolved into implementation constraints: legacy `src.infer` tests,
  stale legacy inference docs, PEFT adapter fatal checks, HF score trace
  alignment, no-resize Qwen image processing, and minimal V1 mAP consumer
  ownership.
- OpenSpec contradiction requiring immediate contract rewrite: none found.
- Docs contradiction carried as implementation risk: at source-study time,
  current `docs/eval` and `docs/ARTIFACTS.md` still described legacy inference
  routes and artifact names. These docs were later patched after implementation
  to route through CoordExp-Swift first.
- Implementation approval: later granted.

## Local Environment Evidence

- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`.
- CodeGraph index for this worktree: `1309` files, `23068` nodes, `53555`
  edges.
- Installed packages observed by source/probe lanes:
  - `transformers==4.57.1`
  - `peft==0.17.1`
  - `torch==2.9.1`
  - `accelerate==1.10.1`
  - `tokenizers==0.22.0`
- Adapter/source-gate test slice:
  `pytest tests/adapters/test_dora_setup.py tests/adapters/test_source_gates.py -q`
  returned `38 passed`.
- Current implementation absence check: no live `src/infer/` package and no
  live `src/inference/` package existed before implementation.

## Owner Boundary Findings

The current rebuilt training code already has useful owners, but many of them
are training-shaped:

- `src/config/loader.py::load_train_config` returns `ResolvedTrainConfig`.
- `src/config/writer.py::write_resolved_config_artifacts` currently accepts
  `ResolvedTrainConfig`.
- `src/config/paths.py::resolve_run_directory` accepts `TrainConfig`, and
  path-resolution helpers include training path fields.
- `src/qwen/loading.py::load_qwen_components` and `_load_model` are shaped
  around training config.
- `src/artifacts/manager.py` imports training schedule/config types.
- `src/adapters/source_gates.py` and `src/adapters/dora.py` are the correct
  owner surfaces for DoRA setup, target discovery, source-study gates, and
  adapter validation.

Implementation constraint:

- Inference-facing code SHALL NOT import or call `TrainConfig`,
  `ResolvedTrainConfig`, `ResolvedStepSchedule`, `load_train_config()`, or
  unallowlisted `src.training.*`.
- V1 has an empty `src.training.*` allowlist.
- Shared behavior should be generalized in the owning modules only when it can
  be made type-neutral. Inference runtime should coordinate those owner APIs,
  not copy training-specific objects.

## Qwen Processor, Tokenizer, And Special Token Evidence

Real local model path used for probes:
`/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.

Lightweight processor/model probe:

- Transformers config class: `Qwen3VLConfig`.
- Processor class: `Qwen3VLProcessor`.
- Image processor class: `Qwen2VLImageProcessorFast`.
- Processor `patch_size=16` equals `vision_config.patch_size=16`.
- Processor `merge_size=2` equals `vision_config.spatial_merge_size=2`.
- Processor `temporal_patch_size=2` equals `vision_config.temporal_patch_size=2`.

Tokenizer atom probe:

| token | ids | single atom |
| --- | --- | --- |
| `<|object_ref_start|>` | `[151646]` | yes |
| `<|object_ref_end|>` | `[151647]` | yes |
| `<|box_start|>` | `[151648]` | yes |
| `<|box_end|>` | `[151649]` | yes |
| `<|im_end|>` | `[151645]` | yes |
| `<|coord_100|>` | `[151770]` | yes |
| `<|coord_200|>` | `[151870]` | yes |
| `<|coord_300|>` | `[151970]` | yes |
| `<|coord_400|>` | `[152070]` | yes |

Additional tokenizer facts:

- `pad_token_id=151643`.
- `eos_token_id=151645`, matching `<|im_end|>`.

Implementation constraint:

- V1 inference stop behavior must use Qwen `<|im_end|>`, not
  `<|endoftext|>`.
- Raw traces must preserve special tokens. Parser-facing views may strip the
  terminal `<|im_end|>` only with explicit trace/provenance evidence.
- No-resize inference must validate processor/model vision identity and image
  dimensions before processor reshape.

## Transformers Qwen3-VL And Generation Evidence

Installed upstream source studied:

- `transformers/generation/utils.py`
- `transformers/models/qwen3_vl/processing_qwen3_vl.py`
- `transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`

Key findings:

- HF generation scores are opt-in. `GenerationConfig()` defaults include
  `return_dict_in_generate=False` and `output_scores=False`.
- `GenerateDecoderOnlyOutput.scores` are per-generation-step tensors shaped
  `(batch_size, vocab_size)` and exist only when score output is requested.
- HF stores post-logits-processor `next_token_scores`; normalized transition
  logprobs must be gathered with explicit alignment to generated suffix tokens.
- HF examples slice generated tokens from `outputs.sequences[:, input_length:]`
  and use `compute_transition_scores(..., normalize_logits=True)` for transition
  scores.
- Stop/pad handling is a correctness boundary. Finished rows append
  `pad_token_id` after EOS, so V1 must retain the stop token and exclude or
  explicitly flag post-stop padding.
- Qwen3-VL processor returns text inputs plus `pixel_values`,
  `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw`.
- Qwen3-VL uses the Qwen2-VL image processor family; upstream defaults include
  `do_resize=True`. CoordExp inference must override and prove `do_resize=False`.
- Qwen3-VL forward validates visual placeholder count, scatters image/video
  embeds into text embeddings, computes prefill MRoPE positions, stores
  `rope_deltas`, and then derives generated-step positions from cache position
  plus deltas.

Implementation constraints:

- HF backend must pass `return_dict_in_generate=True` and `output_scores=True`
  whenever scored inference is requested.
- Constant fallback scores are rejected for V1 mAP-style scored inference.
- Trace extraction must test variable prompt widths, generated suffix indexing,
  `<|im_end|>` retention, and post-stop padding exclusion.
- A tiny real Qwen generate trace remains required during backend implementation.

## PEFT And Adapter Evidence

Installed PEFT source studied:

- `peft/peft_model.py`

Key findings:

- `PeftModel.load_adapter(...)` returns a concrete `load_result`.
- A loaded adapter is not automatically active; `set_adapter(...)` must be
  called.
- `load_result.missing_keys` is filtered to adapter/tuner-specific keys before
  being returned.
- `get_model_status()` reports `enabled`, `active_adapters`,
  `merged_adapters`, and `requires_grad`, and can report `"irregular"` for
  inconsistent adapter states.

Current local owner evidence:

- `src/adapters/dora.py` already records adapter mode/type/name/path/base,
  target modules, PEFT config, trainables, and package versions.
- Existing DoRA tests and source-gate tests pass.
- Current local adapter code validates important training-side properties, but
  a V1 inference reload/status receipt still needs to capture PEFT
  missing/unexpected keys, activate the adapter, and check status.

Implementation constraints:

- Adapter load policy belongs in `src/adapters/`, not inline in
  `src/inference/runtime.py`.
- Inference runtime may coordinate an adapter-owner result, but warning-only
  PEFT load paths are not acceptable.
- Fatal adapter checks must cover base identity, captured missing keys,
  captured unexpected keys, `set_adapter`, `get_model_status().enabled`, active
  adapter names, irregular status fields, and unexpected merged state.

## Special Token Embedding Delta Evidence

Existing source study and owner code establish the V1 mechanism:

- V1 uses a CoordExp-owned additive embedding delta wrapper, not PEFT
  `TrainableTokens`, for selected special-token embeddings.
- Saved identity metadata includes base model path, base config SHA, tokenizer
  SHA, token strings, token ids, dtype, tensor key, and tensor shape.
- Loader validation already supports expected base/tokenizer identity checks.
- The selected token group is the four object/box wrappers plus the 1000
  coordinate tokens.

Implementation constraints:

- Inference must validate embedding-delta identity before generation.
- It must record whether the run is base-only, base+adapter, or
  base+adapter+embedding-delta.

## Legacy Inference Reference Study

Reference-only legacy surfaces inspected:

- `/data/CoordExp/src/infer/*`
- `/data/CoordExp/configs/infer/*`
- `docs/eval/CONTRACT.md`
- `docs/eval/WORKFLOW.md`
- `docs/ARTIFACTS.md`

Accepted lessons:

- Legacy HF code already demonstrates the need for `return_dict_in_generate`,
  `output_scores`, generated-length checks, and per-token logprob extraction.
- Legacy checkpoint helpers contain useful adapter and token-embedding identity
  lessons.
- Eval docs correctly distinguish raw `gt_vs_pred.jsonl` from benchmark-bearing
  `gt_vs_pred_scored.jsonl`.
- For mAP claims, V1 must consume scored artifacts, preserve row identity and
  GT fields, and write portable score provenance.

Rejected legacy behavior:

- Do not recreate the legacy `src/infer/` package.
- Do not keep `configs/infer/*` as canonical V1 config authority.
- Do not use `scripts/run_infer.py` as the V1 public entry.
- Do not keep top-level `resolved_config.json`; V1 uses
  `configs/resolved.json` and `configs/resolved.yaml`.
- Do not use constant-score compatibility materialization for V1 scored mAP.
- Do not bring confidence post-op, guarded artifacts, LVIS proxy bundles,
  Oracle-K, visual galleries, or Stage-2 eval-step materialization into minimal
  V1 unless separately approved.
- Do not bridge to the legacy evaluator by default. Minimal V1 should rebuild a
  focused `src.eval` detection consumer for `gt_vs_pred_scored.jsonl`.

## Legacy Test Triage

Mechanical inventory:

- `35` live test files directly import or reference `src.infer`.
- `1` live test file references `configs/infer` without a direct `src.infer`
  import:
  `tests/analysis/sorted_random_no_newline_phenotype/test_rollout_phenotype.py`.
- Broad legacy strings appear in more files, but many are analysis or retired
  training/objective checks rather than V1 inference targets.

Port-to-new `tests/inference/` or `tests/eval/`:

- `tests/test_decode_backend_trace_contract.py`
- `tests/test_infer_decode_request_mapping.py`
- `tests/test_detection_prompt_input_codec.py`
- `tests/test_parser_policy_parity.py`
- `tests/test_infer_artifact_metadata.py`
- `tests/test_decode_provenance_contract.py`
- `tests/test_score_policy_fingerprint.py`
- `tests/test_detection_eval_output_parity.py`
- `tests/test_qwen_generation_contract.py`
- `tests/test_inference_runtime_backend_facade.py`
- `tests/test_infer_batch_decoding.py`
- `tests/test_infer_checkpoint_resolution.py`
- `tests/test_prompt_parity_guard.py`
- `tests/test_infer_compact_full_policy_contract.py`
- `tests/test_detection_scene_phase4_infer_eval_projection.py`
- `tests/test_removed_decode_constraint_surface_contract.py`

Retain with explicit out-of-scope or legacy allowlist reason:

- `tests/test_stage2_rollout_runtime.py`
- `tests/test_stage2_ab_vllm_server_mode_smoke.py`
- `tests/test_vllm_server_rollout_contract.py`
- `tests/test_rollout_matching_decoding_cfg.py`
- `tests/test_stage1_detection_eval.py`
- `tests/test_stage2_ab_prompt_alignment_contract.py`
- `tests/test_stage2_rollout_correction_target_boundary.py`
- `tests/test_prepare_samples_for_rollout_vllm_multimodal.py`
- `tests/test_swift_rollout_endpoints_contract.py`
- `tests/test_vllm_server_adapter_payload.py`
- `tests/test_vllm_server_multimodal_payload_contract.py`
- `tests/test_rollout_hf_length_coherence_gate.py`

Remove, archive, or split later with rationale:

- `tests/test_run_infer_legacy_shared_runtime.py`
- `tests/test_infer_pipeline_shared_decode_request.py`
- `tests/test_infer_layout_import_gates.py`
- `tests/test_gt_vs_pred_visualization.py`
- `tests/test_oracle_k_eval.py`
- `tests/test_unmatched_proposal_verifier_scorer.py`
- `tests/test_unified_infer_pipeline.py`

Implementation constraint:

- These tests are not an excuse to recreate `src/infer/`.
- The implementation wave should add a residue check that rejects unallowlisted
  direct `src.infer` imports after V1 tests are ported.

## Eval And Artifact Gate

V1 benchmark eligibility requires:

- `gt_vs_pred_scored.jsonl`, not raw-only `gt_vs_pred.jsonl`, for COCO/LVIS/mAP
  claims.
- One scored row per input row with stable image identity, dimensions, GT, row
  order, and `pred: []` for unscoreable rows.
- Row-local `pred_score_source`, integer `pred_score_version`, finite scores in
  `[0, 1]`, and portable score provenance.
- `pred_token_trace.jsonl` sufficient to recompute selected-token scores.
- Minimal rebuilt `src.eval` consumer that refuses missing or mismatched score
  provenance before metric computation.

## OpenSpec Impact

The active OpenSpec and superpower roadmap already encode the source-study
constraints:

- `src/infer.py` plus `src/inference/`, no `src/infer/` package.
- V1 inference config under `configs/coordexp_swift/infer/`, not
  `configs/infer/*`.
- HF scored generation with full token trace; no constant score fallback.
- Qwen no-resize image plan and processor/model vision parity.
- Fatal adapter validation in the adapter owner.
- Minimal V1 `src.eval` scored-artifact consumer.

No immediate OpenSpec rewrite is required before implementation. The stale
legacy docs remain a known adoption/documentation cleanup item after the V1
implementation lands.

## Remaining Gates Before Source Implementation

- User approval for implementation kickoff.
- Real tiny HF/Qwen trace proof during backend implementation.
- Adapter reload/status receipt implementation and negative fixtures under the
  adapter owner.
- V1 legacy-test allowlist or porting residue check during implementation.
- Real smoke fixture pinning and adapter-enabled smoke fixture pinning.
- Fixed val200 validation handles after implementation smokes pass. Full
  validation-dataset evaluation is optional and not required for V1 readiness.
