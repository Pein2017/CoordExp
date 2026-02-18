## 1. Deprecated Keys (Fail-Fast, No Warnings, No Legacy Support)

- [ ] 1.1 Enforce that deprecated/legacy evaluator keys are rejected (fail-fast) with an actionable error message naming the unsupported keys:
  - `unknown_policy`
  - `semantic_fallback`
- [ ] 1.2 Add a unit test asserting that including `unknown_policy` or `semantic_fallback` in eval config fails fast (no warning-only behavior).
- [ ] 1.3 Enforce that deprecated training key `training.packing_length` is rejected (fail-fast) with an actionable error message:
  - Removal guidance: use `global_max_length` and/or `template.max_length` instead.
- [ ] 1.4 Add a unit test asserting that including `training.packing_length` fails fast (no warning-only behavior).

## 2. Silent-Failure Policy Enforcement (Datasets/Trainers/Infer/Eval)

- [ ] 2.1 Operationalize “explicit sink” compliance with an objective allowlist:
  - Define the initial allowlist (minimal) for exception suppression sites that are permitted to be best-effort.
  - Add `tests/test_silent_failure_policy.py` that fails CI if forbidden suppression patterns appear outside the allowlist (start with `except Exception: pass`).
- [ ] 2.2 Add/standardize a deterministic mechanism for temporary `template.system` overrides that guarantees restoration (via `finally`) or fails fast.
- [ ] 2.3 Add a minimal unit test covering prompt override restoration (no leakage across two sequential encodes).
- [ ] 2.4 Scan core paths (at least `src/trainers/`, `src/infer/`, `src/eval/`) for blanket exception swallowing and either:
  - delete it (preferred), or
  - narrow catches to explicit exception types and return an explicit failure sentinel (no silent `pass`), or
  - move it behind an allowlisted explicit sink (only when failure cannot affect model inputs/labels/metrics artifacts).

## 3. Dataset Silent-Failure Cleanup (Fusion + Dense Caption)

- [ ] 3.1 Remove blanket `except Exception: pass` around sample metadata attachment in `src/datasets/unified_fusion_dataset.py` and fail fast on metadata write failure.
- [ ] 3.2 Tighten `src/datasets/dense_caption.py` prompt injection: stop swallowing setter failures and enforce deterministic restoration.
- [ ] 3.3 Tighten `src/datasets/dense_caption.py` debug metadata handling: do not silently suppress unexpected exceptions in core encoding paths.

## 4. Fusion Metadata Regression Tests

- [ ] 4.1 Add a regression test asserting that encoded fusion samples include `sample_id`, `dataset`, and `base_idx`.
- [ ] 4.2 Add a regression test asserting that metadata attachment failure is fatal (e.g., encoded output is not a mutable mapping).

## 5. Detection Evaluator Regression Tests (Encoder + Deprecated Keys)

- [ ] 5.1 Add a unit test asserting evaluation fails loudly when the semantic encoder cannot be loaded (simulate via monkeypatch; no network dependency).
- [ ] 5.2 Add a unit test asserting deprecated keys fail fast (see 1.2) and that no warning-only behavior remains in the evaluator/pipeline path.

## 6. Dead Code Removal (Verified Zero Call Sites)

- [ ] 6.1 Remove unused Stage-2 scheduler helpers (`Stage2ABSchedulerMixin._stage2_b_step_mode`, `_stage2_policy_channel_for_step`) after verifying no call sites.
- [ ] 6.2 Remove unused Stage-2 training RNG helper (`Stage2ABTrainingTrainer._maybe_seed_hf_sampling_rollout`) after verifying no call sites.
- [ ] 6.3 Remove unused rollout-matching helpers (`RolloutMatchingSFTTrainer._post_rollout_pack_scope`, `_rollout_one`, `_vllm_server_total_world_size`) after verifying no call sites.
- [ ] 6.4 Remove unused inference pipeline helper (`_write_gt_vs_pred_plot_rows`) after verifying no call sites.

## 7. Simplify Deprecated/No-Op Surfaces (Safe Removals)

- [ ] 7.1 Remove the deprecated `HardSampleMiningConfig` placeholder from `src/config/schema.py` (configs already fail fast on `custom.hard_sample_mining`).
- [ ] 7.2 Remove unused legacy prompt alias `USER_PROMPT_JSON` from `src/config/prompts.py` (verify no in-repo imports).

## 8. Evaluation/Pipeline Refactors (Parity-Gated)

- [ ] 8.1 Deduplicate ROOT_IMAGE_DIR resolution in `src/infer/pipeline.py` into a single helper while preserving precedence.
- [ ] 8.2 Consolidate duplicated evaluator prep helpers (`_prepare_all` and `_prepare_all_separate`) in `src/eval/detection.py` and add a parity test on a small fixture.

## 9. Verification and Reproducibility Checkpoints

- [ ] 9.1 Run `openspec validate simplify-silent-failures-and-dead-code --strict` and fix any reported contradictions or missing artifacts.
- [ ] 9.2 Run targeted compile sanity (`conda run -n ms python -m py_compile` on touched modules).
- [ ] 9.3 Run focused unit tests for inference/eval/dataset behavior (add tests where missing).
- [ ] 9.4 Repro checkpoint: record a canonical config path + `run_name` + seed and confirm output artifacts are created under the resolved run dir (`gt_vs_pred.jsonl`, `summary.json`, `eval/`, `vis/` as applicable).
