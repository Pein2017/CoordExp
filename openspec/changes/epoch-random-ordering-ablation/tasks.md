## 1. Shared Ordering Contract

- [x] 1.1 Extend inference config parsing and resolved-config plumbing to support `infer.object_ordering` with default `sorted`, invalid-value fail-fast behavior, and artifact metadata recording.
- [x] 1.2 Thread the active ordering policy through the shared dense prompt resolver so training, trainer-driven rollout/eval, and standalone inference consume the same ordering-aware prompt inputs.
- [x] 1.3 Add contract tests for ordering defaults and prompt parity in `tests/test_prompt_variants.py`, `tests/test_dense_caption_prompt_override.py`, and `tests/test_infer_batch_decoding.py`.
- [x] 1.4 Add an automated inference contract test that `infer.object_ordering` defaults to `sorted`, accepts `random`, and is recorded in `resolved_config.json` and summary artifacts.

## 2. Stage-1 Packing And Cache Parity

- [x] 2.1 Propagate epoch changes through the static packed dataset wrapper without regenerating `raw_plan` or `aligned_plan`.
- [x] 2.2 Add a fail-fast guard that allows epoch-varying object order under static packing only when per-index planning length is invariant across epochs.
- [x] 2.3 Keep encoded-sample cache parity explicit for the ablation arms by disabling encoded-sample cache in both sorted-order and random-order ablation configs so random-order runs remain cache-ineligible and sorted-order ablation configs do not gain an unintended cache-only advantage.
- [x] 2.4 Add or update tests in `tests/test_packing_wrapper.py`, `tests/test_stage1_static_packing_runtime_config.py`, `tests/test_encoded_sample_cache.py`, and `tests/test_dataset_multworker_determinism_probe.py` to verify epoch propagation, length-invariance enforcement, and cache-policy behavior.

## 3. Stage-2 Channel-A And Rollout Alignment

- [x] 3.1 Update Stage-2 Channel-A teacher-forced payload and canonical-prefix construction so `custom.object_ordering` controls object sequence and numbering for both `sorted` and per-epoch `random`.
- [x] 3.2 Keep Channel-B appearance-order semantics unchanged while routing trainer-driven rollout/eval prompt rebuilding through the same ordering-aware dense prompt resolver.
- [x] 3.3 Add or update tests in `tests/test_stage2_ab_prompt_alignment_contract.py`, `tests/test_stage2_rollout_aligned.py`, and `tests/test_stage2_two_channel_training.py` to verify Channel-A ordering parity and rollout/eval prompt parity under `object_ordering: random`.

## 4. Ablation Configs And Reproducibility Checkpoints

- [x] 4.1 Add explicit sorted-vs-random ablation YAML leaves for stage-1 and stage-2 A-only runs with `custom.object_ordering`, cache policy, and reproducibility-critical knobs pinned in config rather than CLI.
- [x] 4.2 Make the ablation config names, run names, seeds, and output-artifact expectations explicit enough that a reviewer can identify which arm produced a given run directory without opening source code.
- [x] 4.3 Ensure inference-side artifacts record resolved `prompt_variant` and `object_ordering`, and document the expected artifact checks for `resolved_config.json` and summary outputs.
- [x] 4.4 Add automated config-resolution tests for the new ablation YAML leaves so ordering policy, cache policy, seed, run naming, and output paths are asserted instead of checked only manually.
- [x] 4.5 Add a prod-like Stage-2 smoke YAML that extends `configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml` and overrides only smoke runtime knobs needed to exercise the new ordering feature (`training.max_steps`, sample limits, output/log dirs, and other smoke-only runtime settings).

## 5. Verification

- [x] 5.1 Run `conda run -n ms python -m pytest tests/test_prompt_variants.py tests/test_dense_caption_prompt_override.py tests/test_infer_batch_decoding.py -q`.
- [x] 5.2 Run `conda run -n ms python -m pytest tests/test_packing_wrapper.py tests/test_stage1_static_packing_runtime_config.py tests/test_encoded_sample_cache.py tests/test_dataset_multworker_determinism_probe.py -q`.
- [x] 5.3 Run `conda run -n ms python -m pytest tests/test_stage2_ab_prompt_alignment_contract.py tests/test_stage2_rollout_aligned.py tests/test_stage2_two_channel_training.py -q`.
- [x] 5.4 Run `conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py tests/test_stage2_ab_profile_leaf_contract.py -q`.
- [x] 5.5 Do one end-to-end reproducibility check with the ablation configs by confirming the resolved config/run artifacts expose the intended arm, ordering policy, seed, and output paths before any long training job is launched.
- [x] 5.6 After the new ordering feature is enabled, run a two-GPU smoke on the Stage-2 smoke YAML from `4.5`, confirm the run reaches the Stage-2 A-only path, and verify with runtime logs/artifacts that any observed instability is not caused by ordering/prompt/packing contract failures. Completed with repeated two-GPU smoke/debug runs on `2,3` and `0,1`; `resolved_config.json`/`run_metadata.json` were emitted, the feature path was exercised, and the remaining runtime asymmetry was localized to the downstream Qwen3-VL visual forward path rather than the random-ordering implementation.
