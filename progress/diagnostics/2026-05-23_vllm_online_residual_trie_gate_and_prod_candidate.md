---
doc_id: progress.diagnostics.vllm_online_residual_trie_gate_and_prod_candidate_2026_05_23
layer: progress
doc_type: diagnostic-note
status: current
domain: training
summary: vLLM 6:2 online residual-trie gate, adapter-only save validation, and production-candidate method selection.
tags: [stage2, vllm, online-learning, residual-trie, adapter-save, production-candidate]
updated: 2026-05-23
---

# vLLM Online Residual-Trie Gate And Production Candidate

Scope: Stage-2 online residual-trie infrastructure and method selection for the
compact-full ET-RMP-CE checkpoint-3664 family. Evidence here is smoke/mini only:
`train128_val64`, COCO80 view, K=4 rollout attempts, decode batch size 4. This
is not full validation.

## Infrastructure Result

Implemented native vLLM full-sync materialization for adapter-backed
coord/schema token rows:

- helper: `src/trainers/rollout_runtime/vllm_sync_materialization.py`;
- server integration: `src/trainers/rollout_runtime/vllm_server.py`;
- colocate integration: `src/trainers/rollout_runtime/vllm_engine.py`;
- tests:
  - `tests/tokens/test_vllm_sync_materialization.py`;
  - `tests/test_stage2_rollout_runtime.py`;
  - `tests/test_training_config_strict_unknown_keys.py`.

The runtime sync remains `rollout_matching.vllm.sync.mode=full`, but before
native vLLM receives weights the learner materializes active token-row adapter
effects into ordinary `embed_tokens.weight` / `lm_head.weight` rows and strips
learner-only LoRA/module-save keys. This preserves adapter-only checkpoint
saving while avoiding native vLLM errors for `coord_offset_adapter` keys.

Validation:

```bash
conda run -n ms python -m pytest \
  tests/tokens/test_vllm_sync_materialization.py \
  tests/test_stage2_rollout_runtime.py \
  tests/test_training_config_strict_unknown_keys.py \
  -q
# 205 passed in 1.58s

openspec status --change materialize-vllm-full-sync-adapter-rows
# 4/4 artifacts complete

openspec validate materialize-vllm-full-sync-adapter-rows --strict
# Change 'materialize-vllm-full-sync-adapter-rows' is valid
```

## vLLM 6:2 Gate

Config:

- `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml`

Launch shape:

```bash
server_gpus=0,1,2,3,4,5 train_gpus=6,7 \
config=configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml \
WAIT_TIMEOUT=900 WAIT_INTERVAL=2 MASTER_PORT=29674 \
conda run --no-capture-output -n ms bash scripts/train_stage2.sh
```

Artifact root:

- `output/stage2_ab/smoke/coco80_view_online_residual_trie_train128_val64/tail_append_zero_fp_lr1e5_decode4_vllm6srv2lr_gate/gate_4steps-coco80_view-train128-val64-online_residual_trie-tail_append-zero_fp-lr1e5-decode4-vllm6srv2lr-compact_full-et_rmp_ce_ckpt3664/v3-20260523-183336`

Evidence:

- topology: 6 vLLM server ranks and 2 learner ranks, ratio 3:1;
- decode batch cap: `channel_b_decode_batch_size=4`, `eval_decode_batch_size=4`;
- server GPU memory peaked around 62.9 GiB and learner memory around 59.8 GiB;
- no OOM;
- materialization log appeared at sync:
  `materialized coord_offset_adapter rows for vLLM full-sync`;
- native vLLM used `/update_flattened_params/`;
- no `coord_offset_adapter` unknown-parameter error;
- POST `/infer/` succeeded with HTTP 200;
- parse drops and truncation stayed zero in rollout and final eval.

Final gate metrics:

| scope | recall | fn | precision | mAP | parse trunc | invalid/drop ambiguous |
|---|---:|---:|---:|---:|---:|---:|
| final rollout step | 0.5227 | 231 | 0.5315 | n/a | 0 | 0 |
| final eval step 4 | 0.5341 | 205 | 0.5087 | 0.4647 | 0 | 0 |

Throughput:

- `time/rollout_generate_s=30.28` at final step;
- `rollout/gen_tokens_per_s=130.37`;
- train runtime `298.01s` for the 4-step gate.

## Adapter-Only Save Gate

Config:

- `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_1step_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_save_gate.yaml`

Artifact root:

- `output/stage2_ab/smoke/coco80_view_online_residual_trie_train128_val64/tail_append_zero_fp_lr1e5_decode4_vllm6srv2lr_save_gate/save_gate_1step-coco80_view-train128-val64-online_residual_trie-tail_append-zero_fp-lr1e5-decode4-vllm6srv2lr-compact_full-et_rmp_ce_ckpt3664/v1-20260523-184751`

Checkpoint:

- `checkpoint-1`, size `38M`;
- files: `adapter_model.safetensors`, `adapter_config.json`,
  `additional_config.json`, `trainer_state.json`, `training_args.bin`,
  `README.md`;
- no full-model shard files.

`adapter_config.json` records LoRA plus `modules_to_save=["coord_offset_adapter"]`
against base model
`/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`.

## Small-Sample Method Read

Scored smoke/mini runs show a consistent trade-off:

| family | representative run | eval recall | fn | precision | mAP | parse trunc | read |
|---|---|---:|---:|---:|---:|---:|---|
| raw `decode4-hf` gate | `.../coco80_view_decode4_gate/.../v8-20260523-101030` | 0.5409 | 202 | 0.4516 | 0.4740 | 0.75 | best tiny mAP/recall, but parser health and duplicate counters are poor |
| online residual-trie HF | `.../online_residual_trie.../v1-20260523-143720` | 0.5386 | 203 | 0.5290 | 0.4687 | 0 | almost same recall, much cleaner syntax |
| online residual-trie vLLM | `.../vllm6srv2lr_gate/.../v3-20260523-183336` | 0.5341 | 205 | 0.5087 | 0.4647 | 0 | small metric cost, large decode speedup |
| offline/prepared stage2-trie CE mini | `sorted_zero_fp.../v6-20260523-094338` | 0.4068 | 261 | 0.6755 | 0.3967 | 0.7188 | not a production parent for online learning |

The online residual-trie family is the better production direction despite a
small smoke mAP deficit: it removes parse truncation in this gate, dramatically
reduces duplicate-pathology counters, and is now compatible with native vLLM
server-mode throughput.

For the evaluated stage2-trie CE mini group, `sorted_zero_fp` led that offline
mini slice (`mAP=0.3967`, `recall=0.4068`), while tail-append was better aligned
with current online residual-trie defaults and legacy FN-tail semantics. Treat
`sorted` as a paired ablation, not the default.

## Method Recommendation

Use online residual-trie as the production-candidate core:

- `token_ce` on Channel A only;
- `stage2_trie_ce` on Channel B only, preset `rollout_trie_hard_ce`;
- K=4 rollout attempts with temperatures `[0.0, 0.4, 0.7, 1.0]`;
- `compact_full` + `compact_grammar`;
- `fp_policy.mode=zero_loss_context`;
- `pseudo_positive.enabled=false`;
- UL/pseudo admission through residual-state consensus only:
  `ul_consensus_ratio=1.0`, `min_ul_valid_rollouts=4`;
- `tail_append` as conservative default, `sorted` as the required paired ablation;
- native vLLM server-mode 6:2 with decode batch size 4 and
  `gpu_memory_utilization=0.80`.

Do not promote the old `prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
as a parent for this line. It is a legacy clean-prefix pseudo-positive profile.
The useful idea from that lineage is recovered-GT / explorer-evidence as a
recall signal, but the old K4 pseudo-positive path was coord-heavy and
anchor-centric; it does not prove production recall improvement.

New production-candidate configs:

- `configs/stage2_two_channel/prod/ab_mixed_coco1024_online_residual_trie_vllm_tail_append.yaml`
- `configs/stage2_two_channel/prod/ab_mixed_coco1024_online_residual_trie_vllm_sorted.yaml`

These inherit the production vLLM/dataset shell, then explicitly override the
online residual-trie objective/runtime keys. They are intended for cfg-only and
preflight first, not blind full production launch.

Cfg-only validation passed after keeping the currently implemented compact-full
invalid-rollout policy:

```bash
conda run --no-capture-output -n ms python -m src.sft \
  --config configs/stage2_two_channel/prod/ab_mixed_coco1024_online_residual_trie_vllm_tail_append.yaml \
  --cfg-only
# status=ok, eval_steps=300, save_steps=300, gradient_accumulation_steps=96

conda run --no-capture-output -n ms python -m src.sft \
  --config configs/stage2_two_channel/prod/ab_mixed_coco1024_online_residual_trie_vllm_sorted.yaml \
  --cfg-only
# status=ok, eval_steps=300, save_steps=300, gradient_accumulation_steps=96
```

Attempting `invalid_rollout_policy: dump_and_continue` failed cfg-only because
`compact_full` currently requires `fallback_gt_fn_append_only` until alternate
invalid-rollout behavior is implemented. The production-candidate configs use
the implemented policy.

## Required Next Gates

1. Config-load both production-candidate leaves.
2. Run the already passing 4-step vLLM gate before any larger run.
3. Run the 1-step save gate whenever checkpoint-save behavior changes.
4. For first production-like training, keep `server_gpus=0,1,2,3,4,5` and
   `train_gpus=6,7`, decode batch size 4, and do not raise vLLM memory
   utilization above 0.80 until a longer run confirms headroom.
5. Report labeled-GT recall/FN separately from UL-promoted local evidence.
6. Track parse truncation, invalid/drop ambiguous counts, confidence trace
   fallback, duplicate counters, residual atom/sequence count, UL promoted /
   quarantined / rejected, and recovered-GT style evidence.
7. Promote `tail_append` only if it beats or ties `sorted` on the same
   production-like eval surface; otherwise flip the default deliberately.
