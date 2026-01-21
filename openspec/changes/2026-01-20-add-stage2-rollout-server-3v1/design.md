# Design: 3v1 Rollout Server + Stage_2 Learner

## Overview
This change proposes a **disaggregated actor/learner** topology for stage_2 rollout-matching SFT.

- **Actors** (rollout side): vLLM server(s) running on a dedicated GPU subset to maximize rollout throughput.
- **Learner** (training side): existing stage_2 rollout-matching trainer running teacher-forced forward/backward on packed sequences.

Key principle: keep stage_2 correctness and reproducibility while eliminating the GPU-memory contention that causes OOM in colocate mode.

## Target Topology (3 vs 1)
Single node, 4 GPUs:
- GPUs 0-2: vLLM rollout server (3 GPUs)
- GPU 3: learner (1 GPU) running `RolloutMatchingSFTTrainer` (single training process / `world_size == 1`)

This is intentionally compatible with the fact that Qwen3-VL-8B can fit on a single GPU, and it leaves room to reduce `global_max_length` (e.g. 16k -> 12k) if needed.

## Data Flow (E-step / M-step)
Stage_2 already has a clear two-phase structure inside one `compute_loss` call:

1) rollout (no grad)
2) strict parse + match
3) build teacher-forced `Y_train`
4) teacher-forced forward/backward with masking

In the 3v1 design we only change **where** step (1) runs.

### E-step (fresh rollouts)
For each raw micro-batch (identity-collated list of samples):
1. Learner builds vLLM requests from `messages` (+ image payload via template) using the same template contract.
2. Learner sends requests to vLLM server.
3. Server returns, per sample:
   - `prompt_token_ids`
   - `response_token_ids`
   - decoded text (optional, for logging)
4. Learner performs strict parse/match and constructs:
   - append-ready prefix token boundary (prefix-only trimming allowed)
   - mandatory FN append fragment
   - final teacher-forced training target
     `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`
5. Learner tokenizes/encodes the teacher-forced inputs and prepares per-sample supervision masks.
6. Learner applies post-rollout packing (micro/window scheduling) to produce packed rows of max length `global_max_length`.

### M-step (teacher forcing)
- Learner runs standard teacher-forced forward/backward on the packed rows.
- Optimizer step updates parameters.

### Repeat
- After the optimizer step, learner synchronizes weights to the rollout server (see Weight Sync).
- Next E-step uses the updated weights.

### Interaction with `rollout_buffer`
If `custom.extra.rollout_matching.rollout_buffer.enabled: true` and `m_steps > 1`:
- The learner generates fresh rollouts only on E-steps.
- The learner reuses the prepared teacher-forced batches for subsequent M-steps.
- Weight sync to server is only required on E-steps (because no new rollouts are requested on reuse steps).

This preserves the existing semantics: buffering is the throughput knob, and it naturally reduces server-sync frequency.

## Dataloader Iteration Semantics
The learner owns the dataloader. This keeps iteration deterministic and avoids splitting dataset ownership across processes.

Rules:
- The learner consumes exactly the same sequence of raw micro-batches as in current stage_2.
- When `rollout_buffer.m_steps > 1`, the existing accumulation-window repeater remains responsible for repeating raw windows so that no dataset samples are skipped.
- Window-aware packing (`post_rollout_pack_scope: window`) continues to use the existing lookahead wrapper to provide full-window visibility without prefetching outside the dataloader contract.

Non-goal (v1): actors owning a separate dataloader. This would complicate reproducibility (samplers, sharding, resume) and is not required to solve the OOM instability.

## Packed `Y_train` Delivery (What Crosses Process Boundaries)
### v1 (recommended): deliver rollouts only
Cross-process data is intentionally minimized:
- learner -> server: inference requests (messages + image payload)
- server -> learner: `prompt_token_ids`, `response_token_ids`, and decoded text
- learner -> server: weight sync payload (full weights or adapter)

Packed `Y_train` and all supervision masks remain learner-local.

Why:
- Packed sequences can be up to 16k tokens; sending them over IPC/NCCL adds overhead and complexity.
- Learner already has tokenizer/template context and must compute match/masks.
- This keeps the stage_2 implementation close to current code paths.

### Future option (not in v1): server delivers pre-tokenized segments
If learner-side CPU becomes a bottleneck, we can optionally allow actors/server to return:
- append-ready prefix token boundary (or prefix token ids)
- pre-encoded teacher-forced segment tensors + metadata

This would require a stable serialization contract for all required masks/targets and is deferred.

## Weight Sync (No Offload/Reload)
We reuse ms-swift's proven GRPO server sync mechanism:
- training client initializes an NCCL communicator with the vLLM server via a `/init_communicator/` endpoint.
- weights are broadcast in-memory using NCCL (no disk checkpoints).

Two modes (configured via `custom.extra.rollout_matching.vllm.sync.mode`; see spec for fallback rules):

### Default: full merged weights sync (robust)
- Recommended when:
  - training uses DoRA (dlora)
  - multimodal LoRA on ViT/aligner is enabled
  - vLLM LoRA is unstable on the local stack
- Learner merges adapters into base weights, then broadcasts the merged weights.
- This matches the operational pattern used in Qwen3-VL GRPO configs (e.g. `vllm_enable_lora: false`).

### Optional: adapter-only sync (fast)
- Recommended only when vLLM LoRA is known to work for the model stack.
- Learner broadcasts only LoRA/DoRA adapter tensors.
- Lower bandwidth and faster per-iteration sync.

Sync cadence:
- MUST sync after any learner update *before* requesting rollouts that are claimed to be "latest-policy" rollouts.
- With rollout buffering enabled, SHOULD sync only on E-steps.

## Determinism / Seeds
Determinism goals:
- same config + same seed -> same dataset order + same rollout requests
- greedy decoding should be stable, but we still record seeds in artifacts

Requirements:
- Learner MUST derive a per-request seed deterministically from:
  - `training.seed`
  - optimizer-step index (`state.global_step`)
  - within-step sample index
- Learner MUST log:
  - server endpoint(s)
  - sync mode (full vs adapter)
  - sync step counters

## Practical Launch Sketch (Single Node)
1) Start the rollout server on 3 GPUs:
- `CUDA_VISIBLE_DEVICES=0,1,2 swift rollout ...` with data-parallel size 3 (or equivalent multi-GPU settings).

2) Start the learner on 1 GPU:
- `CUDA_VISIBLE_DEVICES=3 conda run -n ms python -m src.sft --config <stage2_server_yaml>`

The stage_2 YAML specifies `custom.extra.rollout_matching.vllm.mode: server` plus `custom.extra.rollout_matching.vllm.server` and `custom.extra.rollout_matching.vllm.sync` settings.

## Compatibility Notes
- This is designed to be compatible with `transformers` + `ms-swift`:
  - vLLM server protocol and weight sync come from ms-swift (same as GRPO).
  - learner remains a normal HF/ms-swift trainer.
- vLLM cannot run backward; all SFT training remains on the learner.
- DoRA (dlora) remains the only tuning method in scope.
