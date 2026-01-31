# Stage-2 AB Training Runbook (A: Expectation Loop, B: Rollout-Matching)

Stage-2 AB training in CoordExp is enabled via:

`custom.trainer_variant: stage2_ab_training`

It composes:
- **Channel-A**: multi-iteration “soft self-context” / expectation loop (no rollout generation).
- **Channel-B**: rollout -> parse -> match -> FN-append -> teacher-forced losses (rollout-matching SFT).

This runbook focuses on **end-to-end infrastructure correctness** (server-mode, GPU allocation, and key health metrics).
For rollout-matching details, see `STAGE2_ROLLOUT.md`.

## Current Architecture (Scheduler + Channels)

Stage-2 AB uses a deterministic, optimizer-step scheduler:

- Config: `custom.extra.stage2_ab.schedule.pattern: ["A", "A", "B"]` (example).
- Runtime: channel is chosen by `TrainerState.global_step % len(pattern)` (A/B only).

Channel semantics (high level):

- **A-step** (expectation loop): builds a teacher-forced target from GT objects (no rollouts), then runs the normal packed SFT forward/backward.
- **B-step** (rollout matching): generates rollouts (no grad), parses + matches, builds `Y_train`, then runs packed SFT forward/backward on the teacher-forced target.

Key config surfaces:

- Stage-2 AB schedule + loss weights: `custom.extra.stage2_ab.*`
- Rollout + parsing + matching knobs (shared with Stage_2): `custom.extra.rollout_matching.*`
- Packing knobs (required for Channel-B step mode): `training.packing`, `global_max_length`, `training.packing_buffer`, `training.packing_min_fill_ratio`, `training.packing_drop_last`

### Channel-B modes (micro vs step)

Channel-B supports two execution modes under `custom.extra.stage2_ab.channel_b.mode`:

- `micro` (legacy): each micro-step independently runs rollout → pack → learn (one packed sequence per micro-step when packing is enabled).
- `step` (current default in `configs/stage2_ab/prod/base.yaml`): “step-budgeted” rollouts + packing.
  - Collect raw dataset samples across the full gradient-accumulation window.
  - Run rollout generation + parse/match.
  - Pack into a **variable number of packed sequences** (each capped by `global_max_length`) and run forward/backward once per pack.
  - The outer Trainer still performs **exactly one** `optimizer.step()` for the optimizer step.

Important constraints for `mode: step`:

- Requires `training.packing: true` (the learner microbatch stays 1 packed sequence).
- Requires `training.dataloader_drop_last: true` to avoid partial accumulation windows at the end of an epoch.
- Keep `custom.extra.rollout_matching.post_rollout_pack_scope: micro` (step mode does packing inside the optimizer step).
- The number of raw dataset samples available per optimizer step is primarily controlled by:
  - `gradient_accumulation_steps` (derived from `training.effective_batch_size`), and
  - `custom.extra.rollout_matching.rollout_generate_batch_size` (optional stacking of raw samples per micro-step when packing is enabled).
- `custom.extra.stage2_ab.channel_b.rollouts_per_step` is a *global per-optimizer-step* budget. If omitted, it defaults to:
  - `per_device_train_batch_size * world_size * gradient_accumulation_steps`
  - In vLLM server mode, `world_size == 1`, so this is effectively `gradient_accumulation_steps` when per-device is 1.

### Channel-B pipelining (server mode only)

`custom.extra.stage2_ab.channel_b.enable_pipeline: true` overlaps rollout generation (server GPUs) with learner packing/learning.

- Requires `custom.extra.rollout_matching.rollout_backend: vllm`
- Requires `custom.extra.rollout_matching.vllm.mode: server` (dedicated rollout GPUs)
- Tuning: `custom.extra.stage2_ab.channel_b.rollout_decode_batch_size` controls the decode chunk size used by the pipeline.

## Data Contract

Stage-2 AB consumes the same JSONL contract as other public detection sources (no embedded `messages` required):
- `images`, `objects`, `width`, `height`

See: `../data/JSONL_CONTRACT.md`.

## Recommended GPU Topology (Single Node)

Server mode (recommended for long rollouts) runs rollouts on dedicated GPUs and keeps the learner single-process:

- **4 GPUs**: `server_gpus=0,1,2` + `train_gpus=3` (3v1)
- **8 GPUs**: `server_gpus=0,1,2,3,4,5,6` + `train_gpus=7` (7v1)

Notes:
- vLLM server mode constraint: the learner must be **world_size == 1** (do not use `torchrun`).
- Server GPUs will be idle during Channel-A steps; this is expected.

## Launch (vLLM Server Mode)

Use the helper launcher (starts `swift rollout` + learner, disables proxies, waits for `/health/`, and cleans up on exit):

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost

bash scripts/stage2_ab_server_train.sh \
  server_gpus=0,1,2,3,4,5,6 train_gpus=7 \
  vllm_gpu_memory_utilization=0.75 \
  config=configs/stage2_ab/prod/ab_mixed.yaml
```

## “Long rollout” length constraints

Two separate limits interact:
- `custom.extra.rollout_matching.max_new_tokens`: rollout generation budget (e.g., 3084).
- `global_max_length`: model/server max length (also passed as vLLM `--vllm_max_model_len` in server mode).

Rule of thumb:
- Ensure `global_max_length >= prompt_len + max_new_tokens`.
- If you set `max_new_tokens: 3084`, `global_max_length: 2048` is usually too small; use `4096` (or higher).

## DoRA (dlora)

If you want DoRA instead of “standard LoRA”, ensure configs include:
- `tuner.use_dora: true`

## Key Health Metrics (in `logging.jsonl`)

Channel scheduling:
- `stage2/channel_a` (1 on A steps)
- `stage2/channel_b` (1 on B steps)

Step-budgeted Channel-B (when `custom.extra.stage2_ab.channel_b.mode: step`):
- `stage2/raw_rollouts` (raw samples rolled out this optimizer step; per-rank in training logs)

Rollout health (B steps):
- `stage2/invalid_rollout` (should be near 0)
- `rollout/parse_truncated_rate` (should be low unless you hit max length)
- `rollout/sample_valid_pred_rate` (should be high)

Throughput (where your wall time goes):
- `time/rollout_generate_s` (often dominates long-rollout runs)
- `time/forward_s` (learner compute)
- `rollout/gen_tokens_per_s`

## Troubleshooting quick map

“0% GPU utilization” is not always a bug:
- If you are on **Channel-A steps**, the rollout server can idle and `time/rollout_generate_s == 0` is expected.

If training is stuck at the first B-step (server mode):
- `/health/` can be OK while the **communicator `group_port` is not listening**.
- See the troubleshooting section in `STAGE2_ROLLOUT.md` for checks and mitigation steps.
