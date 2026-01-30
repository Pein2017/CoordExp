# Stage-2 AB Training Runbook (A: Expectation Loop, B: Rollout-Matching)

Stage-2 AB training in CoordExp is enabled via:

`custom.trainer_variant: stage2_ab_training`

It composes:
- **Channel-A**: multi-iteration “soft self-context” / expectation loop (no rollout generation).
- **Channel-B**: rollout -> parse -> match -> FN-append -> teacher-forced losses (rollout-matching SFT).

This runbook focuses on **end-to-end infrastructure correctness** (server-mode, GPU allocation, and key health metrics).
For rollout-matching details, see `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md`.

## Data Contract

Stage-2 AB consumes the same JSONL contract as other public detection sources (no embedded `messages` required):
- `images`, `objects`, `width`, `height`

See: `docs/DATA_JSONL_CONTRACT.md`.

## Recommended GPU Topology (Single Node)

Server mode (recommended for long rollouts) runs rollouts on dedicated GPUs and keeps the learner single-process:

- **4 GPUs**: `server_gpus=0,1,2` + `train_gpus=3` (3v1)
- **8 GPUs**: `server_gpus=0,1,2,3,4,5,6` + `train_gpus=7` (7v1)

Notes:
- v1 constraint: the learner must be **world_size == 1** (do not use `torchrun`).
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
- See the troubleshooting section in `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` for checks and mitigation steps.
