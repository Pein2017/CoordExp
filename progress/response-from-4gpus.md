# Response from 4‑GPU Node (Stage‑2 AB + vLLM **Server Mode**)

Date: 2026-01-29

This note summarizes what was verified on the 4‑GPU machine, so an 8‑GPU agent can run longer / more thorough tests.

## TL;DR (What’s confirmed)

- ✅ vLLM **server mode** works end-to-end with Stage‑2 AB on this repo: server comes up, learner connects, training steps run, logs are produced.
- ✅ A dedicated GPU smoke test passes: `tests/test_stage2_ab_vllm_server_mode_smoke.py` (gated by env flag).
- ✅ Proxy hygiene is handled: launch flow unsets `http_proxy/https_proxy` and sets `NO_PROXY=127.0.0.1,localhost` by default to avoid localhost traffic going through a global proxy.
- ⚠️ This was a **smoke**, not a quality verdict: in a 6‑step AB-mix run, both Channel‑B steps had `stage2/invalid_rollout=1` and large loss spikes.

## Environment (versions observed)

From the `ms` conda env:

- `torch==2.8.0+cu128`
- `transformers==4.57.1`
- `vllm==0.11.0`
- `xformers==0.0.32.post1`
- `flash_attn==2.8.2`

Notes:
- vLLM server mode previously tripped a version constraint when `flash-attn>2.8.2` with `xformers==0.0.32.post1`. This node is now consistent (`flash-attn==2.8.2`).

## Checkpoint used

User-provided (restored) checkpoint:

- `output/1-26/checkpoint-1516-merged`

## What was run

### 1) 4‑GPU server+learner smoke (AB mix)

Launcher:
- `scripts/stage2_ab_server_train.sh`

Config:
- `configs/stage2_ab/smoke_bbox_max60_ckpt1516_ab_mixed_vllm_server_3v1.yaml`

Command:

```bash
bash scripts/stage2_ab_server_train.sh \
  server_gpus=0,1,2 train_gpus=3 \
  config=configs/stage2_ab/smoke_bbox_max60_ckpt1516_ab_mixed_vllm_server_3v1.yaml
```

Topology:
- vLLM server: GPUs `0,1,2` (DP=3, TP=1) via `conda run -n ms swift rollout --infer_backend vllm`
- learner: GPU `3` (single process; server mode currently requires world_size==1 learner)

Output dir (run produced on this node):
- `output/stage2_ab/smoke_bbox_max60_ckpt1516/v0-20260129-072634/ab_mixed_vllm_server_3v1_smoke/`
- `logging.jsonl`: `output/stage2_ab/smoke_bbox_max60_ckpt1516/v0-20260129-072634/ab_mixed_vllm_server_3v1_smoke/logging.jsonl`

### 2) Pytest: vLLM server-mode smoke test

Command:

```bash
COORDEXP_RUN_4GPU_SMOKE=1 \
COORDEXP_STAGE2_AB_MODEL=output/1-26/checkpoint-1516-merged \
conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py
```

Result:
- `1 passed in 164.45s (0:02:44)`

## Results (AB mix 6-step smoke)

The config uses the AB pattern `A, A, B` repeating, with `max_steps=6` and small sample limits (`train_sample_limit=32`, `val_sample_limit=8`).

Parsed from `logging.jsonl`:

```text
step  ch  loss       ce         coord      invalid_rollout  parse_trunc_rate  rollout_len  gen_tok_s
1     A   0.224204   0.0488142  0.17539    0.0             0.0              0.0         0
2     A   0.144977   0.0584658  0.0865108  0.0             0.0              0.0         0
3     B   20.0745    18.6161    1.45842    1.0             0.0              30.0        1.22
4     A   1.20599    0.0481938  1.1578     0.0             0.0              0.0         0
5     A   0.554072   0.0420265  0.512046   0.0             0.0              0.0         0
6     B   19.6183    17.7629    1.85543    1.0             1.0              256.0       40
```

Interpretation (only what the smoke can support):
- Channel scheduling is correct (A,A,B repeats).
- vLLM server mode is active on B steps (`rollout/backend_vllm==1` in the raw logs).
- Both B steps were marked `stage2/invalid_rollout=1`, and step 6 had `rollout/parse_truncated_rate=1.0`.
  This likely explains the B-step loss spikes (~20) and means the rollouts weren’t usable for matching on these samples.
- A-step losses are “reasonable-looking” but too few/noisy to claim a real trend.

## Proxy / launch behavior

Both launch entrypoints default to disabling proxy for safety (unless you set `disable_proxy=false`):
- `scripts/train.sh` unsets `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/all_proxy/ALL_PROXY` and sets `NO_PROXY/no_proxy` to include `127.0.0.1,localhost`.
- `scripts/stage2_ab_server_train.sh` does the same before starting `swift rollout` and the learner.

This prevents localhost vLLM health/infer calls from being routed through a global proxy.

## Notes about “GPU memory not freed”

During discussion, we saw `nvidia-smi` report GPU memory usage on some devices without any visible processes inside this container namespace (`[Not Found]` / empty process table).
- This does **not** appear to be caused by the launcher (the vLLM server is torn down via process-group kill on exit, and port 8000 was not reachable after completion).
- Likely explanation: other host-side workloads / namespace visibility differences / persistence-mode allocator behavior.

If needed, check on the host for the owning PIDs/users.

## What the 8‑GPU agent should do next (recommended)

Focus: determine whether Channel‑B becomes “healthy” (low invalid rollouts + meaningful match metrics) over longer runs.

1) Run a longer AB-mix server-mode session (e.g. 100–1000 steps) and summarize:
   - `stage2/invalid_rollout` rate (should go down)
   - `rollout/parse_truncated_rate` (ideally low)
   - `rollout/match_rate`, `rollout/f1`, `rollout/sample_valid_pred_rate`
   - loss decomposition on B steps (`loss/ce`, `loss/coord`, bbox losses)
2) Consider allocating more GPUs to server DP (keep learner single GPU):
   - Example: `server_gpus=0,1,2,3,4,5,6 train_gpus=7` (DP=7)
3) If invalid/truncated rollouts remain high:
   - inspect raw rollout outputs (JSON validity, coord token formatting, truncation vs parsing)
   - tune rollout constraints (`max_new_tokens`, prompt length, server max_model_len) as needed

## Files added/used for this workflow

- `scripts/stage2_ab_server_train.sh` (new launcher: starts `swift rollout` + learner, handles proxy + health wait + cleanup)
- `configs/stage2_ab/smoke_bbox_max60_ckpt1516_ab_mixed_vllm_server_3v1.yaml` (AB mixed, server mode, uses `output/1-26/checkpoint-1516-merged`)
- `tests/test_stage2_ab_vllm_server_mode_smoke.py` (end-to-end GPU smoke; gated behind `COORDEXP_RUN_4GPU_SMOKE=1`)
