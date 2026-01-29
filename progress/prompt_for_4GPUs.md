# Prompt for 4-GPU Machine: Stage-2 AB + vLLM **Server Mode** Diagnosis

This repo is currently on a temporary branch that contains Stage-2 AB trainer wiring + a set of smoke configs.
The current node’s GPUs are busy, so we could not validate vLLM server mode here. The goal on the 4‑GPU node
is to run a *minimal but end-to-end* Stage‑2 AB training smoke with rollouts served by `swift rollout` on a
separate GPU, and confirm all critical Stage‑2 invariants.

## Branch to Pull

Pull and checkout:

- Branch: `temp/2026-01-29-stage2ab-servermode-diagnosis`

This branch includes:
- `src/trainers/stage2_ab_training.py` (Stage2‑AB trainer)
- `configs/stage2_ab/*` (bbox-only v1 configs + smokes)
- fixes in `src/trainers/rollout_matching_sft.py` that unblock Stage‑2 AB training and vLLM diagnostics
- tests in `tests/test_stage2_ab_training.py`

## Stage-1 Checkpoint (Base for Stage-2 AB)

Use this checkpoint as the initial model:

`output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/mixed-merged-ckpt-1516`

## Dataset for Stage-2 AB (bbox-only, coord-token JSONL)

We run Stage‑2 AB on LVIS bbox-only max60:
- Train: `public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl`
- Val: `public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl`

The dataset records are bbox-only `{images, objects, width, height}`; the dataset pipeline builds
`messages` + `assistant_payload` internally (required by Stage‑2 AB).

## Critical Stage-2 AB / Rollout-Matching Invariants to Validate

### A) Channel-A (Expectation Loop)
- Runs `n_softctx_iter` forwards per optimizer step.
- Iterations `0..n-2` run under `no_grad`, final iteration runs with grads.
- `use_cache=False` and no `past_key_values` leakage.
- Coord-slot embedding update uses logits at `p-1` (shift alignment).

### B) Channel-B (Rollout → Parse/Match → FN append → Teacher-forced loss)
- Rollout happens with no-grad on server GPU.
- Learner parses strictly, matches with Hungarian + maskIoU gating, and appends FN.
- Logs include `rollout/seed_base` and server metadata; rollouts are reproducible for greedy decode.
- `stage2/drop_poly` stays 0 (bbox-only v1), and `stage2/invalid_rollout` counters behave.

### C) Packing Contract (Post-rollout packing only)
- `training.packing=true` triggers post-rollout packing inside trainer.
- Rollout generation is *not* packed.
- Carry-only mode requires `training.packing_drop_last=true`.

## On the 4‑GPU Node: Minimal End-to-End Diagnosis (1 GPU server + 1 GPU learner)

### Step 0) Pull branch

```bash
git fetch origin
git checkout temp/2026-01-29-stage2ab-servermode-diagnosis
```

### Step 1) Sanity: run unit tests (CPU-only)

```bash
conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py
```

### Step 2) Launch vLLM rollout server (GPU0)

Keep this running in a separate tmux pane/session.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift rollout \
  --model output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/v0-20260126-162638/epoch_4-pure_ce-LRs-2e-4_1e-4_4e-4-from-base-4B/merged_checkpoint-1516_20260128-130701 \
  --host 0.0.0.0 --port 8000 \
  --infer_backend vllm \
  --vllm_data_parallel_size 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_max_model_len 8192 \
  --vllm_enable_lora false
```

Notes:
- `group_port` for weight sync defaults to `51216` in the learner YAML. If that port is taken, pick a new one
  and update the learner YAML (or override by editing `configs/stage2_ab/smoke_bbox_max60_ckpt1516_b_only_vllm_server_1v1.yaml`).

### Step 3) Health check the server (from the same node)

```bash
conda run -n ms python - <<'PY'
import requests
print("health", requests.get("http://127.0.0.1:8000/health/").status_code)
print("world_size", requests.get("http://127.0.0.1:8000/get_world_size/").json())
PY
```

Expected:
- `/health/` returns 200
- `/get_world_size/` returns a JSON with `world_size` (should be 1 for DP=1 server)

### Step 4) Run Stage‑2 AB learner (GPU1), Channel‑B only, vLLM server mode

Learner must be **single-process** (no torchrun) for server mode.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage2_ab/smoke_bbox_max60_ckpt1516_b_only_vllm_server_1v1.yaml
```

Config file:
- `configs/stage2_ab/smoke_bbox_max60_ckpt1516_b_only_vllm_server_1v1.yaml`

What to verify in logs (`output/stage2_ab/smoke_bbox_max60_ckpt1516/*/b_only_vllm_server_1v1_smoke/logging.jsonl`):
- `stage2/channel_b == 1.0`
- `rollout/backend_vllm == 1.0`
- `rollout/decode_mode_greedy == 1.0`
- `rollout/rollout_len_mean > 0`
- `rollout/seed_base` present
- `vLLM rollout server engine_type` logged once (client handshake)
- No errors from `init_communicator` / weight sync

### Step 5) Run Stage‑2 AB Channel‑A only (GPU1), expectation-loop smoke

This does not depend on the rollout server.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage2_ab/smoke_bbox_max60_ckpt1516_a_only.yaml
```

Expected:
- completes 2 steps
- logs include `stage2/channel_a == 1.0`

## Optional: Use all 4 GPUs (3v1 topology)

If you want higher rollout throughput and still isolate rollouts from learner VRAM:

Server (GPUs 0,1,2):
```bash
CUDA_VISIBLE_DEVICES=0,1,2 conda run -n ms swift rollout \
  --model output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/v0-20260126-162638/epoch_4-pure_ce-LRs-2e-4_1e-4_4e-4-from-base-4B/merged_checkpoint-1516_20260128-130701 \
  --host 0.0.0.0 --port 8000 \
  --infer_backend vllm \
  --vllm_data_parallel_size 3 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_max_model_len 8192 \
  --vllm_enable_lora false
```

Learner (GPU3):
```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage2_ab/smoke_bbox_max60_ckpt1516_b_only_vllm_server_1v1.yaml
```

Then update `custom.extra.rollout_matching.vllm.server.servers` in the learner YAML to list 3 servers
if you actually launch 3 distinct server processes (common is a single server process that uses DP internally,
so keep one `base_url`).

## Troubleshooting Cheatsheet

- `/health/` not reachable:
  - wrong port/host, server not started, firewall rules.
- Communicator init fails (server mode):
  - pick a different `group_port`, ensure it’s reachable; increase `timeout_s`.
- OOM on server:
  - reduce `--vllm_max_model_len`, reduce `--vllm_gpu_memory_utilization`, reduce `max_new_tokens` in learner YAML.
- Rollout outputs missing or non-JSON:
  - ensure dataset uses correct `ROOT_IMAGE_DIR` (runner auto-sets it to JSONL parent).

