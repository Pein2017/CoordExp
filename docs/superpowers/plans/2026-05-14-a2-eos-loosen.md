# A2 EOS-Loosen Clean Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, preflight, launch, and track a clean A2+EOS-loosen ablation without prefix-rollin or reduced supervised-token density.

**Architecture:** Add an opt-in `objective.eos` path to the existing A2 `random_permutation_et_rmp_ce` objective. The schema and runtime pass the existing EOS trust config into target construction; the target builder only changes `TokenTarget.loss_weight` for assistant `<|im_end|>` positions.

**Tech Stack:** Python config schema, latest compact detection dataset/objective code, YAML training configs, `pytest`, `src.sft`, `torchrun`/`scripts/train.sh`.

---

## Status

Running status: implementation complete; unit/schema tests passed; exact DDP8 smoke passed; 8-GPU production launch is running under `compact-full-support2-eos-loosen-a2e`.

## Corrected Research Framing

A2 is the trusted/default compact-full anchor. A3 and A4 are prefix-rollin
mechanism probes, not fair causal replacements for A2. A4 is A3+EOS loosen, not
A2+EOS loosen.

Previous A2-vs-A3/A4 claims must be caveated because A3/A4:

- sample a GT prefix `K`;
- mask prefix labels and supervise only the suffix;
- therefore supervise roughly `E[N-K] = N/2` object entries per image exposure
  while retaining prefix+suffix transformer attention compute;
- use different objectized support/balance, boundary, and type-gate sections;
- change EOS trust only inside the prefix-rollin surface.

The clean EOS ablation is `A2E-support2-eos-loosen`: A2 plus EOS trust weighting
only. Supervised-token-matched A3 or an A2/A3 mixture is a separate future fair
prefix-rollin ablation.

## Files

- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/detection/objective.py`
- Modify: `tests/test_detection_training_dataset.py`
- Modify: `tests/test_prefix_rollin_schema.py`
- Create: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml`
- Create/update: `progress/diagnostics/2026-05-14_a2_eos_loosen_ablation.md`

## Task 1: A2E Objective Plumbing

- [x] Add failing tests proving A2 can accept `objective.eos` while retaining
  `variant=random_permutation_et_rmp_ce`.
- [x] Allow optional `objective.eos` for `random_permutation_et_rmp_ce` while
  keeping `rollin`, `target`, `boundary`, and `type_gate` prefix-only.
- [x] Pass `objective.eos.eos_trust_weight` into the latest detection dataset.
- [x] Compute EOS trust for ordinary A2 examples and apply it only to assistant
  `<|im_end|>` token targets.
- [x] Verify A2E does not emit `rollin_k` metadata and does not mask object
  labels.

Verification:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_training_dataset.py::test_latest_detection_dataset_applies_eos_trust_without_prefix_rollin \
  tests/test_prefix_rollin_schema.py::test_random_permutation_schema_accepts_eos_ablation_without_prefix_rollin \
  tests/test_prefix_rollin_schema.py::test_random_permutation_eos_ablation_requires_experiment_surface \
  tests/test_prefix_rollin_schema.py::test_random_permutation_eos_ablation_configs_materialize_from_repo_paths \
  -q
```

Observed:

```text
4 passed
```

## Task 2: Configs

- [x] Add production ablation config:
  `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml`.
- [x] Add DDP8 prodlike smoke config:
  `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml`.
- [x] Preserve A2 train/val data, model, token rows, optimizer, schedule,
  batch/effective batch, and four-epoch settings.

Production config:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml
```

Smoke config:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml
```

Expected production artifact root pattern:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-<UTC>
```

## Task 3: Broader Preflight

- [x] Run the latest detection unit/schema test surface:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_training_dataset.py \
  tests/test_prefix_rollin_schema.py \
  -q
```

- [x] Materialize the production and smoke configs and print the key contract
  fields:

```bash
conda run -n ms python - <<'PY'
from src.config.loader import ConfigLoader

paths = [
    "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml",
    "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml",
]
for path in paths:
    cfg = ConfigLoader.load_materialized_training_config(path)
    print(path)
    print("variant", cfg.objective.variant)
    print("support", cfg.objective.trie_support_weight)
    print("balance", cfg.objective.trie_balance_weight)
    print("eos", cfg.objective.eos.eos_trust_weight.source if cfg.objective.eos else None)
    print("rollin", cfg.objective.rollin)
    print("surface", cfg.experiment.surface if cfg.experiment else None)
    print("run_name", cfg.training["run_name"])
    print("artifact_subdir", cfg.training["artifact_subdir"])
PY
```

- [x] Run a DDP smoke:

```bash
PYTHONPATH=. \
OMP_NUM_THREADS=8 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,5 \
conda run --no-capture-output -n ms torchrun \
  --master_addr=127.0.0.1 \
  --master_port=29618 \
  --nproc_per_node=2 \
  -m src.sft \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml
```

Observed artifact root:

```text
temp/recursive_detection_ce_latest/output/compact_full_support2_eos_loosen_ddp8_preflight/smoke-compact-full-support2-eos-loosen-ddp8-preflight/v0-20260514-035850
```

Smoke created `resolved_config.json`, `effective_runtime.json`,
`experiment_manifest.json`, `run_metadata.json`, train/eval provenance, and
rank heartbeat/log evidence. Metrics included
`recursive_detection_ce/eos_trust_weight`, `trie_support_weight=2.0`, and
`trie_balance_weight=1.0`.

This initial DDP2 smoke is superseded for launch gating by the exact DDP8
preflight recorded in the tracking note:

```text
temp/recursive_detection_ce_latest/output/compact_full_support2_eos_loosen_ddp8_preflight/smoke-compact-full-support2-eos-loosen-ddp8-preflight/v1-20260514-062139
```

## Task 4: Production Launch

- [x] Launch only after Task 3 passes and all 8 GPUs are visible/free.

Preferred wrapper command:

```bash
config=configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml \
gpus=all \
COORDEXP_TRAIN_HEARTBEAT=1 \
train_log_dir=temp/prod_launch_a2e_eos_loosen_20260514 \
conda run --no-capture-output -n ms bash scripts/train.sh
```

Direct equivalent:

```bash
PYTHONPATH=. \
OMP_NUM_THREADS=8 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n ms torchrun \
  --master_addr=127.0.0.1 \
  --master_port=29619 \
  --nproc_per_node=8 \
  -m src.sft \
  --config configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml
```

Observed production artifact root:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803
```

Launcher log:

```text
temp/a2e_eos_loosen_launch_logs/prod/compact_full_support2_eos_loosen-20260514T062522Z.log
```

The earlier hidden-context GPU blocker was resolved after the prior regular
8-GPU tests were terminated. The guard then observed a clean all-8-GPU window,
completed the exact DDP8 smoke, observed a second clean window, and started the
production command.

## Task 5: First Evaluation After Training

- [ ] Evaluate the trained checkpoint on A2 `val200` greedy cap1024/rp1.10.
- [ ] If cheap, also evaluate greedy cap3084/rp1.10.
- [ ] Record raw metrics, guarded metrics, parse/drop counters, duplicate
  guard report, prediction count, and artifact roots.
- [ ] If object count or duplicate behavior changes materially, generate a
  manual audit packet before interpreting the result.

Decision rule:

- A2E helps if valid-object emission increases while A2 stability mostly holds.
- A2E fails unsafe if duplicate/collapse increases materially.
- A2E is neutral if recall/valid-object emission does not improve.
