---
title: Ledger Adapter-Save Smoke Handoff
updated: 2026-06-24
branch: codex/ledger-auxiliary-loss
claim_scope: handoff
---

# Ledger Adapter-Save Smoke Handoff

This handoff is for continuing the ledger auxiliary-loss and adapter-reload
smoke work on another host node.

## Checkout

Repository/worktree used here:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git branch --show-current
# codex/ledger-auxiliary-loss
```

Remote branch:

```text
origin/codex/ledger-auxiliary-loss
```

On another host, fetch the branch from an existing CoordExp clone and create or
update a worktree:

```bash
cd /data/CoordExp
git fetch origin
git worktree add .worktrees/ledger-auxiliary-loss origin/codex/ledger-auxiliary-loss
cd .worktrees/ledger-auxiliary-loss
```

If the worktree already exists on that host:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git fetch origin
git pull --ff-only origin codex/ledger-auxiliary-loss
```

For one-off config-loader checks, include the local ms-swift checkout:

```bash
export PYTHONPATH=/data/CoordExp/.worktrees/ledger-auxiliary-loss:/data/ms-swift
```

Pytest already wires `/data/ms-swift` through `tests/conftest.py`, but raw
`python - <<'PY'` config scripts need the `PYTHONPATH` prefix above.

## What Is In This Branch State

This branch prepares the current ledger 128-sample smoke path for saved adapter
reload checks. The central behavior is:

- training may still save `coverage_ledger_head` when ledger is enabled;
- HF inference prepares a temporary adapter view that drops training-only
  `coverage_ledger_head` from `modules_to_save` and `adapter_model.safetensors`;
- `token_embeddings_adapter` remains preserved and validated for compact
  object-box-closed inference;
- the source adapter checkpoint directory is left unchanged.

Key code paths:

- `src/infer/checkpoints.py`
  - `prepare_adapter_checkpoint_for_inference(...)`
  - `TRAINING_ONLY_MODULES_TO_DROP_FOR_INFERENCE = ("coverage_ledger_head",)`
- `src/infer/runtime.py`
  - HF adapter inference uses the filtered runtime adapter view.
- `src/sft.py`
  - ledger training appends `coverage_ledger_head` to `modules_to_save`;
  - non-ledger runs remove stale `coverage_ledger_head` save entries.
- `src/training/coverage_ledger/qwen_capture.py`
  - hidden-state capture avoids mistaking PEFT wrappers for Qwen base modules.
- `src/trainers/metrics/batch_contract.py`
  - visual batch contract now fails when `image_grid_thw` is present but
    `pixel_values` are missing.

Research notes updated:

- `research/ideas/ledger-auxiliary-loss/discussion.md`
- `research/ideas/ledger-auxiliary-loss/index.md`

## Prepared 128 Smoke Configs

Both configs stay in the existing `coverage_ledger_closed_hard_sft_128*`
family instead of adding a long stacked suffix.

Baseline comparator:

```text
configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml
```

Ledger arm:

```text
configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml
```

Both now save inference-oriented adapter artifacts:

```yaml
training:
  save_strategy: steps
  save_steps: 128
  save_total_limit: 2
  save_delay_steps: 0
  save_last_epoch: true
  save_model_only: false
```

Resolved artifact subdirs:

```text
baseline: coverage_ledger_closed_hard_sft_128_baseline_adapter_save
ledger:   coverage_ledger_closed_hard_sft_128_adapter_save
```

The smoke scope is the 128-sample overfit trial: train on the deterministic
selected 128 rows, save adapters, reload adapters through the inference engine,
and evaluate on the same 128 rows. Do not treat this as validation evidence.

## Important Current Limitation

The ledger branch does not yet contain the stable `main` implementation of the
new weighted four-family `token_type_mass` objective, and it does not yet wire
the planned `geometry_valid_tail` term. In this worktree today:

- `hard_sft` still rejects `continuation_margin.enabled=true`;
- `token_type_mass` is still effectively enabled-only on this branch unless the
  stable main split is ported or merged;
- the smoke YAMLs intentionally keep `token_type_mass.enabled=false` and
  `continuation_margin.enabled=false` so they parse now;
- `geometry_valid_tail` is not enabled in the smoke YAMLs.

Therefore the next real benchmark cannot honestly claim
`ledger + mandatory type + geometry + continuation` until the stable type-loss
split is merged into this branch and the ledger-specific continuation/geometry
terms are implemented.

## Commands Already Run

YAML whitespace check:

```bash
git diff --check -- \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml
```

Result: passed.

Materialized config check:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/ledger-auxiliary-loss:/data/ms-swift \
/root/miniconda3/envs/ms/bin/python - <<'PY'
from pathlib import Path
from src.config.loader import ConfigLoader

paths = [
    Path("configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml"),
    Path("configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml"),
]
for path in paths:
    cfg = ConfigLoader.load_materialized_training_config(path)
    terms = cfg.objective.terms
    print(path)
    print("  artifact_subdir=", cfg.training["artifact_subdir"])
    print("  run_name=", cfg.training["run_name"])
    print("  max_steps=", cfg.training["max_steps"])
    print("  save_strategy=", cfg.training["save_strategy"])
    print("  save_steps=", cfg.training["save_steps"])
    print("  save_total_limit=", cfg.training["save_total_limit"])
    print("  save_delay_steps=", cfg.training["save_delay_steps"])
    print("  save_last_epoch=", cfg.training["save_last_epoch"])
    print("  save_model_only=", cfg.training["save_model_only"])
    print("  token_type_mass=", terms.token_type_mass.enabled)
    print("  continuation_margin=", terms.continuation_margin.enabled)
    print("  coverage_ledger=", terms.coverage_ledger.enabled)
PY
```

Result: passed. Baseline resolves `coverage_ledger=False`; ledger resolves
`coverage_ledger=True`. Both resolve `save_strategy=steps`, `save_steps=128`,
`save_total_limit=2`, `save_delay_steps=0`, `save_last_epoch=True`, and
`save_model_only=False`.

Inference adapter/reload compatibility tests:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/ledger-auxiliary-loss:/data/ms-swift \
/root/miniconda3/envs/ms/bin/python -m pytest \
  tests/test_coverage_ledger_head_install.py \
  tests/test_infer_checkpoint_resolution.py -q
```

Result: `36 passed`.

Narrow inference resolver only:

```bash
/root/miniconda3/envs/ms/bin/python -m pytest tests/test_infer_checkpoint_resolution.py -q
```

Result: `23 passed`.

One broad checkpoint-policy command was tried:

```bash
/root/miniconda3/envs/ms/bin/python -m pytest \
  tests/test_infer_checkpoint_resolution.py \
  tests/test_checkpoint_weight_only_policy.py -q
```

Result: `42 passed, 1 failed`. The failure is in
`test_instance_trie_configs_use_restartable_public_checkpoint_policy`, on
archived instance-trie configs rejected by the current pipeline registry as
unknown top-level domains. Treat that as an unrelated archived-config issue, not
as evidence against the ledger adapter inference path.

## Smoke Launch When GPUs Are Free

Do not launch if another training run is occupying GPU memory. The user asked
to stop actual launch in that case.

Recommended order when GPUs are available:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss

config=configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
gpus=8 \
scripts/train.sh

config=configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
gpus=8 \
scripts/train.sh
```

After each run, inspect the output root under:

```text
temp/detection_teacher_forcing/output/
```

Expected new run-family subdirs:

```text
coverage_ledger_closed_hard_sft_128_baseline_adapter_save
coverage_ledger_closed_hard_sft_128_adapter_save
```

Before any metric claim:

- verify `adapter_config.json` exists in the chosen saved checkpoint;
- verify `modules_to_save` includes `token_embeddings_adapter`;
- for ledger checkpoints, verify inference preparation drops only
  `coverage_ledger_head`;
- run strict parser/eval on the same 128 selected training rows;
- use the requested inference decode setting with repetition penalty
  `rp=1.10` when comparing saved adapters;
- compare baseline and ledger on the same selected 128 rows and same decode
  policy.

## Unresolved Decisions / Next Work

1. Port or merge the stable main bidirectional `token_type_mass` implementation
   into this ledger branch before enabling mandatory type-gating in the ledger
   smoke config.
2. Implement and test the planned ledger-only `continuation_margin` and
   `geometry_valid_tail` terms before claiming the full
   `ledger + mandatory type + geometry + continuation` stack.
3. Pin the exact "well-trained sorted pure-CE" adapter path before adding a
   `model.adapters` warm-start to production or smoke configs. Several sorted
   pure-CE references exist in the repo; this handoff does not guess the path.
4. The post-hoc `compact_span_drop_salvage` operation is not implemented in
   this branch state.
5. The OpenSpec CLI was not available in the earlier implementation environment;
   strict OpenSpec validation remains pending wherever `openspec` is installed.

## Skills For The Next Agent

Use these CoordExp skills if continuing from this handoff:

- `coordexp-router-context` for repo routing and current-vs-history boundaries.
- `review-convergence-loop` before promoting continuation, geometry, or salvage.
- `superpowers:subagent-driven-development` for the remaining implementation
  plan.
- `full-pipeline-smoke` before interpreting train128 results.
- `coordexp-infer-eval-workflow` for saved-adapter inference/eval artifacts.
- `git-hygiene` before committing or pushing follow-up work.
