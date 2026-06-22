# FN-Rescue Phase-3 ScriptMaster Orchestration Plan

Status: implementation plan

Scope root:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes
```

Artifact scope:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
```

## Objective

Create a worktree-local ScriptMaster that runs the current Phase-3
attention-guided causal binding workflow without interpreting the 8 available
GPUs as production training.

The ScriptMaster must run these lanes in dependency order:

1. `target_mask`
2. `competitor_source`
3. `sink_triage`
4. `case_linked`
5. `report`

## Implementation Tasks

- Add `scripts/analysis/launch_autoreg_fn_rescue_attention_guided_causal_binding_tmux.sh`.
- Derive default `REPO_ROOT` from the launcher path.
- Resolve `paths.artifact_root` from the Phase-3 YAML and write commands under
  `$artifact_root/logs/`.
- Guard existing lane outputs unless `ALLOW_OVERWRITE=1`.
- Assign the first two available GPUs to the two decode/intervention lanes.
- Run sink/case/report stages with `CUDA_VISIBLE_DEVICES=`.
- Support `DRY_RUN=1` to write and print the command file without starting tmux.
- Keep this as analysis orchestration only; no training entrypoints.

## Verification

- Add launcher dry-run tests that assert generated commands use the worktree
  root and the Phase-3 runner.
- Run the Phase-3 unit suite, Phase-2 suite, targeted continuation launcher
  suite, and compile checks.
- Run a dry-run launch against the smoke config.
- Run smoke artifacts through the schema/count checker after any real smoke.

## Current Limitation

The current Phase-3 intervention lanes are not shard-safe.  The first
ScriptMaster therefore uses the available GPUs as a resource pool but only
binds one GPU per decode lane.  Using all eight GPUs for one lane requires a
separate shard-output and merge contract.
