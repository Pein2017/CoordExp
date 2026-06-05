# A3.3 Post-X1 Instance-Basin Tomography Real-Tiny Smoke

Date: 2026-06-05

Worktree: `/data/CoordExp/.worktrees/fn-rescue-attention-probes`

Artifact root:

`/data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_real_tiny`

## Scope

This is a real GPU smoke for the A3.3 post-x1 slot-posterior probe. It is not a full validation run and must not be interpreted as detector accuracy or AP/F1 ranking.

The run compares three checkpoints as mechanism probes:

- `fullobj_random_pure_ce_ckpt3668`
- `fullobj_sorted_pure_ce_ckpt3668`
- `et_rmp_ce_ckpt3664`

The ET-RMP checkpoint remains a `reference_anchor` with `template_objective_confounded_reference`; it is not a clean sorted-vs-random comparison member.

## Commands

Index and prefix materialization:

```bash
python scripts/analysis/run_post_x1_instance_basin_tomography.py \
  --config /tmp/a33_real_tiny.yaml \
  --stages data_root_audit,case_universe,prefix_states \
  --allow-overwrite
```

Tiny single-row smoke:

```bash
A33_GPU_ID=0 A33_REAL_SLOT_LIMIT=1 \
python scripts/analysis/run_post_x1_instance_basin_tomography.py \
  --config /tmp/a33_real_tiny.yaml \
  --stages slot_posterior \
  --real-runtime \
  --allow-overwrite \
  --shard-id 0
```

Expanded real-tiny smoke:

```bash
for spec in 0:0 1:1 2:5; do
  shard=${spec%%:*}
  gpu=${spec##*:}
  A33_GPU_ID=$gpu A33_REAL_SLOT_LIMIT=6 \
  python scripts/analysis/run_post_x1_instance_basin_tomography.py \
    --config /tmp/a33_real_tiny.yaml \
    --stages slot_posterior \
    --real-runtime \
    --allow-overwrite \
    --shard-id "$shard" &
done
wait
```

Finalize:

```bash
python scripts/analysis/run_post_x1_instance_basin_tomography.py \
  --config /tmp/a33_real_tiny.yaml \
  --stages slot_merge,trajectory,attraction_matrix,prefix_sensitivity,greedy_continuation,report,gallery \
  --allow-overwrite
```

## Artifact Gate

`evaluate_status(artifact_root)` returned:

```json
{
  "status": "final_artifacts_present",
  "failed_gates": [],
  "row_counts": {
    "case_universe_rows": 256,
    "prefix_state_rows": 1024,
    "slot_posterior_rows": 162,
    "slot_posterior_shard_rows": 162
  },
  "checkpoint_role_counts": {
    "et_rmp_ce_ckpt3664": 54,
    "fullobj_random_pure_ce_ckpt3668": 54,
    "fullobj_sorted_pure_ce_ckpt3668": 54
  },
  "schema_versions": {
    "a3.3.v1": 162
  }
}
```

## Main Observation

In this smoke, low recall/post-x1 binding errors are not explained by EOS or non-coordinate tokens taking probability mass at these forced slot states.

Full-vocab coordinate mass is consistently high:

| checkpoint_role | rows | mean coord mass | min coord mass | low coord-mass rows |
| --- | ---: | ---: | ---: | ---: |
| `fullobj_random_pure_ce_ckpt3668` | 54 | 0.9906 | 0.9653 | 0 |
| `fullobj_sorted_pure_ce_ckpt3668` | 54 | 0.9922 | 0.9763 | 0 |
| `et_rmp_ce_ckpt3664` | 54 | 0.9887 | 0.9768 | 0 |

This means the next-token distribution is overwhelmingly on coordinate tokens once the prompt is forced to `desc + x1` and the model is asked for `y1/x2/y2`.

The failure surface is therefore better described as coordinate-basin or instance-binding ambiguity, not continuation-vs-stop or coordinate-vs-noncoordinate mass collapse, at least for this post-x1 smoke.

## Deduplicated Slot Taxonomy

The configured prefix modes include two aliases that canonicalize to `same_desc_good_prefix`. To avoid double-counting those duplicated semantic prefixes, the following table deduplicates rows by:

`(checkpoint_role, prefix_row_id, slot)`

After deduplication:

- rows: 108
- taxonomy counts:
  - `target`: 54
  - `tied`: 21
  - `background`: 19
  - `other_desc_object`: 8
  - `competitor_same_desc`: 6

By checkpoint:

| checkpoint_role | rows | target rate | taxonomy counts | mean coord mass |
| --- | ---: | ---: | --- | ---: |
| `fullobj_random_pure_ce_ckpt3668` | 36 | 0.528 | target 19, tied 8, background 5, other-desc 3, competitor 1 | 0.9905 |
| `fullobj_sorted_pure_ce_ckpt3668` | 36 | 0.528 | target 19, tied 8, competitor 4, background 3, other-desc 2 | 0.9921 |
| `et_rmp_ce_ckpt3664` | 36 | 0.444 | target 16, background 11, tied 5, other-desc 3, competitor 1 | 0.9885 |

By prefix mode:

| prefix_mode | rows | target rate | taxonomy counts |
| --- | ---: | ---: | --- |
| `empty` | 54 | 0.426 | target 23, tied 11, background 8, other-desc 6, competitor 6 |
| `same_desc_good_prefix` | 54 | 0.574 | target 31, background 11, tied 10, other-desc 2 |

In this tiny smoke, adding a same-desc non-target prefix improved post-x1 target-basin rate, but also left non-trivial background/tied mass. This should be treated as a probe signal, not a training recommendation yet.

## Slot-Level Pattern

Before deduplication, slot taxonomy over 162 rows:

| slot | rows | target rate | taxonomy counts |
| --- | ---: | ---: | --- |
| `y1` | 54 | 0.370 | target 20, background 11, other-desc 10, tied 7, competitor 6 |
| `x2` | 54 | 0.759 | target 41, tied 10, background 3 |
| `y2` | 54 | 0.444 | target 24, background 16, tied 14 |

The post-x1 state is not uniformly stable across coordinate slots. `x2` is much more often target-bound than `y1` and `y2` in this sample. This suggests that giving x1 is not sufficient to guarantee stable instance binding through the remaining box trajectory.

## Examples

On `case_id=a33-train-9-bowl-1`, target bbox is `[486, 9, 986, 486]`.

At `desc=bowl, x1=486`, all three checkpoints put high coordinate mass, but `y1` top coordinate is `0`, not target `9`, and is classified as `other_desc_object` because another visible object has a nearby top edge. `x2` is target-bound near `983` vs target `986`. `y2` is near but sometimes falls just outside the strict R95 radius and is classified as `background` for two checkpoints.

This is a useful example because the model is clearly emitting coordinate mass, but the coordinate basin can attach to a different object edge or nearby/background edge.

## Cautions

- Evidence scope is `real_tiny`: 3 shards, `A33_REAL_SLOT_LIMIT=6`, 162 raw slot rows, 108 deduplicated slot rows.
- Prefix mode aliases currently duplicate one semantic same-desc-good prefix unless rows are deduplicated by `(checkpoint_role, prefix_row_id, slot)`.
- The current real runtime probes only post-x1 slots: `y1`, `x2`, `y2`. It does not yet probe pre-x1 candidate cardinality or free greedy continuation.
- GPU usage intentionally avoided busy cards; this smoke used GPUs 0, 1, and 5 because other cards were occupied at launch.
