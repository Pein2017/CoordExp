# Grid Anchor Compact Detection Pending Record

Date: 2026-05-31

Status: pending / deprecated as a primary direction

Branch/worktree used for the attempt:

- Branch: `codex/grid-anchor-compact-detection`
- Worktree: `/data/CoordExp/.worktrees/grid-anchor-compact-detection`
- Main checkout policy: implementation stayed isolated in the worktree. Main should not contain the grid-anchor implementation after cleanup.

## Summary

We explored an anchor-first formulation for autoregressive V-LLM object detection on COCO. The intended compact row format was:

```text
<|object_ref_start|><|grid_anchor_i|>{desc}<|box_start|>{x1}{y1}{x2}{y2}
```

The anchor was designed as a coarse residual-state / coverage-state handle, not a perfect instance ID. The intended behavior was:

1. Select a spatial basin.
2. Describe/refine one object in that basin.
3. Make already covered regions easier to downweight in later autoregressive steps.

The implementation reached a working G16 prototype with strict template/profile handling, offline vocabulary expansion, grid-anchor-aware token rows, residual object-count anchor supervision, natural decoding, and bbox-only metric artifacts. However, the G16 experiments showed a large detection quality regression relative to the non-anchor compact baseline. The degradation shape suggests that mandatory anchor-first generation moved the difficulty from object selection to exact spatial-token selection, then still left object selection and stop/continue calibration unresolved.

Current recommendation:

- Pause/deprecate mandatory grid-anchor-first as the main research direction.
- Do not launch another full production run for G16.
- Do not spend a full production run on G8 unless a tiny diagnostic first gives an unexpectedly strong signal.
- Redirect the next serious work to non-anchor residual-set objectives, object ordering, EOS/continuation calibration, dense-scene slices, and possibly prefix-rollin.

## Implemented Design In The Worktree

The worktree implementation covered the following intended surfaces:

- Shared grid-anchor resolver in `src/detection/grid_anchor.py`.
- Flat anchor tokens `<|grid_anchor_i|>`.
- Supported grid sizes in the design: `{12, 14, 16, 20}` with G16 used for production experiments.
- Anchor assignment from `norm1000_xyxy` bbox centers, not image pixels and not Qwen patch-token coordinates.
- Offline vocabulary expansion and verification tools:
  - `scripts/tools/expand_grid_anchor_vocab.py`
  - `scripts/tools/verify_grid_anchor_vocab.py`
- Distinct profile `compact_full_grid_anchor`.
- Newline-free grid-anchor rendering.
- Strict grid-anchor parser diagnostics.
- Token-row validation for `1000 + 2 + G*G` rows.
- G16 row count: `1258`.
- Persisted `coord_offset_adapter.coord_ids` for compatibility; the adapter key was not renamed.
- Runtime/checkpoint/eval metadata carrying grid size and vocabulary fingerprint.
- `TokenRole.GRID_ANCHOR` and `SemanticRole.GRID_ANCHOR`.
- Residual object-count weighted multi-positive anchor target:

```text
q(anchor | prefix) = count(remaining objects in anchor) / count(remaining objects)
```

- Natural decoding only. No constrained decoding policy was added.
- Metric-bearing artifacts remained bbox-only after successful strict parse.

Important vocabulary artifact:

- Expanded G16 model/tokenizer:
  `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-grid-anchor-g16`
- Verified G16 fingerprint:
  `ad3b0aaba075b322dddf88c753d6c472467e12803e49c354fa4633d69acc8fb3`

## Dataset And Config Notes

The first production attempt initially risked pointing at a legacy max-object artifact. We corrected this by regenerating/materializing a grid-anchor-aware no-cap G16 COCO artifact:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_grid_anchor_g16_len12000
```

Interpretation preserved:

- Raw COCO JSONL records remain model-independent.
- Grid anchors are rendered from normalized bbox centers by the template.
- Grid anchors are not embedded directly into raw object records.

No-cap evidence:

- The first 200 validation rows matched the legacy baseline split exactly.
- First 200 validation rows contained `1444` GT objects.
- First 200 validation rows had max object count `53`.
- The full regenerated artifact retained dense rows above the old max-object cap.

## Main Baselines And Results

### Non-Grid Reference

Best relevant non-anchor natural baseline:

```text
/data/CoordExp/outputs/infer/recursive_detection_ce/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_natural_legacyparse_clean
```

Val200 summary:

| Metric | Value |
|---|---:|
| AP | `0.422314` |
| AP50 | `0.564169` |
| AP75 | `0.447855` |
| AR100 | `0.464796` |
| F1ish@0.50 loc micro | `0.599604` |
| Guarded AP | `0.408285` |
| Guarded AP50 | `0.545490` |
| Predictions | `1115` |
| Parser/errors | `2` |
| Duplicate suppressions | `242` |

This baseline matters because it is not evidence that residual-set / multiple-positive ideas are broken. It suggests the non-anchor residual/support-style objective can work well.

### G16 Balance-Heavy Production Checkpoint

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce/compact_full_grid_anchor_g16_len12000_et_rmp_ce_balance2_bsz16_4epoch_tokenrows_v1/compact-full-grid-anchor-g16-len12000-et-rmp-ce-balance2-bsz16-4epoch-tokenrows-v1/v1-20260530-191218/checkpoint-3668
```

Val200 inference artifact:

```text
/data/CoordExp/outputs/infer/recursive_detection_ce/compact_full_grid_anchor_g16_balance2_ckpt3668_val200_bsz4_temp0_rep1p05_clean
```

Val200 summary:

| Metric | Value |
|---|---:|
| AP | `0.245636` |
| AP50 | `0.348411` |
| AP75 | `0.262514` |
| AR100 | `0.277104` |
| F1ish@0.50 loc micro | `0.428150` |
| Guarded AP | `0.236044` |
| Guarded AP50 | `0.331663` |
| Predictions | `982` |
| Parser/errors | `14` |
| Duplicate suppressions | `197` |

Parser/error categories observed:

- `empty_pred`: `7`
- `invalid_geometry`: `5`
- `misplaced_grid_anchor`: `1`
- `wrong_coord_arity`: `1`

Anchor diagnostics:

| Diagnostic | Value |
|---|---:|
| Anchor tokens | `1054` |
| Pred anchor in GT-anchor set | `0.279886` |
| Near anchor, L1 <= 1 | `0.714421` |
| GT unique anchor coverage | `0.226054` |
| Anchor vs predicted box-center exact | `0.528681` |

### Repetition Penalty Sweep

Repetition penalty did not explain the degradation:

| RP | AP |
|---:|---:|
| `1.00` | `0.2106` |
| `1.05` | `0.2456` |
| `1.10` | `0.2346` |

Conclusion: RP changes the score but does not close the gap to the non-grid baseline.

### G16 Support-Heavy Warm Restart

Hypothesis tested:

- Balance-heavy `support:balance = 1:2` may have undertrained valid residual support.
- Try support-heavy `support:balance = 2:1` from the G16 checkpoint.

Important training correction:

- Plain resume restored near-zero learning rate because global step was also resumed.
- Correct warm restart used `resume_only_model: true` and `ignore_data_skip: true`.

Training output:

```text
/data/CoordExp/.worktrees/grid-anchor-compact-detection/temp/recursive_detection_ce/output/grid_anchor_g16_continue_support2_from_balance2_200steps/grid-anchor-g16-continue-support2-from-balance2-200steps/v2-20260531-104127
```

Teacher-forced eval diagnostics:

| Step | Support Mass | Anchor Top1 | Anchor Top5 | Grid CE | Eval Loss |
|---:|---:|---:|---:|---:|---:|
| `50` | `0.129854` | `0.163981` | `0.416589` | `3.905365` | `1.325141` |
| `100` | `0.137944` | `0.169809` | `0.424129` | `3.858662` | `1.299586` |
| `150` | `0.142918` | `0.176971` | `0.433680` | `3.810154` | `1.283999` |
| `200` | `0.142457` | `0.180706` | `0.428617` | `3.801848` | `1.281457` |

Reference old balance-heavy final:

| Diagnostic | Value |
|---|---:|
| Support mass | `0.103030` |
| Anchor Top1 | `0.170678` |
| Anchor Top5 | `0.428815` |
| Grid CE | `3.727829` |

Interpretation:

- Support-heavy training improved residual support mass.
- It slightly improved anchor top1.
- It did not resolve the detection gap.

Best support-heavy warm restart inference:

```text
/data/CoordExp/outputs/infer/recursive_detection_ce/compact_full_grid_anchor_g16_continue_support2_ckpt150_val200_bsz4_temp0_rep1p05
```

Val200 summary for checkpoint 150:

| Metric | Value |
|---|---:|
| AP | `0.256974` |
| AP50 | `0.368957` |
| AP75 | `0.254914` |
| AR100 | `0.283362` |
| F1ish@0.50 loc micro | `0.468492` |
| Guarded AP | `0.248027` |
| Guarded AP50 | `0.355821` |
| Guarded F1ish@0.50 loc micro | `0.441958` |
| Predictions | `900` |
| Parser/errors | `12` |
| Duplicate suppressions | `164` |

Anchor diagnostics:

| Diagnostic | Value |
|---|---:|
| Anchor tokens | `994` |
| Pred anchor in GT-anchor set | `0.346076` |
| Near anchor, L1 <= 1 | `0.771630` |
| GT unique anchor coverage | `0.233716` |
| Anchor vs predicted box-center exact | `0.529708` |

Checkpoint 200 was not better:

| Metric | Value |
|---|---:|
| AP | `0.246793` |
| AP50 | `0.352354` |
| AR100 | `0.271171` |
| Predictions | `903` |
| Parser/errors | `10` |

## What Was Ruled Out

The investigation did not find evidence that the main degradation was caused by:

- Wrong validation rows for val200.
- The old max-object cap in the val200 comparison.
- Repetition penalty.
- Gross G16 vocabulary expansion failure.
- Gross adapter row-count mismatch.
- Missing G16 token rows.
- A constrained decoding mismatch, because the experiments used natural decoding as intended.

The degradation shape was coherent with a method/surface problem:

- Fewer predictions.
- Lower AR100.
- More parse/geometry friction than non-grid.
- Anchor near-cell accuracy much better than exact-cell usefulness.
- Only about half of predicted boxes exactly matched the emitted anchor cell.
- Support-heavy training improved internal anchor support metrics but not enough detection quality.

## Working Diagnosis

The mandatory G16 anchor token appears to act as a brittle extra commitment at the most fragile point of generation.

Original non-anchor problem:

```text
Which remaining object should be emitted next?
```

Grid-anchor formulation changed this into:

```text
Which remaining spatial basin should be emitted next?
Which object inside that basin should be emitted?
Will the later description/box stay coupled to the earlier anchor?
Should decoding continue or stop?
```

This moved the difficulty rather than removing it.

The G16 token space likely made the first decision too fine:

- G16 has `256` anchors.
- Cell width is about `1000 / 16 = 62.5` normalized coordinate units.
- The model often predicts a nearby cell but not the exact cell.
- The anchor is supposed to be a coarse basin, but G16 behaved closer to a semi-precise coordinate precondition.

Because decoding is natural, the anchor must be learned as a real generative habit. It cannot rely on constrained decoding. The evidence suggests it was neither fully ignored nor fully obeyed. That middle state is costly:

- It adds one more mandatory token per object.
- It raises the cost of continuing.
- It can increase under-generation.
- It duplicates spatial information already present in bbox tokens.
- It only weakly binds the final bbox under natural decoding.

## Why G8 Was Not Pursued As A Production Relaunch

G8 might reduce the exact-token burden:

- G8 has only `64` anchors.
- Cell width is `125` normalized coordinate units.
- Boundary jitter and exact-anchor prediction should be easier.

However, G8 also weakens the binding value:

- More objects per cell.
- More same-category collisions.
- More dense-scene ambiguity.
- More reliance on the later description and box tokens.

Expected risk:

```text
G8 may improve anchor accuracy without improving detection AP/AR.
```

Therefore G8 is not recommended as a full production relaunch. If reopened, G8 should first be a tiny diagnostic only.

## Current Decision

Pause and deprecate mandatory grid-anchor-first as a primary direction.

Keep the conceptual lesson:

- Spatial coverage supervision may still be useful.
- Anchor-as-auxiliary may still be useful.
- Coarse region hints may still be useful.

But the current mandatory anchor-first surface is not the best next research use of compute.

## Suggested Reopen Criteria

Only reopen grid-anchor-first if a small diagnostic experiment shows clear detection gains, not just better anchor metrics.

Minimum suggested criteria for a G8 or G12 diagnostic:

| Metric | Required Signal |
|---|---|
| AP | Clearly above G16 support-heavy `0.257`; preferably above `0.30` on val200 |
| AP50 | Preferably above `0.40` |
| AR100 | Clearly above `0.283`; preferably above `0.33` |
| Prediction count | Closer to non-grid `1115`; preferably above `1000` on val200 |
| Parser errors | Not worse than G16 |
| Duplicate suppressions | Not exploding from coarse-cell collisions |
| Anchor-box consistency | Clearly better than G16's roughly `0.53` |
| Detection coupling | Anchor metric improvements must translate to AP/AR improvements |

If anchor metrics improve but detection does not, the anchor is learnable but not useful enough as a mandatory output token.

## Recommended Next Research Direction

The next serious branch should focus on the original failure directly:

```text
Can training make continuing with a valid remaining object feel as safe as stopping?
```

Recommended next work:

1. No-anchor objective triage:
   - plain CE
   - random-permutation CE
   - residual multiple-positive support-heavy
2. Spatial ordering baseline:
   - coarse spatial row-major ordering without anchor tokens
3. EOS / continuation diagnostics:
   - EOS probability/rank after each object
   - aggregate valid remaining-object mass vs EOS
   - generated count vs GT count
4. Dense-scene slices:
   - `1-5`, `6-10`, `11-20`, `20+` GT object buckets
5. Prefix-rollin / committed-prefix residual training:
   - train recovery after imperfect generated prefixes
6. Annotation incompleteness probe:
   - manually inspect dense-scene false positives and missed objects

These directions preserve the core residual-set research idea while avoiding a mandatory extra spatial token that the current experiments show can become a new bottleneck.

## Cleanup Note

The implementation worktree was intentionally isolated. After this record is exported, the worktree can be removed without bringing grid-anchor implementation edits into `main`.

Persistent artifacts outside the worktree may remain useful for comparison:

- G16 expanded tokenizer/model in `model_cache`.
- G16 no-cap COCO artifact in `public_data/coco`.
- Stage-1 G16 checkpoints in `outputs/stage1_2b`.
- Val200 inference/eval artifacts in `outputs/infer`.

