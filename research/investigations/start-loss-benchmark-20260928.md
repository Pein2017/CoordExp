# Fourth-loss continuation benchmark

Status: CPU checks passed; source decode reproduced; final nonzero-update smoke in progress.

From the untied geo_sorted_xy three-loss step-2444 checkpoint, does an additional
GT-prefix onset objective improve natural greedy COCO val200 detection after
matched short continuation? The unchanged three-loss continuation is the main
control; extra onset CE distinguishes novel objective value from simple emphasis.

The source is `/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-illegal-mass001-ebs24-4epoch/checkpoints/step-2444`.
Its language DoRA and both independent special-token embedding deltas must load.
Use the existing warm-start copier with unchanged language-only topology; require
no newly initialized or ignored adapter tensors. Optimizer and scheduler restart
identically in every arm. This is not exact continuation of the old optimizer.

The authoritative frozen input/config and budget record is
`/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/protocol.json`.
Raw outputs, calibration, logs, process receipts and metrics live beside it.
Source owner: `scripts/start_loss_benchmark.py`, `src/losses/start_coordinate.py`.
No production promotion, rollout training, unknown-object negative labeling,
inference masking, or modification of the historical checkpoint is authorized.

## Fixed contrast

- CE 1, type gate 0.1, conditional illegal mass 0.01 remain fixed.
- Arms: unchanged; extra x1/y1 coordinate-family CE; local coordinate mass;
  GT-instance onset margin. The CE arm is conditional on the coordinate family,
  so it does not also reweight type supervision.
- The same 8192 training images, with two independently shuffled image orders
  (17 and 29); object order stays x then y. No validation-driven cohort selection.
- Each arm/order: up to 256 updates, checkpoint 64 diagnostic and 256 primary.
  Fresh AdamW, language LR 2e-5 / independent token deltas 1e-5, same cosine
  schedule, global batch 24, packing 12000, BF16; two concurrent four-GPU jobs.
- Decode: HF, greedy, RP 1.0, max_new_tokens 3084, batch 4/device,
  coordinate_output_norm median. All use the exact training prompt and val200.
- Primary: final COCO mAP .50:.95. Secondary: class-aware FN at IoU .5,
  maxDets100, AP50/AP75, parser/geometry drops, exact repeats and length caps.
  Matched direction in both orders is exploratory candidate evidence only.
- User budget: 16 wall hours / at most 8 simultaneous GPUs, bounded at
  2026-09-28 19:21 UTC (128 GPU-hour ceiling). Stop affected work on numerical,
  checkpoint, identity or evaluation failure; no automatic scope expansion.

## Objective and calibration semantics

All additions average over both onset slots of complete objects, then images,
using the existing segment-balanced global/DDP reducer. Ineligible margin slots
and empty-support images contribute zero without disappearing from denominators.
The neighborhood radius is min(4, floor(0.02 * GT axis extent)). A neighborhood
is positive localization supervision, not a certificate that all boxes meet IoU.

For margin 0.2, compare the current GT instance's highest neighborhood logit
with the highest disjoint neighborhood logit from another same-description GT
instance in the same image. At y1 the competitor must also be compatible with
the actually supplied x1. Shared/overlapping coordinate projections are skipped.
Other true instances are competing labels under the fixed serialized target;
this does not call them physically illegal or infer any unknown-object label.

Before training, use one real four-rank, global-batch24 smoke step. All its
micro-forwards precede the first optimizer update. Log exact squared gradients
with respect to coordinate logits, including equal-max subgradients. Set each
candidate's fixed weight to 0.1 times CE norm divided by candidate norm. This
equalizes initial coordinate-logit gradient magnitude on that calibration batch,
not parameter gradients or later dynamics. Zero support blocks that candidate's
calibration; it is not a scientific negative. The smoke checkpoint is discarded
as an experimental anchor and used only for cold-load inference qualification.

## Prior evidence and acceptance

Earlier Gaussian/IoU-Gibbs trials did not establish superiority over their CE
controls. Their ordering, RP and supervision differ from this comparison.
Prior expectation hinge versus illegal mass also changed type-gate status, so
it does not isolate geometry. The new comparison holds all three base losses
fixed and directly evaluates free decode, not teacher-forced token accuracy.

CPU checks cover causal rows, same-image/object/description association,
overlapping coordinate aliases, empty support, segment normalization, gradient
direction and analytic calibration versus autograd. Before full execution,
qualify four-rank warm-start -> update -> untied checkpoint -> fresh HF decode
and scoring. Fresh source evaluation provides a current-runtime anchor.

## Qualification receipts

- Source repeat reproduces mAP 0.4556510010795014, FN50 614, strict repeats 460,
  invalid geometry 220 and two length caps.
- 103 CPU loss/config tests pass. Cross-image competitor mutation is rejected.
- Initial launch failures (required determinism environment, JSONL-relative image
  references, explicit packing preparation) were repaired before any benchmark
  arm. Original failed artifacts are retained under the output root. Data path
  repair does not change image identities, contents, annotations or order.
- Calibration uses 24 micro-forwards across four ranks before any nonzero update.
  Frozen weights: CE 0.1, local mass 0.22967969404405367, instance margin
  0.6663237728525236. All three base losses remain 1 / 0.1 / 0.01.
- The calibration smoke's first scheduler LR is zero. Its saved 588 adapter
  tensors and both delta tensors are bitwise equal to the source, which proves
  warm-start preservation but does not prove parameter mutation. A separate
  margin-update smoke uses zero warmup and nonzero LR solely to qualify actual
  backward/update/persistence/cold decode; it is not an experimental anchor.
- Parallel CPU preparation materializes the two 8192-row caches; each group's
  arms reuse its existing cache. No GPU is needed for preparation.
