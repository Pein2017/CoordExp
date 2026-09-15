# COCO GT-correction cold evaluation and reduction qualification

Date: 2026-09-07

Status: **candidate mechanics QUALIFIED; scientific status remains
`MECHANICS_ONLY_NO_MODEL_QUALITY_CLAIM`.** No final checkpoint or production
train256/dev128 evaluation was launched here.

## Implemented boundary

`scripts/research/eval_coco_gt_correction.py` is the probe-specific cold native
HF entry. It clones the sealed Source resolved config and changes only the
input/output locations, source-gate location, and declared checkpoint payload.
The entry validates checkpoint, live bank-manifest, Source composition, update,
surface, payload hash and the exact W row universe before model loading. DoRA
uses the saved adapter; W keeps the immutable Source adapter and installs the
saved FP32 output-row residual. Native generation and the existing scorer are
unchanged.

The cold receipt records arm, checkpoint ID, bank ID, Source identity, update
and payload identity in `model_identity.coco_gt_correction`. W decoding fails if
the hook is not invoked. Optimizer state is never loaded. Current native
embedding loading requires the accepted source-gate evidence, so the entry
defaults to `<checkpoint parent>/source-gate` and allows an explicit override.

`scripts/research/reduce_coco_gt_correction.py` validates every complete scored
artifact before applying the unchanged detection consumer. It requires the
sealed bank and Source-topology receipt, exact train/dev paths and hashes,
FP32-SDPA HF generation, batch 2, 3,084-token cap, repetition penalty 1.0,
eight active ranks, expected cold checkpoint identity, and identical all-image
GT/owner denominators. It reports exact gained/lost owner IDs at IoU50/60/80,
selected/later-owner train cohorts, unmatched predictions, duplicates, invalid
and dropped rows, EOS/cap counts and generated lengths. Unmatched predictions
remain a monitor, not automatic hallucinations.

## Cold evidence

Durable root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/qualification/cold-consumer-v1/`.

Source reuse is sealed separately in
`qualification/source-eval-topology.json`, SHA256
`9e805279a7fde9574260ab4888120c452088158e1747ccf02953961246f43a21`.
It binds both historical Source runs to their split inputs, raw/run-manifest
hashes, exact model/adapter/embedding identities, FP32 SDPA generation and the
eight-rank, per-device-batch-2 topology.

### Actual W payload

Receipt: `cold-consumer-v1/w-payload-qualification.json`, SHA256
`57c44202941f010cb9f4dc1bb5b332950b1b7e0d3155a8eaf186052be435d863`.

- Checkpoint ID:
  `f35245868170f233a7be3209e61183f1460bf7dfe381be47ce499fedc6f1d10c`.
- Payload: FP32 `[1136, 2048]`, 2,326,528 nonzero entries, L2 norm
  `0.014886380173265934`.
- A witness from the actual saved payload gives a nonzero selected-logit delta;
  the consumer's delta equals `hidden @ residual_rows.T` exactly. Unselected
  logits and same-prefix hidden state are bit-exact unchanged. A zero hook and
  hook removal both restore Source exactly.
- Missing, wrong-arm and unused payload cases fail in the focused caller tests.

### Native decode and evaluator

| Surface | Run | Result |
|---|---|---|
| W | `cold-consumer-v1/w/w-fresh-step1-image368-v5/` | completed; one scored image, 130 generated tokens, 14 scoreable predictions, native `im_end`; run-manifest SHA256 `d8fe907400babfba82701416ede7df423f7ee7df9aa1d98fc4627b9e4a83fbf6` |
| DoRA/R | `cold-consumer-v1/dora/r-fresh-step1-image368-v1/` | completed; one scored image, 130 generated tokens, 14 scoreable predictions, native `im_end`; run-manifest SHA256 `e59ce0c1c1619dd138245e9469603593315c2639eeee6352c8139682ecc055c9` |
| W two-rank | `cold-consumer-v1/w-two-rank/w-fresh-step1-four-image-v2/` | completed controller/worker merge; ranks 0/1 used physical tokens 2/3, four scored images, 296 generated tokens, 32 scoreable predictions, all native `im_end`; run-manifest SHA256 `1d53d60f119af3da98ea7a500f54c4596917347c0095290d78cee214d3bb57e0` |

The two-rank slice exercised the actual custom worker launcher, cold-loaded the
same W checkpoint independently in both workers, preserved its checkpoint ID in
both shard manifests, and passed the reducer's topology/payload validator.

Qualification exposed and fixed three real integration mismatches before the
accepted runs: generated leaf configs must pass through the canonical config
namespace; current special-token embedding loading requires the source gate;
and the worker environment builder requires an explicit `base_env`. Earlier
failed attempt directories remain under `cold-consumer-v1/w/` or
`cold-consumer-v1/w-two-rank/`; none is an accepted result.

## Fresh checks

```text
conda run -n ms pytest -q \
  tests/research/test_eval_coco_gt_correction.py \
  tests/research/test_reduce_coco_gt_correction.py \
  tests/research/test_train_coco_gt_correction.py
# 8 passed

conda run -n ms python -m py_compile \
  scripts/research/eval_coco_gt_correction.py \
  scripts/research/reduce_coco_gt_correction.py
```

The real Source train run, real one-rank W/R runs and real two-rank W merge all
pass the reducer's topology validator. The historical Source train run also
passes the sealed reuse receipt check.

## Fixed production evaluation packet

Run only after the lead accepts four valid update-64 checkpoints and reconciles
GPU/output ownership. Each candidate evaluation uses all eight visible GPUs;
run the arms serially rather than overlapping them.

```bash
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1
BANK="$ROOT/bank/manifest.json"
TRAIN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl
DEV=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/dev.jsonl

for ARM in R B M W; do
  for SPLIT in train dev; do
    if [ "$SPLIT" = train ]; then
      INPUT="$TRAIN"; PANEL=train256
    else
      INPUT="$DEV"; PANEL=dev128
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n ms \
      python scripts/research/eval_coco_gt_correction.py run \
      --checkpoint "$ROOT/$ARM/checkpoint-000064" \
      --arm "$ARM" \
      --bank-manifest "$BANK" \
      --expected-completed-update 64 \
      --input-jsonl "$INPUT" \
      --artifact-root "$ROOT/evaluation/$ARM/$SPLIT" \
      --run-name "${ARM}-${PANEL}-native-v1" \
      --expected-active-ranks 8
  done
done
```

After all eight candidate runs are complete, reduce them against the sealed
Source pair:

```bash
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1
SOURCE_ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1
TRAIN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl
DEV=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/dev.jsonl

conda run -n ms python scripts/research/reduce_coco_gt_correction.py \
  --train-input "$TRAIN" --dev-input "$DEV" \
  --train-run "Source=$SOURCE_ROOT/qwen3-vl-2b-sft256-source-train256-natural-v1" \
  --train-run "R=$ROOT/evaluation/R/train/R-train256-native-v1" \
  --train-run "B=$ROOT/evaluation/B/train/B-train256-native-v1" \
  --train-run "M=$ROOT/evaluation/M/train/M-train256-native-v1" \
  --train-run "W=$ROOT/evaluation/W/train/W-train256-native-v1" \
  --dev-run "Source=$SOURCE_ROOT/qwen3-vl-2b-sft256-source-dev128-natural-v1" \
  --dev-run "R=$ROOT/evaluation/R/dev/R-dev128-native-v1" \
  --dev-run "B=$ROOT/evaluation/B/dev/B-dev128-native-v1" \
  --dev-run "M=$ROOT/evaluation/M/dev/M-dev128-native-v1" \
  --dev-run "W=$ROOT/evaluation/W/dev/W-dev128-native-v1" \
  --bank-manifest "$ROOT/bank/manifest.json" \
  --source-topology-receipt "$ROOT/qualification/source-eval-topology.json" \
  --expected-active-ranks 8 --expected-completed-update 64 \
  --evaluation-root "$ROOT/evaluation/detection-reduction-v1" \
  --out "$ROOT/evaluation/portfolio-reduction-v1.json"
```

This packet authorizes no launch by itself. Terminal model-quality evidence and
interpretation remain owned by `unit.md` and the lead.
