# Paired natural endpoint readiness

Status: **CPU-ready; trained endpoints still missing. No evaluation model call
or GPU launch was made by this preparation.**

## Frozen comparison

- Both trained arms are compared to the same N16 anchor, not to Stable50.
- The old exposed panel reuses the completed `completion-v11.json` N16
  `scaled_terminal` rows: 640 images. Its existing strata remain visible as
  train11, reference54, legacy-retention319, old-fresh256, and
  remaining-retention575.
- The independent confirmation panel reuses the new accepted N16 anchor: 256
  images. The already frozen source-blind32 IDs are retained unchanged.
- Total per endpoint: 896 images with natural empty history, original exact
  COCO-80 prompt, FP32/SDPA, greedy T0/top-p1/top-k0/RP1, and
  `max_new_tokens=3084` per image.
- Matching/parsing stays on the accepted native path. GT-unmatched predictions
  and later unreviewed blind proposals remain neutral.

The real CPU preparation consumed all 896 existing anchor rows and found 6,493
parsed predictions, 69 parser drops (67 geometry-invalid), 199 strict later-row
repeats, 896 EOS and zero caps. These are N16 anchor facts, not A/B outcomes.

Supersession note: an earlier worker message reported packet SHA256
`9ea2990130fd02dae5bc6ec9cf9aa677a6955fbe17bef33361ebde8b47b1cacb`.
Before root acceptance, that preparation was regenerated in place after adding
the actual 896-row CPU parser/scorer/overlap replay and then request/batch
identity checks; it is no longer present. The current packet below is the only
pending preparation and becomes launchable only through a successful bind. No panel, prompt, anchor row, training input, metric,
or scientific field changed. This note makes that prior overwrite explicit;
future versions must be preserved rather than rewritten in place.

## Bound artifacts

- Packet: `evaluation/paired-preparation/packet.json`, SHA256
  `fa7e156f8d0b19ec36a34c05548258fe443e1a6365079d60e29be610b30e7232`
- Exact reused anchor rows: `evaluation/paired-preparation/n16-anchor-rows-896.jsonl`,
  SHA256 `4b317b225baadcaa98a1d2d6611f727874d102c90d73a5ccc57cc4c831759b32`
- Readiness: `evaluation/paired-preparation/readiness.json`, SHA256
  `dd8e0599ec1e6a5a674d551fffdc62f9ae2cef4efd4ecec0787a7c8caf966ca1`
- Pre-output blind identity freeze:
  `evaluation/paired-preparation/blind-review-freeze.json`, SHA256
  `7890cbc8ee40431c9177d052818266bf78bd8e729d4987b84e58ddf7cd90ccb7`
- Exact sealed training input required by both arms:
  `training/inputs-sealed-v1.json`, SHA256
  `0c3659f2b8e3ac7c0172f72bce6abae74e2e7d0c9776eaa15f463deec4a2f548`

The blind freeze contains one literal source image per review item and no
prediction or physical label. After both natural endpoints cold-validate, the
consumer writes a mixed source-blind queue plus a separately sealed source map.
It does not invent labels; root may route later single-image visual review to
Luna.

### Training-binding transport correction

The pending packet remains byte-identical at SHA256 `fa7e156...`; it was not
regenerated for this correction. Its historical producer binding is code
SHA256 `b4ad279c91a70a6bc73ff3dea3c3c60a543e34c179f93c7cc658ca5de54def55`.
Current bind/producer code is SHA256
`c0b22bd0effe8c930f0f9063ae1ae7c15819f12766ceb91e3457d6321cdd94e9`.
A successful bound packet will record the latter and retain the pending packet
as `base_packet`.

Owner-successor training emits exact two-field bindings
`{path, sha256}`. Evaluation-owned artifacts continue to use
`{path, sha256, size_bytes}`. The bind boundary now explicitly projects only
the frozen evaluation `training_input` to the two training fields, and verifies
all receipt/cold/rank records with the exact training shape plus live path and
SHA256 checks. It does not accept arbitrary extra fields or unverified paths.
The actual integrated-smoke bindings exercise this accepted producer shape in
the test, but its one-update receipt remains rejected as a scientific endpoint.

## Exact remaining boundary

Binding requires **both** actual 256-update/eight-rank training receipts and
their passing cold checks. A receipt must name the correct arm, bind the exact
sealed input, contain all 256 global updates and eight rank receipts, and expose
a byte-verified non-anchor adapter. The integrated one-update smoke is rejected
and cannot become either scientific endpoint.

```bash
python -m probes.owner_successor_scale.paired_evaluation bind \
  --packet /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-preparation/packet.json \
  --receipt-a <FULL_A_ROOT>/receipt.json --cold-a <FULL_A_ROOT>/cold-check.json \
  --receipt-b <FULL_B_ROOT>/receipt.json --cold-b <FULL_B_ROOT>/cold-check.json \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-preparation/packet-bound.json
```

After root separately accepts and schedules the bound endpoint, run each arm
once on eight GPUs (the concrete IDs remain root's runtime choice):

```bash
python -m probes.owner_successor_scale.paired_evaluation launch \
  --packet <packet-bound.json> --arm A --gpus 0,1,2,3,4,5,6,7 --output <A_NATURAL_ROOT>
python -m probes.owner_successor_scale.paired_evaluation merge \
  --packet <packet-bound.json> --arm A --output <A_NATURAL_ROOT>

python -m probes.owner_successor_scale.paired_evaluation launch \
  --packet <packet-bound.json> --arm B --gpus 0,1,2,3,4,5,6,7 --output <B_NATURAL_ROOT>
python -m probes.owner_successor_scale.paired_evaluation merge \
  --packet <packet-bound.json> --arm B --output <B_NATURAL_ROOT>

python -m probes.owner_successor_scale.paired_evaluation consume \
  --packet <packet-bound.json> \
  --rows-a <A_NATURAL_ROOT>/rows.jsonl --rows-b <B_NATURAL_ROOT>/rows.jsonl \
  --output <PAIRED_CONSUMER_ROOT>
```

The cold consumer rejects prompt/media/grid/token/text/parser/scorer/overlap,
row-count/order, adapter, and packet drift. It emits per-arm and per-stratum
owner gain/retained/loss ledgers; annotated versus neutral first-owner
preservation; strict `IoU>0.95` later-row repeats plus 0.90/0.80 drift
diagnostics; raw/valid/invalid/malformed counts; EOS/caps; and owner changes by
output-position bin. No result automatically promotes a checkpoint.

## Verification

```text
pytest -q probes/owner_successor_scale/tests/test_paired_evaluation.py
5 passed

python -m probes.owner_successor_scale.paired_evaluation prepare --output .../evaluation/paired-preparation
anchor_images=896, blind_images=32, model_calls=0

git diff --check -- probes/owner_successor_scale/paired_evaluation.py probes/owner_successor_scale/tests/test_paired_evaluation.py
clean
```
