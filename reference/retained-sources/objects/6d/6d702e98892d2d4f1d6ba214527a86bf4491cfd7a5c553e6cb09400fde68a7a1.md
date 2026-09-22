# COCO22 cumulative expansion — lead accepted

The S main arm completed256 updates on8 GPUs. At saved128 and256, natural greedy covers all376 frozen COCO80 owners with correct classes, FN0 and annotation-relative F1=1.0. Complete-output review is closed at both endpoints. Old227 owners are retained at every saved checkpoint. The earliest sustained saved milestone is128; no claim is made about unsaved intervals.

| Update | Old owners matched | New owners matched | Frozen FN | F1 | Output review |
|---|---|---|---|---|---|
| 0 | 227/227 | 51/149 | 98 | 0.8262 | open |
| 8 | 227/227 | 87/149 | 62 | 0.8626 | open |
| 16 | 227/227 | 89/149 | 60 | 0.8815 | open |
| 32 | 227/227 | 104/149 | 45 | 0.8910 | open |
| 64 | 227/227 | 128/149 | 21 | 0.9556 | open |
| 128 | 227/227 | 149/149 | 0 | 1.0000 | closed |
| 256 | 227/227 | 149/149 | 0 | 1.0000 | closed |

Geometric matched counts at early checkpoints may include wrong/unresolved classes; the separate completion decision includes class correctness. Final output has376 valid predictions, zero unmatched predictions and no unresolved, malformed, invalid, wrong-class, out-of-scope or EOS/cap debt under the frozen evaluator.

## Question and scope

This supports absorbing11 new image conditions through full22-image replay from the prior Sample-arm final256 parameters. It establishes in-sample cumulative fitting with this teacher and recipe; it does not establish held-out generalization, memory-limited continual learning, general instance binding or a visual-sink mechanism. Conditional Source is not triggered because the main arm succeeded. The unit stops here.

The frozen bank contains22 images and376 trusted owners:227 old and149 new. Discovery accounting remains173 PROCESSED and648 HOLD out of821 raw proposals. HOLD is not teacher admission or final-output debt. The versioned historical annotation ledger is preserved, including excluded/ambiguous entries. The user-accepted visibility rule excludes prior-only tiny objects from future obligations; the cited tiny kite was already excluded here, and no denominator changed after results.

## Runtime and acceptance

Fresh AdamW, sample-equal CE plus0.01 shared geometry hinge, fixed geo_sorted_xy, no refresh. All22 images participate in every update:5632 exposures. Eight ranks used qualified microbatch1; readback batch3. Training took22.66 minutes; the controller including saved readbacks took32.76 minutes. There are132 saved-step readbacks plus22 cold step0 requests.

The tuple/list publication defect was reproduced before repair;31 tests then passed and the existing66-request real-entry qualification replay passed without regeneration. Lead replayed terminal/cold-checkpoint validation, collected and scored all saved readbacks, checked per-image decisions, and rehashed the final adapter and frozen data bindings. All native child workers and owned runtime processes have ended. Historical `closeout-v1/final-receipt.json` remains an immutable BLOCKED snapshot; this accepted receipt supersedes its current disposition.

## Exact continuation artifact

Final adapter: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/S/training/checkpoints/step-00256/adapter`

Fingerprint: `513d3daef205c39ae17bf60f7c69959b9881bc2d4f29e7c007090358ac76a2ed`.

Exact config, metrics, diagnostics, test evidence and artifact digests are bound in [lead final receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/lead-final-receipt.json). [Per-image final scoring](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/S-step-00256.json) and [trajectory](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-v1/trajectory-summary.json) preserve reproducible evidence.

## Frozen identities

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/annotations-v5/annotations.jsonl`
  SHA256 `8a3ddfc03edb3064de417e25e444383dfdc83cc1a08a6bbcee08ddd7435e1959`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/data-v1/bank.json`
  SHA256 `270fe47723918a992092b822b2f78ccc2a42ce42160177381fe8a50ec33f93b6`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/evaluation-preparation-v1/preparation.json`
  SHA256 `6cb0a9ac32b389a81d8047f3f4fdcd70c7dd4fd1771e91edbc339eb496fa3224`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/gt-review-admission-v1/lead-admission.json`
  SHA256 `c97615a9635889088b190231446e0d9a42e18b3b5fba62fa94b73bc5cb678c35`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/trial.json`
  SHA256 `742024cc9c3a8ef97c2b340c2ba5156c420b3a00aa47f3def41a7a2d76fbefed`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/S/training-manifest.json`
  SHA256 `0f375c14d8ad44e102fc2051732335b457eda937f1311a51873a547452108c95`
