---
title: Existing-COCO GT-correction portfolio results
description: Fixed four-arm pilot outcome and verified natural-evaluation readout.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-07-coco-gt-correction-portfolio
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# Existing-COCO GT-correction portfolio: verified result readout

Date: 2026-09-07
Status: **lead-accepted bounded scientific readout; fixed pilot complete; no promotion**
The lead freshly replayed the saved-artifact verifier below, including all 42
paired contrasts and all-image/checkpoint identity checks. [Owning unit](unit.md).

## Decision

Only **M (selected-owner-only suffix loss)** meets the frozen `observed
promising pilot` rule. On the fixed dev128 / 891-owner set it changes IoU50
coverage by **29 gains, 21 losses, net +8** versus Source, while IoU60 and
IoU80 are also nonnegative (net `+3`, `+3`) and strict duplicates, invalid
predictions, and capped outputs do not increase (`3/0/0` versus Source
`7/0/0`). This is a single-seed, fixed-dose pilot result, not robust
generalization or promotion evidence.

R and B show train recovery but no dev IoU50 recovery and are therefore
**memorization-compatible**; neither is promising. R is additionally
**mixed** across dev thresholds (`-5/+1/-5`). W is **mixed**: dev IoU50 is
`+7` versus Source, but IoU60/80 are `-1/-5`. B is harmful at this recipe: it
emits much longer outputs with more duplicates, caps, drops, and unmatched
predictions while losing annotated-owner coverage.

## Owner coverage and debt

Each cell is `matched / debt`, where debt is the fixed annotated-owner
denominator minus matched owners. All 256 train images / 1,955 owners and all
128 dev images / 891 owners remain in the denominators, including capped
rows.

| Split | Arm | IoU50 | IoU60 | IoU80 |
|---|---|---:|---:|---:|
| train | Source | 1259 / 696 | 1190 / 765 | 908 / 1047 |
| train | R | 1321 / 634 | 1241 / 714 | 947 / 1008 |
| train | B | 1292 / 663 | 1222 / 733 | 947 / 1008 |
| train | M | 1314 / 641 | 1230 / 725 | 958 / 997 |
| train | W | 1270 / 685 | 1194 / 761 | 925 / 1030 |
| dev | Source | 614 / 277 | 585 / 306 | 451 / 440 |
| dev | R | 609 / 282 | 586 / 305 | 446 / 445 |
| dev | B | 576 / 315 | 547 / 344 | 429 / 462 |
| dev | M | **622 / 269** | **588 / 303** | **454 / 437** |
| dev | W | 621 / 270 | 584 / 307 | 446 / 445 |

## Paired owner gains and losses

Cells are `gains / losses = net` on identical owner IDs; aggregate-count
subtraction alone was not used.

### Versus Source

| Split | Arm | IoU50 | IoU60 | IoU80 |
|---|---|---:|---:|---:|
| train | R | 103 / 41 = +62 | 95 / 44 = +51 | 95 / 56 = +39 |
| train | B | 96 / 63 = +33 | 95 / 63 = +32 | 105 / 66 = +39 |
| train | M | 104 / 49 = +55 | 99 / 59 = +40 | 103 / 53 = +50 |
| train | W | 63 / 52 = +11 | 57 / 53 = +4 | 64 / 47 = +17 |
| dev | R | 31 / 36 = -5 | 32 / 31 = +1 | 32 / 37 = -5 |
| dev | B | 33 / 71 = -38 | 31 / 69 = -38 | 28 / 50 = -22 |
| dev | M | **29 / 21 = +8** | **27 / 24 = +3** | **32 / 29 = +3** |
| dev | W | 28 / 21 = +7 | 23 / 24 = -1 | 29 / 34 = -5 |

### Versus R (the primary policy/supervision/surface contrasts)

| Split | Contrast | IoU50 | IoU60 | IoU80 |
|---|---|---:|---:|---:|
| train | B-R | 49 / 78 = -29 | 52 / 71 = -19 | 66 / 66 = 0 |
| train | M-R | 43 / 50 = -7 | 42 / 53 = -11 | 59 / 48 = +11 |
| train | W-R | 47 / 98 = -51 | 48 / 95 = -47 | 73 / 95 = -22 |
| dev | B-R | 25 / 58 = -33 | 18 / 57 = -39 | 18 / 35 = -17 |
| dev | M-R | **25 / 12 = +13** | **20 / 18 = +2** | **25 / 17 = +8** |
| dev | W-R | 42 / 30 = +12 | 34 / 36 = -2 | 41 / 41 = 0 |

The primary mechanism contrast therefore favors focusing correction loss on
the selected missing row (M-R), not backfilling after an already-emitted
crossing row (B-R). The strongest alternative explanation remains finite
training-string memorization plus output-distribution changes: R/B recover
train owners without transferring at dev IoU50, and B's degradation travels
with extreme length/unmatched growth. M's clean dev vector is evidence against
*purely* train-string memorization for that arm, but one fixed dev set and one
seed cannot establish generalization mechanism.

## Selected- and later-owner train diagnostic

Matched counts at `IoU50 / IoU60 / IoU80`; denominators are 160 selected
owners and 1,201 later owners. Source selected counts are zero by construction
of the frozen missed-owner cohort.

| Arm | Selected (160) | Later (1201) |
|---|---:|---:|
| Source | 0 / 0 / 0 | 665 / 622 / 433 |
| R | 31 / 26 / 15 | 705 / 650 / 450 |
| B | 25 / 23 / 13 | 683 / 638 / 456 |
| M | **41 / 34 / 24** | 688 / 631 / 446 |
| W | 16 / 11 / 6 | 671 / 622 / 446 |

M has the strongest selected-owner recovery, but its later-owner IoU50/60
counts are below R. The aggregate M-R dev advantage therefore does not prove
that full-remainder supervision is generally harmful; it only identifies the
better arm at this fixed recipe.

## Output diagnostics

`Dup` is the reducer's strict duplicate-candidate count; `amb` is ambiguous
duplicate attribution; `drop` is parser-dropped predictions; `cap` is
cap/nontermination images; `EOS` is natural `<|im_end|>` stops; `len` is mean
/ maximum generated tokens; `unmatched50` is category-compatible predictions
unmatched to annotations at IoU50.

| Split | Arm | Dup / amb | invalid / drop | cap / EOS | len mean / max | unmatched50 |
|---|---|---:|---:|---:|---:|---:|
| train | Source | 17 / 11 | 0 / 794 | 4 / 252 | 115.96 / 3084 | 1064 |
| train | R | 7 / 16 | 2 / 1232 | 5 / 251 | 136.36 / 3084 | 1131 |
| train | B | 54 / 14 | 0 / 2650 | 13 / 243 | 240.12 / 3084 | 2601 |
| train | M | 348 / 15 | 0 / 1571 | 7 / 249 | 160.86 / 3084 | 1506 |
| train | W | 20 / 12 | 0 / 1306 | 5 / 251 | 136.82 / 3084 | 1116 |
| dev | Source | 7 / 8 | 0 / 58 | 0 / 128 | 86.48 / 788 | 489 |
| dev | R | 12 / 6 | 0 / 307 | 1 / 127 | 107.80 / 3084 | 551 |
| dev | B | 26 / 9 | 0 / 1107 | 6 / 122 | 247.02 / 3084 | 1706 |
| dev | M | **3 / 7** | **0 / 14** | **0 / 128** | **84.27 / 734** | 497 |
| dev | W | 5 / 9 | 0 / 202 | 1 / 127 | 120.69 / 3084 | 806 |

M's dev unmatched50 count is `+8` versus Source while annotated matches are
also `+8`; under partial COCO annotation, `unmatched` is an annotation-relative
status and is **not** a hallucination count. The larger train duplicate count
for M is reported rather than hidden; the frozen promising rule is owned by
dev128.

## Identity, execution, and verification

- Result: [portfolio-reduction-v1.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/evaluation/portfolio-reduction-v1.json), SHA256
  `cb0aaa0b99344c813f563b543e9165715a654048883fc85b3bce2c1348f39b19`.
- Saved raw/scored run roots and their raw/run-manifest SHA256 values are
  provenance-bound under the result's `runs.{train,dev}.{Source,R,B,M,W}`
  metadata. The verifier reread all ten raw and scored JSONLs; their ordered
  row IDs exactly equal the 256/128 frozen input rows.
- Bank ID: `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
  Final checkpoint IDs: R
  `4efbec695067ec4f5a7c1de30add419046e0ac210067a27bd013aa9df71c033d`,
  B `67684f1a842c3c454b45c7efd0c2706c9a7248cb571496353025471117da5d9d`,
  M `4dcccaa341763841d008441bd22bd66ac6f90ba810c2ec38cfeab64b6e79b03b`,
  W `1a068bf17c237bda42c4f73c592c81331dfa001e3fc4228383f42b089b54a0da`;
  every candidate run manifest binds its arm to completed update 64.
- [Evaluation launch receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/launch-evaluation/):
  `2026-09-07T09:35:44Z` through `2026-09-07T11:23:36Z`; all eight panel exits
  and the reducer exit are zero; terminal marker is
  `COCO_EVALUATION_AND_REDUCTION_COMPLETED`.
- The fresh CPU-only verifier below hashes the result/bank/topology/input/raw/
  manifests, checks input-to-raw-to-scored row identity and checkpoint/split
  bindings, and recomputes all 42 stored gain/loss/net contrasts from
  matched-owner ID sets without rerunning detection or inference.

```bash
conda run -n ms python -c "$(cat <<'PY'
import hashlib, json
from pathlib import Path

root = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1')
result_path = root / 'evaluation/portfolio-reduction-v1.json'
r = json.loads(result_path.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert r['status'] == 'completed' and r['policy']['all_images_retained'] is True
assert sha(result_path) == 'cb0aaa0b99344c813f563b543e9165715a654048883fc85b3bce2c1348f39b19'
assert sha(r['bank']['manifest_path']) == r['bank']['manifest_sha256']
assert sha(r['source_topology_receipt']['path']) == r['source_topology_receipt']['sha256']
expected = {
    'R': '4efbec695067ec4f5a7c1de30add419046e0ac210067a27bd013aa9df71c033d',
    'B': '67684f1a842c3c454b45c7efd0c2706c9a7248cb571496353025471117da5d9d',
    'M': '4dcccaa341763841d008441bd22bd66ac6f90ba810c2ec38cfeab64b6e79b03b',
    'W': '1a068bf17c237bda42c4f73c592c81331dfa001e3fc4228383f42b089b54a0da',
}
contrast_count = 0
for split, n, owners in [('train', 256, 1955), ('dev', 128, 891)]:
    inp = Path(r['inputs'][split]['path'])
    assert sha(inp) == r['inputs'][split]['sha256']
    rows = [json.loads(line) for line in inp.open()]
    ids = [f"coco2017_train_{row['image_id']:012d}" for row in rows]
    assert len(rows) == len(set(ids)) == n
    assert sum(len(row['objects']) for row in rows) == owners
    s = r['splits'][split]
    assert (s['image_denominator'], s['annotated_owner_denominator']) == (n, owners)
    for arm in ('Source', 'R', 'B', 'M', 'W'):
        run = r['runs'][split][arm]
        run_dir = Path(run['run_dir'])
        assert sha(run_dir / 'gt_vs_pred.jsonl') == run['raw_sha256']
        assert sha(run_dir / 'run_manifest.json') == run['run_manifest_sha256']
        raw = [json.loads(line) for line in (run_dir / 'gt_vs_pred.jsonl').open()]
        scored = [json.loads(line) for line in (run_dir / 'gt_vs_pred_scored.jsonl').open()]
        assert [row['row_id'] for row in raw] == ids == [row['row_id'] for row in scored]
        manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        assert manifest['terminal_status'] == 'completed'
        if arm != 'Source':
            ci = manifest['model_identity']['coco_gt_correction']
            assert (ci['arm'], ci['completed_update'], ci['checkpoint_id']) == (arm, 64, expected[arm])
        monitor = s['arms'][arm]['monitors']
        valid = monitor['generated_row_count'] - monitor['invalid_prediction_count']
        for threshold in ('0.50', '0.60', '0.80'):
            item = s['arms'][arm][f'iou_{threshold}']
            matched = set(item['matched_owner_ids'])
            assert len(matched) == item['matched_owner_count']
            assert item['unmatched_prediction_count'] == valid - item['matched_owner_count']
    for threshold in ('0.50', '0.60', '0.80'):
        sets = {arm: set(s['arms'][arm][f'iou_{threshold}']['matched_owner_ids'])
                for arm in ('Source', 'R', 'B', 'M', 'W')}
        for arm in ('R', 'B', 'M', 'W'):
            for base in ('Source', 'R'):
                if arm == base:
                    continue
                item = s['paired'][f'iou_{threshold}'][f'{arm}_minus_{base}']
                gains, losses = sets[arm] - sets[base], sets[base] - sets[arm]
                assert (len(gains), len(losses), len(gains) - len(losses)) == (
                    item['gain_count'], item['loss_count'], item['net_count'])
                contrast_count += 1
assert (root / 'launch-evaluation/terminal-marker.txt').read_text().strip() == 'COCO_EVALUATION_AND_REDUCTION_COMPLETED'
assert all((root / f'launch-evaluation/{arm}-{split}.exit').read_text().strip() == '0'
           for arm in 'RBMW' for split in ('train', 'dev'))
assert (root / 'launch-evaluation/reducer.exit').read_text().strip() == '0'
print(f'VERIFIED result_sha256={sha(result_path)} train=256/1955 dev=128/891 '
      f'runs=10 candidate_panels=8 paired_contrasts={contrast_count} checkpoints=64 terminal=completed')
PY
)"
```

Evidence line:
  `VERIFIED result_sha256=cb0aaa0b99344c813f563b543e9165715a654048883fc85b3bce2c1348f39b19 train=256/1955 dev=128/891 runs=10 candidate_panels=8 paired_contrasts=42 checkpoints=64 terminal=completed`.

## Claim boundary and stop

This portfolio identifies relative behavior among R/B/M/W. Because it has one
fixed dose and no canonical-only arm, it cannot attribute absolute improvement
to adding corrections rather than canonical CE alone. It also does not prove
capacity limits, universal superiority of selected-only supervision, or robust
generalization; W used a different trainable surface under the same nominal LR,
not a matched effective functional step size. No combination, promotion,
extra round, sweep, or new experiment follows from this closeout.
