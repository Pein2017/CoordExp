---
title: COCO owner-focus versus correction-weight ablation results
description: Fixed-recipe result and independently verified paired owner readout.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-07-coco-owner-focus-ablation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# COCO owner-focus ablation: verified result readout

Status: **lead-accepted bounded scientific readout; complete; no promotion**.
This readout closes the fixed [R/M/Rweak unit](unit.md) without another run.

## Decision

The frozen observed owner-focus advantage rule is **not met**. On the fresh
holdout512, M versus Rweak at IoU50 is `96 gains / 148 losses = -52`, with the
image-bootstrap 95% total-equivalent interval `[-116, 3]`. M is positive in
the two other IoU50 point contrasts (`+25` versus R and `+33` versus Source),
but both intervals also cross zero. M additionally has IoU80 net `-6` versus
Source and increases Source-relative strict duplicate candidates by `61` and
cap/length stops by `7`; these independently violate the frozen rule.

This is a **definitive failure of the fixed recipe's observed acceptance rule
on this panel**, not proof against the owner-focus mechanism or proof of a
negative population effect. The strongest alternative—M only weakens
correction—is consistent with Rweak's observed IoU50 advantage and lower
cap/drop/length burden (though Rweak has more duplicate candidates), but is not
established because the genuine M-Rweak interval includes zero. Rweak matches
supervised-token mass, not gradient norms. This
is one new seed, with no canonical-only arm; finite intervals crossing zero
are uncertain, not evidence of equivalence.

Rweak is not a promotion candidate either: its Source-relative coverage vector
is the best observed (`+85/+47/+6`), but holdout strict duplicate candidates
increase from 21 to 276. R and M have 384 and 82 respectively. Reducing
correction weight remains a viable explanation/alternative, not an accepted
stable solution. No arm is promoted from this ablation.

## Exact owner coverage

Cells are `matched annotated owners / fixed denominator`. All 256 train images
and all 512 holdout images are retained, including capped rows.

| Split | Arm | IoU50 | IoU60 | IoU80 |
|---|---|---:|---:|---:|
| train | Source | 1259 / 1955 | 1190 / 1955 | 908 / 1955 |
| train | R | 1321 / 1955 | 1239 / 1955 | 958 / 1955 |
| train | M | 1305 / 1955 | 1223 / 1955 | 945 / 1955 |
| train | Rweak | 1318 / 1955 | 1240 / 1955 | 948 / 1955 |
| holdout | Source | 2225 / 3759 | 2055 / 3759 | 1534 / 3759 |
| holdout | R | 2233 / 3759 | 2057 / 3759 | 1502 / 3759 |
| holdout | M | 2258 / 3759 | 2075 / 3759 | 1528 / 3759 |
| holdout | Rweak | 2310 / 3759 | 2102 / 3759 | 1540 / 3759 |

## Exact paired owner contrasts

Gains and losses are set differences over identical durable owner IDs, not
aggregate subtraction. Bootstrap intervals resample the paired per-image
difference for 10,000 draws with seed `20260908`; `mean 95%` is owners/image
and `total-eq 95%` is the corresponding owner-total scale. These intervals are
diagnostic and are reported without dichotomizing them as significance tests.

### Fresh holdout512

| IoU | Contrast | Gains | Losses | Net | mean 95% | total-eq 95% |
|---:|---|---:|---:|---:|---:|---:|
| 50 | M-Rweak | 96 | 148 | -52 | [-0.2265625, 0.005859375] | [-116, 3] |
| 50 | M-R | 117 | 92 | +25 | [-0.068359375, 0.18359375] | [-35, 94] |
| 50 | M-Source | 185 | 152 | +33 | [-0.072265625, 0.20703125] | [-37, 106] |
| 60 | M-Rweak | 100 | 127 | -27 | [-0.166015625, 0.048828125] | [-85, 25] |
| 60 | M-R | 113 | 95 | +18 | [-0.07421875, 0.162109375] | [-38, 83] |
| 60 | M-Source | 176 | 156 | +20 | [-0.08984375, 0.173828125] | [-46, 89] |
| 80 | M-Rweak | 81 | 93 | -12 | [-0.11328125, 0.05864257812499929] | [-58, 30.024999999999636] |
| 80 | M-R | 98 | 72 | +26 | [-0.029296875, 0.1328125] | [-15, 68] |
| 80 | M-Source | 142 | 148 | -6 | [-0.107421875, 0.083984375] | [-55, 43] |

### Train256 stability diagnostic

| IoU | Contrast | Gains | Losses | Net | mean 95% | total-eq 95% |
|---:|---|---:|---:|---:|---:|---:|
| 50 | M-Rweak | 42 | 55 | -13 | [-0.15625, 0.05078125] | [-40, 13] |
| 50 | M-R | 29 | 45 | -16 | [-0.1484375, 0.015625] | [-38, 4] |
| 50 | M-Source | 101 | 55 | +46 | [0.05859375, 0.29296875] | [15, 75] |
| 60 | M-Rweak | 39 | 56 | -17 | [-0.1640625, 0.0234375] | [-42, 6] |
| 60 | M-R | 36 | 52 | -16 | [-0.13671875, 0.0078125] | [-35, 2] |
| 60 | M-Source | 97 | 64 | +33 | [0.01953125, 0.23828125] | [5, 61] |
| 80 | M-Rweak | 48 | 51 | -3 | [-0.08984375, 0.06640625] | [-23, 17] |
| 80 | M-R | 42 | 55 | -13 | [-0.12890625, 0.01953125] | [-33, 5] |
| 80 | M-Source | 98 | 61 | +37 | [0.05078125, 0.23828125] | [13, 61] |

Train gains versus Source show that the optimization surface moved. Their
failure to produce the frozen holdout advantage is compatible with
train-specific correction and output-distribution change; train256 is not a
substitute for the decision-owning holdout.

## Output debt

Every value is `M - reference`. `dup` is the strict geometry-derived duplicate
candidate count, not a confirmed semantic duplicate; `cap` equals the number
of cap/nontermination images here. `mean tok` and `total tok` are generated
token deltas; maximum length delta is zero in every contrast (all maxima are
the fixed 3084-token cap).

| Split | Contrast | dup | invalid | dropped | cap | length-stop | mean tok | total tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| holdout | M-Rweak | -194 | -1 | +1982 | +10 | +10 | +59.880859375 | +30659 |
| holdout | M-R | -302 | 0 | -1304 | -5 | -5 | -26.26171875 | -13446 |
| holdout | M-Source | +61 | -1 | +1280 | +7 | +7 | +55.064453125 | +28193 |
| train | M-Rweak | +419 | 0 | +1297 | +5 | +5 | +62.34765625 | +15961 |
| train | M-R | +75 | 0 | +310 | +1 | +1 | +14.54296875 | +3723 |
| train | M-Source | +415 | 0 | +980 | +4 | +4 | +57.2578125 | +14658 |

Unmatched predictions remain annotation-relative under partial COCO labels and
are not called hallucinations. The holdout was outcome-blind and excluded the
documented local train/selection IDs, but Source had historical COCO-val
forward-evaluation exposure; this result makes no no-operational-exposure claim.

## Identity and execution

- Result: [owner-focus-reduction-v1.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/evaluation/owner-focus-reduction-v1.json), SHA256 `418bfeea25d07399917b644bca3361477373bb0d38a20f9da8cfc94c40f08491`.
- Inputs: [train256](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl) `05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`; [holdout512](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/inputs-v1/holdout512.jsonl) `62aff40429cfc10f0a640a6e86d298b560ab79d25ce0a776393757a12d15accd`; [holdout manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/inputs-v1/manifest.json) `c402610384a3589958f95bfb0a37f176b4d7ba2ba2fce6486079b22adeeebb25`.
- Bank: [manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json) SHA256 `4486f173a98e3ba253798f8aa868a0712056b0480b9a4ee8fc081dfa90412ed2`; bank ID `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
- Source step-2444 identity SHA256 `200c8fee10d20ee29b93a13c117613d27660686d3a07b9ebf202050484b58528` (adapter tensor `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`). Final update64 checkpoints: [R](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/R/checkpoint-000064/) `f85bf97eb7040ff9c45240aa8314093c5725509a13336383394333728bcded6f`; [M](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/M/checkpoint-000064/) `030791834471d93a01b7c244ee1f4fb2a2b6a9fbd67a4493fc4ee3d18afbfc9a`; [Rweak](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/Rweak/checkpoint-000064/) `6b1f5b32dc1a5b7503ad81b691855ffcda3571f6d5433bc09838de07a88bf964`.
- Frozen code: [holdout builder](/data/CoordExp/.worktrees/coco-gt-correction-portfolio/scripts/research/build_coco_owner_focus_holdout.py) `a35793314dcd6db93ef217a6444c028f42e8535f1c982c8c636987393cdb7d47`; [trainer](/data/CoordExp/.worktrees/coco-gt-correction-portfolio/scripts/research/train_coco_gt_correction.py) `f59a457c90046eba693eb7d1f5c91982bc131e3d28069be58f8913a1f1626f97`; [evaluator](/data/CoordExp/.worktrees/coco-gt-correction-portfolio/scripts/research/eval_coco_owner_focus.py) `693da524eec702b1a32c0a40ba9ac95d89df42488a9b263955d8091ebcdd14aa`; [reducer](/data/CoordExp/.worktrees/coco-gt-correction-portfolio/scripts/research/reduce_coco_owner_focus.py) `9ab06d765e3fbe93efbb626075e49a3790a79071b591af962debc12b01a73b48`.
- Training ran `13:54:04Z–14:37:53Z`; Source evaluation `14:25:52Z–15:49:32Z`; candidate evaluation `14:39:31Z–17:07:02Z`. Every evaluation used two ranks and per-device decode batch 4; all eight panel exits and the reducer exit were zero.

The reducer binds the eight run roots. For direct audit, these are the raw /
scored / run-manifest SHA256 values:

| Split | Arm | raw | scored | run manifest |
|---|---|---|---|---|
| holdout | Source | `06eb70ad6f1b3907469a7d58132a307e0fce13cd12202745a6a314f997fcd22a` | `fcb424be6a567fe1e80fd579c64ae12a45a34bbca7fa75e38b736fe4ef20ee36` | `bbe993b3b390d9c3226b97e7f14078573bb400045b6dfe20b1392818b8f608f6` |
| holdout | R | `a7ec31d09ce5cb6639efa65848cf37698dd62b02cab146f6c4f796f0623a1715` | `75a910b0afa5e46d8a8a24484aefb7ba6bd91456ae9e0522b08af4ea064c16b5` | `48ee62673c3c6522f6b831a4ca7eb27831ab47ad418c9c7c49f9ed02e7fa7eb9` |
| holdout | M | `4952881bf6fe1e954d74628d5e3951c8347f76d693dfa17e3cd786c1032b0630` | `4847d6065b41cf8f377ef1ef08bee8fd2a02869160a6a9c8f50e34912217b66c` | `6f97f2b1082be4caa19c7f2ba1f27a74581fcd8b780f63000653cba795ffef39` |
| holdout | Rweak | `c4f69751d2830684661f49d9a678c39dad82a9a368f372b64fb40c92e92e9316` | `6cb132d88e2f3c681a2589b01f0b40f853f172cc52ce0a715429c1f98fab4b10` | `32fd2299e6b7bd89578401ed1bd78c7108eec010de1a971c06cb4110009a8944` |
| train | Source | `9318f45f2c5ccb722f5db262a6c5ec943a4f6fcebd25e88e7e43385a97294601` | `aec35c0e5100e40c3474fd46ef81c4b915e8226cb8c56037eea2aa25b8e88b7e` | `6c9abecd93483a5ce46b7e24626838e1116e84730adf13564a8cdae7e683885d` |
| train | R | `5af77f5c93c71e644063002281ec6a240d5876ed93ae6fee46ffcd8c8ec44c3a` | `8196758ff4691a6b3d03c26aa1e2eefba604360db047e59e0b2fd30aace84510` | `225021cc04a7300cb51e0dccbc19f83d04e4efc382b8c982bf9e17395b794a6e` |
| train | M | `d05b48b718069a662c8b4dd22fbb36225084dcdc5546d9363dbe5247b973344d` | `0931a9e88675ab60d2e4b9f0f5d7a9064fc8596eab40a1de53615d30439bf540` | `bf61c9870bbc812dadb670a9527dcdcb6457f5d49b2db5eef5cf2a6e6e1be095` |
| train | Rweak | `8a95e246dd6bcd5f16bd0640f9bbb5b40f2e6f0b3c256cedf4e9a24ca80b23a6` | `0316c1a0dcba89dee03a8724fd81d2ee96b960bffc1c4bce92d0691df567aa4b` | `0ebd957b67cfce955c452e6992ade0aee5be7e5874b1ed6dbec262279df2efa6` |

## Reproducible CPU verifier

This check does not call the evaluator or reducer and writes nothing. It
hashes the result, frozen code, inputs, bank, manifests, raw and scored files;
checks all eight row sets and checkpoint bindings; independently implements a
cardinality-first maximum-IoU assignment; recomputes every stored owner-ID
gain/loss/net set; and replays the paired image-bootstrap association and exact
stored intervals.

```bash
conda run --no-capture-output -n ms python - <<'PY'
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import normalize_coco_category_name

ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1')
RESULT=ROOT/'evaluation/owner-focus-reduction-v1.json'
CODE={
 'scripts/research/build_coco_owner_focus_holdout.py':'a35793314dcd6db93ef217a6444c028f42e8535f1c982c8c636987393cdb7d47',
 'scripts/research/train_coco_gt_correction.py':'f59a457c90046eba693eb7d1f5c91982bc131e3d28069be58f8913a1f1626f97',
 'scripts/research/eval_coco_owner_focus.py':'693da524eec702b1a32c0a40ba9ac95d89df42488a9b263955d8091ebcdd14aa',
 'scripts/research/reduce_coco_owner_focus.py':'9ab06d765e3fbe93efbb626075e49a3790a79071b591af962debc12b01a73b48'}
CHECKPOINTS={'R':'f85bf97eb7040ff9c45240aa8314093c5725509a13336383394333728bcded6f','M':'030791834471d93a01b7c244ee1f4fb2a2b6a9fbd67a4493fc4ee3d18afbfc9a','Rweak':'6b1f5b32dc1a5b7503ad81b691855ffcda3571f6d5433bc09838de07a88bf964'}
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 xs=[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
 out={str(x['row_id']):x for x in xs}; assert len(out)==len(xs); return out
cat=lambda x: normalize_coco_category_name(x.get('description',''))
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
 z=max(0,x2-x1)*max(0,y2-y1)
 return z/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z) if z else 0.
def matched(row,t):
 w,h=row['image_width'],row['image_height']
 gt=[(cat(o),coord_bins_to_pixel_xyxy(o['bbox'],image_width=w,image_height=h,field='gt'),int(o['object_id'])) for o in row['gt']]
 pr=[(cat(o),tuple(map(float,o['bbox']))) for o in row['pred'] if cat(o) and len(o.get('bbox',()))==4 and o['bbox'][0]<o['bbox'][2] and o['bbox'][1]<o['bbox'][3]]
 n=len(gt)+len(pr); score=np.zeros((n,n),dtype=np.int64); score[:len(gt),:len(pr)]=-10**18
 for i,(gc,gb,_) in enumerate(gt):
  for j,(pc,pb) in enumerate(pr):
   q=iou(gb,pb)
   if gc==pc and q>=t: score[i,j]=10**15+round(q*10**9)
 rr,cc=linear_sum_assignment(score,maximize=True)
 return {gt[i][2] for i,j in zip(rr,cc) if i<len(gt) and j<len(pr) and score[i,j]>0}

assert sha(RESULT)=='418bfeea25d07399917b644bca3361477373bb0d38a20f9da8cfc94c40f08491'
r=json.loads(RESULT.read_text()); assert r['status']=='completed' and r['policy']['all_images_retained'] is True
for p,want in CODE.items(): assert sha(p)==want
for k in ('train','holdout'): assert sha(r['inputs'][k]['path'])==r['inputs'][k]['sha256']
assert sha(r['bank']['path'])==r['bank']['sha256'] and sha(r['holdout_manifest']['path'])==r['holdout_manifest']['sha256']
bank=json.loads(Path(r['bank']['path']).read_text()); source_id='200c8fee10d20ee29b93a13c117613d27660686d3a07b9ebf202050484b58528'
assert bank['bank_id']==r['bank']['bank_id'] and bank['source_identity']['sha256']==source_id
comparisons={'M_minus_Rweak':('M','Rweak'),'M_minus_R':('M','R'),'M_minus_Source':('M','Source')}
checks=0
for split,nimages,nowners in [('train',256,1955),('holdout',512,3759)]:
 by_arm={}
 for arm,run in r['runs'][split].items():
  d=Path(run['run_dir']); raw=d/'gt_vs_pred.jsonl'; scored=d/'gt_vs_pred_scored.jsonl'; manifest=d/'run_manifest.json'
  assert sha(raw)==run['raw_sha256']==run['detection_metrics']['evaluation_receipt']['artifacts']['gt_vs_pred.jsonl']['sha256']
  assert sha(scored)==run['detection_metrics']['evaluation_receipt']['artifacts']['gt_vs_pred_scored.jsonl']['sha256']
  assert sha(manifest)==run['run_manifest_sha256']
  a,b=rows(raw),rows(scored); assert len(a)==len(b)==nimages and set(a)==set(b)
  m=json.loads(manifest.read_text()); assert m['model_identity_fingerprint']==run['model_identity_fingerprint']
  if arm=='Source': assert 'coco_owner_focus' not in m['model_identity']
  else:
   ident=m['model_identity']['coco_owner_focus']
   assert ident['checkpoint_id']==CHECKPOINTS[arm] and ident['completed_update']==64 and ident['bank_id']==bank['bank_id'] and ident['source_identity_sha256']==source_id
  by_arm[arm]=b
 universe={int(o['object_id']) for row in by_arm['Source'].values() for o in row['gt']}
 assert len(universe)==nowners and r['splits'][split]['image_denominator']==nimages and r['splits'][split]['annotated_owner_denominator']==nowners
 for t in (.50,.60,.80):
  key=f'iou_{t:.2f}'; per={arm:{rid:matched(row,t) for rid,row in data.items()} for arm,data in by_arm.items()}; total={arm:set().union(*x.values()) for arm,x in per.items()}
  for arm in total: assert sorted(total[arm])==r['splits'][split]['arms'][arm][key]['matched_owner_ids']
  for name,(a,b) in comparisons.items():
   s=r['splits'][split]['paired'][key][name]; gains=total[a]-total[b]; losses=total[b]-total[a]
   assert sorted(gains)==s['gain_owner_ids'] and sorted(losses)==s['loss_owner_ids']
   assert (len(gains),len(losses),len(gains)-len(losses))==(s['gain_count'],s['loss_count'],s['net_count'])
   ids=sorted(per[a]); diff=np.array([len(per[a][x])-len(per[b][x]) for x in ids],dtype=float)
   boot=np.random.default_rng(20260908).choice(diff,size=(10000,len(ids)),replace=True).mean(axis=1); lo,hi=np.quantile(boot,[.025,.975],method='linear'); bs=s['image_level_paired_bootstrap']
   assert (bs['image_count'],bs['draws'],bs['seed'],bs['unit'])==(len(ids),10000,20260908,'matched annotated owners per image')
   np.testing.assert_allclose([diff.mean(),diff.sum(),lo,hi,lo*len(ids),hi*len(ids)],[bs['observed_mean_difference'],bs['observed_total_difference'],*bs['percentile_95_mean_interval'],*bs['percentile_95_total_equivalent_interval']],rtol=0,atol=1e-12)
   checks+=1
print(f'VERIFIER PASS panels=8 images=768 owners=5714 paired_contrasts={checks} bootstraps={checks}')
PY
```

Lead freshly replayed the embedded verifier verbatim:
`VERIFIER PASS panels=8 images=768 owners=5714 paired_contrasts=18 bootstraps=18`.

## Stop

Science stops here under the frozen unit: no sweep, extra seed, canonical-only
arm, dose extension, promotion, or continuation proposal is implied by this
negative result.
