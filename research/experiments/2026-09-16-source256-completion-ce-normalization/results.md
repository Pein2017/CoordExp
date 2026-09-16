# B-normalized: accepted execution, no promotion

The 64-update successor completed. CPU verification recomputed saved readback scores and all paired comparisons; manifest, producer and checkpoint hashes were checked. This is a technically accepted scientific result, not a runtime failure.

## Natural greedy endpoint

| Endpoint | Train FN | Train gained/lost vs Source | Dev FN | Dev gained/lost vs Source |
|---|---:|---:|---:|---:|
| Source0 | 688 | — | 271 | — |
| A64 | 628 | 95/35 | 280 | 17/26 |
| B64 | 667 | 96/75 | 284 | 24/37 |
| Bnormalized64 | 652 | 91/55 | 276 | 20/25 |

G/L counts compare matched owner identities with Source; train has 1,988 owners on 256 images, dev 891 on 128 images. Inputs are processed rescale_32_1024_bbox_len12000_xy_sorted, not raw COCO.

## Output errors

| Endpoint | Train malformed / repeat / cap | Dev malformed / repeat / cap |
|---|---:|---:|
| Source0 | 794 / 467 / 4 | 58 / 111 / 0 |
| A64 | 152 / 432 / 2 | 18 / 96 / 0 |
| B64 | 1445 / 1153 / 8 | 295 / 228 / 1 |
| Bnormalized64 | 660 / 303 / 3 | 4 / 140 / 0 |

Invalid geometry is zero. Repeat is the frozen geometry-overlap proxy, not confirmed physical duplication. Annotation-unmatched predictions remain unknown, not FP. EOS debt equals cap debt in these endpoints.

## Interpretation and stop

Normalization reduces original B train loss 75→55 and output errors, but gains also fall 96→91. Of the original 96 train gains, 71 remain, 25 disappear, and 20 different gains enter. Dev retains 11 of B’s 24 gains, loses 13, and adds 9 different gains. This meets the predeclared weaker-learning pattern, not selective preservation with gains intact. It does not prove uniform gradient weakening or isolate EOS/prefix causality: completion CE strength and its ratio to geometry both changed.

No promotion: train FN652 is worse than canonical A628; train losses55 exceed A35; dev FN276 remains worse than Source271. Dev improves over A280 and B284, and dev losses25 meet A26, but that does not close the missing recall gate. Output debt also remains mixed versus A.

Step16 was diagnostic: train FN670, G62/L44; dev FN271, G16/L16. It does not replace the frozen step64 primary endpoint. Category-consistent IoU50/60/80 counts, per-image identities and all diagnostics remain in the bound machine-readable result and receipt.

Stop at the authorized fixed dose. No extra seed, updates, refresh, visual census or weak-bank experiment. Original negative results are preserved.

## Evidence

- [Lead acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-completion-ce-normalization/lead/final-acceptance-v1.json) — config/manifest, checkpoint files, producer verification, metrics and hashes.
- [Result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-completion-ce-normalization/runtime/main-normalized-v1/evaluation/result.json) — SHA256 `d4674776e5ae509923462148028f428dda3b28c12af61789c82919827fb2cfa4`.
- [Training terminal](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-completion-ce-normalization/runtime/main-normalized-v1/B-normalized/training/terminal.json) — 64 updates, fresh optimizer, 2,048 model calls, 4,096 logical forwards.
- New bs4 natural readback: 768 image requests across 32 shards at steps16/64. Existing Source/A/B controls reused with exact identity checks; no control rerun.
- One seed and historically used dev limit generalization; no claim about all rollout learning.
