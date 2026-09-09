---
title: CE Completes the Fixed N2 Fit While Squared-Margin Adam Remains Partial
description: One fresh CE arm reaches the common gate at step 66 and exact cold natural coverage 65/65, against the sealed 300-step partial margin control.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-05-dora-ce-margin-n2-ablation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# Outcome

**Scientific verdict:** `CE_COMPLETE_MARGIN_INCOMPLETE_FIXED_PROFILE`.
Cross-entropy (CE) with Adam completes the exact fixed N2 fit at step 66. The
sealed squared-margin Adam control did not complete it within 300 steps.
This fixed recipe does not support replacing CE with the tested squared-margin
loss. It does not rule out every margin objective or a differently tuned recipe.

**Mechanics:** accepted. One authorized CE training invocation and one new-process
cold verification exited successfully; all 588 unmerged adapter tensors match
exactly after cold loading. Source restoration and its original adapter SHA-256
were independently reverified. No new QP, margin-control, or larger-panel run.

## Matched scope and measured result

The [unit](unit.md) and [frozen packet (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-dora-ce-margin-n2-ablation/launch-v1.json) own the exact Source,
images 6040/16228, 65 owners, 592 full canonical token positions including EOS,
teacher-forced prefixes, FP32 runtime, 196 magnitude vectors / 573,440 scalars,
AdamW lr=0.003 and remaining optimizer settings, per-step checks and budgets.
The sole training intervention is native full-vocabulary CE summed over tokens
and divided by 592, rather than the squared worst-competitor margin deficit.
Both backward passes accumulate before a single shared Adam step.

| Quantity | Fresh CE + Adam | Sealed squared-margin + Adam |
| --- | ---: | ---: |
| Updates | 66; common gate reached | 300; step budget reached |
| Common squared-margin metric | 0 | 0.0002465234 |
| Worst target margin | 0.0227622986 | -0.1335630417 |
| Positions below 0.00998 | 0 / 592 | 213 / 592 |
| Cold natural owners IoU50 / 60 / 80 | 65 / 65 / 65 | 37 / 28 / 21 |
| Confirmed duplicate + malformed + cap debt | 0 | 4 |
| Valid unmatched, unknown | 0 | 33 |
| Training process seconds | 208.080 | 897.102 |
| Cold process seconds | 61.700 | 67.420 |
| Combined process seconds | 269.780 | 964.522 |
| Magnitude distance from Source | 40.02207 | 23.19594 |

The CE arm's receipt `initial/final/best.loss` fields deliberately remain the
**common squared-margin metric**, not CE values. Its separate training-loss
receipt identifies `ce`, denominator 592. Both Source scans are exactly equal:
loss 0.5347199440, minimum margin -6.6791734695, and 262 deficient positions.

At CE step 65, one deficient position remained (minimum margin -0.0204830170);
step 66 removed it. Cold minimum margins are 0.2776336670 on image 6040 and
0.0227622986 on image 16228. Native greedy exactly reproduces both canonical
routes, covers 15 and 50 owners respectively at every IoU, emits EOS naturally,
and has no duplicates, malformed spans, cap, or unmatched predictions.

Training uses 132 image forward/backwards and 134 margin forwards. Peak reserved
memory is 32,189,186,048 bytes. The new measured train+cold work totals about
4.50 single-GPU process minutes, excluding Conda/shell overhead. The control is
reused, not concurrently randomized: this is not a hardware-normalized speedup
benchmark. Its time-to-success is undefined because it never reached the gate.
The unequal-success norm values do not establish a low-norm margin advantage.

## Interpretation and stop

- **Observed:** CE reaches the complete margin and natural-generation gates on
  the same parameter surface; this squared-margin Adam recipe does not.
- **Supported:** the current margin objective provides no observed optimization
  advantage over CE at the frozen Adam schedule. Optimizing CE can also drive
  the entire squared-margin metric to zero.
- **Ruled out here:** inability of this magnitude surface to fit N2, or an
  unavoidable teacher-forcing-to-greedy mismatch on these exact complete routes.
- **Unresolved:** the causal contribution of worst-competitor selection,
  threshold saturation, squared-deficit gradient strength, or loss-specific
  optimizer conditioning. No such mechanism is established by this one contrast.
- **Not claimed:** general loss dominance, best-tuned performance, seed
  repeatability, self-rollout learning, held-out dense recall, or production use.

The next decision is a user discussion about the intended bottleneck, not an
automatic objective sweep or self-rollout experiment. This unit is closed.

## Evidence and reproduction

Authoritative aggregate:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-ce-margin-n2-ablation/run-v1/comparison.json`

SHA-256: `5f5a088b9508aa9729d2223b38c944c20bac2ead03f0ed229b97b43a9ee5eb0d`.
It binds both new terminal receipts and both immutable control receipts.
New train SHA-256:
`0d6db0d6b1aa9df50fd290eac35d8a31fe681530aa6c06ab9419dbcd4f90d33c`;
new cold SHA-256:
`96be8b2f74da3dc9ab14030c974d5a178234870f358b56086cc71805c8040c3e`.

Recompute with a fresh destination (overwrite is refused):

```bash
conda run -n ms python scripts/research/analyze_dora_ce_margin_n2.py \
  --ce-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-ce-margin-n2-ablation/run-v1 \
  --output /tmp/dora-ce-margin-n2-replay.json
```

The lead's replay was byte-identical. Actual-receipt sensitivity checks reject
a non-CE treatment label and a claimed complete fit with one natural owner
removed. Fresh focused tests: 22 passed. CE value and gradients match native
CE across two-image accumulation; doubling its normalization makes the check
fail. A pre-launch `verify` CLI regression from the new loss metadata was
reproduced, corrected, and checked before any GPU invocation; it consumed no
scientific attempt. Default margin semantics, illegal QP+CE rejection, and
the inherited finite/save/cold contracts remain covered.
