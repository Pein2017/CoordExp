---
title: Source256 canonical CE versus refreshed trajectory RLOO
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
status: complete
evidence_status: observed_fixed_panel
updated: 2026-09-06
---

# Source256 canonical CE versus refreshed trajectory RLOO

Question: from the SAMEoriginalSource, do four freshly sampled on-policy RLOO
updates change natural annotated-owner coverage differently from four canonical
CE updates under the same image population and optimizer hyperparameters?
Strongest alternative: any apparent difference comes from greater RLOO compute
or a changed reward/preservation policy; report unequal generation/replay/time
costs and do not claim equal-compute superiority.

User-accepted reward: per-image r=TP50/GTcount from global one-to-one category-
consistent owner matching. K=4, A_k=r_k-mean(other3). RLOO objective is the image
and K mean of -A times SUM of generated action token log probabilities. CE is
image mean of per-transcript MEAN token NLL on native canonical targets including
EOS. There is no extra unknown/EOS/length/duplicate term, token credit mask,
whitening, PPO ratio, KL/reference, or critic. Reward neutrality is not token-
gradient neutrality: ordinary trajectory advantage applies to all action tokens.

All256x4 cells participate regardless of malformed/dropped predictions, unknown
rows, duplicate extras, length stops or flat rewards. Valid predictions within
an imperfect trajectory still match normally. Append native im_end for replay
ONLYif actually observed; length-cap actions contain no fabricated EOS. Missing
cells, wrong prompt/media/current-policy identity or corrupt token provenance
HALTtechnically, never silently become zero reward or replacement samples.

Two independent full588-tensor FP32/SDPA chains, Source-started,4rounds each;
AdamW lr2.5e-6/betas.9,.999/eps1e-8/wd0/clip1, constantLR and persistent moments.
Eachround is one full256-image optimizer update. RLOO samples K4 from its own
immediately preceding adapter at T1/top-p1/RP1/max3084 with seeds derived from
2026090601..04 plusround; no stale bank reuse. Save adapter and receipt-bound
optimizer moments aftereachround. Cold dev128 everyround, train256 atterminal4;
reuse immutable Source natural artifacts. This is population/update/LR matched,
NOTtoken/FLOP/wall matched. Four rounds are the stop even if results are null.

Shared constants, authority and resource/acceptance boundary:
[successor portfolio](../2026-09-06-ce-controls-rloo-successor/unit.md).

The canonical inference config remains B2/T0/top-p1 as required by its native
frontend contract. The existing research sampler is an explicitly recorded
sampled counterfactual: one sequence per call, CLI T1/top-p1/RP1, and new opt-in
`--raw-softmax`. This uses a fresh generation config with top-k0 and no inherited
model defaults; it also retains actually emitted pad tokens before observed
EOS. Sampling metadata, not the canonical frontend's greedy settings, owns
the behavior policy. Old unmarked/top-k-inheriting banks are rejected. The
reward definition is unchanged; no historical result is reclassified here.

The eight-image update/save/persistent-optimizer/native-cold qualification is
lead-accepted in the portfolio's rloo-qualification-lead-acceptance-v1.json.
The full course completed16:54:40UTC on2026-09-06. All eight updates, four fresh
1024-cell banks, persistent optimizer states and ten cold reads are lead-accepted.
The [results](results.md) record terminal development RLOO618/589/451 versus
CE612/589/452 and Source614/585/451. This is a small primary fixed-panel gain,
not equal-compute superiority or uniform improvement across thresholds.
Four rounds remain the stop; no additional run or architecture promotion follows.
