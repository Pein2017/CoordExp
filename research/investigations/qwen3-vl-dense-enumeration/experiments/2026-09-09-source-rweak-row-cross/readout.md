# First-row crossing: limited conditional gains, checkpoint dependence

Scientific status: **lead-accepted bounded result; no training or mechanism
promotion**. The separate model-routing benchmark is complete; its acceptance
and cost boundaries are in `benchmark/readout.md`.

## Outcome

Crossing the first divergent Rweak row into Source exposes a modest selected-panel
improvement and one particularly useful complete-output witness. It does **not**
establish a transferable owner-state mechanism, a generally better first row,
or an identified training-gradient problem. Checkpoint-dependent continuation
remains after visible row history is fixed.

This is the fixed outcome-stratified32-image panel (16 any-loss /8 gain-only /
8 equal-owner-set token-changed),401 GT annotations. It is not an estimate of
population mechanism frequencies or independent generalization. In the original
full512, Source/Rweak had2225/2310 IoU50 owners out of3759 GT; the panel below
deliberately oversamples Source-owner losses and cannot reverse that full-set
finding. Twenty-five common prefixes are empty. All64 forced actions are rows;
no scientific EOS intervention was selected.

## Complete-output results

Cell order is recipient checkpoint, action donor:0=Source,1=Rweak.

| Cell | Meaning | TP50 /60 /80 | Valid predictions | FP50 | Micro F1@50 | Strict repeats / parser drops / caps |
|---|---|---|---:|---:|---:|---|
| 00 | Source native | 210 /176 /110 | 372 | 162 | .543338 | 4 /1 /0 |
| 01 | Source after Rweak row | 212 /176 /110 | 371 | 159 | .549223 | 3 /2 /0 |
| 10 | Rweak after Source row | 195 /163 /103 | 685 | 490 | .359116 | 312 /360 /2 |
| 11 | Rweak native | 200 /168 /104 | 621 | 421 | .391389 | 252 /91 /1 |

Strict repeats count each later prediction once when pixel IoU exceeds.95 with
an earlier prediction, independent of category or GT. Parser drops are not valid
predictions. Unmatched predictions are annotation-relative FP, not automatically
hallucinations. Capped outputs remain in every denominator and score.

- **01 versus00:**8 gained,6 lost,204 retained owners; net+2 and F1+0.005885.
  At IoU60/80, gains/losses are5/5 and3/3: no net stricter-IoU owner improvement.
- **01 versus11:**30 gained,18 lost,182 retained; this is a fixed-visible-history
  checkpoint contrast, not the effect of replacing Source's row.
- **10 versus00:**14 gained,29 lost,181 retained. Versus11:5 gained,10 lost,
  190 retained. Extra repeats, drops and caps prevent calling isolated gains a
  better complete outcome.

Macro recall/F1@50 for00/01/10/11 are respectively
.598699/.624087, .608558/.634254, .565800/.566150, .603212/.605650.
Complete owner IDs, all thresholds, per-case counts and tokens are preserved in
the lossless reduction, not replaced by these summaries.

Sensitivity only, **not a filtered primary result**: the Rweak-native capped
case566923 accounts for much of01's advantage over11. Leaving that one case
out leaves31 images/389 GT, a01-versus11 exchange of22 gained/18 lost owners,
TP201 versus197 and micro F1.538874 versus.522546. Their macro F1 becomes
.627418 versus.624464. At IoU60/80 the TP ordering reverses to165 versus166
and102 versus103. The remaining IoU50 advantage is small; a broad stricter-
geometry advantage is not supported. The official32-case scores above retain it.

## Strongest useful witness:211674

For `coco2017_val_000000211674` (16 GT), cell00/01/10/11 has:
- TP50:6/8/6/5; TP60:3/5/3/2; TP80:1/3/1/1.
- Valid predictions:15/15/15/14; F1@50:.387097/.516129/.387097/.333333.
- Natural EOS in all cells and no strict repeats;01 has one parser drop.

Against Source,01 gains GT166500,1763804,2000560 and loses1730992. This is
**not owner-set dominance**. At IoU60/80 it gains two without losses. Neither
forced row/prefix has any compatible GT edge at any reported threshold; these
changes therefore survive the direct-row and action-induced matching explanation.
At the same prediction count,01 produces a better complete annotated outcome.

One recovered bus owner demonstrably includes category correction, not newly
discovered localization: Source already has a box with pixel IoU.89613 against
GT166500 but labels it `person`;01 labels a nearby box `bus` at IoU.98632.
This does not establish an owner-memory representation. Geometry,
category and annotation-relative ownership must not be conflated with latent
object-state discovery.

The two other IoU50 gains also have same-category Source candidates already:
best geometric IoU for1763804 rises.43216→.57447, and for2000560
.44970→.59310. The lost1730992 moves.54952→.47582. Thus all three IoU50 gains
in this witness are explainable by category correction or threshold-crossing
localization changes; the count increase is **not proof of newly enumerated
physical entities**. The stricter-threshold gains involve their own owner sets.

Another useful but different witness,369370, gives01 a strict Source owner
superset (+313304, no loss) at the same six predictions, F1.769231→.923077;
the new owner is directly action-compatible, and its gain disappears at IoU80.
It is not a clean suffix-recovery example.

## What the four cells identify

For fixed image/owner, let `z_ma` indicate complete-output owner membership.
Observed controlled contrasts are the row effects `z01-z00`, `z11-z10`, the
fixed-visible-history checkpoint effects `z10-z00`, `z11-z01`, and interaction
`z11-z10-z01+z00`. These are finite deterministic interventions on the frozen
system. They do not isolate a hidden-state mediator or independent training causes.

Among29 native Source-owner losses,24 have pattern1100 (present under Source
with either row, absent under Rweak with either row);2 have1010 (follow action0),
1 has1000,2 have1110. Among19 native Rweak gains,13 have0011 (follow Rweak),
4 have0101 (follow action1),2 have0001. Bits are00/01/10/11. These are selected
owner-pattern counts, **not percentages of loss caused by independent mechanisms**.

Three conclusion-changing safeguards:
1. With outcomes1110, assigning the native loss along00→01→11 attributes it
   entirely to checkpoint change; along00→10→11 it attributes it to row change.
   Both are exact decompositions. There is no unique causal-share percentage
   without an extra allocation convention. In211674, the row's TP effect is+2
   under Source but−1 under Rweak, with count interaction−3.
2. Direct B→C action replacement with identical empty suffixes already produces
   patterns1010/0101. Moreover, global matching can change a suffix owner's ID
   without changing the suffix prediction: adding a B-only action can reassign
   a fixed B/C-compatible suffix prediction from B to C. The reducer's
   `pure_tail` field is incidence-filtered suffix realization, not isolated
   causal suffix change. Primary gains/losses are never removed.211674's empty
   action-incidence graph avoids this particular ambiguity.
3. Four greedy trajectories do not determine useful off-greedy continuations,
   internal representations or gradients. An explicit owner-state model and a
   checkpoint-specific history lookup can realize the same four outputs. Endpoint
   logit margins `d_c(w)=1+c*w*(w-1)` agree at0/1 while their derivative at0 has
   arbitrary sign. A gradient-interference claim is not identified by this panel.

## Next decision and stop

The211674 witness justifies posing a **complete natural-outcome learning question**,
not launching an objective or copying a convenient crossed trajectory. A possible
future contrast is complete-outcome preference/credit versus matched-compute
trajectory imitation on training candidates, with every lost owner and the
prediction/F1/repeat/drop/cap burden kept visible. Any chosen TP-versus-burden
tradeoff remains user-owned. Forced-history likelihood is not its acceptance
metric; fresh natural decoding would be required under separate authorization.

The cheapest current alternative explanations are category/localization changes,
checkpoint-specific trajectory compatibility/repetition, and annotation/global-
assignment changes. Existing boxes, stricter IoUs and the controlled cell contrasts
already discriminate some of them. No new GPU arm, training or architecture is
needed to close this panel. **Stop scientific execution here.**

## Reproduction and provenance

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/`.
- Frozen manifest:`data-v1/manifest.json`, SHA256
  `af3fd69e05bea3c148c293e528a2ea738ac49e45631bb0b71a94d7eb56d3fcd8`.
- Full reduction:`reduction-v1/reduction.json`, SHA256
  `8ff4e5b2a2729a36e51a4f44ee3368ed1d38e7f8ac578a997d6dc8ee8fa444d9`.
- Compact projection:`science-evidence-v1.json`, SHA256
  `2eb4c6c6ab5789aa8119f52e269082f05cc9266664f211b710d8ccc36e2021c1`.

`technical-acceptance.md`, `run_panel.py`, the immutable formal launch/completion
receipts, and `reduction/README.md` provide exact commands and consumer evidence.
Both independent engineering implementations passed actual qualification. The
single formal panel used only one implementation; there was no duplicate formal
experiment. All generation used native FP32/SDPA, RP1.0 and a3084-token whole
trajectory cap including the common prefix and forced action, excluding the input
prompt/image. At most0.6414602 allocated GPU-hours including both qualifications;
all GPUs released, zero optimizer steps. Model-routing PK is recorded separately
in `benchmark/acceptance.md` and its final accounting/readout.
