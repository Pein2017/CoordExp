# Final Results

This file closes out the `unmatched-proposal-verifier-ablation` change with the
final offline diagnosis.

It is intentionally evidence-first and follows the authority-first framing in:

- `proposal.md`
- `design.md`
- `specs/unmatched-proposal-verifier-study/spec.md`

The original question was not merely whether `counterfactual` beats
`commitment` on clean synthetic negatives.
The real question was:

- when a rollout proposal is unmatched to GT, is it mostly dead / duplicate /
  bad,
- or is it often a real visible object that rollout can see more easily than
  the dataset annotations express,
- and is the verifier strong enough to support later soft pseudo-label
  promotion?

## Evidence Stack Used

The final diagnosis uses all three evidence layers:

1. `Layer A`: clean GT positives vs GT-derived hard negatives
2. `Layer B`: real rollout proposal populations from authoritative temperatures
   `0.0 / 0.3 / 0.5 / 0.7`
3. `Layer C`: completed manual audit on a stratified 96-sample unmatched subset

Authoritative aggregate artifacts:

- `temp/stage2_parity_authoritative_summary.csv`
- `temp/stage2_parity_unmatched_topk_summary.csv`
- `temp/stage2_parity_rollout_shape_summary.csv`
- `temp/stage2_parity_rollout_image_patterns.txt`
- `temp/stage2_parity_scoring_failures.csv`

Manual audit artifacts:

- `output/analysis/unmatched-proposal-verifier-manual-audit-v1/manual_audit_recommended96.csv`
- `output/analysis/unmatched-proposal-verifier-manual-audit-v1/manual_audit_labels.jsonl`

Additional comparative artifacts for the later checkpoint extension:

- `temp/stage1_stage2a_authoritative_summary.csv`
- `temp/stage1_stage2a_report_summary.csv`
- `temp/stage1_stage2a_image_trajectories.csv`
- `temp/stage1_stage2a_high_cardinality_cases.json`
- `temp/stage1_stage2a_unmatched_topk_summary.csv`

## Layer A: Clean Verifier Benchmark

This layer is stable across temperatures because it does not depend on rollout
population quality.

For `ul-res_1024-ckpt_300_merged`:

- `commitment`: AUROC `0.5033`, AUPRC `0.3033`
- `counterfactual`: AUROC `0.6163`, AUPRC `0.4636`
- `combined_linear`: AUROC `0.5471`, AUPRC `0.3990`

For `ul-res_1024-v2-ckpt_300_merged`:

- `commitment`: AUROC `0.5019`, AUPRC `0.3015`
- `counterfactual`: AUROC `0.6244`, AUPRC `0.4695`
- `combined_linear`: AUROC `0.5504`, AUPRC `0.4023`

Conclusion:

- `counterfactual` is the strongest single proxy on the clean slice.
- `combined_linear` improves over `commitment`, but does not beat
  `counterfactual`.
- This is stable across both checkpoints.

## Layer B1: Rollout Collection Health

The authoritative `stage2_parity_vllm` rerun fixed the earlier collection
contract mismatch and produced valid rollout populations for all eight
`checkpoint x temperature` runs.

Main collection trend:

- proposal count drops sharply as temperature rises
- duplicate-like rate drops even faster
- unmatched count also drops with temperature

Average across checkpoints:

- `t=0.0`: `pred_count_total=1295.5`, `unmatched_count=996.0`,
  `duplicate_like_rate=0.5531`
- `t=0.3`: `pred_count_total=804.5`, `unmatched_count=550.0`,
  `duplicate_like_rate=0.3224`
- `t=0.5`: `pred_count_total=763.0`, `unmatched_count=500.0`,
  `duplicate_like_rate=0.1424`
- `t=0.7`: `pred_count_total=556.0`, `unmatched_count=304.0`,
  `duplicate_like_rate=0.0098`

Interpretation:

- the main bottleneck is indeed rollout behavior, not clean teacher-forced
  recognition
- temperature strongly changes proposal population quality
- low temperature produces many more proposals, but also many more duplicates
  and pathological oversized proposals

## Layer B2: Rollout Proposal Benchmark

This layer is where the story becomes more nuanced.

### Matched vs Unmatched Separation

Across the eight authoritative runs:

- `rollout AUROC`: best proxy was
  - `combined_linear` in `4/8`
  - `counterfactual` in `3/8`
  - `commitment` in `1/8`
- `rollout AUPRC`: best proxy was
  - `combined_linear` in `5/8`
  - `commitment` in `2/8`
  - `counterfactual` in `1/8`

This differs from the clean slice.

Interpretation:

- `counterfactual` remains the best single proxy for grounding signal
- but on real rollout proposals, `combined_linear` becomes the most practical
  reranker more often than either single proxy alone
- `commitment` and `counterfactual` are genuinely complementary rather than
  redundant

### Decode / Cardinality Pathology

Single-image trajectory inspection shows the strongest evidence for the
user’s original hypothesis:

- the bottleneck is more about `continue-vs-stop / repetition / cardinality`
  than geometry
- the same image can move between empty output, dense runaway output, and
  sparse partial output depending on temperature

Representative cases from `temp/stage2_parity_rollout_image_patterns.txt`:

- `images/val2017/000000315450.jpg`:
  `128 -> 0 -> 21 -> 1`
- `images/val2017/000000238410.jpg`:
  `0 -> 35 -> 1 -> 36`
- `images/val2017/000000568439.jpg`:
  `66 -> 7 -> 2 -> 1`
- `images/val2017/000000104612.jpg`:
  empty at `t=0.0`, then `7` broccoli proposals at `t=0.7`

This is not consistent with a pure geometry bottleneck.

### Proxy Failure Mode: Large-Box Bias

The most important rollout-side pathology is the bbox-mask
`counterfactual` large-box bias.

From `temp/stage2_parity_rollout_shape_summary.csv`:

- low-temperature unmatched proposals are often huge
- top-ranked unmatched proposals by `counterfactual` or `combined_linear`
  are frequently near full-frame boxes

Examples:

- base `t=0.0`, top-25 unmatched by `counterfactual`:
  mean area fraction `0.8432`
- base `t=0.0`, top-25 unmatched by `combined_linear`:
  mean area fraction `0.9269`
- v2 `t=0.0`, top-25 unmatched by `counterfactual`:
  mean area fraction `0.9129`

This means:

- masking a huge region can artificially inflate the counterfactual drop
- high `counterfactual` does not automatically mean tight visual grounding

## Layer C: Manual Audit

The completed stratified 96-sample audit provides the final trust anchor.

From `manual_audit_recommended96.csv`:

- total audited proposals: `96`
- every row has an audit note

Final label distribution:

- `real_visible_object`: `26`
- `wrong_location`: `46`
- `dead_or_hallucinated`: `18`
- `uncertain`: `6`
- `duplicate_like`: `0`

The absence of `duplicate_like` in this audited subset is expected because:

- the audited queue came from the cleaner `0.5 / 0.7` regimes
- and duplicate suppression had already been treated as a separate surface

### Manual Audit by Run

`ul-res_1024-ckpt_300_merged @ t=0.5`

- real: `10/24`
- wrong-location: `11/24`
- dead: `2/24`
- uncertain: `1/24`

`ul-res_1024-ckpt_300_merged @ t=0.7`

- real: `8/24`
- wrong-location: `9/24`
- dead: `5/24`
- uncertain: `2/24`

`ul-res_1024-v2-ckpt_300_merged @ t=0.5`

- real: `1/24`
- wrong-location: `15/24`
- dead: `6/24`
- uncertain: `2/24`

`ul-res_1024-v2-ckpt_300_merged @ t=0.7`

- real: `7/24`
- wrong-location: `11/24`
- dead: `5/24`
- uncertain: `1/24`

Interpretation:

- the base checkpoint is materially better than v2 on this pseudo-label-facing
  audit subset
- `t=0.7` improves v2 sharply relative to `t=0.5`
- even in the better conditions, `wrong_location` remains the dominant failure
  mode

### Manual Audit by Score Quantile

This is one of the most diagnostic findings.

- `q1_top`: `5 real / 23 wrong / 3 dead / 1 uncertain` across `32` audited rows
- `q2_upper_mid`: `15 real / 11 wrong / 3 dead / 2 uncertain` across `31` rows
- `q3_lower_mid`: `3 real / 7 wrong / 5 dead / 2 uncertain`
- `q4_tail`: `3 real / 5 wrong / 7 dead / 1 uncertain`

Interpretation:

- the very top score bucket is not the cleanest bucket
- instead, it is dominated by `wrong_location`
- this strongly confirms the large-box / group-box over-reward problem
- the middle score bucket is actually the best precision region in this audited
  sample

### Note Themes

Manual notes repeat a small number of themes very consistently.

For `wrong_location`, the notes repeatedly mention:

- “覆盖了整张图 / 全图 / 全景”
- “包含太多无关区域”
- “应该拆成多个框”
- “只包含局部 / 没有框全”

For `real_visible_object`, the notes repeatedly mention:

- visible but partially occluded objects
- local object parts that are still clearly semantically grounded
- GT category mismatch or GT underspecification

For `dead_or_hallucinated`, the notes repeatedly mention:

- semantic mismatch (`not that object`)
- box on the wrong region entirely
- degenerate local regions that happen to co-occur with the right class nearby

This note-level evidence strongly supports a mixed-distribution interpretation:

- unmatched proposals are not pure hallucinations
- but they are also not promotion-ready by default
- the largest mass is wrong-location / oversized / group-box proposals

## Additional Checkpoint Extension: Stage1 vs Stage2a

After the main UL-checkpoint study closed out, the same authoritative
`stage2_parity_vllm` experiment was repeated for two additional checkpoints:

- `stage1-coco80-ckpt-1832-merged`
- `stage2a-eff64-softctx2-merged`

The experiment contract was kept fixed:

- same subset size (`200`)
- same dataset (`coco 1024 val`)
- same prompt controls (`coco_80`, `desc_first`)
- same temperatures (`0.0 / 0.3 / 0.5 / 0.7`)
- same verifier/scoring pipeline

These runs did **not** include a fresh manual audit layer, so they provide a
useful rollout-personality comparison, but they do not change the main
pseudo-label-readiness conclusion.

### Clean-Slice Comparison

The two checkpoints are nearly indistinguishable on the clean verifier slice.

For `stage1-coco80-ckpt-1832-merged`:

- `commitment`: AUROC `0.5015`, AUPRC `0.3004`
- `counterfactual`: AUROC `0.6183`, AUPRC `0.4644`
- `combined_linear`: AUROC `0.5460`, AUPRC `0.3935`

For `stage2a-eff64-softctx2-merged`:

- `commitment`: AUROC `0.5019`, AUPRC `0.3019`
- `counterfactual`: AUROC `0.6231`, AUPRC `0.4734`
- `combined_linear`: AUROC `0.5529`, AUPRC `0.4056`

Interpretation:

- the main difference between these two checkpoints is not clean teacher-forced
  verifier ability
- the main difference is rollout behavior

### Rollout Collection Personality

`stage2a-eff64-softctx2-merged` is more aggressive at low temperature.

At `t=0.0`:

- `stage1`: `pred_total=879`, `matched=429`, `unmatched=450`,
  `duplicate_like_rate=0.1854`, `degenerate=5`
- `stage2a`: `pred_total=1087`, `matched=460`, `unmatched=626`,
  `duplicate_like_rate=0.2199`, `degenerate=72`

At `t=0.7`:

- `stage1`: `pred_total=590`, `matched=294`, `unmatched=291`,
  `duplicate_like_rate=0.0085`
- `stage2a`: `pred_total=614`, `matched=267`, `unmatched=334`,
  `duplicate_like_rate=0.0228`

Interpretation:

- both checkpoints share the same broad temperature trend
- but `stage2a` preserves a wider, noisier unmatched population
- `stage1` is more conservative and less degenerate

### Single-Image Trajectory Behavior

Object-level trajectory files again support the cardinality hypothesis.

From `temp/stage1_stage2a_image_trajectories.csv`:

- `stage1`: `mean_swing=6.32`, `median_swing=3`, `max_swing=117`
- `stage2a`: `mean_swing=6.72`, `median_swing=4`, `max_swing=125`

Representative mode-switch examples:

- `stage1`, `images/val2017/000000258911.jpg`: `128 -> 11 -> 14 -> 16`
- `stage1`, `images/val2017/000000039405.jpg`: `1 -> 27 -> 22 -> 0`
- `stage2a`, `images/val2017/000000425361.jpg`: `127 -> 9 -> 9 -> 2`
- `stage2a`, `images/val2017/000000381587.jpg`: `23 -> 0 -> 28 -> 27`

These trajectories again show that rollout mode-switching, not pure geometry,
dominates the behavior.

### High-Cardinality Cases

The most extreme proposal explosions are not all the same.

#### Clear collapse / duplication cases

- `stage1 @ t=0.0`, `images/val2017/000000258911.jpg`
  - `pred_count=128`
  - `person x128`
  - `dup_any_rate=0.977`
- `stage2a @ t=0.0`, `images/val2017/000000425361.jpg`
  - `pred_count=127`
  - `cup x119`
  - `dup_any_rate=0.976`
  - `near_full_rate=0.984`
- `stage1 @ t=0.3`, `images/val2017/000000046804.jpg`
  - `pred_count=26`
  - `sheep x26`
  - `dup_any_rate=0.885`

Interpretation:

- `stage1` low-temperature collapse is more often person/sheep-centric
- `stage2a` low-temperature collapse is more likely to involve household /
  clutter categories such as `cup`

#### High-count but partly healthy crowded-scene cases

- `stage1 @ t=0.0`, `images/val2017/000000238410.jpg`
  - `pred_count=34`
  - `person/chair/bottle/wine glass` mixture
  - `dup_any_rate=0.0`
- `stage2a @ t=0.3`, same image
  - `pred_count=32`
  - similar mixture
  - `dup_any_rate=0.031`
- `stage2a @ t=0.5`, `images/val2017/000000381587.jpg`
  - `pred_count=28`
  - `bowl/cup/person` mixture
  - `dup_any_rate=0.0`

Interpretation:

- not every high-cardinality image is bad
- some genuinely reflect crowded-scene proposal richness
- but the collapse cases remain numerous enough to matter

### Rollout Proxy Comparison

Overall matched-vs-unmatched separation is slightly better for `stage1` in the
more practical `0.5 / 0.7` regime.

Examples:

- `stage1 @ t=0.5`: best rollout AUROC `combined_linear = 0.6340`
- `stage2a @ t=0.5`: best rollout AUROC `combined_linear = 0.6089`
- `stage1 @ t=0.7`: best rollout AUROC `combined_linear = 0.6530`
- `stage2a @ t=0.7`: best rollout AUROC `counterfactual = 0.6404`

However, `stage2a` has an interesting top-k behavior:

- for several `t>=0.3` settings, `commitment` top-25 unmatched proposals have
  surprisingly strong weak-IoU quality
- e.g. `stage2a @ t=0.3`, top-25 commitment-ranked unmatched has
  `IoU>=0.3 = 0.72` with very low duplicate rate

Interpretation:

- `stage2a` is not simply “worse”
- it is wider and noisier at the population level
- but some of its semantic top-k slices are stronger than expected

### Missing Token-Level Trace

One limitation of this checkpoint extension is that all eight runs wrote empty
`pred_token_trace.jsonl` files.

This means:

- token-by-token generation trajectory is not available
- the usable trajectory evidence is object-level only
- all conclusions in this section are based on:
  - `gt_vs_pred.jsonl`
  - `proposal_table.jsonl`
  - collection/proxy summaries

So the comparison is still informative, but only at proposal-population scale.

### Comparative Conclusion

The additional checkpoints reinforce the main study logic:

- clean verifier ability is fairly stable across checkpoints
- rollout personality differs materially across checkpoints
- low-temperature behavior remains the main source of pathological variance

More specifically:

- `stage1-coco80-ckpt-1832-merged` is more conservative and rollout-stable
- `stage2a-eff64-softctx2-merged` is more aggressive, more degenerate at low
  temperature, and more semantically biased toward some clutter/household
  classes

This extension does not change the main final decision:

- rollout behavior varies substantially across checkpoints
- verifier-guided triage remains promising
- but promotion-readiness still requires stronger control of wrong-location and
  oversized-box failure modes

## Manual Audit Extension: Stage1 vs Stage2a

To raise the newer checkpoint comparison above pure automatic evidence, a small
follow-up audit was completed on a `48`-proposal queue built from:

- `stage1-coco80-ckpt-1832-merged @ t=0.5 / 0.7`
- `stage2a-eff64-softctx2-merged @ t=0.5 / 0.7`

The labeled audit file is:

- `output/analysis/manual-audit-stage1-stage2a-v1/manual_audit_recommended48.csv`

Every row in this audit has a free-form note, so the conclusions below use both
labels and note content.

### Label Distribution

Final label counts across the `48` audited proposals:

- `wrong_location`: `30`
- `real_visible_object`: `10`
- `dead_or_hallucinated`: `8`

No rows were labeled `duplicate_like` or `uncertain` in this smaller follow-up
set.

This immediately shows that the newer checkpoint extension does **not** weaken
the main conclusion from the first audit:

- `wrong_location` is still the dominant failure mode
- even after restricting to `0.5 / 0.7`

### By Checkpoint and Temperature

`stage1-coco80-ckpt-1832-merged @ t=0.5`

- real: `4/12`
- wrong-location: `6/12`
- dead: `2/12`

`stage1-coco80-ckpt-1832-merged @ t=0.7`

- real: `3/12`
- wrong-location: `7/12`
- dead: `2/12`

`stage2a-eff64-softctx2-merged @ t=0.5`

- real: `1/12`
- wrong-location: `10/12`
- dead: `1/12`

`stage2a-eff64-softctx2-merged @ t=0.7`

- real: `2/12`
- wrong-location: `7/12`
- dead: `3/12`

Interpretation:

- `stage1` is clearly better than `stage2a` on this pseudo-label-facing audit
  slice
- `stage2a @ t=0.5` is the weakest condition in this extension
- both checkpoints remain dominated by wrong-location proposals, but `stage1`
  contains a materially larger share of proposals that are at least recognizable
  as real visible objects

### By Score Quantile

The same pathology from the earlier 96-sample audit shows up again.

`q1_top`:

- real: `3/16`
- wrong-location: `12/16`
- dead: `1/16`

`q2_upper_mid`:

- real: `1/16`
- wrong-location: `11/16`
- dead: `4/16`

`q3_lower_mid`:

- real: `6/16`
- wrong-location: `7/16`
- dead: `3/16`

Interpretation:

- the highest-score bucket is again heavily polluted by wrong-location cases
- this is consistent with bbox-mask `counterfactual` over-rewarding broad or
  coarse proposals
- unlike a well-calibrated promotion score, the top bucket is *not* the most
  trustworthy bucket

### By Overlap Bucket

`0.1_to_0.3`:

- real: `2/12`
- wrong-location: `9/12`
- dead: `1/12`

`0.3_to_0.5`:

- real: `1/12`
- wrong-location: `10/12`
- dead: `1/12`

`ge_0.5`:

- real: `4/12`
- wrong-location: `8/12`

`lt_0.1`:

- real: `3/12`
- wrong-location: `3/12`
- dead: `6/12`

Interpretation:

- high weak-IoU is still not enough to certify promotion quality
- `ge_0.5` proposals are often still wrong-location rather than genuinely good
  single-instance boxes
- `lt_0.1` is much more likely to be truly dead or semantically off

### Notes-Based Failure Themes

The notes are highly consistent and sharpen the automatic diagnosis.

For `wrong_location`, the notes repeatedly mention:

- “整张图片 / 全局”
- “包含了太多无关信息”
- “只框了一小部分 / 只框了上半部分 / 只框了下半部分”
- “本来是对的类，但 bbox 太大、太宽、或没有把物体框全”

For `real_visible_object`, the notes often describe:

- real objects that are partially visible
- small but semantically clear objects
- region proposals that point to the correct entity but do not fully capture it

For `dead_or_hallucinated`, the notes repeatedly describe:

- extremely tiny or unreadable regions
- clear class mismatch
- boxes on background or irrelevant elongated regions

This notes-based evidence again supports the same practical interpretation:

- the current verifier is better at separating “obviously dead” from “something
  semantically present”
- but it is still not strong enough to separate “good single-instance proposal”
  from “coarse wrong-location proposal”

### Score Statistics by Human Label

Average scores on this 48-row audit:

- `real_visible_object`
  - mean commitment `-0.2324`
  - mean counterfactual `1.8066`
  - mean combined `1.5742`
  - mean nearest-GT IoU `0.4170`
- `wrong_location`
  - mean commitment `-0.3729`
  - mean counterfactual `2.9146`
  - mean combined `2.5417`
  - mean nearest-GT IoU `0.3817`
- `dead_or_hallucinated`
  - mean commitment `-0.3478`
  - mean counterfactual `0.5072`
  - mean combined `0.1593`
  - mean nearest-GT IoU `0.0590`

Interpretation:

- `dead_or_hallucinated` does score lower on average, which is good
- but `wrong_location` still scores *higher* than `real_visible_object` on the
  current proxy stack
- so the dominant unresolved issue is not “hallucination vs truth”
- it is “coarse wrong-location vs genuinely useful object proposal”

### Updated Comparative Conclusion

The manual audit extension makes the checkpoint comparison much sharper:

- `stage1-coco80-ckpt-1832-merged` is the more trustworthy rollout source of
  the two
- `stage2a-eff64-softctx2-merged` remains more aggressive and noisier
- neither checkpoint is close to promotion-ready under the current bbox-mask
  verifier

Most importantly, this extension confirms that the main unresolved problem is
still the same as in the UL-checkpoint audit:

- `wrong_location` dominates
- top-scoring unmatched proposals are often too coarse
- and the verifier still over-rewards broad regions

## Final Decision Answers

### 1. Which single proxy is strongest?

Most reliable answer:

- `counterfactual`

Reason:

- it is the strongest single proxy on the clean slice across both checkpoints
- it remains competitive on rollout scoring
- it best captures the “visual evidence matters” signal

But:

- it is not sufficient on its own for rollout promotion because of the
  large-box bias

### 2. Does `commitment + counterfactual` materially outperform either one alone?

Answer:

- on the clean slice: `no`
- on the rollout slice: `often yes`, especially in the more useful `0.5 / 0.7`
  regimes

So the practical answer is:

- `combined_linear` is the best rollout reranker more often than either single
  score
- but this is a rollout-only advantage, not a universally stronger proxy

### 3. Is the signal stable across checkpoints?

Answer:

- `clean-slice signal`: yes
- `rollout-ranking signal`: partially
- `promotion-facing quality`: mixed

Specifically:

- both checkpoints agree that `counterfactual > commitment` on the clean slice
- both checkpoints agree that temperature changes proposal population strongly
- but the two checkpoints do not have the same promotion-facing quality
- v2 at `t=0.5` is notably weaker than base at `t=0.5`

### 4. Is the proxy good enough to be used for soft pseudo-label promotion of unmatched proposals?

Final answer:

- **not yet**

Reason:

- there is real verifier signal
- there are real unmatched proposals in the pool
- but the dominant audited failure mode is still `wrong_location`
- and the top score bucket is actually overrun by wrong-location cases

So the correct final status is:

- `promising, but not yet promotion-ready`

### 5. What are the main observed failure modes?

Primary:

- oversized / near-full-frame boxes
- group boxes that merge multiple instances
- semantically plausible but poorly localized proposals
- top-ranked `counterfactual` rewards caused by masking too much area

Secondary:

- true semantic hallucinations
- abstention / empty outputs on some crowded scenes
- temperature-sensitive runaway repetition on certain images

## Reliable Final Conclusion

The final diagnosis is:

- the user’s original intuition was correct:
  the main bottleneck is decode / cardinality behavior rather than geometry
- `counterfactual` is the strongest single verifier proxy in principle
- `combined_linear` is the most practical reranker on real rollout proposals
- unmatched proposals do contain genuine real visible objects
- but the current bbox-mask verifier still over-rewards large or overly broad
  boxes
- therefore the study supports verifier-guided triage as a promising direction,
  but it does **not** yet validate soft pseudo-label promotion as reliable

This is the strongest conclusion justified by the complete evidence stack.
