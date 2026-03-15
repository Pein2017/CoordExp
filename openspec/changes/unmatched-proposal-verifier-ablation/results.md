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
