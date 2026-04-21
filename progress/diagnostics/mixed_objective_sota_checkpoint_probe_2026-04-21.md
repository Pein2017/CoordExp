---
title: Mixed-Objective SOTA Checkpoint Probe
date: 2026-04-21
status: complete
topics: [stage1, 2b, mixed-objective, adapter, eval, oracle-k, recall]
tags: [2b, diagnostics, mixed-objective, hard-ce, soft-ce, gate, oracle-k]
summary: Focused probe of the new 2B mixed-objective adapter checkpoint showing strong val200 detection, stable output surface, moderate Oracle-K recoverability, and a low-recall profile dominated by weak-visual misses rather than suppressed positives.
---

# Mixed-Objective SOTA Checkpoint Probe (2026-04-21)

This note records a focused follow-up probe on the new 2B mixed-objective adapter checkpoint:

- adapter:
  `output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

The goal was not to rerun the full earlier coord-family project, but to answer a narrower question:

- why does this checkpoint reach the remembered `eval200`-scale `~0.39` to `0.41` range?
- is it behaving like the older weak `hard_soft_ce_2b` family, or like a stronger and more stable successor?
- does its remaining recall gap look like “model saw it but did not say it”, or more like weak visual support / systematic miss?

This study was executed in a dedicated worktree branch:

- branch:
  `codex/mixed-objective-sota-probe`
- worktree:
  `/data/CoordExp/.worktrees/mixed-objective-sota-probe`

## Scope

The probe covered four layers.

1. Contract audit
2. Matched `val200` infer + proxy eval bundle
3. `val64` verifier / unmatched-proposal recall lane
4. `val64` Oracle-K + recall-mechanism probe

Primary copied artifacts for long-term retention:

- [contract_audit_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/contract_audit_summary.json)
- [val200_infer_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/val200_infer_summary.json)
- [val200_proxy_eval_bundle_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/val200_proxy_eval_bundle_summary.json)
- [val64_verifier_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/val64_verifier_summary.json)
- [val64_oracle_k_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/val64_oracle_k_summary.json)
- [val64_recall_probe_summary.json](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/artifacts/mixed_objective_sota_probe_2026-04-21/val64_recall_probe_summary.json)

Study provenance:

- [design spec](/data/CoordExp/.worktrees/mixed-objective-sota-probe/docs/superpowers/specs/2026-04-21-mixed-objective-sota-checkpoint-probe-design.md)
- [implementation plan](/data/CoordExp/.worktrees/mixed-objective-sota-probe/docs/superpowers/plans/2026-04-21-mixed-objective-sota-checkpoint-probe.md)

## Contract Read

The contract audit confirmed that this checkpoint is a standard adapter-runtime case rather than a merged model:

- checkpoint type:
  `adapter`
- runtime load pattern:
  `model_checkpoint + adapter_checkpoint`
- base model:
  `model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- infer mode:
  `coord`
- bbox format:
  `xyxy`
- runtime `pred_coord_mode`:
  `auto`

This matters because it means the checkpoint should be treated as a first-class modern adapter family, not as a legacy merged artifact or a special-case debug run.

## Val200 Detection Read

The most important result is that the checkpoint is genuinely strong on matched `val200`.

Main `coco_real` view:

- `bbox_AP = 0.4137`
- `bbox_AP50 = 0.5910`
- `bbox_AP75 = 0.4203`
- `bbox_AR100 = 0.5113`
- `f1@0.50 loc micro = 0.6993`
- `precision@0.50 = 0.7526`
- `recall@0.50 = 0.6530`

More conservative views remain strong:

- `coco_real_strict`:
  `bbox_AP = 0.4110`
- `coco_real_strict_plausible`:
  `bbox_AP = 0.3923`

Interpretation:

- the checkpoint is not a brittle overfit that only looks good under permissive filtering
- the `~0.39` remembered benchmark scale was real, and this run lands slightly higher
- the family should now be treated as a serious strong 2B baseline

## Output-Surface Stability

The `val64` verifier collection lane shows a notably healthy surface:

- `pred_count_total = 436`
- `matched_count = 341`
- `unmatched_count = 95`
- `invalid_rollout_count = 0`
- `parser_failure_counts = {}`

This is one of the most important qualitative differences versus the older weak `base / hard_soft` families from the earlier comparison:

- no obvious invalid-geometry collapse
- no parser-failure family dominating the run
- no evidence that the score comes from a fragile output contract

So the checkpoint’s strength looks real and operational, not like a lucky but unstable decoding surface.

## Verifier Read

The verifier lane shows a usable but not magic confidence structure.

GT vs hard negative:

- `commitment auroc = 0.5117`
- `counterfactual auroc = 0.5992`
- `combined_linear auroc = 0.5436`

Matched vs unmatched:

- `commitment auroc = 0.6906`
- `counterfactual auroc = 0.6221`
- `combined_linear auroc = 0.7445`

Interpretation:

- plain commitment alone is not the main story
- the family does separate matched vs unmatched proposals fairly well
- but the detector is still better understood as a structured grounding model than as a model with a single clean scalar “confidence” knob

## Oracle-K and Recall Mechanism

Oracle-K shows a meaningful but bounded recoverable gap.

Primary localization recovery:

- baseline localization FN:
  `223`
- recoverable localization FN:
  `60`
- systematic localization FN:
  `163`
- `oracle_k_recovery_rate = 0.2691`

This places the family in an interesting middle regime:

- better than the old weak `hard_soft_ce_2b` family, which had almost no recoverable space
- less decode-sensitive than the earlier `raw_text_xyxy_pure_ce` family, which showed a much larger recoverable block
- somewhat reminiscent of the stronger `center_parameterization` family, though not identical

The mechanism split is especially informative:

- `suppressed_fn_rate = 0.0`
- `competitive_fn_rate = 0.0807`
- `weak_visual_fn_rate = 0.9193`

So the remaining recall gap is mostly not:

- “the model confidently saw it but refused to emit it”

It is much more consistent with:

- weak visual support
- incomplete grounding support
- and a smaller but real competitive subset

This makes the family look more like a stable engineering checkpoint than a model with a huge amount of hidden latent recall waiting to be unlocked by sampling.

## Most Important Operational Finding

The biggest integration issue during the probe was not model behavior but evaluation plumbing:

- `proxy_eval_bundle` and `oracle_k` were treating `model_cache/all-MiniLM-L6-v2-local` as a HuggingFace repo id instead of a local path under worktree execution

This was fixed in:

- [proxy_eval_bundle.py](/data/CoordExp/.worktrees/mixed-objective-sota-probe/src/eval/proxy_eval_bundle.py)
- [oracle_k.py](/data/CoordExp/.worktrees/mixed-objective-sota-probe/src/eval/oracle_k.py)

with the repair committed as:

- `05e1fa8` `fix(eval): resolve local semantic model paths in worktrees`

This is worth recording because it was a real worktree-path bug, not a missing-model bug. The local sentence-embedding weights were present the whole time.

## Final Read

Five conclusions are safe.

1. This checkpoint is a genuinely strong 2B adapter baseline, not a noisy one-off.
2. Its `val200` performance is strong under both permissive and stricter proxy-eval views.
3. Its output surface is stable and does not resemble the invalid-geometry-heavy older weak `hard_soft` family.
4. It has a moderate Oracle-K recoverable block, but not a huge hidden reservoir of suppressed positives.
5. Its remaining misses are mostly weak-visual rather than suppressed-emission failures.

The practical recommendation is:

- keep this checkpoint in the active shortlist as a strong mixed-objective 2B baseline
- treat it as more “stable and deployment-friendly” than “mysteriously under-decoded”
- compare it downstream against `center_parameterization` and `raw_text_xyxy_pure_ce` primarily on basin shape, crowded-scene behavior, and instance competition rather than only raw AP

## Relationship To Earlier Notes

This note extends, but does not replace, the earlier coord-family comparison and synthesis work:

- [coord_family_basin_and_recall_comparison_2026-04-20.md](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/coord_family_basin_and_recall_comparison_2026-04-20.md)
- [raw_text_continuity_and_coord_family_synthesis_2026-04-20.md](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/diagnostics/raw_text_continuity_and_coord_family_synthesis_2026-04-20.md)

It also updates the practical meaning of the older benchmark note:

- [stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md](/data/CoordExp/.worktrees/mixed-objective-sota-probe/progress/benchmarks/stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md)

That older benchmark established the family as promising. This probe shows that the new adapter checkpoint is not only promising on AP, but also stable enough to deserve ongoing diagnostic attention.
