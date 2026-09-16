---
title: Research frontier and agent entry
role: research-entry
authority: non_normative_research
updated: 2026-09-16
---
# Research: trustworthy natural set completion

## Objective and current task

The user's primary objective is to reduce physical false negatives on unlabeled or incompletely labeled images; benchmark mAP is secondary. Compile trustworthy explorer support and useful continuations into ordinary model parameters so original-image, empty-prefix natural greedy covers more real owners without exchanging misses for false instances, incumbent-owner loss, repetition, invalid geometry or abnormal stopping. A complete training teacher is one possible means, not the research endpoint. Sampling union, executable complete routes and greedy realization are distinct. An explicit internal owner ledger is neither assumed nor established.

This repository studies this research topic as a whole. `research/` is its knowledge entry, without a redundant topic wrapper. **Fast catch-up:** read this page, the [Source256 learning comparison](experiments/2026-09-16-source256-fixed-prefix-completion/unit.md) and its [state](experiments/2026-09-16-source256-fixed-prefix-completion/state.json), then the [direction workshop](experiments/2026-09-16-research-direction-workshop/results.md). Retrieve the [accepted 22-image result](experiments/2026-09-15-coco22-cumulative-expansion/results.md), [CE-normalization result](experiments/2026-09-15-coco227-ce-normalization/results.md) and [paired-start predecessor](experiments/2026-09-14-training-set-completion-curriculum/dual-start-results.md) when those factors matter. **Deep catch-up:** continue through [the story](story.md), the relevant question below, and its decisive sources. Definitions and historical aliases live in [the glossary](glossary.md).

## Exact continuation boundary

The [paired16 complete-output ranking repair result](experiments/2026-09-16-source256-output-ranking-repair/results.md) is technically accepted and closed. R improves train/dev FN to632/265 versus P638/284, but outside-reference coverage and output-debt gates fail; actual starting91 gains retain59 with32 lost and31 replacements. Neither arm is promoted. [State](experiments/2026-09-16-source256-output-ranking-repair/state.json) owns the stop boundary; no further run is authorized.

The [Source256 completion CE normalization result](experiments/2026-09-16-source256-completion-ce-normalization/results.md) is technically accepted and [closed](experiments/2026-09-16-source256-completion-ce-normalization/state.json). Normalization reduced original B's forgetting and output errors, but gains also shrank; canonical and Source-dev advancement gates remain unmet. The fixed dose is complete; no additional run, weak-bank or visual review is authorized.

The [Source256 fixed-history completion](experiments/2026-09-16-source256-fixed-prefix-completion/unit.md)
is now [closed](experiments/2026-09-16-source256-fixed-prefix-completion/state.json):
[no incremental value at the registered64-update dose](experiments/2026-09-16-source256-fixed-prefix-completion/results.md).
Both arms trained and all80 natural bs4 readback shards were scored. B64 covers
1321/1988 train owners versus A64's1360, and607/891 dev owners versus A64's611
and Source's620. Source-incumbent loss and output burdens are higher in B.
The final143/256 eligible images and27.93% correction fraction passed scarcity;
this is a negative matched learning result, not an A=A feasibility failure.
Further view_images stopped by user. Do not automatically extend dose, seeds,
refresh, cohort or objectives; the result owns the claim limits and recovery evidence.

The2026-09-16 [research-direction workshop](experiments/2026-09-16-research-direction-workshop/results.md)
and its [state](experiments/2026-09-16-research-direction-workshop/state.json)
restore this objective after the fixed-teacher fitting stage. Distinguish added
trusted owner supervision from trajectory/prefix learning and refresh. The
workshop is a bounded synthesis, not a new training launch. Review the nearest
successful baseline and counterexample before proposing another direction.

The completed training-first task used original GT plus independently admitted old/new physical owners and required finishing each cumulative stage before adding images. Early held-out validation was deferred for that task; this historical boundary does not set the next study's acceptance rules. Generalization is not established by the fit.

The completed fixed-teacher stage is [11-to-22-image cumulative expansion](experiments/2026-09-15-coco22-cumulative-expansion/unit.md), using the latest Sample-arm final256 adapter, full old11 replay, sample-equal CE, fixed256 full-cohort updates and all8 GPUs. The user removed hour caps and preauthorized a matched Source joint-fit only after a genuine main-arm failure. Extra verified real COCO80 owners are permitted after manual visual review and must be written into the corresponding annotation JSONL `unlabeled` field. Frozen task completion and complete-output review are separate verdicts; annotation F1<1 from a verified extra does not alone trigger failure. The19 historical class-unknown owners remain in the full ledger and do not block this versioned expansion. The main arm is now lead-accepted: all376 owners, FN0/F1=1 and clean complete outputs at saved128 and256, with old227 retained at every saved checkpoint. Source was not triggered; all workers ended. See the [accepted result](experiments/2026-09-15-coco22-cumulative-expansion/results.md); the [closed state](experiments/2026-09-15-coco22-cumulative-expansion/state.json) preserves that stage's boundary. The [compression handoff](experiments/2026-09-15-coco22-cumulative-expansion/handoff.md) preserves predecessor bindings.

The [accepted CE-normalization result](experiments/2026-09-15-coco227-ce-normalization/results.md) remains closed: both arms reached clean227/227 at saved16 and maintained it through256, with old218 retained and new9 learned. Its [closed state](experiments/2026-09-15-coco227-ce-normalization/state.json) is retained. The earlier [A/B result](experiments/2026-09-14-training-set-completion-curriculum/dual-start-results.md) and [closed state](experiments/2026-09-14-training-set-completion-curriculum/state.json) preserve the initialization comparison. Shared geometry integration is accepted; do not reopen it as a new objective.

The result owns the numbers, loss curve, per-image ledger and evidence handles. Its frozen training population differs from the later all-known supplemental population. Newly discovered evaluation owners are not retroactive training targets. The latest physical review inherits class-agnostic one-to-one IoU≥0.5 matches and visually adjudicates only unmatched valid rows; earlier all-row strict-extent reviews are not directly comparable without matched re-evaluation.

Unknown category and physical identity are separate axes. CE masking removes direct target loss, not the literal conditioning history or all downstream gradients. Removing unknown rows invalidates an old suffix's original conditioning certificate. Existing confirmed false-object debt may temporarily remain under this task but must not be positively reinforced; new confirmed errors must be repaired before growth. Repeats, malformed rows, invalid geometry and caps have no completion exemption. Intermediate training regression was allowed; final stage acceptance still requires the joint per-image outcome. The state's preserved protocol owns the exact conditions.

## Question map and strongest retained evidence

| Question | What is known, with the limit kept attached |
|---|---|
| [Capacity and readout](questions/capacity-and-readout.md) | Shared finite-panel fitting works through output readout and internal parameters; this is not arbitrary-image capacity or transfer. |
| [Visual designation and causal use](questions/visual-designation-and-causal-use.md) | Painted cues and exact internal replay can steer rows; held-out representation quality did not make the tested learned bridge safe or target-specific. |
| [Discovery and route value](questions/discovery-and-route-value.md) | Useful conditional and sampled routes exist; an owner union, a useful full route, and greedy realization are different objects. |
| [Greedy compilation](questions/greedy-compilation.md) | Conditional fit, natural entry and full completion separate; current residual errors do not uniquely identify their cause. |
| [Preservation and credit](questions/preservation-and-credit.md) | Better local losses and small average KL do not certify incumbent-owner preservation. |
| [History, repetition and stopping](questions/history-repetition-stopping.md) | Coordinate and prefix state affect future rows; neither selected patches nor immediate suppression establish a universal set ledger. |
| [Physical evaluation](questions/physical-evaluation.md) | Annotation-relative quality may rise while reviewed physical coverage falls; unmatched is not a physical truth label. |
| [Runtime and evidence](questions/runtime-and-evidence.md) | Numerical, packing, verifier and attribution defects can change the apparent answer; technical invalidity is not a scientific null. |

## Next reasoning step and retrieval

The 2026-09-16 user ruling registers [TIDE-aligned unmatched review](../docs/eval/UNMATCHED_REVIEW.md) project-wide and prefers Co-DETR without a default VLM judge. The [retained-output diagnostic, design and accepted user adjudication](experiments/2026-09-16-codetr-only-review-proxy/results.md) is [closed](experiments/2026-09-16-codetr-only-review-proxy/state.json): detector support helps triage but cannot automatically become GT. Full-image/context evidence plus residual lead/subagent review is the next proposed proxy; new calibration and teacher admission remain unproven.

The22-image cumulative stage is closed successfully. Further image-count expansion or mechanism work needs a separate bounded contract; no automatic larger-cohort or Source launch remains authorized by this unit. A proposal must name the precise unresolved behavior, nearest tested predecessor, strongest counterexample and cheapest distinguishing observation. [Alternatives](alternatives.md) preserves important unanswered branches, not a standing launch queue. [The experiment catalog](experiments/catalog.jsonl) indexes historical and current records, including earlier denoising, painted-cue and binding lineages; search it rather than loading it wholesale.

Read [CONVENTIONS.md](CONVENTIONS.md) before maintaining knowledge. Current user instructions and fresh registered Project/runtime observations govern actions. Old `running`, `active`, mandatory review ceremonies and compute grants are historical data. [Historical source recovery and retirement accounting](../docs/history/research-records/2026-09-15-root-collapse/README.md) provide original bytes and paths without restoring old categories or executable assumptions.
