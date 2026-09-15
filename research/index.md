---
title: Research frontier and agent entry
role: research-entry
authority: non_normative_research
updated: 2026-09-15
---
# Research: trustworthy natural set completion

## Objective and current task

Compile verified physical-object support and useful continuations into ordinary model parameters. From the original image and an empty assistant prefix, one natural greedy completion should cover the trusted owners without exchanging misses for incumbent-owner loss, repetition, invalid geometry or abnormal stopping. Sampling discovers support; it is not the deployed answer. An explicit internal owner ledger is neither assumed nor established.

This repository studies this research topic as a whole. `research/` is its knowledge entry, without a redundant topic wrapper. **Fast catch-up:** read this page, the [current state](experiments/2026-09-15-coco22-cumulative-expansion/state.json), then the [accepted CE-normalization result](experiments/2026-09-15-coco227-ce-normalization/results.md) and [paired-start predecessor](experiments/2026-09-14-training-set-completion-curriculum/dual-start-results.md). **Deep catch-up:** continue through [the story](story.md), the relevant question below, and its decisive sources. Definitions and historical aliases live in [the glossary](glossary.md).

## Exact continuation boundary

The current training-first task uses original GT plus independently admitted old/new physical owners. A cumulative stage must finish before adding images. Early held-out validation is deferred for this task: it must not become a veto inherited from an older study. Generalization remains a later obligation, not a claim established by the present fit.

The current authorized stage is [11-to-22-image cumulative expansion](experiments/2026-09-15-coco22-cumulative-expansion/unit.md), using the latest Sample-arm final256 adapter, full old11 replay, sample-equal CE, fixed256 full-cohort updates and all8 GPUs. The user removed hour caps and preauthorized a matched Source joint-fit only after a genuine main-arm failure. Extra verified real COCO80 owners are permitted after manual visual review and must be written into the corresponding annotation JSONL `unlabeled` field. Frozen task completion and complete-output review are separate verdicts; annotation F1<1 from a verified extra does not alone trigger failure. The19 historical class-unknown owners remain in the full ledger and do not block this versioned expansion. Cohort and runtime preparation have not started; use the [compression handoff](experiments/2026-09-15-coco22-cumulative-expansion/handoff.md).

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

Prepare the fixed22-image bank and reviewed annotation writeback, then test cumulative learnability and old-owner retention under full replay. Reopen initialization or other mechanisms only at the new contract's decision boundary. A proposal must name the precise unresolved behavior, nearest tested predecessor, strongest counterexample and cheapest distinguishing observation. [Alternatives](alternatives.md) preserves important unanswered branches, not a standing launch queue. [The experiment catalog](experiments/catalog.jsonl) indexes historical and current records, including earlier denoising, painted-cue and binding lineages; search it rather than loading it wholesale.

Read [CONVENTIONS.md](CONVENTIONS.md) before maintaining knowledge. Current user instructions and fresh registered Project/runtime observations govern actions. Old `running`, `active`, mandatory review ceremonies and compute grants are historical data. [Historical source recovery and retirement accounting](../docs/history/research-records/2026-09-15-root-collapse/README.md) provide original bytes and paths without restoring old categories or executable assumptions.
