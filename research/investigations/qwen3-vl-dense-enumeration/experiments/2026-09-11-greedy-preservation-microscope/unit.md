---
title: Why average reference KL did not preserve natural greedy owners
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-11-greedy-preservation-microscope
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: verified
updated: 2026-09-11
---

## Question and fixed population

Executed disposition: [results](results.md). All56 paired records and the
fresh root consumer are accepted; no further scoring is authorized here.

On the56 reference trajectories explicitly protected during positive32
training, did small mean teacher-forced KL coexist with greedy margin crossings
at the actual first natural divergence, followed by owner loss?

These same56 images lost33/gained6 IoU50 annotated owners at the accepted
natural endpoint (416->389 TP). Therefore reference coverage alone cannot
explain all preservation damage. The strongest alternative is a large/diffuse
distribution shift, not a few weak argmax decisions or later exposure to a
changed prefix. This is descriptive localization, not causal rescue yet.

Use unchanged Stable50 and the exact positive32 adapter, original literal
reference prompts/actions and admitted KL masks:56cases,6056action tokens,
6047selected positions, including the known360573 invalid-row exclusion.
Do not create new reference labels, masks, GT edits or owner assignments.
Bind actual A natural outputs and the retained Stable50 actions by image.

## One production-shaped scoring slice

Replay each original reference trajectory once per checkpoint, with gradients
disabled, through the already accepted native scorer route. Compute full-vocab
KL(Stable50||positive32) at admitted positions, per-image means/maxima, source
literal-token versus best-other margins under both models, and source-argmax
retention/flip counts. Store compact statistics, not full-vocabulary logits.

Find the actual first natural action-token divergence and report whether its
position was protected, the literal old/new tokens, token category, relevant
margins/KL, and whether the current replay argmax equals the observed A token
at that shared prefix. Preserve raw margins; report numerical near-ties
separately using fixed epsilon1e-3 rather than pretending signs near zero prove
a robust transition. Source literal tokens need not be silently assumed argmax
under a different parallel replay arithmetic: record exceptions explicitly.

Join this to existing per-image gained/lost/retained annotated-owner sets and
new repeat/drop burden. Describe concentration of KL and margin violations;
association with owner loss is not proof that restoring one token rescues it.
If the evidence points to weak crossings, a later single-token rescue or
margin-preservation training contrast needs its own fixed contract. No such
intervention is part of this read.

## Bounds, ownership and stop

Eight independent workers, seven cases each, two sequential model loads per
worker;112scoring/model/image forwards globally,16loads, no generation. Hold
only bounded detached reference probabilities needed for local KL; source
cache at most2GiB/rank, CUDA allocated/reserved at most24GiB, RSS at most24GiB,
and wall at most600seconds/rank. Do not keep both full models on GPU. No
distributed optimizer or collectives; full caches must not be published as
research payload. Emit token-position statistics and exact source identities.

Root grants runtime only after the bounded CPU preparation is accepted. All
model/frontend/scorer identities and masks are reused, not a new execution
framework. Fail closed on identity/mask/alignment mismatch; preserve failure,
no automatic retry. Stop after all56paired records and one root consumer check.
This lane is independent of the checkpoint-history inference cross; the root
assigns non-overlapping GPU time. No child agents or training authority.
