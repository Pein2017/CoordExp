---
title: Natural complete-output opportunity in the retained Source256 bank
description: Retained-data census separating achievable sampled outputs from greedy realization and learning claims.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-09-natural-candidate-opportunity
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-09
---

# Question and authority

The user authorizes continuing the question: distinguish no observed good
output from a good output not realized by greedy/learning. This first phase
uses retained evidence only: **zero GPU forwards, zero optimizer steps, no new
sampling, no protected confirmation read**. Extra sampling or training needs
a separately frozen question and cost authorization.

From original Source step2444 on train256, do its four retained natural
temperature1 samples per image contain better single complete outputs than
its same-checkpoint, same-prompt/media greedy output? A sample union is not a
complete output. The strongest alternatives are owner exchanges, category or
localization changes, repetition/cap avoidance, and model/prompt mismatch.

Finding a better sample establishes a finite conditional opportunity, not
learnability or a learning failure. No better sample among K4 does not prove
zero model probability, absence of legal good sequences, or impossibility.
This training population and retrospective bank provide no generalization claim.

# Fixed evidence

Bank plan:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json`.
Use its eight named acquisition shards, all256 images and exactly4 seeds per
image, plus its named train JSONL. No outcome-based image selection or bank
replacement. Sampling is native from prompt, T1/top-p1/top-k0/RP1/cap3084.
Stored sampled bodies omit observed terminal EOS; append it only where the
historical stop is im_end when accounting for complete token length. Never
invent EOS for a length stop or change the saved body/text/parser evidence.

Matched original greedy root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1/`.
Use its raw/scored rows, image plans, prompt/token traces and run manifest as
needed. Verify checkpoint/adapter/embedding identity, image/GT identity, prompt
IDs and media/grid correspondence before comparing. Sampling temperature versus
greedy is the intended difference; a hidden RP or prompt difference is not.
Record exact sources/hashes and current analysis code separately. Retired
historical configuration paths do not require resurrecting a worktree or
importing historical producer code; persisted effective evidence owns the run.

# Outcome accounting

Primary remains category-consistent global one-to-one IoU50 owner sets on each
complete output. Preserve all256 greedy and1024 sampled outputs, including
parser drops, invalid/empty predictions, caps and repeats. Carry IoU60/80,
TP/FP/FN/F1/recall, prediction count, category/GT-independent later pixel-IoU>.95
repeats, parser drops and complete generated length/stop. Unmatched predictions
are annotation-relative, not automatically hallucinations.

Report three nested descriptive witness levels, not training admission rules:
1. Net owner improvement: sample TP50 > greedy TP50, with gains/losses explicit.
2. Owner-preserving improvement: a strict superset of greedy owner IDs.
3. Strong joint witness: level2 and no increase in annotation-relative FP,
   strict repeats, parser drops or cap indicator. Report lengths, but do not
   forbid extra correct predictions or impose a new scalar utility.

For each level report the number of images with any such single sample and the
number of samples, plus complete metrics and gained/lost/retained IDs. Stronger
levels never filter or replace the primary population.

For descriptive best-by-owner selection, use highest TP50, then lower FP50,
fewer repeats, fewer drops, shorter complete token length, then lower seed.
Show sampled best-of4 separately from the optimistic oracle that may retain
greedy. Neither is a deployment policy. Keep full per-candidate scorecards so
different tradeoffs remain visible.

Report the per-image owner union as a separate unattained combination bound,
including cases whose union gains owners but no single sample improves TP.
Never use that union as evidence of one good achievable complete trajectory.

For gained/lost owner examples, report best baseline same-category and any-
category geometric IoU, selected matching and relevant boxes. These are
continuous explanatory diagnostics, not an automatic physical-entity novelty
or causal-classification rule. Matching ambiguity remains explicit.

# Implementation, verification and stop

Use the maintained dora_owner_learning package's row projection, shared native
parser and src.eval.assignment.global_matches; do not duplicate a matcher or
create a generic bank/runtime framework. Keep new analysis local to this package.
Root owns this unit, source acceptance and interpretation. An implementation
owner may write the scoped analysis module/tests and versioned output artifact.
Preserve all unrelated dirty research/base/skill changes.

The consumer must reject missing/duplicate cells, wrong checkpoint/prompt/media,
corrupt token/text/parser/count/budget identity, and wrong GT geometry. Recompute
raw metrics and reproduce baseline/source-plan reward counts. Use a real saved
CPU fixture plus counterexamples for union-only gain, improvement with owner
loss, false improvement via repeats, and score/identity corruption.

Stop after one validated full retained-bank reduction and interpretation. If
evidence is insufficient, return the exact gap and a bounded next proposal;
do not automatically sample, train, increase K, or reopen retired study arms.

# Closure

The retained-data phase is lead-accepted. All256 images /1024 samples passed
the full consumer and independent byte-identical replay. The [results](results.md)
identify54 images with a net-improving complete sample and32 with a strong
IoU50 joint witness. This establishes observed sampling opportunity, not
learning failure. A supplementary read-only join to the already executed
round1 plan describes signed learning-signal supply without changing the
primary estimand. The missing immediate post-round1 train greedy read remains
the next discriminator; no new GPU work, sampling or training is authorized
by this closure.

Subsequent user authorization led to the completed
[round1 greedy read](../2026-09-09-round1-greedy-realization/results.md), which
closes that missing outcome under a separate execution bound. The retained
CPU result and its original evidence boundary remain unchanged.
