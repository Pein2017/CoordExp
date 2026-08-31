---
title: Image2299 matched sequence-optimizer ablation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-31-image2299-matched-sequence-optimizer-ablation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: cold_greedy_41_owner_success_with_frozen_recipe_negatives
updated: 2026-08-31
---

# Image2299 matched sequence-optimizer ablation

## Question, contrast, and stop rule

On one frozen r32 Image2299 instance, is ordinary-greedy 41-owner compilation
dependent on (a) the successful model-emitted sequence rather than an
owner-equivalent canonical sequence, and/or (b) protected-null QP rather than
matched real autograd CE?  The only four cells are `M/G × QP/CE`.  The unit
stops after its pre-registered cells, or immediately on an identity, static-G,
surface, solver, parser/matcher, cold-parity, or resource HOLD.  It must not
search checkpoints, add rows/rank/cap/updates, or touch the completed 46/46
mainline.

## Immutable inputs and route ledger

- r32 checkpoint: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step`
  (checkpoint readback SHA `83491665c2a918ae85b67cd57f5dc2ae64c84ffa8d1f1c01e131dba4ca3aeca7`).
- M receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-dyadic-norm-release-distillation/20260830T-image2299-dyadic-norm-release-distillation-v1/receipt.json`, SHA `9811bdb632c5c564607537fb80892c38b66a584b31a09da97fa887b6fc90a5e6`.
- Target library: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-26-image2299-full-root-detached-margin/target-root/target-root-v2/target.json`, SHA `22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988`.
- M route: SHA `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`, 370 tokens, final EOS `151645`; its ordered production-matcher ledger is
  `30,23,29,16,7,38,44,31,6,40,28,21,3,24,37,26,13,27,41,25,2,19,39,10,4,17,42,33,5,12,15,36,22,20,1,0,35,32,14,34,18`.
  These are 38 persons and ties `gt10/gt12/gt44`; excluded owners are exactly
  `gt11/gt43/gt45/gt8/gt9`.

`G` is mechanically constructed by iterating that ledger, appending each
owner's complete canonical nine-token `target.json.rows[*].token_ids`, then
one EOS.  The frozen construction has 370 tokens, pre-EOS row hash
`c7aa4f659ff3eca7d81ed6d86046d2f9193530245d186bce37419533f3476b25`,
and route SHA `e238e67122aa46b54cf9290d11093b07a7490746d6730d6f843f8d4ee61d677d`.
Before any teacher forcing or optimization, the new runner must call the
production parser/global matcher on G and fail closed unless it reports 38
persons, ties `gt10/12/44`, exactly 41 unique strict owners, zero hard debt,
and one final row-aligned EOS.  Token equality is scaffold-only; this
owner-equivalent static gate is the admission criterion.

## Shared preflight and parameter surface

With zero residual, teacher-force both admitted M and G using the identical
prompt/image/checkpoint.  A positive is a route position whose target is not
the strict logit top-1.  Freeze, before any solver or backward call:

`S_union = sorted(set(M positive target IDs) union set(G positive target IDs))`.

Record its ordered IDs, count, per-route positive positions/counts, and
`sha256(json.dumps(S_union, separators=(",", ":")).encode())`.  The historic
M nine rows/12 positives are not this union and may not be reused.  A failed
preflight, empty/changed union, or later row expansion is a HOLD.
Its JSON-serializable identity explicitly binds the live runner source SHA,
checkpoint/readback, prompt-token/image, start/frozen surfaces and sentinels,
ordinary/M/G routes, positive/S_union receipts, and per-route
basis/protected hashes; every cell recaptures and compares these identities.

For each sequence `c in {M,G}`, build its protected matrix and basis only
after `S_union` freezes.  Both cells for `c` use the same FP64 normalized output-only
surface `D_{c,s} = ||W_s|| X_{c,s} B_c^T`, with `X_c` the sole trainable variable and exactly
the rows in `S_union`.  Base model, DoRA, tied input/output embeddings, existing
embedding delta, aligner, vision tower, and every nonselected output row are
frozen.  Runtime applies the FP64 correction to output logits only.  Record
positive count/rank, protected count/hash, basis hash, S_union hash, variable
count, protected correction maximum, and residual norm.  Require correction
`<= 1e-10`, margin `0.01`, and normalized norm `<= 9/8`; no input-side path.

## Cell programs and acceptance

For `c+QP`, minimize `1/2 ||X_c||^2` subject, for each positive target, to
margin `>= .01` against every other movable S_union row and the strongest
fixed `V\\S_union` competitor.  Use HiGHS feasibility then SLSQP minimum-norm
primal; dual is diagnostic only.  Require exhaustive full-vocabulary runtime
recheck and the common null/norm checks.

For `c+CE`, use only true target-token autograd CE:
`SGD(X_c, lr in {2^-13,2^-12,2^-11,2^-10}, momentum=0, weight_decay=0)`;
zero gradients, `loss.backward()`, step, then project `X_c` to normalized norm
`<=9/8`.  Each of 41 object rows contributes its mean nine-token CE and EOS
is a separate singleton action: loss is their mean over 42 actions.  There is
no QP/margin loss, reward, KL, unlikelihood, or auxiliary term.  The panel is
exactly 50 updates per LR, with observations at `0,1,2,4,8,16,32,50`; no extra
LR, step, or optimizer.  Evaluate every listed milestone.  If any warm
milestone passes, select the earliest update, then the lower LR (numeric) on a
tie; only that payload enters cold verification.  If none passes, the CE cell
is a frozen-recipe negative.

Warm and fresh-subprocess cold acceptance both require ordinary greedy
(`do_sample=False`, no beam, forcing, prefix controller/table, or logits
processor), the same 41-owner gate above, all M parent owners retained, zero
duplicate/unmatched/unsupported/malformed/ambiguity/unknown/token-budget debt,
and natural EOS.  Cold additionally requires exact route, ledger, residual,
selected rows, basis/protected identities, and frozen surfaces.  Teacher loss,
QP slack, token equality, and warm-only output are not success.

## Boundaries and interpretation

Run one immutable preflight first, then eventual cells as four independent world-size-1 GPU processes; no DDP or
allreduce.  The preflight is the only shared admission/resource accounting and records its exact receipt SHA; each cell receives that SHA, recaptures every identity, and has at most two model loads, 16 GiB peak reserved, 3600 s,
and 200 MB output.  Each receipt records observed counts and measured usage
against those limits; QP has one solve/one warm decode, CE has four 50-update
traces.  Save payload only after warm success.  The completed canonical-five-
tie 46/46 artifacts are immutable external boundary, never an input payload.

This unit can support only the attachment's conditional comparisons on this
single augmented r32 Image2299 instance.  In particular, all-CE failure
rejects this four-LR CE recipe only; G failure does not show the model cannot
represent its owners.  It makes no base-r32, transfer, cross-image, training,
or 46/46 claim.

## Completed evidence

The authoritative execution is preflight v6 plus cell v5 receipts, finalized
as `cold_greedy_41_owner_success`: M+QP alone passed its warm and fresh-cold
ordinary-greedy 41-owner gate. M+CE and G+CE are frozen-recipe negatives; G+QP
is feasible but cap-gated before a warm candidate. See [results](results.md)
and [pilot receipt](pilot-receipt.md). Preflight v1--v5 and cell v1--v4 remain
technical-invalid provenance only, not scientific evidence.
