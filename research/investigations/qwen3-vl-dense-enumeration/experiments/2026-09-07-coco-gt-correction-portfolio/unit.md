---
title: Existing-COCO GT Correction Portfolio
description: Fixed-dose prefix-policy, suffix-supervision and update-surface pilot on existing COCO.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-07-coco-gt-correction-portfolio
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# Existing-COCO GT Correction Portfolio

## Status and authority

- Scientific status: **complete; bounded pilot evidence verified**.
  [Results and reproducible readout](results.md) own the outcome.
- Implementation status: **lead-accepted mechanics and final cold evaluation**;
  this does not promote a model, architecture or deployment.
- Technical owner: [preserved OpenSpec proposal (archive ref `archive/research-restructure-20260909/coco-gt-correction-portfolio` at `c00043abc`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/openspec/changes/add-coco-gt-correction-portfolio/proposal.md)
  and [preserved implementation tasks (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/openspec/changes/add-coco-gt-correction-portfolio/tasks.md).
- Scientific owner: this unit. No model/architecture/publication promotion.
- User-accepted on 2026-09-07: existing COCO only; four independent fixed-dose
  arms followed by natural evaluation and STOP; eight GPUs available.
- User authorized apply and autonomous execution on 2026-09-07: "确认,并自主推进".
  The reviewed recipe below is now the fixed apply contract. The user also asks
  for economical agent allocation and minimal Astra lead-token expenditure.
- Excluded: COCO-LVIS, GT enrichment/completeness research, compulsory prior
  canonical-SFT-versus-self-prefix-SFT comparison, tuning, extra rounds,
  refreshed rollout banks, winner combinations and model promotion.

## Final disposition

The fixed four-arm pilot reached its stop: each arm trained 64 updates and all
eight native evaluation panels plus the reducer completed. Training took
47m13s wall time; evaluation/reduction took 1h47m52s
(`09:35:44Z`–`11:23:36Z`). Lead replayed all-image/hash/checkpoint checks and all
42 paired owner-set contrasts from the saved evidence; no extra run was needed.

M alone meets the frozen **observed promising pilot** rule on dev128: IoU50
614→622 owners (+29 gained/−21 lost), IoU60 +3, IoU80 +3, with duplicate/invalid/
cap counts 3/0/0 versus Source 7/0/0. It exceeds R by +13/+2/+8 owners at the
three thresholds. R and B have train recovery but not dev IoU50 recovery;
W's +7 dev IoU50 is mixed with −1/−5 at IoU60/80 and one capped image. B loses
38 dev IoU50 owners and markedly expands output/debt at this dose.

Important limit: M's train strict duplicates rise from 17 to 348, despite its
cleaner dev vector. The dev-based promising label is not an overall stability
claim. Single seed, small fixed dev set, and no canonical-only arm preclude a
robust-generalization claim or attribution of absolute gains to correction CE
alone. The full matrices and cohort diagnostics are retained in results.

**STOP reached:** no added updates, sweeps, refreshed banks, winner composition,
deployment or promotion. A later experiment requires a new user decision.
The sections below preserve the frozen protocol and chronological execution
receipts; this final disposition governs the current status.

Final acceptance checks: nine owned Markdown documents and eleven local links
passed whitespace/link checks; unit/result frontmatter is complete, verified,
and not promoted. Strict OpenSpec validation and `git diff --check` passed;
the research-base decision graph check reports six decisions/six edges valid.
All eight panel exits and the reducer exit are zero, launch source hashes
remain unchanged, and both owned driver PIDs are gone. No unresolved blocking
correction or owned job remains. All twelve implementation tasks are complete;
Git publication, OpenSpec archival and model promotion were not performed.

## Question and strongest alternative

From exact Source step-2444, does annotation-authored correction after a
model-produced prefix improve natural annotated-owner coverage on the fixed
COCO dev128, and do prefix policy, suffix supervision or update surface change
that outcome under one common fixed-dose recipe?

Hypothesis: learning a missed annotation in the model's own context can repair
enumeration. Strongest alternatives are finite training-string memorization,
mere output-length growth/duplication, or a geometry repair misdescribed as new
object discovery. Training CE and forced-prefix recovery are diagnostics, not
the decision outcome. This portfolio compares policies; without a new
canonical-only arm it does not isolate the absolute benefit of adding
corrections over canonical CE alone.

## Population and immutable-source binding

Use existing artifacts only, under `/data/CoordExp/outputs/research/`:

- `qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/`:
  `train.jsonl`, `dev.jsonl`, `manifest.json`; 256 train images / 1,955 annotated
  owners and 128 disjoint dev images / 891 annotated owners. Train JSONL SHA256:
  `05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`.
- Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
  Source adapter: `eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/adapter`;
  adapter tensor SHA256
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
  Use its original sibling selected embeddings, not an updated CE checkpoint.
- Canonical prompt and owner-row token source:
  `qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/ce/round-1/plan.json`.
- Raw Source capture:
  `qwen3-vl-dense-enumeration/2026-09-01-scalable-annotated-owner-shared-dora-pilot/g0/cross-image-v1/g0.4-source-signal-v3/source-trajectories.json`.
  Pair against the baseline's `natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1/`
  non-pad trace and prompt identities before reuse. Exclude trace padding, not
  genuine generated tokens. Do not copy old synthesized terminal contexts.

The apply bank manifest must seal all referenced file hashes, base/tokenizer/
template/embedding identities, prompt/image/media identities, owner IDs,
decoder configuration, code revision and train/dev disjointness before updates.
Current source inspection finds 1,259 covered / 696 missed train annotations at
IoU50 and 160 images with misses. These are planning counts to revalidate during
bank sealing, not new execution results.

Planned durable root (not created by this proposal):
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/`.
Separate `qualification/`, `bank/`, `R/`, `B/`, `M/`, `W/` and `evaluation/`
within this root; preserve failed attempts rather than overwrite their identity.

## Bank and ordering policy

1. Use the existing category-compatible global one-to-one owner matcher at
   IoU50 on the entire Source trajectory; freeze its assignment. Coverage at a
   prefix means the assigned prediction row has completed by that cut.
   Ambiguous/near-match cases remain tagged; they do not trigger a whole-image
   exclusion or a new physical-existence gate.
2. Select one earliest canonical-order missed owner per affected image, tie by
   annotation ID. This bounds the first pilot to at most one paired correction
   per image (expected 160); all 256 images retain the canonical anchor. Later
   missed owners are still present in full-remainder targets when uncovered.
3. Interception prefix h_pre is the exact Source token slice immediately before
   the first complete emitted row whose frozen `geo_sorted_xy` key reaches or
   passes the selected owner's key. Backfill h_post includes that crossing row,
   cutting at its complete closure. Neither rewrites earlier emitted history.
4. If there is no crossing row, both prefixes end at the last complete row,
   before EOS or an incomplete capped fragment. No complete row means empty
   generated history. Mark these pairs as identical-policy events and report
   their count; do not filter them out to manufacture a stronger difference.
5. For each retained prefix, author all not-yet-covered annotation rows in the
   existing canonical x-then-y order. Record the selected owner's exact row
   span even when other not-yet-covered annotations precede it. Keep complete
   category/coordinate/closure tokens; do not claim the retained generated
   history plus canonical suffix is globally sorted. No sampling-success gate.
6. Correction suffixes end at the final `<|box_end|>` and have no invented EOS.
   The ordinary canonical full-image CE target keeps its native EOS in every
   arm. This controls termination supervision without pretending COCO is a
   complete census. There is no inference-time insertion/rewind/second pass.

## Four independent arms

| Arm | Correction prefix / teacher suffix | Correction loss | Trainable surface |
|---|---|---|---|
| R: reference | h_pre / canonical uncovered remainder | All object tokens | Full existing language DoRA A/B/m |
| B: actual-history backfill | h_post / its canonical uncovered remainder | All object tokens | Same DoRA |
| M: selected-owner only | Exactly R's complete string | Selected missing owner's complete row only | Same DoRA |
| W: output readout | Exactly R's complete string | Exactly R's mask | Shared selected-output-row residual; Source otherwise frozen |

Each starts independently from the same Source. Primary contrasts: B−R for
the backfill policy, M−R for suffix-object supervision, W−R for the update
surface. These are not a full factorial; no interactions are identified.
Ordering remains a practical canonical prior, not a proven optimal policy or
a constraint imposed on natural inference. A sorted-vs-random study is absent.

## Loss, dose and resources (accepted apply recipe)

- Per-image loss = canonical mean action-token CE (native EOS included) plus
  correction masked-token sum divided by the full R correction length. The
  latter denominator is fixed across all four arms, including B and M. Images
  without a correction contribute zero correction loss. Global image mean;
  anchor/correction coefficients both 1. Prefix/prompt direct loss is zero.
- M retains R's whole teacher-forced string; only nonselected object numerator
  positions are zero. The selected row's coefficient and terminal anchor do
  not change. Full-vocabulary CE, no QP or selected-row softmax.
- Eight complete train passes: 64 global updates at 32 images/update, same
  seeded per-epoch image ordering (`seed=20260907`), physical batch one with
  gradient accumulation. No held-out selection during training.
- AdamW, constant learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`,
  weight decay 0, global gradient clip 1; full FP32 replay/trainable state,
  native SDPA, dropout disabled. No LR schedule or hyperparameter sweep.
- W uses zero-initialized FP32 residual rows for the train-independent union of
  existing 80-class category/grammar tokens, all 1,000 coordinate tokens and
  native termination tokens. Row IDs/count/dimension are sealed at apply.
  DoRA rank/layout stays Source's, with no rank search.
- Qualification first, then four arms × two GPUs; no automatic GPU borrowing
  or overlapping duplicate job. Nominal production bound is 13,312 image-action
  training forwards: `4 * 8 * (256 + 160)`, before bounded qualification/eval.
  No per-epoch image/token-bank regeneration. W still uses native full-prefix
  forward, not a separately cached approximation. Measure max sequence length,
  forward/backward/save wall time, peak GPU/RSS and artifact size in qualification.
- Operational ceiling: 12 hours per training arm including in-process save
  time; preserve partial artifacts and report incomplete if crossed. A valid
  interrupted arm may resume from a completed update within the same cumulative
  ceiling. No extra optimization steps. Mechanical smoke is at most two
  updates per surface, not a fifth scientific arm.

The fixed LR and number of updates match a recipe, not effective functional
step size or compute across W/DoRA. Report these differences; do not interpret
W failure as proof of representational impossibility.

## Primary evidence and decision rule

Cold natural greedy train256 and dev128 evaluation for each final artifact,
using Source's frozen native prompt/image/decoder settings (including its
token cap and repetition penalty), with no teacher prefix supplied. Replay
Source under that same evaluation topology before scoring arm deltas; reuse a
Source result only when its sealed lineage/topology matches.

Decision-owning outcome: dev128 annotated-owner coverage at IoU50, reported as
integer gains, losses and net versus Source and versus R on the same 891 owners.
Report IoU60/80 as geometry checks, selected-owner and later-owner train
recovery, duplicate/invalid/cap rates, natural EOS, lengths and unmatched
predictions. All 128 dev images stay in denominators, including capped output.
Use the unchanged evaluator, not a new hand-selected subset or repaired output.

Label an arm an **observed promising pilot** only if dev IoU50 improves over
Source without an aggregate drop at IoU60/80 or an increase in duplicate,
invalid or capped-output counts. Report every raw delta regardless of this
label. Unmatched-prediction growth is visible but not automatically a count of
hallucinations under partial COCO labels. A conflicting vector is **mixed**;
train-only gains are **memorization-compatible**; mechanical failure is
**unresolved**, not a negative learning result. Single seed and fixed dev set
do not establish robust generalization or a universally superior policy.

## Acceptance, stop and continuation

Mechanical acceptance requires verified bank/masks, finite gradients on only
the declared surface, one-/two-rank loss/update agreement, completed-boundary
resume, atomic save and fresh cold payload consumption. None substitutes for
natural model-quality evidence. Correct failed mechanics without changing this
estimand/dose; bundle corrections and recheck the original counterexample.

After the fixed four-arm pilot and terminal natural evaluation, write one
results table and close this unit at its actual evidence level. No further
rounds, sweeps, winner combination, deployment or promotion. A conclusion-changing
scientific/resource change requires the user, not an automatic fallback.

Team plan: one owner for the bank/objective/trainer boundary; delegate only a
disjoint cold-evaluation consumer or a named acceptance risk with clear net
savings. Four GPU jobs do not require four agent schedulers. Observe actual
handoff/rework costs during apply; revise the shared native-team skill only
for a demonstrated coordination problem and after a scoped check. No skill
change is justified by this planning turn alone.

## Apply admission receipt

Implementation worktree: `/data/CoordExp/.worktrees/coco-gt-correction-portfolio`,
branch `probe/coco-gt-correction-portfolio`, forked from verified clean
`research-probes` HEAD `73b8b3cc2614db052055c7c49fc33d6207b0f80a`.
Planning source: `/data/CoordExp/.worktrees/dora-prox-linear-n2` at
`e4f932faa80718fc487fc56744e9e16f1abf62a7` plus only the five authored planning
documents. Unrelated dirty work was not copied. Pre-apply SHA256 identities:

| Document | SHA256 |
|---|---|
| proposal.md | `3419326a3acc050f635ed7847a79dc9ccf681668c8d2a818b9b14768cf121576` |
| design.md | `2e62b4a8a6e1212724e59e60d5e9afd0936c8d337504cd615b1ebe83556a0db3` |
| tasks.md | `cbf8bf2fc3bb7a7b0432a0ed69e59bcacafc7e199b1daa384d501d34089d314c` |
| delta spec | `7f01edf3243a9675f29c1c6cf61d42aa984ef865a9cff795d60018bbf1d12f30` |
| unit.md | `42aebdeff6a0fc1b359406921165ace5f076ac756c676eda0183966401e1cd23` |

GPU admission snapshot: eight A100 80GB devices; GPUs 0–5 essentially empty,
6/7 shared with other work. Preserve unrelated processes. Initial qualification
uses 0/1; cold-consumer mechanics may use 2/3 under its separate owner.

## Prelaunch acceptance (2026-09-07)

Evidence: [training qualification](training-qualification.md) and
[cold consumer qualification](cold-evaluation-qualification.md). Lead replayed
all four focused test files after the review corrections: **12 passed**;
the real-checkpoint qualification verifier reports **QUALIFIED**. Strict
OpenSpec validation and scoped whitespace checks passed. One independent
read-only review rebuilt all 256 bank records and checked exact prefix/owner,
mask/denominator/EOS and unequal-work DDP semantics. It found two fail-closed
gaps, not wrong existing training records: self-consistently resealed owner
corruption and foreign optimizer lineage. Both reproduced RED and were repaired
by their single owner; lead replayed the original sensitivity tests in the
12-test suite. No second review round and no remaining blocking findings.

Standards verdict: **lead-accepted at the probe boundary**. Scientific-intent
verdict: **the frozen four-arm contrast is preserved**; acceptance is mechanics
only, not improvement evidence. Bank ID remains
`37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`;
the sealed bank bytes did not change during corrections. Qualification receipt
SHA256 is `c97a7255ef3d0b6f33cb86e8fc2d1d867486910f9b8ae563d4787d716a06c411`.

Final source hashes:
- bank: `25df771ecc10ab81582c7b7fa5aeb86c138b494e4a892ba09bcea5aaca9c2abb`
- trainer: `d9d796c948a3464411b7efe713108ff3ad6d0f68c8afc848e5edd28f08f9a69e`
- cold entry: `8f72bd23a4e363022da010c4e3c2b1e83dee68fd7557c9f31fddb08d4c3bcde8`
- reducer: `3fc93466b0b2c0bc6c6dfeacc58714e9912464472cb5f6824e42dfe20bf3633e`

Launch uses [the fixed shell driver (preserved from archive ref `archive/research-restructure-20260909/coco-gt-correction-portfolio` at `c00043abc`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/run-training.sh): R=GPUs 0/1, B=2/3,
M=4/5, W=6/7, one independent two-rank optimizer lineage per arm. The final
admission snapshot found all eight GPUs empty, no matching jobs, and 1.8 TB
free disk versus a 43.4 GB retained-checkpoint upper bound. The measured
worst-length extrapolation is 2.10 hours per arm, not 12 hours: 12 hours is only
the external per-arm timeout. Preserve partial completed-update checkpoints if
that timeout is reached; do not silently relaunch or extend the dose.

Native evaluation is FP32 SDPA, RP1.0, batch 2, cap 3,084, eight-rank
controller/worker; the exact Source train/dev reuse bindings are in
`pilot-v1/qualification/source-eval-topology.json`. Candidate panels run
serially across all eight GPUs after training settles, per the cold-consumer
packet. Their terminal results remain outstanding.

## Training launch provenance

Started `2026-09-07T08:41:33Z`, tmux session
`coco-gt-correction-pilot-v1`, driver PID `1543722` (pane PID `1543720`).
All four independent two-rank processes were observed loading their Source
models with the exact 64-update commands. Durable witnesses are under
`pilot-v1/launch/`: `driver.log`, per-arm logs and terminal `.exit` files,
`started-utc.txt`, `finished-utc.txt` when settled, and `code-sha256.txt`.
Launch/liveness is not a completed-update or model-quality claim. Join this
invocation on continuation; never run the driver again over its existing roots.

## Training terminal receipt

All four arms exited 0 and the driver settled at `2026-09-07T09:28:46Z`:
47 minutes 13 seconds wall time from launch, not the conservative 2.10-hour
extrapolation. Lead verified exactly 64 checkpoint manifests per arm, final
update 64, unchanged shared bank ID and the fixed recipe. Each arm records
3,328 image-action forwards; all four total 13,312, matching the declared dose.
The launch code-hash receipt still verifies all producer/consumer files.

| Arm | Final checkpoint ID | Recorded training seconds |
|---|---|---:|
| R | `4efbec695067ec4f5a7c1de30add419046e0ac210067a27bd013aa9df71c033d` | 2817.52 |
| B | `67684f1a842c3c454b45c7efd0c2706c9a7248cb571496353025471117da5d9d` | 2799.99 |
| M | `4dcccaa341763841d008441bd22bd66ac6f90ba810c2ec38cfeab64b6e79b03b` | 2798.57 |
| W | `1a068bf17c237bda42c4f73c592c81331dfa001e3fc4228383f42b089b54a0da` | 1240.80 |

Training is mechanically complete. Natural evaluation and scientific
interpretation remain outstanding; training-loss comparisons do not establish
the primary outcome. The one-shot training monitor
`12371f26-0a98-4573-8ad6-e8f2eaf7cadf` fired on the terminal log and was read
once; its pointer was not used as a substitute for these artifact checks.

## Natural-evaluation launch provenance

The [evaluation invocation](evaluation-execution.md) started at
`2026-09-07T09:35:44Z`, tmux `coco-gt-correction-eval-v1`, driver PID `1588621`.
The [fixed serial driver (preserved from the same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/run-evaluation.sh) runs exactly eight candidate
panels then the existing provenance-locked reducer; it has no retry branch.
Lead verified the live driver and its code-hash receipt. The production cold
loader accepted all four final payloads before launch. First-panel eight-worker
liveness is recorded, but no completed natural-quality result is claimed yet.
Continuation must inspect `pilot-v1/launch-evaluation/driver.log` and
`terminal-marker.txt`, not launch another driver into the existing roots.

## Native-team feedback at the completed execution boundary

The two Sol-high builders owned training and cold-consumer packages and
coordinated their shared checkpoint interface directly. One Sol-high bounded
prelaunch review found two reproducible validator gaps; the original owner
fixed them and the lead rechecked the counterexamples without a second review.
A fresh Sol-medium execution owner handled the changed launch permission,
delivered the durable invocation, and stopped rather than staying active to
poll. Another Sol-medium package owns the final saved-artifact readout; lead
acceptance remains separate. No nested team or additional scheduler was used.

This supplies no new coordination-policy defect requiring a shared skill edit:
the existing native-team guide already prescribes these ownership, fresh-phase,
bounded-review and durable-wait behaviors. The observed validator defects were
corrected in code, not generalized into another agent ritual. Therefore
`native-subagents-guide/SKILL.md` is intentionally unchanged for this run.
No measured model-cost comparison is claimed; the monitor's avoided-poll
estimates are not dollar or token savings measurements.
