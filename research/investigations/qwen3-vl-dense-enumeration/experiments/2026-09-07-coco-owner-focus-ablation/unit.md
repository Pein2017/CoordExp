---
title: COCO owner-focus versus correction-weight ablation
description: Three fixed-dose arms with real training batches and a fresh COCO holdout.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-07-coco-owner-focus-ablation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# COCO owner-focus versus correction-weight ablation

## Final disposition

**Complete at the fixed stop; mechanics and bounded readout lead-accepted;
no model or mechanism promotion.** [Verified results](results.md) own the
scientific readout. The sections below preserve the accepted protocol and
chronological execution; this disposition supersedes their pending/live states.

On fresh holdout512 / 3,759 annotated owners, M fails the frozen observed
owner-focus advantage rule. M-Rweak at IoU50 is -52 owners (96 gained, 148 lost),
with paired image-bootstrap 95% total-equivalent interval [-116,3]. M-R is +25
[-35,94] and M-Source is +33 [-37,106]; none establishes a robust positive
holdout contrast. M also has six fewer net matches than Source at IoU80 and raises strict
duplicate candidates 21→82 and capped images 11→18. The failure of the
predeclared acceptance rule is decisive for this run, not a proof that owner
focus is ineffective in the population or equivalent to weaker correction.

Rweak has the strongest observed Source-relative coverage vector (+85/+47/+6)
and fewer caps (11→8), but strict duplicate candidates rise 21→276. Therefore
it is not a stable accepted winner either. The token-weight explanation remains
viable; token-mass matching does not match gradient norms. No canonical-only
arm or multi-seed population claim is present.

All three arms finished 64 updates. All eight natural panels exited zero and
the fixed reducer completed without retry, filtering or new scientific runs.
Source evaluation took 1h23m40s; the three candidate pairs ran concurrently for
2h27m31s, ending `2026-09-07T17:07:02Z`. The lead freshly replayed frozen hashes,
all-image/checkpoint identities, independently reconstructed owner matching,
all 18 paired contrasts and all 18 bootstrap intervals. The sole result JSON
SHA256 is `418bfeea25d07399917b644bca3361477373bb0d38a20f9da8cfc94c40f08491`.
All three owned training/Source/candidate drivers have exited. Monitor service
registration later timed out; the remaining durable job was joined through its
existing log/PID without restarting services or experiments.

**STOP:** no further dose, sweep, seed, refreshed bank, combined winner,
promotion, commit, push or archive. The user-requested historical skill audit
found no required text change. The bounded Astra/low execution trial passed
without substantive correction, but supports no comparative cost/ranking claim.

## User authorization and question

User approved this successor and parallel autonomous execution on 2026-09-07:
"好的,快速并行推进.我们要迅速获得结论性的 ablations. decode可以 batch size = 4 起步."
The completed [portfolio](../2026-09-07-coco-gt-correction-portfolio/unit.md)
remains immutable scientific evidence, not an authorization to extend its runs.

From the same step-2444 Source, does selected-owner correction (M) improve
natural annotated-owner coverage beyond full-remainder correction (R), even
when the latter has the same supervised-token weight budget (Rweak), on a
fresh, outcome-blind 512-image COCO holdout?
Strongest alternative: M merely weakens correction rather than benefiting
from owner selection. Secondary risk: duplicate/length degeneration despite
small aggregate coverage gains. No physical completeness or hallucination
claim follows from partial COCO annotations.

## Frozen scientific recipe

- Independently initialize R, M, Rweak from the original Source, never M's
  completed checkpoint. Same existing sealed train256 correction bank:
  `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
- Reuse its exact prompts, media, prefixes, full teacher strings, owner masks,
  complete-row cuts and canonical EOS anchor. No refreshed bank or filtering.
- R and M have exactly the prior per-image objectives. For a correction image,
  Rweak multiplies R correction loss by `sum(M_mask) / sum(R_mask)` from that
  image's sealed bank event. Canonical loss is unchanged. All arms retain the
  full R denominator. This matches token weight mass, NOT gradient norms.
- Same full language DoRA trainable surface, optimizer and LR as the prior
  portfolio; 64 global updates, effective image batch 32, seed `20260908` for
  paired ordering. Actual microbatch is qualified below, not a dose change.
- Freeze 512 existing COCO images without using model outcomes; exclude all
  discoverable prior local train/selection panel IDs and document exclusion
  scope and Source training provenance. Do not label historical Source training
  overlap unknown as proven absent. If 512 cannot be admitted, ask the lead;
  do not silently weaken the population claim or acquire another dataset.
- Evaluate Source and all three final checkpoints on the same new holdout.
  Evaluate train256 for stability diagnostics. Existing dev128 is historical
  selection evidence, not a fresh test and need not be decoded again.
- Native greedy FP32 SDPA, repetition penalty 1.0 and cap 3084 remain fixed.
  Decode starts at actual per-device batch 4; all compared arms including
  Source use the same qualified batching profile. No filtered failures/caps.

## Decision and stop rule

Primary: paired annotated-owner IoU50 coverage M-Rweak and M-R, with M-Source
as the usefulness baseline. Report IoU60/80 and per-image paired uncertainty,
not only pooled owner counts. Strict duplicate, invalid/drop, cap and length
vectors accompany all claims. An observed focus advantage requires positive
IoU50 contrasts against both controls and Source, no negative IoU60/80 versus
Source, and no increase in duplicate/invalid/cap counts versus Source.
Report uncertainty honestly: a small or inconclusive contrast is not a
conclusive mechanism claim. Token-budget matching does not isolate all
gradient-scale effects. No canonical-only arm is introduced.

STOP after three 64-update runs and fixed final natural evaluation. No sweep,
early dev selection, extra seed, automatic dose extension, M+W combination,
refreshed bank, model promotion, commit/push or archival. Mechanical recovery
preserves failed artifacts and changes no scientific recipe.

## Engineering and allocation

This successor owns its bounded implementation delta; the previous OpenSpec
and prior production recipes must not be weakened in place. Prefer minimal
probe-specific entry reuse, no generic trainer or scheduler. A successor mode
may share implementation only with explicit identity-bound old/new guards.

- Training owner: actual padded multimodal microbatching and Rweak objective,
  training manifests/checkpoints, one production-shaped two-rank qualification.
  GPU 0/1 exclusively during qualification. Preserve exact image-mean weighting,
  causal supervision, native positions, frozen tensors and cold payload identity.
- Evaluation owner: outcome-blind holdout, cold Source/R/M/Rweak consumer and
  paired reducer, batching/throughput qualification. GPU 2/3 exclusively during
  qualification. Coordinate checkpoint interface directly with training owner.
- Independent history adviser: read-only agent-call/acceptance audit of the
  requested historical task; no code or skill writes and no experiment gate.
- Lead: scientific contract, GPU allocation, cross-package decisions and one
  acceptance replay. No nested delegation. Owners complete their own fix/check
  loops. No automatic additional reviewer.

Before production, compare the same small update at microbatch 1 and a real
larger batch, including uneven correction counts/lengths and longest-row memory.
Check masked loss/gradients or update equivalence within declared numerical
tolerance and a saved cold consumer. Benchmark only a bounded batch ladder;
select by useful tokens/s or images/s, elapsed time and peak memory, not merely
allocated memory. Keep effective batch32 and 64 updates unchanged. Fail closed
on unsupported padding/position/serialization behavior. Decode batch4 versus
single-example sensitivity check must cover native EOS and long outputs.

Planned production: three concurrent two-GPU train jobs on 0/1, 2/3, 4/5,
with 6/7 available for Source evaluation once evaluation is qualified. May use
all eight GPUs for final panel evaluation after training; one invocation owns
each output root. Bound each train arm by the inherited 12-hour safety ceiling,
not a target duration. Qualify finalizer, payload and wall/RSS bounds before
launch. Production starts only after lead acceptance of package receipts.

Durable output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/`.
Package records [training qualification (preserved from archive ref `archive/research-restructure-20260909/coco-gt-correction-portfolio` at `c00043abc`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/training-qualification.md) and [evaluation qualification (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/evaluation-qualification.md)
will bind exact code/input/holdout/checkpoint identities and executable commands.

## Admission clarification

The evaluation owner found 4,952 existing canonical COCO-val rows and selected
512 images / 3,759 annotated owners using outcome-blind SHA256 ranking seeded
20260908. The local exclusion scan covers 26 input artifacts / 2,259 unique
IDs, including 20 COCO-val exclusions; its manifest owns the precise scope.
Source step-2444 optimizer training used COCO-train. Historical Source forward
evaluation was configured on COCO-val, so this is a fresh successor panel, not
a claim of no prior operational exposure. The fixed terminal Source checkpoint
was not selected using this new panel. No scientific dose or contrast changed.

Training mb2 qualified; one final mb4 production-shaped one-update mechanics
comparison is allowed. Stop the batch ladder at four: retain two on OOM or at
most 5% throughput improvement, otherwise select four only with the same parity
and sufficient memory margin. This is execution qualification, not another
scientific arm. All three production arms use the selected physical batch.

## Training lead acceptance

Lead replayed the training consumer tests (5 passed) and artifact qualification
(`QUALIFIED_CANDIDATE`), inspecting padded native positions and the global
image-mean scaling at the real call boundary. Select actual microbatch2 for
all arms: the final mb4 rung was 16.81% slower (57.179s versus 47.569s), reserved
81.28GB, and exceeded the frozen component-loss tolerance (5.245e-6 versus
4e-6). Its retained evidence is not relabeled as passing. No further batch rung.
The training receipt owns full parity/resource bounds and immutable checkpoint
IDs; this is mechanics acceptance, not a scientific result.

For useful overlap, the launch allocation is R on GPU0/1, M on GPU4/5, Rweak
on GPU6/7. GPU2/3 stay with the evaluation owner, then the Source baseline.
Only device placement changes; all train jobs remain two-rank/global32/mb2.
The [fixed launcher (preserved from the same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/run-training.sh) has no retry or additional dose. Lead
inspected the actual Rweak step1 cold-consumer completed summary, four scored
rows and exact checkpoint identity. Its run-manifest SHA256 is
`16c8a069a0d310864ff3330e60d13f00cc4cab9024b53f88e078f3d405dbf6cd`.
This closes the real saved-artifact consumption gate without another review.

## Live training invocation

All three fixed 64-update arms launched in durable tmux
`coco-owner-focus-train-v1`; pane PID at launch `1944163`. Exact start time,
driver PID, source hashes, per-arm logs/exits and final completion markers are
owned by `focus-v1/launch-training/` under the durable output root.
Do not relaunch an existing arm or driver. Success marker:
`FOCUS_ALL_TRAINING_COMPLETED`; failure markers `FOCUS_ARM_FAILED` or
`FOCUS_TRAINING_FAILED`. No scientific result is available yet. Evaluation
qualification continues independently on GPU2/3, then Source evaluation may
overlap the three training jobs on those same GPUs after its own acceptance.

## Evaluation lead acceptance and execution handoff

Lead freshly replayed 12 focused old/new consumer tests and
`verify_coco_owner_focus_evaluation.py`: `QUALIFIED`, old frozen files unchanged,
Source composition and frozen raw-row guards accepted, holdout 512/3,759,
single-versus-batch4 tokens/predictions identical, real two-rank Rweak cold
consumer completed. [Evaluation qualification (preserved from the same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/evaluation-qualification.md)
owns exact identities and the bounded input exclusion receipt.

All Source/candidate panels use two ranks and per-device batch4. The deliberately
long qualification panel showed LOWER throughput than batch1, not a speedup;
its 9.8-hour per-arm linear extrapolation is a long-tail capacity scenario, not
the ordinary-panel forecast. Actual elapsed time will be recorded. No batch8
or generic inference rewrite is added to the critical path.

Primary holdout512 is decoded before the train256 stability diagnostic for each
arm, without intermediate selection or changing the final stop. Source on2/3
may overlap training; candidates require complete update64 identity acceptance.
A fresh Astra/low execution owner is authorized to launch Source only and
prepare the three-arm candidate/reduction launch commands. This is an actual
package trial requested by the user, not evidence that Astra/low is already
faster, cheaper or interchangeable with Sol/high. Lead retains acceptance.

## Source live execution and bounded Astra/low observation

Source launched at `2026-09-07T14:25:52Z` in
`coco-owner-focus-source-eval-v1`, driver PID `2061168`, on GPU2/3, holdout first.
The [execution receipt (preserved from the same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/evaluation-execution.md) binds controller/worker startup,
fixed code/input hashes, script hashes and all terminal markers. Candidate and
reduction scripts are prepared but NOT launched. Lead inspected all three
scripts, freshly replayed their bash syntax and the seven source/launcher hash
checks, and verified the live Source pane. The execution package is accepted;
Source startup is not result completion.

This Astra/low package needed no substantive correction. Its documented
first-clock-to-candidate interval is 3m48s, only a lower bound, not exact
spawn-to-acceptance. No matched Sol/high counterfactual or verified dollar price
is available. It supports using Astra/low for another similar bounded execution
package, not a general model ranking or lower-cost claim. Shared skill unchanged.

Next action is event-driven evaluation completion. Only after all eight panels
complete, run the prepared reducer and write the single final scientific
readout. No owner remains alive merely to poll a job.

## Training complete; candidate evaluation launched

Training terminated at `2026-09-07T14:37:53Z`, 43m49s after the durable driver
start. All three exits are zero. Lead loaded all three final checkpoints through
the cold successor consumer and checked all 192 update manifests for the exact
paired seeded batch schedule, two ranks, frozen tensors/no unexpected gradient,
and complete update numbering. Each arm executed 3,328 trajectory forwards and
backwards in 1,024 actual model forwards, preserving the fixed eight-pass dose.
All final recipes retain effective32/actual-mb2/seed20260908; producer hashes
match the qualified implementation and launch receipts.

| Arm | Accepted update64 checkpoint ID |
|---|---|
| R | `f85bf97eb7040ff9c45240aa8314093c5725509a13336383394333728bcded6f` |
| M | `030791834471d93a01b7c244ee1f4fb2a2b6a9fbd67a4493fc4ee3d18afbfc9a` |
| Rweak | `6b1f5b32dc1a5b7503ad81b691855ffcda3571f6d5433bc09838de07a88bf964` |

The prepared candidate launcher was started with explicit lead update64
acceptance in tmux `coco-owner-focus-candidate-eval-v1`, pane PID `2109486`.
Its `focus-v1/launch-candidate-evaluation/` receipts own exact start/PID/hashes,
per-arm panel logs and exits. R/M/Rweak evaluate concurrently on0/1,4/5,6/7,
while Source continues on2/3; every arm uses the same two-rank batch4 profile.
No results are inferred from training losses or incomplete panels.

Source evaluation completed at `2026-09-07T15:49:32Z` (1h23m40s). Lead replayed
the frozen-input, Source-composition, complete-artifact and two-rank batch4
checks on all 512 holdout and 256 train rows; both panel exits are zero.
Rweak holdout also exited zero, while the candidate driver remains active.
The paired result and scientific disposition remain pending all six candidate
panels; no partial-result selection or additional run is authorized.
