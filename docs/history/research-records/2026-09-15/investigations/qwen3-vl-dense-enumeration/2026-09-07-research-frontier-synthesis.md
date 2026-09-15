---
title: Research Frontier Synthesis — 2026-09-07
description: Decision-oriented synthesis of readout compilation, internal optimization, CE generalization, termination, and trajectory credit.
type: investigation
role: research-synthesis
authority: non_normative_research
architecture_promotion_status: not_promoted
status: complete
updated: 2026-09-07
---

# Research frontier — 2026-09-07

Web reading mirror: [Notion research frontier](https://app.notion.com/p/3d49d9ce3f5981cdb5dbdca4c4e7d379).
The mirror includes the decision-bearing tables and interpretation, not merely
local-path links. Raw model payloads, logs and source datasets remain local.

This is a dated reading snapshot, not a new experiment or launch authorization.
It covers the Qwen3-VL dense-enumeration program, with fresh September results
from this worktree and read-only evidence from the two named sibling worktrees.
It is not an audit of every CoordExp production or historical research project.
Existing unit results own their scientific claims; executable artifacts remain
local. The September CE-control portfolio is closed at every registered stop.
The separate N256 scaling result remains partial, not silently closed with it.

## Executive interpretation

**The central question has shifted from whether the model can fit dense scenes
to which learned corrections survive beyond the fitted routes and images.**

- Shared output-head QP can compile Human13 completely, but ordinary CE can
  also fit the same thirteen images through existing internal DoRA parameters.
  Same-panel fit is no longer a discriminating algorithmic advantage.
- Output-QP semantic routes compress better than coherent wrong-route controls
  through N16. That is structured readout compatibility, not proven transfer.
- The tested internal prox-linear QP and soft-owner QP correction have not
  beaten their simpler respective controls on their registered outcomes.
- On COCO train256/dev128, stronger training fit does not ensure development
  improvement. Broader data mitigates degradation; suppressing EOS supervision
  trades some owner recall for severe generation debt.
- Four refreshed RLOO updates give a small terminal primary gain without the
  EOS-zero development cap explosion, but not uniform threshold improvement,
  seed robustness, or equal-compute superiority.

## Vocabulary and non-comparable evidence surfaces

- **Readout W / output QP:** modify selected output-head rows by one shared
  residual while retaining the hidden network. Its convex certificate belongs
  to captured teacher-forced states; natural decoding needs separate evidence.
- **Internal DoRA:** change existing language-network adapter parameters.
  Magnitude-only means196 vectors; full A/B/m means588 tensors. These are not
  the output-QP surface and their norms are not directly comparable.
- **Prox-linear QP:** solve a local linearized internal-network optimization
  subproblem. Here proposals were bounded and inexact, not exact global solves.
- **Soft-owner QP:** correct a proposed RLOO parameter update using local owner
  constraints. This tests a preservation signal, not output-readout capacity.
- **CE:** cross-entropy on canonical GT transcripts. **RLOO:** REINFORCE with
  leave-one-out baseline; here K4 current-policy trajectories, TP50/GT reward.
- **Owner count:** category-consistent globally one-to-one matched annotated
  GT instances. IoU50/60/80 use thresholds0.5/0.6/0.8. Summing threshold counts
  counts an owner multiple times; it is not a number of additional objects.
- **Development:** historically used panel, not untouched test. Unmatched
  valid predictions remain unknown under incomplete annotations, not automatic FP.

## 1. Readout compilation and semantic compression

### Observation

Human13 shared output QP reaches65/65,123/123,392/392 at N2/N4/N13, including
all three IoU thresholds, natural EOS and zero registered hard debt. A fixed
Image2299-authored payload nevertheless reduces a separate legacy12 cohort
from138 to18 IoU50 owners on its particular checkpoint. Those are distinct
experiments; do not describe that negative as Human13 leave-one-image-out.

The sibling N256 study fixes1136 output rows and compares semantic routes with
four complete-route derangements. Its certified train-only aggregates show:

| Stage | Semantic / smallest-null norm | Semantic / best-null difficulty-normalized cost |
|---|---:|---:|
| N4 | 0.19745 | 0.38495 |
| N8 | 0.21477 | 0.53017 |
| N16 | 0.23369 | 0.57604 |

N4 energy is93.5948% coordinate rows,6.4022% category rows and0.0030%
format/EOS. Thus formatting alone is insufficient to explain its advantage.
However, before refitting, the N4 payload changes new N8-position satisfaction
from116/186 to112/186; the N8 payload changes new N16 satisfaction from328/536
to326/536. Shared train fit is not an automatically reusable correction.
The original target-blind transfer ladder stopped at N2 with42 Source-owner
losses. N32 has a semantic receipt but no complete matched-null aggregate at
this read; the N32 verdict remains unresolved. No N256 result is claimed.

### Interpretation and possible next discriminator

Keep output QP as a certificate and representation diagnostic. The strongest
alternative is image/prefix-specific address lookup on structured features.
A useful next question is whether a frozen correction helps never-fitted
states/owners before refitting, rather than whether another larger training
system can be solved. Any new transfer experiment must remain separate from
the closed original transfer ladder. This report does not take over N32.

## 2. Internal capacity, loss and optimizer

### Observation

On the same N2 magnitude surface, the tested squared-margin Adam ends at
37/28/21 natural owners after300 updates; bounded inexact prox-linear QP ends
at24/19/8 with greater measured cost. Neither meets complete fit. A fresh CE
arm then reaches65/65/65 at step66. A separate fresh Human13 CE replay reaches
392/392/392 at step140 from Source173/159/91, training only196 magnitude vectors.
The N13 payload matches the historical same-seed CE payload exactly; this is
independent execution reproduction, not a new-seed replication.

### Interpretation and possible next discriminator

Internal capacity and canonical-to-greedy realization are sufficient on this
finite panel. Do not keep selling internal QP as necessary for fitting it.
If internal constrained optimization is pursued, define its added value as
equal-success cost, effective change, preservation or transfer. The accepted
prox-linear steps predicted finite reduction well; the tested bottleneck was
inner-model descent/solve quality, not a demonstrated universal failure of
local linearization. A different optimizer is a new experiment, not a repair
that retroactively erases this bounded negative.

## 3. Population CE: data breadth and parameter dose

All terminal entries below use the same original Source and fixed diagnostic
panels. Native courses have256 updates; the density2048 arm has the same total
16384 image presentations but eight rather than64 repeats per training image.
Owner/token presentations are not exactly matched. The train256 read is only
the original nested subset of the broader2048 training population.

| Native course at step256 | Dev128 IoU50/60/80, 891 owners | Train256 IoU50/60/80, 1955 owners |
|---|---|---|
| Original Source | 614 /585 /451 | 1259 /1190 /908 |
| Full A/B/m CE, lr1e-5 | 596 /563 /418 | 1394 /1317 /1046 |
| Magnitude-only CE, lr1e-5 | 612 /585 /451 | 1256 /1191 /911 |
| Magnitude-only CE, lr0.003 | 522 /491 /367 | 1688 /1639 /1507 |
| Full CE, density2048 | 611 /578 /449 | 1319 /1239 /943 |

**Inference:** broader exposure reduces the original development damage but
does not beat Source on primary terminal IoU50. Small-lr magnitude-only changes
little; large-lr magnitude-only fits strongly while damaging development.
The latter changes dose as well as surface relative to full CE; it cannot
isolate a pure parameter-surface effect. Training success is not the scarce
evidence here. We need evidence for reusable corrections and preserved owners.

The cheapest first analysis, before another training grid, is to classify the
existing lost/gained owners into missing-object discovery, box extent/binding,
termination and duplicate behavior. This is a proposed descriptive analysis,
not a causal answer or authorization to relabel the fixed metrics.

## 4. EOS supervision and incomplete-label objectives

The EOS-zero control removes only terminal EOS from the CE numerator and keeps
the original denominator. It does not introduce censored or set likelihood.

| Development checkpoint | EOS-zero IoU50/60/80 | Length caps /128 |
|---|---|---:|
| Step64 | 629 /594 /456 | 35 |
| Step256 | 620 /581 /428 | 70 |

Source has zero development caps. Terminal prediction count rises1103→3449;
dropped predictions rise58→19635. The primary gains are real matched owners,
but longer output provides more matching opportunities and high-IoU quality
later regresses. **Neither “EOS is irrelevant” nor “remove EOS and solve recall”
fits the evidence.** Missing annotations plausibly create false stopping
supervision, but this mechanism has not been isolated by this control.

A bounded next discriminator could compare the retained outputs at a common
output-token budget, without replacing the registered primary result. It tests
whether the gain survives equal generation opportunity, not whether the whole
training intervention is causal. Any new partial-label objective needs an
explicit definition of what remains supervised and how termination is learned;
simply treating every unknown as positive or suppressing all EOS is unsupported.

## 5. Refreshed RLOO and owner-preserving updates

Two Source-started FP32/SDPA full-DoRA chains used the same256 images, persistent
AdamW lr2.5e-6 and four global updates. RLOO uses complete generated actions,
per-image TP50/GT and no extra EOS/length/duplicate/unknown penalty. Neutral
reward for an unmatched prediction does not make its token gradient zero.

| Development checkpoint | CE IoU50/60/80 | RLOO IoU50/60/80 |
|---|---|---|
| Source | 614 /585 /451 | 614 /585 /451 |
| Round1 | 619 /587 /449 | 619 /589 /450 |
| Round2 | 616 /587 /449 | 616 /587 /448 |
| Round3 | 613 /588 /452 | 616 /585 /447 |
| Round4 | 612 /589 /452 | 618 /589 /451 |

Terminal train CE is1264/1199/912; RLOO is1278/1209/923. Development RLOO minus
CE is+6/0/-1, or+0.67 percentage points at primary IoU50. Both terminal dev
reads have natural EOS on128/128 images; both terminal training reads have
four caps, like Source. The four-step course is not the256-step CE experiment.
RLOO uses4096 versus1024 replay forward/backward calls,297272 versus74980 action
tokens, plus4096 sampled trajectories. All groups and actual actions remain
included. Four banks and eight persistent optimizer states passed checks.

Separately, the C-anchored train248 sibling experiment finds refreshed RLOO
+1/+4/+6 versus local soft-owner QP +1/+2/+5 relative to its own anchor.
Its248 images/1798 owners and updated starting adapter differ from Source256;
the numbers must not be pooled. All eight QP-threatened owners were retained
by ordinary RLOO too, so the QP did not rescue a realized loss. Its local signal
failed to identify the actual discrete route differences at this dose.

**Interpretation:** RLOO remains a plausible, comparatively simple behavioral
baseline, not a demonstrated superior algorithm. Seed variation, unequal
effective movement and unequal compute are strong alternatives for its small
Source256 advantage. If another training test is chosen, ask for a replicated
fixed-endpoint contrast and explicitly choose update-, replay- or wall-matching.
Do not call these matching rules interchangeable or add a QP guard without
evidence that it predicts real owner loss.

## 6. Autoregressive state, geometry and explicit architecture

Historical fixed-state interventions show that earlier complete rows, ordering
and coordinates can alter later owner access; the native recurrence is not an
order-invariant covered-owner ledger. This is a mechanism observation, not a
proof that a new ledger/commit/bridge architecture is required. The newer CE
same-panel success removes one reason to infer an expressivity failure.

Keep representation availability, owner binding, box extent, visitation and
termination separate. A useful architecture-facing discriminator would change
only an explicitly defined owner/coverage state while holding learned policy,
candidate evidence and budget fixed. That is a proposal requiring its own
authority. No such architecture change or causal future-use result is newly
established by the September optimization portfolio.

## Decision menu — recommendations, not launches

1. **If the goal is better development behavior:** first inspect existing
   gain/loss and matched-output-budget evidence; then decide one bounded
   replicated CE/RLOO or data-exposure contrast. Do not launch all knobs together.
2. **If the goal is a mechanism claim:** prioritize pre-refit transfer of a
   readout correction or an owner-specific state intervention over another
   same-panel fit demonstration. Compression alone is not the endpoint.
3. **If the goal is a new optimizer:** require equal-success or actual-owner-
   preservation benefit over CE/Adam or ordinary RLOO. Current QP variants have
   not supplied that evidence. Inner-solve progress and KKT convergence alone
   do not satisfy the behavioral objective.
4. **If the goal is incomplete-label learning:** retain annotated-owner recall
   as a bounded metric and separate extra supported instances, duplicates and
   hallucinations through adjudication. The EOS-zero result motivates a
   question; it does not select the replacement objective.

My preferred ordering is existing-output diagnosis → one simple learning
contrast → only then a justified new loss/optimizer/architecture. A new seed
tests robustness; a new held-out panel tests a different question. Neither
should be silently substituted for the other. The user owns the next choice.

## Evidence owners

- [Human13 output-QP](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/results.md)
  and [fixed Image2299 payload transfer](experiments/2026-08-31-image2299-g46-payload-legacy12-transfer/results.md).
- [N2 prox-linear QP](experiments/2026-09-05-dora-prox-linear-n2/results.md),
  [N2 CE/margin control](experiments/2026-09-05-dora-ce-margin-n2-ablation/results.md),
  [fresh Human13 CE](experiments/2026-09-05-human13-pure-ce-replay/results.md).
- [SFT256 baseline](experiments/2026-09-05-sft256-dev128-baseline/results.md),
  [native-control portfolio](experiments/2026-09-06-ce-controls-rloo-successor/unit.md),
  [EOS](experiments/2026-09-06-eos-numerator-control/results.md),
  [magnitude](experiments/2026-09-06-dora-surface-control/results.md),
  [density2048](experiments/2026-09-06-density2048-exposure-control/results.md),
  [Source CE/RLOO](experiments/2026-09-06-source256-ce-rloo/results.md).
- Sibling N256 artifact owner: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-n256-shared-output-qp-norm-scaling/`;
  `train-n4-v3/aggregate-train-n4.json`, `train-n8-v1/aggregate-train-n8.json`,
  `train-n16-v1/aggregate-train-n16.json`. N32 lacks a full aggregate at read time.
- Sibling soft-owner result: `/data/CoordExp/.worktrees/c-anchored-owner-mechanism-audit/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical/results.md`.
- Exact source hashes and Notion readback are recorded in the adjacent
  `2026-09-07-research-frontier-sync.json`. This synthesis does not modify
  sibling ownership, raw artifacts, historical claims or stopping rules.
