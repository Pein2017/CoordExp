---
title: Human-13 Row-Contrast and Geometry-Preservation Successor
description: A two-arm, low-dose same-panel probe of row-level duplicate redirection, rectangle-valid greedy decisions, and first-order preservation of previously greedy-visible owner coordinates.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_in_progress
unit_id: 2026-08-13-human13-row-contrast-geometry-preservation-successor
topic: qwen3-vl-dense-enumeration
status: authorized_in_progress
evidence_status: design_frozen_execution_pending
updated: 2026-08-13
---

# Human-13 Row-Contrast and Geometry-Preservation Successor

## Decision

Run one bounded successor to the completed
[Human-13 K-union-to-greedy screen](../2026-08-12-human13-k-union-to-greedy-overfit-screen/results.md).
The successor compares only two fresh-Source treatments at cumulative panel
exposures one and two:

- **R1 — row contrast + rectangle gate:** retain the A4 any-valid K-hit target
  direction and Source replay, replace final-coordinate duplicate
  unlikelihood with hierarchical complete-row contrast, and require greedy
  `x2/y2` decisions to remain rectangle-valid.
- **R2 — R1 + G preservation:** apply the same objective, then project an
  adverse accumulated trainable-parameter gradient away from a frozen
  Source-`G` coordinate watch gradient before clipping and AdamW.

The already-authoritative A4 exposure-two outcome is the historical baseline;
it is not rerun.  No third treatment, long dose, architecture change, or
same-batch update/decode loop is authorized.

The execution topology with the shortest expected wall clock uses six GPUs:
two independent world-size-one training jobs and four independent HF
fp32/SDPA evaluations for `R1@1`, `R1@2`, `R2@1`, and `R2@2`. Two additional
available GPUs remain operational reserve.

## Why this is the next smallest experiment

The predecessor provides three conclusion-changing observations:

1. A4 at exposure two reaches `H+8/G-5/M+5`, but emits `32` duplicates and
   `14` malformed rows. It is useful enough to repair, not safe enough to
   extend.
2. Exact re-parsing shows almost every malformed row is a geometry-invalid
   rectangle (`x1>=x2` or `y1>=y2`), not a general wrapper/parser failure.
3. The current duplicate loss acts at only the duplicate row's final
   coordinate, while all five examined low-dose `G` losses for both A4 and A6
   are coordinate-IoU drift rather than duplicate filtering or matcher
   competition.

Thus the next experiment tests a better loss geometry and a first-order
retention constraint on the existing native architecture. It does not reopen
the external owner bridge or ask whether K-miss owners need new visual support.

## Frozen substrate

The successor MUST reuse, byte-for-byte where applicable:

- Source checkpoint, base model, language-only DoRA surface, frozen vision
  tower, frozen multimodal aligner, frozen embeddings, fresh AdamW state, and
  optimizer values from the predecessor;
- the exact 13-image/392-owner panel;
- sealed Human-13 manifest SHA-256
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`;
- frozen sets `G=173`, `H=73`, `M=146` and the same selected native H rows;
- chronological class-agnostic prediction-to-prediction duplicate definition
  `IoU>0.95`, before owner matching;
- original-prompt clean-greedy HF fp32/SDPA batch-one evaluation at repetition
  penalty `1.0`; and
- cardinality-first, maximum-total-IoU one-to-one owner matching.

`M` remains gradient-neutral. A finite K-miss is not a negative label, and no
partial K-union sequence licenses a positive terminal token.

## Frozen successor ledger

The canonical sidecar contains all 78 aligned duplicate states for diagnosis.
The actual R1/R2 low-dose training subset is frozen to the 12 sealed-manifest
events and one deterministic native alias per uncovered owner. The measured
alternative (all optional states/aliases) was 1,178 segments, 160 packs and
1,845,364 packed tokens per exposure; the chosen subset is 411 segments, 55
packs and 624,922 packed tokens per exposure. This is a compute decision, not
evidence that the omitted static states are invalid.

The successor sidecar is derived without model execution from:

1. all complete duplicate events already sealed in Source and K trajectories;
2. complete duplicate rows in the authoritative A4 exposure-one and
   exposure-two raw outputs, provided exact tokenizer/parser alignment and
   artifact identity validation succeed; and
3. all Source `G` rows and selected native `H` rows needed for valid candidates,
   rectangle sites, and coordinate watch sites.

Each event binds image, trajectory/artifact identity, exact raw token prefix,
complete duplicate-row tokens, four coordinate positions, covered owner IDs,
uncovered `T=G union H` owner IDs, candidate owner IDs and aliases, and content
hashes. Events are deduplicated only by identical image, prefix, and duplicate
row tokens. No event is dropped because an image is duplicate-heavy.

If an A4 output cannot be exactly aligned, its events are excluded with a
typed reason; the sealed Source/K events remain usable. Silent approximate
span recovery is forbidden.

## R1 objective

### Hierarchical row contrast

For a candidate row `r` at an exact decision prefix, define the length-
normalized teacher-forced scores

```text
S_box(r) = mean log p(r_q | prefix, r_<q), q in {x1,y1,x2,y2}
S_row(r) = mean log p(r_q | prefix, r_<q), q in owner-distinguishing
           description tokens plus {x1,y1,x2,y2}
```

Aliases first combine within physical owner:

```text
Q_o = logmeanexp(S(r) for aliases r of owner o)
```

For duplicate `d`, use the first applicable case:

1. If an uncovered valid owner has the same normalized description, compare
   `S_box(d)` with `logsumexp_o Q_o` over those owners.
2. Else if another uncovered valid owner exists, compare `S_row(d)` with the
   owner-normalized full-row alternative.
3. Else apply stable token unlikelihood at all four duplicate coordinate sites.
   Do not redirect toward `STOP`.

Cases one and two use

```text
softplus(m_dup + S(d) - logsumexp_o Q_o)
```

with selected candidates and log-sum-exp in fp32. Losses average within image
then across eligible images. This is an optimization signal, not a claim that
the teacher-forced candidate is the eventual greedy row.

### Rectangle-valid greedy gate

At a trusted positive row's `x2` site, valid next tokens are coordinate tokens
whose decoded value is strictly greater than the already emitted `x1`; at
`y2`, valid tokens are coordinate tokens strictly greater than `y1`. All other
vocabulary items are invalid at that site. The loss is

```text
relu(m_rect + max_invalid(logit) - max_valid(logit))
```

with argmax identities treated as stop-gradient selectors. The gate is applied
to positive Source-`G` replay and native-`H` candidate paths already present in
the R1 packs. It enforces a rectangle, not a token-identical GT coordinate.

### Shared target and replay

R1 retains A4's exact fixed-parameter global any-valid candidate objective and
the existing masked Source-body replay. The terminal token remains masked.
The active normalized families are:

```text
A4 any-valid target + Source replay + hierarchical row contrast
+ rectangle-valid gate
```

All coefficients are frozen in the resolved plan before the vertical slice.
No coefficient is tuned from exposure-one output.

## R2 preservation

At each panel exposure, accumulate R1's trainable-parameter gradient `g_R1` and
a separate owner/image-normalized Source-`G` coordinate CE watch gradient
`g_G` over the exact frozen Source prefixes. Before global clipping:

```text
if dot(g_R1, g_G) < 0:
    g = g_R1 - dot(g_R1,g_G) / (||g_G||^2 + eps) * g_G
else:
    g = g_R1
```

This makes the raw first-order direction non-adverse to the watch loss. The
receipt records the pre/post dot product, both norms, projection coefficient,
and zero/non-finite handling. The guarantee ends before clipping, AdamW
preconditioning, and nonlinear free-running decode; `G` retention remains an
empirical primary outcome.

R2 is world-size one. It MUST fail before optimizer mutation if complete R1
and watch gradients cannot be accumulated at one parameter state.

## Contrast and decision-owning outcome

| Arm | Source | Target | Duplicate | Geometry | Preservation | Dose |
| --- | --- | --- | --- | --- | --- | --- |
| Historical A4 | prior immutable run | any-valid | final-y2 UL | none | replay | exposure 2 |
| R1 | fresh | any-valid | hierarchical row | x2/y2 argmax | replay | 1,2 |
| R2 | fresh | any-valid | hierarchical row | x2/y2 argmax | replay + gradient projection | 1,2 |

The primary projection remains the tuple

```text
(H gained, G lost, M gained, unique owners,
 duplicates, malformed rows, cap stops, rows, generated tokens)
```

Never collapse it into one scalar. The most useful successor has more H gain
than historical A4@2 with fewer G losses and no increase in duplicate or
malformed burden, but this is a comparison description, not a hard promotion
threshold. Report image-level gained/retained/lost identities so owner exchange
cannot hide in pooled counts.

## Execution gates and stop rules

1. CPU tests must close ledger alignment, owner/alias normalization, all three
   row-contrast branches, coordinate token masks, selector gradients,
   projection algebra, packing, dry-run, and analyzer compatibility.
2. A real image-14038 vertical slice must complete one R1 update, checkpoint
   write/read, HF batch-one clean greedy decode, and analyzer projection.
3. Stop before the two-arm run on any non-finite loss/gradient, empty required
   family, pack/site mismatch, wrong trainable surface, checkpoint readback
   mismatch, HF surface mismatch, or unresolved OOM.
4. After the vertical passes, run R1 and R2 independently to cumulative
   exposures one and two, then evaluate the four immutable checkpoints.
5. Stop. Do not extend the dose, refresh targets online, or optimize again
   after inspecting same-batch outputs.

## Claim boundary

This is an adaptive, same-panel, overfit-only mechanism screen. It can show
that a native gradient recipe changes these exact clean-greedy outputs. It
cannot establish validation gain, transfer, population prevalence, production
safety, duplicate elimination, full-set mastery, or architecture sufficiency.

Implementation must favor the shortest conclusion-changing path. Do not add a
new trainer, general loss registry, distributed choreography, evidence
framework, or exhaustive review layer when the existing Human-13 path can be
extended locally.
