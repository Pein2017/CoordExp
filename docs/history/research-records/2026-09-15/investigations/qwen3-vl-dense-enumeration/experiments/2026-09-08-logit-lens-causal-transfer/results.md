---
title: Late readout is not late causal availability, and preference erasure is not decision transfer
description: Bidirectional state grafts on Image2299 and 12 additional Human13 training images support a context-suffix-dependent transfer framework with important scale and metric limits.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-08-logit-lens-causal-transfer
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

## Decision

**Lead-accepted:** a local causal-transfer effect, direction-specific limitations,
and a bounded functional framework. This is not a demonstrated natural
enumeration mechanism, held-out improvement, or a new trained model.

The [unit](unit.md) froze one Image2299 pilot and one fixed replication on the
remaining Human13 images. Both are complete. [Lead acceptance](lead-acceptance.json)
binds execution and independent reductions. [Framework](framework.md) separates
pre-result predictions from subsequent observations and unresolved alternatives.

The two most useful findings are:

1. **A coordinate token need not yet be top-one in a fixed-head lens for its
   intermediate state to have a donor-specific downstream effect.**
2. **Erasing the recipient's preference is not equivalent to transferring the
   donor's decision.** A two-token margin alone can obscure this distinction.

## Evidence surface and population

Same sorted-xy Source step-2444 and Human13 magnitude-only CE overfit adapters
as the parent diagnostic; common base, image, prompt, selected embeddings,
final norm and output head. Frozen overfit-generated prefixes are replayed on
both models with no KV cache. Interventions alter residuals at decoder outputs,
not weights, tokens, visual inputs, rotary positions or decoding policy.

- Pilot: Image2299, 12 coordinate decisions from first/middle/last generated rows.
- Replication: 12 additional **training** images, four coordinates of the middle
  generated row per image; 48 population sites and 47 distinct-coordinate-endpoint
  sites eligible for normalized transfer. Image14439 site1 has equal endpoint
  tokens; it remains in raw records but is excluded from the undefined ratio.
- Intended and executed replication images are identical:
  `1584,2685,4134,5001,6040,7511,10707,13348,13923,14038,14439,16228`.
- Total new causal study: 13 images, 60 population sites, 59 normalized-ratio
  sites. Pilot and replication are reported separately, not pooled as independent
  confirmation. All images belong to the overfit training population.

Define a and s as the two native checkpoint top-one tokens at the *same* prefix,
and `m = logit(a) - logit(s)`. Transfer R is the sum of patch-minus-receiver
margins divided by the sum of donor-minus-receiver margins. It is not accuracy,
probability improvement or a percentage of recovered objects. R is not clipped.
Replication tables first normalize within image, then weight images equally.

Current-state graft replaces one prediction position only; full-prefix graft
replaces all contextual states including visual/prompt/history/current states.
Four seeded random-delta controls match donor displacement norm at the current
position. Each current patch is isolated; future prediction sites are not
simultaneously patched. Block28 is a common-head algebraic identity control,
not scientific evidence that the final block discovered the answer.

## 1. Causal availability precedes top-one lens readout in the pilot

The parent Image2299 probe had zero actual coordinate tokens ranked first
through block24 (12 sites, 1-based layer numbering). Yet current-state graft
overfit→Source at block24 transfers R=0.664 of the endpoint margin difference,
versus 0.080 mean random control, and selects the overfit token in 8/12 sites.
All 12 coordinate-site margins move donorward. Full-prefix transfer is 0.740.

This is a direct counterexample to equating 'not top-one in Logit Lens' with
'cannot yet influence the downstream decision'. It is **not** proof that the
model naturally stores a complete correct box at block24, or that this state
is necessary for native behavior. We do not extrapolate the pilot's zero
early top-one counts to all replication images without measuring their lenses.

## 2. Transfer profile repeats across the remaining 12 images

Image-equal means, overfit donor→Source receiver:

| Block, 1-based | Current-state R | Full-prefix R | Random-delta R |
| --- | ---: | ---: | ---: |
| 16 | 0.0735 | 0.1919 | 0.0034 |
| 24 | 0.4864 | 0.5992 | 0.0347 |
| 27 | 0.7486 | 0.8605 | 0.0099 |

At block24, current-state R ranges 0.322–0.649 across images; the corresponding
random-seed means range 0.005–0.077. At block27 these ranges are 0.596–0.916
and -0.031–0.031. These are observed image ranges, not confidence intervals.

Full-vocabulary donor choice supports that this direction is more than generic
recipient disruption: at block27 the current graft selects the overfit token
in **46/47 eligible sites**, full-prefix graft in **47/47**, whereas random
controls do so in **8/188 site-seed trials**. The image-equal current donor-win
rate is 97.22%; its difference from 46/47 reflects equal-image weighting and
one image's three-site eligible denominator.

The current position accounts for more of the full-graft effect at later
tested depths, but context does not become universally irrelevant: the
overfit→Source full-minus-current R gap remains approximately 0.11 at blocks
24 and27. These three sampled depths do not establish smooth dynamics between
them or a unique sharply localized computational layer.

## 3. Reverse transfer exposes a metric trap and directional asymmetry

Source donor→overfit receiver, image-equal means:

| Block | Current R | Full R | Random R | Current donor top-one rate |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.4617 | 0.8827 | 0.1896 | 0.00% |
| 24 | 0.9598 | 0.9985 | 0.6223 | 32.64% |
| 27 | 1.0108 | 1.0207 | 0.5983 | 55.56% |

At block24, current graft selects the Source donor token in only 15/47 sites;
24/47 instead select a **third token**, despite aggregate R≈0.96. At block27
the corresponding counts are 26/47 and18/47. Random controls can also erase
much of the overfit endpoint preference while rarely restoring Source's token.

Thus 'R near one' means the chosen two-token margin has reached the donor
level, **not** that the donor's output distribution or decision has been
restored. This constrains the interpretation of both patching and any
four-corner decomposition built from this margin. It also argues against a
simple symmetric exchange of an identical self-contained decision variable.

At block16 reverse full-prefix R=0.883 substantially exceeds current R=0.462.
The donor context states affect the receiver suffix's decision. Because that
context includes visual and prompt positions, this is not specifically a
generated-history or remaining-owner-ledger effect.

## 4. A quantitative state–context–suffix framework

For a fixed prefix x, write the decision as `m = M_receiver_suffix(H_l(x))`.
H contains all prefix residual states, while the ordinary lens reads only
the current position through the fixed final norm/head.

Using the two native and two full-state hybrid margins, define m[d,r] where
d owns the upstream state and r owns the suffix. Then

```
Upstream = ((m[O,S]-m[S,S]) + (m[O,O]-m[S,O])) / 2
Suffix   = ((m[S,O]-m[S,S]) + (m[O,O]-m[O,S])) / 2
Interaction = m[O,O] - m[O,S] - m[S,O] + m[S,S]
```

Upstream + Suffix exactly equals the native checkpoint margin difference.
Interaction is an order-dependence term, not a third quantity to add to that
sum. This is an exact two-factor allocation for the defined hybrid experiment,
not a fitted predictive model or an intrinsic fraction of the LLM's reasoning.

Replication image-equal normalized results:

| Block | Upstream allocation | Suffix allocation | Interaction |
| --- | ---: | ---: | ---: |
| 16 | 0.5373 | 0.4627 | 0.6908 |
| 24 | 0.7989 | 0.2011 | 0.3993 |
| 27 | 0.9406 | 0.0594 | 0.1602 |

The additive, direction-independent account is poor at the earlier boundary:
the interaction is large and positive on all 12 images at block16. Later,
current-state donor choice becomes more transferable, with smaller but still
nonzero directional interaction. Splitting later also places more parameters
upstream, so an increasing allocation alone would not be a surprising
mechanism discovery; the current/full and donor-choice controls are essential.

### Radius and direction are a remaining alternative

Saved-state geometry replicates a systematic checkpoint difference:

| Block | Mean overfit/Source radius ratio | Mean matched-state cosine |
| --- | ---: | ---: |
| 16 | 0.9560 | 0.9778 |
| 24 | 0.7740 | 0.8252 |
| 27 | 0.7341 | 0.5314 |

Close vectors at block16 can still have substantial functional interaction;
cosine similarity is not functional interchangeability. Equal absolute delta
norm also does not equalize perturbation size relative to the receiver or the
resulting state's radius. Hence reverse random sensitivity is not yet proof
of a semantically fragile or worse model.

A useful candidate dynamical parametrization is `h_l = r_l u_l`. The exact
residual update determines both radius and direction; a smaller radius can
make a fixed orthogonal update produce a larger angular change. But the real
adapters can change the update itself, so this remains a hypothesis, not an
established explanation. [Framework](framework.md) gives the exact recurrence
and the separately proposed radius-only/direction-only discriminating grafts.

## Supported and unsupported conclusions

**Supported within this intervention surface:** late fixed-head readability
and earlier downstream causal leverage are different observables; current
state and context contributions differ by depth and direction; transferring
a donor choice differs from erasing the recipient's preference. The operational
state–context–suffix framework organizes and quantitatively checks these facts.

**Still unresolved:** radial scaling versus angular/state-content contribution,
native necessity versus cross-checkpoint compatibility, a true remaining-owner
representation, the cause of full-generation row-count/EOS differences, and
generalization outside the overfit panel. No training-time dynamics were
measured. No layer was ablated during natural full-image generation, no IoU
or unique-owner outcome was measured, and no model improvement is claimed.

**Next proposed discriminator, not executed:** radius-only versus direction-only
current-state grafts at the existing late boundary would address the largest
remaining interpretation confound before any claim about learned state
content. Natural continuation/owner evaluation would require a separate
behavioral estimand. No extra sweep or training follows automatically.

## Mechanics, artifacts and stop

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-causal-transfer`.

- Pilot: `run-v1/evaluation-v2/receipt.json` owns CPU-recovered acceptance of
  unchanged raw execution. The original terminal-failed receipt and exact
  launched source remain. A combined absolute/relative random-norm predicate
  was incorrectly summarized as absolute-only; no forward or scientific metric
  was affected and no GPU rerun occurred. Pilot self-check evidence is explicitly
  inferred from its fail-fast path, not individually cold remeasured.
- Replication: `stage-b-v1/receipt.json` binds 12 per-image receipts with persisted
  self-patch, native/text parity, final donor identity, position guards and hashes.
- Independent reductions: `pilot-analysis-v2`, `replication-analysis-v1`,
  `pilot-geometry-v1`, `replication-geometry-v1`. The earlier margin-only pilot
  reduction remains; v2 exposes donor/third-token outcomes.
- [Replication transfer figure](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-causal-transfer/replication-analysis-v1/transfer-framework.png)
  and [state geometry](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-causal-transfer/replication-geometry-v1/state-geometry.png).

Lead independently replayed seven focused tests, both raw reductions, exact
four-corner arithmetic, all 1,632 random norm checks, raw/per-image hashes and
exclusion counts; all passed. GPU0 was released. New causal execution took
319.86 + 1060.47 seconds, approximately 23 minutes total, within the 60-minute
bound. Replication peak allocation was 18.25 GB. No model training occurred.

The requested bounded series is closed: pilot, fixed replication and framework
synthesis. Code and records remain uncommitted in the isolated direction
worktree; unrelated work and other experiments are unchanged.
