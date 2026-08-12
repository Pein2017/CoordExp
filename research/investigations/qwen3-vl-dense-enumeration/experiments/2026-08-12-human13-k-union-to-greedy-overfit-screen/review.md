---
title: Human-13 K-Union-to-Greedy Overfit Screen Independent Review
type: investigation
role: review
authority: non_normative_research
unit_id: 2026-08-12-human13-k-union-to-greedy-overfit-screen
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-12
---

# Independent review and reconciliation

## Review scope

At the user's request, a read-only `claude-fable-5` reviewer at maximum
reasoning effort independently evaluated the proposed thirteen-image overfit
screen. The packet asked it to arbitrate prefix construction, suffix depth,
cross-entropy versus remaining-valid normalization and marginal objectives,
owner exchange, compute, and the smallest staged experiment.

The reviewer returned `PASS` for a hub-and-spoke eight-arm design. Its strongest
contributions were:

- separate K-hit `H` from K-miss `M` in Stage 1;
- use a Source-body replay control and a no-preservation ablation;
- compare H1, H2, and full-H suffix depth around one common hub;
- treat uniformly normalized per-owner CE as the same fixed-parameter gradient
  as packed H1 events, rather than paying for a duplicate arm;
- keep any-valid log-sum-exp as a falsification arm because it can concentrate
  on the already easiest owner;
- exclude Shapley/rollout marginal weighting from Stage 1 because it values
  sample diversity rather than one greedy path;
- stratify appended targets by geo-order-compatible `H_tail` versus
  out-of-order `H_mid`; and
- add a full-GT capacity control plus log-spaced exposure milestones.

## Conclusion-changing audit findings

The primary agent and two independent internal auditors did not accept the
reviewer's first draft verbatim.

### Terminal conflict

The draft replayed the Source terminal at the exact state where treatment
supervised another row. Those are contradictory labels. It also supervised a
terminal after full `H`, even though `M` may remain. The adopted design masks
terminal loss from all Stage-1 replay and treatment paths. Terminal supervision
is permitted only after a separately declared exhaustive target sequence.

### Set-mass dose

The draft repeated the any-valid mass objective `|H|` times to match owner
dose. That is not gradient-equivalent to uniform owner CE except in the
accidental case of uniform native row probabilities, and it turns one all-row
score into approximately quadratic physical work. The adopted design evaluates
the set objective once per image exposure and reports its effective owner count
instead of claiming matched owner credit.

### Gradient equivalence boundary

Uniform owner-mean CE equals separate H1 gradients only when every H1 segment
is accumulated at the same parameters before one optimizer step, with isolated
packing and explicit owner averaging. Sequential AdamW owner updates are not
equivalent. The unit now freezes the former semantics.

### Prefix and margin attribution

A sampled donor prefix changes history and its covered set, so it is an
algorithmic recipe rather than a pure prefix effect. The highest-likelihood
donor selector was removed. The pre-terminal margin was renamed a shared
continue gate because its first divergent row-start token is generally not
owner-specific.

### Missing capacity discriminator

All H-only arms could fail because the trainable DoRA/pipeline cannot memorize
the panel at all. A full-GT body-CE control was added outside the eight-arm
matrix.

## Final disposition

### K-hit, duplication, and argmax amendments

The user subsequently froze three conclusion-changing refinements:

- Stage 1 treats only native K-hit rows that already pass the owner and IoU
  matcher; GT-derived coordinate search is therefore a different support-
  expansion estimand and is excluded.
- Sampled discovery uses K=16 as four physical batches of four explicit `n=1`
  requests per image with repetition penalty `1.10`; clean greedy retains the
  Source-matched decode recipe.
- Every later class-agnostic prediction-to-prediction IoU greater than `0.95`
  is deleted from synthetic treatment prefixes and receives an unlikelihood
  event at its original raw prefix. No event is capped, while duplicate-free
  output remains a goal rather than a promotion gate.

A second read-only Fable-5/max audit returned `NARROW`: add a no-update
bottleneck census and one token-rank treatment, keep any-valid mass as a foil,
and move candidate-tree/on-policy refresh to Stage 2. Its first treatment draft
placed independent H1 singleton margins at the same exact prefix. The lead and
independent mathematical audit rejected that construction: at the first sibling
divergence it asks two distinct valid tokens to be strict global argmax and has
a permanent loss floor. Fable accepted the counterexample in a focused
follow-up. The adopted A8-prime arm is instead one coherent full-H chain, paired
with A1 and differing only in CE versus violation-only bottleneck loss.

The user then authorized this research-unit revision, one OpenSpec planning
change, one linked Superpowers implementation plan, and independent Sol/Fable
review. That authority does not include implementation or accelerator use; the
available eight GPUs are a later ceiling.

`NARROW` for the pre-OpenSpec scientific design. The user-requested independent
Sol-xhigh and Fable-xhigh review of the complete research-unit/OpenSpec/
Superpowers packet is still pending. Implementation and launch remain `HOLD`
until that packet converges and the user separately grants execution and
material-cost authority.

With the amendments above, the unit is a coherent same-panel optimization
screen. It remains incapable of validation, generalization, architecture
promotion, or production claims. The reviewers' dispositions are preserved as
advisory provenance; the amended [unit](unit.md) owns the executable scientific
meaning.

## Frozen-packet Sol/Fable audit and closure

The user-requested final audit used one identical seven-file packet for an
independent `gpt-5.6-sol/xhigh` semantic reviewer and
`claude-fable-5/xhigh` major-decision reviewer. Both were read-only and were
instructed to report only conclusion-changing P0/P1 findings. The first packet
ledger SHA-256 was
`6885134dbb7c0c83bd2d90f5b6f6ffab9a01040bd0e13c8f0dc86d681df600e9`.

Sol returned `HOLD` with no P0 and three P1 findings. Fable returned `NARROW`
with one P0 and one P1. The lead accepted and reconciled all five without
adding an arm or general framework:

1. freeze the language-DoRA-only AdamW, learning rate, schedule, clipping, and
   H/replay/duplicate family coefficients, remove subjective pre-16 behavior
   stops, and make any long-dose screen a fresh Source/optimizer run;
2. insert a separately authorized full-panel Source-plus-208-K discovery,
   canonical-manifest freeze, and no-update census before any target-dependent
   update;
3. rename A0 to the shared no-H background control because it contains both
   Source replay and duplicate correction;
4. apply the user's chronological `IoU>0.95` duplicate rule before owner
   matching, exclude every later duplicate from positive owner/support/replay/
   target/candidate sets and from final owner credit, and preserve it only as a
   raw-state unlikelihood event; and
5. compute duplicate unlikelihood as the stable fp32 logit identity
   `softplus(z_target-logsumexp(z_non_target))`, with a saturated-margin finite
   regression.

The corrected packet ledger SHA-256 is
`d0eaf04d58910b152ef2c9b55bf7049f05775c509f7a1ba5f95d81c14ece6830`.
Each original reviewer then performed exactly one focused correction re-review.
Sol returned `PASS_FOR_USER_APPROVAL`; Fable independently returned
`PASS_FOR_USER_APPROVAL` and confirmed that all five findings were closed with
no new P0/P1.

### Final audit disposition

`PASS_FOR_USER_APPROVAL` for the planning packet. `HOLD` for implementation,
model execution, and GPU launch until the user explicitly authorizes the next
gate. The pass certifies only the corrected seven-file planning target; it does
not certify future code or any runtime artifact.
