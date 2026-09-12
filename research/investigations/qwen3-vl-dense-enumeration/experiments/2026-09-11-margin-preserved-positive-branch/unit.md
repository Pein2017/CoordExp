---
title: Positive-branch learning with worst-token margin preservation
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-11-margin-preserved-positive-branch
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: verified
updated: 2026-09-11
---

## Frozen question and authorization

**Closeout:** the paired smoke, one C32/cold and one 384+6 endpoint are complete
and lead-accepted technically. The [results](results.md) report improved margin
and reference preservation plus aggregate owner/burden gains, but two natural
positive-case losses and a worse fixed-history conditional drop count. The
no-weakening gate is not passed; no promotion or additional run is authorized.
The frozen question, formula and stop below remain unchanged.

From unchanged Stable50, does adding one-sided worst-token reference-margin
preservation to the exact positive32 objective reduce natural owner loss
without erasing the three learned loop repairs?

The user's autonomous overnight goal authorizes this one bounded learning
contrast. Root owns formula, semantics, acceptance and stop. This is one new
32-update candidate C, compared with the already accepted positive32 A and
Stable50 endpoints; no redundant full A run, event-risk arm, QP, architecture,
token change, GT edit, checkpoint promotion or coefficient sweep.

Evidence motivating this contrast: the accepted56-reference microscope finds
586 protected argmax flips and55/55 actual first divergences at protected
positions with exact current replay/natural next-token agreement. Only2/55
first divergences are epsilon1e-3 near-ties. Mean image KL is0.00583332, while
the56 protected images lose33/gain6 annotated owners. Most images without
owner loss also flip tokens: crossing is not by itself an owner-loss label.
The separate history cross establishes that entry selection and conditional
continuation quality can vary independently across the four fixed images.

## Exact objective

Use the identical3 verified complete-row positives, their verified conditional
successors and exact56 normal references/masks from positive32. Keep
`L_A = mean3(-sum logp(c)) + 10 mean3(KL_cond) + 100 mean56(KL_normal)`.
Do not protect the bad repeat at the positive fork, move successor prefixes,
append a new EOS label, or change any original KL mask.

At original protected position t of normal image i, let y_it be the literal
Stable50 token and `m0_it = z0(y_it) - max(v != y_it) z0(v)`, from the accepted
microscope's Stable50 replay. Eligible E_i contains exactly original-mask
positions whose source literal is argmax and `m0_it > 0.001`.
There are6030 eligible positions; the17 original-mask near-ties keep their
original KL but receive no new margin term. The9 excluded invalid-row tokens
of360573 remain excluded. No selection by owner loss, GT or A's first flip.

Freeze each target floor `tau_it = min(0.1, 0.5*m0_it)`. At current parameters:

```
m_it(theta) = z_theta(y_it | original prefix) - max(v != y_it) z_theta(v)
R_i(theta) = max(t in E_i) relu(tau_it - m_it(theta))
R(theta) = mean56 R_i(theta)
L_C(theta) = L_A(theta) + 10*R(theta)
```

The max is over all eligible tokens of an image, not only A's first flip and
not a token average. Empty E_i would contribute differentiable zero, but
all56 current cases are nonempty. Initial value/gradient are zero. The source
floor allows some erosion rather than preserving every original confidence.
The best-other token is recomputed from current logits, not frozen to A's
competitor. Use original selected-token positions; do not penalize aliases,
add GT neighborhoods, or give forced rows free credit.

Coefficient10 is chosen once, before C exists: at retained A, R=0.38309709,
so its weighted penalty3.8309709 is comparable to A's remaining positive NLL
3.51984596 (pre-step32), rather than an effectively zero squared-small-gap
term. This is scale calibration, not a guarantee of gradient balance.
Do not adjust the coefficient/floor after seeing C outcomes.

## Execution and acceptance

Same Stable50 payload/base/embedding/frontend, FP32 SDPA, language-only DoRA
588tensors/18,006,016scalars, fresh AdamW32steps, lr1e-5, betas(.9,.999),
eps1e-8, wd0, foreachFalse, clip1, deterministic eval-mode scorer and original
DDP normalization. Each rank owns7 normal references and all3 positive and
3 conditional items. The new normal contribution is `10*8/56 * R_i`; DDP
averaging yields the declared global mean. Reuse each normal replay's logits:
no additional model/image forwards, replay items, backwards or collectives.
Record the margin loss separately from raw KL and positive loss. After the
last update, replay the exact56 normal references once with no gradients on
the already loaded final model (7/rank, no new loads/caches). Preserve actual
post-update eligible argmax flips, floor violations, raw KL, R and active worst
positions. Pre-last-update statistics must not be labeled final-checkpoint
evidence. These56 extra scoring forwards do not change training or its
backward/collective counts; this closes the final-margin acceptance surface.

First CPU/golden preparation; then a root-granted paired2step real smoke
(weight0 unchanged-A parity, then weight10 candidate), each from Stable50.
Weight0 must match retained A's two-step adapter or demonstrate a concrete
numeric-route mismatch before training. Weight10 must show zero initial
margin force and a nonzero consumed margin gradient on a violated fixture /
actual post-first-step state, correct full-vocabulary competitor and exact
distributed denominator. Preserve all receipts and failures. Full C requires
a separate explicit launch grant after this vertical slice; no automatic retry.

Full32 bounds match A training topology:8workers/80initial reference forwards,
3510training-route model/image forwards including initial/final positive scoring,
plus56post-update normal-reference score forwards,3566total. The corresponding
two-step smoke total is356 rather than300. Record the56final-reference reads
separately; they add no optimizer steps, backwards or gradient synchronizations.
3328backwards,256gradient synchronizations,8loads, no sampling. At most1500s
per rank,24GiB CUDA allocated/reserved and24GiB RSS; compact margin table,
no extra full-vocabulary cache. Declare measured smoke cost/peaks before full.
Cold reload must reproduce the three saved final positive scores. No optimizer
or source-artifact overwrite; preserve prior accepted code/packets.

After technically valid C32/cold, execute exactly one384natural+6conditional
endpoint under the same prompt/greedy/RP1/cap3084/evaluator/geometry contract as
accepted A. Explicitly label C; do not fabricate a second A/B run. Endpoint
uses8independent workers,390continuations/image forwards/8loads, at most
15000generated/model-forward tokens perworker (120000 globally) and1500s/24GiB
CUDA/RSS perworker. Enforce the worker ceiling before another model forward;
do not merely check the global cap after spending it. Keep each original
rollout's3084/remaining-prefix budget unchanged. If a worker exhausts its
total budget, preserve completed records and an explicit incomplete/budget-stop
terminal, without a fabricated complete quality result or automatic rerun.
Both retained baselines plus worst-case conditional allowance fit this fixed
allocation (Stable50 maximum13620/rank). Bind exact
free-prefix/forced partition and all parser drops as before.

Report paired owner gains/losses/retention at50/60/80, F1, row starts, strict
repeats, geometry-invalid/other malformed drops, EOS/caps and the three h-only
and h+c results. Reference56 is training protection, development128 is an
already exposed screen, not an independent holdout. The three natural positive
cases have A TP50 counts12/11/16 (351017/417044/477415); do not hide their loss
behind improved reference averages. Report39654 separately without adding it
to the protection set or relabeling its broad branch as a correct instance.

## Decision and final stop

Encouraging preservation requires fewer eligible margin violations AND fewer
reference-owner losses than A's33 (with net reference coverage improved),
without weakening the three positive natural cases or their conditional
branches. A joint quality candidate additionally needs no net384-owner loss
relative to Stable50 and no worse repeat/invalid/cap burden; full paired metrics
remain visible even if these gates fail. No automatic checkpoint promotion.

If margins improve without owners, literal decision preservation is an
insufficient surrogate here. If preserving references erases positive repair,
the result is a gain/preservation tradeoff, not a proof of impossibility.
If reference repair succeeds but development still degrades, do not claim
generalization. Stronger constraint force may merely reduce effective positive
dose; the direct positive/conditional and natural-case checks discriminate it.

Stop after the one paired smoke, one C32/cold and one endpoint (or a concrete
technical/signal stop). No new coefficient, dose, reference refresh, rescue
branch or rollout sweep. Combine this with the two completed probes to close
the selected overnight portfolio with an evidence-ranked recommendation.
