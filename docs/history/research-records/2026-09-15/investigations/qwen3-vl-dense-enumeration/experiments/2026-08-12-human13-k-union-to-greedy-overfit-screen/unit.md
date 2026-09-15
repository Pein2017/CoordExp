---
title: Human-13 K-Union-to-Greedy Overfit Screen
description: A same-panel, native-transformer overfit laboratory that separates K-hit consolidation, coherent suffix depth, prefix provenance, probability mass, and token-rank supervision without making validation or generalization claims.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: implemented_and_partially_executed
unit_id: 2026-08-12-human13-k-union-to-greedy-overfit-screen
topic: qwen3-vl-dense-enumeration
status: complete_bounded_same_panel
evidence_status: original_prompt_clean_greedy_matrix_complete
updated: 2026-08-12
---

# Human-13 K-Union-to-Greedy Overfit Screen

## Decision and outcome

The authorized implementation and bounded matrix have completed. The current
scientific disposition is owned by [results.md](results.md): native low-dose
updates can improve same-panel greedy owner coverage, but no executed arm
compiles the full K-hit union safely. The useful region is one panel exposure;
later exposures cause owner exchange and severe output growth. A4, A6, and
A8-prime are mechanically absent and remain unknown, not negative results.

This unit asks whether repeated natural sampling has already exposed useful
physical-owner support that can be consolidated into one ordinary clean-greedy
completion by updating only the existing language-tower Weight-Decomposed
Low-Rank Adaptation payload. The vision tower and multimodal aligner remain
frozen. No external owner bridge, detector, inference-time ledger, or new
gradient path is part of this screen.

The thirteen images are deliberately used as an **overfit-only optimization
laboratory**. Their labels, sampled trajectories, local diagnostics, and
post-update greedy outputs may all influence arm selection and later training
on the same images. There is no validation split. Consequently, the strongest
possible result is that a recipe can fit these exact images under the declared
runtime; no result from this unit estimates generalization, prevalence, or
production value.

The decision-owning outcome is one unforced completion from the original
prompt under the frozen clean-greedy decode recipe. Teacher-forced likelihood,
fixed-prefix margins, training loss, and sampled recall are diagnostics only.

This document retains the approved scientific design. Execution provenance,
the complete milestone table, mechanical failures, interpretation, and next
decision are in [results.md](results.md). Checkpoint promotion, a long run,
support-expansion successor, stable-spec sync, and production changes remain
unauthorized.

## Originating intent and approved scope

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Use the exact thirteen manually refined dense images for repeated fitting | User decision, 2026-08-12 | scientific invariant | cohort and claim boundary | approved for overfit-only use |
| Do not reserve validation images inside this panel | User decision, 2026-08-12 | scientific invariant | evaluation and checkpoint selection | approved; no transfer claim |
| Explore prefix, suffix, and loss construction boldly | User decision, 2026-08-12 | scientific invariant | compared treatments | approved for design |
| Begin with owners present in the fixed K-rollout union but absent from Source greedy | User decision, 2026-08-12 | scientific invariant | primary target stratum | approved |
| Keep K-miss owners separate before deciding support expansion | User decision, 2026-08-12 | scientific invariant | target strata and successor route | approved |
| Use K-batched sampling and no-padding packed training | User preference, 2026-08-12 | conservative design choice | implementation and compute | proposed default |
| Submit sampled decode in physical batches of four and use sampling repetition penalty `1.10` | User decision, 2026-08-12 | scientific invariant | frozen `K` ledger | approved |
| Prefer the shortest conclusion-bearing implementation and avoid speculative framework, redundant receipts, or ceremonial audits | User decision, 2026-08-12 | conservative design choice | implementation and evidence cost | approved |
| Prepare the research unit, OpenSpec change, Superpowers plan, and independent Sol/Fable review | User decision, 2026-08-12 | conservative design choice | planning authority | approved |
| Use at most eight GPUs after a later launch decision | User decision, 2026-08-12 | material-cost boundary | execution topology | approved as ceiling; not launch authority |
| Train or launch the planned arms | not yet granted | material-cost authority | implementation and GPU execution | needs user decision |

## Frozen substrate

### Model

Use the plain native checkpoint `S`, not the permanent-owner-bridge
checkpoint:

```text
/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/
2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444
```

Its source identities and wrapper are owned by the
[static/dynamic owner-interface crossover](../2026-08-05-static-dynamic-owner-interface-crossover/unit.md):

- base model: `Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`;
- stable x-then-y ordering, `geo_sorted_xy`;
- native `object_box_closed` wrapper;
- language-tower Weight-Decomposed Low-Rank Adaptation trainable;
- vision tower and multimodal aligner frozen; and
- fresh AdamW state for every independent arm.

Every arm must begin from byte-identical Source adapter and embedding payloads.
No arm inherits weights or optimizer moments from another arm.

### Panel

Use the derived x-then-y thirteen-image panel:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-05-static-dynamic-owner-interface-crossover/inputs/
human-refined-13.geo_sorted_xy.coord.jsonl
```

Its SHA-256 is
`5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
The source panel contains 392 manually calibrated owners: 346 in the legacy
twelve images and 46 in image `2299`. Every report must show the legacy-twelve
slice and image `2299` separately before a pooled thirteen-image table, as
required by the
[panel-admission contract](../2026-08-04-sorted-prospective-13-image-panel-admission/unit.md).

The twelve legacy images are currently rejected by the general StateBank blind
cohort guard in `src/rollout_calibration/state_bank.py`. Implementation must add
a hash-bound, experiment-local, overfit-only opt-in; it must not delete, weaken,
or silently bypass the default guard.

## Owner sets and immutable ledgers

For image `i`, freeze the following sets before any optimizer update:

```text
G_i       Source clean-greedy owners
U_i^K     trusted owner union of G_i and the fixed K sampled trajectories
H_i       U_i^K minus G_i: K-hit greedy misses
M_i       GT_i minus U_i^K: K-miss owners
F_i       H_i union M_i: all Source-greedy misses under this fixed recipe
```

`H` is the primary Stage-1 target. `M` supplies no gradient in Stage 1 and is
never a negative; incidental `M` recovery is reported separately. Zero hits in
K trajectories means only "unobserved under this fixed sampling recipe", not
visually absent or unlearnable.

Duplicate classification precedes owner assignment. Parse each raw Source and
K trajectory, scan its complete rows in generation order, retain the earliest
row, and mark every later row whose class-agnostic prediction-to-prediction IoU
with an already retained row is greater than `0.95` as a duplicate. By the
user's frozen rule, a later duplicate is ineligible for one-to-one owner
assignment, `G/U/H` support, Source replay, positive target-row selection, and
A4 candidates even if a cardinality-first matcher could otherwise assign it to
a distinct dense GT owner. It remains in the raw provenance and contributes
duplicate unlikelihood. The one-to-one same-category owner matcher runs only on
the retained non-duplicate complete rows. Thus no row can be both a trusted
owner positive and a duplicate negative.

The ledger must bind checkpoint, image and prompt hashes, tokenizer and wrapper,
decode backend and version, K, seeds, temperature, top-p, repetition penalty,
maximum new tokens, physical request batch and request-local seed, generated
token identifiers, parser status, row index, category, box, matcher version,
owner assignment, and truncation status.

Proposed discovery recipe, to be frozen before launch:

```text
K                       16 trajectories per image
sampled seeds           21001 through 21016
physical request batch  4 independent n=1 requests
temperature             0.4
top-p                   0.95
repetition penalty      1.10
maximum new tokens      512
```

For each image, dispatch four successive `engine.generate` calls containing the
explicit seed groups `21001..21004`, `21005..21008`, `21009..21012`, and
`21013..21016`. Every request has `n=1` and owns its own seed; do not substitute
one request with `n=4` or infer scientific seed identity from vLLM child-index
expansion. The compact collector receipt records observed cache counters when
the runtime exposes them, but missing cache telemetry does not block the
scientific ledger and possible reuse is never reported as observed reuse.

The sampling-only repetition penalty `1.10` changes the frozen discovery
distribution and therefore belongs to the `K` identity. It does not change the
deployment readout: clean-greedy evaluation remains the Source-matched,
batch-size-one HF recipe with repetition penalty `1.0`, because physical batch
shape and decode processors have previously changed first actions in this model
family.

Owner assignment over the retained non-duplicate rows uses one-to-one,
same-category matching at Intersection over Union at least `0.50`, plus the
frozen ambiguity policy. A complete, retained, strictly matched row remains
usable as positive support even if later tokens in its trajectory hit the
generation cap; a K-miss statement may use only the predeclared complete
trajectory set and must expose truncation rather than treating it as a miss.

For every `H` owner, select one native target-row occurrence before outcomes
are seen. Use a deterministic rule based on strict geometry quality, then seed
and row index; do not select by train-time likelihood. The selected exact row
tokens are reused across prefix arms. Canonical GT rows are reserved for the
full-GT capacity control and later `M` support-expansion arms.

Each selected `H` row has already been emitted by the native model and passed
the declared owner and IoU matcher. Stage 1 therefore tests discovery-to-greedy
consolidation under observed native support; it does not teach a new GT box or
claim that localization is invariant to a changed prefix. A GT-derived IoU
coordinate oracle would change the estimand and is excluded.

## Prefix, duplication, and ordering terminology

- **Raw Source stop prefix (`P_raw`)**: the exact Source clean-greedy tokens up
  to, but excluding, its natural terminal token. It owns provenance and every
  duplicate-unlikelihood decision state; it is never called deduplicated.
- **Clean treatment stop prefix (`P_clean`)**: a synthetic context made by
  scanning complete rows in `P_raw` from left to right, retaining the first row
  and deleting every later complete row whose class-agnostic prediction-to-
  prediction IoU with an already retained row is greater than `0.95`. Exact
  retained token spans are concatenated without decode/re-tokenize. All `P_stop`
  treatment arms below use `P_clean`.
- **Sampled donor prefix (`P_K`)**: the sampled-trajectory tokens up to, but
  excluding, the selected target owner's first strict occurrence, with the same
  sequential duplicate removal applied before a treatment suffix is appended.
- **Source-body replay**: positive replay of trusted Source owner-row tokens;
  duplicate rows, the Source terminal token, and any unmatched, malformed, or
  unresolved row receive no replay loss.

Unmatched non-duplicate complete rows remain in treatment context but receive
no direct loss. Malformed and geometry-invalid spans also remain context-only
in this first screen. The frozen `G/U/H/M/F` sets are computed once from the
pre-update natural trajectories after the mandatory chronological duplicate
exclusion above; they are never recomputed from a synthetic treatment prefix.

Every observed later duplicate in Source greedy and all sixteen artifact-valid
K trajectories receives one unlikelihood event at its original raw trajectory
state; there is no event cap, sampling, or down-weighting by chronology.
To avoid penalizing shared category/schema tokens, the selected negative site is
the final coordinate token that closes the duplicate box:

```text
q_e      = logsumexp(z_(v != d_e))
L_dup(e) = softplus(z_(d_e) - q_e)
         = -log(1 - p(d_e | raw prefix and duplicate row before d_e))
```

The logit-difference form is required so a saturated target remains finite in
fp32; computing `1 - softmax(z)[d_e]` directly is forbidden. Every frozen event
is consumed, and image-balanced normalization prevents one duplicate-heavy
image from silently owning the whole panel update. Duplicate-free output is an
optimization goal and reported readout, not a hard promotion metric. Distinct
GT owners in this panel can overlap materially below `0.95`, so GT separation
does not make the pred-pred rule safe by itself. The mandatory exclusion before
matching is what prevents contradictory positive and negative labels; this
panel-local rule is not promoted into a general duplicate policy.

`P_clean` replaces the earlier ambiguous treatment name `P0`, which other
repository units use for prompt-only states. `P_raw` remains available whenever
an exact realized history is scientifically required.

For the stable x-then-y target order, define:

- `H_tail`: an `H` owner whose canonical sort key does not move backward past
  any matched Source-greedy owner already emitted; and
- `H_mid`: an `H` owner that would need an out-of-order recovery if appended
  after `P_stop`.

Also record the integer sort-backtrack distance. All `P_stop` results are
reported by this stratum. An `H_mid` deficit does not by itself identify a
visual-representation failure; it may be an ordering or prefix mismatch.

## Shared treatment semantics

One **panel exposure** means that every eligible image is presented once. For
one image exposure, every declared target owner is scored exactly once and all
owner losses are accumulated before one optimizer update. Packed segments must
be attention-isolated, reset their position/MRoPE semantics per segment, use no
padding, and perform no optimizer step inside the accumulation.

This distinction is load-bearing. At fixed parameters,

```text
mean_a gradient(L_a) = gradient(mean_a L_a)
```

so uniformly weighted H1 events and an owner-mean multi-positive objective are
the same gradient only when all owners enter the same update. Sequential AdamW
updates are not equivalent because parameters, moments, clipping, weight decay,
and the learning-rate schedule change between owners.

Every correction arm uses the same Source-body replay family unless the arm
explicitly ablates it. The terminal token is masked in all Stage-1 treatment
and replay paths. Supervising `P_stop -> terminal` while also supervising
`P_stop -> H row` is a contradictory label. Likewise, reaching all `H` owners
does not license a positive terminal while any `M` owner remains.

Every Stage-1 arm, including the no-`H` control, also consumes the same frozen
duplicate-event family and `L_dup` coefficient. The full-GT capacity control is
outside this treatment family and remains plain masked body CE so that it tests
trainable capacity rather than duplicate correction.

### Frozen update contract

The initial screen freezes one optimizer and coefficient table before any
outcome is observed:

```text
trainable parameters       language-tower DoRA payload only
frozen parameters          vision tower, multimodal aligner, token embeddings,
                           base language weights
optimizer                  torch AdamW
adapter learning rate      1.0e-5
betas / epsilon            (0.9, 0.999) / 1.0e-8
weight decay               0.0
scheduler                  cosine_with_warmup, warmup_steps=0
initial scheduler horizon  16 panel updates
global gradient clip       1.0

family coefficients        H objective  Source replay  duplicate UL
A0 shared no-H background        0.0          1.0           1.0
A1/A3/A4/A6/A8-prime             1.0          1.0           1.0
A7                                1.0          0.0           1.0
full-GT capacity                  1.0          0.0           0.0
```

Each nonzero family first applies its declared owner/image normalization, then
the three normalized family scalars are multiplied by these coefficients and
summed. No arm may rescale active families to keep their sum constant. A0 is
therefore the **shared no-H background control**—Source replay plus the common
duplicate correction—not a replay-only control. A3 versus A7 owns the replay
ablation.

The first authorized matrix is a fixed sixteen-update run. A later 32/64/100
screen, if separately authorized, restarts from byte-identical Source and fresh
AdamW with its own predeclared 100-update cosine schedule; it does not resume
the short run or pool shared milestones as the same optimizer trajectory.

### Packing and optimizer topology

Each logical candidate is a complete multimodal segment with isolated causal
attention and per-segment Qwen multimodal rotary-position reset. Use one
physical batch row, no padding, varlen FlashAttention-2, and the existing
`global_max_length=12000`. Deterministically order independent segments by
descending encoded length with image/event identity as the tie-breaker, then
first-fit them into packs. This changes execution order but not the complete
panel objective because all packs are evaluated at the same parameters before
one optimizer update.

A1, A8-prime, and the full-GT control use one coherent segment per image. A3,
A7, A6, Source replay, and duplicate events may share packs across images. A4's
complete candidate set for one image is atomic: all of its row scores must
remain in one forward graph. A CPU length preflight fails that arm if an atomic
image group or any coherent segment exceeds 12,000 tokens; this screen does not
grow a speculative multi-pass candidate engine.

The complete panel denominator is known before backward. Micro-pack losses are
scaled by that global owner/image/event denominator, accumulated with no
optimizer mutation between packs, and followed by exactly one AdamW step per
panel exposure. Packing saves padding and launches; it does not claim image-
encoder, prefix-KV, or forward-FLOP reuse, because every isolated segment still
contains its own prompt, image, and prefix.

When later authorized, each training arm runs as an independent one-rank
Accelerate process on one GPU. At most eight arms run concurrently. This uses
the available cards for independent scientific work instead of introducing
unequal-rank DDP packing, cross-rank denominators, or new collective
choreography. A one-image vertical slice must confirm the current 12,000-token
bound before any matrix launch; absent a measured OOM, no speculative fallback
packing mode is designed.

## Loss geometries

For target owner `a`, exact target row `y^a`, and context `c`, let

```text
s_a(c) = sum_t log p(y^a_t | c, y^a_<t)
u_CE(a,c) = -s_a(c) / length(y^a)
```

The owner mean, rather than a raw token mean over the whole physical pack,
prevents row length and scene density from silently changing owner credit.

### Uniform all-owner row cross-entropy

```text
L_all(c) = mean_(a in H_i) u_CE(a,c)
```

This is the precise implementation of uniformly normalized remaining-valid
owner supervision. It does not require a separately named "multi-positive"
arm when its H1 events are accumulated in the same update.

### Coherent full-H token bottleneck margin

Fix one complete A1 residual chain before any train-time score is observed:

```text
P_clean -> y^(pi_1) -> ... -> y^(pi_|H|)
```

where `pi` is the same stable residual order and the exact same selected native
rows used by A1. For every non-terminal target token `y_t` in that one coherent
chain, let

```text
r_t = z(y_t) - max_(v != y_t) z(v)
L_bottleneck = mean_owner mean_token relu(m - r_t)
```

The competitor identity is a stop-gradient argmax selection; objective math is
fp32. Row wrappers and row terminators participate, while the final chat
terminal remains masked. There is no CE continuation term. A satisfied site has
zero direct gradient, so this arm tests rank crossing rather than continuing to
concentrate probability on a selected row.

The positive margin `m` is frozen before any update from the no-update
cross-surface census: take the maximum observed absolute target-versus-
competitor margin drift between the packed training surface and the declared
HF clean-greedy scoring surface, then add `1e-4`. If the surfaces cannot provide
aligned finite scores or the required `m` exceeds `0.5`, the bottleneck arm is
mechanically blocked rather than silently retuned. This calibration is a
numerical-stability allowance, not an outcome-tuned hyperparameter.

Independent H1 bottleneck losses at the same `P_clean` are forbidden. At the
first token where two valid H rows diverge, they would require mutually
exclusive siblings to be strict global argmax and create a permanent loss
floor. A coherent full-H chain has one target at every reached state and avoids
that contradiction.

### Deferred continue-gate diagnostic

At `P_stop`, compare the shared valid row-start token with the Source terminal
token once per image:

```text
L_gate = softplus(m - [z(valid_row_start) - z(terminal)])
         + lambda * mean_(a in H_i) continuation_CE(a)
```

This is a saturating continue-versus-stop gate, not an owner-specific routing
margin. Repeating the shared gate once per owner would silently multiply its
dose. It is retained as a Stage-2 diagnostic definition but is not an
independent Stage-1 arm: the coherent bottleneck chain already scores each
distinct row boundary once.

### Any-valid native row mass

After exact-token deduplication, complete candidate rows must contain an
explicit row terminator so they form disjoint, prefix-free actions. Then:

```text
L_any(c) = -log sum_(a in H_i) exp(s_a(c))
```

If the row strings are not prefix-free, compute union probability through a
token trie; do not call a length-normalized candidate energy a probability.
This objective is evaluated once per image exposure, not repeated `|H_i|`
times. Its gradient weights owners by their current native row probability and
therefore may concentrate on the already easiest owner. Report

```text
w_a = softmax_a(s_a)
effective_owner_count = 1 / sum_a w_a^2
```

and the owner-gain Gini coefficient. `L_any` is a registered falsification arm,
not an assumed union objective.

K-rollout rarity, leave-one-out union contribution, and Shapley weighting are
not Stage-1 objectives. They value complementarity among sampled trajectories,
whereas deployment asks one greedy trajectory to contain the union. They may
become a budget-limited tie-breaker later; rollout frequency and teacher-forced
deficit are recorded as covariates now.

## Stage 0.25: model-derived discovery and freeze

CPU dry-run materialization does not create the scientific ledger. After a
separate model/GPU authorization and before any optimizer update, execute the
Source-matched HF batch-size-one clean-greedy decode on all thirteen images and
the exact 208 K requests described above. Finalize duplicate classifications,
retained-row owner assignments, `G/U/H/M/F`, selected native rows, residual
order, donor eligibility, raw/clean prefixes, and duplicate events into one
canonical full-panel manifest.

A partial one-image or fixture manifest may test mechanics but cannot select
the decision-bearing vertical-slice image, decide A6 applicability, calibrate
A8-prime, or substitute for this gate. The training runner fails closed on a
partial manifest.

## Stage 0.5: no-update bottleneck census

After the K ledger, target rows, prefixes, duplicate events, and residual order
are frozen, run one no-update teacher-forced census before any arm trains. Build
an exact-token trie over the selected native H rows at `P_clean`; at every node
report whether the actual top-1 token is one of the viable native children and
the strongest viable-child margin. Separately score every site in the fixed A1
and A8-prime coherent chain and report:

- first non-argmax token and minimum strict margin;
- boundary, schema, description, coordinate, and row-terminator site counts;
- target/competitor identities, tie count, and packed-versus-HF margin drift;
- the trie-projected native row if the current top-1 remains inside the tree;
  and
- STOP, confirmed duplicate, invalid, or unclassified competitor status.

The census is diagnostic only. It cannot change selected target rows, their
order, `G/H/M`, or another arm's owner weights. A8-prime is skipped as a true
no-op if no coherent-chain site violates its frozen margin. The trie is not a
training candidate tree and tokens outside the frozen native leaves are not
declared invalid.

## Stage 0: anchors and capacity control

These controls are outside the treatment arms:

| Control | Update | Purpose |
| --- | --- | --- |
| Frozen Source | none | Exact behavior anchor. |
| Full-GT body cross-entropy | prompt to all 392 per-image canonical GT rows in stable x-then-y order; terminal masked by default | Capacity/optimization upper bound: can this DoRA and pipeline fit the thirteen image-to-owner sequences at all? |

The full-GT control may supervise the final terminal only after the user
explicitly declares the panel exhaustive for this purpose. Without this
control, failure of all H-only arms cannot distinguish a bad consolidation
algorithm from a pipeline or trainable-capacity failure.

## Stage 1: narrowed hub-and-spoke screen

The hub is `A3`. Every arm restarts from Source with a fresh optimizer. Unless
shown otherwise, target scope is `H` only, terminal loss is masked, Source-body
replay is present, and the same per-image owner multiset and weights are used.

| Arm | Prefix | Suffix topology | Supervision | Single question |
| --- | --- | --- | --- | --- |
| A0 shared no-H background control | native Source trajectory | trusted Source rows only plus shared duplicate events at their raw states | owner-mean row CE plus `L_dup`, terminal masked | How much does the common replay-plus-duplicate background move greedy behavior without `H` treatment? |
| A1 full-H chain CE | `P_clean` | every `H` row in one frozen residual order | `L_all`, no terminal | Does an explicit coherent residual chain provide a same-panel consolidation existence proof? |
| A3 uniform H1 hub | `P_clean` | every `H` owner as an independent H1 segment | `L_all` | Does equal owner credit at the natural stop boundary consolidate existing support? |
| A4 any-valid mass | `P_clean` | all `H` rows as atomic candidate actions | one `L_any` per image exposure | Does aggregate valid-row mass outperform equal owner credit, or hide owner starvation? |
| A6 natural donor H1 | paired `P_K` | the same selected owner row used by A3 | `L_all` | Does model-native sampled history transfer better, especially for `H_mid` owners? |
| A7 no-preservation H1 | `P_clean` | identical to A3 | `L_all`, no Source-body replay | Does Source-body replay carry retention, or merely dilute the target update? |
| A8-prime full-H bottleneck | `P_clean` | exactly the A1 coherent residual chain | `L_bottleneck`, no CE and no terminal | Does crossing only the chain's token-rank bottlenecks transfer to clean greedy better than chain CE? |

A6 is an algorithmic donor-prefix arm, not a pure prefix causal effect: its
covered set and preceding route differ from `P_stop`. Its target row is held
identical to A3, the donor rule is outcome-independent, and all prior row
identities are reported. A "highest teacher-forced likelihood donor" is not
used because it would select easier examples and confound the comparison.
A6 is executed only when the frozen pre-update ledger contains at least one
`H_mid` owner with an eligible native donor prefix. Otherwise it is recorded as
not applicable rather than replaced by a weaker case.

A2 continue-gate and A5 role-balanced H2 are deferred. A2 is largely contained
by the once-per-boundary A8-prime chain, while A5 becomes decision-bearing only
after H1 produces a partial clean-greedy gain. Native candidate-tree rank
training, online prefix refresh, and GT-IoU search are also absent from Stage 1.

Images with `H_i` empty contribute only the shared replay family and are
excluded from target-loss denominators; their count remains visible.

## Milestones, dose, and compute

Record clean-greedy outputs after panel exposures:

```text
0, 1, 2, 4, 8, 16, 32, 64, 100
```

All applicable Stage-1 arms first run through exposure 16 unless an exact
mechanical stop fires. Every Pareto-nondominated or mechanistically distinct
recipe may receive a separately authorized fresh 100-update run, but the long
run restarts Source and AdamW and uses its own 100-update schedule. It is not an
optimizer continuation of the short run. This is adaptive same-panel
screening, not a statistical winner selection. Every executed milestone and
run family remains in the result; no best-only table is allowed.

Do not equate owner dose with compute. Each arm receipt records:

- eligible images and target owners;
- per-owner positive weight and cumulative owner exposure;
- selected loss-token count and physical packed-token count;
- candidate-row forwards, optimizer updates, and gradient accumulation;
- raw and clipped gradient norm, parameter-delta norm, and AdamW moments;
- K-generation and greedy-evaluation token counts;
- GPU seconds, wall time, and peak allocated memory; and
- terminal, malformed, truncation, and non-finite events.

Run one real image through source selection, packing, backward, checkpoint
write-read, clean greedy, matching, and the final ledger before estimating or
authorizing the complete matrix. The timing receipt from that vertical slice,
not a speculative GPU-hour number, owns the launch budget.

## Primary readout

Apply the same chronological class-agnostic pred-pred IoU `>0.95` exclusion to
every clean-greedy output before final owner assignment. Later duplicate rows
count in duplicate/output burden and receive no owner credit; the declared
cardinality-first one-to-one matcher runs on retained non-duplicate rows only.
This keeps training support, Source/H ledgers, and outcome credit aligned.

For every image and milestone, freeze exact identities and report separately:

```text
H gained                       primary consolidation gain
G retained and lost            owner exchange / preservation
M gained incidentally           support-expansion spillover
final unique trusted owner set
prediction rows and tokens
coverage at fixed row budgets and at natural stop
duplicate, unmatched, malformed, invalid, cap-stop, and repetition burden
common-owner IoU and coordinate error
```

An image exhibits **safe in-panel consolidation** when it gains at least one
`H` owner, loses no `G` owner, and creates no new malformed or cap-stop failure.
Full `H` mastery requires `G_i union H_i` to be present in the final greedy set.
Report the number of images satisfying each predicate; a pooled net gain cannot
hide owner exchange.

The primary comparison is the Pareto surface over `H` gain, `G` loss, burden,
and compute relative to Frozen Source and A0. Fixed-prefix margins and
teacher-forced deficits explain outcomes but cannot promote an arm whose
original-prompt clean greedy output does not improve.

## Stage 2 branches

Stage 2 starts from a fresh Source checkpoint and fresh optimizer for every
branch. It never continues an A-arm checkpoint merely because that checkpoint
looked attractive.

1. **H1 or the fixed chain moves a local margin but greedy gains at most one
   owner.** Compare the deferred role-balanced H2 and the executed full-H arms,
   then run an iterative frontier-refresh
   arm: decode the current greedy policy, version a new `P_stop`, apply a
   bounded update, and use the next panel pass as the next training state. Keep
   `H` fixed; only prefix versions change. Stop after two no-gain refreshes, a
   declared update cap, or mastery.
2. **Order is decision-bearing.** Under the winning loss, compare the same
   full-H owner multiset and dose using stable geo order and a K-native
   candidate-tree projection. The projection may choose only frozen strict-
   matching native H leaves, selects at each synthetic state the remaining
   viable child with the best token margin, and never calls an unobserved
   coordinate token invalid. This is a separately authorized search-and-distill
   algorithm, not ordinary CE or a GT-IoU oracle. Random permutations are added
   only if these two materially differ; they are not a ritual baseline.
3. **Owner exchange dominates.** Compare Source-body replay weight, a frozen
   Source logit anchor, and their combination. Any gradient projection remains
   native-parameter training and must expose the watch-gradient set; it is not
   inferred safe from target loss alone.
4. **Natural donor prefixes win for `H_mid`.** Expand to multiple frozen donor
   prefixes per owner and a mixed `P_stop`/`P_K` recipe. Keep target row tokens
   paired and report prior covered sets.
5. **H-only consolidation succeeds.** Open a separate support-expansion
   contrast using canonical GT rows: `M` only versus `H union M`, with the same
   terminal masking and separate `H`, `M`, and `G` ledgers. This is a new
   support question, not a continuation of the H-only claim.
6. **All H-only arms fail.** If the full-GT capacity control also fails, stop
   and diagnose trainable capacity, objective plumbing, or optimizer dose. If
   full-GT fits but H-only fails, the evidence localizes a proxy-to-greedy or
   prefix/suffix construction failure and motivates the frontier-refresh and
   greedy-projection branches, not an external bridge by default.

Rollout-marginal or Shapley weighting may be tested only after a later budget
constraint creates a real owner-selection problem. A K-union reward assigned
to every trajectory is excluded: tied group rewards give no within-group
advantage, and trajectory complementarity is not one-path greedy consolidation.

## Falsification and stop rules

Stop an arm and preserve its partial receipt before exposure 16 only when one
of the following exact mechanical predicates occurs:

- step-zero output differs from the frozen Source under the same runtime;
- non-finite loss, gradient, parameter, or checkpoint payload;
- row parsing or terminal semantics differ from the declared masks;
- the exact owner multiset or Source restart identity drifts;
- an atomic/coherent segment exceeds 12,000 tokens or the declared runtime OOMs;
- the applied-step count, checkpoint write-read, or clean-greedy/analyzer
  execution contract fails; or
- the vertical slice cannot reproduce packing, write-read, and clean-greedy
  evaluation through the final analyzer.

Cap stops, malformed rows, duplication, unmatched rows, repetition, and output
growth are mandatory readouts through exposure 16, not subjective early-stop
phrases. They may prevent promotion or motivate a later user decision, but
they do not silently change executed dose in this first screen.

Stop A8-prime when every coherent-chain site meets its frozen margin at two
successive milestones but original-prompt clean greedy gains no `H` owner. That
is evidence for a fixed-prefix-to-realized-trajectory transfer failure and
routes to Stage-2 frontier refresh; it is not repaired by increasing the
margin or adding a GT coordinate oracle.

Failure of one loss is evidence only against that exact prefix, suffix,
normalization, optimizer, and dose. Failure of all arms does not prove that
K-union owners are mutually unrealizable unless the capacity control and
iterative frontier branch also fail under their own valid receipts.

## Claim boundary

A positive result may say:

> From the frozen step-2444 Source, this exact native training recipe can fit
> these thirteen images so that specified K-hit owners enter one unforced
> clean-greedy completion while named Source owners and output safety are
> retained.

A negative result may say:

> This exact treatment did not consolidate the frozen K-hit targets into clean
> greedy on these thirteen images within the executed dose.

This unit cannot establish validation performance, cross-image transfer,
generalization, population prevalence, a final production loss, architecture
necessity, or safety outside the observed thirteen-image optimization surface.

## Planned artifact root

Each immutable run identifier publishes under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-12-human13-k-union-to-greedy-overfit-screen/<run-id>/
```

No run identifier is reused after a partial or failed execution. A future
`results.md` must separate `Observed`, `Supported`, `Ruled out`, `Unresolved`,
and `Not claimed` before this unit can close.
