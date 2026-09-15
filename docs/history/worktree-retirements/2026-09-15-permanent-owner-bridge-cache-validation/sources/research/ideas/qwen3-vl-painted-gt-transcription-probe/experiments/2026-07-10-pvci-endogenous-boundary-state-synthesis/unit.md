---
title: PVCI Endogenous Boundary-State Synthesis
description: Tests whether clean Qwen visual features of an oracle-selected instance can synthesize causal continuation, identity, and geometry boundary deltas on held-out images.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-10-pvci-endogenous-boundary-state-synthesis
topic: qwen3-vl-painted-gt-transcription-probe
status: completed
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - causal-patching
  - endogenous-state
updated: 2026-07-10
---

# PVCI Endogenous Boundary-State Synthesis

## Question

Can clean Qwen visual features from a GT-selected instance support, used only
as a research oracle, predict residual deltas that causally recover the
continuation, current-instance identity, and geometry effects previously
obtained from painted/post-scatter donor trajectories?

This unit tests synthesizability of useful causal handles. It does not assume
that residual injection, three fixed adapters, a cursor schedule, or any other
candidate implementation should become the final architecture.

## Decision Relevance

- Functional capability or uncertainty: whether the useful boundary-local
  states are endogenous to clean Qwen instance features rather than available
  only by replaying a painted/post-scatter trajectory.
- Costly or hard-to-reverse choice this result could change: whether future
  training should synthesize boundary-local control directly, preserve a
  persistent input-level visual designation that lets Qwen compile the states,
  or first introduce a stronger instance representation.
- Outside this unit's scope: autonomous object selection, coverage, commit,
  STOP calibration, mixture training, Qwen backbone training, an external
  detector/backbone, production detection metrics, and final architecture.

## Competing Hypotheses

- H1: clean instance features predict reusable boundary deltas.
  - Expected signature: a tiny held-out mapper produces continuation,
    identity, and geometry deltas whose causal rollout effects recover a
    substantial fraction of exact-donor effects and clearly exceed wrong-
    instance, shuffled, geometry-only, global-only, and random controls.
  - Meaningful falsifier: predicted deltas fit training targets but fail on
    image-held-out causal rollouts or do not separate the selected instance
    from within-image alternatives.
- H2: the target deltas are transferable but not linearly readable from the
  tested clean features.
  - Expected signature: a train-basis low-rank exact-target projection retains
    causal efficacy on held-out rows, while ridge synthesis from clean support
    features does not.
  - Meaningful falsifier: the low-rank exact-target ceiling itself loses the
    donor effect at the tested ranks.
- H3: painted/post-scatter conditioning must remain available so Qwen can
  compile trajectory-dependent states.
  - Expected signature: exact full-state donors or persistent post-scatter
    remain effective, but additive exact deltas, low-rank projections, and
    synthesized deltas fail even under matched textual prefixes.
  - Meaningful falsifier: a finite additive delta schedule recovers the donor
    effects on held-out rows.
- H4: apparent synthesis comes from geometry or schedule shortcuts rather than
  instance-specific visual evidence.
  - Expected signature: geometry-only/global-only predictors perform as well
    as support-conditioned predictors, or same-class within-image swaps do not
    change the controlled row.
  - Meaningful falsifier: support-conditioned synthesis wins on causal rollout
    and same-class swaps follow the swapped instance rather than the original
    row or class prior.

## Completion Promise

This unit is complete when:

- Evidence gate: a 32-row source ledger is audited; synthesis fitting and
  causal evaluation use only rows whose persistent-post first prediction both
  forms a closed row and names the marked source instance; prefix-matched clean
  and post targets are captured at row-entry layer 19, post-object-start layer
  23, and pre-`x1` layer 23; source images are kept disjoint across fit and test;
  mapper selection uses fit images only; and held-out causal interventions
  include exact, low-rank, synthesized, wrong-instance, shuffled, geometry-
  only or global-only, and norm-matched random controls.
- Acceptable evidence: exact runtime/model/tokenizer/feature-store identities;
  target and split receipts; raw and normalized regression diagnostics; native
  free-continuation outcomes; hook/cache/application receipts; phrase
  attribution; all four coordinates; IoU/L1; same-class swap strata; and
  deterministic artifact regeneration where practical.
- Insufficient evidence: hidden-state MSE or cosine alone; row-level training
  fit without image-held-out causal rollout; a row-random split that leaks the
  same image; a control set without wrong-instance and random baselines; or
  success only from exact painted/post-scatter donors.

## Outcome Interpretation

- If support-conditioned synthesis recovers at least 70% of the exact additive
  donor phrase effect and materially improves geometry over identity-only on
  held-out images: direct endogenous boundary-state synthesis becomes more
  plausible.
- If exact additive deltas work but learned synthesis fails: the causal handle
  is valid but the tested clean feature/readout family is inadequate.
- If low-rank exact-target projections work but feature prediction fails: the
  state is compressible but not linearly predictable from the tested support
  representation; a small nonlinear follow-up may be justified.
- If only full-state replacement or persistent post-scatter works: direct
  additive scheduling becomes less plausible and input-level persistent
  designation remains the stronger route.
- If wrong-instance or same-class swaps do not separate: the learned signal is
  not yet current-instance binding, even if aggregate phrase accuracy rises.
- If geometry-only matches visual support: the oracle geometry is carrying the
  effect and the visual-identity claim is demoted.
- If the result is negative at this handle: do not infer that endogenous
  control is impossible; boundary layer, additive-vs-replacement semantics,
  feature pooling, sample size, or linearity may be mismatched.
- Possibilities this probe cannot distinguish: autonomous selection quality,
  how an instance support would be obtained, coverage representation, and the
  final runtime delivery interface.

## Evidence Scope

- Checkout or branch:
  `/data/CoordExp/.codex/worktrees/69ed/CoordExp`,
  `codex/continue-handoff-session`.
- Baseline commit: `e07c6b73` plus the uncommitted research instrumentation
  from the predecessor unit.
- Predecessor unit:
  `../2026-07-10-pvci-sequential-control-variable-causal-schedule/unit.md`.
- Config:
  `configs/coordexp_swift/infer/research/pvci_sequential_control_e1_debug4.yaml`.
- Checkpoint: E1 anti-copy step 484 through the research-only byte-identical
  replay surface recorded by the predecessor unit.
- Source feature store:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_val32/feature_store/`.
- Source rows: 32 rows from 17 source images, of which 17 rows from 13 images
  have a persistent-post first prediction that names the marked source
  instance. Clean image-token embeddings are
  reconstructed as `painted_image_embeds.float() - delta_image_embeds`; no
  additional vision model is introduced.
- Planned artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_endogenous_boundary_state_synthesis/`.
- Frozen plan:
  `/data/CoordExp/outputs/painted_gt/pvci_endogenous_boundary_state_synthesis/plan.json`,
  SHA-256
  `9e32b328c4bbfe064887a6a12a1b8524882ea034077d6ceb805763a6b748982f`.
- Capture artifacts:
  - `capture/summary.json`, SHA-256
    `7613877c43a649241d0fe3753dda10e139349f58e0307cf208e1a9b95034a990`;
  - `capture/target_states.pt`, SHA-256
    `d8c9bebd06353931805ec009be1139b32c8abdc8b6034ed9e61eb3bc99026cc9`;
  - `capture/target_ledger.jsonl`, SHA-256
    `2592b659ec3fa6fb41bbb63da133aab2ba435aefe949d65c3f2b5cc893fc905a`.
- Fit artifacts:
  - `fit/summary.json`, SHA-256
    `e5c34564547c3313920b2369798b0b309de88b4b3a4cf7ba61080cc777843e00`;
  - `fit/mapper_predictions.pt`, SHA-256
    `1a5029004192e7ec34e0237956a3ef63f834467c0891991e7dc2b0a9137a3dde`;
  - `fit/fit_rows.jsonl`, SHA-256
    `562f68328933e3bb93f795579b7f2eeccd41cdf4e8bccf95508862c28e7587cc`.
- Causal evaluation artifacts:
  - `causal_eval/summary.json`, SHA-256
    `86f664bd6d06afbe5d165446558a931928a8d3abc6ed6b3580d052e69d3eb0b4`;
  - `causal_eval/conditions.jsonl`, SHA-256
    `bde2965730663c4bc3db8003b601d46ff4dc59b66b920c239d7487d0986cf6d5`.
- Primary metrics: held-out continuation-to-post rate, post-phrase recovery,
  row closure, coordinate L1/IoU to post, causal recovery relative to the exact
  additive-delta ceiling, correct-vs-wrong instance separation, and same-class
  swap behavior.
- Known limitations: one checkpoint, 17 synthesis-eligible rows, 13 images,
  one synthesis-eligible source-image same-class stratum, GT instance support
  as a research oracle, and residual
  deltas as experimental handles rather than a proposed production interface.

## Procedure

1. Freeze a source-image-disjoint split before fitting. Put the two images with
   same-class marked-instance pairs in the held-out diagnostic stratum and
   choose the remaining held-out images deterministically without reading
   outcomes.
2. Capture prefix-matched clean and persistent-post layer-input states at the
   continuation, identity, and geometry boundaries for all eligible rows.
3. Materialize compact clean features from selected support mean, source-ring
   mean, global mean, and normalized support geometry. Preserve feature-family
   ablations rather than silently concatenating every source.
4. Fit ridge/low-rank maps using fit images only. Choose regularization and
   rank without held-out rollout outcomes.
5. Establish causal ceilings in order: exact full-state replacement, exact
   additive delta, train-basis low-rank exact-target projection, then learned
   synthesis.
6. Run boundary-isolated and full-schedule held-out continuations. Include
   wrong-instance, within-image shuffled, same-class swap, geometry-only,
   global-only, and norm-matched random controls.
7. Interpret causal rollout evidence against the predeclared outcome map and
   stop. Do not continue automatically into Qwen training or final architecture.

## Pre-Launch Contract And Risk Gate

- Intended contract: the Qwen model is frozen; only CPU-side tiny mappers are
  fit; GT support is research-oracle input; test images never contribute to
  mapper fitting or hyperparameter selection.
- Runtime contract: native HF generation and KV cache are preserved; residual
  interventions are receipted; additive and replacement semantics are explicit
  and never silently interchanged.
- Artifact contract: split identities, feature-store hashes, target tensor
  hashes, mapper fit rows, hyperparameters, intervention receipts, parser
  outcomes, and summary denominators are persisted.
- Hold conditions: any image leakage, missing mandatory boundary capture,
  failure of exact predecessor replay, additive/replacement ambiguity, silent
  zero fallback, missing wrong-instance control, or causal claims based only on
  representation fit.

## Observations

- Risk-gate correction: syntactic post-row validity was not a sufficient
  selected-instance target contract. Only 17 of the 32 source rows produced a
  first persistent-post prediction whose description matched the marked source
  instance. The first plan was held before interpretation, the semantic gate
  was added, and the final image-disjoint split contains 12 fit rows from nine
  images and five evaluation rows from four images. The evaluation set contains
  the only fully eligible same-class pair (`wine glass` / `wine glass`).
- Target capture: 32 continuation pairs and 26 identity/geometry pairs were
  captured. Persistent-post next-token replay was exact for 32/32
  continuation prefixes, 26/26 identity prefixes, and 23/26 geometry prefixes.
  The three geometry-prefix misses remain recorded; no rows were silently
  removed after state capture.
- Exact causal handle: on the five synthesis-eligible held-out rows, exact
  additive continuation/identity/geometry deltas produced the marked phrase on
  5/5 rows, closed 4/5 rows, and achieved mean post-box IoU `0.450` over
  comparable rows. Exact full-state replacement produced 5/5 phrases, 5/5
  closed rows, and mean IoU `0.523`; persistent post-scatter remained the exact
  5/5 and IoU `1.0` ceiling.
- Linear synthesis: direct support-conditioned synthesis produced the marked
  phrase on 1/5 rows, identical to the clean, global-only, and norm-matched
  random baselines at the primary scale. Doubling the schedule reached 2/5 but
  mean IoU remained `0.032`; quadrupling reduced closure to 1/5. The
  support-conditioned representation did not separate the same-class pair.
- Low-rank ceiling: increasing the train-basis exact-target projection from
  rank 1 through rank 12 never exceeded 2/5 marked phrases. Rank 12 reached
  mean post-box IoU `0.135`, far below exact additive deltas.
- Stage attribution:
  - exact continuation + synthesized identity + exact geometry produced 2/5
    phrases, exactly matching the no-identity and geometry-only/global identity
    controls. Scaling synthesized identity to 2x or 4x did not improve it;
  - exact continuation + exact identity + synthesized geometry preserved 5/5
    phrases and reached mean IoU `0.371`, but no-geometry reached `0.350`, and
    geometry-only/global geometry controls also reached `0.371`;
  - synthesized support continuation opened 4/5 rows when later stages were
    exact, identical to global continuation. Geometry-only continuation opened
    5/5, so a coarse row-opening signal may be easier than instance binding,
    but this panel does not establish visual-instance-specific continuation.
- Execution receipts: all 162 intervention rows restored hooks, and every
  continuation anchor applied exactly once. Identity/geometry anchors applied
  in 154 rows and were explicitly recorded as optional-unreached in eight rows
  whose earlier intervention did not naturally reach those boundaries.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_endogenous_boundary_state_synthesis/`.

## Interpretation

- Supported reading: the finite additive boundary schedule is a valid causal
  handle on held-out selected-instance rows. The failure is not that useful
  continuation/identity/geometry changes require permanent post-scatter drive;
  exact clean/post deltas can reproduce much of the effect.
- Supported negative reading: mean-pooled clean source-box and source-ring
  features do not linearly synthesize reusable current-instance identity or
  geometry deltas on this image-held-out panel. Their causal behavior is
  indistinguishable from geometry-only/global controls at the relevant stages.
- H1 is not supported at the tested feature and mapper family. The predeclared
  70% exact-donor phrase-effect gate failed: primary support synthesis recovered
  none of the exact additive phrase gain over clean at 1x and only one of four
  gained rows at 2x.
- H2 is not supported at the tested train-derived ranks. Exact held-out targets
  projected into the rank-12 fit basis retained only 2/5 marked phrases, so the
  failure cannot be attributed only to ridge feature prediction.
- H3 is falsified at this handle: exact additive deltas work without persistent
  post-scatter conditioning, although persistent post remains the exact ceiling.
- H4 is supported: geometry/global predictors match support-conditioned
  predictors, and same-class swapping does not reveal instance-specific
  control.
- Alternative reading: 12 fit rows are too few for a general readout, or mean
  pooling destroys the spatial/nonlocal structure needed to generate the
  states. A nonlinear mapper trained on this sample would not distinguish
  those explanations reliably and is therefore not justified as the immediate
  follow-up.
- Remaining uncertainty: whether a structured image-token renderer can use a
  selected support to write the input-level spatial/nonlocal pattern from which
  frozen Qwen compiles the useful residual schedule.
- More likely after this result: input-level structured rendering, persistent
  cue tokens, or a richer spatial instance representation as the synthesis
  handle.
- Less likely after this result: three independent linear adapters from pooled
  instance means directly to decoder residual deltas.
- Candidate implementations still unresolved: residual schedule, input-level
  feature delta, appended cue tokens, instance slots, selector, coverage, and
  STOP mechanism.

## Research Unit Closeout

Observed:

Exact additive boundary deltas are causally effective on five held-out
mark-follow rows, but tiny linear maps from pooled clean support features do not
recover the instance-specific identity or geometry effects and do not beat
non-instance controls.

Evidence gate:

Passed after one fail-closed plan correction. Source-image disjointness,
mark-follow target eligibility, fit-only hyperparameter selection, boundary
capture receipts, exact/additive/low-rank/synthesized/wrong/random controls,
same-class evaluation, native free continuation, all-coordinate parsing, and
hook/application receipts were all materialized. The small sample size remains
an explicit limitation.

Supported:

The causal schedule itself and additive-delta semantics are viable experimental
handles. Direct linear endogenous synthesis from the tested pooled clean
features is not supported.

Not supported yet:

Reusable endogenous current-instance states, a nonlinear state generator,
autonomous selection, coverage, commit, STOP, mixture stability, and any final
architecture.

Architecture update:

No architecture is promoted. Direct pooled-feature-to-residual adapters are
demoted. A structured input-level renderer becomes more plausible than direct
state prediction, but remains a hypothesis rather than a chosen module.

Next decider:

Test a reusable local post-scatter rendering alphabet without training Qwen:
fit a translation-shared mean or token-conditioned map from clean source-
support tokens to the exact local image-token delta, place the synthesized
field on held-out source supports, and compare causal marked-instance follow
against exact local mark delta, full image-token delta, wrong-location,
same-class-swap, global-mean, and norm-matched random controls. Existing
component artifacts already show that exact mark-local delta follows the marked
instance on 16/32 source rows versus 17/32 for the full image-token delta, so a
bounded renderer test has higher information gain than fitting an MLP directly
to residual states.

Promotion decision:

Not promoted. Close this unit as a negative result for direct linear state
synthesis and a positive result for the finite additive causal schedule.
