---
title: N256 Shared Output-QP Norm Scaling and Target-Blind Transfer
description: Fixed-surface semantic-versus-derangement scaling of one shared output-only residual on 256 train images, with target-blind readout on 128 disjoint images.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: cpu_contract_verified_bounded_n2_smoke_authorized
unit_id: 2026-09-01-n256-shared-output-qp-norm-scaling
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: mechanics_only
updated: 2026-09-01
---

# N256 Shared Output-QP Norm Scaling and Target-Blind Transfer

## Decision and question

From the exact four-coordinate `geo_sorted_xy` step-2444 Source, does one
shared output-only residual fit correct routes on a nested 256-image train
cohort with lower standardized minimum energy than matched cross-image route
derangements, and does that residual improve never-solved screen-dev images
without losing Source owners or degrading hard-debt and EOS gates?

The decision has two independent axes:

1. **semantic compression:** correct image-to-route assignments cost less than
   all admitted derangement replicates on one fixed output-row surface; and
2. **target-blind transfer:** on the disjoint screen-dev cohort, the semantic
   payload improves target margins and fresh natural-greedy annotated-owner
   coverage beyond all admitted null payloads while preserving Source owners.

The strongest alternative is high-dimensional finite-address lookup plus
shared formatting, coordinate, category, or EOS calibration. Training fit,
raw Frobenius norm, rank saturation, or numerical failure alone cannot reject
that alternative.

## Frozen terminology

- `N` is the number of train images in a nested stage:
  `N in {2, 4, 8, 16, 32, 64, 128, 256}`.
- `K` is the requested number of independent matched-derangement null
  replicates, not an image count or matrix condition number. `K=4` means one
  semantic solve plus four null solves at a stage. At `N=2`, only one distinct
  fixed-point-free route swap exists; receipts must report
  `requested_K=4` and `unique_K=1`, not four independent controls.
- Matrix conditioning is reported separately as `cond` or `kappa` diagnostics.

## Source, population, and target blindness

The machine-readable [policy (preserved from archive ref `archive/research-restructure-20260909/n256-shared-output-qp-norm-scaling` at `6b846c7c`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/n256-shared-output-qp-norm-scaling/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-01-n256-shared-output-qp-norm-scaling/policy-v1.json) owns exact paths, hashes,
derivation rules, and expected counts.

Planning receipts: [resource census (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/n256-shared-output-qp-norm-scaling/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-01-n256-shared-output-qp-norm-scaling/resource-census-v1.md) and
[harness benchmark (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/n256-shared-output-qp-norm-scaling/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-01-n256-shared-output-qp-norm-scaling/harness-benchmark-v1.md).

- Source checkpoint: step-2444 from the four-coordinate `geo_sorted_xy`
  lineage, with the original prompt, tokenizer, parser, evaluator, margin
  `0.01`, certificate tolerance `2e-5`, RP1.0 greedy decode, and natural EOS.
- `geo_sorted_xy` defines sealed canonical target routes only. Natural model
  predictions may appear in any valid row order; decoded order is neither a
  validity gate nor debt, and owner scoring uses global order-agnostic matching.
- Reuse the independently validated G0 `256 train + 128 screen-dev` COCO
  train2017 split and its checkpoint-aligned execution media. Raw COCO
  annotations remain the sole owner/category/continuous-geometry authority.
- The train cohort alone may define nested stages, canonical routes, baseline
  deficits, null assignments, constraints, solver state, or retries.
- The planner consumes the sealed train-only
  [train plan (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/n256-shared-output-qp-norm-scaling/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-01-n256-shared-output-qp-norm-scaling/train-plan-v1.json), whose records contain only image ID,
  density stratum, and owner count. It must not parse the joint annotated-owner
  ledger that also contains screen-dev target rows.
- Screen-dev target rows must not be read by the planning, row-universe, null,
  capture-selection, solve, checkpoint, retry, or stage-continuation paths.
  They are admitted only by the sealed readout path after an immutable payload
  exists.
- `val2017` remains untouched.

The train order is derived once from the frozen cohort and density labels by
the policy seed, SHA-256 ordering within each stratum, and the repeated stratum
cycle `high, low, medium`. Each stage is an exact prefix. Expected stage
burdens are:

| N | owners | decision positions | high / low / medium |
|---:|---:|---:|---:|
| 2 | 14 | 132 | 1 / 1 / 0 |
| 4 | 29 | 273 | 2 / 1 / 1 |
| 8 | 48 | 459 | 3 / 3 / 2 |
| 16 | 103 | 995 | 6 / 5 / 5 |
| 32 | 213 | 2,037 | 11 / 11 / 10 |
| 64 | 447 | 4,281 | 22 / 21 / 21 |
| 128 | 998 | 9,584 | 43 / 43 / 42 |
| 256 | 1,955 | 18,745 | 86 / 85 / 85 |

## Fixed intervention surface

Every stage and arm uses the same grammar-defined output-row universe:

- all coordinate tokens `<|coord_0|>` through `<|coord_999|>`;
- tokenized COCO category names inside the native object-row format;
- native object/box format tokens; and
- natural `<|im_end|>`.

The frozen tokenizer yields 1,136 unique rows. This surface is independent of
stage and cohort target identities. The base model, tied input embeddings,
checkpoint, hidden layers, prompt, ordering, route serialization, margin,
objective, and certificate remain unchanged. No per-image payload, image-ID
branch, model training, DoRA, merge, hidden-layer update, row pruning, sampled
certificate, or post-readout tuning is allowed.

At fixed row count `R=1136`, the registered target-versus-full-vocabulary
constraint count is exactly `P * R`; N256 projects to 21,294,320 registered
constraints. A resource or numerical stop is mechanics evidence only.

## Arms and matched null

### Semantic arm

Teacher-force each image's complete canonical raw-owner route on its own
checkpoint-aligned execution media and solve the exact minimum-Frobenius
selected-surface QP.

### Complete-route derangement nulls

For each stage, permute complete canonical image routes across receiving images.
The route, including every category and complete four-coordinate bbox row,
remains atomic; no coordinate position or token is independently permuted.
Every mapping is a complete permutation with no image fixed point, so the
global category, bbox, coordinate-token, row-token, and selected-row target
multisets are exact.

After all semantic train captures are sealed, but before any arm solve, compute
four deterministic matched assignments from the policy seeds. Pairing cost uses
only train-side category composition, owner/route length, coordinate-token
summary, semantic baseline-deficit energy, and selected-row burden. Later
replicates penalize reuse of earlier edges; screen-dev is never consulted.
Record requested and actually unique `K` at every stage and fail closed on an
incomplete or fixed-point assignment.

The null is a coherent-route control: each receiving image is teacher-forced
with its assigned donor route before solving. It therefore tests whether the
same readout can compile matched cross-image nonsense routes, not merely whether
target labels can be shuffled on frozen semantic states.

## Required readouts

For every stage and admitted arm, report:

1. images, owners, decision states, deficient positions, raw and standardized
   deficit energy, registered and active constraints, fixed rows, hidden rank,
   free variables, and solver/certificate counters;
2. `R_N = ||Delta W_N||_F`, `R_N/sqrt(owners)`, `R_N^2/owners`, and
   `R_N^2/standardized_deficit_energy`;
3. incremental energy per added owner and deficit from `N/2` to `N`;
4. effective rank, rank-95, largest-row energy share, and condition diagnostics;
5. functional norm on the frozen screen-dev reference-state distribution;
6. before refitting, the fraction and margin of newly added semantic
   constraints helped by the prior semantic payload; and
7. target-blind screen-dev target-margin change plus fresh natural-greedy
   gained, retained, and lost annotated owners, duplicate, malformed, cap,
   unmatched, and natural-EOS counts.

The common deficit scale is the RMS of positive N256 semantic Source deficits,
sealed before any solve. A zero scale is `MECHANICAL_INVALID`.

## Screen-dev gate

Evaluate Source once, then evaluate every stage's semantic payload and each
unique null payload in a fresh process, batch one. A process loads exactly one
payload and runs all 128 images; payload switching inside a process is
forbidden.

A stage has strict target-blind transfer only when:

- semantic mean target-margin change is positive and exceeds every admitted
  null replicate;
- semantic annotated-owner net gain is positive and exceeds every null;
- every Source-matched annotated owner is retained (`lost=0`);
- confirmed duplicate, malformed, and cap debt do not increase; and
- every candidate reaches natural EOS.

COCO unmatched predictions remain an explicit unknown monitor because missing
annotations cannot identify them as false positives. They do not receive hard
debt credit without separate entity/category and geometry review.

## Outcome and claim boundary

- **LOOKUP_COMPATIBLE:** at N256, semantic standardized energy is not below
  every admitted null and target-blind transfer is nonpositive.
- **SHARED_COMPRESSION_ONLY:** semantic standardized energy is below every
  admitted null at N256, but strict target-blind transfer does not pass.
- **TRANSFER_EVIDENCE:** semantic standardized energy is below every admitted
  null at N256 and the N256 strict target-blind transfer gate passes.
- **MIXED_EVIDENCE:** transfer and compression disagree, stage-wise evidence is
  materially inconsistent, or fewer than the requested distinct nulls prevent
  the stronger comparison.
- **MECHANICAL_OR_RESOURCE_BOUND:** the frozen problem was not solved and
  certified within an approved resource boundary.

The strongest positive claim is held-out annotated-owner improvement for this
frozen COCO train2017 split by one output-head residual. It is not population
generalization, base-model learning, hidden-state internalization, full-scene
completeness, an owner mechanism, precision, production readiness, or
architecture promotion.

## Implementation and launch boundary

Study-owned implementation surfaces are:

- `policy-v1.json`;
- `train-plan-v1.json`;
- `scripts/research/run_n256_shared_output_qp_norm_scaling.py`;
- `scripts/research/run_n256_n2_smoke_bounded_process.py`; and
- `tests/research/test_n256_shared_output_qp_norm_scaling.py`.

The runner may reuse the Human13 hook, capture, QP, immutable-receipt, and
evaluator mechanics only after adapting them to the fixed surface and this
unit. It must use exact block separation, a compact matrix-free active operator,
and an exhaustive FP64/FP32 certificate; approximation, rank truncation, or
post-hoc row selection is forbidden.

Implementation and CPU verification are authorized. Broad GPU/model execution
is not authorized. The sole exception is the bounded N2 plumbing smoke below.
Before the first production-shaped full N2/N4 slice, freeze and get
user acceptance for wall time, host RSS, GPU memory, active constraints,
artifact bytes, attempt count, corrected-launch policy, and the projected N256
critical path. The slice proves mechanics only. Stop at the first approved
resource boundary and never reinterpret it as a scientific negative.

## CPU acceptance receipt

On 2026-09-01, the focused CPU suite passed `14/14`; policy validation reproduced
the 1,136-row hash, all eight nested stages, and the N256 count of 21,294,320
registered constraints. Python compilation, diff whitespace checks, and Serena
Python diagnostics were clean. Sensitivity checks cover row identity,
train/screen access separation, N4 distinct-null construction, N2 unique-null
collapse, block/scalar separation parity, active-operator adjoint/objective/
gradient parity, dual-gap rejection, full-vocabulary pre-refit competition,
screen-cell completeness, unauthorized model commands, joint-ledger access
rejection, prompt-target non-exposure, strict `im_end` EOS, arbitrary decoded
prediction order, launch identity, and resource-bound classification. The
bounded-process wrapper separately passed both a no-op completion probe and an
RSS-breach termination sensitivity check.

The first N4 planner test failed with only two unique assignments before the
deterministic no-good branch correction, then passed with four. The planner
also now reads only the sealed train plan rather than parsing the joint
train/screen target ledger.

At the time of this receipt, legacy/full `capture` and `screen` remained
fail-closed. The bounded N2 entries require immutable lead launch packets and
had not yet executed. This receipt is CPU mechanics evidence only; it is not a
capture, solve, decode, metric, transfer, feasibility, or model-quality result.

## Authorized N2 plumbing smoke

The user authorized one non-scientific N2 vertical on 2026-09-01:

- one GPU; total wall time at most two hours;
- GPU memory at most 40 GiB; host RSS at most 64 GiB; artifacts at most 10 GiB;
- N2 semantic plus its sole unique complete-route swap, with at most 20,000
  active constraints per solve;
- one model-entry attempt, no result-driven or corrected-launch retry, and at
  most three 4,000-iteration numerical continuations on the same dual state;
- a target-blind SHA-ordered two-image screen subset, evaluated in one fresh
  Source process and one fresh process per payload: six natural decodes total;
- stop and report after plumbing/resource evidence. The subset cannot produce
  any compression, transfer, feasibility, or model-quality outcome class.

The machine-readable smoke policy and launch packet must bind the exact GPU,
inputs, payloads, process identities, counters, and measured peaks before this
exception can execute. Full 128-image screen evaluation and every later N stage
remain unauthorized.
