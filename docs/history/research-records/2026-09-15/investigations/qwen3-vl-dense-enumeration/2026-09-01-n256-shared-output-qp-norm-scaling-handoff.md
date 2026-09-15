---
title: N256 shared output-QP norm-scaling handoff
description: Historical September 2 continuation snapshot and frozen N256 protocol; current route is maintained in the compass.
type: investigation
role: research-handoff
authority: transport_only
status: historical_snapshot
updated: 2026-09-09
---

# N256 shared output-QP norm-scaling handoff

> **Superseded continuation snapshot — 2026-09-09.** Use the
> [current compass](compass.md) and [experiment router](experiments/index.md)
> for the current evidence boundary. The September 2 continuation below,
> including its then-unlaunched N8 next step, is historical. Its protocol,
> measurements, source identities, and receipt links are preserved unchanged.

## Historical continuation point — 2026-09-02

The design below has now been implemented and partially executed. Treat this
section and the linked terminal receipts as current; the original proposal is
retained afterward for provenance.

- The full target-blind ladder stopped at N2. Semantic lost 42 Source owners
  and increased duplicate/malformed debt; the null also failed natural EOS.
  See [full-n2-results-v1.md](experiments/2026-09-01-n256-shared-output-qp-norm-scaling/full-n2-results-v1.md).
- A separately frozen train-only N4 discriminator completed with four unique
  matched derangements. Semantic norm is `0.329872`, about `5.06x` below the
  lowest null, and its difficulty-normalized cost is about `2.60x` below the
  lowest null. See
  [n4-train-only-results-v1.md](experiments/2026-09-01-n256-shared-output-qp-norm-scaling/n4-train-only-results-v1.md).
- CPU-only decomposition shows `93.5948%` of semantic residual energy on
  coordinate rows, `6.4022%` on category rows, and `0.0030%` on format/EOS.
  Removing coordinates breaks 104/116 coordinate-target positions; removing
  category rows breaks 9/37 category-target positions. This rejects a pure
  format/EOS calibration account. See
  [n4-mechanism-results-v1.md](experiments/2026-09-01-n256-shared-output-qp-norm-scaling/n4-mechanism-results-v1.md).
- Effective rank is `8.515`, but exact margin retention is tail-sensitive:
  rank 64 keeps `99.57%` energy yet only `166/273` positions; the first frozen
  tested rank restoring all positions is 128. Low norm is therefore not the
  same claim as tiny exact rank.

The shortest next discriminator is train-only N8 semantic versus four matched
nulls on the same sealed capture and 1,136-row surface. It must remain separate
from the stopped screen-dev transfer ladder. N8 has not been launched.

## Checkout and isolation boundary

- Worktree: `/data/CoordExp/.worktrees/n256-shared-output-qp-norm-scaling`
- Branch: `codex/n256-shared-output-qp-norm-scaling`
- Base commit: `0dc1dfc7b4fd793c1b51505f6c1375c99aab4b2e`
- Fork source: `/data/CoordExp/.worktrees/human13-output-qp-identity-generalization`

The source checkout was heavily dirty when this worktree was created. This
worktree intentionally contains only the committed base and this handoff; it
does **not** silently copy the source checkout's unrelated or in-progress DoRA,
G0.4, OpenSpec, runner, test, or research-record edits. Read those files and
port only an explicitly audited minimal set if this unit needs them. Do not
write into the source checkout.

This handoff is transport, not experimental authority. Before model execution,
create a new research unit that freezes the cohort, fixed readout surface,
semantic/null contrast, metrics, resource bounds, attempt budget, and stop rule.

## Current decision

**GO to an independent N=256 norm-scaling study, but raw Frobenius norm alone
cannot decide whether the residual contains general knowledge.** The
decision-bearing question is:

> As one shared output-only residual is fit on a nested sequence of images, do
> semantically correct constraints exhibit compression and forward transfer
> beyond a matched cross-image derangement null?

Call the image count `N`; reserve `K` for permutation/null replicates to avoid
collision with existing experiment terminology.

## Correct Human13 premise

Human13 did **not** use thirteen per-image payloads. The accepted N13 result used
one immutable shared `Delta W_out` across all thirteen images, with no image-ID
branch, per-image checkpoint, per-image residual, or inference-time payload
selection. It achieved `392 / 392` owners at IoU50/60/80, natural EOS on all
thirteen images, and zero hard debt.

The accepted claim is finite-panel shared-output compilability only. All
thirteen target routes and hidden states entered the solve, so it is compatible
with a high-dimensional finite lookup and is not held-out generalization.

### Existing nested-stage measurements

| N | owners | `||Delta W||_F` | `||Delta W||/sqrt(N)` | `||Delta W||^2/owner` | effective rank | rank-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 65 | 0.683992783 | 0.483655935 | 0.007197633 | 24.84 | 71 |
| 4 | 123 | 1.033198908 | 0.516599454 | 0.008678862 | 34.36 | 101 |
| 13 | 392 | 1.914429690 | 0.530967262 | 0.009349594 | 62.75 | 232 |

Across these three points, a descriptive log-log fit is approximately
`||Delta W_N|| proportional to N^0.55`, close to the `sqrt(N)` lookup null.
This is preliminary lookup-compatible evidence, not a conclusion: there are
only three stages, image difficulty changes, and the trainable selected-row
surface expanded from 217 to 360 to 772 rows.

The N13 hidden-state span had already reached rank 2048, so N>13 may enter a
different regime. Numerical or resource failure at larger N is mechanics
evidence unless infeasibility is certified under the frozen mathematical
problem.

## Why the proposed binary norm rule must be rejected

For a fixed parameter space and nested constraints,

\[
R_N^2=\min_x\|x\|_2^2\quad\text{subject to}\quad A_Nx\ge b_N.
\]

The dual gives `x* = A_N^T lambda*`; therefore `R_N` is determined by the
constraint deficits and Gram geometry, not by semantic meaning itself.
Semantic and target-permuted problems can have the same norm curve.

- Approximately orthogonal equal-cost constraints give `R_N proportional to
  sqrt(M_N)`, where `M_N` is independent constraint burden.
- Repeated or already-satisfied constraints give a plateau. That proves
  redundancy under this readout, not an owner mechanism.
- A genuinely useful shared component plus irreducible image-specific noise can
  still give `R_N^2 = R_shared^2 + c M_N`; divergence does not rule out useful
  shared structure.
- `R_N / N` is not diagnostic because it declines as `1/sqrt(N)` even under the
  pure lookup null.

The new unit may derive and register monotonicity only after freezing one
common parameter surface. The historical stages changed selected output rows,
so their raw curve does not satisfy that strict theorem premise.

## Proposed frozen contrast

### Cohorts

- One density-stratified nested train cohort with stages
  `N in {2, 4, 8, 16, 32, 64, 128, 256}`.
- One disjoint target-blind screen-dev cohort, never used for constraint
  selection, target aliases, solver choices, checkpoint choice, or retries.
- Keep `val2017` untouched unless a later unit explicitly promotes to a
  generalization study.

The already prepared `256 train + 128 screen-dev` G0 cohort is a candidate
mechanical input, not automatically this unit's authority. If reused, verify
its immutable ledger and freeze that choice in the new unit.

### Data and geometry

- Source checkpoint: `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
- Execution media must follow the checkpoint-aligned `1024 * global max length
  12000` processed dataset, not raw COCO JPEG dimensions:
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl`.
- Raw COCO annotations remain the sole owner/category/continuous-geometry
  authority. Processed objects must not add, remove, or override owners.
- Freeze prompt, tokenizer, `geo_sorted_xy`, quantization, evaluator, margin,
  repetition penalty, EOS, and cap semantics.

### Fixed intervention surface

Every N stage must optimize the same preregistered output-row universe. Do not
let the surface expand with the tokens observed at each N, because that
confounds added constraints with added parameters. A train-cohort-wide frozen
row universe or a grammar-defined coordinate/category/format row universe is
acceptable; screen-dev target identities must not define it.

Use one shared output-only residual at every stage. No per-image payload,
image-ID selection, model training, DoRA, merge, or hidden-layer update belongs
in this unit.

### Controls

For each stage compare:

1. **Semantic:** correct image-to-target constraint assignment.
2. **Matched derangement null:** whole target bundles reassigned across images
   with no fixed points, matching at least category composition, row/token
   length, coordinate-token distribution, baseline deficit, and selected-row
   burden.

Do not use independently permuted coordinate positions. A bbox/route bundle is
the atomic target unit. Freeze the null seed and matching rule before solving.

## Required readouts

Report N and the actual burden: images, owners, decision states, deficient
positions, standardized deficit energy, registered constraints, active
constraints, selected rows, and free variables.

For semantic and null arms report:

1. `R_N = ||Delta W_N||_F`;
2. `R_N / sqrt(owners)` and `R_N^2 / owners`;
3. incremental energy `R_N^2 - R_(N/2)^2` per added owner/deficit;
4. effective rank, rank-95, largest-row energy share, and selected-row count;
5. a functional norm on a frozen reference state distribution, e.g.
   `||Delta W_N Sigma_h^(1/2)||_F`;
6. before refitting, the fraction and margin of newly added semantic
   constraints already helped by `Delta W_(N/2)`;
7. on screen-dev, target-margin change plus fresh natural-greedy gained,
   retained, and lost owners, duplicate/unmatched/malformed/cap debt, and
   natural EOS.

The key comparisons are semantic-versus-null excess compression and target-
blind forward transfer. Training fit and raw norm do not own the conclusion.

## Outcome classes and stop boundary

- **LOOKUP_COMPATIBLE:** semantic normalized norm/rank growth is not better
  than the matched null and target-blind forward transfer is nonpositive.
- **SHARED_COMPRESSION_ONLY:** semantic constraints compress better than the
  null, but unseen natural-greedy behavior does not improve safely.
- **TRANSFER_EVIDENCE:** semantic beats the null and improves never-solved
  images while preserving Source owners and hard-debt/EOS gates.
- **MECHANICAL_OR_RESOURCE_BOUND:** the exact frozen problem was not solved or
  certified. Do not reinterpret this as absence of shared structure.

Even `TRANSFER_EVIDENCE` supports only held-out annotated-owner improvement for
the frozen split. It is not base-model learning, hidden-state internalization,
full-scene completeness, or production readiness.

Before the first costly model run, project forwards, constraint materialization,
active-set state, host RSS, GPU memory, payload size, and wall time from a
production-shaped small stage. N13 already had 2,807,764 registered constraints
and 8,079 active constraints; a naive N256 extension is not automatically
tractable. Stop at the first registered resource boundary rather than silently
changing the estimand or solver.

## Minimum reading path

1. This handoff.
2. Final live Human13 record in the source checkout:
   `/data/CoordExp/.worktrees/human13-output-qp-identity-generalization/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-31-human13-shared-output-qp-identity-generalization/results.md`.
3. Its owning unit beside that result. The older committed handoff in this new
   checkout contains superseded counts and is historical evidence only.
4. Exact residual receipts:
   `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-human13-shared-output-qp-same-panel-overfit/20260831T-n{2-v2,4-v1,13-v5-certificate-polish-corrected}/solve/`.
5. Candidate reusable runner and focused test, currently untracked in the dirty
   source checkout:
   `/data/CoordExp/.worktrees/human13-output-qp-identity-generalization/scripts/research/run_human13_output_qp_same_panel.py` and
   `/data/CoordExp/.worktrees/human13-output-qp-identity-generalization/tests/research/test_human13_output_qp_same_panel.py`.
6. The user's norm-scaling discussion packet:
   `/data/CoordExp/.codex/attachments/7b5b3abc-eecf-4263-a05f-8cab3af65182/pasted-text.txt`.

Reverify every volatile path, hash, GPU state, and dirty-file ownership before
implementation or launch. The first next action is a research unit and bounded
dimension/resource census, not a 256-image launch.
