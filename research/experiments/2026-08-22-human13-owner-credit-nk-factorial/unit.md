---
title: Human-13 Owner-Credit N/K Factorial
description: A standalone BF16/FA2 census and N/K factorial testing whether more sampled trajectories (K) and more images (N) yield a training-credit direction that safely consolidates K-hit owners into Human-13 clean greedy without eroding existing Source owners.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-08-22-human13-owner-credit-nk-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-28
---

# Human-13 Owner-Credit N/K Factorial

## Decision and outcome

This unit records, as a records-only return under
`reclaim-research-probes-lifecycle` (D5, D9), the frozen scientific protocol
and executed evidence of a standalone (non-production) owner-credit probe that
ran as three sequential standalone executions on branch
`codex/human13-nk-factorial-probe`: a one-image existence check, a four-image
K16 factorial, and a thirteen-image paired K4/K8 factorial. The probe's code,
`contract.md`, and CPU-only tests were never returned to `research-probes`;
only this unit and its result are returned. The lane is retired
(`probe-final/human13-nk-factorial-probe`) after this unit closes.

The executed facts, milestone tables, and bounded interpretation are in
[results.md](results.md). In summary: the N=13 paired K4/K8 matrix is valid,
independently-replayed evidence that K8 is not supported over K4 under the
required dual-repetition-penalty criterion — K8's small RP1.0 aggregate gain
is not replicate-robust and RP1.10 owner loss is worse. Widening the image
cohort from N=4 to N=13 does not by itself make the current credit direction
coherent.

## Question

Can information from multiple sampled trajectories (K) and a wider image
cohort (N) be combined into a scalable training-credit signal that improves
Human-13 clean-greedy physical-owner coverage while retaining owners already
found by the Source model, under dual-repetition-penalty (RP 1.0 and 1.10)
greedy evaluation?

## Competing explanation

The strongest alternative to "more K and more N denoise the update" is that
additional sampled trajectories and images add conflicting gradient signal
rather than a shared, generalizable direction — i.e. any observed owner gain
is image-specific churn (some images gain, others lose) rather than a stable
cross-image improvement. The paired K4/K8 replicate design and the separate
dual-RP evaluation are the controls meant to separate these: a real denoising
effect should show K8 winning consistently across replicates and RP settings,
not only in one pooled RP1.0 aggregate.

## Primary observation

The N=13 K4/K8 milestone-4 dual-RP owner table (TP/FP/FN by decode and K,
reproduced verbatim in [results.md](results.md)) is the primary observation:
K8 nets a small aggregate RP1.0 gain (TP+2/FP-10/FN-2) but a materially worse
RP1.10 outcome (TP-19/FP+16/FN+19), and is better/equal/worse than K4 in only
1/2/1 (RP1.0) and 1/0/3 (RP1.10) of four replicates. This directly falsifies
"more K is a safe scale-up" for the frozen credit-fusion objective.

## Frozen scientific surface

The following is the frozen contract for the standalone probe, as it existed
on `codex/human13-nk-factorial-probe` at its retired tip `a904e3ae3`
(`git show codex/human13-nk-factorial-probe:research/2026-08-22-human13-owner-credit-nk-factorial/contract.md`).
It governed the N=13 paired K4/K8 corrected-geometry execution; the earlier
N=1 and N=4/K16/T4 standalone runs (commits `32bc918d4` and `b36216f10`) are
its informal predecessors under the same standalone owner-credit design and
share its census/credit/rollback mechanics without this contract's explicit
eligibility and seed-namespace ledger.

This probe is a standalone BF16/FA2 census and factorial driver. It does not
import or invoke the production all-HF/CudaAdapter, compiler, transaction,
checkpoint, promotion, DDP, or vLLM paths. CPU dry planning is the only action
authorized in the source package; live model, GPU, sampling, backward, and
optimizer execution belong to a later launch package (which is what actually
executed to produce the results in this unit).

### Census and credit definitions

- Census all 13 manifest images with clean greedy at RP 1.0 and 1.10 plus K32
  sampling at temperature 0.4, top-p 1.0, RP 1.0, and 512 new tokens.
- `G_current` is the union of same-category one-to-one IoU-0.5 owners on the
  two current clean-greedy surfaces. `H_reachable` is the K32 sampled owner
  union minus `G_current`; every other GT owner is neutral `M`.
- Eligibility requires at least one current G owner, at least two reachable H
  owners, and at least four aggregate first-hit observations across reachable
  H. Rank by descending H count, descending aggregate H support, ascending
  image ID. N4 is the first four; Nlarge is the first eight when available,
  otherwise every eligible image. Fewer than four images is a typed HOLD.
  Cohorts nest.
- Each distinct N/K/replicate cell starts from exact Source with fresh AdamW,
  LR 3e-6, preservation coefficient 1.0, independent per-image RLOO, an
  image-balanced objective, persistent optimizer for T=2, and audits at
  0/1/2. Fixed K16 evaluation seeds are shared across cells and disjoint from
  census and training. Training seeds are disjoint across cell, step, and
  image.
- Credit keeps existing unmatched, invalid, duplicate, malformed, and
  premature stop burden semantics. Current H and M are derived only from
  census evidence; legacy manifest G/H/M are historical metadata. An image
  with no current-G preservation token rows contributes a scalar zero
  preservation term.
- Every live cell unconditionally restores the exact CPU trainable-parameter
  snapshot and writes no model checkpoint.

### Failure-mode matrix

| Invariant | Executable owner | Minimal counterexample | Closing evidence |
| --- | --- | --- | --- |
| G/H/M are rebased from both current greedy surfaces and K32 owners | census merge | legacy H appears in credit although absent from K32 | focused synthetic merge test asserts exact G/H/M partition |
| cohort eligibility, stable order, and nesting are deterministic | census merge | tied support reverses image-ID order or Nlarge excludes N4 | eligibility/HOLD/nesting tests |
| seed namespaces never overlap | pure seed planner | one K16 cell seed equals census/fixed-eval/other-cell seed | all-pairs disjointness test across cell, step, image, and bank |
| per-image trajectory mean and outer image mean use actual K and N | cell objective | K32 loss divided by 16 or images weighted by row count | sensitivity test with unequal synthetic per-image totals |
| absent preservation rows are a zero term | cell objective | empty mask raises or removes the image denominator | focused zero-preservation test |
| rollback is unconditional, exact, and checkpoint-free | cell runner | exception after optimizer step leaves a changed tensor | injected-failure rollback test plus output-plan assertion |
| dry mode cannot load a model or touch CUDA | CLI planner | `--dry-plan` reaches loader or CUDA APIs | monkeypatched fail-on-load/CUDA test and eight-cell plan assertion |
| Nlarge=N4 does not invent duplicate cells | plan builder | exactly four eligible images produce eight duplicate jobs | plan test expects four cells and explicit collapsed-factor label |
| future outputs cannot overwrite the historical root | CLI/path validator | output equals the 2026-08-21 artifact root | path rejection test and unique per-shard/cell roots |
| capped trajectories use the canonical public-ledger terminal kind | evidence/credit adapter | a capped K trajectory carries `cap` and fails ledger construction | capped-trajectory projection test reaches the supported burden ledger with `cap_stop` |
| non-finite gradients or physical deltas fail before evidence publication | cell update/rollback | NaN gradient reaches AdamW or NaN parameter delta enters JSON | injected pre-step gradient and post-step delta tests raise and restore exact Source |
| every cell artifact binds decision-bearing lineage and accounting | cell result/validator | merged census, seed plan, objective, or burden counters mutate without detection | digest/identity mutation tests plus per-step credit, burden, optimizer-state receipts |

Load-bearing tests for rebasing, seeds, objective accounting, rollback, and
dry planning were required to demonstrate sensitivity by failing against a
deliberate mutant or counterexample before their final green receipt was
accepted. Neither the tests nor the CPU planner code were returned by this
lifecycle change; the contract text above is preserved as the frozen design
record, and the executed evidence it produced is owned by
[results.md](results.md).

## Alignment

The proxy unit is the standalone CPU-census-plus-live-cell probe described
above; the final evaluation surface is exact-HF batch-size-one clean-greedy
decoding at RP 1.0 and RP 1.10 on the same thirteen images used throughout the
Human-13 investigation line. No intervening detector, external owner bridge,
or production DDP/checkpoint path sits between the trained adapter and the
scored surface, so no separate transfer claim is required beyond the
same-checkpoint, same-decode evaluation already performed. Ambiguous evidence
(image-level churn, non-replicate-robust aggregates) is treated as neutral,
not as support for K8, per the Scientific interpretation gate.

## Outline

- Owner surfaces: language-tower DoRA adapter only, vision tower and
  multimodal aligner frozen (inherited from the Human-13 standalone-probe
  design family).
- Reused infrastructure: standalone BF16/FA2 runner
  (`scripts/research/run_human13_standalone_owner_credit_probe.py` at tag
  `probe-final/human13-nk-factorial-probe`), independent of production
  all-HF/CudaAdapter/compiler/transaction/checkpoint/DDP/vLLM paths.
- Non-goals: no production checkpoint promotion, no stable-spec sync, no
  claim beyond the exact Human-13 cohort and this credit-fusion objective.
- Stop rule: any non-finite gradient/parameter delta, inexact rollback, or
  broken dual-RP Source reproduction fails the cell before evidence
  publication (see failure-mode matrix above); scientifically, K-widening is
  stopped by the replicate-inconsistent result in results.md.
- Representative smoke: a one-image standalone existence check at commit
  `32bc918d4` preceded the multi-image factorials.

## Scope

- Checkpoint: exact Source, `Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
  family, language-tower DoRA trainable, vision tower and aligner frozen
  (same Source lineage as the rest of the Human-13 investigation).
- Cases: all 13 Human-13 manifest images for census; N=1 (existence check),
  N=4 (K16/T4), and N=13 paired K4/K8 (T4, replicates r0-r3) for the trained
  cells.
- Changed factor: number of sampled trajectories per image (K4 vs K8) and
  image-cohort breadth (N1 -> N4 -> N13), holding LR, preservation
  coefficient, and update horizon fixed.
- Invariants: fresh AdamW per cell, exact Source restart, no checkpoint
  write, dual-RP (1.0/1.10) clean-greedy evaluation as the decision surface.
- Decode semantics needed to interpret the pilot: clean greedy is exact-HF,
  batch-size-one, at repetition penalty 1.0 and 1.10; sampling is K-trajectory
  vLLM at temperature 0.4, top-p 1.0 (N13 stage) or 0.95 (N1/N4 stage per the
  earlier standalone design), RP 1.0, 512 max new tokens.

## Artifact handle

Logical output roots (verified present on disk at close of this lifecycle
change):

- N=1 standalone result (existence check):
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/full-gpu1.json`
  at commit `32bc918d468baa41fbe218dd81998f86eb5eb226`. The same directory
  also holds `sentinel-gpu0.json` (56.4K) as a companion sentinel run.
- N=4/K16/T4 standalone result:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/multi-image-n4-k16-t4-gpu1.json`
  (373.4K) at commit `b36216f10b89f7a585b2c08dcaed6fe997b0ff9d`.
- N=13, paired K4/K8, four-replicate matrix (decision-bearing evidence):
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-23-human13-n13-k4k8-corrected-geometry-probe/v3`
  at correction commit `a904e3ae38405cf018bccbc176c3497be29313c9`. On disk
  this root contains `cells/`, `logs/`, `preflight/`, and `receipts/`
  subdirectories (eight cell directories under `cells/`, each with its
  canonical `result.json` and terminal receipt, per results.md).

Replay entry point: `scripts/research/run_human13_standalone_owner_credit_probe.py`,
preserved only at tag `probe-final/human13-nk-factorial-probe` (the retired
lane's tip, `a904e3ae3`); it is not present on `research-probes` HEAD.

## Terminology

- **N**: number of Human-13 images in the trained cohort for a given cell
  (N1, N4, N13).
- **K**: number of sampled trajectories per image used to derive reachable
  owner credit (K4, K8, K16, K32 depending on stage).
- **G/H/M**: current-greedy owners, sampling-reachable-but-greedy-missed
  owners, and neutral (unreached) owners, rebased per-cell from the current
  census, as defined in the "Census and credit definitions" section above.
- **RP**: decode repetition penalty; this unit's decision surface is scored
  at RP 1.0 and RP 1.10 in parallel, never RP 1.0 alone.
- **Cell**: one distinct (N, K, replicate) training-and-evaluation unit,
  starting from exact Source with fresh AdamW.
