# Current Project Memory

Last verified for the physical-owner duplication unit: 2026-07-22T21:25:00Z.

## Closed result: physical-owner duplication

The physical-owner duplication unit is complete. Equal-update Source-only
controls show that the duplicate-cleaned trajectory training recipe, rather
than extra optimizer steps alone, produces the only favorable bounded result.
Relative to its 6-update Source control it is `+4` matched unique owners and `-51`
strict duplicate candidates on train-256, including `+3` owners and `-48`
duplicates on 240 never-trained images. On the disjoint twelve-image
human-refined panel it is `+3` owners and `-9` duplicates versus the equal-update
control, and `+4` owners and `-3` duplicates versus frozen Source.

These controls repeat Source rows, so they do not match event composition or
Source exposure and cannot isolate cleaned semantics as the sole cause. The
complete recipe is promising but not uniformly safe: image `10707` loses a
laptop and
develops a repeated remote row. Recovery-positive and local rejection reduce
duplicates but often lose owners; the combined local-plus-cleaned profile is
rejected. Do not automatically promote the current expanded overlap queue:
most extra immediate-recovery cases are concentrated in one image, and many
longer candidates contain unresolved intervening rows. The next scale axis is
more exact self-rollout trajectories plus row-level physical-owner review.

Authoritative result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`

Parallel breadth-screen and Pi-worker state below was not revalidated by this
closeout.

Last verified for the breadth screen: 2026-07-22T17:17:09Z. The previously recorded eight sampled
workers were no longer running, and the trajectory-panel directory still had
only short model-load logs with no completed sampled JSON shard. Treat that
breadth-screen launch as interrupted and re-audit it before any continuation.

The parallel Pi worker-ablation execution state was last verified at
`2026-07-23T01:35:06Z`; the port-9090 infrastructure rerun is complete and no
Pi process is live. The investigation used no graphics-processing unit.

## Active objective

Run the constant-dose image-breadth treatment screen for Qwen3-VL dense object
enumeration. Hold sampled-route event count, Source-preservation event count,
optimizer updates, object-count-band allocation, and event difficulty as fixed
as feasible while comparing supervision concentrated in 118 images with the
same dose spread across 496 images.

This is a bounded test of whether narrow image breadth caused the previous
owner exchange. It is not full-size promotion and not an architecture
commitment.

## Closed parallel objective: physical-owner duplication

The user authorized an independent long-running goal that proceeded in
parallel with the interrupted breadth screen:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/unit.md`

The frozen question is whether generated history causes repeated selection of
one physical instance and whether training can reject the repeated row while
recovering into a later valid, uncovered owner. Official unmatched predictions
remain neutral unless review confirms a category error or entity hallucination;
verified false negatives and physical-owner duplicates are actionable signals.

The dedicated implementation contract is:

`openspec/changes/add-physical-owner-duplicate-rejection-and-recovery-training/`

The OpenSpec proposal, design, capability spec, and tasks are complete and
strict validation passes. Planned controls and treatments are frozen Source,
Source-preservation-only, recovery-positive-only, local duplicate rejection
and recovery, duplicate-cleaned counterfactual trajectory imitation, and their
combined profile. Diagnosis and training proceed in parallel; causal
results control interpretation rather than canceling a valid training smoke.

The initial 256-image census found 113 annotation-anchored repeated-owner rows
in 38 trajectories. Thirty-four trajectories later recover an uncovered
annotated owner. The safest initial subset is 34 near-exact repeated rows in 12
images. The existing 144 geometry-derived ambiguous overlap candidates remain
excluded from automatic duplication supervision.

## Parallel bounded investigation: Pi external worker

Pi `0.81.1` is installed in a non-global output prefix. Four read-only Stage 0
task fixtures and hidden verifiers are frozen under:

`/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/fixtures-v1/`

The research owner is:

`research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/unit.md`

The user completed worktree-local OAuth, specified the required port-9090
proxy, and authorized reopening the twelve Pi cells. All worktree-local Bash
and Pi entry points now export the proxy through `127.0.0.1:9090`. The rerun
completed with positive token usage in all cells and no workspace mutation.

Luna, Terra, and Sol all pass artifact inventory and mechanical aggregation.
The audited Task 2 verifier passes Luna and Terra; Pi Sol and native Sol share
one exact `HFBackendSession` capitalization error. All Pi Task 3 cells have the
correct verdict, identifiers, and logical argument but fail the original
lexical limitations checks. Native Sol remains three of four under the
original frozen verifiers.

Stage 0 supports a larger frozen benchmark for mechanically verifiable tasks,
not an adapter or default route. Native Codex lacks comparable token and cost
receipts, so no total-cost advantage or causal harness effect is established.
See `memories/notes/2026-07-23-pi-stage0-proxy9090-rerun-result.md`.

## Most recent closed evidence

The 118-image Source-route-preservation screen is complete. Its treatment was
learnable but narrow:

* selected Source-missed owners were recovered approximately four to six times
  as often as non-selected missed owners;
* admitted images could gain owners and box quality and standard detection
  metrics improved;
* non-admitted images lost owners at every milestone;
* more updates did not repair transfer.

The result supports route-conditioned owner redistribution, not safe final-set
expansion. The two live alternatives are insufficient image breadth versus an
intrinsic owner-exchange limitation of positive complete-row imitation.

## Frozen current unit

The unit is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`

The 2,432-image label-only pool was frozen before route inspection and split by
image identity and object-count band into:

* 2,048 training-candidate images, 512 per band;
* 256 development images, 64 per band;
* 128 held-out images, 32 per band.

Each treatment arm will contain 496 sampled-route rows plus 496 exact Source
rows: 992 unique events, 31 optimizer updates, and effective event batch size
32. The broad arm uses 496 physical images; the concentrated arm uses a nested
118-image subset. The twelve human-refined images remain safety evidence only
and never supply gradients.

The selector must record one of three matching modes:

1. exact object-count-band by selection-rank matching;
2. coarse object-count-band by rank-one, rank-two, rank-three, and
   rank-four-or-deeper matching;
3. policy-only rank-one fallback.

Interpretation must follow the emitted mode. Only exact matching supports a
breadth-effect claim at fixed ordinal-rank distribution. Coarse matching has a
weaker claim, and policy-only fallback cannot identify physical image breadth
as the cause.

## Breadth-screen execution state

The missing 2,176-image sampled panel is running at:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-v1/`

The previously recorded workers are no longer present. Each shard log ends
after model loading and no sampled JSON shard is present. Do not describe this
panel as running or complete. Its intended request semantics remain one image,
temperature 0.4, nucleus probability 0.95, repetition penalty 1.0, 512 generated
tokens, and per-image seeded reset, but a fresh continuation receipt is needed
before relaunch.

## Current understanding

Established or bounded-supported:

* Low-temperature sampling exposes real physical owners that greedy decoding
  misses.
* Prefix order is causally active; no order-free covered-object ledger has been
  demonstrated.
* Entity discovery and complete box geometry are separate outcomes.
* Fixed-prefix coordinate correction can be learned without improving clean
  rollout.
* Positive complete-row imitation can strongly favor selected owners while
  exchanging away other Source owners.
* Standard mean Average Precision can improve even when unique physical-owner
  coverage does not.

Tentative or unresolved:

* Whether greater physical image breadth turns narrow owner redistribution
  into a transferable rule.
* Whether complete-row positive imitation intrinsically exchanges owners even
  after matching training dose and event difficulty.
* Whether any explicit covered-set carrier, object slot, cursor, or other
  architecture change is necessary.

Rejected or held:

* Do not interpret longer output, terminal suppression, selected-owner
  recovery, or a mean Average Precision gain alone as set expansion.
* Do not scale the previous 118-image treatment unchanged.
* Do not add a second cohort, external detector, object slot, terminal
  suppression, canonical supervised-fine-tuning mixture, or online refresh to
  this pilot.

## Immediate next actions

1. Treat the physical-owner duplication unit as closed bounded evidence; do not
   promote the combined profile or automatically expand the current overlap
   queue.
2. If duplication work continues, open a separate expansion unit for more
   exact self-rollout trajectories and row-level physical-owner review.
3. Treat Pi Stage 0 as closed; require a new frozen benchmark before adapter or
   default-route promotion.
4. Re-audit the interrupted breadth-screen launcher and artifacts before any
   relaunch or claim about its execution state.
5. Preserve separate-thread work by intent and verify the live worktree before
   staging, committing, or resuming any unit.

## Minimum reading path

1. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/unit.md`
2. `openspec/changes/add-physical-owner-duplicate-rejection-and-recovery-training/design.md`
3. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/results.md`
4. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`
5. `research/investigations/qwen3-vl-dense-enumeration/compass.md`

The comprehensive recap remains the long historical reconstruction. The
breadth-screen checkpoint is historical evidence for an interrupted run; the
closed physical-owner and Pi notes above are the current operational handoffs.
