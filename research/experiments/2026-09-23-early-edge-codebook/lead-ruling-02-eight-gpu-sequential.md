# Ruling02: eight-GPU training, then eight-GPU inference

2026-09-23. The user explicitly requires all eight GPUs for training and inference
only after training finishes. This supersedes unit.md's four-rank accumulation2
and overlapping training/evaluation clauses. The original unit and launch-v1
remain immutable. All other architecture, objective, population, endpoint,
scoring, evidence limits and resource ceilings remain in force.

## Exact authorized topology and ordering

Use eight training ranks on GPUs0..7, gradient accumulation1, effective batch
eight packs/update. Keep the exact ordered global eight-pack batch for each of
the984 updates, the same seed1729,16epochs,1024 training identities,492-pack cache,
7872 global consumed packs and16384 image presentations. This is a rank/sharding
change, not a doubled batch, exposure change or new optimization setting.
Preserve the nominal per-group LRs, warmup10/constant scheduler, clipping,
three-loss coefficients/global segment normalization and904 trainable tensors.
Do not scale learning rates or silently change packing/cache order.

Training owns all eight GPUs until the fixed fit is terminal and its distributed
process group is stopped/joined. Only then launch native evaluation/inference
workers across all eight GPUs. No overlapping native generation, teacher
evaluation, historical-comparator work or address diagnostic during training.
Checkpoint492 is saved during training; its96-image epoch8 evaluation is queued
until training ends. Final984 remains the endpoint and takes budget priority.
Unrelated CPU preparation can proceed independently. No outcome selects a
checkpoint or changes the ordering. No additional GPU inference qualification
is needed merely because the training topology changes.

## Repair and actual eight-rank qualification

The first four-rank fit is terminal with a later-pack geometry error after
passing its first-two-update instrumentation; no reusable checkpoint exists.
Preserve its launch, logged work, rank receipts, failure and cost. Those receipts
remain four-rank evidence and do not establish eight-rank correctness.

Authorize the narrow edge-arithmetic repair. First reproduce the actual failing
grid/coordinates on CPU. Correct the mathematical footprint construction, not
the accepted geometry: prefer integer patch boundaries divided once by their
extent if split FP32 division/addition is the cause. Do not weaken validation or
apply a broad clamp that hides wrong grids, ordering or image transforms. Verify
the actual failed case, loaded processor patch ordering, endpoint edges and
representative frozen-panel grids; retain the failing-before/passing-after
counterexample. This is technical repair, not a scientific negative or a change
from norm1000. Version source captures and launch/config bindings.

Start a fresh fit from the same mature source with a collision-free output
identity and the same independent P initialization seed. Do not resume failed
in-memory updates or reset the package clock. Existing source-OFF/reload and
unchanged objective evidence may be reused within their boundaries.

Before broad continuation, the first two genuine eight-rank optimizer calls
must verify exact actual global pack IDs/order, global eligible-segment counts,
the actual accumulation1/DDP-eight-rank gradient scaling, independent objective
and gradient accounting, intended trainable coverage/updates, frozen hashes,
and valid LR0 behavior. Reject an incorrect old four-rank/accumulation2 scaling
in the nearest CPU/actual-caller check. Compare against the declared mathematical
global objective; do not require bitwise equality across different GPU topologies.
Use existing numerical acceptance bounds, not a relaxed tolerance. Passing
genuine updates1/2 remain in the scientific trajectory; continue automatically
to984 without waiting for another lead ACK. No replay under a new label.

## Costs, ownership and return

Original model-wall start remains1790131536.7124155. The inspected cost snapshot
already includes1283.9224786758423 allocated GPU-seconds; this is a snapshot,
not a new baseline or cap reset. Count every failed/loading/repair interval.
Retain8wallh/64allocatedGPUh/16GiB, in-flight reservations and7.75h admission
cutoff. If the original envelope cannot complete all queues, preserve unfinished
cells as HOLD and return the conflict; no silent extension or shorter fit.

This ruling authorizes repair and fresh eight-rank production after the above
mechanical checks. Worker922 owns execution and child integration. Send the
corrected eight-rank first-entry evidence, material conflicts and final candidate
directly to the lead. No watcher, self-acceptance, successor, commit or publication.
