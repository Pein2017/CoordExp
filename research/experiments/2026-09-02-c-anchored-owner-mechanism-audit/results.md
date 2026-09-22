# C-Anchored zero-update owner-mechanism audit results

## Final disposition

- Mechanics: **`MECHANICALLY_VALID`**.
- Science: **`GO_C_ANCHORED_ACTIVE_SET_FEASIBILITY`**.
- Production and projected training: **HOLD**.

The frozen D0 adapter did learn naturally executable target signal on its own
optimization cohort: relative to Source it gained 63 positive-event owners at
IoU50.  It also lost 36 Source preservation owners.  Five of those losses, on
five different images, have an exact behavior-linked margin lesion: D0 reaches
the immutable Source boundary, follows the incumbent row to the actual first
divergence, and at that decision the target margin changes from positive under
Source to non-positive under D0 while aggregate preservation loss decreases.
The supported set fits the registered capacity of 16.

This authorizes only a separate **zero-update C-anchor small-dual feasibility
test**.  It does not authorize projected training, a new screen, an OpenSpec,
or production promotion.  The authoritative receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-owner-mechanism-audit/audit-v1.json`
(SHA-256
`40a51a1e34950b9efaf72f25e648e1c3e2e02764929d98655195ceadb4f163af`).

## Population correction

The earlier `1,955` owner count belongs to the 256-image candidate universe,
not the executed natural-decode cohort.  The eight mechanically excluded
images contain 157 owners, leaving **248 eligible images and 1,798 owners**.
Direct comparison with both authoritative and geometry-sorted annotation views
found no crop or owner loss inside either population.  All natural metrics
below therefore use `248 / 1,798`; excluded owners are not imputed as misses.

## A1: natural realization on the optimization cohort

| Arm | IoU50 | IoU60 | IoU80 | IoU50 owner micro | IoU50 image macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 1,221 | 1,154 | 882 | 0.6791 | 0.7717 |
| C | **1,259** | **1,186** | 896 | **0.7002** | **0.7905** |
| D0 | 1,248 | 1,177 | **897** | 0.6941 | 0.7822 |

At IoU50, D0 gains 63 and loses 36 owners relative to Source, for net `+27`;
all 63 gains are positive-event owners and all 36 losses are preservation
owners.  C gains 67 positive-event owners and loses 29 preservation owners,
for net `+38`.  D0 is `-11` versus C (`39` gains and `50` losses).  Thus D0's
old screen regression was not an optimizer no-op, but C remains the stronger
scalable anchor on D0's own training cohort as well.

| Arm | Positive owners / 576 | Preservation owners / 1,222 | Generated valid rows | Valid unmatched, unknown | Length caps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 1 | 1,220 | 1,712 | 491 | 0 |
| C | **68** | **1,191** | 2,226 | 967 | 3 |
| D0 | 64 | 1,184 | 2,229 | 981 | 1 |

C has 842 geometry-invalid and three malformed dropped spans, dominated by its
three capped trajectories; D0 has 288 geometry-invalid and one malformed span,
with one cap.  Natural ordering violations occur on `25 / 43 / 43` images for
Source/C/D0 and remain monitor-only.  Unmatched valid rows remain unknown, not
hallucinations.

## A2-A4: alternatives and state reuse

- Of 101 multi-owner positive base events, the hard grouped-softmin winner
  changes only six times from Source to D0.  Mean top-candidate weight is
  `0.8153` for Source and `0.8220` for D0.  Raw multiplicity therefore does not
  establish an average-CE conflict and remains descriptive.
- C reaches 507/1,659 Source-bank boundaries with exact tokens.  Another 754
  have the same locked physical-owner set and row boundary but different
  tokens; 396 differ in owner set or boundary and two are absent.  A successor
  must rematerialize from C.  Class 2 is semantic context, not permission to
  reuse an off-policy token prefix.
- C received about 11.9 times D0's row/token supervision.  Its advantage is
  still compatible with coherent scaffold or supervision density; the audit
  does not isolate a pure conditioning mechanism.

## A5: fixed-state margins and exact lesions

On the 576 positive candidates, D0 improves the historical raw loss for 391
and improves the weakest margin for 285.  No candidate moves from an
uncertified to a fully greedy-certified complete row at the fixed Source
state, despite 63 natural target-owner gains.  The useful target signal is
therefore realized along changed natural trajectories rather than as a
complete-row certificate at the stale Source boundary.

Across 1,222 preservation rows, mean historical raw loss decreases from
`1.116968` to `1.092611`.  Nevertheless:

- 560 Source-certified rows become uncertified under D0;
- 421 of those crossings are hidden even within an individually improved
  candidate loss;
- 21 natural owner losses diverge before the tested Source boundary and are
  classified as prefix cascades;
- five owner losses have the exact observed decision crossing required for a
  behavior-linked lesion.

The five exact lesions are on images `101636`, `347671`, `359310`, `360573`,
and `545632`.  Three diverge at coordinate tokens and two at description
tokens.  This is a concrete counterexample to any claim that lower mean
preservation CE guarantees every incumbent greedy decision.  It is not proof
that projection will improve a C-anchored update.

## Mechanics

- Parameter updates, backward passes, optimizer steps, DDP collectives, and
  checkpoints written: **0**.
- Fresh natural decodes: Source, C, and D0; no old Source capture was reused.
- Fixed-prefix forwards: `6,914` (`3,457` for each of Source and D0).
- Decode seconds: Source `1,422.84`, C `2,731.36`, D0 `2,225.39`.
- Peak natural-decode GPU memory: `14,924,666,880` allocated and
  `19,562,233,856` reserved bytes.
- Nine model jobs ran with peak concurrency six.  Host RSS was not captured
  because `/usr/bin/time` is absent; no replacement sampler was added.

Attempt 1 failed before any model process because `/usr/bin/time` was absent;
the unchanged second attempt completed.  Wave-0 scorer status files
misreported 164 images because they recorded shell argument count; the frozen
shard plan and receipts independently prove 82 images per arm.  A wave-1
prelaunch packet used wrong telemetry field names and was superseded before
launch.  These are preserved non-scientific telemetry defects.

The attempt-2 packet also overstates the unit by saying that any completed cap
blocks interpretation.  The research unit's decision matrix is authoritative:
caps are A1/A4 safety monitors here, while identity, row, prefix, adapter,
parser, receipt, or incomplete-artifact failures are mechanically invalid.

## Exactly one next discriminator

Create a fresh research unit for a **no-update C-anchored local-feasibility
test**:

1. rematerialize gain and incumbent states from fresh C natural prefixes;
2. compare the unconstrained C-anchor gain direction with at most 16
   image-balanced critical incumbent gradients;
3. solve only the small dual QP and require a nonzero feasible predicted gain.

If the unconstrained direction does not threaten incumbents, projection is
unnecessary and the route falls back to the simpler refreshed successor.  If
the dual direction is zero, ambiguous, or needs more than 16 constraints, stop
locally.  Only a nonzero feasible direction may justify a later one-update
vertical slice under a separate contract.

## Claim boundary

This audit establishes train-cohort natural uptake, C-prefix reachability,
fixed-Source complete-row margins, and five exact Source-to-D0 behavioral
counterexamples.  It does not establish an internal covered-set
representation, a causal DoRA mechanism, full-scene precision or recall under
missing annotations, held-out generalization, global feasibility, or the value
of projected C-anchored training.
