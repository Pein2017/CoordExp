# Signed native historical-read influence at the first coordinate fork

Frozen successor to the CPU-only [design](unit.md), 2026-09-24.
Lead admission is limited to the first three calls below; remaining seven
are a fixed proposal requiring lead qualification review. No user permission
renewal is needed inside the user's existing autonomous research authority.

## Decision

At the first confirmed bowl revisit, does withholding the latest completed
row from current-row attention favor the fixed new-owner coordinate or the
old-owner coordinate, with native tokens and positions preserved?

The direct-copying explanation predicts that withholding that row raises
new-minus-old preference. The opposing account predicts that the historical
read helps progression, so withholding it lowers new-minus-old preference.
An indifferent response leaves this specific direct-read route unresolved.
This is a prospective directional test at a native state, not another
artificial rotary displacement or a claim that one attention source is faulty.

## Frozen sources and finite support

Use the accepted candidate registry
`supporting/native-window-candidates.json`, SHA256
`dcdc2f20a6dc64b6ec1110faf7cc2a7c4e4cf3dcbd4355c9ee2338b4c07996c4`,
bound to original mature untied native raw/trace/receipts and the independently
loaded input/output deltas. Preserve checkpoint, processor, prompts, original
four-request order, padding/EOS behavior, dtype and attention implementation.
Do not import or execute retained source captures.

All offsets here are raw target-completion indices; preparation must crosswalk
them to actual padded physical slots. End-exclusive ranges are used.

| State | Native target / batch | Current-row prefix, before x1 | Latest completed row | Optional earlier-row comparator | Fixed old / new x1 IDs |
| --- | --- | --- | --- | --- | --- |
| Bowl revisit | mature313465 row1, fresh-18 / 3 | [10,15) | row0 [0,10) | none | 151675 / 151827 |
| Bowl progress | same source row2 / 3 | [20,25) | row1 [10,20) | row0 [0,10) | 151675 / 151827 |
| Cow progress | mature479075 row1, fresh-26 / 3 | [9,13) | row0 [0,9) | none | 151823 / 151980 |

Bowl fixed extents are native row1 A `[5,197,217,407]` and native row2 B
`[157,0,415,241]`; B is unvisited at both bowl states. Cow extents are native
row0 A and native row1 B. This finite pair is not total physical-owner mass.
Later native B is retrospective support, not held-out discovery. Bowl versus
cow is not a matched causal cohort. Bowl state differences include history
content and position; the within-state attention contrast is the intervention.

## Exact intervention and execution route

Use one original-source-shaped full-prefix forward per cell, ending before
the native x1 is generated. Prefer the maintained full-forward preparation
already used by first-arrival probes; no cached/full splitting is needed.
Every cell recomputes the original image and complete prefix. No free tokens,
forced alternative x1, coordinate-row scores, model gradients or training.

For the target request only, on all 28 text layers and all heads, make the
selected completed-row keys inaccessible to every query in the current-row
prefix. Modify only those entries of the actual causal attention mask. Keep
all historical-query rows, image/prompt columns, other generated-row columns,
current-row causal/self entries, and every companion unchanged. Use the native
mask's blocked-value convention. No token, position, K, V or parameter patch.
Historical states must remain identical; current-row Q/K/V and downstream
states may respond to the intervention and must not be described as clamped.
Softmax redistribution over the remaining keys is part of this intervention;
it is not an isolated value-copy estimate or complete erasure of history.

Per state execute native, identity-mask sham, and latest-row mask, in that
order. At bowl progress also execute the earlier-row mask after qualification.
The total proposed panel is 10 model / 10 vision full-prefix forwards:
3 bowl-revisit, 4 bowl-progress, 3 cow-progress. Only the first 3 are admitted
now. If the live model route cannot consume this exact mask without changing
attention implementation or source semantics, return the conflict before GPU.

## Primary readout and falsification

Let M=z(new x1)-z(old x1), and delta=M(masked)-M(native), using raw logits.
Freeze a numerical interpretive deadband of 0.01 nat: delta>0.01 means this
removal favors the new coordinate; delta<-0.01 means removal favors the old;
otherwise it is numerically indifferent for this assay, not a statistical null.
Both signs are informative; do not tune layers, masks, thresholds or cases to
obtain a preferred direction.

The first-case direct-copy prediction is delta>0.01. A negative first-case
delta rejects that directional prediction for this intervention and endpoint.
It does not exclude history carried indirectly in other contextual states.
After separate admission, compare the two bowl deltas and report their signed
difference; report the cow and equal-size older-row comparator separately.
Similar normal-transition effects weaken failure specificity. Do not convert
three states into population rates or call a difference a coveredness effect.

Always retain full-vocabulary native/sham/masked vectors, old/new P/logP/rank,
global winner/runner/gap and full-vocabulary TV. A changed pairwise margin can
coexist with a third global winner. No physical owner switch, complete-box
realization, natural onset mechanism or recovery is claimed without a later
separately frozen continuation test. This test determines whether that next
step has a useful directional basis.

## Mechanical qualification, bounded cost and stop

Before GPU, bind current producer/import bytes, original full-batch source
identity and observed geometry; freeze exact commands and a shape-aware cost
estimate. Use a small CPU fixture on the actual mask-edit caller to reject
wrong target, row span or query range, unwanted companion/historical edits,
and mutation of the source mask. An identity edit must exercise the same path.
No generic hook framework or large tensor atlas is needed.

Native source chosen/top2/logsumexp parity is required on target and active
companions at 2e-4. Ended companions retain original EOS/pad inputs; their
after-EOS source trace is undefined. Native/sham full-vector parity is required
on all four batch rows at 2e-4 before the mask arm. Treatment companion vectors
must remain within 2e-4 of native. Record actual mask consumption at all 28
layers, exact native tokens/positions, selected query/key rectangles, unchanged
complement hashes and historical-state equality. Retain compact observed mask
windows plus reconstructible mask evidence, not 28 copies of a full mask.

Save raw vectors and consumer evidence before reduction; a separate CPU
readback verifies the bindings and recomputes the signed endpoint. Failure is
technical-invalid, not a scientific negative; preserve the attempt and charge
all setup/diagnostics. No silent retry or extra forward.

First-case cap: 360 allocated GPU seconds including setup/failures/diagnostics.
Proposed entire panel cap: 900 seconds, still inside sequence cap 8 GPU-hours.
Prior accepted sequence charge is 0.2381858424494664 GPU-hours, bound by the
predecessor y1-value three-case ledger. First-case output planning limit is
64 MiB, whole-panel 256 MiB. Measure actual wall time, allocated GPU interval,
model/vision call counts, RSS/GPU peaks, artifact bytes and terminal job state.
Reforecast remaining seven after the first case; do not launch them yet.

Worker owns `probes/training_set_completion/recurrence_native_history_read/`,
this unit's new attempt-specific supporting records and candidate report, and
raw output under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-attempt-001/`.
No Git action, predecessor/peer/shared-runtime edits, new sample, or successor.
Return the three-call candidate or the first decision-bearing failure directly
to lead; only lead can accept or admit the remaining fixed cells.
