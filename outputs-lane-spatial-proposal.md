# Lane B launch proposal and retained pilot boundary (2026-09-19)

The spatial unit is frozen in `research/experiments/2026-09-19-recurrence-spatial-source/unit.md`.

The retained pilot qualification bindings are the accepted numerical-feedback records
`tied-417044-failure` and `untied-417044-failure` in
`2026-09-18-numerical-recurrence-feedback/selection.json`. They use source row
11 and a 119-token prefix for tied, and source row 16 and a 169-token prefix
for untied; each prefix hash is checked before native execution. The prior
rows0-4 pilot is preserved as a selection-mismatched mechanical receipt.

The concrete transform is a 128-pixel horizontal translation, four 32-pixel
processor cells. For each source `(W,H)`, every cell uses `(W+256,H)`, with the
untouched source copied at x=128 for `00`, x=0 for the negative sign, and
x=256 for the positive sign. No resampling or clipping occurs. History boxes
are translated through the exact canvas-to-bin map using round-half-even; the
same descriptions, order, row count and token count are checked before native
execution. Invalid source boxes are retained and their source/mapped validity
plus any rounding-induced change are recorded. The seven cells are `00`,
`10-/+`, `01-/+`, and `11-/+`.

Qualification uses the two exact mature original-policy model states, with
centered `00` requiring valid complete rows while retaining the frozen
recurrence/proxy classification. A mapped existing-bank match is an optional
grounding witness; its absence limits physical interpretation but does not
discard numerical recurrence. Failure of the recurrence/valid-row gate is an
admission-HOLD and no transform search. I will then use the unchanged
native incremental path with a rebuilt image prefix per cell; forced x1 logits
are stored as conditional diagnostics and free paths remain original RP1,
32 rows/512 tokens/EOS.

This 14-cell qualification is partial Lane B evidence. Lane A has now frozen
the shared mechanism panel; the CPU entry resolved all 45 states and binds the
current pilot only where the exact state appears. Final model calls remain
pending parent device release.
