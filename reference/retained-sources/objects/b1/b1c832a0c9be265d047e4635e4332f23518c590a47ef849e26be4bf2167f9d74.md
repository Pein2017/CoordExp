# Lane C milestone

2026-09-19: execution contract frozen in
`research/experiments/2026-09-19-recurrence-conditional-mass/unit.md` and
`state.json`. The primary conditioning is the native panel prefix plus the
exact serialized current description through `<|box_start|>`; opener and
description-entry evidence remain separate boundary diagnostics. The horizon
is derived from the actual serializer before launch. All 256 draws per state
count in the denominator, including EOS, grammar escapes and invalid extents;
the same-description historical <=8-bin union is deduplicated once, with a
separate literal-invalid flag. Full-vocabulary temperature-1/RP1 native
sampling uses fixed domain-separated RNG streams, no retries or filters.
The scientific sampler batch is frozen at eight; each draw retains its chunk
stream seed and within-chunk offset for exact replay.

Mechanical qualification completed on GPU 6 using the accepted
`untied-417044-failure` source binding from the numerical-feedback selection,
with the corrected boundary after `source_row.end` and the saved `next_row` as
the audited next event. The earlier before-source-boundary qualification is
preserved under `qualification-preboundary/` and is excluded from the unit.
The mature untied loader was required because the checkpoint sidecar is the
declared `input_embed_delta`; the generic shared-delta loader failed closed on
that metadata mismatch and was not used. The successful qualification reports
native prompt/media parity, a finite 152670-way softmax with mass 1.0, row
opener/EOS competition, and saved-native-next-event membership at the exact
conditioned prefix (`true`, description-compatible, legal bbox and terminator).
The required duplicated native batch-4 and scientific batch-8 parity passed
against the accepted source trace: prompt/image grid and dtype identity match,
positions are identical, batch-4 and batch-8 maximum logit deltas are
`3.1471e-05` and `3.5286e-05`, and source-trace raw-logit deltas are
`7.6294e-06` and `2.2888e-05`, under the fixed `2e-4` limit. The corrected
qualifier makes no scientific draws; its final receipt records both batch paths
and the accepted trace.
Artifacts: `outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/qualification/qualification.json`,
`run-manifest.json`, and `self-consistency.json`.

Lane A subsequently published the final `shared-panel.json` and
`shared-sources.json`. The scientific run consumed those exact files and their
recorded hashes: 45 states, 23 split-aware image identities, and 256 draws per
state. The panel remains intentionally imbalanced (21 failures, 24 proxies;
19 untied, 26 tied; all ten prospective proxies tied); it was not rebalanced.
The final raw draw file contains 11,520 records. The first counted batch-8
chunk is recorded in `sampler-entry.json` with the native full-vocabulary,
temperature-1/RP1 settings, stream seed and offsets, token log-probabilities,
and no grammar constraint or retry.

The CPU reducer independently recomputes event status from raw tokens and
frozen history. The primary event is a complete legal bbox plus terminator in
the deduplicated same-description <=8-bin historical union; a draw near
multiple boxes counts once. Image summaries key on bound split-aware
`source_example_id`, never a bare numeric image ID. The result is 441 legal
repeat draws out of 11,520 (q=0.0383 descriptive captured-draw fraction),
alongside 4 invalid-geometry near-repeats, 939 invalid extents and 17 grammar
escapes. No pooled or cross-image interval is claimed; Wilson intervals remain
per-state only, and the image table is descriptive one-image-one-unit output.
The saved native greedy next-row audit has 29 legal repeats and 16 legal
non-repeats and is reported separately from sampled q. `1-q` is not a new-owner
probability and the event is not a physical-duplicate estimate.

Among the ten prospective tied proxies, next-row descriptions differ from the
source-row description for image IDs 185502, 119636, 477785, 328462, 259312,
523815 and 354063. Their same-description unions are empty, so q is
structurally zero under this conditioning. The matching-description proxies
59540, 171564 and 131490 each have a union of size one. Raw draw records omit
`source_policy`; the state-binding crosswalk retains the tied-original policy
and split-aware source identity.

The scientific run recorded 7,290 model forwards, 1,530 vision forwards and
1,440 native generation calls, with 7,708.110 seconds summed per-state elapsed
time on logical GPU6. Qualification receipts account for 13 model and 13
vision forwards across all preserved attempts; the final successful receipt is
5/5. GPU-kernel seconds were not separately instrumented.
Earlier failed-closed launch attempts and corrected qualification attempts are
preserved under `launch-failures.json`, `qualification-attempts.json` and the
three qualification archive directories. No recurrence-mass process remains.

CPU entrypoints pass: `python -m probes.training_set_completion.recurrence_mass.reduce --self-test`, the raw-draw reduction command recorded in `lane-c-gate.json`, and module compilation. The self-consistency record shows native full-vocabulary sampling emits grammar escapes while the constrained-coordinate control emits none; the control is diagnostic only. Lane status is candidate complete with root replay and scientific acceptance pending.
