# Do nearby coordinate inputs stay locally continuous in the trained readout?

## Authority and question

This is Lane D of the released `2026-09-19` recurrence distribution package.
Root owns scientific interpretation, acceptance, and integration. This lane
owns the CPU input/output-row provenance audit, deterministic state selection,
and bounded paired forward captures under its output root. It does not own the
shared recurrence panel, model-loading helpers, the root reducer, or the root
terminal.

The first question is descriptive: after the published step-2444 tied and
untied packages are loaded with their actual base rows and DoRA/special-token
payloads, are adjacent coordinate rows closer or more regular than
distance-matched and non-adjacent rows? The source initializer is audited
separately from trained rows. A smooth initializer is not evidence that
trained input rows, output rows, or greedy choices are smooth.

The paired question is narrower: at frozen failure and non-recurrent-proxy
histories, does changing the last feasible coordinate in the native history
by one bin change the next target-coordinate score through an input embedding
difference that remains visible at a later fixed target boundary? This is an
input-coordinate continuity probe. It is not a physical identity claim,
owner or annotation audit, training-origin claim, causal layer intervention,
or population estimate.

The strongest alternative is that row geometry is locally regular while
context, validity/order constraints, and the decoder's nonlinear state make
the next-coordinate winner discontinuous. The paired captures distinguish
these by preserving native batch, source order, position IDs, companions, and
a fixed continuation suffix.

## CPU provenance and geometry

`probes/training_set_completion/coordinate_continuity/cpu_geometry.py` audits
the actual natural-adjacent initializer and saved tied/untied step-2444
effective tensors. It checks the identities:

```
input_rows = base_rows + input_delta
output_rows = base_rows + output_delta
```

and records actual adapter and special-token payload bindings from
`untied_shared.py` package identities. It computes all 1000 adjacent
distances, distance-matched/non-adjacent comparison distances, row norms,
cosines, and largest local jumps. Initializer reconstruction uses the actual
digit-token mean, structured features, scale, seed, and projection source in
`expand_coord_vocab.py`; it does not assume initializer metadata proves the
trained geometry.

The CPU receipt must include same-anchor witnesses from the same row base or
trained surface, such as a reported d4 versus d499/d999 pair, in addition to
aggregate means. A statement about non-monotonicity is supported by these
same-anchor comparisons and recorded local jumps; it is not inferred from
means over different pair populations. Untied input/output geometry is the
row-wise `||input_row-output_row||` surface, not an adjacent-distance summary
of a difference matrix.

## Frozen panel and deterministic state selection

Lane A supplies the shared panel at:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json`.

No state is launched before that file and its `shared-sources.json` binding
exist. The selector accepts boundaries with `source_row`,
`coordinate_offsets`, `native_tokens`, and `target_slots`. It classifies
`kind=failure` as `failure`; `kind=healthy`, `kind=proxy`, and
`nonrecurrent_proxy` strata as `proxy`. Unknown strata are excluded and
recorded.

The selector takes eight failures and eight proxies, or stops with a
missing-panel/insufficient-stratum receipt. Within each stratum it takes at
most one boundary per `(model,image_id)` scene, alternates available models,
and gives each model four slots when both tied and untied candidates exist.
It fills any remaining slots in stable order without reusing a scene. Stable
order is `(model,image_id,group,boundary_id)` using the panel file name when
present. There is no outcome replacement, random draw, or manual difficulty
choice. Selected state IDs and panel hash are written before any model call.

For each selected boundary, the input site is the latest coordinate token in
`source_row.coordinate_offsets` for which at least one of `value-1` and
`value+1` remains in the coordinate vocabulary `[0,999]`. A site with no
in-vocabulary direction is excluded. Strict `x1<x2, y1<y2` geometry is never
used as a feasibility gate: invalid source histories remain eligible, and a
direction that changes validity or order is retained as a recorded confound.
All in-vocabulary directions and the native value are evaluated. The source
role, absolute native offset, original value, replacement values, vocabulary
validity, geometry validity, and order status are recorded for every variant.

The immediate score site is the first `x1` target slot with `row_delay=0`.
The later score site is the first `x1` target slot with the smallest positive
`row_delay`. A slot's `offset` is the token being predicted, so its logit is
read at the preceding input position (`offset-1`) after the exact native
prefix through that position. A boundary without both sites is excluded. The
fixed suffix for every direction is the unchanged native token span from one
position after the replacement through the later score prefix; no generated
token is sampled or imported from a changed continuation.

This rule keeps the same position IDs and all native intervening tokens. It
compares context and validity/order status explicitly rather than calling
nearby coordinate values physically equivalent.

The selected site role distribution is reported as part of the frozen plan.
If the latest-coordinate rule is dominated by `y2`, that is the observed
scope of this input audit; it is not generalized to `x1`, `y1`, or `x2`, and
no extra role sweep is added to fill coverage. The source initializer's use of
the same embedding construction across coordinate roles remains a source
fact, while contextual sensitivity here is role-specific and conditional on
the selected native histories. Any forward result is therefore conditional on
these frozen failure/proxy histories, not evidence about the unconditional
natural-history distribution.

## Paired forward capture

The producer reuses the accepted `load_model`, `build_bound_native_requests`,
`prepare_native_inputs`, and native full-prefix replay seam. It builds the
actual four-member native group for each boundary. Only the target member's
recorded history coordinate token is replaced; companions, media, masks,
positions, padding, source order, and fixed suffix remain unchanged.

For every selected state and feasible direction, run a native control and an
observational-hook replay. The hook clones values before any in-place
DeepStack update and does not mutate model, cache, scores, or input IDs. One
full-prefix replay reaches the later site and captures both immediate and
later score positions; this avoids rewalking the accepted 3084-token
continuation. Qualification is counted in the same budget.

Each capture records model/policy/image/group/batch identity,
panel/source/receipt hashes, prompt identity, target role and position, native
prefix and fixed-suffix hashes, all companion token identities, native and
replaced token IDs, feasible direction, row validity/order flags, target input
embedding delta, compact signed per-layer residual/input deltas, final
normalized/head input deltas, full coordinate logits when available,
full-vocabulary top competitors, native winner/margin/rank, EOS and
coordinate-family probabilities, emitted/native target token, and
no-hook-versus-hook score/head/input differences with winner parity at both
score sites.

The reducer treats the two boundaries as paired observations. It does not
interpret companion outputs as independent outcomes or infer causality from a
logit change alone.

## Qualification, budget, and stop

The first selected state is the first balanced failure state in stable
selector order. Its native and all in-vocabulary replacement directions
qualify before broad execution. At the real entry, each native full-prefix
replay must agree with the saved native trace at both score sites: exact
winner and top-two logits within the established FP32 absolute ceiling
`2e-4`. Uninstrumented and observational-hook runs must also have the same
target scores and winners. Source prompt/media/position identities and
unchanged companion tokens must match exactly. A failure is preserved and
stops this lane; tolerance is never widened. Hook/no-hook self-parity alone
does not qualify the full-prefix/native seam.

The hard ceiling is 16 states, at most three variants per state (native,
minus-one when feasible, plus-one when feasible), two replay modes, and one
full-prefix call per variant/mode. The declared upper bound is therefore 96
native batch forwards, below the 512 additional-forward budget including
qualification. The full-prefix horizon is the later target prefix only: no
free generation, sampling, EOS change, cache transplant, mask change, or
3084-token continuation replay. This lane uses only GPU 7, with a provisional
ceiling of two GPU-hours and four GiB retained tensors within the parent's
24-GPU-hour total.

Stop after selected states, captures, and CPU parity/reconstruction complete.
Stop with `NEEDS_LEAD` on missing or changed source bindings, unsupported
target geometry, native/full-prefix parity failure, or budget conflict. Do
not add states, layers, coordinate doses, models, checkpoints, training,
annotations, visual review, integrated terminal, package events, or a
successor unit.

## Owned artifacts

This lane owns this unit and state record, the task-local probe directory,
and artifacts under:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-input-continuity`.

The parent owns the shared panel and final integration. This lane returns the
CPU geometry receipt, frozen selection/execution plan, per-state captures, and
actual process/cost receipts; it does not write the parent terminal or
integration event log.
