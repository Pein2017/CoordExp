# Lane C result record

Status: **candidate; root acceptance pending**.

The complete scientific result is preserved at the output-root artifact below.
This repo-owned record is the authoritative pointer required by `state.json`;
it does not duplicate the raw draws or replace the standalone reducer.

## Frozen evidence

- Unit: `2026-09-19-recurrence-conditional-mass`.
- Panel: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json`, SHA256 `005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`.
- Sources: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-sources.json`, SHA256 `8f7b43d2d9f4014bb42024324543eea27bad222b6c8432d3e44c93b45743fe9e`.
- Bound states/images: 45 states, 23 split-aware image identities; 21 failure states and 24 proxy states.
- Sampling denominator: 45 × 256 = 11,520 original-policy native draws. Every draw, including grammar escapes and invalid extents, remains in the denominator.

## Result and uncertainty boundary

The legal numerical-repeat event occurred 441/11,520 times, a descriptive
captured-draw fraction of `q=0.03828125`. The full outcome table, per-state
Wilson intervals, greedy next-row audit and image-level descriptive summaries
are in the [output result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/results.md).

The pooled fraction and model/source strata are not treated as iid draws from
one common q, so no pooled or cross-image confidence interval is claimed.
Wilson intervals are retained only per state. Image summaries give each bound
image one unit and are descriptive. The event is numerical membership in the
frozen same-description `<=8`-bin union, not physical duplicate probability;
`1-q` is not new-owner probability.

The ten prospective proxy states are all tied. Seven states use a next-row
description different from the source-row description: image IDs `185502`,
`119636`, `477785`, `328462`, `259312`, `523815` and `354063`. Each has an
empty same-description historical union (`repeat_union_size=0`), so q is
structurally zero there; this is a conditioning/denominator property, not
model suppression. The remaining prospective proxy IDs `59540`, `171564` and
`131490` have matching descriptions and a union of size one. Raw draw records
leave `source_policy` null; `state-bindings.json` and the frozen panel/source
manifests provide the canonical tied-original crosswalk.

## Cost and exclusions

The scientific run recorded 7,290 model forwards, 1,530 vision forwards and
1,440 native generation calls, with 7,708.110 summed per-state elapsed seconds
on logical GPU 6. Qualification receipts account for 13 model and 13 vision
forwards across the preserved pre-boundary, pre-batch-4, pre-stream and final
qualification attempts; the final successful receipt is 5/5. Failed launch
guards produced zero scientific draws and are preserved in
`launch-failures.json`. No owned process remains.

## Acceptance and replay

Root owns scientific acceptance. The raw draw file, state/source bindings,
sampler-entry receipt, preserved prior result, reducer output and launch
receipts are under:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/`

CPU acceptance commands:

```text
python -m probes.training_set_completion.recurrence_mass.reduce --self-test
python -m probes.training_set_completion.recurrence_mass.reduce --draws /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/draws.jsonl --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/reduction.json
```

No policy promotion, physical-identity claim, successor launch or new sampling
arm is authorized by this record.
