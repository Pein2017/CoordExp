# Spatial progress gate

Status: `candidate technical HOLD`; lead acceptance is pending.

The frozen admission contained one target-first-revisit boundary, one matched
control boundary, and eight planned scientific cells. Neither allowed
qualification campaign produced an analyzable scientific cell. The first
campaign failed before model entry because it looked up the boundary in the
source panel instead of the frozen shared panel. The repair campaign issued
seven model and seven vision forwards, then failed with `KeyError('tokens')`
in source-trace validation. Both campaigns are consumed.

Lane A therefore has `0/8` executed/analyzable scientific cells. H1 remains
unidentified and untested; this technical failure is not a scientific null and
does not favor H1 because Lane B weakened H2. No release, rerun, replacement
cell, or successor was launched.

Evidence:

- Integrated two-lane candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/candidate-manifest-integrated-v2.json` (`1001bdfb20eaddc9455d4c228c022b876c3ed7f074df834a5f66459698ca9ae4`).
- Reduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/reduction-v2.json` (`a8ea05dc0a3a2d192b7a0894d2b435782ec1d9094b44eeea599ebb204c803aa7`). The preserved v1 reduction sampled the current clock; v2 changes only wall-time derivation to use retained start and terminal records.
- Qualification receipts: `qualification-01/receipt.json` (`41be5b01f853fde2957c1d37ac94ccde4a5a6eba0dfd40578f39702b8cd6cab9`) and `qualification-repair-01/receipt.json` (`8563815249ee7794645b062283f1270ea4fc7250f69f4c4aae7020f41a41b58e`).
- Cost: seven model forwards, seven vision forwards, `22.046531915664673` GPU-seconds, and `4,866,144` retained evidence bytes.

CPU checks:

```text
python -m probes.training_set_completion.spatial_progress_gate.selfcheck
python -m probes.training_set_completion.spatial_progress_gate.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate --out <new-path>
```
