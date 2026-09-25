# Readout geometry

This family owns output-only norm interventions, readout direction/component and
coordinate-margin diagnostics, and tied/untied gradient accounting. Readout
control is not automatically a training-origin explanation or physical recall
improvement. Accepted claims remain in the [capacity/readout question](../../research/questions/capacity-and-readout.md)
and each original experiment record.

## Priority reusable method: coordinate norm equalization

`readout_norm_fresh.py` retains the frozen paired fresh128 producer and CPU
checks. `untied_natural.py` retains the separate mature tied/untied experiment.
Both preserve coordinate-only scaling and non-coordinate logits; their effective
weight composition and identity checks remain profile-specific. The scale uses
the selected rows' median norm, not an implicit mean or an arbitrary new target.

`src.qwen.readout_norm.load_fixed_tied_coefficients` verifies the frozen tied
selected-delta coefficients. It neither selects a dataset nor silently accepts
the untied profile. Concrete model loading is in
[model profiles](../model_profiles/README.md). Input/output embeddings, bias and
effective row identities must still match the chosen method.

Coordinate-margin/pair, first-coordinate and history-readout profiles retain
their own interventions and independent readbacks. CPU arithmetic and captured
state checks are separate from fresh native continuation and physical scoring.

`python -m probes.readout_geometry.readout_norm_fresh --help` describes the
current entry. The existing frozen panel is not an arbitrary-data launcher.
Do not update accepted receipt hashes to current source values. Tests and saved
readback exercise only their declared CPU surfaces; model replay needs its own
inputs, budget and authorization.
