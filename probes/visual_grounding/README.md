# Visual grounding

This family owns visual-instance binding and fixed-grid visual-detail dependence.
The profiles ask whether designated visual evidence is used under controlled
conditions, not whether every unmatched prediction is a false object. See the
[visual causal-use question](../../research/questions/visual-designation-and-causal-use.md)
for interpretation and the original unit for its frozen conditions.

`visual_instance_binding` retains compositing, runtime and independent reduction
and verification. `visual_detail_dependence` retains preparation, availability
and qualification. Their image/grid, crop, composition, conditional-score and
control definitions remain local to the profile instead of becoming global knobs.

Common model loading and mature saved-source reconstruction come from
[model profiles](../model_profiles/README.md); native input and history operations
come from `src.qwen` and `src.inference`. No other research family's runner is a
utility dependency. Tests live alongside the profiles and use local CPU stand-ins.

The family move changes current import paths, not old research records, image
identity, geometry, output payloads or interpretation. Existing raw-data readers
remain distinct from producer summaries; model qualifications are not rerun by
architecture validation.
