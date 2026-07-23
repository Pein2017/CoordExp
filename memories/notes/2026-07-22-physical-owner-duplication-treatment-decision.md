# Physical-owner duplication treatment decision

The bounded duplication unit closed with one treatment worth expanding:
the duplicate-cleaned trajectory training recipe. Equal-update Source-only
controls rule out optimizer-step count alone, but do not match event
composition or Source exposure and therefore do not isolate cleaned semantics
as the sole cause. The recipe increases unique matched owners and reduces
strict duplicate candidates on train-256, on the 240-image never-trained
cohort, and on the disjoint twelve-image human-refined panel.

The result does not support generic terminal suppression, local duplicate
rejection alone, or the combined local-plus-cleaned profile. The combined
profile is harmful on owner coverage and duplication. Cleaned imitation also
has a real counterexample: image `10707` loses a laptop and creates repeated
remote output.

Future expansion must collect and explicitly review more exact self-rollout
trajectories. Do not automatically promote the current expanded overlap queue:
it is concentrated in a few images and unresolved rows intervene in many
candidate suffixes. Delete only reviewed complete duplicate rows, retain only
contiguous reviewed valid suffixes, and keep unmatched official false
positives neutral unless visual review establishes semantic error or entity
hallucination.

Authoritative result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`
