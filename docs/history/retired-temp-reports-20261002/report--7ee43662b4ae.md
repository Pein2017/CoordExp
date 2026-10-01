# Qwen3-VL Instance Binding Mechanism Report

Artifact root: `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424`

## Current Conclusion

mixed view supported: weak/partial pre-x1 binding exists, but x1/y1 remains the hard instance-disambiguation boundary in difficult same-desc scenes.

Convergence status: `converged_first_pass_mixed_soft_pre_x1_coordinate_hardening`.

## Evidence

- Cases: `64`.
- Cohorts: `{'priority_same_desc': 56, 'sparse_single_instance_control': 8}`.
- Target ordinal histogram: `{'0': 17, '1': 9, '10': 2, '11': 2, '12': 1, '2': 6, '3': 6, '4': 5, '5': 5, '6': 4, '7': 3, '8': 2, '9': 2}`.
- Best pre-x1 probe accuracy: `0.422`.
- Best post-x1/post-y1 probe accuracy: `0.594`.
- X1 target strict-best mass rate: `0.547`.
- X1 target/other mean mass: `0.168` / `0.145`.
- Schema-context patch mean absolute margin delta: `0.068`.
- Previous-geometry patch mean absolute margin delta: `0.002`.
- Current-desc patch mean absolute margin delta: `0.001`.
- Schema-context top-candidate flip rate: `0.375`.
- Donor schema-context mean donor/target mass delta: `0.029` / `-0.064`.
- Donor current-desc / previous-geometry mean donor mass delta: `-0.000059` / `-0.000597`.
- Donor schema-context changed-to-donor rate: `0.196`.
- Good/bad basin proxy labels: `{'ambiguous_bad_leaning_proxy': 9, 'ambiguous_good_leaning_proxy': 17, 'bad_basin_proxy': 20, 'good_basin_proxy': 18}`.
- Rollout failure labels: `{'duplicate_collapse_like': 13, 'healthy_same_desc_multi': 30, 'healthy_single_same_desc': 8, 'near_duplicate_like': 9, 'wrong_or_missing_target_desc': 4}`.
- Rollout healthy/failure-like counts: `38` / `26`.
- Rollout vs basin proxy cross-tab: `{'ambiguous_bad_leaning_proxy': {'duplicate_collapse_like': 2, 'healthy_same_desc_multi': 4, 'near_duplicate_like': 2, 'wrong_or_missing_target_desc': 1}, 'ambiguous_good_leaning_proxy': {'duplicate_collapse_like': 1, 'healthy_same_desc_multi': 13, 'near_duplicate_like': 3}, 'bad_basin_proxy': {'duplicate_collapse_like': 10, 'healthy_same_desc_multi': 7, 'near_duplicate_like': 2, 'wrong_or_missing_target_desc': 1}, 'good_basin_proxy': {'healthy_same_desc_multi': 6, 'healthy_single_same_desc': 8, 'near_duplicate_like': 2, 'wrong_or_missing_target_desc': 2}}`.

## Interpretation

Evidence supports the mixed view rather than a pure H0 or pure H1. Instance identity is weakly decodable before x1, but the pre-x1 coordinate distribution is still multi-modal and only modestly favors the target. Decodability hardens after x1/y1, and rollout contrast contains both healthy same-desc continuations and duplicate/near-duplicate failures. Schema-context attenuation is causally high-impact, so these tokens are not inert punctuation; the remaining uncertainty is whether they carry identity directly or route/read out geometry-bearing state. Donor patching strengthens this: schema-context copies increase donor x1 mass by 0.029 on average and reduce target mass by 0.064, while current-desc and previous-geometry donor-mass deltas stay near zero.

## Next Steps

- Expand the same-desc rollout split beyond 64 cases to check stability.
- Add a wrong-image control for the schema-context attenuation result.
- Repeat donor patching with randomized donor controls to separate content transfer from positional disruption.
