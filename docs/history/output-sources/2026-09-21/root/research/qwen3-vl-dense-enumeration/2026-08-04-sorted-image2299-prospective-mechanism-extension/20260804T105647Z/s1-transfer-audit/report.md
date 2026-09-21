# Image 2299 sealed calibration-transfer audit

## Primary conclusion

Frozen transfer is **14/19 = 0.737**, below the frozen 0.8 gate.
All **27** native-FN dispositions remain withheld. S2 and S3 do not open.

## Post-hoc TP category sensitivity

- person: 13/16 = 0.812
- tie: 1/3 = 0.333

These are sensitivities under the unchanged frozen gate, not fitted category thresholds.

## Failed native-TP controls at the exact due boundary

| owner | category | due boundary | L failed | U failed | owner rank L/U | category rank | continue-stop margin |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| gt:2299:0 | person | 2299:boundary-001 | peak_lift | peak_lift | 12/12 | 1 | 11.039826 |
| gt:2299:10 | tie | 2299:boundary-007 | local_concentration | none | 1/1 | 1 | 11.839073 |
| gt:2299:25 | person | 2299:boundary-017 | peak_lift | peak_lift | 8/8 | 1 | 2.412645 |
| gt:2299:37 | person | 2299:boundary-020 | peak_lift | peak_lift | 6/6 | 1 | 3.474419 |
| gt:2299:9 | tie | 2299:boundary-005 | local_concentration | none | 1/1 | 1 | 10.569430 |

## Native-FN descriptive dispositions

Role: `descriptive_only_nontransferring`. Counts: `{"persistent_no_tested_localization_support": 11, "resolved_tested_localization_support": 16}`.
These counts are not a valid visual-recall rate.

## Free query-suffix sidecar reachability

The diagnostic uses owner-level any-hit over 50 sealed box sidecars with inherited Canvas norm1000-to-pixel conversion and category-local strict assignment.
- native_tp: 19/19 owners with any hit (person 16/16; tie 3/3)
- native_fn: 1/27 owners with any hit (person 1/22; tie 0/5)

A sidecar miss does not establish that an owner lacks visual support. This free diagnostic does not change the transfer gate or authorize S2/S3.
