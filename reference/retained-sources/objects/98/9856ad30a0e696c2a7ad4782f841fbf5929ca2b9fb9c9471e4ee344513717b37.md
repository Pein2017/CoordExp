## Corrected broad-v1 saved-output reduction (candidate, unreviewed)

The corrected broad attempt executed the unchanged 45-state panel after the v3 source-route parity gate. It generated 43 new states and reused the 14 corrected-pilot cells. The 43-state run produced **38404 model forwards**, **759 vision forwards**, **253 cells**, and closed with return code 0 across devices 0-7. Eight broad states failed the frozen common-00 admission and therefore have only their 00 receipt; no state was replaced or selected by outcome. The two pilot states remain local HOLDs.

The primary denominator is the 35 admitted broad states (12 failure, 23 proxy). Full panel counts are 21 failure and 24 proxy; both pilot failures and eight broad holds stay in the denominator record but are excluded from the admitted-only contrast. All 253 broad reduced receipts agree with the corrected parser's complete-row count.

### Primary coherent comparison

| panel kind | admitted states | 00 recurrent | 11- recurrent | 11+ recurrent | both coherent signs | either coherent sign |
|---|---:|---:|---:|---:|---:|---:|
| failure | 12 | 12 | 9 | 11 | 9 | 11 |
| proxy | 23 | 6 | 10 | 4 | 4 | 10 |

`10-/10+/01-/01+` are mismatch diagnostics. Their admitted-only recurrence counts are:
| panel kind | 10- | 10+ | 01- | 01+ |
|---|---:|---:|---:|---:|
| failure | 6 | 8 | 10 | 5 |
| proxy | 3 | 4 | 8 | 3 |

These are numerical recurrence predicates. A missing grounding witness limits physical-instance interpretation and does not remove the numerical cell.

### All executed cell accounting

| cell | present | recurrent | exact | near | complete rows | valid | invalid | malformed | row-cap EOS | natural EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 45 | 18 | 14 | 18 | 800 | 755 | 45 | 0 | 12 | 33 |
| 10- | 37 | 9 | 9 | 9 | 428 | 341 | 87 | 0 | 5 | 32 |
| 10+ | 37 | 12 | 8 | 12 | 637 | 612 | 25 | 0 | 8 | 29 |
| 01- | 37 | 20 | 16 | 20 | 773 | 725 | 48 | 0 | 17 | 20 |
| 01+ | 37 | 8 | 8 | 8 | 476 | 463 | 13 | 0 | 7 | 30 |
| 11- | 37 | 21 | 20 | 21 | 704 | 658 | 46 | 0 | 14 | 23 |
| 11+ | 37 | 16 | 14 | 16 | 680 | 621 | 59 | 0 | 10 | 27 |

The recurrence columns count accepted consecutive triple windows at the state/cell level; they are not all-pairs counts. Long runs expose multiple overlapping triple windows, and the compact per-cell records also retain longest exact/near run lengths. Complete invalid boxes remain in the recurrence input and are reported separately.

### Forced-description and opener diagnostics

At the common row boundary, opener/EOS and coordinate-family values are full-vocabulary quantities. Forced x1 old/moved windows are conditional measurements with fixed sign centers; masses are raw full-vocabulary masses and are not window-renormalized. The complete per-cell values are in `result.json`.

| cell | n | median opener log p | median EOS log p | median opener-EOS | median coordinate-family mass | median moved-old log p (- sign) | median moved-old log p (+ sign) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | 45 | -0.00021550717065110803 | -8.454578399658203 | 8.454362892487552 | 6.076196945765944e-14 | -7.15212869644165 | -5.297971725463867 |
| 10- | 37 | -0.002075067488476634 | -6.180227279663086 | 6.178152212174609 | 6.168097174512333e-13 | -4.101774215698242 | -2.2686967849731445 |
| 10+ | 37 | -0.00011193125828867778 | -9.100419044494629 | 9.10030711323634 | 2.187078372139356e-14 | -2.8829216957092285 | -0.5030536651611328 |
| 01- | 37 | -0.00011848701251437888 | -9.048973083496094 | 9.04885459648358 | 4.1397434254446946e-14 | 1.671426773071289 | -4.09572172164917 |
| 01+ | 37 | -0.001547330990433693 | -6.473026752471924 | 6.47147942148149 | 3.7196309400633343e-13 | -4.640170097351074 | 3.607725143432617 |
| 11- | 37 | -0.00031704644788987935 | -8.06246566772461 | 8.06214862127672 | 4.925997692975352e-14 | 6.551761627197266 | -2.3076086044311523 |
| 11+ | 37 | -0.0005907459417358041 | -7.437367916107178 | 7.436777170165442 | 7.900435541234943e-14 | -2.5661849975585938 | 8.792781829833984 |

The sign-matched conditional moved-vs-old **window log-mass** change relative to each state's common `00` baseline is:
| panel kind | mismatch | paired state-signs | median delta vs 00 | positive | min | max |
|---|---|---:|---:|---:|---:|---:|
| failure | 10 | 24 | 3.7219568490982056 | 20 | -4.636860370635986 | 6.4761881828308105 |
| failure | 01 | 24 | 13.099534273147583 | 24 | 5.945224285125732 | 18.582609176635742 |
| failure | 11 | 24 | 16.91586196422577 | 24 | 5.695436716079712 | 21.339067220687866 |
| proxy | 10 | 46 | 6.239798307418823 | 45 | -0.26860880851745605 | 41.71055293083191 |
| proxy | 01 | 46 | 3.892768144607544 | 41 | -2.443003296852112 | 18.376228094100952 |
| proxy | 11 | 46 | 12.303161174058914 | 46 | 0.7732467651367188 | 42.7571074962616 |

These conditional x1 values help separate spatial diagnostics from free trajectory recurrence. They are description-conditioned score shifts, not recovery or physical-owner evidence.


### Geometry and exclusions

The frozen common-canvas rule is lossless copy into the fixed canvas, no resize/interpolation, 32-pixel grid, and horizontal +/-128-pixel displacement. Across the 45x7 frozen manifests there are 3521 supplied history boxes (503 source rows before cell replication), three source-invalid rows, and zero inverse drift above one bin. Five rounding/order entries are retained as a discretization confound:
- `tied-7511-healthy` cell `00` row `10`: source `[716, 575, 717, 596]` -> mapped `[677, 561, 677, 578]` -> inverse `[716, 575, 716, 595]`, mapped_valid=False, order_preserved=False
- `tied-7511-healthy` cell `10-` row `10`: source `[716, 575, 717, 596]` -> mapped `[677, 561, 677, 578]` -> inverse `[716, 575, 716, 595]`, mapped_valid=False, order_preserved=False
- `tied-7511-healthy` cell `10+` row `10`: source `[716, 575, 717, 596]` -> mapped `[677, 561, 677, 578]` -> inverse `[716, 575, 716, 595]`, mapped_valid=False, order_preserved=False
- `tied-7511-healthy` cell `01+` row `8`: source `[644, 575, 645, 597]` -> mapped `[709, 652, 709, 670]` -> inverse `[644, 575, 644, 597]`, mapped_valid=False, order_preserved=False
- `tied-7511-healthy` cell `11+` row `8`: source `[644, 575, 645, 597]` -> mapped `[709, 652, 709, 670]` -> inverse `[644, 575, 644, 597]`, mapped_valid=False, order_preserved=False

The eight broad admission holds are listed in `result.json` with model, kind, source group, row counts, invalid rows, and the exact missing transformed-cell denominator. Candidate-v1 raw/reduced outputs were preserved separately and are excluded from every corrected scientific total.

### Source strata, cost, and replay

State-level source/model/policy/runtime bindings are retained in `result.json`; `val-extra-runtime-stratum`, mature-panel groups, and prospective-new-cohort groups are not silently pooled. Tied versus untied remains a package comparison.

The standalone CPU reducer is `probes/training_set_completion/recurrence_spatial/reduce.py`. The aggregate and artifact map are:
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/result.json`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/artifact-map.json`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/run-receipt.json` and `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/reduction-run-receipt.json`

This is a lane candidate pending parent/root independent verification and acceptance; it does not identify a unique physical instance or causal circuit.

