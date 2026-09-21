# Lane B: visual position × history position

**Lead acceptance (2026-09-19):** Corrected 45-state spatial contrast accepted: 35 admissions, 10 local HOLD, 267 cells. Conditional visual/history sensitivity and persistent free numerical recurrence; no physical recovery or unique circuit. Original invalid attempt remains excluded. [Independent receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/integration/lead-acceptance.json). Candidate snapshots and raw evidence are preserved. No successor is released.

## Corrected broad-v1 saved-output reduction (lead-accepted, bounded)

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

## Technical source-route parity gate: selector correction pending

The saved Lane-C qualification reuse for untied `417044` passed exact
source/prefix, native-token, raw-trace, image, prompt, grid, and batch-1/4/8
winner/logit checks. Its source-row boundary is 169. The tied v1/v2 source-route
artifacts are preserved byte-for-byte, but the v2 free-row comparison is not a
valid parity result: `run_gate.py` selected `source_logits[target_index, -1]`
and `target_logits[0, 0]` for `row_boundary`. Both routes must use the final
position of their compact six-position output for the same completed prefix.
The forced-description x1 stage already used `-1` on both routes.

A CPU-only six-position, position-coded fixture reproduces the defect: the old
selector creates a winner mismatch and a `19.0` max delta, while the corrected
`target_logits[0, -1]` selector matches the winner with a `1.0e-05` delta under
the frozen `2e-4` tolerance. Therefore the prior tied row-boundary values
(`30.5304298401`, winner `151646` to `151670`) are an invalid-selector result,
not a verified runtime parity failure. The forced x1 result remains a valid
position-`-1` check, but the target-only free row-boundary position-`-1` full
vocabulary tensor was not saved, so CPU cannot repair that comparison.

The bounded tensor audit found no exact full-vocabulary gate tensor in the v2
artifact directory. Closest saved captures are coordinate-only tensors from
other routes and cannot establish this target-only versus heterogeneous batch-4
comparison. Root then released exactly two additional tied forwards after the
parent-owned CPU consumer/readback gate passed. Those two forwards are recorded
below in technical-gate-v3. No forced x1, untied rerun, generation,
replacement, or spatial cell was added.

The prior eight forwards remain provenance only: v1 four mechanically invalid
readback forwards and v2 four forwards whose row-boundary selector was wrong.
The 45-state panel remains unchanged with 43 states unexecuted, and all 14
corrected pilot cells remain preserved. Lane-B broad launch remains pending
the parent/root decision despite the corrected route parity result; no spatial
recurrence claim is made from this technical gate.

Correction artifacts:

- Selector receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/selector-correction-receipt.json`
- CPU falsification: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/selector-falsification.json`
- Correction diff: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/selector-correction.diff`
- Saved-tensor audit: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/saved-tensor-audit.json`
- Preserved preliminary v2 comparison: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/comparison-tied.json`

CPU falsification command:

```bash
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/selector-falsification.py
```

The former `cpu-acceptance.json` and `reduce_gate.py` output are retained as
superseded records; they must not be used to claim a verified parity failure.

## Corrected v3 source-route capture: parity passed

The parent CPU consumer gate passed its archived position-coded RED/GREEN
checks before launch. The v3 producer then ran exactly two tied forwards on
GPU0: one original heterogeneous batch-4 source route and one original-image
target-only batch-1 route. Both captures were saved before comparison with
`logits_to_keep=1` and full FP32 vocabulary vectors.

The capture consumer passed with max full-vocabulary delta
`3.9458275e-05` (tolerance `2e-4`), source and target winners both `151646`,
and saved-source trace delta `1.5258789e-05`. The source/target target-slot
input IDs and position IDs match after removing padding; prompt count is
`1320` on both routes, image grid is `[1,54,72]` on both, and media SHA-256 is
`66b4ed305acb5aa1c198309d4ce0ec6ed5edb25563ae8e9c2b04574ef18bda62` on both.
This closes the technical source-route parity gate for this tied state. It does
not authorize or answer the spatial recurrence experiment; parent/root owns
the decision to release the remaining 43 states.

The v3 capture used 2 native model forwards, 0 generation calls, `3.0708867`
measured GPU seconds, and `8.9505216` seconds enclosing load/prepare wall
time. Closure found no owned producer or consumer process and no remaining
compute application. The 45-state panel denominator is unchanged: 2 pilot
states preserved, 43 future states unexecuted, 0 new spatial cells, 0
replacements, and 0 `00` reruns.

V3 artifacts:

- Capture receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/capture-receipt.json`
- Source capture: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/source.pt`
- Target capture: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/target.pt`
- Consumer result: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/compare.json`
- Capture integrity: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/capture-integrity.json`
- Pre-call snapshot: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/capture-snapshot.json`
- Device/job closure: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v3/closure.json`

## Attempt ledger and budget reconciliation

The standalone attempt ledger preserves every invalid, superseded, corrected, and
not-run denominator without treating an invalid run as a scientific null:

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/attempt-ledger.json`
- Candidate-v1 qualification receipt: 7,712 attempted model forwards and 72 vision forwards; its final 14-cell reuse denominator is 4,256 model forwards and 42 vision forwards, leaving the receipt-defined 3,456 model and 30 vision forwards outside that final denominator. The qualification rows share receipt paths, so they are not independently re-summed.
- The candidate-v1 cost summary separately reports 40 zero-forward missing-`image_plan` planning failures (20 first-batch plus 20 rerun-v2) and 20 corrected new-cohort state attempts.
- The standalone corrected pilot-v2 receipt reports 4,192 model forwards and 42 vision forwards. The aggregate cost receipt reports `pilot_model_forwards=4256`; both fields are retained with their source scopes, and the 64-forward difference is not inferred or silently reconciled.
- Technical gates add 4 invalid v1 forwards, 4 selector-bug v2 forwards, and 2 corrected v3 forwards. V3 measured 3.07088671875 GPU seconds and 8.950521640479565 seconds enclosing load/prepare wall time.
- The unchanged panel has 43 future states and zero additional forwards. The conservative enclosing budget is 14.601111111111111 GPU-hours before the released corrected attempt, an 8-hour corrected-attempt allowance, and a 22.60111111111111 GPU-hour projected package upper bound; these are reservation bounds separate from measured GPU seconds.

The final aggregate cost receipt reports 42,270 model forwards, 801 vision
forwards and 4,408.466894 elapsed seconds. It overlaps the preserved
qualification denominators and is retained as an aggregate summary rather than
added to the attempt rows.

## Corrected pilot-v2: bounded HOLD

The parent released one corrected, production-shaped pilot on 2026-09-19. It reuses the frozen first-pilot boundary (`refined-03`, image `417044`) for one tied and one untied failure state, with no outcome-based selection or replacement. The immutable source snapshot was captured before model calls. The full 45-state panel was nevertheless constructed first: 45/45 states, 315/315 cells, 135 images and 3,521 history rows; 21 invalid source rows were retained, with zero non-coordinate drift, y-role changes, crop mismatches or source-identity mismatches. Only the two released states received model calls.

Both common-`00` admissions held under the corrected consecutive-triple predicate for the two executed pilot states. This is a local pilot HOLD, not an all-panel closure or an inference that relevant failures vanish across the unchanged 45-state panel. The six signed cells per model were run with the fixed `skip_gate` diagnostic flag after the unchanged `00` hold; they expose mapped recurrence, mismatch behavior and logits without rescuing admission. No `00` rerun, replacement, tuning or broad panel launch occurred; the remaining 43 states are pending the parent runtime gate.

### Geometry and source binding

The rule is a lossless copy of the 1,152×864 scene into a fixed 1,408×864 black canvas, with no resize or interpolation, 32-pixel grid and ±128-pixel horizontal displacement from the centered 00 origin. All four history coordinates receive the same horizontal affine map; y roles remain unchanged. Raw invalid rows remain in the history, and free outputs are inverse mapped to source bins.

| cell | visual offset | history relative shift | interpretation |
|---|---:|---:|---|
| `00` | 128 px | 0 px | common centered baseline |
| `10−` | 0 px | 0 px | visual-only left mismatch |
| `10+` | 256 px | 0 px | visual-only right mismatch |
| `01−` | 128 px | −128 px | history-only left mismatch |
| `01+` | 128 px | +128 px | history-only right mismatch |
| `11−` | 0 px | −128 px | coherent left translation |
| `11+` | 256 px | +128 px | coherent right translation |

The native target-only request route is bound to `build_bound_native_requests`; the original heterogeneous companion rows are retained as provenance, while each transformed target rebuilds its own image and prefix. Every model cell kept one prompt identity, `[1,54,88]` image grid and input-id tensor hash. `00`/`01` and `10`/`11` media pairs matched exactly; the distinct visual offsets had distinct media identities.

### Admission, recurrence and mapped coordinates

| model | common `00` rows | valid / invalid | known witnesses | failure predicate | admission |
|---|---:|---:|---:|---:|---|
| tied | 31 | 31 / 0 | 24 | false | HOLD |
| untied | 26 | 26 / 0 | 18 | false | HOLD |

The transformed recurrence outcomes are diagnostic because both baselines held:

| model | `00` | `10−` | `10+` | `01−` | `01+` | `11−` | `11+` |
|---|---:|---:|---:|---:|---:|---:|---:|
| tied | no | no | no | yes | no | yes | no |
| untied | no | no | no | yes | no | yes | yes |

Here “yes” is the corrected failure predicate (a complete legal box recurrence with the frozen consecutive-triple exact/near primitive). Exact/near triple counts and row validity are retained below; long runs therefore have more triple edges than unique repeat anchors.

| model | cell | exact triples | near triples | complete rows | valid / invalid | injected row-cap EOS |
|---|---|---:|---:|---:|---:|---:|
| tied | `00` | 0 | 0 | 31 | 31 / 0 | no |
| tied | `10−` | 0 | 0 | 30 | 29 / 1 | no |
| tied | `10+` | 0 | 0 | 32 | 32 / 0 | yes |
| tied | `01−` | 23 | 26 | 32 | 32 / 0 | yes |
| tied | `01+` | 0 | 0 | 32 | 32 / 0 | yes |
| tied | `11−` | 13 | 21 | 32 | 32 / 0 | yes |
| tied | `11+` | 0 | 0 | 32 | 32 / 0 | yes |
| untied | `00` | 0 | 0 | 26 | 26 / 0 | no |
| untied | `10−` | 0 | 0 | 26 | 25 / 1 | no |
| untied | `10+` | 0 | 0 | 25 | 25 / 0 | no |
| untied | `01−` | 26 | 28 | 32 | 32 / 0 | yes |
| untied | `01+` | 0 | 0 | 21 | 21 / 0 | no |
| untied | `11−` | 15 | 22 | 32 | 31 / 1 | yes |
| untied | `11+` | 18 | 28 | 32 | 19 / 13 | yes |

Since neither `00` cell has an accepted recurrence anchor, the coherent `00`↔`11` mapped shift is undefined for this pilot. The signed cells that do recur map back to source bins; representative exact anchors are tied `01−`: `(-111,260,3,288)`, `(-111,275,3,290)`, `(-111,275,3,288)`, `(-111,275,-38,288)` and tied `11−`: `(0,226,81,255)`, `(0,116,65,145)`, `(0,126,75,163)`, `(0,116,75,163)`. Untied `01−` has `(-111,197,-22,216)`, `(-111,197,-19,216)`, `(-111,0,1110,999)`; untied `11−` has `(0,193,81,216)`, `(0,193,75,216)`, `(0,0,57,70)`, `(0,0,57,53)`, `(0,0,46,53)`; untied `11+` has `(13,238,60,247)` and `(21,247,59,247)`. These are numerical source-bin outputs, not physical-object identity. The `10` cells are non-recurrent in both packages, while `01−` is recurrent in both; this is a mismatch diagnostic, not a detection-quality control.

### Full-vocabulary opener/EOS and forced x1 windows

The conditional forced-description measurement uses the same description and reports raw full-vocabulary quantities. The old and moved windows are fixed for every cell: old `91`, moved `0` for `−` and `182` for `+`, radius 8. `Δlog p` and `Δlog mass` are moved minus old; no window renormalization is used. Opener is the chosen free-continuation log probability, EOS is its full-vocabulary log probability, and coordinate-family mass is the raw 1,000-bin family mass.

| model | cell | recurrence | valid / invalid | rows | cap EOS | opener log p | EOS log p | coord mass | Δlog p −/+ | Δlog mass −/+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | `00` | no | 31 / 0 | 31 | no | −0.001 | −7.425 | 1.08e−14 | −5.902 / −4.497 | −7.434 / −4.645 |
| tied | `10−` | no | 29 / 1 | 30 | no | −0.002 | −6.180 | 1.09e−13 | −4.961 / −3.754 | −6.446 / −3.890 |
| tied | `10+` | no | 32 / 0 | 32 | yes | −0.003 | −5.787 | 1.43e−13 | −2.411 / +1.923 | −3.937 / +2.263 |
| tied | `01−` | yes | 32 / 0 | 32 | yes | −0.001 | −6.917 | 1.41e−12 | +1.671 / −3.041 | −0.029 / −2.875 |
| tied | `01+` | no | 32 / 0 | 32 | yes | −0.001 | −6.939 | 4.79e−14 | +1.658 / +8.878 | −0.071 / +8.750 |
| tied | `11−` | yes | 32 / 0 | 32 | yes | −0.001 | −7.333 | 9.68e−14 | +5.318 / −3.223 | +3.903 / −2.902 |
| tied | `11+` | no | 32 / 0 | 32 | yes | −0.001 | −7.163 | 1.84e−14 | +0.337 / +9.623 | −1.179 / +9.897 |
| untied | `00` | no | 26 / 0 | 26 | no | −0.000 | −7.708 | 6.95e−14 | −5.001 / −1.855 | −6.835 / −1.899 |
| untied | `10−` | no | 25 / 1 | 26 | no | −0.002 | −6.004 | 3.33e−12 | −4.771 / −2.205 | −6.465 / −2.355 |
| untied | `10+` | no | 25 / 0 | 25 | no | −0.001 | −6.806 | 3.85e−13 | −2.883 / +1.197 | −4.144 / +1.501 |
| untied | `01−` | yes | 32 / 0 | 32 | yes | −0.002 | −6.388 | 1.03e−11 | +1.770 / −2.740 | +0.151 / −2.571 |
| untied | `01+` | no | 21 / 0 | 21 | no | −0.001 | −6.684 | 3.72e−13 | +0.349 / +5.970 | −1.245 / +5.811 |
| untied | `11−` | yes | 31 / 1 | 32 | yes | −0.001 | −6.808 | 1.04e−12 | +4.259 / −2.221 | +2.795 / −2.176 |
| untied | `11+` | yes | 19 / 13 | 32 | yes | −0.001 | −7.598 | 7.90e−14 | −0.065 / +9.280 | −1.718 / +9.581 |

The row-cap receipt distinguishes natural EOS from injected EOS; a reported `im_end` stop reason does not erase the explicit row-cap provenance. The forced measurements are conditional diagnostics and do not supply free-continuation recovery credit.

### Strata, cost and stop

The CPU denominator is the complete 45-state panel (21 failure, 24 proxy), but the model denominator is only the released `refined-03` failure source: image `417044`, two model packages, seven cells each, 14 free continuations. No proxy model call was made in this bounded pilot. Tied and untied remain package strata, not an untie-only estimate. The attempt used 4,192 model forwards and 42 vision forwards on GPU0; initial `00` phase wall time was 70.413 s, signed diagnostic phase wall time was 313.788 s, and the enclosing device interval was 454.020 s. GPU seconds were not instrumented, so they remain null. No owned producer process remains and parameter mutation is false. The existing exact saved native/no-op gate is scoped to the separate untied numerical-feedback `untied-885-failure` runtime (batch size 3); current Lane-B target-only request count 1 versus original source-group count 4 has no saved logits/token parity receipt, so that gate is identified but does not close the current Lane-B route.

The executed pilot remains a bounded local `HOLD`; the lane status is `pending_parent_gate`. The unchanged panel still has 43 unexecuted states, so this record does not close Lane B or support an all-failures-vanish inference. Root must first verify the current target-only versus original four-case source-route parity gate before releasing any remaining model work.

### Corrected pilot artifacts

- Receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/pilot-receipt.json` (SHA-256 `aa8343f65727c353a7aecb0323cda63a1427b5353ffffed3d2b1834edce9aabf`)
- Independent reduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/reduced/pilot-summary.json`
- Immutable pre-call snapshot: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/source-snapshot.json` (SHA-256 `b3667e69db551feb54f7a4cdbce75410c7828bc98307a91690480fafa61f338a`)
- Input construction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/input-construction-receipt.json`
- Pending parent gate and remaining-state/source inventory: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/pending-parent-gate.json` (43 remaining states; current bs4 parity marked missing) (SHA-256 `cefc89eeb84fbe685a12ec317ca87e74b60f3ae162bbf0fb0131e1a24686ded9`)
- Existing scoped native/no-op parity gate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/coordination/baseline-noop-launch-v1.json` (passes for separate untied `untied-885-failure`, not current Lane-B route)
- Raw tied/untied outputs: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/raw/`
- CPU reducer: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/pilot_reducer.py`

Acceptance commands for this corrected attempt:

```bash
python3 -m py_compile probes/training_set_completion/recurrence_spatial/*.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/red_witness.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/green_witness.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/correction_audit.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/pilot_reducer.py
```


## Correction status: bounded HOLD

The candidate-v1 model outputs and result are preserved byte-for-byte under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source-candidate-v1`.
They are technically invalid for the intended contrast and must not be used as
Lane-B scientific evidence. The pre-correction producer mapped y coordinates
with x geometry, both producer and reducer inverse-mapped all four roles with x
geometry, the row limiter counted `OBJ_END` before a complete box, recurrence
admission dropped invalid/out-of-source rows and used adjacent-link near runs,
and forced x1 used history offset single bins rather than fixed per-source,
per-sign windows.

The CPU correction gate is complete at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/correction-receipt.json`.
It checked all 45 bound inputs and 267 saved cells, with zero GPU forwards.
The RED witnesses reproduce each defect through the candidate-v1 helpers; the
GREEN witnesses pass role-aware transforms/inverses, full `BOX_END` row caps,
the accepted consecutive-triple recurrence primitive while retaining invalid
complete rows, and saved full-vocabulary ±8-bin forced windows. The row
limiter now measures only the free suffix and records whether it injected EOS;
the candidate-v1 receipts predate that provenance, so their 76 parser-derived
row-bound classifications are not native stop-reason evidence. The correction
receipt reports 6,679 history y-coordinate changes, 76 truncated row-bound
cells, 15 changed recurrence predicates, and 164 wrong historical fixed-sign
scalar window assignments across 534 recomputed sign windows. Its conservative
enclosing occupancy bound is 10.601 GPU-hours for prior A/B/D work plus the
reserved 4 GPU-hours for C, leaving 9.399 GPU-hours under the 24-hour package
ceiling; missing intervals are not counted as zero.
The corrected pilot-v2 reduction is reported above. The recurrence,
mapped-shift, mismatch, opener/EOS, and forced-window tables below remain the
historical candidate-v1 diagnostics and are retained only for provenance; they
are not scientific evidence because that runtime was technically invalid.

Lifecycle is `HOLD`; root acceptance, policy promotion, and successor launch remain outside this worker boundary. The frozen question is whether numerical recurrence follows moved visual content, moved coordinate history, or fixed canvas preferences. The primary comparison is common `00` against coherent `11-`/`11+`, after inverse mapping output boxes to source bins. `10` and `01` are mismatch diagnostics.

## Frozen source and geometry

- Shared panel: `005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`; 45 states (21 failure, 24 proxy). Two states (`tied-417044-failure`, `untied-417044-failure`) reuse the accepted 14-cell pilot; the other 43 states have current native runtime receipts.
- Each cell copies the unchanged scene into a fixed black common canvas with no resize/interpolation, 32 px grid, round-half-even coordinate remapping, and ±128 px horizontal displacement relative to the centered 00 origin (00 uses visual/history offset 128 px; 10−/10+ use visual 0/256 with history 128; 01−/01+ use visual 128 with history 0/256; 11−/11+ use 0/0 and 256/256). History boxes receive the same affine map as the visual content. Cells are `00`, `10-`, `10+`, `01-`, `01+`, `11-`, `11+`; `00` is shared between signs.
- Source rows, including invalid source rows, remain in the bound histories. In the 45-state panel there are 503 source history rows: 3 invalid source rows across 3 states; 5 rounding-validity-change entries in `tied-7511-healthy`; mapped invalid rows and order checks are retained in the JSON result.

## Admission and primary comparison

| panel kind | states | admitted 00 + all 7 cells | 00 recurrent | 11− recurrent | 11+ recurrent | both 11 signs | either 11 sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| failure | 21 | 14 | 14 | 10 | 12 | 9 | 13 |
| healthy | 24 | 23 | 5 | 9 | 5 | 5 | 9 |

The declared denominator is shown in the first state column; the admitted column excludes common-00 admission holds. These values are descriptive candidate-v1 outputs under the invalid pre-correction runtime and are not a Lane-B scientific result. Failure states all had recurrent 00 among the 14 admitted states. Coherent 11 recurrence persists for both signs in 9/14 admitted failure states and 5/23 admitted proxy states. Predicate agreement with 00 is 10/14 for `11-` and 12/14 for `11+` in failure states; proxies are 19/23 and 21/23. The mapped repeat-anchor shift summaries among cells with anchors are `11−`: median `[1.5, 26.5]` source bins (n=15), `11+`: median `[1.0, -60.5]` (n=16). These are numerical mapped shifts, not physical owner identity.

## Mismatch diagnostics

| cell | present | recurrent | exact | near | alternative recurrence versus 00 |
|---|---:|---:|---:|---:|---:|
| 10- | 37 | 11 | 10 | 11 | 1 |
| 10+ | 37 | 13 | 8 | 13 | 0 |
| 01- | 37 | 14 | 10 | 14 | 0 |
| 01+ | 37 | 6 | 6 | 6 | 0 |

These cells deliberately break visual/history coherence and are not detection-quality controls. Anchor shifts are summarized in the result JSON by cell; missing anchors are retained as null rather than imputed.

## Per-cell recurrence and geometry totals

| cell | present | recurrent | exact | near | valid rows | invalid rows | malformed rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | 45 | 19 | 13 | 19 | 713 | 80 | 14 |
| 10- | 37 | 11 | 10 | 11 | 346 | 83 | 7 |
| 10+ | 37 | 13 | 8 | 13 | 611 | 24 | 9 |
| 01- | 37 | 14 | 10 | 14 | 697 | 59 | 17 |
| 01+ | 37 | 6 | 6 | 6 | 453 | 12 | 6 |
| 11- | 37 | 19 | 18 | 19 | 646 | 44 | 14 |
| 11+ | 37 | 17 | 14 | 17 | 617 | 50 | 9 |

All 267 candidate-v1 free continuations emitted the EOS token; the historical parser classified 191 below the 32-row bound and 76 at the old row-bound threshold. Because candidate-v1 did not record free-suffix/injected-EOS provenance, these are parser classifications rather than confirmed native stop reasons. None reached the 512-token cap. The raw and reduced receipts retain longest runs, descriptions, source-mapped boxes, pair-edge counts, output token counts, border rows, and out-of-bounds rows for each cell. The corrected audit uses the accepted consecutive triple for exact and near admission; long runs may still have quadratic pair-edge counts in descriptive summaries.

## Opener/EOS and forced-description diagnostics

| cell | n | median opener log p | median EOS log p | median opener−EOS | median coordinate-family mass | median forced moved−old x1 log p |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 45 | -0.000216 | -8.454578 | 8.454363 | 6.076e-14 | 0.000000 |
| 10- | 37 | -0.001236 | -6.698269 | 6.697033 | 6.168e-13 | 0.000000 |
| 10+ | 37 | -0.000112 | -9.100419 | 9.100307 | 2.187e-14 | 0.000000 |
| 01- | 37 | -0.000118 | -9.048973 | 9.048855 | 4.140e-14 | 1.235125 |
| 01+ | 37 | -0.001857 | -6.290899 | 6.289041 | 4.307e-13 | 3.468387 |
| 11- | 37 | -0.000317 | -8.062466 | 8.062149 | 4.926e-14 | 6.879394 |
| 11+ | 37 | -0.000750 | -7.198788 | 7.198038 | 2.253e-13 | 8.532781 |

The opener and coordinate-family quantities are full-vocabulary values. The forced-description x1 old/moved window measurement is conditional and is not free continuation quality; no window renormalization is used. Full top competitors, coordinate distributions, and old/moved logits/log probabilities remain in each runtime receipt.

## Source and runtime strata

| source group | states | failure | proxy | tied | untied |
|---|---:|---:|---:|---:|---:|
| val-extra | 12 | 6 | 6 | 6 | 6 |
| refined-01 | 3 | 1 | 2 | 1 | 2 |
| refined-02 | 4 | 2 | 2 | 2 | 2 |
| refined-03 | 4 | 2 | 2 | 2 | 2 |
| fresh-18 | 2 | 0 | 2 | 1 | 1 |
| new-08 | 2 | 2 | 0 | 1 | 1 |
| new-18 | 2 | 2 | 0 | 1 | 1 |
| new-17 | 2 | 2 | 0 | 1 | 1 |
| new-14 | 2 | 2 | 0 | 1 | 1 |
| new-10 | 1 | 1 | 0 | 0 | 1 |
| new-27 | 1 | 1 | 0 | 0 | 1 |
| new-09 | 2 | 0 | 2 | 2 | 0 |
| new-06 | 1 | 0 | 1 | 1 | 0 |
| new-25 | 1 | 0 | 1 | 1 | 0 |
| new-19 | 1 | 0 | 1 | 1 | 0 |
| new-13 | 1 | 0 | 1 | 1 | 0 |
| new-28 | 1 | 0 | 1 | 1 | 0 |
| new-02 | 1 | 0 | 1 | 1 | 0 |
| new-07 | 1 | 0 | 1 | 1 | 0 |
| new-20 | 1 | 0 | 1 | 1 | 0 |

Model/policy/runtime/source identity is retained per state. Tied versus untied is a package comparison; it is not an untie-only estimate. Existing mature records, feedback selections, and new-cohort saved `image_plan` receipts are bound by path and SHA in the state summaries.

## Excluded or no-longer-recurrent states

The frozen admission rule generated no transformed cells for these eight states; this is an explicit denominator outcome, not a replacement selection:

| state | kind | model | source group | image | valid / complete rows | source invalid |
|---|---|---|---|---:|---:|---:|
| untied-885-failure | failure | untied | val-extra | 885 | 10 / 10 | 0 |
| tied-885-failure | failure | tied | val-extra | 885 | 10 / 10 | 0 |
| untied-5586-failure | failure | untied | val-extra | 5586 | 23 / 23 | 0 |
| tied-5586-failure | failure | tied | val-extra | 5586 | 23 / 26 | 3 |
| untied-7511-failure | failure | untied | refined-01 | 7511 | 18 / 21 | 3 |
| tied-7511-healthy | healthy | tied | refined-01 | 7511 | 0 / 0 | 0 |
| untied-train-196924-failure | failure | untied | new-10 | 196924 | 0 / 0 | 0 |
| untied-train-505967-failure | failure | untied | new-27 | 505967 | 16 / 16 | 0 |

The 7 failure holds no longer retained the frozen recurrence predicate in common 00. `tied-7511-healthy` is a proxy hold because common 00 emitted no valid row; it is retained as a healthy-control admission outcome.

## Cost and acceptance

- 267 free continuations: 45 common-00 plus 222 transformed cells; 42,270 model forwards and 801 vision forwards; summed producer wall time 4,408.467 s. GPU seconds were not instrumented and remain null; wall time is not relabeled as GPU time.
- Runtime JSON bytes: 56440507; reduced JSON bytes: 10881425. The 40 pre-fix new-cohort attempts (20 first batch, 20 v2 rerun) failed before native forward on missing `image_plan`; they are mechanical qualification attempts and are excluded from the scientific continuation denominator. The corrected 20-state rerun used saved Lane-A planning fields and verified input identity.
- No producer or launch processes remain. No model parameters were mutated.

### Artifacts

- Candidate result: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/result.json`
- Cost receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/cost-receipt.json`
- Job closure: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/job-closure.json`
- Manifest reconciliation: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifest-reconciliation.json`
- CPU reducer: `probes/training_set_completion/recurrence_spatial/reduce.py`
- CPU correction audit: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/correction-receipt.json`
- RED/GREEN witnesses: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/red-witness.json`, `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/green-witness.json`
- Saved-output reducer receipt/output: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/saved-reducer-receipt.json`, `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/saved-reducer-untied-885-healthy.json`
- Saved-output summary reducer: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/summarize_candidate.py`
- Frozen source snapshot: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/source-snapshot.json`

Acceptance commands:

```bash
python3 -m py_compile probes/training_set_completion/recurrence_spatial/*.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/red_witness.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/green_witness.py
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/correction/correction_audit.py
python3 -m probes.training_set_completion.recurrence_spatial.reduce --model tied --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-169872-failure.json --runtime-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-169872-failure.json --reduced-path /tmp/recur-spatial-recheck.json
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/summarize_candidate.py --check-only
```

This is a lane candidate. The recurrence and mapped-location results do not identify a unique physical object or causal circuit, and optional grounding-bank absence limits physical interpretation.
