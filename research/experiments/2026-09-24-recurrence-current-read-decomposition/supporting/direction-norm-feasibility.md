# Direction versus norm feasibility — CPU candidate

Status: **FEASIBLE** for fixed analytical G/D at the saved current-y1 query. This is source and arithmetic feasibility only; no model treatment, score, threshold or physical claim.

## Bound source and calculation

Original refined-03 four-request target2 train351017; A raw[:9], replayed six current IDs, query physical1376, history keys[1362,1371), width1377. All six accepted attempt002 cells contributed all28 layers ×16 query heads = 2,688 head records. Every cell's input JSON matches the same freshly reconstructed original CPU full input; original source/media/positions, QKV/rotary/actual-mask bindings are in [JSON](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-24-recurrence-current-read-decomposition/supporting/direction-norm-bindings.json).

For each saved cell's own incoming Q/K/V, the maintained FP32-RoPE/FP64 reduction reconstructs N=H+R and R under the native readable mask. Per head, n=||N||₂ and r=||R||₂; G=(r/n)N, D=(n/r)R. Controls were checked in FP64 and cast to FP32 once for prospective errors. No gain fitting, clipping, epsilon, head or layer selection.

## Complete feasibility counts and ranges

Valid heads **2,688/2,688**; HOLD heads 0. G gain>1 in 284 heads; D gain>1 in 2,404; exact equal norms 0. Gains above one are amplification, not attenuation.

| Quantity | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|
| n | 0.192344914 | 4.88852976 | 43.9939104 | 368.030228 |
| r | 0.103558428 | 4.44371646 | 38.044058 | 347.730629 |
| G gain r/n | 0.0733089365 | 0.963968682 | 1.00099112 | 1.03828905 |
| D gain n/r | 0.963122937 | 1.03737814 | 1.62455798 | 13.6409018 |
| cos(N,R) | -0.131229206 | 0.997866609 | 0.999997801 | 1 |
| G FP32 signed norm error | -4.17716262e-06 | 2.86134616e-10 | 1.30105011e-07 | 2.53154275e-06 |
| D FP32 signed norm error | -1.95372542e-06 | -2.93653768e-10 | 1.42327045e-07 | 1.78299996e-06 |

All per-head values, p05/p25/p75 quantiles, signed FP32 errors, and cast direction cosines are retained in the JSON; the table does not rank or select heads.

| Saved cell | Valid heads | G gain>1 | D gain>1 | G gain min–max | cos(N,R) min–max |
|---|---:|---:|---:|---:|---:|
| native_anchor | 448 | 36 | 412 | 0.107488–1.03326 | -0.10718–1 |
| current_mask_anchor | 448 | 58 | 390 | 0.0754714–1.03829 | -0.0800012–1 |
| identity_reconstruction | 448 | 36 | 412 | 0.107487–1.03326 | -0.107181–1 |
| full_mask_bridge | 448 | 58 | 390 | 0.0754715–1.03829 | -0.0800012–1 |
| remove_H | 448 | 56 | 392 | 0.0733089–1.03771 | -0.0562807–1 |
| redistribute_R | 448 | 40 | 408 | 0.134659–1.0337 | -0.131229–1 |

The asymmetric two-head CPU fixture passed the intended per-head norm and direction identities, rejected zero N and zero R, and rejected a global-norm gain and swapped G/D controls. This validates the algebraic route, not whole-model behavior.

## Smallest later route for lead review only

A possible six-call sequence is fresh N, fresh R removal, independent N reconstruction identity, independent R write identity, then G and D. The N reference is accepted native_anchor; the R reference is accepted remove_H, **not** current_mask_anchor. Each prospective arm would recompute its gain from its own incoming Q/K/V at every layer/head, cast once to FP32, and write only target2 physical1376 pre-o_proj. Original native mask, other queries, history and companions stay fixed. The actual writer's selected-output and untouched-complement checks, all-layer consumer observation, finally hook restoration, and separate cold verifier would be retained. The first four cells would require the accepted all-four vector gates before either novel control; lead must set new technical norm bounds and the full-vocabulary scientific endpoint before any launch.

Measured attempt002 parent outer 50.671184s for six calls; a 2× same-shape planning estimate is 101.342369s, not a runtime bound. Its raw tree is 1,996,222,285 bytes, with accepted 25% evidence allowance 2,491,132,218 bytes inside 3GiB. Same width/pixels and six calls are a finite proposal, not GPU authority. Peak memory and exact future overhead require fresh qualification.

No saved tensor has a zero/nonfinite/unrepresentable per-head norm or control under the stated arithmetic. That establishes computational feasibility on these six old inputs only. Adaptive later-layer Q/K/V can change G/D and the final distribution; no semantic content, copied-coordinate, natural-recurrence or mediation claim follows.

## Provenance and accounting

Brief SHA-256 `4af3a4eaadd1112708393e9b1d6dd279d67d092e8c523c20ea521abe23be5420`; lead acceptance `3f255b53adc45804c2e930319bc6c0a2d7f50fb874362c556687de18c6c5a667`; maintained producer `1d346e3e9534dc0ff1cd8949f1fc6736996cc4f458aa7bffb3554e3de7dfd0d4` and byte-identical preserved capture `1d346e3e9534dc0ff1cd8949f1fc6736996cc4f458aa7bffb3554e3de7dfd0d4`. Accepted receipt `610faec502b3239b0a00f53b4b0bc6537f7f28b07a8922501ac0ce582467d9e2`, cold readback `92d068c9f555df957b332185eec2d3f1561635778e793301da8dd76c4572fb4b`, preflight `b9edbb727ee8ad7e0ee6f6307020ab0e1fcbd05487b8a433b847f8a4ebb8383e`. Six raw/input hashes and every head record are in [direction-norm-bindings.json](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-24-recurrence-current-read-decomposition/supporting/direction-norm-bindings.json). Only this report and that JSON were written. **0 model loads, 0 model/vision forwards, 0 CUDA calls, 0 generated tokens, 0 GPU-hours added.** Prior accepted sequence stays 0.9321684536531304 GPUh. Lead owns any later contract.
