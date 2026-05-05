# Core6 FN Boundary Length-Bias Read

Source artifacts:

- Boundary probe root: `/data/CoordExp/temp/core6_fn_boundary_probe_20260501`
- Pair rows: `*/pair_scores.jsonl`
- Candidate rows: `*/candidate_scores.jsonl`
- Derived row cache: `length_bias_rows.json`

Definitions:

- `close_adv = log p(close first token) - log p(continue first token)`.
- Positive `close_adv` means the model prefers closing over separator/object continuation at the boundary.
- `frac_continue` is the fraction where the continue first-token margin beats close.
- Empty-prefix rows are structurally special and should not be interpreted as long-context stop pressure.

## Linear Trend Excluding Empty Prefix

| scorer | n | Pearson(length, close_adv) | slope per token | slope per 100 tokens | mean close_adv |
| --- | ---: | ---: | ---: | ---: | ---: |
| base1332 | 375 | 0.385 | 0.00391 | 0.391 | 1.391 |
| et_rmp_step300 | 375 | 0.379 | 0.00221 | 0.221 | 1.009 |

## Token-Binned Boundary Read

| scorer | token bin | n | mean close_adv | median close_adv | frac_continue | mean continue mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base1332 | 0-75 | 22 | 0.648 | -0.035 | 0.636 | 0.446 |
| base1332 | 75-125 | 37 | 0.227 | 0.252 | 0.324 | 0.465 |
| base1332 | 125-175 | 38 | 1.095 | 0.213 | 0.237 | 0.315 |
| base1332 | 175-225 | 43 | 1.707 | 1.805 | 0.233 | 0.143 |
| base1332 | 225-275 | 62 | 1.011 | 0.100 | 0.129 | 0.279 |
| base1332 | 275-325 | 26 | 1.052 | 0.100 | 0.115 | 0.346 |
| base1332 | 325-375 | 98 | 1.821 | 2.930 | 0.000 | 0.219 |
| base1332 | 375-450 | 19 | 2.080 | 2.238 | 0.000 | 0.079 |
| base1332 | 450-650 | 30 | 2.529 | 2.579 | 0.000 | 0.001 |
| et_rmp_step300 | 0-75 | 22 | 0.896 | 0.000 | 0.318 | 0.320 |
| et_rmp_step300 | 75-125 | 37 | 0.437 | 0.281 | 0.000 | 0.351 |
| et_rmp_step300 | 125-175 | 38 | 0.232 | 0.252 | 0.000 | 0.484 |
| et_rmp_step300 | 175-225 | 43 | 0.840 | 0.523 | 0.000 | 0.318 |
| et_rmp_step300 | 225-275 | 62 | 1.021 | 0.760 | 0.000 | 0.248 |
| et_rmp_step300 | 275-325 | 26 | 0.944 | 0.826 | 0.000 | 0.215 |
| et_rmp_step300 | 325-375 | 98 | 1.519 | 1.406 | 0.000 | 0.082 |
| et_rmp_step300 | 375-450 | 19 | 1.222 | 1.137 | 0.000 | 0.034 |
| et_rmp_step300 | 450-650 | 30 | 1.256 | 1.313 | 0.000 | 0.046 |

## Object-Count-Binned Boundary Read

| scorer | object-count bin | n | mean close_adv | median close_adv | frac_continue | mean continue mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base1332 | 1-3 | 32 | 0.494 | -0.005 | 0.594 | 0.444 |
| base1332 | 4-6 | 57 | 0.551 | 0.252 | 0.281 | 0.434 |
| base1332 | 7-9 | 65 | 1.788 | 2.127 | 0.169 | 0.126 |
| base1332 | 10-12 | 71 | 0.903 | 0.100 | 0.141 | 0.321 |
| base1332 | 13-15 | 101 | 1.768 | 2.695 | 0.000 | 0.227 |
| base1332 | 16-19 | 19 | 2.080 | 2.238 | 0.000 | 0.079 |
| base1332 | 20+ | 30 | 2.529 | 2.579 | 0.000 | 0.001 |
| et_rmp_step300 | 1-3 | 32 | 0.688 | 0.000 | 0.219 | 0.347 |
| et_rmp_step300 | 4-6 | 57 | 0.382 | 0.281 | 0.000 | 0.409 |
| et_rmp_step300 | 7-9 | 65 | 0.754 | 0.429 | 0.000 | 0.307 |
| et_rmp_step300 | 10-12 | 71 | 1.033 | 0.693 | 0.000 | 0.254 |
| et_rmp_step300 | 13-15 | 101 | 1.499 | 1.224 | 0.000 | 0.086 |
| et_rmp_step300 | 16-19 | 19 | 1.222 | 1.137 | 0.000 | 0.034 |
| et_rmp_step300 | 20+ | 30 | 1.256 | 1.313 | 0.000 | 0.046 |

## Best Single-Step Split

| scorer | threshold token | left n | left mean close_adv | right n | right mean close_adv | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base1332 | 151 | 75 | 0.258 | 300 | 1.674 | 1.416 |
| et_rmp_step300 | 197 | 107 | 0.458 | 268 | 1.229 | 0.772 |

