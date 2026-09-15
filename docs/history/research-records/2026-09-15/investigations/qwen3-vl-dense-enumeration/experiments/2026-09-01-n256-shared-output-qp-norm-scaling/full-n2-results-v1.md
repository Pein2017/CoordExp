# Full 128-image N2 stage: terminal scientific receipt

## Disposition

**STOP at N2. Do not construct or launch N4.**

Run `n256-full-n2-v1-20260901T145100Z` completed one sealed N256 semantic
capture, the sole unique N2 derangement, both exact N2 solves, and fresh
128-image Source, semantic, and null-0 screens. The terminal aggregate records
`strict_target_blind_transfer: false` and
`continue_or_stop.decision: STOP` because the semantic payload failed Source
owner preservation and hard-debt non-increase.

This is a decision-bearing N2 stage result, not an N256 outcome. The full
ladder scientific verdict remains `null`; no N4 or later-stage artifact was
created.

## Frozen identity

- Launch packet: [launch-packet-full-n2-v1.json](launch-packet-full-n2-v1.json),
  SHA-256
  `c19dd4925d853091ebffc4867d270a1fa91b7ba02aa3d89cd52e00e61e8d2eee`.
- Thin real runner:
  `scripts/research/run_n256_shared_output_qp_scaling_full.py`, SHA-256
  `140218a441ec9e3e55f168f4bab6e1482431be7451b41c574708ee0701da23fe`.
- Terminal aggregate:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-n256-shared-output-qp-norm-scaling/full-n2-v1/aggregate-full-n2.json`,
  SHA-256
  `98f514c734a95a02b7f5920757d3783694f360f7226f593b9b46b8a29f35cbf2`.
- Natural decoded prediction order is arbitrary and normal. All owner scoring
  used category-compatible global one-to-one IoU50 matching; no decoded
  sortedness gate or debt was applied.

## Train capture and exact solves

The unique semantic capture used all 256 train images, 1,955 owners, and
18,745 decision positions in one model process (`PID 3769205`): 256
teacher-forced forwards, zero natural decodes, and one model load. The common
N256 positive-deficit RMS was `2.0026697456508598` over 7,412 positive
deficits. Tensor SHA-256:
`1e0e724cd7824110caa6a10ae3ba71fb5f06489540d39e00cb4e301e535d48bf`.

At N2, requested `K=4` collapsed to the only possible unique derangement
(`unique_K=1`), the complete swap. The plan was train-only and records
`screen_consulted: false`.

| arm | payload SHA-256 | Frobenius norm | standardized deficit energy | norm^2 / standardized energy | effective rank | condition (>1e-12) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| semantic | `664c4ea970ffe69ee90ed5e5e824296001c0aa5701597a7aa313a513f45cfb6d` | 0.275629 | 75.012194 | 0.00101279 | 4.76724 | 1388.39 |
| null-0 | `052b0e8c030595a9ebd44c99dc8bc50dc90e5e5fcfb866e63556e1adc4872763` | 1.127251 | 501.250561 | 0.00253505 | 7.87796 | 2525.37 |

Both exact solves certified the fixed `R=1136` surface. Semantic used 197
active constraints and 1,326 optimizer iterations, with maximum FP64 violation
`8.92e-6` and minimum FP32 hook margin `0.00999069`. Null-0 used 549 active
constraints and 10,073 iterations, with maximum FP64 violation `6.59e-7` and
minimum FP32 hook margin `0.00999451`.

**Observation:** the N2 semantic payload is much smaller and has lower
standardized train deficit energy than its one admitted null. **Inference
boundary:** one two-image prefix and one possible null cannot establish N256
shared compression, scaling, or generalization.

## Disjoint 128-image screen

The three arms used distinct fresh processes and a common runtime identity
hash `1b995a278ba526adb675393d440381175a96807c42eb91d10dba5340c422bd3f`:
Source `PID 3793554`, semantic `PID 4018823`, null-0 `PID 4077826`. Each loaded
the model once and performed 128 natural decodes plus 128 teacher-forced margin
forwards.

| arm | mean margin | change vs Source | matched owners | gained | lost | net | duplicate | malformed | cap | unmatched monitor | all natural EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Source | 5.546764 | 0 | 608 | 0 | 0 | 0 | 113 | 58 | 0 | 495 | yes |
| semantic | 5.492892 | -0.053871 | 587 | 21 | 42 | -21 | 203 | 98 | 0 | 629 | yes |
| null-0 | 5.343238 | -0.203526 | 381 | 21 | 248 | -227 | 2004 | 3565 | 7 | 2885 | **no** |

The semantic arm independently triggers the frozen stop rule: it loses 42
Source-matched owners and increases duplicate debt by 90 and malformed debt by
40. Its 21 gained owners do not compensate under the strict retention gate.
The null arm is worse and also has seven length-capped, non-natural-EOS cases.
Unmatched predictions remain an unknown monitor and are not treated as hard
false-positive debt.

Therefore the evidence falsifies strict target-blind transfer at N2 for this
frozen payload. It does not show that scaling would or would not repair the
effect, because the authorized ladder stops before testing that counterfactual.

## Resources, attempts, and residue

Resources were observed only, never used as a scientific ceiling. Across the
three screens, elapsed time was `7719.958 s`; the peak host RSS was
`12,266,954,752` bytes, peak PyTorch GPU allocation `11,838,387,712` bytes,
peak reservation `13,629,390,848` bytes, and final output tree
`690,642,466` bytes. The unusually long null screen (`5470.396 s`) was allowed
to finish and exposed genuine length/cap failure rather than a resource stop.

One shell command before the successful semantic invocation named a
nonexistent runner path and exited before Python entry, model loading, screen
access, or artifact creation. The sealed semantic attempt then ran once under
its launch-bound attempt ID. No result-driven retry or payload correction was
performed. Post-stage checks found no matching screen process and GPU 7
returned to zero reported memory use.

## Stop boundary

`aggregate-full-n2.json` is the terminal stage receipt. N4 through N256 remain
unexecuted by the frozen early-stop rule. The N2 train-side semantic/null gap
may be reported only as a bounded observation alongside the failed held-out
screen gate; it is not an architecture-promotion or population-generalization
claim.
