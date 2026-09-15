# N4 train-only low-norm discriminator: terminal receipt

## Disposition

**YES — the frozen N4 train prefix has low-norm structure relative to all four
admitted matched derangements.**

Run `n256-train-n4-v3-20260902T021851Z` satisfies the predeclared positive
gate: the semantic exact minimum has both lower raw Frobenius norm and lower
`R^2 / standardized_deficit_energy` than every `K=4` unique null. The terminal
machine receipt records `N4_LOW_NORM_STRUCTURE_EVIDENCE` and
`next_action: STOP_AFTER_N4`.

This answers only the train-only N4 mechanism question in
[n4-train-only-amendment-v1.md](n4-train-only-amendment-v1.md). The failed N2
target-blind transfer result remains unchanged; this is not N256 scaling,
held-out transfer, population generalization, or architecture promotion.

## Decision evidence

The stage contains four train images, 29 annotated owners, 273 decision
positions, one semantic assignment, and four unique complete-route
derangements.

| arm | Frobenius norm | standardized deficit energy | `R^2 / standardized deficit energy` | effective rank |
| --- | ---: | ---: | ---: | ---: |
| semantic | **0.329872** | 98.8921 | **0.00110035** | 8.515 |
| null-0 | 1.839523 | 1029.3891 | 0.00328724 | 16.736 |
| null-1 | 1.804803 | 1032.0042 | 0.00315630 | 16.976 |
| null-2 | 1.811899 | 1118.3519 | 0.00293555 | 16.674 |
| null-3 | 1.670622 | 976.4154 | 0.00285839 | 16.155 |

Against the most favorable null, semantic requires only `0.19745x` the raw
norm (about `5.06x` lower) and `0.38495x` the difficulty-normalized cost
(about `2.60x` lower). The second comparison is decision-bearing because it
remains favorable after accounting for the fact that semantic Source deficits
are themselves much smaller than deranged deficits.

From N2 to N4, semantic owners increase `14 -> 29` and positions
`132 -> 273`, while norm increases only `0.275629 -> 0.329872` (`1.197x`) and
`R^2 / owner` falls to `0.691x` its N2 value. This is compatible with shared
sublinear structure between N2 and N4, but two stages do not establish an N256
scaling law.

## Exact-solve acceptance

All five solves use the fixed 1,136-row surface, margin `0.01`, tighter FP64
separation tolerance `1e-5`, and the frozen FP32 replay tolerance `2e-5`.
Maximum FP64 violations range from `2.59e-6` to `9.40e-6`; minimum FP32 hook
margins range from `0.00998306` to `0.00999069`. Duality-gap magnitudes are at
most `3.74e-8`. Therefore every arm passes the exact minimum-norm and replay
certificates before comparison.

## Technical correction provenance

- Launch v1 completed deterministic planning only, then was superseded before
  model entry because GPU 7 acquired an active 45.8 GiB workload.
- Launch v2 used free GPU 1 and sealed the four null captures in one process:
  16 teacher-forced forwards, zero natural decodes, and one model load. The
  capture receipt SHA-256 is
  `b626867c6ff0f322279c22caf25227577f9398af3ce779ff51b5e37dc1f98af3`.
- The first v2 solve became technical-invalid when null-0 FP32 replay missed
  the frozen acceptance boundary by `7.52e-7`. The shared solver had consumed
  the full replay tolerance during FP64 separation, leaving no rounding
  budget. No v2 partial solve output contributes to the verdict.
- A regression first failed because the solver lacked a distinct FP64
  tolerance. The smallest correction adds that tolerance, solves N4 to
  `1e-5`, and retains the original `2e-5` replay gate. The corrected suite
  passed `20/20`. Launch v3 reuses the hash-bound valid v2 plan and null
  captures and recomputes all five solves.

No screen input or screen GT was bound or opened by the N4 runner.

## Identity and resources

- Launch packet:
  [launch-packet-train-n4-v3.json](launch-packet-train-n4-v3.json), SHA-256
  `dc126853b6b82bbbb8cf5175cc2cf17a528487151dfb6e630412a560f2f9a0e9`.
- Corrected core runner SHA-256:
  `380ac5bb0834af03942ee4ba6410243a5e797d906a911e3782113b4dfaf9afac`.
- N4 thin runner SHA-256:
  `86184e0c1b0a61ffafd5d8a539c860a29c53b755a2b228dd092f3d3365f620b4`.
- Terminal aggregate:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-n256-shared-output-qp-norm-scaling/train-n4-v3/aggregate-train-n4.json`,
  SHA-256
  `999ffa8c72b3774bf09022bf533573b94266c8633fbb397212e288abadcca038`.

The valid null capture took `22.15 s`, peaked at `9,086,589,440` GPU allocated
bytes, `9,865,003,008` reserved bytes, and `12,072,628,224` host RSS bytes.
The five corrected CPU solves took about `700 s` in total. Resources were
observe-only and did not define the result.

## Stop boundary

The N4 low-norm question is answered positively and this amendment is closed.
Do not infer transfer from this result: the earlier N2 screen still lost 42
Source-matched owners and increased duplicate/malformed debt. N8 and later
stages remain unexecuted and require a separate user-owned question and stop
rule.
