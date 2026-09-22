---
title: Human13 Shared Output-QP Same-Panel Overfit Results
description: One shared output-only residual passes the complete 392-owner Human13 same-panel gate under fresh-cold RP1.0 natural greedy.
type: investigation
role: research-result
authority: non_normative_research
status: completed_human13_overfit_pass
evidence_status: immutable_n13_primary_acceptance
updated: 2026-08-31
---

# Human13 Shared Output-QP Same-Panel Overfit Results

## Decision

- **N=2: PASS.** One shared output-only residual compiled all `65 / 65`
  registered owners at IoU50, IoU60, and IoU80 under fresh-cold RP1.0, with
  zero hard debt, natural EOS, exact canonical replay, and exact
  Source-A -> candidate-B -> Source-A restoration.
- **N=4: PASS.** One newly solved shared residual compiled all `123 / 123`
  owners under the same gate and restoration checks.
- **N=13: `HUMAN13_OVERFIT_PASS`.** One shared residual compiled all
  `392 / 392` owners at IoU50, IoU60, and IoU80. Legacy-12 contributes
  `346 / 346`; Image2299 contributes `46 / 46`. All thirteen fresh candidate
  processes reached natural EOS with zero duplicate, unmatched, malformed, or
  cap debt, and all thirteen fresh Source processes exactly restored the frozen
  pre-candidate Source identity.

The accepted claim is:

> From the exact `geo_sorted_xy` step-2444 Source, one output-only residual
> shared across the fixed thirteen-image Human13 panel compiled its complete
> 392-owner policy under fresh-cold RP1.0 natural greedy without hard debt.

This is finite-panel shared-output overfit or compilability only. It is not
held-out transfer, identity or distributional generalization, semantic sharing,
hidden-state internalization, base-model learning, or production readiness.

## Frozen execution identity

- Decision input SHA-256:
  `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
- Source adapter / special-embedding SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da` /
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.
- Primary decode: one fresh Hugging Face fp32/SDPA process per cell, batch one,
  RP `1.0`, original prompt, strict category-consistent global one-to-one
  actual-pixel matcher.
- RP `1.10` is monitor-only and does not own acceptance. It was not extended
  to N13 after the primary stop rule fired.
- Every stage uses one immutable payload; no image-specific residual,
  checkpoint, branch, or inference-time payload selection exists.

## Stage evidence

| Readout | N=2 | N=4 | N=13 |
|---|---:|---:|---:|
| accepted run | `20260831T-n2-v2` | `20260831T-n4-v1` | `20260831T-n13-v5-certificate-polish-corrected` |
| images / owners | 2 / 65 | 4 / 123 | 13 / 392 |
| decision states `P` | 592 | 1,147 | 3,637 |
| unique route targets `U` | 237 | 401 | 832 |
| deficient positions | 262 | 498 | 1,540 |
| selected output rows | 217 | 360 | 772 |
| hidden-span rank | 592 | 1,147 | 2,048 |
| free variables | 128,464 | 412,920 | 1,581,056 |
| registered constraints | 128,464 | 412,920 | 2,807,764 |
| final active constraints | 832 | 1,836 | 8,079 |
| outer solves / optimizer iterations | 9 / 7,567 | 10 / 15,390 | 16 / 76,176 |
| payload bytes | 3,557,232 | 5,901,288 | 12,654,792 |
| minimum FP32 hook margin | 0.00999069 | 0.00998688 | 0.00998688 |
| max FP64 violation | 8.87e-06 | 1.47e-05 | 1.0843e-05 |
| RP1.0 IoU50 / 60 / 80 owners | 65 / 65 / 65 | 123 / 123 / 123 | 392 / 392 / 392 |
| hard debt / natural EOS | 0 / 2 of 2 | 0 / 4 of 4 | 0 / 13 of 13 |
| exact canonical replay | 2 of 2 | 4 of 4 | 13 of 13 |
| exact Source A/B/A | 2 of 2 | 4 of 4 | 13 of 13 |

Primary receipts and payloads:

- N2 aggregate SHA-256:
  `66e1ab053d5dfcf2419c98f6dd49d2605f6966dd879c70eb19d97b9aa40571fe`;
  payload SHA-256:
  `fcece90e3ab5d5623de164c5402a947b92e34dab3af0a5994ce5363c982ac03a`.
- N4 aggregate SHA-256:
  `679984524773bfda023e33c7c81ee15d18ae8e980bc22029ebb1fa0e30ca3583`;
  payload SHA-256:
  `eff0f85eb0e3fc93635c84f284df380d61209c522ecdac626a4c6de5b98c8019`.
- N13 artifact root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-human13-shared-output-qp-same-panel-overfit/20260831T-n13-v5-certificate-polish-corrected`.
- N13 aggregate SHA-256:
  `3e99ca1a13491bbfdef848c31149bb81cfb9e86b49b5c0a2e566f4a3c62e05ac`.
- N13 extended acceptance SHA-256:
  `f5e28abb8902731e26d55631720e2f365d319ec509b4cebc39e7fc27e1ff2f87`.
- N13 payload / sidecar SHA-256:
  `fc22b930d487021a9859d08c2dba9f7fa33bc235f3e40350a9ec9f63657092db` /
  `8f9ca751cd63121af2a7112b40d8b165cb9e00e639f354e3497060cda1e3c3f6`.
- N13 bindings / launch packet / solve log SHA-256:
  `c315c59c651aef56e431ba488b44d5ef334cecbe6c9089f58ba5a4cb458457f0` /
  `0daf9e94e0ba4c190bfd6d002c7d549017bb6d552853c0a14e294152eba89846` /
  `51a70fc16247973b2e152d7528e2357713c647109f56c8240ed8cc5e46b55ca6`.

The extended acceptance receipt binds all thirteen candidate receipts and all
thirteen post-candidate Source receipts. The latter match the frozen
pre-candidate Source on generated-token hash, parser-text hash, stop reason, and
generated-token count. An initial Image2685 post-Source shell command named a
nonexistent module and failed before model load or receipt creation; the one
corrected invocation is the admitted evidence and is explicitly named in the
acceptance receipt.

## N13 numerical recovery and certificate

The original N13 solve and diagnostic replay established slow active-set
convergence but stopped before payload creation. The accepted recovery changed
only optimizer continuation at a terminal active subproblem:

1. retain the exact terminal dual and continue the same L-BFGS-B problem in at
   most three `4,000`-iteration segments;
2. when a status-0 terminal point still missed the unchanged certificate and no
   new cut existed, polish that same active problem with `ftol=0`; and
3. admit a payload only after the unchanged exhaustive FP64 and FP32 checks.

The target routes, selected rows, registered constraints, zero start, margin
`0.01`, certificate tolerance `2e-5`, full-vocabulary partition, and objective
were not relaxed or reselected. The accepted solution has:

- objective `0.5 * ||Delta W_out||_F^2 = 1.8325205182823667`;
- `||Delta W_out||_F = 1.9144296896372908`;
- maximum exhaustive FP64 violation `1.0842741027028424e-05`;
- minimum FP32 hook margin `0.00998687744140625`;
- duality gap `1.1209528860689488e-08`.

Run v3 stopped only `1.31e-06` above the frozen tolerance. Run v4 is a purely
mechanical invalid launch: repeated `--capture` options collapsed an argparse
`nargs+` value before optimizer or model execution. Run v5 is the single
corrected scientific entry and owns the result.

## Capacity and mechanism monitors

| Monitor | N=2 | N=4 | N=13 |
|---|---:|---:|---:|
| residual parameter effective rank | 24.84 | 34.36 | 62.75 |
| residual parameter rank-95 | 71 | 101 | 232 |
| largest row-energy share | 4.75% | 4.37% | 2.82% |
| residual first-direction energy | not decision-owning | not decision-owning | 5.52% |
| functional logit-delta effective rank on registered states | 2.09 | 2.27 | not required after primary stop |
| largest squared baseline-deficit share | 7.07% | 4.51% | not recomputed after primary stop |

The row-concentration trigger did not fire at any stage. The N13 residual is
less row-concentrated than the Image2299 G46 payload, but that is only a
mechanism diagnostic: hidden states may still let one shared matrix implement a
finite lookup. The non-gating N13 deficit-share and functional-rank extensions
were not run after the primary stop rule. No semantic-sharing inference follows
from rank or concentration.

The N2/N4 RP1.10 monitor was fragile (`39 / 65` and `53 / 123` at IoU50), so
the accepted payloads are not described as decode-robust policies. N13 RP1.10
and the optional identity-permutation null were skipped after
`HUMAN13_OVERFIT_PASS`; neither can revoke or strengthen the registered primary
claim.

## Stop boundary

The unit's N13 success stop rule has fired. No larger surface, per-image payload,
rank escalation, architecture promotion, or production action follows.
`HOLD_PRODUCTION` remains unconditional. A held-out or distributional test must
be a separate successor with its own frozen estimand and leak-free protocol.
