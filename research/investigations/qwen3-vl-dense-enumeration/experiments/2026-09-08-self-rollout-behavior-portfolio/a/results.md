# A: same-prefix downstream-value probe

Status: candidate, awaiting root acceptance. Fixed inference stop reached;
no optimizer update, model promotion, extra sample or further GPU run.

## Decision-bearing observation

The fixed16 images contain124 annotated owners; their single native Source
baselines cover90 at category-consistent pixel IoU>=0.50. All64 independently
sampled actions reached box-end; all64 full branches subsequently ended with
native EOS, without a3084-token cap.

All64 next actions are valid single rows, with four distinct sampled token
actions per image. One final trajectory (`287484`, sample2) has two
geometry-invalid later rows dropped by the native parser; the trajectory and
all recoverable predictions remain in the denominator. The other63 final
trajectories have fully accepted parses. There are no strict >0.95
category-agnostic geometric-repeat pairs in immediate or final outputs.

| Accounting unit | GT denominator | Matched | Unmatched valid predictions | Missing GT owners |
| --- | ---: | ---: | ---: | ---: |
| 16 single-greedy baselines | 124 | 90 | 40 | 34 |
| 64 prefix+action outcomes | 496 | 212 | 100 | 284 |
| 64 completed forced-prefix branches | 496 | 362 | 195 | 134 |

Branch sums repeat each image's denominator four times. They are not a
best-of-K union, not64 independent images, and not a deployed-model gain.

Among96 within-image unordered branch pairs,16 have equal immediate owner-count
increment but different final increment, spread across6/16 images. There are
two strict immediate-versus-final ranking reversals, both on image486123.
The96 pairs are not independent observations; this historically selected
training-side panel does not estimate population frequency or transfer.

This closes the narrow question affirmatively on the sampled support:
immediate count gain does not fully determine a row action's eventual native
greedy owner coverage. It does not establish an effective training objective,
a better deployed model, a missing ledger architecture, or benefits from
best-of-K selection/union.

## Concrete reversal and strongest limitation

For `coco2017_train_000000486123`, the exact midpoint prefix covers5 owners.
The four sampled actions are each10 tokens with the identical traffic-light
description and schema. Only their four coordinate-token positions differ.

| Sample index | Immediate increment | Final increment | Final full tokens |
| --- | ---: | ---: | ---: |
| 0 | 0 | 1 | 78 |
| 1 | 1 | 2 | 87 |
| 2 | 1 | 2 | 87 |
| 3 | 0 | 3 | 142 |

Thus equal starting states, action lengths and entity descriptions still yield
different later access. However sample3's box is pixel `[796,607,797,614]`, a
1x7-pixel annotation-unmatched box. Its subsequent route covers more GT owners
than the immediately matched alternatives. This is not proof of hallucination;
it is a concrete warning that downstream owner return is not a correctness or
groundedness score for the action receiving credit. A credit learner could
prefer an annotation-unmatched routing cue. Do not silently resolve this
user-owned tradeoff by adding an unknown/geometry penalty or by claiming the
proposal is safe/superior.

## Technical qualification and identity

The first baseline/branch qualification passed exact midpoint native-greedy
token replay, iid K4 sampling, frozen native continuation, native parser/global
cardinality-first matcher, and disk reload. A corrupted token identity was
rejected. Qualification took32.71 seconds and256 observed model forwards.

A fresh CPU consumer independently replayed160 token/parser/matcher artifacts
(16 full baselines,16 midpoint prefixes,64 immediate outputs,64 final outputs)
and all16 midpoint derivations. A corrupted owner count was rejected.
The package owner replayed all three reducer tests after its correction:
3 passed, including positive-prefix EOS support, appended-malformed action,
paired-return sensitivity, and the exact strict geometric-IoU boundary.

Attempt1 failed before model/GPU load because the root-produced audit panel
had absolute image refs while the native loader requires relative refs.
Attempt2 selected the identical canonical raw_train256 examples through frozen
IDs, original-row hashes and resolved media hashes. The sealed panel and its
cohort were not rewritten or changed. Attempt2 completed with process exit0.

- Elapsed attempt2:428.25 seconds, including cold model setup/qualification.
- Observed model forwards:4367; optimizer updates:0.
- Peak CUDA allocated/reserved:9.53GB/10.75GB (decimal).
- Peak host RSS:11.98GB (decimal).
- Source16 baseline SHA256:`0117a01b7d2b4b28489d18b6d71120af5f77053221cd732eaff217d1c177c077`.
- Branch-bank SHA256:`970446748b6b3c73937b3f09a06acf605f85e7da10d56185399ff47435a4cb7c`.
- Shared manifest SHA256:`b680f78e6fb8ded1ebc3b5c089aef62d8a42a1e3c839df853655f3c7be838261`.

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a/`.
Its immutable `baselines.jsonl` and `baseline-receipt.json` were published for B
before the remaining A branches completed. Each baseline, sampled action and
completed branch was persisted separately before final aggregate publication.
`runtime.json`, `qualification.json`, `run-receipt.json`,
`consumer-verification.json`, attempt logs and receipts bind the real execution.
`analysis.json` and output-root `results.md` provide full denominators, per-image
and per-branch owner gains/losses, support and parse/repetition diagnostics.

## Next design, not a launch

See `training-proposal.md`: one fresh-bank row-only RLOO update using iid raw
T1/top-p1/RP1 actions and frozen RP1.10 greedy suffix returns. No forced greedy
reference enters the iid baseline, and no prefix/suffix token receives loss.
Equal-return legal siblings are not canonicalized into negatives; lower-return
legal siblings and unknowns can nevertheless receive negative relative credit.
These observations justify discussing downstream credit, not executing training
or automatically extending the completed probe.

## CPU reproduction

Run from `/data/CoordExp/.worktrees/dora-prox-linear-n2` with the configured
Python environment:

```bash
python /data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-self-rollout-behavior-portfolio/a/verify_a.py
python /data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-self-rollout-behavior-portfolio/a/reduce_a.py
```

The verifier is CPU-only. The reducer atomically republishes derived outputs,
not raw evidence. The finished GPU runner must not be relaunched.
