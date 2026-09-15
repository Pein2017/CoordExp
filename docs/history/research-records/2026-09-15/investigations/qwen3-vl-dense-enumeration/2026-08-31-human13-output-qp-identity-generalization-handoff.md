---
title: Human13 shared output-QP same-panel overfit handoff
description: Phase-boundary transport from the Image2299 finite-policy witness to a fixed-panel 13-image shared-output overfit test.
type: investigation
role: research-handoff
authority: transport_only
status: human13_overfit_pass
updated: 2026-09-02
---

# Human13 shared output-QP same-panel overfit handoff

Direction instantiated on `2026-08-31` at
`/data/CoordExp/.worktrees/human13-output-qp-identity-generalization`, branch
`probe/human13-output-qp-identity-generalization`. The active research
authority is the [Human13 shared output-QP same-panel overfit unit](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/unit.md).
This handoff transports context only.

## Semantic correction before execution

The original handoff incorrectly made held-out generalization and leave-one-out
transfer the active question. The user corrected the intent before any Human13
model forward:

> Image2299 proved that one fixed image could be overfit. First test whether
> the same output-only mechanism can overfit thirteen fixed images. Measure
> real generalization only after that capacity result exists.

The stable filename, branch, and unit ID retain the old
`identity-generalization` label for provenance and backlinks. They do not own
the revised estimand. The retired held-out plan created no evidence.

## What Image2299 established

The completed [Image2299 continuation](2026-08-31-image2299-augmented-greedy-38-to-46-owner-continuation.md)
contains a fresh-cold ordinary-greedy `46 / 46` witness. Its strongest direct
path is the [canonical G46 global QP](experiments/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md):

\[
z(h)=hE^\top+h\Delta W_{out}^\top.
\]

The language tower, multimodal aligner, vision tower, and input embedding stay
frozen. The payload changes output logits only. The bounded claim is that one
oracle-authored complete route on one fixed image can be compiled into a
natural-greedy output policy. It is not base-model learning or transfer.

A post-closeout audit found `99.1492%` first-singular-direction energy, one
coordinate-token row holding almost all parameter energy, and six hard token
decisions holding `87.10%` of squared deficit mass. Therefore total norm or
parameter rank alone cannot identify a semantic mechanism. Human13 must also
measure functional logit-change rank and row/constraint concentration.

## Correct Human13 question

> From the exact four-coordinate `geo_sorted_xy=(x1,y1)` step-2444 Source, can
> one output-only `Delta W_out` shared across the fixed Human13 panel compile
> all 392 canonical GT owners under fresh-cold original-prompt natural greedy,
> with strict owner validity, zero hard debt, and natural EOS?

All thirteen target routes and states may enter the joint fit. That is the
point of a same-panel overfit test. The one-matrix requirement still excludes
image-ID branches, per-image payloads, per-image checkpoints, and test-time
payload selection. A finite state-to-token lookup implemented inside the
shared matrix counts as overfit success; it simply does not count as
generalization.

The older `12 target-bearing + 1 preservation-only` statement came from a
narrow missing-owner ledger. It is wrong for this full-GT predicate. Here all
13 images are target-bearing: legacy-12 contributes 346 owners and Image2299
contributes 46.

Primary success is strict same-category global one-to-one actual-pixel
Intersection over Union at least `0.5` coverage of `392 / 392`, zero confirmed
duplicate/unsupported/malformed/cap debt, and valid natural EOS at repetition
penalty `1.0`. Report legacy-12 and Image2299 separately before pooling. Exact
canonical token replay is a stronger diagnostic, not a second success gate.
Repetition penalty `1.10` is a nonblocking robustness monitor.

## Live frozen inputs

The historical panel was rebuilt rather than replaced by a new semantic
substitute. Twelve raw authority rows were preserved byte-for-byte except for
the historical relative image-prefix normalization; the expected legacy hash
then matched exactly. The existing `research-base-v2` builders reproduced the
13-row admission panel and x-then-y derivative. All five published hashes now
match:

| Identity | SHA-256 |
|---|---|
| legacy-12 JSONL | `cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85` |
| admitted-13 JSONL | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| admission receipt | `ad78c174897509dca07c24c57d12c897fdad42724d83af08147d79c9644c5414` |
| `geo_sorted_xy` JSONL | `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23` |
| x-then-y receipt | `cd1273f627f7bdfcb16e9ca6e4a50e9d51f0163081ea7458d6d3deabe9bb82c0` |

The decision input is:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl`.

The mainline Source is intentionally independent of the Image2299 QP lineage:

- checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- base:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- adapter SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- special-embedding SHA-256:
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`
- prompt config SHA-256:
  `d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b`.

## Smallest staged program

The authoritative unit freezes the exact solver and checks. The operational
ladder is:

1. **Stage -1:** identity, Source decode, zero-delta parity, canonical-route
   ledger, and Source-A -> candidate-B -> Source-A contamination check.
2. **N=2:** one delta for images `6040, 16228` and all 65 owners.
3. **N=4:** one delta for nested images `4134, 6040, 13923, 16228` and all 123
   owners.
4. **N=13:** one delta for all 392 owners.

N=2 failure stops N=4; N=4 failure stops N=13. N=13 success stops the unit and
reports same-panel overfit. There are no folds. The pooled protected-null rank
is a capacity monitor, not a gate. Solver infeasibility is a bounded negative
for this output surface, not evidence that the representation lacks semantic
information.

## Execution update

The [completed results](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/results.md)
supersede this handoff's pre-launch state. N2 passes on all 65 owners, N4 on all
123 owners, and N13 on all 392 owners at IoU50, IoU60, and IoU80 under RP1.0,
with zero hard debt, natural EOS, exact canonical replay, and exact
Source-A/B/A restoration. The strongest current claim is therefore complete
thirteen-image same-panel shared-output compilation.

The accepted N13 run is
`20260831T-n13-v5-certificate-polish-corrected`. It retains the frozen margin,
certificate tolerance, targets, constraints, and objective; bounded continuation
plus same-active-problem certificate polish closes the earlier numerical
boundary. Its maximum exhaustive FP64 violation is
`1.0842741027028424e-05` against tolerance `2e-5`, and its minimum FP32
hook margin is `0.00998687744140625`. Thirteen fresh candidate processes pass,
and thirteen independent Source-only processes exactly restore generated-token
hash, parser-text hash, stop reason, and generated-token count. The immutable
acceptance receipt classifies the result as `HUMAN13_OVERFIT_PASS`.

At every N, bind provenance and resource counters; constraint/slack/certificate
counts; norm and rank scaling; largest row and constraint; per-image owner and
hard-debt behavior; warm/cold parity; and the functional effective rank of
`H_eval Delta W_out^T`. Run one sensitivity only if the largest row reaches
`90%` residual energy or one constraint reaches `50%` of squared baseline-
deficit mass.

## Intuition for the stronger null

Suppose the true route needs coordinate tokens `100, 200`, while an easy fake
route uses `600, 700`. If the true route fits and that fake route fails, the
result may only say that some token rows are easier to move; it does not show
that the correct owner-state pairing matters.

The stronger identity-permutation null keeps the same target-token multiset
and matched local difficulty, but shuffles which hidden-state decision is
paired with which target token. In plain language: keep the same answer pieces
and the same exam difficulty, but assign the answer pieces to the wrong
questions.

- If the shuffled pairing fits just as cheaply, the output head behaves like a
  finite lookup/capacity device.
- If the correct pairing needs materially less norm/rank/slack, panel-bounded
  semantic alignment becomes plausible.

For the current objective, **both outcomes would remain valid 13-image
overfit**. The optional null was not run after the primary success stop rule
fired.

## Optional Image2299-payload transfer ablation

The Image2299-trained information question was tested independently. The
registered contrast is native `R` versus `R+Q` on the legacy-12 images,
with no refit, scaling, sign choice, row deletion, or payload selection:

- `R`:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step`
- `Q`:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/20260831T-image2299-canonical-g46-global-qp-v3/g46_global_qp_protected_null_output_residual.safetensors`
- `Q` SHA-256:
  `22d9df392f586979e8b191a89068ea929943356d8bfce86856e8203c0e0f51fd`.

The one immutable [transfer result](experiments/2026-08-31-image2299-g46-payload-legacy12-transfer/results.md)
is strongly negative. Strict IoU50 coverage falls from `138 / 346` under
`R` to `18 / 346` under `R+Q` (`delta50=-120`); hard debt rises from
`434` to `4004`, and eleven of twelve treatment decodes hit the cap. The
Image2299 positive control remains `46 / 46` with zero hard debt, and exact
`R -> R+Q -> R` restoration passes. This is negative behavioral transfer of
that fixed payload on that fixed cohort conditional on `R`. It does not test
`Q` portability to step-2444, distributional generalization, or the newly
fitted Human13 residual.

## Continuation and authority boundary

The completed implementation reused only the existing step-2444 loading,
prompt, cold-decode, and strict owner-evaluation seams; it did not restore the
retired Human13 runner or copy the Image2299 specimen-bound import chain.

Current state: input recovery and the N2/N4/N13 ladder are complete; the N13
success stop rule has fired. No further same-panel optimization, null, RP sweep,
or larger parameter surface is required by this unit. `HOLD_PRODUCTION`
remains unconditional.

## Later direction-local successors

The output-QP success did not establish held-out behavior. The later
[Human13 magnitude-only finite-overfit result](experiments/2026-09-01-human13-dora-magnitude-finite-overfit/results.md)
compiled all `392 / 392` fixed-panel owners with one shared unmerged
magnitude-only adapter per stage. That is fixed-panel internal-network
programmability only.

The predecessor [G0 result](experiments/2026-09-01-scalable-annotated-owner-shared-dora-pilot/results.md)
stopped before gradients because its registered exact whole-bundle K1 null was
infeasible in 101 image-imbalanced strata. That closes only the frozen control
route.

The subsequent [Direct C-D0 result](experiments/2026-09-01-annotated-owner-direct-c-d0-pilot/results.md)
completed the image-disjoint test and returned
`SCIENTIFIC_STOP_DIRECT_C_D0`. D0 matched 606 IoU50 owners versus C's 630 on
the frozen 128-image, 891-owner screen, retained
`586 / 614 = 95.44%` of Source-covered owners, and produced one capped decode.
The action is to stop this frozen objective without rescue. The result does not
rule out shared DoRA, actual-prefix learning generally, refreshed-prefix
successors, or full-COCO training. The next separate phase begins with a
C-anchored zero-update mechanism audit; constrained optimization is
conditional. `HOLD_PRODUCTION` remains unconditional.

## Minimal reading path

1. the completed [same-panel results](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/results.md)
   and [unit](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/unit.md);
2. the [Image2299 continuation](2026-08-31-image2299-augmented-greedy-38-to-46-owner-continuation.md);
3. the [direct G46 result](experiments/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md);
4. the [fixed-payload legacy-12 transfer result](experiments/2026-08-31-image2299-g46-payload-legacy12-transfer/results.md);
5. the [prospective panel-admission unit](experiments/2026-08-04-sorted-prospective-13-image-panel-admission/unit.md);
6. the [step-2444 owner-interface unit](experiments/2026-08-05-static-dynamic-owner-interface-crossover/unit.md).
