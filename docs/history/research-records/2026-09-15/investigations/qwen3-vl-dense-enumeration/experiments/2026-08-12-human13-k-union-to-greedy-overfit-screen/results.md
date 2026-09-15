---
title: Human-13 K-Union-to-Greedy Overfit Screen Results
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-08-12-human13-k-union-to-greedy-overfit-screen
topic: qwen3-vl-dense-enumeration
status: complete_bounded_same_panel
updated: 2026-08-12
---

# Human-13 K-Union-to-Greedy Overfit Screen Results

## Disposition

`COMPLETE_BOUNDED_SAME_PANEL_NARROW`.

The native language-tower DoRA/AdamW path can move sampled-but-greedy-missed
owners into an unforced original-prompt greedy completion on these exact
thirteen images. The useful dose is extremely small: every executed treatment
has its best gain/retention region at one panel exposure, while later exposures
produce owner exchange, duplication, malformed rows, output growth, and cap
stops. No executed arm safely compiles the full K-hit union into greedy.

This is an overfit-only optimization result. It is not validation or evidence
of cross-image generalization.

## Frozen evidence

Artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-12-human13-k-union-to-greedy-overfit-screen
```

- Source: exact HF fp32/SDPA, physical batch one, original prompt, greedy,
  repetition penalty `1.0`, thirteen images.
- Discovery: `208` vLLM requests = thirteen images times sixteen explicit
  `n=1` seeds, submitted as four physical batches of four per image at
  repetition penalty `1.10`.
- Manifest SHA-256:
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.
- Frozen owner ledger: `392` GT owners; `G=173`, `H=73`, `M=146`; `73`
  selected native H rows; `12` duplicate events; `12` target-bearing images.
- Matrix plan SHA-256:
  `f5600850c724339a5901d15a47ba2abb21303a852028b7709d4853b322a18b1a`.
- Combined raw-output ledger: `338` rows = frozen Source plus five arms times
  five milestones times thirteen images; SHA-256
  `054b548f082d8d2a16fb4588e05b976763796b9e6266e82f6f0d446829aff481`.
- Authoritative analysis: `analysis/matrix-v1-analysis.json`, SHA-256
  `cff3f9f26ba61417a74e06bf8fce178c20f067e51528e95e6329a0b0a4cfaa8e`.
- Posthoc plan/eval crosswalk and mechanical-disposition record:
  `analysis/execution-reconciliation-v1.json`, SHA-256
  `f2bb0cd76047452463d19177fa8b39d15a8ba02d55fab3757f52a3acc22f7fda`.
  This record does not replace the missing contemporaneous A4/A6/census
  stderr receipts.

Chronological class-agnostic pred-pred IoU `>0.95` exclusion is applied before
the final cardinality-first, maximum-total-IoU one-to-one owner matcher. Later
duplicates receive no owner credit.

## Complete pooled table

`H+`, `G-`, and `M+` are owner identities, not row counts. `Rows/Tok` are
generated output burden. `Dup`, `Malformed`, and `Cap` are kept separate from
owner coverage.

| Arm | Exposure | Unique | H+ | G- | M+ | Rows | Tok | Dup | Unmatched | Malformed | Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Source | 0 | 173 | 0 | 0 | 0 | 255 | 2360 | 9 | 62 | 11 | 0 |
| A0 | 1 | 173 | 2 | 3 | 1 | 239 | 2215 | 12 | 54 | 0 | 0 |
| A0 | 2 | 171 | 6 | 8 | 0 | 232 | 2153 | 8 | 53 | 0 | 0 |
| A0 | 4 | 174 | 10 | 10 | 1 | 234 | 2170 | 9 | 50 | 1 | 0 |
| A0 | 8 | 161 | 5 | 18 | 1 | 558 | 5079 | 61 | 65 | 271 | 1 |
| A0 | 16 | 165 | 8 | 17 | 1 | 558 | 5079 | 59 | 62 | 272 | 1 |
| A1 | 1 | 174 | 5 | 4 | 0 | 249 | 2306 | 15 | 59 | 1 | 0 |
| A1 | 2 | 170 | 6 | 10 | 1 | 305 | 2809 | 63 | 65 | 7 | 0 |
| A1 | 4 | 163 | 8 | 19 | 1 | 633 | 5757 | 182 | 84 | 204 | 1 |
| A1 | 8 | 169 | 10 | 18 | 4 | 704 | 6395 | 229 | 127 | 179 | 1 |
| A1 | 16 | 170 | 11 | 17 | 3 | 1339 | 12100 | 272 | 184 | 713 | 3 |
| A3 | 1 | 177 | 6 | 3 | 1 | 305 | 2810 | 35 | 83 | 10 | 0 |
| A3 | 2 | 174 | 4 | 6 | 3 | 305 | 2809 | 34 | 85 | 12 | 0 |
| A3 | 4 | 168 | 9 | 18 | 4 | 647 | 5883 | 182 | 97 | 200 | 1 |
| A3 | 8 | 169 | 10 | 16 | 2 | 1042 | 9434 | 253 | 142 | 478 | 2 |
| A3 | 16 | 168 | 10 | 20 | 5 | 2155 | 19885 | 367 | 206 | 1414 | 6 |
| A7 | 1 | 176 | 5 | 2 | 0 | 310 | 2855 | 58 | 72 | 4 | 0 |
| A7 | 2 | 173 | 6 | 9 | 3 | 312 | 2872 | 41 | 93 | 5 | 0 |
| A7 | 4 | 170 | 9 | 15 | 3 | 660 | 5999 | 180 | 107 | 203 | 1 |
| A7 | 8 | 167 | 8 | 17 | 3 | 2181 | 19813 | 517 | 218 | 1279 | 6 |
| A7 | 16 | 168 | 9 | 21 | 7 | 2483 | 22829 | 499 | 252 | 1564 | 7 |
| Full GT | 1 | 188 | 12 | 5 | 8 | 338 | 3112 | 54 | 92 | 4 | 0 |
| Full GT | 2 | 174 | 12 | 18 | 7 | 689 | 6270 | 169 | 134 | 212 | 1 |
| Full GT | 4 | 178 | 17 | 20 | 8 | 817 | 7439 | 269 | 164 | 206 | 1 |
| Full GT | 8 | 188 | 24 | 16 | 7 | 749 | 6811 | 234 | 139 | 188 | 1 |
| Full GT | 16 | 187 | 23 | 19 | 10 | 760 | 6919 | 257 | 154 | 162 | 1 |

## Observed

### The native optimization surface is responsive on this panel

Full-GT body CE at one exposure moves unique owners from `173` to `188`:
`H+12`, incidental `M+8`, and `G-5`. This is direct clean-greedy evidence that
the frozen vision/aligner plus language-tower DoRA path has a responsive native
optimization surface on the panel. An external owner or gradient bridge is not
required to obtain an initial recall increase here. This does not establish
sufficient capacity to fit all `392` owners.

That control is not safe or selective: rows increase `255 -> 338`, duplicates
`9 -> 54`, and unmatched rows `62 -> 92`. It is capacity evidence, not the
winning Stage-1 recipe.

### The narrow K-hit treatments have a one-exposure trade-off region

- A3@1 has the largest pooled H gain among the executed one-exposure
  H-supervised narrow treatments: `H+6`, `G-3`, `M+1`, final unique `177`
  (`+4` net versus Source). Two H gains are paired with G losses on the same
  images, so this is partly owner exchange. It also raises duplicate burden to
  `35` and rows to `305`.
- A7@1 trades one fewer H gain for one fewer G loss: `H+5`, `G-2`, final
  unique `176`, but duplicates rise to `58`.
- A1@1 is the lower-output alternative: `H+5`, `G-4`, final unique `174`,
  rows `249`, duplicates `15`.

No single arm dominates all axes. A3@1 is not a Pareto winner: it has the
largest pooled H gain, A1@1 has the smallest targeted burden and the largest
safe-image count, and A7@1 has one fewer pooled G loss but much higher
duplication. A0@1 already contributes `H+2/G-3` with lower burden, so A3 adds
roughly four pooled H gains over the shared background without improving
pooled G loss.

### Repeating optimization does not snowball safely

For A1/A3/A7, more exposures raise some H counts but lose at least as many G
owners and rapidly inflate output. At exposure sixteen:

- A3: `H+10`, `G-20`, `2155` rows, `367` duplicates, `1414` malformed, six
  cap stops.
- A7: `H+9`, `G-21`, `2483` rows, `499` duplicates, `1564` malformed, seven
  cap stops.

Thus the proposed sample-by-sample `optimize-until-satisfied` loop is rejected
under this exact static-target objective. It would select into memorization,
owner exchange, and output-length failure long before stable set mastery.

### Source replay mitigates late output pathology, not owner loss

A3 and A7 share independent H1 supervision and duplicate correction; A7 omits
Source-body replay. At exposure eight, A3 versus A7 is `9434` versus `19813`
generated tokens, `253` versus `517` duplicates, and two versus six cap stops.
At exposure sixteen the corresponding values are `19885/22829`, `367/499`,
and `6/7`. The G-loss comparison changes sign across milestones and never
supports replay-based owner preservation. Replay only shows late-dose burden
mitigation here, and A3 still collapses.

### The common background is not neutral

A0 contains Source replay plus duplicate correction and no H target. It still
moves H owners: by exposure four it has `H+10`, `G-10`, `M+1`. Therefore an H
gain cannot be attributed to target suffixes without the A0 contrast. At one
exposure, A3 adds four H gains over A0 while holding pooled G loss at three,
but it also adds rows and duplicates.

## Mechanical dispositions

- Full Source/K acquisition and the sealed `13/392` manifest completed.
- The one-image A1 vertical slice completed one real packed update, checkpoint
  write/read, HF clean-greedy readout, and analyzer. On image 14038 it gained
  one H owner and lost one G owner; pooled over the panel it was `H+4/G-2`,
  proving mechanics but not clean consolidation.
- Full-GT, A0, A1, A3, and A7 completed all sixteen finite optimizer updates
  and checkpoints `1/2/4/8/16`. All use zero-padding packing.
- A4 was not executed. Its independent full multimodal candidate bundle is
  atomic; exact processor lengths exceed the `12000` bound for eleven of the
  twelve target-bearing images. A deterministic posthoc CPU preflight
  reproduces the first failure at image 1584 (`20846` tokens). No multi-pass
  fallback was invented, and the original launcher stderr was not persisted.
- A6 failed closed before model load, forward, or update because the encoded
  donor prefix mismatched sealed donor provenance. The failure is reproducible
  in CPU payload validation, but the original stderr was not persisted and a
  plan-side binding defect remains possible. It was not retried.
- A8-prime was not executed. The no-update census produced no artifact after
  three live-entry attempts: prompt identity and FA2-device bugs were fixed,
  then the HF coherent segment still carried an invalid prompt boundary. Per
  the declared stop rule, no third repair was attempted and no partial logits
  were interpreted.

Per-exposure packing/runtime:

| Arm | Packs | Logical = packed tokens | 16-update wall s | Peak bytes | Eval wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full GT | 2 | 20780 | 111.6 | 13715179008 | 2338.8 |
| A0 | 4 | 37618 | 144.4 | 10035874304 | 1324.4 |
| A1 | 5 | 56268 | 209.5 | 9878460416 | 2256.5 |
| A3 | 13 | 148594 | 537.2 | 9907854336 | 3150.3 |
| A7 | 12 | 129172 | 463.1 | 9643120640 | 4135.3 |

Packing removed padding but did not reuse image or prefix computation. The
clean-greedy evaluation, not training, dominated wall time once output grew.

## Supported

1. A native, very-low-dose parameter update can consolidate several K-hit
   owners into greedy on this panel.
2. The most promising immediate native recipe is a single exposure of
   owner-balanced H1 or full-residual body CE with an explicit preservation
   constraint; do not continue the same batch until mastery.
3. K-miss owners are not necessarily visually absent: full-GT@1 recovers eight
   M owners. A later support-expansion contrast is justified, but it must keep
   H, M, and G ledgers separate.
4. Some preservation mechanism remains necessary. Source replay is evidence
   only for late output-burden mitigation, not for Source-owner preservation.

## Ruled out for the next route

- External owner/gradient bridge as the first remedy.
- Shared K-union reward or GRPO-style within-group union credit as a mechanism
  for compiling multiple samples into one greedy path.
- Static-target optimize-until-satisfied on one image/batch.
- Sixteen-exposure dosing of the present A1/A3/A7 losses.
- Treating duplication or retention as secondary metrics.

## Unresolved

- Whether a calibrated one-update mixture can keep the full-GT@1 recall gain
  while restoring Source owners and controlling duplication.
- Whether preservation should be Source row CE, sparse frozen-logit anchoring,
  or an explicit constrained gradient step. The screen shows the need, not the
  optimal mechanism.
- Whether A4 any-valid mass, A6 donor prefixes, or A8 token-rank crossing help;
  their contrasts are mechanically absent, not negative scientific results.
- Generalization and fresh-image transfer are entirely unmeasured.

## Next decision

If the user authorizes a successor, restart from Source and compare only a
small low-dose frontier:

1. A3-style owner-balanced H1 at one exposure;
2. full residual body CE at one exposure, with `H` and `M` weights reported
   separately; and
3. the same target direction with a stronger preservation constraint.

Select on the Pareto tuple `(H gained, G lost, M gained, duplicate/malformed/
cap burden, generated tokens)`. Do not resume any checkpoint from this screen,
do not add a long run, and do not update after observing the same batch's
post-update decode.

## Not claimed

No validation performance, transfer, population prevalence, production loss,
architecture necessity, full-set mastery, safe optimizer direction, or
duplicate-free decoding is claimed. The result does not establish that K-miss
owners are learned representations; only their same-panel supervised
recoverability was observed.

## 2026-08-13 missing-arm successor

This bounded successor repaired and executed the three mechanically missing
contrasts without changing the sealed `13/392` owner ledger or reusing a
trained checkpoint. Its authoritative analysis is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor/analysis/a4-a6-final.json`
(SHA-256 `2f2ac938ad9fe553c2cd87c3684a65afe2f4331d01d4b0e806fb923992b64e98`).

### Mechanical recovery

- A4 now preserves its one-per-image prefix-free union estimand while allowing
  physical splitting. At fixed parameters it scores every candidate, forms one
  global fp32 softmax weight vector per image, replays the identical candidates
  with detached weights, accumulates gradients over all packs, and applies one
  AdamW step. The production-shaped image-14038 slice covered 31 candidates in
  five score forwards and six replay packs, produced a finite gradient, and
  wrote/read checkpoint step 1. The full panel used 309 candidates, 44 score
  forwards plus 45 replay forwards per exposure, and no padding.
- A6 cleanly removed every earlier frozen duplicate-row span from each donor
  prefix. All `73/73` sealed donor bindings matched, including image 16228 owner
  `gt:16228:19`; the full arm then completed sixteen finite updates.
- A8 produced a complete immutable no-update census rather than another partial
  failure. Across 680 aligned sites all values were finite, but maximum packed
  versus HF target-margin drift was `1.4143247604`, yielding required margin
  `1.4144247604 > 0.5`. A8-prime was therefore mechanically omitted, not
  interpreted as an algorithmic null. The census is
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor/census/v2/no-update-census.json`
  (SHA-256 `82d3f87883121ca7e215dc9d919660c5e426e4fc190a5e69f5406b35fa27235d`).

### Original-prompt clean-greedy outcome

All rows below use HF fp32/SDPA, physical batch size one, greedy decoding, and
repetition penalty 1.0. `H+` is newly recovered frozen K-hit owners, `G-` is
lost Source owners, and `M+` is incidental K-miss recovery.

| Arm | Exposure | Unique owners | H+ | G- | M+ | Rows | Duplicates | Unmatched | Malformed | Cap stops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source | 0 | 173 | 0 | 0 | 0 | 255 | 9 | 62 | 11 | 0 |
| A4 | 1 | 175 | 4 | 2 | 0 | 310 | 34 | 86 | 15 | 0 |
| A4 | 2 | 181 | 8 | 5 | 5 | 311 | 32 | 84 | 14 | 0 |
| A4 | 4 | 182 | 9 | 8 | 8 | 554 | 88 | 130 | 154 | 1 |
| A4 | 8 | 171 | 8 | 19 | 9 | 2432 | 570 | 206 | 1485 | 7 |
| A4 | 16 | 184 | 17 | 14 | 8 | 2438 | 483 | 238 | 1533 | 7 |
| A6 | 1 | 169 | 2 | 6 | 0 | 238 | 11 | 58 | 0 | 0 |
| A6 | 2 | 175 | 7 | 5 | 0 | 244 | 11 | 51 | 7 | 0 |
| A6 | 4 | 168 | 13 | 19 | 1 | 632 | 112 | 86 | 266 | 1 |
| A6 | 8 | 168 | 13 | 19 | 1 | 576 | 76 | 67 | 265 | 1 |
| A6 | 16 | 171 | 14 | 17 | 1 | 571 | 69 | 66 | 265 | 1 |

### Interpretation

A4 supplies direct same-panel evidence that globally fused K-native union
information can be compiled into greedy coverage: at exposure two it reaches
`181` unique owners (`H+8/G-5/M+5`), and at exposure four it reaches `182`.
This is not a safe recipe. Output burden grows first and then collapses into a
long-generation regime; exposures eight and sixteen each generate about 22.5k
tokens with seven cap stops. Their five-checkpoint evaluation takes 4272.3 s,
while A4 training takes 2483.3 s and 45 replay packs per exposure.

A6 has a narrower low-dose window. Exposure two reaches `175` owners with
`H+7/G-5`, 244 rows, and 11 duplicates, but later exposures exchange owners and
inflate output. A6 training takes 505.0 s with 13 packs per exposure; it is much
cheaper than A4 but does not snowball safely.

Both arms did receive frozen duplicate-event unlikelihood. That objective only
covered 12 selected duplicate atoms from the frozen Source/K ledger, versus
309 A4 union candidates and 2909 H target tokens. During A4 exposures one
through eight, the H raw loss falls `20.43 -> 15.05`, while duplicate raw loss
stays approximately `0.031-0.033`: the registered old duplicate branches are
already low-probability. The later greedy duplicates arise at newly reached
prefixes or token/box aliases that the static ledger never supervised.
Consequently this result does not reject duplicate unlikelihood; it rejects a
static twelve-event unlikelihood set as sufficient preservation for repeated
union updates.

### Successor disposition

The bounded successor is complete. A4 and A6 are promoted only as low-dose
same-panel algorithm evidence, not checkpoints or general recipes. No
100-exposure continuation, K-miss supervision, checkpoint promotion, stable
spec sync, or OpenSpec archive is authorized. The smallest next discriminator
is a fresh-Source low-dose A4 contrast in which duplicate negatives are
refreshed from the current natural rollout and STOP/row-count preservation is
co-primary. It must retain the exact global union normalization; chunk-local
softmax remains an invalid substitute. Blindly increasing the coefficient on
the same static twelve duplicate events is not the preferred contrast.
