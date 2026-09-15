---
title: Next-Row Likelihood Change and Causal Source Trace Results
description: Exact-row scoring supports local owner suppression and a geometry-structured successor effect, while the natural equal-depth discriminator stopped at its admission gate.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-next-row-probability-transition-and-causal-source-trace
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: stopped_at_natural_sibling_admission_gate
updated: 2026-07-17
---

# Next-Row Likelihood Change and Causal Source Trace Results

## Verdict

This unit establishes a high-confidence but local result:

> Appending one naturally generated complete object row sharply reduces the
> exact-row likelihood of generating that same physical owner again.

The result is not explained solely by category anti-repetition or by a generic
increase in row depth. On image `2299`, all scored objects are people, yet
appending the row for one person suppresses that person while increasing four
frozen row variants owned by two other people. The change is concentrated in
the first horizontal box coordinate, `x1`, rather than in the description or
the row-entry-versus-terminal decision.

The broader claim that a commit uniformly redistributes probability to every
uncovered object is false in this panel. Other-object changes are positive,
near zero, or negative depending on the owner and prefix. A geometry-sorted
successor or moving spatial frontier remains a plausible explanation.

The unit does **not** establish an explicit object ledger, a persistent covered
set, or a general physical-object commit mechanism. The required natural
equal-row-depth discriminator was not admitted: after the initial 32 samples
and the predeclared cap of 128 additional samples, no alternative owner had at
least three distinct exact natural row variants. Per the stop rule, no layer or
source trace and no training experiment was run.

## Evidence Identity

All exact-row scores used the geometry-sorted Qwen3 Vision-Language
2-billion-parameter adapter at checkpoint step `4887`:

```text
/data/CoordExp/.worktrees/research-probes/configs/coordexp_infras/infer/
  qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml
```

The historical configuration component `gaussian_rps` means Gaussian
soft-target coordinate cross-entropy plus a cumulative-distribution
regularizer. The adapter also uses Weight-Decomposed Low-Rank Adaptation
(`DoRA`). This checkpoint is not a pure cross-entropy baseline.

The exact-row scorer used full teacher forcing, no key-value cache, no
repetition penalty, Scaled Dot Product Attention (`SDPA`), and 32-bit floating
point (`float32`) log-softmax accumulation. The `float32` confirmation runs
materialized all `2,149,097,472` parameters as `float32`.

The image-`15254` configuration-precision and full-`float32` receipts are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  smoke-image15254-row0-config-v3/receipt.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  stage1-image15254-row0-fp32-v1/receipt.json
```

The image-`7574` receipts are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  stage1-image7574-row0-config-v1/receipt.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  stage1-image7574-row0-fp32-v1/receipt.json
```

The image-`2299` native-prefix factorial receipts are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  stage1-image2299-prefix-factorial-config-v1/receipt.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  stage1-image2299-prefix-factorial-fp32-v1/receipt.json
```

The request-scoped sampling runtime was independently attested at temperatures
`0.2`, `0.4`, and `0.6`:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  runtime-attestation-current-v1/
  request-scoped-sampling-three-policy-cuda.json
```

For all three policies, the attestation records exact request-order replay,
exact result-artifact replay, exact selected-token-score replay, and identical
stock-versus-custom processed-logit tensor hashes. Its aggregate payload
fingerprint is:

```text
2bcb65e374938117ddfd807606eabdbfeb2c6ebb8311c97c6700048cb415c0b6
```

## Row-Zero Exact-Owner Suppression

The tables below report paired changes in full-row summed log-likelihood after
appending the named donor row. Negative means the same frozen candidate row
became less likely.

### Image 15254

| Appended donor owner | Scored owner | Frozen variants | Full-row changes in `float32` |
|---|---|---:|---:|
| bowl | bowl | 2 | `-9.1038`, `-8.0172` |
| bowl | carrot | 2 | `+0.6882`, `+0.3101` |
| carrot | carrot | 2 | `-11.4332`, `-13.0001` |
| carrot | bowl | 2 | `-1.6108`, `-1.1347` |

Each appended row strongly suppresses its own owner. The other-owner response
is not uniform: a bowl row slightly raises the carrot variants, while a carrot
row also lowers the bowl variants. Because the bowl and carrot are spatially
nested, this image cannot by itself distinguish physical-object commit from
local semantic or geometric interaction.

### Image 7574

| Appended donor owner | Scored owner | Frozen variants | Full-row changes in `float32` |
|---|---|---:|---:|
| bowl | bowl | 2 | `-5.9958`, `-6.0201` |
| bowl | microwave | 2 | `+0.2363`, `-0.0194` |
| microwave | microwave | 2 | `-8.9358`, `-8.3629` |
| microwave | bowl | 2 | `-7.1714`, `-6.9532` |

Again, own-owner suppression is large and sign-consistent. The bowl-to-
microwave effect is approximately zero and variant-dependent, while the
microwave row substantially lowers both microwave and bowl variants. This
rules out broad, uniform uncovered-object redistribution in this case.

Across images `15254` and `7574`, all eight donor-owned frozen variants have a
large negative change, ranging from `-5.9958` to `-13.0001`. Every full-row
change in both tables has the same sign under configuration precision and
full-model `float32`. The largest absolute precision difference among these
paired full-row changes is `0.8088` log units on image `15254` and `0.4743` on
image `7574`; neither changes the interpretation.

Image `12576` was not scored because the available historical boundary did not
match the active prompt exactly. It was excluded rather than repaired or
spliced.

## Image 2299: Same-Category Native Prefix Factorial

Image `2299` is a dense scene with several spatially separated people. The
frozen native chain is:

```text
P0: prompt only
PA: prompt + person A
PAB: prompt + person A + person B
```

`P0`, `PA`, and `PAB` are exact contiguous native states. A fourth state that
contains person B without person A is a forced replay and remains descriptive
only; it does not own any native-state conclusion.

### Appending person A at row zero

Under full-model `float32`, the `P0` to `PA` transition changes the frozen rows
as follows:

| Scored owner | Full-row change |
|---|---:|
| person A | `-7.1465` |
| person B | `+3.9752` |
| person C, four variants owned by two people | `-0.3362` to `+0.8172` |

This resembles a local geometry-sorted successor: the immediate next person
becomes much easier, while later people do not rise uniformly.

### Appending person B after person A

Under the native `PA` to `PAB` transition:

| Scored owner | Full-row change | Description change | `x1` change |
|---|---:|---:|---:|
| person A | `-0.1022` | `+0.0034` | `-0.4786` |
| person B | `-6.2517` | `+0.0034` | `-3.8327` |
| remaining person, variant 1 | `+2.4343` | `+0.0034` | `+2.6337` |
| remaining person, variant 2 | `+2.4160` | `+0.0034` | `+2.8408` |
| remaining person, variant 3 | `+2.0100` | `+0.0034` | `+1.9604` |
| remaining person, variant 4 | `+1.3919` | `+0.0034` | `+1.2824` |

All rows share the description `person`. Category anti-repetition therefore
cannot explain why person B falls while four other person rows rise. The
description contribution is essentially unchanged; almost the entire
owner-specific separation appears in geometry, led by `x1`.

The row-entry-minus-terminal margins remain strongly positive:

| Native state | Configuration precision | Full-model `float32` |
|---|---:|---:|
| `P0` | `11.6250` | `11.7224` |
| `PA` | `11.1250` | `11.1442` |
| `PAB` | `11.0000` | `11.0252` |

The model remains strongly in continue mode throughout. A broad change in
terminal preference is therefore not the cause of the owner-specific matrix.

Configuration precision and full-model `float32` agree on the sign of every
full-row change in both native transitions. For the key `PA` to `PAB`
transition, person B is `-6.2938` versus `-6.2517`, and the four remaining rows
are `+1.4332` to `+2.3034` versus `+1.3919` to `+2.4343`. The largest absolute
full-row precision difference in this transition is `0.3099` log units.

This precision agreement is intentionally restricted to conclusion-owning
native transitions. One near-zero cell in the descriptive forced-B-at-row-zero
replay changes from `-0.0382` at configuration precision to `+0.0113` in
full-model `float32`. That forced state and its factorial interaction remain
descriptive only.

## Natural Equal-Depth Discriminator and Stop Rule

The remaining ambiguity was whether the image-`2299` pattern represented a
physical-object commit or only a one-way geometry-sorted frontier. The planned
discriminator required at least two different physical owners to occur
naturally as the next row from the exact same `PA` boundary, with at least
three distinct exact row variants for each alternative owner. That would have
allowed reciprocal, equal-row-depth scoring without forcing an off-support
history.

The initial 32-sample wave is recorded at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  wave1-image2299-after-owner0002-discovery-k32/
  owner-admission/admission-manifest.json
```

It produced 31 samples for person owner `0001` and one sample for person owner
`0006`. Owner `0001` was admitted; owner `0006` had only one exact variant.

The first 64 additional samples are recorded at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  wave1c-image2299-after-owner0002-discovery-k64/
  owner-admission/admission-manifest.json
```

They produced 62 samples for owner `0001`, one for owner `0006`, and one
unmatched row. The second 64 additional samples are recorded at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/
  wave1d-image2299-after-owner0002-discovery-k64/
  owner-admission/admission-manifest.json
```

They produced 60 samples for owner `0001`, two distinct variants for owner
`0004`, and two unmatched rows. Across the capped 128 additional samples, no
alternative owner reached three distinct exact row variants. Pooling the
initial and additional waves also leaves owner `0006` with only two distinct
variants and owner `0004` with two.

The predeclared action was to stop rather than lower the admission threshold,
force a rare branch, or continue sampling after the 128-additional-sample cap.
Across all 160 independent samples from the exact boundary, no alternative
owner passed the three-distinct-variant gate. The equal-depth reciprocal test is
therefore unidentified, not negative.

## Supported

- A naturally appended complete row causes large, repeatable suppression of
  exact row variants owned by that same physical object.
- Simple category anti-repetition is not the sole cause: same-category person
  rows move in opposite directions after the same prefix update.
- Generic row depth or terminal preference is not the sole cause: the
  row-entry-versus-terminal margin stays strongly positive while owner-specific
  geometry likelihoods separate.
- The first horizontal coordinate, `x1`, is the dominant measured decision
  point in the strongest same-category transition.
- A local geometry-sorted successor or frontier is compatible with the
  observed pattern.

## Ruled Out Within This Evidence Scope

- A commit does not uniformly increase every uncovered candidate row.
- The observed image-`2299` transition is not explained only by suppressing the
  repeated word `person`.
- The effect is not merely a broad switch from terminal output to another row.

## Unresolved

- Whether the native transition represents a physical-object commit, a
  geometry-sorted traversal frontier, or a mixture of both.
- Whether a persistent covered-set state exists beyond the immediately
  appended row.
- Whether the same effect generalizes across checkpoints, prompts, random-order
  supervision, dense categories, or later row depths.
- Which language layer, visual pathway, or prefix pathway computes the
  owner-specific `x1` shift.
- Whether the local suppression can be strengthened by training without
  damaging naming, geometry, or enumeration.

## Not Claimed

- No explicit object ledger, slot, object file, covered-set carrier, or
  physical-object memory has been identified.
- No layer, attention, multilayer-perceptron, residual-stream, visual-token, or
  language-model-head causal source has been localized.
- No architecture or training loss is promoted.
- No forced prefix or off-support coordinate path is treated as native causal
  evidence.
- The exact-row likelihood changes are not a closed probability distribution
  over objects and do not by themselves prove probability-mass conservation.

## Decision and Next Research Seed

Close this unit at the natural sibling admission gate. Do not continue the
same image-`2299` sampling search, do not relax the three-variant requirement,
and do not begin a layer/source trace from this incomplete discriminator.

The next high-value unit should search existing bagging artifacts for a case
that already contains two or more naturally recurring, uniquely owned sibling
objects at the same exact prefix boundary. Prefer a same-category scene and
require the reciprocal owner branches before any new graphics-processing-unit
sampling. If such a case exists, test whether appending a later geometry-ranked
owner suppresses an earlier still-unemitted owner:

- reciprocal survival supports physical-object commit; and
- one-way suppression of earlier owners supports a geometry-sorted frontier.

Only after that discriminator passes should the research trace the dominant
`x1` change through final logits, language layers, prefix state, and visual
features. Training remains closed until the source of the useful transition is
identified or a separate treatment-screen unit is explicitly authorized.

## Verification

Every path cited above was resolved locally. The configuration-precision and
full-model-`float32` receipts agree on every conclusion-owning sign. All three
sampling admission manifests report the expected sample counts, unique seeds,
and complete merged receipts. The current runtime attestation independently
verifies the request-scoped sampling path used by this research family.

This result file is synthesized interpretation under `research/`; the receipts
and admission manifests under `outputs/research/` remain the executed evidence.
No current-behavior documentation, stable specification, production code,
model architecture, or training configuration is changed by this result.
