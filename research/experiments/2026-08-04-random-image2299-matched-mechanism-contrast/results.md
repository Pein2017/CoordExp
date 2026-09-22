---
title: Random Image 2299 Matched Mechanism Contrast - Results
type: investigation
role: research-results
authority: non_normative_research
unit_id: 2026-08-04-random-image2299-matched-mechanism-contrast
status: complete
evidence_status: complete
updated: 2026-08-04
---

# Results

## Native rollout

Under the matched HF fp32, greedy, `rp=1.0`, `max_new_tokens=3084`
runtime, the original random-order step-4887 checkpoint stopped naturally after
14 valid rows. Strict ambiguity-neutral one-to-one matching found 3/46 owners,
leaving 43 false negatives and 11 unmatched rows. Ten of the unmatched rows
were the exact repeated full-canvas box `<0,0,999,999>` for `person`.

The completed sorted reference produced 19/46 strict owners from 24 valid rows.
Random retained exactly 3 sorted owners, gained none, and lost 16. This is not a
mere shorter-list effect: the random trajectory entered a repeated full-canvas
geometry basin after two initially correct left-side people, briefly recovered
one correct lower-left owner, and then returned to the same basin.

## Conditional landscape

The full random capture completed all `30/30` query groups with no plan
modification. At the identical empty assistant history, random is not weaker
than sorted in conditional GT-localized likelihood. Across the 46 physical
owners, the median random-minus-sorted deltas are:

| Readout | Median delta |
| --- | ---: |
| Exact GT-anchor four-coordinate log probability | `+4.4715` |
| Best tested local-candidate log probability | `+4.4413` |
| Local concentration | `+1.1712` |
| Category-population peak lift | `+2.4776` |
| Owner-relative margin to the best same-category owner | `+2.3381` |
| Continue-versus-stop margin | `+0.6760` |

The exact GT-anchor score improved for `38/46` owners: `32/38 person` and
`6/8 tie`. Among the 16 owners emitted by sorted but lost by random, `12/16`
still have a higher exact-anchor score under random. The matched root evidence
therefore rejects the simple explanation that random native recall is low
because those owners uniformly lost conditional visual/geometry support.

The stronger support does not become a coherent greedy coordinate policy. In
the random checkpoint's 15 self-prefix contexts, forced-category free-box
decoding strictly hits only four owners in total: the same three native person
owners plus one tie. The sorted reference hits 20 owners over its own 25
contexts. This comparison includes endogenous history after root and is not a
pure checkpoint intervention, but it directly measures each checkpoint's
realized trajectory.

### Full-canvas basin

The random `person` free-box sidecar is local at boundaries 0 and 1, becomes
the exact full-canvas `[0,0,999,999]` at boundaries 2 through 4, briefly returns
to local boxes at boundaries 5 and 6, and is full-canvas again at every
boundary from 7 through 14.

At boundaries 3 and 4, the full-canvas greedy sequence is respectively `0.51`
and `1.07` nats *worse* than the best tested GT-local candidate sequence. The
token-wise greedy decoder nevertheless enters it. This is evidence for local
argmax/search-path failure before the basin becomes dominant at sequence
level. From boundary 7 onward the full-canvas sequence is `2.75` to `6.05`
nats above the best tested GT-local candidate, so the later prefix has turned
the failure into a genuine high-probability geometry attractor.

The forced `tie` sidecar similarly returns essentially one lower-image box in
all 15 contexts; it strictly matches only one tie once. Thus random ordering
did not produce a broad instance selector on this dense one-category scene. It
produced broad teacher-forced local support but a narrow, history-sensitive
coordinate argmax policy.

## Decode-only `rp=1.10` discriminator

Changing only repetition penalty from `1.0` to `1.10` eliminates the ten exact
full-canvas duplicates. The random checkpoint then emits 19 row spans: 18
valid and one invalid tie box. Strict matching recovers 11/46 owners, versus
3/46 at `rp=1.0`; all three original owners are retained and eight are added.
Four of the 11 are owners not hit by the sorted `rp=1.0` trajectory.

This is a large decode-processor rescue, but not a solution. One near-canvas
person box remains, seven valid rows are unmatched, the output introduces a
`chair`, and one tie has invalid geometry. It remains well below sorted's
19/46. Repetition penalty therefore breaks the repeated-coordinate basin and
reveals latent alternative owners, while also trading into geometry and
unsupported-output errors.

## Primary conclusion

For image `2299`, random-order training appears to have learned a wider
conditional owner/box support surface than its native recall suggests. Its
main failure is converting that surface into one stable without-replacement
greedy trajectory. The failure has two stages:

1. an early token-wise greedy path can choose a worse full box sequence than a
   tested GT-local alternative; and
2. after more self-prefix rows, the full-canvas sequence becomes a dominant
   high-probability attractor.

This is meaningfully different from both `visual support absent` and simple
premature STOP. It points to sequence-level coordinate search, repeated-token
dynamics, and history-conditioned instance routing as the immediate random
checkpoint bottlenecks on this image.

## Interpretation boundary

The sorted frozen support classifier is not transferred. Random has only three
native-TP controls, only two eligible due boundaries, and clears the old rule
for one; the transfer is underpowered and non-validity-bearing. The descriptive
random FN split (`12 resolved`, `30 persistent`, `1 bank-insufficient`) is not
a recoverability prevalence estimate.

Teacher-forced coordinate likelihood after a forced category query is not the
marginal probability of proposing an owner, and the 17-role bank is finite.
The `rp=1.10` run is a post-hoc decode discriminator, not part of the clean
random-versus-sorted checkpoint contrast.

## Evidence

- Random run root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-random-image2299-matched-mechanism-contrast/20260804T134835Z`
- Native `rp=1.0` receipt: `s0-native/receipt.json`
- Complete candidate shard: `s1-shard-full-b16/shard-receipt.json`
- Continuous owner analysis: `s1-analysis/owner-context-features.jsonl`
- Matched contrast: `s2-contrast/comparison.json`
- Owner-by-owner contrast: `s2-contrast/owner-comparison.jsonl`
- Post-hoc `rp=1.10` rollout: `s3-rp1p10/greedy.json`
