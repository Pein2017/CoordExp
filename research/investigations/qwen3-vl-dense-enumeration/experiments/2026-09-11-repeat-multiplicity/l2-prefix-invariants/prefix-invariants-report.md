# Repeat-multiplicity prefix candidate report

**Status:** candidate, CPU-verified; no model call, GPU launch, GT use, or
owner admission. This is a synthetic conditional diagnosis only.

## Frozen source and cohort

The only source is the immutable natural-row census from R:

- Census: [/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/census/census.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/census/census.json)
- Census SHA-256: `664856efcf2555097a419011f6f9485c87ebe3978a1064e2d78d0151444a447a`
- R packet SHA-256: `48087dcf05e03fc9b6c2eb9de1e2d66e1efe09c972ce22e8c6ae2774f95aa4ba`

The fixed cohort is exactly `9813, 158044, 417044, 502725`; there is no
backfill. A/B/C are **literal source-row labels**, not physical-owner or GT
identities. Every selected row is from the original-image natural action and
has no strict-repeat reference in R.

| case | natural strict repeats; first strict row | description | A source row | B source row | C shared-final source row | row tokens | prefix tokens |
|---:|---:|---|---:|---:|---:|---:|---:|
| 9813 | 0; none | person | 0 (`0:9`) | 6 (`54:63`) | 10 (`90:99`) | 9 | 45 |
| 158044 | 5; row 9 | book | 13 (`119:128`) | 16 (`146:155`) | 20 (`182:191`) | 9 | 45 |
| 417044 | 291; row 8 | donut | 1 (`9:19`) | 2 (`19:29`) | 4 (`39:49`) | 10 | 50 |
| 502725 | 38; row 5 | knife | 0 (`0:9`) | 2 (`18:27`) | 3 (`27:36`) | 9 | 45 |

The spans are zero-based half-open offsets in each unchanged Stable50 action
token array. Full source-row digests and coordinates are frozen in
[`check_prefix_invariants.py`](check_prefix_invariants.py); the selected rows
are canonical, accepted, geometry-valid, four-coordinate rows, and pairwise
strict-IoU distinct. Their pairwise native pixel IoUs are:

| case | A/B | A/C | B/C |
|---:|---:|---:|---:|
| 9813 | 0 | 0 | 0 |
| 158044 | 0 | 0 | 0 |
| 417044 | 0 | 0 | 0.051734 |
| 502725 | 0 | 0.157998 | 0.007097 |

These geometry values are only a row-distinctness check. They do not certify
three physical instances. The prior visual review explicitly leaves book
grouping, clipped donuts, and overlapping knife/utensil identity uncertain.

## Conditional prefix family (eight arms per case)

Materialize each prefix as exact concatenation of the stored source token IDs;
do not decode/re-tokenize, pad, translate, repair, or append EOS. `C` is a
one-row common suffix, so every arm has five complete rows, the same unique
source-row label set `{A,B,C}`, the same token count, and byte-identical final
row `C`.

| arm | varied block | A multiplicity | B multiplicity | reversal mate |
|---|---|---:|---:|---|
| `alt_fwd` | `ABAB` | 2 | 2 | `alt_rev` |
| `alt_rev` | `BABA` | 2 | 2 | `alt_fwd` |
| `Aheavy_fwd` | `AAAB` | 3 | 1 | `Aheavy_rev` |
| `Aheavy_rev` | `BAAA` | 3 | 1 | `Aheavy_fwd` |
| `Bheavy_fwd` | `BBBA` | 1 | 3 | `Bheavy_rev` |
| `Bheavy_rev` | `ABBB` | 1 | 3 | `Bheavy_fwd` |
| `block_fwd` | `AABB` | 2 | 2 | `block_rev` |
| `block_rev` | `BBAA` | 2 | 2 | `block_fwd` |

The actual prefixes are the table block followed by `C` (for example,
`AAAB|C`). `alt` versus `block` is an adjacency control at balanced
multiplicity; A-heavy and B-heavy are symmetric count reallocations. Reversal
mates counterbalance direction/recency but do not make the contrast a pure
count intervention.

## Literal acceptance checks

`check_prefix_invariants.py` performs read-only checks and exits nonzero with
`HOLD` on failure. It verifies:

1. R census byte hash and all selected raw-row hashes;
2. accepted, geometry-valid, complete canonical rows with one object wrapper,
   one box wrapper, four coordinate tokens, no EOS/PAD, and consistent span;
3. same description, pairwise-distinct A/B/C token rows, equal row-token
   lengths, and no pairwise strict geometry duplicate;
4. exactly eight arms, five rows and the same `{A,B,C}` set per arm;
5. exact per-case prefix token counts (`45,45,50,45` in cohort order), no EOS,
   and exact byte-level final C row;
6. exact reversal pairing for all four forward/reverse pairs.

Run:

```bash
python /data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-11-repeat-multiplicity/l2-prefix-invariants/check_prefix_invariants.py
```

Expected result is `PASS_CPU_LITERAL_INVARIANTS` for all four fixed cases and
eight arms each. No candidate is substituted if the command fails.

## Exact falsifier and interpretation boundary

After all literal checks pass, this lane has **no detected multiplicity effect
at this scope** if all eight arms in every case produce the same first complete
free row or EOS and the same declared free-summary outcomes (valid rows,
strict repeats, invalid rows, stop reason), while the reversal-balanced score
contrasts are zero under the packet's fixed FP32 comparison tolerance. A
reversal-only difference that cancels between mates falsifies a pure-count
interpretation, even if it establishes order/recency sensitivity.

For any scalar readout `Y`, the registered count contrasts are
`Delta_A = mean(Y[Aheavy_fwd], Y[Aheavy_rev]) -
mean(Y[alt_fwd], Y[alt_rev])` and
`Delta_B = mean(Y[Bheavy_fwd], Y[Bheavy_rev]) -
mean(Y[alt_fwd], Y[alt_rev])`. The exact null is `Delta_A = Delta_B = 0`
for every case/readout **and** no arm-wise free-output difference; a zero
average alone is not enough because opposite changes can cancel.

Conversely, a persistent A-heavy/B-heavy difference after averaging each
forward/reverse pair is only evidence of **conditional history sensitivity** on
these four synthetic panels. Exact candidate-row log probability is not total
probability mass over an IoU neighborhood; free next-row identity, repeat,
invalid-geometry, and stopping must be reported separately.

## HOLD conditions and limits

Hold the whole lane (no backfill) for any source-hash drift, missing row,
description mismatch, unequal row lengths, malformed/invalid candidate, EOS or
PAD in a prefix, changed unique-row set, changed final C token IDs, or failed
reversal/token-count check. Hold physical-owner language even when the CPU
checks pass: R provides generated geometry, not exhaustive instance identity;
the existing review records uncertainty for these small/grouped objects.

The prefixes are off-policy synthetic histories with no natural preamble. Four
hand-selected cases, one checkpoint, deterministic greedy decoding, and a
512-token conditional horizon cannot estimate prevalence, training origin,
generalization, or deployment quality. Equal token count fixes absolute length
and final-row boundary, but A/B content, adjacency, and recency necessarily
change; reversal averaging reduces rather than removes that confounding. The
one-row C suffix says nothing about longer suffix memory. No result here
authorizes training, stronger punishment, a circuit claim, or a physical-owner
quality score.
