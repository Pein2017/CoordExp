---
title: Person 25 Dominant-Owner Commit and Persistence Closeout Results
type: investigation
role: research-results
authority: executed_evidence
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Person 25 Dominant-Owner Commit and Persistence Closeout Results

## Verdict

The historical random-order pure cross-entropy adapter has a strong,
row-content-sensitive **immediate transition**, but this image does not support
a stable object-level covered-set ledger.

After a clean row for person-only rank `25` is appended, sampled continuations
move away from a clean person-25 box and usually recover the distinct,
overlapping person-only rank `18`. The same transition occurs naturally when a
control rollout first emits person 25. However:

- greedy decoding still emits a vertically merged box that covers both people;
- the clean and merged continuations share their description, `x1`, `y1`, and
  `x2` neighborhood and separate mainly at `y2`;
- writing person 18 into an earlier row does not reliably exclude it after the
  same final person-25 row; person 18 still returns in `55/96` samples;
- that earlier-history intervention also creates full-canvas and malformed
  geometry, so its effect is not a clean object-ledger operation.

The most precise interpretation is therefore:

> The prefix carries a geometry-conditioned transition state with strong
> dependence on the most recent row in this case. It can move the next row
> toward an uncovered object, but the model does not atomically commit a
> complete physical instance or reliably preserve a set of all earlier
> committed instances.

One fixed-prefix logit probe also exposes a local reason sampling can succeed
where greedy fails. At the final `y2` decision, the single highest token is
`<|coord_999|>`, while the aggregate probability over the person-18 boundary
neighborhood is much larger. Greedy compares individual tokens and chooses the
isolated spike; sampling can enter the broad, physically coherent coordinate
cluster.

No architecture or training method is promoted from this one image and one
checkpoint. The result does keep a loss- or decoding-calibration route open
before introducing an explicit slot or covered-set carrier.

## Executed contract

All behavior panels used:

- image `2299`, with the current near-complete manual relabel containing 38
  people and 8 ties;
- the historical random-order adapter at checkpoint `3668`:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/
compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/
compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/
v1-20260601-062428/checkpoint-3668
```

- full-model 32-bit floating point and eager attention;
- the exact compact historical prompt and image bytes;
- temperature `0.4`, top-p threshold `0.95`, repetition penalty `1.0`;
- exact raw token append, without decode and retokenize;
- one structurally complete row per transition, except for the explicitly
  labeled three-row descriptive horizon.

The adapter, image, prompt, coordinate-token offset, and prefix hashes are
stored in every arm artifact.

## Immediate same-depth panel

The parent history was the canonical person sequence `[0, 1, 2]`. The control
appended canonical person `3`; each treatment appended one naturally sampled,
strictly matched person-25 row from the existing historical artifact.

| Final appended row | Donor Intersection over Union to person 25 | Strict person-25 matches | Other nearest owners | Valid rows | Mean continuous Intersection over Union to person 25 |
|---|---:|---:|---|---:|---:|
| canonical person 3 control | not applicable | `21/24` | nearest person 25 `23/24`; person 18 `1/24` | `24/24` | `0.8233` |
| person 25, donor seed 21 | `0.8152` | `7/24` | person 18 `16/24`; person 19 `1/24` | `24/24` | `0.3464` |
| person 25, donor seed 0 | `0.9288` | `5/24` | person 18 `18/24`; person 19 `1/24` | `24/24` | `0.3073` |
| person 25, donor seed 9 | `0.9745` | `5/24` | person 18 `18/24`; person 19 `1/24` | `24/24` | `0.3073` |

The frozen immediate gate passes. Relative to the paired control, the three
donors remove `14`, `16`, and `16` strict person-25 outcomes, with zero paired
switches in the opposite direction. Exact paired two-sided probabilities are
`0.000122`, `0.0000305`, and `0.0000305` respectively. All 72 sampled treatment
rows are syntactically valid.

The three donor variants are robustness treatments, not three independent
replicates. Donor seeds 0 and 9 produce identical owner outcomes for all 24
paired seeds; seed 21 differs on two seeds.

Greedy does **not** improve. Every donor arm greedily emits:

```text
<|object_ref_start|>person<|box_start|>
<|coord_0|><|coord_298|><|coord_93|><|coord_999|>
```

The strict matcher assigns this box to person 25, but coordinate review shows
that it spans both person 18 and person 25 and should not be interpreted as a
clean physical-instance repeat.

## Coordinate-level decomposition

The two manual boxes are vertically aligned and overlap:

| Person-only rank | Manual normalized box |
|---:|---|
| 18 | `[3, 298, 100, 552]` |
| 25 | `[1, 415, 98, 928]` |

Their box Intersection over Union is `0.2120`. The seed-0 donor for person 25
is `[0, 393, 97, 922]`.

After that donor, the 24 sampled rows split as follows:

- `18/24` clean person-18 rows: `x1=0`, median `y1=296`, median
  `x2=96`, and median `y2=559`;
- `5/24` merged, person-25-matched rows: `x1=0`, median `y1=297`, median
  `x2=93`, and `y2=999` in every case;
- `1/24` person-19 row in another region.

The same coordinate pattern holds for all three person-25 donors. Thus the
clean person-18 path and the merged path do not first separate at description
or `x1`. They share almost the entire row and primarily separate at the final
vertical boundary `y2`.

The larger 96-sample base-history arm contains 15 rows assigned to person 25.
All 15 begin in the person-18 corridor and end at `y2=999`. Their mean box
Intersection over Union is `0.845` to the geometric union of people 18 and 25,
compared with `0.697` to person 25 alone and `0.342` to person 18 alone. Every
one is closer to the two-person union than to either physical person. Across
that arm, `87/96` samples enter essentially the same corridor through
`x1,y1,x2`; 72 close near person 18's lower boundary and 15 expand into the
merged extent.

This is a concrete counterexample to treating `x1` as a universal completed
instance-binding point. For aligned, overlapping, same-description instances,
physical extent can remain unresolved until the final coordinate.

## Fixed-prefix `y2` probability competition

The fixed recipient was:

```text
[canonical people 0, 1, 2]
+ [raw seed-0 person-25 donor]
+ <|object_ref_start|>person<|box_start|>
  <|coord_0|><|coord_298|><|coord_93|>
```

One full-model 32-bit floating-point forward pass scored the next token. The
coordinate-token vocabulary carries `0.9999993` of the full-vocabulary
probability after temperature `0.4`, so the comparison is not caused by text
tokens competing at this slot.

| Candidate | Raw full-vocabulary probability | Probability after temperature `0.4` | Aggregate log-probability margin over `coord_999` after temperature `0.4` |
|---|---:|---:|---:|
| single `coord_999` | `0.00828` | `0.04288` | reference |
| observed person-18 band `514..576` | `0.25087` | `0.49543` | `+2.4470` |
| broader person-18 band `500..600` | `0.36007` | `0.62478` | `+2.6790` |

`coord_999` is the highest individual coordinate logit. The next highest
individual coordinates are `564`, `563`, `558`, `567`, `562`, and nearby
values. In other words, a single erroneous extreme token beats every one
correct-neighborhood token, while the correct-neighborhood aggregate mass
beats the extreme by more than an order of magnitude.

This is an exact local distributional explanation for the branch at this fixed
`y2` state. It does not by itself explain every sampled trajectory or prove
that all rescued objects or all four coordinate slots have the same structure.

## Same final row with different earlier histories

The raw seed-0 person-25 donor is token-identical after earlier treatment
owners `2`, `3`, `4`, and `14`. With 24 paired samples, their next-owner
distributions are nearly identical:

| Earlier treatment person | Person 18 | Person 25 | Other people |
|---:|---:|---:|---:|
| 2 | `18/24` | `5/24` | person 19 `1/24` |
| 3 | `17/24` | `4/24` | person 4 `2/24`; person 12 `1/24` |
| 4 | `17/24` | `5/24` | person 19 `1/24`; person 32 `1/24` |
| 14 | `18/24` | `5/24` | person 19 `1/24` |

This shows strong last-row dominance, but those four earlier people were weak
successor candidates and therefore could not test object-specific persistence.

The follow-up held the final donor exactly fixed and increased to 96 paired
seeds. It compared a base history, a nearby spatial control, a distant control,
and the key treatment in which dominant successor person 18 had already been
written earlier.

| Earlier treatment person | Person 18 next | Person 25 next | Unresolved geometry | Malformed | Greedy result |
|---:|---:|---:|---:|---:|---|
| 2, base | `72/96` | `15/96` | `0` | `0` | person-25-associated merged box |
| 11, nearby control | `70/96` | `14/96` | `0` | `0` | person-25-associated merged box |
| 17, distant control | `64/96` | `10/96` | `2` | `0` | person-25-associated merged box |
| 18, already committed | `55/96` | `6/96` | `12` | `1` | full-canvas person box `[0,0,999,999]` |

For person-18 recurrence, the paired exact comparison of the person-18 history
against the base history gives `p=0.00599`; against the nearby person-11
control, `p=0.0135`; and against the distant person-17 control, `p=0.136`.

The earlier person-18 row therefore remains causally relevant after the same
final row, but the effect is neither complete nor clean. Person 18 is still the
next strict match in `55/96` samples, and 12 additional samples collapse to the
full-canvas box `[0,0,999,999]`. A distant earlier row also changes the
distribution. This supports long-range prefix influence, not a reliable
physical-object exclusion ledger.

## Three-row descriptive horizon

The horizon is descriptive because later rows diverge.

In the same-depth control, person 25 is a strict first-row match in `21/24`
trajectories. Given those 21 naturally emitted person-25 rows, the second row
moves to another uncovered person in `20/21` cases, including person 18 in
`19/21`. Only `1/21` immediately repeats person 25. This independently
reproduces the donor-based immediate transition without forced person-25 row
tokens.

After the fixed person-25 donor, the first strict return time of person 25 is:

| First strict return lag | Trajectories |
|---:|---:|
| next row | `5/24` |
| two rows later | `2/24` |
| three rows later | `2/24` |
| no strict return within three rows | `15/24` |

The traversal is not unique-object coverage. Among the 18 trajectories whose
first row is nearest to person 18, the second row repeats person 18 in `11/18`.
The model can therefore express strong immediate suppression for person 25
while failing to apply the same operation reliably to the next person.

## Hypothesis decisions

| Explanation | Decision | Evidence |
|---|---|---|
| Person 25 is a completely static candidate unaffected by its own row | Rejected for sampled clean extent | Three donor variants and natural within-trajectory reproduction move most mass to person 18. |
| Appending any fourth row is sufficient | Rejected | The same-depth person-3 control remains person-25 dominated. |
| Only one literal person-25 token string is suppressed | Disfavored | Three geometrically distinct raw person-25 rows produce the same transition signature. |
| The adapter has a stable physical-object covered-set ledger | Not supported | Earlier person 18 still repeats in `55/96`; full-canvas failures expand; person 18 also repeats in `11/18` next steps. |
| Earlier history is completely forgotten after the latest row | Rejected | The token-identical final row still changes under earlier person 18, including greedy collapse. |
| The immediate effect is an atomic complete-object switch | Rejected in this case | Clean and merged paths share description and first three coordinates and separate mainly at `y2`. |
| Greedy failure is partly caused by fragmented valid coordinate mass | Supported at the fixed `y2` state | `coord_999` is the single-token argmax, while `514..576` owns `49.54%` mass at temperature `0.4`. |

## Research implication

This result changes the treatment priority.

An explicit slot or persistent covered-set carrier is not yet the first move.
The native decoder already uses the emitted row as a causal transition state,
and it already places substantial probability on the correct next extent. The
immediate failure is that this support is distributed over many nearby
coordinate tokens while a contaminated extreme boundary wins greedy argmax.

The smallest treatment-linked successor should therefore test whether a
coordinate-neighborhood or complete-row objective can consolidate a sampled
clean extent into the greedy path under the model's own prefix. It must compare:

- a clean uncovered-object row;
- the merged or repeated row chosen by greedy;
- other already committed objects;
- invalid and full-canvas geometry.

A plain tail-append sequence is unlikely to identify this distinction. A
training screen should operate at the actual self-prefix decision and preserve
the two-axis distinction between entity discovery and physical extent.

Before a 256-image training screen, replicate this exact analysis on 4 to 8
crop-reviewed dense cases: identify where sampled clean and greedy contaminated
rows first diverge, measure aggregate valid-coordinate mass against the greedy
single-token spike, and require the entity and geometry interpretation to agree.
Only if the native valid mass is repeatedly absent should the route escalate to
a new visual bridge or explicit state carrier.

## Limitations

- one image, one checkpoint, one training seed, and one dense same-class region;
- person 18 and person 25 overlap and are unusually aligned, making the case
  useful for binding analysis but not representative prevalence;
- the `514..576` interval is derived from the manual person-18 extent and the
  observed clean-output range, not a preregistered universal tolerance;
- donor rows are natural continuations after a forced canonical prefix, not
  complete natural rollouts;
- the three donor variants are correlated robustness checks;
- a three-row free horizon cannot identify a persistent hidden state because
  intermediate generated rows diverge.

## Artifact roots

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/
  immediate-fp32-greedy-plus-k24-v1/
  common-final-person25-history-fp32-k24-v1/
  dominant-successor-history-fp32-k96-v1/
  horizon3-fp32-k24-v1/
  y2-fixed-prefix-fp32-v1/
```

The executed one-time runner and fixed-prefix scorer are:

```text
scripts/research/run_person25_commit_closeout.py
scripts/research/score_person25_y2_competition.py
```

Targeted tests:

```text
tests/analysis/test_person25_commit_closeout.py
tests/analysis/test_person25_y2_competition.py
```

**Replay note.** Producer scripts deleted from `research-probes` on 2026-08-28 (reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`: `git worktree add <tmp> research-base-v2`.
