# Accepted visual audit: credit changes have several meanings

The lead accepts the complete CPU evidence package and closes this unit after
reviewing all23 frozen A_A→D_I category-owner transitions on ten source images.
The user requested completion of round11 followed by a stop for reporting and
discussion. No round12 or other research is scheduled. User scientific acceptance
remains false; all visual judgments are candidates, not ground truth.

The [gallery](../../../outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/index.html)
retains the paired contexts and full-image evidence. The separate
[lead adjudication](../../../outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/lead-adjudication-01.json)
records entity representation, category, localization, evaluator reason, selected
alternative rows and source locators for every case. Historical case JSONs retain
their original pending fields; this later adjudication owns the completed review.

| Descriptive disposition | Cases | Meaning and limit |
|---|---:|---|
| Candidate dedicated localized additions |3|One person and two donuts acquire an individual local box. Full-output alternatives were inspected; this is not validated physical recovery.|
| Candidate dedicated localized removals |3|One person, one donut and one chair lose an individual local box. This is not a population physical-FN estimate.|
| Plausibly persistent entity with changed geometry |8|A candidate for the same visible entity plausibly remains; changed localization changes reference credit.|
| Unchanged candidate with assignment exchange |1|The exact candidate remains, but its evaluator credit moves to another reference.|
| Unresolved individual entity correspondence |8|Occlusion, tiny targets, crowded/merged extents or poorly located alternatives prevent a confident physical interpretation.|

These are qualitative review counts, not a new binary representation metric or
a net physical-success score. The original panel contains10 gains and13 losses.
It excludes unlabeled entities, both-missed entities, representation changes among
retained-credit owners and the two repaired-context images7511/351017. Therefore
neither 3−3 nor 10−13 estimates overall physical recovery. Unknown/unmatched
predictions remain annotation-relative unknowns.

The six localized-change candidates are gains `2299/-17`, `417044/1079910`,
`417044/1083295` and losses `2299/-27`, `417044/1083375`, `477415/1586761`.
A separate Astra review tried to falsify these by finding overlooked local or
partial same-entity candidates in the complete saved pools and raw records; none
was found. Crucially, the donut scene contains a huge strip
`[703,0,999,999]` in both arms (A_A p32, D_I p33). It overlaps several targets,
so “no prediction overlaps the entity” would be false. The narrower supported
description is addition/removal of a dedicated local candidate. Visual category
compatibility does not promote any prediction or existing annotation to truth.

Two concrete counterexamples constrain interpretation. For `16228/-53`, p4 is
exactly `[44,460,77,562]` with the same person description and IoU0.520548 to the
target in both arms. Its assignment changes from−53 to−61; the box did not
disappear. For `5001/2017429`, the broad credited p7 crosses just below0.5,
while the clear head/upper-body candidate p9 remains in both outputs. For the
tiny `16228/-39`, a four-bin right-edge change moves IoU0.51138→0.49471;
individual identity remains unresolved despite the clear numerical reason.

The observation is that evaluator gains/losses mix geometry, assignment,
ambiguous identity and plausible changes in localized depiction. It does not
identify a training mechanism. The strongest prior alternative—every credit
loss is a physical omission—is contradicted; the opposite claim that every
change is harmless threshold noise is also unsupported by the six candidates.
No annotation rewrite, training/admission/acquisition feedback or new experiment
follows from these model judgments.

Technical acceptance reuses the worker's successful focused adapter falsifier,
real renderer boundary and full-pool/coordinate readback. An independent mapping
review verified all23 joins, reference/crop geometry and1650 case-side candidate
occurrences, including exact pixels for a nonsquare source crop. The lead checked
the terminal and five bound records, all23 case and ten full-pool identities,
source diff, exact disposition keys, cited rows and the unchanged-box assignment
counterexample. All ten originals and23 paired contexts were visually inspected;
unmarked crops and complete alternative pools were used where needed. No model,
GPU, tokenizer object, checkpoint, fresh inference or test-suite rerun was used.
The CPU package took13.339s active time, peaked at1.652GB RSS and retained about
41.7MB before terminal evidence, within30min/8GiB/1GiB.

Immutable output root is canonical
`outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11/`.
`lead-adjudication-01.json` SHA256 is
`ee06ffb7397674912a7d5d160e8694788e5204d2acc55363194bd41047623c8a`;
`lead-acceptance-01.json` SHA256 is
`b3dbb82315916b25276cb6f9659b45f4d1f3969165f9930496e391746d817b5c`.
The clean CPU candidate source was `d97cf13bf31e704bd81b33a4518aad8a8835376e`.
Source-image and checkpoint hashes were reused, not regenerated. Historical
evidence is unchanged. Round11 is complete; stop for the user discussion.
