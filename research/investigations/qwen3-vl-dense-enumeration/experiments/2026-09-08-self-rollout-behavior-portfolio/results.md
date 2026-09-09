# First-wave synthesis — lead accepted, bounded stop reached

Historical closeout. The later [owner visual follow-up](../2026-09-08-a-owner-visual-followup/visual-readout.md)
supersedes interpreting the strict 4/16 review conjunction as an owner-validity
count or upper bound. [Updated route/cost evidence](../../../../archive/self-rollout-a-owner-visual-followup-benchmark/readout.md)
uses the user's supplied prices and does not retire Sol globally. The original
official metrics below remain unchanged.

All four inference lanes are complete. Root replayed the decision-bearing
consumers, exact identity/matching checks and routing qualifications. No model
training, architecture promotion, new sample extension or configuration change
followed. Raw evidence root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior`.

## Owner binding, not description correctness

The user's later correction owns P's interpretation: a category error is
acceptable if the box binds a real single owner. `desc` can be masked; its
correctness is not an owner veto. Reusing the sealed24-box blind review,
**4/16 unmatched boxes** now pass visible entity + single-instance + acceptable
localization, versus1/16 under the superseded description-required rule.
Matched controls pass7/8; the remaining control has a localization concern.
These are model-reviewed candidates, not confirmed supplemental unique owners.

On the original64 scored Source trajectories, the separate category-agnostic
pixel-IoU>=0.50 global one-to-one diagnostic matches295/498 owners, versus287/498
under the frozen official category-consistent metric. `coord_mean` AUC is
0.85166 for auxiliary matched/unmatched discrimination (official AUC0.83976).
The score is useful for triage; five-class semantic ordering, true-unlabeled
labels and a calibrated owner-acceptance threshold remain unestablished.
The high-confidence projector/television category mismatch is not evidence
against the clarified owner-binding criterion.

See `reviewer/result.md`, `reviewer/integrate.py`, and
`reviewer/owner_binding.py`. Historical raw likelihood evidence was unavailable
after bounded recovery, so P freshly scored existing exact trajectories rather
than generating replacements. The reviewer used native vision tools, not an
external service or a new infrastructure layer.

## A, B and F conclusions

These retain their frozen category-consistent annotated-owner metric; the new
P diagnostic does not silently redefine their scientific contrast.

| Lane | Observation | Decision |
|---|---|---|
| A: same-prefix downstream value |16 of96 within-image branch pairs have equal immediate gain but different final gain, across6/16 images;2 ranking reversals | Immediate owner gain is insufficient to characterize later coverage on sampled support. Worth pursuing as a credit-design question, not yet an accepted training objective. |
| B: last-two-row history swap | Eligible12:83/110 owners before and after;3 gained,3 lost; remaining realization44/71 unchanged | Local history sensitivity, no aggregate recovery or evidence that a new memory module is needed. Close this stress test. |
| F: feedback off | Same dev64:231→278/498 owners,63 gained/16 lost; paired-bootstrap95% interval for +47 is[24,76]; caps2→0 | Inference contribution is harmful conditional on the feedback-trained DoRA. Do not promote this feedback variant. |

A has a concrete caution independent of description errors: on486123 all four
sampled rows have the same traffic-light description and length, differing only
in coordinates. Immediate increments[0,1,1,0] lead to final increments[1,2,2,3].
The highest-return row is an annotation-unmatched1x7-pixel box. Downstream
coverage can reward an extra routing cue, so masking `desc` alone does not
address action grounding. This is not proof of hallucination or a reason to
silently add a geometry/unknown penalty.

F-off remains below Source287 and near training control280. The ablation does
not isolate joint-training co-adaptation or establish a new superior model.
A/B use a small historically selected training-side16 panel; F/P use exploratory
development64. Neither is held-out confirmation. All branch denominators,
owner gains/losses, drops and stops remain in the raw reductions.

Lane records: `a/results.md`, `a/training-proposal.md`, `b/results.md`,
`b/recovery-training-proposal.md`, and `model-routing/feedback-off-results.md`.
Training proposals are candidates only. In particular, coordinate-only masking
would need its own explicit surrogate/sampling and owner-acceptance contract;
it is not identical to the drafted full-action row RLOO.

## Sol versus Astra

Both implementations passed the same real qualification on their first attempt
and required zero lead repairs. Astra-low took6m06s and8875 output tokens;
Sol-high took14m41s and24072 output tokens. Uncached input was63153 versus96129.
Root independently verified exact fresh-task boundaries and raw telemetry.
Astra-low was2.41x faster in this work sample and used fewer tokens. Dollar
comparison is unavailable because the local rates omit Astra.

Recommendation: use **Astra-low instead of Sol-high for this bounded engineering
slot**, with Sol retained as a manual fallback. One sample does not establish
universal family superiority. No model configuration was changed. See
`model-routing/result.md` for acceptance, comparison limits and receipts.

## Acceptance and execution bounds

`lead-acceptance.json` binds exact decision outputs and root verification.
A:160 parser/matcher artifact replays,16 midpoint replays,3 reducer tests and an
independent raw paired-count recount. B: exact swaps/pins/fresh reduction and10
tests. P: exact-score corruption sensitivity, original identities and sealed
blind-review reaggregation. F: cold hook parity, persisted full64, original
weight identities, global matches and paired-bootstrap replay.

GPU execution intervals including model setup: A428.25s, B70.51s, P55.65s,
F433.71s; the two engineering qualifications took15.08s and16.17s. These are
overlapping single-device intervals, not portfolio wall time. All six recorded
GPU worker PIDs are gone. A's first attempt failed pre-GPU on the root-produced
absolute-path audit serialization; canonical-source recovery preserved the
frozen IDs and hashes. That input issue is not a Sol/Astra work-sample defect.

Stop: no further GPU invocation, optimizer update, automatic reviewer expansion,
proxy promotion, model-config removal or experiment extension.
