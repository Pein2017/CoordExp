---
title: Image 2299 Near-Complete Human Relabel Successor Transition Results
description: A 96-call paired panel supports immediate physical-owner-sensitive successor routing while leaving multi-row covered-object state versus last-row spatial routing unresolved.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-18-image2299-near-complete-human-relabel-successor-transition
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: immediate_owner_sensitive_transition_supported
updated: 2026-07-18
---

# Image 2299 Near-Complete Human Relabel Successor Transition Results

## Verdict

At this exact dense-person state, appending a naturally generated row for one
physical person changes the next-person distribution in an owner-specific way
and excludes the just-emitted person from all 96 immediate successor samples.
This is strong output-level evidence for a commit-like transition. It is not
yet evidence that the model maintains a general multi-row covered-object set.

The main unresolved alternative is a last-row-conditioned spatial successor
rule: the final box may route the decoder to a learned nearby person without
retaining the complete earlier history.

## Evidence Scope

- Image: `2299`, a dense school group photograph at `1216 x 736` pixels.
- Relabel authority:
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`,
  line 28.
- Relabel record SHA-256 checksum:
  `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b`.
- Source image SHA-256 checksum:
  `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`.
- Relabel contents: 38 `person` objects and 8 `tie` objects.
- Model: the 2-billion-parameter Qwen3 Vision-Language model with the
  geometry-sorted Gaussian coordinate target Weight-Decomposed Low-Rank
  Adaptation (`DoRA`) adapter at step `4,887`.
- Shared parent: the same natural prefix containing two previously emitted
  people, person ranks 1 and 0.
- Treatment: append one complete naturally sampled row for person rank 3, 2,
  4, or 14, then sample exactly one successor row.
- Robustness: three exact natural coordinate variants per emitted person and
  the same eight seeds for all variants.
- Total: `4 owners x 3 variants x 8 paired seeds = 96` calls.
- Decode policy: temperature `0.4`, top-p `0.95`, repetition penalty `1.0`,
  and a nine-token maximum horizon.
- Person ranks are zero-based ranks among the 38 relabeled people in the
  geometry-sorted serialization.

The new relabel resolves 30 of 32 rows from the earlier discovery panel, versus
23 of 32 under the older audit ledger. It therefore removes missing annotation
as the main interpretation risk for this image.

## Observed Successor Distributions

| Emitted person | Valid matched successors | Other outcomes | Earlier-rank / later-rank successors | Immediate self repeats | Earlier parent repeats |
|---|---|---|---:|---:|---:|
| rank 3 | rank 2: 18; rank 4: 4; rank 6: 1 | unresolved: 1 | 18 / 5 | 0 / 24 | 0 / 24 |
| rank 2 | rank 5: 6; rank 6: 18 | none | 0 / 24 | 0 / 24 | 0 / 24 |
| rank 4 | rank 3: 19; rank 2: 3; rank 14: 2 | none | 22 / 2 | 0 / 24 | 0 / 24 |
| rank 14 | rank 10: 13; rank 28: 4; rank 16: 3; rank 18: 2 | malformed: 2 | 13 / 9 | 0 / 24 | 0 / 24 |

Overall:

- 93 of 96 successors match one relabeled person uniquely under Intersection
  over Union at least `0.5` and a top-versus-second margin at least `0.05`;
- one valid `person` row remains unresolved;
- two rows are malformed because they contain only three valid coordinate
  tokens;
- no successor is a `tie` and no call stops before attempting a row;
- no successor repeats the just-emitted physical person;
- no successor repeats either of the two earlier parent people; and
- among the 93 uniquely matched people, 53 move backward and 40 move forward
  in the trained geometry order.

## Owner-Specific Evidence

Every treatment owner is reachable when a different person was appended:

| Candidate person | Successor after its own row | Successor after the other three owners |
|---|---:|---:|
| rank 2 | 0 / 24 | 21 / 72 |
| rank 3 | 0 / 24 | 19 / 72 |
| rank 4 | 0 / 24 | 4 / 72 |
| rank 14 | 0 / 24 | 2 / 72 |

This matters because zero self-repeat cannot be explained by declaring those
people globally unreachable. The contrast is strongest for ranks 2 and 3.
Ranks 4 and 14 have lower counterfactual frequency, so their individual
self-suppression evidence is weaker even though they follow the same pooled
pattern.

Under the canonical coordinate variant, seven of eight paired seeds produce
four distinct successors across the four emitted owners. One seed produces
three distinct successors. This rejects a source-insensitive generic row
progression explanation for the selected state.

Exact-coordinate variant agreement within each owner is:

| Emitted person | Seeds for which all three natural variants agree |
|---|---:|
| rank 3 | 4 / 8 |
| rank 2 | 8 / 8 |
| rank 4 | 7 / 8 |
| rank 14 | 5 / 8 |

Small coordinate changes can therefore change the sampled successor, but they
do not remove the owner-conditioned exclusion pattern.

## What This Rules Out in This Case

1. **Whole-category anti-repetition.** Ninety-four calls produce a valid
   `person` row and no call switches to `tie`.
2. **Generic continuation.** The emitted owner sharply changes the successor
   distribution under paired seeds.
3. **A strict one-way geometry frontier.** More than half of the uniquely
   matched successors are earlier in geometry order. Rank 14 frequently returns
   to rank 10.
4. **Missing annotation as the main apparent effect.** Ninety-three outcomes
   receive a unique physical owner under the user relabel.
5. **A single accidental coordinate rendering.** The main exclusion result
   survives three natural exact-row variants for every owner.

## What Remains Unresolved

The panel identifies an immediate physical-owner-sensitive transition, but two
mechanisms remain compatible with it:

1. the prefix retains a multi-row record of already emitted physical people;
2. the final row selects a learned spatial successor while older rows have
   little or no causal effect.

The absence of the two earlier parent people from all 96 outcomes is consistent
with multi-row memory, but is not decisive because those people may be unlikely
successors from every selected final-row state.

The current panel also does not establish:

- an explicit hidden-state ledger;
- generalization beyond image 2299, this checkpoint, or this parent prefix;
- a training objective or architectural remedy; or
- that every person has equally strong selective self-suppression.

## Highest-Information Next Experiment

Hold the final emitted physical person fixed while changing only an earlier
covered person. Use equal-depth, naturally occurring histories:

```text
History A: ... person j ... final person i
History B: ... person k ... final person i
```

Then sample one paired successor row.

- If person `j` is suppressed only after History A, an earlier committed row
  remains causally active after the same final owner.
- If the successor distributions are effectively the same, the current result
  is better explained by last-row spatial routing.

Existing bagging trajectories should be searched first for naturally
convergent histories. A forced row should not own the conclusion.

## Artifacts

- Frozen per-call analysis and aggregate transition table:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-image2299-near-complete-human-relabel-successor-transition/analysis-v1/analysis.json`
  with SHA-256 checksum
  `ccb2c8baac83ba96da4282a01f8989923310034a34efccb41d8a32f0a170a5d3`.
- Canonical four-owner panel:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-image2299-near-complete-human-relabel-successor-transition/pilot-canonical-four-owner-paired-k8-v1`
- Additional two variants per owner:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-image2299-near-complete-human-relabel-successor-transition/variant-robustness-additional-two-per-owner-paired-k8-v1`
- One-call execution smoke:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-image2299-near-complete-human-relabel-successor-transition/smoke-one-call-owner0003-exact-token-v5`
- Offline analyzer:
  `scripts/research/analyze_image2299_near_complete_relabel_successor_transition.py`.

The replay used exact donor prompt token identifiers because the nested natural
assistant prefix does not survive a decode-and-retokenize round trip. It also
used the local direct sampling context because the historical persisted runtime
attestation points to a removed worktree path. Every call bundle still records
the executed model, tokenizer, decode policy, seed, prompt hash, and runtime
identity. This is sufficient for this bounded exploratory panel but is weaker
lineage than a current persisted runtime capability.
