---
title: Sampled-History Target Reachability and Complete-Row Causal Value Results
description: Final bounded evidence on whether natural sampled row history makes later physical objects greedily reachable and safely improves unique-object coverage.
type: investigation-results
role: evidence-record
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-07-19-sampled-history-target-reachability-and-complete-row-value
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: stage_7_complete
updated: 2026-07-19
---

# Sampled-History Target Reachability and Complete-Row Causal Value Results

## Final verdict

Natural sampled history is causally active, but the evidence does not establish
a clean object-coverage state or a general benefit.

Three admissible same-parent complete-row interventions all changed later
physical-owner access. All three branch-row pairs named the same physical
owner and differed only in coordinates. Only image `7816` produced a safe
final gain: person owner `211764` was added without losing another owner or
adding a duplicate, increasing the final strict unique-owner count from `9`
to `10`. Images `12576` and `18380` only exchanged later owners.

Deeper controls reject one simple carrier. On image `7816`, either sampled
`x1` or sampled `y1` in row 4 is sufficient to cross into the rescued-person
route, while `y2` alone is not. On image `12576`, neither sampled row 3 nor
sampled row 4 alone recovers the sampled endpoint; their exact joint history
is required.

The evidence supports a distributed, coordinate-sensitive prefix state that
changes later object competition. It does not establish a semantic object
ledger, a monotonic covered set, a universal latest-row carrier, or a safe
training target. Training and architecture promotion remain blocked.

## Stage 1 interim verdict

Stage 1 changes the interpretation of the discovery pool. Three of six frozen
"sampled-only within eight rows" targets are not final greedy omissions:
greedy emits each of them at zero-based row 8, immediately after the original
eight-row window. The other three remain absent when greedy naturally
terminates, and crop-assisted review confirms that they are distinct physical
objects rather than annotation or matching artifacts.

This evidence is sufficient to start a stratified sampled-prefix sufficiency
ladder. It is not sufficient to propose a training objective or claim a
covered-set mechanism.

## Frozen execution evidence

The canonical Stage 1 artifact is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-19-sampled-history-target-reachability-and-complete-row-value/
stage1-extended-root-greedy-fp32-v1/union.json
```

Its SHA-256 digest is:

```text
ac3ee031f75f3344610322a1baba89cadb305ddfcab8bfc708b4b977226b6875
```

All eight shards passed exact model, adapter, special-token embedding,
inference-config, source-data, image, prompt, processed-media, and first-eight
raw-token parity checks. The run used Hugging Face generation, 32-bit floating
point model parameters, physical batch size one, greedy decoding, repetition
penalty 1.0, and a total budget of 512 newly generated token identifiers.

## Stage 1 target classification

| Image | Frozen physical owner | Extended greedy result | First greedy hit | Interpretation |
|---|---|---|---:|---|
| `2299` | person `-2` | natural terminal at row 24 | none | genuine terminal omission |
| `7816` | person `211764` | natural terminal at row 10 | none | genuine terminal omission |
| `12576` | cup `678023` | natural terminal after later rows | row 8 | route delay, not final omission |
| `18380` | person `2030411` | 512-token cap | row 8 | route delay; final set remains censored |
| `19109` | person `1726831` | natural terminal at row 27 | none | genuine terminal omission |
| `19432` | chair `384172` | natural terminal after later rows | row 8 | geometry/refinement diagnostic only |

Images `9400` and `9590` remain negative discovery controls because the frozen
eight-row artifacts contain no eligible sampled-only physical owner.

## Crop-assisted physical-owner review

The three terminal omissions survive direct review of enlarged crops and the
original images.

- Image `2299`, owner `-2`, is a distinct boy in the top row. The sampled box
  reaches him with Intersection over Union 0.561, although it includes some
  adjacent-person pixels. The closest greedy row belongs to neighboring owner
  `-9`, not the target.
- Image `7816`, owner `211764`, is a small, heavily occluded white-shirted
  person between two larger adults. Greedy emits the neighboring red-shirted
  adult twice and never emits the target. The sampled target row has
  Intersection over Union 0.594.
- Image `19109`, owner `1726831`, is a distinct dark-clothed person near the
  cafe tables. Greedy emits the adjacent tan-coated person and later enters a
  repeated motorcycle/localization basin before stopping. The sampled target
  row has Intersection over Union 0.689.

The unresolved greedy rows in these images mostly depict real objects with
partial, shifted, oversized, or mixed geometry. They are not automatically
hallucinations and remain excluded from strict owner-set claims.

## Root-decision control

Image `19109`, owner `1680320`, is retained as a clean same-prefix stochastic
choice control rather than a history-ladder target. Sample seed 11 emits the
striped-hat person at row 0 with Intersection over Union 0.809. Root greedy
chooses a different person and never emits owner `1680320` before natural
termination.

Because both choices start from the empty assistant prefix, this case proves
that some sampling-only access exists without accumulated sampled history. It
prevents us from attributing every sampled rescue to history or commitment.

## Stage 2 admission strata

The complete frozen decision table is
[stage2-admission.json](stage2-admission.json). It was written before any
prefix-ladder generation.

Run the sampled-prefix sufficiency ladder on five primary or provisional
targets, but preserve their different scientific meanings:

| Stratum | Images | Allowed interpretation |
|---|---|---|
| terminal omission | `2299`, `7816`, `19109` | sampled history may alter final target reachability |
| route delay | `12576` | sampled history may accelerate or reorder access |
| route delay with final censoring | `18380` | bounded route effect only |

Retain image `19432` only as a geometry/refinement diagnostic. Its sampled row
0 already emits an extremely large chair box whose top physical candidate is
the frozen chair owner `384172`. A later tight row may refine an ambiguously
represented owner rather than discover a new owner.

For image `12576`, the only earlier unresolved row is a chair while the target
is a cup. For image `18380`, earlier unresolved person boxes have zero target
overlap and large center distance. For image `19109`, direct crop review shows
that earlier unresolved rows depict other people or motorcycles. These checks
make the three histories usable as provisional mechanism probes without
silently promoting unresolved rows to negative evidence.

## Claims still forbidden after Stage 1

Stage 1 does not show that:

- sampled history causes target access;
- any single sampled row improves later safe unique-object coverage;
- the model has or lacks a covered-set carrier;
- a local row preference is a justified training treatment; or
- delayed access is a final recall failure.

The next evidence must come from the exact sampled-prefix ladder and, only at
an adjacent unreachable-to-reachable transition, the same-parent complete-row
intervention defined in [unit.md](unit.md).

The remainder of this record supplies that evidence and supersedes the
Stage-1-only restrictions above where it makes a narrower final decision.

## Stages 2 through 7 canonical evidence

All paths below are under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-19-sampled-history-target-reachability-and-complete-row-value/
```

| Stage | Artifact | SHA-256 digest |
|---|---|---|
| 2: sampled-prefix ladder | `stage2-prefix-sufficiency-ladder-fp32-v1/union.json` | `cdfc50132fed1a5cf1b2a4ecd3c12b34e0ee2e4dae09fe6e0b3447e88b6d82e3` |
| 3: same-parent complete-row intervention | `stage3-same-parent-complete-row-intervention-fp32-v1/union.json` | `3d32b66c55379452515cb4cc361ee233ab62ba035601e363cd16241a3b7c1ee9` |
| 4: image-`7816` final horizon | `stage4-final-horizon-fp32-v1/image-7816.json` | `9a5661855502b18cad253b522314fc73a935ca0b2a71a9c438ca0cf569218f2e` |
| 5: image-`7816` row-0 by row-4 crossover | `stage5-coordinate-history-crossover-fp32-v1/image-7816.json` | `fddec82836a86809ffd05a28181912e5aaf8bb2aa5606b384ac201f9bd973445` |
| 6: image-`7816` row-4 coordinate factorial | `stage6-row-four-coordinate-factorial-fp32-v1/image-7816.json` | `9f349fe9ee0d2a97b083739629b95fe5fbb42adf5887db6b91c99ba15c343baf` |
| 6: candidate-row scoring | `stage6-row-four-coordinate-factorial-fp32-v1/candidate-scoring-fp32-v1/receipt.json` | `b8c5fbfab2ea0dc6f642cd13b7be6fb5ef0f8822d374a22e7f979c41cf139198` |
| 7: image-`12576` row-3 by row-4 crossover | `stage7-image-12576-row-three-row-four-mediation-crossover-fp32-v1/image-12576.json` | `464d4daa2c11be6ee83872c42e9ceb054d44ecea6755b94fe794bfd78921e09b` |

All stages retained the frozen model and numerical contract. Exact model,
adapter, special coordinate-token embedding, prompt, image, processed-media,
configuration, source-file, and raw-prefix checks passed. Relevant native
endpoint and no-op token-parity gates also passed.

## Stage 2: sampled-prefix reachability is real but not monotonic

Let `P_k` denote the exact sampled prefix after `k` complete sampled rows. The
observed target-reachability sequences were:

| Image | Reachability from `P_0, P_1, ...` | Interpretation |
|---|---|---|
| `2299` | miss, miss | sampled history never makes the frozen target greedy within the tested ladder |
| `7816` | miss, hit, unresolved, hit, hit, unresolved | early access exists, but later prefixes do not preserve a monotonic state |
| `12576` | miss, miss, miss, miss, hit, hit, hit, hit | a stable tested transition appears after four sampled rows |
| `18380` | miss, miss, hit, hit, hit, hit | a stable tested transition appears after two sampled rows |
| `19109` | miss, miss, hit, miss, miss, miss | access appears and then disappears |
| `19432` | unresolved, hit, hit, hit, hit, hit, hit, hit | geometry or refinement diagnostic only |

Only adjacent clean-miss-to-hit transitions entered Stage 3. This left images
`7816`, `12576`, and `18380`.

The non-monotonic cases reject a simple account in which every emitted row only
adds durable information to a growing covered set. They remain compatible with
a fragile recurrent state whose later content can reinforce, overwrite, or
redirect earlier route information.

## Stage 3: one complete same-owner row changes later access

Every admitted sampled row and native greedy row names the same physical owner,
has the same nine-token structure, and differs only in coordinates.

| Image | Branch owner | Sampled minus native coordinate bins | Later owners gained | Later owners lost | Strict unique-owner change |
|---|---:|---|---|---|---:|
| `7816` | `2153918` | `[-1, +1, +2, +2]` | `211764` | none | `+1` |
| `12576` | `1509406` | `[+6, -1, +3, +1]` | `678023` | `1571077` | `0` |
| `18380` | `1330599` | `[+7, 0, 0, -3]` | `1329602`, `2030411` | `1225902`, `1317070` | `0` |

The native-row no-op replay passes in all three cases. The later frozen target
appears only in the sampled-row arm for each admitted comparison.

This establishes causal sensitivity to natural same-owner row geometry. It
does not establish whether the coordinates act as a spatial frontier, object
commit, language continuation cue, or a mixture. Images `12576` and `18380`
are route exchanges, not coverage gains.

## Stage 4: the image-`7816` gain survives natural termination

Both image-`7816` arms naturally terminate at row `10` under the 512-token
contract.

- Native-coordinate arm: `9` strict unique physical owners, duplicate owner
  `205108`, target owner `211764` absent.
- Sampled-coordinate arm: `10` strict unique physical owners, no duplicate,
  target owner `211764` present, and no verified owner lost.

This is the unit's only safe final unique-object improvement. It proves that a
natural coordinate-history variation can improve final enumeration on one
image. One image is not enough to define a general training objective.

## Stage 5: row 4 is sufficient within image `7816`

The row-0 by row-4 crossover holds the later suffix fixed and swaps the native
or sampled coordinate variants at the two changed positions.

| Row 0 variant | Row 4 variant | First later owner |
|---|---|---|
| native | native | duplicate person `205108` |
| sampled | native | duplicate person `205108` |
| native | sampled | rescued person `211764` |
| sampled | sampled | rescued person `211764` |

Within this exact image and route, row 4 screens the earlier row-0 variation.
This permits a row-4-local follow-up. It does not establish that the latest row
is generally sufficient in other images.

## Stage 6: row-4 coordinates act separately and directionally

Native row 4 is person owner `205108` at coordinate bins
`[877, 132, 927, 432]`; sampled row 4 is the same owner at
`[874, 129, 927, 421]`.

The eight-arm factorial varies the natural native or sampled value of `x1`,
`y1`, and `y2`, while holding identical `x2` fixed.

- Native `x1` plus native `y1` returns owner `205108`, regardless of `y2`.
- Sampled `x1` is sufficient to return owner `211764`, even with native `y1`
  and `y2`.
- Sampled `y1` is also sufficient to return owner `211764`, even with native
  `x1` and `y2`.
- Sampled `y2` alone is insufficient.

All eight factorial arms naturally terminate at row `10`. Candidate-row scores
change mainly at the target row's `x1` and `y1`; the row-entry versus terminal
margin stays strongly positive. The effect is not caused by forcing the model
to continue.

Two controls are equally important:

- the reverse natural coordinate direction generates an unmatched hybrid box
  `[894, 145, 935, 347]`;
- an orthogonal `x2`-only control generates another unmatched hybrid box
  `[894, 134, 935, 432]`.

Neither control retrieves the target. The signal is coordinate-position-
specific and direction-sensitive, but it is not a clean semantic object
pointer. Arbitrary coordinate mixing can create geometry belonging to no
verified owner.

## Stage 7: an independent image rejects a universal latest-row carrier

Image `12576` supplies a bounded replication with rows 5 and 6 held fixed.
Rows 3 and 4 each have native and sampled coordinate variants for the same
physical cup owners. The four exact 65-token prefixes produce:

| Row 3 variant | Row 4 variant | Greedy row 7 |
|---|---|---|
| native | native | pizza owner `1571077` |
| sampled | native | pizza owner `1571077` |
| native | sampled | pizza owner `1571077` |
| sampled | sampled | cup target owner `678023` |

Both endpoint arms exactly replay their frozen nine-token rows. Both crossed
arms return the exact native pizza row. All four outputs are complete and
parser-clean.

The strongest allowed conclusion is:

> With rows 5 and 6 fixed, the sampled row-7 branch on image `12576` requires
> the joint sampled row-3 and sampled row-4 coordinate history. Neither changed
> row is independently sufficient in the two crossed prefixes.

This is consistent with nonlinear interaction, sub-threshold accumulation, or
compatibility with one of two familiar endpoint prefixes. The crossed prefixes
are artificial combinations that the model did not naturally generate, so the
result is not formal hidden-state mediation and does not prove semantic
multi-row reasoning. The artifact's local field
`primary_mediation_claim_allowed: true` records that its executable parity gate
passed; it is not accepted here as permission to claim independent row-4
mediation.

## Mechanism update

The most economical account consistent with all stages is:

```text
image evidence + exact generated-prefix state
    -> route-conditioned competition among candidate rows
    -> an early category or coordinate branch
    -> autoregressive completion of one physical-object row
```

The prefix state is not merely a list of covered object identifiers. Coordinate
tokens from earlier rows can change which later region or owner becomes locally
greedy-compatible. The state can persist across intervening common rows, but it
can also disappear or require a conjunction of multiple earlier variations.

| Claim | Decision |
|---|---|
| Previous rows causally influence later object selection. | Supported in bounded exact-prefix cases. |
| The influence can safely improve final unique-object coverage. | Supported only on image `7816`. |
| The influential state is a semantic, monotonic covered-object ledger. | Not supported; several results argue against this simple form. |
| One latest complete row generally owns the effect. | Rejected by image `12576`. |
| A local complete-row or coordinate preference is ready for training. | Rejected by the multi-image promotion gate. |

## Training and architecture decision

The preregistered promotion gate required at least three safe positive cases
and at least four fully interpretable targets for a general training proposal.
Only one case produces a safe final unique-owner gain. Two other causal cases
exchange owners, and the deeper controls reveal different temporal structures.

Therefore:

- do not launch the proposed 256-image training screen from this unit;
- do not train a universal sampled-row preference;
- do not treat sampled coordinate variants as positive labels in general;
- do not add a persistent ledger, object slot, or terminal suppression based on
  this evidence; and
- do not extend Stages 6 or 7 post hoc with more arms.

If this mechanism is revisited, it requires a separately admitted independent
cohort whose primary endpoint is safe final unique-owner gain. Candidate-row
scoring may then distinguish gradual probability shifts from abrupt joint
branch changes. That is a future discriminator, not an active training plan.

## Final limitations

- The admitted causal cohort contains only three images.
- All Stage-3 branch rows are same-owner coordinate variants; different-object
  commit is not tested.
- Physical-owner matching is conservative and cannot turn ambiguous geometry
  into negative evidence.
- Image `18380` remains right-censored at the long horizon.
- The image-`7816` and image-`12576` deeper controls use selected positive or
  route-exchange cases and do not estimate prevalence.
- Stage-7 crossed prefixes are off-policy constructions and may expose endpoint
  compatibility rather than a naturally used internal computation.

The unit is complete. It closes the route from sampled novelty directly to a
training objective while preserving the more modest finding that natural row
geometry is an executable component of Qwen3-VL's recurrent traversal state.
