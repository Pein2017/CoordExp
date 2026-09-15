---
title: Prefix-State Phrase-Geometry Factorial
description: Exact-prefix causal factorization of the description and geometry carried by one completed detection row.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-prefix-state-phrase-geometry-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Prefix-State Phrase-Geometry Factorial

The unit is complete. See [the verified results and bounded
verdict](results.md).

## Question

At the exact image-`12576` **prefix state 56 (`P56`)**, meaning the legal
assistant continuation containing 56 generated tokens immediately before the
sampled target-pizza row, which part of an appended complete row controls the
first subsequent object action:

1. the description token;
2. the four coordinate tokens;
3. their coherent conjunction; or
4. only the generic fact that one more valid row was completed?

The unit is falsifiable: a description main effect, geometry main effect,
description-by-geometry interaction, or absence of all three leads to a
different next research route.

## Competing Explanations

The leading explanation from the preceding [sampled-rescue causal-replay
unit](../2026-07-14-sampled-rescue-object-transition-causal-replay/results.md)
is a geometry/order/frontier-like successor process plus generic complete-row
advancement. The strongest alternatives are category or lexical inhibition,
a phrase-geometry conjunction, and a distributed prefix-state update that is
insensitive to the row's object content.

The primary control is a two-by-two factorial in which the description and
geometry are crossed while every row remains a grammar-valid, equal-length
nine-token complete row appended at the identical `P56` boundary.

## Primary Observation

For each condition, classify the first complete free next-row action as:

- target pizza `coco-ann:1571077`;
- left cup `coco-ann:678023`;
- right cup `coco-ann:678923`;
- already covered pizza `coco-ann:1571233`;
- another supported object;
- unsupported or uncertain object;
- immediate terminal action; or
- invalid output.

The object-reference opening token is structural and is not object evidence.
Phrase owner, geometry owner, parser validity, and first free description-token
evidence remain separate diagnostics.

## Factorial and Controls

All complete-row conditions use this exact token layout:

```text
object-reference start
one description token
object-reference end
box start
four coordinate tokens
box end
```

The four primary conditions are:

| Complete-row condition | Description factor | Geometry factor |
|---|---|---|
| `pizza-description__target-pizza-geometry` | `pizza` | rescued target pizza `coco-ann:1571077` |
| `pizza-description__left-cup-geometry` | `pizza` | competing left cup `coco-ann:678023` |
| `cup-description__target-pizza-geometry` | `cup` | rescued target pizza `coco-ann:1571077` |
| `cup-description__left-cup-geometry` | `cup` | competing left cup `coco-ann:678023` |

Three controls are retained:

- `no-appended-row`: unchanged `P56`; this is intentionally not part of the
  equal-length factorial estimate;
- `covered-pizza-description__covered-pizza-geometry`: exact nine-token append
  of the already committed pizza `coco-ann:1571233`, whose geometry is distinct
  from the rescued target pizza;
- `unsupported-chair-description__unsupported-region-geometry`: exact
  nine-token append of the previously reviewed unsupported-chair row. This is
  an unsupported-row control under the frozen case review, not proof that the
  region contains no possible object.

Every complete row is newly appended at `P56`. Historical
`syntax_only/context_mismatch` receipts may supply reviewed token factors but
must not supply the causal prefix state.

## Prediction Matrix

| Explanation | Required pattern |
|---|---|
| Geometry/order/frontier state | Conditions sharing geometry have similar next-action distributions; target-pizza geometry advances toward the left cup and left-cup geometry advances toward the right cup, regardless of description. |
| Category or lexical inhibition | Conditions sharing description have similar next-action distributions and geometry swaps have little effect. |
| Phrase-geometry conjunction | Coherent diagonal rows differ from the two crossed rows; neither single factor explains the result. |
| Generic complete-row advancement | All equal-length rows have approximately the same next-action distribution, including the unsupported-row control; no description or geometry main effect appears. |

Object-specific commit-to-uncovered redistribution is not restored merely by
advancing to the left cup. Such a claim would require the completed object to
be selectively suppressed and probability to move to a predeclared uncovered
object more specifically than under the unsupported and covered-duplicate
controls.

## Execution Outline

1. Reuse the current model loader, legal assistant-continuation renderer,
   request-scoped sampling backend, parser, and compact receipt writer.
2. Add one experiment-local composed-row seam that reads a reviewed
   description source row and geometry source row, validates both canonical
   rows, composes one nine-token row, and appends it to the exact recipient
   prefix.
3. Run one greedy plus two paired sampled requests for the four factorial arms
   and no-appended-row smoke. Stop on any token-count, grammar, runtime-identity,
   or exact-prefix mismatch.
4. If the smoke passes, run one greedy plus eight paired temperature-`0.4`,
   top-p nucleus-threshold-`0.95` requests for all six complete-row conditions
   and the no-appended-row control. Sampling uses repetition penalty `1.0` and
   physical batch size four.
5. Close on the first-action distribution matrix. Add candidate forward scores,
   a second image, or another temperature only if the primary matrix is
   genuinely ambiguous.

## Scope and Invariants

- Worktree: `/data/CoordExp/.worktrees/research-probes`.
- Image: Common Objects in Context validation image `12576` only.
- Inference configuration:
  `/data/CoordExp/.worktrees/research-probes/configs/coordexp_infras/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml`.
- Source data:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`.
- Checkpoint: step `4,887` from the configuration above.
- Recipient state: exact `P56` prompt and generated-prefix token identifiers,
  not decoded-text equivalence alone.
- Vision input, processor, tokenizer, prompt, adapter, special-token embeddings,
  parser, maximum token budget, repetition penalty, temperature, and top-p
  nucleus threshold remain fixed across compared conditions.
- All four factorial rows and both equal-length controls contain exactly nine
  tokens and occupy the same appended prefix positions.
- Sampling seeds are paired across all conditions and remain explicit in every
  receipt.

## Artifact Handle

The logical artifact root is:

```text
outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-prefix-state-phrase-geometry-factorial/<immutable-run-id>/
```

The resolved durable root will be under `/data/CoordExp/outputs/research/` and
will contain one compact receipt per condition, raw call bundles, a composed-row
provenance record, and one cross-condition summary.

## Non-Goals and Stop Rules

- no training, OpenSpec change, residual replay, attention sweep, final
  architecture, or population metric;
- no claim that a forced textual row is a native generated commit;
- no reuse of historical context-mismatched controls as causal evidence;
- stop before the full panel if any condition is not an exact `P56` append, is
  not a valid nine-token row, or changes the runtime identity;
- if all equal-length conditions collapse to one successor, close with bounded
  support for generic row advancement and do not escalate within this unit;
- if a clear main effect or interaction appears, record it as a one-case
  mechanism result. Independent replication remains required before training
  or architecture promotion.
