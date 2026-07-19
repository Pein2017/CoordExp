---
title: Sampled-History Target Reachability and Complete-Row Causal Value Results
description: Interim evidence from the extended greedy screen and crop-assisted physical-owner review, before the sampled-prefix sufficiency ladder.
type: investigation-results
role: evidence-record
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-07-19-sampled-history-target-reachability-and-complete-row-value
topic: qwen3-vl-dense-enumeration
status: active
evidence_status: stage_1_complete
updated: 2026-07-19
---

# Sampled-History Target Reachability and Complete-Row Causal Value Results

## Current verdict

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

## Claims still forbidden

Stage 1 does not show that:

- sampled history causes target access;
- any single sampled row improves later safe unique-object coverage;
- the model has or lacks a covered-set carrier;
- a local row preference is a justified training treatment; or
- delayed access is a final recall failure.

The next evidence must come from the exact sampled-prefix ladder and, only at
an adjacent unreachable-to-reachable transition, the same-parent complete-row
intervention defined in [unit.md](unit.md).
