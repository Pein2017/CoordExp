---
title: Native Qwen3-VL text-coordinate validation raw-material synthesis
type: investigation
role: archival-synthesis
authority: non_normative_research
status: complete_benchmark_baseline
updated: 2026-08-29
---

# Native text-coordinate Validation-200 baseline

## Session context and result

The repaired runner restored the pretrained chat template, rendered all 80
COCO names into the prompt, and enforced strict JSON/ontology parsing. The
greedy baseline scored mAP `0.147820`; the low-temperature repetition-penalty
sweep reached `0.199962` at RP 1.10, with fewer severe duplicate bursts but not
eliminating collapse. This is an ecological decode comparison, not a causal
serialization ablation.

The source benchmark record was
`/data/CoordExp/outputs/research/qwen3-vl-native-text-coordinate-val200/benchmark.md`;
its decision-bearing contents are copied into this packet before the raw root
was removed. No claim is made that the deleted per-row traces remain
reconstructable.

## Raw disposition

The 1.6G root was mostly raw JSONL/token traces and per-run metadata. This
Markdown preserves the prompt/template correction, the two reported mAP values,
the repetition observation and their interpretation. The deleted traces and
their detailed per-row audits are not preserved here. The historical disposition
was `DISTILL-DELETE`.
