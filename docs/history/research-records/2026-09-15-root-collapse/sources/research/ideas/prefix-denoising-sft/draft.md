---
type: idea
title: Prefix Denoising SFT Draft
description: Original V1 idea and rationale for clean/noisy prefix-denoising Stage-1 compact detection SFT.
tags: [stage1, prefix-denoising, draft]
updated: 2026-06-20
---

# Prefix Denoising SFT Draft

## Motivation

The V1 idea targets a specific Stage-1 compact detection failure mode: a model
that is trained only under clean teacher-forced coordinate histories may be
fragile when free autoregressive decoding drifts. The proposed intervention is
to train the same clean target continuation under both clean and mildly
coordinate-noised prefixes.

The idea is deliberately Stage-1 first. It uses the existing compact detection
teacher-forcing route and avoids changing Stage-2 rollout-correction behavior
in the first slice.

## V1 Shape

Each base image yields two full multimodal views:

- `clean_full`: clean GT object sequence as input and labels.
- `noisy_full`: valid bbox-coordinate-corrupted object sequence as input, clean
  GT object sequence as labels.

The main CE objective is branch-balanced hard clean-label CE:

```text
0.5 * CE(clean_full_input, clean_labels)
+ 0.5 * CE(noisy_full_input, clean_labels)
```

When KL is enabled, sparse selected-object coordinate sites compare
`stopgrad(clean_full)` teacher logits against `noisy_full` student logits inside
a GT-centered local coordinate-token window.

## Accepted Constraints

- Use hard clean-label CE only for V1; do not activate valid-set marginal CE,
  multi-positive trie CE, coordinate SoftCE, or recursive objective features.
- Keep object order deterministic and clean-GT sorted.
- Corrupt bbox coordinates as valid whole boxes in norm1000 `xyxy` space.
- Require all four quantized bbox coordinates to change for noisy objects.
- Keep schema, class, description, delimiter, and stop tokens clean.
- Disable encoded-sample cache because noise and selected KL sites may vary by
  epoch while length remains stable.
- Treat `prefix_denoising.current_object_kl.weight = 0.0` as the CE-only
  denoising ablation, not as a separate feature surface.

## Non-Goals

- Do not implement free-decode or rollout evaluation in the V1 implementation
  slice.
- Do not add invalid, malformed, hard-negative, text, class, or schema
  corruption.
- Do not add a new clean-baseline config; use matched external comparison
  artifacts when drawing claims.
- Do not edit upstream HF/Qwen model files.
- Do not interpret launch-health as exposure-bias improvement.

## Sources

- `progress/directions/prefix_denoising_sft_v1.md`
- `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
