# Prefix-Denoising Surgery Probing Design

Date: 2026-06-25

Status: approved design record for the next mechanistic research round.

## Purpose

Diagnose how prefix-denoising SFT changes the internal mechanism of a causal
V-LLM during object perception, object binding, coordinate-basin formation,
object-span emission, transition to the next object, duplication collapse, false
negative formation, and premature termination.

The primary comparison is not sorted versus random. The primary comparison is
prefix-denoising versus the closest non-denoising control under matched image,
template, prefix state, object role, token position, layer, and component.
Sorted/random ordering is a blocking factor and a source of useful nuisance
variation, not the main causal axis.

## Checkpoints And Rollout Evidence

Treatment checkpoints:

```text
sorted prefix-denoising:
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908

random prefix-denoising:
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_random_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-random-bsz1x128-4epoch/v2-20260623-133126/checkpoint-908
```

Available val200 free-rollout artifacts for harvesting cases:

```text
sorted prefix-denoising val200:
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_fullwrap_desc_sorted_fixed_fullwrap_desc_sorted_fixed_ckpt908_val200_rp1p10_ckpt908_val200_bsz8_temp0_rp1p10_max3084_chatfix_2gpu

random prefix-denoising val200:
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_fullwrap_desc_random_fixed_fullwrap_desc_random_fixed_ckpt908_val200_rp1p10_ckpt908_val200_bsz8_temp0_rp1p10_max3084_chatfix_2gpu
```

First control to find:

```text
closest compact_object_box_closed desc_first full-wrapper pure-CE checkpoint,
preferably with matching base model, template, token_embeddings_adapter surface,
ordering regime, and val200 rollout artifacts.
```

Fallback controls are allowed only after documenting the mismatch. Older
compact-full pure SFT checkpoints can be used as weak controls for mechanism
triangulation, but template mismatch must be carried through interpretation.

## Core Estimand

For a matched image base, token prefix, object/span episode, token position,
layer, and component, estimate the paired internal-state effect of
prefix-denoising training:

```text
Delta_internal =
  state_or_effect(prefix_denoising_model, image_base, prefix_state, position, layer, component)
  -
  state_or_effect(non_denoising_control, image_base, prefix_state, position, layer, component)
```

Observed AP, duplicate-guard counts, parse errors, and f1ish summaries are case
harvesting signals only. They do not define success and should not become the
main explanation.

## Sample-Base Doctrine

The unit of deep research is a representative `sample(image)-base`, not a broad
metric slice. A sample-base is one image plus the GT objects, rollout objects,
selected target objects, prefixes, generated spans, and manual rationale needed
to study a mechanism fork repeatedly across models and token positions.

Select a deliberately small initial panel:

```text
initial deep panel: 8-12 image bases
optional sanity panel: 2 easy controls
```

An image base is eligible only if it exposes a sharp mechanism question. Avoid
normal or well-learned images unless they serve as sanity controls for a specific
readout or intervention.

Required initial categories:

```text
2-3 false-negative bases:
  A visible object is omitted in free rollout or lost under a specific prefix,
  but the image/prefix offers a plausible path to test perception versus
  language-side guidance, delayed evidence, or wrong basin lock.

2-3 duplication-burst bases:
  The model emits near-duplicate boxes, repeated semantic anchors, repeated
  coordinate anchors, or a burst later suppressed by duplicate guards.

1-2 divergent same-image bases:
  Prefix-denoising and control models choose different object orders, spans,
  coordinate basins, or termination behavior on the same image.

1-2 boundary or termination bases:
  Empty prediction, premature `<|box_end|>`, premature `<|im_end|>`, wrapper
  router collapse, or strict-template mismatch creates a token-mode fork.

0-2 sanity controls:
  Easy correct images, included only to verify that probes and patch directions
  are not globally pathological.
```

Each selected image base must carry:

```text
image_id
dataset split
artifact roots used for harvest
models where the phenomenon appears
target object(s) and generated span(s)
failure family tags
short manual rationale: why this image is mechanistically valuable
minimum planned probes
manual-review status
```

Sample selection should prefer images that support multiple tests from the same
base: false-negative plus delayed-evidence check, duplication plus repeated
anchor check, boundary plus termination check, or control/treatment divergence
under the same prefix.

## Blocking Variables

Carry these variables through every row and summary:

```text
model_id
checkpoint_path
adapter surface, especially token_embeddings_adapter
training objective family: prefix_denoising or pure_ce
ordering regime: sorted or random
image_id
dataset split: train or val
object category
object index/order position
object scale and crowding tags
prefix mode
token position
layer
component/site
manual sample-base id
```

## Token Positions

Probe the state trajectory, not only the failure token:

```text
image/context end
object_ref_start
descriptor midpoint
descriptor end
object_ref_end
box_start
pre_x1
after_x1
after_x1_y1
after_x1_y1_x2
after_full_box / pre_box_end
box_end
next_object_ref_start
termination pressure / im_end pressure
```

For coordinate slots, track the coordinate-token basin over `<|coord_0|>` to
`<|coord_999|>`, not just exact next-token accuracy. Prior evidence says geometry
locality can survive while smoothness is damaged, so the probe must distinguish:

```text
exact target rank
distance-to-target rank
local window mass
coordinate-family mass
coordinate smoothness / neighborhood shape
wrapper-token competition
semantic-token competition
```

## Probe Families

Start with model/case registry and post-hoc harvesting. Escalate into model
runs only after the sample-base panel is curated.

1. Layerwise state atlas:
   capture hidden states, residual stream projections, coordinate-family mass,
   wrapper mass, semantic anchor logits, and termination pressure at all planned
   positions for each selected sample-base.

2. Paired residual difference flow:
   compare prefix-denoising versus control on the same image/prefix/position,
   especially around layers 16-18 for coordinate-basin onset and later layers
   for boundary/termination routing.

3. Component decomposition:
   separate layer input, self-attention output, MLP output, and final residual
   effects. Prior work shows boundary/router rows and coordinate-mode x1-onset
   rows must not be pooled.

4. Cross-model activation surgery:
   patch treatment/control states in both directions. Measure whether a state
   from one model can move the other model into a different coordinate basin,
   object span, duplicate loop, or termination decision.

5. Continuation surgery:
   after a targeted patch, continue generation far enough to observe whether
   the object span repairs, duplicates, shifts to another object, or terminates.

6. Perception versus language-guidance false-negative probe:
   for missing objects, compare pure visual evidence under empty/minimal prefix,
   correct language guidance, wrong-object guidance, correct coordinate seed,
   wrong coordinate seed, and hidden-state patching. A recoverable missing object
   is not a visual perception failure by default.

7. Coordinate-token surface probe:
   inspect the new `<|coord_*>` embeddings and output directions for local
   geometry, damaged smoothness, attraction basins, and layerwise emergence of
   coordinate-neighborhood mass.

## Interpretation Rules

- Do not assume prefix denoising succeeded mechanistically because the checkpoint
  was trained with a denoising objective.
- Do not assume prefix denoising failed mechanistically because clean/noisy CE or
  KL was small on average.
- Do not pool wrapper-mode boundary failures with coordinate-mode x1-onset
  failures.
- Do not call a false negative a visual-perception failure until guided prefix,
  coordinate-seed, and hidden-state intervention probes fail.
- Do not treat broad val200 metrics as the explanation.
- When a path is promising and has high influence over the final mechanism
  picture, it is acceptable to dive deeper and dynamically adjust the roadmap.

## First Durable Output

The first implementation artifact should be:

```text
sample_base_registry.jsonl
sample_base_summary.md
```

These files should list the selected image bases, why they matter, what evidence
selected them, and the minimum model-backed probes each one deserves.
