# Pixel-Painted Row Coverage Design

Status: branch-local research design; not an OpenSpec contract.

Date: 2026-06-08

Scope: `configs/stage1/detection_teacher_forcing/` and the active
`stage1_detection_teacher_forcing` training route.

Related prior design:
`docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-design.md`

## Decision

Build **pixel-painted row coverage** as a separate opt-in upper-bound probe for
explicit visual external memory.

The mechanism should repaint the input image for each row state so the visual
encoder sees coverage markers from the beginning of the visual pipeline:

```text
image_k = Paint(image, coverage(objects before row k))
V_k = vision_encoder(image_k)
p(row_k | prompt_with_marker_instruction, text_prefix(S_k), V_k)
```

This differs from the earlier latent row-conditioned visual coverage prototype:

```text
V = vision_encoder(image)
V'_k = V + coverage_residual(coverage(objects before row k))
p(row_k | prompt, text_prefix(S_k), V'_k)
```

The new feature preserves the earlier latent implementation as a comparator.
It must not silently replace `row_conditioned_visual_coverage` behavior.

## Research Meaning

This feature tests whether explicit coverage memory helps at all when presented
as a strong visual signal. It is an upper-bound experiment, not an efficiency
claim.

The motivating hypothesis is that latent residual injection may happen too late
or too weakly in the visual pipeline. Pixel painting forces coverage markers to
participate in early patch embedding, visual attention, object grouping, visual
salience, projector input construction, and language-side decoding.

The mechanism does not add new information in the Bayes-optimal sense because
coverage is a deterministic function of the prefix. Its value is an inductive
bias and optimization shortcut: the model no longer has to reconstruct visual
coverage from text coordinates alone.

## Goals

- Test whether explicit visual external memory reduces duplicate enumeration,
  repeated-region bursts, object count error, and premature EOS.
- Test whether early visual intervention helps state tracking and object
  binding more than late latent residuals.
- Keep existing latent row coverage and no-coverage row-state baselines as
  separate comparators.
- Allow an explicit system-prompt instruction that tells the model how to
  interpret painted regions.
- Prioritize scientific clarity over visual-forward efficiency.

## Non-Goals

- Do not make this the global default Stage-1 training behavior; the feature
  remains explicitly gated by `pixel_painted_row_coverage.enabled`.
- Do not claim deployment efficiency; repeated visual forward passes are
  acceptable for this probe.
- Do not replace or reinterpret the existing latent
  `row_conditioned_visual_coverage` config.
- Do not introduce residual-set correction, duplicate-unlikelihood, forced EOS,
  fallback decoding, or new non-CE objectives in the first version.
- Do not treat teacher-forced `token_acc` as the primary success criterion.

## Initial Contract

The first version should be opt-in through a new config section rather than by
overloading `row_conditioned_visual_coverage`.

Canonical working name:

```yaml
pixel_painted_row_coverage:
  enabled: true
  version: v1
  row_state_policy: prefix_expansion
  marker:
    style: boundary_translucent_tint
    boundary_rgb: [0, 255, 255]
    interior_rgb: [0, 255, 255]
    interior_alpha: 0.18
    boundary_width_px: 4
  prompt:
    marker_instruction: true
  training:
    unpacked_only: true
    include_terminal_state: true
```

The feature is limited to unpacked Stage-1 detection teacher-forcing row states
until a later design explicitly handles packing, cache reuse, or deployment
rollout efficiency.

The config surface is intentionally strict and minimal. V1 does not expose
arbitrary marker text, learned paint parameters, cache paths, rollout knobs, or
alternate marker styles. `style` is fixed to `boundary_translucent_tint` to
match the accepted non-occluding marker semantics. RGB, alpha, and boundary
width are explicit for reproducibility.

V1 implementation order is **training-path first**. The first implementation
should paint teacher-forced row-state training samples and structure the painter
so rollout can reuse it later. Fully wiring generation-time rollout painting is
deferred until the training/config/data path is verified.

## Training Surface Default

The canonical V1 pixel-painted experiment should compete with the strongest
all-layer Stage-1 benchmark, not only the earlier freeze-ViT comparator.

The default checked-in pixel-painted training config therefore inherits the
all-layer LoRA surface from the matched row-coverage SFT config:

- `training.train_type: lora`;
- `training.freeze_vit: false`;
- `training.freeze_aligner: false`;
- LoRA targets include `model.visual.blocks.*`, visual merger/deepstack merger
  MLPs, and all language-model layers;
- `training.optimizer: multimodal_coord_offset`;
- `token_rows.enabled: true`, so the coordinate token rows and
  `coord_offset_adapter` participate in training.

This is the intended competitive default for the upper-bound probe. A
freeze-ViT version may still be useful as a diagnostic ablation, but it is not
the primary claim path.

Pixel-painted checked-in configs should write under a pixel-painted artifact
root rather than the latent row-coverage root. This keeps artifact provenance
readable even when the config inherits most training knobs from a latent
row-coverage comparator.

V1 is mutually exclusive with latent row-conditioned visual coverage. A config
must not enable both:

```yaml
row_conditioned_visual_coverage:
  enabled: true
pixel_painted_row_coverage:
  enabled: true
```

This keeps the first claim interpretable: a run tests either late latent
feature marking or early pixel-space optical marking, not a mixture of both.

V1 paints images **in memory** for teacher-forced row-state training. It does
not materialize a persistent painted-image cache in the first implementation.
The original dataset image identity remains the provenance handle, while the
row-state sample carries pixel-paint metadata describing the applied marker.

This keeps the upper-bound probe reversible and avoids creating an image tree
for every `(image, row_state_k)` pair. The painter should still be structured
as a reusable function so later rollout code or visualization tools can call the
same optical operation.

The row-state semantics match the earlier row-coverage protocol:

```text
state 0: paint empty coverage -> predict row 1
state 1: paint coverage(row 1) -> predict row 2
state N: paint coverage(rows 1..N) -> predict assistant stop
```

The target row and future rows must not be included in the painted coverage.

## Prompt Contract

V1 may explicitly describe the marker in the system or task prompt. This is
intentional: the experiment tests a strong, instruction-aware version of the
coverage idea.

The first implementation defaults to a fixed built-in system-prompt marker
instruction when pixel painting is enabled:

```yaml
pixel_painted_row_coverage:
  enabled: true
  prompt:
    marker_instruction: true
```

The instruction text is code-owned, not arbitrary YAML text. This keeps runs
reproducible and allows a clean ablation with `marker_instruction: false`.

Prompt wording should preserve the distinction between:

```text
marked = already emitted in the prefix
```

and unsupported claims such as:

```text
marked = not an object
marked = background
marked = never attend
marked = always stop when most of the image is marked
```

The prompt should encourage enumeration of remaining unmarked objects while
still allowing valid overlapping, partially covered, or contained objects.

## Marker Semantics

V1 uses a fixed optical paint marker, not a learnable residual perturbation and
not an erasure operation.

The accepted marker style is **boundary plus non-occluding translucent interior
tint**:

- draw a visible bbox boundary around previously emitted objects;
- apply a light translucent tint inside the bbox;
- preserve object evidence inside the painted region;
- keep boundary edges visually distinguishable from the original image;
- avoid hard masking, blurring, blackout, or replacing pixels with a flat fill.

The model should still perceive object information inside painted regions while
also learning that the marker means:

```text
this region or object extent has already been emitted in the prefix
```

This is important for crowded and overlapping scenes. A painted region may
still contain valid remaining visual evidence, so the marker must not encode
"background" or "ignore all pixels here."

## Evidence Gates

The first claim path requires matched comparisons:

- no row-state coverage baseline;
- latent row-conditioned visual coverage;
- pixel-painted row coverage without prompt instruction;
- pixel-painted row coverage with prompt instruction.

The primary comparison should use the same all-layer trainable surface as the
SOTA benchmark. Freeze-ViT comparisons are secondary diagnostics only.

Launch preflight:

- resolve the checked-in pixel-painted config and confirm all-layer trainability
  knobs, `multimodal_coord_offset`, token rows, pixel-enabled, and latent-off;
- run a narrow real Qwen/MS-Swift template encode smoke with one in-memory
  painted PIL image, `do_resize=false`, and expected image tensor/grid fields;
- confirm pixel-paint diagnostic metadata is stripped before `model(**inputs)`.

Primary diagnostics:

- duplication rate;
- repeated-region burst rate;
- object count error;
- premature EOS rate;
- invalid parse/drop counters;
- recall and AP/AR;
- crowded and high-overlap slices;
- rollout examples with rendered overlays.

Teacher-forced token accuracy is a secondary diagnostic only.

## Open Design Forks

Resolved marker semantics:

- V1 uses boundary plus non-occluding translucent interior tint.
- The marker preserves object evidence inside painted regions.
- Hard occlusion, blur, blackout, and coverage-only replacement are out of
  scope for V1.
- V1 implementation starts with teacher-forced row-state training. Rollout
  reuse is a design constraint, but fully wired rollout painting is deferred
  from the first implementation step.
- V1 is mutually exclusive with latent `row_conditioned_visual_coverage`.
- V1 defaults to a built-in system-prompt marker instruction, with
  `prompt.marker_instruction: false` as the no-instruction ablation.
- V1 training uses in-memory painted image objects. No persistent
  painted-image cache is introduced in the first implementation.
- V1 uses a strict minimal config surface with fixed
  `marker.style: boundary_translucent_tint`, default cyan marker colors,
  `interior_alpha: 0.18`, and `boundary_width_px: 4`.
- V1's canonical checked-in experiment config defaults to the all-layer
  trainable surface: visual blocks, visual projector/merger MLPs, language
  layers, and coordinate token rows/`coord_offset_adapter`.

Remaining forks: tests and rollout diagnostics must be resolved before
implementation.
