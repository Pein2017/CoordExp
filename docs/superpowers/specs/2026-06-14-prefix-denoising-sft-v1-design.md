# Prefix-Denoising SFT V1 Implementation Design

Date: 2026-06-14

Status: design approved for durable recording. Implementation is not yet authorized
until the user reviews this written spec and approves the follow-up
implementation plan.

Owner surface: Stage-1 compact detection teacher forcing under
`stage1_detection_teacher_forcing`, with new prefix-denoising V1 code isolated
from Stage-2 rollout-correction surfaces.

Primary direction note:
`progress/directions/prefix_denoising_sft_v1.md`.

## Purpose

Implement prefix-denoising SFT V1 as a packed, hard-CE Stage-1 training
extension that teaches the autoregressive coord model to recover from valid but
drifted bbox-coordinate prefixes.

For each image, V1 builds a paired hybrid sample:

```text
clean_full:
  input  = clean GT object sequence
  labels = clean GT object sequence

noisy_full:
  input  = bbox-wise coordinate-corrupted object sequence
  labels = clean GT object sequence
```

The CE objective is hard clean-label SFT CE only. No valid-set marginal CE,
multi-positive trie target, coordinate SoftCE, or prior recursive detection
feature should be activated in the V1 CE branch. Sparse asymmetric KL is added
only at selected objects' coordinate sites when
`prefix_denoising.current_object_kl.weight > 0.0`.

## Locked Decisions

| Area | Decision |
|---|---|
| Training surface | V1 targets Stage-1 compact detection teacher forcing first. Do not change Stage-2 rollout-correction behavior. |
| CE objective | Use hard clean-label SFT CE for both `clean_full` and `noisy_full`. Do not activate marginal, multi-positive, trie, SoftCE, or other previous objective features. |
| KL objective | Use asymmetric local-window coord KL: `stopgrad(clean_full) -> noisy_full`. |
| Branches | Build exactly two full multimodal segments per hybrid sample: `clean_full` and `noisy_full`. KL metadata is not a third segment. |
| Noise | Corrupt bbox coordinates bbox-wise in norm1000 integer `xyxy` space. Require valid boxes and all four quantized coordinates changed for every noisy object. |
| Object order | Require deterministic clean-GT sorted order. Reject inherited random/permuted ordering when prefix denoising is enabled. |
| Packing | Production-style V1 launch-health uses packed training. Do not relax existing teacher-forcing/recursive packing guards globally; add a narrow V1 hybrid-packed path. |
| Cache | Disable encoded-sample cache for V1. Static packing may cache a length/pack plan because planned hybrid lengths are epoch-invariant. |
| Visual inputs | Treat clean/noisy segments as separate multimodal segments with separate visual ownership, even when they reference the same image. |
| Decode | Keep free autoregressive decode untouched. Do not add logits constraints or rollout/free-decode eval in the first implementation slice. |
| Model | Do not edit upstream HF/Qwen model files. Production training starts from `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`. |

## Non-Goals

- Do not implement rollout or free-decode evaluation in V1 implementation.
- Do not add invalid, malformed, hard negative, text, class, or schema
  corruption.
- Do not add a clean-baseline config. The clean baseline is an external
  comparison artifact.
- Do not add user-facing sampler knobs beyond
  `prefix_denoising.current_object_kl.num_objects_per_image`.
- Do not add a current-object KL `enabled` boolean. `weight = 0.0` disables KL.
- Do not introduce temperature as a V1 config field. Local KL temperature is
  fixed at `1.0`.
- Do not enable encoded-sample cache for dynamic noise or epoch-varying KL site
  selection.
- Do not use global `logits_to_keep` for this objective; CE and KL diagnostics
  need full relevant logits.

## Architecture

V1 is a separate prefix-denoising path layered beside the existing Stage-1
teacher-forcing route. The existing route supplies prompt/template/tokenizer
contracts and dataset/runtime integration, but V1 should not inherit the
current `pure_valid_set_marginal` random-permutation target semantics.

The implementation is organized as contract-first layers:

```text
typed config and runtime gates
  -> constructive bbox noiser
  -> hybrid sample and segment builder
  -> hard CE objective and metrics
  -> sparse local-window KL objective and diagnostics
  -> hybrid-aware packed collation and boundary rewriting
  -> tiny packed launch-health configs
```

This layering lets unit tests inspect unpacked objects and tensors locally, but
all named launch-health and smoke configs should exercise the intended packed
path.

## Component Design

### 1. Config Schema And Runtime Gates

Add an optional top-level `prefix_denoising` section to
`DetectionTrainingConfig`:

```yaml
prefix_denoising:
  enabled: true
  noise:
    center_shift_frac: 0.08
    uniform_scale_range: [0.92, 1.08]
  current_object_kl:
    weight: 0.05
    window_radius: 8
    num_objects_per_image: 1
```

Disabled semantics:

- omitted `prefix_denoising` or `enabled: false` means ordinary clean teacher
  forcing;
- `enabled: true` with `prefix_denoising.current_object_kl.weight: 0.0` means
  CE-only paired denoising;
- `enabled: true` with positive KL weight means CE plus sparse KL.

Validation requirements:

- `detection_template.id: compact_full`;
- `detection_template.coordinate_surface: coord_token`;
- `detection_template.bbox_format: xyxy`;
- `data.object_ordering: sorted`;
- hard clean-label CE profile for V1, not valid-set marginal CE;
- `training.encoded_sample_cache.enabled: false`;
- `training.use_logits_to_keep` absent or false;
- no non-V1 teacher-forcing packing escape hatch;
- unknown nested keys rejected by strict dataclass parsing.

The runtime gate should leave existing teacher-forcing/recursive rejection
paths intact for non-V1 configs. Prefix-denoising gets a narrow positive
eligibility branch only when the hybrid builder, collator, and boundary map are
active.

### 2. Constructive Bbox Noiser

Add bbox noising helpers under `src/datasets/geometry.py`, keeping geometry math
outside trainer code.

The helper operates on norm1000 integer `xyxy` boxes and returns a structured
result containing:

- clean bbox and noisy bbox;
- clean and noisy coord bins;
- per-coordinate changed flags;
- validity status;
- skip or infeasible reason;
- sampled center/scale parameters;
- RNG/provenance payload.

The helper must construct from the feasible set of valid boxes:

```text
0 <= x1' < x2' <= 999
0 <= y1' < y2' <= 999
x1' != x1
y1' != y1
x2' != x2
y2' != y2
```

It should not repair invalid boxes after sampling. If the feasible set is empty
under the configured strength, the builder records an explicit infeasible
reason and excludes the whole hybrid sample during pre-plan eligibility.
Generated invalid boxes are exceptional and should produce warning counters.

### 3. Hybrid Data Model And Builder

Introduce a focused prefix-denoising module, for example
`src/detection/prefix_denoising.py`, with data objects equivalent to:

```text
HybridPrefixDenoisingSample
  base_sample_id
  hybrid_sample_id
  clean_full: PrefixDenoisingSegment
  noisy_full: PrefixDenoisingSegment
  noisy_object_map
  selected_kl_sites
  skip/provenance metadata

PrefixDenoisingSegment
  segment_id
  branch_id: clean_full | noisy_full
  input_ids
  labels
  attention/input visual payload references
  supervised positions
  CE denominator
```

The builder should:

- load the normal offline-prepared JSONL/image row;
- resolve sorted clean-GT object order;
- build one clean object map;
- build one noisy object map by applying the shared noiser to every valid
  nondegenerate object bbox;
- render two complete conversations with identical length and clean labels;
- ensure only bbox coordinate input ids differ in `noisy_full`;
- attach selected KL site metadata only when KL weight is positive;
- exclude whole hybrid samples before packing when overlength or deterministic
  noise infeasibility is known.

The builder should not emit clean/noisy segments as unrelated dataset rows.
They are one semantic unit because CE branch balance, noising provenance, KL
site selection, and packing eligibility are all per hybrid sample.

### 4. Hard CE Objective

V1 CE is hard clean-label CE over supervised assistant-response tokens:

```text
L_full =
    0.5 * CE(clean_full_input, clean_labels)
  + 0.5 * CE(noisy_full_input, clean_labels)
```

This branch supervises descriptions, bbox markers, coordinate tokens,
separators, schema/control tokens, and assistant stop tokens in both views.
Prompt and non-response context labels remain masked as in standard SFT.

The optimized CE scalar is branch-balanced, not token-pooled. A separate
token-pooled monitor may be logged, but it must not define the training loss
scale when branch denominators differ.

### 5. Sparse Local-Window KL

When `prefix_denoising.current_object_kl.weight > 0.0`, select
`K_i = min(num_objects_per_image, object_count_i)` objects per image using a
deterministic internal sampler. The sampler can vary by epoch, but it is not a
user-facing knob and does not change text-token length.

For each selected object, compute KL at the four coordinate sites:

```text
x1, y1, x2, y2
```

Use the clean GT bin to define the clipped support:

```text
support(c, r) = [max(0, c-r), ..., min(999, c+r)]
```

The local KL uses fp32 softmax/log-softmax over the clipped support:

```text
KL(stopgrad(clean_full local coord distribution)
   || noisy_full local coord distribution)
```

The KL term is averaged over selected objects and candidate sites. Increasing
`K` increases coverage and compute, not effective KL weight scale.

Required diagnostics include raw KL, weighted KL, candidate site count,
effective site count, identical-prefix site count, support-bin count,
edge-truncation count, teacher/student support mass, full coordinate-vocab
GT-bin probability, local conditional GT-bin probability, top-1-is-GT, and
teacher-minus-student deltas by slot.

### 6. Metrics

Prefix-denoising metrics should use the typed `MetricEvent` path but publish
distinct flat keys with branch/view encoded in the key. Do not emit the same
flat metric key with only different `MetricEvent.channel` values.

Required monitor families:

- standard `llm_loss`;
- global full-vocab token top-1 and top-5 accuracy;
- clean/noisy branch CE;
- branch-balanced CE;
- optional token-pooled CE monitor;
- raw and weighted local-window KL;
- KL support and teacher-quality diagnostics;
- skip/infeasible counters;
- selected-object and site-count counters.

The design should preserve existing dashboard continuity while adding enough
branch-level structure to catch noisy CE collapse, KL starvation, or misleading
local-window renormalization.

### 7. Hybrid Packing

The atomic packing unit is one `HybridPrefixDenoisingSample`, measured by:

```text
len(clean_full segment) + len(noisy_full segment)
```

Multiple hybrid samples may be packed into one physical row, but one hybrid
sample must never be split across rows. If the next hybrid sample cannot fit
fully in the current pack, it moves to the next pack. If one hybrid sample
exceeds `global_max_length`, it is excluded before the static pack plan and
logged as `overlength_hybrid_sample`.

Introduce `PackedHybridBoundaryMap` or an equivalent structure with:

- physical packed row index;
- hybrid sample id;
- segment id and branch id;
- segment token start/end offsets;
- supervised-label start/end offsets;
- sidecar local-to-packed offsets;
- position-id reset boundaries;
- FlashAttention varlen boundaries when padding-free packed mode is used;
- image placeholder span ownership;
- `pixel_values` and `image_grid_thw` slice ownership;
- CE and KL denominator contribution.

The boundary map is the contract for rewriting CE positions, KL sites, metric
denominators, and visual ownership after packing. A plain 2D attention mask is
not enough for sidecar-active packed teacher forcing.

### 8. Config Leaves And Launch Health

Add new prefix-denoising config leaves rather than mutating the existing
canonical baseline config. The current canonical Stage-1 teacher-forcing config
uses random object ordering, so V1 leaves must explicitly set:

```yaml
data:
  object_ordering: sorted
```

The first runnable ladder is:

```text
packed CE-only denoising:
  prefix_denoising.current_object_kl.weight = 0.0

packed CE plus KL:
  prefix_denoising.current_object_kl.weight = 0.05
```

Tiny smoke runs may use a well-tuned pure-CE adapter only for numerical wiring
and launch-health. Production or quality-bearing training starts from the
model-cache coord 2B full checkpoint.

## Data Flow

```text
raw JSONL row + image
  -> sorted clean DetectionScene / object map
  -> feasible bbox noising for every valid object
  -> clean_full rendered conversation and labels
  -> noisy_full rendered conversation and clean labels
  -> optional selected KL site metadata
  -> HybridPrefixDenoisingSample eligibility filter
  -> static hybrid pack plan
  -> packed model forward with isolated segments
  -> hard CE over clean/noisy labels
  -> optional local-window KL from clean logits to noisy logits
  -> typed metric events and standard training monitors
```

## Error Handling

Fail fast for contract violations:

- unsupported template, coordinate surface, bbox format, or object order;
- unexpected nested config keys;
- enabled encoded-sample cache;
- logits pruning;
- non-V1 packing path with sidecar-active teacher forcing;
- mismatched clean/noisy segment lengths;
- label mismatch between clean/noisy branches;
- non-coordinate token differences in `noisy_full`;
- KL support missing the clean GT bin;
- non-finite CE, KL, probabilities, or metrics;
- planned hybrid sample skipped at fetch time;
- visual slice or placeholder ownership mismatch after packing.

Skip with explicit counters only for deterministic data eligibility cases:

- degenerate clean GT bbox;
- infeasible valid 4-coordinate bbox perturbation;
- overlength hybrid sample before pack planning.

Invalid bbox generation after feasible-set construction is treated as a bug or
infeasible configuration, not a normal repair path.

## Testing Strategy

The implementation plan should be test-first and organized around these gates:

1. Config/schema tests for enabled, disabled, CE-only, KL-on, sorted-order
   validation, unknown-key rejection, encoded-cache rejection, and packing guard
   preservation.
2. Geometry tests for valid constructive noising, boundary boxes, tiny boxes,
   full-image boxes, deterministic seeds, feasible-set construction, and
   `4/4` coordinate-bin changes.
3. Hybrid builder tests for two segments only, stable lengths, clean labels in
   both branches, coordinate-only input differences, object order preservation,
   selected KL-site metadata, and deterministic pre-plan exclusions.
4. Hard CE tests proving branch-balanced CE equals
   `0.5 * CE_clean + 0.5 * CE_noisy` and that token-pooled CE is diagnostic
   only.
5. KL numerics tests for support clipping near `0` and `999`, fp32 local
   softmax, stopgrad teacher distribution, student gradient path, denominator
   accounting, first-object `x1` identical-prefix behavior, and finite metrics.
6. Metric tests proving flat prefix-denoising keys do not collide and preserve
   `llm_loss`, top-1, and top-5 monitors.
7. Packing tests for atomic hybrid sample placement, overlength exclusion,
   boundary-map offsets, position-id resets, visual slice ownership, CE
   positions, KL positions, and metric denominators.
8. Epoch tests proving selected KL objects and noisy coordinate bins may vary
   while hybrid sample length, planned ids, and skip decisions remain invariant.
9. Tiny packed smoke tests for CE-only and KL-on launch-health, using adapter
   checkpoints only for wiring and numerical sanity.

## Acceptance Criteria

V1 implementation is ready for launch-health only when:

- omitted/disabled prefix-denoising still runs ordinary teacher forcing;
- CE-only denoising produces clean/noisy hard-CE branch metrics without KL
  sidecar cost;
- KL-on denoising produces finite raw/weighted KL and teacher-quality metrics;
- packed CE and KL sidecar positions survive boundary rewriting;
- standard `llm_loss`, token top-1, and token top-5 monitors are still emitted;
- skip counters and noising-difficulty reports are present;
- encoded-sample cache is disabled or rejected for V1;
- non-V1 sidecar packing guardrails still hold;
- tiny packed CE-only smoke passes before tiny packed KL-on smoke;
- no result is interpreted as an exposure-bias or rollout improvement without a
  later matched rollout/free-decode evaluation.

## Open Implementation Details

No blocking research-design decision remains after approving hard clean-label
CE. The implementation plan still needs to choose exact class/function names
and the smallest code insertion points after inspecting current symbols, but
those are engineering details rather than research-policy choices.

## User Review Gate

After this spec is written, the user should review it before the implementation
plan is created. The next Superpowers step is `superpowers:writing-plans`, which
will turn this design into a task-by-task implementation roadmap with exact
files, failing tests, implementation steps, and verification commands.
