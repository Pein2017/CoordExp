---
doc_id: progress.directions.prefix-denoising-sft-v1
layer: progress
doc_type: direction
status: active-draft
domain: training
summary: Infra-aware hybrid V1 research direction for prefix-denoising SFT on Stage-1 compact detection teacher forcing.
tags: [progress, directions, stage1, compact-full, teacher-forcing, prefix-denoising, hybrid-denoising, sparse-kl]
updated: 2026-06-14
---

# Prefix-Denoising SFT V1: Infra-Aware Hybrid

## Decision

Contract status: this is an active research direction with an initial runtime
implementation on the `codex/prefix-denoising-sft` worktree. The
`prefix_denoising` YAML surface, hard CE objective, sparse local coordinate KL,
packed hybrid boundary map, packed KL-site rewrite, encoded-cache guardrails,
and launch-health config leaves are implemented for the Stage-1 compact
teacher-forcing route. This note remains the research rationale and launch
checklist rather than a stable cross-surface contract.

V1 targets Stage-1 compact detection teacher forcing first, under the current
`stage1_detection_teacher_forcing` route. The names and interfaces should stay
general enough for a later Stage-2 rollout-prefix extension, but V1 should not
start by changing the active Stage-2 rollout-correction surface.

Primary quality-bearing training should start from the model-cache 2B coord
full checkpoint, not from a well-tuned pure-CE adapter checkpoint. The
production starting point is:

```text
model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
```

That checkpoint already provides the coordinate-capable model architecture,
token machinery, and initialized coordinate-offset state. The first production
trainable boundary is coord offset plus LLM-tower DoRA through the existing
ms-swift/PEFT tuner path. ViT and aligner should stay frozen for the first
quality-bearing denoising run so the result reads as coordinate-prefix
robustness, not a change in visual feature extraction or connector behavior.

V1 requires coord-token compact detection mode. It should fail fast if the
active data/template route cannot represent each bbox coordinate as the expected
single coordinate-token/bin. There should be no fallback to numeric
text-coordinate corruption in the first implementation; that would make
clean/noisy same-length replacement and packed sidecar alignment harder to
reason about.

Well-tuned pure-CE adapters may still be used for tiny launch-health runs and
numerical verification of the KL term. Those runs are for sidecar alignment,
finite/scaled KL, gradient direction, and metric denominators. They are not the
production baseline and must be labeled as `tiny` or `launch-health`.

## Core Revision

The previous all-object A/B denoising design is semantically clean but too
expensive as the default training surface. V1 should not expand every object in
every image into separate A/B KL probes by default. That expansion repeats
image/prompt tokens and historical object prefixes many times. Padding-free
packing can reduce padding waste, but it cannot remove the repeated-prefix
attention cost.

The default V1 surface is therefore a hybrid:

1. `full_sequence_denoising_ce`: an efficient full-sequence CE branch.
2. `sparse_current_object_kl`: sparse clean-to-noisy prefix KL on selected
   object coordinate sites.

The full-sequence branch carries broad CE coverage. The KL branch is a
selected-site consistency regularizer: it uses `clean_full` as the teacher and
`noisy_full` as the student. It should not create extra KL segments in the
V1 default.

## Branch A: Full-Sequence Denoising CE

For each image / GT object sequence, construct two full-sequence views:

```text
clean_full:
  input_ids = clean GT object sequence
  labels    = clean GT object sequence

noisy_full:
  input_ids = coordinate-corrupted GT object sequence
  labels    = clean GT object sequence
```

Prompt and non-response context labels remain masked as in ordinary SFT. Within
the assistant response, all clean target tokens can be CE-supervised in both
views: descriptions, bbox markers, coordinate tokens, separators, and terminal
tokens. In V1, class/description/control/delimiter tokens remain clean in the
`noisy_full` input; only bbox coordinate tokens are corrupted. For simplicity
and unification, `noisy_full` corrupts every valid nondegenerate object bbox in
the sequence. V1 should not add a separate `noisy_full` object-corruption
probability, maximum count, or density/sampling knob. Clean-prefix exposure is
provided by `clean_full`, and noisy-prefix difficulty is controlled by the
shared bbox-noise strength knobs.

This means V1 consolidates format and schema requirements in both views. The
clean and noisy branches both supervise clean class/description text, structural
special tokens, bbox markers, object separators, and assistant stop tokens with
ordinary teacher-forced CE. The noisy branch is not a format-corruption branch:
its non-coordinate input tokens stay clean, while its coordinate input tokens
are bbox-wise perturbed and its labels remain the clean full response.

`noisy_full` uses the ordinary assistant-response CE mask. It supervises every
clean response token, including coordinate positions such as `x1` whose causal
prefix has not yet seen the current object's noisy coordinates. The
causal-difference filter belongs to sparse current-object KL, not the
full-sequence CE branch.

This branch keeps the core denoising philosophy:

```text
noisy coordinate prefix -> clean GT continuation
```

For example, when predicting `object_k.x2` in `noisy_full`, the causal prefix
may contain corrupted boxes for previous objects and corrupted `object_k.x1`
and `object_k.y1`, while the target label remains the clean `object_k.x2`.
When predicting tokens after the object bbox, the noisy complete bbox remains
in the causal prefix. This trains recovery from wrong-but-valid coordinate
history without requiring one A/B KL probe per object.

The full-sequence CE branch should be normalized so that adding the noisy view
does not silently double the ordinary SFT loss scale. The working V1 loss is:

```text
L_full_sequence_denoising =
    0.5 * CE(clean_full_input, clean_labels)
  + 0.5 * CE(noisy_full_input, clean_labels)
```

The CE denominator is supervised clean-label tokens, not raw physical tokens or
packed row length. The clean/noisy full-sequence CE terms use an equal
branch-level average, not one pooled CE over both branches' supervised tokens.
This keeps clean-prefix competence and noisy-prefix recovery balanced even if
future filtering or skip handling makes the two branches differ in token count.

## Branch B: Sparse Clean-To-Noisy Prefix KL

V1 default uses the existing full-sequence views for KL. For each base image and
epoch, sample up to `K` object indices. The teacher is `clean_full`; the student
is `noisy_full`:

```text
KL teacher distribution:
  clean_full logits

KL student distribution:
  noisy_full logits
```

For each selected object `k`, KL compares the two views at that object's
coordinate prediction sites. The teacher and student share:

- same image;
- same clean-GT-derived object order;
- same current object index `k`;
- same clean target coordinate bins and KL support;
- same metadata/provenance seed lineage.

The intended difference is the whole denoising condition:

```text
teacher prefix = clean previous prefix + clean current-object prefix
student prefix = noisy previous prefix + noisy current-object prefix
```

This is less mechanistically isolated than the previous same-noisy-history
current-object teacher idea, but it better matches the V1 philosophy: under
wrong, slightly wrong, or drifted prefixes, make the model behave more like it
does under clean teacher-forced prefixes. The same-noisy-history current-object
teacher remains a future diagnostic ablation, not the V1 default.

The sparse KL branch does not own broad full-continuation CE. `clean_full` and
`noisy_full` already provide CE coverage. KL only adds selected-object
distribution matching on coordinate sites. V1 should not apply KL to schema,
description, separator, marker, or stop-token sites; ordinary CE in both views
is the format/schema consolidation path.

## Sparse Current-Object KL

The KL direction is asymmetric:

```text
KL(
  stopgrad(clean_full local coord distribution)
  || noisy_full local coord distribution
)
```

KL applies at selected objects' coordinate prediction sites whose causal
conditioning prefix differs between clean and noisy full-sequence views. In the
standard `x1 y1 x2 y2` order, the previous prefix already differs before `x1`
for any selected object after the first, and the current-object prefix differs
from `y1` onward. For V1 simplicity and uniform selected-object accounting, the
canonical KL sites are all four coordinate sites:

```text
x1, y1, x2, y2
```

The KL support is a local coordinate-token window centered on the clean GT
coordinate bin for that prediction site. View A and View B distributions are
restricted to that support and renormalized before KL. The support must include
the clean GT coordinate bin. Softmax/log-softmax for this local KL should run in
fp32 and assert finite probabilities.

Support bins are clipped to the coordinate vocabulary:

```text
support(c, r) = [max(0, c-r), ..., min(999, c+r)]
```

The support never wraps, pads, mirrors, or includes invalid coordinate-token
bins. It must be non-empty, unique, and include the clean GT bin. Launch-health
metrics should log the actual support-bin count and edge-truncation count/rate.

Support values are coordinate bins, not full-vocabulary token ids. The KL
implementation must map each support bin through the tokenizer's coord-token id
row before indexing logits. Full coordinate-vocab mass diagnostics must sum over
all coord-token ids, not raw vocab columns `0..999`.

All CE, token-accuracy, and KL label sites use causal-LM alignment:
`labels[position]` is predicted by `logits[position - 1]`. Physical position
`0` and each packed segment's first token must remain unsupervised or be
excluded from loss/metric denominators.

For uniform V1 accounting, all selected objects contribute four candidate KL
sites: `x1`, `y1`, `x2`, and `y2`. If the first selected object has an identical
clean/noisy causal prefix at `x1`, that site remains in the KL denominator and
should contribute zero or near-zero KL. Log `candidate_kl_site_count`,
`effective_kl_site_count`, and `identical_prefix_site_count` so this choice is
visible instead of silently changing the scale.

Accepted risk: the teacher is not an oracle. The clean-prefix teacher is still
the same model under teacher forcing, not ground truth logits. If the clean
teacher distribution is poor or overconfident, asymmetric KL can provide bad
guidance. V1 accepts this risk and relies on small KL weight, local GT-centered
support, CE anchoring, mild valid bbox noise, and launch-health diagnostics to
detect when the clean teacher is not locally healthier than the `noisy_full`
student.

Teacher-quality diagnostics must include both local-window conditional
statistics after renormalization and pre-renormalization mass statistics before
the window restriction. At minimum, report teacher/student support mass in the
GT-centered window, full coordinate-vocab GT-bin probability, local conditional
GT-bin probability, entropy or top-1 distance where available, and
teacher-minus-student deltas by `x1/y1/x2/y2`, object index/depth, and prefix
condition. KL should not be interpreted as meaningful if the clean teacher only
looks good after local-window renormalization while assigning little original
mass to the GT window.

The numeric KL knobs remain explicit config fields:

- `prefix_denoising.current_object_kl.weight`;
- `prefix_denoising.current_object_kl.window_radius`;
- `prefix_denoising.current_object_kl.num_objects_per_image`;

V1 should not expose a separate current-object sampler knob. Current object
selection is internally fixed to deterministic seeded uniform selection or a
deterministic per-image permutation/cycle over the clean-GT ordered object
indices, and it may rotate selected objects by epoch while preserving length
invariance. The only user-facing coverage knob is
`prefix_denoising.current_object_kl.num_objects_per_image`. The first default is
`1`. Larger values are coverage/throughput ablations after the `K=1` path is
numerically sane.

The first KL weight default is `prefix_denoising.current_object_kl.weight =
0.05`. The initial lambda ladder should be `[0.00, 0.05, 0.10]` after smoke
verification. Larger values such as `0.20` should wait until `0.10` is clearly
stable and too weak. The first default temperature is
hard-coded at `1.0`. Because the KL support is already local and GT-centered,
temperature smoothing such as `2.0` should be treated as a follow-up numeric
ablation if the raw local KL is too spiky or unstable, not a V1 config field.
`prefix_denoising.current_object_kl.window_radius` remains configurable because
it defines the local coordinate-token support. The default is `8`, with a small
interpretable support-radius ladder such as `[4, 8, 16]` if support sensitivity
needs to be studied.

## Objective

The V1 objective is:

```text
L_total =
    L_full_sequence_denoising
  + lambda_current_object_kl * L_sparse_current_object_KL
```

where:

```text
L_full_sequence_denoising =
    0.5 * CE(clean_full_input, clean_labels)
  + 0.5 * CE(noisy_full_input, clean_labels)
```

and:

```text
L_sparse_current_object_KL =
    mean over sampled current objects k and eligible coord sites:
        KL(
          stopgrad(local coord distribution from clean_full),
          local coord distribution from noisy_full
        )
```

The KL term is averaged over selected objects and eligible sites, not summed.
Increasing `prefix_denoising.current_object_kl.num_objects_per_image` therefore
increases coverage and compute, but should not automatically increase the
effective KL weight scale. In V1, increasing `K` does not add segments; it only
adds selected-object coordinate sites inside the same `clean_full` and
`noisy_full` forwards.

The current-object KL branch is not an all-object CE expansion. The clean-only
CE baseline already exists as a prior trained baseline artifact and does not
need to be recreated through a `prefix_denoising` config. The CE-only ablation
from the earlier all-object A/B design is superseded by an infra-aware
comparison ladder:

- existing clean full-sequence SFT baseline artifact;
- clean plus noisy full-sequence CE, no current-object KL;
- clean plus noisy full-sequence CE plus sparse current-object KL.

The clean baseline must be matched by base checkpoint, train/eval data, object
ordering, packing/cache policy, optimizer budget, route, and eval config before
it can support improvement claims. If the available clean baseline is an older
artifact with a different schedule, adapter state, ordering, or route, it should
be labeled `historical_anchor` and used only for orientation. The first
implementation ladder should still separate same-noise CE-only denoising from
CE+KL so any KL gain is not attributed to broad noisy CE by mistake.

V1 launch-health may establish data construction, packing, CE/KL numerics, and
teacher-forced synthetic valid-bbox prefix robustness. It must not be
interpreted as evidence that rollout exposure bias is solved. Exposure-bias
claims require a later matched free-decode or rollout evaluation of clean CE,
clean+noisy CE, and clean+noisy CE+KL, with parse/drop counters, duplicate or
guarded metrics, and self-prefix coordinate-locality probes preserving
`x1/y1/x2/y2` splits.

Optional full-clean-to-full-noisy coordinate-logit consistency may be studied
later at broader token scopes, but the primary V1 KL mechanism is already
clean-to-noisy full-sequence consistency at selected coordinate sites. Option A,
the same-noisy-history current-object teacher, remains a future diagnostic
ablation for isolating current-object prefix perturbation.

## Object Ordering

For V1, object ordering is deterministic and shared. Use the same object order
for `clean_full`, `noisy_full`, and every sampled current-object KL site group.
Compute the order from clean GT boxes and do not re-sort after perturbation.
The clean teacher logits and noisy student logits must index the same ordered
object positions.

V1 prefix-denoising configs must use the clean fixed order, such as
`data.object_ordering: sorted`, and must fail fast if they inherit
`random_permutation` or `random` from a nearby compact-full production config.
V1 should use the existing sorted-source invariant; it must not silently sort raw
rows after loading and must never re-sort after perturbing noisy boxes. Any
random or permuted object-order variant is a separately named ablation with a
matched baseline.

Keeping order fixed makes V1 about prefix denoising, not order augmentation.
Random or permuted object order can be a later named ablation.

## Noise Policy

Coordinate corruption should be valid and geometrically plausible. V1 uses
whole-bbox perturbation, not independent coordinate-token jitter:

- shift the bbox center relative to bbox width/height;
- apply mild width/height scaling;
- keep the result valid and inside the coordinate range;
- render the perturbed valid bbox back into coordinate tokens.

Class, description, schema, delimiter, and terminal tokens stay clean in the
first version. Invalid objects, malformed text, hard negative prefixes, and
format-breaking corruption are out of V1 scope.

The geometry helper should live under `src/datasets/geometry.py`, matching the
repo rule that bbox math routes through the dataset geometry module. It should
stay unaware of View A/B, KL, branch ids, and metrics. Training-specific code
owns branch construction, labels, CE masks, KL site selection, and counters.

The bbox noise maker should return a structured result, not only a new bbox:

- clean bbox and noisy bbox;
- clean and noisy coord tokens/bins;
- per-coordinate changed flags;
- validity status and skip reason;
- perturbation parameters;
- RNG/provenance payload.

The V1 bbox-noise helper operates in norm1000 integer `xyxy` coordinate-bin
space. For each clean bbox, it first computes a deterministic feasible set of
noisy integer bboxes satisfying:

```text
0 <= x1' < x2' <= 999
0 <= y1' < y2' <= 999
x1' != x1
y1' != y1
x2' != x2
y2' != y2
```

The feasible set must also respect the configured center-shift/scale envelope,
including any explicit minimum one-bin movement rule needed to make quantized
movement meaningful. If the feasible set is empty, return a structured
infeasible result during preflight; do not sample-and-retry in the training hot
path. If it is nonempty, sample directly from that feasible set with the
sample/epoch RNG. Do not rely on post-hoc clamp/repair to turn an invalid
sampled box into a valid one.

The earlier `4/4 changed` token rule remains useful for `noisy_full`, because a
no-op current perturbation silently removes KL meaning.
For the full-sequence noisy branch, all valid nondegenerate object bboxes are
corrupted with the shared bbox-noise maker. The full branch should not expose a
separate object-level corruption probability; the first V1 surface controls
noise only through a small set of strength/level knobs such as center shift and
uniform scale range.

The same `4/4` changed-token contract applies to the entire noisy object map,
not only to selected KL sites. For every valid nondegenerate object that enters
`noisy_full`, the noisy bbox must remain valid and all four quantized coordinate
tokens must differ from the clean bbox:

```text
x1' != x1
y1' != y1
x2' != x2
y2' != y2
```

If this is infeasible for an object under the configured noise strength, the
affected hybrid sample should be skipped or counted explicitly rather than
silently emitting a partial/no-op corruption.

Because the bbox-noise maker is intended to be valid by construction, generated
invalid bboxes are exceptional. If any object needed by a hybrid sample cannot
produce valid `4/4` noise, skip the entire hybrid sample, increment explicit
skip counters, and emit warning-level diagnostics. Expected counters include
`degenerate_gt_bbox`, `noise_infeasible_valid_bbox`, and
`noise_infeasible_4coord_changed`. Repeated invalid-bbox warnings should be
treated as evidence of a noise-maker bug or infeasible configuration, not as a
normal repair path.

Full-sequence `noisy_full` and selected-object KL sites should share the same V1
noise-strength defaults. The first design should not make KL easier by silently
using a smaller current-object perturbation. If local-window KL is too unstable,
the first response should be an explicit KL numeric
ablation, such as lower `prefix_denoising.current_object_kl.weight` or larger
`prefix_denoising.current_object_kl.window_radius`, not a hidden
branch-specific noise regime. Temperature `2.0` may be reopened later as a
named numeric ablation if needed.

The first smoke/default noise profile remains
`prefix_denoising.noise.center_shift_frac = 0.08` and
`prefix_denoising.noise.uniform_scale_range = [0.92, 1.08]` even though
`noisy_full` corrupts every valid nondegenerate object bbox. This preserves
comparability with the earlier noise-strength assumptions. If tiny smoke
evidence shows excessive skip rates, invalid-bbox warnings, a severe
noisy-full CE gap, or token-accuracy collapse, the first named milder fallback
is `prefix_denoising.noise.center_shift_frac = 0.04` and
`prefix_denoising.noise.uniform_scale_range = [0.96, 1.04]`.

The V1 config should be one combined `prefix_denoising` group with nested
subsections, not unrelated top-level groups:

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

This keeps the strategy discoverable as one feature while preserving the
separation between full-sequence CE, sparse selected-object KL sites, and the
shared noisy object map. Full-sequence clean/noisy CE is implicit whenever
`prefix_denoising.enabled = true`; V1 should not expose a
`prefix_denoising.full_sequence.enabled` toggle. Disabling the whole feature
uses the existing trained clean CE baseline rather than a new V1 config mode.
Disabled modes are intentionally two-level:

- omitted `prefix_denoising` or `prefix_denoising.enabled: false` means ordinary
  clean teacher forcing with no prefix-denoising branches, sidecars, or metrics;
- `prefix_denoising.enabled: true` with `current_object_kl.weight: 0.0` is the
  CE-only denoising ablation;
- `prefix_denoising.enabled: true` with positive `current_object_kl.weight`
  enables the KL sidecar path.
The current-object KL path should not have a separate `enabled` boolean:
`prefix_denoising.current_object_kl.weight = 0.0` disables KL and gives the
clean/noisy full-sequence CE-only denoising ablation; positive weight enables
the KL contribution. When the weight is `0.0`, the builder should not construct
KL site sidecars at all; the CE-only denoising ablation should not pay KL
metadata or loss cost.
The KL direction and support are algorithmic invariants, not config knobs:
teacher logits use `stopgrad(clean_full) -> noisy_full`, and support is always
the local coordinate-token window around the clean GT bin. V1 should not expose
`teacher_stopgrad` or `coord_only` fields.

## Data Construction

For each base image, the hybrid data builder should conceptually produce:

```text
1. clean_full sequence
2. noisy_full sequence
3. zero or more selected-object KL site groups, bounded by K
```

Each base image/epoch should first produce one clean object map and one noisy
object map. The noisy object map is generated by applying the shared valid
bbox-noise maker to every valid nondegenerate object. All hybrid branches reuse
these maps:

```text
clean_full:
  input  = clean object map
  labels = clean object map

noisy_full:
  input  = noisy object map
  labels = clean object map

KL for selected object k:
  teacher distribution = clean_full logits at object k x1/y1/x2/y2
  student distribution = noisy_full logits at object k x1/y1/x2/y2
  support              = GT-centered local coordinate-token windows
```

Current-object KL site selection should not independently resample noise in V1.
Reusing the same per-object noisy bbox map keeps
full-sequence CE and current-object KL tied to the same perturbation regime and
simplifies metadata, debugging, and launch-health interpretation.

The builder must not select every object for KL by default. Selected object
count is bounded by `prefix_denoising.current_object_kl.num_objects_per_image`.
For image `i`, `K_i = min(K, object_count_i)` object site groups are created
when KL is enabled. The internal current-object sampler draws `K_i`
distinct object indices uniformly or from a deterministic per-image permutation
over the clean-GT ordered object sequence. The sampler may vary selected object
indices by epoch to improve object-level KL coverage. This does not change
static-packing length because selected objects are metadata/site choices, not
extra segments. The sampler policy is not a V1 config knob.

Changing selected KL objects or noisy coordinate token ids across epochs changes
encoded content but not text-token length. Encoded sample caching for
prefix-denoising must therefore be disabled or made epoch-aware. Static packing
can still use a stable pack plan because the planned length stays invariant.

The current object index `0` is valid. Single-object images remain usable for
current-object KL because they still provide noisy-current-prefix sites after
the first coordinate. For uniform V1 accounting, selected objects expose
`x1/y1/x2/y2` KL sites; `x1` for the first object may have identical causal
prefixes and therefore contribute little or zero KL.

The semantic data unit should be one `HybridPrefixDenoisingSample`, not a set of
unrelated rows. This container owns the base sample id, image reference, clean
object map, noisy object map, noising provenance, skip policy, branch-balance
accounting, and every segment derived from that base image. Its members are
attention-isolated `PrefixDenoisingSegment`s:

```text
HybridPrefixDenoisingSample
  clean_full segment
  noisy_full segment
  optional selected-object KL site metadata when current_object_kl.weight > 0
```

`clean_full` and `noisy_full` are a paired semantic container because they share
the same clean target, noisy object map, and branch-level CE balance. They should
not be emitted as unrelated dataset rows that could lose provenance or silently
change the intended `0.5 * CE_clean + 0.5 * CE_noisy` scale. When
`prefix_denoising.current_object_kl.weight > 0.0`, selected-object KL site
metadata should be attached to the same hybrid container. When the weight is
`0.0`, KL site metadata should not be built.

Each segment remains a complete teacher-forced multimodal conversation with its
own tokenized text, labels, image placeholders, and local sidecar metadata. The
container gives implementation code one place to enforce that paired views have
identical token lengths, object order, clean-label targets, and visual payload
ownership before they reach the packer.

## Metadata

The hybrid design needs metadata that separates CE coverage from current-object KL
alignment. Required metadata should include:

- `base_sample_id`;
- `hybrid_sample_id`;
- `segment_id`;
- segment `branch_id`: `clean_full` or `noisy_full`;
- loss/metric `channel`: `sparse_current_object_kl` for selected KL site groups;
- `view_id`: `clean_full`, `noisy_full`;
- local segment token length and supervised-label span;
- clean-GT object order and ordered object ids;
- full-branch noisy object map;
- per-object clean/noisy bbox tokens and changed flags;
- sampled KL current object indices;
- `current_object_index`;
- `history_object_count`;
- KL-eligible coordinate slots;
- clean GT bins for KL sites;
- local-window support radius and support bins;
- noising seed and perturbation parameters;
- sampler seed, epoch, and per-image object permutation/cycle state;
- skip reason counters;
- CE supervised-token denominators;
- KL site denominators.

This metadata is not only for debugging. It is what makes later ablations
interpretable: full-sequence CE may look healthy while current-object KL is
starved, or current-object KL may look healthy while noisy-full CE is too hard.
`sparse_current_object_kl` is never a third `PrefixDenoisingSegment` and never a
third attention segment. It is a loss/metric/site channel attached to paired
positions in the existing `clean_full` and `noisy_full` segments.

## Metrics

Metrics must preserve the standard training-health monitors and add
denoising-specific splits. The standard `llm_loss` monitor and token top-1/top-5
accuracy monitors must remain visible to existing dashboards; prefix-denoising
metrics should add structure without replacing those monitors.

- `llm_loss` over supervised clean labels;
- top-1 token accuracy over supervised clean labels;
- top-5 token accuracy over supervised clean labels;
- clean-full CE;
- noisy-full CE;
- full-branch aggregate CE;
- raw sparse current-object KL;
- weighted sparse current-object KL;
- KL eligible site count;
- KL support/window diagnostics;
- teacher/student local-window GT-bin probability diagnostics;
- skip counters by reason.

Global `llm_loss`, token top-1 accuracy, and token top-5 accuracy must be
computed from supervised-token accounting, not as an unweighted mean of
branch/view metrics. Branch/view splits should be logged separately so the
aggregate cannot hide that noisy-full CE is failing or that current-object KL
has no eligible sites.

Where the existing semantic token accounting makes it cheap, launch-health
metrics should also retain branch/view splits for schema, description,
coordinate, object-control, separator, and stop-token groups. These splits are
diagnostic only; they should not introduce schema-specific loss weighting in V1.
The first experiment should rely on ordinary CE in both branches to consolidate
format requirements.

Metrics should use the repo's typed metric-event path rather than ad hoc scalar
side channels, aligned with `docs/training/METRICS.md` and
`src/metrics/events.py`. Under the current flattener, V1 should materialize the
branch/view channel in the metric key itself or first change the flattener with
tests. Do not emit the same flat key with only different `MetricEvent.channel`
values, because current metric reduction treats that as an identity collision.
Candidate V1 keys are:

- `prefix_denoising/global/loss/ce_balanced`;
- `prefix_denoising/global/loss/ce_token_pooled`;
- `prefix_denoising/clean_full/loss/ce`;
- `prefix_denoising/noisy_full/loss/ce`;
- `prefix_denoising/global/token_acc/full_vocab/top1`;
- `prefix_denoising/global/token_acc/full_vocab/top5`;
- `prefix_denoising/clean_full/token_acc/full_vocab/top1`;
- `prefix_denoising/noisy_full/token_acc/full_vocab/top1`;
- `prefix_denoising/kl/local_window/raw`;
- `prefix_denoising/kl/local_window/weighted`;
- `prefix_denoising/kl/local_window/site_count`;
- `prefix_denoising/kl/local_window/candidate_site_count`;
- `prefix_denoising/kl/local_window/identical_prefix_site_count`;
- `prefix_denoising/kl/local_window/support_bin_count`;
- `prefix_denoising/kl/local_window/edge_truncated_count`;
- `prefix_denoising/kl/local_window/teacher_support_mass`;
- `prefix_denoising/kl/local_window/student_support_mass`;
- `prefix_denoising/kl/local_window/teacher_gt_prob_conditional`;
- `prefix_denoising/kl/local_window/student_gt_prob_conditional`;
- `prefix_denoising/kl/local_window/teacher_gt_prob_full_coord_vocab`;
- `prefix_denoising/kl/local_window/student_gt_prob_full_coord_vocab`;
- `prefix_denoising/kl/local_window/teacher_minus_student_gt_prob_conditional`;
- `prefix_denoising/kl/local_window/teacher_top1_is_gt`;
- `prefix_denoising/kl/local_window/student_top1_is_gt`.

Required count/skip diagnostics include:

- `prefix_denoising/skipped_hybrid_sample_count`;
- `prefix_denoising/noise_invalid_bbox_warning_count`;
- `prefix_denoising/noise_infeasible_4coord_changed_count`;
- `prefix_denoising/current_object_kl/selected_object_count`.

The optimized CE scalar is `prefix_denoising/global/loss/ce_balanced`, and it
must equal `0.5 * CE_clean_full + 0.5 * CE_noisy_full`. A token-pooled CE monitor
may also be logged as `prefix_denoising/global/loss/ce_token_pooled`. The
standard `llm_loss` dashboard monitor is the actual optimized scalar
backpropagated by the trainer: CE-only runs log the branch-balanced CE, and
KL-on runs log `CE_balanced + kl_weight * KL_raw`. Token-pooled CE must not be
used to verify the `0.5/0.5` objective scale when branch denominators differ.
Token accuracy metrics remain denominator-weighted over supervised clean-label
tokens.

The implementation should keep CE and KL separated in objective outputs and
metrics even though training uses one scalar total loss. `L_total` is
`L_full + lambda_current_object_kl * L_sparse_current_object_KL`, but launch
health must still expose clean CE, noisy CE, raw KL, weighted KL, teacher-quality
diagnostics, token top-1/top-5 accuracy, and the standard `llm_loss` monitor.

## Packing and Infra

The full-sequence branch is the throughput path. It is packing-friendly because
it behaves like ordinary SFT paired sequences: one clean sequence and one noisy
sequence per base example. KL does not add extra segments in the V1 default; it
adds selected coordinate sites inside the existing clean/noisy forwards.

The sparse current-object KL branch is bounded by `K`. It may still require
sidecar-safe packed target-position rewriting and attention-boundary metadata,
but its site count is intentionally capped. The implementation should not rely
on all-object KL expansion by default and then hope packing makes it cheap.

Current CoordExp compact teacher-forcing packing remains guarded for ordinary
teacher-forcing because precise target-position sidecar rewriting is not a
generic feature. That safety boundary remains important. V1 adds a narrow
positive eligibility path: `prefix_denoising.enabled: true` plus the
hybrid-aware packed builder/collator and `PackedHybridBoundaryMap` offset
rewrite. Teacher-forcing or recursive configs without this explicit V1 path
must continue to reject
`training.packing`, `training.eval_packing`, `packing.static_packing`,
`packing.padding_free_packed`, encoded-sample cache, and logits pruning as they
do today. The hybrid design changes the cost target: make full-sequence
denoising efficient first, then add bounded selected-object KL-site sidecars
with tests that prove pack offsets, labels, KL sites, and metrics survive
flattening.

The first authored launch-health leaves are:

- `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml`;
- `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml`;
- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`;
- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`.

For production-style V1 training, the first implementation target is static
hybrid pack planning under `training.packing: true`: the whole
`HybridPrefixDenoisingSample` is the atomic planning item, and the actual
collated packed batch must prove position reset, varlen attention boundaries,
and visual ownership through `PackedHybridBoundaryMap`. The separate
`packing.padding_free_packed` escape hatch remains out of V1. The algorithmic
design should not depend on all-object KL expansion as the packing unit. KL
sites are sparse auxiliary metadata, not extra CE data segments.

The V1 packed KL shape is:

```text
K = 0 / weight = 0:
  clean_full
  noisy_full

K = 1:
  clean_full
  noisy_full
  KL sites for object_k0: x1/y1/x2/y2

K = 2:
  clean_full
  noisy_full
  KL sites for object_k0: x1/y1/x2/y2
  KL sites for object_k1: x1/y1/x2/y2
```

In general, a hybrid sample has two complete multimodal segments regardless
of `K`: `clean_full` and `noisy_full`. `K_i = min(K, object_count_i)` controls
how many objects contribute KL sites, not how many extra segments are built. The
first default remains `K=1`; `K=2` is the first coverage/throughput ablation.
Larger `K` values should wait for KL-site count, throughput, and teacher-quality
evidence.

The atomic packing item should be the whole `HybridPrefixDenoisingSample`,
measured by the sum of its segment text-token lengths. A packer may place
multiple hybrid samples into one physical packed row, but it must not split one
hybrid sample across rows. If one hybrid sample already exceeds
`global_max_length`, skip that whole sample and log a skip counter rather than
splitting the image or dropping only one view. If the current row has room for
some but not all segments of the next hybrid sample, leave that whole sample for
the next pack.

Overlength handling is owned by the prefix-denoising hybrid pre-plan filter,
before generic static packing. Generic Stage-1 static packing remains fail-fast
for ordinary atomic samples. The V1 wrapper records `overlength_hybrid_sample`
exclusions in skip counters/manifests, then passes only eligible atomic hybrid
samples to the static pack plan.

The static pack plan is allowed because the hybrid sample length is invariant
across epochs: same two segments, same clean/noisy segment templates, same
coord-token representation, and deterministic skip/drop decisions. The selected
KL object indices and noisy coordinate token ids may change by epoch, but this
changes metadata/content, not text-token length. Therefore encoded-sample
caching must be disabled or epoch-aware for prefix-denoising; a cache keyed only
by base sample id would be stale when epoch-varying noise or KL object selection
is enabled.

Hybrid sample inclusion is decided before static length cache/plan construction
by an epoch-invariant feasibility predicate. Planned samples must never be
skipped at fetch time. The noiser must be constructive for every planned sample
and epoch; any fetch-time `4/4` or valid-box failure is an invariant violation
and must abort rather than returning a shorter or empty hybrid sample. Overlength
and deterministic infeasible-noise exclusions are recorded in pre-plan skip
manifests/counters, and the static packing fingerprint includes the
prefix-denoising schema version, noise config, and eligibility policy.

This encoded-sample cache is separate from the static packing cache. Static
packing cache stores or reuses a length/packing plan. Encoded-sample cache
stores already-built dataset samples, such as tokenized `input_ids`, labels, and
sidecar-like metadata, in fingerprinted shard files so later epochs or later
runs can load the encoded payload instead of re-rendering and re-tokenizing the
base JSONL record. That is useful for deterministic sorted SFT, but dangerous
for prefix-denoising V1: if noise or selected KL object indices vary by epoch, a
base-sample-id cache hit would replay the old noisy bbox map and old KL-site
metadata. V1 should therefore disable encoded-sample cache rather than add an
epoch-aware fingerprint/key surface in the first implementation.

With encoded-sample cache disabled, V1 dynamically fetches the offline-prepared
raw JSONL/image record and rebuilds the hybrid `clean_full`/`noisy_full` sample
for the current epoch. This is an accepted V1 cost because it preserves
epoch-varying bbox augmentation and selected-object KL coverage without adding a
new cache-key contract.

Inside a physical packed row, segments must remain attention-isolated even when
they come from the same source image. A packed row therefore needs a
`PackedHybridBoundaryMap` or equivalent runtime structure with, at minimum:

- `packed_row_index`;
- `hybrid_sample_id`;
- `segment_id`;
- `branch_id` and `view_id`;
- token `start`/`end` offsets in the packed row;
- supervised-label `start`/`end` offsets;
- local-to-packed position offset for every sidecar;
- position-id reset boundary;
- FlashAttention varlen boundaries when the padding-free path uses them:
  `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`;
- image placeholder span ownership;
- `pixel_values` / `image_grid_thw` slice ownership;
- CE denominator contribution;
- KL-site denominator contribution.

This boundary map is the contract between packing, sidecar rewriting, metrics,
and model input construction. It must agree with the actual flattened physical
order of `input_ids`, `labels`, image placeholders, visual grids, and sidecars.
Do not rely on a plain 2D `attention_mask` to represent multiple independent
segments inside one packed row; it is insufficient for sidecar-active packed
teacher forcing.

Visual inputs should be logically owned per segment. Even when two segments come
from the same source image, V1 should not allow one segment to attend across a
boundary to reuse another segment's image tokens. A later implementation can
deduplicate image preprocessing or cached pixel tensors by image id, but the
packed segment contract should still expose self-contained visual ownership for
placeholders, `pixel_values`, and `image_grid_thw` slices. This is a correctness
contract, not a new V1 config surface: current packing policy still uses
text-token length for fit decisions. V1 launch-health must verify that visual
slice counts and image-grid order match the segment boundary map.

Visual payload deduplication should remain an implementation-only cache
optimization. It must not change semantic per-segment visual ownership or permit
cross-segment attention.

The first implementation should intentionally accept duplicate visual encoder
work across the clean/noisy segments. If `clean_full` and `noisy_full` use the
same source image, V1 still treats them as two
self-contained multimodal segments with separate image placeholders,
`pixel_values`, and `image_grid_thw` ownership. This is not mathematically
required forever, but it is the simplest ms-swift-compatible shape that can be
launched correctly without custom shared-prefix attention, visual-feature cache
contracts, or upstream model-forward surgery.

V1 should introduce a narrow hybrid-aware packing wrapper/helper that reuses the
existing static packing cache and intact-atomic-sample philosophy where possible,
but emits `PackedHybridBoundaryMap` directly. This keeps the new sidecar-active
semantics local instead of changing generic packing first.

Sidecar-safe packing should be implemented as an explicit offset rewrite, not as
ad hoc post-hoc correction inside the loss. The rewrite must shift target IR atom
positions, current-object KL-site positions, local-window support metadata,
sample ids, and metric denominators from segment-local positions into packed-row
positions. For packed rows, existing teacher-forcing conventions that require
`batch_index` should see the physical packed row index, while
`hybrid_sample_id`/`segment_id` preserve semantic ownership.

Narrow unit and debug construction checks may run unpacked when that makes
tensor/label/metadata inspection simpler. Any named smoke, launch-health, or
train config should use the packed path so the evidence matches the intended
infra target.

When `prefix_denoising.current_object_kl.weight = 0.0`, the CE-only denoising
path should build only the full-sequence clean/noisy CE branch and should avoid
KL-site sidecars entirely. When
`prefix_denoising.current_object_kl.weight > 0.0`, the implementation must use
sidecar-safe packing before any training interpretation: current-object KL
requires aligned `clean_full` teacher and `noisy_full` student logits, KL-site
metadata, local-window support, and denominator accounting to survive packing.

Use one canonical prefix-denoising config surface. CE-only verification should
use `prefix_denoising.current_object_kl.weight = 0.0`; KL-on verification should
use the same surface with a positive weight such as `0.05`. The first code slice
may provide two tiny smoke leaves or two exact run commands for convenience, but
it should not introduce a separate clean-baseline config or additional feature
toggles.

Downstream rollout or free-decode eval policy is intentionally out of this V1
config/design surface. The next implementation gate is data/loss/packing
correctness. The already trained clean CE baseline remains a comparison
artifact; this note does not define a new rollout-eval protocol.

The first code slice should not implement rollout/free-decode eval. It should
prove data construction, packing, sidecar offsets, CE/KL loss numerics, and
launch-health metrics before any rollout policy is reopened.

## Verification Gates

Before any interpretation, V1 needs narrow launch-health checks:

- config/schema tests proving omitted/disabled prefix-denoising remains ordinary
  teacher forcing, `weight=0.0` is CE-only denoising, positive weight enables KL
  sidecars, inherited random/permuted object ordering is rejected, unknown nested
  keys are rejected, and non-V1 teacher-forcing packing guards remain intact;
- geometry noise tests for valid boxes, deterministic seeds, and changed-token
  assertions, including tiny boxes, boundary boxes, full-image boxes,
  width-1/height-1 skinny boxes, feasible-set construction, and no retry counter
  in the successful path;
- local-window support tests for clean bins near `0` and `999`, proving support
  bins are clipped, unique, valid, include the clean GT bin, and never wrap or
  pad;
- full-sequence clean/noisy label tests showing `input_ids` may differ while
  labels stay clean;
- clean/noisy view-difference tests proving only bbox coordinate input ids
  differ between branches, non-coordinate assistant input tokens stay identical,
  and labels are identical full-clean labels across branches;
- hybrid-container tests proving clean/noisy full segments have stable lengths,
  object order, image ownership, clean target metadata, and selected KL-site
  metadata;
- loss-scale tests showing `0.5 * CE_clean + 0.5 * CE_noisy` is on the expected
  scale, including an unequal-denominator formula test that distinguishes
  optimized balanced CE from token-pooled CE;
- metric-flattening tests proving clean/noisy/global prefix-denoising metrics
  publish distinct flat keys without `MetricEvent` identity collisions;
- current-object KL tests proving `clean_full` teacher and `noisy_full` student
  share image, order, selected object ids, clean bins, local support, and packed
  offsets;
- KL-site tests proving selected objects' `x1/y1/x2/y2` coordinate sites are
  eligible;
- stopgrad tests proving the `clean_full` teacher distribution is detached and
  the `noisy_full` student receives KL gradient;
- teacher-quality diagnostics comparing teacher vs student local-window GT-bin
  probability, pre-renormalization support mass, full coordinate-vocab GT-bin
  probability, entropy/top-1 distance where available, and top-1-is-GT at
  selected KL sites;
- metadata tests for branch ids, selected object indices, KL-site positions, and
  denominators;
- epoch-variation tests proving selected KL object indices and noisy coordinate
  values can change across epochs while `HybridPrefixDenoisingSample` text-token
  length, segment count, planned `hybrid_sample_id`s, and skip decisions remain
  invariant;
- `K > 1` tests proving V1 adds selected KL-site groups, not extra segments, and
  averages KL over selected objects/sites rather than summing;
- coverage diagnostics by object count, selected object index, history object
  count, coordinate slot, and identical-prefix site count;
- encoded-cache tests proving prefix-denoising disables cache reuse or keys it by
  epoch when content varies by epoch;
- packed/collator tests proving segment boundaries, position-id resets,
  FlashAttention varlen boundaries, image placeholder spans, `pixel_values` /
  `image_grid_thw` slices, target IR offsets, KL-site offsets, and metric
  denominators survive flattening;
- overlength-container tests proving a whole hybrid sample is skipped when its
  total segment text length exceeds `global_max_length`;
- a CE-only packed smoke before the KL-on packed smoke, while still implementing
  the KL path in V1;
- one-batch smoke proving finite CE, KL, total loss, token accuracy, and skip
  counters.

Before interpreting training quality, V1 also needs a noising-difficulty report:
per-slot bin deltas, clean/noisy IoU, center/size deltas, skip/exclusion rates,
noisy-full CE gap, and teacher/student GT-window changes for the default profile
and the named milder fallback. `4/4` changed proves non-noop corruption, not
that the perturbation is behaviorally meaningful or in-distribution.

If the `clean_full` KL teacher is not locally healthier than the `noisy_full`
student in tiny launch-health diagnostics, KL-on training should be treated as
numerically wired but not yet meaningful. This check must be per-slot and must
consider pre-renormalization support mass; a teacher that only looks healthy
after local-window renormalization is not enough. The first response should be
to lower noise, lower KL weight, or run CE-only denoising longer before
interpreting KL as helpful.

Adapter-backed tiny runs may use the recent well-tuned sorted pure-CE
checkpoint only for launch-health. Production or quality-bearing runs start
from the model-cache coord 2B full checkpoint. The first verification sequence is
packed CE-only denoising with `current_object_kl.weight = 0.0`, followed by the
same packed path with KL enabled at the first positive weight.

## Rationale

The original all-object A/B design cleanly expressed the mechanism, but it
scaled poorly. Expanding every object into A/B current-object KL probes repeats
the expensive prefix many times. Dense images would dominate training, and
packing would only reduce padding, not repeated attention over duplicated
image/prompt/history tokens.

The hybrid design separates two jobs:

- broad clean-target CE coverage belongs to full-sequence denoising;
- prefix insensitivity belongs to sparse clean-to-noisy KL on selected
  coordinate sites.

This preserves the central philosophy, "noisy coordinate prefix -> clean GT
continuation", while keeping the training surface closer to high-throughput
SFT. The full-sequence noisy view exposes every object and every coordinate
site to denoising pressure. Sparse clean-to-noisy KL then explicitly asks the
noisy-prefix student to match the clean-prefix teacher at selected object
coordinate sites.

This V1 KL is a prefix-insensitivity regularizer rather than an isolated
current-object-prefix diagnostic. The teacher uses clean previous and current
object prefixes; the student uses noisy previous and current object prefixes.
That makes the guidance less surgical but better aligned with the exposure-bias
goal: if generated or teacher-forced prefixes drift, the local coordinate
distribution should remain close to the clean-prefix behavior.

The design intentionally makes `K` a site-coverage knob, not a segment-count
knob. `K > 1` increases selected objects and KL sites inside the same two
forwarded segments. Option A, the same-noisy-history current-object teacher, can
remain a future diagnostic ablation if we later need to isolate current-object
prefix effects.

Keeping class/description/control tokens clean in V1 avoids conflating
geometric robustness with text or format robustness. Invalid and hard malformed
prefixes can be studied later, but the first experiment should test whether
small valid bbox drift can be corrected under teacher-forced clean labels.

## Consequence

The previous all-object A/B sample-unit design is superseded. The default CE
data surface is full-sequence clean/noisy denoising. The same two segments also
provide KL logits when KL is enabled: `clean_full` is the detached teacher and
`noisy_full` is the student.

Implementation planning should prioritize:

1. shared valid bbox noise maker;
2. `HybridPrefixDenoisingSample` and segment construction;
3. full-sequence clean/noisy sequence construction with clean labels;
4. CE scale and token-accuracy metrics;
5. `PackedHybridBoundaryMap` and sidecar-safe packing tests;
6. sparse selected-object KL-site selection and construction;
7. local-window clean-to-noisy KL.

Implementation should favor the simplest launchable ms-swift-compatible path
over visual deduplication, shared-prefix attention, or custom vision-cache
optimizations. Those optimizations are later throughput work, not V1 correctness
dependencies.

Approved integration decisions:

- Compute V1 with one packed model forward containing `clean_full` and
  `noisy_full`; detach only the local clean coordinate distributions used as KL
  teacher distributions.
- Add a top-level optional `prefix_denoising` config section to the
  `DetectionTrainingConfig` surface, because the strategy crosses data
  construction, packing, and loss; do not hide it under `objective.modules`.
- Disable encoded-sample cache for prefix-denoising V1 rather than making the
  cache epoch-aware immediately.
- Seed epoch-varying bbox noise and selected KL object indices by sample id and
  epoch so they are reproducible but still act like augmentation across epochs.
- Keep fixed segment order inside each hybrid sample: `clean_full`, then
  `noisy_full`.
- Build a stable pack plan only from hybrid samples whose V1 noise construction
  can succeed deterministically; if valid `4/4` changed bbox noise is infeasible,
  exclude the whole hybrid sample before packing and log a counter.
- Keep the training hot path constructive, without a normal reject/repair loop;
  repeated noising failure means the sample/config is infeasible or the noise
  maker has a bug.
- Use an internal deterministic epoch-shifted object permutation for selected KL
  object coverage, then take `K_i = min(K, object_count_i)`.
- Require full-logit availability for V1 CE, top-1/top-5 token accuracy, and
  selected local-window KL; reject logits-pruning paths unless proven to preserve
  all required monitor and KL-support tokens.
- Keep KL coordinate-only; do not apply KL to schema, description, separator,
  marker, or stop-token sites.
- Add tests proving only coordinate input ids differ between clean/noisy views,
  while full-clean labels and non-coordinate assistant input ids match.
- Keep semantic token metrics visible by branch/view where cheap, especially for
  schema, description, coordinate, object-control, separator, and stop groups.
- Do not introduce schema-specific weighting in V1.

The first comparison ladder should compare the existing clean SFT baseline
artifact against two prefix-denoising runs:

```text
matched_clean_full_ce_baseline
clean_full_ce + noisy_full_ce
clean_full_ce + noisy_full_ce + sparse_current_object_kl
```

If the existing clean SFT baseline is not matched to the V1 starting checkpoint,
data, object order, packing/cache policy, optimizer budget, route, and eval
config, label it `historical_anchor` instead of `matched_clean_full_ce_baseline`
and do not use it for improvement claims. The same-noise CE-only denoising run
is required before claiming that sparse KL adds value beyond broad noisy CE.

The old all-object KL-probe expansion and the same-noisy-history current-object
teacher can remain diagnostic ablations, but they are not the V1 default
training surface.
Mechanism attribution remains coarse until that Option A diagnostic exists:
successful V1 CE+KL supports broad clean-vs-noisy prefix consistency, not a
precise claim that current-object prefix perturbation alone was isolated.

## Open Grill Queue

No research-design decision currently blocks implementation. Remaining open
questions should be limited to implementation details discovered while wiring the
schema, hybrid builder, constructive noiser, sidecar-safe packer, loss code, or
launch-health metrics.

## Evidence

- Scope: `none-yet`
- Handles:
  - direction doc:
    `progress/directions/prefix_denoising_sft_v1.md`
  - Stage-1 route:
    `stage1_detection_teacher_forcing`
  - current Stage-1 packing guardrail:
    `docs/data/PACKING.md`
  - packing eligibility helper:
    `src/detection/packing.py`
  - detection config schema:
    `src/config/schema.py`
  - metric event contract:
    `src/metrics/events.py`
  - production starting checkpoint:
    `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
  - tiny launch-health adapter candidate:
    `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668`
