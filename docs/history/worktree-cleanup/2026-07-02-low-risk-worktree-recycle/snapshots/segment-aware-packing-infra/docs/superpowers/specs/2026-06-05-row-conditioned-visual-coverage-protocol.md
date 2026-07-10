# Row-Conditioned Visual Coverage Protocol

Status: research protocol; not an OpenSpec contract.

Date: 2026-06-05

Primary design: `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-design.md`

Progress record: `progress/directions/2026-06-05_row_conditioned_visual_coverage.md`

Evidence scope: `none-yet`. This protocol is a pre-implementation research
specification. It defines what the first prototype must mean and how results
may be interpreted; it does not claim that the mechanism works.

## Decision

Use this protocol as the next documentation layer before implementation. The
existing design document records the conceptual claim. This protocol records the
implementable boundaries, evidence gates, artifact handles, and OpenSpec
boundary for a first prototype.

The first prototype remains:

```text
row-conditioned visual coverage
+ standard teacher-forced SFT with CE
+ faithful row-boundary re-prefill rollout
```

It does not add residual-set correction, trie/set supervision,
duplicate-unlikelihood, attention-bias routing, or fallback decoding.

Interpret the prototype as a **Direct Feature Painting Oracle**. Its purpose is
to test whether explicit row-conditioned visual coverage is useful under a
faithful tensor-flow path. KV-cache efficiency, append-coverage-token decoding,
and other cache-friendly approximations are separate future comparisons.

## Documentation Stack

Use the following stack for this branch:

1. Concept design:
   `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-design.md`
2. Research protocol:
   `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md`
3. Future implementation plan, after protocol review:
   `docs/superpowers/plans/2026-06-05-row-conditioned-visual-coverage-prototype.md`
4. Direction record:
   `progress/directions/2026-06-05_row_conditioned_visual_coverage.md`
5. OpenSpec:
   deferred until a stable config schema, runtime behavior, artifact contract,
   or metric contract is ready.

The direction record explains why the idea exists. The design and protocol
constrain the first prototype. The implementation plan may be executed later,
but no implementation is part of this protocol.

Do not link this protocol from `docs/training/README.md`, `docs/AGENT_INDEX.md`,
or `docs/catalog.yaml` as a supported training surface until there is runnable
config-first support and verification evidence. At this stage it is a
repo-local research protocol only.

## Experimental Config Boundary

Use a detection-config-native experimental section for v1:

```yaml
row_conditioned_visual_coverage:
  enabled: true
  version: v1
  row_state_policy: prefix_expansion
  tune_operator: additive_residual
  descriptor:
    boundary_channel: true
    interior_channel: true
    boundary_band_width_tokens: 1.0
    accumulation: clamp_0_1
  residual:
    boundary_alpha_init: 0.0
    interior_alpha_init: 0.0
    boundary_alpha_max: 0.25
    interior_alpha_max: 0.08
  training:
    unpacked_only: true
    include_terminal_state: true
    ordering_modes: [random_sft, sorted_sft]
  rollout:
    strategy: row_boundary_reprefill
```

This bucket is intentionally not a stable schema contract. The implementation
must still parse and validate it strictly inside the row-coverage module so
experiments fail early on misspelled keys.

Do not add this section to OpenSpec or stable operator docs for v1.

The first comparison must include a no-coverage row-boundary re-prefill
baseline. Without this baseline, any rollout change is confounded by the
prefill protocol itself rather than coverage.

## Row-State Dataset Protocol

For each detection scene with ordered objects `O_1..O_N`, materialize
row-state examples:

```text
k = 0: prefix = empty rows, coverage = empty, target = O_1
k = 1: prefix = O_1, coverage = O_1, target = O_2
k = r: prefix = O_1..O_r, coverage = O_1..O_r, target = O_{r+1}
k = N-1: prefix = O_1..O_{N-1}, coverage = O_1..O_{N-1}, target = O_N
k = N: prefix = O_1..O_N, coverage = O_1..O_N, target = assistant stop
```

The terminal state is ordinary chat-template CE over the assistant stop target.
It is not a custom stop objective. It exists because rollout also reaches a
row-conditioned stop decision after all committed objects.

Teacher-forced tensors may include the supervised target row in the encoded
assistant text so shifted CE has label positions to supervise. The row-state
prefix and coverage, however, must exclude the target row. In implementation
terms:

- prefix/coverage object indices for object row `k` are `0..k-1`;
- rendered teacher-forcing indices for object row `k` may be `0..k` so the
  target row exists in `input_ids`;
- labels/target IR are built only for row `k`;
- rows before `k` are prefix context and must not be supervised for that
  row-state example.

For `random_sft`, the ordered object sequence comes from the deterministic
per-epoch random-permutation ordering already used by CoordExp detection
datasets. For `sorted_sft`, it comes from the sorted order. The coverage
mechanism, descriptor, CE target, and rollout rule must remain identical across
the two modes.

Reports must keep `random_sft` and `sorted_sft` separate. They answer different
questions:

- `random_sft` tests whether coverage supports arbitrary enumeration without a
  fixed spatial policy;
- `sorted_sft` tests whether coverage helps when next-row behavior is already
  partly structured by object order.

Packing is disabled for the first prototype. The coverage state sidecar is
unpacked-only until packed offset mapping is designed explicitly.

## Coverage State Sidecar

Each row-state training sample should carry a sidecar with the semantic state,
not pre-painted visual tensors:

```text
sample_id
base_idx
row_state_k
target_kind: object_row | assistant_stop
target_object_index: int | null
prefix_object_indices: list[int]
coverage_object_indices: list[int]
rendered_teacher_forcing_indices: list[int]
supervised_object_indices: list[int]
image_width
image_height
coverage_boxes_norm1000_xyxy
ordering_strategy
ordering_seed
```

Painting should happen after `image_grid_thw` is known, so the descriptor uses
the actual model processor grid rather than a guessed pixel lattice.

## Bbox-To-Visual-Lattice Protocol

The painter consumes:

```text
coverage boxes in norm1000 xyxy
image width and height
image_grid_thw from the model batch
visual spatial merge size
```

For v1:

- paint one image sample at a time, using that sample's one-image,
  one-time-step visual grid;
- derive the effective tuned-token lattice from post-merge visual features:
  `t * (h // merge_size) * (w // merge_size)`;
- reject grids where `h` or `w` is not divisible by `merge_size`;
- convert norm1000 boxes to image-space coordinates through
  `src/datasets/geometry.py`;
- compute rectangular token footprints in row-major order;
- compute interior mass as fractional token-footprint overlap with the bbox;
- compute boundary mass as fractional overlap with a soft bbox-edge band;
- set band width to one effective visual-token footprint by default;
- accumulate masses additively across covered boxes and clamp to `[0, 1]`;
- reject descriptor length mismatches against image-feature length.

Coordinate alignment is the highest-priority validation item. Before any
metric interpretation, run a visual/debug validation that traces:

```text
norm1000 bbox
-> original image coordinates
-> processor grid metadata
-> patch grid
-> post-merge visual-token lattice
-> painted boundary/interior token cells projected back onto the image
```

The validation must show that painted token cells correspond to the intended
image regions. If this alignment fails, every downstream coverage claim is
blocked, even if loss decreases.

The `merge_size` must come from the active model or processor path, such as
`model.visual.spatial_merge_size`, `model.model.visual.spatial_merge_size`, or
`processor.image_processor.merge_size`. Do not assume raw patch-grid cells are
the tuned feature tokens.

Boundary mass should be computed from an outer-minus-inner rectangle band:

```text
outer = expand(box, one token footprint)
inner = shrink(box, one token footprint)
boundary_area = area(token intersect outer) - area(token intersect inner)
boundary_mass = clamp(boundary_area / area(token), 0, 1)
```

Interior mass is:

```text
interior_mass = clamp(area(token intersect box) / area(token), 0, 1)
```

Do not globally normalize masses over the image.

Boundary mass is bbox-boundary coverage over claimed box extent. It is not a
true instance contour. A later mask/segmentation version may claim contour
coverage only if masks are actually used in the painter.

Golden cases required before training:

- empty coverage produces all-zero boundary/interior mass;
- a token-aligned box paints the expected interior rectangle;
- a tiny box smaller than one visual token still produces a bounded visible
  coverage mark, with boundary mass dominating interior mass;
- adjacent boxes accumulate without changing unrelated tokens;
- overlapping and nested boxes accumulate locally and saturate only where their
  footprints overlap;
- clipped boxes are clipped before painting;
- degenerate boxes are rejected or normalized through the same geometry policy
  used by detection serialization.

The default boundary band straddles the bbox edge through the
outer-minus-inner construction above. If a later implementation uses an
inside-only band, it must be treated as an ablation rather than the v1 protocol.

## Visual Feature Tuning Protocol

The first tuning operator is:

```text
V'_i = V_i
     + alpha_b * boundary_mass_i * e_boundary
     + alpha_i * interior_mass_i * e_interior
```

Implementation requirements:

- `e_boundary` and `e_interior` are trainable vectors with the same hidden size
  as the final image features consumed by the language model;
- alpha parameters are trainable scalar gates initialized near identity;
- alpha values are bounded before use;
- original visual features are preserved by residual addition;
- empty coverage returns byte-for-byte equal features when alpha is zero;
- no covered token is zeroed, masked out, or removed from attention.

Residual monitoring is required. Alpha caps bound only the scalar gates; they
do not bound residual size if learned embedding norms grow. Coverage-active
smokes must record at least:

```text
row_coverage/delta_to_feature_norm_ratio
row_coverage/max_token_delta_to_feature_norm_ratio
row_coverage/mean_token_delta_to_feature_norm_ratio
row_coverage/boundary_embedding_norm
row_coverage/interior_embedding_norm
```

If the residual norm ratio becomes large enough to dominate original visual
features, interpretation must be held until a conservative variant is tested:
smaller initialization, tighter alpha caps, embedding-norm control, or explicit
regularization.

Interior semantics are "already enumerated", not "visually irrelevant". The
interior residual must never be described as direct suppression. Any duplicate
reduction or attention shift must be learned through CE and validated against
recall, crowded scenes, overlap, and nested-object slices.

The first proof must include capacity controls:

- coverage off;
- zero-mass descriptor through the same tuning helper;
- boundary-only;
- interior-only;
- shuffled coverage boxes within the same image;
- wrong-image coverage;
- stale coverage from a previous prefix state.

V1 coverage is union memory. These controls do not test object identity memory
because the descriptor does not carry identity, count, class text, confidence,
or recency. Future ablations may add count/log-count channels, class-summary
coverage, confidence-weighted coverage, or instance-memory variants, but those
are not part of the v1 claim path.

These controls are required because a positive result from a larger wrapper or
adapter is otherwise ambiguous: it may come from extra capacity, ordering, or a
prefix/runtime change rather than coverage.

The implementation must not edit upstream Hugging Face Qwen files. Use a local
wrapper or context-managed forward helper that applies the same tuning function
in training and rollout.

For Qwen3-VL-style `ForConditionalGeneration` wrappers, the helper must resolve
and patch the inner visual-feature owner that actually calls
`get_image_features`, such as `model.model`, not only the top-level wrapper.

Deepstack policy for v1:

- tune the final image features consumed by the language model;
- if the active model path also returns or forwards deepstack visual embeds,
  either tune them with the same coverage descriptor at the matching effective
  lattice or reject coverage-active forward with an explicit unsupported-policy
  error;
- do not silently leave deepstack visual embeds on a different coverage state
  while claiming first-version tensor-flow evidence.

## Forward Boundary

The model-forward boundary should treat coverage as a bridge-consumed input,
not as an argument forwarded into the HF model:

```text
raw batch sidecar -> collator -> TrainerLossBridge -> coverage forward helper
```

The helper may intercept the model's image-feature path in a local wrapper, but
the tensor computation must remain:

```text
image features V
coverage descriptor M_k
tuned features Tune(V, M_k)
standard autoregressive decoder forward
```

The same helper must be callable from training and rollout. If a path cannot
call the same helper, that path is not eligible for first-version evidence.

The shared helper state must include the coverage config, painter, merge-size
resolver, tuner module, and no-cache policy. Rollout must use the same trained
tuner object or a loaded state with the same recorded hash as training.

Tensor-flow parity requirement for any fixed prefix state `S_k`:

```text
training conditional:
  p(row_k | prompt, text_prefix(S_k), Tune(V, coverage(S_k)))

rollout conditional:
  p(row_k | prompt, text_prefix(S_k), Tune(V, coverage(S_k)))
```

The source of `S_k` may differ. The tensor computation for a given `S_k` must
not.

## Rollout Protocol

First proof uses row-boundary re-prefill:

```text
raw_V = image_features(image)
committed_rows = []

for row_state_k in 0..max_rows:
    coverage_state = coverage(committed_rows)
    tuned_V = Tune(raw_V, coverage_state)
    prefill prompt + tuned_V + committed row text
    decode next row or assistant stop
    if assistant stop: finish
    if row parse-invalid: record invalid and finish or apply configured strict failure
    commit valid row
```

The image encoder should run once per image in the rollout helper when the
underlying model interface allows precomputed image features. If the first
engineering slice must temporarily re-encode during a smoke test, the artifact
must label that as a smoke-only runtime limitation and it must not be used for
the primary mechanism claim.

KV-cache efficiency is explicitly deferred. The rollout proof recomputes the
decoder prefix at row boundaries so text-prefix hidden states are conditioned
on the current visual coverage state.

Coverage-active row-boundary prefill must not reuse cross-row `past_key_values`.
It must remove stale `past_key_values` and force `use_cache=false` for the
coverage-conditioned prefill. Within-row decode cache is allowed only after the
row has been prefilling under the current tuned visual state, and artifacts must
record `cache_reused_across_rows=false`.

Do not mix this faithful re-prefill path with the cache-friendly
append-coverage-token variant unless the experiment is explicitly comparing the
two. A cache-compatible path cannot be used to interpret the Direct Feature
Painting Oracle unless it proves tensor-flow equivalence or is reported as a
separate approximation.

Artifacts must also record whether precomputed image features were used:

```text
image_encoder_calls
uses_precomputed_image_features
reencode_policy
evidence_scope
```

Any primary mechanism claim is blocked when `image_encoder_calls > 1` unless
the run is explicitly labeled `tiny` or `smoke-only runtime limitation`.

## Evidence Gates

Use the following claim ladder:

```text
no_claim
mechanism_alive
rollout_safe
val200_promising
full_eval_candidate
```

Stage 0: unit and smoke contract.

- config validates;
- row-state expansion produces correct object and terminal states;
- coverage painting respects lattice size and bounded mass;
- empty coverage is identity;
- covered regions remain observable by construction;
- training and rollout call the same tuning helper.
- no-coverage row-boundary re-prefill is runnable.
- prefix rows are context only and are not re-supervised for row-state `k`;
- current target row and terminal assistant-stop row receive CE loss;
- norm1000-to-lattice visual alignment overlays pass inspection;
- residual norm ratios and embedding norms are logged or exported.

Stage 1: mechanism gate.

- compare coverage off vs on under matched prefix states;
- inspect early coordinate binding, especially `x1/y1`;
- report `x1`, `y1`, `x2`, and `y2` separately;
- report target rank, target-neighborhood mass, same-description competitor
  rank, residual-vs-EOS margin, and residual-vs-emitted margin;
- report `random_sft` and `sorted_sft` separately;
- record whether coverage changes target-region rank or local duplicate-basin
  mass before interpreting rollout.

Stage 2: rollout guardrail.

- duplicate rate must not improve by lowering recall;
- overlap and crowded slices must not regress silently;
- invalid parse rate must not increase materially;
- EOS or row-count truncation must be reported directly;
- COCO partial annotation caveats must be separated from near-identical
  same-description duplicates.

Stage 3: exposure-gap guardrail.

- compare clean teacher-forced coverage with corrupted-prefix coverage;
- include bbox jitter, dropped previous objects, duplicated previous objects,
  enlarged/shrunk boxes, wrong label with right box, right label with shifted
  box, and optional confidence-weighted coverage as controlled corruptions;
- report row-wise recovery after prefix corruption, not only final image-level
  AP/AR.

Metric-bearing claims must come from score-bearing evaluation artifacts. Debug
F1-ish, proxy, tiny, val200, partial, and full-val results must be labeled with
their evidence scope and must not be collapsed.

Primary rollout readouts:

- AP, AP50, AP75, and AR100 when a score-bearing evaluator artifact exists;
- F1-ish precision and recall only when clearly labeled as debug/proxy;
- predicted row count;
- invalid parse rate;
- premature assistant-stop rate;
- same-description duplicate rate at IoU greater than `0.95`;
- duplicate burst rate;
- repeated same-instance emission rate;
- FN and FP count;
- class-found-but-instance-missed rate;
- crowded, overlap, nested-object, small-object, and same-description slices.
- row-wise recovery after prefix corruption;
- residual norm ratio summaries.

No positive rollout claim is allowed when duplicate reduction is bought by:

- lower recall or AR100;
- fewer predicted rows without an explicit recall gain;
- higher assistant-stop or premature-stop rate;
- increased invalid parse rate;
- post-hoc duplicate guarding rather than changed generation;
- collapsing `random_sft` and `sorted_sft` into one aggregate;
- score-free debug artifacts reported as official metrics.

Failure taxonomy to use in reports:

- coverage suppression;
- overlap or nested-object suppression;
- same-description competitor rebinding;
- duplicate relocation rather than duplicate removal;
- premature assistant stop;
- parse instability;
- object-count truncation;
- prefix-state mismatch;
- current/future row leakage;
- patch-resolution mismatch;
- partial-annotation FP ambiguity.

## Artifact Policy

The first prototype should write artifacts under:

```text
/data/CoordExp/outputs/analysis/row_conditioned_visual_coverage/
```

Recommended subdirectories:

```text
mechanism_probe/
rollout_smoke/
rollout_val200/
reports/
```

Recommended first artifact names:

```text
resolved_config.json
coverage_config.json
coverage_state_rows.jsonl
coverage_descriptor_stats.json
coverage_alignment_overlays/
coverage_alignment_rows.jsonl
row_coverage_residual_norms.json
loss_mask_audit_rows.jsonl
mechanism_probe_rows.jsonl
rollout_rows.jsonl
rollout_summary.json
guardrail_metrics.json
report.md
```

These names are research artifacts for this branch, not stable evaluator or
inference contracts.

First reports should include a no-claim ledger. Each apparent improvement must
state whether it is usable evidence for coverage, and if not, which no-claim
rule blocked interpretation.

## Review And Promotion Gates

Before implementation:

- protocol and implementation plan must be reviewed;
- no stable `docs/training`, `docs/catalog.yaml`, `docs/AGENT_INDEX.md`, or
  OpenSpec route may present this as supported behavior;
- the implementation plan must keep bbox math routed through
  `src/datasets/geometry.py` where geometry conversion is needed;
- upstream Hugging Face model files must remain untouched.

Before promotion to operator-facing `docs/training`:

- there is a runnable config-first workflow;
- narrow tests and at least one smoke artifact exist;
- the doc gives an exact run command and known unsupported modes;
- artifact and metric scope labels are present;
- `random_sft` and `sorted_sft` remain separate;
- the workflow is intended as current experimental behavior, not a one-off
  probe.

Before OpenSpec promotion:

- at least one compatibility-sensitive surface is intentionally stabilized;
- an active OpenSpec change exists with proposal, design, tasks, and capability
  deltas;
- config, runtime, artifact, or metric tests cover the stabilized behavior;
- strict OpenSpec validation passes.

## OpenSpec Boundary

Do not create OpenSpec for this branch now.

OpenSpec becomes appropriate only after at least one surface is ready to become
a supported contract:

- top-level `row_conditioned_visual_coverage` or another stable
  detection-native config schema;
- row-state dataset behavior;
- coverage descriptor artifact names and schema;
- rollout re-prefill runtime behavior;
- coverage-specific metric semantics.

Until then, design, protocol, plan, and progress notes are the correct durable
surfaces.
