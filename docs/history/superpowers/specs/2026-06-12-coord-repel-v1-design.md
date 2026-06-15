# Coord-Repel V1 Design Spec

Status: implementation in progress; unit/config gates passed through packed-remap and config leaves; GPU smoke gates still pending
Date: 2026-06-12
Branch: `codex/coord-repel-conservative-design`
Source record:
`progress/explorations/2026-06-12_coord_repel_stage1_sft_design_decisions.md`

## Goal

Add a conservative Stage-1 teacher-forced SFT auxiliary loss that calibrates
coordinate basins. The loss should discourage the model from assigning high
coordinate probability mass to prior/seen coordinate basins when predicting the
next object's early coordinates.

The mechanism is intentionally training-time only. It must not change KV cache
behavior, attention, decoding, sampling, inference, or rollout policy.

## Scope

V1 applies only to Stage-1 teacher-forced detection SFT:

- `objective.id: teacher_forcing`;
- teacher-forced forward-pass training;
- teacher-forced forward-pass Stage-1 eval;
- packed and unpacked batches;
- static packing aligned with ms-swift / Transformers FlashAttention v2.

V1 does not apply to:

- Stage-2 rollout-aware training;
- decoded rollout predictions;
- inference-time duplicate suppression;
- attention or KV-cache intervention;
- geometry-level box repulsion;
- `x2` / `y2` extent slots.

## Core Concept

Coord-repel is a coordinate-basin calibration loss.

For the current coordinate slot, the desired condition is:

```text
probability mass near current GT coordinate basin
>
probability mass near prior/seen wrong coordinate basins
```

For a current `x1` or `y1` slot:

```text
B+ = current GT coordinate band
B- = prior/seen same-role coordinate bands
     intersected with current wrong high-probability coordinate top-k
```

The loss is:

```text
softplus(margin + log P(B-) - log P(B+))
```

where `P(B+)` and `P(B-)` are conditional probability masses over the coordinate
vocabulary only.

This is not generic object diversity. It is not "do not look at old boxes." It
specifically trains the model not to treat prior/seen coordinate bands as the
next object's coordinate attractor when the current logits are already assigning
high probability to those wrong basins.

## B+ And B- Definitions

V1 roles:

```text
roles = [x1, y1]
```

Positive band:

```text
B+ = current GT coordinate token +/- 4 bins
```

Candidate negative bands:

```text
same source sample
same coordinate role
object_order_index < current.object_order_index
negative_source = last_and_same_desc by default
band radius = +/- 4 bins
```

`last_and_same_desc` means:

```text
last previous object
union all previous objects with matching desc/category key
```

`all_prior` remains a stress/diagnostic mode, not the balanced default.

Activated negative band:

```text
coord_logits = logits_at_slot[coord_token_ids]
coord_probs = softmax(coord_logits)
top_k_wrong = topk(coord_probs, k=32), excluding B+
B- = top_k_wrong intersected with prior same-role coordinate bands
```

If `B-` is empty, the slot contributes no coord-repel loss. If `B+` is empty,
that is a target/support contract error and should fail fast.

Future GT objects, rollout predictions, model-generated boxes, and objects from
other packed source samples must not contribute to `B-`.

## Probability Space

Both `B+` and `B-` are computed in the coordinate-token vocabulary:

```text
coord_log_probs = log_softmax(logits_at_slot[coord_token_ids])
log_p_pos = logsumexp(coord_log_probs[B+])
log_p_neg = logsumexp(coord_log_probs[B-])
loss = softplus(margin + log_p_neg - log_p_pos)
```

The computation should run in fp32.

`coord_token_ids` must be an ordered 1000-bin coordinate vocabulary resolved
from the tokenizer/coordinate-token utility, not inferred by sorting the
coordinate-token ID set. The ordered tuple may be checked against the
`RoleVocab.coord_token_ids` set, but bin identity must come from the tokenizer
contract.

Full-vocabulary format correctness remains the job of the main teacher-forcing
objective. Under `hard_sft`, that objective is singleton CE. Under
`pure_valid_set_marginal`, it is valid-set marginal next-token likelihood.
Coord-repel must not entangle its basin signal with punctuation, JSON/schema
tokens, description tokens, or special tokens.

## Reduction And Strength

Reduce over active coordinate slots:

```text
active slot = x1 or y1 slot with non-empty B-
slot_loss = softplus(margin + log_p_neg - log_p_pos)
loss_sum = sum(slot_loss over active slots)
raw_loss = loss_sum / active_slots
weighted_loss = coord_repel.weight * raw_loss
total_loss = main_teacher_forcing_loss + weighted_loss
```

Initial V1 strength:

```yaml
coord_repel:
  weight: 0.05
  margin: 0.25
  top_k: 32
  negative_source: last_and_same_desc
```

The balanced/main setting uses `weight: 0.05`. A stress-smoke override may use
`weight: 0.1` to confirm that the mechanism is visible, but the primary
ablation should start from the balanced setting unless telemetry shows the term
is inactive.

## Public Config Surface

Expose only the minimal V1 experiment levers:

```yaml
objective:
  modules:
    coord_repel:
      enabled: true
      weight: 0.05
      margin: 0.25
      top_k: 32
      negative_source: last_and_same_desc
```

Keep these mechanism choices internal for V1, but log them in resolved metadata
or metrics:

```text
roles = [x1, y1]
positive_radius = 4
negative_radius = 4
distribution = coordinate_vocab_conditional
reduction = active_slot_mean
```

## Packing Architecture

V1 must work with static packing and FlashAttention v2. The design aligns with
the current ms-swift / Transformers surface rather than introducing custom
attention code.

Architecture:

```text
src/data_collators/packed_layout.py
  generic packed physical-row layout

src/training/teacher_forcing/packing.py
  remap per-source teacher-forcing IRs into one IR per physical packed row

src/training/teacher_forcing/coord_repel.py
  typed coord-repel slot extraction
  B+/B- construction
  pure tensor loss
  diagnostics/result object

src/training/objectives/teacher_forcing.py
  integrate main teacher-forcing loss and coord-repel auxiliary term
```

The collator-side packed layout owns physical packing offsets and source sample
boundaries. The teacher-forcing remapper emits one merged
`TeacherForcingTargetIR` per physical packed row. Coord-repel then consumes the
already-remapped IR and must not infer offsets itself.

V1 does not add a new public `packing.remap_contract` key. Packed
teacher-forcing requires the existing
`objective.target_ir.exact_packing_mapping.enabled=true` guard and records the
effective remap contract as `teacher_forcing_atoms_v1`. Unpacked
teacher-forcing may keep the current false/default value.

Packed teacher-forcing is eligible only for the validated static-packing
contract: `training.packing=true`, `training.eval_packing=true`,
`packing.static_packing=true`, `packing.padding_free_packed=false`,
per-device train batch size 1, right padding, `flash_attention_2`, full logits,
encoded-sample cache disabled, and exact teacher-forcing remap enabled. The
`packing.padding_free_packed` config mode is a separate experimental runtime
mode and is not the ordinary ms-swift static packing path.

The effective packed remap contract is `teacher_forcing_atoms_v1` in merged
IR metadata and `teacher_forcing_target_ir_exact_mapping_v1` in static-packing
fingerprints. Those identities prevent cache/fingerprint collisions with
ordinary `random_order_sft` static packing.

## Coord-Repel Context

Coord-repel V1 derives context from the remapped `TeacherForcingTargetIR`; it
does not introduce a separate dataset or collator sidecar.

The implementation should use a typed extractor that promotes required
IR/provenance fields into:

```text
CoordRepelSlot:
  source_sample_id
  physical_batch_index
  object_index
  object_order_index
  object_instance_id
  coord_role
  coord_bin
  selected_token_id
  logit_position
  target_position
```

`object_order_index` comes from
`TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]`.
`object_index` and `object_instance_id` may come from atom provenance, but only
inside the typed extractor/validator.

If coord-repel is enabled and required fields are missing, fail fast with a
clear contract error.

## Metrics And Diagnostics

Log raw and weighted loss contributions, effective strength, and activation
statistics. Required metrics:

```text
coord_repel/raw_loss
coord_repel/weighted_loss
coord_repel/active_slots
coord_repel/eligible_x1_slots
coord_repel/eligible_y1_slots
coord_repel/empty_bminus_slots
coord_repel/p_pos_mean
coord_repel/p_neg_mean
coord_repel/log_margin_mean
coord_repel/topk_intersection_count
coord_repel/wrong_top1_in_bneg_rate
coord_repel/b_seen_coverage_fraction
coord_repel/weighted_loss_to_main_tf_loss_ratio
coord_repel/negative_source_mode
```

Metric names should make clear that `P(B+)`, `P(B-)`, and top-k are conditional
on the coordinate vocabulary. Ratio metrics use the current main
teacher-forcing loss as denominator and must not be named as CE unless the run
profile is explicitly `hard_sft`.

## Verification Gates

Do not launch production training until the following narrow gates pass:

1. Formula probe:
   deterministic logits where high `B+` mass gives low loss and high `B-` mass
   gives higher loss.
2. Band construction probe:
   positive radius, negative prior bands, top-k intersection, empty `B-` skip,
   and empty `B+` fail-fast.
3. Causal order probe:
   only prior teacher-forced objects contribute to `B-`; future objects do not.
4. Packing probe:
   packed `A+B` rows do not allow `A` prior bands to affect `B`, or vice versa.
5. Packing runtime contract probe:
   packed teacher-forcing is accepted only with
   `objective.target_ir.exact_packing_mapping.enabled=true`,
   `teacher_forcing_atoms_v1`, padding-free static packing, FlashAttention v2,
   per-device train batch size 1, and encoded cache disabled.
6. Coordinate-vocab probe:
   coord-repel uses the ordered tokenizer coordinate vocabulary and fails fast
   if it disagrees with the role-vocab coordinate-token set.
7. Accumulation scaling probe:
   unequal active-slot counts reduce by global active-slot mean, not average of
   microbatch means.
8. Config probe:
   only `enabled`, `weight`, `margin`, `top_k`, and `negative_source` are
   public V1 knobs.
9. Eval-forward probe:
   Stage-1 eval computes the same forward-only coord-repel diagnostics without
   rollout assumptions.
10. Canonical-failure activation probe:
   known duplicate/prefix-sensitive cases produce non-empty `B-`, measurable
   `wrong_top1_in_bneg_rate`, and interpretable `log P(B+) - log P(B-)`.
11. Negative-source probe:
   compare `last_only`, `same_desc`, `last_and_same_desc`, and `all_prior` on
   canonical cases before promoting any source mode beyond V1 default.
12. Box-end template smoke:
   verify the `compact_box_end` template inserts exactly one `<|box_end|>` after
   `y2`, uses no newline, and keeps the token trainable/saved.
13. Real-GPU launch-health smoke:
   run a short training smoke on real GPU hardware, not only unit tests or CPU
   formula probes, and verify coord-repel loss and monitoring metrics are
   active, finite, and scaled as intended.
14. COCO length-12000 stability smoke:
   use the COCO Stage-1 route configured for `global_max_length: 12000`; do not
   substitute a `max_objects: 60` capped dataset for launch evidence.
15. LLM-tower-only trainability probe:
   verify the launch trains the LLM tower/token embeddings only and keeps the
   vision tower and other non-LLM parameter groups frozen unless separately
   approved.
16. Loss-numerics stability gate:
   follow the `loss-numerics-sanity` evidence standard for a longer smoke run:
   finite total/component losses, nonzero valid counts, stable denominators,
   interpretable raw/weighted coord-repel contribution, stable gradient norms,
   and a decreasing main-loss trend over a meaningful step window.
17. Task-health gate:
   decreasing loss is not sufficient. Parser/template validity, eval-forward
   metrics, packed-remap integrity, and no cross-source negative leakage must
   remain healthy before production training.
18. Base-2B control arm:
   run the same real-GPU smoke and longer stability gates on the base 2B model
   as a required control arm. Use the same COCO length-12000, LLM-tower-only,
   packed teacher-forcing contract so failures can be attributed to the
   objective/runtime rather than checkpoint-specific history.

## Launch And Production Gates

The production run is not unlocked by passing unit tests alone. After
implementation, the launch sequence is:

1. Run deterministic formula, config, packed-remap, objective-integration, and
   eval-forward tests.
2. Launch short real-GPU smoke runs to verify the intended loss and monitoring
   metrics are live on actual hardware.
3. Include the base 2B model as a required control arm for those real-GPU smoke
   runs.
4. Launch longer real-GPU stability runs on the COCO Stage-1 length-12000
   dataset, with no `max_objects: 60` cap substituted for evidence collection.
   The base 2B arm must pass this same gate.
5. Train LLM tower/token embeddings only during the launch-health and stability
   runs. Record trainable parameter groups in the run artifact.
6. Interpret early curves with the `loss-numerics-sanity` checklist. Required
   evidence includes authored and resolved configs, one real target example,
   one collated batch, deterministic formula probes, raw and weighted losses,
   valid counts, gradient norm, learning rate, step, accumulation count, finite
   checks, memory/utilization, and artifact handles.
7. Only after both numerics and task-health gates pass, launch production
   training with 8 GPUs, 4 epochs, static packing, padding-free packed batches,
   FlashAttention v2, and the exact teacher-forcing packing remap contract.
   If a non-base candidate checkpoint receives a production launch, include the
   base 2B model as a matched production/control arm under the same contract
   unless the production scope is explicitly narrowed later.

Production launch blockers:

- coord-repel enabled but inactive or absent from logs;
- `coord_repel/active_slots` is zero on duplicate-sensitive smoke fixtures;
- `coord_repel/weighted_loss_to_main_tf_loss_ratio` is missing, nonfinite, or
  outside an interpretable range without explanation;
- total loss decreases while parser/template/eval-forward artifacts degrade;
- trainable parameter report shows vision-tower or other non-LLM parameters
  unexpectedly trainable;
- packed samples share coord-repel prior bands across original source samples;
- COCO length-12000 evidence is replaced by a max-60-object capped dataset.
- base 2B control-arm smoke or stability run fails the same numerics or
  task-health gates.

## Risks

The main risks are:

- loss becomes invisible despite being enabled;
- loss dominates the main teacher-forcing loss and destabilizes coordinate
  learning;
- future-object leakage creates oracle supervision;
- packed samples leak prior bands across sample boundaries;
- active-slot denominator is wrong under packing, accumulation, or DDP;
- `x2/y2` overcorrection is introduced accidentally.
- default negatives become too broad in crowded scenes.
- real-GPU telemetry looks healthy while the auxiliary term is inactive,
  detached, incorrectly reduced, or hidden by packing/accumulation scaling.

V1 mitigates these by limiting the mechanism to `x1/y1`, using top-k-activated
negative bands, defaulting to `last_and_same_desc`, reducing by active slots,
requiring targeted tests before production training, and gating production on
both `loss-numerics-sanity` launch evidence and task-health artifacts.

## Template Ablation Axis

The coord-repel experiment should include an independent template axis:

```text
compact:
  <object_ref_start>{desc}<box_start>x1 y1 x2 y2<object_ref_start>...

compact_box_end:
  <object_ref_start>{desc}<box_start>x1 y1 x2 y2<|box_end|><object_ref_start>...
```

Hard constraints:

```text
no newline
no <object_ref_end>
<|box_end|> is the only row-commit token
```

Recommended 2x2 matrix:

```text
1. compact + no coord-repel
2. compact_box_end + no coord-repel
3. compact + coord-repel
4. compact_box_end + coord-repel
```

This separates coordinate-basin calibration from row-commit structure.

## Current Non-Goals

V1 intentionally does not implement:

- same-desc-only as the balanced/default negative source;
- `x2/y2` repulsion;
- geometry-level box repulsion;
- rollout prediction negatives;
- inference-time deduplication;
- attention/KV intervention;
- broad hyperparameter surface over radii or roles;
- positive-band anchor loss.

These can be revisited after the first numerically validated smoke run.
