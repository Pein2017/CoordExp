---
title: Coord-Repel Stage-1 SFT Design Decisions
date: 2026-06-12
status: discussion-decisions
scope: none-yet
---

# Coord-Repel Stage-1 SFT Design Decisions

This note records resolved design decisions from the `grill-me-with-docs`
pressure-test for the conservative coord-repel experiment. It is not an
implementation report and does not describe current runtime behavior.

## Decision: V1 Scope Is Stage-1 SFT Only

V1 coord-repel is scoped behaviorally to the Stage-1 SFT / detection
teacher-forcing phase only, using the active
`stage1_detection_teacher_forcing` surface under
`configs/stage1/detection_teacher_forcing/`.

The implementation design may use neutral names such as `coord_repel` and
slot-context metadata so that the math can be reused later, but V1 must not
wire Stage-2 behavior, Stage-2 configs, rollout training, inference-time
constraints, sampler changes, attention/KV changes, visual probes, or extra
model forwards.

## Rationale

The goal is to isolate whether a conservative coordinate-slot repulsion signal
helps the Stage-1 SFT compact-output regime. Pulling Stage-2 into the first
implementation would change the research question, increase compatibility
surface area, and make ablation results harder to interpret.

## Consequence

- Coord-repel config leaves should live with the Stage-1 detection
  teacher-forcing configs, not Stage-2 training configs.
- Loss wiring should target the Stage-1 teacher-forcing objective path.
- Metrics should make Stage-1 coord-slot eligibility, activation, and skip
  reasons visible before any larger training run.
- Stage-2 reuse is a future design question, not part of V1 acceptance.

## Evidence

- Scope: `none-yet`
- Handles:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `configs/stage1/detection_teacher_forcing/`
  - `src/training/objectives/teacher_forcing.py`

## Decision: Reuse Teacher-Forcing Infrastructure And Support Packing

Coord-repel V1 should reuse the existing Stage-1 teacher-forcing target IR,
objective runner, role vocab, batch-extras, and supervision-row resolution
infrastructure. It should not introduce a coord-repel-specific trainer path,
sampler, packed-batch format, or extra model forward.

Packing compatibility is a V1 requirement. The current no-packing policy for
latest Stage-1 teacher-forcing may be rebuilt or broken, provided the
replacement implements and validates exact atom-position remapping for packed
batches.

## Rationale

The existing `teacher_forcing` path already owns the relevant coordinate-token
rows and has a dormant `objective.target_ir.exact_packing_mapping.enabled`
config concept. Coord-repel should extend that path instead of duplicating it.

The current runtime rejects `training.packing=true`,
`training.eval_packing=true`, `packing.static_packing=true`, and
`packing.padding_free_packed=true` for latest teacher-forcing because exact
atom-position packing mapping is not implemented yet. That is the policy to
replace, not a reason to keep coord-repel unpacked.

## Consequence

- V1 implementation must include a packed teacher-forcing sidecar remap plan or
  it is incomplete.
- Coord-repel must consume teacher-forcing atoms after any packed-position
  remapping, so its `logit_position`, `target_position`, `batch_index`, and
  sample provenance match the physical logits tensor.
- Tests must cover packed and unpacked teacher-forcing batches before larger
  coord-repel training claims.
- Existing no-packing docs/config comments will need updates if the packing
  path becomes validated behavior.

## Evidence

- Scope: `none-yet`
- Handles:
  - `configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml`
  - `docs/data/PACKING.md`
  - `src/detection/runtime.py`
  - `src/detection/dataset.py`
  - `src/detection/packing.py`
  - `src/trainers/metrics/teacher_forcing.py`

## Decision: Align Packed Teacher Forcing With ms-swift And FlashAttention

Stage-1 packed teacher forcing should align with the existing ms-swift packing
and padding-free training contract rather than inventing a CoordExp-specific
attention path.

The intended packed runtime is:

```text
static pack plan
  -> one packed training row per device forward
  -> ms-swift padding_free packed collator
  -> reset position_ids per original sample inside the packed row
  -> Transformers flash_attention_2 varlen path
  -> one merged teacher-forcing IR per packed row
  -> coord-repel consumes remapped atoms from that merged IR
```

Packed raw examples such as `A+B+C` are one physical forward row, but remain
causally isolated ordinary SFT examples through reset `position_ids` /
FlashAttention varlen boundaries. CoordExp should not treat `A+B+C` as one
continuous causal story unless a future experiment explicitly changes that
training meaning.

## Rationale

Local ms-swift already treats `packing` as implying `padding_free`, and its
template `packing_row` concatenates packed samples while resetting
`position_ids`. Local Transformers routes reset-position packed rows through the
FlashAttention varlen path when `flash_attention_2` is active. Matching that
surface maximizes FlashAttention v2 efficiency and avoids model attention
monkeypatches.

The necessary CoordExp work is not a new attention implementation. It is a
packed teacher-forcing layout/remap layer that preserves sample offsets,
rewrites atom positions, and merges per-sample teacher-forcing IRs into the
single physical packed row seen by the objective runner.

## Consequence

- V1 should set or require `model.attn_impl: flash_attention_2` whenever
  packed teacher-forcing is enabled.
- V1 should use `training.packing=true`, `training.packing_mode=static`,
  `per_device_train_batch_size=1`, and an `effective_batch_size` counted in
  packed-sequence units.
- `packing.static_packing` and `packing.padding_free_packed` should be treated
  as aligned for this surface, not competing modes.
- The collator-side sidecar path needs a first-class packed layout describing
  source sample ids, source lengths, source offsets, and packed row ranges.
- `TeacherForcingTargetIREnricher` should remap and merge packed sidecars using
  that layout instead of rejecting packed batches.
- Coord-repel must remain a consumer of the remapped teacher-forcing atoms, not
  a separate owner of packing offsets or FA2 metadata.

## Evidence

- Scope: `none-yet`
- Handles:
  - `/data/ms-swift/swift/arguments/sft_args.py`
  - `/data/ms-swift/swift/template/base.py`
  - `/data/ms-swift/swift/dataset/packing.py`
  - `/root/miniconda3/envs/ms/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py`
  - `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py`
  - `docs/standards/upstream/FLASH_ATTENTION.md`
  - `src/data_collators/enrichers.py`

## Decision: CoordExp Owns Packed Sidecar Layout In The Collator Wrapper

Packed teacher-forcing sidecar layout should be built in CoordExp's
batch-extras collator wrapper, not by modifying ms-swift's generic
`Template.packing_row()`.

The CoordExp wrapper already receives the raw packed batch structure such as
`[[A, B, C]]` before/around the ms-swift collator call. It should derive a
first-class packed layout from that raw structure, call the existing ms-swift
collator, verify the packed tensor surface matches the layout, and then remap
and merge teacher-forcing sidecars into one packed-row IR.

## Rationale

ms-swift should remain the owner of generic packing, padding-free collation,
reset `position_ids`, multimodal collation, and Transformers/FlashAttention
compatibility. CoordExp should own only CoordExp-specific sidecars and research
semantics: teacher-forcing atoms, sample provenance, coord-repel slot context,
and metric ownership.

Putting CoordExp offset metadata into ms-swift would couple a generic upstream
template primitive to one research stack's sidecars. Computing the layout in
CoordExp keeps the interface local, testable, and easier to evolve with
teacher-forcing/coord-repel.

## Consequence

- Refactor the current batch-extras collator/enricher shape as needed so packed
  layout is a real local interface instead of scattered ad hoc checks.
- The packed layout should record source sample ids, source lengths, source
  offsets, packed row ranges, and enough provenance to diagnose atom-position
  mismatches.
- `TeacherForcingTargetIREnricher` should stop fail-fast rejecting packed
  sidecars once exact remapping is implemented.
- ms-swift `Template.packing_row()` should be treated as an upstream-compatible
  dependency surface, not a place for CoordExp-specific sidecar code.
- Refactors are allowed when they make the packed teacher-forcing layout deeper
  and safer; unrelated churn remains out of scope.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/data_collators/batch_extras_collator.py`
  - `src/data_collators/enrichers.py`
  - `src/datasets/wrappers/packed_caption.py`
  - `/data/ms-swift/swift/template/base.py`

## Decision: Emit One Merged Teacher-Forcing IR Per Packed Row

Packed teacher-forcing collation should emit one merged
`TeacherForcingTargetIR` for each physical packed row. It should not pass
multiple per-source IRs plus a layout map downstream for the objective runner to
interpret.

For a packed row such as `A+B+C`, the collator-side remap layer should apply
the source offsets, set atom `batch_index` to the physical packed row, rewrite
`target_position` and `logit_position`, and merge the remapped atoms into one
packed-row IR.

## Rationale

The objective runner and teacher-forcing objective already operate on physical
logit rows. Emitting a merged/remapped IR keeps the loss path simple: every atom
already names the correct physical `batch_index`, `target_position`, and
`logit_position`.

Keeping multiple source IRs and making the objective interpret offsets would
leak packing concerns into objective math and coord-repel. Packing should be
resolved before loss computation.

## Consequence

- The collator-side remap layer owns packed source offsets.
- The teacher-forcing objective stays row-local and offset-free.
- Coord-repel consumes remapped coordinate atoms and does not know whether the
  source batch was packed or unpacked.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/ir.py`
  - `src/trainers/metrics/teacher_forcing.py`
  - `src/training/objectives/teacher_forcing.py`
  - `src/data_collators/enrichers.py`

## Decision: Keep IR As Transport, Deepen Layout/Remap Interfaces

`TeacherForcingTargetIR` remains acceptable as the name of the immutable
teacher-forcing supervision carrier, but it should not become a catch-all class
for packing, coord-repel, or attention-runtime concerns.

The current IR class is intentionally lightweight: `schema_version`, `atoms`,
and `metadata`. That is a reasonable transport shape, but too shallow to carry
the new packed-remap complexity by itself. The design should add typed helper
interfaces for packed layout and IR remapping rather than hiding more behavior
inside untyped `metadata` or `provenance` dictionaries.

## Rationale

The useful abstraction is not "an IR object that knows everything." The useful
abstraction is a small set of explicit pipeline artifacts:

- a teacher-forcing target carrier (`TeacherForcingTargetIR`);
- a packed physical-row layout;
- a remapper that turns per-source IRs plus layout into packed-row IRs;
- objective code that consumes only already-remapped atoms.

This preserves compatibility with the existing teacher-forcing objective while
creating a deeper interface where the new complexity actually lives.

## Consequence

- Avoid putting packing offsets, FA2 details, or coord-repel negative-selection
  logic directly on `TeacherForcingTargetIR`.
- Prefer typed dataclasses/functions over expanding `metadata: Mapping[str,
  Any]` and atom `provenance: Mapping[str, Any]` for required runtime fields.
- Treat `metadata`/`provenance` as audit/debug payloads unless a field is
  deliberately promoted to a typed interface.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/ir.py`
  - `src/training/supervision/distributions.py`
  - `src/training/objectives/teacher_forcing.py`

## Exploration Finding: Architecture Pressure-Test

The current difficulty is not primarily the `TeacherForcingTargetIR` name or
class shape. The harder issue is ownership split across dataset encoding,
static packing, batch-extras collation, semantic sidecars, runtime config
guards, and teacher-forcing objective validation.

The smallest coherent implementation center is a collator-owned packed layout
and teacher-forcing remap contract. That layer should be built once per
collated batch, validated against the physical packed tensors, and used by
sidecar enrichers that need source-to-packed-row offsets.

## Architecture Candidates

1. `PackedBatchLayout` / remap layer.
   Build one explicit layout from the raw packed batch plus the collated tensor
   batch. Use it to merge per-source teacher-forcing IRs into one packed-row IR
   per physical row. This is the recommended implementation center.

2. Teacher-forcing context helpers.
   Keep `TeacherForcingTargetIR` as transport and add typed helper code for
   packing remap and coord-repel slot context. Coord-repel should consume
   already-remapped atoms, not raw sample provenance or attention-specific
   offsets.

3. Runtime/config gate simplification.
   Replace blanket teacher-forcing packing rejections with a validated condition:
   static packing, padding-free FA2, per-device batch size one, compact-full
   teacher-forcing IR, exact remap contract enabled, logits-to-keep disabled,
   and encoded sample cache disabled unless separately validated.

4. Sidecar convergence.
   The codebase already has semantic sidecar types, but the active collator path
   still carries loose raw keys and `BatchExtras`. This can be cleaned after the
   packed-layout layer proves out, but should not be the first refactor.

## Decision: Use Packed Layout Plus Merged Row IR As V1 Design Center

The approved implementation center for V1 is a `PackedBatchLayout`-style
collator/remap layer plus one merged `TeacherForcingTargetIR` per physical
packed row.

The packed layout layer should own source sample ranges, physical row offsets,
pack membership, and validation against the collated tensor batch. The
teacher-forcing remap step should consume that layout, offset per-source atoms,
set atom `batch_index` to the physical packed row, and emit one already-remapped
IR for each packed row.

## Rationale

This keeps packing concerns out of objective math and coord-repel. The
teacher-forcing objective can continue to consume physical logit rows, while
coord-repel receives coordinate atoms that already point at the correct packed
positions.

The rejected alternatives were:

- stuffing packed-offset behavior into `TeacherForcingTargetIR` itself;
- passing multiple per-source IRs plus offsets into the objective;
- modifying ms-swift or FlashAttention integration for CoordExp-specific
  sidecars.

Those alternatives either make the IR too broad, leak packing into loss code, or
couple CoordExp research semantics to upstream generic collation.

## Consequence

- The next implementation plan should start from the packed layout/remap layer.
- `TeacherForcingTargetIR` stays a lightweight transport carrier.
- Coord-repel should be implemented against remapped coordinate atoms and should
  not own packing offsets.
- Runtime/config guards can be relaxed only after remap validation and loss
  equivalence tests exist.
- Sidecar convergence remains a follow-up unless implementation proves the
  current raw-key/semantic-sidecar split blocks the packed layout.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/data_collators/batch_extras_collator.py`
  - `src/data_collators/enrichers.py`
  - `src/datasets/wrappers/packed_caption.py`
  - `src/training/teacher_forcing/ir.py`
  - `src/training/objectives/teacher_forcing.py`
  - `src/trainers/metrics/teacher_forcing.py`

## Decision: Split Generic Packed Layout From Teacher-Forcing Remap

V1 should use two small modules rather than concentrating all packed
teacher-forcing behavior in one collator/enricher file.

- `src/data_collators/packed_layout.py` owns generic physical packed layout:
  packed rows, source segments, sample ids, source ranges, offsets, lengths,
  and validation against the collated tensor batch.
- `src/training/teacher_forcing/packing.py` owns teacher-forcing-specific
  remap and merge semantics: rewriting atom `batch_index`, `target_position`,
  and `logit_position`; preserving useful provenance; and emitting one merged
  `TeacherForcingTargetIR` per physical packed row.
- `TeacherForcingTargetIREnricher` remains a thin integration layer that calls
  the layout builder and teacher-forcing remapper.

## Rationale

The physical layout is not teacher-forcing-specific. Token-type sidecars,
dataset meta, diagnostics, and future sidecars may also need the same packed
source ranges. Keeping layout generic avoids coupling that contract to one
objective.

The atom remap is teacher-forcing-specific. It should live near the IR,
validation, and objective code so the semantics of atom positions remain
auditable.

The rejected alternative was putting everything in `src/data_collators/`.
That would be faster in the first patch, but it would make the collator the
owner of research-objective semantics and likely recreate the readability
problem this refactor is meant to reduce.

## Consequence

- The implementation plan should introduce both modules in the first slice.
- Collator code may know that teacher-forcing has a remapper, but should not
  implement atom-position semantics inline.
- Tests should cover the generic layout independently from teacher-forcing IR
  remap behavior.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/data_collators/packed_layout.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/data_collators/enrichers.py`
  - `src/training/teacher_forcing/ir.py`

## Decision: Infer Remap Contract Internally For V1

V1 should not add a new public config key for the packed teacher-forcing remap
contract. The contract should be inferred internally when Stage-1
teacher-forcing runs with static packing.

In practice:

- `objective.id: teacher_forcing`
- `training.packing: true`
- `training.packing_mode: static`

imply the effective remap contract `teacher_forcing_atoms_v1`.

## Rationale

The goal for this version is to stay light and fast to implement. A public key
such as `packing.remap_contract: teacher_forcing_atoms_v1` is auditable, but it
adds another user-facing knob that can contradict `training.packing`,
top-level `packing.*`, or the old `objective.target_ir.exact_packing_mapping`
shape.

V1 should reduce the number of policy surfaces, not rename the existing
fragmentation.

## Consequence

- Schema/runtime code should validate the inferred contract rather than require
  users to author it.
- Docs may describe the effective contract, but configs should not need to set
  it.
- The old `objective.target_ir.exact_packing_mapping.enabled` path should not be
  expanded as the V1 user-facing control.
- Tests should assert the inferred behavior and negative cases where the
  contract cannot be satisfied.

## Evidence

- Scope: `none-yet`
- Handles:
  - `configs/stage1/detection_teacher_forcing/`
  - `src/config/schema.py`
  - `src/detection/runtime.py`
  - `src/data_collators/enrichers.py`
  - `src/training/teacher_forcing/packing.py`

## Decision: Support Packed Stage-1 Eval Alongside Packed Training

V1 should support packing for both Stage-1 teacher-forcing training and
Stage-1 teacher-forcing eval. Stage-1 eval is not rollout; it is the same
teacher-forced forward-pass path as training, with potentially different
monitoring and metric aggregation.

## Rationale

The packed-remap contract is a forward-pass tensor contract. If train and eval
share the same teacher-forcing collator/objective path, disabling eval packing
would create an artificial divergence and reduce the usefulness of train/eval
comparisons.

The valid distinction is metric accounting, not remap mechanics. Eval may need
clearer sample-level counters, monitor dumps, or debug payloads, but those
should consume the same remapped packed-row IR produced for training.

## Consequence

- The implementation should not treat Stage-1 eval as rollout.
- Train and eval packed teacher-forcing should share the same packed layout and
  IR remap code.
- Eval-specific metrics must preserve source sample accounting from the packed
  layout, even though loss computation consumes physical packed rows.
- Tests should include at least one eval-style forward-only packed path or
  metric accounting check, not only train loss equivalence.

## Evidence

- Scope: `none-yet`
- Handles:
  - `configs/stage1/detection_teacher_forcing/`
  - `src/data_collators/packed_layout.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/trainers/metrics/teacher_forcing.py`
  - `src/training/objectives/teacher_forcing.py`

## Decision: Coord-Repel V1 Compares Same-Sample Same-Role Object Slots

Status: superseded by `Decision: Reframe Coord-Repel As Coordinate-Basin
Calibration` and `Decision: Use X1/Y1 Basin-Mass Bands For V1`.

Coord-repel V1 should repel coordinate-token predictions only between
different object slots from the same original sample and the same coordinate
role.

The comparison scope is:

- `x1` atoms against other `x1` atoms;
- `y1` atoms against other `y1` atoms;
- `x2` atoms against other `x2` atoms;
- `y2` atoms against other `y2` atoms;
- different object slots only;
- same source sample only, even when several samples share one physical packed
  row;
- teacher-forced coordinate logits only, not rollout predictions.

V1 should not repel across coordinate roles, across packed source samples, or at
geometry/box level.

## Rationale

Same-role token repulsion is the smallest tensor-level version that directly
targets coordinate collapse while keeping grouping auditable. It avoids treating
`A+B+C` packed rows as one image and avoids introducing box-level geometry
claims before the token-logit mechanism is validated.

Using teacher-forced logits keeps the mechanism aligned with the Stage-1 SFT
forward path. Rollout behavior remains out of scope.

## Consequence

- Coord-repel context needs source sample id, object slot id, coordinate role,
  physical batch row, logit position, and target token id for each coordinate
  atom.
- Packed remap must preserve source sample identity after atom positions are
  shifted into the physical packed row.
- Tests must include a packed row with at least two source samples and verify
  that coord-repel pairs are formed within each source sample, not across them.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/ir.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/objectives/teacher_forcing.py`
  - `src/data_collators/packed_layout.py`

## Decision: Coord-Repel V1 Uses Margin Ranking Over Competing Target Tokens

Status: superseded by `Decision: Reframe Coord-Repel As Coordinate-Basin
Calibration` and `Decision: Use X1/Y1 Basin-Mass Bands For V1`.

Coord-repel V1 should use a local margin-ranking / unlikelihood-style loss over
competing coordinate target tokens for same-sample, same-role, different-object
pairs.

For a valid pair `(i, j)`:

- at atom `i`'s logit position, prefer `i.target_token_id` over
  `j.target_token_id`;
- at atom `j`'s logit position, prefer `j.target_token_id` over
  `i.target_token_id`.

A representative symmetric form is:

```text
relu(margin - logp_i(target_i) + logp_i(target_j))
+ relu(margin - logp_j(target_j) + logp_j(target_i))
```

reduced over valid coord-repel pairs.

## Rationale

This directly targets coordinate-token confusion between object slots while
remaining local to the teacher-forced forward pass. It avoids rollout decoding,
box parsing, or value-distance assumptions in V1.

The rejected alternative is geometry/value-distance repulsion over decoded
coordinates. That may be useful later, but it introduces extra token-to-value
and box-level semantics before the simpler logit mechanism is validated.

## Consequence

- Coord-repel needs deterministic valid-pair construction and explicit
  denominator semantics.
- The formula should run in a numerically safe dtype for `log_softmax` and
  margin arithmetic.
- Logs must include raw coord-repel loss, weighted contribution, pair counts,
  skipped-pair counts, and the configured margin/weight.
- Calibration must be designed separately so the auxiliary term does not
  dominate or vanish relative to standard CE.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/ir.py`
  - `src/training/teacher_forcing/packing.py`

## Decision: Reframe Coord-Repel As Coordinate-Basin Calibration

Coord-repel V1 is a training-time, loss-only coordinate-basin calibration term.
It must not modify KV cache behavior, attention, decoding, sampling, or
inference policy. It only reads teacher-forced coordinate-slot logits during
training/eval forward passes.

For a current coordinate slot, the intended pressure is:

```text
probability mass near the current GT coordinate basin
>
probability mass near previously emitted / seen wrong coordinate basins
```

For a slot such as `y1`:

```text
B+ = a small band around the current GT y1, for example y1 +/- 4 bins
B- = previous/seen boxes' y1 bands
     intersected with the current model top-k wrong high-probability coordinate
     tokens for this slot
```

The representative loss is:

```text
softplus(margin + log P(B-) - log P(B+))
```

where `P(B+)` and `P(B-)` are probability masses under the current coordinate
slot's logit distribution.

## Rationale

This supersedes the broader pairwise target-token margin framing. The target is
not generic object diversity. The target is a local coordinate attractor: the
model should not treat an old/seen coordinate band as the next object's
coordinate well when its own logits are already assigning high probability to
that wrong basin.

The design is conservative because not every previous box is repelled. A
previous/seen box contributes negative pressure only when its coordinate band
intersects the model's current high-probability wrong top-k region. That avoids
penalizing legitimate overlapping objects, neighboring objects, and extent
slots where broad repulsion can overcorrect.

## Consequence

- Coord-repel should operate on probability mass over coordinate-token bands,
  not only on individual competing target tokens.
- Negative bands must be model-activated: `previous/seen coordinate band`
  intersected with current wrong high-probability top-k coordinate tokens.
- The objective should keep hard CE dominant and act only as a conservative
  basin correction term.
- No inference-time behavior changes are part of V1.
- Pair construction must preserve source-sample identity under packing.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/teacher_forcing/ir.py`
  - `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md`
  - `progress/diagnostics/2026-04-21_raw_text_coordinate_mechanism_findings.md`

## Decision: Use X1/Y1 Basin-Mass Bands For V1

Coord-repel V1 should apply only to `x1` and `y1` coordinate slots. It should
compare probability mass over positive and negative coordinate bands, not
individual competing target tokens.

For each eligible current `x1` or `y1` atom:

```text
B+ = current GT coordinate band
     fixed radius: +/- 4 bins

B- = union(previous/seen coordinate bands for the same coordinate role)
     fixed radius: +/- 4 bins
     intersected with current wrong high-probability coordinate top-k
```

The initial top-k activation scope is:

```text
top_k = 32
exclude tokens inside B+
exclude the current GT token
intersect remaining top-k tokens with previous/seen bands
```

The V1 loss for a slot is:

```text
softplus(margin + log P(B-) - log P(B+))
```

If `B-` is empty, the slot contributes no coord-repel loss. If `B+` is empty,
the implementation should fail fast because the coordinate target/support
contract is broken.

## Rationale

The Stage-1 duplication diagnosis points to early `x1/y1` coordinate escape as
the primary separator. `x2/y2` are extent/closure slots and have higher
overcorrection risk, especially for legitimate overlapping, aligned, or
similarly sized objects.

Using fixed small bands keeps V1 light and auditable. The top-k intersection
makes negative pressure model-activated: previous/seen coordinates matter only
when the current forward pass is actually assigning them high probability.

## Consequence

- V1 coord-repel context must provide previous/seen coordinate bands per source
  sample and role.
- Packed remap must preserve source-sample identity so previous/seen bands do
  not cross physical packed-row sample boundaries.
- Metrics should log `B+` mass, `B-` mass, active slot counts, empty-negative
  counts, and top-k intersection counts.
- `x2/y2` support should remain a later extension, not hidden behavior in V1.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/teacher_forcing/ir.py`
  - `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md`
  - `progress/diagnostics/2026-04-21_raw_text_coordinate_mechanism_findings.md`

## Decision: B- Uses All Prior Teacher-Forced Objects, Activated By Top-K

Status: superseded by `Decision: Use Last-And-Same-Desc As Balanced Negative
Source`.

For a current object slot `k`, coord-repel V1 should build candidate negative
coordinate bands from all previous objects in the same source sample's
teacher-forced rendered order:

```text
seen_boxes = objects with object_slot_index < k
```

It should not restrict the candidate set to same-desc objects. It should also
not include future GT objects, decoded rollout predictions, model-generated
boxes, or objects from other packed source samples.

The actual penalized `B-` remains conservative:

```text
B- = union(all prior same-role coordinate bands)
     intersected with current wrong high-probability coordinate top-k
```

Only the top-k-activated intersection contributes loss.

## Rationale

The prefix-visible coordinate basins are all prior teacher-forced boxes, not
only same-desc boxes. Restricting to same-desc would make the loss miss
geometry-driven attraction to earlier coordinates when semantic labels differ
or when the failure is dominated by coordinate history rather than text.

The top-k intersection is the safety valve: a prior object is not penalized
unless the current forward pass is actually assigning high probability to its
coordinate band.

## Consequence

- Coord-repel context must preserve teacher-forced object order and object slot
  index for each coordinate atom.
- Packed remap must preserve source-sample boundaries so prior-object bands do
  not leak across packed samples.
- Metrics should distinguish candidate prior bands from activated `B-` tokens.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/ir.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/objectives/teacher_forcing.py`
  - `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md`
  - `progress/diagnostics/2026-04-21_raw_text_coordinate_mechanism_findings.md`

## Decision: Use Last-And-Same-Desc As Balanced Negative Source

Coord-repel V1 should use `last_and_same_desc` as the balanced/main negative
source:

```text
B_seen =
  band(last previous object for the same role)
  union bands(previous objects with matching desc/category key for the same role)
```

`all_prior` remains available as a stress/diagnostic source mode, not the
balanced default. The useful source modes are:

```text
last_only
same_desc
last_and_same_desc
all_prior
```

The activated negative band is still:

```text
B- = wrong_topk outside B+
     intersected with B_seen
```

## Rationale

The prior `all_prior` default is not conservative enough for crowded COCO-style
scenes. Many unrelated objects can share similar `x1` or `y1` coordinates
because of shelves, rows, vertical alignment, or repeated layout structure.
Using all prior bands can turn `B_seen` into a broad coordinate minefield.

`last_and_same_desc` keeps the two main intended attractors:

- immediate prefix drag from the most recent object;
- same-desc/history attractors that often drive duplicate bursts.

Keeping `all_prior` as a stress mode preserves a way to test cross-description
geometry drag without making it the default experiment claim.

## Consequence

- `negative_source` should be public or at least resolved/logged because it
  changes research meaning.
- Balanced/main config should use `negative_source: last_and_same_desc`.
- Stress diagnostics may use `negative_source: all_prior`.
- Canonical-failure activation probes should compare source modes before
  promoting any non-default source.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/coord_repel.py`
  - `src/training/objectives/teacher_forcing.py`
  - `docs/superpowers/specs/2026-06-12-coord-repel-v1-design.md`

## Decision: Compute B- Top-K Within Coordinate Vocabulary Only

Coord-repel V1 should compute wrong top-k over the coordinate-token vocabulary,
not the full model vocabulary.

For a current coordinate slot:

```text
coord_logits = logits_at_slot[coord_token_ids]
coord_probs = softmax(coord_logits)
top_k_wrong = topk(coord_probs, k=32), excluding B+
B- = top_k_wrong intersected with prior same-role coordinate bands
```

## Rationale

The target mechanism is coordinate-basin attraction. Full-vocabulary top-k would
mix coordinate tokens with punctuation, schema tokens, description tokens, and
special tokens. That would make `B-` sensitive to unrelated surface-format
competition and weaken the objective's interpretation.

Coordinate-vocabulary-only top-k matches the existing coordinate-locality
diagnostics, which evaluate probabilities over `<|coord_0|>` through
`<|coord_999|>`.

## Consequence

- Coord-repel implementation needs the resolved coordinate token id list and a
  stable token-id-to-coordinate-bin map.
- `P(B+)`, `P(B-)`, and top-k diagnostics should be reported in the conditional
  coordinate vocabulary distribution.
- Full-vocabulary CE remains the main objective; coord-repel is an auxiliary
  coordinate-basin term.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/ir.py`
  - `src/coord_tokens/codec.py`
  - `src/trainers/losses/sft_gaussian_coord_soft_ce.py`

## Decision: Use Conditional Coordinate-Vocabulary Mass For B+/B-

Coord-repel V1 should compute `P(B+)` and `P(B-)` within the conditional
coordinate-token distribution, not the full model vocabulary distribution.

For a coordinate slot:

```text
coord_log_probs = log_softmax(logits_at_slot[coord_token_ids])
log_p_pos = logsumexp(coord_log_probs[B+])
log_p_neg = logsumexp(coord_log_probs[B-])
loss = softplus(margin + log_p_neg - log_p_pos)
```

The computation should run in fp32 and handle empty support explicitly. Empty
`B-` means no coord-repel term for that slot. Empty `B+` is a target/support
contract error.

## Rationale

The mechanism asks whether the model's coordinate distribution is attracted to
the current GT basin or to a previous/seen coordinate basin. Full-vocabulary
probability mass would entangle this signal with punctuation, JSON/schema
tokens, text tokens, or other non-coordinate competition.

Full-vocabulary format correctness remains the job of the main CE objective.
Coord-repel should remain a coordinate-basin auxiliary term.

## Consequence

- Coord-repel must build a stable coordinate-token id list and coordinate-bin
  index map.
- Logs for `P(B+)`, `P(B-)`, top-k, and active masses should all be clearly
  labeled as coordinate-vocabulary-conditional.
- The objective should not mask or alter the main full-vocabulary CE labels.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/coord_tokens/codec.py`
  - `src/trainers/losses/sft_gaussian_coord_soft_ce.py`

## Decision: Reduce By Active Slots And Use Visible Initial Weight

Coord-repel V1 should reduce over active coordinate slots: `x1` or `y1` slots
where `B-` is non-empty after top-k intersection.

```text
slot_loss = softplus(margin + log_p_neg - log_p_pos)
raw_loss = mean(slot_loss over active slots)
weighted_loss = coord_repel.weight * raw_loss
total_loss = main_teacher_forcing_loss + weighted_loss
```

Initial V1 strength:

```yaml
coord_repel:
  weight: 0.05
  margin: 0.25
```

## Rationale

Reducing over active slots keeps the loss tied to actual basin-attraction
events rather than all coordinate atoms, all objects, all previous boxes,
sequence length, or physical packed rows. Empty `B-` slots are not failure
opportunities and should not dilute the term.

The balanced/main weight is `0.05`. This keeps the module visible while reducing
the risk that it hijacks coordinate learning. A stress-smoke config may use
`weight: 0.1` to confirm that the mechanism can move metrics, but the primary
ablation should use `0.05` unless telemetry shows the term is inactive.

The margin remains conservative: `0.25` in log-probability space asks `P(B+)`
to exceed `P(B-)` by roughly `exp(0.25)`, not by an aggressive ratio.

## Consequence

- Logs must expose both raw and weighted coord-repel contributions so the
  effective strength can be audited.
- Early launch checks should compare weighted coord-repel against the main
  teacher-forcing loss and gradient norm.
- If the term dominates or destabilizes training, the first fallback is a
  weight sweep, not changing the core `B+`/`B-` definition.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/teacher_forcing/ir.py`

## Decision: Do Not Add Positive-Band Anchor In V1

Coord-repel V1 should not add a separate positive-band anchor term such as:

```text
L_pos = -log P(B+)
```

The only V1 differentiable coord-repel term is the active negative-basin
calibration loss:

```text
softplus(margin + log P(B-) - log P(B+))
```

with no contribution when `B-` is empty.

## Rationale

A positive-band anchor is plausible, but it changes the experiment. It adds a
local target-basin shaping objective closer to SoftCE/locality supervision,
while the current V1 claim is anti-basin calibration: penalize the prior/seen
coordinate attractor only when the model's current coordinate distribution is
actively assigning mass to that basin.

The main teacher-forcing CE already anchors the exact GT token. If V1 also adds
`-log P(B+)`, later behavior changes would be harder to attribute to
coord-repel rather than extra local positive shaping.

## Consequence

- Empty `B-` slots remain skipped, not converted into positive-anchor slots.
- Metrics must still log `P(B+)`, active-slot rate, and empty-`B-` rate so an
  inactive module is visible.
- If canonical-failure probes show sparse activation, the first response should
  be to inspect negative-source alignment, not silently add a positive anchor.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/coord_repel.py`
  - `src/training/objectives/teacher_forcing.py`
  - `docs/superpowers/specs/2026-06-12-coord-repel-v1-design.md`

## Decision: Expose Minimal Coord-Repel Config Knobs In V1

Coord-repel V1 should expose only the minimal experiment levers:

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

The following V1 mechanism choices should remain internal constants, surfaced
through logs/resolved metadata rather than as user-authored knobs:

```text
roles = [x1, y1]
positive_radius = 4
negative_radius = 4
distribution = coordinate_vocab_conditional
reduction = active_slot_mean
```

## Rationale

`enabled`, `weight`, `margin`, `top_k`, and `negative_source` are the useful
first-pass experiment controls. `negative_source` is included because it changes
research meaning: the difference between `last_and_same_desc` and `all_prior`
is not merely a numeric tuning choice. The other values define the mechanism
itself and should remain internal in V1.

The hidden constants must still be observable. They should be logged in resolved
config, run metadata, or metric payloads so later interpretation is not
ambiguous.

## Consequence

- Schema/config work should add only the five public keys for V1.
- Implementation should log the internal constants.
- Any later expansion to `x2/y2`, different radii, same-desc filtering, or
  alternative reductions should be a deliberate follow-up, not incidental
  configurability.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/config/schema.py`
  - `configs/stage1/detection_teacher_forcing/`
  - `src/training/objectives/teacher_forcing.py`

## Decision: Derive Coord-Repel Context From Remapped IR

Coord-repel V1 should derive its slot context from the already-remapped
`TeacherForcingTargetIR`, not from a new dataset or collator sidecar.

The implementation should introduce a typed extractor that promotes the small
set of required IR/provenance fields into explicit runtime slots:

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

`object_order_index` should be derived from
`TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]`.
`object_index` and `object_instance_id` may come from atom provenance, but only
inside the typed extractor/validator.

## Rationale

The remapped IR is already the source of truth for teacher-forcing supervision.
It contains the physical packed-row positions after remap and already carries
the selected object order. A separate coord-repel sidecar would duplicate
object/order/coordinate state and create another drift surface.

Keeping provenance access inside one typed extractor avoids spreading untyped
dictionary reads through objective code while still keeping V1 light.

## Consequence

- If coord-repel is enabled and required IR fields are missing, fail fast with a
  clear contract error.
- Coord-repel should not read raw dataset objects at loss time.
- Packed remap must preserve `stable_sample_id`, selected object order, atom
  provenance, and physical positions.
- Tests should cover missing/malformed provenance and packed-row remap cases.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/ir.py`
  - `src/detection/teacher_forcing/target_builder.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/training/objectives/teacher_forcing.py`

## Decision: Keep Coord-Repel As One Small Teacher-Forcing Helper Module

Coord-repel V1 should live in one focused teacher-forcing helper module:

```text
src/training/teacher_forcing/coord_repel.py
```

That module should own:

- `CoordRepelConfig` runtime normalization if needed;
- `CoordRepelSlot`;
- slot extraction from remapped `TeacherForcingTargetIR`;
- `B+` / `B-` construction;
- the pure tensor loss function;
- a small result object carrying loss sums, active counts, and diagnostics.

`src/training/objectives/teacher_forcing.py` should remain the integration
point that calls this helper after the main teacher-forcing loss has prepared
logits, labels, and sidecars.

## Rationale

Putting coord-repel directly into the objective file would make the objective
harder to read and harder to test. Splitting it across several modules would be
premature for V1. One pure helper module keeps the surface small while giving
the formula and slot extraction dedicated tests.

This also keeps `src/data_collators/packed_layout.py` and
`src/training/teacher_forcing/packing.py` focused on physical packing/remap,
not on the auxiliary loss mechanism.

## Consequence

- No new trainer subclass or ms-swift modification is needed for coord-repel.
- Objective code should pass only tensors, remapped IRs, coordinate-token ids,
  and config values into the helper.
- If the helper grows beyond V1, later split points are context extraction and
  tensor loss math.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/coord_repel.py`
  - `src/training/objectives/teacher_forcing.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/data_collators/packed_layout.py`

## Decision: Preserve Active-Slot Mean Across Packing, DDP, And Accumulation

Coord-repel should return both numerator and denominator:

```text
loss_sum = sum(slot_loss over active x1/y1 slots)
active_slots = count(active x1/y1 slots with non-empty B-)
raw_loss = loss_sum / active_slots
```

The integration must preserve active-slot mean semantics across static packing,
gradient accumulation, and distributed training. It should not average
microbatch means when active-slot counts differ.

## Rationale

The Gaussian packed-SFT fix showed that auxiliary coordinate losses can look
reasonable while being scaled incorrectly under packing/accumulation. Coord-repel
has the same risk because active slots vary by object count, prior-object
count, model top-k intersections, and packing layout.

The intended unit is the active coordinate-basin correction event, not
microbatch, physical row, packed sequence, object, or raw coordinate token.

## Consequence

- The result object should expose `loss_sum`, `active_slots`, `raw_loss`, and
  `weighted_loss`.
- Tests must include unequal active-slot counts across accumulation samples and
  verify the accumulated result equals global active-slot mean.
- Logs should include both raw and weighted values so the `0.1` weight is
  auditable.

## Evidence

- Scope: `none-yet`
- Handles:
  - `src/training/teacher_forcing/coord_repel.py`
  - `src/training/objectives/teacher_forcing.py`
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/progress/handoffs/2026-06-11-packed-gaussian-sft-retrain.md`
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/tests/test_sft_gaussian_coord_soft_ce.py`

## Decision: Verification Gates For V1

The first implementation should include narrow tests before any production
training:

1. Pure formula probe:
   deterministic logits where high `B+` mass gives low loss and high `B-` mass
   gives higher loss.
2. Band construction probe:
   `B+` radius, `B-` prior bands, `top_k` intersection, empty `B-` skip, and
   empty `B+` fail-fast.
3. Causal order probe:
   only prior teacher-forced objects contribute to `B-`; future objects do not.
4. Packing probe:
   packed `A+B` rows do not allow `A` prior bands to affect `B`, or vice versa.
5. Accumulation scaling probe:
   unequal active-slot counts reduce by global active-slot mean.
6. Config probe:
   only `enabled`, `weight`, `margin`, `top_k`, and `negative_source` are
   public V1 knobs.
7. Eval-forward probe:
   Stage-1 eval can compute the same forward-only coord-repel diagnostics
   without rollout assumptions.
8. Canonical-failure activation probe:
   known duplicate/prefix-sensitive cases should produce non-empty `B-`,
   interpretable `wrong_top1_in_bneg_rate`, and meaningful
   `log P(B+) - log P(B-)` before production training.
9. Negative-source probe:
   compare `last_only`, `same_desc`, `last_and_same_desc`, and `all_prior` on
   canonical cases before promoting any source mode beyond the balanced default.
10. Box-end template smoke:
    verify that `compact_box_end` inserts exactly one `<|box_end|>` after `y2`,
    uses no newline, and keeps the token trainable/saved.

## Rationale

These tests target the highest-risk silent failures: wrong target scope,
future-object leakage, cross-packed-sample leakage, invisible loss strength,
and incorrect accumulation scaling.

## Consequence

- Do not start a production coord-repel run until these gates pass.
- The first smoke should report active-slot counts and raw/weighted loss before
  interpreting any quality metric.

## Evidence

- Scope: `none-yet`
- Handles:
  - `tests/`
  - `src/training/teacher_forcing/coord_repel.py`
  - `src/training/teacher_forcing/packing.py`
  - `src/data_collators/packed_layout.py`

## Decision: Include Compact Box-End As Independent 2x2 Ablation Axis

The coord-repel study should include the template axis:

```text
compact
compact_box_end
```

`compact_box_end` appends one row-commit token after `y2`:

```text
<object_ref_start>{desc}<box_start>x1 y1 x2 y2<|box_end|><object_ref_start>...
```

Hard constraints:

```text
no newline
no <object_ref_end>
<|box_end|> is the only row-commit token
```

Recommended matrix:

```text
1. compact + no coord-repel
2. compact_box_end + no coord-repel
3. compact + coord-repel
4. compact_box_end + coord-repel
```

## Rationale

`<|box_end|>` tests row-commit structure. Coord-repel tests coordinate-basin
calibration. These are independent mechanisms and should not be conflated.

Without a row-commit token, `y2` carries coordinate completion, instance
completion, coverage update, and transition pressure. A single `<|box_end|>`
token may make that transition more learnable without changing coord-repel.

## Consequence

- Template ablation should be tracked separately from coord-repel loss
  ablation.
- The token must be trainable/saved and must not introduce newline behavior.
- The first implementation plan should treat template work as an independent
  slice if it is implemented in the same branch.

## Evidence

- Scope: `none-yet`
- Handles:
  - `docs/superpowers/specs/2026-06-12-coord-repel-v1-design.md`
  - `configs/stage1/detection_teacher_forcing/`

## Reference Finding: Gaussian SoftCE Calibration Precedent

The 2x2 ablation worktree
`/data/CoordExp/.worktrees/fully-compact-2x2-ablation` provides the closest
local precedent for auxiliary coordinate-loss scaling.

Relevant evidence:

- `progress/handoffs/2026-06-11-packed-gaussian-sft-retrain.md` records a fix
  for packed Gaussian Stage-1 SFT aux-loss scaling so it uses true
  accumulation-window coord-token `token_mean`.
- The checked production config
  `configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed.yaml`
  uses `gaussian_mixture_weight: 0.5`,
  `gaussian_r95_axis_fraction: 0.04`, and
  `gaussian_r95_cap_bins: 8`.
- `src/trainers/losses/sft_gaussian_coord_soft_ce.py` infers per-coordinate
  R95 radii from compact-full `xyxy` quads:
  `floor(min(cap_bins, axis_fraction * axis_length))`, with x roles using box
  width and y roles using box height.
- The same loss computes coordinate log-probabilities in fp32 and reduces by
  valid coordinate-token count, with distributed accumulation-window correction.
- `tests/test_sft_gaussian_coord_soft_ce.py` verifies packed coord positions and
  accumulation-window coord-token mean scaling.
- `progress/diagnostics/2026-05-18_gaussian_softce_a5_a6_coord_logit_locality.md`
  reports that broad/default Gaussian behavior was weak, while CE-anchored
  Gaussian mix-0.2 preserved more of the hard-CE coordinate basin. This is
  `val200` mechanism evidence, not full validation.

Implication for coord-repel:

- Use coord-token/atom-level denominator semantics, not packed-row or sequence
  denominators.
- Keep the first strength setting conservative and log raw versus weighted
  contribution.
- Do not reuse Gaussian smooth-target tolerance as coord-repel's negative-pair
  scope. Gaussian SoftCE is a locality/smoothness mechanism; coord-repel is an
  anti-basin calibration mechanism.
- Do not infer that broad auxiliary coordinate shaping is automatically good;
  A5/A6 suggests exact-token pressure must remain dominant.

## Evidence

- Scope: `val200` for A5/A6 mechanism diagnostics; `smoke` for packed Gaussian
  scaling tests and preflight notes.
- Handles:
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/progress/handoffs/2026-06-11-packed-gaussian-sft-retrain.md`
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/src/trainers/losses/sft_gaussian_coord_soft_ce.py`
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/tests/test_sft_gaussian_coord_soft_ce.py`
  - `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/progress/diagnostics/2026-05-18_gaussian_softce_a5_a6_coord_logit_locality.md`

## Configuration Pressure Points

The current Stage-1 detection teacher-forcing config surface has multiple
overlapping packing owners:

- `training.packing` / `training.eval_packing`;
- top-level `packing.static_packing` / `packing.padding_free_packed`;
- `objective.target_ir.exact_packing_mapping.enabled`.

For V1, the desired authoring surface should be simpler: packing is a stable
runtime contract, while coord-repel is an objective module/ablation knob.
The old `exact_packing_mapping.enabled` knob should become either an internal
validated capability or a concise `packing.remap_contract` value, not another
user-facing switch that can contradict runtime packing.

## CodeGraph Initialization Status

Worktree-local CodeGraph was initialized and synced in
`/data/CoordExp/.worktrees/coord-repel-conservative-design`.

- `codegraph init -i /data/CoordExp/.worktrees/coord-repel-conservative-design`
  reports the worktree is already initialized.
- `codegraph sync /data/CoordExp/.worktrees/coord-repel-conservative-design`
  reports the index is already up to date.
- `codegraph status /data/CoordExp/.worktrees/coord-repel-conservative-design`
  reports 1,172 files, 20,206 nodes, 51,735 edges, and an up-to-date local DB.

The CodeGraph MCP wrapper still reports a warning that its index belongs to
`/data/CoordExp`, even when called with the worktree project path. Until that
wrapper is restarted or reconfigured with `codegraph serve --path
/data/CoordExp/.worktrees/coord-repel-conservative-design --mcp`, use the CLI
from the worktree for CodeGraph-backed exploration.

## Open Design Questions

- Whether the first implementation should also update stable docs/specs, or
  keep changes branch-local until the first smoke proves the mechanism is worth
  promoting.
- Whether `margin=0.25` should be swept after the first balanced `weight=0.05`
  smoke, or held fixed until the loss is numerically validated.
- Whether sidecar convergence should be a follow-up cleanup after packed
  teacher-forcing and coord-repel pass their targeted tests.
