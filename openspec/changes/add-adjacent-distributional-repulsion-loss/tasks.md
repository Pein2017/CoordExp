## 1. OpenSpec Foundation

- [ ] 1.1 Keep the first implementation aligned with the agreed scope:
  - immediate previous object only
  - `same_desc | global` filter toggle
  - primary v1 signal is a size-aware edge-band box-copy term
  - edge-only bands, not a full 2D region
  - Stage-1 wording is GT-context
  - Stage-2 wording is rollout-context
  - decoded-box CIoU companion is deferred
- [ ] 1.2 Keep the delta spec set in sync for:
  - `adjacent-distributional-repulsion-loss`
  - `coord-aux-loss`
  - `teacher-forcing-unified-loss-registry`
  - `stage2-ab-training`
  - `rollout-matching-sft`
  - `trainer-metrics-components`
- [ ] 1.3 Re-validate the change after each artifact pass:
  - `openspec validate add-adjacent-distributional-repulsion-loss --type change --strict --json --no-interactive`

## 2. Coord-Distribution Adjacent Atom

- [ ] 2.1 Define one shared helper for the adjacent-distributional repulsion term.
- [ ] 2.2 Implement the per-slot edge-band overlap computation against the
  immediately previous object's target coord bins.
- [ ] 2.3 Define the edge-band half-width from previous-box size using:
  - `adjacent_repulsion_margin_ratio`
  - minimum one coord bin on each axis
- [ ] 2.4 Keep the taper family normative in v1:
  - linear taper
  - per-edge half-width ratio semantics
- [ ] 2.5 Compose the four slot overlaps into one thresholded box-copy score so
  the loss focuses on rerollout of the same box rather than generic proximity.
- [ ] 2.6 Add focused unit coverage for:
  - overlap rises when current coord mass approaches previous-object edge bins
  - larger previous boxes induce wider edge bands
  - exact no-adjacent-object case is zero contribution
  - partial-edge matches stay below `copy_margin`
  - `same_desc` and `global` filter behavior

## 3. Stage-1 GT-Context Adapter

- [ ] 3.1 Extend the Stage-1 coord auxiliary config surface under
  `custom.coord_soft_ce_w1` with:
  - `adjacent_repulsion_weight`
  - `adjacent_repulsion_filter_mode`
  - `adjacent_repulsion_margin_ratio`
  - `adjacent_repulsion_copy_margin`
- [ ] 3.2 Tighten the Stage-1 `coord_soft_ce_w1` parser so unsupported adjacent
  repulsion keys fail fast.
- [ ] 3.3 Thread the Stage-1 loss path through a reusable GT-context adapter
  rather than creating a second divergent loss definition.
- [ ] 3.4 Extend Stage-1 object grouping so the runtime can recover:
  - per-sample boundaries
  - object order
  - bbox quartets
  - object-local normalized desc for `same_desc` mode
  - packed-row boundary resets at successive CoordJSON containers
- [ ] 3.5 Fail fast when `same_desc` mode is enabled but object-local desc
  association is ambiguous.
- [ ] 3.6 Add Stage-1 tests covering:
  - config validation
  - unsupported nested-key rejection
  - GT-order adjacent pairing
  - adjacency never crossing sample boundaries
  - ambiguous desc grouping fail-fast behavior
  - partial-edge agreement does not over-trigger the box-copy margin

## 4. Stage-2 Rollout-Context Integration

- [ ] 4.1 Extend `coord_reg.config` with:
  - `adjacent_repulsion_weight`
  - `adjacent_repulsion_filter_mode`
  - `adjacent_repulsion_margin_ratio`
  - `adjacent_repulsion_copy_margin`
- [ ] 4.2 Keep the implementation as a coord-side `coord_reg` sub-term rather
  than a standalone bbox module.
- [ ] 4.3 Define and preserve a grouped Stage-2 carrier from `target_builder`
  through `bbox_geo` into `coord_reg`, including enough metadata for:
  - previous-object edge lookup
  - canonical clean-order adjacency
  - `same_desc | global` gating
- [ ] 4.4 Ensure rollout-context adjacency uses canonical edited clean target
  order rather than supervision append order.
- [ ] 4.5 Ensure the current object must already belong to a positive
  coord-supervised rollout subset before adjacent repulsion applies.
- [ ] 4.6 Add targeted rollout-context tests covering:
  - adjacent pairing in edited clean order
  - clean-order adjacency differing from append order
  - previous clean-order object retained as context but not bbox-supervised
  - `same_desc` vs `global`
  - partial-edge agreement remains below the copy margin
  - no deprecated `self_context` naming or assumptions

## 5. Registry, Config, And Metrics

- [ ] 5.1 Extend the shared loss-registry semantics with canonical adjacent
  repulsion naming for `gt` and `rollout` contexts.
- [ ] 5.2 Tighten config validation in:
  - tighten Stage-1 coord auxiliary parser validation
  - Stage-2 `coord_reg.config`
  - rollout-matching `coord_reg.config`
- [ ] 5.3 Add canonical metrics for:
  - Stage-1 adjacent-repulsion contribution
  - Stage-2 `loss/B_coord/adjacent_repulsion`
  - adjacent-pair count diagnostics
  - adjacent box-copy score diagnostics
- [ ] 5.4 Make metric ownership explicit:
  - `coord_reg` emits raw adjacent diagnostics
  - Stage-2 objective logging projects rollout provenance to `loss/B_coord/*`
- [ ] 5.5 Update docs after implementation:
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`

## 6. Validation

- [ ] 6.1 Re-run narrow config/spec coverage:
  - `conda run -n ms pytest tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms pytest tests/test_stage2_ab_config_contract.py`
- [ ] 6.2 Re-run Stage-1 focused coverage:
  - coord auxiliary and Stage-1 bbox/packing tests touched by the adapter path
  - `conda run -n ms pytest tests/test_stage1_metric_key_parity.py`
- [ ] 6.3 Re-run Stage-2 focused coverage:
  - `coord_reg` / objective projection tests
  - Stage-2 training tests touching rollout-context bbox groups and metrics
- [ ] 6.4 Run a focused smoke or synthetic acceptance check for:
  - partial-edge overlap staying below `adjacent_repulsion_copy_margin`
  - canonical clean-order adjacency behavior
- [ ] 6.5 Validate the OpenSpec change at the end:
  - `openspec validate add-adjacent-distributional-repulsion-loss --type change --strict --json --no-interactive`

## 7. Deferred Follow-Up

- [ ] 7.1 Evaluate whether a decoded-box CIoU-style adjacent companion should be
  added after the distributional v1 signal is validated.
- [ ] 7.2 Evaluate whether adjacency should later expand beyond the immediately
  previous object.
