# Coord Gaussian RPS Stage-1 Auxiliary Plan

## Goal

Add a main-based ordinary Stage-1 coordinate smoothing auxiliary for full-wrapper
detection SFT:

- single standard forward only: base CE plus coordinate Gaussian and RPS/CPRS
- GT-centered coordinate mean
- object bbox shape-aware variance recovered from adjacent compact-full `xyxy`
  coordinate quads
- compact four-way type gate enabled in the new launch profile: `coord`, `desc`,
  wrapper/schema, and `eos`
- no W1/`coord_soft_ce_w1` naming or dependency for the new mechanism

Legacy `custom.coord_soft_ce_w1` stays available for old configs unless a config
explicitly opts into `coord_gaussian_rps`.

## Source Handles

- Worktree: `/data/CoordExp/.worktrees/coord-shape-gaussian-rps`
- Current compact type groups: `src/detection/token_types.py`
- Recursive type-gate paradigm:
  `src/detection/loss.py::_compute_type_gate_loss` and
  `src/detection/objective.py::_apply_compact_type_gate`
- Ordinary Stage-1 hook to replace for this branch:
  `src/trainers/metrics/coord_losses.py`, `src/bootstrap/trainer_setup.py`,
  and `src/sft.py`
- Old donor idea only:
  `/data/CoordExp/.worktrees/fully-compact-2x2-ablation/src/trainers/losses/sft_gaussian_coord_soft_ce.py`
- Target launch profile:
  `configs/stage1/profiles/2b/pure_ce_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml`

## Tasks

1. Add failing tests for Gaussian RPS math, shape-aware R95 inference, ordinary
   loss behavior, config schema, trainer wiring, and the desc-first full-wrapper
   launch config.
2. Add `src/coord_tokens/gaussian_rps.py` with normalized Gaussian targets from
   R95 radii and discrete RPS/CPRS over coordinate-bin CDFs.
3. Add `src/trainers/losses/coord_gaussian_rps.py` to compute coord-only CE,
   Gaussian soft CE, RPS/CPRS, and compact type-gate allowed-mass loss from the
   same ordinary forward logits.
4. Add `src/trainers/metrics/coord_gaussian_rps.py` and wire it through
   `src/trainers/metrics/mixins.py`, `src/bootstrap/trainer_setup.py`,
   `src/sft.py`, and `src/detection/runtime.py`.
5. Add `objective.auxiliaries.coord_gaussian_rps` schema with nested
   `type_gate.enabled`, `type_gate.mode: allowed_type_mass`, and weights
   `struct`, `coord`, `desc`, `eos`.
6. Add prod and tiny smoke configs for the desc-first sorted full-wrapper profile
   using `compact_object_box_closed`, `bbox_format: xyxy`, and default type gate
   weights `1.0, 1.0, 1.0, 0.5`.
7. Update current docs/research notes to state that CPRS is implemented as
   discrete RPS and that this is not the old W1 surface.
8. Verify with the new narrow tests, relevant config/runtime contract slices,
   legacy W1 tests, and `git diff --check`.
