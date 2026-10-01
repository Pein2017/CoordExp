# Raw-Text Coordinate Mechanism Update (2026-04-21)

## New bundles

- Pre-burst margin, model-native: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-model-native-20260421`
- Pre-burst margin, pretty-inline: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-pretty-inline-20260421`
- Pre-burst surface comparison: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-surface-comparison-20260421.json`
- FN suppression, model-native: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421`
- FN review gallery: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/review_gallery/review.html`

## Pre-burst anchor-collapse findings

- Both surfaces completed with `selected_case_count = 16` and `num_case_variant_rows = 64`.
- The strongest causal intervention remains `source_x1y1_from_gt_next`.
- On `model_native`, base-only stays in a bad basin on average:
  - baseline `mean_margin_mean_logprob = -0.2443`
  - `source_x1y1_from_gt_next = -0.2874`
- On `model_native`, the raw-text adapter is much closer to the decision boundary and the x1/y1 intervention pushes it toward GT:
  - baseline `mean_margin_mean_logprob = -0.0061`
  - `source_x1y1_from_gt_next = 0.0719`
  - `mean_delta_from_baseline_mean_logprob = +0.0780`
  - `positive_margin_mean_rate = 0.625`
- The surface comparison is small for the adapter and modest for base-only. The adapter’s aggregate pre-burst result is effectively surface-stable between `model_native` and `pretty_inline`.
- The `source_bbox_from_gt_next` rows collapse to zero margin by construction, so they behave as the intended sanity check rather than an independent effect.

## FN suppression findings

- The FN suppression probe completed with `selected_case_count = 5` and `num_case_model_rows = 10`.
- In all selected cases, `continue_with_gt` loses to `eos_now` on joint sequence score for both models:
  - base-only `positive_continue_minus_eos_sum_rate = 0.0`
  - base+adapter `positive_continue_minus_eos_sum_rate = 0.0`
- But per-token evidence is often less anti-object than the joint score suggests:
  - base-only `positive_continue_minus_eos_mean_rate = 0.6` and `stop_pressure_rate = 0.6`
  - base+adapter `positive_continue_minus_eos_mean_rate = 0.4` and `stop_pressure_rate = 0.4`
- This is consistent with a real EOS-vs-continue suppression signature: on several recovered FN cases, continuation is not tokenwise implausible, but it still loses badly in joint probability because the stop action is dramatically shorter.

## FN review surface

- The selected 5 FN cases now have a dual-panel HTML review gallery:
  - baseline miss on the left
  - best recovered oracle hit on the right
- Gallery path:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/review_gallery/review.html`
- Each card includes:
  - GT category and image id
  - best oracle run label and IoU
  - base-only and base+adapter stop-pressure readouts
  - a heuristic badge: `both_models_stop_pressure`, `mixed_stop_pressure`, or `weak_stop_pressure_signal`
- This is now the intended user-facing path for deciding whether the recovered FN cases look like clean labeled-GT recoveries versus visually ambiguous examples.

## Code paths landed in the worktree

- `src/analysis/raw_text_coordinate_continuation_scoring.py`
- `src/analysis/raw_text_coordinate_preburst_probe.py`
- `src/analysis/raw_text_coordinate_fn_probe.py`
- `src/analysis/raw_text_coordinate_fn_review.py`
- `scripts/analysis/run_raw_text_coordinate_preburst_margin_probe.py`
- `scripts/analysis/run_raw_text_coordinate_fn_suppression_probe.py`
- `scripts/analysis/build_raw_text_coordinate_fn_review_gallery.py`
- `configs/analysis/raw_text_coordinate_mechanism/preburst_margin_probe.yaml`
- `configs/analysis/raw_text_coordinate_mechanism/preburst_margin_probe_pretty_inline.yaml`
- `configs/analysis/raw_text_coordinate_mechanism/fn_suppression_probe.yaml`
- `configs/analysis/raw_text_coordinate_mechanism/fn_review_gallery.yaml`

## Recommended next reads

1. `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-model-native-20260421/summary.json`
2. `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-pretty-inline-20260421/summary.json`
3. `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-surface-comparison-20260421.json`
4. `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/summary.json`
5. `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/review_gallery/review.html`
