# Conditional illegal-mass replacement (2026-09-26)

Status: production and matched val200 evaluation completed on 2026-09-27. Supersedes the expected-coordinate hinge for new training; historical axis001 checkpoints and evidence are unchanged.

The auxiliary is `conditional_order_gate`, weight 0.01. For a complete supervised xyxy object, x2 is conditioned on the same object's actual teacher-prefix x1 token, and y2 on y1. Each slot uses `logsumexp(coordinate logits) - logsumexp(strictly legal coordinate logits)`, equivalent to `-log(1 - illegal coordinate-family probability mass)`. Equality is illegal. There is no expected-coordinate calculation, future-GT restriction on x1/y1, or inference mask. This auxiliary does not choose among valid next objects or target a particular legal bin; CE still supervises the annotated continuation. Finite-weight training is not a rollout legality guarantee.

Reduction: mean of the two constrained slots, mean of complete boxes per supervised segment, then the existing global supervised-segment denominator and DDP compensation. Segments without complete boxes contribute connected zero. Incomplete boxes are counted and skipped; missing required predecessors, mismatched prefix tokens and inconsistent object metadata fail closed.

Implementation reuses the research-probes `conditional_order_gate` computation and adapts its per-segment result to the current loss runner. The old hinge module, binding, schema key and runnable axis YAMLs are removed; old schema keys fail validation instead of silently acquiring new semantics.

Production config: `configs/train/geo_sorted_xy/untied_illegal_mass.yaml`.
Eight-GPU qualification config: `configs/smoke/eight_gpu_geo_sorted_xy_untied_illegal_mass.yaml`.

This starts fresh from Qwen3-VL-2B-Instruct-coordexp-natural-adjacent, with untied selected-token deltas, four epochs, EBS24, packing12000, language LR2e-4 and token LR1e-4. Model, optimizer, training, packing, template, data, adapter, checkpoint, eval and resume settings were compared to the frozen September18 production YAML. Checkpoint input is null; optimizer state is not resumed. The numerical auxiliary weight remains0.01, but the new objective has a different scale from the old hinge.

Validation receipts: `/data/CoordExp/outputs/infra_base/illegal-mass-preparation-20260926/`. CPU tests cover strict boundary gradients, the legal-mean/illegal-argmax counterexample, legal-mass redistribution, prefix and object identity, packed segments, compact logits, BF16 and DDP gradients, configuration rejection and trainer integration. A mutation admitting equality is rejected. At this preparation checkpoint, no GPU training or full cache preparation had been launched; subsequent results are recorded below.

## Three-loss configuration update

User authorized enabling all three objectives. Current production and smoke YAMLs explicitly use CE1.0, token_type_gate0.1 (`mode: enabled`), and conditional_order_gate0.01. Type gate is therefore an additional change relative to the September18 zero-weight baseline. These are starting coefficients, not measured equal gradient contributions or an optimized ratio. The updated effective configuration and validation are `resolved-config-three-losses.json` and `three-losses-validation.json` in the preparation receipt directory; they supersede the previous effective configuration. All three runner bindings were verified active before launch.

## Production and val200 result (2026-09-27)

The eight-GPU production run completed 2444/2444 updates over four epochs with no skipped or nonfinite updates. The final checkpoint is `/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-illegal-mass001-ebs24-4epoch/checkpoints/step-2444`; a fresh native HF process reloaded both independent embedding deltas exactly and verified their separate forward effects. Launch evidence is under `/data/CoordExp/outputs/infra_base/illegal-mass-launch-20260927/`.

The fixed val200 protocol reused the same 200 rows, training prompt, HF BF16/FlashAttention2 backend, eight active ranks, batch size 4, greedy decoding, 3084 new-token cap, and repetition penalty 1.0 (no penalty). The normalized arm used `coordinate_output_norm: median`. Full inference/evaluation artifacts and the verified comparison are under `/data/CoordExp/outputs/infra_base/untied-illegal-mass-val200-20260927/`.

| Checkpoint | Coordinate norm | bbox mAP | FN50 | Geometry-invalid drops | Truncated rows |
|---|---|---:|---:|---:|---:|
| New illegal-mass | disabled | 0.448517 | 626 | 565 | 2 |
| New illegal-mass | median | 0.455651 | 614 | 220 | 2 |
| Historical axis001 | disabled | 0.463153 | 623 | 497 | 2 |
| Historical axis001 | median | 0.480328 | 608 | 89 | 2 |

Median normalization raised the new checkpoint's mAP by 0.007134 and reduced FN50 by 12. Against axis001 under the matching norm setting, the new checkpoint was lower by 0.014636 mAP without normalization and 0.024677 with it. Almost all geometry-invalid drops in the new run came from two length-limited images (`coco2017_val_000000000885` and `coco2017_val_000000005586`), so the count does not represent a broad per-image failure rate. The changed objective includes both the replacement geometry term and newly enabled token-type gate; this comparison does not isolate either term's causal effect. The val200 point estimates show no quality improvement over axis001.
