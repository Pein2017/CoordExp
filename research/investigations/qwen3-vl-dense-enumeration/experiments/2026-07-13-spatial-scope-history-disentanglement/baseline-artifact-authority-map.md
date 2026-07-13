# Baseline Artifact Authority Map

Evidence date: 2026-07-13.

This note records provenance only. It does not promote a historical result into
the current inference or evaluation contract. The bounded evidence set is the
two step-4887 validation-200 artifact roots and their launch logs, one current
CoordExp-Swift contract smoke artifact, the two exact checkpoint manifests,
`docs/eval/WORKFLOW.md`, and the stable inference-scoring and detection-evaluator
specifications.

## Authority verdict

- The two legacy validation-200 runs are historical motivation and paired
  diagnostics only. They used deterministic temperature `0.0`, repetition
  penalty `1.10`, and `max_new_tokens=3084`.
- Their `confidence_postop:v2` scores do not satisfy the current
  `compact-object-selected-token-score-v1` contract. In particular, legacy
  score provenance is row-level and each prediction lacks the current
  structured, replayable `pred_score_source`. Their legacy metric files are
  therefore not eligible for current benchmark claims.
- The current `max_new_tokens=512` CoordExp-Swift smoke artifact is a separate,
  explicitly benchmark-ineligible reference. It must not be mixed into the
  paired legacy comparison: it uses a different inference implementation,
  artifact schema, score policy, decode horizon, and prompt-policy identity.
- The spatial-scope/history-disentanglement experiment will establish its own
  sampled baseline with repetition penalty `1.0` and its own run receipts. No
  metric or score from the artifacts below substitutes for that baseline.

## Checkpoint identities

Both checkpoints compose the same Qwen3-VL-2B base model with
Weight-Decomposed Low-Rank Adaptation (DoRA) and a special-token embedding
delta. The manifest fingerprint, rather than the directory name, is the
checkpoint provenance handle.

| Condition | Exact checkpoint manifest | Secure Hash Algorithm 256-bit (SHA-256) | Adapter fingerprint | Special-token embedding fingerprint |
|---|---|---|---|---|
| Gaussian coordinate objective | `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json` | `613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536` | `ba25619faf51ef7d1eb732e514f250efd37bce3d7f54d1db9eb5e83d8cc2a6c6` | `c8df057b9e3be1479c24e2272338ba627c19a663d19333a024e558488e64ebeb` |
| Pure cross-entropy objective with type gate | `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json` | `c8ad1ab01550fc640c67457fec9ad1f8b3bd1b8cef351cb90d41666233b80da1` | `35fd88b1586b946943b936f24531ee95d4a4c57c8efc358bc81a06af9ecbb6c3` | `b5bef0f097c4ddf8f8879e71183795ab877f55c5702c2fb48979d9c555f2c9c9` |

## Paired legacy validation-200 evidence

The exact roots are:

1. `/data/CoordExp/outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200_bsz4_temp0_rp1p10_max3084_8gpu`
2. `/data/CoordExp/outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_step4887_val200_bsz4_temp0_rp1p10_max3084_8gpu`

All paths in the next table are relative to the corresponding exact root.
JavaScript Object Notation Lines (`JSONL`) row counts describe the physical line shape of each
artifact; the legacy token trace stores one row per input rather than the
current one-row-per-generated-token trace shape.

| Artifact | Gaussian SHA-256 / rows | Pure cross-entropy SHA-256 / rows |
|---|---|---|
| `gt_vs_pred.jsonl` | `b98a37ceefbc6d63893ab216540065ef2639f599c7a4cf741d06b6651be0d78b` / 200 | `341974d7292ee9b15abb8a808d4bd714a816210ec1295b334cbda90e9dc15ccb` / 200 |
| `gt_vs_pred_scored.jsonl` | `d41c9b101461612e1c70a410f9bd818a1ba9a5bfd3927dbda0c1e12fef570e7a` / 200 | `3c09e163239861c000966042825ff756f3e9980125b1e629c6233a9ebdafba45` / 200 |
| `pred_token_trace.jsonl` | `bac5b3134f02e5dca2466b7c4ba5dbbe9fef7260b0fc6ee58f2331992213718f` / 200 | `6e99399c4c35881a486df5dc5e540fffa3de6f644e27bfa1e0cc4fb9f79a7903` / 200 |
| `pred_confidence.jsonl` | `ab58e8a83c466160377b9656dfb59fe052ed52d35b63a24387c9d6cb4081f3a3` / 200 | `9bcabe5a2f27274026cd742ffc1ce8997c739e2c134952a84d6a49510a6aa15e` / 200 |
| `confidence_postop_summary.json` | `ceaaf4d76f6be2921b812c3a9dbd127187dd9d919a039a4489768e704e4a9135` | `6f08666ed0c2dec3a1e61ec57587bf11746184f50567cf450be147a586a5ce4d` |
| `summary.json` | `3d034f07ba0ff6ecc3d2624ec15a7b17fe3073889e241ed01d384a4ef5a03f1a` | `a0dd7eab9a58b983177b6faab2370a7bac8670fbb9b1fd9ed04181f7e269226c` |
| `gt_vs_pred_scored.jsonl.provenance.json` | `0f9c46b07f3f1dcce61c7d0458c5e73dfbb9242b4c1741b63f9ac91cf69f2fdb` | `bb09c58df27a15e82d66de2b2dbed9f999dcf8b30ea63a0ff2097c5472a7d793` |
| `eval/metrics.json` | `49457c7fe75ef5233f73bb6588e88af0690c40ea8b1cdfc0b09e8daadff8c563` | `02e390b511f8665f071b38f5cfcf29136ec16530be6ea654b1465be279d5b06b` |

Both runs read and emitted 200 rows, used batch size `4`, seed `42`, no image
resize, Common Objects in Context 80-class (`COCO-80`) prompting,
description-first fields, geometry-sorted object order, and the compact closed
object-box template. Their generation settings were temperature `0.0`, top-p
`0.9`, repetition penalty `1.10`, and `max_new_tokens=3084`. The launch logs
warn that sampling-only flags such as temperature and top-p may be ignored in
deterministic generation.

The Gaussian run retained 1,336 of 1,336 confidence-scored predictions and
recorded one dropped invalid object. The pure cross-entropy run retained 1,294
of 1,294 confidence-scored predictions and recorded no inference error. Those
counts are diagnostic properties of the historical scorer, not current score
eligibility.

Execution receipts:

- Gaussian log:
  `/data/CoordExp/outputs/infer/coordexp_swift/launch_logs/gaussian_rps_step4887_val200_main_8gpu_20260710T070749Z.log`,
  SHA-256 `7b0871905fd869a6e0817263987b55a8f1eaaf9bba8275174aee86d73ef74db7`.
- Gaussian status:
  `/data/CoordExp/outputs/infer/coordexp_swift/launch_logs/gaussian_rps_step4887_val200_main_8gpu_20260710T070749Z.status`,
  content `0`, SHA-256
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
- Pure cross-entropy log:
  `/data/CoordExp/outputs/infer/coordexp_swift/launch_logs/pure_ce_typegate_step4887_val200_main_8gpu_20260710T065353Z.log`,
  SHA-256 `b2cbaa3657850bb5d64aef17856b3bd182fb3d6774ca208c09139a7bbe774eb2`.
- Pure cross-entropy status:
  `/data/CoordExp/outputs/infer/coordexp_swift/launch_logs/pure_ce_typegate_step4887_val200_main_8gpu_20260710T065353Z.status`,
  content `0`, SHA-256
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The launch logs name legacy configuration paths that no longer exist in the
current checkout. The immutable `resolved_config.json`, summaries, logs, and
artifact hashes above are the reproduction evidence; the missing source
configuration files must not be presented as current configuration authority.

## Separate current-contract smoke reference

Exact root:
`/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-gaussian-rps-dora-r16a32-step4887-val200`.

| Artifact | SHA-256 | Rows |
|---|---|---:|
| `gt_vs_pred.jsonl` | `10f64e506cbfb45d014fc8bd6f5b1e05419d6813211157ac03e3e738857d725a` | 200 |
| `gt_vs_pred_scored.jsonl` | `fab7dbac5bd3ff13b46d9cf132b7bf8e095a1bf1ac2ab301aa20b7d9de295011` | 200 |
| `pred_token_trace.jsonl` | `9c6e74e55cec4b91edde45003db65e85bc9f760c980c74f6674a9d913c4cc12a` | 26,669 |
| `summary.json` | `901f26e837b5b063e5082cbb99cd9eb776a4a94ccf71cf007bc8bc5e28ed6e29` | not applicable |
| `gt_vs_pred_scored.jsonl.provenance.json` | `23559439299d096d00302597552b24804063f0ddad8b9f178a3889d8a0ca3077` | not applicable |
| `run_manifest.json` | `89ddb9c744ecb701e0e5f0022316f9325f742926976b626d7248af3d7f17abf5` | not applicable |
| `eval/metrics.json` | `5af3fc47aea2e154d3bd0dac45c3d5bc57a71a2506410bb71fbce6ad056619ee` | not applicable |
| `eval/evaluation_receipt.json` | `231f06a948174f94b9b11658b6a77d321c41c7483c8721a65474f083947c2f4e` | not applicable |

This run used deterministic generation with temperature `0.0`, top-p `1.0`,
repetition penalty `1.10`, `max_new_tokens=512`, batch size `4`, no image
resize, and the current compact parser/scorer. It materialized 1,281 scoreable
predictions. Its manifest and evaluator receipt both explicitly state
`benchmark_eligible=false`; its metric file is a smoke result, not a baseline
for the fresh experiment.

Unlike the legacy artifacts, its Version 1 score is
`exp(sum(selected_token_logprobs) / n_selected)` over exactly eight tokens:
the four object/box wrapper tokens and four coordinate tokens. Description and
category text are excluded. Each prediction carries structured row, object
span, token-index, token-identifier, token-text, log-probability, selected-count,
and score-policy evidence, with trace replay and raw/scored row binding.

## Current reusable entry points

Current behavior is owned by `docs/eval/WORKFLOW.md` and the stable
`openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md` and
`openspec/specs/coordexp-swift-detection-evaluator/spec.md` contracts.

- Inference command: `conda run -n ms python -m src.infer --config <config>`.
- Compact parser: `src.inference.parsing.parse_compact_object_box_closed`.
- Selected-token scorer: `src.inference.scoring.score_prediction` with policy
  identifier `compact-object-selected-token-score-v1`.
- Canonical artifact writer and validator:
  `src.inference.artifacts.write_inference_artifacts` and
  `src.inference.artifacts.validate_scored_artifact_set`.
- Direct evaluator function:
  `src.eval.detection_consumer.evaluate_scored_detection_artifacts`.
- Direct evaluator command:
  `conda run -n ms python scripts/evaluate_detection.py --artifact-dir <run-dir> --out-dir <run-dir>/eval`.

The current evaluator accepts only a contract-valid raw/scored/provenance/trace
family. It validates raw-to-scored identity, row parity, model, processor,
prompt, generation, parser, and score-policy bindings before metric reduction.
Legacy confidence post-processing, duplicate-guard, per-image, per-class, and
semantic diagnostic families remain historical unless a current contract
explicitly rebuilds them. A path or a legacy `metric_bearing=true` field is a
provenance handle, not proof of current benchmark eligibility.
