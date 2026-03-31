## Bootstrap Stage. Hard-Case Selector

- [x] B.1 Add a bootstrap selector manifest/config that defines the frozen candidate pool, dataset split, checkpoint pair, evaluator settings, and deterministic ranking tuple.
- [x] B.2 Implement deterministic bootstrap image ranking using `image_only`, greedy, default-length baseline cells for `original` and `a_only`, separately for `train` and `val`.
- [x] B.3 Exclude rollout-health-invalid images from the final bootstrap ranking and record the reason per excluded image.
- [x] B.4 Emit a ranked bootstrap table plus frozen split-specific `Hard-32` and `Hard-16` manifests, with `Hard-16` as the top-16 prefix subset of `Hard-32` within the same split.
- [x] B.5 Verify that rerunning the bootstrap selector from the same inputs produces byte-identical split-specific `Hard-32` and `Hard-16` manifests.

## Stage 0. Study Manifest, Subset Freeze, And Config Surfaces

- [x] 0.1 Add a manifest/config schema for fixed-checkpoint rollout-factor studies under `configs/analysis/` that resolves checkpoint aliases, checkpoint artifact kinds, checkpoint fingerprints, `dataset_split`, split-specific JSONL, explicit image ids and image order for the fixed study subset, subset name, subset-selection provenance, bootstrap selector provenance, image root, prompt variant, prompt hash, object field order, evaluator settings, backend, seed schedule, output directory, and optional checkpoint provenance sidecars.
- [x] 0.2 Add initial study manifests for the canonical split-specific `Hard-16` and `Hard-32` `original` vs `a_only` comparisons, freezing:
  - `original = output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
  - `a_only = output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`
  - `a_only_config_source = output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml`
  - `train = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - `val = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- [x] 0.3 Add resolved-manifest artifact writing so every study run persists checkpoint aliases, paths, dataset split, split-specific JSONL, fixed image subset, prompt contract, preprocessing invariants, evaluator settings, backend, seeds, and output locations before rollout cells start.
- [x] 0.4 Add stage-specific manifests or equivalent provenance artifacts for bootstrap, subset freeze, baseline, sampling, prefix, length, and reporting so later stages can resume from frozen earlier outputs.
- [x] 0.5 Add stage identifiers and logical-cell / execution-shard ids to the study plan so every later artifact can be placed in the staged experiment program.
- [x] 0.6 Ensure Stage 0 only consumes frozen bootstrap outputs and cannot mutate the already-selected split-specific `Hard-16` / `Hard-32` manifests.

## Stage 1. Deterministic Baseline

- [x] 1.1 Implement baseline cell execution for fixed checkpoints using the authoritative HF backend with parity-safe prompt construction and `do_resize=false`.
- [x] 1.2 Emit per-cell artifacts for the deterministic baseline, including `gt_vs_pred` JSONL, run summary, resolved generation settings, and stable cell identifiers.
- [x] 1.3 Add baseline aggregation that compares `original` vs `a_only` on `train / Hard-16`, `train / Hard-32`, `val / Hard-16`, and `val / Hard-32`, and writes checkpoint-level summary tables plus joinable qualitative outputs.
- [x] 1.4 Add rollout-health outputs for baseline cells, including non-empty prediction rate, parse-valid rate, invalid-rollout count, duplicate-like rate, prediction count, and truncation/stop anomalies when available.
- [x] 1.5 Validate the baseline layer on a small monitored subset and record a reproducibility command using one GPU, fixed seed, and explicit output dir.

## Stage 2. Image-Only Union-of-K Sampling Coverage

- [x] 2.1 Implement sampled cell execution with configurable `K`, temperature, top-p, repetition penalty, and seed schedule while preserving `logical_cell_id` plus `execution_shard_id`.
- [x] 2.2 Implement object-level union-of-`K` coverage analysis that computes per-GT hit frequency, supporting `never_hit` / `rare_hit` / `often_hit` / `always_hit` buckets, and union recall per image and checkpoint.
- [x] 2.3 Add sampled-layer rollout-health gating so unhealthy cells remain visible but do not enter the main sampling-recovery tables.
- [x] 2.4 Add sampled-layer aggregation outputs that separate deterministic recall from sampling-recoverable recall and preserve checkpoint × factor provenance.
- [x] 2.5 Validate sampled coverage on a small subset with at least one `K>1` cell and confirm the report distinguishes never-hit vs sometimes-hit GT objects.

## Stage 3. Prefix-Order Matrix

- [x] 3.1 Implement prefix-mode support for `image_only`, `oracle_gt_prefix_train_order`, `oracle_gt_prefix_random_order`, and `self_prefix`, with explicit prefix provenance, prefix ordering rule, prefix content hash, and prefix length metadata.
- [x] 3.2 Implement fixed-seed random-order prefix construction so the same GT object set can be serialized in training order or random order for the same image.
- [x] 3.3 Implement the normative continuation partition contract by serializing prefix-injected predictions first, recording `prefix_pred_count` plus `continuation_pred_start_index`, and computing continuation-only recovery from only the continuation tail.
- [x] 3.4 Ensure prefix-injected objects cannot satisfy continuation-only recovery by construction, even when whole-scene review artifacts include them.
- [x] 3.5 Add rollout-health gating for prefix-order cells so parser-collapse or degenerate continuation cells do not enter the main prefix-attribution tables.
- [x] 3.6 Validate the full prefix-order matrix on `train / Hard-16` and `val / Hard-16`, and confirm that train-order, random-order, and self-prefix cells are separately labeled in artifacts before scaling the same stage to the corresponding `Hard-32` cohorts.

## Stage 4. Switched And Broken Prefix Stress Tests

- [x] 4.1 Extend prefix-mode support to `switched_prefix` and `broken_prefix`.
- [x] 4.2 Implement prefix mutation helpers for at least deletion, adjacent swap, and single-object insertion while preserving valid CoordJSON structure.
- [x] 4.3 Add rollout-health gating for switched/broken cells so invalid stress-test outputs are visible but do not pollute causal attribution.
- [x] 4.4 Validate switched-prefix and broken-prefix behavior on the fixed subset, prioritizing images or GT objects still unresolved after Stages 1-3.

## Stage 5. Sequence-Length Cross-Checks

- [x] 5.1 Add default-length vs extended-length study cells that preserve `max_new_tokens`, generated token count, emitted object count, stop reason, and EOS metadata per rollout.
- [x] 5.2 Implement per-image and per-checkpoint summaries that compare object-count and token-count behavior between default-length and extended-length cells.
- [x] 5.3 Add rollout-health gating for length cells so unhealthy comparisons are excluded from the main length-bias conclusions.
- [x] 5.4 Add a report section that flags cases where extended-length cells recover additional GT objects versus cases where they only add unmatched predictions.
- [x] 5.5 Validate the length-bias layer on the fixed subset and confirm that the report can join recovered objects back to the corresponding cell and image.

## Stage 6. Recovery Attribution And Review Queue

- [x] 6.1 Add an object-level recovery table that assigns `deterministic_hit`, `decode_selection_miss`, `prefix_sensitive_miss`, `length_bias_miss`, or `persistent_unrecovered` using minimal-intervention precedence.
- [x] 6.2 Make `(record_idx, gt_idx)` the normative GT-object key for per-object analysis artifacts and preserve `image_id` plus `file_name` when available.
- [x] 6.3 Add derived `canonical_gt_index` only when review or overlay artifacts are materialized.
- [x] 6.4 Add supporting quality flags such as `annotation_mismatch_candidate` and `semantic_confusion_candidate`.
- [x] 6.5 Add joinable qualitative artifact export and a review queue for the most informative cells, including stable links from overlays/panels back to logical cell ids, execution shard ids, `record_idx`, `gt_idx`, and `canonical_gt_index` where applicable.
- [x] 6.6 Ensure the review queue preserves separate audit slices for `train` vs `val` and for `Hard-16` vs `Hard-32`.

## Stage 7. Multi-GPU Orchestration And Final Reporting

- [x] 7.1 Implement logical-cell / execution-shard sharding over four local GPUs, with one process per GPU and per-shard recording of GPU id, checkpoint alias, backend, seed, and factor settings.
- [x] 7.2 Merge per-GT outputs in deterministic sort order by `logical_cell_id`, `execution_shard_id`, `record_idx`, `gt_idx`.
- [x] 7.3 Add final study reporting that separates deterministic baseline, sampled union-of-`K`, prefix sensitivity, and sequence-length evidence instead of collapsing them into one score.
- [x] 7.4 Add a stage-wise recovery waterfall so later recoveries are interpreted relative to earlier evidence.
- [x] 7.5 Report `train` and `val` separately, report `Hard-16` and `Hard-32` separately within each split, and treat split-matched `Hard-32` as the extension check for conclusions first seen on split-matched `Hard-16`.
- [x] 7.6 Run an end-to-end smoke study on a small fixed subset using the resolved manifest, a fixed run name, fixed seeds, and four-GPU sharding, then verify that all expected artifacts exist and are mutually joinable.
