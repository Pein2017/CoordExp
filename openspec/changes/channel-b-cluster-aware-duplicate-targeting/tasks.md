## 1. Spec And Contract Alignment

- [ ] 1.1 Update Channel-B rollout preparation to replace sequential duplicate bursts with deterministic cluster-aware duplicate-like grouping while preserving anchor-rooted clean-prefix editing.
- [ ] 1.2 Keep `loss_duplicate_burst_unlikelihood` as the canonical B-only module and update the runtime metadata builder so it emits cluster-aware first-divergence targets on the existing payload shape.

## 2. Runtime Implementation

- [ ] 2.1 Port the minimal duplicate-like relation from analysis code into a repo-owned Stage-2 helper that operates on parsed bbox rollout objects without post-hoc confidence inputs.
- [ ] 2.2 Add crowd-safety guards that distinguish tight duplicate collapse from spread-out or explorer-supported same-description crowded objects.
- [ ] 2.3 Wire the new duplicate-like grouping through [rollout_views.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py), [target_builder.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py), and the Stage-2 trainer logging path without introducing new CLI flags.

## 3. Diagnostics And Verification

- [ ] 3.1 Emit the cluster-aware duplicate diagnostics required by the spec, including `dup/duplicate_like_max_cluster_size`, `dup/desc_entropy`, and `stage2_ab/channel_b/dup/N_duplicate_like_clusters`.
- [ ] 3.2 Add deterministic unit coverage for tight duplicate clusters, spread-out crowded same-description objects, and explorer-supported same-description objects.
- [ ] 3.3 Run a Stage-2 smoke under `conda run -n ms python -m src.sft --config <stage2 smoke config>` and verify duplicate-target counts, duplicate-collapse metrics, and output artifacts remain stable across repeated runs.

## 4. End-To-End Validation

- [ ] 4.1 Produce a short Stage-2 comparison run with identical config path, run name convention, and seed except for the new duplicate-targeting behavior.
- [ ] 4.2 Re-run inference/eval on the resulting checkpoint and compare duplicate-collapse scenes plus `coco_real` metrics against the pre-change baseline.
- [ ] 4.3 Confirm the run directory still contains the expected reproducibility artifacts, including `resolved_config.json`, `effective_runtime.json`, `pipeline_manifest.json`, and the standard trainer metrics outputs.
