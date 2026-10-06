# Retained audit evidence

These original audits are historical technical evidence, not current gates or
run instructions. Read current [documentation](../../docs/README.md) and
[research questions](/data/CoordExp/.worktrees/research-probes/research/index.md)
first. Retention is limited to the independent uses below.

| Source | Independent evidence use |
|---|---|
| [Stage-1 SoftCE accounting](2026-01-22_stage1_softce_logging.md) | Accumulation-dependent train/eval logging counterexample; loss-only A100 timings distinguish full-vocabulary gate cost from SoftCE cost. Proposed W1 removal was not measured. |
| [Channel-A coordinate loss](2026-02-25_stage2_channel_a_coord_loss.md) | The observed run contradicts the proposed F1-down/AP-up diagnosis: F1 improved while AP stayed roughly flat; A1 coordinate diagnostics were absent. Keeps the exact run, step table and final-softctx versus A1 loss distinction. |
| [Type/schema audit](2026-05-03_type_schema_architecture_audit.md) | Bounded encoded-cache acceptance and unresolved static-packing/prediction/state follow-ups. Typed internal records preserved serialized keys; the broad audit was not exhaustive implementation closure. |
| [Instance-Gaussian implementation audit](2026-05-14-instance-trie-gaussian-post-implementation-audit.md) | Candidate sidecar causality, teacher-candidate consistency and ignored-tau rejection before model smoke; distinguishes this successor from IoU/CIoU-Gibbs. |
| [Instance-Gaussian smoke](2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md) | Target-shape, tiny and DDP8 measurements, including candidate sharpening and resource cost; four-step readiness does not establish free-rollout quality. |
| [DetectionScene source checkpoint](2026-05-31_detection_scene_clean_break_archive_checkpoint.md) | Exact pre-replacement Git identity used by the following classification and validation records. |
| [DetectionScene classification](2026-05-31_detection_scene_phase0_surface_classification.md) | Original replacement-before-deletion contract, including private bridges, strict versus salvage outputs and unchanged artifact names; interprets the validation's allowed leftovers. |
| [DetectionScene validation](2026-06-01_detection_scene_cleanup_validation.md) | Actual narrow caller/config checks, skips and classified leftovers after the clean break; preserves the boundary of what passed. |
| [Prefix-denoising plan audit](2026-06-14_prefix_denoising_sft_v1_audit.md) | Concrete sidecar and packed-isolation counterexamples, quantization-induced small-box exclusion, reversed skip-rate fallback, and unequal-denominator risk. A coherent objective did not qualify its implementation. |

Current ownership and governance are maintained in
[documentation](../../docs/documentation.md), [storage](../../docs/storage-ownership.md)
and [engineering principles](../../docs/engineering-principles.md).
