# Runtime Footguns

Role: quick recall for durable config/runtime traps and compatibility boundaries.

Canonical pointers:
- `docs/SYSTEM_OVERVIEW.md`
- `docs/ARTIFACTS.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/training/METRICS.md`
- `docs/eval/WORKFLOW.md`
- `openspec/specs/runtime-architecture-refactor-program/spec.md`
- `openspec/specs/stage2-ab-training/spec.md`
- `openspec/specs/rollout-matching-sft/spec.md`

High-signal traps:
- YAML-first config validation is strict; unknown keys fail fast. Route config debugging through `src/config/loader.py` first, then `src/config/schema.py` and rollout-specific schema helpers.
- Single-dataset training is the default posture; fusion-config training is legacy or experimental.
- Offline-prepared JSONL is the contract: preserve geometry alignment, keep runtime resize disabled, and use `src/datasets/geometry.py` for coordinate semantics.
- Treat Stage-2 surfaces separately: `stage2_two_channel` is the current operator-facing path, while `stage2_rollout_aligned` is the supported compatibility variant with its own config family and specs.
- Do not cross-wire Stage-2 config families: author `stage2_ab.pipeline.*` for `stage2_two_channel` and `rollout_matching.pipeline.*` for `stage2_rollout_aligned`.
- Route training entry, provenance, and trainer assembly through `src/sft.py` and `src/bootstrap/{pipeline_manifest,trainer_setup,run_metadata}.py` before diving into trainer internals.
- Inference and evaluation are separate seams: `src/infer/{pipeline,engine,backends,artifacts}.py` vs `src/eval/{detection,orchestration,artifacts}.py`.
- Offline evaluator metrics and trainer-native Stage-2 rollout metrics use different key families; do not compare them as if they were the same surface.
- Artifact/provenance behavior is centralized in `docs/ARTIFACTS.md`; route there instead of relying on memory for file-level artifact details.

Routing hints:
- config contract changes -> `src/config/loader.py`, `src/config/schema.py`
- data contract/rendering -> `src/datasets/geometry.py`, `src/datasets/builders/jsonlines.py`, `src/datasets/dense_caption.py`
- bootstrap/provenance -> `src/bootstrap/`
- Stage-2 two-channel behavior -> `src/trainers/stage2_two_channel.py`, `src/trainers/stage2_two_channel/`, `src/trainers/stage2_coordination.py`
- rollout-aligned behavior -> `src/trainers/stage2_rollout_aligned.py`, `src/trainers/rollout_aligned_targets.py`, `src/trainers/rollout_aligned_evaluator.py`, `src/trainers/rollout_runtime/`
- inference/eval runtime seams -> `src/infer/`, `src/eval/`