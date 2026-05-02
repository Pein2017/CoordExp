# Compact Detection Sequence Phase 1 Training-Infra Plan

> **Management boundary:** Linear owns the overall research lifecycle and
> cross-phase progress. This super-power plan owns only the code implementation,
> tests, and smoke verification for the current branch.

## Goal

Merge an off-by-default Stage-1 training-infrastructure slice for compact
Pixel2Seq-style detection sequences on Qwen3-VL 2B.

This branch does **not** need to finish inference, `val200`, or research
interpretation. Those start after production training produces checkpoints.

## Phase 1 Scope

Implement and verify:

- compact assistant-target rendering for Stage-1 data,
- `custom.detection_sequence_format`,
- compact prompt/system-prompt wiring,
- native Qwen marker reuse:
  - `<|object_ref_start|>` as `structural_ce_only`,
  - `<|box_start|>` as `structural_ce_only`,
  - `<|coord_0|>`..`<|coord_999|>` as `coord_geometry`,
- role-separated trainable token-row offsets through
  `custom.trainable_token_rows`,
- coordinate-only loss/mask semantics for coord softCE/W1/regression-family
  behavior,
- static-packing and encoded-cache fingerprint separation by detection format
  and sample limits,
- a two-GPU Stage-1 smoke for `compact_full`,
- tests and docs needed to merge this training infrastructure to `main`.

## Explicitly Deferred To Phase 2

Linear should track these as later gates after production checkpoints exist:

- production training completion and checkpoint selection,
- inference config matrix,
- compact inference parse diagnostics,
- confidence post-op scoring policy,
- `val200` metrics,
- final benchmark note under `progress/benchmarks/`,
- Notion research interpretation and conclusion.

Phase 1 may contain parser helpers if they are low-risk and off by default, but
Phase 1 does not claim inference/eval readiness or measured AP behavior.

## Implemented Files

Training infrastructure:

- `src/common/detection_sequence.py`
- `src/config/schema.py`
- `src/config/prompts.py`
- `src/config/loader.py`
- `src/datasets/builders/jsonlines.py`
- `src/datasets/dense_caption.py`
- `src/sft.py`
- `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`

Token-role refactor:

- `src/tokens/__init__.py`
- `src/tokens/qwen_native.py`
- `src/tokens/roles.py`
- `src/tokens/row_offsets.py`
- `src/tokens/coord/`
- `src/tokens/structural/compact_markers.py`
- `src/coord_tokens/*.py` compatibility wrappers

Tests:

- `tests/test_detection_sequence_format.py`
- `tests/test_prompt_variants.py`
- `tests/test_coord_utils.py`
- `tests/test_encoded_sample_cache.py`
- `tests/test_encoded_sample_cache_runtime_config.py`
- `tests/test_stage1_set_continuation_config.py`
- `tests/test_stage1_static_packing_runtime_config.py`
- `tests/tokens/test_token_roles.py`
- `tests/tokens/test_row_offsets.py`
- `tests/coord_tokens/test_offset_adapter.py`

Docs:

- `docs/superpowers/specs/2026-05-01-compact-detection-sequence-ablation-design.md`
- `docs/superpowers/plans/2026-05-01-compact-detection-sequence-ablation.md`
- `docs/superpowers/research-management-pilot.md`

## Completed Phase 1 Tasks

- [x] Add compact detection-sequence renderer/parser unit surface.
- [x] Add `custom.detection_sequence_format` config support.
- [x] Add compact Stage-1 prompt/system-prompt wiring.
- [x] Integrate compact assistant-target rendering in JSONL/dense-caption
      training data paths.
- [x] Refactor token handling toward `src/tokens/` with
      `src/coord_tokens/` compatibility wrappers.
- [x] Add token role sets:
      `coord_geometry` and `structural_ce_only`.
- [x] Reuse native Qwen markers without tokenizer/model expansion.
- [x] Make `<|object_ref_start|>` and `<|box_start|>` trainable row IDs while
      keeping them outside coordinate loss IDs.
- [x] Keep coordinate loss IDs exactly `<|coord_0|>`..`<|coord_999|>`.
- [x] Include detection-sequence format in static-packing and encoded-cache
      fingerprints.
- [x] Include `train_sample_limit` in static-packing fingerprint to prevent
      stale smoke plan reuse.
- [x] Add compact-full Stage-1 smoke config.
- [x] Run two-GPU Stage-1 smoke with raw COCO JSONLs and static packing.

## Verified Evidence

Targeted tests:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_sequence_format.py \
  tests/test_prompt_variants.py \
  tests/test_coord_utils.py \
  tests/test_stage1_set_continuation_config.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_encoded_sample_cache.py \
  tests/tokens/test_token_roles.py \
  tests/tokens/test_row_offsets.py \
  tests/coord_tokens/test_offset_adapter.py -q
```

Observed:

```text
144 passed, 4 warnings
```

Lint:

```bash
conda run -n ms python -m ruff check \
  src/tokens src/coord_tokens src/config/schema.py src/config/__init__.py \
  src/sft.py src/common/detection_sequence.py src/common/prediction_parsing.py \
  src/datasets/builders/jsonlines.py src/datasets/dense_caption.py \
  src/infer/engine.py tests/test_detection_sequence_format.py tests/tokens \
  tests/coord_tokens/test_offset_adapter.py \
  tests/test_stage1_static_packing_runtime_config.py
```

Observed:

```text
All checks passed!
```

Diff hygiene:

```bash
git diff --check
```

Observed: exit code 0.

Two-GPU smoke:

```bash
config=configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml \
gpus=0,1 \
conda run -n ms bash scripts/train.sh
```

Observed final artifact:

```text
temp/compact_detection_sequence/output/stage1/smoke/compact_full_tiny/
  smoke_2steps-stage1-2b-compact_full-native_qwen_markers/
  v1-20260501-164229
```

Observed run facts:

```text
detection_sequence_format: compact_full
train_sample_limit: 384
val_sample_limit: 8
packing: true / static
packing_length: 12000
train static packs: 47 raw, 48 aligned
world_size: 2
accum/current_grad_steps: 12.0 on both logged steps
final global_step/max_steps: 2/2
train_loss: 56.97003746
```

Artifact files present:

```text
config_source.yaml
resolved_config.json
effective_runtime.json
eval_data_provenance.json
experiment_manifest.json
logging.jsonl
run_metadata.json
runtime_env.json
train_data_provenance.json
```

## Remaining Before Merge

- [x] Update this plan/spec after Phase 1 scope refinement.
- [x] Update `AGENTS.md` so the repo-level work routine says:
      Linear owns overall research/process state; super-power owns
      branch-specific implementation and smoke verification.
- [ ] Reconcile this branch with current `main`.
- [ ] Re-run targeted tests after reconciliation.
- [ ] Re-run `git diff --check`.
- [ ] Review diff scope for accidental inference/eval promises.
- [ ] Commit the Phase 1 implementation in logical commits.
- [ ] Merge to `main` or open a PR.
- [ ] Only clean/free the worktree after the branch is safely merged or pushed.

## Linear-Owned Follow-Up

After merge, Linear should track:

- launch compact-full production training from `main`,
- decide whether to train baseline E and compact A/B/C/D variants in the same
  production wave or in staged waves,
- monitor production artifacts and checkpoint health,
- create Phase 2 implementation issue for inference/eval/`val200`,
- create final research memo task after measured results exist.
