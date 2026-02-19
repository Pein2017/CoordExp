## Context

This change is an audit + hardening pass over the end-to-end training pipeline described in:
- `progress/full_idea.md` (semantic baseline),
- Stage-1 launcher: `scripts/train.sh` + `configs/stage1/ablation/geometry_first_coco80.yaml`,
- Stage-2 AB server-mode launcher: `scripts/train_stage2.sh` + `configs/stage2_ab/prod/ab_mixed.yaml`.

Stage-2 AB adds an external vLLM rollout server (via `swift rollout`) plus packed teacher-forced learning
under two channels (A/B). Correctness depends on strict boundaries:
data contract -> chat-template render -> tokenize -> pack -> forward/loss masks -> metrics/eval -> artifacts.

This pipeline also depends on upstream library contracts (ms-swift, transformers, vLLM, torch). The audit
treats those dependencies as explicit integration surfaces: we verify versions/provenance are captured and
we add CPU-only contract tests for the specific upstream APIs/protocols CoordExp relies on.

Constraints (non-negotiable):
- Config-first (YAML); avoid adding new CLI hyperparameter flags.
- Preserve geometry invariants (never drop/reorder coords); training uses `do_resize=false`.
- Keep Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model internals (e.g., `modeling_qwen3_vl.py`).

Operational constraint for this investigation:
- GPU smoke runs are temporarily unavailable; we will prioritize CPU-only checks + unit tests now and defer
  GPU-dependent end-to-end runs as explicit tasks.

## Goals / Non-Goals

**Goals:**
- Produce a pipeline map (data -> transforms/packing -> training/inference -> artifacts) grounded in the
  operational entrypoints and the actual module owners in `src/`.
- Run a comprehensive audit/diagnosis pass with evidence handles (file path + symbol/line, config keys,
  or a test/command) for each risk area.
- Apply minimal fixes that improve correctness/reproducibility/eval validity, and add CPU-only tests that
  lock down invariants so regressions are caught early.
- Ensure Stage-2 AB rollout behavior is diagnosable with stable metrics and that known failure modes have
  deterministic fallbacks (no silent objective drift).

**Non-Goals:**
- No large architecture forks (no DETR-style heads, no bespoke RL loops, no custom vision backbones).
- No behavior changes that require new CLI flags.
- No broad refactors unrelated to correctness/reproducibility/eval validity.

## Decisions

1) **Audit method: breadth pass -> depth pass -> hardening**
   - Breadth: identify the true operational flow from configs + launchers + entrypoint wiring.
   - Depth: deep-dive the highest-risk boundaries (packing masks, template/tokenizer alignment, Stage-2
     rollout parsing + matching + FN injection, loss masking/weighting, checkpoint artifact contents).
   - Hardening: implement the smallest fixes + tests to prevent recurrence.

2) **Evidence gate**
   - No audit claim is “done” unless it has a concrete handle:
     - `path:line` evidence, or
     - a config key path, or
     - a test/command with an expected outcome.

3) **CPU-first verification now; GPU smokes later**
   - Add/extend unit tests and lightweight tools that validate invariants without GPUs.
   - Encode GPU-dependent operational smokes as deferred tasks with exact commands and pass criteria.

4) **Spec-aligned, config-first changes**
   - Prefer tightening validation and adding tests over changing semantics.
   - If a true requirement change is needed, capture it explicitly in a follow-up spec change.

## Risks / Trade-offs

- **[Risk] Some failures only reproduce under GPU/vLLM load (deadlocks, throughput collapse).**
  -> Mitigation: strengthen CPU contract tests (DDP control-flow invariants, request chunking, parsing)
  and defer a small number of operational smokes with explicit acceptance checks.

- **[Risk] Making validation stricter can cause existing runs to fail earlier.**
  -> Mitigation: limit fail-fast to objective-changing or eval-invalidating cases, and ensure errors are
  actionable (full dotted-path keys, suggested remediation).

- **[Risk] Adding diagnostics can bloat logs or add overhead.**
  -> Mitigation: keep diagnostics aggregate-only, bounded, and best-effort unless they affect the loss.

## Operational Entrypoints Under Audit

This audit is anchored to the *operational* entrypoints that are intended to be used by operators:

- Stage-1 launcher: `scripts/train.sh`
  - Anchored config: `configs/stage1/ablation/geometry_first_coco80.yaml`
- Stage-2 AB server-mode launcher (rollout server + learner): `scripts/train_stage2.sh`
  - Anchored config: `configs/stage2_ab/prod/ab_mixed.yaml`

### Resolved Run Identity And Directories (Materialized YAML)

We treat the following as the stable run identity inputs (post-`extends`, pre-runtime auto-versioning).
Both launchers `cd` into the repo root before executing Python, so relative dirs resolve under repo root.

Stage-1 (`scripts/train.sh` + `configs/stage1/ablation/geometry_first_coco80.yaml`):
- `training.run_name`: `epoch_4-softce_w1-coco80-geometry_first-LRs-2e-4_1e-4_4e-4-from-base-4B`
- `training.output_dir`: `./output/stage1/coco_bbox_max60-coco80-geometry_first`
- `training.logging_dir`: `./tb/stage1/coco_bbox_max60-coco80-geometry_first`

Stage-2 AB (`scripts/train_stage2.sh` + `configs/stage2_ab/prod/ab_mixed.yaml`):
- `training.run_name`: `prod_ab_mixed_bbox_max60_ckpt1516_ep2`
- `training.output_dir`: `output/stage2_ab/prod/ab_mixed`
- `training.logging_dir`: `tb/stage2_ab/prod/ab_mixed`

Verification (CPU-only; avoids ms-swift TrainArguments side-effects like deepspeed init):

```bash
PYTHONPATH=. conda run -n ms python -c "import os; from src.config.loader import ConfigLoader; tc=ConfigLoader.load_materialized_training_config('configs/stage1/ablation/geometry_first_coco80.yaml'); print(tc.training['run_name']); print(os.path.abspath(tc.training['output_dir'])); print(os.path.abspath(tc.training['logging_dir']))"

PYTHONPATH=. conda run -n ms python -c "import os; from src.config.loader import ConfigLoader; tc=ConfigLoader.load_materialized_training_config('configs/stage2_ab/prod/ab_mixed.yaml'); print(tc.training['run_name']); print(os.path.abspath(tc.training['output_dir'])); print(os.path.abspath(tc.training['logging_dir']))"
```

Note on runtime `output_dir` auto-versioning:
- ms-swift (and/or our config integration) materializes a per-run directory under:
  - `<training.output_dir>/<training.run_name>/v{N}-YYYYMMDD-HHMMSS`
- The `v{N}` component is auto-incremented based on existing directories, so it is **not stable** across
  repeated invocations and should not be treated as a “resolved constant” in audit artifacts.

## Pipeline Map (One Page)

The following map lists the concrete owner modules (repo file paths) for each pipeline boundary, and the
high-signal YAML keypaths that drive that boundary for the anchored entrypoints.

**1) Launch + Entrypoints**
- Stage-1 launcher: `scripts/train.sh`
- Stage-2 launcher: `scripts/train_stage2.sh` (also launches `swift rollout`)
- Python entrypoint (both): `src/sft.py` (`python -m src.sft --config <...>`)

Key YAML:
- `training.*` (run_name, output_dir, logging_dir, save/eval strategies, batch sizes)
- `model.*` (model path, dtype/attn backend, template selection)
- `custom.*` (dataset + prompt + coord-token policies)

Artifacts:
- `run_metadata.json` (run manifest/provenance; see `src/sft.py`)

**2) Config Loading + Strictness**
- YAML merge/materialization + strict schema: `src/config/loader.py`, `src/config/schema.py`
- Prompt defaults: `src/config/prompts.py` (`get_template_prompts`)

Key YAML:
- `extends:` (YAML inheritance)
- `custom.*` prompt keys (`custom.extra.prompt_variant`, `custom.object_field_order`, overrides)

**3) Data Ingestion + Sample Construction**
- Primary dataset (stage_1 + stage_2): `src/datasets/dense_caption.py`
- Message builder (single-image JSONL -> messages): `src/datasets/builders/jsonlines.py` (`JSONLinesBuilder`)
- Fusion dataset (legacy/experimental): `src/datasets/unified_fusion_dataset.py`

Key YAML:
- `custom.train_jsonl`, `custom.val_jsonl`
- `custom.train_sample_limit`, `custom.val_sample_limit`
- `data.dataloader_num_workers`, `data.dataloader_pin_memory`

Artifacts:
- Cooked JSONL files under `public_data/**` (offline pipeline outputs; see task 3.6)

**4) Geometry + Coord Tokens**
- Geometry utilities and invariants: `src/datasets/geometry.py`
- CoordJSON transpilation to strict JSON: `src/utils/coordjson_transpiler.py`
- Downstream parsing boundary (rollout/eval): `src/common/prediction_parsing.py`

Key YAML:
- `custom.coord_tokens.*` (enabled, skip_bbox_norm, etc.)
- `custom.object_field_order` (desc_first vs geometry_first)
- `template.max_pixels` (hard cap; runtime resizing forbidden)

**5) Chat Template Render + Tokenization (Qwen3-VL / causal AR)**
- Prompt selection: `src/config/prompts.py`
- Message materialization: `src/datasets/builders/jsonlines.py`
- Upstream template execution (ms-swift, Qwen3-VL): external (`/data/ms-swift/...`)

Key YAML:
- `model.template` / template selection and special token conventions
- `custom.user_prompt`, `custom.system_prompt` (or prompt variants)

**6) Packing + Collation + Masks**
- Packing wrapper (deterministic grouping + template flags): `src/datasets/wrappers/packed_caption.py`
- Packing/mask correctness assertions and tests:
  - `tests/test_packing_wrapper.py`
  - `tests/test_stage2_ab_packing_mask_gradients.py`
  - `tests/test_packed_labels_and_coord_targets.py`

Key YAML:
- `training.packing` (+ packing fill knobs under `custom.packing.*` if used)
- `model.attn_impl` (required flash-attn backend under packing; enforced in `src/sft.py`)

**7) Trainer Forward + Loss Composition**
- Stage-1 / SFT objective wiring: `src/sft.py` + ms-swift Trainer surfaces
- Stage-2 AB learner: `src/trainers/stage2_ab_training.py`
- Rollout-matching trainer components: `src/trainers/rollout_matching_sft.py`
- Loss helpers (Stage-1): `src/trainers/losses/*`, `src/trainers/metrics/mixins.py`

Key YAML:
- `stage2_ab.*` (schedule b_ratio, softctx iterations, channel semantics)
- `rollout_matching.*` (backend, max_new_tokens, decode batch sizing, debug/monitor dumps)

**8) Evaluation + Metrics + Logging**
- Trainer metric emission sites:
  - Stage-2 AB: `src/trainers/stage2_ab_training.py`
  - Rollout-matching: `src/trainers/rollout_matching_sft.py`
- TensorBoard/WandB wiring is largely upstream (ms-swift/transformers), with run metadata in `src/sft.py`.

Key YAML:
- `training.do_eval`, `training.eval_strategy`, `training.eval_steps`
- `training.report_to`, logging sinks

**9) Checkpointing + Artifacts**
- Checkpoint policy and metadata persistence:
  - `src/sft.py` (run metadata + config snapshot logic)
  - trainer save hooks in `src/trainers/*` and upstream Trainer

Key YAML:
- `training.save_strategy`, `training.save_steps`

**10) Stage-2 AB Rollout Server Integration (vLLM server-mode)**
- Launcher preflight contract resolver: `src/trainers/rollout_matching/preflight.py`
- Combined launcher (server + learner): `scripts/train_stage2.sh`
- Learner/server interaction logic:
  - `src/trainers/rollout_matching_sft.py`
  - `src/trainers/stage2_ab_training.py`

Key YAML:
- `rollout_matching.rollout_backend`
- `rollout_matching.vllm.mode: server`
- `rollout_matching.vllm.server.servers` (single-server enforced)

## Existing CPU-Runnable Contract Tests (Inventory)

High-signal CPU-only tests already covering critical boundaries:

- Template / CoordJSON rendering and parse-boundary:
  - `tests/test_chat_template_regression.py`
  - `tests/test_coordjson_transpiler.py`
  - `tests/test_coordjson_parsing_inventory_gate.py`
  - `tests/test_coordjson_parse_boundary_inventory.py`
- Packing + masks + packed-boundary correctness:
  - `tests/test_packing_wrapper.py`
  - `tests/test_stage2_ab_packing_mask_gradients.py`
  - `tests/test_packed_labels_and_coord_targets.py`
  - `tests/test_stage2_ab_channel_local_packing_buffer.py`
- Stage-2 AB + server-mode contracts:
  - `tests/test_stage2_preflight_path_resolution.py`
  - `tests/test_stage2_ab_prompt_alignment_contract.py`
- Runtime rescale prohibition / max_pixels:
  - `tests/test_max_pixels_enforcement.py`
- Upstream integration contracts:
  - `tests/test_transformers_trainer_contract.py`
  - `tests/test_swift_rollout_cli_flags.py`

Known gaps (to be covered by pending tasks):
- Stage-1/Stage-2 objective masking and loss-weight correctness (Section 6).
- Stage-2 AB DDP failure propagation semantics for rollout-server weight sync (8.9).
- ms-swift rollout request/response schema pinning (12.4) + endpoint surface contracts (12.7).
