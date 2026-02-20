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
- Required run manifest:
  - `resolved_config.json` (resolved YAML snapshot; see `src/utils/run_manifest.py`)
  - `runtime_env.json` (key env metadata; see `src/utils/run_manifest.py`)
  - `config_source.yaml` / `base_config_source.yaml` (best-effort copies of the input YAML files)
- Extended provenance:
  - `run_metadata.json` (git + upstream provenance + launcher metadata; see `src/sft.py`)

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
- Stage-1 evaluation (teacher-forced):
  - Uses upstream ms-swift/transformers evaluation loop over `eval_dataset` (optionally packed when `training.eval_packing=true`).
- Stage-2 evaluation (rollout-style; no teacher-forced loss):
  - `src/trainers/rollout_matching_sft.py:RolloutMatchingSFTTrainer.evaluate` runs: rollout -> parse -> Hungarian match.
  - Eval batches are `list[dict]` (identity collator); batching is controlled by `rollout_matching.decode_batch_size`
    via `src/sft.py:_apply_rollout_decode_batch_size_override`.
- Trainer metric emission sites:
  - Stage-2 AB: `src/trainers/stage2_ab_training.py`
  - Rollout-matching: `src/trainers/rollout_matching_sft.py`
- TensorBoard/WandB wiring is largely upstream (ms-swift/transformers), with run metadata in `src/sft.py`.

Key YAML:
- `training.do_eval`, `training.eval_strategy`, `training.eval_steps`
- `training.report_to`, logging sinks

**9) Checkpointing + Artifacts**
- Checkpoint policy and metadata persistence:
  - `src/utils/run_manifest.py` (required: resolved config + runtime env snapshots)
  - `src/sft.py` (extended: run metadata + upstream provenance)
  - trainer save hooks in `src/trainers/*` and upstream Trainer

Key YAML:
- `training.save_strategy`, `training.save_steps`
- `training.save_only_model: true` (weight-only persistence; no optimizer/scheduler/rng state)

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

## Prompt Construction Trace (Stage-1 / Stage-2 AB)

This section documents *how* the system/user/assistant turns are materialized into the exact text and
token sequences fed into Qwen3-VL, with emphasis on special tokens and prompt/target boundaries.

### Stage-1 (SFT) Teacher-Forced Prompt

Owner modules:
- Prompt selection: `src/config/prompts.py:get_template_prompts` (dense system/user wording, prompt variants, object-field order wording)
- Message materialization: `src/datasets/builders/jsonlines.py:JSONLinesBuilder.build_many`
- Learner encoding: `src/datasets/dense_caption.py:__getitem__` (`self.template.encode(merged, return_length=True)`)

Conversation shape:
- The builder emits one round of messages:
  - user: multimodal content list containing one or more `{"type": "image", "image": <url>}` entries followed by
    a `{"type": "text", "text": <user_prompt>}` entry.
  - assistant: `{"type": "text", "text": <CoordJSON>}` where CoordJSON is the *training target*.
- The system prompt is injected at chat-template render time (by the upstream ms-swift/Qwen3-VL template), not by the builder.

Special tokens + role boundaries:
- Qwen3-VL chat template renders message roles using `<|im_start|>{role}` and closes each turn with `<|im_end|>`.
- Image placeholders are rendered as `<|vision_start|><|image_pad|><|vision_end|>` within the user turn.
- Evidence: `tests/test_chat_template_regression.py` asserts `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`
  and the vision placeholder are present in the final chat text.

### Stage-2 AB Channel A (Teacher-Forced GT Payload)

Owner modules:
- Channel selection + batch prep: `src/trainers/stage2_ab_training.py:_prepare_batch_inputs_a`

Conversation shape:
- Dataset samples must include a `messages` list (same shape as Stage-1).
- For Channel A, the trainer replaces the assistant response with a teacher-forced payload derived from GT objects:
  - Build assistant CoordJSON payload: `_build_teacher_forced_payload(...)` + `dumps_coordjson(...)`
  - Tokenize the assistant text into ids: `template.tokenizer.encode(..., add_special_tokens=False)`
  - Inject token ids into the assistant message and run `template.encode(...)` to obtain full `input_ids` and `labels`.

Boundary alignment:
- Prompt/target boundary (`prompt_len`) is located by scanning `labels` for the first non-`-100` index.
- This is packing-safe and does not assume fixed chat-template layout (it follows the upstream template’s masking).

### Stage-2 AB Channel B (Rollout Prompt)

Operationally, Channel B generation happens on the rollout server (ms-swift + vLLM). The learner relies on:
- Prompt tokens are built from system + user turns, with a generation prompt that opens an assistant turn but does not close it.
- Evidence: `tests/test_chat_template_regression.py` asserts that `apply_chat_template(..., add_generation_prompt=True)`
  ends with an `<|im_start|>assistant` turn and that the tail contains no `<|im_end|>`.

### Cross-Process Tokenization Alignment (Learner vs Server)

Stage-2 AB must ensure the exact same tokenizer and chat-template boundary behavior between:
- the rollout server (prompt_token_ids produced during generation), and
- the learner (teacher-forced `template.encode(...)` used to compute loss masks/offsets).

Evidence:
- The stage-2 learner asserts prompt-token-id prefix equality and prompt_len equality when `prompt_token_ids` are available:
  `src/trainers/stage2_ab_training.py` (prompt alignment checks inside batch prep).
- Additional alignment guards exist for rollout-matching trainer flows:
  `src/trainers/rollout_matching_sft.py` (prompt-token-id alignment and tokenizer consistency checks).

## Stage-2 Performance Instrumentation (Channel A/B)

Stage-2 AB has two execution regimes:
- **Channel A**: teacher-forced (no rollouts).
- **Channel B**: rollout-matching (external rollout server + parse/match + post-rollout packing).

### Timing Metrics (`time/*`)

Rollout-matching trainer emits:
- `time/forward_s`: learner forward/backward wall time for an optimizer step.
- `time/mask_build_s`: packed attention/label mask construction time.
- `time/rollout_generate_s`: vLLM server generation wall time (**only when Channel B runs**).
- `time/rollout_parse_match_s`: parse + match time (**only when Channel B runs**).
- `time/rollout_teacher_encode_s`: teacher-encode time for the derived training target (**only when Channel B runs**).
- `time/post_rollout_pack_s`: post-rollout repacking time (only when repacking occurred).

Stage-2 AB Channel A batch path emits:
- `time/channel_a_teacher_encode_s`
- `time/channel_a_pack_s`

Contract:
- `time/rollout_*` keys must not be emitted as misleading `0.0` scalars on Channel A steps.

Evidence:
- `tests/test_rollout_time_metrics_gating.py`
- `tests/test_stage2_ab_time_metrics_gating.py`

### Debug/Monitor Dumps (`monitor_dump`, `vllm.server.debug_dump`)

CoordExp uses two qualitative dump channels for diagnosing Stage-2 behavior:
- `rollout_matching.monitor_dump`: per-step structured payload (JSON + optional Markdown) including messages, rollout text, prefix, derived training target, and parse/match stats.
- `rollout_matching.vllm.server.debug_dump`: minimal GT text vs rollout text dumps to diagnose server-mode formatting without printing giant blobs to stdout.

These dumps can be large (and are intentionally allowed to be unbounded via `max_text_chars<=0` / `max_chars<=0`).
They must not destabilize training or silently “brick” runs via filesystem issues.

Safety rails implemented:
- **Best-effort I/O:** dump write failures do not crash training (warnings only).
- **Async write by default:** dump I/O is offloaded to a single-thread executor.
- **Bounded async queue:** `max_pending_writes` bounds in-flight dump tasks; new dumps are skipped if the queue is full.
- **Low-disk guard:** `min_free_gb` skips dumps when free disk is below a threshold (prevents “disk full” surprises caused by diagnostics).
- **Existing hard caps:** `max_events`, `max_samples` continue to bound total dump volume.

Key YAML:
- `rollout_matching.monitor_dump.{enabled,every_steps,dump_first_step,only_world_process_zero,max_events,max_samples,max_text_chars,async_write,max_pending_writes,min_free_gb,out_dir,write_markdown}`
- `rollout_matching.vllm.server.debug_dump.{enabled,every_steps,dump_first_step,only_world_process_zero,max_events,max_samples,max_chars,async_write,max_pending_writes,min_free_gb,out_dir}`

Evidence:
- `tests/test_monitor_dump_io_best_effort.py`
- `tests/test_dump_clip_text_safety.py`
- `tests/test_vllm_server_debug_dump_io_best_effort.py`

## Config-First Stage-2 Performance Optimization Knobs (When GPUs Are Available)

This audit intentionally avoids adding new CLI hyperparameter flags. The primary optimization levers are YAML knobs.

How to verify improvements once GPUs are available:
- `rollout/gen_tokens_per_s` (higher is better; held against quality/eval).
- `time/rollout_generate_s`, `time/rollout_parse_match_s`, `time/post_rollout_pack_s` (identify dominant bottleneck).
- `stage2_ab/b_ratio_realized` (confirm schedule matches intended compute split).
- GPU-side: vLLM server memory utilization, request queueing, and “prefill vs decode” balance (out-of-band profiling).

High-leverage knobs (ordered by typical impact):
- Reduce rollout work:
  - `rollout_matching.max_new_tokens` (hard cap on decode cost).
  - `stage2_ab.schedule.b_ratio` (rollout frequency vs teacher-forced frequency).
  - `rollout_matching.decode_batch_size` (too small underutilizes server; too large increases latency/OOM risk).
- Improve vLLM server throughput (server-mode):
  - `rollout_matching.vllm.tensor_parallel_size` and rollout-server `data_parallel_size` (via launcher flags).
  - `rollout_matching.vllm.max_model_len` (must be coherent with `global_max_length`).
  - vLLM engine knobs (when safe/needed): `enable_chunked_prefill`, `max_num_seqs`, `max_num_batched_tokens`.
- Improve learner packing efficiency:
  - `training.packing: true` (already required).
  - Packing wrapper knobs (stage_1 + stage_2): `custom.packing_length`, `custom.packing_buffer`, `custom.packing_min_fill_ratio`, `custom.packing_drop_last`, `custom.packing_allow_single_long`.
  - Post-rollout packing telemetry: `packing/post_rollout_fill`, `packing/post_rollout_segments`, `time/post_rollout_pack_s`.
- Reduce diagnostics overhead in production runs:
  - Disable or downsample `monitor_dump` / `debug_dump` once a run is stable.

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
  - `tests/test_stage2_launcher_preflight_contract.py`
  - `tests/test_stage2_ab_prompt_alignment_contract.py`
  - `tests/test_ddp_vllm_sync_failure_propagation.py`
  - `tests/test_stage2_ab_bresenham_schedule_long_horizon.py`
  - `tests/test_rollout_seed_nonzero_contract.py`
- Runtime rescale prohibition / max_pixels:
  - `tests/test_max_pixels_enforcement.py`
- Upstream integration contracts:
  - `tests/test_transformers_trainer_contract.py`
  - `tests/test_swift_rollout_cli_flags.py`
  - `tests/test_rollout_hf_length_coherence_gate.py`
  - `tests/test_vllm_server_rollout_contract.py`
  - `tests/test_vllm_server_multimodal_payload_contract.py`
  - `tests/test_swift_rollout_endpoints_contract.py`
- Training infrastructure contracts:
  - `tests/test_run_manifest_files.py`
  - `tests/test_checkpoint_weight_only_policy.py`
  - `tests/test_stage2_pending_metrics_aggregation.py`
  - `tests/test_trainer_metrics_payload_contract.py`
  - `tests/test_batch_extras_failure_not_silent.py`

Known gaps (to be covered by pending tasks):
- Broad silent-failure scan + audit report (Section 10).
- Deferred GPU smokes (Section 11).
