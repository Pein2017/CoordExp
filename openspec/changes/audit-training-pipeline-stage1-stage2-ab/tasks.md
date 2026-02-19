## 1. Baseline And Scope

- [ ] 1.1 Capture the exact operational entrypoints + configs under audit (stage_1 and stage_2 AB) and record the resolved `training.run_name`, `training.output_dir`, and `training.logging_dir` for each.
- [ ] 1.2 Produce a one-page “pipeline map” (data -> transforms/packing -> training/inference -> artifacts) with concrete owner modules (file paths) for each boundary.
- [ ] 1.3 Reconcile operator-facing entrypoints in docs vs repo reality (e.g., `scripts/train_stage2.sh` vs references to `scripts/stage2_ab_server_train.sh`) and define the desired single canonical name.
- [ ] 1.4 Inventory existing CPU-runnable tests already covering stage_1 / stage_2 AB contracts (start from `tests/test_stage2_ab_training.py`, `tests/test_rollout_matching_sft.py`, `tests/test_packing_wrapper.py`) and list known gaps.

## 2. Config Loading And Contract Strictness

- [ ] 2.1 Verify strict unknown-key fail-fast for the stage_1 and stage_2 configs (including nested `stage2_ab.*` and `rollout_matching.*`) via existing config-strictness tests; add a targeted regression if any uncovered section is silently ignored.
- [ ] 2.2 Verify Stage-2 AB “profile leaf explicitness” for `configs/stage2_ab/prod/ab_mixed.yaml` matches the `stage2-ab-training` spec (required pinned high-signal keys); add/adjust the contract test if needed.
- [ ] 2.3 Validate coherence constraints among `global_max_length`, vLLM `max_model_len`, and `rollout_matching.max_new_tokens` (no silent truncation surprises); add fail-fast checks or diagnostics where currently ambiguous.
- [ ] 2.4 Confirm coord-token mode invariants for the audited configs (`custom.coord_tokens.enabled`, `custom.coord_tokens.skip_bbox_norm`, and the `<|coord_0|>.. <|coord_999|>` id range) and add a unit test that detects double-normalization or id-range drift.
- [ ] 2.5 Verify that server-launcher runtime knobs that are intentionally config-free (e.g., vLLM server `gpu_memory_utilization`, `enforce_eager`, dtype) are not simultaneously present as misleading YAML keys; either plumb them through preflight or document the separation explicitly.
- [ ] 2.6 Verify ms-swift TrainArguments placeholder dataset requirements remain satisfied (`data.dataset`/`data.val_dataset` dummy placeholders) and that our loader does not accidentally drop them while still using `custom.train_jsonl`/`custom.val_jsonl` as the true data source.

## 3. Data Factory And Data Processing (Raw -> Cooked -> Samples)

- [ ] 3.1 Trace the code path from `custom.train_jsonl` / `custom.val_jsonl` to dataset construction, including any caching/cooking layers, and document the intermediate artifacts (if any) and their schemas.
- [ ] 3.2 Verify JSONL contract validation is strict and fail-fast for cooked GT (no silent object drops) and enumerate the explicit edge-case policies (empty objects, invalid arity, missing width/height).
- [ ] 3.3 Verify geometry invariants in preprocessing/augmentation: coordinate order is preserved, no coord dropping/reordering, and training-side `do_resize=false` is respected end-to-end.
- [ ] 3.4 Audit sampling and determinism: `train_sample_limit`, `val_sample_limit`, with/without replacement, multi-worker shuffling/seed handling, and packing-buffer determinism under DDP.
- [ ] 3.5 Verify image path resolution and `ROOT_IMAGE_DIR` handling is consistent across learner and rollout server (server-mode), including relative-path behavior.
- [ ] 3.6 Audit offline public-data pipeline drop policies (object drops, record drops, max-object filtering) and ensure every drop mode is surfaced as explicit counters in manifests and carried into run artifacts for audit signoff.
- [x] 3.7 Verify dataset seeding is derived from `training.seed` (or is explicitly recorded in run artifacts) rather than hardcoded constants; add a regression test that detects seed drift across entrypoints.
- [ ] 3.8 Add a CPU-only determinism probe for a tiny dataset slice with `data.dataloader_num_workers>0` to detect order-sensitive RNG behavior (sample_id sequence + packed lengths stable across two runs, or explicitly documented nondeterminism with rationale).

## 4. Chat Template Construction (Qwen3-VL, Causal AR)

- [ ] 4.1 Trace prompt building for stage_1 and stage_2 AB: system/user/assistant message construction, role boundaries, and special tokens (`<|im_start|>`, `<|im_end|>`) under the Qwen3-VL chat template.
- [x] 4.2 Validate assistant rendering format: CoordJSON container `{\"objects\": [...]}` with bare coord-token literals in geometry, plus key-order handling for `custom.object_field_order` (desc-first vs geometry-first).
- [x] 4.3 Verify the parse boundary contract: assistant CoordJSON must be transpiled to strict JSON before `json.loads` for downstream matching/eval; add an inventory gate to prevent reintroducing direct `json.loads` on CoordJSON.
- [x] 4.4 Verify tokenizer consistency across learner and rollout server in vLLM server mode: identical vocab/special tokens/coord-token IDs, and prompt-token-id alignment checks are enforced with actionable errors.
- [x] 4.5 Add/extend regression tests using `scripts/tools/inspect_chat_template.py` fixtures to lock down canonical serialization + tokenization for both `desc_first` and `geometry_first`.
- [x] 4.6 Add an upstream integration gate for runtime resizing: forbid ms-swift runtime rescaling for training/infer (images must be pre-rescaled offline). Treat `template.max_pixels` as a hard input constraint and fail fast if any record exceeds it (do not silently rescale); add a CPU-only regression that verifies oversize inputs raise with actionable guidance.

## 5. Packing + Masks (Packed Sequence Training)

- [x] 5.1 Audit packing wrapper + collator behavior: multimodal fields are preserved per-sample, concatenation order is stable, and `position_ids` generation remains correct for Qwen3-VL (mRoPE) under packing.
- [x] 5.2 Validate attention mask correctness under packing (no cross-sample leakage, correct causal masking, correct padding-free behavior) and ensure label masking aligns with packed boundaries.
- [x] 5.3 Validate Stage-2 post-rollout packing selection behavior (deterministic, no segment splitting, oldest-segment inclusion) and ensure supervision offsets remain correct after packing.
- [x] 5.4 Add/extend unit tests for packing-related gradient correctness and masking invariants (start from `tests/test_stage2_ab_packing_mask_gradients.py` and `tests/test_packing_wrapper.py`).
- [x] 5.5 Add an integration gate for attention backend compatibility under packing: when `training.packing=true`, require a known-safe attention implementation for padding-free packed training (and fail fast or emit an actionable warning if misconfigured).

## 6. Model Forward + Loss Composition (Correctness)

- [ ] 6.1 Stage-1 objective audit: base full-vocab CE applies only to non-coord targets; coord-soft losses (softCE/W1/gate) apply only at coord labels; reduction/normalization is packing-safe and does not dilute under grad-accum.
- [ ] 6.2 Coord-offset adapter audit: `tie_head` semantics are correct, optimizer parameter groups match config LRs, and merged-export behavior is consistent with `docs/training/COORD_OBJECTIVE_AND_ADAPTER.md`.
- [ ] 6.3 Channel-A audit (Stage-2 AB): verify the A1 CE anchor split vs final-iteration geometry loss, `n_softctx_iter` semantics, and `softctx_grad_mode` behavior (unroll vs em_detach) match `progress/full_idea.md`.
- [ ] 6.4 Channel-B audit (Stage-2 AB): verify rollout prefix + strict parse + match + FN injection builds a single teacher-forced target; verify FP-neutral and closure-supervision masking semantics match the runbook.
- [ ] 6.5 Add targeted unit tests that validate shape consistency, masking correctness, and gradient propagation for the coord-logit pathways (without requiring full model weights).

## 7. Training Infrastructure (Logging, Eval, Checkpoints)

- [ ] 7.1 Audit logging/metrics flow: payload contract versioning, aggregation semantics under grad-accum/DDP, and “best-effort diagnostics vs fail-fast objective” boundaries.
- [ ] 7.2 Audit evaluation triggers and behavior for stage_1 and stage_2 AB (eval_strategy, eval_steps, eval_packing) and confirm eval does not silently change semantics under packing.
- [ ] 7.3 Verify checkpoint saving policy matches project intent (weight-only persistence; no full trainer state). If behavior differs between stage_1 and stage_2 trainers, make it explicit and test it.
- [ ] 7.4 Add a reproducibility “run manifest” requirement: persist resolved config, key environment metadata, and seed information under `training.output_dir` for every run.
- [x] 7.5 Extend the run manifest to include upstream provenance (transformers/torch/vllm/swift versions, ms-swift git SHA + dirty status, and rollout-server launch flags actually used) so results remain paper-auditable across upstream drift.

## 8. Stage-2 AB Stability (vLLM Server + Learner Interaction)

- [ ] 8.1 Audit Stage-2 launcher preflight contract resolution and validation (backend/mode/server URLs/model path, `ROOT_IMAGE_DIR`, and vLLM `max_model_len`); ensure failures are early and actionable.
- [ ] 8.2 Audit server/learner topology safeguards in the launcher: disjoint GPU sets, derived DP/TP behavior, readiness polling, and cleanup/termination behavior.
- [ ] 8.3 Audit vLLM server-mode weight sync control-flow under DDP: rank0-only sync, strict barriers, and “all ranks take identical control-flow” invariants to prevent deadlocks.
- [ ] 8.4 Verify Channel A/B deterministic scheduler correctness (Bresenham schedule) and confirm `stage2_ab/b_ratio_realized` matches expectation over long horizons.
- [ ] 8.5 Enumerate abnormal behaviors (invalid rollouts, truncation at max_new_tokens, closure-marker alignment failures, server timeouts) and ensure each has a deterministic fallback and diagnosis metric.
- [ ] 8.6 Verify launcher config resolution is self-consistent: the YAML used by the learner and the arguments passed to the rollout server do not drift in ways that change effective behavior silently (tokenizer/model path, max_model_len, LoRA enablement, sampling knobs).
- [ ] 8.7 Audit any Stage-2 rollout queue / staleness controls (versioning, windowing, queue limits, drop/backpressure behavior) and ensure off-policy gap is bounded and observable when async pathways are enabled.
- [x] 8.8 Verify the combined stage-2 launcher behaves intentionally for multi-server configs: if `rollout_matching.vllm.server.servers` has length > 1, either fail fast with actionable guidance or document the required external orchestration (do not silently ignore additional servers).
- [ ] 8.9 Expand DDP-safe weight sync audit to include failure propagation: any rank0 sync failure must trigger a synchronized global abort (or equivalent) so non-rank0 learners do not hang or continue with partially updated rollout state.
- [ ] 8.10 Audit rollout seeding semantics end-to-end (including seed=0 edge cases) across learner, vLLM server, and ms-swift RequestConfig handling to prevent hidden nondeterminism/diversity drift.
- [x] 8.11 Add a “run from non-repo cwd” validation for the stage-2 launcher preflight: relative paths in configs (JSONL paths, model paths, root image dir) must resolve relative to the config or repo root, not `Path.cwd()`.

## 9. Stage-2 Training Performance (Bottlenecks)

- [ ] 9.1 Identify throughput bottlenecks on the Channel-B path using existing `time/*` metrics (generate/parse/match/encode/pack) and confirm timings are emitted only when Channel-B executes.
- [ ] 9.2 Audit debug/monitor dumps (`monitor_dump`, `debug_dump`) for bounded overhead and safety; ensure large dumps cannot stall training or exhaust disk unexpectedly.
- [ ] 9.3 Propose config-first optimizations (no new CLI flags): decode microbatching, server DP/TP choice, packing fill ratio knobs, and schedule `b_ratio` trade-offs; define how to verify improvements when GPUs are available.

## 10. Hidden Risks Inventory (Broad Scan)

- [ ] 10.1 Run/extend silent-failure policy scans for core training/eval paths (no blanket exception suppression) and confirm best-effort behavior is limited to I/O-only sinks.
- [ ] 10.2 Audit docs/README drift that could cause operator error (entrypoint naming, required env vars like `ROOT_IMAGE_DIR`, and “server vs colocate” backend selection).
- [ ] 10.3 Verify no new behavior is introduced via ad-hoc flags; all new knobs must be YAML-specified and strict-parsed.
- [ ] 10.4 Create an “audit report” (P0/P1/P2) with file:line evidence and explicit verification commands for each finding; store it alongside this change for review.

## 11. Deferred GPU Smokes (Run Later)

- [ ] 11.1 (Deferred; GPU required) Run a short stage_1 smoke using `scripts/train.sh` with the audited stage_1 config and a debug/sample limit; verify metrics keys, packing behavior, and checkpoint artifacts.
- [ ] 11.2 (Deferred; GPU required) Run a stage_2 AB server-mode smoke using `scripts/train_stage2.sh` with `configs/stage2_ab/smoke/ab_mixed.yaml`; verify readiness gates, weight sync, A/B schedule telemetry, and Channel-B rollout metrics.

## 12. Upstream Integration Contracts (Transformers / ms-swift / vLLM / Torch)

- [x] 12.1 Record upstream dependency provenance for this audit: ms-swift source path + git SHA, and `transformers`, `torch`, `vllm`, `swift` versions from env `ms`; ensure they are also captured in the run manifest task (7.5).
- [x] 12.2 Add an upstream API contract test for transformers `Trainer` helper methods used by CoordExp optimizer wiring (signature/behavior drift should fail fast against the pinned transformers version).
- [ ] 12.3 For `rollout_backend=hf`, add a length-coherence gate: validate `prompt_len + max_new_tokens` against model `max_position_embeddings` and fail fast (or warn loudly) in truncation-risk regimes.
- [ ] 12.4 Pin ms-swift rollout request/response schema assumptions: server-mode rollouts must use `return_details=True` and responses must include `prompt_token_ids` and per-choice `token_ids`; add CPU-only contract tests around request construction and response validation.
- [x] 12.5 Add a compatibility gate for `swift rollout` CLI flag surface used by `scripts/train_stage2.sh` (flags must exist and remain semantically compatible across ms-swift upgrades).
- [ ] 12.6 Add a lightweight compatibility test for Qwen3-VL multimodal payload shape construction through the same code path used in Stage-2 rollouts (detect tuple/list serialization edge cases early, without a live server).
- [ ] 12.7 Pin the swift rollout HTTP endpoint surface the launcher depends on (`/health/`, `/infer/`, `/get_world_size/`, communicator init endpoints) and add a contract test that fails fast if endpoints drift or are disabled.
