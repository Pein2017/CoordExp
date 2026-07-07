# Tasks

Implementation is blocked until this OpenSpec baseline is reviewed and the
user explicitly authorizes implementation. These tasks define the intended
roadmap and acceptance gates.

## 1. OpenSpec Baseline Draft And Validation

- [x] 1.1 Draft `design.md` from `DECISIONS.md`, `BLUEPRINT.md`, and accepted
  review verdicts.
- [x] 1.2 Draft `coordexp-swift-config-runtime` delta spec.
- [x] 1.3 Draft `coordexp-swift-data-template-encoding` delta spec.
- [x] 1.4 Draft `coordexp-swift-packing-forward` delta spec.
- [x] 1.5 Draft `coordexp-swift-supervision-losses` delta spec.
- [x] 1.6 Draft `coordexp-swift-adapters-embeddings-optim` delta spec.
- [x] 1.7 Draft `coordexp-swift-training-artifacts` delta spec.
- [x] 1.8 Draft `coordexp-swift-vertical-smoke` delta spec.
- [x] 1.9 Run OpenSpec validation and doc hygiene checks.
- [x] 1.10 Run four read-only review lanes and patch accepted P0/P1 findings.

## 2. Source Studies, Probes, And Legacy Invariant Inventory

- [x] 2.1 Study MS-Swift, Transformers, PEFT, and local usage for DoRA,
  `use_dora`, LoRA target discovery, and any existing DoRA-like behavior.
- [x] 2.2 Define the public `dora` adapter as PEFT DoRA-backed LoRA using
  `LoraConfig(use_dora=True)`, and reject `dlora` as a V1 schema value.
- [x] 2.3 Build a minimal DoRA round-trip probe that verifies initialization,
  save, load, forward compatibility, and DoRA magnitude-vector coverage when
  applicable before `adapter.type: dora` validates.
- [x] 2.4 Study custom Qwen wrappers, PEFT `TrainableTokens`, and LoRA
  `trainable_token_indices` for selected special-token embedding deltas.
- [x] 2.5 Decide and document the embedding mechanism that preserves selected
  full embedding updates, tied input/output behavior, compact checkpoints,
  additive-versus-absolute payload semantics, and base-plus-adapter-plus-delta
  loading.
- [x] 2.6 Study installed Transformers Qwen3-VL/Qwen2-VL processor behavior for
  no-resize image admissibility, `patch_size`, `merge_size`,
  `image_grid_thw`, placeholder expansion, and forward-output shape.
- [x] 2.7 Study and probe Qwen3-VL packed MRoPE requirements, including 4-row
  `[text,t,h,w]` position ids, per-segment reset points, and differences from
  running upstream helpers over a whole packed row.
- [x] 2.8 Study and probe FlashAttention varlen segment isolation, including
  attention implementation selection, dtype constraints, cumulative sequence
  lengths, max lengths, and branch-level evidence.
- [x] 2.9 Inventory legacy `src/` correctness invariants worth porting as tests
  or contracts, especially geometry round-trip, no-resize validation,
  coord-token span alignment, packing isolation, MRoPE reset points, and loss
  denominator semantics.
- [x] 2.10 Review source studies, probes, and legacy invariant inventory before
  source implementation begins.

## 3. Smoke Fixture Pinning And Materialization

- [x] 3.1 Locate current training JSONL sources with `len12000` naming
  and candidate single-image rows.
- [x] 3.2 Select at least two short valid single-image source rows with safe
  descriptions and no-resize-compatible image dimensions, including at least
  one exactly two-object row where practical.
- [x] 3.3 Materialize `tests/fixtures/smoke/qwen3_vl_single_image_pack/` with
  copied image file(s) under `images/`, canonical `examples.jsonl`,
  `config.yaml`, `checksums.json`, and README.
- [x] 3.4 Ensure generated smoke outputs go under ignored `run.artifact_root`,
  not under the fixture source directory.
- [x] 3.5 Generate and freeze `expected_rendered.json` only after the real
  renderer exists, using the renderer path rather than a hand-authored snapshot.

## 4. Archive, Skeleton, And Config Foundations

- [x] 4.1 Move old active `src/` intact to `reference/legacy_src/` after
  explicit implementation approval.
- [x] 4.2 Create minimal new `src/` package skeleton with no placeholder
  framework code and no old-source import shims.
- [x] 4.3 Implement small contract-error vocabulary.
- [x] 4.4 Implement strict config loading, inheritance, resolved config
  fingerprints, run identity, output directory policy, logits-memory budget
  checks, Qwen attention/dtype validation, and dry config trace.
- [x] 4.5 Implement planned-step schedule resolution from dataset length,
  world size, effective batch size, epochs, debug max steps, and deterministic
  tail-fill behavior.

## 5. Data, Template, And Qwen Encoding

- [x] 5.1 Implement `RawExample` loading, image path resolution, geometry
  validation, and canonical example ids.
- [x] 5.2 Implement English-only object/box template rendering with
  `source_order` and deterministic `random` ordering, while rejecting legacy
  `sorted` and reserving any future geometric sort for an explicitly approved
  name such as `geometry_sorted`.
- [x] 5.3 Implement assistant suffix and typed span validation for
  `supervised_response_text`.
- [x] 5.4 Implement Qwen component loading and tokenizer token-identity
  preflight for wrappers, coordinate tokens, and the `<|im_end|>\n` split.
- [x] 5.5 Implement no-resize Qwen image encoding, processor-derived
  admissible-dimension validation, explicit raw-pixel and merged-visual-token
  budget checks, and image-grid validation.
- [x] 5.6 Implement rendered-to-token span alignment and `EncodedExample`
  validation under `packing.global_max_length`.

## 6. Packing And Qwen Forward

- [x] 6.1 Implement no-padding pack planning with overflow commit behavior.
- [x] 6.2 Implement packed supervision remapping with invertible logical-to-
  physical position traces.
- [x] 6.3 Implement Qwen MRoPE position input construction for packed isolated
  segments with per-segment computation, 4-row `[text,t,h,w]` Qwen boundary
  validation, and reset points matching `PackedSegment` boundaries.
- [x] 6.4 Implement Qwen forward wrapper with model-side loss disabled,
  `use_cache=False`, no `inputs_embeds`, full-vocabulary logits over either the
  full sequence or explicitly selected supervised rows, output-shape validation,
  placeholder/grid validation, and Qwen forward receipt.
- [x] 6.5 Prove FlashAttention varlen segment isolation through branch-level
  evidence, not only shape checks.
- [x] 6.6 Implement deterministic packing-cache reuse with 16 CPU workers as
  the default cache-miss materialization policy, record the resolved worker
  count in packing/cache receipts, and verify that worker count does not affect
  semantic cache identity or packed micro-step order.

## 7. Losses, Normalizers, And Finite Gates

- [x] 7.1 Implement `TokenAtom`, `TokenSpan`, `TokenSequence`, same-segment
  target/logits position validation, and dense-label parity helpers.
- [x] 7.2 Implement `LossContext`, selected fp32 logits, `BaseTokenCE`,
  exact group-mass `TokenTypeGateLoss`, and closed V1 token-type vocabulary
  groups.
- [x] 7.3 Implement planned-step `segment_balanced` loss normalizers and
  backend scaling guards.
- [x] 7.4 Implement `LossRunner` and `LossBundle` with weighted per-term loss
  metrics, top-level `acc_top1`, top-level `acc_top5`, counts, and diagnostics.
- [x] 7.5 Implement pre-backward scalar finite gate and post-backward
  gradient/overflow gate with all-rank consensus.

## 8. Adapters, Embeddings, And Optimizer

- [x] 8.1 Implement adapter loading/initialization only after source-study gates
  are passed.
- [x] 8.2 Implement DoRA setup as the default adapter-enabled training path.
- [x] 8.3 Implement selected special-token embedding deltas and compact
  checkpoint payloads.
- [x] 8.4 Implement explicit optimizer group matching for vision, aligner,
  language, adapter parameters, and selected embedding deltas.
- [x] 8.5 Emit trainable-surface and optimizer-group receipts before the first
  backward pass.
- [x] 8.6 Implement the DoRA seed-mode hierarchy:
  `initialize_new`, `load_existing`, and `warm_start_expand_dora`.
- [x] 8.7 Implement config-driven target expansion where required targets are
  always created, complete source target tensors are reused by exact key,
  missing source targets remain freshly initialized, and partial source targets
  fail before optimizer construction.
- [x] 8.8 Load configured selected-token embedding delta payloads before
  optimizer construction for `warm_start_expand_dora`.

## 9. Trainer, Runtime, Artifacts, Metrics, Checkpoints, And Eval

- [x] 9.1 Implement `SupervisedTrainer` as loop orchestration over approved
  components.
- [x] 9.2 Implement `TrainRuntime` for device/backend mechanics, Accelerate
  integration, rank guards, backward, clipping, optimizer stepping, scheduler
  stepping, metric gathering, and rank-safe saves.
- [x] 9.3 Implement artifact manager, run manifest, resolved config artifacts,
  schedule receipt, subsystem receipts, and metric event streams.
- [x] 9.4 Implement checkpoint writer for adapter payloads, selected embedding
  deltas, metadata, unpadded step ids, `checkpoint-final.json`, and optional
  `best_acc_top1.json`.
- [x] 9.5 Implement `eval.forward` as packed forward-only evaluation using the
  same encoding, packing, Qwen forward, loss, and metric stack as training.

## 10. Five-Step Vertical Smoke Acceptance

- [x] 10.1 Run the DoRA adapter-enabled five-planned-step smoke after DoRA and
  selected-embedding source gates pass.
- [x] 10.2 Use real `packing.global_max_length`, fixture-local sample limit,
  `training.effective_batch_size: 2`, `max_steps: 5`, and two scheduled
  `eval.forward` runs, normally from explicit smoke `eval.forward.steps:
  [2, 4]`.
- [x] 10.3 Verify resolved config, manifest, Qwen setup receipt, pack plan,
  at least one multi-segment packed forward receipt, loss plan,
  trainable-surface receipt, optimizer receipt, metrics, eval-forward
  summaries, checkpoint metadata, and `checkpoints/checkpoint-final.json`.
- [x] 10.4 Verify DeepSpeed status labels `schema_accepted` and, when actually
  proven, `conflict_validation_implemented` without claiming
  `systems_smoke_verified` or `production_supported` until a later systems
  smoke passes.
- [x] 10.5 Stop at smoke acceptance; do not expand V1 with rollout training,
  hidden-state losses, hidden-state/KV/runtime feature caches, video,
  multi-image, vLLM, or exact resume unless a later approved change promotes
  them. Deterministic packing-cache reuse is an approved V1 infrastructure
  exception.
