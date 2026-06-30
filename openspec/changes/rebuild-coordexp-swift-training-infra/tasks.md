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

## 2. DLoRA And Special-Token Embedding Source Studies

- [ ] 2.1 Study MS-Swift, Transformers, PEFT, and local usage for DoRA,
  `use_dora`, LoRA target discovery, and any existing dLoRA-like behavior.
- [ ] 2.2 Define dLoRA precisely as DoRA-backed PEFT behavior or a
  CoordExp-owned mechanism.
- [ ] 2.3 Build a minimal dLoRA round-trip probe that verifies initialization,
  save, load, and forward compatibility before `adapter.type: dlora` validates.
- [ ] 2.4 Study custom Qwen wrappers, PEFT `TrainableTokens`, and LoRA
  `trainable_token_indices` for selected special-token embedding deltas.
- [ ] 2.5 Decide and document the embedding mechanism that preserves selected
  full embedding updates, tied input/output behavior, compact checkpoints, and
  base-plus-adapter-plus-delta loading.

## 3. Smoke Fixture Pinning And Materialization

- [ ] 3.1 Locate current training JSONL sources with `len12000` naming
  and candidate single-image rows.
- [ ] 3.2 Select or deterministically reduce one short valid two-object source
  row with safe descriptions and no-resize-compatible image dimensions.
- [ ] 3.3 Materialize `tests/fixtures/smoke/qwen3_vl_single_image_pack/` with
  copied image, canonical `examples.jsonl`, `config.yaml`,
  `checksums.json`, README, and `expected_rendered.json`.
- [ ] 3.4 Ensure generated smoke outputs go under ignored `run.artifact_root`,
  not under the fixture source directory.

## 4. Archive, Skeleton, And Config Foundations

- [ ] 4.1 Move old active `src/` intact to `reference/legacy_src/` after
  explicit implementation approval.
- [ ] 4.2 Create minimal new `src/` package skeleton with no placeholder
  framework code and no old-source import shims.
- [ ] 4.3 Implement small contract-error vocabulary.
- [ ] 4.4 Implement strict config loading, inheritance, resolved config
  fingerprints, run identity, output directory policy, and dry config trace.
- [ ] 4.5 Implement planned-step schedule resolution from dataset length,
  world size, effective batch size, epochs, and debug max steps.

## 5. Data, Template, And Qwen Encoding

- [ ] 5.1 Implement `RawExample` loading, image path resolution, geometry
  validation, and canonical example ids.
- [ ] 5.2 Implement English-only object/box template rendering with
  `source_order` and deterministic `random` ordering, while rejecting legacy
  `sorted` and reserving any future geometric sort for an explicitly approved
  name such as `geometry_sorted`.
- [ ] 5.3 Implement assistant suffix and typed span validation for
  `supervised_response_text`.
- [ ] 5.4 Implement Qwen component loading and tokenizer token-identity
  preflight for wrappers and coordinate tokens.
- [ ] 5.5 Implement no-resize Qwen image encoding, processor-derived
  admissible-dimension validation, explicit raw-pixel and merged-visual-token
  budget checks, and image-grid validation.
- [ ] 5.6 Implement rendered-to-token span alignment and `EncodedExample`
  validation under `packing.global_max_length`.

## 6. Packing And Qwen Forward

- [ ] 6.1 Implement no-padding pack planning with overflow commit behavior.
- [ ] 6.2 Implement packed supervision remapping with invertible logical-to-
  physical position traces.
- [ ] 6.3 Implement Qwen MRoPE position input construction for packed isolated
  segments.
- [ ] 6.4 Implement Qwen forward wrapper with model-side loss disabled,
  `use_cache=False`, no `inputs_embeds`, full logits, output-shape validation,
  placeholder/grid validation, and Qwen forward receipt.
- [ ] 6.5 Prove FlashAttention varlen segment isolation through branch-level
  evidence, not only shape checks.

## 7. Losses, Normalizers, And Finite Gates

- [ ] 7.1 Implement `TokenAtom`, `TokenSpan`, `TokenSequence`, target/logits
  position validation, and dense-label parity helpers.
- [ ] 7.2 Implement `LossContext`, selected fp32 logits, `BaseTokenCE`,
  `TokenTypeGateLoss`, and closed V1 token-type vocabulary groups.
- [ ] 7.3 Implement planned-step loss normalizers and backend scaling guards.
- [ ] 7.4 Implement `LossRunner` and `LossBundle` with weighted per-term loss
  metrics, top-level `acc_top1`, top-level `acc_top5`, counts, and diagnostics.
- [ ] 7.5 Implement pre-backward scalar finite gate and post-backward
  gradient/overflow gate with all-rank consensus.

## 8. Adapters, Embeddings, And Optimizer

- [ ] 8.1 Implement adapter loading/initialization only after source-study gates
  are passed.
- [ ] 8.2 Implement dLoRA setup as the default adapter-enabled training path.
- [ ] 8.3 Implement selected special-token embedding deltas and compact
  checkpoint payloads.
- [ ] 8.4 Implement explicit optimizer group matching for vision, aligner,
  language, adapter parameters, and selected embedding deltas.
- [ ] 8.5 Emit trainable-surface and optimizer-group receipts before the first
  backward pass.

## 9. Trainer, Runtime, Artifacts, Metrics, Checkpoints, And Eval

- [ ] 9.1 Implement `SupervisedTrainer` as loop orchestration over approved
  components.
- [ ] 9.2 Implement `TrainRuntime` for device/backend mechanics, Accelerate
  integration, rank guards, backward, clipping, optimizer stepping, scheduler
  stepping, metric gathering, and rank-safe saves.
- [ ] 9.3 Implement artifact manager, run manifest, resolved config artifacts,
  schedule receipt, subsystem receipts, and metric event streams.
- [ ] 9.4 Implement checkpoint writer for adapter payloads, selected embedding
  deltas, metadata, unpadded step ids, `checkpoint-final.json`, and optional
  `best_acc_top1.json`.
- [ ] 9.5 Implement `eval.forward` as packed forward-only evaluation using the
  same encoding, packing, Qwen forward, loss, and metric stack as training.

## 10. Five-Step Vertical Smoke Acceptance

- [ ] 10.1 Run the dLoRA adapter-enabled five-planned-step smoke after dLoRA and
  selected-embedding source gates pass.
- [ ] 10.2 Use real `packing.global_max_length`, fixture-local sample limit,
  `training.effective_batch_size: 1`, `max_steps: 5`, and two scheduled
  `eval.forward` runs, normally from explicit smoke `eval.forward.steps:
  [2, 4]`.
- [ ] 10.3 Verify resolved config, manifest, Qwen setup receipt, pack plan,
  loss plan, trainable-surface receipt, optimizer receipt, metrics,
  eval-forward summaries, checkpoint metadata, and `checkpoints/checkpoint-final.json`.
- [ ] 10.4 Verify DeepSpeed status labels `schema_accepted` and, when actually
  proven, `conflict_validation_implemented` without claiming
  `systems_smoke_verified` or `production_supported` until a later systems
  smoke passes.
- [ ] 10.5 Stop at smoke acceptance; do not expand V1 with rollout training,
  hidden-state losses, persistent caches, video, multi-image, vLLM, or exact
  resume unless a later approved change promotes them.
