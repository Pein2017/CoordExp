## Why

CoordExp-swift needs a clean, self-contained training infrastructure baseline so
future research work can be understood, debugged, and extended without
reconstructing context across MS-Swift, Transformers, old `src/`, and scattered
historical OpenSpec contracts. This change turns the reviewed architecture
blueprint into the first stable OpenSpec contract set for the rebuild.

The priority order is accuracy and precision first, training/system efficiency
second, simplicity third, and extension capability fourth. The new OpenSpec
baseline should therefore protect exact token/image/loss/artifact semantics
before adding convenience abstractions or future rollout surfaces.

## What Changes

- Establish a fresh OpenSpec baseline for CoordExp-swift, with the previous
  `openspec/` tree archived as reference material under
  `reference/legacy_openspec_2026-06-29/`.
- **BREAKING**: Archived legacy OpenSpec specs and archived legacy changes are
  no longer current behavior authority for this worktree's rebuilt `src/`
  architecture.
- Define the V1 training pipeline around Qwen3-VL single-image supervised
  training with Transformers/PyTorch as substrate and CoordExp-owned contracts
  for data, templates, encoding, packing, supervision, losses, optimizer
  grouping, runtime, artifacts, metrics, checkpointing, and forward eval.
- Preserve the end-to-end execution flow:
  Raw Dataset -> Dataset Processing -> Chat Template -> Packing ->
  Tokenization -> Visual Processing -> Model Construction -> Forward Pass ->
  Loss Computation -> Backward Pass -> Distributed Training -> Checkpoint
  Management -> Inference/Eval.
- Require dLoRA setup definition/source study before any `adapter.type: dlora`
  config validates, then use dLoRA as the first adapter-enabled smoke path.
- Require a special-token embedding mechanism study before implementation,
  comparing custom Qwen wrappers against PEFT `TrainableTokens` and LoRA
  `trainable_token_indices`, while preserving compact checkpoint payloads and
  tied input/output behavior.
- Define the first vertical smoke as a real five-planned-step training path
  with real `packing.global_max_length`, sample-limited data, scheduled
  `eval.forward`, metrics, checkpoint metadata, and
  `checkpoints/checkpoint-final.json`.
- Keep rollout, hidden-state losses, persistent caches, richer inference/eval,
  DeepSpeed production support, and old production coordinate-soft-CE objective
  parity as future capabilities unless explicitly promoted by later changes.

## Capabilities

### New Capabilities

- `coordexp-swift-config-runtime`: strict resolved config, run identity,
  inheritance, cadence, runtime backend status, and planned-step schedule
  contracts.
- `coordexp-swift-data-template-encoding`: raw example shape, JSONL loading,
  object ordering, chat-template rendering, Qwen processor/tokenizer encoding,
  no-resize image policy, and token/span alignment contracts.
- `coordexp-swift-packing-forward`: no-padding packed-sequence construction,
  packed supervision remapping, Qwen MRoPE position ids, visual payload
  handling, and FlashAttention varlen forward-boundary contracts.
- `coordexp-swift-supervision-losses`: `TokenAtom` / `TokenSpan` /
  `TokenSequence`, `LossContext`, protected `BaseTokenCE`,
  `TokenTypeGateLoss`, planned-step `LossNormalizers`, finite gates, metrics,
  and future auxiliary-loss seams.
- `coordexp-swift-adapters-embeddings-optim`: dLoRA/LoRA source-study gates,
  adapter target discovery, special-token embedding deltas, explicit LR/WD
  groups, optimizer construction, and trainable-parameter receipts.
- `coordexp-swift-training-artifacts`: `SupervisedTrainer`, `TrainRuntime`,
  backward/optimizer/scheduler policy, distributed rank safety, checkpoint
  writer contracts, metric streams, manifest layout, and forward-eval artifacts.
- `coordexp-swift-vertical-smoke`: the permanent tiny Qwen3-VL single-image
  smoke fixture and the first end-to-end train/eval/checkpoint acceptance
  contract.

### Modified Capabilities

- None. This is a clean rebuild baseline; previous OpenSpec capabilities are
  archived as historical reference and are not modified in place.

## Impact

- Affected source roots: `src/`, `tests/fixtures/smoke/`,
  `configs/`, `openspec/`, `docs/architecture/proposals/`, and
  `reference/`.
- Affected external dependencies: Transformers, PyTorch, PEFT, Accelerate,
  DeepSpeed, flash-attn, and MS-Swift as a reference-only source.
- Implementation remains blocked until the relevant module/card approvals and
  source-study gates are satisfied.
- The archived legacy OpenSpec tree remains available for comparison, but new
  specs must be authored from `DECISIONS.md` and `BLUEPRINT.md`, not copied from
  old Stage-1/Stage-2 contracts.
