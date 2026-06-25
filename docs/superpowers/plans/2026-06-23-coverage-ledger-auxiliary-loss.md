# Coverage Ledger Auxiliary Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a strict Stage-1 Qwen3-VL coverage-ledger auxiliary loss that trains hidden states to encode emitted-object coverage and binds each current row to its visual region, with a paired 128-sample closed-template hard-SFT smoke baseline before any production training.

**Architecture:** Keep the active teacher-forcing CE path intact, add a `CoverageLedgerSidecar` under `TrainingSidecars.supervision.payloads`, register a trainable `CoverageLedgerHead` on the actual model before optimizer construction, capture final Qwen hidden states and post-merger image embeddings from the same forward pass, compute coverage BCE plus one-vs-all row-object binding loss in a bridge-owned auxiliary path, and publish explicit metric events and smoke/debug artifacts.

**Tech Stack:** Python, PyTorch, pytest, ms-swift trainer integration, Hugging Face Qwen3-VL model conventions, CoordExp Stage-1 research teacher-forcing configs, CoordExp metric events, PIL/Matplotlib-compatible overlay artifact generation.

---

## Approval Gate

This file is an implementation roadmap. Do not edit source code, configs, tests, or launch training from this plan until the user explicitly approves implementation.

Implementation worktree:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git status --short --branch
codegraph init -i
```

Expected branch:

```text
codex/ledger-auxiliary-loss
```

## Source Of Truth

- Design spec: `docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md`
- Design audit: `docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design-audit.md`
- Research idea packet: `research/ideas/ledger-auxiliary-loss/`
- Stage-1 objective docs: `docs/training/STAGE1_OBJECTIVE.md`
- Qwen upstream boundary: `docs/standards/UPSTREAM.md`, `docs/standards/upstream/QWEN_VL.md`
- Active config schema: `src/config/schema.py`
- Teacher-forcing trainer mixin: `src/trainers/metrics/teacher_forcing.py`
- Loss bridge: `src/training/bridge/loss_bridge.py`
- Sidecar container: `src/training/sidecars.py`
- Metric events: `src/metrics/events.py`, `src/training/objectives/types.py`
- Optimizer split: `src/optim/token_embeddings_adapter_optimizer.py`

## Non-Negotiable Contracts

- V0 template is exactly `detection_template.id: compact_object_box_closed`.
- Both `<|object_ref_end|>` and `<|box_end|>` are mandatory.
- `compact_full` remains the old chat-template/schema label and is invalid as a Stage-1 `detection_template.id`.
- No new special tokens and no global production default change.
- V0 comparison is closed-wrapper hard-SFT baseline versus closed-wrapper hard-SFT plus coverage ledger.
- `coverage_ledger.enabled: true` rejects packing, static packing, padding-free packing, multi-image samples, video samples, zero-object samples, missing image metadata, malformed boxes, and missing sidecars.
- V0 uses per-device batch size 1, one long unpadded sequence per forward pass.
- The bridge does not own trainable modules. `CoverageLedgerHead` is attached to the trainable model before optimizer creation.
- The Qwen visual tower is not recomputed for the ledger path. Capture happens from the helpful same-forward hook/wrapper around `get_image_features`.
- Visual-region mapping uses projected post-merger `image_embeds`, processed image dimensions, and a minimal half-open token-cell rectangle enclosing object pixels.
- Row-object binding marks the current row object positive and all other annotated objects negative.
- Coverage and row-object binding AUC/accuracy are monitoring metrics from the same forward pass, not rollout-quality claims.
- Smoke artifacts must include all-128 alignment debug records and 16 overlay images.
- OpenSpec is deferred until the experimental loss/config/artifact contract is promoted to stable behavior.

## Task 1: Config Contract And Runtime Gating

**Files:**
- Modify: `src/config/schema.py`
- Modify: `tests/test_teacher_forcing_config_contract.py`
- Modify: `tests/test_training_config_strict_unknown_keys.py`

- [ ] Add failing config tests for the new term under `objective.terms.coverage_ledger`.

  Required test cases:

  - `profile: hard_sft` with `coverage_ledger.enabled: true` and `detection_template.id: compact_object_box_closed` loads successfully.
  - `coverage_weight`, `region_anchor_weight`, `ledger_projection_dim`, `temperature`, `normalize_eps`, and `pos_weight` reject invalid types and invalid ranges.
  - Unknown nested keys under `objective.terms.coverage_ledger` fail fast.
  - `coverage_ledger.enabled: true` rejects `detection_template.id: compact`.
  - `coverage_ledger.enabled: true` rejects `detection_template.id: compact_box_closed`.
  - `coverage_ledger.enabled: true` rejects `detection_template.id: compact_object_closed`.
  - `coverage_ledger.enabled: true` rejects `detection_template.id: compact_full`.
  - `coverage_ledger.enabled: true` rejects `packing: true`, `static_packing.enabled: true`, and `padding_free_packed: true`.

  Run:

  ```bash
  python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_training_config_strict_unknown_keys.py -q
  ```

  Expected before implementation: the new `coverage_ledger` term is rejected as an unknown `objective.terms` key.

- [ ] Implement `TeacherForcingCoverageLedgerConfig` in `src/config/schema.py`.

  Required fields and defaults:

  ```python
  enabled: bool = False
  coverage_weight: float = 0.1
  region_anchor_weight: float = 0.1
  ledger_projection_dim: int = 256
  temperature: float = 0.2
  normalize_eps: float = 1.0e-6
  pos_weight: float = 1.0
  log_auc: bool = True
  log_accuracy: bool = True
  overlay_sample_count: int = 16
  smoke_sample_count: int = 128
  smoke_sample_seed: int = 20260623
  ```

  Validation rules:

  - `enabled` is a plain `bool`.
  - `coverage_weight` and `region_anchor_weight` are finite floats greater than or equal to `0`.
  - `ledger_projection_dim` is a positive integer.
  - `temperature` is finite and greater than or equal to `0.05`.
  - `normalize_eps` is finite and greater than or equal to `1e-8`.
  - `pos_weight` is finite and greater than `0`.
  - `overlay_sample_count` is exactly `16` for the V0 smoke recipe.
  - `smoke_sample_count` is exactly `128` for the V0 smoke recipe.
  - `smoke_sample_seed` is exactly `20260623` for the V0 smoke recipe.

- [ ] Add `coverage_ledger` to `TeacherForcingModulesConfig` allowed `objective.terms` keys.

  Do not revive retired `objective.modules` authoring. Keep all new authoring under `objective.terms`.

- [ ] Keep hard-SFT gating precise.

  `TeacherForcingObjectiveConfig.__post_init__` currently rejects valid-set modules under `profile: hard_sft`. The ledger is bridge-local auxiliary supervision, not a valid-set semantic module, so hard-SFT must allow `terms.coverage_ledger.enabled: true`.

- [ ] Add cross-field validation in the runtime/config support function that already sees detection template and packing settings.

  Required behavior when `coverage_ledger.enabled` is true:

  - Resolve `detection_template.id` through `resolve_detection_template_contract`.
  - Require `contract.template_id == "compact_object_box_closed"`.
  - Require `contract.include_object_ref_end is True`.
  - Require `contract.include_box_end is True`.
  - Reject `compact_full` with an error message that names old chat-template/schema usage.
  - Require all packing modes disabled.
  - Require `per_device_train_batch_size == 1` or the repo's resolved equivalent when that field is available at validation time.

  Run:

  ```bash
  python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_training_config_strict_unknown_keys.py -q
  ```

  Expected after implementation: all config contract tests pass and existing compact-template tests remain unchanged.

## Task 2: Closed-Template Smoke Config Pair

**Files:**
- Add: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml`
- Add: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`
- Add or modify: `tests/test_coverage_ledger_smoke_configs.py`

- [ ] Add a config-load test for the paired smoke configs.

  The test must assert both configs resolve these shared fields:

  ```yaml
  pipeline:
    id: stage1_research_teacher_forcing
  detection_template:
    id: compact_object_box_closed
  objective:
    id: research_teacher_forcing
    profile: hard_sft
  ```

  The test must assert both configs disable packing:

  ```yaml
  packing: false
  static_packing:
    enabled: false
  padding_free_packed: false
  ```

  The test must assert the adapter structural rows include all four wrapper tokens:

  ```text
  <|object_ref_start|>
  <|object_ref_end|>
  <|box_start|>
  <|box_end|>
  ```

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_smoke_configs.py -q
  ```

  Expected before implementation: the two config paths do not exist.

- [ ] Create the baseline config.

  Required differences from the ledger config:

  - `objective.terms.coverage_ledger.enabled: false`
  - run identity and output root identify the baseline

  Required shared runtime:

  - training sample count `128`
  - seed `20260623`
  - `max_steps: 256`
  - `per_device_train_batch_size: 1`
  - `gradient_accumulation_steps: 1`
  - `do_resize: false` in the processor/runtime surface that already owns this contract

- [ ] Create the ledger config.

  Required ledger fields:

  ```yaml
  objective:
    terms:
      coverage_ledger:
        enabled: true
        coverage_weight: 0.1
        region_anchor_weight: 0.1
        ledger_projection_dim: 256
        temperature: 0.2
        normalize_eps: 1.0e-6
        pos_weight: 1.0
        log_auc: true
        log_accuracy: true
        overlay_sample_count: 16
        smoke_sample_count: 128
        smoke_sample_seed: 20260623
  ```

- [ ] Add a resolved-config diff test.

  The diff allowlist is:

  - ledger enablement and ledger weights
  - run name
  - output directory
  - artifact root
  - smoke/preflight debug artifact settings

  The diff test must fail if template, object field order, source dataset, model/checkpoint, processor resize policy, packing policy, batch size, grad accumulation, or max steps differ.

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_smoke_configs.py -q
  ```

## Task 3: Coverage Ledger Sidecar Types And Extraction

**Files:**
- Add: `src/training/coverage_ledger/__init__.py`
- Add: `src/training/coverage_ledger/sidecars.py`
- Add: `src/training/coverage_ledger/sidecar_builder.py`
- Modify: `src/training/sidecars.py` only if export plumbing is needed
- Modify the Stage-1 data/collator path that currently emits `teacher_forcing_target_ir`
- Modify: `tests/test_model_input_bundle_contract.py`
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`
- Add: `tests/test_coverage_ledger_sidecar_builder.py`

- [ ] Add immutable dataclasses in `src/training/coverage_ledger/sidecars.py`.

  Required shape:

  ```python
  @dataclass(frozen=True, slots=True)
  class CoverageLedgerObjectEntry:
      object_instance_id: str
      source_object_index: int
      emitted_order_index: int
      image_index: int
      bbox_norm1000_xyxy: tuple[int, int, int, int]
      box_start_position: int
      coord_label_positions: tuple[int, int, int, int]
      object_ref_end_position: int
      box_end_position: int

  @dataclass(frozen=True, slots=True)
  class CoverageLedgerSidecar:
      sample_id: str
      prompt_end_position: int
      object_entries: tuple of CoverageLedgerObjectEntry values
      image_grid_thw: tuple[int, int, int]
      processed_width: int
      processed_height: int
      image_identity: str
  ```

  Validation requirements:

  - `object_entries` is non-empty.
  - `sample_id` and `image_identity` are non-empty strings.
  - `prompt_end_position`, positions, dimensions, and grid values are non-negative integers, with dimensions/grid strictly positive.
  - every `image_index` is `0`.
  - every bbox is finite integer norm-1000 `xyxy` with `0 <= x1 < x2 <= 1000` and `0 <= y1 < y2 <= 1000`.
  - emitted order indices are exactly `0..N-1`.
  - object instance ids are unique inside the sidecar.

- [ ] Add sidecar extraction tests from structured tokenized metadata.

  The builder must use tokenized/rendered span labels, not raw token string search.

  Required extraction rules:

  - `prompt_end_position = first_object_entry.entry_span.start - 1`
  - `box_start_position = object_entry.bbox_start_span.start`
  - `coord_label_positions = tuple(span.start for span in object_entry.coord_spans)`
  - `object_ref_end_position` comes from the unique `control_span.label == "object_ref_end"`
  - `box_end_position` comes from the unique `control_span.label == "box_end"`
  - each required control span must cover exactly one token after tokenization

  Required rejection tests:

  - plain `compact` lacks `box_end` and fails
  - `compact_box_closed` lacks `object_ref_end` and fails
  - `compact_object_closed` lacks `box_end` and fails
  - duplicate `CoverageLedgerSidecar` payloads fail
  - missing payload when ledger is enabled fails
  - disagreement between explicit `training_sidecars` and raw-batch `training_sidecars` fails
  - packed batch sidecar offset rewriting is not attempted and fails

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_sidecar_builder.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_model_input_bundle_contract.py -q
  ```

- [ ] Route sidecars through `TrainingSidecars.supervision.payloads`.

  The collator path should append one `CoverageLedgerSidecar` per sample when `coverage_ledger.enabled` is true. It must not add the payload to model inputs, metadata forwarded to Qwen, or generic dict keys that bypass `TrainingSidecars`.

## Task 4: Trainable Coverage Ledger Head Ownership

**Files:**
- Add: `src/training/coverage_ledger/head.py`
- Modify: `src/sft.py`
- Modify: `src/optim/token_embeddings_adapter_optimizer.py`
- Modify: `tests/tokens/test_token_embeddings_adapter_optimizer.py`
- Add: `tests/test_coverage_ledger_head_install.py`

- [ ] Add tests proving the head is real trainable model state.

  Required assertions:

  - enabled config installs `coverage_ledger_head` on the actual trainable model after `prepare_model` and before trainer/optimizer construction.
  - disabled config does not install the head.
  - `dict(model.named_parameters())` contains `coverage_ledger_head.state_projection.weight`, `coverage_ledger_head.region_anchor_state_projection.weight`, and `coverage_ledger_head.object_projection.weight`.
  - all ledger head parameters have `requires_grad=True`.
  - a backward plus optimizer step changes at least one ledger head parameter in a tiny synthetic loss.
  - `state_dict()` contains the ledger head weights.
  - loading a saved `state_dict()` restores the ledger head weights.

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_head_install.py tests/tokens/test_token_embeddings_adapter_optimizer.py -q
  ```

  Expected before implementation: `coverage_ledger_head` is absent.

- [ ] Implement `CoverageLedgerHead`.

  Required module fields:

  ```python
  state_projection: nn.Linear
  region_anchor_state_projection: nn.Linear
  object_projection: nn.Linear
  ```

  Required constructor inputs:

  ```python
  hidden_size: int
  visual_dim: int
  ledger_projection_dim: int
  normalize_eps: float
  ```

  The head owns only projection weights. Temperature, loss weights, and `pos_weight` stay in the objective config, not in module parameters.

- [ ] Add an install helper.

  Required behavior:

  - install exactly once under attribute name `coverage_ledger_head`
  - raise if an existing attribute is present with incompatible type or dimensions
  - infer `hidden_size` from the language model config used for final hidden states
  - infer `visual_dim` from captured post-merger image embeddings in a dry synthetic test when model config lacks a direct field
  - move the head to the model parameter device
  - use a floating dtype compatible with model trainable parameters

- [ ] Wire install in `src/sft.py`.

  Install immediately after:

  ```python
  sft.model = sft.prepare_model(
      train_args, sft.model, template=sft.template, train_dataset=dataset
  )
  ```

  and after token-embedding hook reattachment. It must happen before the `instantiate_trainer` call and before optimizer creation.

- [ ] Update `create_multimodal_token_embeddings_adapter_optimizer`.

  Required behavior:

  - include `coverage_ledger_head` params exactly once
  - use main training LR and weight decay for ledger head params in V0
  - keep them even when `model_meta.model_arch` splits vision/aligner/language prefixes and would otherwise miss the head
  - preserve existing token-embedding adapter parameter grouping

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_head_install.py tests/tokens/test_token_embeddings_adapter_optimizer.py -q
  ```

## Task 5: Same-Forward Qwen Capture Helper

**Files:**
- Add: `src/training/coverage_ledger/qwen_capture.py`
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`
- Add: `tests/test_coverage_ledger_qwen_capture.py`

- [ ] Add fake-Qwen contract tests before implementation.

  Required fake model shape:

  - outer conditional-generation module exposes `.model` and `.lm_head`
  - lower-level `.model.get_image_features` returns post-merger image embeddings
  - lower-level model forward returns final hidden states
  - `.lm_head(hidden_states)` returns logits

  Required assertions:

  - capture returns full logits, final hidden states, and image embeds
  - logits from capture equal logits from the model's normal forward within tolerance on the fake model
  - `get_image_features` is called exactly once
  - sidecar-only keys are not forwarded to Qwen
  - `logits_to_keep` remains rejected
  - logits time dimension still matches `input_ids`

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_qwen_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
  ```

- [ ] Implement `CoverageLedgerForwardCapture`.

  Required behavior:

  - use the existing `prepare_forward_inputs` convention from `src/trainers/teacher_forcing/forwards.py`
  - unwrap the active trainable model without dropping LoRA/adapters/hooks
  - temporarily wrap the lower-level `model.model.get_image_features`
  - call the lower-level Qwen model exactly once for hidden states and captured `image_embeds`
  - apply the same `lm_head` path to get full logits
  - preserve dtype/device/autocast behavior
  - restore the original `get_image_features` method in a `finally` block
  - hard fail if the model is not a Qwen3-VL compatible conditional-generation object
  - hard fail if image embeds are missing, empty, or count-mismatched with image-token slots/grid metadata

- [ ] Add a tiny real-model parity probe if the local test environment can load the tiny configured Qwen checkpoint without a long download.

  Command:

  ```bash
  python -m pytest tests/test_coverage_ledger_qwen_capture.py -q -m "not slow"
  ```

  If the real-model probe is skipped because the checkpoint is unavailable, record the skip reason in the final implementation summary.

## Task 6: Visual Token Region Mapping

**Files:**
- Add: `src/training/coverage_ledger/visual_regions.py`
- Add: `tests/test_coverage_ledger_visual_regions.py`

- [ ] Add synthetic mapping tests.

  Required test cases:

  - full-image bbox maps to every post-merger visual token cell
  - centered bbox maps to the minimal enclosing half-open token-cell rectangle
  - tiny valid bbox maps to at least one cell after clamping
  - bbox on the right/bottom edge clamps within grid bounds
  - degenerate bbox hard fails
  - processed dimensions inconsistent with image grid and patch size hard fail
  - multi-frame or multi-image grid hard fails in V0

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_visual_regions.py -q
  ```

- [ ] Implement minimal enclosing post-merger token-cell mapping.

  Required formula:

  ```text
  x1_px = bbox_x1 / 1000 * processed_width
  y1_px = bbox_y1 / 1000 * processed_height
  x2_px = bbox_x2 / 1000 * processed_width
  y2_px = bbox_y2 / 1000 * processed_height

  col_start = floor(x1_px / cell_width)
  row_start = floor(y1_px / cell_height)
  col_end = ceil(x2_px / cell_width)
  row_end = ceil(y2_px / cell_height)
  ```

  Required bounds:

  - output interval is half-open: `[row_start, row_end) x [col_start, col_end)`
  - valid tiny boxes clamp to one cell
  - no nearest-top-k fallback is used
  - returned indices are flattened in row-major order and index into projected post-merger `image_embeds`

- [ ] Pool object visual embeddings.

  For each object, gather the mapped token cells and average their captured post-merger embeddings. Detach visual embeddings before projection for V0 so ledger gradients do not update the vision tower or aligner through the auxiliary path.

## Task 7: Ledger Target Builder And Loss Math

**Files:**
- Add: `src/training/coverage_ledger/loss.py`
- Add: `tests/test_coverage_ledger_loss.py`
- Modify: `tests/test_objective_precision_policy.py` if a precision-policy regression test is needed

- [ ] Add synthetic loss tests before implementation.

  Required coverage-state target tests with `N=3` objects:

  - prompt-end target is `[0, 0, 0]`
  - row 0 `<box_end>` target is `[1, 0, 0]`
  - row 1 `<box_end>` target is `[1, 1, 0]`
  - row 2 `<box_end>` target is `[1, 1, 1]`

  Required region-anchor tests:

  - row 0 `<box_start>` uses only object 0 as positive
  - row 1 `<box_start>` uses only object 1 as positive
  - row 2 `<box_start>` uses only object 2 as positive
  - non-current objects are masked and do not contribute negative terms

  Required numerical tests:

  - lower temperature increases logit magnitude before BCE
  - `normalize_eps` prevents division by zero for zero vectors
  - non-finite logits or losses raise `FloatingPointError`
  - zero `coverage_weight` disables coverage contribution but still emits valid diagnostic counts
  - zero `region_anchor_weight` disables region-anchor contribution

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_loss.py tests/test_objective_precision_policy.py -q
  ```

- [ ] Implement target construction.

  Required state positions:

  - `prompt_end_position`
  - every `object_entry.box_end_position`

  Required anchor positions:

  - every `object_entry.box_start_position`

  Required causal hidden-state gather:

  - gather final hidden states at these exact token positions
  - do not shift positions through `LabelLogitRowMap`; this objective reads hidden state after observed prefix tokens, not logits predicting those tokens

- [ ] Implement coverage BCE.

  Required computation:

  - project hidden states with `CoverageLedgerHead.state_projection`
  - project detached pooled visual object embeddings with `CoverageLedgerHead.object_projection`
  - L2-normalize both projections using `normalize_eps`
  - compute pairwise logits as dot product divided by `temperature`
  - flatten valid state-object pairs
  - apply `binary_cross_entropy_with_logits` with `pos_weight`
  - compute in `torch.float32` using `ObjectivePrecisionPolicy`

- [ ] Implement one-vs-all row-object binding.

  Required computation:

  - project `<box_start>` hidden states with `CoverageLedgerHead.region_anchor_state_projection`
  - project detached pooled visual object embeddings with the shared `CoverageLedgerHead.object_projection`
  - L2-normalize both projections using `normalize_eps`
  - for row `k`, score against every annotated object
  - set object `k` target to 1 and all other objects to 0
  - compute BCE-with-logits over the full row-object matrix

- [ ] Return a typed result.

  Required result fields:

  ```python
  total_loss
  coverage_loss
  region_anchor_loss
  weighted_loss
  metric_events
  debug_rows
  ```

  `weighted_loss = coverage_weight * coverage_loss + region_anchor_weight * region_anchor_loss`.

## Task 8: Ledger Metrics And Reducer Semantics

**Files:**
- Add: `src/training/coverage_ledger/metrics.py`
- Add: `tests/test_coverage_ledger_metrics.py`
- Modify: `docs/training/METRICS.md` only after the producer is implemented

- [ ] Add metric producer tests.

  Required flat keys:

  ```text
	  teacher_forcing/loss/coverage_ledger_auxiliary/contribution
	  teacher_forcing/ledger/coverage_ledger_auxiliary_pair_normalized
	  teacher_forcing/ledger/coverage_bce
	  teacher_forcing/ledger/row_object_binding_bce
	  teacher_forcing/ledger/coverage_auc
	  teacher_forcing/ledger/coverage_accuracy
	  teacher_forcing/ledger/row_object_binding_auc
	  teacher_forcing/ledger/row_object_binding_accuracy
	  teacher_forcing/ledger/coverage_state_count
	  teacher_forcing/ledger/coverage_pair_count
	  teacher_forcing/ledger/object_count
	  teacher_forcing/ledger/row_object_binding_pair_count
	  ```

  Required reducer tests:

  - BCE producer calls `weighted_mean_event` with `key`, `value=loss_sum / valid_count`, `weight=valid_count`, and the required metric metadata
  - weighted loss producer emits `CoverageLedgerLossResult.weighted_loss` as the exact `last` scalar added to runner loss
  - pair-normalized weighted loss reconstruction is diagnostic-only under `teacher_forcing/ledger/coverage_ledger_auxiliary_pair_normalized`
  - AUC is omitted when a batch has only one class
  - AUC denominator is comparable positive-negative pairs `n_pos * n_neg`
  - AUC tie credit is `0.5`
  - accuracy is computed from `sigmoid(logit) >= 0.5`
  - zero denominator metrics are omitted, not logged as `0.0`

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_metrics.py tests/test_metric_events.py -q
  ```

- [ ] Implement exact coverage AUC from one forward pass.

  The implementation may compute pairwise rank AUC directly because V0 uses one unpadded sample per forward and object counts are small. Use vectorized PyTorch or a simple tensor sort, but preserve tie credit `0.5`.

- [ ] Mark diagnostic versus objective metrics correctly.

  Only `teacher_forcing/loss/coverage_ledger_auxiliary/contribution` is objective-relevant and it reports the exact scalar added to training loss. Coverage BCE, row-object binding BCE, pair-normalized auxiliary loss, AUC, accuracy, counts, and debug gauges are diagnostic metrics.

## Task 9: Bridge And Trainer Integration

**Files:**
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/training/observability/service.py` only if existing flattening cannot be reused
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`
- Modify: `tests/test_training_runtime_sft_integration.py`
- Add: `tests/test_coverage_ledger_bridge_integration.py`

- [ ] Add integration tests for disabled behavior.

  Required assertions:

  - with `coverage_ledger.enabled: false`, `TrainerLossBridge` calls the model through the existing path
  - disabled ledger does not require `CoverageLedgerSidecar`
  - disabled ledger does not require `coverage_ledger_head`
  - existing teacher-forcing CE tests still pass

- [ ] Add integration tests for enabled behavior.

  Required assertions:

  - exactly one `CoverageLedgerSidecar` is required
  - exactly one `coverage_ledger_head` is required
  - `CoverageLedgerForwardCapture` path is used
  - full logits validation still runs
  - base teacher-forcing CE loss and ledger weighted loss are summed into returned loss
  - metric events include both base CE events and ledger events
  - metric events are flattened and logged through `SwiftMetricReporter`
  - sidecars are never forwarded to Qwen

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_bridge_integration.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_training_runtime_sft_integration.py -q
  ```

- [ ] Extend `TrainerLossBridgeResult`.

  Add a defaulted field:

  ```python
  metric_events: tuple of MetricEvent values, defaulting to an empty tuple
  ```

  Keep `objective_result` as the base typed runner result for compatibility. Set `metric_events` to `objective_result.metric_events + coverage_ledger_result.metric_events` when ledger is enabled.

- [ ] Update `TeacherForcingObjectiveMixin.compute_loss`.

  Required changes:

  - resolve `objective_cfg.terms.coverage_ledger`
  - pass ledger config to the bridge as part of a typed or mapping bridge setting
  - pass `training_sidecars` through if already carried by raw inputs
  - log `flatten_metric_events(result.metric_events)` through `SwiftMetricReporter`
  - continue returning `(result.loss, result.outputs)` when `return_outputs` is true

- [ ] Keep the objective-runner registry unchanged for ledger.

  `coverage_ledger` is sidecar-driven, so do not add it to `SUPPORTED_OBJECTIVES_BY_DISTRIBUTION_KIND` as if it supervised `TeacherForcingTargetDistribution` spans. The existing `teacher_forcing` objective remains responsible for hard-SFT CE.

## Task 10: Preflight, Sample Manifest, And Overlay Artifacts

**Files:**
- Add: `scripts/training/coverage_ledger_preflight.py`
- Add: `src/training/coverage_ledger/preflight.py`
- Add: `src/training/coverage_ledger/artifacts.py`
- Add: `tests/test_coverage_ledger_preflight_artifacts.py`
- Modify: `docs/ARTIFACTS.md` after artifact writer exists

- [ ] Add artifact writer tests.

  Required output tree under `temp/coverage_ledger_preflight_smoke/ledger/`:

  ```text
  selected_samples.json
  alignment_debug.jsonl
  overlays/
  overlays/index.json
  ```

  Required `selected_samples.json` fields:

  ```text
  schema_version
  source_jsonl_path
  source_jsonl_repo_path
  source_jsonl_resolved_path
  source_jsonl_sha256
  dataset_id
  split
  selected_row_indices
  selected_sample_ids
  selection_seed
  selection_algorithm
  template_id
  object_field_order
  tokenizer_id
  model_id
  processor_do_resize
  image_grid_metadata_version
  samples[]
  ```

  The writer must fail fast instead of merging output when `ledger/` or
  `ledger/overlays/` already contains stale files.

  Required per-sample fields:

  ```text
  row_index
  sample_id
  object_count
  image_identity
  processed_width
  processed_height
  image_grid_thw
  ```

  Required `alignment_debug.jsonl` coverage:

  - exactly 128 lines for the V0 preflight
  - each line records prompt end, each object row's `<object_ref_end>`, `<box_start>`, coord positions, `<box_end>`, bbox, mapped visual cells, and failure status
  - failure status is `"ok"` for all 128 samples in a passing preflight

  Required overlays:

  - exactly 16 image files
  - overlay includes original image, GT bbox, mapped visual-token rectangle/cells, sample id, object index, and template id
  - `overlays/index.json` points to all 16 files and their source sample/object ids

  Run:

  ```bash
  python -m pytest tests/test_coverage_ledger_preflight_artifacts.py -q
  ```

- [ ] Implement the preflight script.

  Required command:

  ```bash
  python scripts/training/coverage_ledger_preflight.py \
    --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
    --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
    --output-root temp/coverage_ledger_preflight_smoke
  ```

  Required behavior:

  - load and resolve both configs
  - enforce the resolved-config diff allowlist from Task 2
  - select the 128 training samples using seed `20260623`
  - build detection rendering, tokenization, teacher-forcing IR, and coverage ledger sidecar for all selected samples
  - build visual-region mappings
  - write all artifacts
  - exit non-zero on the first strict alignment failure
  - never launch training

## Task 11: Tiny Smoke Training And Documentation

**Files:**
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/ARTIFACTS.md`
- Modify: `research/ideas/ledger-auxiliary-loss/index.md`
- Add or modify: `tests/test_artifact_contract_docs.py`

- [ ] Run the docs-only artifact consistency tests after docs edits.

  Command:

  ```bash
  python -m pytest tests/test_artifact_contract_docs.py -q
  ```

- [ ] Update docs only after code emits the behavior.

  Required doc updates:

  - `docs/training/STAGE1_OBJECTIVE.md` records the experimental closed-wrapper hard-SFT ledger smoke route
  - `docs/training/METRICS.md` records ledger metric keys and reducer semantics
  - `docs/ARTIFACTS.md` records `ledger/selected_samples.json`, `ledger/alignment_debug.jsonl`, and `ledger/overlays/`
  - research index links the new implementation plan and marks OpenSpec as deferred

- [ ] Run preflight before training.

  Command:

  ```bash
  python scripts/training/coverage_ledger_preflight.py \
    --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
    --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
    --output-root temp/coverage_ledger_preflight_smoke
  ```

- [ ] Launch tiny smoke training only after user confirms the runtime cost.

  The first launch must use the two checked-in smoke configs from Task 2 and write to separate output roots. The implementation summary must label the result as `tiny/smoke`, not validation.

  Required post-run checks:

  - both runs contain `resolved_config.json`
  - both runs contain `effective_runtime.json`
  - both runs contain `experiment_manifest.json`
  - both runs contain `run_metadata.json`
  - both runs contain `train_data_provenance.json`
  - ledger run contains `ledger/selected_samples.json`
  - ledger run contains all-128 `ledger/alignment_debug.jsonl`
  - ledger run contains 16 overlays
  - ledger metrics appear in train logs
  - baseline and ledger config diff remains allowlisted

## Task 12: Review Convergence And Production Gate

**Files:**
- Modify: `docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md` only if implementation discovers a spec mismatch
- Add: an implementation review/audit note under `docs/superpowers/specs/` if review-convergence produces material findings

- [ ] Launch review-convergence after unit tests and preflight pass.

  Required independent review lanes:

  - Qwen same-forward capture correctness and upstream-boundary compliance
  - sidecar/position and visual-region geometry correctness
  - optimizer/checkpoint ownership correctness
  - metric reducer and artifact reproducibility correctness
  - smoke-run interpretation and no-overclaiming correctness

- [ ] Fix every P0/P1 finding before smoke training interpretation.

  P2 findings may remain only if they are documented with an explicit follow-up and do not affect the V0 mechanism claim.

- [ ] Keep production training gated.

  Production training is not approved by implementation completion. It requires:

  - strict preflight pass on the exact production training input
  - smoke baseline and smoke ledger artifacts complete
  - evidence that ledger metrics are observable
  - evidence that overlays match object pixels and mapped visual token cells
  - explicit user approval for production runtime cost

## Final Verification Bundle

Run this unit implementation bundle before claiming the code/config/docs portion
is complete. Passing this pytest bundle alone does not claim preflight success,
smoke success, production readiness, or usable local launch state.

```bash
python -m pytest \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_coverage_ledger_smoke_configs.py \
  tests/test_coverage_ledger_sidecar_builder.py \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_model_input_bundle_contract.py \
  tests/test_coverage_ledger_head_install.py \
  tests/tokens/test_token_embeddings_adapter_optimizer.py \
  tests/test_coverage_ledger_qwen_capture.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  tests/test_coverage_ledger_visual_regions.py \
  tests/test_coverage_ledger_loss.py \
  tests/test_coverage_ledger_metrics.py \
  tests/test_metric_events.py \
  tests/test_coverage_ledger_bridge_integration.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_coverage_ledger_preflight_artifacts.py \
  tests/test_artifact_contract_docs.py \
  -q
```

Then run the preflight only when the local public-data/model assets exist:

```bash
python scripts/training/coverage_ledger_preflight.py \
  --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  --output-root temp/coverage_ledger_preflight_smoke
```

Final implementation summary must report:

- changed files
- verification commands and outcomes
- preflight artifact root
- whether preflight was run or skipped because local assets were unavailable
- whether real-Qwen capture parity was run or skipped
- whether smoke training was run or still awaiting approval
- residual risks, especially small-object visual-token alignment and smoke-only evidence scope
