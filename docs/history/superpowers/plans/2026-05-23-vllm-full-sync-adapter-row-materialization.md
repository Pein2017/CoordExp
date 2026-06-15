# vLLM Full-Sync Adapter Row Materialization Implementation Plan

> **Execution status:** Implemented after explicit user approval in the long-goal takeover. The checklist below is retained as the implementation/audit trail.

**Goal:** Preserve frozen-base, adapter-only Stage-2 training and checkpointing while native vLLM full-sync receives an ordinary-weight snapshot that includes CoordExp token-row adapter effects.

**Architecture:** Add a small learner-side materialization helper that runs after current PEFT state-dict canonicalization and before native vLLM `load_weights()` or ms-swift server bucket update. The helper clones and patches only affected ordinary row tensors, strips learner-only key families, and fails fast when active token-row state cannot be represented safely.

**Important boundary:** Current PEFT full-sync may temporarily call `merge_adapter()` and later `unmerge_adapter()`. This change requires no lasting live-base mutation and adapter-only save semantics; it does not require a zero-temporary-mutation implementation in Phase 1.

**Tech Stack:** Python 3.12, PyTorch, PEFT/ms-swift, native vLLM Qwen3VL loader, CoordExp Stage-2 rollout runtime, `conda run -n ms python -m pytest`.

---

## File Structure

- Create: `src/trainers/rollout_runtime/vllm_sync_materialization.py`
  - Owns active coord/token-row adapter discovery, row patching, forbidden-key validation, tied-head safety checks, and diagnostics.
- Modify: `src/trainers/rollout_runtime/vllm_server.py`
  - Calls the helper inside `sync_vllm_server_full_weights` before `owner._vllm_server_update_state_dict(client, state_dict)`.
- Modify: `src/trainers/rollout_runtime/vllm_engine.py`
  - Calls the helper inside `sync_vllm_full_weights_if_needed` before `engine.inner_model.load_weights(state_dict.items())`.
- Create: `tests/tokens/test_vllm_sync_materialization.py`
  - Unit tests for tied/untied parity, row values, forbidden keys, no-op, and fail-fast behavior.
- Modify: `tests/test_stage2_rollout_runtime.py`
  - Server and colocate integration tests with fake owner/client/engine.
- Modify: `tests/test_training_config_strict_unknown_keys.py`
  - Canonical `rollout_matching.vllm.*` path and alias rejection tests if not already covered.
- Modify after implementation: `docs/training/STAGE2_RUNBOOK.md`
  - Short operational note for adapter-only save plus materialized full-sync runtime semantics.

## Task 0: Reconfirm The Draft Gate

- [ ] Confirm OpenSpec validates before implementation:

```bash
openspec validate materialize-vllm-full-sync-adapter-rows --strict
openspec status --change materialize-vllm-full-sync-adapter-rows
```

- [ ] Confirm the user approved implementation after reviewing this plan.
- [ ] Confirm current worktree dirt is still limited to the OpenSpec, this plan, and the vLLM gate config unless the user has added new work.

## Task 1: Add Failing Unit Tests First

**Files:**
- Create: `tests/tokens/test_vllm_sync_materialization.py`

- [ ] Add test scaffolding with two toy models:

```python
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from src.tokens.row_offsets import install_coord_offset_adapter
from src.trainers.rollout_runtime.vllm_sync_materialization import (
    materialize_state_dict_for_vllm_full_sync,
)


class TinyTiedModel(nn.Module):
    def __init__(self, *, tie_config: bool | None = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        if tie_config is not None:
            self.config = types.SimpleNamespace(tie_word_embeddings=bool(tie_config))

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


class TinyUntiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = types.SimpleNamespace(tie_word_embeddings=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


def _non_coord_ids(coord_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    coord = {int(x) for x in coord_ids.tolist()}
    return torch.tensor([idx for idx in range(vocab_size) if idx not in coord], dtype=torch.long)
```

- [ ] Add `test_materialized_tied_coord_offset_matches_hook_logits_and_rows`:
  - Install a tied `coord_offset_adapter` with coord rows `[2, 5]`.
  - Set `embed_offset` to non-zero, non-symmetric values.
  - Build a state dict with `embed_tokens.weight`, `lm_head.weight`, and `coord_offset_adapter.*` tensors.
  - Call `materialize_state_dict_for_vllm_full_sync`.
  - Assert hook logits match a plain model loaded from the materialized rows.
  - Assert `embed_tokens.weight[coord_ids] == base_embed[coord_ids] + embed_offset`.
  - Assert `lm_head.weight[coord_ids] == base_head[coord_ids] + embed_offset`.
  - Assert non-coordinate rows are unchanged for both tensors.
  - Assert no key containing `coord_offset_adapter`, `modules_to_save`, `original_module`, or `lora_` remains.
  - Assert the input `state_dict` tensors were not mutated.

- [ ] Add `test_materialized_untied_coord_offset_patches_distinct_head_rows`:
  - Use `TinyUntiedModel`.
  - Install with `tie_head=False`.
  - Set distinct `embed_offset` and `head_offset`.
  - Assert embedding rows use `embed_offset`.
  - Assert output rows use `head_offset`, not `embed_offset`.
  - Assert logits match hook behavior and non-coordinate rows are unchanged.

- [ ] Add `test_peft_modules_to_save_wrapper_active_adapter_is_discovered`:
  - Wrap or simulate the active `CoordOffsetAdapter` through PEFT `ModulesToSaveWrapper` when PEFT is importable.
  - Verify the helper discovers the active saved module, not the stale pre-wrapper instance.
  - Skip with `pytest.importorskip("peft")` if the local test env lacks PEFT.

- [ ] Run one focused test before implementation and confirm it fails from missing helper import:

```bash
conda run -n ms python -m pytest tests/tokens/test_vllm_sync_materialization.py::test_materialized_tied_coord_offset_matches_hook_logits_and_rows -q
```

## Task 2: Implement The Materialization Helper

**Files:**
- Create: `src/trainers/rollout_runtime/vllm_sync_materialization.py`

- [ ] Implement public helper:

```python
def materialize_state_dict_for_vllm_full_sync(
    *,
    model: Any,
    state_dict: Mapping[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    ...
```

- [ ] Key matching rules:
  - Treat any component named `coord_offset_adapter` as CoordExp token-row state.
  - Treat `modules_to_save.<adapter>.` as learner-only state for every adapter name, not only `default`.
  - Treat any component named `original_module` as learner-only state.
  - Treat any component starting with `lora_` as LoRA/DoRA residue.
  - Validate after filtering that no forbidden key remains.

- [ ] Adapter discovery rules:
  - First discover direct `CoordOffsetAdapter` instances from `model.named_modules()`.
  - Also support PEFT `ModulesToSaveWrapper` by selecting the active adapter from `active_adapters` and `modules_to_save`.
  - If the state dict contains CoordExp token-row keys but no active adapter can be discovered, raise `ValueError`; never silently filter those keys.
  - If there is no active adapter and no CoordExp token-row key, return a new mapping with any residual LoRA/original/module-save keys removed.

- [ ] Row patching rules:
  - Locate ordinary weights by suffix, accepting prefixed names such as `model.embed_tokens.weight` and `base_model.model.lm_head.weight`.
  - Prefer the shortest matching suffix key for deterministic behavior.
  - Clone only tensors that receive row updates.
  - Validate `coord_ids` is 1D, offsets are 2D, row counts match, hidden sizes match, and token IDs are in bounds.
  - Convert offsets to the target tensor dtype/device before addition.
  - Do not mutate the input mapping or input tensors.

- [ ] Tied-head rules:
  - If `lm_head.weight` is present, patch it with `embed_offset` for `tie_head=True`.
  - If `lm_head.weight` is absent and `tie_head=True`, proceed only when the model confirms tied output embeddings through `config.tie_word_embeddings`, `config.text_config.tie_word_embeddings`, or shared input/output embedding storage.
  - If tying is unknown, raise `ValueError` with `tie_word_embeddings` or `lm_head.weight` in the message.

- [ ] Untied-head rules:
  - For `tie_head=False`, require `head_offset` and `lm_head.weight`.
  - Missing `head_offset` or missing `lm_head.weight` must fail before vLLM sync.

- [ ] Diagnostics:
  - Log one compact info line only when materialization is applied.
  - Include row count, embed key, optional head key, and tie mode.
  - Do not log large key lists unless raising; exception previews may show the first few forbidden keys.

- [ ] Run unit tests from Task 1 and keep failing tests visible until the helper passes:

```bash
conda run -n ms python -m pytest tests/tokens/test_vllm_sync_materialization.py -q
```

## Task 3: Add Fail-Fast And Hygiene Coverage

**Files:**
- Modify: `tests/tokens/test_vllm_sync_materialization.py`

- [ ] Add `test_coord_keys_without_discoverable_adapter_fail_fast`:
  - Use a model without `coord_offset_adapter`.
  - Pass a state dict containing `coord_offset_adapter.embed_offset` or `modules_to_save.alt.coord_offset_adapter.embed_offset`.
  - Assert `ValueError` and no filtering-only success.

- [ ] Add `test_no_adapter_noops_without_mutating_input_mapping`:
  - Use a model without adapter and a state dict with ordinary weights plus residual `lora_A.default.weight`, `lora_magnitude_vector.default.weight`, and `original_module.weight`.
  - Assert residual forbidden keys are removed from the returned dict.
  - Assert ordinary tensors in the input mapping are unchanged and the input mapping still has its original keys.

- [ ] Add `test_forbidden_key_detection_is_not_default_adapter_specific`:
  - Cover `modules_to_save.default.*`, `modules_to_save.alt.*`, `coord_offset_adapter.modules_to_save.default.*`, `.original_module.`, `lora_A`, `lora_B`, `lora_embedding_A`, and `lora_magnitude_vector`.

- [ ] Add tied-head missing-head tests:
  - `test_tied_missing_lm_head_allowed_when_config_and_storage_confirm_tying`.
  - `test_tied_missing_lm_head_rejected_when_tying_is_unknown`.
  - `test_tied_missing_lm_head_rejected_when_config_says_untied`.

- [ ] Add shape and bounds tests:
  - `coord_ids` not 1D.
  - offset row count mismatch.
  - offset hidden size mismatch.
  - negative coord ID.
  - coord ID >= vocab size.

- [ ] Run:

```bash
conda run -n ms python -m pytest tests/tokens/test_vllm_sync_materialization.py -q
```

## Task 4: Integrate Server And Colocate Full-Sync

**Files:**
- Modify: `src/trainers/rollout_runtime/vllm_server.py`
- Modify: `src/trainers/rollout_runtime/vllm_engine.py`

- [ ] In `vllm_server.py`, import:

```python
from .vllm_sync_materialization import materialize_state_dict_for_vllm_full_sync
```

- [ ] In `sync_vllm_server_full_weights`, call after current PEFT canonicalization and before `owner._vllm_server_update_state_dict(client, state_dict)`:

```python
state_dict = materialize_state_dict_for_vllm_full_sync(
    model=owner.model,
    state_dict=state_dict,
    logger=logger,
)
owner._vllm_server_update_state_dict(client, state_dict)
```

- [ ] In `vllm_engine.py`, import the same helper.

- [ ] In `sync_vllm_full_weights_if_needed`, call after current PEFT canonicalization and before `engine.inner_model.load_weights(state_dict.items())`:

```python
state_dict = materialize_state_dict_for_vllm_full_sync(
    model=train_model,
    state_dict=state_dict,
    logger=getattr(owner, "logger", None),
)
engine.inner_model.load_weights(state_dict.items())
```

- [ ] Preserve the existing `finally: train_model.unmerge_adapter()` and `finally: owner.model.unmerge_adapter()` behavior. Add regression coverage before changing that structure.

## Task 5: Add Runtime Integration Tests

**Files:**
- Modify: `tests/test_stage2_rollout_runtime.py`

- [ ] Add `test_vllm_server_full_sync_materializes_coord_offset_rows`:
  - Build a fake owner with `model=TinyTiedModel(tie_config=True)` plus active coord adapter.
  - Use a fake `_vllm_server_update_state_dict(client, state_dict)` that stores the state dict.
  - Call `sync_vllm_server_full_weights(owner=owner, client=object(), logger=_Logger())`.
  - Assert captured state contains patched `embed_tokens.weight` and `lm_head.weight` rows.
  - Assert non-coordinate rows are unchanged.
  - Assert no forbidden key is captured.

- [ ] Add `test_vllm_colocate_full_sync_materializes_coord_offset_rows`:
  - Build a fake owner with `_ensure_vllm_engine()` returning an object whose `inner_model.load_weights()` records `(name, tensor)` pairs and whose `engine.reset_prefix_cache()` records a call.
  - Set `_vllm_last_loaded_step=-1`, `state.global_step=1`, and `accelerator=None`.
  - Call `sync_vllm_full_weights_if_needed(owner=owner)`.
  - Assert loaded weights contain patched rows, no forbidden keys, and prefix cache reset happened.

- [ ] Add `test_peft_merge_is_unmerged_when_materialization_raises`:
  - Monkeypatch `accelerate.utils.is_peft_model` to return `True`.
  - Use a fake model whose `merge_adapter()` sets `merged=True` and whose `unmerge_adapter()` restores `merged=False`.
  - Force materialization to raise after merge, for example with active coord keys but missing `embed_tokens.weight`.
  - Assert the sync function raises and `merged` is `False` afterward.

- [ ] Run:

```bash
conda run -n ms python -m pytest tests/tokens/test_vllm_sync_materialization.py tests/test_stage2_rollout_runtime.py -q
```

## Task 6: Config Guards And Documentation

**Files:**
- Modify: `tests/test_training_config_strict_unknown_keys.py`
- Modify if needed: `src/config/rollout_matching_schema.py`
- Modify: `docs/training/STAGE2_RUNBOOK.md`

- [ ] Confirm or add validation that active token-row adapters do not use native vLLM adapter-only LoRA sync:
  - Canonical rejected path: `rollout_matching.vllm.enable_lora=true`.
  - Canonical full-sync path: `rollout_matching.vllm.sync.mode=full`.

- [ ] Add strict unknown-key coverage for legacy/top-level aliases:
  - `vllm.enable_lora`
  - `vllm.sync.mode`
  - `enable_lora`
  - `sync.mode`
  - Error should point users toward `rollout_matching.vllm.enable_lora` and `rollout_matching.vllm.sync.mode`.

- [ ] Add runbook paragraph:

```markdown
For adapter-backed compact-full checkpoints with `coord_offset_adapter`,
Stage-2 native vLLM rollout uses `rollout_matching.vllm.sync.mode=full` as a
runtime materialized ordinary-weight snapshot. This does not permanently merge
the base model and does not change adapter-only checkpoint saving. The sync
snapshot patches coord/schema token rows before native vLLM `load_weights()`
and rejects filtering-only behavior. True native vLLM adapter-only row-sync is
a future performance project.
```

- [ ] Run:

```bash
conda run -n ms python -m pytest tests/test_training_config_strict_unknown_keys.py -q
```

## Task 7: vLLM Gate Verification After Approval

**Config:**
- `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml`

**Log:**
- `temp/vllm6srv2lr_tail_gate.log`

- [ ] Run cfg-only:

```bash
conda run --no-capture-output -n ms python -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml \
  --cfg-only
```

Expected: `status: ok`.

- [ ] Launch the gate only after implementation and user approval:

```bash
tmux new-session -d -s vllm6srv2lr_tail_gate \
  "cd /data/CoordExp/.worktrees/unified-training-infra-refactor && \
   server_gpus=0,1,2,3,4,5 train_gpus=6,7 \
   config=configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml \
   WAIT_TIMEOUT=900 WAIT_INTERVAL=2 MASTER_PORT=29674 \
   conda run --no-capture-output -n ms bash scripts/train_stage2.sh \
   2>&1 | tee temp/vllm6srv2lr_tail_gate.log"
```

- [ ] Monitor:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 200 temp/vllm6srv2lr_tail_gate.log
```

Expected evidence:
- No native vLLM unknown-key error for `coord_offset_adapter`.
- Server topology remains 6 vLLM ranks and 2 learner ranks, ratio 3:1.
- Decode batch cap remains 4.
- First rollout reaches inference requests or exposes the next concrete blocker.
- GPU memory/OOM result is observable rather than inferred.

## Execution Result

Implemented and verified. Targeted tests passed, the OpenSpec validates, the
vLLM 6-server / 2-learner gate completed without OOM or unknown
`coord_offset_adapter` sync failure, and the 1-step save gate produced an
adapter-only checkpoint.
