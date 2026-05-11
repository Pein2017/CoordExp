thread_id: 019d4bfc-882a-76e3-8441-908b4f5df121
updated_at: 2026-04-02T02:29:18+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/02/rollout-2026-04-02T02-18-54-019d4bfc-882a-76e3-8441-908b4f5df121.jsonl
cwd: /data/CoordExp
git_branch: main

# The user investigated coord-offset training/merge behavior for a Qwen3-VL tie-head checkpoint, confirmed the adapter was trained and merged correctly, then asked to rewrite misleading merge-script messages.

Rollout context: repo root was `/data/CoordExp`. The user was training from `configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first_1024_lvis_proxy.yaml` and saw `scripts/merge_coord.sh` log `lm_head.weight not found; adapter uses tie_head=True, so only embed_tokens.weight will be patched.` They asked to add a fail-fast safeguard for unsafe coord-weight initialization, then clarified they wanted to investigate whether the tie-head behavior was actually desired, and finally asked to update the script messages so they would not be misleading anymore.

## Task 1: Investigate coord-offset training / merge behavior and add a safeguard

Outcome: success

Preference signals:
- The user asked: `Please help me add a safe guard to avoid initializing the coord weights and should fail fast.` -> they wanted unsafe coord-weight handling to stop loudly rather than silently continue.
- After seeing the initial diagnosis, the user said: `Sorry, I'm not sure whether this is actually desired effect for the tie lm head model architecture. Please help me investigate it.` -> they wanted evidence-based investigation before assuming the merge log was wrong.
- The user’s concern was not just a generic bug report; they specifically cared about whether tie-head Qwen3-VL behavior should patch only `embed_tokens.weight`.

Key steps:
- Confirmed the stage-1 config enables `custom.coord_offset.enabled: true` and `tie_head: true` in `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`.
- Verified `src/sft.py` installs `coord_offset_adapter`, appends `coord_offset_adapter` to `modules_to_save`, and reattaches coord hooks after `prepare_model()`.
- Read `src/coord_tokens/offset_adapter.py` and confirmed `CoordOffsetAdapter` initializes `embed_offset` to zeros, uses `head_offset = None` in tie-head mode, and shares the same tensor for logits when `tie_head=True`.
- Inspected the real checkpoint at `output/stage1/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy/epoch_2-continue-from-4b/v1-20260401-120504/checkpoint-1564` and found `adapter_model.safetensors` contains `base_model.model.coord_offset_adapter.embed_offset` with shape `(1000, 2560)`, `absmax≈0.008423`, and `2,558,372 / 2,560,000` elements above `1e-6`, so the coord adapter was actually trained.
- Examined `trainer_state.json`; coord diagnostics were active (`coord_diag/enabled: 1.0`) and the coord losses moved during training, so the training path was live.
- Loaded the merged checkpoint with `Qwen3VLForConditionalGeneration.from_pretrained('output/stage1/coco_bbox-lvis_proxy-merged')` and verified `get_input_embeddings().weight` and `get_output_embeddings().weight` are tied after load (`same_data_ptr=True`). This proved that for this architecture, patching only `embed_tokens.weight` is the correct merge path when `tie_word_embeddings=true`.
- Compared merged vs base merged coord-token rows (`151670:152669`) and found the delta was non-zero (`coord_delta_absmax=0.0084228515625`, `coord_delta_nonzero_gt1e-6=2558372`), confirming the merge actually baked in the learned coord offsets.
- Added a fail-fast guard in `scripts/tools/inject_coord_offsets.py`: if a tie-head adapter has no standalone `lm_head.weight` but the merged config is not explicitly tied, it now raises a `RuntimeError` instead of proceeding on an unsafe path.
- Updated `scripts/tools/verify_coord_tokens.py` so it correctly treats tie-head adapters with only `embed_offset` as valid rather than failing because `head_offset` is absent.
- Verified both changes with `py_compile`, the updated verifier on the real checkpoint, and a synthetic unsafe/safe merge fixture.

Failures and how to do differently:
- The first verifier invocation used the wrong CLI flag (`--adapter_dir` instead of `--adapter`) and failed.
- `AutoModelForCausalLM` was the wrong loader for `Qwen3VLConfig`; the correct class was `Qwen3VLForConditionalGeneration`.
- One synthetic safety test was interrupted by deleting the temp directory too early; rerunning the fixture setup and merge sequentially resolved it.
- The original merge log wording was misleading because it sounded like skipping `lm_head.weight` might be suspicious, when for tied Qwen3-VL it is expected.

Reusable knowledge:
- For Qwen3-VL tied checkpoints, `tie_head=True` means there is only one learned coord-offset tensor (`embed_offset`), and logits reuse that same tensor.
- In this repo, `modules_to_save: ['coord_offset_adapter']` is the mechanism that persists coord offsets inside the adapter checkpoint.
- `scripts/merge_coord.sh` calls `scripts/tools/inject_coord_offsets.py`; the latter is the right place to enforce merge-time safety.
- The real checkpoint proved that coord training can succeed even when `head_offset` is absent; absence of `head_offset` is not a bug in tie-head mode.

References:
- [1] `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`: `custom.coord_offset.enabled: true`, `tie_head: true`, coord id range `151670..152669`.
- [2] `src/coord_tokens/offset_adapter.py`: tie-head uses shared `embed_offset`; `head_offset` is `None` in tie mode.
- [3] `output/stage1/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy/epoch_2-continue-from-4b/v1-20260401-120504/checkpoint-1564/adapter_model.safetensors`: `base_model.model.coord_offset_adapter.embed_offset` had `absmax≈0.008423` and was non-zero across almost all elements.
- [4] `output/stage1/coco_bbox-lvis_proxy-merged/config.json`: `tie_word_embeddings: true` at top level and in `text_config`.
- [5] `Qwen3VLForConditionalGeneration.from_pretrained('output/stage1/coco_bbox-lvis_proxy-merged')` -> `same_data_ptr True`, proving tied weights after load.
- [6] `scripts/tools/inject_coord_offsets.py`: new fail-fast error when tie-head adapter meets an untied/unknown merged config.
- [7] `scripts/tools/verify_coord_tokens.py`: now prints `No head_offset tensor; adapter uses tie_head=True` and treats that as success.

## Task 2: Update `scripts/merge_coord.sh` messages to avoid misleading wording

Outcome: success

Preference signals:
- The user asked: `Help me update the messages of scripts/merge_coord.sh to avoid misleading anymore` -> they wanted the script output to match the confirmed tie-head semantics, not just the code behavior.
- This followed the investigation task, so the user wanted the user-facing logs changed to reflect the proven architecture behavior.

Key steps:
- Read `scripts/merge_coord.sh` and identified the misleading areas:
  - the comment above coord injection
  - the `tied` / `untied` / `unknown` detection messages
  - the final `tie_word_embeddings` warning block.
- Updated the comment to explicitly say tie-head adapters store only a shared `embed_offset` tensor and that, for tied Qwen-family checkpoints, patching `embed_tokens.weight` is expected because `lm_head` resolves to the same weights after load.
- Added an explicit informational branch for `COORD_OFFSETS_MODE=tied`:
  - `Adapter uses tie_head=True (shared coord offsets).`
  - `For tied Qwen-family checkpoints, injecting embed_tokens.weight is the expected merge path.`
- Refined the final config warning block so it distinguishes:
  - `tie_word_embeddings=true` -> informational, expected tied behavior
  - `tie_word_embeddings=false` -> warning for untied export
  - `unknown` -> warning to verify reload behavior.
- Verified with `bash -n scripts/merge_coord.sh`.

Failures and how to do differently:
- The original wording implied `lm_head.weight` being absent was potentially suspicious; for tied Qwen3-VL checkpoints that message should instead state that this is expected.
- The script had collapsed `true`, `false`, and `unknown` config states into a single warning path; that was too coarse for this architecture.

Reusable knowledge:
- `scripts/merge_coord.sh` is the human-facing place where merge semantics should be described clearly; `scripts/tools/inject_coord_offsets.py` is the enforcement point.
- For tied checkpoints, the absence of a standalone `lm_head.weight` shard is expected and should not be framed as an error condition.
- The repo already has a validated tie-head default path in `scripts/tools/expand_coord_vocab.py`, which forces and verifies tied embeddings during vocab expansion.

References:
- [1] `scripts/merge_coord.sh` now contains explicit tie-head commentary and a `COORD_OFFSETS_MODE == tied` info branch.
- [2] Final config messaging now branches on `true`, `false`, and `unknown` instead of only warning whenever the value is not `true`.
- [3] Verification command: `bash -n scripts/merge_coord.sh` passed after the edit.
