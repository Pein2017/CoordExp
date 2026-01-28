# Audit Report: Stage-2 AB (Iterative Soft Self-Context + Rollout Matching) bbox-only v1 — Tensor Flow (positions/masks/logits/caches)

## Executive summary
**Status: PASS (Channel‑B as currently implemented) / CONDITIONAL PASS (Channel‑A proposal)**.

- The existing Stage‑2 *rollout-matching* trainer already contains a correct and **necessary** guard for Qwen‑VL mRoPE + packed/padding‑free training: it reconstructs a **4‑row `position_ids`** `[text_position_ids; mRoPE(t/h/w)]` before the model forward so that Transformers’ Qwen3‑VL uses the correct *text* positions for packed boundaries.
- With `flash_attention_2` (preferred), packed boundaries are enforced either by (a) **precomputed varlen kwargs** (`cu_seq_lens_*`, `max_length_*`) that ms‑swift emits for packed batches, or (b) Transformers inferring boundaries from **`position_ids==0`**. The audit reproduces and demonstrates the **silent correctness hazard** if the wrong row is used as `text_position_ids`.
- Channel‑A (“iterative soft self‑context”) is not implemented in-repo; it can be safe, but only if it **reuses the exact same packed boundary contract** (either ms‑swift `cu_seq_lens_*` or correct `text_position_ids` row) and never reuses/accidentally enables kv‑cache across iterations.

## Scope
The audit is grounded in the plan text around the proposal bullets that materially constrain tensor contracts:
- `openspec/changes/2026-01-27-add-stage2-ab-iterative-softctx-bboxonly/proposal.md:10` (Channel‑A iterative full‑forwards) through `openspec/changes/2026-01-27-add-stage2-ab-iterative-softctx-bboxonly/proposal.md:15` (compatibility with ms‑swift/Transformers; no upstream patches).

Focus areas (per request): token positions, attention masks, logits computation, logits storage/caching, and tensor shape/layout changes across CoordExp + ms‑swift + Transformers (Qwen3‑VL) with `flash_attention_2`.

## Repo safety / working tree
- `git status --porcelain` at audit start/end shows only: `?? openspec/changes/2026-01-27-add-stage2-ab-iterative-softctx-bboxonly/` (untracked).
- No repo source files were modified. Evidence scripts/logs were written only under `temp/`.

## Environment (from evidence run)
From `temp/audit_stage2_tensor_flow.log`:
- torch: `2.8.0+cu128`, cuda: `12.8`
- transformers: `4.57.1`
- flash_attn: `2.8.3`

## Files & functions participating in tensor flow (with roles)

### Proposal scope anchor
- `openspec/changes/2026-01-27-add-stage2-ab-iterative-softctx-bboxonly/proposal.md:10` (through line 15)
  - Role: declares the two-channel Stage‑2 flow and “no upstream patches” constraint that makes the *tensor contracts* the critical correctness surface.

### CoordExp (our repo)
- `src/trainers/rollout_matching_sft.py:5594-5783` — `RolloutMatchingSFTTrainer.compute_loss`
  - Role: **central training forward**; strips helper keys, constructs **4‑row `position_ids`** for Qwen‑VL + packing, calls `model(**inputs_for_model)`, then computes CE + coord losses from logits.
  - Key tensor operations:
    - position fix: concatenates `text_position_ids` onto `position_ids` (mRoPE) when `model_type.startswith("qwen")` and shapes match.
    - logits: reads `outputs.logits` and builds `logits_next = logits[:, :-1, :]`.
    - coord supervision: `logits_full = logits_next[b_t, logit_pos, :]`, then `logits_coord = logits_full.index_select(-1, coord_ids_t)`.

- `src/trainers/rollout_matching_sft.py:2802-2952` — `RolloutMatchingSFTTrainer._rollout_one`
  - Role: single-sample rollout helper; **drops `position_ids` / `text_position_ids` for generation**, ensuring generation uses upstream Qwen3‑VL generation contract (position_ids computed internally).

- `src/trainers/rollout_matching_sft.py:3829-4035` — `RolloutMatchingSFTTrainer._rollout_many_hf`
  - Role: batch rollout via HF `generate`; similarly **pops `position_ids` / `text_position_ids`** before `template.generate`.

- `src/trainers/rollout_matching_sft.py:4684-5242` — `RolloutMatchingSFTTrainer._prepare_batch_inputs`
  - Role: builds teacher-forced training batches from rollout outputs; creates the dict that later flows into `compute_loss` (includes `input_ids`, `attention_mask` or `None` in padding_free packing, `position_ids`, `text_position_ids`, and rollout meta).

- `src/trainers/rollout_matching_sft.py:1221-1307` — `_build_labels_and_coord_targets_for_sample`
  - Role: creates masked CE labels and coord supervision indices; ensures coord tokens are excluded from CE and only supervised via coord loss.

- `src/trainers/rollout_matching_sft.py:1310-1452` — `_build_labels_and_coord_targets_for_batch`
  - Role: batch wrapper supporting packed/unpacked meta; produces `labels_masked` and `(batch,pos,bin,is_prefix)` vectors used to gather from logits.

- `src/trainers/rollout_matching_sft.py:1491-1512` — `_copy_prepared_batch_for_training_step`
  - Role: shallow-copy input dict before training_step/compute_loss mutates via `pop`; prevents **buffered batch reuse** from being corrupted.

- `src/trainers/rollout_matching_sft.py:1722-1769` — `_RolloutWindowBuffer.select_batch`
  - Role: caching/reuse of prepared batches across optimizer steps (M‑steps); ensures reused batches are copied to avoid mutation.

### ms-swift (external; used by our trainer/template)
- `/data/ms-swift/swift/llm/template/template/qwen.py:383-393` — `Qwen2VLTemplate.forward_context`
  - Role: (legacy) patch path for older Transformers; with transformers>=4.53 this is effectively a no-op (returns super), but included for completeness.

- `/data/ms-swift/swift/llm/template/template/qwen.py:395-405` — `Qwen2VLTemplate._post_encode`
  - Role: multimodal training hook: converts `input_ids` -> `inputs_embeds` and inserts visual features (so model forward often sees `inputs_embeds`, not `pixel_values`).

- `/data/ms-swift/swift/llm/template/template/qwen.py:414-422` — `Qwen2VLTemplate.packing_row`
  - Role: in padding_free packing, concatenates per-sample position_ids along seq dim.

- `/data/ms-swift/swift/llm/template/template/qwen.py:424-443` — `Qwen2VLTemplate._get_position_ids`
  - Role: calls HF Qwen base model `get_rope_index(...)` to compute mRoPE position ids; then prepends a **sequential text row** via `_concat_text_position_ids`.

- `/data/ms-swift/swift/llm/template/template/qwen.py:445-456` — `Qwen2VLTemplate._data_collator`
  - Role: emits `position_ids` as **3-row mRoPE** and separately emits `text_position_ids` (row0), and (for transformers>=4.53) emits packed FA2 kwargs via `get_packed_seq_params(text_position_ids)`.

- `/data/ms-swift/swift/llm/template/base.py:1598-1713` — `Template._data_collator`
  - Role: constructs batch tensors; for `padding_free=True`, flattens sequences into batch_size=1 packed tensor and sets attention_mask semantics.

- `/data/ms-swift/swift/llm/template/base.py:1332-1348` — `Template.pre_forward_hook`
  - Role: forward pre-hook that applies `_post_encode` and preserves keys like `position_ids`, and optionally forwards FA2 varlen kwargs (`cu_seq_lens_*`, `max_length_*`). Drops `input_ids` if `inputs_embeds` is provided.

- `/data/ms-swift/swift/llm/template/base.py:1999-2003` — `Template._concat_text_position_ids`
  - Role: constructs sequential `text_position_ids = arange(seq_len)` and concatenates it as row0 before the 3-row mRoPE.

- `/data/ms-swift/swift/llm/utils.py:320-336` — `get_packed_seq_params`
  - Role: emits FA2 varlen kwargs based on boundaries where `text_position_ids==0`: `cu_seq_lens_{q,k}`, `max_length_{q,k}`.

- `/data/ms-swift/swift/trainers/trainers.py:307-408` — `Seq2SeqTrainer.compute_loss`
  - Role: baseline compute_loss that `pop()`s helper keys (incl. `text_position_ids`) and calls `model(**inputs)`; our rollout-matching trainer bypasses this logic by overriding compute_loss.

### Transformers (HF upstream; local installation)
- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1314-1373` — `Qwen3VLForConditionalGeneration.forward`
  - Role: computes **logits** via `lm_head(hidden_states[:, slice_indices, :])`; returns logits + past_key_values.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1106-1239` — `Qwen3VLModel.forward`
  - Role: multimodal wrapper; optionally computes `position_ids` via `get_rope_index` when `position_ids is None`, then forwards into text model.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:782-874` — `Qwen3VLTextModel.forward`
  - Role: **splits 4-row `position_ids`** into `text_position_ids` (row0) + mRoPE rows; builds causal mask via `create_causal_mask(..., position_ids=text_position_ids)`; computes rotary embeddings; loops decoder layers.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:415-457` — `Qwen3VLTextAttention.forward`
  - Role: applies rotary embeddings to Q/K, updates kv cache via `past_key_values.update(...)` (in-place), then dispatches to attention implementation (`flash_attention_2` preferred).

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:487-519` — `Qwen3VLTextDecoderLayer.forward`
  - Role: passes `position_ids` and FA2 kwargs down to `Qwen3VLTextAttention.forward`.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/masking_utils.py:745-836` — `create_causal_mask`
  - Role: produces mask for the selected attention backend. For `flash_attention_2`, returns `None` if fully causal (no padding). For SDPA/eager, uses `position_ids` for packed sequence detection.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/masking_utils.py:637-661` — `find_packed_sequence_indices`
  - Role: detects packed boundaries via `(diff(position_ids) != 1).cumsum(-1)`.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/masking_utils.py:525-561` — `flash_attention_mask`
  - Role: for FA2 returns `None` if no padding; otherwise returns the 2D attention_mask for unpadding.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/integrations/flash_attention.py:14-84` — `flash_attention_forward`
  - Role: calls `_flash_attention_forward` with provided `position_ids` and/or precomputed varlen kwargs.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:529-668` — `_flash_attention_forward`
  - Role: implements FA2 padding-free packed attention; if attention_mask is None, uses either (a) **precomputed** `(cu_seq_lens_*, max_length_*)` or (b) infers via `position_ids`.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:316-357` — `prepare_fa_kwargs_from_position_ids`
  - Role: infers varlen boundaries where `position_ids==0` and computes `cu_seq_lens` and `max_length`.

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:360-395` — `_prepare_from_posids`
  - Role: flattens Q/K/V and calls `prepare_fa_kwargs_from_position_ids`.

## Call stacks / symbol chains (Stage‑2 tensor flow)

### Chain 1 — Channel‑B training (teacher-forced) logits path
1) `src/trainers/rollout_matching_sft.py:6213` `RolloutMatchingSFTTrainer.training_step`
   -> builds prepared batch via `_prepare_batch_inputs` and possibly buffer reuse.
2) `src/trainers/rollout_matching_sft.py:5594` `RolloutMatchingSFTTrainer.compute_loss`
   -> ensures Qwen packed safety: constructs 4-row `position_ids` from `text_position_ids` + mRoPE.
3) `/data/ms-swift/swift/llm/template/base.py:1332` `Template.pre_forward_hook`
   -> may convert `input_ids` to `inputs_embeds` and forwards `position_ids` and `cu_seq_lens_*`.
4) `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1314` `Qwen3VLForConditionalGeneration.forward`
   -> `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1106` `Qwen3VLModel.forward`
   -> `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:782` `Qwen3VLTextModel.forward`
   -> `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:415` `Qwen3VLTextAttention.forward`
   -> `/root/miniconda3/.../transformers/integrations/flash_attention.py:14` `flash_attention_forward`
   -> `/root/miniconda3/.../transformers/modeling_flash_attention_utils.py:529` `_flash_attention_forward`
   -> logits computed in `Qwen3VLForConditionalGeneration.forward` via `lm_head`.

### Chain 2 — Packed boundary inference (the core hazard)
- `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:782` `Qwen3VLTextModel.forward`
  - if `position_ids.shape[0]==4`: uses row0 as `text_position_ids`.
  - else: uses `position_ids[0]` as `text_position_ids`.
- `_flash_attention_forward` (FA2 padding-free) uses either:
  - explicit `cu_seq_lens_*` (from ms-swift), OR
  - `prepare_fa_kwargs_from_position_ids` which splits sequences at every `position_ids==0`.

### Chain 3 — Channel‑B rollout generation + caching path
1) `src/trainers/rollout_matching_sft.py:2802` `_rollout_one` (or `src/trainers/rollout_matching_sft.py:3829` `_rollout_many_hf`)
   -> `model_inputs.pop("position_ids")` and `pop("text_position_ids")`.
2) HF generation:
   -> `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1375` `Qwen3VLForConditionalGeneration.prepare_inputs_for_generation` (forces `position_ids=None`)
   -> `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1314` forward
   -> caching via `DynamicCache` and in-place `past_key_values.update(...)` in `Qwen3VLTextAttention.forward`.

## Evidence artifacts

### Instrumented script
- Script: `temp/audit_stage2_tensor_flow.py`
  - Non-invasive: monkeypatches in-memory Transformers functions to log shapes/dtypes and short stacks.
  - Logs key points: embedding output, mask creation (`create_causal_mask`), packed boundary detection (`find_packed_sequence_indices`), FA2 varlen inference (`_prepare_from_posids`), logits (`lm_head`), and caching (`DynamicCache`).

### Command(s) executed
- `conda run -n ms python temp/audit_stage2_tensor_flow.py 2>&1 | tee temp/audit_stage2_tensor_flow.log`

### Key log excerpts (GOOD vs BAD packed boundaries)
From `temp/audit_stage2_tensor_flow.log` (line numbers from `nl -ba`):

```text
# GOOD: correct text_position_ids boundaries
find_packed_sequence_indices ... head=[0, 1, 2, 0, 1, 2, 3, 4] -> ... head=[0, 0, 0, 1, 1, 1, 1, 1]
FA _prepare_from_posids ... head=[0, 1, 2, 0, 1, 2, 3, 4] -> cu_seq_lens_q=[0, 3, 8]

# ms-swift style: explicit cu_seq_lens (bypasses position_id inference)
Qwen3VLTextAttention ... cu_seq_lens_q=shape=(3,) ... max_length_q=tensor(5, ...)  (and no FA _prepare_from_posids log)

# BAD: if the model misuses mRoPE temporal row as text_position_ids
find_packed_sequence_indices ... head=[0, 1, 2, 0, 0, 1, 2, 3] -> ... head=[0, 0, 0, 1, 2, 2, 2, 2]
FA _prepare_from_posids ... head=[0, 1, 2, 0, 0, 1, 2, 3] -> cu_seq_lens_q=[0, 3, 4, 8]
```

### Cache evidence
From the same run:
- `prefill ... past_key_values=DynamicCache seq_len=8`
- `decode1 ... past_key_values=DynamicCache seq_len=9`

## Risk table

| Risk | Severity | Confidence | Recommended verification steps |
|---|---:|---:|---|
| **Packed boundary corruption** if Channel‑A softctx uses 3-row mRoPE `position_ids` without (a) 4-row fix or (b) `cu_seq_lens_*` kwargs (FA2) | High | High | Add an explicit unit/integration check that Stage‑2 Channel‑A forwards always satisfy: `position_ids.shape[0]==4` OR `cu_seq_lens_q` present when `attention_mask is None`. Re-run `temp/audit_stage2_tensor_flow.py` adapted to the Channel‑A forward wrapper. |
| **Cache leakage across softctx iterations** (reusing `past_key_values` or accidentally enabling `use_cache=True`) causing stale attention | Medium | Medium | In Channel‑A implementation, hard‑set `use_cache=False` and never pass `past_key_values`. Add assertion that outputs.past_key_values is None during training forwards. |
| **Silent shape mismatch** if any path enables `logits_to_keep` (slicing logits) while CoordExp losses assume full `seq_len` logits | Medium | Medium | In Stage‑2 configs, keep `use_logits_to_keep=None/False` (default for multimodal). In trainer, optionally strip `logits_to_keep` from inputs (or assert it is 0). |
| **Incorrect boundary inference if attention_mask is accidentally provided in packed/padding_free mode** (should be None) | Medium | Low | Add an assertion in Stage‑2 packing path: if packed (bsz==1 and multiple segments), require `attention_mask is None` and rely on `cu_seq_lens_*`/`position_ids`. |
| **Iterative softctx positional misalignment** if softctx inserts/removes tokens but does not update `text_position_ids` resets / cu_seq_lens | High | Medium | Provide a deterministic tiny test that modifies the prefix and validates that `text_position_ids==0` occurs exactly at packed boundaries and that downstream `cu_seq_lens_q` matches segment lengths. |

## Verification checklist (explicit pass/fail checks)

### 1) Token positions
- [PASS] **Where set:** ms‑swift computes mRoPE via `get_rope_index` and constructs `text_position_ids` via `arange(seq_len)`.
  - Code: `/data/ms-swift/swift/llm/template/template/qwen.py:424` and `/data/ms-swift/swift/llm/template/base.py:1999`.
- [PASS] **No unexpected overwrite in Channel‑B training:** CoordExp reconstructs 4‑row `position_ids` when needed and otherwise passes through.
  - Code: `src/trainers/rollout_matching_sft.py:5594`.
- [PASS] **Generation paths avoid passing stale positions:** rollout generation pops `position_ids` and HF generation sets `position_ids=None`.
  - Code: `src/trainers/rollout_matching_sft.py:2802`, `src/trainers/rollout_matching_sft.py:3829`, `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1375`.
- [TODO / acceptance] Channel‑A must guarantee either 4‑row `position_ids` or `cu_seq_lens_*` varlen kwargs whenever `attention_mask is None`.

### 2) Attention masks
- [PASS] For `flash_attention_2`, `create_causal_mask` returns `None` when fully causal (no padding), relying on FA2 `is_causal` + varlen packing.
  - Code: `/root/miniconda3/.../transformers/masking_utils.py:525` and `/root/miniconda3/.../transformers/masking_utils.py:745`.
- [PASS] Packed boundary detection agrees with expected segment lengths in GOOD case; differs in BAD case (demonstrated by evidence logs).
  - Evidence: `temp/audit_stage2_tensor_flow.log` lines around `find_packed_sequence_indices` and `FA _prepare_from_posids`.

### 3) Logits compute & storage
- [PASS] Logits computed via `lm_head(hidden_states)`; CoordExp uses logits directly (no detach/in-place modification of logits tensors).
  - Code: `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:1314` and `src/trainers/rollout_matching_sft.py:5594`.
- [PASS] No logits caching/storage in rollout buffer; buffer caches prepared batches only and copies metadata to avoid mutation.
  - Code: `src/trainers/rollout_matching_sft.py:1491` and `src/trainers/rollout_matching_sft.py:1722`.

### 4) Caches / soft prompts / iterative updates
- [PASS] Channel‑B training sets `use_cache` off implicitly by trainer (and evidence script uses `use_cache=False` in training-style cases).
- [PASS] Generation uses `DynamicCache` and in-place cache updates; this is expected and isolated to rollout/generation.
  - Evidence: `temp/audit_stage2_tensor_flow.log` shows `DynamicCache seq_len=8 -> 9`.
- [TODO / acceptance] Channel‑A must ensure no `past_key_values` are carried across iterations; recommend an assertion: `outputs.past_key_values is None`.

### 5) Upstream library assumptions / in-place ops
- [PASS] `Qwen3VLTextAttention.forward` updates kv-cache in-place (`past_key_values.update(...)`), so carrying caches across iterations would be unsafe.
  - Code: `/root/miniconda3/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:415`.
- [PASS] FA2 padding-free varlen boundaries are inferred from `position_ids==0` unless `cu_seq_lens_*` are provided.
  - Code: `/root/miniconda3/.../transformers/modeling_flash_attention_utils.py:316` and `/root/miniconda3/.../transformers/modeling_flash_attention_utils.py:529`.

## Acceptance checks (what to run / what to look for)
1) Run the tensor-flow audit:
   - `conda run -n ms python temp/audit_stage2_tensor_flow.py 2>&1 | tee temp/audit_stage2_tensor_flow.log`
2) Confirm in the log:
   - GOOD case shows `cu_seq_lens_q=[0, 3, 8]` (two segments: 3 + 5).
   - BAD case shows extra boundary `cu_seq_lens_q=[0, 3, 4, 8]` (incorrect split).
   - ms‑swift style case shows `cu_seq_lens_q=shape=(3,)` and does **not** log `FA _prepare_from_posids` (i.e., it bypassed inference).
   - Cache case shows `DynamicCache seq_len=8` then `seq_len=9`.
