# Change: Add rollout-matching SFT trainer variant (JSON output, matching, self-context coord supervision)

## Why
This proposal introduces a rollout-matching SFT trainer variant (alias: `stage_2` in `progress/full_idea.md`).

Baseline training (alias: `stage_1`, see `progress/pretrain/first_stage_v2.md`) stabilizes **JSON-only** structured outputs and teaches the model to emit `<|coord_k|>` tokens reliably using **distributional coord supervision** (`softCE + W1 + gate`) while applying standard CE to non-coord tokens.

`progress/full_idea.md` motivates an **EM-ish** refinement loop:
- **E-step**: rollout the current model to obtain a hypothesis output (and its tokenization),
- **latent alignment**: match predicted objects to GT objects (Hungarian),
- **M-step**: update parameters using losses that are consistent with the rollout context.

To proceed, CoordExp needs a training infrastructure that can run **rollout → parse → match → single teacher-forced forward → masked losses** efficiently, while keeping the project’s core constraints:
- **JSON-only assistant output** (no `<obj><desc><box>` protocol),
- **coord supervision remains token-distributional** (`softCE + W1 + gate`) at coord slots, not expectation-decoding or L1/GIoU,
- **YAML-first** configuration (no new hyperparameter CLI flags).

## What Changes
- Add a new opt-in trainer variant `custom.trainer_variant: rollout_matching_sft` that implements rollout-matching SFT training (alias: `stage_2`):
  - **Rollout loop** (no grad): generate the assistant JSON response from the current model with configurable decoding.
  - **Strict parsing + alignment**: parse the rollout into schema-conformant predicted objects in *appearance order* (no key sorting, no token-inserting JSON repair), and deterministically align each valid object’s coord slots to token indices in the rollout token sequence (bbox_2d and poly). Suffix-only trimming of truncated rollouts is allowed to drop incomplete tails without changing earlier tokens.
  - **Hungarian matching (MVP baseline)**: align valid predicted objects to GT objects using Hungarian assignment with dummy augmentation, candidate pruning, `maskIoU` geometry costs, and pre-assignment gating.
  - **Single-path training sequence + one forward**: construct one teacher-forced assistant target per sample:
    - `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`
    - `Y_rollout_prefix` is the rollout assistant tokens with suffix-only trimming (drop `<|im_end|>`; drop incomplete tail; drop final top-level `}` so the JSON object is open for append).
    - unmatched GT objects (FN) are **always** appended to recover recall.
  - **One loss with per-token masks** (single forward pass):
    - **Rollout prefix self-context:** apply coord-only distributional supervision (`softCE + W1 + gate`) at matched predicted coord-slot token indices (bbox_2d direct; poly via Sinkhorn OT + barycentric projection targets).
    - **Appended GT tail:** apply hard CE on non-coord tokens and `softCE + W1 + gate` on coord tokens (baseline semantics).
    - No expectation decoding; no L1/GIoU/mask losses; no IoU/maskIoU metric logging.
- Deprecate `line` geometries throughout the pipeline (prompts/contract/parsing/metrics): CoordExp standardizes on `bbox_2d` and `poly` only.
- Provide a YAML config template under `configs/` for rollout-matching runs and a minimal runbook entry under `progress/` (paper-ready naming, deterministic seeds, key counters).
- Enforce known constraints explicitly:
  - **Packing is not supported** for rollout-matching generation; enabling `training.packing: true` while `trainer_variant=rollout_matching_sft` SHALL fail fast with a clear error.

## Impact
- Affected specs:
  - **New capability:** `rollout-matching-sft` (added)
  - Existing capabilities (used, not modified): `coord-token-mode` (distributional coord losses), `inference-engine` (shared parsing utilities are reused)
- Affected code (expected):
  - `src/sft.py` (trainer selection by `custom.trainer_variant`; rollout-specific collator routing)
  - `src/trainers/` (new rollout-matching trainer)
  - `src/eval/parsing.py` (reference for shared schema/geometry helpers; training uses strict parsing without repair)
  - `src/coord_tokens/soft_ce_w1.py` (reuse for distributional coord loss computation)
  - `configs/` (new rollout-matching YAML)
  - `tests/` (parser/matcher/trainer unit tests)

## Backward Compatibility
- Default behaviour is unchanged; rollout-matching training (alias `stage_2`) is opt-in via `custom.trainer_variant`.
- This introduces a new failure mode only when explicitly enabled: packing is rejected to avoid silent mis-training.

## Non-goals
- No expectation-decoding, no continuous L1/GIoU geometry losses.
- No migration to a non-JSON output protocol.
- No changes to upstream Qwen3-VL internals.
- vLLM rollout acceleration is desirable but not required for the initial MVP; the trainer architecture SHOULD keep a clean path to add it later.
