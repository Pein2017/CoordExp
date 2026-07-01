## 1. Implementation
- [x] 1.1 Add a rollout-matching config surface (YAML-first; alias: `stage_2`):
  - [x] `custom.trainer_variant: rollout_matching_sft`
  - [x] `custom.extra.rollout_matching` dict for rollout/matching knobs (decoding params, gating thresholds, weights).
- [x] 1.2 Add a rollout-matching trainer implementation under `src/trainers/` that:
  - [x] runs rollouts (no grad) using transformers generation (MVP),
  - [x] performs strict, schema-conformant parsing (no repair) to extract predicted objects in appearance order,
  - [x] extracts per-object coord token indices via token-aligned parsing (bbox_2d and poly; no decoded-text pattern search),
  - [x] runs Hungarian matching with dummy augmentation, candidate pruning, maskIoU costs, and pre-assignment gating,
  - [x] constructs a single teacher-forced assistant sequence per sample:
    - [x] `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`
    - [x] `Y_rollout_prefix` uses suffix-only trimming (strip `<|im_end|>`; drop incomplete tail; drop final top-level `}` so JSON is open for append),
    - [x] FN append is mandatory (unmatched GT objects are always appended),
  - [x] runs exactly one forward pass on `Y_train` and computes one total loss using token masks:
    - [x] coord-token supervision (`softCE + W1 + gate`) on matched prefix coord slots (self-context),
    - [x] hard CE on non-coord tokens in the appended GT tail,
    - [x] coord-token supervision (`softCE + W1 + gate`) on coord tokens in the appended GT tail,
    - [x] poly targets use Sinkhorn OT + barycentric projection ONLY (no mixture),
    - [x] no decoded-coordinate losses or IoU/maskIoU metrics logging.
- [x] 1.3 Integrate rollout-matching trainer selection into `src/sft.py` via `custom.trainer_variant`.
- [x] 1.4 Adjust the collator path for rollout-matching:
  - [x] rollout-matching SHOULD use an identity/pass-through collator so `messages` / `assistant_payload` / GT metadata are available in the trainer.
  - [x] Existing baseline collator + metrics MUST remain unchanged for non-rollout-matching trainers.
- [x] 1.5 Add explicit fail-fast checks:
  - [x] reject `training.packing: true` under rollout-matching,
  - [x] reject missing required GT metadata for rollout-matching (e.g., missing `assistant_payload` when needed).
- [x] 1.6 Add a rollout-matching YAML config template under `configs/` (placeholders only; no “default hyperparams”).
- [x] 1.7 Add unit tests:
  - [x] strict rollout parsing (no repair): invalid objects are dropped, valid objects preserved in appearance order,
  - [x] coord token index alignment via token-aligned parsing (bbox_2d + poly),
  - [x] Hungarian matching correctness (dummy augmentation + gating) on a tiny synthetic example,
  - [x] OT+barycentric target construction sanity (poly involved),
  - [x] loss masking invariants:
    - [x] rollout prefix non-coord tokens are ignored,
    - [x] coord tokens never contribute to full-vocab CE,
    - [x] appended tail non-coord tokens contribute to CE,
    - [x] supervised coord indices are within assistant span (sanity checks).
- [x] 1.8 Add training-time logging counters:
  - [x] valid vs invalid predicted objects (dropped),
  - [x] match rate, #matched objects, #FN appended, #gating rejections,
  - [x] distributional coord loss breakdown (prefix coord supervision vs appended tail coord supervision),
  - [x] optional debug dump on parse failures (bounded).

## 1.9 Rollout parsing robustness + desc masking policy (current rollouts)
- [x] 1.9.1 Parsing: treat `<|im_end|>` as a hard stop (strip it even when fused) before strict scanning/trimming.
- [x] 1.9.2 Parsing: suffix-only trimming to the last complete object boundary MUST yield an append-ready prefix (ends with `{` or `}`), or fall back to `{` when no open brace is found.
- [x] 1.9.3 Loss masking: ignore `desc` value tokens for CE (both rollout prefix and appended GT tail). Coord-token supervision remains unchanged.
- [x] 1.9.4 Add unit tests for (1.9.1-1.9.3).

## 2. Rollout-Matching Smoke Test Checklist (paper-ready; alias: stage_2)
NOTE: This section is an operational checklist for running stage_2 experiments. It is intentionally NOT tracked
as OpenSpec tasks (i.e., no checkboxes) so it doesn't block archiving code/spec changes.

- 2.1 Pick a baseline checkpoint as the starting point (`<BASE_CKPT>`).
- 2.2 Pick a small train/val JSONL and set a strict sample limit (`<TRAIN_JSONL>`, `<VAL_JSONL>`, N≈50–200).
- 2.3 Run rollout-matching with deterministic seed + greedy decoding.
- 2.4 Verify:
  - no NaNs/infs,
  - strict parsing yields a non-trivial valid-object fraction (invalid objects are dropped, not repaired),
  - match gating rejects obvious failures and FN append is non-zero when expected,
  - training loss decreases and logs are emitted.
- 2.5 Record run artifacts: resolved YAML, key counters, and a few debug examples.
