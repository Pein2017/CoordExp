# Stage-2 AB: Continuous Soft-Context Mismatch vs Discrete Rollouts (and Stage-1 Loss Alignment)

Date: 2026-02-22

This note records an important diagnosis from Stage-2 AB Channel-A-only experiments (COCO bbox max60, COCO-80 prompt variant), focusing on rollout degradation modes and why a continuous self-context expectation update can diverge from real (discrete) rollout behavior.

Related docs:
- `progress/stage2_ab_channel_a_only_coord_gate_diagnosis_2026-02-21.md` (coord gate loss + em_detach CE-grad bug fix)
- `progress/stage2_channel_a_only_infer_eval_2026-02-01.md` (earlier infer/eval notes on repetition_penalty)


## Executive Summary

Observations from `geo_first` artifacts suggest that a major failure mode is not pure coordinate regression quality, but *open-loop generation control*:
- repeating identical `bbox_2d + desc` many times (runaway list continuation)
- injecting non-coordinate tokens inside `bbox_2d` lists causing `wrong_arity` drops (e.g., inserting `\u7b2c\u4e5d` as the first list element)

This strongly motivates two architectural conclusions:
1. Stage-1 (teacher-forced) training remains the best scalable way to learn robust discrete coord tokens + format.
2. Stage-2 alignment for open-loop rollouts likely requires either:
   - some amount of actual rollout training (Channel-B), or
   - decode-time constraints / termination rules,
   and **Channel-A continuous expectation updates alone may be an insufficient proxy**.

A key discovery: Channel-A updates coord token embeddings using a *continuous expected embedding* derived from coord logits, which can create a train/infer mismatch because real rollouts are discrete coord tokens.


## Background: What Stage-1 vs Stage-2 Channel-A Optimize

### Stage-1 (SFT) coordinate learning
Stage-1 uses teacher forcing with:
- base CE on non-coordinate tokens (coord tokens masked out)
- coord distribution losses on coord-token positions (soft CE + W1 + gate)

Implementation pointers:
- `src/trainers/losses/coord_soft_ce_w1.py` (`compute_coord_soft_ce_w1_loss`, `coord_vocab_gate_loss`, etc.)

Properties:
- directly trains discrete coord token distributions
- strongly improves diagnostics like `coord_diag/p_gt_mean`, `coord_diag/acc_top5`, `coord_diag/expected_bin_mae`

### Stage-2 AB Channel-A (self-context expectation)
Stage-2 AB Channel-A does multiple forward passes and replaces coord-slot input embeddings with an expected embedding computed from previous-iteration logits.

Implementation pointers:
- `src/trainers/stage2_ab_training.py` (`Stage2ABTrainingTrainer.compute_loss`)
  - self-context update loop uses `probs @ coord_embedding_table` to build `exp_emb`
  - CE anchoring: Channel-A uses `logits_a1` for CE (A1 teacher-forced), not the final softctx iteration
  - bbox losses: SmoothL1 + CIoU computed on expectation-decoded coords (`E[k]/999`)

Key risk:
- The model is trained under a *continuous relaxation* of coordinates in context, but eval rollouts are *discrete token* generations. This mismatch can show up exactly as "format looks fine under teacher forcing, but open-loop gets weird".


## Evidence: Current `geo_first` Failure Modes from Monitor Dumps

Artifacts (geo-first, A-only):
- `output/stage2_ab/coco_bbox_max60/a_only/geometry_first_merged_ckpt1832_ep2/v0-20260221-162254`

### A) `wrong_arity` due to non-coord tokens inside bbox list
From:
- `.../monitor_dumps/step_000900.json`

We observed a sample with:
- `parse_dropped_invalid = 64`
- `parse_dropped_invalid_by_reason = {"wrong_arity": 64}`

Inspection showed many `bbox_2d` lists like:
- `"bbox_2d": [<NONCOORD>, <|coord_*|>, <|coord_*|>, <|coord_*|>]`

The specific non-coord token repeated was the Chinese token `\u7b2c\u4e5d`.

This is highly diagnostic:
- It is not dataset noise.
- It is the model "losing control" of the intended sublanguage inside a coord list.
- In `geometry_first` formatting, the first element after `[` is a coord token; inserting any non-coord token immediately produces an invalid object.

### B) Repeating identical objects (runaway list continuation)
From:
- `.../monitor_dumps/step_000600.json`

We observed a sample with:
- `valid_pred_objects = 128`
- `gt_objects = 9`
- `parse_truncated = true` (hit max_new_tokens)
- exact duplicates of `(desc, bbox)` repeated many times, e.g. `('person', (778, 354, 999, 999))` repeated 32 times.

This looks like classic greedy decoding degeneracy: once the model falls into a loop, it keeps emitting another object record until the token budget is exhausted.


## Why This Can Happen (Algorithmic Perspective)

### 1) Exposure bias + long JSON lists
Even with perfect teacher-forced training, open-loop generation is fragile:
- once the model deviates slightly, it enters a prefix state not seen in training
- locally, "emit another object" is always syntactically valid
- greedy decoding will deterministically follow the highest-probability continuation, which can be a repeating attractor

### 2) Continuous softctx is not discrete rollout
The Channel-A self-context update is:
- take coord logits (over 1000 bins)
- softmax -> probabilities
- expected embedding = `sum_k p(k) * E(coord_k)`

But real rollouts feed the model:
- an actual coord token id `coord_k` (discrete)
- an embedding `E(coord_k)`

If the model learns to rely on the "in-between" embedding states, this can harm discrete rollout behavior.

This mismatch is conceptually similar to:
- training with soft targets / continuous relaxations without matching inference
- scheduled sampling vs pure teacher forcing

### 3) Geometry-first is brittle at the coord list boundary
In `geometry_first`, the model enters coord mode immediately:
- `{"bbox_2d": [<coord>, <coord>, <coord>, <coord>], "desc": ...}`

Any non-coord emission inside the list causes arity mismatch or parse failure.
`desc_first` tends to be more forgiving because it anchors the object with stable text tokens before entering coord lists.


## Should We Discretize Softctx Expectation?

### Recommendation: worth trying, but as a controlled ablation (not the main bet)
Discretizing the self-context update directly targets the mismatch:
- training context becomes closer to discrete rollouts

However, it is not free:
- hard discretization introduces non-differentiability
- you must choose an estimator:
  - stop-gradient (treat discrete selection as constant)
  - straight-through estimator
  - Gumbel-softmax / sampling-based estimators

Given current evidence, the primary wins for SOTA behavior will likely come from:
- Stage-1 scaling + strong discrete coord distribution training
- some amount of rollout-based alignment (Channel-B) and/or constrained decoding

Discretized softctx is a medium-risk / medium-reward idea that can be tested after we have a stable baseline.

### Candidate discretization variants
For coord-slot updates in Channel-A, instead of expected embedding:
1. Argmax embedding:
   - `k_hat = argmax_k p(k)`
   - `exp_emb = E(coord_{k_hat})`
   - gradient: stop-grad (simple) or straight-through
2. Rounded expectation embedding:
   - `k_hat = round(E[k])`
   - `exp_emb = E(coord_{k_hat})`
   - gradient: stop-grad or straight-through
3. Top-k sparse mixture:
   - keep top-k coord embeddings weighted by renormalized probs
   - still continuous, but less "blurry"

Success criteria:
- reduces `wrong_arity` and other parse drops
- reduces duplicate object loops (secondary)
- improves rollout F1 / recall without sacrificing format stability


## Can We Apply `bbox_ciou` Loss in Stage-1?

Yes, it is technically feasible.

Stage-1 already has:
- teacher-forced full-vocab logits
- labels containing discrete coord tokens

To compute bbox-level losses in Stage-1:
1. Identify coord-token positions corresponding to each bbox_2d (groups of 4)
2. Extract coord logits at those positions (previous-token logits for next-token prediction)
3. Decode coord values (expectation or argmax)
4. Compute SmoothL1 and/or CIoU against GT box
5. Add to the total loss with a tunable weight

This would make Stage-1's geometry learning more aligned with Stage-2's bbox regression objective, without needing rollouts.

Caveats:
- If computed on expectation decode, multimodal distributions can yield averaged coordinates that are not representative of the discrete mode.
- Using argmax decode for CIoU can better match discrete behavior but is less smooth.
- The weight must be tuned to avoid overpowering the token-level coord distribution losses.


## Practical Bet for SOTA Dense Captioning / Detection

If compute allows any rollouts:
- Prefer a mixed pipeline:
  - Stage-1 scale for discrete coord vocab + text
  - Stage-2 Channel-B rollouts for open-loop alignment (object count, stopping, repetition)
  - optional Channel-A (continuous or discretized) as a regularizer

If rollouts are severely constrained:
- Stage-1 scale + decode constraints (grammar/state machine + termination rules)
- optionally add bbox-level losses (SmoothL1/CIoU) in Stage-1
- treat discretized Channel-A as an experimental proxy


## Proposed Experiments (Minimal, High-Signal)

1. Stage-1 + bbox-level loss ablation
- Add bbox SmoothL1/CIoU to Stage-1 with small weight
- Compare:
  - teacher-forced coord diagnostics
  - offline rollout eval (same decoding config)

2. Channel-A softctx: continuous vs discrete update
- Keep everything else fixed
- Compare:
  - parse drop reasons (`wrong_arity`, `unexpected_keys`)
  - duplication metrics (exact duplicate bbox+desc count)
  - rollout F1/recall

3. Representation choice: `desc_first` vs `geometry_first`
- Hypothesis: `desc_first` has fewer catastrophic coord-list boundary failures


## Open Questions

- Is `wrong_arity` primarily a decoding issue (greedy) or a learned formatting drift?
- How much of the rollout degradation is dominated by a small number of outlier images?
- For polygons, what is the best constrained decoding strategy (token filter vs grammar-based) compatible with vLLM/HF backends?

