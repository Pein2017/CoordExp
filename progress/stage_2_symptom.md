# Stage-2 (AB) symptoms after `b_ratio=0.85`: rollout length growth + stability crash

Last updated: 2026-02-17

This note summarizes **what goes wrong** when Stage-2 AB training runs with **Channel-B dominating** (`stage2_ab.schedule.b_ratio=0.85`) for a few hundred steps, especially when rollouts are allowed to grow long (e.g., `rollout_matching.max_new_tokens=3084`). It also maps symptoms to likely root causes and gives config-first cures with concrete verification steps.

The intent is to help you audit whether the current architecture/algorithm matches the design described in `progress/full_idea.md` and to identify which knobs are causing the visible degeneracy (over-generation, duplication, malformed preds, recall plateau).

---

## 0) Executive summary (what to fix first)

**Observed (P0 symptoms):**
- **Rollouts get longer over training** and frequently hit the decode cap, causing **truncation** and **invalid JSON**. This collapses training signal quality and stability.
- Model **over-generates objects** (predicted object count becomes 2-3x GT) while **recall stays ~0.5** and **precision collapses**.
- Outputs develop **degenerate modes**: repeated labels (e.g., `flagpole` spam), duplicated objects, and malformed coordinate slots (non-`<|coord_k|>` tokens).

**Likely root causes (P0):**
- **Schedule mismatch vs design intent**: the design recommends Channel-A hot / Channel-B cold (e.g., 95% / 5%), but prod configs run **B hot** (`b_ratio=0.85`).
- **No hard stop on object count**: `rollout_matching.repeat_terminate.max_object_keys` is `null` (base config), so decoding has no "max 60 objects" safeguard and relies on token-repeat heuristics that don't trigger for "object spam".
- **Token budget too large in continue**: `ab_mixed_continue.yaml` sets `rollout_matching.max_new_tokens=3084`, which gives the model room to over-generate and still often truncates (because it expands into the budget).

**Highest-leverage cures (config-first):**
1. **Restore A-hot / B-cold schedule** (start with `b_ratio=0.05` per design; ramp only after stability).
2. **Set `rollout_matching.repeat_terminate.max_object_keys: 60`** (since you train on `bbox_max60`).
3. **Bring `max_new_tokens` back down (e.g., 2048 or lower)**, especially until object-count termination is enforced.

These three should directly reduce truncation, invalid parses, and duplicate spam, while preserving Stage-1's already-good geometry.

---

## 1) Context: what Stage-2 AB is supposed to do (design intent)

From `progress/full_idea.md` ("Stage-2: EM-ish Training Loop"), the *recommended* schedule is explicitly:
- Channel-A hot path: **~95% steps**
- Channel-B cold path: **~5% steps**

The rationale stated there is that Stage-1/SFT already gives strong geometry/format stability, and Stage-2's main remaining instability is **self-context generation** (long JSON, permutation, missing/extras/format). Channel-B should target discrete/set-level failures, while Channel-A anchors the model and prevents drift.

This matters because the observed failures look exactly like what you'd expect when the "cold path" dominates the optimization.

---

## 2) Runs and artifacts referenced (evidence sources)

### 2.1 Baseline Stage-2 run (prod)
- Run dir: `output/stage2_ab/prod/ab_mixed/v8-20260214-194010/prod_ab_mixed_bbox_max60_ckpt1516_ep2/`
- Config: `configs/stage2_ab/prod/ab_mixed.yaml`
  - `stage2_ab.schedule.b_ratio: 0.85`
  - `rollout_matching.max_new_tokens: 2048`

### 2.2 Continue Stage-2 run (prod continue from ckpt_400)
- Run dir: `output/stage2_ab/prod/ab_mixed_continue/v1-20260215-170827/prod_ab_mixed_continue_from_ckpt_400/`
- Config: `configs/stage2_ab/prod/ab_mixed_continue.yaml`
  - `stage2_ab.schedule.b_ratio: 0.85`
  - `rollout_matching.max_new_tokens: 3084` (**important difference**)
  - debug dumps: `.../vllm_server_debug/step_*.json`

### 2.3 20-sample inference + visualization (ckpt_continue_800)
- Checkpoint: `output/stage2_ab/experimental/ab_mixed_ckpt_continue_800/`
- Outputs:
  - `output/bench/ab_mixed_ckpt_continue_800-experimental/gt_vs_pred.jsonl`
  - `output/bench/ab_mixed_ckpt_continue_800-experimental/vis/vis_*.png`

---

## 3) Symptoms after a few hundred steps with `b_ratio=0.85`

### 3.1 Metric regression: precision collapses, recall stuck ~0.5, truncation + invalid drops explode

The evaluation logs (from `logging.jsonl`, keys `eval_rollout/*`) show:

| Run | Step | F1 | Precision | Recall | Trunc rate | Dropped invalid | Pred objs | GT objs | mIoU (matched mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (`ab_mixed`) | 100 | 0.441 | 0.391 | 0.506 | 0.0139 | 5 | 5,985 | 4,619 | 0.769 |
| baseline (`ab_mixed`) | 400 | 0.408 | 0.341 | 0.510 | 0.0873 | 186 | 6,918 | 4,619 | 0.769 |
| continue (`ab_mixed_continue`) | 100 | 0.323 | 0.229 | 0.551 | 0.1806 | 677 | 11,129 | 4,619 | 0.766 |
| continue (`ab_mixed_continue`) | 800 | 0.296 | 0.210 | 0.501 | 0.3730 | 4,067 | 11,012 | 4,619 | 0.774 |

Key interpretation:
- **Recall stays around ~0.5** across both runs and does not climb with more steps/rollouts.
- **Precision collapses** in the continue run (0.39 -> 0.21 range), while predicted object count roughly **doubles**.
- **Truncation and invalid drops are the clearest "stability crash" signals**:
  - baseline: truncation remains low (<0.09), invalid drops stay small.
  - continue: truncation grows to **0.37** and invalid drops reach **4,067** (in a single eval sweep).
- **mIoU stays ~0.77** even when everything else degrades:
  - geometry is not the primary failure; it's **set-level correctness + format + over-generation**.

### 3.2 Rollout length growth: token budget saturation becomes common

On training Channel-B steps, rollout token statistics show that the continue run frequently hits the max token cap.

From the continue run training logs (`rollout/gen_new_tokens_*`):
- At step 100: `gen_new_tokens_p90 = 3084` and truncation rate ~0.286.
- At step 800: `gen_new_tokens_mean = 1351.5`, `gen_new_tokens_p90 = 3084`, truncation rate ~0.393.
- At step 800: `rollout/repeat_terminate_triggered_sequences = 0.0` (repeat terminate active, but not stopping the sequences).

By contrast, the baseline run's training rollouts remain much shorter:
- step 100 baseline: `gen_new_tokens_mean ~318`, `p90 ~711`, truncation ~0.0.

This is the rollout-length instability pattern you described: **the model learns to fill the available generation budget**, and then the system starts training on more truncated / invalid rollouts.

### 3.3 Over-generation: "more rollouts" does not translate to "more recall"

The continue run's eval predicted object count is consistently about:
- `pred_objects ~ 10k-11k` vs `gt_objects = 4,619` (ratio ~2.2-2.4x)

But recall stays ~0.5. This indicates that extra generation is mostly:
- duplicates,
- false positives,
- malformed objects that get dropped before matching,
- or plausible-but-unlabeled objects (dataset incompleteness can exist, but the duplication/malformed patterns are not explainable by incompleteness alone).

### 3.4 Malformed outputs: invalid JSON + non-coordinate tokens in bbox slots

#### 3.4.1 vLLM debug dumps: rollout JSON frequently cannot be parsed

From `output/stage2_ab/prod/ab_mixed_continue/.../vllm_server_debug/step_*.json` (4 files x 4 samples = 16 rollouts):
- Valid GT JSON: 16 / 16
- Valid rollout JSON: 10 / 16 (**62.5%**)
- The failures are JSON decoding errors like:
  - `Unterminated string ...` (strongly consistent with truncation mid-string)
  - `Expecting ',' delimiter ...` (malformed structure / missing braces)

So even before set-matching, a large fraction of generated rollouts are **not even valid JSON**.

#### 3.4.2 20-sample inference from ckpt_continue_800: coordinate-slot "contamination"

From `output/bench/ab_mixed_ckpt_continue_800-experimental/gt_vs_pred.jsonl` (20 samples):
- `raw_pred_total = 362` objects (from `raw_output_json`)
- `valid_pred_total = 269` objects (after strict parsing)
- Estimated dropped objects: `93` (**25.7%** drop rate)
- Worst duplication: a single sample emits **90** duplicates of `flagpole`:
  - image: `images/train2017/000000000165.jpg`

More importantly: many raw bbox slots contain tokens that are not `<|coord_k|>`.

Counts of invalid bbox-slot tokens across these 20 samples:
- invalid bbox-slot tokens total: 275
- plain digits (e.g., `"0"`, `"720"`): 95
- non-ASCII tokens in bbox slots: 178
  - most frequent token: `\"\\u5c3c\\u65af\"` appears 172 times (this corresponds to a CJK token observed literally in the raw outputs)
- leading-space tokens: 1 (e.g., `" Country"`)

This is a concrete mechanism behind "malformed preds": even when the model stays inside a JSON-like format, it is not respecting the **coord-token vocabulary contract**.

### 3.5 "Rough glance" / image-prior guessing (semantic drift)

Qualitatively, both vLLM debug dumps and the 20-sample inference show:
- plausible but "default" objects injected (e.g., `billboard`, `streetlight`, `tray`, `book`) at high frequency,
- repeated labels with near-identical boxes,
- weak alignment to GT label set once the output starts drifting.

This is consistent with a model that is being optimized on a regime where:
- producing additional objects is not strongly penalized,
- and the training signal for *which* objects to produce is weakened by over-dominant rollout-based updates.

---

## 4) Root causes (with evidence and mechanism)

### 4.1 Root cause A (P0): schedule mismatch vs the intended AB mixture

Evidence:
- Design intent (`progress/full_idea.md`): Channel-A hot / Channel-B cold, e.g., 95% / 5%.
- Config reality:
  - `configs/stage2_ab/base.yaml`: `stage2_ab.schedule.b_ratio: 0.05`
  - `configs/stage2_ab/prod/ab_mixed.yaml`: overrides to `b_ratio: 0.85`
  - `configs/stage2_ab/prod/ab_mixed_continue.yaml`: also uses `b_ratio: 0.85`

Mechanism:
- With `b_ratio=0.85`, you are effectively optimizing the model primarily on **self-generated contexts**.
- If Channel-B's loss is even slightly permissive to false positives or format drift, the model will exploit that freedom.
- Channel-A's stabilizing effect (Stage-1-like teacher-forcing + soft-context geometry anchor) becomes too weak to prevent drift.

The observed pattern (precision collapse + duplication + malformed) is exactly what "B-hot" should produce if Channel-B isn't carefully FP-penalized and format-regularized.

### 4.2 Root cause B (P0): decode budget too large + no object-count termination = truncation/invalid feedback loop

Evidence:
- Continue config sets `rollout_matching.max_new_tokens: 3084` (vs baseline 2048).
- `configs/stage2_ab/base.yaml` keeps `rollout_matching.repeat_terminate.max_object_keys: null`.
- Continue training logs at step 800:
  - `rollout/gen_new_tokens_p90 = 3084` (budget saturation)
  - `rollout/parse_truncated_rate ~ 0.393`
  - `rollout/repeat_terminate_triggered_sequences = 0.0`
- vLLM debug dumps show rollout JSON parsing failures consistent with truncation mid-string.

Mechanism:
1. Model begins to over-generate objects (for any reason: permissive loss, drift).
2. Longer outputs become common -> JSON strings get truncated or malformed.
3. Truncated/malformed rollouts are either dropped or weakly supervised.
4. The model gets even less "format discipline" signal -> generates more malformed outputs.

This is a classic self-training failure mode: **once invalid rollouts become common, they poison the training distribution**.

### 4.3 Root cause C (P0/P1): "strict parsing + dropping" makes invalid outputs cheap

Evidence:
- Eval metrics show `eval_rollout/parse_dropped_invalid` rising from 5 -> 4,067.
- Channel-B training logs show strict-drop counts and reasons (e.g., `non_coord_token`, `key_invalid`).
- In 20-sample inference, ~25.7% of raw objects are dropped by strict parsing.

Mechanism:
- If invalid objects are dropped downstream, they may:
  - not contribute to loss, or
  - contribute less than valid objects,
  - or be excluded from matching/supervision.
- The model can "escape" penalties for over-generation by drifting into malformed outputs (which are then ignored).

This makes format correctness and coord-token correctness a *first-class training objective*, not just a post-processing detail.

### 4.4 Root cause D (P1): set-level objective mismatch encourages duplication

Even if a prediction is "semantically plausible", duplicates are not useful:
- duplicates don't increase recall,
- duplicates do increase FP count and hurt precision,
- duplicates inflate sequence length and increase truncation probability.

If the algorithm does not explicitly penalize duplicates (or strictly cap object count), the easiest local optimum is "spam plausible objects".

### 4.5 What seems *not* to be the main issue: geometry itself

Evidence:
- `eval_rollout/matched_maskiou_mean` stays ~0.77 across runs, even in the unstable regime.
- Many parsed predictions have reasonable boxes, but the model produces too many extras and too many invalid objects.

Interpretation:
- Stage-1 did its job for **coord decoding**.
- Stage-2 AB failures are dominated by **generation control + format discipline + set cardinality**.

---

## 5) Recommendations (cures) and why they should help

### 5.1 P0 cure: restore the intended AB schedule (A hot, B cold)

**Recommendation**
- Change `stage2_ab.schedule.b_ratio` back toward the design default (start with 0.05).
- If you need more Channel-B later, ramp slowly (e.g., 0.05 -> 0.10 -> 0.20), only after truncation/invalid are under control.

**Why it helps**
- Channel-A provides a stabilizing teacher-forced anchor that preserves Stage-1 behavior.
- Channel-B can then be used as a targeted "set-level fix" rather than the dominant training distribution.

**How to verify**
- Within the first 100-200 steps:
  - `eval_rollout/precision` should stop collapsing.
  - `eval_rollout/parse_truncated_rate` should decrease (or at least stop increasing).
  - `eval_rollout/parse_dropped_invalid` should fall sharply.

### 5.2 P0 cure: enforce a hard object-count stop (`max_object_keys: 60`)

**Recommendation (config-first)**
Set (or override) the repeat terminate object cap:

```yaml
rollout_matching:
  repeat_terminate:
    enabled: true
    max_object_keys: 60
```

**Why it helps**
- You train on `bbox_max60`, so decoding beyond 60 objects is always waste and almost always harmful.
- A hard cap prevents:
  - runaway duplication modes,
  - length-induced truncation,
  - malformed JSON from exhausting `max_new_tokens`.

**How to verify**
- Training logs:
  - `rollout/gen_new_tokens_p90` drops (should stop pegging at `max_new_tokens`).
  - `rollout/parse_truncated_rate` drops.
  - `rollout/repeat_terminate_triggered_sequences` becomes > 0 at least occasionally (if the model tries to exceed the cap).
- vLLM debug dumps:
  - rollout JSON should parse at >95% (ideally 100%).
  - number of `object_*` keys should be <= 60.

### 5.3 P0 cure: lower `max_new_tokens` (especially in continue runs)

**Recommendation**
- Align the continue run with baseline's safer budget (e.g., 2048), or even lower while debugging (e.g., 1536).

```yaml
rollout_matching:
  max_new_tokens: 2048
```

**Why it helps**
- Large token budgets invite the model to "use the space" once it starts over-generating.
- Lower budgets reduce truncation length, reduce time per rollout, and reduce the chance of invalid JSON mid-string.

**How to verify**
- `eval_rollout/parse_truncated_rate` should decrease.
- vLLM debug dumps should show fewer `Unterminated string` errors.

### 5.4 P1 cure: make invalid/malformed rollouts expensive (don't let the model escape)

**Recommendation**
- Ensure invalid parse cases contribute an explicit training signal:
  - either by upweighting structure CE on invalid instances,
  - or by adding a dedicated "format correctness" penalty,
  - or by not dropping invalids silently.

**Why it helps**
- Current evidence shows large invalid-drop volumes (thousands) and significant coord-slot contamination.
- If invalid outputs are cheap, the model will keep producing them as an escape hatch.

**How to verify**
- `eval_rollout/parse_dropped_invalid` drops across steps instead of rising.
- In inference dumps, invalid bbox-slot tokens (digits / non-ASCII) drop to near-zero.

### 5.5 P1 cure: add explicit anti-duplication / cardinality control beyond token repeats

**Recommendation**
- Prefer controls that target *object-level repetition*, not token-level repetition:
  - object-key cap (`max_object_keys`) is the first and easiest.
  - optionally add "stop if the next object is a near-duplicate of a previous one" (bbox IoU + same desc).

**Why it helps**
- Token n-gram repetition checks often won't trigger when the model changes just a few coord tokens.
- The failure mode here is "repeat the same semantic object with slightly changed coords", which is object-level duplication.

**How to verify**
- In `gt_vs_pred.jsonl`, the max duplicate count per sample should fall dramatically (no more 20-90 repeats).

### 5.6 P2 cure: if stability still crashes, reduce Channel-A softctx coupling

**Recommendation**
- Use the fallback described in `progress/full_idea.md`:
  - set soft self-context grad mode to a detached EM-ish variant (e.g., `softctx_grad_mode: em_detach`) if supported by your trainer.

**Why it helps**
- If unrolled softctx creates unstable cross-object credit assignment, detaching reduces feedback coupling.
  - Note: based on current evidence, the *dominant* instability is B-hot + long rollouts; do not skip P0 fixes.

**How to verify**
- Loss curve becomes smoother (fewer spikes), and truncation/invalid trends stop worsening.

---

## 6) Verification checklist (what to watch to confirm the cure works)

### 6.1 Training-time health signals (log keys)
- `rollout/gen_new_tokens_p90` (should not peg at `max_new_tokens`)
- `rollout/parse_truncated_rate` (should be low, ideally < 0.05)
- `stage2_ab/channel_b/strict_drop/N_drop_invalid` (should fall)
- `rollout/repeat_terminate_triggered_sequences` (should become >0 when model tries to exceed caps)

### 6.2 Eval-time quality signals
- `eval_rollout/precision` (should improve as over-generation is controlled)
- `eval_rollout/recall` (may improve only after the model stops wasting budget on duplicates/invalids)
- `eval_rollout/parse_dropped_invalid` (should drop by an order of magnitude)
- `eval_rollout/parse_truncated_rate` (should decrease, not increase with step)
- `eval_rollout/pred_objects / eval_rollout/gt_objects` ratio (should move toward 1x)

### 6.3 Artifact sanity checks
- vLLM debug dumps: `rollout_text` should parse as JSON nearly always.
- `output/bench/.../gt_vs_pred.jsonl`:
  - dropped_est / raw_pred_total should be small
  - invalid bbox-slot tokens should be near 0
  - max duplicate label count per sample should be small

---

## 7) Suggested ablation plan (fast, reproducible)

Keep everything identical to prod, only override a few keys in a derived YAML:

1. **Fix termination only**
   - `repeat_terminate.max_object_keys: 60`
   - keep `b_ratio: 0.85`
   - keep `max_new_tokens: 3084`
   - Goal: verify truncation + invalid JSON improve immediately.

2. **Fix schedule only**
   - `b_ratio: 0.05`
   - keep other knobs as-is
   - Goal: verify precision/duplication improves even without changing decode budget.

3. **Fix both (recommended)**
   - `b_ratio: 0.05`
   - `repeat_terminate.max_object_keys: 60`
   - `max_new_tokens: 2048`

Run each for ~200 steps with the same eval interval and compare:
- truncation trend,
- invalid-drop trend,
- pred/gt ratio,
- and qualitative duplication in a 20-sample rollout visualization.

---

## 8) Final take: is the model "learning well"?

There is one piece of good news:
- **Matched mIoU stays stable (~0.77)** even in the unstable regime, so Stage-1's geometry and coord-token decoding aren't fundamentally broken.

But overall, under `b_ratio=0.85` + long rollouts:
- the system is not converging toward "higher recall with controlled precision";
- instead it is converging toward a degenerate local optimum: **spam plausible objects, drift in coord-token validity, and rely on truncation/drop behavior**.

That points to **algorithmic control (schedule + termination + invalid penalties)** as the primary fix, not a model-architecture change.
