# Raw-Text Coordinate Mechanism Resume Handoff

## State

- Worktree: `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism`
- Branch: `codex/raw-text-coordinate-mechanism`
- Scope intentionally paused after finishing the in-flight tasks only.
- No new jobs should be assumed to still be running; verify with `ps` / `nvidia-smi` after reconnect.

## Completed In This Session

### 1. Good-basin pilot finished cleanly

- Bundle: `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421`
- Key files:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/summary.json`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/report.md`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/pilot/summary.json`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/pilot/per_coord_scores.jsonl`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/pilot/token_length_control_summary.json`

### 2. Duplicate-burst onset probe rerun on two explicit surfaces

- Model-native surface:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-model-native-20260421/summary.json`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-model-native-20260421/per_coord_scores.jsonl`
- Pretty-inline surface:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-pretty-inline-20260421/summary.json`
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-pretty-inline-20260421/per_coord_scores.jsonl`
- Surface comparison artifact:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-surface-comparison-20260421.json`

### 3. Review surface remains available

- Review gallery:
  - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism/review_gallery/review.html`

## Strongest Current Findings

### Good basin exists, but summed score is strongly token-length biased

- Raw pilot says local continuity is real under `pretty_inline`:
  - base `mass@4`: `x1=0.684`, `y1=0.728`
  - pure_ce pilot is also complete in the same bundle
- But the raw argmax is badly confounded by token count:
  - in `token_length_control_summary.json`, best-by-sum is 1-token on `123/128` base probes and `126/128` pure_ce probes
  - best-by-sum and best-by-mean disagree on `85/128` base probes and `86/128` pure_ce probes
- After holding token count fixed to the GT token span, continuity survives strongly:
  - base `same_gt_token_mass_at_4_mean = 0.8774`
  - pure_ce `same_gt_token_mass_at_4_mean = 0.8907`
  - base `same_gt_token_best_is_gt_rate = 0.8281`
  - pure_ce `same_gt_token_best_is_gt_rate = 0.8203`

Interpretation:

- The local basin signal is not just noise.
- But the current summed-logprob pilot should not be treated as mechanism-grade without explicit length control.

### Duplicate-burst onset bad basin is real, and previous-anchor metrics are now explicit

- Both duplicate-burst runs completed with:
  - `candidate_rows_total = 767`
  - `num_probes = 26`
- The rerun now records `previous` center metrics in addition to `pred` and `gt`.
- Example aggregate pattern from the summaries:
  - base-only `x1` onset probe:
    - model-native `pred_center_mass_at_4 = 0.6876`
    - model-native `previous_center_mass_at_4 = 0.0945`
    - model-native `gt_center_mass_at_4 = 0.1378`
  - base-only `y1` onset probe:
    - model-native `pred_center_mass_at_4 = 0.7863`
    - model-native `previous_center_mass_at_4 = 0.3669`
    - model-native `gt_center_mass_at_4 = 0.0915`
  - base+adapter surface deltas are nearly zero between `model_native` and `pretty_inline` in the current aggregate comparison.

Interpretation:

- The onset-object bad basin is strong.
- The source/previous anchor is now measurable rather than hidden.
- Serializer sensitivity is modest on the base model and negligible in the current adapter onset aggregate.

## Important Methodological Fixes Landed

- `model_native` is no longer a fake label in the duplicate-burst runner.
  - The rerun reconstructs native prefix style from sibling `pred_token_trace.jsonl`.
- The scorer now supports wrapped native JSON for span replacement.
  - Fenced JSON blocks are handled in `raw_text_coord_continuity_scoring.py`.
- Bad-basin summaries now include `previous` center metrics and `previous_minus_gt_mass_at_4`.

## Returned Subagent Guidance

### FN mechanism next step

Best immediate experiment:

- Start from the already-mined `suppressed_fn` cases in:
  - `/data/CoordExp/output/analysis/coord-family-recall-rawtext-val64/summary.json`
- There are `2` promising `suppressed_fn` cases.
- For each case, run a direct teacher-forced chunk comparison:
  - `EOS now`
  - versus `continue with missed GT object, then close`

Reason:

- This is the fastest reusable path to test “perceptually supported but decode-suppressed” without building a new FN framework.

Main risk:

- Do not over-call EOS suppression unless a quick visual audit rules out unlabeled-positive / evaluator ambiguity.

### Pre-burst causal mechanism next step

Best minimal next probe:

- Use existing onset-mined duplicate cases, but stop at the pre-onset prefix:
  - `prefix_objects = preds[:object_idx]`
- Compute pairwise margin before the duplicate object is emitted:
  - `M_base = score(gt_next | preburst_prefix, image) - score(exact_duplicate | preburst_prefix, image)`
- Repeat under a geometry-specific prefix intervention:
  - preferred: `source_x1y1_from_gt_next`
  - optional sanity check: `drop_source`
- Use the margin delta as the causal readout.

Reason:

- This directly tests earlier `preburst_anchor_collapse`, not only onset-local attraction after the duplicate object already exists.

## Code Changes In Worktree

- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/src/analysis/raw_text_coordinate_exploratory.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/src/analysis/raw_text_coord_continuity_scoring.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/src/analysis/raw_text_coord_continuity_report.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/src/analysis/raw_text_coord_continuity_probe.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/scripts/analysis/run_raw_text_coordinate_duplicate_burst_probe.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/configs/analysis/raw_text_coordinate_mechanism/duplicate_burst_probe.yaml`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/configs/analysis/raw_text_coordinate_mechanism/duplicate_burst_probe_pretty_inline.yaml`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/tests/test_raw_text_coord_continuity_scoring.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/tests/test_raw_text_coord_continuity_probe.py`
- `/data/CoordExp/.worktrees/raw-text-coordinate-mechanism/tests/test_raw_text_coordinate_exploratory.py`

## Verification Already Done

- Focused regression suite:
  - `rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_scoring.py tests/test_raw_text_coordinate_exploratory.py tests/test_raw_text_coord_continuity_probe.py -q`
  - result: `54 passed`
- Syntax check:
  - `rtk conda run -n ms python -m py_compile src/analysis/raw_text_coordinate_exploratory.py src/analysis/raw_text_coord_continuity_scoring.py src/analysis/raw_text_coord_continuity_report.py src/analysis/raw_text_coord_continuity_probe.py scripts/analysis/run_raw_text_coordinate_duplicate_burst_probe.py`

## Recommended Restart Order

1. Re-check there are no stale workers still alive.
2. Read this handoff plus:
   - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/pilot/token_length_control_summary.json`
   - `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-surface-comparison-20260421.json`
3. Decide whether to implement next:
   - the FN `EOS now` vs `continue with GT object` chunk probe
   - or the pre-onset pairwise margin probe with `source_x1y1_from_gt_next`
4. Only after that, consider new heatmaps or larger reruns.
