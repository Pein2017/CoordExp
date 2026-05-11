thread_id: 019d86c9-0fbe-7672-baab-24e7f5a201c1
updated_at: 2026-04-18T10:21:09+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/13/rollout-2026-04-13T12-20-16-019d86c9-0fbe-7672-baab-24e7f5a201c1.jsonl
cwd: /data/CoordExp
git_branch: codex/raw-text-continuity-probe

# Implemented and validated a V2 agent-research control plane on top of CoordExp’s runtime layer

Rollout context: the user wanted the repo to evolve toward an agent-friendly research workflow where natural-language intent becomes executable training/inference/eval work. In this rollout, the main concrete implementation work happened in a worktree at `/data/CoordExp/.worktrees/agent-research-runtime`, where a new V2 runtime/control-plane was patched and then regression-tested. The surrounding repo already had strong docs/specs for training, inference, evaluation, artifacts, and OpenSpec-driven change control; this rollout focused on making the runtime more faithful to those contracts and less ambiguous for future agent-led execution.

## Task 1: V2 runtime/control-plane hardening

Outcome: success

Preference signals:

- The user’s original framing strongly emphasized that future work should be driven by natural language and should support a “research command + agent execution” style. That implies future agents should proactively preserve execution provenance, decision traces, and replayable artifacts rather than only optimizing for one-off script success.

Key steps:

- Patched the V2 runtime in `src/runtime/` to tighten the control plane around `mission`, `run`, `preflight`, `review`, and adapter execution.
- Added run identity and provenance improvements: `mission_schema_version`, `resolved_recipe_hash`, `run_path` ref, `summary_ref`, `preflight_report` ref, and richer `review.evidence` backlinks.
- Added stricter rerun-closure checks so the runtime no longer reports exact rerun possible unless recipe snapshot, dataset refs, seed bundle, workspace, and key input paths are present and valid.
- Added explicit stage dependency hydration so `upstream_stage_refs` are populated on stage result records.
- Changed the training adapter flow so a trained `selected_checkpoint` always becomes the checkpoint handed to downstream infer/eval, instead of being silently shadowed by the original `model_checkpoint`.
- Added a training-config overlay that injects mission `train_jsonl` / `val_jsonl` into the legacy training override, so the legacy training surface stays aligned with the mission inputs.
- Tightened scale-promotion logic so a smoke run only promotes to scale when it is truly accepted: `review.status == completed` and `scale_readiness.status == ready`.
- Tightened baseline selection so only `review.status == completed` runs are eligible for baseline comparison.

Failures and how to do differently:

- The first test pass failed on a few mechanical issues: missing `run_record_path` import in `src/runtime/run.py`, a test that wrote a second `run.json` before creating its parent directory, and one timing issue where `summary_ref` / `preflight_report_ref` were populated too late for review evidence.
- The fix pattern was: patch the runtime first, then immediately rerun the focused V2 tests, then fix import/timing/test setup issues in-place until green.
- The rollout showed that when adding runtime evidence fields, the order of assignment matters: fields that feed `review.evidence` must be present before `derive_run_review()` is called.

Reusable knowledge:

- In the V2 runtime, `review.evidence` is a good place to keep durable backlinks: `stage_result_refs`, `summary_ref`, `preflight_report_ref`, `metrics_json_ref`, `eval_dir_ref`, and `vis_dir_ref` are now surfaced there.
- Scale promotion should be treated as a gated promotion from accepted smoke evidence, not as a generic “next action == scale” shortcut.
- If a training adapter can emit a new best checkpoint, downstream inference should consume that selected checkpoint explicitly rather than the pre-training checkpoint input.
- Baseline discovery should ignore blocked or incomplete runs; only completed reviewed runs are safe candidates for comparison.

References:

- [1] `src/runtime/run.py` — added `resolved_recipe_hash`, `run_path`, `summary_ref`, `upstream_stage_refs`, adapter output backfill, and stricter rerun-closure validation.
- [2] `src/runtime/mission.py` — added `_supports_scale_promotion()`, prevented invalid scale promotion, and recorded richer review-event refs.
- [3] `src/runtime/preflight.py` — added training dataset alignment checks and a stronger scale-gate check tied to prior accepted smoke evidence.
- [4] `src/runtime/review.py` — baseline selection now requires `review.status == completed`; review evidence now includes more concrete artifact refs.
- [5] `src/runtime/adapters.py` — training override now injects mission `train_jsonl` / `val_jsonl` into the legacy training surface.
- [6] `tests/test_v2_runtime_control_plane.py`, `tests/test_v2_runtime_adapters.py` — new regressions for invalid scale promotion, checkpoint handoff, baseline selection, and evidence backlinks.
- [7] Validation commands that passed:
  - `rtk conda run -n ms python -m pytest tests/test_v2_runtime_control_plane.py tests/test_v2_runtime_adapters.py tests/test_v2_core_contracts.py` → `27 passed`
  - `rtk conda run -n ms python -m pytest tests/test_unified_infer_pipeline.py tests/test_detection_eval_output_parity.py tests/test_gt_vs_pred_visualization.py tests/test_run_manifest_files.py tests/test_run_metadata_file.py tests/test_experiment_manifest_file.py` → `68 passed`
  - `openspec validate refactor-codebase-for-agent-researcher` → valid

## Task 2: OpenSpec / change validation and repo status check

Outcome: success

Preference signals:

- The user asked for a system that the agent can evolve autonomously; that makes spec-driven changes and validation especially important for future runs, because the agent should not rely on undocumented behavior.

Key steps:

- Verified the OpenSpec change `refactor-codebase-for-agent-researcher` was fully complete via `openspec instructions apply --change refactor-codebase-for-agent-researcher --json`.
- Verified the change was valid with `openspec validate refactor-codebase-for-agent-researcher`.
- Checked the worktree status and noticed the change set was broader than the runtime files alone, including pre-existing or side-effect edits in `.codex/skills/`, `AGENTS.md`, `src/__init__.py`, and untracked `openspec/changes/refactor-codebase-for-agent-researcher/`, `src/core/`, `src/runtime/`, and the new V2 tests.

Failures and how to do differently:

- No functional failure here, but the worktree status check showed that future agents should be careful not to assume a narrow diff scope when working in this branch/worktree; there may be unrelated deletions/modifications already present.

Reusable knowledge:

- The current OpenSpec change was already fully executed (`31/31 complete`), so future work in this branch should treat the spec as the current contract baseline rather than reopening the proposal phase.
- When validating this area, it is useful to run both the V2 runtime regressions and the legacy infer/eval artifact tests, because the runtime changes can affect artifact/backlink timing even when the primary control-plane tests are green.

References:

- `openspec instructions apply --change refactor-codebase-for-agent-researcher --json` → `state: all_done`, `progress: 31/31 complete`
- `openspec validate refactor-codebase-for-agent-researcher` → valid
- `rtk git status --short` showed outstanding edits/untracked items in the worktree, including `.codex/skills/graphify/...`, `AGENTS.md`, `src/__init__.py`, `src/runtime/`, and new V2 tests.
