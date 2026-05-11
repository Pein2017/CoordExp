thread_id: 019d86be-74f9-7443-8c25-abcc589d913a
updated_at: 2026-04-13T12:48:31+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/13/rollout-2026-04-13T12-08-41-019d86be-74f9-7443-8c25-abcc589d913a.jsonl
cwd: /data/CoordExp
git_branch: main

# Added first-class experiment metadata and unified run manifest support, then documented/verified/pushed the change

Rollout context: The user wanted to stop encoding rich experiment intent inside `run_name` / `run_dir`, and instead track run purpose, hypothesis, deviations, runtime settings, and comments in a dedicated structured-but-human-readable block. The work evolved into a cross-cutting contract change with an OpenSpec change, new artifact, schema updates, docs, tests, a representative Stage-1 config, then a follow-up question about whether the new Stage-1 profile was safe to launch as a production run, and finally a request to commit and push the local changes.

## Task 1: Design and implement structured experiment metadata

Outcome: success

Preference signals:

- The user said they wanted to stop overloading `run_name` / `run_dir` with hyperparameter and ablation details, and instead have each run carry natural-language descriptions of purpose, hypothesis, baseline deviations, runtime settings, and comments. This indicates a durable preference for explicit authored run context rather than semantic path parsing.
- The user then clarified: "You may refactor and change the artifacts. Unify and design an optimal and scalable way for both hard runtime information and soft natural language information." This suggests they prefer a first-class artifact/model split, not a thin compatibility shim.
- In the follow-up Stage-1 experiment request, the user explicitly framed the run as an exploratory attempt for a new bbox parameterization feature, with a hypothesis about duplication collapse and early coordinate decoding. That implies future similar runs should store the experimental motivation and hypothesis in the config itself instead of only in ad hoc notes.

Key steps:

- Inspected the existing training/runtime artifact path: `resolved_config.json`, `effective_runtime.json`, `pipeline_manifest.json`, and `run_metadata.json` were already distinct.
- Confirmed strict config parsing: unknown keys fail fast, and `custom.extra` is the only intentional extension bucket today.
- Created OpenSpec change `add-structured-experiment-metadata` with proposal, design, spec, and tasks.
- Added a typed top-level `experiment` section in `src/config/schema.py` with strict validation and serialization.
- Added `src/bootstrap/experiment_manifest.py` to write `experiment_manifest.json` as a run-level overview artifact.
- Refactored `src/bootstrap/run_metadata.py` to expose a reusable payload builder while keeping `run_metadata.json` as the low-level provenance sidecar.
- Wired `src/sft.py` to emit both `run_metadata.json` and `experiment_manifest.json` during rank-0 bootstrap.
- Updated `src/analysis/unmatched_proposal_verifier.py` to recognize `experiment_manifest.json` as a candidate metadata source before falling back.
- Updated docs (`docs/ARTIFACTS.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/IMPLEMENTATION_MAP.md`) to explain the split.
- Added targeted tests for schema parsing, manifest writing, and metadata consumer fallback.

Failures and how to do differently:

- A YAML quoting issue appeared twice during parsing checks because colon-containing prose in list items was being interpreted as mappings. The fix was to quote the affected strings in the `experiment` block. Future authored experiment prose that includes `:` should be quoted by default.
- One broad test slice surfaced an unrelated workspace-dependent checkpoint-path failure in `tests/test_unmatched_proposal_verifier.py::test_resolve_checkpoint_path_labels_common_root`. That was not caused by the metadata change and should be kept isolated unless the task is specifically about checkpoint discovery.
- A graph rebuild was attempted twice and timed out on this repo. The code and targeted tests were still completed, but future similar runs may need a longer rebuild budget or a lighter graphify mode.

Reusable knowledge:

- The current repo contract is now split into roles that should stay distinct:
  - `resolved_config.json` = exact authored config authority
  - `effective_runtime.json` = executed runtime authority
  - `pipeline_manifest.json` = pipeline structure authority
  - `run_metadata.json` = low-level provenance and launcher/cache metadata
  - `experiment_manifest.json` = primary human-facing run summary combining authored experiment context, runtime summary, provenance summary, and artifact pointers
- For this repo, `scripts/train.sh` is env-var driven; the intended invocation form is `config=... gpus=... bash scripts/train.sh`, not positional arguments.
- The Stage-1 `center_log_size` path is explicitly documented as a narrow V1 experiment path, not a stable baseline recipe.
- The new `experiment` section should be treated as authored narrative / intent, not as the source of exact machine truth; exact values still belong in the existing authoritative artifacts.

References:

- [1] OpenSpec artifacts: `openspec/changes/add-structured-experiment-metadata/{proposal.md,design.md,specs/experiment-metadata/spec.md,tasks.md}`
- [2] New schema and artifact code: `src/config/schema.py`, `src/bootstrap/experiment_manifest.py`, `src/bootstrap/run_metadata.py`, `src/sft.py`
- [3] Updated consumer fallback: `src/analysis/unmatched_proposal_verifier.py` now checks `experiment_manifest.json` before `run_metadata.json`
- [4] Representative authored config: `configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml`
- [5] Tests added/updated: `tests/test_experiment_manifest_file.py`, `tests/test_training_config_strict_unknown_keys.py`, `tests/test_stage1_static_packing_runtime_config.py`, `tests/test_unmatched_proposal_verifier.py`
- [6] Focused verification passed:
  - `conda run -n ms python -m pytest -q tests/test_experiment_manifest_file.py tests/test_run_metadata_file.py tests/test_run_manifest_files.py tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py -k stage2_center_size_smoke_config_resolves_bbox_geo_parameterization`
  - `conda run -n ms python -m pytest -q tests/test_unmatched_proposal_verifier.py -k experiment_manifest_pointer`

## Task 2: Advise whether the new Stage-1 profile can be run as a production job

Outcome: success

Preference signals:

- The user asked directly whether they could run `config=configs/stage1/profiles/2b/center_log_size_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml gpus=all bash scripts/train.sh` as a production run next. That suggests they want a direct command-level recommendation, not a long theoretical answer.

Key steps:

- Checked `scripts/train.sh` and confirmed it is env-var driven, with a hard error on positional args.
- Verified the Stage-1 profile parses cleanly after the new `experiment` block was added.
- Confirmed docs for Stage-1 `center_log_size` frame it as a narrow V1 experiment that keeps external artifacts canonical `xyxy` and does not support confidence post-op.
- Answered with the correct launcher form and clarified that `gpus=all` expands to `0,1,2,3,4,5,6,7`.

Failures and how to do differently:

- The initial command the user suggested used positional arguments (`... yaml gpus=all bash scripts/train.sh`) instead of env vars. Future similar advice should explicitly restate the env-var form to avoid accidental launcher misuse.
- The profile is valid, but the docs make clear it should be treated as an exploratory `center_log_size` experiment, not a canonical baseline. Future agents should distinguish “launchable” from “production-baseline stable.”

Reusable knowledge:

- `scripts/train.sh` resolves config paths from the `config` env var and validates JSONL contracts before launching.
- `gpus=all` is accepted by the script and expands to all 8 visible GPU slots (`0..7`) on this launcher.
- `center_log_size` Stage-1 runs are compatible with the repo’s canonical downstream evaluation artifacts, but confidence post-op is unsupported for this path.

References:

- [1] `scripts/train.sh` usage: env-var only, no positional args
- [2] `docs/training/STAGE1_OBJECTIVE.md` section `Stage-1 center_log_size V1 experiment`
- [3] `configs/stage1/profiles/2b/center_log_size_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml`

## Task 3: Commit and push local changes

Outcome: success

Preference signals:

- The user asked to "commit the local changes and push to remote," which indicates they expect the agent to handle git hygiene and remote publication once the implementation is ready.

Key steps:

- Checked branch/remote state and confirmed the repo was on `main` with `origin` configured.
- Staged only the relevant modified files, including the new OpenSpec change artifacts and the new experiment-manifest implementation/tests.
- Created one commit: `5b1b9b0` with message `Add structured experiment metadata manifests`.
- Pushed `main` to `origin` successfully.

Failures and how to do differently:

- An attempt to read a missing `.codex/skills/git-hygiene/SKILL.md` file failed; that was non-blocking and should not be repeated unless the repo actually has that skill file.

Reusable knowledge:

- The commit was cleanly scoped to the experiment metadata work; the final worktree was clean after the commit.
- Remote push target was `git@github.com:Pein2017/CoordExp.git` on branch `main`.

References:

- [1] Commit: `5b1b9b0` — `Add structured experiment metadata manifests`
- [2] Push: `origin/main` updated from `78185ca` to `5b1b9b0`
- [3] Final staged scope included `src/bootstrap/experiment_manifest.py`, schema/runtime changes, docs, tests, OpenSpec change artifacts, and the Stage-1 experiment config update

