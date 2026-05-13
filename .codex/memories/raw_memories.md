# Raw Memories

Merged stage-1 raw memories (stable ascending thread-id order):

## Thread `019dd46a-66dd-7321-8e69-211fff081375`
updated_at: 2026-04-28T14:10:00+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-28T14-07-16-019dd46a-66dd-7321-8e69-211fff081375.jsonl
rollout_summary_file: 2026-04-28T14-07-16-z4mp-conda_bashrc_init_clean_shell_fix.md

---
description: Fixed a session-dependent Conda lookup failure by moving `conda init` above the interactive early return in `~/.bashrc`; verified that clean bash shells can now run `conda run -n ms python -V`.
task: Diagnose why `conda` was sometimes missing and make shell init reliable for `ms`
task_group: shell-init-conda
task_outcome: success
cwd: /data/CoordExp
keywords: conda, bashrc, conda init, clean shell, PATH shadowing, non-interactive shell, ms, conda run, bash_profile, profile, environment initialization
---

### Task 1: Diagnose conda availability and patch shell init

task: Inspect `~/.bashrc` and fix Conda initialization so `conda run -n ms python` works in clean bash shells
task_group: shell-init-conda
task_outcome: success

Preference signals:
- when the user said "Please refer to `~/.bashrc` and add necessary initialization" -> future agents should inspect the actual shell startup files and patch them directly rather than suggesting a generic Conda workaround.
- when the user asked why they "sometimes got `/bin/bash: line 1: conda: command not found`" -> future agents should test both the current shell and a minimal clean shell, because the problem may only appear when inherited PATH state is absent.
- when the user asked to "use `conda -n ms` to launch python interpreter" -> future agents should be ready to correct the command shape to `conda run -n ms python` or `conda activate ms && python`, because that is the actual working usage.

Reusable knowledge:
- `conda` existed at `/root/miniconda3/bin/conda`, and `/root/miniconda3/etc/profile.d/conda.sh` was present.
- The root cause was the position of `[ -z "$PS1" ] && return` in `~/.bashrc`: it came before the `conda init` block, which prevented non-interactive bash shells from initializing Conda.
- Moving the entire `conda init` block above the early return made `conda` available as a function in clean shells.
- `~/.profile` sources `~/.bashrc` for login shells on this host; there was no `~/.bash_profile` file.
- Clean-shell verification with `env -i HOME=$HOME TERM=$TERM bash -lc 'type conda; conda run -n ms python -V'` confirmed the fix and returned `Python 3.12.11`.

Failures and how to do differently:
- The current shell initially looked healthy because `PATH` already included Miniconda, which could have hidden the bug. Always verify with a minimal environment when debugging shell startup issues.
- The command `conda -n ms python` is not a valid execution form; use `conda run -n ms python` for one-shot execution or `conda activate ms` for an interactive interpreter.
- `~/.bash_profile` was absent, so the actionable file was `~/.bashrc`; the fix belonged there, not in a login-only file.

References:
- `/root/.bashrc` before fix: the `conda init` block was below `[ -z "$PS1" ] && return`.
- `/root/.bashrc` after fix: the `conda init` block was moved above the early return.
- Verification command/output: `env -i HOME=$HOME TERM=$TERM bash -lc 'type conda; echo "---"; conda run -n ms python -V'` -> `conda is a function` / `Python 3.12.11`.
- Related clean-shell failure before fix: `env -i HOME=$HOME TERM=$TERM bash --noprofile --norc -lc 'conda run -n ms python -V'` -> `bash: line 1: conda: command not found`.

## Thread `019ddf39-2843-7430-b949-910dd6db152f`
updated_at: 2026-04-30T17:19:17+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-30T16-29-18-019ddf39-2843-7430-b949-910dd6db152f.jsonl
rollout_summary_file: 2026-04-30T16-29-18-WSsL-stage1_set_continuation_support2_eval_evolution.md

---
description: Step-wise eval diagnosis of support-reweighted ET-RMP-CE on COCO val200; main gain was precision/duplicate suppression, not large recall or capacity increase
task: inspect eval_detection artifacts across training steps and analyze support reweight + RMP effects
task_group: /data/CoordExp Stage-1 set-continuation production contracts, runtime optimization, and FN-focused diagnostics
task_outcome: success
cwd: /data/CoordExp
keywords: stage1_set_continuation, et_rmp_ce, support reweight, RMP, eval_detection, val200, confidence_postop, bbox_logprob_confidence_exp, gt_vs_pred.jsonl, pred_token_trace.jsonl, AP, F1, duplicate suppression, high-count recall
---

### Task 1: Inspect latest eval artifacts and compare across training steps

task: inspect output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/.../eval_detection across steps 100-916 and compare outputs
task_group: /data/CoordExp stage1 set-continuation eval analysis
task_outcome: success

Preference signals:
- when the user asked to compare outputs “across different training steps,” that suggests future similar requests should be handled as step-wise artifact evolution, not a single metric snapshot.
- when the user said the goal was to understand “capacity, object emission behavior, and decoding dynamics,” that suggests future similar analyses should foreground behavioral diagnosis (emission count, duplicates, FN/FP balance, termination) rather than AP alone.

Reusable knowledge:
- The actual run directory existed under `support2-eff_bs_128-v1`, not the user-typed `support2_eff_bs_128-v1`.
- Every step had valid parsing and no empty predictions; the main changes were in emission count, duplicate suppression, and precision/recall tradeoff.
- The eval surface was `val200` with `sample_limit=200`, COCO coord-token `xyxy`, greedy decoding (`temperature=0`, `top_p=1`, `repetition_penalty=1.1`), and confidence scoring via `bbox_logprob_confidence_exp`.

Failures and how to do differently:
- The user-supplied path had a naming mismatch; future agents should verify the exact run directory before assuming artifacts are missing.
- `config_source.yaml` / manifest prose contained stale `bsz32` / `256` language while the resolved runtime was `16/128`; prefer resolved config and runtime files over authored prose when provenance conflicts.

References:
- actual run directory: `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2-eff_bs_128-v1/v0-20260429-162104/eval_detection`
- step 100 metrics: `bbox_AP=0.399900383`, `f1ish@0.30_f1_full_micro=0.645468998`, `pred_total=1107`, `fp_full=260`, `fn_full=632`
- step 900 metrics: `bbox_AP=0.422105991`, `f1ish@0.30_f1_full_micro=0.705931847`, `pred_total=960`, `fp_full=94`, `fn_full=605`
- example emission collapse: `images/val2017/000000009590.jpg` changed from `123` predictions at step 100 (`118` bowls) to `23` predictions at step 900 (`14` bowls)`

### Task 2: Diagnose support reweight and RMP effects

task: determine whether support reweight + RMP increases object-emission capacity or mainly regularizes decoding
task_group: /data/CoordExp model behavior diagnosis
task_outcome: success

Preference signals:
- when the user explicitly asked about “support reweight” and “RMP” effects on “capacity,” future similar tasks should separate training-side branch-mass metrics from free-rollout behavior.

Reusable knowledge:
- After step 100 the model rapidly suppresses duplicate/loop-like emissions; exact duplicates disappear after step 200 and near-duplicate same-desc pairs collapse from thousands to single digits.
- Total predictions stabilize around `~960-971` on the 200-image val slice, or roughly `0.66-0.67` predictions per GT, which is below full coverage.
- Low-count scenes are already strong; the main weakness is crowded/high-count continuation.
- High-count scenes (`gt>=11`) remain bottlenecked: recall stays around `0.47-0.49` while precision rises into the `0.85` range.
- Training-side RMP metrics improved (`rmp/valid_child_mass_mean`, `rmp/valid_child_top1_acc`), so the objective is shaping valid-branch preference, but free rollout still under-emits in dense scenes.

Failures and how to do differently:
- A nearby `et_rmp_ce_v1` run is only a directional comparator, not a pure ablation, so future comparisons should be labeled non-isomorphic if controls differ.
- The observed effect is cleaner valid continuations, not dramatic recall/capacity gains; do not over-claim increased object emission capacity without stronger high-count recall evidence.

References:
- best support2 step 900: `AP=0.422105991`, `F1@0.30=0.705931847`, `P@0.30=0.899249732`, `R@0.30=0.581024931`, `pred_total=960`
- high-count bucket at step 900: `gt=853`, `pred=501`, `tp=418`, `fp=73`, `fn=435`, `P=0.851323829`, `R=0.490035170`
- training log at step 900: `rmp/valid_child_mass_mean=0.33303473`, `rmp/valid_child_top1_acc=0.37238674`, `loss/rmp_branch_support=0.66095452`, `loss/rmp_branch_total=2.11728554`

### Task 3: Resolve provenance and runtime contract

task: confirm actual resolved runtime/provenance and note stale manifest language
task_group: /data/CoordExp provenance verification
task_outcome: success

Preference signals:
- when the user asked about “latest artifacts,” treat exact run provenance as part of the analysis rather than assuming the visible folder name is always the source of truth.

Reusable knowledge:
- The actual resolved training contract is `per_device_train_batch_size=16`, `gradient_accumulation_steps=1`, `effective_batch_size=128`.
- The objective is `entry_trie_rmp_ce` with `branch_support_weight=2` and `branch_balance_weight=1`.
- `eval_data_provenance.json` confirms `dataset_seed=17`, `sample_limit=200`, and COCO val source `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`.

Failures and how to do differently:
- Stale prose in `config_source.yaml` / `experiment_manifest.json` still mentions `bsz32` / `256`; future agents should use resolved config and runtime files for truth.

References:
- `resolved_config.json`: `training.per_device_train_batch_size: 16`, `training.gradient_accumulation_steps: 1`, `training.effective_batch_size: 128`
- `effective_runtime.json` / manifest prose still contained stale `bsz32` / `effective_batch_size=256` language
- `eval_data_provenance.json`: `dataset_seed=17`, `sample_limit=200`, `dataset_jsonl=public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## Thread `019ddf3b-8db4-7433-9243-314cb0a9ca31`
updated_at: 2026-04-30T16:41:43+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-30T16-31-55-019ddf3b-8db4-7433-9243-314cb0a9ca31.jsonl
rollout_summary_file: 2026-04-30T16-31-55-AiiJ-stage1_et_rmp_ce_support2_eval_evolution_across_steps.md

---
description: Step-by-step analysis of a support2 Stage-1 ET-RMP-CE eval run showed clean validity throughout, stable but only modest recall/FN gains over training, and a strong precision/stability shift from support reweight + RMP rather than a broad object-recall expansion.
task: analyze rollout evolution across training steps for support2 ET-RMP-CE eval_detection
product_task_group: /data/CoordExp stage1 set-continuation eval/diagnostics
cwd: /data/CoordExp
keywords: eval_detection, gt_vs_pred.jsonl, gt_vs_pred_scored.jsonl, metrics.json, per_image.json, pred_token_trace.jsonl, TensorBoard, support reweight, RMP, AP, FN, precision, recall, validity, CoordJSON, coco80
---

### Task 1: Analyze step-by-step rollout evolution

task: compare rollout outputs across training steps for output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/.../eval_detection

task_group: stage1 set-continuation evaluation diagnostics

task_outcome: success

Preference signals:
- when the user asked to “Compare rollout outputs across different steps” and “Identify how predictions evolve during training”, future similar analyses should inspect per-step artifacts directly instead of only final metrics.
- when the user asked “Does the model emit more valid objects over time?” and “Do FN cases decrease?”, future similar analyses should report object counts, TP/FP/FN, and validity/stability signals together.
- when the user asked “Does decoding become more stable or less stable?”, future similar analyses should check prediction-set retention and token-tail behavior, not just AP.

Reusable knowledge:
- The actual on-disk run-name segment was `support2-eff_bs_128-v1`; the user-provided path used `support2_eff_bs_128-v1`.
- Every step from `step_0000100` through `step_0000916` had merged `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`, `per_image.json`, `matches.jsonl`, `matches@0.30.jsonl`, `pred_token_trace.jsonl`, `confidence_postop_summary.json`, `coco_gt.json`, and `coco_preds.json`.
- All steps were schema-valid and non-empty: `invalid_json=0`, `invalid_geometry=0`, `empty_pred=0`; confidence post-op kept all predictions (`kept_fraction=1.0`).
- Step-level evolution should be read from the merged step artifacts, not from `infer_summary.json` alone, because the backend checkpoint field there remained the same across steps even though the step artifacts differed.

Failures and how to do differently:
- Do not trust the repeated `model_checkpoint` field in `infer_summary.json` as evidence of which training step was evaluated; compare step artifact hashes and per-step metrics instead.
- Do not paraphrase the run path label into a different separator form; the filesystem path was the authoritative provenance.

References:
- `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2-eff_bs_128-v1/v0-20260429-162104/eval_detection`
- `metrics.json` trend: `bbox_AP 0.3999 -> 0.423 (best at step 700) -> 0.4181 final`; `f1ish@0.30_fn_loc 618 -> 603`.
- `pred_token_trace.jsonl`: many rows continue after `<|im_end|>` with `<|endoftext|>` tails; tail length shortened over time.

### Task 2: Interpret support reweight + RMP

task: analyze effects of support reweight and RMP on recall, class diversity, and decoding capacity

task_group: stage1 ET-RMP objective analysis

task_outcome: success

Preference signals:
- when the user asked how `support reweight` changes class diversity or object recall, future similar analyses should compare class coverage/entropy and recall separately.
- when the user asked how RMP affects “capacity and decoding behavior”, future similar analyses should separate structured-output validity/capacity from recall quantity.

Reusable knowledge:
- Support2 predicted class coverage stayed roughly flat at `73-74` GT classes covered, with class entropy around `4.46 -> 4.40`; it did not meaningfully increase class diversity.
- Support2 primarily improved precision by suppressing FPs (`246 -> 92` at `@0.30`), while recall only increased slightly (`0.572 -> 0.582`).
- Compared with the sibling no-support ET-RMP run, support2 ended up more conservative: `support2@916 = TP 841 / FP 92 / FN 603 / AP 0.418` vs `no-support@300 = TP 848 / FP 111 / FN 596 / AP 0.420`.
- RMP is the main reason the model stays in a usable structured-output regime; the older candidate-balanced/bidirgate run had parse validity around `0.50`, many empty predictions, and much worse AP/recall, while the ET-RMP runs had `parse_valid_rate=1.0`, zero empty predictions, and much higher AP/recall.
- The remaining bottleneck appears to be crowded/small-object recall and branch sharpness, especially the coordinate branch, where `train/rmp/valid_child_mass_coord` stayed much smaller than the desc/text branch.

Failures and how to do differently:
- Support reweight should not be assumed to be a recall booster; in this run it behaved more like a precision/stability pressure.
- The candidate-balanced/bidirgate run is a capacity baseline rather than a controlled ablation, so use it as a rough contrast, not a mechanistic proof.

References:
- Support2 final metrics: `bbox_AP=0.4181`, `f1ish@0.30_precision_loc_micro=0.9014`, `f1ish@0.30_recall_loc_micro=0.5824`, `f1ish@0.30_fn_loc=603`.
- TensorBoard support2 endpoints: `train/rmp/valid_child_mass_mean 0.2785 -> 0.2998`; `train/rmp/valid_child_mass_coord 0.0185 -> 0.0234`; `train/rmp/valid_child_mass_desc_text 0.3875 -> 0.4053`.
- Older candidate-balanced/bidirgate example: `eval/det_parse_valid_rate` around `0.514999... -> 0.504999...`, `eval/det_empty_pred` around `97 -> 99`, `eval/det_bbox_AP` around `0.193 -> 0.213`.

### Task 3: Summarize decoding stability and token-tail behavior

task: analyze whether decoding becomes more stable or less stable over training

task_group: stage1 set-continuation decoding diagnostics

task_outcome: success

Preference signals:
- when the user asked whether decoding becomes “more stable or less stable”, future similar analyses should mention both object-level retention and stop/tail behavior.

Reusable knowledge:
- The run is schema-stable throughout: all steps had valid JSON and no empty predictions.
- Object-level stability improves over time: adjacent-step retained-prediction Jaccard rises from `0.634` for `100->200` to `0.909` for `900->916`.
- The generation tail is not perfectly clean: most rows emit `<|im_end|>`, but many continue with `<|endoftext|>` after that token. The tail is shorter later in training, so the model is cleaner but not perfectly stop-disciplined.
- Decoding is image-dependent; some crowded examples improve a lot while others regress or plateau.

Failures and how to do differently:
- Do not reduce “stability” to validity alone; the token tail and per-image object retention matter here.
- A single example image can be misleading, so pair a representative stable image with an improving and a regressing example.

References:
- Adjacent-step retention: `100->200 retained@cls+IoU.5=806, jacc=0.634`; `900->916 retained@cls+IoU.5=915, jacc=0.909`.
- Step 100 tail behavior: about `56/200` rows ended exactly at `<|im_end|>` and the average tail after `<|im_end|>` was about `158.6` tokens.
- Step 916 tail behavior: about `58/200` rows ended exactly at `<|im_end|>` and the average tail after `<|im_end|>` was about `100.1` tokens.
- Representative images: `000000019109.jpg` improved sharply; `000000017959.jpg` regressed; `000000000139.jpg` stayed mostly flat.

### Task 4: Identify best step for different goals

task: determine which training step is best for AP vs FN/F1 in the support2 run

task_group: checkpoint selection for stage1 evaluation

task_outcome: success

Preference signals:
- When the user asks for evolution across steps, future similar analyses should state the best step for each criterion rather than only the final step.

Reusable knowledge:
- Step `700` was best for AP in this run (`bbox_AP=0.423`, `bbox_AP50=0.562`, `APl=0.563`).
- Step `900` was best for FN/F1 behavior (`FN@0.30=594`, `TP@0.30=850`, `F1@0.30=0.715`).
- Final step `916` regressed slightly from step `900` on FN/F1 and from step `700` on AP, so the last checkpoint was not the best checkpoint on either axis.

References:
- Step 700 metrics: `bbox_AP=0.423`, `f1ish@0.30_fn_loc=601`, `f1ish@0.30_f1_loc_micro=0.707`.
- Step 900 metrics: `bbox_AP=0.422`, `f1ish@0.30_tp_loc=850`, `f1ish@0.30_fn_loc=594`, `f1ish@0.30_f1_loc_micro=0.715`.
- Step 916 metrics: `bbox_AP=0.418`, `f1ish@0.30_tp_loc=841`, `f1ish@0.30_fn_loc=603`, `f1ish@0.30_f1_loc_micro=0.708`.

## Thread `019ddf43-6d02-7822-a895-7ec68ceba053`
updated_at: 2026-04-30T17:14:11+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-30T16-40-31-019ddf43-6d02-7822-a895-7ec68ceba053.jsonl
rollout_summary_file: 2026-04-30T16-40-31-0a0H-notion_linear_connector_probing_and_session_boundary.md

---
description: User repeatedly probed Notion and Linear connector availability; Notion eventually responded, but Linear was not exposed in-session and could not be installed from chat. Highest-value takeaway: verify connector liveness by probing the tool surface, but if Linear is unavailable, stop and ask for app/OAuth enablement in the session/workspace.
task: connector-availability-probe-for-notion-and-linear
task_group: connector_setup
task_outcome: partial
cwd: /data/CoordExp
keywords: Notion, Linear, MCP startup failed, handshake timeout, connector availability, OAuth, plugin skill, tool registry, superpowers, Codex app
---

### Task 1: Notion connector availability check

task: probe Notion connector availability and confirm whether the account/session is connected
task_group: connector_setup
task_outcome: partial

Preference signals:
- The user repeatedly asked to “Connect Notion” / “try to use notion” and then asked “Have you connected to my Notion account?” -> the user wants a live check of connector state rather than assumptions or generic setup guidance.
- The terse imperative phrasing suggests future responses should quickly probe the connector and report the actual state.

Reusable knowledge:
- `_notion_get_teams {}` and `_notion_get_users {"user_id":"self"}` are the probe calls used here.
- The failure mode was `MCP startup failed: timed out handshaking with MCP server after 30s`.
- A later retry of `_notion_get_teams {}` succeeded and returned `{"joinedTeams":[],"otherTeams":[],"hasMore":false}`.

Failures and how to do differently:
- Early Notion calls failed because the MCP client did not start in time; one retry later succeeded, so a transient startup delay should be retried once before declaring the connector unavailable.

References:
- `_notion_get_teams {}` → `tool call error: failed to get client … MCP startup failed: timed out handshaking with MCP server after 30s`
- `_notion_get_users {"user_id":"self"}` → same timeout
- Successful `_notion_get_teams {}` output: `{"joinedTeams":[],"otherTeams":[],"hasMore":false}`

### Task 2: Linear connector / app availability check

task: determine whether Linear can be connected/installed in the current session and explain the `linear` skill pointer
task_group: connector_setup
task_outcome: uncertain

Preference signals:
- The user repeatedly asked to “try to connect to `linear`” and “help me install this `Linear` App/plugin in this session” -> they want the agent to actively verify session-level availability, not just describe Linear in the abstract.
- The user asked “[$linear:linear](...) what does this do?” -> they want skill/plugin pointers explained in plain English when they appear.

Reusable knowledge:
- The visible tool registry surfaced GitHub and Serena namespaces, but no Linear namespace.
- The local skill file `/data/CoordExp/.codex/plugins/cache/openai-curated/linear/886026e9/skills/linear/SKILL.md` states that Linear work assumes connected OAuth-backed tools.
- That skill instructs that if Linear tools are unavailable, the agent should pause and ask the user to enable the bundled Linear app, complete OAuth, and restart Codex/the session if the tools still do not appear.
- The `[$linear:linear](...)` reference is only a skill pointer; it does not install or enable Linear by itself.

Failures and how to do differently:
- Linear could not be installed from chat because the connector was not exposed in-session.
- The correct boundary is to stop and ask the user to enable/auth the Linear app in the UI/session, then retry in a fresh session.

References:
- `/data/CoordExp/.codex/plugins/cache/openai-curated/linear/886026e9/skills/linear/SKILL.md`
- Skill excerpt: “If Linear tools are unavailable, pause and ask the user to connect the Linear app… Restart Codex or the current session if the tools still do not appear.”
- User wording: `help me install this \\`Linear\\` App/plugin in this session`
- User wording: `[$linear:linear](...) what does this do?`
- Discovery attempts surfaced GitHub / Serena tool namespaces, but no Linear tool namespace

## Thread `019e14e9-2b24-7420-a7ca-c711472368f8`
updated_at: 2026-05-12T16:15:06+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/sessions/2026/05/11/rollout-2026-05-11T02-41-25-019e14e9-2b24-7420-a7ca-c711472368f8.jsonl
rollout_summary_file: 2026-05-11T02-41-25-LODS-coordexp_baidudisk_union_sync_skill_and_asset_sync_design.md

---
description: User worked through CoordExp large-asset sync on Baidu Netdisk, then wanted the workflow packaged as a reusable Codex skill; key durable takeaway is append-only union sync (new files auto-sync, deletes manual, no overwrite), with BaiduPCS-Go as the transport layer and skills self-contained under .codex/skills.
task: design-and-package-baidudisk-union-sync-skill
 task_group: CoordExp / Baidu Netdisk asset-sync workflow
 task_outcome: success
cwd: /data/CoordExp
keywords: BaiduPCS-Go, Baidu Netdisk, union sync, append-only, no-delete, no-overwrite, Codex skill, SKILL.md, config-template, manifest, conflict detection, tmux, HTTPS PAT, outputs, artifact sync
---

### Task 1: Pull latest main and learn output/provenance conventions

task: git pull --ff-only origin main; inspect docs about output/model_cache/public_data provenance
 task_group: CoordExp repo sync / provenance policy
 task_outcome: success

Preference signals:
- When the user said “你不需要编写和设计。我的远端repo已经规划好了，你只需要知悉” after the initial pull, that indicates that in similar sync situations the agent should stop at awareness/reading instead of redesigning the workflow.
- The user’s repeated emphasis on syncing `output/`, `model_cache`, `public_data`, artifacts, and using Baidu Netdisk across nodes suggests durable concern for cross-machine recovery and provenance rather than ephemeral local cleanup.

Reusable knowledge:
- `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md` defines `model_cache/` and raw `public_data/` as not Baidu sync surfaces, and `output/` as the default Baidu sync surface.
- `manifests/public_data_provenance/` is the git-tracked provenance location for processed `public_data/`.
- The repo already had a manual BaiduPCS-Go transfer skill, so future sync work should build on that instead of improvising a new transfer stack.

Failures and how to do differently:
- After the pull, lots of `.codex/memories/*` untracked files surfaced; the safe behavior was to leave them untouched unless the user explicitly asked for cleanup.

References:
- `git pull --ff-only origin main` fast-forwarded `750834d..4dbc9e4`.
- New policy docs were added in the pull: `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md`, `docs/superpowers/specs/2026-05-11-output-sync-and-public-data-provenance-design.md`, and `manifests/public_data_provenance/README.md`.

### Task 2: Fix Baidu upload failures by renaming special PNG filenames and uploading mappings

task: recover 89 failed PNG uploads under outputs/analysis by renaming Baidu-hostile filenames and re-uploading
 task_group: BaiduPCS-Go artifact upload recovery
 task_outcome: success

Preference signals:
- When the user later said “我想要走`2`，简单直接一些，可以吗？如果有必要，可以再同步修改相应的 references。” after the upload failures, that indicates a preference for the direct rename-based fix over archive-based recovery when the artifact set is browsable and small enough.
- The user allowed reference updates only if necessary, which means future fixes should check for actual textual references before editing manifests/docs.

Reusable knowledge:
- BaiduPCS-Go / Baidu Netdisk rejected the original PNG names that contained `:` and `->`; a safe rename plus mapping manifest resolved the issue.
- The working rename rule was `->` -> `_to_` and `:` -> `_`.
- The mapping files were stored in `outputs/_baidu_filename_mapping/20260511_special_png_filename_sanitization.{json,tsv}` and uploaded successfully.
- A small probe upload with a safe filename confirmed the issue was filename compatibility, not file corruption.

Failures and how to do differently:
- Retrying the same original filenames failed again; the correct pivot was to rename locally and re-upload, not to keep retrying the same paths.
- The initial broad upload failed only on the special-name PNGs; the rest of the outputs were already fine.

References:
- Failed file count: 89; total size about 3.8 MB.
- Successful remote listing showed renamed files such as `1000_9_4_to_5__base__baseline__gt.png`.
- Probe success: a renamed `safe_probe.png` uploaded to `/CoordExp/upload_probe/` successfully.

### Task 3: Design a Git-like Baidu sync pipeline and package it as a reusable skill

task: think through sync architecture for multiple environments, then package it as a self-contained Codex skill
 task_group: Cross-node large-asset sync design
 task_outcome: success

Preference signals:
- The user asked to “将整个开发环境的sync 的pipeline 搭建好”, “模仿git的那种手感”, and later wanted the result “打包成一个 skills” with all scripts and references inside the skill folder, not scattered in the codebase.
- The user explicitly wanted the skill to be “可泛化、通用” so another environment’s Codex agent can understand and execute it without extra explanation.
- The user wanted “新增的内容，自动sync，而删除需要纯手动操作”; that is a durable default for future sync design: append-only automation, manual deletion.

Reusable knowledge:
- The correct high-level model is an **append-only union sync**, not a mirror sync.
- A useful conceptual split is: Git stores rules/skills/denylists/docs; Baidu stores large assets and append-only manifests.
- The skill should expose `doctor`, `scan`, `status`, `push`, `pull`, and `sync` commands, with `--apply` required for actual transfer.
- Pull should stage to a local temp directory and merge via `rsync --ignore-existing`; push should use BaiduPCS-Go with skip-existing semantics and never overwrite.
- Conflicts should be same-path different-content cases that halt the run and require manual resolution.
- Deletion should be manual and can optionally be guarded by a denylist so older nodes don’t re-upload deleted paths.

Failures and how to do differently:
- Using `BaiduPCS-Go upload --policy rsync` or `download --ow` would be too close to a mirror/overwrite workflow; the safer design is to control union semantics in the wrapper and keep overwrite/delete out of the automation path.
- The direct repo-codebase design was rejected by the user in favor of a skill package, so future work should default to `.codex/skills/...` when the goal is cross-environment reuse.

References:
- The append-only union semantics were documented in the new skill’s `SKILL.md` and `references/semantics.md`.
- The user’s desired sync handoff was effectively: new files should propagate both ways; deletes are always manual.

### Task 4: Package the sync design as a standalone Codex skill and sync only that change

task: add a self-contained `baidudisk-union-sync` skill under `.codex/skills/` and commit/push only that skill
 task_group: Codex skill authoring / repo hygiene
 task_outcome: success

Preference signals:
- The user requested “请只`commit and sync`你的修改而忽略其他的dirty changes” -> in future similar situations, only stage the files belonging to the current request and leave unrelated dirty state alone.
- The user wanted the sync system to be reusable by another environment’s Codex agent, which supports the decision to package the logic as a self-contained skill folder.

Reusable knowledge:
- The new skill directory is `.codex/skills/baidudisk-union-sync/` and contains everything needed to use the workflow elsewhere:
  - `SKILL.md`
  - `scripts/baidu_union_sync.py`
  - `references/config-template.json`
  - `references/semantics.md`
  - `agents/openai.yaml`
- The skill validation passed under `conda run -n ms python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/baidudisk-union-sync`.
- The commit/push for the skill was successful on `main` via HTTPS remote.

Failures and how to do differently:
- The repo had unrelated dirty state in `.codex/memories/rollout_summaries/*.md` and an untracked `scripts/tools/commit_codex_memories.sh`; those were intentionally not staged or committed.
- `quick_validate.py` under the plain system Python failed because `yaml` was missing; future validation in this repo should prefer `conda run -n ms`.
- `py_compile` created a `__pycache__` under the skill directory, which had to be removed to keep the skill portable and clean.

References:
- Commit: `ac0e0d8 chore(codex): add baidudisk union sync skill`.
- Push: `82d5b26..ac0e0d8  main -> main`.
- Validation: `Skill is valid!`.
- The current worktree still had unrelated dirty deletions in `.codex/memories/rollout_summaries/*.md`, but they were intentionally ignored for this commit.

## Thread `019e1c92-7573-7ae0-8c1d-db40e1124ba8`
updated_at: 2026-05-12T16:15:06+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/sessions/2026/05/12/rollout-2026-05-12T14-23-40-019e1c92-7573-7ae0-8c1d-db40e1124ba8.jsonl
rollout_summary_file: 2026-05-12T14-23-40-y0Z2-mattpocock_skills_import_prune_and_commit.md

---
description: Installed mattpocock/skills, discovered it writes to ./.agents/skills rather than .codex/skills, then pruned the set down to three CoordExp-local helper skills (grill-me, handoff, zoom-out), customized their wording for this repo, validated them, and committed only those three files.
task: install and relocate mattpocock/skills into repo-local Codex skills, prune unnecessary ones, customize survivors, commit
task_group: codex-skills
task_outcome: success
cwd: /data/CoordExp
keywords: skills, mattpocock/skills, .codex/skills, .agents/skills, skill-creator, quick_validate.py, git-commit-push, chore(codex), Codex home, pruning, frontmatter
---

### Task 1: Install, inspect, prune, and customize imported skills

task: run `npx skills@latest add mattpocock/skills`, move usable skills into `.codex/skills`, remove unnecessary ones, and rewrite the survivors for CoordExp
task_group: codex skill installation and workspace-local customization
task_outcome: success

Preference signals:
- user asked to “properly move those skills to the codex directory” and was unsure whether they fit in `codex` -> future runs should verify the actual install target instead of assuming package output matches the desired home.
- user said `.codex/skills/to-prd, .codex/skills/to-issues` are unnecessary because this repo is maintained personally -> future runs should feel free to drop issue/PRD/triage workflow skills when they conflict with the repo’s actual operating model.
- user said “We may just remove some skills completely if you suggest” and later “You may customize/adjust those skills when helpful and remove when unnecessary” -> future runs should prune aggressively and rewrite retained skills for local fit.

Reusable knowledge:
- `npx --yes skills@latest add mattpocock/skills -y --copy` installs into `./.agents/skills`, not directly into `.codex/skills`.
- The imported pack contained 14 skills: `caveman`, `diagnose`, `grill-me`, `grill-with-docs`, `handoff`, `improve-codebase-architecture`, `prototype`, `setup-matt-pocock-skills`, `tdd`, `to-issues`, `to-prd`, `triage`, `write-a-skill`, `zoom-out`.
- Final retained skills were reduced to three: `grill-me`, `handoff`, `zoom-out`.
- The workspace validator is `conda run -n ms python .codex/skills/.system/skill-creator/scripts/quick_validate.py <skill_dir>`; it reported `Skill is valid!` after the final edits.

Failures and how to do differently:
- The package’s native install surface was a temp-workdir `.agents/skills` tree, so a follow-up copy into `.codex/skills` was necessary.
- The first retained versions of `handoff` and `zoom-out` failed validation because of unsupported frontmatter keys (`argument-hint`, `disable-model-invocation`); those keys had to be removed while preserving the body text.

References:
- `npx --yes skills@latest add mattpocock/skills -y --copy`
- temp install output: `Source: https://github.com/mattpocock/skills.git`, `Installing all 14 skills`, `Installing skills`
- validator errors:
  - `Unexpected key(s) in SKILL.md frontmatter: argument-hint. Allowed properties are: allowed-tools, description, license, metadata, name`
  - `Unexpected key(s) in SKILL.md frontmatter: disable-model-invocation. Allowed properties are: allowed-tools, description, license, metadata, name`
- final kept paths: `.codex/skills/grill-me/SKILL.md`, `.codex/skills/handoff/SKILL.md`, `.codex/skills/zoom-out/SKILL.md`

### Task 2: Commit the customized helper skills

task: stage and commit only the three customized `.codex/skills/*` directories
task_group: git hygiene / Codex workspace maintenance
task_outcome: success

Preference signals:
- user said “commit them properly” -> future runs should stage narrowly, verify the staged diff, and commit the intended skill changes as one logical unit.

Reusable knowledge:
- Repo remote is HTTPS: `origin https://github.com/Pein2017/CoordExp.git`.
- `github_personal_token.txt` is ignored by `.gitignore` (`.gitignore:2:* github_personal_token.txt`).
- For new untracked skill directories, plain `git diff` is empty until staged; use `git diff --cached --stat` / `--name-only` to review the actual commit surface.
- The final commit on `main` was `5e9dd6c chore(codex): add workspace helper skills`.

Failures and how to do differently:
- No failure in the final commit path; the main caution is to review the staged diff rather than expecting plain `git diff` to show new files.

References:
- `git remote -v` -> `origin	https://github.com/Pein2017/CoordExp.git (fetch/push)`
- staged diff: `.codex/skills/grill-me/SKILL.md`, `.codex/skills/handoff/SKILL.md`, `.codex/skills/zoom-out/SKILL.md`, `3 files changed, 79 insertions(insertions)`
- `git diff --cached --check` passed
- commit: `[main 5e9dd6c] chore(codex): add workspace helper skills`

## Thread `019e1cb2-be00-77c3-84b3-c6521215606d`
updated_at: 2026-05-12T16:19:01+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/sessions/2026/05/12/rollout-2026-05-12T14-58-56-019e1cb2-be00-77c3-84b3-c6521215606d.jsonl
rollout_summary_file: 2026-05-12T14-58-56-wZOt-remove_linear_notion_references_repo_cleanup.md

---
description: Deep cleanup of explicit Linear/Notion references in /data/CoordExp, including repo docs, hidden plugin/cache/state surfaces, and verification that remaining exact matches are benign math/technical uses. Important takeaway: remove explicit connector/plugin/session artifacts separately from generic `linear`/`notion` prose; `.codex/sessions` still contains `*.jsonl` logs unless deleted explicitly.
task: clean-linear-notion-references-repo-wide
task_group: /data/CoordExp cleanup / verification
task_outcome: success
cwd: /data/CoordExp
keywords: Linear, Notion, cleanup, verification, hidden caches, plugin metadata, sessions jsonl, docs rewrite, exact-term sweep, benign linear math, .codex, .claude
---

### Task 1: Repo-wide Linear/Notion cleanup

task: exhaustively inspect /data/CoordExp and remove explicit Linear/Notion workflow material while preserving unrelated content

task_group: repo cleanup

task_outcome: success

Preference signals:
- when the user said "Exhaustively inspect every file at every directory level" and "Use subagents to split discovery, cleanup, and verification work" -> future cleanup tasks should start with broad discovery, then separate cleanup and verification passes rather than editing opportunistically.
- when the user said "Avoid removing unrelated research-management content unless it is tied to Linear or Notion" -> future cleanup should preserve repo-local research/process material unless it explicitly names those tools.
- when the user required a final verification search for `Linear`, `linear`, `Notion`, and `notion` -> future similar tasks should end with exact-term sweeps and report any remaining matches with justification.

Reusable knowledge:
- Hidden plugin/cache/state surfaces were the highest-signal cleanup targets, along with repo policy docs and super-power planning docs that encoded management boundaries.
- Exact tool-reference patterns that were useful for sweeping were: `notion@openai-curated`, `linear@openai-curated`, `connector_name: Notion`, `connector_name: Linear`, `mcp__codex_apps__notion`, `mcp__codex_apps__linear`, `app.notion.com`, `notion-faq`, `linear_notion`, `Notion migration`, `Linear workspace`, and `Linear tickets`.
- Exact-term verification should separate capitalized tool names from lowercase technical wording; in this repo, lowercase `linear` often appears in math/probe contexts and should not be removed unless the surrounding text is tool/process guidance.

Failures and how to do differently:
- The first broad search hit a very large amount of irrelevant content in vendored caches, generated artifacts, and model/tokenizer files; narrowing to explicit connector names, plugin metadata, and repo-owned docs made the cleanup tractable.
- Some hidden session-history files under `.codex/sessions` were not removed in the main cleanup pass; if the goal is total removal of session logs, delete that tree explicitly and then re-run verification.

References:
- `AGENTS.md`, `docs/AGENT_ENGINEERING_CONSTITUTION.md`, `docs/superpowers/specs/2026-05-01-compact-detection-sequence-ablation-design.md`, `docs/superpowers/plans/2026-05-01-compact-detection-sequence-ablation.md`, `docs/superpowers/plans/2026-05-06-compact-full-prefix-rollin-multipositive-unification.md`, `docs/superpowers/specs/2026-05-06-compact-full-prefix-rollin-multipositive-unification-design.md`, `progress/benchmarks/2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md`, `progress/directions/full_idea_v3.md`
- `.codex/config.toml`, `.claude/plugins/marketplaces/claude-plugins-official/.claude-plugin/marketplace.json`, `.codex/skills/handoff/SKILL.md`, `.codex/skills/grill-me/SKILL.md`, `.codex/skills/.system/plugin-creator/references/plugin-json-spec.md`, `.codex/superpowers/skills/writing-skills/SKILL.md`
- `external/label-studio/.github/workflows/algolia-crawler-hs-docs.yml`, `external/label-studio/docs/.gitignore`

### Task 2: Session-log follow-up

task: check whether `.codex/sessions` JSONL files were removed

task_group: hidden state verification

task_outcome: success

Preference signals:
- when the user asked "Did you remove the `*.jsonl` in the `.codex/sessions`?" -> future cleanup work should verify whether session history logs are actually deleted rather than assuming tree-level cleanup covered them.

Reusable knowledge:
- `.codex/sessions` is a separate history surface from `.codex/cache`, `.codex/.tmp`, and `.codex/memories`; removing one does not imply the others were removed.
- Direct `find ... -name '*.jsonl'` verification is the reliable way to confirm whether session logs are still present.

Failures and how to do differently:
- The directory `.codex/sessions` still existed and still contained JSONL logs after the cleanup pass; a future pass must delete it explicitly if that is the desired outcome.

References:
- `find /data/CoordExp/.codex/sessions -type f -name '*.jsonl'`
- Example surviving files before any explicit deletion:
  - `/data/CoordExp/.codex/sessions/2026/05/07/rollout-2026-05-07T03-27-53-019e007a-4507-7881-8b73-d0ea97b17886.jsonl`
  - `/data/CoordExp/.codex/sessions/2026/05/12/rollout-2026-05-12T15-02-24-019e1cb5-eb40-7d22-9be0-446d605d13c7.jsonl`

### Task 3: Final verification of remaining matches

task: distinguish explicit tool references from benign linear/notion wording after cleanup

task_group: verification

task_outcome: success

Preference signals:
- when the user said they wanted the repo to "look as though those tools were never part of the project" -> final verification should classify surviving exact-word hits as benign or residual with justification, not just dump search output.

Reusable knowledge:
- The tool-only search patterns returned no matches after cleanup.
- Remaining `Linear` hits were benign technical/math text such as `nn.Linear`, `Linear fit`, `Linear interpolation`, and `Linear Frequency`.
- Remaining `notion` hits were generic noun usage or third-party data, not project workflow guidance.

Failures and how to do differently:
- A raw exact-word sweep produces many false positives from math, dataset text, and vendored dependencies; always pair it with a tool-only pattern sweep for cleanup tasks like this.

References:
- Benign examples: `tests/test_rollout_offload_context.py`, `progress/benchmarks/2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md`, `external/label-studio/web/libs/editor/src/components/Timeline/Controls/SpectrogramControl.tsx`, `scripts/analysis/visualize_packing_results.py`.
- Explicit tool patterns verified absent: `notion@openai-curated`, `linear@openai-curated`, `connector_name: Notion`, `connector_name: Linear`, `mcp__codex_apps__notion`, `mcp__codex_apps__linear`, `app.notion.com`, `linear_notion`, `Notion migration`, `Linear workspace`, `Linear tickets`.

## Thread `019e1cea-f7f7-7e11-9edf-835efccf7a88`
updated_at: 2026-05-12T16:46:55+00:00
cwd: /data/CoordExp
rollout_path: /data/CoordExp/.codex/sessions/2026/05/12/rollout-2026-05-12T16-00-21-019e1cea-f7f7-7e11-9edf-835efccf7a88.jsonl
rollout_summary_file: 2026-05-12T16-00-21-2IoE-codex_sessions_missing_recovery_memory_refresh_root_cause.md

---
description: Recovered several deleted Codex rollout JSONL files, restored `.codex/memories` to the pre-latest-refresh state, and traced the likely cause to a Codex Desktop/app-server 0.130.0 refresh/reset around 2026-05-12 16:01 UTC rather than a plain shell delete command.
task: recover missing rollout-*.jsonl files and restore .codex/memories pre-refresh
task_group: CoordExp / Codex runtime recovery
task_outcome: partial
cwd: /data/CoordExp
keywords: codex app-server, rollout-*.jsonl, .codex/sessions, session_index.jsonl, state_5.sqlite, logs_2.sqlite, .codex/memories, refresh memories, deleted inode, lsof +L1, git reflog, origin/main, CODEX_HOME, Codex Desktop 0.130.0
---

### Task 1: Recover missing rollout-*.jsonl files

task: locate and recover missing Codex rollout JSONL sessions under /data/CoordExp and related homes
task_group: runtime recovery / Codex sessions
task_outcome: partial

Preference signals:
- when the user said "我有好多sessions似乎丢失了，帮我找一找。我现在只能看到 4 个 sessions。我是在用本地的Codex APP通过远程SSH连接到此环境。" -> future similar incidents should assume a concrete recovery investigation, not just a conceptual explanation
- when the user added "可能跟我升级了`codex cli`有关" and later "我的 HOME应该在当前的`.codex/`下" -> future similar incidents should check CODEX_HOME and upgrade transitions early
- when the user asked "是有`codex进程`在后台运作吗？可能是某个进程误删了吗" -> future similar incidents should inspect live Codex processes and open file handles for deleted files

Reusable knowledge:
- `state_5.sqlite` can preserve thread metadata even when the corresponding `rollout-*.jsonl` files are missing.
- A live `codex app-server` can hold deleted rollout JSONL files open long enough to recover them via `/proc/<pid>/fd/<fd>`.
- `lsof -nP 2>/dev/null | rg '/data/CoordExp/\.codex/(sessions|memories|state_5|logs_2|session_index)'` is a useful first snapshot for Codex recovery work.

Failures and how to do differently:
- A first pass only saw one visible rollout JSONL and could have incorrectly concluded that almost everything was gone; later `lsof +L1` showed deleted-but-open files. Check open deleted file descriptors early.
- Once the app-server closed the relevant file descriptors, no more JSONL recovery was possible from the process.

References:
- Recovered rollout files now exist under `/data/CoordExp/.codex/sessions/...`:
  - `rollout-2026-05-07T03-27-53-019e007a-4507-7881-8b73-d0ea97b17886.jsonl`
  - `rollout-2026-05-11T02-41-25-019e14e9-2b24-7420-a7ca-c711472368f8.jsonl`
  - `rollout-2026-05-12T14-23-40-019e1c92-7573-7ae0-8c1d-db40e1124ba8.jsonl`
  - `rollout-2026-05-12T14-58-56-019e1cb2-be00-77c3-84b3-c6521215606d.jsonl`
  - `rollout-2026-05-12T15-02-24-019e1cb5-eb40-7d22-9be0-446d605d13c7.jsonl`
  - `rollout-2026-05-12T16-00-21-019e1cea-f7f7-7e11-9edf-835efccf7a88.jsonl`
- Recovery mirror: `/data/CoordExp/temp/codex_session_recovery_20260512_1614/`

### Task 2: Recover memory summaries and revert `.codex/memories` before the latest refresh

task: recover deleted memory summary markdown files and restore `.codex/memories` to the pre-refresh state
task_group: git-tracked memory recovery / .codex/memories
task_outcome: success

Preference signals:
- when the user asked "请同步查看一下我的`git history`，跟我同步`.codex/memories`有关吗？" -> future similar recoveries should inspect tracked-memory commits as well as runtime files
- when the user said "我的`.codex/memories`不应该是`5-12`16点左右创建的才对啊。那一刻，发生了什么，是否可以复原？我的`.codex/memories`应该存在了好几个月了才对，可能就是那个时候‘覆盖’了？" -> future agents should distinguish first creation from a refresh/overwrite event
- when the user clarified "基本就是回退到最近一次的`refresh memoreis`之前的状态" -> restore to the parent of the refresh commit, not just to the remote tip

Reusable knowledge:
- `ae7e0dfc` is the latest `refresh memories` commit and `cf812b55` is its parent/pre-refresh state for `.codex/memories` in this checkout.
- The latest refresh deleted a batch of historical rollout-summary files and introduced `phase2_workspace_diff.md`.
- Git can recover the markdown summaries because they are tracked; it cannot directly recover ignored JSONL session files.

Failures and how to do differently:
- Restoring from `origin/main` did not solve the problem because the remote already contained the latest `refresh memories` commit. Check HEAD vs remote first; if they are identical, restore from the desired parent commit instead.
- The user’s request not to push local commits meant the right action was a worktree restore, not publishing a new commit.

References:
- Git restore source used: `cf812b55`
- Latest refresh commit: `ae7e0dfc`
- Recovered markdown summaries stored in `/data/CoordExp/temp/codex_session_recovery_20260512_1614/recovered_memory_summaries_from_git/`
- The restored summary for the key thread was `2026-05-11T06-50-47-82L9-public_data_provenance_jsonl_checksums_and_manifest_handoff.md`

### Task 3: Determine likely root cause and whether a background Codex process mis-deleted files

task: root-cause investigation for missing Codex sessions and memory refresh side effects
task_group: Codex runtime / root-cause analysis
task_outcome: partial

Preference signals:
- when the user said "帮我尝试挽救，最主要的是找到最可疑的根因" -> prioritize root-cause analysis over cosmetic fixes
- when the user asked "是有`codex进程`在后台运作吗？可能是某个进程误删了吗" -> explicitly check live Codex processes and their file handles
- when the user asked whether this was tied to `.codex/memories` -> investigate both runtime session state and tracked memory state, not only one side

Reusable knowledge:
- Live `codex app-server` / `codex app-server proxy` processes were running with `CODEX_HOME=/data/CoordExp/.codex` and `cwd=/root` during the incident.
- The key log symptom was `state db reconcile_rollout extraction failed ... No such file or directory (os error 2)` for the missing rollout path.
- `.gitignore` excludes `.codex/sessions`, `session_index.jsonl`, `state_5.sqlite`, and `logs_2.sqlite`, so Git history does not explain the missing runtime JSONL files.

Failures and how to do differently:
- No explicit shell `rm -rf`, `git clean`, `rsync --delete`, or similar destructive command targeting `.codex/sessions` was found. The absence of a shell delete command suggests an internal app/runtime reset path rather than a plain terminal delete.
- The current evidence does not prove the exact initiating command or internal method. A controlled reproduction in a temporary `CODEX_HOME` would be needed for certainty.

References:
- Live processes observed:
  - `node /root/.nvm/versions/node/v22.22.0/bin/codex app-server --listen unix://`
  - `codex app-server proxy`
- Important log entry:
  - `state db reconcile_rollout extraction failed /data/CoordExp/.codex/sessions/2026/05/11/rollout-2026-05-11T06-50-47-019e15cd-7907-76b0-902a-83af0aaee1f3.jsonl: No such file or directory (os error 2)`
- Timestamped runtime file evidence:
  - `.codex/sessions` birth: `2026-05-12 16:01:30+00:00`
  - `.codex/session_index.jsonl` birth: `2026-05-12 16:01:36+00:00`
  - `phase2_workspace_diff.md` birth: `2026-05-12 16:01:35+00:00`

### Task 4: Restore `.codex/memories` to the pre-latest-refresh worktree state without pushing

task: restore .codex/memories to pre-refresh state and avoid local push

task_group: git/worktree recovery

task_outcome: success

Preference signals:
- when the user said "由于我的`.codex/memories`似乎被重置了。请不要推送我本地的commit，而是让远端的`.codex/memories`覆盖下来。" -> do not push local commits; restore in the worktree instead
- when the user clarified they wanted the state before the latest `refresh memoreis` -> revert to the parent of the refresh commit, not to the remote tip

Reusable knowledge:
- `git fetch origin main` followed by `git rev-list --left-right --count main...origin/main` showed `0 0`, meaning the remote already matched local HEAD and could not by itself restore the pre-refresh state.
- `git restore --source=cf812b55 --staged --worktree -- .codex/memories` restored the memory tree to the version before the latest refresh.
- Unstaging after the restore left the worktree changed but avoided a commit/push.

Failures and how to do differently:
- Restoring from `origin/main` was insufficient because `origin/main` already contained the refresh commit. Use the pre-refresh parent commit when the user wants the previous state.
- The operation left `.codex/memories` showing expected working-tree changes; that is acceptable for a rollback recovery, but future agents should be explicit that it is a worktree restore, not a published history rewrite.

References:
- Backup archive created before the restore: `/data/CoordExp/temp/codex_session_recovery_20260512_1614/local_codex_memories_before_origin_restore_20260512T164050Z.tar.gz`
- Restore source: `cf812b55`
- Latest refresh commit: `ae7e0dfc`
- Current worktree reflects the pre-refresh memory content in `.codex/memories`.

