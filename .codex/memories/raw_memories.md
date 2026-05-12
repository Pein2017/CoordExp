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

