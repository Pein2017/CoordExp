thread_id: 019dca7b-7ad7-7393-b10b-bce65beae793
updated_at: 2026-04-28T14:10:15+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/26/rollout-2026-04-26T15-49-43-019dca7b-7ad7-7393-b10b-bce65beae793.jsonl
cwd: /data/CoordExp
git_branch: main

# Investigated why Stage-1 set-continuation training loss improved while eval mAP fell on `val200`

Rollout context: the user asked for a diagnosis of the training logs at `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_pem_close_suppress/setcont-coco1024-sota1332-pem-close-suppress/v2-20260426-114031/logging.jsonl`, under `configs/stage1/set_continuation/production.yaml`, because training loss decreases but evaluation mAP drops from roughly 0.38 at the original checkpoint to about 0.16 in the run.

## Task 1: Diagnose loss-vs-mAP discrepancy for Stage-1 set-continuation

Outcome: partial

Preference signals:
- the user asked for a “detailed diagnosis” and explicitly framed the question as either “A computational or implementation error” vs “Genuine model degradation” -> future similar debugging tasks should prioritize root-cause analysis over a shallow metric summary.
- the user pointed to a specific log path and config path -> future agents should inspect the exact artifact and resolved config first, rather than generalizing from memory.

Key steps:
- inspected the rollout skill/docs first, then the exact `logging.jsonl`, `config_source.yaml`, `resolved_config.json`, `experiment_manifest.json`, `eval_data_provenance.json`, and the generated eval sidecars under the run directory.
- confirmed the run was Stage-1 set-continuation from `checkpoint-1332-merged-full`, with `val_sample_limit=200`, `eval_detection.limit=200`, coord-token `xyxy`, `score_mode: confidence_postop`, strict parsing, and greedy decoding.
- compared against earlier `val200` coord-token reference artifacts for `checkpoint-1332` and also against related set-continuation run families (`pem_close_suppress`, `closefix`, `boundaryfix`, `schemafix`, `bidirgate_warmup10`).
- traced the active trainer code paths in `src/trainers/stage1_set_continuation/{branch_encoder.py,losses.py,sampling.py,serialization.py,trainer.py}` and the eval callback in `src/callbacks/stage1_detection_eval.py`.
- wrote a factual report to `temp/stage1_set_continuation_training_report_20260428.md` instead of inventing a root cause.

Failures and how to do differently:
- no single definitive root cause was proven from the current evidence; the result should be treated as a diagnosis of symptoms plus code-path facts, not a completed postmortem.
- the rollout showed that train-time eval is callback-driven on the live model, so a future investigator should not assume the summary path alone identifies the exact weights used; verify the callback/model wiring explicitly.
- the early `pem_close_suppress` logs are strongly non-comparable to later candidate-balanced/schema-aware runs because the objective/config changed materially; future comparisons should keep eval scope and decode settings matched.

Reusable knowledge:
- `Stage1DetectionEvalCallback.on_evaluate()` uses the live trainer model (`engine.model = runtime_model`), switches it to eval, runs inference, restores training mode, and then evaluates the resulting `gt_vs_pred.jsonl`; a configured checkpoint path in the summary is not, by itself, sufficient evidence of which weights were actually used during the callback.
- the current set-continuation branch encoder is continuation-aware: non-terminal candidate branches append `, `, terminal branches append the global `]}` close, and tokenizer-span masks are built from the real chat-template rendering so merged boundary tokens are still supervised.
- `compute_candidate_full_entry_logprob()` scores coord tokens with coord-vocab normalization and non-coord tokens with full-vocab logprob; the production candidate-balanced path optimizes `loss/candidate_balanced`, while PEM/threshold-loss is a legacy path.
- the report found that the strongest regression pattern is not a raw parsing crash but a combination of malformed continuation shape, empty parsed predictions, and recall collapse in crowded/high-GT-count images.
- the best artifact in the current family was `schemafix/v0 step100` (`bbox_AP=0.4025`, `pred_total=1191`, `empty_pred=8`), but it degraded by step 200 (`bbox_AP=0.3140`, `pred_total=764`, `empty_pred=54`).
- `boundaryfix` still had a schema-start bug at step 100: 103/200 traces began as bare object entries instead of the `{"objects": ...}` wrapper.
- later schema/gate runs started with the correct `{"objects": ...}` wrapper in all 200 traces, but many outputs still parsed empty because the generated text was malformed or overlong.

References:
- [1] user request: analyze `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_pem_close_suppress/setcont-coco1024-sota1332-pem-close-suppress/v2-20260426-114031/logging.jsonl` under `configs/stage1/set_continuation/production.yaml`; explain why loss drops but mAP falls from ~0.38 to ~0.16.
- [2] run artifacts: `resolved_config.json`, `experiment_manifest.json`, `eval_data_provenance.json`, `logging.jsonl`, `eval_detection/step_0000100/{metrics.json,infer_summary.json,confidence_postop_summary.json,gt_vs_pred.jsonl,pred_token_trace.jsonl}`.
- [3] key code paths: `src/trainers/stage1_set_continuation/branch_encoder.py::encode_set_continuation_branch`, `src/trainers/stage1_set_continuation/losses.py::{compute_candidate_full_entry_logprob,compute_mp_pem_losses,compute_bidirectional_token_gate_loss}`, `src/trainers/stage1_set_continuation/sampling.py::{sample_subset_and_candidates,select_tail_protected_candidates}`, `src/trainers/stage1_set_continuation/trainer.py::_process_sample`, `src/callbacks/stage1_detection_eval.py::Stage1DetectionEvalCallback.on_evaluate`.
- [4] helpful artifact written for later reference: `temp/stage1_set_continuation_training_report_20260428.md`.

## Task 1 (continued): Distill the root symptom pattern

Outcome: partial

Preference signals:
- the user explicitly asked whether the issue is a computational/implementation error or genuine degradation -> future similar analyses should separate code-path evidence from model-behavior evidence.

Key steps:
- evaluated cross-run metrics and parsed-output hygiene, including empty prediction rates, wrapper-vs-bare-object starts, max-token truncation, extra-object-after-close patterns, and GT-count bucket breakdowns.
- compared reference coord-token `checkpoint-1332` evals (`rp1.00`, `rp1.05-ish`, `rp1.10`) with the newer continuation runs.
- extracted that the training losses steadily decrease across runs, while eval AP/mAP is not monotonic and often collapses in crowded buckets.

Failures and how to do differently:
- the evidence supports “real degradation plus decode/serialization path issues” more strongly than “single computational bug,” but it does not fully isolate one root cause; future agents should test the boundary between generation, parsing, and eval separately.

Reusable knowledge:
- the current rollouts show strong bucket sensitivity: empty parsed predictions rise sharply for rows with 7+ GT objects in several later runs, which is a useful pivot signal for future debugging.
- malformed outputs observed in the traces include object emission after `]}`, extra top-level keys after the `objects` array, and repeated `<|endoftext|>` tails; these are likely more diagnostic than aggregate AP alone.

References:
- [5] metrics contrast: `schemafix/v0 step100` had `bbox_AP=0.4025`, `pred_total=1191`, `empty_pred=8`; `schemafix/v0 step200` dropped to `bbox_AP=0.3140`, `pred_total=764`, `empty_pred=54`; `bidirgate_warmup10/v0 step100` had `bbox_AP=0.1927`, `pred_total=663`, `empty_pred=97`.
- [6] parse hygiene summary: `boundaryfix` step100 had 103 bare-object starts; later schema/gate runs had correct wrappers but high empty-pred rates and malformed continuations.

