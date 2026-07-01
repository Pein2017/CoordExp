## 1. Study Scaffold

- [x] 1.1 Add a manifest-driven analysis entrypoint under `src/analysis/` plus a YAML config family under `configs/analysis/` for the duplication-collapse study, with no new CLI flags.
- [x] 1.2 Implement strict manifest resolution for checkpoint aliases, checkpoint paths, prompt/order controls, subset provenance, and the exact HF baseline decode contract: `temperature = 0.0`, `top_p = 0.9`, `repetition_penalty = 1.05`, `max_new_tokens = 3084`, and `seed = 42`.
- [x] 1.3 Add fail-fast validation that rejects missing checkpoints, missing subset inputs, and attempts to route the study through training-only surfaces.
- [x] 1.4 Add isolated-workspace support so deep-probe code and machine-readable case bundles can live under a separate root-level research directory while remaining checkpoint-read-only.

## 2. Artifact Audit And Comparison Matrix

- [x] 2.1 Implement an artifact-audit stage that reuses existing `gt_vs_pred.jsonl`, `pred_token_trace.jsonl`, `pred_confidence.jsonl`, and evaluator outputs when they already exist for a checkpoint.
- [x] 2.2 Add checkpoint-matrix metadata fields for stage, objective family, geometry regime, rollout-training regime, and known interpretation confounds.
- [x] 2.3 Emit a checkpoint inventory artifact that makes unavailable comparison cells explicit instead of silently dropping them.
- [x] 2.4 Add deterministic, recall-heavy case-selection logic that prioritizes clear duplication failures from all locally available Stage-1 and Stage-2 historical artifacts, filters the checkpoint or artifact names to entries containing `merged`, and records the exact source artifact and selection reason for each case.
- [x] 2.5 Extend checkpoint-family metadata and report inputs so the study can explicitly compare:
  - clean pure-CE Stage-1 references when they are locally available,
  - otherwise CE-like or `coord_soft_ce_w1`-disabled continuation branches,
  - and soft-coordinate-supervised Stage-1 references,
  while recording missing-family gaps and excluding token-incompatible base-pretrained checkpoints from the formal family comparison.
- [x] 2.6 Extend the checkpoint inventory artifact to record, for every discovered `merged` family:
  - `coord_soft_ce_w1_state`,
  - parent-checkpoint provenance when known,
  - `has_infer_artifact`,
  - and `best_probe_surface`,
  and split the cohort into ready-to-probe versus fresh-inference-needed families.

## 3. Fixed-Setting Deep Probe

- [x] 3.1 Wire a fixed-setting rollout-reproduction stage that replays selected cases with the HuggingFace backend under the exact baseline decode contract, and require baseline reproduction before any secondary `top_k` or `top_p` diagnostic is interpreted.
- [x] 3.2 Implement rollout-first, re-forward-second duplication surgery so each failure case records the onset object index, onset token span, onset field-phase marker, and the immediately preceding and following steps for comparison.
- [x] 3.3 Add mandatory deep-probe instrumentation that captures raw forward logits, decode-processed token competition, and per-layer / per-head attention summaries at and around the duplication-onset step.
- [x] 3.4 Add mandatory LLM hidden-state or residual-stream summaries plus LLM-to-visual-token attention summaries, and add native vision-tower probe paths when feasible without making raw tensor dumps the default artifact.
- [x] 3.5 Add explicit probe-surface status outputs so skipped or partially available LLM-side, vision-facing, or native vision-tower evidence remains visible in every case bundle and final report.
- [x] 3.6 Add upstream-local dependency mapping for probe hooks, including any required exploration of local `transformers` or `ms-swift` sources, and record those dependency roots in manifests or case bundles when they influence instrumentation.
- [x] 3.7 Extend the probe outputs to preserve checkpoint-comparable layerwise group-mass summaries that can test the "vision consulted, then overwritten by recent history" hypothesis, including separate prior-coord and generated-history buckets when available.
- [x] 3.8 Add targeted intervention-probe helpers for late-layer visual-token biasing, late-layer prior-history attenuation, and coord-phase-only intervention windows, without mutating checkpoint weights.

## 4. Controlled Comparisons And Reporting

- [x] 4.1 Add a controlled-comparison stage that makes same-desc `gt_next` versus `exact_duplicate` mandatory when applicable, and otherwise records a documented fallback control trace.
- [x] 4.2 Extend the counterfactual outputs to record coordinate-distribution shape metrics, including entropy, top-1 probability, target-bin probability, previous-box probability, previous-box neighborhood mass, and expectation-versus-argmax summaries when logits are available.
- [x] 4.3 Add case-level reporting for earliest observable precursors, including prediction-count growth, `max_desc_count`, same-desc near-duplicate counts, continuation-sensitive signals, and duplication-onset internal-state summaries.
- [x] 4.4 Add a report builder that preserves separate conclusions for artifact audit, fixed-setting rollout reproduction, deterministic re-forward surgery, controlled counterfactual evidence, and deep onset probes.
- [x] 4.5 Add final summary tables that distinguish coordinate-dominant, continuation-dominant, internal-state-dominant, mixed, and insufficient-evidence findings without overclaiming from one evidence layer.
- [x] 4.6 Ensure the final report explicitly treats FN injection as out of primary scope unless a direct recorded probe contradicts that assumption, and explicitly excludes heuristic duplicate suppression or decode-policy mitigation as the primary outcome of this change.
- [x] 4.7 Add cohort-level analysis outputs that aggregate mechanism signals across multiple checkpoints and diverse duplicated samples, clearly separate hero-case findings from checkpoint-family findings, and surface `coord_x1` / `coord_y1` escape metrics as first-class summaries.
- [x] 4.8 Add intervention-evaluation reporting that records both internal signal shifts and behavioral outcomes so "reduced duplication via undercounting" is not misread as a successful re-grounding intervention, with `predicted_object` vs `exact_duplicate` preferred when available.
- [x] 4.9 Add an explicit family-comparison report section that answers, for each available regime, why duplication is absent or present:
  - whether the checkpoint is a clean pure-CE baseline or only a CE-like proxy continuation,
  - whether it retains sharp coord discrimination that blocks the copy basin,
  - or whether soft coordinate supervision introduces a smooth local basin plus history-dominant final coord decisions.
- [x] 4.10 Add an explicit readiness split in the final report so ready-to-probe merged families and fresh-inference-needed merged families are reported separately rather than silently mixed.

## 5. Verification

- [x] 5.1 Add focused tests for manifest resolution, missing-checkpoint failure, onset detection, and probe-surface status recording.
- [x] 5.2 Add a small smoke config that runs the study on one or two existing checkpoints and a tiny fixed subset under the fixed HF decode contract, then verify the run emits a resolved manifest, machine-readable onset-probe bundles, and a final report under the main analysis root or isolated research workspace.
- [x] 5.3 Run targeted verification with `PYTHONPATH=. conda run -n ms python -m pytest` on the new analysis tests plus any touched infer/eval artifact readers, and record the exact commands in the change notes.
- [x] 5.4 Add a checkpoint-cohort study config and verification run that covers multiple available Stage-1 and Stage-2 `merged` checkpoints and a diverse set of historical duplication cases, then verify the run emits cohort-level summaries rather than only one-case reports.
- [x] 5.5 Add a targeted intervention smoke run on at least one anchored duplication case and verify that the output captures both attention-mass changes and candidate-ranking changes for each intervention.
- [x] 5.6 Add a family-comparison verification run, when the local checkpoint inventory supports it, that aligns either:
  - one clean pure-CE reference and one soft-coordinate-supervised reference,
  - or one CE-like soft-disabled continuation proxy and one soft-coordinate-supervised reference,
  under the same decode contract and verifies the report records the provenance caveat rather than overstating the CE-side baseline.
- [x] 5.7 Add a checkpoint-inventory verification check that confirms the emitted manifest/report records `coord_soft_ce_w1_state`, parent checkpoint when known, infer-artifact readiness, and the ready-to-probe versus fresh-inference-needed split.

## 6. Hypothesis Verification Extensions

- [x] 6.1 Update the study artifacts and follow-up configs so the next round explicitly covers:
  - matched crowding / class-prior cohort analysis,
  - rollout perturbation / interpolation without retraining,
  - and coord-split oracle probes centered on `coord_x1` / `coord_y1`.
- [x] 6.2 Add a follow-up analysis runner under `src/analysis/` plus config(s) under `configs/analysis/` that can:
  - aggregate local same-class crowding summaries for duplicated and healthy same-desc cases,
  - preserve class-family labels,
  - and emit machine-readable duplicate-versus-control summaries after crowding stratification.
- [x] 6.3 Add a no-retraining rollout-perturbation probe that edits the immediate pre-collapse prefix, reruns rollout under the same fixed decode contract, and records whether the model remains in the duplicate basin, escapes to a non-duplicate same-desc continuation, escapes semantically, or undercounts.
- [x] 6.4 Extend the controlled-comparison surface to preserve `predicted_object` versus `exact_duplicate` and coord-split oracle comparisons such as `x1/y1`-oracle and `x2/y2`-oracle when same-desc alternative targets are available.
- [x] 6.5 Run the crowding / class-prior follow-up analysis on a diverse duplicated-plus-normal panel and save machine-readable summaries under the isolated research workspace.
- [x] 6.6 Run the rollout-perturbation and coord-split follow-up probes across multiple checkpoints and anchored duplication cases under the fixed decode contract, using parallel multi-GPU execution where useful.
- [x] 6.7 Record an executive task list and verification note that explains which hypotheses were tested, which artifacts support or weaken each hypothesis, and which items remain open.

## Verification Notes

- `PYTHONPATH=. conda run -n ms python -m py_compile src/analysis/duplication_collapse_analysis.py scripts/analysis/run_duplication_collapse_analysis.py tests/test_duplication_collapse_analysis.py`
- `PYTHONPATH=. conda run -n ms python -m pytest tests/test_duplication_collapse_analysis.py -q`
- `PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/smoke.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/family_compare.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/intervention_smoke.yaml`
- `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/bootstrap_stage1_softce_w1_core.yaml`
- `CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/bootstrap_stage1_adjrep_global_core.yaml`
- `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_collapse_analysis.py --config configs/analysis/duplication_collapse/bootstrap_stage1_hard_soft_ce_2b_core.yaml`
- `PYTHONPATH=. conda run -n ms python -m py_compile src/analysis/duplication_collapse_analysis.py src/analysis/duplication_followup.py scripts/analysis/run_duplication_followup.py`
- `PYTHONPATH=. conda run -n ms python -m pytest tests/test_duplication_collapse_analysis.py -q`
- `openspec validate add-duplication-collapse-analysis-study`
- `PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/crowding_class_panel.yaml`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_stage2.yaml`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_ce_ciou.yaml`
- `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_softce4b_shard_a.yaml`
- `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_softce4b_shard_b.yaml`
- `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_pure_ce_shard_a.yaml`
- `CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_pure_ce_shard_b.yaml`
- `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. conda run -n ms python scripts/analysis/run_duplication_followup.py --config configs/analysis/duplication_followup/prefix_perturb_center_shard_a.yaml`
