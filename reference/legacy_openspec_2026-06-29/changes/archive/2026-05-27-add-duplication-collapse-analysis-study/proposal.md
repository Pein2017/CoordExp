## Why

Same-desc duplication collapse is now showing up across existing Stage-1 and Stage-2 checkpoints, and the current evidence is fragmented across infer artifacts, offline diagnostics, and dated progress notes. The latest onset-localized probe evidence suggests a more specific mechanism than generic "attention ignores vision": the model can allocate substantial late-middle-layer attention mass to vision tokens, but the final decision layers often hand control back to recent generated history and prior coord tokens, especially at coord emission steps. Healthy same-desc controls can still show some history overwrite while avoiding collapse, so the stronger separator now appears to be whether `coord_x1` / `coord_y1` rapidly escape the previous or local bbox neighborhood. We need one inference-only, fixed-checkpoint study contract that can test whether the primary failure mechanism is a coordinate basin / weak local escape barrier, late-layer overwrite of visual grounding, continuation calibration, train-inference mismatch, or some mixed combination before we touch the objective again.

The next missing piece is causal comparison across training regimes. The study should not stop at saying that duplicated checkpoints attend to prior coord tokens. It needs to explain why that shortcut becomes available after Stage-1 soft coordinate supervision and why CE-like branches do not appear to enter the same basin as easily. Base pretrained checkpoints are not part of this causal comparison because they do not support the expanded coordinate-token space used by the current training and inference pipeline. The current on-disk `merged` inventory also matters: the strongest CE-like checkpoints available today are soft-disabled continuations, not clean from-scratch pure-CE baselines, so the study must preserve that provenance rather than silently relabeling them.

## What Changes

- Introduce a fixed-checkpoint, inference-only duplication-collapse analysis study that compares all locally available Stage-1 and Stage-2 checkpoints whose names include `merged`, without launching new training.
- Standardize the study manifest for checkpoint inventory, artifact provenance, prompt/order controls, subset selection, and decode settings so results are reproducible across Stage-1 and Stage-2 families.
- Fix the authoritative inference configuration to the current production-style HuggingFace rollout baseline:
  - `temperature = 0.0`,
  - `top_p = 0.9`,
  - `repetition_penalty = 1.05`,
  - `max_new_tokens = 3084`,
  - `seed = 42`,
  - no temperature sweep,
  - no repetition-penalty sweep,
  - and no decode-configuration sweep as the primary study axis.
- Allow limited secondary decode diagnostics such as `top_k` or `top_p` variants only after the baseline failure has been reproduced under the fixed authoritative profile, and require those diagnostics to remain subordinate to the main mechanism study.
- Require a deep mechanism probe for same-desc duplication that captures:
  - a rollout-first reproduction of the failure,
  - a deterministic re-forward of the emitted prefix at the duplication onset,
  - the exact decoding step where duplication first emerges,
  - logits evolution immediately before and after that step,
  - attention probabilities across layers and heads,
  - inter-layer signal propagation inside Qwen3-VL,
  - LLM-to-vision attention concentration at onset,
  - late-layer shifts in attention mass from visual tokens toward recent generated tokens or prior coord spans,
  - and any cross-step feedback loop that reinforces repeated object emission.
- Require new surgery-like probing tools for Qwen3-VL as a first-class deliverable:
  - mandatory LLM-tower probes for logits, decode scores, hidden or residual summaries, and per-layer or per-head attention,
  - mandatory vision-facing probes through LLM-to-visual-token attention summaries,
  - and native vision-tower summaries when they can be exposed cleanly, with explicit missingness markers when they cannot.
- Treat upstream-local dependency exploration as in-scope for instrumentation design:
  - local `transformers` sources under `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`,
  - local `ms-swift` sources under `/data/ms-swift/swift/`,
  - and probe development may inspect those directories directly when Qwen3-VL rollout or hook behavior depends on upstream implementation details.
- Require a multi-layer evidence program for duplication diagnosis:
  - rollout canary metrics on existing `gt_vs_pred.jsonl` / `pred_token_trace.jsonl` artifacts,
  - fixed-setting rollout reproduction under the authoritative baseline,
  - deterministic re-forward surgery at the duplication onset,
  - fixed-setting prefix-conditioned continuation probes,
  - teacher-forced / controlled counterfactual probes that compare failure and non-failure trajectories at the duplication onset,
  - and targeted inference-time intervention probes that can selectively weaken recent-history shortcuts or selectively strengthen late-layer visual access without changing checkpoint weights.
- Require the study to report earliest observable precursors of duplication collapse, including continuation-vs-close imbalance, same-desc burst onset, coord copy-mass, coord entropy / sharpness, token competition, and cross-step state accumulation.
- Require the study to prioritize `predicted_object` versus `exact_duplicate` over `gt_next` versus `exact_duplicate` as the main causal comparison when rollout produced a usable next object, and to treat `coord_x1` / `coord_y1` escape from the previous/local neighborhood as the primary mechanism surface.
- Require the study to test whether duplication becomes unavoidable when final-layer state stops preserving visual evidence and instead resolves coord choices from recent text/history, especially prior emitted coord tokens.
- Require the study to treat late history-overwrite as a secondary amplifier by default and to promote it to a primary-cause claim only when controlled interventions show stronger evidence than the current coord-basin signals.
- Require the study to explicitly compare available checkpoint families that represent:
  - clean pure-CE Stage-1 behavior when such checkpoints are locally available,
  - otherwise CE-like or `coord_soft_ce_w1`-disabled continuation branches,
  - and soft-coordinate-supervised Stage-1 behavior,
  so the final diagnosis can explain why duplication is absent or weakened in one regime and emerges after soft coordinate supervision in the other.
- Require family-comparison outputs to distinguish two different reasons duplication may be absent:
  - the model retained sufficiently sharp coord discrimination to avoid the local copy basin,
  - or the model avoids duplication only by drifting into some other rollout failure mode such as conservative under-enumeration or semantic misbinding.
- Require the checkpoint inventory to record, for each `merged` family:
  - whether `coord_soft_ce_w1` is enabled, disabled, or unknown,
  - any obvious parent-checkpoint provenance,
  - whether a paired infer artifact already exists,
  - and the best current probe surface handle.
- Require the study to split the discovered `merged` cohort into:
  - ready-to-probe families with directly usable infer surfaces,
  - and fresh-inference-needed families that exist on disk but do not yet share the same probe-ready artifact surface.
- Require the report to avoid overstating the available CE comparison:
  - a soft-disabled continuation MAY be used as a practical CE-like reference,
  - but it MUST NOT be presented as a clean pure-CE baseline unless config provenance supports that claim.
- Require multi-checkpoint and multi-case evidence before promoting a mechanism-level hypothesis from a hero-case finding to a checkpoint-family conclusion.
- Require cohort-level summaries that preserve heterogeneity across cases rather than overfitting conclusions to one or two obvious catastrophic samples.
- Require follow-up cohort analyses that explicitly test:
  - whether duplicated cases remain separable from healthy same-desc controls after matching or stratifying on same-class crowding signals,
  - and whether suspected class-prior effects survive after conditioning on local crowding rather than only raw class identity.
- Define a recall-heavy duplication-case miner whose primary job is to avoid missed failures:
  - same-desc bursts with similar box size,
  - narrow spatial drift around an existing object cluster,
  - and obvious repeated-localization regimes even when strict IoU criteria would miss them.
- Define the checkpoint cohort around all locally available Stage-1 and Stage-2 checkpoint or artifact roots that are compatible with the current pipeline, but include only entries whose checkpoint or artifact names contain `merged`.
- Adopt an exploratory, iterative research loop that can refine hypotheses, add targeted probes, and prioritize curiosity-driven mechanism discovery over fixed heuristic explanations while keeping all work inference-only.
- Explicitly extend the exploratory loop to include intervention design and counterfactual probing inside Qwen3-VL, including:
  - late-layer history attenuation,
  - late-layer visual-token biasing,
  - previous-object span suppression,
  - and phase-specific coord-step interventions,
  with the goal of testing which internal shortcuts make duplication self-reinforcing.
- Explicitly extend the exploratory loop to include rollout-perturbation and rollout-interpolation probes that approximate randomized local context without retraining:
  - perturb the immediate pre-collapse prefix object,
  - interpolate that prefix object toward the alternative same-desc target,
  - drop or replace the most recent same-desc object when the onset is local,
  - and verify whether the rollout escapes the duplicate basin under the same fixed decode contract.
- Require the next experiment round to prioritize:
  - `coord_x1` / `coord_y1` escape as the primary diagnostic surface,
  - `predicted_object` versus `exact_duplicate` as the primary causal comparison,
  - coordinate-basin / weak-local-escape interpretations ahead of purely attention-based stories,
  - and late history-overwrite as a secondary amplifier unless stronger controlled evidence appears.
- Allow a separate root-level research workspace for custom instrumentation, controlled experiments, and isolated artifacts, without disturbing the main training and analysis paths.
- Keep the change analysis-only: no new training flows, no objective redesign in this change, no checkpoint mutation, no evaluator contract changes, and no heuristic duplicate suppression or decode-policy mitigation work as the primary deliverable.

## Capabilities

### New Capabilities
- `duplication-collapse-analysis-study`: inference-only, fixed-checkpoint study for diagnosing same-desc duplication collapse at the mechanism level under production-like low-temperature HuggingFace decoding.

### Modified Capabilities

## Impact

- Expected code impact centers on analysis and reporting surfaces such as `src/analysis/`, `configs/analysis/`, and supporting infer/eval artifact readers.
- The change may also add a separate root-level research workspace for custom probing tools, deep-probe experiment drivers, and isolated mechanism-analysis artifacts.
- The study will consume existing artifacts under `output/infer/`, `output/eval/`, and all locally available Stage-1 and Stage-2 checkpoint roots that are compatible with the current pipeline, filtered to checkpoint or artifact names containing `merged`.
- When available, the study should also preserve whether a checkpoint family is acting as:
  - a clean pure-CE reference,
  - a CE-like soft-disabled continuation,
  - or a soft-coordinate-supervised reference,
  because that family labeling is part of the causal question the study is meant to answer.
- The study may also depend on upstream-local source exploration in the active `ms` environment, especially `transformers` and `ms-swift`, when implementing or validating Qwen3-VL probe hooks.
- No public inference, evaluation, dataset, or training contracts are expected to change in this proposal; the main reproducibility impact is adding one canonical analysis workflow, a machine-readable case-bundle contract, and a thin report layer for onset-localized duplication diagnosis.
- Verification should be artifact-driven: manifest checks, deterministic onset localization, case-bundle outputs, and study reports, not new training runs.
