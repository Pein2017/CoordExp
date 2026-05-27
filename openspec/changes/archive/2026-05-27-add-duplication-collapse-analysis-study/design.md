## Context

CoordExp already has several partial analysis surfaces for rollout pathologies:

- free-generation artifacts under `output/infer/` and `output/eval/`,
- fixed-checkpoint decode studies under `output/analysis/`,
- Oracle-K temperature sweeps under `output/oracle_k/`,
- teacher-forced proxy scorers in `src/analysis/unmatched_proposal_verifier.py`,
- and rollout / duplication diagnostics in `progress/diagnostics/`.

What is missing is one canonical study that answers a narrower question end-to-end:

- when same-desc duplication collapse appears, is the first failure signal mainly:
  - overly smooth or high-entropy coordinate distributions,
  - weak continuation / stop calibration,
  - self-prefix amplification under rollout,
  - attention or inter-layer concentration on a repeated region,
  - cross-step state accumulation inside Qwen3-VL,
  - late-layer overwrite of visual grounding by recent generated tokens,
  - or some mixed combination of those factors?
- and, across available checkpoint families that are compatible with the current
  expanded coord-token pipeline, why does this collapse:
  - stay absent or weaker under CE-like branches,
  - but emerge after soft coordinate supervision such as soft-CE / W1?

The current operational constraint is strict:

- no further training is allowed,
- only existing checkpoints may be analyzed,
- and conclusions must come from evaluation or inference-time behavior.

That means the study must treat existing checkpoint families and existing artifacts as the ablation surface, rather than proposing new loss sweeps that require fresh optimization.

The current research posture is also explicit:

- this is a hard, exploratory mechanism problem,
- the first version should bias toward curiosity and failure discovery rather than conservative surface reporting,
- and the study should prefer high-recall failure mining over overly strict duplicate definitions that might hide the earliest onset regime.

## Goals / Non-Goals

**Goals:**
- Define one reproducible, manifest-driven study for diagnosing same-desc duplication collapse across existing Stage-1 and Stage-2 checkpoints.
- Reuse existing infer/eval artifacts when possible, and run only targeted inference-time probes when an evidence layer is missing.
- Separate artifact-level rollout evidence from teacher-forced counterfactual evidence so the report can distinguish train-inference mismatch from latent scoring pathologies.
- Require explicit coordinate-distribution measurements, including entropy / sharpness, target probability, and previous-box copy mass, rather than relying only on final AP or duplicate counts.
- Make early `coord_x1` / `coord_y1` neighborhood escape the primary diagnosis surface and treat whole-object averages as supporting summaries.
- Require deep, onset-focused instrumentation that can capture attention probabilities, logits evolution, and cross-step state at the moment duplication begins.
- Require the study to test whether vision evidence is present in late-middle layers but overwritten by recent generated tokens or prior coord spans in the final decision layers.
- Require rollout-first, surgery-like re-forward probing as the canonical workflow for deep cases.
- Require Qwen3-VL probe tooling that is strong on the LLM side and vision-facing attention side, while attempting native vision-tower summaries when feasible.
- Explicitly allow upstream-local exploration of `transformers` and `ms-swift` in the active `ms` environment when probe behavior depends on upstream implementation details.
- Require report outputs that surface the earliest observable precursors of collapse, not just the terminal catastrophic samples.
- Require checkpoint-family conclusions to be supported by a diverse cohort of cases rather than one or two hero examples.
- Require matched crowding and class-prior follow-up analyses so duplicated cases can be compared against healthy same-desc controls within similar local same-class crowding regimes.
- Require rollout-perturbation or rollout-interpolation probes that approximate randomized local context without retraining by editing the immediate pre-collapse prefix and checking whether rollout escapes the duplicate basin.
- Require the final diagnosis to compare available training-signal regimes that
  are token-compatible with the current pipeline, especially:
  - clean pure-CE Stage-1 checkpoints when they are locally available,
  - otherwise CE-like or `coord_soft_ce_w1`-disabled continuation branches,
  - and Stage-1 checkpoints with soft coordinate supervision,
  so the study can explain why the duplication basin is introduced rather than
  merely describing what duplicated checkpoints look like.
- Require the report to distinguish two reasons duplication may be absent:
  - preserved coord discrimination that prevents the local copy basin from
    forming,
  - versus escape into a different failure mode such as conservative
    under-enumeration or semantic drift.
- Require the checkpoint inventory and family-comparison report to preserve the
  actual on-disk provenance of each comparison family, including:
  - whether `coord_soft_ce_w1` is enabled, disabled, or unknown,
  - whether the checkpoint is a clean baseline or a continuation,
  - whether a paired infer artifact already exists,
  - and the best current probe surface available without additional inference.
- Require the study to split the discovered `merged` cohort into:
  - ready-to-probe families that already have direct infer or eval artifact
    surfaces,
  - and fresh-inference-needed families that exist on disk but need matching
    rollout artifacts before they can join the same comparison set.
- Keep the workflow config-first under `configs/analysis/` and avoid new CLI flags.
- Keep the authoritative inference configuration fixed to the current production-style HuggingFace rollout baseline:
  - `temperature = 0.0`,
  - `top_p = 0.9`,
  - `repetition_penalty = 1.05`,
  - `max_new_tokens = 3084`,
  - `seed = 42`.
- Allow an isolated root-level research workspace for custom probes and deep-probe artifacts.

**Non-Goals:**
- Do not redesign Stage-1 or Stage-2 objectives in this change.
- Do not add any new training workflow, checkpoint mutation path, or optimizer sweep.
- Do not change the infer/eval artifact contract, rollout parser, or detection evaluator semantics.
- Do not force one single causal story when the evidence is mixed; the study should preserve competing signals rather than flatten them into one scalar.
- Do not require every checkpoint to support every probe; unavailable evidence should be surfaced explicitly rather than silently synthesized.
- Do not run temperature sweeps or repetition-penalty sweeps in the authoritative study path.
- Do not treat `top_k` or `top_p` as the primary explanation surface; any such diagnostics remain secondary to baseline reproduction.
- Do not center the diagnosis on FN injection; that factor remains out of primary scope unless a direct probe contradicts current evidence.
- Do not drift into heuristic mitigation, duplicate suppression, or decode-policy tweaking as the main output of this change.
- Do not promote a late-layer overwrite hypothesis to a checkpoint-family conclusion unless the effect repeats across multiple checkpoints and a diverse case cohort.
- Do not treat late history-overwrite as the primary cause when healthy controls show similar overwrite but stronger early coord escape.
- Do not treat crowded scenes or high-count classes as a sufficient explanation unless duplicated cases remain separable from healthy same-desc controls after matched crowding analysis.

## Decisions

### 1. Make the study fixed-checkpoint and inference-only by contract

Decision:
- The study will consume all locally available existing Stage-1 and Stage-2 checkpoints that are compatible with the current pipeline, plus existing or newly generated inference/eval artifacts derived from those checkpoints.
- The checkpoint-discovery rule for the canonical study cohort will filter to checkpoint or artifact names containing `merged`.
- The checkpoint inventory must record, for each discovered family:
  - stage,
  - family label,
  - `coord_soft_ce_w1` state when it can be inferred,
  - parent-checkpoint provenance when it is obvious from config,
  - whether a paired infer artifact is already available,
  - and the best current probe surface handle.
- The inventory output must also split the cohort into:
  - ready-to-probe families with matched infer/eval surfaces,
  - and fresh-inference-needed families that require rollout before fair
    mechanism comparison.
- The study will never launch training, mutate checkpoints, or rely on synthetic retraining ablations.
- Checkpoint comparison will therefore be observational: the “ablation matrix” is the set of already-available objective families, geometry variants, and rollout-training variants.

Rationale:
- This is the only design compatible with the current GPU constraint.
- It also keeps the causal claims honest: objective-level conclusions must be phrased in terms of checkpoint-family evidence, not imagined clean-room retrains.
- Using all locally available `merged` checkpoints widens the observational surface without opening the door to incompatible or partial artifacts that do not represent the deployed checkpoint family cleanly.
- Explicit provenance fields keep the comparison honest when the available CE-like references are continuations rather than clean from-scratch pure-CE baselines.

Alternatives considered:
- New training sweeps over CE / soft-CE / W1 / geometry / UL: rejected because it violates the operational constraint.
- Pure artifact-only analysis with no new inference-time probes: rejected because existing artifacts do not expose every needed sharpness / continuation diagnostic.

### 1b. Add matched crowding and class-prior follow-up audits

Decision:
- The study will add a follow-up cohort-analysis layer that computes local
  same-class crowding summaries for duplicated and healthy same-desc cases,
  including:
  - same-class GT count,
  - maximum same-class IoU,
  - nearest-neighbor center-distance summaries,
  - and per-class or per-scene-family labels when available from existing
    artifacts.
- The follow-up layer will explicitly test whether:
  - duplicated cases remain separable from healthy same-desc controls after
    matching or stratifying on local crowding,
  - and duplicate-prone class families remain enriched after conditioning on
    crowding rather than only raw class identity.

Rationale:
- Current evidence shows that many failure anchors are dense same-class
  clusters, but some low-cardinality pairs still drift.
- A matched crowding audit is the cleanest way to tell whether crowding is a
  trigger, a confounder, or a sufficient explanation.

Alternatives considered:
- Treat raw duplicate-label frequency as a class-prior verdict: rejected
  because high-count classes and crowded scenes are confounded.
- Treat annotation noise as the default explanation: rejected because the
  current artifact base does not yet show direct label-error evidence.

### 2. Fix the authoritative decode contract to an exact HF baseline

Decision:
- The study will use the HuggingFace backend as the authoritative rollout backend.
- The authoritative baseline decode contract will match the current production-style artifact profile:
  - `temperature = 0.0`,
  - `top_p = 0.9`,
  - `repetition_penalty = 1.05`,
  - `max_new_tokens = 3084`,
  - `seed = 42`.
- `top_k` is not part of the baseline contract and should default to the framework's unset behavior unless a secondary diagnostic explicitly overrides it.
- The study will not treat decode sweeps as part of the canonical workflow.
- Any limited `top_k` or `top_p` counterfactuals must occur only after a case has been reproduced under the exact authoritative baseline and must be labeled as secondary diagnostics.

Rationale:
- The stated goal is to explain why the current checkpoint duplicates under the settings we actually care about, not to chart a broad decode-response surface.
- HF gives the most direct access to token logits, attentions, hidden states, and deterministic rollout control.

Alternatives considered:
- vLLM as the primary deep-probe backend: rejected because probe control is weaker for onset-level internal instrumentation.
- Temperature or repetition sweeps as a first-class study axis: rejected because they would shift the study away from mechanism diagnosis under production-like decoding.

### 3. Make rollout-first, re-forward-second surgery the canonical deep-probe workflow

Decision:
- The study will preserve four evidence layers:
  1. artifact audit / rollout canary,
  2. fixed-setting controlled rollout reproduction,
  3. deterministic re-forward surgery on the exact emitted prefix and onset window,
  4. controlled continuation or teacher-forced counterfactual probes.
- Each layer will record whether it was executed, skipped, or partially available.
- The final report will preserve which conclusions come from which layer.
- The deep-probe path must always reproduce the failure in rollout first, then re-forward the exact generated prefix and onset window before adding any further controls.

Rationale:
- Duplicate collapse can look similar in terminal images while arising from different mechanisms.
- Artifact canaries capture actual rollout behavior.
- Fixed-setting rollout reproduction captures the exact production-like decoding regime we need to explain.
- Deterministic re-forward surgery is the cleanest way to inspect logits, attentions, and state without changing the decoded history we are trying to explain.
- Controlled continuation and teacher-forced probes test whether the model already under-scores `gt_next` versus duplicate candidates under controlled context.
- Deep onset probes test whether the first repeated emission is preceded by attention concentration, token-competition collapse, or cross-step amplification.

Alternatives considered:
- One monolithic “duplicate score” study: rejected because it would blur causal interpretation.
- Teacher-forced-only study: rejected because it hides train-inference mismatch.
- Rollout-only study: rejected because it cannot isolate coordinate scoring under matched same-desc candidates.

### 4. Use a recall-heavy duplicate-family miner and a normative onset rule

Decision:
- The study will use a recall-heavy duplicate-family miner whose primary bias is “prefer false positives over missed early failures.”
- A candidate duplicate-family onset object is any emitted object that:
  - matches a prior object description within the current repeated-desc regime,
  - and either overlaps strongly with a prior object, or has similar bbox size with narrow local drift around the same object cluster.
- The object-level onset is the earliest emitted object that satisfies the duplicate-family detector.
- The token-level onset is the first description token of that onset object.
- The study will also record a field-phase onset marker with values such as:
  - `continue_or_open`,
  - `desc`,
  - `coord_x1`,
  - `coord_y1`,
  - `coord_x2`,
  - `coord_y2`,
  - `close`.
- Ties are broken by earliest object index, then earliest token index.
- For cases where the prefix object and the ground-truth next object share the same description, the canonical counterfactual set will include:
  - `gt_next`,
  - `exact_duplicate`,
  - and optional spatial jitters around the previous object.
- These same-desc comparisons will be treated as the primary coordinate-isolating probe because semantic-token differences are removed by construction.
- The study will record coordinate-only score margin, full score margin, and per-slot coordinate-distribution summaries for those candidates.
- When rollout yields a concrete next object at the same prefix, the canonical comparison SHOULD prioritize `predicted_object` versus `exact_duplicate` ahead of `gt_next` versus `exact_duplicate`, because it measures the actual rollout escape decision rather than only the teacher-aligned target.

Rationale:
- The strongest hypothesis under review is that duplication collapse is driven by coordinate-distribution smoothness / ambiguity, but the failure must first be found reliably.
- Same-desc GT-next vs exact-duplicate is a useful teacher-aligned control, but the actual generated next object is the cleaner causal probe for whether rollout escaped the duplicate basin.
- Existing bad cases like the `broccoli` and `apple` loops are already in this form.

Alternatives considered:
- Generic matched-vs-unmatched proposal scoring only: rejected because semantic differences can dominate the comparison.
- Exact-overlap-only duplicate candidates: rejected because prior diagnostics show the failure often lives in local same-desc drift rather than perfect copies.

### 5. Make same-desc controls and expectation-versus-argmax analysis mandatory when applicable

Decision:
- The study will never use coord entropy alone as the sole explanation for duplication collapse.
- Whenever the onset case supports same-desc controls, the deep-probe package must include `gt_next` versus `exact_duplicate` under the same prefix.
- Whenever rollout produced a concrete onset-local continuation, the deep-probe package must also include `predicted_object` versus `exact_duplicate`, and the report should treat that comparison as primary unless a case-specific limitation is recorded.
- Whenever logits are available for the coordinate slots, the deep-probe package must include both expectation-style and argmax-style summaries and explicitly record any disagreement between them.
- Each case report will preserve both:
  - continuation structure signals such as close-vs-continue imbalance or post-object continuation preference,
  - coordinate-distribution signals such as entropy, top-1 probability, target-bin probability, previous-box copy mass, and expectation / argmax disagreement,
  - and internal-state signals such as per-layer / per-head attention probabilities, logits trajectories, hidden-state or residual drift, and cross-step accumulation.
- The study summary will classify evidence as:
  - coordinate-dominant,
  - continuation-dominant,
  - internal-state-dominant,
  - mixed,
  - or insufficient evidence.

Rationale:
- Existing evidence already shows that some shallow duplicate basins can be escaped by decode penalties, while some deep basins are driven by almost-certain continuation.
- A pure “entropy control” explanation is therefore too strong unless continuation signals are measured and ruled out.
- Current healthy-vs-failure controls also show that history-overwrite is not sufficient by itself; the stronger separator is whether `coord_x1` / `coord_y1` rapidly evacuate mass away from the previous/local neighborhood.

Alternatives considered:
- Encode one scalar “duplication sharpness index”: rejected because it would hide whether the failure starts in continuation or coordinates.

### 6. Treat late-layer overwrite of visual grounding as a first-class hypothesis and intervention target

Decision:
- The study will explicitly test a "visual consulted, then overwritten" hypothesis for duplicated cases.
- For each deep-probed case, attention summaries must preserve layerwise group masses so the report can detect whether:
  - visual-token access rises in late-middle layers,
  - then falls in the final decision layers,
  - while recent generated history or prior coord spans gain control.
- The study will treat prior emitted coord tokens as a separate shortcut source rather than merging them into generic history when the probe surface allows that separation.
- The study will add targeted inference-time intervention probes that do not modify checkpoint weights, including:
  - late-layer attenuation of attention to previous-object text or coord spans,
  - late-layer positive bias toward visual-token groups,
  - phase-specific interventions that activate only at `coord_x1`, `coord_y1`, `coord_x2`, or `coord_y2`,
  - and paired controls that compare "boost vision" against "suppress prior-object history".
- The study will evaluate intervention success using candidate ranking shifts, not only attention-mass changes:
  - whether `predicted_object` or `gt_next` begins to outrank `exact_duplicate`,
  - whether previous-box copy mass falls,
  - and whether the rollout escapes the duplicate basin without simply collapsing into undercounting or semantic drift.
- Until stronger controlled evidence appears, the default interpretation of late history-overwrite will remain "secondary amplifier" rather than "primary cause."

Rationale:
- Current repaired onset probes show that visual-token mass can be substantial in late-middle layers at the duplicate onset, yet final coord decisions are still history-dominant.
- That pattern is more specific and more actionable than a generic claim that the model ignores vision.
- It also suggests the cleanest intervention surface is the final arbitration stage rather than global vision amplification.

Alternatives considered:
- Treat all history tokens as one undifferentiated group: rejected because prior coord spans appear to be a particularly important shortcut.
- Rely on raw attention mass alone as the intervention outcome: rejected because candidate ranking and rollout behavior are the stronger causal readouts.

### 6b. Approximate randomized local context with rollout perturbation and interpolation instead of retraining

Decision:
- The study will add a follow-up rollout-perturbation layer that edits the
  immediate pre-collapse prefix rather than training a random-order control
  checkpoint.
- For anchored same-desc collapse cases, the perturbation set SHOULD include:
  - dropping the most recent same-desc source object when legal,
  - replacing that source object with a same-desc alternative such as
    `gt_next` when available,
  - interpolating the source-object box toward the same-desc alternative,
  - and source-box jitter variants that preserve desc identity while changing
    local geometry.
- Each perturbation branch MUST be evaluated under the same fixed decode
  contract as the baseline case.
- The perturbation output MUST record whether rollout:
  - remains in the same duplicate basin,
  - escapes to a non-duplicate same-desc continuation,
  - escapes via semantic drift,
  - or collapses into early stop / undercount.

Rationale:
- The user explicitly prefers a no-retraining approximation to randomized
  object-sequence effects.
- Local-prefix perturbation is the cleanest causal approximation because it
- directly tests whether the duplicate basin is rigid or escapable once the immediate
  anchor is changed.

Alternatives considered:
- Retraining a random-order Stage-1 control: rejected by the operational
  constraint.
- Interpreting attention-only interventions as a replacement for prefix
  perturbation: rejected because the current evidence points more strongly to
  a coord basin / local escape barrier than to attention alone.

### 6c. Add oracle-style coord-split probes to isolate early-slot causality

Decision:
- The study will add same-prefix coord-split probes that compare the exact
  duplicate against alternatives such as:
  - `x1/y1` from a same-desc alternative target with `x2/y2` kept from the
    duplicate candidate,
  - `x1/y1` from the duplicate with `x2/y2` from the alternative same-desc
    target,
  - and `predicted_object` versus `exact_duplicate` whenever rollout produced a
    concrete continuation.
- The report will treat these coord-split probes as stronger evidence for
  early-slot causality than whole-object averages.

Rationale:
- Current evidence consistently points to `coord_x1` / `coord_y1` as the point
  where duplication becomes difficult to escape.
- Coord-split probes let the study test whether the early slots are the real
  branch point instead of only correlated summary features.

Alternatives considered:
- Whole-object candidate scoring alone: rejected because it blurs the early
  coord-slot decision that appears to separate healthy from collapsed rollout.

### 7. Make the minimum deep-probe surface explicit for Qwen3-VL

Decision:
- The study must provide surgery-like probe helpers for Qwen3-VL that always capture:
  - raw forward logits and decode-processed scores on the LLM side,
  - per-layer or per-head attention summaries on the LLM side,
  - hidden-state or residual summaries on the LLM side,
  - and LLM-to-visual-token attention summaries.
- Native vision-tower self-attention or intermediate feature summaries are strongly preferred when the backend exposes them cleanly.
- If native vision-tower internals are unavailable for a run, the case bundle must record that absence explicitly instead of silently downgrading the probe.
- Per-case outputs should be machine-readable first, with a thinner markdown synthesis built on top.

Rationale:
- The user's brief explicitly prioritizes dynamic, surgery-like probes for both the vision-facing and language-facing sides of Qwen3-VL.
- A mandatory minimum surface keeps the study from collapsing into a shallow report with inconsistent telemetry.

Alternatives considered:
- LLM-only onset probes: rejected because they undershoot the stated research goal.
- Full raw-tensor dumps for everything: rejected because they are expensive and not necessary as the default artifact format.

### 8. Require checkpoint-family and cohort-level evidence before promoting a mechanism claim

Decision:
- The study will separate hero-case analysis from cohort-level conclusions.
- A hero-case finding may motivate a hypothesis, but checkpoint-family conclusions must additionally include:
  - multiple checkpoints when the relevant families are available,
  - multiple cases per checkpoint family when historical artifacts support them,
  - and explicit diversity in duplicated descriptions or scene structure whenever available.
- The report will preserve case-level distributions and checkpoint-family summaries for key signals such as:
  - late-layer visual-token mass,
  - late-layer generated-history mass,
  - prior-coord shortcut mass,
  - candidate score margins,
  - and intervention effect sizes.
- The report must label hypotheses as:
  - hero-case only,
  - repeated across a checkpoint cohort,
  - or repeated across multiple checkpoint families.
- Where sample counts are large enough, the report should include simple distributional or bootstrap-style uncertainty summaries rather than only point estimates.

Rationale:
- Duplication collapse is heterogeneous across classes and scenes.
- The study is explicitly exploratory, so it needs room for hero-case discovery, but it also needs a guardrail against overfitting claims to one memorable sample.

Alternatives considered:
- Hard-code one fixed sample count threshold for every claim: rejected because available case counts differ across checkpoint families.
- Leave cohort support implicit: rejected because it would encourage over-reading the first successful probe.

### 9. Make the training-signal comparison explicit rather than implicit

Decision:
- The study will treat the pure-CE / soft-coordinate-supervised comparison as a
  first-class causal question rather than a side note.
- Base pretrained checkpoints are out of scope for this causal comparison
  because they do not support the expanded coord-token space and are not
  execution-compatible with the current Stage-1 training and inference path.
- When the compatible families are available, the study will align them on:
  - decode contract,
  - prompt and field-order controls,
  - case mining rules,
  - and onset-localized probe outputs.
- If a clean pure-CE Stage-1 checkpoint is not locally available, the study may
  use a CE-like or `coord_soft_ce_w1`-disabled continuation branch as a
  practical comparison family, but the report must label it explicitly as a
  continuation rather than a clean pure-CE baseline.
- The report must explicitly answer, for each available family:
  - whether duplication is absent because coord decisions remain sharp enough to
    avoid copying the previous object,
  - whether duplication is absent only because the rollout fails differently,
  - or whether duplication is present because same-desc continuation combines
    with a smooth local coord manifold and late-layer history overwrite.
- The study will therefore treat "attention collapses onto prior coord tokens" as an onset symptom, then test which training signals made that shortcut cheap or dominant.

Rationale:
- The user's question is not only "what happens when duplication starts?" but
  also "what changed in Stage-1 that made this basin exist at all?"
- A duplicated checkpoint can look history-dominant for many reasons. Without
  an explicit family comparison, the study cannot separate:
  - preserved sharp coord discrimination,
  - alternative non-duplication failure modes,
  - and objective-induced coordinate smoothness.
- The current merged inventory already shows that the practical CE-like
  references on disk are soft-disabled continuations, so the design needs an
  explicit provenance rule or the study will accidentally overclaim the purity
  of the comparison.

Alternatives considered:
- Leave the family comparison to the final narrative only: rejected because it would make the most important causal question optional.
- Use base pretrained checkpoints as a formal comparison family: rejected
  because they do not share the expanded coord-token space and are not
  compatible with the current training and inference pipeline.

### 10. Use an isolated research workspace for invasive probes while keeping the main path config-first

Decision:
- The study may create a separate root-level research workspace for deep-probe code, instrumentation helpers, and large intermediate artifacts.
- The workspace will remain analysis-only and checkpoint-read-only.
- Canonical manifests and final report outputs can still point back to main-repo artifacts and checkpoints.

Rationale:
- Attention capture, hidden-state tracing, and per-step deep probes are more invasive than the existing analysis helpers.
- Keeping that work in an isolated workspace reduces blast radius and keeps the main codebase clean while the mechanism study is still exploratory.

Alternatives considered:
- Force all deep-probe code directly into the existing analysis tree immediately: rejected because the probing surface is still iterative and may churn quickly.

### 11. Reuse existing analysis primitives where possible rather than introducing a bespoke framework

Decision:
- The study should compose existing primitives where possible:
  - `src/analysis/small_object_duplication_diagnostics.py`
  - `src/analysis/unmatched_proposal_verifier.py`
  - `scripts/run_infer.py`
  - `scripts/postop_confidence.py`
  - `scripts/evaluate_detection.py`
- New work, if needed, should mostly be orchestration, manifest normalization, and report synthesis rather than a full new probing stack.

Rationale:
- The repo already contains targeted decode, prefix, and teacher-forced scoring building blocks.
- Reusing them lowers reproducibility risk and keeps the OpenSpec grounded in real code paths.

Alternatives considered:
- Build an entirely new stand-alone study script: rejected because it would duplicate checkpoint loading, prompt handling, parsing, and scoring contracts that already exist.

### 12. Treat local upstream sources as valid exploration surfaces for probe design

Decision:
- The study may inspect locally installed upstream dependencies when Qwen3-VL instrumentation depends on implementation details outside the CoordExp tree.
- The primary upstream-local dependency roots for this study are:
  - `transformers` at `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`,
  - `ms-swift` at `/data/ms-swift/swift/`.
- Manifest or case-bundle metadata should record when a probe relied on upstream-local behavior or hook placement derived from those trees.

Rationale:
- Attention hooks, hidden-state capture, and multimodal token plumbing often depend on upstream model wrappers rather than only repo-local code.
- Making these dependencies explicit keeps the research workflow honest and reproducible.

Alternatives considered:
- Treat upstream behavior as a black box and probe only via repo-local wrappers: rejected because it would make some Qwen3-VL instrumentation fragile or under-specified.

## Risks / Trade-offs

- [Risk] Existing checkpoint families are not perfectly controlled ablations because model size, init path, and objective differ together. → Mitigation: require the report to label confounds explicitly and prefer within-family comparisons when available.
- [Risk] Some saved artifacts lack the telemetry needed for every precursor metric. → Mitigation: make evidence-layer availability explicit and allow targeted inference-time regeneration for only the missing probe surface.
- [Risk] Counterfactual teacher-forced results may overstate confidence relative to rollout. → Mitigation: keep teacher-forced evidence separate from rollout canaries and prefix-conditioned probes.
- [Risk] Internal probes can generate large artifacts and high GPU pressure. → Mitigation: allow an isolated research workspace, keep the authoritative decode settings fixed, and parallelize only across available GPUs.
- [Risk] High-recall duplicate mining can include obvious false positives. → Mitigation: preserve candidate-mining outputs separately from confirmed deep-probe cases, and keep the onset rule deterministic.
- [Risk] HF instrumentation can perturb the exact rollout being studied. → Mitigation: make rollout reproduction and deterministic re-forward separate layers and require the case bundle to record whether the deep probe was collected from the original rollout or a matched re-forward pass.
- [Risk] Upstream-local hook points can drift across `transformers` or `ms-swift` revisions. → Mitigation: record the upstream dependency roots and any hook-placement assumptions in manifests or case bundles.
- [Risk] A same-desc coordinate probe may still miss failures driven by object birth or ordering on mixed-desc scenes. → Mitigation: keep the study focused on same-desc duplication collapse and allow “out of scope for same-desc probe” labeling for other failure modes.
- [Risk] The report could drift into ad hoc researcher notes instead of a reproducible artifact contract. → Mitigation: require manifest, per-case outputs, and summary tables as first-class study artifacts.
- [Risk] Current prior belief about FN injection or soft supervision could bias interpretation. → Mitigation: treat FN injection as out-of-scope for the primary study axis and require evidence-backed conclusions tied to recorded probes.
- [Risk] A visually plausible hero case could dominate the narrative and overfit the mechanism story. → Mitigation: require checkpoint-family and cohort-level support before promoting a hero-case pattern to a study conclusion.
- [Risk] Attention interventions may appear to help only because they suppress generation or force undercounting. → Mitigation: evaluate interventions against candidate ranking and rollout escape quality, not just reduced duplication.

## Migration Plan

1. Add the new study capability and its requirements.
2. Implement a manifest-driven analysis entrypoint under existing analysis/config conventions plus an isolated research workspace for deep probes if needed.
3. Reuse existing artifacts where available and materialize missing inference-only evidence layers only as needed under the fixed HF decode contract.
4. Generate one canonical report over the current checkpoint set.
5. If the workflow proves useful, future changes may refine the checkpoint matrix or add richer case classification, but no artifact migration is required because this is an additive analysis capability.

## Open Questions

- Which exact `top_k` secondary diagnostics, if any, are worth preserving once the baseline reproduction is stable?
- How should the first version group visual tokens for LLM-to-vision attention summaries: native image-token ranges only, or region-aware buckets derived from prompt or parse state?
- Should the first canonical run define a fixed hero-case cohort in-spec, or only require deterministic subset provenance and allow the concrete cohort to remain config-driven?
- Which upstream-local hook sites in `transformers` or `ms-swift` give the cleanest native vision-tower summaries without destabilizing rollout reproduction?
