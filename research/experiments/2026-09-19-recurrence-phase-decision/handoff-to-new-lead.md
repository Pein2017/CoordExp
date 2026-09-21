---
title: Duplication mechanism — handoff to a new research lead
role: research-handoff
authority: transport_only
snapshot_date: 2026-09-21
source_lead_task: 01a0a3d5-dc45-7693-8467-4801aa7190df
---

# Duplication mechanism: research-lead handoff

## Read this first

**Objective:** explain why natural autoregressive detection sometimes repeatedly emits the same or nearby boxes, and distinguish the cause of first recurrence from its persistence. The eventual product objective is more distinct real objects in original-image, empty-prefix natural greedy output, while preserving credible old objects and avoiding invalidity, hallucination and abnormal stopping. The immediate research objective is causal understanding of duplication, **not exact matching or zero FN**.

**Current decision:** the last experiment is lead-accepted and closed. Small coordinate edits often change output trajectories, but do not consistently remove numerical recurrence. Stop extending that particular small-perturbation contrast merely to find another escape. **The broader mechanism investigation remains open.** No new experiment, training run, embedding rewrite or checkpoint-stage sweep is launched by this handoff.

**Working explanation, not a discovered circuit:** history-conditioned spatial progression does not reliably enforce exclusion of already emitted owners; coordinate readout scale amplifies some unfavorable choices; subsequent autoregressive feedback sustains some low-new-coverage routes. Existing evidence supports pieces of this account, not its complete causal identification. Several failure mechanisms may coexist.

**Most valuable unresolved distinction:** does the model lack a usable covered-owner representation, or does usable coverage information fail to control the serialized next-row decision? Also separate initiation, maintenance and relapse. A treatment that improves output need not identify the cause.

**Next action:** reconcile this snapshot with the current owners below, then discuss/design a materially distinguishing causal contrast. The user explicitly wants fresh thought, including methodology analogous to separating etiology, maintenance and symptomatic treatment in medicine. Do not turn that analogy into a presumed diagnosis or a new orchestration framework. Register one decision-bearing question and stop rule before new costly work.

**This document is transport, not a second research state or a launch grant.** Exact numbers, annotation versions and lifecycle remain owned by the cited results/receipts. The user is creating the new lead task; this handoff creates no task.

## 1. Minimum reading path, in authority order

1. Current user instructions and local agent contract; [research placement and authority convention](/data/CoordExp/.worktrees/research-probes/research/CONVENTIONS.md).
2. [Current frontier](/data/CoordExp/.worktrees/research-probes/research/index.md).
3. Latest [state](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-19-recurrence-phase-decision/state.json), [lead result](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-19-recurrence-phase-decision/lead-results.md), and [acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/lead-acceptance.json). Use the **lead** result, not the superseded derived count in the immutable worker result.
4. [History, repetition and stopping](/data/CoordExp/.worktrees/research-probes/research/questions/history-repetition-stopping.md) and [capacity and readout](/data/CoordExp/.worktrees/research-probes/research/questions/capacity-and-readout.md): combined evidence, counterexamples and limits.
5. [Research story](/data/CoordExp/.worktrees/research-probes/research/story.md) for why we moved away from finite fitting; [catalog](/data/CoordExp/.worktrees/research-probes/research/experiments/catalog.jsonl) for exact predecessors. The story contains historical proposed next steps; it does not override the current frontier.
6. [Physical evaluation](/data/CoordExp/.worktrees/research-probes/research/questions/physical-evaluation.md), [metric interpretation](/data/CoordExp/.worktrees/research-probes/docs/eval/INTERPRETATION.md), and [TIDE-aligned unmatched vocabulary](/data/CoordExp/.worktrees/research-probes/docs/eval/UNMATCHED_REVIEW.md) before translating a numerical failure into an owner claim.
7. [Literature map](/data/CoordExp/.worktrees/research-probes/research/literature/index.md) only for the selected question. External papers and Pro replies are hypotheses/methods, not local experimental evidence. Reopen their primary sources before relying on them.

Do not read every old transcript before acting. Retrieve the nearest accepted predecessor and its strongest counterexample. The rest of this document preserves the context needed to choose those sources.

## 2. User intent and current collaboration boundary

- Priority: understand duplication and eventually reduce physical misses on incomplete annotations. Benchmark mAP, exact coordinate agreement and FN=0 are secondary to this mechanism question.
- Strong preference: independent research judgment, curiosity, falsifiable alternatives, direct lead decisions and artifact-backed acceptance. Do not repeatedly ask permission for already authorized routine work.
- The user corrected routing explicitly on 2026-09-21: **decision-bearing analysis belongs to the lead and Astra advisers; Luna is for bounded extraction, statistics, implementation or replay chores.** This supersedes any old blanket Luna-only restriction for scientific advisers. Root retains conclusions and acceptance. The preceding recap used one Astra/xhigh challenger; its assessment supported branch closure but not a solved root cause.
- Chinese for direct user exchanges; concise English for technical records and worker messages.
- Eight GPUs are available for an authorized unit. Intermittent stress workloads are expected and yield to requested work; do not wait for an idle `nvidia-smi`. Still freeze sensible bounds and avoid duplicate work just to light up devices.
- No Notion: the user explicitly said it is unavailable. Optional repository Project Memory was not demonstrably maintained and remains unused. Do not activate it or write Codex-managed memory as part of this handoff.
- No wake-me-up plugin: use native completion/direct messages. Root reviews worker candidates; an idle task or delivered message is not scientific acceptance.
- Preserve unrelated dirty changes, accepted data and original failure receipts. No cleanup, commits, publication, source-label changes or architecture changes are requested here.
- Checkpoint-stage probes were explicitly deferred because undertrained checkpoints confound unstable decoding. Reopening needs a new reason and user-aligned contract, not an old compute grant.
- Exhaustive raw-proposal review and full annotation completion must not become a gate for every mechanism experiment. Use local HOLDs and preserve denominators.

## 3. What has already been accomplished

| Research stage | Accepted knowledge | What remains unproved |
|---|---|---|
| Shared geometry and finite fitting | Initial dual-start tests, sample-vs-token CE and the 22-image expansion reached clean 227/227 and then 376/376 known-owner fits. The 22-image main arm completed at saved update128/256 with old227 retained; conditional Source was not triggered. Shared geometry integration is accepted. Earlier Human13 pure CE also fit the fixed panel. | General enumeration, online refresh benefit, or completeness of arbitrary scenes. Another tiny fit is not the current question. |
| Source256 learning | Canonical learning versus fixed-prefix completion and its CE-normalized follow-up exposed preservation costs. Normalization reduced forgetting/errors but also gains; promotion gates were not met. | Whether online refresh or added missing labels would independently help. |
| Output ranking repair | R16 improved local train/dev FN versus positive CE but failed joint gates: reference-external coverage worsened; repeat proxy263→659; only59/91 starting gained identities retained. | Safe owner credit or a stable transferred repair. This is a different model lineage from mature2444. |
| Human review | Many nonduplicate GT-unmatched predictions were real objects; extent, visibility and part/group granularity also explain mismatches. | A population unlabeled rate; blanket positive admission for UNKNOWN; that GT noise caused repetition. |
| Norm intervention | On R16 fresh128, known matches578→592 (G30/L16), invalid rows476→5, strict near-repeat rows604→40, caps2→0. Net+13 came from9 unhealthy images, +1 from119 healthier images; paired bootstrap interval includes zero. | Population physical recall improvement, global policy promotion, unique mechanism or training-origin defect. |
| Mechanism interventions | Particular output decisions causally redirect complete continuations; history and retained state can matter; some routes relapse after apparent rescue. | A universal repetition circuit, a reliable covered-owner ledger or a general intervention that restores distinct-owner enumeration. |

The oldest A/B denotes initialization (full-teacher step256 versus original geo-sorted-xy step2444 with its embeddings); later Source256 A/B denotes learning recipes. Do not merge their meanings. Finite fitting used fixed teachers; it is not evidence of online rollout refresh. The [frontier's completed-evidence table](/data/CoordExp/.worktrees/research-probes/research/index.md) links the exact owning result for each row above.

## 4. Mechanistic evidence that changes the explanation

### 4.1 Readout scale is a causal contributor, not an established etiology

Changing only the effective coordinate output-row norms can change actual winners and empty-prefix generation. Many endpoint winners survive equalization. Mature untied+axis also has extreme output endpoint norms despite separated input/output deltas. Tying is therefore unnecessary for these observed peaks and for recurrence.

Magnitude-matched reflected/sign controls also reduce numerical recurrence in some selected cases; normalization is not uniquely privileged by that evidence. Sparse0/1 readout support explains selected donut routes through only one or two actual argmax changes, but other scenes retain null/adverse effects. Forcing the initial0→1 alone helps one untied donut route and fails on tied; continuing operators can work while restoring the original0 or the entire first row. Do not call coord1 an escape identity.

The [forward provenance result](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-19-coordinate-margin-provenance/results.md) reconstructed eight states and16 paired prefix replays. At one tied slot, raw0−30=+0.332750 decomposed into directional−0.589133 plus length+0.921884. Equalization reverses that pair, but the global winner is23. Positive0−1 can coexist with global winners52/46/23. FinalMLP27 is a large positive0−1 contributor across all eight states, including nonzero-winner states; contribution rank is not recurrence specificity or causal blame. History differences are distributed and opposed across residual updates.

### 4.2 History is used; abstract coveredness is not isolated

Earlier rows affect candidate scores even with the same literal final row. On a bounded donut common-tail contrast, older history moves complete-row relative preference away from the earlier owner but moves the first-x1 preference toward it. Both absolute candidate probabilities can increase. This is relative preference, not proven absolute inhibition.

Coveredness remains entangled with geometry, displacement, order, contextual states and synthetic-history compatibility. A model can react to history without reliably excluding already completed objects. A scan-progress account and a covered-owner account have not been cleanly separated.

### 4.3 Greedy access and full-row preference are different failures

At the genuine R16 donut first-confirmed physical recurrence, the repeated owner's retained complete rows beat retained supported unvisited-owner alternatives. Width4 search improves the repeated owner's row likelihood; it does not recover a higher-likelihood new-owner candidate. Candidate mass is tiny, so this is not an exhaustive owner-probability result.

Thus two explanations must remain: useful routes can be inaccessible to local greedy, and the model can actually prefer repeating even under full-row scoring. Top-k availability alone does not force greedy realization. Legal alternatives must be coherent boxes of the same owner, not independently combined coordinate tolerances.

### 4.4 Tokens and cache both matter, in different interventions

Initial corner checks found no cache/full argmax mismatch at363 sampled slots. This weakens a simple implementation mismatch, not all cache mechanisms.

On selected bird routes, native cache rebuilt from the changed emitted token reproduced the rescued continuation; special donated KV was unnecessary for that route. In another intervention, identical initial emitted tokens coexisted with altered retained KV and a later divergence; rebuilding removed that drift. Neither universal token-only nor universal cache-irrelevance is supported.

Full donor-state restoration reconstructs a successful computation, but partial swaps did not recover owners. Summed margin increments can cross an argmax threshold without a special joint module. A successful restoration is not a selective disease mechanism.

### 4.5 The phenomenon is broader than donuts, and not purely fixed to0/999

The mature145 panel contains deliberately enriched difficult scenes. Long runs dominate its row/pair totals. Prospective seed19-128 original-policy outputs show near-repeat exposure7/128 tied and10/128 untied+axis, exact exposure3/128 and2/128, no caps. Do not pool these strata or read pair counts as owner counts.

Categories extend to person, book, bottle, bowl, knife, cow, bicycle, car and others; rare-class denominators are small. High input-side density and repeated descriptions are associated with recurrence, but longer output opportunities and annotation context are confounded. Output-derived density would itself be contaminated by repetition.

In the corrected spatial factorial, image-only and history-only horizontal movement both shift supplied-description x1 window odds. Among admitted failures, median visual/history/joint log-ratio changes are3.722/13.100/16.916. Joint coherent movement still recurs9/12 and11/12 by sign, often around the inverse-mapped numerical anchor. This weakens a solely fixed-canvas-bin account. It neither identifies a physical owner nor proves history is stronger than vision: intervention strengths and conditioning differ.

### 4.6 Low conditional event mass can coexist with repetitive greedy

The conditional-mass probe produced441 legal numerical-repeat events from11,520 full-softmax short draws over45 states. Failure-state median event probabilities were0.0273 tied and0.0625 untied. The event is a supplied-description same-description <=8-bin union, not every physical duplicate. Seven proxy unions are empty, making their zeros structural.

A greedy row can have small joint probability while winning token by token. This does not show that the remaining probability mass contains useful new owners, or that sampling would solve enumeration.

### 4.7 Actual coordinate initialization is structured

The initializer is `natural_adjacent`: shared projected linear/Fourier features over1,000 bins,8 frequencies, scale0.02, seed0; exact CPU reconstruction passed. It is **not independent random initialization per bin**. Highest-band adjacent phase is about0.805 radians; a continuous formula need not vary slowly. Trained input/readout geometry must be measured separately, especially for untied.

Same-anchor distances are non-monotone: at anchor0, distance to bin4 can exceed distance to499 or999. This is a geometry observation, not proof of a recurrence cause. The first continuity study edited only y2:1/32 immediate and0 later fixed-suffix winners changed. That did not test general x1 or free-trajectory stability; the latest unit did extend those surfaces.

## 5. Latest accepted experiment — exact continuation state

Owner: [phase-decision lead result](/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-19-recurrence-phase-decision/lead-results.md). Output root: [recurrence-phase-decision](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision).

**Question:** across seed/first-repeat/third-row phases, do x1/y2±1 history edits alter decisions and free recurrence, and do effects survive two supplied native rows?

- Frozen45-state source panel:21 failure/24 proxy;45 native profiles/150 rows. Counterfactual subset16 failure/4 proxy sources.
-451 ready saved free cells:352 edits and99 same-boundary native controls.25 local bridge HOLDs lack two complete native rows;22 OOV edit directions were not replaced.
- The first qualifying triple is chronological and fully pairwise within8 bins for the same description. The prior exact-first selector sometimes chose a later seed. Phases are observational histories, not a pure repeat-count manipulation.
- **Corrected first-ever numerical-repeat subgroup:6/16, not4/16.** Both row0 val885 seeds were omitted in the worker's derived summary. Raw tokens, plan and crosswalk were already correct. Frozen worker bytes remain unchanged; the root-corrected summary owns the correction.
-4/352 first-free switches are opener/EOS changes in one untied train196924 source, outside that subgroup. They are not coordinate-sensitivity counts.
-120/352 free trajectories diverge across15 source/model identities. Root posthoc first-fork roles: x1=25,y1=10,x2=47,y2=32,row-boundary=6.78/114 complete coordinate-fork row pairs remain same-description and <=8 bins apart; no physical-identity conclusion follows.
- Among166 edits with both release modes: neither diverges94, immediate-only24, bridge-only3, both45. Shared intervening tokens do not reset hidden state or remove access to the earlier edited row. Direct rereading, relay, different release phase and decay remain alternatives.
- At jointly observed16-row failure horizons, longest near-run lower/same/higher: immediate11/102/6 of119; bridge8/98/5 of111. Missing short-EOS horizons are not zero repetition.85 invalid complete boxes in78 cells are retained.
- A posthoc8-bin first-fork peak is dominated by two sources (27/29 cases from untied7511/14038); do not promote it to a Fourier-frequency mechanism.

**Disposition:** local decision/trajectory sensitivity without consistent durable numerical escape or a general onset mechanism. The small-perturbation branch is closed. No physical recovery claim, successor, training or architecture change was accepted.

**Verification:**21 candidate bindings; full CPU reducer replay byte-identical; all4 first-free switches checked from full FP32 vectors and actual emitted tokens. Capture/free full-vocabulary parity over451 cells: maximum4.84e-5, zero winner mismatch. Historical source qualification is top2 except explicitly tested full-vector families. One bounded semantic review found no bridge/control blocker.

**Cost and closure:**54,206 model-forward calls;20 GPU jobs exit0;2.476 receipt-summed GPUh;2.96GiB payload. These are not kernel-utilization measurements. No owned model job remained at acceptance. Reverify present task/process state before any future dispatch; old closure is not a live process monitor.

**Important artifact limit:** full logits are saved at registered native/fixed positions and initial release; later free steps have selected/top-competitor evidence, not full vocabulary at every divergence. Complete per-cell MRoPE arrays were not saved. No rerun was ordered just to fill those omissions.

## 6. Lead's hypothesis and uncertainty ledger

These are interpretation/proposal notes preserved for the new lead, not new accepted findings. Ranking reflects current explanatory reach, not mutually exclusive diagnoses.

| Candidate explanation | Why it remains plausible | Strongest restraint | What would change our belief |
|---|---|---|---|
| **Unreliable covered-owner exclusion within spatial progression** | Older history, coordinate/order changes, translation-following recurrence and relapse all matter. | Earlier-history suppression/relative preference exists; “no memory” is too strong. | Separate coveredness from location, displacement and common-tail state on credible owners; test whether owner-selective exclusion survives those controls. |
| **Coverage information is available but loses in serialized decisions** | Full-row and first-coordinate preferences can move oppositely; sparse choices sometimes redirect useful continuations. | Repeated rows can also win full-row likelihood; bounded candidate support is incomplete. | At a native first recurrence, distinguish a preference already favoring repetition from one lost at token-level realization, with coherent owner alternatives and their actual fork costs. |
| **Context-dependent readout scale/direction amplifies bad choices** | Output-only norm interventions change decisions and structural failures; endpoint lengths are prominent. | Many winners survive equal norms; alternative directions help; benefits depend on timing/history; tying is unnecessary. | Predict sign and specificity of native competition changes beyond equal-magnitude controls, rather than merely demonstrate another rescue. |
| **Feedback maintains a low-novelty region through multiple nearby trajectories** | Many coordinate forks preserve numerical recurrence; some changed histories continue low recurrence after withdrawal, others relapse. | No strict periodic orbit, universal self-reinforcement or isolated feedback variable is identified. | A prospectively defined state/decision marker predicts onset or maintenance across held-out scenes, and reciprocal intervention changes the predicted outcome. |
| **Density, visibility and supervision granularity create weak instance separation** | Human review and exposure tables associate difficult repeated instances with failures. | Density, output length and category opportunities are confounded; many unmatched are valid; annotation inconsistency is not causal evidence. | A matched contrast separates distinguishability/granularity from mere longer generation or more same-class opportunities. |
| **Initialization or SFT optimization created the bias** | Structured input geometry is non-monotone; effective output norms diverge despite untie. | Inference intervention does not identify training origin; gradient accounting remains technical-invalid; no isolated initializer/untie control. | A qualified training-origin measurement or matched authorized training contrast. Do not revive stage sweeps by default. |

Questions I would keep visible:

1. Does the model estimate “already emitted” incorrectly, or estimate it but give it too little decision weight? A readable probe alone cannot establish use.
2. Is the first repeated row a good box of a real already-seen owner, an ambiguous extent/group, or already a degenerate numeric output? These may require different mechanisms.
3. Are onset, persistence and relapse distinct? A rescue inside a mature loop need not diagnose its entry.
4. Why does normalization have strong structural benefits in some packages but weak/adverse effects elsewhere? Cause, compensation and downstream treatment remain alternatives.
5. How does image evidence compete with historical location and serialization order? A score shift under supplied description is narrower than a free owner-selection decision.
6. Does probability fragment across many legitimate owners/extents while greedy follows a repeat route? This is a candidate explanation, not established by the small conditional-repeat mass.
7. Where is continuity needed: coordinate value→embedding, embedding→scores, or scores→full free trajectory? Argmax discontinuity makes these different. Geometrically adjacent outputs need not have identical task identity in dense scenes.
8. Can we obtain a marker that predicts future repetition before it occurs on scenes not used to choose the marker? Another large late-layer difference measured after failure is not enough.

## 7. Next-stage methodology: possible discriminators, not a launch packet

The user specifically suggested thinking like a medical investigation. The useful analogy is **etiology versus maintenance versus treatment response**, with heterogeneous phenotypes and matched controls. No biological claim or literature-backed causal guarantee is implied.

### Candidate A — first-onset case/control comparison

Identify native first numerical recurrence broadly, with a smaller physically admitted subset. Match nonrecurrent decision opportunities by checkpoint, description, prior same-description exposure, row position/length and available scene-density proxies. Keep severe enriched cases separate from prospective cases. Measure the same predeclared quantities at opener, description and each coordinate role *before* onset; avoid selecting features after seeing the outcome.

Decision: is there a selective precursor beyond ordinary late-sequence/dense-scene behavior? Strong alternative: the “marker” merely measures difficult scenes or output length. Stop if no distinguishing signal appears under the declared comparison; do not continue sweeping every layer/head until something looks different. This is still observational until an intervention tests the proposed mediator.

### Candidate B — coveredness versus spatial progress

Use credible owners and a narrow set of coherent candidate rows. Construct histories that differ in whether a target was covered while matching recent position, direction, scale, token length and tail as far as the scene permits; use reciprocal owner roles and extent controls. The earlier common-tail study is a predecessor, not an untested idea. Displacement-control admission previously failed to isolate the contrast.

Decision: does prior coverage independently reduce the target's preference, and where does that response survive or disappear between full-row scoring and actual greedy decisions? Stop with a local identification HOLD if the scene cannot separate identity from geometry; do not manufacture a causal result by accepting a confounded replacement.

### Candidate C — maintenance, withdrawal and mediation

If A/B nominate a concrete information path, distinguish access to old-row entries from propagation through later contextual states. Common-token bridges alone do not separate these. A targeted reversible intervention would need matched collateral-effect controls, real runtime qualification, immediate scores and free outcomes. Reciprocal induction in a matched nonrecurrent state is stronger than rescue alone, but out-of-distribution damage can also induce repetition.

Decision: does the proposed mechanism selectively explain persistence rather than generic computational disruption? Stop if only full successful-state restoration works or matched controls reproduce the effect. Do not reopen a full attention/KV/layer sweep without a prediction.

### Candidate D — coordinate representation / training etiology

Keep this as a separate axis if evidence points there. First distinguish input geometry, output lengths/directions, valid clipping, token frequency and actual decision margins. A norm treatment is not a matched training intervention. Any initializer replacement, objective change, head redesign or training-stage study requires its own authorization/contrast. The current user request is a handoff, not authorization to run those studies.

Across candidates: define the phenotype, competing accounts, precise intervention, native/negative controls, outcome, artifact scope, bound and stop rule **before** launch. A mechanism study can be useful without higher mAP. Conversely, reduced numeric repetition alone cannot certify useful owner recovery. Do not require all image annotations to be complete just to measure a well-defined numerical mechanism.

## 8. Dead ends, narrowed explanations and reopening conditions

- **More tiny-set SFT/QP to prove fit:** already answered. Reopen for a different estimand such as preservation, transfer or matched-success cost.
- **Uniform “bad KV cache implementation”:** sampled native/full checks and subsequent position-qualified replays weaken this. A concrete runtime counterexample can reopen it; saved-state effects are not automatically implementation errors.
- **All recurrence comes from fixed endpoints or tied embeddings:** inconsistent with untied recurrence and translated-history evidence. They may still modulate particular cases.
- **More ±1 rescues, dose/temperature/seed scans:** not the next step without a distinct prediction. Small edits already alter many routes without consistent escape.
- **Largest residual/attention/MLP contribution is the culprit:** accounting is not selective intervention. Full donor restoration is not sufficient localization.
- **Pure absent-memory or monotonic repeat-count reinforcement:** existing history effects and non-monotone fixed-tail results constrain these. Neither is a general theorem against all memory/feedback accounts.
- **NMS, repetition penalty or hard duplicate masks solve the question:** they can alter symptoms/output and may be engineering baselines later; post-hoc filtering cannot recover spent generation budget or missed owners. The user prefers understanding before changing RP.
- **GT-unmatched means hallucination; overlap means duplicate:** false. A legitimate box may contain neighboring owners. Keep identity, existence, extent, class, visibility and group/part granularity distinct.
- **A technical-invalid attempt is a scientific null:** false. Original spatialB and gradient-accounting invalid attempts remain excluded. Local HOLD is not a whole-panel null.

## 9. Checkpoints, datasets and frozen panels

The model is the local Qwen3-VL dense-enumeration pipeline with1,000 coordinate bins0–999 and wrapper tokens. Do not infer the full effective model from an adapter directory alone: paired embeddings, base model, decode/config and data identities belong to the manifest.

| Identity | Exact existing path / owner |
|---|---|
| Mature tied2444 | [checkpoint](/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444) |
| Mature untied+axis2444 | [checkpoint](/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444) |
| Older ranking R16 adapter | [adapter](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/runtime/main-v1/R/training/checkpoints/step-00016/adapter); [paired-source acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/lead/final-acceptance-v1.json) |
| Mature package configs and memberships | [panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json) |
| Frozen high-confidence18 runtime inputs | [refined18.runtime.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/refined18.runtime.jsonl), [input audit](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/input-audit.json) |
| Human13 source | [frozen13 JSONL](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl) |
| Refined5 source | [frozen snapshot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl) |
| Mutable refinement counterpart | [public working.norm.jsonl](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/working.norm.jsonl); never silently substitute it for the frozen version |
| Latest shared45-state panel | [shared-panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json), [source/model bindings](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-sources.json) |

The18scene frozen panel contains Human13's392 positives plus Refined5's178 =570 positives. Five refined train IDs:7116,309264(birds),351017,417044(donuts),477415. Images are under `/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/` with twelve-digit filenames. Useful val burst cases also include885,5586,7511,14038,632 under the corresponding `val2017` directory; they are not all refined-owner scenes. The user explicitly said not to restrict the main question to donuts or refined-only images.

Human13 SHA256: `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
Refined5 snapshot SHA256: `1d8d7c6d63e982f2d5060fa96a7cbc825c0314246276b181aaefff9ee80fed85`.
Shared45 panel SHA256: `005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`.

The user omitted visually indeterminate objects during refinement. There is no verified explicit ignore mask encoding all such omissions; absence from these positives is not a physical negative. Birds309264 changed from10 historical to14 current positives, with retained/added/removed identities; do not mix versions. The strongest physical donut onset results use **R16**, while recent broad numerical studies use **mature2444**. Do not transfer owner attribution or checkpoint conclusions silently.

When human review is needed, show full-image plotted boxes plus context crops; thick outlines can obscure tiny objects. Co-DETR crop+resize is optional corroboration, never automatic GT. The user's review found chair/dog examples initially called detector errors were valid, while the person-hand false detection was wrong. TIDE terms are useful but must allow UNKNOWN/weak visibility/extent/granularity rather than forcing every unmatched into FP. New verified training owners require explicit admission and the appropriate original JSONL `unlabeled` update; discussion alone changes no labels.

## 10. Evidence entry points and replay

Four-lane predecessor: [integrated result and artifact map](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/integration/ARTIFACTS.md), [lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/integration/lead-acceptance.json). Its conservative18.568GPUh bound includes invalid attempts, not measured utilization. A's overwritten original metadata and D's original overwritten shard receipt were not recovered. Accepted current/raw evidence remains checkable; do not claim complete archival recovery.

SpatialB had real implementation defects: horizontal affine applied to y roles, incorrect row stopping, dropped invalid triples, drift-chain near-run semantics, incorrect fixed windows, then a last-six-position logit comparison bug. Invalid attempts are preserved and excluded. Corrected position119 native gate saved both full vectors and passed maxdelta3.945827e-5; gate lineage totals10. These concrete failures justify checking **actual role/position/event semantics at a production-shaped entry**, not building a generic runtime framework. The corrected panel had12/21 failure and23/24 proxy admissions,10 local HOLDs,267 cells;48 signed cells unexecuted and12 pilot signed cells diagnostic-only.

Latest immutable [candidate manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/candidate-manifest-final.json): SHA256 `d073e8aa69118104038113489da95382d89a9492f0a1e3e87a46c85b625b1b13`.
Latest [lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/lead-acceptance.json): SHA256 at handoff `76c13f0ec208244e15136ddacfa9c5ae688468296131c1e21ef1605afc3b73e2`.
Accepted [reduction-02](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/attempts/reduction-02/reduction.json): SHA256 `a9dbb04fec27059ab5b9aa4d0e005cefc8222a7f24d766dbcae9a7f2050c6a88`.

Latest [root verification directory](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/coordination/root-acceptance) contains manifest, byte-identical replay, first-free vectors, corrected first-ever denominator, fork-role audit, bridge semantics and closure checks. The immutable worker `results.md` and manifest-bound `unit.md` were deliberately left unchanged; use `lead-results.md` and `corrected-summary.json` for the correction.

Already completed acceptance command, retained here for later CPU reproduction only; use a **fresh** output path:

```bash
# cwd: /data/CoordExp/.worktrees/research-probes
PYTHONPATH=. python probes/training_set_completion/recurrence_phase_decision/reduce.py --selfcheck

PYTHONPATH=. python probes/training_set_completion/recurrence_phase_decision/reduce.py \
  --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/selection/plan.json \
  --capture /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/reduction-inputs/capture \
  --release /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/reduction-inputs/release \
  --windows /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision/selection/windows.json \
  --output /tmp/recurrence-phase-new-lead-replay.json
```

Use bare `python`/`python3` to preserve the configured runtime, normally `ms`. The root's first CPU replay lacked `PYTHONPATH=.` and failed before any model work; the corrected command passed. Reuse accepted replay evidence unless source drift or a new claim makes rerunning useful. Hash-bound source/config snapshots in manifests outrank a subsequently edited checkout.

## 11. External consultations retained without making them authority

The original Pro replies remain local; no need to copy their long text here. These are dated proposals already partly tested, not current instructions. Opening summaries below preserve their decision impact; consult the originals for full arguments and compare with subsequent accepted results.

| Packet | Preserved impact | SHA256 |
|---|---|---|
| [Decision control versus cause](</data/CoordExp/.codex/attachments/fda17252-c1db-49a1-80d9-7fae752687dd/pasted-text.txt>) | Prioritized fresh paired norm evaluation and shadow decisions; that population experiment is now completed. | `f10a95aeafbb3c04815a54fe6ee5fe9a62edfc242c02636180856ebe869b68f5` |
| [Multiple maintenance mechanisms](</data/CoordExp/.codex/attachments/0f5135ee-a64e-42d4-a8f7-4a2fae90ad15/pasted-text.txt>) | Historical continuation versus visually driven reselection, plus readout amplifier; a hypothesis, not identified subtypes. | `37169d6f07d02e634223a133ce4a964a8e4a12ad3cf32e9af87f287ff8a31675` |
| [Physical onset advice](</data/CoordExp/.codex/attachments/0ab6fa36-f660-440e-9264-f05b2d498043/pasted-text.txt>) | Native first physical recurrence and evidence limits; later onset/branch studies supply the updated result. | `f827f59abec3a4a34fcdf3a874b38c00eaf6f82cee06926e1bb6e8db404c3a4b` |
| [Conditional impulse response](</data/CoordExp/.codex/attachments/e6b76aa4-ed03-4d0b-a602-3ad44ae75804/Pasted text.txt>) | Fixed suffix versus free response, directional history information; recent numerical feedback and phase studies now constrain it. | `65dae70e2d82a8ed79ed2cf0854e8f64e5cc0494e450274674c0c9d8189fd258` |
| [Short history-target feedback](</data/CoordExp/.codex/attachments/0a066862-8875-45f9-8e91-b56ee119cfcd/Pasted text.txt>) | Avoid making physical HOLD/gradient repair prerequisites; require selective feedback beyond generic influence. | `60ad1c03d20aa79c6c125ff2f931508b62f666154b09c0a1ee6c1b8ce1ea05df` |
| [Move beyond local0/1 attribution](</data/CoordExp/.codex/attachments/da8380e5-90d2-4cc3-bfad-021696130176/Pasted text.txt>) | Separate image position, history position and fixed numeric preference; spatial/distribution package now completed. | `6bf54045061016bdaaea38a021b9b448f28a4ed94f39124521fbcc5969542009` |
| [Equivalent histories and coveredness](</data/CoordExp/.codex/attachments/207bcaa4-30d8-42b2-a748-835a4d845719/Pasted text.txt>) | Task-equivalent history stability versus genuinely changed coverage; still relevant but do not relabel old permutation/common-tail studies as new. | `bbab90ea083700b77b7a867f3158c0d957b65e9db18310055a7f0263d61c4289` |

Earlier consultations and all user adjudications remain in the source lead task. The [literature map](/data/CoordExp/.worktrees/research-probes/research/literature/index.md) also records external session intake. In particular, temporal growth, readable features and attention mass need appropriate nulls before causal interpretation. A VIT-register analogy was raised by the user but was never established as the cause of these outputs.

## 12. Checkout, tasks and transport

- Verified cwd: `/data/CoordExp/.worktrees/research-probes`.
- HEAD at handoff preparation: `13b5e434930942a82cda7002a40b6acd06d71926`; branch `research-probes`. The checkout was already dirty:4 tracked modifications and33 untracked entries before this handoff. HEAD alone does not identify experiment code; use source snapshots and manifests. Preserve all unrelated work.
- This handoff is under the last experiment to obey the flat research layout; do not create a separate active `research/handoffs/` bucket or a competing current-state ledger.
- No new owning OpenSpec change is created. Research meaning and completion for the closed probe are owned by its existing unit/state/lead result. Consult an actual owning OpenSpec before making future shared infra changes.
- Original lead: `codex://threads/01a0a3d5-dc45-7693-8467-4801aa7190df`.
- Most recent persistent execution worker: `codex://threads/01a0ba19-8004-7400-b8af-98ce9b9df0cd` (9-19-worker). Last verified model configuration was user-selected Sol/xhigh; preserve it unless explicitly changing the arrangement. It has completed the last unit and received a closure notice. It is not authorized to invent a successor.
- Earlier execution context: `codex://threads/01a0a81a-9e32-7db1-bd07-86fa601f4276` (9-16-worker), no current ownership.
- Untied-training discussion: `codex://threads/01a0b28f-ddec-74f0-89bd-7d3f094059bd`. External-paper intake: `codex://threads/01a0afd6-87bb-7b83-a606-47da710dac2f`. These are retrieval pointers, not current action requests.

Follow the current [lead-worker skill](/data/CoordExp/.codex/skills/lead-worker/SKILL.md) and [native team guidance](/data/CoordExp/.codex/skills/native-agent-team-guide/SKILL.md) with the user's latest routing override. Root owns decisions/acceptance; advisers advise; a worker executes a bounded contract. Reconcile task/model/cwd/current turn and write ownership before dispatch. Do not launch a parallel CLI continuation of an App-owned worker.

For an explicitly requested send, discover the current native thread tools first. Prior direct delivery used [worker_turn.py](/data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py) through `/data/CoordExp/.codex/app-server-control/app-server-control.sock` because no native send wrapper was available. The helper's default Astra-low pairing did not match the user-selected Sol/xhigh task; root used a one-off exact identity/config check rather than changing helper defaults or silently changing the model. Native `turn/start` was used when idle, `turn/steer` with the current turn when running. Save a unique receipt; a timeout can mean delivery-unknown, so never blindly resend. Reverify tool/runtime availability rather than assuming this fallback is still needed.

## 13. Persistence limits and first action for the new lead

This document preserves the decision-critical story, hypotheses, counterexamples, exact artifact routes and recovery details. It is not a duplicate of every transcript, tensor or figure. Full evidence stays in the cited immutable artifacts; some explicitly disclosed data were never retained or were overwritten and cannot be reconstructed from prose. Local links depend on this host/filesystem; they are not an off-host backup.

The handoff does not refresh every checkpoint tensor hash, worker runtime, current public-data file or tool registration. Reverify those volatile facts if they become inputs to a new run. No new scientific conclusion or GPU execution is pending acceptance for the closed unit.

**Suggested first new-lead response:** state what is known about control versus cause; name the strongest two competing explanations; propose the cheapest contrast that makes them predict different outcomes; identify which part is onset, maintenance or repair. Ask the user only for a material semantic/cost/architecture decision that the current conversation has not settled. Then bind the agreed unit and let a worker execute it. Do not restart with another generic sensitivity scan or another proof that the tiny training set can be fit.
