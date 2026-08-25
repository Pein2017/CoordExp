# Research-Influential Session Synthesis

This is a bounded intake from May and June 2026 session JSONL. The session archive is provenance, not current scientific authority. No entry below promotes a mechanism, architecture, training result, or deployment claim beyond the observed evidence surface.

### A2 is the stable reference; A3/A4 remain probes
- Sources: `/data/CoordExp/.codex/sessions/2026/05/13/rollout-2026-05-13T01-57-39-019e1f0d-d12b-7b13-9bc3-db8a2d50ef60.jsonl` (`019e1f0d-d12b-7b13-9bc3-db8a2d50ef60`); `/data/CoordExp/.codex/sessions/2026/05/13/rollout-2026-05-13T01-59-40-019e1f0f-a8e2-7bb1-bf41-a5619690cd9d.jsonl` (`019e1f0f-a8e2-7bb1-bf41-a5619690cd9d`); `/data/CoordExp/.codex/sessions/2026/05/13/rollout-2026-05-13T01-59-40-019e1f0f-a989-7682-b10a-62990eb20420.jsonl` (`019e1f0f-a989-7682-b10a-62990eb20420`).
- Current owner: unowned in this intake; the sessions route to the older A1/A2/A3/A4 research notes and current decision router, which must be checked before reuse.
- Question and contrast: whether prefix-rollin variants A3/A4 should replace the stable compact-full A2/support2 reference for duplication and continuation behavior.
- Observed evidence surface: repository archaeology and experiment/log review. A2 is treated as the stable compact-full support2 reference; A3/A4 use sampled ground-truth prefix roll-in and are not exposure-matched to A2.
- Scientific disposition: bounded historical interpretation only. A3/A4 show continuation/recall pressure and tail-risk tradeoffs, but the sessions do not establish a fair causal replacement comparison.
- Technical/infrastructure disposition: future comparisons require explicit prefix exposure, denominator, conditioning, and artifact identity; raw logs alone are insufficient.
- Decision or continuation impact: retain A2 as the reference while designing a fair controlled comparison; do not promote A3/A4 from these sessions.
- Not claimed: no claim of superior natural decoding, trainability, generalization, or mechanism truth.
- Notion coverage: `needs-summary` — Autoregressive state/visitation/termination owner `3c79d9ce-3f59-812b-a0e7-cd61a84de65a`; exact same evidence and claim boundary were not verified in the snapshot.

### Full-train anchor statistics favor 16x16 as a budget tradeoff, not a model result
- Sources: `/data/CoordExp/.codex/sessions/2026/05/29/rollout-2026-05-29T11-02-29-019e7366-5dd3-75e2-851a-da2f0a42a924.jsonl` (`019e7366-5dd3-75e2-851a-da2f0a42a924`); `/data/CoordExp/.codex/sessions/2026/05/29/rollout-2026-05-29T11-02-14-019e7366-26b3-7693-a8e1-a75c66d1d816.jsonl` (`019e7366-26b3-7693-a8e1-a75c66d1d816`).
- Current owner: unowned; candidate route is the anchor-grid design and serialization/geometry track.
- Question and contrast: which fixed grid gives a useful anchor-before-category binding tradeoff under token budget and dense/small-object pressure.
- Observed evidence surface: full-train COCO statistics and data-pipeline archaeology. The reported 16x16 point has 87.29% unique anchor-plus-category assignment, 12.71% same-category collision, 80.43% unique assignment in images with at least 40 objects, and 69.2% for small objects; 14x14 reports 85.01%, 14.99%, 76.91%, and 65.2% respectively.
- Scientific disposition: exploratory data-statistics evidence; it supports a provisional design choice, not detection quality or binding causality.
- Technical/infrastructure disposition: recommended input is the uncapped COCO train JSONL for dense-tail statistics, with image grid from `ceil(height/32)` and `ceil(width/32)` and boxes from `bbox_2d`; max-object caps can hide the densest scenes.
- Decision or continuation impact: use 16x16 as the higher-budget candidate and 14x14 as the conservative ~200-token candidate; ablate rather than declare a winner.
- Not claimed: no AP/recall/duplication reduction, no learned state, and no evidence that anchor tokens cause better generation.
- Notion coverage: `needs-summary` — Serialization, geometry, ordering owner `3c79d9ce-3f59-81c0-91a6-cf798be94c96`.

### Custom objective integration must preserve sidecar semantics and normalization ownership
- Sources: `/data/CoordExp/.codex/sessions/2026/05/19/rollout-2026-05-19T10-31-08-019e3fca-157e-7673-bee8-c5276dbc8d82.jsonl` (`019e3fca-157e-7673-bee8-c5276dbc8d82`).
- Current owner: `/data/CoordExp/src/training` and the teacher-forcing objective bridge; no scientific result owner is established by this session.
- Question and contrast: how to add full-vocabulary/sidecar objective terms without violating the upstream VLM trainer, labels, generation, or evaluation contracts.
- Observed evidence surface: local Transformers/Qwen3-VL API and trainer/collator source inspection. Differentiable objective logic belongs in `compute_loss`; callbacks are observational. A wrapped collator should preserve prepared assistant-span labels and attach sidecars; `DataCollatorForLanguageModeling` can mask or recreate labels incorrectly. `num_items_in_batch` normalization must not silently replace sidecar normalization, and Qwen3-VL-MoE router loss needs explicit handling if enabled.
- Scientific disposition: mechanics-only, not a training or model-quality observation.
- Technical/infrastructure disposition: preserve labels and sidecars, make normalization explicit, and keep router-loss treatment explicit before any objective comparison.
- Decision or continuation impact: use this as a precondition for teacher-forcing or auxiliary-loss experiments; it does not choose an objective.
- Not claimed: no loss improvement, no causal effect, and no evidence of natural-decoding transfer.
- Notion coverage: `needs-summary` — Training and research infrastructure owner `3c79d9ce-3f59-8186-838b-c775b98c787c`.

### FN-rescue artifact metadata must point to final merged files
- Sources: `/data/CoordExp/.codex/sessions/2026/06/02/rollout-2026-06-02T07-08-48-019e8729-dced-7132-a66a-2e51472513eb.jsonl` (`019e8729-dced-7132-a66a-2e51472513eb`).
- Current owner: `/data/CoordExp/src/analysis/autoreg_fn_rescue_continuation.py` and its focused test; scientific route remains unowned here.
- Question and contrast: whether the FN-rescue continuation merge produces durable, consumer-trustworthy artifacts rather than only a successful temporary merge.
- Observed evidence surface: read-only code-quality review plus a tiny temporary-root reproduction. `merge_summary.json` recorded paths under a deleted temporary merge directory while root-level merged files existed; targeted tests reported `71 passed` before the proposed regression assertion.
- Scientific disposition: affected artifact evidence is technically untrusted until metadata is rewritten against the final artifact root; no FN-rescue quality conclusion follows.
- Technical/infrastructure disposition: build output metadata after final replacement and assert every recorded path exists under the artifact root with matching row count and SHA-256.
- Decision or continuation impact: block downstream analysis that consumes stale paths; rerun or revalidate the merge after the repair.
- Not claimed: no change in rescue rate, recall, or model behavior.
- Notion coverage: `needs-summary` — Set coverage/all-HF/credit owner `3c79d9ce-3f59-814e-aeef-d9f755c10afe`.

### Row-conditioned coverage requires rebuilding each prefill from committed valid rows
- Sources: `/data/CoordExp/.codex/sessions/2026/06/06/rollout-2026-06-06T07-11-43-019e9bc5-faaf-76f1-a7d7-54d44aea124e.jsonl` (`019e9bc5-faaf-76f1-a7d7-54d44aea124e`).
- Current owner: `/data/CoordExp/.worktrees/row-conditioned-visual-coverage` rollout and artifact surface; no current research-unit owner was confirmed.
- Question and contrast: whether row-conditioned coverage forward passes actually condition on the prompt plus parse-valid committed rows, rather than stale rollout metadata or fixed input tensors.
- Observed evidence surface: mechanics review. The repaired helper calls `build_prefill_inputs` for each row, forwards HF-safe `input_ids`/`attention_mask`/`use_cache`, decodes the next row, and commits only valid parsed rows. A strict fake model test checks row-0/row-1 tensor differences and rejects stale unknown kwargs; `13 passed in 1.07s`, `py_compile` and `git diff --check` passed.
- Scientific disposition: mechanics-only; it establishes the intended conditioning path, not coverage improvement or model quality.
- Technical/infrastructure disposition: keep invalid decoded text out of the committed prefix and preserve feature isolation/finish-reason metadata.
- Decision or continuation impact: this is a prerequisite before any row-coverage scientific pilot; natural behavior still requires a separate transfer/evaluation check.
- Not claimed: no causal coverage-memory mechanism, recall gain, or trainability claim.
- Notion coverage: `needs-summary` — Representation and binding owner `3c79d9ce-3f59-81ec-95f9-db7c27fe182b`.

### Coverage-ledger design defines a precise intervention surface but not a promoted mechanism
- Sources: `/data/CoordExp/.codex/sessions/2026/06/28/rollout-2026-06-28T16-33-03-019f0f13-cbfd-71b1-af54-151be4850a4f.jsonl` (`019f0f13-cbfd-71b1-af54-151be4850a4f`); `/data/CoordExp/.codex/sessions/2026/06/29/rollout-2026-06-29T04-17-39-019f1198-e054-7282-8c5d-fd6d9d5f74d8.jsonl` (`019f1198-e054-7282-8c5d-fd6d9d5f74d8`).
- Current owner: `/data/CoordExp/.worktrees/coverage-ledger-mechanistic-probing/research/` and the named ledger auxiliary-loss worktree; current decision authority was not verified here.
- Question and contrast: whether an auxiliary coverage ledger can shape object binding/coverage state, with premerge versus postmerge intervention compared under the same template and parent baseline.
- Observed evidence surface: design/spec and checkpoint-provenance archaeology. The design names prompt-end uncovered state, row-completion state at `<box_end>`, row-object binding at `<box_start>`, post-merge projected visual-token rectangles, a sidecar payload, same-forward capture, and separate diagnostic metric/loss namespaces. For the two named checkpoints, `resolved_config.json` and trainer metrics were stronger run-level evidence than stale or unavailable source-config paths.
- Scientific disposition: planned mechanism probe plus provenance caveat; no mechanism promotion or causal result.
- Technical/infrastructure disposition: compare exact sorted pure-CE parent, template/token surface, intervention location, and immutable run lineage. The source config for the exact parent was not available at the current path, so semantic comparators are not exact provenance.
- Decision or continuation impact: preserve pre/post-merge as separate contrasts and repair provenance before interpreting a ledger effect.
- Not claimed: no evidence that the ledger changes natural decoding, recall, duplication, or hidden-state memory.
- Notion coverage: `needs-adjudication` — Set coverage/all-HF/credit owner `3c79d9ce-3f59-814e-aeef-d9f755c10afe`; the exact-parent provenance gap can change the comparison claim.

### CoordExp-Swift pre-kickoff still had unresolved runtime authority questions
- Sources: `/data/CoordExp/.codex/sessions/2026/06/30/rollout-2026-06-30T02-12-42-019f164c-d6e9-7c92-9c0e-712724a99162.jsonl` (`019f164c-d6e9-7c92-9c0e-712724a99162`).
- Current owner: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra`; implementation authority remains separate from this audit.
- Question and contrast: whether config/runtime/artifact contracts are sufficiently fixed for a safe pre-kickoff rebuild.
- Observed evidence surface: strict OpenSpec validation and read-only audit. The audit left tail-fill authority and whether a smoke fixture may also serve as eval unresolved; the worktree was broadly dirty with deleted legacy paths and untracked rebuild docs.
- Scientific disposition: unexecuted; no model or training evidence.
- Technical/infrastructure disposition: resolve tail policy and fixture-eval binding, then preserve Wave-0 status and narrow staging before implementation.
- Decision or continuation impact: do not treat a green spec validation as a completed vertical smoke or reproducibility proof.
- Not claimed: no claim about training quality, packing correctness, or final metrics.
- Notion coverage: `needs-summary` — Training and research infrastructure owner `3c79d9ce-3f59-8186-838b-c775b98c787c`.

### Boundary-tail review found artifact and matching contracts that can invalidate interpretation
- Sources: `/data/CoordExp/.codex/sessions/2026/05/13/rollout-2026-05-13T06-35-11-019e200b-e79a-71b2-b2f8-a3f792561c8f.jsonl` (`019e200b-e79a-71b2-b2f8-a3f792561c8f`); `/data/CoordExp/.codex/sessions/2026/05/13/rollout-2026-05-13T06-35-11-019e200b-e7f8-7da0-964d-b8a469d3923f.jsonl` (`019e200b-e7f8-7da0-964d-b8a469d3923f`).
- Current owner: `/data/CoordExp/.worktrees/boundary-tail-direction-gate/src/analysis` and its diagnostic artifact root; the scientific decision owner is not established in this package.
- Question and contrast: whether boundary/tail risk diagnostics can compare runs without silently mis-grouping images, misassigning objects, or trusting stale artifact counts.
- Observed evidence surface: read-only evidence-validity and implementation audits. Findings covered worktree/artifact-root binding, global candidate matching by tier then IoU, canonical `image_id` grouping across differing image strings, finite confidence handling, and stale report counts.
- Scientific disposition: the affected diagnostic verdict is neutral until matching, grouping, and report counts are mechanically reconciled; no boundary or tail mechanism is established here.
- Technical/infrastructure disposition: use exact artifact roots, deterministic one-to-one matching, canonical image identity, finite-value guards, and fresh recounts before manual audit or threshold transfer.
- Decision or continuation impact: preserve this as a validity gate for any boundary/tail experiment; repairs require fresh immutable artifacts.
- Not claimed: no duplication-burst estimate, valid-object gain, or threshold recommendation.
- Notion coverage: `needs-summary` — Causal evaluation/evidence owner `3c79d9ce-3f59-8173-ba20-c5cade809d97`.

## Package closeout

- Coverage: every May/June source file has one manifest row; all paths exist and are inside the frozen date roots.
- Sampling: 11 deep-read high/medium rows and 10 skip rows were inspected; the method records the required 10+10 sample.
- L2 routes: none started because no callable `gpt-5.6-luna` route was exposed in this session; this is recorded as a topology gap rather than treated as scientific evidence.
- Unknowns: hidden reasoning, exact current authority for several historical routes, exact parent-config provenance for the ledger checkpoints, and Notion page content beyond the routing map.
