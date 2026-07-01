## Context

The immediate research problem is not training a new detector, but explaining why two fixed 2B checkpoints behave so similarly on offline rollout:

- `original`: the Stage-1 2B checkpoint that already shows strong semantic objectness,
- `a_only`: the Stage-2 Channel-A-only continuation checkpoint that appears only modestly different in aggregate, while looking slightly healthier in some crowded scenes.

The core uncertainty is now at rollout time:

- Are missing GT objects truly never represented by the model?
- Or are they sometimes available but not selected under greedy or canonical rollout?
- Does the learned prefix/order prior steer continuation away from certain objects?
- Does the model stop early because the sequence already “looks complete enough” under the training distribution?

The study must therefore freeze training and compare checkpoints only through offline rollouts on the same dataset split and image set. This avoids confounding factor analysis with simultaneous checkpoint updates. The canonical study shape is a dense vertical ablation over a small fixed subset of `train` or `val` images, not a broad one-pass benchmark over the full split.

The initial canonical comparison pair is fixed:

- `original = output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
- `a_only = output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

The initial canonical dataset sources are fixed:

- `train = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `val = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

The study should also preserve the `a_only` checkpoint provenance sidecar:

- `output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml`

A quick split audit suggests this should be treated as a seen-vs-held-out comparison, not a gross density-shift comparison:

- `train`: `117247` images, mean `7.238` GT objects/image, `p95=22`, `max=60`
- `val`: `4951` images, mean `7.326` GT objects/image, `p95=22`, `max=55`

So if FN remains strong on the `train` split, the likely explanation is not that `train` is simply much denser or structurally different than `val`; it is more likely to be tied to rollout selection, ordering, stopping, or annotation-policy ambiguity.

This study is an extension of the archived unmatched-proposal verifier ablation in two ways:

- it adopts the same authority-first staged evidence philosophy,
- and it reuses the same discipline of explicit validity gates before interpreting deeper downstream signals.

Constraints and invariants:

- Use the offline-prepared `train` and `val` JSONL and preserve current geometry invariants.
- Preserve Qwen3-VL chat-template compatibility and `do_resize=false` parity.
- Prefer config-first study orchestration over new ad-hoc CLI surfaces.
- Keep checkpoint comparison reproducible with explicit manifests and seeds.
- Use the same fixed image subset and image order for every cell inside one study run.
- Gate downstream interpretation on rollout-health checks instead of assuming every collected cell is equally meaningful.
- Avoid changing the stable training or inference contract unless clearly required.

Data flow:

`fixed checkpoints + study config + split-specific offline JSONL`
→ `bootstrap selector over frozen candidate pool`
→ `curated hard-case subset / manifest resolution`
→ `baseline rollouts + health checks`
→ `factor sweeps (decode / prefix / length / K)`
→ `gt_vs_pred artifacts + summaries + matches`
→ `object-level recovery analysis + aggregate reports + qualitative panels`

Stakeholders:

- research operators comparing `original` vs `a_only`,
- future checkpoint authors who want to add stronger geometry modules or random-order SFT and evaluate them under the same study harness,
- paper/report consumers who need reproducible evidence rather than impressionistic visual comparison.

## Goals / Non-Goals

**Goals:**

- Provide a reproducible offline study workflow that compares fixed checkpoints on the same fixed small-image subset, prompts, preprocessing, and evaluation settings.
- Provide two canonical hard-case subsets, `Hard-16` and `Hard-32`, for each dataset split, intentionally concentrating the poorest `train` or `val` samples instead of mixing in easy images.
- Measure the impact of rollout factors that plausibly drive FN:
  - checkpoint identity,
  - deterministic vs sampled decoding,
  - prefix continuation regime, especially training-order vs random-order vs rollout-order prefixes,
  - sequence-length / EOS pressure,
  - and targeted switched/broken prefix stress tests.
- Preserve `dataset_split` as a first-class factor so the study can separately answer:
  - which FN persist on held-out `val`,
  - and which FN still persist on in-distribution `train`, where persistent misses are a stronger warning sign.
- Produce object-level union-of-`K` coverage outputs that distinguish:
  - deterministic hits,
  - sampling-recoverable misses,
  - prefix-sensitive misses,
  - length-sensitive misses,
  - persistent unrecovered objects under tested interventions,
  - and likely annotation-mismatch cases.
- Preserve enough artifact structure that the study can be resumed, audited, and extended to future checkpoints.
- Use four local GPUs by sharding study cells rather than requiring training-style distributed inference.

**Non-Goals:**

- Training or fine-tuning new checkpoints as part of this study workflow.
- Changing the public training contract or the general-purpose inference pipeline for everyday runs.
- Solving annotation incompleteness in COCO; the workflow may surface it, but it does not redefine the evaluator.
- Claiming causality from a single metric; conclusions must remain stratified by deterministic, sampled, and qualitative evidence.
- Replacing large-scale benchmark evaluation; this study is for intensive diagnosis on a small fixed subset.
- Producing a balanced or representative subset; these cohorts are intentionally difficulty-biased.

## Decisions

### Decision 1: Introduce a dedicated fixed-checkpoint study workflow rather than extending the general inference pipeline

The study requires behavior that is more specialized than standard infer/eval:

- paired checkpoint comparison,
- prefix-conditioned continuation,
- union-of-`K` object-level recall analysis,
- and multiple sweep axes over the same frozen image set.

Instead of widening the inference pipeline contract, the change will introduce a study-specific workflow under `configs/analysis/`, `scripts/analysis/`, and `src/analysis/` that reuses existing inference and parity helpers where possible.

Why this over modifying `scripts/run_infer.py` directly:

- it keeps the day-to-day inference interface stable,
- it avoids introducing many study-only knobs into a general entrypoint,
- and it allows us to enforce stronger provenance for checkpoint-comparison experiments.

Alternatives considered:

- Extend the unified inference pipeline with prefix-intervention and sweep controls.
  Rejected because it would overload a stable operator workflow with study-specific complexity.
- Use ad-hoc notebooks or one-off scripts per study.
  Rejected because the evidence would be harder to reproduce and compare across checkpoints.

### Decision 2: Freeze all study inputs through an explicit manifest

Each study run will resolve a manifest that records:

- checkpoint aliases and paths,
- checkpoint artifact kind and fingerprint,
- dataset split,
- split-specific offline JSONL,
- optional checkpoint provenance sidecars such as `config_source.yaml`,
- explicit frozen image ids and image order,
- subset kind and subset-selection provenance,
- image root,
- prompt variant,
- prompt hash,
- object field order,
- preprocessing invariants including `do_resize=false`,
- evaluator settings,
- backend choice,
- generation defaults,
- prefix construction defaults,
- and seed schedule.

The study will never compare checkpoints unless these inputs are explicitly resolved and written to artifacts before execution.

Why:

- the main failure mode in checkpoint-comparison studies is accidental drift in prompt, backend, or dataset subset;
- manifest-first execution makes the resulting outputs auditable.

Alternatives considered:

- infer defaults from whichever run dir or script happens to be active.
  Rejected because it is too easy to compare non-equivalent rollouts.

### Decision 2a: Canonical subsets are hard-case cohorts, not mixed controls

The canonical subsets are:

- `Hard-16`
- `Hard-32`

Both subsets must be selected from poor-performing images within one fixed dataset split rather than from a balanced or representative sample.
`Hard-16` MUST be a strict top-16 prefix subset of `Hard-32` within the same dataset split.

Recommended selection signals:

- low image-level matched recall under deterministic baseline,
- high unresolved GT count,
- high crowding or dense-object count,
- and persistence across both checkpoints rather than a one-off outlier for only one model.

Selection rule:

- a deterministic bootstrap selector pass runs before subset freeze,
- the selector operates independently per dataset split, defaulting to the corresponding offline split JSONL unless a smaller candidate pool is explicitly manifested,
- the selector ranks only rollout-health-valid images from the bootstrap pass,
- `Hard-32` is the top `32` images from that split-specific ranking,
- `Hard-16` is the top `16` images from the same split-specific ranking, and therefore a strict subset of `Hard-32`.

Default ranking tuple, in order:

1. descending mean unresolved GT count across `original` and `a_only`,
2. descending mean unmatched prediction count across `original` and `a_only`,
3. descending GT object count,
4. ascending `image_id` as the final tie-break.

Why:

- the user wants to know why FN persists on the images that actually matter,
- FN that persist on `train` are a stronger sign of rollout blocking or incapacity than equally strict FN on `val`,
- and easy images dilute the causal signal while consuming factor-budget.

### Decision 2b: Preserve dataset split as a first-class causal factor

The study will treat `dataset_split ∈ {train, val}` as part of the logical analysis condition rather than a side annotation.

Canonical cohorts are therefore:

- `train / Hard-16`
- `train / Hard-32`
- `val / Hard-16`
- `val / Hard-32`

Interpretation rule:

- conclusions about persistent FN must be reported separately for `train` and `val`,
- `Hard-16` remains the inner hard cohort and `Hard-32` remains the extension cohort within one split,
- and a GT object that remains unrecovered on `train` should be treated as stronger evidence of incapacity or rollout blocking than an otherwise similar `val` miss.

Why:

- the user explicitly wants to understand whether the model is missing objects it has already seen during SFT,
- and the split comparison is only meaningful if it is preserved all the way through manifests, aggregation, and final reporting.

### Decision 3: Use HF only for the first version of the study

The study backend will be `hf` only.

Rationale:

- HF is simpler for exact prefix control, continuation from manipulated prefixes, and consistent per-step intervention.
- HF deterministic behavior is easier to reason about for “same checkpoint / same prefix / same seed” comparisons.
- The study is checkpoint-analysis oriented, not throughput-bound training.

How four GPUs are used:

- one process per GPU,
- each process executes independent study cells,
- no tensor/model parallelism is required for the default workflow.

Alternatives considered:

- Use `vllm` everywhere for speed.
  Rejected because prefix intervention and deterministic continuation control are more awkward there.
- Use a mixed HF/`vllm` study.
  Rejected for the first version because backend variance would complicate the FN diagnosis more than it helps.

### Decision 4: Organize the study into four evidence layers

The study will explicitly separate:

1. **Deterministic baseline layer**
   - greedy or temperature-0 rollout,
   - no prefix intervention,
   - checkpoint-vs-checkpoint comparison.

2. **Sampled coverage layer**
   - `K` rollouts per image at controlled temperature cells,
   - union-of-`K` recall and per-GT hit frequency.

3. **Prefix sensitivity layer**
   - continuation under controlled prefix modes and lengths,
   - comparison of downstream object recovery and continuation behavior.

4. **Sequence-length layer**
   - matched default-length vs extended-length cells,
   - comparison of downstream object recovery and stop behavior.

Why:

- each layer answers a different question,
- and mixing them would make interpretation much weaker.

Alternatives considered:

- one giant factorial grid without interpretation layers.
  Rejected because the resulting evidence would be broad but hard to reason about.

### Decision 4a: Add a rollout-health gate before deeper causal attribution

Before a logical cell participates in the main sampled, prefix, or length comparison tables, the study will record rollout-health signals and decide whether the cell is healthy enough for interpretation.

Minimum health signals:

- non-empty prediction rate,
- parse-valid rate,
- invalid-rollout count,
- duplicate-like rate,
- prediction count,
- and truncation or stop anomalies when available.

The rollout-health gate will be manifest-resolved rather than hardcoded, but its formulas, thresholds, and invalid-reason precedence must be explicit and frozen per run.

Default invalid-reason precedence:

1. `parse_invalid`
2. `invalid_rollout`
3. `truncation_anomaly`
4. `too_few_nonempty`
5. `too_few_predictions`
6. `too_duplicate_like`

Interpretation rule:

- unhealthy cells remain visible in artifacts and appendices,
- but they do not dominate the main causal FN attribution tables.

Why:

- this is the main extension learned from the archived unmatched-verifier study,
- and it prevents parser collapse or degenerate rollouts from being mistaken for meaningful prefix or length effects.

### Decision 5a: Define a normative continuation partition for prefix-conditioned cells

Prefix-conditioned cells must preserve a deterministic boundary between prefix-injected predictions and continuation-emitted predictions.

Normative contract:

- the final prediction array for a prefix-conditioned cell must serialize all prefix-injected predictions first and all continuation-emitted predictions after them,
- each prefix-conditioned artifact must record:
  - `prefix_pred_count`
  - `continuation_pred_start_index`
- `continuation_pred_start_index` must equal `prefix_pred_count`,
- continuation-only recovery is computed by applying the normal matching rule to the continuation tail only, excluding prefix-injected predictions from the candidate set.

Interpretation rule:

- prefix-injected objects may appear in whole-scene review artifacts,
- but they do not count toward continuation-only recovery,
- and a continuation duplicate of a prefix-injected object does not create new GT recovery beyond what the prefix already satisfied.

Why:

- `prefix_sensitive_miss` is only meaningful if injected prefix objects cannot satisfy recovery by construction,
- and the existing shared whole-scene visualization contracts do not provide this split automatically.

### Decision 6: Support explicit prefix-intervention modes instead of treating prefix as an implicit side effect

The study will model prefix intervention as a first-class factor.

Initial modes:

- `image_only`: standard rollout from image plus prompt only.
- `oracle_gt_prefix_train_order`: continue from the first `N` GT objects serialized in the same order used by training targets.
- `oracle_gt_prefix_random_order`: continue from the same first `N` GT objects, but serialized in a fixed seeded random order for that image.
- `self_prefix`: continue from the checkpoint’s own first `N` generated objects after truncating the continuation boundary.
- `switched_prefix`: continue one checkpoint from another checkpoint’s prefix for the same image.
- `broken_prefix`: continue from a perturbed version of a plausible prefix, such as an adjacent swap, deletion, or small wrong object insertion.

Prefix lengths:

- start with `N ∈ {1, 2, 4, 8}`.

Why:

- this directly operationalizes the user hypothesis that rollout can flip when the early prefix changes,
- it makes training-order vs random-order vs rollout-order a first-class causal comparison,
- and it allows us to distinguish order sensitivity from simple capacity limits.

Alternatives considered:

- only compare final outputs and infer prefix effects indirectly.
  Rejected because it leaves the main hypothesis untested.

### Decision 7: Treat union-of-K coverage plus recovery precedence as the main test for why FN persists

For sampled runs, the study will compute object-level coverage over GT objects:

- a GT object is “covered” if it is matched by at least one rollout in the `K` sample set,
- and per-GT hit frequency is the fraction of rollouts in which it is matched.

The study will not stop at raw coverage. Each GT object will receive a mutually exclusive recovery status with minimal-intervention precedence:

- `deterministic_hit`
- `decode_selection_miss`
- `prefix_sensitive_miss`
- `length_bias_miss`
- `persistent_unrecovered`

The study will also preserve supporting frequency buckets:

- `never_hit`
- `rare_hit`
- `often_hit`
- `always_hit`

Why:

- this is the cleanest way to distinguish latent knowledge from deterministic decode selection,
- and the precedence rule prevents the same missed GT object from being “explained” by several interventions at once.

Alternatives considered:

- compare only aggregate recall/F1 across sampled runs.
  Rejected because aggregate metrics cannot tell whether the same GT objects are always missing.

### Decision 7a: Use evaluator-domain GT identity as the normative per-object key

The normative GT-object key for analysis artifacts is:

- `(record_idx, gt_idx)`

Required supporting provenance:

- `image_id`
- `file_name` when available

When review or overlay artifacts are materialized, they must also preserve a derived visualization bridge:

- `canonical_gt_index`

Interpretation rule:

- aggregation, recovery tables, and shard merges operate on `(record_idx, gt_idx)`,
- `canonical_gt_index` is derived only for visualization/review joins and does not replace the evaluator-domain key.

Why:

- the repo already uses `(record_idx, gt_idx)` as the normative GT key in Oracle-K style analyses,
- and canonical visualization uses a separate GT identity domain that should be treated as a bridge, not the primary analysis key.

### Decision 8: Make sequence-length bias explicit through controlled length cells

The study will test whether apparent FN is coupled to implicit object-count or token-count priors by comparing:

- default `max_new_tokens` cells,
- extended `max_new_tokens` cells,
- emitted object count,
- emitted token count,
- stop reason,
- and EOS position when available.

The analysis will be tied to:

- per-image GT object count,
- per-rollout predicted object count,
- and training-style ordering assumptions.

Why:

- “stops too early because the list already looks complete” is a plausible failure mode that cannot be isolated from standard recall metrics alone.

Alternatives considered:

- rely on temperature sweeps alone.
  Rejected because temperature variation does not cleanly isolate stop-length bias.

### Decision 9: Use dense hard-case subsets instead of a sparse full-split sweep

The canonical study run will target fixed hard-case subsets of `train` and `val` images and will spend compute budget on vertical combinations rather than breadth.

Why:

- the immediate research question is about per-image, per-object rollout behavior,
- the user wants to compare many interventions on the same image,
- and a dense matrix over poor images is more informative here than a shallow pass over the entire dataset split.

Study consequences:

- the manifest must record explicit image ids and image order,
- every factor cell must reuse the same subset,
- the study should treat `dataset_split × {Hard-16, Hard-32}` as named cohorts rather than generic subset sizes,
- and qualitative review is a first-class output, not a sidecar afterthought.

### Decision 10: Freeze the initial canonical checkpoint pair and keep the alias system extensible

The study must support arbitrary fixed checkpoint aliases, but the initial canonical pair is:

- `original = output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
- `a_only = output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

The initial manifests must also freeze:

- `train = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `val = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- `a_only_config_source = output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml`

Why:

- the user has now named the exact canonical pair and datasets for the first implementation,
- freezing them removes an avoidable source of ambiguity,
- and alias-based manifests still keep the harness reusable for future checkpoint additions.

This also makes the harness reusable for:

- a future stronger-geometry 2B checkpoint,
- and a future random-order Stage-1 SFT checkpoint.

### Decision 11: Separate logical analysis cells from execution shards

The study will distinguish:

- `logical_cell_id`: one analysis condition defined by checkpoint, subset, decode family, prefix mode, length mode, and prompt contract.
- `execution_shard_id`: one physical execution unit for a `logical_cell_id`, such as one `sample_idx`, one GPU assignment, or one retry.

Why:

- `K`-sampling naturally produces multiple physical runs for one analysis cell,
- resume/retry logic is cleaner when it is explicit,
- and multi-GPU provenance stays accurate without overloading the meaning of “cell.”

Merged per-GT outputs must be written in deterministic sort order:

1. `logical_cell_id`
2. `execution_shard_id`
3. `record_idx`
4. `gt_idx`

## Canonical Experiment Program

The study is intentionally a staged experiment design first and an implementation target second. The goal is not to throw every factor into one flat sweep, but to use a fixed subset and progressively increase intervention strength while preserving causal interpretability.

Each stage should emit its own manifest or equivalent frozen provenance artifact so later stages can be resumed from earlier outputs without recomputing the full matrix. The minimum recommended stage-manifest family is:

- `bootstrap_manifest.json`
- `subset_manifest.json`
- `baseline_manifest.json`
- `sampling_manifest.json`
- `prefix_manifest.json`
- `length_manifest.json`
- `report_manifest.json`

### Bootstrap Stage: Hard-Case Selector

Purpose:

- produce one deterministic ranking over a broader frozen candidate pool for one dataset split,
- derive `Hard-32` and `Hard-16` from that one ranking,
- and resolve the hard cohorts before the canonical study begins.

Mandatory contract:

- checkpoint in `{original, a_only}`,
- dataset split in `{train, val}`,
- prefix mode `image_only`,
- decode family `greedy`,
- length mode `default` only,
- one frozen candidate pool,
- and rollout-health gating before images enter the final ranking.

Outputs:

- bootstrap selector manifest,
- split-specific image-level difficulty ranking table,
- `Hard-32` manifest for that split,
- and `Hard-16` manifest for that split derived as the top-16 prefix of `Hard-32`.

### Stage 0: Study Subset Freeze

Purpose:

- lock the exact split-specific images before any rollout cell runs,
- consume the bootstrap selector outputs without mutating them,
- and ensure every later comparison is cell-matched on the same image order.

Default shape:

- four named fixed cohorts:
  - `train / Hard-16`
  - `train / Hard-32`
  - `val / Hard-16`
  - `val / Hard-32`
- each cohort intentionally drawn from the poorest samples inside its own dataset split,
- with emphasis on:
  - known hard crowded scenes,
  - monitor images already discussed by the user,
  - images with persistent strict-FN burden,
  - and images where annotation mismatch vs true incapacity is genuinely ambiguous.

Outputs:

- resolved manifest with explicit image ids and image order,
- dataset split and split JSONL,
- per-image GT inventory,
- subset-selection provenance showing why each image entered its split-specific `Hard-16` or `Hard-32`,
- and a stable study subset id reused by all later stages.

### Stage 1: Deterministic Baseline

Purpose:

- establish the reference rollout for each checkpoint,
- identify baseline misses on the exact study subset,
- and create the initial qualitative panel.

Mandatory cells:

- checkpoint in `{original, a_only}`,
- dataset split in `{train, val}`,
- prefix mode `image_only`,
- decode family `greedy`,
- length mode `default`.

Outputs:

- deterministic `gt_vs_pred` artifacts,
- per-image summaries,
- rollout-health summaries,
- and the initial baseline FN set that later stages will try to recover.

### Stage 2: Image-Only Sampling Coverage

Purpose:

- answer whether a baseline FN is truly never emitted or just not chosen under greedy decode,
- before any prefix manipulation is introduced.

Mandatory cells:

- same contract as Stage 1,
- but with sampled decode and union-of-`K` aggregation,
- using one shared seed ladder per checkpoint.

Validity rule:

- sampled cells must pass the rollout-health gate before they enter the main sampling-recovery tables.

Recommended first pass:

- one low-to-moderate temperature cell,
- one moderate-coverage `K` such as `8`,
- and an optional larger `K` such as `32` only after the first pass is stable.

Interpretation:

- if a GT object is recovered here, it becomes `decode_selection_miss`,
- and later prefix or length recovery should not override that attribution.

### Stage 3: Prefix-Order Matrix

Purpose:

- test whether FN depends on the ordering and content of the early rollout prefix.

Mandatory prefix families:

- `oracle_gt_prefix_train_order`
- `oracle_gt_prefix_random_order`
- `self_prefix`

Mandatory factors:

- checkpoint in `{original, a_only}`,
- prefix length in `{1, 2, 4, 8}`,
- length mode `default`,
- and continuation-only scoring.

Validity rule:

- prefix-conditioned cells must preserve valid continuation artifacts and pass the rollout-health gate before they enter main prefix-attribution tables.

Interpretation:

- train-order recovery suggests exposure to the training serialization helps continuation,
- random-order degradation suggests an order prior,
- self-prefix behavior measures whether the model can continue its own rollout more reliably than externally supplied prefixes.

### Stage 4: Switched And Broken Prefix Stress Tests

Purpose:

- test whether continuation is checkpoint-specific and brittle to small local prefix corruption.

Mandatory families:

- `switched_prefix`
- `broken_prefix`

Minimum broken-prefix mutations:

- deletion,
- adjacent swap,
- single-object insertion.

Recommended scope:

- run this stage on the same fixed subset,
- but prioritize images and GT objects that remain unresolved or become prefix-sensitive after Stages 1-3.

### Stage 5: Sequence-Length Cross-Checks

Purpose:

- test whether unresolved FN is blocked by the length budget or early stopping behavior,
- after the main prefix explanations have already been explored.

Mandatory comparison:

- default `max_new_tokens`
- extended `max_new_tokens`

Validity rule:

- length cells are interpreted only if both matched cells remain rollout-healthy enough to compare.

Outputs:

- per-rollout token count,
- object count,
- stop reason,
- EOS position when available,
- and additional recovered-vs-added-unmatched comparisons.

Interpretation:

- recovery here suggests a length-sensitive blocking effect,
- while no recovery but more unmatched output suggests the model is not simply being cut off too early.

### Stage 6: Recovery Attribution And Manual Audit

Purpose:

- turn cell-level outcomes into object-level explanations,
- and separate strict-metric misses from likely annotation or matching ambiguity.

Required outputs:

- object-level recovery table with minimal-intervention precedence,
- supporting quality flags such as `annotation_mismatch_candidate`,
- and a review queue with overlays for each main bucket.

Primary buckets:

- `deterministic_hit`
- `decode_selection_miss`
- `prefix_sensitive_miss`
- `length_bias_miss`
- `persistent_unrecovered`

### Stage 7: Synthesis Report

Purpose:

- summarize how the two checkpoints differ on the same images,
- and answer the user’s core question: whether FN reflects incapacity or rollout blocking.

The final report should answer, per checkpoint and per subset:

- which GT objects are already robustly found,
- which appear under sampling but not greedy,
- which depend on train-order vs random-order vs rollout-order prefixes,
- which depend on longer sequences,
- and which stay unrecovered under all tested interventions.

## Primary Matrix And Expansion Policy

The canonical run is the full matrix over the fixed small subset, but “full matrix” here means all mandatory stages above, not an unconstrained Cartesian product over every conceivable temperature, mutation, and prefix variant.

Primary matrix:

- bootstrap selector over one frozen candidate pool per dataset split
- two checkpoints: `original`, `a_only`
- two dataset splits: `train`, `val`
- two fixed subsets per split: `Hard-16`, `Hard-32`
- deterministic baseline with default length
- image-only sampled coverage
- full prefix-order matrix
- switched/broken prefix stress tests
- sequence-length cross-checks
- rollout-health gating
- object-level attribution and review queue

Expansion policy:

- add more `K`,
- add more temperatures,
- add more random-order seeds,
- or add more broken-prefix mutations

only after the primary matrix is complete and interpretable.

## Risks / Trade-offs

- **[Risk] Prefix intervention becomes too custom to one backend** → Keep the first version HF-only so the study is easy to interpret and debug.
- **[Risk] Factor grid explodes combinatorially** → Use the staged program and primary-matrix policy so “full exploration” still stays interpretable and finite.
- **[Risk] Results are dominated by annotation incompleteness rather than model behavior** → Preserve qualitative audit artifacts and annotate conclusions as strict-metric vs human-semantic when needed.
- **[Risk] The missing `a_only` step-300 weights make exact replay impossible** → Use explicit checkpoint manifests and treat recorded dumps as reference-only evidence, not executable checkpoints.
- **[Risk] Random-order future checkpoints confound the current fixed-checkpoint study** → Keep the current study workflow checkpoint-agnostic but freeze each comparison pair per run; future checkpoints are added as new manifests, not mixed into old outputs.
- **[Risk] Multi-GPU orchestration introduces hidden non-determinism** → Shard by logical cell/execution shard, not by model parallelism; each shard records GPU id, seed, and resolved config independently.
- **[Risk] Small subsets overfit the narrative to a few images** → Require explicit image-id provenance and preserve per-image audit outputs so conclusions stay grounded in concrete examples.

## Migration Plan

1. Add the new study capability and its spec.
2. Implement the bootstrap selector over frozen `train` and `val` candidate pools and write split-specific `Hard-32` / `Hard-16` manifests.
3. Implement manifest-driven checkpoint comparison over those frozen subsets.
4. Implement Stages 1-2 first so deterministic and sampling evidence are available before prefix intervention.
5. Implement Stage 3, the full prefix-order matrix, using the continuation partition contract.
6. Implement Stage 4, switched/broken prefix stress tests.
7. Implement Stage 5 sequence-length cross-checks, then Stage 6 object-level attribution and Stage 7 reporting.
8. Validate on the initial `original` vs `a_only` pair across both `train` and `val`, then extend to future checkpoints.

Rollback strategy:

- because this is a study workflow rather than a core serving path, rollback is simply:
  - stop using the new analysis configs/scripts,
  - preserve already written artifacts,
  - and continue using existing inference/eval flows.

## Open Questions

- Should sequence-length bias be analyzed only via `max_new_tokens` sweeps, or should the study also record per-step EOS-vs-object logits when available?
- How much qualitative manual audit is required before labeling a GT object as “likely annotation mismatch” rather than true FN?
- Should the first bootstrap pass over `train` use the full `117247`-image split or a frozen larger candidate pool cap for runtime control, as long as that cap itself is manifested and reproducible?
