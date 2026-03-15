# Unmatched Proposal Verifier Blueprint

This file is the authoritative handoff blueprint for continuing the
`unmatched-proposal-verifier-ablation` change in a fresh conversation.

It is intentionally more execution-oriented than `proposal.md` /
`design.md` / `spec.md` and is meant to answer:

- what the study is really trying to prove,
- what has already been established,
- what evidence is still missing,
- and exactly how to continue without repeating past mistakes.

## 1. Study Intent

The real question is not merely:

- “does `counterfactual` beat `commitment` on synthetic negatives?”

The real question is:

- “when a rollout proposal is unmatched to GT, can we trust a verifier proxy
  enough to use that proposal later for soft pseudo-label promotion?”

That requires stronger evidence than a clean teacher-forced benchmark alone.

## 2. Why The Previous Runs Were Insufficient

The previous implementation established useful but incomplete evidence:

- on the clean GT-vs-hard-negative slice,
  `counterfactual` consistently outperformed plain `commitment`.

However, that evidence is not enough for a trustworthy unmatched-proposal
conclusion because:

1. the successful full runs mainly supported the clean GT slice,
2. rollout unmatched evidence did not reliably materialize into usable
   `matched_vs_unmatched` comparisons,
3. temperature effects were not trustworthy when rollout collection validity was
   not explicitly gated,
4. there is still no manual unmatched audit anchoring what high-scoring
   unmatched proposals actually mean.

Therefore, the study must now be treated as an authority-first evidence
pipeline rather than a single monolithic run.

## 3. Current Repository State

### Code / Runtime State

Implementation for the original offline study harness was merged into `main`.

Relevant code paths already present:

- runner:
  - `scripts/analysis/run_unmatched_proposal_verifier.py`
- main module:
  - `src/analysis/unmatched_proposal_verifier.py`
- parsing helper:
  - `src/trainers/rollout_matching/parsing.py`
- configs:
  - `configs/analysis/unmatched_proposal_verifier/default.yaml`
  - `configs/analysis/unmatched_proposal_verifier/smoke.yaml`
- docs:
  - `docs/eval/UNMATCHED_PROPOSAL_VERIFIER_STUDY.md`

### OpenSpec State

The active change has been rewritten toward a stricter evidence standard:

- `openspec/changes/unmatched-proposal-verifier-ablation/proposal.md`
- `openspec/changes/unmatched-proposal-verifier-ablation/design.md`
- `openspec/changes/unmatched-proposal-verifier-ablation/specs/unmatched-proposal-verifier-study/spec.md`
- `openspec/changes/unmatched-proposal-verifier-ablation/tasks.md`

### Git / Workspace State

At the time this blueprint was written:

- `main` contains the implementation merge,
- the four OpenSpec change files above are locally modified but not yet
  committed in the current working tree,
- this blueprint file is also new and uncommitted.

## 4. What Has Been Successfully Verified

### 4.1 Smoke Harness

An end-to-end smoke harness has been run successfully for:

- `1 checkpoint`
- `8 sampled images`

Smoke outputs exist at:

- `output/analysis/unmatched-proposal-verifier-smoke/`

Confirmed smoke artifacts include:

- subset meta
- `gt_vs_pred.jsonl`
- `eval/matches.jsonl`
- GT positive / hard-negative tables
- proposal score tables
- checkpoint summary
- top-level report
- unmatched audit pack

### 4.2 Clean GT Slice Signal

Across successful runs, the clean GT-vs-hard-negative slice consistently showed:

- `counterfactual > commitment`
- `combined_linear` did not clearly beat `counterfactual`

This is the strongest currently trustworthy conclusion.

### 4.3 Full Temperature Sweep Outcomes So Far

Two full temperature sweep attempts were made.

The most informative one is:

- `output/analysis/unmatched-proposal-verifier-temperature-sweep-v2/`

Successful temperatures:

- `0.0`
- `0.1`
- `0.5`
- `1.0`

Failed temperatures:

- `0.05`
- `0.2`
- `0.3`
- `0.7`

But even the successful temperatures did **not** yield trustworthy rollout
unmatched conclusions, because the rollout-facing metrics were not informative
enough.

## 5. What Must Not Be Over-Claimed

Do **not** claim from the current results that:

- unmatched rollout proposals are already trustworthy pseudo-label candidates,
- temperature has no effect on proposal quality,
- or the verifier is already promotion-ready.

The current trustworthy claim is narrower:

- `counterfactual` appears to be the strongest single proxy on the clean
  GT-vs-hard-negative benchmark.

## 6. Authority-First Evidence Stack

The study should now be executed through three evidence layers.

### Layer A: Clean Verifier Benchmark

Purpose:

- determine whether the verifier proxies themselves carry grounding signal.

Inputs:

- GT positives
- GT-derived hard negatives

This layer answers:

- strongest single proxy?
- does combination help?

This layer does **not** answer:

- whether unmatched rollout proposals are promotion-ready.

### Layer B: Rollout Proposal Benchmark

Purpose:

- test the verifier on real rollout proposal populations.

This layer must be split into:

#### B1. Collection Validity Gate

Required outputs per checkpoint × temperature:

- `pred_count_total`
- `pred_count_per_image_mean`
- `nonempty_pred_image_rate`
- `matched_count`
- `unmatched_count`
- `ignored_count`
- `invalid_rollout_count`
- parser failure counts
- duplicate-like rate
- `collection_valid`
- `collection_invalid_reason`

Only collection-valid runs may enter the authoritative temperature comparison.

Suggested initial gate:

- `nonempty_pred_image_rate >= 0.30`
- `pred_count_total >= 100`
- `unmatched_count >= 50`

#### B2. Rollout Proposal Scoring

Only collection-valid runs should produce authoritative rollout comparison
tables.

Required outputs:

- matched-vs-unmatched separation
- unmatched top-k proxy analysis
- proposal score distributions
- checkpoint × temperature proposal stats

### Layer C: Small Manual Audit Benchmark

Purpose:

- establish what high-scoring unmatched proposals actually are.

Target size:

- roughly `96` to `128` unmatched proposals total

Required stratification:

- checkpoint
- temperature
- score quantile
- nearest-GT weak-overlap bucket

Suggested audit labels:

- `real_visible_object`
- `duplicate_like`
- `wrong_location`
- `dead_or_hallucinated`
- `uncertain`

Without this layer, the final recommendation must remain downgraded.

## 7. Authoritative Temperature Scope

The main temperature sweep should be exactly:

- `0.0`
- `0.3`
- `0.5`
- `0.7`

Interpretation rule:

- only these four temperatures may enter the authoritative main conclusion
  tables,
- any other temperature is appendix-only / exploratory,
- do not mix them into the final headline result.

## 8. Recommended Staged Execution Plan

This is the recommended continuation order.

### Stage 0: Commit The Spec Reset

Before touching implementation again:

- commit the updated OpenSpec files plus this blueprint,
- so the next conversation starts from a clean declared plan.

### Stage 1: Refactor Into Stages

Refactor `src/analysis/unmatched_proposal_verifier.py` into explicit stages:

1. subset / GT preparation
2. rollout collection
3. collection-health summarization
4. rollout scoring
5. report aggregation
6. manual audit prep / ingestion

Each stage should emit a manifest.

Suggested manifests:

- `subset_manifest.json`
- `collection_manifest.json`
- `scoring_manifest.json`
- `report_manifest.json`
- `manual_audit_manifest.json`

### Stage 2: Collection-Only Sanity Run

Run a smaller collection-only matrix first:

- subset: `64` images
- checkpoints: the 2 merged UL checkpoints
- temperatures: `0.0 / 0.3 / 0.5 / 0.7`

Goal:

- determine which checkpoint × temperature runs are collection-valid.

Do **not** start interpreting proxy conclusions yet.

### Stage 3: Authoritative Full Scoring Run

Only for collection-valid temperatures:

- subset: `200` images
- same 2 checkpoints
- same 4 temperatures

Run:

- clean GT slice scoring
- rollout proposal scoring
- aggregate reporting

### Stage 4: Manual Audit

Prepare and label the unmatched audit subset:

- `96-128` proposals total
- stratified as defined above

Then rerun final aggregation using the audit labels.

### Stage 5: Final Recommendation

Only then answer:

1. strongest single proxy?
2. does combination help?
3. stable across checkpoints?
4. stable across authoritative temperatures?
5. good enough for soft pseudo-label promotion?

## 9. Final Recommendation Standard

A strong pseudo-label-promotion recommendation should only be made if all of
the following hold:

1. `counterfactual` clearly beats `commitment` on Layer A.
2. most checkpoint × temperature runs pass the collection-validity gate.
3. rollout proposal scoring remains informative on Layer B.
4. manual audit shows that high-scoring unmatched proposals are often
   `real_visible_object`.

If any of these fail, the report must explicitly downgrade the conclusion to:

- `promising but not yet promotion-ready`

## 10. Immediate To-Do For The Next Conversation

When continuing in a fresh conversation, the first concrete actions should be:

1. Commit:
   - `proposal.md`
   - `design.md`
   - `spec.md`
   - `tasks.md`
   - `blueprint.md`
2. Validate the change:
   - `openspec validate unmatched-proposal-verifier-ablation --type change --strict --no-interactive`
3. Do **not** resume the old broad temperature sweep.
4. Implement the staged workflow and collection-validity gate first.
5. Only then run the authoritative four-temperature collection sanity matrix.

## 11. Files To Open First In The Next Conversation

- `openspec/changes/unmatched-proposal-verifier-ablation/blueprint.md`
- `openspec/changes/unmatched-proposal-verifier-ablation/tasks.md`
- `openspec/changes/unmatched-proposal-verifier-ablation/specs/unmatched-proposal-verifier-study/spec.md`
- `src/analysis/unmatched_proposal_verifier.py`
- `configs/analysis/unmatched_proposal_verifier/default.yaml`
- `docs/eval/UNMATCHED_PROPOSAL_VERIFIER_STUDY.md`
