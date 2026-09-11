---
doc_id: docs.eval.index
layer: docs
doc_type: router
status: canonical
domain: eval
summary: Router for inference and evaluation contracts, workflows, and artifacts.
tags: [eval, infer, workflow]
updated: 2026-07-11
---

# Evaluation & Inference

Use this folder for current Swift inference, scoring, evaluation and metric
interpretation. Historical export and analysis routes are labeled separately.

coordexp-infras note:

- Start with the named config/artifact and select the relevant page below.
- The accepted Swift V1 validation gate is the fixed val200 inference/eval run,
  not a full validation-dataset run.
- Full validation-dataset or official test-dev evaluation remains optional and
  should be launched only when explicitly requested.

## Select the task

Use [WORKFLOW.md](WORKFLOW.md) for a run, [CONTRACT.md](CONTRACT.md) for
artifact compatibility, and [INTERPRETATION.md](INTERPRETATION.md) for metric,
matching, category or annotation questions. Read [../ARTIFACTS.md](../ARTIFACTS.md)
only when artifact ownership is unclear. Official test-dev export has a separate
[submission scope](COCO_TEST_SUBMISSION.md); it is not a prerequisite for a local
research comparison.

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - evaluator inputs, record shape, coordinate handling, scoring rules, output invariants, failure policy, and shared visualization contract
- [WORKFLOW.md](WORKFLOW.md)
  - current Swift inference, selected-token scoring, aggregate COCO evaluation and direct run visualization
- [../ARTIFACTS.md](../ARTIFACTS.md)
  - full infer/eval/training artifact inventory, provenance, helper ownership, and run-directory expectations
- [COCO_TEST_SUBMISSION.md](COCO_TEST_SUBMISSION.md)
  - historical mainline runbook for 1024-budget COCO test-dev inference and
    official submission export; verify current Swift support separately
- [drafts/UNMATCHED_PROPOSAL_VERIFIER_STUDY.md](drafts/UNMATCHED_PROPOSAL_VERIFIER_STUDY.md)
  - draft supplementary study; not part of the default infer -> score -> evaluate path

- [Legacy evaluation reference](../history/evaluation/2026-09-09-legacy-eval-reference.md)
  - preserved MS-Swift/mainline commands and schemas for historical reconstruction

## Normative Specs

Use these only when exact stable contract semantics matter:

- [`coordexp-infras-infer-pipeline/spec.md`](../../openspec/specs/coordexp-infras-infer-pipeline/spec.md)
- [`coordexp-infras-infer-backend-trace/spec.md`](../../openspec/specs/coordexp-infras-infer-backend-trace/spec.md)
- [`coordexp-infras-infer-scoring-artifacts/spec.md`](../../openspec/specs/coordexp-infras-infer-scoring-artifacts/spec.md)
- [`coordexp-infras-detection-evaluator/spec.md`](../../openspec/specs/coordexp-infras-detection-evaluator/spec.md)

## Use This Router For

- "What JSONL does the evaluator expect?"
- "What is the current production workflow?"
- "How do I run a real COCO test-dev benchmark and upload it?"
- "Which artifacts should exist after a valid infer/eval run?"
- "Is val200 enough for the coordexp-infras V1 readiness gate?"
- "How do I compare one baseline decode against repeated stochastic rollouts?"
- "How do I run the unmatched-proposal verifier ablation on a small COCO subset?"
