---
doc_id: docs.eval.index
layer: docs
doc_type: router
status: canonical
domain: eval
summary: Router for inference and evaluation contracts, workflows, and artifacts.
tags: [eval, infer, workflow]
updated: 2026-05-03
---

# Evaluation & Inference

Use this folder for the current infer -> score -> evaluate workflow, official
COCO export, and additive analysis studies.

## Read Order

1. [CONTRACT.md](CONTRACT.md)
2. [WORKFLOW.md](WORKFLOW.md)
3. [../ARTIFACTS.md](../ARTIFACTS.md)
4. [COCO_TEST_SUBMISSION.md](COCO_TEST_SUBMISSION.md) for official test-dev benchmarking

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - evaluator inputs, record shape, coordinate handling, scoring rules, output invariants, failure policy, and shared visualization contract
- [WORKFLOW.md](WORKFLOW.md)
  - YAML-first operational flow from inference to visualization, including raw-text norm1000, non-canonical bbox caveats, duplicate control, LVIS proxy views, and Oracle-K analysis
- [../ARTIFACTS.md](../ARTIFACTS.md)
  - full infer/eval/training artifact inventory, provenance, helper ownership, and run-directory expectations
- [COCO_TEST_SUBMISSION.md](COCO_TEST_SUBMISSION.md)
  - end-to-end runbook for 1024-budget COCO test-dev inference and official submission export
- [UNMATCHED_PROPOSAL_VERIFIER_STUDY.md](UNMATCHED_PROPOSAL_VERIFIER_STUDY.md)
  - draft supplementary study; not part of the default infer -> score -> evaluate path

## Normative Specs

Use these only when exact stable contract semantics matter:

- [`inference-pipeline/spec.md`](../../openspec/specs/inference-pipeline/spec.md)
- [`inference-engine/spec.md`](../../openspec/specs/inference-engine/spec.md)
- [`detection-evaluator/spec.md`](../../openspec/specs/detection-evaluator/spec.md)
- [`runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md)

## Use This Router For

- "What JSONL does the evaluator expect?"
- "What is the current production workflow?"
- "How do I run a real COCO test-dev benchmark and upload it?"
- "Which artifacts should exist after a valid infer/eval run?"
- "How do I compare one baseline decode against repeated stochastic rollouts?"
- "How do I run the unmatched-proposal verifier ablation on a small COCO subset?"
