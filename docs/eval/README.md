---
doc_id: docs.eval.index
layer: docs
doc_type: router
status: canonical
domain: eval
summary: Router for inference and evaluation contracts, workflows, and artifacts.
tags: [eval, infer, workflow]
updated: 2026-03-10
---

# Evaluation & Inference

Use this folder for the current infer -> score -> evaluate workflow and the additive Oracle-K repeated-sampling analysis path.

## Read Order

1. [CONTRACT.md](CONTRACT.md)
2. [WORKFLOW.md](WORKFLOW.md)
3. [COCO_TEST_SUBMISSION.md](COCO_TEST_SUBMISSION.md) for official test-dev benchmarking
4. [UNMATCHED_PROPOSAL_VERIFIER_STUDY.md](UNMATCHED_PROPOSAL_VERIFIER_STUDY.md) for the offline unmatched-proposal verifier ablation
5. [../ARTIFACTS.md](../ARTIFACTS.md)

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - evaluator inputs, outputs, invariants, and failure modes
- [WORKFLOW.md](WORKFLOW.md)
  - YAML-first operational flow from inference to visualization, including Oracle-K analysis
- [COCO_TEST_SUBMISSION.md](COCO_TEST_SUBMISSION.md)
  - end-to-end runbook for 1024-budget COCO test-dev inference and official submission export
- [UNMATCHED_PROPOSAL_VERIFIER_STUDY.md](UNMATCHED_PROPOSAL_VERIFIER_STUDY.md)
  - offline ablation workflow for commitment / counterfactual proposal verification
- [../ARTIFACTS.md](../ARTIFACTS.md)
  - run artifacts and provenance surfaces

## Use This Router For

- "What JSONL does the evaluator expect?"
- "What is the current production workflow?"
- "How do I run a real COCO test-dev benchmark and upload it?"
- "Which artifacts should exist after a valid infer/eval run?"
- "How do I compare one baseline decode against repeated stochastic rollouts?"
- "How do I run the unmatched-proposal verifier ablation on a small COCO subset?"
