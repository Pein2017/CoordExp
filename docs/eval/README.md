---
doc_id: docs.eval.index
layer: docs
doc_type: router
status: canonical
domain: eval
summary: Router for inference and evaluation contracts, workflows, and artifacts.
tags: [eval, infer, workflow]
updated: 2026-03-09
---

# Evaluation & Inference

Use this folder for the current infer -> score -> evaluate workflow.

## Read Order

1. [CONTRACT.md](CONTRACT.md)
2. [WORKFLOW.md](WORKFLOW.md)
3. [../ARTIFACTS.md](../ARTIFACTS.md)

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - evaluator inputs, outputs, invariants, and failure modes
- [WORKFLOW.md](WORKFLOW.md)
  - YAML-first operational flow from inference to visualization
- [../ARTIFACTS.md](../ARTIFACTS.md)
  - run artifacts and provenance surfaces

## Use This Router For

- "What JSONL does the evaluator expect?"
- "What is the current production workflow?"
- "Which artifacts should exist after a valid infer/eval run?"
