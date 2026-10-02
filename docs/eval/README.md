---
doc_id: docs.eval.index
layer: docs
doc_type: router
status: canonical
domain: eval
summary: Short route to inference and evaluation owners.
tags: [eval, infer, router]
updated: 2026-09-30
---

# Evaluation and inference

This folder routes readers to the current evaluator owners; operational instructions live in the `coordexp-infer-eval-workflow` Skill.

- Current commands and output-root selection: the Skill's `references/current-workflow.md`.
- Artifact compatibility: [CONTRACT.md](CONTRACT.md) and the owning inference, scoring, and evaluator specs.
- Metric meaning: the current research owner.
- Broader run artifacts: [ARTIFACTS.md](../ARTIFACTS.md).
- Historical COCO test-dev: the Skill reference `historical-coco-testdev.md`; verify current scope before use.

Full-dataset and official test-dev runs are separate from the ordinary Swift V1 val200 path.
