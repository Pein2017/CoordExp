---
name: model-innovation-risk-audit
description: Use when auditing a new CoordExp model/objective/data/tokenizer/decode/eval/runtime innovation for silent train/eval mismatch risk.
---

# Model Innovation Risk Audit

Default to read-only audit. The goal is to find mismatches that do not crash, can pass smokes, and can produce normal-looking losses while changing the actual training signal or eval claim.

## Contract Triangulation

Extract the intended algorithm contract, then compare it across:

- design intent and config schema;
- materialized config;
- data JSONL, geometry, ordering, image roots;
- tokenizer/chat template, EOS/pad/stops, assistant spans;
- dataset/collator labels, masks, sidecars, padding, offsets;
- model forward logits, shifts, slicing, trainable groups;
- loss support, weights, EOS/type gates, precision;
- metrics and effective weighted contributions;
- decode/eval/parser behavior;
- artifacts and manifests.

## Probes

Prefer narrow evidence:

- real tokenizer ids and generation config;
- resolved config;
- one encoded sample and one collated batch;
- tiny logits loss calculation;
- JSONL/image scan;
- artifact manifest check;
- targeted unit test or smoke command.

If the contract is wired plausibly but behavior remains unproven, hand off to `model-diagnosis` for a tiny probe before expensive training.

## Findings-First Report

Use:

```text
Findings
Confirmed OK
Decision Questions
Patch Recommendations
Unit Tests And Diagnostics
Smoke Run Suggestions
Residual Risks
```

Finding format:

```text
[P0/P1/P2/P3] Title
Evidence
Impact
Fix direction
Minimal test/diagnostic
```

Severity:

- `P0`: current implementation likely trains/evals the wrong objective or corrupts core results.
- `P1`: silent mismatch can plausibly alter training signal, decode behavior, or eval validity.
- `P2`: reproducibility, diagnostics, maintainability, or edge-case risk.
- `P3`: future-proofing or cleanup.

Do not inflate severity because a finding is interesting.

## Parallel Audit Split

When independent surfaces exist and subagents are available, split by disjoint scope:

- objective/loss;
- dataset/collator;
- tokenizer/template/decode;
- config/runtime;
- artifact/eval;
- tests/diagnostics.

## References

Load only when needed:

- `references/risk-taxonomy.md`
- `references/subagent-prompts.md`
- `references/report-template.md`
