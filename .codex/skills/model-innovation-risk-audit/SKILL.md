---
name: model-innovation-risk-audit
description: "Use when reviewing a planned or newly wired CoordExp model/objective/data/tokenizer/decode/eval/runtime change before trusting training or eval claims, especially for silent train/eval/config/artifact mismatch risk."
---

# Model Innovation Risk Audit

Default to read-only audit. This skill is a **contract and provenance gate**, not a symptom debugger. The goal is to find mismatches that do not crash, can pass smokes, and can produce normal-looking losses while changing the actual training signal or eval claim.

## Role Boundary

Use this skill when the question is:

- "Can we trust this new mechanism, config, data path, tokenizer/template, loss, decode path, eval path, or runtime integration?"
- "Could this innovation silently train or evaluate a different contract than intended?"
- "Before launching or interpreting a run, are schema, materialized config, data, loss, decode/eval, and artifacts aligned?"

Do **not** use this skill as the main tool when the user already has a concrete behavioral symptom such as a metric drop, FP/FN shift, invalid rollout spike, duplication burst, length collapse, train/eval divergence, or optimization instability. Start with `model-diagnosis` for those symptoms, then return here only if the diagnosis points to silent config/runtime/eval contract drift.

Hand off to `model-diagnosis` when the contract appears wired correctly but behavior remains unproven or abnormal.

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

Probe goal: prove or falsify a contract mismatch. Do not explain a metric regression from aggregate scores alone; that is `model-diagnosis` territory.

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

## Stage-2 Production Gate

For Stage-2 rollout-correction, residual-set, trie, or vLLM candidates, return a bounded launch/promote decision instead of an open-ended audit. Check:

- config load and resolved pipeline namespace;
- prepared-data versus live-rollout mode;
- prompt/template/decode parity and vLLM adapter/token-row synchronization;
- parser health, invalid/drop counters, and metric-bearing eval validity;
- residual, trie, duplicate, and unlikelihood counters with scope labels;
- `run_metadata.json`, `resolved_config.json`, `pipeline_manifest.json`, and artifact roots;
- whether a paired baseline or ablation is required before promotion.

Output one of: `promote`, `hold`, `rerun gate`, or `needs user decision`, with the smallest verification command or artifact check that would change the verdict.

## References

Load only when needed:

- `references/risk-taxonomy.md`
- `references/subagent-prompts.md`
- `references/report-template.md`
