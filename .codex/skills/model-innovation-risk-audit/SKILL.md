---
name: model-innovation-risk-audit
description: "Use when a planned or newly wired CoordExp mechanism, objective, loss, or eval path needs a pre-launch or pre-interpretation trust gate before training/eval claims, especially for silent train/eval/config/artifact mismatch risk."
---

# Model Innovation Risk Audit

Default to read-only audit. This skill is a **contract and provenance gate**, not a symptom debugger. The goal is to find mismatches that do not crash, can pass smokes, and can produce normal-looking losses while changing the actual training signal or eval claim.

## Role Boundary

Use this skill when the question is:

- "Can we trust this new mechanism, config, data path, tokenizer/template, loss, decode path, eval path, or runtime integration?"
- "Could this innovation silently train or evaluate a different contract than intended?"
- "Before launching or interpreting a run, are schema, materialized config, data, loss, decode/eval, metrics, and artifacts aligned?"

Do **not** use this skill as the main tool when the user already has a concrete behavioral symptom such as a metric drop, FP/FN shift, invalid rollout spike, duplication burst, length collapse, train/eval divergence, or optimization instability. Start with `model-diagnosis` for those symptoms, then return here only if the diagnosis points to silent config/runtime/eval contract drift.

Hand off to `model-diagnosis` when the contract appears wired correctly but behavior remains unproven or abnormal.

## Minimal Contract Diff

For every innovation, produce a compact contract diff:

- intended contract: design, spec, or plan;
- authored config: YAML keys and inheritance chain;
- resolved contract: materialized config and schema dataclass;
- runtime contract: dataset, collator, trainer, loss, decode, and eval objects actually used;
- artifact contract: resolved config, manifests, parser/drop counters, and metric keys;
- evidence verdict: matched, mismatched, or unproven.

Do not trust a smoke run if any row silently falls back to legacy behavior.

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

## Config And Loss Footguns

- `ConfigLoader.load_yaml_with_extends()` deep-merges dicts but replaces lists wholesale; inspect final resolved objective lists.
- Reject legacy keys in semantic modes instead of warning or ignoring.
- Separate monitoring-only knobs from differentiable objective weights.
- Verify raw loss terms and effective weighted contributions are both logged.
- Check zero-weight targets, EOS/type-gate composition, duplicate multiplicity, and teacher-token membership.
- Use deterministic tiny-logit tests for scalar formulas before trusting training curves.

## Loss/Numerics Gate

For new or changed objectives, verify one authored config, one resolved config, one encoded sample, one collated batch, and one deterministic tiny-logit formula probe before trusting training curves.

Report raw and weighted loss terms, valid-count denominators, mask density, zero-mask behavior, target support membership, finite checks, dtype/fp32 islands, gradient path, accumulation, distributed reduction semantics, and metric/logging names for each term. If wiring is correct but behavior is already abnormal, hand off to `model-diagnosis`.

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
Minimal Contract Diff
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

## Launch/Promote Gate

For high-risk model innovations, return a bounded launch/promote decision instead of an open-ended audit. For Stage-2 rollout-correction, residual-set, trie, or vLLM candidates, check:

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
- `references/contract-diff.md`
