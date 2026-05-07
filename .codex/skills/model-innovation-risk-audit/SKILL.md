---
name: model-innovation-risk-audit
description: Use when auditing a new model, training objective, loss, data/collator path, tokenizer/template change, decoding/evaluation change, packing/runtime change, or algorithmic innovation before production training or serious ablation. Focuses on hidden mismatch risks that may not crash, may not show in loss metrics, and may silently degrade training or evaluation validity.
---

# Model Innovation Risk Audit

## Mission

Audit a proposed or implemented model/training innovation before serious training, ablation, or production validation.

The goal is not to prove the code runs. The goal is to find hidden mismatch risks that:

- do not raise exceptions,
- can pass smoke tests,
- can produce normal-looking loss curves,
- can evade ordinary runtime metrics,
- but change the actual training signal, decode contract, eval meaning, or reproducibility claim.

Treat the innovation as a contract that must remain consistent across design, config, data, collator, model forward, logits, loss, metrics, decode, evaluation, and artifacts.

## When To Use

Use this skill when the user asks to audit or prepare any new innovation involving:

- model fine-tuning objectives,
- loss functions,
- multi-positive targets,
- autoregressive target construction,
- tokenizer or chat-template behavior,
- EOS / stop-token semantics,
- data serialization or geometry,
- dataset/collator sidecars,
- packing, padding, logits slicing, or precision,
- distributed effective batch semantics,
- inference/decode/eval pipeline changes,
- artifact or manifest provenance for research claims.

Also use when the user says phrases like:

- hidden risk, mismatch, misalignment, silent failure, or train/eval parity,
- production training, serious ablation, full-val, paper-facing, or launch gate,
- loss metrics cannot show it,
- 不会报错但会影响训练,
- 错配 / 错位 / 隐藏风险,
- 检查训练和 decode 是否一致.

## Operating Rules

Default to read-only audit. Do not patch code unless the user explicitly asks for fixes.

First extract the intended algorithm contract, then audit implementation surfaces against that contract.

Prefer evidence over intuition:

- exact source references,
- materialized config values,
- real tokenizer/model probes when relevant,
- synthetic invariant tests,
- small smoke runs,
- artifact metadata checks,
- command outputs with scope labels.

When the audit has multiple independent surfaces and subagents are available, use parallel subagents.

If fixes are requested, use test-first changes for behavior-affecting patches. Patch only the smallest owner surface that can enforce the contract.

## Core Method: Contract Triangulation

For every innovation, triangulate these layers:

| Layer | Question |
|---|---|
| Design intent | What should the algorithm do mathematically and operationally? |
| Config schema | Can the intended behavior be expressed without hidden aliases or stale knobs? |
| Materialized config | Does the resolved config say what runtime actually does? |
| Data contract | Are labels, geometry, object order, image paths, and row filters valid? |
| Tokenizer/template | Are special tokens, chat stops, prompt text, image markers, and assistant spans identical across paths? |
| Dataset/collator | Are labels, masks, sidecars, padding, and offsets preserved into the batch? |
| Model forward | Are logits full-length, unsliced, correctly shifted, and computed with the intended inputs? |
| Loss math | Are support/balance/weights/type-gates/EOS priors applied to the intended terms? |
| Precision | Are log-softmax/logsumexp/probability math performed in fp32 when needed? |
| Metrics | Do metrics measure effective weighted contributions, not misleading raw terms? |
| Decode | Does generation use the same EOS/pad/template/preprocessing contract as training? |
| Eval/parser | Does evaluation parse exactly the intended output surface and report drops? |
| Artifacts | Can the run be reproduced from config, checkpoint, base model, tokenizer, and manifests? |

## Workflow

### Step 1: Extract The Innovation Contract

Write the intended algorithm in precise terms.

Capture:

- objective formula,
- target distribution,
- sampling/state distribution,
- masking rules,
- stop/EOS policy,
- data assumptions,
- tokenizer/template assumptions,
- decode/eval assumptions,
- production vs ablation status.

If the contract is ambiguous, ask the smallest number of blocking questions.

### Step 2: Build The Surface Map

List the files and runtime surfaces that can change the contract.

Always include:

- config schema and YAML,
- dataset and collator,
- template/tokenization,
- objective/target builder,
- loss function,
- trainer forward,
- inference/decode,
- evaluation/parser,
- artifacts/manifests,
- tests.

### Step 3: Create Invariants

Turn the contract into concrete invariants. Prefer invariants that can be tested.

Examples:

```text
teacher token is in positive set
target.position maps to labels after collation
EOS id is the chat stop id, not pad/text terminator
resolved config normalization equals target sidecar normalization
generation uses the same image preprocessing as training
loss formula matches scalar expectation on tiny logits
```

### Step 4: Run Read-Only Probes

Before proposing fixes, run narrow probes when allowed:

- real tokenizer ids,
- generation config,
- materialized config,
- one synthetic encoded sample,
- one collated batch,
- tiny logits loss calculation,
- JSONL/image scan,
- artifact manifest check.

Label each probe with scope.

### Step 5: Findings-First Report

Report findings before summaries.

For each finding include:

- severity,
- title,
- evidence with file/line or command output,
- impact on model training/eval,
- recommended patch,
- unit test or diagnostic.

Also list confirmed OK checks to avoid rediscovering the same questions.

### Step 6: Patch Plan Only If Asked

If the user asks for fixes, produce a minimal patch plan.

Prioritize:

1. fail-fast guards that prevent silent wrong training,
2. config/runtime truthfulness,
3. train/decode/eval parity,
4. deterministic unit tests,
5. smoke/diagnostic probes,
6. docs/artifact provenance.

Use TDD for behavior changes.

## Required Report Sections

Every completed audit should include:

```text
Findings
Confirmed OK
Decision Questions
Patch Recommendations
Unit Tests And Diagnostics
Smoke Run Suggestions
Residual Risks
```

If no findings are found, say so explicitly and identify remaining blind spots.

## Severity Rubric

| Severity | Meaning |
|---|---|
| P0 | Current implementation very likely trains/evals the wrong objective or corrupts core results now. Stop launch. |
| P1 | Silent mismatch can plausibly change training signal, decode behavior, or eval validity. Fix before serious run unless explicitly accepted. |
| P2 | Reproducibility, diagnostics, maintainability, or edge-case risk. Fix soon or document clearly. |
| P3 | Nice-to-have guardrail, cleanup, or future-proofing. |

Do not inflate severity just because a finding is interesting. Severity should track result risk.

## Subagent Pattern

When useful, dispatch independent subagents with disjoint audit surfaces.

Recommended subagents:

| Subagent | Scope |
|---|---|
| Objective/Loss Auditor | target math, support/balance, EOS/type-gate weighting, precision |
| Dataset/Collator Auditor | JSONL, geometry, object order, image root, masks, sidecar alignment |
| Tokenizer/Template/Decode Auditor | tokenizer ids, chat template, EOS/pad, generation config, parser |
| Config/Runtime Auditor | schema, materialized config, hidden overrides, effective batch, packing guards |
| Artifact/Eval Auditor | manifests, metrics, eval surface, checkpoint portability |
| Test Designer | unit tests, diagnostics, smoke commands, failure-injection cases |

Subagents should return findings-first reports with:

```text
[P0/P1/P2/P3] finding title
Evidence
Impact
Fix direction
Minimal test/diagnostic
Confirmed OK checks
Commands run
```

Do not duplicate the same task across subagents unless asking for independent adversarial confirmation.

## Launch Gate Recommendation

Before a serious training run, require:

- all P0 fixed,
- all P1 fixed or explicitly accepted by the user,
- P2 tracked or documented,
- targeted tests passing,
- at least one production-like smoke run,
- resolved config and artifacts prove the actual objective identity,
- train/decode/eval parity checked for tokenizer, image processing, EOS/pad, and parser.

## References

Load these only when needed:

- `references/risk-taxonomy.md`: detailed risk classes and probes.
- `references/subagent-prompts.md`: copy-ready parallel audit prompts.
- `references/report-template.md`: reusable audit report skeleton.
