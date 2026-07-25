---
name: model-innovation-risk-audit
description: Gate a high-risk or decision-grade CoordExp mechanism before launch or interpretation when silent config, data, runtime, objective, or evaluation drift could change the claim.
---

# Model Innovation Risk Audit

Run a read-only **contract gate** for mismatches that can pass a smoke and still
train or evaluate a different mechanism. Use `model-diagnosis` when an existing
behavioral symptom is the main question.

Do not invoke the full gate merely because a pilot is new. For an exploratory
first observation, protect checkpoint/config/input identity, the declared
factor, semantic alignment, actual path execution, objective wiring when
training, and raw output attribution. Escalate when silent drift can reverse the
observation or the evidence is being promoted.

## Gate

1. **Write the minimal contract diff.**
   - Compare intended behavior, authored config, resolved config/schema,
     executable data/model/loss/decode/eval objects, emitted artifacts, and
     evidence claim.
   - Complete when every layer is `matched`, `mismatched`, or `unproven`.

2. **Identify executable owners.**
   - Trace who actually controls tokenizer/template, data and collator,
     trainable groups, forward and loss, optimizer/scheduler/distributed
     behavior, decode/parser/eval, and artifact side effects.
   - Inspect installed source or run a tiny executable probe when upstream owns
     the semantics.
   - Complete when mocks or plans are no longer standing in for the owner.

3. **Probe the fragile seam.**
   - Prefer one real encoded sample, collated batch, deterministic formula or
     logits probe, resolved config, JSONL/image scan, manifest check, or targeted
     smoke.
   - For objectives, verify support, masks, denominators, raw and weighted
     terms, finite precision, gradient path, and logged names only where the
     claim depends on them.
   - Complete when the suspected silent mismatch is proved, falsified, or
     localized.

4. **Return a launch decision.**
   - Report `promote`, `hold`, `rerun gate`, or `needs user decision` with the
     smallest evidence that would change it.
   - Do not inflate severity or harden exploratory infrastructure without a
     demonstrated conclusion-changing failure.
   - Complete when claim scope and runtime evidence support the verdict.

## Report

Lead with P0/P1/P2 findings, then the contract diff, confirmed OK checks,
decision questions, correction or probe direction, verification, and residual
risk. Each finding needs an evidence handle and an explicit decision impact.

Load only when needed:

- [risk-taxonomy.md](references/risk-taxonomy.md) for failure classes;
- [contract-diff.md](references/contract-diff.md) for a standalone contract
  table;
- [report-template.md](references/report-template.md) for a durable report.
