---
name: codebase-design
description: Review and rank CoordExp architecture friction read-only, or design and compare module interfaces and seam placement when behavior needs deeper ownership, testability, or research-semantic clarity.
---

# Codebase Design

Design a **deep module**: substantial behavior behind a small, honest interface
at the owner of the concept. Keep user-owned scientific choices visible.

## Choose The Mode

- **Review:** follow current authority to live contracts, config, tests,
  artifacts, callers, and source. Find concrete caller knowledge spread across
  modules, multiple policy owners, shallow orchestration, compatibility leakage,
  or tests crossing internals. Rank candidates before proposing an interface.
- **Design:** design only the selected concept and owner. Compare meaningfully
  different interfaces when the seam is consequential.

Architecture review is read-only. Do not implement, record a durable proposal,
or change research meaning without separate authorization.

For each review candidate report the current owner, exact friction evidence,
deepening direction, knowledge hidden and retained, semantic risk, benefit,
verification, and confidence. Give one top recommendation and stop at `drop`,
`probe assumption`, `ready for interface decision`, `ready for implementation
approval`, or `needs user decision`.

## Decide Whether A Seam Exists

Use a stable interface only when variation is real, a second consumer exists,
a compatibility boundary already exists, or moving the seam later is
materially costly. Keep a first exploratory consumer experiment-local.

A useful interface concentrates caller knowledge, failure handling, and
verification. A shallow wrapper merely relocates complexity.

Read [DEEPENING.md](DEEPENING.md) when restructuring an existing cluster and
[DESIGN-IT-TWICE.md](DESIGN-IT-TWICE.md) for a consequential interface choice.

## Design

1. **Name the concept and owner.**
   - State the promise in one sentence and identify current callers.
   - Complete when one module can truthfully own the behavior.

2. **Inventory caller knowledge.**
   - Include inputs, outputs, invariants, order, config, failure modes,
     performance, artifacts, and research semantics.
   - Complete when hidden coupling is visible.

3. **Place the seam.**
   - Keep stable internals internal; expose only real variation and observable
     receipts. Accept dependencies explicitly when ownership varies.
   - Complete when deleting the proposed module would spread its complexity back
     across callers.

4. **Protect scientific control.**
   - Keep choices about model behavior, data and geometry, targets and
     objectives, optimization, statistics, metrics, and artifact meaning visible
     to the user.
   - Choose reversible code structure without escalating incidental details.
   - Complete when the interface hides implementation burden without hiding
     meaning.

5. **Verify through the interface.**
   - Make callers and tests cross the same semantic seam. Compare a second design
     when the choice is costly.
   - Complete when behavior and contract checks survive internal refactoring.

## Report

For review, return ranked candidates with exact evidence, one recommendation,
verification, residual risk, and approval state. For design, state current owner
and friction, proposed promise, hidden versus visible knowledge, alternatives
when consequential, migration cost, research-semantic risk, and verification.
