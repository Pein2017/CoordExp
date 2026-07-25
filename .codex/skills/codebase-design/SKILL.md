---
name: codebase-design
description: Design or compare CoordExp module interfaces and seam placement when behavior needs deeper ownership, testability, or research-semantic clarity.
---

# Codebase Design

Design a **deep module**: substantial behavior behind a small, honest interface
at the owner of the concept. Keep user-owned scientific choices visible.

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

State current owner and friction, proposed promise, hidden versus visible
knowledge, alternatives when consequential, migration cost, research-semantic
risk, and verification.
