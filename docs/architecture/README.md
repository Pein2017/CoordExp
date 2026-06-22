---
doc_id: docs.architecture.index
layer: docs
doc_type: proposal-router
status: proposal
domain: architecture
summary: Router for the CoordExp architecture simplification proposal documents.
updated: 2026-06-15
---

# CoordExp Architecture Notes

This directory contains architecture reviews and proposal-only migration notes for converging the CoordExp codebase around fewer core concepts, fewer execution paths, and clearer ownership boundaries.

The documents are written for implementation agents working inside the repository. They are not a request to refactor immediately. They define the intended direction, migration concerns, and concrete planning units for future changes.

## Documents

1. [Independent Architecture Review](INDEPENDENT_ARCHITECTURE_REVIEW.md)
   - Review of risks, stale-target language, and implementation-boundary concerns.
   - Use this before treating any proposal document as actionable.

2. [Codebase Simplification Proposal](proposals/2026-05-31-simplification/CODEBASE_SIMPLIFICATION_PROPOSAL.md)
   - High-level diagnosis.
   - Target architecture.
   - Simplification principles.
   - Major recommendations and rationale.

3. [Subsystem Ownership Boundaries](proposals/2026-05-31-simplification/OWNERSHIP_BOUNDARIES.md)
   - Proposed ownership boundaries for training, inference, data, evaluation, metrics, configuration, artifacts, and experiment code.
   - Active / compatibility / retired classification guidance.
   - Boundary rules for future changes.

4. [Refactoring Roadmap](proposals/2026-05-31-simplification/SIMPLIFICATION_ROADMAP.md)
   - Phased migration plan.
   - Validation strategy.
   - Risk areas and migration concerns.
   - Suggested PR sequencing.

## How to use these documents

Start with `INDEPENDENT_ARCHITECTURE_REVIEW.md` to understand what is current-vs-target. Then use `proposals/2026-05-31-simplification/CODEBASE_SIMPLIFICATION_PROPOSAL.md` for the architectural diagnosis and target model, `proposals/2026-05-31-simplification/OWNERSHIP_BOUNDARIES.md` when deciding where a proposed change should live, and `proposals/2026-05-31-simplification/SIMPLIFICATION_ROADMAP.md` when turning the proposal into implementation tasks.

When these documents conflict with existing canonical behavior, treat the current `docs/`, `openspec/specs/`, and tests as the executable source of truth. This proposal describes a migration direction, not a completed refactor.
