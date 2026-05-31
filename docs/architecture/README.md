---
doc_id: docs.architecture.index
layer: docs
doc_type: proposal-router
status: proposal
domain: architecture
summary: Router for the CoordExp architecture simplification proposal documents.
updated: 2026-05-31
---

# CoordExp Architecture Simplification Proposal

This directory contains a reusable architecture improvement proposal for converging the CoordExp codebase around fewer core concepts, fewer execution paths, and clearer ownership boundaries.

The documents are written for implementation agents working inside the repository. They are not a request to refactor immediately. They define the intended direction, migration concerns, and concrete planning units for future changes.

## Documents

1. [Codebase Simplification Proposal](CODEBASE_SIMPLIFICATION_PROPOSAL.md)
   - High-level diagnosis.
   - Target architecture.
   - Simplification principles.
   - Major recommendations and rationale.

2. [Subsystem Ownership Boundaries](OWNERSHIP_BOUNDARIES.md)
   - Proposed ownership boundaries for training, inference, data, evaluation, metrics, configuration, artifacts, and experiment code.
   - Active / compatibility / retired classification guidance.
   - Boundary rules for future changes.

3. [Refactoring Roadmap](SIMPLIFICATION_ROADMAP.md)
   - Phased migration plan.
   - Validation strategy.
   - Risk areas and migration concerns.
   - Suggested PR sequencing.

## How to use these documents

Start with `CODEBASE_SIMPLIFICATION_PROPOSAL.md` to understand the architectural diagnosis and target model. Use `OWNERSHIP_BOUNDARIES.md` when deciding where a change should live. Use `SIMPLIFICATION_ROADMAP.md` when turning the proposal into implementation tasks.

When these documents conflict with existing canonical behavior, treat the current `docs/`, `openspec/specs/`, and tests as the executable source of truth. This proposal describes a migration direction, not a completed refactor.
