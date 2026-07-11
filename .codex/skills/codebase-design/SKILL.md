---
name: codebase-design
description: Use when designing or improving module interfaces, deciding seam placement, comparing architectural alternatives, deepening shallow code, making behavior testable, or when another CoordExp skill needs shared deep-module and research-contract vocabulary.
---

# Codebase Design

Design deep modules: substantial behavior behind a small, honest interface,
placed at a clean seam and testable through that interface. Optimize for
leverage for callers, locality for maintainers, and legibility of research
meaning for the user.

## Shared Language

Use these terms consistently while preserving domain names from CoordExp docs,
configs, artifacts, and tests.

- **Module**: anything with an interface and an implementation: function,
  class, package, or cross-layer slice.
- **Interface**: everything a caller must know to use the module correctly,
  including types, invariants, ordering, errors, configuration, performance,
  and research semantics. It is wider than a type signature.
- **Implementation**: behavior hidden inside the module.
- **Depth**: leverage at the interface. A deep module provides substantial
  behavior through a small interface; a shallow module exposes nearly as much
  complexity as it contains.
- **Seam**: a place where behavior can vary without editing the caller. It is
  where an interface lives.
- **Adapter**: a concrete implementation occupying a seam.
- **Leverage**: capability delivered per unit of interface the caller learns.
- **Locality**: change, knowledge, bugs, and verification concentrated in one
  owner instead of scattered across callers.

Prefer these terms to vague substitutes such as `component`, `service`, `API`,
or `boundary` when discussing this architecture model.

## Core Principles

- **Depth belongs to the interface, not the line count.** Internal complexity
  can be large and well-factored while the external interface remains small.
- **Apply the deletion test.** If deleting a module makes complexity disappear,
  it was probably pass-through structure. If the complexity reappears across
  callers, the module was hiding something useful.
- **The interface is the test surface.** Callers and tests should cross the same
  semantic seam. A need to test past it often reveals the wrong module shape.
- **One adapter is a hypothetical seam; two adapters make variation real.** Do
  not add indirection for an imagined future.
- **Accept dependencies instead of creating them invisibly.** Keep variable
  infrastructure behind an explicit internal seam.
- **Return observable results instead of forcing callers to inspect side
  effects.** When side effects are inherent, expose their receipt or artifact.
- **Research contracts are interface facts.** Config keys, geometry/order,
  token roles, forward semantics, targets, loss normalization, metric scope,
  artifact names, cache identity, and provenance are caller knowledge.

Read [DEEPENING.md](DEEPENING.md) when restructuring an existing cluster. Read
[DESIGN-IT-TWICE.md](DESIGN-IT-TWICE.md) when comparing alternative interfaces.

## Human And Agent Control

The goal is not to make the user learn implementation detail. The agent should
map dependencies, identify ownership, generate alternatives, explain trade-offs,
and implement the selected design. The user should retain control over choices
that alter:

- algorithmic behavior or model forward semantics;
- data construction, sampling, geometry, ordering, or leakage assumptions;
- targets, loss terms, normalization, optimization, or training budget;
- statistical assumptions, estimands, thresholds, or evidence requirements;
- metric comparability, artifact meaning, and supported research claims.

Expose those choices in architectural language: what the module promises, what
it hides, which invariant moves, which evidence would decide, and what becomes
hard to change. Use `grill-me` to resolve one such decision at a time. Choose
ordinary code structure and reversible implementation details without making
the user act as a programmer.

## Design Procedure

1. **Name the concept and owner.** State what behavior or contract the module
   owns in one sentence.
2. **Inventory caller knowledge.** List every fact callers currently need:
   inputs, outputs, order, config, failure modes, artifacts, performance, and
   research semantics.
3. **Classify dependencies.** Use [DEEPENING.md](DEEPENING.md) to decide which
   dependencies remain internal and which justify a seam.
4. **Propose the interface before internals.** Include types, invariants,
   errors, ordering, configuration, and observable receipts.
5. **Run the control check.** Verify that incidental complexity is hidden while
   user-owned scientific decisions remain visible.
6. **Test through the interface.** Select behavior, contract, artifact, or smoke
   evidence that survives internal refactoring.
7. **Design it twice when the decision is consequential.** Compare meaningfully
   different interfaces before locking a high-cost seam.

## Relationships

- A module presents one coherent interface to its callers and tests.
- Depth is judged against that interface.
- A seam locates variation; adapters occupy the seam.
- Depth creates leverage for callers and locality for maintainers.
- Research contracts constrain the interface even when no Python type expresses
  them.

## Rejected Framings

- **Depth as implementation-lines divided by interface-lines**: padding is not
  leverage.
- **Interface as only public methods or a language keyword**: invariants and
  semantic contracts also count.
- **Every dependency deserves an interface**: unvarying internals do not need
  speculative ports.
- **Pure functions everywhere**: extraction without locality can leave the real
  bugs in orchestration.
- **Cleaner code over research meaning**: a refactor that silently changes data,
  forward behavior, loss, or evaluation is an invalid design.

## Philosophy

A good module does not hide everything. It hides incidental implementation
complexity while exposing the choices that determine meaning. This distinction
lets an agent carry the coding burden without taking scientific control from
the user.

Deep design is therefore a coordination tool. The user reasons about promises,
evidence, and trade-offs at a high level; the agent reasons about files, types,
dependencies, and tests. The interface is where those two understandings meet.
