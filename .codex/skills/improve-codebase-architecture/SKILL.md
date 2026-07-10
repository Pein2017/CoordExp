---
name: improve-codebase-architecture
description: Use when the user wants CoordExp architecture review, refactoring opportunities, deeper modules, cleaner research workflow seams, or code that is easier to test, audit, and navigate.
---

# Improve Codebase Architecture

Surface architectural friction and propose **deepening opportunities**:
refactors that turn shallow modules into deeper, easier-to-test modules without
changing research meaning or stable contracts by accident.

## Glossary

Use these terms consistently in suggestions. Full definitions live in
[LANGUAGE.md](LANGUAGE.md).

- **Module** — anything with an interface and an implementation (function, class, package, slice).
- **Interface** — everything a caller must know to use the module: types, invariants, error modes, ordering, config. Not just the type signature.
- **Implementation** — the code inside.
- **Depth** — leverage at the interface: a lot of behaviour behind a small interface. **Deep** = high leverage. **Shallow** = interface nearly as complex as the implementation.
- **Seam** — where an interface lives; a place behaviour can be altered without editing in place. (Use this, not "boundary.")
- **Adapter** — a concrete thing satisfying an interface at a seam.
- **Leverage** — what callers get from depth.
- **Locality** — what maintainers get from depth: change, bugs, knowledge concentrated in one place.

## Prompt And Comparison Modes

When the user asks for a prompt to send to multiple architecture reviewers, produce one shared read-only prompt. Set background, purpose, scope, and evidence expectations, but do not force a traversal order, hierarchy, or checklist unless requested. The prompt should let each reviewer reveal its own architecture taste.

When comparing review artifacts, give a direct verdict on the user's stated axis. Separate hierarchy/design taste, factual grounding, implementation safety, and actionability. Do not flatten "better" into a generic score.

Key principles:

- **Deletion test**: imagine deleting the module. If complexity vanishes, it was a pass-through. If complexity reappears across N callers, it was earning its keep.
- **The interface is the test surface.**
- **One adapter = hypothetical seam. Two adapters = real seam.**

CoordExp architecture is constrained by research semantics: image/geometry
alignment, config inheritance, training/eval parity, artifact contracts,
provenance, and evidence scope matter as much as code shape.

## Process

### 1. Explore

Start from the repo route before broad source search:
`docs/AGENT_INDEX.md`, `docs/catalog.yaml`, then relevant docs, stable specs,
configs, tests, artifacts, `research/` notes, and legacy `progress/`
provenance when explicitly relevant.

Use CodeGraph when a correct local index exists to map modules, call chains,
grouped source context, and impact radius before token-heavy file reads. In
linked worktrees, initialize/query the exact worktree and pass `projectPath` to
CodeGraph MCP calls when ambiguous. Treat CodeGraph as the scout: after it
identifies Python files or symbols, switch to Serena for exact symbol overview,
references, declarations/implementations, diagnostics, and symbolic edits. Use
`rg`/`rtk` for exact literal search in docs, configs, specs, artifacts, and
logs. Use subagents when parallel architecture audits materially help or the
user explicitly asks for parallel agent work.

Look for friction:

- Where does understanding one concept require bouncing between many small modules?
- Where are modules **shallow** — interface nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called (no **locality**)?
- Where do tightly-coupled modules leak across their seams?
- Which parts of the codebase are untested, or hard to test through their current interface?
- Where does code shape obscure config inheritance, loss semantics, geometry/order preservation, artifact contracts, or eval validity?

Apply the **deletion test** to anything you suspect is shallow: would deleting it concentrate complexity, or just move it? A "yes, concentrates" is the signal you want.

### 2. Present candidates

Present a numbered list of deepening opportunities. For each candidate:

- **Files** — which files/modules are involved
- **Problem** — why the current architecture is causing friction
- **Solution** — plain English description of what would change
- **Benefits** — locality, leverage, testability, auditability, and research-safety gain
- **Verification** — the narrow check that would prove behavior or contracts stayed intact

Use CoordExp vocabulary from the relevant docs/specs and the architecture
vocabulary from [LANGUAGE.md](LANGUAGE.md). Prefer names already present in
configs, artifacts, docs, and tests over new terminology.

If a candidate contradicts a stable spec, documented workflow, or active
experiment constraint, surface that conflict clearly and explain whether the
proposal needs OpenSpec, a docs update, or a smaller compatibility-preserving
shape.

Do NOT propose interfaces yet. Ask the user: "Which of these would you like to explore?"

### 3. Grilling loop

Once the user picks a candidate, drop into a grilling conversation. Walk the design tree with them — constraints, dependencies, the shape of the deepened module, what sits behind the seam, what tests survive.

Side effects happen inline as decisions crystallize:

- **Stable behavior or workflow changes?** Update the routed `docs/` page.
- **Compatibility-sensitive contract changes?** Use OpenSpec, only when the contract is genuinely stable and normative.
- **Empirical or historical reasons?** Record new interpretation in `research/`;
  read `progress/` only as legacy provenance.
- **Implementation checklists or handoff notes?** Keep them in the active super-power plan/spec when available.
- **User rejects the candidate with a load-bearing reason?** Record it in the right durable surface using `$grill-me record=local` guidance so future architecture reviews do not re-suggest it.
- **Want to explore alternative interfaces for the deepened module?** See [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md).
