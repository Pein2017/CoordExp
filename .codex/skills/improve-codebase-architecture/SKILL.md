---
name: improve-codebase-architecture
description: Use for read-only CoordExp architecture reviews, ranked deepening opportunities, or selected module/interface design while preserving research contracts and user-owned scientific decisions.
---

# Improve Codebase Architecture

Find architectural friction, rank deepening opportunities, and design an
approved interface without silently changing research meaning. Default to
read-only analysis; do not turn a review into implementation or a durable
decision record without explicit approval.

## Design language

Use these terms precisely:

- **Module**: a function, class, package, or cross-layer slice with an interface
  and implementation.
- **Interface**: everything callers must know: types, invariants, order, errors,
  configuration, performance, artifacts, and research semantics.
- **Depth**: substantial behavior behind a small, honest interface.
- **Seam**: where behavior can vary without editing the caller; adapters occupy
  a seam.
- **Leverage**: capability delivered per unit of caller knowledge.
- **Locality**: change, knowledge, bugs, and verification concentrated in one
  owner instead of scattered across callers.

Research contracts are interface facts: config/schema, data geometry and order,
token/template and forward semantics, targets and loss normalization, metric
scope, artifact names, cache identity, and provenance.

Apply three checks throughout:

- **Deletion test**: if deleting a module removes complexity, it was probably
  pass-through structure; if complexity spreads into callers, it bought
  locality.
- **Interface test**: callers and tests should cross the same semantic seam.
- **Variation test**: one adapter is hypothetical; add a seam when real
  variation or a correctness boundary justifies it.

Read [deepening.md](references/deepening.md) when consolidating an existing
cluster. Read [design-it-twice.md](references/design-it-twice.md) only after a
consequential interface or seam has been selected.

## Authority and tools

Start with `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, and the smallest relevant
canonical docs, stable specs, configs, tests, and artifacts. Treat historical
notes and old worktrees as evidence, not current authority.

Use CodeGraph only when its index matches the exact worktree. Use Serena for
precise Python symbol bodies and references when available. Use `rg` and direct
reads for docs, YAML, specs, artifacts, metrics, and manifests.

## Review workflow

1. **Map ownership and callers.** Trace one concept across entrypoint, config,
   implementation, tests, artifacts, and docs.
2. **Find friction.** Look for scattered caller knowledge, pass-through layers,
   tests coupled to internals, duplicated policy, large orchestration owners,
   compatibility leaking into canonical behavior, or hidden research meaning.
3. **Rank candidates.** For each, report files/current owner, concrete friction,
   deepening direction, knowledge hidden versus kept visible, research risk,
   benefit, verification, and strength: `strong`, `worth exploring`, or
   `speculative`.
4. **Recommend one candidate.** Give the direct reason and stop for selection;
   do not prematurely lock its interface.

Review output defaults to concise chat. Use a compact Mermaid diagram only when
three or more interacting modules make the relationship clearer. When comparing
existing reviews, judge factual grounding, design depth, research safety, and
actionability against the user's requested axis.

## Selected-interface workflow

After the user selects a candidate:

1. Name the concept and single owner.
2. Inventory caller knowledge: inputs, outputs, order, configuration, failure
   modes, artifacts, performance, and research semantics.
3. Classify dependencies and seams using
   [deepening.md](references/deepening.md).
4. Propose the full interface before internals, including invariants, errors,
   ordering, configuration, and observable receipts.
5. Keep incidental complexity hidden while exposing user-owned scientific
   choices.
6. Test through the interface with behavior, contract, artifact, replay, or
   smoke evidence that survives internal refactoring.
7. For a consequential seam, compare genuinely different designs with
   [design-it-twice.md](references/design-it-twice.md).

The user owns choices that alter algorithm or forward semantics, data
construction/geometry/order, targets, loss or normalization, optimization or
training cost, statistical assumptions, metric comparability, artifact meaning,
or supported research claims. Use `grill-me` for one such decision at a time.
Choose reversible code structure without making the user decide code aesthetics.

## Approval boundary

End with one state:

- `drop candidate`;
- `probe architecture assumption`;
- `ready for interface decision`;
- `ready for implementation approval`;
- `needs user decision`.

Do not implement until the candidate and intended interface are approved. After
approval, use an ordinary plan for reversible refactoring, OpenSpec only for a
stable compatibility-sensitive contract, or `research/` for empirical rationale.
Update canonical docs only when current behavior or recommended workflows change.
