# Design It Twice

Use this when a selected seam or interface is consequential enough that the
first reasonable design should not win by momentum.

## Frame the problem

State:

- the concept and current owner;
- constraints every design must satisfy;
- dependency categories from [deepening.md](deepening.md);
- research contracts that must remain visible or unchanged;
- one small caller example that grounds the problem without prejudging it.

## Generate alternatives

Generate at least three meaningfully different interfaces. Use subagents only
when the user explicitly asks for parallel agent work; otherwise generate them
locally with independent constraints. Useful shapes are:

- **Minimal**: one to three entry points with maximum leverage.
- **Flexible**: supports verified variation without exposing internals.
- **Common-case**: makes the dominant workflow explicit and trivial.
- **Contract-first**: centers the data, geometry, forward, loss, metric, or
  artifact invariant when it is the dominant risk.

For each alternative provide:

1. the full interface, including invariants, order, errors, and configuration;
2. a caller example;
3. what becomes hidden;
4. dependency and adapter strategy;
5. user-owned semantic decisions;
6. contract and verification impact;
7. where depth and locality are gained or lost.

## Compare and decide

Present alternatives separately before comparing depth, locality, seam
placement, contract visibility, testability, migration cost, and user burden.
Give a recommendation. If the choice alters algorithm, forward semantics, data,
loss, statistics, evaluation meaning, or experiment cost, use `grill-me` to ask
one decision question. Otherwise choose the soundest reversible implementation.
