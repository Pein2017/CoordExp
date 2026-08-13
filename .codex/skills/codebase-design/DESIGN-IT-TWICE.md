# Design It Twice

Use this when a chosen seam or interface is consequential enough that the first
reasonable design should not become the default by momentum.

## Frame The Problem

State:

- the concept and current owner;
- constraints every design must satisfy;
- dependency categories from [DEEPENING.md](DEEPENING.md);
- research contracts that must remain visible or unchanged;
- a small illustrative caller example that grounds the problem without
  prejudging the answer.

## Generate Alternatives

Generate at least three meaningfully different interfaces. Use subagents only
when the user explicitly asks for parallel agent work; otherwise generate them
locally with independent design constraints:

- **Minimal interface**: one to three entry points, maximum leverage.
- **Flexible interface**: supports verified variation without exposing
  internals.
- **Common-case interface**: makes the dominant workflow explicit and trivial.
- **Contract-first interface**: centers the data, geometry, forward, loss,
  metric, or artifact invariant when it is the dominant risk.

For each alternative provide:

1. the full interface, including invariants, order, errors, and config;
2. a caller example;
3. what becomes hidden;
4. dependency and adapter strategy;
5. user-owned semantic decisions;
6. contract and verification impact;
7. where depth and locality are gained or lost.

## Compare And Decide

Present alternatives separately before comparing them. Contrast depth,
locality, seam placement, contract visibility, testability, migration cost, and
the burden placed on the user.

Give a recommendation. If the choice alters algorithm, forward semantics, data,
loss, statistics, evaluation meaning, or experiment cost, ask one direct
decision question. Use `grill-me` only when the user explicitly invokes it.
Otherwise select the soundest reversible implementation without making the user
choose code aesthetics.
