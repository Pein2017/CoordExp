# Interface Design

When the user wants to explore alternative interfaces for a chosen deepening
candidate, use this pattern. Based on "Design It Twice" (Ousterhout): the first
idea is unlikely to be the best.

Uses the vocabulary in [LANGUAGE.md](LANGUAGE.md) — **module**, **interface**, **seam**, **adapter**, **leverage**.

## Process

### 1. Frame the problem space

Before spawning sub-agents, write a user-facing explanation of the problem space for the chosen candidate:

- The constraints any new interface would need to satisfy
- The dependencies it would rely on, and which category they fall into (see [DEEPENING.md](DEEPENING.md))
- A rough illustrative code sketch to ground the constraints — not a proposal, just a way to make the constraints concrete

Show this to the user, then immediately proceed to Step 2. The user reads and thinks while the sub-agents work in parallel.

### 2. Generate alternatives

Generate at least three meaningfully different interfaces. Use subagents only
when the user explicitly asks for parallel agent work; otherwise produce the
alternatives locally.

Use the same technical brief for each alternative: file paths, coupling
details, dependency category from [DEEPENING.md](DEEPENING.md), what sits behind
the seam, and any config/artifact/eval contracts. Apply a different design
constraint to each alternative:

- Minimal interface: aim for 1-3 entry points max and maximize leverage per entry point.
- Flexible interface: support extension without exposing implementation details.
- Common-case interface: make the dominant CoordExp workflow trivial and explicit.
- Contract-first interface: center config, artifact, metric, or geometry invariants when they are the true risk.

Use both [LANGUAGE.md](LANGUAGE.md) vocabulary and CoordExp domain names from
the relevant docs/specs/configs/artifacts.

Each alternative should include:

1. Interface (types, methods, params — plus invariants, ordering, error modes)
2. Usage example showing how callers use it
3. What the implementation hides behind the seam
4. Dependency strategy and adapters (see [DEEPENING.md](DEEPENING.md))
5. Contract impact and verification path
6. Trade-offs: where leverage is high, where it is thin

### 3. Present and compare

Present designs sequentially so the user can absorb each one, then compare them in prose. Contrast by **depth** (leverage at the interface), **locality** (where change concentrates), and **seam placement**.

After comparing, give your own recommendation: which design you think is strongest and why. If elements from different designs would combine well, propose a hybrid. Be opinionated — the user wants a strong read, not a menu.
