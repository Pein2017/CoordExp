---
name: improve-codebase-architecture
description: Use when the user wants a read-only CoordExp architecture review, ranked deepening/refactoring opportunities, visual architecture report, or a user-guided path from codebase friction to an approved module/interface design.
---

# Improve Codebase Architecture

Find architectural friction and propose deepening opportunities without
silently changing research meaning. Use [codebase-design](../codebase-design/SKILL.md)
for the shared module/interface philosophy; this skill owns CoordExp-specific
discovery, evidence, ranking, and user coordination.

## Default Posture

Architecture review is read-only and analysis-first unless the user explicitly
asks to record or implement the result. Do not turn a review request into a
refactor, docs rewrite, OpenSpec change, or durable decision record.

The agent should absorb code-level exploration and explain the architecture at
the level the user needs to control. The user owns choices that alter algorithm
semantics, model forward behavior, data construction/geometry/order, loss and
normalization, optimization/training trade-offs, statistical assumptions,
metric comparability, or artifact meaning. Reversible code organization is the
agent's responsibility.

## Review Modes

- `report=chat`: default. Return a ranked candidate list with exact evidence.
- `report=visual`: use when the user asks for a visual report or when three or
  more interacting modules make the relationship materially clearer. Prefer a
  compact Mermaid diagram in chat; use a self-contained `/tmp/` HTML report
  only when the requested comparison needs richer before/after visuals.
- `compare=reviews`: compare existing architecture reports on factual grounding,
  hierarchy/design taste, research safety, and actionability; give a direct
  verdict on the user's stated axis.
- `prompt=reviewers`: write one shared read-only prompt for multiple reviewers.
  Set scope and evidence expectations without forcing them into one design
  taste.

## Process

### 1. Load authority and design language

Read `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, and the relevant canonical
docs/specs/configs/tests/artifacts before broad source search. Read
[codebase-design](../codebase-design/SKILL.md) for the shared vocabulary. Read
[DEEPENING.md](../codebase-design/DEEPENING.md) when consolidating a candidate,
and [DESIGN-IT-TWICE.md](../codebase-design/DESIGN-IT-TWICE.md) only after a
candidate has been selected for interface exploration.

Use CodeGraph only as a broad map when the exact worktree has a correct index.
Once Python files or symbols are known, use Serena for precise bodies,
references, declarations, and diagnostics. Use `rg`/raw reads for docs, YAML,
specs, artifacts, metrics, and manifests.

### 2. Explore friction organically

Look for places where:

- understanding one concept requires bouncing across many shallow modules;
- callers need nearly as much knowledge as the implementation contains;
- pure helpers were extracted for unit testing but orchestration owns the real
  failure;
- config, forward, data, loss, metric, or artifact policy has multiple owners;
- compatibility or diagnostic paths leak into canonical behavior;
- a large entrypoint accumulates decisions that belong to a source owner;
- tests cross internal structure because no honest interface exists;
- research meaning is hidden behind generic framework abstractions;
- user-facing knobs encode choices the implementation should simply get right.

Apply the deletion test. If deleting a module makes complexity disappear, it
was likely pass-through structure. If the complexity spreads across callers,
the module was buying locality.

### 3. Present ranked candidates

For each candidate include:

- **Files and current owner**;
- **Friction**, with concrete call/config/artifact evidence;
- **Deepening direction**, without prematurely fixing the interface;
- **Hidden versus visible knowledge** after the change;
- **Research-semantic risk**: forward, data, loss, statistics, metrics, or
  artifacts touched;
- **Benefits** in depth, locality, testability, auditability, and navigation;
- **Verification** that would prove behavior and contracts stayed intact;
- **Strength**: `strong`, `worth exploring`, or `speculative`.

End with one top recommendation and why. Do not propose interfaces yet. Ask the
user which candidate to explore, one question only.

### 4. Design the chosen interface

For the selected candidate, follow
[DESIGN-IT-TWICE.md](../codebase-design/DESIGN-IT-TWICE.md) when the seam is
consequential. Generate meaningfully different alternatives before choosing by
momentum. Compare depth, locality, contract visibility, testability, migration
cost, and the burden placed on the user.

Use `grill-me` for user-owned semantic forks. Ask one decision at a time, attach
the recommended answer, and wait. Do not ask the user to choose between class or
function layouts unless those layouts encode a real architectural or research
trade-off.

### 5. Stop at an approval boundary

Finish the review with exactly one state:

- `drop candidate`;
- `probe architecture assumption`;
- `ready for interface decision`;
- `ready for implementation approval`;
- `needs user decision`.

Do not implement until the user explicitly approves the candidate and intended
interface. After approval, use the smallest appropriate carrier: ordinary plan
for reversible refactoring, OpenSpec for stable compatibility-sensitive
contracts, or `research/` for empirical rationale. Update canonical docs only
when behavior or recommended workflows actually change.

## Philosophy

Architecture is not professional-coder theater. Its job is to compress the
implementation burden while making semantic choices easier to see and control.
A beautiful hierarchy that obscures model behavior, data meaning, loss, or
statistical assumptions is worse than plain code.

The agent should bring codebase literacy, alternatives, and evidence. The user
should be able to reason about promises, trade-offs, and experimental meaning
without mastering every implementation detail. The chosen interface is the
coordination surface between those responsibilities.
