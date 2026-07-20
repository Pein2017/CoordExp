---
title: Reconciliation of the Independent Implementation Audits
description: Cross-review of the lead audit, Codex audit, and Opus audit after the lead's independent ranking was frozen.
type: investigation
role: audit-reconciliation
authority: non_normative_research
audit_status: reconciled_real_smoke_pending
updated: 2026-07-20
---

# Reconciliation of the Independent Implementation Audits

## Review Order

The lead first completed and recorded
`implementation-race-independent-audit.md`. Only afterward did the lead read
the two external reviews in full:

- `agent_review/codex-audit.md`;
- `agent_review/opus-audit.md`.

This ordering preserves the independent judgment while allowing direct
adversarial probes from the external reviews to correct it.

## External Review Quality Ranking

1. **Codex audit**
2. **Opus audit**

The Codex audit is stronger because it found two executable, conclusion-level
defects that neither the lead's first pass nor the Opus audit found:

- the command-line interface implementation admits `description` in the state
  bank but the loss vocabulary accepts `desc_text`, so a normal transition
  event fails at the gate; and
- the application implementation gates sites from disabled objectives in its
  single-objective arms, so those arms are not isolated.

It also used direct adversarial records to establish candidate-provenance and
coordinate-owner failures. Its recommendation to retain the command-line
interface implementation as the base while importing only value-based runtime
coordinate identity from the application implementation is well supported.

The Opus audit is useful for scope discipline, integration simplicity, and its
warning that the application implementation's forced per-step family
composition changes the bank's exposure distribution. However, it ranks
ClaudeX first while missing several semantic admission holes later confirmed
by direct probes: weak producer-role binding, arbitrary terminal identity,
coordinate value-to-token mismatch, ownerless geometry, and missing
prefix-covered-owner evidence. Its preference for fewer lines therefore
outweighs scientific identity more than this experiment permits.

## Corrected Static Ranking

1. **Command-line interface implementation**
2. **ClaudeX implementation**
3. **Application implementation**

The lead's initial placement of the application implementation in second
place is withdrawn. The correction is evidence-driven:

- its single-objective token-type gate is deterministically contaminated by
  disabled-objective sites;
- producer greedy versus sampled role is not schema-bound;
- coordinate owner can disagree with candidate owner; and
- its forced event-family scheduler changes per-event exposure and can repeat
  an event inside one optimizer step.

The application scheduler must therefore **not** be cherry-picked. If the
joint arm needs every objective family represented in a step, that property
must come from a frozen bank order or a sampler that preserves declared
per-event exposure without duplication or hidden reweighting.

## Consolidated Mandatory Fixes for the Command-Line Interface Base

Before real Smoke A, the selected base must satisfy all of the following:

1. Use one canonical token-type name end to end; the existing
   `description`/`desc_text` mismatch must be removed and covered by a real
   state-bank-to-loss test.
2. Store coordinate values, not independently trusted token identifiers, and
   resolve values through the active ordered coordinate-token identity at
   runtime.
3. Require the coordinate correction owner to match the candidate trajectory
   owner, except for an explicitly typed independently reviewed same-owner
   correction route.
4. Prove first-wrong ordering: every earlier coordinate must be inside its own
   reviewed accepted set and the selected coordinate must be the first outside
   its accepted set.
5. Require every nonterminal owner-resolution interval to begin at candidate
   offset zero, the shared next-row decision.
6. Keep producer-declared greedy versus sampled roles and the explicit set of
   physical owners represented in the prefix.
7. Load the source selected-token embedding delta for exact source identity
   but freeze it; optimizer ownership must contain language-tower
   Weight-Decomposed Low-Rank Adaptation parameters only.
8. Persist the trainable-surface receipt and post-backward finite-gradient
   report, including finite nonzero gradient norm.
9. Run one unmocked executable path from immutable state bank through exact
   image/token replay, packing, model forward, losses, backward, optimizer,
   checkpoint reload, and ordinary inference.

## What to Reuse from the Losing Implementations

Reuse only narrow evidence-backed ideas:

- from the application implementation, coordinate-value-to-runtime-token
  resolution and early manifest/checkpoint identity rejection;
- from ClaudeX, the tiny synthetic margin-direction-after-update regression
  test and its use of existing generic micro-step metadata.

Do not merge the application loss runner or scheduler. Do not replace the
command-line interface producer-provenance and prefix-coverage model with the
thinner ClaudeX schema.

## Remaining Decision Gate

This corrected ranking is still a static implementation judgment. A final
execution ranking requires a lead-owned neutral fixture compiled into each
native schema and a real shared Smoke A/B. Any implementation that cannot
consume the exact fixture without changing scientific meaning fails the
execution gate visibly; the fixture must not be weakened to make a branch
pass.
