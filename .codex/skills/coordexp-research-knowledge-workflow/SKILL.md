---
name: coordexp-research-knowledge-workflow
description: Preserve, migrate, synthesize, or close CoordExp research knowledge while keeping raw provenance, active interpretation, and formal authority distinct.
---

# CoordExp Research Knowledge Workflow

Use this skill when the durable product is **research knowledge** rather than
current-behavior documentation, production code, or an ordinary audit.

## Authority

- Current operator guidance owns current behavior.
- Stable specifications own compatibility-sensitive contracts.
- The active research record owns ideas, investigations, mechanisms,
  decisions, negative results, and continuation context.
- Historical records are provenance; legacy progress archives receive no new
  records.
- Reviews, handoffs, packets, and memory route to owners but do not become
  authority.

Use `audit-review` for correctness findings, `model-diagnosis` for abnormal
behavior, `coordexp-infer-eval-workflow` for run operations, and
`git-hygiene` for repository publication.

## Work

1. **Bound the source and product.**
   - Record exact roots, worktrees, date window, globs, artifacts, and whether
     the result is raw intake, migration, synthesis, a research unit, or a
     decision packet.
   - Complete when scope cannot silently expand.

2. **Preserve provenance before interpretation.**
   - Keep source path, checkout, content identity, and divergence visible.
   - For cross-worktree unions, distinguish identical content, new paths, and
     same-path divergent content before cleanup.
   - Complete when every retained claim can be traced to a source.

3. **Synthesize into an owning surface.**
   - Build a reading path rather than mirroring raw notes.
   - Separate planned protocol, executed evidence, interpretation, decision,
     mechanism promotion, and implementation authorization.
   - For research units, apply
     [Research Graph and Unit Contract](references/research-graph-contract.md).
   - Complete when one tracked unit/result/decision owns the current statement.

4. **Resolve semantic deltas.**
   - Compare the originating question and approved route with any later change
     to cohort, predicate, estimand, control, claim, stop rule, critical path,
     or material cost.
   - Record old versus new, reason, claim/evidence/cost impact, authorization,
     and detached work.
   - Record decision-owning outcome, intervention or proxy, final evaluation,
     transfer assumption, preservation risk, and signal supply.
   - Use [Research Alignment Examples](references/research-alignment-examples.md)
     only when the boundary is unclear.
   - Complete when derived constraints are either authorized or explicitly
     labeled proposals.

5. **Close routing once.**
   - When evidence changes the route, update the owning result/unit, experiment
     index, and current decision or compass. Refresh project memory only if
     continuation state changed; create a handoff only for a real transfer.
   - Promote current behavior to operator guidance or stable contracts only
     through their own authorized change.
   - Complete when no live router points at superseded transport or provenance.

6. **Verify the product.**
   - Check tracked scope, links, source counts, lifecycle state, artifact
     attribution, and that the verdict does not exceed its evidence.
   - Complete when the reading path is coherent and the authority boundary is
     explicit.

## Conditional Branches

- For raw Markdown union collection, preserve a manifest and byte-faithful
  snapshots before synthesis.
- For supervisor or independent-model packets, make the packet
  decision-focused and keep reviewer output advisory.
- For Open Knowledge Format style migration, use the repository's native
  research routers and light frontmatter; avoid app-specific wiki structure.
- For executed units, keep durable artifacts under the owning output root and
  record immutable run identity once comparison requires it.

## Report

State source boundary, created or updated owners, raw versus synthesized
material, authority caveats, verification, unresolved gaps, and next
continuation point.
