## Why

The shared CoordExp Notion workspace mixes current research, generated synthesis, stale guidance, and legacy records, while the local repository, worktrees, sessions, `research/`, and legacy `progress/` contain the evidence needed to distinguish them. Consolidating that evidence now will make Notion a reliable shared reading and navigation layer without turning it into a second executable source of truth or an unbounded transcript archive.

## What Changes

- Refactor the existing CoordExp home, Global Research OS, track pages, and four existing databases in place; preserve stable page URLs and use reversible lifecycle/status changes rather than deletion.
- Replace stale or over-strong research wording with evidence-bounded summaries, especially for Image2299, serialization, compact sequence, causal mechanisms, and current production behavior.
- Record the current serialization contract precisely: fully closed object wrappers are concatenated with no per-row separator; the assistant payload receives one terminal `<|im_end|>\n` suffix. Keep production `geo_sorted` as the current y-then-x validation route while documenting x-then-y as the preferred future experimental direction and a possible future production migration, not an approved switch.
- Import only bounded summaries and canonical pointers from Codex sessions, registered worktrees, `research/`, and legacy `progress/`; do not copy raw transcripts, large artifacts, or duplicate histories into Notion.
- Mark abandoned or superseded routes as historical, stale, rejected, closed, or on hold according to evidence, and interlink active questions, decisions, claims, methods, and provenance.
- Remove generated lifecycle/strength hierarchies that overstate authority. Use plain evidence-surface, status, scope, and source wording instead.
- Verify each write by refetching the affected page and maintain a concise reconciliation receipt in this change.

Non-goals are changing local research records, worktrees, model code, configs, production ordering, training or inference behavior; importing raw session JSONL or bulky `progress/` artifacts; adding databases, relations, sync automation, or dependencies; deleting Notion pages; or committing/pushing repository changes.

Protected semantics: repository code/configs/specs and current research artifacts remain authoritative for executable behavior and scientific evidence; technical validity and scientific conclusions stay separate; unexecuted, invalid, confounded, or diagnostic-only work remains neutral; historical URLs and provenance remain intact.

## Capabilities

### New Capabilities

None. This is a documentation and knowledge-governance refactor; `.openspec.yaml` sets `skip_specs: true`.

### Modified Capabilities

None. No stable compatibility-sensitive behavior changes.

## Impact

- Local: only `openspec/changes/consolidate-coordexp-notion-research-os/` is added or edited.
- Notion: existing CoordExp home, Research OS, research/track pages, database records, views, and legacy routers may be updated in place; new pages are allowed only when a bounded summary has no suitable existing owner.
- Runtime/API/dependencies: none.
- Authority: Notion remains a shared reading and synthesis layer; `/data/CoordExp` current code, configs, specs, artifacts, and active research remain the source of truth for implementation and claims.
