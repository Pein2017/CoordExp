## Why

The previous Notion consolidation inventoried the local Codex session corpus and summarized major research routes, but it did not prove that every decision-bearing research session or unique finding was covered. A source-level reconciliation is needed to recover valuable research knowledge while excluding routine development/tooling sessions and avoiding raw transcript expansion.

## What Changes

- Classify the complete local Codex session corpus by research relevance using metadata, user intent, cwd/worktree, referenced research paths, and decision-bearing content.
- Deep-read the research-relevant subset in date-sharded Luna packages; skip sessions primarily about generic software/plugin/tool development unless they materially changed CoordExp research evidence, methodology, or authorization.
- Inventory and deduplicate current and historical research Markdown across the root and registered research worktrees, preserving source path, checkout, lifecycle, and divergence.
- Extract candidate findings as evidence-bounded records: source, question, executed surface, scientific disposition, technical disposition, decision impact, not-claimed boundary, and current owner.
- Produce a coverage manifest mapping each candidate session/document to `covered`, `duplicate`, `historical`, `unexecuted`, `needs-summary`, or `needs-adjudication`.
- Defer Notion mutation until the integrated manifest is reviewed; no raw session or document dump is imported.

Non-goals: auditing developer productivity or token cost; documenting generic Codex/plugin/tool development; modifying research conclusions, code, configs, worktrees, sessions, `research/`, or `progress/`; launching experiments; or claiming exhaustive semantic coverage from keyword matches alone.

Protected semantics: repository research owners and executed artifacts outrank session prose; technical mechanics and scientific evidence remain separate; forced, teacher-forced, retrieval, smoke, invalid, partial, and unexecuted evidence retain their exact boundaries; the newest authorized route supersedes stale handoffs without erasing provenance.

## Capabilities

### New Capabilities

None. This is research-knowledge intake and documentation work; `.openspec.yaml` sets `skip_specs: true`.

### Modified Capabilities

None.

## Impact

- Local writes are confined to `openspec/changes/extract-research-knowledge-from-codex-sessions/`.
- Read surfaces include `/data/CoordExp/.codex/sessions/`, `/data/CoordExp/.codex/memories/`, root `research/`, and research documents in registered CoordExp worktrees.
- Notion remains unchanged during the exploration wave; a later reviewed wave may update existing owners.
- No runtime, API, dependency, model-training, or production behavior changes.
