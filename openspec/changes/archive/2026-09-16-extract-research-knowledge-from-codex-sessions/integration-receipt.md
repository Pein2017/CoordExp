# Integration receipt

## Products

- `coverage.tsv`: one contract row for every captured source path.
- `research-findings.md`: lead-deduplicated research synthesis.
- `notion-update-plan.md`: reviewed owner-level update packets; no Notion writes.

## Counts

- Session sources: 6,278 (`May=1,043`, `June=1,543`, `July=2,230`, `August=1,462`).
- Research-document sources: 9,746 (`root=51`, `worktree=9,695`; one live worktree path above the frozen pre-discovery count).
- Integrated rows: 16,024; unique paths: 16,024; malformed 14-column rows: 0; missing paths: 0.
- Relevance: `high=1,946`, `medium=7,354`, `skip=6,724`.
- Disposition: `covered=1,059`, `development-skip=3,925`, `duplicate=9,096`, `needs-adjudication=20`, `needs-summary=1,142`, `unexecuted=782`.

The manifest merge command was the deterministic concatenation of the four package manifests, dropping repeated headers. No source row or package disposition was rewritten during integration.

## Independent lead checks

- Replayed manifest headers, row counts, allowed source roots, path existence, and uniqueness.
- Recomputed a sample of research-document SHA-256 duplicate keys.
- Opened raw session samples across high, medium, and skip boundaries for every date package.
- Forced correction rounds after finding generic-development false positives and research-bearing false skips.
- Verified the corrected August identities for Image2299 and the split between `permanent-owner-bridge` V1 and `owner-commit-binding` successor work.
- Compared the prior serialization Notion receipt with live renderer/token code and retained the newline-order conflict as `needs-adjudication`.

## Residual limitations

- No callable Luna L2 route was available inside any L1 package. Four Luna L1 agents completed the census and synthesis directly; L0 independently sampled and reconciled them.
- Session classification remains heuristic. Latest-user-task classification can miss earlier research turns, while assistant-tail classification can over-promote generic development. Encrypted/inherited child tasks are the hardest boundary. Exact raw JSONL and current research owners remain authoritative.
- The coverage manifest proves exhaustive path enumeration, not exhaustive semantic interpretation of every turn.
- No Notion page was read or mutated in this wave; `notion-update-plan.md` is a proposal for the next reviewed write wave.

## Mutation boundary

Only files under `openspec/changes/extract-research-knowledge-from-codex-sessions/` were created or updated. No source code, config, experiment, GPU job, research document, progress record, session file, Notion page, commit, push, deletion, or move occurred.
