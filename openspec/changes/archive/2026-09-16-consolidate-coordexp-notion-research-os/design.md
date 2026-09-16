## Context

See [proposal.md](proposal.md). The current Notion workspace already has a useful home, a Global Research OS, four databases, track pages, and preserved legacy URLs. Its main defects are semantic drift, generated hierarchies presented as authority, sparse cross-links, and a few stale current-state pages. Local evidence is much larger than Notion: 6,265 session JSONL files (~17.47 GB), 34 registered worktrees, 51 active `research/` Markdown files, and 231 legacy `progress/` records. Notion must therefore index and summarize rather than mirror.

## Goals / Non-Goals

**Goals:**

- Give each current topic one concise Notion owner and link evidence, decisions, methods, and historical records to it.
- Keep every scientific claim bounded by execution status, evidence surface, scope, and canonical source.
- Make stale/abandoned material visibly historical without breaking URLs or provenance.
- Leave a verifiable receipt for every page mutation.

**Non-Goals:**

- No bidirectional sync, transcript ingestion pipeline, new database, relation migration, schema deletion, or custom automation.
- No change to repository content outside this OpenSpec change, and no scientific or production decision beyond the user's explicit rulings.

## Decisions

### 1. Reuse the existing information architecture

The home remains the entry point; Global Research OS remains a shared reading map; Research Units, Methods, Claims, and Decision Log remain the only databases; track and legacy pages remain routers. We will improve content and views in place before considering schema changes.

Alternative considered: build a new canonical database set. Rejected because it would duplicate records, break established URLs, and create a migration problem without improving scientific authority.

### 2. Treat Notion as a projection, not an authority replica

Each summary states its evidence status and links canonical repository paths. Sessions are sampled through metadata, memory/index summaries, and decision-bearing outer messages; worktrees are deduplicated by current semantic owner; `research/` is active interpretation; `progress/` is historical provenance. Raw JSONL, logs, images, and bulky artifacts stay local.

Alternative considered: import all records or attach files. Rejected due to size, duplication, privacy/noise, and inevitable drift.

### 3. Use evidence-bearing language instead of a synthetic ladder

Records use plain fields/sections: lifecycle status, evidence surface, scope, source, current interpretation, and not-claimed boundary. The generated V→S→A→R→T→E sequence may remain only as a diagnostic checklist, never as a mandatory lifecycle or accepted scientific ontology. The E0–E5 ordinal ladder is removed from current guidance because its dimensions are not ordinal.

Alternative considered: preserve the hierarchy and merely add caveats. Rejected because an accepted-looking ladder would continue to imply authority and comparability it does not have.

### 4. Apply a fetch–mutate–refetch transaction per page

Before each Notion write, fetch the current page, mutate by stable page ID, then refetch and verify distinctive text/properties. Writes are small, grouped into waves, and stop on any ambiguous replacement or tool error. The receipt records page ID, intended outcome, verification marker, and result.

Alternative considered: bulk rewrite all pages. Rejected because partial failures and concurrent GPT-Web edits would be hard to detect or roll back.

### 5. Preserve exact serialization and ordering boundaries

The serialization page will distinguish four facts:

1. Each object is fully closed as `<|object_ref_start|>{desc}<|object_ref_end><|box_start|>{x1}{y1}{x2}{y2}<|box_end|>`.
2. Object rows are concatenated with no separator or newline between rows.
3. The complete assistant content receives one terminal `<|im_end|>\n` suffix; the newline follows `<|im_end|>` and is not a row separator.
4. Current production `geo_sorted` validates y-then-x authored order. Directional bounded evidence favors an explicit future x-then-y experimental route; a production migration is possible later but is not part of this change.

### 6. Archive reversibly and by evidence, never by age alone

Abandoned routes become Closed, Rejected, Superseded, Historical, Stale, or Hold in their owning record. Legacy pages become thin routers with a historical callout. No page is trashed, and no URL is moved merely because it is old.

## Risks / Trade-offs

- [Concurrent Notion edits could invalidate text replacement] → Fetch immediately before writing; use page IDs and refetch verification; stop rather than overwrite ambiguous content.
- [Summaries could inflate weak evidence] → Include evidence surface, scope, and explicit “not claimed” boundaries; preserve unexecuted/invalid status.
- [A small projection omits useful history] → Keep canonical local paths and legacy URLs so detail remains discoverable without copying it.
- [Sparse database properties remain] → Prefer better views and page bodies now; defer destructive schema cleanup until usage proves columns unnecessary.
- [Notion query limits constrain a full database audit] → Operate on the already fetched inventory and exact page IDs; verify writes individually rather than issuing more broad queries.

## Migration Plan

1. Freeze local and Notion inventory in the receipt.
2. Pilot four high-value pages: home, Global Research OS, serialization, and Image2299. Refetch all four; stop if any verification marker is absent.
3. Reconcile compact-sequence records, blank methods, and legacy router pages using reversible lifecycle wording.
4. Add bounded cross-worktree/session/research/progress synthesis to existing owners; create at most one local-index page only if no current owner can hold the pointers.
5. Verify all touched pages, links, current-state boundaries, and no raw-data expansion; complete task checkboxes only from those receipts.

Rollback is page-by-page using the pre-write content captured by the connector response/current fetch. Because URLs and databases are preserved and no pages are deleted, rollback does not require data recreation.
