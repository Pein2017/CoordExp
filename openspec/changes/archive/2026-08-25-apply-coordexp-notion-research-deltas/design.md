## Context

See [proposal.md](proposal.md). The source census and semantic integration are already accepted in `../extract-research-knowledge-from-codex-sessions/`; this change owns only reviewed Notion writeback. Existing Notion pages and stable IDs are the write targets, while repository research owners remain authoritative.

## Goals / Non-Goals

**Goals:**

- Apply one compact delta per existing semantic owner.
- Preserve citations to exact local owners and distinguish evidence, interpretation, decision, and unresolved work.
- Produce a reproducible refetch receipt for every external mutation.

**Non-Goals:**

- No raw transcript/document mirroring, new database/schema, page duplication, deletion, moving, experiment execution, or production mutation.
- No resolution of remaining scientific contradictions beyond labeling and routing them.

## Decisions

### 1. Update existing owners in place

Fetch each page immediately before mutation, then use exact search-and-replace or bounded append operations. This preserves stable URLs and backlinks. Creating replacement pages was rejected because it would split authority and increase Notion volume.

### 2. Treat the reviewed update plan as the content router

`../extract-research-knowledge-from-codex-sessions/notion-update-plan.md` owns the target list and content boundaries; `research-findings.md` supplies evidence-level detail. The live repository and current Notion page are rechecked before each delta. Raw manifests remain local provenance only.

### 3. Verify external state after every write wave

Refetch mutated pages and check a distinctive marker plus preserved authority/lifecycle language. Record page ID, operation, marker, and result in `writeback-receipt.md`. A failed or ambiguous update stops that page without broad replacement.

### 4. Preserve the canonical suffix but mask the newline

The serialization owner will state the resolved three-layer contract: the Qwen text ends in `<|im_end|>\n`; `<|im_end|>` is the final supervised token; the terminal newline is intentionally ignored. This does not change production behavior.

## Risks / Trade-offs

- [Page content drift makes exact replacement unsafe] → Refetch immediately and prefer a compact appended reconciliation section.
- [Summary overstates evidence] → Reuse the integrated not-claimed boundaries and keep contradictions as `needs-adjudication`.
- [Notion expansion] → Update only named owners, omit manifests/raw prose, and create no new databases or parallel pages.
- [Partial connector failure] → Record per-page PASS/FAIL and retry only the affected page after refetch.

## Migration Plan

1. Fetch all target owners and freeze their current content identities in the receipt.
2. Apply compact deltas in independent owner groups, beginning with global contract and serialization.
3. Refetch and verify each group before continuing.
4. Validate the OpenSpec and close only when every intended page has a receipt.

Rollback is page-local: retain the exact pre-write section in the receipt and replace only the inserted reconciliation block if a correction is required. No destructive archive operation is part of this change.
