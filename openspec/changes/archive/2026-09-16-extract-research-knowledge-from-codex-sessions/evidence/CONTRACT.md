# Exploration Contract

Capture time: `2026-08-25T12:35:18Z`

## Frozen source boundary

- Sessions: `/data/CoordExp/.codex/sessions/2026/{05,06,07,08}/**/*.jsonl` whose mtime/content existed at capture. Counts: May 1,043; June 1,543; July 2,230; August 1–25 1,462; total 6,278.
- Memory/index routing aids: `/data/CoordExp/.codex/memories/MEMORY.md` and directly referenced rollout summaries. These aid triage but do not replace raw-session verification.
- Research documents: `/data/CoordExp/research/**/*.md` (51 files) and Markdown under registered CoordExp worktrees (9,694 paths before deduplication). Worktree path, HEAD, branch, dirty state, and content identity must remain visible.
- Canonical fixed point: `/data/CoordExp` current checkout. Canonical research evidence: locked `/data/CoordExp/.worktrees/research-probes`; bounded infrastructure owner: locked `/data/CoordExp/.worktrees/research-probe-infras`. A newer explicit decision or a narrower current unit may supersede an older compass/handoff.

## Session relevance

Classify every source row as `high`, `medium`, or `skip`.

`high` if the session contains decision-bearing CoordExp research: hypothesis or mechanism work; experiment design/execution/result; artifact/metric interpretation; model behavior diagnosis; data, serialization, geometry, evaluation, or claim semantics; negative/invalid/retired route; continuation/authorization decision.

`medium` if research influence is plausible but indirect: infrastructure or workflow changed whether evidence was valid, reproducible, attributable, recoverable, or authorized. Deep-read enough to decide whether it contains a unique research-relevant finding.

`skip` for generic software/plugin/tool/UI/refactor/install/review/maintenance work with no material effect on a CoordExp research question, evidence surface, scientific validity, current route, or authorization. Record a short skip reason; do not synthesize its development details.

Include research-specific child/subagent sessions, but group them with their root thread where identity is available. Child repetition is `duplicate`, while a unique child finding retains its own source row.

## Required `manifest.tsv` columns

Use one header exactly:

```text
source_path	source_id	root_or_group_id	date	cwd_or_worktree	relevance	disposition	topics	evidence_surface	execution_lifecycle	current_owner	duplicate_key	confidence	reason
```

Allowed `disposition`: `covered`, `needs-summary`, `needs-adjudication`, `duplicate`, `historical`, `unexecuted`, `invalid`, `development-skip`.

Every session source file in the package must have a row, even when skipped. Research-document rows cover every discovered Markdown path after the package's declared root expansion; byte-identical duplicates may share a duplicate key but retain individual source rows.

## Required `synthesis.md` entry

For each unique valuable finding:

```text
### Plain-language finding title
- Sources: exact session/document paths and IDs
- Current owner: exact unit/result/decision/track, or unowned
- Question and contrast:
- Observed evidence surface:
- Scientific disposition:
- Technical/infrastructure disposition:
- Decision or continuation impact:
- Not claimed:
- Notion coverage: covered | needs-summary | needs-adjudication, with page/owner if known
```

Separate planned protocol, executed evidence, interpretation, decision, mechanism promotion, and implementation authorization. Forced-prefix, teacher-forced, retrieval, smoke, mechanics-only, invalid, partial, and unexecuted evidence never promote themselves.

## Package acceptance

- L1 may spawn at most two `gpt-5.6-luna` L2 workers with `fork_turns: "none"`, explicit effort, independent subranges, no further delegation, and no writes outside its package directory.
- L1 must inspect L2 output, replay counts, sample at least 10 high/medium rows and 10 skips (or all if fewer), reconcile duplicates, and return one compact acceptance packet.
- Authored Markdown uses `apply_patch`. A deterministic bulk TSV generator is allowed for the large mechanical manifest only if its command/script and source boundary are recorded in `METHOD.md`; no dependency installation.
- No Notion mutation, experiment launch, code/config/research/progress edit, commit, push, deletion, or raw transcript/document dump.
- Stop with `NEEDS_CONTEXT` on unreadable/corrupt sources or ambiguous identity; use `HOLD` for a conclusion-changing semantic conflict; do not silently guess.

## Package write roots

- `evidence/session-early/`
- `evidence/session-july/`
- `evidence/session-august/`
- `evidence/research-docs/`

Each package must create `METHOD.md`, `manifest.tsv`, and `synthesis.md` only under its own root.
