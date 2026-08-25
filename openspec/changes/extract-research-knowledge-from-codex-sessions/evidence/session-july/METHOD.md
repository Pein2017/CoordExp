# July session inventory method

## Frozen boundary

- Source root: `/data/CoordExp/.codex/sessions/2026/07/**/*.jsonl`.
- Capture contract: `evidence/CONTRACT.md`, captured `2026-08-25T12:35:18Z`.
- Raw count at capture and replay: **2,230** files.
- No network, Notion mutation, experiment, GPU work, install, deletion, or
  repository mutation was performed.

## Mechanical inventory

`manifest.tsv` was generated deterministically from the sorted July glob. The
generator read each source's first `session_meta` record for `source_id`, parent
thread, timestamp, and cwd; retained the source path verbatim; read a bounded
120,000-byte tail for assistant/event evidence; and computed a full-file
SHA-256 for duplicate detection. It emitted exactly the contract header and
one row per source. All 2,230 paths are under `/data/CoordExp/.codex/sessions/2026/07`
and exist at verification time. No exact-content duplicates were found.

The correction classifier first extracts user-task messages after removing
injected recommended-plugin, app-context, AGENTS, permissions, skills,
environment, and aborted-turn scaffolding. It assigns `high` only when the
sanitized task names a decision-bearing experiment, model behavior, metric,
evaluation, dataset, artifact interpretation, mechanism, or research route;
`medium` covers research-adjacent runtime, serialization, checkpoint, cache,
architecture, and reproducibility seams without a scientific result. Generic
Serena/MCP/tool-stack, Codex/subagent/session, plugin, release, hook, router,
config, install, and ordinary code-review tasks are `skip`. For child sessions
whose parent prompt is encrypted and unavailable, a strict fallback requires a
specific CoordExp research anchor plus an evidence/experiment anchor; otherwise
the row is skipped. `root_or_group_id` preserves the parent thread when present;
child rows remain present when grouped.

Final relevance counts:

| relevance | rows |
|---|---:|
| high | 139 |
| medium | 103 |
| skip | 1,988 |
| total | 2,230 |

Final disposition counts are `covered=6`, `needs-summary=240`, and
`development-skip=1,984`; no exact duplicate, invalid, historical, or
unexecuted row was promoted solely by the classifier.

## Deep-read and sensitivity checks

I independently replayed the six covered research candidates against raw JSONL
and extracted their session IDs, parent IDs, cwd, and assistant conclusions:

- painted-GT frozen baseline diagnosis: `019f307b-b066-7e11-a52e-4456fbcfaa93`;
- Qwen dense-enumeration / scope-history audit: `019f5ac5-c940-7463-af2c-3b3ed40f0555`;
- exact-history HF seam audit: `019f9a33-85e1-7223-870b-051e78fa523c`;
- presentation-cache preparation audit: `019fa7af-f7fd-7b71-974e-5ec73834bc69`;
- bounded online permutation result: `019fad9e-a47e-71b0-a555-7750fd99e7c4`;
- Codex/Claude memory bootstrap: `019fb270-0e93-7223-a1f5-c94aaa2d8385`.

I sampled the first 10 high rows, first 10 medium rows, and first 10 skip rows
from the corrected sorted manifest and opened their raw JSONL user/task text
and assistant tails. Six skip-classified rows had research-bearing
artifact/scoring traces on this check; they remain manually promoted and are
recorded in their `reason` fields:
`019f237c-df75-73f3-8f11-095bb56acd60`,
`019f3726-482a-7d10-bb69-5c8a0f17ca63`,
`019f515c-9e25-72a1-a8ed-5f91aad78776`,
`019f5492-9eaa-7c53-85ea-f455485702ec`,
`019f5492-c599-71d1-8e60-e4f7575fb094`, and
`019f5492-f2e8-7902-87ca-938f4ef1770a`. This is the skipped-boundary
sensitivity result; the remaining 1,988 skips retain a short
generic-development reason and were not synthesized. Four initially high
rows were demoted to medium after this round because they described
research-adjacent infrastructure/spec seams without a decision-bearing result.

No L2 receipt was available: this session exposed no callable
`gpt-5.6-luna` route, so no delegated result is claimed or substituted.

## Correction receipt

- Previous tail-keyword pass: `high=1,028`, `medium=967`, `skip=235`.
- Corrected user-task-first pass: `high=139`, `medium=103`, `skip=1,988`.
- Concrete false-positive checks now classify both
  `019f9767-cee2-75f2-b1b5-8d21b5fef410` (Codex wait/config research) and
  `019fb6da-5f84-7e10-9b95-31667140897b` (Serena Light release audit) as
  `development-skip`.
- The residual boundary is encrypted child prompts: when no usable user task
  is present, only a strict specific-research-plus-evidence fallback can retain
  a row; generic assistant-tail vocabulary is insufficient.

## Evidence boundary

The inventory is a triage and provenance product. `synthesis.md` promotes only
findings directly supported by the named raw sessions. Mechanics smoke,
teacher-forced, sampled/retrieval, cache, memory, and infrastructure findings
remain bounded to their observed surface; they do not establish natural-model
behavior, trainability, deployment readiness, or current authority.
