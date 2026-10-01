---
name: codex-usage-ledger
description: Audit persisted Codex rollout usage, model-effort routing cost, or acceptance-cost evidence for a bounded task or date window. Not for ordinary session recall or live quota checks.
---

# Codex Usage Ledger

Produce bounded, evidence-based reports from persisted Codex rollout JSONL. Keep rollout data, `$CODEX_HOME`, and runtime state read-only; write only requested reports, preferably to a temporary or task-owned directory. Resolve `$CODEX_HOME` and require its `sessions/` directory; never infer it from `~` or `$HOME`. Do not install packages or modify config, caches, app-server, or session state.

## Choose and run a bounded audit

1. Select the date window or task/model slice. Default output is subagent-only; include root usage only when requested. For a lead task, `--root-thread-id` follows `parent_thread_id`; `--include-root` adds the lead’s own usage. A child thread is a container: when possible, the report joins it to the parent `spawn_agent` invocation by event ID; otherwise it records thread-level fallback evidence. Exact `--thread-id` and `--session-id` filters are mutually exclusive with that subtree filter. An empty match fails rather than becoming a global scan.
2. For exact known pages, repeat `--rollout PATH` for every relevant physical page. Filters cannot discover missing pages outside that input set. Do not combine explicit pages with `--sessions` or `--max-files`. Read [Invocation Examples](references/invocation-examples.md) and use the example matching the scope.
3. Use strict outcomes when actual lead/verifier labels exist. Only explicit `--outcomes` labels count as acceptance; `--require-outcomes` also requires strict policy and labels every emitted record. Generate/edit the template as needed; duplicate identifiers fail closed. Unknown labels are never acceptance evidence. Proxy policies are descriptive only.
4. Read `summary.json` first. Check scope, parse errors, priced/unpriced counts, pricing snapshot and date, and disposition definition; consult [Report fields](references/report-fields.md) when meanings are unclear. The summary is compact by default; use `--full-summary` only when `groups` or `attempt_routes` are needed. Re-run after inputs, labels, or prices change; output-file existence is not a cache.

## Preserve accounting identity and limits

Modern `token_usage_record` receipts are counted once per `(thread_id, response_id)` across pages; identical duplicates count once, conflicting usage is excluded and reported, and foreign-thread receipts are excluded. Legacy cumulative `event_msg/token_count` uses deltas, treats counter decreases as resets, and excludes inherited fork history at the task boundary. Mixed files use modern receipts to avoid double counting. Usage is post-boundary persisted token evidence, not invoice billing; reasoning output is a subset of output and is not added twice. Model and effort attribution uses the latest matching persisted context at or before a receipt, so later context cannot receive earlier usage.

Calendar `--since/--until` dates are inclusive, followed by a half-open UTC receipt window. Explicit `--receipt-since/--receipt-until` is start-inclusive/end-exclusive, applies to receipts rather than lifecycle wall time, and cannot be combined with calendar filters. Boundaries must come from task evidence. An unwindowed resumed thread is cumulative: never add it to its earlier report. Use disjoint evidenced windows and distinct task-chain attempt IDs for incremental repair cost. Record pages, boundaries, and outcomes used.

Treat prices as dated estimates, never invoices. Report missing prices as unknown; priced partial totals are not complete costs. `attempts.cost_per_accepted_task` is `accepted_estimated_cost / priced_accepted_attempts`; it is the mean of priced accepted records, excluding separately labeled failures/rework, not full task acceptance cost. For that, combine every incremental attempt through acceptance and attributable lead costs, leaving unknown costs explicit. `completed` and `followup_aware` are proxies, not human acceptance: the latter treats a completed child without an observed follow-up as accepted proxy and one with a follow-up as rework proxy. Compare model × effort only across comparable role, task class, brief, verifier, and task surface; route pairs are descriptive and do not justify global rankings or token-savings claims without direct evidence.

## Return

Give scope and snapshot time, scan and parse counts, pricing coverage, strict acceptance cost only if supported (otherwise say unavailable), requested proxy cost clearly labeled, and a small model × effort table with sample counts and caveats. Include formula and denominator, priced/unpriced counts, price snapshot timestamp, and strongest limitation for each headline cost. Link the JSONL detail and summary artifacts.

Use `scripts/run_ledger.py` as the stable entrypoint; it locates the source through `CODEX_USAGE_LEDGER_ROOT` or the workspace default without changing the environment. The ledger recomputes reports and has no persistent result cache.
