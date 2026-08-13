# Codex Usage Ledger

Read-only, offline reporting for Codex rollout sessions. It identifies
subagent threads, extracts persisted token receipts, joins model/reasoning
contexts, and optionally estimates cost from a user-owned provider price table.

The tool does not modify Codex, the app-server, `$CODEX_HOME`, or rollout
files. It uses only the persisted `session_meta`, `turn_context`, lifecycle,
`event_msg/token_count`, `event_msg/sub_agent_activity`, and agent tool-call
records.

## Quick start

```bash
cd /data/CoordExp/codex-usage-ledger

python -m codex_usage_ledger \
  --sessions "$CODEX_HOME/sessions" \
  --since 2026-08-01 \
  --format jsonl \
  --output /tmp/codex-subagent-usage.jsonl \
  --summary-out /tmp/codex-subagent-summary.json
```

The default output is one JSON object per subagent rollout. Use `--format csv`
for a spreadsheet-friendly view or `--format json` for one combined document.
Use `--max-files` while validating the command against a large session tree.

## Scope one parent task

Use `--root-thread-id` when the lead in Session A should account only for its
own rollout tree. It follows `parent_thread_id` recursively, so it includes
the lead and every descendant thread when combined with `--include-root`:

```bash
export CODEX_HOME=/data/CoordExp/.codex
python -m codex_usage_ledger \
  --sessions "$CODEX_HOME/sessions" \
  --root-thread-id "$SESSION_A_THREAD_ID" \
  --include-root \
  --prices prices-gpt56-standard.toml \
  --format json \
  --summary-out /tmp/session-a-summary.json \
  --pretty
```

For child-agent cost only, omit `--include-root`; this keeps the parent
process's own tokens out of the child ledger. `--thread-id ID` selects one
exact rollout, while `--session-id ID` selects records whose persisted
`session_id` matches exactly. These three scope options are mutually
exclusive. The summary records the selected mode in `filters` and
`scan.scope_filter`/`scan.scope_records`.

## Build the package

Build a wheel without changing the active Codex environment:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

The CLI entry point is `codex-usage-ledger`; the skill wrapper can also run the
source tree directly through `CODEX_USAGE_LEDGER_ROOT`.

## Pricing

Codex does not provide a universal provider price table. Copy
`prices.example.toml` to a private file, fill in the current rates, and pass it
with `--prices`:

```bash
python -m codex_usage_ledger \
  --sessions "$CODEX_HOME/sessions" \
  --prices /path/to/private/prices.toml \
  --since 2026-08-01 \
  --format csv \
  --output /tmp/codex-subagent-cost.csv
```

## Routing attempts and acceptance cost

The report attaches each child thread to its parent `spawn_agent` activity
when rollout receipts contain the matching event ID. A child thread is a
container; the routing unit is the parent invocation attempt. Optional lead
dispositions are JSONL, one object per line:

```json
{"attempt_id":"call_...","disposition":"accepted","note":"verifier passed"}
```

Valid dispositions are `accepted`, `rework`, `escalated`, and `failed`. Use
strict accounting when only explicit lead labels should count:

```bash
python -m codex_usage_ledger \
  --sessions "$CODEX_HOME/sessions" \
  --prices prices-gpt56-standard.toml \
  --outcomes /path/to/routing-outcomes.jsonl \
  --disposition-policy strict \
  --format json \
  --summary-out /tmp/codex-routing-summary.json
```

Historical rollouts usually do not persist an explicit acceptance label. For
an auditable exploratory estimate, `completed` treats a terminal
`task_complete` receipt as an explicitly named completion proxy, while
`followup_aware` also marks a completed child with an observed
`followup_task` as `rework`. The summary exposes the definition and
`attempts.cost_per_accepted_task`; neither proxy is human acceptance.

The report keeps `pricing.status` explicit:

- `ok`: a unique model and matching rate were found;
- `partial_or_missing`: at least one persisted model/effort segment has no
  matching rate or cannot be attributed safely;
- `missing_rate` and `ambiguous_model`: segment-level statuses explaining why
  a complete total was not produced;
- `missing_usage`: no persisted token receipt was found.

The summary JSON also exposes `totals.fully_priced_cost` (only sessions with
every route priced), `totals.known_route_cost` (priced route portions, a lower
bound when partial records exist), and `totals.unpriced_tokens`.

The cost calculation reports uncached input, cached input, cache-write input,
output, and reasoning dimensions separately. Cache-write and reasoning
inclusion are explicit flags in the rate file because providers do not all
bill those fields the same way.

`prices-gpt56-standard.toml` is a local snapshot populated from the user's
GPT-5.6 Standard API price screenshot on 2026-08-06. Treat it as an estimate
input and replace it when the provider price sheet changes.

## Interpretation limits

`latest_total_usage` is the cumulative usage recorded for the thread. Forked
subagent rollouts may embed the parent thread's history before the child's own
task; the ledger detects the first `task_started` boundary and reports the
post-boundary delta as `measured_usage`. The `usage_by_route` field uses those
cumulative-token deltas and the surrounding `turn_context` timeline to
approximate model/effort segments. It is useful for triage, but it is not an
invoice-grade per-response join; inspect multi-model threads before using them
as matched routing evidence.

This is a measurement and routing aid, not an invoice. Keep the provider,
gateway, pricing source, currency, and effective date with any decision-bearing
cost comparison.
