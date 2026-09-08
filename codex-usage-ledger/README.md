# Codex Usage Ledger

Read-only, offline reporting for Codex rollout sessions. It identifies
subagent threads, extracts persisted token receipts, joins model/reasoning
contexts, and optionally estimates cost from a user-owned provider price table.

The tool does not modify Codex, the app-server, `$CODEX_HOME`, or rollout
files. It uses only the persisted `session_meta`, `turn_context`, lifecycle,
`token_usage_record`, `event_msg/token_count`, `event_msg/sub_agent_activity`,
and agent tool-call records.

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
The default summary is compact: it keeps `route_pairs` and omits redundant
task-level `groups` and role-level `attempt_routes`. Pass `--full-summary` when
an existing consumer needs those legacy arrays. Use `--max-files` while
validating the command against a large session tree.

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

Declare `[metadata].effective_date` and `[metadata].source` in the price TOML.
The summary's `pricing_snapshot` binds the resolved path, SHA-256, metadata,
currencies, sources, and loaded rate keys to the resulting estimate. Duplicate
provider/model rates and non-boolean billing flags fail closed.

## Routing attempts and acceptance cost

The report attaches each child thread to its parent `spawn_agent` activity
when rollout receipts contain the matching event ID. A child thread is a
container; the routing unit is the parent invocation attempt. Optional lead
dispositions are JSONL, one object per line:

```json
{"attempt_id":"call_...","disposition":"accepted","note":"verifier passed"}
```

Valid dispositions are `accepted`, `rework`, `escalated`, and `failed`. Use
`--outcomes-template-out` to create one context-bearing placeholder row per
attempt, edit it down to explicit valid outcomes, then use strict accounting:

```bash
python -m codex_usage_ledger \
  --sessions "$CODEX_HOME/sessions" \
  --prices prices-gpt56-standard.toml \
  --outcomes-template-out /tmp/routing-outcomes-template.jsonl \
  --summary-out /tmp/codex-routing-summary.json
```

After replacing `REPLACE_ME` dispositions:

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
`attempts.cost_per_accepted_task`; neither proxy is human acceptance. Attempt
rows also expose interaction/completion counts and bounded
`proxy_ambiguity_reasons`; these do not alter disposition or classify message
text.

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
bill those fields the same way. Each `route_pairs` entry aggregates these
billable dimensions, measured usage, estimated cost, and labeled rollout wall
time with observation count, total, mean, median, and nearest-rank P90.

`prices-gpt56-standard.toml` is a local snapshot populated from the user's
GPT-5.6 Standard API price screenshot on 2026-08-06. Treat it as an estimate
input and replace it when the provider price sheet changes.

## Interpretation limits

Modern `token_usage_record` entries are the primary accounting source. The
ledger validates their per-response usage, keeps receipts owned by the
persisted `thread_id`, and deduplicates `(thread_id, response_id)` across
pagination files. Duplicate receipts with the same usage are counted once;
metadata differences are reported under `scan.usage.metadata_conflicts`.
Duplicates with conflicting usage are excluded and reported under
`scan.usage`. A receipt whose thread identity differs from the enclosing file
is excluded as foreign. Route attribution uses the latest matching context at
or before the receipt timestamp, so a future context cannot receive earlier
usage.

Files without modern receipts use the older cumulative `event_msg/token_count`
fallback. A reset to a lower cumulative counter starts a new legacy epoch, and
the existing task boundary still excludes inherited fork history. Files that
contain both formats report `usage_format: "mixed"` and use modern receipts so
the legacy events cannot double-count the same responses.

`--since` and `--until` are inclusive calendar-date filters in the CLI. After
file discovery, receipt and legacy event timestamps are applied as a half-open
UTC window; a receipt without a valid timestamp is excluded from a bounded
window and reported. The summary records the effective `receipt_since` and
`receipt_until` boundaries.

Reasoning output is a subset of `output_tokens`; it is retained as a diagnostic
dimension and is never added to `total_tokens` a second time. These persisted
receipts are measurement evidence, not invoice-grade provider billing.

This is a measurement and routing aid, not an invoice. Keep the provider,
gateway, pricing source, currency, and effective date with any decision-bearing
cost comparison.
