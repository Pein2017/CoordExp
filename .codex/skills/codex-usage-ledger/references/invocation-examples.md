# Ledger Invocation Examples

Read the entrypoint boundaries first. These examples do not authorize a broader
scan or root-thread inclusion than requested. Keep the existing CODEX_HOME.

## Date-window report (subagents only)

```bash
: "${CODEX_HOME:?CODEX_HOME must be set}"
ledger_report_dir=$(mktemp -d)
conda run -n ms python /data/CoordExp/.agents/skills/codex-usage-ledger/scripts/run_ledger.py \
  --sessions "$CODEX_HOME/sessions" \
  --since YYYY-MM-DD \
  --until YYYY-MM-DD \
  --prices /data/CoordExp/codex-usage-ledger/prices-gpt56-standard.toml \
  --disposition-policy followup_aware \
  --format jsonl \
  --output "$ledger_report_dir/attempts.jsonl" \
  --summary-out "$ledger_report_dir/summary.json" \
  --pretty
```

## One root task and its descendants

Set SESSION_A_THREAD_ID to the persisted lead thread ID. This example includes
the root; omit --include-root unless that total was requested.

```bash
: "${CODEX_HOME:?CODEX_HOME must be set}"
: "${SESSION_A_THREAD_ID:?Set the persisted lead thread ID}"
ledger_report_dir=$(mktemp -d)
conda run -n ms python /data/CoordExp/.agents/skills/codex-usage-ledger/scripts/run_ledger.py \
  --root-thread-id "$SESSION_A_THREAD_ID" \
  --include-root \
  --prices /data/CoordExp/codex-usage-ledger/prices-gpt56-standard.toml \
  --disposition-policy followup_aware \
  --format json \
  --summary-out "$ledger_report_dir/session-a-summary.json" \
  --pretty
```

## Exact worker pages and a corrective phase

Use paths and UTC boundaries from retained task evidence. The following paths,
IDs and times are illustrative; replace them. The final boundary is exclusive.
Repeat --rollout for all physical pages needed by the scope, including context
pages when route attribution requires them. These flags replace --sessions.

```bash
python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
  --rollout /path/to/rollout-worker-page1.jsonl \
  --rollout /path/to/rollout-worker-page2.jsonl \
  --thread-id WORKER_THREAD_ID \
  --include-root \
  --receipt-since 2026-09-13T10:00:00Z \
  --receipt-until 2026-09-13T10:10:00Z \
  --prices /path/to/dated-prices.toml \
  --outcomes /path/to/repair-outcome.jsonl \
  --disposition-policy strict --require-outcomes \
  --output /tmp/repair-usage.jsonl \
  --summary-out /tmp/repair-summary.json
```

`--include-root` is needed for root records; omit it for child-only accounting.
The outcomes file may label the exact worker thread for this window. When the
same worker has another phase, use a separate scoped outcome file; do not put
two conflicting labels for its thread ID into one file. In delegation receipts,
use nonoverlapping phase IDs and incremental cost, not the full resumed total.

`--require-outcomes` checks labeling only. Inspect parse errors, usage conflicts,
missing prices and the actual acceptance evidence as usual. Unwindowed strict
reports without the flag remain useful for usage discovery, but unknown labels
cannot supply an acceptance denominator. Recompute after changed inputs rather
than skipping because an older output file already exists.
