# Ledger Invocation Examples

Read the entrypoint boundaries first. These examples do not authorize a broader
scan or root-thread inclusion than requested. Keep the existing CODEX_HOME.

## Date-window report (subagents only)

```bash
: "${CODEX_HOME:?CODEX_HOME must be set}"
ledger_report_dir=$(mktemp -d)
conda run -n ms python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
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
conda run -n ms python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
  --root-thread-id "$SESSION_A_THREAD_ID" \
  --include-root \
  --prices /data/CoordExp/codex-usage-ledger/prices-gpt56-standard.toml \
  --disposition-policy followup_aware \
  --format json \
  --summary-out "$ledger_report_dir/session-a-summary.json" \
  --pretty
```
