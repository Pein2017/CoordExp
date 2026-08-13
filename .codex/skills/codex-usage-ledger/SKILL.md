---
name: codex-usage-ledger
description: Read-only offline audit of Codex rollout sessions for subagent token usage, model-effort route cost, parent-invocation attempts, completion or acceptance evidence, and cost_per_accepted_task. Use when a session needs to inspect many subagents, scope the report to one root thread and its descendants, compare routing choices, estimate historical cost, or verify a ledger report from $CODEX_HOME/sessions.
---

# Codex Usage Ledger

Use this skill to produce an evidence-bounded routing-cost report from persisted
Codex rollout JSONL. Keep the ledger and `$CODEX_HOME` read-only; write only
reports requested by the caller, preferably under a temporary directory.

## Required boundaries

- Use `$CODEX_HOME`, never `~`, `$HOME`, or an inferred home directory.
- Resolve `CODEX_HOME` before running anything; stop with a clear error if it is
  unset or its `sessions/` directory is absent.
- Do not modify `config.toml`, model caches, app-server state, rollout JSONL,
  or Codex session state. Do not install packages into the active environment.
- Keep root-thread usage separate unless the caller explicitly asks for
  `--include-root`; the default report is subagent-only.
- Treat the provider price file as an estimate input, not an invoice.

## Run an audit

1. Identify the inclusive date window, task slice, or model slice. For a large
   session, prefer one lead-owned scan and let workers analyze its saved JSONL;
   if parallel scans are necessary, use disjoint windows and unique output
   paths.
2. Set a task-specific report directory, then invoke the bundled wrapper:

   ```bash
   export CODEX_HOME=/data/CoordExp/.codex
   ledger_report_dir=$(mktemp -d)
   python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
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

   To keep an audit attached to one lead task, pass the lead's persisted
   `thread_id` as `--root-thread-id`. The ledger then follows
   `parent_thread_id` recursively and ignores unrelated sessions. Add
   `--include-root` when the lead's own usage belongs in the total; omit it
   when the report should contain child-agent usage only:

   ```bash
   export CODEX_HOME=/data/CoordExp/.codex
   python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
     --root-thread-id "$SESSION_A_THREAD_ID" \
     --include-root \
     --prices /data/CoordExp/codex-usage-ledger/prices-gpt56-standard.toml \
     --disposition-policy followup_aware \
     --format json \
     --summary-out "$ledger_report_dir/session-a-summary.json" \
     --pretty
   ```

   `--thread-id ID` selects one exact rollout. `--session-id ID` selects
   records with an exact persisted `session_id`. These options are mutually
   exclusive with `--root-thread-id`; a scope that matches no rollout fails
   rather than silently producing a global report. The selected scope is
   recorded in `filters` and `scan.scope_filter`/`scan.scope_records`.

3. Run `--disposition-policy strict` when a real lead or verifier outcomes
   JSONL is available. Pass it with `--outcomes`; use `attempt_id` from the
   report and one disposition (`accepted`, `rework`, `escalated`, or `failed`)
   per line.
4. Read `summary.json` first. Check `scan.parse_errors`, the scan window,
   `totals.unpriced_segments`, the price source, and the disposition definition
   before quoting a number. Load `references/report-fields.md` when a field
   meaning is unclear.

## Interpret acceptance cost

- Treat `attempts.cost_per_accepted_task` as authoritative only when
  `attempts.policy == "strict"` and explicit outcomes exist.
- Label `completed` and `followup_aware` values as proxies in every report.
  `followup_aware` counts a completed child with no observed `followup_task` as
  an accepted proxy and a completed child with an observed `followup_task` as
  rework proxy. It is not proof that the parent accepted the work.
- Compare model × effort pairs only within comparable role, task class, brief,
  verifier, and surface. Do not produce a global model ranking from mixed
  roles, missing rates, interrupted tasks, or raw call volume.
- Use `route_pairs` for descriptive cost and disposition summaries. Require a
  meaningful sample and comparable acceptance evidence before changing a
  routing default.
- Report the formula, denominator, priced/unpriced counts, snapshot timestamp,
  and the strongest limitation alongside every headline cost.

## Return format

Return a compact audit with:

1. Scope and snapshot timestamp.
2. Scan counts and parse/pricing coverage.
3. Strict acceptance cost, if available; otherwise state that it is unavailable.
4. Explicitly labeled proxy cost, if requested.
5. A small route-pair table with sample counts and caveats.
6. Paths to the JSONL detail and summary artifacts.

## Bundled resources

- Use `scripts/run_ledger.py` as the stable entrypoint. It locates the source
  package through `CODEX_USAGE_LEDGER_ROOT` or the workspace default and never
  changes the environment.
- Consult [references/report-fields.md](references/report-fields.md) for the
  summary schema and evidence vocabulary.
