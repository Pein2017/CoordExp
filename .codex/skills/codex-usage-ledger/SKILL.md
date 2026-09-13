---
name: codex-usage-ledger
description: Audit persisted Codex rollout usage, model-effort routing cost, or acceptance-cost evidence for a bounded task or date window. Not for ordinary session recall or live quota checks.
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

   Read [Invocation Examples](references/invocation-examples.md) before
   running the wrapper; use the example matching the requested scope.

   To keep an audit attached to one lead task, pass the lead's persisted
   `thread_id` as `--root-thread-id`. The ledger then follows
   `parent_thread_id` recursively and ignores unrelated sessions. Add
   `--include-root` when the lead's own usage belongs in the total; omit it
   when the report should contain child-agent usage only.

   `--thread-id ID` selects one exact rollout. `--session-id ID` selects
   records with an exact persisted `session_id`. These options are mutually
   exclusive with `--root-thread-id`; a scope that matches no rollout fails
   rather than silently producing a global report. The selected scope is
   recorded in `filters` and `scan.scope_filter`/`scan.scope_records`.

3. Run `--disposition-policy strict` when a real lead or verifier outcomes
   JSONL is available. Generate a starting file with
   `--outcomes-template-out PATH`, replace the `REPLACE_ME` values, remove
   attempts that remain unlabeled, and pass the result with `--outcomes`. Use
   one disposition (`accepted`, `rework`, `escalated`, or `failed`) per line;
   duplicate identifiers fail closed. For decision-bearing labeled reports, add
   `--require-outcomes`: it requires strict policy, an outcomes file and explicit
   labels for every emitted record. Ordinary strict audits without outcomes
   remain allowed, but unknown dispositions are not acceptance evidence.
4. Read `summary.json` first. Check `scan.parse_errors`, the scan window,
   `totals.unpriced_segments`, `pricing_snapshot`, and the disposition
   definition before quoting a number. The default summary is compact. Pass
   `--full-summary` only when task-level `groups` or role-level
   `attempt_routes` are needed. Load `references/report-fields.md` when a field
   meaning is unclear.

## Interpret acceptance cost

- `attempts.cost_per_accepted_task` is the mean cost of priced records labeled
  accepted; it excludes separately labeled failed/rework records. Even with
  strict outcomes, it is not the complete cost through task acceptance. Use
  the delegation task chain for that denominator, retaining all attempts and
  attributable lead costs. Missing pricing or lead costs stay unknown.
- Label `completed` and `followup_aware` values as proxies in every report.
  `followup_aware` counts a completed child with no observed `followup_task` as
  an accepted proxy and a completed child with an observed `followup_task` as
  rework proxy. It is not proof that the parent accepted the work.
- Compare model × effort pairs only within comparable role, task class, brief,
  verifier, and surface. Do not produce a global model ranking from mixed
  roles, missing rates, interrupted tasks, or raw call volume.
- Use `route_pairs` for measured/billable token, rollout wall-time, estimated
  cost, and disposition distributions. Require a meaningful sample and
  comparable acceptance evidence before changing a routing default. Wall time
  includes waiting and orchestration delay; it is not compute time.
- Report the formula, denominator, priced/unpriced counts, snapshot timestamp,
  and the strongest limitation alongside every headline cost.

## Resumed workers and bounded input

When exact physical rollout paths are already known, pass repeatable
`--rollout PATH` to bypass session-tree discovery. Supply every relevant page;
thread/subtree filters only see the supplied input set. Do not combine this
with `--sessions` or `--max-files`. Prefer this route over copying an entire
session tree or rescanning it once per worker. Keep root inclusion explicit.

A resumed thread's unwindowed report is cumulative. Do not add the first-turn
report to the resumed cumulative report. For a repair, use explicit timezone-
aware `--receipt-since START --receipt-until END` boundaries: start inclusive,
end exclusive. Do not mix them with calendar `--since/--until`. Boundaries must
come from retained task/receipt evidence, not guessed wall time. One window is
not automatic phase detection: metadata, attempt IDs and lifecycle wall time
still describe the rollout. Supply outcomes specifically for the selected
window; assign a distinct task-chain attempt ID when exporting it.

Use the existing [delegation evidence workflow](../native-agent-team-guide/references/evidence-workflow.md)
for incremental attempt costs and final task acceptance. Record the input pages,
receipt window, outcomes and dated prices in cost evidence. Check reconciliation
warnings and parse/pricing coverage before using numbers. Shared lead overhead
must not be silently allocated to every child or counted again as takeover.

The CLI recomputes reports; it has no persistent result cache. Reusing a filename
is not proof of freshness. If inputs, scope, outcomes or prices change, rerun the
bounded invocation. For decision-bearing work prefer immutable input snapshots
and a task-owned report directory. Do not build a cache or background scanner
just to avoid a small explicit-file scan.

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
