# Report fields

Use these fields when checking a ledger result.

| Field | Meaning |
| --- | --- |
| `scan.files_seen` | Rollout files considered after the date/window filter. |
| `scan.scope_filter` | `all`, `thread`, `session`, or `root_thread_subtree`. |
| `scan.scope_records` | Records matching the requested scope before root exclusion. |
| `scan.records_emitted` | Subagent records emitted; root sessions are excluded by default. |
| `scan.parse_errors` | JSONL parse/read errors in emitted records. |
| `totals.measured_tokens` | Post-boundary token delta measured by the ledger. |
| `totals.known_route_cost` | Sum of priced route segments; partial records can make this a lower bound. |
| `totals.unpriced_segments` | Route segments excluded from complete pricing. |
| `attempts.attempts` | One child invocation/lifecycle record. |
| `attempts.accepted_attempts` | Records classified as accepted under the selected policy. |
| `attempts.priced_accepted_attempts` | Accepted records with a complete price. This is the denominator for the cost metric. |
| `attempts.cost_per_accepted_task` | `accepted_estimated_cost / priced_accepted_attempts`; only authoritative with explicit outcomes. |
| `attempts.accepted_definition` | The exact strict or proxy definition used for the number. |
| `pricing_snapshot` | Resolved price path, SHA-256, effective date, sources, currencies, and loaded rate keys; `unconfigured` when no price file was selected. |
| `route_pairs` | Compact descriptive model × effort aggregates; do not treat mixed-task pairs as a benchmark. |
| `route_pairs[].rollout_wall_seconds` | Observation count plus total, mean, median, and nearest-rank P90 elapsed wall time; this includes orchestration and waiting. |
| `route_pairs[].measured_tokens` | Distributions for persisted post-boundary token dimensions. |
| `route_pairs[].billable_tokens` | Distributions for fully priced uncached, cached, cache-write, output, and reasoning dimensions. |
| `route_pairs[].estimated_cost` | Distribution over fully priced attempt costs. |

The matching CLI filters are also copied into `filters`:
`thread_id` (one exact rollout), `session_id` (exact persisted session ID), and
`root_thread_id` (that thread plus all descendants). Only one can be set. A
root-subtree report still excludes the root record unless `include_root` is
true; this is useful when separating parent-process cost from child-agent cost.
`filters.summary_mode` is `compact` by default. `--full-summary` changes it to
`full` and restores `groups` plus `attempt_routes`; detailed session output is
unchanged.

## Evidence vocabulary

- `strict`: only `--outcomes` labels count as acceptance.
- `completed`: terminal `task_complete` is a completion proxy.
- `followup_aware`: completion with no observed `followup_task` is an accepted
  proxy; completion followed by `followup_task` is a rework proxy.
- `spawn_event_id`: the attempt was joined to a parent `spawn_agent` activity.
- `thread:<id>`: no matching spawn activity was persisted; the record is a
  thread-level fallback evidence.
- `proxy_ambiguity_reasons`: structural warnings from persisted lifecycle and
  interaction receipts; never a semantic classifier of agent messages.
