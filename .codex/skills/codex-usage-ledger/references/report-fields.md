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
| `route_pairs` | Descriptive model × effort aggregates; do not treat mixed-task pairs as a benchmark. |

The matching CLI filters are also copied into `filters`:
`thread_id` (one exact rollout), `session_id` (exact persisted session ID), and
`root_thread_id` (that thread plus all descendants). Only one can be set. A
root-subtree report still excludes the root record unless `include_root` is
true; this is useful when separating parent-process cost from child-agent cost.

## Evidence vocabulary

- `strict`: only `--outcomes` labels count as acceptance.
- `completed`: terminal `task_complete` is a completion proxy.
- `followup_aware`: completion with no observed `followup_task` is an accepted
  proxy; completion followed by `followup_task` is a rework proxy.
- `spawn_event_id`: the attempt was joined to a parent `spawn_agent` activity.
- `thread:<id>`: no matching spawn activity was persisted; the record is a
  thread-level fallback evidence.
