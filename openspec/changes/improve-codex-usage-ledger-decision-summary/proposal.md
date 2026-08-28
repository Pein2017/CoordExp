## Why

The ledger reliably extracts and prices rollout sessions, but its default summary repeats nearly one task-level group per session while omitting the route-level token, duration, and price-snapshot evidence needed to compare model-effort choices by completed task. The result is costly to load into an agent and still requires ad hoc post-processing for decisions such as `sol-medium` versus `terra-high/xhigh/max`.

## What Changes

- **BREAKING**: Make the default summary compact by omitting task-level `groups` and `attempt_routes`; retain them behind `--full-summary` while leaving detailed session output unchanged.
- Extend each `route_pairs` entry with bounded distributions for rollout wall time, measured usage, billable-token dimensions, and estimated cost so token efficiency and price effects can be compared together.
- Add a top-level pricing snapshot receipt containing the selected price path, SHA-256, effective date, sources, currencies, and rate keys.
- Add an outcomes-template output for strict human labeling, reject duplicate outcome identifiers and price-rate keys, and expose structural ambiguity signals without inferring acceptance from natural-language messages.
- Keep the package read-only, offline, standard-library-only, and explicit that non-strict dispositions are proxies.
- Repair local test/import and static-type diagnostics touched by the change.

## Capabilities

### New Capabilities

- `codex-usage-ledger-decision-reporting`: Compact, auditable model-effort decision summaries with price provenance and a fail-closed strict-outcome workflow.

### Modified Capabilities

None. No existing stable OpenSpec owns this standalone workspace tool.

## Impact

- Affects `codex-usage-ledger` CLI summary JSON, route aggregation, pricing/outcome validation, tests, package documentation, and the installed `codex-usage-ledger` skill documentation.
- Existing consumers that require `groups` or `attempt_routes` must pass `--full-summary`.
- No Codex rollout, app-server, cache, config, or session state is modified; no dependency, network lookup, database, dashboard, or plugin manifest is added.
