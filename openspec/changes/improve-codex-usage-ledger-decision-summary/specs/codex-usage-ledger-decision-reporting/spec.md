## Purpose

Provide compact, provenance-bearing model-effort summaries that let operators compare token usage, elapsed rollout time, estimated cost, and explicit acceptance evidence without loading redundant task-level aggregates.

## ADDED Requirements

### Requirement: Compact summary is the default
The ledger SHALL emit totals, attempt totals, route-pair aggregates, scan evidence, filters, and pricing provenance in the default summary. It SHALL omit task-level `groups` and role-level `attempt_routes` unless the operator explicitly requests the full summary, while detailed per-session output remains unchanged.

#### Scenario: Default summary
- **WHEN** an operator requests a summary without the full-summary option
- **THEN** the summary contains `route_pairs` and does not contain `groups` or `attempt_routes`

#### Scenario: Full summary migration path
- **WHEN** an operator requests the full-summary option
- **THEN** the summary additionally contains the legacy `groups` and `attempt_routes` arrays

### Requirement: Route pairs expose decision-bearing distributions
Each model-effort route pair SHALL report sample counts and bounded distributions for rollout wall time, measured token dimensions, billable-token dimensions, and estimated cost. Missing timestamps or pricing SHALL remain visible through observation and priced counts rather than being converted to zero.

#### Scenario: Fully measured route pair
- **WHEN** a route pair contains sessions with start/end timestamps, token receipts, and matching prices
- **THEN** its summary reports total, mean, median, and P90 values for available measurements and preserves the existing disposition and accepted-cost counts

#### Scenario: Partially priced route pair
- **WHEN** at least one session in a route pair lacks a matching rate
- **THEN** priced and unpriced counts identify the partial coverage and missing billable measurements are not treated as zero-cost sessions

### Requirement: Summary binds the price snapshot
When a price file is supplied, the summary SHALL identify that exact file using its resolved path and SHA-256 and SHALL report its declared effective date, source strings, currencies, and loaded rate keys. When no price file is supplied, the summary SHALL explicitly report that no pricing snapshot was selected.

#### Scenario: Price file supplied
- **WHEN** the ledger loads a valid price TOML
- **THEN** `pricing_snapshot` provides a deterministic receipt for the selected contents

#### Scenario: No price file supplied
- **WHEN** the ledger runs without `--prices`
- **THEN** `pricing_snapshot` explicitly records an unconfigured state instead of implying current prices

### Requirement: Strict outcomes are easy to label and fail closed
The ledger SHALL optionally write one human-editable JSONL template row per attempt for strict outcome labeling. Duplicate outcome identifiers and duplicate provider-model price keys SHALL fail before a report is written.

#### Scenario: Generate outcomes template
- **WHEN** an operator requests an outcomes template
- **THEN** the ledger writes attempt identity and routing context with an unmistakable placeholder disposition without changing rollout data

#### Scenario: Duplicate outcome identifier
- **WHEN** an outcomes JSONL repeats an identifier
- **THEN** the ledger fails with an error that names the duplicate

#### Scenario: Duplicate price key
- **WHEN** a price TOML repeats a provider-model key
- **THEN** the ledger fails with an error that names the duplicate

### Requirement: Proxy ambiguity remains structural and explicit
The ledger SHALL expose observed interaction and completion counts plus bounded ambiguity reasons on attempts. It MUST NOT infer acceptance by classifying natural-language child or parent messages.

#### Scenario: Reused or continued child
- **WHEN** persisted activity shows repeated completion events or interactions not represented by the follow-up count
- **THEN** the attempt contains the structural counts and corresponding ambiguity reasons while retaining the selected strict or proxy disposition policy

### Requirement: Offline read-only operation is preserved
The decision report SHALL use only persisted rollout files and a user-selected local price table. It MUST NOT update Codex state, fetch prices from the network, install runtime dependencies, or mutate session data.

#### Scenario: Decision audit
- **WHEN** an operator runs the enhanced ledger against `$CODEX_HOME/sessions`
- **THEN** only explicitly selected report and template output paths are written
