## Context

See [proposal.md](proposal.md). The existing package already has the required session parsing, route attribution, billable-token calculation, pricing segments, attempt annotations, and JSON writers. The change should deepen those existing owners instead of adding a second analysis layer.

## Goals / Non-Goals

**Goals:**

- Make the default summary small enough to inspect directly while preserving an explicit compatibility path.
- Aggregate the existing per-session evidence into one model-effort decision surface.
- Bind every cost comparison to the exact local price input.
- Make strict outcome labeling practical and make ambiguous proxy evidence visible.

**Non-Goals:**

- Inferring semantic acceptance, task class, quality, or task equivalence from message text.
- Adding a database, cache/index service, dashboard, generic query language, network price lookup, dependency, or plugin manifest.
- Changing rollout parsing or the detailed per-session JSONL schema except for additional attempt annotations.

## Decisions

### Keep one enriched item representation

Route metrics will be computed from the existing enriched and annotated item dictionaries. Billable dimensions will be summed from their existing pricing segments, and measured dimensions will use `measured_usage`. This avoids a parallel report model and keeps missing evidence distinguishable from zero.

Alternative considered: add a dataframe or analytics dependency. Rejected because twelve route pairs do not justify another runtime or representation.

### Use compact-by-default with one explicit compatibility flag

The CLI will always emit `route_pairs`. It will add the existing `groups` and `attempt_routes` only with `--full-summary`, and the selected summary mode will be recorded in `filters`.

Alternative considered: keep the large default and add a second decision-summary file. Rejected because it leaves the skill's primary `summary.json` misleadingly expensive and creates two summary authorities.

### Use deterministic bounded statistics

Each numeric distribution will report observations, total, mean, median, and nearest-rank P90. Wall time will be labeled as rollout wall time, not active compute time. Missing timestamp or billing observations are omitted from the distribution and remain visible through observation counts and existing priced/unpriced counts.

Alternative considered: arbitrary percentile configuration. Rejected until a measured consumer needs it.

### Treat the price table as a receipt-bearing input

Pricing will retain its current per-segment assumptions. A small receipt helper will hash the selected file and summarize optional top-level metadata plus the loaded rates. The bundled TOMLs will declare metadata explicitly; no date will be guessed from comments or source strings.

Alternative considered: automatically fetch current prices. Rejected because the tool is intentionally offline and provider/gateway billing is user-owned evidence.

### Generate, but never auto-fill, strict outcomes

`--outcomes-template-out` will write JSONL rows containing attempt and routing context with a placeholder disposition. Existing strict loading remains the authority once the operator replaces or removes placeholders. Duplicate identifiers and rate keys fail closed.

Alternative considered: classify final messages or treat follow-ups as semantic rework. Rejected because sampled histories show both false acceptance and false rework.

### Expose proxy risk through existing lifecycle receipts

Attempt annotations will add interaction and completion counts and small reason labels derived only from persisted activity kinds. They do not alter disposition.

## Risks / Trade-offs

- [Default summary breaks consumers expecting task-level arrays] → `--full-summary` restores those arrays and the README/skill will name the migration.
- [Wall time includes waiting and orchestration delay] → label it `rollout_wall_seconds` and do not call it compute time.
- [P90 is unstable for tiny route samples] → include observation counts and keep the skill's matched-task/sample-size warning.
- [Template placeholders are not valid strict outcomes] → name them `REPLACE_ME` and fail clearly if an unedited template is passed back.
- [Price metadata can be absent in private tables] → report null metadata while preserving path, hash, loaded keys, and per-rate sources.

## Migration Plan

1. Existing detailed JSONL consumers continue unchanged.
2. Summary consumers that need `groups` or `attempt_routes` add `--full-summary`.
3. Operators regenerate historical decision summaries so their price hashes and effective dates are recorded.
4. Rollback consists of passing `--full-summary`; no data migration or session mutation is required.
