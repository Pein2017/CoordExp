## 1. Freeze observable behavior

- [x] 1.1 Add interface-level tests that fail on the current default/full summary shape and route-pair duration, measured-token, billable-token, and cost distributions.
- [x] 1.2 Add tests for pricing snapshot receipts, duplicate price/outcome rejection, outcomes-template rows, and structural proxy ambiguity annotations.

## 2. Implement the decision summary

- [x] 2.1 Make the default summary compact, preserve legacy aggregates behind `--full-summary`, and add route-pair distributions using the existing enriched records.
- [x] 2.2 Add deterministic local price receipts and fail-closed price-table validation without changing per-segment billing semantics.
- [x] 2.3 Add strict outcomes-template output, fail-closed outcome validation, structural ambiguity fields, and repair affected static-type diagnostics.

## 3. Update operator contracts

- [x] 3.1 Update the package README, bundled price examples, skill instructions, and report-field reference for compact/full migration, decision metrics, pricing receipts, and strict outcomes labeling.

## 4. Verify the accepted surface

- [x] 4.1 Run the full unit suite, Ruff, compile checks, and static diagnostics for the touched package.
- [x] 4.2 Run a real-session wrapper smoke and a 14-day audit to verify read-only operation, zero parse regressions, pricing coverage, decision fields, and reduced default-summary size.
- [x] 4.3 Validate the OpenSpec change strictly, inspect the frozen diff for scope/residue, and record only acceptance conditions actually demonstrated.
