# Governance And Claim Checks

Use this reference when an audit touches stable contracts, OpenSpec state, benchmark claims, or reviewer closure.

## Audit Mode Detail

- `change/spec audit`: check implementation, docs, stable specs, active change deltas, tests, and migration/default behavior together.
- `artifact/run audit`: start from the artifact root and label scope before interpreting metrics.
- `launch gate`: return `promote`, `hold`, `rerun gate`, or `needs user decision` with the smallest proof that would change the verdict.
- `claim validity audit`: name the exact claim, comparator, evidence, counterevidence, missing baseline, and metric scope.
- `implementation-vs-contract audit`: prove code/config/runtime/artifacts match the current documented contract.

## OpenSpec Governance

- Check active-change state when it affects the answer.
- Stable specs are current contracts; active changes are scoped proposals unless explicitly in scope.
- Validate changed specs strictly before closure.
- When behavior, schema, artifact names, metric semantics, loss semantics, or entrypoints move, verify docs/specs were updated in the same change.
- Treat stale, incomplete, or deprecated changes as residual risk instead of current behavior.

## Review Closure

- Explicit reviewer pass is required to close a gate.
- Timeout, disconnection, or missing reviewer output is unresolved.
- Style-only notes are not blockers unless they hide contract, correctness, eval-validity, or maintainability risk.
- If the user asks for blocker-only review, do not drift into open-ended audit.

## Claim Validity

Report:

- `claim`: exact wording being supported or challenged.
- `scope`: tiny, smoke, val200, full-val, proxy, raw-text, coord-token, checkpoint, dataset slice, decode shape.
- `baseline`: matched comparator or reason missing.
- `evidence`: artifact paths, metric files, commands, or docs/spec handles.
- `counterevidence`: negative or ambiguous signals.
- `verdict`: supported, unsupported, partial, stale, or cannot interpret yet.
