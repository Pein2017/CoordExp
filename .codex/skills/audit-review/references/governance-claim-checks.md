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

- Require an explicit independent reviewer pass only when the governing contract names it as a gate; otherwise proportionate lead verification may close the decision.
- Missing required review remains unresolved. An unavailable optional adviser does not create a new blocker.
- Reuse accepted evidence for the same phase/target/risk across skills; switching audit names does not authorize another review.
- Style or optional maintainability notes are not blockers without a demonstrated impact on the declared decision or acceptance invariant.
- If the user asks for blocker-only review, do not drift into open-ended audit.
- `approve`/`promote` is an evidence recommendation, not implementation, launch, publication, or semantic-change authorization.

## Claim Validity

Report:

- `claim`: exact wording being supported or challenged.
- `scope`: tiny, smoke, val200, full-val, proxy, raw-text, coord-token, checkpoint, dataset slice, decode shape.
- `baseline`: matched comparator or reason missing.
- `evidence`: artifact paths, metric files, commands, or docs/spec handles.
- `counterevidence`: negative or ambiguous signals.
- `verdict`: supported, unsupported, partial, stale, or cannot interpret yet.
