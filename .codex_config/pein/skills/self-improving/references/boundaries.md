# Self-Improving Boundaries

Use this skill to improve execution quality, not to accumulate sensitive personal data.

## Never Store

- credentials, tokens, API keys, or access details
- financial, medical, biometric, or government-identifying data
- third-party personal details
- home addresses, routines, or sensitive location patterns
- manipulative notes such as what makes the user comply faster

## Safe To Store

- explicit workflow preferences
- explicit response-style preferences
- repo-specific conventions
- reusable lessons from corrections
- scoped project overrides

If the memory root is git-tracked, apply an extra filter:

- store only content you would be comfortable committing into the repository history
- avoid machine-specific noise unless it is truly important for cross-machine continuity
- prefer project conventions over personal scratch notes

## Consent Rules

- explicit correction: okay to log as a correction
- explicit "always/never/prefer" statement: okay to store in the matching namespace
- repeated pattern without explicit confirmation: keep tentative unless clearly durable
- inferred preference from silence: never store

## Deletion Rules

- never delete by default
- archive or supersede when preferences change
- if the user asks to forget something, remove it from the relevant files and confirm what changed
