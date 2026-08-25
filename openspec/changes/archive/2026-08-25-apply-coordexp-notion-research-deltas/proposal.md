## Why

The exhaustive Codex-session and research-document intake is complete, but its reviewed owner-level deltas have not yet been written back to Notion. The only prior blocker—the terminal suffix order—has now been resolved from the live CoordExp-Swift template and supervision path.

## What Changes

- Update only the existing Notion owners listed in `../extract-research-knowledge-from-codex-sessions/notion-update-plan.md` with compact, evidence-bounded deltas.
- Record the intentional terminal contract: canonical serialized suffix `<|im_end|>\n`, no separator between object rows, and loss supervision on `<|im_end|>` while the final newline is ignored.
- Preserve current scientific versus technical disposition, planned/executed/partial/retired boundaries, and current repository authority.
- Refetch every mutated page and record exact page IDs, markers, and outcomes in a local writeback receipt.
- Avoid raw transcript/document imports, duplicate pages, new databases, destructive archive operations, and production changes.

## Capabilities

### New Capabilities

None. This is an external documentation writeback; `.openspec.yaml` sets `skip_specs: true`.

### Modified Capabilities

None.

## Impact

- External writes are limited to the existing CoordExp Notion Research OS pages named in the reviewed update plan.
- Local writes are limited to this change directory, plus the separately authorized one-line explanatory comment in the active CoordExp-Swift renderer.
- No code behavior, configuration, model, experiment, artifact, session, `research/`, or legacy `progress/` content changes.
