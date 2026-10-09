## Why

Pi's default post-run threshold check compacts a normally completed answer and delays prompt settlement. The user wants the Codex default timing: preserve the completed answer and compact before the next model request.

## What Changes

- Defer threshold maintenance after a completed response in the locally maintained compaction plugin.
- Retain Pi's next-request compaction, manual compaction, active-task continuation and overflow recovery.
- Publish and adopt an immutable local plugin release for the shared CLI/Web profile while sessions are idle.

## Capabilities

### New Capabilities

- `pi-compaction-scheduling`: Local plugin scheduling for completed responses and the next model request.

### Modified Capabilities

None.

## Impact

`codex-tools/pi-extensions/packages/pi-codex-compact`, its real SDK consumer tests and local release metadata; the shared profile's selected package and idle Web extension reload. Pi core source and compression protocol remain unchanged.
