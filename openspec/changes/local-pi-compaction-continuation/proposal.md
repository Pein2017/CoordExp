## Why

Successful opaque compaction in Pi can leave the next active-task response at `OK`; a fresh user continuation restores useful work. The user selected a locally maintained fork of `@narumitw/pi-codex-compact`, with local releases shared by CLI and Web-Pi.

## What Changes

- Maintain the upstream monorepo under `codex-tools/pi-extensions` and modify only the compaction package plus local release tooling.
- Preserve task continuation after active automatic context-management compaction using a bounded, deterministic replay item; leave manual and settled compaction idle.
- Preserve existing checkpoints, provider authentication, cancellation, and newer user instructions.
- Build versioned local release artifacts and replace the shared profile's npm plugin with the verified local release.

## Capabilities

### New Capabilities

- `pi-local-compaction`: Local release ownership and task-sensitive remote compaction continuation for the shared Pi runtime.

### Modified Capabilities

None.

## Impact

Independent repository `/data/CoordExp/codex-tools/pi-extensions`, its compaction package and release helper, and `/data/CoordExp/.pi/settings.json`. No public fork, npm publication, upstream PR, provider migration, SDK modification, or interruption of active user sessions.
