## Why

The adaptive stop-signal-damping experiment has proven harmful in practice. It
pushes rollout behavior toward duplicate-heavy object enumeration, especially in
dense or crowded scenes, and it is no longer considered a safe or supported
training surface.

## What Changes

- Remove the live `token_ce.config.stop_signal_damping` configuration surface by
  making authored presence fail fast with actionable guidance to delete it.
- Remove the runtime stop-signal weighting path and all related objective atom
  / metric emission from active training code.
- Scrub checked-in configs and stable training docs that still advertise the
  experiment.
- Archive the original `adaptive-stop-signal-damping` implementation change as
  historical context without syncing its delta specs into `openspec/specs/`.

## Impact

- Affected code is centered in the teacher-forcing token CE config/runtime path
  plus Stage-2 objective/metric projection.
- Affected authored surfaces include Stage-2 canonical YAML profiles and the
  stable training docs.
- Stable main specs remain unchanged because the experiment was never promoted
  into `openspec/specs/`.
