## Why

Persistent research sessions already run in parallel, but peer coordination requires long identity-heavy commands and reports can become obsolete before consumption. Separate collaboration guidance from transport, make short source-linked communication convenient, and give the existing Python helper maintainable boundaries without adding another agent runtime.

## What Changes

- Refine the existing lead-worker skill and task/report examples: lead retains scientific decisions and acceptance; workers own bounded outcomes and may directly coordinate factual dependencies with named peers.
- Add explicitly selected participant aliases and sender/recipient routing for independent Pi Web sessions. Resolve aliases from a small transport-owned block in the existing task state, then validate exact session IDs and physical working directories; never infer a target from display names.
- Add concise update/decision/correction briefs with optional report path, section, revision and supersession metadata. Send references, not report contents; separate machine routing fields from human-facing summaries.
- Refactor `pi_worker_turn.py` into a thin compatibility entrypoint and a small standard-library package separating CLI/orchestration, routing/message values, and HTTP/receipt concerns. Retain the existing create, inspect, model lookup, queue/steering and recovery behavior.
- Add focused failure-oriented tests and CPU loopback caller-to-receipt qualification. Keep existing commands and generated return routes usable.
- Defer broadcast/fanout, shared editable boards, passive inbox, schedulers, daemon processes, cross-harness transport, automatic message withdrawal and read/acceptance tracking.

## Capabilities

### New Capabilities

- `peer-session-coordination`: Explicit independent-session routing, short source-linked briefs, preserved delivery/recovery semantics, and collaboration guidance that does not confer authority.

### Modified Capabilities

None. Existing main capability inventory has no persistent peer-transport specification. Internal module extraction does not itself add normative requirements.

## Impact

- Main owns this change and the maintained `.codex/skills/lead-worker/` implementation and references. The shared native-team guide remains the owner of generic outcome-package and acceptance rules; do not duplicate them.
- Expected code surfaces: `.codex/skills/lead-worker/scripts/pi_worker_turn.py`, its focused tests, and a new sibling `peer_transport/` Python package. Codex `worker_turn.py` behavior is unchanged.
- Expected documentation surfaces: `lead-worker/SKILL.md`, `references/pi-transport.md`, and a compact coordination examples reference. Research task states remain owned by their existing research units; implementation does not modify active research states or sessions.
- No third-party dependency, Pi Web/Core source change, plugin install, runtime restart, model/GPU execution or Git publication is required.
- Compatibility: retain the current script path, legacy pair flags, explicit model/effort checks, session/cwd guards, exclusive receipts, and unknown-delivery handling. Queue still starts an idle recipient; this is not passive notification.
