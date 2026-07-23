# Pi Stage 0 Readiness Checkpoint

Captured: `2026-07-22T16:44:04Z`.

Source handles:

- `handoff/2026-07-22-pi-lightweight-worker-ablation.md`;
- `research/investigations/pi-lightweight-worker-ablation/`;
- `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/tooling/pi-coding-agent-0.81.1-install-receipt.md`;
- `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/fixtures-v1/manifest.md`;
- `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/sandbox/chroot-v1-receipt.md`.

## What changed

The user authorized continuation of the Pi capability evaluation and asked
that installation be delegated to the Luna subagent if Pi was absent. Pi was
absent from `PATH`; Luna installed the official
`@earendil-works/pi-coding-agent` package at exact version `0.81.1` into a
non-global output prefix and verified only version and help surfaces. No model
session or graphics-processing-unit work ran.

The Stage 0 research unit is now explicit. Four worker-visible tasks and four
hidden verifiers are frozen: artifact inventory, current inference source
routing, narrow claim audit, and deterministic aggregation. The source-routing
fixture is a Git archive of commit
`bb26ed613c5393255dc8784d0bbd1d122c3befd6`. Every verifier passes its hidden
positive-control answer; no worker answer has executed.

## Safety result and blocker

Bubblewrap and unshare cannot create mount or user namespaces in the current
container. A task-specific chroot fallback works under numeric user and group
`65534`: the worker can see one read-only fixture and its private writable
home, cannot write the fixture, and cannot see `/data/CoordExp`, host Pi or
Codex auth files, or graphics-processing-unit devices. Pi resource discovery
is disabled and only `read`, `grep`, `find`, and `ls` tools are enabled.

The chroot shares the host network namespace because the container lacks
`CAP_NET_ADMIN` and a usable network-namespace helper. The model cannot invoke
shell or network tools, but the kernel does not restrict the trusted Pi process
to provider domains. The isolated Pi home also has no authentication. A host
Pi auth file exists but was not read or copied because that requires explicit
secret-handling authority.

Execution is blocked until the user chooses fresh isolated OAuth versus
explicit reuse of the existing Pi auth file and either accepts the shared-
network residual risk or supplies a stronger container boundary. After auth,
the lead must inspect Pi's available model catalog and determine whether it
can match a native Codex worker's effective model and reasoning level. Native
Codex token and monetary-cost accounting is also not currently exposed by the
`agents` tool. Any run without both parity and common accounting is a
capability scout, not a causal Pi-versus-Codex harness result.

The Pi investigation must not interfere with the active constant-dose image-
breadth screen or use its graphics-processing units.
