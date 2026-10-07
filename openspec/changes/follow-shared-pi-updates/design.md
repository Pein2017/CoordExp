## Context

See proposal.md for motivation. The official release layout is already atomic and versioned. Web externalizes the four Pi packages and uses an in-process SDK. Next 16.3.6's dev CLI restarts its worker on exit code 77 under the same CLI PID and port; a real isolated smoke has exercised resolver replacement and unchanged SessionManager history.

## Goals / Non-Goals

**Goals:** One official installation owner independent of profile; safe automatic adoption in the local dev service; readable/indexed source without core modifications.

**Non-Goals:** Hot-swapping SDK objects, upgrading an already running interactive CLI, qualifying unknown future API compatibility, changing authentication/Tunnel policy, modifying upstream Pi source, or issuing paid model calls.

## Decisions

1. `codex-tools/pi-core` is a normal unmodified official Git checkout at the runtime tag; its `.local/install` and `.local/bin/pi` hold the existing managed distribution. `.git/info/exclude` keeps runtime/index artifacts private without source changes. `.pi` remains the profile. Moving all repositories into another stack directory would add path migrations without improving runtime ownership; keep Web/extensions in place.
2. The root CLI wrapper and Web launcher explicitly select this install. Web's binder reads `PI_MANAGED_INSTALL_ROOT`, validates all four SDK package versions, then atomically replaces its scope symlink. There is no legacy profile/install fallback after migration.
3. A process-global Web update coordinator polls the selected version every two seconds, only when enabled by the local dev launcher. It waits for pending starts, wrapper commands, SDK prompt/compaction/bash work, queued inputs, closing wrappers and open terminal shells. It synchronously closes new session/command/terminal admission before awaiting idle session shutdown. It then runs the existing binder and exits the Next worker with the installed restart constant. HMR updates callbacks without duplicating the observer. A synchronous CLI update hook that waits for Web idle could deadlock when invoked from a Web bash tool; avoid it. Another daemon/supervisor would duplicate Next's existing lifecycle.
4. Validation or activation failures are logged without request bodies/credentials; the coordinator retains the old binding and reopens admission. Shutdown failures prevent adoption. Browser-selected idle sessions are resumable and do not themselves block adoption. Open terminal shells defer adoption, even when not producing output.
5. Each active repository under `codex-tools` owns its `.codegraph` index. The user selected this layout after identifying that a Pi-only parent index could catch queries from other repositories without child indexes. Initialize child indexes and remove the newly created parent index/config. Queries explicitly choose the repository owner; cross-module exploration uses separate queries. The official source reference is initially pinned to 1.0.3; future runtime updates do not silently switch reference source. Document explicit tag checkout plus index sync when the user wants reference/runtime alignment.

## Risks / Trade-offs

- [Next's dev restart protocol can change] → import its actual restart constant and retain a real isolated integration test; this mechanism is enabled only in the managed local dev launcher.
- [Update races with a new prompt or terminal] → one process-global synchronous admission gate, exercised at actual consumers; include preflight and starts in the busy predicate.
- [Extension shutdown hangs or fails] → no worker exit or binding switch before settled shutdown; expose the failure in logs instead of forcing termination.
- [Future core APIs break Web/plugin] → official update's CLI smoke does not imply application compatibility; log loaded/selected versions and let explicit user adaptation address the failure.
- [Reference tag differs from a future runtime] → document the separate selected runtime version and reference tag rather than imply automatic source synchronization.

## Migration Plan

Preserve the existing version, marker, profile and release artifact. Verify Web is idle and exact service PID, gracefully stop only its owned process, move the managed install/launcher, update explicit callers, activate shared SDK and start one local service. Verify CLI version, Web SDK path, profile/package configuration and HTTP consumers. Retain historical recovery receipts; they are not active fallback paths. On a concrete migration failure, repair the named paths using the preserved installation rather than reinstalling or changing session data.
