## 1. Source and behavior

- [x] 1.1 Establish independent local upstream-derived checkout and confirm HEAD, dirty ownership, and shared runtime profile.
- [x] 1.2 Add exact task-continuation replay and strict persisted validation; verify RED/GREEN, legacy checkpoints, and arbitrary-user rejection.
- [x] 1.3 Exercise real SDK automatic/manual/settled/cancel/new-input behavior and verify no duplicate completed tool or extra turn.

## 2. Local release and adoption

- [x] 2.1 Add local version/release ownership documentation and immutable local packaging; verify artifact contents, loader, and repository checks/tests.
- [ ] 2.2 Qualify a bounded hosted continuation task without new prompt; record observed behavior and limits.
- [x] 2.3 Replace the npm plugin with the verified local release in the shared profile; verify CLI/Web loading, preserve active sessions, and record local release identity.

## Qualification evidence

Source owner: `/data/CoordExp/codex-tools/pi-extensions`, local `main` commit
`033773f06e69461465b9aefee33335707462170d`. Release: `0.55.0-coordexp.1`.
The release receipt is `.local/releases/0.55.0-coordexp.1/release.json` in that
repository; adoption/backups are `.local/adoption/20261006-coordexp-1/`.

The package's 377 tests pass. Six continuation regressions pass with actual Pi
1.0.3 against source, generated output and the released package. The release's
lazy settings menu loads, and cancelled menus start no compaction. Dirty-source
and existing-version releases are rejected. `npm run check` passes. Four unrelated
full-suite failures reproduce against unmodified upstream source; their logs and
the resolved old overflow expectation are recorded at the module's documentation.

The shared profile configures the local release once; actual resource loading
deduplicates the physically identical user/project scope. CLI listing and Web
plugin diagnostics identify the selected release with no diagnostics. The old
npm files were removed. Compaction protocol and other settings were preserved.
The Web service was kept running; no active task was reloaded or prompted.

**HOLD on 2.2:** The one hosted attempt timed out on its first normal request,
before any tool or compaction. Its receipt is
`.local/qualification/continuation-9UkwKX/result.json`. No retry was made. The
deterministic checks establish lifecycle/persistence behavior, not live model
continuation through an opaque checkpoint. This change remains unarchived.

## Historical-scope closeout — 2026-10-09

The user approved closing stale OpenSpec acceptance debt rather than retrying the original hosted request. Item 2.2 remains **unchecked**: the native model/provider continuation was never observed in this scope. The local Pi 1.0.3 SDK, release and shared-profile adoption evidence above remains valid only at its tested boundary; later Pi local-compaction scheduling and releases have independent records. Archive this historical change without implying hosted success. Any future live continuation qualification should be separately selected against the then-current installed model/provider and deployment, with a fresh bounded cost/stop policy.

