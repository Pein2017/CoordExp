# DSH Local Source Migration Implementation Plan

**Goal:** Move the `web` profile from the installed npm CLI to `/data/deepseek-harness`, preserve the existing `/data/CoordExp/.dsh` state, enable the source Agent Teams Host/Web layers, and replace port 3080 only after a backup and alternate-port smoke test.

**Architecture:** Keep `DSH_HOME` as the sole durable state root. Build and launch the checked-out source CLI through its existing `pnpm dsh` entrypoint; add the two documented Agent Teams bundle layers to the existing `web` profile rather than editing source package code. Use an isolated timestamped backup and an alternate listener as the rollback boundary.

**Tech Stack:** Node 22, pnpm 11, TypeScript/tsx, Cordis profiles, DSH Web, tmux.

**Spec:** User request in the active task.

## Global Constraints

- `DSH_HOME` remains `/data/CoordExp/.dsh`.
- Do not reset, clean, overwrite, or delete unrelated files or Git state.
- Back up profile/session/credential/settings/model/catalog/plugin/patch/Serena/MCP state before mutation; do not print secrets.
- Compare installed npm DSH with the checked-out source HEAD; do not change source Git history.
- Run dependency verification and `pnpm run build` in `/data/deepseek-harness`.
- Smoke the source Web server on an alternate port before replacing 3080.
- Remove only the DSH npm runtime after successful replacement; keep npm/npx and their general caches.

### Task 1: Audit and backup

- [ ] Record versions, Git status/HEAD, active processes/listeners, profile composition, session and credential metadata, and the last 100 `tmux dsh` lines (with secrets redacted).
- [ ] Copy the exact current profile manifests/patches, session storage, credential files, settings, model/catalog files, plugin manifests/patches, Serena hook/MCP files, and a manifest/hash receipt into a timestamped backup below `/data/CoordExp/.dsh`.

### Task 2: Build and source profile composition

- [ ] Verify the checked-out dependency state with the locked pnpm workspace, then run `pnpm run build`.
- [ ] Add `packages/experimental/agent-team-profile` and `packages/experimental/agent-team-web-profile` to `/data/CoordExp/.dsh/profiles/web` using the source CLI/plugin mechanism, preserving existing bundle order and patches.
- [ ] Confirm the composed source tree with `DSH_HOME=/data/CoordExp/.dsh pnpm dsh --profile web --dump-config`, including Team Host/UI rows, existing models, Serena MCP, and hooks.

### Task 3: Alternate-port smoke and cutover

- [ ] Start `DSH_HOME=/data/CoordExp/.dsh pnpm dsh web --no-open --port <free-port>` in `tmux dsh-source-smoke` and verify HTTP readiness, process ownership, dump-config, Team tool rows, and profile/plugin resolution without sending a model request.
- [ ] Stop the old service only if one exists, then start `DSH_HOME=/data/CoordExp/.dsh pnpm dsh web --no-open --port 3080` in the existing `tmux dsh` session and verify HTTP readiness and the source process command line.
- [ ] Preserve rollback commands that stop the source process and restart the previously recorded npm CLI with the same `DSH_HOME` and port.

### Task 4: Remove only the DSH npm runtime and report

- [ ] After the 3080 source service is healthy, remove the installed `@deepseek-ai/dsh` package/symlink only; leave npm, npx, and `/root/.npm/_npx` intact.
- [ ] Re-run source version/readiness/config checks and report versions, changed profile entries, backup path, smoke/cutover evidence, rollback command, and any unresolved compatibility issues.

### Task 5: Runtime log diagnosis

- [ ] Classify the captured `tmux dsh` errors by source, reproducibility, and impact; separate stale Serena workspace/LSP warnings and OOM from DSH startup/network/model failures.
- [ ] Include the classification and practical consequence in the migration report without changing unrelated Serena/LSP configuration unless required for the requested cutover.
