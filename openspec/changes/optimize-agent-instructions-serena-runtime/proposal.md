## Why

The current DSH Web profile carries avoidable runtime overhead and can project the same instruction file twice: `$DSH_HOME/AGENTS.md` is a symlink to the workspace's `.codex/AGENTS.md`, but discovery tracks the two logical paths separately. The profile also installs a Serena reminder hook that currently fails on every matching call, while its absolute paths make the profile dependent on this machine and launch layout.

This change targets those three concrete issues without weakening native filesystem observation, broadening content deduplication across intentional directory scopes, or overwriting pre-existing workspace changes.

## What Changes

- Deduplicate only the user-global instruction file and a project candidate that resolve to the same canonical filesystem target, including dynamic nested projections; preserve distinct same-content files from unrelated directories.
- Add regression coverage for symlink-target deduplication and for retaining ordinary cross-directory instruction scopes.
- Remove the nonfunctional DSH Serena reminder hook from the Web profile by default; retain the broader repeat-tool guard and leave the standalone Codex hook configuration untouched.
- Replace profile-local absolute DSH paths with loader-resolved home paths and make the Serena executable/state location explicitly configurable while preserving the current workspace-relative `.codex/serena` default.
- Preserve existing dirty `.codex/AGENTS.md` and `.codex/serena/serena_config.yml` changes; do not rewrite or clean them.

## Capabilities

### New Capabilities

- `agent-instruction-scope-dedup`: Canonical-target-aware projection of user-global and nested instruction aliases without collapsing independent directory scopes.
- `serena-profile-runtime`: Portable, single-owner Serena Web profile wiring with no ineffective duplicate reminder hook.

### Modified Capabilities

<!-- No existing CoordExp stable capability owns DSH Web instruction or Serena profile behavior. -->

## Impact

- DSH source: `packages/context/agent-instructions` in `/data/deepseek-harness`, including tests and the exported file metadata shape used internally by the package.
- CoordExp profile configuration: `/data/CoordExp/.dsh/profiles/web/cordis.patch.yml`; the inactive hook JSON remains available for explicit compatibility use.
- Model-visible history becomes smaller when a global instruction file is reached again through its canonical nested alias; all native `read/write/edit` observation and refresh semantics remain intact.
- Serena startup remains host-level and does not become per-Web-session; multi-workspace isolation and project-state basename collisions remain separate follow-up work.
- No new runtime dependency is required and no existing CoordExp-Swift stable spec is changed.
