## 1. Canonical instruction projection

- [x] 1.1 Carry an optional canonical target identity from host and provider-backed instruction discovery through baseline and dynamic metadata, while preserving logical scope and display paths; verify the agent-instructions package typecheck passes.
- [x] 1.2 Suppress only project candidates whose canonical identity matches the current user-global candidate, retain logical-path behavior when identity is unavailable, and restore normal projection when an alias retargets; verify focused reconciliation tests cover set, removal, and retarget behavior.
- [x] 1.3 Add filesystem-backed regression coverage for a user-global symlink reached as a nested candidate and for different directories containing equal bytes at different targets; verify the new tests fail on the old behavior and pass after the implementation.

## 2. Serena Web profile cleanup

- [x] 2.1 Remove the profile-local `serena-remind-hook` insertion while leaving the base `repeat-tool-reminder` and standalone Codex hook configuration unchanged; verify the effective profile dump contains `mcp-serena` but no `serena-remind-hook`.
- [x] 2.2 Replace profile-local absolute DSH paths with `dshHomePath(...)`, resolve the Serena executable through an environment override with a PATH-based default, and make the Serena state directory environment-overridable with the current workspace-relative `.codex/serena` default; verify the loader accepts the expressions and the current profile resolves to the existing paths.
- [x] 2.3 Make the retained hook JSON's command paths runtime-resolved for explicit compatibility use without enabling the Web hook; verify it remains valid JSON and contains no `/data/CoordExp` or `/root/.local/bin/serena` literals.

## 3. Verification and residue checks

- [x] 3.1 Run the targeted agent-instructions regression suite, host build, and changed-file lint; verify the build, lint, and all 155 reproducible tests pass, with the pre-existing mode-000 readability test excluded because this root-run environment can still read it.
- [x] 3.2 Run OpenSpec validation and inspect the final diff/status; verify the new change artifacts and intended DSH/profile changes are present, while pre-existing `.codex/AGENTS.md` and `.codex/serena/serena_config.yml` edits (and tooling-generated DSH `.codex/` state) remain untouched.
