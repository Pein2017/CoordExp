## Context

See `proposal.md` for motivation and scope. The current instruction loader preserves logical display paths and only deduplicates trimmed content within one directory. Its filesystem provider already resolves symlinks to an opaque target identity, while the host-filesystem path currently has no equivalent identity in the loaded metadata. Dynamic reconciliation therefore cannot recognize that a nested candidate is an alias of the previously loaded user-global file.

The Web profile is assembled from YAML loader expressions. DSH-home paths can already be resolved by the loader, but the local profile currently embeds this machine's DSH home and Serena executable/state paths. The profile-local reminder hook is not functioning in the current runtime and is redundant with the generic repeat-tool guard.

## Goals / Non-Goals

**Goals:**

- Prevent only the confirmed global-to-nested canonical alias from producing a second model-visible instruction message.
- Preserve logical scope, precedence, removal, retargeting, and native filesystem observation behavior.
- Make the Web profile relocatable and configurable without changing the intended default Serena state layout.
- Remove the ineffective Web-only reminder hook while retaining the generic repeated-tool guard.

**Non-Goals:**

- Do not deduplicate arbitrary equal-content files from different directories.
- Do not add path exclusions or change nested instruction discovery semantics in this change.
- Do not make Serena per-Web-session or solve multi-workspace server ownership.
- Do not alter the pre-existing `.codex/AGENTS.md` or `.codex/serena/serena_config.yml` edits.
- Do not change native `read`, `write`, `edit`, sandbox, or observation contracts.

## Decisions

### 1. Carry canonical target identity through instruction metadata

Add an optional internal identity to discovered, loaded, and probed candidates. The native `FileSystem` path uses the provider's resolved `FsTarget.targetKey`; the host path uses the resolved real path when available. The logical `absolutePath` and `displayPath` remain unchanged for scope keys, source attribution, and rendering.

Alternative considered: deduplicate all cross-directory equal-content files. Rejected because directory scope and precedence can be intentional even when prose happens to match.

### 2. Suppress only project candidates aliasing the user-global target

Baseline rendering and dynamic reconciliation compare a project candidate's identity with the current user-global candidate identity. A match suppresses the project alias; a missing identity falls back to current logical-path behavior. If the alias later resolves to a different target, normal project-scoped set/replace behavior resumes. Existing state and removal handling remain keyed by logical scope.

Alternative considered: canonicalize every logical path before constructing scope keys. Rejected because it would erase the observable distinction between directory scopes and could complicate removal notices and precedence.

### 3. Disable the profile-local Serena reminder, not the generic guard

Remove only the Web profile's `serena-remind-hook` insertion. The `repeat-tool-reminder` base row remains unchanged, and standalone Codex hook configuration is not edited. This is behavior-preserving for the observed current runtime, where the profile hook only produces infrastructure-refusal records.

Alternative considered: repair the hook sandbox in this change. Rejected because the failure belongs to the runtime sandbox backend and would expand scope beyond profile optimization.

### 4. Use loader expressions and environment overrides for profile paths

Use `dshHomePath('...')` for DSH-owned paths. Resolve the Serena command from `SERENA_COMMAND` with `serena` as the PATH-based default. Resolve the Serena state from `SERENA_HOME` or `DSH_SERENA_HOME`, falling back to the launch workspace's `.codex/serena` path so the current layout remains the default without embedding `/data/CoordExp`.

Alternative considered: move Serena state unconditionally under `$DSH_HOME`. Rejected because the current profile intentionally shares the `.codex/serena` state namespace with the configured Serena tooling.

## Risks / Trade-offs

- **[Risk]** A provider may expose an identity that is not stable across sessions. → Treat identity as an optimization hint only; when absent, preserve existing logical behavior, and keep logical scope/version state authoritative.
- **[Risk]** Suppressing a nested alias could hide a desired display-path distinction. → Apply suppression only for the exact user-global target match and preserve the global source attribution; a different target remains independently visible.
- **[Risk]** Removing the reminder hook loses its future reminder behavior. → The generic repeat guard remains active; the profile hook can be reintroduced after its sandbox backend is repaired.
- **[Risk]** PATH-based Serena resolution may fail in a launcher with a restricted PATH. → Support an explicit `SERENA_COMMAND` override and validate the optimized profile with the current launcher environment.
- **[Risk]** Dynamic Web workspaces can still bind one host Serena to the launch project. → Keep this known limitation explicit; workspace ownership is a separate follow-up.

## Migration Plan

1. Apply the source metadata/reconciliation change and its focused tests in `/data/deepseek-harness`.
2. Update the profile patch in `/data/CoordExp` and restart DSH Web so the new composition is loaded.
3. Verify the profile dump, a symlink-alias instruction scenario, and the targeted package test.
4. Roll back by restoring the prior profile row and source package if the checks reveal unexpected scope or state behavior; no user data or pre-existing dirty files are rewritten.
