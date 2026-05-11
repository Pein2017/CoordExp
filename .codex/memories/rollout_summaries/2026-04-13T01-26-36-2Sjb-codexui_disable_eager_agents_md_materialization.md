thread_id: 019d8472-9b77-78a3-94c1-5f6f07f7adfd
updated_at: 2026-04-13T01:32:05+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/13/rollout-2026-04-13T01-26-36-019d8472-9b77-78a3-94c1-5f6f07f7adfd.jsonl
cwd: /data/CoordExp
git_branch: main

# Disabled eager `AGENTS.md` materialization in codexUI startup sync after tracing the actual code path

Rollout context: The user was running `bash mcp/codexUI/scripts/codexapp-current-dir.sh` from `/data/CoordExp` and suspected it was creating a symlink involving `AGENTS.md` under `./codex/`. They explicitly asked to use Serena MCP to explore the `codexUI` project and disable the symlink creation unless necessary.

## Task 1: Find and stop `AGENTS.md` symlink/materialization behavior in codexUI

Outcome: partial

Preference signals:
- The user asked to “use serena MCP to explore my `codexUI` project,” which suggests that for similar repo-exploration tasks the agent should prefer Serena’s symbol/pattern tools first and avoid blind full-file reads.
- The user asked to “disable the symlink creation unless it’s necessary,” which indicates they care about avoiding eager filesystem mutation during startup and prefer a narrow repair-only behavior rather than unconditional normalization.

Key steps:
- Traced the launcher script `mcp/codexUI/scripts/codexapp-current-dir.sh` and confirmed it only sets `CODEX_HOME=$PWD/.codex` and then execs the built app; it does not itself create the `AGENTS.md` link.
- Searched `mcp/codexUI` for `AGENTS.md`, `ln -s`, `symlink`, and related strings.
- Identified the relevant code path in `mcp/codexUI/src/server/skillsRoutes.ts`: startup sync always called `ensureCodexAgentsFilesArePlainFiles()`.
- Read the startup helper and found it previously deleted/recreated both `.codex/AGENTS.md` and `.codex/skills/AGENTS.md` unconditionally, even when neither existed.
- Patched the helper so it only repairs paths that already exist as a symlink or other non-file hazard, and returns early when both paths are missing or already plain files.
- Verified the current on-disk `.codex/AGENTS.md` and `.codex/skills/AGENTS.md` were plain files, not symlinks.

Failures and how to do differently:
- The first TypeScript verification attempt used `npm --prefix mcp/codexUI exec vue-tsc --noEmit`, which printed the TypeScript CLI help instead of running the check because `npm exec` did not forward the flags as expected in this environment.
- A graph refresh command required by the repository workflow failed because the Python module `graphify` is not installed in the current shell (`ModuleNotFoundError: No module named 'graphify'`).
- The better direct verification path was `./node_modules/.bin/vue-tsc --noEmit` run from `mcp/codexUI`, which completed successfully after the patch.

Reusable knowledge:
- In `mcp/codexUI`, the launcher script is not the source of `AGENTS.md` link behavior; the real startup mutation lives in `src/server/skillsRoutes.ts`.
- `ensureCodexAgentsFilesArePlainFiles()` is the startup hook that controls whether `.codex/AGENTS.md` and `.codex/skills/AGENTS.md` are rewritten.
- The patched logic now uses `lstat` to distinguish `missing` / `file` / `unsafe` and only rewrites when a non-file hazard is present.
- For this repo, direct tool invocation from the project directory can be more reliable than `npm exec` for `vue-tsc` verification.

References:
- [1] Launcher script: `mcp/codexUI/scripts/codexapp-current-dir.sh` — sets `CODEX_HOME="${CODEX_HOME:-${launch_dir}/.codex}"` and then `exec node ...`.
- [2] Relevant server startup hook: `mcp/codexUI/src/server/skillsRoutes.ts#L999-L1043`.
- [3] Patched logic (condensed):
  - `getPathState(targetPath)` uses `lstat` and returns `'missing' | 'file' | 'unsafe'`.
  - `needsRepair = codexAgentsState === 'unsafe' || skillsAgentsState === 'unsafe'`.
  - `if (!needsRepair) return` prevents eager file creation.
  - Only the unsafe path(s) are removed and rewritten.
- [4] On-disk state check: `ls -l /data/CoordExp/.codex/AGENTS.md /data/CoordExp/.codex/skills/AGENTS.md` returned two regular files, size 0.
- [5] Verification detail: `./node_modules/.bin/vue-tsc --noEmit` succeeded when run from `mcp/codexUI`.
- [6] Verification failure detail: `python3 -c "from graphify.watch import _rebuild_code; ..."` failed with `ModuleNotFoundError: No module named 'graphify'`.

