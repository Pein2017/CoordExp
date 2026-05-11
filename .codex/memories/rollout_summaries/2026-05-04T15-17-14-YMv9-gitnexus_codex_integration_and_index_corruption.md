thread_id: 019df390-9c9e-7ca3-973c-72a3da67e1db
updated_at: 2026-05-05T06:26:10+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/04/rollout-2026-05-04T15-17-14-019df390-9c9e-7ca3-973c-72a3da67e1db.jsonl
cwd: /data/CoordExp
git_branch: main

# GitNexus was installed, adapted for Codex, then later partially rolled back after its index became unhealthy.

Rollout context: The user worked in `/data/CoordExp` and wanted GitNexus evaluated, installed, and integrated for Codex use alongside Serena. The user later asked for the AGENTS file to describe both tools, then for a commit-triggered GitNexus refresh hook, then reported that GitNexus was unhealthy because of WAL corruption, and finally asked to uninstall GitNexus and remove the hook.

## Task 1: Evaluate and install GitNexus for Codex use

Outcome: partial

Preference signals:
- The user said they cared more about whether GitNexus could give future Codex agents better indexing/help than about changing their workflow, indicating they value agent-facing structure and context quality over minimal workflow disruption.
- The user explicitly asked that installation, initialization, embeddings, and skills be placed under `$CODEX_HOME` rather than root paths, indicating a strong preference for repo-local / Codex-local state.
- The user clarified: “我用的是`codex`，不是`claude`！” and asked to replace `.claude` with `.codex`, indicating they wanted Codex-native naming and paths, not Claude-oriented defaults.

Key steps:
- Installed `gitnexus@1.6.3` globally after an initial install was interrupted and later re-run non-interactively.
- Determined the installation was pulling native deps like `onnxruntime-node` and `tree-sitter-*`, which explained the long install time.
- Found the default embedding path needed `NODE_USE_ENV_PROXY=1` because Node 22 `fetch` did not honor the existing proxy settings for Hugging Face downloads.
- Added repo-local Codex config entries so GitNexus MCP would be launched from `/data/CoordExp/.codex` with repo-local HF cache and repo-local registry state.
- Created Codex-local wrapper scripts under `/data/CoordExp/.codex/bin/` to launch MCP, register the repo, and refresh the index.
- Copied GitNexus’s bundled skills into `/data/CoordExp/.codex/skills` and rewrote references from `.claude` to `.codex` where applicable.

Failures and how to do differently:
- A direct `npm install -g gitnexus` first failed because `onnxruntime-node` tried to download a build list and hit `HTTP 302` / timeout behavior through the proxy path.
- The first attempt at refresh/indexing became unstable because multiple GitNexus MCP processes and the write-heavy indexing flow were running concurrently.
- GitNexus’s own `index --force` path did not reliably write a usable registry entry when `meta.json` was missing, so a Codex-local wrapper had to synthesize registry/meta state instead.
- GitNexus’s `.gitnexus` store later became corrupted (`WAL file is corrupted`), so future similar work should avoid leaving multiple long-lived MCP readers open while forcing re-indexes.

Reusable knowledge:
- On this machine, GitNexus can be installed, but embedding downloads need `NODE_USE_ENV_PROXY=1` for Node 22 to respect the proxy.
- GitNexus’s repo graph lives in `.gitnexus/`, while the global registry is in `~/.gitnexus/registry.json` unless `GITNEXUS_HOME` is overridden.
- GitNexus supports Codex MCP via `[mcp_servers.gitnexus]`, but its default skill paths still assume `.claude`; a Codex-local wrapper is needed if the user wants `.codex` naming consistently.

References:
- [1] `npm install -g gitnexus@1.6.3` succeeded after rerun with proxy-aware settings; install logs showed native deps and `added 260 packages in 36s`.
- [2] `NODE_USE_ENV_PROXY=1` fixed the Hugging Face fetch path for the embedding model; without it, `initEmbedder()` failed with `TypeError: fetch failed` / `ETIMEDOUT`.
- [3] GitNexus default embedding model was `Snowflake/snowflake-arctic-embed-xs`.
- [4] Codex-local wrapper scripts were created under `/data/CoordExp/.codex/bin/gitnexus-codex-mcp.sh`, `/data/CoordExp/.codex/bin/gitnexus-codex-refresh.sh`, and `/data/CoordExp/.codex/bin/gitnexus-codex-register.sh`.
- [5] The repo-local registry was written to `/data/CoordExp/.codex/gitnexus/registry.json`.
- [6] GitNexus CLI output eventually showed `No indexed repositories found` / `Checksum verification failed, the WAL file is corrupted` for queries, confirming the index had become unhealthy.

## Task 2: Compare GitNexus vs Serena and document the division of labor

Outcome: success

Preference signals:
- The user said they would keep both tools, so future agents should treat them as complementary and not force an either/or decision.
- The user asked for the comparison to be added to `AGENTS.md`, meaning they wanted the team workflow doc to encode the tool split for future Codex runs.

Key steps:
- Compared GitNexus and Serena using official docs and repo outputs.
- Concluded Serena is the symbol-level / IDE- or LSP-backed precision layer, while GitNexus is the graph/index/process layer.
- Updated `AGENTS.md` to say both tools remain available, Serena is the primary symbol-aware layer, GitNexus is the graph/impact layer, and the preferred order is GitNexus first for concept/process questions and Serena next for concrete symbol edits.

Failures and how to do differently:
- The comparison evolved over several rounds because the user first cared about whether GitNexus was stronger than Serena for Codex agents, then about documentation and workflow placement. Future agents should expect the user to want tool-role clarity translated into repo guidance, not just a verbal comparison.

Reusable knowledge:
- Serena is best treated as the “ground truth” for definitions, references, implementations, and rename-safe edits.
- GitNexus is best treated as a process / execution-flow / blast-radius map for unfamiliar or large codebases.
- For this repo, the AGENTS file explicitly now recommends: use GitNexus for concept/process/diff-impact exploration, then Serena for symbol-level edits.

References:
- [1] `AGENTS.md` was updated to add a short Serena vs GitNexus section describing the complementary roles and preferred order.
- [2] The user explicitly chose to keep both tools.

## Task 3: Add a commit-triggered GitNexus refresh hook for Codex

Outcome: partial

Preference signals:
- The user asked for a hook that automatically re-indexes when a git commit is detected, indicating they want the index freshness problem handled proactively by the environment.
- The user later reported that the hook message appeared during non-commit actions, which means they care about noisy false positives being removed.

Key steps:
- Confirmed Codex supports hooks in its own config and that a `PostToolUse` hook can be wired to `Bash`-type tool usage.
- Added a hook concept that was intended to detect commit completion and then trigger GitNexus refresh.
- Reduced the hook’s noisiness by removing the always-on status message when the user complained it fired during normal operations.

Failures and how to do differently:
- The hook was initially too broad: the user saw `PostToolUse - Checking whether GitNexus should refresh after git commit` even for non-commit operations.
- The fix was to make the hook quieter and narrower, but the larger lesson is that a Bash/PostToolUse hook is easy to overmatch unless it explicitly checks for successful `git commit` and a changed HEAD.
- Future agents should keep hook output silent unless the triggering condition is truly met.

Reusable knowledge:
- Codex hooks are the right place for this kind of post-action automation; the hook should be event-specific and commit-specific, not a general “Bash happened” notifier.
- If the hook only needs to act on successful commits, the gating logic should inspect the command text and actual HEAD change, not merely the presence of Bash activity.

References:
- [1] The user reported the noisy message exactly as: `PostToolUse - Checking whether GitNexus should refresh after git commit`.
- [2] The hook message was later removed so normal Bash operations would not show it.

## Task 4: Diagnose GitNexus health and handle corrupted index state

Outcome: fail

Preference signals:
- The user relied on another agent’s health report saying listing worked but context/impact failed with WAL corruption, indicating they expect the environment to be fixed or clearly rolled back when the index is unhealthy.
- The user then asked to uninstall GitNexus and remove the hook, showing they preferred a clean reset over continuing to troubleshoot a corrupted state in place.

Key steps:
- Verified that `gitnexus list` could still show the repo but `query`, `context`, and `impact` failed with `Checksum verification failed, the WAL file is corrupted`.
- Confirmed the active `.gitnexus` directory existed but the underlying query path was unhealthy.
- Observed that multiple GitNexus MCP processes were still running, which likely contributed to the corruption / instability.
- Determined the active index was not trustworthy and that further queries should not be treated as valid.

Failures and how to do differently:
- Rebuilding the index in-place while multiple MCP readers were alive and while commit-triggered refreshes were also in play was not stable.
- The repository ended up with both a new active `.gitnexus` and an older `.gitnexus.corrupt-*` backup, but the active store still exhibited WAL errors.
- Future agents should not treat `list_repos()` success as proof that `query/context/impact` are healthy; those can still fail if WAL or FTS layers are broken.

Reusable knowledge:
- `list_repos()` reads registry state and can succeed even when the underlying graph store is corrupt.
- `query/context/impact` are the real health checks for GitNexus; WAL corruption shows up there, not necessarily in repo listing.
- On this machine, corrupted WAL errors manifested as `Storage exception: Checksum verification failed, the WAL file is corrupted` and `FTS extension load failed`.

References:
- [1] Another agent reported: “GitNexus is unfortunately not healthy in this session: listing repos works, but context and impact fail with Storage exception: Checksum verification failed, the WAL file is corrupted.”
- [2] The current active `.gitnexus` and a backup `.gitnexus.corrupt-20260505T043904Z` both existed during diagnosis.
- [3] GitNexus query output repeatedly showed `WAL file is corrupted` and zero processes/results.

## Task 5: Uninstall GitNexus and remove the hook

Outcome: partial

Preference signals:
- The user explicitly asked: “帮我先卸载gitnexus，我将手动重新安装。同时删除这个`hook`,” indicating they wanted the environment reset to a clean manual-install starting point.

Key steps:
- Began removing GitNexus-related Codex config and AGENTS guidance.
- Attempted to remove running GitNexus processes, repo-local skills, the hook, and the global `gitnexus` package.
- Confirmed the global package was still present during cleanup (`gitnexus@1.6.3` was still installed at one point), and the repo-local GitNexus skills still existed before deletion.
- Started deleting local GitNexus artifacts and the hook path, but the cleanup was interrupted by process/session issues before a clean final verification was completed.

Failures and how to do differently:
- Cleanup was only partially completed because the session hit errors while trying to kill long-lived GitNexus processes, and the final removal pass was not fully validated.
- Because the session was interrupted, future agents should re-check the remaining presence of:
  - global `gitnexus`
  - `/data/CoordExp/.codex/bin/gitnexus-codex-*`
  - `/data/CoordExp/.codex/skills/gitnexus-*`
  - Codex hook entries in `/data/CoordExp/.codex/config.toml`
  - the `.gitnexus` store and any backup directories

Reusable knowledge:
- There were many lingering `gitnexus mcp` processes from earlier attempts, so a full uninstall/cleanup should verify process termination as well as file deletion.
- The repo-local GitNexus artifacts were spread across `/.codex/bin`, `/.codex/skills`, `/.codex/gitnexus`, and `/.gitnexus`; cleaning one layer was not enough.

References:
- [1] The user’s explicit request was to uninstall GitNexus and delete the hook before hand-reinstalling.
- [2] `npm ls -g --depth=0 gitnexus` showed the global package as present during the cleanup attempt.
- [3] Repo-local GitNexus skill directories were present before deletion: `gitnexus-gitnexus-cli`, `gitnexus-gitnexus-debugging`, `gitnexus-gitnexus-exploring`, `gitnexus-gitnexus-guide`, `gitnexus-gitnexus-impact-analysis`, `gitnexus-gitnexus-pr-review`, `gitnexus-gitnexus-refactoring`.

