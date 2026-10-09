# Agent integration and task routing

This page owns local integration facts, not another constitution or mandatory
reading stack. Start from the task's actual source, command, artifact or question.

## Instruction ownership

`/data/CoordExp/.codex/AGENTS.md` is the single user-wide contract. Project AGENTS
add only responsibilities, local risks and conditional owner pointers. Skill
entrypoints expose reusable procedures; detailed references load only when needed.
Current source/typed configs/tests own behavior, stable OpenSpec specs own
compatibility, and canonical research owns scientific evidence and interpretation.

For substantial change management use local OpenSpec. Do not duplicate its plans
in agent handoffs or memory. Documentation ownership and recovery follow [the shared policy](documentation.md).

## Loading and skill maintenance

Use actual harness instruction-source output to verify loading; a Markdown link
is not automatic injection. The existing WebCodex Runner `instructions.files`
facility is the intended global loader. Verify the constitution appears exactly
once in each Project after any configuration change; do not copy it into worktrees.
Native harness aliases may refer to the same file and are not additional owners.

Shared local skills live at `/data/CoordExp/.codex/skills`. Their descriptions
control discovery, not permission. Use the installed skill-creator for edits;
preserve vendor/system/synced package ownership. A rejected symlink or a collection
folder without its own SKILL.md is a discovery integration issue, not a reason to
copy packages, invent definitions, or weaken path checks. Remove obsolete local
skills and their live references directly; Git history is recovery.

## Local execution

Use bare `python`/`python3` in the configured shell: startup preserves a selected
interpreter or wraps it with `conda run --no-capture-output -n ms`. Absolute
interpreter paths bypass that wrapper. Verify the real worker interpreter for
version-sensitive work; do not assume a structured process uses shell functions.
The selected checkout's dependency declarations and qualification consumer own
runtime admission, not an old receipt or another worktree's supported version.

For CoordExp design trade-offs, preserve this order: accuracy/precision, training
and system efficiency, simplicity/low implementation overhead, then scalability
and future extension. This does not authorize changing a frozen scientific metric.

## Local Pi customization

Pi Web keeps its built-in subagent engine. Local profiles live in
`/data/CoordExp/.pi/agents/`; `general-purpose`, `explore` and `plan` are disabled
in that directory's `settings.json`, not removed from vendor source. The enabled
IDs `coordexp-scout`, `coordexp-builder`, `coordexp-research` and `coordexp-runner`
identify source, implementation, research and runtime **resource bundles**, not
fixed worker roles. Each selects CodeGraph exploration and Shared Memory
`context`, `search` and `read`, not memory writes or nested `Agent` tools.
Extension selection remains limited to RTK for builder/runner; scout/research
load no extensions. The Notion profile is unchanged.
Their tools, skills, extensions and MCP selections stay configurable; model,
thinking, turn budget, parent-context inheritance, background mode and isolation
are explicit task-level choices of the main/lead. The profile files contain no
execution defaults or fixed report word limits. No legacy alias profiles are
kept: saved child metadata retains historical IDs and frozen resource snapshots,
so resume does not require the renamed profile or gain its newly selected
`context` operation. New children use the current definitions; already-live
catalogs/wrappers may remain stale until reloaded or naturally replaced. Disk
selection is not evidence of live tool availability. See the single
[invocation policy](../.codex/AGENTS.md#model-routing) and
[Pi Web's interface note](../codex-tools/pi-web/docs/agents/subagents.md).
Temporary roles and output requirements belong in the assignment, which grants
no additional research launch, process-stop or publication authority.

CodeGraph queries must name the indexed source owner explicitly:

- CoordExp source: `/data/CoordExp`.
- Pi Web source: `/data/CoordExp/codex-tools/pi-web`.
- Official Pi reference: `/data/CoordExp/codex-tools/pi-core` (tag `v1.0.3`).
- Codex source: `/data/CoordExp/external/harness/codex`.

The older `external/harness/pi` is a distinct versioned checkout, not the installed
SDK. `projectPath` selects the nearest index, not a query subtree. Do not use the
`node_modules/@earendil-works` symlink as a runtime-source index; ignored compiled
vendor paths can leave an examples-only index. A result from an unrelated file is
not evidence merely because the query returned successfully.
See [Pi source navigation](../codex-tools/pi-web/docs/pi-source-navigation.md)
for index commands and reference/runtime version alignment.

The Pi RTK adapter is `/data/CoordExp/.pi/extensions/coordexp-rtk.ts`; command
rewrite semantics remain at `.codex/hooks/rtk-pretooluse.py`. Compression is for
human engineering diagnostics, never an exact-output or scientific artifact
contract. Use `RTK_HOOK_DISABLE=1` for a raw command. Do not register a second RTK
extension alongside the auto-discovered one.

For observation, keep raw journals/logs at their existing owner and return bounded
projections plus source pointers. A subagent's terminal tool details may carry a
read-time lifecycle-scoped usage/model/artifact projection; persisted assistant
responses are not exact SDK turn counts, and absent provider routing or termination
metadata must stay unknown. Durable external jobs use the existing
[native-team job handoff](../.codex/skills/native-agent-team-guide/SKILL.md#durable-job-handoff),
not a second scheduler or run ledger. Agent interruption is not job cancellation.

## Experimental shared memory

The self-built cross-harness shared-memory MCP is separate from Codex automatic
memories. Its owner is `codex-tools/shared-memory-mcp/skills/shared-memory/SKILL.md`
(exposed by the shared skill path where supported). That skill owns caller-context,
candidate, approval and supersession mechanics. The user-wide contract owns the
capture/publication grant. Never substitute another memory store or the parent's
caller identity when that MCP is unavailable. Research records remain the source
of scientific meaning; memory links to them rather than maintaining a second
frontier or metric ledger.
