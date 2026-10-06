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

## Experimental shared memory

The self-built cross-harness shared-memory MCP is separate from Codex automatic
memories. Its owner is `codex-tools/shared-memory-mcp/skills/shared-memory/SKILL.md`
(exposed by the shared skill path where supported). That skill owns caller-context,
candidate, approval and supersession mechanics. The user-wide contract owns the
capture/publication grant. Never substitute another memory store or the parent's
caller identity when that MCP is unavailable. Research records remain the source
of scientific meaning; memory links to them rather than maintaining a second
frontier or metric ledger.
