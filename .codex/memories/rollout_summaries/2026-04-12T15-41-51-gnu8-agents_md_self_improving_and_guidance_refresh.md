thread_id: 019d825b-3ee9-7b80-965e-454853e71e18
updated_at: 2026-04-14T03:18:53+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/12/rollout-2026-04-12T15-41-51-019d825b-3ee9-7b80-965e-454853e71e18.jsonl
cwd: /data/CoordExp
git_branch: main

# The user asked for an AGENTS.md refresh focused on the first 27 lines, and the assistant updated the repo guidance to match the current CoordExp docs/spec structure.

Rollout context: The session centered on keeping repo guidance current with the codebase and on making the self-improving trigger easier to use. The final task asked to re-scan the codebase and update the outdated top section of AGENTS.md.

## Task 1: Make self-improving easier to trigger

Outcome: success

Preference signals:
- The user said: "Please help me update the `prompt` in `AGENTS.md` to make the skill to be triggered more easily. It's not your fault. But try to improve your global instruction." -> the user wants the agent to proactively trigger self-improving after repeated mistakes, workflow friction, or other reusable lessons, not only when explicitly named.
- The user later said: "Please help me update my `AGENTS.md` for first 27 lines since it's kind of out-dated. Please re-scan my codebase and update with a new global instruction guidance." -> the user expects AGENTS guidance to be kept current by re-scanning the repo rather than being rewritten from stale memory.

Key steps:
- The assistant inspected the current `AGENTS.md` self-improving section and the repo-local self-improving skill files under `.self-improving/`.
- The assistant patched `AGENTS.md` so `self-improving` can be activated proactively after repeated mistakes, relaunch loops, environment footguns, or obviously reusable workflows, while still keeping memory writes conservative.
- The updated file was re-read to verify the new trigger language was present.

Failures and how to do differently:
- No substantive failure here; the main takeaway is that the agent should not wait for an exact skill-name prompt once a reusable pattern is obvious.

Reusable knowledge:
- The repo-local self-improving memory lives under `.self-improving/`, and the skill is intended to be triggered when repeated mistakes or reusable workflow lessons are apparent.
- `AGENTS.md` is the right place to broaden the trigger language if the user wants more proactive memory capture in this workspace.

References:
- [1] `.codex/skills/self-improving/SKILL.md` — says to activate on explicit mention, asks to remember, asks what has been learned, or repeated mistakes/workflows worth capturing; do not learn from silence.
- [2] `.self-improving/memory.md`, `.self-improving/corrections.md`, `.self-improving/projects/coordexp.md` — confirmed repo-local memory layout.
- [3] `AGENTS.md` updated text: "Also activate it proactively when a session shows repeated mistakes, repeated rework, multi-turn debugging/relaunch loops, environment footguns, or a clearly reusable successful workflow that is likely to recur in this workspace."

## Task 2: Refresh the first 27 lines of AGENTS.md based on the current codebase

Outcome: success

Preference signals:
- The user said: "Please help me update my `AGENTS.md` for first 27 lines since it's kind of out-dated. Please re-scan my codebase and update with a new global instruction guidance." -> the user wants the top-level agent instructions to be re-derived from the current repo docs, not left as stale prose.
- The user specifically narrowed scope to the first 27 lines -> preserve that scope and avoid broad stylistic rewrites when they ask for a targeted refresh.

Key steps:
- The assistant re-scanned the canonical guidance layers: `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and the graphify report.
- The assistant patched only the top section of `AGENTS.md`, updating Mission / Defaults / Guardrails / Workflow / Repo Safety to reflect the current repo shape.
- The new guidance now explicitly emphasizes `openspec/specs/` precedence, `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for routing, offline-prepared JSONL as the default data surface, active Stage-1 and Stage-2 training surfaces, artifact/manifest completeness, and narrow verification first.

Failures and how to do differently:
- No code or test failure; this was a docs-only update.
- The assistant correctly avoided touching the later navigation/model sections because the user only requested the outdated first 27 lines.

Reusable knowledge:
- Current repo precedence is more nuanced than the old top-level AGENTS wording; `openspec/specs/` is normative above `docs/`, with `progress/` reserved for historical evidence.
- The repo currently treats Stage-1 baseline and Stage-2 rollout-aware training as active first-class surfaces, with offline-prepared JSONL and artifact/manifests as core operational concerns.

References:
- [1] `docs/PROJECT_CONTEXT.md` — canonical precedence and universal read order.
- [2] `docs/SYSTEM_OVERVIEW.md` — current end-to-end flow: offline conversion/resize/coord-tokenization -> JSONL contract -> dataset build -> training -> inference/eval -> artifacts.
- [3] `docs/IMPLEMENTATION_MAP.md` — active code entrypoints and tests for data, Stage-1, Stage-2, inference, evaluation, and run manifests.
- [4] `AGENTS.md:3-27` — updated Mission / Defaults / Guardrails / Workflow / Repo Safety section.
- [5] `graphify-out/GRAPH_REPORT.md` — confirmed the repository is large enough that graph structure adds value.
