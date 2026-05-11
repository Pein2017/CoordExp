thread_id: 019d85ff-c4a5-7961-a58c-90188bbeed87
updated_at: 2026-04-13T08:41:44+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/13/rollout-2026-04-13T08-40-25-019d85ff-c4a5-7961-a58c-90188bbeed87.jsonl
cwd: /data/CoordExp
git_branch: main

# Updated AGENTS.md to explicitly activate the repo-local `self-improving` workflow and point memory at `.self-improving/`.

Rollout context: The user asked to update `/data/CoordExp/AGENTS.md` so the `self-improving` skill is activated and files under `.self-improving` are exported. The repo already had a portable Codex skill at `.codex/skills/self-improving/SKILL.md` and existing repo-local memory files under `.self-improving/` (including `.self-improving/projects/coordexp.md`). The change was instruction-only; verification was via `git diff -- AGENTS.md`.

## Task 1: Update AGENTS.md for self-improving

Outcome: success

Preference signals:
- The user asked: "Please update my `AGENTS.md` to activate the `self-improving` skill and export the files under `.self-improving`" -> future similar repo-instruction edits should explicitly wire in the relevant skill and mention the repo-local memory root rather than assuming hidden conventions.
- The user’s wording "export the files under `.self-improving`" plus the existing repo layout suggests they want `.self-improving/` treated as visible/shared workspace state, not as private scratch or something buried inside the skill directory.

Key steps:
- Read the current `AGENTS.md` and the portable skill definition at `.codex/skills/self-improving/SKILL.md`.
- Confirmed the repo already had `.self-improving/` files, including `.self-improving/projects/coordexp.md` and `.self-improving/index.md`, so the change could reinforce an existing convention.
- Patched `AGENTS.md` with a new `## Self-Improving` section that:
  - activates the skill when the user explicitly names it or asks to remember reusable preferences/corrections/workflows,
  - keeps mutable memory under `.self-improving/`, not in `.codex/skills/self-improving/` or a machine-global home directory,
  - marks `.self-improving/` as exported repo-local project state intended to be shared/visible in the workspace.
- Verified the diff directly with `git diff -- AGENTS.md`.

Failures and how to do differently:
- No functional failure occurred.
- The main guardrail was to align the AGENTS wording with the existing repo-local `.self-improving/` layout rather than inventing a parallel convention; future changes in this area should check for an existing workspace-local memory root before adding new instructions.

Reusable knowledge:
- This repo already uses a Codex-portable skill at `.codex/skills/self-improving/SKILL.md` and a workspace-local memory root at `.self-improving/`.
- `.self-improving/index.md` showed the workspace had at least one project-scoped file under `projects/`, confirming the local memory root is active and meant to be used.
- For instruction-only edits, `git diff -- AGENTS.md` was sufficient verification; no runtime tests were necessary.

References:
- [1] Skill guidance read from `.codex/skills/self-improving/SKILL.md`: "Store mutable memory outside the skill directory in a workspace-local folder. Default local memory root: `.self-improving/`"
- [2] Existing repo-local memory evidence: `.self-improving/projects/coordexp.md` contained "Keep the self-improving memory for this workspace under `.self-improving/`, not inside the portable skill folder and not in a machine-global home-directory location."
- [3] `AGENTS.md` diff added:
  - `## Self-Improving`
  - activate skill on explicit naming / reusable preference / learned lessons / repeated mistakes,
  - keep mutable memory under `.self-improving/`,
  - treat `.self-improving/` as exported repo-local project state.

