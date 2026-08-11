# CoordExp Risk-First Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the installed Superpowers plugin as an allowed execution discipline while making CoordExp complex work retire production-shape, scale, artifact, and activation risks before broad implementation.

**Architecture:** Keep stable triggers and authority in `AGENTS.md`, plan-shape and audit frequency in `openspec/config.yaml`, and executable methods in existing CoordExp skills. Do not create a second workflow skill or duplicate OpenSpec authority.

**Tech Stack:** Markdown, YAML, Agent Skills, OpenSpec project configuration, Git.

## Global Constraints

- Preserve all unrelated dirty changes and stage only this plan's paths or hunks.
- The installed Superpowers plugin is execution discipline; OpenSpec remains scope, semantic, compatibility, and completion authority.
- Do not edit generated OpenSpec apply skills for project-local policy.
- Reuse `full-pipeline-smoke`, `model-innovation-risk-audit`, `agent-routing`, `audit-review`, and `handoff`; add no new skill.

---

### Task 1: Restore Superpowers authority wording and plan gates

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/AGENT_INDEX.md`
- Modify: `docs/PROJECT_CONTEXT.md`
- Modify: `docs/architecture/README.md`
- Modify: `openspec/config.yaml`

**Interfaces:**
- Consumes: current repository authority hierarchy and OpenSpec PWSG rules.
- Produces: unambiguous Superpowers/OpenSpec ownership, risk-first slices, resource budgets, and bounded audit gates.

- [x] **Step 1: Verify the current text exhibits the approved gaps**

Run:

```bash
rg -n "removed super-power|super-power plans|verification plus an independent audit|Gate \(verify \+ audit\)|End each wave" docs/AGENT_INDEX.md docs/PROJECT_CONTEXT.md docs/architecture/README.md openspec/config.yaml
```

Expected: legacy wording can be read as banning the installed plugin, and every wave requires independent audit.

- [x] **Step 2: Add the minimal stable trigger contract**

Edit `AGENTS.md` with one concise Development section covering OpenSpec/Superpowers authority, production-shaped risk retirement, scale budgets, and fresh phase handoffs.

- [x] **Step 3: Correct legacy wording and OpenSpec execution rules**

Edit `docs/AGENT_INDEX.md`, `docs/PROJECT_CONTEXT.md`, `docs/architecture/README.md`, and `openspec/config.yaml` so legacy repo-local super-power material remains non-authoritative while the installed plugin is explicitly permitted. Make routine gates executable and lead-owned; reserve independent audit for frozen decision-bearing or high-risk integration targets.

- [x] **Step 4: Verify the authority and gate language**

Run:

```bash
rg -n "installed Superpowers|OpenSpec remains|production-shaped|frozen.*high-risk|routine waves" AGENTS.md docs/AGENT_INDEX.md docs/PROJECT_CONTEXT.md docs/architecture/README.md openspec/config.yaml
```

Expected: every concept is owned exactly once and the old blanket prohibition is gone.

### Task 2: Add early and final production-shaped smoke contracts

**Files:**
- Modify: `.codex/skills/full-pipeline-smoke/SKILL.md`
- Test: `.codex/skills/full-pipeline-smoke/SKILL.md`

**Interfaces:**
- Consumes: a changed behavior and current production entry/runtime.
- Produces: an early risk-retirement slice and a final frozen integration smoke with measurable resource bounds.

- [x] **Step 1: Use historical failures as RED evidence**

Confirm the current skill lacks explicit early/final modes and required bounds for forward counts, collectives, cache passes, wall/RSS/workers, artifact payload, finalization, and recovery.

- [x] **Step 2: Add minimal early/final modes and budget contract**

Add an early risk-retirement path before broad implementation and retain the final frozen smoke. Require exact CLI/runtime, real wrapper, multi-process evidence where distributed shape matters, complete artifact publication/reload, downstream consumption, and representative resource bounds.

- [x] **Step 3: Validate the skill**

Run:

```bash
conda run -n ms python /data/CoordExp/.codex/skills/.system/skill-creator/scripts/quick_validate.py /data/CoordExp/.codex/skills/full-pipeline-smoke
```

Expected: `Skill is valid!`

### Task 3: Extend high-risk taxonomy and phase handoff

**Files:**
- Modify: `.codex/skills/model-innovation-risk-audit/SKILL.md`
- Modify: `.codex/skills/model-innovation-risk-audit/references/risk-taxonomy.md`
- Modify: `.codex/skills/handoff/SKILL.md`

**Interfaces:**
- Consumes: one high-risk mechanism or a substantial phase boundary.
- Produces: execution-topology, scale/resource, artifact/activation risk checks and a compact fresh-task continuation contract.

- [x] **Step 1: Add one-shot audit timing and three risk classes**

Make the risk audit run once before costly implementation and repeat only after a material target change. Add execution topology, scale/resource, and artifact/activation lifecycle risks to the taxonomy.

- [x] **Step 2: Add design-to-implementation and implementation-to-launch handoffs**

Require exact target identity, owning OpenSpec change, risk ledger, receipts, dirty scope, authority, consumed claims, and stop rule while keeping the handoff compact.

- [x] **Step 3: Validate both skills**

Run:

```bash
conda run -n ms python /data/CoordExp/.codex/skills/.system/skill-creator/scripts/quick_validate.py /data/CoordExp/.codex/skills/model-innovation-risk-audit
conda run -n ms python /data/CoordExp/.codex/skills/.system/skill-creator/scripts/quick_validate.py /data/CoordExp/.codex/skills/handoff
```

Expected: both return `Skill is valid!`

### Task 4: Verify, stage precisely, and commit

**Files:**
- Verify: all files listed above
- Commit: only the approved governance paths and the new execution plan

**Interfaces:**
- Consumes: completed fixed diff.
- Produces: one scoped commit and a receipt of remaining unrelated dirt.

- [x] **Step 1: Run repository checks**

Run:

```bash
openspec validate --all --strict
git diff --check
```

Expected: zero failures.

- [x] **Step 2: Inspect complete patch and stage exact paths/hunks**

Use explicit paths for clean files and an index-only patch for the new `AGENTS.md` section so the pre-existing Runtime hunk remains unstaged.

- [x] **Step 3: Verify staged scope**

Run:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Expected: only this plan's approved governance changes.

- [x] **Step 4: Commit**

Run:

```bash
git commit -m "docs: restore risk-first Superpowers workflow"
```

- [x] **Step 5: Confirm remaining dirt is unrelated**

Run:

```bash
git status --short
git show --stat --oneline --decorate -1
```

Expected: the commit contains only approved files; unrelated pre-existing changes remain.
