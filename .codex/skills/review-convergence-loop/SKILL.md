---
name: review-convergence-loop
description: Use when the user asks to iterate a CoordExp task through doing work, subagent review/audit/discussion, revision, and repeated convergence checks, especially for architecture decisions, specs, implementation plans, code changes, launch gates, or research designs that need parallel critique before approval.
---

# Review Convergence Loop

Use this to run an explicit convergence loop:

```text
produce artifact/work
-> independent review/audit/discussion lanes
-> triage findings
-> revise
-> repeat until convergence or explicit gate
```

This is an orchestration skill. Use narrower skills inside the loop when relevant: `grill-me-with-docs` for durable research/design pressure-testing, `audit-review` for read-only severity-ranked audits, `model-innovation-risk-audit` for launch-risk gates, `openspec-lifecycle` for stable contract changes, and Superpowers review/execution skills for implementation-plan execution.

## Use When

Use when the user asks for any of:

- "loop until convergence"
- "do -> subagents review -> revise"
- "launch multiple subagents to audit/review/discuss"
- "draft/refine with review agents"
- "review-gated implementation"
- "pressure-test, revise, and wait for approval"
- "iterate until no blocking findings remain"

Do not use for simple one-shot answers, narrow command output, or ordinary code review unless the user explicitly wants iteration.

## Mode

Classify the loop before acting:

- `research/design`: brainstorm, mechanism design, architecture decision, no implementation unless later approved.
- `docs/spec/plan`: draft durable docs, specs, OpenSpec changes, or implementation plans.
- `implementation`: make code/config/docs changes from an approved plan.
- `audit/launch gate`: decide whether to promote, hold, rerun, or request user decision.
- `packaging`: turn repeated workflow evidence into skills, agents, automations, or deliberate skips.

Respect the mode. Do not turn a design loop into implementation. Do not promote stable docs, OpenSpec, production runs, or launch claims unless that is explicitly in scope.

## Loop

### 1. Set Scope

State:

- objective
- mode
- allowed mutation level: read-only, docs-only, code/config allowed, or launch/deploy allowed
- source-of-truth surfaces
- stop condition
- approval gates

If the user asks for a side conversation or read-only exploration, do not mutate files.

### 2. Produce The First Artifact

Create or inspect the work product appropriate to the mode:

- research/design: synthesis, decision tree, architecture note, or ranked alternatives
- docs/spec/plan: draft doc, plan, OpenSpec change, or approval packet
- implementation: narrow patch with targeted tests
- audit/launch gate: severity-ranked findings and verification handles
- packaging: shortlist first, asset only after approval

Keep work scoped. Use current repo/docs/artifacts as authority.

### 3. Dispatch Independent Review Lanes

Use subagents only when the user explicitly requested subagents or the active workflow already permits them.

Use 2-6 lanes. Keep them independent. Common lanes:

- architecture/code-boundary reviewer
- upstream dependency reviewer
- test/verification reviewer
- docs/governance reviewer
- artifact/eval-validity reviewer
- implementation quality reviewer
- research failure-mode reviewer

Each subagent prompt must include:

```text
Scope:
- exact artifact/work to review
- exact files/docs/artifacts allowed
- read-only or mutation rules

Output:
- P0/P1/P2 findings
- evidence handles
- impact
- fix direction
- verification
- confirmed OK checks
- unresolved questions

Rules:
- do not edit unless explicitly assigned an implementation slice
- reviewer timeout/disconnection is unresolved, not approval
- do not broaden beyond assigned lane
```

Do not wait idly. While agents run, continue local non-overlapping work.

### 4. Triage Findings

Classify every returned issue:

- `P0`: invalidates correctness, research meaning, reproducibility, or launch safety
- `P1`: substantial risk to supported workflow, contract, metrics, artifacts, or maintainability
- `P2`: clarity, coverage, or future-maintenance issue
- `non-blocking`: useful but not required now
- `wrong`: reject with technical reason and evidence
- `duplicate`: merge into existing finding

Timeouts, missing reviewers, or vague reviewer claims are not approval.

### 5. Revise

Revise only the surfaces allowed by the current mode.

For each accepted P0/P1, update the artifact/work and add or update verification coverage. Fix P2 only when cheap and aligned.

For docs/spec/plan loops, capture important review resolutions in a review log or plan section. For implementation loops, run targeted tests after fixes.

### 6. Check Convergence

Converged only when all required conditions are true:

- all P0/P1 findings are fixed, explicitly rejected with evidence, or converted into a user decision
- no reviewer output is pending if it is needed for the stop condition
- verification commands or doc/routing checks cover the actual requirement
- approval gates are explicit
- stable docs/specs/launch claims do not exceed the evidence scope
- final artifact states what is approved, what is not approved, and what remains gated

Default max rounds:

- design/docs: 2 review rounds unless new P0/P1 appears
- implementation: 3 review/fix rounds per task before escalating
- launch gate: 1 focused review round, then rerun only if evidence changed

Continue beyond the default only when each round is closing material findings. Do not churn on style-only P2s.

### 7. Stop Correctly

Stop state must be one of:

- `approved to implement`: plan/docs converged and user explicitly approved implementation
- `ready for user approval`: docs/design/plan converged, implementation not started
- `implemented and verified`: code/config/docs changed and verification passed
- `hold`: blocking issue remains with concrete reason
- `needs user decision`: the next fork changes research meaning, compatibility, cost, or irreversible behavior

Never imply production readiness from skipped hardware smoke, timed-out review, partial tests, or narrow evidence.

## Verification

For docs/spec/plan loops:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for path in ["docs/catalog.yaml", "progress/index.yaml"]:
    if Path(path).exists():
        yaml.safe_load(Path(path).read_text())
        print(f"{path}: ok")
PY
rg -n "approval gate|not production eligible|OpenSpec|TODO|TBD|\\.\\.\\." docs progress openspec
git diff --check
```

For skill changes:

```bash
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/<skill>
python - <<'PY'
import pathlib, yaml
for path in pathlib.Path(".codex/skills").glob("*/agents/openai.yaml"):
    yaml.safe_load(path.read_text())
PY
git diff --check
```

For implementation loops, run the targeted tests named in the plan, docs, OpenSpec change, or `docs/IMPLEMENTATION_MAP.md` before broad tests.

## Output

Report:

- mode and mutation scope
- artifact/work produced
- review lanes launched
- accepted findings and revisions
- rejected findings with reason
- verification run
- remaining gates
- exact next state: `ready for user approval`, `hold`, `needs user decision`, or `implemented and verified`

Keep the final report concise and evidence-backed.
