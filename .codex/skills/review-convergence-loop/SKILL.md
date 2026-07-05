---
name: review-convergence-loop
description: Use when the user explicitly asks to iterate a CoordExp task through work, review or audit lanes, revision, and a convergence or approval gate.
---

# Review Convergence Loop

Run an explicit loop:

```text
produce or inspect work -> focused review lanes -> triage -> revise -> gate
```

Do not use for ordinary one-shot answers, simple code review, narrow command
output, or normal implementation. Route those to the narrow skill that owns the
surface.

## Scope

State the loop before acting:

- objective;
- mode: `research/design`, `docs/spec/plan`, `implementation`,
  `audit/launch-gate`, or `packaging`;
- mutation scope: read-only, docs-only, code/config allowed, or launch allowed;
- source-of-truth surfaces;
- stop condition and approval gate.

Respect the mode. Do not turn a design loop into implementation or a partial
smoke into a launch claim.

## First Pass

Create or inspect the smallest artifact that can be reviewed:

- `research/design`: decision tree, alternatives, mechanism note, or synthesis;
- `docs/spec/plan`: draft doc, plan, OpenSpec change, or approval packet;
- `implementation`: narrow patch with targeted verification;
- `audit/launch-gate`: findings and evidence handles;
- `packaging`: shortlist first, asset only after approval.

Use current docs, specs, configs, code, artifacts, and research notes as
authority. Use `grill-me record=local` when the loop is mainly a durable
decision pressure test.

## Review Lanes

Use subagents only when the user explicitly requested subagents, review agents,
parallel lanes, or a workflow that already permits them. When allowed and the
evidence surface is broad, dispatch independent lanes instead of stacking all
raw context into one reviewer. Keep lanes independent and bounded; prefer 2-6
only when the work is broad enough.

Each lane prompt must include:

```text
Scope:
- exact artifact or work to review
- allowed files, docs, artifacts, or mutation boundary

Output:
- P0/P1/P2 findings
- evidence handles
- impact
- fix direction
- verification
- confirmed OK checks
- unresolved questions

Rules:
- do not broaden beyond the assigned lane
- do not edit unless assigned an implementation slice
- timeout or disconnection is unresolved, not approval
```

While review lanes run, continue local non-overlapping work.

## Triage

Classify every claim:

- `P0`: invalidates correctness, research meaning, reproducibility, or launch
  safety;
- `P1`: substantial risk to supported workflow, contract, metrics, artifacts, or
  maintainability;
- `P2`: clarity, coverage, or future-maintenance issue;
- `non-blocking`, `wrong`, or `duplicate`.

Accept findings with evidence, reject with evidence, or convert the fork into a
user decision. Do not let vague reviewer output block forever.

## Revise And Gate

Revise only allowed surfaces. For accepted P0/P1 findings, update the
artifact/work and add or rerun targeted verification. Fix P2 only when cheap and
aligned.

Converged means:

- all P0/P1 findings are fixed, rejected with evidence, or converted into a user
  decision;
- required review lanes are complete or explicitly marked unresolved;
- verification covers the actual requirement;
- stable docs/specs/launch claims do not exceed evidence scope;
- the final state is one of `ready for user approval`, `approved to implement`,
  `implemented and verified`, `hold`, or `needs user decision`.

Default max rounds: 2 for design/docs, 3 for implementation, 1 for launch gates
unless new evidence changes the decision.

## Verification

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

For implementation loops, run the targeted tests named by the plan, docs,
OpenSpec change, or `docs/IMPLEMENTATION_MAP.md`.

## Output

Report the mode, mutation scope, reviewed artifact, lanes used, accepted and
rejected findings, revisions made, verification run, remaining gates, and final
state.
