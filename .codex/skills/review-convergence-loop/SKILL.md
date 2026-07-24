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
-> decide fix/narrow/drop/probe/user decision
-> revise
-> repeat until convergence or explicit gate
```

This is an orchestration skill. Use narrower skills inside the loop when relevant: `grill-me record=local` for durable research/design pressure-testing, `audit-review` for read-only severity-ranked audits, and `model-innovation-risk-audit` for launch-risk gates. When an OpenSpec change is in scope, follow the installed OpenSpec workflow and the change's CLI-generated artifact instructions directly.

## Use When

Use when the user asks for any of:

- "loop until convergence"
- "do -> subagents review -> revise"
- "launch multiple subagents to audit/review/discuss"
- "draft/refine with review agents"
- "review-gated implementation"
- "pressure-test, revise, and wait for approval"
- "iterate until no blocking findings remain"

Do not use for simple one-shot answers, narrow command output, ordinary code review, ordinary implementation, or simple read-only audits unless the user explicitly wants iteration. Route one-shot work to `audit-review`, `model-diagnosis`, `model-innovation-risk-audit`, or the relevant workflow skill.

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
- decision at stake
- evidence that would change the decision
- artifact or evidence delta since any prior review
- stop condition
- approval gates

If the user asks for a side conversation or read-only exploration, do not mutate files.

If the request says "all", "every detail", "fully understand", "maximize subagents", or similar broad language, compress it into one decision and one stop condition before dispatching lanes.

### 2. Produce The First Artifact

Create or inspect the work product appropriate to the mode:

- research/design: synthesis, decision tree, architecture note, or ranked alternatives
- docs/spec/plan: draft doc, plan, OpenSpec change, or approval packet
- implementation: narrow patch with targeted tests
- audit/launch gate: severity-ranked findings and verification handles
- packaging: shortlist first, asset only after approval

Use current repo/docs/artifacts as authority.

### 3. Dispatch Independent Review Lanes

Use subagents only when the user explicitly requested subagents or the active workflow already permits them.

Dispatch according to independent decision surfaces, not available agent slots
or a mechanical lane count. Give each surface one current owner. A second agent
on the same surface must replace or audit the first, not create a duplicate
implementation or general-review lane. Use task-specific generic briefs and
select the model from `.codex/MODEL_ROUTING.md`. Useful independent lane types
include unknown-surface discovery, upstream dependency tracing, contract or
artifact audit, scientific interpretation, runtime receipt, research
synthesis, and bounded implementation. These are task descriptions, not
permanent role profiles.

Before every review wave, record:

- the open decision risk owned by each lane;
- the changed artifact or new evidence since the last wave;
- the action each possible verdict would trigger;
- why a deterministic check or concrete probe is insufficient.

If a lane has no distinct risk or no possible decision impact, do not dispatch
it. Default to one reviewer for one decision surface. Use a second lane only
when two genuinely independent judgment surfaces are both required.

Each prompt must name the exact evidence scope, mutation boundary, owned risk,
decision impact, and stop condition. For review output, use the `audit-review`
output contract rather than restating its severity and disposition schema here.
Reviewers must not broaden or edit outside their lane, and must report only new
findings or status changes when a prior ledger exists.

Do not wait idly. While agents run, continue local non-overlapping work.

### 4. Triage Findings

Use `audit-review` as the owner of severity vocabulary and exploratory blocking
criteria. Review findings are advisory evidence: the lead owns acceptance,
deduplication, synthesis, and final disposition. This loop owns orchestration
and convergence.
Apply its classification and disposition rules, reject or merge findings with
evidence, and treat missing or vague required reviews as unresolved rather than
approval.

### 5. Decision Turn

Before revising, apply the `audit-review` decision disposition. Do not patch
through `narrow`, `drop`, `probe`, or `needs user decision`. Before treating a
research-meaning change as `fix`, apply the
[originating-intent and semantic-delta gate](../coordexp-research-knowledge-workflow/references/research-graph-contract.md#originating-intent-and-semantic-delta-gate).
If the originating source does not determine the answer, pause at the owning
decision rather than letting review convergence manufacture a requirement.

### 6. Revise

Revise only the surfaces allowed by the current mode.

For accepted `fix` findings, update the allowed surface and its verification.
For `probe`, obtain the discriminator first; for `narrow` or `drop`, revise the
claim or scope rather than code by default. Fix lower-priority items only when
cheap and aligned.

For docs/spec/plan loops, capture important review resolutions in a review log or plan section. For implementation loops, run targeted tests after fixes.

### 7. Check Convergence

Convergence requires a disposition for every required P0/P1, completion of
reviews and verification named by the stop condition, bounded claims, and an
explicit next gate. The final artifact must distinguish approved, unsupported,
and still-gated scope.

For an exploratory research pilot, convergence does not mean every finding is
fixed. It means every conclusion-threatening finding has a disposition and the
remaining items are explicitly recorded as limitations, deferred debt, or
promotion blockers. Let the first representative model observation precede
production-style completeness.

Default max rounds:

- design/docs: 2 review rounds unless new P0/P1 appears
- implementation: 2 review/fix rounds; a third requires changed evidence on an
  unresolved P0/P1
- launch gate: 1 focused review round, then rerun only if evidence changed

Continue beyond the default only when each round is closing material findings. Do not churn on style-only P2s.

If a review wave produces no new accepted P0/P1 and no decision change, close
that surface immediately. Continue with a probe, bounded revision, narrowed
claim, user decision, or stop; do not seek another reviewer for reassurance.

Do not run a third clean review wave. Changed evidence, a new artifact version,
or an unresolved high-stakes decision may justify a focused follow-up on the
open finding, not another general review.

### 8. Stop Correctly

Stop state must be one of:

- `approved to implement`: plan/docs converged and user explicitly approved implementation
- `ready for user approval`: docs/design/plan converged, implementation not started
- `implemented and verified`: code/config/docs changed and verification passed
- `hold`: blocking issue remains with concrete reason
- `needs user decision`: the next fork changes research meaning, compatibility, cost, or irreversible behavior
- `probe required`: implementation/revision should wait for a concrete discriminating check
- `narrowed/dropped`: review changed the claim, scope, mechanism, or launch path rather than producing a patch

Never imply production readiness from skipped hardware smoke, timed-out review, partial tests, or narrow evidence.

For each loop, record the artifact/version reviewed, review lanes used, accepted/rejected findings, decision implication for every P0/P1, revision made or reason skipped, and the next gate or convergence decision.

## Verification

For docs/spec/plan loops:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for path in ["docs/catalog.yaml"]:
    if Path(path).exists():
        yaml.safe_load(Path(path).read_text())
        print(f"{path}: ok")
PY
rg -n "approval gate|not production eligible|OpenSpec|TODO|TBD|\\.\\.\\." docs research openspec
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

For custom-agent changes:

```bash
python - <<'PY'
import pathlib, tomllib
required = {"name", "description", "developer_instructions"}
for path in pathlib.Path(".codex/agents").glob("*.toml"):
    data = tomllib.loads(path.read_text())
    missing = required - data.keys()
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    print(f"{path}: ok")
PY
git diff --check
```

For implementation loops, run the targeted tests named in the plan, docs, OpenSpec change, or `docs/IMPLEMENTATION_MAP.md` before broad tests.

## Output

Report the mode, decision and stop condition; reviewed artifact and lanes;
accepted or rejected findings and their dispositions; revision and verification;
remaining gates; and the exact stop state from Step 8.
