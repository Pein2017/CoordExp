---
name: skill-doctor
description: Audit local agent skills against real conversation evidence, identify retrieval or workflow gaps, and propose or apply authorized improvements. Use for skill effectiveness and skill-library maintenance, not ordinary task status or generic code review.
---

# Skill Doctor

Use real conversations to decide whether a skill changed the work, whether a
missing rule would have prevented a failure, and which current owner needs an
update. Installed names, a green validator, or a skill mention do not demonstrate
effective use. The output is an evidence-backed maintenance decision; a scorecard
is optional.

## Scope and evidence

Read [supported harnesses](references/supported-harnesses.md) before collection.
Infer scope from the user's named repository, research topics, date window and
skill families. Ask only when a missing choice changes what may be examined.
Keep transcripts and extracts local; do not upload or publish them. Use a fresh
scratch directory for audit artifacts. Change actual skills/docs only within the
user's authorization; authorization already given in this task remains valid.

Use memory or a session index to locate relevant source runs, then inspect the
actual messages, tool results and affected artifacts. Prefer a bounded sample
covering the reported failure modes and successful counterexamples. State how it
was selected; parent/child fragments and resumed copies are not independent tasks.
Do not broaden into every repository or every installed plugin by default.

The bundled collector is a convenience:

```bash
python scripts/collect_sessions.py --harness codex --codex-home /path/to/codex \
  --repo /path/to/repo --skills-dir /path/to/skills \
  --days 35 --max-sessions 12 --out /fresh/scratch/directory
```

Resolve these paths from this skill and the selected environment; use the
applicable source flags from the harness reference. Inspect `inventory.json`
before scoring: scope, unique session identities, captured user/assistant/tool
entries, truncation and sampling exclusions matter. A zero sample after nonzero
in-scope records is not evidence of zero activity or zero skill use. Check one
known source record for a collector/format mismatch; if needed, use a bounded
manual extraction and report the coverage limitation. Do not manufacture scores
from missing evidence or repair an unrelated ingestion system for the audit.

## Attribute before changing instructions

For each decision-bearing episode, record:

- the task and exact user constraint;
- an observed mistake, avoidable rework, or correctly prevented failure;
- the actual read/invocation and outcome evidence, when available;
- whether the current instruction was missing, undiscoverable, stale, ignored,
  or unrelated to the cause;
- the smallest useful change and its owning surface, or why no edit is warranted.

A tool call mentioning a skill is not proof the file was successfully loaded or
followed. Re-read the current instruction before proposing a fix: a past gap may
already have been repaired. Distinguish source/code defects from instruction
problems and model variance. A repaired output does not erase avoidable rework;
a missing full diff does not justify a code-quality verdict.

Use [efficiency](scorers/efficiency.md) and
[code quality](scorers/code-quality.md) rubrics if a grade was requested, or as
optional lenses for an evidence matrix. Do not turn selected failure episodes
into population-wide performance or causal model-family claims. Exclude
insufficient code evidence from code-quality aggregation. Report the denominator
and raw labels; a synthetic overall grade is not needed for a maintenance plan.

## Choose the knowledge owner

- A skill description should make a recurring task discoverable. For a missing
  read, inspect the pointer first: does it name the trigger, target and decision
  supported? Strengthen that pointer before copying the target into context.
  Check one representative task that should trigger the read and one that should
  not; distinguish a readable file from evidence that it was actually loaded.
- Its body should guide the task and load only relevant knowledge.
- Conditional local procedures belong in its references.
- Shared model/domain semantics belong in one maintained document or contract,
  linked from the relevant skills; do not copy a manual into several skills.
  Keep shared policy at its contract, role actions in the skill, and branch-only
  details behind conditional references. Replace duplicate clauses with precise
  links; do not make a new constitution or registry just to index the old ones.
- An invariant with a deterministic counterexample usually needs an executable
  test at its code owner, not another paragraph of warnings.
- A decision/exception for one experiment belongs in that research record, not
  a universal rule for future work.

Preserve distinctions between diagnosis, mechanism-fidelity review, scientific
interpretation, execution and launch acceptance when their outputs differ.
Merge duplicated policy/reference content before combining distinct skills.
Retire a skill when its supported workflow has actually disappeared or another
owner fully replaces it; non-use in a small sample alone is insufficient.

## Make and validate the change

Follow [skill improvements](references/skill-improvements.md) for failure-driven
repairs and `skill-creator` for skill structure. For an explicitly requested
knowledge-library redesign, current source/ownership and demonstrated retrieval
failures also justify routing/documentation changes; label that evidence
separately from failed-session attribution.

Prefer replacement over accumulated instructions. Do not restate generic model
knowledge or add an audit gate simply because a technical reference is useful.
If edits are not authorized, save reviewable proposals/diffs in the scratch
report. If authorized, preserve unrelated work and implement the selected changes
without asking the same permission again.

Validate changed frontmatter and local references, inspect exact diffs, and test
changed scripts with a real failing input before the fix. For a substantial
skill, a bounded independent forward-test can check a realistic task using the
new guidance. Do not give the evaluator the desired answer. A successful
forward-test demonstrates usability for those cases, not measured productivity
improvement; do not launch another open-ended review.

## Deliver

Report concrete findings, changes or proposed dispositions, source evidence,
validation and remaining limits. A Markdown report is sufficient. The bundled
`render_report.py` is a legacy branded scorecard; use it only when the user asks
for that format. Do not add unrelated promotion or an automatic permission
question to the user's result.
