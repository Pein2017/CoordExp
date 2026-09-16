# Learn delegation routes from accepted work

Use this reference only when a requested or decision-changing cost comparison
needs structured records. Ordinary delegation does not require these receipts.
This is an optional evidence workflow, not a scheduler or automatic model ranking.
The lead owns labels and route decisions. No new benchmark, reviewer, background
service or model call is required to maintain it. Use task evidence already
collected; do not rescan full sessions just to fill a receipt.

## Optional use for a requested comparison

1. Before a consequential assignment, read a relevant existing summary if it is
   available. Compare task class, contract/acceptance, risk, verifier, brief and
   topology. With no comparable evidence, keep the choice tentative and use the
   current task's verifier and uncertainty. Record the chosen route and short rationale in the ordinary brief.
2. When the comparison needs an acceptance, repair, switch or invalidation
   update, the lead writes one full task snapshot to a task-owned
   `routing/revision-0001/delegation-outcomes.jsonl`. A later snapshot goes in
   `revision-0002/` with the same stable task ID and incremented revision. Keep
   prior attempts in order. Only that task's owner writes there; unrelated leads
   use their own artifact directories. This does not authorize mutation outside
   the task's output scope. Worker self-report alone is not acceptance.
3. Refresh the summary when new relevant evidence could change a route, or when
   the user requests it. Pass explicit task directories or files; do not scan all
   of CODEX_HOME, memory or the repository by default. Summaries are disposable;
   use a caller-owned output directory, not one shared writable global report.
4. Change a route preference only after inspecting comparable acceptance and
   cost evidence. One run is an observation, not a default. Include scope,
   evidence links and uncertainty in the ordinary decision record. Scripts never
   rewrite SKILL.md, model settings or persistent policy. A changed model version,
   brief, verifier or task scope can make earlier evidence inapplicable.

Paths in the commands below are examples: replace TASK_OUTPUT and REPORT_OUTPUT
with the current task's actual owned directories. No daemon or installation.

```bash
python /data/CoordExp/.codex/skills/native-agent-team-guide/scripts/routing_evidence.py validate TASK_OUTPUT/routing
python /data/CoordExp/.codex/skills/native-agent-team-guide/scripts/routing_evidence.py summarize TASK_OUTPUT/routing --output-dir REPORT_OUTPUT --task-class code-repair
```

The second command defaults to `--origin production`. Use `--origin benchmark`
for exams, or `--origin all` to see both in separate groups. Optional
`--comparison-key KEY` narrows to the lead-declared comparable question;
`--since YYYY-MM-DD` filters latest record dates. It is not a pricing refresh or
model-version resolver. With valid records but no filter matches, the report
explicitly shows zero tasks. An empty input directory or malformed input fails.

Read summary.md first (at most30 groups and3 task examples per group, with
explicit omission notices); summary.json and record_sources retain full detail.
One explicitly supplied JSONL file may have any filename. Directory discovery
only reads files named delegation-outcomes.jsonl. Inputs remain read-only.
The small CLI validates syntax/accounting shape, not truth of evidence or quality.

## Receipt schema v1

See [record-example.json](record-example.json) for a valid illustrative record.
Serialize one JSON object per line. The example is not an actual accepted task;
its filename deliberately excludes it from automatic directory discovery.

Required task fields:

- `schema_version`: 1; `task_id`: globally stable task/package ID, e.g.
  `THREAD_UUID/package-name`; `revision`: positive integer;
  `recorded_at`: ISO timestamp with timezone.
- `origin`: production or benchmark; never promote an exam to production.
- `task_class`: short reusable class, e.g. code-repair or research-analysis.
  `comparison_key`: lead-owned cohort of comparable task constraints/acceptance,
  not a model name. If comparability is unknown, use a task-specific key instead
  of pooling unrelated work. It does not prove equal difficulty or randomization.
- `risk`: low/medium/high; `verifier`: deterministic/integration/review.
  `brief_style`: a concise brief version/shape label; `topology`: direct/flat or
  a concise owned-package arrangement. All are grouping conditions. Use actual
  model identifiers and meaningful versions to prevent accidental pooling.
- `outcome`: accepted, failed, pending, invalidated. Pending means the task is
  unresolved after recorded attempts; do not invent costs for a running attempt.
  A later discovered acceptance error creates an invalidated higher revision;
  it must not silently remove the earlier spend.
- `evidence`: nonempty paths/references to the task contract, actual acceptance
  checks, scope/brief rationale and cost records as applicable.
- `attempts`: ordered list of **all incremental attempts** through the current
  boundary, including failures, repair, escalation and takeover. Never add a
  cumulative worker-resume total to its first-turn total.

Each attempt has these required fields:

- `attempt_id`: globally unique, e.g. worker thread ID plus nonoverlapping turn
  range; `model`, `effort`, `fork`: actual route, including lead takeover when
  applicable. Fork is none, all, or a positive integer string. Supported effort
  labels here are low/medium/high/xhigh/max; runtime availability remains a
  dispatch check. Higher Astra efforts are not excluded.
- `outcome`: accepted/rework/failed/escalated; `failure_kind`:
  none/implementation/reasoning/brief/environment/verifier/unknown.
  A corrective follow-up is a new incremental attempt even on the same worker.
  Internal worker self-testing remains in the original attempt.
- `worker_usd`, `lead_usd`, `runtime_usd`: nonnegative finite USD estimates or
  null. Zero means measured/not applicable, not missing. Lead costs cover only
  attributable briefing, review and integration not already billed as worker
  takeover. Runtime includes billed resources, not elapsed time alone.
- `cost_source`: receipt/ledger path and dated rate provenance, or explicit
  explanation of missing costs. Use codex-usage-ledger when necessary; do not
  recreate its token/pricing parser here. Reasoning is included in output.

One accepted task needs a final accepted attempt; earlier attempts cannot also
be accepted. A task later invalidated can retain its historical accepted attempt.
A revision preserves previous attempt IDs/order, but can correct their labels or
cost estimates with evidence. Exact duplicate task/revision rows are deduplicated;
conflicting rows fail. The latest revision wins before date filtering. Reusing an
attempt ID across tasks fails, but the script cannot detect overlapping token
ranges under different IDs: the lead still owns that accounting boundary.

For nested delegation, choose one accounting unit: either the package chain
including its descendants or disjoint child tasks. Do not record the same spend
both ways. Keep shared/unattributable lead costs separate in the ordinary report;
leave lead_usd null instead of spreading them evenly across routes.

## Interpret the summary

Groups preserve all task conditions and the **full observed route sequence**,
including effort/fork and repeat attempts. A Luna-to-Astra rescue is not credited
as an Astra-only success. Failed tasks remain in group spend. Total
cost_per_accepted_usd is sum of all group attempt costs divided by accepted tasks,
only when all costs are known and all tasks are terminal accepted/failed.
With unknown cost, pending work, invalidation or zero accepted tasks it is null.
Worker-only cost is labeled separately; known spend is merely a subtotal.

Observed chains are descriptive, not estimates of an initial model's expected
cost: a repaired chain is selected by its first failure. Inspect all matching
starting routes and failure/takeover evidence before deciding. Do not compare
only successful tasks, sort a global leaderboard, infer probabilities from one
exam, or reward a route merely because its lead costs were not measured.

Use comparable real tasks to try an alternative route only when both choices
fit the existing quality/resource boundary. No fixed exploration quota, duplicate
execution or automatic extra budget. Include the cost of gathering evidence;
stop maintaining it when it would cost more than it can reasonably inform.

## Maintainer verification

```bash
python /data/CoordExp/.codex/skills/native-agent-team-guide/scripts/test_routing_evidence.py -v
```

These black-box CLI tests cover failed/unknown spend, route chains, revisions,
duplicates, source discovery and invalid input. They validate accounting
behavior, not whether a model preference improves real research outcomes.
