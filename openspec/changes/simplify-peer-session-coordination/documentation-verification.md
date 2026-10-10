# Documentation verification — tasks 5.1–5.3

## Scope and qualification

This record covers only the documentation/skill work in tasks 5.1–5.3. The
Pi helper CLI is being implemented by its code owner and remains a moving
consumer. This pass does not inspect or change scripts/tests and does not claim
that command examples parse against the completed CLI; the parent owns that
follow-up verification, task checkboxes and final verification. Examples below
are static walkthroughs of the agreed OpenSpec design, with illustrative
paths/revisions rather than live research state.

## Task map

| Task | Documentation evidence | Verification boundary |
|---|---|---|
| 5.1 | `.codex/skills/lead-worker/SKILL.md` now directs generic package, continuation and acceptance policy to the native-team guide, distinguishes destination-backend transport, and links `references/coordination-examples.md`. The examples include peer/dependency-aware assignments, factual coordination, update/decision/urgent briefs, unaffected continuation, escalation, presentation-only supersession and independent initial advisor views. | Static content and link checks only; no skill-trigger, behavior or token/efficiency benchmark. |
| 5.2 | `.codex/skills/lead-worker/references/pi-transport.md` retains legacy pair examples and documents exact Pi sender/recipient IDs with explicit cwd, version-1 `pi_peer_transport` aliases, report pointers separate from message bodies, and queue/steering/receipt limits. | Static command-shape review against the design; CLI parsing is deliberately deferred to the parent after implementation completes. |
| 5.3 | The bounded three-case walkthrough below checks audience, source/revision, action and authority boundary. | Illustrative static cases, not a live send, current-state review or research conclusion. |

## Three-case static walkthrough

| Case | Audience and source/revision | Required action | Authority boundary |
|---|---|---|---|
| Schema dependency coordination | The adapter worker and the named schema-owner peer; task contract `contracts/input-schema.md`, revision `r3`, relevant field section. | Ask the peer for the exact field name/type; continue unrelated adapter work while that fact is resolved. | A factual answer is not approval to change scientific meaning. A schema change affecting the agreed sample, interpretation, cost or stop boundary returns to the existing decision owner. |
| Supersede a CE presentation report | Consumers of the report and the lead; `reports/ce-presentation.md`, revision `r2`, `summary-table`, superseding only the `r1` presentation table. | Stop reusing the superseded displayed table and use the corrected section. | The example limits supersession to presentation; keyed observations and their original source remain intact, and scientific interpretation stays with its existing owner. |
| Urgent invalidated target binding | The worker about to act and the lead/decision owner as needed; `config/target-binding.json`, revision `r6`, which invalidates the previously selected `r5` binding. | Do not start using `r5`; reread `r6` and report whether any action already started. Request steering only when a busy recipient needs the correction at its next processing boundary. | Steering is neither process cancellation nor proof of consumption; it does not authorize a replacement target or broaden the task. Reconcile any already-started external action with its owner. |

## Static limits

Static checks passed: local Markdown links/anchors and fenced blocks resolve in
all four documentation files; all eight Pi transport shell examples pass
`bash -n` syntax parsing only (not execution); the participant JSON parses and
matches the documented version-1 shape; and explicit-peer, alias, report-pointer
and coordination example fields are present. This is not CLI argument parsing,
which remains for the parent after the implementation settles.

No live Pi Web message, model/GPU work, child session, or paid skill benchmark
was run. No behavioral efficiency, token savings, delivery/read/acceptance or
research-outcome claim is made.

## Parent consumer check

After implementation, the parent loaded the retained public CLI with `runpy`,
extracted all eight shell examples, substituted test-owned directories, canonical
fixture IDs and a temporary version-1 participant state, and called `parse_args`
for each. All **8/8** parsed successfully (exit **0**). No `operate` or API call
was executed. This closes the deferred CLI argument check, not live delivery.
