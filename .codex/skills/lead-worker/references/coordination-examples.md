# Coordination examples

These are compact patterns, not a second team policy. Follow the
[native-team guide](../../native-agent-team-guide/SKILL.md) for outcome packages,
continuation, escalation and acceptance. Use source names/revisions from the
current task owner; the paths and wording below are illustrative.

## Assignment: make the dependency visible

> **Outcome / owner:** Worker A delivers the input adapter against the agreed
> schema; Worker A owns its implementation and report.
> **Dependency / peer:** Worker B owns the schema. Ask B directly for the field
> name, type and schema revision needed by the adapter; do not ask B to approve
> unrelated scientific meaning.
> **Source / report:** Use the task's schema contract at revision `r3`; return
> the result to the existing task report at `reports/adapter.md` and send the
> lead a concise status.
> **Continue:** Tests and implementation that do not depend on the unresolved
> field may proceed.
> **Escalate:** A schema change that alters the agreed sample, interpretation,
> material cost or stop boundary goes to the existing decision owner before
> that affected work proceeds.

The assignment identifies relevant peers and dependencies without making them
co-owners of the deliverable. Keep public shared index/state at its existing
owner; do not create a board or broadcast channel. Use normal task checkpoints,
not automatic inbox polling or acknowledgments for every phase.

## Briefs: state intent and evidence

- **Update:** “The adapter now accepts `feature_id` as a string. Verified
  against `contracts/input-schema.md`, revision `r3`, section `fields`. The
  independent unit tests can continue; report is at `reports/adapter.md`.”
  Send to the lead and only peers whose next step depends on the change.
- **Decision request:** “The current schema at revision `r4` makes `target_key`
  optional. Should the agreed task require it, or should this go back to the
  schema owner? This affects only the dependent integration step; unrelated
  checks continue. Source: `contracts/input-schema.md`, section `target_key`.”
  Address the existing decision owner, state the exact decision and impact, and
  do not treat a peer reply as a change of authority.
- **Urgent correction:** “Do not start using target binding `r5`: the task-owned
  validation at `config/target-binding.json`, revision `r6`, invalidates that
  selection. Re-read `r6` and report whether any action already started.” Send
  to the person about to act and the lead when the decision owner needs to know;
  explicitly request steering only if a busy recipient needs the correction at
  its next processing boundary. A receipt is not proof it was seen, and steering
  does not cancel an external job.

The words *update*, *decision* and *urgent* describe the message, not a delivery
mode, permission or mandatory acknowledgment. Short messages need no rigid word
limit; link a report when its detail matters rather than pasting it into every
message.

## Supersede only the affected presentation

> “Use the CE presentation table in `reports/ce-presentation.md`, revision `r2`,
> section `summary-table`; it supersedes the displayed table in `r1` because the
> labels were transposed. This correction is presentation-only: the underlying
> keyed observations and their source are unchanged.”

The supersession reference names the affected presentation, not all earlier
work or evidence. Keep the original source and report history intact; consumers
should stop using only the superseded table. Record the report path, section,
revision and superseded reference as pointers, not as permission to read or
rewrite the source.

## Independent advisors

When two advisors are asked for independent initial judgments, collect each
initial view before showing either advisor the other's conclusion. Then share
only what is needed for comparison and preserve each original assessment. This
reduces avoidable anchoring without changing who owns the decision.
