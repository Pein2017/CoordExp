# Current Model Priors

> Time-sensitive empirical state for the `agent-routing` skill. The lead uses
> this reference when surface, model, or effort selection is material. Workers
> receive the selected route in their brief and do not load this file.

## Availability

- Verify the live spawn interface before every non-default route; availability
  is version- and account-sensitive.
- The installed Claude Code interface currently exposes Haiku, Sonnet, Opus,
  and Fable, each with `low`, `medium`, `high`, `xhigh`, and `max` effort.
- Codex Multi-Agent V2 currently exposes Terra and Sol in this environment.
  Legacy Multi-Agent V1 may expose the wider picker, including Luna; use Luna
  only when the active interface advertises it. Do not switch agent versions
  merely to obtain another model without a task-specific reason.
- Treat surface, model, effort, and role as separate choices. Cross-provider
  capability and spend are not a single permanent ranking.

## Model Priors

| Model | Provisional fit | Important boundary |
| --- | --- | --- |
| Claude Haiku | Fast mechanical work, extraction, smoke tests, and bounded scans behind a verifier | Its detections can be trusted behind a verifier; its narrative diagnoses cannot. Read the receipt, not the summary |
| Claude Sonnet | Balanced bounded implementation and ordinary coding | Require structural acceptance; avoid repeating expensive real probes across tests |
| Claude Opus | Cross-file integration, architecture seams, broad review, and source archaeology | Calibrate severity, deduplicate findings, and constrain scope |
| Claude Fable | Core planning, architecture alternatives, and difficult decision discussion | Local task-class evidence is sparse; do not spend it on routine implementation |
| Codex Luna | Fast, lightweight, high-volume mechanical work when V1 advertises it | Not currently available through the active V2 spawn surface |
| Codex Terra | Fast scouting, stable self-contained implementation, integration, and contract review | Do not hard-code moving external snapshots; verify structural behavior |
| Codex Sol | Research reasoning, contract and concurrency review, silent-correctness seams, and conclusion-critical synthesis | Strong reasoning does not guarantee exact mechanical details |

Effort tunes depth within a model; it does not define the role or improve every
model monotonically. Treat model and effort as an interacting route pair. Start
at low for straightforward verified work, medium for ordinary bounded agents,
and high for cross-file reasoning or review. Use `xhigh` or `max` only for a
localized, consequential uncertainty that lower effort has not resolved.
When a capable model broadens scope, redesigns unnecessarily, or turns a
builder lane into an audit, tighten the brief or lower effort before escalating;
that is a different failure from insufficient reasoning depth.

## Observed Differentiators

- Serena Light audits on 2026-07-28–29 reinforced a complementary route:
  Sol/high-xhigh was concise on contract, concurrency, and silent-correctness,
  including two sequential P1 freshness counterexamples; Opus/high-max was
  broader on repository archaeology, lifecycle, environment, and runtime
  evidence but needed tighter scope and severity calibration. This was
  role-shaped, not a paired ranking; any HOLD blocked release, and provider
  diversity did not replace lead verification.
- In one matched implementation task on 2026-07-28, Sonnet/medium,
  Terra/medium, Opus/high, and Sol/high all passed focused tests initially. A
  later external-repository change exposed hard-coded snapshot assertions in
  the Terra and Sol results, while Sonnet and Opus used structural set
  assertions. Opus also found the adjacent renderer-schema seam; Sonnet was
  adequately scoped but repeated an expensive real probe across tests.
- In one three-lane research-probe build on 2026-07-29 (Claude Code, all lanes
  given disjoint write surfaces and a lead-owned mechanical verifier),
  Haiku/default correctly DETECTED a real config-fingerprint mismatch but
  attached a causal diagnosis it had not tested ("configs need synchronizing";
  the actual delta was one derived output-path field), and shipped a module that
  could not be imported via `spec_from_file_location`. Sonnet/default matched an
  exact 24-cell acceptance table with no tuning and independently found a defect
  in the LEAD's frozen contract — a join key that was not unique across
  checkpoints — then failed closed rather than joining ambiguously. Opus/default
  took the highest silent-error lane, deviated from the brief by reimplementing
  a bulk logprob path (the canonical helper discards the logits its outputs
  needed), disclosed the deviation, and validated it bitwise against the
  canonical helper; the lead's independent recomputation agreed at exactly 0.0.
  Routing reading: cheap lanes need their *prose* verified, not just their
  artifacts; a strong model earned its slot by disclosing a necessary deviation
  rather than by avoiding one.
- In one bounded-online training-infrastructure task on 2026-07-29, role-shaped
  routing mattered more than a simple model ranking. Terra/medium was fast and
  useful for scoped implementation but twice left cross-owner artifact and
  lifecycle closure to the lead; Sol/high closed those integration seams.
  Fable/high produced useful architecture and resource estimates but missed an
  image-plan determinant and one lifecycle detail. Sonnet/medium efficiently
  updated profiles and authority text but needed narrow corrections to exact
  dequeue semantics. Opus/high found launch-blocking full-horizon/cap defects
  with an executable probe after broad tests had passed, at materially higher
  latency. Haiku/low launched and monitored the mechanically specified W8 smoke
  correctly but inferred fresh-loadability from file presence until the lead ran
  the payload inspectors. Treat this as provisional task-class evidence, not a
  paired benchmark: use cheap lanes behind exact receipts, reserve strong audit
  lanes for silent launch risk, and keep final cross-owner closure with the lead.
- Treat API retries, quota exhaustion, completion delay, and lifecycle failures
  as surface or runtime evidence, not model-quality evidence.

## Maintenance

- Keep only evidence that changes a routing decision. One result is provisional;
  repeated comparable evidence may change a default.
- Revisit affected priors when a model revision, agent version, tool surface,
  or common task class changes. Keep unavailable models inactive rather than
  deleting their evidence.
- Edit recommendations in place and delete stale or absorbed prose. Do not
  require a benchmark, profiling pass, per-task ledger, or formal routing
  matrix.
