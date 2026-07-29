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
| Claude Haiku | Fast mechanical work, extraction, smoke tests, and bounded scans behind a verifier | Multi-turn repository work is not yet locally calibrated |
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

- In one paired conclusion-critical audit on 2026-07-28, Sol/high was more
  concise and calibrated on contract, concurrency, and silent-correctness
  seams. Opus/high was broader on repository archaeology, lifecycle,
  environment, and migration risks but needed stronger deduplication and
  severity calibration. Both initially missed the same tool-to-protocol
  semantic mismatch, so provider diversity did not replace lead verification.
- In one matched implementation task on 2026-07-28, Sonnet/medium,
  Terra/medium, Opus/high, and Sol/high all passed focused tests initially. A
  later external-repository change exposed hard-coded snapshot assertions in
  the Terra and Sol results, while Sonnet and Opus used structural set
  assertions. Opus also found the adjacent renderer-schema seam; Sonnet was
  adequately scoped but repeated an expensive real probe across tests.
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
