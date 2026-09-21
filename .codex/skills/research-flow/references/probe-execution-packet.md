# Frozen Probe Execution Packet

Use this only to implement or run an already authorized research probe. The
owning research unit keeps the question, contrast, population, claim and stop
rule; this packet grants no launch, interpretation or scope-expansion authority.

## Admission

Before writing code or calling a model, bind the worker to:

- the active-contract sentence and authoritative unit state;
- exact checkout, checkpoint/config and input artifact identities;
- intervention boundary, population and denominator;
- the real consumer and its output schema, including valid empty, `null`, EOS,
  unavailable and `HOLD` states;
- output root, owner, attempt/repair budget and stop condition.

Reuse the owning unit or existing launch packet instead of duplicating it. If a
decision-bearing field is missing or conflicting, return `NEEDS_CONTEXT` or
`HOLD` to the lead; do not infer research meaning from convenient examples.

## Implement

State the source-to-transform-to-output relationship and the nearest plausible
wrong result. Read the task's domain execution skill, such as
`qwen3-vl-execution`, for model-specific behavior. Preserve the consumer's
schema for nonempty and empty cases, and include a check that distinguishes the
intended interpretation from that wrong result.

Do not start a reducer or dependent lane until its upstream manifest and schema
are frozen. When a peer changes a bound input, either rebind the dependent work
explicitly or keep it on `HOLD`.

## Qualify And Run

- Before the first model call, preserve or immutably bind the producer source,
  input manifest and effective runtime identity.
- Exercise the smallest real entry that reaches the unresolved model, artifact
  or consumer seam. Use `full-pipeline-smoke` only when that real seam is the
  named risk, not as another approval stage.
- Include one positive case and one mutation, corruption or boundary case that
  would fail under the nearest wrong implementation.
- Preserve failed attempts. A repair uses a new output path, records the changed
  producer identity and counts against the frozen attempt budget.
- Keep one live invocation per owned unit and publish immutable artifacts plus a
  manifest sufficient for fresh readback.

## Return

Report `candidate`, `NEEDS_CONTEXT`, `HOLD` or `BLOCKED`; changed paths; producer
and input identities; commands and checks; attempt and resource counters; output
artifacts; and the exact unresolved gap. Keep scientific interpretation and
acceptance with the owning lead.
