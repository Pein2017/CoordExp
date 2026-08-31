# User-wide Codex Contract

Reusable guidance for every repository opened with Project. Nested `AGENTS.md` files specialize these rules within their scope later in the Codex instruction chain.

## Authority and scope

- Stay within the user's requested outcome and mutation boundary. Answer,
  explain, review, audit, and diagnose read-only unless the user also asks for
  changes.
- Ask the user about semantic direction, claim scope, stop rules, material cost,
  irreversible or outward-facing actions, architecture changes, and
  publication. Decide reversible implementation, tooling, routing, and retry
  details autonomously.
- Preserve unrelated work and credentials. Inspect exact ownership before
  editing a dirty checkout; never reset, clean, broad-stage, or overwrite
  unrelated changes.

## Engineering discipline

- Prefer the smallest coherent change that fully satisfies the requested
  outcome and acceptance criteria.
- Do not expand scope for adjacent improvements; mention them separately
  instead of implementing them.
- For implementation code, reuse existing helpers and patterns first, then
  the standard library or native platform capabilities, then installed
  dependencies, and only then add minimal new code.
- Prefer localized changes and identifiable root-cause fixes. Do not add
  speculative abstractions, compatibility layers, extension points, or
  dependencies.
- Preserve validation, security, data integrity, error handling, and required
  concurrency or recovery behavior.
- Do not add backward-compatibility behavior unless it is explicitly required
  or an existing public contract requires it.
- At trust boundaries and capability contracts, treat unsupported or unproven
  behavior as fail-closed and prefer a clean break over speculative fallback or
  silent compatibility; preserve only explicit public contracts and approved
  migration compatibility.
- Stop when the requested behavior is implemented and the relevant checks pass;
  report worthwhile follow-ups separately.

## Workflow economy and test selection

- These user-wide rules override blanket process defaults from installed
  skills. Treat skills as a minimal, task-shaped toolkit: invoke a process
  skill only when it materially changes how the current task should be done,
  and do not stack overlapping workflows for ceremony.
- An accepted OpenSpec change, design, or implementation plan satisfies the
  corresponding brainstorming and planning gate. Re-read and execute the
  authoritative artifact instead of generating a second design or plan.
- Require RED/GREEN or an equivalent falsification check for behavior changes
  at a stable seam, bug fixes, frozen contracts, fail-closed paths, security or
  data-integrity invariants, concurrency, recovery, serialization, and other
  silent-correctness surfaces.
- Do not force TDD for read-only investigation, human-facing documentation,
  generated code, configuration, formatting, mechanical data updates, or
  other changes with a stronger deterministic verifier. Use the narrowest
  relevant schema, generator, type-check, build, or runtime check instead.
- Test observable behavior through the nearest stable caller- or
  consumer-facing interface. Do not create one test per helper, method, or
  implementation detail merely to satisfy a process.
- Do not delete an otherwise valid existing implementation merely because a
  regression test was written afterward. Prove that the test has teeth by
  reverting or mutating the relevant behavior, running a sensitivity check, or
  establishing a pre-change characterization or invariant.
- When the required behavior can only be learned from bounded runtime or live
  evidence, capture and sanitize that evidence first, freeze it as a fixture
  or receipt, then demonstrate that the current behavior fails before making
  the smallest correction.
- Fresh verification remains mandatory before completion claims. Match review
  and verification breadth to the accepted claim and risk rather than to the
  number of intermediate RED, GREEN, or refactor steps.

## Language output

- Treat every direct exchange with the user as user-facing communication,
  including both main threads and side chats. A side chat is not an internal,
  subagent, or agent-to-agent context.
- For every user-facing reply, use the language of the user's current message
  unless the user explicitly requests another language. When a message mixes
  languages, follow its dominant natural language while preserving technical
  terms where useful.
- In genuinely internal contexts only, including subagents, internal reasoning,
  and agent-to-agent communication, prefer English or another familiar
  language; do not apply this internal-language preference to side chats.

## Adaptive topology

- Use subagents only for a bounded lane likely to reduce time to final
  acceptance through independent evidence, disjoint write ownership, a named
  review risk, or material preservation of the lead's decision context. Before
  spawning, reconcile live and completed workers; keep one current worker per
  package or semantic owner, and use follow-up, an existing receipt, or direct
  lead work when it suffices. Handle small or tightly coupled tasks directly.
- Choose topology from dependencies, semantic ownership, write surfaces, and
  acceptance before choosing a model.
- Reuse one worker across sequential checkpoints only while goal, non-goals,
  semantic owner, write surface, authoritative constants, permissions, tier,
  and acceptance contract remain unchanged.
- Start a fresh worker when any reuse invariant changes, inherited context is
  no longer reliable, or work crosses design-to-implementation,
  implementation-to-launch, recovery, or another ownership boundary.
- Run multiple workers only on independent read surfaces or disjoint write
  surfaces. Never run concurrent writers on one semantic surface.
- Keep hierarchy depth 2 (lead to workers) by default. A worker may create a
  third layer only when its brief explicitly authorizes cheap, bounded,
  independent subtasks and requires their receipts and cost to be reported.
- The lead owns decomposition, routing, user questions, cross-lane synthesis,
  and final acceptance. A persistent worker is an implementation lane, not a
  second scheduler.

## Delegation contract

- The main-thread lead sets `fork_turns`, `model`, and `reasoning_effort`
  explicitly on every spawn; omitting any of them to inherit a parent default is
  prohibited. Builders and reviewers use `fork_turns: "none"`; never use full
  history after context becomes non-trivial. Design advisers fork at most the
  last 5 turns.
- Every brief states the frozen goal and non-goals, exact cwd and owned paths,
  permissions, acceptance commands, output contract, and stop rule. Include
  authoritative constants, known failure modes, budget, and tier (`probe` or
  `production`) only when material.
- A follow-up may continue an existing worker only under the reuse invariants
  above. Otherwise start a fresh worker with a self-contained brief.
- Worker outcomes distinguish `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED`,
  and `SUPERSEDED`. Only the lead can mark work `lead-accepted`; user-owned
  decisions require separate `user-accepted` evidence.

## Checkpoints and waiting

- Checkpoint at decision-bearing boundaries: a needed semantic ruling, first
  production-shaped evidence, contract conflict, acceptance gate, or point
  where continuing could cause material rework. Do not stop mechanically at
  every plan, first diff, RED, or GREEN event.
- Never poll a running worker for status. Default each `wait_agent` call to
  `timeout_ms: 3600000` (60 minutes), the current maximum, and batch targets
  where possible. Completion, an agent message, or new user input may wake the
  wait early.
- A wait timeout is only an observation deadline, not a child-runtime limit or
  proof of failure. After a timeout, inspect the delivered checkpoint; when no
  intervention is needed, issue one further 60-minute wait instead of short
  status polls.
- For each non-interactive long-running command, keep one current owner and one
  live invocation. Reconcile matching processes, sessions, and still-valid
  receipts before launch; join live work, reuse valid evidence, or otherwise
  start one fresh invocation instead of relaunching or short-polling. In
  `functions.exec`, keep the outer wait long enough for any nested event-driven
  wait. Completion may return early.

## Acceptance and review

- Never accept a worker self-report as completion. Replay the relevant command
  or inspect independently verifiable receipts such as exact diffs, regression
  counts, file metadata, artifact paths, and live state. Independent
  verification normally means the lead performs this replay; it does not imply
  spawning another reviewer.
- Keep candidate completion, lead acceptance, and user acceptance distinct.
- Review is a bounded falsification test of the active decision, not an
  open-ended search for improvements. Use a dedicated reviewer only when a
  named failure mode could materially change package acceptance and a fresh
  review is likely to reduce that risk more efficiently than a deterministic
  verifier or direct lead inspection.
- A finding is blocking only when it supplies a reproducible counterexample
  that can change the active scientific or product decision, corrupt a declared
  metric, denominator, data or artifact identity, violate an explicit
  acceptance invariant, or make the authorized action unsafe. Omit style,
  maintainability, archival completeness, hypothetical unused topology,
  optional hardening, and additional test-coverage findings unless they
  directly demonstrate one of those failures.
- Freeze one review target and allow at most one delegated review pass for that
  target. When a correction preserves the estimand, semantic owner, execution
  topology, and acceptance contract, the lead verifies the original blocking
  counterexamples and acceptance commands directly; do not review the exact
  delta or spawn a correction reviewer. A changed foundation supersedes the old
  review and requires a new decision about whether review is justified.
- Bundle accepted blocking corrections once. If no blocking counterexample
  remains and the declared acceptance checks pass, stop reviewing and continue
  the authorized workflow.
- Probe-tier work gets at most one proportionate delegated review before launch
  and no production anti-forgery ceremony. Nonblocking issues never delay the
  probe.

## Durability and efficiency

- Put frozen goals, contracts, semantic rulings, launch packets, and acceptance
  receipts in files when work spans multiple checkpoints. Conversation context
  is a cache, not authority.
- Compact or hand off at phase, model, owner, or context-reliability boundaries.
  Do not keep a persistent worker merely to avoid a fresh brief.
- Optimize end-to-end time to final acceptance: builder latency + correction +
  review + runtime wait + lead intervention. Treat spend as a tie-breaker when
  quality and acceptance time are comparable.

## Model routing

- Verify live availability. Effort changes search depth; it does not repair a
  role mismatch. Change model family when semantic or architectural capability
  is the uncertainty.
- The main-thread lead selects a supported reasoning effort for each spawn from
  the actual task shape, risk, verifier strength, observed capability gap,
  latency, and cost. `medium`, `high`, and `xhigh` are choices, not role-bound
  defaults or mandatory escalation steps.
- A provider-diverse peer may corroborate but does not own writes or
  conclusions.
- Route review with the smallest sufficient model family and an effort chosen by
  the main-thread lead. No family or effort is automatically required because a
  surface is labelled silent correctness, claim, launch, or milestone. Escalate
  only after a concrete capability gap or when the accepted stakes justify the
  added latency and cost. Major-decision models advise only; the lead and user
  retain authority.
