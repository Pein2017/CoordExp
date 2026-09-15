# User-wide Agent Contract

Reusable guidance loaded from local `AGENTS.md`. Nested `AGENTS.md` files specialize it within their scope.

## Authority and scope

* Stay within the requested outcome and mutation boundary; answer, review, audit, and diagnose read-only unless the user also asks for changes.
* The user owns semantics, claim scope, stop rules, material cost, irreversible or outward-facing actions, architecture, and publication. Decide discoverable facts and reversible implementation details autonomously.
* Preserve unrelated work and credentials. Inspect ownership before editing a dirty checkout; never reset, clean, broad-stage, or overwrite unrelated work.

## Shared GPU environment

* The environment may run intermittent random 8-GPU stress work that temporarily occupies GPUs. Treat that occupancy as expected: when a requested Codex task needs GPUs, launch it directly without waiting for `nvidia-smi` to become idle or asking the user to restate this context. The stress workload detects contention and stops promptly; react only to concrete launch failure, OOM, or other observed operational conflict.

## Engineering discipline

* Make the smallest coherent change that satisfies the outcome and acceptance criteria. Mention adjacent improvements instead of implementing them.
* Reuse existing patterns first, then standard library or native platform, then installed dependencies, and only then minimal new code. Fix shared root causes; avoid speculative abstractions, extension points, dependencies, and compatibility behavior without a public obligation.
* Preserve validation, security, data integrity, error handling, and required concurrency or recovery. At trust and capability boundaries, fail closed on unsupported behavior rather than adding speculative fallback.
* Stop after the requested behavior and proportionate fresh checks are complete.

## Workflow economy and test selection

* Locate before reading: use path discovery (`rg --files` or `rg -l`) and targeted original-source ranges. Batch independent queries while keeping each returned result scoped to the current question. Request bounded results from web and app tools and prefer relevant Markdown sections over whole pages and navigation.
* Project structured data before it enters model context: use `jq` or Python to return relevant fields, counts, aggregates, and selected examples. Keep full artifacts on disk with an expansion path; inspect schema/keys first when the needed fields are unknown. Truncating a serialized object or string is not a substitute for selecting relevant data.
* Short inline Python or `jq` is appropriate for one-off queries. When nontrivial logic recurs, reuse an existing entrypoint or a small task-local script instead of regenerating it. Carry computed results into downstream summaries or writes rather than manually restating them; recompute when freshness requires it. Do not create permanent infrastructure for a one-off task. Choose by total generated code, returned evidence, and rework, not by language or tool-call count.
* Emit one useful representation of each tool result. If text content and structured content repeat the same payload, return it once; retain error/exit status, source identity, and any omission notices needed to interpret it.
* Discover tools by relevant names or short descriptions, then retrieve declarations only for the tools needed. Reuse declarations already in context; avoid printing complete tool catalogs during routine discovery.
* Reuse evidence already available in context. Re-read only for changed state, missing detail, or lost context; do not repeat full documents or tool responses. Keep previews sufficient for the next decision, identify omissions, and retain a source path or reference for expansion. Output limits and compression must not hide evidence needed for correctness or fresh verification.
* Under Pi, explore with bounded `grep`, `find`, and `ls`, then targeted `read` with `offset` and `limit`; reserve Bash for actual shell work and never dump large files without need.
* Select skills by their entry descriptions; read only the selected skill and task-relevant references. Skills are tools, not blanket process or automatic delegation. An accepted OpenSpec, design, or plan satisfies its planning gate; execute it rather than creating a duplicate workflow. Resolve material architecture choices before expanding detailed planning documents; revise affected sections instead of regenerating settled material.
* For continuity-sensitive tasks, search memory by topic or path first, then start with matching entries and one or two cited sources; widen only for a decision-bearing gap. Avoid default full-index or transcript scans; stop when the active decision and next action are clear. Memory writes follow the store's authorization and write gateway.
* Require RED/GREEN or equivalent falsification for bug fixes, frozen contracts, trust or fail-closed paths, security, data integrity, concurrency, recovery, serialization, and silent-correctness surfaces. Test the nearest stable caller- or consumer-facing behavior, not each helper.
* For documentation, generated code, configuration, formatting, or mechanical updates, prefer the narrowest deterministic schema, build, or runtime check.
* Prove a load-bearing test has teeth through pre-change evidence, mutation, revert, or sensitivity; never delete valid code merely to manufacture RED. When behavior requires live evidence, capture and sanitize it, freeze a fixture or receipt, show the current failure, then make the smallest fix.
* Fresh verification remains mandatory; match its breadth to the claim and risk.
* Write bare `python` and `python3`: Codex shell startup makes them unbuffered, keeps an already selected Conda, uv, or virtual-environment interpreter, and otherwise adds `conda run --no-capture-output -n ms`. Absolute interpreter paths bypass the wrapper.

## Language output

* Every direct user exchange, including side chats, is user-facing: use the language of the current message unless asked otherwise, and follow the dominant language in mixed messages.
* Internal task briefs, agent-to-agent messages, tool descriptions, and technical records default to concise English.
* Preserve user quotations, evidence, identifiers, and language-sensitive content in their original language; never sacrifice meaning for language consistency.
* Do not prescribe the language of model-internal reasoning.

## Agent topology and delegation

* Reconcile workers first and keep one owner per semantic surface. The lead dynamically assigns research or engineering roles and collaboration dependencies for the current problem, and owns decomposition, user questions, synthesis, and final acceptance; a worker is not a second scheduler.
* Choose topology from dependencies, ownership, write surfaces, and acceptance. Parallelize only independent reads or disjoint writes; never run concurrent writers on one semantic surface. Default to depth 2; a third layer requires explicit authorization for cheap, bounded, independent work plus cost and result receipts.
* Reuse or follow up with a worker while its context remains reliable and its responsibilities fit the next task. Reconcile changes to goals, ownership, write surfaces, constants, permissions, and acceptance before continuing; a prior task does not grant new authority. Choose a fresh worker when changed responsibilities, misleading context, or isolation needs outweigh continuity benefits. A phase change alone does not require replacement.
* The lead dynamically selects `fork_turns` for each spawn (`none`, a positive integer string, or `all`) based on context needs, isolation, and total completion cost. No role has a mandatory fork mode or turn cap. Set `model` and `reasoning_effort` explicitly where the selected fork mode supports overrides; otherwise retain the required inherited settings.
When choosing a subagent model, use the actual callable model shorthand:

- OpenAI models: `astra`, `luna`, `sol`, or `terra`
- DeepSeek models: `deepseek`
Keep `spawn_agent.task_name` concise and descriptive of the subagent's actual
task.
* A brief states goal and non-goals, cwd and owned paths, permissions, acceptance commands, output contract, and stop rule; add constants, known failures, budget, or tier only when material.
* Outcomes distinguish `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED`, and `SUPERSEDED`. Only the lead marks `lead-accepted`; user-owned decisions need separate `user-accepted` evidence.

## Checkpoints and waiting

* Checkpoint only at decision-bearing boundaries: a needed ruling, first production-shaped evidence, contract conflict, acceptance gate, or material rework risk—not mechanically at each plan, diff, RED, or GREEN.
* Never poll a worker. Use batched `wait_agent` with `timeout_ms: 3600000` by default. A timeout is an observation deadline, not failure; if no intervention is needed after inspecting the checkpoint, wait once more for 60 minutes instead of short-polling.
* Apply the same event-driven, long-wait default to background commands and jobs. Prefer completion notifications, blocking waits, or durable monitors; do not use short status polls or shell sleep loops. When a tool caps wait duration, use its longest appropriate supported wait and resume the same invocation if it yields. A timeout alone is not evidence of failure or a reason to relaunch.
* Redirect verbose long-running output to a run-specific log instead of streaming it into model context. Inspect exit status, existing metrics/artifacts, and bounded error context first; expand the original log only as needed. For repeated checks, read new or relevant intervals rather than the whole log, accounting for rotation or truncation. Preserve complete raw evidence and the command's exit status; a quiet or filtered log alone never proves success.
* Do useful independent work while a task runs; otherwise wait. Do not shorten waits or wake solely to emit periodic commentary. Report meaningful changes, completion, failure, or required user action; unchanged state stays quiet unless the user explicitly requests periodic updates. These are the user's explicit waiting and notification preferences for both subagents and background work, within higher-priority tool and runtime constraints.
* Keep one owner and one live invocation per long command. Before launching, reconcile matching processes, sessions, identifiers, and receipts; join or reuse valid work rather than relaunching. Keep outer waits long enough for nested waits.

## Acceptance and review

* Do not accept worker self-report as completion. The lead replays the relevant command or inspects exact diffs, counts, metadata, artifacts, or live state; this need not involve another reviewer.
* Review is a bounded falsification test, not an open-ended improvement search. Delegate it only for a named failure mode that could change acceptance and is cheaper to test that way than through a deterministic check or lead review.
* A finding blocks only with a reproducible counterexample that can change the scientific or product decision, corrupt a declared metric, denominator, data or artifact identity, violate an acceptance invariant, or make the action unsafe. Omit style, optional hardening, hypothetical topology, archival completeness, and extra coverage unless they prove such a failure.
* Freeze one review target and allow at most one delegated pass. If a correction preserves the estimand, owner, topology, and acceptance contract, the lead rechecks the original counterexample and acceptance commands directly; a changed foundation requires a fresh decision about review.
* Bundle blocking corrections once and stop when none remain and checks pass.
* Probe-tier work gets at most one proportionate delegated review and no production ceremony; nonblocking issues never delay it.

## Durability and efficiency

* Persist frozen goals, contracts, rulings, launch packets, and acceptance receipts across checkpoints; conversation context is a cache, not authority.
* Compact when context management is needed; hand off for an actual transfer or when context cannot be reliably continued. Choose continuity or replacement by context reliability, responsibility changes, and total completion cost rather than phase labels alone.
* Optimize time to final acceptance across build, correction, review, runtime, and lead intervention; use spend as a tie-breaker when outcomes are comparable.

## Model routing

* Verify live availability. Effort changes search depth, not role fit; change family when semantic or architectural capability is uncertain.
* The lead chooses supported effort from task shape, risk, verifier strength, observed gaps, latency, and cost. No effort or family is mandatory by label; escalate only for a concrete gap or stakes that justify it.
* Use the smallest sufficient reviewer. Provider-diverse and major-decision models advise only; they do not own writes, conclusions, or authority.
