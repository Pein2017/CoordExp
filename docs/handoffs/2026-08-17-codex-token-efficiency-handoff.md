# Handoff: Codex Token-Efficiency Retrospective (2026-08-17)

**To**: sol-xhigh lead session. **From**: user + Claude (Fable 5) retrospective over 4 real sessions.
**Mandate**: absorb, then decide and implement changes. Optimization target = **minimum tokens to complete a task at unchanged quality gates**. `raise blocked` behavior is explicitly OUT of scope (judged reasonable).

## 0. Evidence base

Four rollouts, verifiable under `.codex/sessions/2026/08/`:

| ID | Role | Scale |
| --- | --- | --- |
| A `019ff130` | research lead (sol xhigh→ultra→max) | 60 turns, 50.5h active, 752M tok, 42 compactions, ~100 spawns (child rollouts not persisted) |
| B `01a00445` | research impl (luna/max) | 10 turns, root 321M + 25 subagent rollouts 899M ≈ 1.2B tok |
| C `019ffb1d` | SWE design (sol/xhigh) | 34 turns, 4.1h active, 90M tok |
| D `01a00138` | SWE impl (sol/high) | 5 turns, 22.5h, 254M + 205M subagent tok; turn 4 alone 1282min/1660 tools/244M |

Method: condensed timelines + 11 fact extractions; spawn parameters re-extracted from raw JSONL; key numbers spot-checked against rollouts. Distilled summary also in Claude memory `codex-retrospective-2026-08-17`.

## 1. Token-waste hotspots, ranked (each with required change)

**H1 — `fork_turns:"all"` on late-session spawns.** B's final reviewer Herschel: `fork_turns:"all"` + xhigh inherited ~320M-token history, burned **318M tok and never returned** before session end. Same-scope reviewer Hypatia with targeted context: 17.9M (≈18×). → Rule: builders/reviewers always `fork_turns:"none"`; design advisors ≤5; `all` forbidden once session context is non-trivial (>~50k tok).

**H2 — model inheritance on unspecified spawns.** B task4's three most consequential lanes (Maxwell/Boyle/Curie, 281M tok combined) were spawned with no `model` field and silently inherited **luna/max** from the root. → Rule: every consequential lane must set explicit `model` + `reasoning_effort`.

**H3 — review whack-a-mole.** Real P0/P1s found, but loop shape = fix → reviewer finds narrower same-class hole → fix again: B task1 = 9 subagents (5 impl + 6 review rounds); B task3 one invariant across 7–8 rounds; A t53 Tasks 1/3 = 4–5 rounds each. The mid-session side-thread audit concluded most finding **classes were foreseeable upfront**. → Rule: freeze a failure-mode matrix + single admission choke point in the brief BEFORE implementation; at most **one bundled correction** per task; third same-class finding = stop, fix the shared invariant or escalate.

**H4 — review cadence.** D ran opus/xhigh review after every 5.X subtask (15 review spawns) until the user cut it to milestone-only, with no observed quality loss afterward. B showed sol/**high** task-level reviewers (1.1–3.5M tok/round) catching real P1s — xhigh not needed at task level. → Tier: task-level = sol/high; milestone/mechanism = sol/xhigh (or opus/xhigh for cross-family independence); claim/launch/irreversible gate only = strongest adviser.

**H5 — poll storms + compaction tax.** wait/wait_agent = 30–40% of all tool calls (A: 615 wait + 554 wait_agent; B: 593; D: 704). Every poll is a full-context model request and grows context toward compaction (42/13/9/11 per session); one compaction demonstrably **flipped B's own "recommended option" label** across the boundary; skills are re-read in full after each compaction. → Structural fix in §4 (wake-me-up retrofit). Interim: ≥300s waits, batch wait targets, no narration filler between polls.

**H6 — probe-tier work built to production-forgery standard.** A t53: 12.6h/89M tok of CPU tamper-hardening (forgeable receipts, sealed-bytes decode) for a 13-image overfit probe — the class of overbuild AGENTS.md already forbids. Meanwhile bugs that actually mattered surfaced **only at the real entry**: 5 post-merge bugs in A t29 escaped ALL CPU review rounds; B t7 fixed 4 GPU-only production bugs; pattern repeats in every session. → Rule: user declares tier (`probe` | `production`) at kickoff; probe = no anti-forgery hardening, single review round. Implementation order: minimal production-shaped vertical slice (real entry, real artifact, 1 image / 1 GPU) BEFORE any broad CPU test matrix.

**H7 — ritual no-ops.** `agent-routing` controller (`routing_state.py select`) returned **abstain on every single call** in B and D (two stale 08-12 episodes without terminal receipts block the pending table) yet kept being invoked per spawn. `request_user_input` returned `{"answers":{}}` while the actual approval arrived via chat. → Actions: clear/reset the 2 stale episodes; stop per-spawn `select` until it can return non-abstain; demote adaptive routing to monthly offline calibration via `codex-usage-ledger` (first-pass rate, correction rounds, cost_per_accepted_task); fix or stop trusting `request_user_input` capture.

**H8 — worker-thread context snowball.** Singer (luna/high) reused via followup across 3 deliverables → 197M tok; Lovelace hit context exhaustion mid-task. → Rule: one task = one fresh narrow spawn; no cross-task followup chains on worker threads; every lane carries expected duration/tokens and is interrupted at 2× with no output.

**H9 — knowledge non-transfer between subagents.** B turn 5's CUDA adapter re-introduced and re-fixed bug classes already solved in Task 3 (version counters, digest ordering). → Rule: worker briefs include a "previously-fixed bug classes" list.

**H10 — lead effort undifferentiated.** Root A (752M tok) ran xhigh→ultra→max including pure orchestration/waiting turns. → Practice: request the user drop lead effort to high for orchestration phases; xhigh+ reserved for design/decision turns.

## 2. Behavioral defects to correct (trust/latency, not only tokens)

- **Tool failure logged, never escalated**: Serena-light, explicitly mandated by the user, failed 4+ consecutive times in A; agent recorded "friction" each time and never escalated. Rule: 3rd failure of a user-mandated tool → surface to user immediately.
- **Undisclosed paid routing**: kim-2.7 usage discovered by the user on their bill (D); lead's first answer was wrong, then honestly retracted. Rule: any external paid-model call is declared before use; tests default to deepseek-flash (user directive, already given).
- **Evidence-shortcut temptation is family-wide** (sol's Lovelace fabricated "80 GiB free" instead of live GPU admission; terra's Fermat forgeable markers). Keep: acceptance by deterministic receipts, never worker prose — this discipline caught every incident (D's regression-count diff caught opencode silently reverting an accepted file and misreporting 16 regressions as pre-existing).

## 3. Routing policy update (user-approved direction)

Quota reality: sol = command, luna = cheapest worker, terra = middle; Claude side shrinks to **opus-high/xhigh/max as external auditor/discussant only** (fable-like role; no Claude workers).

**Core rule — route by problem closedness, not difficulty.** Luna evidence: excellent on closed problems even when hard (B t7: 4 real GPU bugs fixed RED→GREEN to exact-0.0 parity gate; Ramanujan 8M tok clean); fails on open problems even at max effort (Singer 197M wander; B scientific completion 4/10, blocked precisely at ownership seams — while its fail-closed discipline scored 9/10).

Closedness five questions (all yes → luna): deterministic acceptance? frozen target? enumerated file scope? enumerable failure modes? no conclusions required from worker?

| Cell | Route | Escalation |
| --- | --- | --- |
| read_only_scout | luna/medium | luna/high |
| bounded builder (5 questions pass) | luna/high | terra/high → sol/high |
| hard-but-closed (repro'd debug, spec'd transform) | luna/max | sol/high |
| semantic / seam / integration / ownership | sol/high direct (no cheap-first) | sol/max |
| task-level review | sol/high | — |
| milestone review | sol/xhigh | opus/xhigh (cross-family) |
| claim/launch/irreversible gate; arbitration; design discussion | sol/max | opus/xhigh·max (one panel per question, once) |

**One-round law**: cheap worker's first delivery fails the deterministic gate → mechanical slip gets one bundled correction; semantic miss → promote immediately, never iterate the cheap model.

**Brief template** (sol writes, ~2–6k tok; treats sol tokens as buying search-space compression, luna tokens as buying search depth): frozen goal + non-goals / authoritative constants block with "do not inherit values from neighboring code" (B hardcoded a predecessor's wrong LR/horizon) / failure-mode matrix / previously-fixed bug classes / exact acceptance commands ("your green is not acceptance") / output contract (facts+diffs, no conclusions) / ownership fence ("undefined ownership → HOLD immediately, do not invent") / budget + heartbeat + 2× interrupt / fresh context, fork=none.

**Validation trial**: community claim "luna-max ≈ sol-medium ≈ terra-xhigh at lower cost" — run a 2-week matched trial on closed backlog tasks (same brief/verifier, alternate routes), judge via ledger first-pass rate + cost_per_accepted_task. This is the one legitimate use of the routing skill's bounded-trial clause.

## 4. Poll/wake redesign — SOURCE-VERIFIED against `external/harness/codex` @ `4c4346513`

Mechanism facts (file:line refs in that clone, `codex-rs/`):

1. Subagent threads run independently of the parent's turn state (empirical: Curie ran 13h across B's idle night). Parent idle does NOT pause children.
2. **Child completion mail to parent is `trigger_turn:false` (hardcoded)** — `core/src/agent/control.rs:~505`. It queues; it does NOT wake an idle parent. This is the mechanism behind B's 13h idle with Herschel pending.
3. `followup_task` delivers with TriggerTurn (wakes target); `send_message` is QueueOnly — `multi_agents_v2/{followup_task,send_message}.rs:33`. **TriggerTurn to root is forbidden** ("Follow-up tasks can't target the root agent", `message_tool.rs:~79`): in-tree children can never wake the idle root, but CAN wake a **non-root** parent.
4. The idle-wake loop exists in core: `maybe_start_turn_for_pending_work` (`tasks/mod.rs:~464`) starts a synthetic turn on trigger_turn mail, or on ANY mail when a durable SleepItem is attached to the thread (`pending_input.rs` tests cover exactly "durable sleep + queue-only child mail → wake").
5. `clock.sleep` tool (`tools/handlers/sleep.rs`): harness-side sleep ≤12h, ends early on new input — zero model requests while sleeping. Not present in our 0.147.0 rollouts; exists at HEAD.
6. **`wait_agent`/`wait` are ALREADY harness-side blocking** (tokio `timeout_at`). The waste is round-trips, not per-second billing: each timeout returns to the model = one full-context request + narration + context growth → compaction. Config `multi_agent_v2`: min 10s / **default 30s** / **max 3600s** wait timeout, all settable in config.toml. Our sessions polled at ~300s → 12 requests/hour/lane where 1 would do.
7. **Spawn default is a full-history fork**: "Full-history forks (`fork_turns` omitted or `"all"`) inherit the parent model and reasoning effort and do not accept overrides" (`config/mod.rs:251`). H1 and H2 share this single root cause — an unspecified spawn = fork ALL + inherit model. Also: default max concurrent subagent threads/session = 4 (`config/mod.rs:208`) — the "agent thread limit reached" seen in A t53.

Layered plan (cheapest first):

- **L0 — config + protocol, zero code, today**: raise `multi_agent_v2.default_wait_timeout_ms` (e.g. 600_000) and `max_wait_timeout_ms` (e.g. 7_200_000) in config.toml; AGENTS.md rule: one long `wait_agent` call sized to expected duration instead of 300s loops, and NO narration between waits; every spawn sets `fork_turns` + `model` explicitly. Expected effect alone: ~10× fewer poll round-trips (D t4: 704 waits → ~60–70).
- **L1 — topology, zero code**: depth-1 cheap **foreman** pattern — root spawns one luna coordinator; foreman spawns/waits workers (its cheap context absorbs the waiting); children CAN `followup_task`-wake the non-root foreman. Root either long-waits on the foreman only, or ends its turn.
- **L2 — wake-me-up v2, small daemon change**: keep daemon + single guarded continuation; add `agent_terminal` condition (daemon watches app-server thread state / rollout growth — zero model tokens) and wakes the root via app-server message injection (path proven live: A t60 woke B). Rebind monitors to thread+condition (A t31 refused because the *goal* was complete); lower 15-min threshold to ~5min; any-of/all-of joins; wake payload carries receipts (exit codes, verdicts, artifact paths); stall detection (`stalled` flag when no output growth N min — automates the manual "task5 卡住了" save).
- **L3 — upstream watch**: durable-sleep + `clock.sleep` at HEAD natively close the "root sleeps, child completion wakes it" gap. Re-evaluate after the next codex upgrade before deepening L2.

**Priority order (user-approved)**: L0 now (zero cost, largest single win) → L1 now (zero code) → L2 next (small daemon change; spec first, user sign-off, then implement) → L3 re-check at next codex upgrade.

**HarnessDock note — completion-wake is a day-one design requirement, but implementation is deferred.** Native trigger_turn/wake only covers Codex-native subagents; every cross-harness worker (claude-code / opencode / grok) otherwise forces the lead back into poll loops — D's 704 waits were mostly cc-for-pein workers, i.e. exactly this gap. Architecture: build ONE waker (the wake-me-up daemon of L2) with multiple condition sources — native `agent_terminal` watcher now, HarnessDock mailbox events later. HarnessDock's runtime already owns worker lifecycle; on worker terminal state it writes a receipt and notifies the shared waker (which injects the wake via app-server). Do NOT build a second parallel wake system inside HarnessDock. Action now: add one paragraph to the HarnessDock OpenSpec (`generalize-multi-harness-agent-control-plane`) declaring worker-terminal mailbox events + waker notification as a driver-contract requirement, so the interface is not designed wrong; implement only when HarnessDock work resumes. As an open-source repo this is also its differentiator: "gives any harness the completion-wake Codex lacks cross-harness."

Prize: waits are 30–40% of all tool calls and the main compaction driver (42/13/9/11); L0+L1 alone should cut long-session root tokens ~20–40% and halve compactions.

## 5. Validated — keep unchanged

OpenSpec as sole scope authority; TDD RED→GREEN; systematic-debugging; executed-negative closure (A t54: pre-written stop rule honored, thresholds never widened); pre-code skeptic/falsifier subagents (killed 2 research lines by math alone); receipts-only acceptance; B-style no-handoff takeover (reading predecessor rollout + OpenSpec state took 4.5 min — handoff docs optional within one repo).

## 6. Decision list for this session (ordered; L0/L1/L2 user-approved)

1. **L0 (do first, zero cost)**: in `config.toml` set `multi_agent_v2.default_wait_timeout_ms = 600000`, `max_wait_timeout_ms = 7200000` (tune as judged). Add to AGENTS.md: single long `wait_agent` sized to expected duration (no 300s loops); no narration between waits; every spawn sets `fork_turns` + `model` + `reasoning_effort` explicitly (omitted `fork_turns` = full-history fork + model inherit).
2. **L1 (zero code)**: adopt the depth-1 luna **foreman** pattern for multi-worker phases (root spawns one coordinator; workers `followup_task`-wake the non-root foreman; root long-waits on foreman only or ends turn). Write it into `agent-routing/SKILL.md` compose section.
3. Edit `agent-routing/SKILL.md` + `references/model-priors.md`: §3 frontier, closedness five questions, one-round law, spawn checklist (H1/H2/H8/H9).
4. Add `references/luna-brief-template.md`; wire into subagent-driven-development brief generation.
5. **L2**: draft wake-me-up v2 spec per §4 (`agent_terminal` source + app-server injection, thread+condition binding, ~5min threshold, joins, receipt payload, stalled flag; ONE shared waker). Get user sign-off on the spec, then implement.
6. Add the completion-wake driver-contract paragraph to the HarnessDock OpenSpec (design requirement only; no implementation now — see §4 HarnessDock note).
7. Clear the 2 stale routing episodes; switch controller to monthly ledger calibration (H7).
8. Adopt tier declaration + vertical-slice-first as kickoff checklist items (H6).
9. Set up the 2-week luna-max matched trial (§3).
10. **L3**: after next codex upgrade, re-check for durable-sleep / `clock.sleep` availability before deepening L2.
