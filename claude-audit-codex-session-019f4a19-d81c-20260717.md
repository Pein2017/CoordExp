# Independent Third-Party Review — Codex Session `019f4a19-d81c-75a2-84b0-2c20379e686e`

- **Reviewer**: Claude (Fable 5), acting as independent research PI / mechanistic-interpretability researcher / research-systems auditor
- **Date**: 2026-07-17
- **Session reviewed**: `019f4a19-d81c-75a2-84b0-2c20379e686e` (2026-07-10T03:37Z → 2026-07-17T03:08Z, still active at review time; forked from `019f49c4-7af3`)
- **Transcript**: `/data/CoordExp/.codex/sessions/2026/07/10/rollout-2026-07-10T03-37-15-019f4a19-d81c-75a2-84b0-2c20379e686e.jsonl` (264.8 MB, 78,178 records)
- **Workspace**: `/data/CoordExp/.worktrees/research-probes` (dense-enumeration hub) plus the earlier PVCI phase merged from two now-deleted `.codex/worktrees/` roots
- **Prior reviews consumed by the session itself** (not duplicated here): `claude-audit-codex-session-019f4a19.md` and `codex-audit.md` (07-12, PVCI implementation), `session-review-claude.md` / `session-review-codex.md` (07-14), `agent_review/{fable,fable-continue,pro-new,pro-continue}.md` (07-16). This review reads the complete session plus live repository and artifacts, and focuses on what those point-in-time reviews could not see: the whole arc and the artifact-level ground truth.

---

## 1. Executive Verdict

This is one of the most disciplined AI-led research sessions I have audited, and simultaneously one of the most expensive per unit of decision-relevant knowledge. Both halves matter.

**Scientific integrity: strong, and verified.** Every conclusion-owning artifact I spot-checked reproduces exactly — SHA-256 receipts match, headline counts recompute from raw receipts (the 11/32-vs-21/32 pizza/cup fragmentation, the image-`7818` donor `IoU = 0.800259` and mean release `+1.084567`, the `63/64` x1→x2 transport). Result documents systematically bound their claims, negative results are closed honestly (the expensive PVCI bridge terminated as `shortcut_or_row_prior`; the count-balanced soft-cross operator closed as `no_adjudication` when its own matched controls failed), and demoted claims are preserved rather than deleted. The bfloat16/batch-shape confound discovery — that physical batch layout alone flips low-margin coordinate decisions, removed under full-model float32 — is a genuinely conclusion-protecting contribution that most research groups would have missed and shipped.

**Scientific strategy: increasingly at risk of building a mechanistic theory of idiosyncrasies.** After 33+ executed units across two investigations, nearly every "supported" mechanism claim rests on 1–2 hand-picked images and exact states (P56/image-12576, image-139, image-7574, image-7818, one fork, one chair field). No claim has a prevalence tier. Two standing controls that the program itself identified — a random-ordering-trained adapter and the base no-adapter model — have never been run, despite being the cheapest way to separate "object commit mechanism" from "geometry-sorted serialization grammar learned from the training data" and "adapter-induced" from "pretrained" behavior. The highest-leverage hypothesis for the actual goal (incomplete dense labels teach conservative termination) has been listed as "plausible / possibly dominant" since day one and has never been directly tested, while ten-plus micro-causal probes were.

**Efficiency: self-aware but still heavy.** Main-thread consumption was ~2.62B input tokens (98.4% cache-served; ~42M uncached) and 4.0M output tokens, running at `gpt-5.6-sol ultra` reasoning for 171 of 258 turns, across 109 context compactions, 22 rollbacks, and 8 days — 64% of a weekly Pro rate limit. The visible thread is the tip: 346 child subagent rollouts attribute to this session and sum to **~856M output tokens, 213× the parent** (measured from the children's own final usage counters; 93% of all Codex output on this machine for the week). The goal tracker the user sees reported per-goal figures of 0.8–7.3M — the same order-of-magnitude cost invisibility the 07-12 audit flagged for PVCI persists for the whole week. The 07-14 workflow reforms measurably lightened the loop (one-day multi-unit cadence on 07-15), but the artifact style remains disproportionate: a 45KB pre-registered `unit.md` and a 76KB single-use runner for a two-target probe is the *reformed* weight.

The distilled scientific output is real: the week converted "the model has weak instance binding" into a specific, falsifiable, staged decision model (mode fragmentation at a fixed state → phrase-geometry transaction gate → phase-separated semantic/geometry routing → late pre-`x1` owner-basin collapse → autoregressive geometry transport → traversal-state-but-not-coverage cross-row update). That is a publishable skeleton. What it is missing is exactly what the current trajectory keeps deferring: population evidence, ordering/base-model controls, and one demonstration that any discovered mechanism can be moved by training.

---

## 2. What Was Actually Done (Session Reconstruction)

Timeline reconstructed from 194 user messages, 1,906 agent messages, goal records, and the research tree. Token figures are main-thread cumulative; subagent traffic is additional.

| Phase | Dates | Content | Outcome |
|---|---|---|---|
| Handoff + framing | 07-10 | Continuation from session `019f2d8b` via handoff doc; ~280KB ChatGPT synthesis pasted 3×; grill-me; user redirects from "final architecture" to experimentalist loop; 5-hypothesis recap after user reports being "lost and suspicious" | Research posture reset; PVCI direction chosen |
| PVCI causal proposal bridge | 07-10 → 07-12 | Doc-level contract → implementation → fail-closed gates → matched DoRA training (A/B/C arms, no D) on 8×A100 → 16-cell own-prefix panel, 320 held-out images | **Well-controlled negative**: held-out representation gate passes (29 checks), but behavioral bridge effect reproduced by another-image and token-permutation controls → `shortcut_or_row_prior`; safety non-inferiority failed. Goal segment consumed ~7.3M tokens; user: "ages… unbearable efficiency" |
| Meta round 1 | 07-12 → 07-13 | Two external audits ranked; principles distilled into skills; infra synced from `CoordExp-swift`; two temp worktrees harvested and deleted; `research/` "brain center" graph established; new worktree `research-probes` | Research-knowledge architecture (compass/overview/index/units) created |
| Spatial scope & history | 07-13 → 07-14 | 90KB ChatGPT "Gap analysis" pasted 7×; ~15-message 同意 design chain; 3 independent spec reviews ranked; masked/tile/bagging/cumulative-prefix panel; 51-file, ~33.5k-line initial commit; 14 sequential fixes | Masked reset < equal-call bagging; cumulative accepted-row prefix harmful; tiling worst. User's visual review: "几乎是完美答卷" |
| Meta round 2 | 07-14 | Session-wide retrospective (2 external verdicts ranked); research-flow reforms: shrink contracts, 4–8-sample case studies, visual review over metrics, flexible main-thread authority; model-routing race (luna/sol tiers) | Reforms encoded in AGENTS.md/skills; sampled-rescue unit authorized |
| Fixed-state micro-causal chain | 07-14 → 07-16 | Sampled-rescue P56 (11/32 pizza vs 21/32 cup; terminal state 0/32); phrase-geometry factorial; two visual-support counterfactuals; bfloat16/batch confound discovered → 4-unit numerics investigation → float32 policy; fixed-encoding spatial-eligibility family (hard/soft/query-phase/cross-region/count-balanced); residual portability chain → image-7818 one-way `x1` basin switch | The week's core mechanism ladder; several operators honestly closed (`no_adjudication`, gate failures) |
| Human-audited review | 07-16 | Purpose-built HTML annotation tool (tmux-served); user annotated all 119 unmatched bagging candidates with per-box comments | 108/119 semantic-exact vs 28/119 geometry-acceptable; y2 bottom under-coverage 35/39; **but 0/119 entity references captured → planned genealogy waves dead; unit redirected** |
| Complete-box factorial | 07-16 → 07-17 | Fixed-prefix teacher-forced box lattice + progressive coordinate release on fork (part vs whole) and dense-chair states | `x1→x2` transport 63/64; fork stays part-like 27/32 despite whole-object x1,y1 forcing; emitted geometry flips next-row category (chair 16/16 vs person) |
| Current frontier | 07-17 | Belief-update synthesis; cross-row influence-horizon question (direct vs mediated vs insertion); grill-me; 898-line unit spec drafted; user approved 03:08 | Planned: object-specific geometry transport + decision phase + cross-row horizon, 4 audited cases |

Cost accounting (main thread): 2,616,796,949 input tokens (2,574,402,304 cached), 4,010,216 output (1,183,409 reasoning). Fleet: ~530 `spawn_agent` calls (peaks: 145 on 07-13, 114 on 07-15, 113 on 07-11), 2,313 `wait_agent` + 1,784 `wait` calls, 620 `send_message`, 64 `interrupt_agent`. **Fleet cost measured from child rollouts: 346 children whose session meta references this parent sum to 856,388,107 output tokens (≈213× the parent; average ≈2.5M output tokens per child lane).** 8,504 shell executions, 708 patch applications. 168 session-meta records (reconnects — matching repeated "连接断开了,请继续"). Runtime artifacts: 13GB under `outputs/research/qwen3-vl-dense-enumeration/` (21 experiment roots) plus 13GB PVCI-era `outputs/probes/`.

---

## 3. Evidence Integrity Audit (Sampled, All Passed)

I independently verified, from disk, not from the docs:

1. **Receipts exist and hash-match.** All three conclusion-owning receipts named in the weekly report reproduce their published SHA-256 digests exactly (`6148fa75…`, `2046d578…`, `73805bc2…`).
2. **Headline counts recompute from raw receipts.** Aggregating `call_bundles[].first_action` across the four rescue-entry receipt files yields exactly 21 cup + 11 pizza first actions in 32 sampled draws, in two tight, disjoint geometric clusters — the fragmentation claim is not an artifact of parsing or selective reporting.
3. **Image-7818 numbers trace end-to-end.** `owner_attribution.donor_iou = 0.8002593939…`, `replacement_coordinate_release = 1.0845668…` (= mean of the four per-coordinate deltas whose x1 component is `+4.335076`), owner switch `664730 → 661523`, and the layer-13 negative control — all present in the receipt with matching values and explicit no-op self-replacement gates at zero drift.
4. **Docs volunteer their own weaknesses.** The human-review unit reports its own failed annotation gate (`119 approved / 0 entity references`) instead of quietly using the data; the complete-box unit explicitly labels its 63/64 result as batch-4 evidence with a 4/4 batch-1 greedy control; near-ties are declared near-ties.
5. **Session-to-doc consistency.** The 07-17 in-chat synthesis matches the frozen `results.md` and compass rows; an in-flight transposed-table error was caught by the session's own recheck plus an independent audit before commit.
6. **Repo hygiene at review time.** `research-probes` is clean except the planned 07-17 unit (untracked by design) and four modified index docs; the last commits map 1:1 to closed units.

I found no evidence of fabricated results, silently overwritten failures, or claims exceeding their receipts in the sampled set. The provenance chain (transcript → unit.md → runner + tests → receipts → results.md → compass/weekly) is genuinely navigable — rare and valuable.

Minor defects noted: the human-review results table lists a "five-image purposive cohort" but enumerates four image rows (the fifth image evidently contributed zero unmatched candidates — say so explicitly); `progress/`-era layer-17/21 claims are correctly quarantined as historical, but the weekly report's "Historical Late-Middle Language-Layer Evidence" section is the only place where non-replicated ms-swift-era numbers still do argumentative work.

---

## 4. Scientific Assessment

### 4.1 What is genuinely strong

- **Claim discipline.** The Supported / Ruled Out / Unresolved / Not Claimed structure, demoted-claims register, and per-unit stop rules are executed, not performative. Example: the image-7818 result — the single strongest causal finding of the week — is voluntarily narrowed from "geometry-state portability" to "one-way first-coordinate basin switch, 99.9% of effect at x1" by the session's own analysis.
- **Controls at the right layer.** Wrong-object / another-image / token-permutation / position-only / norm-matched controls killed the PVCI bridge's flattering interpretation. No-op parity gates, self-replacement drift checks, and paired seeds are standard practice by 07-15.
- **The numerics catch.** Discovering that bfloat16 physical batch shape flips low-margin coordinate argmaxes — then spending four bounded units separating "real broad branch (image-7574)" from "1–2-pixel micro-jitter", localizing divergence after `get_image_features`, and adopting a float32-for-conclusion-critical-probes policy — is exactly what "runtime assurance only where it protects a conclusion" should look like. Most VLM-interpretability work never checks this and is contaminated by it.
- **Honest negatives that close branches.** `shortcut_or_row_prior`, `no_adjudication_close_count_balanced_soft_cross_operator`, the masked-policy rejection, and the fork's refusal to extend to whole-object extent are all preserved as load-bearing negative results.
- **Real intellectual progress.** The staged decision model (fragmented next-object modes → phrase-geometry transaction gate → phase-separated earlier-vs-row-query computation → late pre-x1 basin → x1-conditioned transport → traversal-inertia-not-coverage) is a much sharper object than 07-10's "binding is the root cause" intuition. The 07-17 reframing — "the model may not lack a state carrier; it lacks training constraints that make that carrier mean unique physical coverage" — is a genuinely good hypothesis and is more original than the ledger/slot framings it displaced.

### 4.2 Core weaknesses (severity-ranked)

**W1 — The program has an existence-proof engine and no prevalence engine.**
Every mechanism claim in the compass belief register is bounded to 1–2 images and often one exact prefix state. That is correct labeling, but the *portfolio* is unbalanced: ~14 consecutive units produced point evidence, zero units produced population evidence. The candidate paper thesis ("repeated sampling exposes competing valid modes at one exact state; trajectory determines availability") already commits to a distributional claim — "how often does a greedy-missed object have stable support at some reachable state?" is answerable with an automated 32–64-image screen using existing runners, no human annotation required for a first pass. Without that tier, reviewers can dismiss the entire atlas as anecdotes, and — worse — the program itself cannot rank which mechanism matters for recall at scale. The compass's own update rule ("replicate only when a route decision requires prevalence") has become a way to never require prevalence.

**W2 — Two standing controls were identified and never run.**
(a) *Random-ordering adapter.* The 07-13 unit explicitly states attribution to training order "requires a random-order-trained checkpoint", and the user said on 07-11 they would train one in parallel. No subsequent unit uses it. This control cuts through the single largest interpretive ambiguity that recurs in *five* separate units: geometry-sorted successor grammar vs object commit (the P56 cup→right-cup transition, the phrase-geometry gate, the x1→x2 transport, the traversal-rank alternatives all carry this caveat). One afternoon of paired probes on the random-order adapter would either collapse or dramatically strengthen half the belief register.
(b) *Base model without the adapter.* "Base model" appears in the artifacts only as provenance metadata. Whether mode fragmentation, part-extent basins, and conservative stopping are *created by DoRA fine-tuning on incomplete geometry-sorted labels* or are *pretrained Qwen3-VL grounding behavior* is unknown — yet the north star explicitly requires preserving pretrained capability, and the treatment options differ completely (data/objective fixes vs architecture support). This is the cheapest untested discriminator in the program.

**W3 — Discriminator treadmill; data- and training-side hypotheses are starved.**
Each closed unit names "exactly one next discriminator", which is good hygiene locally but has produced a globally unbounded recursion of micro-causal probes: since PVCI closed (07-12), every unit has been an inference-time intervention on the same frozen step-4887 checkpoint. Meanwhile: the annotation-completeness hypothesis (compass: "plausible", listed alternative: "dominant cause of conservative termination") has a concrete stated discriminator — exhaustive labels vs original vs controlled thinning on the same images — that is now *cheaper than ever* (the HTML annotation tool exists, the user demonstrated willingness to annotate 119 boxes with comments) and has been deferred for the entire week. The 256-image training-screen gate is well-designed but its precondition structure means it can be deferred indefinitely by exactly the kind of unresolved micro-questions the treadmill keeps generating. When the user asked on 07-16 for "direction and stopping criteria", the answer produced… the next probe. A program-level stop rule is missing: something like *"a mechanism thread may spawn at most K consecutive n≤4 units before it must either produce a prevalence estimate, consume a standing control, or hand a testable lever to the training screen."*

**W4 — Case sourcing has a selection-effect structure that the language occasionally outruns.**
Cases enter the microscope conditioned on being interesting failures (bagging-unmatched candidates, rescue events, images with qualified donors after screening 6 → 3 → 1). That is fine for existence claims, and the docs mostly respect it — but compass summary lines like "current-row visual reads strongly route geometry and spatial ownership" quietly generalize from ≤6 cases of which 1 completed a full owner switch, 1 was phase-specific, and 3 were one-sided/destructive. Similarly "at sufficient resolution, hard routing can compile a tight instance-owned geometry path" rests on 3 images surviving a screen designed to find them. The unit docs are careful; the register prose is where enthusiasm leaks in. Since the compass is the document future work (and GPT-Pro) will actually read, its rows should carry the denominator, not just the direction.

**W5 — Statistics are honest but under-powered for the directional weight they now carry.**
Counts are reported plainly (good), but: 16/16 endpoint-family results at temperature 0.4 have a 95% binomial lower bound of ~79% — fine for "strong effect at this state", not for the flat language "target-like 16/16" invites downstream; nothing in the program computes even this. Multiple small panels per unit (5 hypotheses × several arms) with no note on how many comparisons were run means some "one of five cases completed a switch" results are compatible with base-rate noise under hard, destructive interventions. A house rule of exact binomial CIs on every count and a per-unit comparison budget would cost nothing and materially harden the record.

**W6 — One person-day of the user's highest-value labor was partially wasted by a schema omission.**
The 119-candidate manual review is the program's only human-verified evidence and directly seeded the current (good) coordinate-coherence direction. But the annotation tool shipped without the entity-reference field that the unit's own Wave 1+ gates required, so the planned genealogy/coverage waves died and the export is qualitatively rich but quantitatively orphaned (no owner consolidation, no unique-entity recall — exactly the covered-set questions the program cares most about). The unit's redirection was the right recovery, but this was a foreseeable contract failure: the unit.md predeclared the entity-ledger gate; the tool was never checked against it before the user spent the evening annotating. The reformed "smoke before infrastructure" rule was applied to model runners but not to the human-facing instrument.

**W7 — Interpretation debt around "hard" interventions.**
Several positive results use interventions with no natural scale (hard key exclusion, full residual replacement). The program partially covered this (soft-bias dose response showed hard endpoints are not smooth continuations; matched-control failures closed the soft-cross operator), but the asymmetric chimera and the one-way portability results remain interpretable as artifacts of abrupt phase switching — the compass says so, and then the discriminator queue moves elsewhere. If late-state portability is to anchor the eventual training story ("teach the model to synthesize the privileged state"), the naturalness question is not optional: does *any* naturally occurring rollout state resemble the hard-routed donor state? A cheap representational-similarity screen between donor states and sampled-rescue states would connect the two strongest threads of the week and has not been proposed.

### 4.3 Where the direction should look next (reviewer's judgment)

The 07-17 planned unit (transport specificity + decision phase + cross-row horizon with clamping) is a good design — the direct/mediated/insertion decomposition of cross-row influence is the right causal frame, and the synthetic-vs-real x1 cue contrast is the single best next micro-experiment in the queue. My challenge is not to its content but to its exclusivity: run it, and in the same wave spend the marginal GPU-hours on (i) the random-order-adapter replication of the P56 transition panel, (ii) a base-model replication of the fragmentation probe, and (iii) the first automated prevalence screen. All three reuse existing runners nearly unchanged. If the transport/horizon results reproduce on the random-order adapter, "traversal grammar" loses; if they vanish, half the atlas needs relabeling *before* more depth is added to it.

---

## 5. Implementation & Workflow Assessment

### 5.1 What the 07-14 reforms demonstrably fixed

Before: PVCI's `unit.md` alone is 77.7KB; the 07-13 spatial-scope unit opened with a 51-file, ~33.5k-line commit and needed 14 sequential fixes before first results; audit-gated everything; "It's been ages… unbearable efficiency" (user, 07-12).
After: 07-15 closed *nine* bounded units in one day on 8×A100 with paired seeds and clean stop rules; failures were classified rather than re-audited; later smokes caught real assumptions early (request-identity comparator bug, non-homogeneous predecessor batch, GT-vs-realized donor row conflation) and were repaired without unit-scale rework. The reforms were real and the retrospective that produced them (user's #1–#4 critique: premature generalization, excessive runtime assurance, audit perfectionism) was correctly diagnosed and encoded into `AGENTS.md`/skills.

### 5.2 What is still heavy

- **Spec and runner weight is still an order of magnitude beyond the evidence weight.** The *reformed* pattern is a 45KB/898-line pre-registered unit.md, a 50–76KB bespoke runner, and a per-unit test file — for probes whose scientific payload is a handful of counts on ≤4 cases. 35 runner scripts (~1.2MB) now exist with limited cross-reuse (adjacent fixed-encoding runners share only ~100 identical lines). One-time code is allowed to be ugly, but this isn't ugly-cheap, it's polished-expensive: each runner re-implements prompt reconstruction, receipt plumbing, gate logic, and attestation. A single `research_probe` harness owning (fixture load → prompt hash check → decode policy → receipt emission) with ~5–10KB experiment-specific intervention modules would cut both implementation tokens and audit surface by well over half. The 07-13 ambition to "design reusable modules" produced `src/analysis/` subpackages for three units and then reverted to bespoke runners.
- **Fleet economics dominate and are invisible at the steering wheel.** The parent thread's 4.0M output tokens are 0.5% of the session's true ~860M fleet output. `sol ultra` for 66% of parent turns, ~170 audit-lane subagents, and triple-review ranking rituals for specs and even external essays is a cost profile justified only for conclusion-critical junctures — and because per-goal trackers report only parent-side numbers, nobody in the loop (user or lead agent) sees the real burn rate when deciding whether to spawn another lane. The session's own routing doc (weekly report) already prescribes cheaper tiers for most lanes; the routing decision point lacks the cost signal to enforce it. A per-goal fleet-token ledger (sum of child rollout counters, refreshed at each goal update) would make R3/R7 self-policing.
- **Context economics.** The same 90KB ChatGPT conversation entered the context 7 times on 07-13 (~630KB, likely resend/UI retries), and a ~205KB one 3× on 07-10. With a 258k-token window and 109 compactions, these pastes are a material driver of compaction churn — and compaction is when provenance and nuance get lossy. Attachments-by-path (which the harness already supports) or a one-line "already ingested, skipping" guard would remove this class entirely.
- **Mega-thread architecture.** One 265MB 8-day thread with 168 reconnects and 109 compactions *worked* here only because the docs-as-external-memory design is good: after each compaction the compass/units carry the state. But the failure mode showed up anyway (the transposed-table error, conclusions drafted from compacted memory then fixed against artifacts). Per-goal forked sessions seeded from the compass + latest unit — the pattern the user already uses for external reviews — would give the same continuity with far less compaction risk, and would make future session audits tractable (this one required condensing 265MB).

### 5.3 The multi-model review economy

The session consumed at least ten external reviews (ChatGPT syntheses, GPT-Pro packages, Fable essays, Claude/Codex audits) and ran an explicit subagent "model race". Verdict: the *audits with artifact access* earned their cost (the 07-12 pair changed the workflow; the fixed-point audits caught real errors, including a conclusion-wording overreach on batch scope). The *essay-ranking rituals* mostly returned agreement plus vocabulary; their conclusions ("Luna for bounded implementation, Sol for judgment") were adopted from n≈2 anecdotes. Model-routing evidence remains soft; the 07-16 routing policy is reasonable but should be labeled as prior, not finding.

### 5.4 Specific process defects worth fixing

1. Human-facing instruments must pass the same pre-registration gate as runners (W6). Fix the annotation tool schema (entity reference, per-boundary structured codes) before the next human batch; the current export cannot be retrofitted.
2. `AgentGuard` blocked deletion of one stray config (`coordinate_release_max5.yaml`), correctly reported for manual cleanup — but the same run's guard posture allows 13GB+13GB artifact accumulation with no retention policy. Define per-unit artifact retention (receipts + summaries forever; token-level dumps for conclusion-owning runs only).
3. Duplicated user messages (same text 2–7×) appear throughout the transcript; where these were prompts, some triggered duplicated agent work before dedup. A cheap idempotency check ("this exact message was just processed") would help.
4. The weekly report's own packaging warning ("substantial parts of the tree are untracked") was accurate on 07-16 and is *stale* today — commits landed 07-17. Regenerate before shipping to GPT-Pro so the external reviewer isn't warned off files that are now tracked.

---

## 6. Novelty and Publication Readiness

Positioned against what I know of the literature (through early 2026): the individual ingredients — exposure bias in autoregressive structured prediction, sampling-recovers-what-greedy-misses, attention-is-not-causation, activation patching — are established. The *specific composite* this program has built is, to my knowledge, novel and interesting:

1. **Fixed-state next-object mode fragmentation in an AR detector**, with exact-state replay showing greedy mode concentration (not missing perception) as a recall mechanism — cleaner than the usual bagging-gap observation because the state is held byte-identical.
2. **The phrase-geometry transaction gate**: only a coherent row advances the traversal — a crisp, replicable unit of "what a committed row means to the decoder".
3. **Phase-separated routing** (earlier-query semantic compatibility vs row-query geometry routing) and the **one-way pre-x1 basin portability** — a concrete decomposition of where the next-instance decision lives, with unusually strong numerical-trust hygiene (float32 control) behind it.
4. **Coordinate transport without extent commitment** (x1→x2 coupling coexisting with part-extent basins) plus **traversal-inertia-not-coverage** cross-row state — jointly reframing "binding" into calibration of an existing state carrier.

For a mechanistic-interpretability venue, (1)+(2)+(4) with a prevalence tier and an ordering control could stand as a paper skeleton today. For the program's stated ambition (close the detector gap), the atlas is necessary-but-preparatory: no training lever has yet been demonstrated to move any of these mechanisms (the one attempt, the PVCI bridge, correctly failed its specificity controls). The stated candidate thesis is well-formed; its evidential gaps are exactly W1/W2 plus one successful mechanism-predicted training intervention.

Honest framing for the user: the current corpus proves the *diagnosis method*; the *therapy claim* remains fully open, and the week's most repeated silent assumption — that understanding this one geometry-sorted 2B DoRA checkpoint transfers to the goal — is untested along all three axes (ordering, adapter, scale).

---

## 7. Ranked Recommendations

**R1. Institute a prevalence tier (highest scientific leverage).** Before the next micro-discriminator beyond the already-approved 07-17 unit: one automated screen over 32–64 val images that, per image, harvests bagging-rescue events, replays each at its rescue-entry state ×16 samples, and reports the population frequency of (a) fixed-state multi-mode support, (b) rescue-entry vs terminal-state contrast, (c) coherent-row successor sensitivity. Existing runners cover ~90% of this. Every compass belief row gets a "prevalence: measured/unknown" column.

**R2. Run the two standing controls (cheapest falsification available).** Random-order-adapter and base-model replications of the P56 transition panel and the x1→x2 transport panel. Half a day each on existing scripts. Outcomes rewrite or dramatically harden five belief rows at once.

**R3. Cap the treadmill with a program-level rule.** Max K (suggest K=3) consecutive n≤4 units per mechanism thread before the thread must produce a prevalence estimate, consume a standing control, or emit a training-screen lever. Encode it in the compass update rule, since that is the document the loop actually obeys.

**R4. Execute the annotation-completeness experiment.** It is the compass's most-cited alternative explanation, the user has already built the labeling muscle, and it directly gates the training screen's Gate 3. Exhaustive-label vs original vs thinned on 8–16 dense images, measuring termination and coverage behavior of the *existing* checkpoint (no training needed for the first pass).

**R5. Build the thin probe harness.** One shared fixture→prompt→decode→receipt spine; experiment-specific intervention plugs. Target: next unit's new code ≤10KB, unit.md ≤300 lines with gates expressed as pytest checks rather than prose. This also shrinks audit lanes, which currently re-read prose contracts.

**R6. Fix the human-review instrument contract** (entity references, structured per-boundary codes, gate check before the user touches it), then re-run the genealogy waves on the existing 119-candidate cohort — the images and predictions are frozen; only the ledger is missing.

**R7. Right-size the meta-layer.** Main loop at `sol xhigh` except designated synthesis turns; audits only on conclusion-owning units (drop pre-launch audit for n≤4 probes that already carry no-op gates); one external review per milestone instead of ranked triplets; stop pasting >50KB conversations into context (attach by path).

**R8. Session architecture.** Fork a fresh session per research goal from compass + active unit; retire the mega-thread at the next natural boundary. The knowledge system already makes this safe; the 265MB thread is now the single largest audit and provenance liability.

**R9. Statistics floor.** Exact binomial CIs on all counts; per-unit declared comparison budget; batch-4-sampled vs batch-1-greedy evidence always labeled (the complete-box unit already models this — make it the template).

**R10. Connect the two strongest threads.** Representational-similarity screen: do naturally occurring rescue-entry states resemble hard-routed donor states in the pre-x1 residual subspace? A positive would give the first endogenous-synthesis target and a concrete training objective candidate; a negative would demote the portability thread before more is invested in it.

---

## 8. Closing Verdict

Judged as a research artifact, the session's output is trustworthy: I attempted to break its headline claims at the receipt level and failed. Judged as a research *strategy*, the session is over-invested in depth-at-a-point and under-invested in breadth, controls, and the data/training side of its own north star — and its self-correction machinery, which is excellent at claim-level honesty, has no mechanism that forces portfolio-level rebalancing. Judged as a development process, it has already fixed its worst 07-10-era habits and has a candid written retrospective, but the reformed steady state still spends roughly an order of magnitude more specification, audit, and orchestration than the evidence payload requires.

The single sentence I would put on the next session's wall:

> Before explaining another exact state, first establish how many states need the explanation — and check whether the base model and a random-order adapter need it too.

---

### Appendix A — Verification Log

| Check | Method | Result |
|---|---|---|
| Receipt digests (3 conclusion-owning) | `sha256sum` vs published | exact match |
| P56 fragmentation 11/32 vs 21/32 | recount from `call_bundles[].first_action` across 4 receipts | 21 cup + 11 pizza, disjoint clusters |
| Image-7818 IoU / release / owner switch | field extraction from receipt JSON | 0.8002594 / +1.0845668 / 664730→661523; layer-13 control present |
| x1 dominance (+4.335076; 99.9%) | per-coordinate deltas in results vs receipt-mean consistency | consistent (mean = +1.084567) |
| Human-review export | gate block in results.md vs frozen JSON path + digest | gate honestly failed; export present |
| Worktree state | `git status`, `git log` | clean except planned 07-17 unit; commits map to units |
| Session statistics | `jq` aggregation over 78,178 rollout records | figures as cited in §2 |
| Fleet cost | per-file final `total_token_usage` over 736 week rollouts; child attribution via parent-id in session meta | 346 children, 856,388,107 output tokens (93% of week's total on this machine) |
| Random-order / base-model control usage | `grep` across investigation + results | never consumed (unit.md defers explicitly) |

### Appendix B — Key Evidence Handles

- Transcript: `/data/CoordExp/.codex/sessions/2026/07/10/rollout-2026-07-10T03-37-15-019f4a19-d81c-75a2-84b0-2c20379e686e.jsonl`
- Hub: `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/{compass.md, overview.md, index.md, 2026-07-13-to-2026-07-16-weekly-research-report.md, experiments/}`
- PVCI phase: `/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/` (esp. `2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md`)
- Artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/` (21 roots, 13GB)
- Planned successor: `experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/unit.md` (untracked at review time)
