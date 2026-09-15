## ADDED Requirements

### Requirement: Permanent owner-bridge algorithm accuracy battery

Before production launch, the implementation SHALL pass deterministic reference checks for global injective matching, normalized uncovered-set routing, four-branch causal owner use including gradient reach into selected values and the shared atomizer, RMS clipping, zero-bridge parity, block-20-to-final causal visibility, activation-checkpoint on/off forward-and-gradient parity with single-count diagnostics, packed-segment isolation, teacher-forced versus forced-token incremental execution, serialization reload, and per-sequence HF lifecycle isolation. Reference checks MUST use independently computed expected results or paired execution paths and MUST fail on semantic disagreement rather than merely checking finite values.

#### Scenario: Common activation pulse is added to causal-swap branches
- **WHEN** identical logit energy is added to every correct and swapped branch
- **THEN** the reference and implementation `L_use` margins MUST remain unchanged within tolerance

#### Scenario: Same-seam router is substituted
- **WHEN** a test reads the boundary at block 20 instead of after upper-layer owner-conditioned KV formation
- **THEN** the causal-visibility check MUST distinguish it from the approved final-normalized route

#### Scenario: Checkpoint recomputation loses bridge context
- **WHEN** activation checkpointing recomputes any ordinary or owner-use branch with missing or different typed bridge context
- **THEN** checkpoint-on/off parity MUST fail before a production smoke can pass

### Requirement: Distributed collective choreography acceptance

Before the eight-rank smoke is treated as current evidence, readiness SHALL include a distributed choreography acceptance battery. A failure of this battery is runtime evidence only and MUST NOT be recorded as a model-quality, objective, or algorithm-accuracy result.

The battery MUST include an actual two-rank data-parallel test whose ranks are given deliberately uneven eligible owner-use branch counts and a mixed real/shadow slot assignment. It MUST assert exactly one wrapped anchor forward per physical accumulation slot per rank, exactly one combined backward per slot, exactly one optimizer update for the window, and DoRA/selected-embedding/bridge gradient parity against a single-process reference within declared tolerance. Mocked or single-process-simulated process groups do not satisfy this requirement.

The battery MUST then compare an exact one-step W8 choreography digest — the per-rank ordered collective kinds and counts for one planned step — and require it to be identical across all eight ranks before the full bounded W8 smoke is rerun to its step budget.

The battery MUST also produce a packed-versus-isolated owner-use receipt covering branch logits, `L_use`, gradients, and the realized FA2 `cu_seq_lens` boundaries.

#### Scenario: Ranks enqueue different collective counts
- **WHEN** the one-step choreography digest differs on any rank
- **THEN** readiness MUST fail before the full W8 smoke is rerun
- **AND** the failure MUST be reported as runtime choreography evidence rather than as a training-quality outcome

#### Scenario: Standalone control timeout is raised instead of repairing choreography
- **WHEN** a distributed hang is addressed by raising the static rank-report control timeout constant beyond the effective NCCL watchdog
- **THEN** that change MUST NOT be accepted as the repair
- **AND** the bounded constant MUST be restored and the choreography battery MUST still pass

#### Scenario: Uneven branch counts change gradients
- **WHEN** two ranks with different eligible branch counts produce gradients that disagree with the single-process reference
- **THEN** the two-rank acceptance MUST fail
- **AND** global denominator and world-size scaling MUST be re-verified as unchanged from the pre-repair definition

### Requirement: Permanent owner-bridge smoke runs

Readiness SHALL include at least one exact-module single-rank smoke and one bounded eight-rank Accelerate smoke derived from the production configuration. Together they MUST exercise the existing COCO/no-resize-1024/global-12000 data path or a provenance-bound fixture with identical semantics, all four loss terms, safe optimizer stepping, distributed reduction, bridge checkpoint publication, payload reload, and batched greedy HF lifecycle. The eight-rank smoke MUST use the production trainable surface and global EBS derivation while limiting steps and data for mechanical validation.

Smoke acceptance MUST require finite losses/gradients, complete owner assignment, exact artifact identity, no distributed hang, no lifecycle leakage, and successful reload. It MUST NOT require improved recall, duplication, atom AP, or stopping from a smoke-trained checkpoint.

An eight-rank smoke that terminates on divergent collectives, a watchdog timeout, or a control-gather failure does not satisfy this requirement and MUST be rerun on the repaired tree after the choreography acceptance battery passes.

#### Scenario: Eight-rank smoke terminates on a collective divergence
- **WHEN** an eight-rank smoke fails with unequal per-rank enqueued collective counts and zero completed steps
- **THEN** that run MUST NOT count as the required eight-rank smoke
- **AND** its evidence MUST be scoped to runtime choreography rather than to model behavior

#### Scenario: Eight-rank smoke completes with poor recall
- **WHEN** every mechanical and algorithm-identity check passes but the tiny smoke checkpoint has poor natural recall
- **THEN** the smoke remains mechanically accepted
- **AND** poor recall MUST be recorded without blocking production launch

#### Scenario: Smoke cannot assign a labeled owner
- **WHEN** any smoke example reaches loss computation without one injectively matched atom per labeled owner
- **THEN** smoke acceptance MUST fail as an algorithm-contract error

### Requirement: Conditional eight-GPU production launch

The production run MAY launch immediately after strict OpenSpec validation, the algorithm accuracy battery, the distributed collective choreography acceptance battery, the single-rank smoke, the eight-rank smoke, and fixed-tree engineering/intent audits pass with no unresolved P0/P1. Every one of these results MUST be produced by the frozen tree that will be launched; an earlier passing result from a tree that predates a runtime repair does not satisfy this requirement. Launch MUST use the exact Stage 1 production config, eight Accelerate ranks, a unique collision-safe artifact root, live GPU ownership/headroom checks, and an at-most-once command guard. It MUST preserve unrelated GPU processes and MUST record the resolved config, code/tree identity, command, parent checkpoint, process ids, visible devices, and run root as soon as startup succeeds.

No tiny-smoke model-quality threshold SHALL be part of readiness. A launch is confirmed only after all eight ranks initialize, the run identity is durable, and at least one safe optimizer step plus its finite/gradient/artifact heartbeat is observed.

If the singleton production claim was consumed by an attested pre-run infrastructure failure, it MUST remain immutable and MUST NOT be deleted, renamed, overwritten, reused, or treated as automatically renewable. Recovery MAY occur only after fresh explicit user authorization and SHALL use one fixed canonical append-only parent-linked successor. That successor MUST declare `attempt_ordinal=1` and `max_recovery_attempts=1`; use a distinct recovery policy/schema while leaving the original v1 policy and validator unchanged; bind the original claim, intent, activation, preflight, all eight worker admissions, failed run-binding receipt, stdout, and stderr by fixed exact path and SHA-256; bind the parent intent's exact run artifact root/output directory/run name and the current frozen repository/resolved-production-config identities; and use a fresh nonce plus recovery-specific intent identity. Original and successor launch evidence MUST remain independently readable.

The fixed successor path MUST be reserved using no-replace creation and parent-directory durability before deriving its nonce or payload. The reservation MUST NOT be removed on any failure path. An empty, partial, or malformed reserved successor is terminally consumed and MUST make later recovery requests fail before parsing or process creation. Recovery MUST also reject independently on recovery-claim residue, any ordinal or maximum other than `1`, or an intent identity that is not bound to the parent claim and attempt ordinal.

Recovery eligibility MUST fail before successor reservation or process creation if the exact terminal failed-binding evidence with the accepted process-exited error and nonzero return code is absent, the exact parent launcher PID/start-time/boot identity remains live, the original attempt has a successful run binding, a parent-nonce `run.json` under the parent-bound artifact root, model/optimizer heartbeat, mutated parent evidence, a reused nonce, an existing or malformed successor/residue, a recovery-preflight run-root mismatch, or any ordinal other than the single authorized attempt. A regression MUST prove that `run.json` publication precedes the sole production model/optimizer construction path before parent run absence is used as no-model/no-optimizer evidence. The successor's intent, activation, worker admissions, pre-model quorum, run binding, and failures MUST be attempt-scoped. Every production worker MUST select exactly one claim from the closed original/successor set by nonce and MUST record the selected path and SHA-256 before distributed initialization; unreadable, zero-match, or multiple-match candidates MUST fail closed without fallback. If reservation, completion, activation, binding, or the first finite/applied heartbeat fails or is uncertain, all evidence MUST be retained and no further activation MAY be issued.

#### Scenario: Readiness passes and GPUs are free
- **WHEN** all declared readiness evidence is current for the frozen implementation and eight GPUs have safe headroom
- **THEN** the Stage 1 production training MAY be launched without another approval turn
- **AND** its durable launch identity MUST be reported

#### Scenario: Another job owns required GPU memory
- **WHEN** live inspection shows that an eight-rank launch would evict or materially interfere with unrelated work
- **THEN** launch MUST remain pending without killing or reconfiguring that work
- **AND** the blocker and unchanged readiness evidence MUST be reported

#### Scenario: Consumed claim failed before a run existed
- **WHEN** the immutable original claim is bound to a failed activation with no run binding, run root, model construction, optimizer mutation, or finite/applied heartbeat
- **AND** the user explicitly authorizes one recovery after the repaired frozen tree passes its gates
- **THEN** the guard MAY publish exactly one canonical parent-linked successor with a fresh nonce and attempt ordinal `1` of `1`
- **AND** it MUST preserve and hash-bind all original claim and failure evidence

#### Scenario: Successor reservation fails before payload completion
- **WHEN** the fixed successor path has been reserved but nonce or payload derivation fails, or the reserved bytes are empty, partial, or malformed
- **THEN** the reservation MUST remain in place as terminal evidence
- **AND** every later recovery invocation MUST reject before parsing it or creating a process

#### Scenario: Recovery would become a third activation
- **WHEN** a recovery successor already exists or the single successor activation is failed, uncertain, or consumed
- **THEN** the guard MUST reject another process creation before changing any launch evidence
- **AND** it MUST NOT delete, rotate, or repurpose either the original claim or its successor

#### Scenario: Parent failure evidence drifts
- **WHEN** any anchored original claim, intent, activation, preflight, admission, failed run-binding, stdout, or stderr file no longer matches the fixed exact path and SHA-256 authorized for recovery
- **THEN** successor validation MUST fail before nonce publication or process creation

#### Scenario: Closed claim selection fails
- **WHEN** zero or multiple fixed claim candidates match the worker nonce, or either candidate is unreadable
- **THEN** production startup MUST fail before distributed initialization without falling back to the other claim

#### Scenario: Parent run identity is not conclusively absent
- **WHEN** the terminal failed-binding receipt is absent or non-terminal, a successful parent binding exists, the recovery preflight changes the parent-bound run root, or a parent-nonce `run.json` exists under that root
- **THEN** recovery MUST fail before successor reservation
