---
name: audit-review
description: Use for read-only CoordExp audits, fixed-point diff/code reviews, approval or launch gates, and explicitly requested bounded review-revise loops, with engineering and intent/contract judgments kept separate.
---

# Audit Review

Audit correctness, reproducibility, governance, pipeline integrity, and evaluation validity. Default to a concise, blocker-first result. Expand only when requested or needed to justify a decision.

## Route Before Reviewing

- Use `model-diagnosis` first for metric drops, malformed outputs, duplication, length/stop changes, train/eval divergence, or other model-behavior symptoms.
- Use `model-innovation-risk-audit` for pre-launch trust gates on a new mechanism, objective, loss, or eval path.
- Use `debug-feedback-loop` for a reproducible code, config, CLI, runtime, integration, flake, or performance failure.
- Use this skill for independent correctness, contract, claim-validity, diff, approval, or launch judgment.

One-shot audits are read-only. Enter convergence mode only when the user explicitly asks for review, revision, and repeated checking; honor the mutation scope they authorized.

## Bound The Decision

Before broad reading, name:

- mode: `approval`, `change/spec`, `artifact/run`, `launch`, `claim validity`, `implementation-vs-contract`, or `diff/code review`
- decision at stake and evidence scope
- fixed point or artifact version
- stop condition
- prior finding ledger, if any

Compress broad requests into a falsifiable question. For blocker-only review, stop after blocking findings, confirmed-OK checks, verdict, and residual risk.

## Authority And Routing

Start with `docs/AGENT_INDEX.md` and `docs/catalog.yaml`, then follow the relevant canonical docs and `docs/IMPLEMENTATION_MAP.md`. Use stable specs for compatibility-sensitive contracts and active OpenSpec artifacts only when the user puts that change in scope. Treat `research/`, old worktrees, memories, and `progress/` as historical evidence, not current behavior; do not create new `progress/` records.

Current route examples:

- training: `src/train.py` -> `src/training/pipeline.py` -> `src/training/supervised_trainer.py`
- inference: `src/infer.py` -> `src/inference/pipeline.py` -> `src/inference/runtime.py` / `src/inference/backend.py`
- detection evaluation: `scripts/evaluate_detection.py` -> `src/eval/detection_consumer.py`

Verify these routes against the live docs before relying on them; paths preserved only in history are not current architecture.

## Fixed-Point Diff Review

When reviewing a branch, commit range, staged/unstaged work, or PR-equivalent snapshot:

1. Use the user-supplied base. If none is supplied, use an unambiguous canonical merge base and state it; ask only when the choice materially changes scope.
2. Capture the base, merge base, commit list, changed paths, diff stat, and worktree status once.
3. Separate committed, staged, unstaged, and untracked changes. Stop on an invalid ref or empty in-scope diff.
4. Resolve intent from the user brief, explicitly scoped active change, stable contracts, canonical docs/configs/tests, then commit text as weak evidence.
5. Read [code-review-baseline.md](references/code-review-baseline.md) when smell and standards guidance is useful.

Useful handles:

```bash
git rev-parse --verify <base>
git merge-base <base> HEAD
git log <base>..HEAD --oneline
git diff --stat <base>...HEAD
git diff --name-only <base>...HEAD
git status --short
```

## Two Independent Judgment Axes

**Engineering Standards** covers ownership, readability, depth, coupling, proportionality, testability, and repository conventions.

**Intent And Contract** covers requested behavior, algorithm and forward semantics, data/geometry/order, targets and loss normalization, optimization, statistical assumptions, metric comparability, and artifact/provenance contracts.

Do not let a pass on one axis cancel a failure on the other. The agent owns code-level evidence, implementation risk, and verification design. The user owns changes to research meaning, forward semantics, data/loss meaning, statistical assumptions, metric interpretation, expensive runs, and compatibility or publication trade-offs.

## Findings And Decisions

Severity:

- `P0`: likely invalidates correctness, reproducibility, evaluation claims, or launch safety.
- `P1`: substantial risk to supported workflows, contracts, metrics, artifacts, or maintainability.
- `P2`: clarity, coverage, or future-maintenance risk.

Every material finding needs an evidence handle, impact, decision implication, fix/probe direction, and smallest realistic verification. Classify each P0/P1 before proposing changes:

- `fix`: the direction remains valid; make a bounded correction.
- `narrow`: reduce the claim, workflow, support matrix, or launch scope.
- `drop`: stop the mechanism or path as framed.
- `probe`: obtain the cheapest discriminating runtime, artifact, baseline, ablation, or upstream receipt.
- `needs user decision`: the fork changes research meaning, compatibility, cost, destructive behavior, or publication/launch risk.

Prefer `probe` to speculative fixes when executed semantics or matched evidence is missing. Reviewer timeout, disconnection, vague output, or a skipped hardware check is unresolved, not approval.

## Evidence Routine

Prioritize the highest-risk touched flows rather than auditing everything:

- data schema, image/geometry/order, and coordinate transitions
- config strictness and removed-key rejection
- training assembly, packing/cache ownership, loss normalization, and manifests
- inference runtime, scoring evidence, merge parity, and resolved configuration
- evaluator input contract, metric scope, and artifact provenance

For run or claim audits, identify code/config identity, checkpoint, data/sample scope, training budget, decode/runtime settings, relevant metric artifacts, and whether the comparator is matched or confounded.

Route runtime-dependent P0/P1 confirmation through `probe_runner` when available. Treat source reading, mocks, or plans as insufficient when upstream/vendor/runtime behavior owns the result. Prefer targeted tests from `docs/IMPLEMENTATION_MAP.md`, artifact/manifest checks, and narrow receipts over broad reruns.

Open references only when useful:

- [pipeline-checklist.md](references/pipeline-checklist.md) for end-to-end semantics and reproducibility
- [governance-claim-checks.md](references/governance-claim-checks.md) for OpenSpec, claims, and closure
- [grep-seeds.md](references/grep-seeds.md) for scoped discovery
- [report-template.md](references/report-template.md) for a requested standalone report

## Explicit Convergence Mode

Use this only when the user asks to iterate work through review and revision.

1. State objective, mode, allowed mutation level, artifact/version, decision, evidence that could change it, approval gates, and stop condition.
2. Produce or inspect the first bounded artifact.
3. If subagents are allowed or requested, use two independent lanes by default. Assign exact scope, read/write boundary, evidence output, and stop condition. Suitable roles include `contract_auditor`, `repo_scout`, `upstream_relation_tracer`, `model_diagnostician`, and `probe_runner`; use `implementation_worker` only for an explicitly owned patch.
4. Triage results as P0/P1/P2, non-blocking, wrong, or duplicate. Reject findings only with technical evidence.
5. Classify every accepted P0/P1 as `fix`, `narrow`, `drop`, `probe`, or `needs user decision` before revision.
6. Revise only authorized surfaces. Do not patch through research-meaning, compatibility, cost, destructive, or publication decisions.
7. Re-review changed evidence. Default to at most two review/revision rounds. Continue only when a round closes a material finding; do not run another clean wave without changed evidence or an unresolved high-stakes decision.

Convergence requires all P0/P1 findings to be resolved, narrowed, dropped, probed, evidence-rejected, or assigned to the user; required reviewers and verification must be complete; and claims must not exceed evidence.

End in exactly one state:

- `approve` (one-shot gate)
- `approved to implement`
- `ready for user approval`
- `implemented and verified`
- `hold`
- `needs user decision`
- `probe required`
- `narrowed/dropped`

## Output Contract

Lead with severity-ranked blockers and concrete handles. For diff/code review, use:

1. snapshot and fixed point
2. Engineering Standards findings
3. Intent And Contract findings
4. confirmed OK / ruled-out checks
5. verdict and residual risk

If no findings remain, say so and name skipped checks or residual risk. Keep open questions only when they block a reliable conclusion. For convergence mode, also state artifact/version, lanes used, accepted/rejected findings, decisions and revisions, verification, remaining gate, and exact stop state.

## Guardrails

- Do not modify production code, configs, docs, specs, or artifacts during a one-shot audit.
- Do not invent results or promote hypotheses into findings.
- Label benchmark and artifact scope precisely; do not compare incompatible dataset slices, bbox/coordinate surfaces, checkpoints, parsers, or launch shapes.
- Use Git only when the audit scope depends on it.
- Read an in-scope finding ledger first and report only new findings, status changes, rejected findings, or closure evidence.
- Ask before writing temporary probes inside the repository; prefer `/tmp/`.
