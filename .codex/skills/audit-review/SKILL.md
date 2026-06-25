---
name: audit-review
description: "Use when producing a read-only CoordExp audit of code, configs, specs, artifacts, docs, progress notes, or OpenSpec changes for correctness, reproducibility, governance, pipeline, and eval-validity risks."
---

# Audit Review

Produce read-only audits for another implementer. Prioritize correctness, reproducibility, governance, pipeline integrity, and eval validity over style commentary.

## Role Boundary

Use this for severity-ranked audits, not implementation. If the user gives a concrete model symptom such as a metric drop, invalid spike, duplication burst, or length collapse, start with `model-diagnosis`; return here only for independent correctness or claim-validity review. If the user asks whether a new mechanism is contract-safe before launch, use `model-innovation-risk-audit`.

## Audit Mode Selector

Name the mode before searching broadly:

- `approval audit`: pre-implementation, pre-merge, pre-launch, or final approval review; return `approve`, `hold`, `reject`, or `needs user decision`.
- `change/spec audit`: compare implementation, docs, stable specs, and active OpenSpec deltas.
- `artifact/run audit`: start from the exact artifact root; label scope and do not generalize beyond it.
- `launch gate`: decide `promote`, `hold`, `rerun gate`, or `needs user decision`.
- `claim validity audit`: identify the claim, scope, baseline/ablation, metrics, counterevidence, and falsification gap.
- `implementation-vs-contract audit`: verify code/config/runtime/artifacts implement the documented contract.

If the user requests blocker-only review, stop after blocking findings, confirmed OK checks, and residual risks.

## Authority Model

Use current repo truth in this order:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when explicitly in scope
7. `progress/` only for history, diagnostics, benchmark evidence, or empirical failures

Use `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for routing. Treat `progress/audits/` and other temporary notes as removable evidence, not durable codebase references.

## Governance Checks

For stable contracts or OpenSpec work:

- Check active-change state when it matters.
- Validate specs strictly when specs changed.
- Verify code/docs/spec sync when schema, artifact names, metrics, loss semantics, entrypoints, or recommended workflows move.
- Treat incomplete proposals as active or explicitly deprecated; do not let stale changes masquerade as current contract.
- Reviewer timeout or disconnection is unresolved, not approval.

## Output Contract

Lead with findings, ordered by severity:

- `P0`: likely invalidates correctness, reproducibility, or evaluation claims.
- `P1`: substantial risk to supported workflows, artifacts, metrics, or config contracts.
- `P2`: maintainability, clarity, or missing-coverage risks that could become failures.

Each finding needs:

- evidence handle: file path, symbol, config key, artifact path, command output summary, or doc/spec line reference
- impact: why it matters for correctness, reproducibility, eval validity, or maintainability
- fix direction: what an implementer should change
- verification: the smallest realistic command, artifact check, or targeted test that would prove the fix

Also include:

- confirmed OK / ruled out checks that prevent backtracking
- open questions only when they block a reliable conclusion
- suggested next actions for an implementer

For approval audits, also include the verdict, validation run, skipped checks, and residual risks. If the user asks for a report artifact, write a standalone Markdown report and verify section structure, placeholder markers, and whitespace.

Use `references/report-template.md` when a skeleton is helpful.

## Read-Only Guardrails

- Do not modify production code, configs, docs, specs, or artifacts during an audit.
- Do not invent results. Label unverified ideas as hypotheses and keep them out of severity-ranked findings.
- Do not treat benchmark scopes as interchangeable. Always label `tiny`, `val200`, `limit=200`, first-200, full-val, proxy view, raw-text, coord-token, bbox format, checkpoint id, and launch shape when relevant.
- Do not use `progress/` as current behavior when `docs/` or stable specs cover the contract.
- Use Git inspection only when the audit scope depends on dirty state, a PR/change diff, or the user asks for it; otherwise do not run Git by reflex.
- For Python code exploration, route docs/configs first, then use a correct local CodeGraph index only for broad "where should I look?" maps. Once files or symbols are known, switch to Serena for exact references, bodies, declarations/implementations, diagnostics, and edit-risk checks. In linked worktrees, do not trust CodeGraph results from another checkout.
- For broad approval audits, use `contract_auditor` as the default custom-agent role. Add `coordexp_mapper` for unknown surfaces and `upstream_relation_tracer` for cross-root dependencies; reconcile every lane into one verdict. A timed-out or vague lane is unresolved, not approval.
- If a temporary probe is unavoidable, prefer `/tmp/`. Ask before writing under repo `temp/`.

## Audit Workflow

### 1. Bound The Surface

Identify the smallest relevant set of:

- docs: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, domain docs
- stable specs: only exact `openspec/specs/` contracts needed by the question
- progress: only matching benchmark, diagnostic, exploration, direction, pretrain, or audit evidence
- code: likely `src/config/`, `src/datasets/`, `src/detection/`, `src/trainers/`, `src/infer/`, `src/eval/`, `src/bootstrap/`, `src/common/`
- configs: the concrete YAML profiles under review
- tests/artifacts: targeted surfaces from `docs/IMPLEMENTATION_MAP.md` or artifact manifests

### 2. Trace High-Risk Flows

Prioritize 3-5 flows with the highest impact:

- Data contract and geometry: JSONL schema, `bbox_2d` xor `poly`, ordering, pixel/norm1000/token transitions, image-root resolution.
- Training: config schema, `src/sft.py`, `src/training_runtime/plan.py`, trainer variant, collator family, packing owner, cache eligibility, manifests.
- Stage-1 compact detection: `LatestDetectionTrainingConfig`, `DetectionTrainingDataset`, recursive detection objective, sidecar/packing policy.
- Stage-2 rollout correction: `stage2_rollout_correction`, rollout runtime, residual-set planning, duplicate filtering before assignment, teacher-forcing modules, diagnostic events, and metric keys.
- Infer/eval: `src/infer/pipeline.py::run_pipeline`, `resolved_config.json`, `resolved_config.path`, confidence post-op compatibility, `src/eval/detection.py::evaluate_and_save`, guarded metrics.
- Artifacts/provenance: `summary.json`, `metrics.json`, `run_metadata.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `effective_runtime.json`, durable copied summaries.

### 3. Look For Failure Classes

Check for:

- silent fallback where fail-fast is expected
- geometry drop/reorder/renormalization without contract support
- stale config keys accepted as no-ops
- artifact names or metric keys that drift from docs/specs
- benchmark claims missing scope labels
- confidence/eval paths applied to incompatible bbox formats
- temporary audit/progress notes being used as durable docs
- broad runtime fusion assumptions on paths that should use offline-prepared JSONL

### 4. Validate Only When In Scope

When validation is allowed or requested:

- prefer existing targeted tests from `docs/IMPLEMENTATION_MAP.md`
- Codex shells initialize the `ms` conda environment by default; use `rtk pytest ...` for noisy test output in this repo
- prefer artifact and manifest checks over broad reruns
- for long or sharded runs, check merged summaries/manifests rather than log lines

If validation is not allowed or too expensive, provide exact verification steps and expected failure signals.

## Resources

Open only when helpful:

- `references/report-template.md`: audit report skeleton
- `references/grep-seeds.md`: high-signal `rg` starting points
- `references/pipeline-checklist.md`: end-to-end correctness and reproducibility checklist
- `references/governance-claim-checks.md`: OpenSpec governance, claim-validity, and review-closure checklist
