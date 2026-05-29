---
name: audit-review
description: "Use when producing a read-only CoordExp audit of code, configs, specs, artifacts, docs, progress notes, or OpenSpec changes for correctness, reproducibility, pipeline, and eval-validity risks."
---

# Audit Review

Produce read-only audits that help another implementer change CoordExp safely. Optimize for correctness, reproducibility, pipeline integrity, and eval validity over style commentary.

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

Use `references/report-template.md` when a skeleton is helpful.

## Read-Only Guardrails

- Do not modify production code, configs, docs, specs, or artifacts during an audit.
- Do not invent results. Label unverified ideas as hypotheses and keep them out of severity-ranked findings.
- Do not treat benchmark scopes as interchangeable. Always label `tiny`, `val200`, `limit=200`, first-200, full-val, proxy view, raw-text, coord-token, bbox format, checkpoint id, and launch shape when relevant.
- Do not use `progress/` as current behavior when `docs/` or stable specs cover the contract.
- Use Git inspection only when the audit scope depends on dirty state, a PR/change diff, or the user asks for it; otherwise do not run Git by reflex.
- For Python code exploration, narrow first with `rg` or `rtk grep`, then use Serena symbol tools.
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
- Stage-2: `stage2_ab.pipeline` vs `rollout_matching.pipeline`, rollout runtime, teacher-forcing modules, duplicate-control losses, metric keys.
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
