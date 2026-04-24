---
name: audit-review
description: Use when producing a read-only CoordExp audit of code, configs, specs, artifacts, docs, progress notes, or OpenSpec changes for correctness, reproducibility, pipeline, and eval-validity risks.
---

# Audit Review

## Overview

Produce an audit report that helps an implementer safely change/refine CoordExp without guessing.
Optimize for correctness, reproducibility, and contract/pipeline integrity rather than style refactors.

Use the repo's source-of-truth order: `openspec/specs/` -> `docs/` -> `openspec/changes/<active-change>/` -> `progress/`.
Use `progress/` for evidence, diagnostics, benchmark scope, and history; do not answer current-behavior questions from it when docs/specs cover the contract.

## Output Contract (What You Deliver)

- Severity-ranked findings (`P0`/`P1`/`P2`) with concrete evidence handles (`path:line`, config keys, exact commands, or tool output).
- “Confirmed OK / ruled out” notes to prevent backtracking.
- Verification steps: exact commands/tests to reproduce or validate each claim.
- Open questions: smallest set of clarifications required to remove ambiguity.
- Suggested next actions for an implementer (do not implement changes yourself).

Use `references/report-template.md` if you want a ready-made skeleton.

## Guardrails (Read-Only Audit)

- Do not modify production code/configs/specs. No `apply_patch` against `src/`, `configs/`, `openspec/`, etc.
- Prefer read-only exploration: `rg`, `find`, `git diff`, `sed`, `python -m pytest`, `python -m py_compile`.
- If you must write a temporary test or probe:
- Default: write under `/tmp/` so the repo stays clean.
- If you need it under `temp/` for sharing, ask the user first and keep artifacts minimal.
- Always check and report `git status --porcelain` at the start; if the worktree is dirty and it matters, ask before proceeding.
- For Python code exploration: Serena MCP is mandatory (symbol-aware navigation; provide `relative_path` constraints).
- Never invent results. If a claim cannot be verified, label it as a hypothesis and keep it out of severity-ranked findings.
- Do not conflate benchmark scopes. Report `val200`, `limit=200`, first-200, proxy view, full-val, raw-text vs coord-token, checkpoint ids, and launch shape when they affect the claim.

## Workflow (Breadth Pass -> Depth Pass -> Report)

### Step 0: Clarify The Ask (Smallest Unblocking Questions)

- If scope is ambiguous, ask 1–3 questions max:
- What exact artifact(s) are we auditing: path(s) or concept?
- Is the goal: “spec/design review only” or “implementation vs spec audit”?
- Any constraints: time budget, no-network, specific configs/datasets, must-pass tests?

Assume the deliverable is a report for a separate implementer unless the user explicitly asks you to change code.

### Step 1: Snapshot + Map The Surface Area (Breadth Pass)

- Safety snapshot:
- Run `git status --porcelain` and note any dirty files.
- If auditing a change/PR, capture `git diff --name-only` (or change directory file listing) to bound the search.
- Identify entrypoints and contracts:
  - Docs/specs: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, relevant `openspec/specs/`, relevant domain docs.
  - Progress: `progress/index.yaml`, `progress/README.md`, and the matching category router when you need empirical evidence.
  - Code: likely entrypoints (`src/bootstrap/`, `src/config/loader.py`, `src/datasets/geometry.py`, `src/trainers/`, `src/infer/`, `src/eval/`, `public_data/`).
  - Tests: locate tests adjacent to the target area and any policy scans.
- Grep for relevant context (fast, wide net):
- Use `rg` to find: config keys, CLI flags, spec terms, artifact filenames, error messages.
- Use `references/grep-seeds.md` when you need good starting patterns.
- Build a short “context index”:
- Key files with 1-line reason each.
- Key symbols to inspect (class/function names) with file paths.

### Step 2: Inspect The Highest-Risk Flows (Depth Pass)

Pick 3–5 top risk areas based on impact and likelihood, then deep-dive with evidence:

- Pipeline and process flow:
  - Trace data flow: input -> transforms -> packing -> training/infer/eval -> artifacts.
  - Verify invariant-sensitive steps (geometry, ordering, normalization).
  - Route geometry checks through `src/datasets/geometry.py`, not ad hoc bbox math.
- Configuration and contracts:
  - Check strict parsing / unknown-key behavior (fail-fast vs silently ignored).
  - Check backward-compat surfaces (stable CLI contracts, deprecated keys policy).
  - Check that stable workflows stay YAML-first instead of adding CLI flags.
- Artifacts and eval validity:
  - Verify training manifests: `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`.
  - Verify infer/eval artifacts: `summary.json`, `resolved_config.json`, `resolved_config.path`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`, and guarded companions when enabled.
  - For current infer behavior inspect `src/infer/pipeline.py::run_pipeline`; for current eval behavior inspect `src/eval/detection.py::evaluate_and_save`.
- Determinism and reproducibility:
  - Look for ordering-dependent behavior, random seeds, multiprocess I/O, filesystem-dependent nondeterminism.
- Silent failure policy:
  - Ensure unexpected exceptions are not swallowed in core paths; best-effort behavior should be narrow and justified.

### Step 3: Validate Or Falsify With Targeted Tests (Optional, But High Value)

- Prefer running existing targeted tests first.
- If a hypothesis needs a minimal repro, write a temporary test:
- Put it in `/tmp/` and run it with `PYTHONPATH=.` so the repo stays unchanged.
- Keep it tiny and single-purpose; delete it afterwards (or ask before deleting if the user wants to keep it).
- When tests are too expensive to run, provide a verification plan with expected artifacts and failure signals.

### Step 4: Write The Audit Report

- Lead with findings (ranked). Each finding must include:
- Evidence handle (`path:line`, config key, or command output summary).
- Why it matters (correctness/repro/eval validity/maintainability).
- Suggested fix direction (for implementer) and how to verify.
- Add “confirmed OK / ruled out” checks that reduce backtracking.
- End with open questions (only what’s truly needed).

## Resources (optional)

Open these only when helpful (progressive disclosure):

- `references/report-template.md`: audit report skeleton (P0/P1/P2 + evidence + verification).
- `references/grep-seeds.md`: high-signal `rg` starting points for broad context discovery.
- `references/pipeline-checklist.md`: checklist for pipeline/process correctness and reproducibility risks.
