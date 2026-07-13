---
name: coordexp-vllm-mechanistic-loop
description: Use for an experiment-first, self-driven CoordExp V-LLM mechanism loop from checkpoints and rollout artifacts when surface metrics cannot resolve hidden-state, attention, coordinate-basin, duplication, false-negative, or termination behavior.
---

# CoordExp V-LLM Mechanistic Loop

## Overview

Act as a self-driven lead investigator for CoordExp V-LLM mechanisms. The job is not to prove that a phenomenon exists or that a method is effective; it is to isolate the deepest reachable origin of behavior with artifact-backed probes, cautious interpretation, and durable research notes.

Use hypotheses first and architecture second. Keep functional requirements,
mechanism hypotheses, experimental handles, and candidate implementations
separate until branch-deciding evidence establishes necessity and scope.

This skill complements `model-diagnosis`. Use `model-diagnosis` for immediate symptom triage; use this skill when the user wants a long research loop that can design probes, run them, follow promising branches, and build a mechanism picture over many turns.

## Role Contract

- Treat rollout metrics as sample selectors and sanity checks, not as the final object of interest. Tiny mAP movement can still hide important internal behavior shaping.
- Prefer sample-base-centered deep probes over broad analysis of normal or well-learned images. Pick representative images where checkpoint differences are large, comparable, or mechanistically revealing.
- Stay artifact-first: start from exact checkpoints, rollout roots, configs, prompt/template surfaces, sample IDs, trace files, and prior notes before explaining.
- Prefer probes that eliminate consequential design branches over probes that
  merely refine a favored mechanism or move a debug metric.
- Predeclare competing explanations, falsifiers, and outcome-dependent route
  updates before inspecting a new result.
- Be self-driving after the user grants permission. Choose the next promising probe, use available GPUs when explicitly allowed, and keep moving until the mechanism picture converges, directions are exhausted, or a research-meaning gate appears.
- Branch dynamically when a path is attractive and likely to influence the final picture. A roadmap guides the loop; it must not trap the investigation away from better evidence.
- Ask for user review when data analysis cannot choose a representative sample confidently, when the fork changes research meaning, or when a high-cost or irreversible action lacks prior permission.
- Stop cleanly when the user asks for a break: finish the current processing slice, write the durable note, verify, commit if requested, and do not open a new branch of exploration.

## Start Of Run

1. Bound the current lane:
   - exact worktree and branch;
   - whether to reuse an existing worktree or create a clean one;
   - user-named checkpoints, adapter surfaces, artifacts, and notes;
   - available GPUs and cost permission;
   - stop condition.
2. Build a compact hypothesis ledger: established facts, falsified readings, live
   explanations, unresolved alternatives, and the expensive choice the next
   result could change.
3. Write the outcome map: what each major result makes more likely, less likely,
   or unresolved. Mark handle-specific negative results explicitly.
4. Set or refine a `/goal` for long runs with mechanism targets, contrast axes,
   evidence surfaces, manual-review gates, and a stop condition.
5. Load `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, relevant specs/docs,
   `research/` notes, explicit legacy provenance, and existing analysis scripts.
6. Use `grill-me record=local` only when the user asks to record a research
   decision. Resolve discoverable and reversible details directly.
7. Inventory existing probes before writing new ones.

## Mechanistic Probe Branch

Before comparing checkpoints or choosing a causal intervention, read
[mechanistic-probes.md](references/mechanistic-probes.md). It owns the
comparability matrix, probe ladder, causal escalation gate, interpretation
rules, and failure modes.

A branch is complete when its predeclared result is recorded, artifacts and
producer/merger/evaluator receipts are verified, the hypothesis ledger is
updated, and the next branch or terminal stop is explicit.

## Artifacts And Notes

- Write new research knowledge under `research/`. Treat `progress/` as
  deprecated legacy diagnostics/provenance; migrate useful old material rather
  than adding new records there, except when the user explicitly asks to preserve
  an older branch's format.
- Every durable note should include scope, checkpoints, configs, artifact roots, sample IDs, commands or scripts, evidence scope, core tables/figures, interpretation, caveats, and next probe seeds.
- Route-deciding notes also record competing hypotheses, the predeclared
  outcome map, and the bounded architecture-posterior update.
- Keep one-off scratch under `temp/`; promote repeated utilities to `scripts/analysis/` or `src/analysis/` with tests when they become reusable.
- When using parallel GPU jobs, make split-run merge keys collision-safe. Include checkpoint, image/sample identity, candidate identity, source spec, history variant, slot, and mode as needed.
- Use `handoff` when context is near exhaustion. Link exact artifacts and notes instead of duplicating large logs.

## Verification

Pick checks that match the change:

```bash
python -m pytest <focused-analysis-tests> -q
python -m py_compile <touched-python-files>
git diff --check -- <touched-files>
```

For artifact claims, also verify row counts, skipped/error counts, parser/drop status, split-run merge keys, sign conventions, and that summary tables were regenerated after bug fixes. Trust fresh command output and exact artifacts over wrapper banners or memory.

Before a metric-bearing panel, prove the producer, merger, and evaluator on real
preflight artifacts of the same schema. During long jobs, keep routine health in
logs and report only state changes, failures, cost changes, and decision
boundaries. Label inferred GPU-hours as estimates.

## Output Contract

Report:

- current role, objective, worktree, and source surfaces;
- representative samples and why they were selected;
- executed probes and artifact roots;
- mechanism picture so far, with evidence scope and caveats;
- competing hypotheses and what became more likely, less likely, or unresolved;
- next branch-deciding probe or manual-review gate, chosen by information gain;
- conditional architecture implications only where evidence warrants them;
- changed files, verification commands, and skipped checks.
