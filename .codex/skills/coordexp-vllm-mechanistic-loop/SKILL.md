---
name: coordexp-vllm-mechanistic-loop
description: Use when continuing or starting a long-running CoordExp V-LLM mechanistic research loop from checkpoints, rollout artifacts, hidden states, attention routes, residual/logit-lens probes, coordinate-token basins, duplication/FN/termination failures, coverage-ledger or prefix-denoising comparisons, or user asks for self-driven experiment design beyond surface metrics.
---

# CoordExp V-LLM Mechanistic Loop

## Overview

Act as a self-driven lead investigator for CoordExp V-LLM mechanisms. The job is not to prove that a phenomenon exists or that a method is effective; it is to isolate the deepest reachable origin of behavior with artifact-backed probes, cautious interpretation, and durable research notes.

This skill complements `model-diagnosis`. Use `model-diagnosis` for immediate symptom triage; use this skill when the user wants a long research loop that can design probes, run them, follow promising branches, and build a mechanism picture over many turns.

## Role Contract

- Treat rollout metrics as sample selectors and sanity checks, not as the final object of interest. Tiny mAP movement can still hide important internal behavior shaping.
- Prefer sample-base-centered deep probes over broad analysis of normal or well-learned images. Pick representative images where checkpoint differences are large, comparable, or mechanistically revealing.
- Stay artifact-first: start from exact checkpoints, rollout roots, configs, prompt/template surfaces, sample IDs, trace files, and prior notes before explaining.
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
2. Set or refine a `/goal` for long runs. If the user's words are rough, translate them into a precise objective with mechanism targets, contrast axes, evidence surfaces, and manual-review gates.
3. Load local routing first: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, relevant specs/docs, prior `research/` notes, legacy `progress/` provenance when explicitly relevant, and existing analysis scripts. Use `coordexp-router-context` when navigation is nontrivial.
4. If the user asks to grill or record local decision context, use `grill-me record=local` before implementation. Batch obvious decisions with recommendations; reserve one-at-a-time questions for forks that change research meaning.
5. Inventory existing probes before writing new ones. Reuse or extend local analysis surfaces for hidden states, attention evidence, residual streams, source ablation, coordinate logit locality, duplication collapse, and visualization.

## Comparability Matrix

Before comparing checkpoints, make a small matrix that records:

- checkpoint path, adapter/loading surface, and whether modules such as MLP aligner, token embeddings adapter, LoRA/DoRA, coord offsets, or coverage-ledger heads differ;
- prompt/template, tokenizer, special token surface, and `<|coord_0|>` through `<|coord_999|>` handling;
- dataset slice, image IDs, row/order policy, bbox representation, and `do_resize=false`;
- free rollout vs teacher forcing, grammar state, row separator, temperature, repetition penalty, `max_new_tokens`, and parser/drop status;
- artifact roots for raw output, scored predictions, token traces, sidecars, visualizations, and probe outputs.

Avoid coarse labels such as `checkpoint_family` when they hide the actual mechanism surface. Compare the effective surfaces that can change behavior.

## Probe Ladder

Move from cheap structure to causal mechanisms, but do not stop at the cheap layer when the user has asked for roots.

1. **Case and onset ledger**: classify parse validity, emitted count, pair onset, component onset, component growth, normalized row position, desc/class, spatial basin, and stop/termination behavior.
2. **Slot and boundary readouts**: keep `x1`, `y1`, `x2`, `y2`, `box_end`, stop/continue, and type/schema tokens separate. Do not collapse them into one aggregate score.
3. **Prefix and guidance splits**: compare full history, no-history, wrong-control, same-desc competitor, GT-guided, and model-generated bad-prefix conditions when relevant.
4. **Coordinate-token geometry**: inspect special coordinate token embeddings, LM-head/readout neighborhoods, valid-token mass, target-bin ranks, and local smoothness. Stable schema/type loss is helpful evidence, not a reason to skip readouts.
5. **Hidden-state and residual flow**: trace where object identity, region availability, coordinate-basin preference, and target-bin margins form across layers.
6. **Attention routing**: use attention as evidence for candidate routes, not as causal proof by itself.
7. **Causal interventions**: patch, ablate, mask, or source-swap only after the row/window/slot is pinned. Prefer paired controls across checkpoint groups.
8. **Sublayer and final-layer decomposition**: when a layer transition looks decisive, split attention, MLP, residual, and final norm/LM-head effects before making a mechanism claim.

Good probe families include object pointer trajectory, guidance separability matrix, visual-language binding, prior-object history routes, coordinate target-readout margins, false-negative visual-vs-language guidance, duplication onset dynamics, and termination/boundary control.

## Interpretation Rules

- Separate visible symptom, candidate mechanism, causal evidence, and unresolved alternative.
- Prefer "probe handle mismatch" or "inconclusive-needs-mechanism-panel" over "mechanism absent" when a negative result does not actually target the suspected surface.
- Treat attention-only findings as route hypotheses until paired with ablation or patch evidence.
- Distinguish a local slot rescue from full object-span recovery. A patch that fixes `y1` but fails `x2/y2/box_end` is evidence for split onset vs span-binding mechanisms.
- Distinguish x1-boundary basin selection from later coordinate-slot target-bin movement. The first coordinate can choose a route while later slots still determine available-vs-duplicate geometry.
- Do not let aggregate metrics override stronger sample-level internal evidence without reconciliation.
- For coverage-ledger, prefix-denoising, or other perturbation checkpoints, interpret them as controlled representation-shaping interventions unless the training objective and runtime contract prove a stronger claim.

## Artifacts And Notes

- Write new research knowledge under `research/`. Treat `progress/` as
  deprecated legacy diagnostics/provenance; migrate useful old material rather
  than adding new records there, except when the user explicitly asks to preserve
  an older branch's format.
- Every durable note should include scope, checkpoints, configs, artifact roots, sample IDs, commands or scripts, evidence scope, core tables/figures, interpretation, caveats, and next probe seeds.
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

## Common Mistakes

- Reporting AP/mAP deltas as the answer when the user asked for internal behavior shaping.
- Studying only normal images instead of high-value divergent samples.
- Using the first duplicate-like pair as the burst onset when the largest component starts later.
- Comparing checkpoints without aligning prompt/template, adapter/module loading, decode protocol, and token surface.
- Treating attention as causal proof.
- Porting every old helper into a new worktree instead of referencing prior tools and copying only what is necessary.
- Flattening slot-wise or condition-wise evidence into a single metric.
- Ignoring special coordinate-token embeddings or newly trained token surfaces.
- Ending after a shallow observation when the user has granted a self-driven, cost-tolerant mechanism search.

## Output Contract

Report:

- current role, objective, worktree, and source surfaces;
- representative samples and why they were selected;
- executed probes and artifact roots;
- mechanism picture so far, with evidence scope and caveats;
- next recommended branch or manual-review gate;
- changed files, verification commands, and skipped checks.
