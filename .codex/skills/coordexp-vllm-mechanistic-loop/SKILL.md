---
name: coordexp-vllm-mechanistic-loop
description: Use when continuing or starting a long-running CoordExp vision-language large language model (V-LLM) mechanistic research loop from checkpoints, rollout artifacts, hidden states, attention routes, residual/logit-lens probes, coordinate-token basins, duplication, false-negative, or termination failures, coverage-ledger or prefix-denoising comparisons, or user asks for self-driven experiment design beyond surface metrics.
---

# CoordExp Vision-Language Large Language Model (`V-LLM`) Mechanistic Loop

## Overview

Act as a self-driven lead investigator for CoordExp vision-language large
language model (`V-LLM`) mechanisms. The job is not to prove that a phenomenon
exists or that a method is effective; it is to isolate the deepest reachable
origin of behavior with artifact-backed probes, cautious interpretation, and
durable research notes.

This skill complements `model-diagnosis`. Use `model-diagnosis` for immediate symptom triage; use this skill when the user wants a long research loop that can design probes, run them, follow promising branches, and build a mechanism picture over many turns.

## Role Contract

- Treat rollout metrics as sample selectors and sanity checks, not as the final
  object of interest. Tiny mean Average Precision movement can still hide
  important internal behavior shaping.
- Prefer sample-base-centered deep probes over broad analysis of normal or well-learned images. Pick representative images where checkpoint differences are large, comparable, or mechanistically revealing.
- Stay artifact-first: start from exact checkpoints, rollout roots, configs,
  prompt/template surfaces, sample identifiers, trace files, and prior notes
  before explaining.
- Be self-driving after the user grants permission. Choose the next promising
  probe, use available graphics processing units when explicitly allowed, and
  keep moving until the mechanism picture converges, directions are exhausted,
  or a research-meaning gate appears.
- Branch dynamically when a path is attractive and likely to influence the final picture. A roadmap guides the loop; it must not trap the investigation away from better evidence.
- Ask for user review when data analysis cannot choose a representative sample confidently, when the fork changes research meaning, or when a high-cost or irreversible action lacks prior permission.
- Stop cleanly when the user asks for a break: finish the current processing slice, write the durable note, verify, commit if requested, and do not open a new branch of exploration.

## Start Of Run

1. Bound the current lane:
   - exact worktree and branch;
   - whether to reuse an existing worktree or create a clean one;
   - user-named checkpoints, adapter surfaces, artifacts, and notes;
   - available graphics processing units and cost permission;
   - stop condition.
2. Set or refine a `/goal` for long runs. If the user's words are rough, translate them into a precise objective with mechanism targets, contrast axes, evidence surfaces, and manual-review gates.
3. Load local routing first: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, relevant specs/docs, prior `research/` notes, legacy `progress/` provenance when explicitly relevant, and existing analysis scripts.
4. Select or create one owning research unit before implementation. Write a
   compact outline: one falsifiable question, strongest alternative, primary
   contrast, smallest observation, expected owner surfaces, reused
   infrastructure, non-goals, stop rule, rough cost, and logical
   `outputs/research/<investigation>/<unit-id>/<run-id>/` root. Use
   `coordexp-research-knowledge-workflow` for the unit lifecycle. Do not freeze a
   speculative code interface or production-shaped protocol for an exploratory
   probe.
5. If the user asks to grill or record local decision context, use `grill-me record=local` before implementation. Resolve discoverable facts and reversible implementation details yourself; for user-owned research decisions, follow `grill-me` and ask exactly one recommended question at a time.
6. Inventory existing probes before writing new ones. Reuse or extend local analysis surfaces for hidden states, attention evidence, residual streams, source ablation, coordinate logit locality, duplication collapse, and visualization.

## Evidence-First Research Flow

Use this default order:

1. compact research outline;
2. one scientific-design review of the contrast and strongest confounds;
3. minimal implementation on experiment-local surfaces;
4. representative real graphics-processing-unit smoke;
5. sample-level visualization or behavior anatomy;
6. targeted audit of risks exposed by the smoke;
7. stop, revise, or promote.

Do not require every review finding to close before the first model observation.
Block an exploratory run only when a finding could change the scientific
conclusion by running the wrong input/model/prompt, corrupting semantic or
geometry alignment, changing more than the declared factor, losing request
attribution, or invalidating the primary observation. Record other findings as
known limitations or promotion blockers.

For inference-led case studies and mechanism probes, start with four to eight
carefully selected cases and normally stay at or below sixteen. Use metrics to
select and sanity-check cases; when labels, matching, or aggregation are known
to be incomplete, let direct visual and trace review own the interpretation.
Scale only after confirming that the measured quantity represents the intended
phenomenon.

For training-led probes, use progressive data scales such as 256, 512, 1024,
and 2048 samples. Maximize physical batch size, packing efficiency, sequence
length utilization, data throughput, and available graphics-processing-unit
occupancy. Preserve matched effective examples or tokens per optimizer update,
loss normalization, optimizer schedule, and packing isolation across compared
arms unless one of those is an explicit experimental variable.

## Minimal Implementation And Runtime Assurance

- Expand the implementation only to obtain the primary observation or protect
  its interpretation. Defer future reuse, broad resume behavior, exhaustive
  manifests, and unobserved edge cases.
- Keep the first consumer experiment-local. Promote shared interfaces only
  after another real consumer demonstrates the same semantic seam.
- Prefer existing model loading, batching, workers, parsing, scoring, and
  artifact writers. Add only the missing intervention or observation seam.
- For an exploratory run, retain a compact receipt containing source identity,
  checkpoint/config/case identities, request conditions and seeds, raw output,
  parser/failure status, and the representative smoke result.
- Do not add live tensor inventories, mutation seals, adversarial capability
  objects, replay attestations, or parity systems unless a demonstrated runtime
  ambiguity could alter the primary conclusion.
- Within an authorized `/goal`, let the lead adapt controls, diagnostics,
  implementation, and small recursive branches. Seek user direction for a
  major route change or material added critical-path time; about two hours is a
  flexible discussion trigger rather than a rigid cap.

## Comparability Matrix

Before comparing checkpoints, make a small matrix that records:

- checkpoint path, adapter/loading surface, and whether modules such as a
  multilayer-perceptron aligner, token-embedding adapter, Low-Rank Adaptation
  (`LoRA`), Weight-Decomposed Low-Rank Adaptation (`DoRA`), coordinate offsets,
  or coverage-ledger heads differ;
- prompt/template, tokenizer, special token surface, and `<|coord_0|>` through `<|coord_999|>` handling;
- dataset slice, image identifiers, row/order policy, bounding-box
  representation, and `do_resize=false`;
- free rollout vs teacher forcing, grammar state, row separator, temperature, repetition penalty, `max_new_tokens`, and parser/drop status;
- artifact roots for raw output, scored predictions, token traces, sidecars, visualizations, and probe outputs.

Avoid coarse labels such as `checkpoint_family` when they hide the actual mechanism surface. Compare the effective surfaces that can change behavior.

Here `do_resize=false` means the image processor must not resize the input;
`max_new_tokens` is the maximum number of generated tokens; `x1` and `y1` are
the left and top bounding-box coordinates; `x2` and `y2` are the right and
bottom coordinates; `box_end` is the closing bounding-box wrapper token; and
`desc` is the object-description span. The literal token prefix `coord` means a
discrete coordinate token.

## Probe Ladder

Move from cheap structure to causal mechanisms, but do not stop at the cheap layer when the user has asked for roots.

1. **Case and onset ledger**: classify parse validity, emitted count, pair onset, component onset, component growth, normalized row position, desc/class, spatial basin, and stop/termination behavior.
2. **Slot and boundary readouts**: keep `x1`, `y1`, `x2`, `y2`, `box_end`, stop/continue, and type/schema tokens separate. Do not collapse them into one aggregate score.
3. **Prefix and guidance splits**: compare full history, no-history,
   wrong-control, same-description competitor, ground-truth-guided, and
   model-generated bad-prefix conditions when relevant.
4. **Coordinate-token geometry**: inspect special coordinate token embeddings,
   language-model-head/readout neighborhoods, valid-token mass, target-bin ranks,
   and local smoothness. Stable schema/type loss is helpful evidence, not a
   reason to skip readouts.
5. **Hidden-state and residual flow**: trace where object identity, region availability, coordinate-basin preference, and target-bin margins form across layers.
6. **Attention routing**: use attention as evidence for candidate routes, not as causal proof by itself.
7. **Causal interventions**: patch, ablate, mask, or source-swap only after the row/window/slot is pinned. Prefer paired controls across checkpoint groups.
8. **Sublayer and final-layer decomposition**: when a layer transition looks
   decisive, split attention, multilayer-perceptron, residual, and final
   normalization/language-model-head effects before making a mechanism claim.

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
- For an exploratory note, retain only the scope and changed factor,
  checkpoint/config/case identities, artifact handle, primary raw output or
  trace, primary visualization or metric primitive, observed behavior,
  limitations, and next discriminator. Include the exact command or script only
  when it is needed to locate or rerun the observation.
- Keep disposable scratch under `temp/`. Put thin experiment command-line
  interfaces under
  `scripts/research/`; promote repeated semantic logic to `src/analysis/` with
  tests. Do not revive legacy `scripts/analysis/` as the default owner.
- Resolve durable run artifacts under `/data/CoordExp/outputs/research/` so they
  survive temporary-worktree retirement. Use the compact exploratory receipt
  defined above. After a result is promoted to decision-grade evidence, add
  authored/resolved config identities, declared invariants, complete material
  failures, metric primitives, analyzer identities, and terminal status.
- Keep scientific hypotheses, cohorts, thresholds, and verdicts in the research
  unit. Use OpenSpec only when a reusable implementation or compatibility
  contract is authorized separately.
- When using parallel graphics-processing-unit jobs, make split-run merge keys
  collision-safe. Include checkpoint, image/sample identity, candidate identity,
  source specification, history variant, slot, and mode as needed.
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

- Reporting Average Precision or mean Average Precision deltas as the answer
  when the user asked for internal behavior shaping.
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
