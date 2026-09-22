---
title: Sampled-Rescue Object Transition Execution Plan
description: Minimal staged implementation, graphics-processing-unit scheduling, review gates, and adaptive subagent routing for the sampled-rescue causal-replay unit.
type: investigation
role: execution-plan
authority: non_normative_research
implementation_status: complete_for_unit
unit_id: 2026-07-14-sampled-rescue-object-transition-causal-replay
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-07-14
---

# Sampled-Rescue Object Transition Execution Plan

This plan supports the owning [research unit](unit.md). It deliberately avoids
freezing a reusable software interface before runtime evidence exists. The user
authorized implementation and graphics-processing-unit execution for Waves 0
through 4 on 2026-07-14. Training and architecture changes remain outside this
authorization.

## Success Criterion Before Construction

Construction may begin after the user approves a plan that has:

1. one falsifiable primary question and one strongest competing explanation;
2. four through eight named case candidates with direct-review roles;
3. exact fixed-prefix and donor-row semantics;
4. one representative smoke and explicit stop rules;
5. only conclusion-threatening runtime checks;
6. an owner for every implementation surface and no duplicate worker lanes.

## Minimal Execution Waves

Execution disposition: Waves 0, 1, 2, and 3 completed. No 32-sample Wave 3
expansion was needed because the one-greedy-plus-eight-sample complete-row
panel produced a clear failed crossover. Wave 4 did not enter because
first-description-token forcing already carried source-consistent phrase and
geometry through a complete row; the remaining uncertainty is cross-row
prefix-state factorization. See the verified [results](results.md).

### Wave 0: Existing-artifact anatomy

- Read the two completed full-image bagging roots and existing visual review.
- Verify inherited decode modes. The inherited `FULL_SINGLE` arm is sampled at
  temperature `0.4`, so add one exact neutral-repetition greedy anchor per
  retained image before rescue attribution; do not relabel `FULL_SINGLE` as
  greedy.
- Build a disposable case table containing image, trajectory, geometry mode,
  category variants, first divergence, rescue row, and the four boundary
  roles declared by the research unit.
- Freeze the per-boundary verified, covered, uncovered, uncertain, and
  unsupported object ledger before new replay outputs are inspected.
- Select four through eight cases. No graphics processing unit is required.
- Stop if raw trajectories cannot be bound back to exact prompts, prefixes,
  seeds, and parsed rows.

### Wave 1: One-image real smoke

- Implement only the missing seam needed to replay one exact prefix and project
  the immutable generated trace to its first complete row or terminal action.
  Retain the full raw trace; do not add a backend stopping-criteria interface
  because later causal tokens cannot alter the already generated first row.
- Construct the prefix through the current legal open-assistant continuation
  renderer; do not concatenate arbitrary token identifiers by hand.
- Run one image, one boundary, one greedy request with sampling disabled, and
  eight temperature-`0.4`, top-p nucleus-threshold-`0.95` sampled requests on
  one graphics processing unit.
- Use physical batch size four when request shapes permit it.
- Inspect raw text, token boundary, parsed row, seed uniqueness, and emitted-row
  token log probabilities before expanding. Add a teacher-forced alternative
  candidate scorer only if the empirical fixed-prefix panel remains ambiguous;
  no current inference seam owns that diagnostic.
- For adjacent nested donor arms, retain a token-diff trace proving that only
  the declared donor suffix changed.

This wave owns the first implementation truth. Mock-only and parser-only tests
cannot promote the unit to the next wave.

### Wave 2: Fixed-prefix distribution panel

- Run only cases and boundary roles that survived Wave 1.
- Partition immutable request identities across up to eight graphics processing
  units. Use one process per device and physical batch size four when shapes
  permit.
- Use all eight devices only when enough independent requests exist to keep
  them useful; do not create extra seeds or cases merely to occupy hardware.
- Start with eight seeds per temperature using top-p nucleus threshold `0.95`.
  Expand one state to 32 seeds only under the research unit's declared
  temperature-`0.4` confirmation rule.
- Report temperature-specific hit counts and binomial bounds; do not pool
  temperatures as one sampling distribution.
- Merge by image, exact prefix identity, boundary role, temperature, and seed.

### Wave 3: Token-level causal replay

- Run only sampled-rescue cases that show an auditable fixed-state mode or a
  clearly localized earlier divergence.
- Add the missing donor-span forcing seam only after Wave 2 passes. Ordinary
  generation fields are not treated as a token-forcing interface.
- Assign independent request partitions for continuation, description,
  coordinate, complete-row, wrong-object, and irrelevant controls.
- Release greedy generation immediately after the declared donor span.
- Review phrase owner, geometry owner, row validity, closure, and next-row
  redistribution before computing any broad aggregate.
- Run complete rescued-object, wrong-object, already-covered duplicate, and
  irrelevant-row controls before using the word commit in a result claim.

### Wave 4: Conditional residual replay

- Enter only after Wave 3 identifies a source-specific phase but fails to carry
  the intended behavior through the full row.
- Start from the historically supported layer band for that phase and compare
  correct donor, wrong-object donor, clean no-op, and source-swapped control.
- Use 32-bit floating-point reductions and rerun any numerically marginal case
  before interpretation.
- Stop after the smallest layer band answers the phase question; no broad head
  atlas belongs to this unit.

## Expected Source Ownership

The exact code interface remains deliberately unfrozen. Current expected
ownership is:

| Need | Reuse first | Add only if missing |
|---|---|---|
| Model and adapter composition | `src/inference/runtime.py::assemble_runtime`, `src/qwen/runtime_loading.py`, and `src/adapters/dora.py::load_inference_dora_adapter` | No new loader. |
| Decode request and generation | `src/inference/backend.py::DecodeGenerationPolicy`, `DecodeRequest`, and `HFGenerateBackend.generate_batch` | One experiment-local fixed-prefix and first-row projection seam; donor-span forcing remains a later gated seam. |
| Prompt and token boundaries | `src/inference/prompt.py::build_prompt_record`, its `AssistantContinuation`, `src/templates/`, and `src/qwen/encoding.py` | A legal arbitrary fixed-prefix helper only if the current continuation renderer cannot represent the selected prefix exactly. |
| Parsing and raw artifacts | `src/inference/parsing.py::parse_compact_object_box_closed` and `src/inference/artifacts.py` | Compact research receipt, not a new artifact framework. |
| Candidate forward scores | `src/eval/forward.py`, `src/qwen/forward.py` | Thin candidate-row scorer if current forward evaluation cannot expose token phases. |
| Analysis | Existing bagging artifacts and the historical implementation at `f5af926ba:src/analysis/spatial_scope_history/postrun_loader.py` (`load_postrun_evidence`) | Start experiment-local; promote to `src/analysis/` only after a second consumer. Do not inherit the prior sealed schedule unless its exact semantics are needed. |
| Thin command entry | `scripts/research/` | One command-line entry for the first consumer. |
| Visualization | `src/vis/` and prior review images | A small comparison sheet only when it changes case interpretation. |

Do not modify stable inference schemas or create an OpenSpec change unless the
smoke proves that a reusable compatibility-sensitive surface is actually
needed.

The existing step-4,887 inference configuration uses repetition penalty
`1.10`; this unit's primary decode policy is an explicit research-time override
to `1.0`, not a silent mutation of the production configuration. Existing
sampled-execution attestation is schedule-bound. Reuse only the minimum source
and runtime validation required by the backend, and revalidate the new fixed-
prefix request semantics instead of pretending that the old schedule covers
them.

The frozen configuration and checkpoint identities are declared in the owning
[research unit](unit.md). The historical filename token `gaussian_rps` expands
to **Gaussian Soft-Target Coordinate Cross-Entropy with Ordered
Cumulative-Distribution Penalty** and is provenance only; new code, reports,
and run identifiers must use the complete operational name rather than minting
another unexplained shorthand.

## Compact Run Receipt

Each executed request needs only:

- source commit or dirty-diff identity;
- checkpoint and resolved inference identity;
- image, boundary role, exact prefix hash, donor-row identity, condition, and
  seed;
- raw output, token boundary, parser status, and failure reason;
- candidate-path score primitives when used;
- one primary object-transition classification.

Do not add nested seals, live tensor inventories, exhaustive manifests, or
resume machinery unless the first smoke exposes an ambiguity that can change
the conclusion.

## Adaptive Subagent Routing

This unit pilots output-quality-based routing instead of assigning a powerful
model solely from task labels.

### Initial assignment

| Task shape | Agent capability | Starting model and reasoning effort | Inherited recent turns | Typical work |
|---|---|---|---:|---|
| Bounded discovery or artifact lookup | Repository scout | Generative Pre-trained Transformer 5.6 Luna (`GPT-5.6 Luna`) with medium reasoning effort | `1` | Locate paths, symbols, run roots, exact rows, and current owners. |
| Runtime-dependent smoke or attestation | Probe runner | GPT-5.6 Luna with medium reasoning effort | `1` | Execute the smallest command that proves runtime semantics and return receipts. |
| Thin mechanical implementation | Generic worker with explicit ownership | GPT-5.6 Luna with high reasoning effort | `1` | One experiment-local analyzer or command-line seam with focused checks. |
| Difficult but localized implementation repair | Same current worker, then one replacement if necessary | GPT-5.6 Luna with extra-high reasoning effort | `1` | Resolve a concrete hook, tensor-shape, batching, or parsing failure after a real smoke. |
| Scientific protocol, control, or artifact audit | Contract auditor | Generative Pre-trained Transformer 5.6 Sol (`GPT-5.6 Sol`) with medium reasoning effort | `2` | Challenge changed factors, controls, evidence scope, runtime identity, and artifact meaning. |
| Initial model-mechanism interpretation | Dedicated scientific reviewer with a model-behavior prompt | GPT-5.6 Sol with medium reasoning effort | `2` | Interpret causal replay and model behavior without expanding into contract review. |
| Unresolved causal contradiction | Model diagnostician | GPT-5.6 Sol with extra-high reasoning effort | `2` | One final synthesis only after focused lower-cost work and an initial GPT-5.6 Sol interpretation with medium reasoning effort leave the same high-severity contradiction. |

The Terra model family is not used in this unit. Full conversation history is
not inherited; current-unit work uses only the one or two recent turns declared
below.

Every subagent spawn must set `fork_turns` explicitly. This unit uses one
recent turn for bounded discovery, runtime probes, and implementation, and two
recent turns for scientific review or causal synthesis. It does not use full-
history inheritance.

Model and reasoning effort cannot be changed inside one subagent generation.
Adaptation occurs at a task boundary through a focused follow-up, reassignment,
or one routed escalation. Specialized agent roles whose model and effort are
runtime-fixed retain those settings; the lead changes the task route rather
than pretending to override them.

### Output-quality score

The lead scores each substantive subagent result from zero to two on five
dimensions. Zero means absent, contradicted, or unverifiable; one means partial
or weakly evidenced; two means directly evidenced and sufficient for the
assigned decision.

1. **Scope adherence**: stayed inside assigned files, mutations, and stop rule.
2. **Evidence traceability**: supplied exact paths, symbols, artifacts, or fresh
   checks rather than unsupported recollection.
3. **Semantic correctness**: preserved checkpoint, image, prefix, token,
   geometry, condition, and historical-versus-current meaning.
4. **Decision utility**: answered the assigned fork, named the strongest
   remaining risk, and did not bury the recommendation.
5. **Verification quality**: used the smallest check that proves the claimed
   work or clearly marked execution as skipped.

Acceptance requires a total score of at least eight, semantic correctness equal
to two, evidence traceability at least one, verification quality at least one,
and no hard failure. A causal-interpretation task also requires evidence
traceability equal to two. Six or seven points receives one focused follow-up
to the same agent. Five or fewer points, or failure of a critical dimension,
triggers routing by failure type rather than another blind retry.

Hard failures are: invented evidence, wrong checkout, unauthorized mutation,
historical/current authority conflation, hidden failed checks, or omission of a
control whose absence changes the scientific conclusion.

Priority zero means a finding invalidates the intended scientific result.
Priority one means a finding can materially change the interpretation or route
choice. Lower-priority findings do not automatically block the exploratory
smoke.

### Escalation rules

1. Correct missing details through `followup_task` on the same agent before
   spawning another agent.
2. Allow at most one initial pass and one focused follow-up at the same tier.
   Do not run a third cheap-agent retry unless the task definition changes.
3. Escalate factual uncertainty to a repository scout or execution probe, not
   automatically to a more expensive reasoning model.
4. Replace a failed implementation worker at most once and preserve valid
   discovery rather than repeating it.
5. Escalate interpretive scientific disagreement to one dedicated GPT-5.6 Sol
   reviewer with medium reasoning effort; missing evidence goes to an
   execution probe instead.
6. Use Sol with extra-high reasoning only when the same Priority zero or
   Priority one issue remains after a focused lower-cost attempt and review.
7. Preserve useful discovery from a weak synthesis; ask a stronger agent to
   reason over the evidence instead of repeating the lookup.
8. Do not create duplicate implementation lanes. One worker owns each code
   surface; another agent may audit or replace it, never race it.

If a replacement worker or one routed escalation still fails its acceptance
gate, end the task as `STOP_UNRESOLVED`; do not launch another automatic retry.

If GPT-5.6 Luna passes the acceptance gates, keep the next similar task on that
model family. If GPT-5.6 Sol with medium reasoning effort resolves the
scientific issue, do not escalate to extra-high effort merely because the topic
is important.

Every task ends in one explicit disposition: `ACCEPT`, `FOLLOW_UP`, `REASSIGN`,
`ESCALATE`, or `STOP_UNRESOLVED`. The lead records only a compact per-task
routing row: assigned task, capability, model, reasoning effort, first-pass
score, repair outcome, final score, disposition, escalation reason, and a rough
elapsed-time or tool-call cost signal when available. This is an efficiency
diagnostic, not a permanent leaderboard or model-quality claim.

## Review and Promotion Gates

- One scientific-design review occurs before implementation approval.
- After the first real smoke, review only risks exposed by execution.
- A failed non-critical review item becomes a limitation or promotion blocker;
  it does not automatically stop the exploratory run.
- A unit-wide rewrite is not justified by formatting, ideal abstractions,
  exhaustive reproducibility, or speculative future consumers.
- Training, a 256-image screen, architecture changes, and stable contracts each
  require a later evidence-backed decision. None is authorized here.

## User-Owned Gate

The user approved construction and bounded execution on 2026-07-14. The goal
closed without training or architecture promotion. The next Prefix-State
Phrase-Geometry Factorial is a separate research unit; the 256-image training
screen remains unauthorized.
