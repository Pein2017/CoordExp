# Identical Implementation Benchmark Prompt

You are one of three independent implementation agents starting from an
identical CoordExp Git commit. Work only in the current checkout. Do not read,
diff, inspect, or copy from either sibling benchmark worktree.

## Objective

Implement the bounded rollout-only own-prefix entity-transition and coordinate-
boundary calibration pilot described by the owning Research Unit and OpenSpec.
The purpose of this run is to compare implementation quality across agents,
not to redesign the scientific treatment or launch the formal experiment.

## Required Reading

Read these files completely before editing:

1. `AGENTS.md`
2. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/unit.md`
3. `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/proposal.md`
4. `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/design.md`
5. `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/specs/coordexp-infras-own-prefix-calibration-training/spec.md`
6. `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/specs/coordexp-infras-supervision-losses/spec.md`
7. `openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/tasks.md`

Then inspect the canonical current owners through `docs/AGENT_INDEX.md`,
`docs/catalog.yaml`, and the narrow training modules referenced by those
routers. Reuse the existing coordexp-infras trainer, model composition, packing,
optimizer, checkpoint writer, inference, and artifact surfaces.

## Fixed Scientific Contract

- Training examples come only from frozen, reviewed, checkpoint-owned rollout
  state banks.
- Do not mix canonical supervised-fine-tuning rows into this pilot.
- Historical prefix tokens are conditioning input, not teacher-forced targets.
- Apply loss only at explicitly eligible rollout-derived causal sites.
- Every positive and harmful gradient-bearing site receives the small intended
  token-type gate defined by the spec.
- At a premature terminal boundary, the intended type is object-row schema,
  not terminal output.
- Keep entity trust and geometry trust separate. Unknown supervision produces
  zero gradient on the unknown axis.
- Entity-transition preference compares physical owners and supports multiple
  valid uncovered owners.
- Geometry preference targets only the first wrong coordinate with a reviewed
  discrete acceptable-token set.
- Recompute the full image and exact token prefix on every training forward.
  Do not use saved hidden states or key-value tensors as training input.
- Do not add a detector, object slot, object query, persistent ledger, crop
  policy, custom inference controller, second trainer, or alternate checkpoint
  format.
- Do not add full-row base cross-entropy, canonical replay, Kullback-Leibler
  anchoring, Gaussian coordinate targets, or online state-bank refresh.
- Keep ordinary supervised-training and inference defaults unchanged.

The completed 12-image human-refined dense cohort is blind evaluation data.
Do not use image IDs `1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348,
13923, 14038, 14439, 16228` for implementation examples, fixtures, smoke data,
candidate mining, tuning, or arm selection.

## Benchmark Scope and Shared-Fixture Boundary

Implement the minimum coherent code and focused tests corresponding to
OpenSpec tasks 1 through 4. Do not mark an acceptance item complete when its
real-fixture evidence is absent. Implement any task-5 smoke wiring that can be
tested without inventing scientific inputs.

The real lead-owned Smoke A and Smoke B fixture required by task 0.1 is not
part of this benchmark checkout. Do not select a substitute real event, tune a
private fixture, or claim that task 0.1 or the real task-5 smoke has passed.
Expose the smallest clear fixture-loading seam required by the spec and report
the missing shared fixture as the remaining external blocker. Synthetic unit-
test records are allowed only for deterministic code tests and must be labeled
synthetic; they are not scientific smoke evidence.

Do not launch the formal 256-image state-bank build or the 12-job graphics-
processing-unit training matrix. Do not modify the Research Unit or OpenSpec
scientific meaning to make implementation easier. If a genuine contradiction
is found, preserve it in the final receipt instead of silently choosing a new
contract.

## Execution Expectations

1. Confirm the checkout path, branch, starting commit, and clean status.
2. Inventory the current training and loss owners before adding code.
3. Make the smallest implementation that satisfies the normative specs.
4. Prefer focused tests and one narrow end-to-end synthetic or existing-safe
   pipeline smoke over broad infrastructure or speculative abstractions.
5. Keep 32-bit floating-point loss math at the grouped research-loss boundary.
6. Preserve atomic event groups across packing and optimizer-step planning.
7. Record raw and weighted losses, eligibility denominators, target margins,
   token-type legal mass, ignored/unknown counts, and finite status through the
   existing rank-zero artifact path.
8. Run strict configuration, state-bank, replay, loss, pipeline, checkpoint,
   and inference-composition checks proportional to the implemented scope.
9. Run `openspec validate add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot --strict` and `git diff --check`.
10. Commit the completed implementation to the current benchmark branch.

## Required Final Receipt

Create
`openspec/changes/add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot/implementation-receipt.md`
and include:

- starting and final commit;
- changed owner surfaces and why each was necessary;
- exact tests and commands run;
- passed, failed, and skipped checks;
- synthetic versus real evidence labels;
- known scientific or runtime limitations;
- task checklist status;
- the unresolved shared-fixture blocker; and
- any deviation from this prompt or the frozen OpenSpec.

Your implementation will be judged primarily on semantic correctness and
scientific-contract fidelity, then on focused verification, simplicity,
reuse of existing infrastructure, runtime behavior, and code quality. More
code or more abstractions do not receive extra credit.
