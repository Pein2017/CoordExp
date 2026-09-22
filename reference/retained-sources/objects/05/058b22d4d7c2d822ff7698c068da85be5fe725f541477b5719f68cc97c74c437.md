# Greedy compilation: fitting a route, reaching it, and finishing it

**Question:** which required decisions remain unlearned, which are not naturally reached, and which later decisions undo earlier success?

## Operational decomposition

For a fixed complete target token sequence, positive target-versus-all-competitor margin at every preceding target prefix, including EOS, implies exact greedy replay from the same starting condition when execution is deterministic and aligned. A lower **average** CE loss does not imply those individual inequalities. A certificate beginning at a supplied intermediate prefix does not imply natural access from an empty assistant prefix. Different acceptable full trajectories need not share one literal target sequence.

Thus measure conditional fitting, natural entry, complete free continuation, incumbent preservation and stopping separately. Do not infer owner damage from any literal token divergence; the owner-level consequence owns that claim. Two incompatible unique next-token/next-row demands at an identical condition cannot both be the unique greedy action.

## Evidence chain

[Coordinate-boundary training](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-256-image-coordinate-boundary-training-screen/unit.md) improved exact-prefix margins without improving clean rollout under the registered courses. [Mixed/refreshed corrections](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/results.md) changed candidate and target composition too, so they did not isolate a pure prefix-only cause.

[Fixed-witness route access](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-fixed-witness-route-access/results.md) found some likelihood gains without first-fork crossing. [Native entrance CE](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-native-entrance-ce-feasibility/results.md) did recover both chosen targets, but brought collateral burden. These results refute opposite shortcuts: neither “nothing learned” nor “the target learned, so the problem is solved” follows from a local metric.

In [label versus compilation](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/results.md), common-history continuations realized the trusted first row and immediate geometric successor, yet some later owner sets changed and a completed target could be repeated. For those cases, strengthening only the already-realized entrance omits the outstanding obligation. This does not uniquely diagnose exposure bias or a missing owner-memory slot.

## Prefix denoising: unresolved does not mean untried

The [original clean/noisy design](../../docs/history/research-records/2026-09-15-root-collapse/sources/research/ideas/prefix-denoising-sft/draft.md) optimized clean continuations under both clean and valid coordinate-noised histories, optionally matching selected clean/noisy coordinate distributions. Setting the optional KL weight to zero was still denoising CE, not denoising-OFF. After correcting branch leakage, the [root-cause analysis](../../docs/history/research-records/2026-09-15-root-collapse/sources/research/ideas/prefix-denoising-sft/experiments/2026-06-17-inert-objective-root-cause/unit.md) found almost unchanged branch distributions for the tested compact checkpoint. Its comparison used a different coordinate objective, training budget and adapter/full-merge surface, so it could not identify denoising benefit or harm.

The [axis-sort result](../../docs/history/research-records/2026-09-15-root-collapse/sources/research/ideas/prefix-denoising-sft/experiments/2026-06-16-axis-sort-negative-result/unit.md) separated making boxes materializable from recovering useful localization. The matched denoising-OFF question remains open; later self-rollout and coordinate interventions are related evidence, not that missing control. Perturbing a genuinely consumed history channel, including a self-rollout or relative-encoding variant, remains an alternative rather than a demonstrated fix. Do not transfer this compact checkpoint's insensitivity diagnosis to later source checkpoints.

## Current training-first task

[State](../experiments/2026-09-14-training-set-completion-curriculum/state.json) owns the latest boundary. [The accepted fourth-fit result](../experiments/2026-09-14-training-set-completion-curriculum/results.md) owns the fixed versus supplemental population, per-image changes, loss curve and review rules. The [original protocol and phase history](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-training-set-completion-curriculum/unit.md) is frozen provenance, not a live launch document.

The unchanged continuation improved natural coverage and stopping without a recipe/target refresh. This is evidence that optimization exposure was useful in that run. It does not establish convergence, the sufficiency of more steps, a unique cause of the remaining misses, or unseen-image transfer. A zero teacher-forced expectation-based geometry hinge coexisted with greedy geometry errors; it is not a hard raw-box guarantee.

The current goal permits different legitimate output orders and representations, requires cumulative per-image task completion before growth, and defers an early validation veto. Do not silently substitute canonical transcript perfection for owner coverage, or use old generalization concerns to veto this separately chosen training task.

## Reopening condition

Classify the current unresolved cases using already retained evidence before changing the learning problem. A next contrast should separate still-negative required margins, naturally visited-history errors, harmful later choices and loss of old owners. A failed fixed dose is not a general negative; a proposal for more dose or one refresh is not evidence that it will work. The current user stop remains in force until changed explicitly.
