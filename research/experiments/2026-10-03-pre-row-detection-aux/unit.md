# Pre-row detection auxiliary on a fixed self-prefix bank

## Question and authority

On the existing step256 anchor and 18-image / 570-label cohort, does adding a
training-only pre-row class and joint-box objective to the existing full-label
row objective improve ordinary greedy generation after matched fixed-prefix
training? User requested preparation of this next research experiment on
2026-10-03 and selected a lead-worker running GPT-6.1-Sol at high effort. This
reopens this unit's preparation only; closed overnight units remain closed.

The lead owns scientific design, input/target selection, release and acceptance.
The persistent worker owns its assigned implementation and package checks. The
initial assignment is CPU discovery/preparation; no GPU/model invocation is
released. The eventual exact native packet and cost require lead acceptance of
the prepared source and bounded execution path. State and results belong here;
this is not authority to schedule another unit.

## Contrast and predecessor boundary

Both arms will use identical frozen input histories, positive row targets,
base-loss reductions and optimization exposure. A uses the existing sequence /
owner-region row objective; B adds one training-only auxiliary predicting the
same verified target's class and full box from a causal pre-description state.
Auxiliary gradients must reach the generator's trainable parameters. Inference
uses the ordinary autoregressive path without the auxiliary head, cache edits,
coverage masks or new control tokens. A is a new fixed-bank baseline, not a
reproduction of the historical online-refresh trajectory.

The full-label predecessor already uses GT owner regions on self-rollout-derived
contexts: [computation](../2026-10-02-full-label-self-rollout-fit/unit.md) and
[results](../2026-10-02-full-label-self-rollout-fit/results.md). Selected local
fitting and finite greedy fitting are established on other tasks; they do not
answer whether this direct auxiliary helps current native outcomes. Earlier PVCI
proposal supervision learned target-related information without reliable native
transfer; feedback routes incurred nonselective burden. See
[visual use](../../questions/visual-designation-and-causal-use.md). This unit
tests class/joint-box supervision without adding a feedback route.

The strongest alternatives are an auxiliary head that learns without useful
generator transfer, extra gradient exposure rather than better task structure,
and shared-parameter interference that exchanges previously covered owners.
Neither head accuracy nor falling conditional loss is the primary outcome.

## Inputs and preparation boundary

Use the predecessor's full-label snapshot, manifest and named shared
`start-loss-instance-margin-order17-step256/payload` checkpoint. The initial CPU
candidate source for a frozen bank is version0 of the accepted balanced
constant-LR invocation, located by `lower_lr_followup.arms.constant.run_output`
in the predecessor's state.json. Reuse immutable raw records, prompt IDs,
credit plans and their accepted identities; do not rewrite historical artifacts,
run old source, silently substitute a baseline, or regenerate trajectories.

Before implementation, the worker reports exact available inputs, branch/row
counts, pre-row positions, ambiguous image-plus-prefix targets, current model
loading/training/export/evaluation entrypoints and the smallest integration.
The lead then freezes numerical loss/optimizer choices, event eligibility,
step/evaluation schedule and resource bounds here. A conflicting target at an
identical causal input is a design question, not permission to silently choose a
different object or supervise a single-box mean.

## Lead-frozen CPU implementation design

Freeze version0 records and credit plans once. Reuse the full-label branch
schedule and all existing trace/CHAIN/redirect loss terms and reductions in A
and B. Neither arm refreshes its bank from its own later outputs. Both start
independently from the same anchor with fresh optimizers and run a proposed
16 full-bank updates; no dose extension or best-checkpoint selection. This is a
single finite-dose screen, not a convergence claim.

Auxiliary eligibility is every positive owned row actually supervised by that
schedule: retained trace M where present, CHAIN B and relocated M, and positive
redirect rows. Exclude legal-only/unknown rows, harmful continuations and EOS.
Use each owner's annotation GT class and box, never the sampled row's point box.
Read the final post-normalization language hidden state at the consumed
`<|object_ref_start|>` token, before any target description or coordinate token.
Preparation must derive and assert its absolute position from the actual tokens.

Group by image/input identity plus the exact prefix through that opener. If a
group has different class/GT-box targets, exclude the entire group from the
auxiliary only, preserving A/B's baseline objective unchanged. Keep and report
all excluded rows, owners, branches and total weight. Same-target aliases retain
their existing weighted multiplicity. The lead reviews the actual inventory
before native release; a loss of the correction population changes that decision.

Use one FP32 affine projection from the row state to C+4 values. C is the sorted
set of exact description strings in the frozen 570-label input; there is no
background, unknown-negative or globally-complete class. For four outputs a,b,c,d,
predict normalized xyxy as x1=sigmoid(a), y1=sigmoid(b),
x2=x1+(1-x1)*sigmoid(c), y2=y1+(1-y1)*sigmoid(d), and compare with GT/1000.
This has nonnegative extents; finite-precision saturation can be degenerate and
must remain finite and visible in diagnostics, not silently filtered or claimed
to guarantee strict generated-box legality. Use a numerically defined GIoU for
such boxes and the positive GT boxes.

Per-row auxiliary is CE/log(max(C,2)) + mean-four-coordinate L1 + .5*(1-GIoU).
Within an image, normalize by the sum of eligible positive-row weights, retaining
their existing relative M/B/redirect coefficients. Then average over all18
images, with zero contribution for an empty eligible image. B adds .1 times
this auxiliary; A adds none. This is an explicit first dose, not gradient-matched
or optimized. A positive result identifies the combined auxiliary including its
stricter GT point-box target, not uniquely a shorter credit path.

Keep the existing backbone trainables, base learning rates, AdamW parameters and
backbone clipping: language DoRA1e-5, independent input/output deltas5e-6,
betas(.9,.999), eps1e-8, weight_decay0, clip1, seed92711. The head uses its own
AdamW at1e-3 with the same betas/epsilon/weight-decay and separate clip1; it must
not enter the backbone clipping denominator. Initialize the head with an isolated
seeded RNG so it cannot perturb shared runtime randomness; record the exact
initializer in preparation. Head state is saved separately for reproducibility,
excluded from the ordinary deployed model. No warmup, lambda sweep or trainable
module expansion. Tiny CPU fixtures must falsify accidental detach and leakage.

Proposed primary evaluation is ordinary empty-history native greedy at the shared
anchor and both step16 endpoints on all18 images, retaining full570 references and
the original decoder limits. Preparation must also enumerate a bounded conditional
free-row evaluation from saved correction prefixes, label actual versus synthetic
histories explicitly, and return its exact selection/counts for the lead to freeze.
Do not manufacture eligible cases. No head prediction is substituted for a decoded
box. Exact forward, request, token, artifact and wall bounds will be derived from
the frozen input inventory before any runtime release.

CPU acceptance includes input/owner/position binding, ambiguous-prefix handling,
unchanged A/B base-loss reductions and gradients, nonzero auxiliary gradients into
shared trainables, finite box loss at saturated outputs, independent optimizer
clipping, saved head/export separation, and persisted consumer rejection of identity
drift. Reuse existing meaningful tests and pure functions. The current
`CaptureHiddenRows` is intentionally detached diagnostic storage and cannot provide
the trainable path; reuse text-stack resolution and a minimal differentiable
collection at the selected norm output without changing diagnostic semantics.

## Outcomes, limits and stop

Retain conditional free-row completion and empty-history greedy outcomes as
separate evidence. Report annotation-relative coverage, fixed-baseline
gains/retention/losses, duplicate/near-repeat, invalid/malformed, EOS/cap and
resource costs. Preserve unknown or annotation-unmatched cases as unknown;
known labels do not exhaust physical reality. Improvements at a proxy or fixed
prefix do not establish physical-FN recovery, causation, robustness or held-out
generalization. No perfect-score or per-update zero-regression quality gate.

Current stop: a CPU-prepared, reviewable package with exact launch proposal,
meaningful checks and unresolved runtime risks. No model load/forward, GPU
qualification, training, new generation, checkpoint selection, native retries or
parameter sweeps are released by the initial assignment. Tiny CPU fixtures and
tokenizer/processor metadata preparation are allowed; scientific changes return
to the lead. No modification of historical inputs, shared assets or `/external`.

Storage: maintained implementation in `probes/` (reuse `src/` mechanisms), tests
in `tests/probes/`; this unit owns protocol/state/results; disposable messages
in `.local/scratch/pre-row-detection-aux-12/`; machine evidence and future runs
in this worktree's `outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/`.
