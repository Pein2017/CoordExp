# Early four-edge coordinate codebook: matched training pilot

## Authority, pair and question

2026-09-23. The user approved full left/right/top/bottom coordinate vectors,
early patch-edge injection, preservation of existing RoPE, and execution by
922-worker. This is a new bounded package, not reopening the closed predecessor.
Lead: `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d`; worker:
`01a0c726-ad7c-7cc0-89b7-d76ac6fcf027` (922-worker).
Live inspection found the worker idle at `gpt-6-sol/xhigh`, different from the
historical Astra/low setting. Preserve the current user-owned setting; do not
change either task's model or effort. Communicate directly in English using
[lead-worker](/data/CoordExp/.codex/skills/lead-worker/SKILL.md), including return
messages to the lead. No wake-me-up, watcher, or wake-then-fetch-chat chain.

From the same mature untied source, does replacing late masked-center injection
with early full-vector patch-edge injection improve native fitted-image
completion at the same three-loss 16-epoch dose, while containing output failures?

The [accepted predecessor](../2026-09-22-coordinate-codebook-three-loss/lead-results.md)
learned on the training panel but did not establish injection benefit; restoring
the auxiliary losses did not repair the prospective failure limits. The working
hypothesis is that early content/address interaction and explicit edge slots
make coordinate information more usable. Ordinary additional adapter capacity,
different optimization, and incomplete fitting remain alternatives. This is a
bundled architecture comparison: timing, center-versus-edge encoding, full-vector
use and a new projection change together. It cannot isolate early timing or
establish physical-instance binding. Patch edges are not object boundaries or
pixelwise annotations.

## Frozen architecture

Use the maintained model/trainer and the accepted live mature untied+axis001
step2444 source. Preserve its adapter tensors and both embedding deltas.
Disable the old main-merger codebook injection in the new candidate; do not
combine early and late injections or add new DeepStack injection hooks.

Inject once into raw patch-token states after the existing learned vision
position addition and before the first vision block. Preserve native vision
positions, vision RoPE, text MRoPE, token counts, masks and DeepStack topology.
The actual source has vision width1024, output-codebook width2048, patch size16,
and spatial merge2; bind these to the loaded source rather than general defaults.

For each raw patch, derive its footprint `(xL,xR,yT,yB)` in processed image
coordinates. Use the real processor's patch permutation, per-image boundaries
and crop/padding/resize mapping. Do not assume flattened row-major order. Use
the existing norm1000 coordinate convention: continuous lookup
`t=clip(1000*u,0,999)` and adjacent-row interpolation. The validity-loss margin
`1/999` does not redefine that convention. Include left/top0 and clipped
right/bottom999. Bind corner, interior, rectangular and packed-image examples.
No GT object coordinate may enter this construction.

`E(t)` is the full live effective OUTPUT coordinate row (base plus independent
trainable output delta), using only the1000 coordinate tokens. Let
`n(z)=z/sqrt(mean(z^2)+1e-6)` in FP32. Fixed concatenation slots carry edge/axis
identity; do not sum four undifferentiated vectors or delete alternating channels.

```text
s = concat(n(E(xL)), n(E(xR)), n(E(yT)), n(E(yB)))
z = P(s)                         # Linear(4 * output_width, vision_width), no bias
h_new = h + softplus(raw_gain) * stopgrad(RMS(h)) * n(z)
```

Initialize P with seeded Xavier-uniform, independently seeded so it does not
consume/change the inherited adapter initialization RNG stream. Gain starts
at0.05; it is trainable, not capped at0.05. Cast the residual at addition only.
Keep interpolation, normalization and projection differentiable into P, gain
and the live output delta. RMS normalization, not unit L2 normalization, defines
the intended initial relative residual scale. At exact zero/disabled gain the
new route must preserve the declared source computation. Do not require the
enabled architecture to reproduce source logits.

All intended three-tower all-linear rank16/alpha32 DoRA targets, independent
selected input/output deltas and the new P/gain train. Base weights, patch
convolution, norms and positional parameters remain outside the trainable
allowlist. Preserve mature targets and new-only execution-device magnitude
initialization semantics. Save/load P, gain and architecture metadata explicitly;
old late-injection checkpoints retain their own loader meaning.

## Matched recipe, data and execution

Reuse the exact accepted three-loss1024/256 identities, original retained32,
96-image training sentinel,492-pack cache and first984-update schedule. Each
image has16 presentations:7872global packs,16384image presentations. Seed1729;
four ranks times accumulation2; effective batch8packs; BF16/FA2. Start fresh
from the mature source, never from a trained late checkpoint or qualification.
Save62/123/246/492/984; only984 is the endpoint,492 is a diagnostic. No extra
seed, LR search, dose extension, early stopping on quality or checkpoint selection.

Keep the accepted AdamW settings, ten-step warmup then constant LR, weight decay0,
clip1.0 and dropout0. Language/aligner DoRA LR2e-5, vision5e-6, input/output
deltas1e-5, gain1e-3; new P LR2e-5/weight decay0. New parameter capacity is a
declared contrast difference. Freeze the resolved config and exact delta before
model entry. `data.eval` remains null; use maintained explicit evaluation queues.

The three defaults are CE1, token-type gate0.2 over desc_text/schema/coordinate/eos,
raw-axis validity hinge0.01 with margin1/999, gaussian0, segment_balanced.
Use existing implementations and global denominators. No auxiliary reconstruction
loss or other new training objective. Native evaluation retains the same prompt,
geo_sorted_xy targets, tokenizer, geometry, greedy policy, repetition penalty1,
3084 cap and parser/scoring semantics. No constrained decoding, sampling or NMS.

Reuse1280 accepted source cells and1280 late-three-loss16 cells, plus96 late8
sentinel cells if their identities pass. Produce1280 early16 cells and96 early8
sentinel cells. Thus1376 new and2656 reused cells;4032 analytical cells, not4032
independent images. Preserve provenance of predecessor reuse; no new replication
claim. Reuse failure is HOLD for the affected comparison, not permission to
quietly rerun source or change the denominator. Prior CE-only cells are context,
not a required new comparison arm.

Primary: paired train clean completion, IoU50/IoU80 coverage and coordinate
fidelity, especially the frozen dense stratum and retained32/additions992 split.
Report all bad/cap/owner-recurrent/severe transitions versus BOTH source and
late16, per-image severity and exact-row versus annotation-owner repetition.
UNKNOWN remains unmatched, not physically false. Retain source-relative limits
train51/10/51/10, validation12/2/12/2. Validation coverage/CE alone cannot select
or veto. All incomplete cells remain HOLD. Report numerical tradeoffs without
inventing an after-outcome improvement threshold or claiming promotion.

## Address-use diagnostic and evidence limits

On the frozen original32 training images at final984, authorize at most96
additional teacher-forced forwards: correct addresses, identity replay through
the intervention path, and one fixed within-image cyclic permutation of whole
four-edge tuples (shift `max(1,N//2)` for N raw patch tokens). Keep image pixels,
GT history, native positions, model weights and each recipient token's visual
RMS scale fixed. Freeze the cases/order/shift before outcomes. Identity replay
must meet the inherited2e-4 numerical tolerance. Singleton/no-change grids
are ineligible for the corruption contrast, not negative evidence.

Compare coordinate-token versus description-token full-vocabulary CE with fixed
denominators, by image and role. This checks sensitivity to address corruption;
it does not by itself prove retained exact numerical addresses, semantic
specificity of the perturbation, component training benefit or owner binding.
Report residual/gain scales at the actual injection and preserve the distinction
between exact constructed input addresses, decoder sensitivity and actual native
localization. Do not claim that information survived simply because it was added.
This secondary diagnostic does not select training or veto the primary comparison;
if infeasible, close it HOLD and complete the main fixed experiment.

## Qualification, ownership and stop

Cwd: `/data/CoordExp/.worktrees/research-probes`.
Output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-early-edge-codebook`.
Use existing `probes/training_set_completion/coordinate_codebook_alignment/`
drivers, configs and reducers; no second trainer or generic execution framework.
Worker owns its execution/candidate/delegation notes in this unit, new run
artifacts and scoped implementation/tests. Lead owns unit/state/catalog/frontier
and acceptance. Do not edit predecessor records, annotations or common assets.

Allowed shared seams are codebook implementation and its direct config,
optimizer, pipeline and checkpoint/inference consumers, plus relevant tests.
Inspect dirty ownership and freeze an exact file allowlist before edits; preserve
other work. Escalate a different shared seam or architecture/meaning change.
Use source captures outside outputs; do not execute captures or write code/docs
under outputs. Reuse qualification evidence only where the executed surface
is unchanged.

Before broad fitting, run the smallest real training/save/fresh-load/generation
slice that proves patch order, edge-slot identity, differentiability through
live output rows, intended trainable updates and frozen source identity. Include
wrong-order/edge-swap/detached-codebook mutations, source OFF parity on the same
three qualification cases, and old-checkpoint loading compatibility. Preserve
valid LR0, zero hinge and delayed-gradient states. Independently check the actual
four-rank first-two-update loss/gradient/denominator and allowlist; passing genuine
production updates remain in the trajectory. No redundant replay under a new name.
Reuse accepted distributed comparisons where unchanged; do not build another
acceptance ceremony around identical plumbing.

Use the same bounded8wall-hour/64allocatedGPU-hour envelope for this new package,
including failures, qualification, diagnostic and evaluation. Start the clock at
first model-entry process; never reset it. Keep artifacts within16GiB, reserve
in-flight costs, stop new model admission by7.75h and preserve unfinished HOLDs.
Measure early-injection memory/throughput with true grid sizes before broad work.
Use all8 GPUs for independent useful work: four-rank training and ready eval on
the other four, then all8 for the remaining queue. Do not repeat already accepted
cells to occupy devices. Do not wait for ambient stress occupancy to disappear.

After mechanical qualification passes, continue the fixed fit/evaluation within
this contract without another ACK. Report first real entry, material conflict
and final candidate directly. Ordinary repairs remain local within the envelope;
preserve attempts and return changed science/cost/authority questions promptly.
Stop at the stable candidate; no self-acceptance, successor, full-data training,
commit, publication, global skill or memory edit.

Worker may reuse Luna children for disjoint bounded implementation/checks,
normally Luna-max for the fragile live-codebook/caller checks, recording actual
model/effort. Parent retains config/integration/launch responsibility. In one
delegation-notes.md retain the original brief, semantic correction, concrete
caller/counterexample evidence and any parent takeover; no model-ranking claim.
Apply the prior lesson: explicit effective defaults, legitimate zero cases and
whole promised artifact compatibility, not leaf-test counts alone.
