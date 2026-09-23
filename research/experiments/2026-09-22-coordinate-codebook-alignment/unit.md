# Live coordinate-codebook alignment: fitted-image training pilot

Date: 2026-09-22. Status: user-authorized implementation and bounded execution;
no scientific result or lead acceptance yet.

## Authorization and ownership

The user selected one recipe, without architectural ablations: mature untied
checkpoint, trainable input/output coordinate embeddings, all-linear DoRA,
early visual addresses linked to output coordinate rows, and canonical
`geo_sorted_xy` SFT. The user subsequently made fitted-image natural generation
the first decision surface, with held-out evidence informative rather than an
early veto, and explicitly requested execution through 922-worker plus reusable
training infrastructure. This is a new grant; the closed address-readout pilot
and its budgets remain closed.

- Lead: `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d`.
- Persistent worker: `922-worker`, `01a0c726-ad7c-7cc0-89b7-d76ac6fcf027`,
  user-selected `gpt-6-astra` / `low`; verified idle with these settings before
  dispatch. Do not change task settings or create another sidebar main.
- Cwd: `/data/CoordExp/.worktrees/research-probes`.
- Direct reports: use the existing lead-worker transport with this pair and
  `--to lead`; no wake marker, watcher, or request to reread worker chat.
- Lead owns this protocol, state, catalog, frontier, scientific amendments and
  acceptance. Worker owns implementation, frozen execution manifests, runtime,
  candidate results, and its implementation children. Routine repair is local;
  report decision-bearing failures promptly. There is no arbitrary repair-count
  ceiling and no worker authority to change the scientific program.

## Question, predecessors and interpretation

From the mature untied+axis001 step-2444 anchor, can a live output-codebook visual
address connection, jointly trained with selected input/output rows and three-
tower DoRA using full-response SFT, learn reliable empty-prefix greedy completion
on one frozen small training panel within a bounded optimization search?

The previous [address-readout pilot](../2026-09-22-address-readout-pilot/lead-results.md)
trained a small module over a frozen tied backbone and failed fresh transfer.
This package replaces that interface and trains the agreed upstream surfaces;
it is not another dose of that closed recipe. Existing
[pure-CE Human13 fitting](../2026-09-05-human13-pure-ce-replay/results.md)
already establishes ordinary finite-panel fitting. Another successful fit is
execution/feasibility evidence, not proof that the new address connection is
necessary or that the original burst cause has been identified.

Prioritized hypothesis: task-visible visual positions and output coordinate
decisions benefit from a shared learned coordinate basis. Strong alternatives:
the network memorizes the fitted image/sequence pairs, or unresolved instance
selection/history policy remains the limit. This single-recipe package cannot
isolate causal credit among its trainable components. Preserve that limitation.
No fresh-transfer improvement is required to pass this first feasibility stage.

## Frozen source, data and supervision

Start from `mature-untied-axis001-xy-step2444` in
[RESEARCH_ASSETS](../../assets.md), including its original base,
DoRA adapter and BOTH selected-token deltas. Checkpoint root:
`/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444`.
Bind actual source/config/tokenizer/processor/payload identities. This source
also differs in historical axis loss and optimization; it is not an untie-only
comparison. Historical auxiliary losses are not automatically active now.

Before model outcomes, freeze a 32-image fit panel. Prefer the maintained
Human13 plus refined5 positive-label sources where they are compatible with the
canonical task, then fill from the registered COCO source using deterministic
metadata-based selection, including ordinary and dense/same-class scenes.
Resolve duplicate image identity, exact labels, descriptions, source groups,
dimensions, target lengths and admission exclusions before freezing. Do not
invent/repair labels or reinterpret unmatched predictions as physical errors.
If source compatibility prevents this admission without changing label meaning,
report the conflict; do not silently create a new teacher. The exact admitted
population and annotation-completeness boundary belong in admission.json.

Freeze 64 disjoint COCO monitor images, balanced between ordinary and dense
annotation strata. They are held out from THIS training, not claimed unseen by
the mature checkpoint. Prior development images are not fresh confirmation.
Monitor outcomes must not choose LR/checkpoint, trigger routine early stopping,
or veto fitted-image feasibility. This round authorizes no 1k/4k/full-data fit.

Use the existing COCO-80 description-first canonical response and correct
`geo_sorted_xy` ordering: lexicographic x1, then y1, stable source ties. Preserve
norm1000 geometry and wrappers. Full-response full-vocabulary CE supervises
description, schema, coordinate and EOS atoms; mask prompt/image/padding as in
the maintained trainer. Keep `losses.normalizer: segment_balanced`, whose
denominator is the planned step's eligible segments; do not transplant the old
sidecar's coordinate-only token-pooling objective. No auxiliary alignment,
axis, repetition or geometry loss is added in this first recipe.

## Frozen address interface and trainable surfaces

One explicit injection occurs after the MAIN vision merger, in LLM hidden
dimensions, before its visual tokens enter the language sequence. Keep existing
absolute position, vision RoPE, language MRoPE and DeepStack routing. Do not add
separate explicit injections at DeepStack layers or alter rotary frequencies.

For bin b, W[b] is the live effective OUTPUT coordinate row: base row plus its
independent output delta. Never use the wrapper's base-only `.weight`, the
untied INPUT table, or a detached/cached initial codebook. Only the 1000 coordinate
IDs participate in address lookup, excluding the four selected wrapper IDs.

For every actual merged image cell, derive normalized x/y cell centers from
processed image geometry and the real processor/vision token order, independent
of any target annotation. Map u to t=clip(1000*u, 0, 999) and linearly interpolate
the adjacent effective output rows. Verify rectangular grids, merge order and
any crop/padding transform against the real caller before admitting it. Mixed
packed images retain their own geometry and boundaries.

Let n(z)=z/sqrt(mean(z^2)+1e-6), calculated in FP32. Let Mx select even hidden
channels and My select odd hidden channels. These fixed complementary masks
distinguish axes without a dense learned projection. For each merged token v:

    a = Mx * n(interpolate(W, tx)) + My * n(interpolate(W, ty))
    v_new = v + g * stopgrad(RMS(v)) * n(a)

Use a positive learnable gain g=softplus(raw_gain), initialized to 0.05. Keep
normalization and interpolation differentiable into W; only the visual RMS
reference is detached. Cast the additive residual to the actual activation
dtype at the addition. Axis masks and geometry convention are checkpointed
architecture metadata. Each axis initially uses half of the mature row's
channels; this is an explicit design tradeoff, not guaranteed lossless alignment
or an object-boundary/owner representation. x1/x2 and y1/y2 roles remain supplied
by autoregressive context. Native output logits retain their original dot-product
definition; no output-logit normalization or family-mass alteration is added.

Train BOTH independent selected input/output deltas (existing coordinate and
wrapper selection), the address gain, and DoRA in all intended linear modules
of language, vision and aligner, including primary/DeepStack mergers. lm_head
selected rows are trained by deltas, not another head DoRA. Keep base weights,
patch convolution, norms and positional parameters outside that allowlist.
Preserve existing adapter targets/weights; use the maintained expansion path
to add missing tower targets with a function-preserving initialization. Do not
reset the existing adapter or retie/reinitialize the two embedding payloads.

## Reusable implementation, not a second trainer

Reuse `src.train` -> `src.training.pipeline` -> `SupervisedTrainer`, existing
packing/supervision/Accelerate, optimizer coverage, checkpoint publication and
native evaluation. Worker may make the narrow maintained changes necessary to
wire untied embeddings and this opt-in address module end to end. Existing
tied/no-injection behavior stays covered by focused regression checks.

Current integration gaps are known: the ordinary pipeline uses the tied-only
embedding implementation; the reusable `src/qwen/untied_embeddings.py` supports
two deltas; optimizer grouping rejects unknown trainables; ordinary checkpoint
publication does not automatically save an arbitrary added module. Close these
at their existing owners, retaining fail-closed coverage and atomic save/reload.
Use dynamic HF reconstruction initially; untied dense/vLLM folding is not part
of this authorization. No install-site monkey patch, feature-cache training,
new trainer framework, dispatch registry, or unrelated cleanup.

Expected reusable deliverable: one config-driven entry supporting explicit
dataset/ID manifests, scale, seed, update/epoch limits, batch/accumulation,
parameter-group LR, evaluation cadence, resume and checkpoint roots. Later
small/large fits should change config/data, not copy the loop. Provide a prepared
full-dataset example with 1/2/4-epoch checkpoints, clearly NOT authorized to run.
Keep scientific selection/reduction in
`probes/training_set_completion/coordinate_codebook_alignment/`.

Worker owns that package, a narrow new config directory, necessary concrete
`src` integration points and their focused tests, its candidate notes in this
unit, and source captures. Record the exact implementation allowlist before
editing shared files; assign disjoint child surfaces. Existing dirty work,
including deleted CLAUDE.md/GEMINI.md, is unrelated and must be preserved.

## Qualification and bounded optimization

Begin with CPU invariants and a tiny real slice drawn from the frozen fit panel:
rectangular/ordinary/dense examples, genuine full-response updates, checkpoint,
independent process reload and empty-prefix native generation. Verify the OFF
injection expanded model matches the composed source under the same runtime,
effective input/output tensors are distinct, intended towers update, and the
address branch itself sends a gradient to output deltas. Use a focused
counterexample to show the caller check detects stale/detached codebook, wrong
grid association and omitted checkpoint state. Do not create architecture
ablation arms from these mechanical checks.

Compare the real loss/gradient against the maintained segment-balanced formula
using unequal response lengths. If using distributed training, qualify unequal
two-rank work and single/distributed normalization before broad jobs. Verify
actual trainable-gradient checkpointing and live vision gradients; no frozen
feature cache is valid here. Measure throughput/memory on the real full path,
including long native decoding, before freezing queue estimates.

Before the first scientific update, save the source-bound launch config and
admission. Default optimizer proposal released by the lead: AdamW, betas
(0.9,0.999), eps 1e-8, weight decay 0, grad clip 1.0, effective batch 8, ten
linear warmup updates then constant LR. Nominal group LRs: language DoRA 2e-5,
vision DoRA 5e-6, aligner DoRA 2e-5, input/output deltas 1e-5, raw_gain 1e-3.
Use existing source rank/alpha, dropout 0. Use seed 1729 and a recorded shuffled
repeat schedule. Worker may resolve batching/precision/attention/checkpointing
mechanics using source parity and measured throughput; do not silently change
loss, trainable surfaces or the scientific interface.

This is an overfit-capable schedule, not a 1-epoch transfer screen: up to 512
optimizer updates per fit. Run the nominal LR first. If competent fitting is
not reached, up to two fresh starts at 0.3x and 3x all group LRs are authorized,
using the same source/cohort/schedule. These are optimization settings of ONE
architecture, not mechanism ablations. Avoid rejecting a slow stable fit solely
for initial lack of progress. Record observed numerical failures and candidate
curves. If qualification shows a reproducible immediate scale disruption, a
single initial-gain reduction to 0.01 is authorized, documented before that
scientific fit; do not form a gain x LR grid. Other regularizers or structural
changes require a lead ruling, not user reapproval for routine repairs.

Evaluate fitted-image natural generation at source and saved updates 32,128,512
(or terminal), retaining all denominators and curves. Earlier clean completion
may be checked again after another 32 updates; stop that fit when stable. Select
by fitted-image native coverage/validity/repetition, then coordinate fidelity;
never by held-out or aggregate CE alone. One second-seed repeat (2718) of the
chosen feasible configuration is authorized if it fits the package cap; both
seed results remain visible. No selection of a lucky seed as the sole outcome.
Technical repairs are not extra scientific settings, but all their costs count.

Qualification-to-fitting continuation is automatic after these checks pass and
the frozen launch is saved. Send the first real-entry milestone promptly; do
not wait for lead approval unless there is a material conflict. Never replay a
passing execution under a new qualification/scientific label. A scientific
state starts from its declared source; qualification updates are not hidden
warm-start exposure.

## Evidence, compute and stop

Primary: ordinary empty-assistant-prefix greedy generation on all fit images,
same prompt/parser, RP1.0, no sampling/NMS/forced escape, max_new_tokens=3084.
Before admission verify canonical target lengths fit this generation cap;
otherwise report the population/cap conflict before scientific execution.
Report per-image one-to-one known-positive coverage, gains/losses, recurrence
proxies/run lengths, invalid/malformed/missing outputs, unknown matches and
EOS/cap. Physical identities require existing physical labels; ordinary IoU
matches remain annotation proxies. Report IoU50 and IoU80, target-token margins
and coordinate error to distinguish box fidelity from mere token changes.

The clean finite-panel target is complete admitted known-positive coverage at
IoU50 without repeat/invalid/parser/cap burden and with natural EOS. Report
partial improvement honestly; failure to reach this target is not proof of
representational impossibility. Teacher-forced loss without natural completion
is not a fit-success claim. With deterministic matched execution, a target that
wins at EVERY canonical prefix must replay greedily; contradictions require
execution/conditioning diagnosis. Sparse average-loss improvement does not
establish that premise.

Evaluate the source and the chosen final configuration on the 64 monitor images
after fitted-image selection. Monitor regression/flatness is reported separately
and does not veto fitted-image feasibility or select another checkpoint. The
lead decides whether any larger-data phase is worth funding; no full-dataset,
1k/4k expansion, broad search or successor is automatically released.

Use all eight GPUs where useful: independent LR work after nominal diagnosis,
qualified distributed fitting, and queued evaluation. Prefer a work-conserving
queue with fixed scientific cells over static image buckets that strand GPUs.
Do not wait for expected stress occupancy; react to concrete runtime conflicts.

Lead-set initial envelope for THIS package: at most 8 model-execution wall hours
from its first real model entry and 64 allocated GPU-hours, with all failures,
qualification, training and evaluation included. These are new lead containment
limits, not a quotation of a user budget. At half the envelope, report cost/
throughput and remaining decision work. Preserve/join jobs before the cap;
unfinished work is HOLD. No resetting clocks or hidden CPU-launched producers.
Retain exact source/config/commands and raw evidence; no gratuitous giant logit
dumps or per-step full-model snapshots.

Return a stable candidate manifest plus candidate-results.md, measured reusable
entry/resume/reload capability, exact commands, selected parameters/curves,
per-image traces and saved-output reducer, costs, and terminal job status.
Give bounded interpretations: technical validity, same-panel feasibility,
held-out monitoring and causal attribution separately. Stop at candidate or
budget/conflict; no self-acceptance, commit, publication or production launch.

Acceptance includes focused caller tests with meaningful falsification, a fresh
saved-output reducer replay, trained checkpoint fresh-process HF readback,
`python scripts/research/check_research_knowledge.py check`,
`python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs`,
and `git diff --check`. Report unrelated pre-existing failures separately.

Artifacts:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment/`.
Use maintained source imports and the existing source-provenance helper. Do not
execute captured/history/output code or create human-authored notes in outputs.
