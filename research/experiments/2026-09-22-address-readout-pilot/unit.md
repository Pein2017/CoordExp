# Address-assisted readout and genuine visual-detail pilot

Date: 2026-09-22. Status: user-authorized bounded execution, lead acceptance pending.
This unit owns both parallel lanes and their common evaluation contract.

Transport amendment: [direct session reporting](communication-update.md)
supersedes this unit's wake-marker return instructions only. Scientific gates
and budgets retain their owning contracts.

## Authorization, pair and checkpoint decision

The user approved the proposed small frozen-backbone training pilot and parallel
visual-detail comparison, then explicitly said: “好的,理解了,可以执行.
另外, 用untie的 checkpoint 会不会更好? 由你决定. 可以开工.”
This supersedes the earlier preparation-only instruction for this package.
All eight GPUs may be used; the approved ceiling is four hours of model-execution
wall time and 32 allocated GPU-hours, including qualifications and failures.

- Lead: `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d` (current Astra/ultra).
- New persistent execution worker: `01a0c726-ad7c-7cc0-89b7-d76ac6fcf027`,
  named `922-worker`, user-selected `gpt-6-astra`, `low`; settings verified live.
- Cwd: `/data/CoordExp/.worktrees/research-probes`.
- Lead owns scientific decisions, this protocol/state/catalog/frontier, training
  qualification acceptance and final acceptance. Worker owns implementation,
  preparation, job supervision, candidate evidence and independent child checks.
- Worker may implement directly and/or use Luna-family children for disjoint
  implementation surfaces, with explicit live-supported model/effort. No new
  sidebar main, no child authority to expand the research program. The historical
  Luna-only recovery trial is finished and does not constrain this new package.

Use the mature **tied step-2444** throughout this first pilot. The available
untied checkpoint also changes axis loss and other training conditions; it is
not a clean untie comparator. The sidecar is independently trainable while the
original tied embeddings/head remain frozen. No tied/untied sweep is authorized.

Exact source loader: `probes/training_set_completion/untied_shared.py`, tied
configuration from the retained
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json`.
Base model:
`/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
Source checkpoint:
`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
Bind actual resolved paths, model/processor/config and trained delta hashes at
preparation; a path label alone is not source parity.

## Question, alternatives and decision

Hypothesis A: implicit visual-position-to-coordinate readout is a limiting
computation. Supplying correctly associated visual addresses to a small learned
readout can improve coordinate grounding and, if this defect materially drives
bursts, improve distinct-owner coverage during native enumeration.

Strong alternative: coordinate calibration can improve while the native policy
still fails to commit to a distinct instance or update history-dependent
selection. That outcome downgrades the ruler as the main burst remedy/cause,
even if the auxiliary readout learns successfully.

Hypothesis B: insufficient retained visual detail is a material bottleneck.
Compare baseline image processing with retaining more genuinely available
original detail. A positive result is input-detail dependence, not unique proof
of a vision-encoder defect: token count, spatial positions and attention burden
also change. More interpolation pixels alone do not constitute new detail.

Prior constraints, not launch instructions:
- [Visual binding](../2026-09-21-visual-instance-binding/lead-results.md): finite
  conditional full rows retain differentiated local support; native first-owner
  choice and physical identity remain unresolved.
- [Spatial progress](../2026-09-21-spatial-progress-recovery/lead-results.md): the
  noncrossed control improves comparably; no selective progress-gate cause.
- [Native row choice](../2026-09-21-native-row-choice/lead-results.md): zero of
  three tested N witnesses beats the repeated full row, a finite search only.

Proceed only with the fixed pilot. Calibration improvement plus preserved/gained
physical owners in natural decoding supports continuing address-assisted readout.
Calibration improvement without reduced same-owner recurrence downgrades it as
the main burst explanation. Failed learning or invalid qualification is
unidentified, not a scientific null. No unique circuit/training-origin claim.

## Shared admission and evaluation

The worker prepares one immutable admission/config manifest before model work.
Use trusted existing positive boxes and canonical `geo_sorted_xy` serialization;
retain image, source split, description, annotation/physical-owner provenance,
decode config, original dimensions, processed grids, exclusions and reasons.
`geo_sorted_xy` means lexicographic left-top `(x1,y1)`, not raster-row scanning.
Do not change labels, prompt, sorting, coordinate vocabulary or EOS supervision.

Target 128 training images and 32 independent calibration-validation images,
split by original image identity. Select deterministically from eligible existing
data, not by sidecar results. Retain valid positive targets from incomplete
annotations without treating missing annotations as negatives or complete-set
obligations. No target coordinates in a supplied referent description.

Evaluation is disjoint by image from both training and calibration validation:
the existing six-image/11-source-trajectory recurrence pool is development
evidence only (only tied trajectories are executable in this pilot), plus up to
32 fresh natural-evaluation images selected deterministically before model
outcomes. Freeze density/same-class strata from source annotations if available;
do not select a new panel by baseline or intervention failures. “Fresh” means
not used in these pilot splits/development, not guaranteed absent from backbone
pretraining/SFT. Publish unavailable cases as HOLD, not silent replacements.
If trustworthy sources cannot fill the proposal, return exact availability to
the lead before changing the scientific denominator.

Use native original-image, empty-prefix greedy enumeration as the primary
behavioral endpoint, with matched prompt/budget/parser and model mode across
conditions. Freeze a bounded recurrence-prefix continuation diagnostic separately;
its effect is not empty-prefix transfer. No sampling, forced escape, new NMS or
repetition penalty. Retain full token/box traces, termination reason and cap.

Report per-image credible owner gains and losses separately, distinct-owner
coverage, same-owner revisits and burst length, invalid geometry, EOS/cap and
unmatched/unknown. Apply existing physical review semantics and preserved owner
identities; do not equate exact-coordinate repetition with same-owner repetition
or every annotation-unmatched prediction with a false object. Show annotation
proxies as proxies if physical review is incomplete. No automated judge upgrade.

Calibration includes coordinate CE/error and a diagnostic with naturally unique
class/clear referent supplied without coordinates, freely generating all four
coordinates. Teacher-forced y2 after three GT coordinates is not this diagnostic.
Freeze these diagnostic cases before training and keep them separate from the
native enumeration claim.

## Lane A: minimal coarse address-assisted readout

Freeze the complete original model: vision encoder, primary merger, DeepStack,
LLM, DoRA, trained embeddings/head and tied relationship. Train only two
2048-to-64 projections Q/K, four 64-dimensional role embeddings and one scalar
gain (about 263k parameters, exact count from code). No new full LoRA, upstream
residual injection, ledger, learned local offsets or tokenization changes.

For a causal coordinate slot t, let h_t be the actual effective head input after
final norm, and v_i the frozen primary post-merger visual tokens. Role r is
derived solely from the consumed prefix. Define
`a_i = softmax_i((Q h_t + e_r)^T K v_i / sqrt(64))`.
For a merged grid Hm by Wm, token addresses are its actual cell centers:
`cx=(col+0.5)/Wm`, `cy=(row+0.5)/Hm`, mapped through the real resize/padding to the
annotation coordinate frame. Validate visual-token ordering and image ownership.
Never infer this bank from text-sequence RoPE positions or a DeepStack tuple.

For bins b=0..999 and the role's axis, use a fixed Gaussian address kernel
`K_ib = softmax_b(-0.5*((b/1000-c_i)/sigma_axis)^2)` with sigma equal to half one
merged-cell width on that axis in the same coordinate frame. Compute stably in
log space, and `R_b=sum_i a_i*K_ib`. The data geometry uses b/1000; the older
embedding initializer's b/999 convention is not the output geometry contract.
Native logits retain the within-cell information; this tests coarse address
assistance and cannot refute every finer ruler/interface on a null result.

Add gain times log R to native coordinate logits and renormalize within the
coordinate family C to preserve its native logsumexp:
`z'_b=z_b+delta_b-logsumexp_C(z+delta)+logsumexp_C(z)`.
Leave all non-coordinate vocabulary logits untouched at the same prefix.
Only the 1000 coordinate IDs belong to C, not all 1004 selected training IDs.
Use prefix-admitted coordinate roles only. This preserves family/EOS competition
at a fixed prefix, but changed histories can change later stop decisions.

Initialize Q/K nonzero and gain zero for source parity. Demonstrate gain gradients
first and Q/K/role gradients after the gain moves; no dead double-zero branch.
Keep the backbone under no_grad and train the sidecar outside it. Detached cached
states may be reused across arms only for the exact same teacher-forced history;
each freely generated arm computes its own states on its own generated prefix.

Three arms, with two paired training seeds:
1. Frozen original source.
2. Trained correctly associated visual-feature/address pairs.
3. Identical architecture, initial weights, optimizer, batches, updates and budget,
   with a deterministic frozen nonidentity permutation of address pairs relative
   to features (bind mapping per grid shape before outcome inspection).
Do not permute pixels, feature order, native positional embeddings or bin meanings.
The wrong-address control is not information-free; position may be recoverable
from keys. A positive requires benefit over the original, not merely degradation
of the permuted arm. No outcome-selected permutation or checkpoint.

Use positive coordinate-token CE with full-vocabulary probability accounting and
explicit causal alignment/global token denominator. No coverage/repetition loss,
new stop/count target, artificial corrupted-prefix curriculum or EOS loss.
At most 256 optimizer updates per trained arm/seed. Freeze exact optimizer,
batching, numeric mode, paired seeds, update schedule and final checkpoint rule in
the preparation manifest before training. Use the final fixed update, not the
best failure-panel score. Preparation may propose a smaller equal-arm schedule
from measured throughput; lead accepts it before substantive training.

## Lane B: genuine visual-detail comparison

Keep the original frozen source; do not combine this lane with the trained bridge
in this pilot. For eligible originals, compare the exact baseline processing
against one fixed higher-detail processing setting, chosen from original
dimensions and budget before results. Preserve field of view/aspect/content,
prompt, decode limits and annotation mapping. No outcome-selected crop/zoom.

Add a control at the same larger processed grid that first discards detail down
to baseline resolution and then upsamples. This separates added visual samples
from merely increasing token count/grid where possible; it does not perfectly
isolate every resampling effect. Retain exact compositor/interpolation settings,
pixel hashes, effective grids and attention-length costs. If originals contain no
additional detail, mark that image ineligible for this question rather than call
upsampling detail recovery. Share source baselines when truly identical.

## Real-entry qualification and staged authority

Implementation, CPU preparation/checks and a bounded actual-model qualification
are authorized now. This first dispatch ends at lead-review-ready or a material
blocker. Broad Lane A training and scientific Lane B production require the lead
to inspect this qualification and freeze the prepared admission/config first.
This is a lead verification gate under existing user authorization, not another
user permission request. Independent CPU preparation may continue in parallel.

Qualify the exact maintained caller, persistence and cached generation path:
1. Bridge-off/zero-gain complete-vocabulary and short-greedy parity to the mature
   source. When enabled, only admitted slots and the actual coordinate family
   change; verify family logsumexp conservation and non-coordinate equality.
2. Rectangular-grid corner/address identity, spatial merge order, x/y units and
   batch/image boundaries. Deliberately corrupt an axis/order and make the check
   fail. Keep original model vision/DeepStack consumption unchanged.
3. Future-target/suffix mutation cannot change earlier features/roles/predictions.
   Target index j uses causal state j-1 including compact selected-row mappings;
   prefix-only parsing agrees with full-prefix parsing at every admitted slot.
4. Real gradient/update evidence and optimizer allowlist; frozen original tensors
   remain unchanged. A tiny fit must show learning signal beyond a scalar-only
   dead sidecar. Preserve finite gradients and post-update parameter deltas.
5. Tiny train -> sidecar save with source/config/token bindings -> fresh-process
   reload -> free complete-box generation. Full replay agrees with incremental
   cached generation; per-request visual banks do not leak between images.
6. Lane B baseline replay and composition/geometry checks, one actual eligible
   original through all three settings; count qualification outputs as scientific
   cells if the frozen final inputs/conditions are identical. Never rerun merely
   to rename qualification evidence as production.

Use the smallest real dataset slice exercising these risks. Bind all source
captures before the corresponding execution, including current shared imports.
Retain failures and costs. No arbitrary one-repair ceiling: worker supervises
ordinary repairs within budget, escalates conclusion-changing conflicts promptly,
and does not expand data/architecture/conditions to turn a failed check positive.

## Ownership, budget, artifacts and return

Worker-owned maintained code:
`probes/training_set_completion/coordinate_address_readout/` and
`probes/training_set_completion/visual_detail_dependence/`.
Keep shared pilot admission in the first package with a single owner; agree a
small interface before independent lane work. Worker-owned research records:
this unit's `preparation.md`, `candidate-results.md`, and bounded execution notes.
Lead owns `unit.md`, `state.json`, catalog, frontier and acceptance records.
No edits to historical results, labels, shared infrastructure or unrelated files
without reporting a necessary scope conflict. Reuse maintained shared operations.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot/`.
Use distinct selection, qualification, lane-a and lane-b directories underneath;
failed/repair attempts get fresh paths. Source captures follow
`docs/OUTPUT_STORAGE_POLICY.md` outside outputs. No code/Markdown in outputs.

Use GPUs 0..7 productively for independent preparation, paired training seeds and
evaluation as dependencies permit. Prefer single-GPU independent jobs to adding
DDP solely to occupy cards. Expected stress occupancy is not a reason to wait;
react to an actual launch/OOM conflict. Record each owned process and join it;
never leave an unaccounted producer or schedule duplicate work.

The four-hour wall clock starts at this package's first model call; the total
32 allocated GPU-hours includes failed jobs, cached feature passes, training,
evaluation and duplicated evidence. Record model/vision forwards, allocated
GPU-seconds, wall start/terminal, updates/tokens and artifact bytes. Estimate
remaining stages from qualification before scaling; reduce no frozen denominator
silently. Stop before exceeding either ceiling. This is a pilot ceiling, not a
spend target. Retain minimal necessary states rather than full activations/KVs.

At the qualification gate, return a stable candidate manifest, source/admission/
config bindings, exact run/check commands, failed mutation evidence, parameter
and gradient checks, parity maxima, save/reload evidence, costs/forecast, changed
paths and all job states. Include Lane B preparation/qualification in the same
handoff. Lead acceptance releases the fixed remaining pilot in a separate message.
At final closeout, additionally provide the deterministic saved-output reduction,
per-image outputs/uncertainty, physical identity evidence and all costs/failures.

Acceptance uses actual caller selfchecks/replay and manifest bindings, fresh
reload evidence, `python -B scripts/research/check_research_knowledge.py check`,
`python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs`,
and `git diff --check`. Supply precise new module commands when implemented.
Do not invent pass results for planned checks. Candidate is not lead acceptance.
No commit, publication, successor, new architecture, extra seeds or data sweep.

## Worker dispatch and durable return

Read `/data/CoordExp/.codex/skills/lead-worker/SKILL.md`, relevant local AGENTS,
`qwen3-vl-execution`, and `full-pipeline-smoke` for the named execution risks.
This unit is the English task profile for both lanes; it is not permission to
interpret historical protocols as new execution authority.

At the first decision-bearing handoff, finish/stop dependent jobs, save the
candidate or blocker, and append exactly one JSON line to output-root
`lead-events.jsonl`: `{"event":"LEAD_REVIEW_READY","result_ref":"absolute manifest path"}`
or `{"event":"LEAD_BLOCKED","result_ref":"absolute evidence path","reason":"short reason"}`.
The file is append-only: never truncate, rotate or overwrite it. These markers
are routing evidence only, never scientific success. Then end your turn with the
same candidate/blocker and await lead steering; do not launch the held stages.
The lead arms one durable log monitor and independently reads back the artifacts.
