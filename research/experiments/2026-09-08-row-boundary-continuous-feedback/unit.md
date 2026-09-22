---
title: Completed-row continuous feedback for remaining-owner enumeration
description: One matched Source-started pilot with full image and causal history retained.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-08-row-boundary-continuous-feedback
topic: qwen3-vl-dense-enumeration
status: running
evidence_status: partial
updated: 2026-09-08
---

# Frozen question and authority

From Source step-2444, does feeding a completed row's continuous final state
into the next row opener improve unique annotated-owner coverage under natural
greedy decoding with repetition penalty (RP) 1.10, relative to equally trained
feedback-off control?

The user approved this bounded route on 2026-09-08, requested lead-directed
subagents and long event-driven waits without polling, named Human13/Image2299
for the small slice, and authorized GPU use without further permission requests.
This unit owns scientific meaning; the lead owns acceptance. It authorizes
experiment-local code, not relaxation of shared forward contracts or a reusable
architecture promotion. No commit, cleanup, publication, or parameter sweep.

Strongest alternative: the model learns a generic continuation or short-prefix
training bias rather than a usable covered-set transition; lower loss, fewer
duplicates, or forced-prefix improvement need not improve full greedy coverage.
The cheapest decision-bearing evidence is the matched, image-disjoint natural
generation contrast below. This is a pilot, not a confirmatory generalization test.

## Intervention

At each legal completed `<|box_end|>`, retain the contextual final normalized
hidden state. Add one bias-free learned linear projection of that state to the
input embedding of the next `<|object_ref_start|>` belonging to the next row.
The projection starts at zero, with fixed external scale 1. There is no extra
learned gate. Retain the full image and all causal history/KV; no new vocabulary,
history compression/masking, vision change, STOP controller, or new RL objective.
An owner is the joint `{description, xyxy}` instance, not its description alone.

Timing is **post-admission, pre-description/instance selection**. The preceding
box-end logits already chose opener versus EOS. This intervention cannot directly
change that same boundary's admission logits; no premature-EOS solution is claimed.
Forced common-opener tests, if used, are diagnostics, never the primary outcome.

Training uses sequential row segments with graph-connected KV and prior end state;
do not detach that state or label a detached recurrence full backpropagation.
Gradient checkpointing is disabled for the initial cached path. Each trajectory
encodes its image once. Inference keeps native greedy/RP/EOS semantics and resets
feedback state per request. Unsupported boundaries fail closed.

## Anchor and implementation boundary

- Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- Source: `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
- Adapter `adapter/adapter_model.safetensors` SHA256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
- Adapter config SHA256:
  `0781d04d0579c57a45949193ca3b4afeedbadf92ff237873b070450e3b262739`.
- Reuse Source selected-token embedding delta; freeze and verify its live hash
  in the execution receipt. Base, vision, embedding delta, and output head stay fixed.
- Both arms train the same existing language DoRA parameters; only the feedback
  arm additionally learns/uses the new projection. Source, not the newer dedup
  checkpoint, is the common starting anchor. Dedup auxiliary weight is zero in
  both arms. Source choice isolates this first feedback contrast from that branch.
- Reuse current native tokenizer, compact formatting, geometry, materialization,
  generation, parser and matcher. Start FP32/SDPA as in Source256; record effective
  dtype/attention backend and
  immutable code/config/input hashes before any conclusion-bearing execution.
- Output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback`.
- Starting worktree HEAD: `e4f932faa80718fc487fc56744e9e16f1abf62a7`.
  Existing unrelated dirty records/configuration are preserved.

## Gate 1: real slice, mechanics only

GPU 0 initially; one real Image2299 trajectory with at least two completed rows,
one update, save, fresh-process reload, and greedy RP1.10 readback. Image2299 has
46 annotated owners, including 38 people; the person count is not the full denominator.
GPU runtime cap is 30 minutes, including diagnosis/reload; CPU implementation is
outside that cap. Stop on unresolved top execution risk, preserving failed evidence.

Required evidence: feedback-off native parity; legal injection positions/counts;
nonzero next-row gradient to the projection and preceding end state; image forward
count one; correct cache/position growth; finite update; explicit projection and
DoRA optimizer/save payload; cold restored behavior; no cross-request state leak.
Zero initialization naturally gives zero prior-state gradient at the first forward:
use a separately identified nonzero diagnostic projection for gradient sensitivity,
then restore zero initialization for the real update and prospective pilot.

Record exact commands/exits, forwards, tokens, cache lengths, parameter and payload
sizes, wall time, peak GPU allocation/reservation and host RSS. Numerical parity
tolerance must be declared, not enlarged after failure to call it a pass. This gate
proves mechanics, not model quality; no broad pilot before lead acceptance.

## Gate 2: one matched pilot

Target 128 eligible training images and 64 image-disjoint development images, one
seed (`20260908`), 16 matched optimizer updates, and no intermediate model selection.
Keep preexisting sealed confirmation images untouched. The CPU data receipt binds
the exact image IDs, candidate/eligible/excluded denominators, source provenance,
prompt/media hashes, token slices, target identities, and split exclusions before
pilot launch. If suitable sampled Source prefixes are unavailable, acquire only
the missing common bank; never substitute GT-only prefixes or another checkpoint.

Both arms consume identical frozen Source self-rollout prefixes ending at legal
row boundaries, at most four completed prefix rows for this first bounded graph.
For each image, choose up to two distinct remaining annotated owners using the
frozen seed; any corresponding complete legal next-owner row is a valid target.
Use existing category-consistent one-to-one IoU>=0.50 matching for prefix coverage.
Do not claim a unique canonical next owner or treat unmatched predictions as
established hallucinations. Prefix labels are ignored. Supervise the complete
next-owner span (opener, description, coordinates, box end), with no fabricated
EOS after a single row. Average token NLL within target, then alternatives within
image, then images. This is a simple common continuation objective, not a new set
likelihood or an explicit covered-set label.

Each update traverses all 128 images; same order and persistent AdamW in both arms:
learning rate 2.5e-6, betas (0.9, 0.999), epsilon 1e-8, weight decay 0, global norm
clip 1, constant learning rate, eval-mode/dropout off with gradients enabled.
Run the same segmented scoring path in both arms so feedback is the only execution
factor. The image count/update count is matched, not parameter count/FLOPs/wall time.
No development outputs select data, loss, layer, gate, learning rate, or stopping step.

After measured slice cost, the lead will freeze an executable launch bound here
without asking again. Conservative ceiling: two independent single-GPU training
arms, at most four hours per arm, at most four additional GPU-hours for required
data acquisition and final inference; total ceiling 12.5 GPU-hours including slice.
Launch only if projected work fits; do not silently reduce images/updates or add
distributed machinery to force it through. One matched round is the scientific stop.

## Evidence and decision

Final cold full-image generation: Source reference and both terminal arms on all
64 development images, natural greedy RP1.10 primary; RP1.0 diagnostic. No forced
prefix, external dedup/postprocessing, target hints, or EOS suppression in primary
generation. Use the Source256 cap of 3084 new tokens and the same geometry in all
arms; report actual length caps separately from observed EOS.

Primary: unique category-consistent annotated-owner true positives at IoU>=0.50,
feedback minus equally trained control; preserve gained/lost owner IDs and per-image
deltas. Report paired-image bootstrap uncertainty as exploratory, not confirmation.
Also report annotated FP/FN, precision/recall, output count, parse/stop/cap failures,
and later geometric repeats (prediction-prediction pixel IoU>0.95, irrespective of
class) as diagnostics. Fewer duplicates alone is not enumeration improvement.

Complete this version after the single fixed round. No natural coverage gain, only
training/conditional gains, or merely shorter output closes this version without
an automatic layer/gate/loss/seed sweep. A positive pilot only supports deciding
whether independent replication is worth doing; it does not promote the architecture
or establish a native symbolic covered set. Technical failure leaves the affected
scientific contrast unanswered, not negative.

## Current evidence

CPU data is lead-accepted after replaying the complete provenance/geometry/split
check, including observed overlap and bad-Source-provenance sensitivity failures.
[Data receipt](data-receipt.md) owns the details; immutable manifest:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/manifest.json`,
SHA256 `63b89eed7cf05947ccb68facbf4ef245b13517c76ec6b97a44210fd63f21249b`.
The checked producer SHA256 is
`f94c477e75e984b1cd8551de756f3beae8168fa0d12b3999729dc7d2a1f86304`.
The actual sampled Source bank was reusable: no new rollout acquisition is needed.
256 candidates yielded 236 eligible training images; 128 were frozen, with 233
legal remaining-owner target alternatives. Dev64 contains only raw image/GT rows.
It is disjoint from this pilot's train, Human13 and reserved confirmation sets,
but had prior development-evaluation exposure; do not call it a fresh confirmation
or globally unseen panel. No model outputs were read to select those 64 IDs.

The Image2299 mechanics slice is lead-accepted at its explicitly reduced 128-token
generation scope. The lead replayed the four CPU checks, inspected the gradient,
cache/position, payload and separate-process evidence, and closed a reproduced
grammar defect: naked/malformed end markers no longer activate feedback; native
batch>1 fails closed. The same saved update was cold-read after that correction,
without retraining, and valid application sites and warm/cold tokens agree.
Receipt: `slice-v1/attempt-4/receipt-grammar-correction.json` under the output root,
SHA256 `2565432bebac8a82b5565e591dbf1099e568e373e1663b94e0f2ea1a09b9c232`.
Measured aggregate GPU-visible wall time was 145.42 seconds; peak allocated/reserved
memory 25.16/26.68 GB. This remains mechanics evidence only.

The pilot runner must first cross its own real target-only loss/save/cold-consumer
entry using four frozen training representatives (including maximum prefix/prompt
shapes) and full-cap 3084-token generation. This small integration check is charged
to the existing budget, starts from Source, and never seeds the scientific arms.
Then freeze measured full-course bounds and run the original two arms from Source.
No extra architecture or auxiliary research gate is added. Broader dirty routers
are not rewritten during construction.

## Lead launch acceptance and frozen runtime

The four-image pilot-entry qualification is lead-accepted: the lead replayed five
focused tests, inspected target-only loss and 128-image/alternative reduction code,
verified the frozen producer identity, and consumed all four cold raw records with
the final reducer, reproducing TP/FP/FN and matched owner IDs. Actual visual/model
hooks measured 8 image and 34 segment forwards, 8 backwards, 78 supervised target
positions and zero prefix labels. All four full-cap native generations reached
`im_end`; this is execution evidence, not training/development quality acceptance.

Qualification receipt `pilot-qualification-v1/receipt.json` SHA256:
`361d139883e51b92024655ee5ed56af1ae305e9bea2b3c8bbbbf9a3ebc9a9361`.
Fixed runner SHA256:
`5e9238e58566ed8a62dab1d6034494d5fd14dff307aa9eb7b4b3b2cb29923d7b`.
The launch candidate and measured extrapolation live at
`pilot-launch-candidate.json` (SHA256
`cac3dd13049ec3c9d97759cefa114abc842e1438bca424a8919fa9b8f75d9e2b`).
Measured update time was 14.75 s for 8 alternatives; peak allocated/reserved memory
28.85/31.01 GB. Linear projection is 1.91 GPU-hours per training arm and 0.91 GPU-hours
for all six final evaluations, about 4.80 GPU-hours including qualifications.
This is an estimate, not permission to exceed the frozen ceiling.

[Fixed launcher (preserved from `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-row-boundary-continuous-feedback/launch_pilot.sh) owns two independent single-GPU chains: control
GPU 0 and feedback GPU 1, followed by their respective two cold evaluations. Source
reference runs on GPU 2. Each arm starts from Source, never a qualification payload.
Per-training command: 14340 s then at most 60 s interrupt/kill grace; each of the
six evaluation commands: 2340 s plus at most 60 s grace. These sum to at most 12
GPU-hours, leaving the qualification reserve within the original 12.5 ceiling.
No automatic retries or partial-image substitutes. `pilot-v1/launch/` records exact
runtime hashes, launch identity, per-command logs/exits and driver terminal exit;
`pilot-v1/results.json` is the final six-panel reduction. The lead must still inspect
terminal counters, raw artifact identities, and the primary result before closure.
