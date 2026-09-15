---
title: Image versus output-history dependence of small-box repetition
description: Bounded conditional continuation diagnosis before stronger punishment.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-11-small-owner-repeat-origin
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-11
---

## Question and decision

From unchanged Stable50, does a fixed-prefix continuation follow the image,
the coordinates in its latest complete output row, or both, near the onset
and after accumulation of repetitive output? The decision is whether stronger
punishment is justified, and what action would need credit. This is a mechanism
diagnostic, not a training or improved-policy experiment.

The user authorized independent bounded experiments and specifically asked
why a small owner can seed a drifting repeat chain, before deciding stronger
punishment. The [dedup32 study](../2026-09-11-stable50-geometric-dedup/results.md)
is closed and not promoted. Its valid-repeat decline largely hid invalid-row
growth; increasing that coefficient is not an arm here.

## Competing explanations and scope

- Image-dependent repeated retrieval: the current picture keeps inducing the
  same region/category despite history.
- History-dependent continuation: recently emitted geometry/category helps
  sustain a repetitive sequence, potentially beyond supporting image evidence.
- Mixed dependence, including numerical/coordinate-boundary attraction. A
  local coordinate effect does not identify an attention head, KV ledger,
  training-data cause, or physical-instance representation.

The original source picture and raw onset rows receive visual review: a first
row may be a correct owner, a poor extent, or unsupported. GT-unmatched is not
used as a hallucination label. No first detection is trained against here.

## Frozen exploratory contrast

Use the same eight deliberately selected cases as dedup32: 9813,158044,248167,
274509,351017,417044,477415,502725. These are seven known repeat cases plus one
clean control, not a population sample. Inspect all original raw rows, including
invalid geometry. The CPU census owns token boundaries, not owner truth.

At an early and a later complete-row boundary, cross two factors:

1. Original picture versus one different existing picture with exactly the
   same dimensions and visual-token grid. Donors are visually checked for lack
   of the target repeat category; this is an intentionally off-policy image
  substitution, not an image-removal or attention-ablation experiment.
   Donor-alone behavior is a necessary interpretive control: reuse the three
   existing hash-bound Stable50 donor trajectories (13043/46432: one airplane;
   81205: one fire hydrant), and acquire native donor-alone trajectories for
   new same-size donors3125/3514. This addition was fixed before any model call.
2. Original literal prefix versus replacement of only the four coordinate
   tokens of its final complete row by a translation. Preserve its width,
   height, validity status, class, wrappers, token count, and every earlier
   token. Do not repair invalid seeds or shrink full-canvas boxes to create
   a translation. Concrete eligible rows and translations are fixed in the
   packet before any model call; unavailable contrasts stay unavailable.

The factorial pairs have identical sequence lengths, prompt IDs, visual grid,
and free-token budgets. Every cell recomputes its own native prefix/cache;
there is no direct KV or attention modification. Early/late differences are
descriptive: row count and accumulated content differ, so they do not alone
identify a repetition-dose effect. Forced coordinate rows and image-swapped
histories are interventions, never claimed as natural detections.

## Model, evidence and accounting

The checkpoint is `positive7-support50-81/training/adapter` with unchanged
Source embeddings, Qwen3-VL2B base, FP32/SDPA and patch linearization. The
packet hashes its checkpoint, image, input and producer bytes. Decode remains
greedy, temperature0, repetition penalty1, top-p1, top-k0, model defaults off,
native EOS151645. No sampling, postprocessing, GT edits, or optimizer steps.

Each case first regenerates the original natural trajectory, cap3084, to
verify its saved anchor identity. Each conditional cell receives at most512
new tokens and never exceeds3084 total action tokens including its prefix.
The 512-token horizon is censoring, not evidence of indefinite repetition or
resolution of the original3084 cap.

Retain literal prefix/free IDs, raw text, native parser drops, valid later-row
IoU>0.95 repeats, exact raw repeated rows (including invalid rows), descriptions,
coordinate/center traces, and native EOS versus horizon. Compare the first
successor and first five/20 complete free rows, with missing successors
explicit. Plot original/translated seed centers and generated centers to
distinguish persistent redirection from one-row copying or immediate return.
These geometric diagnostics do not assign physical owner identity.

Image-swapped outputs have no original-image owner-quality interpretation.
Only unchanged-image outputs may carry supplementary annotation-relative
gained/retained/lost accounting, excluding artificial prefix replacements from
claims of recovered owners. A smaller parsed-repeat count alone is never success.

## Cost, admission and stop

Root owns design, packet, launch, visual adjudication and conclusion; Sol-high
owns the eight-case CPU onset census; Luna-max owns the bounded native runner
and CPU invariant tests. No children own promotion or extra model calls.

One immutable real smoke (351017 natural plus its four early factorial cells)
before scaling, then eight independent GPU workers, one case/model per GPU.
At most79 continuations including smoke,68,740 generated tokens, nine model
loads, and 1500 seconds per worker. Model
and image forward counts, wall time, peak CUDA/RSS and process exits are
recorded; shared jobs are untouched. No repeats of a scientific arm, donor
sweep, dose sweep, extra training or automatic cohort expansion.

CPU packet serialization, source identity and prefix/budget invariants must
pass before loading. The smoke must traverse model load, prompt/media/grid,
natural continuation, both intervention factors, durable raw output, native
parser and cold readback.
Mismatch or a runtime failure leaves only the affected contrast unanswered.

Stop after this panel and root interpretation, even if mixed. Stronger
punishment requires a separately specified valid-versus-invalid, row-level
contrast and preservation of plausible first owners; this diagnostic does
not authorize that launch. Architecture and training-origin causes remain
unproven without additional evidence.

## Artifact owner

[Output root](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin)
contains the CPU census, immutable `packet.json`, one smoke, per-rank raw
records/receipts, and one reduction. This unit owns interpretation and closure.

Frozen packet SHA256: `48087dcf05e03fc9b6c2eb9de1e2d66e1efe09c972ce22e8c6ae2774f95aa4ba`.
The five-cell real smoke passed exact Stable50 natural replay, both factors,
same prompt/grid, distinct image content, native parsing and cold readback.
It used4130 generated tokens, one model load and319.66s wall. This admits
the unchanged eight-rank panel, not a mechanism or policy-quality conclusion.

### Technical input-envelope repair

The initial eight-rank launch completed six cases. Ranks3/7 (274509/502725)
failed before any model forward or continuation: the lead's two outside384
donor records used absolute image references and pixel GT coordinates rather
than the caller's relative-path/coord-token input schema. This was a packet
construction error, not a model outcome or a GPU-memory failure.

The original packet, code and failed receipts remain unchanged. A new
`packet-repair-01.json` replaces only those two donor input-record envelopes
with existing canonical coord-JSONL rows and correctly rebased relative paths.
Actual image bytes, dimensions, prompt IDs, visual grids, model, prefixes,
jobs and decode are unchanged. `repair-preflight.json` reproduces both old
errors, then passes the actual native request/materialization caller on CPU.
Only the two unexecuted cases are reacquired under `repair-01`; the six
completed cases are reused. The generation budget remains79 total cells;
two failed model loads add technical cost (11 loads including smoke/retry,
rather than the planned9). This does not reopen any scientific arm.

## Closure

The [result](results.md) closes all74 planned full-panel cells from six preserved
and two input-envelope-repaired ranks, plus the five-cell smoke. Eight natural
anchors and16 unchanged-image/native-prefix suffixes exactly reproduce saved
Stable50 tokens. The lead personally inspected all8 onset and8 continuation
figures. Conditional image/history dependence is established on this selected
panel; a circuit, training-origin cause, small-object-specific mechanism, and
safe learned repair are not.

Total execution used79 continuations,29,536 generated tokens and11 model loads,
including the two pre-forward failures. Summed GPU-assigned worker wall time
was2393.66 seconds (0.665 GPU-hours), not utilization percentage. No training,
checkpoint promotion or further model call is pending. The recommendation is
not to increase the closed coordinate-only UL coefficient; any future row-level,
validity-aware learning contrast needs its own bounded design.
