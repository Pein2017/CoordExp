# Small-owner drifting repetition: mixed image and history dependence

## Decision

**Some longer repetitive histories can sustain the old category without the
original picture. The first box is not uniformly a hallucination, and moving
the last box is not a reliable fix. Do not simply increase the previous
coordinate-only unlikelihood coefficient.**

The frozen eight-case diagnostic is complete:74 full-panel cells plus5 smoke
cells, unchanged Stable50, no training. This establishes conditional behavior,
not the training-origin cause, a KV/attention circuit, a small-object-specific
mechanism, or a successful deduplication policy. Stop at this panel.

## Contrast and evidence boundary

Seven known repeat cases and one clean control were deliberately selected from
the preceding dedup study. At early/late literal-prefix boundaries, cross:

- Original image versus a visually checked, different image with identical
  dimensions, prompt IDs and visual-token grid.
- Native history versus translation of only the last complete row's coordinate
  tokens by250 bins along the available-room axis/sign. Preserve box extents,
  validity, description, wrappers, prefix length and every earlier token.

Every cell recomputes its native prefix/cache. This is not KV editing. Early
means immediately before the first strict repeat; late includes the eighth
same-description emission starting at that repeat. The clean control uses its
separately frozen boundaries. Early/late changes history length, count and
content together, so it is not a controlled repetition-dose experiment. Nor
does this compare small versus large owners, or matched x versus y shifts.

Original natural rollouts have cap3084; conditional cells have at most512 free
tokens and total action length at most3084. A length stop is censored, not proof
of indefinite looping. All results below count **newly generated complete
rows**, including invalid geometry where stated; forced prefix rows are never
credited as recovered owners. Strict repeat counts use class-blind IoU>0.95
against any earlier valid row, once per later row, including earlier prefix
references. Same-category counts are not physical-owner identity labels.

The five donor-alone controls all terminate without the target repeat class:
13043/46432 each one airplane,81205 one hydrant,3125 two people plus a truck,
3514 three people plus a frisbee. The first three reuse hash-bound Stable50
trajectories; the latter two were acquired in the frozen panel.

## What the intervention actually shows

### 1. Bottle and knife histories can outlast the original visual support

| Case; unchanged history, substituted image | Early prefix | Later prefix |
|---|---|---|
|351017 bottle → airplane image|One airplane, EOS after11 free tokens; no new bottle|51 bottle rows,18 valid +33 geometry-invalid;512-token horizon|
|502725 knife → frisbee image|Three people and a frisbee, EOS after39 free tokens; no new knife|56 knife rows,11 valid +45 geometry-invalid;512-token horizon|

The first bottle prefix contains a person and one bottle row; the later prefix
contains a person and nine bottle rows. Knife prefixes contain five versus13
knife rows. These are accumulated natural histories, not duplicated rows
artificially appended by the probe.

**Inference:** the original image is no longer necessary to sustain these two
conditional chains at the later boundary. The old-category behavior depends
strongly on accumulated history under the same donor image. This supersedes
the smoke-only impression that replacing the picture necessarily stops the
bottle chain. It does not show independence from all visual input, isolate
repetition count from other history changes, or identify an internal circuit.

### 2. Last-row coordinates affect subsequent traversal, not just one next box

On the unchanged original image:

- **417044, donuts:** native early/late continuations both hit512 tokens with
  51 valid rows and44/49 strict repeats. Translation produces20/22 valid rows,
  zero strict repeats, zero invalid rows, then EOS at201/221 tokens. The lead
  visually checked that successor boxes move to different actual donuts in the
  display, not merely jitter around one narrow strip.
- **502725, knives:** native early/late continuations both reach512 tokens.
  Translating the latest box permits one visually credible cake box and EOS
  after10 tokens at either boundary.
- **248167, vase:** translation permits finite six/eight-row continuations,
  zero strict repeats and invalid rows, with boxes on bed/chair/book regions.
- **351017, bottles:** translation releases table/glass/rear-bottle regions,
  but both original-image continuations still reach512 tokens and retain30/29
  strict repeats. Changing where generation goes is not necessarily stopping
  repetition.

**Inference:** emitted geometry can influence a multi-row continuation. A
spatial-order/search-position prior is a concrete alternative to a missing
owner ledger: changing the latest coordinate may change where enumeration
continues. These data do not distinguish those explanations. No annotated
or physical-owner preservation score is claimed for these forced histories.

### 3. Strong counterexamples prevent a universal self-loop or repair claim

- **477415, chair:** the early shift releases people/chairs, but the later
  seed is already zero-height at `[0,999,999,999]`. Shifting it upward while
  preserving invalidity still returns to that bottom-edge line:56 invalid
  rows and the512-token horizon on the original image. Replacing the picture
  with the airplane instead yields one airplane and EOS at the later boundary.
  The current image/context can matter even after a long bad history; this is
  not a universal last-row copying mechanism.
- **274509, books:** latest-coordinate translation largely returns to the
  original book region; swapping in the truck scene makes the output finite
  and short. The early donor continuation still includes one book description.
- **9813, clean control:** original-image continuations are finite and contain
  no strict repeats or invalid geometry. On the hydrant donor, the native late
  prefix stops immediately; translating that prefix instead produces41 person
  rows, including3 invalid rows and17 strict repeats, before EOS.
- **502725, early donor counterexample:** native history on the frisbee image
  is clean, but translating its latest box induces56 knife rows,54 invalid,
  reaching512 tokens. The same kind of edit can create rather than remove a
  pathology.

The clean-control failure is especially important: intentionally mismatched
image/history interventions can induce behavior absent from native rollout.
They diagnose conditional sensitivity; they are not deployable repairs or
automatically valid learning trajectories.

## Did the first small box correspond to something real?

The lead personally used `view_image` on all8 onset figures and all8 final
continuation figures, plus four original cap-case photos and all five donor
photos. The figures preserve context and mark forced boxes separately; only
the first five free complete rows are overlaid. Later-row claims come from
the complete raw records and reduction, not an assertion of exhaustive visual
inspection of every generated box.

-477415 starts on a real left-edge chair back.502725 starts over real
  overlapping knives/utensils, with individual versus grouped extent uncertain.
  417044 has real partially clipped donuts at the left edge. These first
  detections must not inherit a blanket hallucination label from later loops.
-351017 starts at the extreme upper-left dark border, where the lead found
  no convincing bottle support; actual rear-wall bottles are elsewhere. Fine
  image-detail uncertainty remains, so this is a bounded visual observation.
-158044/274509 contain real books, but stack/group extents and different book
  regions prevent labeling a whole category run as one physical-owner chain.
  248167's clipped decorative-container region is plausible, with uncertain
  class and individual extent.

Repetition is not merely an artifact of invalid geometry. Zero-based first
strict-repeat versus first geometry-invalid indices include351017:2 versus147,
477415:7 versus11,502725:5 versus59.417044 repeats through valid complete rows;
its final parser drop is an incomplete capped row. Invalid geometry can become
a later failure state, but it is not required to initiate these chains.

## Full factorial accounting — diagnostic, not a policy score

|16 cells per condition|EOS within horizon|Free tokens|Strict valid later repeats|Geometry-invalid free rows|
|---|---:|---:|---:|---:|
|Original image, native history|8|5126|299|122|
|Original image, translated history|13|2990|76|60|
|Donor image, native history|14|1697|53|88|
|Donor image, translated history|15|2186|33|108|

These overlapping-prefix, deliberately selected cells are not independent
population samples. More EOS or fewer strict repeats alone does not imply
quality: wrong-category outputs, invalid rows, omitted detections and forced
context remain. There is no held-out or generalization result here.

## Implication for punishment and the next research decision

**Suppressing the later pathological chain is worth targeting; a larger weight
on the same coordinate-only loss is not supported.** The previous fixed32 run
already reduced valid strict repeats582→487 while increasing geometry-invalid
rows788→888, with unchanged four caps and three fewer annotated TP50 matches.
The current probe independently shows that a loop can mix valid repeated and
invalid rows, and that its first detection can be physically plausible.

For any future learning design, the decision-bearing requirements are:

1. Preserve the user's simple IoU>0.95 selector for later valid repeated rows;
   do not convert all GT-unmatched detections or plausible first owners into
   negative labels. Pairwise geometry alone does not recognize every drifting
   or group-extent failure, and this panel does not supply a trusted matcher.
2. Make the bad output row/continuation the credit unit and account explicitly
   for invalid-geometry alternatives. This is not equivalent to indiscriminate
   tokenwise UL on shared wrappers, descriptors or all nearby coordinates;
   that could suppress legitimate enumeration rather than the repeated action.
3. Verify fewer actual repeated/invalid emissions and less cap burden in
   **fresh natural rollout**, jointly with preservation and novel detections.
   A forced continuation that can emit a cake/different donut is only a
   conditional capability witness, not a demonstrated autonomous repair.

Validity-aware negative credit and useful-next-row credit are bounded design
hypotheses, not a tested algorithm or license to amplify a coefficient now.
Spatial traversal versus owner coverage, the origins in training, small-object
susceptibility and KV/special-token circuit roles remain unresolved. No further
training, architecture surgery or model call was launched after the frozen stop.

## Technical acceptance, cost and worker trial

- Same Stable50 adapter and Source special embeddings; FP32/SDPA, native
  greedy policy, temperature0, RP1, top-p1, top-k0, EOS151645. No updates.
- Eight independent GPU workers handled the eight cases after a five-cell
  real smoke. All8 natural trajectories exactly reproduce saved Stable50
  action IDs. All16 unchanged-image/native-prefix suffixes exactly reproduce
  the corresponding natural slices; smoke/full overlap is exact on all5 cells.
- The initial ranks3/7 failed after model load, before any model forward or
  continuation, because the **lead's** two new donor records violated the
  relative-image-path/coord-token input schema. Original failed receipts and
  packet remain unchanged. CPU native-call preflight reproduced the error and
  validated two canonical-record envelope repairs. Same image bytes, prompts,
  grids, model, prefixes, jobs and decode; only those two unexecuted cases were
  reacquired. Six successful ranks were not rerun. The original panel terminal
  correctly remains failed; the combined reduction owns complete74-cell scope.
- All74 raw cells passed literal-token/native-parser cold readback and packet,
  prefix and budget checks. Sol-high produced the census and visual packages;
  Luna-max produced the native runner and seven CPU invariant tests. Root
  corrected a forced-history owner-credit flag and required missing CUDA-peak
  accounting before acceptance; Luna implemented the corrections. The donor
  input-envelope error belongs to root, not the worker. This is a successful
  task-specific delivery with lead correction, not a blanket quality verdict.
- Accepted full panel:25,406 generated tokens/model forwards,74 image forwards,
  eight model loads,2038.47 summed GPU-assigned worker seconds. Smoke adds4130
  tokens, five image forwards, one load and319.66 seconds. Two failed loads add
  35.53 seconds and zero forwards/tokens. Total:79 continuations,29,536 generated
  tokens/model forwards,79 image forwards,11 loads,2393.66 seconds
  (**0.665 allocated GPU-hours**, not measured utilization percentage).
- Peak CUDA allocated9,982,440,448B; peak RSS12,150,964,224B. The two technical
  failures explain the11 versus nine planned loads; generation stays within
  the frozen79-cell/68,740-token bounds. Stable50 remains unchanged; no default,
  GT, confirmation512 axis, Git commit or production artifact was changed.

## Reproduction and artifact entry points

- [Protocol and frozen stop](unit.md)
- [Authoritative74-cell reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/reduction.json)
- [Original packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/packet.json)
- [Input-envelope repair packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/packet-repair-01.json)
- [CPU repair preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/repair-preflight.json)
- [Onset visual observations](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/visual-review.json)
- [Continuation visual observations](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/continuation-visual-review.json)
- [All8 continuation figures and literal overlays](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/continuation-visualizations/manifest.json)
- [Final lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/lead-acceptance.json)

Original packet SHA256:
`48087dcf05e03fc9b6c2eb9de1e2d66e1efe09c972ce22e8c6ae2774f95aa4ba`.
Repair packet SHA256:
`3a460d88f0e7697fd892459c380417a85da9dd03bed412b6413dcfc8dff556c3`.
The reducer binds each raw cell to its actual producer packet. Its literal
readback and summarization functions can be replayed without model calls;
`reduce_panel.py` uses exclusive output creation to avoid overwriting evidence.
