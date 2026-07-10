# Prefix-Denoising Surgery Probing Roadmap

Date: 2026-06-25

Use this plan with `superpowers:subagent-driven-development` for independent
lanes. The plan is intentionally sample-base-centered: select a small set of
mechanistically rich images first, then spend GPU time deeply on those images
instead of broadening over normal learned cases.

## Objective

Build an artifact-backed mechanism picture of how prefix-denoising SFT changes,
or fails to change, the V-LLM internal path from image evidence to object span:

```text
visual evidence -> object/semantic binding -> coordinate basin ->
box span emission -> next-object transition or duplication/termination collapse
```

The study compares prefix-denoising checkpoints against the closest
non-denoising controls, with sorted/random treated as blocking variables.

## Task 0: Registry And Control Discovery

- [ ] Create a model registry with exact checkpoint paths, rollout artifact
      paths, training config paths, adapter surfaces, and template identifiers.
- [ ] Confirm that both prefix-denoising checkpoints use
      `token_embeddings_adapter` as the trainable new-token surface.
- [ ] Find the closest pure-CE compact_object_box_closed desc_first full-wrapper
      control checkpoint and val200 artifact.
- [ ] If no close control exists, document fallback controls and template
      mismatches before running surgery.
- [ ] Record prior average-denoising evidence only as context, not as a causal
      conclusion.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/model_registry/model_registry.json
outputs/analysis/prefix_denoising_surgery_probing/model_registry/model_registry.md
```

## Task 1: Sample-Base Harvest And Curation

- [ ] Parse available val200 rollout artifacts for sorted and random
      prefix-denoising checkpoints.
- [ ] Parse the chosen control artifact(s) when available.
- [ ] Join predictions with GT and image metadata at image_id granularity.
- [ ] Identify candidate image bases for false negatives, duplication bursts,
      divergent same-image behavior, boundary/termination faults, and a small
      sanity-control set.
- [ ] Select 8-12 deep image bases, plus at most 2 easy sanity controls.
- [ ] For every selected base, write a manual-review-ready rationale explaining
      why this image is worth deep probing.
- [ ] Avoid spending model-backed work on normal/well-learned images unless
      they are explicit sanity controls.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/sample_base_registry/sample_base_candidates.jsonl
outputs/analysis/prefix_denoising_surgery_probing/sample_base_registry/sample_base_registry.jsonl
outputs/analysis/prefix_denoising_surgery_probing/sample_base_registry/sample_base_summary.md
```

Current focus-panel artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/focus_panel/v1_surgery_focus_no_sanity/focus_panel.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/focus_panel/v1_surgery_focus_no_sanity/focus_panel_summary.md
```

Snapshot: the strict surgery panel excludes easy sanity controls and broad
complete-row failures for this deep round, selecting image bases `17899`,
`2685`, `12670`, `16228`, and `2299`. The panel deliberately prioritizes
observed atlas anomalies and local transition/basin failure over generic high
metric severity: `17899` is now an observed terminal-router plus coordinate
basin case; `2685` is a wine-glass/person repeated-anchor basin; `12670` is a
person/backpack/handbag crowd-binding basin; `16228` is a large person/train
over-emission basin; `2299` is a person/tie case where the model families fail
oppositely at the boundary. Broad empty/normal ids held out for this phase:
`18380`, `9590`, `17207`, `9772`.

## Task 2: Position Plan For Selected Bases

- [ ] Materialize token-position plans for each selected image base across the
      treatment and control models.
- [ ] Include descriptor, wrapper, coordinate, post-box, next-object, and
      termination positions.
- [ ] Preserve image_id, target object id/category, rollout span id, generated
      object order, GT box, prediction box, prefix text, token ids, and template
      compatibility fields.
- [ ] Flag rows as coordinate-mode, wrapper-mode, semantic-mode, or termination
      pressure where possible.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/position_plan/position_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/position_plan/position_summary.md
```

Current v1 artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_summary.md
```

Snapshot: 14 selected sample bases, two prefix-denoising treatment models,
11,648 token-position rows, 896 complete generated object spans, no skipped
selected image/model pairs. The plan preserves token text and rollout logprob;
`pred_token_trace.jsonl` does not carry token ids, so token ids are explicitly
marked as post-hoc tokenizer-join fields rather than silently invented.

Early case-shaping observation from the row plan: high-value selected images
show large repeated-class basins rather than normal learned behavior. Examples
include `wine glass`, `person`, `bowl`, and `chair` bursts in random denoising,
and a distinct `donut` collapse basin for sorted denoising on image 17899. Treat
these as next-probe selection cues, not as final mechanistic conclusions.

## Task 3: Layerwise Atlas

- [ ] Run hidden-state/logit atlas over selected bases and planned positions.
- [ ] Track coordinate-family mass, local coordinate-window mass, target rank,
      nearest-coordinate rank, wrapper-token mass, semantic-token pressure, and
      termination pressure.
- [ ] Compare prefix-denoising versus control pairwise on the same image/prefix.
- [ ] Stratify rows by false-negative, duplication, boundary, divergent, and
      sanity families.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/atlas_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/atlas_summary.md
```

Current smoke artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_smoke_image17899_sorted_prex1_2rows_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_smoke_image17899_random_prex1_2rows_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_image17899_sorted_prex1_term_layers0_8_16_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_image17899_random_prex1_term_layers0_8_16_last_gpu1
```

Smoke scope: image 17899, two `pre_x1` source rows per model, layers `0,-1`.
All four atlas rows per model reported `readout_status=ok` and
`coord_lens_available=True`. Early layer rows had low coordinate-family mass
and non-coordinate top token text (`k`), while final-layer rows moved into
high coordinate-family mass and exact or near-exact coordinate basins. In the
random checkpoint smoke, the second selected object landed one bin off
(`coord_281` for target `coord_282`), making coordinate-basin near-miss rows a
promising next expansion target.

Expanded image-17899 scope: all `pre_x1` plus terminal-pressure source rows for
both denoising checkpoints at layers `0,8,16,-1`. The final layer shows strong
coordinate-family capture but many repeated-class rows miss the intended x1
basin by tens of bins: sorted `donut` rows often land near a neighboring donut
anchor, and random `cake` rows often land near repeated cake anchors. The
terminal row exposed a sharp next-object-vs-stop router fork: sorted ended with
`P(object_ref_start)=0.5307` and `P(im_end)=0.4684`, while random ended with an
almost exact `P(im_end)=P(object_ref_start)=0.499711` tie resolved toward
`im_end`. Treat this as a high-priority site for termination/continuation
surgery and not yet as a global conclusion.

Expanded focus-panel atlas artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairA_sorted_2685_12670_prex1_term_layers0_8_16_20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairA_random_2685_12670_prex1_term_layers0_8_16_20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairB_sorted_16228_2299_prex1_term_layers0_8_16_20_24_last_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairB_random_16228_2299_prex1_term_layers0_8_16_20_24_last_gpu3
```

Expanded scope: four additional focus bases, both denoising checkpoints,
positions `pre_x1,termination_pressure`, layers `0,8,16,20,24,-1`; 296 source
positions and 1,776 atlas rows, all with `readout_status=ok`. Early evidence:

- Coordinate-family mass becomes decisive only late, but late confidence does
  not mean correct binding. Many final-layer rows have `coord_vocab_mass` near
  0.99 and target rank under 10 while top-1 collapses to another anchor.
- `2685` cleanly separates the checkpoints: random emits 54 `wine glass` spans
  and has 19/61 final x1 rows at distance >=32, while sorted has 0/23 far rows
  and mean final x1 distance 4. This is a strong first coordinate-basin surgery
  candidate.
- `16228` and `12670` expose a border/anchor basin: many repeated `person`
  spans choose `coord_0` at final x1 even when the emitted target x1 is hundreds
  of bins away. Examples include random image `16228` object 45 with target
  `coord_926`, top-1 `coord_0`, target rank 5, and final coord mass 0.9929.
  Treat small ranks here cautiously because many coordinate bins are tied or
  near-tied.
- Terminal readout shows a repeatable stop/continue router family. Sorted
  continues on `17899` and `2685`, but stops strongly on `2299`, `12670`, and
  `16228`; random is exactly tied on `17899`, `2685`, and `12670`, continues on
  `16228`, and weakly stops on `2299`. These rows should feed direct
  termination/continuation surgery.

Current direct tail-slot all-layer atlas:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_16228_obj16_layers0_4_8_12_16_20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_16228_first_token_only_layers0_4_8_12_16_20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_2685_obj3_layers0_4_8_12_16_20_24_last_gpu2
```

Scope: exact generated-object-index filtering over `16228` object 16,
`16228` objects 34/45/51, and `2685` object 3, positions
`pre_x1,post_x1_pre_y1,post_y1_pre_x2,post_x2_pre_y2`, layers
`0,4,8,12,16,20,24,final`; 160 atlas rows, all `readout_status=ok`.

Bridge finding: tail-slot tensor flow now points to a layer-24-to-final
transition. Layer 24 can carry nearby or low-rank target hints without full
coordinate-family saturation; the final layer saturates the coordinate-token
simplex. Under correct guided tail prefixes this often materializes the target
coordinate, while under `pre_x1` state-entry collapse it can materialize the
wrong border or repeated anchor. Treat the next component patch as a test of
which layer-24/final component turns a hidden hint into the final coordinate
basin.

## Task 4: Paired Difference Flow And Component Split

- [ ] Compute treatment-minus-control residual/logit/readout deltas by layer
      and token position.
- [ ] Split layer input, self-attention output, MLP output, and final residual
      contributions when hooks support it.
- [ ] Do not pool coordinate-mode x1-onset rows with wrapper/router boundary
      rows.
- [ ] Prioritize layers 16-18 for coordinate-basin onset and later layers for
      boundary/termination routing, but let the atlas reveal unexpected sites.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/difference_flow/difference_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/difference_flow/difference_summary.md
```

## Task 5: Cross-Model Surgery

- [ ] Patch control states into prefix-denoising models and prefix-denoising
      states into controls at matched image/prefix/position/layer/site.
- [ ] Measure immediate next-token movement and coordinate-basin movement.
- [ ] Continue generation for selected high-effect rows to test whether patching
      repairs an FN, triggers or suppresses duplication, changes object order, or
      changes termination.
- [ ] Include no-op/self-patch controls and incompatible-image controls.
- [ ] Shard by GPU across image bases, models, layers, and patch sites.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/cross_model_surgery/surgery_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/cross_model_surgery/continuation_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/cross_model_surgery/surgery_summary.md
```

Current residual/readout-space surgery artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_coord_focus_2685_16228_layers20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_terminal_focus_2685_16228_layers20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_terminal_17899_2685_layers24_last_gpu2
```

Scope note: these are normalized readout-space perturbations before `lm_head`,
not full forward activation patches or continued generation. They test whether
the selected hidden states are locally movable along target-minus-antagonist
directions. They should feed, not replace, the full continuation/patching pass.

Findings from this first surgery pass:

- Coordinate basins are highly movable once the readout is asked the right
  question. For 12 random-denoise coordinate-collapse source states from
  images `16228` and `2685`, 252 readout-surgery rows produced 89 exact
  flips-to-target, 167 coordinate-distance improvements, and 216 rank
  improvements.
- The `16228` random `person` coord-0 collapse is especially diagnostic.
  Across 11 selected rows, final-layer mean x1 distance starts near 745.8 bins;
  alpha `0.05` along the target-minus-antagonist readout direction flips 8/11
  rows to exact target and reduces mean distance to about 1.36 bins. Layer 24
  is already close: alpha `0.1` flips 9/11 and reduces mean distance to about
  0.18 bins.
- `2685` random has a selected person/wine-glass-basin row where final-layer
  x1 is 519 bins off at alpha 0 but becomes exact by alpha `0.02`. This is a
  compact candidate for full continuation surgery.
- Terminal router states are also locally fragile. In the `17899/2685`
  terminal comparison across both model families, final-layer stop-vs-continue
  readouts flip to `im_end` by alpha `0.005`; layer 24 generally needs about
  alpha `0.05`. This supports a late router-knife-edge hypothesis rather than
  a hard, inaccessible termination decision.

Interpretation guardrail: because the perturbation is applied in readout space,
these results show local basin movability and target-direction accessibility.
They do not yet prove that a full hidden-state patch will repair free
autoregressive continuation. The next causal step should patch or steer actual
layer outputs around layer 24/final and continue generation for the selected
rows.

Current full-forward prefill activation-patch bridge:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_coord16228_bridge_true_logits_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_coord2685_bridge_true_logits_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_terminal_bridge_true_logits_layers24_last_gpu2
```

Scope note: the `v2_*` activation-patch rows inject the same
target-minus-antagonist direction into actual decoder-layer outputs at the
assistant-prefix decision token and read true `outputs.logits`. Earlier
`v1_*` activation-patch rows are superseded because they reconstructed logits
from returned `hidden_states[-1]`, which a diagnostic showed can miss
final-layer output-hook effects.

Bridge findings:

- Coordinate next-token logits are causally movable in the true forward pass.
  Image `16228` random-denoise repeated-person coord-0 rows flip exactly in
  12/32 layer/alpha rows, with layer 27 typically needing alpha `0.02` and
  layer 24 alpha `0.05`. Image `2685` random object 3 flips by alpha `0.01`
  at both layers 24 and 27.
- Terminal-router logits are extremely fragile. Across images `17899` and
  `2685` for sorted/random denoising, all four selected terminal-pressure
  states flip toward `<|im_end|>` by alpha `0.005` at both layers 24 and 27.
- This bridges readout-space surgery to real next-token causal evidence, but
  it still does not prove multi-token continuation repair. The next Task 5
  step should reuse these exact source states for short patched continuation:
  coordinate patch should produce coherent y1/x2/y2 continuation, and terminal
  patch should stop without reopening a duplicated object.

Current short-continuation bridge:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_coord16228_continuation_repair_labels_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_coord2685_continuation_repair_labels_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_terminal_continuation_repair_labels_gpu2
```

Bridge findings:

- Coordinate repairs split into two mechanisms. `16228` object 16 becomes a
  full-tail coherent repair: patching x1 to `coord_829` is followed by
  `coord_384, coord_861, coord_453, box_end`. But `16228` objects 34/45/51
  and `2685` object 3 are first-token-only or near-tail repairs: x1 flips, but
  later slots are pulled to nearby coordinate basins.
- Terminal repairs are clean for sorted denoise. Sorted `17899` and sorted
  `2685` switch from `<|object_ref_start|>` continuation to one-token
  `<|im_end|>` stops by alpha `0.005` at layers 24 and 27.
- Random terminal rows expose a tie/greedy distinction rather than a repair
  need: manual greedy already emits `<|im_end|>` at alpha 0, even when the
  readout margin is zero or top-k reporting can make continuation look equally
  plausible.

Current direct tail-slot continuation bridge:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_16228_obj16_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_16228_first_token_only_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_2685_obj3_layers24_last_gpu2
```

Bridge findings:

- Exact object-index filtering now lets this branch spend GPU work on
  representative sample bases instead of normal or well-learned rows. The first
  narrow set is `16228` object 16, `16228` objects 34/45/51, and `2685` object
  3.
- Direct tail prefixes where the target is already rank-1 collapse the
  target-minus-antagonist direction to zero. Treat this as a probe-handle
  boundary and a clue: under exact guidance, those tail slots are locally
  learned. Their free-rollout failure is upstream in state entry or tail
  retention, not direct local token availability.
- Non-saturated tail rows split into three outcomes: immediate-token-only
  repair, coherent remaining-tail repair, and weak nearby-basin movement.
  `16228` object 51 `post_y1_pre_x2` and `2685` object 3 `post_x2_pre_y2`
  are the strongest direct-tail coherent repairs.

Current component-site continuation bridge:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj16_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj34_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj45_layers24_last_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj51_layers24_last_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_2685_obj3_layers24_last_gpu0
```

Bridge findings:

- Four patch sites were tested over the same selected object bases:
  `layer_output`, `layer_input`, `self_attn`, and `mlp`.
- Across 480 valid component rows, first-token target repair counts were
  `layer_output=38`, `layer_input=35`, `mlp=27`, `self_attn=11`; coherent
  tail repairs were `layer_output=13`, `layer_input=13`, `mlp=12`,
  `self_attn=6`.
- This localizes most coordinate-token steering to MLP/layer-output routes.
  Attention-only patches can move some rows but are consistently weaker.
- The `pre_x1` rows remain first-token-only: component patches can repair x1,
  but do not restore y1/x2/y2 as an object-tail manifold.

Sample-base selection rule for the next round: treat the image base as the
primary unit of deep research. Spend narrow GPU/probe budget on representative
failure-rich or contrastive bases first: duplication bursts, false negatives,
premature-stop/continuation ties, and same-image object pairs where one span is
full-tail coherent while a neighbor is first-token-only. Normal or well-learned
images should be used as controls, not as the main discovery pool.

Current object-state selector:

```text
src/analysis/prefix_denoising_surgery_probing/object_state_selector.py
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v4_v8_v9_component_site_join_diverse
```

Selector findings:

- The selector ranks exact `(image, model, object, position)` states from the
  val200 position plan, joins activation-patch evidence post hoc, and caps
  selected rows per image/model/family to keep the deep panel representative.
- The current joined panel has `5376` candidates, `960` activation rows joined,
  and `32` selected states over image bases `2157`, `2299`, `2685`, `12670`,
  `14439`, `16228`, and `19432`.
- The top selected contrast remains `16228/2685`: `pre_x1` states are
  first-token-only while later tail slots have coherent repairs.
- Fresh `v9` probes show two saturated `coord_0` duplicate anchors:
  `19432` random object 12 pre_x1 and `12670` sorted object 0 pre_x1 both give
  `40/40` `direction_error` rows when reinforcing the emitted `coord_0`.
- `2157` sorted object 13 pre_x1 gives `26/40` first-token-only repairs and no
  coherent tail repair, making it a clean state-entry/tail-binding case.
- `2299` random object 27 termination pressure already emits `<|im_end|>` in
  `40/40` rows, so that state is a terminal stability/control row rather than a
  stop-repair target.

Next Task 5 focus: stop treating coordinate repair as a binary next-token
event. Local coordinate-token steering is now partially localized; the deeper
open problem is state-entry and tail binding. Prioritize probes that test which
component couples a repaired x1 to the subsequent y1/x2/y2 trajectory, not just
which component can move one coordinate logit. For saturated anchor rows, stop
reinforcing the emitted `coord_0`; build anti-anchor or alternate-GT target
directions and ask whether the basin can be escaped before the final coordinate
simplex. For terminal rows, preserve tie semantics explicitly and ask why
sorted denoise keeps continuation pressure where random greedy already falls to
stop.

Anchor-escape bridge completed:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_plan.py
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_plan/v2_saturated_direction_coord0_gt_candidates
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_reduce/v2_v12_layer_sweep
```

Bridge findings:

- The strict planner reduced saturated `coord_0` duplicate-anchor rows to four
  same-description, non-border GT-candidate x1 targets over the two strongest
  bases: `12670/sorted/person` and `19432/random/chair`.
- Full layer sweep scope was all decoder layers `0..27`, patch sites
  `layer_output,layer_input,mlp,self_attn`, alphas
  `0,0.005,0.01,0.02,0.05,0.1`, and `continuation_steps=0`.
- The best anchor-escape rows are late and residual-stream/readout dominated:
  all group bests are layer 27 `layer_output` at alpha `0.1`.
- `19432/random/chair` shows a strong alternate-coordinate basin: target
  `coord_537` flips exactly at late `layer_output/layer_input`, and target
  `coord_351` reaches rank 1 with top1 `coord_350`.
- `12670/sorted/person` is weaker and more basin-confused: target `coord_344`
  reaches a near top1 `coord_346`, while target `coord_492` is pulled toward
  `coord_429`.
- Post-hoc destination checks show these wrong/near bins are not arbitrary:
  `12670` destinations around `414/427/429` align with neighboring
  person/backpack/teddy-bear GT or prediction anchors, while `19432`
  destinations around `350/537/598/414` align with real chair basins.
- Attention-only patching is weak for this mechanism; prioritize
  `layer_output`, `layer_input`, and secondarily MLP for the next local probes.

Revised next Task 5 direction:

- [x] For `19432/random/chair`, use the exact/near x1 escape rows as a
      tail-binding receiver panel: after x1 moves to `coord_537` or near
      `coord_351`, test whether y1/x2/y2 can be made to follow the same GT
      object or snap back to the old repeated-chair manifold.
- [ ] For `12670/sorted/person`, inspect destination bins `414, 427, 429,
      346` against GT and predicted boxes, then test whether these are real
      neighboring person basins, mixed object/coordinate attractors, or learned
      coordinate priors.
- [ ] Restrict the next value/region/head analysis for this bridge to layers
      `22..27`, unless a control requires earlier layers.

Tail-binding bridge completed:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding.py
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_reduce/v1_strict4_all_modes_iou_quality
```

Bridge findings:

- `19432/random/chair` is the current positive tail-binding case. Free
  continuation remains non-target, but forcing alternate x1 yields target IoU
  `0.584/0.632`, and forcing x1+y1 yields target IoU `0.900/0.996`.
- `12670/sorted/person` is the current negative tail-binding case. Free
  continuation replays the original huge border box. Forcing x1, x1+y1, or
  x1+y1+x2 still gives low-IoU boxes, usually with shallow y2 around `415`;
  only forcing the full box makes closure exact.
- The shallow `12670` y2 around `415` aligns with existing predicted
  person/backpack anchors (`415/416`) and nearby GT y2 bins, while the recovered
  `19432` chair y2 values align with chair GT/prediction basins. Treat y2
  closure as destination-basin competition, not random tail noise.
- The next mechanism question is therefore no longer "does alternate x1 exist?"
  but "why does the chair case have a recoverable y1/x2/y2 object scaffold
  while the person case lacks vertical-extent closure?"

Revised next Task 5.1 direction:

- [ ] Compare visual/value routes at layers `22..27` for `19432/random/chair`
      target GT 8/5 versus `12670/sorted/person` target GT 15/8, focused on
      y1/x2/y2 and especially y2 closure.
- [ ] Build a tiny destination-basin table for the generated tails: original
      anchor replay, valid non-target, low-IoU target overlap, target IoU50,
      target IoU75.
- [x] Probe whether the `12670` shallow-y2 around `415` aligns with a real
      neighboring person/teddy/backpack basin or a learned small-box extent.

## Task 6: False-Negative Perception Versus Guidance Panel

- [ ] For selected missing objects, test empty/minimal prefix, correct language
      guidance, wrong-object guidance, correct coordinate seed, wrong coordinate
      seed, and hidden-state patching.
- [ ] Label an object as visually unavailable only if the model fails under
      multiple guidance and patching routes.
- [ ] Compare train-set failures and val-set failures when matching image bases
      or analogous object layouts are available.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/fn_guidance/fn_guidance_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/fn_guidance/fn_guidance_summary.md
```

## Task 7: Coordinate Token Surface

- [ ] Inspect `<|coord_0|>` through `<|coord_999|>` input embeddings, output
      directions, adapter deltas, neighborhood geometry, and smoothness.
- [ ] Compare prefix-denoising checkpoints, control checkpoints, and base model
      surfaces where available.
- [ ] Connect coordinate-surface effects to selected sample-base basins instead
      of reporting only global embedding geometry.

Acceptance:

```text
outputs/analysis/prefix_denoising_surgery_probing/coord_token_surface/coord_surface_rows.jsonl
outputs/analysis/prefix_denoising_surgery_probing/coord_token_surface/coord_surface_summary.md
```

## Task 8: Synthesis And Next Fork

- [ ] Write a synthesis note only after at least one model-backed atlas or
      surgery pass over selected sample bases.
- [ ] Separate visible symptoms, intervention evidence, likely mechanism, and
      unresolved alternatives.
- [ ] Decide whether to deepen on coordinate-basin onset, boundary/router
      collapse, FN guidance/perception, coordinate-token surface, or a limited
      training-step intervention.

Acceptance:

```text
progress/diagnostics/2026-06-25_prefix_denoising_surgery_probing_findings.md
```

## Stop Conditions

Stop for user review if:

```text
the selected sample-base panel requires manual visual adjudication before model-backed runs
the closest control checkpoint cannot be established without a high-mismatch fallback
the first atlas shows no stable paired differences and a new probe family is needed
the cross-model surgery produces ambiguous movement that needs research-priority choice
```

Otherwise, keep moving through the sample-base-centered loop and dynamically
deepen any path that materially changes the final mechanism picture.
