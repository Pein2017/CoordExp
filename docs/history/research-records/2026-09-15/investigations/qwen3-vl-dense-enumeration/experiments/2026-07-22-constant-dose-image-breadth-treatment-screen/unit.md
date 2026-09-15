---
title: Constant-Dose Image-Breadth Treatment Screen
description: A matched experiment that holds row-event count and optimizer updates fixed while spreading sampled-route and Source-preservation supervision across more physical images.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-22-constant-dose-image-breadth-treatment-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_no_heldout_image_breadth_advantage
updated: 2026-07-23
---

# Constant-Dose Image-Breadth Treatment Screen

## Execution Outcome

The screen is complete. At the frozen 992-event, 31-update dose, spreading
supervision over 496 images instead of concentrating it in 162 matched images
did not improve annotated-owner coverage on the pre-admission held-out cohort.
The selected sampled-route owners were nevertheless recovered much more often
than non-selected owners on gradient images. This strongly enriched recovery is
consistent with owner-specific uptake of the supervision, but it is not a
same-owner untreated counterfactual. This screen did not establish that greater
image breadth turns that local effect into a transferable set-expansion rule.

The primary conclusion uses repetition penalty 1.0. A separately executed
repetition-penalty-1.10 sensitivity panel changed many owner identities and
lowered the Source matched-owner count without reliably increasing treatment
matched-owner count. It must therefore be treated as a decoding-policy
intervention, not as a harmless duplicate-only adjustment. The first attempted
1.10 panel is invalid because the runtime silently remained at 1.0; only the
validated second panel contributes evidence.

See `results.md` for the complete result, artifact identities, uncertainty,
and the next research decision. Do not scale this unchanged positive-only
complete-row treatment by adding more images or epochs.

## Question

Did the previous treatment exchange physical owners because 496 sampled-route
events were concentrated in only 118 images, or is owner exchange intrinsic to
positive-only complete-row imitation under the current autoregressive model?

This unit changes image breadth while holding the training dose fixed. It is
not a larger epoch and is not promotion to full-size training.

## Evidence That Motivates the Test

The completed 256-image Source-route-preservation screen established three
facts:

1. selected physical owners are recovered approximately four to six times as
   often as non-selected missed owners;
2. admitted images can gain owners and box quality improves;
3. non-admitted images lose owners at every evaluated milestone.

The supervision is therefore learnable, but its effect is narrow. More updates
do not repair transfer. A plausible explanation is that too few different
images expose the rule that training should learn. The strongest alternative
is that complete-row imitation merely changes which greedy route wins, so
broader images will change the identities of gained and lost owners without
increasing the final set.

## Competing Hypotheses

### Image-breadth hypothesis

The current gradient is dominated by repeated rows and routes from a small
number of images. Spreading the same amount of credit across many more images
will reduce image-specific route fitting and produce net owner gains on images
that never supplied gradients.

Predictions:

- the broad arm exceeds the concentrated arm on never-trained images;
- gained owners outnumber lost Source owners at both intersection-over-union
  thresholds 0.30 and 0.50;
- the admitted-positive and non-admitted-negative split shrinks or disappears;
- output validity, geometry, and duplicate behavior remain acceptable.

### Intrinsic owner-exchange hypothesis

Positive complete-row imitation changes local route preference but does not
teach a stable rule for preserving the rest of the object set. More image
diversity will change which owners are recovered, but a comparable number of
Source owners will still disappear.

Predictions:

- selected-owner recovery remains enriched;
- total owner delta remains near zero or negative on never-trained images;
- gained and lost owners remain balanced or losses dominate;
- standard mean Average Precision may improve without owner-set expansion.

### Optimization-dose alternative

The treatment may be useful only early in optimization. This explanation
predicts an early broad-arm milestone that transfers better than later
milestones. Holding total updates at 31 and saving the same milestones keeps
this explanation visible without confounding it with a longer epoch.

## Frozen Source and Data Surface

Use the geometry-sorted, description-first, pure-cross-entropy plus token-type
gate Qwen3-VL 2-billion-parameter Source checkpoint at step 4,887. Freeze the
vision tower, multimodal aligner, selected-token embedding delta, prompt,
tokenizer, coordinate surface, and greedy decode policy.

Create a deterministic 2,432-image label-only candidate pool from the same
Common Objects in Context training source and selector used by the earlier
768-image pool. Use the same selection seed so the earlier pool is expected to
be an exact subset. Verify the same image identities, resolved source images,
and all non-path row fields rather than assuming it. Byte equality is not
expected because the selector rebases relative image paths to the output file.

Logical pool root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
candidate-pool-v1/candidate-pool-2432.coord.jsonl
```

The pool contains 608 images in each of four object-count bands. Before
inspecting route eligibility, split it by image identity and band into 2,048
training-candidate images, 256 development images, and 128 held-out images.
Each band contributes 512, 64, and 32 images respectively. The split uses only
label-side object counts and a fixed seeded allocation; sampled-route outcomes must not
affect membership. Write the split and its source hash to an immutable receipt.
Keep byte-preserving split and complement JSONL files beside the candidate pool
so their relative image paths retain exactly the same meaning.

Regenerate the complete 2,432-image trajectory panel with one unified vLLM
high-throughput inference backend. Do not combine the earlier Hugging Face
256-image routes with new vLLM routes. For each image, collect sixteen sampled
trajectories at temperature 0.4, nucleus probability 0.95, repetition penalty
1.0, and a 1,024-token generation allowance. Identify them by `sample_index`
0 through 15; request seed is not a research variable.

Use eight disjoint image shards, one graphics-processing-unit worker and one
vLLM engine per shard, with scheduler capacity 32 and 16 images per persisted
batch. Set the prompt-plus-generation model length to 4,096. Require natural
closure for every route and stop a worker after persisting evidence if any
route ends because of length.

Do not include a greedy route in this production panel. A targeted four-image
smoke showed that greedy decoding at repetition penalty 1.0 entered stable
repeated-row loops and consumed the full token allowance, whereas all 64
matched low-temperature samples closed naturally. Raising the greedy limit,
allowing truncation, or changing its repetition penalty would each change the
comparison rather than produce a valid matched completion. The complete
production panel therefore estimates sampled object support only. Its receipt
and exact claim boundary are recorded in
`trajectory-panel-vllm-receipt.md`.

Collect a separate finite Source baseline before route admission. Define
`Source@B16` as greedy decoding at repetition penalty 1.0, stopped immediately
after the sixteenth complete object row or at an earlier natural image-end.
Here `B16` means a budget of sixteen complete object rows.
It is a semantic row-budget view, not a token-length-truncated completion. A
run that reaches the token limit before either condition is incomplete and
cannot define Source ownership. The runtime may generate beyond row sixteen
when that is operationally simpler, but no later row participates in the
baseline. Project every sampled trajectory to its first sixteen complete rows
for the matched comparison. Greedy decoding at repetition penalty 1.10 is a
separately named decoding-policy sensitivity check, not the route-admission
baseline.

Treat `Source@B16` as a policy-conditioned, immutable realized baseline, not as
a batching-invariant property of the checkpoint. The 64-image qualification
showed that changing only the request batch partition can change meaning-bearing
greedy rows. Freeze vLLM 0.14.1, Brain Floating Point 16-bit computation,
`max_num_seqs=32`, `image_batch_size=16`, eight stable strided image shards,
request order, prompt and model identities, and the exact artifact realization.
Two independent Graphics Processing Unit runs under the frozen per-engine
batch policy reproduced all 64 raw and projected token hashes exactly. The
qualification and full-panel evidence are recorded in
`source-b16-vllm-receipt.md`.

An image with `failed_invalid_before_budget` or
`failed_token_limit_before_budget` has no usable Source baseline. This does not
mean that its Source owner set is empty. Persist the evidence, continue the
remaining collection, and exclude the image from admission. Do not retry it
under another batch grouping to obtain a favorable Source realization.

Development and held-out trajectories may be materialized for later
evaluation, but no admission statistic from those images may affect either
training bank.

The twelve human-refined validation images remain development safety evidence
only. They never supply gradients and are not part of the 2,432-image pool.

## Route Admission and Matched Event Selection

Apply the same conservative route admission semantics as the completed screen:

- parser-accepted and naturally closed route;
- verified annotated physical-owner set within its first sixteen complete rows
  strictly expands the `Source@B16` owner set;
- no increase in confirmed duplicates or malformed rows;
- only trusted first-occurrence physical owners receive gradient;
- entity identity and coordinate trust remain separate;
- unresolved or annotation-ambiguous rows receive no gradient.

Build Source-preservation events only from trusted first occurrences inside
`Source@B16`. Rows emitted after the sixteenth complete row do not expand the
Source owner set and do not supply preservation events. Unresolved rows remain
neutral; they are neither trusted owners nor negative examples.

Run route admission only inside the frozen 2,048-image training-candidate
split and only on the intersection of Source-eligible and sampled-eligible
images. Report Source ineligibility by split and object-count band. All owner
claims and feasibility denominators must name this joined eligible cohort.
For each image, define a trusted pair count as the smaller of its eligible
sampled-event count and trusted Source-preservation-event count.

The first gate is a census. Training is feasible only if this condition holds:

```text
eligible unique training images >= 496
```

Every eligible image must provide at least one trusted sampled event and one
trusted Source-preservation event. The gate therefore proves that the broad
bank can use 496 different physical images with no second event from any image.
If it fails, stop before training rather than weakening trust rules.

The original equal-band design failed a stronger pre-training feasibility
audit and is superseded prospectively. The joined eligible training cohort has
797 images: 33 sparse images with one to three annotated objects, 158 medium
images with four to seven objects, 294 dense images with eight to fifteen
objects, and 312 very-dense images with sixteen or more objects. Selecting 124
broad images from every band is impossible because only 33 sparse images are
eligible. The original fixed 118-image concentrated cohort, with band counts
30, 30, 29, and 29, also cannot supply 124 distinct trusted pairs per band:
its capacities are respectively 39, 74, 123, and 191. These failures were
found before a StateBank was frozen or training began. Do not weaken trust,
clone events, or substitute another image in response.

For the validated `coordexp_vllm_trajectory_panel.v2` execution contract only,
freeze the prospective protocol amendment
`v2_capacity_constrained_capped_max_min_breadth_v1`. In plain English, first
allocate 496 broad images and pair events by deterministic capped max-min
fairness over eligible object-count bands in the canonical sparse, medium,
dense, very-dense order. Then, within each selected broad band cohort, use the
shortest deterministic image-hash prefix whose cumulative usable trusted pair
capacity reaches that band's event quota. Usable pair capacity is the smaller
of raw distinct pair capacity and eight for each image, because every pair
materializes two complete-row records and the canonical StateBank permits at
most sixteen such records per image. Never skip a prefix image or swap in a
capacity-richer image. Legacy execution contracts retain the original
equal-band 496-versus-118 behavior.

The frozen current-data census is:

| Object-count band | Eligible images | Broad images and pair events | Concentrated prefix images | Raw distinct capacity at prefix minus one / prefix | Usable capacity at prefix minus one / prefix | Pair events per concentrated image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse, 1 to 3 | 33 | 33 | 26 | 31 / 33 | 31 / 33 | 1.2692 |
| Medium, 4 to 7 | 158 | 155 | 72 | 154 / 157 | 154 / 157 | 2.1528 |
| Dense, 8 to 15 | 294 | 154 | 37 | 156 / 161 | 150 / 155 | 4.1622 |
| Very dense, 16 or more | 312 | 154 | 27 | 164 / 174 | 152 / 160 | 5.7037 |

The broad arm therefore uses 496 images and 496 pairs. The concentrated arm
uses 162 images and the same 496 pair-event quotas, for an overall ratio of
3.0617 pair events per concentrated image. The existing independent
within-band round-robin supplies exactly 33, 155, 154, and 154 distinct pairs
without cloning or selecting more than eight pairs for any image. Record the
exact allocation proof in every selection and arm receipt.

Build two newly matched `StateBank` artifacts. Here, a StateBank is the
persisted collection of model states, prefixes, target rows, and token-level
training supervision consumed by the training pipeline. Each contains exactly:

- 496 sampled-route treatment events;
- 496 Source-route preservation events;
- 992 total events;
- 31 optimizer updates at effective event batch size 32.

First freeze exactly 496 broad images under the execution-contract-specific
band quotas and deterministic image-hash order. Then freeze the concentrated
subset under the corresponding execution-contract-specific rule. For v2,
that is the 162-image minimum-prefix cohort above. The subset choice inspects
only distinct trusted pair capacity needed to establish the shortest prefix;
it does not inspect candidate-row score, downstream value, or training result.

Use the same event-ranking rule in both banks. Within an image, prioritize
marginal route-added owners, then total route-added owners, fewer unresolved
rows, lower seed identifier, and earlier trusted Source-anchor order.

Before freezing either bank, run a pair-supply census. The broad arm and the
concentrated arm must first attempt to match their selected-pair histogram by
object-count band and within-image selection rank. Allocate the same
execution-contract-specific pair quota to each band in both arms, then assign
the broad arm's one pair per image so its rank histogram matches the
concentrated arm. This keeps image breadth from silently becoming an
easier-row versus harder-row comparison. If exact ranks are infeasible, try
one predeclared coarsening:
rank 1, rank 2, rank 3, and rank 4 or deeper. If that is also infeasible, the
run may proceed only as a comparison of two data-allocation policies; it must
not claim that physical image breadth alone caused the difference. Record the
feasibility result and the final interpretation before training.

For the broad bank:

1. select exactly one sampled event and one Source event per admitted image;
2. use 496 physical images and no repeated image;
3. keep the sampled and Source families on the same physical-image set;
4. satisfy the accepted band-by-rank matching rule;
5. divide each image's credit equally between the two families.

For the concentrated bank, use the within-band round-robin rule over its
execution-contract-specific nested image cohort until the same per-band pair
quotas and 496 total pairs are selected. Fail if the trusted pair supply is
insufficient. Concentration necessarily gives more event credit to each
image; that is the intended changed factor. No event may be cloned.

The v2 claim boundary is a capacity-constrained and canonical-StateBank-cap-
constrained 496-versus-162 image allocation comparison. It is not the rejected
equal-band 496-versus-118 design, not a natural Common Objects in Context
prevalence comparison, and not a uniform four-times concentration contrast.
Because pair events per image differ materially by band, every treatment
result must report object-count bands separately; an aggregate result cannot
establish a band-uniform image-breadth effect.

Do not increase total row count, family ratio, optimizer updates, or per-image
total credit to make either arm fit. Rescale all event weights independently in
each arm so their mean is exactly 1.0 and their total is exactly 992. Verify
event count, mean weight, and total weight in the assembly receipt. Record
event-rank, route-count, row-depth, and complete-row coordinate-token
supervision distributions so
any remaining difference in event difficulty is visible rather than silently
attributed to image count alone.

## Compared Treatments

### Source checkpoint

Evaluation only.

### Newly matched concentrated 162-image treatment

Train the newly assembled concentrated StateBank. The historical 118-image
checkpoints are context only because their cohort and route-selection rule were
not constructed from the same frozen reservoir.

### Newly matched broad-image treatment

Train the new 992-event StateBank for 31 optimizer updates with the same
learning rate `1e-5`, gradient clipping 1.0, effective event batch size 32,
32-bit floating-point loss computation, Weight-Decomposed Low-Rank Adaptation,
and selected token-type gate. Save steps 10, 20, 30, and 31.

Train both treatment arms under two fixed, matched training seeds. The two arms
use the same seed pair, optimizer schedule, and checkpoint steps. Seed-level
results remain separate; averaging is additional evidence, not permission to
hide a sign disagreement.

No object slot, covered-set carrier, negative unmatched row, terminal
suppression, external detector, Kullback-Leibler divergence, canonical
supervised-fine-tuning mixture, or online refresh is added.

## Evaluation Split and Milestone Choice

Materialize clean greedy inference for Source and every treatment milestone by
executing the frozen full 2,432-image request panel. This preserves the exact
worker partition, image batches, prompt realization, and vLLM numerical path of
the Source run. The development JSONL is an analysis filter over those full
panels; it is not a smaller inference input.

Before comparing milestones, define one fixed complete-case development cohort
containing only images that are Source@B16-eligible in Source and in all sixteen
treatment panels:

```text
two arms x two training seeds x four optimizer steps
```

Use this same fixed image set for every owner, duplicate-candidate, and mean
Average Precision comparison. Report ineligible and invalid counts separately
over all 256 development images. Never turn an ineligible image into a
zero-owner loss.

Select one shared optimizer step for both arms by maximizing the
broad-minus-concentrated matched annotated-owner count at intersection over
union 0.30, averaged across the two matched training seeds. Keep each seed's
contrast visible. Break exact ties by the same averaged contrast at
intersection over union 0.50, then by fewer annotation-anchored duplicate
candidates at intersection over union 0.30 summed across both arms and seeds,
then by the mean standard mean Average Precision across the four arm-seed
panels. Freeze that one shared step before reading the held-out comparison.
Paired image-level uncertainty and object-count-band results inform how strong
the development signal is, but do not replace this mechanical milestone rule.

The primary owner-gain and owner-retention comparison uses the same bounded
`Source@B16` policy. After the shared milestone is frozen, also evaluate the
canonical repetition-penalty-1.10 greedy policy as a decoding-policy
sensitivity check. This secondary policy cannot change route admission,
training events, or milestone selection, and its conclusion must be reported
separately from the fixed-budget neutral-policy conclusion.

Run only the frozen shared step for both arms and both training seeds on the 128
held-out images. The held-out split was fixed before route admission and owns
the transfer conclusion. Also report training-cohort results separately for
the 162 concentrated images, the additional broad-only gradient images, and
the whole broad gradient set. Historical checkpoints remain context and do not
participate in shared-step selection.

Report for every arm:

- unique annotated physical owners found;
- retained, lost, and gained Source owners;
- selected-owner and non-selected-owner recovery;
- category and per-coordinate geometry behavior;
- prediction count, natural closure, duplicate candidates, invalid rows,
  dropped spans, malformed rows, and truncation;
- mean Average Precision and mean Recall as secondary detection summaries.

Unmatched Common Objects in Context predictions are review-needed evidence,
not automatic hallucinations.

## Falsification and Stop Rule

The image-breadth hypothesis is supported only if the broad arm produces a net
owner gain over Source on the pre-admission held-out slice and improves over
the newly matched concentrated arm without an unacceptable geometry or output-
health regression.
The sign should agree at intersection over union 0.30 and 0.50 or any
disagreement must be explained by direct owner and geometry review.

For both broad-over-Source and broad-over-concentrated comparisons, report
paired image-level uncertainty and object-count-band results separately. Both
training seeds must agree in sign for the result to support or reject the
image-breadth hypothesis. A seed disagreement, mixed object-count-band signs,
or an uncertainty interval that does not distinguish the alternatives is an
inconclusive result, not evidence for intrinsic owner exchange.

Stop the positive-only route-imitation treatment family if either condition
holds:

- both seeds and paired uncertainty support a non-positive held-out owner delta
  at both thresholds;
- selected-owner gains remain paired with comparable non-selected or Source-
  owner losses.

If stopped, the next treatment must change the learning information, such as a
same-prefix comparison between a valid uncovered branch and a harmful branch,
or add a compact task-state route. Do not respond by adding more epochs.

If the broad arm yields consistent held-out owner growth with stable output and
geometry, it becomes eligible for a separately designed larger-event or full-
size replication. This unit itself does not authorize that replication.

## Minimal Execution Path

1. Create and verify the 2,432-image label-only pool.
2. Freeze the label-only 2,048/256/128 split before generating or inspecting
   new trajectories.
3. Generate the complete 2,432-image panel with the unified vLLM backend:
   sixteen sampled Source trajectories per image and no greedy trajectory. Do
   not reuse or merge the earlier Hugging Face panel.
4. Validate exact model identity, prompt policy, all sixteen sample indexes per
   image, complete image coverage, natural closure, and zero length
   truncation. Use this panel to estimate sampled owner support and mine
   candidate owner-prefix transitions inside the 2,048-image training split.
5. Qualify `Source@B16` on 64 training-candidate images selected without route
   inspection: sixteen from each object-count band. Freeze the execution policy
   and require exact same-policy replay. Then collect one immutable
   `Source@B16` realization over the complete 2,432-image panel, recording and
   excluding per-image ineligibility without aborting a worker.
6. Compare sampled and Source owner sets at the common first-sixteen-row
   horizon, apply the feasibility condition and v2 prospective allocation
   amendment, then assemble and audit the newly matched concentrated and broad
   992-event StateBanks. Verify the exact 496-versus-162 nested image cohorts,
   band-specific prefix-capacity proofs, and 992 unique events per arm before
   training.
7. Run one real mixed-step smoke, then train both arms for two matched seeds and
   31 updates on eight Graphics Processing Units.
8. Select one shared milestone on the development slice and evaluate only that
   step on the held-out and human-refined slices.
9. Close the treatment family or authorize a larger replication from the
   owner-ledger evidence.

## Non-Goals

- claiming that the previous 256-image signal was random noise;
- increasing optimizer updates together with image count;
- estimating hallucination from incomplete Common Objects in Context labels;
- redesigning the final architecture;
- generalizing the experiment-local assembler before a second real consumer;
- using the held-out slice to choose a checkpoint.

## Artifact Handle

Logical output root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/<run-id>/
```

Every run identifier is immutable. Preserve raw trajectories, admission
receipts, StateBank identity, training receipts, clean rollout artifacts, and
owner-ledger analyses needed to reconstruct the conclusion.
