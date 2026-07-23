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
status: running
evidence_status: partial
updated: 2026-07-23
---

# Constant-Dose Image-Breadth Treatment Screen

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

Development and held-out trajectories may be materialized for later
evaluation, but no admission statistic from those images may affect either
training bank.

The twelve human-refined validation images remain development safety evidence
only. They never supply gradients and are not part of the 2,432-image pool.

## Route Admission and Matched Event Selection

Apply the same conservative route admission semantics as the completed screen:

- parser-accepted and naturally closed route;
- verified annotated physical-owner set strictly expands the Source greedy
  owner set;
- no increase in confirmed duplicates or malformed rows;
- only trusted first-occurrence physical owners receive gradient;
- entity identity and coordinate trust remain separate;
- unresolved or annotation-ambiguous rows receive no gradient.

The sampled-only production panel does not contain the source greedy owner set
referenced by the first admission rule. Before any StateBank is frozen, define
a separate finite source-baseline decoding policy or revise the admission rule
and its claims. Do not silently substitute a length-truncated greedy loop, a
greedy run with a different repetition penalty, or an arbitrary sampled route.
Until this decision is recorded, trajectory collection is complete but
StateBank assembly remains pending.

Run route admission only inside the frozen 2,048-image training-candidate
split.
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

Build two newly matched `StateBank` artifacts. Here, a StateBank is the
persisted collection of model states, prefixes, target rows, and token-level
training supervision consumed by the training pipeline. Each contains exactly:

- 496 sampled-route treatment events;
- 496 Source-route preservation events;
- 992 total events;
- 31 optimizer updates at effective event batch size 32.

First freeze exactly 496 broad images in deterministic object-count-band and
image-hash order. Then freeze a 118-image concentrated subset of the broad set
with the same band balance.
The subset choice does not inspect candidate-row score or downstream value.

Use the same event-ranking rule in both banks. Within an image, prioritize
marginal route-added owners, then total route-added owners, fewer unresolved
rows, lower seed identifier, and earlier trusted Source-anchor order.

Before freezing either bank, run a pair-supply census. The broad arm and the
concentrated arm must first attempt to match their selected-pair histogram by
object-count band and within-image selection rank. Allocate the same number of
pairs to each object-count band in both arms, then assign the broad arm's one
pair per image so its rank histogram matches the concentrated arm. This keeps
image breadth from silently becoming an easier-row versus harder-row
comparison. If exact ranks are infeasible, try one predeclared coarsening:
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

For the concentrated bank, use a within-band round-robin rule over its 118
images until the same per-band pair quotas and 496 total pairs are selected.
Fail if the trusted pair supply is insufficient. Concentration necessarily
gives more event credit to each image; that is the intended changed factor. No
event may be cloned.

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

### Newly matched concentrated 118-image treatment

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

Run clean greedy inference for Source and every treatment milestone on the 256
development images. Select one shared optimizer step for both arms by
maximizing the broad-minus-concentrated unique-owner difference at intersection
over union 0.30, averaged across the two matched training seeds. Break ties by
the same contrast at intersection over union 0.50, then fewer confirmed
duplicate candidates, then standard mean Average Precision. Freeze that one
shared step before reading the held-out comparison.

Run only the frozen shared step for both arms and both training seeds on the 128
held-out images. The held-out split was fixed before route admission and owns
the transfer conclusion. Also report training-cohort results separately for
the 118 concentrated images, the additional broad-only gradient images, and
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
3. Derive the 2,176 images not covered by the old panel, then generate one
   greedy and sixteen sampled Source trajectories over eight Graphics
   Processing Units.
4. Merge the old and new request-scoped trajectory panels, validate exact
   model identity, prompt policy, seed coverage, and image coverage, then run
   admission only inside the 2,048-image training candidates.
5. Apply the feasibility condition, then assemble and audit the newly matched
   concentrated and broad 992-event StateBanks.
6. Run one real mixed-step smoke, then train both arms for two matched seeds and
   31 updates on eight Graphics Processing Units.
7. Select one shared milestone on the development slice and evaluate only that
   step on the held-out and human-refined slices.
8. Close the treatment family or authorize a larger replication from the
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
