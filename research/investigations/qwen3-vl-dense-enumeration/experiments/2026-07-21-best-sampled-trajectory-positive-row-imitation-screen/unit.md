---
title: Best Sampled Trajectory Positive Row Imitation Screen
description: A one-treatment screen that tests whether exact-prefix imitation of one verified higher-coverage sampled path can improve native greedy object enumeration.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-21-best-sampled-trajectory-positive-row-imitation-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: executed_bounded_route_level_shift_without_net_coverage_gain
updated: 2026-07-21
---

# Best Sampled Trajectory Positive Row Imitation Screen

> **Closed result:** the full one-epoch screen completed. Every evaluated
> milestone reduced aggregate unique annotated-owner coverage, but a targeted
> follow-up shows a route-level shift toward owners represented by the selected
> sampled routes, offset by losses of other stable owners. Only 118 of the 238
> route-added owners were direct positive-row event targets, so this does not
> establish direct owner-wise imitation. This exact treatment is not promoted
> to 1,024 images; its route-conditioned credit hypothesis is retained for a
> preservation-aware 256-image successor. See
> [results.md](results.md).

## Question

Can positive-only training on exact complete rows from one naturally sampled,
verified, higher-coverage path per image make the source model recover more
unique physical objects under ordinary greedy decoding?

The screen tests one treatment against the frozen source checkpoint. It does
not attempt to isolate every auxiliary contribution. It is promising only when
clean greedy behavior improves, rather than merely lowering replay loss or
making sequences longer.

## Why This Unit Exists

The earlier exact-terminal treatment was too sparse and too late: only three
unique events survived a 256-image census, and its one-event smoke induced a
person repetition burst. A corrected branch pilot then showed that supplying
the sampled history before the current row can make the later target owner
available, whereas changing only an early local branch did not.

The completed 256-image source panel contains one greedy and sixteen
low-temperature sampled trajectories per image. At a fixed budget of sixteen
complete rows, 122 images have at least one parser-accepted, naturally closed
sampled route that is a strict verified-owner superset of greedy without more
confirmed duplicates or malformed rows. Deterministic best-route selection
adds 243 verified owners across those images. This is enough to test whether a
whole sampled path is useful training material; it is not proof that imitation
will transfer to self-generated greedy prefixes.

## Competing Explanations

### Working explanation: a complete useful path supplies missing credit

Sampling sometimes enters a valid route that greedy never visits. Training the
verified rows on that path, under their exact sampled prefixes, may strengthen
the sequence of decisions that reaches additional objects.

### Alternative: replay learning does not survive self rollout

The model may memorize the stored rows at fixed prefixes without reproducing
their path under greedy decoding. This is the primary failure mode because the
preceding coordinate treatment improved exact-prefix margins but not clean
rollout.

### Alternative: treatment only increases continuation

If output length rises while unique verified owners do not, the treatment has
mainly changed a generic continuation tendency rather than object discovery.

### Alternative: sampled geometry contaminates instance identity

A sampled phrase can name a real category while its box mixes neighboring
instances. The initial treatment therefore requires either a unique category
owner in that image or a strong same-category geometric match.

## Training Data

The frozen cohort contains 256 physical training images, one greedy trajectory
per image, and sixteen independent sampled trajectories per image. Route
selection is deterministic:

1. admit only parser-accepted, naturally terminated sampled routes;
2. require a strict verified-owner superset of greedy at the same sixteen-row
   budget;
3. require confirmed duplicate and malformed-row counts to be no worse;
4. maximize added owners, then minimize duplicate change, malformed change,
   unresolved rows, and finally sampled-seed identifier.

For the selected route, keep exact integer token identifiers and exact sampled
prefixes. A row receives gradient only when it is the first occurrence of a
verified physical owner and occurs no later than the last added-owner row.
Rows matched to an owner greedy also finds remain eligible because they form
the sampled history that led to later additions.

An eligible row must also satisfy one of these physical-owner trust rules:

- its category has exactly one labeled owner in the image; or
- its same-category match has intersection over union at least `0.75`.

Duplicates, malformed rows, unresolved rows, rows after the last added owner,
terminal tokens, and all historical prefix tokens receive zero gradient.
Unresolved prefix rows remain exact context only. Unmatched predictions are not
treated as hallucinations because Common Objects in Context annotations are
incomplete.

## Compared Models

### Source checkpoint

Evaluation only. The source is the geometry-sorted, description-first,
pure-cross-entropy checkpoint at step 4,887.

### Positive path imitation treatment

For admitted image `i` with `n_i` eligible rows, let `Y_i,j` be an exact row and
`P_i,j` its actual sampled prefix. Give every row in that image weight
`w_i,j = 1 / n_i`, then rescale all weights to mean one across the training
bank. This makes each admitted image contribute equal total weight regardless
of route length.

The treatment loss is:

```text
L_row(i,j) = w_i,j * mean_nonempty(
  mean negative log probability over schema-and-description sites,
  mean negative log probability over trusted coordinate sites
)
```

The two token groups are averaged separately before their non-empty group means
are combined. This prevents a longer description or a fixed number of
coordinate tokens from silently dominating the event. A small token-type gate
is applied at the same selected sites as a language-format stabilizer; it is
not a scientific comparison arm. There is no negative branch, terminal loss,
canonical supervised-fine-tuning mixture, or trajectory-quality multiplier.

## Model and Optimization Scope

- Freeze the vision tower and multimodal aligner.
- Train only the language-tower Weight-Decomposed Low-Rank Adaptation (`DoRA`)
  payload. Keep selected-token embedding deltas frozen.
- Use a token-type gate only as a stability constraint on selected row sites;
  it is not an independent scientific arm.
- Use learning rate `1e-5`, gradient clipping at `1.0`, one epoch, eight
  Graphics Processing Units, and effective event batch size `32`.
- Make the bank length exactly divisible by `32` through a recorded,
  deterministic, minimum-loss image exclusion. The expected bank is 512 events
  from 118 images, which produces sixteen optimizer steps.
- Save checkpoints at steps 5, 10, 15, and the final step 16. The final
  checkpoint is the primary evaluation target.
- Do not add an object slot, detector, coverage ledger, external teacher,
  Kullback-Leibler divergence penalty, or inference-time controller.
- Reuse the current exact-prefix `StateBank` event store, meaning the
  repository's frozen bank of token-exact rollout prefixes and candidate rows,
  together with its replay, packing, compact-logit, and streaming training
  path. Add only the missing positive-only weighted row-loss behavior and
  experiment-local data assembler.

## Primary Observation

The primary comparison is final treatment minus the frozen source checkpoint
on ordinary clean greedy rollout, not training loss and not fixed-prefix replay
alone.

The planned evidence order was:

1. Exact-prefix replay would confirm that selected complete-row likelihood
   moves in the intended direction with finite gradients.
2. Clean greedy rollout on the 256 training images tests whether the learned
   preference survives self-generated prefixes.
3. The twelve human-refined development images test behavioral transfer and
   safety without entering training or model selection.
4. A broader validation panel is optional for this first screen and is run only
   if the preceding evidence is promising.

The completed screen has real finite-gradient smoke and clean-rollout evidence,
but no full-bank pre/post exact-prefix likelihood receipt. Its final
interpretation therefore relies on clean behavioral transfer and does not claim
that the entire bank was independently verified by fixed-prefix scoring.

Report unique matched physical owners, confirmed duplicates, malformed and
dropped rows, native termination, row count, entity discovery, category
retention, full-box intersection over union, center error, and box-size error.
Entity discovery and geometry quality remain separate measurements.

## Interpretation and Promotion

- **Clean greedy unique-owner coverage improves without obvious safety loss:**
  the treatment is promising; prepare a 1,024-image replication.
- **Exact-prefix likelihood improves but clean rollout does not:** the stored
  row objective moves locally but does not transfer through the model's own
  trajectory.
- **Treatment increases rows but not unique owners:** the objective mainly
  strengthens generic continuation.
- **Treatment improves entity discovery while geometry worsens:** preserve the
  route treatment as a selection result, but do not claim complete detection
  improvement; geometry needs a separate coherent-box treatment.
- **The treatment does not improve:** do not scale self-imitation; return to local
  remaining-object completion or an explicit compact task-state intervention.

## Stop Rule

There is no fixed numerical promotion margin for this exploratory screen. Scale
to 1,024 images only when the combined evidence is promising: unique physical
owner recall, false-negative rate, F1 score, or mean Average Precision improves
enough to justify a larger test, while confirmed duplicates, malformed output,
unsupported entities, truncation, repetition bursts, and trusted geometry show
no obvious damaging trend. A training-loss decrease alone is not a promotion
signal.

## Minimal Execution Path

1. Freeze and validate the completed greedy-plus-sixteen-sampled panel.
2. Select one best admitted route per image and assemble exact-prefix positive
   row events with deterministic receipts.
3. Audit enlarged crops for a compact set of high-gain, crowded, unresolved,
   and near-threshold cases.
4. Run a one-event, one-step real-model smoke and verify exact replay, selected
   sites, finite gradients, checkpoint loading, and ordinary inference.
5. Train the 512-event bank for one full eight-GPU epoch and retain steps 5,
   10, 15, and 16.
6. Evaluate Source and final treatment with identical greedy inference.
7. Decide whether the evidence justifies a 1,024-image replication.

## Non-Goals

- selecting a final architecture;
- proving that every sampled route is better than greedy;
- recovering the full union of all sampled objects with one trajectory;
- isolating the independent effect of the token-type gate;
- comparing continuous trajectory-quality weighting with a shuffled control;
- estimating population-level effect size from the twelve development images;
- treating unmatched predictions as hallucinations;
- full-dataset training before the 256-image screen closes.

## Artifact Handle

Planned logical root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/<run-id>/
```

The frozen source collection is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-earliest-shared-prefix-branch-and-trajectory-treatment/
full-trajectory-k16-source-256/
```

Implementation and one-epoch training completed. The immutable bank contains
512 events from 118 images. The one-event smoke and all sixteen full-training
updates were finite; checkpoints were saved at steps 5, 10, 15, and 16.
Identical clean greedy evaluation on train-256 and twelve human-refined images
found lower unique annotated-owner coverage at every evaluated milestone.
Step 10 and step 15 sometimes improved official mean Average Precision through
shorter, cleaner output and slightly tighter retained boxes. A subsequent
owner-identity intersection shows that the checkpoints recover more owners
added by the selected sampled routes, but lose other ordinary owners at nearly
the same rate. Step 15 leaves total annotated-owner coverage unchanged on the
118 admitted images while the 138 non-admitted images regress. The final step
16 also produces a large regression on the twelve-image panel. The identical
1,024-image replication is canceled; a matched-arm, preservation-aware
256-image successor is the recommended next treatment. Complete evidence and
interpretation are in
[results.md](results.md).
