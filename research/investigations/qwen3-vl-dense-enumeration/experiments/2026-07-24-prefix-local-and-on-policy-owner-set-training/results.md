---
title: Prefix-Local and On-Policy Final Owner-Set Training Results
description: Materialization, completed long training, matched-batch milestone and transfer evaluation, protocol deviations, and bounded clean-greedy owner-set evidence.
type: investigation-result
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-24-prefix-local-and-on-policy-owner-set-training
topic: qwen3-vl-dense-enumeration
status: transfer_evaluation_ready_for_user_discussion
evidence_status: directional_heldout_net_gain_not_robustly_established
updated: 2026-07-25
---

# Prefix-Local and On-Policy Final Owner-Set Training Results

## Current Verdict

All four 1,440-event long runs completed 90 finite applied optimizer updates.
Their 64-image training-panel dose curves show that positive net-owner movement
is real but that the complete-action and grouped objectives can turn set
expansion into unbounded output growth. Under the matched final policy
(`batch_size=4`, `max_new_tokens=3084`, greedy decoding, repetition penalty
`1.0`), Source naturally stops on all 64 images and matches 389 owners.

The only evaluated treatment with zero 3084-token safety-cap stops is the
first-divergence transition control at step 36. It gains 21 Source-missed
owners, loses 18 Source owners, and therefore has a small net gain of three
owners while producing 155 fewer predictions and 14 fewer strict duplicate
candidates. This is a bounded favorable result, not strong set expansion. The
other promoted or final checkpoints have one to ten safety-cap stops. Their
larger apparent owner gains are accompanied by hundreds or thousands of extra
predictions and cannot be promoted as usable improvement.

The matched transfer gate strengthens this result without turning it into a
robust promotion. On development-256, transition step 36 gains 128 owners,
loses 60, and is `+68` net. On disjoint heldout-128, it gains 46, loses 39, and
is `+7` net. The held-out direction is favorable and is achieved with 118
fewer predictions and 13 fewer strict duplicate candidates, but only 21 images
are net positive versus 17 net negative and 90 unchanged. It also introduces
one 3084-token length stop where held-out Source has none. This is promising
directional transfer, not a robust usable-improvement claim.

The result still does not establish objective superiority, a genuine
on-policy estimator, a full-pool gain, or a final architecture. The approved
round is ready for user discussion before any broader evaluation or successor
direction is launched.

## Materialized Event Supply

The assembler replays the receipt-bound sampled trajectories, retains an exact
model-produced non-empty prefix, and pairs one trusted clean owner-expansion
row with a constructed one-token premature stop action. The stop is an audited
counterfactual action, not a claim that a natural sampled stop was observed.

The full replay found:

- 46,477 positive groups across 1,814 images;
- 44,576 non-empty-prefix groups across 1,463 images;
- 69,306 positive row occurrences in 32,064 trajectories; and
- 1,901 row-zero groups excluded only because StateBank version 1 cannot encode
  an entity transition from an `empty` root state. This is a storage boundary,
  not a scientific event predicate.

The two long-run banks each select one deterministic event from 1,440 different
images. Prefix depth is 1 for 1,353 events, 2 for 79, 3 for 7, and 4 for 1.

| Bank | Events | Selected positive aliases | StateBank identity |
|---|---:|---:|---|
| singleton complete-action pairwise | 1,440 | 1,440 | `5951466193dfdb52b2c517b2b80c622302a6f2f06cd50d13f4d965dbe8ba5934` |
| grouped owner-conditioned | 1,440 | 1,749 | `6bf8e839ac345835fc10891493327a6185c80e655aeac0e5db83c7c312a2a6bd` |

The grouped bank caps aliases at two and normalizes within the positive owner
group. Its additional aliases therefore change the audited candidate aggregate
without silently increasing event or image weight.

## Implemented Objective Path

The training path now has two explicit profiles sharing one complete-action
scorer:

- `complete_action_pairwise`: summed autoregressive log likelihood for one
  positive complete row versus one harmful complete action, optimized with
  `softplus(score_harmful - score_positive)`;
- `owner_conditioned_candidate`: stable weighted `logsumexp` over all selected
  aliases for one physical owner versus the harmful group, followed by the same
  logistic comparison.

Candidate identifiers and token sequences are deduplicated, grouped positives
must bind one owner, pairwise records must contain exactly one positive, and
all score aggregation is performed in 32-bit floating point. The existing
`transition_only` first-divergence profile is retained as a distinct control;
it is not renamed as complete-action pairwise training.

Targeted config, loss, and trainer-integration verification completed with
`85 passed`. All four long configs also resolve successfully with frozen
fingerprints before launch.

## Short Clean-Greedy Gate

The decision panel contains 64 training images bound to the materialized event
bank. Every arm uses deterministic Hugging Face greedy decoding, repetition
penalty `1.0`, the original prompt, one autoregressive completion, and the
existing row schema. Source repeated exactly: 386 matched owners, 636
predictions, and zero owner, prediction, duplicate, or geometry delta.

The table compares each checkpoint with that frozen Source. `Owner + / -`
means treatment-only owners and Source-only owners. `Net` is their difference.
`Prediction delta` is not treated as owner gain. `Strict duplicate delta` uses
the strict physical-owner duplicate candidate counter.

| Objective and checkpoint | Owner + / - | Net | Prediction delta | Strict duplicate delta | Common-owner mean Intersection over Union delta |
|---|---:|---:|---:|---:|---:|
| complete-action pairwise, learning rate `1e-5`, step 1 | 19 / 6 | +13 | +17 | -10 | -0.000510 |
| complete-action pairwise, learning rate `1e-5`, step 4 | 22 / 6 | +16 | +19 | 0 | +0.000389 |
| complete-action pairwise, learning rate `3e-6`, step 1 | 11 / 5 | +6 | -2 | -1 | -0.002050 |
| complete-action pairwise, learning rate `3e-6`, step 4 | 14 / 11 | +3 | -14 | -7 | -0.000177 |
| first-divergence transition control, learning rate `1e-5`, step 1 | 17 / 12 | +5 | -30 | -24 | -0.000499 |
| first-divergence transition control, learning rate `1e-5`, step 4 | 24 / 11 | +13 | -2 | -1 | -0.000583 |
| grouped owner-conditioned, learning rate `1e-5`, step 1 | 22 / 6 | +16 | +14 | +1 | -0.000833 |
| grouped owner-conditioned, learning rate `1e-5`, step 4 | 22 / 7 | +15 | +18 | -9 | -0.001094 |

The gains exceed the corresponding lost-owner counts, so they are not pure
owner exchange. Dose behavior is non-monotonic: the lower-rate pairwise arm
falls from `+6` to `+3`, while the transition control rises from `+5` to `+13`.
Milestone evaluation is therefore required; the last checkpoint must not be
assumed best. The pairwise `1e-5` step-4 result also contains one pre-existing
pathological repeated-owner image whose strict duplicate count rises by four,
offset elsewhere in the aggregate. This is a watch case, not evidence of a
panel-wide duplication burst.

The trained event owner itself was already found by Source in 62 of 64 panel
images, so most aggregate gain is shared-parameter transfer to other owners,
not direct rescue of the selected owner. That is a useful final-set observation
but weakens any narrow claim that the loss acts only on its named owner.

## Protocol Deviation and Unfinished Arms

The planned zero-update influence matrix was not executed before the short
optimizer updates. Runtime records do establish finite nonzero gradients,
correct immediate target-versus-stop margin movement, and applied updates; the
one-step and four-step clean-greedy evaluations provide a stronger direct
behavioral test of net owner gain. They do not reconstruct the planned
first-order score changes for every Source, non-target, duplicate, invalid, and
unresolved candidate.

The long launch therefore uses an explicitly adapted exploratory gate, not a
claim that the original zero-update requirement was literally satisfied. This
deviation limits mechanistic attribution and must remain visible in the final
comparison. It does not erase the observed one-completion final-owner gains.

Genuine on-policy and hybrid arms are not launched. The current repository has
frozen-StateBank scoring and ordinary inference, but no verified loop that
generates with the current trainable parameters, differentiably rescores the
same sampling policy, binds the exact adapter fingerprint, computes trusted
final-set utility, and refreshes after each accepted update. Reusing the frozen
candidate bank would not satisfy the defined estimator. Matched ordinary row
cross-entropy and Source-preservation-only controls are also still pending, so
no comparative objective conclusion is yet warranted.

## Completed Long-Training Receipt

Each run used 1,440 events, effective batch size 16, 90 optimizer updates, ten-
percent milestone checkpoints, one A100, and an isolated collision-fail
artifact root. Every run has 90 finite step records; the last record reports
`optimizer_update_status=applied`, and `checkpoints/final.json` resolves to
step 90. Adapter and special-token embedding payloads exist at steps 9 through
90 in increments of nine.

| Arm | Config fingerprint | Final update | Final checkpoint |
|---|---|---:|---:|
| complete-action pairwise, learning rate `1e-5` | `4945f95044ebf78caf97554a8b52c389c2442df0be7f6eabcb997ba32f30bbdd` | 90 | 90 |
| complete-action pairwise, learning rate `3e-6` | `6d18e2c33568e38bc539d1ebc3f1be4e99ab16a690a344fdc774cb8587292960` | 90 | 90 |
| first-divergence transition, learning rate `1e-5` | `0a6ef72c91e30ed53349addb84f0cb1351facf2d79d5478abab1ecb70bc36395` | 90 | 90 |
| grouped owner-conditioned, learning rate `1e-5` | `793c3ffdf655350340e43898359476c26af8f0e74cbfd81adfd1ad1904ca2f50` | 90 | 90 |

The original durable workers all exited with status zero. The run logs,
checkpoints, and receipts are under the experiment artifact root stated below.

## Milestone Dose Curve at 512 New Tokens

All 40 milestone runs completed on the frozen 64-image panel. This bounded
screen is useful for detecting expansion pressure, but a 512-token stop is not
automatically a final failure: Source has one 512-token stop that naturally
closes when allowed the established 3084-token horizon.

The dose curve nevertheless exposes a clear difference between objectives.
The complete-action pairwise `1e-5` arm grows from two length stops at step 9
to thirteen at step 90. Grouped owner-conditioned training grows from three to
twelve. The `3e-6` pairwise arm stays between one and four stops, while the
transition control retains exactly the single short-horizon Source stop at
every milestone. The complete table and per-run owner ledgers are rooted at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-24-prefix-local-and-on-policy-owner-set-training/
owner-comparisons-64-long-milestones-v1/
```

## Matched Final-Horizon Comparison

The conclusion-bearing comparison freezes `batch_size=4`,
`max_new_tokens=3084`, deterministic greedy decoding, repetition penalty
`1.0`, one completion, the original prompt, and the canonical row schema for
both Source and treatment. Source naturally stops on all 64 images, matches
389 owners, and owns the comparison baseline below.

| Objective and checkpoint | Owner + / - | Net | Prediction delta | Strict duplicate delta | Length stops |
|---|---:|---:|---:|---:|---:|
| grouped owner-conditioned `1e-5`, step 9 | 22 / 11 | +11 | +169 | +8 | 1 |
| grouped owner-conditioned `1e-5`, step 90 | 35 / 16 | +19 | +2,449 | +7 | 10 |
| complete-action pairwise `1e-5`, step 9 | 24 / 10 | +14 | +467 | +15 | 1 |
| complete-action pairwise `1e-5`, step 90 | 36 / 9 | +27 | +2,486 | +4 | 9 |
| complete-action pairwise `3e-6`, step 27 | 10 / 9 | +1 | +487 | -6 | 2 |
| complete-action pairwise `3e-6`, step 54 | 24 / 6 | +18 | +557 | +10 | 2 |
| complete-action pairwise `3e-6`, step 90 | 24 / 8 | +16 | +610 | +2 | 2 |
| first-divergence transition `1e-5`, step 36 | 21 / 18 | +3 | -155 | -14 | 0 |
| first-divergence transition `1e-5`, step 90 | 23 / 21 | +2 | +103 | -16 | 1 |

No run has parser, score, or image-validation failures. Only transition step 36
meets the Source-matched zero-length-stop validity target. Its gain is not
explained by longer output and is accompanied by fewer strict duplicate
candidates, but it still exchanges 18 Source owners for 21 new owners. The net
gain of three on training images is too small and too in-sample to establish a
usable model improvement.

The authoritative matched table and per-image ledgers are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-24-prefix-local-and-on-policy-owner-set-training/
owner-comparisons-64-long-promotion-max3084-matched-b4-v3/
```

## Matched Development and Held-Out Transfer

The user approved a narrow transfer gate for Source and first-divergence
transition step 36 only. Development and heldout are disjoint by example ID and
contain 256 and 128 images. All four runs use the original prompt, one
completion, `batch_size=4`, `max_new_tokens=3084`, greedy decoding, repetition
penalty `1.0`, scoring enabled, and the same eight-rank controller-worker
execution. Within each split, Source and treatment have the same shard-plan
fingerprint and rank-to-device mapping. Their only intended model-payload
difference is the transition step-36 adapter replacing the inherited
production step-4887 Source adapter; the special-token embedding payload is
content-identical by hash.

All 768 run rows decode and score successfully. No run has a parser, score, or
image-validation failure.

| Split | Source / treatment owners | Owner + / - | Net | Net-positive / net-negative / unchanged images | Prediction delta | Strict duplicate delta | Common-owner mean Intersection over Union delta | Source / treatment length stops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development-256 | 1,452 / 1,520 | 128 / 60 | +68 | 60 / 21 / 175 | +350 | -14 | +0.002068 | 1 / 3 |
| heldout-128 | 729 / 736 | 46 / 39 | +7 | 21 / 17 / 90 | -118 | -13 | +0.006447 | 0 / 1 |

Development owner coverage rises from `55.50%` to `58.10%`, a 2.60 percentage-
point increase. Held-out owner coverage rises from `55.95%` to `56.49%`, a
0.54 percentage-point increase. The held-out result is important because the
owner gain coexists with fewer total predictions, fewer strict duplicate
candidates, no invalid-prediction increase, and slightly better common-owner
geometry. It is therefore not simple verbosity or duplicate expansion.

The held-out magnitude remains uncertain: the net is only seven owners across
1,303 annotated owners, positive and negative images are close at 21 versus
17, and one treatment row reaches the fixed length horizon. In 100,000
post-hoc paired image bootstrap resamples with NumPy seeds `20260725` and
`20260726`, development mean owner delta is `+0.2656` with a percentile
interval of `[+0.1484, +0.4023]`, while held-out mean is `+0.0547` with
`[-0.0781, +0.1875]`. This diagnostic was not a
pre-registered acceptance threshold and is used only to bound the claim. The
defensible conclusion is directional transfer with a favorable quality
profile, not a statistically or operationally robust promotion.

The receipt-bound configs, complete run artifacts, and authoritative paired
ledgers are:

```text
configs/coordexp_swift/infer/research/
qwen3_vl_2b_transition_step36_transfer_max3084_matched_b4_v1/

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-24-prefix-local-and-on-policy-owner-set-training/
clean-rollouts-transition-step36-transfer-max3084-matched-b4-v1/

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-24-prefix-local-and-on-policy-owner-set-training/
owner-comparisons-transition-step36-transfer-max3084-matched-b4-v2/
```

## Evaluation Runtime Corrections

An initial final-horizon table compared Source decoded at batch size 16 with
treatments decoded at batch size 4. A real same-checkpoint replay found that
57 of 64 raw greedy decodes differed across those batch policies, although
their aggregate stop counts agreed. The mixed-batch table under
`owner-comparisons-64-long-promotion-max3084-b4-v2/` is therefore rejected as
conclusion evidence and retained only as provenance.

Batch size 16 had first been abandoned because Hugging Face
`compute_transition_scores` stacked every generated step before fp32
log-softmax. A 1784-step batch requested an additional 16.23 GiB and failed;
another run reached 3084 steps before the same failure. Policy likelihood
extraction now invokes the same Transformers method in bounded 32-step chunks.
A focused red-to-green regression and all 12 current `HFBackendSession` tests
pass. A real batch-16 replay of the prior 3084-step failure completed with 64
rows, two expected model length stops, no artifact failures, peak allocated
memory 43,988,510,208 bytes, and peak reserved memory 69,128,421,376 bytes.
This fixes the runtime OOM but does not make batch sizes behaviorally
interchangeable. The matched result above therefore remains frozen at batch
size 4.

`tests/inference/test_backend_trace.py` is a stale legacy test surface: it
imports the removed `HFGenerateBackend` and constructs the pre-current
`TokenTrace(logprob=...)` interface. Its 34 failures predate and do not exercise
the current `HFBackendSession` correction; they were not broadened into this
research fix.

## Stop Boundary

The approved long training, matched training-panel comparison, and narrow
development/held-out transition-step-36 transfer gate are complete. Do not
start full-pool, successor-direction, or final-architecture work before
discussing this evidence with the user. The unresolved question is no longer
whether the direction can transfer at all; it is whether the small and
uncertain held-out gain is strong enough to justify a broader confirmation or
a change to the research mechanism.
