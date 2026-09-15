---
title: Sampled Object-Span Likelihood and Consensus Filtering of the K=16 Union
description: Pre-registered teacher-forced likelihood replay of every parsed object span in the Sorted, Random, and Permutation K=16 sampled rollouts, testing whether internal confidence and cross-trajectory consensus can reject badly grounded union clusters while retaining the owners that only sampling recovers.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-sampled-span-likelihood-and-consensus-union-filtering
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Sampled Object-Span Likelihood and Consensus Filtering of the K=16 Union

This unit is pre-registered and not yet executed. Every threshold, label
semantic, and stop rule below was fixed with the research lead **before** any
likelihood replay ran. The alignment record is the point of this document; a
threshold chosen after seeing the likelihood distributions would not be a
pre-registration.

The research lead authorized execution on 2026-07-29, after the protocol above
was frozen. Stage 0 and Stage 1 are in flight; `evidence_status` stays `none`
until their receipts land and are checked. The conclusion-bearing chain —
replay, span slicing, medoid aggregation, retention curve — had never been
exercised on a real case at the time of authorization, so Stage 1 doubles as the
representative smoke that the unit contract requires.

Originating transport is
`handoff/2026-07-29-three-checkpoint-sampled-span-likelihood-mining.md`. That
handoff is provenance only. This unit owns the protocol.

## Terminology

Declared once, used throughout.

- **Sorted checkpoint**: the geometry-sorted row-order training arm,
  `qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32` at step 4887.
- **Random checkpoint**: the random row-order training arm at the same step.
- **Permutation checkpoint**: the random-permutation-bundle training arm at the
  same step.
- **Object span**: one parsed
  `<|object_ref_start|>…<|object_ref_end|><|box_start|>…<|box_end|>` row inside
  one generated trajectory, carrying a stable `object_span_id`.
- **Raw-model likelihood**: FP32 `log_softmax` over unmodified language-model
  head logits at the chosen token, computed by one teacher-forced full forward.
- **Policy likelihood**: the same quantity after the executed generation
  processors (temperature 0.4, top-p 0.95). Deferred in this unit; never
  aliased onto raw-model likelihood.
- **Cluster**: one class-aware complete-link IoU≥0.50 group of object spans
  within one image, produced by the canonical union matcher.
- **Medoid**: the cluster member maximizing summed IoU to all other members.
  The canonical union prediction is the medoid's box, so the medoid is the
  object the filter would actually accept or reject.
- **Support**: the number of distinct sampling seeds (out of 16) contributing at
  least one member span to a cluster.
- **Per-span independent best-IoU label**: a label assigned to each span from
  its own maximum same-class ground-truth IoU, independent of the canonical
  one-to-one assignment. Duplicate correct spans all count as true positives,
  so per-span totals do not sum to the canonical union true-positive count.
- **Cluster-inherited label**: the canonical union verdict of a span's cluster,
  propagated to every member. Totals reconcile with canonical metrics, but a
  geometrically correct span inherits a false-positive label when its cluster
  lost the greedy assignment.
- **Catastrophic false positive**: an unmatched cluster whose maximum same-class
  ground-truth IoU is below 0.10. Split into **class-absent** (the image has no
  ground-truth object of that class at all, so the maximum is vacuously zero)
  and **mis-grounded** (the class exists in the image but the box is placed
  elsewhere).
- **Ordinary false positive**: an unmatched cluster with maximum same-class IoU
  at or above 0.10. Split into **duplicate** (maximum IoU ≥ 0.50 against a
  ground-truth owner already claimed by another cluster) and **loose**
  (0.10 ≤ maximum IoU < 0.50).
- **Greedy-missed union-recovered owner**: a ground-truth object matched by the
  K=16 sampled union but not by the checkpoint's native greedy decode. This set
  is the entire incremental value of the union operation over greedy.
- **Leave-one-image-out (LOIO) cross-validation**: threshold selected on 11 of
  the 12 panel images and evaluated on the held-out image; the 12 held-out
  evaluations are pooled into one out-of-sample estimate.

## Decision and Decision-Owning Outcome

The decision at stake is whether a cluster-level confidence post-operation on
the K=16 sampled union is worth building at all, and if so whether its
controlling signal is internal likelihood or cross-trajectory consensus.

The outcome is judged on **union clusters**, because that is the unit a filter
would accept or reject. Exactly one of:

1. **Likelihood-filterable tail** — catastrophic false positives concentrate at
   low description or coordinate likelihood; a confidence post-operation is
   promising.
2. **Consensus-only tail** — bad clusters are not low likelihood but have low
   support or high spatial dispersion; likelihood alone is insufficient.
3. **Confident binding failure** — badly grounded clusters carry high likelihood
   *and* high support; internal confidence cannot verify grounding truth, and
   the route turns to an image-conditioned verifier, candidate ranker, or
   explicit geometry check.

## Primary Question

Within each checkpoint, does raw-model likelihood over an object span's
coordinate and description tokens separate true-positive clusters from
catastrophic false-positive clusters strongly enough to reject a useful share of
the catastrophic tail while retaining the greedy-missed union-recovered owners?

## Strongest Alternative

The model is confidently wrong. Coordinate tokens for a badly grounded box are
emitted with likelihood comparable to a correct box, because the language model
is confident in *emitting a well-formed row*, not in *having grounded it*.

The control that separates this from the working hypothesis is the schema
wrapper component. Wrapper tokens (`<|object_ref_start|>`, `<|object_ref_end|>`,
`<|box_start|>`, `<|box_end|>`) carry grammar confidence only. **If wrapper
likelihood separates true from catastrophic clusters but coordinate likelihood
does not, the result is rejected as grammar confidence, not grounding
confidence**, regardless of headline AUROC.

## Frozen Panel, Checkpoints, and Source Artifacts

Panel: the 12 human-refined images in
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl`,
carrying 346 ground-truth owners.

Source rollout root (read-only; this unit must not mutate it):

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084`

Per checkpoint: `<root>/{sorted,random,permutation}/sampled/shard-{0,1}.json`,
`<root>/{sorted,random,permutation}/f1-metrics.json`, and the native greedy
manifest `<root>/visualization/{sorted,random,permutation}/manifest.json`.

The sampled shards record the executed sampling policy — temperature `0.4`,
top-p `0.95`, repetition penalty `1.0`, seeds `21001`–`21016`, max new tokens
`3084` — which differs from the infer configs' greedy blocks (temperature `0.0`,
repetition penalty `1.10`). The configs' generation block must not be used to
describe these rollouts.

Each shard records `infer_config_path` under the
`permutation-bundle-coordinate-noise-pilot` worktree, not this checkout. That
origin worktree is still on disk, and its resolved configs reproduce the
recorded fingerprints `d44ab5a6…` (Sorted), `ed8d677f…` (Random), and
`1a862866…` (Permutation) exactly. This checkout's configs resolve to different
fingerprints, but the difference is confined to `run.artifact_root`; see the
semantic-delta gate.

Replay volume: 576 full-sequence teacher-forced forwards (3 checkpoints × 12
images × 16 seeds) covering about 9,340 object spans, at most 1,372 prompt plus
396 generated tokens each. Each sequence is replayed exactly once; likelihood is
sliced per span afterwards. One forward per object row is forbidden.

## Pre-Registration Baseline

Derived read-only from the existing artifacts above **before** the protocol was
frozen. These are inputs to the pre-registration, not evidence produced by this
unit. Because these numbers define the stop rule, they must stay re-derivable:
`scripts/research/derive_likelihood_mining_baseline.py` reproduces every value
in this section and exits non-zero if any gate fails. Its own gates are that the
canonical recomputation is byte-identical, the declared aggregates match, and
greedy ground-truth indexing aligns.

Canonical union aggregates were reproduced from the shards by re-running
`scripts/research/compute_sampled_union_f1_metrics.py::compute`. All three
checkpoints reproduced their stored `f1-metrics.json` **byte-identically**,
which establishes that cluster membership and matched ground-truth identity —
absent from the stored `cluster_receipts` — are fully reconstructible.

| Checkpoint | Clusters | Union TP | Union FP | Union FN | Greedy TP |
|---|---:|---:|---:|---:|---:|
| Sorted | 667 | 191 | 476 | 155 | 107 |
| Random | 593 | 162 | 431 | 184 | 132 |
| Permutation | 592 | 169 | 423 | 177 | 121 |

False-positive taxonomy and owner recovery:

| Checkpoint | Catastrophic class-absent | Catastrophic mis-grounded | Ordinary duplicate | Ordinary loose | Catastrophic total (share of FP) | Greedy-missed union-recovered owners |
|---|---:|---:|---:|---:|---:|---:|
| Sorted | 13 | 184 | 65 | 214 | 197 (41%) | 88 |
| Random | 4 | 127 | 70 | 230 | 131 (30%) | 44 |
| Permutation | 2 | 136 | 67 | 218 | 138 (33%) | 60 |

Two consequences were used in freezing the protocol. First, the class-absent
subclass is small (13/4/2), so the catastrophic tail is not an artifact of
vacuous maxima; it is dominated by genuine mis-grounding. Second, the greedy
ground-truth indices in the visualization manifests align exactly with the COCO
annotation ordering used by the union matcher (0 of 346 positional mismatches
per checkpoint, and matched-pair counts equal the greedy true-positive counts),
so greedy-missed union-recovered owners are a lookup rather than a re-match.

All descriptions map to known COCO category names
(`unknown_category_prediction_cluster_count = 0` for all three), so the
same-class predicate underlying the catastrophic definition is well-formed.

## Protocol

### Stage 0 — contract gates, before any scoring

1. Recompute the SHA-256 of every stored `prompt_token_ids` and
   `generated_token_ids` and compare against the recorded digests.
2. Rebuild native prompt inputs through
   `_materialize_native_inputs` and require executed prompt-token parity, image
   content identity, and the recorded `observed_image_grid_thw`.
3. Verify model identity against the configs that produced the rollouts. Every
   resolved config field except `run.artifact_root` must equal the origin
   worktree's resolved config, and the origin config's fingerprint must equal
   the `resolved_fingerprint` recorded in the shard. Any other differing field
   fails the gate. See the semantic-delta gate for why this replaced a
   whole-config fingerprint comparison.
4. **Span alignment gate.** For all 576 sequences, using the tokenizer alone,
   run `src/inference/scoring.py::_locate_object_interval`,
   `_token_char_ranges`, and `_trace_for_span` over every parsed prediction.
   Every object span must map to exactly one contiguous generated-token
   interval, and every schema and coordinate sub-span must map to exactly one
   token. Existing alignment code is reused rather than reimplemented.
5. Re-derive the canonical union aggregates and require exactly `191/155/476`,
   `162/184/431`, and `169/177/423`.

Any failure stops the unit as invalid. Fuzzy matching, span dropping, or
partial decision-bearing scores are forbidden.

### Stage 1 — span-level replay and separation

One FP32 teacher-forced full forward per sequence via
`src/inference/hf_backend.py::teacher_forced_chosen_token_logprobs`, on GPU 6,
one checkpoint at a time, GPU 7 left free for unrelated work.

Per object span emit: checkpoint, image, seed, generated order, span id;
row-entry token likelihood; description token likelihoods and their
length-normalized mean; each of the four coordinate token likelihoods with
their mean and minimum; schema-wrapper mean; full-row mean and sum; the span's
generated-prefix position and the trajectory's total row count; **both** row
labels (per-span independent best-IoU and cluster-inherited); matched
ground-truth index and IoU; cluster id, support, medoid/member role, and
spatial dispersion.

Stage 1 answers the separation question: within each checkpoint, do
true-positive, ordinary false-positive, and catastrophic false-positive spans
separate by description likelihood, coordinate mean, or weakest coordinate?
It also reports whether low likelihood occurs preferentially later in a
trajectory, and whether trajectory coverage correlates with coordinate
likelihood P10.

**Stage 1 stops for research-lead review.** Stage 2 does not start
automatically. If likelihood shows no separation from the catastrophic tail,
running five cluster confidence families only draws curves through a known null.

### Stage 2 — cluster confidence and retention, gated on Stage 1

Aggregate span likelihood to the cluster. **Primary aggregation is the medoid
member's likelihood**, because the medoid's box is the canonical union
prediction and therefore the object a threshold accepts or rejects. Member
mean, minimum, maximum, and standard deviation are stored as alternative
features and compared, but do not define the primary statistic.

Compare these confidence families without hard-coding a production formula:
support only; coordinate likelihood only; description plus coordinate
likelihood; support plus likelihood; support plus likelihood plus spatial
dispersion.

Thresholds are selected under **leave-one-image-out cross-validation** and the
pooled out-of-sample estimate is the reported number. The in-sample value is
reported alongside, explicitly labeled as an upper bound.

Leave-one-image-out produces 12 thresholds per checkpoint, not one. Report the
spread of those 12 selected thresholds alongside pooled retention and rejection.
A family whose pooled numbers pass but whose fold-wise thresholds are widely
dispersed has no deployable operating point, and the handoff's promotion rule
requires checkpoint-specific threshold behavior rather than a single pooled
score.

## Stage 1 Observed

Execution is complete. The decision-bearing interpretation is owned by
[results.md](results.md); this section records Stage 1's executed facts only.

Two later readings constrain it. Support's apparent dominance here does not
survive the pre-registered operating point, because support is confounded with
the owners the filter must retain. More strongly, the area under the curve
reported below is **not a valid ranking of filter candidates** — see
[Ranking Quality Versus Usable Rejection](../2026-07-29-ranking-quality-versus-usable-rejection/results.md).
Read this section as a descriptive separation measurement only.

Stage 0 and Stage 1 executed on 2026-07-29.

Contract gates: all four pass, receipt at `stage0-gates-receipt.json`. All 576
sequences passed stored-hash recomputation, executed prompt-token parity, image
grid and media identity, and logprob-length checks; all 9,340 spans resolved to
exactly one contiguous token interval and all 74,720 schema and coordinate
sub-spans to exactly one token each. The replay is 576 full forwards, 465 s of
compute across three checkpoint loads.

The narrowed model-identity gate is exhaustive rather than a whitelist,
independently confirmed by the lead: perturbing `data.input_jsonl`,
`generation.top_p`, `model.processor.do_resize`, or `embedding_delta.path` is
each caught, and `run.artifact_root` is the sole exempted path. The receipt now
records that exemption as actually exercised — an exemption never observed to
fire is indistinguishable from a comparison that never ran.

Independent verification by the lead, not the implementing lane: chosen-token
logprobs recomputed in a fresh process with
`hf_backend.teacher_forced_chosen_token_logprobs` for 38 spans spanning three
images and seeds agreed with the emitted artifact at **exactly 0.0** absolute
difference on row sum, row-entry, and all four coordinate components. The
cluster reconstruction reproduces the pre-registration baseline table in every
cell, with 1,852 clusters each carrying exactly one medoid.

**Separation at the medoid, area under the ROC curve, true positive versus
catastrophic false positive, within checkpoint:**

| Checkpoint | coordinate mean | weakest coordinate | description mean | schema wrapper | full row | support |
|---|---:|---:|---:|---:|---:|---:|
| Sorted | 0.799 | 0.742 | 0.709 | 0.636 | 0.808 | **0.894** |
| Random | 0.758 | 0.601 | 0.417 | 0.442 | 0.728 | **0.821** |
| Permutation | 0.794 | 0.647 | 0.409 | 0.492 | 0.767 | **0.829** |

Median values, and the consensus baseline:

| Checkpoint | coord mean TP / catastrophic | description mean TP / catastrophic | support TP / catastrophic |
|---|---|---|---|
| Sorted | −2.603 / −3.218 | −0.152 / −0.383 | 12 / 1 |
| Random | −3.077 / −3.579 | −0.704 / −0.558 | 7 / 1 |
| Permutation | −3.101 / −3.652 | −0.615 / −0.540 | 7 / 1 |

**Observed.**

1. Coordinate likelihood carries real separation of catastrophic false positives
   in all three checkpoints, around 0.76–0.80.
2. The pre-registered grammar control passes decisively. Schema-wrapper
   likelihood is saturated — median −0.000 for true positives, ordinary false
   positives, and catastrophic false positives alike — so it carries
   essentially no information, and its area under the curve sits at or below
   chance for Random and Permutation. The separation is not grammar confidence.
3. Description likelihood is **inverted** for Random and Permutation (0.417 and
   0.409): badly grounded clusters are named *more* confidently than correct
   ones. Confident category naming is not evidence of correct grounding. Sorted
   does not show this inversion (0.709).
4. Trajectory support alone outranks every likelihood feature on every
   checkpoint (0.821–0.894 against a best likelihood feature of 0.728–0.808).
5. Low likelihood does **not** accumulate later in a trajectory. Median
   coordinate likelihood by position third is −3.399 / −3.518 / −3.185 (Random)
   and −2.678 / −2.942 / −2.751 (Sorted): non-monotone, with the final third no
   worse than the first. The "entry into a bad autoregressive basin" reading is
   not supported.

**Not claimed.** No branch of the decision-owning outcome is selected. The
pre-registered decision requires the owner-retention tradeoff, which is Stage 2.
Nothing here shows that likelihood adds anything *beyond* support, which is the
question Stage 2 must answer; on this evidence consensus is the stronger single
signal and likelihood is a candidate increment, not a replacement.

**Next discriminator.** At leave-one-image-out retention of at least 95% of
greedy-missed union-recovered owners, does any likelihood-bearing confidence
family reject more catastrophic clusters than support alone?

## Primary Estimand and Pre-Registered Stop Rule

Primary retention denominator is the **greedy-missed union-recovered owner**
set: 88 (Sorted), 44 (Random), 60 (Permutation). Retention over all union
true-positive owners (191/162/169) is reported alongside but is not primary,
because a filter that preserves overall recall while deleting the
sampling-only recoveries has destroyed the union operation's entire value.

Fixed before execution:

> At leave-one-image-out out-of-sample retention of at least 95% of
> greedy-missed union-recovered owners, if the best cluster confidence family
> rejects less than 20% of the catastrophic false-positive clusters, the outcome
> is decided as **confident binding failure** and threshold tuning stops.

The percentage is the rule; the counts follow from it. Passing requires
rejecting at least `ceil(0.20 × N)` catastrophic clusters, which is **40 of 197**
for Sorted, **27 of 131** for Random, and **28 of 138** for Permutation. These
counts are emitted as `catastrophic_rejection_pass_threshold` by the baseline
derivation script so the rule and the numbers cannot drift apart.

Because the Random denominator is only 44 owners, 95% retention permits losing
at most 2 owners and the point estimate is unstable to a single owner. Retention
on the primary denominator carries a deterministic bootstrap interval, and
Random's point estimate is not treated as a transferable threshold.

Interpretation rules, also fixed before execution:

- low-likelihood catastrophic tail → likelihood post-operation viable;
- high-likelihood but low-support tail → consensus is the controlling signal;
- high-likelihood and high-support catastrophic tail → internal confidence
  insufficient; recommend external image-conditioned verification;
- wrapper separates but coordinates do not → reject as grammar confidence.

Improved AUROC alone does not promote anything.

## Non-Goals and Preserved Behavior

Training semantics, inference policy, the prompt or output contract, the parser,
and the canonical matcher are unchanged. No resampling: exact teacher-forced
likelihood does not require regenerating trajectories. Policy likelihood, a
crossed checkpoint scorer matrix, and any production filter implementation are
out of scope. Source rollout artifacts are read-only. Unrelated modified and
untracked files in this worktree are preserved and not staged, reverted, or
cleaned.

## Cross-Checkpoint Claim Boundary

Absolute log-likelihoods are not comparable across the three checkpoints: they
are different models producing different row counts (3,570 / 2,852 / 2,918
parsed spans). Cross-checkpoint likelihood medians are reported as
**descriptive only**, with the confound stated. Every filtering claim is
within-checkpoint. Cluster features carry both within-checkpoint and
within-image percentiles; threshold scanning uses within-checkpoint, because a
deployed filter does not know image difficulty in advance.

## Execution Routing

Implementation is split into three lanes with disjoint write surfaces, so no two
workers own one semantic surface. The research lead froze the inter-lane data
contract before launch, which is what allows the lanes to run concurrently
rather than in sequence; the lead retains the protocol, interpretation, and
final acceptance.

| Lane | Owned file | Role and model | Acceptance mechanism |
|---|---|---|---|
| A — Stage 0 contract gates | `scripts/research/verify_likelihood_mining_contracts.py` | mechanical verifier, Haiku | Receipt must cover 576 sequences and 9,340 spans; exits non-zero on any gate failure; must reject a deliberately corrupted span |
| B — Stage 1 replay | `scripts/research/run_span_likelihood_replay.py` | builder, Opus | Span counts 3,570 / 2,852 / 2,918; same-sequence replay bitwise identical; plumbing cross-validated against the independent production greedy token trace |
| C — cluster labels and features | `scripts/research/build_cluster_confidence.py` | builder, Sonnet | Must reproduce the pre-registration baseline table cell for cell, asserted in-script |

Lane B carries the highest silent-error risk — a wrong logits slice or prompt
width yields plausible but wrong likelihoods everywhere downstream — so it was
routed directly to the strongest builder rather than cascaded, and it carries an
independent-artifact cross-check rather than only self-consistency. Lane C is
semantically dense but exactly pinned by known counts, so a mid-tier builder
behind that verifier is sufficient. Lane A is a bounded scan behind an
executable check.

Frozen inter-lane contract: Lane B emits `span-likelihood.jsonl` carrying only
likelihood and provenance fields; Lane C emits `span-labels.jsonl` and
`cluster-confidence.json`. Labels never enter Lane B's file and likelihoods are
never recomputed in Lane C's. The join key is
**`(checkpoint, object_span_id)`** — `object_span_id` alone is not unique across
checkpoints, and a row lacking `checkpoint` must abort the join rather than
resolve it ambiguously.

Lane C's outputs are accepted: re-derived independently from its own artifacts,
the acceptance table matches in every cell, all 1,852 clusters carry exactly one
medoid, and `(checkpoint, object_span_id)` yields 9,340 distinct keys.

Threshold scanning, leave-one-image-out validation, plots, and interpretation
are not delegated.

## Artifact Root

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1/`

This co-locates derived evidence with the source rollouts rather than using the
`outputs/research/<investigation>/<unit-id>/<run-id>/` layout, so that the
replay sits next to the trajectories it replays; `likelihood-mining-v1` is the
immutable run identifier. Expected products: `span-likelihood.jsonl`,
`cluster-confidence.json`, `summary.json` with the Stage 0 gate receipts,
`report.md`, distribution and retention plots, and optionally a filtered 1×3
visualization in the existing `Sorted | Random | Permutation` panel order.

**Replay note.** Producer scripts deleted from `research-probes` on 2026-08-28 (reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`: `git worktree add <tmp> research-base-v2`.

## Semantic-Delta Gate

Conditions added during alignment that the originating handoff did not fix.

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Two stages with a research-lead review gate between separation and retention curves | Alignment round 1 | conservative design choice | Cost; Stage 1 alone cannot promote | Approved by research lead |
| Both per-span independent best-IoU and cluster-inherited row labels emitted; per-span primary for distributional claims, cluster-inherited for retention | Alignment round 1 | semantic fork — the handoff assumed a canonical per-span label that does not exist | Claim; the reported separation number differs by label | Approved by research lead |
| Catastrophic false positives split into class-absent and mis-grounded; ordinary split into duplicate and loose | Alignment rounds 1–2 | conservative design choice | Claim; prevents distinct failure modes being pooled | Approved by research lead |
| Pre-registered stop rule at ≥95% retention and <20% catastrophic rejection | Alignment round 1 | conservative design choice | Stop rule | Approved by research lead |
| Primary retention denominator is greedy-missed union-recovered owners, with bootstrap interval | Alignment round 2 | scientific invariant — the handoff's promotion rule names this set | Estimand | Inherited and made explicit |
| Cluster likelihood aggregated at the medoid member | Alignment round 2 | semantic fork — undefined in the handoff | Claim; member mean/min/max give different answers | Approved by research lead |
| Leave-one-image-out cross-validation; the pre-registered threshold binds to the out-of-sample estimate | Alignment round 3 | conservative design choice | Claim and stop rule; materially harder to pass than in-sample | Approved by research lead |
| Raw-model likelihood only; policy likelihood deferred | Handoff | inherited | None | Inherited |
| Model-identity gate narrowed from whole-config fingerprint equality to "every resolved field except `run.artifact_root` matches the origin worktree, and the origin config reproduces the stored fingerprint" | Stage 0 execution, 2026-07-29 | conservative design choice — corrects a defect in the pre-registration | Stop rule | Approved by research lead after the field-level diff |
| Cross-lane join key corrected from `object_span_id` to `(checkpoint, object_span_id)` | Lane C, 2026-07-29 | scientific invariant — the original key is not unique | Claim; an ambiguous join would mix checkpoints | Inherited; the defect was mine, the fix is forced |

No condition in this table narrows the originating objective without the
research lead's explicit acceptance, and none was introduced after observing
likelihood values.

Two of these were forced by execution rather than chosen, and both are defects
in the pre-registration that execution exposed:

- The **fingerprint gate** as written compared the whole resolved config. Only
  `run.artifact_root` differs between this checkout and the origin worktree, and
  that field is a derived output directory — a function of which worktree the
  process runs in, not of the model. The gate therefore tested checkout
  location, not model identity, and would fail by construction in any worktree
  but one. Narrowing it is a correction, not a relaxation: the exemption is
  restricted to exactly one named field, every other field must match, the
  origin config must still reproduce the stored fingerprint, and the actual
  identity burden is carried per sequence by executed prompt-token parity,
  `executed_media_sha256`, and `observed_image_grid_thw`. A future difference in
  any model-bearing field still fails.
- The **join key** `object_span_id` has the form
  `{example_id}:seed-{seed}:span-{n}`. All three checkpoints replay the same 12
  images under the same 16 seeds, so the same identifier recurs once per
  checkpoint with a different box and description: 9,340 spans carry only 3,761
  distinct identifiers. Joining on it alone would silently merge checkpoints.
  The composite key is the only correct one.

## Alignment

The intervention unit is the object span; the final evaluation surface is the
union cluster; the transfer claim is that span-level likelihood aggregated at
the medoid predicts cluster-level grounding correctness. Behavior that must be
preserved is the greedy-missed union-recovered owner set. Ambiguous evidence —
ordinary false positives of the duplicate subclass — stays neutral, since a
duplicate of a correctly grounded owner is a different failure from a
mis-grounded box. Signal supply is adequate for the separation question (197 /
131 / 138 catastrophic clusters) but thin for the retention question on Random
(44 owners), which is why that denominator carries an interval and no
transferable threshold.

## Cost and Stop

Roughly 3–5 hours, dominated by alignment and analysis rather than GPU compute;
576 forwards of a 2-billion-parameter model in FP32 are minutes of compute plus
three checkpoint loads.

Stop after Stage 1 for review. Stop the unit entirely on any Stage 0 gate
failure, or when the pre-registered stop rule decides confident binding failure.
Do not launch training, change inference policy, promote a filter, extend the
panel, or run the crossed checkpoint scorer matrix without a separate decision.
