# Source256 completion CE normalized by canonical target length

## Authorized question and predecessor

From original Source step2444, does changing only eligible completion CE from
suffix-token mean to same-image canonical-target-token normalization preserve
B's gained known owners while reducing incumbent loss and output errors at64
updates, under the original canonical/development advancement gates?

The user explicitly authorized this independent successor and execution after
loss accounting and actual-entry checks on2026-09-16. The [predecessor result](../2026-09-16-source256-fixed-prefix-completion/results.md)
remains accepted and unchanged: canonical A has train FN628/dev280; original B
has667/284 versus Source688/271. A gains95/loses35 train owners; B gains96/loses75.
B also has more malformed/repeated/capped output. This is not a runtime failure
or evidence against every rollout-learning method.

The remaining question is whether short-suffix mean normalization contributed
to B's preservation failure. The strongest competing explanation is simply
weaker learning: both gains and losses could shrink under lower CE strength.
The fixed natural greedy endpoint and per-owner G/L distinguish these outcomes;
no additional proxy/gradient screen or visual census is a prerequisite.

## Single changed factor

Call the successor **B-normalized**. For each eligible completion presentation:

`CE = sum(NLL over active suffix tokens including EOS) / canonical_full_target_token_count_for_same_image`

The canonical denominator includes every canonical target token, including
syntax, coordinates and EOS, and excludes prompt tokens. It is the frozen
canonical route length, not a generated length or batch aggregate. For suffix10
and canonical40, new CE is old CE/4; geometry stays exactly unchanged.

All common/canonical/fallback CE terms retain the original sample-equal mean.
Preserve the exact original B histories, targets, binary0/1 CE and geometry masks,
image schedule,50/50 branches, Source-prefix input gradients, standard EOS,
geometry implementation/normalization/weight0.01 and branch/rank normalization.
Do not place fractional values in CE masks and do not scale the complete loss.
No teacher, owner, locator, category, ordering, weak-bank or annotation changes.

This changes completion CE strength and its ratio to geometry. It cannot alone
identify an EOS mechanism, a prefix mechanism or the optimal normalization.

## Immutable cohort, anchor and controls

Inherit the [original protocol](../2026-09-16-source256-fixed-prefix-completion/unit.md)
for exact checkpoint, paired embeddings, tokenizer, processing, optimizer,
metric and debt semantics except the declared CE denominator delta.

- Preparation: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json`.
  SHA256 `9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242`.
- Train256 with1988 admitted owners; dev128 with891 owners. Processed
  `rescale_32_1024_bbox_len12000_xy_sorted`, not rawCOCO. Original annotations and
  versioned34 additions/one exclusion remain fixed; unknown remains unknown.
- Exactly143 eligible images;1144/4096 completion presentations (27.9296875%).
  Reuse admission; no new unmatched or view_images review.
- Source step2444 adapter SHA256
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`;
  paired embeddings SHA256
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.
- Reusable matched-control trial SHA256
  `2cfd2a39d3aa3a682002586fe7849225b44a3d4dc80c643d39b33f5e989e409d` at
  original output root `runtime/main-v1/trial.json`.
- Reusable scored natural outputs SHA256
  `1c45305af5c3b7689e75465092773437005a312632946ad2a83e706e48a8083d` at
  original output root `runtime/main-v1/evaluation/result.json`.
  Reuse Source/A/B saved outputs only after exact identity/metric checks.
  Bind raw readbacks and repaired evaluator, not historical counts alone.

Fresh CPU projection of143 eligible routes: suffix/canonical token ratio has
median0.6153846, mean0.6000887, min0.0970874 and max0.9738372. Thus prior
suffix-mean CE amplification relative to the proposed denominator has median
1.625x and max10.3x. These are target accounting facts, not measured gradients
or proof of the hypothesis; the bound preparation is their source.

## Dose, execution and qualification

One successor arm, seed19,64 applied updates, global batch64,4096 presentations.
Start from the original Source and paired embeddings with fresh AdamW, not B64
or qualification state. Preserve rank16/alpha32/dropout0 language DoRA,
LR1e-5, AdamW(0.9,0.999),eps1e-8,weight_decay0,clip1,cosine64,no warmup.
Frozen vision/base/readout/selected-token embeddings and precision match B.

Use the original four-rank training/microbatch2 topology to keep the reused B
control matched. The eight available GPUs serve existing evaluation shards;
no topology/throughput benchmark or added arm. Expected main training2048 model
calls and4096 logical forwards; record actual active-token denominators, CE
numerators/scales, geometry and memory. Reuse previous unchanged execution
qualification evidence and add a bounded two-update actual-entry successor
check that consumes the new denominator and exercises save/reload. Main64 is
released by the lead after CPU loss accounting and this real-entry acceptance.

Save16/32/64; new natural greedy readbacks at16/64 on train256/dev128,bs4,RP1,
cap3084,empty-prefix, no rescue. Step64 is primary;16 is diagnostic, not selected
post hoc. New evaluation budget768 image requests,192 batch4 calls,32 shards;
Source and prior A/B reads are reused. Bind configs/code/checkpoints/artifacts.
Long producers must live in independent tmux sessions and emit both success and
failure events for wake-me-up; subagents do not maintain polling loops. Preserve
completed units and resume only missing work. A derived reducer error does not
authorize repeating model execution. One scoped repair/affected-stage retry;
repeated failure pauses the affected stage, not a scientific null.

## Evidence and decision

Primary: natural greedy global one-to-one class-agnostic IoU50 known-owner FN,
G/L relative to Source, and incumbent retention, pooled and per-image. Report
class-consistent50/60/80 separately. Compare B-normalized to original B and A;
report gains retained/lost/replaced as owner identities where available.
Separate strict-repeat proxy, malformed, invalid geometry and cap/EOS debt.
No automatic confirmed-FP count from annotation-unmatched rows.

Support for continuing this normalization route requires B's new gains to be
preserved (trainG at least originalB's96), fewer old-owner losses than B's75,
and decreasing output errors without trading one debt category for another.
Report dev coverage and retention against B, A and Source. If gains and losses
both shrink, classify as weaker learning rather than a selective preservation
success. Mixed outcomes remain mixed; no scalar debt/coverage composite.

Advancement remains stricter: train FN below A628 and Source688; train Source
loss no worse A35; dev coverage at least A611 and Source620, with dev Source
loss no worse A26; output debt no worse A. Unresolved physical false-instance
judgments limit the claim and never trigger an automatic visual census.
One seed/historically used dev does not establish population generalization.

At the fixed endpoint, publish a technical acceptance status and separate
scientific conclusion. Stop before new seeds, extra dose, refresh, full visual
review, weak-bank work, new objectives or architecture. Co-DETR weak-bank is a
separate future unit; localization repair and owner addition remain distinct.

## Ownership and artifacts

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-completion-ce-normalization/`.

- Lead: this protocol/state, integration, qualification/main release and result.
- `/root/normalized_training`, luna-max: CE accounting and training entry,
  focused invariants, new training/trial modules and preparation/qualification.
- `/root/normalized_evaluation`, terra-max: control reuse, new evaluation path,
  durable main producer and result receipt. Coordinate interface with training.

No nested delegation. Preserve bound predecessor producer bytes and unrelated
dirty work. Assess these model routes from accepted package quality and actual
repair burden; this one trial cannot establish universal replacement of Sol.
