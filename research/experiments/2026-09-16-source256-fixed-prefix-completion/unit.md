# Source256: fixed Source-history completion versus canonical SFT

## Authority and question

The user approved this execution package on 2026-09-16 in task
`01a0a81a-9e32-7db1-bd07-86fa601f4276`, explicitly said to start, and confirmed
the 50/50 mixture, 64-update dose and 128-candidate review budgets. Eight GPUs
are available within this contract. Preparation, implementation, bounded
corroboration/review and execution are authorized, subject to the new scarcity
gate below. This supersedes the earlier draft-only boundary.

**From original geo_sorted_xy Source step2444, does a fixed mixture of canonical
replay and completion after the Source's actual trusted prefixes reduce natural
greedy known-owner FN relative to canonical SFT, with the same owner information
and image exposures, without worse incumbent retention or development recall?**

The ultimate objective is fewer physical misses under incomplete annotation.
This first contrast tests a learning package: conditioning history AND supervision
allocation change. It does not isolate a pure prefix mechanism, label-supply
effect, EOS mechanism, refresh benefit or fully annotation-free learning.

Confirmed by the user in this planning turn:

- Reuse train256/dev128 and the common original Source anchor selected below.
- Advancement requires training FN benefit, incumbent retention no worse than
  the control, and no development recall regression. Training-only improvement
  remains a limited result and does not qualify for advancement.
- Both arms retain standard EOS supervision. EOS denotes the end of the current
  trusted list, not proof that every physical object has been found. EOS masking
  is a separate possible ablation, not part of this first contrast.
- Co-DETR crop+resize corroboration belongs after rollout matching and before
  learning-trajectory construction and masks. It does not automatically admit GT.

## Predecessor recap and changed question

The [workshop](../2026-09-16-research-direction-workshop/results.md) owns the full
recap. Human13 and COCO22 establish bounded SFT fitting; useful sampled outputs
exist; higher sequence likelihood does not ensure greedy entry; entry success
does not ensure later preservation. July route imitation and prefix training
already showed owner exchange and output debt. This is not a new claim that SFT,
prefixes or suffix imitation have never been tried.

The incremental contrast holds the trusted owner bank, anchor and training
population fixed, uses one transparent full-remaining-bank target construction,
and judges the resulting empty-prefix complete outputs against matched canonical
SFT. A positive result would justify testing later refresh. No incremental value
at the fixed dose closes this candidate at that dose, rather than triggering more
labels, K, seeds or a KV/architecture search automatically.

The [previous train256 SFT course](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/results.md)
improved training IoU50 matches by 135 while losing 18 on dev128. Its endpoint is
not a promoted starting adapter. The [retained K4 census](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-natural-candidate-opportunity/results.md)
contains 32 strong-witness images, but selecting only them would change the
population. All 256 images remain in the training and reporting denominator.

## Fixed population and anchor

Use the existing image IDs and canonical processed images from
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/`:

| Input | Images | Original annotated objects | SHA256 |
|---|---:|---:|---|
| train.jsonl |256|1955|`05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`|
| dev.jsonl |128|891|`b2ba42e6be18ca179cbdac387ed7af5b8400cf88a5a7d63f3b46643f2f61a46c`|

The previous turn freshly checked image existence, digests and zero train/dev ID
overlap. These object counts are source counts, not automatically admitted new
teacher denominators. Publish the versioned exclusions and additions before
training; preserve the original files. Report the existing density strata and
opportunity strata without selecting images by treatment success.

Common checkpoint root:
`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/`.

- Load `adapter/` and its paired `special_token_embeddings/`.
- Adapter weights SHA256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
- Embedding weights SHA256:
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.
- Base/tokenizer/processor:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- Tokenizer SHA256:
  `ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8`.

These bytes match the retained Source acquisition. Source already underwent COCO
supervision; neither source-corpus exposure nor base pretraining absence is
claimed. Dev128 is a historically used development panel held out of THIS
update, not an untouched final test. Later Sample/COCO22 adapters explicitly
fitted five dev128 images and are excluded from this comparison. Do not merge
COCO22 into an additional independent test denominator.

## One acquisition and corroboration stage, shared by both arms

1. Prefer the retained Source greedy and four natural samples per training image
   from the K4 census, after checkpoint, prompt, image, decode and artifact
   identity checks. This is reuse of one fixed acquisition, not a fresh rollout
   claim. If a decisive identity gap makes reuse invalid, prepare one bounded
   replacement acquisition of the same 256 x (greedy + K4), and amend its exact
   identity before execution; do not mix incompatible banks or increase K.
2. Match against the seed annotation ledger and deduplicate unmatched candidates
   into owner hypotheses. Preserve raw proposals, parse failures, locator spans,
   policy/seed and every crop-to-original coordinate transform.
3. Run Co-DETR on unpainted unmatched-centered context crops: inherit 3x context,
   minimum 128 source pixels per side, clipping and the existing native
   keep-ratio Resize (2048,1280). Bind its model/config before calling it. Render
   the full image with candidate/detector boxes alongside the context crop for
   review. Do not supply painted boxes to the detector.
4. Use inherited score >=0.50 and same-category candidate/detector IoU >=0.75
   as a support-priority signal, not a precision guarantee or admission rule.
   Check identity, part/whole, duplicate, extent and competing-instance evidence.
   A miss by Co-DETR is unknown, not a negative label. A supported hand box is
   not an independent person; overlap with a neighbor does not itself invalidate
   a person's box. No coordinate averaging between uncertain owners.
5. Approved initial human/agent review budget: one pass over at most 128 unique
   owner hypotheses, selected before training by support/conflict priorities with
   a round-robin spread over images and at most two per image. These are review
   limits, not bounds on true object counts. Unprocessed or ambiguous candidates
   remain HOLD and do not block freezing the bank. Do not exhaust raw proposals.
6. Only visually lead-admitted owners or repairs become positive targets. Write
   confirmed additions to the matching source-format JSONL `unlabeled` field in
   a versioned annotation successor, with stable owner IDs and provenance;
   preserve the original `objects`, old versions and rejected/HOLD entries.
   Existing GT is not exempt from the current weak-visual/extent policy. Do not
   invent a blanket small-area deletion threshold or silently remove a real
   tiny but visually identifiable object.
7. Freeze ONE train bank and its evaluation binding, then give A and B identical
   admitted owner/category/box information. Re-match the fixed Source histories
   against this bank before constructing masks. No teacher changes during
   training; no development labels enter the training bank.

The shared [unmatched policy](../../../docs/eval/UNMATCHED_REVIEW.md) and
[latest user adjudication](../2026-09-16-codetr-only-review-proxy/results.md)
own physical/TIDE-aligned interpretation. This stage measures neither the causal
benefit of Co-DETR nor automatic-labeling precision. The full image pool stays
fixed even when a candidate is not admitted.

## Frozen learning contrast

Call the arms **A: canonical SFT** and **B: fixed Source-history completion**.
They are not the earlier two-initialization A/B experiment.

Each matched optimizer update contains 64 image presentations: 32 common
canonical presentations and 32 variable presentations. Use a fixed balanced
schedule so every image receives eight common and eight variable presentations
over the fixed 64 updates. A uses canonical targets in both branches. B uses
the following correction in its variable branch when eligible, otherwise the
identical canonical target. Average each branch over its 32 images, then combine
with weights 0.5/0.5; a fallback does not disappear from either denominator.

### B prefix, target and masks

- Use the frozen Source natural greedy trajectory. Take its longest exact
  complete-object-row prefix before the first unknown, false, duplicate,
  malformed or otherwise inadmissible row, or its observed termination. Every
  retained row must confidently refer to a distinct bank owner under the frozen
  category/identity/extent policy. Do not skip a bad internal row and call the
  resulting history the original trajectory.
- Require a nonempty trusted prefix and at least one bank owner still missing.
  Otherwise use canonical fallback. This never excludes the image from training.
- Preserve prefix token IDs and the original image/prompt. Omit the original EOS
  from the context. Append ALL remaining bank owners with unchanged trusted
  target rows in geo_sorted_xy order, followed by standard EOS. Preserve the
  whole suffix, including owners that Source had emitted later; do not credit
  only the first repaired row.
- This suffix is a newly constructed supervised target. It is not an observed
  sampled transcript, and a sample union does not certify it as one. Its
  row-level validity and bank coverage must be checked under the new history.
- Mask prompt and prefix labels; supervise every suffix token, including syntax,
  coordinates and EOS. Geometry auxiliary loss applies ONLY to supervised target
  rows. A masked prefix remains input context and may carry gradients through
  its hidden states; this is not a detach or delete operation.
- If the entire target exceeds the frozen training length bound, use canonical
  fallback rather than truncate owners. If even the canonical form cannot fit,
  surface an input-contract failure; do not silently drop that image.

All active-token CE denominators respect their masks. B generally has fewer
supervised tokens and a larger EOS fraction in correction examples. Preserve
the user's standard-EOS choice; do not add the adviser's alternate EOS masking
rule. Log active lengths, effective EOS weight, target-owner exposures, prefix
owner counts and fallback reasons. Differences in credit allocation are part of
the learning package, not proof of a pure history effect. Zero eligible
corrections means no executable contrast; it is not evidence against learning.

## Frozen dose and invariant training recipe

The user confirmed these numeric defaults on 2026-09-16:

- One paired seed, 19; **64 applied updates per arm**, effective image batch 64:
  4096 image presentations per arm, 16 image passes. No automatic extra dose.
- Full language DoRA A/B/magnitude, rank16/alpha32/dropout0. Base, vision,
  aligner, selected-token embeddings and readout stay frozen in both arms.
- Fresh AdamW per arm: language LR 1e-5, betas (0.9,0.999), eps1e-8,
  weight decay0, clip1.0; cosine schedule across64 updates, zero warmup.
- Sample-equal active-token CE and the same accepted shared geometry hinge
  weight0.01. Reuse its implementation/normalization; do not reopen shared loss
  or normalization as another study axis.
- Identical image schedule, parameter surface, optimizer policy, precision and
  train/eval preprocessing. Supervised-token and physical compute counts are
  recorded separately; equal updates are not claimed to mean equal total cost.
- Save16/32/64; full natural train256/dev128 reads at Source0,16,64. The primary
  endpoint is64, with16 a predeclared learning/debt diagnostic. Do not select a
  favorable intermediate checkpoint after inspecting dev results.

## Evaluation and advancement

Use original images, empty-prefix natural greedy and the same native decode
contract as Source: RP1, cap3084, no forced row opener, supplied prefix or rescue.
Retained outputs may be rescored against the new fixed reference, but do not
reuse historical counts as if the revised reference/metric were identical.

Freeze the trusted reference and scoring before training. Primary known-owner
coverage uses global one-to-one class-agnostic IoU >=0.5; retain class-consistent
IoU50/60/80 separately, plus confirmed identity/extent diagnoses. This is a
geometry-based known-owner FN measure, not a census of all physical misses.
Group/individual/part scopes are recorded separately and a group box earns no
invented atomic-owner count. Publish both pooled and per-image gains/losses.

For each split, relative to the common Source define gained owners G and lost
incumbents L. Candidate advancement at the fixed64 endpoint requires:

1. Train: B has fewer FN than A and than Source; L(B) <= L(A).
2. Dev: B coverage >= A coverage and >= Source coverage; L(B) <= L(A).
3. No increase over the control in confirmed false-instance, repeat, malformed,
   invalid-geometry or cap debt. Keep each debt type separate, rather than hiding
   it in a scalar score. Report Source comparisons as well.
4. All claimed differences have mechanically valid, comparably reviewed evidence.
   Unknown unmatched rows remain unknown, not zero false positives. Uncertainty
   capable of reversing a claimed physical win limits advancement/claim scope.

Endpoint unmatched triage uses the same Co-DETR route and blinded comparative
review: initially at most128 unique decision-relevant hypotheses across A/B,
with policy identity hidden where practical. Unreviewed rows stay HOLD, with
coverage reported. This is a bounded decision review, not all-proposal completion.
New endpoint discoveries do not rewrite the primary training/reference target;
an explicitly supplemental symmetric rescore of Source/A/B may use them.

Latest user amendment during execution: stop further `view_images` and converge.
Initial review therefore closes at120 visually reviewed hypotheses out of128
selected:34 admitted additions and one targeted original-owner exclusion;
the8 unreviewed selected hypotheses remain HOLD. No further visual review is
launched under this continuation, including at the endpoint. Use existing visual
evidence and fixed-reference endpoint metrics; unresolved physical-debt evidence
limits advancement claims rather than triggering another review expansion.

Training-only benefit => `in_sample_gain_no_advancement`. Conditional-prefix
improvement without natural benefit => `conditional_only`. Clean endpoint tie or
no advantage => `no_incremental_value_at_registered_dose`. Technical invalidity
=> the affected scientific contrast remains unanswered. A qualifying win is a
candidate for a separately defined refresh comparison, not proof of general
unlabeled-data learning. No significance or population-generalization claim
follows from one seed and this historically used development set.

## Eight-GPU execution plan

- Reconcile existing workers/jobs, then use all eight devices for independent
  crop-detector or natural-readback shards when those stages run. Shared ambient
  stress occupancy is not a reason to wait before an authorized launch.
- Default training topology is A on four GPUs and B on four GPUs concurrently,
  preserving global batch64 per arm. If measured startup/throughput/memory makes
  sequential eight-GPU arms faster, use that matched topology for BOTH arms.
- One bounded operational benchmark may compare these two topologies and at
  most two safe microbatch sizes, using at most32 disposable applied updates in
  total. Restore Source before the real run. Measure completed image/token
  throughput, GPU memory, wall time and loss/gradient accounting; utilization
  percentage alone does not decide. No scientific arm is added by this benchmark.
- One production-shaped two-update qualification per arm must exercise actual
  image entry, exact prefix/masks, full-bank coverage, EOS, geometry masking,
  branch/rank normalization, optimizer application, save/reload and natural
  readback. Reuse its valid runtime evidence; do not chain redundant smokes.
- Latest user amendment on 2026-09-16 supersedes batch-size maximization:
  use **natural greedy readback/evaluation batch=4**. Perform only the minimal
  parity qualification needed for valid results; do not search batch8 or larger
  or spend time on optional throughput/topology benchmarks. The user explicitly
  requests acceptance and scientific research focus. Default paired training
  remains four GPUs per arm with the current safe microbatch.
- Main runs begin from Source, with fresh optimizers, not benchmark/qualification
  states. Capture config/source/input hashes and real counters in one run receipt.
  Runtime owner reuses existing maintained training/eval components; exact entry
  and resolved configs are implementation products, not falsely ready today.

Once started, allow scoped technical repair and one affected-stage rerun, with
the failure retained. A repeated technical failure pauses that stage. After the
paired64 endpoint, close the result before any refresh, new seed, larger dataset,
changed EOS rule, new objective or architecture work. Eight available GPUs do not
create an obligation to add experimental factors.

## Authorized execution and scarcity gate

Before the main GPU training course, freeze and verify at least **64 of 256**
images with eligible trusted Source-prefix completions. Under the fixed 50/50
schedule this means at least **12.5% of all training presentations** actually
consume completion targets. The user explicitly confirmed these thresholds.
Report structural candidates separately from fully admitted, length-valid
completions; include fallback reasons, prefix/suffix lengths and owner exposures.
If either bound fails, **stop and report before main training**. Do not increase
K, review budget, select a smaller denominator, alter mixture or run a nearly
A=A comparison. This is a feasibility stop, not a scientific negative.

After bank admission and this gate, proceed through the registered real-entry
qualification and paired course without redundant per-step approval. Routine
implementation choices remain with execution owners. User-owned semantic
changes and the existing post-endpoint stop rule remain in force.

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/`.
Teacher/evaluation digests, eligibility census, resolved configs and real-entry
receipts are preparation/execution products, not assumed completed gates.

## Current implementation ownership

- Lead `/root`: protocol/state, Co-DETR and visual admission coordination,
  scarcity gate, integration and scientific acceptance.
- `/root/source_bank`: retained identity, shared-bank preparation, exact prefix
  data/masks and eligibility receipt; new `source256_data` code and preparation
  artifacts.
- `/root/paired_runtime`: paired trainer/evaluation consumer and real-entry
  execution; new `source256` runtime code and runtime artifacts. GPU execution
  begins only after the lead releases the required readiness boundary.

No nested delegation. Existing unrelated dirty work is preserved. This section
routes package dependencies; execution receipts own actual results.
