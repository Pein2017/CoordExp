# COCO227 support expansion: sample-equal versus token-equal CE

**Closed; lead-accepted.** Both CE arms reach sustained clean227-owner completion
at saved16 and remain clean through256. See [results.md](results.md).

## Authority and question

The user approved the lead's proposed next round on2026-09-15, including bounded
throughput qualification, a227-owner trusted teacher, the two CE reductions,
and all8 GPUs. Previous dual-start-v3 is closed and remains immutable evidence.

From the accepted A final new-step256 parameter checkpoint, does globally
token-equal CE change the observed speed of clean227-owner natural-greedy
completion, relative to sample-equal CE, while preserving the original218 owners
and the same fixed training dose?

Predecessor: [accepted paired result](../2026-09-14-training-set-completion-curriculum/dual-start-results.md).
Output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization`.

## Frozen research contrast

- One common source: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/A/training/checkpoints/step-00256/adapter`,
  with the same bound original step2444 additive special-token embedding delta.
- Same11 original images. Teacher227 is exactly the previously trusted218 plus
  all9 current-known verified COCO-80 owners not in218: one person, seven donuts,
  one chair. Reuse verified owner geometry/class/provenance; no new guesses.
- Keep every old218 row's relative order and literal fields. Append added rows
  per image in stable owner-ID order before EOS; both arms receive identical
  freshly rebuilt sequences. Do not introduce another sorting or refresh axis.
- Preserve old218/new9, historical232 and full current-known248 ledgers. The19
  class-unknown and2 known non-COCO owners remain outside this training teacher.
- Arm S: mean over CE-eligible images of their active-token mean NLL.
- Arm T: sum of masked NLL across the global update divided by the global number
  of active tokens. Binary mask controls numerator and denominator; all-masked
  samples are excluded from CE-eligible-image denominator. EOS remains supervised
  and masked input context remains present. Never substitute sample sum/image mean.
- Geometry stays equal-image mean of per-image mean complete-box expected-axis
  hinge, weight0.01 and margin1/999, independent of CE reduction.
- Both fresh seed42 AdamW, lr1e-5, betas(.9,.999), eps1e-8, weight_decay0,
  foreachFalse, clip1; same language DoRA surface, base/vision/embedding/lm_head
  frozen; same fp32 SDPA/model.eval numerical recipe.
- Same global11-image update and contiguous3/3/3/2 partition across4 ranks per
  arm (S physical0-3,T physical4-7). Qualify microbatch execution before selection;
  both scientific arms use the same selected microbatch/checkpointing strategy.

## Dose, evidence and decision

- Each arm256 NEW updates,2816 logical image exposures, maximum7200s training
  wall time. Preserve checkpoints8,16,32,64,128,256. No dose extension.
- Source0 is common; re-admit the11 retained prior A-final256 natural outputs
  only under exact source/config/prompt/media/stop identity, without relabeling
  their original provenance. New readback is132 requests: two arms x six saved
  checkpoints x11 images. Every request cap3084, original image, empty assistant
  prefix, greedy temperature0/top_p1/top_k0/RP1, EOS151645. Each readback worker
  has7200s wall limit, and the entire132-request phase also has a7200s wall
  deadline including queued endpoint jobs. Use at most one live endpoint worker
  per GPU; reuse completed rows and recover only missing work.
- Predeclared trajectory outcome: earliest saved checkpoint in8/16/32/64/128/256
  that is clean-complete and remains so at all later saved checkpoints. If none,
  record not attained by256. This is a sampled milestone, not an exact hitting
  time or convergence claim.
- Clean-complete means227 valid predictions matched one-to-one to227 targets
  at IoU>=.5, correct known descriptions, zero malformed/geometry-invalid/non-COCO/
  duplicate/physical-error burden and natural EOS on all11 images.
- Report new9 coverage/FN and retained/lost original218 at every saved point.
  Primary old218/new9 counts partition one joint227 one-to-one assignment; never
  independently credit one prediction to both partitions. Independently matched
  old218 can be a separately labeled diagnostic only. Recompute the common source
  partition rather than assuming which new owners were absent.
  Final256 FN/F1 and complete-output errors remain decision-bearing. Report all
  historical232/current-known248 deficits separately and IoU>=.8 as diagnostic.
  Matching remains class-agnostic, cardinality-first, one-to-one. F1 uses all valid
  predictions; unmatched is not physical FP. No hidden composite score.
- Same matched final quality with earlier sustained clean milestone supports
  faster observed acquisition under that CE objective. Tradeoffs with old-owner
  retention/errors are explicit; a trajectory crossing need not yield a winner.

## Throughput qualification: mechanical, not a scientific arm

- Training owner may run at most6 short configurations, each at most2 updates
  and600s, maximum132 logical image exposures total. Cover actual global11
  sample/token reduction parity, uneven rank workloads, microbatch1 versus2/3,
  and reduced activation checkpointing only if within this budget and useful.
- Readback owner may run at most36 image requests total, cap3084 each, to compare
  serial and microbatch2/3 on this11-image common source. Maximum1200s/config.
  Use retained original outputs as an additional reference, not a substitute for
  live timing. Original identity, stop, owner coverage and output errors must
  agree; tiny coordinate-bin differences require detection-level review and
  cannot silently change eligibility or matching.
- Training qualification uses GPUs0-3; readback qualification GPUs4-7. Model
  timings, forward calls, padding, rank imbalance, memory and checkpointing
  strategy are measured. Pick a correct measured faster configuration; if no
  advantage is established, retain the validated serial path. A failed speed
  option does not block the scientific trial when an accepted route exists.
- All model work uses named tmux and durable logs/exit receipts. No automatic
  extra tuning beyond these bounds. Root releases full training after teacher
  and real execution admission; already authorized release is an internal gate.

## Interpretation limits and stop

Both arms share the expanded teacher, so their contrast estimates normalization
under that expansion. It cannot isolate the effect of support expansion versus
the predecessor. Seven of nine additions are donuts on one image; do not claim
a population-wide normalization winner. CE loss/gradient/clipping differences
are diagnostics, not causal substitutes for natural readback.

No held-out/generalization claim, online refresh, extra category adjudication,
new-owner discovery, optimizer sweep, seed sweep, or downstream publication.
Stop after the bounded batch and its acceptance. Technical failures do not become
scientific negatives; retain valid training and repair consumers without retraining.

## Ownership

Root owns this unit, bindings, launch release and final decision. Reused data/
evaluation owner builds227 teacher and generalized scoped scoring; reused training
owner owns new batched CE execution and paired controller; fresh readback owner
owns batched generation qualification and durable readback. All are L1, no L2.
Prior frozen producer source files and old artifacts remain unchanged; import
existing helpers through new scoped modules rather than edit old source bindings.

## Preparation evidence

### Readback scheduler recovery

The controller failed after both training arms completed256 updates and saved
all six checkpoints. The actual runtime lacks `os.pidfd_open`; the first S8
endpoint had already started when scheduler registration raised AttributeError.
This is a consumer scheduling failure, not an invalid training result. The
root validated both training terminals and replayed all512 update formulas
with zero error; both arms had identical initial per-image route terms.

The live original S8 worker (PID986468, GPU0, attempt controller-001-job-001)
is preserved. Recovery owns only missing readback work under the original
7200s phase deadline (UNIX1789476142.59); no training replay, extra requests or
deadline reset is authorized. Frozen producer sources and the trial remain
unchanged; a separate portable recovery entry owns the scheduling repair.
Evidence:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/readback-recovery-v1/original-worker-reconciliation.json`.
Monitor88d5cb3b-e6e6-42cf-a56d-09d2c73591c6 was consumed exactly once with
view=decision; it witnessed the failure log after1418.89s, not success.

The full trial was released and launched on2026-09-15 at10:17UTC in tmux
`coordexp-coco227-ce-normalization-trial`, with S on GPUs0-3 and T on GPUs4-7.
Frozen trial file SHA256:
`b086ba7a2f4debd61f8db89e7fa563554a6b83efdb39c0063c81bc78560ba95a`.
Release, launch identity, logs and terminal receipts belong under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/trial-v1`.

Training qualification completed six two-update runs,132 logical exposures,
all exit0. The lead replayed every terminal validation and all four batched
comparisons;10 training/controller tests passed fresh. Microbatch2/3 failed the
frozen parity gate, so both scientific arms retain microbatch1 with activation
checkpointing. Mean max-rank forward/backward times were5.1235/4.8539/4.6934s.
For microbatch2, normalized global CE differed by at most1.65e-6, per-image CE
by8.56e-6, but raw per-image NLL sums and gradient norms exceeded the common
absolute gate. Microbatch3 also exceeded the relative gradient tolerance.
The fallback does not establish a batching bug; no tolerance was relaxed after
observing these results. Detailed fields:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/lead-admission-v1/batch-parity-breakdown.json`.
Full-trial admission also confirmed that arm manifests differ only in CE
reduction, run name/output root and their content digest. Receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/lead-admission-v1/training-trial-acceptance.json`.

Readback qualification is lead-accepted at batch3: live serial/batch2/batch3
used exactly33 requests, all token/prompt/media/grid identities matched the
retained common source, and all33 ended at EOS. Five focused tests passed in
the root replay. Generation times were158.004/143.757/140.370 seconds;
batch3 reduced this measured time11.16% versus serial, with10.535GB peak
allocated GPU memory. This is a single short qualification, not an end-to-end
throughput guarantee. Receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/lead-admission-v1/readback-acceptance.json`.

The lead admitted the CPU teacher and evaluator after six fresh focused tests,
literal old218 token/weight/trace/box preservation checks on all11 images, and
an exact replay of the common-source score. Receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/lead-admission-v1/data-evaluation-acceptance.json`.
This is not GPU execution or full-launch acceptance.

The expanded teacher has2176 active tokens. Common source0 covers218/227,
partitioned as old218=218/218 and new9=0/9, with no output-error debt.
The frozen teacher implies that token-equal CE gives image417044 (seven new
donuts,400 active tokens)2.022 times its sample-equal image weight. The other
two images containing additions receive1.582 and1.506 times their sample-equal
weights. These are objective weights, not evidence that either arm learns
better. Full per-image weights are recorded in
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/lead-admission-v1/ce-weight-design.json`.
