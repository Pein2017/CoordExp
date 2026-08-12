# Measurement Contract: harden-optimize-coordexp-swift-training-infrastructure

Status: frozen for candidate execution on 2026-08-09. Amendments require a
dated entry that explains why the earlier contract could not decide the slice;
results collected before an amendment remain labeled with the older contract.

## Decision rule

Correctness is a prerequisite, not a performance metric. A candidate is
eligible for promotion only when its semantic oracle is green, its receipts are
complete, it stays inside every resource ceiling, and its paired end-to-end
effect clears the practical noise band below. A local microbenchmark,
utilization increase, lower input-build time, or lower memory use cannot by
itself promote a production default.

The compatibility reference is the accepted implementation at commit
`2b0a2165a88499f0314572c5a73c9a308a990154`, replayed from the same current
code/checkout as the candidate through a private test or benchmark seam when a
mechanism changes. Cross-commit timings are provenance only. The reference
policies are:

- packing: `source_order_next_fit`;
- input provider: `synchronous`;
- attention: installed `flash_attention_2` route;
- resume: disabled;
- object/row order: resolved sorted order within each image is exact and
  immutable; deterministic cross-image and cross-batch reorder is allowed only
  in the Wave 6 comparators.

## Frozen workloads

### W0-CPU: cache preparation and admission

- Base config:
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml`.
- Train/eval sample limits: the committed `256/64` values; no sampling change.
- Seed: `17`; source-order input; `geo_sorted` intra-image row policy.
- One new temporary cache root per accepted comparison pair. The cache is built
  once, validated at payload level, then reused read-only by both arms.
- Cold preparation is measured from entry into
  `prepare_training_pack_caches` through both committed train/eval
  publications. Warm admission is measured separately after one unmeasured
  validation pass; the host page-cache state is reported, never silently
  described as cold.
- Shared `.cache/coordexp_swift/packing` is read-only evidence and is never a
  measurement destination.

### W0-GPU: production-shaped compatibility run

- Same base config and exact `256/64` examples, rendered tokens, pack stream,
  seed, model, adapter, losses, evaluation, and checkpoint policy.
- Eight ranks on eight otherwise-idle A100 80 GB GPUs. A temporary derived
  config changes only `run.name`, `run.artifact_root`,
  `training.max_steps: 5`, `eval.forward.steps: [3]`, and
  `checkpoint.steps: [5]`; scheduled eval therefore runs after step 3 and final
  checkpoint publication after step 5. Existing null fraction cadences remain
  unchanged. The derived config and resolved fingerprint are stored with the
  receipt.
- Steps 1-2 are warm-up. Steps 3-5 form the steady-state interval. Time to
  model load, first successful optimizer step, eval hydration/evaluation, and
  checkpoint publication are separate phases and are excluded from
  steady-state step wall clock.
- The measured step duration is the all-rank maximum already reduced through
  the training metric collective. Per-rank input build/wait values and skew are
  retained when available; a rank-zero-only resource value is labeled as such.
- A launch is invalid if another job has more than 1 GiB allocated or more than
  5% utilization on any selected GPU during the final preflight. Invalid
  launches are not performance observations and do not authorize disturbing
  the other job.

### Slice-specific workloads

- Wave 1 cache tests use synthetic/temp assets plus W0-CPU; no production cache
  write.
- Wave 2 FA2 parity uses the smallest real Qwen packed example that exercises
  every text layer, paired with separately executed segments and one corrupted
  boundary negative control.
- Wave 3 protected-loss comparison uses W0-GPU's fixed first measured pack and
  compares the zero-weight reference against the detached/no-grad candidate.
- Wave 4 eval hydration uses the fixed 64-example eval publication at eight
  ranks and preserves the accepted modulo rank assignment, canonical ordinals,
  and exact global denominators. Because every rank may reference every coarse
  chunk, retained/decode RSS and deserialized-step count are primary; chunk
  skips and bytes-read reduction are workload-dependent observations.
- Wave 5 provider selection uses W0-GPU and its exact cache/pack stream. The
  three distinct arms are `R`, the current synchronous CPU-build-then-transfer
  compatibility provider; `D`, the legacy fused device-direct path selected by
  no provider; and `O`, the depth-one bounded-overlap provider.
- Wave 6 planner CPU comparison uses the encoded lengths from W0-CPU; any
  surviving changed-order planner then uses the smallest matched five-step
  training comparison with exact per-image row order.
- Wave 7 resume compares one uninterrupted five-step run with a run interrupted
  after optimizer step 3 and resumed to step 5 at the same world size.
- Wave 8 reruns the accepted compatibility matrix without dependency upgrades,
  under pinned runtime-baseline schema 3 digest
  `cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`
  and the current model-free r2 native-runtime attestation.

## Semantic and numerical oracles

- Exact equality: config/cache/policy fingerprints, example and pack IDs,
  each-once coverage, intra-image row identity and order, token IDs, labels,
  segment boundaries, supervision masks, integer MRoPE positions, checkpoint
  inventory/lineage, counters, categorical statuses, and finite/non-finite
  decisions.
- Same-forward FP32 protected-loss diagnostics computed from one frozen logits,
  target, atom, denominator, and weight identity: `rtol=1e-5`, `atol=1e-6`.
- BF16 real-model supervised outputs, total loss, and per-term scalars derived
  across distinct BF16 forwards: `rtol=5e-3`, `atol=5e-3` after comparison in
  FP32.
- Same-forward genuinely FP32 gradients or parameter updates:
  `rtol=1e-4`, `atol=1e-5`. Wave 2 cross-forward gradients are classified by
  compute provenance rather than storage dtype: all 589 expected trainables are
  BF16-compute-derived and use `rtol=5e-3`, `atol=5e-3` after detached FP32
  conversion.
- Aggregate accuracy and loss statistics are compared from exact integer
  numerators/denominators where owned; derived floating values use
  `rtol=1e-6`, `atol=1e-8`.
- Any mismatch outside tolerance is a correctness failure. Tolerances are not
  widened after seeing a candidate result.

## Timing, ordering, and repetitions

- Use a monotonic clock. The end-to-end scope starts immediately before the
  public entry point and ends only after its terminal artifact is durable.
  Phase durations are nested within, not substituted for, this scope.
- Candidate comparisons run in fresh processes and alternate order `A/B`,
  `B/A`, `A/B`, where A is compatibility reference. Both arms use the same
  immutable cache publication and workload identity.
- The Wave 5 three-arm exception uses complete within-repetition triads in the
  predeclared order `R/D/O`, `D/O/R`, `O/R/D`, so every arm appears once in
  every launch position. Pairwise deltas are computed within each triad; `R`
  remains the compatibility reference.
- Three accepted paired observations are the repetition cap. At most one
  contaminated launch per wave may be replaced when the receipt proves an
  external-job collision, host failure, or incomplete artifact; a merely
  unfavorable or noisy result is not contamination.
- Report every observation plus median paired delta, min/max, and paired median
  absolute deviation. Do not report only the best run.

## Practical noise bands and promotion thresholds

- Steady-state optimizer-step wall clock: candidate must improve the paired
  median by more than `max(5%, 2 * paired_MAD/reference_median)`.
- Startup, cache preparation/admission, eval, or checkpoint phase: candidate
  must improve by more than `max(5%, 1.0 second)` when that phase is the
  decision owner and must not regress end-to-end wall clock beyond 2%.
- CPU/GPU memory: reductions below `max(5%, 512 MiB)` are descriptive only;
  any increase above that band requires a demonstrated end-to-end win and must
  remain under the hard ceiling.
- Packing utilization is descriptive until matched training is semantically
  green and clears the steady-state end-to-end threshold.
- If the confidence/noise rule is inconclusive at the repetition cap, retain
  the compatibility default and close the candidate as inconclusive or
  experimental.

## Resource and cost ceilings

- GPU memory: stop an arm before the next step if any rank reaches 76 GiB used
  or CUDA OOM occurs. No automatic batch/sequence reduction is allowed.
- Host memory: stop at 64 GiB RSS for one rank/process or 256 GiB summed across
  the eight training ranks.
- Temporary cache: 10 GiB per sample-limited publication; temporary run
  artifacts: 20 GiB per arm. Exceeding either stops the slice before expanding
  scope.
- W0-GPU launch wall clock: 30 minutes per arm. A three-pair Wave 5 comparison
  is capped at 24 GPU-hours, including one replacement launch. Other GPU waves
  are capped at 8 GPU-hours unless their task explicitly invokes the Wave 5
  three-arm matrix. The authorized final Wave 9 run is outside these probe
  caps but still requires its own frozen launch packet.
- No dependency install/upgrade, no production-cache mutation/deletion, and no
  eviction or termination of unrelated GPU work.

## 2026-08-11 production-path disposition

The lead has ended the Wave 3 through Wave 6 performance campaign before any
fresh successor marker was consumed. No further zero-weight efficiency,
eval-startup/resource, input-provider performance, or packing-performance arm
is part of this change's active execution plan.

- Wave 3 closes without memory/time/efficiency promotion; the protected-loss
  correctness implementation and immutable historical failures remain bounded
  evidence.
- Wave 4 retains the current correctness-tested rank-selective eval hydration
  behavior, with no RSS, I/O, startup, timing, or efficiency promotion claim.
- Wave 5 retains `synchronous`; alternative providers remain experimental and
  no result from shared load or a resource-stop packet selects a new default.
- Wave 6 remains pending for future design and matched training. Its CPU planner
  packet is descriptive only and `source_order_next_fit` remains the production
  default.

These dispositions supersede the repetition schedules and GPU budgets above
for Waves 3-6: those arms are not launched. They do not reinterpret missing
measurements as favorable results. The active costly path is the bounded Wave 7
exact-resume sequence, followed by Wave 8 compatibility and Wave 9 production
convergence. Shared-GPU timing/resource observations from that path remain
non-promotional.

## Required receipt fields

Every accepted observation records executed repository/dependency provenance,
resolved config and workload fingerprints, cache manifest/fingerprint, exact
comparison arm and private selector, world size/ranks/devices, seed, packing
and provider policies, phase timestamps/durations/statuses, warm-up and measured
step counts, per-step wall/build/wait values, CPU RSS and I/O high-water marks,
GPU allocated/reserved/device-use high-water marks, pack utilization, eval
coverage/denominators, checkpoint publication identity, semantic-oracle result,
and terminal/steady-state eligibility.

## Stop rules

Stop the current wave and do not advance when any mandatory semantic oracle
fails, provenance is unavailable for a decision-bearing comparison, dependency
or backend identity drifts, a required cache is stale/corrupt/unpublished, a
resource/cost ceiling is crossed, an artifact is incomplete, or a P0/P1 review
finding remains. Performance failure rejects/demotes that candidate; it does
not justify tuning the workload, widening tolerances, or changing the protected
intra-image sorted row order.

## Amendments

### 2026-08-09: executable provider arms and bounded hydration claim

An independent Wave 0 audit established that the shipped synchronous provider
already performs CPU construction followed by a distinct device transfer, so
the earlier labels `synchronous` and `two-phase synchronous` named the same
implementation. The provider comparison above now names three executable paths
(`R`, `D`, and `O`) and freezes a balanced triad order. The same audit proved
that modulo eval assignment can make every rank reference every coarse payload
chunk, so Wave 4 no longer assumes chunk-skip or I/O savings and preserves the
existing rank assignment. No Wave 4 or Wave 5 candidate result preceded this
amendment; the W0-CPU cache receipts are unaffected.

### 2026-08-09: dtype-band classification for Wave 2 cross-arm loss-term scalars

Wave 2's packed and separate arms are distinct BF16-autocast FlashAttention
forwards whose production Accelerator wrapper emits graph-connected FP32
logits to the loss path.
Every cross-arm per-term loss scalar (`raw_loss`, `weighted_loss`,
`segment_mean_numerator`, and `token_weighted_diagnostic`) is FP32 arithmetic
over those independently produced BF16-compute-derived FP32 logits, and the
`weighted_loss` values sum to the total loss assigned to the BF16-derived band.
Cross-arm per-term scalars therefore use the existing BF16 real-model band,
`rtol=5e-3`, `atol=5e-3`, after comparison in FP32. Every term and field remains
a mandatory gate; workload, sample identity, exact denominator semantics,
negative control, claim, and stop rule are unchanged.

For the fixed one-packed-context versus two-separate-context arms, exact
denominator semantics means exact term inventory, `term_name`, scope,
eligible/skipped segment counts, selected-atom counts, term weights,
normalizer/formula version, and resulting planned-step denominator.
`context_count` is validated separately as structural evidence and must be
exactly `1` for packed and `2` for shared-separate; it is not a cross-arm
semantic-equality field. This authenticated preflight must pass before
Accelerator construction, model loading, or either comparison forward.

The FP32 protected-loss scalar band, `rtol=1e-5`, `atol=1e-6`, is reserved for
same-forward-input comparisons in which only graph construction differs. Wave
3's zero-weight reference and detached/no-grad raw-diagnostic paths therefore
reuse one content-bound frozen FP32 logits tensor plus identical targets,
semantic atoms, denominators, and weights; a second model forward is not an
admissible strict-diagnostic reference.

No Wave 2 cross-arm numerical result preceded this amendment. The CPU
revalidation attempt stopped at `qwen.parity.cuda_device`, and the first GPU
attempt stopped at the shared-denominator precondition before any
`packed_clean`, `separate_reference`, negative-control, or comparison arm was
executed. The immutable failed GPU receipt has SHA-256
`ff8eb69913a7722b82009626937e5b4413e5181c38274e316757a62ab2196ed5`.
Its byte-identical durable copy is
`receipts/wave2-denominator-precondition-failure.json`; the compact provenance
record is `receipts/wave2-denominator-precondition-failure.md`. Both bind plan
SHA-256 `875750a17489fb674c65073bb10654e2434680e9423ca0a6aab2fddd512fb4af`,
terminal failure code `qwen.parity.denominator_mismatch`, and empty arm,
comparison, proof, negative-control, and measurement inventories.
No tolerance value is introduced or widened by this amendment; it assigns each
quantity to the already-frozen band matching the provenance of its inputs, and
neither band may be widened after a Wave 2 arm comparison is produced.

### 2026-08-10: conditional Wave 2 v3 reopening and simple FP32 comparison rule

The user selected one conditional versioned reopening after the immutable v2
gradient-gate failure and explicitly rejected a more elaborate numerical
oracle. V2 remains a terminal historical failure and is never re-scored. V3
binds parent plan SHA-256
`e5c2b1eb0c7ff99b7a66de8dc172f5af0331959d9b5f12a07e61096dd79b4197`,
parent plan-file SHA-256
`aadcfe046938315d90df05633b1da00d5a86e368eee3304b9d822183c1cdb681`,
whose byte-identical durable repository copy is
`receipts/wave2-v2-parent-plan.json`,
config fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`,
seed `17`, source indices `[0,1]`, example IDs
`coco2017_train_000000000009::aug:hflip` and
`coco2017_train_000000000025::aug:hvflip`, clean boundaries
`[0,1436,2822]`, separate boundaries `[0,1436]` and `[0,1386]`, corrupted
boundaries `[0,2822]`, and base-weight aggregate SHA-256
`e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa`.
No workload, dependency, model, adapter, delta, seed, order, supervision,
denominator, or negative-control substitution is allowed.

The current resolved runtime config is retained exactly as fingerprint
`0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1`.
Its sole allowed parent-compatibility projection removes the later Wave 5
strict default `training.forward_input_provider_mode: synchronous`; the
projected SHA-256 MUST equal the frozen parent fingerprint above. The
versioned attestation rejects a non-synchronous value, an additional removed
path/value, a wrong projected digest, or any other config drift. Ordinary plan
revalidation remains exact, and the live mode is reasserted immediately before
the attempt marker and GPU setup and recorded in the plan, marker, and receipt.

Qwen attention/linear work remains under production BF16 autocast. The frozen
production `Accelerator.prepare` route graph-connects `ConvertOutputsToFp32`
before `LossContext`; high-precision logits and loss arithmetic therefore run
in FP32 and gradients return through the cast to the BF16 upstream graph. The
probe preserves this production route. Only receipt/comparison snapshots are
detached, and those FP32 snapshots of supervised logits, total loss, all four
mandatory per-term scalars, and every expected gradient use elementwise
`torch.allclose(rtol=5e-3, atol=5e-3)`. Cosine
similarity, relative-L2, adaptive near-zero floors, result-dependent tolerance
selection, and storage-dtype tolerance selection are not acceptance metrics.
V3 does not recompute or publish a second legacy storage-dtype comparison; the
immutable v2 receipt is the complete historical record of that rejected gate.

The clean structural trainable inventory is 196 LoRA-A, 196 LoRA-B, 196 DoRA
magnitude, and one shared special-token delta parameter, for 589 total. The
model-free v3 plan binds these counts and strict owner/suffix classification
rules. After CPU model, adapter, and delta installation and before GPU setup,
the exact installed names, shapes, parameter storage dtypes, gradient dtypes,
and BF16 compute provenance are derived from the live model plus authoritative
adapter/delta receipts and bound into the attempt-start marker and terminal
receipt. Every row is present and finite with no extras in packed-primary,
packed-repeat, and separate-streaming arms. Exact-zero individual gradients are
allowed, but the complete inventory must contain a nonzero aggregate signal.

Packed-primary and packed-repeat restore identical model/RNG/input/train-mode
state, clear gradients exactly once immediately before their forward, and each
performs one backward without another clear in between. Their detached FP32 gradient inventories
must satisfy global `max_abs<=2.5e-3`; otherwise the observation is terminal
`unmeasurable`. This repeat gate cannot widen another tolerance or select the
better packed result. Both packed arms must independently pass the same
`5e-3/5e-3` elementwise comparison with the separate reference.
The process environment is frozen with `FLASH_ATTENTION_DETERMINISTIC=1` before
plan preparation and any runtime/model construction. The plan and terminal
receipt authenticate that value. It is used only to make the one-shot
diagnostic repeat decision-bearing and is not a production-default change.

The separate arm clears gradients once, then executes two ordered micro-steps,
each with forward, loss, and immediate `accelerator.backward`; the first uses
the production no-sync/accumulation semantics where applicable. It does not
retain and sum two differentiable graphs. Only detached loss scalars may be
summed for reporting. The boundary-only negative may omit backward and changes
only the FA2 boundary vector from `[0,1436,2822]` to `[0,2822]`; model state,
input IDs, images, position IDs, supervision, masks, loss wiring, and every
other field remain unchanged. It passes only when supervised logits or total
loss leaves the frozen BF16 band.

V3 uses plan schema `coordexp-swift-wave2-packed-parity-plan-v3` and receipt
schema `coordexp-swift-wave2-packed-parity-receipt-v3`. Every terminal status
(`passed`, `failed`, or `unmeasurable`) preserves completed evidence and exact
plan binding. After implementation, CPU mutation tests, and independent
P0/P1-free launch audits, at most one real-model v3 GPU attempt may enter model
GPU setup within the existing eight-GPU-hour ceiling. Immediately before that
setup, the command atomically publishes an absent-target immutable attempt-start
marker bound to the exact plan, command/source/dependency identities, and
receipt target plus the exact installed 589-row trainable inventory. Marker
publication consumes the attempt regardless of outcome;
a pre-existing marker rejects another invocation. A failure before marker
publication is not model evidence and requires a fresh audit, not a blind retry.
The terminal receipt references the marker. There is no automatic retry, sample
switch, tolerance edit, or v4 authorization. Wave 3 remains held unless the one
v3 receipt passes every mandatory gate.

### 2026-08-10: observed v3 result and narrow release decision

The sole v3 attempt is complete and consumed. Its immutable plan, marker, and
terminal receipt have file SHA-256 values
`141b4294dacae69f39907303d3ae75c4c4182d8da7625b399280ed99ed66187d`,
`8d1d16404814e95fa5cd9ced4062f9fd6e051ef870bde208d1f7e886b1231d48`,
and `781838ada548c9c0d9db4dacf1b43b6bc0e767303fa2906006c95a3ce70e6958`.
The receipt remains terminal `failed/qwen.parity.clean_failed`; it is never
rewritten or rescored.

Observed decision-bearing forward evidence passed: all 138 aligned supervised
rows, including every one of 152,670 logits per row, were byte-identical;
total-loss delta was `9.5367431640625e-7`; every mandatory per-term scalar,
denominator projection, semantic-atom key, and finite-value gate passed; the
28/28 text-layer deterministic FA2 varlen proof passed with clean boundaries
`[0,1436,2822]`; and the boundary-only negative changed supervised logits by up
to `35.125` and total loss by more than `2.48`. Packed-primary and packed-repeat
were bit-identical across all 589 gradients and 20,062,208 compared values.

The cross-shape packed-versus-streaming gradient comparison remains a retained
diagnostic failure. Exactly three tensors exceeded the frozen combined band,
while 393 signal-bearing rows exhibited deterministic BF16 reduction-order
differences and 196 cold-start LoRA-A rows were exactly zero in both arms. The
measurement therefore does not support exact cross-shape Jacobian equivalence;
nor does it support interpreting those numerical differences as segment leakage
when the complete supervised forward is byte-identical and the negative control
is decisive.

The user selected the narrow release rule after seeing this immutable outcome:
Wave 2 acceptance is scoped to forward/logit, objective/denominator,
boundary-isolation, and all-layer FA2 semantics. Cross-shape gradient numerical
equality is diagnostic only and cannot veto that scoped release. No tolerance is
widened, no result is recomputed, and no retry, sample change, or v4 is allowed.
Wave 3 may proceed only after future failed-receipt preflight evidence retention
is fixed, tests and strict validation pass, and independent audits find no other
P0/P1.

### 2026-08-10: current projection-v2 readers preserve historical execution

The executed Wave 2 v3 plan and failed receipt retain their historical
projection-v1 current fingerprint
`0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1`.
They are not regenerated. Current Wave 2 reader compatibility and active Wave 3
plan preparation use their respective config projection-v2 schemas and authenticate live
resolved fingerprint
`da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`.
Each projection removes exactly these later strict defaults before reproducing
legacy fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`:

- `training.forward_input_provider_mode: synchronous`;
- `packing.policy: source_order_next_fit`, `packing.window_size: null`,
  `packing.lookahead: null`, `packing.seed: 0`, `packing.worker_count: 1`,
  `packing.fragment_item_budget: 1024`,
  `packing.fragment_byte_budget: 4194304`,
  `packing.cursor_byte_budget: 65536`, and
  `packing.max_packs_per_fragment: null`;
- `resume: {checkpoint_dir: null, mode: disabled}` as one exact object.

Missing, extra, reordered, or changed path/value rows and either digest drift
fail closed. The immutable Wave 2 v3 failed receipt has one narrower reader
exception: only the known plan hash plus exact plan and receipt payloads at
`failed/qwen.parity.clean_failed` and stage `comparisons` may omit the later
`execution.requested_device` and `execution.gpu_idle_preflight` fields. Current
rich receipts still require those fields. This amendment changes reader
compatibility only; it does not rescore the failed receipt or authorize another
Wave 2 launch.

### 2026-08-10: Wave 3 host-watchdog failure and weight-bound replacement

Wave 3 r2 is an immutable incomplete technical artifact, not an unfavorable
model result. Its worker completed CPU model/adapter/delta installation and the
exact 589-row trainable inventory, then the external host watchdog rejected a
terminal zombie/short-lived descendant whose `/proc/<pid>/status` legitimately
lacked `VmRSS`. No model forward or backward ran and no comparison was emitted.
The repaired watchdog skips a PID only after bounded recheck proves it vanished
or entered terminal `Z/X/x`; live missing or malformed `VmRSS`, permission or
other I/O failures, and the 64-GiB ceiling remain fail closed.

The first r3 prelaunch plan used the old v2 artifact schema and was rejected
before marker publication because it did not bind model weight content. It is
preserved at
`outputs/probes/coordexp_swift/wave3_zero_weight/2026-08-10-r3/plan.json`
with file SHA-256
`035857210f78452ad6567d909f37616dad6f87bd2d2fd46c8dc3817eb17b7f37`;
it is non-executable and did not consume the sole replacement run.

The sole eligible replacement used active v3 plan/marker/receipt schemas under
`outputs/probes/coordexp_swift/wave3_zero_weight/2026-08-10-r3-v3`. Its plan
binds frozen base-weight aggregate
`e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa`
plus the complete index/shard inventory, and the worker rehashed it before
model load. Plan, marker, and terminal-receipt file SHA-256 values are
`f4936e01729722ce297f9d9a09bd791967fcb741ada827d32b1321deaa86288e`,
`27b2b957d6f96b9a454d3394773fbc65672b673824e14b45c0f5373847c9b297`,
and `6283baa2b72c0e730bb671179980423564188b9d910778cf29ba66042298958c`;
their internal hashes are
`0d2f9c575968ab1f79c530ac6a501190b183fc6e29fd835cbeb5ce555f98a90a`,
`3df128f6ea3b4c12cf822dbcb5258f5a0da0f74d58a7a6bd46a92e09fbf400d7`,
and `b0be5a695abfb4c657d60afba82988f8381159664859e62ea19b55ace0dd148e`.

The terminal status is `failed/wave3.accelerator` after the exact completed
prefix `plan_revalidated`, `gpu_idle_preflight`, `model_setup_cpu`, and
`attempt_started`. It records the installed 589-row trainable inventory and
full weight identity, but zero model forwards, zero backwards, no arms, and no
oracle/comparison/efficiency result. The selected cache manifest and chunk were
independently rehashed unchanged after the run. The direct Accelerate 1.10.1
one-process default represented the selected device as indexless
`torch.device("cuda")`, which failed exact equality against
`torch.device("cuda:0")`. Current source closes that seam by CPU-only
pre-marker state admission and exact device-environment pinning, followed by a
strict post-marker Accelerator identity check; CPU tests and independent audits
pass, but this remediation has not been rerun on CUDA and is not measurement
evidence. Marker publication consumed the sole replacement. At that 2026-08-10
fixed point there was no r4, retry, sample switch, workload change, or tolerance
change, and tasks 4.4 and 4.5 remained open. The 2026-08-11 production-path
disposition above later closed them without performance promotion or another
launch.

That sentence records the immutable 2026-08-10 fixed point. On 2026-08-11 the
user explicitly authorized continuing the remaining smoke runs on the current
shared eight GPUs, which authorizes exactly one fresh-root Wave 3 v4 successor
without mutating or retrying the consumed v3 root. Before its marker, the v4
plan MUST rebind the live source/config/cache/runtime/full-weight identities and
the selected device's stable shared-process baseline. After every spawned
worker outcome, including timeout, watchdog, or worker failure, it MUST perform
bounded session-wide cleanup and then exactly two subset sweeps at least two
seconds apart. A survivor, new GPU row, incomplete cleanup, or incomplete sweep
makes the one-shot receipt failed. Shared-load time, utilization, memory, and
efficiency observations are non-promotional; only numerical, correctness,
plumbing, and cleanup evidence may be admitted from this run.

The fixed r2 v2 chain remains readable only as `historical_non_executable`;
active entry points reject v2 before CUDA, child-process, cache/model, marker,
receipt, or sidecar effects.

### 2026-08-10: Wave 8 schema-3 native admission fixed point

Wave 8 decision-grade compatibility now binds pinned runtime-baseline schema 3
digest
`cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`.
The current minimal CUDA initialization evidence is
`outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json`,
whose internal receipt SHA-256 is
`829d86ec8d977f30d37e8e979acba6527b3640180825e551785078a4876b1043`.
It attests mapped native runtime identity only and is not a model, cache,
training-quality, or throughput observation. The earlier r1 receipt remains
historical pre-owner-expansion evidence and cannot satisfy current admission.

### 2026-08-11: Wave 7 deterministic-r5 pre-result contract

Wave 7 r4 is preserved as an immutable failed comparison, not retried or
rescored. Its comparison receipt is
`outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-10-r4/exact-resume-comparison-receipt.json`,
with file SHA-256
`1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b`
and internal canonical payload SHA-256
`13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a`.
The receipt proves clean eight-rank interruption, exact cursor/RNG/control
readback, append-only lineage, and BF16 state continuity within the already
frozen `rtol=5e-3`, `atol=5e-3` band. It does not pass task 8.7: independent
reference and parent launches differ in owned integer accuracy numerators at
step 3, before resume, so the observed difference cannot be attributed to
restore.

The sole new r5 hypothesis is that the pre-resume drift is caused by the
previous unbound nondeterministic CUDA/FlashAttention backward route. The
prospective treatment is a supported exact-resume runtime policy named
`strict_cuda_replay_v1`. Every exact-same-world-size run MUST require and attest
all of the following before Accelerator, model, or CUDA materialization:

- launcher-provided `FLASH_ATTENTION_DETERMINISTIC=1`;
- launcher-provided `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `torch.use_deterministic_algorithms(True, warn_only=False)`;
- `torch.backends.cudnn.deterministic=True` and
  `torch.backends.cudnn.benchmark=False`;
- exact rank/world/device mapping, seed, pinned runtime-baseline identity, and
  rank consensus for the resolved policy.

The compatibility default for non-resume runs remains `legacy`; exact resume
is rejected unless `strict_cuda_replay_v1` is selected. This changes runtime
qualification, so no r4 timing or performance observation transfers to r5.
It does not change the model, data, augmentation, image/object order, pack IDs,
tokens, objective, optimizer schedule, seed, checkpoint cadence, numerical
tolerances, or exact integer numerator/denominator oracle.

Before the r5 sequence marker, CPU tests and a two-launch eight-rank synthetic
CUDA preflight MUST prove exact policy/env/rank admission and exact digests for
fixed BF16 CUBLAS, deterministic FA2 forward/backward, and one fixed collective.
The synthetic probe is plumbing evidence only, not model-training evidence.

CPU-heavy preparation and validation MUST use bounded multi-core execution
whenever the production owner supports semantics-preserving parallelism.  In
particular, the fresh r5 pack-cache materialization MUST use the authenticated
`fork_process_pool` route with more than one worker (the current frozen default
is 16 on the 112-logical-CPU host), record the resolved strategy and worker
count in its manifest/receipt, and reject a silent fallback to one worker.
Parallel workers MUST restore exact source/example order and preserve every
single-image object/row ordering, token, supervision, pack-plan, and cache
digest.  Output-affecting worker settings remain semantic determinants unless
exact worker-count equality has been executed and authenticated; serial-only
owners such as atomic publication remain serial rather than being presented as
parallel work.  Independent CPU-only hash, validation, and test partitions MAY
run concurrently when their artifact targets and mutable process state are
disjoint, and their receipts MUST retain the exact worker/concurrency scope.

Complete base-model weight admission MUST likewise hash independent,
index-declared weight payload files with a bounded thread pool after serial
index/layout preflight. Results MUST be collected in canonical shard-path order
so the existing model-weight identity and aggregate digest remain unchanged,
and the model-input attestation MUST record the hashing strategy, resolved
worker count, and payload-file count. The current two-shard model therefore uses
two workers. A genuine standalone one-file payload may use one worker because
there is no independent file-level unit to parallelize; this exception MUST NOT
permit a multi-shard fallback to one worker.

On 2026-08-11 the user explicitly authorized the r5 correctness/plumbing smoke
to share all eight GPUs with already-running work.  A new r5 attempt therefore
MAY admit a bounded, stable pre-existing compute-process baseline instead of
requiring globally idle devices.  Before its marker, two samples at least two
seconds apart MUST bind the exact eight-GPU index/UUID inventory and the exact
pre-existing `(gpu_uuid, driver_pid)` set, while each 81920 MiB GPU has at most
49152 MiB already allocated, preserving at least 32768 MiB of headroom for the
owned smoke.  Utilization is observational under this shared mode.  After
every owned launch and before terminal publication, every observed compute row
MUST be a subset of that frozen pre-existing set; an existing job may finish,
but any new row or surviving owned process/session fails the one-shot attempt.
The controller MUST NOT signal, reset, or otherwise mutate the pre-existing
jobs.  Shared-mode receipts support only deterministic plumbing, numerical,
artifact, interruption, and exact-resume correctness claims.  Their wall time,
GPU utilization, and memory observations are contaminated and MUST NOT support
provider, packing, zero-weight, startup, or throughput promotion; those
performance decisions retain their original matched/otherwise-idle gate.

An earlier same-day authorization allowed optional fresh-root Wave 4 parity,
Wave 5 provider-plumbing, and Wave 6 matched-training correctness successors
under this shared baseline. The later production-path disposition above retired
those optional launches before their markers were consumed. Historical roots
remain immutable and non-executable; no shared-load observation is reused to
close or promote the omitted performance questions.

R5 is one prospective, at-most-once sequence. One absent-only sequence marker
MUST bind this dated amendment, the immutable r4 failure, all source/config and
runtime identities, a fresh private cache, model weights, three absent run
roots, every command and timeout, the pre-child and final comparison targets,
and the unchanged semantic/numerical oracles. The fixed order is:

1. one uninterrupted strict run;
2. one strictly controlled parent interrupted after the durable step-3
   checkpoint event;
3. one read-only pre-child gate;
4. only if that gate passes, one exact-resume child;
5. one final comparator-v2 decision.

The pre-child gate MUST authenticate the parent marker/termination receipt,
step-3 manifest and all eight ranks, optimizer-step boundary, cursor/RNG/control
state, inference-payload manifest, committed progress, logs, exact integer
`top1_correct`, `top5_correct`, and `atom_count`, absence of step 5/final/late
writes/surviving processes, the child config and command, and a stable parent
tree snapshot. The child launcher MUST verify the immutable gate-receipt digest
before any child CUDA or model setup; a human ordering convention is not
sufficient.

Publication events in r5 bind checkpoint-local self-authenticating inference
payload manifests and atomically persist checkpoint-bound progress. Comparator
v2 treats explicit wall-time fields as observations, compares best-checkpoint
selector/path/step separately from metric values, and consumes persisted
integer accuracy sufficient statistics directly. It MUST NOT reconstruct r5
numerators from floating ratios or widen any tolerance. The historical r4
reader may diagnose its legacy rows but cannot relabel r4 as passed.

Publishing the r5 sequence marker consumes the attempt. Any env, source,
schema, rank, operator, numerator, checkpoint, inference, lineage, process,
late-write, or comparison failure publishes a terminal failed receipt and
stops all later arms; there is no selective retry, root switch, seed/sample
change, field deletion, or tolerance change. If the step-3 gate fails, the
child is not launched. If strict r5 still fails exact pre-resume numerators,
tasks 8.7 and 8.9 remain open; a common-step-3-ancestor contrast would be a
different user-owned estimand and requires a separate prospective decision.

### 2026-08-11: consumed r5 outcome and successor authority boundary

The r5 attempt is consumed by this immutable chain:

- plan file SHA-256
  `ea3cef1d96412cd3f425e9c7f0c0cd562ceafa2fe59c39c225755898dbb66a1f`,
  internal canonical payload SHA-256
  `73c502d69ec8b999a9e451620702b54eb1d573bf775e18d32edde08d216f53c4`;
- sequence-marker file SHA-256
  `58aa7425dc6c019a23e790ba072495f0fb787b805eabaf922f079c35eddb19b5`,
  internal canonical payload SHA-256
  `a620caa0e64a12be980120e310a914571c78ee0889d6da0760aae208f3c82224`;
- terminal-receipt file SHA-256
  `d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1`,
  internal canonical payload SHA-256
  `f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656`.

The uninterrupted phase terminated `rc=1` after 20.094 seconds. The controlled
parent, pre-child gate, resume child, and final comparison are all
`not_started`. Bounded cleanup and recovery passed, so the terminal receipt is a
valid one-shot failure record; it is not permission to retry.

CPU-only reproduction identifies the failing seam without extending the
measurement claim: invoking the absolute `src/train.py` path without the
repository root in `PYTHONPATH` raises `ModuleNotFoundError`, while module
invocation succeeds. No model or cache was reached and no training arm ran.
Consequently r5 is diagnostic-only evidence for a launcher/entrypoint plumbing
failure; it supplies no model, cache, training, interrupted-versus-resumed, or
exact-resume result and does not complete tasks 8.2, 8.7, 8.8, or 8.9.

A successor source fix is in progress, but r6 is not authorized. Any new
attempt requires explicit user authorization plus fresh absent roots, runtime
attestation, deterministic preflight, request, immutable plan, marker, and
private cache. It retains r5's no-retry rule, cost ceiling, shared-load
restrictions, and diagnostic claim boundary. Wave 8 tasks 9.3 and 9.5 therefore
remain blocked on an accepted Wave 7 gate. Nothing here reopens the Wave 3-5
performance campaign or promotes Wave 6: their no-promotion/default-retention
dispositions remain intact, Wave 6 matched training remains pending, and
`source_order_next_fit` remains the production default.

### 2026-08-11: Wave 7 r6 one-shot successor authorization

Later on 2026-08-11, the user explicitly authorized one fresh r6 one-shot successor.
This later authorization supersedes only the prospective no-authority
statement immediately above; it does not alter or reinterpret the immutable r5
failure. r6 is not an r5 retry. It is exactly one prospective successor under a
fresh r6 run/cache/runtime/preflight/request/plan namespace, with absent-only
publication roots, a fresh private cache, and its own at-most-once marker.

The r6 authority uses amendment schema
`coordexp-swift-wave7-r6-amendment-v3`, request schema
`coordexp-swift-wave7-exact-resume-sequence-request-v5`, plan schema
`coordexp-swift-wave7-exact-resume-sequence-plan-v5`, marker schema
`coordexp-swift-wave7-exact-resume-sequence-marker-v5`, and terminal schema
`coordexp-swift-wave7-exact-resume-sequence-receipt-v5`. The request retains the
existing `legacy_r4_failure` binding and adds exactly one
`predecessor_sequence_failure` binding to the immutable r5 terminal. That r5
binding is copied unchanged from request to plan and is historical,
non-executable evidence; it cannot satisfy a current terminal, request, plan,
marker, or launch authorization.

Each r6 phase runs at most once with a timeout of at most 600 seconds
(`<=600s/phase`). The complete sequence is bounded to at most 2400 seconds wall
time (`<=2400s wall`) and at most 14400 GPU-device-seconds
(`<=14400 GPU-device-seconds`). Shared-GPU observations remain non-promotional.
Failure at any gate publishes terminal evidence and stops: no later phase may
start, there is no retry or root switch, and there is no automatic r7.

This one successor does not reopen the Wave 3-5 performance campaign or promote
Wave 6. Their no-promotion/default-retention dispositions remain intact, Wave 6
matched training remains pending, and `source_order_next_fit` remains the
production default. Wave 8 tasks 9.3 and 9.5 remain blocked until r6 passes the
Wave 7 gate.

### 2026-08-11: Wave 7 r6 immutable consumed preflight failure

R6 is consumed and failed in deterministic preflight after
`KeyboardInterrupt`. It never reached model loading, training, or sequence-level
request/plan/marker/terminal publication. Its immutable precursors completed
before preflight. The passed config-bundle receipt has file/payload SHA-256
`82214495b1e75e44155303a28e456d6269942fde926959ce846d5c03fdb6aa89` /
`6f4a16b524701dac91cf03a1ac802e590766f1a4ab0881023a400a339809dd3c`,
with semantic-projection SHA-256
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
The canonical private-cache root is
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-11-wave7-r6`; its passed
preparation receipt has file/payload SHA-256
`52df7e0b3bcca23fcfd5bde69f59f7a97030bf455d704a2009e19a6d22a74c3b` /
`bd9a8e4b937f000547d7f7de676bd642773c03edbedd7c5b0fdfee3c0bf9f718`.
The production payload path used `fork_process_pool` with 16 workers. Train has
fingerprint
`8307cf2dd9cac344b8a50bb783a831d683c3686ae73678425d6245e9df9eae87`,
manifest SHA-256
`775c0c87f6c6e63b11f17c4c737c7649cadb2e0b20494b02d4694274d2ad3567`,
and one 32-micro-step chunk SHA-256
`59a38ddb036ae644e15bfb25db022dc7b28ca82216679262bf6af44824ba8956`.
Eval has fingerprint
`881fe84188657eeb705dd1bf2b754f11b9f69d61eb62a83954fc867a9d1744f8`,
manifest SHA-256
`dce7449e1267a9217166b6297249856d94f0def88fd9cb7fd7c3422f09a15634`,
and one 8-micro-step chunk SHA-256
`047135041298ed869a078d78f22a3cf54d949963950ffe3dedc54c120e910ca3`.
Production payload loading passed and the tree remained stable. This completed
cache evidence is bound by two repeated seven-file snapshots with identical
tree SHA-256
`943fdbc325e143dbb8748c97f905f95fecd11836d6f6039b728df8b816a3de85`
and makes no model or training claim. The passed runtime receipt has
file/payload SHA-256
`51822d203651799cf44279c85b2de64d50a2c2a5ba2ad3e8513ab1f2a22f0469` /
`3b06e0ce6eefa31fb414b6d8c9eb50c8a2bdbd2f0ca448da71fbdbdb929fd0d2`.

The immutable preflight chain is:

| Artifact | File SHA-256 | Internal canonical payload SHA-256 |
|---|---|---|
| determinism-preflight plan | `2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` | `9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f` |
| attempt-start marker | `f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` | `967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e` |
| failed terminal receipt | `eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` | `4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab` |

Marker publication consumed the attempt. The failed terminal records
`mismatches=["KeyboardInterrupt"]`, `launch_count=0`, no bound rank receipts,
and no comparison. Direct leaf inspection finds exactly eight passed
`launch-a` receipts for ranks 0-7, each with workload aggregate SHA-256
`41f57c8c5ff14d25b1c7c20711f53becec5853fadcfe68dd0c116ec8fd1d9422`,
and no `launch-b` receipt. Those leaves are supplemental plumbing evidence only:
the immutable terminal cannot incorporate them, the required two-launch
comparison did not reach a fixed point, and r6 is not a preflight or Wave 7
pass.

The superseding post-failure repair identities are:

- `scripts/probes/coordexp_swift/wave7_determinism_preflight.py` SHA-256
  `52d6e2a3b709ede73f5794eefdbfd6c57997714af29f5329e936b8953492346c`;
- `tests/training/test_wave7_determinism_preflight.py` SHA-256
  `8af9a58d1f425c13bd5e25ffeaca3ab60865b9dcf650d0469abb846be48279a8`;
- `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py` SHA-256
  `cadcbf5ec413b4eea7314684181b4a3ef514144212f9814e3f58d2b999ddfaea`;
- `tests/training/test_wave7_exact_resume_sequence.py` SHA-256
  `ae0f1615410cce187722e1473fed93908033e94b1b6ed6dc98caf4c6df41c236`.

An independent read-only audit returned PASS for this repair set. That PASS is
future-authorization engineering evidence only: it neither rewrites the r6
terminal nor admits Wave 8 or an automatic r7. Wave 3-5 remain no-promotion;
Wave 6 matched training remains pending, and `source_order_next_fit` remains the
production default.

### 2026-08-12: Wave 7 r7 one-shot successor authorization

The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot
successor. This is not an r6 retry and does not mutate or reinterpret any r6
artifact. Its absent-only sequence and private-cache roots are
`outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-r7` and
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-r7`. The
sequence root owns fresh private `runtime/`, `determinism-preflight/`,
`amendment-v4.json`, `request-v6.json`, `sequence-plan-v6.json`,
`sequence-marker.json`, and `sequence-receipt.json` namespaces; no path may
switch to or reuse an earlier root.

The prospective packet schemas are
`coordexp-swift-wave7-r7-amendment-v4`,
`coordexp-swift-wave7-exact-resume-sequence-request-v6`,
`coordexp-swift-wave7-exact-resume-sequence-plan-v6`,
`coordexp-swift-wave7-exact-resume-sequence-marker-v6`, and
`coordexp-swift-wave7-exact-resume-sequence-receipt-v6`. Request and plan MUST
carry unchanged:

| Binding | Immutable artifact | File SHA-256 | Internal canonical payload SHA-256 |
|---|---|---|---|
| `legacy_r4_failure` | r4 exact-resume comparison receipt | `1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b` | `13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a` |
| `predecessor_sequence_failure` | r5 sequence terminal | `d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1` | `f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656` |
| `predecessor_preflight_failure.plan` | r6 deterministic-preflight plan | `2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` | `9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f` |
| `predecessor_preflight_failure.attempt_marker` | r6 deterministic-preflight attempt marker | `f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` | `967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e` |
| `predecessor_preflight_failure.terminal_receipt` | r6 deterministic-preflight terminal | `eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` | `4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab` |

R7 retains the fixed one-shot phase order and unchanged semantic/numerical
oracles. Every phase runs at most once with timeout `<=600s`; total cost is
`<=2400s wall` and `<=14400 GPU-device-seconds`. Admission retains the shared
eight-GPU envelope of exactly 81920 MiB total, no more than 49152 MiB
pre-existing allocation, and at least 32768 MiB headroom per device. Shared-load
timing, utilization, memory, and throughput are non-promotional. The first
failed gate publishes terminal evidence and stops all later phases. There is no
retry, root switch, or automatic Wave 7 `r8` successor.

The user's 2026-08-12 instruction already authorizes continuation in the order
Wave 7 -> Wave 8 -> Wave 9, while task 10.3 already authorizes the final
cache/convergence work after its gates. No additional user authority is imposed
for Wave 8 or Wave 9. Wave 8 starts only after the accepted r7 Wave 7 gate and
with its own frozen packet and budget; Wave 9 starts only after the accepted
Wave 8 gate and with its own sealed packet, frozen identities, and budget. R7
does not waive either later gate or transfer its packet or budget downstream.

### 2026-08-12: Wave 7 r7 immutable consumed cache-admission failure

The r7 config-bundle gate passed. Its receipt file SHA-256 is
`f07e4bc9f4e272e6ed831c714b55f8c1c7a6ee721a86c691e00e5cd3eee34ac6`,
its internal payload SHA-256 is
`a72c97856ce2689077c91a19652aca3e6fd6b914c8dbce15bfac9008da46ebdb`,
and its semantic-projection SHA-256 is
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
The frozen config files are:

| Role | File SHA-256 |
|---|---|
| uninterrupted | `88ccc01f69deab5242405593032c31ff8374c455f08247da20322e3709be2fc4` |
| interrupted parent | `f9441f5a9ccb27635289a49739b4449c486770b970736549ea9b4dfee4fdf997` |
| resume child | `8bec80b9a5eaefcd9212968b3f9dd8d38eb0601308791a686a2ff622d9dcb9f8` |

The first real preparation invocation stopped before publishing a receipt
because the fresh private-cache parent did not yet exist. After creation of that
exact authorized parent, the production preparation command ran without the
required strict determinism environment and atomically published a failed
preparation receipt at
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-r7/preparation-receipt.json`.
Its file SHA-256 is
`9077a0458bece11b630ab90c68a9811dfca2941171b0575717b3dba670ae817d`
and internal receipt SHA-256 is
`e7d7d5c4f2d6cf35e823c80faed4c9391f0368449f6be27560db9f628f8bcc2b`.
It records schema `coordexp-swift-pack-cache-preparation-receipt-v1`,
`terminal_status="failed"`, error type `RuntimeContractError`, error code
`runtime.determinism_environment_conflict`, and `result=null`.

This atomic failed admission is the r7 terminal disposition. The private-cache
root contains no cache payload files. The r7 sequence root contains only the
three configs and passed config-bundle receipt: runtime, deterministic preflight,
request, plan, marker, sequence terminal, model load, CUDA, training, comparison,
and post-run evidence were never reached. The first failed mandatory admission
therefore stops the sequence with no retry, root switch, or automatic Wave 7
`r8` successor. R7 is not a Wave 7 pass and supports no cache, model, training,
exact-resume, or performance claim. The already-authorized Wave 7 -> Wave 8 ->
Wave 9 route remains blocked at the Wave 7 gate; Wave 8 and Wave 9 retain their
own packet and budget requirements but cannot start through this failed chain.
