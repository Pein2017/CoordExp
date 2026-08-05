# Design: Streamline CoordExp-Swift Base Infrastructure

## 1. Current Execution-Path Ownership (inspected at commit a19ffceeb)

This section records the as-inspected baseline; implementation MUST revalidate
the live HEAD and call graph before each wave (tasks 0.1, 5.1) rather than
treating this commit as authority.

- `src/train.py` -> `src/training/pipeline.py` assembles config, Qwen
  components, DoRA adapter, embedding deltas, pack cache, schedule, losses,
  optimizer/scheduler, `TrainRuntime`, eval handler, checkpoint handler; then
  `src/training/supervised_trainer.py` runs planned steps.
- The streaming trainer path first moves all micro-steps of the planned step
  (`supervised_trainer.py:258-270`), then forwards them one by one
  (`:285-321`) and MAY break early when the per-micro-step all-rank
  pre-backward finite gate votes unsafe. Any lookahead design must respect
  this planned-step pull-then-consume shape and the early-break exit.
- Forward-input construction is owned by
  `src/qwen/forward.py:build_qwen_forward_inputs`, called synchronously per
  micro-step from `_default_qwen_forward`; image pixels are materialized there
  from lazy plans (`src/qwen/images.py`), including the content-SHA256
  identity check, on the trainer thread.
- Rank pack loading is owned by
  `src/training/pack_cache.py:load_rank_micro_steps_from_cache`; distributed
  admission is `manifest`-level and the eager rank load is the single
  digest-and-payload pass. `_rank_local_pack_indices` derives the rank's
  required indices from the resolved schedule before any payload read.
- `eval.forward` (`src/eval/forward.py:ForwardEvalRunner`) currently loads all
  eval micro-steps on every rank; the cross-rank metric reduction
  (`TrainRuntime._reduce_metric_reports`) is a plain rank mean and is exact
  only because every rank computes identical values.
- Loss streaming path: `LossRunner.prepare_planned_step` gathers global
  denominators once per planned step; `compute_micro_step` applies
  `backend_gradient_scale = world_size`; `finalize_planned_step` merges scalar
  artifacts. Critical reduction fact (drives Seam D): `plan.counts` is built
  from the globally merged base denominator
  (`src/losses/runner.py:252-265, 885-900`) and `finalize_planned_step` writes
  those global counts into the metrics mapping (`:364-395`). Therefore the
  metric `count/supervised_atoms` is identical on every rank and is NOT a
  valid per-rank weight.
- A parallel non-streaming loss path (`LossRunner.compute`, trainer batch
  branch, eval batch branch) exists with no production caller.
- Finite gates: per-micro-step all-rank scalar consensus before backward and a
  per-planned-step gradient gate (`src/runtime/finite_gates.py`), gathered via
  the bounded gloo control-plane collective.
- Run artifacts: rank-zero `RunWriter`; `logging.jsonl` rows carry no timing
  fields; `run.json` is the compact run record.

## 2. Seam A: Bounded Forward-Input Lookahead

### Concept and owner

A `ForwardInputProvider` protocol owned at the trainer/pipeline boundary. The
trainer owns step boundaries explicitly; the provider owns preparation. Two
implementations satisfy the same protocol: a synchronous provider (semantic
reference, builds on demand) and the overlapped provider (depth-one
lookahead). Trainer code changes where lifecycle ownership requires it; a
stateful step-boundary-blind global `qwen_forward` closure is explicitly
rejected because it cannot see early termination.

### Interface (what callers know)

- `begin_planned_step(planned_step_id, moved_micro_steps)`: the trainer hands
  the provider exactly the moved micro-step sequence of the current planned
  step after the existing move loop. The overlapped provider starts preparing
  from ordinal 0.
- `take(ordinal, micro_step) -> QwenForwardInputs`: returns the prepared
  CPU-resident inputs for that ordinal; asserts ordinal and `pack_index`
  identity against the micro-step actually being consumed and fails closed on
  skew. The synchronous provider builds here directly.
- `end_planned_step()`: called on every step exit — normal completion,
  finite-gate early break, or exception (trainer `finally`). Cancels the
  producer's remaining work for the step, drains and discards unconsumed
  prepared items, and leaves the provider idle. No provider state survives
  into scheduled eval/checkpoint handlers or the next planned step.
- `close()`: bounded join of the producer thread; called from the same
  `finally` scope that closes the rank-report gatherer.

### Concurrency and lifecycle model

- One producer `threading.Thread` and one `queue.Queue(maxsize=1)` per
  provider instance: at most one prepared micro-step beyond the executing
  one; host memory bounded by two micro-steps of materialized inputs.
- Cancellation-aware handoff: producer `put` and consumer `get` both poll a
  step-scoped cancel event with short timeouts, so a full queue can never
  deadlock `end_planned_step()` or `close()`; the producer observes
  cancellation between and during handoffs and abandons the current step's
  remaining ordinals.
- Producer performs CPU-only work: file read, SHA256 identity check, PIL
  decode, flip transform, HF image processing, CPU tensorization, FA2 plan on
  CPU. It never touches CUDA; device transfer stays on the consuming step, so
  stream/allocator semantics are unchanged.
- State machine per provider: `idle -> active(step) -> idle` with `active`
  entered only by `begin_planned_step` and left only by `end_planned_step`;
  `take` outside `active`, ordinal skew, or a second `begin` without `end` is
  a `RuntimeContractError`.
- Error propagation: a producer exception is captured with its ordinal and
  enqueued as a poison item; `take` for that ordinal re-raises the original
  exception — the same micro-step at which the synchronous path would have
  failed. Earlier ordinals complete normally; the producer stops for the
  step.
- Covered transitions (each with a dedicated test): normal completion;
  producer error; consumer error (trainer exception mid-step); finite-gate
  early break (remaining prepared work discarded, next step clean); scheduled
  eval/checkpoint between steps (provider idle, no interaction).
- Determinism: materialization has no RNG; both provider modes produce
  byte-identical tensors. The image content-SHA256 check runs in the producer
  with unchanged fail-closed behavior.
- Eval reuse: the eval loop MAY use its own provider instance with the same
  protocol; instances never share queues or threads.
- Mode selection and receipt: the synchronous mode is selectable through a
  debug-only switch (not YAML); the resolved provider mode is recorded once
  in the compact run record so any run's mode is auditable despite the switch
  being non-config.

### Alternatives compared

- Pipeline-global producer walking the full rank-local tuple: rejected — the
  trainer's early-break exit leaves the producer holding an unforwarded
  micro-step, producing ordinal skew or stale work at the next planned step.
  The planned-step-scoped lifecycle removes this class of bug structurally.
- Process-pool producer: rejected — tensor serialization/shared memory,
  duplicated processor state, weaker fail-closed identity handling; decode,
  hashing, and numpy release the GIL for the dominant cost.
- Deeper or configurable queues / DataLoader adoption: rejected per the
  bounded-host-memory requirement; depth one already hides one step's build
  time.
- Materializing pixels into the pack cache: rejected on prior
  `stream-distributed-pack-cache-runtime` evidence (payload and admission
  cost explode; identity checks weaken).

### Failure modes and receipts

Producer failure surfaces as the original `EncodingContractError` /
`QwenForwardContractError` at the affected step. `input_build_seconds` and
`input_wait_seconds` (Seam E) are the steady-state receipts that overlap is
effective; the run record carries the provider mode.

## 3. Seam B: Rank-Selective Chunk Loading

### Contract

`load_rank_micro_steps_from_cache` already knows `required` indices before
touching payloads. New behavior: a chunk whose declared `[start, end)` range
does not intersect `required` is skipped without reading its payload bytes.
Verification contract, explicitly:

- Manifest-level validation of every chunk declaration (contiguity, counts,
  digest syntax, safe paths, file existence) is unchanged and still covers
  skipped chunks.
- Every chunk whose payload is read MUST first match its declared SHA256 and
  MUST pass restricted unpickling and type/count validation, exactly as today.
- Skipping never changes the produced rank-local sequence: selection by
  `required` indices and ordering by the schedule formula are untouched.
- Baseline behavior does NOT trust prepare-time verification: no digest is
  waived for a decoded chunk. A "trust prepared digests" mode is recorded
  only as a deferred option requiring its own justification and threat-model
  review.

### Interaction with the pending stream change

`stream-distributed-pack-cache-runtime` (completed, untouched) words the rank
load as "exactly one full validated payload pass", and its deltas carry fewer
scenarios than the current stable requirements. Handling: (a) task 0.1 makes
sync/archive of that change a blocking reconciliation that preserves the
union of stable scenarios; (b) this change's `Deterministic Packing Cache
Reuse` delta carries the full union (stable + prior change + new), so archive
order cannot erase accepted behavior; (c) this delta relaxes "full" to "at
most the chunks the rank schedule requires" while keeping "validated" and "at
most once".

### Expected effect and honesty about limits

For full-epoch prod schedules the required set can touch every chunk, so the
win concentrates in smokes, short runs, `max_steps` probes, and large-world
configurations; the measurement plan measures both a short-run and a
prod-shaped schedule and accepts either a startup win or neutrality with no
regression.

## 4. Seam C: Rank-Sharded eval.forward With Exact Aggregation

### Sharding

Eval packs are partitioned deterministically and disjointly by
`sequence_ordinal % world_size == rank`, where `sequence_ordinal` is each
micro-step's position in the canonical eval micro-step order (not the
pack's `pack_index` identity label, which is not guaranteed contiguous or
position-matching for every materialized sequence). Every pack is assigned
to exactly one rank; when `pack_count < world_size` the evaluator falls back
to the replicated form, which is trivially exact.

### Complete durable-scalar inventory and reducers

The canonical eval row consists of the scalar metric mapping plus the
row-level `example_count` and `pack_count`. Every durable scalar has a
declared sufficient statistic and reducer; a compact explicit eval reduction
payload (internal, non-durable) carries the rank-local statistics through the
existing bounded gatherer, and every rank derives the identical global row
before rank zero writes it. No generic reduction framework is introduced.

| Row surface | Rank-local sufficient statistic | Global reducer |
| --- | --- | --- |
| `loss/<term>`, `loss/total` | partial segment-mean numerator per term (or the world-size-scaled contribution) | sum of partials divided by the shared global denominator; equivalently the existing mean reduction over W-scaled contributions, with the identity `mean_r(W * c_r) = sum_r c_r` proven by test |
| `acc_top1`, `acc_top5` | integer `top1_correct`, `top5_correct`, local `accuracy_atom_count` | ratio of summed integers: `sum_r c_r / sum_r n_r` |
| `loss/<term>/token_weighted_diag` | `value_r * selected_count_r` and `selected_count_r` | `sum_r(value_r * selected_count_r) / sum_r(selected_count_r)` |
| `example_count`, `pack_count`, rank-local counts | local counts over the rank's disjoint shard | global sums (disjointness makes sums exact; an example belongs to exactly one pack) |
| `loss/<term>/segment_count`, globally merged `count/*` fields | none (already global) | emitted once from the shared global denominator gathered before finalize; MUST NOT be additionally rank-summed |
| `finite/*`, `non_finite_fields` handling | non-finite contributions propagate through the sums naturally | finiteness evaluated on the derived global values; writer normalization (null + `non_finite_fields`) unchanged |

### Reduction modes are explicit

The table above defines the disjoint-shard mode only. The replicated fallback
(`pack_count < world_size`) MUST keep the current replicated semantics:
every rank computes identical values over the full eval set and they are
reduced once (the existing identical-value mean), never summed as disjoint
contributions — otherwise counts and totals would be multiplied by
`world_size`. The active mode (disjoint-shard vs replicated) is carried
explicitly in the internal reduction payload / call path; it MUST NOT be
inferred ambiguously from payload shape. A dedicated multi-rank fixture with
`pack_count < world_size` verifies the entire row is identical to the
pre-change replicated evaluator and no count is multiplied (tasks 4.1).

### Tolerance and selection compatibility

Against the fully replicated evaluator on the same checkpoint and eval set:
integer counts and top-k accuracies MUST be exactly equal; loss and
token-weighted diagnostic scalars MUST agree within relative tolerance 1e-5
(fp32 summation-order effects only). Acceptance compares the entire canonical
eval row, not only loss/top-k. Best-checkpoint selection consumes `acc_top1`,
which is exact, so selection is provably unchanged. The eval row schema is
unchanged.

### Alternative compared

Keeping replication and shortening cadence was rejected: it changes research
observability rather than removing the redundancy, and the global-denominator
machinery already exists.

## 5. Seam D: Exact Train Top-k Reduction

The existing metric collective already ships a per-rank payload; it gains a
small internal extension: rank-local sufficient statistics
(`top1_correct`, `top5_correct`, `accuracy_atom_count` as integers) carried
alongside the metrics mapping. The reducer forms
`acc_topk = sum_r(correct_r) / sum_r(atoms_r)` and writes only the derived
ratios into the durable row; the statistics themselves are not durable
fields. Explicitly rejected implementations: weighting per-rank accuracies by
the metric `count/supervised_atoms` (that count is globally merged and equal
on all ranks, so the weighting degenerates to the current plain mean — the
defect this seam fixes) and reconstructing integer counts from rounded float
ratios. At world size one the reduction degenerates to the rank-local value.
Sharded eval (Seam C) uses the same statistics.

Reducer semantics per key are a fixed declared mapping, not a framework:
plain mean (existing default, still correct for W-scaled loss metrics),
all-rank maximum (Seam E timing keys), and summed-statistics ratios
(accuracy keys).

## 6. Seam E: Step Timing Fields

Measurement boundary: from the start of the planned step's first micro-step
handling through gradient zeroing — inside the compute/optimizer boundary —
excluding the completed-step handler and scheduled eval/checkpoint handlers.
Fields in the train row:

- `step_duration_seconds`: wall time of that boundary; distributed reduction
  is the all-rank maximum, because the slowest rank owns the critical path (a
  rank mean is not distributed wall time).
- `input_build_seconds`: summed forward-input construction time for the
  step's micro-steps (producer-side when overlapped); all-rank maximum for
  bottleneck visibility.
- `input_wait_seconds`: time the consuming step blocked waiting on prepared
  inputs (0 in synchronous mode); all-rank maximum.

A mean is emitted only if separately named as a mean (none planned). Values
travel through the existing metric collective — no new synchronization —
and cost monotonic-clock reads only. Artifact schema policy: additive-only;
no existing field is renamed, removed, or retyped; older rows simply lack the
fields. The run record additionally records the provider mode (Seam A) so
timing interpretation is unambiguous.

## 7. Seam F: Gated Micro-Optimizations

- Gradient finite gate: batch per-parameter checks with
  `torch._foreach_norm` (per device/dtype group) plus a single non-finite
  reduction. Hard gate: NaN/Inf/overflow decision outcomes, reason codes, and
  `grad_norm` diagnostics must be equivalent on the existing
  `tests/runtime/test_finite_gates.py` matrix extended with mixed-dtype
  cases; explicit finite checks are not replaced by a weaker inference from a
  scalar norm unless those tests prove decision and diagnostic equivalence.
  Merged only with a measured planned-step overhead reduction; otherwise
  reverted.
- Cache-materialization CPU: hoist the per-pack `examples_by_id` rebuild
  (`src/training/pipeline.py:1453-1461`) to one map per dataset; adopt
  bisect-based token-span lookup in `src/qwen/encoding.py:426-474` only with
  an exact boundary-equivalence test over a representative encoded corpus
  (identical token indices and identical error classification for every
  span, including the crossing-boundary failure cases).
- Cache identity precision: any edit to `code_identity` source files changes
  fingerprints, not only the `mtime_ns` removal; the final settled source
  state of this change induces one new production fingerprint. All
  intermediate measurement work therefore runs against dedicated temporary
  cache roots (`COORDEXP_SWIFT_PACK_CACHE_ROOT`) and sample-limited derived
  configs; the single production materialization happens once after all
  cache-determinant sources settle, in a user-authorized window. Existing
  cache directories remain historical/disposable and are never auto-deleted.
- Inference scoring interval search: out of scope (no current evidence of
  need), per lead review.

## 8. Seam G: Deletion With Proof

Non-streaming loss path removal proceeds only after a recorded
call-graph/consumer proof at live HEAD: `LossRunner` implements the full
streaming protocol, so `_supports_streaming_loss` is constant-true in
production; the only consumers of `LossRunner.compute` are the trainer batch
branch, the eval batch branch, and tests. Deletion removes the trainer batch
branch, the eval batch branch, `LossRunner.compute`/`_compute_token_term`,
and both `_supports_streaming_loss` copies; the trainer instead asserts the
streaming protocol at construction with a clear contract error. Tests that
exercised the batch path are ported to streaming-path equivalents that
preserve their numeric oracles. Dead helpers (`_loss_plan_artifact`,
`_template_identity`, `_unwrap_checkpoint_model`), the duplicated
`_sync_forward_result_if_requested` calls (`supervised_trainer.py:293-297`),
and the production-dead `build_repeating_micro_step_stream` (moved next to
its test oracle) are removed with grep-residue proof, as is the empty
`src/metrics` stub after a residue check. Deliberately NOT consolidated: the
three `_examples_by_id` copies (each owns a distinct error taxonomy) and the
`_file_sha256`/`_sha256_json` duplicates — local duplication is preferred
over cross-domain coupling, and no `utils` module is created.

## 9. Invariants (global, all seams)

1. Byte-identical model inputs: for any micro-step, the tensors reaching
   `run_qwen_forward` are identical with and without this change.
2. Identical gradient stream: loss values, denominators, gate decisions, and
   optimizer updates are bit-compatible up to fp summation order; the
   objective formula and `backend_gradient_scale` are untouched.
3. Fail-closed everywhere: every existing validation (image SHA256, chunk
   digest, restricted unpickle, prompt parity, manifest structure) fires with
   the same error codes at the same or earlier points.
4. Canonical order: rank-local pack sequence, presentation order, and
   scheduled-event order are unchanged; no provider state crosses a
   planned-step boundary.
5. Artifact compatibility: additive-only logging fields; unchanged eval row
   schema, run-record compactness, checkpoint payloads, cache format version.
6. Bounded resources: lookahead holds at most one extra micro-step of host
   tensors; no new durable per-step receipts; no new collectives.

## 10. Measurement And Stop Rules

`measurement-plan.md` defines the exact harness configs, cache-root
separation, per-wave measurements, and accept/revert decisions for each
benchmark-dependent slice. The final gate additionally requires an
independent review (tracked in `review-triage.md`) with no unresolved P0/P1
findings.
