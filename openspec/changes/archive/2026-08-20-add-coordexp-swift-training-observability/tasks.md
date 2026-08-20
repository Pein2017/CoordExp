## 0. Wave 0 - Prerequisite authority and baseline

- [x] 0.1 Verify `reconcile-coordexp-swift-training-contracts`, `decompose-coordexp-swift-training-orchestration`, and `standardize-coordexp-swift-supervised-losses` are implementation-complete, synced, and archived; record their exact commits and this change's base commit, and stop if stable specs, current code, docs, or archive dispositions disagree.
- [x] 0.2 Rebase this change's `Wide-Step Logging Stream` delta against the complete stable requirement produced by the synced losses change; retain every post-loss paragraph/scenario and prove the canonical computed-term fields are raw/configured-weight/weighted, the ambiguous alias is absent, and an omitted zero-weight optional term has no field family.
- [x] 0.3 Characterize and pin the fixed owner/import graph: `src/runtime/metrics.py` for types/reduction, `src/training/reporting.py` for canonical rows, `src/artifacts/observation_publisher.py` for JSONL-first sinks, and `src/training/session.py` for wiring. Stop if the predecessor has not established these owner seams or if the facade remains a competing owner.
- [x] 0.4 Freeze an exact command manifest with cwd, `conda` environment, commands, configs, world size/devices, artifact roots, expected evidence, and quantitative limits; require a reviewed append-only amendment for any later command change.
- [x] 0.5 Run the focused config/loss/runtime/reporting/artifact/exact-resume suites against that untouched baseline and gate entry with strict OpenSpec validation plus standards and intent-contract audits; do not begin implementation with an unresolved P0/P1.

> **Wave 0 closed (2026-08-20, base `3d390b108`):** predecessors pinned
> (all three archived; validate --all 20/20). Delta rebase complete and
> audit-verified in both directions (zero stable scenarios missing; all
> observability intent retained; three deliberate 2026-08-12 relaxations
> proven authored via `0b98f0561`). Owner seams pinned (new owners
> correctly absent; facade clean; reduction source
> `train_runtime.py::_reduce_metric_reports`). Manifest + amendments 1-3;
> entry baseline 890/0/0 (lead + audit replays). Entry audit
> (`receipts/wave-0-entry-audit.md`): STANDARDS + INTENT both
> PASS-WITH-DISPOSITIONS, 1 P1 / 5 P2 / 2 P3; the P1 and P2s are
> resolved by manifest amend-2/-3 and the amendment below; entry gate
> thereby CLEARED.
>
> **Wave-0 amendment (resolves entry-audit P1-REDUCE; binds Wave 2):**
> the typed reducer set carries no MEAN and no fallback, so the four
> families currently in the plain-mean branch are reclassified as
> RED-first DECLARED VALUE CHANGES in Wave 2 (never silent):
> (1) train `count/packs`+`count/examples` -> SUM (fixes the 2b0a2165a
> mean-over-ranks defect; matches sharded eval);
> (2) train `loss/<term>/token_weighted_diag` -> count-weighted ratio
> sample (fixes the unweighted mean-of-rank-means defect);
> (3) train `finite/*` -> BOOL_ALL (audit-found 0.5-on-split-step
> artifact; matches the all-rank finite decision semantics);
> (4) replicated-eval objective keys -> IDENTICAL, gated on an empirical
> two-rank divergence measurement inside the task-2.5 gloo probe
> (bitwise equality expected under the enforced deterministic env; if
> divergence is observed, a bounded-tolerance IDENTICAL variant with the
> measured bound recorded instead). Task 2.4's "preserving full-row
> equivalence" clause is amended accordingly: equivalence is preserved
> for correctly-reduced families; the three defect families change value
> BY DECLARATION with parity evidence against a world-size-1 reference.
> P2-COMPARATOR carried to Wave 5.2: new fields must be enumerated BY
> NAME in the comparator exclusions (`input_h2d_seconds` evades both the
> v1 TIMING_FIELDS list and v2's suffix predicate).

## 1. Wave 1 - Required presentation config

- [x] 1.1 Add failing config tests proving `observability.steps` is required
  with no default, accepts only positive integers, rejects legacy logging
  aliases, and remains present in `resolved_config.json`.
- [x] 1.2 Add the strict required `ObservabilityConfig` surface to
  `src/config/models.py` without adding enable flags, sink lists, or a
  compatibility alias.
- [x] 1.3 Author an explicit `observability.steps` value in every supported
  training YAML under `configs/coordexp_swift/prod/` and
  `configs/coordexp_swift/smoke/`; leave inference, archived, Stage 1, and Stage
  2 configs untouched.
- [x] 1.4 Update current config fixture builders and strict-key tests so every
  accepted training fixture makes an explicit presentation decision and old
  fixtures fail for the intended missing-field reason.
- [x] 1.5 Gate Wave 1 with the focused config suite, a script that resolves
  every supported active training YAML, strict OpenSpec validation, a residue
  search for an observability default or accepted logging alias, and no
  unresolved test or validation failure.

> **Wave 1 closed (2026-08-20, opus builder, zero correction rounds):**
> `ObservabilityConfig` with exactly `steps: int = Field(gt=0, strict=True)`,
> required, no default, no alias/enable-flag/sink surface, presentation-only
> docstring. RED 42/9 observed; 17 alias/extra params declared
> green-from-birth regression pins. 26 files authored (prod=10, smoke and
> length_isolation=1, live loader-input fixture=1 pre-authorized; text-append
> only, byte-preserving; frozen training_orchestration subtree untouched).
> Active-profile baseline revision `2a297a93a` -> `51cc48de6`, 25/25 digests
> refreshed via the test module's own digest helper; drift-guard allowlist
> deliberately NOT widened. Gate: 435/0/0 (lead replay identical); inventory
> probe exit 0; pin-family check 520/6skip stable across the second
> config-byte drift; residue zero; **determinant no-entry proven twice**
> (redirected archived-baseline probe EQUAL/EQUAL + lead direct recompute
> with payload sha byte-identical). Manifest amend-4 records the archived
> probe-path disposition and the tracked test_gate_ablation flake (2nd
> occurrence, tripwire set at 3). Collateral survey 1889/0/126skip.

## 2. Wave 2 - Explicit distributed metric reduction

- [x] 2.1 Add failing unit tests for scalar and ratio samples, cross-rank
  schema/reducer disagreement, required versus backend-unavailable fields,
  identical-value mismatch, sum-before-divide ratios, and rejection of a
  metric with no reducer.
- [x] 2.2 Implement the narrow immutable metric sample/batch types and reducer
  logic in `src/runtime/metrics.py`; use `BOOL_ALL` for boolean conjunction,
  keep reducer selection producer-owned, and add no
  mutable registry, event names, subscriptions, or implicit mean fallback.
- [x] 2.3 Migrate train metric gathering to typed batches, preserving exact
  integer accuracy statistics and keeping per-rank reports ephemeral rather
  than serializing a normal `per_rank_measurement` trace.
- [x] 2.4 Migrate both replicated and disjoint-shard eval reduction to explicit
  `IDENTICAL`, `SUM`, `MAX`, `BOOL_ALL`, and ratio samples while preserving
  full-row equivalence and best-checkpoint selector values.
- [x] 2.5 Add and execute a real two-process Gloo probe with deliberately
  asymmetric rank-local norms, timings, counts, and numerator/denominator
  statistics; assert rank maximum, sum, ratio, schema agreement, and clean
  collective termination.
- [x] 2.6 Delete the superseded key-name/suffix reducer tables, implicit
  plain-mean branch, and normal-row per-rank serialization after all callers
  use typed batches.
- [x] 2.7 Gate Wave 2 with focused runtime and eval suites, the Gloo receipt,
  strict OpenSpec validation, and searches proving no implicit reducer or
  dynamic registry remains; do not continue with an unresolved test or
  validation failure.

> **Wave 2 closed (2026-08-20, opus builder, zero correction rounds):**
> typed reduction lives in new `src/runtime/metrics.py` (immutable
> samples/batches; SUM/MAX/IDENTICAL/BOOL_ALL/ratio; producer-declared
> reducers; undeclared metric raises `runtime.metric_reducer_undeclared`;
> no registry, no implicit mean). The Wave-0 amendment's four families
> landed as declared RED-first value changes with per-family RED receipts
> (SUM counts, count-weighted ratio diag, BOOL_ALL finite, and
> replicated-eval IDENTICAL implemented EXACT after the gloo probe
> measured bitwise-0.0 divergence). `train_runtime.py` 1165->583 lines:
> plain-mean branch (2b0a2165a TODO discharged), all suffix/prefix
> reducer tables, and the eval dict machinery deleted; normal rows no
> longer serialize `per_rank_measurement` (zero hits in src/; per-rank
> scalars remain ephemeral in ReducedMetricBatch for bounded lifecycle
> receipts). Collective order/count unchanged (single gather; fixtures
> byte-identical; compat suite green). ws-1 parity exclusion list DELETED
> - every reduced key now matches the reference. Gates (lead replays
> identical): 384/0/0 reduction, 379/0 artifacts/reporting/compat, 1446/0
> collateral, strict valid, determinant fingerprints + payload sha
> byte-identical. Deviations accepted: lr/group_* + merged-denominator
> counts -> fail-closed IDENTICAL (value-preserving), NaN-propagating
> MAX, error-code renames (test-only callers), typed
> `EvalRuntimeBoundary.gather_metrics` protocol. Manifest amend-5 pins
> counts, retires the probe placeholder, and records two archived probe
> scripts that would break if re-run (deliberately unedited). Flake
> tripwire count stands at 2 (zero this wave).

## 3. Wave 3 - Truthful optimizer, loss, timing, and resource observations

- [x] 3.1 Add failing CPU optimizer/scheduler tests that distinguish planned
  step, optimizer-wrapper attempt, actual applied update, and scheduler advance.
  Preserve scheduler/eval/checkpoint planned-step policy; add the closed
  `optimizer_boundary_action = apply | scaler_skip | not_attempted` decision and
  prove all ranks select the same action before a wrapper call. Prove
  the runtime-owned `AppliedUpdateReceipt.not_attempted(planned_step_id,
  group_count, reason)` returns `attempted: false`, `applied: false`,
  `step_was_skipped: false`, and null group LRs on a `not_attempted` rejection;
  prove `optimizer_step_count` does not advance for that branch but counts every
  completed wrapper invocation, including a `scaler_skip` finalization;
  `scheduler_step_count` counts completed planned-step advances, and
  `optimizer_update_status` owns actual application without a redundant durable
  applied counter; prove applied LR is the pre-call param-group value only when
  the accepted global outcome is actually applied. Add runtime-owned
  `terminal_not_attempted` and `terminal_post_wrapper` receipt constructors with
  truthful nullable fields and bounded mutation state. A supported pre-backward
  `not_attempted` may remain `unchanged`, but an fp16 `terminal_not_attempted`
  after exactly-once unscale MUST preserve known no-wrapper/no-parameter-
  mutation truth while reporting the composite GradScaler state as
  `divergent_or_unknown`, not `unchanged`; reporter/session code MUST NOT
  synthesize those booleans from exceptions.
- [x] 3.2 Add failing ordering tests proving fp16 has exactly one owner and call
  order: `accelerator.unscale_gradients(optimizer)` exactly once, gradient finite
  inspection and pre-clip norm, then one global boundary action. Prove `apply`
  uses a non-unscaling clip primitive such as `torch.nn.utils.clip_grad_norm_`
  over already-unscaled gradients before the wrapper. Prove `scaler_skip` is
  admitted only when every rank has an active scaler and a current non-finite
  unscaled gradient, performs no clip, calls the wrapper once for scaler
  finalization. Require every real fp16 wrapper call to converge
  `all_skipped | none_skipped | mixed` before scheduler: `apply` accepts only
  none-skipped and `scaler_skip` only all-skipped. Prove mixed candidacy or
  unrelated unsafe fp16 state terminates before a wrapper with parameters and
  underlying optimizer untouched but GradScaler `UNSCALED`/unfinalized and
  `mutation_state=divergent_or_unknown`; prove post-call mixed
  truth uses nullable application state, while unanimous action-contradictory
  truth remains known; `scaler_skip+none_skipped` retains identical pre-call LRs
  actually applied, while other terminal outcomes use null LRs. No rank-local
  raise may precede consensus. Prohibit
  `accelerator.clip_grad_norm_`, return the single runtime-owned bounded update
  receipt, propagate it through `CompletedStepObservation`, and reuse the
  existing all-rank max norm without another gradient-scan collective.
- [x] 3.3 In `src/training/reporting.py`, consume the prerequisite completed loss telemetry without recomputing from backend-scaled tensors or sufficient statistics: preserve raw/configured-weight/weighted fields only for actually computed terms, retain computed zero-weight gate diagnostics, preserve complete omission of a zero-weight optional term, and verify single-rank and asymmetric multi-rank rows.
- [x] 3.4 Capture exact per-step physical-token, supervised-atom, and pack work
  counts before tensors are released and derive global throughput only from
  summed work divided by rank-max step duration.
- [x] 3.5 Extend input timing with accurately completed H2D measurement where
  available, retain honest CPU-build/wait/step scopes, mark unsupported fields
  unavailable, and add a test that normal observation never synchronizes CUDA
  solely for timing.
- [x] 3.6 Extend the resource collector with current and process-lifetime peak
  CUDA allocated/reserved bytes plus per-step allocator retry/OOM deltas;
  verify rank-max bytes, rank-sum deltas, CPU unavailability, and no fabricated
  zero.
- [x] 3.7 Update train/eval row construction in `src/training/reporting.py` to
  emit the new loss, LR, norm, throughput, timing, memory, counter, finite, and
  availability fields while retaining one strict row per completed planned
  step/eval. Add the terminal exception: pre-wrapper terminal unsafe and post-
  wrapper contradictory outcomes publish/converge exactly one row at the
  current planned-step id before failed finalization, without incrementing
  completed/scheduler counts or dispatching eval/checkpoint/exact-resume/
  selector/final-success handlers. If row publication fails, preserve the
  optimizer-boundary code as primary during common failed finalization.
- [x] 3.8 Before any GPU-backed action, review the frozen command manifest,
  obtain fresh user authorization for that action, and record bounds for
  devices/world size, planned steps, model forwards, cache/materialization
  passes, wall time, peak GPU memory, and artifact bytes. Execute genuine CUDA
  fp16 finite and overflow arms through real Accelerate. Both MUST prove exactly
  one `accelerator.unscale_gradients(optimizer)` followed by finite/norm
  authority with no `accelerator.clip_grad_norm_`; the finite arm MUST then prove
  one non-unscaling clip, an applied update, and pre-call LR. The overflow arm
  MUST prove an all-rank-confirmed `scaler_skip`, zero clip calls, one wrapper
  call solely for scaler finalization, post-call skipped truth, null applied LR,
  one completed-wrapper counter increment, planned scheduler progression, and
  no underlying optimizer mutation or false applied status. Also execute a
  bounded two-rank injected-outcome probe covering pre-wrapper mixed/unsupported,
  post-wrapper mixed, `apply+all_skipped`, and `scaler_skip+none_skipped`:
  require truthful terminal fields (including pre-wrapper
  `mutation_state=divergent_or_unknown` after unscale), one terminal row, common
  failed finalization, no rank-local pre-consensus raise, and zero scheduler/
  scheduled-handler progression. Gate
  DDP/cost expansion with focused trainer/loss/runtime/resource/artifact tests,
  collective/CUDA-sync residue checks, strict OpenSpec validation, and the
  single pre-DDP/cost standards plus intent-contract audit; no fp16 claim is
  allowed without both receipts.

> **Wave 3 closed (2026-08-20; commits `bdb29b3fa` 3A boundary,
> `4db58e972` 3B rows/resources, `305b17eb7` probe scripts, `e81f6a91a`
> scaler-keying fix + receipts; pre-DDP audit
> `receipts/wave-3-pre-ddp-audit.md` 0 P0/0 P1, Wave-4 entry CLEARED):**
> all-rank `optimizer_boundary_action` on the existing gradient gather;
> spec-authored fp16-only post-wrapper consensus (non-fp16 byte-unchanged);
> constructor-only `AppliedUpdateReceipt` with derived mutation state;
> truthful rows (availability honesty, rank-max-duration throughput,
> IDENTICAL-not-MAX for already-global grad_norm/lr); bounded terminal-row
> path. **Real-CUDA defect found and fixed with full RED->GREEN receipts**:
> `_scaler_found_inf` keyed GradScaler state on the wrapper optimizer
> (structurally False; discriminating RED proved it could flip the boundary
> action to apply under finite-grads+scaler-inf); attempt-1 fp16 receipts
> retain the fired FINDING as evidence, attempt-2 clean; audit re-derived
> the fix against installed torch 2.9.1/accelerate 1.10.1 internals
> (staleness ruled out). 3.8 executed under the frozen packet on idle GPUs
> 0,1 (standing grant): fp16 finite+overflow arms ws1+ws2 all clauses
> green, injected-outcome six arms x two ranks all-true, cache root
> byte-identical, bounds honoured. Accepted deviations recorded:
> **W3-2 (do NOT delete)** - the receipt-less `_scheduler_lr_metrics`
> legacy fallback publishes post-scheduler LR under `lr/group_<i>`; it is
> production-unreachable but LOAD-BEARING for the frozen characterization
> row - Wave-5 residue searches must not remove it; W3-3 - session-level
> terminal wiring proven by source inspection only; **carried obligation
> bound to task 4.1: the terminal-row + injected-append-failure coverage
> must be BEHAVIORAL at the `_run_training_session` seam**. P3 residuals
> W3-4/W3-5 recorded in amend-6. Gate 832-834/0/0 (lead + audit), 15/15
> fixtures/collective, determinant EQUAL/EQUAL, strict valid. Flake
> tripwire: count stands at 2 (zero this wave).

## 4. Wave 4 - JSONL-first console and TensorBoard publisher

- [x] 4.1 Add failing publisher tests for required `step`, step/total console
  formatting, train interval/terminal dispatch, unconditional eval mirroring,
  rank-zero-only ownership, and proof that JSONL append completes before any
  presentation call. Cover the terminal optimizer-boundary row and an injected
  append failure that still converges failed finalization with the primary
  optimizer-boundary code.
- [x] 4.2 Keep the row builder in `src/training/reporting.py` and implement the
  direct rank-zero observation publisher in
  `src/artifacts/observation_publisher.py` around the existing `RunWriter`
  append/status handshake; expose one train/eval publication interface rather
  than an event dispatcher.
- [x] 4.3 Implement the compact console presenter and segment-local approximate
  ETA without persisting ETA or presentation state and without including sink
  time in `step_duration_seconds`.
- [x] 4.4 Implement a lazy run-local `tensorboard/` sink using
  `torch.utils.tensorboard.SummaryWriter`, deterministic
  `<split>/<canonical-key>` tags, finite numeric values only, canonical planned
  step as `global_step`, bounded queueing, and terminal close.
- [x] 4.5 Add a TensorBoard event-reader test that loads a temporary run's event
  file and proves expected train/eval tags, finite values, and global steps.
- [x] 4.6 Inject TensorBoard import, initialization, `add_scalar`, flush, and close failures; prove the already written JSONL row survives, at most one bounded run warning plus one best-effort stderr warning is emitted, cleanup failure cannot recurse, the sink latches disabled, and later JSONL/eval/checkpoint work continues. Add exact tests that `unavailable_fields` and `non_finite_fields` are sorted/unique and bounded by field count and name bytes with one bounded truncation count/marker.
- [x] 4.7 Wire the publisher only from `src/training/session.py`, delete
  superseded console/TensorBoard/pass-through helpers, and verify non-main ranks
  create neither terminal progress nor event files.
- [x] 4.8 Gate Wave 4 with focused publisher, run-writer, train/eval reporting/
  session, TensorBoard reader, and failure suites; strict OpenSpec validation; searches
  for event buses, generic registries, alternate scalar files, DB/W&B, and
  per-rank event streams; do not continue with an unresolved test or validation
  failure.

> **Wave 4 closed (2026-08-20, opus builder, zero correction rounds):**
> `src/artifacts/observation_publisher.py` owns the direct rank-zero
> publisher (publish/close only; no dispatcher); JSONL append + all-rank
> status broadcast strictly precede any presentation; identity validation
> runs on every rank before the collective. Console (stderr, step/total +
> compact scalars + segment-local approximate ETA never persisted);
> TensorBoard lazy run-local sink with deterministic split/key tags,
> finite-only numerics, canonical global_step, real event-reader test.
> Seven-arm failure injection all isolated (latch, one bounded warning +
> one stderr line, no recursion, run continues). W3-3 carried obligation
> DISCHARGED behaviorally at `_run_initialized_training` (real RunWriter,
> primary boundary code on injected append failure, zero handler dispatch,
> non-main rank creates nothing); seam pins proven by a byte-restored
> mutation check. `_append_logging_row_shared` moved (not aliased) to the
> publisher module with declared-flip pin retargets; no other legacy
> console/TB helper existed to delete (receipted). Retained load-bearing
> fallbacks recorded in amend-7 (W3-2 LR fallback + publisher=None
> publication-only path) - Wave-5 residue must not remove them. Gate
> 735/0/0 (lead replay identical), 15/15 fixtures/collective, collateral
> 2345/0/126skip, residue zero, strict valid, determinant payload sha
> byte-identical. Flake tripwire: count stands at 2 (zero this wave).

## 5. Wave 5 - Resume boundary, docs, and production-shaped acceptance

- [x] 5.1 Exclude the top-level `observability` block from exact-resume
  semantic compatibility and three-run input-attestation projections while
  keeping forward, loss, optimizer, scheduler, data-order, RNG, applied LR,
  pre-clip norm, finite, and update-status comparisons strict.
- [x] 5.2 Update the Wave 7 exact-resume comparator and fixtures to classify new
  timing/resource/availability fields as non-semantic observations, ignore
  console/TensorBoard/ETA, and accept presentation-only config drift without
  accepting loss, LR, norm, or update drift. Prove no exact-resume state,
  checkpoint, best selector, or successful final artifact is published from a
  terminal pre-wrapper or post-wrapper optimizer boundary, and that `run.json`
  records failed status plus the current planned-step id without counting it as
  completed.
- [x] 5.3 Update `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`,
  `docs/IMPLEMENTATION_MAP.md`, and `docs/ARTIFACTS.md` with the required config,
  canonical metric meanings/reducers, run-local TensorBoard path, approximate
  ETA boundary, failure policy, and probe-only metric exclusions; do not revive
  historical Trainer/event-bus terminology.
- [x] 5.4 Run the focused exact-resume, input-attestation, config, runtime,
  loss, eval, artifact, trainer, reporting, publisher, and session suites through
  `conda run -n ms`, then re-resolve every active production/smoke training
  config and run strict JSON/non-finite checks over emitted fixture rows.
- [x] 5.5 With fresh user authorization for this distinct GPU action, record its
  bounds for devices/world size, planned steps, model forwards, cache/
  materialization passes, wall time, peak GPU memory, and artifact bytes, then
  execute the smallest current two-rank production-mimic config that
  performs one real finite optimizer update and scheduled eval; verify one
  shared run tree, one train and one eval canonical row, correct LR/norm/loss/
  timing/resource fields, readable rank-zero TensorBoard events, and no
  rank-local run/event trees. Record exact config, commit, command, devices,
  artifact root, counters, and evidence scope without making a throughput or
  model-quality claim.
- [x] 5.6 Run the relevant broader regression suite, `openspec validate
  add-coordexp-swift-training-observability --strict`, and residue searches
  proving no implicit reducer, old scheduler-derived LR path, normal per-rank
  trace, alternate scalar authority, or unbounded sink error state remains.
- [x] 5.7 Obtain the final standards/code-quality and intent/spec-contract
  audits over the implementation, executed probes, emitted artifacts, docs,
  and exact-resume exclusions; resolve all P0/P1 findings and record any lower
  severity residual risk before requesting OpenSpec verification/archive.


> **Wave 5 closed / change complete (2026-08-20; commits `30b563e6c` part 1,
> `c4d4dc857` smoke receipts, plus this close-out; final audit
> `receipts/wave-5-final-audit.md` STANDARDS+INTENT both
> PASS-WITH-DISPOSITIONS, 0 P0 / 0 P1, completion gate YES):**
> presentation-only `observability` excluded from exact-resume
> compatibility and three-run attestation with RED-observed fail-closed
> semantics retained; comparators classify observation fields BY NAME with
> partition-completeness tests (optimizer truth/counters/loss-weight kept
> semantic); terminal-boundary no-publication proven behaviorally
> (no state/checkpoint/selector/final from a terminal boundary; run.json
> failed at the planned-step id). 5.5 smoke under the frozen packet on
> idle GPUs 0,1 (standing grant): one shared run tree, train+eval
> canonical rows with every new field family live (incl. honest
> unavailable_fields on the first-step allocator deltas), TB 77 tags
> verified with the real event reader, all bounds honoured, cache
> byte-identical, and train/eval base_ce raw values bit-equal to the
> archived losses-change smoke. 5.4 focused 1357/0/0; 5.6 broad
> 2565/0/126skip with zero residue (audit's independent f-string-aware
> sweep also 0). Final-audit P2s discharged in this commit: amend-8
> (placeholder retired, counts pinned, re-pins recorded) and the
> ARTIFACTS.md reducer-attribution fix (pre-clip norm is IDENTICAL by
> design, not MAX). P3s recorded in the receipt. Archive-mechanics
> scenario diff: 0 problems across all six MODIFIED requirements.
> Flake tripwire final count: 2 (never reached 3).
