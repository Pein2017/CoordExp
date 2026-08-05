# Tasks: Streamline CoordExp-Swift Base Infrastructure

Sequencing rules: waves are dependency-ordered; implementation MAY stop after
any completed wave gate with the change still coherent. Wave 1 lands
observability first so later waves are measurable. A benchmark-dependent slice
whose measured gate fails is reverted and recorded as rejected-with-evidence
(stop rule), without blocking unrelated waves. Do not begin the next wave
while the current gate has unresolved P0 or P1 findings.

Shared gate commands (every wave):

- Targeted tests for the touched modules (named per wave).
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict`
- Markdown/diff hygiene: `git diff --check` and repo markdown lint if
  configured; residue grep for deleted symbols.
- Measurements per `measurement-plan.md` where the wave is
  benchmark-dependent.

Review cadence (deliberately not per-wave): a focused independent review
after Wave 3 and Wave 4 (the high-risk concurrency/distributed slices), and
one final independent convergence review at task 6.5. Other waves gate on
tests, validation, and hygiene only, unless a wave exposes a new semantic
risk, in which case a focused review is added for that wave.

## 0. Preconditions And Baseline

- [x] 0.1 BLOCKING: reconcile-and-preserve sync of
      `stream-distributed-pack-cache-runtime`. Before applying this change's
      packing/cache deltas, its deltas must be synced/archived such that the
      union of current stable scenarios is preserved — for `Deterministic
      Packing Cache Reuse`: same-template relaunch, renderer code change,
      production cache miss, worker count, older payload version; for `Cache
      Payload Is Current-Version-Only`: old-version and incomplete-manifest
      rejection — plus that change's own distributed-consumption scenarios.
      Record the observed post-sync stable text. Do not edit that change
      from this change. Additionally revalidate this change's design Section
      1 claims (call graph, line anchors, reduction surfaces) against live
      HEAD; the drafting baseline commit is provenance, not authority.
      DONE 2026-08-03: repaired the completed change's two MODIFIED delta
      specs (`coordexp-swift-packing-forward`,
      `coordexp-swift-pack-cache-semantic-identity`) to carry the full
      stable-scenario union plus its own distributed-consumption scenarios,
      validated `openspec validate stream-distributed-pack-cache-runtime
      --strict` (valid), then archived via `openspec archive
      stream-distributed-pack-cache-runtime --yes` to
      `openspec/changes/archive/2026-08-03-stream-distributed-pack-cache-runtime/`.
      Post-archive stable specs
      (`openspec/specs/coordexp-swift-packing-forward/spec.md`,
      `openspec/specs/coordexp-swift-pack-cache-semantic-identity/spec.md`)
      confirmed to carry every listed scenario under `Deterministic Packing
      Cache Reuse` and `Cache Payload Is Current-Version-Only`, plus the
      untouched `Distributed Training Consumes Prepared Caches` requirement
      preserved as-is. Design Section 1 revalidation against live HEAD:
      current worktree HEAD is `a19ffceeb`, the exact commit design.md cites
      as its inspection baseline (no drift possible). Spot-checked the cited
      call graph directly: `supervised_trainer.py` move-then-forward
      streaming shape and early-break `if not pre_decision.should_call_backward:
      break` (~258-321, matching cited 258-270/285-321, including the
      pre-existing duplicated `_sync_forward_result_if_requested` calls flagged
      for task 5.2); `src/qwen/forward.py:160 build_qwen_forward_inputs`;
      `src/training/pack_cache.py` manifest-level admission
      (`PACK_CACHE_VERIFICATION_LEVELS`, `load_cache_manifest`,
      `load_rank_micro_steps_from_cache`); `src/eval/forward.py:71
      ForwardEvalRunner`; `src/runtime/train_runtime.py:387
      _reduce_metric_reports`; `src/losses/runner.py` counts built from the
      globally merged base denominator (`_build_counts_from_token_sequences`,
      ~245-270/880-905 matching cited 252-265/885-900). No claim
      contradicted; design.md left untouched (out of Wave 0 edit scope).
- [x] 0.2 M0 scope, revised by user decision 2026-08-04: a read-only
      historical-baseline availability receipt only, not a throughput
      measurement. Fail-closed preflight before any M0 claim of
      availability: derive the exact resolved config's cache fingerprint and
      verify the corresponding cache in the shared production root exists,
      is complete/current-version, and admits through validation-only checks
      without any write. If it is absent, incomplete, or
      fingerprint-mismatched, M0 reports "baseline unavailable" — it MUST
      NOT materialize into the shared production root, which stays
      read-only until user-authorized 6.1a. M0 records no time-to-step,
      steps/hour, eval-wall-time, or `COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS`
      figures under this scope; decision-bearing performance evidence for
      Waves 2-5 comes from the paired A/B methodology in
      `measurement-plan.md` instead.
      DONE 2026-08-03 (preflight) / ACCEPTED 2026-08-04 (user decision): the
      fail-closed preflight ran read-only (no cache write) and found all
      four required production packing-cache fingerprints (H-short
      train/eval, H-steady train/eval) absent from
      `.cache/coordexp_swift/packing`; M0 reported "baseline unavailable"
      per the plan's own stop rule. The user reviewed this result and
      decided NOT to build the missing caches; the missing baseline is
      explicitly accepted as permanent, not silently treated as satisfied.
      Task complete on that basis — the preflight receipt itself, and its
      accepted disposition, is the M0 deliverable. GPU 0 was also not idle
      at check time (external process, 97% util, 19.7 GB used), recorded for
      provenance only and no longer gating since M0 performs no GPU launch.
      Full receipt and the user-decision record: `implementation-notes.md`.

## 1. Exactness, Defaults, And Observability (correctness-preserving)

- [x] 1.1 Add failing unit test: with unequal per-rank supervised-atom
      counts, reduced `acc_top1`/`acc_top5` must equal
      `sum_r(correct_r) / sum_r(atoms_r)`. Then implement exact rank-local
      sufficient statistics — integer `top1_correct`, `top5_correct`, and
      local `accuracy_atom_count` — carried in the internal reduction
      payload and summed before the ratio is formed (design Seam D). The
      test must also assert the anti-patterns fail: weighting by the
      globally merged `count/supervised_atoms` metric and reconstructing
      counts from rounded ratios are not equivalent on the fixture.
      DONE: `LossBundle.accuracy_stats` (`src/losses/runner.py`) carries
      exact int `top1_correct`/`top5_correct`/`accuracy_atom_count` from
      `compute`/`compute_micro_step`; `finalize_planned_step` sums them via
      `_merge_accuracy_stats` (integer sum over every micro-step artifact,
      never a rounded-ratio reconstruction; a missing/malformed micro-step
      artifact or a correct count exceeding its atom count fails closed
      rather than silently undercounting). `TrainRuntime.gather_metrics`
      gained an `accuracy_stats` kwarg; `_reduce_metric_reports` requires it
      unconditionally whenever `acc_top1`/`acc_top5` are present in the
      gathered payload (local or any peer missing it, or a peer's correct
      count exceeding its atom count, raises `RuntimeContractError` — never
      a silent plain-mean fallback); `eval.forward`
      (`src/eval/forward.py::_accuracy_stats_from`) now also forwards real
      accuracy_stats from the loss bundle/artifact, so the requirement holds
      uniformly for every production caller that gathers accuracy keys.
      Revised 2026-08-04 per Opus independent review P1-B (see
      `review-triage.md`) — the original cut left local-omission silently
      falling back to a plain mean. Tests:
      `tests/losses/test_runner.py` (accuracy_stats exactness across
      unequal-atom-count micro-steps, missing/malformed/correct-exceeds-atoms
      micro-artifact rejection),
      `tests/runtime/test_train_runtime.py::test_multirank_accuracy_reduction_sums_integer_stats_for_unequal_atom_counts`,
      `::test_multirank_accuracy_reduction_rejects_rounded_ratio_reconstruction`
      (anti-pattern fixtures diverge from the exact result),
      `::test_single_rank_accuracy_stats_degenerate_to_rank_local_value`,
      `::test_single_rank_accuracy_stats_reject_correct_exceeding_atom_count`,
      `::test_multirank_accuracy_metric_without_local_accuracy_stats_fails_closed`,
      `::test_multirank_accuracy_metric_local_missing_peer_present_fails_closed`,
      `::test_multirank_accuracy_metric_without_peer_accuracy_stats_fails_closed`,
      `::test_multirank_accuracy_metric_rejects_local_correct_exceeding_atom_count`,
      `::test_multirank_accuracy_metric_rejects_peer_correct_exceeding_atom_count`,
      `::test_multirank_metrics_without_accuracy_keys_never_require_accuracy_stats`,
      `tests/training/test_pipeline_assembly.py::test_train_logging_forwards_real_loss_runner_accuracy_stats_without_leaking_to_row`
      (production wiring end to end with the real `LossRunner`, no fakes).
- [x] 1.2 Flip `model.fa2_branch_proof` schema default to `first_micro_step`
      in `src/config/models.py`; add config test that an omitted key resolves
      to `first_micro_step` and that explicit `every_forward` still captures
      per-forward proof; verify no prod/smoke config changes meaning
      (all set the key explicitly).
      DONE: schema default flipped
      (`ModelConfig.fa2_branch_proof`). Config tests added:
      `tests/config/test_train_config.py::test_fa2_branch_proof_omitted_resolves_to_first_micro_step`,
      `::test_fa2_branch_proof_explicit_every_forward_stays_explicit`; policy
      behavior tests added:
      `tests/training/test_pipeline_assembly.py::test_fa2_branch_proof_policy_first_micro_step_captures_only_first_step`,
      `::test_fa2_branch_proof_policy_every_forward_captures_every_step`.
      Verification found the proposal's blanket "all current prod/smoke
      configs already set it explicitly" claim did NOT hold: all 5 tracked
      `prod/*.yaml` configs set it explicitly (unaffected), but 8
      `smoke/*.yaml` configs omitted it (`ebs128_1step`,
      `ebs1_memprobe_1step`, `llm_memfit6000_ebs128_1step`,
      `ebs4_streamprobe_1step`, `ebs128_2step`, `ebs16_streamprobe_1step`,
      `accelerate4_ebs128_2step`, `accelerate2_ebs2_1step`) and would have
      silently changed meaning for any multi-micro-step run among them.
      Fixed by adding `fa2_branch_proof: every_forward` explicitly to those
      8 configs, preserving their exact pre-change resolved behavior.
      Confirmed via
      `tests/config/test_train_config.py::test_active_profile_migration_changes_only_infrastructure_allowlist`
      (resolved-config semantic digest over every active profile, unaffected
      by this fix since the *resolved* value did not change).
- [x] 1.3 Add `step_duration_seconds`, `input_build_seconds`,
      `input_wait_seconds` to the completed-step logging row (design Seam E):
      measured inside the planned-step compute/optimizer boundary (first
      micro-step handling through gradient zeroing), excluding completed-step
      and scheduled eval/checkpoint handlers; reduced as the all-rank maximum
      through the existing metric collective with no new synchronization.
      Unit tests: presence, boundary exclusion, max-reduction semantics with
      unequal per-rank timings, non-finite normalization, additive-only
      schema (no existing field renamed/removed/retyped).
      DONE: `CompletedStepObservation` (`src/training/supervised_trainer.py`)
      gained the three additive fields; `_run_streaming_planned_step` times
      the compute/optimizer boundary with `time.monotonic()` around the
      existing method body (start of first micro-step handling through
      `zero_gradients`) and accumulates `input_build_seconds` honestly from
      the existing (unconditional, not profiling-flag-gated) Qwen forward
      receipt field `timings_ns["total_build_inputs_ns"]`
      (`src/qwen/forward.py`) rather than inventing a new timing carrier;
      `input_wait_seconds` is `0.0` for synchronous (pre-Wave-3)
      construction. `pipeline._train_logging_handler` merges the three
      fields into the metrics passed to `gather_metrics`;
      `TrainRuntime._reduce_metric_reports` reduces
      `{step_duration_seconds, input_build_seconds, input_wait_seconds}` as
      the all-rank maximum (`_MAX_REDUCED_METRIC_KEYS`) through the existing
      collective — no new synchronization. Non-finite normalization reuses
      the existing generic `RunWriter` null/`non_finite_fields` handling
      (no new code needed). Non-streaming (production-dead, Wave-5-deletion
      bound) batch path left uninstrumented — its observations report `None`
      ("not measured"), never a fabricated `0.0`; `_train_logging_handler`
      only adds a timing field to the gathered metrics when it is not
      `None`, so that path's durable rows omit the fields entirely rather
      than asserting a false zero-duration measurement (revised 2026-08-04
      per Opus independent review P2-G; see `review-triage.md`).
      Additive-only, no behavior regression since this path has no
      production caller. Tests:
      `tests/training/test_supervised_trainer.py::test_streaming_step_records_input_build_seconds_from_forward_receipt`,
      `::test_streaming_step_duration_excludes_completed_step_and_scheduled_handlers`
      (deterministic incrementing fake-clock proof: the handlers themselves
      consume ticks, so a boundary that wrongly included handler time would
      read a precisely wrong, explicitly asserted value — not merely raise),
      `::test_non_streaming_batch_path_leaves_timing_fields_unmeasured`,
      `tests/runtime/test_train_runtime.py::test_multirank_timing_fields_reduce_to_all_rank_maximum_not_mean`,
      `tests/training/test_pipeline_assembly.py::test_train_row_carries_timing_fields_additively`,
      `::test_train_row_normalizes_non_finite_timing_fields`.
- [x] 1.4 Wave gate: `tests/runtime/`,
      `tests/training/test_supervised_trainer.py`, `tests/config/`,
      `tests/artifacts/`-relevant tests; shared gate commands; confirm
      logging-row consumers (vis tools, scripts) tolerate additive fields.
      DONE: `tests/runtime/ tests/training/ tests/losses/ tests/config/
      tests/artifacts/ tests/eval/ tests/qwen/` — 453 passed (2026-08-04,
      after the Opus review fixes below). `ruff check`
      clean on touched files (3 pre-existing unrelated findings confirmed
      present at HEAD baseline, untouched). `ruff format --check` was not
      used as a gate: the untouched HEAD baseline of every touched file
      already fails it repo-wide (no ruff/pyproject format config present),
      confirmed by diffing baseline vs. working tree — not a regression.
      `git diff --check` clean. `openspec validate
      streamline-coordexp-swift-base-infrastructure --strict` and `openspec
      validate --all --strict` both pass (19/19). Residue grep:
      `_merge_accuracy_metric` fully removed; no remaining
      already-global-count accuracy weighting. Logging-row consumers
      (`scripts/pipelines/train_task_manager.py`) read rows defensively via
      `.get`/key-membership checks, not exact-schema equality — tolerate
      additive fields.

## 2. Cache Identity And Rank-Selective Loading

- [x] 2.1 Add failing test: identical dataset content with changed `mtime_ns`
      must produce an identical semantic fingerprint; changed content must
      change it. Then remove `mtime_ns` from
      `build_packing_cache_determinants`; update the mtime-preserving test at
      `tests/training/test_pack_cache.py:202` to assert timestamp
      independence instead.
      DONE: `"mtime_ns"` removed from the `dataset` determinant block in
      `build_packing_cache_determinants` (`src/training/pack_cache.py`);
      `size_bytes`, `path`, and `sha256` retained, all other determinant
      groups (template, packing, processor, ordering, augmentation, qwen
      identity, code identity) untouched. The existing test at
      `tests/training/test_pack_cache.py:202` (mtime-forced-constant
      content-change probe) renamed/extended to
      `test_packing_cache_fingerprint_is_timestamp_independent_and_tracks_content`:
      now also asserts `"mtime_ns" not in` the determinants dict, that a
      `touch` (mtime advanced 5s, content/size unchanged) produces an
      identical fingerprint, and (unchanged from before) that a byte content
      change produces a different fingerprint independent of timestamp.
- [x] 2.2 Implement chunk skipping in `load_rank_micro_steps_from_cache`
      (design Seam B): skip chunks with empty `[start,end)` intersection with
      the rank's required set; every decoded chunk keeps digest verification,
      restricted unpickling, and type validation. Tests: (a) produced
      rank-local sequence identical to the pre-change loader and to the
      repeating-stream oracle across world sizes and `max_steps` shapes;
      (b) a corrupted required chunk still fails closed; (c) a corrupted
      non-required chunk is not decoded by this rank while manifest-level
      declaration checks still pass/fail as specified.
      DONE: `_iter_required_chunks` (new) skips chunks whose declared
      `[start, end)` does not intersect the rank's `required` index set;
      every chunk it does yield goes through the unchanged
      `_load_validated_chunk` (digest verify, restricted unpickle,
      type/count validation) — factored out of the old
      `_iter_validated_chunks` body so `load_all_micro_steps_from_cache`
      (unrelated, still a full pass) is byte-for-byte unchanged.
      `load_rank_micro_steps_from_cache` gained the internal
      `_force_full_chunk_pass: bool = False` kwarg (not a public
      YAML/CLI surface) for the M2 A/B control; manifest-level
      `_validate_manifest` (contiguity/counts/digest-syntax/path-safety/
      existence over every declared chunk) is called unconditionally
      before chunk selection, so it is untouched by skipping. Tests: (a)
      `tests/training/test_pack_cache.py::test_rank_selective_loading_matches_full_pass_and_repeating_stream_oracle`
      (4 parametrized `(max_steps, grad_accum_steps, world_size)` shapes,
      every rank — asserts identity against both `_force_full_chunk_pass`
      arms and the `build_repeating_micro_step_stream` oracle); (b) required
      chunk corruption still fails closed —
      `test_micro_step_cache_rejects_corrupt_required_chunk` (pre-existing,
      unaffected); (c)
      `test_rank_reader_skips_corrupt_chunk_outside_required_set` (new: a
      corrupted non-required chunk's rank load now succeeds without
      decoding it) plus `test_micro_step_cache_manifest_rejects_chunk_gaps`
      (pre-existing, exercises the non-required chunk — proves manifest
      declaration checks still fire for skipped chunks) and updated
      `test_rank_reader_releases_validated_unselected_steps` (indices in a
      skipped chunk are never loaded at all, not merely
      loaded-then-released). Two pre-existing tests asserted the OLD
      full-pass-only behavior and were updated to match the new intended
      semantics (`test_rank_reader_validates_unselected_chunks` renamed to
      `test_rank_reader_skips_corrupt_chunk_outside_required_set` with the
      assertion inverted; `test_rank_reader_releases_validated_unselected_steps`
      updated for chunk-level, not index-level, decode granularity).
- [x] 2.3 Cache-root discipline for this and later waves: all wave-level A/B
      measurements use dedicated temporary `COORDEXP_SWIFT_PACK_CACHE_ROOT`
      roots and sample-limited derived configs; the shared production cache
      root is never written, deleted, or mutated by measurement work. The
      single production materialization for the final fingerprint is
      deferred to task 6.1a.
      DONE: M2 (below) used `COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m2_probe/cache_root`
      exclusively; shared production root
      (`.cache/coordexp_swift/packing`) verified untouched before/after (35
      fingerprint dirs both times, root mtime predates this session). Full
      receipt in `implementation-notes.md` ("M2" section).
- [x] 2.4 Wave gate: `tests/training/test_pack_cache.py`,
      `test_pipeline_assembly.py`, `test_pipeline_pack_cache_rebuild.py`;
      measurement M2 on temporary cache roots — paired A/B (full
      digest-and-payload pass vs rank-selective loading via the internal
      test/benchmark-only control, same code/config/cache; see
      `measurement-plan.md`), time-to-first-planned-step, short-run and
      prod-shaped schedules; accept improvement or neutrality, reject any
      startup regression; shared gate commands.
      DONE: targeted suite 88 passed. M2 paired A/B (real dataset, real
      Qwen processor/tokenizer, `load_model=False`, 3 repetitions each arm,
      internal `_force_full_chunk_pass` control): H-steady-bounded (derived
      625-micro-step temp cache, 2 chunks) rank-selective median 5.779s vs
      full-pass median 7.432s — 22.2% improvement, every repetition favors
      rank-selective; H-short (existing smoke config, 32 micro-steps — only
      1 chunk at the production default `chunk_size=512`, so structurally
      neutral) shows noise-level parity, no regression. **ACCEPT** —
      rank-selective chunk loading (the new default) improves on at least
      one harness and regresses neither. Functional probes (real
      `prepare_train_cache` CLI, disposable 6-row scratch dataset copy,
      production dataset file itself never written): touch → `build_status:
      "hit"`, same fingerprint, no rebuild; 1-byte content change → new
      fingerprint, `build_status: "built"` (miss, rebuilt). Full receipt,
      exact commands, chunk-layout table, and cache-state disclosure (warm
      cache only — this sandbox denies `/proc/sys/vm/drop_caches`) in
      `implementation-notes.md` ("M2" section). Shared gate commands: `ruff
      check` clean on touched files; `git diff --check` clean; `openspec
      validate streamline-coordexp-swift-base-infrastructure --strict` and
      `openspec validate --all --strict` pass (19/19); residue grep for
      `mtime_ns` confirms it remains only in the unrelated
      `image_stat_fingerprint` provenance mechanism
      (`src/data/images.py`), not in packing-cache determinants.

## 3. Bounded Forward-Input Lookahead

- [x] 3.1 Implement the `ForwardInputProvider` protocol and both providers
      per design Seam A: planned-step-scoped lifecycle
      (`begin_planned_step` / `take` / `end_planned_step` / `close`),
      depth-one queue, CPU-only producer, ordinal+pack identity check,
      poison-item error propagation, cancellation-aware put/get, bounded
      join, run-record provider-mode receipt, debug-only synchronous switch.
      Trainer changes are in scope where lifecycle ownership requires them
      (calling begin/end at step boundaries and take per micro-step); do not
      hide step boundaries inside a stateful global closure.
      DONE (revised 2026-08-04 per Opus independent review, HOLD → fixed;
      see `review-triage.md` "Independent implementation review (Opus,
      2026-08-04) — Wave 3"): new module
      `src/training/forward_input_provider.py` — the `ForwardInputProvider`
      `Protocol` (now also declaring the `last_take_wait_seconds: float`
      attribute the trainer reads, per P2-E), `SynchronousForwardInputProvider`
      (reference: two-phase CPU build + separate device move, no thread —
      revised from the original fused build, see P2-A below), and
      `OverlappedForwardInputProvider` (one `threading.Thread` +
      `queue.Queue(maxsize=1)` **plus a `threading.Semaphore(1)` build
      slot** spawned fresh per `begin_planned_step`, joined in
      `end_planned_step`/`close`). Producer always calls
      `build_qwen_forward_inputs(..., device=None)` (CPU-only,
      never CUDA); `take()` moves the CPU-resident `QwenForwardInputs` to
      the micro-step's real `forward_device` via a new
      `_move_forward_inputs_to_device` helper (`.to(device)` on
      input_ids/position_ids/pixel_values/image_grid_thw/fa2 cu_seq_lens_q,k/
      logits_to_keep; receipt object is carried over unchanged, so
      `total_build_inputs_ns` reflects genuinely CPU-only producer time —
      device transfer happens separately on the consumer/`take()` thread,
      never inside the producer).
      **P1-1 fix (true depth-one residency bound):** the original
      `queue.Queue(maxsize=1)` alone did not bound the producer to one
      built-ahead item — a fast producer could finish building ordinal k+1
      (blocking only on `put()`) while ordinal k still sat in the queue
      unconsumed, momentarily holding two fully-built items. Added a
      step-scoped `threading.Semaphore(1)` "build slot":
      `_produce_forward_inputs` now acquires it (cancellation-aware, via
      new `_cancellation_aware_acquire`) *before* starting each build;
      `take()` releases it immediately after dequeuing the previous item —
      *before* the H2D move — so the producer can never start building
      ordinal k+1 until ordinal k has actually been taken. Cancellation
      correctness is preserved: `end_planned_step()`/`close()` still cannot
      deadlock, since a producer blocked on the semaphore acquire (not just
      the queue put) wakes on the same step-scoped `cancel_event` within one
      poll interval.
      Ordinal+pack-identity check (`_StepBoundState.validate_take`) fails
      closed (`training.forward_input_provider_ordinal_skew` /
      `..._pack_skew`) on skew. Producer exceptions are captured as a poison
      `_PreparedItem(error=...)` and re-raised with the original type at the
      affected `take()`. `_cancellation_aware_put`/`_cancellation_aware_get`
      poll a step-scoped `threading.Event` so a full depth-one queue can
      never deadlock `end_planned_step()`/`close()`; `end_planned_step`
      cancels, drains, and joins with a bounded 30s timeout (raises
      `training.forward_input_provider_join_timeout` if exceeded — never
      hangs silently). Debug-only mode switch:
      `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` env var (not YAML;
      default `synchronous` — revised by the 3.4 stop-rule disposition
      below; `overlapped` selectable for measurement), resolved by
      `resolve_forward_input_provider_mode()`. Run-record receipt: new
      `RunWriter.bind_forward_input_provider_mode()` (additive
      `forward_input_provider_mode` field in `run.json`, bind-once
      immutable, same pattern as `bind_schedule`/`bind_materialization`);
      **P2-G fix:** bound only after `SupervisedTrainer` construction
      succeeds (see 3.3's new fail-closed validation), so the run record
      never claims a provider mode that turned out to be unused by a
      non-streaming loss runner.
      **P2-B centralization:** the logits-to-keep helper is no longer
      duplicated — `forward_input_provider.py` imports
      `supervised_trainer._logits_positions_to_keep` directly (no circular
      import: the trainer only references the provider module under
      `TYPE_CHECKING`).
      Trainer wiring (`src/training/supervised_trainer.py`): new
      `forward_input_provider` constructor kwarg (default `None`, fully
      backward compatible — see 3.3); `_run_streaming_planned_step` calls
      `begin_planned_step` right after the existing move loop and wraps the
      rest of the planned-step body (loss-plan prep through
      `zero_gradients`) in `try/finally` so `end_planned_step()` runs on
      every exit path (normal completion, producer error, consumer error,
      finite-gate early break) — matching the design's explicit rejection of
      a step-boundary-blind global closure.
      **P2-D fix:** `step_duration_seconds` is now captured immediately
      after `zero_gradients()`, *inside* the `try` block and before the
      `finally`'s `provider.end_planned_step()` — the original code captured
      it after the `finally`, so provider teardown (cancellation, drain,
      bounded join) was silently included in the measured compute/optimizer
      boundary.
      **P2-B fix (construction-time guard):** `SupervisedTrainer.__init__`
      now raises `RuntimeContractError`
      (`trainer.forward_input_provider_conflicts_with_custom_qwen_forward`)
      if both `forward_input_provider` and a custom `qwen_forward` are
      passed — the provider path calls `run_qwen_forward` directly and
      would otherwise silently ignore the custom callable.
      **P2-G fix (construction-time guard):** `SupervisedTrainer.__init__`
      also raises `RuntimeContractError`
      (`trainer.forward_input_provider_requires_streaming_loss_runner`) if
      `forward_input_provider` is paired with a non-streaming loss runner —
      the non-streaming batch path never consults the provider, which would
      otherwise leave it silently unused while the run record still claimed
      its mode (no new "unused" mode value was added; the combination is
      rejected outright, per the review's explicit instruction).
      Pipeline wiring (`src/training/pipeline.py`): `_run_initialized_training`
      builds the provider once per run via `resolve_forward_input_provider_mode()`
      / `build_forward_input_provider()`, passes it to `SupervisedTrainer`,
      binds the mode to the run record only after construction succeeds
      (P2-G), and closes it in the same `finally` scope that closes the
      rank-report gatherer — **P2-F fix:** now a *nested* `try/finally`
      (`try: forward_input_provider.close() finally: <close rank-report
      gatherer>`) so a provider-close failure (e.g. the bounded-join
      timeout) cannot skip the rank-report gatherer close.
- [x] 3.2 Unit tests covering every lifecycle transition: (a) prepared
      inputs byte-identical to synchronous construction for a fixture pack
      set; (b) producer exception surfaces at the exact affected micro-step
      with the original error type; (c) consumer error mid-step cancels and
      joins cleanly; (d) finite-gate early break discards
      prepared-but-unconsumed items and the next planned step starts clean
      (no stale/skewed item), including with scheduled eval between the
      steps; (e) full-queue shutdown cannot deadlock (cancellation-aware
      put/get); (f) ordinal/pack skew raises a contract error; (g) at most
      one prepared item beyond the executing step; (h) provider mode
      recorded in the run record in both modes.
      DONE, revised 2026-08-04 (Opus independent review HOLD → fixed same
      day; see `review-triage.md`): code and tests below are complete and
      green, including two new tests fixing P1-2 (the original (g) test was
      tautological — it only checked `queue.maxsize`/`qsize()`, which holds
      trivially regardless of whether the depth-one bound is actually
      enforced by build ordering). A narrow independent re-review
      specifically confirming the P1-1 depth-one fix is still recorded as
      pending in `review-triage.md`'s "Disposition" section, but no longer
      blocks this checkbox now that the fix, its decisive tests, and task
      3.4's stop-rule disposition are all complete and honestly described.
      Test suite: `tests/training/test_forward_input_provider.py` (19 tests,
      real `build_qwen_forward_inputs`/`plan_packed_sequences`/
      `build_qwen_position_inputs` fixtures, no fakes for the
      provider-under-test itself) —
      (a) `test_synchronous_and_overlapped_providers_produce_byte_identical_forward_inputs`
      (both providers' output compared tensor-by-tensor, including FA2
      `cu_seq_lens_q/k`, against a direct fused `build_qwen_forward_inputs`
      call); (b)
      `test_overlapped_provider_surfaces_producer_exception_at_the_affected_ordinal`
      (ordinal 0 succeeds, ordinal 1's monkeypatched failure re-raises the
      exact custom exception type at `take(1, ...)`); (c)
      `test_overlapped_provider_end_planned_step_after_partial_consumption_joins_cleanly`
      (bounded-time join after only one of three ordinals consumed); (d)
      `test_overlapped_provider_discards_unconsumed_items_and_starts_next_step_clean`
      (ordinal 2 never consumed in step 1; step 2 begins at ordinal 0 clean,
      no stale item); (e)
      `test_overlapped_provider_end_planned_step_does_not_deadlock_on_a_full_queue`
      (never calls `take()`, forcing the producer to block on the build slot
      after filling the queue, then asserts bounded-time
      `end_planned_step()`); (f) `test_provider_take_fails_closed_on_ordinal_skew`
      and `test_provider_take_fails_closed_on_pack_identity_skew` (both
      providers, both error codes); (g) **replaced per P1-2** —
      `test_overlapped_provider_gates_next_build_until_prior_item_is_taken`
      (decisive: monkeypatches `_build_forward_inputs` to record build
      order, asserts ordinal 1's build has NOT started after a bounded
      0.3s wait with *no* `take()` call yet — manually verified this test
      FAILS against the pre-fix code, by temporarily stripping the
      `_cancellation_aware_acquire` gate from a scratch copy of the source
      and re-running it: `build_started == [0, 1]` within the wait window,
      confirming the test is decisive, not tautological) and
      `test_overlapped_provider_built_minus_taken_never_exceeds_one`
      (continuous invariant `builds_completed - items_dequeued <= 1`,
      sampled from a background watcher thread across a full 5-micro-step
      step with no artificial delay; `items_dequeued` is measured at the
      internal `_cancellation_aware_get` dequeue point — the same event
      that triggers the build-slot release — not after the caller's
      `take()` including its H2D move, which would race against the
      producer and produce false positives unrelated to the actual bound);
      (h) `tests/artifacts/test_run_artifacts.py::test_forward_input_provider_mode_can_be_bound_once_per_mode`
      (parametrized `overlapped`/`synchronous`) and
      `::test_forward_input_provider_mode_rejects_unknown_values`.
      New tests for the P2 fixes: `test_both_provider_modes_build_on_cpu_and_move_separately_for_comparable_timing`
      (P2-A: both providers' `_build_forward_inputs` calls use `device=None`
      and both call `_move_forward_inputs_to_device` separately, so
      `input_build_seconds` means the same CPU-only thing in both modes);
      `test_move_to_device_covers_every_tensor_field_used_by_to_model_kwargs`
      (P2-C: uses `torch.device("meta")` — no real CUDA needed — to prove
      every tensor field `to_model_kwargs()` sends to the model was actually
      moved; manually verified decisive by temporarily reverting one field's
      `.to(device=...)` call in a scratch copy and confirming the test fails
      with the exact offending field name in the assertion message);
      `tests/training/test_supervised_trainer.py::test_forward_input_provider_rejects_combination_with_custom_qwen_forward`
      and `::test_forward_input_provider_rejects_non_streaming_loss_runner`
      (P2-B/P2-G construction-time guards; both assert zero provider
      lifecycle calls happened, since construction fails before `run()`);
      `::test_streaming_planned_step_duration_excludes_provider_teardown`
      (P2-D: incrementing-fake-clock proof — a `SlowTeardownProvider` test
      double consumes two extra ticks inside `end_planned_step()`;
      `step_duration_seconds` reads the correct pre-teardown value and is
      explicitly asserted NOT to equal the teardown-inclusive value).
      Additional lifecycle coverage beyond the (a)-(h) list:
      `test_provider_lifecycle_state_machine_fails_closed_on_misuse`
      (idle/active state machine: take-before-begin, second begin without
      end, end-while-idle, use-after-close, each asserted for both
      providers), `test_provider_close_is_idempotent_for_both_modes`,
      `test_overlapped_provider_leaves_no_leaked_thread_after_close`
      (`threading.active_count()` returns to baseline), and
      `test_overlapped_provider_producer_always_builds_on_cpu_regardless_of_target_device`
      (regression proof that the producer always calls
      `build_qwen_forward_inputs(..., device=None)` even when the
      micro-step's real target is a non-existent `cuda:7` — the producer
      thread never attempts that device). Trainer-level wiring tests in
      `tests/training/test_supervised_trainer.py` (task 3.3 evidence, below)
      cover lifecycle ordering as seen by the trainer using a lightweight
      `FakeForwardInputProvider` test double.
- [x] 3.3 Wire the provider through training assembly; `input_wait_seconds`
      and `input_build_seconds` populate from the provider.
      DONE: `_run_streaming_planned_step` now calls
      `provider.take(local_micro_step_index, micro_step)` then
      `run_qwen_forward(...)` directly (bypassing the fused
      `self.qwen_forward`/`_default_qwen_forward` path) whenever
      `forward_input_provider` is not `None`; `provider is None` (the
      default) leaves every existing call site byte-for-byte unchanged, so
      the full pre-Wave-3 trainer test suite required zero changes beyond
      one pipeline-assembly fixture gaining the new `writer` method stub.
      `input_wait_seconds` accumulates `provider.last_take_wait_seconds`
      after every `take()` call (0.0 for the synchronous provider by
      construction; real queue-wait time for the overlapped provider,
      timed around `_cancellation_aware_get` in `take()` itself — not
      conflated with the producer-side `total_build_inputs_ns` that feeds
      `input_build_seconds`). Pipeline wiring: `_run_initialized_training`
      passes `forward_input_provider=forward_input_provider` into
      `SupervisedTrainer(...)`. Tests:
      `tests/training/test_supervised_trainer.py::test_streaming_planned_step_uses_forward_input_provider_when_present`
      (asserts exact `["begin", "take:0", "take:1", "end"]` call order and
      `input_wait_seconds`/`input_build_seconds` accumulation from a fake
      provider double),
      `::test_streaming_planned_step_ends_provider_cleanly_on_finite_gate_early_break`
      (`["begin", "take:0", "end"]` — ordinal 1 never requested),
      `::test_streaming_planned_step_ends_provider_on_consumer_error`
      (`run_qwen_forward` raises; `end_planned_step` still runs before the
      exception propagates). Pipeline-level:
      `tests/training/test_pipeline_assembly.py::test_same_dataset_eval_resolves_distinct_full_cache_and_binding`
      extended to assert `provider_modes == ["synchronous"]` (default mode
      bound to the run record end to end through real
      `_run_initialized_training` wiring, mocked trainer/model layers;
      updated from `["overlapped"]` when the default flipped per the 3.4
      stop-rule disposition below).
      REVISED 2026-08-04 per Opus independent review (see
      `review-triage.md`): P2-B and P2-G added two construction-time
      fail-closed guards described in full under 3.1 (custom-`qwen_forward`
      conflict; non-streaming-loss-runner pairing) — both covered by new
      tests `test_forward_input_provider_rejects_combination_with_custom_qwen_forward`
      and `test_forward_input_provider_rejects_non_streaming_loss_runner`.
      P2-D moved the `step_duration_seconds` capture point to exclude
      provider teardown (`test_streaming_planned_step_duration_excludes_provider_teardown`).
      P2-G also moved the pipeline's `writer.bind_forward_input_provider_mode`
      call to after `SupervisedTrainer(...)` construction succeeds (see 3.1);
      the `test_same_dataset_eval_resolves_distinct_full_cache_and_binding`
      assertion above is unaffected since the mocked trainer never raises.
      Still checked: the wiring itself (provider consulted per micro-step,
      timings populated) is unaffected by these additive guards/ordering
      fixes and remains verified by the tests above plus the full targeted
      gate suite.
- [x] 3.4 Wave gate: trainer/pipeline/runtime tests; measurement M3 on
      temporary cache roots — paired A/B (steps/hour, overlapped vs
      synchronous provider via the existing debug switch, same
      code/config/cache) on the production-shaped smoke; loss-stream
      equivalence: identical `loss/*` and gate decisions between modes on a
      fixed 2-step run, including a synthetic finite-gate early-break case.
      Focused independent review of the concurrency lifecycle. Stop rule: if
      steps/hour does not improve, revert the wiring, keep the tests, record
      rejected-with-evidence.
      DONE, stop rule applied 2026-08-04 (Opus independent review HOLD →
      fixed same day; see `review-triage.md`): overlap is fully implemented
      and its semantic equivalence to the synchronous reference path is
      proven decisively (deterministic loss-stream-equivalence harness,
      byte-identical-inputs test, full lifecycle test matrix — all below).
      The world_size=1 M3 evidence collected against the depth-one-corrected
      code did not establish a steps/hour improvement (2 of 5 repetitions
      favored overlap; see the two measurement sessions below) — per the
      stop rule stated by this task, **overlap is rejected as the shipped
      default and reverted to `synchronous`; the tests and implementation
      are kept**, recorded as rejected-with-evidence rather than silently
      dropped. This checkbox is now checked because that disposition is
      itself the honest, complete outcome the stop rule specifies — not
      because the exact named 8-rank harness has been run. A narrow
      independent re-review confirming the P1-1 depth-one fix and this
      final disposition is still recorded as a residual item in
      `review-triage.md` (see its "Disposition" section).
      This is a deliberate narrow deviation from the planning shorthand
      "revert the wiring": the overlapped arm was demoted and the
      synchronous default restored, but the shared two-phase provider wiring
      remains because it owns the validated lifecycle and comparable timing
      boundary. The retained wiring and required future three-arm comparison
      are recorded below; no unmeasured performance claim is made for it.
      Gate suite: `tests/runtime/ tests/training/ tests/losses/
      tests/config/ tests/artifacts/ tests/eval/ tests/qwen/` — 481 passed
      (2026-08-04, after the P1/P2 fixes; up from 475 — net +6 tests: 2
      decisive depth-one tests replacing 1 tautological one, plus P2-A/C/D/G
      coverage). `ruff check` clean on every touched/new file. `git diff
      --check` clean. `openspec validate streamline-coordexp-swift-base-infrastructure --strict`
      and `--all --strict` (19/19) both pass. Loss-stream equivalence: deterministic real-provider,
      real-`build_qwen_forward_inputs`/`run_qwen_forward`, fake-echo-model
      harness in `tests/training/test_forward_input_provider.py` —
      `test_loss_stream_and_gate_decisions_are_identical_between_provider_modes_on_a_fixed_run`
      (fixed 2-step, grad_accum=2 run; `loss_bundle_artifact`,
      `optimizer_update_status`, `finite_status`, `micro_step_count` exactly
      equal between modes) and
      `test_loss_stream_and_gate_decisions_are_identical_between_provider_modes_on_a_synthetic_early_break`
      (call-index-driven synthetic unsafe pre-backward gate forces an early
      break identically in both modes; step 1 breaks after 1 micro-step in
      both, step 2 completes both micro-steps in both — full artifact
      equality asserted). **P2-H (Opus review) — M3 result restated
      honestly, now on TWO measurement sessions (pre- and post-P1-1-fix):**
      the exact named H-short/H-steady harnesses require `accelerate8` (8
      GPU ranks); at every measurement time in this task, GPUs 2-7 were
      either external-process-occupied or the full 8-rank harness was
      otherwise not attempted, so the full named harness has still never
      been run — every number below is a same-code/same-config(-except-
      world-size)/same-temporary-cache scoped single-GPU
      (`CUDA_VISIBLE_DEVICES=0`, world_size=1) substitute (temporary derived
      config `extends`-ing the exact H-steady prod base path,
      `data.train.sample_limit: 64`/`data.eval.sample_limit: 16`/
      `training.max_steps: 6`/`training.effective_batch_size: 4`/
      `checkpoint.save_final: false`, scratch-only, not committed; same
      temporary pack cache reused by every arm, confirmed via identical
      `materializations.train.semantic_fingerprint`; toggling only
      `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`).
      *Session 1 (pre-fix code, 3 repetitions, GPUs 2-7 busy at the time):*
      mean `step_duration_seconds` synchronous `[10.469, 10.509, 10.607]` s
      vs overlapped `[10.374, 10.472, 10.423]` s — overlapped won every
      repetition (median steps/hour 342.6 sync vs 345.4 overlapped, +0.8%).
      *Session 2 (current fixed code — P1-1 depth-one gate + P2-A CPU-only
      synchronous build — 5 repetitions, all 8 GPUs idle at the time):* mean
      `step_duration_seconds` synchronous
      `[10.450, 10.599, 10.524, 10.643, 10.995]` s vs overlapped
      `[10.483, 10.544, 10.568, 10.689, 10.799]` s — **only 2 of 5
      repetitions favored overlapped** (per-repetition steps/hour change:
      `[-0.31%, +0.52%, -0.42%, -0.43%, +1.81%]`); median steps/hour 339.7
      sync vs 340.6 overlapped (+0.27%, i.e. noise-level parity, not a
      consistent win); mean 338.4 vs 339.1. Both sessions show real,
      nonzero `input_wait_seconds` for overlapped (`0.0` for synchronous by
      construction vs `~0.16-0.40s`/step for overlapped, confirming the
      queue-wait mechanism is genuinely exercised), but session 2 also shows
      `input_build_seconds` for the overlapped producer thread running
      consistently ~1.5-2x *higher* than the synchronous provider's CPU
      build time in the same repetition (e.g. rep 5: sync 0.848s vs overlap
      1.519s) despite both now building via the identical CPU-only
      `_build_forward_inputs(..., device=None)` code path (P2-A) — the
      leading hypothesis is CPython GIL contention between the producer
      thread's CPU-bound build work and the main/consumer thread's own
      Python-level GPU-orchestration overhead (tensor dispatch, autograd
      bookkeeping around `model.forward()`/`backward()`), which would
      inflate the producer's *wall-clock* build time without it doing more
      real work; this was not investigated further (would require
      `time.thread_time()`/`time.process_time()` instrumentation, out of
      scope for this fix pass) and is recorded as an open question, not a
      confirmed cause. Both sessions also show build times trending upward
      across repetitions within session 2 specifically (rep 1 sync 0.379s →
      rep 5 sync 0.848s), consistent with host-load drift across the
      session (thermal, page cache, or other contention) adding further
      noise on top of the mode comparison.
      **Final disposition, stop rule applied: overlap is REJECTED as the
      shipped default; `synchronous` ships as the default provider mode.**
      Session 1's "every repetition favors overlapped" result did not
      replicate in session 2 against the depth-one-corrected code with more
      repetitions (2/5, noise-level median difference) — the honest summary
      is that this world_size=1 substitute cannot distinguish a small real
      effect from measurement noise at this scale, so it does not establish
      the "steps/hour improves" condition the stop rule requires for
      keeping overlap as the default. Applying the stop rule literally: the
      wiring is reverted to the synchronous default
      (`resolve_forward_input_provider_mode()` now returns `synchronous`
      when unset), the implementation and full test suite are kept
      (`overlapped` remains fully implemented, semantically proven
      equivalent, and selectable via the existing debug-only
      `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE=overlapped` switch), and
      this is recorded as rejected-with-evidence rather than silently
      dropped or quietly kept as a default without supporting evidence.
      This is not a claim that overlap does not work or never will: (1) no
      *correctness* regression was found under either mode (proven
      decisively and deterministically, independent of GPU timing noise);
      (2) the design's own stated expectation is that this mechanism's win
      concentrates at larger world size and larger CPU-build-to-GPU-compute
      ratios (larger images, more micro-steps per rank) — a single-GPU,
      2B-parameter, single-image-pack probe is explicitly the least
      favorable shape for this mechanism per `design.md`'s own caveat, so
      this null/mixed result at world_size=1 does not by itself prove
      overlap would fail to win at 8-rank production scale; (3) the
      mechanism-level plumbing (`input_wait_seconds` genuinely nonzero and
      bounded by `input_build_seconds`) is demonstrably real — only its net
      wall-clock payoff at this specific tiny scale is unproven. The exact
      named 8-rank H-short/H-steady harness remains **future confirmation
      work, not yet run**, and is the only evidence that can promote
      overlap back to the shipped default — it may do so if it demonstrates
      a real improvement there; absent that confirmation, `synchronous`
      stays the default. Correctness cross-check (unpaired, informational
      only): `loss/total` was bit-identical between arms for planned steps
      1-2 of session 1's first repetition (before optimizer updates could
      compound ordinary cross-process GPU floating-point non-determinism);
      this is why the decisive loss-stream-equivalence claim above uses the
      deterministic CPU-level harness instead of any live GPU comparison.
      No production cache root or unrelated GPU/process was touched in
      either session (GPUs 2-7 untouched in session 1; all 8 GPUs confirmed
      idle before session 2 and only GPU 0 used; production
      `.cache/coordexp_swift/packing` fingerprint-dir count and root mtime
      unchanged before/after both sessions — 35 dirs, mtime predates this
      task); all scratch configs/caches/outputs from both sessions were
      deleted after measurement.
      **P2 recorded for the future 8-rank M3 (not built now):** the
      synchronous provider is no longer the old fused single-call
      construction — P2-A changed it to the same two-phase (CPU build then
      separate device move) shape as the overlapped provider, purely so
      `input_build_seconds` would be comparable across modes. This means
      the *current* synchronous-vs-overlapped A/B no longer also tells us
      whether the two-phase construction itself (independent of
      threading/queueing) costs anything relative to the original
      single-call fused build (i.e. a `forward_input_provider=None` /
      legacy `_default_qwen_forward` arm). The exact future 8-rank M3
      should therefore compare three arms — legacy fused
      (`forward_input_provider=None`), two-phase synchronous, and
      overlapped — rather than only two, so a two-phase-construction
      overhead is not silently attributed to (or hidden inside) the
      overlap/no-overlap comparison. No third-arm benchmark is built as
      part of this fix pass; this is recorded as a P2 for whoever runs the
      eventual 8-rank confirmation.
      OUTSTANDING: (1) the "focused independent review of the concurrency
      lifecycle" named by this task is a separate reviewer pass; the Opus
      pass recorded in `review-triage.md` below is exactly that review and
      returned HOLD, fixed same day including this final stop-rule
      disposition — a narrow independent re-review specifically confirming
      the P1-1 depth-one fix and this final disposition is still recorded
      as pending in `review-triage.md`'s "Disposition" section, but no
      longer blocks this checkbox (the stop-rule outcome itself is complete
      and honestly described). (2) Re-running the full named 8-rank
      H-short/H-steady `accelerate8` harness — with the three-arm
      methodology above — remains the only path to promoting `overlapped`
      back to the shipped default.

## 4. Rank-Sharded eval.forward With Exact Aggregation

- [x] 4.1 Add failing equivalence test over the ENTIRE canonical eval row:
      sharded eval on a multi-rank fixture with unequal per-rank atom counts
      must reproduce the replicated evaluator's complete row — exact integer
      counts and `example_count`/`pack_count` sums, exact top-k from summed
      integer statistics, loss and `token_weighted_diag` scalars within rtol
      1e-5, global-denominator-derived fields emitted once (not
      rank-summed), finite/non-finite handling unchanged (design Seam C
      inventory table). Add the fallback fixture: a multi-rank eval with
      `pack_count < world_size` must run replicated with current replicated
      reduction semantics, produce a row identical to the pre-change
      replicated evaluator, and multiply no count or total by the world
      size.
      DONE: `tests/eval/test_forward_eval.py` gained a 3-pack fixture
      (unequal atom counts 2/1/2 across packs 0/1/2, `pack_count=3 >=
      world_size=2`) driven through the REAL `LossRunner` and REAL
      `TrainRuntime` (not a reimplementation) via a small thread-based
      `_ThreadedRankCollective` that simulates the production bounded
      gatherer with two concurrent rank threads — decisive because it
      exercises the actual production `TrainRuntime._reduce_metric_reports`
      code path, not a test-only stand-in.
      `test_disjoint_shard_eval_reproduces_full_replicated_row_with_unequal_rank_atom_counts`
      asserts both ranks derive an identical row, and that row matches an
      independently-computed world_size=1 replicated reference across the
      entire canonical row: `example_count`/`pack_count` exact sums,
      `acc_top1`/`acc_top5` exact, `loss/total` and every `loss/<term>` and
      `loss/<term>/token_weighted_diag` within rtol 1e-5,
      `loss/<term>/segment_count` and `count/supervised_atoms`/
      `count/eligible_segments`/`count/skipped_segments` exactly equal
      (single-emission, not rank-summed), `count/packs`/`count/examples`
      exact rank-local sums, identical row key sets, and no internal
      `__weight__` key leaking into the durable row.
      `test_disjoint_shard_eval_rejects_naive_disjoint_sum_of_local_losses`
      names the `mean_r(W*c_r) = sum_r(c_r)` identity explicitly against the
      independently-computed reference.
      `test_replicated_fallback_when_pack_count_below_world_size_keeps_pre_change_semantics`
      runs 4 simulated ranks with only 3 packs (`pack_count < world_size`),
      asserts `resolve_active_eval_reduction_mode` returns `replicated`, and
      asserts every rank's row is byte-identical to the replicated reference
      with `pack_count`/`example_count` NOT multiplied by world_size (4).
      Narrower decisive unit tests for the two new `TrainRuntime` reducer
      kinds (`tests/runtime/test_train_runtime.py`):
      `test_eval_disjoint_shard_sum_keys_sum_rank_local_counts_not_mean`,
      `test_eval_disjoint_shard_identical_keys_require_exact_cross_rank_agreement`,
      `test_eval_disjoint_shard_identical_key_mismatch_fails_closed`,
      `test_eval_disjoint_shard_sum_key_rejects_non_integer_count`,
      `test_eval_disjoint_shard_token_weighted_diag_sum_propagates_nonfinite`,
      `test_eval_disjoint_shard_reduction_mode_mismatch_across_ranks_fails_closed`,
      `test_eval_replicated_mode_never_activates_new_sum_or_identical_reducers`
      (proves the default/replicated call path is byte-unaffected by the new
      reducers).
      REVISED 2026-08-04 per Opus independent review P1-1/P1-2 (HOLD, fixed
      same day; see `review-triage.md` "Wave 4"): the original fixture
      matrix only covered the all-finite case, so it missed a real bug --
      `finite/*` keys fell through to the default plain-mean reducer in
      `disjoint_shard` mode, producing a meaningless 0.5 when ranks
      disagreed on finiteness instead of the replicated evaluator's correct
      0.0. Added the decisive fixture
      `test_disjoint_shard_eval_nonfinite_shard_reduces_finite_flags_by_and_not_mean`
      (real `LossRunner`/`TrainRuntime`, a genuinely non-finite loss from a
      NaN logits row on one rank's shard, full `finite/*` and real
      `RunWriter` writer-facing null/`non_finite_fields` parity against the
      replicated reference) plus the narrower
      `test_eval_disjoint_shard_finite_keys_reduce_by_and_not_mean` /
      `test_eval_disjoint_shard_finite_key_rejects_malformed_value` /
      `test_eval_replicated_mode_never_activates_new_finite_and_reducer`
      (`tests/runtime/test_train_runtime.py`). Manually verified decisive
      by reverting the fix in a scratch copy and confirming both the new
      end-to-end and unit tests fail with the exact `0.5 == 0.0` mismatch,
      then restoring the fix and re-confirming green.
- [x] 4.2 Implement deterministic disjoint pack sharding
      (`pack_ordinal % world_size`), the compact explicit eval reduction
      payload carrying rank-local sufficient statistics through the existing
      gatherer, global-denominator gathering via the existing
      `prepare_planned_step` path, and derivation of the identical global
      row on every rank. The active reduction mode (disjoint-shard vs
      replicated) must be explicit in the reduction payload or call path,
      never inferred from payload shape; the replicated fallback
      (`pack_count < world_size`) keeps current replicated semantics with
      identical rank values reduced once, never summed. Include the identity
      test `mean_r(W * c_r) = sum_r c_r` for the W-scaled loss-contribution
      route.
      DONE: `partition_eval_micro_steps_for_rank` (`src/eval/forward.py`)
      filters the canonical, already-ordered `eval_micro_steps` tuple by
      `micro_step.pack.pack_index % world_size == rank` — a pure filter,
      never a reorder, so within-rank forward order is unchanged.
      `resolve_active_eval_reduction_mode(*, pack_count, world_size)` decides
      the active mode exactly once at pipeline-assembly time
      (`_run_initialized_training`, `src/training/pipeline.py`): `world_size
      <= 1` or `pack_count < world_size` unconditionally forces `replicated`
      (the automatic structural fallback, matching the training-artifacts
      delta's "Replicated fallback keeps replicated reduction" scenario);
      otherwise the internal debug/test-only
      `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` env var
      (`resolve_eval_reduction_control()`, values `auto`/`replicated`, unset
      defaults to `auto` after task 4.4's measured promotion) decides whether
      disjoint-shard is attempted
      — mirroring the Wave 3 `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`
      pattern exactly, not a public YAML/CLI surface. The resolved mode is
      threaded explicitly end to end: pipeline.py resolves it once, passes
      it into `_eval_forward_handler(reduction_mode=...)`, which passes it
      into `ForwardEvalRunner(reduction_mode=..., world_size=..., rank=...)`
      for every scheduled eval invocation — never re-inferred from payload
      shape at reduction time. `ForwardEvalRunner.__init__` fails closed
      (`RuntimeContractError`) on an unknown mode, `disjoint_shard` at
      `world_size<=1`, or `disjoint_shard` paired with a non-streaming loss
      runner (mirrors Wave 3's P2-B/P2-G construction-time guard pattern).
      In `disjoint_shard` mode, `_run_streaming_forward_only` now threads
      `denominator_gatherer` (reusing `supervised_trainer.py`'s existing
      private `_runtime_loss_denominator_gatherer` helper, not duplicated),
      `world_size`, and `rank` into `loss_runner.prepare_planned_step` —
      exactly the same global-denominator-gather path training already uses
      — so `backend_gradient_scale=world_size` and every rank's local
      `loss/<term>` metric is already the W-scaled contribution the existing
      plain-mean reducer needs; `replicated` mode's call is completely
      unchanged (bare `prepare_planned_step(tuple(micro_steps))`, no new
      kwargs), guaranteeing byte-identical replicated/world_size=1 rows.
      New compact eval-specific sufficient-statistics payload/reducer
      (`src/runtime/train_runtime.py`, gated by a new `reduction_mode`
      kwarg on `TrainRuntime.gather_metrics`, `None` by default so every
      train call and every replicated eval call is byte-unaffected): a
      `disjoint_shard`-only per-key classification adds two new reducer
      kinds to the existing fixed declared mapping (still no generic
      framework) — `_reduce_eval_sum_metric` (plain cross-rank sum: rank-local
      `example_count`/`pack_count`/`count/packs`/`count/examples`, and the
      pre-weighted `.../token_weighted_diag` product plus its companion
      `.../token_weighted_diag/__weight__`) and `_reduce_eval_identical_metric`
      (already-global fields from the merged denominator —
      `count/supervised_atoms`/`count/eligible_segments`/
      `count/skipped_segments`, `.../segment_count` — validated identical
      across ranks and fail-closed on mismatch, emitted once rather than
      summed). `_checked_eval_count_value` fails closed on any non-finite,
      negative, or non-integer count-like value. The reduction mode itself
      is validated identical across every gathered rank report
      (`runtime.metric_gather_reduction_mode`), alongside the pre-existing
      step/split/key identity checks. `example_count`/`pack_count` are
      injected as ordinary sum-reduced metric keys before the gather
      (`_prepare_disjoint_shard_scalars`) and popped back out as the
      row-level dataclass fields after it (`_finalize_disjoint_shard_scalars`,
      which also divides each `.../token_weighted_diag` sum by its `sum_r`
      weight to recover the exact `sum_r(value_r*count_r)/sum_r(count_r)`
      weighted average and deletes the internal `__weight__` key before it
      can reach the durable row) — so `TrainRuntime`'s new logic stays a
      pure per-key sum/identical extension, with the weighted-average
      derivation living in the eval-specific caller, not baked into the
      shared collective.
      `mean_r(W*c_r) = sum_r(c_r)` identity: proven by
      `test_disjoint_shard_eval_rejects_naive_disjoint_sum_of_local_losses`
      (task 4.1) comparing the sharded row's `loss/total` against the
      independently-computed replicated reference within rtol 1e-5.
      REVISED 2026-08-04 per Opus independent review (HOLD, fixed same day;
      see `review-triage.md` "Wave 4"):
      **P1-1 (finite/* reduction):** `finite/*` keys were missing from the
      per-key classification and fell through to the default plain-mean
      reducer, producing a fractional (meaningless) value whenever ranks
      disagreed on finiteness. Added `_reduce_eval_finite_metric` (logical
      AND via `min` over per-rank 0.0/1.0 flags) and `_checked_eval_finite_flag`
      (fail-closed on any value other than exactly 0.0/1.0), gated the same
      way as the sum/identical reducers -- `disjoint_shard` only, replicated
      path unaffected. See task 4.1's addendum for the decisive test.
      **P2-A (sharding key):** the partition key was
      `micro_step.pack.pack_index % world_size` -- `pack_index` is an
      identity label, not a guaranteed position-matching ordinal for every
      materialized sequence. `partition_eval_micro_steps_for_rank` now
      shards by `enumerate(micro_steps)`'s position
      (`sequence_ordinal % world_size == rank`); pack identity is still
      validated fail-closed (renamed `_validate_pack_identity`, was
      `_pack_ordinal`) but no longer used as the modulus. Decisive
      non-contiguous-pack_index fixture:
      `test_partition_eval_micro_steps_for_rank_uses_sequence_position_not_pack_index_value`
      (`tests/eval/test_forward_eval.py`).
      **P2-B (cross-rank liveness):** the `reduction_mode` identity check
      inside `_reduce_metric_reports` only fires once an eval invocation's
      metric gather runs -- too late, since the `disjoint_shard` path's
      `gather_loss_denominators` collective (inside `ForwardEvalRunner`,
      task 4.2's own new wiring) runs first and only a rank that locally
      resolved `disjoint_shard` ever calls it; a genuine divergence would
      hang a `replicated` peer rather than fail closed. Added
      `TrainRuntime.validate_eval_reduction_consensus(reduction_mode,
      pack_count)` (`src/runtime/train_runtime.py`), reusing the existing
      bounded rank-report gatherer (no new collective machinery, no public
      knob); `_run_initialized_training` (`src/training/pipeline.py`) calls
      it unconditionally on every rank immediately after resolving
      `eval_reduction_mode`, strictly before the sharding filter or any
      handler construction. Fails closed on mode or `pack_count`
      disagreement. Tests:
      `test_eval_reduction_consensus_passes_when_ranks_agree`,
      `test_eval_reduction_consensus_allows_matching_none_pack_count`,
      `test_eval_reduction_consensus_fails_closed_on_mode_mismatch`,
      `test_eval_reduction_consensus_fails_closed_on_pack_count_mismatch`,
      `test_eval_reduction_consensus_is_noop_at_world_size_one`
      (`tests/runtime/test_train_runtime.py`, real `TrainRuntime` gather
      path); pipeline wiring proven by
      `test_same_dataset_eval_resolves_distinct_full_cache_and_binding`'s
      new `consensus_calls` assertion
      (`tests/training/test_pipeline_assembly.py`).
      **P2-D (speculative-code note):** the `local_value is None` handling
      in `_prepare_disjoint_shard_scalars` for a term's `token_weighted_diag`
      value is unreachable under the current `LossRunner` (always a real
      float); assessed and left as-is per the review's explicit
      no-speculative-machinery instruction -- no code change.
      **P2-E (misleading docs):** fixed the module-level comment,
      `partition_eval_micro_steps_for_rank`'s docstring, and the error code
      (`eval_forward.pack_ordinal_missing` -> `eval_forward.pack_identity_missing`)
      to stop calling `pack_index` "the ordinal"; also updated `design.md`
      Seam C's sharding paragraph.
- [x] 4.3 Preserve eval row schema and best-checkpoint selection: assert the
      selector consumes an exact `acc_top1`; assert single `eval` row per
      invocation; optionally reuse the provider protocol for the eval loop
      (gated by the same equivalence test).
      DONE: eval row schema is unchanged — `ForwardEvalObservation`/
      `to_logging_row()` untouched; the new `example_count`/`pack_count`/
      `__weight__` plumbing is entirely internal to the pre-gather/
      post-gather reduction and never reaches the dataclass or the row
      (asserted directly:
      `test_disjoint_shard_eval_reproduces_full_replicated_row_with_unequal_rank_atom_counts`
      asserts `set(sharded_row) == set(replicated_row)` and that no row key
      ends with `__weight__`). `ForwardEvalRunner.run()` still returns
      exactly one `ForwardEvalObservation` per invocation regardless of
      mode (unchanged control flow), so the pre-existing
      `test_five_train_and_two_eval_callbacks_write_exact_wide_rows`
      (`tests/training/test_pipeline_assembly.py`, unmodified, still
      passing) continues to prove one `eval` row per scheduled event.
      `test_best_checkpoint_selector_consumes_exact_acc_top1_from_sharded_row`
      mirrors `pipeline._checkpoint_handler`'s exact selector read
      (`eval_observation.get("acc_top1")`) and asserts the sharded row's
      `acc_top1` is both float-typed and exactly equal to the replicated
      reference's — selection is provably unaffected by which reduction
      mode produced the row. The provider protocol
      (`ForwardInputProvider`, Wave 3) was NOT reused for the eval loop:
      the eval loop's CPU-vs-GPU shape (no backward, no lookahead-worthy
      inter-step gap) does not materially simplify under it, so this
      remains a plain synchronous per-micro-step loop, per the optional/
      not-required clause.
- [x] 4.4 Wave gate: `tests/eval/`, `tests/losses/test_runner.py`,
      trainer/pipeline tests; measurement M4 on temporary cache roots —
      paired A/B (eval invocation wall time at world size 8, sharded vs
      replicated via the internal test/benchmark-only reduction-mode
      control, same code/config/cache); world-size-1 degeneracy
      byte-identical rows; checkpoint-selection replay on a recorded run.
      Focused independent review of the distributed reduction. Stop rule:
      revert to replicated evaluation if full-row exactness cannot be
      demonstrated on the fixture matrix.
      PARTIAL 2026-08-04: test/hygiene gate is fully green; M4's exact
      8-rank GPU evidence and the focused independent review are honestly
      outstanding (GPUs were occupied by unrelated external work at
      measurement time — see `implementation-notes.md` "M4" for the
      before/after `nvidia-smi` receipts). Per the stop rule's own
      criterion ("if full-row exactness cannot be demonstrated on the
      fixture matrix") the fixture matrix in tasks 4.1-4.3 DOES demonstrate
      full-row exactness decisively (real `LossRunner` + real `TrainRuntime`
      reduction, not a reimplementation), so there is no correctness basis
      to revert; `disjoint_shard` remains fully implemented and available
      through the internal control, but `replicated` stays the shipped
      default (`resolve_active_eval_reduction_mode` returns `replicated`
      whenever the `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` control is unset or
      `world_size<=1`/`pack_count<world_size`) because the win itself —
      not correctness — is exactly what the missing 8-rank measurement was
      supposed to establish. This mirrors Wave 3's `overlapped` disposition:
      implemented, proven correct, not promoted to default without
      measured evidence. World-size-1 degeneracy: guaranteed structurally
      (`resolve_active_eval_reduction_mode` forces `replicated` at
      `world_size<=1`, and `ForwardEvalRunner` rejects constructing
      `disjoint_shard` at `world_size<=1`), so the pre-existing
      `test_forward_eval_streaming_counts_and_wide_scalars` /
      `test_forward_eval_returns_one_pure_wide_observation_and_restores_mode`
      (unmodified, still passing) already cover the byte-identical
      world_size=1 row. Checkpoint-selection replay on a recorded
      multi-rank run and the focused independent review both require the
      same unavailable 8-rank launch and remain future confirmation work,
      exactly like Wave 3's residual item. Gate suite: `tests/runtime/
      tests/training/ tests/losses/ tests/config/ tests/artifacts/
      tests/eval/ tests/qwen/` — 499 passed (up from 481 pre-Wave-4; net
      +18). `ruff check` clean on every touched/new file (2 pre-existing
      findings confirmed present at HEAD baseline —
      `RankGradientFiniteReport` unused import, one unused `runtime` local
      in an unrelated test — neither introduced by this wave). `git diff
      --check` clean. `openspec validate
      streamline-coordexp-swift-base-infrastructure --strict` and `--all
      --strict` both pass (19/19). No production pack-cache root or
      unrelated GPU/process was touched — no cache build or GPU launch was
      attempted this wave at all, since correctness work required neither
      and the M4 GPU window was unsafe.
      STILL PARTIAL 2026-08-04 (later same day) after an independent Opus
      review of this wave returned HOLD on two P1 findings (a real
      `finite/*` reduction bug, missing non-finite test coverage) and five
      P2 findings, all fixed the same day -- see task 4.1/4.2's addenda and
      `review-triage.md` "Wave 4" for the full findings table and
      resolutions. This checkbox remains unchecked: the fixes closed the
      HOLD's correctness/testing findings, but neither the exact 8-rank M4
      measurement nor the focused independent review this task names became
      available as a result (both require the same unavailable 8-rank
      launch this session never had). Updated gate suite: `tests/runtime/
      tests/training/ tests/losses/ tests/config/ tests/artifacts/
      tests/eval/ tests/qwen/` — 509 passed, 1 skipped (CUDA-gated
      `tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad`;
      **corrected 2026-08-04 per Wave 5 review P2** from this session's
      original "510 passed", up from 499; net +11 from the
      HOLD-fix tests). `ruff check` still clean on every touched file (the
      same 2 pre-existing baseline findings, untouched). `git diff --check`
      clean. `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict` and `--all --strict` still pass (19/19). Full receipt in
      `review-triage.md` "Wave 4" re-verification section.
      STILL PARTIAL 2026-08-04 (later session): the "focused independent
      review of the distributed reduction" this task names has now been
      performed by a separate agent lane and converged -- verdict HOLD on
      one further P1 finding (a full-row-exactness gap: a configured term
      with globally-zero selected tokens across the entire disjoint-sharded
      eval set crashed instead of matching the replicated reference's `0.0`
      convention), fixed and decisively re-tested the same session, verdict
      **APPROVE**. See `review-triage.md` "Independent implementation review
      (2026-08-04) — Wave 4 distributed reduction" for the full finding,
      fix, and re-verification (102 passed on the targeted suite at that
      time, up from 101). This checkbox remains unchecked regardless: the
      review-availability gate is now satisfied, but the exact 8-rank M4
      paired A/B wall-time measurement and a checkpoint-selection replay on
      a recorded multi-rank run are still unavailable in this session (GPU
      receipt at this session's start: all 8 A100s externally occupied,
      100% util, 2026-08-04T07:52Z) -- both remain future confirmation
      work, exactly as Wave 3's `overlapped` disposition already
      established as this change's pattern for a measurement-only residual.
      **CORRECTED same-day, pre-final Opus P2 closures pass:** an
      end-to-end test driving the real `LossRunner.prepare_planned_step`
      for the identical globally-zero-selected-term scenario (not the
      hand-summed helper functions the original fix's test used) shows the
      scenario is actually UNREACHABLE past `prepare_planned_step` --
      `_build_denominator_from_token_sequences`'s pre-existing, Wave-4-
      independent "at least one eligible segment" check fails closed first,
      symmetrically, in both reduction modes. The `weight <= 0.0` code the
      original P1-1 fix touched is real, harmless, and matches the
      reference `_weighted_average` convention, but is defensive/dead code
      in production use, not a live correctness bug -- the same disposition
      already given to `_prepare_disjoint_shard_scalars`'s own dead
      `local_value is None` branch (task 4.2's P2-D). The fix itself is
      kept unchanged (harmless, consistent with the reference convention).
      Verdict **APPROVE still stands** (no P0/P1 finding survives this
      correction; the correction narrows the finding's reachability
      characterization, it does not reopen a defect). Test count at THIS
      pass's own re-verification: 104 on the same 4-path targeted suite (up
      from 102), 107 on the 6.2 3-path suite (up from 105); see
      `review-triage.md` "Pre-final Opus P2 closures" for the full,
      current, re-run receipts.
      DONE 2026-08-04 (later session): the exact 8-rank M4 paired A/B and
      checkpoint-selection replay this checkbox was still waiting on are now
      complete. All 8 GPUs confirmed idle immediately before each of 3
      back-to-back repetitions; H-short (`pack_count=8 == world_size=8`, so
      the structural fallback does not mask the control), same HEAD
      (`a19ffceeb`), same derived config, same temporary
      `COORDEXP_SWIFT_PACK_CACHE_ROOT` cache reused by all 6 runs (3 reps x
      sharded/replicated), same fixed inputs. Eval invocation wall time
      (max-across-ranks, seconds): sharded median 1.4775 (min 1.4616, max
      1.7801) vs replicated median 10.7365 (min 10.7132, max 10.9179) — a
      7.27x median speedup, ranges non-overlapping across all 3 reps.
      Full-row equivalence: exact on every count/top-k/identity/finite field
      and within rtol 1e-5 (in fact bit-identical) on every loss/diagnostic
      scalar, across all 3 reps x 8 ranks x both modes; identical key sets,
      no `__weight__` leakage, all 8 ranks' rows identical to each other
      within each arm. `validate_eval_reduction_consensus` (pre-sharding
      cross-rank liveness check) passed with no hang/fail-closed error on
      every run. Checkpoint-selection replay: `checkpoints/best.json`
      byte-identical (`step-1`, `acc_top1`, `value: 0.3215420461766575`)
      across all 6 runs. World-size-1 degeneracy: not re-run (already
      structurally guaranteed and covered by existing unmodified tests, per
      this checkbox's earlier PARTIAL note). Focused independent review:
      already converged APPROVE in a prior session (see `review-triage.md`).
      Stop rule does not trigger — full-row exactness is demonstrated, not
      absent. Per `measurement-plan.md`'s own M4 accept criterion ("sharded
      improves eval wall time over replicated at 8 ranks with full-row
      equivalence green"), this evidence is ACCEPT. The owning
      implementation lane then promoted `disjoint_shard` to the shipped
      multi-rank default by making the unset internal control resolve to
      `auto` and updating the matching default-contract test. Explicit
      `replicated`, the `pack_count < world_size` fallback, and the
      world-size-one fallback remain unchanged.
      Full receipts, exact commands, and the per-repetition timing/row
      tables are in `implementation-notes.md` "M4" → "Real 8-rank GPU
      measurement (2026-08-04, later session)". All temporary artifacts
      (`/tmp/coordexp_m4_probe/`) deleted after recording; shared production
      cache root and all unrelated worktrees/processes untouched.

## 5. Weight Loss And Gated Micro-Optimizations

- [x] 5.1 Record the deletion proof for the non-streaming loss path at live
      HEAD (design Seam G): call-graph/consumer inventory showing no
      production owner; then delete the trainer batch branch, eval batch
      branch, `LossRunner.compute`/`_compute_token_term`, and both
      `_supports_streaming_loss` copies; add a construction-time streaming
      protocol assertion; port batch-path tests to streaming equivalents
      preserving their numeric oracles.
      DONE 2026-08-04: deletion proof at live HEAD (this worktree, post-Wave-4)
      -- production call-graph search (`grep -rn "\.compute("` /
      `_supports_streaming_loss` across `src/`) found exactly three
      `LossRunner.compute` call sites: `SupervisedTrainer.run`'s
      non-streaming batch branch (`src/training/supervised_trainer.py`, the
      `else` arm reached only when `_supports_streaming_loss(self.loss_runner)`
      is false), `ForwardEvalRunner._run_forward_only`'s equivalent batch
      arm (`src/eval/forward.py`), and `_compute_token_term`'s three
      internal call sites inside `compute` itself. The real production
      `LossRunner` (`src/losses/runner.py`) has implemented
      `prepare_planned_step`/`compute_micro_step`/`finalize_planned_step`
      since Wave 3-4, so `_supports_streaming_loss(LossRunner(...))` is
      constant-true for every production construction path
      (`LossRunner.from_config` is the only production constructor,
      `src/training/pipeline.py`); the batch branches were therefore dead
      in production and reachable only via test doubles that implemented
      `.compute` but not the streaming trio. No other production caller of
      `.compute`/`_compute_token_term` exists.
      Deleted: `LossRunner.compute` and `_compute_token_term`
      (`src/losses/runner.py`), plus their now-orphaned exclusive private
      helpers `_checked_contexts`, `_build_counts`,
      `_selected_count_by_token_type` (each had zero remaining callers once
      `compute`/`_compute_token_term` were gone -- confirmed by grep before
      deletion); the unused `reduce_segment_balanced_planned_step` import
      that only `_compute_token_term` had consumed. Trainer batch branch
      deleted from `SupervisedTrainer.run` (`src/training/supervised_trainer.py`)
      -- `run()` now unconditionally calls `_run_streaming_planned_step` every
      planned step, no `_supports_streaming_loss` runtime branch. Eval batch
      branch deleted from `ForwardEvalRunner._run_forward_only`
      (`src/eval/forward.py`) -- it now unconditionally delegates to
      `_run_streaming_forward_only`. Both module-local `_supports_streaming_loss`
      copies deleted entirely (not merely their runtime call sites); replaced
      by one construction-time assertion inlined at each owner's `__init__`:
      `SupervisedTrainer.__init__` raises `RuntimeContractError`
      (`trainer.loss_runner_requires_streaming_protocol`) unconditionally
      (not gated on `forward_input_provider` presence as the pre-existing
      partial check was) if `loss_runner` lacks
      `prepare_planned_step`/`compute_micro_step`/`finalize_planned_step`;
      `ForwardEvalRunner.__init__` raises the equivalent
      `eval_forward.loss_runner_requires_streaming_protocol` unconditionally
      (previously only checked when `reduction_mode == disjoint_shard`). The
      now-subsumed provider-specific
      `trainer.forward_input_provider_requires_streaming_loss_runner` and
      mode-specific `eval_forward.disjoint_shard_requires_streaming_loss`
      error codes were removed (both conditions are strict subsets of the
      new unconditional check, so they were unreachable once it was added).
      `LossRunnerBoundary` Protocol (`src/training/supervised_trainer.py`,
      re-exported by `src/training/__init__.py` and imported by
      `src/eval/forward.py`) updated from a stale `.compute(...)`-only
      signature to declare the actual required streaming trio.
      Test porting (numeric oracles and failure behavior preserved,
      verified by direct pytest runs after each port):
      `tests/losses/test_runner.py` -- 5 single-context `.compute((context,))`
      call sites ported to `prepare_planned_step` + `compute_micro_step`
      (byte-identical to the old batch call for a single micro-step, since
      both build the denominator from the same one context); the
      2-context `test_loss_runner_returns_weighted_metrics_and_top_level_accuracy`
      ported to the 3-call streaming protocol and merged with
      `test_loss_runner_streaming_micro_contributions_match_planned_step_compute`
      (renamed `test_loss_runner_streaming_planned_step_reproduces_weighted_metrics_and_top_level_accuracy`)
      since the latter's own oracle was `runner.compute(contexts)`, which no
      longer exists -- the merged test's oracle is instead computed
      independently via `reduce_segment_balanced_planned_step` directly
      (the same lower-level primitive `compute`/`compute_micro_step` both
      called), which is a strictly more decisive equivalence proof than
      comparing against the now-deleted convenience wrapper. The
      `no_coordinate.compute((context,))` zero-eligible-segment failure case
      ported to `no_coordinate.prepare_planned_step(...)` raising the
      identical `loss.segment_balanced_zero_eligible` code (same underlying
      `_build_denominator_from_token_sequences` check the batch path's
      `reduce_segment_balanced_planned_step` also used).
      `tests/training/test_supervised_trainer.py` -- 14 trivial
      `FakeLossRunner` -> `StreamingFakeLossRunner` swaps (event
      dispatch/scheduling/gate/device-selection tests whose assertions never
      depended on batch-vs-streaming call shape); the comprehensive
      `test_supervised_trainer_orchestrates_accumulation_and_runtime_boundaries`
      ported with its exact `log` call-order sequence rewritten for the
      streaming interleaved (move-all, prepare-once, then
      forward/context/loss/pre/backward per micro-step inside
      `runtime.accumulation_context`) shape instead of the batch shape
      (move-all, forward-all, one `loss:N` call); `loss_bundle_artifact`
      assertion changed from an exact-dict-equality oracle
      (`== {"total_loss": 1.0}`) to a key-scoped oracle
      (`["total_loss"] == 1.0`) since the streaming `finalize_planned_step`
      artifact additionally carries `metrics`/`diagnostics` keys the batch
      artifact never had -- the numeric oracle itself (1.0, from two
      micro-steps each contributing the fake 0.5) is unchanged.
      `test_non_streaming_batch_path_leaves_timing_fields_unmeasured` deleted
      outright (not ported) -- its entire premise (the batch path is
      uninstrumented) no longer has a code path to exist under; every
      remaining trainer construction path is now streaming and always
      measures timing, already covered by the pre-existing
      `test_streaming_step_records_input_build_seconds_from_forward_receipt`.
      Added `test_supervised_trainer_rejects_non_streaming_loss_runner`
      (no provider) to prove the new construction-time check fires
      unconditionally, not only in combination with a provider; updated
      `test_forward_input_provider_rejects_non_streaming_loss_runner` to
      assert the new `trainer.loss_runner_requires_streaming_protocol` code.
      `tests/eval/test_forward_eval.py` -- `NonfiniteLossRunner` re-based
      from `FakeLossRunner` (batch, now construction-rejected) onto
      `StreamingFakeLossRunner`, overriding only `finalize_planned_step` to
      return the nonfinite metrics payload (same nonfinite oracle:
      `loss/total` NaN, `diagnostic/max_logit` Inf, no
      `non_finite_fields` key, since `to_logging_row()` still leaves
      normalization to the writer); 4 more `FakeLossRunner` ->
      `StreamingFakeLossRunner` swaps (source-required, forward-raises,
      eval-raises tests) with `forward_raises`/`eval_raises` `log`
      assertions extended by the now-observable `streaming.prepare:1` entry
      that precedes the first micro-step's move/forward (the eval-raises
      case is unaffected since `model.eval()` itself raises before
      `prepare_planned_step` is ever reached); the main
      `test_forward_eval_returns_one_pure_wide_observation_and_restores_mode`
      ported with its `log` sequence rewritten for the streaming shape
      (`streaming.prepare:2` before the move/forward/context/`streaming.loss`
      pair per micro-step, `streaming.finalize:2` before the gather) --
      `result.scalars`/`to_logging_row()`/`runtime.gathered` oracles
      unchanged (verified identical: the streaming
      `finalize_planned_step`'s `{"metrics": ...}` artifact resolves through
      `_metric_scalars`/`_gather_scalars` to the exact same
      `FakeLossBundle.metrics` values the batch path returned). Construction-time
      guard test renamed/split:
      `test_forward_eval_runner_rejects_disjoint_shard_with_non_streaming_loss_runner`
      ->
      `test_forward_eval_runner_rejects_non_streaming_loss_runner_in_disjoint_shard_mode`
      (updated to the new `eval_forward.loss_runner_requires_streaming_protocol`
      code) plus a new
      `test_forward_eval_runner_rejects_non_streaming_loss_runner_in_replicated_mode`
      proving the check fires in the default reduction mode too, not only
      `disjoint_shard`.
      Verification: `tests/losses/test_runner.py` 11 passed,
      `tests/training/test_supervised_trainer.py` 31 passed,
      `tests/eval/test_forward_eval.py` 22 passed (individually and as part
      of the full gate suite below). Residue grep for `.compute(` restricted
      to `src/losses|src/training|src/eval` and their test directories: zero
      matches. Residue grep for `_supports_streaming_loss`,
      `forward_input_provider_requires_streaming_loss_runner`,
      `disjoint_shard_requires_streaming_loss`: zero matches anywhere in
      `src/`/`tests/`.
- [x] 5.2 Remove dead helpers `_loss_plan_artifact`, `_template_identity`,
      `_unwrap_checkpoint_model`; fix duplicated
      `_sync_forward_result_if_requested` calls; move
      `build_repeating_micro_step_stream` into tests next to its oracle;
      delete the empty `src/metrics` stub after a residue check. Keep the
      three `_examples_by_id` copies and the `_file_sha256`/`_sha256_json`
      duplicates as-is (local duplication over cross-domain coupling; no
      utilities layer). Residue grep for every removed symbol.
      DONE 2026-08-04: `_loss_plan_artifact`, `_template_identity`,
      `_unwrap_checkpoint_model` (`src/training/pipeline.py`) each had zero
      call sites anywhere in `src/`/`tests/` (confirmed by grep before
      deletion -- definitions only) and were deleted outright.
      `_sync_forward_result_if_requested` was called 4 times around the
      streaming loss-context construction
      (`src/training/supervised_trainer.py`, pre-existing at live HEAD,
      predating this change -- flagged by task 0.1): two adjacent identical
      calls immediately before `loss_context_factory(...)` and two more
      immediately after. Reduced to the two distinct intended sync points
      (one before, one after context construction), each now called exactly
      once; the unrelated `_sync_loss_bundle_if_requested` calls at three
      distinct points (post-compute, pre-backward, post-backward) were left
      untouched -- they are not adjacent duplicates, each brackets a
      different profiling stage. `build_repeating_micro_step_stream` (zero
      production callers -- confirmed by grep; only consumed by
      `tests/training/test_pack_cache.py`'s rank-selective-loading
      equivalence oracle at tasks 2.2/2.4) moved verbatim from
      `src/training/pipeline.py` into `tests/training/test_pack_cache.py`
      (placed beside `write_micro_step_cache`, its neighboring test-owned
      helper, immediately before its two call sites); the import updated
      from `from src.training.pipeline import build_repeating_micro_step_stream`
      to a local definition, with `Iterator`/`Sequence` added to the test
      file's imports and removed from `pipeline.py`'s (now-unused there).
      `src/metrics/` residue check: grep for `src.metrics`/`from src import
      metrics`/`import metrics` across `src/` and every `pytest.ini`
      `testpaths` directory found zero references; the only referencing
      files are top-level legacy `tests/test_*.py` modules (e.g.
      `tests/test_metric_events.py` importing `src.metrics.events`, a
      submodule that was never present under the real `src/metrics/`
      directory even before deletion) that are excluded from `pytest.ini`'s
      `testpaths` and do not collect under this worktree's test
      configuration -- confirmed already-orphaned, not a residue this
      deletion could introduce. Deleted `src/metrics/` (a single 30-byte
      marker `__init__.py`, tracked at `e1662c2c7`, no submodules).
      Kept as-is per explicit instruction: the three `_examples_by_id`
      copies (`src/qwen/forward.py`, `src/qwen/positions.py`,
      `src/packing/supervision.py`) and the `_file_sha256`/`_sha256_json`
      duplicates (present across `src/qwen/`, `src/adapters/`,
      `src/training/`, `src/inference/`) -- confirmed still present
      unchanged. Residue grep for every removed symbol
      (`_loss_plan_artifact`, `_template_identity`,
      `_unwrap_checkpoint_model`, `build_repeating_micro_step_stream` in
      `src/`, `src.metrics`) across `src/`/`tests/`: zero matches beyond the
      moved test-local definition and the unrelated `chat_template_identity`
      dict-key substring false positive in two legacy out-of-scope test
      files. `ruff check` clean (zero findings) on both touched files
      (`src/training/pipeline.py`, `tests/training/test_pack_cache.py`).
      Combined 5.1+5.2 gate: `tests/runtime/ tests/training/ tests/losses/
      tests/config/ tests/artifacts/ tests/eval/ tests/qwen/` -- 509 passed,
      1 skipped (CUDA-gated `tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad`;
      **corrected 2026-08-04 per Wave 5 review P2** from this session's
      original "510 passed") (unchanged count from the pre-5.1/5.2 baseline:
      -1 losses test merged away, -1 trainer test deleted, +1 trainer test
      added, +1 eval test added, net 0). Full repo `pytest.ini` `testpaths`
      suite: 1121 passed, 1 skipped (same CUDA-gated test; **corrected
      2026-08-04** from "1122 passed").
      `ruff check` across every file touched by 5.1+5.2
      (`src/losses/runner.py`, `src/training/supervised_trainer.py`,
      `src/eval/forward.py`, `src/training/pipeline.py`,
      `tests/losses/test_runner.py`, `tests/training/test_supervised_trainer.py`,
      `tests/eval/test_forward_eval.py`, `tests/training/test_pack_cache.py`):
      clean except one pre-existing baseline finding in
      `src/training/supervised_trainer.py` (`collections.abc.Iterator`
      unused) confirmed present at the untouched committed HEAD baseline
      (`git show HEAD:src/training/supervised_trainer.py | grep -c Iterator`
      = 1, i.e. only the import itself, before any Wave 0-4 or 5.1/5.2
      edits) -- not introduced or worsened by this change, left untouched
      per the no-unrelated-cleanup instruction. `git diff --check` clean.
      `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict` and `--all --strict` both pass (19/19).
- [x] 5.3 Gated: hoist per-pack `examples_by_id` map; bisect token-span
      lookup with the exact boundary-equivalence corpus test (identical
      indices and identical error classification, including
      boundary-crossing failures). Accept only with equivalence green;
      measure materialization wall time (M5a) on a temporary cache root as a
      paired A/B — old linear lookup vs hoisted+bisect lookup, both present
      in the same pre-deletion commit — before the old lookup is deleted.
      DONE 2026-08-04: **both sub-candidates REJECTED with evidence; no
      production change.** Full method/results in `implementation-notes.md`
      "M5a (Wave 5, task 5.3)". Hoisted `examples_by_id`: implemented behind
      a private `_PACK_EXAMPLES_LOOKUP_STRATEGY` module flag, proven
      byte-identical via two paired tests (synthetic multi-segment pack;
      real 128-example/16-pack dataset from
      `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_ebs128_1step.yaml`
      on a temporary cache root), and shown ~9.5x faster in isolation
      (0.323ms legacy vs 0.034ms hoisted per full lookup pass) — but the
      end-to-end `prepare_training_pack_caches` paired A/B (5 reps,
      alternated order) showed only a noise-level ~1.9% mean difference
      (3.181s vs 3.120s) with overlapping distributions, because
      tokenization/image encoding dominate wall time by 3+ orders of
      magnitude at any measurement-compliant sample size. Bisect token-span
      lookup: implemented behind a private `_TOKEN_SPAN_LOOKUP_STRATEGY`
      module flag; matched the legacy linear scan exactly on the real
      single-image fixture corpus (`tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`,
      confirmed-monotonic real `base_offset_mapping`), but a constructed
      adversarial `offset_mapping` containing one non-monotonic zero-width
      special-token-like `(0,0)` entry mid-sequence produced a genuine
      divergence (linear returns `(1, 3)`; bisect returns `(3,)` and
      spuriously raises `qwen.span_token_coverage`) — equivalence is not
      green, so per this task's own rule the candidate is rejected. Both
      candidates were reverted in full: `src/training/pipeline.py` and
      `tests/training/test_pipeline_assembly.py` show zero net change from
      this task (residue grep for `_PACK_EXAMPLES_LOOKUP_STRATEGY`,
      `_encoded_examples_for_pack_from_map`, `_TOKEN_SPAN_LOOKUP_STRATEGY`,
      `_token_indices_for_char_range_bisect` across `src/`/`tests/`: zero
      matches). No production cache was ever opened; all measurement used
      dedicated `tempfile.mkdtemp()` roots.
      **Wave 5 review P2 addendum (2026-08-04)**: the rejected-bisect
      lesson is now preserved durably rather than only in this note and
      `implementation-notes.md` — `src/qwen/encoding.py`'s
      `_token_indices_for_char_range` carries a short comment stating the
      non-monotonicity hazard, and a permanent regression test
      (`tests/qwen/test_encoding.py::test_token_span_lookup_handles_non_monotonic_zero_width_offset_without_bisect`)
      asserts the exact `[(0,5),(5,10),(0,0),(10,15)]`/`[5,15)` counterexample
      resolves to `(1, 3)` with no spurious `qwen.span_token_coverage` error.
      No bisect implementation or private control was reintroduced; only the
      comment and the one new test were added.
- [x] 5.4 Gated: batch gradient finite-gate host syncs per design Seam F;
      extend `tests/runtime/test_finite_gates.py` with mixed-dtype
      NaN/Inf/overflow/missing-grad cases asserting identical decisions,
      reason codes, and `grad_norm` diagnostics. Accept only with
      equivalence green plus measured planned-step overhead reduction (M5b)
      as a paired A/B — old per-micro-step syncs vs batched syncs, both
      present in the same pre-deletion commit — before the old path is
      deleted; otherwise revert.
      DONE 2026-08-04: **REJECTED / deferred pending CUDA evidence; no
      production change.** Full method/results in `implementation-notes.md`
      "M5b (Wave 5, task 5.4)". Implemented
      `_build_gradient_finite_report_batched` (per-(device, dtype) group,
      `torch._foreach_norm`, one host sync per distinct device instead of
      two per parameter, explicit elementwise finiteness kept independent
      of the norm per this task's own instruction) behind no production
      wiring change at all (test/benchmark-only). Extended
      `tests/runtime/test_finite_gates.py` with a 25-case paired corpus
      (mixed dtype fp32/bf16/fp16/fp64, NaN confined to one dtype group,
      mixed +Inf/-Inf, grad-norm overflow from finite fp32 values matching
      the pre-existing `test_build_gradient_report_non_finite_norm_triggers_global_skip`
      oracle, missing grad, all-grads-missing, empty params,
      backend-overflow passthrough, a 4-dtype-group case with a NaN
      injected only in the last element of the last group to defeat
      per-group short-circuiting, and sparse/compressed-sparse gradient
      rejection with identical error code+context) — all 25 green,
      `GateDecision` fields (reason codes, `all_ranks_safe`,
      `should_call_optimizer_step`, diagnostics) compared exact, `grad_norm`
      compared with `pytest.approx(rel=1e-9)` per this change's own
      fp-summation-order invariant. `nvidia-smi` checked twice ~40 minutes
      apart: all 8 GPUs carried substantial external memory/utilization at
      both checks, so no CUDA tensor was allocated for this task
      (`CUDA_VISIBLE_DEVICES=""` throughout) and M5b's overhead measurement
      is CPU-only by necessity — 589 synthetic DoRA/LoRA-shaped parameters
      (matching `adapter.target_modules: all_linear`, `rank: 8`, hidden
      3584) showed only a ~5-7% CPU-side improvement (two runs, 30
      alternated reps each: 1.068x and 1.050x), with per-run standard
      deviation (~8% of the mean) comparable to the effect size — expected,
      since the candidate's actual target (CUDA host-device sync stalls)
      does not exist on CPU, and CPU is not the decision-relevant device
      per this task's own instruction. Fully reverted:
      `src/runtime/finite_gates.py` and `tests/runtime/test_finite_gates.py`
      are byte-identical to their pre-5.4 state (`git status --porcelain`
      empty for both; residue grep for
      `_build_gradient_finite_report_batched` across `src/`/`tests/`: zero
      matches). `build_gradient_finite_report` remains the sole production
      path, unchanged.
- [x] 5.5 Wave gate: full targeted suites for losses, trainer, runtime,
      eval, pack cache; shared gate commands.
      DONE 2026-08-04: since both 5.3 and 5.4 candidates were rejected and
      fully reverted (see above), this wave's net source diff vs. the
      pre-existing Wave 5.1/5.2 baseline is zero — the gate below validates
      that baseline is undisturbed. Targeted suite
      (`tests/losses/ tests/training/ tests/runtime/ tests/config/
      tests/artifacts/ tests/eval/ tests/qwen/ tests/packing/`): 522 passed,
      1 skipped (CUDA-gated
      `tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad`
      -- collected but not exercised without a visible GPU, per this
      session's no-GPU policy; **corrected 2026-08-04 per Wave 5 review P2**
      from this session's original "523 passed", which silently let that one
      test run against a real GPU rather than gating it). Full repo
      `pytest.ini` `testpaths` suite: 1121 passed, 1 skipped (same CUDA-gated
      test; **corrected 2026-08-04** from "1122 passed"; unchanged count
      from the 5.1/5.2 baseline recorded in 5.2's own DONE note once that
      note's own count is read under the same CUDA-hidden correction above).
      `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict` and `--all --strict`: both pass (19/19). `git diff --check`:
      clean. Residue greps for every symbol either sub-candidate introduced
      (`_PACK_EXAMPLES_LOOKUP_STRATEGY`, `_encoded_examples_for_pack_from_map`,
      `_TOKEN_SPAN_LOOKUP_STRATEGY`, `_token_indices_for_char_range_bisect`,
      `_build_gradient_finite_report_batched`): zero matches anywhere in
      `src/`/`tests/`. Targeted `ruff check` on every file touched this
      session (`src/qwen/encoding.py`, `src/training/pipeline.py`,
      `tests/training/test_pipeline_assembly.py`, `src/runtime/finite_gates.py`,
      `tests/runtime/test_finite_gates.py`): clean. No production cache was
      opened; no GPU was used (GPUs were saturated by external jobs for the
      duration of this session per the M5b GPU-availability receipt above).
      No commit made.
      **Wave 5 review P2 fixes addendum (2026-08-04)**: the "net source diff
      is zero" statement above is superseded for `src/qwen/encoding.py` and
      `tests/qwen/test_encoding.py` only — a durable comment plus one
      permanent regression test were added per the 5.3 addendum above (no
      candidate/private-control code reintroduced). Also fixed: the pass
      counts throughout this wave's receipts (this task, 5.2, and
      `review-triage.md` Wave 3/4) were corrected from 523/510/1122 to
      522/509/1121 (each +1 skipped, CUDA-gated) — those higher counts had
      silently included a GPU-executed test rather than gating it, contrary
      to this session's own no-GPU policy; `review-triage.md` P2-G (Wave 3
      table) was annotated noting `trainer.forward_input_provider_requires_streaming_loss_runner`
      was superseded/deleted by Wave 5 task 5.1's unconditional
      `trainer.loss_runner_requires_streaming_protocol` check; and a new
      `## ADDED Requirements` entry ("Loss Runner Streaming Protocol Is
      Required At Construction") was added to
      `specs/coordexp-swift-supervision-losses/spec.md` making the
      construction-time streaming-trio gate (trainer and eval, fail-closed,
      batch path unsupported) a normative delta requirement. Re-verified
      after these fixes: targeted suite (same 8 paths above) 523 passed, 1
      skipped (the correctly-CUDA-gated 522 above, +1 for the new permanent
      regression test); full repo suite 1122 passed, 1 skipped (same +1);
      `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict` and `--all --strict` both pass (19/19); `git diff --check`
      clean; `ruff check` on `src/qwen/encoding.py` and
      `tests/qwen/test_encoding.py` clean. No production cache, GPU,
      commit, stage, or push.

## 6. Final Acceptance Gate

- [x] 6.1a USER-AUTHORIZED: single production cache materialization for the
      final settled fingerprint via `python -m src.prepare_train_cache`,
      scheduled in a window the user approves (launch/material cost). The
      existing cache directories are left in place; their cleanup is a
      separate user-owned decision.
      DONE 2026-08-04: `CUDA_VISIBLE_DEVICES="" python -m src.prepare_train_cache
      --config configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml`
      run against the default production cache root
      (`.cache/coordexp_swift/packing`), no model load
      (`"model_loaded": false`), no GPU. Resolved config fingerprint
      `54122efaf7a0fe69838ccaba90ade06e865da66c43d6e42fc276a78cd1d1f0ef`.
      Both caches were misses and freshly built (`"build_status": "built"`):
      train fingerprint `e68f2699d83705bd2a82a938164ede5e5d800d291c4b9bd64c3ece85db97c668`
      (14660 micro-steps, 29 chunks, 11 GB,
      `.cache/coordexp_swift/packing/e68f2699d83705bd2a82a938164ede5e5d800d291c4b9bd64c3ece85db97c668`),
      eval fingerprint `e9aa216516e72dfdd593afd8177e58d02b2c9f7f8aff64db267138a774324549`
      (619 micro-steps, 2 chunks, 461 MB,
      `.cache/coordexp_swift/packing/e9aa216516e72dfdd593afd8177e58d02b2c9f7f8aff64db267138a774324549`).
      Wall time 09:59:01-10:31:14 UTC (1933.6 s, ~32.2 min); two earlier launch
      attempts in the same session died with zero cache-root residue (no new
      directory, no new lock file) before this run and are not additional
      publications. Pre-run cache root held exactly 35 pre-existing fingerprint
      directories (50 GB) plus 6 lock files; post-run diff confirms all 35
      untouched (mtimes unchanged) and exactly 2 new directories added (37
      total, 61 GB) — no prior cache deleted, repaired, or mutated. Both new
      caches independently pass full payload-level validation via the live
      admission API (`load_cache_manifest(..., level="payloads")` and
      `cache_is_complete(...)`): `status="complete"`,
      `version="coordexp-swift-pack-cache-v2"`, `cache_is_complete=True`.
      `git status`/`git diff --check` after the run are identical to the
      pre-run dirty-worktree state (no source/config touched by the
      materialization). No commit/stage/push.
- [x] 6.1b End-to-end production-shaped smoke per `measurement-plan.md` M6:
      train + scheduled eval + checkpoint on multi-rank against the new
      production cache; record absolute wall-clock timings (M0 is
      unavailable by user decision 2026-08-04 and MUST NOT be used as a
      comparison point); publish those absolute timings alongside the
      paired A/B accept/reject status already recorded per slice in Waves
      2-5 in the change notes.
      DONE 2026-08-04: both required parts run on real 8-rank hardware,
      co-located with an unrelated external 8-rank job on all 8 GPUs
      (including GPU5) per explicit user authorization; no OOM, no
      unrelated-process interference. **H-short** (exact tracked config,
      dedicated disposable `/tmp` cache root): full train+eval+checkpoint
      cycle, `completed_steps: 2`, full process wall time 83.660 s
      (pre-loop init 17.149 s + writer span 53.878 s + teardown 12.633 s),
      approx. time-to-first-completed-step 35.694 s; eval row (step 1)
      bit-identical to the M4 receipt's own recorded row, confirmed on the
      now-default `disjoint_shard` path (`pack_count=8 >= world_size=8`,
      `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` unset ⇒ `auto`); checkpoint
      `step-2`/`final.json` valid canonical selector (no `best.json` since
      this exact tracked config schedules eval and checkpoint at different
      steps -- expected, not a defect). **H-steady** (genuine minimal
      `extends` derived config outside the tracked tree,
      `training.max_steps: 5`, `eval.forward.every_fraction`/
      `checkpoint.every_fraction` overridden to the schema max `1.0` since
      the loader's `config.null_inherited_delete` fail-closed guard blocks
      nulling an inherited non-null cadence -- proven via
      `resolve_planned_step_schedule` to leave steps 1-4 event-free):
      cache-hit proven before model load and confirmed again in the
      completed run's own `materializations` (both fingerprints identical
      to 6.1a's production caches, zero rebuild); 5/5 steady
      `step_duration_seconds` samples 17.7455-18.5796 s, median 17.899 s
      (201.13 steps/hour); full-scale eval at step 5 (4952 examples/619
      packs, the complete non-sample-limited production eval set) on the
      same now-default `disjoint_shard` path; checkpoint `step-5`/
      `best.json` (`acc_top1` selector, value exactly matches the eval
      row) /`final.json`. Production cache root confirmed byte-identical
      (37 directories, mtime unchanged) before and after both runs. Full
      receipts, exact commands, the verbatim derived config, and per-step
      tables: `implementation-notes.md` ("6.1b -- M6 end-to-end
      production-shaped smoke"). All `/tmp/coordexp_m6_probe/` scratch
      deleted after evidence was durably recorded; no production cache,
      unrelated worktree, or unrelated GPU process touched.
- [x] 6.2 Verify artifact compatibility: `logging.jsonl` diff shows
      additive-only train-row fields; eval rows, run record (plus the new
      compact provider-mode field), checkpoint payloads, and cache manifest
      format byte-compatible in schema.
      DONE 2026-08-04: proof is a byte-level diff of every touched producer
      against live HEAD (`git diff HEAD -- <file>`), not a black-box replay,
      cross-checked by the existing/targeted test suite (CUDA-hidden,
      `CUDA_VISIBLE_DEVICES=""`).
      **Train row (`logging.jsonl`) additive-only:** `RunWriter.append_logging_row`
      (`src/artifacts/run_writer.py`) has zero diff this change (confirmed by
      hunk-location: the two hunks in this file only touch the `run.json`
      initial-state dict and add the new `bind_forward_input_provider_mode`
      method, both below `append_logging_row`'s line range) — the writer
      itself cannot have changed row shape. The only row-content change is in
      `_train_logging_handler` (`src/training/pipeline.py`): a new
      `timing_fields` dict (`step_duration_seconds`, `input_build_seconds`,
      `input_wait_seconds`) is merged into `scalar_metrics` only when a value
      is not `None`, before the existing `gather_metrics`/`append_logging_row`
      call — no existing key is renamed, removed, or retyped, and an
      unmeasured (non-streaming, production-dead) path omits the fields
      entirely rather than writing a fabricated zero (task 1.3). The
      `TrainRuntime.gather_metrics` `"reduction"` receipt field (renamed
      `all_rank_mean` -> `all_rank_mixed` by the Wave 1 P2-D fix) is a
      returned-dict label, not a row key: `_train_logging_handler` only
      spreads `gathered["metrics"]` into the row (`row: dict[str, Any] = {...,
      **dict(reduced), ...}`, `src/training/pipeline.py`), so that internal
      rename never reaches the durable schema. Verified by the existing
      `tests/training/test_pipeline_assembly.py::test_train_row_carries_timing_fields_additively`,
      `::test_train_row_normalizes_non_finite_timing_fields`,
      `::test_train_logging_forwards_real_loss_runner_accuracy_stats_without_leaking_to_row`
      (accuracy_stats plumbing is provably reduction-internal, never a row
      key) plus `tests/artifacts/test_run_artifacts.py::test_logging_appends_one_self_contained_train_and_eval_row`.
      **Eval row schema unchanged despite internal sharded payloads:**
      `ForwardEvalObservation` and `to_logging_row()` (`src/eval/forward.py:173-196`)
      sit entirely between two diff hunks (old-line 38->new-line 157 and
      old-line 79->new-line 207) and are confirmed byte-identical to HEAD by
      direct read. The new `example_count`/`pack_count`/`__weight__`
      sharding-reduction plumbing lives only in
      `_prepare_disjoint_shard_scalars`/`_finalize_disjoint_shard_scalars`
      (module-level helpers, not the dataclass) and is popped back out before
      `ForwardEvalObservation` is constructed. Verified by
      `tests/eval/test_forward_eval.py::test_disjoint_shard_eval_reproduces_full_replicated_row_with_unequal_rank_atom_counts`
      (asserts `set(sharded_row) == set(replicated_row)` and no row key ends
      with `__weight__`, task 4.3) plus the full `tests/eval/` suite.
      **Run record gains only the compact `forward_input_provider_mode`
      field:** `src/artifacts/run_writer.py`'s diff is exactly two hunks — one
      adds `"forward_input_provider_mode": None` to the initial `run.json`
      state dict (alongside the pre-existing fixed keys, none renamed/removed),
      the other adds the new bind-once `bind_forward_input_provider_mode`
      method. No other run-record field, and no schedule/materialization
      binding code, is touched. Verified by
      `tests/artifacts/test_run_artifacts.py::test_forward_input_provider_mode_can_be_bound_once_per_mode`,
      `::test_forward_input_provider_mode_rejects_unknown_values`,
      `::test_initialize_writes_only_fixed_run_files_with_no_selector_state`
      (unmodified, still passing — proves no other file/key was added to a
      fresh run tree), `::test_materialization_binding_is_one_time_and_survives_cache_deletion`,
      `::test_schedule_can_be_bound_once_after_early_initialization` (both
      unmodified, still passing — proves the pre-existing bind-once fields are
      untouched by the new one).
      **Checkpoint payload/schema unchanged:** `src/artifacts/checkpoints.py`
      carries zero diff this entire change (absent from `git status --short`)
      — the strongest possible compatibility proof, since the file was never
      opened for edit.
      **Pack cache manifest format unchanged despite fingerprint-determinant
      and rank-selective-load changes:** `PACKING_CACHE_VERSION` constant
      (`"coordexp-swift-pack-cache-v2"`) is untouched, and the manifest's
      top-level key set written by `_publish_micro_step_cache`
      (`version`/`status`/`fingerprint`/`determinants`/`micro_step_count`/
      `chunk_size`/`chunks`/`materialization`/`augmentation`,
      `src/training/pack_cache.py`) has zero diff — the only change inside
      that dict is the removal of the `"mtime_ns"` key from the nested
      `determinants["dataset"]` sub-dict (task 2.1's intended identity fix, a
      determinant *value* change, not a manifest *layout* change), and the
      new `_iter_required_chunks`/`_load_validated_chunk` chunk-skip logic is
      a read-path optimization that never touches manifest-write code at all.
      Every chunk that IS read still goes through the unchanged
      `_load_validated_chunk` digest-and-payload validation. Verified by
      `tests/training/test_pack_cache.py::test_packing_cache_fingerprint_is_timestamp_independent_and_tracks_content`
      (task 2.1) and
      `::test_rank_selective_loading_matches_full_pass_and_repeating_stream_oracle`
      (task 2.2, identical output across the chunk-skip and forced-full-pass
      arms) plus the full targeted suite.
      **Test evidence (CUDA-hidden, `CUDA_VISIBLE_DEVICES=""`, `ms` conda
      env):** `tests/artifacts/test_run_artifacts.py tests/eval/test_forward_eval.py
      tests/training/test_pack_cache.py` — 105 passed, 0 failed at the time
      this note was written. **Corrected 2026-08-04 (pre-final Opus P2
      closures pass): re-verified 107 passed, 0 failed** — the figure was
      stale, not wrong: +1 from the same-day 4.4 independent-review fix
      (`test_disjoint_shard_globally_zero_selected_term_matches_replicated_zero_weight_convention`)
      and +1 from this later pass's own new decisive end-to-end test (see
      `review-triage.md`'s "Pre-final Opus P2 closures" section for the
      re-run command and full lineage). No production
      cache root, GPU, commit, stage, or push used; no temp CPU-only artifact
      harness was needed beyond the existing fixtures since the diff-location
      proof is strictly more decisive (byte-identical-by-construction, not
      merely observed-identical-on-one-run).
- [x] 6.3 Docs touch-ups where behavior is operator-visible
      (`docs/COORDEXP_SWIFT.md` cache/eval notes, timing fields) —
      implementation-time edits, not part of this planning change.
      DONE 2026-08-04: minimal edit to `docs/COORDEXP_SWIFT.md`'s existing
      "Semantic boundaries that docs must preserve" section only — no new
      section, no internal implementation detail (no module/function names,
      no error codes, no env var names). Extended the existing pack-cache
      bullet with mtime-insensitive content-only identity and rank-selective
      chunk-granular loading (digest/manifest validation unaffected).  Added
      one bullet for the three additive completed-step timing fields and the
      synchronous-shipped-default/debug-only-overlap forward-input-provider
      disposition. Added one bullet for the replicated-shipped-default/
      debug-only-sharded eval reduction disposition and its still-pending
      8-rank evidence. Closed with an explicit "no new public knob" statement
      covering both debug-only switches. `updated:` frontmatter bumped to
      2026-08-04. No other doc file touched.
      **Superseded after M4 ACCEPT later on 2026-08-04:** the same operator
      bullet was revised after the exact 8-rank wall-clock and checkpoint-
      selector replay passed. It now correctly records `disjoint_shard` as
      the shipped multi-rank default with replicated structural fallbacks;
      the earlier sentence above is a historical pre-M4 receipt, not the
      current contract.
- [x] 6.4 `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict`, delta inspection against the then-current stable specs, and
      markdown/diff hygiene.
      DONE 2026-08-04: `openspec validate streamline-coordexp-swift-base-infrastructure
      --strict` and `openspec validate --all --strict` both pass (19/19),
      both before and after this task's own edits. **Delta inspection**
      found and fixed one real gap: `coordexp-swift-training-artifacts`'s
      delta spec had `## MODIFIED Requirements` for "Eval Forward Artifacts"
      and "Wide-Step Logging Stream" (both correctly superset the stable
      scenario sets, verified scenario-by-scenario) but no entry at all for
      "Rank-Zero Run Record", despite task 3.1 adding a real, tested,
      bind-once-immutable `forward_input_provider_mode` field to `run.json`
      -- the same class of "normative behavior change with no delta
      requirement" gap Wave 5's own P2-3 finding already fixed once for the
      streaming-protocol construction-time gate. Added a `## MODIFIED
      Requirements` entry for "Rank-Zero Run Record" (prose sentence plus
      one new "Forward-input-provider mode recorded once" scenario, stable
      scenarios otherwise preserved verbatim). Every other MODIFIED delta
      requirement in this change (`coordexp-swift-config-runtime`'s "Packed
      Qwen Runtime Controls", `coordexp-swift-pack-cache-semantic-identity`'s
      "Cache Payload Is Current-Version-Only", `coordexp-swift-packing-forward`'s
      "Deterministic Packing Cache Reuse", `coordexp-swift-supervision-losses`'s
      "Loss Bundle Metrics") verified scenario-by-scenario superset against
      the current stable spec text -- no further gap found. **Markdown/diff
      hygiene:** `git diff --check` clean; every `.md` file touched or added
      this session has balanced code fences and no merge-conflict residue
      (checked directly, since untracked new files are outside `git diff
      --check`'s tracked-diff scope); no trailing whitespace/tabs in any
      touched doc. **Residue grep:** `eval_forward.token_weighted_diag_zero_weight`
      (the raise this session's 4.4 fix removed) is absent from `src/`
      entirely, present only in this session's own historical
      finding/test-docstring text. **Archive-time bookkeeping audit**
      (correcting the Opus-authored Wave 5 P2-1 note): see the new
      "Archive-time bookkeeping audit" section at the end of
      `review-triage.md` -- P2-1's own resolution overclaimed that it
      corrected the Wave 3 re-verification section's test count (it did
      not; only Wave 4's carries the correction annotation); Wave 4's
      original 499 figure is shown by decisive arithmetic
      (`499 + 11 - 1 = 509`, matching the recorded re-verification exactly)
      to have almost certainly counted the CUDA-gated bf16 test as a real
      pass; Wave 1/2's 453 and Wave 3's 481/475 figures have no equivalent
      arithmetic cross-check available and are recorded as an open,
      annotated uncertainty rather than retroactively invented numbers, per
      explicit instruction. This audit changes no checkbox state -- it is a
      receipt-accuracy annotation only, exactly like the P2-1 precedent it
      corrects.
- [x] 6.5 Final independent convergence review recorded in
      `review-triage.md`; the change is complete only with no unresolved
      P0/P1 findings and every benchmark-dependent slice carrying either a
      measured win or a recorded rejected-with-evidence reversal.
      DONE 2026-08-04: an independent read-only Claude Opus 5 review audited
      the full live diff, OpenSpec artifacts, production cache identities,
      exact reduction paths, M4/M6 receipts, defaults, deletions, artifact
      compatibility, and benchmark dispositions. Verdict **APPROVE** with no
      P0/P1. Four cheap documentation P2s were accepted and resolved: the
      post-M4 docs receipt is annotated, the pre-M4 review paragraph is marked
      superseded, the proposal's config claim is corrected, and M3's retained
      two-phase wiring is recorded as a deliberate stop-rule deviation. The
      optional pipeline-level multi-rank default-routing coverage was also
      added to the existing assembly test. Final verification after that
      assertion: full CUDA-hidden suite 1125 passed / 1 skipped; targeted
      pipeline/eval suites 42 passed;
      OpenSpec strict validation 19/19 and `git diff --check` clean. Every
      benchmark-dependent slice is closed: M2/M4 accepted, M3 rejected with
      synchronous default retained, M5a/M5b rejected and reverted, M6 absolute
      timings recorded without comparison to unavailable M0.
