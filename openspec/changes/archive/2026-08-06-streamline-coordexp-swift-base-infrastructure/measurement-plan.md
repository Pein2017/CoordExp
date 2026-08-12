# Measurement Plan: streamline-coordexp-swift-base-infrastructure

Purpose: define the representative measurements that gate benchmark-dependent
slices. No target numbers are requirements; the requirement is a measured
improvement (or explicit neutrality where stated) on these harnesses, else the
slice is reverted (stop rule in `tasks.md`).

## Methodology (revised by user decision, 2026-08-04)

M0's historical production-cache baseline is unavailable (see M0 below and
`implementation-notes.md`); the user decided not to build the missing caches.
Decision-bearing performance evidence for every benchmark-dependent slice
therefore comes from **same-code/same-config/same-temporary-cache paired A/B**
measurements, not from comparing across commits, HEADs, or against M0:

- Both arms of a paired A/B run at the identical worktree HEAD, the identical
  resolved config, and the identical temporary pack cache (built once, reused
  by both arms) — the only thing that differs is which mechanism executes,
  selected by an **internal, debug/test-only control** (a private
  constructor argument, monkeypatch seam, or environment variable consumed
  only by the benchmark/test harness). This isolates the mechanism under test
  from confounds (code drift, config drift, cache-cold/warm drift) that a
  cross-commit or cross-HEAD comparison cannot rule out.
- These controls are internal implementation seams, not public YAML
  compatibility surfaces: no production or smoke config ever gains a
  `use_old_path`-style key, and no control is documented as a supported
  runtime feature. Each control's exact form (kwarg, env var, or monkeypatch
  target) is recorded verbatim in the measurement notes at the wave that
  introduces it, alongside the paired-run results.
- Where the old mechanism is slated for deletion (Wave 5's finite-gate and
  encoding micro-optimizations), the paired A/B runs **before** the old path
  is deleted: both implementations coexist in the same commit for the
  duration of the local benchmark, the paired measurement is taken, and only
  then does the deletion commit land (tasks 5.1/5.3/5.4 gate order).
- Paired A/B slices: M2 (rank-selective vs full-chunk-pass loading), M3
  (overlapped vs synchronous provider — already an internal debug switch),
  M4 (sharded vs replicated eval reduction), M5a (encode micro-optimizations),
  M5b (finite-gate batching).
- M0 is a read-only historical-baseline availability receipt only (see
  below) — it produces no throughput numbers and is never used as a
  comparison point. M6 is an absolute, clearly-labeled final end-to-end
  smoke measurement, not a comparison against M0.

## Harnesses (exact existing configs)

- H-short (8 ranks, EBS 24, 12k packs, 2 planned steps + scheduled eval):
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml`
- H-steady base (production 8-epoch shape):
  `configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml`
  There is no CLI step override in `src/train.py`, and no existing config
  carries a bounded `max_steps` for this base. The bounded window is produced
  at implementation time as a temporary derived config: a small YAML that
  `extends` the H-steady base by absolute path and sets `training.max_steps`
  (enough steps for a stable steps/hour estimate after warmup), a distinct
  `run.name`, and — for A/B slices — a `data.train.sample_limit`. Temporary
  derived configs live outside the tracked config tree (scratch/probe
  location), are recorded verbatim in the measurement notes, and are not
  committed as production configs.
- H-single (world-size-1 degeneracy):
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_single_gpu_full_eval_stream.yaml`
- Profiling overlay: `COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS=1` plus the new
  timing row fields once Wave 1 lands.

All measurements run on the same host, same visible GPUs, page cache state
noted (cold vs warm), three repetitions, median reported with min/max. For
paired A/B slices, both arms of a given repetition run back-to-back against
the same warm/cold cache state to avoid state drift between arms.

## Cache-root discipline (applies to every measurement)

- M0's read-only preflight reads the existing production cache root; nothing
  in this plan writes, deletes, or mutates the existing 50 GB cache root or
  any cache directory in it, automatically or otherwise, before
  user-authorized task 6.1a. Existing directories remain
  historical/disposable; cleanup is a separate user-owned decision.
- M0 preflight (fail-closed, before any M0 claim of availability): derive
  the exact resolved-config cache fingerprint for each harness and verify
  the corresponding cache directory in the production root exists, is
  complete/current-version, and admits through validation-only checks with
  no write. If absent, incomplete, or fingerprint-mismatched, M0 reports
  "baseline unavailable — production cache missing for resolved config" and
  MUST NOT trigger materialization into the shared production root. This
  preflight has already run (2026-08-03): all four required caches (H-short
  train/eval, H-steady train/eval) are absent; the user accepted this
  disposition on 2026-08-04 and decided not to build the missing caches (see
  `implementation-notes.md`). The shared root remains read-only until
  user-authorized task 6.1a.
- Every per-slice paired A/B measurement (M2, M3, M4, M5a, M5b) builds
  exactly one temporary pack cache per harness/config combination under a
  dedicated temporary `COORDEXP_SWIFT_PACK_CACHE_ROOT`, using sample-limited
  derived configs, and reuses that same temporary cache for both arms of the
  pair — so intermediate `code_identity`/determinant churn only rebuilds
  small temporary caches, and neither arm's timing is confounded by a
  separate cache build. Runtime toggles (the provider debug switch,
  `fa2_branch_proof` values, and the new internal A/B controls above) are
  used where they exist instead of config forks.
- Exactly one production-scale materialization for the final settled
  fingerprint happens at task 6.1a, after all cache-determinant source files
  (Waves 2 and 5) are final, in a user-authorized window. Waves 2 and 5 are
  independently implementable and measurable on temporary roots precisely
  because the production rebuild is deferred to that single final step.

## Measurements

- M0 (historical-baseline availability receipt, Wave 0 — CLOSED,
  baseline-unavailable accepted 2026-08-04): the fail-closed read-only
  preflight above, run once against unmodified HEAD `a19ffceeb` with the
  existing production cache root. Outcome: baseline unavailable for both
  harnesses (all 4 required fingerprints absent); GPU 0 was also not idle at
  check time. Per user decision, the missing caches are not built and no
  throughput numbers are claimed for M0. This item produces no
  time-to-first-planned-step, steps/hour, eval-wall-time, or
  `COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS` figures, and none may be reported as
  if it did. Full receipt: `implementation-notes.md`.
- M2 (Wave 2, rank-selective loading + mtime): paired A/B on one temporary
  cache root per harness — full validated digest-and-payload chunk pass
  (pre-Wave-2 behavior, reachable post-Wave-2 only through the internal
  test/benchmark-only control) vs rank-selective chunk loading (the new
  default), same commit/config/cache: time-to-first-planned-step for
  H-short and the bounded H-steady window, cold and warm cache. Accept:
  rank-selective improves over full-pass on at least one harness and
  regresses neither. Functional probes (unpaired, correctness-only):
  `touch` on the dataset produces a cache hit (no rebuild); a one-byte
  content change produces a miss.
- M3 (Wave 3, lookahead): paired A/B via the existing debug-only synchronous
  switch, same commit/config/temporary cache — steps/hour on the bounded
  H-steady window, overlapped vs synchronous provider; `input_wait_seconds`
  distribution; loss-stream equivalence (identical `loss/*` and gate
  decisions between modes on a fixed 2-step H-short run, including a
  synthetic finite-gate early-break case). Accept: overlapped improves
  steps/hour over synchronous; revert wiring otherwise.
  Implementation disposition: the failed arm was demoted and synchronous
  restored as the shipped default, while the shared two-phase provider wiring
  was retained for lifecycle/timing consistency. This deliberate deviation is
  recorded with its future three-arm measurement in `tasks.md` and
  `implementation-notes.md` rather than being described as a literal deletion
  of all provider wiring.
- M4 (Wave 4, sharded eval): paired A/B via an internal test/benchmark-only
  reduction-mode control that forces replicated reduction at a world size
  where sharding would otherwise be active (distinct from the automatic
  `pack_count < world_size` fallback, which stays implicit/production), same
  commit/config/temporary cache — eval invocation wall time on H-short at 8
  ranks, sharded vs replicated; full-row equivalence check per the design
  Seam C inventory (exact counts and top-k, loss and diagnostics rtol 1e-5,
  single-emission global fields); H-single degeneracy (sharding inactive,
  byte-identical rows, no control needed since only one rank exists). Accept:
  sharded improves eval wall time over replicated at 8 ranks with full-row
  equivalence green.
- M5a (Wave 5, encode micro-optimizations): paired A/B run locally before the
  old lookup path is deleted — old linear `examples_by_id`/token-span lookup
  vs hoisted map + bisect lookup, both present in the same pre-deletion
  commit, same temporary cache, sample-limited dataset: cache materialization
  wall time before/after, equivalence tests green. Accept: hoisted+bisect
  improves over the old lookup; else revert (skip the deletion in task 5.2's
  neighborhood and keep the old path).
- M5b (Wave 5, finite-gate batching): paired A/B run locally before the old
  per-sync gate path is deleted — old per-micro-step host syncs vs batched
  gradient finite-gate syncs, both present in the same pre-deletion commit,
  profiled on H-short: planned-step overhead attributable to the
  post-backward gate, decisions/diagnostics equivalence green. Accept:
  batched reduces overhead vs the old per-sync path; else revert.
- M6 (final gate, after user-authorized 6.1a production materialization):
  end-to-end H-short run (train + eval + checkpoint) and the bounded
  H-steady window, reported as **absolute wall-clock timings**, clearly
  labeled as final-state measurements with no baseline to compare against
  (M0 is unavailable by user decision). The change notes publish this
  absolute table alongside the accept/reject status and paired-A/B numbers
  already recorded for each benchmark-dependent slice in Waves 2-5.

## Reporting

Each measurement records: config path (including the verbatim temporary
derived config where used), the exact internal A/B control used and its form
(kwarg/env var/monkeypatch target) for paired slices, HEAD commit, world
size, cache root and state, repetition values, and the accept/revert
decision. M0's disposition (baseline unavailable, accepted 2026-08-04) and
M6's absolute final timings are reported as distinct categories from the
paired A/B results — never merged into a fabricated before/after comparison.
Results live in the change notes (implementation-time), not in new
top-level documents.
