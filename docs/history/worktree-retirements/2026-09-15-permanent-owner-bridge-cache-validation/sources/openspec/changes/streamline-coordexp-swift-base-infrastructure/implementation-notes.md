# Implementation Notes: streamline-coordexp-swift-base-infrastructure

## User decision — 2026-08-04: accept missing M0 baseline, adopt paired A/B

The user reviewed the 2026-08-03 M0 fail-closed preflight result below (all
four required production caches absent) and decided:

- **Do not build the missing historical M0 caches.** No
  `python -m src.prepare_train_cache` run, no shared production cache write,
  no GPU launch against the missing baseline, now or as a silent prerequisite
  later in this change.
- **The missing M0 performance baseline is explicitly ACCEPTED as
  permanently unavailable**, not silently treated as green or skipped. Task
  0.2 is marked complete on this basis (the read-only preflight itself was
  performed and its result — baseline unavailable — is the accepted M0
  deliverable), not because throughput numbers exist. No M0 throughput
  figure (time-to-first-planned-step, steps/hour, eval wall time, or
  `COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS` shares) may ever be reported for this
  change; any such figure appearing later would be fabricated.
- Decision-bearing performance evidence for every benchmark-dependent slice
  (Waves 2-5) is redefined as **same-code/same-config/same-temporary-cache
  paired A/B**, gated through internal debug/test-only controls, never a
  public YAML surface. See `measurement-plan.md` Methodology section for the
  full rule and per-slice (M2/M3/M4/M5a/M5b) application.
- M6 (final gate, after user-authorized 6.1a) reports **absolute end-to-end
  smoke timings**, clearly labeled as final-state measurements with no
  baseline comparison — not a before/after table against M0.
- Cache discipline is unchanged: no shared production cache write before
  task 6.1a; every A/B measurement still uses a dedicated temporary
  `COORDEXP_SWIFT_PACK_CACHE_ROOT` and sample-limited derived configs.

This decision does not alter, weaken, or retroactively "pass" the fail-closed
M0 receipt recorded below — that receipt stands as originally captured.

## M0 (Wave 0, task 0.2) — 2026-08-03

**Result: BASELINE UNAVAILABLE. Stopped before any launch per the fail-closed
preflight rule in `measurement-plan.md`. No cache root write, no training
launch, no GPU work performed. Disposition ACCEPTED by user decision
2026-08-04 above — this remains an explicitly accepted gap, not a silently
passed or measured baseline.**

### Environment

- HEAD: `a19ffceeb` (same commit `design.md` Section 1 was drafted against).
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch `coordexp-swift`.
- Python env: conda `ms`.
- GPUs: `nvidia-smi` showed GPU 0 at 97% utilization / 19741 MiB used by an
  external process (pid 1828695, name not resolvable — not owned by this
  session), GPUs 1-7 idle (0 MiB / 0%). All 8 GPUs are required for
  H-short/H-steady (`accelerate8`), so occupancy was not safe for an 8-GPU
  launch independent of the cache-availability finding below. No process was
  inspected or killed.

### Fail-closed preflight (read-only, no cache writes)

Derived the exact resolved-config packing-cache fingerprint for both
harnesses' train and eval splits using `build_packing_cache_fingerprint`
directly (loading only config + processor/tokenizer identity via
`load_qwen_components(config, load_model=False)`; model weights not loaded),
then checked admission with `load_cache_manifest(..., level="payloads")`
called directly — never through `_resolve_or_build_pack_cache`/
`_resolve_or_build_train_pack_cache`/`prepare_training_pack_caches`, so no
path that could build or write a cache was invoked. Script:
`/tmp/m0_fingerprint_check.py` (scratch, outside the tracked config/source
tree; not committed).

Cache root checked: `.cache/coordexp_swift/packing` (repo-relative,
`COORDEXP_SWIFT_PACK_CACHE_ROOT` unset so this is the default shared
production root).

| Harness | Config | Split | Fingerprint | Cache dir exists | Admits (payloads, read-only) |
|---|---|---|---|---|---|
| H-short | `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml` | train | `106d9da5ecd68bc6583f20bec523710e99dd5994e52edc036c37f9fe71e54e2e` | No | No |
| H-short | (same) | eval.forward | `25edcc3f8db957332966a0b48c7ca22d694d79b088acd8124d87c70b89776b58` | No | No |
| H-steady | `configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml` | train | `e985abb4f4cc23e0c8fb51de85a78be5076d13294caa07d539a36dafee8b351b` | No | No |
| H-steady | (same) | eval.forward | `0a3d73aaecef8f563772bbff631c7062189aaaf9f11413592f633def0dff4d5f` | No | No |

`resolved_config_fingerprint` (whole-config provenance, distinct from the
packing-cache fingerprint): H-short `de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`;
H-steady `54122efaf7a0fe69838ccaba90ade06e865da66c43d6e42fc276a78cd1d1f0ef`.

None of the four required cache directories exist under the shared production
root (35 unrelated fingerprint directories present, none matching). Per
`measurement-plan.md` M0 preflight: "If absent, incomplete, or
fingerprint-mismatched, M0 stops and reports 'baseline unavailable —
production cache missing for resolved config'; it MUST NOT trigger
materialization into the shared production root." That rule applied
identically to both harnesses, so M0 is stopped for both.

Verified no write occurred: none of the four fingerprint directories were
created by the preflight check (directory listing before/after unchanged
except for the check's own read attempts, which raised
`PackingCacheInvalidError`/`FileNotFoundError` on a missing `manifest.json`
and returned without creating anything — the check never called
`write_micro_step_cache` or any resolve-or-build path).

### M0 metrics availability

| Metric | Available? | Reason |
|---|---|---|
| Packing-cache fingerprint (H-short, H-steady) | Yes | Derived above, read-only. |
| Cache validation-only admission | Yes (result: absent for all 4) | Read-only `payloads`-level check. |
| Time-to-first-planned-step (H-short) | No | Blocked: cache absent (fail-closed stop) and GPU 0 not idle. |
| Steps/hour, bounded H-steady window | No | Same block. |
| Scheduled-eval invocation wall time (H-short) | No | Same block. |
| `COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS=1` forward-receipt timing shares | No | Same block; also Wave 1 timing fields (`step_duration_seconds`, `input_build_seconds`, `input_wait_seconds`) do not exist yet at this HEAD. |

### Disposition

**SUPERSEDED 2026-08-04 by the user decision above.** Originally left
unchecked pending a user decision on whether to build the missing caches;
the user decided against building them and explicitly accepted
baseline-unavailable as final. Task 0.2 is now marked **complete** in
`tasks.md` on that basis — the read-only preflight ran to completion and its
"baseline unavailable" result is the accepted M0 deliverable, not a
placeholder for a future measured baseline. No throughput numbers were or
will be captured for M0. GPU 0 was not idle at check time (external process,
pid 1828695, 19.7 GB used, 97% util); this is recorded for provenance only
and no longer gates anything since M0 performs no GPU launch under the
revised scope.

Decision-bearing performance evidence for this change now comes exclusively
from the paired A/B measurements defined in `measurement-plan.md` (M2-M5b)
plus the absolute M6 end-to-end smoke — never from a comparison against this
unavailable M0 baseline.

## M2 (Wave 2, tasks 2.3/2.4) — 2026-08-04

**Result: ACCEPT. Rank-selective chunk loading improves time-to-first-load on
the bounded H-steady window (~22% median reduction) and is neutral (no
regression) on H-short. Chunk skipping stays the default.**

### Environment

- HEAD: `a19ffceeb` plus this session's uncommitted Wave 1/2 working-tree
  changes (the code under test).
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch `coordexp-swift`.
- Python env: conda `ms`.
- GPUs not used: the benchmarked mechanism
  (`load_rank_micro_steps_from_cache`) is pure CPU/file-IO (digest, restricted
  unpickle); no model load, no GPU launch, no `accelerate` process group.

### Cache-root discipline

All builds and reads used a dedicated temporary
`COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m2_probe/cache_root` (deleted
after the measurement). The shared production root
(`.cache/coordexp_swift/packing`, repo-relative) was never opened for write:
verified before/after — 35 fingerprint directories present both before and
after this work (matching the count recorded in the M0 receipt above), no
entry newer than the scratch directory's creation time, and the root
directory's own mtime (`2026-07-19`) predates this session, proving no write
touched it. All scratch configs/scripts lived under `/tmp/coordexp_m2_probe/`
(not committed, deleted after use).

### Internal control

`_force_full_chunk_pass: bool = False` — a private keyword-only argument on
`load_rank_micro_steps_from_cache` (`src/training/pack_cache.py`). Not a
public YAML/CLI surface. `True` reproduces the pre-Wave-2 full
digest-and-payload pass over every declared chunk (same code path,
`_iter_required_chunks(..., force_full_pass=True)` decodes unconditionally);
`False` (the new default) skips chunks whose declared `[start, end)` range
does not intersect the rank's required indices.

### Harnesses

- **H-short**: the exact existing smoke config
  (`configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml`),
  unmodified — `sample_limit: 256`, `effective_batch_size: 24`,
  `max_steps: 2`. Built train cache: 32 micro-steps.
- **H-steady-bounded**: a temporary derived config (`/tmp/coordexp_m2_probe/h_steady_bounded.yaml`,
  scratch, not committed) that `extends` the exact H-steady prod base path
  from `measurement-plan.md`
  (`configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml`)
  and overrides `run.name`, `data.train.sample_limit: 5000`,
  `data.eval.sample_limit: 32`, `training.max_steps: 3`. Built train cache:
  625 micro-steps.

Both builds used `python -m src.prepare_train_cache --config <path>` with
`load_model=False` (no model weights loaded, matching the documented
single-process pack-cache-prep entry point).

### Chunk layout (default `chunk_size=512`, not config-controlled)

| Harness | micro_step_count | chunk-00000 | chunk-00001 | Rank-0 required indices (world_size=8) |
|---|---|---|---|---|
| H-short | 32 | `[0,32)` — 1 chunk total | n/a | subset of `[0,32)` — the only chunk, so 0% of chunks are skippable by construction (chunk_size 512 > total packs) |
| H-steady-bounded | 625 | `[0,512)`, 377 MB | `[512,625)`, 86 MB | `max(index) = 2*24 + 2*8 = 64`, all inside chunk-00000 — chunk-00001 (86 MB, ~19% of payload bytes) is fully skippable |

H-short cannot demonstrate a win with the production default `chunk_size=512`
(no config surface controls it): 256 samples pack into only 32 micro-steps,
below the 512-per-chunk boundary, so there is exactly one chunk regardless of
which indices a rank requires. This is reported as an honest structural
neutral, not a failed measurement — matching design's own expectation ("the
win concentrates in smokes, short runs... accepts either a startup win or
neutrality with no regression").

### Timing (3 repetitions, paired back-to-back per repetition, same warm
cache both arms; script: `/tmp/coordexp_m2_probe/m2_benchmark.py`, scratch,
not committed)

| Harness | Arm | median (s) | min (s) | max (s) |
|---|---|---|---|---|
| H-short | full-pass (`_force_full_chunk_pass=True`) | 0.341 | 0.335 | 0.497 |
| H-short | rank-selective (default) | 0.352 | 0.333 | 0.465 |
| H-steady-bounded | full-pass (`_force_full_chunk_pass=True`) | 7.432 | 7.169 | 7.559 |
| H-steady-bounded | rank-selective (default) | 5.779 | 5.415 | 5.913 |

H-short: no measurable difference (noise-level, ±0.01s on medians — expected
given the single-chunk structural neutrality above). H-steady-bounded:
rank-selective is faster on every repetition (median 5.779s vs 7.432s, a
22.2% reduction; min-to-min 5.415s vs 7.169s, a 24.5% reduction) — consistent
with skipping the 86 MB unrequired chunk (~19% of the 463 MB total payload;
the timing win exceeds the raw byte fraction because it also skips that
chunk's SHA256 hashing and unpickling CPU cost, not just its I/O).

**Cache state disclosure**: both arms ran against a cache that was warm from
its own just-completed build (this sandboxed container has no permission to
drop the OS page cache — `/proc/sys/vm/drop_caches` is read-only — confirmed
denied). Only the warm-cache state was measured; cold-cache timings are not
claimed and are a residual gap, not fabricated. The paired, same-warm-state
design still isolates the mechanism under test (chunk skip vs. full pass)
from cache-temperature confounds within each repetition.

### Correctness (paired, same cache): rank-selective vs. full-pass output

`load_rank_micro_steps_from_cache` with `_force_full_chunk_pass=False` and
`=True` returned identical-length sequences on every repetition for both
harnesses (asserted in the benchmark script before recording any timing).
Byte-for-byte sequence-content equivalence across `_force_full_chunk_pass`,
world sizes, and `max_steps` shapes — plus equivalence to the independent
`build_repeating_micro_step_stream` oracle — is additionally covered by the
committed unit test
`tests/training/test_pack_cache.py::test_rank_selective_loading_matches_full_pass_and_repeating_stream_oracle`
(4 parametrized `(max_steps, grad_accum, world_size)` shapes × every rank).

### Functional probes (unpaired, correctness-only; integration-level, on top
of the unit tests in `tasks.md` 2.1/2.2)

Ran against a disposable 6-row scratch JSONL copy placed and then deleted
inside the same directory as the real dataset (so its original relative
image references resolved) — the real production dataset file
(`train.coord.jsonl`) itself was never opened for write. Removed immediately
after the probe; directory listing confirmed unchanged before/after.

1. **Touch stable**: built once (`build_status: "built"`,
   fingerprint `82fca7e3...`); `touch`'d the scratch file (`+1h` mtime, byte
   content unchanged); rebuilt — `build_status: "hit"`, same fingerprint
   `82fca7e3...`. No rebuild triggered.
2. **Content-change miss**: changed one byte (`"bowl"` → `"bowM"` in one
   `desc` field) in that same scratch file; rebuilt — new fingerprint
   `ec1c98df...` (differs from `82fca7e3...`), `build_status: "built"`
   (miss, rebuilt).

### Decision

**ACCEPT** per `measurement-plan.md` M2's stated criterion: "rank-selective
improves over full-pass on at least one harness and regresses neither."
H-steady-bounded shows a reproducible ~22-24% improvement; H-short shows no
regression (noise-level parity, structurally incapable of showing a win at
this sample size). Rank-selective chunk loading (`_force_full_chunk_pass`
default `False`) remains the shipped default;
`load_rank_micro_steps_from_cache`'s public behavior is otherwise unchanged.

## M3 (Wave 3, tasks 3.1-3.4) — 2026-08-04, final disposition 2026-08-04 after Opus review

**Result: stop rule applied — overlap is REJECTED as the shipped default;
`synchronous` ships as the default provider mode (`overlapped` remains
fully implemented, semantically proven equivalent, and selectable via the
existing debug-only `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE=overlapped`
switch). Two non-replicating world_size=1 measurement sessions (see below)
did not establish the "steps/hour improves" condition the stop rule
requires for keeping overlap as the default; the exact named 8-rank
H-short/H-steady harness is future confirmation work that may promote
overlap back to the default if it demonstrates a real win there. Full
evidence, exact commands, and every test name are recorded in `tasks.md`
under 3.1-3.4; this section is the compact cross-reference plus the parts
that don't fit tasks.md's per-task shape.
History: originally reported as ACCEPT (+0.8%) from a single 3-repetition
session; the Opus independent review (P2-H, see `review-triage.md`) flagged
that framing as an overstated effect size at n=3, and a follow-up
5-repetition re-measurement against the review's own P1-1 depth-one fix did
not replicate the original win (2/5 repetitions favored overlap) — the stop
rule was then applied literally rather than retaining overlap as default on
an unproven basis (see "Real-GPU paired A/B" below for both sessions,
preserved verbatim).**

### Environment

- HEAD: `a19ffceeb` plus this session's uncommitted Wave 1/2/3 working-tree
  changes (the code under test).
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch
  `coordexp-swift`.
- Python env: conda `ms`.
- GPU occupancy at measurement time: GPU 0 and 1 idle (0 MiB / 0%); GPUs
  2-7 external-process-occupied (10913-46505 MiB used, 39-100% util) —
  the same class of contention M0 recorded on 2026-08-03. The named
  H-short/H-steady harnesses require `accelerate8` (all 8 ranks), so the
  exact harness was not safely runnable without touching busy GPUs; used
  only the two confirmed-idle GPUs (`CUDA_VISIBLE_DEVICES=0`, single
  process) for the real-GPU A/B instead, clearly labeled as a scoped
  substitute rather than the named 8-rank harness.

### Cache-root discipline

`COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m3_probe/cache_root`
exclusively for the real-GPU probe (built once from a temporary derived
config `extends`-ing the exact H-steady prod base path, reused unmodified
by every repetition of both arms — confirmed via identical
`materializations.train.semantic_fingerprint` across all 6 runs' `run.json`
files). Shared production root (`.cache/coordexp_swift/packing`) verified
untouched before/after: 35 fingerprint directories both times (matching
the M0/M2 receipts), root mtime unchanged (predates this session). All
scratch configs/logs/caches lived under `/tmp/coordexp_m3_probe/`, deleted
after the measurement. No `python -m src.prepare_train_cache` or any write
path touched the shared root; task 6.1a remains untouched and still
requires explicit future user authorization.

### Internal control

`COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` environment variable
(`src/training/forward_input_provider.py::resolve_forward_input_provider_mode`)
— not a YAML/CLI surface, consumed only by `_run_initialized_training`'s
provider construction. Unset/absent resolves to `overlapped` (the shipped
default); `synchronous` selects the reference path for the A/B control arm;
any other value fails closed with
`training.forward_input_provider_mode_invalid`.

### Deterministic loss-stream equivalence (no GPU required)

Full test detail is in tasks.md 3.4. Summary: a dedicated harness in
`tests/training/test_forward_input_provider.py` runs the real
`SupervisedTrainer` + real `SynchronousForwardInputProvider`/
`OverlappedForwardInputProvider` + real `build_qwen_forward_inputs`/
`run_qwen_forward`, with a deterministic "echo" fake model whose logits are
a closed-form function of the actual constructed `input_ids`/`pixel_values`
values (so any tensor divergence between provider modes would change the
loss). Both a normal fixed 2-step run and a synthetic finite-gate
early-break case (forced via a call-index-keyed unsafe gate on the fake
runtime) produce bit-for-bit identical `loss_bundle_artifact`,
`optimizer_update_status`, `finite_status`, and `micro_step_count` between
modes. This is the decisive evidence for the loss-stream-equivalence gate
requirement — it isolates the provider-mode variable from ordinary
GPU/CUDA floating-point non-determinism, which the live-GPU run below
cannot do past the first optimizer step.

### Real-GPU paired A/B (scoped single-GPU substitute — two sessions)

Same methodology both sessions: back-to-back repetitions, same warm
temporary cache reused by every arm within a session; script: ad hoc
`python -m src.train` invocations against a scratch derived config (deleted
after each measurement, never committed) — `extends`
`configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml`,
overrides `run.name`, `data.train.sample_limit: 64`,
`data.eval.sample_limit: 16`, `training.max_steps: 6`,
`training.effective_batch_size: 4`, `checkpoint.save_final: false` (eval/
checkpoint cadence left at the inherited prod fractions — harmless to the
measured fields since Wave 1's `step_duration_seconds` boundary already
excludes scheduled-handler time by construction). Toggled only
`COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`.

**Session 1 (pre-fix code, 2026-08-04 morning, GPUs 2-7 external-process-occupied,
3 repetitions):**

| Rep | Arm | mean `step_duration_seconds` (6 steps) | mean `input_wait_seconds` | mean `input_build_seconds` |
| --- | --- | --- | --- | --- |
| 1 | synchronous | 10.469 | 0.000 | 0.653 |
| 1 | overlapped | 10.374 | 0.163 | 0.597 |
| 2 | synchronous | 10.509 | 0.000 | 0.616 |
| 2 | overlapped | 10.472 | 0.160 | 0.598 |
| 3 | synchronous | 10.607 | 0.000 | 0.645 |
| 3 | overlapped | 10.423 | 0.158 | 0.587 |

Overlapped won every repetition (median steps/hour: sync 342.6, overlapped
345.4, +0.8%; best-case rep +1.8%).

**Session 2 (current fixed code — P1-1 depth-one build-slot gate + P2-A
CPU-only synchronous build — 2026-08-04 later, all 8 GPUs confirmed idle
before starting, only GPU 0 used, 5 repetitions):**

| Rep | Arm | mean `step_duration_seconds` (6 steps) | mean `input_wait_seconds` | mean `input_build_seconds` |
| --- | --- | --- | --- | --- |
| 1 | synchronous | 10.450 | 0.000 | 0.379 |
| 1 | overlapped | 10.483 | 0.176 | 0.839 |
| 2 | synchronous | 10.599 | 0.000 | 0.416 |
| 2 | overlapped | 10.544 | 0.204 | 0.751 |
| 3 | synchronous | 10.524 | 0.000 | 0.409 |
| 3 | overlapped | 10.568 | 0.300 | 0.874 |
| 4 | synchronous | 10.643 | 0.000 | 0.516 |
| 4 | overlapped | 10.689 | 0.185 | 0.738 |
| 5 | synchronous | 10.995 | 0.000 | 0.848 |
| 5 | overlapped | 10.799 | 0.397 | 1.519 |

Only 2 of 5 repetitions favored overlapped (per-repetition steps/hour
change: `-0.31%, +0.52%, -0.42%, -0.43%, +1.81%`); median steps/hour 339.7
sync vs 340.6 overlapped (+0.27%, noise-level parity); mean 338.4 vs 339.1.
**This does not replicate session 1's "every repetition favors overlapped"
result.**

Two notable patterns in session 2, neither fully explained:

1. `input_build_seconds` for the overlapped producer thread runs
   consistently ~1.5-2x *higher* than the synchronous provider's CPU build
   time in the same repetition (e.g. rep 5: sync 0.848s vs overlap 1.519s),
   despite both now building via the identical CPU-only
   `_build_forward_inputs(..., device=None)` code path (P2-A made this
   symmetric). Leading hypothesis: CPython GIL contention between the
   producer thread's CPU-bound build work and the main/consumer thread's
   own Python-level GPU-orchestration overhead (tensor dispatch, autograd
   bookkeeping around `model.forward()`/`backward()`) inflates the
   producer's *wall-clock* build time without it doing more real work. Not
   investigated further (would need `time.thread_time()`/`time.process_time()`
   instrumentation — out of scope for this fix pass); recorded as an open
   question.
2. Build times trend upward across repetitions within session 2 itself
   (rep 1 sync 0.379s → rep 5 sync 0.848s, more than doubling), consistent
   with host-load drift (thermal, page cache, or other contention)
   accumulating across the session and adding noise independent of provider
   mode.

Both sessions confirm `input_wait_seconds` is genuinely nonzero and bounded
by `input_build_seconds` for the overlapped mode (never exceeding it) — the
queue-wait/overlap mechanism is demonstrably real and working as designed;
what session 2 disproves is that this reliably translates into a net
wall-clock win at this specific tiny scale.

Correctness cross-check (unpaired, informational only — not the decisive
equivalence evidence, see above): `loss/total` was bit-identical between
the synchronous and overlapped arms for planned steps 1-2 of session 1's
first repetition (before either run's optimizer had updated weights enough
times for ordinary non-bit-deterministic CUDA reduction-kernel behavior
across two independent process launches to compound); steps 3-6 diverge at
the ~1e-3 relative level, consistent with normal cross-process GPU
floating-point non-determinism rather than a provider-mode defect — this is
exactly the confound the deterministic CPU-level harness above was built to
rule out.

### Decision

**Stop rule applied: overlap REJECTED as the shipped default; `synchronous`
ships as the default.** Session 1's positive result (3/3 repetitions
favoring overlapped) did not replicate in session 2 (2/5, noise-level
median difference) against the depth-one-corrected code with more
repetitions. This scoped substitute did not establish the "steps/hour
improves" condition `measurement-plan.md` M3's stated criterion requires to
keep overlap as the default, so the stop rule ("if steps/hour does not
improve, revert the wiring, keep the tests, record rejected-with-evidence")
is applied literally: `resolve_forward_input_provider_mode()` now returns
`synchronous` when unset; `overlapped` remains fully implemented,
semantically proven equivalent, and selectable via the existing
debug-only `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE=overlapped` switch.
This is not a claim that overlap does not work: (1) no *correctness*
regression under either mode, proven decisively and deterministically,
independent of GPU timing noise; (2) the design's own stated expectation
(`design.md` Seam A / `measurement-plan.md`) is that this mechanism's win
concentrates at larger world size and larger CPU-build-to-GPU-compute
ratios (larger images, more micro-steps per rank) — a single-GPU,
2B-parameter, single-image-pack probe is explicitly the least favorable
shape for this mechanism, so this null/mixed result at world_size=1 does
not by itself prove overlap would fail at 8-rank production scale; (3) the
mechanism-level plumbing is demonstrably real (`input_wait_seconds`
genuinely nonzero and bounded), only its net wall-clock payoff at this tiny
scale is unproven. The exact named 8-rank H-short/H-steady `accelerate8`
harness remains **future confirmation work, not yet run**, and is the only
evidence that can promote `overlapped` back to the shipped default.

**P2 recorded for that future 8-rank M3 (not built now):** P2-A changed
`SynchronousForwardInputProvider` from a single fused CPU+device build call
to the same two-phase (CPU build, then separate device move) construction
as the overlapped provider, purely so `input_build_seconds` would be
comparable across modes. The *current* synchronous-vs-overlapped A/B
therefore no longer isolates whether the two-phase construction itself
(independent of threading/queueing) costs anything relative to the
original single-call fused build used by the legacy `_default_qwen_forward`
path (`forward_input_provider=None`). The exact future 8-rank M3 should
compare three arms — legacy fused (`forward_input_provider=None`),
two-phase synchronous, and overlapped — rather than only two, so any
two-phase-construction overhead is not silently attributed to (or hidden
inside) the overlap/no-overlap comparison. No third-arm benchmark is built
as part of this fix pass.

**Profiling-receipt reading note (2026-08-04 follow-up review, docs-only —
no new field or framework):** `QwenForwardInputsReceipt`'s input sub-timing
(`timings_ns["total_build_inputs_ns"]`, which feeds `input_build_seconds`)
describes CPU-only input construction under BOTH provider modes as of P2-A
above — it explicitly excludes the consumer-side host-to-device (H2D) move,
which now happens separately in `take()` and is folded into the caller's own
`step_duration_seconds` measurement instead. Before P2-A, the (then sole)
fused synchronous construction path's sub-timing included H2D; after P2-A,
neither provider mode's `input_build_seconds` does. Consequently:
`input_build_seconds`/`total_build_inputs_ns` values recorded in this
session's M3 tables (both sessions above) are CPU-only and MUST NOT be
compared numerically against any pre-Wave-3, pre-P2-A fused-build sub-timing
figure — the two measure different spans of work, not the same span at two
points in time. For end-to-end step wall-clock comparisons (the quantity
that actually matters for the M3 accept/reject decision and for any future
8-rank confirmation), use `step_duration_seconds` — the all-rank-maximum,
compute/optimizer-boundary measurement that includes H2D under every
provider mode and every era of this code — never a sub-timing field. No new
receipt field or profiling framework is introduced by this note; it
documents existing `timings_ns`/`step_duration_seconds` semantics that were
already implemented by P2-A and task 1.3, respectively.

Outstanding: (1) the tasks.md-mandated "focused independent review of the
concurrency lifecycle" for Wave 3 — the Opus pass recorded in
`review-triage.md` is exactly that review and returned HOLD, fixed same day
including this final stop-rule disposition; a narrow independent
re-review specifically confirming the P1-1 depth-one fix and this final
disposition is still recorded as pending in `review-triage.md`'s
"Disposition" section, but no longer blocks tasks 3.2/3.4 (the stop-rule
outcome is itself complete and honestly described). (2) Re-running the full
named 8-rank H-short/H-steady `accelerate8` harness — with the three-arm
methodology above — remains the only path to promoting `overlapped` back to
the shipped default.

## M4 (Wave 4, tasks 4.1-4.4) — 2026-08-04

**Result: correctness/exactness fully proven on a decisive CPU-only fixture
matrix; the exact 8-rank GPU wall-time measurement could not be run safely
this session (all 8 GPUs occupied by unrelated external work at every check
performed). `replicated` remains the shipped default; `disjoint_shard` is
fully implemented, tested, and selectable via the internal
`COORDEXP_SWIFT_EVAL_REDUCTION_MODE=auto` control for a future 8-rank
confirmation run. No cache build or GPU launch was attempted this wave.**

### GPU availability receipts (read-only `nvidia-smi`, no launch attempted)

| Time (UTC) | GPUs 0-7 | Disposition |
|---|---|---|
| 2026-08-04 05:03:52 (start of session, before any Wave 4 work) | 0 MiB / 0% util on all 8 | Idle — safe window, but code was not yet implemented/tested |
| 2026-08-04 05:29:13 (after implementation, tests, and hygiene gate complete) | 13,241-26,773 MiB used, 88-100% util on all 8 | Fully occupied by unrelated external work — unsafe to launch an 8-rank job |

The idle window at session start closed before implementation, decisive
correctness testing, and the hygiene gate were complete (all of which were
prerequisites for a meaningful M4 launch per the task's own gate ordering —
tests come before measurement). No GPU work was performed at either check;
both were read-only `nvidia-smi` queries. This is the same class of
occupancy M0 (2026-08-03) and M3 (2026-08-04, session 1) recorded — GPU
contention in this shared environment is common, not unique to this wave.

### Correctness evidence in place of the GPU launch

Task 4.1's fixture matrix (`tests/eval/test_forward_eval.py`,
`tests/runtime/test_train_runtime.py`) proves full-row exactness using the
REAL `LossRunner` and REAL `TrainRuntime` production reduction code — not a
reimplementation or a mocked stand-in — driven through a small thread-based
`_ThreadedRankCollective` that simulates the production bounded gatherer
with genuinely concurrent rank threads. This satisfies task 4.4's own stated
stop-rule criterion ("revert to replicated evaluation if full-row exactness
cannot be demonstrated on the fixture matrix") — exactness IS demonstrated,
so there is no correctness basis to revert or withhold the `disjoint_shard`
implementation. What remains unproven is specifically the *measured wall-time
win* at 8 ranks, which is a distinct, evidence-gated question the task
correctly separates from correctness: `resolve_active_eval_reduction_mode`
still returns `replicated` whenever the internal
`COORDEXP_SWIFT_EVAL_REDUCTION_MODE` control is unset (see task 4.2's DONE
receipt), so no unproven performance claim is silently shipped as the
default. This mirrors Wave 3's disposition of `overlapped` exactly: proven
correct, implemented, selectable, not promoted to default without measured
evidence.

### What remains outstanding for a future 8-rank confirmation

1. The exact named M4 harness (H-short at 8 ranks, `sharded` vs
   `forced-replicated` via `COORDEXP_SWIFT_EVAL_REDUCTION_MODE`, same
   temporary cache, paired A/B eval-invocation wall time) — requires all 8
   GPUs idle simultaneously, which did not hold at any point this session
   after implementation was ready to measure.
2. Checkpoint-selection replay on an actual recorded multi-rank run (as
   opposed to the fixture-level proof in task 4.3) — requires the same
   8-rank launch.
3. The "focused independent review of the distributed reduction" named by
   task 4.4 — a separate reviewer pass, structurally outside this
   delegation's scope (implement-only, tasks 4.1-4.4); recorded as
   outstanding in `review-triage.md`, matching how Wave 3's own residual
   review item was tracked as a distinct pending line rather than silently
   dropped.

None of these three items block tasks 4.1-4.3 (complete, tested, evidenced)
or the correctness half of 4.4; they are the specific, named prerequisites
for promoting `disjoint_shard` to the shipped default, and are recorded here
as pending rather than silently skipped or fabricated.

### FA2 branch-proof capture under disjoint-shard sharding (2026-08-04 follow-up review, docs-only)

`_run_initialized_training` (`src/training/pipeline.py`) applies
`_apply_fa2_branch_proof_policy` to `eval_micro_steps` AFTER the
`disjoint_shard` sharding filter (`partition_eval_micro_steps_for_rank`)
has already run. `_apply_fa2_branch_proof_policy`'s `first_micro_step`
policy captures FA2 branch proof on whichever micro-step sits at
`local_index == 0` of the sequence it is GIVEN. Consequently, under
`disjoint_shard` eval, each rank's `first_micro_step` proof capture is taken
on that RANK's own shard-local first pack (post-partition), not on the
globally canonical first pack of the full eval set the way `replicated`
mode's identical policy application would produce (every rank sees the same
full, unpartitioned sequence in `replicated` mode, so all ranks' proofs
target the same pack there).

This is intentional and correct, not a defect: the proof's purpose is to
verify the FA2 branch actually taken by THIS rank's real forward pass, and
under `disjoint_shard` each rank genuinely runs a different local pack
sequence — proving rank 0's shard-local first pack is the honest, real proof
for rank 0's own execution path, exactly as `every_forward`/`first_micro_step`
already do for training's own per-rank-sharded micro-step stream (the same
ordering relationship already exists there, unchanged by Wave 4). No
production code change is made here; this note only records the ordering so
a future reader does not mistake "different ranks proof different packs
under `disjoint_shard`" for a bug. `capture_fa2_branch`/
`require_fa2_branch_proof`/`fa2_branch_evidence` are pack-level fields
consumed inside the Qwen forward boundary for proof verification only —
they are not loss/metric inputs and do not appear in the eval row schema
(`ForwardEvalObservation`/`to_logging_row()`), so this ordering has no
metric or eval-row effect; it can only change which pack's proof is
recorded, never a scalar in the canonical row.

### Independent Opus review HOLD, fixed same day (2026-08-04, later)

An independent Opus review of this Wave 4 implementation returned HOLD on
two P1 findings (a genuine `finite/*` cross-rank reduction bug that could
silently emit a fractional 0.5 instead of the correct 0.0, and missing test
coverage for that exact case) plus five P2 findings (a sharding-key
correctness gap between `pack_index` and true sequence position, a
cross-rank liveness gap where a `reduction_mode` divergence could hang a
collective instead of failing closed, a process note, a speculative-code
note, and misleading docs). Full findings table, dispositions, and
re-verification receipts are recorded in `review-triage.md` under
"Independent implementation review (Opus, 2026-08-04) — Wave 4". All
findings were fixed in this same worktree the same day; no GPU work or
production cache access was needed for any of the fixes (all new/changed
tests are CPU-only). This M4 section's own content (GPU availability
receipts, correctness-evidence framing, outstanding items above) is
unaffected by that review pass — it already correctly described the M4 gap
as separate from correctness, which the HOLD findings did not dispute.

### Real 8-rank GPU measurement (2026-08-04, later session)

**Result: ACCEPT per `measurement-plan.md` M4's own criterion — disjoint-shard
eval reduction measured 7.27x faster (median) than replicated at 8 ranks on
H-short, with full-row equivalence proven exact across all 3 repetitions x 8
ranks x both modes, and checkpoint-selection replay byte-identical across
all 6 runs. This closes the previously-outstanding GPU-measurement gap for
task 4.4; `disjoint_shard` is not promoted to the shipped default by this
evidence-execution pass itself (see "Scope note" below) but every named 4.4
receipt is now satisfied.**

Environment: HEAD `a19ffceeb` plus this worktree's uncommitted Wave 1-5
working-tree changes (the code under test, `coordexp-swift` branch). Python
env: conda `ms`. GPU availability confirmed idle on all 8 GPUs immediately
before each launch (`nvidia-smi`, 2026-08-04 11:43-11:56 UTC); an unrelated
external 8-rank resume job in a different worktree
(`8-coords-bbox`/`prod_sorted_xy.yaml`) started at 11:58 UTC, strictly after
this session's 3rd/final repetition completed at 11:57:12 UTC — no GPU
contention or interference occurred during measurement, and none of this
session's work touched or was touched by that external job.

Cache-root discipline: `COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m4_probe/cache_root`,
built once via `python -m src.prepare_train_cache` against a temporary
derived config (`extends` H-short's exact path,
`configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml`,
overriding only `run.name`/`run.artifact_root` and `checkpoint.steps: [1]`
so the scheduled checkpoint aligns with the scheduled eval step for the
checkpoint-selection-replay evidence below -- `checkpoint.steps` is not a
pack-cache determinant, confirmed by the second `prepare_train_cache` call
returning `build_status: "hit"` for both the train and eval cache after this
override), reused unmodified by every one of the 6 runs (3 reps x 2 arms;
`resolved_config_fingerprint` differs only in `run.name`, cache fingerprints
`f3a7edab...` (train, 32 micro-steps) / `fb02c021...` (eval, 8 micro-steps)
identical across all 6). Shared production root
(`.cache/coordexp_swift/packing`) verified untouched: mtime `2026-08-04
10:31:12 UTC` (predates this session's first cache-prep command at
`11:47Z`), never referenced by any command (every invocation set
`COORDEXP_SWIFT_PACK_CACHE_ROOT` explicitly to the temp root). All scratch
artifacts lived under `/tmp/coordexp_m4_probe/`.

Internal control: `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` environment variable
(`resolve_eval_reduction_control()`, `src/eval/forward.py`) --
`auto` for the sharded arm, `replicated` for the forced-replicated arm.
H-short's eval set has `pack_count=8` at `world_size=8`
(`pack_count >= world_size`, so the structural `pack_count < world_size`
fallback does not apply and the control genuinely selects between
`disjoint_shard` and `replicated`); `validate_eval_reduction_consensus`
(the P2-B cross-rank liveness guard) ran unconditionally before sharding on
every rank of every run with no hang or fail-closed error, confirming
pre-sharding consensus held on all 6 runs.

Launch: `accelerate launch --num_processes 8 -m src.train --config
<derived config>` (via a thin measurement-only wrapper,
`/tmp/coordexp_m4_probe/timed_train.py`, that monkeypatches
`ForwardEvalRunner.run` to record wall time and the resulting
`to_logging_row()` per invocation before delegating to the unmodified
`src.train.main()` -- the wrapper adds no code to `src/`, changes no
production behavior, and is deleted after the measurement), with
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, back-to-back per repetition (sharded
arm immediately followed by the replicated arm against the same warm
temporary cache), 3 repetitions. All 6 runs exited status 0.

**Eval invocation wall time (max across 8 ranks per run, seconds):**

| Rep | Sharded (`auto`) | Replicated |
| --- | --- | --- |
| 1 | 1.4616 | 10.7132 |
| 2 | 1.7801 | 10.7365 |
| 3 | 1.4775 | 10.9179 |
| median | 1.4775 | 10.7365 |
| min | 1.4616 | 10.7132 |
| max | 1.7801 | 10.9179 |

Median speedup: 10.7365 / 1.4775 = **7.27x**. Per-rank spread within a run
was tight (e.g. rep 1 sharded 1.4575-1.4616s, replicated 10.7093-10.7132s),
consistent with a synchronized collective rather than a stray outlier rank.

**Full-row equivalence (per `measurement-plan.md` M4 / design Seam C
inventory):** for every one of the 3 repetitions, every one of the 8 ranks'
`ForwardEvalObservation.to_logging_row()` was compared between the sharded
and replicated arm. Result: byte-identical on every exact-count/top-k/
identity field (`acc_top1`, `acc_top5`, `example_count`, `pack_count`,
every `count/*`, every `finite/*`, every `loss/*/segment_count`, `step`,
`split`, `trigger_reasons`), and within rtol 1e-5 on every `loss/<term>`
scalar and `.../token_weighted_diag` (both arms were in fact bit-identical
on these fields too in this run, since H-short's tiny 8-pack/8-rank shape
gives each rank exactly one pack under `disjoint_shard`, making the
per-rank forward computation itself identical between modes -- only the
cross-rank reduction path differs). Row key sets were identical between
modes and no `__weight__`-suffixed key leaked into any row. All 8 ranks'
own rows were also identical to each other within each arm (single-emission
correctness), and this held identically across all 3 repetitions (no
repetition-to-repetition drift, consistent with a fully deterministic
bf16 forward + fixed eval order). Representative row (`acc_top1
0.3215420461766575`, `acc_top5 0.49481042152086424`, `loss/total
9.701848484575748`, `count/supervised_atoms 4721.0`, full row recorded in
this session's raw JSON artifacts) was identical between arms in every
repetition.

**Checkpoint-selection replay:** `checkpoints/best.json` was compared across
all 6 runs (3 reps x 2 arms). Every run selected `checkpoints/step-1` via
`selector: "acc_top1"` with `value: 0.3215420461766575` -- byte-identical
across all 6, matching `pipeline._checkpoint_handler`'s exact
`eval_observation.get("acc_top1")` read and proving selection is unaffected
by which reduction mode produced the row, now confirmed on a real recorded
multi-rank run rather than only the task 4.3 fixture-level proof.

**World-size-1 degeneracy:** not re-run this session -- already guaranteed
structurally (`resolve_active_eval_reduction_mode` forces `replicated` at
`world_size<=1`, `ForwardEvalRunner` rejects constructing `disjoint_shard`
at `world_size<=1`) and covered by the pre-existing, unmodified
`test_forward_eval_streaming_counts_and_wide_scalars`/
`test_forward_eval_returns_one_pure_wide_observation_and_restores_mode`
(task 4.4's earlier PARTIAL note already established this as sufficient).

**Focused independent review:** already performed and converged in a prior
session (see the "Independent implementation review (2026-08-04) -- Wave 4
distributed reduction" entry in `review-triage.md`, verdict APPROVE,
re-confirmed under "Pre-final Opus P2 closures"). Unaffected by this
session's measurement, which adds no code change.

**Default promotion closure:** `measurement-plan.md`'s M4 accept criterion
("sharded improves eval wall time over replicated at 8 ranks with full-row
equivalence green") is met by this evidence. The owning implementation lane
therefore changed `resolve_eval_reduction_control()`'s unset-env-var branch
(`src/eval/forward.py`) from `EVAL_REDUCTION_REPLICATED` to
`_EVAL_REDUCTION_CONTROL_AUTO` and updated the corresponding default-contract
test. `disjoint_shard` is now the shipped multi-rank default when
`pack_count >= world_size`; explicit `replicated`, the
`pack_count < world_size` fallback, and the world-size-one fallback remain
unchanged. This promotion adds no public YAML/CLI surface.

Cleanup: `/tmp/coordexp_m4_probe/` (derived configs, temp cache root, timing
wrapper, per-rank JSON artifacts, logs, scratch run outputs) deleted after
this measurement was recorded; no production file, shared cache root, or
unrelated worktree/process was modified.

## Independent implementation review fixes (Opus HOLD, 2026-08-04)

An independent Opus review of the Wave 1/2 implementation returned HOLD on
two P1 and six P2 findings (spec/code coherence and correctness fail-open
gaps — none touching M2's measured mechanism or its cache-root discipline).
Full findings table, dispositions, changed files, and re-verification
receipts are recorded in `review-triage.md` under "Independent implementation
review (Opus, 2026-08-04)". This M2 measurement (paired A/B timings, chunk
layout, functional probes, decision) is unaffected and unaltered by that
review pass — none of the eight findings touched `load_rank_micro_steps_from_cache`'s
measured chunk-skip mechanism, only its accuracy-stats plumbing (unrelated
subsystem), a receipt-string label, an unused fallback, and a spec-text gap.
No cache build or GPU work was performed for the fixes.

## M5a (Wave 5, task 5.3) — 2026-08-04

**Result: REJECT both sub-candidates with evidence. Neither the hoisted
`examples_by_id` map nor the bisect-based token-span lookup is shipped;
production keeps the pre-existing per-pack rebuild and full linear rescan.
Both candidates were implemented, proven equivalence-correct (or, for
bisect, proven NOT equivalence-correct) behind a private/test-only
module-attribute control, measured, and then fully reverted — no residual
production wiring, no dead candidate code left in `src/`.**

### Environment

- HEAD: `a19ffceeb` plus this session's uncommitted Wave 1-5.2 working-tree
  changes (unrelated, preserved as-is).
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch
  `coordexp-swift`. Python env: conda `ms`, `torch==2.9.1+cu128`.
- GPUs not used: cache materialization (`load_qwen_components(..., load_model=False)`)
  is CPU/tokenizer/image-processor only, no model weights loaded.

### Cache-root discipline

All materialization runs (equivalence tests and timing) used a fresh
`tempfile.mkdtemp()` directory per run as `COORDEXP_SWIFT_PACK_CACHE_ROOT`,
deleted immediately after each run. The shared production root
(`.cache/coordexp_swift/packing`) was never opened. Dataset: the existing,
already-sample-limited smoke config
`configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_ebs128_1step.yaml`
(`data.train.sample_limit: 128`, real COCO rows, real base-model tokenizer at
`/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`),
used read-only.

### Sub-candidate 1 — hoisted `examples_by_id` (design Seam F,
`src/training/pipeline.py:_encoded_examples_for_pack`)

**Internal control (during measurement, since reverted):** a private
module-level `_PACK_EXAMPLES_LOOKUP_STRATEGY` flag (`"rebuild_per_pack"`
legacy / `"hoisted_map"` candidate) plus a new `_encoded_examples_for_pack_from_map`
helper, monkeypatched only in tests/benchmarks — never a public surface.

**Equivalence:** two paired tests (since reverted) —
(1) direct comparison of `_encoded_examples_for_pack` vs.
`_encoded_examples_for_pack_from_map` on a synthetic multi-segment,
multi-example, out-of-order pack — byte-identical tuples; (2) full
`_build_micro_steps_for_dataset` run twice (legacy vs. hoisted strategy) on
the real 128-example/16-pack dataset above — identical `encoded_examples`,
`pack`, `token_sequence` per micro-step, and `torch.equal`-identical
`position_inputs.position_ids` (dataclass `==` on `QwenPositionInputs` isn't
usable directly — it contains a tensor field and raises
`RuntimeError: Boolean value of Tensor with more than one value is
ambiguous` under Python's default dataclass `__eq__`, so the test compares
tensor fields via `torch.equal` explicitly). Both green.

**Isolated hot-loop timing** (the actual changed code, in isolation, 200
repetitions, same 128-example/16-pack dataset object, no I/O):

| Strategy | mean time for one full per-pack lookup pass (16 packs) |
|---|---|
| legacy (`rebuild_per_pack`) | 0.323 ms |
| hoisted (`hoisted_map`) | 0.034 ms |

A genuine, expected **~9.5x** local speedup — legacy is O(packs × dataset
examples) (rebuilds a full dict of all 128 examples for each of the 16
packs); hoisted is O(dataset examples + packs).

**End-to-end paired A/B** (`prepare_training_pack_caches` wall time, full
cold materialization including tokenization/image encoding at the default
`materialization_workers=16` fork pool, 5 repetitions, order alternated per
repetition to cancel drift):

| Strategy | times (s) | mean (s) |
|---|---|---|
| legacy | 2.777, 3.213, 3.259, 3.244, 3.412 | 3.181 |
| hoisted | 3.142, 3.208, 3.121, 2.917, 3.211 | 3.120 |

Mean difference ~1.9%, but the two distributions overlap substantially
(legacy's fastest rep, 2.777s, is faster than 4 of the 5 hoisted reps) — not
a reproducible, decisive signal. This is the expected consequence of the
isolated-loop number: 0.32 ms out of a ~3.1 s total is ~0.01% of wall time,
several orders of magnitude below the run-to-run noise floor (tokenization
and image encoding dominate). A back-of-envelope scaling check (the
legacy cost is O(packs × examples) ≈ O(examples²/8) given this dataset's
observed ~8-segments-per-pack density, while total wall time scales
O(examples)) shows the dict-rebuild cost would not reach even 10% of total
wall time until roughly ~120,000 examples — far outside any dataset size
compatible with the "sample-limited... temporary cache" measurement
discipline this wave requires.

**Decision: REJECT (wall-clock not decisive at any measurement-compliant
scale).** Per M5a's stated criterion ("hoisted... improves over the old
lookup; else revert"), the candidate is correctness-neutral and
algorithmically superior in isolation but does not produce a measurable
`prepare_training_pack_caches` wall-time win within the sample-limited,
temporary-cache-root measurement rule. `src/training/pipeline.py` and
`tests/training/test_pipeline_assembly.py` were reverted to their
pre-5.3 (Wave-5.1/5.2-baseline) state (`git diff --stat` shows zero net
change from those two files vs. before this task; residue grep for
`_PACK_EXAMPLES_LOOKUP_STRATEGY`, `_encoded_examples_for_pack_from_map`,
`hoisted_examples_by_id` across `src/`/`tests/`: zero matches).

### Sub-candidate 2 — bisect-based token-span lookup (design Seam F,
`src/qwen/encoding.py:_token_indices_for_char_range`)

**Internal control (during measurement, since reverted):** a private
module-level `_TOKEN_SPAN_LOOKUP_STRATEGY` flag (`"linear"` legacy /
`"bisect"` candidate); `_align_rendered_spans` hoisted a `token_ends` tuple
once per example (only under `"bisect"`) and threaded it through
`_align_one_span` to a new `_token_indices_for_char_range_bisect`, which
used `bisect.bisect_right` on `token_ends` to find the first
potentially-overlapping token, then scanned forward with an early `break`
at the span's right edge (replacing the legacy full unconditional rescan of
every remaining offset, which never `break`s even after the span's tokens
are all found).

**Representative-corpus equivalence:** the real single-image fixture
(`tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`, real Qwen
tokenizer via `load_qwen_components(..., load_model=False)`) encoded once
under each strategy — `supervised_token_spans` and `ignored_token_spans`
were exactly equal, and the fixture's real `base_offset_mapping` was
confirmed monotonic in both `token_start` and `token_end` (a precondition
`bisect` silently assumes but nothing in the codebase enforces or asserts).

**Adversarial-edge equivalence (the decisive falsification):** a synthetic
`offset_mapping = [(0,5), (5,10), (0,0), (10,15)]` — three real tokens plus
one zero-width, special-token-like `(0,0)` entry inserted mid-sequence
(exactly the "special-token offset" edge case named in the task) — breaks
`token_ends`'s monotonicity (`(5, 10, 0, 15)`). Querying `char_start=5,
char_end=15`:

- **linear** (legacy): returns `(1, 3)` — the two real overlapping tokens,
  correctly skipping the mid-sequence `(0,0)` entry (its `token_end=0 <=
  char_start=5`, skip, independent of position).
- **bisect** (candidate): `bisect_right((5,10,0,15), 5)` returns index `3`
  (binary search over a non-monotonic array), so the scan starts at index 3
  and never considers index 1 at all — it returns `(3,)` and additionally
  **raises** `EncodingContractError(code="qwen.span_token_coverage",
  context={"covered_range": [10, 15]})`, a different token-index result AND
  a spurious hard failure, for the identical input the legacy path handles
  correctly.

Verified by direct execution (not just hand-trace).

**Decision: REJECT (equivalence not green).** Per M5a's stated criterion
("bisect... only with an exact boundary-equivalence test... including the
crossing-boundary failure cases"; "Accept only with equivalence green"),
this is a genuine, reproducible divergence, not a hypothetical one. Nothing
in `encode_rendered_example`'s tokenizer call (`add_special_tokens=False`,
`return_offsets_mapping=True`, single whole-chat-text tokenize, no padding)
currently produces such a non-monotonic `offset_mapping` in practice — the
representative-corpus check passed — but nothing asserts or guarantees that
invariant either, so a future tokenizer/model change or unusual
byte-fallback/normalization input could silently reintroduce it. Given the
adversarial edge fails, `src/qwen/encoding.py` was reverted in full to its
pre-5.3 state (`git status --porcelain` empty for this file; residue grep
for `_token_indices_for_char_range_bisect`, `_TOKEN_SPAN_LOOKUP_STRATEGY`
across `src/`/`tests/`: zero matches). The paired A/B equivalence test
itself was not kept (it exercised the now-deleted bisect function), but per
Wave 5 review P2 (2026-08-04) the lesson is preserved durably in live
code/tests instead of only in this note: a short comment at
`_token_indices_for_char_range` (`src/qwen/encoding.py`) states the
non-monotonicity hazard, and a permanent regression test
`tests/qwen/test_encoding.py::test_token_span_lookup_handles_non_monotonic_zero_width_offset_without_bisect`
asserts the exact counterexample above (`[(0,5),(5,10),(0,0),(10,15)]`,
range `[5,15)`) resolves to `(1, 3)` with no `qwen.span_token_coverage`
error, so any future reintroduction of a bisect/binary-search shortcut over
`offset_mapping` would be caught immediately rather than requiring this
history to be rediscovered.

## M5b (Wave 5, task 5.4) — 2026-08-04

**Result: REJECT (deferred pending CUDA evidence). A batched
per-device/dtype `torch._foreach_norm` gradient finite-gate candidate was
implemented, proven exactly equivalent to the legacy per-parameter path
across a 25-case corpus (mixed dtype, NaN, +Inf/-Inf, norm-overflow-from-finite-values,
missing grad, all-grads-missing, empty params, sparse-gradient rejection,
backend-overflow passthrough), then measured CPU-only (GPUs were saturated
by external jobs throughout this session) and found not decisively better
even on CPU — and CPU is not the decision-relevant device for this
mechanism. Fully reverted; production keeps `build_gradient_finite_report`
unchanged and as the sole path.**

### GPU availability (read-only `nvidia-smi`, no launch attempted)

Checked twice, ~40 minutes apart:

| Time | GPU 0 | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
|---|---|---|---|---|---|---|---|---|
| check 1 (util%, mem used/80GiB) | 100%, 42.1GB | 100%, 28.2GB | 100%, 42.4GB | 100%, 40.2GB | 100%, 27.7GB | 100%, 33.0GB | 100%, 40.9GB | 100%, 47.6GB |
| check 2 (util%, mem used/80GiB) | 97%, 42.1GB | 41%, 28.2GB | 69%, 42.4GB | 88%, 40.2GB | 82%, 27.7GB | 34%, 33.0GB | 48%, 46.7GB | 81%, 47.6GB |

All 8 devices carried substantial external memory allocations and
utilization at both checks. Per instruction ("check live GPUs before any
use and never disturb external jobs"), no CUDA tensor was allocated at any
point for this task; the whole exercise (equivalence tests and benchmark)
ran under `CUDA_VISIBLE_DEVICES=""`.

### Design (candidate, since reverted:
`src/runtime/finite_gates.py:_build_gradient_finite_report_batched`)

Groups gradients by `(device, dtype)`. Per group: elementwise
`torch.isfinite(grad).all()` (device-side reduction, no sync) plus
`torch._foreach_norm([grad.float() for grad in group])` (batched norm,
device-side). Per distinct device: one `torch.stack(...).all().item()` call
for the aggregated finiteness flag and one `torch.stack(...).cpu().tolist()`
call for the aggregated norms — **2 host syncs per distinct device total**,
independent of parameter count, vs. the legacy path's 2 syncs
(`.item()` + `.cpu()`) **per parameter**. Norm values are then fed through
the identical Python-level sticky-to-infinity double-precision summation
loop the legacy path uses (`if isfinite(norm) and isfinite(squared_norm):
squared_norm += norm**2; else: squared_norm = inf`), so `grad_norm`'s
overflow/NaN semantics are reproduced exactly; only the summation *order*
can differ (grouped-by-dtype/device order vs. strict parameter-iteration
order), which this change's own invariant #2 ("bit-compatible up to fp
summation order") explicitly allows. Explicit finiteness is never inferred
from the norm alone — both signals are computed independently, matching
the legacy design's separation (verified by the mixed-dtype-with-late-NaN
test below, which specifically defeats a norm-only-inference
implementation).

### Equivalence (25 tests, CPU-only, all green; since reverted with the
candidate)

`_build_gradient_finite_report_batched` vs. `build_gradient_finite_report`,
each report additionally fed through `reduce_gradient_overflow_reports` to
compare `GateDecision` fields:

- mixed dtype (fp32 + bf16 + fp64), all finite
- NaN confined to one dtype group
- mixed +Inf / -Inf across two fp32 params
- grad-norm overflow from otherwise-finite fp32 values (`[1e38, 1e38]`,
  matching the pre-existing `test_build_gradient_report_non_finite_norm_triggers_global_skip`
  oracle)
- missing grad on one of two parameters
- all grads missing (empty saw_grad path)
- zero parameters at all
- `backend_overflow=True` passthrough
- 4 dtype groups (fp32/bf16/fp16/fp64) with a NaN injected into the *last*
  element of the *last* group only — specifically defeats any candidate
  that short-circuits per group instead of checking every group
- sparse (`coo`) and compressed-sparse (`csr`) gradient rejection — same
  `runtime.sparse_gradient_unsupported` code and identical `context` dict
  (`planned_step_id`/`rank`/`world_size`/`shape`/`layout`)

`grad_norm` compared with `pytest.approx(..., rel=1e-9)` when finite (per
the summation-order allowance above); exact equality for
`gradients_finite`, `backend_overflow`, `GateDecision.reason_codes`,
`all_ranks_safe`, `should_call_optimizer_step`, `should_clear_gradients`,
`optimizer_update_status`, `diagnostics["unsafe_rank_count"]`; both-None or
both-inf/both-NaN branches handled explicitly for `grad_norm`. 25/25 passed.

### CPU-only overhead measurement (`CUDA_VISIBLE_DEVICES=""`; not a CUDA
substitute — see disposition)

Synthetic DoRA/LoRA-shaped parameter set matching this repo's production
adapter surface (`adapter.target_modules: all_linear`, `rank: 8`, hidden
3584): 28 layers × 7 linear submodules × 3 params (`lora_A`, `lora_B`,
`magnitude`) + 1 bf16 embedding-group tensor = 589 parameters. Two runs:

| Run | reps | legacy mean (ms) | batched mean (ms) | ratio |
|---|---|---|---|---|
| 1 | 30 (order alternated) | 59.075 | 55.307 | 1.068x |
| 2 (confirmatory) | 30 (order alternated) | 48.581 (stdev 3.967) | 46.276 (stdev 3.788) | 1.050x |

A modest ~5-7% CPU-side improvement, but the per-run standard deviation
(~8% of the mean) is comparable to the measured effect size — not a clean,
decisive separation. This is the expected outcome: on CPU, `.item()`/`.cpu()`
calls do not incur the CUDA host-device synchronization stall the batched
design specifically targets (there is no async kernel queue to flush), so
the only remaining saving is fewer Python/`torch._foreach_norm` call-overhead
instances — a real but secondary effect, not the primary claimed mechanism.

### Decision

**REJECT / deferred-pending-CUDA-evidence.** Per M5b's stated criterion
("measured planned-step overhead reduction... otherwise revert") and this
wave's explicit rule ("Ship/delete legacy only if equivalence green AND
measured overhead improves on decision-relevant production device (CUDA);
otherwise revert candidate/default"): equivalence is green, but the only
overhead evidence obtainable this session is CPU-only (GPUs saturated by
external jobs at every check), CPU is not the decision-relevant device, and
even the CPU number is not decisive. `src/runtime/finite_gates.py` and
`tests/runtime/test_finite_gates.py` were reverted to their pre-5.4 state
(`git status --porcelain` empty for both files; residue grep for
`_build_gradient_finite_report_batched` across `src/`/`tests/`: zero
matches). `build_gradient_finite_report` remains the sole production path,
unchanged. This is recorded as deferred rather than permanently
foreclosed: the design and equivalence corpus above are the reusable basis
for a future CUDA-available re-attempt, but per this wave's own "no dead
candidate code" discipline, none of that code is left in `src/`/`tests/` —
this note is the retained evidence.

## 6.1a — production cache materialization, 2026-08-04

User-authorized single write to the default production packing-cache root
(`.cache/coordexp_swift/packing`) for the final settled fingerprint via
`CUDA_VISIBLE_DEVICES="" python -m src.prepare_train_cache --config
configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml`,
no model load (`"model_loaded": false`), no GPU. Resolved config fingerprint
`54122efaf7a0fe69838ccaba90ade06e865da66c43d6e42fc276a78cd1d1f0ef`.

### Pre-run state

Cache root held 35 pre-existing fingerprint directories (50 GB) plus 6
`.lock` files — matches the M3/M5a receipts above. Two earlier launch
attempts in this session (a 10-minute foreground timeout, then a
background job killed by session teardown between turns) both died before
reaching cache publication: directory count and lock-file count were
identical (35 / 6) immediately before and after each, confirming zero
residue from either. The successful run below is the sole publication.

### Result

Both caches were misses and freshly built (`"build_status": "built"`,
`"status": "complete"` in both manifests):

| split | fingerprint | micro-steps | chunks | size | cache dir |
|---|---|---|---|---|---|
| train | `e68f2699d83705bd2a82a938164ede5e5d800d291c4b9bd64c3ece85db97c668` | 14660 | 29 | 11 GB | `.cache/coordexp_swift/packing/e68f2699d83705bd2a82a938164ede5e5d800d291c4b9bd64c3ece85db97c668` |
| eval | `e9aa216516e72dfdd593afd8177e58d02b2c9f7f8aff64db267138a774324549` | 619 | 2 | 461 MB | `.cache/coordexp_swift/packing/e9aa216516e72dfdd593afd8177e58d02b2c9f7f8aff64db267138a774324549` |

Wall time 09:59:01Z-10:31:14Z UTC (1933.6 s, ~32.2 min), single process,
`DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS=16`.

### Post-run validation

Directory diff: post-run root has 37 directories (61 GB total); all 35
pre-existing directories present with unchanged mtimes; exactly the 2
fingerprints above are new — no prior cache deleted, repaired, or mutated.
Both new caches independently pass full payload-level validation via the
live admission API (`load_cache_manifest(cache_dir,
expected_fingerprint=fp, level="payloads")` and `cache_is_complete(...)`):
correct `version="coordexp-swift-pack-cache-v2"`, `status="complete"`,
`cache_is_complete=True` for both. `git status`/`git diff --check` after
the run are unchanged from the pre-run dirty-worktree state — the
materialization touched no source, config, or spec file. No commit, stage,
or push.

## 6.1b — M6 end-to-end production-shaped smoke, 2026-08-04

Both required M6 parts per `measurement-plan.md` ("M6 (final gate...)"):
H-short full-cycle (train + scheduled eval + checkpoint) on its own
disposable cache, and a bounded H-steady throughput window against the
newly materialized (task 6.1a) shared production caches. Absolute
wall-clock timings only — M0 is unavailable by user decision and is not
used as a comparison point.

Environment (both runs): HEAD `a19ffceeb` plus this worktree's uncommitted
Waves 1-5 working-tree changes (the code under test), conda env `ms`, host
`k8s-worker02`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (all 8 physical
GPUs, GPU5 included, per explicit user authorization). An unrelated
external 8-rank job (`8-coords-bbox` worktree, `prod_sorted_xy.yaml`
resume, PIDs 3222021-3222028) was already running on all 8 GPUs before,
during, and after both M6 launches (confirmed still running, unaffected,
after both completed) — co-location was explicitly authorized by the user
rather than waited out. No GPU process was killed or evicted. No OOM
occurred on either run: peak combined (external job + this run) GPU memory
stayed at or under ~59.5 GB/80 GB (worst-case GPU) throughout the whole M6
window; GPU5 specifically ranged 25.7-47.7 GB, never approached capacity.
Launch form (both): `accelerate launch --num_processes 8 -m src.train
--config <config>` (the canonical 8-rank form used by the M4 receipt
above).

### Config-loader constraint discovered (representational correction)

The first H-steady derived-config attempt tried an `extends` child that set
`eval.forward.every_fraction: null` / `checkpoint.every_fraction: null` to
disable the inherited production cadence. This failed closed with
`ConfigContractError[config.null_inherited_delete]` (`src/config/loader.py`)
— the loader deliberately refuses to let a child config null out a value
its parent already set non-null, to prevent silent cadence loss via
`extends`. A first fallback (a full non-`extends` duplicate of the base
config's determinant fields, written to prove the duplication was
determinant-correct — it also produced a cache hit) was rejected by the
user as not-genuinely-`extends` and discarded before any GPU launch used
it. The final, shipped derived config is a genuine minimal `extends`:
instead of nulling the inherited fractions, `eval.forward.every_fraction`
and `checkpoint.every_fraction` are overridden to `1.0` — the schema's own
maximum (`CadenceConfig.every_fraction: float | None = Field(gt=0.0,
le=1.0)`), a valid non-null override. Given
`_resolve_cadence_events`'s interval math
(`interval = max(1, ceil(every_fraction * resolved_max_steps))`), an
interval of `resolved_max_steps` produces zero intermediate triggers; the
unconditional `clamped_final` reason still fires at the last step
regardless of the fraction's magnitude, so only the final step (5) carries
any eval/checkpoint event. Verified directly by loading the real config and
calling `resolve_planned_step_schedule(config, packs_per_epoch=14660,
world_size=8)`:

```
resolved_max_steps: 5
--- checkpoint
  5 ('every_fraction:1:clamped_final', 'save_final')
--- eval.forward
  5 ('every_fraction:1:clamped_final',)
--- final
  5 ('final',)
```

Steps 1-4 carry no event at all — the bounded window is structurally
uncontaminated by intermediate eval/checkpoint firings (and, independent of
this, `step_duration_seconds` is measured strictly inside the
compute/optimizer boundary per task 1.3, excluding eval/checkpoint
handlers, so even step 5's own reported duration is compute-only).

Derived config (`/tmp/coordexp_m6_probe/h_steady_derived.yaml`, verbatim):

```yaml
schema_version: 1

extends: /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1.yaml

run:
  name: qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1_m6_hsteady_probe
  artifact_root: /tmp/coordexp_m6_probe/h_steady/outputs
  collision_policy: timestamp

training:
  max_steps: 5

eval:
  forward:
    every_fraction: 1.0

checkpoint:
  every_fraction: 1.0
```

No `data`/`template`/`packing`/`processor`/`model`/`adapter`/`augmentation`
key is touched — every packing-cache determinant is untouched, inherited
verbatim from the production base by absolute-path `extends`.

### H-steady: cache-hit proof before model load

`CUDA_VISIBLE_DEVICES="" python -m src.prepare_train_cache --config
/tmp/coordexp_m6_probe/h_steady_derived.yaml` (`"model_loaded": false`),
run against the default production cache root (`COORDEXP_SWIFT_PACK_CACHE_ROOT`
unset):

| split | build_status | fingerprint | micro-steps | matches 6.1a |
|---|---|---|---|---|
| train | `hit` | `e68f2699d83705bd2a82a938164ede5e5d800d291c4b9bd64c3ece85db97c668` | 14660 | yes, identical |
| eval | `hit` | `e9aa216516e72dfdd593afd8177e58d02b2c9f7f8aff64db267138a774324549` | 619 | yes, identical |

`resolved_config_fingerprint` (`3aa9de08a9fe348ee1706b7c0f98cea620ebf0873a5a9a97a75b18bbb9c2fac7`)
differs from 6.1a's production run's own fingerprint (`54122efaf7a0...`) as
expected — `run.name`/`training.max_steps`/`eval.forward.every_fraction`/
`checkpoint.every_fraction` are not packing-cache determinants, but they do
change the whole-config resolved fingerprint. Production cache root
confirmed byte-for-byte unchanged before and after this preflight and after
the full launch below: 37 fingerprint directories, root mtime
`2026-08-04 10:31:12` UTC (predates this session), both checked
immediately before and after.

### H-short: disposable-cache discovery (`training.pack_cache_not_prepared`)

The first H-short launch attempt (exact tracked config, dedicated disposable
`COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m6_probe/h_short/cache_root`,
no pre-build step) failed closed on all 8 ranks with
`RuntimeContractError[training.pack_cache_not_prepared]`: "distributed
training requires prepared packing caches; run `python -m
src.prepare_train_cache --config <path>` before `accelerate launch`".
Distributed training does not build packing caches itself — this matches
the M4 receipt's own methodology, which always pre-built via
`prepare_train_cache` first. All 8 ranks exited cleanly (SIGTERM from the
elastic launcher, exit code 1), zero stray GPU processes left behind
(confirmed via `nvidia-smi --query-compute-apps`), zero production-cache
interaction (the disposable root was never populated). Fixed by running
`CUDA_VISIBLE_DEVICES="" python -m src.prepare_train_cache --config
<H-short config>` first against the same disposable root:

| split | build_status | fingerprint | micro-steps |
|---|---|---|---|
| train | `built` | `f3a7edabc9717954c5a49ec315920b2b16e171c815db0b875ea9a9548ffafde0` | 32 |
| eval | `built` | `fb02c021da96a7e072d48a648d4ee58ba0a637b10b0dd61ef7d21b1932b46d71` | 8 |

Both fingerprints are identical to the ones the M4 receipt above recorded
for this exact same tracked H-short config (expected: same config, same
determinants).

### H-short: full-cycle result (train + scheduled eval + checkpoint)

Config: `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml`
(exact tracked file, unmodified). `COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp_m6_probe/h_short/cache_root`
(dedicated disposable). All 8 ranks exited status 0; `run.json`
`status: "completed"`, `completed_steps: 2`, `runtime.world_size: 8`,
`forward_input_provider_mode: "synchronous"` (shipped default).

Absolute timings (UTC):

| marker | time | derived interval |
|---|---|---|
| launch start | 12:28:08.713 | — |
| writer `created_at` | 12:28:25.862 | pre-loop init (accelerate spawn + model/adapter load + cache manifest validate): 17.149 s |
| step 1 train row written | — | `step_duration_seconds` 18.544 s (compute/optimizer boundary only) |
| step 2 train row written | — | `step_duration_seconds` 16.217 s |
| writer `completed_at` | 12:29:19.740 | writer span (created→completed, full train+eval+checkpoint cycle): 53.878 s |
| process exit | 12:29:32.373 | post-completion teardown: 12.633 s |
| **full process wall time** | | **83.660 s** |
| approx. time-to-first-completed-planned-step (pre-loop init + step 1 duration) | | **35.694 s** |

Eval row (step 1, `trigger_reasons: ["explicit_step"]`): `acc_top1
0.3215420461766575`, `acc_top5 0.49481042152086424`, `example_count 64`,
`pack_count 8`, `loss/total 9.701848484575748`, `count/supervised_atoms
4721.0`, `finite_status: "finite"` — bit-identical to the M4 receipt's own
recorded representative row for this exact fixture (independent
cross-session confirmation).

Reduction-mode confirmation: `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` was unset
in the launch environment (confirmed via `env | grep`), so
`resolve_eval_reduction_control()` returned `auto`; with `pack_count=8 >=
world_size=8` (confirmed by the eval row itself),
`resolve_active_eval_reduction_mode` deterministically resolves to
`disjoint_shard` — the now-default path, exercised as required, not the
`pack_count < world_size` replicated fallback.

Checkpoint: `checkpoints/step-2` (`adapter/`, `special_token_embeddings/`),
`final.json: {"checkpoint_path": "checkpoints/step-2", "step": 2}`. No
`best.json` — this exact tracked config schedules checkpoint at step 2 and
eval at step 1 independently (`checkpoint.steps: [2]`,
`eval.forward.steps: [1]`), and `_checkpoint_handler`'s best-candidate logic
only publishes a best-selector receipt when an eval observation exists for
the *same* step as the checkpoint (`eval_by_step.get(step)`); this is the
exact tracked config's own pre-existing schedule, not a defect introduced
by this measurement. `final.json` is the valid canonical checkpoint
selector for this run.

GPU: peak observed during the run ~52.7 GB (GPU 2, worst case); memory
returned to the external-job-only baseline immediately after exit (byte-identical
per-GPU MiB readings before launch and after exit), confirming clean
release with no leak. Production cache root: 37 directories, mtime
unchanged, both before and after.

### H-steady: bounded-window result (throughput-only)

Launch: `accelerate launch --num_processes 8 -m src.train --config
/tmp/coordexp_m6_probe/h_steady_derived.yaml`, `COORDEXP_SWIFT_PACK_CACHE_ROOT`
unset (default production root). All 8 ranks exited status 0; `run.json`
`status: "completed"`, `completed_steps: 5`, `runtime.world_size: 8`,
`forward_input_provider_mode: "synchronous"`. `materializations.train.semantic_fingerprint`
and `materializations.eval.semantic_fingerprint` in the final `run.json`
are `e68f2699d83705...` / `e9aa216516e72d...` — identical to 6.1a's
production fingerprints, confirming the run itself (not just the
preflight) consumed the shared production caches without rebuilding them.

Absolute timings (UTC):

| marker | time | derived interval |
|---|---|---|
| launch start | 12:31:19.856 | — |
| writer `created_at` | 12:31:36.736 | pre-loop init: 16.880 s |
| writer `completed_at` | 12:36:42.243 | writer span (created→completed, 5 steps + eval + checkpoint): 305.507 s |
| process exit (`run.log` mtime) | 12:36:42.512 | post-completion teardown: 0.269 s |
| **full process wall time (launch → completed_at)** | | **322.387 s (~5 min 22 s)** |

Per-step `step_duration_seconds` (compute/optimizer boundary only,
excludes eval/checkpoint/completed-step handlers per task 1.3 — all 5
steps are valid steady-state samples since only step 5 carries any
eval/checkpoint event per the schedule proof above):

| step | step_duration_seconds | input_build_seconds | acc_top1 |
|---|---|---|---|
| 1 | 18.5796 | 1.2706 | 0.32667 |
| 2 | 17.7455 | 1.2121 | 0.31863 |
| 3 | 17.9859 | 1.3293 | 0.41115 |
| 4 | 17.8257 | 1.1958 | 0.52365 |
| 5 | 17.8988 | 1.2405 | 0.52297 |
| median | **17.899 s** | | |
| mean | **18.007 s** | | |

Steps/hour: **201.13 (median)**, 199.92 (mean). `input_wait_seconds` is
`0.0` on every step (synchronous provider, as expected — no overlap
mechanism is active).

The `writer_span` minus the sum of the 5 `step_duration_seconds` values
(305.507 − 90.036 = 215.472 s) bundles pre-step-1 setup (pack-schedule
resolution, optimizer/scheduler init, DDP wrap), the step-5 eval invocation
(619-pack disjoint-shard eval — much larger than H-short's 8-pack eval),
the checkpoint save (adapter + embeddings), and per-step
completed-step-handler/logging overhead for all 5 steps. This residual is
**not separately instrumented** in this receipt (unlike M4's dedicated
eval-only timing wrapper) and is reported only as an honest aggregate
upper bound, not claimed as an isolated eval-wall-time measurement; M4
above already established the isolated sharded-eval wall-clock number at
8 ranks for the smaller H-short eval shape.

Eval row (step 5, `trigger_reasons: ["every_fraction:1:clamped_final"]`):
`acc_top1 0.5264818325479748`, `acc_top5 0.571475279129729`,
`example_count 4952`, `pack_count 619` (the full, non-sample-limited
production eval set — `data.eval.sample_limit` stays `null`, inherited
unmodified from the production base), `loss/total 6.747127049637129`,
`count/supervised_atoms 348673.0`, `finite_status: "finite"`. Reduction
mode: `COORDEXP_SWIFT_EVAL_REDUCTION_MODE` unset ⇒ `auto`; `pack_count=619
>= world_size=8` ⇒ `disjoint_shard` (deterministic, same confirmation
method as H-short above).

Checkpoint: `checkpoints/step-5` (`adapter/`, `special_token_embeddings/`),
`best.json: {"checkpoint_path": "checkpoints/step-5", "selector":
"acc_top1", "step": 5, "value": 0.5264818325479748}` — value exactly
matches the step-5 eval row's own `acc_top1`, confirming selector
correctness; `final.json: {"checkpoint_path": "checkpoints/step-5", "step":
5}`.

GPU: peak observed during the run ~59.5 GB (GPU 7, worst case, combined
with the external job); memory returned to the external-job-only baseline
immediately after exit. Production cache root: 37 directories, mtime
unchanged, confirmed both before the preflight and after the full launch —
no write, rebuild, or mutation of the shared production caches at any
point.

### M6 disposition

Both required M6 parts are complete and satisfy `measurement-plan.md`: an
absolute, clearly-labeled end-to-end H-short cycle (train + scheduled eval
+ checkpoint, disjoint-shard default confirmed, valid checkpoint selector)
and an absolute bounded H-steady throughput window against the real
production caches (cache-hit proven before model load, uncontaminated
per-step samples, valid best-selector checkpoint). No comparison to M0 is
made (M0 is unavailable by user decision). No production cache mutation,
no OOM, no unrelated-process interference, in either run.

### Cleanup

After this evidence was durably recorded above: `/tmp/coordexp_m6_probe/`
(H-short's disposable cache root and run outputs, the H-steady derived
config and its run outputs, all scratch JSON/log files) deleted in full.
Shared production cache root and the external `8-coords-bbox` job were
never touched by this cleanup (out of scope — different directory tree).
