# Review Triage: streamline-coordexp-swift-base-infrastructure

Status: first Codex lead review (HOLD, fixable planning defects) and the
final localized Codex acceptance pass (P1-7, P1-8, P2-5) received and
resolved in the planning artifacts on 2026-08-03. A later independent
convergence review (tasks 6.5) still gates final acceptance; findings from
that review will be appended below.

## Review scope requested (for the later independent review)

- Delta-spec correctness against the then-current stable specs, including the
  reconcile-and-preserve sync obligation for
  `stream-distributed-pack-cache-runtime` and the scenario-union delta.
- Semantic-preservation claims: byte-identical forward inputs under the
  planned-step-scoped provider, verification preservation under chunk
  skipping, full-row sharded-eval exactness, additive-only artifact schema.
- Gating discipline: every benchmark-dependent slice has a measured
  accept/revert rule; every deletion has a proof task; cache-root discipline
  prevents production-cache churn.

## Findings

| ID | Severity | Area | Finding (lead review, 2026-08-03) | Disposition | Status | Resolution |
| -- | -------- | ---- | --------------------------------- | ----------- | ------ | ---------- |
| P1-1 | P1 | metrics reduction | Top-k weighting by metric `count/supervised_atoms` degenerates to rank mean because that count is globally merged in `prepare_planned_step`/`finalize_planned_step` | accepted | resolved | Reduction now derives from exact rank-local integer sufficient statistics (`top1_correct`, `top5_correct`, local atom count) summed before the ratio; anti-patterns (global-count weighting, rounded-ratio reconstruction) prohibited by spec and tested. See `design.md` Seams C/D, supervision-losses delta, tasks 1.1; same statistics reused for sharded eval. |
| P1-2 | P1 | sharded eval contract | Eval exactness covered only loss/top-k; `token_weighted_diag`, count fields, `example_count`/`pack_count`, global-denominator-derived fields, and finite handling lacked defined reducers | accepted | resolved | Complete durable-scalar inventory with per-field sufficient statistic and reducer added (`design.md` Seam C table); compact explicit eval reduction payload allowed, no generic framework; acceptance covers the entire canonical eval row. See training-artifacts delta, tasks 4.1-4.2. |
| P1-3 | P1 | lookahead lifecycle | Pipeline-global producer walking the full tuple desynchronizes on the trainer's pull-then-consume shape and finite-gate early break, risking ordinal skew/stale work | accepted | resolved | Redesigned as a planned-step-scoped `ForwardInputProvider` protocol (`begin_planned_step`/`take`/`end_planned_step`/`close`) with cancel/discard on every exit path, cancellation-aware put/get (no full-queue deadlock), and trainer-owned step boundaries; "trainer structurally unchanged" claim removed. Early-break scenario added to the packing-forward delta; lifecycle test matrix in tasks 3.1-3.2. |
| P1-4 | P1 | timing semantics | Rank-mean timing is not distributed wall time; measurement boundary and lookahead-mode receipt were undefined | accepted | resolved | `step_duration_seconds` defined as all-rank maximum inside the planned-step compute/optimizer boundary excluding completed-step and scheduled handlers; `input_build_seconds`/`input_wait_seconds` all-rank maximum; existing metric collective reused with no new synchronization; provider mode recorded once in the compact run record. See training-artifacts delta, `design.md` Seam E, tasks 1.3, packing-forward delta receipt clause. |
| P1-5 | P1 | archive ordering / scenario preservation | Prior change's deltas carry fewer scenarios than stable requirements; mechanical archive could erase accepted behavior | accepted | resolved | Task 0.1 is now a blocking reconcile-and-preserve sync obligation naming the exact scenario union; this change's `Deterministic Packing Cache Reuse` delta carries the full union (five stable scenarios + prior-change distributed scenario + new chunk-skip scenario) so archive order cannot erase behavior. See proposal Sequencing Dependency, packing-forward delta. |
| P1-6 | P1 | measurement executability / cache cost | Ellipsis config names; bounded `max_steps` config pretended to exist; risk of repeated multi-hour production-cache rebuilds; contradictory one-rebuild claim across waves 2/5 | accepted | resolved | Exact H-short/H-steady/H-single paths named; bounded window produced via an implementation-time temporary derived config using `extends` (no CLI override exists); M0 uses the existing production cache read-only; all A/B slices use dedicated temporary `COORDEXP_SWIFT_PACK_CACHE_ROOT` roots with sample limits; exactly one production materialization at user-authorized task 6.1a after determinant sources settle; no automatic deletion/mutation of the existing 50 GB cache; contradictory co-scheduling claim removed. See `measurement-plan.md`, tasks 2.3/6.1a, proposal Impact. |
| P2-1 | P2 | process weight | Per-wave dual standards/intent audits are over-audit | accepted | resolved | Review cadence reduced to focused reviews after Waves 3 and 4 plus one final convergence review (6.5); other waves gate on tests/validation/hygiene unless new semantic risk appears. See tasks header. |
| P2-2 | P2 | deletion scope | `_file_sha256`/`_sha256_json` consolidation is speculative cross-domain coupling | accepted | resolved | Consolidation removed; duplicates explicitly kept alongside the `_examples_by_id` copies; deletion scope limited to proven dead helpers, duplicated sync calls, production-dead loss branch after consumer proof, and the empty metrics stub after residue proof. See `design.md` Seam G, tasks 5.2. |
| P2-3 | P2 | cache identity precision | Framing implied only `mtime_ns` removal changes fingerprints | accepted | resolved | Clarified: any `code_identity` source edit changes fingerprints; the final settled source state induces one new production fingerprint; intermediate measurements never churn the shared production cache; existing directories remain historical/disposable and are not auto-deleted. See proposal Impact, `design.md` Seam F. |
| P2-4 | P2 | baseline authority | Drafting commit risked being read as authoritative | accepted | resolved | `a19ffceeb` labeled as inspection provenance only; tasks 0.1 and 5.1 revalidate live HEAD/call graph before implementation. See `design.md` Section 1, proposal Sequencing Dependency. |
| P1-7 | P1 | eval fallback reduction mode | Disjoint-shard sum reducers applied in the replicated fallback would multiply counts by world size; mode was implicit | accepted | resolved | Explicit dual-mode contract added: disjoint-shard mode sums local statistics; replicated fallback keeps current replicated semantics (identical rank values reduced once, never summed); mode carried explicitly in the reduction payload/call path, never inferred; dedicated `pack_count < world_size` fixture asserts the entire row equals the pre-change replicated evaluator with no count multiplication. See `design.md` Seam C "Reduction modes are explicit", training-artifacts delta (new "Replicated fallback keeps replicated reduction" scenario), tasks 4.1/4.2. |
| P1-8 | P1 | M0 cache preflight | M0 could implicitly materialize into the shared production root if the exact-config cache were absent/mismatched | accepted | resolved | Fail-closed read-only preflight added before any M0 launch: derive the resolved-config fingerprint, verify the cache exists, is complete/current, and admits via validation-only checks with no write; otherwise M0 stops reporting "baseline unavailable"; shared root stays read-only until user-authorized 6.1a. See `measurement-plan.md` cache-root discipline + M0, tasks 0.2. |
| P2-5 | P2 | audit wording | Proposal claimed the audit "measured" the efficiency losses; it identified them without running performance measurements | accepted | resolved | Wording changed to "identified", with measurement ownership explicitly left to M0-M6. See `proposal.md` Why section. |

## Disposition rules

- P0/P1 findings block the affected wave gate and the final acceptance gate
  (tasks 6.5) until resolved with evidence or descoped with user approval.
- P2 findings are scheduled inside the owning wave or recorded as accepted
  residual risk in the change notes.
- Findings that contradict the lead-review decisions recorded in
  `proposal.md` are escalated to the user rather than silently re-decided.

## Resolution log

- 2026-08-03: P1-1..P1-6, P2-1..P2-4 from the Codex lead review resolved by
  revising `proposal.md`, `design.md`, `tasks.md`, `measurement-plan.md`, and
  the packing-forward / supervision-losses / training-artifacts deltas.
  Accepted directions preserved unchanged: loaded required chunks remain
  SHA256-verified; `mtime_ns` removed with content SHA retained; FA2 proof
  default `first_micro_step`; no inference rewrite or formula changes;
  benchmark slices keep accept/revert gates; no generic utilities/reduction
  framework.
- 2026-08-03 (later): P1-7, P1-8, P2-5 from the final localized acceptance
  pass resolved by revising `design.md` (explicit reduction modes),
  the training-artifacts delta (mode contract + replicated-fallback
  scenario), `tasks.md` (0.2 preflight, 4.1 fallback fixture, 4.2 explicit
  mode), `measurement-plan.md` (fail-closed M0 preflight), and
  `proposal.md` (identified vs measured wording).
- 2026-08-04: P1-8's M0 preflight ran and found the historical production
  baseline unavailable for both harnesses (all four required caches absent;
  see `implementation-notes.md`). User decision: do not build the missing
  caches; the gap is explicitly accepted, not silently treated as measured.
  Task 0.2 marked complete on the accepted-preflight-receipt basis, not on a
  captured throughput baseline. Consequently `measurement-plan.md` was
  revised so every benchmark-dependent slice's decision-bearing evidence is
  a same-code/same-config/same-temporary-cache paired A/B via internal
  debug/test-only controls (never a public YAML surface), and M6 (task
  6.1b) now reports absolute end-to-end smoke timings instead of an
  M0-comparison table. This does not reopen or alter P1-6/P1-8's original
  resolutions (cache-root discipline, fail-closed preflight contract); it
  narrows what M0 is used for. No P0/P1 finding is newly outstanding from
  this change.

## Independent implementation review (Opus, 2026-08-04) — Waves 1/2 — verdict HOLD

Scope: the Wave 1/2 source implementation (not the planning artifacts already
resolved above). Verdict HOLD on two P1 findings plus six cheap P2 findings,
all fixed in this same worktree the same day; no P0 findings.

| ID | Severity | Area | Finding | Disposition | Status | Resolution |
| -- | -------- | ---- | ------- | ----------- | ------ | ---------- |
| P1-A | P1 | spec coherence | `coordexp-swift-pack-cache-semantic-identity`'s delta spec only ADDED the new "Rank-Selective Chunk Consumption Preserves Verification" requirement; it left the stable `Cache Payload Is Current-Version-Only` requirement's text ("the eager rank load MUST verify every digest and decoded payload exactly once before forward"; "the eager rank load MUST reject the changed payload") unmodified and therefore contradicting the new chunk-skip behavior once archived | accepted | resolved | Added a complete `## MODIFIED Requirements` entry for `Cache Payload Is Current-Version-Only` in the delta spec, preserving the full stable scenario union (5 scenarios) and qualifying the two affected ones: the eager-rank-load scenario now scopes "every digest and decoded payload" to the rank's resolved-schedule required set; the payload-changes scenario is split into a required-chunk case (unchanged: fail closed) and a new non-required-chunk case (may skip, matching the ADDED requirement). `proposal.md`'s Sequencing Dependency paragraph updated to name both MODIFIED deltas carrying the full union. `openspec validate --strict` and `--all --strict` both pass. |
| P1-B | P1 | metrics reduction fail-open | `TrainRuntime._reduce_metric_reports` only used the exact summed-integer reduction for `acc_top1`/`acc_top5` when the LOCAL rank happened to supply `accuracy_stats`; if the local caller omitted them (a future bug, or any as-yet-unwired caller) while `acc_top1`/`acc_top5` were still present in the metric payload, it silently fell back to the plain rank mean — the exact pre-Wave-1 defect, now reachable through a code path with no explicit test forcing the fail branch | accepted | resolved | `_reduce_metric_reports` now raises `RuntimeContractError` (`runtime.accuracy_stats_missing`) whenever `acc_top1`/`acc_top5` are present in the gathered payload and the local rank omitted `accuracy_stats`, before ever reaching a reducer branch; the existing peer-missing check is preserved. `eval.forward` (`src/eval/forward.py`) — previously the one caller that gathered accuracy keys without stats — now forwards real `accuracy_stats` from the loss bundle/artifact via a new `_accuracy_stats_from` helper, so no production caller is left exempt. Callers gathering metric sets with no accuracy keys are unaffected (`test_multirank_metrics_without_accuracy_keys_never_require_accuracy_stats`). Added: a pipeline-level production-wiring test building a real `LossRunner.finalize_planned_step` artifact and asserting it is forwarded through `_train_logging_handler` into `runtime.gather_metrics` without leaking into the durable row (`test_train_logging_forwards_real_loss_runner_accuracy_stats_without_leaking_to_row`); asymmetric missing-stats tests (local-missing/peer-present, peer-missing/local-present, neither-present) and the exact-pooled-result test already present from Wave 1. |
| P2-C | P2 | test strength | The boundary-exclusion timing test only proved exclusion via a finite tick iterator raising `StopIteration` on any stray call — a real signal, but it never demonstrated what a *wrong* boundary placement would numerically look like | accepted | resolved | Rewrote the test with an incrementing fake clock consumed both by the trainer's own boundary timing AND by the completed-step/scheduled handlers (each explicitly calls `time.monotonic()` once). The correct boundary yields a fixed, explicitly asserted `1.0`-tick duration per step regardless of intervening handler ticks; the test also asserts the duration does NOT equal the value a boundary-inclusive-of-handlers bug would produce (`3.0`), making the wrong placement a directly falsifiable numeric claim, not just a crash. |
| P2-D | P2 | receipt accuracy | `TrainRuntime.gather_metrics`'s `"reduction"` receipt field reported the literal string `"all_rank_mean"` for every multi-rank call, which became factually false once per-key reducers diverged (max for timing, summed-ratio for accuracy, mean for everything else) | accepted | resolved | Renamed to `"all_rank_mixed"` in `src/runtime/train_runtime.py`; updated the one asserting test. |
| P2-E | P2 | input validation | `accuracy_stats` fields were validated as non-negative integers but never cross-validated against each other; a malformed `top1_correct`/`top5_correct` exceeding `accuracy_atom_count` (locally or from a peer) would silently pollute the summed ratio with an impossible >1.0 contribution | accepted | resolved | Added `_check_correct_within_atom_count` in `src/runtime/train_runtime.py`, called from both `_checked_accuracy_stats` (local) and `_reduce_accuracy_metric` (per-peer, before summing) — raises `runtime.accuracy_stats_correct_exceeds_atoms`. Tests for single-rank, local-exceeds, and peer-exceeds cases. |
| P2-F | P2 | dead fallback | `_apply_fa2_branch_proof_policy` read `getattr(config.model, "fa2_branch_proof", "every_forward")` — a stale defensive fallback to the OLD pre-Wave-1 default, unreachable in practice (the validated `ModelConfig` schema always populates the field) but misleading, since the string no longer matches the current schema default | accepted | resolved | Changed to direct attribute access `config.model.fa2_branch_proof` in `src/training/pipeline.py`; confirmed both direct-caller test fixtures already set the field explicitly. |
| P2-G | P2 | fabricated timing | `CompletedStepObservation`'s timing fields defaulted to `0.0`; the production-dead non-streaming batch trainer path never set them, so its rows would report a *fabricated* zero-duration measurement rather than "not measured" | accepted | resolved | Changed the three fields to `float \| None = None` in `src/training/supervised_trainer.py` (the streaming/production path is unaffected — it always sets real measured floats); `pipeline._train_logging_handler` now only adds a timing field to the gathered metrics when the observation's value is not `None`, so an unmeasured path's durable row omits the fields entirely (additive-schema-consistent) instead of asserting a false zero. Added `test_non_streaming_batch_path_leaves_timing_fields_unmeasured`. No new machinery added to the dead path; Wave-5 deletion task is unaffected. |
| P2-H | P2 | undercounting | `LossRunner._merge_accuracy_stats` used `stats.get(field, 0)` per micro-step artifact, so a micro-step artifact missing (or malformed) `accuracy_stats` would silently contribute 0 rather than failing — undercounting the planned-step total without any signal | accepted | resolved | Rewrote to require `accuracy_stats` as a `Mapping` on every artifact and validate each of the three fields as a non-negative integer via a new `_checked_micro_accuracy_stat` helper, plus the same correct-exceeds-atoms cross-check as P2-E, all raising `LossContractError` on violation (`loss.accuracy_stats_missing`, `loss.accuracy_stats_field_type`, `loss.accuracy_stats_field_missing`, `loss.accuracy_stats_correct_exceeds_atoms`). Confirmed no test or production caller ever constructs a `finalize_planned_step` artifact by hand without `accuracy_stats` — every call site uses `LossBundle.to_artifact_dict()` from a real `compute_micro_step` bundle. |

### Files changed for this review pass

`openspec/changes/streamline-coordexp-swift-base-infrastructure/{specs/coordexp-swift-pack-cache-semantic-identity/spec.md,proposal.md,tasks.md,review-triage.md}`,
`src/eval/forward.py`, `src/losses/runner.py`, `src/runtime/train_runtime.py`,
`src/training/pipeline.py`, `src/training/supervised_trainer.py`,
`tests/eval/test_forward_eval.py`, `tests/losses/test_runner.py`,
`tests/runtime/test_train_runtime.py`, `tests/training/test_pipeline_assembly.py`,
`tests/training/test_supervised_trainer.py`.

### Re-verification after fixes (2026-08-04)

- Targeted: `tests/losses/test_runner.py tests/runtime/ tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/training/test_pack_cache.py tests/training/test_pipeline_pack_cache_rebuild.py tests/config/ tests/eval/ tests/qwen/ tests/artifacts/` — 422 passed.
- Broad: `tests/runtime/ tests/training/ tests/losses/ tests/config/ tests/artifacts/ tests/eval/ tests/qwen/` — 453 passed.
- `ruff check` on every touched file (this pass plus original Wave 1/2 files): 3 findings, all confirmed pre-existing and present at the unmodified HEAD baseline (`RankGradientFiniteReport`/`Iterator` unused imports, one unused `runtime` local in a test) — none introduced by this change.
- `git diff --check`: clean.
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict` and `openspec validate --all --strict`: 19/19 pass.
- Residue greps: `all_rank_mean` (none), `getattr(config.model, "fa2_branch_proof"` (none), `mtime_ns` in packing determinants (none), `_merge_accuracy_metric` (none).
- Shared production pack-cache root (`.cache/coordexp_swift/packing`): unchanged — 35 fingerprint directories, root mtime unchanged (predates this session); no cache build or GPU work was needed for these fixes, so the M2 measurement result recorded in `implementation-notes.md` is untouched and still authoritative.
- No commit/push/GPU launch performed.

## Independent implementation review (Opus, 2026-08-04) — Wave 3 — verdict HOLD

Scope: the Wave 3 source implementation (`ForwardInputProvider` protocol and
both concrete providers, trainer wiring, pipeline wiring, run-record
receipt) plus the M3 measurement methodology and its reported result.
Verdict HOLD on two P1 findings (one a genuine correctness gap in the
depth-one bound, one a test-quality gap) plus eight P2 findings, all fixed
in this same worktree the same day; no P0 findings.

| ID | Severity | Area | Finding | Disposition | Status | Resolution |
| -- | -------- | ---- | ------- | ----------- | ------ | ---------- |
| P1-1 | P1 | depth-one bound (correctness) | `queue.Queue(maxsize=1)` alone does not enforce "at most one prepared micro-step beyond the executing one": a fast producer can finish building ordinal k+1 (blocking only on `put()`) while ordinal k still sits in the queue unconsumed, momentarily holding two fully-built items in host memory — violating the design's bounded-memory invariant | accepted | resolved | Added a step-scoped `threading.Semaphore(1)` build slot: the producer acquires it (cancellation-aware, `_cancellation_aware_acquire`) before starting each build; the consumer releases it immediately after dequeuing the previous item, before the H2D move. The producer can now never start building ordinal k+1 until ordinal k has actually been taken. Cancellation/end/full-queue paths remain deadlock-free — a producer blocked on the semaphore acquire wakes on the same step-scoped `cancel_event` within one poll interval, same as the existing queue put/get. See `tasks.md` 3.1, `src/training/forward_input_provider.py`. |
| P1-2 | P1 | test quality | The (g) "at most one prepared item" test only asserted `queue.maxsize == 1` and `qsize() <= 1` — true by construction of `queue.Queue(maxsize=1)` regardless of whether the depth-one bound is actually enforced by build ordering (i.e. tautological; would not have caught P1-1) | accepted | resolved | Replaced with two decisive tests: `test_overlapped_provider_gates_next_build_until_prior_item_is_taken` (asserts ordinal 1's build has not started after a bounded 0.3s wait with no `take()` call yet — manually verified this fails against a scratch copy of the pre-fix source, confirming decisiveness) and `test_overlapped_provider_built_minus_taken_never_exceeds_one` (continuous invariant `builds_completed - items_dequeued <= 1` sampled from a background watcher thread across a full 5-micro-step step with no artificial delay, using the internal dequeue event rather than the caller's post-H2D-move return to avoid a false-positive race). See `tasks.md` 3.2. |
| P2-A | P2 | timing comparability | `SynchronousForwardInputProvider` built fused CPU+device inputs in one call, so its `total_build_inputs_ns` receipt included H2D time; `OverlappedForwardInputProvider` built CPU-only. `input_build_seconds` therefore meant different things in the two modes, making the M3 comparison apples-to-oranges | accepted | resolved | `SynchronousForwardInputProvider.take()` now also builds CPU-only (`device=None`) then calls `_move_forward_inputs_to_device` separately, matching the overlapped provider's two-phase construction; both modes' `total_build_inputs_ns` now mean CPU-only build time. H2D remains real time inside the caller's `step_duration_seconds`, documented explicitly in the module docstring as excluded from both `input_build_seconds` and `input_wait_seconds`. New test: `test_both_provider_modes_build_on_cpu_and_move_separately_for_comparable_timing`. |
| P2-B | P2 | silent misuse / duplication | A custom `qwen_forward` combined with `forward_input_provider` was silently ignored (the provider path calls `run_qwen_forward` directly); the logits-to-keep helper was also duplicated between `supervised_trainer.py` and `forward_input_provider.py`, a drift risk | accepted | resolved | `SupervisedTrainer.__init__` now raises `RuntimeContractError` (`trainer.forward_input_provider_conflicts_with_custom_qwen_forward`) if both are passed. `forward_input_provider.py` now imports `supervised_trainer._logits_positions_to_keep` directly instead of duplicating it (no circular import: the trainer only references the provider module under `TYPE_CHECKING`). New test: `test_forward_input_provider_rejects_combination_with_custom_qwen_forward`. |
| P2-C | P2 | device-move coverage | No test proved the H2D move actually reached every tensor field `to_model_kwargs()` sends to the model (input_ids/position_ids/pixel_values/image_grid_thw/fa2 cu_seq_lens_q,k) — a regression silently dropping one field's `.to(device)` call would not have been caught | accepted | resolved | Added `test_move_to_device_covers_every_tensor_field_used_by_to_model_kwargs`, using `torch.device("meta")` (no real CUDA needed) to assert every tensor field's device actually changed; manually verified decisive by reverting one field's move in a scratch copy and confirming the test fails naming that exact field. |
| P2-D | P2 | timing boundary | `step_duration_seconds` was captured after the `finally` block that calls `provider.end_planned_step()`, so provider teardown (cancellation, drain, bounded join) was silently included in the measured compute/optimizer boundary | accepted | resolved | Capture point moved inside the `try` block, immediately after `zero_gradients()` and before the `finally`. New test: `test_streaming_planned_step_duration_excludes_provider_teardown` (incrementing-fake-clock proof with a teardown-consuming test double, mirroring the existing scheduled-handler-exclusion test's technique). |
| P2-E | P2 | protocol completeness | `ForwardInputProvider` (the `Protocol`) declared only the four lifecycle methods; the trainer also reads `provider.last_take_wait_seconds`, an attribute the protocol never declared | accepted | resolved | Added `last_take_wait_seconds: float` to the `Protocol` body. |
| P2-F | P2 | cleanup ordering | `forward_input_provider.close()` and the rank-report gatherer's `close()` were both plain statements in the same `finally` block; a `forward_input_provider.close()` failure (e.g. the bounded-join timeout) would skip the rank-report gatherer close entirely | accepted | resolved | Nested `try/finally` in `src/training/pipeline.py`: `try: forward_input_provider.close() finally: <close rank-report gatherer>`. |
| P2-G | P2 | false run receipt | The run record always bound a `forward_input_provider_mode`, even in the hypothetical case where the paired loss runner was non-streaming (production-dead batch path, which never consults the provider) — the record would then claim a mode that was silently never used | accepted | resolved | `SupervisedTrainer.__init__` now raises `RuntimeContractError` (`trainer.forward_input_provider_requires_streaming_loss_runner`) if `forward_input_provider` is paired with a non-streaming loss runner — no new "unused" mode value was added, per the review's explicit instruction; the combination is rejected outright. `_run_initialized_training` also moved the `writer.bind_forward_input_provider_mode(...)` call to after `SupervisedTrainer(...)` construction succeeds, so the run record never binds a mode for a construction that failed this check. New test: `test_forward_input_provider_rejects_non_streaming_loss_runner`. **[Superseded 2026-08-04, Wave 5 task 5.1]**: `trainer.forward_input_provider_requires_streaming_loss_runner` was deleted outright once Wave 5 added `SupervisedTrainer.__init__`'s unconditional `trainer.loss_runner_requires_streaming_protocol` check (raised for *every* non-streaming `loss_runner`, provider or not) — this P2-G code was a strict subset of that broader check and became unreachable once it existed, so it was removed rather than left dead. See `tasks.md` 5.1 DONE note and the residue grep confirming zero remaining references. |
| P2-H | P2 | measurement honesty | The M3 result was reported as "median steps/hour 342.6 sync vs 345.4 overlapped, +0.8%", which reads as an established effect size at n=3 repetitions — not statistically supported by that sample size | accepted | resolved, escalated by re-measurement, then stop rule applied | Initially restated as "directional, not established" (per-repetition range +0.35% to +1.77%, still every repetition favoring overlapped). While implementing this fix, also re-ran the same paired A/B against the now-fixed code (P1-1 depth-one gate + P2-A CPU-only synchronous build) with 5 repetitions instead of 3, since GPUs had gone fully idle: **the original win did not replicate** — only 2/5 repetitions favored overlapped, median steps/hour difference +0.27% (noise-level parity). Both sessions' full data are recorded side by side in `implementation-notes.md` "M3". Final disposition: this world_size=1 evidence does not establish the "steps/hour improves" condition task 3.4's own stop rule requires, so the stop rule was applied literally — **overlap rejected as the shipped default; `synchronous` now ships as the default** (`resolve_forward_input_provider_mode()` returns `synchronous` when unset), implementation and tests kept, recorded rejected-with-evidence per the stop rule's own wording. `overlapped` remains fully implemented and selectable via `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE=overlapped`. Exact 8-rank confirmation is the only path to promoting `overlapped` back to the default. Also recorded as a P2 for that future 8-rank M3: compare three arms (legacy fused `forward_input_provider=None`, two-phase synchronous, overlapped) rather than two, since P2-A's fix made the synchronous provider two-phase (no longer the original single-call fused build) — no third-arm benchmark built now. See `tasks.md` 3.4. |

### Files changed for this review pass

`openspec/changes/streamline-coordexp-swift-base-infrastructure/{tasks.md,review-triage.md,implementation-notes.md,specs/coordexp-swift-packing-forward/spec.md}`,
`src/training/forward_input_provider.py`, `src/training/supervised_trainer.py`,
`src/training/pipeline.py`,
`tests/training/test_forward_input_provider.py`,
`tests/training/test_supervised_trainer.py`,
`tests/training/test_pipeline_assembly.py`.

### Re-verification after fixes (2026-08-04)

- Targeted: `tests/training/test_forward_input_provider.py tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/artifacts/test_run_artifacts.py` — 86 passed.
- Broad: `tests/runtime/ tests/training/ tests/losses/ tests/config/ tests/artifacts/ tests/eval/ tests/qwen/` — 481 passed (up from 475: net +6 — two decisive depth-one tests replacing one tautological test, plus five new P2-coverage tests, minus none removed elsewhere).
- Decisiveness manually verified for both P1-2 replacement tests and the P2-C coverage test by temporarily reverting the corresponding fix in a scratch copy of the (untracked) source file and re-running the specific test: `test_overlapped_provider_gates_next_build_until_prior_item_is_taken` fails without the build-slot gate (`build_started == [0, 1]` within the no-`take()` wait window); `test_move_to_device_covers_every_tensor_field_used_by_to_model_kwargs` fails without one field's device move (names the exact field). Both reverted back to the fixed source before continuing.
- `ruff check` on every touched file: 1 finding, confirmed pre-existing at the unmodified HEAD baseline (`Iterator` unused import in `supervised_trainer.py`) — not introduced by this pass.
- `git diff --check`: clean.
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict` and `openspec validate --all --strict`: 19/19 pass.
- Shared production pack-cache root (`.cache/coordexp_swift/packing`): unchanged — 35 fingerprint directories, root mtime unchanged before/after this whole pass (predates this session). GPU work WAS performed for this pass, specifically for P2-H: a 5-repetition re-measurement of the M3 A/B against the fixed code, using only GPU 0 (confirmed idle, along with all other GPUs, before starting) and a dedicated temporary `COORDEXP_SWIFT_PACK_CACHE_ROOT` (deleted after the measurement, never touching the shared production root). All other P1/P2 code fixes required no cache build or GPU work.
- No commit/push performed. GPU launch WAS performed (see above), scoped to
  GPU 0 only with all other GPUs confirmed idle beforehand, and only for the
  P2-H re-measurement.

### Disposition: tasks 3.2 and 3.4 checked, final narrow re-review still pending

Tasks 3.1-3.4 in `tasks.md` are all checked: 3.1/3.3's implementation and
wiring are unaffected by the additive guards/ordering fixes; 3.2's test
suite is complete and green including the P1-2 decisive replacements; 3.4's
stop rule was applied literally once the world_size=1 re-measurement did
not establish an improvement — overlap is rejected as the shipped default,
`synchronous` ships instead, the implementation and tests are kept, and
this is recorded as rejected-with-evidence per the stop rule's own wording.
That stop-rule outcome is itself a complete, honestly-described conclusion,
not an open question — so it no longer blocks the checkboxes the way the
earlier (overstated) "ACCEPT" framing would have.
This Opus pass is itself the "focused independent review of the concurrency
lifecycle" named by task 3.4's own text — its HOLD-then-fixed-same-day
outcome is recorded here. A **narrow independent re-review specifically
confirming the P1-1 depth-one fix and this final stop-rule disposition is
still recorded as pending** — recommended before this change is folded into
a final convergence review (task 6.5), but not gating 3.2/3.4's checkbox
state, consistent with how the Waves 1/2 Opus review above was resolved
same-day but is still recorded as its own dated section rather than
silently folded into the original DONE text.

## Wave 4 (sharded eval.forward) — independent review pending 2026-08-04

Tasks 4.1-4.3 (disjoint pack sharding, the compact eval reduction
payload/reducers, checkpoint-selector/row-schema preservation) are
implemented and tested per `tasks.md`; full-row exactness against the
replicated evaluator is proven on a decisive fixture matrix using the real
`LossRunner`/`TrainRuntime` production reduction code (see `tasks.md` 4.1,
`implementation-notes.md` "M4"). `replicated` remains the shipped default;
`disjoint_shard` is fully implemented and selectable via the internal
`COORDEXP_SWIFT_EVAL_REDUCTION_MODE` control, not promoted without measured
8-rank evidence (GPUs were fully occupied by unrelated external work at
every check this session — see `implementation-notes.md` "M4" for the
before/after `nvidia-smi` receipts).

**Superseded 2026-08-04 after M4 ACCEPT:** the exact 8-rank A/B later passed
with full-row and checkpoint-selector equivalence and a 7.27x median eval
speedup. The unset internal control now resolves to `auto`, making
`disjoint_shard` the shipped multi-rank default when structural fallbacks do
not apply. The paragraph above remains only as the dated pre-M4 review state.

The **"focused independent review of the distributed reduction"** named by
task 4.4 had **not been performed** as of this paragraph's original writing
— this implementation pass was itself scoped to tasks 4.1-4.4 only
(implement-only delegation, no separate reviewer role), so no independent
reviewer had yet examined the new `TrainRuntime` sum/identical reducer
additions, the `denominator_gatherer` reuse in `eval/forward.py`, or the
evidence-gated default-mode control, the way the Opus passes above examined
Waves 1/2 and Wave 3. **UPDATE 2026-08-04 (later session):** that review has
now been performed by a separate agent lane (see "Independent implementation
review (2026-08-04) — Wave 4 distributed reduction" below, after the
original HOLD section) — verdict HOLD on one P1 finding, fixed the same
session, converged **APPROVE**. It does not block tasks 4.1-4.3's checkbox
state (their own gates — tests, hygiene, `openspec validate` — are
independently complete and green) and no longer blocks task 4.4 on the
review-availability ground; task 4.4 remains unchecked only for the
still-outstanding 8-rank M4 measurement and checkpoint-selector replay (see
that section's Disposition).

## Independent implementation review (Opus, 2026-08-04) — Wave 4 — verdict HOLD

Scope: the Wave 4 rank-sharded `eval.forward` implementation delivered
above (`src/eval/forward.py`, `src/runtime/train_runtime.py`,
`src/training/pipeline.py`) and its equivalence fixture matrix. Verdict
HOLD on two P1 findings plus five P2 findings, all fixed in this same
worktree the same day; no P0 findings. Fixed by the same writer lane that
delivered the original Wave 4 implementation (no separate reviewer role was
available for this fix pass, same constraint recorded above for the
original "focused independent review" item).

| ID | Severity | Area | Finding | Disposition | Status | Resolution |
| -- | -------- | ---- | ------- | ----------- | ------ | ---------- |
| P1-1 | P1 | finite/* reduction correctness | `finite/*` keys fell through to the default plain-mean reducer in `disjoint_shard` mode: one rank finite (1.0) and one non-finite (0.0) reduced to 0.5, a meaningless fractional value, instead of the replicated evaluator's correct 0.0 (any non-finite contribution makes the derived global scalar non-finite) | accepted | resolved | Added `TrainRuntime._reduce_eval_finite_metric` (logical AND via `min` over the 0.0/1.0 per-rank flags) for every `finite/*`-prefixed key, gated only by `reduction_mode == "disjoint_shard"` (the default/replicated path is untouched, byte-identical to before). `_checked_eval_finite_flag` fails closed on any value other than exactly 0.0/1.0 (including NaN, since `nan not in (0.0, 1.0)` is `True`). See `src/runtime/train_runtime.py`. |
| P1-2 | P1 | test coverage | No test exercised a genuinely non-finite per-rank contribution under sharding; the equivalence matrix only covered the all-finite case, so the P1-1 defect (and its fix) had no decisive test | accepted | resolved | `test_disjoint_shard_eval_nonfinite_shard_reduces_finite_flags_by_and_not_mean` (`tests/eval/test_forward_eval.py`) drives a real `LossRunner`/`TrainRuntime` two-rank run where rank 1's entire shard (pack 1) has a genuinely non-finite loss (NaN logits row feeding real `BaseTokenCE`/`TokenTypeGateLoss` cross-entropy, not an injected fake scalar); asserts every `finite/*` field and the real `RunWriter.append_logging_row` writer-facing null/`non_finite_fields` normalization match the world_size=1 replicated reference exactly. Manually verified decisive: reverting the P1-1 fix in a scratch copy of the source and re-running this test (plus the narrower `tests/runtime/test_train_runtime.py::test_eval_disjoint_shard_finite_keys_reduce_by_and_not_mean`) reproduces the exact `0.5 == 0.0` failure the fix eliminates, then both were re-verified green against the restored fix. |
| P2-A | P2 | sharding key correctness | `partition_eval_micro_steps_for_rank` sharded by `micro_step.pack.pack_index % world_size` -- `pack_index` is an identity/label assigned at pack-planning time, not a guaranteed contiguous 0..N-1 restatement of a micro-step's position in whatever sequence the function is given; using its VALUE for the modulo could silently produce a non-canonical partition for a non-contiguous or offset `pack_index` sequence | accepted | resolved | Sharding key changed to `enumerate(micro_steps)`'s position (`sequence_ordinal % world_size == rank`); pack identity is still validated fail-closed (`_validate_pack_identity`, renamed from `_pack_ordinal` since its return value is no longer used as the ordinal) but plays no role in the partition itself. `test_partition_eval_micro_steps_for_rank_uses_sequence_position_not_pack_index_value` uses deliberately non-contiguous pack_index values (7, 8, 5) chosen so the old pack_index-value scheme and the new position scheme produce opposite splits, making the fixture decisive between them; `test_partition_eval_micro_steps_for_rank_still_validates_pack_identity_fail_closed` proves identity validation survived the refactor. See `src/eval/forward.py`. |
| P2-B | P2 | cross-rank liveness | The `reduction_mode` identity check added in the original Wave 4 pass only fired inside `TrainRuntime._reduce_metric_reports`, reached from the eval-invocation metric gather -- but the `disjoint_shard` path's `gather_loss_denominators` collective runs BEFORE that, and only a rank that resolved `disjoint_shard` locally ever calls it. A genuine mode divergence across ranks (config drift, a resolution bug) would leave a `replicated` peer waiting forever on a collective it never joins, discovered only via a hang, not a fail-closed error | accepted | resolved | Added `TrainRuntime.validate_eval_reduction_consensus(reduction_mode, pack_count)`, reusing the exact same bounded rank-report gatherer as every other collective (no new framework, no public knob). `_run_initialized_training` (`src/training/pipeline.py`) calls it unconditionally on every rank -- regardless of each rank's own locally-resolved mode -- immediately after computing `eval_reduction_mode`/`eval_pack_count_for_consensus` and strictly before the sharding filter (`partition_eval_micro_steps_for_rank`) or any handler construction. Fails closed on mode disagreement (`runtime.eval_reduction_consensus_mode_mismatch`), pack_count disagreement (`runtime.eval_reduction_consensus_pack_count_mismatch`), or malformed/incomplete rank coverage. The eval-cache-is-None fallback branch (no canonical cross-rank pack_count) passes `pack_count=None` uniformly across ranks, which the consensus check accepts without requiring a real integer. Tested against the real `TrainRuntime` gather path: `test_eval_reduction_consensus_passes_when_ranks_agree`, `test_eval_reduction_consensus_allows_matching_none_pack_count`, `test_eval_reduction_consensus_fails_closed_on_mode_mismatch`, `test_eval_reduction_consensus_fails_closed_on_pack_count_mismatch`, `test_eval_reduction_consensus_is_noop_at_world_size_one` (`tests/runtime/test_train_runtime.py`); pipeline-level wiring proven by `test_same_dataset_eval_resolves_distinct_full_cache_and_binding`'s new `consensus_calls` assertion (`tests/training/test_pipeline_assembly.py`). |
| P2-C | P2 | process | Task 4.4 was checked in the original Wave 4 pass despite the named "focused independent review of the distributed reduction" never having been performed | accepted | resolved | `tasks.md` 4.4 stays unchecked; this HOLD and its fixes are recorded here and in `implementation-notes.md`. 4.1 remains checked now that the nonfinite-parity test (P1-2) is green; 4.2/4.3 remain checked (their own claims -- the reducer/payload design and row-schema/selector preservation -- are unaffected by the P1-1 finite/* bug, which was a reducer-classification gap, not a structural defect in those tasks' own scope). |
| P2-D | P2 | speculative code | `_prepare_disjoint_shard_scalars`'s `local_value is None` handling for a term's `token_weighted_diag` value is unreachable under the current `LossRunner` (`_compute_token_term_contribution` always returns a real float for `token_weighted_diagnostic`, never `None`) | accepted, no action | resolved | Assessed and confirmed unreachable; left as-is (harmless defensive code, not exercised by any test, not worth removing or elaborating). No speculative machinery added around it, per instruction. See `src/eval/forward.py::_prepare_disjoint_shard_scalars`. |
| P2-E | P2 | misleading docs | Several comments/docstrings/error codes referred to `pack_index` as "the ordinal" even before P2-A's fix, and the design doc's Seam C text used `pack_ordinal % world_size` | accepted | resolved | Updated `src/eval/forward.py`'s module-level comment, `partition_eval_micro_steps_for_rank`'s docstring, and the error code (`eval_forward.pack_ordinal_missing` -> `eval_forward.pack_identity_missing`, not asserted by any test at the time of rename); updated `design.md` Seam C's sharding paragraph to name `sequence_ordinal` and explicitly distinguish it from `pack_index`. |

### Files changed for this review pass

`src/eval/forward.py`, `src/runtime/train_runtime.py`,
`src/training/pipeline.py`,
`tests/eval/test_forward_eval.py`, `tests/runtime/test_train_runtime.py`,
`tests/training/test_pipeline_assembly.py`,
`openspec/changes/streamline-coordexp-swift-base-infrastructure/{design.md,tasks.md,review-triage.md,implementation-notes.md}`.

### Re-verification after fixes (2026-08-04)

- Targeted: `tests/eval/test_forward_eval.py tests/runtime/test_train_runtime.py tests/training/test_pipeline_assembly.py tests/losses/test_runner.py` -- 101 passed.
- Broad: `tests/runtime/ tests/training/ tests/losses/ tests/config/ tests/artifacts/ tests/eval/ tests/qwen/` -- 509 passed, 1 skipped (CUDA-gated: `tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad` is collected but not exercised without a visible GPU; **corrected 2026-08-04 per Wave 5 review P2** -- this session's original text said "510 passed", counting that CUDA test as passed rather than noting the skip; up from 499 before this fix pass; net +11 new tests: 3 for P1-1's finite/AND reducer, 1 decisive end-to-end nonfinite fixture for P1-2, 2 for P2-A's ordinal fix, 5 for P2-B's consensus collective).
- `ruff check` on every touched file: 2 findings, both confirmed pre-existing at the unmodified HEAD baseline (`RankGradientFiniteReport` unused import in `train_runtime.py`, one unused `runtime` local in an unrelated `test_train_runtime.py` test) -- neither introduced by this pass.
- `git diff --check`: clean.
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict` and `openspec validate --all --strict`: 19/19 pass.
- No production pack-cache root, commit, push, or GPU work performed for this fix pass.
- Decisiveness manually verified for P1-1/P1-2: reverted the `_reduce_eval_finite_metric` dispatch branch in a scratch copy of `src/runtime/train_runtime.py`, re-ran `test_disjoint_shard_eval_nonfinite_shard_reduces_finite_flags_by_and_not_mean` and `test_eval_disjoint_shard_finite_keys_reduce_by_and_not_mean` -- both failed with the exact predicted `0.5 == 0.0` mismatch -- then restored the fix from a pre-edit backup and re-ran both green.
- No production pack-cache root or GPU work touched by this fix pass (all new tests are CPU-only, using the real `LossRunner`/`TrainRuntime` production code with a thread-simulated collective, not `torch.distributed`).
- No commit/push performed.

### Disposition: tasks 4.1-4.3 checked; 4.4 remains unchecked

4.1's fixture matrix now includes the decisive non-finite case (P1-2), so it
stays checked. 4.2/4.3 are unaffected by the P1-1 finite/* bug (a
reducer-classification gap orthogonal to the sharding/payload design and
row-schema/selector claims those tasks make) and remain checked. 4.4 stays
unchecked: the exact 8-rank M4 performance measurement is still unavailable
(see `implementation-notes.md` "M4"), and, at the time this paragraph was
written, the "focused independent review of the distributed reduction" it
names had still not been performed by a separate reviewer role -- this fix
pass closed the Opus-identified HOLD findings but was not itself that
independent review. **UPDATE 2026-08-04 (later session):** that separate
independent review has since been performed and converged APPROVE after
fixing one further P1 finding it found (a full-row-exactness gap this same
fixture matrix had not covered) -- see "Independent implementation review
(2026-08-04) — Wave 4 distributed reduction" below. 4.4 remains unchecked
purely for the still-outstanding 8-rank M4 measurement and
checkpoint-selector replay, not for lack of independent review.

## Independent implementation review (2026-08-04) — Wave 4 distributed reduction — verdict HOLD, fixed same session, converged APPROVE

This is the "focused independent review of the distributed reduction"
task 4.4 names and the P2-C disposition above records as outstanding: a
separate reviewing pass over the current (post-HOLD-fix) Wave 4 code by an
agent lane that did not author the Wave 4 implementation or the prior
same-day HOLD fixes in this session. Reviewer identity, for evidentiary
accuracy: this pass was performed by Claude (Sonnet 5) under a bounded
delegated task from Codex, not literally the Opus model used for the
earlier same-day passes above — recorded exactly to avoid misattributing
authorship in this document.

Scope: `src/eval/forward.py`'s `partition_eval_micro_steps_for_rank`,
`resolve_active_eval_reduction_mode`, `ForwardEvalRunner`/
`_run_streaming_forward_only`, `_prepare_disjoint_shard_scalars`/
`_finalize_disjoint_shard_scalars`; `src/runtime/train_runtime.py`'s
`_reduce_eval_sum_metric`/`_reduce_eval_identical_metric`/
`_reduce_eval_finite_metric`/`validate_eval_reduction_consensus` and their
key-classification helpers; `src/training/pipeline.py`'s wiring order
(consensus call strictly before the sharding filter and handler
construction, confirmed by direct read). Method: read the current code
fresh (not the prior HOLD's diff/prose) and traced every reducer's
numeric path end to end against the exact replicated-path computation
(`_weighted_average`, `src/losses/runner.py`) it is required to reproduce
byte-for-byte, per task 4.1's own full-row-exactness requirement.

| ID | Severity | Area | Finding | Disposition | Status | Resolution |
| -- | -------- | ---- | ------- | ----------- | ------ | ---------- |
| P1-1 | P1 | full-row exactness (correctness) | A configured term with a globally-zero selected-token count across the ENTIRE disjoint-sharded eval set (e.g. an optional coordinate term on a text-only eval split, or any small eval sample that happens to select zero tokens for that term on every rank) crashed `disjoint_shard` eval with `RuntimeContractError[eval_forward.token_weighted_diag_zero_weight]` in `_finalize_disjoint_shard_scalars`. The replicated reference this reduction must reproduce exactly (`_weighted_average` in `src/losses/runner.py`, used by the SAME `LossRunner.finalize_planned_step` both modes call) resolves an identical zero-total-weight average to `0.0`, never an error -- confirmed by direct execution of both code paths on the identical zero-weight input. The pre-existing equivalence fixture matrix (`tests/eval/test_forward_eval.py`) never exercised this case: every fixture there has a nonzero `selected_count` for every configured term, so full-row exactness was not actually proven for this reachable case despite the task 4.1 DONE note's claim | accepted | resolved | Changed `_finalize_disjoint_shard_scalars`'s `weight <= 0.0` branch from `raise` to `result[base_key] = 0.0`, mirroring `_weighted_average`'s own zero-weight fallback exactly (the two are required to agree by construction: `TrainRuntime._reduce_eval_sum_metric` already fail-closes on a negative per-rank weight via `_checked_eval_count_value` before this function ever runs, so `weight <= 0.0` reaching here only ever means an honest global zero, never corrupted data). New decisive test `test_disjoint_shard_globally_zero_selected_term_matches_replicated_zero_weight_convention` (`tests/eval/test_forward_eval.py`) asserts the fixed result equals the independently-computed `_weighted_average` reference (`0.0`) and that the internal `__weight__` key never leaks out; manually verified decisive by reverting the fix in place and re-running -- fails with the exact predicted `RuntimeContractError[eval_forward.token_weighted_diag_zero_weight]`, then restored and re-verified green. See `src/eval/forward.py`. |

No other finding. Everything else traced cleanly: the sharding key is the
canonical sequence position (not `pack_index`, matching the prior P2-A
fix); the consensus collective (`validate_eval_reduction_consensus`) is
called unconditionally on every rank strictly before the sharding filter
and before `ForwardEvalRunner` construction, exactly as claimed; the
`finite/*` AND-reducer and the sum/identical reducer key classification
(`_is_eval_sum_metric_key`/`_is_eval_identical_metric_key`) have no
overlap or gap against the actual keys `_prepare_disjoint_shard_scalars`
produces; the replicated call path takes zero new branches (`reduction_mode`
kwarg only set when `self.reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD`);
`ForwardEvalObservation`/`to_logging_row()` are untouched (confirmed by
diff-hunk-location, not just prose) so the row schema claim in task 4.3
holds independent of this pass's own finding.

### Files changed for this review pass

`src/eval/forward.py`, `tests/eval/test_forward_eval.py`,
`openspec/changes/streamline-coordexp-swift-base-infrastructure/{tasks.md,review-triage.md}`.

### Re-verification after the fix (2026-08-04)

- New test in isolation: `tests/eval/test_forward_eval.py -k zero_selected_term` -- 1 passed.
- Targeted: `tests/eval/test_forward_eval.py tests/runtime/test_train_runtime.py tests/training/test_pipeline_assembly.py tests/losses/test_runner.py` -- 102 passed (up from 101; net +1, this pass's own regression test).
- `git diff --check`: clean.
- No production pack-cache root, GPU, commit, stage, or push used.

### Disposition: verdict converges to APPROVE; 4.4 remains unchecked

The one finding this pass raised (P1-1) is fixed, decisively tested, and
re-verified green -- no other finding survived inspection, so this
independent pass converges to **APPROVE** for the distributed-reduction
code itself. This closes the "focused independent review of the
distributed reduction" task 4.4 names and the P2-C item above records as
outstanding. Task 4.4 in `tasks.md` stays **unchecked** regardless: the
task also requires the exact 8-rank M4 paired-A/B wall-time measurement and
a checkpoint-selection replay on a recorded multi-rank run, both of which
remain unavailable in this session for the same GPU-occupancy reason
recorded in `implementation-notes.md` "M4" and reconfirmed at this
session's own GPU receipt (all 8 A100s externally occupied,
2026-08-04T07:52Z). Correctness convergence and the outstanding
measurement gate are independent conditions; only the former is satisfied
by this pass.

**CORRECTED, pre-final Opus P2 closures pass (2026-08-04, same day):** the
P1-1 finding's reachability claim above ("full-row exactness ... not
actually proven for this reachable case") is narrowed by a real end-to-end
test built in that later pass — see "Pre-final Opus P2 closures" at the end
of this document. The globally-zero-selected-term scenario is NOT reachable
via any real `LossRunner`-driven call in either reduction mode: a
pre-existing, Wave-4-independent check in `_build_denominator_from_token_sequences`
(`src/losses/runner.py`) fails closed first, symmetrically, before
`compute_micro_step`/`finalize_planned_step` ever runs. The fix itself
(`_finalize_disjoint_shard_scalars`'s `weight <= 0.0` branch resolving to
`0.0`) is kept — harmless, matches the reference convention — but is
correctly re-classified as defensive/dead code in production use, the same
class as the already-accepted P2-D `local_value is None` branch, not a live
correctness bug. **APPROVE still stands**; this narrows the finding's
severity/reachability characterization, it does not reopen a defect or
change task 4.4's checkbox state.

## Wave 5 (gated micro-optimizations) — review P2 fixes, 2026-08-04

Scope: four cheap P2 findings from a review of the Wave 5 (tasks 5.3-5.5)
implementation pass — all fixed in this same worktree the same day; no
GPU/cache/commit/stage/push for any of them. This pass changes only
receipts/docs plus two small, durable additions
(`src/qwen/encoding.py`'s comment, `tests/qwen/test_encoding.py`'s new
regression test); Wave 5's rejected-candidate dispositions (5.3's hoisted
`examples_by_id`/bisect token-span lookup, 5.4's batched finite gate) are
unchanged and still fully reverted from production code.

| ID | Severity | Area | Finding | Disposition | Status | Resolution |
| -- | -------- | ---- | ------- | ----------- | ------ | ---------- |
| P2-1 | P2 | receipt accuracy | Every Wave 5 gate receipt (this pass's own `tasks.md` 5.2/5.5 DONE notes, plus the pre-existing Wave 3/4 `review-triage.md` re-verification receipts) reported pass counts (523/510/1122) that silently counted one CUDA-executed test (`tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad`, gated only by `torch.cuda.is_available()`) as a plain pass, contrary to this session's own no-GPU policy | accepted | resolved | Re-ran the targeted (8-path), 7-path (5.1/5.2 baseline subset), and full repo suites with `CUDA_VISIBLE_DEVICES=""`: 522/509/1121 passed respectively, each with that one test reported `SKIPPED ... CUDA is required for bf16 parity` instead of passed. Corrected every occurrence of 523/510/1122 in `tasks.md` (5.2, 5.5) and `review-triage.md` (Wave 3, Wave 4 re-verification sections) to the CUDA-hidden counts, each annotated "corrected 2026-08-04 per Wave 5 review P2" with the prior (silently-GPU-inflated) figure quoted for traceability. No test logic changed; only the receipt text. |
| P2-2 | P2 | stale error-code reference | `review-triage.md`'s Wave 3 P2-G row (line ~135) names `trainer.forward_input_provider_requires_streaming_loss_runner` as the fix for a false-run-receipt finding, but Wave 5 task 5.1 deleted that error code outright (subsumed by a broader unconditional check) — the row read as if that code still exists | accepted | resolved | Appended a "[Superseded 2026-08-04, Wave 5 task 5.1]" note to the P2-G Resolution cell explaining `trainer.forward_input_provider_requires_streaming_loss_runner` was deleted once `SupervisedTrainer.__init__`'s unconditional `trainer.loss_runner_requires_streaming_protocol` check (raised for every non-streaming `loss_runner`, not only the provider combination) made it a strictly-subsumed, unreachable subset. Original row text left otherwise unchanged (historical record). |
| P2-3 | P2 | missing normative delta | Wave 5 task 5.1 made the streaming-trio construction-time gate (trainer and eval, fail-closed, batch path removed) a hard production behavior change, but no OpenSpec delta requirement captured it as a normative contract — only prose in `tasks.md`'s DONE note | accepted | resolved | Added `## ADDED Requirements` / "Loss Runner Streaming Protocol Is Required At Construction" to `specs/coordexp-swift-supervision-losses/spec.md`, with three scenarios (trainer construction rejects a non-streaming loss runner; eval construction rejects one in every reduction mode; a batch-only `compute`-implementing loss runner is rejected outright, not routed to a batch fallback). The existing `## MODIFIED Requirements` / "Loss Bundle Metrics" entry (and its scenario union) was left untouched — this is a new requirement, not a modification. `openspec validate --strict` and `--all --strict`: 19/19 pass. |
| P2-4 | P2 | lesson not durable | The rejected-bisect finding (task 5.3: a non-monotonic zero-width special-token offset breaks a bisect/binary-search shortcut over `offset_mapping`) existed only in `implementation-notes.md` prose — a future contributor re-attempting the same optimization would have no in-code signal | accepted | resolved | Added a short comment at `_token_indices_for_char_range` (`src/qwen/encoding.py`) stating the non-monotonicity hazard, and a new permanent test `tests/qwen/test_encoding.py::test_token_span_lookup_handles_non_monotonic_zero_width_offset_without_bisect` asserting the exact `[(0,5),(5,10),(0,0),(10,15)]`/`[5,15)` counterexample resolves to `(1, 3)` with no spurious `qwen.span_token_coverage` error. No bisect implementation, dispatch, or private control was reintroduced — the function itself is unchanged except for the comment. |

### Files changed for this review pass

`src/qwen/encoding.py`, `tests/qwen/test_encoding.py`,
`openspec/changes/streamline-coordexp-swift-base-infrastructure/{tasks.md,review-triage.md,implementation-notes.md,specs/coordexp-swift-supervision-losses/spec.md}`.

### Re-verification after fixes (2026-08-04)

- Targeted: `tests/losses/ tests/training/ tests/runtime/ tests/config/ tests/artifacts/ tests/eval/ tests/qwen/ tests/packing/` -- 523 passed, 1 skipped (CUDA-gated) -- the corrected 522 above, +1 for this pass's own new permanent regression test.
- Full repo `pytest.ini` `testpaths` suite: 1122 passed, 1 skipped (CUDA-gated) -- same +1.
- `tests/qwen/test_encoding.py` in isolation -- 6 passed (5 pre-existing + the new permanent regression test).
- `ruff check` on `src/qwen/encoding.py` and `tests/qwen/test_encoding.py`: clean.
- `git diff --check`: clean.
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict` and `openspec validate --all --strict`: 19/19 pass.
- No production pack-cache root, GPU, commit, stage, or push for this pass.

### Disposition: tasks 5.1-5.5 remain checked

All four P2s are documentation/receipt corrections or additive, durable
test/comment guards — none change Wave 5's substantive dispositions (both
5.3 sub-candidates and the 5.4 candidate remain rejected-with-evidence and
fully reverted from production code; 5.1/5.2's deletions are unaffected).
Tasks 5.1-5.5 in `tasks.md` remain checked.

## Archive-time bookkeeping audit: historical test-count collected-vs-passed conflation (2026-08-04)

Task 6.4 requires strict delta/hygiene inspection; this includes auditing
this document's own prior bookkeeping correction (Wave 5 P2-1 above) for
completeness, since it is itself a receipt this change's final acceptance
relies on. P2-1's own resolution text claims it "corrected every occurrence
of 523/510/1122 in `tasks.md` (5.2, 5.5) and `review-triage.md` (Wave 3,
Wave 4 re-verification sections)". Re-reading the actual text as it stands
today: the Wave 4 re-verification section (above) does carry a `**corrected
2026-08-04 per Wave 5 review P2**` annotation with the prior figure quoted.
**The Wave 3 re-verification section does not** — its "481 passed (up from
475: net +6 ...)" line carries no skip/correction annotation at all, so
P2-1's claim of having corrected "Wave 3 ... re-verification sections" does
not match the text it actually produced. This is corrected here rather than
silently left standing.

What can be established from current evidence, without inventing a
retroactive number:

- **Wave 4's original pre-HOLD-fix count (499, `tasks.md` task 4.2/4.4's
  first DONE note) almost certainly included the CUDA-executed bf16 test as
  a real pass, not a skip.** Arithmetic check: the same task 4.4 STILL
  PARTIAL note computes its own re-verified count as "509 passed, 1 skipped
  ... up from 499 before this fix pass; net +11 new tests". `499 + 11 new
  tests = 510`, but the actual CUDA-hidden re-verified total is `509` — a
  discrepancy of exactly one, which is only consistent with `499` itself
  having counted that one CUDA test as a genuine pass (`499 - 1 flipped to
  skip + 11 new = 509`, matching exactly). This is a decisive arithmetic
  reconstruction, not a guess.
- **Wave 3's original count (481, `tasks.md` task 3.4 and the Wave 3
  re-verification section above) and Wave 1/2's count (453, `tasks.md` task
  1.4 and the Wave 1/2 re-verification section above) were never
  independently re-verified under `CUDA_VISIBLE_DEVICES=""`, and no
  equivalent arithmetic cross-check is available for either** (unlike Wave
  4, no later same-session note recomputes a "before/after" delta against
  either figure that would let the CUDA test's inclusion be inferred). The
  Wave 3 HOLD-fix session's own P2-H resolution text confirms a real CUDA
  device was visible and used (GPU 0, for the M3 re-measurement) during that
  same session, which makes it plausible that the "Broad" 481/453 test-suite
  runs in those sessions also had a GPU visible and therefore also counted
  the bf16 test as a real pass rather than a skip — but this is a plausible
  reading, not an established fact, and no historical run log exists to
  settle it. Recorded here as an **open, annotated uncertainty**: neither
  481 nor 453 should be read as confirmed CUDA-hidden counts, and neither is
  retroactively "corrected" to any specific alternate number, since doing so
  would invent evidence this session does not have.

**Disposition:** this is a receipt-accuracy annotation only, exactly like
Wave 5's own P2-1 — it does not change any task's checkbox state. Waves
1-5 remain checked/partial exactly as `tasks.md` already records; the
underlying correctness claims for those waves rest on the specific fixture
matrices and decisive tests cited throughout this document, not on any
aggregate "N passed" receipt, so this bookkeeping gap does not itself cast
doubt on any wave's substantive disposition. No further tasks.md edit was
made for Wave 1/3's own DONE notes beyond this annotation, per the explicit
instruction to annotate uncertainty rather than invent corrected numbers.

## Pre-final Opus P2 closures (2026-08-04, TEST/DOC-only pass)

Scope: five external-Opus-flagged P2 items closed with TEST/DOC-only
changes (no source semantics changed; the one production fix from the
prior Wave 4 distributed-reduction review is kept unchanged, per explicit
instruction). No GPU, cache, output, commit, stage, or push.

### 1. End-to-end sharded-vs-replicated test for a globally-zero-selected term

Built `test_disjoint_shard_eval_globally_zero_selected_term_is_rejected_identically_to_replicated`
(`tests/eval/test_forward_eval.py`), driving the REAL `ForwardEvalRunner` +
`LossRunner` + `TrainRuntime` bounded-collective path (`_replicated_reference_row`/
`_run_sharded_two_ranks`, now threading a `loss_runner` parameter through to
support this fixture) with `token_type_gate` configured against "schema" —
a valid V1 token type globally absent from a restricted 2-pack eval subset
(`_WAVE4_PACKS_ZERO_AND_ONE`).

**This test discovered that the P1-1 finding's premise was wrong in an
important way.** Driving the real `LossRunner.prepare_planned_step` (not
the hand-summed helper functions the original fix's test called directly)
shows a globally-zero-selected term can never reach `compute_micro_step`/
`finalize_planned_step` at all: `_build_denominator_from_token_sequences`
has a pre-existing, Wave-4-independent "at least one eligible segment"
fail-closed check (`loss.segment_balanced_zero_eligible`) that fires on
each rank's own local token sequences before any forward/compute happens.
Since `eligible_segment_count > 0` necessarily implies
`selected_atom_count > 0` for that same rank (a segment only becomes
eligible by having a selected atom), no rank can ever complete
`prepare_planned_step` with a term whose rank-local `selected_count` is
zero — and a *globally* zero-selected term means every rank is locally
zero, so every rank fails this same check, in BOTH reduction modes
identically. Confirmed directly:

```
replicated: LossContractError loss.segment_balanced_zero_eligible
sharded:    LossContractError loss.segment_balanced_zero_eligible
```

The decisive, honest end-to-end proof is therefore symmetric REJECTION
(identical error code, both modes), not a successful row — the test
asserts exactly that. `_finalize_disjoint_shard_scalars`'s `weight <= 0.0`
branch (the prior fix) is real, harmless, and matches the reference
`_weighted_average` zero-weight convention, but is now correctly
re-classified as **defensive/unreachable code in production use, not a
live correctness bug** — the same disposition already given to
`_prepare_disjoint_shard_scalars`'s own dead `local_value is None` branch
(task 4.2's P2-D). **Kept unchanged, per instruction.** The original
narrow pure-function test
(`test_disjoint_shard_globally_zero_selected_term_matches_replicated_zero_weight_convention`)
is kept and its docstring corrected to point at this finding — it remains
valid as a proof that the helper function matches the reference convention,
it just does not by itself establish production reachability.

The Wave 4 distributed-reduction review section above and `tasks.md` 4.4
are both annotated with this correction (verdict **APPROVE still stands**;
this narrows a finding's reachability characterization, it does not reopen
a defect).

### 2. Frozen train-row key-set assertion

Added `test_train_row_key_set_gains_exactly_the_three_timing_keys_and_never_leaks_accuracy_stats`
(`tests/training/test_pipeline_assembly.py`), driving the real
`_train_logging_handler` with a real `LossRunner` artifact. Rather than a
hardcoded literal key list (brittle against loss-runner-owned metric names
the row-schema contract does not itself own), it computes the "pre-existing
key set" from the SAME producer with timing fields left unmeasured
(`None`, the default), then asserts the timed row's key set equals that
baseline union exactly `{step_duration_seconds, input_build_seconds,
input_wait_seconds}` — no other key added, removed, or renamed — and that
`accuracy_stats` never becomes a row key in either row. This complements
(does not replace) the pre-existing
`test_train_logging_forwards_real_loss_runner_accuracy_stats_without_leaking_to_row`.

### 3. Profiling P2 (docs only)

Added a clarifying note to `implementation-notes.md`'s M3 section and a
short clause to `docs/COORDEXP_SWIFT.md`'s existing timing-fields bullet:
`QwenForwardInputsReceipt`'s `total_build_inputs_ns` (feeding
`input_build_seconds`) describes CPU-only input construction under both
provider modes as of P2-A, excluding the consumer-side H2D move that now
happens separately in `take()`; it is not comparable across this change to
any pre-Wave-3 fused-build sub-timing, and is not a substitute for
`step_duration_seconds` (the all-rank-maximum, H2D-inclusive, end-to-end
step measurement) for wall-clock comparisons. No new field or framework
added — this documents existing `timings_ns`/`step_duration_seconds`
semantics.

### 4. FA2 P2 (docs only)

Added a note to `implementation-notes.md`'s M4 section: under
`disjoint_shard` eval, `_apply_fa2_branch_proof_policy` runs AFTER the
sharding filter, so each rank's `first_micro_step` proof capture targets
that rank's own shard-local first pack, not the globally canonical first
pack (which is what `replicated` mode's identical policy application
targets, since every rank there sees the same full sequence). This is
intentional — it proves the actual rank-local FA2 path each rank really
executes — and has no metric/eval-row effect:
`capture_fa2_branch`/`require_fa2_branch_proof`/`fa2_branch_evidence` are
pack-level fields consumed only inside the Qwen forward boundary for proof
verification, never inputs to `ForwardEvalObservation`/`to_logging_row()`.
No code change; the safer existing ordering is untouched.

### 5. Corrected stale receipt counts and Ruff findings, with provenance

Every figure below was re-derived by an actual command run in this pass
(`CUDA_VISIBLE_DEVICES=""`, `ms` conda env), not inferred or invented:

| Suite | Prior figure | Current (this pass) | Provenance |
| --- | --- | --- | --- |
| 6.2's 3-path targeted (`tests/artifacts/test_run_artifacts.py tests/eval/test_forward_eval.py tests/training/test_pack_cache.py`) | 105 passed | **107 passed** | +1 same-day 4.4-fix test, +1 this pass's new end-to-end test |
| Wave-4-review 4-path targeted (`tests/eval/test_forward_eval.py tests/runtime/test_train_runtime.py tests/training/test_pipeline_assembly.py tests/losses/test_runner.py`) | 102 passed | **104 passed** | +1 this pass's end-to-end test, +1 this pass's frozen-key-set test |
| Broad 7-path (`tests/runtime/ tests/training/ tests/losses/ tests/config/ tests/artifacts/ tests/eval/ tests/qwen/`) | 509 passed, 1 skipped (last recorded, Wave 4's own STILL PARTIAL receipt) | **513 passed, 1 skipped** | +1 same-day 4.4-fix test (never separately recorded on this 7-path set), +2 this pass's new tests |
| Wave-5 8-path (adds `tests/packing/`) | 523 passed, 1 skipped (last recorded, Wave 5's own final receipt) | **526 passed, 1 skipped** | +1 same-day 4.4-fix test (never separately recorded on this 8-path set), +2 this pass's new tests |
| Full repo `pytest.ini` `testpaths` | 1121 passed, 1 skipped (last recorded, Wave 5's own final receipt) | **1125 passed, 1 skipped** | +1 same-day 4.4-fix test (never separately recorded on this full-repo set), +2 this pass's new tests |

**Ruff pre-existing-findings count (2 → 3):** every prior wave's own ruff
receipt ("2 pre-existing findings") was scoped to THAT wave's own touched
files, not the full union of files this entire change has ever touched —
Wave 4's touched-file set happens not to include
`src/training/supervised_trainer.py` (Wave 1/3's file, carrying the
pre-existing `Iterator` unused-import finding), so its own "2" count
(`RankGradientFiniteReport` in `train_runtime.py`, one unused `runtime`
local in `test_train_runtime.py`) was internally correct for its own scope
but incomplete as a whole-change total. Re-running `ruff check` across the
full union of every file this change has touched across all six waves
finds exactly **3** findings, confirmed pre-existing at the unmodified HEAD
baseline for each file individually (`git show HEAD:<file> | ruff check -`):

```
src/runtime/train_runtime.py:16:5 F401 RankGradientFiniteReport imported but unused
src/training/supervised_trainer.py:5:49 F401 Iterator imported but unused
tests/runtime/test_train_runtime.py:127:5 F841 Local variable `runtime` is assigned to but never used
```

Zero new findings introduced by this change, across its full touched-file
union, confirmed by this pass. No prior wave's own "2" receipt was
factually wrong for its own stated scope; this note supersedes only the
whole-change total.

### Verification for this pass

- `tests/eval/test_forward_eval.py -k "zero_selected_term or rejected_identically"` — 2 passed.
- `tests/eval/test_forward_eval.py` in isolation — 24 passed.
- `tests/training/test_pipeline_assembly.py -k key_set_gains_exactly` — 1 passed.
- `tests/training/test_pipeline_assembly.py` in isolation — 18 passed.
- All five re-derived suite counts in the table above, each from a direct re-run.
- `ruff check` on the full touched-file union — 3 findings, all confirmed pre-existing at HEAD baseline, zero new.
- `git diff --check` — clean.
- `openspec validate streamline-coordexp-swift-base-infrastructure --strict` and `openspec validate --all --strict` — 19/19 pass.
- No production pack-cache root, GPU, commit, stage, or push used.

### Disposition

All five items closed with TEST/DOC-only changes. The one production code
change from the prior review (`_finalize_disjoint_shard_scalars`'s
`weight <= 0.0` → `0.0`) is kept unchanged, per instruction, now correctly
characterized as defensive/unreachable-in-production code rather than a
live correctness fix. **4.4, 6.1a, 6.1b, and 6.5 are untouched** by this
pass — 4.4 stays unchecked for the same still-outstanding 8-rank M4
measurement and checkpoint-selector replay; 6.1a/6.1b/6.5 were never in
this pass's scope.

## Final independent convergence review (2026-08-04)

**Reviewer:** isolated read-only Claude Opus 5 lane. **Verdict: APPROVE.**
No P0 or P1 finding remains. The reviewer traced the full live diff and
production paths, independently recomputed the current production train-cache
fingerprint, verified every `code_identity` and dataset determinant against
working-tree bytes, and confirmed both settled production caches remain
complete/current-version. It also traced exact top-k/count/loss/finite
reducers, timing boundaries, streaming-only deletion, provider/eval defaults,
artifact compatibility, M4/M6 receipts, and every benchmark stop-rule
disposition.

Accepted P2 dispositions:

| Finding | Disposition |
| --- | --- |
| Task 6.3's original docs receipt still described the pre-M4 replicated default | Annotated as superseded after M4 ACCEPT; live operator docs already carried the correct sharded default. |
| The earlier Wave-4 review paragraph still said replicated remained default | Marked superseded immediately below the historical paragraph. |
| Proposal claimed every prod/smoke config already set `fa2_branch_proof` | Corrected: production configs did; eight omitted smoke profiles were pinned during implementation. |
| M3's planning shorthand said to revert wiring, while validated two-phase wiring was retained | Recorded as a deliberate narrow deviation: synchronous ships, overlap is rejected, retained wiring makes no performance claim and needs the already-recorded future three-arm comparison. |
| Pipeline-level multi-rank default routing had only real-GPU receipt coverage | Added a `world_size=2` pipeline-assembly assertion that the unset control selects and partitions `disjoint_shard`; direct resolver and M4/M6 coverage remain. |

All benchmark-dependent slices are closed: M2 and M4 are measured accepts;
M3 is rejected-with-evidence with synchronous shipped; M5a/M5b are
rejected-with-evidence and reverted; M6 records absolute final-state timings
without comparison to unavailable M0. Task 6.5 may therefore close.
