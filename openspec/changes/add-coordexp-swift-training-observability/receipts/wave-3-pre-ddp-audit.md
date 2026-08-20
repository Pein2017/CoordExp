# Wave-3 Pre-DDP/Cost Standards + Intent-Contract Audit (task 3.8)

- **date**: 2026-08-20
- **identity**: Opus observability wave-3 auditor, spawned by Claude Fable lead
  (distinct from the lead, the 3A/3B/probe builders, and the Wave-0 entry auditor)
- **frozen target**: HEAD `e81f6a91af016142c69ffd40752d97d47ed4170d`, tracked tree
  clean, no untracked paths except this receipt. Verified before any command.
- **commits under audit**:
  - `bdb29b3fa` feat(runtime): all-rank optimizer boundary consensus and update receipts (3A)
  - `4db58e972` feat(training): truthful observation rows, timing, and resources (3B)
  - `305b17eb7` test(probes): wave-3.8 fp16 CUDA and injected-outcome probes
  - `e81f6a91a` fix(runtime): read GradScaler found_inf for the optimizer it actually saw
- **scope**: read-only. No GPU command run; the four fp16 CUDA receipts are the GPU
  evidence and were audited for coherence, not re-executed. Only this file written.

## Verdicts

| axis | verdict |
| --- | --- |
| STANDARDS | **PASS-WITH-DISPOSITIONS** |
| INTENT-CONTRACT | **PASS-WITH-DISPOSITIONS** |
| **Wave-4 entry** | **CLEARED** (0 P0 / 0 P1) |

Counts: **P0 = 0, P1 = 0, P2 = 3, P3 = 6.**

## Findings

| id | axis | sev | finding | disposition |
| --- | --- | --- | --- | --- |
| W3-1 | STANDARDS | P2 | No Wave-3 manifest amendment exists. Append-only verified across all 7 manifest revisions (no command deleted or edited in place; amendments a growing prefix), but amendments stop at `amend-5` and `wave3-fp16-gpu-probes` still carries the literal `TO-FREEZE-IN-WAVE-3-PACKET` placeholder. The three Wave-3 argvs live only in `wave-3-fp16-probe-packet.md`; the fixture/collective argv is frozen **nowhere** (this audit reconstructed it by test-count match, see Commands §3). | `amend-6` in the Wave-3 close-out commit: retire the placeholder, freeze the three packet argvs plus the fixture/collective argv, pin observed counts. |
| W3-2 | INTENT (e) | P2 | `lr/group_<index>` means two different things in one row builder. With a receipt (the production path) it is the pre-call applied LR, per spec. With `update_receipt=None` — the receipt-less legacy observation shape pinned by `test_an_observation_without_wave3_inputs_gains_no_row_key` — `reporting._scheduler_lr_metrics` publishes the **post-scheduler** value under the identical key, which is exactly what the spec's *"Applied learning rate is observed"* scenario forbids (`spec.md:324`). Production-unreachable: `SupervisedTrainer` has a single boundary call site (`supervised_trainer.py:407`) that always builds a receipt. | Accept as a bound deviation tied to the receipt-less legacy shape. **Do not delete** — the fallback is load-bearing for the additive-envelope characterization row. Record in the close-out; retire with the legacy shape at the Wave-5 deletion. |
| W3-3 | INTENT (d) | P2 | The session-level terminal path is proven only by source inspection. `tests/training/test_reporting.py:1206` asserts three substrings in `session.py`. It proves neither publish-before-re-raise ordering, nor zero eval/checkpoint/exact-resume/selector/final-success dispatch, nor boundary-code-primary during failed finalization. The injected-outcome probe closes the *reporting* seam behaviorally (real two-rank gloo, `publish_terminal_boundary_row` + real `RunWriter.append_logging_row`, `row_line_count: 1`, `zero_scheduler_progression`/`zero_scheduled_handler_progression` true on all six arms) but drives its own harness, not `_run_training_session`. P2 not P1 because the structure is sound: `execute_optimizer_boundary` raises out of `trainer.run()`, so every post-boundary handler is unreachable by construction. | **Carried obligation onto task 4.1**, whose text already requires covering the terminal row and an injected append failure with the primary boundary code. It must be **behavioral at the session seam**, not a source grep. |
| W3-4 | INTENT (a) | P3 | `_unwrapped_optimizer` unwraps exactly one level; `Accelerator.unscale_gradients` uses `while isinstance(opt, AcceleratedOptimizer)` (re-derived, Commands §7). A doubly-prepared optimizer would silently reintroduce the keying miss (now a `.get` miss → `False`). | Two-line fix (loop instead of single hop) whenever `_unwrapped_optimizer` is next touched. |
| W3-5 | INTENT (a) | P3 | `_scaler_found_inf` returns `False` silently when `_per_optimizer_states` is a Mapping with no matching/correctly shaped entry; it does not fall through to the accessor and produces no `report_error_code`. Unreachable under torch 2.9.1 (verified: `unscale_` always writes `found_inf_per_device`), but a future torch internal rename degrades silently into the exact pre-fix defect. The standing guard (`tests/runtime/test_wave3_optimizer_boundary.py:890-948`) is a torch-*mimicking* fake. | Record. Consider a torch-internals attribute assertion at runtime construction in a later wave. |
| W3-6 | INTENT (e) | P3 | `pre_clip_grad_norm_rank_max` is the max over ranks with **finite** norms only — `finite_gates.py:299-319` filters non-finite before `max`. Exact under `apply` (all ranks finite); under a partial-candidate `scaler_skip` the published "rank max" silently excludes the non-finite ranks. | Record; name the exclusion in the field's spec text or in `unavailable_fields` in a later wave. |
| W3-7 | INTENT (f) | P3 | At accumulation > 1, `count/physical_tokens` is incremented at the top of the micro-step body **before** `pre_backward` (`supervised_trainer.py:304-308`), so a mid-loop non-finite break counts a micro-step that forwarded but never backwarded; and `count/supervised_atoms` comes from the plan-time merged denominator, so a partial planned step's throughput numerators can exceed completed work. Both only affect rows on already-skipped steps, and `micro_step_count` + `_partial_loss_artifact` keep the partiality visible. | Record. Not a double-count: physical tokens sum over micro-steps then SUM across ranks; supervised atoms are already global and correctly reduce IDENTICAL, never summed twice. |
| W3-8 | STANDARDS | P3 | `wave-3-injected-outcome-receipt.json` carries no `observed_bounds` block although the packet declared bounds for it (6 arms, CPU-only). All four fp16 receipts do. | Coherence gap only; note in `amend-6`. |
| W3-9 | INTENT (e) | P3 | `optimizer_step_count` / `scheduler_step_count` enter the row through `_counter_fields(runtime)` **after** reduction, as rank-0-local values, bypassing the Wave-2 typed-reducer declaration. Consensus keeps ranks in lockstep, so a divergence would fault elsewhere first. | Record; candidate for an IDENTICAL declaration in Wave 4. |

**Health status (not a finding).** Flake tripwire
`tests/losses/test_zero_policies.py::test_gate_ablation_creates_no_autograd_edge_into_the_objective`
(manifest amend-4): **zero occurrences** across every replay in this audit. Count
stands at **2**; the tripwire fires at 3. Restate in the close-out.

## STANDARDS — re-derived, not trusted

- **Gate replay** `wave3-optimizer-gate`: **834 passed / 0 failed / 0 skipped**, exit 0
  (within the expected 832-840 band). Independent replay, not the builder's report.
- **Fixture/collective argv**: **15 passed / 0 failed**. Reconstructed by count-match
  (13 + 2); see W3-1 — the manifest holds no frozen record of this argv.
- **Determinant invariant**: independently re-derived at `e81f6a91a` via the amend-4
  redirected-baseline wrapper (probe module unedited, only `BASELINE_PATH` redirected).
  Both splits **EQUAL** with `payload_equal: true`; `tree_dirty: false`:
  - train `8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f`
  - eval.forward `3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662`
- **Manifest append-only**: verified programmatically across `51cc48de6 → f410a4bfa →
  9f9dbf1ba → bdb29b3fa → 4db58e972 → 305b17eb7 → e81f6a91a`. **No violations.**
  Wave-3's four commits added **no** manifest entry at all (→ W3-1).
- **Strict OpenSpec validation**: `Change 'add-coordexp-swift-training-observability' is valid`.
- **Residue checks**: `accelerator.clip_grad_norm_` — zero hits in `src/` (only the
  unrelated `pre_clip_grad_norm_rank_max` identifier matches the substring); zero
  `synchronize()` in `reporting.py` / `train_runtime.py` / `resources.py`; zero direct
  `all_gather` / `all_reduce` / `broadcast` in `train_runtime.py` (every collective
  goes through the injected `rank_report_gatherer` or the accelerator).
- **Receipt bounds vs packet** (packet limits: ws ≤ 2, devices ≤ 2, 2 planned steps and
  2 forwards per fp16 run, cache passes **0**, wall ≤ 300 s, GPU peak ≤ 2 GiB, artifact
  ≤ 1 MiB):

  | receipt | ws | dev | steps | fwd | cache | wall s | peak alloc | bytes | findings | ok |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | ws1 attempt-1 | 1 | 1 | 2 | 2 | 0 | 0.72 | 18.1 MB | 11.7 KB | **1 (fired)** | true |
  | ws1 attempt-2 | 1 | 1 | 2 | 2 | 0 | 0.771 | 18.1 MB | 11.3 KB | 0 | true |
  | ws2 attempt-1 | 2 | 2 | 2 | 2 | 0 | 3.758 | 18.1 MB | 22.0 KB | **1 (fired)** | true |
  | ws2 attempt-2 | 2 | 2 | 2 | 2 | 0 | 4.134 | 18.1 MB | 21.6 KB | 0 | true |

  Every bound honoured, `cache_or_materialization_passes = 0` on all four,
  `evidence_class: genuine_cuda_fp16_accelerate`, real A100 80GB devices,
  torch 2.9.1+cu128. GPU peak is not separately recorded per-device beyond
  `peak_cuda_allocated_bytes` / `peak_cuda_reserved_bytes` (23.1 MB), both far
  inside 2 GiB.
- **Known-defect protocol honoured**: attempt-1 receipts fire the FINDING and are
  retained as real-CUDA RED evidence; attempt-2 receipts are FINDING-free. The
  attempt-1 FINDING text — *"`_scaler_found_inf(scaler, self.optimizer)` … looks up a
  defaultdict miss. The converged action was still correct because the gradient scan
  is independent evidence"* — matches the defect `e81f6a91a` fixes, and the
  attempt-1 payload corroborates it: overflow arm
  `scaler_found_inf_ground_truth_inner_optimizer: true` /
  `scaler_found_inf_visible_to_runtime: false`, flipping to `true`/`true` in attempt-2.

## INTENT-CONTRACT

### (a) Scaler-keying fix — **correct**

Adjudicated against the **installed** torch 2.9.1 / accelerate 1.10.1 sources, not memory:

- `GradScaler.__init__`: `self._per_optimizer_states: dict[int, dict[str, Any]] = defaultdict(...)`.
  `.get(key)` on a `defaultdict` never invokes the factory, so the lookup is genuinely
  **non-mutating** — this matters, because the real `_found_inf_per_device` accessor
  *does* index the defaultdict (`return self._per_optimizer_states[id(optimizer)][...]`)
  and would insert the very state a probe reads as ground truth.
- Entry shape: `_refresh_per_optimizer_state() -> {"stage": OptState.READY,
  "found_inf_per_device": {}}`, and `unscale_` writes
  `optimizer_state["found_inf_per_device"] = self._unscale_grads_(...)`. The code's
  `isinstance(state, Mapping) and "found_inf_per_device" in state` matches exactly.
- Key identity: `Accelerator.unscale_gradients` does
  `while isinstance(opt, AcceleratedOptimizer): opt = opt.optimizer` then
  `self.scaler.unscale_(opt)`, and `AcceleratedOptimizer.step` does
  `self.scaler.step(self.optimizer, closure)` — the scaler only ever sees the **inner**
  optimizer. `_unwrapped_optimizer`'s `getattr(optimizer, "optimizer", None)` resolves
  to that same object for the single-wrap case (see W3-4 for the nesting caveat).
- **Staleness ruled out**: `GradScaler.update()` ends with
  `self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)`, and
  `AcceleratedOptimizer.step` calls `scaler.update()` immediately after `scaler.step`.
  So a completed wrapper call clears the record, and `_scaler_found_inf` is called
  inside `post_backward` immediately after this step's exactly-once unscale. On the
  `not_attempted` branch nothing unscales and the container is already empty; on a
  terminal branch the run ends. No prior step's entry can be read as current truth.
- **Fallback degrades safely**: `else` (non-Mapping `_per_optimizer_states`) → accessor
  with a `TypeError` arity ladder → `False`; unreadable state raises into
  `post_backward`'s `except`, which converts it into
  `report_error_code="fp16_found_inf_unreadable"` and converges as
  `pre_wrapper_unrelated_unsafe` rather than raising rank-locally. The only silent
  degradation is W3-5.
- **Cannot flip a decision incorrectly.** The fix can only turn `scaler_found_inf`
  `False → True`, which turns `is_safe()` false and `is_scaler_overflow_candidate()`
  true. The pre-fix hazard was the reverse: finite gradients + scaler-recorded inf
  selected `apply`, the wrapper then skipped, and the run took the
  `apply + all_skipped` **terminal**. Post-fix that same state converges the correct
  `scaler_skip`. The opposite direction (runtime sees non-finite grads, scaler does
  not — e.g. a model parameter outside the optimizer's param groups) still converges
  `scaler_skip`, the wrapper applies, and `scaler_skip + none_skipped` catches it as
  terminal with `mutation_state=applied_unsafe`. Both directions are covered.

### (b) fp16-only post-wrapper consensus — **spec-authored, correctly gated**

`_active_fp16_scaler()` returns `None` unless
`_normalize_mixed_precision(accelerator.mixed_precision) == "fp16"`; bf16 normalizes to
`"bf16"`, fp32/None to `"no"`, so `scaler_active` is `False` on every rank, and
`_reduce_boundary_action` takes the retained non-scaler branch
(`apply` / `not_attempted`). `execute_optimizer_boundary` then returns at
`if not decision.scaler_active:` **before** `_converge_post_wrapper_outcome` — so the
non-fp16 path runs the same single gradient gather it ran before Wave 3, byte-unchanged.
A disabled scaler (`is_enabled() is False`) also takes the zero-collective path, which
is right: no skip can occur. Rank-divergent scaler presence cannot reach the wrapper —
`len(scaler_ranks) != len(checked)` converges
`pre_wrapper_scaler_candidacy_divergent` (exercised by the injected probe).
**ws = 1**: `_gather_rank_reports` returns `(local_report,)` with no collective, and
`reduce_post_wrapper_reports` reduces that 1-tuple; the ws1 CUDA receipts confirm
`distributed_type: "NO"` with the full boundary exercised.

### (c) Receipt truth matrix vs tasks 3.1 / 3.2 — **every row matches**

| outcome | LRs | mutation_state | attempted/applied/skipped | counters | evidence |
| --- | --- | --- | --- | --- | --- |
| `apply` + `none_skipped` | pre-call `[0.125]` (≠ post-scheduler `[0.0625]`) | `applied` | true/true/false | opt +1, sched +1 | ws1+ws2 finite arm |
| `scaler_skip` + `all_skipped` | all `null` | `scaler_suppressed` | true/false/true | opt +1, sched +1, `underlying_applied_calls: 0`, param digest unchanged | ws1+ws2 overflow arm |
| `not_attempted` | all `null` | `unchanged` | false/false/false | opt **+0** | CPU suites (green in gate replay) |
| pre-wrapper terminal ×3 | all `null` | `divergent_or_unknown` | false/false/false | opt +0, sched +0 | injected arms 101/102/103 |
| post-wrapper `mixed` | all `null` | `divergent_or_unknown` | true/**null**/**null** | opt +1, sched +0 | injected arm 104 |
| `apply` + `all_skipped` | all `null` | `scaler_suppressed` | true/false/true | opt +1, sched +0 | injected arm 105 |
| `scaler_skip` + `none_skipped` | **pre-call `[0.125]` retained** | `applied_unsafe` | true/true/false | opt +1, sched +0, `underlying_applied_calls: 1` | injected arm 106 |

Each row reproduces `AppliedUpdateReceipt`'s constructors exactly
(`optimizer_boundary.py:176-375`), including the two deliberate asymmetries task 3.2
demands: `mixed` is the *only* branch that nulls application truth, and
`scaler_skip + none_skipped` is the *only* terminal that keeps real LRs — correctly,
because every rank genuinely applied them. `terminal_not_attempted` derives
`divergent_or_unknown` from the observed `unscale_completed` fact rather than a
caller-chosen label, and `GateDecision.unscale_completed` uses `any(...)` not `all(...)`,
so the no-scaler rank in the candidacy-divergent arm still reports
`divergent_or_unknown` — `cross_rank_receipt_identical: true` on all six arms. All
six arms: `process_exit_codes: [0, 0]`, real gloo, `cuda_initialized: false`, no
monkeypatched production modules, `zero_scheduler_progression` and
`zero_scheduled_handler_progression` true.

### (d) Terminal-row path — **structurally correct, behaviorally under-proven** (W3-3)

Exactly one bounded row (`row_line_count: 1`), no metric collective (only the existing
rank-zero append + its outcome broadcast), `publish_terminal_boundary_row` swallows
its own failure via `except BaseException: return` so the primary
`runtime.optimizer_boundary_terminal` code survives common failed finalization, and it
deliberately reads no lifecycle counter. The session catch at `session.py:2388-2401`
publishes then re-raises. What is missing is a behavioral proof at the session seam —
see W3-3.

### (e) Rows — **honest**

No backend-scaled recomputation: `_loss_term_observation_samples` consumes the
finalized artifact's `weight` / `denominator` and adds nothing derived. Availability is
never a fabricated zero: `_GatedObservation` rides a `BOOL_ALL` gate whose placeholder
`0.0` is popped and named in `unavailable_fields` unless *every* rank measured the
group; a reduced `None` is likewise named, not emitted; the first-step allocator delta
and any post-gap delta are unavailable rather than `0`. Throughput denominators are
right: all three numerators are **global** (`count/physical_tokens` SUM across ranks,
`count/supervised_atoms` already global and correctly IDENTICAL, `count/packs` SUM)
over the **rank-max** `step_duration_seconds` (MAX), never a mean. Reducer
declarations are constructed once in `metrics._loss_term_declarations` for both splits
with a single `objective_reducer` switch, so train and eval cannot classify a family
two ways. The IDENTICAL-not-MAX dispositions hold: `grad_norm/pre_clip_rank_max` is
already an all-rank max computed by a pure function of identical inputs, so IDENTICAL
is value-preserving and fails closed instead of hiding divergence under a second max
(caveat W3-6); `lr/group_*` are rank-local samples of a value every rank must agree on,
so IDENTICAL is the fail-closed choice. **The one family that means two things is
W3-2**, and it is production-unreachable.

### (f) Accumulation > 1 and ws = 1 — **no double-counting, no missed truth**

`post_backward` runs **once** per planned step after the micro-step loop, so
`_unscale_gradients_once` (with its `_unscaled_planned_step_id` guard) fires exactly
once; `unscale_calls: 1` on every probe arm confirms it. Work counts sum over
micro-steps then across ranks; the already-global merged-denominator counts are never
summed twice. A mid-loop break yields a supported `not_attempted` with
`mutation_state=unchanged` — truthful, since nothing unscaled. At ws = 1 both
consensus reducers take the local-tuple path with zero collectives and the rank-coverage
check still enforces exactly one report. Residuals are W3-7 only.

## Wave-4 entry statement

**Wave-4 entry is CLEARED.** No P0 and no P1 finding exists. The Wave-3 optimizer
boundary, its all-rank consensus, the runtime-owned receipt vocabulary, the truthful
row surface, and the terminal-row seam are semantically sound and backed by
independently replayed CPU gates, coherent real-CUDA fp16 receipts on both attempts,
a six-arm two-rank injected-outcome truth matrix, and a re-derived determinant
invariant. The three P2s are administrative (W3-1), production-unreachable and
characterization-bound (W3-2), and a proof-strength gap with a named Wave-4 owner
(W3-3). **No fp16 claim is made without both receipts**: attempt-2 ws1 and ws2 are the
final evidence and both are FINDING-free.

## Must-include list for the Wave-3 close-out commit

1. **Manifest `amend-6`** (the largest standards gap): retire the
   `TO-FREEZE-IN-WAVE-3-PACKET` placeholder on `wave3-fp16-gpu-probes`; freeze the
   three packet argvs verbatim; freeze the fixture/collective argv
   (`pytest tests/training/test_orchestration_compatibility.py tests/runtime/test_rank_report_collective.py -q`)
   which no manifest entry currently names; pin observed counts —
   `wave3-optimizer-gate 834/0/0`, fixture/collective `15/15`, four fp16 receipts with
   bounds honoured and cache passes 0, injected-outcome six arms exit `[0, 0]`,
   determinant EQUAL/EQUAL with payload equal at `e81f6a91a`, strict validate green.
   Note W3-8 (injected receipt carries no `observed_bounds`).
2. **tasks.md Wave-3 close-out block** checking 3.1-3.8, in the shape of the Wave-1/2
   blocks, recording both accepted deviations: W3-2 (`lr/group_*` legacy fallback,
   bound to the receipt-less characterization row, retire at the Wave-5 deletion —
   **not** to be deleted now, it is load-bearing for the additive envelope) and
   W3-3 (session-level terminal proof strength).
3. **Carried obligation, explicit**: W3-3 attaches to **task 4.1**, whose existing text
   already requires covering the terminal optimizer-boundary row and an injected append
   failure that converges failed finalization with the primary optimizer-boundary code.
   The close-out must state that this coverage has to be **behavioral at the
   `_run_training_session` seam** — ordering, zero handler dispatch, primary code —
   because the current session-level test is a source grep.
4. **Flake tripwire status**: count stands at **2**; **zero** occurrences across this
   audit's replays (834-test gate, 15-test fixture/collective, determinant probe).
5. **Attempt-1 receipts retained as real-CUDA RED evidence** (already true in-tree);
   the close-out should say so, so a later reader does not read the fired FINDING as an
   unresolved defect.
6. **The scaler-keying fix is validated against installed torch 2.9.1 / accelerate
   1.10.1 internals**, with W3-4 (single-level unwrap vs accelerate's `while` loop) and
   W3-5 (silent `False` on an unshaped state entry) recorded as P3 residuals.

## Commands (frozen argv + observed output)

Host rules honoured throughout: `PYTHONDONTWRITEBYTECODE=1`, `bash -c 'export …; unset …'`
wrappers, never an `env`-prefixed command, script files instead of heredocs. No command
hung; none was killed.

### 1. Frozen-target verification

```
$ git rev-parse HEAD
e81f6a91af016142c69ffd40752d97d47ed4170d
$ git status --porcelain
(empty)
$ git log --oneline -4
e81f6a91a fix(runtime): read GradScaler found_inf for the optimizer it actually saw
305b17eb7 test(probes): wave-3.8 fp16 CUDA and injected-outcome probes
4db58e972 feat(training): truthful observation rows, timing, and resources
bdb29b3fa feat(runtime): all-rank optimizer boundary consensus and update receipts
```

### 2. `wave3-optimizer-gate` replay (frozen manifest argv)

```
$ conda run -n ms pytest tests/training/test_supervised_trainer.py tests/runtime \
    tests/losses tests/training/test_reporting.py tests/artifacts -q
834 passed, 1 warning in 99.27s (0:01:39)
=== EXIT: 0 ===
```
(The single warning is the pre-existing sparse-CSR beta notice from
`tests/runtime/test_finite_gates.py`.)

### 3. Fixture/collective argv — reconstructed, see W3-1

```
$ conda run -n ms pytest tests/training/test_orchestration_compatibility.py \
    tests/runtime/test_rank_report_collective.py -q
15 passed in 33.55s
```

### 4. Determinant invariant (amend-4 redirected-baseline wrapper; probe module unedited)

```
$ conda run -n ms python <wrapper redirecting only BASELINE_PATH to the archived
                          losses wave-0-determinant-baseline.json>
  "commit": "e81f6a91af016142c69ffd40752d97d47ed4170d",
  "splits": {
    "eval.forward": { "baseline_fingerprint": "3b30c157d639...af6662",
                      "recomputed_fingerprint": "3b30c157d639...af6662",
                      "fingerprint_equal": true, "payload_equal": true,
                      "first_difference": null, "verdict": "EQUAL" },
    "train":       { "baseline_fingerprint": "8f11237fc793...76f2f",
                      "recomputed_fingerprint": "8f11237fc793...76f2f",
                      "fingerprint_equal": true, "payload_equal": true,
                      "first_difference": null, "verdict": "EQUAL" }
  },
  "status": "OK", "tree_dirty": false, "findings": [], "wall_seconds": 1.357
EXIT=0
```

### 5. Strict OpenSpec validation

```
$ conda run -n ms openspec validate add-coordexp-swift-training-observability --strict
Change 'add-coordexp-swift-training-observability' is valid
```

### 6. Residue checks

```
$ grep -rn "clip_grad_norm_" src/ | grep -v "torch.nn.utils.clip_grad_norm_"
  -> only `pre_clip_grad_norm_rank_max` identifier hits; zero accelerator.clip_grad_norm_ uses
$ grep -rn "synchronize()" src/training/reporting.py src/runtime/train_runtime.py \
      src/artifacts/resources.py
  -> (empty)
$ grep -rn "all_gather\|all_reduce\|broadcast" src/runtime/train_runtime.py
  -> (empty)
```

### 7. torch / accelerate internals re-derived (basis of the (a) adjudication)

```
$ conda run -n ms python <inspect.getsource script>
torch 2.9.1+cu128 accelerate 1.10.1
GradScaler.__init__:  self._per_optimizer_states: dict[int, dict[str, Any]] = defaultdict(
GradScaler.update():  self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
_refresh_per_optimizer_state() -> {"stage": OptState.READY, "found_inf_per_device": {}}
GradScaler.unscale_:  optimizer_state = self._per_optimizer_states[id(optimizer)]
                      optimizer_state["found_inf_per_device"] = self._unscale_grads_(
                      optimizer_state["stage"] = OptState.UNSCALED
GradScaler._found_inf_per_device: return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
AcceleratedOptimizer.step:  self.scaler.step(self.optimizer, closure) ; self.scaler.update()
AcceleratedOptimizer.__init__: self.optimizer = optimizer
Accelerator.unscale_gradients: while isinstance(opt, AcceleratedOptimizer): opt = opt.optimizer
                               self.scaler.unscale_(opt)
```

### 8. Manifest append-only verification

```
$ python <append-only diff over every manifest revision>
51cc48de6 (first revision): 13 commands, amendments=[amend-1, amend-2, amend-3]
51cc48de6 -> f410a4bfa: +commands=[] +amendments=['amend-4-pin-wave1-gate']
f410a4bfa -> 9f9dbf1ba: +commands=[] +amendments=['amend-5-freeze-wave2-gloo-probe']
9f9dbf1ba -> bdb29b3fa: +commands=[] +amendments=[]
bdb29b3fa -> 4db58e972: +commands=[] +amendments=[]
4db58e972 -> 305b17eb7: +commands=[] +amendments=[]
305b17eb7 -> e81f6a91a: +commands=[] +amendments=[]

HEAD amendments: [amend-1, amend-2, amend-3, amend-4, amend-5]
HEAD wave3-fp16-gpu-probes argv: TO-FREEZE-IN-WAVE-3-PACKET: genuine CUDA fp16 finite + overf...

VIOLATIONS: NONE - append-only holds
```

### 9. Receipt coherence dumps

Six receipts read in full and cross-checked against `optimizer_boundary.py`'s
constructors, the packet bounds table, and the tasks 3.1/3.2/3.7 literal clauses; the
per-arm results are summarised in the STANDARDS bounds table and the (c) truth matrix
above. `wave-3-injected-outcome-receipt.json`: `ok: true`, `findings: []`,
`process_exit_codes: [0, 0]`, `backend: gloo`, `world_size: 2`,
`cuda_initialized: false`, all six arms `ok: true` and
`cross_rank_receipt_identical: true`, 17/17 checks true on every arm,
`monkeypatched_modules: []`.
