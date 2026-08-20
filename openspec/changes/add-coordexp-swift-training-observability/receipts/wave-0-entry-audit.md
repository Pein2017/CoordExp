# Wave-0 Entry Audit (task 0.5)

| Field | Value |
| --- | --- |
| Change | `add-coordexp-swift-training-observability` |
| Base commit | `3d390b108db5c666a7c2bd2f7d79b12237de81be` |
| Date | 2026-08-20 |
| Identity | Opus observability entry auditor, spawned by the Claude Fable lead (distinct from the lead and from the 0.2 rebase builder) |
| Scope | read-only audit; this file is the only write |
| **STANDARDS verdict** | **PASS-WITH-DISPOSITIONS** |
| **INTENT-CONTRACT verdict** | **PASS-WITH-DISPOSITIONS** |
| **Entry gate** | **NOT cleared as-frozen** (one P0/P1: `P1-REDUCE`). Cleared once the Wave-0 close-out commit lands the tasks.md amendment assigning the reducer reclassification, plus the manifest amendments below. |

Frozen target verified at audit start and re-verified immediately before this
write: HEAD `3d390b108`, tracked modifications exactly the three delta spec
files, untracked exactly `receipts/{command-manifest.json,wave-0-baseline.md}`.
This receipt is the intended third untracked receipt file, not a HOLD condition.

---

## 1. STANDARDS — re-derivation of tasks 0.1-0.4

### 1.1 Predecessors and validation (task 0.1) — PASS

All three predecessors are present under `openspec/changes/archive/`:
`2026-08-19-reconcile-coordexp-swift-training-contracts`,
`2026-08-20-decompose-coordexp-swift-training-orchestration`,
`2026-08-20-standardize-coordexp-swift-supervised-losses`. The observability
change is the only active change. `openspec validate --all` returns
**20 passed / 0 failed (20 items)** including
`change/add-coordexp-swift-training-observability`; strict validation of the
change returns `valid`. Independently re-derived, matching the baseline receipt.

### 1.2 THE CORE ITEM — the 0.2 delta rebase — PASS (clean, both directions)

**Direction A: stable-spec preservation.** Every `## MODIFIED Requirements`
block in the three deltas was parsed and diffed against the current stable specs
(`openspec/specs/<capability>/spec.md`) at scenario granularity and at
normalized-sentence granularity.

| Capability | Requirement | Stable scenarios | Delta scenarios | Missing from delta |
| --- | --- | --- | --- | --- |
| config-runtime | Cadence Config And Resolved Step Schedule | 4 | 8 | **0** |
| config-runtime | Planned Step Schedule | 2 | 4 | **0** |
| supervision-losses | Non-Finite Loss And Gradient Gates | 4 | 9 | **0** |
| training-artifacts | Optimizer-Step Order | 1 | 4 | **0** |
| training-artifacts | Wide-Step Logging Stream | 6 | 20 | **0** |

**Zero stable scenarios are missing.** Every non-verbatim stable prose sentence
was individually inspected against the delta text; all are reworded-and-preserved
or strengthened, none dropped. Representative resolutions:

- "for every completed train step" → "for every completed planned optimizer-step
  boundary" plus the terminal-row exception (delta artifacts L80-86) — widened.
- "both MUST use the planned training step as `step`" → "both MUST contain the
  required planned training `step`" (L86-88) — preserved.
- Timing paragraph (L204-218) retains the full boundary definition, all-rank
  maximum, and the explicit-mean clause, and adds `input_h2d_seconds`.
- Writer-prohibition sentence (L252-256) preserved and extended with
  "database, or external tracking machinery".
- `Optimizer-Step Order` canonical order (L11-19) preserved and extended with
  exactly-once unscale / action-specific clipping / post-wrapper consensus.
- losses prose L5-18 preserves the pre/post-backward separation, the recoverable
  clause, the one-reduced-all-rank-decision clause, and the once-per-step
  rank-zero row clause.

Required spot-checks, all present verbatim:

| Spot-check | Location |
| --- | --- |
| "sum of weighted objective terms" | artifacts delta L94 |
| "raw finite status of every computed protected term" | losses delta L11 |
| raw/weighted field family (`loss/<term>/raw` + `loss/<term>/weighted` + term-count + finite-status) | artifacts delta L92-94 |
| no-bare-alias clause ("The removed ambiguous bare `loss/<term>` alias MUST NOT be restored for any term.") | artifacts delta L100-101 |
| omitted-term no-field-family | artifacts delta L96-97 and L121-123 |

**Direction B: observability intent preserved through the rebase.** Every line
the rebase removed (`git diff` on the three deltas: 91 insertions, 19 deletions)
was accounted for:

- the pre-rebase `> Application precondition:` marker was correctly replaced by
  the `> Rebase receipt:` note (artifacts delta L3-7);
- the pre-rebase compressed loss-field paragraph was replaced by the stable
  losses paragraph plus an additive paragraph (L103-116); every item it listed
  (weighted total, raw, configured weight, weighted, denominator scope,
  eligible-segment / selected-atom / skipped-segment counts, `acc_top1`,
  `acc_top5`, update+finite status, group LRs, pre-call LR sampling, null LR on
  skip) survives;
- one reflow line in the losses delta;
- the scenario bullet "…independent of `observability.steps`" was **not** lost —
  it was reworded to "all **other** canonical logging metrics …, independent of
  `observability.steps`" and an extra raw/weighted bullet was added ahead of it
  (artifacts delta L270-273).

Each proposal/design intent was independently located in the rebased text:

| Intent | Evidence |
| --- | --- |
| `observability.steps` required, no default, presentation-only | config delta L11-16; scenarios "Presentation interval omitted" (L41-46), "…is invalid" (L48-52), "Every train step is logged" (L34-39) |
| typed reducers, no implicit mean, `BOOL_ALL` not `ALL` | artifacts delta L228-239; scenario "Unknown reducer is requested" (L459) |
| optimizer-boundary terminal receipts | artifacts delta L125-195; losses delta L20-51; config delta L92-104 |
| JSONL-first sink ordering | ADDED `Rank-Zero Presentation Sinks`, "MUST be successfully published before either derived sink consumes it" + scenario "Presentation interval is reached" |
| sink-failure isolation / one-way latch | ADDED requirement, one-way disabled latch paragraph + scenarios "TensorBoard sink fails", "…flush or close fails after a successful add" |

No intent was lost or weakened by the rebase.

### 1.3 The three deliberate pre-losses-base modifications — PASS (archaeology)

`git log` shows the change directory and all three delta files were created by
exactly one commit, `0b98f0561` (2026-08-12, "docs(training): plan
infrastructure refinement"). Each flagged phrase was checked against that
commit's blobs:

| Phrase | Present at `0b98f0561`? | Verdict |
| --- | --- | --- |
| "unacknowledged corrupted" (losses) | yes, L7 of the 2026-08-12 blob | authored 2026-08-12 intent |
| "or synchronized skips" (losses) | yes, L8 of the same blob | authored 2026-08-12 intent |
| timing-barrier relaxation ("without adding a per-step barrier solely for observation") | yes, artifacts blob L201-202 | authored 2026-08-12 intent |

The entire 2026-08-12 timing paragraph (including `input_h2d_seconds`, the
named-scope clause, and the "host enqueue time MUST NOT be labeled as GPU
execution time" prohibition) is **byte-identical** to the current rebased text.
None of the three is an accidental stale-base survival. Note for the record that
all three are genuine relaxations of stable text and are semantically required
by design decisions 3 and 4 (acknowledged-corruption terminal receipts;
synchronized `scaler_skip`; CUDA-event H2D bracketing).

### 1.4 Owner/import graph (task 0.3) — PASS (re-derived)

- `src/runtime/metrics.py` — ABSENT (ENOENT), as claimed.
- `src/artifacts/observation_publisher.py` — ABSENT (ENOENT), as claimed.
- `grep -rn "SummaryWriter\|tensorboard" src/` — **zero hits**; no competing
  TensorBoard owner anywhere in `src/`.
- `src/training/pipeline.py` — zero `SummaryWriter`/`tensorboard`/`console`/
  `print(` hits; the facade is not a competing owner.
- Existing seams present at the declared sizes: `src/training/reporting.py`
  (368 lines), `src/training/session.py` (3534), `src/runtime/train_runtime.py`
  (1165, holding `gather_metrics`/`_reduce_metric_reports`).

### 1.5 Entry baseline replay (task 0.5) — PASS

`wave0-entry-baseline` argv replayed verbatim from the manifest inside a
`bash -c` export wrapper (`PYTHONDONTWRITEBYTECODE=1`; the nine listed variables
unset; no `env`-prefix), real observed output:

```
890 passed, 7 warnings in 98.19s (0:01:38)
```

**890 passed / 0 failed / 0 skipped**, matching the recorded baseline exactly.

### 1.6 Command manifest assessment (task 0.4) vs the 6-wave/40-task plan

`tasks.md` contains 6 waves and 40 tasks (5/5/7/8/8/7). The manifest is
schema-correct, has an append-only amendment rule, a cache invariant with both
frozen fingerprints, and correctly marks the three GPU/distributed argvs
TO-FREEZE. Gaps found are recorded as `P2-MANIFEST-*` below.

---

## 2. INTENT-CONTRACT — pre-implementation conflict probes

### 2.1 (a) Required no-default `observability.steps` vs frozen config identity — PASS

All four historicized families use a **live-computed** drift predicate rather
than a re-pinned byte constant, so a second migration of the same files keeps
them in the already-drifted branch:

- **reconcile executor**: `tests/training/test_reconcile_exact_resume_packet_executor.py`
  L54-59 computes `_LIVE_BASE_CONFIG_DRIFTED = (not exists) or sha256(live) !=
  BASE_CONFIG_SHA256` and skipif-gates the mechanism suite on it;
  `..._historicized.py` skipifs on `not _LIVE_BASE_CONFIG_DRIFTED` and asserts
  `live != executor.BASE_CONFIG_SHA256`. Its pin
  (`44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`) binds the
  **pre-losses-migration** bytes of the accelerate2_ebs2 smoke config — the very
  file task 1.3 will edit a second time. Adding `observability.steps` keeps
  `live != frozen` true, so the historicized refusal proof keeps running and
  keeps passing, and the mechanism suite stays skipped. **Confirmed: survives a
  second migration.**
- **parity FROZEN_V3** (`tests/qwen/test_packed_parity.py` L1496-1531): the
  suite's own comment records that the unmutated baseline "already drifts … pass
  vacuously" and the unconditional `qwen.parity.runtime_config_drift` refusal is
  asserted directly. Same structure, same outcome.
- **wave3 zero-weight** (`tests/losses/test_wave3_zero_weight_probe_contract.py`)
  and **wave7 bundle** (`tests/training/test_wave7_exact_resume_config_bundle.py`)
  assert drift *refusals* (`ConfigBundleError: identity drifted`) against
  fixture-copied bytes rather than pinning a live supported config's sha, so a
  new required field cannot re-break them.

Resolved-config fingerprint consumers: the exact-resume semantic projection
(`src/artifacts/training_state.py::build_resume_compatibility_projection`,
L161-165) and the three-run attestation projection
(`src/training/input_attestation.py` L108-119) are **denylists** excluding only
`{"run", "resume"}` — a new top-level `observability` block therefore enters
`semantic_config`. This does **not** break at Wave 1, because both projections
compare parent against child and task 1.3 migrates every supported config
symmetrically. It becomes load-bearing only when presentation cadence differs
between parent and child, which is exactly what task 5.1 exists to permit.

### 2.2 (b) Determinant payloads — PASS (stop rule holds)

`src/training/pack_cache.py::build_packing_cache_determinants` (L262-340) builds
a **closed, explicitly enumerated** semantic payload: `version`, `split`,
`dataset`, `template`, `packing`, `processor`, `ordering`, `augmentation`,
`qwen`, `realized_vocab_groups`, `micro_step_runtime_config`,
`micro_step_schema`. There is no whole-config dump and no `getattr`-style
traversal. `src/training/cache_contract.py::micro_step_runtime_config_identity`
returns a hard-coded three-key projection
(`training.precision`, `model.fa2_branch_proof` twice). **`observability.*`
cannot enter any determinant payload; the stop rule holds.**

One adjacency worth pinning (see `P2-CACHE-OWNERS`): the aggregate fingerprint
also folds `owner_source_identity.sha256` for each of the 31 files in
`PACKING_CACHE_DETERMINANT_OWNERS` (`_build_determinant_entries` L1618-1636 →
`_registry_entries_fingerprint` L1669-1676 → `aggregate_fingerprint` →
`_determinant_fingerprint`). Three of those owners sit near this change's work
surface: `src/training/micro_steps.py` (micro_step_schema),
`src/training/cache_contract.py` (micro_step_runtime_config), and
`src/training/pack_cache.py` (cache_serializer). Editing any of them changes the
cache fingerprint. None of the five declared observability owners is on the list,
and the physical-token/atom/pack counts task 3.4 needs already live in
`src/runtime/train_runtime.py` (grep confirms zero `physical_token`/`pack_count`
references in `src/training/micro_steps.py`), so the plan is safe today.

### 2.3 (c) Wave-2 reduction migration — one P1

`src/runtime/train_runtime.py::_reduce_metric_reports` (L584-778) dispatches in
this order:

| # | Branch | Typed-model target |
| --- | --- | --- |
| 1 | `_ACCURACY_METRIC_CORRECT_FIELDS` → pooled `_reduce_accuracy_stats` + `_accuracy_ratio` | ratio (exact int correct / int atoms) — clean map |
| 2 | `not replicated_objective and _is_planned_step_objective_metric_key` → sum | `SUM` — clean map |
| 3 | `_MAX_REDUCED_METRIC_KEYS` | `MAX` — clean map |
| 4 | `eval_sharded and _is_eval_sum_metric_key` | `SUM` / pre-weighted ratio — clean map |
| 5 | `eval_sharded and _is_eval_identical_metric_key` | `IDENTICAL` — clean map |
| 6 | `eval_sharded and key.startswith("finite/")` → `min` | `BOOL_ALL` — clean map |
| 7 | **else → plain mean over world size** | **no typed equivalent exists** |

`_is_replicated_eval_reduction` (L899-910, `split == "eval" and reduction_mode is
None`) is a single shared predicate and maps cleanly onto a per-batch mode flag.
Branches 1-6 all map onto {SUM, MAX, IDENTICAL, BOOL_ALL, ratio} without loss.

Branch 7 is the mismatch the design did not anticipate. Because the typed model
has **no MEAN reducer and no fallback**, every key currently landing there must
be given an explicit reducer, and that is a value-changing decision no task owns.
Three families land there today:

1. train `count/packs` and `count/examples` — mean-reduced, so a distributed
   train row under-reports them by the world size. Documented in the code's own
   TODO (L754-767) as a pre-existing defect introduced by `2b0a2165a`.
2. train `loss/<term>/token_weighted_diag` — mean-reduced, i.e. **unweighted**,
   while sharded eval computes the exact count-weighted average. Also in the
   TODO.
3. train `finite/*` keys — **not named in the TODO**. `src/losses/runner.py`
   L457-462 and L544-548 emit `finite/total_loss` and `finite/<term>` on the
   train path, where `reduction_mode is None` and `split == "train"`, so
   branch 6 is not reached and they fall to the mean. A two-rank step split
   finite/non-finite reduces to **0.5** — precisely the "fractional,
   meaningless value" the eval branch's own comment (L822-830) exists to
   prevent, on the rows that carry the objective's finite status.

Answering the design-conflict question directly: the "preserving full-row
equivalence" clause is **eval-scoped** (task 2.4 and the design's first risk
bullet, which speak of replicated vs sharded eval), so fixing the train defects
does **not** conflict with it. But the fixes change distributed train telemetry
values, so they require their own RED/sensitivity evidence and an explicit task.

A fourth, quieter consequence: replicated-eval objective keys reach branch 7
today and are averaged. Under the typed model they become `IDENTICAL`, which
additionally **rejects** any last-ULP cross-rank divergence instead of averaging
it — converting a previously silent tolerance into a run failure. Wave 2 must
decide exact-equality vs bounded tolerance explicitly.

### 2.4 (d) Exact-resume comparators and wave order — safe, with a named trap

`scripts/probes/coordexp_swift/wave7_exact_resume_compare.py::_projection_for_log`
(L2003-2018) is an **exclusion denylist**: it keeps every row key except
`TIMING_FIELDS`, `per_rank_measurement`, and the `resource/` prefix, and
compares the remainder strictly. `TIMING_FIELDS` (L205-207) is exactly
`{step_duration_seconds, input_build_seconds, input_wait_seconds}`. v2's
`_semantic_log_projection` uses the same shape with the broader
`_is_observational_timing` predicate (L787-794: `TIMING_FIELDS`,
`eval_duration_seconds`, `*_duration_seconds`, `*_wall_seconds`, `time/*`).

Consequence: every Wave-3 field outside those sets becomes a **strict semantic
comparison field** from the moment Wave 3 lands until task 5.2 reclassifies it —
`grad_norm/pre_clip_rank_max`, `lr/group_<index>`, throughput fields, the
loss weight/denominator/count family, allocator counters not under `resource/`,
`unavailable_fields`, and the terminal-receipt fields.

**The wave order is nevertheless safe.** No planned gate exercises the comparator
inside that window: task 3.8's GPU probes do not invoke it; the comparator's own
suites use hand-built fixture rows (`_train_row`/`_eval_row`,
`tests/training/test_wave7_exact_resume_compare.py` L271-299), so they will not
go red on new production fields; and 5.1/5.2 precede 5.4/5.5 within Wave 5.
Wave-1 config migration alone does not break the comparators (see 2.1). This is
therefore a P2 trap, not a P1 sequencing failure.

The specific trap 5.2 must not miss: **`input_h2d_seconds` is a timing field
that evades both exclusion predicates** — it is not in v1 `TIMING_FIELDS`, and
it ends in `_seconds` but not `_duration_seconds`/`_wall_seconds`, so v2's
suffix predicate misses it too. 5.2's disposition must enumerate it explicitly
alongside the other new fields rather than relying on a naming convention.

---

## 3. Findings

| ID | Sev | Area | Description | Disposition |
| --- | --- | --- | --- | --- |
| `P1-REDUCE` | **P1** | intent (c) / tasks.md | The typed reducer set {SUM, MAX, IDENTICAL, BOOL_ALL, ratio} has no MEAN and no fallback, but four families currently reach `_reduce_metric_reports`' plain-mean branch with no task owning their reclassification: train `count/packs`+`count/examples`; train `loss/<term>/token_weighted_diag` (unweighted); **train `finite/*`** (0.5 on a 2-rank split — not in the code's TODO, newly found here); and replicated-eval objective keys, whose MEAN→IDENTICAL move converts silent last-ULP tolerance into a run failure. | Amend `tasks.md` before Wave 1: add a Wave-2 task that (i) assigns SUM to the two train counts, (ii) assigns the exact count-weighted ratio to train `token_weighted_diag`, (iii) assigns `BOOL_ALL` to train `finite/*`, (iv) records an explicit tolerance decision for replicated-eval `IDENTICAL`, and (v) requires RED/sensitivity evidence for each value-changing fix. Record in the task that the design's "full-row equivalence" clause is eval-scoped and does not forbid these train fixes. |
| `P2-COMPARATOR` | P2 | intent (d) / tasks 3.x-5.2 | Comparator log projections are exclusion denylists; every Wave-3 field outside `TIMING_FIELDS`/`per_rank_measurement`/`resource/*` is strictly compared for four waves until 5.2 lands. No planned gate hits it, but `input_h2d_seconds` evades v1 `TIMING_FIELDS` **and** v2's `_is_observational_timing` suffix predicate. | Extend task 5.2 to enumerate the new non-semantic fields by name — `input_h2d_seconds` first — rather than relying on suffix conventions; add a note in Wave 3 that the comparator is knowingly wrong for real runs until 5.2. |
| `P2-MANIFEST-W1` | P2 | standards / manifest | `wave1-config-gate` is only `tests/config tests/losses`, but task 1.4 edits shared config fixture builders consumed by `tests/training`, `tests/runtime`, `tests/artifacts`, `tests/eval`. A fixture-builder change can go red outside the gate. | Append-only amendment widening `wave1-config-gate` to include at least `tests/training tests/runtime tests/artifacts`. |
| `P2-MANIFEST-RESIDUE` | P2 | standards / manifest | Tasks 1.5, 2.7, 4.8, 5.6 all require residue searches and the manifest's `expected` strings reference them, but **no residue-search argv is frozen**. Under the append-only rule they cannot be run as gate evidence until amended in. | Amend the manifest with the explicit residue-search argvs (observability default / logging alias; implicit reducer / dynamic registry; event bus / registry / alternate scalar file / DB / W&B / per-rank stream; scheduler-derived LR path; unbounded sink error state). |
| `P2-MANIFEST-W4W5` | P2 | standards / manifest | `wave4-publisher-gate` names no path for task 4.5's TensorBoard event-reader suite and omits `tests/runtime` although task 4.7 deletes superseded helpers that live in `src/runtime/train_runtime.py`. `wave5-resume-gate` omits `tests/training/test_input_attestation.py`, although task 5.1 changes the three-run input-attestation projection in `src/training/input_attestation.py`. | Amend both entries: add the reader-suite path once 4.5 fixes it, add `tests/runtime` to wave 4, add `tests/training/test_input_attestation.py` to wave 5. |
| `P2-CACHE-OWNERS` | P2 | standards / manifest cache invariant | The invariant says the observability surfaces "are not determinant owners" (true) but never names the 31 files whose byte-sha feeds `aggregate_fingerprint`. Three sit near Wave-3 work: `src/training/micro_steps.py`, `src/training/cache_contract.py`, `src/training/pack_cache.py`. | Amend `cache_invariant` with an explicit no-edit file list naming those three; note that task 3.4's work counts already live in `src/runtime/train_runtime.py`, so no edit is needed today. |
| `P3-CADENCE-NARROW` | P3 | standards / rebase | The config-runtime scenario bullet narrowed "no authored logging cadence may suppress that row" to "`observability.steps` MUST NOT suppress or sample that row". | No action. The requirement prose still forbids any schema cadence that samples or suppresses canonical logging, and the `Legacy cadence alias authored` scenario is retained; coverage is intact. |
| `P3-RELAXATIONS` | P3 | standards / archaeology | All three deliberate modifications are genuine relaxations of stable text (acknowledged corruption; synchronized skips; barrier-only timing prohibition), verified as authored intent rather than stale-base survivals. | Informational. Carry them forward as known scope for the Wave-3/5 audits, since each is load-bearing for design decisions 3 and 4. |

**Counts: P0 = 0, P1 = 1, P2 = 5, P3 = 2.**

---

## 4. Entry statement

Wave 1 **may not begin at the frozen base**: one P1 (`P1-REDUCE`) is
unresolved, and the plan's rule is that no unresolved P0/P1 may precede
implementation. The blocker is a planning gap, not a defect in the frozen
artifacts — the rebase, the archaeology, the owner graph, the determinant stop
rule, the historicized guards, and the 890/0/0 baseline are all clean.

Wave 1 is cleared the moment the Wave-0 close-out commit lands the `tasks.md`
amendment assigning the reducer reclassification, together with the append-only
manifest amendments listed below. No re-audit is required for the P2s
individually; the lead may accept them as recorded dispositions.

### The Wave-0 close-out commit MUST include

1. The three rebased delta specs (currently tracked-modified, unchanged since
   `3d390b108`).
2. `receipts/wave-0-baseline.md`.
3. `receipts/command-manifest.json`.
4. `receipts/wave-0-entry-audit.md` (this file).
5. An **append-only manifest amendment pinning the `wave0-entry-baseline`
   counts at 890/0/0** — the entry's own `expected` text says "counts pinned by
   amendment" and `amendments` is still `[]`.
6. Manifest amendments for `P2-MANIFEST-W1`, `P2-MANIFEST-RESIDUE`,
   `P2-MANIFEST-W4W5`, and `P2-CACHE-OWNERS`.
7. A `tasks.md` amendment resolving `P1-REDUCE` (the Wave-2 reducer
   reclassification task with its RED/sensitivity requirement).
8. Tasks 0.1-0.5 checked off in `tasks.md`.

---

## 5. Commands and observed output

All commands ran from `/data/CoordExp/.worktrees/CoordExp-swift`. The gate
evidence — the manifest baseline replay (row 4) and the two validations (rows
2-3) — ran inside `bash -c` export wrappers: the baseline exported
`PYTHONDONTWRITEBYTECODE=1` and unset all nine `required_absent` variables; the
validation script exported `PYTHONDONTWRITEBYTECODE=1` (the nine variables are
not inputs to `openspec validate`). The remaining rows are unwrapped read-only
analysis commands (`git`, `grep`, `ls`, `wc`, `python3` on scratchpad scripts)
and are not gate evidence.
No `env`-prefixed invocation was used; no heredoc was piped into `conda run`
(analysis scripts were written to files and invoked by path). Every result below
is real observed output, never an exit code alone. No command touched
`.cache/coordexp_swift/packing`; no GPU action was taken. No hang occurred; the
longest command was 98 s.

| # | Command | Observed |
| --- | --- | --- |
| 1 | `git rev-parse HEAD; git status --porcelain` (start and again immediately before this write) | `3d390b108db5c666a7c2bd2f7d79b12237de81be`; exactly the three ` M` delta spec paths; untracked `receipts/` with `command-manifest.json` and `wave-0-baseline.md` |
| 2 | `conda run -n ms openspec validate --all` | `Totals: 20 passed, 0 failed (20 items)`, including `✓ change/add-coordexp-swift-training-observability` |
| 3 | `conda run -n ms openspec validate add-coordexp-swift-training-observability --strict` | `Change 'add-coordexp-swift-training-observability' is valid` |
| 4 | `conda run -n ms pytest tests/config tests/losses tests/runtime tests/training/test_reporting.py tests/artifacts tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py -q` (manifest `wave0-entry-baseline`, verbatim) | `890 passed, 7 warnings in 98.19s (0:01:38)` — 890/0/0 |
| 5 | `ls openspec/changes/archive/` | all three predecessor archives present; `add-coordexp-swift-training-observability` is the only active change |
| 6 | scenario/prose diff scripts (`scendiff.py`, `scendiff2.py`, `prose.py` under the session scratchpad `.../scratchpad/audit/`) | zero stable scenarios missing across all five MODIFIED requirements; every non-verbatim stable sentence individually resolved as preserved or strengthened |
| 7 | `git log --format='%h %ad %s' --date=short -- openspec/changes/add-coordexp-swift-training-observability/` | single commit `0b98f0561 2026-08-12 docs(training): plan infrastructure refinement` |
| 8 | `git show 0b98f0561:…/specs/{coordexp-swift-supervision-losses,coordexp-swift-training-artifacts}/spec.md \| grep` for the three phrases | "unacknowledged corrupted" and "or synchronized skips" present in the losses blob (L7-8); the full timing paragraph incl. "per-step barrier solely for observation" present in the artifacts blob (L188-202), byte-identical to the rebased text |
| 9 | `git diff -U0 -- …/specs/` | 91 insertions / 19 deletions; every deleted line individually accounted for (rebase marker, reflow, replaced loss paragraph, reworded scenario bullet) |
| 10 | `ls src/runtime/metrics.py src/artifacts/observation_publisher.py` | both `No such file or directory` — to-be-created owners correctly absent |
| 11 | `grep -rn "SummaryWriter\|tensorboard" src/` | no output — zero TensorBoard usage anywhere in `src/` |
| 12 | `grep -n "SummaryWriter\|tensorboard\|console\|print(" src/training/pipeline.py` | no output — facade is not a competing owner |
| 13 | `wc -l src/training/reporting.py src/training/session.py src/runtime/train_runtime.py` | 368 / 3534 / 1165 |
| 14 | reads of `src/training/pack_cache.py` L255-340, L1618-1676, L1777-1782; `src/training/cache_contract.py` | determinant payload is a closed enumerated allow-list; `micro_step_runtime_config_identity` is a 3-key projection; `observability.*` cannot enter |
| 15 | `python3 -c` dump of `PACKING_CACHE_DETERMINANT_OWNERS` | 31 owner source files; none of the five observability owners; `micro_steps.py`, `cache_contract.py`, `pack_cache.py` present |
| 16 | reads of `src/runtime/train_runtime.py` L26-80, L584-860, L899-911 | seven-branch reducer dispatch confirmed; plain-mean fallback at L768-778; TODO at L754-767 names two of the four affected families |
| 17 | `grep -rn '"finite/' src/` | `src/losses/runner.py` L457, L462, L544, L548 emit `finite/*` on both train and eval paths — confirming the unnamed train mean defect |
| 18 | reads of `tests/training/test_reconcile_exact_resume_packet_executor{,_historicized}.py`, `tests/qwen/test_packed_parity.py` L1441-1531, `tests/training/test_wave7_exact_resume_config_bundle.py`, `tests/losses/test_wave3_zero_weight_probe_contract.py` | all four historicized families use live-computed drift predicates; a second migration keeps them in the already-drifted branch |
| 19 | reads of `src/artifacts/training_state.py` L132-178, `src/training/input_attestation.py` L103-119 | both projections are `{"run","resume"}` denylists; a new `observability` block enters `semantic_config` but Wave-1 migration is symmetric |
| 20 | reads of `scripts/probes/coordexp_swift/wave7_exact_resume_compare.py` L194-207, L2003-2033 and `…_compare_v2.py` L780-830 | exclusion denylists confirmed; `TIMING_FIELDS` is three keys; `_is_observational_timing` misses `input_h2d_seconds` |
| 21 | `ls tests/training \| grep -i attest` | `test_input_attestation.py` exists and is absent from `wave5-resume-gate` |
