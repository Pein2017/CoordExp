# Wave-5 Final Audit — standards/code-quality + intent/spec-contract

- **change**: `add-coordexp-swift-training-observability`, task 5.7
- **date**: 2026-08-20
- **identity**: Opus observability final auditor, spawned by the Claude Fable
  lead. Distinct from the lead, from every wave builder, and from the Wave-0
  entry auditor and Wave-3 pre-DDP auditor.
- **frozen target**: HEAD `c4d4dc857`, tracked tree clean. Verified commit
  chain with nothing interleaved: `51cc48de6` (W0) → `f410a4bfa` (W1) →
  `9f9dbf1ba` (W2) → `bdb29b3fa` / `4db58e972` / `305b17eb7` / `e81f6a91a` /
  `67efb5372` (W3) → `4426d03b7` (W4) → `30b563e6c` / `c4d4dc857` (W5).
- **scope**: read-only. The only file written is this receipt.
- **method**: re-derivation, not receipt trust. Every quantitative claim below
  was recomputed by this auditor from source bytes, on-disk artifacts, or a
  fresh command execution.

## Verdicts

| # | Dimension | Verdict |
| --- | --- | --- |
| 1 | STANDARDS / CODE-QUALITY | **PASS-WITH-DISPOSITIONS** |
| 2 | INTENT / SPEC-CONTRACT | **PASS-WITH-DISPOSITIONS** |

**Severity counts: 0 P0 / 0 P1 / 2 P2 / 3 P3.**

Both dispositions are additive close-out obligations, not defects in shipped
behavior. Nothing found contradicts an executed probe, an emitted artifact, or
a published claim.

## Completion gate

**CLEARED — the change may be marked complete and archived with spec-delta
sync, contingent on the must-include list below.** No P0 or P1 finding exists.
Archive mechanics were independently re-verified at `c4d4dc857`: every
`MODIFIED` requirement in all three deltas carries the complete stable
scenario list, so the archive tool's fail-closed scenario check will not trip.

### Must-include list for the final close-out commit

1. **`amend-8` in `receipts/command-manifest.json`** (append-only, per the
   manifest's own `amendment_rule` and the `amend-4`…`amend-7` per-wave
   precedent). It must: retire the `wave5-vertical-smoke`
   `TO-FREEZE-IN-WAVE-5-PACKET` placeholder against
   `receipts/wave-5-smoke-packet.md`; pin `wave5-resume-gate` at 1357/0/0 and
   `wave5-broad-regression` at 2565 passed / 126 skipped / 0 failed; and
   record the four Wave-5 declared re-pins (v1 comparator
   `4698eeca…`, v2 comparator `aaac6aee…`, `input_attestation.py`
   `351f2495…`, and the paired seals in `wave7_exact_resume_sequence.py`),
   which currently carry in-place declared-flip comments but no manifest
   record. Carry the two retained load-bearing fallbacks forward. (Resolves
   P2-MANIFEST-W5.)
2. **`docs/ARTIFACTS.md:81`** — move "pre-clip gradient norm" out of the `MAX`
   list into the `IDENTICAL` list (or state the two-stage semantics
   explicitly). (Resolves P2-DOCS-REDUCER.)
3. **`tasks.md`** — check 5.1–5.7 and add the Wave-5 close-out block naming
   `30b563e6c`, `c4d4dc857`, and the close-out commit; the observed smoke
   bounds; the gate counts; and both audit verdicts.
4. **Reference `receipts/wave-5-final-audit.md`** (this file) from that
   close-out block.
5. **Re-run `openspec validate … --strict`** after the `tasks.md` and docs
   edits; both edits are prose-only and must not perturb validation.
6. **Archive with spec-delta sync.** Archive mechanics are clean as of
   `c4d4dc857`; re-run the scenario diff if any delta file is touched again.

## Findings

| id | sev | dimension | finding | disposition |
| --- | --- | --- | --- | --- |
| P2-MANIFEST-W5 | P2 | standards | The frozen command manifest still carries the `TO-FREEZE-IN-WAVE-5-PACKET` placeholder for `wave5-vertical-smoke`; `wave5-resume-gate` and `wave5-broad-regression` counts are unpinned; and the four Wave-5 declared sha re-pins have in-place source comments but no manifest amendment. Waves 1–4 each pinned their gate in that wave's close-out commit (`amend-4`…`amend-7`); Wave 5's close-out has not been written yet. | Must-include #1. Not a behavioral defect: the packet (`receipts/wave-5-smoke-packet.md`) and receipt are committed at `c4d4dc857`, and every re-pin is self-consistent (recomputed below). |
| P2-DOCS-REDUCER | P2 | intent (f) | `docs/ARTIFACTS.md:81` attributes the `MAX` reducer to "pre-clip gradient norm", but `src/training/reporting.py:576` declares `grad_norm/pre_clip_rank_max` with `REDUCER_IDENTICAL` — deliberately, per the code comment immediately above it ("value-preserving and fails closed on a divergence instead of hiding it under a second maximum") and per the Wave-3 close-out ("IDENTICAL-not-MAX for already-global grad_norm/lr"). Task 5.3 requires the docs to state canonical metric **reducers**, so this is a doc-accuracy defect on a required deliverable. | Must-include #2. Blast radius bounded: the row value is a genuine rank maximum (computed upstream in the gradient gather), so no published number is wrong; only the reducer attribution at the declaration boundary is. A maintainer misled by it would declare `MAX` for an already-global value — value-preserving, but it would forfeit the fail-closed divergence check. |
| P3-WEIGHT-NAME | P3 | intent (b) | Spec prose says computed terms expose "its configured weight"; the implemented key is `loss/<term>/weight`, not `loss/<term>/configured_weight`. | **No contract breach.** The delta and stable specs enumerate only `loss/<term>/raw` and `loss/<term>/weighted` as literal keys; "configured weight" is semantic prose. Recorded for precision only. |
| P3-RECEIPT-SHAPE | P3 | standards | `wave-5-smoke-receipt.json` renders `loss/base_ce` as a nested `{raw, weight, weighted}` object, while the canonical row uses flat `loss/base_ce/*` keys. | Values verified byte-exact against the row (see below). Presentation convenience in the receipt; could confuse a future reader diffing receipt against JSONL. No action required. |
| P3-W3-CARRIED | P3 | intent (g) | Wave-3 P3 residuals `W3-4` (single-level unwrap vs Accelerate's `while` loop) and `W3-5` (silent `False` on an unshaped state entry) remain open, recorded in `amend-6`. | Already recorded and dispositioned; not re-opened by this audit. |

### Host note (not a finding)

An unscoped `rg` over this worktree hangs (>2 min, killed) descending into
`outputs/`. Every search in this audit was scoped to `src/ scripts/ tests/
configs/ docs/` or to named files. Recommended for future audits.

## 1. Standards / code-quality — re-derivations

### 1.1 OpenSpec validation (both forms, re-run by this auditor)

```
$ conda run -n ms openspec validate add-coordexp-swift-training-observability --strict
Change 'add-coordexp-swift-training-observability' is valid
EXIT=0

$ conda run -n ms openspec validate --all
✓ change/add-coordexp-swift-training-observability
✓ spec/coordexp-swift-adapters-embeddings-optim
✓ spec/coordexp-swift-config-runtime
✓ spec/coordexp-swift-data-template-encoding
✓ spec/coordexp-swift-detection-evaluator
✓ spec/coordexp-swift-geometry-augmentation
✓ spec/coordexp-swift-infer-backend-trace
✓ spec/coordexp-swift-infer-benchmark-smoke
✓ spec/coordexp-swift-infer-config-runtime
✓ spec/coordexp-swift-infer-data-parallel-runtime
✓ spec/coordexp-swift-infer-execution-model
✓ spec/coordexp-swift-infer-pipeline
✓ spec/coordexp-swift-infer-prompt-parsing
✓ spec/coordexp-swift-infer-scoring-artifacts
✓ spec/coordexp-swift-pack-cache-semantic-identity
✓ spec/coordexp-swift-packing-forward
✓ spec/coordexp-swift-supervision-losses
✓ spec/coordexp-swift-training-artifacts
✓ spec/coordexp-swift-training-resume
✓ spec/coordexp-swift-vertical-smoke
Totals: 20 passed, 0 failed (20 items)
EXIT=0
```

### 1.2 Lead junit cross-check + independent frozen-argv replays

Parsed the lead's XML directly (sum over `<testsuite>` attributes):

| artifact | tests | failures | errors | skipped | lead's claim | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `obs_w54.xml` | 1357 | 0 | 0 | 0 | 1357/0/0 | **matches** |
| `obs_w56.xml` | 2691 | 0 | 0 | 126 | 2691/0/126skip | **matches** |

Companion logs agree (`1357 passed`; `2565 passed, 126 skipped`).

Two cheap frozen argvs replayed by this auditor at `c4d4dc857`:

```
$ conda run -n ms python -m pytest tests/artifacts tests/training/test_reporting.py -q
441 passed in 23.92s

$ conda run -n ms python -m pytest tests/training/test_orchestration_compatibility.py \
      tests/runtime/test_rank_report_collective.py -q
15 passed in 34.26s
```

15/15 fixtures/collective matches every prior wave gate.

### 1.3 Determinant invariant — re-derived, not trusted

Executed the redirected archived-baseline probe
(`run_determinant_probe_archived.py`, which redirects only `BASELINE_PATH` and
uses all probe logic as authored) at `c4d4dc857`:

```
"commit": "c4d4dc85766880641ff821a6e8a5cb413d782486", "tree_dirty": false,
"findings": [], "status": "OK", "wall_seconds": 1.303
train      : baseline 8f11237f… recomputed 8f11237f… fingerprint_equal=true payload_equal=true verdict=EQUAL
eval.forward: baseline 3b30c157… recomputed 3b30c157… fingerprint_equal=true payload_equal=true verdict=EQUAL
PROBE_EXIT=0
```

Both target fingerprints re-derive EQUAL with byte-equal payloads. The
observability surface enters no determinant. (The probe's own `"change"` and
`"task"` metadata still name the archived losses change — it is run unmodified
by design, per `amend-4`.)

### 1.4 The 5.5 smoke receipt vs on-disk reality

Read `outputs/smoke/obs_wave5_smoke_r1/obs_wave5_smoke/logging.jsonl` bytes
directly. **Exactly 2 rows** (train step 1, 62 keys; eval step 1, 37 keys).
Every receipt row value re-checked against the actual bytes:

```
OK  count/physical_tokens                         16494.0
OK  grad_norm/pre_clip_rank_max                   50.098120633612425
OK  input_h2d_seconds                             0.019167743682861327
OK  lr/group_0                                    0.0002
OK  optimizer_boundary_action                     apply
OK  optimizer_mutation_state                      applied
OK  optimizer_step_count                          1
OK  optimizer_update_applied                      True
OK  resource/gpu_current_memory_allocated_bytes   4510186496.0
OK  scheduler_step_count                          1
OK  throughput/physical_tokens_per_second         5012.448166201044
OK  unavailable_fields  ['resource/gpu_alloc_retries_delta', 'resource/gpu_ooms_delta']
```

Zero mismatches. `loss/base_ce/{raw,weight,weighted}` = `5.013396978378296 /
1.0 / 5.013396978378296`; `loss/total` equals the sum of weighted terms.
`non_finite_fields` empty. The zero-weight `token_type_gate` term is **present
with `weight=0.0`, `weighted=0.0`, and full raw/count/finite diagnostics** —
correct: it is a *computed* zero-weight gate (retained by task 3.3), not an
omitted optional term. Availability honesty confirmed: the two allocator-delta
fields are named in `unavailable_fields` rather than published as fabricated
zeros, matching the receipt's "first observed step has no allocator delta
baseline" note.

**TensorBoard, read with the real reader** (`EventAccumulator`):

```
tb files: [('events.out.tfevents.1787267463.k8s-worker02.3665716.0', 5293)]   # exactly one
SCALAR TAG COUNT: 77          # receipt claims 77
train tags: 48   eval tags: 29   other-prefix tags: []
non-finite scalars: []
global_steps observed: [1]
train/loss/total: [(1, 5.0134)]   # receipt claims exactly this
```

Both `train/` and `eval/` tag families present, finite-only, canonical planned
step as `global_step`, deterministic `<split>/<canonical-key>` tags.

**Single shared run tree**: parent `obs_wave5_smoke_r1/` contains exactly
`['obs_wave5_smoke']`; run dir contains exactly `['checkpoints',
'logging.jsonl', 'resolved_config.json', 'run.json', 'tensorboard']`. No
rank-local run or event trees.

**Packet bounds vs observed** — all honoured:

| bound | limit | observed | source |
| --- | --- | --- | --- |
| devices / world size | 2 GPUs / ws=2 | `CUDA_VISIBLE_DEVICES=0,1`, ws=2 | receipt + argv |
| planned steps | exactly 1 planned + applied | `completed_steps: 1`, `optimizer_step_count: 1`, action `apply` | row + `run.json` |
| cache / materialization passes | REQUIRED 0 | 0 | pre/post sha inventory |
| wall time | ≤ 900 s | ~24 s (`created_at` 23:10:42 → `completed_at` 23:11:05) | `run.json` |
| peak GPU per rank | ≤ 32 GiB | 9.36 GB (`gpu_max_memory_allocated_bytes` 9356427264) | row |
| CPU RSS per rank | ≤ 16 GiB | 8.34 GiB (`cpu_max_rss_bytes` 8959954944) | row |
| artifact bytes | ≤ 2.5 GiB | **45717812 (re-walked on disk; exactly the receipt figure)** | `os.walk` |

Cache invariant re-verified independently: `cmp obsw5-pre-cache.sha
obsw5-post-cache.sha` → **identical (exit 0)**, 6 entries. Both predecessor
fingerprints appear in `run.json` `materializations` as admitted
(`8f11237f…` train, `3b30c157…` eval). Free disk at launch is not
re-derivable post hoc — accepted as attested by the packet.

The receipt makes **no throughput or model-quality claim**
(`no_throughput_or_quality_claim: true`); the `throughput/*` fields are
published row values, not claims. Correct.

### 1.5 Manifest append-only across all amendments

Reconstructed every committed version of `command-manifest.json` from git and
diffed consecutively:

```
51cc48de6: amendments=3  [amend-1, amend-2, amend-3]
f410a4bfa: amendments=4  append-only OK (+1)   [+amend-4-pin-wave1-gate]
9f9dbf1ba: amendments=5  append-only OK (+1)   [+amend-5-freeze-wave2-gloo-probe]
67efb5372: amendments=6  append-only OK (+1)   [+amend-6-freeze-wave3-probes]
4426d03b7: amendments=7  append-only OK (+1)   [+amend-7-pin-wave4-gate]
```

No prior amendment object was mutated (byte-compared under `sort_keys`); no
command entry was removed or edited; `amendment_rule`, `cache_invariant`, and
`frozen_at_commit` never changed. **Append-only holds strictly.** The Wave-5
gap is the *absence* of `amend-8`, not a violation (P2-MANIFEST-W5).

### 1.6 Wave close-outs match commits

Each close-out block's named commits exist in the chain and their diffstats
match the described work. `30b563e6c` touches exactly the 4 docs, 3 comparator
scripts, 2 src files, and 6 test files described; `c4d4dc857` adds only the
smoke packet and receipt (125 insertions, 0 deletions). Wave 5's close-out
block is correctly still unwritten — tasks 5.1–5.7 remain unchecked because
5.7 is this audit.

### 1.7 Wave-5 comparator re-pins

All four re-pins recomputed from current source bytes:

```
OK  scripts/probes/coordexp_swift/wave7_exact_resume_compare.py
      4698eeca1439bf65d55d524acf9d780f6078251756e792258172d3c79147c8d2
OK  scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py
      aaac6aeecc825773e1b5d93827a61474581dd8fa894f400713719c2628cada66
OK  src/training/input_attestation.py
      351f24953a9d55d80f3a1d5476c3171e30a80b0588ccf97adb0bca35c768b1fc
```

Each re-pin carries an in-place declared-flip comment naming the previous
seal, the owning task, and an explicit "nothing that was compared as training
semantics stopped being compared" assertion. The comments are substantive and
correct. **Assessment: the close-out commit MUST add the manifest amendment**
(must-include #1) — the manifest is the frozen authority and its own rule
requires an appended entry for any later command change; four seals moved with
no manifest record. This is a receipt-completeness obligation, not a
correctness defect.

## 2. Intent / spec-contract — verdict per dimension

### (a) `observability.steps` presentation-only — **PASS**

*Cannot suppress or sample the canonical row stream.* Structurally proven in
`src/artifacts/observation_publisher.py::publish`: the unconditional
`_append_logging_row_shared(...)` call is at line 266; the cadence decision
`_presentation_due(...)` is at line 270. The interval can only trigger an
early `return` **after** the append has completed. Identity is validated on
every rank *before* the collective (line 263-265) so a rank-local raise cannot
desynchronize it.

Behavioral proof in
`tests/artifacts/test_observation_publisher.py::test_train_rows_are_presented_only_on_the_configured_interval`
(steps=2, total=10, publish steps 1–4):

```python
assert [update.step for update in publisher._presented] == [2, 4]
# Every canonical row is appended regardless of the presentation cadence.
assert [row["step"] for row in _rows(writer)] == [1, 2, 3, 4]
assert {step for _, _, step in board.scalars} == {2, 4}
```

Config surface: `ObservabilityConfig` is required with no default
(`src/config/models.py:539`); the interval is composed once at
`src/training/session.py:2322`. It reaches no other consumer.

*Exclusions accept presentation drift, semantic drift still fails closed.*
`_RESUME_NON_SEMANTIC_CONFIG_BLOCKS = frozenset({"observability", "resume",
"run"})` (`src/artifacts/training_state.py:146`) and
`_NON_SEMANTIC_CONFIG_BLOCKS = frozenset({"observability", "resume", "run"})`
(`src/training/input_attestation.py:49`), both carrying declared-flip
comments. Paired tests added in `30b563e6c`:
`test_resume_compatibility_excludes_the_presentation_only_observability_block`,
`test_admission_accepts_presentation_only_drift_and_still_rejects_lr_drift`,
`test_accepts_presentation_only_observability_drift_across_the_three_runs`.
Comparator side:
`test_v2_still_rejects_lr_update_counter_and_loss_weight_drift`. Loss, LR,
norm, and update-status drift remain strict.

*Task-5.2 terminal no-publication is proven behaviorally, not only in prose.*
`30b563e6c` adds two tests to `tests/training/test_training_session.py`:
`test_terminal_boundary_publishes_no_state_checkpoint_selector_or_final`
(no exact-resume state, checkpoint, best-selector update, or successful final
artifact follows a terminal boundary) and
`test_terminal_boundary_finalizes_run_json_failed_at_the_planned_step`
(`run.json` records failed status at the current planned-step id without
counting it as completed). Both executed **green** inside the 2691-test broad
regression (`tests/training` is in the `wave5-broad-regression` argv;
`obs_w56.xml` reports 0 failures / 0 errors). Comparator-side
classification completeness is likewise behavioral:
`test_every_new_producer_owned_field_is_classified_by_name`,
`test_v2_classifies_every_new_producer_owned_field_by_name`, and
`test_diagnostic_name_lists_are_filtered_not_dropped` — so a future row field
cannot silently escape classification by evading a suffix convention, which
was the Wave-0 `P2-COMPARATOR` concern.

### (b) Typed reducers — **PASS**

Reducer vocabulary in `src/runtime/metrics.py` is exactly four members:
`REDUCER_SUM`, `REDUCER_MAX`, `REDUCER_IDENTICAL`, `REDUCER_BOOL_ALL`. The
module docstring records that no `MEAN` exists and that `ALL` is deliberately
absent (`BOOL_ALL` is the only spelling for boolean conjunction).
`IDENTICAL_ABS_TOLERANCE = 0.0` — exact equality, matching the Wave-2 gloo
probe's measured bitwise-0.0 divergence.

*No implicit mean, no name-convention selection.* An undeclared metric raises
`runtime.metric_reducer_undeclared` at four sites in `metrics.py`, asserted at
four test sites. Searches for `endswith(`-style reducer tables, `SUFFIX`
sets, `REDUCER_TABLE`, and `_reducer_for_name` across `metrics.py`,
`train_runtime.py`, and `reporting.py`: **zero hits**. Docstring is explicit:
"anything else fails closed rather than acquiring a default reducer."

*Producer-declared everywhere.* Every `ScalarSample` in `reporting.py` passes
an explicit `reducer=` (spot-verified across ~12 construction sites: timings
and resources `MAX`; work counts `SUM`; already-global `grad_norm` and `lr`
`IDENTICAL`; availability gate `BOOL_ALL`).

*The four Wave-0-amendment reclassifications landed as declared:*

1. `count/packs` + `count/examples` → `objective_reducer`
   (`metrics.py:915-916`), which is `REDUCER_SUM` when
   `partial_rank_contributions` (disjoint train/sharded work) and
   `REDUCER_IDENTICAL` when replicated (`metrics.py:939-941`). ✓
2. `loss/<term>/token_weighted_diag` → ratio sample weighted by
   `selected_count` (`metrics.py:957`), sum-before-divide. ✓
3. `finite/<term>` and `finite/total_loss` → `REDUCER_BOOL_ALL`
   (`metrics.py:907-908`). ✓
4. Replicated-eval objective keys → `IDENTICAL`, implemented EXACT
   (tolerance 0.0) after the probe measured zero divergence. ✓

Per-family RED receipts are recorded in the Wave-2 close-out and `amend-5`.

### (c) Optimizer boundary — **PASS**

*Receipts truthful per the 3.1/3.2 matrix.* Machine-checked the four fp16
receipts for embedded `findings`:

| receipt | `ok` | findings |
| --- | --- | --- |
| `wave-3-fp16-ws1-receipt.json` (attempt 1) | true | **1 — the `scaler_found_inf` FINDING** |
| `wave-3-fp16-ws2-receipt.json` (attempt 1) | true | **1 — the same FINDING** |
| `wave-3-fp16-ws1-r2-receipt.json` (attempt 2) | true | **0** |
| `wave-3-fp16-ws2-r2-receipt.json` (attempt 2) | true | **0** |

The attempt-1 FINDING text names the exact defect ("`Accelerator.unscale_gradients`
unwraps `AcceleratedOptimizer`, so `_scaler_found_inf(scaler, self.optimizer)`
looks up a defaultdict miss") and correctly states the converged action was
still right because the gradient scan is independent evidence. The real-CUDA
fix (`e81f6a91a`) is therefore evidenced by a genuine RED→GREEN pair on real
hardware, not a green-only claim. This is the strongest evidence in the change.

*No `accelerator.clip_grad_norm_`.* Scoped search over `src scripts tests
configs docs`: the only clip call in `src/` is
`torch.nn.utils.clip_grad_norm_` at `src/runtime/train_runtime.py:584` — the
required non-unscaling primitive. Every other hit is a prohibition assertion:
`tests/runtime/test_wave3_optimizer_boundary.py:111` and a source-scan test at
:466 that accepts only the `torch.nn.utils` spelling; both GPU probes assert
`"accelerator.clip_grad_norm_ is prohibited"`. `unscale_gradients` has exactly
one call site in `src/` (`_unscale_gradients_once`, train_runtime.py:556-567),
documented as "the ONLY place".

### (d) JSONL-first — **PASS**

*Append precedes presentation* — proven in (a): line 266 before line 270/275/276.
The `publish` docstring states the contract ("a derived sink may never describe
an observation that has no canonical row") and a failed publication presents
nothing (`test_failed_canonical_publication_presents_nothing`,
`test_the_canonical_row_is_readable_before_any_presentation_call`).

*Sink failures isolate.* `close()` latches `_tensorboard_disabled = True`
**before** the cleanup call so a close failure cannot recurse (lines 281-293).
Seven-arm failure-injection coverage exists
(`test_a_failed_tensorboard_start_keeps_the_canonical_row`,
`test_an_add_scalar_failure_latches_the_sink_and_keeps_later_rows`,
`test_a_flush_failure_after_a_successful_add_keeps_the_row`, …).

*No second scalar authority, event bus, per-rank streams, W&B/DB.* Scoped
searches over `src/`: `metrics.jsonl|scalars.json|per_rank_measurement|
logging_rank` → **zero hits**; `event_bus|wandb|sqlite|mlflow|log_rotation|
RotatingFileHandler|prometheus|statsd` → **zero hits in the training path**
(the only matches repo-wide are pre-existing vLLM *inference* qualification
receipts under `src/inference/qualification_receipts/`, unrelated to this
change). `tensorboard`/`SummaryWriter` appear in exactly one file:
`src/artifacts/observation_publisher.py`. A dedicated test
(`test_publisher_module_owns_no_event_dispatcher_surface`) asserts the module
source contains none of `wandb`, `sqlite`, `scalars.json`, `metrics.jsonl`
and exposes no `handlers`/`listeners` surface.

### (e) Probe-only exclusions honored — **PASS**

Scoped search for `mfu|tflop|joule|watt|energy` across `src/` and the four
canonical docs: **zero hits in `src/`**. The only two doc hits are explicit
*exclusion statements*, not supported claims:

- `docs/ARTIFACTS.md:106` — "MFU, TFLOPS, energy estimates, and per-rank
  metric traces are probe-only and stay disabled unless an explicitly
  calibrated probe requests them."
- `docs/COORDEXP_SWIFT.md:220-221` — same exclusion, phrased as "stay out of
  normal production rows."

No such field appears in the emitted smoke rows (62 train keys / 37 eval keys
enumerated; none throughput-derived beyond the three declared
`throughput/*` work rates, which are in-scope per task 3.4).

### (f) Docs describe-and-link — **PASS-WITH-DISPOSITION** (P2-DOCS-REDUCER)

All four required docs updated in `30b563e6c` (153 added lines, reviewed in
full from git rather than from the lead's `docsdiff.txt`). They cover every
5.3 requirement: the required config and its prod/smoke conventions, canonical
metric meanings and reducers, the run-local `tensorboard/` path, the ETA
boundary ("never persisted, never restored as exact-resume state, and is not
scheduling evidence"), the sink-failure policy, and the probe-only exclusions.

*Describe-and-link, no second authority:* `COORDEXP_SWIFT.md` explicitly
delegates the field inventory —
"The field-by-field inventory is in
[`ARTIFACTS.md`](ARTIFACTS.md#canonical-scalar-rows-and-their-reducers)" —
rather than duplicating it. `IMPLEMENTATION_MAP.md` and `SYSTEM_OVERVIEW.md`
add owner/test tables pointing at the four owner modules. No doc claims to be
the scalar authority; all four state `logging.jsonl` is.

*No revived Trainer/event-bus terminology:* scoped search of the four docs for
`event bus|dispatcher|subscriber|emit_event` → **zero hits**. The two
`Trainer` hits are legitimate: `SYSTEM_OVERVIEW.md:87` names
`SupervisedTrainer`, which is a **current, live class**
(`src/training/supervised_trainer.py:144`, constructed at
`src/training/session.py:2345`) — accurate description, not revived history;
`COORDEXP_SWIFT.md:289` says "No GRPO trainer is implemented", an inference
-section disclaimer. Both docs explicitly state "four owners and no generic
coordinator" and "The training facade is not a second row, reducer, or sink
owner."

**Disposition**: `docs/ARTIFACTS.md:81` mis-attributes the reducer for
`grad_norm/pre_clip_rank_max` (see P2-DOCS-REDUCER). One-line fix, must-include #2.

### (g) Residue — **PASS**

Both load-bearing fallbacks are present with recorded justification:

1. **W3-2 LR fallback**: `_scheduler_lr_metrics` at
   `src/training/reporting.py:87`, called at :492, with three dedicated tests
   (`test_reporting.py:193-203`). Recorded in `amend-6`/`amend-7` as
   production-unreachable but load-bearing for the frozen characterization
   row; Wave-5 residue must not remove it. **Not removed.** ✓
2. **`publisher=None` publication-only path**: `src/training/reporting.py:319`
   (`if publisher is None:`). Recorded in `amend-7` as load-bearing for frozen
   characterization fixtures. **Not removed.** ✓

`amend-7` records both explicitly ("Wave-5 residue searches must not delete
either"). Nothing else legacy remains — see (h).

### (h) Independent residue sweep — **PASS (0 hits)**

Designed and run by this auditor, independent of the lead's
`audit_residue.sh`. It parses every `src/**/*.py` with `ast` and canonicalizes
**f-strings** (`ast.JoinedStr` → `<expr>` placeholders) so dynamically built
row keys are not missed by literal-string grep — the specific gap the brief
called out.

*Row-key coverage*: 171 row-key-shaped strings found, of which **14 are
f-string patterns**. Every one resolves to a producer-declared reducer family:

```
finite/<expr>                        src/losses/runner.py:462          -> BOOL_ALL
loss/<expr>/raw|weighted|selected_count   src/losses/runner.py:1342-1344 -> objective_reducer
loss/<expr>/segment_count            src/losses/runner.py:1348         -> IDENTICAL
loss/<expr>/token_weighted_diag      src/losses/runner.py:1345         -> ratio sample
loss/<expr>/weight                   src/training/reporting.py:219     -> IDENTICAL
loss/<expr>/denominator_scope|selected_atom_count|skipped_segment_count
                                     src/training/reporting.py:229-232 -> string scopes
lr/group_<expr>                      src/training/reporting.py:101     -> retained W3-2 fallback
resource/cpu_<expr>                  src/artifacts/resources.py:262    -> MAX
resource/gpu_<expr>                  src/training/reporting.py:121     -> MAX / SUM (deltas)
eval_detection.invalid_<expr>_bbox   src/eval/detection_consumer.py:896 -> not a row key
```

No orphan f-string row key. *Legacy/forbidden pattern set* (8 independent
patterns, PCRE2, `src/` scoped):

```
implicit mean reducer       : 0 hit(s)     (REDUCER_MEAN, reducer="MEAN", _reduce_metric_reports)
name-convention reducer     : 0 hit(s)     (_SUFFIXES, REDUCER_TABLE, reducer_for_name, endswith("_sum"…))
normal per-rank trace       : 0 hit(s)     (per_rank_measurement)
old scheduler-derived LR    : 0 hit(s)     (get_last_lr())
alternate scalar authority  : 0 hit(s)     (metrics.jsonl, scalars.json/.csv, summary.csv)
event bus / registry        : 0 hit(s)     (event_bus, subscribe(, emit_event, add_listener, register_handler()
external sink               : 0 hit(s)     (wandb, mlflow, sqlite3, prometheus_client, statsd)
unbounded sink error state  : 0 hit(s)
TOTAL non-clean residue hits: 0
```

The Wave-2 deletion of `_reduce_metric_reports` and the suffix tables is
confirmed complete, and no per-rank trace survives in normal rows.

## 3. Archive-mechanics check (completion-gate critical)

Wrote an independent differ that honors `## ADDED / MODIFIED / REMOVED /
RENAMED Requirements` section headers (the lead's `obs_delta_diff.py` treats
all delta requirements uniformly, so it would not distinguish an ADDED block
from a MODIFIED one). For every **MODIFIED** requirement, diffed the delta's
scenario list against the current stable spec:

```
===== coordexp-swift-config-runtime =====
  MODIFIED 'Cadence Config And Resolved Step Schedule': stable=4 delta=8  new=4  -> OK
  MODIFIED 'Planned Step Schedule':                     stable=2 delta=4  new=2  -> OK

===== coordexp-swift-training-artifacts =====
  MODIFIED 'Optimizer-Step Order':                      stable=1 delta=4  new=3  -> OK
  MODIFIED 'Wide-Step Logging Stream':                  stable=6 delta=20 new=14 -> OK
  ADDED    'Rank-Zero Presentation Sinks': scenarios=7  clashes_with_stable=False

===== coordexp-swift-supervision-losses =====
  MODIFIED 'Non-Finite Loss And Gradient Gates':        stable=4 delta=9  new=5  -> OK

TOTAL ARCHIVE-MECHANICS PROBLEMS: 0
```

**Zero stable scenarios are omitted from any MODIFIED block**, every MODIFIED
requirement name matches a stable requirement, the single ADDED requirement
does not clash with an existing stable name, and no duplicate scenario names
exist within any requirement. The Wave-0 rebase held; nothing drifted through
Waves 1–5. **The archive tool's fail-closed check will not trip.**

## 4. Commands executed (all read-only)

```bash
# environment: PYTHONDONTWRITEBYTECODE=1 exported inside bash -c wrappers;
# cwd /data/CoordExp/.worktrees/CoordExp-swift; script files, no heredoc argv.

conda run -n ms openspec validate add-coordexp-swift-training-observability --strict   # EXIT=0
conda run -n ms openspec validate --all                                                # 20 passed, 0 failed
conda run -n ms python -m pytest tests/artifacts tests/training/test_reporting.py -q    # 441 passed
conda run -n ms python -m pytest tests/training/test_orchestration_compatibility.py \
     tests/runtime/test_rank_report_collective.py -q                                   # 15 passed
conda run -n ms python <scratch>/junit.py                # obs_w54 1357/0/0/0; obs_w56 2691/0/0/126
conda run -n ms python <scratch>/verify_smoke.py         # 2 rows; 12/12 claim checks OK
conda run -n ms python <scratch>/verify_tb.py            # 77 tags, 0 non-finite, steps [1], 45717812 bytes
conda run -n ms python <scratch>/manifest_history.py     # 5 versions, append-only OK, 0 mutations
conda run -n ms python <scratch>/my_delta_diff.py        # TOTAL ARCHIVE-MECHANICS PROBLEMS: 0
conda run -n ms python <scratch>/sha_pins.py             # both comparator seals OK
sha256sum src/training/input_attestation.py              # 351f2495… OK
cmp obsw5-pre-cache.sha obsw5-post-cache.sha             # identical (exit 0)
timeout 900s conda run -n ms python <lead-tmp>/run_determinant_probe_archived.py       # EQUAL/EQUAL, exit 0
git log --oneline 51cc48de6~1..c4d4dc857                 # chain matches, nothing interleaved
conda run -n ms python <scratch>/residue.py              # 0 legacy hits; 14 f-string keys all declared
# scoped rg sweeps over: src scripts tests configs docs (never unscoped — see host note)
```

No audit command hung or exceeded 20 minutes. One exception, recorded: a
single unscoped `rg` was killed at the 2-minute mark (it was descending into
`outputs/`) and re-run scoped — see the host note above. No GPU action, no
cache mutation, no commit.

## 5. Tree state after this audit

```
$ git status --porcelain
?? openspec/changes/add-coordexp-swift-training-observability/receipts/wave-5-final-audit.md
```

This receipt is the only untracked path; the tracked tree is unchanged at
`c4d4dc857`.
