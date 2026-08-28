# `src/` Entropy Audit (task 6.1) — REPORT ONLY

Change: `reclaim-research-probes-lifecycle`, task 6.1 (D7). Frozen against `research-probes`
@ `a46a5b764`, `git status --short | wc -l == 0`. Method: `reclaim-code-entropy` (Survey → Prove Or
Reject A Candidate → compact evidence record), classes 1–9.

**Nothing was deleted, staged, or edited by this audit.** The only file written is this receipt.
Cuts execute only after the user names them (task 6.2), in D6 batch-and-test form.

## A. Method and counts

### Consumer model

Production consumers: `src/**`, `scripts/**` (229 research scripts after wave 5, `scripts/probes`,
`scripts/analysis`, root entrypoints), `configs/**`, real entrypoints `src/train.py`, `src/infer.py`,
`scripts/evaluate_detection.py`, `scripts/visualize_detection.py`, `scripts/run_infer.py`.
Non-production: `tests/**`, `docs/**`. `research/**` + `memories/**` citations are evidence
obligations (RECORD_CITED), never deletable on this audit's evidence alone.

Fork rule (inherited from `receipts/scripts-entropy-ledger.md`): `image2299-mechanism-microscope`
is a full fork of this tree, so a hit inside it counts only when the containing file is **absent
from `research-probes` or differs by md5**. Of 4772 files there, **403** qualify and were scanned.

Three independent passes, because no single one is sound here:

1. **AST import graph** over every tracked `.py` (absolute + relative imports resolved), covering
   118 `src/` modules against 4,462 research-probes text files + the 403 qualifying fork files.
2. **Text scan** for each module's dotted path and file path (catches `importlib`, string dispatch,
   config keys, JSON receipts, docs).
3. **Reachability closure** from the real entrypoints and every `scripts/**` file outside
   `scripts/analysis/`, to separate "imported by something" from "imported by something live".

Dynamic reachability actually bit: `src/artifacts/__init__.py` resolves `CheckpointWriter`,
`RunWriter`, `ResearchProbeAdmission`, `ExecutionEvidenceJournal` through a `__getattr__` +
`_EXPORT_MODULES` string map and `importlib.import_module`. Pass 3 alone reports
`src/artifacts/checkpoints.py` (330 lines) and `src/artifacts/run_writer.py` (613 lines) as
unreachable; they are reached from `src/training/pipeline.py:34
from src.artifacts import CheckpointWriter, RunWriter`. **Both are KEEP.** An attribute-form scan
(`module.symbol`) likewise rescued `run_data_parallel_shards`, `launch_worker_subprocess`,
`terminate_worker_processes`, `validate_vllm_forced_replay_qualification` from a false-dead reading.

### Inventory

| Surface | Count | Lines |
| --- | ---: | ---: |
| tracked files under `src/` | 126 | — |
| `src/**/*.py` modules | 118 | 57,999 |
| non-`.py` (`src/inference/qualification_receipts/*.json`) | 8 | 954 KB |
| `src/` paths differing from `main` (`git diff --name-only main...research-probes -- src`) | 61 | — |

### Module-level class counts (118 modules)

| Class | Count | Lines | Disposition |
| --- | ---: | ---: | --- |
| SHARED — a production importer in `src/`, `scripts/`, a config, or the live fork | 116 | 57,952 | keep |
| TEST_ONLY — only a test consumes it (`src/trace_config.py`, `src/metrics/__init__.py`) | 2 | 47 | 1 candidate, 1 residue |
| RECORD_CITED-only / ZERO | 0 | 0 | — |

`src/` is **not** where this repository's entropy lives. Every research-fork-only package named in
the task brief (`src/rollout_calibration/`, `src/analysis/sampled_rescue_transition/`,
`src/analysis/visual_support_counterfactual/`, `src/losses/human13_k_union.py`,
`src/losses/rollout_calibration.py`, `src/adapters/dora.py`, `src/artifacts/evidence_journal.py`,
`src/inference/vllm_forced_replay.py`, `src/inference/vllm_qualification.py`,
`src/inference/execution_model_composition.py`, `src/qwen/special_token_embeddings.py`,
`src/vis/rendering.py`, `src/inference/qualification_receipts/*.json`) has a live production
consumer — see section F for each.

### Candidate classes actually found (`reclaim-code-entropy` 1–9)

| Class | Found | Where |
| --- | ---: | --- |
| 1 unconsumed surface | 3 | `write_trainable_surface_receipt`, `field_balanced_duplicate_rejection_and_recovery`, `src/inference/execution_context.py` internals |
| 2 mirrored fact | 1 | duplicate test basename collision (B7) |
| 3 speculative generality | 1 | the loss alias (C2) |
| 4 extra route or layer | 1 | `pipeline.run_data_parallel_shards` |
| 5 lifecycle duplication | 0 | — |
| 6 misplaced defense | 0 | — |
| 7 hand-rolled infrastructure | 0 | — |
| 8 support-only residue | 2 | `src/trace_config.py`, `src/metrics/` |
| **9 added-then-abandoned residue** | **large** | 266 broken `src.*` import targets across 130 files; see B1–B4, D |

## B. Top-ranked deletable candidates

Ranked by confidence × net reduction, risk as tiebreak. `[confidence / risk]`.

```text
[high / low] tests/analysis/{5 abandoned packages} — 53 files, 9,258 lines
evidence: class 9. `src/analysis/{candidate_field_cardinality_tomography,
  policy_objective_mechanism_comparison, post_x1_instance_basin_tomography,
  prefix_state_transition_tomography, sorted_random_no_newline_phenotype}` were deleted by
  e1662c2c7 "Rebuild CoordExp Swift training framework" (2026-07-01, an ancestor of BOTH main and
  research-probes); the implementations were moved to `reference/legacy_src/analysis/`. These 52 test
  modules + 1 conftest are the entire 52 of the 53 `pytest tests --collect-only` errors. Non-test
  referrers: 12 files under `scripts/analysis/` (also broken, B2) and historical text only
  (`docs/history/`, `progress/`, `reference/legacy_src/`). No `research/` or `memories/` citation.
cut: the 5 test packages; 52 ModuleNotFoundError collection errors; 5 dead concepts
tradeoff: none observable — these tests have not executed since 2026-07-01
verify: `conda run -n ms python -c "import src.analysis.candidate_field_cardinality_tomography"`
  -> ModuleNotFoundError (ran, confirmed for all five). After the cut,
  `pytest tests --collect-only -q` must report 1 error, not 53.
net: -53 files, -9,258 lines, -52 collection errors
```

```text
[high / low] scripts/analysis/{12 runner files for the same 5 packages} — 551 lines
evidence: class 9, the producer half of B1. `run.py`/`status.py`/`finalize_if_ready.py`/`real_runtime.py`
  under `scripts/analysis/{candidate_field_cardinality_tomography,prefix_state_transition_tomography,
  sorted_random_no_newline_phenotype,policy_objective_mechanism_comparison}` plus
  `run_post_x1_instance_basin_tomography.py`, `run_policy_objective_mechanism_comparison.py`.
  Every one imports a module that does not exist. No config, no record, no fork consumer.
cut: 12 CLI entrypoints that cannot start
tradeoff: none observable
verify: `conda run -n ms python scripts/analysis/candidate_field_cardinality_tomography/run.py --help`
  -> ModuleNotFoundError
net: -12 files, -551 lines
```

```text
[high / low] scripts/analysis/{28 further broken files} — 4,960 lines
evidence: class 9, same 2026-07-01 rebuild. These import 25 further missing package roots
  (`src.trainers.*`, `src.training.teacher_forcing`, `src.detection.*`, `src.datasets.builders`,
  `src.coord_tokens.*`, `src.utils.*`, `src.common.geometry`, `src.infer.runtime`,
  `src.analysis.autoreg_*`, ...). Largest: `run_qwen3_vl_instance_binding_study.py` (1,265),
  `export_coco_lvis_proxy_jsonl.py` + sibling (626), `run_hard_ce_coord_logit_locality.py` (564).
tradeoff: none observable; the corresponding implementations are preserved in `reference/legacy_src/`
verify: for each file, `conda run -n ms python -c "import ast,sys; ..."` broken-target scan
  (`scratchpad/broken.py`) must return 0 broken `src.*` targets afterwards
net: -28 files, -4,960 lines
```

```text
[high / low] tests/{13 legacy root modules} + tests/helpers/training_architecture_fixture_builder.py — 7,213 lines
evidence: class 9. `test_chat_template_regression`, `test_coord_soft_ce_w1_collective_guard`,
  `test_decode_backend_trace_contract`, `test_decode_provenance_contract`,
  `test_detection_prompt_input_codec`, `test_detection_training_config_contract`,
  `test_grad_accum_loss_scale_mixin`, `test_infer_layout_import_gates`,
  `test_stage2_ab_channel_a_pack_count_skew_barrier`, `test_stage2_ab_ddp_phase_monitor_disable`,
  `test_stage2_ab_disable_average_tokens_across_devices`, `test_stage2_ab_vllm_server_mode_smoke`
  (+ the shared fixture builder). They import inside test bodies, so they COLLECT but FAIL.
  Measured: `92 failed, 9 passed, 5 skipped in 7.01s`; sampled failure reason is uniformly
  `ModuleNotFoundError: No module named 'src.trainers'`.
cut: 92 permanently-red test node ids; the illusion that the suite is meaningfully red
tradeoff: the 9 passing + 5 skipped node ids inside these files must be triaged before the cut —
  they are the only reason this is not [high / very low]
verify: `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest <the 13 files> -q` -> 92 failed today;
  after the cut the full-suite failure set loses exactly those 92 ids and gains none
net: -14 files, -7,213 lines, -92 red node ids
```

```text
[high / very low] src/metrics/__init__.py — 1 line
evidence: class 9, the one `src/` remnant of the 2026-07-01 rebuild. A docstring-only package
  marker ("Metrics package marker.") for a package whose contents were deleted. Only referrers are
  two broken tests: `tests/test_grad_accum_loss_scale_mixin.py` imports
  `src.metrics.dataset_metrics` and `tests/test_detection_training_config_contract.py` names the
  string `"src.metrics.reporter.SwiftMetricReporter"` — both already in B4.
cut: `src/metrics/` directory, one dead package concept
tradeoff: none observable
verify: `conda run -n ms python -c "import src.metrics.dataset_metrics"` -> ModuleNotFoundError (ran)
net: -1 file, -1 line, -1 package
```

```text
[medium / low] src/inference/pipeline.py::run_data_parallel_shards (lines 273-342) — 69 lines
evidence: class 4 (extra route). A second, in-process front door to the data-parallel shard loop.
  The real entrypoint `src/infer.py` -> `pipeline.run` always takes
  `_execute_controller_worker_path` (subprocess controller/worker), and `run_data_parallel_shards`
  itself raises `pipeline.vllm_in_process_forbidden` for the vLLM backend, so it can only ever have
  served HF in-process shards. Production callers: **zero**. Only consumers are 4 call sites in
  `tests/inference/test_pipeline.py` (2 dedicated test functions:
  `test_run_data_parallel_shards_propagates_execution_context_to_each_shard`,
  `test_data_parallel_shards_reject_vllm_in_process_execution`).
cut: 69 lines + 2 test functions + the `DataParallelShardRunResult` return concept if unused after
tradeoff: loses an in-process multi-shard debugging path that nothing currently calls; if it is a
  deliberate escape hatch for HF debugging, keep it — that is a user call, not a discoverable fact
verify: `grep -rn "run_data_parallel_shards" --include=*.py --include=*.yaml src scripts configs`
  -> exactly one hit, its own `def` (ran). Then
  `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest tests/inference/test_pipeline.py -q`
net: -69 src lines, -2 tests, -1 route concept
```

```text
[medium / very low] src/optim/trainable_surface.py::write_trainable_surface_receipt (142-160) — 19 lines
evidence: class 1 (unconsumed export). Exported from `src/optim/__init__.py` `__all__`. Its sibling
  `build_trainable_surface_receipt` IS production (`src/training/pipeline.py:698`,
  `scripts/research/human13_live_model.py:1542`); the writer half has no production caller.
cut: 1 public function, 1 `__init__` export + `__all__` entry, ~3 dedicated test blocks in
  `tests/optim/test_trainable_surface_receipts.py`
tradeoff: none observable — the receipt is persisted by `RunWriter`, not by this function
verify: `grep -rn "write_trainable_surface_receipt" --include=*.py --include=*.yaml src scripts configs`
  -> only `src/optim/trainable_surface.py:142` and its two `src/optim/__init__.py` lines (ran)
net: -19 src lines, -1 export, -~40 test lines
```

```text
[medium / low] src/trace_config.py — 46 lines
evidence: class 8 (support-only residue). A `python -m src.trace_config` CLI. Zero importers.
  Only consumer is `tests/config/test_train_config.py::test_trace_config_writes_resolved_artifacts`,
  which invokes it as a subprocess. No `docs/` (current), `configs/`, `research/`, `memories/`, or
  fork reference; the only mentions are `docs/history/architecture/proposals/2026-06-27-coordexp-swift/`
  (historical design notes proposing it).
cut: 1 entrypoint + 1 test; the "dry config tracing" concept
tradeoff: loses a hand CLI for dumping a resolved train config. `write_resolved_config_artifacts`
  (its whole body) stays and is production via `src/inference/pipeline.py`.
verify: `grep -rn "trace_config" docs configs scripts research memories openspec` -> only
  `docs/history/**` (ran)
net: -1 file, -46 lines, -1 test
```

```text
[high / very low] src/analysis/spatial_scope_history/ (untracked on-disk residue)
evidence: class 9. The directory contains **only** `__pycache__/*.pyc` (13 compiled modules:
  schedule, execution_evidence, spatial, metrics, merge, review_ledger, cohort_ledger,
  _review_finalization, ...). No tracked source. `git status --short src/analysis` is empty
  (it is ignored). The only textual references are one historical test-fixture string in
  `tests/inference/test_research_surface_residue.py:92` and two closed `research/` units.
cut: a stale compiled-only package directory that makes `src/analysis/` look like it has 3 packages
tradeoff: none — `.pyc` without source is not importable and not evidence
verify: `find src/analysis/spatial_scope_history -type f -not -name '*.pyc'` -> empty (ran)
net: -1 phantom package
```

```text
[low / low] src/inference/execution_context.py — internal-only public surface, 101 lines
evidence: class 1 (low priority). `validate_journal_plan_reference` (39), `build_execution_context_payload`
  (27), `ExecutionContextArtifact` (18), `require_sha256_digest` (10), `JOURNAL_PLAN_REFERENCE_KEYS`
  (6), `EXECUTION_CONTEXT_SCHEMA_VERSION` (1) are public but referenced only inside their own
  module; consumers import the wrappers instead.
cut: nothing removable — a privacy/naming cleanup (`_`-prefix), not a deletion
tradeoff: any of these may be a deliberate contract surface for the admission owner
verify: `grep -rn "validate_journal_plan_reference" --include=*.py src scripts tests`
net: 0 lines; -6 public names from the module's apparent API
```

```text
[medium / low] tests/{research,analysis}/test_assemble_source_preservation_multi_route_state_banks.py
evidence: class 2 (mirrored fact) + a real coverage hole. Two DIFFERENT test modules share one
  basename (795 lines vs 173 lines, different md5) and neither `tests/research/` nor
  `tests/analysis/` has an `__init__.py`, so pytest's rootdir module naming collides. This is the
  53rd collection error. In a full `pytest tests` run the `tests/analysis/` module wins and the
  795-line `tests/research/` module is NOT collected; in the D10 baseline scope
  (`pytest tests/research tests/artifacts`) the opposite happens. The two never run together.
cut: rename one module (or add `__init__.py` to both dirs) — a fix, not a deletion
tradeoff: none; today one of two contract suites is silently skipped in every full-suite run
verify: `pytest tests --collect-only -q` -> "import file mismatch" error disappears; collected
  count rises by the 795-line module's tests
net: -1 collection error, +1 suite actually executed
```

## C. Report-only silent-corruption findings (D7)

Per D7 and the project contract, these are reported at whatever confidence and **never cut without
the user naming them**, even where the evidence looks decisive.

```text
[report-only: silent-corruption surface] src/losses/__init__.py + src/losses/rollout_calibration.py:651
  — alias `field_balanced_duplicate_rejection_and_recovery`
class 3 (speculative generality). Line 651 is a bare alias:
  `field_balanced_duplicate_rejection_and_recovery = (field_balanced_duplicate_rejection_loss)`,
  commented "useful to small research probes". It is in `src/losses/rollout_calibration.__all__`
  and in `src/losses/__all__`. Zero consumers anywhere: `src/`, `scripts/`, `tests/`, `configs/`,
  `research/`, `memories/`, and the fork.
verify (ran): `grep -rn "field_balanced_duplicate_rejection_and_recovery" src scripts configs tests
  <fork>/scripts` -> only its own definition and the two `__init__` lines.
Two names for one loss is exactly the duplicate-truth this pass hunts, but it is a loss-accounting
surface. **Report only.**
```

```text
[report-only: silent-corruption surface] src/losses/rollout_calibration.py — 1,282 lines
Module-level importer is only `src/losses/__init__.py`, which made pass 1 flag it. It is
load-bearing: `src/training/rollout_calibration.py` consumes `grouped_entity_transition_preference`,
`owner_conditioned_candidate_loss`, `first_wrong_coordinate_preference`,
`positive_path_imitation_loss`, `field_balanced_duplicate_rejection_loss`,
`rollout_site_token_type_gate`, and 12 `configs/coordexp_swift/smoke/*rollout_calibration*.yaml`
select it by config key. **KEEP.** Recorded here so a future pass does not re-open it.
```

```text
[report-only: silent-corruption surface] src/inference/backend_parity.py — 1,122 lines
Consumers: `scripts/probes/coordexp_swift/backend_parity.py` + `tests/inference/test_backend_parity.py`.
A single-probe consumer is thin, but parity is named as a silent-corruption surface by D7 and the
project contract. No candidate raised. 43 of its 61 top-level symbols are private helpers with
internal-only use, which is normal for a validator, not entropy.
```

```text
[report-only: silent-corruption surface] src/losses/human13_k_union.py (993),
  src/templates/{renderer.py (561), spans.py (236)}, src/training/{pipeline.py (2,106),
  supervised_trainer.py (841), rollout_calibration.py (1,907)}, src/packing/{supervision.py (296),
  planner.py (200)}, src/training/pack_cache.py (745)
All have live production consumers (`src/train.py` chain, `configs/**`, kept research scripts, and
for `human13_k_union` the live fork). No candidate raised on any of them. Listed so the audit's
coverage of the masking / supervision-position / loss-accounting / packing surfaces is explicit.
```

## D. Abandoned-test residue (lead 2a)

`pytest tests --collect-only -q` (ran, 8.09s): **6,504 tests collected, 53 errors**.
52 errors are `ModuleNotFoundError: No module named 'src.analysis.<pkg>'`; the 53rd is the
duplicate-basename `import file mismatch` in B7.

| Test package (`tests/analysis/…`) | Modules | Lines | Missing implementation | Non-test referrers still present |
| --- | ---: | ---: | --- | --- |
| `sorted_random_no_newline_phenotype/` | 14 | 5,470 | `src.analysis.sorted_random_no_newline_phenotype` | `scripts/analysis/sorted_random_no_newline_phenotype/{run,status,real_runtime}.py`; `reference/legacy_src/analysis/sorted_random_no_newline_phenotype/*`; `docs/history/superpowers/plans/2026-06-04-*` |
| `post_x1_instance_basin_tomography/` | 9 | 1,557 | `src.analysis.post_x1_instance_basin_tomography` | `scripts/analysis/run_post_x1_instance_basin_tomography.py`; `docs/history/superpowers/{plans,specs}/2026-06-04-post-x1-*` |
| `prefix_state_transition_tomography/` | 11 | 1,127 | `src.analysis.prefix_state_transition_tomography` | `scripts/analysis/prefix_state_transition_tomography/{run,status,finalize_if_ready}.py`; `reference/legacy_src/analysis/prefix_state_transition_tomography/*`; `progress/diagnostics/2026-06-04_*` |
| `candidate_field_cardinality_tomography/` | 15 + conftest | 943 | `src.analysis.candidate_field_cardinality_tomography` (+ `src.common.detection_compact_rows`) | `scripts/analysis/candidate_field_cardinality_tomography/{run,status,finalize_if_ready}.py`; `docs/history/superpowers/{plans,specs}/2026-06-03-*`; `progress/diagnostics/2026-06-02_*` |
| `policy_objective_mechanism_comparison/` | 3 | 161 | `src.analysis.policy_objective_mechanism_comparison` (its `test_existing_harness_configs.py` and `test_policy_objective_mechanism_comparison.py` also import the other three) | `scripts/analysis/policy_objective_mechanism_comparison/run.py`, `scripts/analysis/run_policy_objective_mechanism_comparison.py` |
| **total** | **53 files (52 test modules + 1 conftest)** | **9,258** | 5 packages | 12 `scripts/analysis/` files, all themselves broken |

Module names, for the deletion list:

- `candidate_field_cardinality_tomography/`: `conftest`, `test_artifacts`, `test_attention_components`, `test_basin_attraction`, `test_case_index`, `test_config`, `test_controls`, `test_finalize`, `test_prefixes`, `test_probe_plan`, `test_report`, `test_residual_row_scoring`, `test_runner_cli`, `test_status`, `test_taxonomy`, `test_x1_candidate_field`
- `policy_objective_mechanism_comparison/`: `test_config_and_report`, `test_existing_harness_configs`, `test_policy_objective_mechanism_comparison`
- `post_x1_instance_basin_tomography/`: `test_case_universe`, `test_config`, `test_gallery`, `test_merge_report`, `test_posterior`, `test_prefix_modes`, `test_runner_cli`, `test_status`, `test_trajectory`
- `prefix_state_transition_tomography/`: `test_boundary_scoring`, `test_config`, `test_gallery`, `test_jsonl`, `test_merge_report`, `test_paired_probe`, `test_prefix_rendering`, `test_prefix_state_index`, `test_runner_cli`, `test_status_finalize`, `test_x1_readout`
- `sorted_random_no_newline_phenotype/`: `test_boundary_roles`, `test_config`, `test_data_root_audit`, `test_fn_hint_runtime`, `test_fn_matching`, `test_fn_probe`, `test_gallery`, `test_merge_report`, `test_native_rollout`, `test_paired_probe`, `test_prefix_index`, `test_rollout_phenotype`, `test_runner_cli`, `test_status`

Provenance: all five implementations were removed by `e1662c2c7` "Rebuild CoordExp Swift training
framework" (2026-07-01), **an ancestor of both `main` and `research-probes`** — so this residue is
inherited from production, not created by the research fork. The implementations survive under
`reference/legacy_src/analysis/` (89 files), which is why nothing else broke.

### Wider blast radius of the same event

The broken-import scan (`scratchpad/broken.py`, AST over every tracked `.py`) found **266 distinct
non-existent `src.*` import targets** referenced by ~130 non-`reference/` files:
`scripts/analysis/` 40 files (5,511 lines), `tests/analysis/` 52 (9,258), `tests/` root 13 (7,213
incl. one helper), `public_data/` 18, `scripts/` root 6, `scripts/tools` 3, `vis_tools` 1,
`scripts/diagnostics` 1. Missing package roots include `src.trainers.*` (25 targets),
`src.training.{teacher_forcing,objectives,supervision,stage2,span_adapters,pipelines,templates,
observability,encoding,bridge}` (36), `src.detection.*`, `src.datasets.builders`,
`src.coord_tokens.*`, `src.utils.*`, `src.common.geometry`, `src.infer.runtime`, `src.vis.gt_vs_pred`.
`public_data/`, `vis_tools/`, `scripts/tools/`, `scripts/diagnostics/` were **not** audited here
(outside this task's scope) and are the obvious next report.

## E. `tests/analysis` ImportError diagnosis (lead 2b)

The ledger's "3 pre-existing `ImportError` failures originating in `src/inference/backend.py`" is
**two** ImportErrors plus one unrelated failure. Measured:
`pytest tests/analysis/test_sampled_rescue_transition.py
tests/analysis/test_assemble_source_preservation_multi_route_state_banks.py` -> `3 failed, 22 passed`.

**The two real ones — a live defect, not abandoned surface.** Commit `767e57f5e`
"feat: add support for sampling parameters in HFGenerateBackend" (2026-07-17) shrank
`src/inference/backend.py` from 6,843 to 1,002 lines and deleted, among others,
`_normalized_attested_model_identity_for_runtime_comparison` (was at parent line 5789) and
`canonical_float32_logprob` (was at parent line 6207). Its consumers were never updated:

| Site | Form | Effect today |
| --- | --- | --- |
| `scripts/research/run_sampled_rescue_transition.py:1487` | bare function-local `from src.inference.backend import _normalized_attested_model_identity_for_runtime_comparison` | hard `ImportError` — the donor model-identity comparison in the sampled-rescue replay path aborts. Fails test `test_donor_runtime_identity_allows_relocation_but_rejects_payload_change`. |
| `scripts/research/run_sampled_rescue_transition.py:1145` | same import inside `try: … except Exception as exc: raise SystemExit(f"first free token logprob is not finite float32: {exc}")` | the `ImportError` is **swallowed and remisdiagnosed** as a numeric-finiteness failure. Fails test `test_first_free_token_evidence_requires_matching_executed_float32_trace`. |

Both are lazy (function-body) imports, so the module still imports and the failure only appears on
those two code paths. `run_sampled_rescue_transition.py` is RECORD_CITED (kept by the scripts
ledger), so this is a **repair obligation, not a deletion candidate**: either restore the two
helpers (they exist at `git show 767e57f5e^:src/inference/backend.py`) or rewrite the two call sites
against the current backend contract. The masked-diagnosis form at :1145 is the more dangerous of
the two and is worth fixing regardless of the deletion decision.

**The third failure is unrelated to `backend.py`.**
`tests/analysis/test_assemble_source_preservation_multi_route_state_banks.py::
test_family_weights_are_equal_per_image_and_mean_one` raises
`AssemblyError: family event count 1008 differs from required 1024` from
`scripts/research/assemble_source_preservation_multi_route_state_banks.py:827` — an arithmetic
mismatch between the test's fixture (118 images × 8 + 2 × 32 = 1008) and `TARGET_EVENT_COUNT` 1024.
No missing symbol involved.

## F. Kept / rejected high-value candidates

| Candidate | Lines | Reason class | Evidence |
| --- | ---: | --- | --- |
| `src/artifacts/research_probe_admission.py` | 1,830 | prior decision record | KEEP by the scripts ledger and by design (fail-closed admission owner). Only internal speculative surface found: `TargetTreeIdentity` (19 lines) is public but internally used. Low priority; do not cut. |
| `src/artifacts/checkpoints.py`, `src/artifacts/run_writer.py` | 943 | dynamic reachability | reached via `src/artifacts/__init__.py` `__getattr__` + `_EXPORT_MODULES` string map from `src/training/pipeline.py:34` |
| `src/rollout_calibration/{__init__,planning,replay,state_bank}.py` | 5,239 | real consumer | `src/training/pipeline.py`, `src/training/rollout_calibration.py`, `src/config/loader.py`, 6 kept research scripts, 7 fork-only scripts, `tests/rollout_calibration/`; `state_bank` is cited by a `research/` unit (RECORD_CITED) |
| `src/analysis/sampled_rescue_transition/{__init__,artifacts,comparison}.py` | 763 | real consumer | `run_sampled_rescue_transition.py`, `run_native_sibling_branch_replay.py`, `analyze_native_sibling_branch_value.py` (all kept) + 2 fork-only scripts |
| `src/analysis/visual_support_counterfactual/{__init__,intervention}.py` | 1,113 | real consumer + record | 3 kept `run_fixed_encoding_*` scripts, **7 fork-only scripts in the live image2299 direction**, and 2 `research/` units cite `intervention` |
| `src/losses/human13_k_union.py` | 993 | real consumer + silent-corruption | `human13_on_policy_live.py`, `human13_row_contrast_live.py`, `run_human13_k_union_overfit.py`, 2 fork-only scripts, 2 tests |
| `src/losses/rollout_calibration.py` | 1,282 | real consumer + silent-corruption | see C2 |
| `src/adapters/dora.py` | 1,658 | real consumer | 11 importers incl. `src/training/pipeline.py`, `src/inference/{execution_model,execution_model_composition,hf_backend}.py`, `src/optim/*` |
| `src/artifacts/evidence_journal.py` | 1,061 | real consumer | `src/artifacts/research_probe_admission.py` + 2 kept scripts + 3 tests |
| `src/inference/vllm_forced_replay.py` | 85 | real consumer | `src/inference/vllm_backend.py`, referenced by 5 qualification receipts |
| `src/inference/vllm_qualification.py` | 1,248 | real consumer + compatibility | `src/inference/vllm_backend.py`, `scripts/probes/coordexp_swift/vllm_concurrency.py`, `docs/IMPLEMENTATION_MAP.md`; binds all 7 vLLM receipts |
| `src/inference/execution_model_composition.py` | 783 | real consumer | `src/inference/execution_model.py`, `scripts/probes/coordexp_swift/execution_model_composition.py` |
| `src/qwen/special_token_embeddings.py` | 1,600 | real consumer | 12 importers incl. `src/training/pipeline.py`, `src/artifacts/checkpoints.py`, `src/optim/*`; 4 fork-only scripts |
| `src/vis/rendering.py` | 372 | real consumer | `src/vis/api.py` -> `src/vis/__init__.py` -> `scripts/visualize_detection.py` (a real entrypoint); 3 fork-only scripts |
| `src/inference/qualification_receipts/*.json` | 8 files, 954 KB | compatibility obligation | bound as module constants in `src/inference/vllm_qualification.py:18-33` (`QUALIFICATION_RECEIPT`, `APPLICATION_QUALIFICATION_RECEIPT`, FP32 variants, forced-replay, concurrency) and as `DURABLE_COMPOSITION_RECEIPT_ROOT` in `src/inference/execution_model.py:30`. The two 446 KB files are the vLLM 0.14.1 source-qualification evidence — not entropy. |
| `src/inference/{backend,hf_backend,vllm_backend}.py` | 3,865 | necessary independence | `backend.py` is the protocol + `DecodeRequest`/`DecodeResult`/`BackendLaunch` contract with 64 production importers; `hf_backend`/`vllm_backend` are two genuine implementations of it. Not a one-implementation interface. |
| `src/inference/runtime.py` | 122 | real consumer | inspected in full: it is not a forwarding layer — it projects `InferConfig` into `BackendLaunch` and enforces 4 distinct runtime contracts. 50 production importers. |
| `src/inference/{worker,merge,data_parallel,execution_context}.py` | 4,781 | real consumer | the controller/worker path taken by `src/infer.py` -> `pipeline.run` -> `_execute_controller_worker_path`; `run_data_parallel_shards` is the only dead branch (B6) |
| `src/eval/forward.py` | 284 | churn / tiny | reached only through `src/eval/__init__.py` relative re-export and one test, but `docs/IMPLEMENTATION_MAP.md` names it as the forward-eval surface; too small a win to justify touching an eval path |
| `src/inference/backend_parity.py` | 1,122 | silent-corruption surface | C3 |
## Verification posture

This audit ran no destructive command. The commands quoted per candidate are the smallest decisive
checks and were executed as stated. Any cut approved under task 6.2 must still follow D10: record
the CPU-only failure **set** before and after, compare sets and never counts, and keep the 44
admission tests green. Recovery for anything cut is `git checkout research-base-v2 -- <path>`.
