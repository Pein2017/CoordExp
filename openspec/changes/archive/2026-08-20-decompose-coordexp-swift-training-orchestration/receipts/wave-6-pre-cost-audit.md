# Wave-6 Pre-Cost Standards / Overdesign / Intent Audit

| Field | Value |
| --- | --- |
| Change | `decompose-coordexp-swift-training-orchestration` |
| Audited commit | `88391d6bb208931ba3fa537fb9522ea6b106fe60` (Wave 6, HEAD) |
| Wave-0 predecessor | `eb2dc97ab15e76aa75f9751f2b1fe463e271fa90` |
| Date | 2026-08-20 |
| Auditor | Opus pre-cost auditor, spawned by Claude Fable lead (single independent auditor of record for task 7.7; distinct from every builder and from the lead) |
| Task | 7.7 — the single pre-cost standards/overdesign/intent audit |
| Mode | Read-only. The only file written by this audit is this receipt. No commits, no source/test/fixture/tasks.md edits, no `src.prepare_train_cache`, no GPU command. |
| Working tree at audit start | `git status --porcelain` empty |
| Working tree re-check before write | HEAD still `88391d6bb…`, `git status --porcelain` empty. Receipt is **not** VOID-DRIFT. |

## Verdicts

| # | Area | Verdict |
| --- | --- | --- |
| 1 | STANDARDS — do Waves 0–6 satisfy the change's own rules? | **PASS-WITH-DISPOSITIONS** |
| 2 | OVERDESIGN — did the decomposition add forbidden machinery? | **PASS** |
| 3 | INTENT — does the result match the stated intent, and are the two lead dispositions sound? | **PASS-WITH-DISPOSITIONS** |
| 4 | LAUNCH GATE — Wave-7 cache build + Wave-8 two-rank GPU smoke | **CLEARED** (0 P0, 0 P1) |

Severity counts: **P0 = 0, P1 = 0, P2 = 5, P3 = 5.**

---

## 1. STANDARDS — PASS-WITH-DISPOSITIONS

### Verified

**Wave sequence — one revertible commit per wave, strict order from `eb2dc97ab`.**

```
$ git log --oneline eb2dc97ab..HEAD
88391d6bb refactor(training)!: delete legacy_fused mode and provider env override   <- Wave 6 (code + close-out)
6760a6a8c docs(openspec): close decompose wave 5
32867ef18 refactor(training): move the model-bearing lifetime into TrainingSession   <- Wave 5
8239be128 docs(openspec): close decompose wave 4
4986836b0 refactor(training): extract reporting and pure run-writer internals        <- Wave 4
5e4e3e4bb docs(openspec): close decompose wave 3
ca669017c refactor(training): extract cache contract and workflow owners             <- Wave 3
d44b691b5 docs(openspec): close decompose wave 2
505a14b36 refactor(training): extract execution plan and rank control plane          <- Wave 2
912a57219 docs(openspec): close decompose wave 1
0583938cb refactor(training): extract leaf identity and micro-step owners            <- Wave 1
3279d14bc docs: calibrate superpowers usage in canonical AGENTS.md                   <- disclosed docs-only exception
9aafca917 docs(openspec): close decompose wave 0 baseline
2ee6c4959 test(training): freeze orchestration compatibility ledger                  <- Wave 0
(14 commits)
```

Strict monotone sequence, no interleaving of two waves' code. The one interleaved commit is
disclosed and confirmed non-overlapping:

```
$ git show --stat 3279d14bc
 AGENTS.md | 15 +++++++++++++++
 1 file changed, 15 insertions(+)
```

AGENTS.md only — zero owner-surface, spec, source, test, fixture, or manifest overlap.

**Wave-6 independent revertibility (task 7.7 obligation, not previously executed).**
The wave-6 close-out note (f) records the commit and revert proof as unexecuted. Executed here
read-only:

```
$ git --no-pager diff --binary --no-color 88391d6bb^ 88391d6bb > wave6.patch   # 1062 lines
$ git apply -R --check wave6.patch
REVERT-CHECK: CLEAN
```

The wave-6 patch reverse-applies cleanly to the HEAD tree with no residual conflict; the wave is
independently revertible to Wave 5 (`6760a6a8c`) and touches no cache target (see §4). The
`--check` form performs no write.

**Characterization fixtures byte-identical to their Wave-0 introduction.**

```
$ git diff --stat 2ee6c4959..HEAD -- tests/fixtures/          -> (empty)
$ git log --oneline 2ee6c4959..HEAD -- tests/fixtures/        -> (empty)
$ git rev-parse 2ee6c4959:tests/fixtures/training_orchestration
2fe137ccffc9b51e4eb227a91e2fbe6607dfc528
$ git rev-parse HEAD:tests/fixtures/training_orchestration
2fe137ccffc9b51e4eb227a91e2fbe6607dfc528
```

Identical tree object. No wave regenerated a frozen trace, and no commit after Wave 0 touched
`tests/fixtures/` at all.

**Flip / seam-repoint coverage — sampled at Waves 5 and 6 as instructed.**

The manifest declares 26 `declared_flips.nodes` (`flips_at_wave` histogram
`{1:3, 2:7, 3:10, 4:2, 5:4}`) and 6 `harness_seam_repoints.helpers`.

*Wave 5* (`32867ef18`): every test-expectation change is accounted for.
- The 4 declared wave-5 flips landed exactly as the manifest requires — `WAVE0_PIPELINE_OWNED_HELPERS`
  entries `_run_initialized_training`, `_checkpoint_handler`, `_eval_forward_handler`, `_final_handler`
  move `(5, None) -> (5, session)` (tests/training/test_pipeline_assembly.py:2760-2763). The node
  records the new owner; it was not weakened or deleted.
- Every other changed line in `tests/training/test_pipeline_assembly.py` is a literal
  `pipeline.X -> session.X` module-name swap, matching the `harness_seam_repoints` rule
  ("when a wave moves an owner, the harness re-points to the new owner and the frozen fixture must
  replay UNCHANGED"). No asserted value changed. Fixture invariance above independently confirms
  the replay side.

*Wave 6* (`88391d6bb`): the manifest declares **zero** flips at wave 6. Inspection of every
test-expectation change shows they are all the declared BREAKING deletion:
- `tests/config/test_train_config.py`: `legacy_fused` moves out of the *accepts* parametrize and into
  the *rejects* parametrize (net node count unchanged).
- `tests/artifacts/test_run_artifacts.py`: `["legacy_fused","overlapped","synchronous"] -> ["overlapped","synchronous"]`
  (one node removed); a receipt `"source"` literal moves `deprecated_environment_override -> strict_config`.
- `tests/training/test_orchestration_compatibility.py`: the retired variable is dropped from the
  characterization worker's env-clearing list (fixtures still replay byte-identical).
- `node_paths_in_order` for `wave6-legacy-selector-gate` is **file-level** and unchanged, and the
  gate's `environment_selectors` remain accurate (tests still `setenv` the retired variable to prove
  it is ignored), so `revision_rule` was not triggered by a command/path/selector change.
See finding **S-3** for the bookkeeping gap this leaves.

**Strict OpenSpec validation (re-executed).**

```
$ conda run -n ms openspec validate decompose-coordexp-swift-training-orchestration --strict
Change 'decompose-coordexp-swift-training-orchestration' is valid     (rc=0)
```

**Wave-6 gate replay (re-executed from `test-command-manifest.json` — same argv file list and `-q`;
additive non-collecting flags disclosed at the end of §6).**
The builder-drafted `receipts/wave-6-gate.json` was dropped per lead precedent, so the 547/0/0 claim
existed only as self-report. Replayed independently:

```
$ conda run -n ms pytest tests/config/test_train_config.py \
    tests/training/test_forward_input_provider.py tests/training/test_supervised_trainer.py \
    tests/training/test_pipeline_assembly.py tests/training/test_pipeline_cache_preflight.py \
    tests/training/test_pipeline_exact_resume.py tests/training/test_wave5_provider_benchmark.py \
    tests/artifacts/test_run_artifacts.py tests/training/test_training_module_boundaries.py \
    tests/training/test_orchestration_compatibility.py tests/runtime/test_rank_report_collective.py -q
  (env: PYTHONDONTWRITEBYTECODE=1, -p no:cacheprovider, --junitxml; the nine
   `required_absent` environment selectors unset)

547 passed in 300.63s (0:05:00)     RC=0
JUnit: {'tests': '547', 'failures': '0', 'errors': '0', 'skipped': '0', 'time': '300.615'}
```

Matches the close-out claim exactly: 547 passed / 0 failed / 0 skipped, `expected_red_nodes` empty.

**Wave-0 baseline argv replay at HEAD.**

```
$ conda run -n ms pytest tests/training/test_training_module_boundaries.py \
    tests/training/test_orchestration_compatibility.py tests/training/test_pipeline_assembly.py \
    tests/training/test_pipeline_phase_convergence.py tests/artifacts/test_run_artifacts.py \
    tests/training/test_pack_cache_determinant_registry.py tests/runtime/test_rank_report_collective.py -q

283 passed in 64.06s     RC=0
JUnit: {'tests': '283', 'failures': '0', 'errors': '0', 'skipped': '0'}
```

Matches the close-out claim (285 -> 283 by the recorded node accounting). All 14 Wave-0
`wave0_observed_red_nodes` (10 owner-module obligations + 4 parity-import clearances) have converged.

**Wave-5 disclosure (g) blocking residue independently re-verified closed.** The three probe scripts
and their contract suites, plus every new-owner suite:

```
$ conda run -n ms pytest tests/qwen/test_packed_parity.py tests/qwen/test_wave2_v3_probe.py \
    tests/losses/test_wave3_zero_weight_probe_contract.py tests/training/test_reconcile_exact_resume_probe.py \
    tests/packing/test_wave6_pack_plan_probe.py tests/supervision/test_tokens.py \
    tests/training/test_cache_contract.py tests/training/test_cache_workflow.py \
    tests/training/test_control_plane.py tests/training/test_execution_plan.py \
    tests/training/test_reporting.py tests/training/test_training_session.py \
    tests/artifacts/test_run_schema.py -q

746 passed in 220.83s     RC=0
JUnit: {'tests': '746', 'failures': '0', 'errors': '0', 'skipped': '0'}
```

Total independently replayed at HEAD: **1,576 nodes, 0 failed, 0 errored, 0 skipped.**

### Standards findings

See the consolidated table in §5: **S-1, S-2, S-3, S-4**.

---

## 2. OVERDESIGN — PASS

**Forbidden machinery: absent.** Repository-wide grep over `src/` for
`event_bus|eventbus|EventEmitter|subscribe(|publish_event|plugin_registry|PluginRegistry|dependency_injection|DIContainer|ServiceContainer|Injector|FSDP|FullyShardedDataParallel|BackendProtocol|AbstractBackend|ExecutionBackend`
returns **zero hits in any production module**. The only matches in the tree are six lines inside
pre-existing vendored inference qualification receipts
(`src/inference/qualification_receipts/vllm-0.14.1*.json`) naming
`transformers.integrations.fsdp` in a dependency inventory — unrelated to this change and untouched
by it. The word "registry" appears in the new owners only inside prose comments that explicitly
*disclaim* one (`execution_plan.py:12` "no … callback registry", `session.py:1601` "no phase
subclass, registry, callback container, or alternate …", `reporting.py:9` deferring a metric registry
to `add-coordexp-swift-training-observability`). No second backend, no speculative config key.

**Facade is genuinely a facade.** `src/training/pipeline.py` is 156 lines (6,588 at `eb2dc97ab`), two
public entry points, zero business logic: it builds the plan, opens/binds the control plane, admits
the policies and cache workflow, constructs exactly one `TrainingSession`, returns its result, and
closes in `finally`. `prepare_training_pack_caches` is a one-line delegation to `cache_workflow`,
retained because the Wave-0 compatibility ledger freezes the published name and its `__module__`.

**One-way import graph enforced by a green test**
(`tests/training/test_training_module_boundaries.py::test_training_import_graph_has_no_reverse_edges`,
present and passing in the replayed gate JUnit).

**Public surfaces are small, and none is caller-free.** Module-level public names (AST-enumerated),
each cross-checked for callers across `src/`, `tests/`, `scripts/`:

| Owner | Public names | Zero-caller |
| --- | ---: | --- |
| `src/training/execution_plan.py` | 2 | none |
| `src/training/control_plane.py` | 2 | none |
| `src/training/session.py` | 13 | none |
| `src/training/cache_workflow.py` | 6 | none |
| `src/training/cache_contract.py` | 1 | none |
| `src/training/reporting.py` | 1 | none |
| `src/training/micro_steps.py` | 2 | none |
| `src/artifacts/identity.py` | 19 | none |
| `src/artifacts/run_schema.py` | 0 | — |
| `src/artifacts/run_state.py` | 2 | none |
| `src/training/pipeline.py` | 2 | none |

`cache_workflow.EVAL_SPLIT` and `cache_workflow.REQUIRE_ALL_HIT_VERIFICATION_LEVEL` appear
caller-free only to a cross-file scan; both are consumed inside their own module
(`cache_workflow.py:706,720,731,739,742,752,753`) alongside `TRAIN_SPLIT`. `run_schema.py` exposes no
public name at all and is consumed only by `run_state.py` and `run_writer.py`.

**Cache determinant registry no longer owns the orchestration modules.** Grep of
`src/training/pack_cache.py` for `pipeline.py` / `supervised_trainer.py`: zero hits (task 9.4's
standing residue check already holds at Wave 6).

Findings: **O-1, O-2** (both P3, observations only).

---

## 3. INTENT — PASS-WITH-DISPOSITIONS

**Facade reduction: achieved.** `src/training/pipeline.py` 6,588 -> 156 lines.

**Decomposition follows the design's boundary inventory: achieved and test-enforced.** All ten
intended owner modules exist (`execution_plan`, `control_plane`, `session`, `cache_workflow`,
`cache_contract`, `reporting`, `micro_steps`, `artifacts/identity`, `artifacts/run_schema`,
`artifacts/run_state`), and all four production `src.qwen.parity` import obligations are cleared —
`grep -rn "src\.qwen\.parity" src/` returns zero hits. The 14 Wave-0 RED boundary nodes are all green.

**BREAKING deletion: complete.** `grep -rn "legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"`
returns **zero live references** in `src/`, `configs/`, `docs/COORDEXP_SWIFT.md`,
`docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/ARTIFACTS.md`, and `openspec/specs/`.
Surviving references are exactly three labeled classes: (i) strict-rejection / ignore-the-env test
assertions, (ii) the benchmark probe arm "D" (disposition (b) below), (iii) historical planning
records under `docs/superpowers/plans/` and this change's own authority artifacts. Rejection is
proven green, not merely asserted — four nodes in the replayed gate JUnit:
`test_forward_input_provider_mode_rejects_unknown_or_renamed_values[legacy_fused]`,
`test_environment_cannot_replace_strict_provider_mode[legacy_fused-synchronous]`,
`test_environment_cannot_replace_strict_provider_mode[legacy_fused-overlapped]`,
`test_build_forward_input_provider_rejects_retired_and_unknown_modes[legacy_fused]`.

### Disposition (a) — `_default_qwen_forward` retained in the trainer module: **SOUND**

Evidence:
- `src/training/supervised_trainer.py:561` defines it; `src/eval/forward.py:22` imports it and
  `src/eval/forward.py:269` uses it as that module's production default
  (`self.qwen_forward = qwen_forward or _default_qwen_forward`).
- The dependency is **pre-existing, not introduced by this change**: the identical two lines are
  present at the Wave-0 predecessor (`git show eb2dc97ab:src/eval/forward.py` — lines 22 and 269),
  and `git diff --stat eb2dc97ab..HEAD -- src/eval/` is empty. `src/eval/` is byte-untouched across
  Waves 0–6.
- The trainer-side claim is true: no production trainer path calls it. The default wiring is gone and
  construction fails closed — `code="trainer.forward_input_source_required"` at
  `src/training/supervised_trainer.py:174` and `:311`, asserted at
  `tests/training/test_supervised_trainer.py:1394` (green in the gate replay). Remaining references
  are test call sites that pass it explicitly and one probe that monkeypatches it.

Adjudication: the disposition is correct on its facts and the alternative (deleting the
`qwen_forward` seam) would have rewritten 28 trainer call sites and changed what that suite asserts,
colliding with the change's own "supported-mode tests keep passing unchanged" rule. It is **acceptable
residue**, recorded as **P2 (I-1)** — not because it risks anything at launch, but because an
eval-only production symbol living in a trainer module and imported across packages by its private
name is a real ownership defect that a later change must own. It is **not** a `legacy_fused`
survival and blocks nothing.

### Disposition (b) — benchmark arm "D" retained as an unrunnable fail-closed probe: **SOUND**

Evidence:
- `scripts/probes/coordexp_swift/wave5_provider_benchmark.py:81` still declares
  `"D": {"provider_mode": "legacy_fused", "role": "candidate"}`.
- The fail-closed claim is verified by inspection: `_write_arm_config` (line 2283) writes
  `payload["training"]["forward_input_provider_mode"] = ARM_SPECS[arm]["provider_mode"]` and then
  calls `load_train_config(config_path)` — the same strict loader that the green node
  `tests/config/test_train_config.py::test_forward_input_provider_mode_rejects_unknown_or_renamed_values[legacy_fused]`
  proves rejects `legacy_fused`. Arm D therefore cannot execute; it raises at strict validation.
- The measurement contract is genuinely user-owned frozen research design: `TRIAD_ORDERS` is a
  three-arm Latin square `(("R","D","O"),("D","O","R"),("O","R","D"))` with `CANDIDATE_ARMS=("D","O")`,
  `MIN_PAIRED_OBSERVATIONS=3`, `expected_arm_executions=9`. Re-choosing these for two arms is a new
  design decision, correctly escalated rather than taken.
- No promoted claim depends on it: `load_historical_r2_plan` authenticates by file SHA256
  (`154d13d9f989…`), schema string, and `plan_sha256` only — never by arm identity — and the r2 run
  terminated `resource-stop` / `complete_non_promoting` with zero observations.

Adjudication: **acceptable residue, P3 (I-2/I-3)**. A deleted mode that fails closed at strict config
is the correct behavior, and the retained arm is the honest record of a measurement design that was
never executed. Two subordinate observations are recorded for whoever eventually removes it.

---

## 4. LAUNCH GATE — Wave-7 cache build and Wave-8 GPU smoke: **CLEARED**

There are **no unresolved P0 or P1 findings**, so nothing in this audit blocks tasks 8.1–8.5 or
9.1–9.6.

Precondition spot-checks:

| Precondition | Observed | Status |
| --- | --- | --- |
| Production cache root `.cache/coordexp_swift/packing/` empty | `os.listdir` -> `[]` (directory exists, zero entries) | OK — no intermediate fingerprint was built during Waves 1–6 |
| Old immutable targets intact | `.cache/coordexp_swift/geometry_flip_aug_5step/` holds exactly two targets, `2232c868dae6…` and `9c82bc7578…`, 2 files each, mtimes 1783394670.996 / 1783394682.933 (pre-dating this change) | OK — the train/eval pair Wave 7 must leave untouched |
| Smoke config strict-loads | `load_train_config("configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml")` -> OK; `forward_input_provider_mode=synchronous`, `max_steps=1`, `artifact_root=outputs/smoke/production_mimic`, `collision_policy=timestamp` | OK, with caveat **L-1** |
| Free disk on `/data` | `/dev/nvme0n1p1 3.5T total, 1.8T used, 1.6T avail (53%)` | OK — ample |
| Legacy selector cannot re-enter a cache/GPU run | zero live `legacy_fused` references in `src/` or `configs/`; smoke config resolves to the supported reference mode `synchronous` | OK |

Clearance is for *this audit's gate only*. It is **not** launch authority: tasks 8.2 and 9.2 still
require their own fresh user-authorized packets bound to the exact commit, full argv, absent targets,
and numeric resource bounds, per `launch_bearing_actions` in the manifest ("planning or
implementation approval is not launch authority"). Finding **L-1** is an input those packets must
consume.

---

## 5. Findings

| ID | Sev | Area | Description | Disposition |
| --- | --- | --- | --- | --- |
| S-1 | P2 | Standards / record accuracy | Wave-6 close-out note (f) in `tasks.md` still states "Task 7.7's commit, revert proof, and pre-cost … audit are unexecuted: this wave was produced under an explicit no-commit instruction", and points gate evidence at `receipts/wave-6-gate.json`, **which does not exist** (the lead disposition records that it was dropped per the waves-0..5 precedent). Meanwhile task 7.7 is checked `[x]`. The note is stale and self-contradictory against the committed reality (`88391d6bb`). | Doc-only; no behavioral or launch risk. This audit supplies the missing commit/revert/audit evidence. Lead should correct note (f) and drop the dangling receipt path when closing Wave 6, so the Wave-7/8 packet author is not misled about what was executed. |
| S-2 | P2 | Standards / task 7.7 obligation | The Wave-6 revert proof required by task 7.7 was not executed by the wave owner. | **Discharged by this audit.** `git apply -R --check` on the wave-6 patch is CLEAN against the HEAD tree (§1); the wave reverts to `6760a6a8c` independently and mutates no cache target. Record this receipt as the revert evidence. |
| S-3 | P2 | Standards / manifest bookkeeping | `declared_flips.nodes` carries **no wave-6 entry**, yet Wave 6 removed Wave-0 nodes (e.g. `test_run_artifacts.py::test_forward_input_provider_mode_can_be_bound_once_per_mode[legacy_fused]`, `test_train_config.py::test_forward_input_provider_mode_accepts_exact_strict_values[legacy_fused]`) and rewrote others. The manifest's own note says "Any Wave-0 node NOT listed here encodes a preserved surface: an unexplained difference stops the wave." The BREAKING deletion was known at Wave 0 (design D11), so a flip entry was declarable in advance. | The **substance** is fully authorized — the proposal declares this exact deletion BREAKING — and the difference is *explained*, just in prose (close-out accounting: 8 nodes removed / 18 added, each a legacy-mode parametrize case or a renamed legacy test) rather than through the frozen mechanism. `revision_rule` was not violated: `node_paths_in_order` is file-level and unchanged, and the gate's `environment_selectors` remain accurate. Independently corroborated by replaying both the wave-6 gate (547/0/0) and the wave-0 argv (283/0/0) green. Traceability gap only; accept as residue. |
| S-4 | P3 | Standards / commit shape | Waves 0–5 each used a code commit plus a separate `docs(openspec): close decompose wave N` commit; Wave 6 bundles its close-out notes and docs updates into the single code commit `88391d6bb`. | Deviation from precedent, not from a rule. Independent revertibility is unaffected (verified). No action required. |
| O-1 | P3 | Overdesign / layering | The facade calls collaborator **private** names across module boundaries (`cache_workflow._establish_converged_runtime_determinism`, `session._initialize_model_free_run_owner`), and `run_writer`/`run_state` consume `run_schema._strict_*` privates. | A consistent internal convention of this decomposition, deliberately chosen so moved bodies stayed verbatim and Wave-0 helper names were preserved (Wave-1 disclosure (a)). Not machinery, not a new abstraction. Observation only. |
| O-2 | P3 | Overdesign / balance | `src/training/session.py` is 3,534 lines — roughly 53% of the 6,432 lines removed from `pipeline.py` landed in one new owner, making it the largest module in `src/training/`. | Explicitly sanctioned by design decision 8, and the move was proven verbatim (git detected `session.py` as an 89% copy of pre-wave `pipeline.py`; the builder's `difflib` proof reports 3,007 of 3,023 moved lines verbatim with zero deletions). This is under-decomposition relative to the headline, not overdesign, and is a candidate for a later change. No action in this change. |
| I-1 | P2 | Intent / ownership residue | `_default_qwen_forward` is an eval-only production symbol defined in `src/training/supervised_trainer.py:561` and imported across packages by its private name from `src/eval/forward.py:22`. | Lead disposition (a) is **sound** — the dependency is pre-existing (`src/eval/` byte-untouched since `eb2dc97ab`), the trainer is genuinely dead to it, and construction now fails closed with `trainer.forward_input_source_required`. Deleting it here would have rewritten 28 trainer call sites, colliding with this change's own preservation rule. Accept as deferred residue; the later wave that owns it should move it to the eval domain and make the import public. Blocks nothing. |
| I-2 | P3 | Intent / probe quality | Arm "D" fails closed only *after* `_write_arm_config` has run `config_path.parent.mkdir(parents=True)` and written the YAML, and `TRIAD_ORDERS` places `"D"` second in the first triad — so any future execution of the benchmark would abort mid-matrix leaving a partial artifact. | Lead disposition (b) is **sound**; this is a subordinate probe-quality note, not a defect in the audited change. No promoted claim depends on the r2 run. Note it for whoever eventually re-designs the arm set. |
| I-3 | P3 | Intent / test coupling | `tests/training/test_wave5_provider_benchmark.py` still asserts arm-D metadata mapping (lines 622–700, 1252, 1658), so the retired arm remains load-bearing on a green suite. | Consistent with retaining the arm as historical evidence. Whoever removes arm D must move these nodes with it. No action now. |
| L-1 | P2 | Launch gate / Wave-8 precondition | Task 9.2 requires an **absent artifact root** and "stop without retry on … occupied output", but the smoke config sets `run.artifact_root: outputs/smoke/production_mimic` (which **already exists** on disk) with `run.collision_policy: timestamp`. The config will therefore create a timestamped subdirectory rather than fail closed on an occupied root. | Not a defect in the audited commit and not a P0/P1 — the behavior is non-destructive. But the Wave-8 launch packet cannot rely on the config to enforce its own absent-target stop rule: it must name the exact timestamped run directory as the absent target and verify it externally, or set `collision_policy: fail` for the smoke. Hand this to the task-9.2 packet author. |

**P0: 0. P1: 0. P2: 5 (S-1, S-2, S-3, I-1, L-1). P3: 5 (S-4, O-1, O-2, I-2, I-3).**

Because there is no unresolved P0 or P1, the task-7.7 condition "no cache/GPU action may be
authorized with an unresolved P0/P1" is satisfied: **Wave-7 cache materialization and the Wave-8
two-rank GPU smoke are cleared to proceed to their own authorization packets.**

---

## 6. Commands executed (all read-only)

| # | Command | Observed |
| --- | --- | --- |
| 1 | `git rev-parse HEAD` | `88391d6bb208931ba3fa537fb9522ea6b106fe60` (at start and re-checked before this write) |
| 2 | `git status --porcelain` | empty at start; empty immediately before this write |
| 3 | `git log --oneline eb2dc97ab..HEAD` | 14 commits, listed in §1 |
| 4 | `git show --stat 3279d14bc` | `AGENTS.md \| 15 +++++`, 1 file changed |
| 5 | `git show --stat 88391d6bb` | 19 files, +329 / -193; docs + probes + 6 src files + 7 test files + tasks.md |
| 6 | `git diff --stat eb2dc97ab..HEAD` | 83 files, +21,545 / -8,956 |
| 7 | `git diff --stat 2ee6c4959..HEAD -- tests/fixtures/` | empty |
| 8 | `git log --oneline 2ee6c4959..HEAD -- tests/fixtures/` | empty |
| 9 | `git rev-parse {2ee6c4959,HEAD}:tests/fixtures/training_orchestration` | both `2fe137ccffc9b51e4eb227a91e2fbe6607dfc528` |
| 10 | `git diff --binary --no-color 88391d6bb^ 88391d6bb \| git apply -R --check` | **CLEAN** (1,062-line patch reverse-applies) |
| 11 | `git show eb2dc97ab:src/training/pipeline.py \| wc -l` / `wc -l src/training/pipeline.py` | `6588` -> `156` |
| 12 | `git show eb2dc97ab:src/eval/forward.py \| grep -n _default_qwen_forward` | lines 22, 269 (pre-existing) |
| 13 | `git diff --stat eb2dc97ab..HEAD -- src/eval/` | empty |
| 14 | `conda run -n ms openspec validate decompose-coordexp-swift-training-orchestration --strict` | `Change … is valid` (rc=0) |
| 15 | `conda run -n ms pytest <wave6-legacy-selector-gate argv> -q` | `547 passed in 300.63s`, RC=0; JUnit `tests=547 failures=0 errors=0 skipped=0` |
| 16 | `conda run -n ms pytest <wave0-baseline-red argv> -q` | `283 passed in 64.06s`, RC=0; JUnit `tests=283 failures=0 errors=0 skipped=0` |
| 17 | `conda run -n ms pytest <13 residue + new-owner suites> -q` | `746 passed in 220.83s`, RC=0; JUnit `tests=746 failures=0 errors=0 skipped=0` |
| 18 | `grep -rn 'legacy_fused\|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE' src/ configs/ openspec/specs/ docs/{COORDEXP_SWIFT,SYSTEM_OVERVIEW,IMPLEMENTATION_MAP,ARTIFACTS}.md` | zero hits |
| 19 | `grep -rn '…event_bus\|PluginRegistry\|DIContainer\|FSDP\|AbstractBackend…' src/` | zero production hits (6 vendored-inventory hits in `src/inference/qualification_receipts/vllm-0.14.1*.json`) |
| 20 | `grep -rn 'src\.qwen\.parity' src/` | zero hits |
| 21 | `grep -rn 'pipeline.py\|supervised_trainer.py' src/training/pack_cache.py` | zero hits |
| 22 | AST public-surface scan + caller grep over the 11 owner modules | table in §2; zero caller-free public names |
| 23 | `python -c "os.listdir('.cache/coordexp_swift/packing')"` | `[]` |
| 24 | `python -c` walk of `.cache/coordexp_swift/geometry_flip_aug_5step` | 2 targets `2232c868dae6…`, `9c82bc7578…`; 2 files each; mtimes 1783394670.996 / 1783394682.933 |
| 25 | `conda run -n ms python -c "load_train_config(<smoke config>)"` | `STRICT LOAD OK`; `synchronous`, `max_steps=1`, `outputs/smoke/production_mimic`, `collision_policy=timestamp` |
| 26 | `df -h /data` | `/dev/nvme0n1p1 3.5T 1.8T 1.6T 53%` |

All pytest invocations ran with `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and the manifest's
nine `required_absent` environment selectors unset. Scratch logs and JUnit XML were written to the
session scratchpad, never into the repository.

**Post-write expectation:** `git status --porcelain` will show exactly one untracked path —
`openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-6-pre-cost-audit.md`,
this receipt. Any other change would invalidate this audit.
