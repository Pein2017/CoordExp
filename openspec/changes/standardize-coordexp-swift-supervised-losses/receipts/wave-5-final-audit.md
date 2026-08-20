# Wave-5 Final Completion Audit (task 5.6)

- **Date:** 2026-08-20
- **Identity:** Opus final completion auditor, spawned by the Claude Fable lead.
  Distinct from the lead, all five wave builders, the Wave-0 entry auditor, and
  the Wave-3 pre-DDP auditor.
- **Frozen target:** HEAD `fa8233edf8455cf3c7e1780784887a65dfbe948a`; tracked
  tree clean; untracked = exactly `receipts/wave-5-smoke-packet.md` and
  `receipts/wave-5-smoke-receipt.json` (verified with
  `git status --porcelain --untracked-files=all`), plus the gitignored runtime
  dir `outputs/smoke/wave5_losses_smoke_r1/`. Re-verified before writing.
- **Commit chain audited:** `2a297a93a` (w0) -> `57a1c93cb` (w1) ->
  `e6f923534` (user's own AGENTS.md commit, disclosed) -> `2dde00427` (w2) ->
  `a20673078` (w3) -> `3ee9b02f3` (w3 close) -> `5779c006a` (w4) ->
  `fa8233edf` (w5 part 1). All wave close-out notes match their commits.
- **Scope:** read-only. This file is the only write.

## Verdicts

| Verdict | Result |
| --- | --- |
| **STANDARDS** | **PASS-WITH-DISPOSITIONS** |
| **USER-INTENT / CONTRACT** | **PASS-WITH-DISPOSITIONS** |
| **COMPLETION GATE** | **CLEARED** — 0 P0 / 0 P1; conditional on the close-out commit below |

Severity counts: **P0 = 0, P1 = 0, P2 = 1, P3 = 6.**

Every conclusion below was re-derived by this auditor at the frozen tree. No
verdict rests on a builder or lead self-report.

---

## 1. STANDARDS (re-derived)

### 1.1 Strict validation

```
$ conda run -n ms openspec validate standardize-coordexp-swift-supervised-losses --strict
Change 'standardize-coordexp-swift-supervised-losses' is valid          EXIT=0

$ conda run -n ms openspec validate --all
✓ change/standardize-coordexp-swift-supervised-losses   (+ 20 others)
Totals: 21 passed, 0 failed (21 items)                                  EXIT=0
```

All three spec deltas parse and resolve against the stable specs (strict mode
enforces this). Two are `## MODIFIED Requirements`
(`coordexp-swift-supervision-losses`, `coordexp-swift-training-artifacts`) and
one is `## ADDED Requirements` (`coordexp-swift-config-runtime`, adding
**Strict Supervised Loss Configuration**) — the archive sync must honour that
distinction. The F-7 **Non-Finite Loss
And Gradient Gates** delta appended by the lead is present at
`specs/coordexp-swift-supervision-losses/spec.md:247` and is carried by
`fa8233edf` (the only commit touching `specs/` in this change). It codifies the
raw-keyed pre-backward decision that Wave 2 implemented, including the explicit
"zero-weight gate ablation whose weighted contribution is exactly zero" clause —
so implementation and spec now agree rather than the code carrying an
undocumented behaviour.

### 1.2 Frozen-argv replays

```
$ conda run -n ms pytest tests/losses tests/config tests/artifacts -q
723 passed in 42.19s

$ conda run -n ms pytest tests/training/test_orchestration_compatibility.py \
      tests/runtime/test_rank_report_collective.py -q
15 passed in 41.23s
```

Both green, 0 skipped in the first. These cross-check the lead's full-suite
junit at `/data/CoordExp/.claude/jobs/c9895ff9/tmp/w5full.xml`:
`tests=2441 failures=0 errors=0 skipped=126 time=886.851`.

### 1.3 Judgement on the 126 skips

Parsed every `<skipped>` node in the junit. **All 126 carry one identical
reason** and one classname:

```
126  tests.training.test_reconcile_exact_resume_packet_executor
126  "historicized: live base config migrated by
      standardize-coordexp-swift-supervised-losses; the executor's frozen
      Attempt-6 byte pin fail-closes by design (see
      test_reconcile_exact_resume_packet_executor_historicized.py)"
```

So the 126 are **exactly the historicized executor module** — the 6 vacuous
parity mutations are **not** among them. They are not missing: they live in
`tests/qwen/test_packed_parity.py:1499`
(`test_v3_config_projection_rejects_any_live_config_drift[...]`, 6 params,
`pytest.mark.skip` reason "…every mutation would pass vacuously; the
unconditional drift refusal is asserted by
`test_v3_config_projection_refuses_the_migrated_live_config`"), and the frozen
`wave5-focused-full` argv (`pytest tests/config tests/losses tests/runtime
tests/training tests/artifacts tests/eval -q`) **never collects `tests/qwen`**.
That is an argv-scope note, not a suite regression, and this auditor closed it
by direct replay at HEAD:

```
$ conda run -n ms pytest tests/qwen/test_packed_parity.py \
      tests/training/test_reconcile_exact_resume_packet_executor.py \
      tests/training/test_reconcile_exact_resume_packet_executor_historicized.py -q -rs
285 passed, 132 skipped in 47.17s      # 132 = 126 executor + 6 vacuous parity
```

Verdict: the skip population is fully accounted for and correctly reasoned; no
test was silently disabled. Recorded as **F-G (P3)**.

### 1.4 Determinant invariant (re-run, read-only)

```
$ conda run -n ms python scripts/probes/coordexp_swift/losses_determinant_equality_probe.py
status OK, findings [], model_loaded false, wall_seconds 1.295
train        verdict EQUAL  fingerprint_equal true  payload_equal true
             8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f
eval.forward verdict EQUAL  fingerprint_equal true  payload_equal true
             3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662
cache_materialization_passes 0
```

Both splits EQUAL against `wave-0-determinant-baseline.json`
(baseline_sha256 `b2d9595f…`). The cache root inventory returned by my run —
6 files, `total_bytes 4254174`, per-file sha256 list — is **byte-identical** to
the inventory in `wave-5-determinant-equality-receipt.json`, which was captured
*before* the smoke. Since my run is *after* the smoke, this independently
establishes the packet's acceptance term "cache root inventory unchanged
before/after" from bytes rather than from the lead's prose claim.

Structural confirmation from the change diff: none of the 24 determinant-owner
files was touched (`src/losses/vocab.py`, `src/data/`, `src/packing/` all show
0 files in `git diff --name-only 0cb0ae729 fa8233edf`). The Wave-0 DO-NOT-EDIT
stop rule was honoured.

### 1.5 Smoke packet / receipt / disk coherence

Re-parsed `outputs/smoke/wave5_losses_smoke_r1/wave5_losses_smoke/logging.jsonl`
directly from bytes (3869 bytes, sha256 `498225451854ca05…`):

| Receipt claim | Disk (independently parsed) | Match |
| --- | --- | --- |
| rows = 2 | 2 (`split=train step=1`, `split=eval step=1`) | ✅ |
| train keys = 39 | 39 | ✅ |
| eval keys = 38 | 38 | ✅ |
| train `gate_raw` 1.367613136768341 | `loss/token_type_gate/raw` = 1.367613136768341 | ✅ |
| eval `gate_raw` 0.19928929209709167 | `loss/token_type_gate/raw` = 0.19928929209709167 | ✅ |
| gate weighted = 0.0 | `loss/token_type_gate/weighted` = 0.0 (both rows, exact) | ✅ |
| train `base_ce_raw` 5.013396978378296 | 5.013396978378296 | ✅ |
| eval `base_ce_raw` 3.347256660461426 | 3.347256660461426 | ✅ |
| train `selected_count` 560.0 | 560.0 | ✅ |
| eval `selected_count` 202.0 | 202.0 | ✅ |
| `total_equals_sum_weighted` true | `loss/total` == sum(weighted) exactly, both rows | ✅ |
| `coord_gaussian_rps_keys` 0 | 0 occurrences of the substring in the whole file | ✅ |
| `bare_loss_keys` 0 | 0 two-segment `loss/<term>` keys other than `loss/total` | ✅ |
| `backward_or_backend_keys` 0 | 0 occurrences of `backward` / `backend` / `compensat` / `unscaled` | ✅ |

Canonical families present exactly as the Wave-4 contract requires, per term:
`loss/<T>/{raw,weighted,selected_count,segment_count,token_weighted_diag}` +
`finite/<T>`, plus `finite/total_loss` and `loss/total`.

`run.json` (independently read): `status=completed`,
`final_finite_status=finite`, `final_optimizer_update_status=applied`,
`terminal_error=null`, `completed_steps=1`, `collision_outcome=fail`,
`world_size=2`, and both `cache_preparation` and `cache_publication` carry
`"status": "not_run"` (reasons
`single_process_preparation_precedes_training_launch` /
`immutable_cache_hit`). Materializations record exactly the two admitted
predecessor fingerprints.

Bounds table vs observed:

| bound | limit | observed | |
| --- | --- | --- | --- |
| devices / world size | 2 GPUs / ws=2 | `CUDA_VISIBLE_DEVICES=0,1`, `world_size: 2` in run.json | ✅ |
| planned steps | exactly 1 | 1 planned, 1 applied | ✅ |
| cache / materialization passes | REQUIRED 0 | 0 (both phases `not_run`) | ✅ |
| wall time | ≤ 900 s | created 13:00:42 → completed 13:01:07 ≈ 25 s | ✅ |
| peak GPU memory / rank | ≤ 32 GiB | 9.36 GB | ✅ |
| CPU RSS / rank | ≤ 16 GiB | 8.34 GiB | ✅ |
| artifact bytes | ≤ 2.5 GiB | 45,712,689 B (44 MB on disk) | ✅ |
| model forwards | 1 train/rank, ≤1 eval total | inferred: `micro_step_count=1`, `consumed_packs=1`, 1 train + 1 eval row | ⚠︎ |
| free disk at launch | ≥ 100 GiB | launch-time precondition, not reconstructable post-hoc | ⚠︎ |

⚠︎ = the two bounds that are launch-time preconditions rather than persisted
observations. The forward count is strongly supported (one `step=1` train row
with `micro_step_count=1` and one `step=1` eval row, `consumed_packs=1`) but is
inferred, not directly counted; free disk was checkable only at launch. All 7
other bounds are verified from persisted artifacts.

**Single-writer:** the run root contains exactly one directory,
`wave5_losses_smoke/`, with **no `-rankN` siblings** — a positive contrast with
every other family under `outputs/smoke/` (`production_mimic`, `eval_stream`,
`coord_pure_ce_typegate`, `geometry_flip_augmentation` all carry `-rank1..7`
dirs). Rank-zero-only artifact ownership is demonstrated, not asserted.

### 1.6 Manifest append-only; close-out/commit agreement

`git show --numstat` for every commit touching `command-manifest.json`:

```
2a297a93a  178 added / 0 deleted
57a1c93cb   34 added / 0 deleted
2dde00427   14 added / 0 deleted
3ee9b02f3   14 added / 0 deleted
5779c006a   27 added / 0 deleted
```

Zero deletions across the whole history: **strictly append-only**, matching the
manifest's own `amendment_rule`. 9 amendments recorded (amend-1 … amend-9).
Wave close-out notes in `tasks.md` name commits that exist and carry the work
they claim (verified against `git log` and the diff stats).

---

## 2. USER-INTENT / CONTRACT (per dimension)

### (a) SFT-only scope held — **PASS**

- 0 hits for `rollout|reinforce|ppo|grpo|policy_grad|hidden_state_loss|reward`
  across `src/losses/` and `src/config/models.py`.
- `configs/coordexp_swift/infer/` — 21 files, none carries a `losses` block;
  `git diff --name-only 0cb0ae729 fa8233edf | grep infer` returns **0 files**.
  The inference surface is untouched.
- The Future Auxiliary Loss Seam delta keeps "Hidden-state, rollout-derived, and
  RL loss composition remain unsupported unless a later approved change promotes
  them", with a scenario requiring an explicit unsupported-loss diagnostic.
- Docs state the boundary in three current-root files
  (`SYSTEM_OVERVIEW.md`, `COORDEXP_SWIFT.md`, `IMPLEMENTATION_MAP.md`):
  "there is no rollout-derived, hidden-state, policy, value, KL, or reward
  composition".

### (b) Scientific meaning executed as approved — **PASS**

Verified by **structural YAML parse** of every config under
`configs/coordexp_swift/**` (not grep):

```
configs with a losses block:            25
gate (mode, weight) distribution:       ('enabled', 0.1): 6
                                        ('zero_weight_ablation', 0.0): 19
coord_gaussian_rps under protected:     0
coord_gaussian_rps under auxiliary:     2  (1 prod + 1 smoke)
extra protected keys:                   0
base_ce weight != 1.0:                  0
VIOLATIONS:                             0
by directory: prod 5, smoke 16, smoke/length_isolation 4
```

This matches the approved proposal exactly: enabled gates at `0.1` only, zero
gates only under the *named* `zero_weight_ablation` mode, coordinate
Gaussian/RPS only under the typed auxiliary surface.

**Count reconciliation (21 vs 25), resolved:** entry-audit F-3 declared "21
supported configs = 5 prod + 16 smoke"; Wave 1 migrated 25. The 4 extra are the
measurement routes `configs/coordexp_swift/smoke/length_isolation/
physical_length_{6k,12k}_ebs64_{1,2}step.yaml`, which F-3 did not enumerate
under "prod + smoke". The change diff shows each of those 4 gaining exactly the
`+1` mode line. Nothing in F-3's scope was missed and nothing outside it was
migrated silently — the Wave-1 close-out's 25 is the complete number and the
proposal explicitly named "measurement/config routes" as in scope.

Independent replay of the frozen inventory probe:

```
$ conda run -n ms python scripts/probes/coordexp_swift/losses_config_inventory_probe.py
[result] OK: 25 supported configs resolved strictly; 1 historical config(s) rejected
```

**F-3 user-visibility** is satisfied: the Wave-0 close-out enumerates the exact
per-file scope of the `0.2`/`0.25` → `0.1` change (3 prod + 2 smoke at 0.2, 1
smoke at 0.25), the `0.0` → named-ablation flip, and the 2 auxiliary moves; the
Wave-1 close-out records the executed result (6 / 19 / 2) and states that the
`0.1` constant's authority is this change's approved proposal, design, and spec
deltas — not a builder's choice.

### (c) DDP math — **PASS**

- **Compensation exactly once.** `src/losses/runner.py:621` is the sole site:
  `backward_contribution = weighted if scale == 1.0 else weighted * scale`,
  applied only to the differentiable local contribution. `raw` is computed at
  line 605 with an explicit comment recording the pre-change defect (`raw = raw *
  backend_gradient_scale` made telemetry world-size dependent). At `scale == 1.0`
  the identical tensor is reused, so the world-size-one autograd graph is
  unchanged by the separation.
- **Telemetry clean.** Confirmed on the REAL two-rank smoke rows, not only in
  tests: 0 occurrences of `backward`/`backend`/`compensat` in `logging.jsonl`,
  and `loss/total` equals the exact sum of weighted terms on both rows.
- **Parity receipts sound.** `tests/runtime/test_wave3_two_rank_parity.py`
  documents and enforces `rtol=1e-5, atol=1e-6` for the grad/update comparison
  (stated as ~2 orders above fp32 reassociation noise; Wave 3 recorded mutation
  sensitivity 0.09/0.18, ~5 orders above the bar) and `torch.equal` (bitwise)
  for gate-ablation vs base-CE-only — a justified choice precisely because no
  gate op enters the objective graph. The Wave-3 two-rank probe receipt records
  exit 0, 0 findings, 12 collective ops per rank, 0 GPU, 0 cache passes.
- **Wave-5 remainder discharged.** The pre-DDP audit's I-1/I-6 carry-forward
  (`runner.py` micro-artifact `backward_contribution` fallback) now fail-closes:
  `_checked_micro_backward_contribution` raises `LossContractError` with code
  `loss.micro_artifact_backward_contribution_missing` rather than substituting
  the uncompensated `weighted_loss` (which would have divided the planned step's
  backward objective by the world size with no exception and no telemetry
  difference). Covering test
  `test_loss_runner_finalize_rejects_term_missing_backward_contribution` added
  in `fa8233edf`, and it is inside the `tests/losses` set I replayed green.
  This was the only candidate blocker in the carried-obligation ledger; it is
  closed.

### (d) Zero behavior at every boundary — **PASS**

| boundary | `forbid` (base_ce) | `detached_diagnostic` (token_type_gate) | `omit` (coord_gaussian_rps) |
| --- | --- | --- | --- |
| config | weight ≠ 1.0 rejected by strict validation | zero weight without the named mode rejected | absent / weight 0 accepted as omission |
| composition | runtime backstop `loss.base_ce_weight_forbidden` (runner.py:204-217) | always composed (runner.py:228) | not instantiated; instantiating one is a hard error `loss.auxiliary_omitted_term_instantiated` |
| autograd | n/a (always in objective) | `torch.no_grad()` boundary via `is_detached_diagnostic` (runner.py:400); excluded from the objective sum at runner.py:425 | no term, no call |
| denominator | built | built (inside no-grad) | none built or gathered |
| bundle / finite | in objective | retained; **finite keyed on RAW** (`finite_gates.py:45`, F-2) | no bundle entry, no finite check |
| rows | full family | full family, `weighted` = literal `0.0` | whole family absent |

Weighted zero is `raw.detach().new_zeros(())` — an exact `0.0`, deliberately not
`raw * 0.0`, so a non-finite raw diagnostic cannot leak into the weighted value
or the objective while still being caught by the raw-keyed all-rank gate.

**Observed in the REAL smoke rows** (the decisive evidence, since the smoke ran
the named ablation shape): gate raw **1.367613136768341** (train) and
**0.19928929209709167** (eval) visible; gate weighted exactly **0.0** on both;
`finite/token_type_gate = 1.0` retained; **zero** `coord_gaussian_rps` keys
anywhere in the file bytes.

### (e) Artifact compatibility — **PASS**

- **Old bare fields gone from current roots.** 0 bare `loss/<term>` keys in the
  real smoke rows. A literal sweep finds 24 `"loss/<word>"` hits repo-wide, and
  every one is a *pre-Swift Stage-1/Stage-2* term name — `token_ce`, `struct_ce`,
  `desc_ce`, `schema_format_ce`, `recursive_detection_ce`, `candidate_balanced`,
  `mp`, `pem`, `anti_close_start`, `adjacent_repulsion`, `coord_reg_obj`, … — in
  `tests/test_teacher_forcing_token_ce.py`, `tests/test_stage1_*`,
  `tests/test_stage2_*`, and `docs/history/`. **None** is a CoordExp-Swift
  supervised term (`base_ce`, `token_type_gate`, `coord_gaussian_rps`). No
  residue.
- **Historical evidence preserved.** No dual-write alias exists; docs state
  "historical JSONL keeps the schema of the commit that wrote it". Four frozen
  config-identity pin families that authenticate completed GPU evidence were
  historicized behind explicit skip reasons plus a live fail-closed refusal
  module — constants were never re-pinned.
- **Frozen fixtures byte-identical:**
  `git rev-parse HEAD:tests/fixtures/training_orchestration`
  = `2fe137ccffc9b51e4eb227a91e2fbe6607dfc528` ✅ (matches the required value).
  The only fixture movement in the whole change is the disclosed live
  loader-input file `tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`
  (+1 line), outside the frozen subtree.
- **No accidental historical edits.** `git diff --name-only 0cb0ae729
  fa8233edf` shows 0 files under `docs/history/`, `configs/archive/`,
  `openspec/changes/archive/`, or `outputs/`. The only `research/` touch is the
  change's own handoff note (+4 lines).

### (f) Migration completeness — **PASS**

- 25 supported configs: verified structurally (above), 0 violations.
- Inventory probe replayed: 25 resolved / 1 historical rejected.
- 9-file consumer migration: verified by an **f-string-aware** sweep of my own
  design. Every producer emits a suffixed key
  (`f"loss/{name}/raw"`, `/weighted`, `/selected_count`, `/token_weighted_diag`,
  `/segment_count` at `runner.py:521-530` and `1342-1348`;
  `f"loss/{name}{_TOKEN_WEIGHTED_DIAG_SUFFIX}"` at `eval/forward.py:518`). The
  only unsuffixed `f"loss/{name}"` occurrences are **negative absence
  assertions** (`assert f"loss/{name}" not in row`) in
  `tests/artifacts/test_wave4_loss_telemetry_rows.py:368` and
  `tests/eval/test_forward_eval.py:921-922`. This is exactly the class of
  consumer the Wave-0 literal grep missed (amend-9's methodological note); a
  fresh f-string-aware pass finds **no stragglers**.
- Four historicized pin families all accounted for and green at HEAD
  (285 passed / 132 skipped / 0 failed).

### (g) Overdesign — **PASS**

- `bindings` stayed **private**: `src/losses/__init__.py` contains zero
  references to `bindings`, `binding_for`, or `TokenLossBinding`. The only
  importer outside `src/losses/` is its own test module
  (`tests/losses/test_token_loss_bindings.py`). The `_binding_for_payload` hits
  in `scripts/probes/.../wave7_exact_resume_request.py` are an unrelated
  local function name, not this seam.
- **Closed, no registry/hooks:** 0 hits for
  `importlib|import_module|entry_points|__import__|import_path` in `src/config`
  and `src/losses`. The two `registry` string hits are doc comments *asserting
  the absence* ("there is no registry, no discovery, no import-by-name").
  `_active_token_losses` walks a compiled tuple — "a term that is not in this
  function is not composable at all".
- **Docs describe rather than restate:** the new `COORDEXP_SWIFT.md` section
  opens "Normative semantics belong to [supervision/losses spec] and [config
  runtime spec]; this section only routes operators to the shape a current
  config must author." It gives one authored YAML example and per-policy runtime
  behaviour, with cross-links instead of a second normative copy. No OpenSpec
  text is duplicated. Task 5.1's "without copying the OpenSpec change as a
  second authority" is satisfied.

### (h) Legacy residue — **PASS (structural, not grep)**

- 0 bare loss fields (see (e)).
- 0 protected coordinate placements — structural parse of all 25 configs.
- 0 gate weights outside `{(enabled, 0.1), (zero_weight_ablation, 0.0)}`.
- The `0.2` literals that survive in current configs are **legitimately
  different fields**, correctly distinguished by parsing rather than grepping:
  `rps_weight: 0.2` (2 configs — an auxiliary-internal Gaussian/RPS parameter,
  never a gate weight) and `vertical_prob: 0.2` (7 configs — geometry
  augmentation). No `0.25` remains anywhere in `configs/coordexp_swift/`.

---

## 3. Findings

| ID | Sev | Dimension | Finding | Disposition |
| --- | --- | --- | --- | --- |
| F-A | **P2** | standards / manifest | `command-manifest.json` has no amendment retiring the `wave5-vertical-smoke` placeholder, whose `argv` still reads `TO-FREEZE-IN-WAVE-5-PACKET`. Wave 3's parallel case *was* retired (amend-8, "retire the TO-FREEZE placeholder"). The manifest is untouched by `fa8233edf`, so neither the executed smoke argv nor the Wave-5 determinant run is pinned in the ledger. | Substance is bound: `wave-5-smoke-packet.md` freezes the exact command and the receipt binds to the packet, commit, config, cache identities, and bounds — all independently re-verified above. This is ledger completeness, not evidence loss. **Must-include in the close-out commit** (amend-10). |
| F-B | P3 | provenance | The Wave-5 determinant receipt records `commit: 5779c006a` with `tree_dirty: true` (`started_at` 12:57:29Z) and the full-suite junit is stamped 12:57:21Z — both **predate** `fa8233edf` (committed 12:58:18Z). The smoke packet describes the probe as run "at the same tree". | Content-accurate: the Wave-5 edits were in the working tree when both ran. `git diff --stat 5779c006a fa8233edf` is exactly 10 files — docs (4), the `losses_determinant_equality_probe.py` script, the `runner.py` guard, its test in `tests/losses/test_runner.py`, the `coordexp-swift-supervision-losses` spec delta (+55, the F-7 requirement), the determinant receipt JSON, and the handoff note — and this auditor re-anchored **every** one of those surfaces at the **clean** `fa8233edf`: strict + `--all` validation (covers the spec delta), determinant probe EQUAL/EQUAL (covers the probe script and its receipt), `tests/losses+config+artifacts` 723/0/0 (covers the guard and its test), compat+collective 15/15, pin bundle 285/132/0, and the docs read directly. Record the note; no re-run needed. |
| F-C | P3 | receipt schema | The Wave-3 close-out required the Wave-5 receipt schema to add `commit`, `wall_seconds`, and a cache-root sha inventory. `commit` is present; **`wall_seconds` is absent**, and the inventory appears as the prose claim `"byte-identical before/after (lead diff)"` rather than as bytes. | Substance independently closed here: my post-smoke probe re-derived the full 6-file sha256 inventory (`total_bytes 4254174`) identical to the pre-smoke capture, and `run.json` timestamps bound wall time to ≈25 s against the 900 s limit. Record as a schema-completeness note. |
| F-D | P3 | DDP / fail-closed | `LossTermResult.__post_init__` (`runner.py:69-73`) still defaults `backward_contribution` to `weighted_loss` when `None` — the last residual instance of the I-1/I-6 silent-substitution shape. | Unreachable in the live path: both construction sites set it explicitly (`runner.py:614`, `:621`) and the micro-artifact path now fail-closes. Non-blocking; carry forward as a hardening candidate. |
| F-E | P3 | evidence scope | The production-shaped smoke exercised the **`zero_weight_ablation`** row shape only; enabled-gate `0.1` evidence is unit/integration-tier plus the Wave-3 two-rank parity probe. | Declared in the packet itself ("the smoke therefore exercises the named ablation row shape"), so this is a disclosed scope boundary, not a gap. It happens to be the stronger choice for dimension (d), since the ablation is the shape where zero-behavior can silently fail. |
| F-F | P3 | test hygiene | Wave 4 disclosed a 1-of-9 non-reproducing failure in the untouched Wave-2 probe `test_gate_ablation_creates_no_autograd_edge_into_the_objective` (attributed to cross-module global-state leakage into `plan.backend_gradient_scale`), parked "for Wave 5.5 review". | **Disposed, not skipped:** the full-suite junit shows 0 failures / 0 errors over 2441 tests, and my `tests/losses` replay was green. Pre-existing test-isolation hazard; not absorbed by this change; carry forward. |
| F-G | P3 | standards / argv | The frozen `wave5-focused-full` argv does not collect `tests/qwen`, which holds the FROZEN_V3 parity pin family this change modified (46 lines in `tests/qwen/test_packed_parity.py`). Hence 126 skips in the junit, not 132. | Closed by direct replay at HEAD: 285 passed / 132 skipped / 0 failed. Record the composition and the argv-coverage note in the close-out. |

**Out-of-scope carry-forwards (correctly excluded, source TODOs carry
provenance):** train-side `count/packs` + `count/examples` mean-over-ranks and
unweighted `token_weighted_diag` (introduced `2b0a2165a`); zero-eligible-segment
collective desync (`e1662c2c7` / `3cd40f5f0`). Both predate this change and were
explicitly declared out of scope by Waves 3 and 4.

---

## 4. Completion statement

**The change MAY be marked implementation-complete and archived (spec deltas
synced).** There are **0 P0 and 0 P1 findings**. Every dimension enumerated by
task 5.6 — SFT-only scope, scientific meaning, DDP math, zero behavior, artifact
compatibility, migration completeness, overdesign, legacy residue — verdicts
**PASS**, each re-derived at the frozen tree rather than accepted from a
receipt. The single carried obligation that could have blocked
(`runner.py` micro-artifact `backward_contribution` fallback) is discharged with
a fail-closed guard and a covering test.

The one P2 (F-A) is a ledger-completeness gap with no evidence loss and is
listed as a required element of the close-out commit below.

### The final close-out commit MUST include

1. **`tasks.md`:** tick 5.1–5.6 `[x]` and add the Wave-5 close-out note, naming
   `fa8233edf` + the close-out commit, the three Wave-5 receipts, and the
   F-B/F-C provenance dispositions.
2. **The two untracked smoke receipts** (`wave-5-smoke-packet.md`,
   `wave-5-smoke-receipt.json`) **and this audit receipt**
   (`wave-5-final-audit.md`) committed — the tree must be clean before archive.
3. **`command-manifest.json` amend-10** retiring the `wave5-vertical-smoke`
   `TO-FREEZE` placeholder with the actually-executed argv, devices
   (`CUDA_VISIBLE_DEVICES=0,1`), and observed bounds — mirroring amend-8's
   wording. Recommended amend-11: pin the observed `wave5-focused-full` counts
   (2441 passed / 0 failed / 0 errors / 126 skipped, 886.85 s) and note that the
   argv excludes `tests/qwen`, with the 285/132/0 replay as its closure.
4. **Recorded notes:** F-B (junit and determinant receipt predate `fa8233edf` by
   ~60 s; re-anchored by this audit's clean-HEAD replays), F-C (receipt lacks
   `wall_seconds`; inventory re-derived from bytes here), F-G (126-skip
   composition = the historicized executor module in full).
5. **Archive with spec-delta sync** of all three deltas, including the F-7
   **Non-Finite Loss And Gradient Gates** requirement added in `fa8233edf`.
6. **Carry forward as non-blocking** into the successor change or a research
   note: F-D (`LossTermResult` `None` default), F-F (Wave-4 test-isolation
   flake), and the two pre-existing count defects plus the zero-eligible-segment
   collective desync.

---

## 5. Commands executed (with observed output)

Host rules honoured: `PYTHONDONTWRITEBYTECODE=1` exported inside
`bash -c 'export …; …'` (never an `env` prefix, which is silently no-op'd on
this host); helper scripts written to
`/data/CoordExp/.claude/jobs/c9895ff9/tmp/` rather than heredoc'd into
`conda run`. No command hung; nothing exceeded 20 minutes (longest ≈ 47 s).
No GPU action, no cache mutation, no commit.

```
$ git rev-parse HEAD
fa8233edf8455cf3c7e1780784887a65dfbe948a

$ git status --porcelain --untracked-files=all
?? openspec/changes/.../receipts/wave-5-smoke-packet.md
?? openspec/changes/.../receipts/wave-5-smoke-receipt.json

$ git rev-parse HEAD:tests/fixtures/training_orchestration
2fe137ccffc9b51e4eb227a91e2fbe6607dfc528

$ conda run -n ms openspec validate standardize-coordexp-swift-supervised-losses --strict
Change 'standardize-coordexp-swift-supervised-losses' is valid            EXIT=0

$ conda run -n ms openspec validate --all
Totals: 21 passed, 0 failed (21 items)                                    EXIT=0

$ conda run -n ms pytest tests/losses tests/config tests/artifacts -q
723 passed in 42.19s

$ conda run -n ms pytest tests/training/test_orchestration_compatibility.py \
      tests/runtime/test_rank_report_collective.py -q
15 passed in 41.23s

$ conda run -n ms pytest tests/qwen/test_packed_parity.py \
      tests/training/test_reconcile_exact_resume_packet_executor.py \
      tests/training/test_reconcile_exact_resume_packet_executor_historicized.py -q -rs
285 passed, 132 skipped in 47.17s

$ conda run -n ms python scripts/probes/coordexp_swift/losses_determinant_equality_probe.py
status OK / findings [] / train EQUAL / eval.forward EQUAL /
payload_equal true both / cache_materialization_passes 0 / wall_seconds 1.295
cache inventory: 6 files, total_bytes 4254174 (byte-identical to the pre-smoke receipt)

$ conda run -n ms python scripts/probes/coordexp_swift/losses_config_inventory_probe.py
[result] OK: 25 supported configs resolved strictly; 1 historical config(s) rejected

# junit analysis (helper: tmp/audit_junit.py, tmp/audit_junit2.py)
suite: tests=2441 errors=0 failures=0 skipped=126 time=886.851
all 126 skips -> tests.training.test_reconcile_exact_resume_packet_executor, one reason

# real smoke row re-parse from bytes (helper: tmp/audit_rows.py)
logging.jsonl 3869 bytes sha256 498225451854ca051334a12a1b0e34406d0a87e602114b8781fccfcf24582d21
2 rows; train 39 keys / eval 38 keys; gate weighted 0.0 exact both rows;
loss/total == sum(weighted) exact both rows; 0 coord_gaussian_rps / bare-loss /
backward / backend / compensat occurrences

# structural config parse (helper: tmp/audit_configs.py)
25 configs with a losses block; enabled@0.1: 6; zero_weight_ablation@0.0: 19;
coord under protected: 0; coord under auxiliary: 2; VIOLATIONS: 0

# residue sweep, f-string aware (helper: tmp/audit_residue.sh)
bare loss/<term> literals: 24, all legacy Stage1/Stage2 names + docs/history/
dynamic loss hooks in src/config + src/losses: 0
RL/rollout/hidden-state in src/losses + src/config/models.py: 0
gate weights outside {0.1 enabled, 0.0 ablation}: 0
infer/ files in the change diff: 0
determinant-owner files in the change diff: 0
docs/history|configs/archive|openspec archive edits: 0

$ git show --numstat <each commit> -- receipts/command-manifest.json
178/0, 34/0, 14/0, 14/0, 27/0   -> strictly append-only, 9 amendments
```
