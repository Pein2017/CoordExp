# Wave-8 Final Standards / Overdesign / Intent-Contract Audit

| Field | Value |
| --- | --- |
| Change | `decompose-coordexp-swift-training-orchestration` |
| Audited HEAD | `d598f8894a3e2d0f3a5c407b685b344813a58dd3` (Wave-7 commit; source bytes identical to the pre-cost-audited Wave-6 commit `88391d6bb…`) |
| Date | 2026-08-20 |
| Auditor | Opus final completion auditor, spawned by Claude Fable lead (single independent auditor of record for task 9.6; distinct from every builder, from the lead, and from the Wave-6 pre-cost auditor) |
| Task | 9.6 — the single final read-only standards / overdesign / intent-contract audit |
| Mode | Read-only. The only file written by this audit is this receipt. No commits, no source/test/fixture/tasks.md edits, no `src.prepare_train_cache`, no GPU command, no cache mutation. |
| Working tree at audit start | exactly the declared pending-final-commit set (below) |
| Working tree re-check before this write | HEAD still `d598f8894…`; `git status --porcelain` still exactly the same five paths. Receipt is **not** VOID-DRIFT. |

### Pending-final-commit set (sha256 at audit time)

| Status | Path | sha256 |
| --- | --- | --- |
| `M` | `docs/COORDEXP_SWIFT.md` | `a8eefe1b4904bb8bdc4b9e96936431acdfa39818a8139c4e3170ff6eab945e71` |
| `M` | `openspec/changes/…/tasks.md` | `1fc24771213d64b3bac28541e7013e08f4ac7f37eb40cf8d1f729b316dfb41fc` |
| `??` | `…/receipts/wave-8-compatibility-comparison.json` | `c099b60ca69387e4bd7579efda73c60c49bf2b4a46585ef33639164a3c090c7d` |
| `??` | `…/receipts/wave-8-cpu-matrix.json` | `f6b005cb2e249341e59f249b26f0f8b748e46910b61ddb06278f005ab0743d9c` |
| `??` | `…/receipts/wave-8-gpu-launch-packet.md` | `51191c7c041a29e580d3f67d4aa0261788924325996b4a985b7d13a3f4e14895` |

No tracked modification outside this set; no HEAD drift. This receipt itself becomes a sixth untracked path after the write, which is expected and licensed.

## Verdicts

| # | Area | Verdict |
| --- | --- | --- |
| 1 | STANDARDS — Waves 0–8 against the change's own rules (Wave-7/8 additions verified in depth; Waves 0–6 inherited from `receipts/wave-6-pre-cost-audit.md`, 0 P0 / 0 P1) | **PASS-WITH-DISPOSITIONS** |
| 2 | OVERDESIGN — did Waves 7–8 add machinery beyond receipts, docs rows, and smoke runtime artifacts? | **PASS** |
| 3 | INTENT-CONTRACT — does the completed change deliver its stated intent end-to-end? | **PASS-WITH-DISPOSITIONS** |
| 4 | COMPLETION GATE — may the change take its final commit and be marked complete/archived? | **CLEARED** (0 P0 / 0 P1), conditional on the mandatory final-commit items in §6 |

Severity counts: **P0 = 0, P1 = 0, P2 = 2, P3 = 5.**

---

## 1. STANDARDS — PASS-WITH-DISPOSITIONS

### 1.1 Wave-7 packet / receipts coherence — PASS (verified, not inherited)

All four Wave-7 receipts, the cache-action packet, and the on-disk cache agree exactly.

- Preparation receipt (`wave-7-cache-preparation.json`): schema `coordexp-swift-pack-cache-preparation-receipt-v1`, `terminal_status = "completed"`, `failure = null`, **both splits `build_status = "built"`**, both `status = "complete"`, `micro_step_count` 2 (train) / 1 (eval).
- Verification receipt (`wave-7-cache-verification.json`): schema `coordexp-swift-pack-cache-verification-receipt-v1`, `terminal_status = "completed"`, **both splits `build_status = "hit"`**, `verified_splits = ["train", "eval.forward"]`, `verification_level = "payloads"`, `cache_materialization_authorized = false` — the packet's "no materialization authority" clause is enforced in the receipt itself, not merely asserted in prose.
- Both receipts carry the identical `resolved_config_fingerprint` `fd255b40ecf2e835…`, and both `measurement.context.workload_identity` values equal it.
- Fingerprints are consistent across **all four** surfaces plus disk:

| Surface | train | eval |
| --- | --- | --- |
| `wave-7-determinant-projections.json` `new_fingerprint` | `8f11237f…76f2f` | `3b30c157…af6662` |
| preparation receipt | same | same |
| verification receipt | same | same |
| `wave-7-cache-action-packet.md` absent-target list | same | same |
| `wave-8-gpu-launch-packet.md` "new cache" | same | same |
| smoke `run.json` `materializations` + `policy_identities.cache` | same | same |
| on-disk `.cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/` | present | present |

- Cache root contains **exactly two** fingerprint directories (plus their two lock files) — no third/intermediate fingerprint anywhere under the root.
- Manifest bytes re-hashed independently by this audit: train `manifest.json` sha256 `c1d9fea76d063c6b…e025e8`, eval `8de1fda29784aad2…d92992` — **both match** the `manifest_sha256` recorded in the preparation *and* verification receipts, i.e. the target the `--require-all-hit` pass admitted is byte-identical to the one the build published, and is still byte-identical now, after the Wave-8 GPU smoke consumed it.
- Old immutable evidence unchanged: `.cache/coordexp_swift/geometry_flip_aug_5step/{2232c868dae6…, 9c82bc7578…}` still 2 files each with mtimes `1783394670.996` / `1783394682.933` — **identical to the values the pre-cost audit recorded before any Wave-7/8 action**.
- Projections: 31 determinants per split; `content_identity_changed` **empty** on both splits; `owner_changes` exactly the two declared moves (`micro_step_runtime_config` `pipeline.py`→`cache_contract.py`, `micro_step_schema` `supervised_trainer.py`→`micro_steps.py`); `owner_source_changed` exactly the four declared determinant sources. Content-identity-preserving turnover is proven, not asserted.
- Packet conformance against `test-command-manifest.json` → `launch_bearing_actions.wave_7_cache_materialization.packet_freezes`: both required freezes (one build-capable argv + absent receipt path; one `--require-all-hit` argv with a *distinct* absent receipt path and no materialization authority) are present in the packet and honoured by the two receipts.

### 1.2 Wave-8 CPU-matrix receipt — the 11-failure pollution diagnosis

**The diagnosis is sound and complete. The pass-rule framing is not.** These are separate questions and the receipt/tasks.md note conflates them.

*Diagnosis — independently confirmed by this audit (not accepted on the lead's report):*

- I re-ran the minimal deterministic pair in a fresh process (one CUDA bf16 qwen test, then the CPU-purity probe): **1 failed, 1 passed in 5.48 s**, failing with `Wave6ProbeError: "CUDA was initialized before CPU comparison"`, `code="wave6.gpu_forbidden"`, raised at `scripts/probes/coordexp_swift/wave6_pack_plan_comparison.py:2060`. The claimed mechanism reproduces exactly.
- I re-ran all three files owning the 11 failing nodes in one fresh process at this commit: **48 passed / 0 failed / 0 errors / 0 skipped in 164.05 s**. I then checked the JUnit XML node-by-node: **all 11 of the receipt's declared `failing_nodes` are present in that run and all 11 are green** (0 missing, 0 non-green).
- I re-ran `wave8-final-compatibility-gate` verbatim: **231 passed / 0 failed / 0 errors / 0 skipped in 44.60 s**, confirming the claimed 231/0/0.

The four-receipt argument in `wave-8-cpu-matrix.json` is therefore *complete and true*: a code regression is impossible given that every failing node is green in a same-commit fresh process, and the failure is a fail-closed CPU-purity guard tripping on CUDA state that earlier suites in the same pytest process created. The receipt is also unusually honest in recording the diagnostic CUDA-hidden run that **flips** the failure set (11 pass, 69 CUDA-requiring nodes fail) — that is what proves the frozen argv cannot be simultaneously single-process and CUDA-pure on this host, rather than merely asserting an excuse.

*Pass-rule — see finding **F-1**.* Under the manifest's own `pass_rule`, `wave8-full-cpu-matrix` has `expected_red_nodes = []` and no `declared_flips` remaining unrevised at wave 8, so its expected failure set is **∅**; the observed set is 11. The gate does **not** pass, and `revision_rule` ("Any added or changed command, node path, environment selector, or expected-RED entry requires an explicit manifest revision plus a recorded baseline rerun") offered the compliant path — declaring the 11 as expected-RED with this diagnosis, or declaring a CUDA environment selector / process-isolation change — which was not taken. The tasks.md note's phrase "Zero unexplained failures" borrows the *declared-flips* rule ("an unexplained difference stops the wave"), which is about preserved Wave-0 surfaces, not about whether a manifest command passed. Both statements can be true at once and the note should say both.

### 1.3 Wave-8 GPU packet vs. execution — PASS

Every observed value quoted in the tasks.md Wave-8 note was re-derived by this audit from `outputs/smoke/wave8_vertical_smoke_r1/wave8_vertical_smoke/`:

| Ceiling (packet) | Limit | Observed (verified by this audit) | Within? |
| --- | --- | --- | --- |
| per-rank train model forwards | exactly 1 | 1 train row, `micro_step_count` per rank = 1, `consumed_packs = 1` | yes |
| eval model forwards (total) | ≤ 1 | 1 eval row at step 1 | yes |
| per-rank collective operations | ≤ 100 | **no production counter exists** — see finding F-7 | not measured |
| wall time | ≤ 900 s | `entry_to_terminal.duration_seconds = 40.245` | yes |
| CPU peak RSS per rank | ≤ 16 GiB | rank 0 `8,963,375,104 B` (8.35 GiB), rank 1 `8,959,627,264 B` — **not the 1.18 GiB the note records**, see F-2 | yes |
| GPU high-water per rank | ≤ 32 GiB | rank 0 alloc `9,356,426,240 B` / reserved `10,412,359,680 B`; rank 1 alloc `7,171,412,480 B` / reserved `7,713,325,056 B` | yes |
| artifact bytes under the run root | ≤ 2.5 GiB | 45,712,517 B in 11 files + 20,480 B of directory entries = **45,732,997 B**, exactly the recorded figure | yes |
| free disk on /data | ≥ 100 GiB | `df -BG /data` → 1,597 G available | yes |

Lifecycle claims all confirmed from `run.json`: `status = completed`, `terminal_error = null`, `completed_steps = 1`, `resolved_max_steps = 1`, `final_optimizer_update_status = applied`, `final_finite_status = finite`, `checkpoint_event_count = 1`, `collision_outcome = "fail"`, `warning_counts = {}`, `cache_preparation`/`cache_publication` phases `not_run` (admission-only, as the frozen fixture requires), `steady_state` `not_run` (config-shaped at `max_steps=1`), terminal phase `evaluation_execution` `completed`, `failure_phase = null`.

**Overlay-config legitimacy under "named 1-step config" — sound.** The packet names and sha256-binds the base config `…_ebs2_1step.yaml` (`44c2cd2a…`), and the overlay is three lines that `extends` it verbatim and override only `run.name` / `run.artifact_root` / `run.collision_policy`. This is exactly what pre-cost finding **L-1** demanded: the base config's `outputs/smoke/production_mimic` root *already exists* with `collision_policy: timestamp`, which cannot enforce task 9.2's "absent artifact root" + "stop without retry on occupied output" rule. Setting `collision_policy: fail` on a verified-absent root restores the fail-closed stop rule the task requires; `run.artifact_root = outputs/smoke/wave8_vertical_smoke_r1` and `collision_outcome = "fail"` in the produced `run.json` confirm the overlay took effect. The `run.*` fields are proven non-determinants by the Wave-4 probe, and the produced cache identities are the Wave-7 fingerprints unchanged — so the overlay cannot have altered what was exercised. Packet conformance against `launch_bearing_actions.wave_8_gpu_vertical_smoke.packet_freezes` (world_size=2, ≤2 GPUs, one planned/applied step, the six named ceilings) is complete. Durability note only: F-6.

### 1.4 Wave-8 compatibility-comparison receipt method — honest, but weaker than the evidence allows

The three restrictions were each checked for whether they hide a drift. **None does.**

1. *"timestamp-comparable order"* — I confirmed the factual premise: `cache_identity_resolution`, `cache_publication_admission`, `train_rank_hydration`, `cache_preparation`, `cache_publication`, and `evaluation_execution` genuinely carry `started_at: null` in the production `run.json`, so a timestamp-derived order really can only see 2 of the fixture's 7 phases. But the method_note's stated reason ("run.json phases dict is key-sorted, not execution-ordered") overlooks `measurement.phase_order`, which `src/artifacts/run_writer.py` **appends to at runtime** (lines 389/429/482/536/669) and is therefore a true execution-order list. Comparing on that field, the fixture's `run_state_phase_order` is an **exact 7-element prefix** of the production `phase_order` — I verified this equals `True`. The available evidence is strictly stronger than what the receipt claims, in the same direction. See F-4.
2. *"subset row-schema"* — the fixture's 14 protected keys are all present in the production train row (`missing_from_production_train_row: []`, independently confirmed against `logging.jsonl`), and the production extras are config-shaped (`loss/base_ce`, `loss/token_type_gate`, `lr/group_1`, `count/*`) plus per-rank resource keys the model-free harness never emitted. Honest.
3. *"no-comparator eval row"* — the Wave-0 fixture genuinely contains no eval-row comparator; presence-and-step is the strongest check available from the ledger. Honest.

`run_state_phase_status` is compared in full (7/7 equal, `mismatches: {}`), and the extra phases are exactly the model-bearing lifetime the model-free harness cannot contain. `unexplained_differences: []` is corroborated.

### 1.5 Strict validation

`conda run -n ms openspec validate decompose-coordexp-swift-training-orchestration --strict` → `Change 'decompose-coordexp-swift-training-orchestration' is valid`, **rc = 0**.

---

## 2. OVERDESIGN — PASS

- `git diff 88391d6bb..HEAD --stat -- src/ tests/ scripts/ configs/` → **empty**. Not one source, test, script, or config byte changed after the audited Wave-6 commit.
- The pending dirty set contains **no source path** — one docs row, one tasks.md close-out, three receipts. Nothing else.
- The only new non-repo bytes are the two Wave-7 cache targets under `.cache/` and the smoke run under `outputs/smoke/wave8_vertical_smoke_r1/`; both trees are gitignored (`.gitignore:2` `*` under `.cache/`, `.gitignore:55` `/outputs/`), so neither can leak into the final commit.
- The single docs edit is one table row in `docs/COORDEXP_SWIFT.md` naming the accepted new owners (`execution_plan`, `control_plane`, `cache_workflow`, `session`) alongside the retained `pipeline.py` facade — exactly what task 9.5 authorizes, with no efficiency language.

Waves 7–8 added **only** receipts, one docs row, and runtime artifacts. No abstraction, no helper, no tooling, no machinery.

---

## 3. INTENT-CONTRACT — PASS-WITH-DISPOSITIONS

The change delivers its stated intent end-to-end:

- **Facade decomposition governed by existing stable specs (`skip_specs`).** Strict validation passes with no delta specs; the pre-cost audit verified `pipeline.py` 6,588 → 156 lines with zero caller-free public names across the 11 new owners, and the boundary suite `tests/training/test_training_module_boundaries.py` is green inside my 231/0/0 replay.
- **Single authorized cache transition with content-identity-preserving turnover.** Exactly one build-capable invocation; `content_identity_changed` empty on both splits; exactly two fingerprints on disk; the `--require-all-hit` pass recorded `cache_materialization_authorized = false` and two hits with byte-identical manifests; all pre-existing immutable targets unchanged by mtime.
- **Production-shaped two-rank smoke consuming the new cache, ledger honoured.** `world_size=2` on two GPUs, one applied optimizer step, both Wave-7 fingerprints admitted, `cache_preparation`/`cache_publication` `not_run`, cache manifests still byte-identical after the run, run terminal `completed`, and the ledger comparison recording zero unexplained differences with the two intentional ones evidenced in `run.json` (`forward_input_provider_resolution.source = "strict_config"`, `resolved_mode = "synchronous"`).
- **BREAKING legacy deletion.** Zero `legacy_fused` / `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` hits in production surfaces (pre-cost audit cmd 18, unchanged source bytes since), and the production run resolves the mode from strict config with `environment_variable: null`, `is_semantic_override: false`.

### Adjudications

**(a) The 9.1 disposition — accepted at P2, not cleared as compliant.** Recording a non-green canonical matrix with a fully diagnosed environmental failure set is *substantively* acceptable: task 9.1's own text is "run … and record exact command, commit, pass/fail count, duration, and skipped tests" — a run-and-record obligation, which was discharged verbatim, with zero skips (skips being the thing that rule most cares about, since "a skip is reported, never counted as executed coverage"). The failures are **genuinely explained**, and I confirmed that independently twice rather than accepting the lead's report. What is *not* satisfied is the manifest's `pass_rule` set-equality, and the compliant remedy (`revision_rule`) was available and unused. This is the same class of finding as the pre-cost audit's **S-3** — substance authorized, frozen mechanism not revised — which that audit dispositioned at P2; the recurrence of the class at Wave 8 is worth the lead's notice but does not escalate, because no hidden regression is possible on the replay evidence above. Disposition: accept at P2 with a mandatory disclosure correction (§6).

**(b) The overlay-config approach to 9.2 — sound, no finding above P3.** See §1.3. The overlay is the *stronger* reading of task 9.2, not a weakening: it is the only construction that makes "absent artifact root" and "stop on occupied output" simultaneously enforceable while still binding the named 1-step config. L-1 is fully consumed.

**(c) The collective-count ceiling — accepted at P3.** The `≤ 100 per-rank collectives` ceiling was derived, not measured, because no production per-rank collective counter exists in the runtime. The discharge offered — the green collective-order suite (`tests/runtime/test_rank_report_collective.py`, inside my 231/0/0 replay, asserting frame width, magic, payload bound, control timeout, and eight-rank collective ordering) plus completion in 40.2 s against a 900 s hang bound — is the strongest evidence obtainable at this commit: a mismatched or extra collective on one rank would have deadlocked both ranks into the timeout, and it did not. The tasks.md note is candid that no counter exists. It supports "no collective-order divergence"; it does **not** support a *count* claim, and none is made. Acceptable; a real counter belongs to a later observability change.

**(d) Pre-cost P2 residue consumption — verified.**

| Pre-cost finding | Required of Wave 7/8 | Status |
| --- | --- | --- |
| **S-1** (P2) — stale, self-contradictory Wave-6 close-out note (f) pointing at a non-existent `wave-6-gate.json` | lead corrects note (f) | **LANDED** at `d598f8894` (commit message says so; verified in `tasks.md` lines 553–563: bracketed correction marker, 7.7 recorded as fully executed, revert proof and audit receipt cited, dangling receipt path explicitly retired as dropped-per-precedent) |
| **S-2** (P2) — Wave-6 revert proof not executed by the wave owner | discharged by the pre-cost audit itself | **CLOSED**; note (f) now cites it |
| **S-3** (P2) — no wave-6 entry in `declared_flips` for the BREAKING deletion | accepted as residue | unchanged; recurs in class as F-1 |
| **I-1** (P2) — `_default_qwen_forward` eval-only residue in `supervised_trainer.py` | deferred to a later change | unchanged (source bytes frozen); blocks nothing |
| **L-1** (P2) — smoke config cannot enforce the absent-target stop rule | consumed by the task-9.2 packet | **CONSUMED**; the packet names L-1 explicitly and resolves it with the `collision_policy: fail` overlay on a verified-absent root (§1.3) |

All five pre-cost P2s are consumed, discharged, or knowingly carried. The five P3s are unchanged and none is Wave-7/8-scoped.

---

## 4. Findings

| ID | Sev | Area | Description | Disposition |
| --- | --- | --- | --- | --- |
| **F-1** | **P2** | Standards / manifest pass-rule | `wave8-full-cpu-matrix` declares `expected_red_nodes: []` and carries no unrevised `declared_flips` at wave 8, so its `pass_rule` expected failure set is ∅; the canonical run observed 11 failures. Under the frozen `pass_rule` the command **does not pass**, and `revision_rule` offered the compliant remedy (declare the 11 expected-RED with the diagnosis, or declare the CUDA/process-isolation selector) which was not taken. `tasks.md` records "Zero unexplained failures" — the *declared-flips* rule's language — which is true but is not the pass-rule question and reads as if the gate passed. | Accept at P2. Task 9.1's own text is run-and-record, not "gate"; the diagnosis is complete and I independently reproduced both the pollution mechanism (1F/1P, `wave6.gpu_forbidden`) and the fresh-process green replay of all 11 nodes (48/0/0/0), so a hidden regression is impossible. Same class as pre-cost S-3, dispositioned there at P2. **Mandatory:** the final commit must state the pass-rule deviation plainly rather than only "zero unexplained failures" (§6.2). |
| **F-2** | **P2** | Records / receipt accuracy | The tasks.md Wave-8 note records "CPU RSS 1.18 GiB (<=16 GiB)". Observed `resource_high_water.cpu.max_rss_bytes` is **8,963,375,104 B** (rank 0) and **8,959,627,264 B** (rank 1) — ≈ 8.35 GiB, ~7× the recorded figure. Confirmed in both `run.json` and both `per_rank_measurement` blocks of `logging.jsonl`. | No bound exceedance: 8.35 GiB is well inside the 16 GiB ceiling in either unit, so the launch acceptance is unaffected and the gate still clears. But a resource receipt that misstates its observed value by 7× is not usable evidence for any future bound derivation. `tasks.md` is still uncommitted, so the correction is free. **Mandatory in the final commit** (§6.1). |
| F-3 | P3 | Standards / gate receipts | The manifest's `expected_receipt_path` for `wave8-final-compatibility-gate` is `receipts/wave-8-compatibility.json`, which does not exist; the produced `wave-8-compatibility-comparison.json` is a different artifact (production-run ledger comparison, schema `…-comparison-v2`), and the 231/0/0 gate result lives only in tasks.md prose. The same gap exists for every `wave-{1..7}-gate.json`. | Consistent with the precedent already blessed by the lead disposition and the pre-cost audit's S-1 handling ("gate records live in these notes"), so this is residue, not a new defect. I independently replayed the gate and obtained **231 passed / 0 failed / 0 errors / 0 skipped in 44.60 s**, so the record is now externally corroborated. Optional in the final commit: emit `receipts/wave-8-compatibility.json`, or retire the `expected_receipt_path` fields as a manifest revision in a later change. |
| F-4 | P3 | Records / comparison method | `wave-8-compatibility-comparison.json`'s `method_note` states order can be compared "only where both sides carry timestamps" because "run.json phases dict is key-sorted, not execution-ordered". The premise about the `phases` dict is true, but `measurement.phase_order` **is** a runtime-appended execution-order list (`src/artifacts/run_writer.py:389,429,482,536,669`), and on that field the fixture's 7-element `run_state_phase_order` is an **exact prefix** of the production `phase_order` — a strictly stronger result than the receipt's 2-element subsequence claim. | Understatement, not concealment: the restriction hides no drift, and the stronger check passes. This audit supplies the stronger proof. Optional in the final commit: amend the `method_note` to cite `measurement.phase_order` and the exact-prefix result. |
| F-5 | P3 | Intent / 9.3 coverage | Task 9.3 names "run/checkpoint/final/best files" and "collective order" among the exact-compare surfaces; the production comparison receipt compares neither, delegating both to the code-level fixture replay. | The delegation is stated in the receipt's own `method_note`, and the replay is green (231/0/0, independently confirmed). I additionally compared the production `checkpoints/final.json` and `checkpoints/best.json` against the Wave-0 `run_writer` fixtures: key sets identical (`checkpoint_path`,`step` / `checkpoint_path`,`selector`,`step`,`value`), `checkpoint_path` and `step` identical, only the `value` differs and is config-shaped. No action required. |
| F-6 | P3 | Standards / evidence durability | The launch overlay config lives at `/data/.claude/jobs/c9895ff9/tmp/wave8/wave8_vertical_smoke.yaml`, outside the repository, so the sha256-bound file itself is not durable evidence. | Mitigated in place: all three of its lines are quoted verbatim in the packet and its effect is independently confirmed by the produced `run.json` (`artifact_root`, `collision_outcome = "fail"`). No action required. |
| F-7 | P3 | Intent / unmeasured ceiling | The `≤ 100 per-rank collective operations` ceiling was never measured; no production per-rank collective counter exists. | Accepted — see §3(c). Discharged by the green collective-order suite plus 40.2 s completion inside the 900 s hang bound, which is the strongest evidence available at this commit. The note is candid that no counter exists and makes no count claim. A real counter belongs to a later observability change. |

**P0: 0. P1: 0. P2: 2 (F-1, F-2). P3: 5 (F-3, F-4, F-5, F-6, F-7).**

---

## 5. Completion gate

**CLEARED.** There is no unresolved P0 or P1, so the task-9.6 condition is satisfied: the change **may take its final commit and be marked complete/archived**, provided the final commit carries the mandatory items in §6. Both P2s are record-accuracy/disclosure defects in an uncommitted file; neither implies a behavioural, cache, or launch risk, and both are corrected at zero cost before the commit exists.

## 6. What the final commit must still include

1. **(mandatory, F-2)** Correct the CPU-RSS figure in the tasks.md Wave-8 9.2 note: observed per-rank `max_rss_bytes` is **8,963,375,104 B (rank 0) / 8,959,627,264 B (rank 1) ≈ 8.35 GiB**, within the 16 GiB ceiling — not "1.18 GiB".
2. **(mandatory, F-1)** State the pass-rule deviation plainly in the tasks.md 9.1 note: `wave8-full-cpu-matrix` does **not** satisfy the manifest's `pass_rule` (expected failure set ∅ vs. 11 observed) and is accepted as a recorded, fully diagnosed environmental deviation whose compliant remedy under `revision_rule` was deliberately not taken. Keep the existing "zero unexplained failures / zero skips" statement alongside it; do not let it stand alone.
3. **(mandatory)** `git add` the three untracked Wave-8 receipts plus this audit receipt (`receipts/wave-8-final-audit.md`) — they are untracked and would otherwise be omitted.
4. **(mandatory)** Check task 9.6 `[x]` with a close-out note recording: strict validation rc=0; this audit's verdicts and 0 P0 / 0 P1; the independent 231/0/0 gate replay; and the task-9.6 revert obligation — noting that the final commit is **docs + receipts only** (zero source bytes since `88391d6bb`), so its independent code revert is empty by construction and provably touches no cache target or immutable evidence. Record the cache/evidence-untouched proof this audit executed (manifest sha256s `c1d9fea7…` / `8de1fda2…` unchanged after the smoke; `geometry_flip_aug_5step` mtimes `1783394670.996` / `1783394682.933` unchanged since the pre-cost audit).
5. *(optional, F-3/F-4)* Emit `receipts/wave-8-compatibility.json` for the gate's declared `expected_receipt_path`, and amend the comparison receipt's `method_note` to cite `measurement.phase_order` and the exact-prefix result.

Nothing else is required. No source, test, fixture, config, spec, or manifest change is needed or permitted for completion.

---

## 7. Commands executed (all read-only)

| # | Command | Observed |
| --- | --- | --- |
| 1 | `git rev-parse HEAD` | `d598f8894a3e2d0f3a5c407b685b344813a58dd3` (at start and re-checked immediately before this write) |
| 2 | `git status --porcelain` | exactly the five declared paths, at start and immediately before this write |
| 3 | `git diff 88391d6bb..HEAD --stat -- src/ tests/ scripts/ configs/` | **empty** |
| 4 | `git diff --stat` | `docs/COORDEXP_SWIFT.md \| 2 +-`, `tasks.md \| 63 ++++++--`; 2 files, +59 / −6 |
| 5 | `git diff -- docs/COORDEXP_SWIFT.md` | 1 line changed: the "Entry and assembly" owner row |
| 6 | `git show --stat d598f8894` | 6 files, +664 / −9; receipts + tasks.md only; message records "note (f) corrected per audit finding S-1" |
| 7 | `sha256sum` of the five dirty paths | table in the header |
| 8 | `conda run -n ms openspec validate decompose-coordexp-swift-training-orchestration --strict` | `Change … is valid`, **rc = 0** |
| 9 | `pytest <wave8-final-compatibility-gate argv> -q -p no:cacheprovider --junitxml` | `231 passed in 44.60s`, RC=0; JUnit `tests=231 failures=0 errors=0 skipped=0` |
| 10 | `pytest tests/packing/test_wave6_pack_plan_probe.py tests/training/test_orchestration_compatibility.py 'tests/training/test_pipeline_assembly.py::test_runtime_determinism_consensus_binds_launcher_mapping_and_baseline' -q` | `48 passed in 164.05s`, RC=0; JUnit `tests=48 failures=0 errors=0 skipped=0`; **all 11 declared `failing_nodes` present and green (0 missing, 0 non-green)** |
| 11 | `pytest 'tests/qwen/test_patches.py::test_real_qwen3_vl_patch_embed_linearization_matches_cuda_bf16_forward_and_grad' 'tests/packing/test_wave6_pack_plan_probe.py::test_comparison_records_exact_cpu_semantics_and_bounded_metrics' -q` | `1 failed, 1 passed in 5.48s`, RC=1; `Wave6ProbeError: "CUDA was initialized before CPU comparison"`, `code="wave6.gpu_forbidden"` at `wave6_pack_plan_comparison.py:2060` — **pollution mechanism independently reproduced** |
| 12 | JSON read of `wave-7-cache-preparation.json` | `terminal_status=completed`, `failure=null`, both splits `build_status=built`, `status=complete`, `micro_step_count` 2 / 1 |
| 13 | JSON read of `wave-7-cache-verification.json` | `terminal_status=completed`, both splits `build_status=hit`, `verified_splits=[train, eval.forward]`, `verification_level=payloads`, `cache_materialization_authorized=false` |
| 14 | JSON read of `wave-7-determinant-projections.json` | 31 determinants/split; `content_identity_changed` **empty** both splits; owner changes exactly the two declared moves; `owner_source_changed` = the four declared sources |
| 15 | `find .cache/coordexp_swift/packing -maxdepth 3` | exactly two fingerprint dirs (`8f11237f…`, `3b30c157…`) + their two locks; no third fingerprint |
| 16 | `sha256` of both on-disk cache `manifest.json` | `c1d9fea76d063c6b…e025e8` / `8de1fda29784aad2…d92992` — **match** both receipts' `manifest_sha256`, after the smoke consumed them |
| 17 | `os.stat` walk of `.cache/coordexp_swift/geometry_flip_aug_5step` | 2 targets, 2 files each, mtimes `1783394670.996` / `1783394682.933` — identical to the pre-cost audit's record |
| 18 | JSON read of the smoke `run.json` | `status=completed`, `terminal_error=null`, `completed_steps=1`, `resolved_max_steps=1`, `consumed_packs=1`, `final_optimizer_update_status=applied`, `final_finite_status=finite`, `checkpoint_event_count=1`, `collision_outcome=fail`, `warning_counts={}`, `forward_input_provider_resolution.source=strict_config` |
| 19 | fingerprint grep of the smoke `run.json` | both Wave-7 fingerprints appear twice each (`materializations` + `policy_identities.cache`) |
| 20 | phase inspection of `run.json` `measurement` | 14 phases; `cache_preparation`/`cache_publication`/`steady_state` `not_run`; 6 phases carry `started_at=null` exactly as the comparison receipt's method claims; `entry_to_terminal.duration_seconds=40.245` |
| 21 | `measurement.phase_order` vs `tests/fixtures/training_orchestration/phase_order.json` | fixture's 7-element `run_state_phase_order` is an **exact prefix** of the 14-element production `phase_order` → `True` |
| 22 | `resource_high_water` + `per_rank_measurement` from `run.json` / `logging.jsonl` | rank 0 RSS `8,963,375,104 B`, rank 1 `8,959,627,264 B`; GPU alloc `9,356,426,240` / `7,171,412,480` B → **F-2** |
| 23 | `os.walk` byte count of `outputs/smoke/wave8_vertical_smoke_r1` | 45,712,517 B in 11 files; + 5 dirs × 4096 = **45,732,997 B**, exactly the recorded figure |
| 24 | `df -BG /data` | 1,597 G available |
| 25 | production vs fixture `final.json` / `best.json` | key sets identical; `checkpoint_path` and `step` identical; only `best.value` differs (config-shaped) |
| 26 | `test-command-manifest.json` `pass_rule` / `revision_rule` / `launch_bearing_actions` / per-command `expected_red_nodes` + `expected_receipt_path` | `wave8-full-cpu-matrix` `expected_red_nodes=[]`; `wave-8-compatibility.json` and `wave-{1..7}-gate.json` **missing** → **F-1**, **F-3**; both packets conform to `launch_bearing_actions` |
| 27 | receipt self-hash of `wave-8-cpu-matrix.json` | recomputed over the sorted compact JSON minus `receipt_sha256` = `747e14652f11c089…945678` — **matches** the recorded value |
| 28 | `git check-ignore -v outputs/ .cache/` | both ignored (`.gitignore:55`, `.gitignore:2`) — smoke and cache bytes cannot leak into the final commit |
| 29 | `tasks.md` inspection of Wave-6 close-out note (f) | corrected text present at lines 553–563 with the S-1 marker → **S-1 landed** |

All pytest invocations ran with `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and all nine manifest `required_absent` environment selectors verified unset in the invoking shell. JUnit XML and logs were written to the session scratchpad, never into the repository. No `env -u` form was used for the recorded runs (it is silently no-op'd in this sandbox and produced a false RC=0 on two discarded attempts; those attempts are not counted as evidence).

**Post-write expectation:** `git status --porcelain` shows the five declared paths plus one new untracked path — `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-8-final-audit.md`, this receipt. Any other change would invalidate this audit.
