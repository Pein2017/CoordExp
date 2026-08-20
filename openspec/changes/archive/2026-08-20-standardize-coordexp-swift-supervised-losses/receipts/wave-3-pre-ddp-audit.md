# Wave-3 Pre-DDP / Cost Standards + Intent-Contract Audit (task 3.6)

- **Date:** 2026-08-20
- **Identity:** Opus wave-3 pre-DDP auditor, spawned by the Claude Fable lead.
  Distinct from the lead, the Wave-1/2/3 builders, and the Wave-0 entry auditor.
- **Frozen target:** HEAD `a2067307823d679fd5ede2fd691426744c5c4dd8`
  (Wave-3 code commit), tracked tree clean. Verified at audit start and
  re-verified immediately before writing this file: the only untracked paths
  were `receipts/wave-3-two-rank-probe-packet.md` and
  `receipts/wave-3-two-rank-probe-receipt.json`. Expected post-write state is
  those two plus this receipt (three untracked paths).
- **Audited commits:** `57a1c93cb` (Wave 1), `2dde00427` (Wave 2),
  `a20673078` (Wave 3). The user's own `e6f923534`
  (`docs(agents)`, AGENTS.md + AGENTS.override.md only) interleaved between
  Waves 1 and 2 and is disclosed in the Wave-2 close-out note; it touches no
  owner surface of this change.
- **Scope:** read-only. No commits, no GPU, no cache commands. Every command
  ran through `bash -c 'export PYTHONDONTWRITEBYTECODE=1; ...'`; no
  `env`-prefixed invocation was used (the host hook silently no-ops those).
  No heredoc was piped into `conda run`; scratch scripts live under
  `/data/CoordExp/.claude/jobs/c9895ff9/tmp/`.
- **Hangs:** none. No command exceeded 20 minutes; nothing was killed.

## Verdicts

| audit | verdict |
| --- | --- |
| STANDARDS (Waves 1-3 as committed) | **PASS-WITH-DISPOSITIONS** |
| INTENT-CONTRACT (semantic core) | **PASS-WITH-DISPOSITIONS** |
| **Wave-4 entry** | **CLEARED** (0 P0 / 0 P1) |

Counts: **0 P0, 0 P1, 4 P2, 6 P3.**

## Findings

| id | sev | area | description | disposition |
| --- | --- | --- | --- | --- |
| S-1 | P3 | standards / fixtures | The brief's premise "`tests/fixtures/**` is byte-identical to its decompose-era tree object" is **false as literally written**. The subtree hash moved `086a86b8…` (decompose-era, `2a297a93a`) -> `9404e1bb…` (Waves 1-3) at commit `57a1c93cb`. The **entire** diff is one line: `tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml` gained `mode: enabled` beside its already-canonical `weight: 0.1`. The **frozen orchestration** subtree `tests/fixtures/training_orchestration/` is byte-identical at `2fe137ccffc9b51e4eb227a91e2fbe6607dfc528` across `2a297a93a`, `57a1c93cb`, `2dde00427`, `a20673078` and HEAD. | ACCEPT. The one-line change is the deviation the Wave-1 close-out already declares (live loader-input fixture, not a frozen artifact); entry-audit F-5/F-6's "fixtures stay byte-frozen" applies to `training_orchestration/`, which held. The brief's blanket phrasing is superseded by this row. No action. |
| S-2 | P2 | standards / probe receipt | `wave-3-two-rank-probe-receipt.json` carries **no commit binding, no timestamp, no measured wall time, and no cache-root sha inventory before/after**. Its top-level keys are `probe, task, change, backend, device, world_size, cuda_available, cuda_initialized, declared_tolerances, collective_op_count_by_rank, ungated_metric_keys, arms, findings, status`. Three of the packet's seven bounds (wall time <=600 s, cache passes = 0, artifact bytes) are therefore asserted by the packet, not evidenced by the receipt; and the ordering "receipt was produced by the frozen argv at `a20673078`, not the earlier builder dry-run" is only inferable from filesystem mtimes. | ACCEPT WITH REQUIRED FOLLOW-UP. Not behavioral, purely evidentiary; independently closed below (mtime chain + cache scan + byte count). **The Wave-5 packet must require `commit`, `started_at`/`wall_seconds`, and a cache-root sha inventory as receipt schema fields** so this does not recur at the GPU gate. |
| S-3 | P3 | standards / manifest | The `wave3-two-rank-probe` manifest entry still reads `TO-FREEZE-IN-WAVE-3-PACKET`; no `amend-8` records the frozen argv, so the append-only manifest is no longer self-contained about what was executed. | ACCEPT, close in the close-out commit. The manifest entry itself delegates the freeze to the packet, so the packet is compliant; the close-out must nonetheless append `amend-8` with the frozen argv and the observed gate counts. |
| I-1 | P2 | intent / fail-open | `src/training/supervised_trainer.py::_total_loss` reads `getattr(loss_bundle, "backward_loss", None)` and **silently falls back to `total_loss`** when the attribute is absent or non-tensor. A bundle-like object lacking `backward_loss` would backward the UNcompensated semantic total at world size > 1, halving (1/W-ing) every gradient with no exception, no telemetry difference, and no finite-gate signal. | ACCEPT AS P2, NOT P1: there is exactly one `LossBundle` construction site in `src/` (`src/losses/runner.py:478`) and `LossBundle.__post_init__` sets `backward_loss = total_loss` whenever it is `None`, so no wrong path is reachable at HEAD. Repo contract prefers fail-fast over silent fallback on silent-correctness surfaces: recommend requiring the attribute, or requiring `plan.backend_gradient_scale == 1.0` when it is absent. Wave-4/5 owner. |
| I-2 | P2 | intent / distributed hazard | `LossRunner.prepare_planned_step` builds the **local** denominators (which raise `loss.segment_balanced_zero_eligible` on zero eligible segments) *before* `_resolve_streaming_denominators` performs the cross-rank gather. At W>1 a rank whose shard has no eligible segment for a token-type-filtered term (`token_type_gate` is exactly such a term) raises locally while its peers enter the gather collective -> desync/hang instead of a clean all-rank failure. | PRE-EXISTING, NOT P1: introduced `e1662c2c7` (2026-07-01) / `3cd40f5f0` (2026-07-05), present unchanged at `2ee6c4959`, untouched by Wave 3. The wave-3 probe uses unequal but **non-zero** per-rank counts, so it does not cover this. Record as a known distributed hazard; the Wave-5 production-shaped smoke must either cover it or explicitly declare it out of the tested envelope. |
| I-3 | P2 | intent / gate coverage | `src/runtime/finite_gates.py` derives `total_loss_finite` from `bundle.total_loss`, but the tensor actually handed to `backward()` is `backward_loss = W x total_loss`. An overflow created by the x`W` factor is not covered by the pre-backward all-rank scalar gate. | ACCEPT AS P2 (bounded, not P1): W <= 8 and fp32 objective magnitudes are O(1), so the finite/non-finite classification of `total_loss` and `W x total_loss` cannot diverge in the supported envelope. Term-level labels are already raw-keyed (correct per entry-audit F-2). Optional hardening for Wave 5.5's Non-Finite-Gates delta. |
| I-4 | P3 | intent / duplication | The replicated-eval predicate `expected_split == "eval" and expected_reduction_mode is None` is written **twice** in `_reduce_metric_reports` (inline at the accuracy call site, and as the new local `replicated_objective`) rather than bound once and shared. The literal `"eval"` is also a hardcoded copy of `src.eval.forward.EVAL_FORWARD_SPLIT` — which is *not* `src.training.cache_workflow.EVAL_SPLIT == "eval.forward"`, a genuinely confusable pair. | ACCEPT. Verified textually and semantically identical at both sites today (both read the same two locals; identical truth table over train/eval x None/`disjoint_shard`). `train_runtime` cannot import `eval.forward` (that direction already exists), so a shared local binding plus a comment naming the confusable constant is the cheap fix. |
| I-5 | P3 | intent / artifact keys | `LossBundle.to_artifact_dict()` now emits `backward_loss`, and `LossTermResult.to_artifact_dict()` emits `backward_contribution` + `backend_gradient_scale`; `_merge_term_artifacts` and `finalize_planned_step` propagate them. These are **not** row fields today (see the leak proof below), but Wave 4 owns the row schema and must decide explicitly whether they stay internal. | ACCEPT, hand to Wave 4.1/4.2 as an explicit row-schema decision (recommend: stay out of rows; they are backend detail, not semantics). |
| I-6 | P3 | intent / soft alias | `_merge_term_artifacts` reads `item.get("backward_contribution", item["weighted_loss"])` — a silent cross-version fallback that would mis-total a mixed-version artifact stream. | ACCEPT. Same class as I-1, lower reach (finalization arithmetic only, never backward). Fold into the I-1 fix. |
| I-7 | P3 | intent / telemetry semantics | Reduced train/eval telemetry is numerically **unchanged** by Wave 3 (old: raw x W then mean-over-ranks; new: raw then sum-over-ranks — algebraically identical), but `per_rank_metrics` changes visibly: each rank now reports its true semantic partial instead of a W-scaled value. | ACCEPT as an intended, user-visible correction. The Wave-3 close-out note must state it so nobody reads historical per-rank rows against new ones. |

No P0 or P1 finding was raised. Each P2 above is explicitly argued down from P1
in its disposition column: I-1 has no reachable wrong path at HEAD; I-2 predates
the change and is untouched by it; I-3 cannot diverge inside the supported
envelope; S-2 is evidentiary, not behavioral, and is independently closed here.

## 1. STANDARDS — re-derived, not trusted

### 1.1 Gate argv replay (independent)

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms pytest \
  tests/losses tests/runtime tests/eval/test_forward_eval.py \
  tests/training/test_supervised_trainer.py -q'
-> 366 passed, 1 warning in 54.58s          (exit 0)
```

**366 / 0 / 0 — matches the expected 366/0/0.**

### 1.2 Frozen-fixture / collective argv replay (independent)

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms pytest \
  tests/training/test_orchestration_compatibility.py \
  tests/runtime/test_rank_report_collective.py -q'
-> 15 passed in 32.18s                      (exit 0)
```

**15 / 15 — matches expectation.** Fixture byte-identity, by git tree object
rather than by re-running the tests:

| commit | `tests/fixtures` | `tests/fixtures/training_orchestration` |
| --- | --- | --- |
| `2a297a93a` (decompose-era / wave-0 close) | `086a86b8…` | `2fe137cc…` |
| `57a1c93cb` (Wave 1) | `9404e1bb…` | `2fe137cc…` |
| `2dde00427` (Wave 2) | `9404e1bb…` | `2fe137cc…` |
| `a20673078` (Wave 3) / HEAD / worktree | `9404e1bb…` | `2fe137cc…` |

`git diff 2a297a93a HEAD -- tests/fixtures/` = **1 file, 1 insertion**
(`smoke/qwen3_vl_single_image_pack/config.yaml`, `+ mode: enabled`), attributed
to `57a1c93cb`. The frozen orchestration subtree never moved. See S-1.

### 1.3 Probe receipt coherence

Observed in `wave-3-two-rank-probe-receipt.json`:

| field | observed | packet bound | verdict |
| --- | --- | --- | --- |
| `status` | `OK` | exit 0 required | PASS |
| `findings` | `[]` (and `[]` in all three arms) | zero required | PASS |
| `collective_op_count_by_rank` | `{"0": 12, "1": 12}` | 12 per rank | PASS |
| `declared_tolerances` | `rtol 1e-05 / atol 1e-06` | rtol 1e-5 / atol 1e-6 | PASS |
| `world_size` / `backend` / `device` | `2` / `gloo` / `cpu` | exactly 2, CPU gloo | PASS |
| `cuda_initialized` | **`false`** (`cuda_available: true`) | peak GPU memory 0 | PASS — cross-checked as required |
| arms | `base_ce_only`, `enabled_gate`, `coord_auxiliary`, each with `distributed_rank0/1` + `world_size_one_reference` | 1 planned step x 3 arms | PASS |
| artifact bytes | 28,036 B | <= 1 MiB | PASS |
| cache / materialization passes | `find .cache/coordexp_swift/packing -newermt "2026-08-20 07:00"` -> **0 entries** | required 0 | PASS (see below) |
| wall time | not recorded in the receipt | <= 600 s | UNEVIDENCED — S-2 |

The cache scan covers the whole change window (07:00 precedes Wave 1's
`57a1c93cb` at 07:33), so **no path under the resolved production cache root
`<repo>/.cache/coordexp_swift/packing` was written by any Wave-1/2/3 command**,
not merely by the probe. (The root resolves there because
`COORDEXP_SWIFT_PACK_CACHE_ROOT` is required-absent; `src/training/cache_workflow.py:160`.)

Provenance chain (filesystem mtimes — inference, not receipt content, per S-2):
commit `a20673078` 09:38:33 < packet 09:39:04 < receipt 09:39:42. The receipt
therefore post-dates both the frozen commit and the packet freeze, i.e. it is
the frozen-argv run and not the earlier builder dry-run the packet labels
informational.

Cross-checked separately: both the probe
(`scripts/probes/coordexp_swift/losses_wave3_two_rank_probe.py:622`) and the
in-suite two-rank tests use a **real `torch.nn.parallel.DistributedDataParallel`**
module, not a hand-rolled all-reduce — so the mean-gradient compensation under
test is the production mechanism.

### 1.4 Determinant invariant — recomputed at HEAD

Three independent legs, all green.

**(a) Fingerprint + payload recompute** (scratch script re-running HEAD's
`_build_determinant_entries` / `_registry_entries_fingerprint` /
`_registry_code_identity` / `_validate_determinant_registry` over the frozen
semantic payloads):

```
[train]        recomputed 8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f
               baseline   8f11237f…  fingerprint_equal=True  full_payload_equal=True
[eval.forward] recomputed 3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662
               baseline   3b30c157…  fingerprint_equal=True  full_payload_equal=True
entries_equal=True  code_identity_equal=True  registry_version_equal=True  (both splits)
```

Both equal the manifest's pinned `cache_invariant.train_fingerprint` /
`eval_fingerprint`. The receipt's own `baseline_sha256` was also reproduced
exactly — `sha256(canonical-compact-sorted JSON of the document minus
`baseline_sha256`) = b2d9595f07a0a511abf436e24f47b21c5694760c4d308bed4845fd5712931059`
— so the baseline file is internally self-consistent, not just self-declared.

**(b) Owner-file re-hash:** 31 determinants / 24 owner files (enumerated live
from `PACKING_CACHE_DETERMINANT_OWNERS`), sha256 of every live file vs the same
path at wave-0 base `0cb0ae729`: **`drift_vs_wave0_base: NONE`**. The
DO-NOT-EDIT stop rule held across all three waves.

**(c) Config-content check:** the baseline config
(`configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`)
differs from `0cb0ae729` by exactly one added line,
`losses.protected.token_type_gate.mode: zero_weight_ablation`. No determinant
category in `build_packing_cache_determinants` reads `losses.*` (only
`realized_vocab_groups` -> `src/losses/vocab.py`, which is byte-unchanged),
consistent with the Wave-0 finding.

### 1.5 Manifest append-only + wave close-out fidelity

- `command-manifest.json` blob: `b178037e…` (`2a297a93a`) -> `a68eed73…`
  (`57a1c93cb`) -> `b416e1b2…` (`2dde00427`) -> **unchanged** at `a20673078`.
- Deleted non-header lines in each amending diff: `2a297a93a -> 57a1c93cb` = **0**;
  `57a1c93cb -> 2dde00427` = **0**. Amendments 1-7 are strictly appended;
  no original entry was edited or removed. **Append-only: PASS.** (Gap S-3:
  no amend-8 for the wave-3 argv.)
- Wave 0/1/2 close-out notes vs the commits, spot-checked on load-bearing
  claims: Wave 1's "25 supported configs migrated" -> `57a1c93cb` touches
  exactly **25** `.yaml` files under `configs/` (the 26th yaml is the declared
  fixture deviation, correctly listed separately); Wave 2's "finite derivation
  moved to RAW ... including `src/runtime/finite_gates.py:45`" -> the diff shows
  precisely `term_finite[term.name] = _tensor_is_finite(term.raw_loss)`
  replacing the `weighted_loss` keying; Wave 2's "Trainer and
  `src/eval/forward.py` byte-unchanged (F-4 respected)" -> neither file appears
  in `2dde00427`'s stat. Wave 2 also discloses the interleaved `e6f923534`.
  **Close-out fidelity: PASS.**
- `conda run -n ms openspec validate standardize-coordexp-swift-supervised-losses --strict`
  -> `Change 'standardize-coordexp-swift-supervised-losses' is valid`.

## 2. INTENT-CONTRACT — the semantic core

### (a) Is the compensation applied exactly once, only to the differentiable local contribution?

**YES.** The single multiplication site is
`src/losses/runner.py::_compute_token_term_contribution`:
`backward_contribution = weighted if scale == 1.0 else weighted * scale`, with
`raw = segment_balanced_contribution(...)` no longer scaled (the deleted
`raw = raw * float(backend_gradient_scale)` line is the whole defect being
fixed). A `detached_diagnostic` term gets `weighted = raw.detach().new_zeros(())`
and `backward_contribution = weighted`, i.e. never compensated and never in the
objective. `compute_micro_step` filters `objective_terms` before building either
total, so a detached diagnostic contributes to neither.

Leak hunt — every telemetry surface traced to a semantic source:

| surface | reads | compensated? |
| --- | --- | --- |
| train JSONL rows | `src/training/reporting.py:251` -> `loss_bundle["metrics"]` only | NO — `metrics` contains no `backward_*` key |
| eval scalar | `src/eval/forward.py:600` -> `loss_artifact["total_loss"]` | NO |
| micro-step metrics | `_build_metrics(total_loss=total_loss, ...)` (runner.py:450) | NO |
| finite report | `finite_gates.py` -> `bundle.total_loss` + per-term `raw_loss` | NO |
| finalize metrics | `metrics["loss/total"] = total_loss` from `sum(weighted_loss)` | NO |
| backward | `supervised_trainer.py:346` -> `_total_loss(bundle)` -> `bundle.backward_loss` | YES, once |

The only `backward_*` exposure is inside `to_artifact_dict()` payloads, which
flow micro -> `_merge_term_artifacts` -> `finalize_planned_step` and into
`CompletedStepObservation.to_artifact_dict()`. That observation dict is
**never persisted**: `src/training/session.py:2385` reads only
`optimizer_update_status` / `finite_status` off `result.latest_observation`.
So no artifact row carries a compensated number (see I-5 for the Wave-4
decision this leaves open).

Sharded-eval path: `src/eval/forward.py::_run_streaming_forward_only` is
forward-only and never touches `backward_loss`; its scalars come from
`total_loss` and from `terms[*]` token-weighted diagnostics. **No path was
found where `backward_loss` reaches telemetry or `total_loss` reaches backward
at world size > 1** — with the single fail-open caveat recorded as I-1.

### (b) Is the reducer's new `loss/`-keyed sum branch correct?

**YES.** `_is_planned_step_objective_metric_key` = starts with `loss/` and does
not end with `/token_weighted_diag`, `/token_weighted_diag/__weight__`, or
`/segment_count`. Enumerating every `loss/`-prefixed key that actually reaches
`gather_metrics` (from `finalize_planned_step` plus `_prepare_disjoint_shard_scalars`):

| key | branch taken | correct because |
| --- | --- | --- |
| `loss/total`, `loss/<term>` | **new sum** (train + sharded eval) | rank-local partial numerators over the *globally merged* denominator; the global objective is their sum |
| `loss/<term>/segment_count` | excluded -> eval-identical / train mean | already **global** (merged denominator, identical on every rank); summing would multiply by W. The exclusion is load-bearing. |
| `loss/<term>/token_weighted_diag(/__weight__)` | excluded -> sharded-eval pre-weighted sum / train mean | rank-local diagnostics with their own reducers |

Replicated eval is correctly excluded: `prepare_planned_step` is called there
**without** `world_size` (`src/eval/forward.py:352`), so `backend_gradient_scale
= 1.0`, the denominator scope is the full local window, and every rank already
holds the identical global value — mean preserves it, sum would multiply by W.

Numeric continuity, which is why the frozen orchestration fixtures still pass
byte-unchanged: old = `mean_r(W x local_r)` = `sum_r local_r`; new =
`sum_r local_r`. Identical. The visible delta is confined to `per_rank_metrics`
(I-7).

**Predicate equivalence with the accuracy reduction: confirmed.** Both sites
evaluate `expected_split == "eval" and expected_reduction_mode is None` over the
same two locals; the truth table over {train, eval} x {None, `disjoint_shard`}
is identical. A near-miss was chased to ground: `src/training/cache_workflow.py`
defines `EVAL_SPLIT = "eval.forward"`, but the metric-gather split is
`src/eval/forward.py:27 EVAL_FORWARD_SPLIT = "eval"` (passed at line 310), so
the literal `"eval"` is the right string and replicated eval is **not**
mis-summed. That confusable pair is recorded as I-4.

### (c) Does `_total_loss` preferring `backward_loss` change world-size-1 behavior?

**NO behavior change at world size 1.** At `scale == 1.0` every term sets
`backward_contribution = weighted` (the *same tensor object*), so
`compute_micro_step`'s `all(term.backward_contribution is term.weighted_loss)`
short-circuit makes `backward_loss` **the identical node** as `total_loss` — not
merely an equal value. The autograd graph is bit-identical to the
pre-separation implementation, and `_total_loss` returns that same node.

**Hidden compatibility alias: yes, one — and it is fail-open.** See I-1
(`getattr(..., None)` + silent fallback) and I-6 (the analogous
`item.get("backward_contribution", item["weighted_loss"])` in
`_merge_term_artifacts`). Neither is reachable with a wrong value at HEAD;
both should become fail-closed.

### (d) Is the gate-ablation bitwise parity claim (`torch.equal`) justified?

**YES.** The ablated gate (i) runs its whole denominator + fp32 per-atom math
inside `torch.no_grad()` (runner.py:399-402), (ii) yields a literal
`raw.detach().new_zeros(())` weighted value rather than `raw * 0.0`, and
(iii) is filtered out of `objective_terms` before either total is built. So no
gate operation appears in the objective graph at all, and the ablation arm's
objective is the *same sequence of floating-point operations* as the base-only
arm — which is what `torch.equal` on post-update parameters requires. The test
`test_two_rank_gate_ablation_preserves_the_base_ce_only_step_exactly` asserts
element-wise `torch.equal` on every parameter after one real optimizer update
across two real DDP ranks, plus `optimizer_update_status == "ready_to_step"` on
both arms — and it is green in the 366-node gate.

**Non-finite injection genuinely converges one all-rank unsafe decision BEFORE
backward.** `_non_finite_gate_worker` poisons `TokenTypeGateLoss.per_atom_loss`
on **rank 1 only** and then asserts, *on both ranks*: `all_ranks_safe is False`,
`should_call_backward is False`, `should_call_optimizer_step is False`,
`finite_status == "non_finite"`, `optimizer_update_status ==
"skipped_non_finite_scalar"`, `ranks == [0, 1]`, `reason_codes ==
["rank1:non_finite_scalar"]`, `all(gradient is None ...)`, and
`"post_decision" not in result`. Rank 0 — whose own shard is entirely finite —
reaches the same verdict, and the absence of any gradient plus the absence of a
post-backward decision is what proves the decision precedes backward. It also
pins `weighted_loss == 0.0` for the unsafe term, i.e. only RAW-keyed finite
derivation could have seen it: entry-audit **F-2 is discharged in the
distributed setting**, not merely single-process.

**Audit-owned sensitivity evidence** (this is the recorded proof the parity
suite would fail under an uncompensated implementation, discharging the
green-only-test concern for this wave). Script:
`/data/CoordExp/.claude/jobs/c9895ff9/tmp/ddp_mean_check.py` — a repo-free,
CPU/gloo, two-rank real-DDP check of the mechanism the whole wave rests on:

```
rank 0 {'compensated_equals_ws1_reference': True,
        'uncompensated_equals_ws1_reference': False,
        'uncompensated_over_reference': [0.5, 0.5, 0.5],
        'compensated': [3.0, 6.0, 9.0], 'reference': [3.0, 6.0, 9.0]}
rank 1 { ... identical ... }
MECHANISM_CONFIRMED True                                  (exit 0)
```

DDP reduces gradients as a **mean** over ranks; an uncompensated local partial
yields exactly `1/W` of the world-size-one gradient; multiplying by `W` exactly
once restores parity. This is independent of the repo's own tests and confirms
both that the fix is necessary and that its magnitude is exactly right.
`tests/losses/test_wave3_semantic_vs_backward.py::test_backend_compensation_is_applied_exactly_once_never_twice_or_zero`
independently pins the ratio algebra with explicit anti-double (`W**2`) and
anti-zero (`1.0`) assertions.

### (e) Are the two handed-back defects genuinely pre-existing?

**YES, both — and excluding them from parity gating is honest.** The probe's
`ungated_metric_keys` is exactly `["count/examples", "count/packs",
"*/token_weighted_diag"]`.

- `count/examples`, `count/packs`: in the **train** reducer these fall to the
  trailing `sum(...) / self.world_size` mean branch, reporting a per-rank
  average instead of the global total. That branch and
  `_EVAL_SUM_METRIC_KEY_NAMES` (which rescues them only in *sharded eval*) were
  introduced by `2b0a2165a` (2026-08-05), well before this change's base
  `0cb0ae729`, and are present verbatim at `2ee6c4959`
  (`git show 2ee6c4959:src/runtime/train_runtime.py` -> line 53
  `_EVAL_SUM_METRIC_KEY_NAMES`, line 727 `/ self.world_size`).
- `*/token_weighted_diag`: same train mean branch — an unweighted mean of
  rank-local per-atom means, which is not the count-weighted global mean.
  Same provenance; Wave 3 changed neither.

Honesty check: these are genuinely rank-local shard quantities (the counts) or a
known-wrong reducer, so **no** world-size-2 vs world-size-1 comparison could
match them regardless of Wave 3's correctness — gating on them would produce a
false failure, not a real signal. The receipt names them explicitly rather than
silently widening tolerances, and the parity gate still covers
`count/eligible_segments`, `count/skipped_segments`, `count/supervised_atoms`,
`loss/*`, `finite/*` and `acc_*` at delta `0.0`. **Honest.** Required: the
close-out must restate them as *open pre-existing defects with provenance and an
owner*, not leave them buried in a probe field.

### (f) Double-scaling / missed-scaling corners

- **Gradient accumulation > 1: safe, and structurally so.**
  `validate_accelerator_runtime` hard-fails with
  `runtime.accelerator_accumulation_non_neutral` unless
  `accelerator.gradient_accumulation_steps == 1` ("CoordExp owns
  accumulation"), so Accelerate's own `loss / grad_accum` division can never
  apply. Accumulation is instead expressed through the **global planned-step
  denominator**, with one `accelerator.backward()` per micro-step. Because DDP
  averaging is linear, the accumulated gradient is
  `sum_micro (1/W) sum_r W * g = sum_r sum_micro g` — the exact world-size-one
  gradient. No double scaling, no missed scaling.
- **A rank with zero eligible segments: hazard, pre-existing — I-2.** The local
  zero-eligible raise fires before the gather collective, so at W>1 it desyncs
  rather than failing all-rank cleanly. Not introduced here; not covered by the
  probe; must be covered or explicitly excluded by the Wave-5 smoke.
- **Single-segment planned steps: safe.** The denominator is `>= 1` by
  construction (zero is rejected), the compensation is a scalar multiply
  independent of segment count, and the semantic/backward separation is
  per-term, not per-segment.

## 3. Wave-4 entry

**CLEARED.** Zero P0 and zero P1 findings across both audits. The four P2s are
one evidentiary gap (S-2), one unreachable-at-HEAD fail-open (I-1), one
pre-existing distributed hazard untouched by this change (I-2), and one bounded
gate-coverage gap (I-3); none blocks Wave-4's telemetry work, and none is a
silent-correctness defect reachable by the supported configurations at HEAD.

Note on state, not a finding: tasks 3.1-3.6 are unchecked and no Wave-3
close-out note exists at `a20673078`. That is the expected pre-close-out state —
this audit is task 3.6's final input, and the close-out commit is what discharges it.

### The Wave-3 close-out commit must include

1. Tick tasks **3.1-3.6** and add a Wave-3 close-out note in `tasks.md`, bound
   to commit `a20673078`, recording: the gate replay **366/0/0**, the frozen-fixture
   /collective replay **15/15**, the two-rank probe result (status OK, 0 findings,
   12 collective ops/rank, rtol 1e-5/atol 1e-6, world size 2, 0 GPU, 0 cache
   passes), and the determinant re-verification (`8f11237f…` / `3b30c157…`,
   24 owner files zero drift).
2. Commit the three currently-untracked receipts: the probe **packet**, the
   probe **receipt JSON**, and **this audit receipt**.
3. Append **`amend-8`** to `receipts/command-manifest.json` (append-only, never
   editing `wave3-two-rank-probe`) pinning the frozen probe argv — retiring the
   `TO-FREEZE-IN-WAVE-3-PACKET` placeholder — and the observed Wave-3 gate
   counts, matching the amend-6/amend-7 pattern. (S-3)
4. Restate the three `ungated_metric_keys` as **open pre-existing defects** with
   provenance (`2b0a2165a`, 2026-08-05) and a named owner, plus the
   zero-eligible-segment desync hazard (`e1662c2c7` / `3cd40f5f0`, 2026-07) —
   I-2 — as a Wave-5 smoke-envelope decision.
5. Disclose the **per-rank telemetry semantics change** (I-7): reduced values
   are unchanged, per-rank values are no longer world-size-scaled.
6. Cite the audit-owned **sensitivity evidence** for the wave-3 parity suite
   (`MECHANISM_CONFIRMED True`, section (d)) so the load-bearing parity tests
   are not green-only evidence.
7. Carry forward as Wave-4/5 obligations: fail-close the `_total_loss` /
   `_merge_term_artifacts` fallbacks (I-1, I-6); decide explicitly whether
   `backward_loss` / `backward_contribution` / `backend_gradient_scale` appear
   in the new row schema (I-5, recommend: no); bind the replicated-eval
   predicate once instead of twice (I-4).
8. **Require `commit`, `started_at`/`wall_seconds`, and a cache-root sha
   inventory (before/after) as receipt-schema fields in the Wave-5 packet**, so
   the S-2 evidentiary gap does not recur at the GPU gate.

## Appendix — commands executed (all read-only)

| # | command | observed |
| --- | --- | --- |
| 1 | `git rev-parse HEAD && git status --porcelain` (start and again before writing) | `a2067307823d679fd5ede2fd691426744c5c4dd8`; 2 untracked paths, both expected |
| 2 | `bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms pytest tests/losses tests/runtime tests/eval/test_forward_eval.py tests/training/test_supervised_trainer.py -q'` | `366 passed, 1 warning in 54.58s` (exit 0) |
| 3 | `bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms pytest tests/training/test_orchestration_compatibility.py tests/runtime/test_rank_report_collective.py -q'` | `15 passed in 32.18s` (exit 0) |
| 4 | `bash -c 'export PYTHONDONTWRITEBYTECODE=1; cd …; conda run -n ms openspec validate standardize-coordexp-swift-supervised-losses --strict'` | `Change '…' is valid` |
| 5 | `git rev-parse <commit>:tests/fixtures` / `:tests/fixtures/training_orchestration` over 10 commits; `git diff 2a297a93a HEAD -- tests/fixtures/` | see 1.2; 1 file / 1 insertion |
| 6 | `.../tmp/owner_check.py` (live `PACKING_CACHE_DETERMINANT_OWNERS` vs `git show 0cb0ae729:<path>`) | `determinant_count 31`, `owner_file_count 24`, `drift_vs_wave0_base: NONE` |
| 7 | `.../tmp/determinant_check.py` (HEAD registry code over frozen payloads) | train `8f11237f…`, eval `3b30c157…`, `full_payload_equal=True` both splits |
| 8 | `.../tmp/sha_variants.py` | `MATCH: doc_minus_sha \| sep=(',',':') \| indent=None` -> `b2d9595f…` |
| 9 | manifest blob hashes across 5 commits + `git diff … \| grep -c '^-[^-]'` | 0 deletions in each amending diff; unchanged at `a20673078` |
| 10 | `find .cache/coordexp_swift/packing -newermt "2026-08-20 07:00"` | 0 entries |
| 11 | `ls -la --time-style=full-iso …/wave-3-two-rank-probe-*` | packet 09:39:04 (2,167 B); receipt 09:39:42 (28,036 B) |
| 12 | `.../tmp/ddp_mean_check.py` (two-rank gloo, real DDP, repo-free) | `MECHANISM_CONFIRMED True` (exit 0) |
| 13 | `git show --stat` / `-S` blame on `57a1c93cb`, `2dde00427`, `a20673078`, `e6f923534`, `2b0a2165a`, `2ee6c4959` | see 1.5 and 2(e) |

All Python ran under `conda run -n ms` with `PYTHONDONTWRITEBYTECODE=1`; scratch
scripts were written to files, never piped as heredocs.
