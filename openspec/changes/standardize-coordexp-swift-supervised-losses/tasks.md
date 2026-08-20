## 0. Wave 0 - Pinned predecessors, owner graph, and cache invariant

- [x] 0.1 Verify `reconcile-coordexp-swift-training-contracts` and `decompose-coordexp-swift-training-orchestration` are implementation-complete, synced, and archived; record their exact commits and the current change base commit, and stop if code, stable specs, docs, or archive dispositions disagree.
- [x] 0.2 Record the post-decomposition owner/import graph and prove loss config, loss implementations, reporting consumers, and supported config migrations are outside `PACKING_CACHE_DETERMINANT_OWNERS`; record the exact admitted train/eval determinant payloads, aggregate hashes, target paths, and cache-hit receipts from the predecessor.
- [x] 0.3 Freeze an exact execution-command manifest covering cwd, `conda` environment, commands, configs, world size/devices, artifact roots, expected evidence, and quantitative limits. Require a reviewable append-only amendment for any later command change.
- [x] 0.4 Run the untouched focused config/loss/runtime/reporting/cache-identity/artifact suites and strict OpenSpec validation, then obtain the entry standards and intent-contract audits; do not begin Wave 1 with an unresolved P0/P1.

> **Wave 0 closed (2026-08-20, base commit `0cb0ae729`):** predecessors
> pinned (reconcile `eb2dc97ab`; decompose `ebfa78ff1`/`68191f7ea`, both
> archived; `openspec validate --all` 21/21). Cache invariant established:
> 31 determinants / 24 owner files; the ONLY loss-package owner is
> `realized_vocab_groups` -> `src/losses/vocab.py`; NO determinant field
> reads `losses.*`; recomputed fingerprints equal the two published targets;
> full payloads sha-frozen in `receipts/wave-0-determinant-baseline.json`
> (task 5.3 must recompute with that exact config, entry-audit F-10).
> **DO-NOT-EDIT stop rule:** any edit to one of the 24 owner files is a
> blocking contract review, never a rebuild. Entry baseline
> (`wave0-entry-baseline` argv) 817 passed / 0 failed / 0 skipped;
> independently replayed by the entry audit at 817/0/0. Entry audit
> (`receipts/wave-0-entry-audit.md`): STANDARDS and INTENT-CONTRACT both
> PASS-WITH-DISPOSITIONS, 0 P0 / 0 P1 (4 P2, 6 P3), Wave 1 CLEARED.
> Binding dispositions: F-1 - `src/config/models.py` must NOT top-level
> import from `src.losses` (circular, reproduced); Wave 1.2 uses a deferred
> in-validator import of `V1_TOKEN_TYPES` plus an equality test, and never
> edits `vocab.py`. F-2 - the gate-ablation finite label must derive from
> the RAW diagnostic (runner `_build_finite_status` currently keys on
> weighted); task 3.4's non-finite injection is the covering RED and must
> be observed failing against a weighted-keyed implementation once.
> F-4 - `_default_qwen_forward` stays OUT of scope (observability change
> owns it); `_default_loss_context`/`_runtime_loss_denominator_gatherer`
> are legitimately in Wave-2/3 scope. F-5/F-6 correction: the frozen
> orchestration fixtures carry only `loss/total` (preserved), so **no
> compatibility node needs a declared row-schema flip and fixtures stay
> byte-frozen**; the actual Wave-4 per-term consumer surface is exactly
> four current-root files (tests/losses/test_runner.py,
> tests/runtime/test_train_runtime.py,
> tests/training/test_wave7_exact_resume_compare.py synthetic rows,
> scripts/analysis/coordexp_swift_length_isolation.py); the wave-0-baseline
> receipt's ~20-file claim is superseded, receipt not rewritten (F-8: the
> 2026-08-20 handoff's "protected compatibility surface" line about
> per-term keys is likewise superseded by this change's artifacts delta).
> **F-3, user-visible research-meaning scope of Wave 1.3** (21 supported
> configs = 5 prod + 16 smoke): enabled gate weight `0.2` in 3 prod + 2
> smoke and `0.25` in 1 smoke ALL become `0.1`; gate `0.0` in 2 prod + 13
> smoke becomes the named `zero_weight_ablation`; `coord_gaussian_rps`
> moves from protected to auxiliary in 2 configs (1 prod, 1 smoke). The
> `0.1` constant's authority is this change's approved proposal/design/spec
> deltas. Command-manifest amendments 1-4 recorded (entry-baseline pin,
> validate-all argv, frozen residue argvs, and the 2026-08-20 standing user
> GPU grant replacing per-action authorization requests for waves 3/5 GPU
> probes - packets and bounds still required). F-7 noted for Wave 5.5
> (optional Non-Finite-Gates delta).

## 1. Wave 1 - Strict Supervised Config Contract

- [x] 1.1 Add config-first failing tests for base CE omission/reweighting, the exact gate group tuple, `enabled: 0.1`, `zero_weight_ablation: 0`, every incompatible mode/weight mutation, legacy protected coordinate placement, typed auxiliary placement, optional auxiliary zero, unknown loss names, and import/callable-style hooks.
- [x] 1.2 Implement the strict protected and auxiliary config models, including the discriminated gate mode, exact constants and groups, and migration-oriented validation errors without a legacy alias.
- [x] 1.3 Enumerate the catalog-supported/current CoordExp-Swift production, smoke, and measurement configs; migrate enabled gates to `0.1`, mark zero gates as `zero_weight_ablation`, move coordinate Gaussian/RPS under the typed auxiliary surface, and inspect every config diff without modifying historical/archive roots.
- [x] 1.4 Add an inventory test that resolves every current supported training config and separately proves representative historical configs remain provenance rather than accepted current inputs.
- [x] 1.5 Gate Wave 1 with focused config tests, a dry config-resolution probe over the full supported inventory, strict OpenSpec validation, and searches for legacy protected-coordinate placement/dynamic hooks in current roots; do not begin Wave 2 with an unresolved test or validation failure.

> **Wave 1 closed (2026-08-20, opus builder + one bundled correction round +
> lead module-level disposition):** strict contract implemented in
> `src/config/models.py` (BaseCELossConfig weight==1.0; gate `mode`
> enabled<->0.1 / zero_weight_ablation<->0.0 with migration-oriented errors;
> canonical group tuple validated via deferred in-validator import of
> `V1_TOKEN_TYPES` per entry-audit F-1, plus an equality test;
> `losses.auxiliary.coord_gaussian_rps` typed surface; protected placement
> rejected by name). `src/losses/runner.py` config-read seam only. RED first:
> 51 failed / 30 passed observed before implementation. Migration: 25
> supported configs (6 enabled gates 0.2/0.25 -> 0.1 [3 prod + 3 smoke,
> exactly F-3's scope]; 19 -> zero_weight_ablation; 2 coord blocks ->
> auxiliary); `infer/` untouched; declared deviation: the live loader-input
> fixture `tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`
> gained `mode: enabled` beside its existing canonical 0.1 (the frozen
> `tests/fixtures/training_orchestration/` set is byte-untouched). New
> inventory test + model-free probe
> (`scripts/probes/coordexp_swift/losses_config_inventory_probe.py`): 25
> resolved strictly, 1 historical rejected. **Blocking finding + disposition
> (manifest amend-5):** four frozen config-identity pin families
> (parity FROZEN_V3 fingerprint, wave3 zero-weight, wave7 bundle byte pin,
> reconcile packet-executor Attempt-6 byte pin) authenticate completed GPU
> evidence against the two migrated configs; constants NEVER re-pinned; 19
> nodes flipped to typed fail-closed refusal assertions; the executor
> mechanism suite (126 nodes, already path-bound to this worktree) is
> conditionally historicized behind a live-byte-drift skipif with the live
> refusal proof in a new 4-node module; 6 now-vacuous parity mutation params
> skipped with reason. Entry-audit gap recorded: config-IDENTITY consumers
> were missed by F-3/F-5/F-6. Gate (lead independent replay): config gate
> 436/0/0; pin-family bundle 416 passed / 132 skipped / 0 failed;
> entry-baseline argv 898/0; trainer+assembly 125/0; residue searches clean;
> `openspec validate --strict` valid. **Cache invariant receipts:**
> recomputed determinant fingerprints and the canonical payload
> baseline_sha256 are byte-identical to Wave 0 post-migration; the 24
> owner files show zero diff.

## 2. Wave 2 - Closed Loss Composition And Zero Policies

- [x] 2.1 Add interface-level failing tests for the closed binding inventory and its exact protected/auxiliary role, `segment_balanced` normalizer, and `forbid`/`detached_diagnostic`/`omit` zero policies.
- [x] 2.2 Introduce the private frozen `TokenLossBinding` metadata and refactor loss construction to one explicit closed composition path without registry discovery, import-by-name, callable config, or pass-through factories.
- [x] 2.3 Implement base-CE forbidden-zero behavior, the gate-ablation no-grad diagnostic path, and complete optional-auxiliary omission from construction, denominators, calls, bundles, finite checks, and metrics; delete the superseded ad hoc coordinate/protected branching.
- [x] 2.4 Add correctness probes using construction/call sentinels and autograd saved-tensor or graph inspection to prove an omitted auxiliary creates no term work or retained graph and a gate ablation creates no objective autograd edge; record explicitly that this is not an efficiency or peak-memory claim.
- [x] 2.5 Gate Wave 2 with loss/config unit tests, the executed zero-policy probes, strict OpenSpec validation, and residue searches for old branching and public extension hooks; do not begin Wave 3 with an unresolved test or validation failure.

> **Wave 2 closed (2026-08-20, opus builder, zero correction rounds):**
> private frozen `TokenLossBinding` inventory in `src/losses/bindings.py`
> (base_ce/protected/forbid, token_type_gate/protected/detached_diagnostic,
> coord_gaussian_rps/auxiliary/omit; not re-exported), single composition
> path `LossRunner._active_token_losses()`; seven ad hoc branches deleted
> (enumerated in the builder report). Zero policies: forbid raises
> `loss.base_ce_weight_forbidden` at construction; ablation runs
> denominator+fp32 math under no-grad with weighted = exact detached 0.0
> and no objective autograd edge; omit = no instance/denominator/call/
> bundle/finite/metric (instance at weight 0 is now a hard error).
> **Entry-audit F-2 discharged with RED receipt**: the weighted-keyed
> finite path was observed labelling raw=NaN ablation bundles safe before
> the fix; finite derivation moved to RAW at all four sites including
> `src/runtime/finite_gates.py:45` (`RankScalarFiniteReport` gate — the
> one edit outside src/losses, mandated by F-2's all-rank clause;
> behaviour-preserving for objective terms where weighted = raw x finite
> positive weight). 2.4 probes proven sensitive via three reverted source
> mutations (objective-edge, detached-flag, omit-constructs). Accepted
> unrequested fail-fast: `loss.streaming_plan_runner_mismatch` making the
> plan-from-this-runner invariant explicit. Field names unchanged (Wave 4
> owns the rename); omitted-auxiliary fields now absent per spec, with
> tests/losses/test_runner.py + tests/eval/test_forward_eval.py re-pinned
> to base_ce weight 1.0 as forced by the contract. Trainer and
> src/eval/forward.py byte-unchanged (F-4 respected). Gate (lead
> independent replay) 417/0/0; residue 31 hits all in the closed path;
> strict validation valid; determinant fingerprints + payload baseline
> byte-identical; 24 owner files zero diff. Note: commit `e6f923534`
> (user's own AGENTS.md contract update) interleaved before this wave's
> commit — no owner-surface overlap.

## 3. Wave 3 - Global Objective And DDP Parity

- [x] 3.1 Add failing fixtures that distinguish semantic raw value, weighted semantic value, and backend-compensated local backward contribution at world size one and with unequal rank-local eligible-segment counts.
- [x] 3.2 Refactor streaming planning, micro-step computation, and finalization so fp32 per-atom math and global planned-step `segment_balanced` denominators remain unchanged while backend mean-gradient compensation affects the differentiable local contribution exactly once and never semantic telemetry.
- [x] 3.3 Add parameter-gradient and one-optimizer-update parity tests comparing an unequal-rank distributed planned step with the equivalent world-size-one batch for base CE, enabled gate, and positive-weight coordinate auxiliary within declared tolerances.
- [x] 3.4 Add gate-ablation parity tests proving that finite detached diagnostics preserve the base-CE raw value, objective, gradient, finite decision, and optimizer update of a base-only reference; separately inject a non-finite gate diagnostic and prove all ranks skip before backward under the existing fail-closed scalar gate.
- [x] 3.5 Exercise train and forward-eval reducers with unequal rank-local denominators and verify exact counts, raw/weighted aggregation, no double scaling, and no collective-order divergence.
- [x] 3.6 Before any distributed or GPU-backed action, review the frozen command manifest, obtain fresh user authorization for each GPU action, and record bounds for world size, planned steps, model forwards, cache/materialization passes (required `0`), wall time, peak GPU memory, and artifact bytes. Then gate Wave 3 with focused loss/runtime/eval tests, the authorized two-rank reduction-and-update probe, strict OpenSpec validation, collective/residue review, and the single pre-DDP/cost standards plus intent-contract audit; stop on a bound violation or unresolved P0/P1.

> **Wave 3 closed (2026-08-20, code commit `a20673078`; opus builder, zero
> correction rounds; pre-DDP audit `receipts/wave-3-pre-ddp-audit.md`
> STANDARDS + INTENT both PASS-WITH-DISPOSITIONS, 0 P0/0 P1, Wave-4 entry
> CLEARED):** backend mean-gradient compensation now applied exactly once
> at one site (`backward_contribution`); raw/weighted/total/metrics/finite
> surfaces traced clean of it (the pre-change defect: compensation was
> applied to raw and accidentally cancelled by mean-over-ranks reduction);
> reducers sum uncompensated partials for train at any world size and
> sharded eval, replicated eval untouched. Parity evidence: two-rank gloo
> unequal-rank grads+update vs ws-1 within rtol=1e-5/atol=1e-6 (mutation
> sensitivity 0.09/0.18, ~5 orders above the bar); gate-ablation bitwise
> (`torch.equal`, justified: no gate op in the objective graph);
> distributed non-finite gate injection converges one all-rank
> pre-backward skip (entry-audit F-2's covering receipt; finite_gates.py
> byte-unchanged this wave). 3.6 packet frozen at `a20673078` and executed
> once (CPU gloo, 0 GPU, 0 cache passes, exit 0 / 0 findings / 12
> collective ops per rank; `receipts/wave-3-two-rank-probe-{packet.md,
> receipt.json}`; manifest amend-8). Gates: 366/0/0 (lead + audit
> replays), fixtures/collective 15/15, determinant fingerprints + payload
> baseline re-verified independently twice; audit-owned repo-free DDP
> sensitivity check MECHANISM_CONFIRMED. Disclosures: per-rank telemetry
> semantics changed (per-rank rows now carry uncompensated partials;
> reduced values unchanged); `_total_loss` prefers `backward_loss` with a
> ws-1-identity fallback (audit I-1/I-6: Wave 4/5 must fail-close the
> fallbacks); audit I-4: bind the replicated-eval predicate once; audit
> I-5: Wave 4 must make an explicit row-schema decision for the new
> additive keys. Owned open PRE-EXISTING defects (predate this change,
> excluded from parity gating with in-test comments): train-side
> `count/packs`+`count/examples` mean-over-ranks and unweighted
> `token_weighted_diag` (introduced `2b0a2165a`), zero-eligible-segment
> collective desync (`e1662c2c7`/`3cd40f5f0`) — Wave-4 candidates.
> Wave-5 receipt schema must add `commit`, `wall_seconds`, and a
> cache-root sha inventory. `tests/fixtures/` top-level tree hash moved
> only by the disclosed Wave-1 live-input fixture deviation; the frozen
> `training_orchestration/` subtree is unchanged at `2fe137cc`.

## 4. Wave 4 - Canonical Loss Telemetry

- [x] 4.1 Add artifact-first failing tests for exact train and eval row schemas using `loss/<term>/raw`, `loss/<term>/weighted`, `loss/total`, matching counts/denominators/finite fields, gate-ablation retention, optional-family omission, and JSON-null normalization of computed non-finite diagnostics.
- [x] 4.2 Update loss-bundle finalization and rank-zero train/eval projection to emit the new explicit field families, remove the ambiguous `loss/<term>` aliases, and keep total loss equal to the sum of weighted objective terms.
- [x] 4.3 Search and migrate all current code, tests, scripts, selectors, and operator tooling that consume the old per-term fields; preserve historical JSONL and historical readers as commit-bound evidence rather than rewriting artifacts.
- [x] 4.4 Verify bounded row shape for enabled baseline, gate ablation, enabled coordinate auxiliary, omitted coordinate auxiliary, and unsafe-step serialization without adding per-rank streams or new artifact families.
- [x] 4.5 Gate Wave 4 with artifact/loss/eval integration tests, an executed one-step row projection probe, strict OpenSpec validation, and old-field and duplicate-alias residue searches; do not begin Wave 5 with an unresolved test or validation failure.

> **Wave 4 closed (2026-08-20, opus builder — session dropped once on an
> API error and was resumed with edits intact; zero correction rounds):**
> rows now emit `loss/<term>/raw` + `/weighted` + `/selected_count` with
> the existing `/segment_count`, `/token_weighted_diag`, `finite/<term>`
> namespaces; bare `loss/<term>` aliases removed with NO dual-write; the
> same projection serves train and forward eval, asserted on all five 4.4
> shapes incl. unsafe-step JSON-null normalization (`non_finite_fields`
> exact list, no NaN/Infinity literals in bytes). RED first: 6 failed /
> 4 passed artifact-first module, with the 4 green-at-birth nodes proven
> by reverted mutations (I-5 leak injection; I-1 guard neutralization).
> Consumer migration: 9 files, not entry-audit F-6's 4 (Waves 2-3 added
> four test files; `tests/eval/test_forward_eval.py` builds keys via
> f-strings and was invisible to the literal grep — methodological note
> recorded in amend-9 for Wave 5.5). Carried obligations: **I-5
> discharged** (backward_loss/backward_contribution/backend_gradient_scale
> asserted absent from persisted rows, mutation-receipted); **I-4
> discharged** (`_is_replicated_eval_reduction` single predicate, truth
> table + single-call-site test); **I-1/I-6 partial** — `_total_loss`
> fail-closes missing tensor `backward_loss`
> (`trainer.loss_bundle_backward_loss_missing`), documented duck-typed
> path retained; named remainder for Wave 5:
> `src/losses/runner.py:1044` micro-artifact `backward_contribution`
> fallback. Pre-existing count defects (mean-reduced `count/packs`/
> `count/examples`, unweighted train `token_weighted_diag`) NOT fixed —
> ws-1 row shapes did not force them; source TODO with provenance
> `2b0a2165a` records both. Identity-pin flip accepted (amend-9): V1
> test-module drift guard re-pinned, probe-script provenance seal
> `dfbb4d63` untouched. Flake disclosure: one 1-of-9 non-reproducing
> failure in the untouched Wave-2 probe
> `test_gate_ablation_creates_no_autograd_edge_into_the_objective`
> (graph-shape assert; 200/200 direct trials stable; builder attribution:
> cross-module global-state leakage into `plan.backend_gradient_scale`) —
> recorded as a pre-existing test-isolation hazard for Wave 5.5 review,
> not absorbed. Gate 704/0/0 (builder x7 + lead replay); residue 77 hits
> all suffixed, zero bare; strict validation valid; determinant
> fingerprints + payload baseline byte-identical; frozen subtree
> `2fe137cc` re-asserted and pinned inside the new test module.

## 5. Wave 5 - Documentation And Production-Shaped Acceptance

- [x] 5.1 Update canonical loss/config/artifact operator docs and config examples to describe the protected baseline, named gate ablation, typed auxiliary placement, zero policies, raw/weighted fields, and the explicit SFT-only boundary without copying the OpenSpec change as a second authority.
- [x] 5.2 Run the complete focused config, loss, trainer, runtime, eval, and artifact suites plus repository config-inventory validation; fix failures without weakening the new strict contract or adding compatibility aliases.
- [x] 5.3 Recompute the exact train/eval determinant payloads and hashes after all source/config edits and require equality with the Wave 0 baseline. If either changes, stop for contract review and do not materialize, repair, overwrite, or publish a second cache. Obtain fresh user authorization for this distinct GPU action and record its own bounds for devices/world size, planned steps, model forwards, cache/materialization passes (`0`), wall time, peak GPU memory, and artifact bytes; then run the smallest production-shaped distributed supervised vertical smoke against the predecessor cache and bind the receipt to the frozen exact command, config, commit, artifact root, cache identities, bounds, and exact row fields.
- [x] 5.4 Inspect the smoke for finite objective/update status, raw-to-weighted arithmetic, exact term presence/absence, gate mode, global denominators, and single-writer artifact behavior; do not interpret the zero-work probe or smoke as a throughput or memory improvement claim.
- [x] 5.5 Run strict validation for this change and all affected stable-spec deltas, inspect the full change diff and supported-config migration, and search current roots for legacy protected placement, noncanonical gate constants/groups, dynamic loss hooks, ambiguous loss fields, and accidental historical edits.
- [x] 5.6 Obtain independent standards and user-intent/contract audit verdicts covering SFT-only scope, scientific meaning, DDP math, zero behavior, artifact compatibility, migration completeness, overdesign, and legacy residue; resolve every P0/P1 before marking the change implementation-complete.


> **Wave 5 closed / change complete (2026-08-20, commits `fa8233edf` part 1
> + this close-out; final audit `receipts/wave-5-final-audit.md` STANDARDS
> and USER-INTENT both PASS-WITH-DISPOSITIONS, 0 P0 / 0 P1, completion
> gate YES):** docs updated in four canonical pages (describe-and-link, no
> second authority; two historical-reference pages verified untouched by
> design); determinant equality probe EQUAL/EQUAL lead-executed AND
> audit-re-executed at clean HEAD post-smoke (cache inventory byte-equal
> to the pre-smoke capture, proving zero smoke-side cache bytes from
> bytes); production-shaped two-rank BF16 smoke on idle GPUs 0,1 under the
> frozen packet (standing user GPU grant): completed/applied/finite,
> 1 applied step, both predecessor fingerprints admitted,
> ablation row shape live-verified (gate raw 1.3676 train / 0.1993 eval
> visible, weighted exactly 0.0, zero coord/bare/backward keys,
> loss/total == sum weighted exactly, single rank-zero writer); focused
> full suite 2441/0/0 with 126 reasoned skips; residue searches clean
> structurally (rps_weight/vertical_prob 0.2 literals correctly
> distinguished from gate weights); I-1/I-6 fully discharged
> (`loss.micro_artifact_backward_contribution_missing`, RED-observed);
> F-7 Non-Finite Gates MODIFIED delta authored (lead) — sync carries two
> MODIFIED + one ADDED delta set. Manifest amend-10/11 retire the smoke
> placeholder and pin suite counts. Audit notes recorded: F-B (receipt
> provenance ~60s pre-commit, re-anchored), F-C (receipt schema lacked
> wall_seconds; inventory prose re-derived from bytes), F-G (smoke
> exercised the ablation shape only, packet-declared). Carried forward as
> non-blocking to `add-coordexp-swift-training-observability`: F-D
> (`LossTermResult.__post_init__` unreachable weighted_loss default), F-F
> (wave-2 probe test-isolation flake, 0/2441 recurrence), the two
> pre-existing count defects (`count/packs`+`count/examples`
> mean-over-ranks; unweighted train `token_weighted_diag`; TODO at
> train_runtime.py, provenance 2b0a2165a), and the zero-eligible-segment
> collective desync (`e1662c2c7`/`3cd40f5f0`).
