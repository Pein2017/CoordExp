## 0. Wave 0 - Pinned predecessors, owner graph, and cache invariant

- [ ] 0.1 Verify `reconcile-coordexp-swift-training-contracts` and `decompose-coordexp-swift-training-orchestration` are implementation-complete, synced, and archived; record their exact commits and the current change base commit, and stop if code, stable specs, docs, or archive dispositions disagree.
- [ ] 0.2 Record the post-decomposition owner/import graph and prove loss config, loss implementations, reporting consumers, and supported config migrations are outside `PACKING_CACHE_DETERMINANT_OWNERS`; record the exact admitted train/eval determinant payloads, aggregate hashes, target paths, and cache-hit receipts from the predecessor.
- [ ] 0.3 Freeze an exact execution-command manifest covering cwd, `conda` environment, commands, configs, world size/devices, artifact roots, expected evidence, and quantitative limits. Require a reviewable append-only amendment for any later command change.
- [ ] 0.4 Run the untouched focused config/loss/runtime/reporting/cache-identity/artifact suites and strict OpenSpec validation, then obtain the entry standards and intent-contract audits; do not begin Wave 1 with an unresolved P0/P1.

## 1. Wave 1 - Strict Supervised Config Contract

- [ ] 1.1 Add config-first failing tests for base CE omission/reweighting, the exact gate group tuple, `enabled: 0.1`, `zero_weight_ablation: 0`, every incompatible mode/weight mutation, legacy protected coordinate placement, typed auxiliary placement, optional auxiliary zero, unknown loss names, and import/callable-style hooks.
- [ ] 1.2 Implement the strict protected and auxiliary config models, including the discriminated gate mode, exact constants and groups, and migration-oriented validation errors without a legacy alias.
- [ ] 1.3 Enumerate the catalog-supported/current CoordExp-Swift production, smoke, and measurement configs; migrate enabled gates to `0.1`, mark zero gates as `zero_weight_ablation`, move coordinate Gaussian/RPS under the typed auxiliary surface, and inspect every config diff without modifying historical/archive roots.
- [ ] 1.4 Add an inventory test that resolves every current supported training config and separately proves representative historical configs remain provenance rather than accepted current inputs.
- [ ] 1.5 Gate Wave 1 with focused config tests, a dry config-resolution probe over the full supported inventory, strict OpenSpec validation, and searches for legacy protected-coordinate placement/dynamic hooks in current roots; do not begin Wave 2 with an unresolved test or validation failure.

## 2. Wave 2 - Closed Loss Composition And Zero Policies

- [ ] 2.1 Add interface-level failing tests for the closed binding inventory and its exact protected/auxiliary role, `segment_balanced` normalizer, and `forbid`/`detached_diagnostic`/`omit` zero policies.
- [ ] 2.2 Introduce the private frozen `TokenLossBinding` metadata and refactor loss construction to one explicit closed composition path without registry discovery, import-by-name, callable config, or pass-through factories.
- [ ] 2.3 Implement base-CE forbidden-zero behavior, the gate-ablation no-grad diagnostic path, and complete optional-auxiliary omission from construction, denominators, calls, bundles, finite checks, and metrics; delete the superseded ad hoc coordinate/protected branching.
- [ ] 2.4 Add correctness probes using construction/call sentinels and autograd saved-tensor or graph inspection to prove an omitted auxiliary creates no term work or retained graph and a gate ablation creates no objective autograd edge; record explicitly that this is not an efficiency or peak-memory claim.
- [ ] 2.5 Gate Wave 2 with loss/config unit tests, the executed zero-policy probes, strict OpenSpec validation, and residue searches for old branching and public extension hooks; do not begin Wave 3 with an unresolved test or validation failure.

## 3. Wave 3 - Global Objective And DDP Parity

- [ ] 3.1 Add failing fixtures that distinguish semantic raw value, weighted semantic value, and backend-compensated local backward contribution at world size one and with unequal rank-local eligible-segment counts.
- [ ] 3.2 Refactor streaming planning, micro-step computation, and finalization so fp32 per-atom math and global planned-step `segment_balanced` denominators remain unchanged while backend mean-gradient compensation affects the differentiable local contribution exactly once and never semantic telemetry.
- [ ] 3.3 Add parameter-gradient and one-optimizer-update parity tests comparing an unequal-rank distributed planned step with the equivalent world-size-one batch for base CE, enabled gate, and positive-weight coordinate auxiliary within declared tolerances.
- [ ] 3.4 Add gate-ablation parity tests proving that finite detached diagnostics preserve the base-CE raw value, objective, gradient, finite decision, and optimizer update of a base-only reference; separately inject a non-finite gate diagnostic and prove all ranks skip before backward under the existing fail-closed scalar gate.
- [ ] 3.5 Exercise train and forward-eval reducers with unequal rank-local denominators and verify exact counts, raw/weighted aggregation, no double scaling, and no collective-order divergence.
- [ ] 3.6 Before any distributed or GPU-backed action, review the frozen command manifest, obtain fresh user authorization for each GPU action, and record bounds for world size, planned steps, model forwards, cache/materialization passes (required `0`), wall time, peak GPU memory, and artifact bytes. Then gate Wave 3 with focused loss/runtime/eval tests, the authorized two-rank reduction-and-update probe, strict OpenSpec validation, collective/residue review, and the single pre-DDP/cost standards plus intent-contract audit; stop on a bound violation or unresolved P0/P1.

## 4. Wave 4 - Canonical Loss Telemetry

- [ ] 4.1 Add artifact-first failing tests for exact train and eval row schemas using `loss/<term>/raw`, `loss/<term>/weighted`, `loss/total`, matching counts/denominators/finite fields, gate-ablation retention, optional-family omission, and JSON-null normalization of computed non-finite diagnostics.
- [ ] 4.2 Update loss-bundle finalization and rank-zero train/eval projection to emit the new explicit field families, remove the ambiguous `loss/<term>` aliases, and keep total loss equal to the sum of weighted objective terms.
- [ ] 4.3 Search and migrate all current code, tests, scripts, selectors, and operator tooling that consume the old per-term fields; preserve historical JSONL and historical readers as commit-bound evidence rather than rewriting artifacts.
- [ ] 4.4 Verify bounded row shape for enabled baseline, gate ablation, enabled coordinate auxiliary, omitted coordinate auxiliary, and unsafe-step serialization without adding per-rank streams or new artifact families.
- [ ] 4.5 Gate Wave 4 with artifact/loss/eval integration tests, an executed one-step row projection probe, strict OpenSpec validation, and old-field and duplicate-alias residue searches; do not begin Wave 5 with an unresolved test or validation failure.

## 5. Wave 5 - Documentation And Production-Shaped Acceptance

- [ ] 5.1 Update canonical loss/config/artifact operator docs and config examples to describe the protected baseline, named gate ablation, typed auxiliary placement, zero policies, raw/weighted fields, and the explicit SFT-only boundary without copying the OpenSpec change as a second authority.
- [ ] 5.2 Run the complete focused config, loss, trainer, runtime, eval, and artifact suites plus repository config-inventory validation; fix failures without weakening the new strict contract or adding compatibility aliases.
- [ ] 5.3 Recompute the exact train/eval determinant payloads and hashes after all source/config edits and require equality with the Wave 0 baseline. If either changes, stop for contract review and do not materialize, repair, overwrite, or publish a second cache. Obtain fresh user authorization for this distinct GPU action and record its own bounds for devices/world size, planned steps, model forwards, cache/materialization passes (`0`), wall time, peak GPU memory, and artifact bytes; then run the smallest production-shaped distributed supervised vertical smoke against the predecessor cache and bind the receipt to the frozen exact command, config, commit, artifact root, cache identities, bounds, and exact row fields.
- [ ] 5.4 Inspect the smoke for finite objective/update status, raw-to-weighted arithmetic, exact term presence/absence, gate mode, global denominators, and single-writer artifact behavior; do not interpret the zero-work probe or smoke as a throughput or memory improvement claim.
- [ ] 5.5 Run strict validation for this change and all affected stable-spec deltas, inspect the full change diff and supported-config migration, and search current roots for legacy protected placement, noncanonical gate constants/groups, dynamic loss hooks, ambiguous loss fields, and accidental historical edits.
- [ ] 5.6 Obtain independent standards and user-intent/contract audit verdicts covering SFT-only scope, scientific meaning, DDP math, zero behavior, artifact compatibility, migration completeness, overdesign, and legacy residue; resolve every P0/P1 before marking the change implementation-complete.
