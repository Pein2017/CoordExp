## 1. Authorization and Frozen Inputs

- [x] 1.1 Record explicit user authorization to begin implementation, recheck the exact `/data/CoordExp/.worktrees/research-probes` branch and dirty-file ownership, and confirm that model/GPU execution remains separately gated.
- [x] 1.2 Add `scripts/research/build_human13_k_union_manifest.py` with typed, canonical records for panel, Source, request, trajectory, owner, selected row, raw/clean prefix, duplicate event, arm, and global denominator identities.
- [x] 1.3 Add focused manifest tests under `tests/research/test_build_human13_k_union_manifest.py` for the exact panel hash, overfit-only admission, seed completeness, chronological duplicate-before-matching classification, duplicate/owner-positive disjointness including a dense two-owner collision fixture, raw-to-clean token surgery, `G/H/M`, target selection, terminal masks, zero denominators, and fail-closed generic blind admission.
- [x] 1.4 Add `scripts/research/collect_human13_k16_vllm.py` and tests that plan four batches of four explicit `n=1` requests per image with seeds `21001..21016`, sampling repetition penalty `1.10`, canonical result order, incomplete-request failure, and non-claiming optional cache telemetry.

## 2. Objective Math and No-Update Census

- [x] 2.1 Add failing pure-math tests in `tests/losses/test_human13_k_union.py` for owner-mean masked row CE, once-per-image prefix-free union mass, coherent full-H token bottleneck hinge, and all-event image-balanced duplicate-token unlikelihood including finite loss and gradient at a target margin of at least thirty nats.
- [x] 2.2 Implement the minimal fp32 helpers in `src/losses/human13_k_union.py`, including explicit numerator/denominator outputs, detached bottleneck competitor selection, finite checks, candidate weights, and effective-owner diagnostics.
- [x] 2.3 Add `scripts/research/census_human13_k_union_trie.py` and tests for exact-token row deduplication, coherent-chain first non-argmax/minimum margin/tie/token-role census, cross-surface target-margin drift, A8-prime blocking, and byte-identical frozen targets before and after census.
- [x] 2.4 Gate Wave 1 with targeted CPU tests, strict OpenSpec validation, diff/residue checks for deferred A2/A5/candidate-tree/GT-IoU work, and one bounded standards plus research-intent review with no unresolved P0/P1.

## 3. Packed Panel-Step Runner

- [x] 3.1 Add interface-first tests in `tests/research/test_run_human13_k_union_overfit.py` for logical segment construction, stable descending-length first-fit planning, isolated causal/MRoPE positions, 12,000-token fail-closed preflight, A4 image atomicity, and coherent A1/A8/full-GT segments.
- [x] 3.2 Add `scripts/research/run_human13_k_union_overfit.py` as an experiment-local adapter over the existing Qwen forward, no-padding packing, planned-step accumulation, AdamW/DoRA, finite gate, RunWriter, and CheckpointWriter interfaces; do not add a second trainer or modify generic StateBank admission.
- [x] 3.3 Verify through tests that every physical pack uses the complete panel denominator, model parameters and optimizer state remain unchanged between packs, and exactly one optimizer step occurs per panel exposure.
- [x] 3.4 Add compact performance counters for logical/packed tokens, pack count, padding, utilization, GPU seconds, wall time, and peak memory while explicitly omitting any unmeasured prefix-KV or image-encoder reuse claim.

## 4. Arm Materialization, Launcher, and Analyzer

- [x] 4.1 Add strict research configs under `configs/coordexp_swift/research/human13_k_union/` for Frozen Source, full-GT capacity, A0 shared no-H background, A1, A3, A4, A7, A8-prime, and conditional A6, with byte-identical Source, fresh AdamW state, and exact fail-closed trainable-surface/optimizer/scheduler/clipping/family-coefficient values per arm.
- [x] 4.2 Add a dry-run materializer that emits isolated output roots and resolved arm plans without model load, forward, optimizer, checkpoint write, or GPU allocation, and test that A6 is omitted when no eligible `H_mid` donor exists.
- [x] 4.3 Add `scripts/research/analyze_human13_k_union.py` and tests that apply chronological duplicate exclusion before the declared cardinality-first, maximum-total-IoU one-to-one matcher, award later duplicates no owner credit, and report K-hit gained, Source retained/lost, K-miss incidental gain, duplicate/unmatched/invalid/malformed/cap burden, and per-image, legacy-twelve, image-2299, and pooled views.
- [x] 4.4 Add a bounded independent-arm launcher whose default is dry-run, whose explicit execute mode is separate from plan materialization, which assigns at most one world-size-one Accelerate process per GPU, and which never shares output roots or optimizer state; execution still requires documented user model/GPU authority outside the launcher.
- [x] 4.5 Gate Wave 2 CPU implementation with targeted and integration tests, strict config/OpenSpec validation, dry-run zero-model-action evidence, residue checks, and bounded standards plus research-intent reviews with no unresolved P0/P1.

## 5. Separately Authorized Full-Panel Discovery and Freeze

- [x] 5.1 After explicit model/GPU authorization, acquire or identity-verify all thirteen Source-matched HF batch-size-one clean-greedy outputs and execute the exact 208 K requests; finalize only when every image has sixteen unique declared seeds.
- [x] 5.2 Seal the full-panel canonical manifest after chronological duplicate exclusion and retained-row matching, then run the no-update census to freeze selected rows, donor/A6 applicability, coherent order, cross-surface drift, and A8-prime applicability/margin.
- [x] 5.3 Verify that every training entry rejects fixtures, partial manifests, missing seeds, duplicate/positive-set intersections, or unsealed census fields; publish the no-update identity and compact runtime counters and stop before an optimizer action.

## 6. Separately Authorized Production-Shaped Slice

- [x] 6.1 After a distinct update authorization, select exactly one eligible image from the sealed full-panel manifest and run it through packing, forward/backward, one update, checkpoint write/read, HF batch-size-one clean greedy, declared matching, and final projection from a fresh Source/AdamW state.
- [x] 6.2 Publish the vertical-slice identity, masks, denominators, applied-step count, checkpoint readback, raw output, owner outcome, packed-token/runtime/peak-memory counters, and any measured failure without silently changing the 12,000-token or arm semantics.
- [x] 6.3 Run targeted real-entry regression checks and one bounded standards plus research-intent audit; stop and return the measured full-matrix cost proposal if any P0/P1, nonfinite value, write/read failure, mismatch, OOM, or overlength condition remains.

## 7. Separately Authorized Human-13 Matrix

- [x] 7.1 Obtain a distinct user authorization for the measured matrix and launch only applicable independent arms, at most eight concurrently, at panel exposures `0, 1, 2, 4, 8, 16` from byte-identical Source and fresh optimizer states under the frozen sixteen-update contract.
- [x] 7.2 Freeze all original-prompt HF clean-greedy outputs and publish arm-by-arm owner outcomes and burden without promoting an arm from teacher-forced loss, trie, margin, or exact-token diagnostics.
- [x] 7.3 Stop at the first complete table and request another user decision before a fresh Source/AdamW 100-update run through exposures `32, 64, 100`, changing objectives, supervising K-miss, refreshing targets, or adding a candidate-tree/GT-IoU route.
- [x] 7.4 Reconcile results into the owning research unit and project memory, run strict OpenSpec verification plus final bounded audits, and leave checkpoint promotion, stable-spec sync, archive, commit, and publication to separately authorized follow-up actions.

## Execution disposition — 2026-08-12

The authorized bounded screen stopped at its first complete interpretable table.
At that stop point tasks 5.2 and 5.3 remained honestly incomplete: the
canonical manifest was sealed, but the no-update census emitted no artifact
after its declared repair budget, so A8-prime remained unavailable. A4 was
omitted by its then-frozen atomic packing bound and A6 failed closed before
model load; neither was a scientific null. The separately authorized
2026-08-13 successor below subsequently completed 5.2/5.3 and recovered these
mechanical surfaces. This historical disposition remains the authority for the
first matrix only.

## 8. Authorized Missing-Arm Successor — 2026-08-13

- [x] 8.1 Revise the OpenSpec contract and write a bounded execution plan that
  defines A4 as one logical per-image union objective with exact fixed-theta
  two-pass physical streaming, keeps chunk-local normalization forbidden, and
  limits execution to fresh A4/A6/A8-prime arms.
- [x] 8.2 Add RED/GREEN coverage and repair A6 so donor treatment prefixes
  delete every earlier frozen duplicate-row span and byte-match the sealed
  clean-prefix binding for all eligible donors.
- [x] 8.3 Add RED/GREEN coverage and repair A8 census skeleton cloning so image
  identity, prompt boundary, and owner-row-token metadata survive every packed
  and HF clone; publish the census only as a complete immutable output plus
  receipt.
- [x] 8.4 Add pure gradient-equivalence and runner tests for A4 two-pass
  streaming, then implement no-grad global candidate scoring, fp32 per-image
  weights, unchanged-parameter gradient replay, and one AdamW step per panel
  exposure without chunk-local losses.
- [x] 8.5 Pass the targeted CPU suite, strict OpenSpec validation, immutable
  root preflight, exact manifest/Source/K binding, A6 full-ledger prefix audit,
  and one production-shaped A4 vertical slice before broad execution.
- [x] 8.6 Run a fresh immutable no-update census; if its finite drift admits
  A8-prime, launch fresh A4/A6/A8-prime arms at exposures `1,2,4,8,16` with
  isolated Source/AdamW/output roots, then run original-prompt HF fp32/SDPA
  batch-size-one repetition-penalty-1.0 greedy evaluation. If A8 remains
  mechanically blocked, record the exact complete census disposition and run
  only A4/A6.
- [x] 8.7 Publish one bounded successor analysis against the frozen Source,
  update the owning research record, run conclusion-changing verification,
  and stop without a 100-update continuation, K-miss supervision, checkpoint
  promotion, stable-spec sync, or OpenSpec archive.
