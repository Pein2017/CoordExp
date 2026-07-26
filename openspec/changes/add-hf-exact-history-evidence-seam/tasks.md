## 1. Wave One — Baseline and Red Contract

- [ ] 1.1 Confirm explicit user approval before changing runtime code, tests,
  research scripts, configs, or launching model work; verify the exact
  `research-probes` worktree and preserve unrelated dirt.
- [ ] 1.2 In `ms`, run
  `python -m pytest tests/research tests/rollout_calibration -q -p no:randomly`,
  record every failing node ID, and confirm that failures partition only into
  the reviewed 38 mutable-`unit.md` identity drifts and 3 processed-data
  provenance drifts. Route the latter to the existing data-provenance owner,
  change no pins/tests/`pytest.ini`, and stop this change if a third class
  appears.
- [ ] 1.3 Add failing interface tests under `tests/inference/` for a
  single-request `HFExactHistory`, immutable literal token append and hash,
  rejected invalid IDs, forged/cross-session/closed-session use before model
  forward, and absence of public native objects.
- [ ] 1.4 Add failing inference tests for at-use Qwen position construction,
  ordered FP32 chosen-token log-probabilities, strict-greater vocabulary rank,
  one-token and multi-token continuations, and no full-logit return.
- [ ] 1.5 Add failing receipt tests for observed parameter-dtype counts and
  attention implementation, including explicit `null` rather than fallback to
  declared/configured values when unavailable.

## 2. Wave One — HF-Owned Implementation

- [ ] 2.1 Add the frozen `HFExactHistory` and `HFChosenTokenEvidence` public
  records in the HF inference owner; retain an internal unforgeable session
  binding and private per-request multimodal context that is cleared on close.
- [ ] 2.2 Implement `special_token_ids`, one-request
  `prepare_exact_history`, and immutable `extend_exact_history` by reusing the
  existing processor/media/prompt/grid checks and canonical `token_ids_sha256`,
  without decode/re-tokenize or eager position mutation.
- [ ] 2.3 Implement `teacher_forced_evidence` over the full
  conditioning-plus-continuation sequence with use-time attention and Qwen
  M-RoPE positions, FP32 raw log-softmax, strict-greater rank, and existing
  `RuntimeContractError` ownership. Reuse
  `teacher_forced_chosen_token_logprobs` only if the new tests prove equivalent
  position semantics.
- [ ] 2.4 Populate
  `effective_settings.observed_model_dtype` and
  `effective_settings.observed_attn_implementation` from the loaded model,
  preserving declared launch/config fields separately and using `null` for
  unavailable observations.
- [ ] 2.5 Make the new `tests/inference/` slice green and run the pre-existing
  HF backend/session tests to prove ordinary decode, raw/policy likelihood,
  receipt validation, and close behavior are unchanged.

## 3. Wave One — Two Real Consumers and Gate One

- [ ] 3.1 Migrate only the boundary-scoring evidence path in
  `run_continuation_locality_boundary_scoring.py` to prepare/extend one history,
  request opener and terminal evidence, and keep cohort selection, prefix,
  margin, receipt schema, and claim boundary caller-owned.
- [ ] 3.2 Run the frozen locality smoke for Source and transition step 36 in a
  new temporary output root, selecting
  `same_image_near_continue-image-10040-depth-1-16497c21af80` through the
  declared cohort plus `--limit 1`. Require exact executed prompt/prefix IDs and
  hashes and exact boundary log-probabilities/margins versus the existing
  `smoke-v1/locality` receipts.
- [ ] 3.3 Migrate only `_score_state_target` evidence construction in
  `run_exact_prefix_owner_compositionality.py`; retain its row tokens/phases,
  sums/means, ranks, metadata, forced release, parsing, matching, and owner
  interpretation in the script.
- [ ] 3.4 Run the frozen owner smoke for Source and transition step 36 in a new
  temporary output root with
  `--case-id owner-case-image-225458-depth-1-owner-225458-78480-prefix-eb93c607ad0f`,
  `--runtime-dtype fp32`, `--max-new-tokens 64`, and
  `--malformed-limit 2`. Require exact prompt/prefix/candidate IDs and hashes,
  exact per-token log-probabilities/ranks and phase reductions, and unchanged
  release/parse/match evidence versus the existing `smoke-v2/compositionality`
  receipts.
- [ ] 3.5 Only after both parity checks pass, remove superseded scoring and
  runtime-observation private accesses from the two selected call sites. Do not
  delete shared `_forward_logits`, position, rank, generation, stopping,
  parsing, matching, or release helpers used by untouched consumers; document
  each remaining `_model`, `_tokenizer`, or native-input access in the two
  pilots as vision-parity or release machinery outside this seam.
- [ ] 3.6 Verify that no new config file, artifact schema, reducer, owner policy,
  row-phase policy, generic mapping, or closeout/router automation was added;
  confirm the two migrated receipts remain independently readable by their
  unchanged downstream code.
- [ ] 3.7 **Gate One:** run focused inference tests, the existing continuation
  locality research tests, both two-role real smokes, strict OpenSpec
  validation, diff/residue checks, and one independent review reporting
  separate standards-correctness and scientific-intent verdicts. Do not begin
  Wave Two with any unresolved P0/P1 finding.

## 4. Wave Two — Final Acceptance and Stop

- [ ] 4.1 Resolve only in-scope Gate One findings and rerun the smallest test or
  smoke that proves each disposition; retract the stable seam if either caller
  requires full logits, native tensors/stopping, generation hooks, a different
  history shape, or a relaxed scientific estimand.
- [ ] 4.2 Run the complete default-collected test suite plus
  `tests/inference/`,
  `tests/research/test_continuation_locality_owner_compositionality.py`, and the
  explicit research/rollout-calibration baseline. Confirm no new failure class
  or change to the routed 38/3 baseline.
- [ ] 4.3 Repeat the four representative Source/transition smokes from a clean
  temporary root and verify deterministic exact evidence parity, unchanged
  receipt schemas, unchanged downstream readability, and failure diagnostics
  at the backend/session layer for injected lifecycle/token errors.
- [ ] 4.4 Inspect the final diff and dependency surface: scientific choices
  remain visible in both scripts, only duplicated execution mechanics moved,
  no second state-bank-style registry escaped the live HF session, and the two
  untouched consumers and all historical outputs remain unchanged.
- [ ] 4.5 **Gate Two:** run `openspec validate
  add-hf-exact-history-evidence-seam --strict`, formatting/type checks relevant
  to touched files, final residue checks, and a final independent audit. Report
  verification, routed pre-existing failures, deletions, residual private
  accesses, and residual uncertainty; stop for explicit user approval before
  archive, broader migration, or any new research unit.
