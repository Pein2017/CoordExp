# Tasks

## 1. OpenSpec and Governance

- [x] Convert all change-local specs to OpenSpec delta form with `ADDED` or `MODIFIED` requirements.
- [x] Add modified deltas for packing, coord-token mode, coord aux, bbox aux, trainer metrics, and encoded-cache eligibility.
- [ ] Run `openspec validate add-stage1-set-continuation-training --strict` once the OpenSpec CLI is available on `PATH`. Blocked during smoke-prep check on 2026-04-25 because `openspec` was not found on `PATH`.
- [x] Add or update canonical docs routing so `progress/directions/full_idea_v5.md` is discoverable through progress routers.

## 2. Strict Config and Setup Routing

- [x] Add strict `custom.stage1_set_continuation` config dataclasses.
- [x] Add a typed strict top-level `benchmark` config section for benchmark
      group identity, comparator, intended variable, and comparability label.
- [x] Add `custom.trainer_variant: stage1_set_continuation` routing before ordinary Stage-1 trainer factory fallback.
- [x] Update trainer class composition so ordinary one-sequence Stage-1 loss mixins are excluded for this variant.
- [x] Add a raw-sample-preserving setup path with `remove_unused_columns=false`.
- [x] Validate subset ratios, prefix order, candidate scoring mode/K, structural-close settings, PEM settings, aux adapter settings, and metric schema version at config resolution.
- [x] Require effective coord-token rendering and reject raw-text integer coordinate mode.
- [x] Fail fast when `training.packing=true` before static pack-plan train or eval datasets can be built.
- [x] Fail fast or bypass encoded-sample cache according to `training.encoded_sample_cache.ineligible_policy`.
- [x] Fail fast when an enabled auxiliary loss lacks a set-continuation branch adapter.

## 3. Metadata, Collator, and Branch Encoder

- [x] Preserve `assistant_payload`, `messages` or equivalent image/prompt identity, `metadata`, `sample_id`, `base_idx`, and dataset label through the collator.
- [x] Use `assistant_payload.objects` as the canonical object-entry source for subset/candidate sampling.
- [x] Add a regression test proving `sample["objects"]` is not assumed to contain serialized object entries.
- [x] Implement `Stage1SetContinuationBranchEncoder` or equivalent helper for branch construction.
- [x] Ensure the branch encoder wraps template state, disables packing/padding-free assumptions, preserves image inputs, and runs existing image-token/grid contract checks in smoke tests.
- [x] Ensure non-model extras are stripped before `model(**inputs)`.

## 4. Serialization and Structural Spans

- [x] Implement indexed canonical object-entry fragment extraction from the existing CoordJSON assistant serialization path.
- [x] Avoid content-based substring matching for object spans.
- [x] Include candidate entry delimiter and object-entry terminator in candidate labels.
- [x] Exclude global schema closure, assistant suffix, EOS, and chat-template stop tokens from candidate-entry scores.
- [x] Implement canonical structural-close span extraction for close-start and close-sequence supervision.
- [x] Test empty-prefix, non-empty-prefix, full-prefix, first/middle/last candidate, duplicate identical entries, same-desc different-bbox entries, and supported object field orders.
- [x] Test that `<|im_end|>`, `<|end_of_text|>`, EOS, and object-entry close tokens are excluded from structural-close targets.

## 5. Subset and Candidate Sampling

- [x] Implement empty-prefix, random-subset, leave-one-out, and small full-prefix sampling.
- [x] Implement random prefix ordering by default, plus a preserved dataset-order ablation mode.
- [x] Define sampler RNG as a pure function of resolved seed, epoch, sample identity, rank, and documented microstep salt.
- [x] Implement deterministic small-object-count fallbacks for `|O| = 0`, `|O| = 1`, and invalid mode selections.
- [x] Implement exact all-remaining candidate scoring selection.
- [x] Implement optional uniform candidate subsampling with `K`.
- [x] Log configured and resolved valid-mode mixtures.

## 6. Losses and Objective Semantics

- [x] Implement coord-aware full-entry candidate scoring with full-vocab non-coord logprobs and coord-vocab-normalized coord-token logprobs.
- [x] Implement `mp/logZ_scored_raw`, `mp/logZ_remaining_exact`, and `mp/logZ_remaining_est`.
- [x] Implement `loss/mp` from exact or raw scored-candidate logZ according to candidate mode.
- [x] Implement responsibility statistics over scored candidates.
- [x] Implement PEM threshold-loss objective in v1: `positive_evidence_margin.objective=threshold_loss`.
- [x] Require exactly one of `rho` or `log_rho` when PEM threshold loss is enabled.
- [x] Ensure PEM uses exact logZ or uniform-importance estimated logZ, not raw sampled logZ.
- [x] Log `loss/mp_diagnostic` when PEM threshold loss optimizes `loss/pem`.
- [x] Implement `loss/anti_close_start` for `R != empty`.
- [x] Implement `loss/weak_schema_close` for `R = empty`.
- [x] Exclude `R = empty` and zero-weight full-prefix metric-only samples from MP objective denominators.
- [x] Implement branch-local `coord_soft_ce_w1` auxiliary adapter.
- [x] Implement branch-local `bbox_geo` auxiliary adapter.
- [x] Implement branch-local `bbox_size_aux` auxiliary adapter.
- [x] Aggregate branch-local aux as mean-like candidate atoms uniformly over scored valid candidates.
- [x] Keep responsibility-weighted aux out of v1 unless added as a separately named future mode.

## 7. Metrics, Artifacts, and Benchmark Profiles

- [x] Add variant-specific metric documentation and metric-key parity tests.
- [x] Log all required `loss/*`, `mp/*`, `stop/*`, and `aux/*` keys from the set-continuation spec.
- [x] Log candidate entry token counts, logprob sums, per-token logprobs, coord/non-coord diagnostics, responsibility-vs-length diagnostics, and small-n validity counters.
- [x] Log repeated-forward budget metrics and scored-candidate fractions.
- [x] Preserve ordinary Stage-1 metric parity and prove ordinary SFT does not emit set-continuation MP keys.
- [x] Add artifact checks for `resolved_config.json`, `effective_runtime.json`, `experiment_manifest.json`, and existing run-manifest surfaces.
- [x] Record set-continuation metric schema version in resolved config or run manifest.
- [x] Add a single checked-in production profile with stable `benchmark.group_id` and `benchmark.control_group_id`.
- [x] Add benchmark manifest validation that records intended variable, comparator, eval scope/view, sample count, checkpoint/base identity, inference decoding controls, and comparability label.
- [x] Add benchmark manifest validation that records effective coord-slot scoring, aux objective settings, PEM threshold calibration provenance, realized prefix-mode coverage, and realized branch/token budget.
- [x] Add sparse-label FP caveat/report fields for proxy or partial-label eval views.
- [x] Add distributed train-time detection eval generation so DDP ranks decode
      eval shards while rank 0 merges/scales/logs final metrics. This is the
      existing verified eval baseline and is not part of the train-forward
      runtime stabilization work.

## 8. Documentation

- [x] Update `docs/training/STAGE1_OBJECTIVE.md` with a Stage-1 set-continuation section.
- [x] Update `docs/training/METRICS.md` with variant-specific MP/PEM/structural-close metrics.
- [x] Update `docs/data/PACKING.md` with v1 fail-fast behavior for `stage1_set_continuation`.
- [x] Update encoded-cache docs/spec references with metadata-only eligibility or bypass behavior.
- [x] Update `docs/training/README.md` and `docs/IMPLEMENTATION_MAP.md` with the new trainer variant and setup-path fork.
- [x] Document that v1 uses repeated independent forwards, not prefix sharing or branch masks.
- [x] Document that v1 is coord-token-only and rejects raw-text integer coordinate training.
- [x] Document the production entry config and the required benchmark manifest/report fields.

## 9. Verification

- [x] Add unit tests for strict config parsing and unknown/invalid nested keys.
- [x] Add fail-fast tests for raw-text coordinates, packing, encoded-cache policy, and missing auxiliary adapters.
- [x] Add tests proving ordinary one-sequence Stage-1 mixins are not composed for the variant.
- [x] Add collator/batch-extra tests proving set-continuation metadata survives collation and is stripped before model forward.
- [x] Add unit tests for fragment serialization, object-index spans, duplicate entries, and token-span boundaries.
- [x] Add unit tests for structural close-start and close-sequence token spans.
- [x] Add unit tests for subset sampling ratios, deterministic seeded behavior, small-object-count fallback, and random/preserved prefix order modes.
- [x] Add unit tests for candidate subsampling, exact mode, raw/corrected logZ, and PEM calibration.
- [x] Add unit tests for MP logsumexp, PEM threshold-loss total loss, close-start suppression, and weak schema close.
- [x] Add branch-local auxiliary adapter tests and missing-state failure tests.
- [x] Add a tiny trainer smoke test with one image and at least two objects.
- [x] Add config-resolution tests for the production entry config and its all-feature contract.
- [x] Run the focused pytest command listed in `docs/superpowers/plans/2026-04-25-stage1-set-continuation-training.md`.

## 10. Train-Step Forward Runtime Stabilization

- [x] Refine the OpenSpec design for the production OOM class as a
      train-forward runtime issue, scoped away from eval decoding.
- [x] Add strict config schema for
      `custom.stage1_set_continuation.train_forward`, including branch runtime,
      checkpoint settings, budget policy, fallback policy, prefix reuse, and
      telemetry toggles. Omitted `train_forward` must preserve retained-graph
      scoring, disabled fallback, disabled prefix encoding cache, and disabled
      GPU KV cache.
- [x] Add strict config schema for exact suffix-logit scoring and DDP candidate
      padding policy:
      `train_forward.logits.mode={full,supervised_suffix}` and
      `train_forward.ddp_sync.candidate_padding={max_count,none}`. Omitted
      values must preserve full logits and max-count padding.
- [ ] Extract train-forward planning seams:
      sample/candidate execution plan, branch budget policy, branch scorer, MP
      aggregator, and telemetry projection.
- [x] Preserve retained-graph behavior as a legacy/debug branch runtime mode for
      tiny-fixture parity tests.
- [x] Implement `checkpointed_exact` branch scoring so exact MP trades more
      recompute time for lower peak activation/logit retention while keeping the
      outer HF Trainer loss/backward contract. The immediate bridge must reject
      enabled branch-local aux losses unless all aux-bearing atoms are produced
      by the same differentiable checkpointed branch computation.
- [x] Prove `checkpointed_exact` loss and gradient parity against retained-graph
      mode on deterministic tiny fixtures, including coord-token candidate
      positions, DDP padding-branch behavior, and aux-disabled setup
      validation.
- [ ] Add deterministic budget-triggered approximate fallback to uniform
      candidate subsampling when authored candidate/token/memory budgets would
      be exceeded.
- [ ] Prove fallback never reports exact fidelity: metrics and artifacts must
      include remaining candidate count, scored candidate count, fallback reason,
      logZ estimator, authored candidate scoring mode, and objective-fidelity
      label. Authored `candidates.mode=uniform_subsample` must remain
      approximate even when the budget policy does not fire.
- [ ] Add exact in-sample prefix encoding cache and parity tests for `input_ids`,
      labels, coord masks, image tensors, branch spans, and structural-close
      spans.
- [x] Keep GPU KV prefix caching disabled in the immediate bridge; document any
      future detached KV cache as approximate and any future branch-mask runtime
      as a separate exact-efficiency change.
- [x] Implement exact supervised-suffix logits for candidate MP/PEM and
      structural-close losses, including full-logit parity tests for mixed
      coord/non-coord labels and close losses.
- [x] Implement no-padding DDP candidate synchronization as a config-first
      runtime policy while preserving max-count padding as a rollback mode and
      logging local/max/policy/padding-forward telemetry.
- [ ] Emit per-rank train-forward telemetry for memory, branch runtime mode,
      branch token budgets, exact/approx sample counts, fallback reasons, and
      prefix encoding cache hits/misses, with enough signal to distinguish
      retained-graph savings from unchanged per-forward peak allocations such
      as the coord-offset logits hook.
- [x] Update production set-continuation config to prefer
      smart-batched exact supervised-suffix logits, no DDP candidate padding,
      `training.ddp_find_unused_parameters=false`,
      `training.ddp_broadcast_buffers=false`, and authored cap-8 approximate
      fallback thresholds.
- [x] Add or update smoke coverage that exercises the production lifecycle with
      the new train-forward runtime policy rather than a drifted mini-only path.

## 11. Smart Candidate-Branch Batching Throughput Bridge

- [x] Refine OpenSpec design/specs for `smart_batched_exact`: an exact
      selected-candidate runtime that borrows ms-swift-style dynamic
      constant-volume scheduling for runtime-built MP candidate branches while
      keeping dataset-level packing and true padding-free Qwen3-VL branch
      packing out of scope.
- [x] Add strict config schema for
      `custom.stage1_set_continuation.train_forward.branch_batching`, including
      scheduler strategy, row/token caps, min-fill target, and padding-waste
      warning threshold.
- [x] Add scheduler/unit tests proving branch work items are length-bucketed,
      grouped with a constant-volume policy when possible, and fall back to a
      deterministic scheduler if `binpacking` is unavailable.
- [x] Add batched candidate scorer parity tests proving
      `smart_batched_exact` scores match retained-graph serial scoring for
      mixed coord/non-coord candidate labels, heterogeneous suffix lengths, and
      supervised-suffix logits.
- [x] Implement `smart_batched_exact` candidate scoring as a batched padded-row
      runtime for selected candidates only. Close-start and close-sequence losses
      remain on the existing close branch path in the first bridge.
- [x] Add trainer-level parity tests proving MP/PEM loss and logZ metrics match
      retained-graph mode on deterministic tiny microbatches.
- [x] Emit smart-branch-batching telemetry: scheduler code, branch-batch count,
      rows mean/max, token volume mean/max, padding fraction, and smart-batched
      branch-forward count.
- [x] Update production config and benchmark runtime labels to
      `smart_batched_exact_suffix_no_ddp_padding_cap8_v1`, with config-only
      rollback to retained-graph supervised-suffix mode.
- [x] Run targeted unit/config/runtime tests for the smart-batched bridge.
- [x] Run a real 2-GPU smoke before relaunching production.

## 12. Lightweight Bidirectional Token-Type Gate

- [x] Extend the active OpenSpec design/specs with a Stage-1
      set-continuation-native bidirectional token gate. The gate must be scoped
      to `objective_label_mask`, not `candidate_object_label_mask`, and must not
      re-enable ordinary one-sequence Stage-1 loss mixins.
- [x] Add strict config schema for
      `custom.stage1_set_continuation.bidirectional_token_gate`, including
      `enabled`, `coord_gate_weight`, `text_gate_weight`, `temperature`, and
      `scope=objective_tokens`.
- [x] Add red unit tests for token-type assignment using synthetic and real
      tokenizer/chat-template branches: coord labels map to coord gate,
      schema/description/boundary labels map to text gate, prefix labels do not
      contribute, and special stop tokens do not contribute.
- [x] Add loss-mask alignment tests proving next-token shifted logits are used
      for both gate terms and that supervised-suffix cropping preserves the same
      loss as full logits.
- [x] Add vocabulary-scope tests proving the coord vocabulary is exactly the
      configured `<|coord_0|>` through `<|coord_999|>` id set and that out-of-vocab
      or duplicate coord ids fail fast.
- [x] Add gate-math tests proving high non-coord mass at coord slots increases
      `loss/coord_gate`, and high coord mass at schema/description slots
      increases `loss/text_gate`.
- [x] Add runtime parity tests proving retained-graph serial scoring and
      `smart_batched_exact` scoring produce the same candidate scores, gate
      losses, token counts, and gradients on deterministic tiny fixtures.
- [x] Emit compact gate metrics: `loss/coord_gate`, `loss/text_gate`,
      `gate/coord_slot_coord_mass_mean`, `gate/text_slot_coord_mass_mean`,
      `gate/coord_tokens_count`, and `gate/text_tokens_count`.
- [x] Update metric-key parity tests so ordinary Stage-1 runs do not emit gate
      keys, and set-continuation gate-enabled runs do emit the compact gate
      keys.
- [x] Update `docs/training/STAGE1_OBJECTIVE.md`,
      `docs/training/METRICS.md`, and the super-power design/plan artifacts
      with the mathematical definition, validation checklist, and smoke gate.
- [x] Run a tiny real-data smoke with real tokenizer/chat template before any
      production relaunch. Acceptance is procedural: parse-valid hygiene remains
      healthy, gate losses are finite, coord-slot coord mass is high or moves in
      the intended direction, text-slot coord mass is low or decreases, and no
      special stop tokens contribute to the gate.
