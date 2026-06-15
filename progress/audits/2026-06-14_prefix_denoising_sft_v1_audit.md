# Prefix-Denoising SFT V1 — Read-Only Design/Plan Audit

- **Date:** 2026-06-14
- **Mode:** change/spec audit + implementation-vs-contract audit (read-only)
- **Scope:** `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/` (branch `codex/prefix-denoising-sft`)
- **Artifacts reviewed:**
  - `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md` (the plan)
  - `progress/directions/prefix_denoising_sft_v1.md` (the direction note)
  - `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md` (the design spec)
- **Code truth checked against (this worktree):** `src/config/schema.py`, `src/datasets/geometry.py`,
  `src/detection/packing.py`, `src/detection/runtime.py`, `src/detection/dataset.py`,
  `src/training_runtime/plan.py`, `src/data_collators/enrichers.py`,
  `src/data_collators/batch_extras_collator.py`, `src/metrics/events.py`,
  `src/trainers/metrics/teacher_forcing.py`, `src/trainers/teacher_forcing/forwards.py`,
  `src/bootstrap/trainer_setup.py`, `src/tokens/coord/codec.py`.
- **Safety:** No files mutated. Worktree on `codex/prefix-denoising-sft`. No long jobs run; only read-only `rg`/`grep`/Serena symbol reads.

---

## Verdict

**Hold: blocking issues remain.**

The *research design* is technically coherent and free of design-level P0s: clean-full teacher / noisy-full
student matches the stated exposure-bias objective; CE and KL spans/sites are correct; causal alignment
(`labels[p] -> logits[p-1]`), asymmetric `stopgrad(clean) -> noisy` KL, GT-centered clipped local support,
and bin→coord-token-id mapping are all conceptually right and match repo facts.

However, several **P1 implementation-correctness gaps in the plan's shown code/spec are blocking for a clean
implementation** and would become correctness P0s if coded verbatim:

1. New `prefix_denoising_*` sidecars are never registered in the fail-fast detection sidecar boundary, and the
   proposed loss mixin bypasses that boundary — leaking non-model keys into `model(**inputs)`.
2. The mixin's packing-state detection (`getattr(self, "_packing_enabled", lambda: False)()`) silently resolves
   to `False` on the Stage-1 trainer, disabling the packed position-id contract that provides segment isolation.
3. The single hardest, most novel component — packed multimodal materialization (4-row mRoPE `position_ids`,
   varlen boundaries, `pixel_values`/`image_grid_thw` flattening, boundary-map production seam) — is explicitly
   deferred ("extend the module ... after planner tests pass") and under-specified relative to its risk.

In parallel, the **OpenSpec-vs-experiment-only governance decision is genuinely unresolved and needs a user
decision** before schema/loss/metric/packing contracts are implemented.

None of the above invalidate the research idea; they are spec/plan tightening items plus one governance gate.

---

## Findings by severity

### P0 — none at the research-design level

No finding invalidates the research meaning, loss correctness, or reproducibility of the *design as written*.
See "Confirmed OK" for the design properties that were verified sound. The items below are the blocking set;
P1.1–P1.3 are flagged as "P0-in-effect if implemented from the plan's literal code."

---

### P1 — substantial risk to supported workflow / correctness / contracts

#### P1.1 — New prefix-denoising sidecars are not registered at the fail-fast model-input boundary, and the loss mixin bypasses that boundary (sidecar leak)

- **Evidence:**
  - `src/detection/dataset.py:1111` `strip_non_model_detection_sidecars` raises `ValueError` for **any**
    batch key not in `REGISTERED_DETECTION_SIDECAR_KEYS | DETECTION_DROPPED_BEFORE_MODEL_KEYS |
    DETECTION_MODEL_INPUT_KEYS | TRAINER_BATCH_EXTRA_KEYS` ("Unregistered detection batch extras at model-input
    stripping boundary").
  - Plan introduces `prefix_denoising_hybrid`, `prefix_denoising_segment_meta`, and `packed_hybrid_boundary_map`
    (plan lines ~1143, ~1296, ~830–845) but the Planned File Map (plan lines 49–83) never edits the sidecar
    registry in `src/detection/dataset.py`.
  - Proposed `PrefixDenoisingObjectiveMixin.compute_loss` (plan lines ~1689–1740) uses a private
    `ignored_keys = ["labels", "prefix_denoising_segment_meta", "prefix_denoising_hybrid"]` and **does not** call
    `strip_non_model_detection_sidecars`, unlike the existing `TeacherForcingObjectiveMixin.compute_loss`
    (`src/trainers/metrics/teacher_forcing.py:18` calls `strip_non_model_detection_sidecars(inputs)` +
    `maybe_pop_and_stash_batch_extras`).
  - `DatasetMetaEnricher` (`src/data_collators/enrichers.py:42`) **unconditionally** adds `dataset_labels`,
    `dataset_segments`, and a `pack_num_samples` tensor to every collated batch; `RecursiveDetectionTargetsEnricher`
    and `TeacherForcingTargetIREnricher` are also instantiated unconditionally
    (`src/data_collators/batch_extras_collator.py:80–101`).
- **Impact:** Two-sided failure. (a) Because the mixin's `ignored_keys` omits `dataset_labels`,
  `dataset_segments`, `pack_num_samples`, `sample_id`, `prepare_forward_inputs` keeps them in `inputs_for_model`
  (`forwards.py:16` only drops `ignored_keys`), so they are forwarded to `model(**inputs_for_model)` — an
  immediate forward crash or silent kwarg drop. (b) If any other path routes a prefix-denoising batch through
  `strip_non_model_detection_sidecars` (eval, the `hybrids is None` fallback into `super().compute_loss`, column
  filtering), the unregistered `prefix_denoising_*` keys hard-fail. The runtime integration test (plan Task 4
  Step 5) would catch (a) only if it exercises the *real* collator+model forward, not a stub.
- **Fix direction:** Register `prefix_denoising_hybrid`, `prefix_denoising_segment_meta`,
  `packed_hybrid_boundary_map` (and any KL sidecar) in `REGISTERED_DETECTION_SIDECAR_KEYS` /
  `DETECTION_DROPPED_BEFORE_MODEL_KEYS`; add this edit to the Planned File Map. Have the mixin reuse
  `strip_non_model_detection_sidecars(inputs)` (after popping its own extras) instead of a hand-maintained
  `ignored_keys` list, so the model-input boundary stays single-sourced and fail-fast.
- **Verification:** `tests/test_prefix_denoising_runtime_integration.py` must drive the *actual*
  `build_batch_extras_collator` output through `strip_non_model_detection_sidecars` and a real (or faithfully
  stubbed) Qwen forward, asserting no unregistered-key error and no extra kwargs reach `model.forward`.

#### P1.2 — Mixin packing-state detection defaults to `False` on the Stage-1 trainer (silent loss of the packed position contract)

- **Evidence:**
  - Proposed mixin: `packing_enabled=bool(getattr(self, "_packing_enabled", lambda: False)())`
    (plan line ~1715).
  - `_packing_enabled` is defined **only** on Stage-2 surfaces: `src/trainers/stage2_rollout_runtime.py:1817`
    and `src/trainers/rollout_correction/executors.py:350`. It is not provided by `compose_trainer_class`
    (`src/bootstrap/trainer_setup.py:62`, which composes `TeacherForcingObjectiveMixin` at line 113) nor by the
    base Stage-1 trainer.
  - `prepare_forward_inputs` only enforces the Qwen3-VL 4-row `position_ids` packing contract when
    `packing_enabled=True` (`src/trainers/teacher_forcing/forwards.py:37–47`, raises
    "packing enabled but missing Qwen3-VL 4-row position_ids metadata").
- **Impact:** For a packed prefix-denoising run, `getattr` falls back to `lambda: False`, so the mixin passes
  `packing_enabled=False`. The 4-row mRoPE position-id assertion is skipped and the forward is treated as
  unpacked. This is exactly the silent-fallback class the design warns against ("Do not rely on a plain 2D
  attention_mask ... it is insufficient" — direction note ~852). If the packer/collator did not actually inject
  correct reset position_ids, clean and noisy segments could attend across the boundary, leaking the clean answer
  into the noisy student and destroying the denoising signal — with no error raised.
- **Fix direction:** Resolve packing state explicitly from config/runtime (e.g., `training.packing` +
  `packing.static_packing`, or a value threaded from `compose_trainer_class`/`src/sft.py`), not a defensive
  `getattr` default. Fail fast if packing is configured but the batch lacks the 4-row `position_ids` /
  `PackedHybridBoundaryMap`.
- **Verification:** Add a test asserting that with `prefix_denoising.enabled` + `training.packing=true` the mixin
  calls `prepare_forward_inputs(..., packing_enabled=True)` and that a packed batch missing 4-row `position_ids`
  raises rather than silently running.

#### P1.3 — The packed multimodal materialization (the riskiest, most novel component) is deferred and under-specified

- **Evidence:**
  - Plan Task 7 Step 4 implements only the *assignment* planner (`build_hybrid_pack_plan`, plan lines ~2205–2232)
    and says: "Then extend the module with offset rewrite helpers for actual tensors after the planner tests
    pass" (plan line ~2235). The boundary-map dataclass lists `position_reset_offsets`, `varlen_cu_seqlens`,
    `pixel_values_slice`, `image_grid_thw_slice` (plan lines ~2142–2171) but no code produces them.
  - The model forward *requires* these: `prepare_forward_inputs` needs 4-row mRoPE `position_ids`
    (`forwards.py:39–47`); the runtime requires a flash-attn backend
    (`src/sft.py:2047` "requires model.attn_impl to be a flash-attn backend for padding-free packed training").
  - Static packing is **dataset-owned** in the default Stage-1 plan
    (`src/training_runtime/plan.py:67–68` `dataset_static_packing_allowed=True`,
    `dataset_static_packing_owner="dataset"`), so the *existing* path already produces correct packed
    `position_ids`/visual tensors. The direction note intends to "reuse the existing static packing cache ... but
    emit PackedHybridBoundaryMap directly" (direction ~877), yet the plan's Task 7 hand-rolls a fresh planner and
    leaves the tensor seam (whether it calls the ms-swift/template packing collator that already produces
    `position_ids`/`pixel_values`/varlen, or reimplements them) unspecified.
- **Impact:** The correctness of clean/noisy isolation, KL site row resolution, and visual ownership all depend
  on this materialization. Leaving it to "extend after planner tests pass" means the boundary-map unit tests can
  pass while the live packed forward is wrong (mis-aligned `position_ids`, wrong `pixel_values` slices,
  cross-segment attention). This is the highest residual technical risk and is currently the least specified.
- **Fix direction:** Before implementation, decide and document the materialization seam: strongly prefer
  *reusing* the existing dataset static-packing materializer (which already emits Qwen 4-row `position_ids`,
  varlen metadata, and per-sample `pixel_values`/`image_grid_thw`) and only layering `PackedHybridBoundaryMap` /
  sidecar offset rewriting on top. Treat clean_full and noisy_full as two ordinary packed sub-sequences (the
  existing path already isolates arbitrary samples in a packed row), so isolation is inherited rather than
  reinvented. Specify exactly which function builds `position_ids`/`cu_seqlens`/visual slices.
- **Verification:** A packed integration test that builds ≥2 hybrid samples in one physical row and asserts:
  per-segment `position_ids` reset to 0 at each segment start; varlen/cu_seqlens (when emitted) match segment
  boundaries; `pixel_values`/`image_grid_thw` slice owners match the boundary map; and a numeric probe that the
  noisy student logits are invariant to clean-segment coordinate token edits (proves no cross-segment attention).

---

### P2 — clarity / coverage / maintenance

#### P2.1 — Milder noise fallback pushes the wrong direction for "excessive skip rates"

- **Evidence:** Direction note ~466–469: "If tiny smoke evidence shows excessive skip rates, invalid-bbox
  warnings, a severe noisy-full CE gap, or token-accuracy collapse, the first named milder fallback is
  `center_shift_frac = 0.04` and `uniform_scale_range = [0.96, 1.04]`." The constructive noiser requires all four
  quantized coords to change (`construct_valid_norm1000_bbox_noise`, plan lines ~753–769; `4/4` rule, direction
  ~430–441).
- **Impact:** With the candidate-grid construction, a coordinate changes only when `round(width*shift) >= 1` or
  `round(width*scale) != width`, i.e. roughly `width >= 7` / `height >= 7` norm1000 bins (analysis below). A
  *smaller* envelope makes the 4/4-changed feasible set **smaller**, increasing skips. So the documented fallback
  fixes "noisy-full CE too hard / token-acc collapse" but **worsens** the "excessive skip rate" trigger it is
  also listed under. An operator following the note could increase skips while trying to reduce them.
- **Fix direction:** Split the two failure modes: for high skip rate, the corrective is *larger* shift/scale (or
  relaxing 4/4 to a documented weaker rule), not milder noise. Document opposite-direction remedies.
- **Verification:** Noising-difficulty report (plan Task 9 / direction ~981–985) should log skip rate vs noise
  strength on real data and confirm monotonicity before adopting any fallback.

#### P2.2 — Small/thin boxes are systematically 4/4-infeasible under norm1000 quantization; skip rate + enumeration cost unmeasured

- **Evidence:** `construct_valid_norm1000_bbox_noise` enumerates `dx in [-max_dx, max_dx]`,
  `dy in [-max_dy, max_dy]`, `sx, sy in {low, 1.0, high}` and keeps only valid + all-4-changed candidates
  (plan lines ~746–797). `max_dx = round(width * center_shift_frac)`.
- **Impact (two issues):**
  - *Skip / distribution shift:* For `width < 7` (norm1000 bins) at default 0.08/[0.92,1.08], no candidate
    changes both `x1` and `x2`, so the **entire hybrid sample is skipped** (design: skip whole sample if any
    object can't get 4/4, direction ~439–450). Datasets with many thin/small objects (common in detection;
    `rescale_32_1024_bbox_max60` does not preclude small boxes) could lose a non-trivial, *non-random* fraction
    of samples (bias toward large-object images) — a silent eval/train distribution shift if interpreted as
    full-data training.
  - *Throughput:* For large boxes the grid is `(2*max_dx+1)*(2*max_dy+1)*9` candidates (~10^5 for an 800-bin
    box), rebuilt every epoch because encoded-sample cache is disabled (design Locked Decisions; direction
    ~815–826). This runs on the dataloader hot path.
- **Fix direction:** Measure 4/4-infeasible skip rate per dataset *before* production and report it with scope
  labels; do not present skipped-subset runs as full-data validation. Consider sampling directly from the
  feasible set without full enumeration (or capping the grid) to bound cost; the plan already mandates "no
  reject/repair loop," which is compatible with direct constructive sampling.
- **Verification:** Add a geometry test over realistic small/thin boxes asserting the documented skip counters
  fire, and add the per-slot skip-rate line to the launch-health noising-difficulty report.

#### P2.3 — Plan's own contract tests do not match the plan's shown validator messages / strict-parse behavior

- **Evidence:**
  - `test_prefix_denoising_rejects_valid_set_marginal_teacher_forcing` expects
    `match=r"prefix_denoising.*hard clean-label CE"` (plan line ~357), but the shown validator raises
    `"prefix_denoising requires objective.id=teacher_forcing and objective.profile=hard_sft"`
    (plan line ~562–563) — no "hard clean-label CE" substring.
  - `test_prefix_denoising_rejects_coord_soft_ce` injects `objective["coord_soft_ce"] = {"enabled": True}` and
    expects `match=r"prefix_denoising.*SoftCE"` (plan lines ~361–368), but the shown validator never inspects a
    `coord_soft_ce` objective key; the strict objective parser (`parse_dataclass_strict`, used throughout
    `src/config/schema.py`) would instead raise an "Unknown ... keys" error from the objective dataclass, not the
    prefix-denoising SoftCE message.
- **Impact:** These red tests would fail for the wrong reason or never reach the intended assertion, giving false
  confidence that SoftCE/marginal profiles are specifically rejected by the prefix-denoising contract.
- **Fix direction:** Align validator messages with the test regexes (include "hard clean-label CE" and an
  explicit SoftCE rejection), or change the tests to assert on the actual messages; ensure the SoftCE case is
  rejected by an *intentional* prefix-denoising check, not an incidental strict-parse error.
- **Verification:** Run `python -m pytest tests/test_prefix_denoising_config_contract.py -q` after implementing
  the validator; confirm each rejection test fails on the intended message.

#### P2.4 — `objective.target_ir.rollin_policy` becomes dead/contradictory config under prefix denoising (dual ordering surface)

- **Evidence:** The contract validator requires `data.object_ordering == "sorted"` (plan lines ~560–561), and the
  builder uses sorted clean-GT order and bypasses the teacher-forcing target-IR random-permutation runtime
  (plan line ~585). Yet the base test payload sets `objective.target_ir.rollin_policy = random_permutation`
  (plan lines ~256–258) and the contract does not reject or warn on it. `DetectionDataConfig.object_ordering` is
  `Literal["sorted","random_permutation"]` (`src/config/schema.py:3308`), a separate axis from
  `target_ir.rollin_policy`.
- **Impact:** A config can advertise `rollin_policy: random_permutation` while prefix denoising silently ignores
  it and uses sorted order — a stale-key-accepted-as-no-op footgun that could mislead a reader into thinking
  ordering is randomized.
- **Fix direction:** Either reject a non-trivial `target_ir.rollin_policy` when `prefix_denoising.enabled`, or
  document in the schema/README that prefix denoising ignores target-IR roll-in and that `data.object_ordering`
  is the sole ordering control.
- **Verification:** Add a contract test asserting the chosen behavior (reject, or documented ignore).

#### P2.5 — Cross-accumulation / DDP weighting of branch-balanced CE and mean-KL is unspecified

- **Evidence:** `compute_branch_balanced_hard_ce` returns a per-microbatch `0.5*clean_ce + 0.5*noisy_ce` (plan
  lines ~1513–1545); `compute_local_coord_kl` returns a per-microbatch mean over sites (plan line ~1949). The
  mixin returns this scalar to HF Trainer and ignores `num_items_in_batch` (plan line ~1690).
- **Impact:** Under gradient accumulation / DDP, a mean-of-per-microbatch-means is not a global token-weighted
  (CE) or site-weighted (KL) mean when counts differ across microbatches/ranks, slightly shifting the effective
  CE scale and KL weight. This is mitigated for CE because `clean_full` and `noisy_full` share identical labels
  and therefore identical supervised-token counts per sample (so per-sample clean_den == noisy_den), but
  cross-microbatch imbalance and KL site-count imbalance remain. This matches existing teacher-forcing behavior
  (the current mixin also ignores `num_items_in_batch`, `teacher_forcing.py:18`), so it is not a regression.
- **Fix direction:** Document the accounting explicitly; if exact global weighting matters for the KL ladder,
  reduce with global denominators (all-reduce site/token counts) rather than per-microbatch means.
- **Verification:** A two-microbatch test with unequal site counts asserting the documented aggregation.

#### P2.6 — Zero-object / negative images degenerate to duplicated branches with no denoising signal

- **Evidence:** The builder corrupts "every valid nondegenerate object bbox" (design ~205–208); KL sites come
  from objects. Nothing in the plan special-cases images with zero objects.
- **Impact:** A zero-object image yields `noisy_full == clean_full` (nothing to corrupt): doubled compute, zero
  denoising signal, empty KL sites, and `noisy_full` CE identical to `clean_full` CE — which can flatter the
  noisy-CE monitor. Not incorrect, but wasteful and potentially misleading on datasets with negatives.
- **Fix direction:** Decide explicitly (exclude with a counter, or keep and tag), and surface a counter so the
  noisy-CE monitor isn't silently diluted by no-op samples.
- **Verification:** Builder test on a zero-object row asserting the chosen behavior and counter.

#### P2.7 — Prefer the canonical coord-token-id helper over a bespoke `coord_token_ids` argument

- **Evidence:** KL maps bins→ids via a passed-in `coord_token_ids` tensor (plan lines ~1796–1820,1921–1930).
  A canonical helper already exists: `get_coord_token_ids(tokenizer)` returns ids for `<|coord_0|>..<|coord_999|>`
  (`src/tokens/coord/codec.py:69`) with `build_coord_token_id_mask` (`codec.py:105`); `EXPECTED_COORD_START_ID =
  151670` (`src/tokens/qwen_native.py`) matches the plan's `expected_start` (plan line 237).
- **Impact:** Re-deriving the coord-id row risks drift from the canonical 0..999 contract.
- **Fix direction:** Source `coord_token_ids` from `get_coord_token_ids(self._resolve_tokenizer())` in the mixin
  and pass it down; keep the loss helper tokenizer-agnostic.
- **Verification:** Assert in a test that the mixin-sourced ids equal `get_coord_token_ids(tokenizer)`.

---

### Non-blocking — useful but not required pre-implementation

- **Boundary-map insertion seam into `collated` is implicit.** The enricher's packed branch *checks* for
  `packed_hybrid_boundary_map` in `collated` and raises if absent (plan lines ~1271–1280), but no component is
  specified to *insert* it before the enricher runs. `build_batch_extras_collator` runs enrichers after
  `collate_fn(batch)` (`batch_extras_collator.py:86`). Name the producer (custom packing collator or template
  hook) explicitly. (Enricher signature `(*, collated, raw_batch, packed)` itself is fine — it matches
  `RecursiveDetectionTargetsEnricher`/`TeacherForcingTargetIREnricher`, `batch_extras_collator.py:93–101`.)
- **`import random` in `geometry.py`.** Not strictly required (the annotation `rng: random.Random` is a string
  under `from __future__ import annotations`, and the body only calls `rng.randrange`), but add it if any runtime
  reference to `random` is introduced.
- **Token-pooled-vs-balanced CE test.** Because clean/noisy share labels, denominators are equal in the normal
  path; keep the planned "unequal-denominator formula test" (plan ~942–944) so the distinction stays meaningful
  if a branch is ever dropped.

---

## Confirmed OK / ruled out

- **Research semantics match the objective.** noisy_full input = bbox-corrupted coords, labels = clean GT
  (design ~24–32) directly trains "drifted coordinate prefix -> clean continuation," the stated exposure-bias
  target. KL teacher=clean / student=noisy with asymmetric `stopgrad(clean)->noisy` (direction ~187–193) is a
  coherent prefix-insensitivity regularizer. The "teacher is not an oracle" risk is explicitly acknowledged with
  teacher-quality diagnostics (direction ~238–254).
- **CE/KL sites and causal alignment are correct.** CE supervises clean labels in both branches over the
  assistant response; KL applies only at the four coordinate sites `x1/y1/x2/y2` of selected objects
  (direction ~196–204). `labels[p] -> logits[p-1]` is implemented (`_segment_ce_sum`, plan ~1507; KL
  `clean_row = clean_label_position - 1`, plan ~1925) and the CE helper guards against supervising a segment's
  first physical token (`active_positions <= start` raises, plan ~1505), preventing cross-segment leakage in
  packed rows.
- **Asymmetric KL direction + stopgrad are right.** Teacher slice is `.detach()`ed; student is not
  (plan ~1933–1944); the planned test asserts `clean_logits.grad is None` and `noisy_logits.grad is not None`
  (plan ~1824–1826).
- **Local support semantics are right.** `support(c,r)=[max(0,c-r)..min(999,c+r)]` clips at edges, never wraps,
  includes GT, renormalizes in fp32 over the support; bins are mapped through coord-token ids before indexing
  full-vocab logits (design ~253–273; direction ~211–222). The K>1 = "more sites, not more segments, mean not
  sum" invariant is stated and tested (direction ~312–317; plan ~1831–1842).
- **`hard_sft` profile and config base exist.** `objective.profile` Literal includes `hard_sft`
  (`src/config/schema.py:135,3912–3916`) and `hard_sft` already enforces auxiliary modules disabled
  (`schema.py:3941–3973`). Keeping prefix denoising on `objective.id: teacher_forcing` + `profile: hard_sft`
  (not reviving `objective.id: sft`) is consistent with current schema.
- **Disabled-by-default is backward compatible.** Adding `prefix_denoising` to `_DETECTION_OPTIONAL_SECTIONS`
  (`schema.py:2908`) with `enabled=False` default and validation gated on `enabled` leaves existing configs
  unaffected.
- **Existing packing guards are real and correctly targeted.** `_validate_teacher_forcing_training_packing_contract`
  (`schema.py:2863`) and `_detection_validate_packing_runtime_contract` (`schema.py:3082`) reject `packing`,
  `static_packing`, and `padding_free_packed` for `teacher_forcing` and `recursive_detection_ce`; the plan's
  intent to add a *narrow positive* eligibility path while keeping non-V1 rejection intact is the right shape.
  Static packing is genuinely supported for the base detection objective and is dataset-owned
  (`src/training_runtime/plan.py:64–74`), so the reuse premise is valid.
- **Metric event API matches.** `weighted_mean_event(key, value, weight, *, unit, ..., metric_surface, channel,
  diagnostic_only)` stores `numerator=value*weight, denominator=weight` (`src/metrics/events.py:206`);
  `flatten_metric_events`/`reduce_metric_events` group by `event.key` and **raise on same-key/different-identity**
  (`events.py:354–391`), and `MetricIdentity` includes `channel`/`metric_surface` (`events.py:21–45`). The
  design's "put branch/view in the key, don't rely on channel-only differences" guidance is accurate, and the
  proposed distinct keys avoid collisions. `SwiftMetricReporter.update_many` exists (`src/metrics/reporter.py:196`).
- **Standard monitors preserved.** The mixin sets `flat["llm_loss"]` to the optimized scalar actually
  backpropagated (CE-only: balanced CE; KL-on: `CE_balanced + weight*KL_raw`) and emits global top-1/top-5
  (plan ~1726–1738; design ~286–296), satisfying the "preserve `llm_loss`/top-1/top-5" requirement.
- **Coordinate vocabulary is 1000 coord tokens with a canonical id helper** (`src/tokens/coord/codec.py:69–92`),
  so bin→id mapping for KL has a sound, existing basis.
- **`prepare_forward_inputs` exists with the signature the mixin uses** (`forwards.py:7`), and
  `compose_trainer_class` is the correct composition owner (`src/bootstrap/trainer_setup.py:62,113`).
- **Greenfield:** no existing `prefix_denoising`/`PrefixDenoising` references in `src/`, `openspec/`, or
  `docs/training/METRICS.md`; `docs/training/METRICS.md` and `configs/stage1/detection_teacher_forcing/{README.md,prod,smoke}`
  exist as the plan assumes.

---

## Unresolved questions (materially affect meaning / correctness / cost / governance)

1. **Governance gate (needs user decision).** Plan Task 1 Step 0 requires *either* an OpenSpec change for the
   compatibility-sensitive surface (schema, loss semantics, metric keys, packing eligibility, cache rules) *or*
   an explicit user decision that V1 stays branch-local experiment-only. No active prefix-denoising OpenSpec
   change exists (`openspec/changes/` holds only `detection-scene-clean-break`,
   `harden-unified-training-runtime-boundaries`, `archive`). This must be decided before schema/metric contracts
   land. Given `enabled=False` default and full backward compatibility, experiment-only is defensible — but it is
   a user call.
2. **Packed materialization seam (P1.3).** Does V1 reuse the existing dataset static-packing materializer for
   `position_ids`/varlen/`pixel_values`, or hand-roll them? This determines whether attention isolation is
   inherited (low risk) or reinvented (high risk).
3. **Skip-rate budget (P2.1/P2.2).** What 4/4-infeasible skip rate does the target dataset incur at the default
   noise envelope, and is that acceptable as "full-data" training or must it be labeled a filtered subset?

---

## Suggested next gate (before any implementation)

1. **Resolve governance:** user decides OpenSpec change vs experiment-only; record the decision in the plan.
2. **Patch the plan for P1.1–P1.3:** add the sidecar-registry edit to the file map and make the mixin reuse
   `strip_non_model_detection_sidecars`; replace the `_packing_enabled` `getattr` default with explicit packing
   resolution + fail-fast; specify the packed materialization seam (prefer reusing the existing static packer)
   and the boundary-map producer in the live collator.
3. **Reconcile the contract tests (P2.3)** with the validator messages and strict-parse behavior.
4. **Add a real packed integration probe** to the test plan proving segment isolation (noisy-student logits
   invariant to clean-segment coord edits), 4-row `position_ids` resets, and visual-slice ownership — the
   current unit tests can pass while the live packed forward is wrong.
5. **Run a noising-difficulty pre-measurement** (skip rate by box size, candidate-enumeration cost) on the real
   dataset before the production config, and fix the fallback-direction guidance (P2.1).

Only after 1–2 are addressed and 4 is added should implementation begin; the design itself needs no
research-level rework.
