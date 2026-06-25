---
title: Ledger Auxiliary-Loss Implementation-vs-Contract Audit
date: 2026-06-24
branch: codex/ledger-auxiliary-loss
head_commit: 9e2b6a4f
mode: implementation-vs-contract audit + launch gate (smoke path) + smoke artifact audit
reviewer: read-only audit (Claude)
claim_scope: audit
---

# Ledger Auxiliary-Loss Audit — Implementation vs Contract + Launch Gate

**Scope label:** implementation audit **+** smoke-readiness **+** smoke artifact audit
(I inspected real on-disk preflight and baseline artifacts under
`temp/detection_teacher_forcing/output/`).

**Branch state:** `codex/ledger-auxiliary-loss` @ `9e2b6a4f`, clean, up to date with
`origin`. `git pull --ff-only` = "Already up to date".

**Test run:** `112 passed, 1 failed` across the 12 ledger/infer/batch test modules
(env `/root/miniconda3/envs/ms/bin/python`, `PYTHONPATH=<worktree>:/data/ms-swift`).

---

## Bottom Line

The **code** is well-engineered, strict, fail-fast, and faithful to the design
packet on nearly every contract point I checked: the `0..999` coord contract,
coverage/region-anchor target semantics, fp32 numerics, tie-aware AUC, the
config schema, the PEFT head resolution, the non-mutating inference adapter
filter, and the preflight artifact writer. The preflight ran clean on all 128
real COCO samples, and the baseline adapter-save run correctly **omits**
`coverage_ledger_head` from `modules_to_save`.

The **evidence** is the problem. The ledger arm itself has **never completed a
real run**. The same-forward Qwen capture, the `coverage_ledger_head` checkpoint
save, the ledger metric stream, and the saved-adapter reload are verified only
against **synthetic stubs** in unit tests — never against the real Qwen3-VL
model on real data. The only artifacts that exist are (a) the preflight (which
does not train) and (b) the **baseline** run (ledger disabled). So the hypothesis
this branch exists to test is not yet honestly tested end-to-end.

**Verdict — launch gate (smoke): `rerun gate`.** The implementation is gate-ready
in code; the smoke must actually be *run* (baseline + ledger on identical GPU
topology) and the resulting metric/checkpoint/reload artifacts inspected before
any smoke-readiness or baseline-vs-ledger comparison claim.
**Verdict — production: `hold`** (acknowledged missing objective terms + unproven
real-run path).

Treat all current evidence as `tiny`/`overfit-128`/`preflight`/`proxy`. None of it
is validation.

---

## Findings (severity-ranked)

### P1-1 — Ledger arm has never completed a real run; real-model capture unproven
- **Evidence:**
  - `temp/detection_teacher_forcing/output/coverage_ledger_closed_hard_sft_128/smoke-…-ledger/v0..v5` are **all empty** (failed/aborted launches 02:33–02:48, before the later "Fix coverage ledger convergence blockers" / preflight-identity / Swift-compat commits).
  - No `coverage_ledger_closed_hard_sft_128_adapter_save` directory exists (ledger adapter-save run was never launched).
  - The only completed training run is the **baseline** (`…_baseline_adapter_save/…/v0-20260624-161749`), where `objective.terms.coverage_ledger.enabled=false` — so the bridge's `coverage_ledger_enabled` branch (capture + loss + metrics + head save) was **never taken**.
  - `grep` for `teacher_forcing/ledger` / `coverage_ledger_auxiliary` across every `logging.jsonl` → **0 matches**. The ledger metric stream has never been emitted in a run.
  - The Qwen-capture unit tests use synthetic stubs (`tests/test_coverage_ledger_qwen_capture.py:14-150` — `_FakeLowerQwen`, `_FakeQwenForConditionalGeneration`, `_FakePeftConditionalWrapper`, `_FakeLoraFacade`), not the real `transformers` Qwen3-VL.
- **Impact:** The central same-forward capture (one `get_image_features` call, `last_hidden_state` extraction, `lm_head` logit parity), the ledger-head optimizer participation under PEFT, the `coverage_ledger_head` checkpoint save, and the saved-adapter reload are **un-exercised** on the real model. The "Fix … convergence blockers" commit is itself unverified by a run. This blocks any smoke-success or comparison claim.
- **Mitigating evidence (real Qwen3-VL structure matches the capture's assumptions):** `modeling_qwen3_vl.py:887` `Qwen3VLModel`, `:1138` `image_embeds, deepstack_image_embeds = self.get_image_features(...)` (tuple return — handled by `qwen_capture.py:188-189` `result[0]`), `:872` returns `last_hidden_state`, `:1236` `lm_head(last_hidden_state)`. So the structural assumptions are correct; only the runtime behavior (bf16 + flash-attn2 + grad-checkpointing + 8-way DDP) is unproven.
- **Fix direction:** Launch the ledger arm to completion on the **same** GPU topology as the baseline; confirm finite parity logits, exactly-one `get_image_features` call, a non-empty `teacher_forcing/ledger/*` stream, `coverage_ledger_head` present in the saved `adapter_config.json.modules_to_save`, and a successful inference reload via `prepare_adapter_checkpoint_for_inference`.
- **Verification:**
  ```bash
  # after a real ledger run completes:
  jq -r 'select(.["teacher_forcing/ledger/coverage_auc"]!=null)' <run>/logging.jsonl | head
  python -c "import json;print(json.load(open('<ledger-ckpt>/adapter_config.json'))['modules_to_save'])"
  # expect: ['token_embeddings_adapter','coverage_ledger_head']
  ```

### P2-1 — `gpus=8` in the handoff means **GPU id 8**, not 8 GPUs (launch-doc bug)
- **Evidence:** `scripts/train.sh:69` `GPU_DEVICES="${gpus:-…}"` → `:73` `CUDA_VISIBLE_DEVICES=8` → `:76-81` `NUM_GPUS` = count of comma-separated tokens = **1**; only `:70-71` `gpus=all` expands to `0,1,…,7`. Handoff `handoff-2026-06-24-ledger-adapter-smoke.md:238-244` recommends `gpus=8` for both runs.
- **Impact:** On a standard 8-GPU host (ids 0–7), `CUDA_VISIBLE_DEVICES=8` references a non-existent device → CUDA init failure. On a 9+ GPU host it silently runs a single GPU (id 8). Either way `gpus=8` ≠ "8 GPUs".
- **Fix direction:** In the handoff, use `gpus=<single-free-id>` (e.g. `gpus=0`) and state it is a device id; if multi-GPU is intended, use `gpus=all` or an explicit comma list — but note that conflicts with the single-sequence smoke recipe (see P2-2).
- **Verification:** `gpus=3 bash scripts/train.sh …` then check the launcher log line `GPUs: 3 (num=1)`.

### P2-2 — Baseline smoke ran on 8 ranks → effective batch 8, violating the pinned recipe
- **Evidence:** baseline run dir has `train_heartbeat.rank0..rank7.jsonl` (8 ranks); `checkpoint-128/trainer_state.json`: `train_batch_size=1`, `accum/grad_steps=1`, `num_train_epochs=16`, `global_step=128` (⇒ 16 opt-steps/epoch = 128 samples / (8 ranks × 1)). The smoke config sets `training.effective_batch_size: 1` and the design (`…design.md:949-955`) pins "effective batch size 1, one long unpadded physical sequence per forward."
- **Impact:** Global effective batch = 8, not 1. `effective_batch_size: 1` was silently not honored under 8-way DDP (it cannot be: `1/(1×8) < 1`). The per-forward contract (1 sample/rank) is preserved — `schema.py:3191-3201` enforces `per_device_train_batch_size==1`, and the ledger sidecar path requires exactly one sidecar (`loss_bridge.py:366`) — so this is an optimization-fidelity/repro issue, not a crash. But the existing baseline is **not** the canonical recipe and must not anchor a comparison unless the ledger arm runs on identical topology.
- **Fix direction:** Either run both arms on a single GPU (true effective batch 1) or accept effective batch = world_size and document it; make `effective_batch_size` fail-fast when it cannot be satisfied by the launch topology.
- **Verification:** `ls <run>/train_heartbeat.rank*.jsonl | wc -l` (expect 1 for the pinned recipe) and `jq .accum/grad_steps` in `trainer_state.json`.

### P2-3 — Metric-namespace incoherence: off-contract duplicate count events
- **Evidence:** `loss.py:346-381` `_diagnostic_count_events` emits `training/objectives/coverage_ledger/{object_count,coverage_state_count,coverage_pair_count,region_anchor_pair_count}`; `metrics.py:30-33` emits the **same** counts under the contract namespace `teacher_forcing/ledger/{…}`. `loss_bridge.py:220-226` concatenates **both** (`coverage_ledger_result.metric_events` *and* `coverage_ledger_metric_events(...)`), and `trainers/metrics/teacher_forcing.py:204-218` flattens all of them to the reporter.
- **Impact:** Four count metrics are emitted twice under two namespaces. The design metric table (`…design.md:615-636`) specifies **only** `teacher_forcing/ledger/*` and `teacher_forcing/loss/*`. The `training/objectives/coverage_ledger/*` keys are off-contract and will confuse dashboards/consumers and any "metric namespaces are coherent" check.
- **Fix direction:** Stop emitting `result.metric_events` from the bridge (or remove `_diagnostic_count_events` from `loss.py`), so the only ledger metric surface is `coverage_ledger_metric_events()`.
- **Verification:** `grep -c "training/objectives/coverage_ledger" <ledger-run>/logging.jsonl` should be 0 after the fix.

### P2-4 — Visual-region pixel mapping rounds to int pixels (design specified un-rounded floats)
- **Evidence:** `visual_regions.py:112` calls `norm1000_bbox_to_pixel_bbox` → `geometry.py:38-49` → `coord_utils.py:114-131 denorm_and_clamp` does `int(round(...))` + `clamp_and_round` (→ integer, clamped to `[0,W-1]`). The design (`…design.md:465-470`) specifies "Convert … to pixel floats **without early rounding**" using `x1/999*(W-1)`. The underlying formula in `ints_to_pixels_norm1000` (`coord_utils.py:82-94`) is exactly `v/999*(W-1)` ✓, so the only deviation is the rounding/clamping.
- **Impact:** At a cell boundary, rounding can shift `floor()/ceil()` cell selection by ±1 vs the float path (e.g. `x2_px=28.4` → ceil 2 floats vs round(28)→ceil 1). Effect on the minimal enclosing rectangle is small. **Importantly, the overlay (`artifacts.py:329-333,346-356`) uses the same helper**, so the debug gallery faithfully matches what the loss sees — this is a doc-vs-code discrepancy, not an overlay-vs-loss inconsistency.
- **Fix direction:** Either pool the float pixel endpoints before floor/ceil (match the design), or amend the design doc to record the integer-rounded behavior as canonical. Confirm a boundary unit test pins the chosen behavior.
- **Verification:** `tests/test_coverage_ledger_visual_regions.py` (exact-cell-boundary / straddle cases) — confirm they assert the intended rounding mode.

### P2-5 — One failing unit test (brittle swift monkeypatch), not a product defect
- **Evidence:** `tests/test_coverage_ledger_preflight_artifacts.py:281` `monkeypatch.setattr("swift.llm.get_model_tokenizer", …)` → `AttributeError: 'module' object at swift.llm has no attribute 'get_model_tokenizer'`. Fails during **test setup**, before any product code runs. `preflight.py:317-322` guards this import with a `try/except ImportError` fallback (`swift.model.get_model_processor`), and the **real preflight succeeded** (16 overlays rendered from real images), proving the template-builder path works.
- **Impact:** A red test in the ledger suite that the handoff did not surface (the handoff only ran `head_install` + `infer_checkpoint_resolution`). Masks regressions in that test going forward.
- **Fix direction:** Patch the symbol where it is bound (`src.training.coverage_ledger.preflight.get_model_processor`) or import-then-patch, rather than `swift.llm.get_model_tokenizer` on a namespace package.
- **Verification:** `pytest tests/test_coverage_ledger_preflight_artifacts.py -q` → all pass.

### P2-6 — Production-readiness boundaries (acknowledged, but must stay explicit)
- **Evidence:** Smoke configs force `token_type_mass.enabled: false` and `continuation_margin.enabled: false`; no `geometry_valid_tail` term exists; handoff `:129-146` states the stable weighted four-family `token_type_mass`, `geometry_valid_tail`, and ledger continuation are **not** in this branch. Schema hard-limits `per_device_train_batch_size==1` for the ledger path (`schema.py:3191-3201`); the bridge requires exactly one sidecar per forward (`loss_bridge.py:366`). No `ledger/smoke_interpretation.md` exists (no ledger training run).
- **Impact:** The branch **cannot** honestly claim `ledger + mandatory type + geometry + continuation`. Larger-batch production needs ragged multi-sidecar support. These are correctly out of v0 scope but must not be elided in any forward claim.
- **Fix direction:** Keep claims scoped to "ledger-only auxiliary on closed compact, per_device=1, overfit-128 smoke" until the missing terms and multi-sample batching are implemented and tested.
- **Verification:** resolved-config diff of any production proposal vs this smoke; assert no claim references unimplemented terms.

---

## Confirmed OK / Ruled Out

- **Coord `0..999` contract enforced end to end.** `codec.py:17 value_in_coord_range = 0..999`; `geometry.py:10 MAX_BIN=999` + `validate_norm1000_bbox_xyxy` requires `0≤x1<x2≤999`; sidecar builder `sidecar_builder.py:93-117` rejects non-canonical/`>999` coord tokens (so `<|coord_1000|>` and bbox `>999` are rejected); `visual_regions`, `sidecars`, `artifacts` all import the single `geometry.py` source. Inference contract `checkpoints.py:419-422` validates `<|coord_0|>..<|coord_999|>`. Preflight: 1032 objects across 128 samples, **0** out-of-range/degenerate.
- **Object-span & target semantics correct.** Coverage target matrix (`loss.py:119-128`): prompt-end row all-0; after object k's `box_end`, objects `0..k`=1 (covered=positive), `>k`=0 (negative). Region-anchor (`loss.py:235-248`): current-row object only is positive (`softplus(-logit)` = positive-only BCE), all others masked — matches design ("not a one-vs-all classifier"). Coverage uses `box_end_position`; anchor uses `box_start_position`; `object_ref_end_position` is captured for validation. Sidecar builder enforces exactly one `object_ref_end` + one `box_end` per object and exactly four coord positions; `emitted_order_index` must be `0..N-1` (`sidecars.py:223-235`).
- **Sorted roll-in / sorted ordering consistent.** Both smoke configs set `sample_factory.target_sequence.object_ordering: sorted` and `objective.target_ir.rollin_policy.name: sorted`; the preflight dataset reads the same config keys (`preflight.py:281,295-297`), so preflight and smoke training share ordering.
- **fp32 numerics & guards.** `loss.py:180` `DEFAULT_PRECISION_POLICY.context`, `:287-289 _linear_float32` (`.float()`), L2-normalize with `eps`, temperature divide, BCE-with-logits + `pos_weight` on coverage only, `softplus` for anchor; `_raise_non_finite` on logits/losses/weighted. Empty/edge handling: AUC omitted for single-class (`metrics.py:230`), weighted-loss event only when pairs>0 (`metrics.py:51`).
- **AUC & accuracy.** `metrics.py:222-254` exact tie-aware rank AUC (0.5 credit on ties), no sklearn; accuracy threshold logit `0.0` (`metrics.py:262`). All ledger diagnostics `diagnostic_only=True` except `teacher_forcing/loss/coverage_ledger_auxiliary_weighted` (`diagnostic_only=False`) — matches design.
- **PEFT head resolution picks the ACTIVE head, not the frozen original.** `loss_bridge.py:402-425 _resolve_coverage_ledger_head_module` resolves `modules_to_save[active_adapter]` under `ModulesToSaveWrapper` first, falling back to `original_module` only if no active match. `sft.py:4129 _require_wrapped_coverage_ledger_head_for_training` asserts the three head projections are `requires_grad` after `prepare_model`.
- **modules_to_save save/drop is correct (real-artifact evidence).** Baseline `checkpoint-128/adapter_config.json` → `modules_to_save=['token_embeddings_adapter']` with **no** `coverage_ledger_head`; `adapter_model.safetensors` has `token_embeddings_adapter.*` tensors and **no** `coverage_ledger_head.*`. `sft.py:3119-3127` appends/removes `coverage_ledger_head` per enable state; `:4125 _remove_missing_peft_module_to_save` cleans stale PEFT entries.
- **Inference adapter prep: non-mutating + drops only the training-only head.** `checkpoints.py:31 TRAINING_ONLY_MODULES_TO_DROP_FOR_INFERENCE=('coverage_ledger_head',)`; `:130-209 prepare_adapter_checkpoint_for_inference` copies to a tempdir (ignoring original `adapter_config.json`/`adapter_model.safetensors`), filters out only `coverage_ledger_head` tensors/`modules_to_save`, and returns a no-op view when nothing is dropped — the source checkpoint is never mutated; `token_embeddings_adapter` is preserved.
- **Schema is strict and matches the design.** `schema.py:4047-4140` enforces `temperature≥0.05`, `normalize_eps≥1e-8`, `pos_weight>0`, positive `ledger_projection_dim`, and **exact** `overlay_sample_count==16`, `smoke_sample_count==128`, `smoke_sample_seed==20260623`. `:3150-3201` rejects `compact`/`compact_full`/non-closed templates, all packing modes (`packing`, `eval_packing`, `static_packing`, `padding_free_packed`), and requires `per_device_train_batch_size==1`.
- **Preflight artifact contract fully met (smoke artifact audit).** `…_preflight_20260624T161509Z/ledger/`: `selected_samples.json` (schema `coverage_ledger_preflight_artifacts_v0`, `selection_seed=20260623`, `source_jsonl_sha256=649fb8c1…`, `processor_do_resize=False`, `template_id=compact_object_box_closed`, 128 row indices + sample ids + per-sample grid/dims); `alignment_debug.jsonl` = **128 lines, all `failure_status: ok`**; `overlays/` = **16 PNGs** + `index.json` (16 entries, paths/ids/order). Writer (`artifacts.py:67-141`) enforces exact 128/16 counts and `_reject_existing_nonempty_tree` (`:375-383`) fails fast on a stale `ledger/`/`overlays/` tree (no merge). Overlays render GT bbox + post-merge grid + selected rectangle (`:285-343`). Baseline/ledger config-diff allowlist enforced at preflight (`preflight.py:45-65,190-204`).
- **Data resolution: no worktree-vs-canonical divergence.** Worktree and canonical `public_data/.../train.coord.jsonl` are the **same inode** `125731763`, sha256 `649fb8c1…` (matches the preflight digest); baseline `train_data_provenance.json` resolved to the same canonical path. `model_cache` is a symlink to `/data/CoordExp/model_cache`; the model dir resolves. `model_type: qwen3_vl` is explicit and unambiguous.
- **Canonical run artifacts present for the completed (baseline) run:** `resolved_config.json`, `effective_runtime.json`, `experiment_manifest.json`, `run_metadata.json`, `pipeline_manifest.json`, `train_data_provenance.json`, `runtime_env.json`, plus `checkpoint-128/` with `trainer_state.json`.
- **ObjectiveRunner boundary respected.** Coverage ledger is added bridge-locally (`loss_bridge.py:219 loss = loss + coverage_ledger_result.weighted_loss`); it is not registered as a runner objective; the runner-owned CE/objective loss and its metric events are preserved and only appended to.

---

## Open Questions (block a fully reliable conclusion)

1. Does the **real** Qwen3-VL same-forward capture produce finite logits within tolerance of the normal forward, with exactly one `get_image_features` call, under bf16 + `flash_attention_2` + `gradient_checkpointing` + 8-way DDP? (Structural assumptions match `modeling_qwen3_vl.py`, but no real run has confirmed runtime behavior.)
2. Does the Qwen3-VL **deepstack** path (`image_embeds, deepstack_image_embeds = get_image_features(...)`) preserve the `image_embeds[i] ↔ placeholder offset start+i` correspondence the region pooling assumes? The design's same-forward parity probe should assert this; it has not been run.
3. Will the "convergence blockers" fix actually yield a decreasing coverage/anchor loss and interpretable AUC on the 128-sample overfit? Unverified (no ledger run).

---

## Suggested Next Actions (for the implementer)

1. Fix the handoff `gpus=8` recipe (P2-1) and decide single-GPU vs documented multi-GPU effective batch (P2-2) **before** relaunch.
2. Launch the **ledger** adapter-save arm to completion on the **same** topology as the baseline; capture the `teacher_forcing/ledger/*` stream and a saved `coverage_ledger_head`.
3. Reload the saved ledger adapter through `prepare_adapter_checkpoint_for_inference` and confirm it drops only `coverage_ledger_head`, preserves `token_embeddings_adapter`, leaves the source dir intact, and decodes on the same 128 rows.
4. Remove the off-contract `training/objectives/coverage_ledger/*` duplicate metrics (P2-3).
5. Fix the brittle preflight Swift monkeypatch test (P2-5).
6. Decide and pin the pixel-rounding behavior (P2-4) in code + design.
7. Keep all claims scoped: ledger-only, closed compact, per_device=1, overfit-128 — no `type+geometry+continuation` claim (P2-6).

---

## Verification Commands Used

```bash
# branch state
git -C <wt> fetch origin && git -C <wt> pull --ff-only && git -C <wt> log -5 --oneline

# targeted tests (112 passed, 1 failed)
PYTHONPATH=<wt>:/data/ms-swift /root/miniconda3/envs/ms/bin/python -m pytest \
  tests/test_coverage_ledger_*.py tests/test_final_checkpoint_coverage_ledger.py \
  tests/test_infer_checkpoint_resolution.py tests/test_train_batch_contract.py -q

# preflight artifacts (all 128 ok, 0 bad geometry, seed 20260623, sha 649fb8c1…)
#   <PF>/ledger/{selected_samples.json, alignment_debug.jsonl(128 lines), overlays/(16 png + index.json)}

# baseline checkpoint modules_to_save == ['token_embeddings_adapter'] (no coverage_ledger_head)
#   <baseline>/checkpoint-128/adapter_config.json + adapter_model.safetensors

# launch arithmetic: gpus=8 -> CUDA_VISIBLE_DEVICES=8, NUM_GPUS=1 (scripts/train.sh:69-81)
# data identity: same inode 125731763 for worktree vs /data/CoordExp public_data train.coord.jsonl
```
