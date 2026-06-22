# FN-Rescue Desc-X1 Evidence Routing Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next FN-rescue mechanism loop for checkpoint-3664: mine existing attention rows, test whether desc-end states already contain `x1` binding evidence, and run causal region interventions that distinguish useful target evidence from non-causal visual sink/background mass.

**Architecture:** Keep this as an analysis-only extension under `src/analysis/` and `configs/analysis/`, reading the completed FN-rescue full-linked artifacts as the primary evidence source. Phase 2 must not modify Qwen3-VL upstream model files, must not launch production training, and must avoid treating all non-GT/COCO-unlabeled pixels as bad background; suppression candidates are only "sink" tokens or regions after causal evidence shows they hurt `desc -> x1` binding.

**Tech Stack:** Python stdlib, PyYAML, pytest, existing CoordExp compact-full/FN-rescue helpers, PyTorch/Transformers for GPU-only probe/intervention stages, tmux launchers for sharded 8-card analysis.

---

## Scope

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Primary upstream evidence root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation
```

Expected upstream files:

```text
summary.json
merge_summary.json
rescue_generation_rows.jsonl
rescue_decision_context_rows.jsonl
rescue_candidate_region_rows.jsonl
rescue_attention_region_rows.jsonl
```

New artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2
```

Evidence scope label:

```text
val200_fn_rescue_desc_x1_phase2
```

## Execution Status

Current status as of 2026-06-02:

- Phase 1 attention mining is implemented and completed on the val200 FN-rescue artifacts.
- Phase 2 `x1` logit-lens probing is implemented and completed as a sharded smoke run over the Lane-D hidden-state surface.
- Phase 3 CPU intervention planning is implemented and materialized; GPU no-op parity plus target-mask sanity smoke is complete. Competitor and far-background interventions are still pending.
- No production training was launched. GPU use so far has been analysis-only sharded probing.

Next independent research spec:

```text
docs/superpowers/specs/2026-06-02-fn-rescue-attention-guided-causal-binding-design.md
```

That spec supersedes this plan for future work.  This plan remains the record
of the completed Phase-2 attention mining, x1 logit-lens smoke, intervention
planning, and target-mask sanity smoke.

## Research Guardrails

- Do not copy the LazyStrike/LaSt-ViT framing as a foreground/background classifier. This experiment is query-conditional and decoder-step conditional: `desc_end`, `<|box_start|>`, `pre_x1`, and coordinate continuation states are the objects of study.
- Do not globally punish "background" in COCO. COCO漏标严重；non-GT can include true unlabeled objects. Only call a region suppressible after intervention shows it is high-mass and low- or negative-contribution for target binding.
- Treat attention as diagnostic until causal tests confirm it. Attention mining creates hypotheses; intervention stages test them.
- Keep `same-desc IoU > 0.95` duplication as an outcome bucket only. This phase focuses on instance binding and evidence routing.
- Use 8 GPUs as analysis parallelism, not 8-card production training.

## Phase 1: Attention Head/Layer/Token Mining

**Purpose:** From existing `rescue_attention_region_rows.jsonl`, find heads/layers/roles where target, same-desc competitor, rollout prediction, wrong-control source, context ring, and far-background mass separate successful vs failed `desc -> x1` or `desc+x1 -> y1` continuations.

**Files:**

- Create: `src/analysis/autoreg_fn_rescue_desc_x1_phase2.py`
- Create: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase2.py`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase2/ckpt3664_val200.yaml`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase2/ckpt3664_val200_smoke.yaml`
- Create: `tests/test_autoreg_fn_rescue_desc_x1_phase2.py`

**Outputs:**

```text
fn_rescue_desc_x1_phase2/attention_mining/head_layer_region_summary.jsonl
fn_rescue_desc_x1_phase2/attention_mining/head_layer_rankings.json
fn_rescue_desc_x1_phase2/attention_mining/summary.json
fn_rescue_desc_x1_phase2/report.md
```

**Required metrics:**

- per `(tier, role, layer, head, region_kind)` normalized attention mass mean/count
- target-vs-same-desc-competitor margin
- target-vs-far-background margin
- success-vs-failure attention gaps for `primary_rescue_success`
- correct-`x1` vs wrong-`x1` contrast for `desc_x1` and `desc_x1_wrong_control`
- top suspect sink heads where `far_background` or wrong-control source is high and target margin is low

**Steps:**

- [x] Add a CPU test fixture with a few synthetic generation rows and attention rows.
- [x] Implement streaming JSONL aggregation that joins attention rows to generation outcomes by `(case_id, rescue_tier)`.
- [x] Write head/layer summary and ranking artifacts without loading the 17GB attention file into memory.
- [x] Run targeted tests and a smoke config with `max_attention_rows`.
- [x] Run the full mining stage, preferably in tmux if the 17GB stream is slow.

## Phase 2: Desc-End / Box-Start X1 Latent Probe

**Purpose:** Test whether the model has already formed a spatial commitment before emitting `x1`, or whether `x1` is the first strong binding seed.

**Preferred probe:** logit-lens and/or lightweight linear probe over hidden states at compact-full roles:

```text
desc_end
box_start
pre_x1
post_x1
post_y1
```

**Reusable existing surface:**

```text
src/analysis/autoreg_hidden_state_probe.py
scripts/analysis/run_autoreg_hidden_state_probe.py
configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml
```

**Phase-2-specific requirement:** If reusing Lane D hidden-state extraction, do not confuse Lane C selected cases with FN-rescue rescue tiers. The report must label the evidence source explicitly:

- `lane_d_x1_basin_cases` for existing Lane D reuse
- `fn_rescue_generation_cases` for any new FN-rescue-specific hidden-state replay

**Outputs:**

```text
fn_rescue_desc_x1_phase2/x1_probe/probe_rows.jsonl
fn_rescue_desc_x1_phase2/x1_probe/summary.json
fn_rescue_desc_x1_phase2/x1_probe/report.md
```

**Required metrics:**

- target `x1` logit/rank at each role
- same-desc competitor `x1` logit/rank when available
- target-vs-competitor x1 margin
- role-wise separability for success/failure buckets
- whether `box_start/pre_x1` is already spatially committed before any generated coordinate token

**Steps:**

- [x] Audit existing Lane D hidden-state probe output contract and tests.
- [x] Add or reuse logit-lens `x1` metrics at `desc_end`, `box_start`, and `pre_x1`.
- [x] Run a small GPU smoke with one shard/case limit.
- [ ] If smoke passes, run a broader 8-card sharded analysis in tmux beyond the current smoke scope.

## Phase 3: Causal Region Intervention

**Purpose:** Distinguish harmless attention reservoirs from causally harmful sink/background or same-desc competitor regions.

**Initial intervention type:** image-region occlusion and/or vision-token suppression at replay time. Prefer the smallest intervention that can be safely implemented without editing upstream HF model files.

**Candidate interventions:**

```text
target_gt_mask
same_desc_competitor_mask
wrong_control_source_region_mask
far_background_sink_mask
context_ring_mask
no_op_control
```

**Outputs:**

CPU plan outputs:

```text
fn_rescue_desc_x1_phase2/intervention_plan/selected_interventions.jsonl
fn_rescue_desc_x1_phase2/intervention_plan/summary.json
```

Future GPU intervention outputs:

```text
fn_rescue_desc_x1_phase2/intervention/intervention_rows.jsonl
fn_rescue_desc_x1_phase2/intervention/summary.json
fn_rescue_desc_x1_phase2/intervention/report.md
```

**Required metrics:**

- delta target `x1` logit and rank
- delta generated IoU for short continuation if generation is enabled
- target-vs-competitor binding flip rate
- invalid-parse / format-collapse count
- whether far-background sink suppression improves, hurts, or leaves unchanged `desc -> x1`

**Steps:**

- [x] Start with a CPU-tested intervention plan that records regions and expected comparisons without model execution.
- [x] Run no-op and target-mask smoke on a few cases to validate directionality: masking the target should reduce target binding if the probe is causal.
- [ ] Only after target-mask sanity passes, run competitor and far-background sink interventions.
- [ ] Promote intervention evidence into the final report only with explicit scope labels.

## Execution Order

1. Phase 1 full attention mining from completed FN-rescue artifacts.
2. Phase 2 smoke using existing hidden-state probe surface; add logit-lens metrics if missing.
3. Phase 3 intervention smoke with no-op and target-mask sanity.
4. Full 8-card sharded analysis only for stages whose smoke artifacts validate cleanly.

## Acceptance Criteria

- Targeted unit tests pass for the new phase-2 mining code.
- Existing FN-rescue, attention routing, and hidden-state probe tests still pass or any pre-existing failures are clearly labeled.
- Phase 1 produces `head_layer_rankings.json`, `summary.json`, and `report.md`.
- Phase 2 produces a role-wise `x1` probe summary or a documented blocker showing why current hidden-state artifacts are insufficient.
- Phase 3 produces at least a smoke intervention summary with no-op and target-mask sanity checks, or a documented blocker if upstream Qwen3-VL runtime cannot support the intervention without unsafe model edits.
- Final report separates attention correlation from causal intervention evidence.
