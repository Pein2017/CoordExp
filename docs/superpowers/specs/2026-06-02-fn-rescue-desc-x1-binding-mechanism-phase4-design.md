# FN-Rescue Desc-X1 Binding Mechanism Phase 4 Design

Status: draft executable research spec

Date: 2026-06-02

Scope root:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes
```

## Goal

Convert Phase-3 region-causal evidence into a sharper mechanism diagnosis for
the desc-first autoregressive path:

```text
desc -> x1 -> y1/x2/y2
```

The central question is whether the model commits to an instance before `x1`,
at `x1`, or only after later coordinate tokens.

## Starting Evidence

Phase-3 full root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding
```

Key full-run findings:

- `target_gt_mask` is strongly causal for desc+x1 binding:
  `28/50` success flips, mean target-IoU delta about `-0.380324`.
- Same-description competitor/source masks are not clean rescue controls; they
  often reduce target binding too.
- `wrong_control_source_region_mask` is mostly metric-neutral for target
  success but not generation-neutral.
- `sink_triage` is candidate-only and must not be used as a background
  suppression signal yet.

## Phase-4A: Instance-Level Attention Binding Table

Purpose:

Use existing FN-rescue `rescue_attention_region_rows.jsonl` rows with
`aggregation_scope=instance` to link Phase-3 intervention outcomes to
instance-specific attention evidence.

Inputs:

```text
fn_rescue_continuation/rescue_attention_region_rows.jsonl
fn_rescue_desc_x1_phase3_causal_binding/case_linked/case_mechanism_rows.jsonl
```

Output:

```text
fn_rescue_desc_x1_phase4_binding_mechanism/instance_attention_binding/rows.jsonl
fn_rescue_desc_x1_phase4_binding_mechanism/instance_attention_binding/summary.json
fn_rescue_desc_x1_phase4_binding_mechanism/instance_attention_binding/report.md
```

Each output row must include:

- `case_id`, `rescue_tier`, `intervention_kind`, `mechanism_bucket`;
- target, same-desc competitor, rollout prediction, wrong-source, context-ring,
  and far-background attention masses;
- `target_minus_competitor_instance_attention`;
- `top_instance_attention_heads`;
- the Phase-3 causal outcome fields needed to compare against attention:
  `target_iou_delta`, `primary_success_changed`, `valid_paired_row`.

Interpretation:

This table can test whether `target_dependent` rows have higher target-instance
attention than competitor-instance attention, and whether
`competitor_dependent` rows reverse that pattern.  It is still attention
correlation plus image-region causal outcome linkage, not head-level causal
proof.

## Phase-4B: Desc-X1 Hidden/Logit Binding Linkage

Purpose:

Link existing or newly generated Lane-D hidden/logit probe rows to Phase-3 cases
so that desc/x1 binding can be inspected across compact roles:

```text
desc_end
box_start
pre_x1
post_x1
pre_y1
```

Initial output:

```text
fn_rescue_desc_x1_phase4_binding_mechanism/desc_x1_probe_linkage/summary.json
fn_rescue_desc_x1_phase4_binding_mechanism/desc_x1_probe_linkage/report.md
```

The first implementation should not force a new hidden-state GPU run if no
compatible Lane-D artifact exists.  It should instead produce a manifest-level
linkage report that records:

- whether Lane-D `probe_rows.jsonl` exists;
- which roles and layer groups are available;
- how many Phase-3 cases can be joined;
- which missing fields block hidden/logit claims.

If compatible probe rows exist, the linkage table should report x1 target-rank
or x1 margin by role and mechanism bucket.  If not, the report must say that
Phase-4B remains blocked on a Lane-D FN-rescue-aligned probe run.

## Phase-4C: Context Disentanglement

This is a follow-up, not the first Phase-4 implementation.  It should split
object pixels from context rings and area-matched controls only after Phase-4A
has identified which instance-attention patterns are actually associated with
Phase-3 outcome buckets.

## Phase-4D: Sink Candidate-to-Causal Narrowing

This is a follow-up.  It should select a small, strict subset from
`sink_triage/sink_candidate_rows.jsonl` and compare high-attention background
patch masks against low-attention and area-matched controls.  No background
suppression training signal is allowed before this causal narrowing step passes.

## GPU Policy

Use GPUs as analysis workers, not production training.

Phase-4A is CPU-only because the required instance attention rows already
exist.  Phase-4B may use GPUs only if running a new Lane-D hidden/logit probe.
If that run is needed, it should use shard-local outputs and merge validation,
following the existing Lane-D shard contract.

## Acceptance Criteria

- Phase-4A writes instance-level attention binding rows, summary, and report.
- Phase-4A reports bucket-level target-vs-competitor attention margins.
- Phase-4A keeps `valid_paired_row` gating visible and does not interpret
  invalid rows as mechanism evidence.
- Phase-4B writes a linkage report even when hidden/logit artifacts are absent.
- Reports explicitly separate attention correlation, image-region causal
  evidence, and hidden/logit evidence.
- No training entrypoint is used.
