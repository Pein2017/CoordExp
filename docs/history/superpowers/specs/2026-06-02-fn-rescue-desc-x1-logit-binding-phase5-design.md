# FN-Rescue Desc-X1 Logit Binding Phase 5 Design

## Goal

Build a CPU-first Phase-5 analysis that links Phase-4 instance-attention
binding rows to Lane-D state-dependent `x1_logit_lens_*` rows.  The experiment
tests whether desc-first decoding failures are visible in the coordinate-logit
state around `desc_end`, `box_start`, and `pre_x1`, while explicitly recording
which requested roles are not covered by the current Lane-D artifact.

## Scope

In scope:

- Read existing artifacts only; do not launch production training.
- Join Phase-4 `instance_attention_binding/rows.jsonl` with Lane-D
  `probe_rows.jsonl` by `case_id`.
- Treat `x1_logit_lens_rank`, `x1_logit_lens_top1_bin`, and
  `x1_logit_lens_target_minus_top1` as state-dependent probe fields.
- Treat `x1_target_rank` and `x1_top_peak_attribution` as case-level Lane-C
  selection metadata, not as dynamic per-role logits.
- Report role/layer coverage for the user-requested roles:
  `desc_end`, `pre_x1`, `post_x1`, and `pre_y1`.  The current artifact also
  includes `box_start`, so it is included as a bridge role.
- Split summaries by `mechanism_bucket`, role, layer group, and by whether
  Phase-4 target-minus-competitor instance attention is positive.

Out of scope:

- New GPU hidden-state extraction.
- Attention-head interventions.
- Training objective changes.
- Background-token suppression.

## Data Flow

```text
Phase-4 instance_attention_binding/rows.jsonl
        +
Lane-D x1_logit_lens_smoke/probe_rows.jsonl
        |
        v
Phase-5 x1_logit_binding_probe/rows.jsonl
Phase-5 x1_logit_binding_probe/summary.json
Phase-5 report.md
```

## Interpretation Boundaries

This phase can support statements about current artifact coverage and
coordinate-logit rank/margin correlations.  It cannot yet prove that a hidden
state causes a particular object binding.  If `post_x1` or `pre_y1` lacks
`x1_logit_lens_*` fields, the correct conclusion is a coverage gap, not a
negative result for those roles.

## Success Criteria

- The analysis writes JSON-safe rows, summary, and report artifacts.
- It detects state-dependent logit-lens coverage and missing requested roles.
- It surfaces the critical slice: positive target attention but poor target x1
  logit rank or negative target-minus-top1 logit margin.
- The findings document records both the result and the coverage limitation.

## Phase-5B Slot-Aware Extension

The `post_x1` state should not be interpreted with an x1-only target.  Once
`x1` has been emitted, the next-token decision is `y1`.  Phase-5B therefore
adds `coord_slot_logit_lens_*` fields:

- `pre_x1` targets `x1`;
- `post_x1` targets `y1`;
- `post_y1` targets `x2`.

This extension is still CPU/GPU probe-only analysis, not training.  It should
be used as the preferred evidence for claims about coordinate-chain binding
after `x1`.
