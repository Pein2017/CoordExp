# Recall Input Gap

Real family-level recall analysis for the matched `val200` comparison set is not yet runnable from existing artifacts.

Checked on `2026-04-20`:

- No matching `oracle_k/.../fn_objects.jsonl` artifacts were found for:
  - `base_xyxy_merged`
  - `raw_text_xyxy_pure_ce`
  - `cxcywh_pure_ce`
  - `cxcy_logw_logh_pure_ce`
  - `center_parameterization`
  - `hard_soft_ce_2b`
- No matching `gt_proxy_scores.jsonl` / `proposal_proxy_scores.jsonl` artifacts were found under `output/analysis` for those same family aliases or checkpoint ids.

Implication:

- `coord_family_recall_probe.py` is ready to ingest real artifacts through `artifact_sources`.
- The current blocker is upstream artifact generation, not report or recall-probe code.

Next required step:

1. Produce `oracle_k` false-negative objects for each family on the matched `val200` slice.
2. Produce `gt_proxy_scores.jsonl` and `proposal_proxy_scores.jsonl` for each family on the same slice.
3. Materialize a non-smoke recall config that points `artifact_sources` at those family-specific files.
