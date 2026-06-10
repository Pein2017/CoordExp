from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.probe_plan import build_probe_plan


def test_probe_plan_assigns_sampled_rows_to_shards(minimal_base_row) -> None:
    rows = []
    for idx in range(5):
        row = dict(minimal_base_row)
        row["case_id"] = f"case-{idx}"
        row["case_index_row_id"] = f"ci-{idx}"
        row["desc_text_canonical"] = "person"
        row["same_desc_count_bucket"] = "same_desc_3"
        row["object_size_bucket"] = "medium"
        row["overlap_bucket"] = "low_overlap"
        rows.append(row)

    plan, summary = build_probe_plan(rows, num_shards=2, max_cases=3, seed=7)

    assert len(plan) == 5
    sampled = [row for row in plan if row["probe_sampled"]]
    assert len(sampled) == 3
    assert {row["planned_shard_id"] for row in sampled} <= {0, 1}
    assert summary["gpu_probe_planned_cases"] == 3
