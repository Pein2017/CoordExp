"""Focused tests for the sorted supported false-negative native-prefix
reachability prevalence visualizer.

These tests run the real analyzer over a small synthetic predecessor run root
to produce a genuine ``analysis/`` directory, then exercise the visualizer
against it.  This keeps the visualizer tests honest about the exact artifact
shape the analyzer publishes without duplicating its internal logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402
from scripts.research import (  # noqa: E402
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as analyzer,
)
from scripts.research import (  # noqa: E402
    visualize_sorted_supported_fn_native_prefix_reachability_prevalence as viz,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(analyzer.canonical_json_bytes(row) + b"\n" for row in rows))


def _context(*, context_id, image_id, boundary_index, context_role, prefix_rows=None) -> dict:
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "context_id": context_id,
        "image_id": image_id,
        "boundary_index": boundary_index,
        "context_role": context_role,
        "prefix_admission": {
            "forced_continue_rows_excluded": True,
            "source": "native_greedy_complete_rows_only",
        },
        "prefix_rows": prefix_rows or [],
    }


def _owner_context_row(
    *,
    gt_owner_id,
    context_id,
    image_id,
    boundary_index,
    context_role,
    normalized_description,
    gate_margin,
    category_rank,
    owner_rank,
    best_gt_owner_id,
    passed_state,
) -> dict:
    return {
        "schema_version": merge.OWNER_CONTEXT_SCHEMA_VERSION,
        "owner_context_id": f"{gt_owner_id}@{context_id}",
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "image_id": image_id,
        "boundary_index": boundary_index,
        "context_role": context_role,
        "normalized_description": normalized_description,
        "loop_marking": {"loop_tail": False},
        "frontier_features": {"passed_state": passed_state},
        "category_proposal_channel": {
            "boundary_gate": {"continue_vs_stop_logprob_margin": gate_margin},
            "category_routing_event": (
                None if category_rank is None else {"within_context_rank": category_rank}
            ),
        },
        "owner_competition_u": {
            "rank": owner_rank,
            "best_gt_owner_id": best_gt_owner_id,
            "margin_to_best_owner": 0.0,
            "population_size": 2,
        },
        # This visualizer fixture does not exercise U/L bound divergence
        # (that is covered by the analyzer's own tests), so the L-bound
        # competition simply mirrors the U-bound values here.
        "owner_competition_l": {
            "rank": owner_rank,
            "best_gt_owner_id": best_gt_owner_id,
            "margin_to_best_owner": 0.0,
            "population_size": 2,
        },
    }


def _owner_summary(
    *,
    gt_owner_id,
    image_id,
    normalized_description,
    disposition,
    native_true_positive,
    upper_context_ids,
    lower_context_ids,
    non_loop_context_support=None,
) -> dict:
    return {
        "schema_version": merge.OWNER_SUMMARY_SCHEMA_VERSION,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "normalized_description": normalized_description,
        "disposition": disposition,
        "native_true_positive": native_true_positive,
        "ambiguity_bound_disposition_flip": False,
        "upper_bound_u": {
            "usable_support_context_ids": upper_context_ids,
            "non_loop_context_support": non_loop_context_support or {},
        },
        "lower_bound_l": {"usable_support_context_ids": lower_context_ids},
    }


def _native_sidecar(*, image_id, row_index, pred_row_id, gt_owner_id) -> dict:
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "image_id": image_id,
        "row_index": row_index,
        "pred_row_id": pred_row_id,
        "strict_match_status": "matched",
        "strict_match_gt_owner_id": gt_owner_id,
    }


def build_run_root(tmp_path: Path) -> Path:
    """The same small two-image-pair scenario used by the analyzer tests."""

    root = tmp_path / "predecessor"
    plan_dir = root / "plan"
    presentation_dir = root / "phases" / "presentation"
    plan_dir.mkdir(parents=True)
    presentation_dir.mkdir(parents=True)

    contexts = [
        _context(context_id="9001:boundary-000", image_id="9001", boundary_index=0, context_role="root"),
        _context(
            context_id="9001:boundary-001",
            image_id="9001",
            boundary_index=1,
            context_role="row_boundary",
            prefix_rows=[{"strict_match_status": "matched", "strict_match_gt_owner_id": "gt:9001:9"}],
        ),
        _context(context_id="9003:boundary-000", image_id="9003", boundary_index=0, context_role="root"),
        _context(context_id="9002:boundary-000", image_id="9002", boundary_index=0, context_role="root"),
        _context(context_id="9004:boundary-000", image_id="9004", boundary_index=0, context_role="root"),
    ]

    owner_context_rows = [
        _owner_context_row(
            gt_owner_id="gt:9001:0", context_id="9001:boundary-000", image_id="9001",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=2.0, category_rank=1, owner_rank=1, best_gt_owner_id="gt:9001:0",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9001:1", context_id="9001:boundary-001", image_id="9001",
            boundary_index=1, context_role="row_boundary", normalized_description="cat_b",
            gate_margin=0.5, category_rank=2, owner_rank=2, best_gt_owner_id="gt:9001:9",
            passed_state="ahead_of_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9003:0", context_id="9003:boundary-000", image_id="9003",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=1.0, category_rank=4, owner_rank=3, best_gt_owner_id="gt:9003:5",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9002:0", context_id="9002:boundary-000", image_id="9002",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=3.0, category_rank=1, owner_rank=1, best_gt_owner_id="gt:9002:0",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9004:0", context_id="9004:boundary-000", image_id="9004",
            boundary_index=0, context_role="root", normalized_description="cat_b",
            gate_margin=-2.0, category_rank=2, owner_rank=2, best_gt_owner_id="gt:9004:5",
            passed_state="root_no_frontier",
        ),
    ]

    owner_summaries = [
        _owner_summary(
            gt_owner_id="gt:9001:0", image_id="9001", normalized_description="cat_a",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9001:boundary-000"], lower_context_ids=["9001:boundary-000"],
        ),
        _owner_summary(
            gt_owner_id="gt:9001:1", image_id="9001", normalized_description="cat_b",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9001:boundary-001"], lower_context_ids=["9001:boundary-001"],
        ),
        _owner_summary(
            gt_owner_id="gt:9003:0", image_id="9003", normalized_description="cat_a",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9003:boundary-000"], lower_context_ids=["9003:boundary-000"],
        ),
        _owner_summary(
            gt_owner_id="gt:9002:0", image_id="9002", normalized_description="cat_a",
            disposition=merge.DISPOSITION_TP_CONTROL, native_true_positive=True,
            upper_context_ids=[], lower_context_ids=[],
            non_loop_context_support={"9002:boundary-000": True},
        ),
        _owner_summary(
            gt_owner_id="gt:9004:0", image_id="9004", normalized_description="cat_b",
            disposition=merge.DISPOSITION_TP_CONTROL, native_true_positive=True,
            upper_context_ids=[], lower_context_ids=[],
            non_loop_context_support={"9004:boundary-000": False},
        ),
    ]

    native_sidecars = [
        _native_sidecar(image_id="9002", row_index=0, pred_row_id="pred:sorted:greedy:0:9002:0", gt_owner_id="gt:9002:0"),
        _native_sidecar(image_id="9004", row_index=0, pred_row_id="pred:sorted:greedy:0:9004:0", gt_owner_id="gt:9004:0"),
    ]

    _write_jsonl(plan_dir / "context-registry.jsonl", contexts)
    _write_jsonl(plan_dir / "native-sidecar-registry.jsonl", native_sidecars)
    _write_jsonl(presentation_dir / "owner-context-features.jsonl", owner_context_rows)
    _write_jsonl(presentation_dir / "owner-summaries.jsonl", owner_summaries)

    plan_receipt = {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "output_file_digests": {
            "context-registry.jsonl": analyzer.sha256_bytes((plan_dir / "context-registry.jsonl").read_bytes()),
            "native-sidecar-registry.jsonl": analyzer.sha256_bytes(
                (plan_dir / "native-sidecar-registry.jsonl").read_bytes()
            ),
        },
    }
    plan_receipt["receipt_content_sha256"] = analyzer.sha256_json(plan_receipt)
    (plan_dir / "receipt.json").write_text(json.dumps(plan_receipt), encoding="utf-8")

    merge_receipt = {
        "schema_version": merge.MERGE_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "phase": merge.PHASE_PRESENTATION,
        "output_file_digests": {
            "owner-summaries.jsonl": analyzer.sha256_bytes(
                (presentation_dir / "owner-summaries.jsonl").read_bytes()
            ),
            "owner-context-features.jsonl": analyzer.sha256_bytes(
                (presentation_dir / "owner-context-features.jsonl").read_bytes()
            ),
        },
        "plan": {"receipt_content_sha256": plan_receipt["receipt_content_sha256"]},
        "usable_as_census_conclusion": True,
        "counts": {
            "owner_context_row_count": len(owner_context_rows),
            "owner_summary_row_count": len(owner_summaries),
        },
    }
    merge_receipt["receipt_content_sha256"] = analyzer.sha256_json(merge_receipt)
    (presentation_dir / "merge-receipt.json").write_text(json.dumps(merge_receipt), encoding="utf-8")

    return root


@pytest.fixture(autouse=True)
def _small_denominators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer, "EXPECTED_FN_DENOMINATOR", 3)
    monkeypatch.setattr(analyzer, "EXPECTED_TP_DENOMINATOR", 2)


@pytest.fixture()
def analysis_dir(tmp_path: Path) -> Path:
    run_root = build_run_root(tmp_path)
    output_root = tmp_path / "out"
    exit_code = analyzer.main(["--run-root", str(run_root), "--output-root", str(output_root)])
    assert exit_code == 0
    return output_root / "analysis"


def _reseal_after_owner_records_mutation(analysis_dir: Path) -> None:
    """Reseal the receipt's owner-records digest to the current bytes only.

    Simulates a corrupted-but-internally-consistent pipeline: the receipt
    still reconstructs and still matches the (mutated) owner-records bytes,
    so only the report/records cross-check can catch the row-count mismatch.
    """

    receipt_path = analysis_dir / analyzer.RECEIPT_NAME
    owner_records_path = analysis_dir / analyzer.OWNER_RECORDS_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["owner_records_jsonl_sha256"] = analyzer.sha256_bytes(owner_records_path.read_bytes())
    receipt["receipt_content_sha256"] = analyzer.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_load_artifacts_accepts_a_genuine_analysis_directory(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    assert artifacts.report["false_negative_cohort"]["denominator"] == 3
    assert artifacts.report["native_true_positive_reference"]["denominator"] == 2
    fn_rows = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    tp_rows = [r for r in artifacts.owner_records if r["cohort"] == viz.TP_COHORT]
    assert len(fn_rows) == 3
    assert len(tp_rows) == 2


def test_load_artifacts_fails_closed_on_tampered_owner_records(analysis_dir: Path) -> None:
    owner_records_path = analysis_dir / analyzer.OWNER_RECORDS_NAME
    owner_records_path.write_bytes(owner_records_path.read_bytes() + b'{"tampered": true}\n')
    with pytest.raises(viz.VisualContractError, match="digest"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_tampered_report(analysis_dir: Path) -> None:
    report_path = analysis_dir / analyzer.REPORT_JSON_NAME
    payload = json.loads(report_path.read_text())
    payload["false_negative_cohort"]["denominator"] = 999
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="digest"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_receipt_self_digest_tamper(analysis_dir: Path) -> None:
    receipt_path = analysis_dir / analyzer.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["fn_denominator"] = 12345
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="does not reconstruct"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_report_records_row_count_mismatch(analysis_dir: Path) -> None:
    owner_records_path = analysis_dir / analyzer.OWNER_RECORDS_NAME
    rows = [json.loads(line) for line in owner_records_path.read_text().splitlines() if line.strip()]
    fn_rows = [row for row in rows if row["cohort"] == analyzer.FN_COHORT]
    dropped = [row for row in rows if row is not fn_rows[0]]
    owner_records_path.write_bytes(
        b"".join(analyzer.canonical_json_bytes(row) + b"\n" for row in dropped)
    )
    _reseal_after_owner_records_mutation(analysis_dir)
    with pytest.raises(viz.VisualContractError, match="false-negative row count"):
        viz.load_artifacts(analysis_dir)


def test_owner_matrix_spec_is_grouped_by_image_then_category(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_rows = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    spec = viz.build_owner_matrix_spec(fn_rows)
    assert spec["row_count"] == 3
    keys = [(row["image_id"], row["normalized_description"]) for row in spec["rows"]]
    assert keys == sorted(keys)
    by_owner = {row["gt_owner_id"]: row for row in spec["rows"]}
    assert by_owner["gt:9001:0"]["cells"]["owner_rank"]["text"] == "rank1"
    assert by_owner["gt:9001:0"]["cells"]["competitor"]["text"] == "target"
    assert by_owner["gt:9001:1"]["cells"]["competitor"]["text"] == "covered"
    assert by_owner["gt:9003:0"]["cells"]["competitor"]["text"] == "uncovered"


def test_native_tp_matrix_spec_reflects_support_and_channel(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    tp_rows = [r for r in artifacts.owner_records if r["cohort"] == viz.TP_COHORT]
    spec = viz.build_native_tp_matrix_spec(tp_rows)
    assert spec["row_count"] == 2
    by_owner = {row["gt_owner_id"]: row for row in spec["rows"]}
    t1 = by_owner["gt:9002:0"]
    assert t1["cells"]["support_count"]["text"] == "1"
    assert t1["cells"]["gate"]["text"] == "open"
    assert t1["cells"]["favorable_rank1"]["text"] == "yes"
    t2 = by_owner["gt:9004:0"]
    assert t2["cells"]["support_count"]["text"] == "0"
    assert t2["cells"]["gate"]["text"] == "closed"


def test_summary_panel_small_multiples_cover_every_stratum_and_match_report(
    analysis_dir: Path,
) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_summary_panel_spec(artifacts.report)
    upper = artifacts.report["false_negative_cohort"]["headline"]["upper_bound_u"]

    # This fixture's two images (9001, 9003) and two categories (cat_a,
    # cat_b) must all be covered; against the real twelve-image census this
    # is the twelve-image/all-category coverage invariant.
    per_image_keys = {row["stratum"] for row in spec["per_image_small_multiples"]}
    assert per_image_keys == set(upper["per_image"]) == {"9001", "9003"}
    per_category_keys = {row["stratum"] for row in spec["per_category_small_multiples"]}
    assert per_category_keys == set(upper["per_category"]) == {"cat_a", "cat_b"}

    for row in spec["per_image_small_multiples"]:
        stratum = upper["per_image"][row["stratum"]]
        assert row["owner_count"] == stratum["owner_count"]
        for metric in spec["small_multiple_metrics"]:
            assert row[metric] == stratum[metric]

    for row in spec["per_category_small_multiples"]:
        stratum = upper["per_category"][row["stratum"]]
        assert row["owner_count"] == stratum["owner_count"]
        for metric in spec["small_multiple_metrics"]:
            assert row[metric] == stratum[metric]


def test_render_summary_panel_png_includes_small_multiple_sections(
    analysis_dir: Path, tmp_path: Path
) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_summary_panel_spec(artifacts.report)
    output_path = tmp_path / "summary-panel.png"
    viz.render_summary_panel_png(spec, output_path)
    assert output_path.is_file()

    from PIL import Image

    with Image.open(output_path) as image:
        bars_only_height = 24 + 18 * max(1, len(spec["bars"]))
        assert image.height > bars_only_height


def test_no_cell_color_is_a_continuous_or_cross_image_value(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_rows = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    spec = viz.build_owner_matrix_spec(fn_rows)
    allowed = {None, viz.COLOR_GREEN, viz.COLOR_RED, viz.COLOR_YELLOW, viz.COLOR_GREY, viz.COLOR_BLUE}
    for row in spec["rows"]:
        for cell in row["cells"].values():
            assert cell.get("color") in allowed


def test_main_specs_only_writes_manifest_without_rendering_png(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(
        ["--analysis-root", str(analysis_dir), "--output-dir", str(output_dir), "--specs-only"]
    )
    assert exit_code == 0
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    assert manifest["rendered_paths"]["owner_matrix_png"] is None
    assert not (output_dir / viz.OWNER_MATRIX_NAME).exists()
    assert (output_dir / "owner-matrix-spec.json").is_file()


def test_main_renders_both_matrix_png_files(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(["--analysis-root", str(analysis_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / viz.OWNER_MATRIX_NAME).is_file()
    assert (output_dir / viz.NATIVE_TP_MATRIX_NAME).is_file()
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    assert manifest["rendered_paths"]["owner_matrix_png"] is not None
    assert manifest["analysis_input_digests"]["receipt_content_sha256"]


def test_manifest_seals_exact_sha256_and_size_of_every_emitted_file(
    analysis_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(["--analysis-root", str(analysis_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())

    reconstructed = viz.analyzer.sha256_json(
        {k: v for k, v in manifest.items() if k != "visual_manifest_sha256"}
    )
    assert reconstructed == manifest["visual_manifest_sha256"]

    for name, expected_filename in (
        ("owner_matrix_spec", "owner-matrix-spec.json"),
        ("native_tp_matrix_spec", "native-tp-matrix-spec.json"),
        ("summary_panel_spec", "summary-panel-spec.json"),
        ("owner_matrix_png", viz.OWNER_MATRIX_NAME),
        ("native_tp_matrix_png", viz.NATIVE_TP_MATRIX_NAME),
        ("summary_panel_png", viz.SUMMARY_PANEL_NAME),
        ("representative_case_references", viz.REPRESENTATIVE_CASE_REFERENCES_NAME),
    ):
        entry = manifest["rendered_paths"][name]
        assert entry is not None
        on_disk = (output_dir / expected_filename).read_bytes()
        assert entry["sha256"] == viz.analyzer.sha256_bytes(on_disk)
        assert entry["size_bytes"] == len(on_disk)
        assert entry["path"] == str(output_dir / expected_filename)


def test_manifest_self_digest_fails_closed_on_tamper(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(
        ["--analysis-root", str(analysis_dir), "--output-dir", str(output_dir), "--specs-only"]
    )
    assert exit_code == 0
    manifest_path = output_dir / viz.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["output_dir"] = "/tampered/path"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reloaded = json.loads(manifest_path.read_text())
    reconstructed = viz.analyzer.sha256_json(
        {k: v for k, v in reloaded.items() if k != "visual_manifest_sha256"}
    )
    assert reconstructed != reloaded["visual_manifest_sha256"]


def test_manifest_digest_detects_rewritten_spec_bytes(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(
        ["--analysis-root", str(analysis_dir), "--output-dir", str(output_dir), "--specs-only"]
    )
    assert exit_code == 0
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    sealed_sha256 = manifest["rendered_paths"]["owner_matrix_spec"]["sha256"]

    spec_path = output_dir / "owner-matrix-spec.json"
    spec_path.write_bytes(spec_path.read_bytes() + b"\n")
    rewritten_sha256 = viz.analyzer.sha256_bytes(spec_path.read_bytes())

    assert rewritten_sha256 != sealed_sha256


# ---------------------------------------------------------------------------
# Representative-case selection and external image references
# ---------------------------------------------------------------------------


def test_select_representative_cases_is_deterministic(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_records = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    cases = viz.select_representative_cases(fn_records)

    # O1 (gt:9001:0) is favorable-top3-before/at under both bounds at its one
    # shared context; O2 loses only to a covered competitor; O3 loses only to
    # an uncovered competitor; no owner in this fixture has support only
    # after the frontier.
    assert cases[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]["record"]["gt_owner_id"] == "gt:9001:0"
    assert cases[viz.CASE_ONLY_COVERED_OTHER]["record"]["gt_owner_id"] == "gt:9001:1"
    assert cases[viz.CASE_ONLY_UNCOVERED_OTHER]["record"]["gt_owner_id"] == "gt:9003:0"
    assert cases[viz.CASE_SUPPORT_ONLY_AFTER_FRONTIER] is None

    # Selection must be stable across repeated calls (no hidden randomness).
    again = viz.select_representative_cases(fn_records)
    for case_name in viz.REPRESENTATIVE_CASE_NAMES:
        left = cases[case_name]
        right = again[case_name]
        if left is None:
            assert right is None
        else:
            assert left["record"]["gt_owner_id"] == right["record"]["gt_owner_id"]


def test_resolve_representative_case_references_without_visual_root(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_records = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    cases = viz.select_representative_cases(fn_records)
    references = viz.resolve_representative_case_references(cases, visual_root=None)

    assert references[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]["selected"] is True
    assert references[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]["source_image"] == {
        "reason": "visual_root_not_supplied"
    }
    assert references[viz.CASE_SUPPORT_ONLY_AFTER_FRONTIER] == {
        "selected": False,
        "reason": "no_qualifying_owner",
        "source_image": None,
    }


def test_resolve_representative_case_references_hashes_exact_bytes(
    analysis_dir: Path, tmp_path: Path
) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_records = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    cases = viz.select_representative_cases(fn_records)

    visual_root = tmp_path / "visual_combined"
    visual_root.mkdir()
    for image_id, payload in (("9001", b"not-a-real-png-just-bytes-9001"), ("9003", b"not-a-real-png-just-bytes-9003")):
        (visual_root / f"owner_map__{image_id}.png").write_bytes(payload)

    references = viz.resolve_representative_case_references(cases, visual_root=visual_root)
    favorable = references[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]
    assert favorable["image_id"] == "9001"
    expected_sha256 = viz.analyzer.sha256_bytes(b"not-a-real-png-just-bytes-9001")
    assert favorable["source_image"]["sha256"] == expected_sha256
    assert favorable["source_image"]["size_bytes"] == len(b"not-a-real-png-just-bytes-9001")
    assert favorable["source_image"]["path"] == str(visual_root / "owner_map__9001.png")

    # Tamper visibility: modifying the referenced file changes its digest.
    (visual_root / "owner_map__9001.png").write_bytes(b"tampered-bytes")
    retampered = viz.resolve_representative_case_references(cases, visual_root=visual_root)
    assert (
        retampered[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]["source_image"]["sha256"]
        != expected_sha256
    )


def test_resolve_representative_case_references_fails_closed_on_missing_image(
    analysis_dir: Path, tmp_path: Path
) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    fn_records = [r for r in artifacts.owner_records if r["cohort"] == viz.FN_COHORT]
    cases = viz.select_representative_cases(fn_records)

    empty_visual_root = tmp_path / "empty_visual_root"
    empty_visual_root.mkdir()
    with pytest.raises(viz.VisualContractError, match="source image is absent"):
        viz.resolve_representative_case_references(cases, visual_root=empty_visual_root)


def test_main_with_representative_visual_root_seals_external_images(
    analysis_dir: Path, tmp_path: Path
) -> None:
    visual_root = tmp_path / "visual_combined"
    visual_root.mkdir()
    for image_id in ("9001", "9003"):
        (visual_root / f"owner_map__{image_id}.png").write_bytes(f"bytes-{image_id}".encode())

    output_dir = tmp_path / "viz"
    exit_code = viz.main(
        [
            "--analysis-root", str(analysis_dir),
            "--output-dir", str(output_dir),
            "--representative-visual-root", str(visual_root),
        ]
    )
    assert exit_code == 0

    references = json.loads((output_dir / viz.REPRESENTATIVE_CASE_REFERENCES_NAME).read_text())
    assert references["visual_root_supplied"] is True
    favorable = references["cases"][viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]
    assert favorable["source_image"]["sha256"] == viz.analyzer.sha256_bytes(b"bytes-9001")

    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    external = manifest["external_representative_image_references"]
    assert external[viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS]["sha256"] == viz.analyzer.sha256_bytes(
        b"bytes-9001"
    )
    assert viz.CASE_SUPPORT_ONLY_AFTER_FRONTIER not in external
    assert manifest["visualizer_source_sha256"] == viz.analyzer.sha256_bytes(
        Path(viz.__file__).resolve().read_bytes()
    )
    # The case-references file itself is sealed like every other emitted spec.
    assert manifest["rendered_paths"]["representative_case_references"]["sha256"] == viz.analyzer.sha256_bytes(
        (output_dir / viz.REPRESENTATIVE_CASE_REFERENCES_NAME).read_bytes()
    )


def test_main_fails_closed_when_representative_visual_root_missing_image(
    analysis_dir: Path, tmp_path: Path
) -> None:
    empty_visual_root = tmp_path / "empty_visual_root"
    empty_visual_root.mkdir()
    output_dir = tmp_path / "viz"
    with pytest.raises(SystemExit, match="source image is absent"):
        viz.main(
            [
                "--analysis-root", str(analysis_dir),
                "--output-dir", str(output_dir),
                "--representative-visual-root", str(empty_visual_root),
            ]
        )


def test_main_specs_only_still_seals_external_references_when_root_supplied(
    analysis_dir: Path, tmp_path: Path
) -> None:
    visual_root = tmp_path / "visual_combined"
    visual_root.mkdir()
    for image_id in ("9001", "9003"):
        (visual_root / f"owner_map__{image_id}.png").write_bytes(f"bytes-{image_id}".encode())

    output_dir = tmp_path / "viz"
    exit_code = viz.main(
        [
            "--analysis-root", str(analysis_dir),
            "--output-dir", str(output_dir),
            "--specs-only",
            "--representative-visual-root", str(visual_root),
        ]
    )
    assert exit_code == 0
    assert not (output_dir / viz.OWNER_MATRIX_NAME).exists()
    assert not (output_dir / viz.SUMMARY_PANEL_NAME).exists()
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    assert viz.CASE_ROBUST_FAVORABLE_PREFRONTIER_MISS in manifest["external_representative_image_references"]
