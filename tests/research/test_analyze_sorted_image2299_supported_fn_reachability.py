"""CPU-only fixtures for the image-2299 supported-FN S2 analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_image2299_supported_fn_reachability as az  # noqa: E402
from scripts.research import analyze_sorted_image2299_owner_accessibility as accessibility  # noqa: E402
from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(az.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(az.canonical_json_bytes(row) + b"\n" for row in rows))


def _context(
    context_id: str,
    boundary: int,
    role: str,
    *,
    covered_owner_id: str | None = None,
) -> dict:
    prefix_rows = []
    if covered_owner_id is not None:
        prefix_rows.append(
            {
                "strict_match_status": "matched",
                "strict_match_gt_owner_id": covered_owner_id,
            }
        )
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "context_id": context_id,
        "image_id": az.IMAGE_ID,
        "boundary_index": boundary,
        "context_role": role,
        "prefix_admission": {
            "forced_continue_rows_excluded": True,
            "source": "native_greedy_complete_rows_only",
        },
        "prefix_rows": prefix_rows,
    }


def _feature(
    owner_id: str,
    context_id: str,
    boundary: int,
    role: str,
    state: str,
    *,
    gate_margin: float = 1.0,
    category_rank: int | None = 1,
    owner_rank_u: int | None = 1,
    owner_rank_l: int | None = None,
    loop_tail: bool = False,
    local_candidate_rank: int = 99,
    support_u: bool = False,
    support_l: bool = False,
) -> dict:
    owner_rank_l = owner_rank_u if owner_rank_l is None else owner_rank_l
    return {
        "schema_version": accessibility.CONTEXT_SCHEMA_VERSION,
        "row_kind": "image2299_owner_context_support",
        "owner_context_id": f"{owner_id}@{context_id}",
        "gt_owner_id": owner_id,
        "context_id": context_id,
        "image_id": az.IMAGE_ID,
        "boundary_index": boundary,
        "context_role": role,
        "native_context_only": True,
        "normalized_description": "person",
        "loop_marking": {"loop_tail": loop_tail},
        "frontier_features": {"passed_state": state},
        "localization": {
            "estimand": "category_field_support_at_owner_geometry",
            "proposal_probability": False,
            "generator_local_max_excluding_other_owner_strict": {
                "ambiguity_excluded_l": {
                    "value": 1.0 if support_l else 0.0,
                    "clears_frozen_support_rule": support_l,
                },
                "ambiguity_included_u": {
                    "value": 1.0 if support_u else 0.0,
                    "clears_frozen_support_rule": support_u,
                },
            },
        },
        "proposal_surface": {
            "separate_from_localization": True,
            "combined_with_localization": False,
            "boundary_gate": {"continue_vs_stop_logprob_margin": gate_margin},
            "category_routing_event": (
                None if category_rank is None else {"within_context_rank": category_rank}
            ),
        },
        # Deliberately unrelated: the analyzer must never substitute this for
        # either category-route rank or same-category physical-owner rank.
        "coordinate_candidate_bank_rank": local_candidate_rank,
        "owner_competition_u": {
            "rank": owner_rank_u,
            "margin_to_best_owner": 0.0,
            "population_size": 4,
        },
        "owner_competition_l": {
            "rank": owner_rank_l,
            "margin_to_best_owner": -0.5,
            "population_size": 4,
        },
    }


def _summary(
    owner_id: str,
    *,
    disposition: str = accessibility.DISPOSITION_RESOLVED,
    native_true_positive: bool = False,
    upper: list[str] | None = None,
    lower: list[str] | None = None,
    ambiguity_flip: bool = False,
) -> dict:
    return {
        "schema_version": accessibility.OWNER_SCHEMA_VERSION,
        "row_kind": "image2299_owner_accessibility_summary",
        "gt_owner_id": owner_id,
        "image_id": az.IMAGE_ID,
        "normalized_description": "person",
        "disposition": disposition,
        "native_true_positive": native_true_positive,
        "native_false_negative": not native_true_positive,
        "ambiguity_bound_disposition_flip": ambiguity_flip,
        "upper_bound_u": {"usable_support_context_ids": upper or []},
        "lower_bound_l": {"usable_support_context_ids": lower or []},
    }


def _source_payloads() -> tuple[list[dict], list[dict], list[dict]]:
    owner_a = "gt:2299:a"
    owner_b = "gt:2299:b"
    owner_c = "gt:2299:c"
    covered = "gt:2299:covered"

    contexts = [
        _context("2299:boundary-000", 0, "root"),
        _context("2299:boundary-001", 1, "row_boundary", covered_owner_id=owner_a),
        _context("2299:boundary-002", 2, "row_boundary", covered_owner_id=covered),
        _context("2299:boundary-003", 3, "terminal"),
    ]
    owners = (owner_a, owner_b, owner_c, covered)
    default_ranks = {owner_a: 1, owner_b: 2, owner_c: 3, covered: 4}
    states = {
        owner_a: ("root_no_frontier", "passed_by_frontier", "passed_by_frontier", "passed_by_frontier"),
        owner_b: ("root_no_frontier", "ahead_of_frontier", "passed_by_frontier", "passed_by_frontier"),
        owner_c: ("ahead_of_frontier", "at_frontier", "at_frontier", "passed_by_frontier"),
        covered: ("root_no_frontier", "ahead_of_frontier", "passed_by_frontier", "passed_by_frontier"),
    }
    features: list[dict] = []
    for owner_id in owners:
        for index, context in enumerate(contexts):
            rank_u = default_ranks[owner_id]
            rank_l = rank_u
            # A's root L-bound rank is two; the persistent owner is rank one,
            # preserving one unique S1 rank-one row for the relational join.
            if index == 0:
                if owner_id == owner_a:
                    rank_l = 2
                elif owner_id == covered:
                    rank_l = 1
                elif owner_id == owner_b:
                    rank_l = 3
                elif owner_id == owner_c:
                    rank_l = 4
            features.append(
                _feature(
                    owner_id,
                    str(context["context_id"]),
                    int(context["boundary_index"]),
                    str(context["context_role"]),
                    states[owner_id][index],
                    category_rank=2 if owner_id == owner_b and index == 1 else 1,
                    owner_rank_u=rank_u,
                    owner_rank_l=rank_l,
                    support_u=(owner_id, index) in {(owner_a, 0), (owner_b, 1), (owner_c, 2)},
                    support_l=(owner_id, index) in {(owner_a, 0), (owner_b, 1)},
                )
            )
    summaries = [
        _summary(
            owner_a,
            upper=["2299:boundary-000"],
            lower=["2299:boundary-000"],
        ),
        _summary(
            owner_b,
            upper=["2299:boundary-001"],
            lower=["2299:boundary-001"],
        ),
        _summary(
            owner_c,
            disposition=accessibility.DISPOSITION_FLIP,
            upper=["2299:boundary-002"],
            lower=[],
            ambiguity_flip=True,
        ),
        _summary(
            covered,
            disposition=accessibility.DISPOSITION_PERSISTENT,
            upper=[],
            lower=[],
        ),
    ]
    return summaries, features, contexts


def _write_sources(
    root: Path,
    *,
    summaries: list[dict] | None = None,
    features: list[dict] | None = None,
    contexts: list[dict] | None = None,
) -> az.SourcePaths:
    root.mkdir(parents=True, exist_ok=True)
    default_summaries, default_features, default_contexts = _source_payloads()
    summaries = default_summaries if summaries is None else summaries
    features = default_features if features is None else features
    contexts = default_contexts if contexts is None else contexts

    paths = az.SourcePaths(
        analysis=root / "analysis.json",
        owner_summaries=root / "owner-summaries.jsonl",
        owner_context_features=root / "owner-context-features.jsonl",
        context_registry=root / "context-registry.jsonl",
        source_receipt=root / "receipt.json",
    )
    analysis = {
        "schema_version": accessibility.ANALYSIS_SCHEMA_VERSION,
        "unit_id": az.UNIT_ID,
        "image_id": az.IMAGE_ID,
        "calibration": {
            "content_sha256": az.FROZEN_CALIBRATION_CONTENT_SHA256,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "denominators": {
            "image2299_owner_count": len(summaries),
            "image2299_native_tp_count": sum(row["native_true_positive"] for row in summaries),
            "image2299_native_fn_count": sum(row["native_false_negative"] for row in summaries),
            "pooled_13_image_denominator_created": False,
        },
        "calibration_transfer": {
            "status": "calibration_transfer_underpowered",
            "passes": None,
            "validity_bearing": False,
        },
    }
    receipt: dict = {
        "schema_version": accessibility.RECEIPT_SCHEMA_VERSION,
        "unit_id": az.UNIT_ID,
        "calibration": {
            "content_sha256": az.FROZEN_CALIBRATION_CONTENT_SHA256,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "output_file_digests": {},
    }
    # This is deliberately the concrete sibling S1 product type, not a
    # hand-invented approximation of its file serialization.
    product = accessibility.AnalysisProduct(
        analysis=analysis,
        owner_summaries=summaries,
        owner_contexts=features,
        context_registry=contexts,
        receipt=receipt,
    )
    product_files = product.files()
    receipt["output_file_digests"] = {
        name: az.sha256_bytes(content) for name, content in sorted(product_files.items())
    }
    receipt["receipt_content_sha256"] = az._receipt_digest(receipt)
    for name, content in product_files.items():
        (root / name).write_bytes(content)
    _write_json(paths.source_receipt, receipt)
    return paths


def _reseal(paths: az.SourcePaths) -> None:
    receipt = {
        "schema_version": az.SOURCE_RECEIPT_SCHEMA_VERSION,
        "unit_id": az.UNIT_ID,
        "calibration": {
            "content_sha256": az.FROZEN_CALIBRATION_CONTENT_SHA256,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "output_file_digests": {
            "analysis.json": az.sha256_bytes(paths.analysis.read_bytes()),
            "owner-summaries.jsonl": az.sha256_bytes(paths.owner_summaries.read_bytes()),
            "owner-context-features.jsonl": az.sha256_bytes(paths.owner_context_features.read_bytes()),
            "context-registry.jsonl": az.sha256_bytes(paths.context_registry.read_bytes()),
        },
    }
    receipt["receipt_content_sha256"] = az._receipt_digest(receipt)
    _write_json(paths.source_receipt, receipt)


@pytest.fixture
def sources(tmp_path: Path) -> az.SourcePaths:
    return _write_sources(tmp_path / "s1")


def _by_owner(result: dict) -> dict[str, dict]:
    return {row["gt_owner_id"]: row for row in result["owner_records"]}


def test_s1_analysis_product_fixture_matches_concrete_contract(sources: az.SourcePaths) -> None:
    analysis = az._read_json(sources.analysis, "analysis")
    receipt = az._read_json(sources.source_receipt, "receipt")
    assert analysis["schema_version"] == accessibility.ANALYSIS_SCHEMA_VERSION
    assert receipt["schema_version"] == accessibility.RECEIPT_SCHEMA_VERSION
    assert receipt["output_file_digests"]["analysis.json"] == az.sha256_bytes(
        sources.analysis.read_bytes()
    )
    assert az.run_analysis(sources)["report"]["validation"][
        "calibration_transfer_status"
    ] == "calibration_transfer_underpowered"


def test_resolved_denominator_ladder_bounds_and_ambiguity(sources: az.SourcePaths) -> None:
    result = az.run_analysis(sources)
    report = result["report"]
    records = _by_owner(result)

    assert report["resolved_false_negative_cohort"]["denominator"] == 2
    assert report["validation"]["resolved_native_false_negative_ambiguity_flip_count"] == 0
    assert "gt:2299:covered" not in records
    assert "gt:2299:c" not in records

    owner_a = records["gt:2299:a"]
    assert owner_a["upper_bound_u"]["any_owner_rank_one"] is True
    assert owner_a["lower_bound_l"]["any_owner_rank_one"] is False
    assert owner_a["ambiguity_bound_disposition_flip"] is False


def test_rank_fields_are_distinct_and_competitor_coverage_is_native_prefix_only(
    sources: az.SourcePaths,
) -> None:
    records = _by_owner(az.run_analysis(sources))
    channel_a = records["gt:2299:a"]["upper_bound_u"]["contexts"][0]
    assert channel_a["category_routing_within_context_rank"] == 1
    assert channel_a["owner_rank_within_group"] == 1
    # The unrelated fixture field is 99 and must not leak into either rank.
    assert channel_a["category_routing_within_context_rank"] != 99
    assert channel_a["owner_rank_within_group"] != 99

    owner_b = records["gt:2299:b"]
    channel_b = owner_b["upper_bound_u"]["contexts"][0]
    assert channel_b["category_routing_within_context_rank"] == 2
    assert channel_b["owner_rank_within_group"] == 2
    assert channel_b["best_gt_owner_id"] == "gt:2299:a"
    assert channel_b["competitor_status"] == az.prevalence.COMPETITOR_COVERED_OTHER
    assert owner_b["only_covered_other_among_never_rank_one"] is True


def test_exact_crossing_membership_uses_first_passed_and_excludes_at_frontier(
    sources: az.SourcePaths,
) -> None:
    result = az.run_analysis(sources)
    records = _by_owner(result)
    crossing_a = records["gt:2299:a"]["exact_crossing"]
    assert crossing_a["exact_crossing_exists"] is True
    assert crossing_a["boundary_index"] == 0
    assert crossing_a["p_context_id"] == "2299:boundary-000"
    assert crossing_a["p_plus_e_context_id"] == "2299:boundary-001"
    assert crossing_a["u_favorable_supported"] is True
    assert crossing_a["l_favorable_supported"] is False

    crossing_b = records["gt:2299:b"]["exact_crossing"]
    assert crossing_b["exact_crossing_exists"] is True
    assert crossing_b["boundary_index"] == 1
    assert crossing_b["u_favorable_supported"] is False

    # C's predecessor is at_frontier, not root/ahead. It is not repaired by
    # scanning beyond the first passed boundary.
    inputs = az.load_inputs(sources)
    c_rows = az._owner_context_rows("gt:2299:c", inputs)
    assert az.derive_crossing_boundary(c_rows) is None
    gate = result["report"]["exact_crossing_gate"]
    assert gate["u_primary_exact_crossing_favorable_owner_ids"] == ["gt:2299:a"]


def test_loop_tail_usable_support_is_rejected(sources: az.SourcePaths) -> None:
    rows = az._read_jsonl(sources.owner_context_features, "features")
    rows[0]["loop_marking"]["loop_tail"] = True
    _write_jsonl(sources.owner_context_features, rows)
    _reseal(sources)
    with pytest.raises(az.AnalysisContractError, match="loop_tail"):
        az.run_analysis(sources)


def test_forced_continue_context_is_rejected(sources: az.SourcePaths) -> None:
    rows = az._read_jsonl(sources.context_registry, "contexts")
    rows[0]["prefix_admission"]["forced_continue_rows_excluded"] = False
    _write_jsonl(sources.context_registry, rows)
    _reseal(sources)
    with pytest.raises(az.AnalysisContractError, match="forced-continue"):
        az.run_analysis(sources)


def test_source_identities_are_retained_and_tamper_fails_closed(sources: az.SourcePaths) -> None:
    result = az.run_analysis(sources)
    record = _by_owner(result)["gt:2299:a"]
    assert record["source_context_ids"] == {
        "upper_bound_u": ["2299:boundary-000"],
        "lower_bound_l": ["2299:boundary-000"],
    }
    assert len(record["source_owner_summary_sha256"]) == 64
    assert result["report"]["input_file_sha256"]["owner_context_features"] == az.sha256_bytes(
        sources.owner_context_features.read_bytes()
    )

    sources.owner_context_features.write_bytes(
        sources.owner_context_features.read_bytes() + b'{"tampered":true}\n'
    )
    with pytest.raises(az.AnalysisContractError, match="digest mismatch"):
        az.run_analysis(sources)


def test_wrong_input_schema_and_wrong_calibration_fail_closed(sources: az.SourcePaths) -> None:
    analysis = az._read_json(sources.analysis, "analysis")
    analysis["schema_version"] = "foreign.v1"
    _write_json(sources.analysis, analysis)
    _reseal(sources)
    with pytest.raises(az.AnalysisContractError, match="schema_version"):
        az.run_analysis(sources)

    analysis["schema_version"] = az.SOURCE_ANALYSIS_SCHEMA_VERSION
    analysis["calibration"]["thresholds_retuned"] = True
    _write_json(sources.analysis, analysis)
    _reseal(sources)
    with pytest.raises(az.AnalysisContractError, match="non-retuned"):
        az.run_analysis(sources)


def test_zero_resolved_cohort_has_null_intervals_and_closed_gate(tmp_path: Path) -> None:
    persistent = _summary(
        "gt:2299:persistent",
        disposition=accessibility.DISPOSITION_PERSISTENT,
        upper=[],
        lower=[],
    )
    paths = _write_sources(tmp_path / "zero", summaries=[persistent], features=[], contexts=[])
    result = az.run_analysis(paths)
    report = result["report"]
    assert result["owner_records"] == []
    assert report["resolved_false_negative_cohort"]["denominator"] == 0
    metric = report["resolved_false_negative_cohort"]["headline"]["upper_bound_u"]["metrics"][
        "any_gate_open"
    ]
    assert metric == {"successes": 0, "total": 0, "proportion": None, "lower": None, "upper": None}
    gate = report["exact_crossing_gate"]
    assert gate["resolved_cohort_meets_minimum"] is False
    assert gate["crossing_favorable_cohort_meets_minimum"] is False
    assert gate["s3_gate_open"] is False
    assert gate["s3_launched"] is False
    assert "one shared image encoding" in report["one_image_caveat"]


def test_small_cohort_reports_counts_wilson_and_never_launches_s3(sources: az.SourcePaths) -> None:
    report = az.run_analysis(sources)["report"]
    gate = report["exact_crossing_gate"]
    assert gate["resolved_cohort_count"] == 2
    assert gate["u_primary_exact_crossing_favorable_count"] == 1
    assert gate["u_primary_exact_crossing_favorable_wilson_95"]["total"] == 2
    assert gate["resolved_cohort_meets_minimum"] is False
    assert gate["s3_gate_open"] is False
    assert gate["action"] == "stop_s3_branch_for_insufficient_frozen_cohort"
    assert report["claims"]["s3_launched"] is False


def test_create_or_identical_publication_and_self_seal(
    sources: az.SourcePaths, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    argv = [
        "--analysis",
        str(sources.analysis),
        "--owner-summaries",
        str(sources.owner_summaries),
        "--owner-context-features",
        str(sources.owner_context_features),
        "--context-registry",
        str(sources.context_registry),
        "--source-receipt",
        str(sources.source_receipt),
        "--output-dir",
        str(output),
    ]
    assert az.main(argv) == 0
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(first) == az.OUTPUT_NAMES
    assert az.main(argv) == 0
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first

    receipt = json.loads((output / az.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert az._receipt_digest(receipt) == receipt["receipt_content_sha256"]
    for name in (az.REPORT_JSON_NAME, az.REPORT_MD_NAME, az.OWNER_RECORDS_NAME):
        assert az.sha256_bytes((output / name).read_bytes()) == receipt["output_file_sha256"][name]


def test_nonidentical_existing_publication_is_rejected(
    sources: az.SourcePaths, tmp_path: Path
) -> None:
    result = az.run_analysis(sources)
    files = az._materialize_files(result)
    output = tmp_path / "out"
    assert az.publish_create_or_identical(output, files) == "created"
    (output / az.REPORT_MD_NAME).write_text("changed", encoding="utf-8")
    with pytest.raises(az.AnalysisContractError, match="non-identical"):
        az.publish_create_or_identical(output, files)
