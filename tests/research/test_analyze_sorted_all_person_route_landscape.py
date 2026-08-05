from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import analyze_sorted_all_person_route_landscape as analyzer


def _score_row(request_id: str, value: float) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "raw_model_logprob": {"complete_box_logprob_sum": value},
    }


def _candidate(
    candidate_id: str,
    owner_id: str,
    *,
    lower: list[str],
    upper: list[str],
    family: str = "exact",
    transform: str = "exact_gt_anchor",
    box: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_family": family,
        "transform": transform,
        "generator_gt_owner_id": owner_id,
        "strict_assignment_gt_owner_id": owner_id if lower == [owner_id] else None,
        "lower_bound_owner_ids": lower,
        "upper_bound_owner_ids": upper,
        "decoded_bbox_pixel_xyxy": box or [0, 0, 10, 10],
        "coord_token_ids": [1, 2, 3, 4],
    }


def _plan(
    *,
    owners: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
    contexts: list[dict[str, Any]] | None = None,
    sidecars: list[dict[str, Any]] | None = None,
    requests: list[dict[str, Any]] | None = None,
    receipt_path: Path | None = None,
    receipt: dict[str, Any] | None = None,
) -> analyzer.PlanBundle:
    candidates = candidates or []
    contexts = contexts or [
        {
            "context_id": context_id,
            "full_prefix_token_ids_sha256": f"prefix-{context_id}",
            "donor_row_indices": [],
            "donor_rows": [],
        }
        for context_id in analyzer.CONTEXT_IDS
    ]
    sidecars = sidecars or []
    requests = requests or []
    receipt_path = receipt_path or Path("/tmp/not-used-plan-receipt.json")
    receipt = receipt or {
        "receipt_content_sha256": "plan-content",
        "frozen_runtime_identity": {"image_width": 1000, "image_height": 1000},
    }
    return analyzer.PlanBundle(
        plan_dir=receipt_path.parent,
        receipt_path=receipt_path,
        receipt=receipt,
        owners=tuple(owners),
        owners_by_id={str(row["gt_owner_id"]): row for row in owners},
        candidates=tuple(candidates),
        candidates_by_id={str(row["candidate_id"]): row for row in candidates},
        contexts=tuple(contexts),
        contexts_by_id={str(row["context_id"]): row for row in contexts},
        sidecars=tuple(sidecars),
        sidecars_by_id={str(row["sidecar_id"]): row for row in sidecars},
        requests=tuple(requests),
        requests_by_id={str(row["request_id"]): row for row in requests},
        file_paths={},
    )


def _score_bundle(
    rows: dict[str, dict[str, Any]],
    *,
    input_mode: str = "strict_confirmation",
    drift: float = 0.0,
) -> analyzer.ScoreBundle:
    return analyzer.ScoreBundle(
        rows_by_request_id=rows,
        receipt_paths=(),
        score_paths=(),
        receipts=(),
        source_identity_sha256="source",
        input_mode=input_mode,
        scorer_code_lineage=(),
        batch_score_error_bound_by_context={
            context_id: drift for context_id in analyzer.CONTEXT_IDS
        },
    )


def test_epsilon_uses_eight_repeat_median_deviation_and_floor() -> None:
    requests = [
        {
            "request_id": f"repeat-{index}",
            "request_kind": "numerical_repeat",
            "repeat_index": index,
            "context_id": "self-due-gt17",
            "candidate_id": "primary:gt:7511:2:00:exact_gt_anchor",
        }
        for index in range(8)
    ]
    plan = _plan(owners=[], requests=requests)
    values = [1.0, 1.0, 1.0, 1.0, 1.0002, 1.0, 1.0, 1.0]
    scores = _score_bundle(
        {
            request["request_id"]: _score_row(request["request_id"], value)
            for request, value in zip(requests, values, strict=True)
        }
    )

    receipt = analyzer.compute_epsilon(plan, scores)

    assert receipt["median"] == 1.0
    assert receipt["delta"] == pytest.approx(0.0002)
    assert receipt["epsilon"] == pytest.approx(0.0004)

    floor_scores = _score_bundle(
        {request["request_id"]: _score_row(request["request_id"], 3.0) for request in requests}
    )
    assert analyzer.compute_epsilon(plan, floor_scores)["epsilon"] == 1e-6


def test_tolerance_competition_rank_and_ties_follow_frozen_inequalities() -> None:
    values = {"a": 10.0, "b": 9.95, "c": 8.0}

    assert analyzer._competition_rank("a", values, 0.1) == 1  # noqa: SLF001
    assert analyzer._competition_rank("b", values, 0.1) == 1  # noqa: SLF001
    assert analyzer._tied_owner_ids("a", values, 0.1) == ["a", "b"]  # noqa: SLF001
    assert analyzer._best_other("a", values, 0.1) == (9.95, ["b"])  # noqa: SLF001
    assert analyzer._delta_sign(0.2, 0.1) == "tied"  # noqa: SLF001
    assert analyzer._delta_sign(0.200001, 0.1) == "positive"  # noqa: SLF001


def test_exact_neighborhood_and_ambiguity_bounds_remain_separate() -> None:
    owners = [
        {"gt_owner_id": owner_id, "bbox_pixel_xyxy": box, "original_annotation_index": index}
        for index, (owner_id, box) in enumerate(
            [
                ("gt:7511:2", [0, 0, 10, 10]),
                ("gt:7511:3", [20, 0, 30, 10]),
                ("gt:7511:4", [40, 0, 50, 10]),
            ]
        )
    ]
    candidates = [
        _candidate(
            "a-exact",
            "gt:7511:2",
            lower=["gt:7511:2"],
            upper=["gt:7511:2"],
            box=[0, 0, 10, 10],
        ),
        _candidate(
            "b-exact",
            "gt:7511:3",
            lower=["gt:7511:3"],
            upper=["gt:7511:3"],
            box=[20, 0, 30, 10],
        ),
        _candidate(
            "c-exact",
            "gt:7511:4",
            lower=["gt:7511:4"],
            upper=["gt:7511:4"],
            box=[40, 0, 50, 10],
        ),
        _candidate(
            "ambiguous",
            "gt:7511:2",
            lower=[],
            upper=["gt:7511:2", "gt:7511:3"],
            family="translation",
            transform="translate_right",
            box=[10, 0, 20, 10],
        ),
    ]
    plan = _plan(owners=owners, candidates=candidates)
    rows: dict[str, dict[str, Any]] = {}
    for context_id in analyzer.CONTEXT_IDS:
        for candidate_id, value in {
            "a-exact": 10.0,
            "b-exact": 9.0,
            "c-exact": 8.0,
            "ambiguous": 11.0,
        }.items():
            request_id = f"score:{context_id}:{candidate_id}"
            rows[request_id] = _score_row(request_id, value)

    contexts = analyzer.build_context_statistics(plan, _score_bundle(rows), epsilon=0.1)
    a = contexts["root"]["gt:7511:2"]

    assert a["exact_anchor"]["score"] == 10.0
    assert a["exact_anchor"]["competition_rank"] == 1
    assert a["neighborhood"]["score"] == 10.0
    assert a["neighborhood"]["margin"] == 1.0
    assert a["ambiguity_bounds"]["lower_score"] == 10.0
    assert a["ambiguity_bounds"]["upper_score"] == 11.0
    assert a["ambiguity_bounds"]["best_rank"] == 1
    # Frozen formula counts every j, including the owner's own U when U > L + epsilon.
    assert a["ambiguity_bounds"]["worst_rank"] == 3
    assert a["ambiguity_bounds"]["lower_margin"] == -1.0
    assert a["ambiguity_bounds"]["upper_margin"] == 2.0
    assert a["ambiguity_bounds"]["status"] == "ambiguity_sensitive"
    assert a["peak"]["winning_candidate_ids"] == ["a-exact"]


def test_batch_drift_marks_near_tie_rank_and_margin_unknown() -> None:
    owners = [
        {"gt_owner_id": "a", "bbox_pixel_xyxy": [0, 0, 10, 10]},
        {"gt_owner_id": "b", "bbox_pixel_xyxy": [20, 0, 30, 10]},
    ]
    plan = _plan(owners=owners)
    contexts: dict[str, dict[str, dict[str, Any]]] = {}
    for context_id in analyzer.CONTEXT_IDS:
        contexts[context_id] = {}
        for owner_id, score, margin in [("a", 10.0, 0.0001), ("b", 9.9999, -0.0001)]:
            contexts[context_id][owner_id] = {
                "exact_anchor": {"score": score},
                "neighborhood": {"score": score, "margin": margin},
                "ambiguity_bounds": {
                    "lower_score": score,
                    "upper_score": score,
                    "lower_margin": margin,
                    "upper_margin": margin,
                },
                "peak": {"concentration": {"top1_minus_top2_score": 0.0001}},
                "realized_sidecars": {"entries": []},
            }
    scores = _score_bundle({}, input_mode="raw_capture_salvage", drift=0.001)

    admission = analyzer.attach_batch_scalar_confirmation(
        plan, contexts, scores, epsilon=1e-6
    )

    assert admission["root"]["status"] == "unresolved_needs_scalar_confirmation"
    confirmation = contexts["root"]["a"]["batch_scalar_confirmation"]
    assert confirmation["exact_anchor"]["status"] == "unknown_needs_scalar_confirmation"
    assert (
        confirmation["neighborhood_rank_and_margin"]["status"]
        == "unknown_needs_scalar_confirmation"
    )
    assert contexts["root"]["a"]["peak"]["batch_scalar_confirmation"]["status"] == (
        "unknown_needs_scalar_confirmation"
    )


def test_selectivity_uses_controls_absent_from_both_prefixes_and_ambiguity_bounds() -> None:
    owner_ids = ["target", "control-a", "control-b"]
    owners = [
        {"gt_owner_id": owner_id, "bbox_pixel_xyxy": [index * 20, 0, index * 20 + 10, 10]}
        for index, owner_id in enumerate(owner_ids)
    ]
    plan = _plan(owners=owners)
    contexts: dict[str, dict[str, dict[str, Any]]] = {
        context_id: {} for context_id in analyzer.CONTEXT_IDS
    }
    for context_id in analyzer.CONTEXT_IDS:
        for owner_id in owner_ids:
            contexts[context_id][owner_id] = {
                "neighborhood": {"margin": 0.0},
                "ambiguity_bounds": {
                    "lower_margin": 0.0,
                    "upper_margin": 0.0,
                    "outcome_invariant": True,
                },
            }
    # The frozen function is specifically centered on gt17, so replace target.
    plan_owner = {
        "gt_owner_id": "gt:7511:17",
        "bbox_pixel_xyxy": [0, 0, 10, 10],
    }
    plan = _plan(
        owners=[plan_owner, owners[1], owners[2]],
    )
    for context_id in analyzer.CONTEXT_IDS:
        contexts[context_id]["gt:7511:17"] = contexts[context_id].pop("target")
    due = contexts["self-due-gt17"]
    post = contexts["skip-post-gt17"]
    post["gt:7511:17"]["neighborhood"]["margin"] = -5.0
    post["gt:7511:17"]["ambiguity_bounds"].update(
        {"lower_margin": -5.0, "upper_margin": -5.0}
    )
    post["control-a"]["neighborhood"]["margin"] = -1.0
    post["control-a"]["ambiguity_bounds"].update(
        {"lower_margin": -1.0, "upper_margin": -1.0}
    )
    post["control-b"]["neighborhood"]["margin"] = -2.0
    post["control-b"]["ambiguity_bounds"].update(
        {"lower_margin": -2.0, "upper_margin": -2.0}
    )
    assert due["gt:7511:17"]["neighborhood"]["margin"] == 0.0

    result = analyzer.compute_selectivity(
        plan,
        contexts,
        epsilon=0.1,
        score_error_bound_by_context={context_id: 0.0 for context_id in analyzer.CONTEXT_IDS},
    )

    assert result["control_owner_ids"] == ["control-a", "control-b"]
    assert result["target_D"] == -5.0
    assert result["point_inequality_passed"] is True
    assert result["status"] == "prefix_content_sensitive_descriptive"
    assert result["batch_scalar_confirmation"]["status"].startswith(
        "prefix_content_sensitive_descriptive"
    )


def test_scan_statistic_uses_last_complete_person_and_average_tie_ranks() -> None:
    owners = [
        {"gt_owner_id": "a", "bbox_pixel_xyxy": [0, 100, 10, 110]},
        {"gt_owner_id": "b", "bbox_pixel_xyxy": [0, 300, 10, 310]},
        {"gt_owner_id": "c", "bbox_pixel_xyxy": [0, 500, 10, 510]},
    ]
    contexts = [
        {
            "context_id": context_id,
            "full_prefix_token_ids_sha256": context_id,
            "donor_row_indices": [] if context_id == "root" else [0, 1],
            "donor_rows": (
                []
                if context_id == "root"
                else [
                    {
                        "row_index": 0,
                        "raw_span_text": "<|object_ref_start|>person<|object_ref_end|><|coord_0|><|coord_200|><|coord_10|><|coord_210|>",
                    },
                    {
                        "row_index": 1,
                        "raw_span_text": "<|object_ref_start|>person<|object_ref_end|><|coord_0|><|coord_400|><|coord_10|><|coord_410|>",
                    },
                ]
            ),
        }
        for context_id in analyzer.CONTEXT_IDS
    ]
    plan = _plan(owners=owners, contexts=contexts)
    context_stats = {
        context_id: {
            owner_id: {"neighborhood": {"score": score}}
            for owner_id, score in [("a", 3.0), ("b", 3.0), ("c", 1.0)]
        }
        for context_id in analyzer.CONTEXT_IDS
    }

    result = analyzer.build_scan_statistics(plan, context_stats, epsilon=0.1)

    assert result["root"]["status"] == "not_applicable_root_has_no_frontier"
    cell = result["self-due-gt17"]
    assert cell["frontier"]["donor_row_index"] == 1
    assert cell["per_owner_distances"]["b"]["signed_ordinal_distance"] == -1
    assert cell["per_owner_distances"]["c"]["signed_ordinal_distance"] == 1
    assert cell["eligible_owner_count"] == 3
    assert cell["spearman_owner_score_rank_vs_absolute_scan_distance"] is not None


def _sealed_score_fixture(
    tmp_path: Path,
    *,
    declared_code_sha256: str,
) -> tuple[analyzer.PlanBundle, Path]:
    plan_receipt_path = tmp_path / "plan-receipt.json"
    plan_receipt_path.write_text("{}\n", encoding="utf-8")
    candidate = _candidate("candidate", "owner", lower=["owner"], upper=["owner"])
    request = {
        "request_id": "request",
        "request_kind": "primary",
        "context_id": "root",
        "candidate_id": "candidate",
    }
    plan = _plan(
        owners=[{"gt_owner_id": "owner", "bbox_pixel_xyxy": [0, 0, 10, 10]}],
        candidates=[candidate],
        requests=[request],
        receipt_path=plan_receipt_path,
        receipt={"receipt_content_sha256": "plan-content"},
    )
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    row = {
        "schema_version": analyzer.SCORE_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "request_id": "request",
        "request_kind": "primary",
        "context_id": "root",
        "candidate_id": "candidate",
        "sidecar_id": None,
        "repeat_index": None,
        "coord_token_ids": [1, 2, 3, 4],
        "coord_token_ids_sha256": analyzer.sha256_json([1, 2, 3, 4]),
        "full_prefix_token_ids_sha256": "prefix-root",
        "native_repetition_penalty_stratum": 1.0,
        "raw_model_logprob": {"complete_box_logprob_sum": -1.0},
        "decision_bearing_channel": analyzer.DECISION_CHANNEL,
        "primary_role": True,
        "excluded_from_primary_ranks": False,
    }
    (shard_dir / analyzer.SCORE_NAME).write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    receipt: dict[str, Any] = {
        "schema_version": analyzer.SCORE_RECEIPT_SCHEMA_VERSION,
        "row_schema_version": analyzer.SCORE_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "code": {"path": str(Path(analyzer.__file__).resolve()), "sha256": declared_code_sha256},
        "plan": {
            "receipt_path": str(plan_receipt_path),
            "receipt_sha256": analyzer.sha256_file(plan_receipt_path),
            "receipt_content_sha256": "plan-content",
        },
        "source_identity": {"runtime": "same"},
        "selection": {"shard_request_ids": ["request"]},
        "scoring_backend_admission": {
            "selected_backend": analyzer.FULL_REFORWARD_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            "contexts_scored": ["root"],
            "per_context_accounting": [
                {
                    "context_id": "root",
                    "batch_admission": {
                        "status": "not_requested",
                        "effective_batch_size": 1,
                    },
                }
            ],
        },
        "counts": {"rows": 1},
        "row_ids": ["request"],
    }
    receipt["receipt_content_sha256"] = analyzer.sha256_json(receipt)
    (shard_dir / analyzer.SCORE_RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return plan, shard_dir


def _write_scalar_overlay_shard(
    shard_dir: Path,
    *,
    plan: analyzer.PlanBundle,
    row: dict[str, Any],
    source_identity: dict[str, Any] | None = None,
    schema_version: str = analyzer.SCORE_RECEIPT_SCHEMA_VERSION,
    executed_code_sha256: str = "a" * 64,
    requested_batch_size: int = 1,
) -> None:
    shard_dir.mkdir()
    (shard_dir / analyzer.SCORE_NAME).write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    receipt: dict[str, Any] = {
        "schema_version": schema_version,
        "row_schema_version": analyzer.SCORE_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "code": {
            "path": str(Path(analyzer.__file__).resolve()),
            "executed_source_sha256": executed_code_sha256,
            "receipt_time_file_sha256": executed_code_sha256,
            "source_drift_detected": False,
            "sha256": executed_code_sha256,
        },
        "plan": {
            "receipt_path": str(plan.receipt_path),
            "receipt_sha256": analyzer.sha256_file(plan.receipt_path),
            "receipt_content_sha256": plan.receipt["receipt_content_sha256"],
        },
        "source_identity": source_identity or {"runtime": "same"},
        "selection": {"shard_request_ids": [row["request_id"]]},
        "scoring_backend_admission": {
            "selected_backend": analyzer.FULL_REFORWARD_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            "requested_batch_size": requested_batch_size,
            "contexts_scored": [row["context_id"]],
            "per_context_accounting": [
                {
                    "context_id": row["context_id"],
                    "batch_admission": {
                        "status": "not_requested",
                        "requested_batch_size": requested_batch_size,
                        "effective_batch_size": requested_batch_size,
                    },
                }
            ],
        },
        "counts": {"rows": 1},
        "row_ids": [row["request_id"]],
    }
    receipt["receipt_content_sha256"] = analyzer.sha256_json(receipt)
    (shard_dir / analyzer.SCORE_RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_salvage_preserves_stale_code_hash_but_strict_confirmation_rejects_it(
    tmp_path: Path,
) -> None:
    plan, shard_dir = _sealed_score_fixture(tmp_path, declared_code_sha256="0" * 64)

    salvaged = analyzer.load_score_shards(
        plan, [shard_dir], input_mode="raw_capture_salvage"
    )

    assert salvaged.input_mode == "raw_capture_salvage"
    assert salvaged.scorer_code_lineage[0]["declared_code_sha256"] == "0" * 64
    assert salvaged.scorer_code_lineage[0]["live_matches_declared"] is False
    assert salvaged.scorer_code_lineage[0]["score_sha256"] == analyzer.sha256_file(
        shard_dir / analyzer.SCORE_NAME
    )
    with pytest.raises(analyzer.AnalysisContractError, match="source hash"):
        analyzer.load_score_shards(plan, [shard_dir], input_mode="strict_confirmation")


def test_exact_coverage_never_relaxes_in_salvage_mode(tmp_path: Path) -> None:
    plan, shard_dir = _sealed_score_fixture(
        tmp_path, declared_code_sha256=analyzer.sha256_file(Path(analyzer.__file__).resolve())
    )
    missing_request = {**plan.requests[0], "request_id": "missing-request"}
    plan = analyzer.PlanBundle(
        **{
            **plan.__dict__,
            "requests": (*plan.requests, missing_request),
            "requests_by_id": {**plan.requests_by_id, "missing-request": missing_request},
        }
    )

    with pytest.raises(analyzer.AnalysisContractError, match="exact complete plan coverage"):
        analyzer.load_score_shards(plan, [shard_dir], input_mode="raw_capture_salvage")


def test_scalar_confirmation_manifest_has_all_41_exact_anchors_per_context() -> None:
    owners = [
        {
            "gt_owner_id": f"gt:7511:{index}",
            "bbox_pixel_xyxy": [index * 2, 0, index * 2 + 1, 1],
        }
        for index in range(2, 43)
    ]
    candidates = [
        _candidate(
            f"exact-{owner['gt_owner_id']}",
            str(owner["gt_owner_id"]),
            lower=[str(owner["gt_owner_id"])],
            upper=[str(owner["gt_owner_id"])],
        )
        for owner in owners
    ]
    requests = [
        {
            "request_id": f"score:{context_id}:{candidate['candidate_id']}",
            "request_kind": "primary",
            "context_id": context_id,
            "candidate_id": candidate["candidate_id"],
        }
        for context_id in analyzer.CONTEXT_IDS
        for candidate in candidates
    ]
    plan = _plan(owners=owners, candidates=candidates, requests=requests)
    score_rows = {
        str(request["request_id"]): _score_row(
            str(request["request_id"]), -float(index)
        )
        for index, request in enumerate(requests)
    }
    scores = _score_bundle(score_rows)
    contexts = {
        context_id: {
            str(owner["gt_owner_id"]): {
                "neighborhood": {"score": -float(owner_index)},
                "batch_scalar_confirmation": {
                    "neighborhood_rank_and_margin": {
                        "status": "confirmed_within_batch_scalar_bound"
                    }
                },
                "peak": {
                    "batch_scalar_confirmation": {
                        "status": "confirmed_within_batch_scalar_bound"
                    }
                },
                "realized_sidecars": {"entries": []},
            }
            for owner_index, owner in enumerate(owners)
        }
        for context_id in analyzer.CONTEXT_IDS
    }

    manifest = analyzer.build_scalar_confirmation_manifest(
        plan, scores, contexts, epsilon=1e-6
    )

    assert len(manifest) == 41 * len(analyzer.CONTEXT_IDS)
    assert manifest == sorted(manifest, key=lambda row: (row["context_id"], row["request_id"]))
    for context_id in analyzer.CONTEXT_IDS:
        selected = [row for row in manifest if row["context_id"] == context_id]
        assert len(selected) == 41
        assert all("all_41_exact_anchors" in row["selection_reasons"] for row in selected)


def test_scalar_overlay_v2_replaces_legacy_row_and_reports_full_decision_drift(
    tmp_path: Path,
) -> None:
    plan, legacy_dir = _sealed_score_fixture(
        tmp_path, declared_code_sha256=analyzer.sha256_file(Path(analyzer.__file__).resolve())
    )
    legacy = analyzer.load_score_shards(
        plan, [legacy_dir], input_mode="raw_capture_salvage"
    )
    scalar_row = dict(legacy.rows_by_request_id["request"])
    scalar_row["raw_model_logprob"] = {"complete_box_logprob_sum": -2.0}
    scalar_dir = tmp_path / "scalar"
    _write_scalar_overlay_shard(scalar_dir, plan=plan, row=scalar_row)
    manifest = [{"context_id": "root", "request_id": "request"}]

    overlaid = analyzer.apply_scalar_confirmation_overlay(
        plan, legacy, [scalar_dir], manifest=manifest
    )
    admission = analyzer.build_scalar_overlay_admission(
        raw_manifest=manifest,
        recomputed_manifest=manifest,
        scores=overlaid,
    )

    assert analyzer.score_value(overlaid.rows_by_request_id["request"]) == -2.0
    assert analyzer.score_value(overlaid.legacy_rows_by_request_id["request"]) == -1.0
    assert overlaid.scalar_overlay_request_ids == {"request"}
    assert admission["decision_admission"] == "fully_scalar_confirmed_decision"
    assert admission["scalar_vs_legacy_batch_drift"]["max_absolute"] == 1.0
    assert admission["per_context"]["root"]["status"] == (
        "fully_scalar_confirmed_decision"
    )


def test_scalar_overlay_rejects_nonmanifest_duplicate_and_non_scalar_receipts(
    tmp_path: Path,
) -> None:
    plan, legacy_dir = _sealed_score_fixture(
        tmp_path, declared_code_sha256=analyzer.sha256_file(Path(analyzer.__file__).resolve())
    )
    legacy = analyzer.load_score_shards(
        plan, [legacy_dir], input_mode="raw_capture_salvage"
    )
    scalar_row = dict(legacy.rows_by_request_id["request"])
    scalar_dir = tmp_path / "scalar"
    _write_scalar_overlay_shard(scalar_dir, plan=plan, row=scalar_row)

    with pytest.raises(analyzer.AnalysisContractError, match="not a subset"):
        analyzer.apply_scalar_confirmation_overlay(plan, legacy, [scalar_dir], manifest=[])
    with pytest.raises(analyzer.AnalysisContractError, match="supplied twice"):
        analyzer.apply_scalar_confirmation_overlay(
            plan,
            legacy,
            [scalar_dir, scalar_dir],
            manifest=[{"request_id": "request"}],
        )

    bad_batch_dir = tmp_path / "bad-batch"
    _write_scalar_overlay_shard(
        bad_batch_dir,
        plan=plan,
        row=scalar_row,
        requested_batch_size=2,
    )
    with pytest.raises(analyzer.AnalysisContractError, match="batch-size one"):
        analyzer.apply_scalar_confirmation_overlay(
            plan,
            legacy,
            [bad_batch_dir],
            manifest=[{"request_id": "request"}],
        )

    v1_dir = tmp_path / "v1"
    _write_scalar_overlay_shard(
        v1_dir,
        plan=plan,
        row=scalar_row,
        schema_version=analyzer.LEGACY_SCORE_RECEIPT_SCHEMA_VERSION,
    )
    with pytest.raises(analyzer.AnalysisContractError, match="schema v2"):
        analyzer.apply_scalar_confirmation_overlay(
            plan,
            legacy,
            [v1_dir],
            manifest=[{"request_id": "request"}],
        )


def test_partial_overlay_is_never_labeled_full_confirmation() -> None:
    rows = {
        "one": {"context_id": "root", "raw_model_logprob": {"complete_box_logprob_sum": 1.1}},
        "two": {"context_id": "root", "raw_model_logprob": {"complete_box_logprob_sum": 2.0}},
    }
    legacy_rows = {
        "one": {"context_id": "root", "raw_model_logprob": {"complete_box_logprob_sum": 1.0}},
        "two": {"context_id": "root", "raw_model_logprob": {"complete_box_logprob_sum": 2.0}},
    }
    scores = analyzer.ScoreBundle(
        rows_by_request_id=rows,
        receipt_paths=(),
        score_paths=(),
        receipts=(),
        source_identity_sha256="source",
        input_mode="raw_capture_salvage",
        scorer_code_lineage=(),
        batch_score_error_bound_by_context={context_id: 0.1 for context_id in analyzer.CONTEXT_IDS},
        legacy_rows_by_request_id=legacy_rows,
        scalar_overlay_request_ids=frozenset({"one"}),
        scalar_overlay_lineage=(
            {"code": {"executed_source_sha256": "a" * 64}},
        ),
    )
    raw_manifest = [
        {"context_id": "root", "request_id": "one"},
        {"context_id": "root", "request_id": "two"},
    ]

    admission = analyzer.build_scalar_overlay_admission(
        raw_manifest=raw_manifest,
        recomputed_manifest=raw_manifest,
        scores=scores,
    )

    assert admission["decision_admission"] == "partial_scalar_overlay_bounded_decision"
    assert admission["per_context"]["root"]["status"] == (
        "partial_scalar_overlay_bounded_decision"
    )
    assert admission["missing_original_manifest_request_ids"] == ["two"]


def test_full_scalar_overlay_promotes_nested_decision_statuses_and_preserves_legacy() -> None:
    contexts = {
        "root": {
            "owner": {
                "decision_score_provenance": {
                    "status": "decision_support_scalar_overlaid",
                    "scalar_request_ids": ["score:root:owner"],
                    "legacy_batch_request_ids": [],
                },
                "batch_scalar_confirmation": {
                    "exact_anchor": {"status": "unknown_needs_scalar_confirmation"},
                    "neighborhood_rank_and_margin": {
                        "status": "unknown_needs_scalar_confirmation",
                        "scalar_margin_lower_bound": -0.1,
                        "scalar_margin_upper_bound": 0.1,
                    },
                },
                "ambiguity_bounds": {
                    "batch_scalar_confirmation": {
                        "status": "unknown_needs_scalar_confirmation",
                        "lower_margin_scalar_interval": [-0.1, 0.1],
                    }
                },
                "peak": {
                    "batch_scalar_confirmation": {
                        "status": "unknown_needs_scalar_confirmation",
                        "minimum_confirming_top1_minus_top2": 0.1,
                    }
                },
            }
        }
    }
    batch_admission = {
        "root": {
            "status": "unresolved_needs_scalar_confirmation",
            "complete_box_score_error_bound": 0.001,
            "unknown_needs_scalar_confirmation_owner_ids": ["owner"],
        }
    }
    overlay_admission = {
        "per_context": {
            context_id: {
                "status": (
                    "fully_scalar_confirmed_decision"
                    if context_id == "root"
                    else "discovery_only_raw"
                )
            }
            for context_id in analyzer.CONTEXT_IDS
        }
    }

    analyzer.promote_scalar_confirmed_decision_statuses(
        contexts,
        batch_admission,
        overlay_admission,
    )

    assert batch_admission["root"]["status"] == "scalar_confirmed_decision_support"
    assert batch_admission["root"]["legacy_batch_bound_status"] == (
        "unresolved_needs_scalar_confirmation"
    )
    assert batch_admission["root"]["complete_box_score_error_bound"] == 0.001
    assert batch_admission["root"]["unknown_needs_scalar_confirmation_owner_ids"] == []
    assert batch_admission["root"]["legacy_batch_bound"][
        "unknown_needs_scalar_confirmation_owner_ids"
    ] == ["owner"]
    cell = contexts["root"]["owner"]
    for confirmation in (
        cell["batch_scalar_confirmation"]["exact_anchor"],
        cell["batch_scalar_confirmation"]["neighborhood_rank_and_margin"],
        cell["ambiguity_bounds"]["batch_scalar_confirmation"],
        cell["peak"]["batch_scalar_confirmation"],
    ):
        assert confirmation["status"] == "scalar_confirmed_decision_support"
        assert confirmation["legacy_batch_bound_status"] == (
            "unknown_needs_scalar_confirmation"
        )
        assert confirmation["legacy_batch_bound"]["status"] == (
            "unknown_needs_scalar_confirmation"
        )


def test_context_delta_uses_scalar_overlay_and_retains_legacy_batch_interval() -> None:
    before = {
        "exact_anchor": {"score": 1.0},
        "neighborhood": {"score": 2.0, "competition_rank": 2, "margin": -0.5},
        "ambiguity_bounds": {"lower_margin": -0.6, "upper_margin": -0.4},
    }
    after = {
        "exact_anchor": {"score": 1.25},
        "neighborhood": {"score": 3.0, "competition_rank": 1, "margin": 0.75},
        "ambiguity_bounds": {"lower_margin": 0.7, "upper_margin": 0.8},
    }

    delta = analyzer._context_delta(
        before,
        after,
        epsilon=1e-6,
        before_score_error_bound=0.1,
        after_score_error_bound=0.2,
        before_decision_admission="fully_scalar_confirmed_decision",
        after_decision_admission="fully_scalar_confirmed_decision",
    )

    confirmation = delta["batch_scalar_confirmation"]
    assert confirmation["status"] == "scalar_confirmed_delta"
    assert confirmation["score_delta_error_bound"] == 0.0
    assert confirmation["owner_relative_margin_delta_lower_bound"] == 1.25
    assert confirmation["owner_relative_margin_delta_upper_bound"] == 1.25
    assert confirmation["legacy_batch_bound"]["status"] == (
        "unknown_needs_scalar_confirmation_for_near_threshold_deltas"
    )
    assert confirmation["legacy_batch_bound"]["score_delta_error_bound"] == pytest.approx(0.3)

    partial = analyzer._context_delta(
        before,
        after,
        epsilon=1e-6,
        before_score_error_bound=0.1,
        after_score_error_bound=0.2,
        before_decision_admission="fully_scalar_confirmed_decision",
        after_decision_admission="partial_scalar_overlay_bounded_decision",
    )
    assert partial["batch_scalar_confirmation"]["status"] == (
        "unknown_needs_scalar_confirmation_for_near_threshold_deltas"
    )
    assert "legacy_batch_bound" not in partial["batch_scalar_confirmation"]
