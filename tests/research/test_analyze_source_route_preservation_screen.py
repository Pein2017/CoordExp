"""Focused contract tests for the July 22 preservation-screen analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.analyze_source_route_preservation_screen import (
    CONFIG_BASENAMES,
    RUN_NAMES,
    STATE_BANK_ARM_KEYS,
    STATE_BANK_FAMILIES,
    ScreenAnalysisError,
    analyze_source_route_preservation_screen,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _row(
    image_id: int,
    pred: list[dict[str, object]],
    *,
    stop_reason: str = "im_end",
    dropped_prediction_count: int = 0,
) -> dict[str, object]:
    if image_id == 1:
        gt = [
            {"object_id": "person", "description": "person", "bbox": [0, 0, 500, 500]},
            {"object_id": "cat", "description": "cat", "bbox": [500, 500, 999, 999]},
        ]
    else:
        gt = [{"object_id": "dog", "description": "dog", "bbox": [0, 500, 500, 999]}]
    return {
        "row_id": f"coco2017_val_{image_id:012d}",
        "image_path": f"/tmp/{image_id:012d}.jpg",
        "image_width": 100,
        "image_height": 100,
        "gt": gt,
        "pred": pred,
        "parse_status": "accepted_with_drops" if dropped_prediction_count else "accepted",
        "dropped_prediction_count": dropped_prediction_count,
        "dropped_predictions": [{}] * dropped_prediction_count,
        "decode_stop_reason": stop_reason,
    }


def _prediction(description: str, bbox: list[int]) -> dict[str, object]:
    return {"description": description, "bbox": bbox}


def _write_arm(root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path, Path]:
    gt_vs_pred = root / "gt_vs_pred.jsonl"
    diagnostics = root / "parse_diagnostics.jsonl"
    summary = root / "summary.json"
    _write_jsonl(gt_vs_pred, rows)
    diagnostic_rows = [
        {
            "row_id": row["row_id"],
            "parse_status": row["parse_status"],
            "valid_prediction_count": len(row["pred"]),
            "dropped_prediction_count": row["dropped_prediction_count"],
            "dropped_predictions": row["dropped_predictions"],
        }
        for row in rows
    ]
    _write_jsonl(diagnostics, diagnostic_rows)
    stop_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["decode_stop_reason"])
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    _write_json(
        summary,
        {
            "terminal_status": "completed",
            "row_count": len(rows),
            "raw_row_count": len(rows),
            "decode_success_count": len(rows),
            "parser_failure_count": 0,
            "dropped_prediction_count": sum(int(row["dropped_prediction_count"]) for row in rows),
            "truncated_decode_count": stop_reasons.get("length", 0),
            "decode_stop_reasons": stop_reasons,
        },
    )
    return gt_vs_pred, summary, diagnostics


def _write_run(
    root: Path,
    label: str,
    rows: list[dict[str, object]],
    frozen_input: Path,
    model_paths: tuple[Path, Path],
) -> Path:
    gt_vs_pred, _, _ = _write_arm(root, rows)
    _write_jsonl(root / "gt_vs_pred_scored.jsonl", rows)
    adapter, embedding = model_paths
    generation = {
        "do_sample": False,
        "max_new_tokens": 3084,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    model_identity = {
        "adapter": {"adapter_path": str(adapter.resolve())},
        "embedding_delta": {"identity": {"delta_path": str(embedding.resolve())}},
    }
    manifest = {
        "generation_policy": generation,
        "dataset_identity": {"dataset": "frozen-train-256", "rows": 256},
        "model_identity": model_identity,
        "model_identity_fingerprint": f"{label}-model-fingerprint",
        "adapter_identity": {"fingerprint": f"{label}-adapter"},
        "embedding_delta_identity": {"fingerprint": f"{label}-embedding"},
    }
    _write_json(root / "run_manifest.json", manifest)
    row_ids = [str(row["row_id"]) for row in rows]
    _write_json(
        root / "gt_vs_pred_scored.jsonl.provenance.json",
        {
            "raw_artifact": {"path": "gt_vs_pred.jsonl", "sha256": hashlib.sha256(gt_vs_pred.read_bytes()).hexdigest()},
            "scored_artifact": {
                "path": "gt_vs_pred_scored.jsonl",
                "sha256": hashlib.sha256((root / "gt_vs_pred_scored.jsonl").read_bytes()).hexdigest(),
            },
            "row_binding": {
                "row_count": len(row_ids),
                "row_ids_sha256": hashlib.sha256(
                    json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            **{
                field: manifest[field]
                for field in (
                    "model_identity",
                    "model_identity_fingerprint",
                    "adapter_identity",
                    "embedding_delta_identity",
                    "generation_policy",
                )
            },
        },
    )
    _write_json(
        root / "configs" / "resolved.json",
        {
            "config": {
                "run": {"name": RUN_NAMES[label]},
                "data": {"input_jsonl": str(frozen_input.resolve())},
                "generation": {key: value for key, value in generation.items() if key != "do_sample"},
                "adapter": {"path": str(adapter.resolve())},
                "embedding_delta": {"path": str(embedding.resolve())},
            },
            "resolution": {
                "entry_config_path": str((root / "configs" / CONFIG_BASENAMES[label]).resolve()),
                "path_origins": {"data.input_jsonl": {"resolved_path": str(frozen_input.resolve())}},
            },
        },
    )
    return root


def _write_state_bank(root: Path, label: str, owner_id: str) -> Path:
    records = [
        {
            "event_id": f"event-{index}",
            "image": {"image_id": 1},
            "positive_path_imitation_eligible": index == 0,
            "candidates": ([{"role": "positive", "physical_owner_id": owner_id}] if index == 0 else []),
        }
        for index in range(992)
    ]
    records_path = root / "state-bank" / "records.jsonl"
    _write_jsonl(records_path, records)
    manifest_path = root / "state-bank" / "manifest.json"
    _write_json(
        manifest_path,
        {
            "records_file": "records.jsonl",
            "record_count": 992,
            "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        },
    )
    receipt_path = root / "assembly-receipt.json"
    _write_json(
        receipt_path,
        {
            "arm": STATE_BANK_FAMILIES[label],
            "counts": {"rollout_rows": 992},
            "state_bank_manifest_path": str(manifest_path.resolve()),
        },
    )
    return receipt_path


def _screen_inputs(tmp_path: Path) -> dict[str, Path]:
    source_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []
    multi_rows: list[dict[str, object]] = []
    for image_id in range(1, 119):
        if image_id == 1:
            source_rows.append(_row(image_id, [_prediction("person", [0, 0, 50, 50])]))
            single_rows.append(
                _row(
                    image_id,
                    [
                        _prediction("cat", [50, 50, 100, 100]),
                        _prediction("cat", [50, 50, 100, 100]),
                    ],
                    stop_reason="length",
                )
            )
            multi_rows.append(
                _row(
                    image_id,
                    [
                        _prediction("person", [0, 0, 50, 50]),
                        _prediction("cat", [50, 50, 100, 100]),
                    ],
                )
            )
            continue
        source_box = [1, 50, 51, 100] if image_id == 2 else [0, 50, 50, 100]
        source_rows.append(_row(image_id, [_prediction("dog", source_box)]))
        single_rows.append(_row(image_id, [_prediction("dog", source_box)]))
        multi_rows.append(_row(image_id, [_prediction("dog", [0, 50, 50, 100])]))
    source_rows.append(
        _row(999, [_prediction("dog", [0, 50, 50, 100])], dropped_prediction_count=1)
    )
    single_rows.append(_row(999, [_prediction("dog", [0, 50, 50, 100])]))
    multi_rows.append(_row(999, [_prediction("dog", [0, 50, 50, 100])]))
    source, source_summary, source_diagnostics = _write_arm(tmp_path / "source", source_rows)
    single, single_summary, single_diagnostics = _write_arm(tmp_path / "single", single_rows)
    multi, multi_summary, multi_diagnostics = _write_arm(tmp_path / "multi", multi_rows)
    receipt = tmp_path / "state-banks-v2" / "assembly-receipt.json"
    _write_json(receipt, {"image_count": 118, "image_ids": list(range(1, 119))})
    return {
        "selection": receipt,
        "source": source,
        "source_summary": source_summary,
        "source_diagnostics": source_diagnostics,
        "single": single,
        "single_summary": single_summary,
        "single_diagnostics": single_diagnostics,
        "multi": multi,
        "multi_summary": multi_summary,
        "multi_diagnostics": multi_diagnostics,
    }


def _full_screen_inputs(tmp_path: Path) -> dict[str, object]:
    frozen = tmp_path / "frozen-train-256.jsonl"
    _write_jsonl(frozen, [{"image_id": image_id} for image_id in range(1, 257)])
    model_paths = {
        label: (tmp_path / "models" / label / "adapter", tmp_path / "models" / label / "embedding")
        for label in RUN_NAMES
    }
    source_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []
    multi_rows: list[dict[str, object]] = []
    for image_id in range(1, 257):
        if image_id == 1:
            source_rows.append(_row(image_id, [_prediction("person", [0, 0, 50, 50])]))
            single_rows.append(
                _row(
                    image_id,
                    [_prediction("cat", [50, 50, 100, 100]), _prediction("cat", [50, 50, 100, 100])],
                    stop_reason="length",
                )
            )
            multi_rows.append(
                _row(image_id, [_prediction("person", [0, 0, 50, 50]), _prediction("cat", [50, 50, 100, 100])])
            )
            continue
        source_box = [1, 50, 51, 100] if image_id == 2 else [0, 50, 50, 100]
        source_rows.append(
            _row(
                image_id,
                [_prediction("dog", source_box)],
                dropped_prediction_count=int(image_id == 256),
            )
        )
        single_rows.append(_row(image_id, [_prediction("dog", source_box)]))
        multi_rows.append(_row(image_id, [_prediction("dog", [0, 50, 50, 100])]))
    runs = {
        "source": _write_run(tmp_path / "source-run", "source", source_rows, frozen, model_paths["source"]),
        "single_route": _write_run(
            tmp_path / "single-run", "single_route", single_rows, frozen, model_paths["single_route"]
        ),
        "multi_route": _write_run(
            tmp_path / "multi-run", "multi_route", multi_rows, frozen, model_paths["multi_route"]
        ),
    }
    single_receipt = _write_state_bank(tmp_path / "state-banks-v2" / "single", "single_route", "1:cat")
    multi_receipt = _write_state_bank(tmp_path / "state-banks-v2" / "multi", "multi_route", "1:cat")
    selection = tmp_path / "state-banks-v2" / "assembly-receipt.json"
    _write_json(
        selection,
        {
            "schema_version": "source_preservation_multi_route_state_bank_assembler.v1",
            "status": "assembled",
            "image_count": 118,
            "image_ids": list(range(1, 119)),
            "arms": {
                STATE_BANK_ARM_KEYS["single_route"]: str(single_receipt.resolve()),
                STATE_BANK_ARM_KEYS["multi_route"]: str(multi_receipt.resolve()),
            },
        },
    )
    return {"selection": selection, "frozen": frozen, "runs": runs, "model_paths": model_paths}


def _analyze(paths: dict[str, object], **overrides: object) -> dict[str, object]:
    runs = paths["runs"]
    return analyze_source_route_preservation_screen(
        selection_receipt_path=overrides.get("selection_receipt_path", paths["selection"]),
        frozen_eval_input_path=overrides.get("frozen_eval_input_path", paths["frozen"]),
        source_run_dir=overrides.get("source_run_dir", runs["source"]),
        single_route_run_dir=overrides.get("single_route_run_dir", runs["single_route"]),
        multi_route_run_dir=overrides.get("multi_route_run_dir", runs["multi_route"]),
        expected_model_paths=paths["model_paths"],
    )


def _refresh_raw_provenance(run_dir: Path) -> None:
    raw_path = run_dir / "gt_vs_pred.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    row_ids = [str(row["row_id"]) for row in rows]
    provenance_path = run_dir / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["raw_artifact"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    provenance["row_binding"] = {
        "row_count": len(row_ids),
        "row_ids_sha256": hashlib.sha256(
            json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(provenance_path, provenance)


def test_screen_receipt_reports_scopes_owner_ledgers_and_geometry(tmp_path: Path) -> None:
    receipt = _analyze(_full_screen_inputs(tmp_path))

    assert STATE_BANK_FAMILIES["single_route"] == "single_route_plus_source_preservation"
    assert STATE_BANK_FAMILIES["multi_route"] == "multi_route_plus_source_preservation"
    source = receipt["arms"]["source"]
    assert source["scopes"]["overall"]["image_count"] == 256
    assert source["scopes"]["overall"]["gt_owner_count"] == 257
    assert source["scopes"]["admitted"]["gt_owner_count"] == 119
    assert source["scopes"]["non_admitted"]["image_count"] == 138
    assert source["scopes"]["admitted"]["owner_false_negative_ids"] == ["1:cat"]
    assert source["parser_and_drop"]["dropped_prediction_count"] == 1
    assert receipt["arms"]["single_route"]["duplicate_candidates"]["count"] == 1
    assert receipt["arms"]["single_route"]["closure"]["truncated_decode_count"] == 1
    assert receipt["arms"]["single_route"]["duplicate_candidates"]["interpretation"].startswith(
        "geometry-derived"
    )

    single = receipt["source_to_treatment"]["single_route"]["by_scope"]["overall"]["owner_ledger"]
    assert single["gained_owner_ids"] == ["1:cat"]
    assert single["lost_owner_ids"] == ["1:person"]
    assert single["gained_selected_positive_path_owner_ids"] == ["1:cat"]
    assert single["gained_untargeted_owner_ids"] == []
    multi_geometry = receipt["source_to_treatment"]["multi_route"]["by_scope"]["overall"]["common_owner_geometry"]
    assert multi_geometry["common_owner_count"] == 256
    assert multi_geometry["x1_abs_error_px"]["min"] == -1.0
    assert multi_geometry["x2_abs_error_px"]["min"] == -1.0


def test_screen_fails_for_swapped_arms_wrong_cohort_and_dropped_evidence(tmp_path: Path) -> None:
    paths = _full_screen_inputs(tmp_path / "screen")
    runs = paths["runs"]
    with pytest.raises(ScreenAnalysisError, match="run name"):
        _analyze(paths, single_route_run_dir=runs["multi_route"], multi_route_run_dir=runs["single_route"])

    wrong = tmp_path / "wrong-cohort.jsonl"
    _write_jsonl(wrong, [{"image_id": image_id} for image_id in range(2, 258)])
    with pytest.raises(ScreenAnalysisError, match="frozen evaluation cohort"):
        _analyze(paths, frozen_eval_input_path=wrong)

    raw_path = runs["source"] / "gt_vs_pred.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["dropped_predictions"] = []
    _write_jsonl(raw_path, rows)
    _refresh_raw_provenance(runs["source"])
    with pytest.raises(ScreenAnalysisError, match="dropped prediction evidence"):
        _analyze(paths)


def test_screen_rejects_swapped_artifact_payload_and_non_integer_drop_count(tmp_path: Path) -> None:
    paths = _full_screen_inputs(tmp_path / "swapped-payload")
    runs = paths["runs"]
    source = runs["source"]
    treatment = runs["single_route"]
    for name in ("gt_vs_pred.jsonl", "summary.json", "parse_diagnostics.jsonl"):
        (source / name).write_bytes((treatment / name).read_bytes())
    with pytest.raises(ScreenAnalysisError, match="canonical provenance"):
        _analyze(paths)

    paths = _full_screen_inputs(tmp_path / "float-drop")
    raw_path = paths["runs"]["source"] / "gt_vs_pred.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["dropped_prediction_count"] = 1.9
    _write_jsonl(raw_path, rows)
    _refresh_raw_provenance(paths["runs"]["source"])
    with pytest.raises(ScreenAnalysisError, match="invalid dropped_prediction_count"):
        _analyze(paths)
