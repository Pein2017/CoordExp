from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.build_static_dynamic_owner_interface_h0_ledger import (
    H0LedgerContractError,
    build_h0_ledger,
    canonical_json_bytes,
    sha256_json,
)


def test_cli_help_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/build_static_dynamic_owner_interface_h0_ledger.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Build a strict, CPU-only owner ledger" in result.stdout


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _fixture(
    tmp_path: Path,
    *,
    checkpoint: str = "S",
    specs: list[dict[str, object]] | None = None,
) -> dict[str, Path | dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    panel_path = tmp_path / "source.jsonl"
    derived_path = tmp_path / "derived.jsonl"
    receipt_path = tmp_path / "derived.receipt.json"
    if specs is None:
        specs = [
            {
                "image_id": 2299,
                "owners": [{"category": "person", "ann": 1, "bbox_bins": [100, 100, 500, 500]}],
                "predictions": [{"category": "person", "bbox": [10, 10, 50, 50]}],
            },
            {
                "image_id": 4134,
                "owners": [{"category": "tie", "ann": 2, "bbox_bins": [200, 200, 600, 600]}],
                "predictions": [{"category": "tie", "bbox": [20, 20, 60, 60]}],
            },
        ]
    rows = []
    for spec in specs:
        image_id = int(spec["image_id"])
        objects = []
        for owner in spec["owners"]:  # type: ignore[union-attr]
            owner = dict(owner)
            bins = list(owner["bbox_bins"])
            objects.append({
                "bbox_2d": [f"<|coord_{value}|>" for value in bins],
                "category_name": owner["category"],
                "desc": owner["category"],
                "coco_ann_id": owner["ann"],
            })
        rows.append({
            "image_id": image_id,
            "width": 100,
            "height": 100,
            "images": [f"{image_id}.jpg"],
            "objects": objects,
        })
    # Keep source and derived bytes distinct while preserving owner identity.
    _write_jsonl(panel_path, rows)
    derived_rows = copy.deepcopy(rows)
    derived_path.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in derived_rows))
    source_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    derived_sha = hashlib.sha256(derived_path.read_bytes()).hexdigest()
    mappings = [
        {
            "image_id": row["image_id"],
            "row_index": index,
            "mapping": [
                {
                    "source_index": object_index,
                    "derived_index": object_index,
                    "object_sha256": sha256_json(object_value),
                }
                for object_index, object_value in enumerate(row["objects"])
            ],
        }
        for index, row in enumerate(rows)
    ]
    receipt = {
        "unit_id": "2026-08-05-static-dynamic-owner-interface-crossover",
        "schema_version": 1,
        "ordering": "geo_sorted_xy",
        "sort_key": ["decoded_x1", "decoded_y1", "source_index"],
        "source_sha256": source_sha,
        "derived_sha256": derived_sha,
        "row_count": 2,
        "owner_count": sum(len(row["objects"]) for row in rows),
        "mapping_count": sum(len(row["objects"]) for row in rows),
        "mapping_sha256": sha256_json(mappings),
        "coordinate_arity_verified": True,
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
        "source_to_derived": mappings,
    }
    _write_json(receipt_path, receipt)

    if checkpoint == "S":
        artifact_name = "qwen3-vl-2b-static-dynamic-owner-interface-s-step2444-h0"
        assistant_format = "object_box_closed"
        parser_policy = "compact_object_box_closed_only"
        config_fingerprint = "a" * 64
    else:
        artifact_name = "qwen3-vl-2b-static-dynamic-owner-interface-a3-step2445-h0"
        assistant_format = "object_box_commit"
        parser_policy = "compact_object_box_commit_only"
        config_fingerprint = "e" * 64
    artifact_dir = tmp_path / artifact_name
    artifact_dir.mkdir()
    config = {
        "schema_version": 1,
        "data": {"input_jsonl": str(derived_path)},
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 3084,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "template": {
            "id": "coordexp-swift-template-v1",
            "assistant_format": assistant_format,
            "object_ordering": "geo_sorted_xy",
            "object_field_order": "desc_first",
            "prompt": {"system": "system", "user": "user"},
        },
    }
    _write_json(artifact_dir / "configs.json", config)
    config_path = artifact_dir / "configs.json"
    resolved_path = artifact_dir / "configs" / "resolved.json"
    resolved_path.parent.mkdir()
    _write_json(resolved_path, {"config": config, "resolution": {"fingerprint": config_fingerprint}})

    generation_fingerprint = sha256_json(config["generation"])
    prompt_fingerprint = sha256_json({"template": config["template"], "template_id": "coordexp-swift-template-v1"})
    model_identity = "b" * 64
    manifest = {
        "artifact_schema_version": 1,
        "terminal_status": "completed",
        "failure_class": None,
        "checkpoint": checkpoint,
        "resolved_config_fingerprints": {"infer_config": config_fingerprint},
        "generation_config_fingerprint": generation_fingerprint,
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 3084,
        },
        "prompt_policy_fingerprint": prompt_fingerprint,
        "model_identity_fingerprint": model_identity,
        "processor_identity_fingerprint": "c" * 64,
        "template_identity": config["template"],
        "parser_policy": parser_policy,
        "response_family": "hf",
        "backend": "hf",
        "backend_mode": "generate",
        "scored_artifact_materialized": True,
        "trace_scoring_status": "scored",
        "dataset_identity": {"input_jsonl": str(derived_path)},
    }
    summary = {
        "terminal_status": "completed",
        "failure_class": None,
        "row_count": len(rows),
        "raw_row_count": len(rows),
        "scored_row_count": len(rows),
        "scored_artifact_materialized": True,
    }
    _write_json(artifact_dir / "run_manifest.json", manifest)
    _write_json(artifact_dir / "summary.json", summary)

    raw_rows: list[dict[str, object]] = []
    scored_rows: list[dict[str, object]] = []
    image_plan: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        image_id = int(row["image_id"])
        row_id = f"coco2017_val_{image_id:012d}"
        spec = next(item for item in specs if int(item["image_id"]) == image_id)
        gt_objects = []
        for owner in spec["owners"]:  # type: ignore[index]
            owner = dict(owner)
            gt_objects.append({
                "bbox": list(owner["bbox_bins"]),
                "description": owner["category"],
                "metadata": {"source": {"category_name": owner["category"], "coco_ann_id": owner["ann"]}},
            })
        predictions = []
        scored_predictions = []
        row_traces = []
        step = 0
        for generated_order, prediction_spec in enumerate(spec["predictions"]):  # type: ignore[index]
            prediction_spec = dict(prediction_spec)
            category = prediction_spec["category"]
            object_span_id = f"{row_id}:span-{generated_order}"
            row_start = step
            # The selected scoring steps may omit no tokens in this compact
            # fixture; physical closure is still the box_end token and A adds
            # a commit closure immediately after it.
            token_ids = [151646, 9000 + generated_order, 151647, 151648, 151700 + generated_order, 151649]
            token_text = ["<|object_ref_start|>", str(category), "<|object_ref_end|>", "<|box_start|>", "<|coord_100|>", "<|box_end|>"]
            selected_steps = list(range(row_start, row_start + len(token_ids)))
            for local_step, token_id in enumerate(token_ids):
                row_traces.append({
                    "trace_type": "generated_token",
                    "row_id": row_id,
                    "generated_step_index": row_start + local_step,
                    "token_id": token_id,
                    "token_text": token_text[local_step],
                })
            step += len(token_ids)
            if checkpoint == "A":
                row_traces.append({
                    "trace_type": "generated_token",
                    "row_id": row_id,
                    "generated_step_index": step,
                    "token_id": 151669,
                    "token_text": "<|commit|>",
                })
                step += 1
            pred = {
                "bbox": list(prediction_spec["bbox"]),
                "description": category,
                "generated_order": generated_order,
                "object_span_id": object_span_id,
            }
            scored_predictions.append({
                **pred,
                "score": float(prediction_spec.get("score", 0.9)),
                "pred_score_version": 1,
                "pred_score_source": {
                    "kind": "token_trace_selected_logprob_mean",
                    "row_id": row_id,
                    "object_span_id": object_span_id,
                    "score_policy_fingerprint": "d" * 64,
                    "selected_count": len(selected_steps),
                    "generated_step_indices": selected_steps,
                    "token_ids": token_ids,
                    "token_text": token_text,
                    "selected_logprobs": [-0.1] * len(selected_steps),
                },
            })
            predictions.append(pred)
        row_traces.append({
            "trace_type": "generated_token",
            "row_id": row_id,
            "generated_step_index": step,
            "token_id": 151645,
            "token_text": "<|im_end|>",
            "is_stop": True,
        })
        raw_rows.append({
            "row_id": row_id,
            "row_index": row_index,
            "example_id": row_id,
            "image_path": str(tmp_path / f"{image_id}.jpg"),
            "image_width": 100,
            "image_height": 100,
            "gt": gt_objects,
            "pred": predictions,
            "parse_status": "accepted",
            "metric_bearing": True,
            "valid_prediction_count": len(predictions),
            "dropped_prediction_count": 0,
        })
        scored_rows.append({
            "row_id": row_id,
            "row_index": row_index,
            "example_id": row_id,
            "image_path": str(tmp_path / f"{image_id}.jpg"),
            "image_width": 100,
            "image_height": 100,
            "gt": raw_rows[-1]["gt"],
            "pred": scored_predictions,
        })
        image_plan.append({"row_id": row_id, "row_index": row_index, "status": "ok", "error": None})
        traces.extend(row_traces)
    _write_jsonl(artifact_dir / "gt_vs_pred.jsonl", raw_rows)
    _write_jsonl(artifact_dir / "gt_vs_pred_scored.jsonl", scored_rows)
    _write_jsonl(artifact_dir / "image_plan.jsonl", image_plan)
    _write_jsonl(artifact_dir / "pred_token_trace.jsonl", traces)
    _write_jsonl(artifact_dir / "parse_diagnostics.jsonl", [])
    provenance = {
        "raw_artifact": {"path": "gt_vs_pred.jsonl", "sha256": hashlib.sha256((artifact_dir / "gt_vs_pred.jsonl").read_bytes()).hexdigest()},
        "scored_artifact": {"path": "gt_vs_pred_scored.jsonl", "sha256": hashlib.sha256((artifact_dir / "gt_vs_pred_scored.jsonl").read_bytes()).hexdigest()},
        "row_binding": {"row_count": len(rows), "row_ids_sha256": hashlib.sha256(json.dumps([row["row_id"] for row in raw_rows], separators=(",", ":")).encode()).hexdigest()},
        "generation_config_fingerprint": generation_fingerprint,
        "model_identity_fingerprint": model_identity,
        "processor_identity_fingerprint": "c" * 64,
        "prompt_policy_fingerprint": prompt_fingerprint,
        "template_identity": config["template"],
        "parser_policy": parser_policy,
    }
    _write_json(artifact_dir / "gt_vs_pred_scored.jsonl.provenance.json", provenance)
    return {"source": panel_path, "derived": derived_path, "receipt": receipt_path, "artifact": artifact_dir, "config": resolved_path, "raw": raw_rows, "scored": scored_rows, "config_path": config_path}


def _build(fixture: dict[str, Path | dict[str, object]]) -> dict[str, object]:
    return build_h0_ledger(
        fixture["artifact"], fixture["config"], fixture["source"], fixture["derived"], fixture["receipt"]
    )


def test_builds_native_ledger_and_keeps_legacy12_image2299_separate(tmp_path: Path) -> None:
    result = _build(_fixture(tmp_path))
    ledger = result["ledger"]
    assert ledger["schema_version"] == "static_dynamic_native_h0_owner_ledger.v1"
    assert ledger["history_complete"] is True
    assert ledger["verified_support_claim"] is False
    assert ledger["summary"]["legacy12"]["record_count"] == 1
    assert ledger["summary"]["image2299"]["record_count"] == 1
    assert all("verified_support" not in record for record in ledger["records"])
    assert all(record["verified_support_claim"] is False for record in ledger["records"])
    assert all(record["support_status"] == "not_measured" for record in ledger["records"])
    assert all(record["match_evidence"]["status"] == "tp" for record in ledger["records"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda fixture: fixture["receipt"].write_text(json.dumps({**json.loads(Path(fixture["receipt"]).read_text()), "source_sha256": "0" * 64})), "receipt source"),
        (lambda fixture: fixture["receipt"].write_text(json.dumps({**json.loads(Path(fixture["receipt"]).read_text()), "ordering": "y_then_x"})), "receipt ordering"),
        (lambda fixture: json.loads(Path(fixture["artifact"], "run_manifest.json").read_text()).update({"terminal_status": "failed"}), "run"),
    ],
)
def test_fails_closed_on_stale_receipt_or_incomplete_run(tmp_path: Path, mutation, message: str) -> None:
    fixture = _fixture(tmp_path)
    if message == "run":
        manifest_path = Path(fixture["artifact"], "run_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["terminal_status"] = "failed"
        _write_json(manifest_path, manifest)
    else:
        mutation(fixture)
    with pytest.raises(H0LedgerContractError):
        _build(fixture)


def test_fails_closed_on_wrong_wrapper_and_missing_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = Path(fixture["artifact"], "run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["template_identity"] = {**manifest["template_identity"], "assistant_format": "object_box_commit"}
    provenance_path = Path(fixture["artifact"], "gt_vs_pred_scored.jsonl.provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["template_identity"] = manifest["template_identity"]
    _write_json(manifest_path, manifest)
    _write_json(provenance_path, provenance)
    with pytest.raises(H0LedgerContractError):
        _build(fixture)

    fixture = _fixture(tmp_path / "wrong-checkpoint")
    with pytest.raises(H0LedgerContractError):
        build_h0_ledger(
            fixture["artifact"],
            fixture["config"],
            fixture["source"],
            fixture["derived"],
            fixture["receipt"],
            checkpoint="A",
        )

    fixture = _fixture(tmp_path / "missing")
    raw_path = Path(fixture["artifact"], "gt_vs_pred.jsonl")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    _write_jsonl(raw_path, rows[:1])
    with pytest.raises(H0LedgerContractError):
        _build(fixture)


def test_fails_closed_on_invalid_source_mapping(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt_path = Path(fixture["receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["source_to_derived"][1]["mapping"][0]["derived_index"] = 3
    receipt["mapping_sha256"] = sha256_json(receipt["source_to_derived"])
    _write_json(receipt_path, receipt)
    with pytest.raises(H0LedgerContractError):
        _build(fixture)


def test_fails_closed_on_source_image_identity_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_path = Path(fixture["artifact"], "gt_vs_pred.jsonl")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["image_path"] = str(tmp_path / "different.jpg")
    _write_jsonl(raw_path, rows)
    provenance_path = Path(fixture["artifact"], "gt_vs_pred_scored.jsonl.provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["raw_artifact"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    _write_json(provenance_path, provenance)
    with pytest.raises(H0LedgerContractError):
        _build(fixture)


def test_duplicate_prediction_is_one_to_one_and_recorded_as_unmatched(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    scored_path = Path(fixture["artifact"], "gt_vs_pred_scored.jsonl")
    scored = [json.loads(line) for line in scored_path.read_text().splitlines()]
    duplicate = copy.deepcopy(scored[0]["pred"][0])
    duplicate["generated_order"] = 1
    duplicate["object_span_id"] = "duplicate"
    duplicate["score"] = 0.1
    duplicate["pred_score_source"] = {
        **duplicate["pred_score_source"],
        "object_span_id": "duplicate",
        "generated_step_indices": [6, 7, 8, 9, 10, 11],
    }
    scored[0]["pred"].append(duplicate)
    _write_jsonl(scored_path, scored)
    trace_path = Path(fixture["artifact"], "pred_token_trace.jsonl")
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    target_row_id = scored[0]["row_id"]
    target = [item for item in traces if item["row_id"] == target_row_id]
    other = [item for item in traces if item["row_id"] != target_row_id]
    stop = next(item for item in target if item.get("is_stop") is True)
    target = [item for item in target if item is not stop]
    duplicate_trace = [
        {
            **item,
            "generated_step_index": index + 6,
        }
        for index, item in enumerate(target)
    ]
    stop["generated_step_index"] = 12
    _write_jsonl(trace_path, target + duplicate_trace + [stop] + other)
    provenance_path = Path(fixture["artifact"], "gt_vs_pred_scored.jsonl.provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["scored_artifact"]["sha256"] = hashlib.sha256(scored_path.read_bytes()).hexdigest()
    _write_json(provenance_path, provenance)
    # This duplicate is a legitimate COCO-style FP, but it must not become a
    # second TP for the same physical owner.
    result = _build(fixture)
    unmatched = result["ledger"]["unmatched_predictions"][0]["predictions"]
    assert any(item["status"] == "unmatched_duplicate_gt" for item in unmatched)
    assert result["ledger"]["summary"]["matched_tp_count"] == 2


def _boundary_specs(*, predictions: list[dict[str, object]], owners: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"image_id": 2299, "owners": owners, "predictions": predictions},
        {
            "image_id": 4134,
            "owners": [{"category": "tie", "ann": 100, "bbox_bins": [200, 200, 600, 600]}],
            "predictions": [{"category": "tie", "bbox": [20, 20, 60, 60]}],
        },
    ]


def test_boundary_prefixes_are_root_before_middle_and_pre_stop_fn(tmp_path: Path) -> None:
    owners = [
        {"category": "person", "ann": 1, "bbox_bins": [100, 100, 500, 500]},
        {"category": "tie", "ann": 2, "bbox_bins": [200, 200, 600, 600]},
    ]
    fixture = _fixture(
        tmp_path,
        specs=_boundary_specs(
            owners=owners,
            predictions=[
                {"category": "person", "bbox": [10, 10, 50, 50]},
                {"category": "tie", "bbox": [20, 20, 60, 60]},
            ],
        ),
    )
    result = _build(fixture)
    records = {
        int(item["source_panel_object_index"]): item
        for item in result["ledger"]["records"]
        if int(item["image_id"]) == 2299
    }
    first = records[0]
    middle = records[1]
    assert first["native_tp"] is True
    assert first["natural_boundary"] == first["due_boundary_index"] == 0
    assert first["prefix_semantics"] == "before_queried_owner_row"
    assert first["exact_prefix_token_ids"] == []
    assert first["generated_history_start_step"] is None
    assert first["generated_history_end_step"] is None
    assert first["latest_covered_owner_id"] is None
    assert first["covered_owner_ids"] == []
    assert middle["native_tp"] is True
    assert middle["natural_boundary"] == middle["due_boundary_index"] == 1
    assert middle["prefix_semantics"] == "before_queried_owner_row"
    assert middle["exact_prefix_token_ids"] == [151646, 9000, 151647, 151648, 151700, 151649]
    assert middle["generated_history_start_step"] == 0
    assert middle["generated_history_end_step"] == 5
    assert middle["latest_covered_owner_id"] == "gt:2299:0"
    assert middle["covered_owner_ids"] == ["gt:2299:0"]
    assert middle["queried_owner_not_covered"] is True
    assert 151645 not in middle["exact_prefix_token_ids"]


def test_fn_uses_first_strict_closure_and_excludes_terminal_im_end(tmp_path: Path) -> None:
    owners = [
        {"category": "person", "ann": 1, "bbox_bins": [100, 100, 500, 500]},
        {"category": "tie", "ann": 2, "bbox_bins": [200, 200, 600, 600]},
    ]
    fixture = _fixture(
        tmp_path,
        specs=_boundary_specs(
            owners=owners,
            predictions=[{"category": "person", "bbox": [10, 10, 50, 50]}],
        ),
    )
    result = _build(fixture)
    fn = next(
        item
        for item in result["ledger"]["records"]
        if int(item["image_id"]) == 2299 and int(item["source_panel_object_index"]) == 1
    )
    assert fn["native_fn"] is True
    assert fn["strict_complete_row"] is False
    assert fn["natural_boundary_valid"] is True
    assert fn["natural_boundary"] == fn["due_boundary_index"] == 1
    assert fn["prefix_semantics"] == "after_strict_covered_row_pre_stop"
    assert fn["exact_prefix_token_ids"] == [151646, 9000, 151647, 151648, 151700, 151649]
    assert fn["generated_history_end_step"] == 5
    assert fn["generated_history_stop_step"] == 6
    assert fn["latest_covered_owner_id"] == "gt:2299:0"
    assert fn["covered_owner_ids"] == ["gt:2299:0"]
    assert fn["is_earliest_eligible_boundary"] is True
    assert fn["queried_owner_not_covered"] is True
    assert 151645 not in fn["exact_prefix_token_ids"]


def test_a_commit_is_part_of_latest_covered_closure(tmp_path: Path) -> None:
    owners = [
        {"category": "person", "ann": 1, "bbox_bins": [100, 100, 500, 500]},
        {"category": "tie", "ann": 2, "bbox_bins": [200, 200, 600, 600]},
    ]
    fixture = _fixture(
        tmp_path,
        checkpoint="A",
        specs=_boundary_specs(
            owners=owners,
            predictions=[
                {"category": "person", "bbox": [10, 10, 50, 50]},
                {"category": "tie", "bbox": [20, 20, 60, 60]},
            ],
        ),
    )
    result = build_h0_ledger(
        fixture["artifact"], fixture["config"], fixture["source"], fixture["derived"], fixture["receipt"], checkpoint="A"
    )
    middle = next(
        item
        for item in result["ledger"]["records"]
        if int(item["image_id"]) == 2299 and int(item["source_panel_object_index"]) == 1
    )
    assert middle["native_tp"] is True
    assert middle["natural_boundary"] == 1
    assert middle["generated_history_end_step"] == 6
    assert middle["exact_prefix_token_ids"][-1] == 151669
    assert middle["generated_row_boundaries"][0]["closure_step"] == 6
    assert middle["generated_row_boundaries"][0]["closure_token_id"] == 151669
    assert 151645 not in middle["exact_prefix_token_ids"]


def test_no_strict_tp_is_explicitly_invalid_boundary(tmp_path: Path) -> None:
    owners = [{"category": "person", "ann": 1, "bbox_bins": [100, 100, 500, 500]}]
    fixture = _fixture(
        tmp_path,
        specs=_boundary_specs(
            owners=owners,
            predictions=[{"category": "dog", "bbox": [10, 10, 50, 50]}],
        ),
    )
    result = _build(fixture)
    fn = next(
        item
        for item in result["ledger"]["records"]
        if int(item["image_id"]) == 2299
    )
    assert fn["native_fn"] is True
    assert fn["natural_boundary_valid"] is False
    assert fn["natural_boundary"] is None
    assert fn["due_boundary_index"] is None
    assert fn["boundary_disposition"] == "no_valid_post_covered_boundary"
    assert fn["prefix_semantics"] == "no_valid_post_covered_boundary"
    assert fn["exact_prefix_token_ids"] is None
    assert fn["exact_prefix_sha256"] is None
    assert fn["latest_covered_owner_id"] is None
    assert fn["covered_owner_ids"] == []
    assert fn["is_earliest_eligible_boundary"] is False
