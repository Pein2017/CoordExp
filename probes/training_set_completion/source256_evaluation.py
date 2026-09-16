"""Freeze and reduce the Source256 main64 natural-readback endpoints.

This module is deliberately CPU-only.  ``prepare`` binds the final data bank,
the paired training manifests, and the already-qualified batch-4 decode path
into explicit worker commands.  ``reduce`` admits the saved shard bytes and
scores the fixed Source/16/64 endpoints without making model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from probes.training_set_completion import paired_evaluation
from probes.training_set_completion import readback_selectors
from probes.training_set_completion import source256_readback as readback
from probes.training_set_completion import source256_training as runtime
from probes.training_set_completion import training
from src.eval.assignment import global_matches


SCHEMA = "training_set_completion.source256_evaluation.v1"
WORKTREE = Path(__file__).resolve().parents[2]
ENDPOINTS = (
    ("Source0", "Source", 0),
    ("A16", "A", 16),
    ("A64", "A", 64),
    ("B16", "B", 16),
    ("B64", "B", 64),
)
THRESHOLDS = (0.5, 0.6, 0.8)
COORD = re.compile(r"^<\|coord_(\d{1,3})\|>$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verified_binding(value: Mapping[str, Any], name: str) -> Path:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    path = Path(str(value["path"])).resolve(strict=True)
    require(training.binding(path) == dict(value), f"{name} binding changed")
    return path


def _validate_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value.get("schema") == f"{SCHEMA}.packet", "Source256 evaluation packet schema")
    require(
        value.get("content_sha256")
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "Source256 evaluation packet digest",
    )
    preparation_path = _verified_binding(value["sources"]["preparation"], "preparation")
    trial_path = _verified_binding(value["sources"]["trial"], "trial")
    plan_path = _verified_binding(value["sources"]["readback_plan"], "readback plan")
    qualification_path = _verified_binding(
        value["sources"]["batch4_qualification"], "batch4 qualification"
    )
    _verified_binding(value["sources"]["producer"], "evaluation producer")
    prepared = runtime.validate_preparation(read(preparation_path))
    trial = read(trial_path)
    plan = readback.validate_plan(read(plan_path))
    require(trial.get("schema") == "training_set_completion.source256_trial.v1", "trial schema")
    require(trial.get("mode") == "main", "main64 trial required")
    require(trial.get("preparation") == training.binding(preparation_path), "trial preparation")
    require(plan["plan"]["preparation"] == training.binding(preparation_path), "plan preparation")
    require(readback._qualified_batch(qualification_path) == 4, "frozen batch4 qualification")
    require(
        [tuple((item["label"], item["arm"], item["step"])) for item in value["endpoints"]]
        == list(ENDPOINTS),
        "fixed Source/16/64 endpoints",
    )
    trial_manifests = {
        arm: _verified_binding(trial["arms"][arm], f"trial {arm} manifest")
        for arm in runtime.ARMS
    }
    for endpoint in value["endpoints"]:
        arm = "A" if endpoint["arm"] == "Source" else endpoint["arm"]
        require(
            endpoint["training_manifest"] == training.binding(trial_manifests[arm]),
            f"{endpoint['label']} manifest identity",
        )
    return {
        "packet": dict(value),
        "prepared": prepared,
        "trial": trial,
        "plan": plan,
        "paths": {
            "preparation": preparation_path,
            "trial": trial_path,
            "plan": plan_path,
            "qualification": qualification_path,
        },
    }


def _training_command(manifest: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--module",
        "probes.training_set_completion.source256_training",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]


def _readback_command(
    *,
    plan: Path,
    qualification: Path,
    manifest: Path,
    terminal: Path | None,
    step: int,
    split: str,
    shard: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "probes.training_set_completion.source256_readback",
        "endpoint-worker",
        "--plan",
        str(plan),
        "--qualification",
        str(qualification),
        "--training-manifest",
        str(manifest),
    ]
    if terminal is not None:
        command.extend(("--terminal", str(terminal)))
    command.extend(
        (
            "--step",
            str(step),
            "--split",
            split,
            "--shard",
            str(shard),
            "--output",
            str(output),
            "--device",
            "cuda:0",
        )
    )
    return command


def prepare_packet(
    *,
    preparation_path: Path,
    trial_path: Path,
    plan_path: Path,
    qualification_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Publish a held launch/readback/reducer packet; never launch a process."""

    require(not output.exists() and not output.is_symlink(), f"packet collision: {output}")
    preparation_path = preparation_path.resolve(strict=True)
    trial_path = trial_path.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    qualification_path = qualification_path.resolve(strict=True)
    prepared = runtime.validate_preparation(read(preparation_path))
    require(prepared["gate"]["status"] == "passed", "final scarcity gate")
    trial = read(trial_path)
    require(
        trial.get("schema") == "training_set_completion.source256_trial.v1"
        and trial.get("mode") == "main"
        and trial.get("preparation") == training.binding(preparation_path),
        "final main trial",
    )
    plan = readback.validate_plan(read(plan_path))
    require(plan["plan"]["preparation"] == training.binding(preparation_path), "final readback plan")
    require(readback._qualified_batch(qualification_path) == 4, "batch4 qualification")

    manifests: dict[str, Path] = {}
    for arm in runtime.ARMS:
        path = _verified_binding(trial["arms"][arm], f"{arm} training manifest")
        manifest = runtime.validate_training_manifest(read(path))
        require(
            manifest["arm"] == arm
            and manifest["mode"] == "main"
            and manifest["preparation"] == training.binding(preparation_path)
            and manifest["runtime"]["fresh_optimizer"] is True,
            f"{arm} fresh Source main manifest",
        )
        manifests[arm] = path

    train_launch = []
    for arm, devices in (("A", [0, 1, 2, 3]), ("B", [4, 5, 6, 7])):
        train_root = manifests[arm].parent / "training"
        train_launch.append(
            {
                "arm": arm,
                "visible_devices": devices,
                "command": _training_command(manifests[arm], train_root),
                "output": str(train_root),
                "terminal": str(train_root / "terminal.json"),
                "start_identity": "original Source adapter plus fresh AdamW and cosine64 scheduler",
            }
        )

    endpoints = []
    readback_root = output.parent / "readback"
    for label, arm, step in ENDPOINTS:
        manifest = manifests["A" if arm == "Source" else arm]
        terminal = None if step == 0 else manifests[arm].parent / "training" / "terminal.json"
        jobs = []
        for split in readback.SPLIT_COUNTS:
            for shard in range(readback.ENDPOINT_SHARDS):
                shard_output = readback_root / label / split / f"shard-{shard:02d}.json"
                jobs.append(
                    {
                        "split": split,
                        "shard": shard,
                        "visible_device": shard,
                        "output": str(shard_output),
                        "command": _readback_command(
                            plan=plan_path,
                            qualification=qualification_path,
                            manifest=manifest,
                            terminal=terminal,
                            step=step,
                            split=split,
                            shard=shard,
                            output=shard_output,
                        ),
                    }
                )
        endpoints.append(
            {
                "label": label,
                "arm": arm,
                "step": step,
                "training_manifest": training.binding(manifest),
                "training_terminal": None if terminal is None else str(terminal),
                "jobs": jobs,
            }
        )

    value: dict[str, Any] = {
        "schema": f"{SCHEMA}.packet",
        "status": "held_for_main64_lead_release",
        "scope": "final-bank paired64 training plus fixed batch4 Source/16/64 natural readback and CPU reduction",
        "sources": {
            "preparation": training.binding(preparation_path),
            "trial": training.binding(trial_path),
            "readback_plan": training.binding(plan_path),
            "batch4_qualification": training.binding(qualification_path),
            "producer": training.binding(Path(__file__)),
        },
        "scarcity_gate": prepared["gate"],
        "training_launch": train_launch,
        "endpoints": endpoints,
        "evaluation_contract": {
            "primary": "per-image global cardinality-first class-agnostic one-to-one IoU>=0.5 known-owner coverage",
            "diagnostic": "separate per-image class-consistent one-to-one IoU>=0.5/0.6/0.8 known-owner coverage",
            "comparisons": "owner-qualified retained/gained/lost identities per split for A-vs-B at16/64 and each endpoint vs Source0",
            "unmatched_prediction_semantics": "annotation-unmatched only; no physical false-positive claim",
            "burden": "separate malformed-row, invalid-geometry, cap/EOS-debt, and class-agnostic strict-IoU>0.95 later-repeat counters",
        },
        "bounds": {
            "training_updates_per_arm": 64,
            "training_image_presentations_per_update": 64,
            "training_model_calls_per_arm": 2_048,
            "endpoint_count": len(ENDPOINTS),
            "readback_worker_count": len(ENDPOINTS) * 2 * readback.ENDPOINT_SHARDS,
            "natural_image_requests": len(ENDPOINTS) * sum(readback.SPLIT_COUNTS.values()),
            "natural_batches_at4": len(ENDPOINTS) * sum(readback.SPLIT_COUNTS.values()) // 4,
        },
        "reducer": {
            "command": [
                sys.executable,
                "-m",
                "probes.training_set_completion.source256_evaluation",
                "reduce",
                "--packet",
                str(output),
                "--output",
                str(output.parent / "result.json"),
            ],
            "output": str(output.parent / "result.json"),
        },
        "launch": "held; packet creation makes no model calls",
        "content_sha256": None,
    }
    value["content_sha256"] = training.digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    training.publish(output, value)
    _validate_packet(value)
    return value


def _coord_bins(value: Sequence[Any]) -> list[int]:
    require(len(value) == 4, "four coordinate tokens")
    result = []
    for token in value:
        match = COORD.fullmatch(str(token))
        require(match is not None and 0 <= int(match.group(1)) <= 999, "coordinate token")
        result.append(int(match.group(1)))
    require(result[0] < result[2] and result[1] < result[3], "positive target box")
    return result


def _target(
    *, image_id: int, owner_id: Any, description: str, coord_bins: Sequence[Any]
) -> dict[str, Any]:
    normalized = " ".join(description.strip().casefold().split())
    checked = list(coord_bins)
    require(
        normalized != ""
        and len(checked) == 4
        and all(type(value) is int and 0 <= value <= 999 for value in checked)
        and checked[0] < checked[2]
        and checked[1] < checked[3],
        "valid known-owner target",
    )
    return {
        "image_id": image_id,
        "owner_id": str(owner_id),
        "reference_coord_bins_1000": checked,
        "description": description,
        "normalized_description": normalized,
    }


def _targets(prepared: Mapping[str, Any]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    raw = prepared["preparation"]
    train: dict[int, list[dict[str, Any]]] = {}
    for record in raw["bank"]["records"]:
        image_id = int(record["image_id"])
        train[image_id] = [
            _target(
                image_id=image_id,
                owner_id=owner["owner_id"],
                description=str(owner["description"]),
                coord_bins=owner["coord_bins"],
            )
            for owner in record["owners"]
        ]
    dev_rows = [
        json.loads(line)
        for line in Path(raw["sources"]["dev_jsonl"]["path"]).read_text().splitlines()
        if line
    ]
    dev: dict[int, list[dict[str, Any]]] = {}
    for row in dev_rows:
        image_id = int(row["image_id"])
        dev[image_id] = [
            _target(
                image_id=image_id,
                owner_id=item["coco_ann_id"],
                description=str(item.get("category_name", item["desc"])),
                coord_bins=_coord_bins(item["bbox_2d"]),
            )
            for item in row["objects"]
        ]
    require(len(train) == 256 and len(dev) == 128, "train256/dev128 owner ledgers")
    for split, rows in (("train", train), ("dev", dev)):
        keys = [(image_id, row["owner_id"]) for image_id, values in rows.items() for row in values]
        require(len(keys) == len(set(keys)), f"{split} unique owner identities")
    return {"train": train, "dev": dev}


def _contexts(prepared: Mapping[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    train = {}
    for record in runtime.hydrate_bound_cases(prepared):
        case = record["canonical_route"]["case"]
        train[int(record["image_id"])] = {
            "row_id": str(case["row_id"]),
            "row_index": int(case["row_index"]),
            "image_width": int(case["image_width"]),
            "image_height": int(case["image_height"]),
        }
    dev_rows = [
        json.loads(line)
        for line in Path(prepared["preparation"]["sources"]["dev_jsonl"]["path"]).read_text().splitlines()
        if line
    ]
    dev = {
        int(row["image_id"]): {
            "row_id": f"coco2017_train_{int(row['image_id']):012d}",
            "row_index": index,
            "image_width": int(row["width"]),
            "image_height": int(row["height"]),
        }
        for index, row in enumerate(dev_rows)
    }
    return {"train": train, "dev": dev}


def class_consistent_matches(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Return category-constrained cardinality-first one-to-one owner coverage."""

    require(threshold in THRESHOLDS, "registered class-consistent threshold")
    gt = [
        (str(item["normalized_description"]), tuple(item["reference_coord_bins_1000"]))
        for item in targets
    ]
    pred = [
        (
            " ".join(str(item.get("description", "")).strip().casefold().split()),
            tuple(item["coord_bins_1000"]),
        )
        for item in predictions
    ]
    matches = global_matches(gt, pred, threshold)
    rows = [
        {
            "reference_owner_id": str(targets[gi]["owner_id"]),
            "prediction_id": str(predictions[pi]["prediction_id"]),
            "prediction_order": int(predictions[pi]["generated_order"]),
            "iou": overlap,
        }
        for gi, pi, overlap in matches
    ]
    covered = {row["reference_owner_id"] for row in rows}
    matched_predictions = {row["prediction_id"] for row in rows}
    return {
        "target_count": len(targets),
        "matched_count": len(rows),
        "covered_owner_ids": [str(item["owner_id"]) for item in targets if str(item["owner_id"]) in covered],
        "missing_owner_ids": [str(item["owner_id"]) for item in targets if str(item["owner_id"]) not in covered],
        "annotation_unmatched_prediction_ids": [
            str(item["prediction_id"])
            for item in predictions
            if str(item["prediction_id"]) not in matched_predictions
        ],
        "matches": rows,
    }


def _strict_repeat_rows(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Count each later valid row once when any earlier row has IoU strictly >.95."""

    repeats = []
    for index, row in enumerate(predictions):
        overlaps = [
            readback_selectors.iou_xyxy(
                row["coord_bins_1000"], previous["coord_bins_1000"]
            )
            for previous in predictions[:index]
        ]
        best = max(overlaps, default=0.0)
        if best > 0.95:
            repeats.append(
                {
                    "prediction_id": str(row["prediction_id"]),
                    "generated_order": int(row["generated_order"]),
                    "best_earlier_iou": best,
                }
            )
    return repeats


def _admit_rows(
    *, checked: Mapping[str, Any], endpoint: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packet, plan = checked["packet"], checked["plan"]["plan"]
    expected_manifest = endpoint["training_manifest"]
    expected_terminal = endpoint["training_terminal"]
    admitted, bindings = [], []
    jobs = [job for job in endpoint["jobs"] if job["split"] == split]
    require(len(jobs) == readback.ENDPOINT_SHARDS, f"{endpoint['label']} {split} shard jobs")
    for job in jobs:
        path = Path(job["output"]).resolve(strict=True)
        value = read(path)
        shard = int(job["shard"])
        require(
            value.get("schema") == f"{readback.SCHEMA}.endpoint_shard"
            and value.get("status") == "completed_unscored"
            and value.get("endpoint") == {"arm": endpoint["arm"], "step": endpoint["step"]}
            and value.get("split") == split
            and value.get("shard") == shard
            and value.get("shard_count") == readback.ENDPOINT_SHARDS,
            f"{endpoint['label']} {split} shard identity",
        )
        require(
            value.get("plan") == packet["sources"]["readback_plan"]
            and value.get("qualification") == packet["sources"]["batch4_qualification"]
            and value.get("training_manifest") == expected_manifest,
            f"{endpoint['label']} shard sources",
        )
        expected_terminal_binding = (
            None if expected_terminal is None else training.binding(Path(expected_terminal))
        )
        require(value.get("training_terminal") == expected_terminal_binding, "endpoint terminal binding")
        generation = value.get("generation", {})
        rows = generation.get("rows")
        expected_ids = plan["endpoint_shards"][split][shard]
        require(
            value.get("batch_size") == 4
            and generation.get("configured_batch_size") == 4
            and generation.get("request_count") == len(expected_ids)
            and isinstance(rows, list)
            and [row.get("image_id") for row in rows] == expected_ids
            and all(batch.get("actual_batch_size") == 4 for batch in generation.get("batches", []))
            and all(row.get("actual_batch_size") == 4 for row in rows),
            f"{endpoint['label']} {split} full batch4 shard",
        )
        admitted.extend(dict(row) for row in rows)
        bindings.append(training.binding(path))
    order = [image_id for shard in plan["endpoint_shards"][split] for image_id in shard]
    require(len(admitted) == len(order) == readback.SPLIT_COUNTS[split], "endpoint split count")
    by_image = {int(row["image_id"]): row for row in admitted}
    require(len(by_image) == len(order) and set(by_image) == set(order), "endpoint split cohort")
    cohort_order = checked["plan"]["cohorts"][split]
    return [by_image[image_id] for image_id in cohort_order], bindings


def _score_split(
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[int, Sequence[Mapping[str, Any]]],
    contexts: Mapping[int, Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    from src.inference.parsing import parse_compact_object_box_closed

    per_image = []
    primary_total: Counter[str] = Counter()
    diagnostic_totals = {str(threshold): Counter() for threshold in THRESHOLDS}
    stop_counts: Counter[str] = Counter()
    burden_total: Counter[str] = Counter()
    for saved in rows:
        image_id = int(saved["image_id"])
        ids = saved.get("generated_token_ids")
        text = saved.get("raw_decode_text")
        require(
            isinstance(ids, list)
            and all(type(token) is int and token >= 0 for token in ids)
            and saved.get("generated_token_ids_sha256") == training.digest(ids),
            f"{split} saved token identity: {image_id}",
        )
        require(
            isinstance(text, str)
            and tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
            f"{split} saved decode identity: {image_id}",
        )
        readback._validate_stop(ids, str(saved.get("decode_stop_reason")))
        context = contexts[image_id]
        native_parse = parse_compact_object_box_closed(text, **context).to_artifact_dict()
        parsed = {
            **native_parse,
            "pred": native_parse["predictions"],
        }
        predictions, dropped = paired_evaluation._matchable_rows_with_geometry_debt(parsed)
        image_targets = list(targets[image_id])
        primary = paired_evaluation._ledger_image(image_targets, predictions, threshold=0.5)
        diagnostic = {
            str(threshold): class_consistent_matches(image_targets, predictions, threshold)
            for threshold in THRESHOLDS
        }
        termination = readback_selectors.termination_metrics(
            len(ids), ids, str(saved["decode_stop_reason"]), cap=readback.CAP
        )
        invalid_geometry = [
            row for row in dropped if row.get("drop_code") == "evaluation.geometry_invalid"
        ]
        malformed = [
            row for row in dropped if row.get("drop_code") != "evaluation.geometry_invalid"
        ]
        repeats = _strict_repeat_rows(predictions)
        burden_total.update(
            valid_prediction_count=len(predictions),
            malformed_row_count=len(malformed),
            invalid_geometry_count=len(invalid_geometry),
            annotation_unmatched_prediction_count=len(
                primary["annotation_unmatched_prediction_ids"]
            ),
            strict_repeat_row_count=len(repeats),
            cap_debt=int(termination["cap_debt"]),
            eos_debt=int(termination["eos_debt"]),
        )
        primary_total.update(
            target_count=primary["target_count"],
            matched_count=primary["matched_count"],
            missing_count=len(primary["missing_owner_ids"]),
        )
        for threshold, result in diagnostic.items():
            diagnostic_totals[threshold].update(
                target_count=result["target_count"],
                matched_count=result["matched_count"],
                missing_count=len(result["missing_owner_ids"]),
            )
        stop_counts[str(saved["decode_stop_reason"])] += 1
        per_image.append(
            {
                "image_id": image_id,
                "generated_token_ids_sha256": saved["generated_token_ids_sha256"],
                "decode_stop_reason": saved["decode_stop_reason"],
                "valid_prediction_count": len(predictions),
                "dropped_prediction_count": len(dropped),
                "burden": {
                    "malformed_row_count": len(malformed),
                    "invalid_geometry_count": len(invalid_geometry),
                    "annotation_unmatched_prediction_count": len(
                        primary["annotation_unmatched_prediction_ids"]
                    ),
                    "annotation_unmatched_semantics": "unknown relative to the fixed known-owner bank; not a confirmed false positive",
                    "strict_repeat_rows": repeats,
                    "repeat_semantics": "class-agnostic normalized-bin IoU strictly >0.95 against any earlier valid row; each later row counted once",
                    **termination,
                },
                "primary_class_agnostic_iou50": primary,
                "class_consistent": diagnostic,
            }
        )
    target_count = primary_total["target_count"]
    return {
        "split": split,
        "image_count": len(rows),
        "target_count": target_count,
        "primary_class_agnostic_iou50": {
            **dict(primary_total),
            "coverage": primary_total["matched_count"] / target_count if target_count else 0.0,
        },
        "class_consistent": {
            threshold: {
                **dict(values),
                "coverage": values["matched_count"] / values["target_count"]
                if values["target_count"]
                else 0.0,
            }
            for threshold, values in diagnostic_totals.items()
        },
        "stop_counts": dict(sorted(stop_counts.items())),
        "valid_prediction_count": sum(row["valid_prediction_count"] for row in per_image),
        "dropped_prediction_count": sum(row["dropped_prediction_count"] for row in per_image),
        "burden": dict(burden_total),
        "confirmed_false_instance_count": None,
        "physical_debt_status": "unresolved_no_endpoint_visual_review",
        "per_image": per_image,
    }


def _coverage(score: Mapping[str, Any], *, split: str, metric: str) -> set[tuple[int, str]]:
    split_score = score["splits"][split]
    rows = split_score["per_image"]
    return {
        (int(row["image_id"]), str(owner))
        for row in rows
        for owner in (
            row["primary_class_agnostic_iou50"]["covered_owner_ids"]
            if metric == "primary"
            else row["class_consistent"][metric]["covered_owner_ids"]
        )
    }


def _owner_changes(before: set[tuple[int, str]], after: set[tuple[int, str]]) -> dict[str, Any]:
    def rows(values: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
        return [
            {"image_id": image_id, "owner_id": owner_id}
            for image_id, owner_id in sorted(values, key=lambda item: (item[0], item[1]))
        ]

    per_image = []
    for image_id in sorted({image for image, _ in before | after}):
        old = {owner for image, owner in before if image == image_id}
        new = {owner for image, owner in after if image == image_id}
        per_image.append(
            {
                "image_id": image_id,
                "retained_owner_ids": sorted(old & new),
                "gained_owner_ids": sorted(new - old),
                "lost_owner_ids": sorted(old - new),
            }
        )
    return {
        "retained": rows(before & after),
        "gained": rows(after - before),
        "lost": rows(before - after),
        "retained_count": len(before & after),
        "gained_count": len(after - before),
        "lost_count": len(before - after),
        "per_image": per_image,
    }


def compare_scores(
    baseline: Mapping[str, Any], endpoint: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    require(
        baseline.get("schema") == endpoint.get("schema") == f"{SCHEMA}.endpoint_score",
        "Source256 comparison score schema",
    )
    require(baseline.get("preparation") == endpoint.get("preparation"), "comparison bank identity")
    splits = {}
    for split in readback.SPLIT_COUNTS:
        before = _coverage(baseline, split=split, metric="primary")
        after = _coverage(endpoint, split=split, metric="primary")
        splits[split] = {
            "primary_class_agnostic_iou50": _owner_changes(before, after),
            "class_consistent": {
                threshold: _owner_changes(
                    _coverage(baseline, split=split, metric=threshold),
                    _coverage(endpoint, split=split, metric=threshold),
                )
                for threshold in (str(value) for value in THRESHOLDS)
            },
        }
    return {
        "label": label,
        "baseline": baseline["endpoint"],
        "endpoint": endpoint["endpoint"],
        "splits": splits,
    }


def reduce(*, packet_path: Path, output: Path) -> dict[str, Any]:
    """Cold-admit all saved shards and publish the fixed paired result."""

    require(not output.exists() and not output.is_symlink(), f"result collision: {output}")
    checked = _validate_packet(read(packet_path))
    from transformers import AutoTokenizer

    raw = checked["prepared"]["preparation"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(raw["identity"]["runtime_contract"]["tokenizer_path"]).parent),
        local_files_only=True,
    )
    targets = _targets(checked["prepared"])
    contexts = _contexts(checked["prepared"])
    scores = {}
    for endpoint in checked["packet"]["endpoints"]:
        split_scores, shard_sources = {}, {}
        for split in readback.SPLIT_COUNTS:
            rows, bindings = _admit_rows(checked=checked, endpoint=endpoint, split=split)
            split_scores[split] = _score_split(
                split=split,
                rows=rows,
                targets=targets[split],
                contexts=contexts[split],
                tokenizer=tokenizer,
            )
            shard_sources[split] = bindings
        scores[endpoint["label"]] = {
            "schema": f"{SCHEMA}.endpoint_score",
            "status": "scored_saved_natural_readback",
            "endpoint": {key: endpoint[key] for key in ("label", "arm", "step")},
            "preparation": checked["packet"]["sources"]["preparation"],
            "readback_shards": shard_sources,
            "splits": split_scores,
        }
    source = scores["Source0"]
    comparisons = {
        "A16_vs_Source0": compare_scores(source, scores["A16"], label="A16_vs_Source0"),
        "B16_vs_Source0": compare_scores(source, scores["B16"], label="B16_vs_Source0"),
        "A64_vs_Source0": compare_scores(source, scores["A64"], label="A64_vs_Source0"),
        "B64_vs_Source0": compare_scores(source, scores["B64"], label="B64_vs_Source0"),
        "B16_vs_A16": compare_scores(scores["A16"], scores["B16"], label="B16_vs_A16"),
        "B64_vs_A64": compare_scores(scores["A64"], scores["B64"], label="B64_vs_A64"),
    }
    value = {
        "schema": f"{SCHEMA}.result",
        "status": "completed_saved_readback_evaluation",
        "packet": training.binding(packet_path),
        "preparation": checked["packet"]["sources"]["preparation"],
        "matching_contract": checked["packet"]["evaluation_contract"],
        "scores": scores,
        "comparisons": comparisons,
        "disposition": "No scalar composite or automatic winner; interpret the paired owner identities and fixed coverage endpoints.",
    }
    training.publish(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--preparation", type=Path, required=True)
    prepare.add_argument("--trial", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--qualification", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    reducer = sub.add_parser("reduce")
    reducer.add_argument("--packet", type=Path, required=True)
    reducer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_packet(
            preparation_path=args.preparation,
            trial_path=args.trial,
            plan_path=args.plan,
            qualification_path=args.qualification,
            output=args.output,
        )
    else:
        result = reduce(packet_path=args.packet, output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
