"""Prepare the Source256 fixed-prefix completion bank without model execution.

The producer has two phases.  ``prepare`` verifies and reduces the retained
Source greedy/K4 artifacts, builds the seed/admitted bank, and publishes an
all-256 runtime manifest.  Re-running with a versioned lead-admission ledger
rebuilds the same manifest against the updated bank.  It never calls a model.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy
from src.eval.assignment import global_matches
from src.eval.detection_categories import normalize_coco_category_name

from .training import validate_route


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
INPUT_ROOT = BASE / "2026-09-05-sft256-dev128-baseline/inputs-v3"
TRAIN_JSONL = INPUT_ROOT / "train.jsonl"
DEV_JSONL = INPUT_ROOT / "dev.jsonl"
INPUT_MANIFEST = INPUT_ROOT / "manifest.json"
CANONICAL_SOURCE = Path(
    "/data/CoordExp/public_data/coco/"
    "rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl"
)
GREEDY_ROOT = (
    BASE
    / "2026-09-05-sft256-dev128-baseline/natural-eval-v1"
    / "qwen3-vl-2b-sft256-source-train256-natural-v1"
)
K4_ROOT = BASE / "2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1"
K4_PLAN = K4_ROOT / "plan.json"
OPPORTUNITY_SUMMARY = BASE / "2026-09-09-natural-candidate-opportunity/full-v2/summary.json"
MODEL_ROOT = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
CHECKPOINT_ROOT = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
)
DEFAULT_OUTPUT = (
    BASE
    / "2026-09-16-source256-fixed-prefix-completion/preparation"
    / "source256-data-v1"
)

SCHEMA = "source256.fixed_prefix_preparation.v1"
ADMISSION_SCHEMA = "source256.review_admissions.v1"
REVIEW_SCHEMA = "source256.unmatched_owner_hypothesis.v1"
EOS = 151645
ROW_START = 151646
ROW_END = 151649
COORD_START = 151670
CAP = 3084
IGNORE_INDEX = -100
EXPECTED = {
    "train_sha256": "05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5",
    "dev_sha256": "b2ba42e6be18ca179cbdac387ed7af5b8400cf88a5a7d63f3b46643f2f61a46c",
    "adapter_sha256": "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da",
    "embedding_sha256": "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2",
    "tokenizer_sha256": "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8",
    "canonical_source_sha256": "ecf07a40856ee96a92c9093139abe24facfa600136e04f3c1bcbd02e039aaad1",
    "k4_plan_content_sha256": "de0c8ac01ed78f0f5fc31066257f7c089ae466a080818cce1e425d73c985b78e",
    "greedy_pad_tokens": 20580,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(value: Any, *, newline: bool = True) -> str:
    payload = canonical(value) if newline else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"bound source is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_hash(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{label} binding fields")
    require(binding(value["path"]) == dict(value), f"{label} source bytes changed")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def publish(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value)
    if output.exists():
        require(output.is_file() and output.read_bytes() == payload, f"collision: {output}")
        return
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    require(output.read_bytes() == payload, f"publication readback differs: {output}")


def publish_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical(dict(row)) for row in rows)
    if output.exists():
        require(output.is_file() and output.read_bytes() == payload, f"collision: {output}")
        return
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    require(output.read_bytes() == payload, f"publication readback differs: {output}")


_COORD = re.compile(r"^<\|coord_(\d+)\|>$")


def coord_bins(obj: Mapping[str, Any]) -> list[int]:
    raw = obj.get("bbox_2d")
    require(isinstance(raw, list) and len(raw) == 4, "bank bbox_2d must have four tokens")
    result: list[int] = []
    for token in raw:
        match = _COORD.fullmatch(str(token))
        require(match is not None, f"invalid coordinate token: {token!r}")
        value = int(match.group(1))
        require(0 <= value <= 999, "coordinate bin out of range")
        result.append(value)
    require(result[0] < result[2] and result[1] < result[3], "invalid bank bbox axes")
    return result


def _normal_description(value: Any) -> str:
    result = normalize_coco_category_name(str(value or ""))
    require(bool(result), "empty/unknown description")
    return result


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    require(tokenizer.eos_token_id == EOS, "tokenizer EOS differs")
    required = {
        "<|object_ref_start|>": ROW_START,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": ROW_END,
    }
    require(
        all(tokenizer.convert_tokens_to_ids(name) == token for name, token in required.items()),
        "tokenizer wrapper IDs differ",
    )
    require(
        [tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>") for index in range(1000)]
        == list(range(COORD_START, COORD_START + 1000)),
        "coordinate token IDs differ",
    )
    return tokenizer


def _token_hash(ids: Sequence[int]) -> str:
    """Runtime-facing token digest, identical to training.digest(list(ids))."""

    return digest([int(value) for value in ids])


def _source_token_hash(ids: Sequence[int]) -> str:
    """Historical Source acquisition token digest (canonical JSON, no newline)."""

    return digest([int(value) for value in ids], newline=False)


def _encode_owner(tokenizer: Any, owner: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    description = str(owner["description"])
    description_ids = tokenizer.encode(description, add_special_tokens=False)
    require(
        bool(description_ids)
        and tokenizer.decode(
            description_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == description,
        f"description tokenization is not literal: {description!r}",
    )
    bins = [int(value) for value in owner["coord_bins"]]
    ids = [ROW_START, *description_ids, 151647, 151648, *(COORD_START + value for value in bins), ROW_END]
    coordinate_offsets = list(range(3 + len(description_ids), 7 + len(description_ids)))
    require([ids[index] for index in coordinate_offsets] == [COORD_START + value for value in bins], "coordinate layout")
    return ids, coordinate_offsets


def _seed_bank(train_rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    seen_global: set[str] = set()
    for row in train_rows:
        image_id = int(row["image_id"])
        owners: list[dict[str, Any]] = []
        for obj in row["objects"]:
            owner_id = str(obj["coco_ann_id"])
            require(owner_id not in seen_global, f"duplicate seed owner ID: {owner_id}")
            seen_global.add(owner_id)
            bins = coord_bins(obj)
            owners.append(
                {
                    "owner_id": owner_id,
                    "description": str(obj["desc"]),
                    "normalized_description": _normal_description(obj["desc"]),
                    "coord_bins": bins,
                    "source": "original_objects",
                    "source_record": dict(obj),
                    "admission_id": None,
                }
            )
        owners.sort(key=lambda owner: (*owner["coord_bins"], owner["owner_id"]))
        require(len({owner["owner_id"] for owner in owners}) == len(owners), "image owner IDs")
        result[image_id] = owners
    require(len(result) == 256 and sum(map(len, result.values())) == 1955, "seed bank population")
    return result


def _load_admissions(path: str | Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return {
            "schema": ADMISSION_SCHEMA,
            "version": "seed-only-v1",
            "entries": [],
            "authority": "original objects retained; no new lead decisions supplied",
        }, []
    value = read_json(path)
    require(value.get("schema") == ADMISSION_SCHEMA, "admission schema")
    require(isinstance(value.get("version"), str) and value["version"], "admission version")
    entries = value.get("entries")
    require(isinstance(entries, list), "admission entries")
    ids = [str(entry.get("admission_id", "")) for entry in entries]
    require(all(ids) and len(ids) == len(set(ids)), "admission IDs")
    for entry in entries:
        require(entry.get("disposition") in {"lead-accepted", "rejected", "HOLD"}, "admission disposition")
        require(entry.get("action") in {"add_owner", "replace_owner", "exclude_owner", "none"}, "admission action")
        if entry["disposition"] != "lead-accepted":
            require(entry["action"] == "none", "non-accepted admission must not mutate bank")
    return value, [dict(entry) for entry in entries]


def admission_example() -> dict[str, Any]:
    """Return the exact versioned lead-decision interchange schema by example."""

    return {
        "schema": ADMISSION_SCHEMA,
        "version": "source256-visual-review-v1",
        "authority": "Only lead-accepted mutating entries change the trusted bank.",
        "entries": [
            {
                "admission_id": "example-add-owner",
                "image_id": 123,
                "candidate_id": "candidate-example",
                "disposition": "lead-accepted",
                "action": "add_owner",
                "owner_id": "source256-unlabeled-000001",
                "object": {
                    "desc": "person",
                    "bbox_2d": ["<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"],
                },
                "provenance": {"evidence": [], "reason": "verified distinct visible owner and accepted extent"},
            },
            {
                "admission_id": "example-repair-owner",
                "image_id": 123,
                "candidate_id": "candidate-repair-example",
                "disposition": "lead-accepted",
                "action": "replace_owner",
                "owner_id": "existing-owner-id",
                "object": {
                    "desc": "person",
                    "bbox_2d": ["<|coord_110|>", "<|coord_210|>", "<|coord_310|>", "<|coord_410|>"],
                },
                "provenance": {"evidence": [], "reason": "accepted category or geometry repair"},
            },
            {
                "admission_id": "example-exclude-seed-owner",
                "image_id": 123,
                "candidate_id": None,
                "disposition": "lead-accepted",
                "action": "exclude_owner",
                "owner_id": "existing-owner-id",
                "provenance": {"evidence": [], "reason": "targeted visibility or extent policy exclusion"},
            },
            {
                "admission_id": "example-hold",
                "image_id": 123,
                "candidate_id": "candidate-hold-example",
                "disposition": "HOLD",
                "action": "none",
                "owner_id": None,
                "provenance": {"evidence": [], "reason": "identity, category, or extent remains unresolved"},
            },
        ],
    }


def _apply_admissions(
    bank: dict[int, list[dict[str, Any]]], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    applied: list[str] = []
    for entry in entries:
        disposition = str(entry.get("disposition"))
        action = str(entry.get("action"))
        counts[f"{disposition}:{action}"] += 1
        if disposition != "lead-accepted" or action == "none":
            continue
        require(action in {"add_owner", "replace_owner", "exclude_owner"}, "unknown accepted admission action")
        image_id = int(entry["image_id"])
        require(image_id in bank, "admission image outside train256")
        owners = bank[image_id]
        owner_id = str(entry["owner_id"])
        current = next((owner for owner in owners if owner["owner_id"] == owner_id), None)
        if action == "exclude_owner":
            require(current is not None, "excluded owner absent")
            owners.remove(current)
        else:
            raw = entry.get("object")
            require(isinstance(raw, Mapping), "accepted add/replace requires object")
            candidate_id = entry.get("candidate_id")
            require(isinstance(candidate_id, str) and candidate_id, "accepted add/replace requires candidate ID")
            owner = {
                "owner_id": owner_id,
                "description": str(raw["desc"]),
                "normalized_description": _normal_description(raw["desc"]),
                "coord_bins": coord_bins(raw),
                "source": "lead_admission",
                "source_record": dict(raw),
                "candidate_id": candidate_id,
                "admission_id": str(entry["admission_id"]),
                "provenance": dict(entry.get("provenance", {})),
            }
            if action == "add_owner":
                require(current is None, "added owner already exists")
                owners.append(owner)
            else:
                require(current is not None, "replaced owner absent")
                owners[owners.index(current)] = owner
        owners.sort(key=lambda owner: (*owner["coord_bins"], owner["owner_id"]))
        require(bool(owners), "admission removed every owner from image")
        applied.append(str(entry["admission_id"]))
    all_ids = [owner["owner_id"] for owners in bank.values() for owner in owners]
    require(len(all_ids) == len(set(all_ids)), "post-admission owner IDs are not globally unique")
    return {"entry_counts": dict(sorted(counts.items())), "applied_admission_ids": applied}


def _owner_objects(
    owners: Sequence[Mapping[str, Any]], width: int, height: int
) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [
        (
            str(owner["normalized_description"]),
            tuple(
                float(value)
                for value in coord_bins_to_pixel_xyxy(
                    owner["coord_bins"],
                    image_width=width,
                    image_height=height,
                    field=f"owner[{owner['owner_id']}].coord_bins",
                )
            ),
        )
        for owner in owners
    ]


def _prediction_objects(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, tuple[float, float, float, float]]], list[int]]:
    valid: list[tuple[str, tuple[float, float, float, float]]] = []
    original_indexes: list[int] = []
    for index, prediction in enumerate(predictions):
        box = prediction.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            numeric = tuple(float(value) for value in box)
            description = _normal_description(prediction.get("description"))
        except (TypeError, ValueError):
            continue
        if numeric[0] >= numeric[2] or numeric[1] >= numeric[3]:
            continue
        valid.append((description, numeric))
        original_indexes.append(index)
    return valid, original_indexes


def _matches(
    predictions: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> dict[int, dict[str, Any]]:
    gt = _owner_objects(owners, width, height)
    pred, original_indexes = _prediction_objects(predictions)
    result: dict[int, dict[str, Any]] = {}
    for gt_index, pred_index, overlap in global_matches(gt, pred, 0.5):
        original_index = original_indexes[pred_index]
        result[original_index] = {
            "owner_id": str(owners[gt_index]["owner_id"]),
            "owner_index": gt_index,
            "iou": overlap,
        }
    return result


def _active_greedy_traces(path: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    by_row: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for trace in read_jsonl(path):
        if trace.get("trace_type") != "generated_token":
            continue
        by_row[str(trace["row_id"])].append(trace)
    pad_total = 0
    for row_id, traces in by_row.items():
        traces.sort(key=lambda item: int(item["generated_step_index"]))
        require(
            [int(item["generated_step_index"]) for item in traces] == list(range(len(traces))),
            f"greedy trace indexes differ: {row_id}",
        )
        active = [item for item in traces if not item["is_pad"]]
        require(traces[: len(active)] == active, f"non-pad after pad: {row_id}")
        pad_total += len(traces) - len(active)
        by_row[row_id] = active
    return dict(by_row), pad_total


def _verify_greedy(
    tokenizer: Any,
    greedy_rows: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    require(len(greedy_rows) == len(traces) == 256, "greedy population")
    token_total = 0
    row_total = 0
    for row in greedy_rows:
        row_id = str(row["row_id"])
        active = list(traces[row_id])
        ids = [int(item["token_id"]) for item in active]
        require(bool(ids) and len(ids) <= CAP, f"greedy token budget: {row_id}")
        require(
            tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            == row["raw_decode_text"],
            f"greedy token/text identity: {row_id}",
        )
        require(
            all(
                tokenizer.decode(
                    [int(item["token_id"])],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                == item["token_text"]
                for item in active
            ),
            f"greedy individual token identity: {row_id}",
        )
        stop_positions = [index for index, item in enumerate(active) if item["is_stop"]]
        if row["decode_stop_reason"] == "im_end":
            require(stop_positions == [len(ids) - 1] and ids[-1] == EOS, f"greedy EOS: {row_id}")
        else:
            require(
                row["decode_stop_reason"] == "length"
                and len(ids) == CAP
                and not stop_positions
                and EOS not in ids,
                f"greedy cap: {row_id}",
            )
        parsed_and_dropped = len(row["pred"]) + int(row.get("dropped_prediction_count", 0))
        require(ids.count(ROW_START) == parsed_and_dropped, f"greedy row-start count: {row_id}")
        require(
            ids.count(ROW_END)
            in ({parsed_and_dropped} if row["decode_stop_reason"] == "im_end" else {parsed_and_dropped, parsed_and_dropped - 1}),
            f"greedy row-end count: {row_id}",
        )
        token_total += len(ids)
        row_total += len(row["pred"])
    return {"images": 256, "active_tokens": token_total, "parsed_rows": row_total}


def _verify_k4(tokenizer: Any, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(plan.get("content_sha256") == EXPECTED["k4_plan_content_sha256"], "K4 plan content identity")
    require(plan["population"]["image_count"] == 256 and plan["population"]["k"] == 4, "K4 plan shape")
    acquisition_paths = [Path(item["path"]) for item in plan["sources"]["rollout_artifacts"]]
    rollouts: list[dict[str, Any]] = []
    for item, path in zip(plan["sources"]["rollout_artifacts"], acquisition_paths):
        require(file_hash(path) == item["sha256"], f"K4 shard changed: {path}")
        shard = read_json(path)
        require(shard["rollout_count"] == len(shard["rollouts"]), "K4 shard count")
        rollouts.extend(shard["rollouts"])
    require(len(rollouts) == 1024, "K4 rollout count")
    indexed = {(str(row["example_id"]), int(row["seed"])): row for row in rollouts}
    require(len(indexed) == 1024, "K4 rollout identity")
    row_total = 0
    for group in plan["population"]["groups"]:
        for action in group["actions"]:
            key = (str(group["example_id"]), int(action["seed"]))
            rollout = indexed[key]
            body = [int(value) for value in rollout["generated_token_ids"]]
            require(body == action["generated_token_ids"], f"K4 token identity: {key}")
            require(_source_token_hash(body) == action["generated_token_ids_sha256"], f"K4 token digest: {key}")
            require(
                tokenizer.decode(
                    body,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                == rollout["generated_text"],
                f"K4 token/text identity: {key}",
            )
            complete = body + ([EOS] if rollout["stop_reason"] == "im_end" else [])
            require(complete == action["action_token_ids"], f"K4 terminal identity: {key}")
            require(rollout["prompt_token_ids"] == group["prompt_token_ids"], f"K4 prompt identity: {key}")
            parsed = rollout["predictions"]
            require(parsed["valid_prediction_count"] == len(parsed["predictions"]), f"K4 parser count: {key}")
            row_total += len(parsed["predictions"])
    return rollouts, {"images": 256, "samples": 1024, "parsed_rows": row_total}


def _identity(
    plan: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    greedy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(file_hash(TRAIN_JSONL) == EXPECTED["train_sha256"], "train JSONL digest")
    require(file_hash(DEV_JSONL) == EXPECTED["dev_sha256"], "dev JSONL digest")
    input_manifest = read_json(INPUT_MANIFEST)
    require(input_manifest.get("canonical_source", {}).get("path") == str(CANONICAL_SOURCE), "canonical source path")
    require(
        input_manifest.get("canonical_source", {}).get("sha256")
        == EXPECTED["canonical_source_sha256"]
        == file_hash(CANONICAL_SOURCE),
        "canonical source digest",
    )
    require(
        input_manifest.get("outputs", {}).get("train", {}).get("sha256") == EXPECTED["train_sha256"]
        and input_manifest.get("outputs", {}).get("dev", {}).get("sha256") == EXPECTED["dev_sha256"],
        "prepared input manifest output identity",
    )
    require(
        file_hash(CHECKPOINT_ROOT / "adapter/adapter_model.safetensors") == EXPECTED["adapter_sha256"],
        "adapter weights digest",
    )
    require(
        file_hash(CHECKPOINT_ROOT / "special_token_embeddings/special_token_embeddings.safetensors")
        == EXPECTED["embedding_sha256"],
        "embedding weights digest",
    )
    require(file_hash(MODEL_ROOT / "tokenizer.json") == EXPECTED["tokenizer_sha256"], "tokenizer digest")
    opportunity = read_json(OPPORTUNITY_SUMMARY)
    require(
        opportunity["source_plan_content_sha256"] == EXPECTED["k4_plan_content_sha256"],
        "retained opportunity plan identity",
    )
    for path, expected in opportunity["source_files"].items():
        require(file_hash(path) == expected, f"retained opportunity source changed: {path}")
    require(len(train_rows) == len(greedy_rows) == 256, "train/greedy count")
    train_ids = [int(row["image_id"]) for row in train_rows]
    greedy_ids = [int(str(row["row_id"]).rsplit("_", 1)[-1]) for row in greedy_rows]
    groups = plan["population"]["groups"]
    group_ids = [int(group["image_id"]) for group in groups]
    require(train_ids == group_ids == greedy_ids, "train/K4/greedy image order")
    require(len(set(train_ids)) == 256, "duplicate train image")
    for source, group, greedy in zip(train_rows, groups, greedy_rows):
        image_path = Path(group["image_path"])
        require(image_path.is_file(), f"image missing: {image_path}")
        require(file_hash(image_path) == group["image_content_sha256"], f"image digest: {image_path}")
        require(int(source["width"]) == int(greedy["image_width"]), "image width differs")
        require(int(source["height"]) == int(greedy["image_height"]), "image height differs")
        require(group["annotated_owner_count"] == len(source["objects"]) == len(greedy["gt"]), "GT count differs")
        require(group["example_id"] == greedy["row_id"] == greedy["example_id"], "example identity differs")
        require(_source_token_hash(group["prompt_token_ids"]) == group["prompt_token_ids_sha256"], "prompt digest")
    return {
        "status": "verified",
        "image_count": 256,
        "original_owner_count": 1955,
        "train_dev_overlap": len(set(train_ids) & {int(row["image_id"]) for row in read_jsonl(DEV_JSONL)}),
        "checkpoint": {
            "root": str(CHECKPOINT_ROOT),
            "adapter_weights_sha256": EXPECTED["adapter_sha256"],
            "embedding_weights_sha256": EXPECTED["embedding_sha256"],
        },
        "base_model": str(MODEL_ROOT),
        "tokenizer_sha256": EXPECTED["tokenizer_sha256"],
        "k4_plan_content_sha256": EXPECTED["k4_plan_content_sha256"],
        "input_sha256": {"train": EXPECTED["train_sha256"], "dev": EXPECTED["dev_sha256"]},
        "processed_input_provenance": {
            "view": "rescale_32_1024_bbox_len12000_xy_sorted",
            "canonical_source_path": str(CANONICAL_SOURCE),
            "canonical_source_sha256": EXPECTED["canonical_source_sha256"],
            "canonical_source_rows": int(input_manifest["canonical_source"]["rows"]),
            "selected_train_rows": 256,
            "selected_dev_rows": 128,
            "semantics": "processed rescaled/sorted COCO view; not raw COCO JSON/images",
        },
        "runtime_contract": {
            "base_model_root": str(MODEL_ROOT),
            "adapter_root": str(CHECKPOINT_ROOT / "adapter"),
            "embedding_root": str(CHECKPOINT_ROOT / "special_token_embeddings"),
            "tokenizer_path": str(MODEL_ROOT / "tokenizer.json"),
            "eos_token_id": EOS,
            "row_start_token_id": ROW_START,
            "row_end_token_id": ROW_END,
            "coordinate_token_ids": list(range(COORD_START, COORD_START + 1000)),
            "coordinate_bin_count": 1000,
            "continuation_token_cap": CAP,
            "object_ordering": "geo_sorted_xy",
            "assistant_format": "object_box_closed",
            "model_dtype": "fp32",
            "attention_implementation": "sdpa",
        },
    }


def _prefix_evidence(
    row: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = list(row["pred"])
    matched = _matches(
        predictions,
        owners,
        width=int(row["image_width"]),
        height=int(row["image_height"]),
    )
    drops = sorted(row.get("dropped_predictions", []), key=lambda value: int(value.get("char_start", 1 << 60)))
    first_drop = int(drops[0].get("char_start", 1 << 60)) if drops else 1 << 60
    prefix: list[dict[str, Any]] = []
    seen: set[str] = set()
    boundary: dict[str, Any] | None = None
    for pred_index, prediction in enumerate(predictions):
        if first_drop < int(prediction["char_start"]):
            boundary = {"kind": "malformed_or_dropped_row", "char_start": first_drop}
            break
        match = matched.get(pred_index)
        if match is None:
            boundary = {
                "kind": "unmatched_unknown_duplicate_or_inadmissible",
                "pred_index": pred_index,
                "object_span_id": prediction.get("object_span_id"),
            }
            break
        if match["owner_id"] in seen:
            boundary = {"kind": "duplicate_owner", "pred_index": pred_index, "owner_id": match["owner_id"]}
            break
        seen.add(match["owner_id"])
        prefix.append(
            {
                "pred_index": pred_index,
                "owner_id": match["owner_id"],
                "iou": match["iou"],
                "object_span_id": prediction.get("object_span_id"),
                "raw_span_sha256": prediction.get("raw_span_sha256"),
            }
        )
    if boundary is None and first_drop < (1 << 60):
        boundary = {"kind": "malformed_or_dropped_row", "char_start": first_drop}
    if boundary is None:
        boundary = {"kind": "observed_termination", "stop_reason": row["decode_stop_reason"]}
    ids = [int(item["token_id"]) for item in traces]
    prefix_ids: list[int] = []
    if prefix:
        char_end = int(predictions[len(prefix) - 1]["char_end"])
        pieces: list[str] = []
        token_end = None
        for index, trace in enumerate(traces):
            pieces.append(str(trace["token_text"]))
            joined = "".join(pieces)
            if len(joined) >= char_end:
                require(
                    len(joined) == char_end
                    and joined == str(row["raw_decode_text"])[:char_end],
                    f"prefix trace/text boundary: {row['row_id']}",
                )
                token_end = index + 1
                break
        require(token_end is not None, f"missing prefix token boundary: {row['row_id']}")
        prefix_ids = ids[:token_end]
        require(prefix_ids[-1] == ROW_END and EOS not in prefix_ids, "prefix token boundary")
    owner_ids = [item["owner_id"] for item in prefix]
    remaining = [str(owner["owner_id"]) for owner in owners if str(owner["owner_id"]) not in seen]
    structural = bool(prefix) and bool(remaining)
    return {
        "structural_candidate": structural,
        "prefix_length": len(prefix),
        "prefix_owner_ids": owner_ids,
        "remaining_owner_ids": remaining,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": _token_hash(prefix_ids),
        "boundary": boundary,
        "matched_rows": prefix,
    }


def _trusted_boxes(
    encoded: Sequence[tuple[Mapping[str, Any], Sequence[int], Sequence[int]]],
    *,
    offset: int,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    total = offset + sum(len(ids) for _, ids, _ in encoded) + 1
    weights = [0] * total
    targets = [IGNORE_INDEX] * total
    boxes: list[dict[str, Any]] = []
    cursor = offset
    for owner, ids, coordinates in encoded:
        positions = [cursor + int(index) for index in coordinates]
        bins = [int(value) for value in owner["coord_bins"]]
        for position, value in zip(positions, bins):
            weights[position] = 1
            targets[position] = value
        boxes.append(
            {
                "x1_position": positions[0],
                "y1_position": positions[1],
                "x2_position": positions[2],
                "y2_position": positions[3],
                "expected_bins": bins,
            }
        )
        cursor += len(ids)
    return boxes, weights, targets


def _route(
    *,
    tokenizer: Any,
    route_kind: str,
    source: Mapping[str, Any],
    group: Mapping[str, Any],
    greedy: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    prefix: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prefix_ids = list(prefix["prefix_token_ids"]) if prefix is not None else []
    prefix_owner_ids = list(prefix["prefix_owner_ids"]) if prefix is not None else []
    suffix_owners = [owner for owner in owners if str(owner["owner_id"]) not in set(prefix_owner_ids)]
    encoded = [(owner, *_encode_owner(tokenizer, owner)) for owner in suffix_owners]
    suffix_ids = [token for _, ids, _ in encoded for token in ids] + [EOS]
    continuation = prefix_ids + suffix_ids
    ce_weights = [0] * len(prefix_ids) + [1] * len(suffix_ids)
    labels = [token if weight else IGNORE_INDEX for token, weight in zip(continuation, ce_weights)]
    boxes, geometry_weights, geometry_targets = _trusted_boxes(encoded, offset=len(prefix_ids))
    require(len(continuation) <= CAP, "route exceeds frozen continuation cap")
    owner_ids = [str(owner["owner_id"]) for owner in owners]
    require(prefix_owner_ids + [str(owner["owner_id"]) for owner in suffix_owners] != [], "empty bank")
    route = {
        "route_id": f"{group['example_id']}:{route_kind}",
        "image_id": int(source["image_id"]),
        "example_id": str(group["example_id"]),
        "case": {
            "image_path": str(group["image_path"]),
            "image_plan": {
                "image_content_sha256": str(group["image_content_sha256"]),
                "executed_media_sha256": str(group["executed_media_sha256"]),
                "observed_image_grid_thw": list(group["observed_image_grid_thw"]),
            },
        },
        "image_identity": {
            "image_path": str(group["image_path"]),
            "image_content_sha256": str(group["image_content_sha256"]),
            "executed_media_sha256": str(group["executed_media_sha256"]),
            "observed_image_grid_thw": list(group["observed_image_grid_thw"]),
        },
        "prompt_token_ids": list(group["prompt_token_ids"]),
        "continuation_token_ids": continuation,
        "ce_weights": ce_weights,
        "labels": labels,
        "trusted_boxes": boxes,
        "geometry_weights": geometry_weights,
        "geometry_target_bins": geometry_targets,
        "trusted_complete_support_endpoint": True,
        "provenance": {
            "route_kind": route_kind,
            "bank_owner_ids": owner_ids,
            "prefix_owner_ids": prefix_owner_ids,
            "suffix_owner_ids": [str(owner["owner_id"]) for owner in suffix_owners],
            "prefix_token_ids": prefix_ids,
            "suffix_token_ids": suffix_ids,
            "prefix_token_ids_sha256": _token_hash(prefix_ids),
            "suffix_token_ids_sha256": _token_hash(suffix_ids),
            "source_greedy_generated_token_ids_sha256": _token_hash(
                [int(item["token_id"]) for item in greedy["active_traces"]]
            ),
            "mask_semantics": "prefix_and_prompt_context_masked; every suffix token including EOS supervised",
            "geometry_semantics": "only four coordinate tokens of supervised bank-owner suffix rows",
        },
    }
    validate_route(route, eos_token_id=EOS, coordinate_token_ids=list(range(COORD_START, COORD_START + 1000)))
    return route


def _validate_route_extras(route: Mapping[str, Any]) -> None:
    tokens = route["continuation_token_ids"]
    weights = route["ce_weights"]
    labels = route["labels"]
    geometry = route["geometry_weights"]
    targets = route["geometry_target_bins"]
    require(len(tokens) == len(weights) == len(labels) == len(geometry) == len(targets), "aligned route vectors")
    require(
        labels == [token if weight else IGNORE_INDEX for token, weight in zip(tokens, weights)],
        "route labels differ from CE weights",
    )
    positions: dict[int, int] = {}
    for box in route["trusted_boxes"]:
        for key, value in zip(
            ("x1_position", "y1_position", "x2_position", "y2_position"),
            box["expected_bins"],
        ):
            position = int(box[key])
            require(position not in positions, "overlapping trusted coordinate positions")
            positions[position] = int(value)
    require(geometry == [int(index in positions) for index in range(len(tokens))], "geometry weights")
    require(targets == [positions.get(index, IGNORE_INDEX) for index in range(len(tokens))], "geometry targets")
    provenance = route["provenance"]
    required = {
        "route_kind",
        "bank_owner_ids",
        "prefix_owner_ids",
        "suffix_owner_ids",
        "prefix_token_ids",
        "suffix_token_ids",
        "prefix_token_ids_sha256",
        "suffix_token_ids_sha256",
        "source_greedy_generated_token_ids_sha256",
        "mask_semantics",
        "geometry_semantics",
    }
    require(set(provenance) == required, "route provenance fields")
    prefix = provenance["prefix_token_ids"]
    suffix = provenance["suffix_token_ids"]
    require(tokens == prefix + suffix, "prefix/suffix token partition")
    require(provenance["prefix_token_ids_sha256"] == _token_hash(prefix), "prefix token digest")
    require(provenance["suffix_token_ids_sha256"] == _token_hash(suffix), "suffix token digest")
    require(all(weight == 0 for weight in weights[: len(prefix)]), "prefix CE mask")
    require(all(weight == 0 for weight in geometry[: len(prefix)]), "prefix geometry mask")
    partition = provenance["prefix_owner_ids"] + provenance["suffix_owner_ids"]
    require(
        len(partition) == len(set(partition))
        and set(partition) == set(provenance["bank_owner_ids"]),
        "complete owner coverage",
    )


def _schedule(image_ids: Sequence[int], *, seed: int = 19) -> dict[str, Any]:
    require(len(image_ids) == 256 and len(set(image_ids)) == 256, "schedule population")
    common_batches: list[list[int]] = []
    variable_batches: list[list[int]] = []
    for epoch in range(8):
        common = list(image_ids)
        variable = list(image_ids)
        random.Random(seed + 2 * epoch).shuffle(common)
        random.Random(seed + 2 * epoch + 1).shuffle(variable)
        common_batches.extend(common[index : index + 32] for index in range(0, 256, 32))
        variable_batches.extend(variable[index : index + 32] for index in range(0, 256, 32))
    updates = [
        {
            "step": index + 1,
            "common_image_ids": common_batches[index],
            "variable_image_ids": variable_batches[index],
        }
        for index in range(64)
    ]
    common_counts = collections.Counter(value for batch in common_batches for value in batch)
    variable_counts = collections.Counter(value for batch in variable_batches for value in batch)
    require(set(common_counts.values()) == {8} and set(variable_counts.values()) == {8}, "schedule balance")
    return {
        "seed": seed,
        "update_count": 64,
        "effective_image_batch": 64,
        "branch_weights": {"common": 0.5, "variable": 0.5},
        "updates": updates,
        "common_exposures_per_image": 8,
        "variable_exposures_per_image": 8,
        "total_presentations": 4096,
    }


def _review_proposals(
    greedy_rows: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    bank: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    for row in greedy_rows:
        sources.append(
            {
                "source_kind": "source_greedy",
                "seed": None,
                "example_id": row["example_id"],
                "image_id": int(str(row["row_id"]).rsplit("_", 1)[-1]),
                "image_path": row["image_path"],
                "width": int(row["image_width"]),
                "height": int(row["image_height"]),
                "predictions": row["pred"],
                "drops": row["dropped_predictions"],
                "stop_reason": row["decode_stop_reason"],
                "raw_text_sha256": hashlib.sha256(row["raw_decode_text"].encode()).hexdigest(),
            }
        )
    for rollout in rollouts:
        parsed = rollout["predictions"]
        sources.append(
            {
                "source_kind": "source_k4_sample",
                "seed": int(rollout["seed"]),
                "example_id": rollout["example_id"],
                "image_id": int(rollout["image_id"]),
                "image_path": next(
                    row["image_path"]
                    for row in greedy_rows
                    if row["example_id"] == rollout["example_id"]
                ),
                "width": int(parsed.get("image_width", 0) or next(row["image_width"] for row in greedy_rows if row["example_id"] == rollout["example_id"])),
                "height": int(parsed.get("image_height", 0) or next(row["image_height"] for row in greedy_rows if row["example_id"] == rollout["example_id"])),
                "predictions": parsed["predictions"],
                "drops": parsed["dropped_predictions"],
                "stop_reason": rollout["stop_reason"],
                "raw_text_sha256": hashlib.sha256(rollout["generated_text"].encode()).hexdigest(),
            }
        )
    proposals: list[dict[str, Any]] = []
    parse_debt: list[dict[str, Any]] = []
    for source in sources:
        image_id = int(source["image_id"])
        predictions = source["predictions"]
        matches = _matches(
            predictions,
            bank[image_id],
            width=int(source["width"]),
            height=int(source["height"]),
        )
        first_boundary_pred_index = None
        if source["source_kind"] == "source_greedy":
            first_drop_char = min(
                (int(item.get("char_start", 1 << 60)) for item in source["drops"]),
                default=1 << 60,
            )
            for candidate_index, candidate_prediction in enumerate(predictions):
                if first_drop_char < int(candidate_prediction.get("char_start", 1 << 60)):
                    break
                if candidate_index not in matches:
                    first_boundary_pred_index = candidate_index
                    break
        matched_owner_ids = {value["owner_id"] for value in matches.values()}
        for index, prediction in enumerate(predictions):
            if index in matches:
                continue
            box = prediction.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                parse_debt.append({**source, "predictions": None, "drops": None, "debt": prediction})
                continue
            top = []
            for owner, (_, owner_box) in zip(
                bank[image_id],
                _owner_objects(bank[image_id], int(source["width"]), int(source["height"])),
            ):
                top.append(
                    {
                        "owner_id": owner["owner_id"],
                        "same_category": owner["normalized_description"] == _normal_description(prediction["description"]),
                        "iou": iou_xyxy(tuple(map(float, box)), owner_box),
                    }
                )
            top.sort(key=lambda item: (-item["iou"], item["owner_id"]))
            proposal_core = {
                "image_id": image_id,
                "source_kind": source["source_kind"],
                "seed": source["seed"],
                "object_span_id": prediction.get("object_span_id"),
                "raw_span_sha256": prediction.get("raw_span_sha256"),
            }
            proposal_id = "proposal-" + digest(proposal_core, newline=False)[:20]
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "image_id": image_id,
                    "example_id": source["example_id"],
                    "image_path": source["image_path"],
                    "image_width": source["width"],
                    "image_height": source["height"],
                    "source_kind": source["source_kind"],
                    "seed": source["seed"],
                    "stop_reason": source["stop_reason"],
                    "raw_output_text_sha256": source["raw_text_sha256"],
                    "pred_index": index,
                    "generated_order": prediction.get("generated_order"),
                    "description": prediction.get("description"),
                    "normalized_description": _normal_description(prediction.get("description")),
                    "coord_bins": list(prediction.get("coord_bins", [])),
                    "bbox_pixel_xyxy": list(box),
                    "object_span_id": prediction.get("object_span_id"),
                    "raw_span_text": prediction.get("raw_span_text"),
                    "raw_span_sha256": prediction.get("raw_span_sha256"),
                    "char_span": [prediction.get("char_start"), prediction.get("char_end")],
                    "schema_spans": prediction.get("schema_spans"),
                    "coord_token_spans": prediction.get("coord_token_spans"),
                    "top_seed_owner_overlaps": top[:4],
                    "matched_seed_owner_ids_elsewhere_in_output": sorted(matched_owner_ids),
                    "prefix_blocking_boundary": index == first_boundary_pred_index,
                    "disposition": "candidate",
                    "semantics": "reference-unmatched proposal; not verified real, distinct, or trainable",
                }
            )
        for drop in source["drops"]:
            parse_debt.append(
                {
                    "schema": "source256.parse_debt.v1",
                    "image_id": image_id,
                    "example_id": source["example_id"],
                    "source_kind": source["source_kind"],
                    "seed": source["seed"],
                    "raw_output_text_sha256": source["raw_text_sha256"],
                    "debt": drop,
                }
            )

    by_key: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for proposal in proposals:
        by_key[(proposal["image_id"], proposal["normalized_description"])].append(proposal)
    hypotheses: list[dict[str, Any]] = []
    for (image_id, description), members in sorted(by_key.items()):
        remaining = {member["proposal_id"]: member for member in members}
        while remaining:
            first_id = sorted(remaining)[0]
            component_ids = {first_id}
            frontier = [first_id]
            while frontier:
                current_id = frontier.pop()
                current = remaining[current_id]
                for other_id, other in list(remaining.items()):
                    if other_id in component_ids:
                        continue
                    if iou_xyxy(tuple(current["bbox_pixel_xyxy"]), tuple(other["bbox_pixel_xyxy"])) > 0.95:
                        component_ids.add(other_id)
                        frontier.append(other_id)
            component = [remaining.pop(member_id) for member_id in sorted(component_ids)]
            representative = min(
                component,
                key=lambda item: (
                    0 if item["source_kind"] == "source_greedy" else 1,
                    item["seed"] if item["seed"] is not None else -1,
                    item["pred_index"],
                    item["proposal_id"],
                ),
            )
            core = {
                "image_id": image_id,
                "normalized_description": description,
                "member_proposal_ids": sorted(component_ids),
            }
            hypotheses.append(
                {
                    "schema": REVIEW_SCHEMA,
                    "candidate_id": "candidate-" + digest(core, newline=False)[:20],
                    "image_id": image_id,
                    "example_id": representative["example_id"],
                    "image_path": representative["image_path"],
                    "image_width": representative["image_width"],
                    "image_height": representative["image_height"],
                    "description": representative["description"],
                    "normalized_description": description,
                    "coord_bins": representative["coord_bins"],
                    "bbox_pixel_xyxy": representative["bbox_pixel_xyxy"],
                    "representative_proposal_id": representative["proposal_id"],
                    "member_proposal_ids": sorted(component_ids),
                    "members": component,
                    "source_support_count": len({(item["source_kind"], item["seed"]) for item in component}),
                    "greedy_member": any(item["source_kind"] == "source_greedy" for item in component),
                    "prefix_blocking_boundary": any(item["prefix_blocking_boundary"] for item in component),
                    "dedup_policy": "same-image same-category connected components at IoU>0.95; hypothesis only",
                    "disposition": "candidate",
                    "physical_axes": {
                        "entity": "unresolved",
                        "category": "unknown",
                        "relation_to_annotation": "unresolved",
                        "geometry": "unresolved",
                        "uniqueness": "unresolved",
                    },
                }
            )
    hypotheses.sort(
        key=lambda item: (
            -int(item["prefix_blocking_boundary"]),
            -int(item["greedy_member"]),
            -int(item["source_support_count"]),
            int(item["image_id"]),
            str(item["candidate_id"]),
        )
    )
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in hypotheses:
        by_image[int(item["image_id"])].append(item)
    selected: list[dict[str, Any]] = []
    for round_index in range(2):
        for image_id in sorted(by_image):
            if len(selected) >= 128:
                break
            if round_index < len(by_image[image_id]):
                item = dict(by_image[image_id][round_index])
                item["selection"] = {
                    "selected": True,
                    "round": round_index + 1,
                    "policy": "two-round image-balanced; prefix-boundary/greedy/support priority; max128 and max2/image",
                }
                selected.append(item)
    return hypotheses, selected, parse_debt


def validate_preparation(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(value.get("schema") == SCHEMA, "preparation schema")
    routes = value.get("routes")
    require(isinstance(routes, list) and len(routes) == 256, "all256 routes")
    image_ids = [int(record["image_id"]) for record in routes]
    require(len(set(image_ids)) == 256, "unique route images")
    fully_eligible = 0
    for record in routes:
        require(set(record) == {"image_id", "example_id", "canonical_route", "completion_route", "eligibility"}, "route record fields")
        canonical_route = record["canonical_route"]
        validate_route(canonical_route, eos_token_id=EOS, coordinate_token_ids=list(range(COORD_START, COORD_START + 1000)))
        _validate_route_extras(canonical_route)
        eligibility = record["eligibility"]
        completion = record["completion_route"]
        if eligibility["fully_eligible"]:
            fully_eligible += 1
            require(completion is not None and eligibility["fallback_reason"] is None, "eligible route completion")
            validate_route(completion, eos_token_id=EOS, coordinate_token_ids=list(range(COORD_START, COORD_START + 1000)))
            _validate_route_extras(completion)
            require(completion["provenance"]["prefix_owner_ids"] == eligibility["prefix_owner_ids"], "eligible prefix owners")
            require(completion["provenance"]["suffix_owner_ids"] == eligibility["remaining_owner_ids"], "eligible suffix owners")
        else:
            require(completion is None and isinstance(eligibility["fallback_reason"], str), "fallback integrity")
    schedule = value["schedule"]
    require(schedule["update_count"] == len(schedule["updates"]) == 64, "schedule updates")
    common = collections.Counter()
    variable = collections.Counter()
    for index, update in enumerate(schedule["updates"], start=1):
        require(update["step"] == index, "schedule step")
        require(len(update["common_image_ids"]) == len(update["variable_image_ids"]) == 32, "schedule batch")
        common.update(update["common_image_ids"])
        variable.update(update["variable_image_ids"])
    require(set(common) == set(variable) == set(image_ids), "schedule image coverage")
    require(set(common.values()) == set(variable.values()) == {8}, "schedule exposure balance")
    gate = value["gate"]
    effective = sum(variable[record["image_id"]] for record in routes if record["eligibility"]["fully_eligible"])
    require(gate["fully_eligible_count"] == fully_eligible, "gate eligible count")
    require(gate["effective_completion_presentations"] == effective, "gate effective presentations")
    require(gate["total_presentations"] == 4096, "gate denominator")
    expected_pass = fully_eligible >= 64 and effective / 4096 >= 0.125
    require(gate["passed"] == expected_pass, "gate pass")
    require(
        gate["disposition"] == ("ready_for_runtime_qualification" if expected_pass else "STOP_before_main_GPU_training"),
        "gate disposition",
    )
    if verify_sources:
        for name, source in value["sources"].items():
            if isinstance(source, Mapping) and set(source) == {"path", "sha256", "size_bytes"}:
                _verify_binding(source, f"sources.{name}")
            elif isinstance(source, list):
                for index, item in enumerate(source):
                    _verify_binding(item, f"sources.{name}[{index}]")
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(value["content_sha256"] == digest(content), "preparation content digest")
    return dict(value)


def prepare(output: str | Path, *, admissions: str | Path | None = None) -> dict[str, Any]:
    output = Path(output)
    require(not output.exists() or output.is_dir(), "output path exists and is not a directory")
    tokenizer = _load_tokenizer()
    train_rows = read_jsonl(TRAIN_JSONL)
    greedy_rows = read_jsonl(GREEDY_ROOT / "gt_vs_pred.jsonl")
    plan = read_json(K4_PLAN)
    identity = _identity(plan, train_rows, greedy_rows)
    traces, pad_total = _active_greedy_traces(GREEDY_ROOT / "pred_token_trace.jsonl")
    require(pad_total == EXPECTED["greedy_pad_tokens"], "greedy trailing pad count")
    greedy_receipt = _verify_greedy(tokenizer, greedy_rows, traces)
    rollouts, k4_receipt = _verify_k4(tokenizer, plan)
    bank = _seed_bank(train_rows)
    admissions_value, admission_entries = _load_admissions(admissions)
    admission_receipt = _apply_admissions(bank, admission_entries)
    admissions_binding = binding(admissions) if admissions is not None else None
    groups = {int(group["image_id"]): group for group in plan["population"]["groups"]}
    greedy_by_image = {
        int(str(row["row_id"]).rsplit("_", 1)[-1]): {**row, "active_traces": traces[str(row["row_id"])]}
        for row in greedy_rows
    }
    routes: list[dict[str, Any]] = []
    fallback_counts: collections.Counter[str] = collections.Counter()
    prefix_counts: collections.Counter[int] = collections.Counter()
    structurally_eligible = 0
    fully_eligible = 0
    bank_records: list[dict[str, Any]] = []
    for source in train_rows:
        image_id = int(source["image_id"])
        owners = bank[image_id]
        group = groups[image_id]
        greedy = greedy_by_image[image_id]
        evidence = _prefix_evidence(greedy, greedy["active_traces"], owners)
        prefix_counts[int(evidence["prefix_length"])] += 1
        structurally_eligible += int(evidence["structural_candidate"])
        canonical_route = _route(
            tokenizer=tokenizer,
            route_kind="canonical",
            source=source,
            group=group,
            greedy=greedy,
            owners=owners,
            prefix=None,
        )
        completion_route = None
        fallback_reason: str | None = None
        if not evidence["prefix_owner_ids"]:
            fallback_reason = "empty_trusted_prefix"
        elif not evidence["remaining_owner_ids"]:
            fallback_reason = "no_remaining_bank_owner"
        elif not evidence["structural_candidate"]:
            fallback_reason = "structural_prefix_ineligible"
        else:
            try:
                completion_route = _route(
                    tokenizer=tokenizer,
                    route_kind="fixed_source_prefix_completion",
                    source=source,
                    group=group,
                    greedy=greedy,
                    owners=owners,
                    prefix=evidence,
                )
            except ValueError as exc:
                if "exceeds frozen continuation cap" not in str(exc):
                    raise
                fallback_reason = "completion_exceeds_frozen_3084_cap"
        is_eligible = completion_route is not None
        fully_eligible += int(is_eligible)
        if fallback_reason is not None:
            fallback_counts[fallback_reason] += 1
        eligibility = {
            "structural_candidate": bool(evidence["structural_candidate"]),
            "fully_eligible": is_eligible,
            "prefix_length": evidence["prefix_length"],
            "prefix_owner_ids": evidence["prefix_owner_ids"],
            "remaining_owner_ids": evidence["remaining_owner_ids"],
            "first_boundary": evidence["boundary"],
            "fallback_reason": fallback_reason,
            "admission_semantics": "registered reference/matching plus versioned accepted additions/exclusions; no claim of exhaustive scene review",
        }
        routes.append(
            {
                "image_id": image_id,
                "example_id": group["example_id"],
                "canonical_route": canonical_route,
                "completion_route": completion_route,
                "eligibility": eligibility,
            }
        )
        bank_records.append(
            {
                "image_id": image_id,
                "example_id": group["example_id"],
                "owners": owners,
                "owner_count": len(owners),
            }
        )
    require(len(routes) == 256, "route count")
    source_by_image = {int(row["image_id"]): row for row in train_rows}
    group_by_image = {int(group["image_id"]): group for group in groups.values()}
    review_sidecar_records: list[dict[str, Any]] = []
    for record in bank_records:
        image_id = int(record["image_id"])
        source = source_by_image[image_id]
        group = group_by_image[image_id]
        review_sidecar_records.append(
            {
                "image_id": image_id,
                "example_id": record["example_id"],
                "image_path": group["image_path"],
                "image_width": int(source["width"]),
                "image_height": int(source["height"]),
                "owners": [
                    {
                        "owner_id": owner["owner_id"],
                        "description": owner["description"],
                        "normalized_description": owner["normalized_description"],
                        "coord_bins": owner["coord_bins"],
                        "bbox_pixel_xyxy": list(
                            coord_bins_to_pixel_xyxy(
                                owner["coord_bins"],
                                image_width=int(source["width"]),
                                image_height=int(source["height"]),
                                field=f"review_sidecar[{image_id}].{owner['owner_id']}",
                            )
                        ),
                        "source": owner["source"],
                    }
                    for owner in record["owners"]
                ],
            }
        )
    review_sidecar: dict[str, Any] = {
        "schema": "source256.trusted_owner_review_sidecar.v1",
        "image_count": 256,
        "owner_count": sum(len(record["owners"]) for record in review_sidecar_records),
        "records": review_sidecar_records,
        "semantics": "same-image trusted-bank association for duplicate/localization/repair review; not detector output",
        "content_sha256": None,
    }
    review_sidecar["content_sha256"] = digest(
        {key: value for key, value in review_sidecar.items() if key != "content_sha256"}
    )
    schedule = _schedule([record["image_id"] for record in routes])
    effective = fully_eligible * schedule["variable_exposures_per_image"]
    fraction = effective / schedule["total_presentations"]
    passed = fully_eligible >= 64 and fraction >= 0.125
    gate = {
        "required_fully_eligible_count": 64,
        "required_effective_completion_fraction": 0.125,
        "fully_eligible_count": fully_eligible,
        "effective_completion_presentations": effective,
        "total_presentations": schedule["total_presentations"],
        "effective_completion_fraction": fraction,
        "hypothetical_half_mixture_fraction": 0.5 * fully_eligible / 256,
        "passed": passed,
        "disposition": "ready_for_runtime_qualification" if passed else "STOP_before_main_GPU_training",
        "no_automatic_support_expansion": True,
    }
    hypotheses, selected, parse_debt = _review_proposals(greedy_rows, rollouts, bank)
    sources: dict[str, Any] = {
        "producer": binding(Path(__file__)),
        "train_jsonl": binding(TRAIN_JSONL),
        "dev_jsonl": binding(DEV_JSONL),
        "input_manifest": binding(INPUT_MANIFEST),
        "canonical_processed_source": binding(CANONICAL_SOURCE),
        "greedy_rows": binding(GREEDY_ROOT / "gt_vs_pred.jsonl"),
        "greedy_token_trace": binding(GREEDY_ROOT / "pred_token_trace.jsonl"),
        "greedy_manifest": binding(GREEDY_ROOT / "run_manifest.json"),
        "k4_plan": binding(K4_PLAN),
        "k4_acquisition_shards": [binding(item["path"]) for item in plan["sources"]["rollout_artifacts"]],
        "opportunity_summary": binding(OPPORTUNITY_SUMMARY),
        "tokenizer": binding(MODEL_ROOT / "tokenizer.json"),
        "adapter_weights": binding(CHECKPOINT_ROOT / "adapter/adapter_model.safetensors"),
        "embedding_weights": binding(
            CHECKPOINT_ROOT / "special_token_embeddings/special_token_embeddings.safetensors"
        ),
    }
    if admissions_binding is not None:
        sources["admissions"] = admissions_binding
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "candidate_ready" if passed else "stopped_by_scarcity_gate",
        "sources": sources,
        "identity": {
            **identity,
            "greedy": {**greedy_receipt, "trailing_pad_tokens_excluded": pad_total},
            "k4": k4_receipt,
        },
        "admissions": {
            "schema": admissions_value["schema"],
            "version": admissions_value["version"],
            **admission_receipt,
        },
        "bank": {
            "schema": "source256.trusted_owner_bank.v1",
            "image_count": len(bank_records),
            "owner_count": sum(record["owner_count"] for record in bank_records),
            "records": bank_records,
        },
        "census": {
            "image_count": 256,
            "structurally_eligible_count": structurally_eligible,
            "fully_eligible_count": fully_eligible,
            "prefix_length_counts": {str(key): value for key, value in sorted(prefix_counts.items())},
            "fallback_counts": dict(sorted(fallback_counts.items())),
            "semantics": {
                "structural": "contiguous greedy rows selected by frozen category-constrained one-to-one IoU50 reference matching",
                "fully_eligible": "structural route against current versioned bank, nonempty prefix, remaining owners, and cap fit",
                "limit": "IoU/reference admission is not an exhaustive physical-truth review; unmatched and conflicts remain unknown/HOLD",
            },
        },
        "routes": routes,
        "schedule": schedule,
        "gate": gate,
        "review": {
            "schema": REVIEW_SCHEMA,
            "raw_proposal_count": sum(len(item["members"]) for item in hypotheses),
            "owner_hypothesis_count": len(hypotheses),
            "selected_review_count": len(selected),
            "parse_debt_count": len(parse_debt),
            "review_limit": 128,
            "per_image_limit": 2,
            "all_candidates_file": "unmatched-owner-hypotheses.jsonl",
            "selected_candidates_file": "review-candidates.jsonl",
            "parse_debt_file": "parse-debt.jsonl",
            "trusted_owner_sidecar_file": "trusted-owner-review-sidecar.json",
            "admission_example_file": "review-admissions.example.json",
        },
        "content_sha256": None,
    }
    manifest["content_sha256"] = digest({key: value for key, value in manifest.items() if key != "content_sha256"})
    validate_preparation(manifest)
    publish_jsonl(output / "unmatched-owner-hypotheses.jsonl", hypotheses)
    publish_jsonl(output / "review-candidates.jsonl", selected)
    publish_jsonl(output / "parse-debt.jsonl", parse_debt)
    publish(output / "trusted-owner-review-sidecar.json", review_sidecar)
    publish(output / "review-admissions.example.json", admission_example())
    publish(output / "preparation.json", manifest)
    receipt = {
        "schema": "source256.fixed_prefix_preparation.receipt.v1",
        "command": (
            "python -m probes.training_set_completion.source256_data prepare "
            f"--output {output}"
            + (f" --admissions {Path(admissions).resolve()}" if admissions is not None else "")
        ),
        "preparation": binding(output / "preparation.json"),
        "all_candidates": binding(output / "unmatched-owner-hypotheses.jsonl"),
        "selected_candidates": binding(output / "review-candidates.jsonl"),
        "parse_debt": binding(output / "parse-debt.jsonl"),
        "trusted_owner_sidecar": binding(output / "trusted-owner-review-sidecar.json"),
        "admission_example": binding(output / "review-admissions.example.json"),
        "gate": gate,
        "counts": {
            "images": 256,
            "bank_owners": manifest["bank"]["owner_count"],
            "structurally_eligible": structurally_eligible,
            "fully_eligible": fully_eligible,
            "review_candidates": len(selected),
        },
    }
    publish(output / "result-receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--admissions", type=Path)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--preparation", type=Path, required=True)
    validate_parser.add_argument("--no-verify-sources", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, admissions=args.admissions)
    else:
        result = validate_preparation(
            read_json(args.preparation), verify_sources=not args.no_verify_sources
        )["gate"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
