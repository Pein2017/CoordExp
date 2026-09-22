#!/usr/bin/env python3
"""Certified 46-owner Route-C/Route-S single-image overfit, Phase 0+1 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import traceback
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import build_sorted_owner_basin_census as canonical_matcher
from scripts.research import run_image2299_decision_microscope as microscope
from scripts.research import run_image2299_native_state_successor_lattice as lattice
from scripts.research import run_image2299_xy_single_edge_owner_compilation as full_root
from src.inference.backend import open_backend_session, token_ids_sha256
from src.inference.hf_backend import HFBackendSession
from src.inference.parsing import parse_compact_object_box


SCHEMA_VERSION = "image2299.certified_46_owner_path_overfit.v1"
PHASE2_SCHEMA_VERSION = "image2299.certified_46_owner_path_overfit.phase2.v1"
UNIT_ID = "2026-08-29-image2299-certified-46-owner-path-overfit"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
PARENT_CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-27-image2299-owner-aware-sparse-margin/s_gain/"
    "20260827T-s-gain-step04-parent-materialization-v1/checkpoint-step-04"
)
PARENT_NATIVE = PARENT_CHECKPOINT.parent / "native-step-04.json"
PARENT_ROUTE_SHA256 = "f0b762c6a54b03015e27e5ec2794c8b8c8f2403607f18eece890047b4ffb05fd"
PROMPT_TOKEN_SHA256 = "33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb"
TARGET_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-26-image2299-full-root-detached-margin/target-root/target-root-v2/target.json"
)
TARGET_SHA256 = "22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988"
AUTHORITY_PATH = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl")
AUTHORITY_SHA256 = "81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894"
AUTHORITY_LINE_SHA256 = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
SOURCE_GATE_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
SOURCE_GATE_FILES = {
    "docs/history/architecture/proposals/2026-06-27-coordexp-swift/source-studies/special-token-embeddings.md":
        "e024f8f9754475cfa6ed81136eae6c72becd2aca53b7b03b9fa3047e8da7d193",
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json":
        "7da3b22b11ef8a0957bedfe78cac16498313cf724e87cfc921caa0ab7c9f68e4",
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/special_token_embeddings.json":
        "f97e45aa65e30a1a9bca2ec3fd1e31c6d248594eb5628aec74ffa13d67366b47",
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/special_token_embeddings.safetensors":
        "8b062a33dccb49bad617b06347462ddce4aaf3264e3dae5c32d33563963e7592",
}
MISSING_OWNERS = frozenset(
    {
        "gt:2299:0", "gt:2299:1", "gt:2299:8", "gt:2299:9", "gt:2299:11",
        "gt:2299:14", "gt:2299:18", "gt:2299:20", "gt:2299:32", "gt:2299:34",
        "gt:2299:35", "gt:2299:43", "gt:2299:45",
    }
)
MISSING_OWNER_ORDER = (
    "gt:2299:35", "gt:2299:18", "gt:2299:45", "gt:2299:20", "gt:2299:11",
    "gt:2299:1", "gt:2299:9", "gt:2299:0", "gt:2299:8", "gt:2299:14",
    "gt:2299:43", "gt:2299:34", "gt:2299:32",
)
ROUTE_SHA256 = {
    "C": "5feb64a1059f21e29a6ba8adb2f8538eeb1a44eeb7d71f572694c4efa0feeced",
    "S": "5a46bf9878b2c07f6db6429f09b51d5fcf1bc9f01704b50490f0b8288a8e1656",
}
WORLD_SIZE, ROW_TOKENS, EOS = 8, 9, 151645
LEARNING_RATE, GRAD_CLIP, NATURAL_MAX_TOKENS = 2.0**-11, 1.0, 512
GROUPS = {"C": (0, 1, 2, 3), "S": (4, 5, 6, 7)}
PHASE1_TRIGGER_RECEIPT = (
    OUTPUT_ROOT / "foreground" / "20260829T-certified46-foreground-v1" / "receipt.json"
)
PHASE1_TRIGGER_SHA256 = "12a681c5261f04546329f0f340014837763e2d027b6475979554c7152367e4a3"
PHASE1_TRIGGER_STATUS = "phase1_complete_needs_frontier"
PHASE2_BRANCHES, PHASE2_STEPS = 8, 8


class Certified46Hold(RuntimeError):
    """An immutable identity or Phase-1 mechanics contract failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise Certified46Hold("HOLD: unsafe run_id")
    return value


@lru_cache(maxsize=1)
def _load_authority() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Bind the live source row; the deleted historical panel is never consulted."""
    if _sha256(AUTHORITY_PATH) != AUTHORITY_SHA256:
        raise Certified46Hold("HOLD: authority JSONL SHA drifted")
    matches = [
        candidate for candidate in AUTHORITY_PATH.read_bytes().splitlines(keepends=True)
        if str(json.loads(candidate).get("image_id")) == "2299"
    ]
    if len(matches) != 1 or hashlib.sha256(matches[0]).hexdigest() != AUTHORITY_LINE_SHA256:
        raise Certified46Hold("HOLD: authority image2299 line SHA drifted")
    line = matches[0].decode("utf-8")
    records, owners_by_image, source = canonical_matcher._load_panel(AUTHORITY_PATH)
    owners = owners_by_image.get("2299", [])
    if len(records) < 1 or len(owners) != 46:
        raise Certified46Hold("HOLD: authority matcher row lacks exactly 46 owners")
    return json.loads(line), owners, {"whole_sha256": AUTHORITY_SHA256, "line_sha256": AUTHORITY_LINE_SHA256, **source}


def _parse_and_match(*, tokenizer: Any, generated_token_ids: Sequence[int], label: str, raw_example: Any) -> dict[str, Any]:
    """Decision-microscope parser/matcher shape, but with the training authority row."""
    _line, owners, _binding = _load_authority()
    text = str(tokenizer.decode(list(map(int, generated_token_ids)), skip_special_tokens=False))
    parsed = parse_compact_object_box(
        text, assistant_format="object_box_closed", row_id="coco2017_val_000000002299",
        row_index=0, image_width=raw_example.image.width, image_height=raw_example.image.height,
    )
    predictions = [
        {
            "pred_row_id": f"pred:certified46:{label}:{int(item['generated_order'])}",
            "original_row_index": int(item["generated_order"]),
            "normalized_description": canonical_matcher._normalize_description(item["description"]),
            "bbox_xyxy": list(item["bbox"]),
        }
        for item in parsed.predictions
    ]
    trajectory = {
        "trajectory_id": f"trajectory:certified46:{label}:2299", "image_id": "2299",
        "decode_mode": "greedy", "seed": 0,
        "source_artifact_sha256": hashlib.sha256(str(tuple(generated_token_ids)).encode()).hexdigest(),
        "predictions": predictions,
    }
    return {
        "label": label, "generated_token_ids_sha256": token_ids_sha256(generated_token_ids),
        "parse": parsed.to_artifact_dict(),
        "matcher": canonical_matcher._trajectory_receipt(trajectory, owners, {}),
    }


def _strict_owner_order(ledger: Mapping[str, Any]) -> tuple[str, ...]:
    parse, matcher = ledger["parse"], ledger["matcher"]
    predictions = list(parse.get("predictions", ()))
    receipts = list(matcher.get("prediction_receipts", ()))
    owners = tuple(str(row.get("strict_match_gt_owner_id", "")) for row in receipts)
    if (
        len(predictions) != len(receipts)
        or int(parse.get("dropped_prediction_count", -1)) != 0
        or len(owners) != len(set(owners))
        or not owners
        or any(row.get("strict_match_status") != "matched" for row in receipts)
        or any(not owner for owner in owners)
    ):
        raise Certified46Hold("HOLD: strict unique matcher ledger has debt")
    return owners


def _parent_tokens() -> list[int]:
    payload = json.loads(PARENT_NATIVE.read_text(encoding="utf-8"))
    tokens = list(map(int, payload.get("generated_token_ids", ())))
    if (
        payload.get("generated_token_ids_sha256") != PARENT_ROUTE_SHA256
        or token_ids_sha256(tokens) != PARENT_ROUTE_SHA256
        or len(tokens) != 298
        or tokens[-1:] != [EOS]
        or (len(tokens) - 1) // ROW_TOKENS != 33
    ):
        raise Certified46Hold("HOLD: Parent route identity/shape drifted")
    return tokens


def _assert_missing_membership(current: Sequence[str]) -> None:
    expected_current = {f"gt:2299:{index}" for index in range(46)} - set(MISSING_OWNERS)
    if set(map(str, current)) != expected_current or len(current) != 33:
        raise Certified46Hold("HOLD: current Parent membership does not yield the frozen 13-owner deficit")


def _route_manifest(name: str, owners: Sequence[str], tokens: Sequence[int]) -> dict[str, Any]:
    if len(owners) != 46 or len(tokens) != 415 or tokens[-1] != EOS or EOS in tokens[:-1]:
        raise Certified46Hold("HOLD: compiled route is not 46 rows plus unique EOS")
    return {
        "route": name, "owner_order": list(owners), "owner_order_sha256": _hash(list(owners)),
        "token_count": len(tokens), "token_ids_sha256": token_ids_sha256(tokens),
        "row_hashes": [token_ids_sha256(tokens[index * ROW_TOKENS:(index + 1) * ROW_TOKENS]) for index in range(46)],
    }


def _compile_routes(*, target: Mapping[str, Any], parent_tokens: Sequence[int], parent_ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(target.get("rows", ()))
    if len(rows) != 46 or any(not isinstance(row.get("token_ids"), list) or len(row["token_ids"]) != ROW_TOKENS for row in rows):
        raise Certified46Hold("HOLD: target rows drifted")
    target_order = [str(row["owner"]) for row in rows]
    if len(target_order) != len(set(target_order)):
        raise Certified46Hold("HOLD: target owner order is not unique")
    current = _strict_owner_order(parent_ledger)
    _assert_missing_membership(current)
    if current != tuple(owner for owner in target_order if owner not in MISSING_OWNERS):
        raise Certified46Hold("HOLD: Parent owner order is not the canonical-filtered target subsequence")
    parent_rows = [list(map(int, parent_tokens[index * ROW_TOKENS:(index + 1) * ROW_TOKENS])) for index in range(33)]
    by_parent = dict(zip(current, parent_rows, strict=True))
    by_target = {str(row["owner"]): list(map(int, row["token_ids"])) for row in rows}
    missing_order = [owner for owner in target_order if owner in MISSING_OWNERS]
    if tuple(missing_order) != MISSING_OWNER_ORDER:
        raise Certified46Hold("HOLD: canonical missing owner order drifted")
    route_c_owners = target_order
    route_c_rows = [by_parent.get(owner, by_target[owner]) for owner in route_c_owners]
    route_s_owners = [*current, *missing_order]
    route_s_rows = [*parent_rows, *(by_target[owner] for owner in missing_order)]
    routes = {
        "C": {"owners": route_c_owners, "tokens": [*sum(route_c_rows, []), EOS]},
        "S": {"owners": route_s_owners, "tokens": [*sum(route_s_rows, []), EOS]},
    }
    parent_hashes = [token_ids_sha256(row) for row in parent_rows]
    for name, route in routes.items():
        manifest = _route_manifest(name, route["owners"], route["tokens"])
        if manifest["token_ids_sha256"] != ROUTE_SHA256[name]:
            raise Certified46Hold(f"HOLD: Route {name} golden token identity drifted")
        actual_parent_hashes = [manifest["row_hashes"][route["owners"].index(owner)] for owner in current]
        if actual_parent_hashes != parent_hashes:
            raise Certified46Hold(f"HOLD: Route {name} did not preserve exact Parent rows")
        manifest["parent_row_hashes"] = parent_hashes
        route["manifest"] = manifest
    return routes


def _phase1_trigger() -> dict[str, Any]:
    """Bind the failed Phase-1 receipt without ever using its checkpoint."""
    if not PHASE1_TRIGGER_RECEIPT.is_file() or _sha256(PHASE1_TRIGGER_RECEIPT) != PHASE1_TRIGGER_SHA256:
        raise Certified46Hold("HOLD: Phase-1 frontier trigger receipt identity drifted")
    receipt = json.loads(PHASE1_TRIGGER_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("status") != PHASE1_TRIGGER_STATUS:
        raise Certified46Hold("HOLD: Phase-1 receipt does not authorize the frontier")
    failed_checkpoint = Path(str(receipt.get("checkpoint", "")))
    if not failed_checkpoint or failed_checkpoint == PARENT_CHECKPOINT:
        raise Certified46Hold("HOLD: Phase-1 trigger lacks a distinct failed checkpoint")
    return {
        "path": str(PHASE1_TRIGGER_RECEIPT),
        "sha256": PHASE1_TRIGGER_SHA256,
        "status": PHASE1_TRIGGER_STATUS,
        "failed_checkpoint": str(failed_checkpoint),
        "champion_checkpoint": str(PARENT_CHECKPOINT),
        "failed_checkpoint_never_loaded_as_champion": True,
    }


def _target_rows(target: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [dict(row) for row in target.get("rows", ())]
    if len(rows) != 46:
        raise Certified46Hold("HOLD: target lacks exactly 46 immutable rows")
    by_owner = {str(row.get("owner")): row for row in rows}
    if len(by_owner) != 46 or any(len(list(row.get("token_ids", ()))) != ROW_TOKENS for row in rows):
        raise Certified46Hold("HOLD: target immutable rows are malformed")
    return rows, by_owner


def _parse_compact_row(*, tokenizer: Any, row_tokens: Sequence[int], raw_example: Any, row_index: int) -> tuple[dict[str, Any] | None, str]:
    """Parse one known nine-token boundary; no global assignment is consulted."""
    if len(row_tokens) != ROW_TOKENS or EOS in row_tokens:
        return None, "untrusted_segment"
    try:
        parsed = parse_compact_object_box(
            str(tokenizer.decode(list(map(int, row_tokens)), skip_special_tokens=False)),
            assistant_format="object_box_closed", row_id="coco2017_val_000000002299",
            row_index=row_index, image_width=raw_example.image.width, image_height=raw_example.image.height,
        )
    except Exception:
        return None, "invalid"
    artifact = parsed.to_artifact_dict()
    predictions = list(artifact.get("predictions", ()))
    dropped = list(artifact.get("dropped_predictions", ()))
    if len(predictions) != 1 or dropped or int(artifact.get("dropped_prediction_count", -1)) != 0:
        status = "unsupported" if artifact.get("parse_status") == "unsupported_format" else "invalid"
        return None, status
    return dict(predictions[0]), "valid"


def _causal_locked_ledger(*, tokenizer: Any, tokens: Sequence[int], raw_example: Any) -> dict[str, Any]:
    """Chronological matcher for the causal frontier; intentionally not Hungarian."""
    _line, owners, _binding = _load_authority()
    sequence = list(map(int, tokens))
    locked: set[str] = set()
    valid_people: list[tuple[float, float, float, float]] = []
    rows: list[dict[str, Any]] = []
    first_event: dict[str, Any] | None = None
    cursor = 0
    segmentation_trusted = True

    while cursor < len(sequence):
        if sequence[cursor] == EOS:
            if cursor != len(sequence) - 1:
                segmentation_trusted = False
                break
            if first_event is None and len(locked) < len(owners):
                first_event = {
                    "kind": "premature_eos", "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                    "bad_span_tokens": [EOS], "divergence": 0,
                }
            cursor += 1
            break
        if cursor + ROW_TOKENS > len(sequence) or EOS in sequence[cursor:cursor + ROW_TOKENS]:
            segmentation_trusted = False
            break

        row_tokens = sequence[cursor:cursor + ROW_TOKENS]
        prediction, parse_state = _parse_compact_row(
            tokenizer=tokenizer, row_tokens=row_tokens, raw_example=raw_example, row_index=cursor // ROW_TOKENS,
        )
        record: dict[str, Any] = {
            "row_index": cursor // ROW_TOKENS, "token_start": cursor,
            "row_tokens_sha256": token_ids_sha256(row_tokens), "parse_state": parse_state,
        }
        if prediction is None:
            record["decision"] = parse_state
            if first_event is None:
                first_event = {"kind": parse_state, "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                               "bad_span_tokens": row_tokens, "divergence": 0}
            rows.append(record)
            cursor += ROW_TOKENS
            continue

        normalized = canonical_matcher._normalize_description(prediction.get("description"))
        box = tuple(float(value) for value in prediction["bbox"])
        compatible = [
            (str(owner["gt_owner_id"]), canonical_matcher._iou(box, owner["bbox_xyxy"]))
            for owner in owners
            if canonical_matcher._compatible_description(
                {"normalized_description": normalized}, owner, {}
            )
        ]
        strict = [(owner, iou) for owner, iou in compatible if iou >= canonical_matcher.IOU_THRESHOLD]
        raw_duplicate = normalized == "person" and any(
            canonical_matcher._iou(box, earlier) >= 0.95 for earlier in valid_people
        )
        record.update({
            "normalized_description": normalized, "bbox_xyxy": list(box),
            "strict_edges": [{"owner": owner, "iou": iou} for owner, iou in strict],
            "raw_person_duplicate": raw_duplicate,
        })
        if normalized == "person":
            valid_people.append(box)
        if raw_duplicate:
            record["decision"] = "duplicate"
            if len(strict) == 1:
                record["owner"] = strict[0][0]
            if first_event is None:
                first_event = {
                    "kind": "duplicate", "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                    "bad_span_tokens": row_tokens, "divergence": 0,
                    "raw_person_duplicate": True,
                    **({"locked_owner": strict[0][0]} if len(strict) == 1 else {}),
                }
        elif len(strict) == 1 and strict[0][0] not in locked:
            owner = strict[0][0]
            locked.add(owner)
            record.update({"decision": "accepted", "owner": owner})
        elif len(strict) == 1 and strict[0][0] in locked:
            record.update({"decision": "duplicate", "owner": strict[0][0]})
            if first_event is None:
                first_event = {
                    "kind": "duplicate",
                    "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                    "bad_span_tokens": row_tokens, "divergence": 0,
                    "locked_owner": strict[0][0], "raw_person_duplicate": raw_duplicate,
                }
        elif len(strict) > 1:
            record["decision"] = "ambiguous"
            if first_event is None:
                first_event = {"kind": "ambiguous", "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                               "bad_span_tokens": row_tokens, "divergence": 0}
        else:
            positive = [(owner, iou) for owner, iou in compatible if iou > 0.0]
            best = max((iou for _owner, iou in positive), default=0.0)
            tied = [owner for owner, iou in positive if math.isclose(iou, best, abs_tol=1e-12, rel_tol=0.0)]
            if best > 0.0 and len(tied) == 1:
                record.update({"decision": "near_miss", "near_miss_owner": tied[0], "near_miss_iou": best})
                kind = "near_miss"
            elif best > 0.0:
                record.update({"decision": "ambiguous", "near_miss_ties": tied, "near_miss_iou": best})
                kind = "ambiguous"
            else:
                record["decision"] = "invalid_unmatched"
                kind = "invalid_unmatched"
            if first_event is None:
                first_event = {"kind": kind, "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                               "bad_span_tokens": row_tokens, "divergence": 0,
                               **({"near_miss_owner": tied[0]} if kind == "near_miss" else {})}
        rows.append(record)
        cursor += ROW_TOKENS

    if not segmentation_trusted:
        first_event = {"kind": "untrusted_segmentation", "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                       "bad_span_tokens": sequence[cursor:], "divergence": 0}
    elif cursor == len(sequence) and (not sequence or sequence[-1] != EOS) and first_event is None:
        first_event = {"kind": "nontermination", "event_start": cursor, "row_index": cursor // ROW_TOKENS,
                       "bad_span_tokens": [], "divergence": 0}
    all_owners = {str(owner["gt_owner_id"]) for owner in owners}
    hard = 0 if first_event is None or first_event["kind"] == "premature_eos" else 1
    return {
        "schema": "causal_locked_ledger.v1", "segmentation_trusted": segmentation_trusted,
        "rows": rows, "locked_owner_ids": [row["owner"] for row in rows if row.get("decision") == "accepted"],
        "final_missing_owner_ids": [owner for owner in sorted(all_owners) if owner not in locked],
        "first_event": first_event, "causal_hard_counter_count": hard,
        "token_count": len(sequence), "token_ids_sha256": token_ids_sha256(sequence),
    }


def _assert_parent_causal_ledger(ledger: Mapping[str, Any]) -> None:
    if (
        not ledger.get("segmentation_trusted")
        or len(ledger.get("locked_owner_ids", ())) != 33
        or set(map(str, ledger.get("final_missing_owner_ids", ()))) != set(MISSING_OWNERS)
        or ledger.get("first_event", {}).get("kind") != "premature_eos"
        or int(ledger.get("first_event", {}).get("event_start", -1)) != 297
        or int(ledger.get("causal_hard_counter_count", -1)) != 0
    ):
        raise Certified46Hold("HOLD: Parent causal ledger is not 33 accepts plus EOS at token 297")


def _compile_dynamic_route(*, target: Mapping[str, Any], natural_tokens: Sequence[int], causal_ledger: Mapping[str, Any], preferred_owner: str | None) -> dict[str, Any]:
    if not causal_ledger.get("segmentation_trusted"):
        raise Certified46Hold("HOLD: dynamic route cannot use untrusted segmentation")
    rows, by_owner = _target_rows(target)
    locked = list(map(str, causal_ledger.get("locked_owner_ids", ())))
    if len(locked) != len(set(locked)) or any(owner not in by_owner for owner in locked):
        raise Certified46Hold("HOLD: causal locked backbone identity is malformed")
    natural = list(map(int, natural_tokens))
    row_tokens = {
        str(row["owner"]): natural[int(record["token_start"]):int(record["token_start"]) + ROW_TOKENS]
        for record in causal_ledger.get("rows", ()) if record.get("decision") == "accepted"
        for row in [{"owner": record["owner"]}]
    }
    if set(row_tokens) != set(locked) or any(len(value) != ROW_TOKENS for value in row_tokens.values()):
        raise Certified46Hold("HOLD: causal backbone rows are not exact compact rows")
    missing = [str(row["owner"]) for row in rows if str(row["owner"]) not in row_tokens]
    if not missing:
        raise Certified46Hold("HOLD: causal frontier has no trusted missing target pool")
    if preferred_owner is not None:
        if preferred_owner not in missing:
            raise Certified46Hold("HOLD: frontier preferred owner is not still missing")
        missing = [preferred_owner, *[owner for owner in missing if owner != preferred_owner]]
    owners = [*locked, *missing]
    tokens = [*sum((row_tokens.get(owner, list(by_owner[owner]["token_ids"])) for owner in owners), []), EOS]
    manifest = _route_manifest("dynamic", owners, tokens)
    manifest.update({"backbone_owner_ids": locked, "missing_owner_ids": missing, "preferred_owner": preferred_owner})
    return {"owners": owners, "tokens": tokens, "manifest": manifest}


def _frontier_span_partition(*, start: int, end: int, worker_index: int, worker_count: int) -> tuple[int, ...]:
    if start < 0 or end < start or worker_count != WORLD_SIZE or worker_index not in range(worker_count):
        raise Certified46Hold("HOLD: malformed frontier span partition")
    return tuple(position for position in range(start, end + 1) if (position - start) % worker_count == worker_index)


def _frontier_supervision_span(*, event_start: int, bad_span_tokens: Sequence[int], target_row_tokens: Sequence[int]) -> tuple[int, int, int]:
    divergence = _exact_prefix(bad_span_tokens, target_row_tokens)
    if event_start < 0 or len(target_row_tokens) != ROW_TOKENS or divergence >= ROW_TOKENS:
        raise Certified46Hold("HOLD: frontier target has no divergent compact token")
    return divergence, event_start + divergence, event_start + ROW_TOKENS - 1


def _owner_keyed_route_prefix(*, causal_ledger: Mapping[str, Any], natural_tokens: Sequence[int], route: Mapping[str, Any]) -> int:
    route_rows = {
        str(owner): list(map(int, route["tokens"][index * ROW_TOKENS:(index + 1) * ROW_TOKENS]))
        for index, owner in enumerate(route["owners"])
    }
    natural = list(map(int, natural_tokens))
    prefix = 0
    for record in causal_ledger.get("rows", ()):
        if record.get("decision") != "accepted":
            break
        owner, start = str(record["owner"]), int(record["token_start"])
        if owner not in route_rows or natural[start:start + ROW_TOKENS] != route_rows[owner]:
            break
        prefix += 1
    return prefix


def _admit_route(*, tokenizer: Any, route: Mapping[str, Any], target: Mapping[str, Any], raw_example: Any, label: str) -> dict[str, Any]:
    ledger = _parse_and_match(tokenizer=tokenizer, generated_token_ids=route["tokens"], label=label, raw_example=raw_example)
    gate = full_root._full_root_joint_owner_equivalence_from_ledger(tokens=route["tokens"], ledger=ledger, target=target)
    if not gate["passed"] or gate["matcher"]["committed_owner_count"] != 46:
        raise Certified46Hold(f"HOLD: Route {route['manifest']['route']} failed exact 46/46 joint owner admission")
    return {"ledger": ledger, "joint_gate": gate}


def _macro_actions(token_count: int) -> tuple[tuple[int, ...], ...]:
    if token_count != 415:
        raise Certified46Hold("HOLD: Phase-1 loss requires exactly 415 target tokens")
    actions = tuple(tuple(range(index * ROW_TOKENS, (index + 1) * ROW_TOKENS)) for index in range(46)) + ((414,),)
    if tuple(position for action in actions for position in action) != tuple(range(token_count)):
        raise Certified46Hold("HOLD: macro action partition does not supervise every token exactly once")
    return actions


def _macro_loss(logits: torch.Tensor, tokens: Sequence[int], *, worker_index: int, worker_count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if logits.ndim != 2 or logits.shape[0] != len(tokens) or worker_count < 1 or worker_index not in range(worker_count):
        raise Certified46Hold("HOLD: macro loss inputs are malformed")
    actions = _macro_actions(len(tokens))
    targets = torch.tensor(tokens, dtype=torch.long, device=logits.device)
    chosen = [index for index in range(len(actions)) if index % worker_count == worker_index]
    terms = [F.cross_entropy(logits[list(actions[index])], targets[list(actions[index])]).mean() / len(actions) for index in chosen]
    if not terms:
        raise Certified46Hold("HOLD: rank owns no macro actions")
    loss = torch.stack(terms).sum()
    return loss, {"action_indices": chosen, "action_count": len(chosen), "token_count": sum(len(actions[index]) for index in chosen), "weight": 1.0 / 47.0}


def _strict_margins(logits: torch.Tensor, tokens: Sequence[int]) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[0] != len(tokens) or len(tokens) != 415 or not bool(torch.isfinite(logits).all().item()):
        raise Certified46Hold("HOLD: strict-margin logits are malformed")
    target = torch.tensor(tokens, dtype=torch.long, device=logits.device)
    values, ids = torch.topk(logits, k=2, dim=-1)
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    competitor_ids = torch.where(ids[:, 0] == target, ids[:, 1], ids[:, 0])
    competitor_logits = logits.gather(1, competitor_ids[:, None]).squeeze(1)
    margins = target_logits - competitor_logits
    records = [
        {"position": index, "target_token_id": int(target[index]), "competitor_token_id": int(competitor_ids[index]), "margin": float(margins[index])}
        for index in range(len(tokens))
    ]
    minimum = min(records, key=lambda item: (item["margin"], item["position"]))
    actions = _macro_actions(len(tokens))
    per_action = [min(records[position]["margin"] for position in action) for action in actions]
    return {
        "global_min": float(minimum["margin"]), "weakest_token": minimum,
        "positive_count": sum(item["margin"] > 0.0 for item in records),
        "per_action_min": per_action,
        "weakest_action": min(range(47), key=lambda index: (per_action[index], index)),
        "per_token": records,
    }


def _exact_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    return next((index for index, (a, b) in enumerate(zip(left, right)) if int(a) != int(b)), min(len(left), len(right)))


def _natural_evaluation(
    *, tokenizer: Any, tokens: Sequence[int], target: Mapping[str, Any],
    route_tokens: Sequence[int], raw_example: Any, parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    if len(route_tokens) != 415:
        raise Certified46Hold("HOLD: natural evaluation lacks one compiled witness route")
    ledger = _parse_and_match(tokenizer=tokenizer, generated_token_ids=tokens, label=label, raw_example=raw_example)
    parse, matcher = ledger["parse"], ledger["matcher"]
    receipts = list(matcher.get("prediction_receipts", ()))
    owners = {str(row.get("strict_match_gt_owner_id")) for row in receipts if row.get("strict_match_status") == "matched"}
    gate = full_root._full_root_joint_owner_equivalence_from_ledger(tokens=tokens, ledger=ledger, target=target)
    hard = sum(int(value) for value in gate["hard_raw_counters"].values())
    hard += int(not gate["natural_row_aligned_eos"])
    return {
        "generated_token_ids": list(map(int, tokens)), "generated_token_ids_sha256": token_ids_sha256(tokens),
        "exact_target_prefix": _exact_prefix(tokens, route_tokens),
        "matched_target_owner_count": len(owners), "matched_target_owner_ids": sorted(owners),
        "all_parent_owners_retained": set(parent_owners).issubset(owners), "hard_counter_count": hard,
        "joint_gate": gate, "ledger": ledger,
    }


def _winner(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Lexicographic winner; deliberately excludes mean CE."""
    if set(candidates) != {"C", "S"}:
        raise Certified46Hold("HOLD: winner comparison requires C and S")
    def key(name: str) -> tuple[Any, ...]:
        item = candidates[name]
        return (
            bool(item["all_parent_owners_retained"]),
            float(item["margin_improvement"]), int(item["exact_target_prefix"]),
            int(item["matched_target_owner_count"]), -int(item["hard_counter_count"]), name == "S",
        )
    return max(("C", "S"), key=key)


def _group_gradients(*, model: Any, native_inputs: Mapping[str, Any], route: Mapping[str, Any], parameters: Sequence[torch.nn.Parameter], pad: int, worker_index: int, worker_count: int, group: Any, update: bool, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    logits = full_root._teacher_forced_route_logits(model=model, native_inputs=native_inputs, route_tokens=route["tokens"], pad_token_id=pad)
    loss, allocation = _macro_loss(logits, route["tokens"], worker_index=worker_index, worker_count=worker_count)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise Certified46Hold("HOLD: disconnected/non-finite Phase-1 gradient")
        parameter.grad = gradient.detach()
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
    norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
    if not bool(torch.isfinite(norm).item()) or float(norm) <= 0:
        raise Certified46Hold("HOLD: macro gradient is zero/non-finite")
    if update:
        optimizer.step()
    return {"loss_contribution": float(loss.detach()), "allocation": allocation, "gradient_norm_pre_clip": float(norm), "updated": update}


def _frontier_loss(
    logits: torch.Tensor, teacher_tokens: Sequence[int], *, span_start: int, span_end: int,
    worker_index: int, worker_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if logits.ndim != 2 or logits.shape[0] != len(teacher_tokens):
        raise Certified46Hold("HOLD: frontier teacher logits do not align")
    if not (0 <= span_start <= span_end < len(teacher_tokens)):
        raise Certified46Hold("HOLD: frontier supervision span is out of bounds")
    positions = _frontier_span_partition(
        start=span_start, end=span_end, worker_index=worker_index, worker_count=worker_count,
    )
    targets = torch.tensor(teacher_tokens, dtype=torch.long, device=logits.device)
    terms = (
        F.cross_entropy(logits[list(positions)], targets[list(positions)], reduction="sum")
        if positions else logits.sum() * 0.0
    )
    length = span_end - span_start + 1
    return terms / length, {
        "positions": list(positions), "span_start": span_start, "span_end": span_end,
        "span_length": length, "normalization": "SUM-allreduce / total-span-length",
    }


def _frontier_gradients(
    *, model: Any, native_inputs: Mapping[str, Any], plan: Mapping[str, Any], parameters: Sequence[torch.nn.Parameter],
    pad: int, worker_index: int, group: Any, optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    teacher = list(map(int, plan["teacher_tokens"]))
    logits = full_root._teacher_forced_route_logits(
        model=model, native_inputs=native_inputs, route_tokens=teacher, pad_token_id=pad,
    )
    loss, allocation = _frontier_loss(
        logits, teacher, span_start=int(plan["span_start"]), span_end=int(plan["span_end"]),
        worker_index=worker_index, worker_count=WORLD_SIZE,
    )
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise Certified46Hold("HOLD: disconnected/non-finite frontier gradient")
        parameter.grad = gradient.detach()
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
    norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
    if not bool(torch.isfinite(norm).item()) or float(norm) <= 0:
        raise Certified46Hold("HOLD: frontier gradient is zero/non-finite")
    optimizer.step()
    return {"loss_contribution": float(loss.detach()), "allocation": allocation, "gradient_norm_pre_clip": float(norm)}


def _frontier_candidate_scores(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int, target: Mapping[str, Any],
    natural_tokens: Sequence[int], causal_ledger: Mapping[str, Any], rejected_offset: int,
) -> dict[str, Any]:
    if not causal_ledger.get("segmentation_trusted"):
        raise Certified46Hold("HOLD: frontier target scoring requires trusted segmentation")
    event = causal_ledger.get("first_event")
    if not isinstance(event, Mapping):
        raise Certified46Hold("HOLD: frontier target scoring requires a causal event")
    if event.get("kind") in {"untrusted_segmentation", "nontermination"}:
        raise Certified46Hold("HOLD: frontier event has no trusted finite bad span")
    rows, _by_owner = _target_rows(target)
    locked = set(map(str, causal_ledger.get("locked_owner_ids", ())))
    pool = [row for row in rows if str(row["owner"]) not in locked]
    if not pool:
        raise Certified46Hold("HOLD: frontier target pool is empty")
    event_start = int(event["event_start"])
    natural = list(map(int, natural_tokens))
    bad = list(map(int, event.get("bad_span_tokens", ())))
    if event_start < 0 or event_start > len(natural) or not bad:
        raise Certified46Hold("HOLD: frontier event span is malformed")
    prefix = natural[:event_start]
    scores: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            owner = str(row["owner"])
            if owner in locked:
                continue
            q = list(map(int, row["token_ids"]))
            divergence, span_start, span_end = _frontier_supervision_span(
                event_start=event_start, bad_span_tokens=bad, target_row_tokens=q,
            )
            teacher = [*prefix, *q]
            logits = full_root._teacher_forced_route_logits(
                model=model, native_inputs=native_inputs, route_tokens=teacher, pad_token_id=pad,
            )
            if not bool(torch.isfinite(logits).all().item()):
                raise Certified46Hold("HOLD: non-finite frontier target logits")
            margins: list[float] = []
            for position in range(span_start, span_end + 1):
                target_token = int(teacher[position])
                values = logits[position]
                target_logit = values[target_token]
                nontarget = values.clone()
                nontarget[target_token] = -torch.inf
                margins.append(float((target_logit - nontarget.max()).item()))
            bottleneck = min(range(len(margins)), key=lambda index: (margins[index], index))
            scores.append({
                "owner": owner, "target_row_index": row_index,
                "compiled_suffix_rank": sum(1 for prior in rows[:row_index] if str(prior["owner"]) not in locked),
                "divergence": divergence, "minimum_margin": margins[bottleneck],
                "bottleneck_global_position": event_start + divergence + bottleneck,
                "per_token_margins": margins,
            })
    scores.sort(key=lambda item: (-float(item["minimum_margin"]), int(item["compiled_suffix_rank"]), int(item["target_row_index"]), str(item["owner"])))
    near_miss = event.get("near_miss_owner") if event.get("kind") == "near_miss" else None
    if near_miss in {item["owner"] for item in scores}:
        selected = next(item for item in scores if item["owner"] == near_miss)
    else:
        selected = scores[rejected_offset % len(scores)]
    q = list(map(int, rows[int(selected["target_row_index"])]["token_ids"]))
    _divergence, span_start, span_end = _frontier_supervision_span(
        event_start=event_start, bad_span_tokens=bad, target_row_tokens=q,
    )
    return {
        "event": dict(event), "event_signature": _hash({"kind": event.get("kind"), "row": event.get("row_index"), "bad": bad}),
        "target_owner": str(selected["owner"]), "target_row_index": int(selected["target_row_index"]),
        "rejected_branch_offset": rejected_offset, "scores": scores,
        "teacher_tokens": [*prefix, *q], "teacher_tokens_sha256": token_ids_sha256([*prefix, *q]),
        "span_start": span_start, "span_end": span_end, "span_length": ROW_TOKENS - int(selected["divergence"]),
        "bad_span_tokens": bad, "target_row_tokens": q,
    }


def _natural_tokens(*, model: Any, native_inputs: Mapping[str, Any], pad: int) -> list[int]:
    inputs = lattice.build_event_inputs(native_inputs, (), pad_token_id=pad)
    prompt_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=NATURAL_MAX_TOKENS, repetition_penalty=1.0, eos_token_id=EOS, pad_token_id=pad, do_sample=False, return_dict_in_generate=True, output_scores=False)
    sequences = getattr(output, "sequences", None)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise Certified46Hold("HOLD: natural greedy generate lacks sequences")
    return [int(value) for value in sequences[0, prompt_width:].detach().cpu().tolist()]


def _surface_agreement(surface: Mapping[str, Any], group: Any) -> None:
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, surface["aggregate_sha256"], group=group)
    if len(set(gathered)) != 1:
        raise Certified46Hold("HOLD: rank disagreement after group update")


def _all_parameter_surface(model: Any) -> str:
    """No-update has no permitted mutation, so hash every named parameter."""
    return _hash([
        {"name": name, "sha256": full_root._tensor_sha256(parameter), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
    ])


def _frozen_surface(model: Any, trainable_names: Sequence[str]) -> str:
    trainable = set(trainable_names)
    return _hash([
        {"name": name, "sha256": full_root._tensor_sha256(parameter), "numel": parameter.numel()}
        for name, parameter in model.named_parameters() if name not in trainable
    ])


def _evaluation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "generated_token_ids", "generated_token_ids_sha256", "exact_target_prefix",
            "matched_target_owner_count", "matched_target_owner_ids", "all_parent_owners_retained",
            "hard_counter_count", "joint_gate",
        )
    }


def _causal_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "segmentation_trusted", "locked_owner_ids", "final_missing_owner_ids",
            "first_event", "causal_hard_counter_count", "token_ids_sha256",
        )
    }


def _setup_for_checkpoint(checkpoint: Path) -> dict[str, Any]:
    for relative, expected in SOURCE_GATE_FILES.items():
        path = SOURCE_GATE_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise Certified46Hold(f"HOLD: embedding source-gate authority drifted: {relative}")
    setup = lattice._setup_for_checkpoint(checkpoint)
    frontend = setup["frontend"]
    embedding_delta = dict(frontend.launch.embedding_delta or {})
    if not embedding_delta.get("path"):
        raise Certified46Hold("HOLD: checkpoint lacks an embedding delta")
    launch = replace(
        frontend.launch,
        embedding_delta={**embedding_delta, "source_gate_root": str(SOURCE_GATE_ROOT)},
    )
    return {**setup, "frontend": replace(frontend, launch=launch)}


def _bindings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _sha256(TARGET_PATH) != TARGET_SHA256:
        raise Certified46Hold("HOLD: target SHA drifted")
    # The historical panel was intentionally deleted.  Local process-only substitution
    # keeps existing runtime setup on its canonical matcher while binding its source.
    _load_authority()
    microscope.REFINED_PANEL = AUTHORITY_PATH
    admission = lattice.admit_parent("A")
    if Path(str(admission.get("checkpoint", ""))) != PARENT_CHECKPOINT or admission.get("expected_route_sha256") != PARENT_ROUTE_SHA256:
        raise Certified46Hold("HOLD: Parent A binding drifted")
    setup = _setup_for_checkpoint(PARENT_CHECKPOINT)
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    if (
        setup["target"].get("token_ids_sha256") != target.get("token_ids_sha256")
        or setup["plan"].get("prompt_token_ids_sha256") != PROMPT_TOKEN_SHA256
        or full_root._sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256
    ):
        raise Certified46Hold("HOLD: live Parent-A target/prompt binding drifted")
    return admission, setup, target


def _cold_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], route_tokens: Sequence[int],
    parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    setup = _setup_for_checkpoint(checkpoint)
    with open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise Certified46Hold("HOLD: cold evaluation requires concrete FP32 HF backend")
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
            raise Certified46Hold("HOLD: cold prompt identity drifted")
        tokens = _natural_tokens(model=opened._model, native_inputs=native_inputs, pad=int(opened._tokenizer.pad_token_id))
        evaluation = _natural_evaluation(
            tokenizer=opened._tokenizer, tokens=tokens, target=target,
            route_tokens=route_tokens, raw_example=setup["raw_example"],
            parent_owners=parent_owners, label=label,
        )
        return {**evaluation, "runtime": opened.receipt.to_artifact_dict()}


def _cold_frontier_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], route: Mapping[str, Any], parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    """Cold evaluation keeps the local causal ledger and global evaluator separate."""
    setup = _setup_for_checkpoint(checkpoint)
    with open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise Certified46Hold("HOLD: cold frontier evaluation requires concrete FP32 HF backend")
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
            raise Certified46Hold("HOLD: cold frontier prompt identity drifted")
        tokens = _natural_tokens(model=opened._model, native_inputs=native_inputs, pad=int(opened._tokenizer.pad_token_id))
        causal = _causal_locked_ledger(tokenizer=opened._tokenizer, tokens=tokens, raw_example=setup["raw_example"])
        return {
            "checkpoint": str(checkpoint), "generated_token_ids": tokens,
            "causal_ledger": causal,
            "natural_evaluation": _natural_evaluation(
                tokenizer=opened._tokenizer, tokens=tokens, target=target, route_tokens=route["tokens"],
                raw_example=setup["raw_example"], parent_owners=parent_owners, label=label,
            ),
            "runtime": opened.receipt.to_artifact_dict(),
        }


def _rank0_call(action: Any) -> Any:
    """All rank-zero failures cross the same broadcast before a peer can proceed."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {"ok": True, "value": action()}
        except BaseException as error:
            packet[0] = {"ok": False, "type": type(error).__name__, "error": str(error)}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or not result.get("ok"):
        raise Certified46Hold(f"HOLD: rank-zero frontier failure: {dict(result or {}).get('type')}: {dict(result or {}).get('error')}")
    return result["value"]


def _phase2_observation(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int, target: Mapping[str, Any],
    raw_example: Any, parent_owners: Sequence[str], rejected_offset: int, route: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tokens = _natural_tokens(model=model, native_inputs=native_inputs, pad=pad)
    causal = _causal_locked_ledger(tokenizer=tokenizer, tokens=tokens, raw_example=raw_example)
    observation: dict[str, Any] = {
        "generated_token_ids": tokens, "generated_token_ids_sha256": token_ids_sha256(tokens),
        "causal_ledger": causal,
    }
    if route is not None:
        observation["natural_evaluation"] = _natural_evaluation(
            tokenizer=tokenizer, tokens=tokens, target=target, route_tokens=route["tokens"], raw_example=raw_example,
            parent_owners=parent_owners, label="phase2-warm-observation",
        )
        observation["owner_keyed_route_prefix"] = _owner_keyed_route_prefix(
            causal_ledger=causal, natural_tokens=tokens, route=route,
        )
    if causal.get("first_event") is not None:
        observation["frontier_plan"] = _frontier_candidate_scores(
            model=model, native_inputs=native_inputs, pad=pad, target=target, natural_tokens=tokens,
            causal_ledger=causal, rejected_offset=rejected_offset,
        )
    return observation


def _prepare_frontier_output(output: Path) -> None:
    if output.exists():
        raise Certified46Hold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)


def _run_frontier_stage(*, stage: str, run_id: str) -> Path:
    """Conditional Phase 2; every branch starts from a freshly loaded cold champion."""
    rank = dist.get_rank()
    output = OUTPUT_ROOT / stage / run_id
    try:
        _rank0_call(lambda: _prepare_frontier_output(output))
        dist.barrier()
        trigger = _rank0_call(_phase1_trigger)
        admission, setup, target = _bindings()
        parent_tokens = _parent_tokens()
        branch_records: list[dict[str, Any]] = []
        promotions: list[dict[str, Any]] = []
        stop_reason = "branch_budget_exhausted"
        status = "phase2_negative_bounded"
        champion_checkpoint = PARENT_CHECKPOINT
        champion_owners: set[str] = set()
        champion_prefix = 0
        same_event_count = 0
        prior_event_signature: str | None = None
        rejected_offset = 0
        parent_owners: tuple[str, ...] = ()

        for branch in range(1, PHASE2_BRANCHES + 1):
            branch_dir = output / f"branch-{branch:02d}"
            _rank0_call(lambda: branch_dir.mkdir())
            dist.barrier()
            with open_backend_session(setup["frontend"].launch if champion_checkpoint == PARENT_CHECKPOINT else _setup_for_checkpoint(champion_checkpoint)["frontend"].launch) as opened:
                if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
                    raise Certified46Hold("HOLD: Phase-2 requires concrete FP32 HF backend")
                model, tokenizer = opened._model, opened._tokenizer
                model.eval()
                native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
                if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
                    raise Certified46Hold("HOLD: Phase-2 live prompt identity drifted")
                if not parent_owners:
                    parent_ledger = _causal_locked_ledger(
                        tokenizer=tokenizer, tokens=parent_tokens, raw_example=setup["raw_example"],
                    )
                    _assert_parent_causal_ledger(parent_ledger)
                    parent_owners = tuple(map(str, parent_ledger["locked_owner_ids"]))

                baseline = _rank0_call(lambda: _phase2_observation(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target=target, raw_example=setup["raw_example"], parent_owners=parent_owners,
                    rejected_offset=rejected_offset, route=None,
                ))
                if champion_checkpoint == PARENT_CHECKPOINT:
                    _assert_parent_causal_ledger(baseline["causal_ledger"])
                plan = baseline.get("frontier_plan")
                if not isinstance(plan, Mapping):
                    raise Certified46Hold("HOLD: cold champion reached no frontier before Phase-2 success gate")
                dynamic_route = _rank0_call(lambda: _compile_dynamic_route(
                    target=target, natural_tokens=baseline["generated_token_ids"], causal_ledger=baseline["causal_ledger"],
                    preferred_owner=str(plan["target_owner"]),
                ))
                admission_record = _rank0_call(lambda: _admit_route(
                    tokenizer=tokenizer, route=dynamic_route, target=target, raw_example=setup["raw_example"],
                    label=f"phase2-branch-{branch:02d}-dynamic",
                ))
                baseline = _rank0_call(lambda: {
                    **baseline,
                    "natural_evaluation": _natural_evaluation(
                        tokenizer=tokenizer, tokens=baseline["generated_token_ids"], target=target,
                        route_tokens=dynamic_route["tokens"], raw_example=setup["raw_example"],
                        parent_owners=parent_owners, label=f"phase2-branch-{branch:02d}-baseline",
                    ),
                    "owner_keyed_route_prefix": _owner_keyed_route_prefix(
                        causal_ledger=baseline["causal_ledger"], natural_tokens=baseline["generated_token_ids"], route=dynamic_route,
                    ),
                })
                baseline_owners = set(map(str, baseline["causal_ledger"]["locked_owner_ids"]))
                if champion_owners and baseline_owners != champion_owners:
                    raise Certified46Hold("HOLD: cold champion owner identity drifted between branches")
                if not champion_owners:
                    champion_owners = baseline_owners
                baseline_prefix = int(baseline["owner_keyed_route_prefix"])
                champion_prefix = baseline_prefix
                event_signature = str(plan["event_signature"])
                same_event_count = same_event_count + 1 if event_signature == prior_event_signature else 1
                prior_event_signature = event_signature
                selected_margin = float(next(item for item in plan["scores"] if item["owner"] == plan["target_owner"])["minimum_margin"])
                historical_best, updates_without_best = selected_margin, 0

                names, parameters = full_root._trainable_surface(model, dora_all_only=True)
                if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                    raise Certified46Hold("HOLD: exact 588/18006016 FP32 DoRA surface drifted")
                sentinels = full_root._full_root_nontrainable_sentinels(model)
                _originals, initial_surface = full_root._full_root_surface_snapshot(names, parameters)
                frozen_before = _frozen_surface(model, names)
                optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=0.0, weight_decay=0.0)
                update_records: list[dict[str, Any]] = []
                current_plan: Mapping[str, Any] = plan
                finish_branch = False
                for step in range(1, PHASE2_STEPS + 1):
                    if step % 2:
                        record = _group_gradients(
                            model=model, native_inputs=native_inputs, route=dynamic_route, parameters=parameters,
                            pad=int(tokenizer.pad_token_id), worker_index=rank, worker_count=WORLD_SIZE,
                            group=dist.group.WORLD, update=True, optimizer=optimizer,
                        )
                        record["kind"] = "macro"
                    else:
                        record = _frontier_gradients(
                            model=model, native_inputs=native_inputs, plan=current_plan, parameters=parameters,
                            pad=int(tokenizer.pad_token_id), worker_index=rank, group=dist.group.WORLD, optimizer=optimizer,
                        )
                        record["kind"] = "frontier"
                        record["frontier_target_owner"] = current_plan["target_owner"]
                    _surface_agreement(full_root._full_root_surface_snapshot(names, parameters)[1], dist.group.WORLD)
                    full_root._assert_full_root_sentinels(model, sentinels)
                    if step % 2 == 0:
                        refreshed = _rank0_call(lambda: _phase2_observation(
                            model=model, tokenizer=tokenizer, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                            target=target, raw_example=setup["raw_example"], parent_owners=parent_owners,
                            rejected_offset=rejected_offset, route=dynamic_route,
                        ))
                        record["observation"] = refreshed
                        next_plan = refreshed.get("frontier_plan")
                        if isinstance(next_plan, Mapping):
                            current_plan = next_plan
                            next_margin = float(next(item for item in next_plan["scores"] if item["owner"] == next_plan["target_owner"])["minimum_margin"])
                            if historical_best is None or next_margin > historical_best:
                                historical_best, updates_without_best = next_margin, 0
                            else:
                                updates_without_best += 2
                        else:
                            finish_branch = True
                    update_records.append({"step": step, **record})
                    if updates_without_best >= PHASE2_STEPS:
                        finish_branch = True
                        stop_reason = "eight_updates_without_new_weakest_margin_best"
                    if finish_branch:
                        break

                terminal_surface = full_root._full_root_surface_snapshot(names, parameters)[1]
                if _frozen_surface(model, names) != frozen_before:
                    raise Certified46Hold("HOLD: Phase-2 frozen non-DoRA surface mutated")
                if terminal_surface == initial_surface:
                    raise Certified46Hold("HOLD: Phase-2 branch did not change permitted DoRA surface")
                full_route_margins = _rank0_call(lambda: _strict_margins(
                    full_root._teacher_forced_route_logits(
                        model=model, native_inputs=native_inputs, route_tokens=dynamic_route["tokens"], pad_token_id=int(tokenizer.pad_token_id),
                    ), dynamic_route["tokens"],
                ))
                saved = _rank0_call(lambda: {
                    "checkpoint": str(branch_dir / "checkpoint-candidate"),
                    "readback": full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=champion_checkpoint, destination=branch_dir / "checkpoint-candidate",
                    ),
                })
                local_record = {
                    "branch": branch, "rank": rank, "initial_surface": initial_surface, "terminal_surface": terminal_surface,
                    "frozen_surface_before": frozen_before, "frozen_surface_after": _frozen_surface(model, names),
                    "updates": update_records,
                }
                gathered_records: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered_records, local_record)
                warm_final = _rank0_call(lambda: _phase2_observation(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target=target, raw_example=setup["raw_example"], parent_owners=parent_owners,
                    rejected_offset=rejected_offset, route=dynamic_route,
                ))
                runtime = opened.receipt.to_artifact_dict()
            del model, tokenizer, native_inputs, prompts, opened
            torch.cuda.empty_cache()
            dist.barrier()
            cold = _rank0_call(lambda: _cold_frontier_evaluate(
                Path(saved["checkpoint"]), target=target, route=dynamic_route, parent_owners=parent_owners,
                label=f"phase2-branch-{branch:02d}-cold",
            ))
            candidate_owners = set(map(str, cold["causal_ledger"]["locked_owner_ids"]))
            candidate_prefix = _owner_keyed_route_prefix(
                causal_ledger=cold["causal_ledger"], natural_tokens=cold["generated_token_ids"], route=dynamic_route,
            )
            global_hard = int(cold["natural_evaluation"]["hard_counter_count"])
            causal_hard = int(cold["causal_ledger"]["causal_hard_counter_count"])
            positive_certificate = float(full_route_margins["global_min"]) > 0.0
            branch_record = {
                "branch": branch, "champion_checkpoint": str(champion_checkpoint), "baseline": baseline,
                "dynamic_route": dynamic_route["manifest"], "admission": admission_record["joint_gate"],
                "updates": next(record["updates"] for record in gathered_records if record["rank"] == 0),
                "rank_surface_records": [{
                    key: record[key] for key in (
                        "rank", "initial_surface", "terminal_surface", "frozen_surface_before", "frozen_surface_after",
                    )
                } for record in gathered_records], "candidate_checkpoint": saved,
                "full_route_margins": full_route_margins, "warm_final": warm_final, "cold": cold,
                "runtime": runtime, "candidate_owner_keyed_prefix": candidate_prefix,
            }
            if (
                warm_final["generated_token_ids"] != cold["generated_token_ids"]
                or _evaluation_identity(warm_final["natural_evaluation"]) != _evaluation_identity(cold["natural_evaluation"])
                or _causal_identity(warm_final["causal_ledger"]) != _causal_identity(cold["causal_ledger"])
            ):
                status, stop_reason = "technical_hold_warm_cold_mismatch", "saved_candidate_not_cold_reproducible"
                branch_record["promotion"] = "technical_hold"
                branch_records.append(branch_record)
                break
            if positive_certificate and cold["generated_token_ids"] != dynamic_route["tokens"]:
                status, stop_reason = "technical_hold_path_certificate_mismatch", "positive_all_token_certificate_not_cold_reproduced"
                branch_record["promotion"] = "technical_hold"
                branch_records.append(branch_record)
                break
            if (
                candidate_owners.issuperset(champion_owners)
                and causal_hard == 0 and global_hard == 0
                and (len(candidate_owners) > len(champion_owners) or (len(candidate_owners) == len(champion_owners) and candidate_prefix > baseline_prefix))
            ):
                promotion = {"branch": branch, "from": str(champion_checkpoint), "to": saved["checkpoint"], "owner_count": len(candidate_owners), "owner_keyed_prefix": candidate_prefix}
                promotions.append(promotion)
                branch_record["promotion"] = promotion
                champion_checkpoint = Path(saved["checkpoint"])
                champion_owners, champion_prefix, rejected_offset = candidate_owners, candidate_prefix, 0
                prior_event_signature, same_event_count = None, 0
                if stop_reason == "eight_updates_without_new_weakest_margin_best":
                    stop_reason = "branch_budget_exhausted"
                if len(candidate_owners) == 46 and bool(cold["natural_evaluation"]["joint_gate"]["passed"]):
                    status, stop_reason = "cold_natural_46_46", "first_cold_exact_joint_46_46"
                    branch_records.append(branch_record)
                    break
            else:
                branch_record["promotion"] = "rejected"
                rejected_offset += 1
            branch_records.append(branch_record)
            if branch_record["promotion"] == "rejected" and same_event_count >= 3:
                stop_reason = "same_first_causal_event_three_independent_branches"
                break
            if stop_reason == "eight_updates_without_new_weakest_margin_best":
                break

        if rank == 0:
            receipt = {
                "schema_version": PHASE2_SCHEMA_VERSION, "status": status, "stage": stage, "run_id": run_id, "unit_id": UNIT_ID,
                "runner_sha256": _sha256(Path(__file__)), "phase1_trigger": trigger,
                "bindings": {"parent_checkpoint": str(PARENT_CHECKPOINT), "parent_route_sha256": PARENT_ROUTE_SHA256, "target": str(TARGET_PATH), "target_sha256": TARGET_SHA256, "authority": str(AUTHORITY_PATH), "authority_sha256": AUTHORITY_SHA256, "source_gate_files": SOURCE_GATE_FILES},
                "allowed_surface": "588 FP32 DoRA tensors / 18006016 elements", "frozen_surfaces": "embeddings, aligner, vision tower, all non-DoRA parameters",
                "optimizer": {"name": "SGD", "learning_rate": LEARNING_RATE, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": GRAD_CLIP},
                "protocol": {"world_size": WORLD_SIZE, "max_branches": PHASE2_BRANCHES, "max_steps_per_branch": PHASE2_STEPS, "updates": "macro odd / causal-frontier even", "frontier_loss": "partitioned token CE, SUM allreduce, normalize total span length", "unlikelihood": False},
                "parent_causal_owner_ids": list(parent_owners), "branches": branch_records, "promotions": promotions,
                "final_champion_checkpoint": str(champion_checkpoint), "final_champion_owner_ids": sorted(champion_owners),
                "final_champion_owner_keyed_prefix": champion_prefix, "stop_reason": stop_reason,
            }
        _rank0_call(lambda: _atomic_json(output / "receipt.json", receipt))
        dist.barrier()
        return output
    except BaseException as error:
        if rank == 0 and output.exists():
            _atomic_json(output / "failure.json", {
                "schema_version": PHASE2_SCHEMA_VERSION + ".failure", "status": "mechanical_failure", "stage": stage,
                "run_id": run_id, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            })
        raise


def run_stage(*, stage: str, run_id: str) -> Path:
    if stage not in {"no-update", "one-update", "foreground", "frontier", "phase2"}:
        raise Certified46Hold("HOLD: unknown stage")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise Certified46Hold("HOLD: requires torchrun --nproc_per_node=8")
    run_id = _safe_run_id(run_id)
    dist.init_process_group(backend="nccl")
    rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise Certified46Hold("HOLD: LOCAL_RANK absent")
    torch.cuda.set_device(local_rank)
    if stage in {"frontier", "phase2"}:
        try:
            return _run_frontier_stage(stage=stage, run_id=run_id)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    output = OUTPUT_ROOT / stage / run_id
    try:
        if rank == 0:
            if output.exists():
                raise Certified46Hold(f"HOLD: refusing overwrite: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        admission, setup, target = _bindings()
        parent_tokens = _parent_tokens()
        groups = {name: dist.new_group(ranks=ranks) for name, ranks in GROUPS.items()}
        route_name = "C" if rank < 4 else "S"
        worker_index = rank % 4
        parent_owners: tuple[str, ...] = ()
        diagnostics: dict[str, list[dict[str, Any]]] = {"C": [], "S": [], "winner": []}
        updates: list[dict[str, Any]] = []
        one_step_checkpoints: dict[str, dict[str, Any]] = {}
        rank_group_records: list[dict[str, Any]] = []
        runtime: dict[str, Any] | None = None
        final_warm: dict[str, Any] | None = None
        checkpoint: Path | None = None
        checkpoint_readback: dict[str, Any] | None = None
        winner: str | None = None
        winner_route: Mapping[str, Any] | None = None
        with open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise Certified46Hold("HOLD: requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
                raise Certified46Hold("HOLD: live prompt identity drifted")
            parent_ledger = _parse_and_match(tokenizer=tokenizer, generated_token_ids=parent_tokens, label="parent-route", raw_example=setup["raw_example"])
            parent_owners = _strict_owner_order(parent_ledger)
            _assert_missing_membership(parent_owners)
            routes = _compile_routes(target=target, parent_tokens=parent_tokens, parent_ledger=parent_ledger)
            admissions = {name: _admit_route(tokenizer=tokenizer, route=route, target=target, raw_example=setup["raw_example"], label=f"route-{name}") for name, route in routes.items()}
            names, parameters = full_root._trainable_surface(model, dora_all_only=True)
            if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                raise Certified46Hold("HOLD: exact 588/18006016 FP32 DoRA surface drifted")
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            originals, initial_surface = full_root._full_root_surface_snapshot(names, parameters)
            all_parameters_before = _all_parameter_surface(model) if stage == "no-update" else None
            frozen_before = _frozen_surface(model, names)
            optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=0.0, weight_decay=0.0)

            def observe(step: int, *, natural: bool) -> dict[str, Any] | None:
                if worker_index != 0:
                    return None
                logits = full_root._teacher_forced_route_logits(model=model, native_inputs=native_inputs, route_tokens=routes[route_name]["tokens"], pad_token_id=int(tokenizer.pad_token_id))
                result: dict[str, Any] = {"step": step, "teacher": _strict_margins(logits, routes[route_name]["tokens"])}
                if natural:
                    tokens = _natural_tokens(model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id))
                    result["natural"] = _natural_evaluation(
                        tokenizer=tokenizer, tokens=tokens, target=target,
                        route_tokens=routes[route_name]["tokens"],
                        raw_example=setup["raw_example"], parent_owners=parent_owners,
                        label=f"{route_name}-step{step}",
                    )
                return result

            if stage == "foreground":
                initial = observe(0, natural=True)
                diagnostics[route_name].append(initial) if initial is not None else None
            loops = 1 if stage in {"no-update", "one-update"} else 8
            for step in range(1, loops + 1):
                record = _group_gradients(model=model, native_inputs=native_inputs, route=routes[route_name], parameters=parameters, pad=int(tokenizer.pad_token_id), worker_index=worker_index, worker_count=4, group=groups[route_name], update=stage != "no-update", optimizer=optimizer)
                _surface_agreement(full_root._full_root_surface_snapshot(names, parameters)[1], groups[route_name])
                full_root._assert_full_root_sentinels(model, sentinels)
                updates.append({"step": step, "route": route_name, **record})
                if stage == "no-update":
                    break
                if stage == "one-update" or (stage == "foreground" and step in {2, 4, 6, 8}):
                    item = observe(step, natural=stage == "one-update" or step in {4, 8})
                    diagnostics[route_name].append(item) if item is not None else None
            terminal_surface = full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = _frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise Certified46Hold("HOLD: frozen non-DoRA surface mutated")
            if stage == "no-update":
                if terminal_surface != initial_surface:
                    raise Certified46Hold("HOLD: no-update mutated trainable surface")
                if _all_parameter_surface(model) != all_parameters_before:
                    raise Certified46Hold("HOLD: no-update mutated a parameter")
                full_root._restore_parameters(parameters, originals)
            else:
                if terminal_surface == initial_surface:
                    raise Certified46Hold("HOLD: update did not change permitted DoRA surface")
            if stage == "one-update" and worker_index == 0:
                destination = output / f"checkpoint-route-{route_name}-step-01"
                one_step_checkpoints[route_name] = {
                    "checkpoint": str(destination),
                    "readback": full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=PARENT_CHECKPOINT, destination=destination
                    ),
                }
            rank_group_records.append(
                {
                    "rank": rank, "route": route_name, "updates": updates,
                "initial_surface": initial_surface, "terminal_surface": terminal_surface,
                "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
                }
            )
            if stage == "foreground":
                local_candidate = None
                if worker_index == 0:
                    terminal = next(item for item in diagnostics[route_name] if item is not None and item["step"] == 8)
                    initial = next(item for item in diagnostics[route_name] if item is not None and item["step"] == 0)
                    local_candidate = dict(terminal["natural"]) | {
                        "route": route_name,
                        "margin_improvement": float(
                            terminal["teacher"]["global_min"] - initial["teacher"]["global_min"]
                        ),
                    }
                gathered: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered, local_candidate)
                if rank == 0:
                    candidates = {str(item["route"]): item for item in gathered if item is not None}
                    chosen = _winner(candidates)
                else:
                    chosen = None
                elected = [chosen]
                dist.broadcast_object_list(elected, src=0)
                winner = str(elected[0])
                source = 0 if winner == "C" else 4
                for parameter in parameters:
                    dist.broadcast(parameter.data, src=source)
                _surface_agreement(full_root._full_root_surface_snapshot(names, parameters)[1], dist.group.WORLD)
                winner_route = routes[winner]
                optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=0.0, weight_decay=0.0)
                for step in range(9, 33):
                    record = _group_gradients(model=model, native_inputs=native_inputs, route=winner_route, parameters=parameters, pad=int(tokenizer.pad_token_id), worker_index=rank, worker_count=WORLD_SIZE, group=dist.group.WORLD, update=True, optimizer=optimizer)
                    updates.append({"step": step, "route": winner, **record})
                    full_root._assert_full_root_sentinels(model, sentinels)
                    stop_now = False
                    certificate_mismatch = False
                    if step % 2 == 0 and rank == 0:
                        logits = full_root._teacher_forced_route_logits(model=model, native_inputs=native_inputs, route_tokens=winner_route["tokens"], pad_token_id=int(tokenizer.pad_token_id))
                        item: dict[str, Any] = {"step": step, "teacher": _strict_margins(logits, winner_route["tokens"]), "update": record}
                        if step % 4 == 0 or item["teacher"]["global_min"] > 0.0:
                            tokens = _natural_tokens(model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id))
                            item["natural"] = _natural_evaluation(
                                tokenizer=tokenizer, tokens=tokens, target=target,
                                route_tokens=winner_route["tokens"],
                                raw_example=setup["raw_example"],
                                parent_owners=parent_owners, label=f"winner-step{step}",
                            )
                        diagnostics["winner"].append(item)
                        if item["teacher"]["global_min"] > 0.0:
                            certificate_mismatch = (
                                item["natural"]["generated_token_ids"] != winner_route["tokens"]
                            )
                            stop_now = not certificate_mismatch
                    stop_signal = [{"stop": stop_now, "mismatch": certificate_mismatch}]
                    dist.broadcast_object_list(stop_signal, src=0)
                    if bool(stop_signal[0]["mismatch"]):
                        raise Certified46Hold(
                            "HOLD: warm positive path certificate did not reproduce its witness"
                        )
                    if bool(stop_signal[0]["stop"]):
                        break
                if rank == 0:
                    latest = diagnostics["winner"][-1]
                    tokens = latest.get("natural", {}).get("generated_token_ids")
                    if tokens is None:
                        tokens = _natural_tokens(model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id))
                    final_warm = _natural_evaluation(
                        tokenizer=tokenizer, tokens=tokens, target=target,
                        route_tokens=winner_route["tokens"], raw_example=setup["raw_example"],
                        parent_owners=parent_owners, label="winner-final-warm",
                    )
                    checkpoint = output / f"checkpoint-winner-step-{step:02d}"
                    checkpoint_readback = full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=PARENT_CHECKPOINT, destination=checkpoint
                    )
            runtime = opened.receipt.to_artifact_dict()
            all_parameters_after = _all_parameter_surface(model) if stage == "no-update" else None
        # Do not leave a warm model reachable while the cold roots open fresh sessions.
        del model, tokenizer, native_inputs, prompts, opened
        torch.cuda.empty_cache()
        dist.barrier()
        cold: dict[str, Any] | None = None
        one_step_cold: dict[str, Any] | None = None
        if stage == "one-update" and worker_index == 0:
            saved = one_step_checkpoints[route_name]
            one_step_cold = {
                "route": route_name,
                "checkpoint": saved["checkpoint"],
                "readback": saved["readback"],
                "cold": _cold_evaluate(
                    Path(saved["checkpoint"]), target=target,
                    route_tokens=routes[route_name]["tokens"],
                    parent_owners=parent_owners, label=f"route-{route_name}-one-step-cold",
                ),
            }
        gathered_diagnostics: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_diagnostics, diagnostics)
        gathered_group_records: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_group_records, rank_group_records)
        gathered_one_step_cold: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_one_step_cold, one_step_cold)
        if rank == 0:
            for local in gathered_diagnostics:
                for name in ("C", "S"):
                    if local[name]:
                        diagnostics[name] = local[name]
            if stage == "one-update":
                by_route = {str(item["route"]): item for item in gathered_one_step_cold if item is not None}
                if set(by_route) != {"C", "S"}:
                    raise Certified46Hold("HOLD: missing one-step cold route receipt")
                for name, record in by_route.items():
                    warm = next(item["natural"] for item in diagnostics[name] if item.get("natural") is not None)
                    cold_eval = record["cold"]
                    if _evaluation_identity(warm) != _evaluation_identity(cold_eval):
                        raise Certified46Hold(f"HOLD: Route {name} one-step warm/cold evaluator mismatch")
                one_step_checkpoints = by_route
        status = "phase1_mechanics_complete"
        path_certificate_strict = False
        if stage == "foreground" and rank == 0:
            assert checkpoint is not None and checkpoint_readback is not None
            assert final_warm is not None and winner is not None and winner_route is not None
            cold = _cold_evaluate(
                checkpoint, target=target, route_tokens=winner_route["tokens"],
                parent_owners=parent_owners, label="winner-cold",
            )
            path_certificate_strict = bool(
                diagnostics["winner"][-1]["teacher"]["global_min"] > 0.0
            )
            if _evaluation_identity(cold) != _evaluation_identity(final_warm):
                status = "technical_hold_cold_greedy_mismatch"
            elif path_certificate_strict and cold["generated_token_ids"] != winner_route["tokens"]:
                status = "technical_hold_path_certificate_mismatch"
            elif cold["joint_gate"]["passed"]:
                status = "cold_natural_46_46"
            else:
                status = "phase1_complete_needs_frontier"
        cold_all: list[Any] = [cold]
        dist.broadcast_object_list(cold_all, src=0)
        if rank == 0:
            receipt = {
                "schema_version": SCHEMA_VERSION, "status": status, "unit_id": UNIT_ID, "stage": stage, "run_id": run_id,
                "runner_sha256": _sha256(Path(__file__)),
                "bindings": {"parent_checkpoint": str(PARENT_CHECKPOINT), "parent_route_sha256": PARENT_ROUTE_SHA256, "prompt_sha256": PROMPT_TOKEN_SHA256, "target": str(TARGET_PATH), "target_sha256": TARGET_SHA256, "authority": str(AUTHORITY_PATH), "authority_sha256": AUTHORITY_SHA256, "authority_image2299_line_sha256": AUTHORITY_LINE_SHA256, "embedding_source_gate_root": str(SOURCE_GATE_ROOT), "embedding_source_gate_files": SOURCE_GATE_FILES, "matcher_panel_substitution": {"from": "deleted historical panel", "to": str(AUTHORITY_PATH), "scope": "this process only"}},
                "missing_owner_manifest": [
                    {"owner": owner, "target_row_sha256": token_ids_sha256(next(row["token_ids"] for row in target["rows"] if row["owner"] == owner))}
                    for owner in MISSING_OWNER_ORDER
                ], "parent_owners": list(parent_owners),
                "routes": {name: {"manifest": route["manifest"], "admission": admissions[name]["joint_gate"]} for name, route in routes.items()},
                "protocol": {"surface": "588 FP32 DoRA tensors / 18006016 elements", "optimizer": "SGD", "learning_rate": LEARNING_RATE, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": GRAD_CLIP, "action_macro": "46 row-normalized CE + EOS CE, weight 1/47", "gradient_reduction": "round-robin action ownership and SUM all-reduce", "winner_rule": "retained+zero-debt, min-margin improvement, prefix, owner count, fewer counters, S tie-break"},
                "rank_group_records": gathered_group_records,
                "initial_surface": initial_surface, "terminal_surface": terminal_surface, "updates": updates, "diagnostics": diagnostics,
                "no_update_all_parameter_surface_before": all_parameters_before,
                "no_update_all_parameter_surface_after": all_parameters_after,
                "one_step_checkpoints": one_step_checkpoints,
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "checkpoint_readback": checkpoint_readback,
                "shootout_winner": winner,
                "shootout_winner_route_sha256": None if winner is None else ROUTE_SHA256[winner],
                "path_certificate_strict": path_certificate_strict,
                "warm_final": final_warm, "cold": cold_all[0], "runtime": runtime,
            }
            _atomic_json(output / "receipt.json", receipt)
        dist.barrier()
        return output
    except BaseException as error:
        if rank == 0 and output.exists():
            _atomic_json(output / "failure.json", {"schema_version": SCHEMA_VERSION + ".failure", "status": "mechanical_failure", "stage": stage, "run_id": run_id, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("no-update", "one-update", "foreground", "frontier", "phase2"), required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(run_stage(stage=args.stage, run_id=args.run_id))


if __name__ == "__main__":
    main()
