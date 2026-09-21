"""CPU reduction for the frozen recurrence-phase decision package.

The reducer owns no model execution.  It consumes the frozen plan plus capture
and release receipts, reparses saved output tokens, and keeps numerical
recurrence separate from annotation-relative matches.  Receipt readers accept
either one manifest or per-cell JSON/JSONL/PT files; the semantic checks below
are intentionally strict once a cell is found.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from probes.training_set_completion.numerical_feedback.select import order_stratum
from probes.training_set_completion.recurrence_phase_decision.common import read_plan
from probes.training_set_completion.recurrence_phase_decision.prepare import (
    BASE,
    write_new,
)


EOS = 151645
OBJ_START = 151646
OBJ_END = 151647
BOX_START = 151648
BOX_END = 151649
COORD_LIMIT = BASE + 1000
NEAR_EPS = 8
COORDINATE_ROLES = ("x1", "y1", "x2", "y2")
ROLE_INDEX = {name: index for index, name in enumerate(COORDINATE_ROLES)}
TOKENIZER_ROOT = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
_TOKENIZER: Any | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _same_binding(value: Any, expected: Mapping[str, Any]) -> bool:
    return isinstance(value, Mapping) and dict(value) == dict(expected)


def _binding_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    """Compare the stable path/hash/size fields; old receipts may add kind."""

    if not isinstance(value, Mapping):
        return False
    return all(value.get(key) == expected.get(key) for key in ("path", "sha256", "size_bytes"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _as_ints(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("token vector is not a sequence")
    return [int(item) for item in value]


def _row_same(left: Mapping[str, Any], right: Mapping[str, Any], epsilon: int = 0) -> bool:
    left_desc = tuple(left.get("description_tokens", left.get("description_token_ids", ())))
    right_desc = tuple(right.get("description_tokens", right.get("description_token_ids", ())))
    left_values = tuple(left.get("values", left.get("coord_bins_canvas", ())))
    right_values = tuple(right.get("values", right.get("coord_bins_canvas", ())))
    return bool(left_values and left_desc == right_desc and len(left_values) == 4 and
                len(right_values) == 4 and max(abs(a - b) for a, b in zip(left_values, right_values)) <= epsilon)


def parse_rows(token_ids: Sequence[int]) -> dict[str, Any]:
    """Parse all object starts, retaining complete invalid rows and fragments."""

    tokens = [int(value) for value in token_ids]
    starts = [index for index, token in enumerate(tokens) if token == OBJ_START]
    rows: list[dict[str, Any]] = []
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(tokens)
        segment = tokens[start:stop]
        try:
            description_end = segment.index(OBJ_END, 1)
            complete = (
                OBJ_START not in segment[1:description_end]
                and description_end + 6 < len(segment)
                and segment[description_end + 1] == BOX_START
                and segment[description_end + 6] == BOX_END
                and all(BASE <= token < COORD_LIMIT for token in segment[description_end + 2:description_end + 6])
            )
        except ValueError:
            complete = False
            description_end = -1
        if not complete:
            rows.append({
                "row_index": row_index,
                "status": "malformed",
                "complete": False,
                "token_count": len(segment),
                "raw_token_ids": segment,
            })
            continue
        values = [token - BASE for token in segment[description_end + 2:description_end + 6]]
        rows.append({
            "row_index": row_index,
            "status": "valid" if values[0] < values[2] and values[1] < values[3] else "invalid",
            "complete": True,
            "description_tokens": segment[1:description_end],
            "values": values,
            "coordinate_offsets": [start + description_end + offset for offset in (2, 3, 4, 5)],
            "start": start,
            "end": start + description_end + 7,
            "token_count": description_end + 7,
            "raw_token_ids": segment[:description_end + 7],
        })
    complete = [row for row in rows if row["complete"]]
    malformed_openers = len(rows) - len(complete)
    consumed = sum(int(row["token_count"]) for row in rows)
    return {
        "rows": rows,
        "complete_rows": len(complete),
        "valid_rows": sum(row["status"] == "valid" for row in complete),
        "invalid_rows": sum(row["status"] == "invalid" for row in complete),
        "malformed_rows": malformed_openers,
        "malformed_openers": malformed_openers,
        "unparsed_tokens": max(0, len(tokens) - consumed - int(EOS in tokens)),
        "eos": EOS in tokens,
    }


def _pairs(rows: Sequence[Mapping[str, Any]], epsilon: int) -> list[dict[str, Any]]:
    complete = [row for row in rows if row.get("complete", False)]
    return [
        {
            "earlier": int(left["row_index"]),
            "later": int(right["row_index"]),
            "max_coordinate_delta": max(
                abs(a - b) for a, b in zip(left["values"], right["values"])
            ),
            "description_tokens": list(left.get("description_tokens", ())),
        }
        for index, left in enumerate(complete)
        for right in complete[index + 1:]
        if _row_same(left, right, epsilon)
    ]


def _runs(rows: Sequence[Mapping[str, Any]], epsilon: int) -> list[dict[str, Any]]:
    complete = [row for row in rows if row.get("complete", False)]
    result: list[dict[str, Any]] = []
    for start in range(len(complete)):
        run = [complete[start]]
        for row in complete[start + 1:]:
            if not _row_same(row, run[0], epsilon) or any(not _row_same(row, prev, epsilon) for prev in run):
                break
            run.append(row)
        if len(run) >= 2:
            result.append({
                "start_row": int(run[0]["row_index"]),
                "length": len(run),
                "row_indices": [int(row["row_index"]) for row in run],
                "kind": "exact" if epsilon == 0 else "near",
            })
    return result


def _longest(rows: Sequence[Mapping[str, Any]], epsilon: int) -> int:
    runs = _runs(rows, epsilon)
    return max((int(run["length"]) for run in runs), default=1 if any(row.get("complete", False) for row in rows) else 0)


def _context_repeats(context: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old = [
        {
            "free_row": int(row["row_index"]),
            "context_row": int(old_row["row_index"]),
            "exact": _row_same(row, old_row, 0),
            "near": _row_same(row, old_row, NEAR_EPS),
        }
        for row in rows
        if row.get("complete", False)
        for old_row in context
        if old_row.get("complete", False) and _row_same(row, old_row, NEAR_EPS)
    ]
    within_exact = _pairs(rows, 0)
    within_near = _pairs(rows, NEAR_EPS)
    return {
        "old_context_exact_or_near": old,
        "old_context_exact_count": sum(item["exact"] for item in old),
        "old_context_near_count": len(old),
        "within_free_exact_pairs": within_exact,
        "within_free_near_pairs": within_near,
    }


def _stop_reason(record: Mapping[str, Any], token_ids: Sequence[int], parsed: Mapping[str, Any]) -> str:
    stop = record.get("stop", record.get("stop_reason", record.get("termination")))
    if stop is None and isinstance(record.get("release"), Mapping):
        stop = record["release"].get("stop_reason", record["release"].get("stop"))
    if isinstance(stop, Mapping):
        stop = stop.get("reason", stop.get("status"))
    if stop:
        return str(stop)
    if parsed.get("eos"):
        return "eos"
    if len(token_ids) >= 256:
        return "token_cap"
    return "unknown"


def release_metrics(
    token_ids: Sequence[int],
    *,
    context_tokens: Sequence[int] = (),
    record: Mapping[str, Any] | None = None,
    horizon: int = 16,
) -> dict[str, Any]:
    parsed = parse_rows(token_ids)
    rows = parsed["rows"]
    context = parse_rows(context_tokens)["rows"] if context_tokens else []
    metrics = {
        "token_count": len(token_ids),
        "stop_reason": _stop_reason(record or {}, token_ids, parsed),
        "complete_rows": parsed["complete_rows"],
        "valid_rows": parsed["valid_rows"],
        "invalid_rows": parsed["invalid_rows"],
        "malformed_rows": parsed["malformed_rows"],
        "malformed_openers": parsed["malformed_openers"],
        "unparsed_tokens": parsed["unparsed_tokens"],
        "eos": parsed["eos"],
        "token_cap_reached": len(token_ids) >= 256,
        "cap_without_eos": len(token_ids) >= 256 and not parsed["eos"],
        "token_cap_fragment": len(token_ids) >= 256 and not parsed["eos"] and parsed["unparsed_tokens"] > 0,
        "exact_pairs": _pairs(rows, 0),
        "near_pairs": _pairs(rows, NEAR_EPS),
        "longest_exact_run": _longest(rows, 0),
        "longest_near_run": _longest(rows, NEAR_EPS),
        "rows": rows,
        "horizons": {},
        "physical_recovery": "not_established_by_numerical_metrics",
    }
    metrics.update(_context_repeats(context, rows))
    for target in (8, horizon):
        if parsed["complete_rows"] >= target:
            prefix = [row for row in rows if row.get("complete", False)][:target]
            metrics["horizons"][str(target)] = {
                "status": "reached",
                "complete_rows": target,
                "exact_pairs": _pairs(prefix, 0),
                "near_pairs": _pairs(prefix, NEAR_EPS),
                "longest_exact_run": _longest(prefix, 0),
                "longest_near_run": _longest(prefix, NEAR_EPS),
            }
        else:
            metrics["horizons"][str(target)] = {
                "status": "unobserved",
                "complete_rows": parsed["complete_rows"],
            }
    return metrics


def _annotation_bank(summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    panel_ref = summary.get("source_panel")
    if not isinstance(panel_ref, Mapping) or not panel_ref.get("path"):
        return [], None
    path = Path(str(panel_ref["path"]))
    if not path.is_file() or not _same_binding(panel_ref, _file_binding(path)):
        return [], None
    panel = _read_json(path)
    group = next((item for item in panel.get("groups", []) if item.get("key") == summary.get("group")), None)
    if not group:
        return [], None
    case = next(
        (item for item in group.get("cases", [])
         if int(item.get("input_record", {}).get("image_id", -1)) == int(summary.get("image_id", -2))),
        None,
    )
    objects = case.get("input_record", {}).get("objects", []) if case else []
    bank: list[dict[str, Any]] = []
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ROOT, use_fast=False, local_files_only=True)
        except Exception:
            _TOKENIZER = False
    pattern = re.compile(r"coord_(\d+)")
    for index, obj in enumerate(objects):
        values: list[int] = []
        for token in obj.get("bbox_2d", ()):
            match = pattern.fullmatch(str(token).strip("<>|"))
            if match is None:
                values = []
                break
            values.append(int(match.group(1)))
        if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
            continue
        description = str(obj.get("desc", obj.get("category_name", ""))).strip().lower()
        description_tokens = None
        if _TOKENIZER:
            description_tokens = _TOKENIZER.encode(description, add_special_tokens=False)
        bank.append({
            "owner_id": str(obj.get("coco_ann_id", f"annotation-{index}")),
            "description": description,
            "description_tokens": description_tokens,
            "coord_bins": values,
        })
    return bank, f"{path} group={summary.get('group')} image_id={summary.get('image_id')}" if case else None


def _iou(left: Sequence[int], right: Sequence[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0 else intersection / union


def known_matches(rows: Sequence[Mapping[str, Any]], bank: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        if row.get("status") != "valid" or ("source_geometry_valid" in row and not row.get("source_geometry_valid")):
            continue
        values = row.get("source_values", row.get("values"))
        if not isinstance(values, Sequence) or len(values) != 4:
            continue
        description = str(row.get("description", "")).strip().lower()
        description_tokens = tuple(row.get("description_tokens", ()))
        # Parser-only rows carry token IDs; an annotation match is allowed only
        # when the receipt supplied a decoded/source description explicitly.
        if not description:
            continue
        for target in bank:
            token_match = bool(description_tokens and target.get("description_tokens") and
                               description_tokens == tuple(target["description_tokens"]))
            if not token_match and (not description or description != str(target["description"]).lower()):
                continue
            candidates.append((_iou(values, target["coord_bins"]), row, target))
    candidates.sort(key=lambda item: (-item[0], int(item[1]["row_index"]), str(item[2]["owner_id"])))
    used_rows: set[int] = set()
    used_owners: set[str] = set()
    matches = []
    for score, row, target in candidates:
        row_index = int(row["row_index"])
        owner_id = str(target["owner_id"])
        if score < 0.5 or row_index in used_rows or owner_id in used_owners:
            continue
        used_rows.add(row_index)
        used_owners.add(owner_id)
        matches.append({"row_index": row_index, "owner_id": owner_id, "iou": score})
    return {"status": "bound", "bank_count": len(bank), "matched_count": len(matches), "matches": matches,
            "claim_boundary": "annotation_relative_only; no physical-owner claim"}


def _walk_receipts(value: Any, *, source: Path, stem_id: str | None = None) -> Iterable[tuple[str, dict[str, Any], Path]]:
    """Yield direct cell records from manifests without traversing metric payloads."""

    if isinstance(value, Mapping):
        identifier = value.get("cell_id", value.get("id"))
        if identifier is None and isinstance(value.get("cell"), Mapping):
            identifier = value["cell"].get("id", value["cell"].get("cell_id"))
        if identifier is not None and not isinstance(identifier, (str, int)):
            identifier = None
        if identifier is not None and any(key in value for key in ("token_ids", "free_token_ids", "vectors", "stages", "first_free", "target", "release")):
            yield str(identifier), dict(value), source
            return
        cell_ids = value.get("cell_ids")
        if isinstance(cell_ids, Sequence) and not isinstance(cell_ids, (str, bytes, bytearray)):
            for cell_id in cell_ids:
                record = dict(value)
                record["cell_id"] = str(cell_id)
                yield str(cell_id), record, source
            return
        schema = str(value.get("schema", ""))
        if schema == "recurrence_phase_decision.profile.v1" and value.get("boundary_id"):
            yield f"@boundary:{value['boundary_id']}", dict(value), source
            return
        for key in ("cells", "records", "entries", "results", "releases", "captures"):
            child = value.get(key)
            if isinstance(child, Mapping):
                for child_id, child_value in child.items():
                    if isinstance(child_value, Mapping):
                        record = dict(child_value)
                        record.setdefault("cell_id", str(child_id))
                        yield from _walk_receipts(record, source=source)
            elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                for item in child:
                    yield from _walk_receipts(item, source=source)
            if child is not None:
                return
        # A shard receipt, launch snapshot, or index is a package manifest,
        # not a cell record.  Only schema-less per-cell JSON may use its file
        # stem as the fallback identifier.
        if stem_id is not None and not schema:
            record = dict(value)
            record.setdefault("cell_id", stem_id)
            yield stem_id, record, source
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_receipts(item, source=source)


def _load_torch(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only used for tensor receipts
        raise ValueError(f"cannot read tensor receipt without torch: {path}") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_sha256(value: Any) -> str | None:
    try:
        import torch
        if not isinstance(value, torch.Tensor):
            return None
        value = value.detach().cpu().contiguous()
        return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    except (ImportError, RuntimeError, TypeError):
        return None


def _value_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weight_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.name.endswith("weights.pt") else []
    return sorted(item for item in path.rglob("*.pt") if item.is_file() and item.name.endswith("weights.pt"))


def _load_embedding_weights(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and bind the saved coordinate input embeddings once per model."""

    bank: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for weight_path in _weight_files(path):
        payload = _load_torch(weight_path)
        if not isinstance(payload, Mapping) or payload.get("schema") != "recurrence_phase_decision.coordinate_weights.v1":
            continue
        model = str(payload.get("model", ""))
        rows = payload.get("input_rows")
        coordinate_ids = [int(value) for value in payload.get("coordinate_ids", ())]
        if not model or not hasattr(rows, "shape") or list(rows.shape) != [1000, 2048]:
            raise ValueError(f"malformed coordinate embedding receipt: {weight_path}")
        expected_ids = list(range(BASE, BASE + 1000))
        if coordinate_ids != expected_ids:
            raise ValueError(f"coordinate embedding IDs changed: {weight_path}")
        input_hash = _tensor_sha256(rows)
        output_hash = _tensor_sha256(payload.get("output_rows"))
        if input_hash is None or output_hash is None or input_hash != payload.get("input_rows_sha256") or output_hash != payload.get("output_rows_sha256"):
            raise ValueError(f"coordinate embedding hash mismatch: {weight_path}")
        identity_hash = _value_sha256(payload.get("model_identity", {}))
        binding = _file_binding(weight_path)
        previous = bank.get(model)
        if previous is not None:
            if (previous["input_rows_sha256"], previous["output_rows_sha256"], previous["identity_sha256"]) != (input_hash, output_hash, identity_hash):
                raise ValueError(f"coordinate embedding identity differs across shards: {model}")
            continue
        bank[model] = {
            "input_rows": rows.detach().float().cpu(),
            "input_rows_sha256": input_hash,
            "output_rows_sha256": output_hash,
            "identity_sha256": identity_hash,
            "binding": binding,
        }
        bindings[model] = {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
            "input_rows_sha256": input_hash,
            "output_rows_sha256": output_hash,
            "identity_sha256": identity_hash,
        }
    return bank, bindings


def _embedding_delta(cell: Mapping[str, Any], weights: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    model = str(cell.get("model", ""))
    item = weights.get(model)
    if item is None:
        return {"status": "missing", "model": model, "reason": "no_bound_coordinate_weights"}
    old_id = int(cell["old_token_id"])
    new_id = int(cell["new_token_id"])
    old_index, new_index = old_id - BASE, new_id - BASE
    rows = item["input_rows"]
    if not (0 <= old_index < len(rows) and 0 <= new_index < len(rows)):
        return {"status": "missing", "model": model, "reason": "token_outside_coordinate_weight_rows"}
    delta = (rows[new_index] - rows[old_index]).float()
    values = [float(value) for value in delta.tolist()]
    return {
        "status": "present",
        "model": model,
        "old_token_id": old_id,
        "new_token_id": new_id,
        "dimension": len(values),
        "l2": math.sqrt(sum(value * value for value in values)),
        "max_abs": max((abs(value) for value in values), default=0.0),
        "values": values,
        "input_rows_sha256": item["input_rows_sha256"],
        "output_rows_sha256": item["output_rows_sha256"],
        "identity_sha256": item["identity_sha256"],
    }


def _load_windows(plan_path: Path, windows_path: Path | None) -> tuple[dict[tuple[str, int, str], Mapping[str, Any]], dict[str, Any] | None]:
    """Load the frozen slot windows; never regenerate them from outcomes."""

    if windows_path is None:
        candidate = plan_path.with_name("windows.json")
        windows_path = candidate if candidate.is_file() else None
    if windows_path is None:
        return {}, None
    windows_path = windows_path.resolve(strict=True)
    value = _read_json(windows_path)
    if value.get("schema") != "recurrence_phase_decision.windows.v1" or value.get("status") != "frozen_before_broad_capture":
        raise ValueError("windows receipt is not the frozen phase-decision windows schema")
    if not _binding_matches(value.get("plan"), _file_binding(plan_path)):
        raise ValueError("windows receipt is bound to a different plan")
    slots = value.get("slots")
    if not isinstance(slots, Sequence):
        raise ValueError("windows receipt lacks slots")
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise ValueError("malformed windows slot")
        key = (str(slot.get("boundary_id")), int(slot.get("row_index")), str(slot.get("role")))
        if key in result:
            raise ValueError(f"duplicate windows slot {key}")
        result[key] = dict(slot)
    file_ref = _file_binding(windows_path)
    for slot in result.values():
        slot["binding"] = file_ref
    return result, file_ref


def _receipt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".pt", ".pth"})


def load_receipts(path: Path | Sequence[Path]) -> tuple[dict[str, tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    bindings: list[dict[str, Any]] = []
    paths = [path] if isinstance(path, Path) else list(path)
    for root in paths:
        for file in _receipt_files(root):
            bindings.append(_file_binding(file))
            if file.suffix.lower() == ".jsonl":
                value: Any = [_read_json_line(line, file) for line in file.read_text().splitlines() if line.strip()]
            elif file.suffix.lower() in {".pt", ".pth"}:
                value = _load_torch(file)
            else:
                value = _read_json(file)
            stem = file.stem if file.stem not in {"capture", "release", "manifest", "result"} else None
            found = list(_walk_receipts(value, source=file, stem_id=stem))
            for identifier, record, source in found:
                if identifier in records:
                    raise ValueError(f"duplicate {identifier} receipt: {source} and {records[identifier][1]}")
                records[identifier] = (record, source)
    return records, bindings


def _tensor_to_list(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tensor_to_list(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_to_list(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().float().cpu().tolist()
    return value


def _resolve_vector(value: Any, base: Path) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("values", "logits", "vector", "scores"):
            if key in value:
                return _resolve_vector(value[key], base)
        for key in ("path", "file"):
            if key in value:
                path = Path(str(value[key]))
                if not path.is_absolute():
                    path = base / path
                loaded = _load_torch(path) if path.suffix.lower() in {".pt", ".pth"} else _read_json(path)
                return _resolve_vector(loaded, path.parent)
        return None
    value = _tensor_to_list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _logprob_metrics(logits: Sequence[float], target: int | None, *, historical: Sequence[int] = ()) -> dict[str, Any]:
    if not logits:
        return {"status": "missing"}
    try:
        import numpy as np
        values = np.asarray(logits, dtype=np.float64)
        maximum = float(values.max())
        weights = np.exp(values - maximum)
        total = float(weights.sum())
        probability = lambda index: float(weights[index] / total) if 0 <= index < len(values) else None
        top_indices = np.argpartition(-values, min(2, len(values)) - 1)[: min(2, len(values))]
        order = sorted((int(index) for index in top_indices), key=lambda index: (-float(values[index]), index))
        probabilities = None
    except ImportError:  # pragma: no cover - NumPy is part of the runtime image
        maximum = max(float(value) for value in logits)
        exp_values = [math.exp(float(value) - maximum) for value in logits]
        total = sum(exp_values)
        probabilities = [value / total for value in exp_values]
        probability = lambda index: probabilities[index] if 0 <= index < len(probabilities) else None
        order = sorted(range(len(logits)), key=lambda index: (-float(logits[index]), index))[:2]
    winner = order[0]
    second = order[1] if len(order) > 1 else winner
    result: dict[str, Any] = {
        "status": "present",
        "vocab_size": len(logits),
        "winner_token_id": winner,
        "winner_logit": float(logits[winner]),
        "top2": [[int(index), float(logits[index])] for index in order[:2]],
        "top1_margin": float(logits[winner] - logits[second]),
        "eos_probability": probability(EOS),
        "opener_probability": probability(OBJ_START),
        "eos_logit": float(logits[EOS]) if EOS < len(logits) else None,
        "opener_logit": float(logits[OBJ_START]) if OBJ_START < len(logits) else None,
        "boundary_competition": {
            "eos_vs_opener_logit_margin": (
                float(logits[EOS] - logits[OBJ_START]) if EOS < len(logits) and OBJ_START < len(logits) else None
            ),
            "eos_vs_opener_probability_margin": (
                float(probability(EOS) - probability(OBJ_START))
                if probability(EOS) is not None and probability(OBJ_START) is not None else None
            ),
        },
    }
    if target is not None and 0 <= target < len(logits):
        target_logit = float(logits[target])
        result.update({
            "observed_token_id": int(target),
            "observed_probability": probability(target),
            "observed_logprob": math.log(max(probability(target) or 0.0, 1e-45)),
            "observed_rank": 1 + (int(np.count_nonzero(values > target_logit)) if probabilities is None else sum(float(logits[index]) > target_logit for index in range(len(logits)))),
        })
    family_start, family_stop = BASE, min(COORD_LIMIT, len(logits))
    if probabilities is None:
        family_weights = weights[family_start:family_stop]
        family_total = float(family_weights.sum() / total)
    else:
        family_weights = None
        family_total = sum(probabilities[index] for index in range(family_start, family_stop))
    result["coordinate_family_mass"] = family_total
    result["coordinate_family_conditional_entropy"] = None
    if family_total and probabilities is None:
        normalized = family_weights / float(family_weights.sum())
        result["coordinate_family_conditional_entropy"] = float(-sum(float(value) * math.log(float(value)) for value in normalized if value))
    elif family_total:
        normalized = [probabilities[index] / family_total for index in range(family_start, family_stop) if probabilities[index]]
        result["coordinate_family_conditional_entropy"] = -sum(value * math.log(value) for value in normalized)
    result["coordinate_entropy"] = result["coordinate_family_conditional_entropy"]
    result["coordinate_endpoints"] = {
        "coord_0_full_vocab_probability": probability(BASE),
        "coord_999_full_vocab_probability": probability(BASE + 999),
        "coord_0_conditional_probability": (probability(BASE) / family_total if family_total and probability(BASE) is not None else None),
        "coord_999_conditional_probability": (probability(BASE + 999) / family_total if family_total and probability(BASE + 999) is not None else None),
    }
    historical_ids = sorted({BASE + int(value) for value in historical if 0 <= int(value) < 1000})
    result["historical_coordinate_window_ids"] = historical_ids
    if probabilities is None:
        result["historical_coordinate_window_mass"] = float(weights[historical_ids].sum() / total) if historical_ids else 0.0
    else:
        result["historical_coordinate_window_mass"] = sum(probabilities[index] for index in historical_ids if index < len(probabilities))
    result["historical_coordinate_window_conditional_mass"] = (
        result["historical_coordinate_window_mass"] / family_total if family_total else None
    )
    endpoint_ids = {historical_ids[0], historical_ids[-1]} if historical_ids else set()
    if probabilities is None:
        result["historical_endpoint_mass"] = float(weights[list(endpoint_ids)].sum() / total) if endpoint_ids else 0.0
    else:
        result["historical_endpoint_mass"] = sum(probabilities[index] for index in endpoint_ids if index < len(probabilities))
    return result


def _stage_entries(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("stages", "row_stages", "vectors", "fixed_vectors", "scores"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return [dict(item, slot=str(name)) if isinstance(item, Mapping) else {"slot": str(name), "logits": item}
                    for name, item in value.items()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [dict(item) if isinstance(item, Mapping) else {"logits": item} for item in value]
    capture = record.get("capture")
    if isinstance(capture, Mapping) and isinstance(capture.get("slots"), Sequence):
        registered = {
            int(item["offset"]): item
            for item in record.get("registered_slots", ())
            if isinstance(item, Mapping) and item.get("offset") is not None
        }
        return [
            {
                **dict(item),
                "logits": item.get("logits", item.get("scores")),
                "target_token_id": item.get("target_token_id", item.get("observed_token_id")),
                "offset": item.get("offset"),
                "name": registered.get(int(item.get("offset", -1)), {}).get("name"),
                "boundary_id": record.get("boundary_id"),
                "row_index": record.get("row_index", record.get("target_row_index")),
            }
            for item in capture["slots"]
            if isinstance(item, Mapping)
        ]
    return []


def _variant_name(record: Mapping[str, Any]) -> str:
    value = record.get("variant", record.get("mode", record.get("role", "")))
    return str(value).lower()


def _first_free(record: Mapping[str, Any], base: Path) -> dict[str, Any] | None:
    candidates = record.get("first_free", record.get("first_free_choice", record.get("first_free_fork")))
    if isinstance(candidates, Mapping):
        if any(key in candidates for key in ("native", "changed", "control", "edited")):
            # Caller handles the per-variant map; this path is a container.
            return None
        candidates = dict(candidates)
    if candidates is None:
        candidates = record.get("first_free_argmax")
    if candidates is None:
        candidates = {}
    if isinstance(candidates, Mapping) and "logits" not in candidates and "scores" not in candidates and "vector" not in candidates:
        if any(key in record for key in ("first_free_logits", "first_logits")):
            candidates = dict(candidates)
            candidates["logits"] = record.get("first_free_logits", record.get("first_logits"))
    if not isinstance(candidates, Mapping):
        return {"choice": int(candidates)}
    value = dict(candidates)
    logits = _resolve_vector(value.get("logits", value.get("scores", value.get("vector"))), base)
    if logits is not None:
        target = value.get("chosen_token_id", value.get("token_id", value.get("choice")))
        value["vector_metrics"] = _logprob_metrics(logits, int(target) if target is not None else None)
        value.setdefault("chosen_token_id", value["vector_metrics"].get("winner_token_id"))
        value.setdefault("margin", value["vector_metrics"].get("top1_margin"))
        for key in ("logits", "scores", "vector"):
            value.pop(key, None)
    if value.get("margin") is None and isinstance(value.get("top_competitors"), Sequence):
        top = value["top_competitors"]
        if len(top) >= 2 and isinstance(top[0], Mapping) and isinstance(top[1], Mapping):
            value["margin"] = float(top[0].get("logit", 0.0)) - float(top[1].get("logit", 0.0))
        elif len(top) >= 2 and isinstance(top[0], Sequence) and isinstance(top[1], Sequence):
            value["margin"] = float(top[0][1]) - float(top[1][1])
    value.setdefault("chosen_token_id", value.get("winner_token_id"))
    return value


def _capture_variants(record: Mapping[str, Any], base: Path, windows: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    value = record.get("variants", record.get("branches"))
    if isinstance(value, Mapping):
        for name, item in value.items():
            if isinstance(item, Mapping):
                variants[str(name)] = dict(item)
    for name in ("native", "control", "changed", "edited"):
        item = record.get(name)
        if isinstance(item, Mapping):
            variants.setdefault(name, dict(item))
    first = record.get("first_free")
    if isinstance(first, Mapping):
        for name, item in first.items():
            if isinstance(item, Mapping):
                variants.setdefault(str(name), dict(item))
    if not variants:
        variants[_variant_name(record) or "cell"] = dict(record)
    for name, item in variants.items():
        first_choice = _first_free(item, base)
        if first_choice is None and item is not record:
            first_choice = _first_free(record, base)
        if first_choice is not None:
            item["first_free_metrics"] = first_choice
        item["stages_metrics"] = _reduce_stages(item, base, windows)
    return variants


def _reduce_stages(record: Mapping[str, Any], base: Path, windows: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    # The saved capture keeps the FP32 head input for paired delta checks. Do
    # not copy that large tensor into the JSON reduction; retain compact stage
    # metadata and the derived full-vocabulary metrics instead.
    dropped = {"logits", "scores", "vector", "head_input"}

    def serializable(stage: Mapping[str, Any]) -> dict[str, Any]:
        return {key: _tensor_to_list(value) for key, value in stage.items() if key not in dropped}

    out = []
    for stage in _stage_entries(record):
        slot_name = str(stage.get("name", ""))
        role = stage.get("role")
        if role is None:
            suffix = slot_name.rsplit("_", 1)[-1]
            role = suffix if suffix in ROLE_INDEX else (slot_name if slot_name in ROLE_INDEX else None)
        if role is not None:
            stage["role"] = role
        logits = _resolve_vector(stage.get("logits", stage.get("scores", stage.get("vector"))), base)
        if logits is None:
            out.append(serializable(stage) | {"status": "missing_vector"})
            continue
        target = stage.get("observed_token_id", stage.get("target_token_id", stage.get("token_id")))
        if target is None and stage.get("target_value") is not None:
            target = BASE + int(stage["target_value"])
        supplied_historical = stage.get("historical_coordinate_values", stage.get("historical_values", ()))
        if isinstance(supplied_historical, Mapping):
            supplied_historical = supplied_historical.get("values", ())
        historical = supplied_historical
        window_binding = None
        window_mismatch = False
        if windows is not None:
            key = (str(stage.get("boundary_id", record.get("boundary_id", ""))), int(stage.get("row_index", -1)), str(stage.get("role", "")))
            window = windows.get(key)
            if window is not None:
                frozen_historical = list(window.get("window_bins", ()))
                window_mismatch = bool(historical) and list(historical) != frozen_historical
                historical = frozen_historical
                window_binding = window.get("binding")
        metrics = _logprob_metrics(logits, int(target) if target is not None else None,
                                   historical=[int(item) for item in historical])
        if window_binding is not None:
            metrics["historical_window_binding"] = window_binding
            metrics["historical_window_input_mismatch"] = window_mismatch
        out.append(serializable(stage) | metrics)
    return out


def _fixed_suffix_pairs(record: Mapping[str, Any], base: Path) -> list[dict[str, Any]]:
    value = record.get("fixed_suffix", record.get("fixed", record.get("fixed_vectors")))
    if not isinstance(value, Mapping):
        return []
    native = value.get("native", value.get("control"))
    changed = value.get("changed", value.get("edited"))
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes, bytearray)):
        native = [native] if native is not None else []
    if not isinstance(changed, Sequence) or isinstance(changed, (str, bytes, bytearray)):
        changed = [changed] if changed is not None else []
    result = []
    for index, (left, right) in enumerate(zip(native, changed)):
        left_logits = _resolve_vector(left, base)
        right_logits = _resolve_vector(right, base)
        if left_logits is None or right_logits is None or len(left_logits) != len(right_logits):
            result.append({"index": index, "status": "missing_or_mismatched_vector"})
            continue
        target = None
        if isinstance(right, Mapping):
            target = right.get("target_token_id", right.get("token_id"))
        if target is None and isinstance(left, Mapping):
            target = left.get("target_token_id", left.get("token_id"))
        delta = [b - a for a, b in zip(left_logits, right_logits)]
        centered = sum(delta) / len(delta)
        native_metrics = _logprob_metrics(left_logits, int(target) if target is not None else None)
        changed_metrics = _logprob_metrics(right_logits, int(target) if target is not None else None)
        native_winner = int(native_metrics.get("winner_token_id", -1))
        result.append({
            "index": index,
            "status": "present",
            "target_token_id": target,
            "centered_score_delta": (float(delta[int(target)] - centered) if target is not None and 0 <= int(target) < len(delta) else None),
            "logprob_delta": (
                changed_metrics.get("observed_logprob", 0.0) - native_metrics.get("observed_logprob", 0.0)
                if target is not None and "observed_logprob" in native_metrics and "observed_logprob" in changed_metrics else None
            ),
            "native_winner": native_winner,
            "changed_winner": int(changed_metrics.get("winner_token_id", -1)),
            "native_top2": native_metrics.get("top2"),
            "changed_top2": changed_metrics.get("top2"),
            "original_winner_new_margin": (
                float(right_logits[native_winner] - max(value for idx, value in enumerate(right_logits) if idx != native_winner))
                if 0 <= native_winner < len(right_logits) and len(right_logits) > 1 else None
            ),
            "native_top2": native_metrics.get("top2"),
            "changed_top2": changed_metrics.get("top2"),
        })
    return result


def _capture_fixed_pairs(record: Mapping[str, Any], control: Mapping[str, Any] | None, base: Path, control_base: Path | None) -> list[dict[str, Any]]:
    """Pair native and edited saved slot vectors at the same release boundary."""

    if control is None:
        return []

    def slots(item: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
        capture = item.get("capture")
        if not isinstance(capture, Mapping) or not isinstance(capture.get("slots"), Sequence):
            return {}
        return {
            int(slot["offset"]): slot
            for slot in capture["slots"]
            if isinstance(slot, Mapping) and slot.get("offset") is not None
        }

    left, right = slots(control), slots(record)
    result = []
    for offset in sorted(left.keys() & right.keys()):
        native = _resolve_vector(left[offset].get("logits"), control_base or base)
        changed = _resolve_vector(right[offset].get("logits"), base)
        if native is None or changed is None or len(native) != len(changed):
            result.append({"offset": offset, "status": "missing_or_mismatched_vector"})
            continue
        delta = [b - a for a, b in zip(native, changed)]
        centered = sum(delta) / len(delta)
        target = right[offset].get("target_token_id", left[offset].get("target_token_id"))
        native_metrics = _logprob_metrics(native, int(target) if target is not None else None)
        changed_metrics = _logprob_metrics(changed, int(target) if target is not None else None)
        native_winner = int(native_metrics.get("winner_token_id", -1))
        native_head = _resolve_vector(left[offset].get("head_input"), control_base or base)
        changed_head = _resolve_vector(right[offset].get("head_input"), base)
        head_delta = None
        if native_head is not None and changed_head is not None and len(native_head) == len(changed_head):
            head_delta_values = [b - a for a, b in zip(native_head, changed_head)]
            head_delta = {
                "l2": math.sqrt(sum(value * value for value in head_delta_values)),
                "max_abs": max(map(abs, head_delta_values), default=0.0),
                "dimension": len(head_delta_values),
            }
        result.append({
            "offset": offset,
            "slot": right[offset].get("name"),
            "status": "present",
            "target_token_id": target,
            "centered_score_delta": float(delta[int(target)] - centered) if target is not None and 0 <= int(target) < len(delta) else None,
            "native_winner": native_winner,
            "changed_winner": int(changed_metrics.get("winner_token_id", -1)),
            "original_winner_new_margin": (
                float(changed[native_winner] - max(value for index, value in enumerate(changed) if index != native_winner))
                if 0 <= native_winner < len(changed) and len(changed) > 1 else None
            ),
            "head_input_delta": head_delta,
        })
    return result


def _head_delta(record: Mapping[str, Any], base: Path) -> dict[str, Any] | None:
    native = _resolve_vector(record.get("native_head_input", record.get("native_input_vector")), base)
    changed = _resolve_vector(record.get("changed_head_input", record.get("changed_input_vector")), base)
    if native is None or changed is None or len(native) != len(changed):
        return None
    delta = [right - left for left, right in zip(native, changed)]
    return {"l2": math.sqrt(sum(value * value for value in delta)), "max_abs": max(map(abs, delta), default=0.0), "dimension": len(delta)}


def _head_delta_pair(record: Mapping[str, Any], control: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if control is None:
        return None
    def first_head(value: Mapping[str, Any]) -> Any:
        direct = value.get("first_free_head_input", value.get("head_input"))
        if direct is not None:
            return direct
        capture = value.get("capture")
        if isinstance(capture, Mapping) and isinstance(capture.get("slots"), Sequence):
            slots = [slot for slot in capture["slots"] if isinstance(slot, Mapping) and slot.get("offset") is not None and slot.get("head_input") is not None]
            if slots:
                return min(slots, key=lambda slot: int(slot["offset"])).get("head_input")
        return None
    left = _tensor_to_list(first_head(control))
    right = _tensor_to_list(first_head(record))
    if not isinstance(left, Sequence) or not isinstance(right, Sequence) or len(left) != len(right):
        return None
    try:
        delta = [float(b) - float(a) for a, b in zip(left, right)]
    except (TypeError, ValueError):
        return None
    return {"l2": math.sqrt(sum(value * value for value in delta)), "max_abs": max(map(abs, delta), default=0.0), "dimension": len(delta)}


def _free_tokens(record: Mapping[str, Any]) -> list[int]:
    for key in ("free_token_ids", "generated_token_ids", "suffix_token_ids", "tokens"):
        if key in record:
            return _as_ints(record[key])
    target = record.get("target")
    if isinstance(target, Mapping):
        for key in ("free_token_ids", "generated_token_ids", "token_ids", "tokens"):
            if key in target:
                return _as_ints(target[key])
    if "token_ids" in record:
        return _as_ints(record["token_ids"])
    return []


def _first_divergence(control: Mapping[str, Any] | None, variant: Mapping[str, Any] | None) -> dict[str, Any]:
    if control is None or variant is None:
        return {"status": "missing"}
    left, right = _free_tokens(control), _free_tokens(variant)
    if not left or not right:
        return {"status": "missing"}
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {
                "status": "diverged",
                "position": position,
                "control_token_id": a,
                "variant_token_id": b,
                "control_length": len(left),
                "variant_length": len(right),
            }
    return {
        "status": "identical_prefix",
        "position": min(len(left), len(right)),
        "control_length": len(left),
        "variant_length": len(right),
    }


def _claimed_parse(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("parse", record.get("metrics"))
    if value is None and isinstance(record.get("release"), Mapping):
        value = record["release"]
    if value is None and isinstance(record.get("target"), Mapping):
        value = record["target"].get("release_metrics", record["target"].get("metrics"))
    return value if isinstance(value, Mapping) else None


def _validate_saved_parse(record: Mapping[str, Any], parsed: Mapping[str, Any]) -> None:
    claimed = _claimed_parse(record)
    if claimed is not None:
        for key in ("complete_rows", "valid_rows", "invalid_rows", "malformed_rows", "malformed_openers"):
            if key in claimed and int(claimed[key]) != int(parsed[key]):
                raise ValueError(f"saved-output {key} disagrees with raw token parse")
    saved_rows = record.get("rows")
    if isinstance(saved_rows, Sequence) and not isinstance(saved_rows, (str, bytes, bytearray)):
        if len(saved_rows) != parsed["complete_rows"] + parsed["malformed_rows"]:
            raise ValueError("saved-output rows dropped complete-invalid or malformed row")


def _binding_issues(record: Mapping[str, Any], plan_path: Path, panel_binding: Mapping[str, Any], summary: Mapping[str, Any] | None = None) -> list[str]:
    issues = []
    for key in ("plan", "source_plan", "plan_binding"):
        if key in record and not _binding_matches(record[key], _file_binding(plan_path)):
            issues.append(f"{key}_mismatch")
    for key in ("panel", "source_panel", "panel_binding"):
        if key in record and isinstance(record[key], Mapping) and "sha256" in record[key] and not _binding_matches(record[key], panel_binding):
            issues.append(f"{key}_mismatch")
    if summary is not None:
        source = record.get("source") if isinstance(record.get("source"), Mapping) else record
        source = source.get("source_bindings", source.get("source", source)) if isinstance(source, Mapping) else {}
        if not all(key in source for key in ("raw", "trace", "receipt")):
            issues.append("source_bindings_missing")
        for key in ("raw", "trace", "receipt"):
            if key in source and not _binding_matches(source[key], summary.get("source", {}).get(key, {})):
                issues.append(f"source_{key}_mismatch")
        # Capture/release receipts use top-level ``source_panel`` for the
        # frozen shared panel.  The source annotation panel is carried as
        # ``native_source_panel`` or inside the source binding object.
        source_panel = record.get("native_source_panel")
        if source_panel is None and isinstance(record.get("source"), Mapping):
            source_panel = record["source"].get("source_panel")
        if source_panel is not None and not _binding_matches(source_panel, summary.get("source_panel", {})):
            issues.append("source_panel_mismatch")
    return issues


def _expected_cells(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, Sequence):
        raise ValueError("plan cells missing")
    result: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not cell.get("id"):
            raise ValueError("malformed plan cell")
        identifier = str(cell["id"])
        if identifier in result:
            raise ValueError(f"duplicate plan cell {identifier}")
        result[identifier] = cell
    return result


def _source_index(plan: Mapping[str, Any], shared_panel: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    summaries = {str(item["boundary_id"]): item for item in plan.get("source_summaries", [])}
    boundaries = {str(item["id"]): item for item in shared_panel.get("all_boundaries", [])}
    if set(summaries) != set(boundaries):
        raise ValueError("plan/shared panel source denominator differs")
    for identifier, summary in summaries.items():
        boundary = boundaries[identifier]
        for key in ("raw", "trace", "receipt"):
            ref = summary.get("source", {}).get(key)
            actual = boundary.get(f"{key}_path")
            if not isinstance(ref, Mapping) or not actual or not _binding_matches(ref, _file_binding(Path(actual))):
                raise ValueError(f"source {key} binding changed for {identifier}")
    return summaries, boundaries


def _native_context(boundary: Mapping[str, Any], cell: Mapping[str, Any]) -> list[int]:
    tokens = _as_ints(boundary["native_tokens"])
    end = cell.get("prefix_end")
    if end is None:
        return []
    site = int(cell["site_offset"])
    if not 0 <= site < len(tokens) or int(tokens[site]) != int(cell["old_token_id"]):
        raise ValueError(f"wrong-position or wrong-old-token cell: {cell.get('id')}")
    edited = list(tokens[: int(end)])
    if int(cell.get("delta", 0)):
        edited[site] = int(cell["new_token_id"])
    return edited


def _validate_cell_binding(summary: Mapping[str, Any], boundary: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    phases = [phase for phase in summary.get("phases", ()) if phase.get("name") == cell.get("phase")]
    if len(phases) != 1:
        raise ValueError(f"cell phase is not uniquely bound: {cell.get('id')}")
    phase = phases[0]
    row = phase["row"]
    role = "x1" if int(cell.get("delta", 0)) == 0 else str(cell.get("role"))
    if role not in ROLE_INDEX:
        raise ValueError(f"cell role is not a coordinate role: {cell.get('id')}")
    offset = int(phase["edit_offsets"][role])
    old = BASE + int(row["values"][ROLE_INDEX[role]])
    if int(cell.get("site_offset", -1)) != offset or int(cell.get("old_token_id", -1)) != old:
        raise ValueError(f"cell role/site binding is inconsistent: {cell.get('id')}")
    delta = int(cell.get("delta", 0))
    if int(cell.get("new_token_id", -1)) != old + delta:
        raise ValueError(f"cell coordinate delta is inconsistent: {cell.get('id')}")
    expected_values = list(row["values"])
    expected_values[ROLE_INDEX[role]] += delta
    if list(cell.get("edited_values", ())) != expected_values:
        raise ValueError(f"cell edited geometry is inconsistent: {cell.get('id')}")
    native = _as_ints(boundary["native_tokens"])
    if offset >= len(native) or native[offset] != old:
        raise ValueError(f"cell source token changed: {cell.get('id')}")


def _validate_receipt_cell(record: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    expected_id = str(cell["id"])
    if isinstance(record.get("cell"), Mapping) and str(record["cell"].get("id", record["cell"].get("cell_id", ""))) != expected_id:
        raise ValueError(f"receipt cell identity differs: {expected_id}")
    if isinstance(record.get("cell_ids"), Sequence) and not isinstance(record.get("cell_ids"), (str, bytes, bytearray)) and expected_id not in {str(value) for value in record["cell_ids"]}:
        raise ValueError(f"receipt cell list does not contain {expected_id}")
    expected_variant = "native" if int(cell.get("delta", 0)) == 0 else f"{cell.get('role')}{int(cell.get('delta')):+d}"
    if record.get("variant") is not None and str(record["variant"]) != expected_variant:
        raise ValueError(f"receipt variant differs for {expected_id}")
    mutation = record.get("mutation")
    if isinstance(mutation, Mapping):
        for key in ("offset", "old_token_id", "new_token_id", "delta"):
            if key in mutation and int(mutation[key]) != int(cell[{"offset": "site_offset", "old_token_id": "old_token_id", "new_token_id": "new_token_id", "delta": "delta"}[key]]):
                raise ValueError(f"receipt mutation differs for {expected_id}")
        if "role" in mutation and int(cell.get("delta", 0)) and str(mutation["role"]) != str(cell.get("role")):
            raise ValueError(f"receipt role differs for {expected_id}")


def _validate_receipt_prefix(record: Mapping[str, Any], expected: Sequence[int], cell: Mapping[str, Any]) -> None:
    prefix = record.get("prefix")
    if not isinstance(prefix, Mapping) or "target_prefix_token_ids" not in prefix:
        return
    actual = _as_ints(prefix["target_prefix_token_ids"])
    if actual != list(expected):
        raise ValueError(f"receipt prefix differs for {cell.get('id')}")
    if "consumed_prefix_length" in prefix and int(prefix["consumed_prefix_length"]) != len(expected):
        raise ValueError(f"receipt prefix length differs for {cell.get('id')}")


def _geometry_order(summary: Mapping[str, Any], boundary: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    phase = next(phase for phase in summary["phases"] if phase["name"] == cell["phase"])
    row = dict(phase["row"])
    row_index = int(row["index"])
    profile = {
        int(item["row_index"]): item["row"]
        for item in summary.get("profile_rows", ())
        if item.get("status") == "available" and isinstance(item.get("row"), Mapping)
    }
    previous = profile.get(row_index - 1)
    next_row = profile.get(row_index + 1)
    if previous is None and isinstance(boundary.get("previous_row"), Mapping) and int(boundary["previous_row"].get("index", -2)) == row_index - 1:
        previous = boundary["previous_row"]
    if next_row is None and isinstance(boundary.get("next_row"), Mapping) and int(boundary["next_row"].get("index", -2)) == row_index + 1:
        next_row = boundary["next_row"]
    values = [int(value) for value in row["values"]]
    edited = [int(value) for value in cell["edited_values"]]
    native_order = order_stratum(values, previous, next_row)
    edited_order = order_stratum(edited, previous, next_row)
    return {
        "native_geometry_valid": bool(row.get("valid")),
        "edited_geometry_valid": edited[0] < edited[2] and edited[1] < edited[3],
        "geometry_validity_changed": bool(row.get("valid")) != (edited[0] < edited[2] and edited[1] < edited[3]),
        "native_order_stratum": native_order,
        "edited_order_stratum": edited_order,
        "ordering_changed": native_order != edited_order,
        "previous_row_index": int(previous["index"]) if isinstance(previous, Mapping) else None,
        "next_row_index": int(next_row["index"]) if isinstance(next_row, Mapping) else None,
        "role": cell.get("role"),
        "delta": int(cell.get("delta", 0)),
        "site_offset": int(cell["site_offset"]),
    }


def _pair_first_free(control: Mapping[str, Any] | None, variant: Mapping[str, Any] | None) -> dict[str, Any]:
    def choice(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        item = value.get("first_free_metrics", value.get("first_free", value.get("first_free_choice")))
        if not isinstance(item, Mapping):
            return None
        token = item.get("chosen_token_id", item.get("token_id", item.get("choice", item.get("winner_token_id"))))
        margin = item.get("margin", item.get("top1_margin"))
        return {"token_id": int(token) if token is not None else None, "margin": float(margin) if margin is not None else None,
                "position": item.get("position", item.get("generated_step", 0)), "role": item.get("role"),
                "top2": item.get("top2", item.get("top_competitors"))}
    left, right = choice(control), choice(variant)
    return {
        "control": left,
        "variant": right,
        "choice_changed": left is not None and right is not None and left["token_id"] != right["token_id"],
        "margin_delta": (right["margin"] - left["margin"] if left and right and left["margin"] is not None and right["margin"] is not None else None),
        "status": "present" if left and right else "missing",
    }


def _profile_metrics(record: Mapping[str, Any], windows: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in record.get("rows", ()):
        if not isinstance(item, Mapping):
            continue
        if item.get("status") != "captured":
            rows.append({"row_index": item.get("row_index"), "status": item.get("status", "missing")})
            continue
        stage_record = {
            "boundary_id": record.get("boundary_id"),
            "row_index": item.get("row_index"),
            "registered_slots": item.get("registered_slots", ()),
            "capture": item.get("capture", {}),
        }
        rows.append({
            "row_index": item.get("row_index"),
            "status": "captured",
            "stages": _reduce_stages(stage_record, Path(str(record.get("_source_path", "."))).parent, windows),
        })
    return {
        "boundary_id": record.get("boundary_id"),
        "model": record.get("model"),
        "source": {"source": record.get("source"), "source_panel": record.get("source_panel")},
        "rows": rows,
    }


def _reduce_cell(
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    boundary: Mapping[str, Any],
    capture: Mapping[str, Any] | None,
    release: Mapping[str, Any] | None,
    control_capture: Mapping[str, Any] | None = None,
    control_release: Mapping[str, Any] | None = None,
    *,
    plan_path: Path,
    panel_binding: Mapping[str, Any],
    windows: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
    embedding_weights: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    identifier = str(cell["id"])
    if cell.get("status") != "ready":
        return {"cell_id": identifier, "status": "held", "hold": cell.get("status")}
    if capture is None or release is None:
        return {"cell_id": identifier, "status": "missing", "missing": [name for name, value in (("capture", capture), ("release", release)) if value is None]}
    issues = _binding_issues(capture, plan_path, panel_binding, summary) + _binding_issues(release, plan_path, panel_binding, summary)
    _validate_cell_binding(summary, boundary, cell)
    _validate_receipt_cell(capture, cell)
    _validate_receipt_cell(release, cell)
    context = _native_context(boundary, cell)
    _validate_receipt_prefix(release, context, cell)
    token_ids = _free_tokens(release)
    if isinstance(release.get("prefix_token_count"), int) and release["prefix_token_count"]:
        full_tokens = release.get("token_ids")
        if isinstance(full_tokens, Sequence) and len(full_tokens) >= int(release["prefix_token_count"]):
            token_ids = _as_ints(full_tokens)[int(release["prefix_token_count"]):]
    parsed = parse_rows(token_ids)
    _validate_saved_parse(release, parsed)
    free = release_metrics(token_ids, context_tokens=context, record=release)
    # Decode descriptions only when the release producer supplied them.  The
    # token parser remains the source for numerical recurrence.
    rows = free["rows"]
    saved_rows = release.get("rows")
    if isinstance(saved_rows, Sequence) and len(saved_rows) == len(rows):
        for row, saved in zip(rows, saved_rows, strict=True):
            if isinstance(saved, Mapping):
                for key in ("description", "source_values", "coord_bins_source", "source_geometry_valid"):
                    if key in saved:
                        row[key] = saved[key]
                if "coord_bins_source" in saved and "source_values" not in saved:
                    row["source_values"] = saved["coord_bins_source"]
    bank, bank_source = _annotation_bank(summary)
    known = known_matches(rows, bank) if bank_source else {
        "status": "HOLD_unbound_annotations", "bank_count": None, "matched_count": None, "matches": []
    }
    variants = _capture_variants(capture, Path(str(capture.get("_source_path", "."))).parent, windows)
    capture_variant = variants.get("changed") or variants.get("edited") or variants.get("cell") or next(iter(variants.values()), {})
    control = variants.get("native") or variants.get("control")
    if control is None and control_capture is not None:
        control_variants = _capture_variants(
            control_capture,
            Path(str(control_capture.get("_source_path", "."))).parent,
            windows,
        )
        control = control_variants.get("native") or control_variants.get("control") or next(iter(control_variants.values()), None)
    release_variants = _capture_variants(release, Path(str(release.get("_source_path", "."))).parent)
    release_variant = release_variants.get("changed") or release_variants.get("edited") or release_variants.get("cell") or next(iter(release_variants.values()), {})
    release_control = None
    if control_release is not None:
        release_control_variants = _capture_variants(control_release, Path(str(control_release.get("_source_path", "."))).parent)
        release_control = release_control_variants.get("native") or release_control_variants.get("control") or release_control_variants.get("cell") or next(iter(release_control_variants.values()), None)
    input_delta = _embedding_delta(cell, embedding_weights or {})
    if input_delta.get("status") != "present":
        issues.append("embedding_weights_missing")
    head_delta = (
        _head_delta(capture, Path(str(capture.get("_source_path", "."))).parent)
        or _head_delta_pair(capture, control_capture)
    )
    return {
        "cell_id": identifier,
        "status": "candidate" if not issues else "HOLD_binding_mismatch",
        "issues": issues,
        "source": {"boundary_id": cell["boundary_id"], "model": cell["model"], "kind": cell["kind"],
                    "phase": cell["phase"], "release_mode": cell["release_mode"], "summary_source": summary.get("source"),
                    "annotation_source": bank_source},
        "geometry_order": _geometry_order(summary, boundary, cell),
        "input": {"old_token_id": int(cell["old_token_id"]), "new_token_id": int(cell["new_token_id"]),
                  "prefix_end": int(cell["prefix_end"]), "token_distance": cell.get("token_distance"),
                  "context_token_count": len(context)},
        "free": free,
        "first_unforced_divergence": _first_divergence(control_release, release),
        "known_annotation": known,
        "first_free": _pair_first_free(release_control, release_variant),
        "capture": {
            "variants": {name: {"first_free": item.get("first_free_metrics"), "stages": item.get("stages_metrics", [])}
                         for name, item in variants.items()},
            "fixed_suffix": (
                _capture_fixed_pairs(
                    capture,
                    control_capture,
                    Path(str(capture.get("_source_path", "."))).parent,
                    Path(str(control_capture.get("_source_path", "."))).parent if control_capture is not None else None,
                )
                or _fixed_suffix_pairs(capture, Path(str(capture.get("_source_path", "."))).parent)
            ),
            "input_vector_delta": input_delta,
            "head_input_delta": head_delta,
        },
        "release_receipt": release.get("receipt", release.get("binding")),
    }


def reduce_records(
    plan_path: Path,
    capture_path: Path,
    release_path: Path | Sequence[Path],
    output_path: Path | None = None,
    windows_path: Path | None = None,
) -> dict[str, Any]:
    plan = read_plan(plan_path)
    panel_path = Path(str(plan["panel"]["path"]))
    if not _same_binding(plan["panel"], _file_binding(panel_path)):
        raise ValueError("shared panel binding changed")
    shared_panel = _read_json(panel_path)
    summaries, boundaries = _source_index(plan, shared_panel)
    windows, windows_binding = _load_windows(plan_path, windows_path)
    expected = _expected_cells(plan)
    embedding_weights, embedding_bindings = _load_embedding_weights(capture_path)
    captures, capture_files = load_receipts(capture_path)
    releases, release_files = load_receipts(release_path)
    profiles = {
        key.removeprefix("@boundary:"): _profile_metrics(dict(value, _source_path=str(source)), windows)
        for key, (value, source) in captures.items()
        if key.startswith("@boundary:")
    }
    cells = []
    counters = {"ready": 0, "held": 0, "missing": 0, "candidate": 0, "binding_hold": 0}
    control_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in expected.values():
        if cell.get("status") == "ready" and int(cell.get("delta", 0)) == 0:
            control_lookup[(str(cell["boundary_id"]), str(cell["phase"]), str(cell["release_mode"]))] = cell
    for identifier, cell in expected.items():
        summary = summaries[str(cell["boundary_id"])]
        boundary = boundaries[str(cell["boundary_id"])]
        capture = captures.get(identifier, (None, None))[0]
        release = releases.get(identifier, (None, None))[0]
        control_capture = None
        control_release = None
        control_cell = control_lookup.get((str(cell["boundary_id"]), str(cell["phase"]), str(cell["release_mode"])))
        if control_cell is not None:
            control_id = str(control_cell["id"])
            control_capture = captures.get(control_id, (None, None))[0]
            if control_capture is not None:
                control_capture = dict(control_capture)
                control_capture["_source_path"] = str(captures[control_id][1])
            control_release = releases.get(control_id, (None, None))[0]
            if control_release is not None:
                control_release = dict(control_release)
                control_release["_source_path"] = str(releases[control_id][1])
        if capture is not None:
            capture = dict(capture)
            capture["_source_path"] = str(captures[identifier][1])
        if release is not None:
            release = dict(release)
            release["_source_path"] = str(releases[identifier][1])
        try:
            reduced = _reduce_cell(cell, summary, boundary, capture, release,
                                   control_capture=control_capture,
                                   control_release=control_release,
                                   plan_path=plan_path, panel_binding=plan["panel"], windows=windows,
                                   embedding_weights=embedding_weights)
        except ValueError as exc:
            reduced = {"cell_id": identifier, "status": "HOLD_corrupt_receipt", "error": str(exc)}
        cells.append(reduced)
        if capture is not None and release is not None and cell.get("status") == "ready":
            counters["ready"] += 1
        status = str(reduced["status"])
        if status == "candidate":
            counters["candidate"] += 1
        elif status == "held":
            counters["held"] += 1
        elif status == "missing":
            counters["missing"] += 1
        elif status.startswith("HOLD_"):
            counters["binding_hold"] += 1
    result = {
        "schema": "recurrence_phase_decision.reduction.v1",
        "status": "candidate_cpu_reduction",
        "plan": _file_binding(plan_path),
        "panel": plan["panel"],
        "windows": windows_binding,
        "profiles": profiles,
        "source_bindings": {identifier: summary["source"] for identifier, summary in summaries.items()},
        "receipt_inputs": {
            "capture_files": capture_files,
            "release_files": release_files,
            "embedding_weights": embedding_bindings,
        },
        "denominators": {
            "plan_cells": len(expected),
            "plan_ready": sum(cell.get("status") == "ready" for cell in expected.values()),
            "plan_held": sum(cell.get("status") != "ready" for cell in expected.values()),
            "plan_missing_directions": len(plan.get("missing_directions", [])),
            "actual": counters,
        },
        "cells": cells,
        "interpretation": [
            "Numerical exact/near recurrence is serialized-row accounting; it is not physical-owner identity.",
            "Known matches are annotation-relative one-to-one IoU>=0.5 matches only for the bound source-panel objects.",
            "Coordinate marginals are reported per slot; no coordinate marginals are multiplied into coherent box probability.",
            "Supplied, edited, and bridge rows are excluded from free recovery credit.",
        ],
    }
    if output_path is not None:
        write_new(output_path, result)
    return result


def _selfcheck() -> None:
    native = [OBJ_START, 9, OBJ_END, BOX_START, BASE + 1, BASE + 4, BASE + 8, BASE + 9, BOX_END]
    invalid = [OBJ_START, 9, OBJ_END, BOX_START, BASE + 8, BASE + 4, BASE + 1, BASE + 9, BOX_END]
    original = {"token_ids": native + invalid + [EOS]}
    snapshot = copy.deepcopy(original)
    parsed = parse_rows(original["token_ids"])
    assert parsed["complete_rows"] == 2 and parsed["invalid_rows"] == 1
    try:
        _validate_saved_parse({"token_ids": original["token_ids"], "rows": [{"status": "valid"}]}, parsed)
    except ValueError:
        pass
    else:
        raise AssertionError("dropped-invalid fixture was accepted")
    boundary = {"native_tokens": native, "previous_row": None, "next_row": None}
    wrong = {"id": "wrong-role", "site_offset": 5, "old_token_id": BASE + 1, "new_token_id": BASE + 2,
             "prefix_end": len(native), "delta": 1}
    try:
        _native_context(boundary, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-position fixture was accepted")
    summary = {"phases": [{"name": "seed", "row": {"values": [1, 4, 8, 9]},
                            "edit_offsets": {"x1": 4, "y1": 5, "x2": 6, "y2": 7}}]}
    wrong_role = {"id": "wrong-role", "phase": "seed", "role": "x1", "delta": 1,
                  "site_offset": 7, "old_token_id": BASE + 9, "new_token_id": BASE + 10,
                  "edited_values": [2, 4, 8, 9]}
    try:
        _validate_cell_binding(summary, {"native_tokens": native + [BASE + 9]}, wrong_role)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-role fixture was accepted")
    assert original == snapshot
    assert _longest(parse_rows(native * 3)["rows"], 0) == 3
    assert release_metrics(native + [EOS])["horizons"]["8"]["status"] == "unobserved"
    print("PASS recurrence reducer parser, invalid retention, wrong-role/position rejection, no mutation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--capture", "--capture-root", dest="capture", type=Path)
    parser.add_argument("--release", "--release-root", dest="release", action="append", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.plan or not args.capture or not args.release or not args.output:
        parser.error("--plan, --capture, --release, and --output are required")
    release_paths = [path.resolve(strict=True) for path in args.release]
    result = reduce_records(
        args.plan.resolve(strict=True), args.capture.resolve(strict=True), release_paths, args.output,
        args.windows.resolve(strict=True) if args.windows else None,
    )
    print(json.dumps(result["denominators"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
