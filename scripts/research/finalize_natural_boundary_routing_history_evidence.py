#!/usr/bin/env python3
"""Finalize natural-boundary endpoint evidence without rewriting old bundles.

This finalizer is intentionally CPU-only.  It validates the new explicit
``admission_mode`` contract, keeps a mechanically valid physical-owner
unmatched outcome as scientific ``unmatched``, and computes two different
2x2 utilities: a descriptive matched-neutral arithmetic and the preregistered
``net_charged = |G| - |L| - unmatched_rows``.  A source-specific crossover is
qualified only when every primary cell has a strict source-specific match.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_routing_history_evidence.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt"
ADMISSION_MODES = ("pre_opener_natural", "post_opener_seeded")
PRIMARY_CELLS = ("Y00", "Y10", "Y01", "Y11")


class EvidenceContractError(ValueError):
    """Raised when an endpoint/finalization receipt is not interpretable."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    try:
        return sha256_bytes(Path(path).expanduser().resolve(strict=True).read_bytes())
    except OSError as exc:
        raise EvidenceContractError(f"cannot hash {path}: {exc}") from exc


def document_self_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _read_jsonish(source: str | Path | Mapping[str, Any] | Sequence[Any], label: str) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        try:
            if path.suffix.lower() == ".jsonl":
                value = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            else:
                value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceContractError(f"{label} is not valid JSON/JSONL: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw)}
    if isinstance(source, Mapping):
        value: Any = dict(source)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        value = list(source)
    else:
        raise TypeError(f"{label} must be a path, mapping, or sequence")
    return value, {"inline": True, "sha256": sha256_json(value)}


def _text(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise EvidenceContractError(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceContractError(f"{label} must be a JSON boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise EvidenceContractError(f"{label} must be finite numeric")
    return float(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceContractError(f"{label} must be a non-negative integer")
    return value


def _hash(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise EvidenceContractError(f"{label} must be a lowercase SHA-256")
    return text


def _nested_endpoint(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the common ``endpoint``/``release_receipt`` wrapper once."""

    result = dict(receipt)
    for key in ("endpoint", "release_receipt", "endpoint_receipt", "row_endpoint"):
        nested = receipt.get(key)
        if isinstance(nested, Mapping):
            for child_key, child_value in nested.items():
                result.setdefault(child_key, child_value)
    return result


def _initial_prefix_last(receipt: Mapping[str, Any]) -> int | None:
    for key in ("initial_prefix_last_token_id", "initial_prefix_last_id"):
        if key in receipt:
            value = receipt[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvidenceContractError(f"{key} must be a non-negative integer")
            return value
    for container_key in ("prefix", "initial_prefix", "model_input"):
        container = receipt.get(container_key)
        if isinstance(container, Mapping):
            for key in ("initial_prefix_last_token_id", "last_token_id"):
                if key in container:
                    value = container[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise EvidenceContractError(f"{container_key}.{key} must be a non-negative integer")
                    return value
            for key in ("initial_prefix_token_ids", "token_ids", "input_ids"):
                values = container.get(key)
                if isinstance(values, list) and values:
                    value = values[-1]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise EvidenceContractError(f"{container_key}.{key} has an invalid last token")
                    return value
    for key in ("initial_prefix_token_ids", "prefix_token_ids"):
        values = receipt.get(key)
        if isinstance(values, list) and values:
            value = values[-1]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvidenceContractError(f"{key} has an invalid last token")
            return value
    return None


def _first_generated_token(receipt: Mapping[str, Any]) -> int | None:
    for key in ("first_generated_token_id", "first_token_id", "first_generated_id"):
        if key in receipt:
            value = receipt[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvidenceContractError(f"{key} must be a non-negative integer")
            return value
    values = receipt.get("generated_token_ids")
    if isinstance(values, list) and values:
        value = values[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceContractError("generated_token_ids[0] is invalid")
        return value
    rows = receipt.get("tokens")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        value = rows[0].get("token_id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceContractError("tokens[0].token_id is invalid")
        return value
    return None


def validate_admission_mode(receipt: Mapping[str, Any], *, context: str = "endpoint") -> dict[str, Any]:
    """Validate pre/post-opener mode and return normalized admission fields.

    ``natural`` is intentionally not accepted as a mode.  It is a legacy
    boolean that old seeded rows used, so it can never substitute for the
    explicit cross-field receipt below.
    """

    row = _nested_endpoint(receipt)
    mode = row.get("admission_mode")
    if mode not in ADMISSION_MODES:
        if mode is None and "natural" in row:
            raise EvidenceContractError(f"{context} lacks explicit admission_mode; legacy natural flag is not a mode")
        raise EvidenceContractError(f"{context}.admission_mode must be pre_opener_natural or post_opener_seeded")
    opener_value = row.get("opener_token_id", row.get("row_opener_token_id"))
    if opener_value is None:
        raise EvidenceContractError(f"{context} lacks explicit opener_token_id")
    if isinstance(opener_value, bool) or not isinstance(opener_value, int) or opener_value < 0:
        raise EvidenceContractError(f"{context}.opener_token_id must be a non-negative integer")
    initial_last = _initial_prefix_last(row)
    if initial_last is None:
        raise EvidenceContractError(f"{context} lacks initial_prefix_last_token_id")
    injected = row.get("opener_injected")
    generated = row.get("opener_generated_by_model")
    if not isinstance(injected, bool):
        raise EvidenceContractError(f"{context}.opener_injected must be a JSON boolean")
    if not isinstance(generated, bool):
        raise EvidenceContractError(f"{context}.opener_generated_by_model must be a JSON boolean")
    first_generated = _first_generated_token(row)
    if first_generated is None:
        raise EvidenceContractError(f"{context} lacks first generated token identity")
    seed = row.get("seed_provenance", row.get("opener_seed_provenance"))
    if mode == "pre_opener_natural":
        if initial_last == opener_value:
            raise EvidenceContractError(f"{context} pre_opener_natural prefix already ends with opener")
        if injected:
            raise EvidenceContractError(f"{context} pre_opener_natural cannot inject opener")
        if generated is not (first_generated == opener_value):
            raise EvidenceContractError(f"{context} opener_generated_by_model disagrees with first generated token")
        if seed not in (None, False, ""):
            raise EvidenceContractError(f"{context} pre_opener_natural carries seeded opener provenance")
    else:
        if not injected:
            raise EvidenceContractError(f"{context} post_opener_seeded requires opener_injected=true")
        if initial_last != opener_value:
            raise EvidenceContractError(f"{context} post_opener_seeded prefix does not end with supplied opener")
        if generated:
            raise EvidenceContractError(f"{context} post_opener_seeded opener cannot be model-generated")
        if seed in (None, False, "", {}):
            raise EvidenceContractError(f"{context} post_opener_seeded lacks seed provenance")
    return {
        "admission_mode": mode,
        "opener_token_id": int(opener_value),
        "initial_prefix_last_token_id": int(initial_last),
        "opener_injected": injected,
        "first_generated_token_id": int(first_generated),
        "opener_generated_by_model": generated,
        "seed_provenance_present": seed not in (None, False, "", {}),
    }


def _list_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise EvidenceContractError(f"{label} must be an array")
    result = [_text(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise EvidenceContractError(f"{label} contains duplicate owner IDs")
    return result


def _owner_bookkeeping(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("owner_bookkeeping")
    if not isinstance(raw, Mapping):
        raw = row.get("owner_utility", row.get("owner_delta"))
    raw = raw if isinstance(raw, Mapping) else {}
    gained = _list_ids(raw.get("G", row.get("G", [])), "owner_bookkeeping.G")
    retained = _list_ids(raw.get("K", row.get("K", [])), "owner_bookkeeping.K")
    lost = _list_ids(raw.get("L", row.get("L", [])), "owner_bookkeeping.L")
    if set(gained) & set(retained) or set(gained) & set(lost) or set(retained) & set(lost):
        raise EvidenceContractError("owner_bookkeeping G/K/L must be disjoint")
    observed_net = raw.get("net", row.get("net"))
    net = len(gained) - len(lost) if observed_net is None else _finite(observed_net, "owner_bookkeeping.net")
    if not math.isclose(net, len(gained) - len(lost), rel_tol=0.0, abs_tol=0.0):
        raise EvidenceContractError("owner_bookkeeping.net disagrees with G/L")
    parse = raw.get("parse")
    parse = parse if isinstance(parse, Mapping) else {}
    unmatched_rows = parse.get("unmatched_rows", row.get("unmatched_rows", 0))
    duplicate_rows = parse.get("duplicate_rows", row.get("duplicate_rows", 0))
    unmatched_count = _nonnegative_int(unmatched_rows, "owner_bookkeeping.parse.unmatched_rows")
    duplicate_count = _nonnegative_int(duplicate_rows, "owner_bookkeeping.parse.duplicate_rows")
    if row.get("owner_match", {}).get("status") == "unmatched" if isinstance(row.get("owner_match"), Mapping) else False:
        unmatched_count = max(1, unmatched_count)
    if row.get("duplicate") is True:
        duplicate_count = max(1, duplicate_count)
    return {
        "G": gained,
        "K": retained,
        "L": lost,
        "net": float(net),
        "G_count": len(gained),
        "K_count": len(retained),
        "L_count": len(lost),
        "unmatched_rows": unmatched_count,
        "duplicate_rows": duplicate_count,
    }


def classify_endpoint(receipt: Mapping[str, Any], *, context: str = "endpoint") -> dict[str, Any]:
    """Validate one endpoint and separate mechanical validity from science."""

    row = _nested_endpoint(receipt)
    try:
        admission = validate_admission_mode(row, context=context)
    except EvidenceContractError as exc:
        return {
            "mechanically_valid": False,
            "technical_status": "technical_invalid",
            "scientific_status": "indeterminate",
            "reason": str(exc),
            "admission": None,
            "owner_utility": None,
        }
    try:
        if row.get("mechanically_valid") is False or row.get("technical_valid") is False:
            raise EvidenceContractError(str(row.get("technical_invalid_reason") or "receipt marks mechanical invalidity"))
        native = row.get("native_parse", row.get("parser"))
        if isinstance(native, Mapping):
            if native.get("valid") is not True or native.get("parse_status") not in (None, "accepted"):
                raise EvidenceContractError("native parser receipt is not accepted")
        wrapper = row.get("wrapper_receipt", row.get("wrapper"))
        if isinstance(wrapper, Mapping) and wrapper.get("complete") is False:
            raise EvidenceContractError("wrapper receipt is incomplete")
        generation_status = row.get("generation_status")
        if generation_status is not None and generation_status not in {"complete", "incomplete"}:
            raise EvidenceContractError("generation_status is unknown")
        complete_row = row.get("complete_row")
        if complete_row is not None and not isinstance(complete_row, bool):
            raise EvidenceContractError("complete_row must be boolean")
        match = row.get("owner_match")
        match_status = None
        source_specific = False
        physical_match = False
        owner_id = None
        if isinstance(match, Mapping):
            match_status = match.get("status")
            if match_status not in {"unique", "matched", "unmatched", "ambiguous"}:
                raise EvidenceContractError("owner_match.status is unknown")
            owner_id = match.get("owner_id", match.get("matched_owner_id"))
            source_specific = match.get("source_specific") is True
            physical_match = match.get("physical_match") is True
        else:
            match_status = row.get("owner_match_status")
            if match_status is not None and match_status not in {"unique", "matched", "unmatched", "ambiguous"}:
                raise EvidenceContractError("owner_match_status is unknown")
        utility = _owner_bookkeeping(row)
        stop_reason = row.get("stop_reason")
        stop = bool(row.get("native_stop") is True or stop_reason in {"im_end", "native_stop", "eos"})
        duplicate = bool(row.get("duplicate") is True or utility["duplicate_rows"] > 0)
        invalid_token = bool(row.get("invalid_token") is True or row.get("scientific_status") == "invalid_token")
        unmatched = bool(match_status == "unmatched" or utility["unmatched_rows"] > 0)
        if unmatched:
            scientific_status = "unmatched"
        elif match_status == "ambiguous":
            scientific_status = "ambiguous"
        elif invalid_token:
            scientific_status = "invalid_token"
        elif stop:
            scientific_status = "native_stop"
        elif duplicate:
            scientific_status = "duplicate"
        elif match_status in {"unique", "matched"}:
            scientific_status = "matched"
        else:
            scientific_status = "unresolved_endpoint"
        strict_source_specific = bool(
            match_status in {"unique", "matched"}
            and source_specific
            and physical_match
            and not unmatched
            and not duplicate
            and not stop
            and not invalid_token
            and admission["admission_mode"] == "pre_opener_natural"
        )
        return {
            "mechanically_valid": True,
            "technical_status": "valid",
            "scientific_status": scientific_status,
            "reason": None,
            "admission": admission,
            "owner_id": owner_id,
            "owner_match_status": match_status,
            "source_specific_match": strict_source_specific,
            "physical_match": physical_match,
            "unmatched": unmatched,
            "duplicate": duplicate,
            "native_stop": stop,
            "owner_utility": utility,
        }
    except EvidenceContractError as exc:
        return {
            "mechanically_valid": False,
            "technical_status": "technical_invalid",
            "scientific_status": "indeterminate",
            "reason": str(exc),
            "admission": admission,
            "owner_utility": None,
        }


def _cell_input(value: Any, *, context: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        return value[0]
    return None


def _cell_summary(raw: Mapping[str, Any] | None, *, context: str) -> dict[str, Any]:
    if raw is None:
        return {
            "cell": context.rsplit(".", 1)[-1],
            "mechanically_valid": False,
            "technical_status": "missing",
            "scientific_status": "indeterminate",
            "source_specific_match": False,
            "reason": "missing primary cell",
            "owner_utility": None,
        }
    classified = classify_endpoint(raw, context=context)
    utility = classified.get("owner_utility")
    return {
        "cell": context.rsplit(".", 1)[-1],
        **{key: classified.get(key) for key in (
            "mechanically_valid",
            "technical_status",
            "scientific_status",
            "reason",
            "owner_id",
            "owner_match_status",
            "source_specific_match",
            "physical_match",
            "unmatched",
            "duplicate",
            "native_stop",
            "admission",
        )},
        "owner_utility": utility,
        "raw_sha256": sha256_json(raw),
    }


def _tau(values: Mapping[str, float]) -> float:
    return (values["Y11"] - values["Y10"]) - (values["Y01"] - values["Y00"])


def finalize_crossover(
    cells: Mapping[str, Any],
    *,
    census: Mapping[str, Any] | str | Path | None = None,
    output: str | Path | None = None,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize a primary Y00/Y10/Y01/Y11 endpoint matrix."""

    if not isinstance(cells, Mapping):
        raise EvidenceContractError("cells must be a JSON object")
    summaries: dict[str, dict[str, Any]] = {
        cell: _cell_summary(_cell_input(cells.get(cell), context=f"crossover.{cell}"), context=f"crossover.{cell}")
        for cell in PRIMARY_CELLS
    }
    neutral: dict[str, float | None] = {}
    charged: dict[str, float | None] = {}
    unmatched: dict[str, int] = {}
    duplicates: dict[str, int] = {}
    stops: dict[str, int] = {}
    strict_cells: list[str] = []
    for cell, summary in summaries.items():
        utility = summary.get("owner_utility")
        if summary.get("mechanically_valid") is True and isinstance(utility, Mapping):
            neutral[cell] = float(utility["net"])
            unmatched[cell] = int(utility["unmatched_rows"])
            duplicates[cell] = int(utility["duplicate_rows"])
            stops[cell] = int(bool(utility.get("native_stop", False)) or bool(summary.get("native_stop")))
            charged[cell] = neutral[cell] - unmatched[cell]
        else:
            neutral[cell] = None
            charged[cell] = None
            unmatched[cell] = 0
            duplicates[cell] = 0
            stops[cell] = 0
        if summary.get("source_specific_match") is True:
            strict_cells.append(cell)
    strict_complete = len(strict_cells) == len(PRIMARY_CELLS)
    mechanically_complete = all(
        summaries[cell].get("mechanically_valid") is True
        and neutral[cell] is not None
        and charged[cell] is not None
        for cell in PRIMARY_CELLS
    )
    source_status = "qualified" if strict_complete else "unqualified"
    missing_strict = [cell for cell in PRIMARY_CELLS if cell not in strict_cells]
    neutral_numeric = {cell: float(neutral[cell]) for cell in PRIMARY_CELLS} if mechanically_complete else None
    charged_numeric = {cell: float(charged[cell]) for cell in PRIMARY_CELLS} if mechanically_complete else None
    neutral_tau = _tau(neutral_numeric) if neutral_numeric is not None else None
    charged_tau = _tau(charged_numeric) if charged_numeric is not None else None
    tau_status = "computed" if mechanically_complete else "not_computable"
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "unit_id": UNIT_ID,
        "source_specific_crossover_status": source_status,
        "source_specific_crossover_qualified": strict_complete,
        "strict_source_specific_cells": strict_cells,
        "missing_strict_source_specific_cells": missing_strict,
        "cells": summaries,
        "utility_semantics": {
            "matched_neutral": "G/K/L/net over matched-owner arithmetic; descriptive only",
            "net_charged": "|G| - |L| - unmatched_rows; unmatched is charged safety cost",
            "tau": "(Y11 - Y10) - (Y01 - Y00) for the named utility only",
        },
        "descriptive_matched_neutral": {
            "cell_net": neutral,
            "tau": neutral_tau,
            "tau_status": tau_status,
            "qualified_source_specific": False,
        },
        "descriptive_matched_net_tau": neutral_tau,
        "descriptive_matched_net_tau_status": tau_status,
        "net_charged": {
            "cell_net_charged": charged,
            "unmatched_rows": unmatched,
            "duplicate_rows": duplicates,
            "native_stop": stops,
            "tau": charged_tau,
            "tau_status": tau_status,
        },
        "net_charged_tau": charged_tau,
        "net_charged_tau_status": tau_status,
        "contrast_summary": {
            "matched_neutral": {
                "Delta_static": (
                    neutral_numeric["Y10"] - neutral_numeric["Y00"]
                    if neutral_numeric is not None
                    else None
                ),
                "Delta_dynamic": (
                    neutral_numeric["Y01"] - neutral_numeric["Y00"]
                    if neutral_numeric is not None
                    else None
                ),
                "status": tau_status,
            },
            "net_charged": {
                "Delta_static": (
                    charged_numeric["Y10"] - charged_numeric["Y00"]
                    if charged_numeric is not None
                    else None
                ),
                "Delta_dynamic": (
                    charged_numeric["Y01"] - charged_numeric["Y00"]
                    if charged_numeric is not None
                    else None
                ),
                "status": tau_status,
            },
        },
    }
    if census is not None:
        census_value, census_info = _read_jsonish(census, "census")
        if not isinstance(census_value, Mapping):
            raise EvidenceContractError("census must be a JSON object")
        if census_value.get("schema_version") != "natural_boundary_owner_admission_census.v1" or census_value.get("unit_id") != UNIT_ID:
            raise EvidenceContractError("census schema/unit identity mismatch")
        if census_value.get("self_sha256") != document_self_sha256(census_value):
            raise EvidenceContractError("census self hash mismatch")
        if not isinstance(census_value.get("rows"), list) or len(census_value["rows"]) != 784:
            raise EvidenceContractError("census must contain exactly 784 rows")
        document["census_binding"] = {"sha256": census_info["sha256"], "self_sha256": census_value["self_sha256"], "row_count": 784}
    document["self_sha256"] = document_self_sha256(document)
    result = {"evidence": document}
    if output is not None:
        _write_immutable(Path(output), canonical_json_bytes(document) + b"\n")
        result["path"] = str(Path(output).expanduser().resolve())
    if receipt_output is not None:
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "unit_id": UNIT_ID,
            "evidence_sha256": document["self_sha256"],
            "source_specific_crossover_status": source_status,
            "descriptive_matched_net_tau": document["descriptive_matched_net_tau"],
            "descriptive_matched_net_tau_status": tau_status,
            "net_charged_tau": document["net_charged_tau"],
            "net_charged_tau_status": tau_status,
        }
        receipt["self_sha256"] = document_self_sha256(receipt)
        _write_immutable(Path(receipt_output), canonical_json_bytes(receipt) + b"\n")
        result["receipt"] = receipt
    return result


def finalize_evidence(
    source: Mapping[str, Any] | Sequence[Any] | str | Path,
    *,
    census: Mapping[str, Any] | str | Path | None = None,
    output: str | Path | None = None,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize either a matrix envelope or an event list."""

    value, source_info = _read_jsonish(source, "endpoint evidence")
    if isinstance(value, Mapping) and isinstance(value.get("cells"), Mapping):
        return finalize_crossover(value["cells"], census=census, output=output, receipt_output=receipt_output)
    if isinstance(value, Mapping) and isinstance(value.get("events"), list):
        events = value["events"]
    elif isinstance(value, list):
        events = value
    elif isinstance(value, Mapping):
        events = [value]
    else:
        raise EvidenceContractError("endpoint evidence must be a mapping or list")
    summaries = [
        _cell_summary(item if isinstance(item, Mapping) else None, context=f"events[{index}]")
        for index, item in enumerate(events)
    ]
    strict_count = sum(item.get("source_specific_match") is True for item in summaries)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "unit_id": UNIT_ID,
        "source_specific_crossover_status": "unqualified",
        "event_count": len(summaries),
        "events": summaries,
        "strict_source_specific_event_count": strict_count,
        "input_sha256": source_info["sha256"],
    }
    document["self_sha256"] = document_self_sha256(document)
    result = {"evidence": document}
    if output is not None:
        _write_immutable(Path(output), canonical_json_bytes(document) + b"\n")
    if receipt_output is not None:
        receipt = {"schema_version": RECEIPT_SCHEMA_VERSION, "status": "complete", "unit_id": UNIT_ID, "evidence_sha256": document["self_sha256"], "source_specific_crossover_status": "unqualified"}
        receipt["self_sha256"] = document_self_sha256(receipt)
        _write_immutable(Path(receipt_output), canonical_json_bytes(receipt) + b"\n")
        result["receipt"] = receipt
    return result


def _write_immutable(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        if resolved.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite immutable evidence artifact: {resolved}")
        return
    resolved.write_bytes(payload)


def validate_evidence(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("unit_id") != UNIT_ID:
        raise EvidenceContractError("evidence schema/unit identity mismatch")
    if document.get("self_sha256") != document_self_sha256(document):
        raise EvidenceContractError("evidence self hash mismatch")
    if document.get("source_specific_crossover_status") not in {"qualified", "unqualified"}:
        raise EvidenceContractError("source_specific_crossover_status must be qualified or unqualified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", dest="receipt_output", type=Path)
    args = parser.parse_args(argv)
    result = finalize_evidence(args.input, census=args.census, output=args.output, receipt_output=args.receipt_output)
    print(json.dumps(result.get("receipt", result["evidence"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
