#!/usr/bin/env python3
"""Build an exact-token staircase manifest from three certified sampled routes.

This helper is intentionally offline.  It reads the frozen route manifest and
the immutable greedy/sampled rollout files, verifies their prompt and row
provenance, computes the longest common generated-token prefix, and writes a
small manifest for the earliest-branch pilot.  It never treats a sampled row
as a ground-truth label; owner ids are only frozen review evidence used to
choose the route.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_greedy_prefix_forced_owner_path import (
    BOX_END,
    hash_prefix_token_ids,
    semantic_endpoint_role,
)


SCHEMA_VERSION = "earliest_shared_prefix_branch_cases.v1"
CERTIFIED_CASES = {
    "person-5001-first-sampled-only": {
        "source_case_id": "person-5001-row3",
        "image_id": 5001,
        "sampled_seed": 21015,
        "target_owner_id": "5001:1269467",
        "sampled_donor_row_index": 5,
        "geometry_trust": "trusted_for_diagnostic",
        "reviewed_donor_owner_id": None,
        "complete_row_budget": 16,
    },
    "person-7511-first-sampled-only": {
        "source_case_id": "person-7511-row4",
        "image_id": 7511,
        "sampled_seed": 21012,
        "target_owner_id": "7511:-169",
        "sampled_donor_row_index": 4,
        "geometry_trust": "entity_only_geometry_loose",
        "reviewed_donor_owner_id": "7511:-169",
        "complete_row_budget": 16,
    },
    "wine-glass-2685-first-sampled-only": {
        "source_case_id": "wine-glass-2685-row6",
        "image_id": 2685,
        "sampled_seed": 21008,
        "target_owner_id": "2685:-78",
        "sampled_donor_row_index": 6,
        "geometry_trust": "trusted_for_diagnostic",
        "reviewed_donor_owner_id": None,
        "complete_row_budget": 16,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} must contain non-negative integer ids")
        result.append(int(token))
    return result


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the exact generated-token longest common prefix length."""

    left_ids = _ids(left, label="left generated_token_ids")
    right_ids = _ids(right, label="right generated_token_ids")
    length = 0
    for left_token, right_token in zip(left_ids, right_ids):
        if left_token != right_token:
            break
        length += 1
    return length


def row_spans(token_ids: Sequence[int], *, box_end_token_id: int = BOX_END) -> list[tuple[int, int]]:
    """Return complete-row half-open spans, refusing incomplete tails."""

    ids = _ids(token_ids, label="generated_token_ids")
    spans: list[tuple[int, int]] = []
    start = 0
    for index, token in enumerate(ids):
        if token == int(box_end_token_id):
            spans.append((start, index + 1))
            start = index + 1
    if start != len(ids):
        raise ValueError("generated token sequence ends with an incomplete row")
    return spans


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve_ref(ref: Any) -> Path:
    path = ref.get("path") if isinstance(ref, Mapping) else ref
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("artifact path is required")
    return Path(str(path)).expanduser().resolve(strict=True)


def _select_rollout(payload: Mapping[str, Any], *, image_id: int, seed: int | None) -> Mapping[str, Any]:
    records = payload.get("rollouts")
    if not isinstance(records, list):
        raise ValueError("rollout artifact must contain a rollouts list")
    selected: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or str(record.get("image_id")) != str(image_id):
            continue
        mode = str(record.get("decode_mode", "")).lower()
        if seed is None and mode in {"greedy", "native"}:
            selected.append(record)
        elif seed is not None and mode in {"sample", "sampled", "sampling"} and int(record.get("seed")) == int(seed):
            selected.append(record)
    if len(selected) != 1:
        raise ValueError(f"expected one rollout for image {image_id}, seed {seed}; found {len(selected)}")
    row = selected[0]
    prompt_ids = _ids(row.get("prompt_token_ids"), label="prompt_token_ids")
    generated_ids = _ids(row.get("generated_token_ids"), label="generated_token_ids")
    expected_prompt_hash = row.get("prompt_token_ids_sha256")
    if expected_prompt_hash != hash_prefix_token_ids(prompt_ids):
        raise ValueError("prompt token hash mismatch")
    expected_generated_hash = row.get("generated_token_ids_sha256")
    if expected_generated_hash != hash_prefix_token_ids(generated_ids):
        raise ValueError("generated token hash mismatch")
    return {**dict(row), "prompt_token_ids": prompt_ids, "generated_token_ids": generated_ids}


def _source_case(cases_payload: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    cases = cases_payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("source cases manifest must contain cases")
    for case in cases:
        if isinstance(case, Mapping) and str(case.get("case_id")) == case_id:
            return case
    raise ValueError(f"source cases manifest lacks {case_id}")


def _row_prediction(record: Mapping[str, Any], row_index: int) -> Mapping[str, Any]:
    predictions = record.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("rollout lacks parser predictions")
    rows = predictions.get("predictions")
    if not isinstance(rows, list) or row_index < 0 or row_index >= len(rows):
        raise ValueError("sampled donor row lacks parser prediction")
    row = rows[row_index]
    if not isinstance(row, Mapping):
        raise ValueError("sampled donor prediction is not an object")
    return row


def _staircase_for_row(
    sampled_ids: Sequence[int],
    row_start: int,
    row_end: int,
    lcp_length: int,
    *,
    include_level_a: bool,
    include_level_b: bool,
) -> list[dict[str, Any]]:
    row = list(map(int, sampled_ids[row_start:row_end]))
    if not row:
        raise ValueError("staircase row cannot be empty")
    endpoints: list[tuple[str, int]] = []
    # Level A starts exactly at the first divergent token and keeps the same
    # row.  The common prefix is recorded separately; forcing the common
    # tokens is equivalent but obscures where the intervention begins.
    if include_level_a and row_start == 0 and lcp_length < row_end:
        first_end = max(1, lcp_length - row_start + 1)
        endpoints.append(("level_a_first_divergence", first_end))
        for end in range(first_end + 1, len(row) + 1):
            role = semantic_endpoint_role(row[:end])
            if role in {"x1", "y1", "x2", "y2", "box_end"}:
                endpoints.append((f"level_a_{role}", end))
    # Level B always begins at the target row opener after the complete
    # sampled prefix before it has been supplied.  This does not inject the
    # target owner itself: the target row is still only partially supplied.
    if include_level_b:
        endpoints.append(("level_b_target_row_start", 1))
        for end in range(2, len(row) + 1):
            role = semantic_endpoint_role(row[:end])
            if role in {"description_end", "box_start", "x1", "y1", "x2", "y2", "box_end"}:
                endpoints.append((f"level_b_{role}", end))
    seen: set[tuple[str, int]] = set()
    output: list[dict[str, Any]] = []
    for name, endpoint in endpoints:
        key = (name, endpoint)
        if key in seen:
            continue
        seen.add(key)
        forced = row[:endpoint]
        output.append({
            "rung_name": name,
            "level": "A" if name.startswith("level_a_") else "B",
            "row_start_token_index": int(row_start),
            "row_end_token_index": int(row_end),
            "endpoint_token_count": int(endpoint),
            "endpoint_role": semantic_endpoint_role(forced),
            "forced_row_prefix_token_ids": forced,
            "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(forced),
            "global_lcp_token_count": int(lcp_length),
        })
    return output


def build_manifest(source_manifest_path: Path, *, output_path: Path) -> dict[str, Any]:
    source_manifest = _load_json(source_manifest_path)
    greedy_path = _resolve_ref(source_manifest.get("greedy_artifact"))
    greedy_payload = _load_json(greedy_path)
    output_cases: list[dict[str, Any]] = []
    for case_id, spec in CERTIFIED_CASES.items():
        source_case = _source_case(source_manifest, str(spec["source_case_id"]))
        sampled_path = _resolve_ref(source_case.get("sampled_artifact"))
        sampled_payload = _load_json(sampled_path)
        image_id = int(spec["image_id"])
        sampled_seed = int(spec["sampled_seed"])
        greedy = _select_rollout(greedy_payload, image_id=image_id, seed=None)
        sampled = _select_rollout(sampled_payload, image_id=image_id, seed=sampled_seed)
        if greedy["prompt_token_ids"] != sampled["prompt_token_ids"]:
            raise ValueError(f"prompt mismatch for {case_id}")
        greedy_ids = greedy["generated_token_ids"]
        sampled_ids = sampled["generated_token_ids"]
        lcp_length = longest_common_prefix_length(greedy_ids, sampled_ids)
        if lcp_length >= min(len(greedy_ids), len(sampled_ids)):
            raise ValueError(f"no strict divergence for {case_id}")
        greedy_spans = row_spans(greedy_ids)
        sampled_spans = row_spans(sampled_ids)
        donor_index = int(spec["sampled_donor_row_index"])
        if donor_index >= len(sampled_spans) or donor_index >= len(greedy_spans):
            raise ValueError(f"target row is unavailable for {case_id}")
        target_start, target_end = sampled_spans[donor_index]
        prediction = _row_prediction(sampled, donor_index)
        expected_source_case = source_case.get("expected_donor")
        # Route replacement is intentional, but basic provenance must still
        # be shown in the output for downstream audit.
        donor_row_ids = sampled_ids[target_start:target_end]
        first_diff_row = next((index for index, (left, right) in enumerate(zip(greedy_ids, sampled_ids)) if left != right), None)
        first_diff_row_index = next((index for index, (start, end) in enumerate(greedy_spans) if first_diff_row is not None and start <= first_diff_row < end), None)
        first_diff_row_offset = None if first_diff_row is None or first_diff_row_index is None else first_diff_row - greedy_spans[first_diff_row_index][0]
        output_cases.append({
            "case_id": case_id,
            "source_case_id": source_case.get("case_id"),
            "image_id": image_id,
            "sampled_seed": sampled_seed,
            "greedy_artifact": {"path": str(greedy_path), "sha256": sha256_file(greedy_path)},
            "sampled_artifact": {"path": str(sampled_path), "sha256": sha256_file(sampled_path)},
            "sampled_donor_row_index": donor_index,
            "target_owner_id": spec["target_owner_id"],
            "reviewed_donor_owner_id": spec["reviewed_donor_owner_id"],
            "geometry_trust": spec["geometry_trust"],
            "complete_row_budget": int(spec["complete_row_budget"]),
            "expected_donor": {
                "description": prediction.get("description"),
                "coord_bins": prediction.get("coord_bins"),
                "raw_span_sha256": prediction.get("raw_span_sha256"),
                "source_manifest_expected_donor": expected_source_case,
            },
            "prompt_token_ids_sha256": greedy["prompt_token_ids_sha256"],
            "greedy_generated_token_count": len(greedy_ids),
            "sampled_generated_token_count": len(sampled_ids),
            "global_longest_common_prefix": {
                "token_count": lcp_length,
                "token_ids_sha256": hash_prefix_token_ids(greedy_ids[:lcp_length]),
                "first_divergence_index": lcp_length,
                "first_divergence_row_index": first_diff_row_index,
                "first_divergence_row_offset": first_diff_row_offset,
                "greedy_token": greedy_ids[lcp_length],
                "sampled_token": sampled_ids[lcp_length],
            },
            "target_row_span_in_sampled": {"start": target_start, "end": target_end, "token_count": target_end - target_start, "sha256": hash_prefix_token_ids(donor_row_ids)},
            "sampled_prefix_before_target_row_token_count": target_start,
            "staircase": [
                *_staircase_for_row(
                    sampled_ids,
                    sampled_spans[0][0],
                    sampled_spans[0][1],
                    lcp_length,
                    include_level_a=True,
                    include_level_b=False,
                ),
                *_staircase_for_row(
                    sampled_ids,
                    target_start,
                    target_end,
                    lcp_length,
                    include_level_a=False,
                    include_level_b=True,
                ),
            ],
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": "2026-07-21-earliest-shared-prefix-branch-pilot",
        "source_cases_manifest": {"path": str(source_manifest_path), "sha256": sha256_file(source_manifest_path)},
        "greedy_artifact": {"path": str(greedy_path), "sha256": sha256_file(greedy_path)},
        "total_token_budget": int(source_manifest.get("total_token_budget", 512)),
        "cases": output_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_manifest(args.source_cases.expanduser().resolve(strict=True), output_path=args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
