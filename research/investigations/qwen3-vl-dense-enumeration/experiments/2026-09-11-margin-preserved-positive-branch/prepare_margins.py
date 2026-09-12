#!/usr/bin/env python3
"""Prepare the frozen Stable50 literal-margin table for the C branch.

This is deliberately a CPU-only provenance/package builder.  It consumes the
accepted microscope records and the exact Stable50 reference input, and never
imports a model, tokenizer, CUDA package, or trainer implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


UNIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = UNIT_DIR.parents[5]
MICROSCOPE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-greedy-preservation-microscope"
)
BRANCH_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event"
)
STABLE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-stable50-geometric-dedup"
)
OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-margin-preserved-positive-branch/margin-preparation"
)

MICROSCOPE_INPUT = MICROSCOPE_ROOT / "preparation/input.json"
MICROSCOPE_CONSUMER = MICROSCOPE_ROOT / "execution/consumer.json"
TRAINER_INPUT = BRANCH_ROOT / "trainer-preparation/inputs.json"
MANIFEST = BRANCH_ROOT / "input-preparation/candidate_manifest.json"
STABLE_INPUT = STABLE_ROOT / "inputs.json"
TEST_PATH = UNIT_DIR / "tests/test_margin_inputs.py"

EPSILON = 0.001
EXPECTED_COUNTS = {
    "cases": 56,
    "action_tokens": 6056,
    "original_mask_positions": 6047,
    "eligible_positions": 6030,
    "original_mask_near_ties": 17,
    "invalid_geometry_tokens": 9,
}

EXPECTED_HASHES = {
    "microscope_input": "a69a2b650ccc08bc3e1c457db19b9e96af70ba2cee49aaf9cce2439ec3d6d68b",
    "microscope_consumer": "4728296deadbec32736486699858d0ff0d413121afc7355706ad22c371436466",
    "trainer_inputs": "f10e82fa7106c2c0a24f5f8612299c28e68fdfbbef1c72539fb869efe5d3a83e",
    "candidate_manifest": "2870e777965007b5b06487fbb33b8408d1a992f4c90e2b4bd1f0f564c1e4aa3b",
    "stable50_inputs": "749fee8e60bee1a8a06e5deb6f018ac6270df2dbd0c2feac7b2eb667c9f83fc4",
}

RANK_RECORD_HASHES = {
    0: "04408ec51c459002b7bde05343b155f413000fde51ad31cd46840e51394f3dfa",
    1: "b3791068fa1fe83fefa5ee5cfa84bd8ad5404b17546168e1a2bd50aaaf222ffd",
    2: "e0e3a86b76392ba09adfae36634783733b87f11cab22e765e99fe518a72a9567",
    3: "265c965b922fc56b61024518fd9b6d8b313fc873f02935dcd5536a63238e33e6",
    4: "5ac9e51b7fbbcc81c6a0af1dcf433d7f53d218be90c48b2bec635230136887e4",
    5: "d99a72f19edc64a3adbc65e4399d41bda02aeaa0bfb4235aeffc4ba4f5447cb2",
    6: "92b259db05fed893f71eeb3fd58e2232955c1c3268a0e002acbf4d32878da7a4",
    7: "81439391c70f3b41821d39e89c118f6d61411eba1c7cb1e90d9d744a65db0cfe",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def list_digest(value: Any) -> str:
    """Match the existing list-identity convention used by Stable50."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_hash(name: str, path: Path) -> str:
    require(path.is_file(), f"missing source: {path}")
    actual = file_sha256(path)
    expected = EXPECTED_HASHES.get(name)
    require(expected is not None, f"no expected hash for {name}")
    require(actual == expected, f"{name} hash mismatch: {actual} != {expected}")
    return actual


def classify_margin(*, source_is_argmax: bool, margin: float) -> str:
    """Return the frozen classification; eligibility is strict at epsilon."""
    require(math.isfinite(margin), "source margin must be finite")
    if not source_is_argmax:
        return "source_literal_not_argmax"
    if margin <= EPSILON:
        return "original_mask_near_tie"
    return "eligible"


def margin_floor(margin: float) -> float:
    require(math.isfinite(margin) and margin > EPSILON, "floor requires eligible m0")
    return min(0.1, 0.5 * margin)


def _source_bindings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate the four source documents and all eight rank files."""
    microscope_input_sha = verify_hash("microscope_input", MICROSCOPE_INPUT)
    microscope_consumer_sha = verify_hash("microscope_consumer", MICROSCOPE_CONSUMER)
    trainer_input_sha = verify_hash("trainer_inputs", TRAINER_INPUT)
    manifest_sha = verify_hash("candidate_manifest", MANIFEST)
    stable_input_sha = verify_hash("stable50_inputs", STABLE_INPUT)

    microscope = read_json(MICROSCOPE_INPUT)
    consumer = read_json(MICROSCOPE_CONSUMER)
    trainer = read_json(TRAINER_INPUT)
    manifest = read_json(MANIFEST)
    stable = read_json(STABLE_INPUT)

    require(microscope["schema"] == "greedy_preservation_microscope.inputs.v1", "wrong microscope schema")
    require(consumer["schema"] == "greedy_preservation_microscope.consumer.v1", "wrong consumer schema")
    require(trainer["schema"] == "repeat_recovery_train.inputs.v1", "wrong trainer schema")
    require(stable["schema"] == "stable50_geometric_dedup.inputs.v1", "wrong Stable50 schema")
    require(consumer["input_sha256"] == microscope_input_sha, "consumer does not bind microscope input")
    require(microscope["near_tie_epsilon"] == EPSILON, "microscope epsilon changed")
    require(microscope["counts"]["cases"] == EXPECTED_COUNTS["cases"], "unexpected microscope case count")
    require(microscope["counts"]["action_tokens"] == EXPECTED_COUNTS["action_tokens"], "unexpected microscope action count")
    require(microscope["counts"]["protected_positions"] == EXPECTED_COUNTS["original_mask_positions"], "unexpected microscope mask count")
    require(consumer["counts"]["cases"] == EXPECTED_COUNTS["cases"], "unexpected consumer case count")
    require(consumer["counts"]["action_tokens"] == EXPECTED_COUNTS["action_tokens"], "unexpected consumer action count")
    require(consumer["counts"]["protected_positions"] == EXPECTED_COUNTS["original_mask_positions"], "unexpected consumer mask count")
    require(manifest.get("immutable_candidate_manifest") is True, "candidate manifest is not immutable")
    require(trainer["manifest"]["sha256"] == manifest_sha, "trainer manifest binding changed")
    require(trainer["stable50_inputs"]["sha256"] == stable_input_sha, "trainer Stable50 binding changed")
    require(trainer["counts"]["normal_count"] == EXPECTED_COUNTS["cases"], "trainer normal count changed")
    require(trainer["counts"]["normal_action_tokens"] == EXPECTED_COUNTS["action_tokens"], "trainer action count changed")
    require(trainer["counts"]["normal_kl_positions"] == EXPECTED_COUNTS["original_mask_positions"], "trainer mask count changed")
    require(len(stable["reference_cases"]) == EXPECTED_COUNTS["cases"], "Stable50 case count changed")

    rank_bindings = []
    records: dict[str, dict[str, Any]] = {}
    for rank, expected_sha in sorted(RANK_RECORD_HASHES.items()):
        path = MICROSCOPE_ROOT / f"execution/rank-{rank}/records.jsonl"
        require(path.is_file(), f"missing rank record file: {path}")
        actual_sha = file_sha256(path)
        require(actual_sha == expected_sha, f"rank {rank} record hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(len(rows) == 7, f"rank {rank} record count changed")
        for row in rows:
            require(row["rank"] == rank, f"record rank mismatch in rank {rank}")
            key = row["key"]
            require(key not in records, f"duplicate source record: {key}")
            records[key] = row
        rank_bindings.append({"rank": rank, "path": str(path), "sha256": actual_sha, "record_count": len(rows)})

    require(len(records) == EXPECTED_COUNTS["cases"], "source record population changed")
    bindings = {
        "microscope_input": {"path": str(MICROSCOPE_INPUT), "sha256": microscope_input_sha},
        "microscope_consumer": {"path": str(MICROSCOPE_CONSUMER), "sha256": microscope_consumer_sha},
        "trainer_inputs": {"path": str(TRAINER_INPUT), "sha256": trainer_input_sha},
        "candidate_manifest": {"path": str(MANIFEST), "sha256": manifest_sha},
        "stable50_inputs": {"path": str(STABLE_INPUT), "sha256": stable_input_sha},
        "microscope_rank_records": rank_bindings,
    }
    return microscope, trainer, stable, {"records": records, "bindings": bindings}


def _image_identity(case: dict[str, Any]) -> dict[str, Any]:
    group = case["group"]
    return {
        "example_id": case["example_id"],
        "image_id": str(case["image_id"]),
        "image_path": group["image_path"],
        "image_content_sha256": group["image_content_sha256"],
        "executed_media_sha256": group["executed_media_sha256"],
        "observed_image_grid_thw": group["observed_image_grid_thw"],
        "image_width": case["image_width"],
        "image_height": case["image_height"],
    }


def _build_case(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    key = case["key"]
    require(source["key"] == key, f"source key mismatch: {key}")
    require(str(source["image_id"]) == str(case["image_id"]), f"source image mismatch: {key}")
    action_ids = case["action_ids"]
    prompt_ids = case["prompt_token_ids"]
    action_hash = case["action_ids_sha256"]
    prompt_hash = case["group"]["prompt_token_ids_sha256"]
    require(list_digest(action_ids) == action_hash, f"Stable50 action digest mismatch: {key}")
    require(list_digest(prompt_ids) == prompt_hash, f"Stable50 prompt digest mismatch: {key}")
    require(source["source_action_ids_sha256"] == action_hash, f"source action identity mismatch: {key}")
    require(source["prompt_token_ids_sha256"] == prompt_hash, f"source prompt identity mismatch: {key}")

    positions = source["positions"]
    mask = list(case["initial_layout"]["kl_positions"])
    mask_set = set(mask)
    require(source["protected_positions"] == mask, f"original mask changed: {key}")
    require(len(action_ids) == case["initial_layout"]["action_token_count"] == len(positions), f"action length mismatch: {key}")
    require([row["position"] for row in positions] == list(range(len(positions))), f"position ordering changed: {key}")

    eligible_positions: list[int] = []
    target_ids: list[int] = []
    source_margins: list[float] = []
    floors: list[float] = []
    token_categories: list[str] = []
    excluded: list[dict[str, Any]] = []
    near_ties = 0
    invalid_tokens = 0
    nonargmax = 0

    geometry_invalid = set(range(75, 84)) if str(case["image_id"]) == "360573" else set()
    for row in positions:
        position = int(row["position"])
        expected_protected = position in mask_set
        require(bool(row["protected"]) == expected_protected, f"record mask flag mismatch: {key}:{position}")
        require(int(row["source_token_id"]) == int(action_ids[position]), f"literal token mismatch: {key}:{position}")
        if not expected_protected:
            require(str(case["image_id"]) == "360573" and position in geometry_invalid, f"unexpected unmasked token: {key}:{position}")
            invalid_tokens += 1
            excluded.append({
                "position": position,
                "literal_token_id": int(row["source_token_id"]),
                "literal_token": row["source_token"],
                "token_category": row["token_category"],
                "reason": "invalid_geometry_token",
                "original_mask": False,
            })
            continue

        margin = float(row["reference_source_margin"])
        is_argmax = bool(row["reference_source_is_argmax"])
        classification = classify_margin(source_is_argmax=is_argmax, margin=margin)
        if classification == "eligible":
            eligible_positions.append(position)
            target_ids.append(int(row["source_token_id"]))
            source_margins.append(margin)
            floors.append(margin_floor(margin))
            token_categories.append(row["token_category"])
            continue

        if classification == "original_mask_near_tie":
            near_ties += 1
        elif classification == "source_literal_not_argmax":
            nonargmax += 1
        excluded.append({
            "position": position,
            "literal_token_id": int(row["source_token_id"]),
            "literal_token": row["source_token"],
            "token_category": row["token_category"],
            "reason": classification,
            "original_mask": True,
            "source_literal_is_argmax": is_argmax,
            "source_margin": margin,
            "source_top1_id": int(row["reference_top1_id"]),
        })

    require(nonargmax == 0, f"source literal non-argmax positions appeared in {key}")
    require(invalid_tokens == len(geometry_invalid), f"invalid geometry count mismatch: {key}")
    require(len(eligible_positions) + near_ties == len(mask), f"mask partition mismatch: {key}")
    require(eligible_positions, f"empty eligible set: {key}")

    image_identity = _image_identity(case)
    require(image_identity["image_id"] == str(source["image_id"]), f"image identity mismatch: {key}")
    return {
        "key": key,
        "example_id": case["example_id"],
        "image_id": str(case["image_id"]),
        "image": image_identity,
        "prompt_token_ids_sha256": prompt_hash,
        "action_ids_sha256": action_hash,
        "source_rank": int(source["rank"]),
        "source_record_schema": source["schema"],
        "original_mask_positions": mask,
        "original_mask_positions_sha256": list_digest(mask),
        "eligible_positions": eligible_positions,
        "target_ids": target_ids,
        "literal_target_ids": target_ids,
        "source_margins": source_margins,
        "floors": floors,
        "token_categories": token_categories,
        "excluded_positions": excluded,
        "counts": {
            "action_tokens": len(positions),
            "original_mask_positions": len(mask),
            "eligible_positions": len(eligible_positions),
            "original_mask_near_ties": near_ties,
            "invalid_geometry_tokens": invalid_tokens,
            "source_literal_nonargmax": nonargmax,
        },
    }


def _code_identity() -> dict[str, Any]:
    require(TEST_PATH.is_file(), f"missing package test: {TEST_PATH}")
    return {
        "producer": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
        "tests": {"path": str(TEST_PATH.resolve()), "sha256": file_sha256(TEST_PATH)},
        "runtime": {"python": sys.version.split()[0], "model_imports": False, "tokenizer_imports": False, "gpu_calls": False},
    }


def schema_fixture() -> dict[str, Any]:
    return {
        "schema": "margin_preserved_positive_branch.margin_inputs.fixture.v1",
        "threshold": {
            "epsilon": EPSILON,
            "eligible": "source_literal_is_argmax and m0 > epsilon",
            "near_tie": "source_literal_is_argmax and m0 <= epsilon",
            "floor": "min(0.1, 0.5*m0)",
            "strict_boundary": {
                "m0_equal_epsilon": {"eligible": False, "reason": "original_mask_near_tie"},
                "m0_just_above_epsilon": {"m0": 0.0010001, "eligible": True, "floor": 0.00050005},
            },
        },
        "invalid_geometry_fixture": {"image_id": "360573", "positions": list(range(75, 84)), "reason": "invalid_geometry_token"},
        "required_case_fields": [
            "key", "image_id", "prompt_token_ids_sha256", "action_ids_sha256",
            "original_mask_positions", "eligible_positions", "target_ids",
            "source_margins", "floors", "excluded_positions",
        ],
        "parallel_arrays": ["eligible_positions", "target_ids", "source_margins", "floors", "token_categories"],
        "expected_counts": EXPECTED_COUNTS,
    }


def build_packet() -> dict[str, Any]:
    microscope, trainer, stable, loaded = _source_bindings()
    records = loaded["records"]
    stable_by_key = {case["key"]: case for case in stable["reference_cases"]}
    require(len(stable_by_key) == EXPECTED_COUNTS["cases"], "duplicate Stable50 case keys")
    cases = []
    for key in sorted(stable_by_key):
        require(key in records, f"missing microscope record: {key}")
        cases.append(_build_case(stable_by_key[key], records[key]))

    totals = {
        "cases": len(cases),
        "action_tokens": sum(c["counts"]["action_tokens"] for c in cases),
        "original_mask_positions": sum(c["counts"]["original_mask_positions"] for c in cases),
        "eligible_positions": sum(c["counts"]["eligible_positions"] for c in cases),
        "original_mask_near_ties": sum(c["counts"]["original_mask_near_ties"] for c in cases),
        "invalid_geometry_tokens": sum(c["counts"]["invalid_geometry_tokens"] for c in cases),
        "source_literal_nonargmax": sum(c["counts"]["source_literal_nonargmax"] for c in cases),
    }
    require(
        {key: totals[key] for key in EXPECTED_COUNTS} == EXPECTED_COUNTS
        and totals["source_literal_nonargmax"] == 0,
        f"unexpected aggregate counts: {totals}",
    )
    # These aliases are the already-frozen Sol-high consumer contract.  The
    # descriptive names above remain the canonical human-facing totals.
    totals.update(
        normal_count=totals["cases"],
        normal_action_tokens=totals["action_tokens"],
        normal_kl_positions=totals["original_mask_positions"],
        eligible_margin_positions=totals["eligible_positions"],
        retained_kl_only_positions=totals["original_mask_near_ties"],
    )
    packet = {
        "schema": "margin_preserved_positive_branch.margin_inputs.v1",
        "status": "prepared_no_model_execution",
        "candidate_only": True,
        "claim_boundary": {
            "selection": "accepted Stable50 literal source margin only",
            "not_selected_by": ["A flips", "GT", "owner loss", "natural endpoint outputs"],
            "no_model_execution": True,
        },
        "threshold": {
            "epsilon": EPSILON,
            "eligible_condition": "reference_source_is_argmax == true and m0 > 0.001",
            "near_tie_condition": "reference_source_is_argmax == true and m0 <= 0.001",
            "floor_formula": "tau=min(0.1,0.5*m0)",
        },
        "source_bindings": loaded["bindings"],
        "trainer_input_identity": {
            "path": str(TRAINER_INPUT),
            "sha256": EXPECTED_HASHES["trainer_inputs"],
            "manifest_sha256": EXPECTED_HASHES["candidate_manifest"],
            "stable50_inputs_sha256": EXPECTED_HASHES["stable50_inputs"],
        },
        "code_identity": _code_identity(),
        "counts": totals,
        "excluded_reason_counts": {
            "original_mask_near_tie": totals["original_mask_near_ties"],
            "invalid_geometry_token": totals["invalid_geometry_tokens"],
            "source_literal_not_argmax": totals["source_literal_nonargmax"],
        },
        "cases": cases,
        "schema_fixture": schema_fixture(),
        "cold_readback": {
            "required": True,
            "method": "reload inputs.json and verify content_sha256, schema, counts, and parallel-array lengths",
            "model_execution": False,
        },
    }
    packet_without_digest = dict(packet)
    packet["content_sha256"] = hashlib.sha256(canonical_bytes(packet_without_digest)).hexdigest()
    return packet


def _write_immutable(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        require(path.read_bytes() == encoded, f"immutable output differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)


def _verify_packet(packet_path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    reloaded = read_json(packet_path)
    require(reloaded["schema"] == packet["schema"], "cold schema mismatch")
    digest = reloaded["content_sha256"]
    body = dict(reloaded)
    del body["content_sha256"]
    require(hashlib.sha256(canonical_bytes(body)).hexdigest() == digest, "cold content digest mismatch")
    require(reloaded["counts"] == packet["counts"], "cold count mismatch")
    for case in reloaded["cases"]:
        lengths = [len(case[name]) for name in ("eligible_positions", "target_ids", "literal_target_ids", "source_margins", "floors", "token_categories")]
        require(len(set(lengths)) == 1, f"cold parallel-array mismatch: {case['key']}")
    return {
        "performed": True,
        "schema": reloaded["schema"],
        "content_sha256_verified": True,
        "counts_verified": True,
        "case_count": len(reloaded["cases"]),
    }


def write_outputs(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    packet = build_packet()
    output_root.mkdir(parents=True, exist_ok=True)
    packet_path = output_root / "inputs.json"
    fixture_path = output_root / "schema-fixture.json"
    _write_immutable(packet_path, packet)
    _write_immutable(fixture_path, schema_fixture())
    cold = _verify_packet(packet_path, packet)
    receipt = {
        "schema": "margin_preserved_positive_branch.margin_preparation_receipt.v1",
        "status": "prepared_no_model_execution",
        "packet": {"path": str(packet_path), "sha256": file_sha256(packet_path), "content_sha256": packet["content_sha256"]},
        "schema_fixture": {"path": str(fixture_path), "sha256": file_sha256(fixture_path)},
        "code_identity": packet["code_identity"],
        "source_bindings": packet["source_bindings"],
        "counts": packet["counts"],
        "cold_readback": cold,
        "command": f"python {Path(__file__).resolve()}",
    }
    _write_immutable(output_root / "preparation-receipt.json", receipt)
    return {"packet": packet, "receipt": receipt}


def main() -> int:
    output_root = OUTPUT_ROOT
    if len(sys.argv) == 3 and sys.argv[1] == "--output-dir":
        output_root = Path(sys.argv[2]).resolve()
    elif len(sys.argv) != 1:
        raise SystemExit("usage: prepare_margins.py [--output-dir PATH]")
    result = write_outputs(output_root)
    print(json.dumps({"packet": str(output_root / "inputs.json"), "packet_sha256": result["receipt"]["packet"]["sha256"], "counts": result["packet"]["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
