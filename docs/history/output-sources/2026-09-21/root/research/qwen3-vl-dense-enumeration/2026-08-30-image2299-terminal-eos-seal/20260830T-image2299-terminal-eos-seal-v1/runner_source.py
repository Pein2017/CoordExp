#!/usr/bin/env python3
"""Seal the 32 frozen Image2299 complete-row candidates with controlled EOS."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_progressive_final_person_release as progressive


recursive = progressive.recursive
ota = progressive.ota
base = progressive.base
token_ids_sha256 = progressive.token_ids_sha256

SCHEMA_VERSION = "image2299.terminal_eos_seal.v1"
UNIT_ID = "2026-08-30-image2299-terminal-eos-seal"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    progressive.OUTPUT_ROOT
    / "20260830T-image2299-progressive-final-person-release-v1"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "e36b9eb2a853ff778644bb43ddbc5c28d972692fbc3f1f4480e689a3db0d00ad"
SOURCE_RUN_ID = "20260830T-image2299-progressive-final-person-release-v1"
SOURCE_RUNNER_SHA256 = "854bb7fe8104bc634f89580799ddc9bf331a0848c23350bc7f0765189f2ae332"
SOURCE_BINDINGS_SHA256 = "2fbc234414c1bf52cce1a5c1585d04e78a57a2bfb0e15a392cd30f3c15907ddb"
SOURCE_COUNTS_SHA256 = "824f4130b55d29025302bd977ab21f1b3457cec8e4b33b446fa64ba421f4b310"
SOURCE_RESOURCES_SHA256 = "9c3738b991b01073f7314cc03b15d49b0d0756e4810d08bf115e8d9ec5849f38"
SOURCE_RUNTIME_SHA256 = "9c6fb6862fe09b6ce1d39442f4b9828c4919530227bd17274dd75222f09a651f"
SOURCE_NODES_SHA256 = "0c09a4cdab6a8c31679bdf5d99dc2934ec74289341b041ec17b8c2c497c1407c"
SOURCE_WARM_REPLAY_SHA256 = "ebcabb3cad6c43632f75336c51b3f902bf57cb401f7b094754bc3b2e9693f737"
LEVEL5_TASKS_SHA256 = "74f27fac81b54b077153188149d3cbddde87b023e1f560d91b316af5e56bc56d"
LEVEL5_RECORDS_SHA256 = "143e6ba9757edcf2ee4e18f22ba26b6271b2c4774c166d37f897c7b2b564cff2"
CANDIDATE_RECORD_SHA256 = {
    "level5-e93e7bdae673886b97baf117": "47146a8d504a3556e92213dcd1b675bd718a44cb74e88ff9d8ffd06f1c16fe8c",
    "level5-bebaa5920644d84e1079063e": "3b844a2c19bc0c0144a13b5ecef468d71a95576ec2ba7020714d345d20fada87",
    "level5-329b72361c89af6cddc74f34": "e90e1e14165868a42292855347ff5d51a0ec16a62f0e202ac6fc03ff1317639b",
    "level5-410b44fef8628f7cc57cef2a": "d57ac496dc6c12b0dbe6f0079104ae2d45f8fae859579b69093c9579cbc0bc5f",
    "level5-0af98466b7d52229b5565999": "bcf52149db768cacf56dc30d2cc54d8b1e75f0be2ef665ab4a11858cb73a4dae",
    "level5-47ddefc6a8c05cf8fa0c411f": "d68b7c310f69492e44136bafbe33f416d04ad8d0abcf87ea01f610d3add558a2",
    "level5-73a869f5da21921d5ea50ef7": "a90c942f65a3e1dd0dca75b9e7dd7ace04e71ad3277396d166ce5cb7503f94e8",
    "level5-f21cef316cab80248189473f": "b97df88c055470b4c28ab48ccdcda2cfa93dd9e382baac2922ab2c64776d5ea7",
    "level5-ccd23a0d17684b2b94ba22f2": "b8393aaf7f3574943b15cc9f4aa8d6d70d0fcce948557933aef88e37bd1c4c0b",
    "level5-6b406dcdb58d1817c3a788ce": "1d8e6b75a1e854d1f583db8bbd44c35e2dce09b2a6b63b327cb8fbb6f7706ae4",
    "level5-133ca24f35d4a953df13787f": "5788496a6489309b4c6c60e1f4328d1098d35b16b21a6376d8495bdd1e4a8384",
    "level5-54ca3f8c09f2ca1bbd2daa02": "3f32741609a4820e8f772372c39fe43768d58e52a6d295f32998b3755b04a7fc",
    "level5-9166d6b0351bab60dcbabd3f": "8f2f2fa3fe338351ec6dd0171d857cefdfbe7d0601d136a1801388878f4405ea",
    "level5-fba2924cdf5c4a699ba9e6dd": "b83a4e0230d2a14600529721d9c2e836a1e02bf2ad7639271f2a36dfb9a61289",
    "level5-770623314047be0a8a740f5b": "b6f0d9d8cdd8d9b5d7e0e46834859fa1686fc1d05dd2e9047927f68efa4e99ec",
    "level5-30f10291ae396ca5c7d2d3ce": "0e5069169fce60e7ce349bb78e406e68df2cedc0bd0534bc9c29ad2090a0a914",
    "level5-9fcffad474738482773b2378": "f0ec087b98a231932475d5535304eab5dc08398cc057ddd2e7d91bfac8e0870e",
    "level5-7fb3865af60fad8585a162c7": "9b7de1466fb8648e1d2d6550243eab423e088e6ce917c5a75282171ecc1719b7",
    "level5-99f42580c1116f60339d918f": "d8c91069bedffc1ab74f48c38eee968e5968835259b7d262325232f01a1ad993",
    "level5-9a5eadecb0812df50a97af9f": "b2a5093d13a0115ac5ff7238a43a6cee15366857ddab35de2888b6bc0d0f05ed",
    "level5-33ac39dde4be4e5dc5f1117a": "ddb5a6a1cc00a70e0fe579947358cfc5d0d85c3f7a697051c8c9ca6de2d1b4f5",
    "level5-760e7ffe8ea75927ae5a023e": "5b6222180b78b135fe80a11f05c8913a26ba9f1cc8e0fa8a44b35ce68ea3d25c",
    "level5-90998257234e9caa6066a76a": "87507d2447a5b9f5aa022facaab141fa5ebf37ef594718df69e0cc9c3f03f6eb",
    "level5-71d6c1b584a6b9ca5ddada18": "5edaddefced700cf0611cbbe136c33b7be53f2b17a1a9fbfa7f5a5f34de4d096",
    "level5-d73e001f5af8436fdbbf67a9": "8415a7e0b35ca5d6455a84f3964116877bb39443d72e56c9eb56018024d7ef30",
    "level5-85f229583f00cac1e53ff14f": "d2e1ea6dc92fa9c55e4f87297dcb301003b067370d20c8b014e4c67ca0e5d4b8",
    "level5-88c085b8edc4799505dad521": "86b05ee7dd5934c2a9e9435e4ff3e3f5f5faec221eeec0c727c8e563aa8219a6",
    "level5-cddd3d71069035124cff6c64": "230f24d2e7880f6754b74a5453aee9feb038c3270f31e7863e9ca620c85982c9",
    "level5-621bcbd7e6f9fdfc9a3ac4de": "532903a7b88cd14fd842d4eaa630be7534060d880a15a153da707661631633d2",
    "level5-9b71843011a1894e39588af3": "47d710ba8ab9cd5f60c70b2fd3da098ccf5fec38c63b3f027243f8dcf03cfa33",
    "level5-6f669250a91523ddcec1fb7d": "40b9e14fe9a11dd192267d3df3d45c5374dc7dd82dc996cac03475b861ce38d8",
    "level5-ec89ee36c96b7ae4f627a9b9": "23130ab0a436d3a688b1ae7bf168c25210744d1f88fb4dcf72051b422ef50579",
}

CANDIDATE_COUNT = 32
SOURCE_NODE_COUNT = 4
SOURCE_INTERVENTION_COUNT = 4
SEALED_ROW_COUNT = 41
BAD_SUFFIX_ROW_COUNTS = (1, 2)
ROW_CLOSE = progressive.ROW_CLOSE
ROW_OPEN_TOKEN = recursive.ROW_OPEN[0]
RESOURCE_BOUND = {
    "gpu_count": 1,
    "source_node_count": SOURCE_NODE_COUNT,
    "sealed_candidate_count": CANDIDATE_COUNT,
    "candidate_teacher_logit_forwards": CANDIDATE_COUNT,
    "fresh_source_replay_completions_max": SOURCE_INTERVENTION_COUNT,
    "fresh_replay_teacher_logit_forwards_max": 1,
    "final_eos_generation_count": 0,
    "generated_tokens_per_source_replay_max": base.NATURAL_MAX_TOKENS,
    "max_peak_cuda_reserved_bytes": 64 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
    "wall_time_seconds_max": 1_200,
}
TERMINAL_STATUSES = {"controlled_38_person_success", "eos_seal_exhausted"}


class TerminalEosSealHold(RuntimeError):
    """A source, seal, matcher, replay, identity, or resource contract failed."""


def _hold(message: str) -> TerminalEosSealHold:
    return TerminalEosSealHold(f"HOLD: {message}")


def _load_source_receipt() -> dict[str, Any]:
    try:
        return recursive._load_receipt(
            SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256, label="progressive source",
        )
    except BaseException as error:
        raise _hold(str(error).removeprefix("HOLD: ")) from error


def _sealed_route(record: Mapping[str, Any]) -> dict[str, Any]:
    prefix = list(map(int, record.get("forced_prefix_tokens", ())))
    row = list(map(int, record.get("forced_row_tokens", ())))
    route = [*prefix, base.EOS]
    if (
        not prefix
        or base.EOS in prefix
        or len(row) != base.ROW_TOKENS
        or tuple(row[:4]) != recursive.ROW_OPEN
        or row[-1] != ROW_CLOSE
        or int(record.get("forced_row_token_count", -1)) != base.ROW_TOKENS
        or prefix[-base.ROW_TOKENS :] != row
        or token_ids_sha256(prefix) != record.get("forced_prefix_sha256")
        or len(prefix) != SEALED_ROW_COUNT * base.ROW_TOKENS
        or (len(route) - 1) % base.ROW_TOKENS
        or route[-1] != base.EOS
        or base.EOS in route[:-1]
    ):
        raise _hold("candidate complete-row prefix cannot be sealed at row-aligned EOS")
    return {
        "forced_prefix_tokens": prefix,
        "forced_prefix_sha256": str(record["forced_prefix_sha256"]),
        "sealed_route_tokens": route,
        "sealed_route_sha256": token_ids_sha256(route),
        "eos_position": len(prefix),
    }


def _validate_source_record(
    record: Mapping[str, Any], *, task: Mapping[str, Any], node: Mapping[str, Any],
) -> dict[str, Any]:
    raw = deepcopy(dict(record))
    task_id = str(raw.get("task_id", ""))
    natural = list(map(int, raw.get("generated_token_ids", ())))
    suffix = list(map(int, raw.get("suffix_token_ids", ())))
    parent = list(map(str, raw.get("parent_owner_ids", ())))
    matched = list(map(str, raw.get("matched_owner_ids", ())))
    gate = dict(raw.get("gate", {}))
    evaluation = dict(raw.get("evaluation", {}))
    parse = dict(dict(evaluation.get("joint_gate", {})).get("parser", {}))
    seal = _sealed_route(raw)
    suffix_row_count = (len(suffix) - 1) // base.ROW_TOKENS if suffix else -1
    task_keys = (
        "task_id", "task_index", "level", "node_id", "node_index",
        "parent_route_sha256", "parent_owner_ids", "missing_person_owner_id",
        "alias_provenance", "alias_provenance_sha256", "forced_prefix_tokens",
        "forced_prefix_sha256", "forced_row_tokens", "forced_row_token_count",
        "forced_coordinate_tokens", "forced_coordinate_count", "forced_positions",
        "forced_coordinate_positions",
    )
    if (
        CANDIDATE_RECORD_SHA256.get(task_id) != base._hash(raw)
        or any(raw.get(key) != task.get(key) for key in task_keys)
        or task_id not in CANDIDATE_RECORD_SHA256
        or raw.get("level") != 5
        or raw.get("node_id") != node.get("node_id")
        or raw.get("parent_route_sha256") != node.get("route_sha256")
        or parent != list(map(str, node.get("matched_owner_ids", ())))
        or raw.get("missing_person_owner_id") != node.get("missing_person_owner_id")
        or len(parent) != 40
        or len(set(parent)) != 40
        or len(matched) != 41
        or set(matched) != set(parent) | {str(raw.get("missing_person_owner_id", ""))}
        or int(raw.get("matched_person_count", -1)) != recursive.PERSON_OWNER_COUNT
        or int(raw.get("matched_owner_count", -1)) != 41
        or int(raw.get("matched_tie_count", -1)) != 3
        or raw.get("admitted") is not False
        or gate.get("gained_owner_ids") != [raw.get("missing_person_owner_id")]
        or natural != [*seal["forced_prefix_tokens"], *suffix]
        or token_ids_sha256(natural) != raw.get("generated_token_ids_sha256")
        or token_ids_sha256(suffix) != raw.get("suffix_token_ids_sha256")
        or suffix_row_count not in BAD_SUFFIX_ROW_COUNTS
        or len(suffix) != suffix_row_count * base.ROW_TOKENS + 1
        or suffix[-1] != base.EOS
        or base.EOS in suffix[:-1]
        or any(
            tuple(suffix[index * base.ROW_TOKENS : index * base.ROW_TOKENS + 4])
            != recursive.ROW_OPEN
            or suffix[(index + 1) * base.ROW_TOKENS - 1] != ROW_CLOSE
            for index in range(suffix_row_count)
        )
        or int(parse.get("person_prediction_count", -1)) != recursive.PERSON_OWNER_COUNT
        or int(parse.get("tie_prediction_count", -1)) != 3
        or int(parse.get("valid_prediction_count", -1)) != SEALED_ROW_COUNT + suffix_row_count
        or not math.isfinite(float(raw.get("coordinate_prefix_log_probability", math.nan)))
    ):
        raise _hold(f"frozen level5 candidate binding drifted for {task_id}")
    return raw


def _validate_source_node(
    node: Mapping[str, Any], descriptions: Mapping[str, str],
) -> dict[str, Any]:
    raw = deepcopy(dict(node))
    inherited_keys = {
        "missing_person_owner_id",
        "person_family_sha256",
        "matched_tie_count",
        "source_node_sha256",
        "source_transcript_sha256",
    }
    beam_node = {key: value for key, value in raw.items() if key not in inherited_keys}
    try:
        validated = progressive._validate_source_node(beam_node, descriptions)
    except BaseException as error:
        raise _hold("progressive source node route/transcript/family binding drifted") from error
    if any(raw.get(key) != validated.get(key) for key in inherited_keys):
        raise _hold("progressive source node inherited binding drifted")
    return raw


def _validate_source_receipt(
    receipt: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(receipt)
    source = dict(raw.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    levels = list(raw.get("levels", ()))
    level5 = dict(levels[-1]) if levels else {}
    tasks = [dict(item) for item in level5.get("tasks", ())]
    records = [dict(item) for item in level5.get("candidate_records", ())]
    source_nodes = list(raw.get("source_nodes", ()))
    if (
        raw.get("schema_version") != progressive.SCHEMA_VERSION
        or raw.get("unit_id") != progressive.UNIT_ID
        or raw.get("run_id") != SOURCE_RUN_ID
        or raw.get("status") != "progressive_release_exhausted"
        or not source_path.is_file()
        or source.get("sha256") != SOURCE_RUNNER_SHA256
        or base._sha256(source_path) != SOURCE_RUNNER_SHA256
        or base._hash(raw.get("bindings", {})) != SOURCE_BINDINGS_SHA256
        or base._hash(raw.get("counts", {})) != SOURCE_COUNTS_SHA256
        or base._hash(raw.get("resources", {})) != SOURCE_RESOURCES_SHA256
        or base._hash(raw.get("runtime", {})) != SOURCE_RUNTIME_SHA256
        or base._hash(source_nodes) != SOURCE_NODES_SHA256
        or base._hash(raw.get("source_warm_replay", ())) != SOURCE_WARM_REPLAY_SHA256
        or len(levels) != 4
        or [dict(item).get("level") for item in levels] != list(progressive.LEVELS)
        or level5.get("launch_status") != "completed"
        or base._hash(tasks) != LEVEL5_TASKS_SHA256
        or base._hash(records) != LEVEL5_RECORDS_SHA256
        or len(tasks) != CANDIDATE_COUNT
        or len(records) != CANDIDATE_COUNT
        or len(source_nodes) != SOURCE_NODE_COUNT
        or dict(raw.get("counts", {})).get("total")
        != {
            "fresh_replay_completions": 0,
            "progressive_warm_completions": 112,
            "source_warm_replay_completions": 16,
            "teacher_logprob_forwards": 112,
        }
        or dict(raw.get("bindings", {})).get("checkpoint_readback")
        != contract.get("checkpoint_readback")
        or dict(raw.get("bindings", {})).get("authority")
        != contract.get("authority_source")
        or dict(raw.get("bindings", {})).get("model_receipt_sha256")
        != recursive.MODEL_RECEIPT_SHA256
        or dict(raw.get("bindings", {})).get("start_surface_sha256")
        != recursive.START_SURFACE_SHA256
        or dict(raw.get("bindings", {})).get("frozen_surface_sha256")
        != recursive.FROZEN_SURFACE_SHA256
    ):
        raise _hold("progressive source binding/count/runtime/surface snapshot drifted")
    nodes = [
        _validate_source_node(item, contract["description_map"])
        for item in source_nodes
    ]
    if [node["node_id"] for node in nodes] != list(progressive.SOURCE_NODE_IDS):
        raise _hold("four progressive source node identities drifted")
    by_node = {str(node["node_id"]): node for node in nodes}
    by_task = {str(task.get("task_id", "")): task for task in tasks}
    if len(by_task) != CANDIDATE_COUNT or set(by_task) != set(CANDIDATE_RECORD_SHA256):
        raise _hold("level5 task identity set drifted")
    validated = [
        _validate_source_record(
            record,
            task=by_task[str(record.get("task_id", ""))],
            node=by_node[str(record.get("node_id", ""))],
        )
        for record in records
    ]
    coverage = {node_id: 0 for node_id in progressive.SOURCE_NODE_IDS}
    for record in validated:
        coverage[str(record["node_id"])] += 1
    if set(coverage.values()) != {8} or len({
        _sealed_route(record)["sealed_route_sha256"] for record in validated
    }) != CANDIDATE_COUNT:
        raise _hold("level5 four-node/eight-alias sealed coverage drifted")
    return {"receipt": raw, "nodes": nodes, "records": validated}


def _eos_score(logits: torch.Tensor, *, eos_position: int) -> dict[str, Any]:
    if (
        logits.ndim != 2
        or logits.shape[0] != eos_position + 1
        or min(base.EOS, ROW_OPEN_TOKEN) < 0
        or max(base.EOS, ROW_OPEN_TOKEN) >= logits.shape[1]
    ):
        raise _hold("terminal EOS teacher logits are malformed")
    vector = logits[eos_position].float()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all().item()):
        raise _hold("terminal EOS logit vector is non-finite")
    log_probs = torch.log_softmax(vector, dim=-1)
    eos_logit = float(vector[base.EOS].item())
    row_open_logit = float(vector[ROW_OPEN_TOKEN].item())
    eos_logprob = float(log_probs[base.EOS].item())
    row_open_logprob = float(log_probs[ROW_OPEN_TOKEN].item())
    values = (eos_logit, row_open_logit, eos_logprob, row_open_logprob)
    if not all(math.isfinite(value) for value in values):
        raise _hold("terminal EOS score is non-finite")
    return {
        "position": eos_position,
        "eos_token_id": base.EOS,
        "row_open_token_id": ROW_OPEN_TOKEN,
        "eos_logit": eos_logit,
        "row_open_logit": row_open_logit,
        "eos_log_probability": eos_logprob,
        "row_open_log_probability": row_open_logprob,
        "eos_vs_row_open_margin": eos_logit - row_open_logit,
        "full_logit_vector_sha256": base.full_root._tensor_sha256(vector),
    }


def _seal_gate(
    evaluation: Mapping[str, Any], *, source: Mapping[str, Any],
    node: Mapping[str, Any], descriptions: Mapping[str, str],
) -> dict[str, Any]:
    route = list(map(int, evaluation.get("generated_token_ids", ())))
    expected_route = _sealed_route(source)["sealed_route_tokens"]
    parent = set(map(str, node.get("matched_owner_ids", ())))
    missing = str(source.get("missing_person_owner_id", ""))
    generic = recursive._node_gate(
        evaluation, parent_owner_ids=node.get("matched_owner_ids", ()),
    )
    owners = set(map(str, generic.get("matched_owner_ids", ())))
    people = sorted(owner for owner in owners if descriptions.get(owner) == "person")
    ties = sorted(owner for owner in owners if descriptions.get(owner) == "tie")
    debt = dict(generic.get("debt", {}))
    extra = {
        "generic_strict_gate": generic.get("passed") is not True or bool(debt),
        "node_parent_binding": source.get("node_id") != node.get("node_id")
        or source.get("parent_route_sha256") != node.get("route_sha256")
        or list(map(str, source.get("parent_owner_ids", ())))
        != list(map(str, node.get("matched_owner_ids", ()))),
        "route_identity": route != expected_route,
        "owner_identity": owners != parent | {missing},
        "parent_loss": not parent.issubset(owners),
        "not_all_38_persons": len(people) != recursive.PERSON_OWNER_COUNT
        or set(people)
        != {owner for owner, description in descriptions.items() if description == "person"},
        "parent_tie_loss": any(
            descriptions.get(owner) == "tie" and owner not in owners for owner in parent
        ),
        "row_count": len(route) != SEALED_ROW_COUNT * base.ROW_TOKENS + 1,
    }
    debt.update({name: failed for name, failed in extra.items() if failed})
    return {
        "passed": not debt,
        "debt": debt,
        "generic_strict_gate": deepcopy(dict(generic)),
        "parent_owner_ids": sorted(parent),
        "matched_owner_ids": sorted(owners),
        "matched_owner_count": len(owners),
        "matched_person_owner_ids": people,
        "matched_person_count": len(people),
        "matched_tie_owner_ids": ties,
        "matched_tie_count": len(ties),
        "gained_owner_ids": sorted(owners - parent),
    }


def _evaluate_seal(
    *, source: Mapping[str, Any], node: Mapping[str, Any], model: Any,
    tokenizer: Any, native_inputs: Mapping[str, Any], contract: Mapping[str, Any],
    raw_example: Any,
) -> dict[str, Any]:
    if (
        source.get("node_id") != node.get("node_id")
        or source.get("parent_route_sha256") != node.get("route_sha256")
        or source.get("parent_owner_ids") != node.get("matched_owner_ids")
    ):
        raise _hold("sealed candidate lost its node-specific production matcher parent")
    seal = _sealed_route(source)
    with torch.inference_mode():
        logits = base.full_root._teacher_forced_route_logits(
            model=model,
            native_inputs=native_inputs,
            route_tokens=seal["sealed_route_tokens"],
            pad_token_id=int(tokenizer.pad_token_id),
        )
    score = _eos_score(logits, eos_position=seal["eos_position"])
    evaluation = recursive.manifold._match_evaluation(
        tokenizer=tokenizer,
        tokens=seal["sealed_route_tokens"],
        target=contract["target"],
        witness_tokens=seal["sealed_route_tokens"],
        raw_example=raw_example,
        parent_owners=node["matched_owner_ids"],
        label=f"terminal-eos-seal-{source['task_id']}",
    )
    if list(map(int, evaluation.get("generated_token_ids", ()))) != seal["sealed_route_tokens"]:
        raise _hold("production evaluator route differs from controlled EOS seal")
    gate = _seal_gate(
        evaluation,
        source=source,
        node=node,
        descriptions=contract["description_map"],
    )
    return {
        "candidate_id": f"seal-{source['task_id']}",
        "source_task_id": str(source["task_id"]),
        "source_task_index": int(source["task_index"]),
        "source_candidate_record_sha256": CANDIDATE_RECORD_SHA256[str(source["task_id"])],
        "node_id": str(source["node_id"]),
        "parent_route_sha256": str(source["parent_route_sha256"]),
        "parent_owner_ids": list(map(str, source["parent_owner_ids"])),
        "missing_person_owner_id": str(source["missing_person_owner_id"]),
        "alias_provenance": deepcopy(list(source["alias_provenance"])),
        "alias_provenance_sha256": str(source["alias_provenance_sha256"]),
        "forced_row_tokens": list(map(int, source["forced_row_tokens"])),
        "forced_row_token_count": int(source["forced_row_token_count"]),
        "forced_coordinate_tokens": list(map(int, source["forced_coordinate_tokens"])),
        "forced_coordinate_positions": list(map(int, source["forced_coordinate_positions"])),
        "forced_prefix_tokens": seal["forced_prefix_tokens"],
        "forced_prefix_sha256": seal["forced_prefix_sha256"],
        "sealed_route_tokens": seal["sealed_route_tokens"],
        "sealed_route_sha256": seal["sealed_route_sha256"],
        "coordinate_prefix_log_probability": float(source["coordinate_prefix_log_probability"]),
        "terminal_eos_score": score,
        "matched_owner_ids": gate["matched_owner_ids"],
        "matched_owner_count": gate["matched_owner_count"],
        "matched_person_owner_ids": gate["matched_person_owner_ids"],
        "matched_person_count": gate["matched_person_count"],
        "matched_tie_owner_ids": gate["matched_tie_owner_ids"],
        "matched_tie_count": gate["matched_tie_count"],
        "admitted": bool(gate["passed"]),
        "gate": gate,
        "evaluation": deepcopy(dict(evaluation)),
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(candidate["matched_owner_count"]),
        -float(candidate["coordinate_prefix_log_probability"]),
        -float(dict(candidate["terminal_eos_score"])["eos_vs_row_open_margin"]),
        str(candidate["sealed_route_sha256"]),
        str(candidate["node_id"]),
        str(candidate["source_task_id"]),
    )


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    admitted = [deepcopy(dict(item)) for item in candidates if item.get("admitted") is True]
    if not admitted:
        raise _hold("cannot select from an empty admitted EOS-seal set")
    if any(
        int(item.get("matched_person_count", -1)) != recursive.PERSON_OWNER_COUNT
        or dict(dict(item.get("gate", {})).get("debt", {}))
        for item in admitted
    ):
        raise _hold("admitted EOS seal escaped the 38-person zero-debt gate")
    return sorted(admitted, key=_candidate_sort_key)[0]


def _terminal_status(candidates: Sequence[Mapping[str, Any]]) -> str:
    return (
        "controlled_38_person_success"
        if any(item.get("admitted") is True for item in candidates)
        else "eos_seal_exhausted"
    )


def _verify_selected_replay(
    selected: Mapping[str, Any], replayed: Mapping[str, Any],
) -> None:
    exact_keys = (
        "candidate_id", "source_task_id", "source_task_index",
        "source_candidate_record_sha256", "node_id", "parent_route_sha256",
        "parent_owner_ids", "missing_person_owner_id", "alias_provenance",
        "alias_provenance_sha256", "forced_row_tokens", "forced_row_token_count",
        "forced_coordinate_tokens", "forced_coordinate_positions",
        "forced_prefix_tokens", "forced_prefix_sha256", "sealed_route_tokens",
        "sealed_route_sha256", "coordinate_prefix_log_probability",
        "terminal_eos_score", "matched_owner_ids", "matched_owner_count",
        "matched_person_owner_ids", "matched_person_count", "matched_tie_owner_ids",
        "matched_tie_count", "admitted", "gate", "evaluation",
    )
    if any(selected.get(key) != replayed.get(key) for key in exact_keys):
        raise _hold("fresh selected EOS-seal route/ledger/logit replay drifted")


def _fresh_replay(
    *, selected: Mapping[str, Any], source: Mapping[str, Any],
    node: Mapping[str, Any], model: Any, tokenizer: Any,
    native_inputs: Mapping[str, Any], contract: Mapping[str, Any], raw_example: Any,
) -> dict[str, Any]:
    previous_tokens = list(map(int, contract["parent_route_tokens"]))
    previous_owners = list(map(str, contract["parent_owner_ids"]))
    records: list[dict[str, Any]] = []
    for recorded in node["intervention_transcript"]:
        x1 = int(recorded["x1_token_id"])
        prefix = recursive._splice_branch_prefix(previous_tokens, x1)["tokens"]
        with torch.inference_mode():
            suffix = base.full_root._greedy_release(
                model=model,
                native_inputs=native_inputs,
                prefix=prefix,
                eos_token_id=base.EOS,
                pad_token_id=int(tokenizer.pad_token_id),
            )
        completion = recursive._forced_completion(previous_tokens, x1, suffix)
        evaluation = recursive.manifold._match_evaluation(
            tokenizer=tokenizer,
            tokens=completion["generated_token_ids"],
            target=contract["target"],
            witness_tokens=completion["generated_token_ids"],
            raw_example=raw_example,
            parent_owners=previous_owners,
            label=f"terminal-eos-source-replay-depth-{recorded['depth']}",
        )
        generic = recursive._node_gate(evaluation, parent_owner_ids=previous_owners)
        owners = list(map(str, generic["matched_owner_ids"]))
        replayed = {
            **completion,
            "matched_owner_ids": owners,
            "matched_person_count": recursive._person_count(
                owners, contract["description_map"],
            ),
            "matched_owner_count": len(owners),
            "gate_debt": dict(generic["debt"]),
        }
        try:
            recursive._verify_replay_record(
                recorded, replayed, previous_tokens=previous_tokens,
            )
        except BaseException as error:
            raise _hold("fresh source transcript replay drifted") from error
        records.append(replayed)
        previous_tokens = completion["generated_token_ids"]
        previous_owners = owners
    reconstructed = [*previous_tokens[:-1], *map(int, source["forced_row_tokens"])]
    if (
        len(records) != SOURCE_INTERVENTION_COUNT
        or token_ids_sha256(previous_tokens) != node["route_sha256"]
        or previous_owners != node["matched_owner_ids"]
        or reconstructed != list(map(int, source["forced_prefix_tokens"]))
        or [*reconstructed, base.EOS] != list(map(int, selected["sealed_route_tokens"]))
    ):
        raise _hold("fresh source lineage cannot reconstruct selected complete row/EOS")
    terminal = _evaluate_seal(
        source=source,
        node=node,
        model=model,
        tokenizer=tokenizer,
        native_inputs=native_inputs,
        contract=contract,
        raw_example=raw_example,
    )
    _verify_selected_replay(selected, terminal)
    return {
        "source_records": records,
        "source_completion_count": len(records),
        "terminal_eos_generation_count": 0,
        "terminal_teacher_logit_forward_count": 1,
        "terminal_record": terminal,
        "final_generated_token_ids": terminal["sealed_route_tokens"],
        "final_generated_token_ids_sha256": terminal["sealed_route_sha256"],
        "final_ledger": deepcopy(dict(terminal["evaluation"]).get("ledger", {})),
        "final_gate_debt": deepcopy(dict(terminal["gate"]).get("debt", {})),
    }


def _enforce_counts(counts: Mapping[str, int], *, success: bool) -> None:
    expected = {
        "candidate_teacher_logit_forwards": CANDIDATE_COUNT,
        "fresh_source_replay_completions": SOURCE_INTERVENTION_COUNT if success else 0,
        "fresh_replay_teacher_logit_forwards": 1 if success else 0,
        "terminal_eos_generation_count": 0,
        "model_loads": 1,
    }
    if dict(counts) != expected:
        raise _hold("terminal EOS-seal execution count drifted")


def _require_one_gpu(*, world_size: int, device_count: int) -> None:
    if world_size != 1 or device_count != 1:
        raise _hold("terminal EOS seal requires one visible GPU and no torchrun fan-out")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise _hold(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise _hold("runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _binding_receipt() -> dict[str, Any]:
    contract = recursive._start_contract()
    source = _validate_source_receipt(_load_source_receipt(), contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_cuda",
        "source_receipt": str(SOURCE_RECEIPT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
        "source_counts_sha256": SOURCE_COUNTS_SHA256,
        "source_resources_sha256": SOURCE_RESOURCES_SHA256,
        "source_runtime_sha256": SOURCE_RUNTIME_SHA256,
        "source_nodes_sha256": SOURCE_NODES_SHA256,
        "source_warm_replay_sha256": SOURCE_WARM_REPLAY_SHA256,
        "level5_tasks_sha256": LEVEL5_TASKS_SHA256,
        "level5_records_sha256": LEVEL5_RECORDS_SHA256,
        "candidate_record_sha256": CANDIDATE_RECORD_SHA256,
        "source_node_ids": [node["node_id"] for node in source["nodes"]],
        "candidate_count": len(source["records"]),
        "model_receipt_sha256": recursive.MODEL_RECEIPT_SHA256,
        "checkpoint": str(recursive.START_CHECKPOINT),
        "checkpoint_readback_sha256": base._hash(contract["checkpoint_readback"]),
        "start_surface_sha256": recursive.START_SURFACE_SHA256,
        "frozen_surface_sha256": recursive.FROZEN_SURFACE_SHA256,
        "prompt_token_ids_sha256": base.PROMPT_TOKEN_SHA256,
        "image_sha256": base.IMAGE_SHA256,
        "target_sha256": base.TARGET_SHA256,
        "authority_description_map_sha256": recursive.AUTHORITY_DESCRIPTION_MAP_SHA256,
        "resource_bound": RESOURCE_BOUND,
    }


def run(*, run_id: str) -> Path:
    _require_one_gpu(
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        device_count=torch.cuda.device_count(),
    )
    contract = recursive._start_contract()
    source = _validate_source_receipt(_load_source_receipt(), contract)
    nodes = source["nodes"]
    records = source["records"]
    node_by_id = {str(node["node_id"]): node for node in nodes}
    source_by_task = {str(record["task_id"]): record for record in records}
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    stage = "prepare_output"
    source_snapshot: Mapping[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    replay: dict[str, Any] | None = None
    status: str | None = None
    counts = {
        "candidate_teacher_logit_forwards": 0,
        "fresh_source_replay_completions": 0,
        "fresh_replay_teacher_logit_forwards": 0,
        "terminal_eos_generation_count": 0,
        "model_loads": 0,
    }
    try:
        source_snapshot = _prepare_output(output)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        stage = "single_frozen_model_load_and_candidate_scoring"
        setup = contract["setup"]
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("terminal EOS seal requires concrete FP32 HF backend")
            counts["model_loads"] = 1
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            names, parameters = ota._step_trainable_surface(model, r32_step=True)
            initial_surface = base.full_root._full_root_surface_snapshot(
                names, parameters,
            )[1]
            frozen_before = base._frozen_surface(model, names)
            runtime = opened.receipt.to_artifact_dict()
            if (
                token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
                or initial_surface.get("aggregate_sha256") != recursive.START_SURFACE_SHA256
                or frozen_before != recursive.FROZEN_SURFACE_SHA256
            ):
                raise _hold("single-load prompt/image/model surface identity drifted")
            for record in records:
                candidate = _evaluate_seal(
                    source=record,
                    node=node_by_id[str(record["node_id"])],
                    model=model,
                    tokenizer=tokenizer,
                    native_inputs=native_inputs,
                    contract=contract,
                    raw_example=setup["raw_example"],
                )
                candidates.append(candidate)
                counts["candidate_teacher_logit_forwards"] += 1
            status = _terminal_status(candidates)
            if status == "controlled_38_person_success":
                selected = _select_candidate(candidates)
                selected_source = source_by_task[str(selected["source_task_id"])]
                selected_node = node_by_id[str(selected["node_id"])]
                stage = "fresh_selected_source_replay_and_terminal_reconstruction"
                replay = _fresh_replay(
                    selected=selected,
                    source=selected_source,
                    node=selected_node,
                    model=model,
                    tokenizer=tokenizer,
                    native_inputs=native_inputs,
                    contract=contract,
                    raw_example=setup["raw_example"],
                )
                counts["fresh_source_replay_completions"] = SOURCE_INTERVENTION_COUNT
                counts["fresh_replay_teacher_logit_forwards"] = 1
            after_surface = base.full_root._full_root_surface_snapshot(
                names, parameters,
            )[1]
            runtime_after = opened.receipt.to_artifact_dict()
            if (
                after_surface != initial_surface
                or base._frozen_surface(model, names) != frozen_before
                or runtime_after != runtime
                or any(parameter.grad is not None for parameter in model.parameters())
            ):
                raise _hold("inference-only terminal EOS seal mutated model/runtime identity")
            if replay is not None:
                replay.update({
                    "surface": after_surface,
                    "runtime": runtime_after,
                })
                if (
                    replay["final_generated_token_ids"] != selected["sealed_route_tokens"]
                    or replay["final_generated_token_ids_sha256"]
                    != selected["sealed_route_sha256"]
                    or replay["final_ledger"]
                    != dict(selected["evaluation"]).get("ledger", {})
                    or dict(replay["final_gate_debt"])
                    or replay["surface"] != initial_surface
                    or replay["runtime"] != runtime
                ):
                    raise _hold("fresh final route/ledger/surface/runtime identity drifted")
        success = status == "controlled_38_person_success"
        _enforce_counts(counts, success=success)
        elapsed = time.monotonic() - started
        resources = {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "predeclared_bound": RESOURCE_BOUND,
        }
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or resources["peak_cuda_reserved_bytes"]
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"]
            or status not in TERMINAL_STATUSES
        ):
            raise _hold("terminal EOS-seal resource or terminal-status bound exceeded")
        final_evaluation = None if selected is None else dict(selected["evaluation"])
        final_ledger = (
            None if final_evaluation is None else dict(final_evaluation.get("ledger", {}))
        )
        final = None if selected is None else {
            "generated_token_ids": selected["sealed_route_tokens"],
            "generated_token_ids_sha256": selected["sealed_route_sha256"],
            "matched_owner_ids": selected["matched_owner_ids"],
            "matched_owner_count": selected["matched_owner_count"],
            "matched_person_owner_ids": selected["matched_person_owner_ids"],
            "matched_person_count": selected["matched_person_count"],
            "matched_tie_owner_ids": selected["matched_tie_owner_ids"],
            "matched_tie_count": selected["matched_tie_count"],
            "terminal_eos_score": selected["terminal_eos_score"],
            "gate": selected["gate"],
            "parsed_rows": list(dict(final_ledger.get("parse", {})).get("predictions", ())),
            "ledger": final_ledger,
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "status": status,
            "run_id": run_id,
            "runner_source_snapshot": source_snapshot,
            "bindings": {
                "source_receipt": str(SOURCE_RECEIPT),
                "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                "source_runner_sha256": SOURCE_RUNNER_SHA256,
                "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
                "source_counts_sha256": SOURCE_COUNTS_SHA256,
                "source_resources_sha256": SOURCE_RESOURCES_SHA256,
                "source_runtime_sha256": SOURCE_RUNTIME_SHA256,
                "source_nodes_sha256": SOURCE_NODES_SHA256,
                "source_warm_replay_sha256": SOURCE_WARM_REPLAY_SHA256,
                "level5_tasks_sha256": LEVEL5_TASKS_SHA256,
                "level5_records_sha256": LEVEL5_RECORDS_SHA256,
                "candidate_record_sha256": CANDIDATE_RECORD_SHA256,
                "model_receipt_sha256": recursive.MODEL_RECEIPT_SHA256,
                "checkpoint": str(recursive.START_CHECKPOINT),
                "checkpoint_readback": contract["checkpoint_readback"],
                "start_surface_sha256": recursive.START_SURFACE_SHA256,
                "frozen_surface_sha256": recursive.FROZEN_SURFACE_SHA256,
                "prompt_token_ids_sha256": base.PROMPT_TOKEN_SHA256,
                "image_sha256": base.IMAGE_SHA256,
                "target_sha256": base.TARGET_SHA256,
                "authority": contract["authority_source"],
                "authority_description_map_sha256": recursive.AUTHORITY_DESCRIPTION_MAP_SHA256,
            },
            "protocol": {
                "route": "candidate.forced_prefix_tokens + [EOS]",
                "candidate_count": CANDIDATE_COUNT,
                "model_loads": 1,
                "production_matcher_parent": "candidate node-specific owner set",
                "strict_gate": "generic compact parser/global matcher gate plus terminal 38-person gate",
                "selection": [
                    "greater_matched_owner_count",
                    "higher_forced_coordinate_prefix_log_probability",
                    "higher_eos_vs_row_open_margin",
                    "deterministic_route_identity",
                ],
                "owner_union": False,
                "final_eos_generation": False,
                "gradient": None,
                "optimizer": None,
                "weight_update": False,
                "checkpoint_save": False,
            },
            "source_nodes": nodes,
            "candidate_records": candidates,
            "selection": {
                "selected_candidate_id": None if selected is None else selected["candidate_id"],
                "selected_source_task_id": None if selected is None else selected["source_task_id"],
                "selected_node_id": None if selected is None else selected["node_id"],
                "candidate": selected,
            },
            "intervention_transcript": (
                [] if selected is None else [
                    *deepcopy(node_by_id[str(selected["node_id"])]["intervention_transcript"]),
                    {
                        "control": "terminal_complete_row_plus_eos",
                        "source_task_id": selected["source_task_id"],
                        "forced_prefix_tokens": selected["forced_prefix_tokens"],
                        "forced_prefix_sha256": selected["forced_prefix_sha256"],
                        "sealed_route_tokens": selected["sealed_route_tokens"],
                        "sealed_route_sha256": selected["sealed_route_sha256"],
                        "terminal_eos_score": selected["terminal_eos_score"],
                        "gate_debt": deepcopy(dict(selected["gate"])["debt"]),
                    },
                ]
            ),
            "final": final,
            "fresh_replay": replay,
            "counts": counts,
            "resources": resources,
            "surface": {"before": initial_surface, "after": after_surface},
            "runtime": {"single_load": runtime, "after_replay": runtime_after},
            "wall_time_seconds": elapsed,
            "claim_boundary": (
                "One controlled frozen-r32 Image2299 38-person trajectory only; "
                "not ordinary greedy behavior, model learning, transfer, or 8/8 tie recovery."
            ),
        }
        base._atomic_json(output / "receipt.json", receipt)
        if (output / "receipt.json").stat().st_size > RESOURCE_BOUND["output_artifact_bytes_max"]:
            raise _hold("terminal EOS-seal receipt exceeded artifact bound")
        return output
    except BaseException as error:
        if output.is_dir():
            base._atomic_json(output / "receipt.json", {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": "HOLD",
                "run_id": run_id,
                "hold_stage": stage,
                "stop_reason": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "runner_source_snapshot": source_snapshot,
                "source_receipt": str(SOURCE_RECEIPT),
                "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                "candidate_records_partial": candidates,
                "selection_partial": selected,
                "fresh_replay_partial": replay,
                "counts": counts,
                "wall_time_seconds": time.monotonic() - started,
            })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true", help="CPU-only frozen check")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id:
            raise SystemExit("--check-bindings is CPU-only and cannot be combined with --run-id")
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if not args.run_id:
        raise SystemExit("one-GPU execution requires explicit --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
