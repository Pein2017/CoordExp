#!/usr/bin/env python3
"""World8 beam-backtracking person search for frozen Image2299 r32."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_recursive_matcher_branch_search as recursive


base = recursive.base
ota = recursive.ota
token_ids_sha256 = recursive.token_ids_sha256

SCHEMA_VERSION = "image2299.beam_backtracking_person_search.v1"
UNIT_ID = "2026-08-30-image2299-beam-backtracking-person-search"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    recursive.OUTPUT_ROOT
    / "20260830T-image2299-recursive-matcher-branch-search-v1b"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "d91d032abce24e0abcda445d0790ba1d59b9ff092c74256ae76efcc65485a1b2"
SOURCE_RECEIPT_OBJECT_SHA256 = "9ade77a81416ae2938bf43bc68cf609b5cf793c747eb1c814372412bacb0b359"
SOURCE_BINDINGS_SHA256 = "a8ce0dccf25031186a228b302b9e13fc8486c014400b98b44e54c32c3a4c364b"
SOURCE_COUNTS_SHA256 = "59fa3ad112dc3bb70360dbc474d65a325d7725e84a98f567f656e1c177a0036e"
SOURCE_RESOURCES_SHA256 = "ef0ee65222f0fc0c6130bd361481aa7c820349df55b16eb7c4f38d51fc44a94a"
SOURCE_RUNNER_SHA256 = "0d56fe7df421bd4bf3d6a695ac612fe27dd2e53debae89b07273c47963501f1c"
SOURCE_DEPTH_ZERO_TRANSCRIPT_SHA256 = (
    "7fe233314d3746f994b6822be20fe5b239951c00536a1953f73f5b5e8d52c83b"
)
SOURCE_FINAL_ROUTE_SHA256 = "725c09599da03369e3352e1b4c703492688b915d0995a5ab3734b7649b6f45e1"
DEPTH_ZERO_X1 = 152032
DEPTH_ZERO_ROUTE_SHA256 = recursive.DEPTH_ZERO_ROUTE_SHA256[DEPTH_ZERO_X1]
DEPTH_ZERO_NODE_ID = f"depth0-{DEPTH_ZERO_X1}-{DEPTH_ZERO_ROUTE_SHA256[:16]}"
ROOT_NODE_ID = f"parent33-{recursive.CONTROL_PARENT_ROUTE_SHA256[:16]}"
SEED_X1_TOKENS = (151820, 151821, 151822, 151866, 151867, 151868)
SEED_ROUTE_SHA256 = {
    151820: "03d69bef68e479836828399f8b0d19c3951b881509cdd7a59922d93a37f849e7",
    151821: "043df1e8b62c44d5bb052051f1feffd6858e1ba05fee5be4fd2f3372b19177f1",
    151822: "86d5635d899525e6d82fa06cd32bef70a01ff5e5e212c504bf5675b9c1eb125c",
    151866: "a2c8852e178dd39b2711ec2527cf04a8abb7cf1ea5a7a0b7166e0109d4825dbe",
    151867: "df48eec88324b07f0ceb9931a5959811e6efec83a30f30cc71708a23b9adaa13",
    151868: "585030e0997dbcb2dce41f8ef62ec31e26077133de7832939e79c2b9999e2299",
}
SEED_EVALUATION_SHA256 = {
    151820: "5f5fc3e2641e6dac73b4e8cab82768eae3c79448ca15ba10c4893675fec0a552",
    151821: "30b5928cc3a35b74270a59acd06a5d45197ece99b1a78080956d656a9a7e7190",
    151822: "c0474be6b41e3f721586f5c143f5705089a027a6749b34165fd479a6c52d419c",
    151866: "1557fc46fdff4c72872b85fb067bf929f0431c03055de202e1cd6de4f32c9306",
    151867: "c1ed6c76020f766b70d35d9a6abba4648de1f61c20545c74c023e7f58e4b4fbb",
    151868: "69443f28c17aa453eaf56b7f609bdec8e66bd0f2aefe5bf6cb794fdc12bba16a",
}
SEED_OWNER_FAMILY_SHA256 = {
    151820: "68904afb9e8e7a1d1a6f65e4af3cb320f6448c78d98bad4d8f74e40274731ace",
    151821: "68904afb9e8e7a1d1a6f65e4af3cb320f6448c78d98bad4d8f74e40274731ace",
    151822: "68904afb9e8e7a1d1a6f65e4af3cb320f6448c78d98bad4d8f74e40274731ace",
    151866: "2eb4549659dfee23bd90e27120ec7bd7d8f5a0adaea51cb3e9c969d6d0b747bd",
    151867: "2eb4549659dfee23bd90e27120ec7bd7d8f5a0adaea51cb3e9c969d6d0b747bd",
    151868: "2eb4549659dfee23bd90e27120ec7bd7d8f5a0adaea51cb3e9c969d6d0b747bd",
}
SEED_GAIN = {
    151820: ("gt:2299:32", "gt:2299:35"),
    151821: ("gt:2299:32", "gt:2299:35"),
    151822: ("gt:2299:32", "gt:2299:35"),
    151866: ("gt:2299:18", "gt:2299:32"),
    151867: ("gt:2299:18", "gt:2299:32"),
    151868: ("gt:2299:18", "gt:2299:32"),
}

WORLD_SIZE = 8
BEAM_WIDTH = 8
FAMILY_CAP = 2
FIRST_EXPANSION_INTERVENTION = 3
MAX_INTERVENTIONS = 6
MAX_ACTIONS_PER_NODE = 15
MAX_WARM_COMPLETIONS = 250
MAX_REPLAY_COMPLETIONS = 6
MAX_BRANCH_LOGIT_FORWARDS = len(SEED_X1_TOKENS) + BEAM_WIDTH * (
    MAX_INTERVENTIONS - FIRST_EXPANSION_INTERVENTION
)
SEED_WARM_COMPLETIONS = len(SEED_X1_TOKENS) * 2
PERSON_OWNER_COUNT = recursive.PERSON_OWNER_COUNT
RESOURCE_BOUND = {
    "gpu_count": WORLD_SIZE,
    "beam_width": BEAM_WIDTH,
    "person_owner_family_cap": FAMILY_CAP,
    "cumulative_interventions_max": MAX_INTERVENTIONS,
    "actions_per_node_max": MAX_ACTIONS_PER_NODE,
    "warm_seed_replay_completions": SEED_WARM_COMPLETIONS,
    "warm_controlled_completions_max": MAX_WARM_COMPLETIONS,
    "cold_replay_completions_max": MAX_REPLAY_COMPLETIONS,
    "branch_logit_forwards_max": MAX_BRANCH_LOGIT_FORWARDS,
    "generated_tokens_per_decode_max": base.NATURAL_MAX_TOKENS,
    "model_forward_calls_total_max": (
        (MAX_WARM_COMPLETIONS + MAX_REPLAY_COMPLETIONS) * base.NATURAL_MAX_TOKENS
        + MAX_BRANCH_LOGIT_FORWARDS
    ),
    "max_peak_cuda_reserved_bytes_per_rank": 64 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
    "wall_time_seconds_max": 1_200,
}
TERMINAL_STATUSES = {
    "controlled_38_person_success",
    "beam_search_exhausted",
    "beam_depth_exhausted",
}


class BeamSearchHold(RuntimeError):
    """A frozen identity, beam, collective, replay, or resource contract failed."""


def _hold(message: str) -> BeamSearchHold:
    return BeamSearchHold(f"HOLD: {message}")


def _load_source_receipt() -> dict[str, Any]:
    try:
        return recursive._load_receipt(
            SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256, label="v1b recursive-search source",
        )
    except BaseException as error:
        raise _hold(str(error).removeprefix("HOLD: ")) from error


def _person_owner_ids(
    owner_ids: Sequence[str], descriptions: Mapping[str, str],
) -> list[str]:
    owners = list(map(str, owner_ids))
    if len(owners) != len(set(owners)) or not set(owners).issubset(descriptions):
        raise _hold("node owner identity is duplicated or outside authority")
    return sorted(owner for owner in owners if descriptions[owner] == "person")


def _node_sort_key(node: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ascending key for the frozen descending-value policy."""
    return (
        -int(node["matched_person_count"]),
        -int(node["matched_owner_count"]),
        int(node["cumulative_intervention_count"]),
        -float(node["cumulative_selected_x1_logit"]),
        str(node["route_sha256"]),
        str(node["node_id"]),
    )


def _validate_node(node: Mapping[str, Any], descriptions: Mapping[str, str]) -> None:
    tokens = list(map(int, node.get("route_tokens", ())))
    owners = list(map(str, node.get("matched_owner_ids", ())))
    people = _person_owner_ids(owners, descriptions)
    transcript = list(node.get("intervention_transcript", ()))
    lineage = list(map(str, node.get("lineage_node_ids", ())))
    interventions = int(node.get("cumulative_intervention_count", -1))
    if (
        not node.get("node_id")
        or token_ids_sha256(tokens) != node.get("route_sha256")
        or not tokens
        or tokens[-1] != base.EOS
        or list(map(str, node.get("matched_person_owner_ids", ()))) != people
        or int(node.get("matched_person_count", -1)) != len(people)
        or int(node.get("matched_owner_count", -1)) != len(owners)
        or len(transcript) != interventions
        or not 2 <= interventions <= MAX_INTERVENTIONS
        or len(lineage) != interventions + 1
        or lineage[0] != ROOT_NODE_ID
        or lineage[-1] != node.get("node_id")
        or lineage[-2] != node.get("parent_node_id")
        or not transcript
        or transcript[0].get("x1_token_id") != DEPTH_ZERO_X1
        or transcript[0].get("parent_route_sha256")
        != recursive.CONTROL_PARENT_ROUTE_SHA256
        or base._hash(transcript[0]) != SOURCE_DEPTH_ZERO_TRANSCRIPT_SHA256
        or transcript[-1].get("generated_token_ids_sha256") != node.get("route_sha256")
        or list(map(int, transcript[-1].get("generated_token_ids", ()))) != tokens
        or list(map(str, transcript[-1].get("matched_owner_ids", ()))) != owners
        or dict(transcript[-1].get("gate_debt", {}))
        or sum(float(item["x1_logit"]) for item in transcript)
        != float(node.get("cumulative_selected_x1_logit", float("nan")))
    ):
        raise _hold("beam node route/owner/transcript/lineage binding drifted")
    previous_route = recursive.CONTROL_PARENT_ROUTE_SHA256
    previous_owners = list(map(str, transcript[0].get("parent_owner_ids", ())))
    for index, entry in enumerate(transcript):
        if (
            int(entry.get("depth", -1)) != index
            or entry.get("parent_route_sha256") != previous_route
            or list(map(str, entry.get("parent_owner_ids", ()))) != previous_owners
        ):
            raise _hold("beam transcript ancestry drifted")
        previous_route = str(entry.get("generated_token_ids_sha256", ""))
        previous_owners = list(map(str, entry.get("matched_owner_ids", ())))
    evaluation = dict(node.get("evaluation", {}))
    if evaluation and (
        list(map(int, evaluation.get("generated_token_ids", ()))) != tokens
        or list(map(str, evaluation.get("matched_target_owner_ids", ()))) != owners
    ):
        raise _hold("beam node evaluation identity drifted")


def _seed_node(
    depth_zero: Mapping[str, Any], candidate: Mapping[str, Any],
    descriptions: Mapping[str, str],
) -> dict[str, Any]:
    x1 = int(candidate["x1_token_id"])
    entry = recursive._transcript_entry(
        candidate,
        parent_tokens=depth_zero["generated_token_ids"],
        parent_owner_ids=depth_zero["matched_owner_ids"],
        depth=1,
    )
    transcript = [deepcopy(dict(depth_zero)), entry]
    route = list(map(int, candidate["generated_token_ids"]))
    owners = list(map(str, candidate["matched_owner_ids"]))
    node_id = f"seed-{x1}-{candidate['generated_token_ids_sha256'][:16]}"
    node = {
        "node_id": node_id,
        "parent_node_id": DEPTH_ZERO_NODE_ID,
        "lineage_node_ids": [ROOT_NODE_ID, DEPTH_ZERO_NODE_ID, node_id],
        "route_tokens": route,
        "route_sha256": str(candidate["generated_token_ids_sha256"]),
        "matched_owner_ids": owners,
        "matched_person_owner_ids": _person_owner_ids(owners, descriptions),
        "matched_person_count": int(candidate["matched_person_count"]),
        "matched_owner_count": int(candidate["matched_owner_count"]),
        "cumulative_intervention_count": 2,
        "cumulative_selected_x1_logit": sum(
            float(item["x1_logit"]) for item in transcript
        ),
        "intervention_transcript": transcript,
        "evaluation": deepcopy(dict(candidate["evaluation"])),
        "source_task_id": f"frozen-seed-{x1}",
    }
    _validate_node(node, descriptions)
    return node


def _validate_source_receipt(
    receipt: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    bindings = dict(receipt.get("bindings", {}))
    counts = dict(receipt.get("counts", {}))
    resources = dict(receipt.get("resources", {}))
    transcript = list(receipt.get("intervention_transcript", ()))
    replay = dict(receipt.get("fresh_replay", {}))
    runtime = dict(receipt.get("runtime", {}))
    depths = list(receipt.get("depths", ()))
    final = dict(receipt.get("final", {}))
    if (
        receipt.get("schema_version") != recursive.SCHEMA_VERSION
        or receipt.get("unit_id") != recursive.UNIT_ID
        or receipt.get("run_id")
        != "20260830T-image2299-recursive-matcher-branch-search-v1b"
        or receipt.get("status") != "controlled_search_exhausted"
        or not source_path.is_file()
        or str(source_path.parent) != str(SOURCE_RECEIPT.parent)
        or source.get("sha256") != SOURCE_RUNNER_SHA256
        or base._sha256(source_path) != SOURCE_RUNNER_SHA256
        or base._hash(bindings) != SOURCE_BINDINGS_SHA256
        or base._hash(counts) != SOURCE_COUNTS_SHA256
        or base._hash(resources) != SOURCE_RESOURCES_SHA256
        or dict(counts.get("total", {}))
        != {
            "branch_logit_forwards": 5,
            "replay_completions": 4,
            "warm_controlled_completions": 33,
        }
        or bindings.get("checkpoint") != str(recursive.START_CHECKPOINT)
        or bindings.get("start_surface_sha256") != recursive.START_SURFACE_SHA256
        or bindings.get("frozen_surface_sha256") != recursive.FROZEN_SURFACE_SHA256
        or bindings.get("authority_description_map_sha256")
        != recursive.AUTHORITY_DESCRIPTION_MAP_SHA256
        or bindings.get("person_owner_count") != PERSON_OWNER_COUNT
        or bindings.get("tie_owner_count") != recursive.TIE_OWNER_COUNT
        or bindings.get("world_size") != WORLD_SIZE
        or bindings.get("control_parent_route_sha256")
        != recursive.CONTROL_PARENT_ROUTE_SHA256
        or bindings.get("model_receipt_sha256") != recursive.MODEL_RECEIPT_SHA256
        or bindings.get("discovery_receipt_sha256")
        != recursive.DISCOVERY_RECEIPT_SHA256
        or bindings.get("prompt_token_ids_sha256") != base.PROMPT_TOKEN_SHA256
        or bindings.get("image_sha256") != base.IMAGE_SHA256
        or bindings.get("target_sha256") != base.TARGET_SHA256
        or bindings.get("parent_owner_ids") != contract["parent_owner_ids"]
        or len(transcript) != 4
        or replay.get("completion_count") != 4
        or replay.get("final_generated_token_ids_sha256") != SOURCE_FINAL_ROUTE_SHA256
        or final.get("generated_token_ids_sha256") != SOURCE_FINAL_ROUTE_SHA256
        or final.get("matched_person_count") != 37
        or replay.get("final_generated_token_ids") != final.get("generated_token_ids")
        or runtime.get("warm") != runtime.get("fresh_rank0")
        or len(depths) != 5
    ):
        raise _hold("v1b source/checkpoint/surface/authority/count/resource binding drifted")
    if (
        base._hash(transcript[0]) != SOURCE_DEPTH_ZERO_TRANSCRIPT_SHA256
        or transcript[0].get("x1_token_id") != DEPTH_ZERO_X1
        or transcript[0].get("generated_token_ids_sha256") != DEPTH_ZERO_ROUTE_SHA256
        or transcript[0].get("matched_person_count") != recursive.DEPTH_ZERO_PERSON_COUNT
        or transcript[0].get("matched_owner_count") != 36
        or transcript[0].get("gate_debt") != {}
        or depths[0].get("selected_x1_token_id") != DEPTH_ZERO_X1
    ):
        raise _hold("v1b selected depth0 route drifted")
    replay_records = list(replay.get("records", ()))
    if len(replay_records) != len(transcript):
        raise _hold("v1b fresh replay record cardinality drifted")
    previous_tokens = list(map(int, contract["parent_route_tokens"]))
    for recorded, replayed in zip(transcript, replay_records, strict=True):
        try:
            recursive._verify_replay_record(
                recorded, replayed, previous_tokens=previous_tokens,
            )
        except BaseException as error:
            raise _hold("v1b fresh replay differs from recorded path") from error
        previous_tokens = list(map(int, recorded["generated_token_ids"]))

    alternatives = [dict(item) for item in dict(depths[1]).get("alternatives", ())]
    eligible = {
        int(item.get("x1_token_id", -1)): item
        for item in alternatives
        if bool(item.get("admitted")) and int(item.get("matched_person_count", -1)) == 35
    }
    if tuple(sorted(eligible)) != SEED_X1_TOKENS or len(eligible) != len(SEED_X1_TOKENS):
        raise _hold("v1b six-seed set drifted")
    descriptions = dict(contract["description_map"])
    depth_zero = dict(transcript[0])
    seed_nodes: list[dict[str, Any]] = []
    for index, x1 in enumerate(SEED_X1_TOKENS):
        candidate = eligible[x1]
        gate = dict(candidate.get("gate", {}))
        owners = list(map(str, candidate.get("matched_owner_ids", ())))
        evaluation = dict(candidate.get("evaluation", {}))
        if (
            int(candidate.get("action_index", -1)) != index
            or int(candidate.get("cumulative_intervention_count", -1)) != 2
            or candidate.get("generated_token_ids_sha256") != SEED_ROUTE_SHA256[x1]
            or token_ids_sha256(candidate.get("generated_token_ids", ()))
            != SEED_ROUTE_SHA256[x1]
            or base._hash(evaluation) != SEED_EVALUATION_SHA256[x1]
            or base._hash(owners) != SEED_OWNER_FAMILY_SHA256[x1]
            or tuple(gate.get("gained_owner_ids", ())) != SEED_GAIN[x1]
            or gate.get("parent_owner_ids") != depth_zero["matched_owner_ids"]
            or gate.get("matched_owner_ids") != owners
            or gate.get("passed") is not True
            or gate.get("debt") != {}
            or candidate.get("matched_person_count") != 35
            or candidate.get("matched_owner_count") != 38
            or len(_person_owner_ids(owners, descriptions)) != 35
            or evaluation.get("generated_token_ids")
            != candidate.get("generated_token_ids")
            or evaluation.get("matched_target_owner_ids") != owners
        ):
            raise _hold(f"v1b seed route/gate/owner family drifted for x1={x1}")
        seed_nodes.append(_seed_node(depth_zero, candidate, descriptions))
    if (
        len({node["node_id"] for node in seed_nodes}) != len(seed_nodes)
        or len({node["route_sha256"] for node in seed_nodes}) != len(seed_nodes)
        or base._hash(receipt) != SOURCE_RECEIPT_OBJECT_SHA256
    ):
        raise _hold("v1b source object or seed identity drifted")
    return {
        "receipt": dict(receipt),
        "depth_zero_transcript": depth_zero,
        "seed_nodes": seed_nodes,
    }


def _family_id(person_owner_ids: Sequence[str]) -> str:
    return base._hash(sorted(map(str, person_owner_ids)))


def _deduplicate_routes(
    nodes: Sequence[Mapping[str, Any]], descriptions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept: list[dict[str, Any]] = []
    by_route: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    for raw in sorted((dict(item) for item in nodes), key=_node_sort_key):
        _validate_node(raw, descriptions)
        route = str(raw["route_sha256"])
        existing = by_route.get(route)
        if existing is None:
            by_route[route] = raw
            kept.append(raw)
            continue
        if (
            existing["route_tokens"] != raw["route_tokens"]
            or existing["matched_owner_ids"] != raw["matched_owner_ids"]
            or existing["matched_person_owner_ids"] != raw["matched_person_owner_ids"]
        ):
            raise _hold("duplicate route hash has conflicting node semantics")
        duplicates.append({
            "route_sha256": route,
            "kept_node_id": str(existing["node_id"]),
            "discarded_node_id": str(raw["node_id"]),
        })
    return kept, duplicates


def _select_diverse(
    nodes: Sequence[Mapping[str, Any]], descriptions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduplicated, duplicates = _deduplicate_routes(nodes, descriptions)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for node in deduplicated:
        family = tuple(map(str, node["matched_person_owner_ids"]))
        groups.setdefault(family, []).append(node)
    families: list[dict[str, Any]] = []
    capped: list[dict[str, Any]] = []
    for family in sorted(groups):
        ordered = sorted(groups[family], key=_node_sort_key)
        kept = ordered[:FAMILY_CAP]
        capped.extend(kept)
        families.append({
            "family_id": _family_id(family),
            "matched_person_owner_ids": list(family),
            "candidate_node_ids": [str(item["node_id"]) for item in ordered],
            "candidate_route_sha256": [str(item["route_sha256"]) for item in ordered],
            "kept_node_ids": [str(item["node_id"]) for item in kept],
            "discarded_by_cap_node_ids": [
                str(item["node_id"]) for item in ordered[FAMILY_CAP:]
            ],
        })
    selected = sorted(capped, key=_node_sort_key)[:BEAM_WIDTH]
    if len({node["node_id"] for node in selected}) != len(selected):
        raise _hold("selected beam node IDs are not unique")
    return selected, {
        "family_cap": FAMILY_CAP,
        "beam_width": BEAM_WIDTH,
        "families": families,
        "duplicate_routes": duplicates,
        "globally_selected_node_ids": [str(item["node_id"]) for item in selected],
    }


def _best_node(
    nodes: Sequence[Mapping[str, Any]], descriptions: Mapping[str, str],
) -> dict[str, Any]:
    deduplicated, _duplicates = _deduplicate_routes(nodes, descriptions)
    if not deduplicated:
        raise _hold("cannot select from an empty node set")
    return sorted(deduplicated, key=_node_sort_key)[0]


def _flatten_frontier(
    beam: Sequence[Mapping[str, Any]], *, descriptions: Mapping[str, str],
    aliases: Sequence[Mapping[str, Any]], cumulative_intervention_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not FIRST_EXPANSION_INTERVENTION <= cumulative_intervention_count <= MAX_INTERVENTIONS:
        raise _hold("expansion intervention depth is outside 3..6")
    tasks: list[dict[str, Any]] = []
    frontiers: list[dict[str, Any]] = []
    ordered_beam = sorted((dict(item) for item in beam), key=_node_sort_key)
    for node_index, node in enumerate(ordered_beam):
        _validate_node(node, descriptions)
        actions = recursive._missing_person_actions(
            node["matched_owner_ids"], descriptions=descriptions, aliases=aliases,
        )
        if len(actions) > MAX_ACTIONS_PER_NODE:
            raise _hold("node action catalog exceeds frozen limit")
        missing = sorted(
            owner for owner, description in descriptions.items()
            if description == "person" and owner not in set(node["matched_owner_ids"])
        )
        frontiers.append({
            "node_id": node["node_id"],
            "missing_person_owner_ids": missing,
            "actions": actions,
        })
        for node_action_index, action in enumerate(actions):
            task_index = len(tasks)
            identity = {
                "cumulative_intervention_count": cumulative_intervention_count,
                "node_index": node_index,
                "node_id": node["node_id"],
                "node_action_index": node_action_index,
                "x1_token_id": int(action["x1_token_id"]),
            }
            tasks.append({
                **identity,
                "task_index": task_index,
                "task_id": (
                    f"i{cumulative_intervention_count}-"
                    f"{base._hash(identity)[:24]}"
                ),
                "parent_route_sha256": node["route_sha256"],
                "parent_owner_ids": node["matched_owner_ids"],
                "action": action,
            })
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise _hold("flattened frontier task IDs are not unique")
    return tasks, frontiers


def _precheck_warm_budget(completed: int, generated_frontier: int) -> None:
    if (
        completed < SEED_WARM_COMPLETIONS
        or generated_frontier < 0
        or completed + generated_frontier > MAX_WARM_COMPLETIONS
    ):
        raise _hold("generated frontier would exceed 250 warm completions")


def _warm_replay_seed(
    *, seed_index: int, seed: Mapping[str, Any], model: Any, tokenizer: Any,
    native_inputs: Mapping[str, Any], contract: Mapping[str, Any], raw_example: Any,
) -> dict[str, Any]:
    previous_tokens = list(map(int, contract["parent_route_tokens"]))
    previous_owners = list(map(str, contract["parent_owner_ids"]))
    replay_records: list[dict[str, Any]] = []
    for recorded in seed["intervention_transcript"]:
        x1 = int(recorded["x1_token_id"])
        prefix = recursive._splice_branch_prefix(previous_tokens, x1)["tokens"]
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
            label=f"beam-warm-seed-{seed_index}-depth-{recorded['depth']}",
        )
        gate = recursive._node_gate(evaluation, parent_owner_ids=previous_owners)
        owners = list(gate["matched_owner_ids"])
        replayed = {
            **completion,
            "matched_owner_ids": owners,
            "matched_person_count": recursive._person_count(
                owners, contract["description_map"],
            ),
            "matched_owner_count": len(owners),
            "gate_debt": dict(gate["debt"]),
        }
        recursive._verify_replay_record(
            recorded, replayed, previous_tokens=previous_tokens,
        )
        replay_records.append(replayed)
        previous_tokens = completion["generated_token_ids"]
        previous_owners = owners
    return {
        "seed_index": seed_index,
        "seed_task_id": f"warm-seed-{seed_index}-{seed['node_id']}",
        "node_id": seed["node_id"],
        "completion_count": len(replay_records),
        "records": replay_records,
        "final_route_sha256": token_ids_sha256(previous_tokens),
        "final_owner_ids": previous_owners,
        "final_gate_debt": dict(replay_records[-1]["gate_debt"]),
    }


def _verify_warm_seed_gather(
    gathered: Sequence[Mapping[str, Any]], *, seeds: Sequence[Mapping[str, Any]],
    parent_route_tokens: Sequence[int],
) -> list[dict[str, Any]]:
    ranks = [int(item.get("rank", -1)) for item in gathered]
    records = [dict(record) for item in gathered for record in item.get("records", ())]
    if (
        sorted(ranks) != list(range(WORLD_SIZE))
        or len(ranks) != len(set(ranks))
        or any(item.get("error") is not None for item in gathered)
        or any(record.get("evaluation_error") is not None for record in records)
    ):
        raise _hold("warm seed collective or replay failed")
    by_index = {int(record.get("seed_index", -1)): record for record in records}
    if sorted(by_index) != list(range(len(seeds))) or len(records) != len(seeds):
        raise _hold("six warm seed transcripts were not distributed exactly once")
    ordered: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        record = by_index[seed_index]
        if (
            int(record.get("rank", -1)) != seed_index % WORLD_SIZE
            or record.get("seed_task_id")
            != f"warm-seed-{seed_index}-{seed['node_id']}"
            or record.get("node_id") != seed["node_id"]
            or record.get("completion_count") != 2
            or record.get("final_route_sha256") != seed["route_sha256"]
            or record.get("final_owner_ids") != seed["matched_owner_ids"]
            or record.get("final_gate_debt") != {}
        ):
            raise _hold("warm seed replay differs from sealed seed")
        previous_tokens = list(map(int, parent_route_tokens))
        replay_records = list(record.get("records", ()))
        if len(replay_records) != 2:
            raise _hold("warm seed replay record cardinality drifted")
        for recorded, replayed in zip(
            seed["intervention_transcript"], replay_records, strict=True,
        ):
            try:
                recursive._verify_replay_record(
                    recorded, replayed, previous_tokens=previous_tokens,
                )
            except BaseException as error:
                raise _hold("warm seed replay differs from sealed intervention") from error
            previous_tokens = list(map(int, recorded["generated_token_ids"]))
        ordered.append(record)
    return ordered


def _evaluate_task(
    *, task: Mapping[str, Any], node: Mapping[str, Any], logit_packet: Mapping[str, Any],
    model: Any, tokenizer: Any, native_inputs: Mapping[str, Any],
    contract: Mapping[str, Any], raw_example: Any,
) -> dict[str, Any]:
    if (
        task["node_id"] != node["node_id"]
        or task["parent_route_sha256"] != node["route_sha256"]
        or task["parent_owner_ids"] != node["matched_owner_ids"]
    ):
        raise _hold("flattened task parent binding drifted")
    candidate = recursive._evaluate_action(
        model=model,
        tokenizer=tokenizer,
        native_inputs=native_inputs,
        pad=int(tokenizer.pad_token_id),
        action_index=int(task["task_index"]),
        action=task["action"],
        parent_tokens=node["route_tokens"],
        parent_owner_ids=node["matched_owner_ids"],
        target=contract["target"],
        raw_example=raw_example,
        descriptions=contract["description_map"],
        logit_packet=logit_packet,
        depth=int(task["cumulative_intervention_count"]) - 1,
    )
    candidate.update({
        "task_id": task["task_id"],
        "task_index": task["task_index"],
        "node_action_index": task["node_action_index"],
        "parent_node_id": node["node_id"],
        "parent_route_sha256": node["route_sha256"],
        "parent_owner_ids": node["matched_owner_ids"],
    })
    return candidate


def _verify_frontier_gather(
    gathered: Sequence[Mapping[str, Any]], *, tasks: Sequence[Mapping[str, Any]],
    beam: Sequence[Mapping[str, Any]], cumulative_intervention_count: int,
) -> list[dict[str, Any]]:
    ranks = [int(item.get("rank", -1)) for item in gathered]
    records = [dict(record) for item in gathered for record in item.get("records", ())]
    if (
        sorted(ranks) != list(range(WORLD_SIZE))
        or len(ranks) != len(set(ranks))
        or any(item.get("error") is not None for item in gathered)
        or any(record.get("evaluation_error") is not None for record in records)
    ):
        raise _hold(f"intervention {cumulative_intervention_count} candidate evaluation failed")
    by_index = {int(record.get("task_index", -1)): record for record in records}
    by_node = {str(node["node_id"]): node for node in beam}
    if sorted(by_index) != list(range(len(tasks))) or len(records) != len(tasks):
        raise _hold("flattened world8 task IDs were not covered exactly once")
    ordered: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        record = by_index[task_index]
        node = by_node[str(task["node_id"])]
        gate = dict(record.get("gate", {}))
        if (
            int(record.get("rank", -1)) != task_index % WORLD_SIZE
            or record.get("task_id") != task["task_id"]
            or record.get("parent_node_id") != node["node_id"]
            or record.get("parent_route_sha256") != node["route_sha256"]
            or record.get("parent_owner_ids") != node["matched_owner_ids"]
            or gate.get("parent_owner_ids") != sorted(node["matched_owner_ids"])
            or int(record.get("x1_token_id", -1))
            != int(task["action"]["x1_token_id"])
            or int(record.get("cumulative_intervention_count", -1))
            != cumulative_intervention_count
        ):
            raise _hold("candidate is not bound to its flattened node-specific parent")
        ordered.append(record)
    return ordered


def _child_node(
    candidate: Mapping[str, Any], *, parent: Mapping[str, Any],
    descriptions: Mapping[str, str],
) -> dict[str, Any]:
    gate = dict(candidate.get("gate", {}))
    if (
        candidate.get("admitted") is not True
        or gate.get("passed") is not True
        or gate.get("debt")
        or candidate.get("parent_node_id") != parent["node_id"]
        or candidate.get("parent_route_sha256") != parent["route_sha256"]
        or candidate.get("parent_owner_ids") != parent["matched_owner_ids"]
        or gate.get("parent_owner_ids") != sorted(parent["matched_owner_ids"])
    ):
        raise _hold("admitted child lost its node-specific parent gate")
    intervention_count = int(candidate["cumulative_intervention_count"])
    entry = recursive._transcript_entry(
        candidate,
        parent_tokens=parent["route_tokens"],
        parent_owner_ids=parent["matched_owner_ids"],
        depth=intervention_count - 1,
    )
    route = list(map(int, candidate["generated_token_ids"]))
    owners = list(map(str, candidate["matched_owner_ids"]))
    identity = {
        "parent_node_id": parent["node_id"],
        "task_id": candidate["task_id"],
        "route_sha256": candidate["generated_token_ids_sha256"],
    }
    node_id = f"node-i{intervention_count}-{base._hash(identity)[:24]}"
    node = {
        "node_id": node_id,
        "parent_node_id": parent["node_id"],
        "lineage_node_ids": [*parent["lineage_node_ids"], node_id],
        "route_tokens": route,
        "route_sha256": str(candidate["generated_token_ids_sha256"]),
        "matched_owner_ids": owners,
        "matched_person_owner_ids": _person_owner_ids(owners, descriptions),
        "matched_person_count": int(candidate["matched_person_count"]),
        "matched_owner_count": int(candidate["matched_owner_count"]),
        "cumulative_intervention_count": intervention_count,
        "cumulative_selected_x1_logit": (
            float(parent["cumulative_selected_x1_logit"])
            + float(candidate["x1_logit"])
        ),
        "intervention_transcript": [*deepcopy(parent["intervention_transcript"]), entry],
        "evaluation": deepcopy(dict(candidate["evaluation"])),
        "source_task_id": candidate["task_id"],
    }
    _validate_node(node, descriptions)
    return node


def _public_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in node.items() if key != "evaluation"}


def _initial_family_receipt(seeds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families: dict[tuple[str, ...], list[str]] = {}
    for node in seeds:
        families.setdefault(tuple(node["matched_person_owner_ids"]), []).append(node["node_id"])
    return [
        {
            "family_id": _family_id(family),
            "matched_person_owner_ids": list(family),
            "seed_node_ids": node_ids,
        }
        for family, node_ids in sorted(families.items())
    ]


def _terminal_status(
    *, has_success: bool, admitted_child_count: int, cumulative_intervention_count: int,
) -> str | None:
    if has_success:
        return "controlled_38_person_success"
    if admitted_child_count == 0:
        return "beam_search_exhausted"
    if cumulative_intervention_count >= MAX_INTERVENTIONS:
        return "beam_depth_exhausted"
    return None


def _enforce_budget(
    counts: Mapping[str, int], *, replay_interventions: int, expected_warm: int,
) -> None:
    if (
        set(counts)
        != {"warm_controlled_completions", "branch_logit_forwards", "replay_completions"}
        or int(counts.get("warm_controlled_completions", -1)) != expected_warm
        or expected_warm < SEED_WARM_COMPLETIONS
        or expected_warm > MAX_WARM_COMPLETIONS
        or not 0 <= int(counts.get("branch_logit_forwards", -1)) <= MAX_BRANCH_LOGIT_FORWARDS
        or int(counts.get("replay_completions", -1)) != replay_interventions
        or not 2 <= replay_interventions <= MAX_REPLAY_COMPLETIONS
    ):
        raise _hold("beam-search execution budget drifted")


def _require_world8(value: int) -> None:
    if value != WORLD_SIZE:
        raise _hold("beam search requires torchrun --nproc_per_node=8")


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
        "source_receipt_object_sha256": SOURCE_RECEIPT_OBJECT_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_status": source["receipt"]["status"],
        "source_final_route_sha256": SOURCE_FINAL_ROUTE_SHA256,
        "model_receipt": str(recursive.MODEL_RECEIPT),
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
        "person_owner_count": PERSON_OWNER_COUNT,
        "tie_owner_count": recursive.TIE_OWNER_COUNT,
        "control_parent_route_sha256": recursive.CONTROL_PARENT_ROUTE_SHA256,
        "depth_zero_x1_token_id": DEPTH_ZERO_X1,
        "depth_zero_route_sha256": DEPTH_ZERO_ROUTE_SHA256,
        "seed_x1_token_ids": list(SEED_X1_TOKENS),
        "seed_route_sha256": SEED_ROUTE_SHA256,
        "seed_owner_family_sha256": SEED_OWNER_FAMILY_SHA256,
        "seed_gain": SEED_GAIN,
        "world_size": WORLD_SIZE,
        "resource_bound": RESOURCE_BOUND,
    }


def run(*, run_id: str) -> Path:
    _require_world8(int(os.environ.get("WORLD_SIZE", "0")))
    contract = recursive._start_contract()
    source = _validate_source_receipt(_load_source_receipt(), contract)
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    stage = "distributed_initialization"
    owns_output = False
    source_snapshot: Mapping[str, Any] | None = None
    depth_records: list[dict[str, Any]] = []
    terminal_node: dict[str, Any] | None = None
    status: str | None = None
    local_counts = {
        "warm_controlled_completions": 0,
        "branch_logit_forwards": 0,
        "replay_completions": 0,
    }
    warm_completion_count = SEED_WARM_COMPLETIONS
    try:
        dist.init_process_group(backend="nccl")
        _require_world8(dist.get_world_size())
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise _hold("LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        stage = "warm_seed_replay_and_beam_search"
        setup = contract["setup"]
        descriptions = contract["description_map"]
        seeds = [deepcopy(item) for item in source["seed_nodes"]]
        beam = seeds
        warm_runtime: Mapping[str, Any] | None = None
        initial_surface: Mapping[str, Any] | None = None

        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("warm beam search requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            if (
                token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
            ):
                raise _hold("warm prompt/image identity drifted")
            names, parameters = ota._step_trainable_surface(model, r32_step=True)
            initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_before = base._frozen_surface(model, names)
            if (
                initial_surface.get("aggregate_sha256") != recursive.START_SURFACE_SHA256
                or frozen_before != recursive.FROZEN_SURFACE_SHA256
            ):
                raise _hold("warm selected-r32-step surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)

            local_seed_records: list[dict[str, Any]] = []
            for seed_index in range(rank, len(seeds), WORLD_SIZE):
                try:
                    record = _warm_replay_seed(
                        seed_index=seed_index,
                        seed=seeds[seed_index],
                        model=model,
                        tokenizer=tokenizer,
                        native_inputs=native_inputs,
                        contract=contract,
                        raw_example=setup["raw_example"],
                    )
                except BaseException as error:
                    record = {
                        "seed_index": seed_index,
                        "evaluation_error": {
                            "type": type(error).__name__, "error": str(error),
                        },
                    }
                record["rank"] = rank
                local_seed_records.append(record)
                local_counts["warm_controlled_completions"] += 2
            gathered_seeds: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_seeds, {
                "rank": rank, "records": local_seed_records, "error": None,
            })
            seed_verification = base._rank0_call(lambda: _verify_warm_seed_gather(
                gathered_seeds,
                seeds=seeds,
                parent_route_tokens=contract["parent_route_tokens"],
            ))
            ota._agree_hash(seed_verification, label="beam-warm-six-seed-verification")
            depth_records.append({
                "cumulative_intervention_count": 2,
                "kind": "warm_verified_initial_seed_beam",
                "input_frontier": [],
                "seed_verification": seed_verification,
                "family_grouping": _initial_family_receipt(seeds),
                "selected_beam": [_public_node(node) for node in beam],
            })

            for intervention_count in range(
                FIRST_EXPANSION_INTERVENTION, MAX_INTERVENTIONS + 1,
            ):
                tasks, frontier = _flatten_frontier(
                    beam,
                    descriptions=descriptions,
                    aliases=contract["aliases"],
                    cumulative_intervention_count=intervention_count,
                )
                if not tasks:
                    terminal_node = _best_node(beam, descriptions)
                    status = "beam_search_exhausted"
                    depth_records.append({
                        "cumulative_intervention_count": intervention_count,
                        "kind": "generated_frontier",
                        "input_frontier": [_public_node(node) for node in beam],
                        "node_actions": frontier,
                        "generated_tasks": [],
                        "logit_packets": {},
                        "candidate_records": [],
                        "admitted_child_node_ids": [],
                        "family_grouping": {
                            "family_cap": FAMILY_CAP,
                            "beam_width": BEAM_WIDTH,
                            "families": [],
                            "duplicate_routes": [],
                            "globally_selected_node_ids": [],
                        },
                        "selected_beam": [],
                        "terminal_status": status,
                    })
                    break
                _precheck_warm_budget(warm_completion_count, len(tasks))
                beam_by_id = {str(node["node_id"]): node for node in beam}
                actions_by_node = {
                    str(item["node_id"]): list(item["actions"]) for item in frontier
                }
                logit_packets: dict[str, dict[str, Any]] = {}
                for node in sorted(beam, key=_node_sort_key):
                    actions = actions_by_node[str(node["node_id"])]
                    if not actions:
                        continue
                    packet = base._rank0_call(
                        lambda node=node, actions=actions, model=model,
                        native_inputs=native_inputs,
                        tokenizer=tokenizer: recursive._logit_packet(
                            model=model,
                            native_inputs=native_inputs,
                            pad=int(tokenizer.pad_token_id),
                            node_tokens=node["route_tokens"],
                            actions=actions,
                            depth=intervention_count - 1,
                        )
                    )
                    if rank == 0:
                        local_counts["branch_logit_forwards"] += 1
                    ota._agree_hash(
                        packet, label=f"beam-i{intervention_count}-{node['node_id']}-logits",
                    )
                    logit_packets[str(node["node_id"])] = packet

                local_records: list[dict[str, Any]] = []
                for task_index in range(rank, len(tasks), WORLD_SIZE):
                    task = tasks[task_index]
                    node = beam_by_id[str(task["node_id"])]
                    try:
                        record = _evaluate_task(
                            task=task,
                            node=node,
                            logit_packet=logit_packets[str(node["node_id"])],
                            model=model,
                            tokenizer=tokenizer,
                            native_inputs=native_inputs,
                            contract=contract,
                            raw_example=setup["raw_example"],
                        )
                    except BaseException as error:
                        record = {
                            "task_index": task_index,
                            "task_id": task["task_id"],
                            "parent_node_id": task["node_id"],
                            "evaluation_error": {
                                "type": type(error).__name__, "error": str(error),
                            },
                        }
                    record["rank"] = rank
                    local_records.append(record)
                    local_counts["warm_controlled_completions"] += 1
                gathered: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered, {
                    "rank": rank, "records": local_records, "error": None,
                })
                candidates = base._rank0_call(lambda: _verify_frontier_gather(
                    gathered,
                    tasks=tasks,
                    beam=beam,
                    cumulative_intervention_count=intervention_count,
                ))
                ota._agree_hash(
                    candidates, label=f"beam-i{intervention_count}-candidate-records",
                )
                warm_completion_count += len(tasks)
                children = [
                    _child_node(
                        candidate,
                        parent=beam_by_id[str(candidate["parent_node_id"])],
                        descriptions=descriptions,
                    )
                    for candidate in candidates if bool(candidate.get("admitted"))
                ]
                selected, grouping = _select_diverse(children, descriptions)
                successes = [
                    node for node in children
                    if int(node["matched_person_count"]) == PERSON_OWNER_COUNT
                ]
                status = _terminal_status(
                    has_success=bool(successes),
                    admitted_child_count=len(children),
                    cumulative_intervention_count=intervention_count,
                )
                if successes:
                    terminal_node = _best_node(successes, descriptions)
                    selected_for_receipt = [terminal_node]
                elif not children:
                    terminal_node = _best_node(beam, descriptions)
                    selected_for_receipt = []
                else:
                    beam = selected
                    selected_for_receipt = beam
                    if status == "beam_depth_exhausted":
                        terminal_node = _best_node(beam, descriptions)
                depth_records.append({
                    "cumulative_intervention_count": intervention_count,
                    "kind": "generated_frontier",
                    "input_frontier": [_public_node(node) for node in beam_by_id.values()],
                    "node_actions": frontier,
                    "generated_tasks": [
                        {key: deepcopy(value) for key, value in task.items() if key != "action"}
                        for task in tasks
                    ],
                    "logit_packets": logit_packets,
                    "candidate_records": candidates,
                    "admitted_child_node_ids": [node["node_id"] for node in children],
                    "family_grouping": grouping,
                    "selected_beam": [
                        _public_node(node) for node in selected_for_receipt
                    ],
                    "terminal_status": status,
                })
                if status is not None:
                    break

            if status not in TERMINAL_STATUSES or terminal_node is None:
                raise _hold("beam search exited without a terminal node/status")
            after_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if (
                after_surface != initial_surface
                or base._frozen_surface(model, names) != frozen_before
                or any(parameter.grad is not None for parameter in model.parameters())
            ):
                raise _hold("inference-only warm beam search mutated model state")
            warm_runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, parameters, names
        torch.cuda.empty_cache()
        dist.barrier()
        stage = "fresh_rank0_replay"
        replay = base._rank0_call(lambda: recursive._cold_replay(
            transcript=terminal_node["intervention_transcript"], contract=contract,
        ))
        if rank == 0:
            local_counts["replay_completions"] = len(
                terminal_node["intervention_transcript"]
            )
        ota._agree_hash(replay, label="beam-fresh-rank0-replay")
        if (
            replay.get("final_generated_token_ids_sha256") != terminal_node["route_sha256"]
            or replay.get("final_generated_token_ids") != terminal_node["route_tokens"]
            or replay.get("completion_count")
            != terminal_node["cumulative_intervention_count"]
            or replay.get("surface") != initial_surface
            or replay.get("runtime") != warm_runtime
        ):
            raise _hold("fresh replay route/surface/runtime identity drifted")

        local_resources = {
            "rank": rank,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(local_rank)),
            "device_total_memory_bytes": int(
                torch.cuda.get_device_properties(local_rank).total_memory
            ),
        }
        gathered_resources: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_resources, local_resources)
        per_rank_counts, total_counts = recursive._gather_totals(local_counts)
        _enforce_budget(
            total_counts,
            replay_interventions=terminal_node["cumulative_intervention_count"],
            expected_warm=warm_completion_count,
        )
        elapsed = time.monotonic() - started
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or max(int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources)
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes_per_rank"]
        ):
            raise _hold("beam search resource bound exceeded")
        final_evaluation = dict(terminal_node["evaluation"])
        final_parent_owners = terminal_node["intervention_transcript"][-1][
            "parent_owner_ids"
        ]
        final_gate = recursive._node_gate(
            final_evaluation, parent_owner_ids=final_parent_owners,
        )
        if (
            not final_gate["passed"]
            or final_gate["matched_owner_ids"] != terminal_node["matched_owner_ids"]
            or status == "controlled_38_person_success"
            and terminal_node["matched_person_count"] != PERSON_OWNER_COUNT
        ):
            raise _hold("terminal node no longer passes its exact parent gate")
        if rank == 0:
            ledger = dict(final_evaluation["ledger"])
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": status,
                "run_id": run_id,
                "runner_source_snapshot": source_snapshot,
                "bindings": {
                    "source_receipt": str(SOURCE_RECEIPT),
                    "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                    "source_receipt_object_sha256": SOURCE_RECEIPT_OBJECT_SHA256,
                    "source_runner_sha256": SOURCE_RUNNER_SHA256,
                    "model_receipt": str(recursive.MODEL_RECEIPT),
                    "model_receipt_sha256": recursive.MODEL_RECEIPT_SHA256,
                    "checkpoint": str(recursive.START_CHECKPOINT),
                    "checkpoint_readback": contract["checkpoint_readback"],
                    "start_surface_sha256": recursive.START_SURFACE_SHA256,
                    "frozen_surface_sha256": recursive.FROZEN_SURFACE_SHA256,
                    "prompt_token_ids_sha256": base.PROMPT_TOKEN_SHA256,
                    "image_sha256": base.IMAGE_SHA256,
                    "target_sha256": base.TARGET_SHA256,
                    "authority": contract["authority_source"],
                    "authority_description_map_sha256": (
                        recursive.AUTHORITY_DESCRIPTION_MAP_SHA256
                    ),
                    "control_parent_route_sha256": recursive.CONTROL_PARENT_ROUTE_SHA256,
                    "depth_zero_route_sha256": DEPTH_ZERO_ROUTE_SHA256,
                    "seed_route_sha256": SEED_ROUTE_SHA256,
                },
                "protocol": {
                    "world_size": WORLD_SIZE,
                    "initial_beam": "six warm-replayed 35-person v1b siblings",
                    "search": (
                        "complete per-node routes only; owner sets are never unioned"
                    ),
                    "action_source": (
                        "sealed OTA aliases for each node's authoritative missing persons only"
                    ),
                    "value": [
                        "matched_person_count",
                        "matched_owner_count",
                        "fewer_cumulative_interventions",
                        "cumulative_selected_x1_logit",
                        "deterministic_route_and_node_identity",
                    ],
                    "diversity": (
                        "deduplicate exact route hash; cap two routes per exact matched-person-owner set; keep best eight"
                    ),
                    "gradient": None,
                    "optimizer": None,
                    "weight_update": False,
                    "checkpoint_save": False,
                },
                "depths": depth_records,
                "selected_final_path": _public_node(terminal_node),
                "intervention_transcript": terminal_node["intervention_transcript"],
                "final": {
                    "node_id": terminal_node["node_id"],
                    "lineage_node_ids": terminal_node["lineage_node_ids"],
                    "generated_token_ids": terminal_node["route_tokens"],
                    "generated_token_ids_sha256": terminal_node["route_sha256"],
                    "matched_owner_ids": terminal_node["matched_owner_ids"],
                    "matched_owner_count": terminal_node["matched_owner_count"],
                    "matched_person_owner_ids": terminal_node[
                        "matched_person_owner_ids"
                    ],
                    "matched_person_count": terminal_node["matched_person_count"],
                    "matched_tie_count": sum(
                        descriptions[item] == "tie"
                        for item in terminal_node["matched_owner_ids"]
                    ),
                    "gate": final_gate,
                    "parsed_rows": list(
                        dict(ledger.get("parse", {})).get("predictions", ())
                    ),
                    "ledger": ledger,
                },
                "fresh_replay": replay,
                "counts": {"per_rank": per_rank_counts, "total": total_counts},
                "resources": {
                    "per_rank": gathered_resources,
                    "max_peak_cuda_allocated_bytes": max(
                        int(item["peak_cuda_allocated_bytes"])
                        for item in gathered_resources
                    ),
                    "max_peak_cuda_reserved_bytes": max(
                        int(item["peak_cuda_reserved_bytes"])
                        for item in gathered_resources
                    ),
                    "predeclared_bound": RESOURCE_BOUND,
                },
                "runtime": {"warm": warm_runtime, "fresh_rank0": replay["runtime"]},
                "wall_time_seconds": elapsed,
                "claim_boundary": (
                    "One frozen-r32 controlled-decoding trajectory on Image2299 only; "
                    "not ordinary greedy learning, transfer, exhaustive tree search, "
                    "or recovery of all eight ties."
                ),
            }
            base._atomic_json(output / "receipt.json", receipt)
            if (
                (output / "receipt.json").stat().st_size
                > RESOURCE_BOUND["output_artifact_bytes_max"]
            ):
                raise _hold("beam receipt exceeded artifact bound")
        dist.barrier()
        return output
    except BaseException as error:
        if dist.is_initialized() and dist.get_rank() == 0 and owns_output:
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
                "depths_partial": depth_records,
                "terminal_node_partial": (
                    None if terminal_node is None else _public_node(terminal_node)
                ),
                "counts_rank0_partial": local_counts,
                "wall_time_seconds": time.monotonic() - started,
            })
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


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
        raise SystemExit("GPU execution requires explicit --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
