#!/usr/bin/env python3
"""World8 progressive final-person coordinate release for frozen Image2299 r32."""

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

from scripts.research import run_image2299_beam_backtracking_person_search as beam


recursive = beam.recursive
ota = recursive.ota
base = recursive.base
token_ids_sha256 = recursive.token_ids_sha256

SCHEMA_VERSION = "image2299.progressive_final_person_release.v1"
UNIT_ID = "2026-08-30-image2299-progressive-final-person-release"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    beam.OUTPUT_ROOT
    / "20260830T-image2299-beam-backtracking-person-search-v1"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "d34eab98475525df264f78e490687b7264049190222eeed45d03649a7575fedc"
SOURCE_BINDINGS_SHA256 = "5973a64e1736d09cf57e8ff1758b890b40d53f4666e71ce0c0bf7eb23e8ca8aa"
SOURCE_COUNTS_SHA256 = "e5af03dd1d6d4e0627cd8e589c208a2fd50a24f2052ab49c802098f3b1a3d95c"
SOURCE_RESOURCES_SHA256 = "dfe7416abd5754d2f3332f7562dabaeb0335335ee8653a89ed0a28c10cd164df"
SOURCE_SELECTED_NODES_SHA256 = "128fc21d404c255df71bcce60a3addb06f8a2b8943b08569f2d099aefaba224b"
SOURCE_RUNNER_SHA256 = "0dd12c4771249fb25dd542adea01ffebc5ad066da5e8a22016c70893e2e6d8a1"
SOURCE_RUN_ID = "20260830T-image2299-beam-backtracking-person-search-v1"
SOURCE_NODE_IDS = (
    "node-i4-b00bf0302feccfdaa21a50fd",
    "node-i4-2ef943de8068c1bb968aa9fc",
    "node-i4-5b032c71eab750f7f1d22df8",
    "node-i4-36beb965099871fd4325f205",
)
SOURCE_NODE_BINDINGS: dict[str, dict[str, str]] = {
    SOURCE_NODE_IDS[0]: {
        "route_sha256": "d807156352c73ec783cf31a26c44da390ed98fe6a4503bc8e5d2d0a950e4fca1",
        "node_sha256": "26d266901a1a249b1d13df05939118f9dedfff997eb6f56ac7af873ff1f3d4a5",
        "transcript_sha256": "02f546a1a4bab1762abea4c271931b98ecc0bb647bd74f50ede1af59dde4a190",
        "family_sha256": "50d7d68b553731b6d057be4f9aa9e475ebf4849de4d287b566fc904b3fe724ac",
        "missing_person_owner_id": "gt:2299:35",
    },
    SOURCE_NODE_IDS[1]: {
        "route_sha256": "b8813d6266b843fe6f82e0e5c19bee6a13a93b82cc8688e8edb7e18e334ef2eb",
        "node_sha256": "ddbb717c1dcab3c9da581f6b7709b1de8e588f694035dcd332cf44b8bd662bbb",
        "transcript_sha256": "328faec7cef84a1d1b9dd93a257750d2207633a82502d1ba2ec016f7b5e3a5f9",
        "family_sha256": "50d7d68b553731b6d057be4f9aa9e475ebf4849de4d287b566fc904b3fe724ac",
        "missing_person_owner_id": "gt:2299:35",
    },
    SOURCE_NODE_IDS[2]: {
        "route_sha256": "7c50867e1af117888a31b8ff0ec3b14ad172d97c47ecc05747f89dd7b8b8b10a",
        "node_sha256": "b4c3ab246498d01f0574e9a7737a451e2ba080733eb67bd87b933fc08d8fd18d",
        "transcript_sha256": "98318840f36e9075bdb5d6fe4974dbfd5d03a7ae1550582e1f8549995153294e",
        "family_sha256": "9efc37f0566aff21ef9c88561b43c2261badc8f15d60075fda17d677a28d4f58",
        "missing_person_owner_id": "gt:2299:18",
    },
    SOURCE_NODE_IDS[3]: {
        "route_sha256": "8a84de2991c63706f7b3c78c650e9eade478710a3270a8a618bdda48d93d901a",
        "node_sha256": "9782b35c6869780ac8e461ab6a1dfee8c72bb86918b01c52f7c63b7069b38fe5",
        "transcript_sha256": "e342eeb412813e17490c8d3bdd75b91b5266a62429fd88d956ba33f05d056ea5",
        "family_sha256": "9efc37f0566aff21ef9c88561b43c2261badc8f15d60075fda17d677a28d4f58",
        "missing_person_owner_id": "gt:2299:18",
    },
}
SOURCE_FAMILY_SHA256 = (
    "50d7d68b553731b6d057be4f9aa9e475ebf4849de4d287b566fc904b3fe724ac",
    "9efc37f0566aff21ef9c88561b43c2261badc8f15d60075fda17d677a28d4f58",
)

WORLD_SIZE = 8
LEVELS = (2, 3, 4, 5)
FORCED_ROW_TOKENS = {2: 6, 3: 7, 4: 8, 5: 9}
FORCED_COORDINATES = {2: 2, 3: 3, 4: 4, 5: 4}
EXPECTED_LEVEL_TASK_COUNTS = {2: 20, 3: 28, 4: 32, 5: 32}
ROW_CLOSE = 151649
SOURCE_TRANSCRIPT_COMPLETIONS = 4 * 4
MAX_WARM_COMPLETIONS = 128
MAX_REPLAY_COMPLETIONS = 5
MAX_PROGRESSIVE_TASKS = MAX_WARM_COMPLETIONS - SOURCE_TRANSCRIPT_COMPLETIONS
RESOURCE_BOUND = {
    "gpu_count": WORLD_SIZE,
    "source_node_count": 4,
    "aliases_per_missing_person_max": 8,
    "progressive_levels": list(LEVELS),
    "progressive_tasks_per_level": EXPECTED_LEVEL_TASK_COUNTS,
    "source_warm_replay_completions": SOURCE_TRANSCRIPT_COMPLETIONS,
    "progressive_warm_completions_max": MAX_PROGRESSIVE_TASKS,
    "warm_completions_total_max": MAX_WARM_COMPLETIONS,
    "teacher_logprob_forwards_max": MAX_PROGRESSIVE_TASKS,
    "fresh_replay_completions_max": MAX_REPLAY_COMPLETIONS,
    "generated_tokens_per_decode_max": base.NATURAL_MAX_TOKENS,
    "max_peak_cuda_reserved_bytes_per_rank": 64 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
    "wall_time_seconds_max": 1_200,
}
TERMINAL_STATUSES = {
    "controlled_38_person_success",
    "progressive_release_exhausted",
}


class ProgressiveReleaseHold(RuntimeError):
    """A frozen source, prefix, matcher, replay, or resource contract failed."""


def _hold(message: str) -> ProgressiveReleaseHold:
    return ProgressiveReleaseHold(f"HOLD: {message}")


def _load_source_receipt() -> dict[str, Any]:
    try:
        return recursive._load_receipt(
            SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256, label="beam-search source",
        )
    except BaseException as error:
        raise _hold(str(error).removeprefix("HOLD: ")) from error


def _validate_source_node(
    node: Mapping[str, Any], descriptions: Mapping[str, str],
) -> dict[str, Any]:
    raw = deepcopy(dict(node))
    node_id = str(raw.get("node_id", ""))
    binding = SOURCE_NODE_BINDINGS.get(node_id)
    try:
        beam._validate_node(raw, descriptions)
    except BaseException as error:
        raise _hold(f"source node validation failed for {node_id}: {error}") from error
    owners = list(map(str, raw.get("matched_owner_ids", ())))
    people = list(map(str, raw.get("matched_person_owner_ids", ())))
    missing = sorted(
        owner for owner, description in descriptions.items()
        if description == "person" and owner not in set(people)
    )
    tie_count = sum(descriptions.get(owner) == "tie" for owner in owners)
    if (
        binding is None
        or base._hash(raw) != binding["node_sha256"]
        or raw.get("route_sha256") != binding["route_sha256"]
        or token_ids_sha256(raw.get("route_tokens", ())) != binding["route_sha256"]
        or base._hash(raw.get("intervention_transcript", ()))
        != binding["transcript_sha256"]
        or base._hash(people) != binding["family_sha256"]
        or missing != [binding["missing_person_owner_id"]]
        or int(raw.get("cumulative_intervention_count", -1)) != 4
        or int(raw.get("matched_person_count", -1)) != 37
        or int(raw.get("matched_owner_count", -1)) != 40
        or tie_count != 3
        or len(list(raw.get("intervention_transcript", ()))) != 4
        or any(dict(item.get("gate_debt", {})) for item in raw["intervention_transcript"])
    ):
        raise _hold(f"source node route/transcript/family binding drifted for {node_id}")
    return {
        **raw,
        "missing_person_owner_id": missing[0],
        "person_family_sha256": binding["family_sha256"],
        "matched_tie_count": tie_count,
        "source_node_sha256": binding["node_sha256"],
        "source_transcript_sha256": binding["transcript_sha256"],
    }


def _validate_source_receipt(
    receipt: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(receipt)
    bindings = dict(raw.get("bindings", {}))
    counts = dict(raw.get("counts", {}))
    resources = dict(raw.get("resources", {}))
    source = dict(raw.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    depths = list(raw.get("depths", ()))
    selected = (
        list(dict(depths[2]).get("selected_beam", ())) if len(depths) > 2 else []
    )
    if (
        raw.get("schema_version") != beam.SCHEMA_VERSION
        or raw.get("unit_id") != beam.UNIT_ID
        or raw.get("run_id") != SOURCE_RUN_ID
        or raw.get("status") != "beam_search_exhausted"
        or not source_path.is_file()
        or source.get("sha256") != SOURCE_RUNNER_SHA256
        or base._sha256(source_path) != SOURCE_RUNNER_SHA256
        or base._hash(bindings) != SOURCE_BINDINGS_SHA256
        or base._hash(counts) != SOURCE_COUNTS_SHA256
        or base._hash(resources) != SOURCE_RESOURCES_SHA256
        or base._hash(selected) != SOURCE_SELECTED_NODES_SHA256
        or len(depths) != 4
        or dict(depths[2]).get("cumulative_intervention_count") != 4
        or [item.get("node_id") for item in selected] != list(SOURCE_NODE_IDS)
        or dict(counts.get("total", {}))
        != {
            "branch_logit_forwards": 18,
            "replay_completions": 4,
            "warm_controlled_completions": 126,
        }
        or resources.get("predeclared_bound") != beam.RESOURCE_BOUND
        or int(resources.get("max_peak_cuda_reserved_bytes", -1))
        > beam.RESOURCE_BOUND["max_peak_cuda_reserved_bytes_per_rank"]
        or bindings.get("model_receipt_sha256") != recursive.MODEL_RECEIPT_SHA256
        or bindings.get("checkpoint") != str(recursive.START_CHECKPOINT)
        or bindings.get("checkpoint_readback") != contract["checkpoint_readback"]
        or bindings.get("start_surface_sha256") != recursive.START_SURFACE_SHA256
        or bindings.get("frozen_surface_sha256") != recursive.FROZEN_SURFACE_SHA256
        or bindings.get("authority_description_map_sha256")
        != recursive.AUTHORITY_DESCRIPTION_MAP_SHA256
        or bindings.get("control_parent_route_sha256")
        != recursive.CONTROL_PARENT_ROUTE_SHA256
        or bindings.get("prompt_token_ids_sha256") != base.PROMPT_TOKEN_SHA256
        or bindings.get("image_sha256") != base.IMAGE_SHA256
        or bindings.get("target_sha256") != base.TARGET_SHA256
        or bindings.get("authority") != contract["authority_source"]
    ):
        raise _hold("source binding/count/resource/surface snapshot drifted")
    nodes = [_validate_source_node(item, contract["description_map"]) for item in selected]
    if (
        [node["missing_person_owner_id"] for node in nodes].count("gt:2299:35") != 2
        or [node["missing_person_owner_id"] for node in nodes].count("gt:2299:18") != 2
        or {node["person_family_sha256"] for node in nodes}
        != set(SOURCE_FAMILY_SHA256)
    ):
        raise _hold("source node complementary family cardinality drifted")
    return {"receipt": raw, "nodes": nodes}


def _sealed_aliases_for_node(
    node: Mapping[str, Any], *, aliases: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str], parent_route_tokens: Sequence[int],
) -> list[dict[str, Any]]:
    missing = str(node.get("missing_person_owner_id", ""))
    owners = set(map(str, node.get("matched_owner_ids", ())))
    actual_missing = sorted(
        owner for owner, description in descriptions.items()
        if description == "person" and owner not in owners
    )
    if actual_missing != [missing]:
        raise _hold("node does not have exactly its bound sole missing authoritative person")
    result: list[dict[str, Any]] = []
    for alias in aliases:
        if str(alias.get("owner", "")) != missing:
            continue
        row = list(map(int, alias.get("token_ids", ())))
        global_match = dict(alias.get("global_match", {}))
        witness = ota._compile_alias_witness(parent_route_tokens, row)
        if (
            alias.get("admitted") is not True
            or global_match.get("passed") is not True
            or dict(global_match.get("debt", {}))
            or len(row) != base.ROW_TOKENS
            or tuple(row[:4]) != recursive.ROW_OPEN
            or row[-1] != ROW_CLOSE
            or base.EOS in row
            or token_ids_sha256(row) != alias.get("row_sha256")
            or token_ids_sha256(witness) != alias.get("witness_sha256")
            or not alias.get("name")
            or not alias.get("source")
        ):
            raise _hold(f"sealed alias row drifted for {missing}")
        result.append(deepcopy(dict(alias)))
    result.sort(key=lambda item: (str(item["row_sha256"]), str(item["name"])))
    if not result or len(result) > 8 or len({item["row_sha256"] for item in result}) != len(result):
        raise _hold(f"sealed alias coverage/budget drifted for {missing}")
    return result


def _splice_progressive_prefix(
    node_tokens: Sequence[int], row_tokens: Sequence[int], *, level: int,
) -> dict[str, Any]:
    node = list(map(int, node_tokens))
    row = list(map(int, row_tokens))
    forced_count = FORCED_ROW_TOKENS.get(level)
    coordinate_count = FORCED_COORDINATES.get(level)
    if (
        forced_count is None
        or coordinate_count is None
        or not node
        or node[-1] != base.EOS
        or base.EOS in node[:-1]
        or (len(node) - 1) % base.ROW_TOKENS
        or len(row) != base.ROW_TOKENS
        or tuple(row[:4]) != recursive.ROW_OPEN
        or row[-1] != ROW_CLOSE
        or base.EOS in row
    ):
        raise _hold("progressive source route/row/level is malformed")
    forced_row = row[:forced_count]
    start = len(node) - 1
    prefix = [*node[:-1], *forced_row]
    positions = list(range(start, start + forced_count))
    coordinate_positions = list(range(start + 4, start + 4 + coordinate_count))
    coordinate_tokens = row[4 : 4 + coordinate_count]
    if (
        [prefix[position] for position in positions] != forced_row
        or [prefix[position] for position in coordinate_positions] != coordinate_tokens
        or base.EOS in prefix
    ):
        raise _hold("progressive forced position binding drifted")
    return {
        "forced_prefix_tokens": prefix,
        "forced_prefix_sha256": token_ids_sha256(prefix),
        "forced_row_tokens": forced_row,
        "forced_row_token_count": forced_count,
        "forced_coordinate_tokens": coordinate_tokens,
        "forced_coordinate_count": coordinate_count,
        "forced_positions": positions,
        "forced_coordinate_positions": coordinate_positions,
    }


def _alias_provenance(alias: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owner": str(alias["owner"]),
        "name": str(alias["name"]),
        "source": str(alias["source"]),
        "row_sha256": str(alias["row_sha256"]),
        "witness_sha256": str(alias["witness_sha256"]),
        "row_tokens": list(map(int, alias["token_ids"])),
    }


def _build_level_tasks(
    nodes: Sequence[Mapping[str, Any]], *, aliases: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str], parent_route_tokens: Sequence[int], level: int,
) -> dict[str, Any]:
    if level not in LEVELS:
        raise _hold("progressive release level is outside 2..5")
    tasks: list[dict[str, Any]] = []
    frontiers: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for node_index, node in enumerate(nodes):
        sealed = _sealed_aliases_for_node(
            node,
            aliases=aliases,
            descriptions=descriptions,
            parent_route_tokens=parent_route_tokens,
        )
        grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for alias in sealed:
            forced = tuple(map(int, alias["token_ids"][: FORCED_ROW_TOKENS[level]]))
            grouped.setdefault(forced, []).append(_alias_provenance(alias))
        node_task_ids: list[str] = []
        for forced_row, provenance in sorted(grouped.items()):
            key = (str(node["node_id"]), forced_row)
            if key in seen:
                raise _hold("duplicate node/forced-prefix task escaped deduplication")
            seen.add(key)
            binding = _splice_progressive_prefix(
                node["route_tokens"], provenance[0]["row_tokens"], level=level,
            )
            provenance.sort(key=lambda item: (item["row_sha256"], item["name"]))
            identity = {
                "level": level,
                "node_id": str(node["node_id"]),
                "forced_row_tokens": list(forced_row),
            }
            task_index = len(tasks)
            task_id = f"level{level}-{base._hash(identity)[:24]}"
            task = {
                **identity,
                "task_index": task_index,
                "task_id": task_id,
                "node_index": node_index,
                "parent_route_sha256": str(node["route_sha256"]),
                "parent_owner_ids": list(map(str, node["matched_owner_ids"])),
                "missing_person_owner_id": str(node["missing_person_owner_id"]),
                "alias_provenance": provenance,
                "alias_provenance_sha256": base._hash(provenance),
                **binding,
            }
            tasks.append(task)
            node_task_ids.append(task_id)
        frontiers.append({
            "node_id": str(node["node_id"]),
            "missing_person_owner_id": str(node["missing_person_owner_id"]),
            "sealed_alias_count": len(sealed),
            "deduplicated_task_count": len(grouped),
            "task_ids": node_task_ids,
        })
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise _hold("progressive task IDs are not unique")
    return {"level": level, "frontiers": frontiers, "tasks": tasks}


def _build_all_levels(
    nodes: Sequence[Mapping[str, Any]], *, aliases: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str], parent_route_tokens: Sequence[int],
) -> dict[int, dict[str, Any]]:
    levels = {
        level: _build_level_tasks(
            nodes,
            aliases=aliases,
            descriptions=descriptions,
            parent_route_tokens=parent_route_tokens,
            level=level,
        )
        for level in LEVELS
    }
    counts = {level: len(levels[level]["tasks"]) for level in LEVELS}
    if counts != EXPECTED_LEVEL_TASK_COUNTS or sum(counts.values()) != MAX_PROGRESSIVE_TASKS:
        raise _hold("progressive deduplicated task-count budget drifted")
    return levels


def _coordinate_prefix_log_probability(
    logits: torch.Tensor, *, positions: Sequence[int], token_ids: Sequence[int],
) -> float:
    indices = list(map(int, positions))
    targets = list(map(int, token_ids))
    if (
        logits.ndim != 2
        or not indices
        or len(indices) != len(targets)
        or min(indices) < 0
        or max(indices) >= logits.shape[0]
        or min(targets) < 0
        or max(targets) >= logits.shape[1]
        or indices != sorted(indices)
    ):
        raise _hold("teacher log-probability positions/tokens are malformed")
    selected = torch.log_softmax(logits[indices].float(), dim=-1)
    value = selected[
        torch.arange(len(targets), device=selected.device),
        torch.tensor(targets, dtype=torch.long, device=selected.device),
    ].sum()
    if not bool(torch.isfinite(value).item()):
        raise _hold("teacher log-probability is non-finite")
    return float(value.item())


def _forced_completion(
    task: Mapping[str, Any], suffix_tokens: Sequence[int],
) -> dict[str, Any]:
    prefix = list(map(int, task.get("forced_prefix_tokens", ())))
    suffix = list(map(int, suffix_tokens))
    full = [*prefix, *suffix]
    forced_positions = list(map(int, task.get("forced_positions", ())))
    forced_row = list(map(int, task.get("forced_row_tokens", ())))
    if (
        not prefix
        or not suffix
        or suffix[-1] != base.EOS
        or base.EOS in suffix[:-1]
        or base.EOS in prefix
        or token_ids_sha256(prefix) != task.get("forced_prefix_sha256")
        or [full[position] for position in forced_positions] != forced_row
        or (len(full) - 1) % base.ROW_TOKENS
        or len(full) >= base.NATURAL_MAX_TOKENS
    ):
        raise _hold("forced progressive prefix did not release to row-aligned EOS")
    return {
        "forced_prefix_tokens": prefix,
        "forced_prefix_sha256": str(task["forced_prefix_sha256"]),
        "suffix_token_ids": suffix,
        "suffix_token_ids_sha256": token_ids_sha256(suffix),
        "generated_token_ids": full,
        "generated_token_ids_sha256": token_ids_sha256(full),
    }


def _progressive_gate(
    node_gate: Mapping[str, Any], *, matched_owner_ids: Sequence[str],
    descriptions: Mapping[str, str], parent_owner_ids: Sequence[str],
) -> dict[str, Any]:
    owners_list = list(map(str, matched_owner_ids))
    owners = set(owners_list)
    parent_list = list(map(str, parent_owner_ids))
    parent = set(parent_list)
    people = sorted(owner for owner in owners if descriptions.get(owner) == "person")
    ties = sorted(owner for owner in owners if descriptions.get(owner) == "tie")
    debt = dict(node_gate.get("debt", {}))
    extra = {
        "node_gate": node_gate.get("passed") is not True or bool(debt),
        "node_parent_binding": list(map(str, node_gate.get("parent_owner_ids", ())))
        != sorted(parent),
        "owner_identity": len(owners_list) != len(owners) or not owners.issubset(descriptions),
        "parent_identity": len(parent_list) != len(parent) or not parent,
        "parent_loss": not parent.issubset(owners),
        "not_proper_superset": not owners > parent,
        "not_38_person": len(people) != recursive.PERSON_OWNER_COUNT,
        "prior_tie_loss": any(descriptions.get(owner) == "tie" and owner not in owners for owner in parent),
    }
    debt.update({name: failed for name, failed in extra.items() if failed})
    return {
        "passed": not debt,
        "debt": debt,
        "node_gate": deepcopy(dict(node_gate)),
        "parent_owner_ids": sorted(parent),
        "matched_owner_ids": sorted(owners),
        "matched_person_owner_ids": people,
        "matched_person_count": len(people),
        "matched_tie_owner_ids": ties,
        "matched_tie_count": len(ties),
        "gained_owner_ids": sorted(owners - parent),
    }


def _candidate_record(
    *, task: Mapping[str, Any], suffix_tokens: Sequence[int],
    evaluation: Mapping[str, Any], descriptions: Mapping[str, str],
    coordinate_prefix_log_probability: float,
) -> dict[str, Any]:
    completion = _forced_completion(task, suffix_tokens)
    if list(map(int, evaluation.get("generated_token_ids", ()))) != completion["generated_token_ids"]:
        raise _hold("production evaluator route differs from progressive completion")
    owners = list(map(str, evaluation.get("matched_target_owner_ids", ())))
    node_gate = recursive._node_gate(
        evaluation, parent_owner_ids=task["parent_owner_ids"],
    )
    gate = _progressive_gate(
        node_gate,
        matched_owner_ids=owners,
        descriptions=descriptions,
        parent_owner_ids=task["parent_owner_ids"],
    )
    return {
        **{key: deepcopy(value) for key, value in task.items()},
        **completion,
        "coordinate_prefix_log_probability": float(coordinate_prefix_log_probability),
        "matched_owner_ids": gate["matched_owner_ids"],
        "matched_owner_count": len(gate["matched_owner_ids"]),
        "matched_person_owner_ids": gate["matched_person_owner_ids"],
        "matched_person_count": gate["matched_person_count"],
        "matched_tie_owner_ids": gate["matched_tie_owner_ids"],
        "matched_tie_count": gate["matched_tie_count"],
        "admitted": bool(gate["passed"]),
        "gate": gate,
        "evaluation": deepcopy(dict(evaluation)),
    }


def _evaluate_task(
    *, task: Mapping[str, Any], node: Mapping[str, Any], model: Any,
    tokenizer: Any, native_inputs: Mapping[str, Any], contract: Mapping[str, Any],
    raw_example: Any,
) -> dict[str, Any]:
    if (
        task.get("node_id") != node.get("node_id")
        or task.get("parent_route_sha256") != node.get("route_sha256")
        or task.get("parent_owner_ids") != node.get("matched_owner_ids")
        or task.get("missing_person_owner_id") != node.get("missing_person_owner_id")
    ):
        raise _hold("progressive task lost its node-specific parent")
    prefix = list(map(int, task["forced_prefix_tokens"]))
    with torch.inference_mode():
        logits = base.full_root._teacher_forced_route_logits(
            model=model,
            native_inputs=native_inputs,
            route_tokens=prefix,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        log_probability = _coordinate_prefix_log_probability(
            logits,
            positions=task["forced_coordinate_positions"],
            token_ids=task["forced_coordinate_tokens"],
        )
        suffix = base.full_root._greedy_release(
            model=model,
            native_inputs=native_inputs,
            prefix=prefix,
            eos_token_id=base.EOS,
            pad_token_id=int(tokenizer.pad_token_id),
        )
    completion = _forced_completion(task, suffix)
    evaluation = recursive.manifold._match_evaluation(
        tokenizer=tokenizer,
        tokens=completion["generated_token_ids"],
        target=contract["target"],
        witness_tokens=completion["generated_token_ids"],
        raw_example=raw_example,
        parent_owners=node["matched_owner_ids"],
        label=f"progressive-level-{task['level']}-{task['task_id']}",
    )
    return _candidate_record(
        task=task,
        suffix_tokens=suffix,
        evaluation=evaluation,
        descriptions=contract["description_map"],
        coordinate_prefix_log_probability=log_probability,
    )


def _verify_source_replay_gather(
    gathered: Sequence[Mapping[str, Any]], *, nodes: Sequence[Mapping[str, Any]],
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
        raise _hold("source warm replay collective or evaluation failed")
    by_index = {int(record.get("source_node_index", -1)): record for record in records}
    if sorted(by_index) != list(range(len(nodes))) or len(records) != len(nodes):
        raise _hold("four source transcripts were not distributed exactly once")
    ordered: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        record = by_index[index]
        if (
            int(record.get("rank", -1)) != index % WORLD_SIZE
            or record.get("source_replay_task_id")
            != f"source-replay-{index}-{node['node_id']}"
            or record.get("node_id") != node["node_id"]
            or int(record.get("completion_count", -1)) != 4
            or record.get("final_route_sha256") != node["route_sha256"]
            or record.get("final_owner_ids") != node["matched_owner_ids"]
            or dict(record.get("final_gate_debt", {}))
        ):
            raise _hold("source warm replay differs from bound node")
        previous = list(map(int, parent_route_tokens))
        replayed_records = list(record.get("records", ()))
        if len(replayed_records) != 4:
            raise _hold("source warm replay transcript cardinality drifted")
        for recorded, replayed in zip(
            node["intervention_transcript"], replayed_records, strict=True,
        ):
            try:
                recursive._verify_replay_record(
                    recorded, replayed, previous_tokens=previous,
                )
            except BaseException as error:
                raise _hold("source warm replay differs from sealed transcript") from error
            previous = list(map(int, recorded["generated_token_ids"]))
        ordered.append(record)
    return ordered


def _verify_level_gather(
    gathered: Sequence[Mapping[str, Any]], *, tasks: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]], level: int,
) -> list[dict[str, Any]]:
    ranks = [int(item.get("rank", -1)) for item in gathered]
    records = [dict(record) for item in gathered for record in item.get("records", ())]
    if (
        sorted(ranks) != list(range(WORLD_SIZE))
        or len(ranks) != len(set(ranks))
        or any(item.get("error") is not None for item in gathered)
        or any(record.get("evaluation_error") is not None for record in records)
    ):
        raise _hold(f"level {level} candidate evaluation failed")
    indices = [int(record.get("task_index", -1)) for record in records]
    if sorted(indices) != list(range(len(tasks))) or len(indices) != len(set(indices)):
        raise _hold(f"level {level} world8 tasks were not covered exactly once")
    by_index = {int(record["task_index"]): record for record in records}
    by_node = {str(node["node_id"]): node for node in nodes}
    ordered: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        record = by_index[index]
        node = by_node.get(str(task["node_id"]))
        gate = dict(record.get("gate", {}))
        if (
            node is None
            or int(record.get("rank", -1)) != index % WORLD_SIZE
            or record.get("task_id") != task["task_id"]
            or int(record.get("level", -1)) != level
            or record.get("node_id") != node["node_id"]
            or record.get("parent_route_sha256") != node["route_sha256"]
            or record.get("parent_owner_ids") != node["matched_owner_ids"]
            or list(map(str, gate.get("parent_owner_ids", ())))
            != sorted(node["matched_owner_ids"])
            or record.get("forced_row_tokens") != task["forced_row_tokens"]
            or record.get("forced_prefix_tokens") != task["forced_prefix_tokens"]
            or record.get("forced_prefix_sha256") != task["forced_prefix_sha256"]
            or record.get("forced_positions") != task["forced_positions"]
        ):
            raise _hold(f"level {level} candidate lost its node-specific parent/prefix")
        ordered.append(record)
    return ordered


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(candidate["matched_person_count"]),
        -int(candidate["matched_owner_count"]),
        int(candidate["forced_row_token_count"]),
        -float(candidate["coordinate_prefix_log_probability"]),
        str(candidate["generated_token_ids_sha256"]),
        str(candidate["alias_provenance_sha256"]),
        str(candidate.get("task_id", "")),
    )


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    admitted = [deepcopy(dict(item)) for item in candidates if item.get("admitted") is True]
    if not admitted:
        raise _hold("cannot select from an empty admitted candidate set")
    if any(int(item.get("matched_person_count", -1)) != recursive.PERSON_OWNER_COUNT for item in admitted):
        raise _hold("admitted candidate escaped the 38-person gate")
    return sorted(admitted, key=_candidate_sort_key)[0]


def _terminal_status(
    candidates_by_level: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[str, int | None]:
    for level in LEVELS:
        if any(item.get("admitted") is True for item in candidates_by_level.get(level, ())):
            return "controlled_38_person_success", level
    return "progressive_release_exhausted", None


def _verify_progressive_replay_record(
    recorded: Mapping[str, Any], replayed: Mapping[str, Any],
) -> None:
    exact_keys = (
        "level",
        "node_id",
        "parent_route_sha256",
        "parent_owner_ids",
        "missing_person_owner_id",
        "forced_row_tokens",
        "forced_row_token_count",
        "forced_coordinate_tokens",
        "forced_coordinate_count",
        "forced_prefix_tokens",
        "forced_prefix_sha256",
        "forced_positions",
        "forced_coordinate_positions",
        "suffix_token_ids",
        "suffix_token_ids_sha256",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "coordinate_prefix_log_probability",
        "matched_owner_ids",
        "matched_owner_count",
        "matched_person_owner_ids",
        "matched_person_count",
        "matched_tie_owner_ids",
        "matched_tie_count",
        "admitted",
    )
    recorded_debt = dict(dict(recorded.get("gate", {})).get("debt", {}))
    replayed_debt = dict(dict(replayed.get("gate", {})).get("debt", {}))
    if (
        any(recorded.get(key) != replayed.get(key) for key in exact_keys)
        or recorded_debt != replayed_debt
    ):
        raise _hold("progressive replay differs from selected candidate")


def _cold_replay(
    *, selected: Mapping[str, Any], node: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(recursive.START_CHECKPOINT)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if (
            type(opened) is not base.HFBackendSession
            or opened._model is None
            or opened._tokenizer is None
        ):
            raise _hold("fresh replay requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
            setup["requests"][:1]
        )
        names, parameters = ota._step_trainable_surface(model, r32_step=True)
        surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
        if (
            token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
            or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
            or surface.get("aggregate_sha256") != recursive.START_SURFACE_SHA256
            or base._frozen_surface(model, names) != recursive.FROZEN_SURFACE_SHA256
        ):
            raise _hold("fresh replay source/surface identity drifted")
        previous_tokens = list(map(int, contract["parent_route_tokens"]))
        previous_owners = list(map(str, contract["parent_owner_ids"]))
        source_records: list[dict[str, Any]] = []
        for recorded in node["intervention_transcript"]:
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
                raw_example=setup["raw_example"],
                parent_owners=previous_owners,
                label=f"progressive-fresh-source-depth-{recorded['depth']}",
            )
            source_gate = recursive._node_gate(
                evaluation, parent_owner_ids=previous_owners,
            )
            owners = list(source_gate["matched_owner_ids"])
            replayed = {
                **completion,
                "matched_owner_ids": owners,
                "matched_person_count": recursive._person_count(
                    owners, contract["description_map"],
                ),
                "matched_owner_count": len(owners),
                "gate_debt": dict(source_gate["debt"]),
            }
            try:
                recursive._verify_replay_record(
                    recorded, replayed, previous_tokens=previous_tokens,
                )
            except BaseException as error:
                raise _hold("fresh source replay differs from sealed transcript") from error
            source_records.append(replayed)
            previous_tokens = completion["generated_token_ids"]
            previous_owners = owners
        if (
            token_ids_sha256(previous_tokens) != node["route_sha256"]
            or previous_owners != node["matched_owner_ids"]
        ):
            raise _hold("fresh source replay final node identity drifted")
        progressive = _evaluate_task(
            task=selected,
            node=node,
            model=model,
            tokenizer=tokenizer,
            native_inputs=native_inputs,
            contract=contract,
            raw_example=setup["raw_example"],
        )
        _verify_progressive_replay_record(selected, progressive)
        if (
            progressive.get("admitted") is not True
            or int(progressive.get("matched_person_count", -1)) != recursive.PERSON_OWNER_COUNT
            or dict(dict(progressive.get("gate", {})).get("debt", {}))
            or any(parameter.grad is not None for parameter in model.parameters())
        ):
            raise _hold("fresh progressive replay lost success/debt/frozen-state identity")
        return {
            "source_records": source_records,
            "progressive_record": progressive,
            "completion_count": len(source_records) + 1,
            "final_generated_token_ids": progressive["generated_token_ids"],
            "final_generated_token_ids_sha256": progressive["generated_token_ids_sha256"],
            "final_matched_owner_ids": progressive["matched_owner_ids"],
            "final_matched_person_count": progressive["matched_person_count"],
            "final_matched_tie_count": progressive["matched_tie_count"],
            "final_gate_debt": dict(progressive["gate"]["debt"]),
            "surface": surface,
            "runtime": opened.receipt.to_artifact_dict(),
        }


def _enforce_budget(
    counts: Mapping[str, int], *, launched_task_count: int, success: bool,
) -> None:
    expected_keys = {
        "source_warm_replay_completions",
        "progressive_warm_completions",
        "teacher_logprob_forwards",
        "fresh_replay_completions",
    }
    source = int(counts.get("source_warm_replay_completions", -1))
    progressive = int(counts.get("progressive_warm_completions", -1))
    forwards = int(counts.get("teacher_logprob_forwards", -1))
    replay = int(counts.get("fresh_replay_completions", -1))
    if (
        set(counts) != expected_keys
        or source != SOURCE_TRANSCRIPT_COMPLETIONS
        or progressive != launched_task_count
        or forwards != launched_task_count
        or launched_task_count < EXPECTED_LEVEL_TASK_COUNTS[2]
        or launched_task_count > MAX_PROGRESSIVE_TASKS
        or source + progressive > MAX_WARM_COMPLETIONS
        or replay != (MAX_REPLAY_COMPLETIONS if success else 0)
        or replay > MAX_REPLAY_COMPLETIONS
    ):
        raise _hold("progressive-release execution budget drifted")


def _require_world8(value: int) -> None:
    if value != WORLD_SIZE:
        raise _hold("progressive release requires torchrun --nproc_per_node=8")


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
    levels = _build_all_levels(
        source["nodes"],
        aliases=contract["aliases"],
        descriptions=contract["description_map"],
        parent_route_tokens=contract["parent_route_tokens"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_cuda",
        "source_receipt": str(SOURCE_RECEIPT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_node_ids": list(SOURCE_NODE_IDS),
        "source_node_bindings": SOURCE_NODE_BINDINGS,
        "source_family_sha256": list(SOURCE_FAMILY_SHA256),
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
        "person_owner_count": recursive.PERSON_OWNER_COUNT,
        "source_matched_tie_count": 3,
        "levels": list(LEVELS),
        "level_task_counts": {
            str(level): len(levels[level]["tasks"]) for level in LEVELS
        },
        "world_size": WORLD_SIZE,
        "resource_bound": RESOURCE_BOUND,
    }


def run(*, run_id: str) -> Path:
    _require_world8(int(os.environ.get("WORLD_SIZE", "0")))
    contract = recursive._start_contract()
    source = _validate_source_receipt(_load_source_receipt(), contract)
    nodes = source["nodes"]
    level_specs = _build_all_levels(
        nodes,
        aliases=contract["aliases"],
        descriptions=contract["description_map"],
        parent_route_tokens=contract["parent_route_tokens"],
    )
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    stage = "distributed_initialization"
    owns_output = False
    source_snapshot: Mapping[str, Any] | None = None
    source_replay: list[dict[str, Any]] = []
    level_records: list[dict[str, Any]] = [
        {
            "level": level,
            "launch_status": "pending",
            "frontiers": deepcopy(level_specs[level]["frontiers"]),
            "tasks": deepcopy(level_specs[level]["tasks"]),
            "candidate_records": [],
            "admitted_task_ids": [],
            "selected_task_id": None,
        }
        for level in LEVELS
    ]
    selected: dict[str, Any] | None = None
    selected_node: dict[str, Any] | None = None
    selected_level: int | None = None
    status: str | None = None
    replay: Mapping[str, Any] | None = None
    local_counts = {
        "source_warm_replay_completions": 0,
        "progressive_warm_completions": 0,
        "teacher_logprob_forwards": 0,
        "fresh_replay_completions": 0,
    }
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
        stage = "warm_source_replay_and_progressive_levels"
        setup = contract["setup"]
        initial_surface: Mapping[str, Any] | None = None
        warm_runtime: Mapping[str, Any] | None = None
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("warm release requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            if (
                token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path))
                != base.IMAGE_SHA256
            ):
                raise _hold("warm prompt/image identity drifted")
            names, parameters = ota._step_trainable_surface(model, r32_step=True)
            initial_surface = base.full_root._full_root_surface_snapshot(
                names, parameters,
            )[1]
            frozen_before = base._frozen_surface(model, names)
            if (
                initial_surface.get("aggregate_sha256") != recursive.START_SURFACE_SHA256
                or frozen_before != recursive.FROZEN_SURFACE_SHA256
            ):
                raise _hold("warm model surface identity drifted")

            local_source_records: list[dict[str, Any]] = []
            for source_index in range(rank, len(nodes), WORLD_SIZE):
                node = nodes[source_index]
                try:
                    record = beam._warm_replay_seed(
                        seed_index=source_index,
                        seed=node,
                        model=model,
                        tokenizer=tokenizer,
                        native_inputs=native_inputs,
                        contract=contract,
                        raw_example=setup["raw_example"],
                    )
                    record.update({
                        "source_node_index": source_index,
                        "source_replay_task_id": (
                            f"source-replay-{source_index}-{node['node_id']}"
                        ),
                    })
                except BaseException as error:
                    record = {
                        "source_node_index": source_index,
                        "evaluation_error": {
                            "type": type(error).__name__,
                            "error": str(error),
                        },
                    }
                record["rank"] = rank
                local_source_records.append(record)
                local_counts["source_warm_replay_completions"] += 4
            gathered_source: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_source, {
                "rank": rank,
                "records": local_source_records,
                "error": None,
            })
            source_replay = base._rank0_call(lambda: _verify_source_replay_gather(
                gathered_source,
                nodes=nodes,
                parent_route_tokens=contract["parent_route_tokens"],
            ))
            ota._agree_hash(source_replay, label="progressive-source-warm-replay")

            candidates_by_level: dict[int, list[dict[str, Any]]] = {
                level: [] for level in LEVELS
            }
            for level in LEVELS:
                stage = f"warm_progressive_level_{level}"
                tasks = level_specs[level]["tasks"]
                local_records: list[dict[str, Any]] = []
                for task_index in range(rank, len(tasks), WORLD_SIZE):
                    task = tasks[task_index]
                    node = nodes[int(task["node_index"])]
                    try:
                        record = _evaluate_task(
                            task=task,
                            node=node,
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
                            "node_id": task["node_id"],
                            "evaluation_error": {
                                "type": type(error).__name__,
                                "error": str(error),
                            },
                        }
                    record["rank"] = rank
                    local_records.append(record)
                    local_counts["progressive_warm_completions"] += 1
                    local_counts["teacher_logprob_forwards"] += 1
                gathered: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered, {
                    "rank": rank,
                    "records": local_records,
                    "error": None,
                })
                candidates = base._rank0_call(lambda level=level, tasks=tasks: (
                    _verify_level_gather(
                        gathered, tasks=tasks, nodes=nodes, level=level,
                    )
                ))
                ota._agree_hash(
                    candidates, label=f"progressive-level-{level}-candidates",
                )
                candidates_by_level[level] = candidates
                record = level_records[LEVELS.index(level)]
                record.update({
                    "launch_status": "completed",
                    "candidate_records": candidates,
                    "admitted_task_ids": [
                        item["task_id"] for item in candidates
                        if item.get("admitted") is True
                    ],
                })
                admitted = [item for item in candidates if item.get("admitted") is True]
                if admitted:
                    selected = _select_candidate(admitted)
                    selected_node = nodes[int(selected["node_index"])]
                    selected_level = level
                    record["selected_task_id"] = selected["task_id"]
                    for later in LEVELS[LEVELS.index(level) + 1 :]:
                        level_records[LEVELS.index(later)]["launch_status"] = (
                            "not_launched_after_first_success"
                        )
                    break
            status, first_level = _terminal_status(candidates_by_level)
            if first_level != selected_level:
                raise _hold("first admitted level/selection stop drifted")
            if (
                status == "controlled_38_person_success"
                and (selected is None or selected_node is None)
            ):
                raise _hold("success has no selected candidate/node")
            if (
                status == "progressive_release_exhausted"
                and any(item["launch_status"] != "completed" for item in level_records)
            ):
                raise _hold("exhaustion did not execute all four levels")
            after_surface = base.full_root._full_root_surface_snapshot(
                names, parameters,
            )[1]
            if (
                after_surface != initial_surface
                or base._frozen_surface(model, names) != frozen_before
                or any(parameter.grad is not None for parameter in model.parameters())
            ):
                raise _hold("inference-only warm release mutated model state")
            warm_runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, parameters, names
        torch.cuda.empty_cache()
        dist.barrier()
        if status == "controlled_38_person_success":
            assert selected is not None and selected_node is not None
            stage = "fresh_rank0_complete_lineage_replay"
            replay = base._rank0_call(lambda: _cold_replay(
                selected=selected,
                node=selected_node,
                contract=contract,
            ))
            if rank == 0:
                local_counts["fresh_replay_completions"] = MAX_REPLAY_COMPLETIONS
            ota._agree_hash(replay, label="progressive-fresh-rank0-replay")
            if (
                replay.get("completion_count") != MAX_REPLAY_COMPLETIONS
                or replay.get("final_generated_token_ids")
                != selected["generated_token_ids"]
                or replay.get("final_generated_token_ids_sha256")
                != selected["generated_token_ids_sha256"]
                or replay.get("final_matched_owner_ids")
                != selected["matched_owner_ids"]
                or replay.get("final_matched_person_count")
                != recursive.PERSON_OWNER_COUNT
                or dict(replay.get("final_gate_debt", {}))
                or replay.get("surface") != initial_surface
                or replay.get("runtime") != warm_runtime
            ):
                raise _hold("fresh complete-lineage replay identity drifted")

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
        launched_task_count = sum(
            len(level_specs[level]["tasks"])
            for level in LEVELS
            if level_records[LEVELS.index(level)]["launch_status"] == "completed"
        )
        _enforce_budget(
            total_counts,
            launched_task_count=launched_task_count,
            success=status == "controlled_38_person_success",
        )
        elapsed = time.monotonic() - started
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or max(int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources)
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes_per_rank"]
        ):
            raise _hold("progressive-release resource bound exceeded")
        if status not in TERMINAL_STATUSES:
            raise _hold("progressive release exited without a terminal status")
        if rank == 0:
            final_evaluation = (
                None if selected is None else dict(selected["evaluation"])
            )
            final_ledger = (
                None if final_evaluation is None
                else dict(final_evaluation.get("ledger", {}))
            )
            progressive_transcript = (
                None if selected is None
                else {
                    key: deepcopy(value)
                    for key, value in selected.items()
                    if key != "evaluation"
                }
            )
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": status,
                "run_id": run_id,
                "runner_source_snapshot": source_snapshot,
                "bindings": {
                    "source_receipt": str(SOURCE_RECEIPT),
                    "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                    "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
                    "source_counts_sha256": SOURCE_COUNTS_SHA256,
                    "source_resources_sha256": SOURCE_RESOURCES_SHA256,
                    "source_selected_nodes_sha256": SOURCE_SELECTED_NODES_SHA256,
                    "source_runner_sha256": SOURCE_RUNNER_SHA256,
                    "source_node_bindings": SOURCE_NODE_BINDINGS,
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
                    "control_parent_route_sha256": (
                        recursive.CONTROL_PARENT_ROUTE_SHA256
                    ),
                },
                "protocol": {
                    "world_size": WORLD_SIZE,
                    "source_replay": (
                        "each of four intervention-4 source transcripts warm-replayed "
                        "once from clean Parent33"
                    ),
                    "levels": {
                        "2": "force row through y1; release x2,y2,row_close,EOS",
                        "3": "force row through x2; release y2,row_close,EOS",
                        "4": "force row through y2; release row_close,EOS",
                        "5": "force complete row; release EOS",
                    },
                    "deduplication": (
                        "exact (node_id, forced row-token tuple), retaining all alias provenance"
                    ),
                    "selection": [
                        "matched_person_count",
                        "matched_owner_count",
                        "fewer_forced_row_tokens",
                        "higher_exact_coordinate_prefix_teacher_log_probability",
                        "deterministic_route_and_alias_identity",
                    ],
                    "first_admitted_level_stop": True,
                    "owner_union": False,
                    "gradient": None,
                    "optimizer": None,
                    "weight_update": False,
                    "checkpoint_save": False,
                },
                "source_nodes": nodes,
                "source_warm_replay": source_replay,
                "levels": level_records,
                "selection": {
                    "selected_level": selected_level,
                    "selected_node_id": (
                        None if selected_node is None else selected_node["node_id"]
                    ),
                    "selected_task_id": (
                        None if selected is None else selected["task_id"]
                    ),
                    "candidate": selected,
                },
                "intervention_transcript": (
                    [] if selected_node is None
                    else [
                        *deepcopy(selected_node["intervention_transcript"]),
                        progressive_transcript,
                    ]
                ),
                "final": (
                    None if selected is None
                    else {
                        "generated_token_ids": selected["generated_token_ids"],
                        "generated_token_ids_sha256": selected[
                            "generated_token_ids_sha256"
                        ],
                        "matched_owner_ids": selected["matched_owner_ids"],
                        "matched_owner_count": selected["matched_owner_count"],
                        "matched_person_owner_ids": selected[
                            "matched_person_owner_ids"
                        ],
                        "matched_person_count": selected["matched_person_count"],
                        "matched_tie_owner_ids": selected["matched_tie_owner_ids"],
                        "matched_tie_count": selected["matched_tie_count"],
                        "gate": selected["gate"],
                        "parsed_rows": list(
                            dict(final_ledger.get("parse", {})).get("predictions", ())
                        ),
                        "ledger": final_ledger,
                    }
                ),
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
                "runtime": {
                    "warm": warm_runtime,
                    "fresh_rank0": None if replay is None else replay["runtime"],
                },
                "wall_time_seconds": elapsed,
                "claim_boundary": (
                    "One frozen-r32 controlled Image2299 38-person trajectory only; "
                    "not ordinary greedy learning, transfer, exhaustive coordinate "
                    "search, or recovery of all eight ties."
                ),
            }
            base._atomic_json(output / "receipt.json", receipt)
            if (
                (output / "receipt.json").stat().st_size
                > RESOURCE_BOUND["output_artifact_bytes_max"]
            ):
                raise _hold("progressive receipt exceeded artifact bound")
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
                "source_node_ids": list(SOURCE_NODE_IDS),
                "source_warm_replay_partial": source_replay,
                "levels_partial": level_records,
                "selection_partial": selected,
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
