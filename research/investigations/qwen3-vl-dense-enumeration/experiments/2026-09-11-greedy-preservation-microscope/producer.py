"""Bounded Stable50 versus positive32 teacher-forced preservation microscope.

``prepare`` and ``verify-preparation`` are CPU-only.  ``launch`` and ``rank``
are GPU-bearing and require the root's later runtime grant.  The GPU path makes
one score forward per checkpoint and image, never generates and never runs a
backward pass.  Full-vocabulary Stable50 probabilities exist only as a bounded
detached per-rank in-memory cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
SOURCE_ROOT = BASE / "2026-09-11-positive-branch-vs-repeat-event"
RAW_ROOT = BASE / "2026-09-11-greedy-preservation-microscope"
PREPARATION = RAW_ROOT / "preparation"
HERE = Path(__file__).resolve().parent
UNIT = HERE / "unit.md"

TRAINER_INPUTS = SOURCE_ROOT / "trainer-preparation" / "inputs.json"
MANIFEST = SOURCE_ROOT / "input-preparation" / "candidate_manifest.json"
ENDPOINT_PACKET = SOURCE_ROOT / "endpoint-preparation" / "packet.json"
ENDPOINT_CONSUMER = SOURCE_ROOT / "endpoint-A" / "consumer.json"
FULL_A_RECEIPT = SOURCE_ROOT / "full-A" / "receipt.json"
TRAINER = Path("probes/dora_owner_learning/repeat_recovery_train.py").resolve()
CONFIG = Path("probes/dora_owner_learning/configs/source256.yaml").resolve()

EXPECTED_HASHES = {
    str(TRAINER_INPUTS): "f10e82fa7106c2c0a24f5f8612299c28e68fdfbbef1c72539fb869efe5d3a83e",
    str(MANIFEST): "2870e777965007b5b06487fbb33b8408d1a992f4c90e2b4bd1f0f564c1e4aa3b",
    str(ENDPOINT_PACKET): "560006e73f3f0fc416e7d58751fe96c936aba9478fb6b320ab6118db2bcd5053",
    str(ENDPOINT_CONSUMER): "d61454f058793fd40338f19d1ebeac9023a3889df58912c3b960224dd8b91a37",
    str(FULL_A_RECEIPT): "dd14419bf11a07aa36c783b26207addf0e4294242d13e4d6219d10674dfeaa40",
    str(TRAINER): "bc6d1728dc5c156b6d617ecb9b99a71b1e81ee305e338c2f1476b3cdc9035308",
}
RUNTIME_FILES = (
    CONFIG,
    Path("probes/dora_owner_learning/runtime.py").resolve(),
    Path("src/qwen/native.py").resolve(),
    Path("src/inference/runtime.py").resolve(),
    Path("src/qwen/runtime_loading.py").resolve(),
    Path("src/qwen/special_token_embeddings.py").resolve(),
)

SCHEMA = "greedy_preservation_microscope.inputs.v1"
SHARD_SCHEMA = "greedy_preservation_microscope.shard.v1"
CONSUMER_SCHEMA = "greedy_preservation_microscope.consumer.v1"
WORLD_SIZE = 8
CASES_PER_WORKER = 7
TOTAL_CASES = 56
TOTAL_ACTION_TOKENS = 6056
TOTAL_PROTECTED_POSITIONS = 6047
MAX_RANK_SECONDS = 600
MAX_CUDA_BYTES = 24 * 1024**3
MAX_RSS_BYTES = 24 * 1024**3
MAX_REFERENCE_CACHE_BYTES = 2 * 1024**3
NEAR_TIE_EPSILON = 1e-3
EOS = 151645
PAD = 151643


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def digest_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def publish(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    require(load_json(path) == payload, f"published payload changed on reload: {path}")


def first_divergence(old: Sequence[int], new: Sequence[int]) -> dict[str, Any] | None:
    for position, (old_id, new_id) in enumerate(zip(old, new)):
        if old_id != new_id:
            return {
                "position": position,
                "common_prefix_length": position,
                "shared_prefix_ids_sha256": digest_ids(old[:position]),
                "old_token_id": int(old_id),
                "new_token_id": int(new_id),
                "kind": "token_substitution",
            }
    if len(old) == len(new):
        return None
    position = min(len(old), len(new))
    return {
        "position": position,
        "common_prefix_length": position,
        "shared_prefix_ids_sha256": digest_ids(old[:position]),
        "old_token_id": int(old[position]) if position < len(old) else None,
        "new_token_id": int(new[position]) if position < len(new) else None,
        "kind": "new_ended" if len(new) < len(old) else "old_ended",
    }


def _source_fields(logits: torch.Tensor, source_ids: Sequence[int]) -> list[dict[str, Any]]:
    require(logits.dtype == torch.float32 and logits.ndim == 2, "score logits must be FP32 rank2")
    targets = torch.tensor(list(source_ids), dtype=torch.long, device=logits.device)
    require(logits.shape[0] == targets.numel() and logits.shape[1] > 1, "source/logit alignment")
    require(bool(torch.isfinite(logits).all()), "nonfinite score logits")
    require(bool(((targets >= 0) & (targets < logits.shape[1])).all()), "source token vocabulary")
    logp = torch.log_softmax(logits, dim=-1)
    source_logp = logp.gather(1, targets[:, None]).squeeze(1)
    source_logits = logits.gather(1, targets[:, None]).squeeze(1)
    top_values, top_ids = logits.max(dim=-1)
    two = torch.topk(logits, k=2, dim=-1)
    best_other = torch.where(two.indices[:, 0] == targets, two.values[:, 1], two.values[:, 0])
    margins = source_logits - best_other
    values = torch.stack((source_logp, source_logits, margins), dim=1).detach().cpu().tolist()
    winners = top_ids.detach().cpu().tolist()
    return [
        {
            "position": position,
            "source_token_id": int(source_ids[position]),
            "source_logprob": float(values[position][0]),
            "source_logit": float(values[position][1]),
            "source_margin": float(values[position][2]),
            "top1_id": int(winners[position]),
            "source_is_argmax": int(source_ids[position]) == int(winners[position]),
            "near_tie": abs(float(values[position][2])) <= NEAR_TIE_EPSILON,
        }
        for position in range(len(source_ids))
    ]


def reference_snapshot(
    logits: torch.Tensor, source_ids: Sequence[int], protected_positions: Sequence[int]
) -> dict[str, Any]:
    protected = list(protected_positions)
    require(
        protected == sorted(set(protected))
        and all(type(position) is int and 0 <= position < len(source_ids) for position in protected),
        "protected position identity",
    )
    fields = _source_fields(logits, source_ids)
    with torch.no_grad():
        protected_logp = torch.log_softmax(logits[protected], dim=-1).detach().cpu()
    require(
        protected_logp.dtype == torch.float32
        and protected_logp.shape == (len(protected), logits.shape[1])
        and not protected_logp.requires_grad
        and protected_logp.grad_fn is None,
        "detached FP32 reference probability cache",
    )
    for row in fields:
        row["protected"] = row["position"] in set(protected)
    return {"protected_logp": protected_logp, "positions": fields}


def compare_snapshot(
    current_logits: torch.Tensor,
    source_ids: Sequence[int],
    protected_positions: Sequence[int],
    reference_logp: torch.Tensor,
    reference_positions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protected = list(protected_positions)
    require(
        reference_logp.dtype == torch.float32
        and reference_logp.shape == (len(protected), current_logits.shape[1]),
        "reference/current vocabulary alignment",
    )
    require(len(reference_positions) == len(source_ids), "reference position coverage")
    current = _source_fields(current_logits, source_ids)
    with torch.no_grad():
        current_logp = torch.log_softmax(current_logits[protected], dim=-1)
        kl = torch.nn.functional.kl_div(
            current_logp, reference_logp.to(current_logits.device),
            reduction="none", log_target=True,
        ).sum(-1).detach().cpu()
    require(bool(torch.isfinite(kl).all()) and bool((kl >= -1e-5).all()), "finite reference KL")
    kl_by_position = {position: float(value) for position, value in zip(protected, kl.tolist())}
    rows = []
    for reference, observed in zip(reference_positions, current, strict=True):
        position = int(reference["position"])
        require(
            position == observed["position"]
            and reference["source_token_id"] == observed["source_token_id"],
            "reference/current token position identity",
        )
        reference_argmax = bool(reference["source_is_argmax"])
        rows.append({
            "position": position,
            "source_token_id": int(reference["source_token_id"]),
            "protected": position in kl_by_position,
            "reference_kl": kl_by_position.get(position),
            "reference_source_logprob": float(reference["source_logprob"]),
            "current_source_logprob": float(observed["source_logprob"]),
            "reference_source_margin": float(reference["source_margin"]),
            "current_source_margin": float(observed["source_margin"]),
            "reference_top1_id": int(reference["top1_id"]),
            "current_top1_id": int(observed["top1_id"]),
            "reference_source_is_argmax": reference_argmax,
            "current_source_is_argmax": bool(observed["source_is_argmax"]),
            "source_argmax_retained": (
                bool(observed["source_is_argmax"]) if reference_argmax else None
            ),
            "near_tie_reference": bool(reference["near_tie"]),
            "near_tie_current": bool(observed["near_tie"]),
        })
    return {"positions": rows}


def token_category(token: str) -> str:
    if token.startswith("<|coord_") and token.endswith("|>"):
        return "coordinate"
    exact = {
        "<|object_ref_start|>": "object_ref_start",
        "<|object_ref_end|>": "object_ref_end",
        "<|box_start|>": "box_start",
        "<|box_end|>": "box_end",
        "<|im_end|>": "im_end",
    }
    return exact.get(token, "description_or_content")


def _slim_adapter(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {key: identity[key] for key in (
        "root", "kind", "version", "fingerprint", "file_count", "files", "semantic_identity"
    )}


def _verify_payload(identity: Mapping[str, Any], label: str) -> None:
    root = Path(identity["root"])
    files = identity["files"]
    require(identity["file_count"] == len(files) and files, f"{label} file count")
    for row in files:
        path = root / row["relative_path"]
        require(
            path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and file_hash(path) == row["sha256"],
            f"{label} payload changed: {path}",
        )


def _runtime_identity() -> list[dict[str, Any]]:
    paths = (Path(__file__).resolve(), UNIT, TRAINER, *RUNTIME_FILES)
    return [{"path": str(path), "sha256": file_hash(path)} for path in paths]


def assemble_input_packet() -> dict[str, Any]:
    from probes.dora_owner_learning.repeat_recovery_train import validate_inputs

    for path, expected in EXPECTED_HASHES.items():
        require(file_hash(path) == expected, f"accepted source hash changed: {path}")
    trainer_inputs, manifest = validate_inputs(TRAINER_INPUTS, verify_sources=True)
    require(trainer_inputs["counts"] == manifest["counts"], "trainer/manifest counts")
    endpoint_packet = load_json(ENDPOINT_PACKET)
    endpoint_rows = load_json(ENDPOINT_CONSUMER)
    full_a = load_json(FULL_A_RECEIPT)
    require(
        endpoint_packet.get("schema") == "positive_branch_vs_repeat_event.endpoint_packet.v1"
        and endpoint_packet.get("status") == "candidate_ready_for_exported_adapters_no_gpu_grant"
        and len(endpoint_packet["eval_records"]) == 384,
        "accepted endpoint packet",
    )
    require(
        isinstance(endpoint_rows, list) and len(endpoint_rows) == 384
        and full_a.get("status") == "completed"
        and full_a.get("arm") == "A"
        and full_a.get("updates") == 32
        and full_a["source_adapter"] == trainer_inputs["stable50_adapter"],
        "accepted positive32 endpoint/model receipt",
    )
    stable_by_id = {str(row["image_id"]): row for row in endpoint_packet["eval_records"]}
    current_by_id = {str(row["image_id"]): row for row in endpoint_rows}
    require(len(stable_by_id) == len(current_by_id) == 384, "unique endpoint image identities")

    cases = []
    totals = defaultdict(int)
    for source in manifest["normals"]["cases"]:
        image_id = str(source["image_id"])
        stable = stable_by_id.get(image_id)
        current = current_by_id.get(image_id)
        require(stable is not None and current is not None, f"missing natural endpoint: {image_id}")
        require(
            stable["split"] == "reference56"
            and stable["example_id"] == source["example_id"]
            and stable["stable_ids"] == source["action_ids"]
            and stable["prompt_token_ids"] == source["prompt_token_ids"]
            and current["split"] == "reference56"
            and current["example_id"] == source["example_id"]
            and current["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"]
            and current["executed_media_sha256"] == source["image"]["executed_media_sha256"]
            and current["observed_image_grid_thw"] == source["image"]["observed_image_grid_thw"]
            and current["packet_sha256"] == EXPECTED_HASHES[str(ENDPOINT_PACKET)]
            and current["adapter_receipt_sha256"] == EXPECTED_HASHES[str(FULL_A_RECEIPT)]
            and current["adapter_fingerprint"] == full_a["saved_adapter"]["fingerprint"],
            f"old/new natural identity mismatch: {image_id}",
        )
        positions = list(source["initial_layout"]["kl_positions"])
        divergence = first_divergence(source["action_ids"], current["action_ids"])
        if divergence is not None:
            divergence["protected"] = divergence["position"] in positions
        stable_owners = sorted(map(str, stable["stable_score"]["50"]["owners"]))
        current_owners = sorted(map(str, current["score"]["50"]["owners"]))
        stable_set, current_set = set(stable_owners), set(current_owners)
        natural = {
            "stable_action_ids_sha256": digest_ids(source["action_ids"]),
            "current_action_ids": list(current["action_ids"]),
            "current_action_ids_sha256": digest_ids(current["action_ids"]),
            "first_divergence": divergence,
            "stable_iou50_owners": stable_owners,
            "current_iou50_owners": current_owners,
            "lost_iou50_owners": sorted(stable_set - current_set),
            "gained_iou50_owners": sorted(current_set - stable_set),
            "stable_strict_repeats": int(stable["stable_score"]["strict_repeats"]),
            "current_strict_repeats": int(current["score"]["strict_repeats"]),
            "stable_parser_drops": int(stable["stable_score"]["parser_drops"]),
            "current_parser_drops": int(current["score"]["parser_drops"]),
            "stable_stop_reason": stable["stable_score"]["stop_reason"],
            "current_stop_reason": current["score"]["stop_reason"],
        }
        case = {
            "key": source["key"],
            "source_index": int(source["source_index"]),
            "example_id": source["example_id"],
            "image_id": image_id,
            "source_case": source,
            "natural": natural,
        }
        cases.append(case)
        totals["stable_tp"] += len(stable_owners)
        totals["positive32_tp"] += len(current_owners)
        totals["lost_owners"] += len(natural["lost_iou50_owners"])
        totals["gained_owners"] += len(natural["gained_iou50_owners"])
        for field in (
            "stable_strict_repeats", "current_strict_repeats",
            "stable_parser_drops", "current_parser_drops",
        ):
            totals[field] += natural[field]
        totals["identical_natural_actions"] += divergence is None
    require([case["source_index"] for case in cases] == list(range(TOTAL_CASES)), "reference order")
    shards = [[case["key"] for case in cases[rank::WORLD_SIZE]] for rank in range(WORLD_SIZE)]
    case_by_key = {case["key"]: case for case in cases}
    base_config = load_json(Path(trainer_inputs["model_identity"]["base_model"]) / "config.json")
    text_config = base_config.get("text_config", base_config)
    vocab_size = int(text_config["vocab_size"])
    shard_plan = []
    for rank, shard in enumerate(shards):
        action_count = sum(len(case_by_key[key]["source_case"]["action_ids"]) for key in shard)
        protected_count = sum(
            len(case_by_key[key]["source_case"]["initial_layout"]["kl_positions"])
            for key in shard
        )
        shard_plan.append({
            "rank": rank, "cases": len(shard), "action_tokens": action_count,
            "protected_positions": protected_count,
            "estimated_reference_cache_bytes": protected_count * vocab_size * 4,
        })
    require(max(row["estimated_reference_cache_bytes"] for row in shard_plan)
            <= MAX_REFERENCE_CACHE_BYTES, "planned reference cache ceiling")
    exclusion = manifest["normals"]["mask_summary"]["geometry_invalid_excluded_positions"]
    packet = {
        "schema": SCHEMA,
        "status": "prepared_no_model_execution",
        "question": "Did small mean reference KL coexist with greedy margin crossings at actual first natural divergence and owner loss?",
        "trajectory_surface": "original Stable50 literal reference trajectory under teacher forcing",
        "near_tie_epsilon": NEAR_TIE_EPSILON,
        "source_bindings": {
            "trainer_inputs": {"path": str(TRAINER_INPUTS), "sha256": file_hash(TRAINER_INPUTS)},
            "manifest": {"path": str(MANIFEST), "sha256": file_hash(MANIFEST)},
            "endpoint_packet": {"path": str(ENDPOINT_PACKET), "sha256": file_hash(ENDPOINT_PACKET)},
            "endpoint_a_consumer": {"path": str(ENDPOINT_CONSUMER), "sha256": file_hash(ENDPOINT_CONSUMER)},
            "full_a_receipt": {"path": str(FULL_A_RECEIPT), "sha256": file_hash(FULL_A_RECEIPT)},
        },
        "runtime_identity": _runtime_identity(),
        "config": trainer_inputs["config"],
        "model": {
            "base_model": trainer_inputs["model_identity"]["base_model"],
            "raw_config_declared_source_adapter": trainer_inputs["model_identity"]["declared_packet_adapter"],
            "stable50_adapter": _slim_adapter(trainer_inputs["stable50_adapter"]),
            "positive32_adapter": _slim_adapter(full_a["saved_adapter"]),
            "source_embedding": _slim_adapter(trainer_inputs["source_embedding"]),
            "checkpoint_override_required": True,
        },
        "counts": {
            "cases": TOTAL_CASES,
            "action_tokens": TOTAL_ACTION_TOKENS,
            "protected_positions": TOTAL_PROTECTED_POSITIONS,
            "workers": WORLD_SIZE,
            "cases_per_worker": CASES_PER_WORKER,
        },
        "bounds": {
            "model_loads_per_worker": 2,
            "score_forwards_per_model_per_worker": CASES_PER_WORKER,
            "score_forwards_global": TOTAL_CASES * 2,
            "image_forwards_global": TOTAL_CASES * 2,
            "generation_forwards": 0,
            "backwards": 0,
            "collectives": 0,
            "max_rank_seconds": MAX_RANK_SECONDS,
            "max_cuda_allocated_bytes": MAX_CUDA_BYTES,
            "max_cuda_reserved_bytes": MAX_CUDA_BYTES,
            "max_rss_bytes": MAX_RSS_BYTES,
            "max_reference_cache_bytes_per_rank": MAX_REFERENCE_CACHE_BYTES,
        },
        "geometry_invalid_exclusion": {
            "image_id": "360573", "positions": list(exclusion["360573"]),
        },
        "endpoint_summary": {
            "iou50": {key: totals[key] for key in (
                "stable_tp", "positive32_tp", "lost_owners", "gained_owners"
            )},
            "burden": {
                "stable_strict_repeats": totals["stable_strict_repeats"],
                "positive32_strict_repeats": totals["current_strict_repeats"],
                "stable_parser_drops": totals["stable_parser_drops"],
                "positive32_parser_drops": totals["current_parser_drops"],
            },
            "identical_natural_actions": totals["identical_natural_actions"],
        },
        "shards": shards,
        "shard_plan": shard_plan,
        "cases": cases,
        "claim_boundary": {
            "descriptive": True,
            "not_claimed": [
                "a margin crossing causes owner loss",
                "a one-token rescue restores an owner",
                "increasing reference coverage or KL weight is indicated",
                "negative efficacy for the repeat-event arm",
            ],
        },
    }
    packet["content_sha256"] = digest_json(packet)
    return packet


def validate_input_packet(packet: Mapping[str, Any], *, verify_sources: bool) -> None:
    from probes.dora_owner_learning.repeat_recovery_train import validate_inputs

    require(packet.get("schema") == SCHEMA and packet.get("status") == "prepared_no_model_execution",
            "input packet schema/status")
    content = dict(packet)
    observed_digest = content.pop("content_sha256", None)
    require(observed_digest == digest_json(content), "input packet content digest")
    require(packet["counts"] == {
        "cases": TOTAL_CASES, "action_tokens": TOTAL_ACTION_TOKENS,
        "protected_positions": TOTAL_PROTECTED_POSITIONS,
        "workers": WORLD_SIZE, "cases_per_worker": CASES_PER_WORKER,
    }, "fixed microscope counts")
    require(packet["bounds"] == {
        "model_loads_per_worker": 2,
        "score_forwards_per_model_per_worker": CASES_PER_WORKER,
        "score_forwards_global": TOTAL_CASES * 2,
        "image_forwards_global": TOTAL_CASES * 2,
        "generation_forwards": 0,
        "backwards": 0,
        "collectives": 0,
        "max_rank_seconds": MAX_RANK_SECONDS,
        "max_cuda_allocated_bytes": MAX_CUDA_BYTES,
        "max_cuda_reserved_bytes": MAX_CUDA_BYTES,
        "max_rss_bytes": MAX_RSS_BYTES,
        "max_reference_cache_bytes_per_rank": MAX_REFERENCE_CACHE_BYTES,
    }, "fixed microscope bounds")
    cases = packet["cases"]
    require(len(cases) == TOTAL_CASES and len({row["key"] for row in cases}) == TOTAL_CASES,
            "unique case population")
    action_tokens = protected_positions = 0
    for case in cases:
        source = case["source_case"]
        natural = case["natural"]
        action_tokens += len(source["action_ids"])
        protected_positions += len(source["initial_layout"]["kl_positions"])
        require(
            case["key"] == source["key"]
            and case["image_id"] == str(source["image_id"])
            and digest_ids(source["action_ids"]) == source["action_ids_sha256"]
            and digest_ids(source["action_ids"]) == natural["stable_action_ids_sha256"]
            and digest_ids(natural["current_action_ids"]) == natural["current_action_ids_sha256"],
            "literal old/new action identity",
        )
        observed = first_divergence(source["action_ids"], natural["current_action_ids"])
        if observed is not None:
            observed["protected"] = observed["position"] in source["initial_layout"]["kl_positions"]
        require(observed == natural["first_divergence"], "natural first divergence")
        old, new = set(natural["stable_iou50_owners"]), set(natural["current_iou50_owners"])
        require(
            natural["lost_iou50_owners"] == sorted(old - new)
            and natural["gained_iou50_owners"] == sorted(new - old),
            "owner outcome sets",
        )
    require(action_tokens == TOTAL_ACTION_TOKENS and protected_positions == TOTAL_PROTECTED_POSITIONS,
            "action/mask totals")
    shards = packet["shards"]
    require(
        len(shards) == WORLD_SIZE and all(len(shard) == CASES_PER_WORKER for shard in shards)
        and [key for shard in shards for key in shard]
        == [case["key"] for rank in range(WORLD_SIZE) for case in cases[rank::WORLD_SIZE]],
        "eight independent seven-case shards",
    )
    model_config = load_json(Path(packet["model"]["base_model"]) / "config.json")
    vocab_size = int(model_config.get("text_config", model_config)["vocab_size"])
    by_key = {case["key"]: case for case in cases}
    expected_plan = []
    for rank, shard in enumerate(shards):
        action_count = sum(len(by_key[key]["source_case"]["action_ids"]) for key in shard)
        protected_count = sum(
            len(by_key[key]["source_case"]["initial_layout"]["kl_positions"])
            for key in shard
        )
        expected_plan.append({
            "rank": rank, "cases": len(shard), "action_tokens": action_count,
            "protected_positions": protected_count,
            "estimated_reference_cache_bytes": protected_count * vocab_size * 4,
        })
    require(packet["shard_plan"] == expected_plan and
            max(row["estimated_reference_cache_bytes"] for row in expected_plan)
            <= MAX_REFERENCE_CACHE_BYTES, "rank resource plan")
    require(packet["geometry_invalid_exclusion"] == {
        "image_id": "360573", "positions": list(range(75, 84)),
    }, "known invalid-row exclusion")
    require(packet["endpoint_summary"] == {
        "iou50": {"stable_tp": 416, "positive32_tp": 389,
                  "lost_owners": 33, "gained_owners": 6},
        "burden": {"stable_strict_repeats": 0, "positive32_strict_repeats": 7,
                   "stable_parser_drops": 1, "positive32_parser_drops": 2},
        "identical_natural_actions": 1,
    }, "accepted endpoint summary")
    if verify_sources:
        for source in packet["source_bindings"].values():
            require(file_hash(source["path"]) == source["sha256"], f"bound source changed: {source['path']}")
        for row in packet["runtime_identity"]:
            require(file_hash(row["path"]) == row["sha256"], f"runtime code changed: {row['path']}")
        validate_inputs(Path(packet["source_bindings"]["trainer_inputs"]["path"]), verify_sources=True)
        _verify_payload(packet["model"]["stable50_adapter"], "Stable50 adapter")
        _verify_payload(packet["model"]["positive32_adapter"], "positive32 adapter")
        _verify_payload(packet["model"]["source_embedding"], "source embedding")


def combine_resource_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(observations, "resource observations absent")
    return {
        "phases": [str(row["phase"]) for row in observations],
        "peak_cuda_allocated_bytes": max(int(row["peak_cuda_allocated_bytes"]) for row in observations),
        "peak_cuda_reserved_bytes": max(int(row["peak_cuda_reserved_bytes"]) for row in observations),
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in observations),
        "elapsed_seconds": max(float(row["elapsed_seconds"]) for row in observations),
    }


def _resource_observation(device: torch.device, started: float, phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "elapsed_seconds": time.monotonic() - started,
    }


def _token_string(tokenizer: Any, token_id: int) -> str:
    value = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(value, str):
        return value
    return tokenizer.decode([int(token_id)], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def _load_checkpoint(packet: Mapping[str, Any], adapter: Mapping[str, Any], device: torch.device):
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.inference.runtime import assemble_frontend
    from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
    from probes.dora_owner_learning.repeat_recovery_train import loaded_composition_evidence
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy

    base = load_research_infer_config(CONFIG).config
    require(base.model_dump(mode="json") == packet["config"], "live Source-shaped config changed")
    declared = packet["model"]["raw_config_declared_source_adapter"]["path"]
    require(
        base.adapter is not None and str(base.adapter.path) == declared
        and declared not in {
            packet["model"]["stable50_adapter"]["root"],
            packet["model"]["positive32_adapter"]["root"],
        },
        "raw Source config checkpoint trap not preserved",
    )
    config = checkpoint_config(base, adapter["root"])
    require(
        str(config.adapter.path) == adapter["root"]
        and config.model.dtype == "fp32"
        and config.backend.hf.attn_implementation == "sdpa"
        and config.backend.hf.patch_embed_linearization == "enabled"
        and config.embedding_delta is not None
        and str(config.embedding_delta.path) == packet["model"]["source_embedding"]["root"],
        "explicit FP32/SDPA checkpoint override",
    )
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    qwen, loaded = load_policy(config, device=device)
    identity = loaded["model_identity"]
    inspected = inspect_special_token_embedding_delta_payload(
        identity["embedding_delta"]["identity"]["delta_path"], packet["model"]["base_model"],
    )
    composition = loaded_composition_evidence(
        loaded_identity=loaded,
        expected_base=packet["model"]["base_model"],
        expected_adapter=adapter["root"],
        expected_embedding=packet["model"]["source_embedding"],
        inspected_embedding=inspected,
    )
    require(composition["passed"], "loaded checkpoint composition")
    require(qwen.token_identity.im_end_token_ids == (EOS,) and qwen.tokenizer.pad_token_id == PAD,
            "native terminal identity")
    model = qwen.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, frontend, qwen, model, loaded, composition


def _install_forward_counters(model: torch.nn.Module, counters: dict[str, int]) -> None:
    def count_model(*_: Any) -> None:
        counters["model_forwards"] += 1
        require(counters["model_forwards"] <= 14, "rank model-forward ceiling")
    model.register_forward_pre_hook(count_model)
    visuals = [module for name, module in model.named_modules() if name.endswith("visual")]
    require(len(visuals) == 1, "single visual module")
    def count_image(*_: Any) -> None:
        counters["image_forwards"] += 1
        require(counters["image_forwards"] <= 14, "rank image-forward ceiling")
    visuals[0].register_forward_pre_hook(count_image)


def _model_logits(model: torch.nn.Module, entry: Mapping[str, Any]) -> torch.Tensor:
    from src.qwen.native import prepare_replay

    case = entry["case"]
    with torch.no_grad():
        replay = prepare_replay(
            model, entry["inputs"], prompt_token_ids=entry["prompt_ids"],
            continuation_token_ids=case["action_ids"],
        )
        logits = replay.aligned_logits(model(**replay.inputs).logits)
    require(replay.target_ids.tolist() == case["action_ids"], "native replay target identity")
    return logits


def execute_rank(input_path: Path, output_root: Path, rank: int) -> None:
    from src.data import load_raw_examples
    from probes.dora_owner_learning.repeat_recovery_train import _materialize_case

    require(0 <= rank < WORLD_SIZE and os.environ.get("CUDA_VISIBLE_DEVICES") == str(rank),
            "rank requires its one exact physical GPU")
    packet = load_json(input_path)
    validate_input_packet(packet, verify_sources=True)
    require(file_hash(input_path) == packet.get("published_sha256", file_hash(input_path)),
            "rank packet publication identity")
    run = output_root / f"rank-{rank}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device("cuda:0")
    counters = {
        "model_loads": 0, "model_forwards": 0, "image_forwards": 0,
        "generation_forwards": 0, "backwards": 0, "collectives": 0,
    }
    observations: list[dict[str, Any]] = []
    status, error = "failed", None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError(f"{MAX_RANK_SECONDS}-second rank ceiling")))
    signal.alarm(MAX_RANK_SECONDS)
    try:
        torch.cuda.set_device(device)
        # Torch 2.9.1 rejects an explicit cuda:0 before lazy CUDA initialization;
        # the accepted native endpoint resets the current device with no argument.
        torch.cuda.reset_peak_memory_stats()
        raw = {
            str(row.example_id): row
            for row in load_raw_examples(packet["config"]["data"]["input_jsonl"])
        }
        by_key = {case["key"]: case for case in packet["cases"]}
        cases = [by_key[key] for key in packet["shards"][rank]]
        require(len(cases) == CASES_PER_WORKER, "rank seven-case assignment")

        stable_config, stable_frontend, stable_qwen, stable_model, stable_loaded, stable_comp = \
            _load_checkpoint(packet, packet["model"]["stable50_adapter"], device)
        counters["model_loads"] += 1
        _install_forward_counters(stable_model, counters)
        publish(run / "stable50-loaded.json", stable_loaded)
        publish(run / "stable50-composition.json", stable_comp)
        reference_cache: dict[str, torch.Tensor] = {}
        reference_rows: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            source = case["source_case"]
            require(source["example_id"] in raw, "rank raw example absent")
            entry = _materialize_case(
                qwen=stable_qwen, frontend=stable_frontend, config=stable_config,
                raw=raw[source["example_id"]], case=source, positive=False,
            )
            snapshot = reference_snapshot(
                _model_logits(stable_model, entry), source["action_ids"],
                source["initial_layout"]["kl_positions"],
            )
            reference_cache[case["key"]] = snapshot["protected_logp"]
            reference_rows[case["key"]] = snapshot["positions"]
            del entry, snapshot
        cache_bytes = sum(value.numel() * value.element_size() for value in reference_cache.values())
        require(cache_bytes <= MAX_REFERENCE_CACHE_BYTES, "detached reference cache ceiling")
        del stable_model, stable_qwen, stable_frontend, stable_config
        gc.collect()
        torch.cuda.empty_cache()
        observations.append(_resource_observation(device, started, "after_stable50_release"))

        current_config, current_frontend, current_qwen, current_model, current_loaded, current_comp = \
            _load_checkpoint(packet, packet["model"]["positive32_adapter"], device)
        counters["model_loads"] += 1
        _install_forward_counters(current_model, counters)
        publish(run / "positive32-loaded.json", current_loaded)
        publish(run / "positive32-composition.json", current_comp)
        records = []
        for case in cases:
            source = case["source_case"]
            entry = _materialize_case(
                qwen=current_qwen, frontend=current_frontend, config=current_config,
                raw=raw[source["example_id"]], case=source, positive=False,
            )
            compared = compare_snapshot(
                _model_logits(current_model, entry), source["action_ids"],
                source["initial_layout"]["kl_positions"], reference_cache[case["key"]],
                reference_rows[case["key"]],
            )
            tokenizer = current_qwen.tokenizer
            for row in compared["positions"]:
                source_token = _token_string(tokenizer, row["source_token_id"])
                row["source_token"] = source_token
                row["token_category"] = token_category(source_token)
            divergence = dict(case["natural"]["first_divergence"] or {}) or None
            if divergence is not None:
                old_id, new_id = divergence["old_token_id"], divergence["new_token_id"]
                divergence["old_token"] = None if old_id is None else _token_string(tokenizer, old_id)
                divergence["new_token"] = None if new_id is None else _token_string(tokenizer, new_id)
                divergence["token_category"] = (
                    "old_ended" if old_id is None else token_category(divergence["old_token"])
                )
                scored = next((row for row in compared["positions"]
                               if row["position"] == divergence["position"]), None)
                divergence["scored_on_reference_trajectory"] = scored is not None
                divergence["current_replay_argmax_equals_observed_new"] = (
                    None if scored is None or new_id is None else scored["current_top1_id"] == new_id
                )
            records.append({
                "schema": "greedy_preservation_microscope.case.v1",
                "key": case["key"], "image_id": case["image_id"], "rank": rank,
                "source_action_ids_sha256": source["action_ids_sha256"],
                "prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
                "protected_positions": list(source["initial_layout"]["kl_positions"]),
                "natural": {**case["natural"], "first_divergence": divergence},
                "positions": compared["positions"],
            })
            del entry, compared
        require(len(records) == CASES_PER_WORKER, "rank record count")
        observations.append(_resource_observation(device, started, "terminal"))
        resources = combine_resource_observations(observations)
        require(
            counters == {"model_loads": 2, "model_forwards": 14, "image_forwards": 14,
                         "generation_forwards": 0, "backwards": 0, "collectives": 0}
            and resources["elapsed_seconds"] <= MAX_RANK_SECONDS
            and resources["peak_cuda_allocated_bytes"] <= MAX_CUDA_BYTES
            and resources["peak_cuda_reserved_bytes"] <= MAX_CUDA_BYTES
            and resources["peak_rss_bytes"] <= MAX_RSS_BYTES,
            "rank execution bounds",
        )
        with (run / "records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        receipt = {
            "schema": SHARD_SCHEMA, "status": "completed", "rank": rank,
            "input": {"path": str(input_path), "sha256": file_hash(input_path)},
            "records": {"path": str(run / "records.jsonl"),
                        "sha256": file_hash(run / "records.jsonl"), "count": len(records)},
            "case_keys": [record["key"] for record in records],
            "counters": counters, "reference_cache_bytes": cache_bytes,
            "resources": resources,
        }
        publish(run / "receipt.json", receipt)
        status = "completed"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.alarm(0)
        terminal = {"status": status, "rank": rank, "error": error,
                    "counters": counters, "elapsed_seconds": time.monotonic() - started}
        publish(run / "terminal.json", terminal)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _quantile(values: Sequence[float], q: float) -> float:
    require(values and 0 <= q <= 1, "quantile input")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def _concentration(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    result = {"total_kl": total}
    for fraction, label in ((0.01, "top_1pct"), (0.05, "top_5pct"), (0.10, "top_10pct")):
        count = max(1, math.ceil(len(ordered) * fraction))
        result[label] = {"positions": count, "share_of_total_kl": (
            sum(ordered[:count]) / total if total > 0 else 0.0
        )}
    return result


def summarize_records(records: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]) -> dict[str, Any]:
    require(len(records) == TOTAL_CASES and len({row["key"] for row in records}) == TOTAL_CASES,
            "consumer case population")
    by_key = {row["key"]: row for row in records}
    require(set(by_key) == {row["key"] for row in packet["cases"]}, "consumer exact case keys")
    flat, per_image, divergences = [], [], []
    category_values: dict[str, list[float]] = defaultdict(list)
    category_flips: dict[str, int] = defaultdict(int)
    for source_case in packet["cases"]:
        row = by_key[source_case["key"]]
        positions = row["positions"]
        source = source_case["source_case"]
        require(
            len(positions) == len(source["action_ids"])
            and [item["position"] for item in positions] == list(range(len(positions)))
            and [item["source_token_id"] for item in positions] == source["action_ids"],
            "consumer token-position alignment",
        )
        protected = [item for item in positions if item["protected"]]
        require(
            [item["position"] for item in protected] == source["initial_layout"]["kl_positions"]
            and all(item["reference_kl"] is not None for item in protected)
            and all(item["reference_kl"] is None for item in positions if not item["protected"]),
            "consumer exact mask alignment",
        )
        flat.extend(protected)
        for item in protected:
            category_values[item["token_category"]].append(item["reference_kl"])
            category_flips[item["token_category"]] += item["source_argmax_retained"] is False
        kls = [item["reference_kl"] for item in protected]
        flips = sum(item["source_argmax_retained"] is False for item in positions)
        natural = row["natural"]
        divergence = natural["first_divergence"]
        if divergence is not None:
            scored = next((item for item in positions if item["position"] == divergence["position"]), None)
            if scored is not None:
                divergence = {**divergence,
                    "reference_kl": scored["reference_kl"],
                    "reference_source_margin": scored["reference_source_margin"],
                    "current_source_margin": scored["current_source_margin"],
                    "reference_source_is_argmax": scored["reference_source_is_argmax"],
                    "current_source_is_argmax": scored["current_source_is_argmax"],
                    "near_tie_reference": scored["near_tie_reference"],
                    "near_tie_current": scored["near_tie_current"],
                }
            divergences.append({"key": row["key"], "image_id": row["image_id"], **divergence})
        per_image.append({
            "key": row["key"], "image_id": row["image_id"],
            "action_tokens": len(positions), "protected_positions": len(protected),
            "mean_reference_kl": sum(kls) / len(kls), "max_reference_kl": max(kls),
            "source_argmax_reference_exceptions": sum(
                not item["reference_source_is_argmax"] for item in positions),
            "source_argmax_checkpoint_flips": flips,
            "near_tie_reference_positions": sum(item["near_tie_reference"] for item in positions),
            "near_tie_current_positions": sum(item["near_tie_current"] for item in positions),
            "lost_iou50_owners": natural["lost_iou50_owners"],
            "gained_iou50_owners": natural["gained_iou50_owners"],
            "strict_repeat_delta": natural["current_strict_repeats"] - natural["stable_strict_repeats"],
            "parser_drop_delta": natural["current_parser_drops"] - natural["stable_parser_drops"],
            "first_divergence": divergence,
        })
    require(len(flat) == TOTAL_PROTECTED_POSITIONS, "consumer global protected count")
    kls = [float(row["reference_kl"]) for row in flat]
    groups = {}
    for label, predicate in (
        ("owner_loss", lambda row: bool(row["lost_iou50_owners"])),
        ("no_owner_loss", lambda row: not row["lost_iou50_owners"]),
    ):
        selected = [row for row in per_image if predicate(row)]
        groups[label] = {
            "images": len(selected),
            "lost_owners": sum(len(row["lost_iou50_owners"]) for row in selected),
            "mean_of_image_mean_kl": sum(row["mean_reference_kl"] for row in selected) / len(selected),
            "images_with_checkpoint_flip": sum(row["source_argmax_checkpoint_flips"] > 0 for row in selected),
            "checkpoint_flips": sum(row["source_argmax_checkpoint_flips"] for row in selected),
        }
    return {
        "schema": CONSUMER_SCHEMA,
        "status": "completed_descriptive_localization",
        "input_sha256": packet.get("published_sha256"),
        "counts": {
            "cases": len(records), "action_tokens": sum(len(row["positions"]) for row in records),
            "protected_positions": len(flat), "natural_divergences": len(divergences),
            "identical_natural_actions": TOTAL_CASES - len(divergences),
        },
        "endpoint_summary": packet["endpoint_summary"],
        "reference_kl": {
            "mean": sum(kls) / len(kls), "max": max(kls),
            "p50": _quantile(kls, 0.50), "p90": _quantile(kls, 0.90),
            "p95": _quantile(kls, 0.95), "p99": _quantile(kls, 0.99),
            "concentration": _concentration(kls),
        },
        "greedy_margin": {
            "reference_literal_argmax_exceptions": sum(
                not row["reference_source_is_argmax"] for record in records for row in record["positions"]),
            "checkpoint_argmax_flips": sum(
                row["source_argmax_retained"] is False for record in records for row in record["positions"]),
            "near_tie_reference_positions": sum(
                row["near_tie_reference"] for record in records for row in record["positions"]),
            "near_tie_current_positions": sum(
                row["near_tie_current"] for record in records for row in record["positions"]),
            "epsilon": NEAR_TIE_EPSILON,
        },
        "by_token_category": {
            category: {
                "protected_positions": len(values), "mean_reference_kl": sum(values) / len(values),
                "max_reference_kl": max(values), "checkpoint_argmax_flips": category_flips[category],
            }
            for category, values in sorted(category_values.items())
        },
        "owner_loss_association": groups,
        "first_natural_divergences": divergences,
        "per_image": per_image,
        "claim_boundary": packet["claim_boundary"],
    }


def consume(input_path: Path, output_root: Path, *, publish_result: bool) -> dict[str, Any]:
    packet = load_json(input_path)
    validate_input_packet(packet, verify_sources=True)
    receipts, records = [], []
    for rank in range(WORLD_SIZE):
        run = output_root / f"rank-{rank}"
        receipt = load_json(run / "receipt.json")
        require(
            receipt.get("schema") == SHARD_SCHEMA and receipt.get("status") == "completed"
            and receipt["rank"] == rank and receipt["input"]["sha256"] == file_hash(input_path)
            and receipt["case_keys"] == packet["shards"][rank]
            and file_hash(receipt["records"]["path"]) == receipt["records"]["sha256"],
            f"completed exact shard {rank}",
        )
        rank_records = _read_jsonl(Path(receipt["records"]["path"]))
        require(len(rank_records) == CASES_PER_WORKER, "shard record count")
        receipts.append({"path": str(run / "receipt.json"), "sha256": file_hash(run / "receipt.json")})
        records.extend(rank_records)
    result = summarize_records(records, packet)
    result["shard_receipts"] = receipts
    result["input_sha256"] = file_hash(input_path)
    if publish_result:
        publish(output_root / "consumer.json", result)
    return result


def launch(input_path: Path, output_root: Path) -> int:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7",
            "launcher requires exact GPUs 0..7")
    packet = load_json(input_path)
    validate_input_packet(packet, verify_sources=True)
    output_root.mkdir(parents=True, exist_ok=False)
    command_base = [sys.executable, str(Path(__file__).resolve()), "rank",
                    "--input", str(input_path), "--output-root", str(output_root)]
    publish(output_root / "group-launch.json", {
        "input": {"path": str(input_path), "sha256": file_hash(input_path)},
        "workers": WORLD_SIZE, "command_base": command_base,
    })
    processes = []
    logs = []
    for rank in range(WORLD_SIZE):
        log = (output_root / f"rank-{rank}.log").open("x", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        process = subprocess.Popen(
            [*command_base, "--rank", str(rank)], stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        processes.append(process)
        logs.append(log)
    exits = []
    for rank, process in enumerate(processes):
        code = process.wait()
        logs[rank].close()
        exits.append({"rank": rank, "exit_code": code})
    publish(output_root / "group-completion.json", {"status": (
        "completed" if all(row["exit_code"] == 0 for row in exits) else "failed"
    ), "exits": exits})
    if any(row["exit_code"] != 0 for row in exits):
        return 1
    consume(input_path, output_root, publish_result=True)
    return 0


def prepare(output: Path) -> dict[str, Any]:
    require(not output.exists(), "preparation output already exists")
    packet = assemble_input_packet()
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / "input.json"
    publish(input_path, packet)
    # Publication bytes are an outer receipt field, not part of the packet's self digest.
    published_sha = file_hash(input_path)
    commands = {
        "gpu_after_root_grant": (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python {Path(__file__).resolve()} launch "
            f"--input {input_path} --output-root {RAW_ROOT / 'execution'}"
        ),
        "root_consumer_check_after_gpu": (
            f"python {Path(__file__).resolve()} verify-output --input {input_path} "
            f"--output-root {RAW_ROOT / 'execution'}"
        ),
        "no_gpu_before_grant": True,
    }
    publish(output / "proposed-commands.json", commands)
    preflight = {
        "schema": "greedy_preservation_microscope.preflight.v1",
        "status": "prepared_cpu_only_gpu_not_started",
        "input": {"path": str(input_path), "sha256": published_sha},
        "code": {"path": str(Path(__file__).resolve()), "sha256": file_hash(__file__)},
        "counts": packet["counts"], "bounds": packet["bounds"],
        "endpoint_summary": packet["endpoint_summary"],
    }
    publish(output / "preflight.json", preflight)
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=PREPARATION)
    validate_parser = commands.add_parser("verify-preparation")
    validate_parser.add_argument("--input", type=Path, required=True)
    rank_parser = commands.add_parser("rank")
    rank_parser.add_argument("--input", type=Path, required=True)
    rank_parser.add_argument("--output-root", type=Path, required=True)
    rank_parser.add_argument("--rank", type=int, required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--input", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = commands.add_parser("verify-output")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output), sort_keys=True))
    elif args.command == "verify-preparation":
        packet = load_json(args.input)
        validate_input_packet(packet, verify_sources=True)
        print(json.dumps({"status": "passed", "input_sha256": file_hash(args.input),
                          "counts": packet["counts"]}, sort_keys=True))
    elif args.command == "rank":
        execute_rank(args.input, args.output_root, args.rank)
    elif args.command == "launch":
        return launch(args.input, args.output_root)
    else:
        observed = load_json(args.output_root / "consumer.json")
        expected = consume(args.input, args.output_root, publish_result=False)
        require(observed == expected, "consumer replay differs")
        print(json.dumps({"status": "passed", "consumer_sha256": file_hash(
            args.output_root / "consumer.json"), "counts": expected["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
