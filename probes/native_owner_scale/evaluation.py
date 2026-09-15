"""Independent evaluation for the native-owner scale-and-state unit.

The module deliberately owns only the evaluation seam:

* identity-only, salted selection of the fresh 256-image panel and a fixed
  32-image review subset;
* CPU construction of the native input packet; and
* a Stable50-only natural endpoint producer with a frozen two-image slice and
  no-rerun remaining-254 phase; and
* cold consumption of natural endpoint rows plus a source-blind mixed-review
  queue.

The scaled model endpoint remains lane/root responsibility.  The baseline
producer delegates model loading, generation, parsing and scoring to the
accepted native runtime; the consumer reuses the accepted native parser,
global matcher and strict-repeat scorer instead of implementing a second metric
path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

from probes.dora_owner_learning import margin_preserved_endpoint as accepted
from probes.dora_owner_learning.candidate_opportunity import file_hash, require, score
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores
from probes.dora_owner_learning.geometric_dedup_eval import overlap_counts


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-12-native-owner-scale-and-state/evaluation"
PRIOR_SELECTION = BASE / "2026-09-12-parallel-owner-research/transfer/selection.json"
STABLE_PACKET = BASE / "2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json"
SOURCE = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl")
NATIVE_SOURCE = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl"
)
LEGACY_EXPOSED_PACKET = STABLE_PACKET
CURRENT_EVALUATION_PACKET = ROOT / "packet-v5.json"
TRAINING_INPUT_V2 = BASE / "2026-09-12-native-owner-scale-and-state/scale/training/preparation/inputs-v2.json"
BASELINE_CONTINUATION_PACKET = ROOT / "baseline-continuation-packet.json"
BASELINE_CONTINUATION_CONSUMER = ROOT / "baseline-continuation-full-consumer/consumer.json"
BASELINE_CONTINUATION_RESULT = ROOT / "baseline-continuation-full-consumer/result.json"
EVALUATION_SNAPSHOT = ROOT / "snapshots/evaluation-before-legacy384-extension.py"
CAP = 3084
EOS = 151645
PANEL_SIZE = 256
BLIND_SIZE = 32
LEGACY_EXPOSED_SIZE = 384
COMBINED_PANEL_SIZE = LEGACY_EXPOSED_SIZE + PANEL_SIZE
ADMITTED_IMAGE_COUNT = 11
REFERENCE_IMAGE_COUNT = 54
TRUSTED_TARGET_COUNT = 16
PANEL_SALT = "native-owner-scale-evaluation-2026-09-12:"
BLIND_SALT = "native-owner-scale-blind-review-2026-09-12:"
BOOTSTRAP_SEED = 20260912
BASELINE_GPUS = (4, 5)
BASELINE_SLICE_IDS = (64523, 395633)
BASELINE_ARM = "Stable50"
BASELINE_PHASES = ("slice", "full")
CANDIDATE_GPUS = tuple(range(8))
CANDIDATE_MAX_RANK_SECONDS = 36000
CANDIDATE_MAX_CUDA_ALLOCATED_BYTES = 24 * 1024**3
CANDIDATE_MAX_CUDA_RESERVED_BYTES = 24 * 1024**3
CANDIDATE_MAX_RSS_BYTES = 32 * 1024**3
# The root-granted slice was launched with this exact producer.  The only
# subsequent producer edits add model/source receipt serialization and CPU
# receipt backfill; endpoint generation semantics are unchanged.
BASELINE_LAUNCH_PRODUCER_SHA256 = "c7c2d69bf1997cfe59ba1510c5bbed3c3cc54afc24b8739d0e59d3dc8bfc923b"


def read(path: str | Path) -> Any:
    """Read one JSON artifact without accepting JSONL by accident."""

    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def publish(path: str | Path, value: Any) -> None:
    """Publish JSON atomically and fail on an occupied artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted.publish(path, value)


_IMAGE_PATTERNS = (
    re.compile(r"coco2017_(?:train|val)_(\d{1,12})(?=[^\d]|$)"),
    re.compile(r"(?:train|val)2017/(\d{1,12})(?=[^\d]|$)"),
    re.compile(r"image[-_](\d{1,12})(?=[^\d]|$)"),
)


def image_ids(value: Any, key: str = "") -> set[int]:
    """Project image identities only; never use scores, counts or outputs.

    Numeric values are accepted only from explicit ``image_id``/``image_ids``
    fields.  Path/example-id strings use the same conservative forms as the
    accepted fresh-transfer selector.
    """

    result: set[int] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            result.update(image_ids(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.update(image_ids(child, key))
    elif type(value) is int and (key == "image_id" or key.endswith("image_ids")):
        result.add(value)
    elif isinstance(value, str):
        for pattern in _IMAGE_PATTERNS:
            result.update(int(match.group(1)) for match in pattern.finditer(value))
    return result


def _ids_from_source(value: Any) -> set[int]:
    """Extract all identity-bearing fields from a declared manifest."""

    result = image_ids(value)
    if isinstance(value, Mapping):
        for key in ("eval_records", "records", "cases", "scenes", "image_ids", "excluded_image_ids"):
            if key in value:
                result.update(image_ids(value[key], key))
    return result


def select_ids(
    universe: Iterable[int], exclusions: Iterable[int], count: int, salt: str = PANEL_SALT
) -> list[int]:
    """Select by identity-only salted SHA256 ordering with no backfill."""

    universe_list = [int(value) for value in universe]
    require(len(universe_list) == len(set(universe_list)), "duplicate source image identity")
    eligible = set(universe_list) - {int(value) for value in exclusions}
    require(len(eligible) >= count, "insufficient unseen panel; no alternate source/backfill")
    return sorted(eligible, key=lambda image_id: hashlib.sha256(f"{salt}{image_id}".encode()).hexdigest())[
        :count
    ]


def validate_selection(selection: Mapping[str, Any], universe: Iterable[int] | None = None) -> None:
    """Validate panel, blind subset, and exclusion invariants."""

    panel = [int(value) for value in selection["image_ids"]]
    blind = [int(value) for value in selection["blind_review_ids"]]
    excluded = {int(value) for value in selection["excluded_image_ids"]}
    require(len(panel) == PANEL_SIZE and len(set(panel)) == PANEL_SIZE, "fresh256 identity/denominator")
    require(len(blind) == BLIND_SIZE and len(set(blind)) == BLIND_SIZE, "blind32 identity/denominator")
    require(set(blind) <= set(panel), "blind32 must be preselected from fresh256")
    require(not set(panel) & excluded, "fresh256 overlaps declared exclusions")
    require(not set(blind) & excluded, "blind32 overlaps declared exclusions")
    if universe is not None:
        source = {int(value) for value in universe}
        require(set(panel) <= source and set(blind) <= source, "selection outside source universe")
    require(selection["image_ids_sha256"] == digest(panel), "fresh256 identity digest")
    require(selection["blind_review_ids_sha256"] == digest(blind), "blind32 identity digest")


def _declared_manifest(path: str | Path, role: str) -> dict[str, Any]:
    path = Path(path)
    require(path.exists(), f"missing {role} exclusion manifest: {path}")
    value = read(path)
    status = value.get("status") if isinstance(value, Mapping) else None
    accepted_statuses = {
        "acknowledged",
        "candidate_cpu_verified",
        "cpu_ready_no_model_calls",
        "frozen_cpu_no_model_calls",
        "frozen",
        "frozen_before_new_outputs",
        "frozen_before_treatment_outputs",
        "lead_accepted",
        "ready",
        "ready_for_launch",
    }
    require(status in accepted_statuses, f"{role} exclusion manifest is not acknowledged/frozen")
    ids = _ids_from_source(value)
    require(ids, f"{role} exclusion manifest has no image identities")
    return {
        "role": role,
        "binding": binding(path),
        "status": status,
        "image_ids": sorted(ids),
        "image_count": len(ids),
    }


def build_exclusion_inventory(
    *,
    prior_selection: str | Path = PRIOR_SELECTION,
    stable_packet: str | Path = STABLE_PACKET,
    lane_manifests: Sequence[tuple[str, str | Path]] = (),
) -> tuple[set[int], list[dict[str, Any]]]:
    """Build a bounded identity inventory from manifests, not raw traces.

    The prior selection is itself a provenance-qualified reconciliation of
    earlier training/selection/adjudication identities.  Its selected IDs
    and projected exclusions are both protected because the selected panel was
    exposed.  ``lane_manifests`` is explicit: callers may not silently widen
    the audit to raw traces or arbitrary output logs.
    """

    prior = read(prior_selection)
    require(prior.get("status") == "frozen_before_new_outputs", "prior transfer selection not frozen")
    prior_ids = set(int(value) for value in prior.get("excluded_image_ids", [])) | set(
        int(value) for value in prior.get("image_ids", [])
    )
    require(prior_ids, "prior transfer selection has no identities")
    sources = [
        {
            "role": "prior-transfer-selection",
            "binding": binding(prior_selection),
            "status": prior["status"],
            "image_ids": sorted(prior_ids),
            "image_count": len(prior_ids),
            "provenance_boundary": prior.get("provenance_boundary"),
        }
    ]

    stable = read(stable_packet)
    stable_records = stable.get("eval_records", [])
    stable_ids = _ids_from_source({"eval_records": stable_records})
    require(len(stable_ids) == 384, "Stable50 exposed union384 identity")
    sources.append(
        {
            "role": "stable50-exposed-union384",
            "binding": binding(stable_packet),
            "status": stable.get("status"),
            "image_ids": sorted(stable_ids),
            "image_count": len(stable_ids),
            "claim_boundary": "exposed Stable50 source universe; not pretraining/SFT disjointness",
        }
    )
    excluded = prior_ids | stable_ids
    for role, path in lane_manifests:
        manifest = _declared_manifest(path, role)
        sources.append(manifest)
        excluded.update(manifest["image_ids"])
    return excluded, sources


def freeze_selection(
    *,
    output: str | Path = ROOT,
    lane_manifests: Sequence[tuple[str, str | Path]] = (),
    coordination: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Freeze fresh256 and blind32 before any new model outputs."""

    output = Path(output)
    selection_path = output / "selection.json"
    require(not selection_path.exists(), "evaluation selection already frozen")
    source_rows = read_jsonl(SOURCE)
    universe = [int(row["image_id"]) for row in source_rows]
    excluded, sources = build_exclusion_inventory(lane_manifests=lane_manifests)
    panel = select_ids(universe, excluded, PANEL_SIZE, PANEL_SALT)
    blind = select_ids(panel, (), BLIND_SIZE, BLIND_SALT)
    result: dict[str, Any] = {
        "schema": "native_owner_scale_state.evaluation.selection.v1",
        "status": "frozen_before_new_outputs",
        "source": binding(SOURCE),
        "native_source": binding(NATIVE_SOURCE),
        "source_images": len(universe),
        "excluded_image_ids": sorted(excluded),
        "excluded_source_images": len(set(universe) & excluded),
        "eligible_images": len(set(universe) - excluded),
        "image_ids": panel,
        "image_ids_sha256": digest(panel),
        "blind_review_ids": blind,
        "blind_review_ids_sha256": digest(blind),
        "panel_salt": PANEL_SALT,
        "blind_review_salt": BLIND_SALT,
        "selection_rule": "SHA256(salt + decimal image ID); first256 from source minus identity exclusions; blind32 is the first salted subset of fresh256; no output/object-count/visual backfill",
        "exclusion_sources": sources,
        "coordination": list(coordination),
        "provenance_boundary": "Identity-disjoint from the declared prior transfer ledger, exposed Stable50 union384 and any explicitly acknowledged lane manifests. This is a bounded manifest reconciliation, not an exhaustive raw-trace or outside-investigation audit and makes no pretraining/SFT-disjointness claim.",
        "blind_review_boundary": "Source-blind mixed predictions from the two declared arms on preselected fresh32 identities; reviewed proposal coverage is not exhaustive scene recall or true-recall estimation.",
    }
    validate_selection(result, universe)
    publish(selection_path, result)
    return result


def _row_maps(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    require(len(result) == len(rows), f"duplicate example_id in endpoint rows: {path}")
    return result


def _source_rows(path: str | Path) -> tuple[dict[int, tuple[int, dict[str, Any], str]], dict[int, dict[str, Any]]]:
    indexed: dict[int, tuple[int, dict[str, Any], str]] = {}
    by_id: dict[int, dict[str, Any]] = {}
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = int(row["image_id"])
        require(image_id not in indexed, "duplicate native source image identity")
        indexed[image_id] = (index, row, line)
        by_id[image_id] = row
    return indexed, by_id


def _native_records(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Construct CPU native prompt/media/GT records for the frozen IDs."""

    from probes.dora_owner_learning.runtime import build_request
    from src.config.inference import InferConfig
    from src.data.examples import raw_example_from_jsonl_row
    from src.qwen.native import prepare_native_inputs
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options

    config = copy.deepcopy(read(STABLE_PACKET)["config"])
    config["data"]["input_jsonl"] = str(NATIVE_SOURCE)
    infer_config = InferConfig.model_validate(config)
    qwen = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=read(STABLE_PACKET)["model"]["base_model_path"],
            dtype="fp32",
            attn_implementation="sdpa",
            patch_embed_linearization="enabled",
            load_model=False,
        )
    )
    native, native_by_id = _source_rows(NATIVE_SOURCE)
    original, original_by_id = _source_rows(SOURCE)
    records: list[dict[str, Any]] = []
    for image_id in selection["image_ids"]:
        image_id = int(image_id)
        index, row, raw_line = native[image_id]
        _, source_row, _ = original[image_id]
        require(
            sorted(digest(obj) for obj in row["objects"])
            == sorted(digest(obj) for obj in source_row["objects"]),
            "native serialization changed owner-object multiset",
        )
        require(
            (NATIVE_SOURCE.parent / row["images"][0]).resolve()
            == (SOURCE.parent / source_row["images"][0]).resolve()
            and (row["width"], row["height"]) == (source_row["width"], source_row["height"]),
            "native serialization changed image bytes/dimensions",
        )
        raw = raw_example_from_jsonl_row(
            row, jsonl_path=NATIVE_SOURCE, row_number=index + 1, raw_line=raw_line
        )
        request, image_plan, _ = build_request(raw, config=infer_config, qwen=qwen, row_index=index)
        batch = prepare_native_inputs(qwen.processor, [request], device="cpu", record_media_identity=True)
        plan = image_plan.to_artifact_dict()
        plan.update(
            observed_image_grid_thw=list(batch.image_grids[0]),
            executed_media_sha256=batch.media_sha256[0],
            backend_prompt_token_count=len(batch.prompt_token_ids[0]),
            backend_projection_evidence_kind="hf_executed_tensors",
        )
        case = {
            "row_id": str(raw.example_id),
            "row_index": index,
            "image_path": str(raw.image.path),
            "image_width": raw.image.width,
            "image_height": raw.image.height,
            "input_record": row,
            "image_plan": plan,
        }
        golden = {
            "row_id": case["row_id"],
            "row_index": index,
            "image_path": case["image_path"],
            "image_width": case["image_width"],
            "image_height": case["image_height"],
            "example_id": str(raw.example_id),
            "gt": [obj.to_artifact_dict() for obj in raw.objects],
        }
        records.append(
            {
                "example_id": str(raw.example_id),
                "image_id": image_id,
                "split": "fresh256",
                "blind_review": image_id in set(selection["blind_review_ids"]),
                "case": case,
                "golden": golden,
                "prompt_token_ids": list(batch.prompt_token_ids[0]),
            }
        )
        del batch
    require(len(records) == PANEL_SIZE, "native record denominator")
    require({int(record["image_id"]) for record in records} == set(selection["image_ids"]), "native record identity")
    return records


def _adapter_identity_bridge(adapter: Mapping[str, Any], base_model_path: str | Path) -> dict[str, Any]:
    """Accept only the recorded swift->infras version/fingerprint rename."""

    from src.adapters.dora import inspect_dora_adapter_payload

    actual = inspect_dora_adapter_payload(adapter["root"], base_model_path)
    immutable = {key: value for key, value in actual.items() if key not in ("version", "fingerprint")}
    stored = {key: value for key, value in adapter.items() if key not in ("version", "fingerprint")}
    require(immutable == stored, "Stable50 adapter payload changed beyond version/fingerprint bridge")
    require(
        adapter.get("version") in {"coordexp-swift-dora-adapter-v1", "coordexp-infras-dora-adapter-v1"}
        and actual.get("version") == "coordexp-infras-dora-adapter-v1",
        "unsupported Stable50 adapter version bridge",
    )
    return actual


def prepare_packet(*, output: str | Path = ROOT) -> dict[str, Any]:
    """Materialize CPU input identities and the root launch/consumer contract."""

    output = Path(output)
    selection = read(output / "selection.json")
    source_rows = read_jsonl(SOURCE)
    validate_selection(selection, [int(row["image_id"]) for row in source_rows])
    packet_path = output / "packet.json"
    require(not packet_path.exists(), "evaluation packet already prepared")
    stable = read(STABLE_PACKET)
    records = _native_records(selection)
    model = copy.deepcopy(stable["model"])
    model["current_adapter"] = _adapter_identity_bridge(
        model["current_adapter"], model["base_model_path"]
    )
    config = copy.deepcopy(stable["config"])
    config["data"]["input_jsonl"] = str(NATIVE_SOURCE)
    packet = {
        "schema": "native_owner_scale_state.evaluation.packet.v1",
        "status": "cpu_prepared_candidate_pending",
        "selection": binding(output / "selection.json"),
        "records": records,
        "model": model,
        "config": config,
        "stable50": {
            "adapter": model["current_adapter"],
            "source_embedding": model["source_embedding"],
            "endpoint_role": "unchanged Stable50 baseline",
        },
        "scaled_terminal": {
            "status": "pending_A_terminal_adapter",
            "endpoint_role": "one frozen scaled terminal adapter only when A has N>=16 admitted packages",
        },
        "generation": {
            "dtype": "fp32",
            "attention": "sdpa",
            "patch_embed_linearization": "enabled",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": CAP,
            "eos_token_id": EOS,
            "natural_prefix_ids": [],
            "forced_credit": False,
        },
        "consumer": {
            "module": "probes.native_owner_scale.evaluation",
            "consume_command": "python -m probes.native_owner_scale.evaluation consume --packet <packet.json> --stable-rows <rows.jsonl> --candidate-rows <rows.jsonl>",
            "metric_path": "accepted native_record -> candidate_opportunity.score -> margin_preserved_endpoint.burden/owner_changes",
            "visualization_path": "evaluation/visualization-manifest.json with literal source-canvas image paths",
        },
        "root_launch_gate": {
            "minimal_full_path_slice": "two selected identities (one blind-review ID and one non-blind ID) x Stable50 + scaled_terminal, natural greedy/RP1 FP32/SDPA, full 3084 allowance, persisted rows then consume",
            "must_precede": "remaining 254-image endpoint expansion",
            "no_model_calls_performed_by_prepare": True,
            "candidate_condition": "A terminal receipt proves admitted_packages >= 16 and binds exactly one unmerged adapter",
        },
        "provenance_boundary": "Fresh panel is identity-disjoint from declared manifests only; no pretraining/SFT-disjointness claim.",
        "producer": binding(Path(__file__).resolve()),
        "native_source": binding(NATIVE_SOURCE),
        "stable_packet": binding(STABLE_PACKET),
    }
    publish(packet_path, packet)
    return packet


def _baseline_phase_ids(packet: Mapping[str, Any], phase: str) -> list[int]:
    require(phase in BASELINE_PHASES, f"unknown baseline phase: {phase}")
    ids = [int(value) for value in packet["phases"][phase]["image_ids"]]
    require(ids and len(ids) == len(set(ids)), f"baseline {phase} identity/denominator")
    return ids


def _verify_file_binding(value: Mapping[str, Any], label: str) -> Path:
    require(isinstance(value, Mapping) and value.get("path"), f"missing {label} binding")
    path = Path(value["path"])
    try:
        current = binding(path)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{label} missing") from exc
    require(current == dict(value), f"{label} changed")
    return path


def _validate_panel_union(slice_ids: Sequence[int], remaining_ids: Sequence[int], panel_ids: Sequence[int]) -> None:
    left = [int(value) for value in slice_ids]
    right = [int(value) for value in remaining_ids]
    panel = {int(value) for value in panel_ids}
    require(len(left) == 2 and len(set(left)) == 2, "completed slice denominator")
    require(len(right) == PANEL_SIZE - 2 and len(set(right)) == PANEL_SIZE - 2, "continuation remaining254 denominator")
    require(not set(left) & set(right), "continuation overlaps completed slice")
    require(set(left) | set(right) == panel, "continuation 2+254 panel union")


def prepare_baseline(
    *, packet_path: str | Path, output_path: str | Path = ROOT / "baseline-packet.json"
) -> dict[str, Any]:
    """Freeze the Stable50-only producer contract without invoking a model."""

    packet_path = Path(packet_path)
    output_path = Path(output_path)
    require(not output_path.exists(), "baseline packet already prepared")
    packet = read(packet_path)
    require(packet.get("schema") == "native_owner_scale_state.evaluation.packet.v1", "evaluation packet schema")
    selection = read(packet["selection"]["path"])
    source_rows = read_jsonl(SOURCE)
    validate_selection(selection, [int(row["image_id"]) for row in source_rows])
    records = {int(record["image_id"]): record for record in packet["records"]}
    require(len(records) == PANEL_SIZE, "evaluation packet fresh256 denominator")
    require(set(records) == set(int(value) for value in selection["image_ids"]), "evaluation packet selection identity")
    require(set(BASELINE_SLICE_IDS) <= set(records), "baseline slice outside frozen panel")
    full_ids = [int(value) for value in selection["image_ids"] if int(value) not in BASELINE_SLICE_IDS]
    require(len(full_ids) == PANEL_SIZE - len(BASELINE_SLICE_IDS), "baseline remaining254 denominator")
    stored_adapter = packet["stable50"]["adapter"]
    adapter = _adapter_identity_bridge(stored_adapter, packet["model"]["base_model_path"])
    phases_root = output_path.parent
    phases = {
        "slice": {
            "image_ids": list(BASELINE_SLICE_IDS),
            "run_root": str(phases_root / "baseline-slice"),
            "consumer_root": str(phases_root / "baseline-slice-consumer"),
        },
        "full": {
            "image_ids": full_ids,
            "run_root": str(phases_root / "baseline-full"),
            "consumer_root": str(phases_root / "baseline-full-consumer"),
        },
    }
    absolute_packet = output_path.resolve()
    launch_command = f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {absolute_packet} --phase slice"
    merge_command = lambda phase: f"python -m probes.native_owner_scale.evaluation baseline-merge --packet {absolute_packet} --phase {phase}"
    consume_command = lambda phase: f"python -m probes.native_owner_scale.evaluation consume-baseline --packet {absolute_packet} --phase {phase} --rows {phases[phase]['run_root']}/rows.jsonl --output {phases[phase]['consumer_root']}"
    result = {
        "schema": "native_owner_scale_state.evaluation.baseline_packet.v1",
        "status": "baseline_cpu_ready_no_model_calls",
        "arm": BASELINE_ARM,
        "input_packet": binding(packet_path),
        "selection": binding(packet["selection"]["path"]),
        "records": PANEL_SIZE,
        "stored_adapter": stored_adapter,
        "adapter": adapter,
        "model": packet["model"],
        "config": packet["config"],
        "generation": packet["generation"],
        "physical_gpus": list(BASELINE_GPUS),
        "phases": phases,
        "resource_bounds": {
            "workers": 2,
            "model_loads_per_worker": 1,
            "natural_continuations": {"slice": 2, "full": PANEL_SIZE - len(BASELINE_SLICE_IDS)},
            "max_action_tokens": {"slice": 2 * CAP, "full": (PANEL_SIZE - len(BASELINE_SLICE_IDS)) * CAP},
            "retry_policy": "none; preserve partial shard and terminal artifacts",
        },
        "producer": binding(Path(__file__).resolve()),
        "consumer": {
            "module": "probes.native_owner_scale.evaluation",
            "merge_command": "python -m probes.native_owner_scale.evaluation baseline-merge --packet <baseline-packet.json> --phase <slice|full>",
            "consume_command": "python -m probes.native_owner_scale.evaluation consume-baseline --packet <baseline-packet.json> --phase <slice|full> --rows <rows.jsonl> --output <consumer-dir>",
            "metric_path": "accepted native_record -> candidate_opportunity.score -> margin_preserved_endpoint.burden",
            "claim_boundary": "Stable50 baseline only; no candidate, paired transfer, GT edit, or hallucination inference",
        },
        "launch_gate": {
            "slice_ids": list(BASELINE_SLICE_IDS),
            "slice_order_is_frozen": True,
            "slice_must_pass_before_full": True,
            "full_excludes_slice_without_rerun": True,
            "natural_policy": "greedy T0/top-p1/RP1 FP32/SDPA, empty prefix, full 3084 allowance",
            "launch_command": launch_command,
            "raw_to_consumer": {
                "slice_merge": merge_command("slice"),
                "slice_consume": consume_command("slice"),
                "full_launch_after_slice": f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {absolute_packet} --phase full",
                "full_merge": merge_command("full"),
                "full_consume": consume_command("full"),
            },
        },
        "provenance_boundary": "Uses only the already frozen evaluation packet and identity selection; Stable50 is an exposed unchanged baseline and is not a new holdout claim.",
    }
    publish(output_path, result)
    request_path = output_path.with_name("baseline-launch-request.json")
    require(not request_path.exists(), "baseline launch request already prepared")
    publish(
        request_path,
        {
            "schema": "native_owner_scale_state.evaluation.baseline_launch_request.v1",
            "status": "ready_for_root_gpu_grant",
            "model_calls_performed": False,
            "packet": binding(output_path),
            "physical_gpus": list(BASELINE_GPUS),
            "slice": {
                "image_ids": list(BASELINE_SLICE_IDS),
                "count": len(BASELINE_SLICE_IDS),
                "launch": launch_command,
                "merge": merge_command("slice"),
                "consume": consume_command("slice"),
            },
            "remaining254": {
                "count": len(full_ids),
                "launch_after_slice_consumer": f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {absolute_packet} --phase full",
                "merge": merge_command("full"),
                "consume": consume_command("full"),
            },
            "natural_contract": result["launch_gate"]["natural_policy"],
            "raw_to_consumer": "worker shard rows.jsonl -> baseline-merge rows.jsonl -> consume-baseline consumer.json/result.json",
            "no_rerun_rule": "full phase is exactly panel minus frozen slice; baseline-launch rejects full until slice result status is cold_verified_baseline_only",
        },
    )
    return result


def prepare_baseline_continuation(
    *, prior_packet_path: str | Path, output_path: str | Path = ROOT / "baseline-continuation-packet.json"
) -> dict[str, Any]:
    """Bind the completed slice to a current-code, remaining254-only packet."""

    prior_packet_path = Path(prior_packet_path)
    output_path = Path(output_path)
    require(not output_path.exists(), "baseline continuation packet already prepared")
    prior = read(prior_packet_path)
    # The v3 packet is intentionally accepted only as a sealed parent.  It
    # must not be allowed to launch the current producer directly because its
    # producer receipt predates live model/source serialization.
    validate_baseline_packet(prior, allow_legacy_producer=True)
    require(prior.get("schema") == "native_owner_scale_state.evaluation.baseline_packet.v1", "continuation parent schema")
    prior_slice = prior["phases"]["slice"]
    prior_full = prior["phases"]["full"]
    slice_result_path = Path(prior_slice["consumer_root"]) / "result.json"
    slice_consumer_path = Path(prior_slice["consumer_root"]) / "consumer.json"
    slice_rows_path = Path(prior_slice["run_root"]) / "rows.jsonl"
    slice_result = read(_verify_file_binding(binding(slice_result_path), "slice result"))
    require(slice_result.get("status") == "cold_verified_baseline_only" and slice_result.get("images") == 2, "completed slice consumer gate")
    slice_consumer = read(_verify_file_binding(binding(slice_consumer_path), "slice consumer"))
    require([int(row["image_id"]) for row in slice_consumer] == list(BASELINE_SLICE_IDS), "completed slice identity")
    slice_rows = read_jsonl(_verify_file_binding(binding(slice_rows_path), "slice raw rows"))
    require([int(row["image_id"]) for row in slice_rows] == list(BASELINE_SLICE_IDS), "completed slice raw identity")
    require(slice_result["packet"] == binding(prior_packet_path), "completed slice parent packet identity")
    evaluation = read(prior["input_packet"]["path"])
    selection = read(prior["selection"]["path"])
    panel_ids = [int(value) for value in selection["image_ids"]]
    remaining_ids = [image_id for image_id in panel_ids if image_id not in BASELINE_SLICE_IDS]
    _validate_panel_union(BASELINE_SLICE_IDS, remaining_ids, panel_ids)
    require(remaining_ids == [int(value) for value in prior_full["image_ids"]], "continuation remaining IDs changed")
    continuation_root = output_path.parent / "baseline-continuation-full"
    continuation_consumer = output_path.parent / "baseline-continuation-full-consumer"
    continuation = copy.deepcopy(prior)
    continuation.update(
        {
            "schema": "native_owner_scale_state.evaluation.baseline_packet.v2",
            "packet_role": "remaining254_continuation",
            "status": "baseline_cpu_ready_no_model_calls",
            "producer": binding(Path(__file__).resolve()),
            "phases": {
                "slice": copy.deepcopy(prior_slice),
                "full": {
                    "image_ids": remaining_ids,
                    "run_root": str(continuation_root),
                    "consumer_root": str(continuation_consumer),
                },
            },
            "continuation": {
                "parent_packet": binding(prior_packet_path),
                "completed_slice": {
                    "image_ids": list(BASELINE_SLICE_IDS),
                    "rows": binding(slice_rows_path),
                    "consumer": binding(slice_consumer_path),
                    "result": binding(slice_result_path),
                },
                "panel_union": {
                    "panel_ids_sha256": digest(panel_ids),
                    "slice_ids_sha256": digest(list(BASELINE_SLICE_IDS)),
                    "remaining_ids_sha256": digest(remaining_ids),
                    "slice_plus_remaining_equals_panel": True,
                },
                "model_generation_unchanged_from_parent": {
                    "model": digest(prior["model"]),
                    "config": digest(prior["config"]),
                    "generation": digest(prior["generation"]),
                    "adapter": digest(prior["adapter"]),
                },
            },
        }
    )
    absolute_packet = output_path.resolve()
    continuation["launch_gate"] = copy.deepcopy(prior["launch_gate"])
    continuation["launch_gate"].update(
        {
            "launch_command": f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {absolute_packet} --phase full",
            "raw_to_consumer": {
                "completed_slice": "sealed in continuation.completed_slice; no rerun",
                "full_launch": f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {absolute_packet} --phase full",
                "full_merge": f"python -m probes.native_owner_scale.evaluation baseline-merge --packet {absolute_packet} --phase full",
                "full_consume": f"python -m probes.native_owner_scale.evaluation consume-baseline --packet {absolute_packet} --phase full --rows {continuation_root / 'rows.jsonl'} --output {continuation_consumer}",
            },
        }
    )
    require(continuation["model"] == prior["model"], "continuation model changed")
    require(continuation["config"] == prior["config"], "continuation config changed")
    require(continuation["generation"] == prior["generation"], "continuation generation changed")
    require(continuation["adapter"] == prior["adapter"], "continuation adapter changed")
    publish(output_path, continuation)
    request_path = output_path.with_name("baseline-continuation-request.json")
    require(not request_path.exists(), "baseline continuation request already prepared")
    publish(
        request_path,
        {
            "schema": "native_owner_scale_state.evaluation.baseline_continuation_request.v1",
            "status": "ready_for_root_gpu_grant",
            "model_calls_performed": False,
            "packet": binding(output_path),
            "parent_packet": binding(prior_packet_path),
            "completed_slice": {
                "image_ids": list(BASELINE_SLICE_IDS),
                "result": binding(slice_result_path),
                "consumer": binding(slice_consumer_path),
                "rows": binding(slice_rows_path),
            },
            "remaining254": {
                "image_ids_sha256": digest(remaining_ids),
                "count": len(remaining_ids),
                "launch": f"CUDA_VISIBLE_DEVICES=4,5 python -m probes.native_owner_scale.evaluation baseline-launch --packet {output_path.resolve()} --phase full",
                "merge": f"python -m probes.native_owner_scale.evaluation baseline-merge --packet {output_path.resolve()} --phase full",
                "consume": f"python -m probes.native_owner_scale.evaluation consume-baseline --packet {output_path.resolve()} --phase full --rows {continuation_root / 'rows.jsonl'} --output {continuation_consumer}",
            },
            "natural_contract": continuation["launch_gate"]["natural_policy"],
            "no_duplicate_slice": True,
            "source_change_policy": "worker binds frozen packet/source before first remaining inference; any changed binding fails closed",
        },
    )
    return continuation


def validate_baseline_packet(packet: Mapping[str, Any], *, allow_legacy_producer: bool = False) -> None:
    """Fail closed on the baseline identity and no-rerun contract."""

    schema = packet.get("schema")
    require(schema in {"native_owner_scale_state.evaluation.baseline_packet.v1", "native_owner_scale_state.evaluation.baseline_packet.v2"}, "baseline packet schema")
    require(packet.get("status") == "baseline_cpu_ready_no_model_calls", "baseline packet status")
    input_binding = packet["input_packet"]
    input_path = Path(input_binding["path"])
    require(binding(input_path) == input_binding, "evaluation input packet changed")
    evaluation = read(input_path)
    require(evaluation.get("schema") == "native_owner_scale_state.evaluation.packet.v1", "evaluation input schema")
    selection_binding = packet["selection"]
    selection_path = Path(selection_binding["path"])
    require(binding(selection_path) == selection_binding, "baseline selection changed")
    selection = read(selection_path)
    source_rows = read_jsonl(SOURCE)
    validate_selection(selection, [int(row["image_id"]) for row in source_rows])
    records = {int(record["image_id"]): record for record in evaluation["records"]}
    require(len(records) == PANEL_SIZE, "baseline input record denominator")
    require(set(records) == set(int(value) for value in selection["image_ids"]), "baseline input identities")
    _verify_file_binding(evaluation["native_source"], "native source")
    require(packet["stored_adapter"] == evaluation["stable50"]["adapter"], "baseline stored Stable50 adapter binding")
    require(packet["adapter"] == _adapter_identity_bridge(packet["stored_adapter"], evaluation["model"]["base_model_path"]), "baseline Stable50 adapter bridge")
    require(packet["generation"] == evaluation["generation"], "baseline generation binding")
    require(packet["generation"]["max_new_tokens"] == CAP, "baseline cap changed")
    require(packet["generation"]["dtype"] == "fp32" and packet["generation"]["attention"] == "sdpa", "baseline numerics changed")
    require(tuple(int(value) for value in packet["physical_gpus"]) == BASELINE_GPUS, "baseline GPU assignment changed")
    slice_ids = _baseline_phase_ids(packet, "slice")
    full_ids = _baseline_phase_ids(packet, "full")
    require(slice_ids == list(BASELINE_SLICE_IDS), "baseline slice identities/order changed")
    require(not set(slice_ids) & set(full_ids), "baseline full phase reruns slice identity")
    require(set(slice_ids) | set(full_ids) == set(records), "baseline phase population")
    require(len(slice_ids) == 2 and len(full_ids) == PANEL_SIZE - 2, "baseline phase denominators")
    current_producer = binding(Path(__file__).resolve())
    producer = packet["producer"]
    require(producer["path"] == current_producer["path"], "baseline producer path changed")
    if schema == "native_owner_scale_state.evaluation.baseline_packet.v1":
        require(
            producer == current_producer
            or (allow_legacy_producer and producer["sha256"] == BASELINE_LAUNCH_PRODUCER_SHA256),
            "baseline producer changed beyond receipt-only compatibility",
        )
    else:
        require(packet.get("packet_role") == "remaining254_continuation", "baseline continuation role")
        require(producer == current_producer, "baseline continuation producer must be current code")
        continuation = packet.get("continuation")
        require(isinstance(continuation, Mapping), "baseline continuation metadata")
        parent_binding = continuation.get("parent_packet")
        parent_path = _verify_file_binding(parent_binding, "continuation parent packet")
        parent = read(parent_path)
        require(parent.get("schema") == "native_owner_scale_state.evaluation.baseline_packet.v1", "continuation parent schema")
        validate_baseline_packet(parent, allow_legacy_producer=True)
        require(packet["model"] == parent["model"], "continuation model changed")
        require(packet["config"] == parent["config"], "continuation config changed")
        require(packet["generation"] == parent["generation"], "continuation generation changed")
        require(packet["adapter"] == parent["adapter"], "continuation adapter changed")
        panel_ids = [int(value) for value in selection["image_ids"]]
        panel_union = continuation.get("panel_union")
        require(isinstance(panel_union, Mapping), "continuation panel union metadata")
        require(panel_union.get("panel_ids_sha256") == digest(panel_ids), "continuation panel digest")
        require(panel_union.get("slice_ids_sha256") == digest(list(BASELINE_SLICE_IDS)), "continuation slice digest")
        require(panel_union.get("remaining_ids_sha256") == digest(full_ids), "continuation remaining digest")
        require(panel_union.get("slice_plus_remaining_equals_panel") is True, "continuation union acknowledgement")
        completed = continuation.get("completed_slice")
        require(isinstance(completed, Mapping), "continuation completed slice metadata")
        rows_path = _verify_file_binding(completed.get("rows"), "continuation slice rows")
        consumer_path = _verify_file_binding(completed.get("consumer"), "continuation slice consumer")
        result_path = _verify_file_binding(completed.get("result"), "continuation slice result")
        slice_result = read(result_path)
        require(slice_result.get("status") == "cold_verified_baseline_only" and slice_result.get("images") == 2, "continuation slice result")
        require(slice_result.get("packet") == parent_binding, "continuation slice parent identity")
        slice_rows = read_jsonl(rows_path)
        slice_consumer = read(consumer_path)
        require([int(row["image_id"]) for row in slice_rows] == list(BASELINE_SLICE_IDS), "continuation slice raw identity")
        require([int(row["image_id"]) for row in slice_consumer] == list(BASELINE_SLICE_IDS), "continuation slice consumer identity")
        parent_sha = parent_binding["sha256"]
        for row in slice_rows + slice_consumer:
            require(row.get("arm") == BASELINE_ARM, "continuation slice arm")
            require(row.get("adapter_fingerprint") == packet["adapter"]["fingerprint"], "continuation slice adapter identity")
            require(row.get("baseline_packet_sha256") == parent_sha, "continuation slice packet identity")


def _baseline_phase(packet: Mapping[str, Any], phase: str) -> dict[str, Any]:
    validate_baseline_packet(packet)
    ids = _baseline_phase_ids(packet, phase)
    value = dict(packet["phases"][phase])
    value["image_ids"] = ids
    return value


def baseline_worker(
    *, packet_path: str | Path, phase: str, shard: int, output: str | Path
) -> None:
    """Produce Stable50 natural rows through the accepted native runtime."""

    import torch

    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.eval.native_rows import native_detection_record as native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    packet_path = Path(packet_path)
    packet = read(packet_path)
    phase_packet = _baseline_phase(packet, phase)
    require(shard in (0, 1), "baseline shard")
    physical_gpu = int(packet["physical_gpus"][shard])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "baseline worker physical GPU")
    require(torch.cuda.device_count() == 1, "baseline worker requires one visible GPU")
    output = Path(output)
    require(not output.exists(), "baseline worker output occupied")
    output.mkdir(parents=True)
    eval_path = Path(packet["input_packet"]["path"])
    evaluation = read(eval_path)
    records = {int(record["image_id"]): record for record in evaluation["records"]}
    phase_records = [records[image_id] for image_id in phase_packet["image_ids"]]
    shard_records = phase_records[shard::2]
    terminal: dict[str, Any] = {
        "schema": "native_owner_scale_state.evaluation.baseline_terminal.v1",
        "status": "running",
        "arm": BASELINE_ARM,
        "phase": phase,
        "shard": shard,
        "physical_gpu": physical_gpu,
        "packet_sha256": file_hash(packet_path),
        "evaluation_packet_sha256": file_hash(eval_path),
        "expected_continuations": len(shard_records),
        "continuations": 0,
        "new_tokens": 0,
        "model_forwards": 0,
        "image_forwards": 0,
        "model_loads": 0,
    }
    publish(output / "launch.json", terminal)
    handles: list[Any] = []
    started = time.monotonic()
    try:
        config = checkpoint_config(
            InferConfig.model_validate(evaluation["config"]), packet["adapter"]["root"]
        )
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live_adapter = identity["model_identity"]["adapter"]
        require(
            live_adapter["adapter_path"] == packet["adapter"]["root"]
            and live_adapter["merged_adapters"] == [],
            "loaded Stable50 adapter identity changed",
        )
        require(
            identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
            and identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
            "live Stable50 FP32/SDPA identity changed",
        )
        publish(
            output / "model.json",
            {
                "schema": "native_owner_scale_state.evaluation.baseline_model_receipt.v1",
                "status": "live_model_loaded",
                "packet": binding(packet_path),
                "evaluation_packet": binding(eval_path),
                "phase": phase,
                "shard": shard,
                "physical_gpu": physical_gpu,
                "identity": identity,
                "adapter": packet["adapter"],
                "generation": packet["generation"],
            },
        )
        publish(
            output / "source.json",
            {
                "schema": "native_owner_scale_state.evaluation.baseline_source_receipt.v1",
                "status": "frozen_input_bound",
                "packet": binding(packet_path),
                "evaluation_packet": binding(eval_path),
                "selection": binding(evaluation["selection"]["path"]),
                "phase": phase,
                "shard": shard,
                "image_ids": [int(record["image_id"]) for record in shard_records],
                "records": [
                    {
                        "image_id": int(record["image_id"]),
                        "example_id": record["example_id"],
                        "image_path": record["case"]["image_path"],
                        "prompt_token_ids_sha256": digest(record["prompt_token_ids"]),
                        "executed_media_sha256": record["case"]["image_plan"]["executed_media_sha256"],
                        "observed_image_grid_thw": record["case"]["image_plan"]["observed_image_grid_thw"],
                    }
                    for record in shard_records
                ],
                "native_source": evaluation["native_source"],
            },
        )
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "visual hook ambiguity")
        handles.append(visual[0].register_forward_pre_hook(lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        policy = NativeGenerationPolicy(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_model_defaults=False,
        )
        rows_path = output / "rows.jsonl"
        with rows_path.open("x") as stream:
            for frozen in shard_records:
                requests, _ = build_requests(qwen, evaluation["config"], [frozen["case"]])
                batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
                plan = frozen["case"]["image_plan"]
                require(
                    list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"]
                    and batch.media_sha256[0] == plan["executed_media_sha256"]
                    and list(batch.image_grids[0]) == plan["observed_image_grid_thw"],
                    "frozen natural input identity changed",
                )
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        batch,
                        extensions=[[]],
                        budgets=[CAP],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                ids = list(generated.token_ids)
                stop = generated.stop_reason
                require(generated.request_id == frozen["example_id"], "baseline request identity")
                accepted._checked_action(ids, stop, CAP)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], stop)
                observed = {
                    "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                    "executed_media_sha256": batch.media_sha256[0],
                    "observed_image_grid_thw": list(batch.image_grids[0]),
                }
                row = {
                    "schema": "native_owner_scale_state.evaluation.baseline_natural.v1",
                    "arm": BASELINE_ARM,
                    "phase": phase,
                    "shard": shard,
                    "example_id": frozen["example_id"],
                    "image_id": frozen["image_id"],
                    "split": frozen["split"],
                    "request_id": generated.request_id,
                    "action_ids": ids,
                    "prefix_ids": [],
                    "forced_ids": [],
                    "remaining_budget": CAP,
                    "text": text,
                    "stop_reason": stop,
                    "parsed": parsed,
                    "score": score(parsed, seed=None, length=len(ids), stop=stop),
                    "overlap_counts": overlap_counts(parsed),
                    "batch_identity_sha256": digest(observed),
                    "packet_sha256": file_hash(eval_path),
                    "baseline_packet_sha256": file_hash(packet_path),
                    "adapter_fingerprint": packet["adapter"]["fingerprint"],
                    **observed,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(ids)
                del batch
        require(terminal["continuations"] == len(shard_records), "baseline shard denominator")
        require(terminal["new_tokens"] <= len(shard_records) * CAP, "baseline token cap")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        publish(output / "terminal.json", terminal)


def baseline_launch(*, packet_path: str | Path, phase: str) -> dict[str, Any]:
    """Launch exactly two independent Stable50 workers; never retries."""

    packet_path = Path(packet_path)
    packet = read(packet_path)
    phase_packet = _baseline_phase(packet, phase)
    if phase == "full":
        slice_consumer = Path(packet["phases"]["slice"]["consumer_root"]) / "result.json"
        require(slice_consumer.exists(), "baseline slice consumer must pass before full phase")
        require(read(slice_consumer).get("status") == "cold_verified_baseline_only", "baseline slice not cold-verified")
    output = Path(phase_packet["run_root"])
    require(not output.exists(), "baseline phase output occupied; no automatic relaunch")
    output.mkdir(parents=True)
    commands = []
    processes = []
    logs = []
    for shard, physical_gpu in enumerate(packet["physical_gpus"]):
        command = [
            sys.executable,
            "-m",
            "probes.native_owner_scale.evaluation",
            "baseline-worker",
            "--packet",
            str(packet_path),
            "--phase",
            phase,
            "--shard",
            str(shard),
            "--output",
            str(output / f"shard-{shard}"),
        ]
        commands.append(command)
        log = (output / f"shard-{shard}.log").open("x")
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        processes.append((shard, subprocess.Popen(command, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)))
    publish(
        output / "launch.json",
        {
            "schema": "native_owner_scale_state.evaluation.baseline_launch.v1",
            "status": "running",
            "packet": binding(packet_path),
            "phase": phase,
            "image_ids": phase_packet["image_ids"],
            "physical_gpus": list(packet["physical_gpus"]),
            "commands": commands,
            "no_retry": True,
        },
    )
    exits = []
    for (shard, process), log in zip(processes, logs, strict=True):
        exits.append({"shard": shard, "exit_code": process.wait()})
        log.close()
    publish(output / "outer-exits.json", exits)
    require(all(value["exit_code"] == 0 for value in exits), "baseline producer failure; preserve partial outputs")
    return {"status": "completed", "phase": phase, "images": len(phase_packet["image_ids"]), "output": str(output)}


def baseline_merge(*, packet_path: str | Path, phase: str) -> dict[str, Any]:
    """Merge completed raw shards without reparsing; the cold consumer follows."""

    packet_path = Path(packet_path)
    packet = read(packet_path)
    phase_packet = _baseline_phase(packet, phase)
    output = Path(phase_packet["run_root"])
    exits = read(output / "outer-exits.json")
    require(len(exits) == 2 and all(value["exit_code"] == 0 for value in exits), "baseline outer exits")
    rows: list[dict[str, Any]] = []
    terminals = []
    for shard in (0, 1):
        shard_root = output / f"shard-{shard}"
        terminal = read(shard_root / "terminal.json")
        require(terminal.get("status") == "completed" and terminal.get("exit_code") == 0, "baseline shard terminal")
        if packet.get("packet_role") == "remaining254_continuation" and phase == "full":
            model_receipt = read(shard_root / "model.json")
            source_receipt = read(shard_root / "source.json")
            require(model_receipt.get("status") == "live_model_loaded", "continuation live model receipt")
            require(source_receipt.get("status") == "frozen_input_bound", "continuation live source receipt")
            require(model_receipt.get("packet") == binding(packet_path), "continuation model packet identity")
            require(source_receipt.get("packet") == binding(packet_path), "continuation source packet identity")
            require(source_receipt.get("image_ids") == [int(value) for value in phase_packet["image_ids"][shard::2]], "continuation source identity")
        terminals.append(binding(shard_root / "terminal.json"))
        shard_rows = read_jsonl(shard_root / "rows.jsonl")
        require(len(shard_rows) == terminal["expected_continuations"] == len(phase_packet["image_ids"][shard::2]), "baseline shard row count")
        rows.extend(shard_rows)
    expected = [int(value) for value in phase_packet["image_ids"]]
    by_id = {int(row["image_id"]): row for row in rows}
    require(len(by_id) == len(rows) == len(expected) and set(by_id) == set(expected), "baseline merged identity")
    ordered = [by_id[image_id] for image_id in expected]
    rows_path = output / "rows.jsonl"
    require(not rows_path.exists(), "baseline merged rows occupied")
    with rows_path.open("x") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "schema": "native_owner_scale_state.evaluation.baseline_merge.v1",
        "status": "baseline_raw_merged_pending_cold_consumer",
        "packet": binding(packet_path),
        "phase": phase,
        "image_ids": expected,
        "rows": binding(rows_path),
        "terminals": terminals,
        "consumer_command": f"python -m probes.native_owner_scale.evaluation consume-baseline --packet {packet_path} --phase {phase} --rows {rows_path} --output {phase_packet['consumer_root']}",
    }
    publish(output / "merge.json", result)
    return result


def baseline_receipts(*, packet_path: str | Path, phase: str) -> dict[str, Any]:
    """Backfill sealed source/model receipts when an older worker omitted them."""

    packet_path = Path(packet_path)
    packet = read(packet_path)
    phase_packet = _baseline_phase(packet, phase)
    output = Path(phase_packet["run_root"])
    rows_path = output / "rows.jsonl"
    require(rows_path.exists(), "baseline merged rows missing")
    require(read(output / "merge.json").get("status") == "baseline_raw_merged_pending_cold_consumer", "baseline merge receipt")
    rows = read_jsonl(rows_path)
    expected_ids = [int(value) for value in phase_packet["image_ids"]]
    require([int(row["image_id"]) for row in rows] == expected_ids, "baseline receipt row identity")
    model_path = output / "model.json"
    source_path = output / "source.json"
    require(not model_path.exists() and not source_path.exists(), "baseline receipts already occupied")
    terminals = [binding(output / f"shard-{shard}" / "terminal.json") for shard in (0, 1)]
    terminal_values = [read(output / f"shard-{shard}" / "terminal.json") for shard in (0, 1)]
    require(all(value.get("status") == "completed" and value.get("exit_code") == 0 for value in terminal_values), "baseline terminal receipt")
    adapter_fingerprints = sorted({str(row["adapter_fingerprint"]) for row in rows})
    require(adapter_fingerprints == [packet["adapter"]["fingerprint"]], "baseline row adapter identity")
    publish(
        model_path,
        {
            "schema": "native_owner_scale_state.evaluation.baseline_model_receipt.v1",
            "status": "sealed_model_identity_from_completed_workers",
            "packet": binding(packet_path),
            "evaluation_packet": binding(packet["input_packet"]["path"]),
            "phase": phase,
            "adapter": packet["adapter"],
            "model": packet["model"],
            "generation": packet["generation"],
            "terminals": terminals,
            "observed_worker_counts": {
                "model_loads": sum(int(value["model_loads"]) for value in terminal_values),
                "continuations": sum(int(value["continuations"]) for value in terminal_values),
                "model_forwards": sum(int(value["model_forwards"]) for value in terminal_values),
                "image_forwards": sum(int(value["image_forwards"]) for value in terminal_values),
                "adapter_fingerprints_in_rows": adapter_fingerprints,
            },
            "receipt_boundary": "The original worker completed before live identity serialization was added; this CPU receipt binds the sealed packet, completed terminal counts and row adapter fingerprint, but is not a replacement for an omitted live identity object.",
        },
    )
    evaluation = read(packet["input_packet"]["path"])
    records = {int(record["image_id"]): record for record in evaluation["records"]}
    publish(
        source_path,
        {
            "schema": "native_owner_scale_state.evaluation.baseline_source_receipt.v1",
            "status": "sealed_source_identity_from_completed_rows",
            "packet": binding(packet_path),
            "evaluation_packet": binding(packet["input_packet"]["path"]),
            "selection": binding(evaluation["selection"]["path"]),
            "phase": phase,
            "image_ids": expected_ids,
            "records": [
                {
                    "image_id": image_id,
                    "example_id": records[image_id]["example_id"],
                    "image_path": records[image_id]["case"]["image_path"],
                    "prompt_token_ids_sha256": digest(records[image_id]["prompt_token_ids"]),
                    "executed_media_sha256": records[image_id]["case"]["image_plan"]["executed_media_sha256"],
                    "observed_image_grid_thw": records[image_id]["case"]["image_plan"]["observed_image_grid_thw"],
                }
                for image_id in expected_ids
            ],
            "native_source": evaluation["native_source"],
            "receipt_boundary": "CPU projection of the sealed input and endpoint row identities; no source selection or output-based substitution.",
        },
    )
    return {"status": "baseline_receipts_bound", "model": binding(model_path), "source": binding(source_path)}


def consume_baseline(
    *, packet_path: str | Path, phase: str, rows_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Cold-consume one Stable50 phase without requiring a candidate arm."""

    from transformers import AutoTokenizer

    packet_path = Path(packet_path)
    packet = read(packet_path)
    phase_packet = _baseline_phase(packet, phase)
    rows_path = Path(rows_path)
    require(rows_path.exists(), "baseline merged rows missing")
    output = Path(output)
    require(not output.exists(), "baseline consumer output occupied")
    eval_packet = read(packet["input_packet"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(eval_packet["model"]["base_model_path"], local_files_only=True)
    expected = {int(record["image_id"]): record for record in eval_packet["records"] if int(record["image_id"]) in set(phase_packet["image_ids"])}
    rows = read_jsonl(rows_path)
    require(len(rows) == len(expected), "baseline consumer denominator")
    actual_ids = [int(row["image_id"]) for row in rows]
    require(actual_ids == [int(value) for value in phase_packet["image_ids"]], "baseline consumer order/identity")
    checked = []
    eval_sha = file_hash(Path(packet["input_packet"]["path"]))
    adapter_fp = packet["adapter"]["fingerprint"]
    baseline_sha = file_hash(packet_path)
    for row in rows:
        require(row.get("schema") == "native_owner_scale_state.evaluation.baseline_natural.v1", "baseline row schema")
        require(row.get("phase") == phase and row.get("baseline_packet_sha256") == baseline_sha, "baseline row packet/phase identity")
        checked.append(_validate_natural_row(row, expected[int(row["image_id"])], tokenizer, BASELINE_ARM, eval_sha, adapter_fp))

    continuation_meta: dict[str, Any] | None = None
    live_receipts: list[dict[str, Any]] = []
    if packet.get("packet_role") == "remaining254_continuation" and phase == "full":
        continuation = packet["continuation"]
        parent_binding = continuation["parent_packet"]
        _verify_file_binding(parent_binding, "continuation parent packet")
        completed = continuation["completed_slice"]
        completed_rows_path = _verify_file_binding(completed["rows"], "continuation slice rows")
        completed_consumer_path = _verify_file_binding(completed["consumer"], "continuation slice consumer")
        completed_result_path = _verify_file_binding(completed["result"], "continuation slice result")
        completed_rows = read_jsonl(completed_rows_path)
        completed_consumer = read(completed_consumer_path)
        completed_result = read(completed_result_path)
        require(completed_result.get("status") == "cold_verified_baseline_only" and completed_result.get("images") == 2, "continuation completed slice gate")
        require(completed_result.get("packet") == parent_binding, "continuation completed slice packet")
        require([int(row["image_id"]) for row in completed_rows] == list(BASELINE_SLICE_IDS), "continuation completed raw identity")
        require([int(row["image_id"]) for row in completed_consumer] == list(BASELINE_SLICE_IDS), "continuation completed consumer identity")
        parent_sha = parent_binding["sha256"]
        for row in completed_rows + completed_consumer:
            require(row.get("arm") == BASELINE_ARM, "continuation completed arm")
            require(row.get("adapter_fingerprint") == adapter_fp, "continuation completed adapter identity")
            require(row.get("baseline_packet_sha256") == parent_sha, "continuation completed packet identity")
        panel_ids = [int(value) for value in read(packet["selection"]["path"])["image_ids"]]
        _validate_panel_union(BASELINE_SLICE_IDS, [int(value) for value in phase_packet["image_ids"]], panel_ids)
        by_id = {int(row["image_id"]): row for row in completed_consumer + checked}
        require(len(by_id) == PANEL_SIZE and set(by_id) == set(panel_ids), "continuation combined denominator/identity")
        checked = [by_id[image_id] for image_id in panel_ids]
        for shard in (0, 1):
            shard_root = Path(phase_packet["run_root"]) / f"shard-{shard}"
            model_path = _verify_file_binding(binding(shard_root / "model.json"), "continuation live model receipt")
            source_path = _verify_file_binding(binding(shard_root / "source.json"), "continuation live source receipt")
            model_receipt = read(model_path)
            source_receipt = read(source_path)
            require(model_receipt.get("status") == "live_model_loaded", "continuation live model receipt status")
            require(source_receipt.get("status") == "frozen_input_bound", "continuation live source receipt status")
            require(source_receipt.get("native_source") == eval_packet["native_source"], "continuation source binding")
            live_receipts.append({"shard": shard, "model": binding(model_path), "source": binding(source_path)})
        continuation_meta = {
            "parent_packet": parent_binding,
            "completed_slice": {
                "images": 2,
                "rows": completed["rows"],
                "consumer": completed["consumer"],
                "result": completed["result"],
                "evidence_boundary": "Original slice was cold-verified before live model/source serialization; its CPU receipts are not relabeled as live-object receipts.",
            },
            "remaining254": {
                "images": len(phase_packet["image_ids"]),
                "live_model_source_receipts": live_receipts,
            },
            "combined_images": PANEL_SIZE,
            "slice_plus_remaining_equals_panel": True,
        }
    output.mkdir(parents=True)
    result = {
        "schema": "native_owner_scale_state.evaluation.baseline_result.v2" if continuation_meta else "native_owner_scale_state.evaluation.baseline_result.v1",
        "status": "cold_verified_baseline_only",
        "arm": BASELINE_ARM,
        "phase": phase,
        "phase_role": "remaining254_continuation" if continuation_meta else phase,
        "packet": binding(packet_path),
        "evaluation_packet": binding(packet["input_packet"]["path"]),
        "rows": binding(rows_path),
        "images": len(checked),
        "quality": aggregate_scores([row["score"] for row in checked]),
        "burden": accepted.burden(checked),
        "denominators": {"fresh256": PANEL_SIZE, "slice": len(BASELINE_SLICE_IDS), "remaining254": PANEL_SIZE - len(BASELINE_SLICE_IDS)},
        "claim_boundary": "Unchanged Stable50 baseline only; no paired candidate claim, no GT edit, and no hallucination inference from unmatched predictions.",
        "next_step": "No remaining Stable50 phase; candidate pairing remains conditional on A's admitted terminal adapter." if continuation_meta else "After slice cold verification, launch full phase exactly on the remaining254 IDs; never rerun the frozen slice.",
    }
    publish(output / "consumer.json", checked)
    if continuation_meta:
        result["continuation"] = continuation_meta
        result["combined_consumer"] = binding(output / "consumer.json")
    publish(output / "result.json", result)
    return result


def validate_candidate_binding(candidate: Mapping[str, Any]) -> None:
    require(candidate.get("label") == "scaled_terminal", "candidate endpoint label")
    require(int(candidate.get("admitted_packages", 0)) >= 16, "candidate below N16 training stop")
    adapter = candidate.get("adapter")
    require(isinstance(adapter, Mapping) and adapter.get("root") and adapter.get("fingerprint"), "candidate adapter identity")
    require(candidate.get("status") in {"cold_verified", "ready_for_launch", "completed"}, "candidate endpoint not cold-ready")
    require(candidate.get("max_new_tokens", CAP) == CAP, "candidate cap changed")
    require(candidate.get("dtype", "fp32") == "fp32" and candidate.get("attention", "sdpa") == "sdpa", "candidate numerics changed")
    if candidate.get("schema") == "native_owner_scale_state.evaluation.candidate_endpoint.v1":
        require(candidate.get("fit", {}).get("updates") == 256, "candidate fit schedule")
        require(candidate.get("fit", {}).get("world_size") == 8, "candidate fit world size")
        for name in ("training_receipt", "cold_check", "training_completion", "training_input"):
            require(isinstance(candidate.get(name), Mapping) and candidate[name].get("path"), f"candidate {name} binding")
        require(candidate.get("claim_boundary") and "natural" in candidate["claim_boundary"], "candidate claim boundary")


def _verify_adapter_files(adapter: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Verify saved adapter payload files without loading model weights."""

    require(isinstance(adapter, Mapping) and adapter.get("root"), f"missing {label} root")
    root = Path(str(adapter["root"]))
    files = adapter.get("files", [])
    require(root.is_dir() and isinstance(files, list) and files, f"missing {label} files")
    require(int(adapter.get("file_count", 0)) == len(files), f"{label} file count")
    for descriptor in files:
        require(isinstance(descriptor, Mapping), f"{label} file descriptor")
        relative = str(descriptor.get("relative_path", ""))
        require(relative and not Path(relative).is_absolute(), f"{label} relative file path")
        actual = binding(root / relative)
        require(actual["sha256"] == descriptor.get("sha256"), f"{label} file hash changed")
        require(actual["size_bytes"] == descriptor.get("size_bytes"), f"{label} file size changed")
    return {"root": str(root), "fingerprint": str(adapter["fingerprint"]), "files": copy.deepcopy(files)}


def _candidate_launch_contract(
    *, packet_path: str | Path, output_root: str | Path, physical_gpus: Sequence[int] = CANDIDATE_GPUS
) -> dict[str, Any]:
    """Build the exact bounded natural producer/merge/consumer entry commands."""

    packet_path = Path(packet_path)
    output_root = Path(output_root)
    gpus = [int(value) for value in physical_gpus]
    require(len(gpus) == 8 and len(set(gpus)) == 8, "candidate physical GPU denominator")
    shards = list(range(len(gpus)))
    image_counts = [COMBINED_PANEL_SIZE // len(gpus) + (index < COMBINED_PANEL_SIZE % len(gpus)) for index in shards]
    worker_commands = [
        [
            sys.executable,
            "-m",
            "probes.native_owner_scale.evaluation",
            "candidate-worker",
            "--packet",
            str(packet_path),
            "--shard",
            str(shard),
            "--physical-gpu",
            str(gpu),
            "--output",
            str(output_root / f"shard-{shard}"),
        ]
        for shard, gpu in zip(shards, gpus, strict=True)
    ]
    launch_command = [
        f"CUDA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in gpus)}",
        sys.executable,
        "-m",
        "probes.native_owner_scale.evaluation",
        "candidate-launch",
        "--packet",
        str(packet_path),
        "--output",
        str(output_root),
    ]
    merge_command = [
        sys.executable,
        "-m",
        "probes.native_owner_scale.evaluation",
        "candidate-merge",
        "--packet",
        str(packet_path),
        "--output",
        str(output_root),
    ]
    consume_output = output_root.with_name(f"{output_root.name}-consumer")
    consume_command = [
        sys.executable,
        "-m",
        "probes.native_owner_scale.evaluation",
        "consume-candidate-panel",
        "--packet",
        str(packet_path),
        "--candidate-rows",
        str(output_root / "rows.jsonl"),
        "--output",
        str(consume_output),
    ]
    return {
        "module": "probes.native_owner_scale.evaluation",
        "physical_gpus": gpus,
        "workers": len(gpus),
        "image_count": COMBINED_PANEL_SIZE,
        "image_counts_per_worker": image_counts,
        "max_new_tokens_per_image": CAP,
        "max_new_tokens_total": COMBINED_PANEL_SIZE * CAP,
        "max_new_tokens_per_worker": max(image_counts) * CAP,
        "max_model_forwards_per_worker": max(image_counts) * (CAP + 1),
        "max_image_forwards_per_worker": max(image_counts),
        "model_loads_per_worker": 1,
        "max_rank_seconds": CANDIDATE_MAX_RANK_SECONDS,
        "max_cuda_allocated_bytes": CANDIDATE_MAX_CUDA_ALLOCATED_BYTES,
        "max_cuda_reserved_bytes": CANDIDATE_MAX_CUDA_RESERVED_BYTES,
        "max_rss_bytes": CANDIDATE_MAX_RSS_BYTES,
        "retry_policy": "none; preserve partial shard and terminal artifacts",
        "natural_policy": "greedy temperature0/top-p1/top-k0/RP1 FP32/SDPA, empty prefix, full 3084 allowance",
        "launch_command": launch_command,
        "worker_commands": worker_commands,
        "merge_command": merge_command,
        "consume_command": consume_command,
        "output_root": str(output_root),
        "consumer_output": str(consume_output),
    }


def prepare_candidate_endpoint(
    *,
    panel_path: str | Path,
    completion_path: str | Path,
    output_path: str | Path,
    natural_output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize A's actual training/cold receipts into the pending candidate schema."""

    panel_path = Path(panel_path)
    completion_path = Path(completion_path)
    output_path = Path(output_path)
    require(not output_path.exists(), "candidate endpoint output occupied")
    panel = read(panel_path)
    _validate_candidate_panel_packet(panel)
    require(panel.get("status") == "candidate_adapter_pending", "candidate panel already bound")
    completion = read(completion_path)
    require(completion.get("schema") == "native_owner_scale.training_completion.v1", "training completion schema")
    require(completion.get("status") == "technically_completed_and_cold_passed_evaluation_pending", "training completion status")
    receipt_binding = _verify_declared_sha_binding(completion.get("receipt", {}), "candidate training receipt")
    cold_binding = _verify_declared_sha_binding(completion.get("cold_check", {}), "candidate cold check")
    input_binding = _verify_declared_sha_binding(completion.get("input", {}), "candidate training input")
    receipt_path = Path(receipt_binding["path"])
    cold_path = Path(cold_binding["path"])
    input_path = Path(input_binding["path"])
    for key in ("gate", "root_acceptance"):
        if key in completion:
            _verify_declared_sha_binding(completion[key], f"candidate {key}")
    receipt = read(receipt_path)
    cold = read(cold_path)
    training_input = read(input_path)
    require(receipt.get("schema") == "parallel_owner_training.receipt.v1", "candidate receipt schema")
    require(receipt.get("status") == "technically_completed_cold_pending", "candidate receipt status")
    require(receipt.get("arm") == "native_owner_scale_fixedP", "candidate receipt arm")
    require(int(receipt.get("updates", 0)) == 256 and int(receipt.get("world_size", 0)) == 8, "candidate receipt schedule")
    require(len(receipt.get("step_records", [])) == 256 and len(receipt.get("terminals", [])) == 8, "candidate receipt records")
    require(cold.get("schema") == "parallel_owner_training.cold_check.v1" and cold.get("status") == "passed", "candidate cold check")
    require(cold.get("training_receipt", {}).get("path") == str(receipt_path)
            and cold.get("training_receipt", {}).get("sha256") == file_hash(receipt_path), "candidate cold receipt identity")
    require(cold.get("saved_adapter") == receipt.get("saved_adapter") == completion.get("saved_adapter"), "candidate adapter identity")
    require(int(cold.get("model_loads", 0)) == 1 and int(cold.get("score_forwards", 0)) == TRUSTED_TARGET_COUNT, "candidate cold counters")
    require(
        all(all(float(delta) == 0.0 for delta in row.values()) for row in cold.get("live_score_deltas", {}).values()),
        "candidate cold score deltas",
    )
    require(training_input.get("schema") == "parallel_owner_training.inputs.v1", "candidate training input schema")
    require(len(training_input.get("positive_records", [])) == TRUSTED_TARGET_COUNT, "candidate positive denominator")
    require(len(training_input.get("normal_keys", [])) == REFERENCE_IMAGE_COUNT, "candidate normal denominator")
    receipt_input = receipt.get("input", {})
    require(receipt_input.get("path") == str(input_path) and receipt_input.get("sha256") == file_hash(input_path), "candidate input identity")
    panel_input = panel["training_input_identity_only"]["inputs_v2"]
    require(panel_input.get("path") == str(input_path) and panel_input.get("sha256") == file_hash(input_path), "candidate panel/input identity")
    expected_positive_ids = {str(record["record_id"]) for record in training_input["positive_records"]}
    require(set(receipt["final_live_positive_scores"]) == expected_positive_ids, "candidate positive score identity")
    model_identity = receipt.get("model_identity", {})
    backend = model_identity.get("backend", {}).get("hf", {})
    require(model_identity.get("base_model") == panel["model"]["base_model_path"], "candidate base model identity")
    require(model_identity.get("dtype") == "fp32" and backend.get("attn_implementation") == "sdpa", "candidate model numerics")
    adapter = copy.deepcopy(receipt["saved_adapter"])
    _verify_adapter_files(adapter, "candidate adapter")
    require(adapter.get("files", [{}])[0].get("sha256") == completion["saved_adapter"]["files"][0].get("sha256"), "candidate adapter manifest")
    require(panel["generation"]["max_new_tokens"] == CAP and panel["generation"]["natural_prefix_ids"] == [], "candidate generation contract")
    if natural_output_root is None:
        tag = panel_path.stem.removeprefix("candidate-panel-")
        natural_output_root = ROOT / f"candidate-natural-{tag}"
    launch_contract = _candidate_launch_contract(packet_path=panel_path, output_root=natural_output_root)
    candidate = {
        "schema": "native_owner_scale_state.evaluation.candidate_endpoint.v1",
        "status": "cold_verified",
        "label": "scaled_terminal",
        "admitted_packages": TRUSTED_TARGET_COUNT,
        "adapter": adapter,
        "training_receipt": binding(receipt_path),
        "cold_check": binding(cold_path),
        "training_completion": binding(completion_path),
        "training_input": binding(input_path),
        "fit": {
            "arm": receipt["arm"],
            "updates": int(receipt["updates"]),
            "world_size": int(receipt["world_size"]),
            "stop_reason": receipt.get("stop_reason"),
            "positive_records": TRUSTED_TARGET_COUNT,
            "normal_reference_records": REFERENCE_IMAGE_COUNT,
        },
        "model": {
            "base_model_path": model_identity["base_model"],
            "dtype": model_identity["dtype"],
            "attention": backend["attn_implementation"],
            "patch_embed_linearization": backend.get("patch_embed_linearization"),
            "merged_adapters": [],
        },
        "generation": copy.deepcopy(panel["generation"]),
        "max_new_tokens": CAP,
        "dtype": "fp32",
        "attention": "sdpa",
        "resource_bounds": launch_contract,
        "launch_contract": launch_contract,
        "fit_claim_boundary": completion["claim_boundary"],
        "claim_boundary": "Cold-loaded N16 adapter is mechanically bound; natural owner recovery, retention, and trusted-target recovery remain unmeasured until root consumes the 640 natural rows. Fit evidence is not recovery credit.",
        "producer": binding(Path(__file__).resolve()),
    }
    validate_candidate_binding(candidate)
    publish(output_path, candidate)
    return candidate


def bind_candidate(
    *, packet_path: str | Path, candidate_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Seal A's one terminal endpoint without changing the frozen selection."""

    packet_path = Path(packet_path)
    candidate_path = Path(candidate_path)
    output_path = Path(output_path) if output_path is not None else packet_path.with_name("packet-bound.json")
    packet = read(packet_path)
    candidate = read(candidate_path)
    validate_candidate_binding(candidate)
    bound = dict(candidate)
    bound["receipt"] = binding(candidate_path)
    packet["scaled_terminal"] = bound
    packet["status"] = "ready_for_root_launch_gate"
    # The original CPU packet remains immutable; only the pending endpoint arm
    # is filled in a new bound packet after A's terminal receipt exists.
    require(not output_path.exists(), "bound evaluation packet already exists")
    packet["base_packet"] = binding(packet_path)
    packet["selection"] = dict(packet["selection"])
    publish(output_path, packet)
    return packet


_SEALED_STABLE_FIELDS = (
    "action_ids",
    "image_id",
    "example_id",
    "stop_reason",
    "text",
    "parsed",
    "score",
    "overlap_counts",
)


def _write_jsonl_exclusive(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write a small immutable JSONL artifact without silently replacing it."""

    path = Path(path)
    require(not path.exists(), f"JSONL artifact already occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return binding(path)


def _sealed_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    require(all(field in row for field in _SEALED_STABLE_FIELDS), "sealed Stable50 row missing lossless field")
    return {field: copy.deepcopy(row[field]) for field in _SEALED_STABLE_FIELDS}


def _annotate_projection(
    row: Mapping[str, Any],
    *,
    source_record: Mapping[str, Any],
    source_kind: str,
    source_file_sha256: str,
    source_adapter_fingerprint: str,
) -> dict[str, Any]:
    """Add provenance to a projected row without changing sealed endpoint fields."""

    projected = copy.deepcopy(dict(row))
    projected["projection_schema"] = "native_owner_scale_state.evaluation.sealed_stable_row.v1"
    projected["source_kind"] = source_kind
    projected["source_record_sha256"] = digest(source_record)
    projected["source_file_sha256"] = source_file_sha256
    projected["source_adapter_fingerprint"] = source_adapter_fingerprint
    projected["projection_exact_fields_sha256"] = digest(_sealed_fields(projected))
    return projected


def _project_legacy_stable_row(
    source: Mapping[str, Any], *, source_packet_sha256: str, source_adapter_fingerprint: str
) -> dict[str, Any]:
    """Project one embedded legacy ``eval_records`` trace to the native row shape."""

    parsed = source["stable_parsed"]
    plan = source["case"]["image_plan"]
    require(parsed.get("raw_decode_text") is not None, "legacy Stable50 trace has no raw decode text")
    require(parsed.get("decode_stop_reason") is not None, "legacy Stable50 trace has no stop reason")
    row = {
        "schema": "native_owner_scale_state.evaluation.sealed_stable_row.v1",
        "arm": "Stable50",
        "example_id": str(source["example_id"]),
        "image_id": int(source["image_id"]),
        "split": source.get("split"),
        "action_ids": list(source["stable_ids"]),
        "prefix_ids": [],
        "forced_ids": [],
        "remaining_budget": CAP,
        "stop_reason": parsed["decode_stop_reason"],
        "text": parsed["raw_decode_text"],
        "parsed": copy.deepcopy(parsed),
        "score": copy.deepcopy(source["stable_score"]),
        "overlap_counts": copy.deepcopy(source["stable_overlap_counts"]),
        "request_id": str(source["example_id"]),
        # This is the identity recorded with the old endpoint, not a new live
        # object receipt.  The rename-only bridge is kept in the sidecar.
        "adapter_fingerprint": source_adapter_fingerprint,
        "source_adapter_fingerprint": source_adapter_fingerprint,
        "packet_sha256": source_packet_sha256,
        "source_packet_sha256": source_packet_sha256,
        "prompt_token_ids_sha256": digest(source["prompt_token_ids"]),
        "executed_media_sha256": plan["executed_media_sha256"],
        "observed_image_grid_thw": plan["observed_image_grid_thw"],
    }
    return row


def _training_strata_ids(
    training_input: Mapping[str, Any], legacy_records: Sequence[Mapping[str, Any]]
) -> tuple[set[int], set[int]]:
    """Resolve admitted and reference identities from the authoritative v2 input."""

    require(training_input.get("schema") == "parallel_owner_training.inputs.v1", "training input schema")
    require(training_input.get("status") == "prepared_no_model_execution", "training input is not CPU-prepared")
    positives = training_input.get("positive_records", [])
    normal_keys = training_input.get("normal_keys", [])
    require(len(positives) == 16, "training positive-record denominator")
    require(len(normal_keys) == REFERENCE_IMAGE_COUNT and len(set(normal_keys)) == REFERENCE_IMAGE_COUNT, "training reference denominator")
    target_ids = {
        int(record["image"]["image_id"])
        for record in positives
    }
    require(len(target_ids) == ADMITTED_IMAGE_COUNT, "training admitted image denominator")
    legacy_by_example = {str(record["example_id"]): record for record in legacy_records}
    require(len(legacy_by_example) == len(legacy_records), "legacy example identity")
    reference_ids: set[int] = set()
    for key in normal_keys:
        record = legacy_by_example.get(str(key))
        require(record is not None, f"training reference missing from exposed packet: {key}")
        reference_ids.add(int(record["image_id"]))
    require(len(reference_ids) == REFERENCE_IMAGE_COUNT, "training reference image identity")
    require(not target_ids & reference_ids, "admitted/reference strata overlap")
    return target_ids, reference_ids


def _verify_declared_sha_binding(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Verify a historical path/hash declaration that predates size-bearing bindings."""

    require(isinstance(value, Mapping) and value.get("path") and value.get("sha256"), f"missing {label} binding")
    current = binding(value["path"])
    require(current["sha256"] == value["sha256"], f"{label} changed")
    return current


def _trusted_target_score(
    row: Mapping[str, Any], target_objects: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score natural predictions against reviewed c targets, never against row GT.

    ``target_objects`` use the accepted scorer's coordinate-bin GT shape, so
    ``score`` performs the same class-normalization and native-pixel IoU
    matching as ordinary endpoint metrics.  All targets from one image are
    scored jointly; global assignment prevents one duplicate prediction from
    being credited to multiple reviewed owners.
    """

    parsed = row.get("parsed", row)
    require(isinstance(parsed, Mapping), "trusted-target parsed row")
    scoring_row = {
        "row_id": str(row.get("example_id", row.get("row_id", "trusted-target"))),
        "image_width": int(parsed.get("image_width", row.get("image_width", 0))),
        "image_height": int(parsed.get("image_height", row.get("image_height", 0))),
        "gt": [copy.deepcopy(target) for target in target_objects],
        "pred": copy.deepcopy(parsed.get("pred", [])),
        "dropped_prediction_count": int(parsed.get("dropped_prediction_count", 0)),
    }
    require(scoring_row["image_width"] > 0 and scoring_row["image_height"] > 0, "trusted-target image dimensions")
    return score(
        scoring_row,
        seed=None,
        length=len(row.get("action_ids", [])),
        stop=str(row.get("stop_reason", parsed.get("decode_stop_reason", "im_end"))),
    )


def _build_trusted_target_ledger(
    *, output_path: str | Path, training_input_path: str | Path = TRAINING_INPUT_V2
) -> dict[str, Any]:
    """Freeze the 16 reviewed c packages and their GT-independent target geometry."""

    from transformers import AutoTokenizer
    from src.inference.parsing import parse_compact_object_box_closed

    output_path = Path(output_path)
    require(not output_path.exists(), "trusted-target ledger already occupied")
    training_input_path = Path(training_input_path)
    training_input = read(training_input_path)
    positive_records = training_input.get("positive_records", [])
    require(len(positive_records) == TRUSTED_TARGET_COUNT, "trusted-target positive-record denominator")
    require(training_input.get("status") == "prepared_no_model_execution", "trusted-target training input status")
    acquisition_declarations = [record["candidate_owner_provenance"]["acquisition_result"] for record in positive_records]
    review_declarations = [record["candidate_owner_provenance"]["visual_review"] for record in positive_records]
    require(len({(item["path"], item["sha256"]) for item in acquisition_declarations}) == 1, "trusted-target acquisition binding drift")
    require(len({(item["path"], item["sha256"]) for item in review_declarations}) == 1, "trusted-target review binding drift")
    acquisition_path = Path(acquisition_declarations[0]["path"])
    review_path = Path(review_declarations[0]["path"])
    acquisition_binding = _verify_declared_sha_binding(acquisition_declarations[0], "trusted-target acquisition")
    review_binding = _verify_declared_sha_binding(review_declarations[0], "trusted-target final review")
    acquisition = read(acquisition_path)
    review = read(review_path)
    require(acquisition.get("schema") == "native_owner_scale.acquisition_result.v1", "trusted-target acquisition schema")
    require(review.get("schema") == "native_owner_scale.visual_reviews.final.v1", "trusted-target final review schema")
    require(review.get("status") == "lead_accepted_physical_bank_admission", "trusted-target final review status")
    accepted_priorities = [int(value) for value in review.get("accepted_priorities_frozen_order", [])]
    require(len(accepted_priorities) == TRUSTED_TARGET_COUNT and len(set(accepted_priorities)) == TRUSTED_TARGET_COUNT, "trusted-target accepted priority denominator")
    acquisition_rows = acquisition.get("rows", [])
    by_job = {str(row["job_id"]): row for row in acquisition_rows}
    require(len(by_job) == len(acquisition_rows), "trusted-target acquisition duplicate job")
    tokenizer = AutoTokenizer.from_pretrained(
        read(STABLE_PACKET)["model"]["base_model_path"], local_files_only=True
    )
    package_ids: set[str] = set()
    owner_ids: set[str] = set()
    targets: list[dict[str, Any]] = []
    for ordinal, record in enumerate(positive_records):
        provenance = record.get("candidate_owner_provenance")
        require(isinstance(provenance, Mapping), "trusted-target provenance missing")
        case_id = str(record["case_id"])
        record_id = str(record["record_id"])
        owner_id = str(provenance.get("owner_id", ""))
        require(case_id == record_id.removesuffix(":c"), "trusted-target case/record identity")
        require(case_id not in package_ids and owner_id, "trusted-target package/owner duplicate")
        package_ids.add(case_id)
        owner_ids.add(owner_id)
        acquisition_row = by_job.get(case_id)
        require(acquisition_row is not None, f"trusted-target acquisition row missing: {case_id}")
        require(str(acquisition_row["example_id"]) == str(record["example_id"]), "trusted-target acquisition example identity")
        require(str(acquisition_row["owner_id"]) == owner_id, "trusted-target acquisition owner identity")
        priority = int(acquisition_row["admission_priority"])
        require(priority in accepted_priorities, "trusted-target admission priority not accepted")
        decision = review.get("decisions", {}).get(case_id)
        require(isinstance(decision, Mapping) and decision.get("status") == "accept", "trusted-target final decision")
        require(decision.get("w_single_owner_nonduplicate") is True, "trusted-target witness decision")
        target_token_ids = record.get("target_token_ids")
        require(isinstance(target_token_ids, list) and target_token_ids, "trusted-target token sequence")
        image = record["image"]
        target_text = tokenizer.decode(target_token_ids, skip_special_tokens=False)
        target_parsed = parse_compact_object_box_closed(
            target_text,
            row_id=str(record["example_id"]),
            row_index=int(image["row_index"]),
            image_width=int(image["image_width"]),
            image_height=int(image["image_height"]),
        ).to_artifact_dict()
        predictions = target_parsed["predictions"]
        require(target_parsed["parse_status"] == "accepted" and target_parsed["dropped_prediction_count"] == 0 and len(predictions) == 1, "trusted-target c parser")
        prediction = predictions[0]
        scoring_object = {
            "object_id": owner_id,
            "description": str(prediction["description"]),
            "bbox": list(prediction["coord_bins"]),
        }
        targets.append(
            {
                "ordinal": ordinal,
                "record_id": record_id,
                "case_id": case_id,
                "example_id": str(record["example_id"]),
                "image_id": int(image["image_id"]),
                "owner_id": owner_id,
                "provenance_kind": str(provenance["kind"]),
                "image": {
                    "image_id": int(image["image_id"]),
                    "image_path": str(image["image_path"]),
                    "image_sha256": str(image["image_sha256"]),
                    "image_width": int(image["image_width"]),
                    "image_height": int(image["image_height"]),
                    "executed_media_sha256": str(image["executed_media_sha256"]),
                    "observed_image_grid_thw": list(image["observed_image_grid_thw"]),
                },
                "target": {
                    "token_ids": list(target_token_ids),
                    "token_ids_sha256": digest(target_token_ids),
                    "text": target_text,
                    "parsed": target_parsed,
                    "prediction": copy.deepcopy(prediction),
                    "target_bbox_pixels": list(prediction["bbox"]),
                    "scoring_object": scoring_object,
                },
                "acquisition": {
                    "job_id": str(acquisition_row["job_id"]),
                    "owner_id": str(acquisition_row["owner_id"]),
                    "admission_priority": priority,
                    "row_sha256": digest(acquisition_row),
                },
                "final_review": {
                    "status": str(decision["status"]),
                    "decision_sha256": digest(decision),
                    "decision": copy.deepcopy(decision),
                },
                "source_record_sha256": digest(record),
                "candidate_owner_provenance": copy.deepcopy(provenance),
            }
        )
    require(set(priority for priority in (item["acquisition"]["admission_priority"] for item in targets)) == set(accepted_priorities), "trusted-target priority union")
    require(len(owner_ids) == TRUSTED_TARGET_COUNT, "trusted-target owner identity")
    ledger = {
        "schema": "native_owner_scale_state.evaluation.trusted_target_ledger.v1",
        "status": "frozen_reviewed_c_targets_no_model_calls",
        "targets": targets,
        "target_count": TRUSTED_TARGET_COUNT,
        "image_count": len({int(item["image_id"]) for item in targets}),
        "source_bindings": {
            "training_input": binding(training_input_path),
            "acquisition_result": acquisition_binding,
            "final_review": review_binding,
        },
        "matching_policy": {
            "thresholds": [50, 60, 80],
            "matcher": "probes.dora_owner_learning.candidate_opportunity.score -> src.eval.assignment.global_matches",
            "class_matching": "existing normalized description class matching",
            "geometry": "existing coordinate-bin-to-native-pixel conversion and IoU",
            "joint_assignment_per_image": True,
            "one_prediction_cannot_recover_two_targets": True,
            "forced_credit": False,
            "gt_independent": True,
        },
        "claim_boundary": "Reviewed c target instances are trusted local-owner targets, not an exhaustive scene ledger and not GT edits; absent/unclear/malformed predictions do not become negative labels.",
    }
    publish(output_path, ledger)
    return ledger


def _trusted_target_panel_summary(
    stable_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Report per-package trusted-owner recovery separately from GT metrics."""

    require(int(ledger.get("target_count", 0)) == TRUSTED_TARGET_COUNT, "trusted-target ledger denominator")
    stable_by_image = {int(row["image_id"]): row for row in stable_rows}
    candidate_by_image = {int(row["image_id"]): row for row in candidate_rows}
    require(len(stable_by_image) == len(stable_rows) and len(candidate_by_image) == len(candidate_rows), "trusted-target endpoint duplicate image")
    by_image: dict[int, list[Mapping[str, Any]]] = {}
    for target in ledger["targets"]:
        by_image.setdefault(int(target["image_id"]), []).append(target["target"]["scoring_object"])
    packages: list[dict[str, Any]] = []
    for target in ledger["targets"]:
        image_id = int(target["image_id"])
        require(image_id in stable_by_image and image_id in candidate_by_image, "trusted-target endpoint missing image")
        target_objects = by_image[image_id]
        stable_score = _trusted_target_score(stable_by_image[image_id], target_objects)
        candidate_score = _trusted_target_score(candidate_by_image[image_id], target_objects)
        owner_id = str(target["owner_id"])
        packages.append(
            {
                "ordinal": int(target["ordinal"]),
                "record_id": target["record_id"],
                "case_id": target["case_id"],
                "example_id": target["example_id"],
                "image_id": image_id,
                "owner_id": owner_id,
                "provenance_kind": target["provenance_kind"],
                "source_record_sha256": target["source_record_sha256"],
                "candidate_owner_provenance": copy.deepcopy(target["candidate_owner_provenance"]),
                "admission_priority": target["acquisition"]["admission_priority"],
                "Stable50": stable_score,
                "scaled_terminal": candidate_score,
            }
        )
    thresholds: dict[str, Any] = {}
    for threshold in ("50", "60", "80"):
        baseline_recovered = [item for item in packages if item["owner_id"] in item["Stable50"][threshold]["owners"]]
        candidate_recovered = [item for item in packages if item["owner_id"] in item["scaled_terminal"][threshold]["owners"]]
        baseline_ids = {item["record_id"] for item in baseline_recovered}
        candidate_ids = {item["record_id"] for item in candidate_recovered}
        thresholds[threshold] = {
            "targets": TRUSTED_TARGET_COUNT,
            "Stable50_recovered": len(baseline_ids),
            "scaled_terminal_recovered": len(candidate_ids),
            "gained": len(candidate_ids - baseline_ids),
            "lost": len(baseline_ids - candidate_ids),
            "retained": len(baseline_ids & candidate_ids),
            "unmatched_neutral": TRUSTED_TARGET_COUNT - len(baseline_ids | candidate_ids),
        }
    return {
        "schema": "native_owner_scale_state.evaluation.trusted_target_recovery.v1",
        "status": "natural_rows_scored_against_frozen_reviewed_targets",
        "targets": TRUSTED_TARGET_COUNT,
        "images": len(by_image),
        "thresholds": thresholds,
        "packages": packages,
        "claim_boundary": "Per-package owner recovery against the frozen reviewed c ledger only; separate from train11 GT aggregate scores, no forced credit, no GT edit, and unmatched rows are neutral rather than negative labels.",
    }


def _candidate_panel_strata(
    records: Sequence[Mapping[str, Any]],
    *,
    admitted_ids: Iterable[int],
    reference_ids: Iterable[int],
    fresh_ids: Iterable[int],
) -> dict[str, dict[str, Any]]:
    """Return deterministic, mutually disjoint image strata for the 640 panel."""

    ids = [int(record["image_id"]) for record in records]
    require(len(ids) == COMBINED_PANEL_SIZE and len(set(ids)) == COMBINED_PANEL_SIZE, "candidate panel denominator/duplicates")
    all_ids = set(ids)
    admitted = {int(value) for value in admitted_ids}
    references = {int(value) for value in reference_ids}
    fresh = {int(value) for value in fresh_ids}
    require(admitted <= all_ids and references <= all_ids and fresh <= all_ids, "candidate strata outside panel")
    require(not admitted & references and not admitted & fresh and not references & fresh, "candidate strata overlap")
    legacy = all_ids - fresh
    require(admitted <= legacy and references <= legacy, "admitted/reference must be legacy identities")
    legacy_retention = legacy - admitted - references
    remaining = legacy_retention | fresh
    require(len(admitted) == ADMITTED_IMAGE_COUNT, "admitted target stratum count")
    require(len(references) == REFERENCE_IMAGE_COUNT, "reference stratum count")
    require(len(legacy_retention) == LEGACY_EXPOSED_SIZE - ADMITTED_IMAGE_COUNT - REFERENCE_IMAGE_COUNT, "legacy retention count")
    require(len(fresh) == PANEL_SIZE and len(remaining) == PANEL_SIZE + len(legacy_retention), "remaining retention count")
    require(admitted | references | remaining == all_ids, "candidate strata union")

    def ordered(values: set[int]) -> list[int]:
        return [image_id for image_id in ids if image_id in values]

    return {
        "train11": {
            "role": "admitted_target",
            "image_ids": ordered(admitted),
            "count": len(admitted),
        },
        "reference54": {
            "role": "reference",
            "image_ids": ordered(references),
            "count": len(references),
        },
        "legacy_retention": {
            "role": "legacy_retention",
            "image_ids": ordered(legacy_retention),
            "count": len(legacy_retention),
        },
        "fresh256": {
            "role": "fresh256",
            "image_ids": ordered(fresh),
            "count": len(fresh),
        },
        "remaining_retention": {
            "role": "remaining_retention",
            "image_ids": ordered(remaining),
            "count": len(remaining),
        },
    }


def _projection_descriptor(
    ordinal: int, row: Mapping[str, Any], *, source_file_sha256: str
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "example_id": str(row["example_id"]),
        "image_id": int(row["image_id"]),
        "source_kind": row["source_kind"],
        "source_record_sha256": row["source_record_sha256"],
        "source_file_sha256": source_file_sha256,
        "projected_row_sha256": digest(row),
        "exact_fields_sha256": digest(_sealed_fields(row)),
    }


def _build_sealed_stable_projection(
    *,
    legacy_packet_path: Path,
    baseline_consumer_path: Path,
    baseline_packet_path: Path,
    baseline_result_path: Path,
    current_packet_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the 384 legacy + 256 sealed-baseline projection without reruns."""

    legacy_binding = binding(legacy_packet_path)
    consumer_binding = binding(baseline_consumer_path)
    baseline_binding = binding(baseline_packet_path)
    baseline_result_binding = binding(baseline_result_path)
    current_binding = binding(current_packet_path)
    legacy_packet = read(legacy_packet_path)
    baseline_packet = read(baseline_packet_path)
    baseline_result = read(baseline_result_path)
    current_packet = read(current_packet_path)
    require(len(legacy_packet.get("eval_records", [])) == LEGACY_EXPOSED_SIZE, "legacy exposed denominator")
    fresh_rows = read(baseline_consumer_path)
    require(isinstance(fresh_rows, list) and len(fresh_rows) == PANEL_SIZE, "sealed fresh baseline denominator")
    require(baseline_packet.get("input_packet") == current_binding, "baseline/current packet binding")
    require(baseline_result.get("evaluation_packet") == current_binding, "baseline result/current packet binding")
    require(baseline_result.get("status") == "cold_verified_baseline_only", "sealed baseline status")
    old_adapter = legacy_packet["model"]["current_adapter"]
    old_packet_sha256 = legacy_binding["sha256"]
    rows: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for source in legacy_packet["eval_records"]:
        projected = _project_legacy_stable_row(
            source, source_packet_sha256=old_packet_sha256, source_adapter_fingerprint=old_adapter["fingerprint"]
        )
        projected = _annotate_projection(
            projected,
            source_record=source,
            source_kind="legacy_exposed_eval_record",
            source_file_sha256=old_packet_sha256,
            source_adapter_fingerprint=old_adapter["fingerprint"],
        )
        rows.append(projected)
        descriptors.append(_projection_descriptor(len(descriptors), projected, source_file_sha256=old_packet_sha256))
    current_records = current_packet["records"]
    current_ids = [int(record["image_id"]) for record in current_records]
    fresh_ids = [int(row["image_id"]) for row in fresh_rows]
    require(fresh_ids == current_ids, "sealed fresh baseline order/identity")
    for source in fresh_rows:
        require(source.get("arm") == BASELINE_ARM, "sealed baseline arm")
        adapter_fingerprint = str(source["adapter_fingerprint"])
        projected = _annotate_projection(
            source,
            source_record=source,
            source_kind="fresh256_baseline_consumer_array",
            source_file_sha256=consumer_binding["sha256"],
            source_adapter_fingerprint=adapter_fingerprint,
        )
        rows.append(projected)
        descriptors.append(_projection_descriptor(len(descriptors), projected, source_file_sha256=consumer_binding["sha256"]))
    require(len(rows) == COMBINED_PANEL_SIZE, "combined stable projection denominator")
    require(len({int(row["image_id"]) for row in rows}) == COMBINED_PANEL_SIZE, "combined stable projection overlap")
    sources = {
        "legacy_exposed_packet": legacy_binding,
        "fresh_baseline_consumer": consumer_binding,
        "baseline_packet": baseline_binding,
        "baseline_result": baseline_result_binding,
        "current_evaluation_packet": current_binding,
    }
    metadata = {
        "schema": "native_owner_scale_state.evaluation.sealed_stable_projection.v2",
        "status": "sealed_no_rerun_lossless_projection",
        "rows": descriptors,
        "sources": sources,
        "exact_fields": list(_SEALED_STABLE_FIELDS),
        "lossless_boundary": "stable_ids/action_ids, image_id, example_id, stop, raw text, parsed fields, score and overlap counts are copied exactly; source adapter identities remain provenance fields and are not relabeled as live receipts",
    }
    return rows, descriptors, metadata


def _validate_sealed_baseline_projection(
    projection_path: str | Path, rows_path: str | Path
) -> list[dict[str, Any]]:
    """Cold-check projection parity and reject dropped, reordered or changed rows."""

    projection_path = Path(projection_path)
    projection = read(projection_path)
    output_binding = projection.get("output_rows")
    require(isinstance(output_binding, Mapping), "projection output binding")
    require(Path(output_binding["path"]).resolve() == Path(rows_path).resolve(), "projection rows path mismatch")
    _verify_file_binding(output_binding, "sealed projection rows")
    sources = projection["sources"]
    legacy_path = _verify_file_binding(sources["legacy_exposed_packet"], "legacy exposed packet")
    fresh_path = _verify_file_binding(sources["fresh_baseline_consumer"], "fresh baseline consumer")
    baseline_packet_path = _verify_file_binding(sources["baseline_packet"], "baseline continuation packet")
    baseline_result_path = _verify_file_binding(sources["baseline_result"], "baseline continuation result")
    current_packet_path = _verify_file_binding(sources["current_evaluation_packet"], "current evaluation packet")
    rows = read_jsonl(rows_path)
    descriptors = projection.get("rows", [])
    require(len(descriptors) == len(rows) == COMBINED_PANEL_SIZE, "sealed projection denominator")
    legacy_packet = read(legacy_path)
    fresh_rows = read(fresh_path)
    baseline_packet = read(baseline_packet_path)
    baseline_result = read(baseline_result_path)
    current_packet = read(current_packet_path)
    current_binding = binding(current_packet_path)
    require(baseline_packet.get("input_packet") == current_binding, "projection baseline/current binding")
    require(baseline_result.get("evaluation_packet") == current_binding, "projection result/current binding")
    require([int(row["image_id"]) for row in fresh_rows] == [int(row["image_id"]) for row in current_packet["records"]], "projection fresh identity")
    old_by_id = {int(record["image_id"]): record for record in legacy_packet["eval_records"]}
    fresh_by_id = {int(record["image_id"]): record for record in fresh_rows}
    require(len(old_by_id) == LEGACY_EXPOSED_SIZE and len(fresh_by_id) == PANEL_SIZE, "projection source identity")
    expected_ids = [int(item["image_id"]) for item in descriptors]
    actual_ids = [int(row["image_id"]) for row in rows]
    require(actual_ids == expected_ids, "sealed projection dropped or reordered row")
    seen: set[int] = set()
    old_adapter = legacy_packet["model"]["current_adapter"]
    legacy_sha = sources["legacy_exposed_packet"]["sha256"]
    fresh_sha = sources["fresh_baseline_consumer"]["sha256"]
    for descriptor, actual in zip(descriptors, rows, strict=True):
        image_id = int(descriptor["image_id"])
        require(image_id not in seen, "sealed projection duplicate image")
        seen.add(image_id)
        kind = descriptor["source_kind"]
        if kind == "legacy_exposed_eval_record":
            source = old_by_id.get(image_id)
            require(source is not None, "legacy projection source row missing")
            expected = _annotate_projection(
                _project_legacy_stable_row(
                    source,
                    source_packet_sha256=legacy_sha,
                    source_adapter_fingerprint=old_adapter["fingerprint"],
                ),
                source_record=source,
                source_kind=kind,
                source_file_sha256=legacy_sha,
                source_adapter_fingerprint=old_adapter["fingerprint"],
            )
        elif kind == "fresh256_baseline_consumer_array":
            source = fresh_by_id.get(image_id)
            require(source is not None, "fresh projection source row missing")
            expected = _annotate_projection(
                source,
                source_record=source,
                source_kind=kind,
                source_file_sha256=fresh_sha,
                source_adapter_fingerprint=str(source["adapter_fingerprint"]),
            )
        else:
            raise ValueError(f"unsupported projection source kind: {kind}")
        require(str(actual["example_id"]) == str(expected["example_id"]), "projection example identity")
        require(actual.get("source_record_sha256") == descriptor["source_record_sha256"], "projection source row hash")
        require(digest(_sealed_fields(actual)) == descriptor["exact_fields_sha256"], "projection exact-field hash")
        require(_sealed_fields(actual) == _sealed_fields(expected), "projection lossless fields changed")
        require(digest(actual) == descriptor["projected_row_sha256"], "projection row changed")
    require(len(seen) == COMBINED_PANEL_SIZE, "projection union denominator")
    return rows


def _validate_candidate_panel_packet(packet: Mapping[str, Any]) -> None:
    require(packet.get("schema") in {
        "native_owner_scale_state.evaluation.candidate_panel.v2",
        "native_owner_scale_state.evaluation.candidate_panel.v3",
    }, "candidate panel schema")
    require(packet.get("status") in {"candidate_adapter_pending", "ready_for_root_launch_gate"}, "candidate panel status")
    records = packet.get("records", [])
    require(len(records) == COMBINED_PANEL_SIZE, "candidate panel record denominator")
    strata = packet.get("strata", {})
    require(strata.get("train11", {}).get("count") == ADMITTED_IMAGE_COUNT, "candidate train11 denominator")
    require(strata.get("reference54", {}).get("count") == REFERENCE_IMAGE_COUNT, "candidate reference54 denominator")
    require(strata.get("remaining_retention", {}).get("count") == COMBINED_PANEL_SIZE - ADMITTED_IMAGE_COUNT - REFERENCE_IMAGE_COUNT, "candidate remaining denominator")
    ids = [int(record["image_id"]) for record in records]
    require(len(ids) == len(set(ids)) == COMBINED_PANEL_SIZE, "candidate panel image identity")
    panel_ids = set(ids)
    train_ids = {int(value) for value in strata["train11"]["image_ids"]}
    reference_ids = {int(value) for value in strata["reference54"]["image_ids"]}
    remaining_ids = {int(value) for value in strata["remaining_retention"]["image_ids"]}
    legacy_ids = {int(value) for value in strata["legacy_retention"]["image_ids"]}
    fresh_ids = {int(value) for value in strata["fresh256"]["image_ids"]}
    require(len(train_ids) == ADMITTED_IMAGE_COUNT and len(reference_ids) == REFERENCE_IMAGE_COUNT, "candidate named stratum identity")
    require(not train_ids & reference_ids and not train_ids & remaining_ids and not reference_ids & remaining_ids, "candidate strata overlap")
    require(train_ids | reference_ids | remaining_ids == panel_ids, "candidate strata union")
    require(len(legacy_ids) == LEGACY_EXPOSED_SIZE - ADMITTED_IMAGE_COUNT - REFERENCE_IMAGE_COUNT and len(fresh_ids) == PANEL_SIZE, "candidate source stratum counts")
    require(not legacy_ids & fresh_ids and legacy_ids | fresh_ids == remaining_ids, "candidate retention strata union")
    require(packet.get("generation", {}).get("max_new_tokens") == CAP, "candidate panel cap")
    require(packet.get("generation", {}).get("natural_prefix_ids") == [], "candidate panel natural prefix")
    require(packet.get("generation", {}).get("dtype") == "fp32" and packet.get("generation", {}).get("attention") == "sdpa", "candidate panel numerics")
    if packet.get("schema") == "native_owner_scale_state.evaluation.candidate_panel.v3":
        ledger = packet.get("trusted_target_ledger")
        require(
            isinstance(ledger, Mapping)
            and ledger.get("path")
            and ledger.get("sha256")
            and ledger.get("size_bytes") is not None,
            "candidate trusted-target ledger binding",
        )
        require(int(packet.get("trusted_target_count", 0)) == TRUSTED_TARGET_COUNT, "candidate trusted-target denominator")


def _validate_trusted_target_ledger(
    ledger: Mapping[str, Any], packet: Mapping[str, Any]
) -> None:
    """Validate the frozen reviewed-target source boundary before consuming rows."""

    require(ledger.get("schema") == "native_owner_scale_state.evaluation.trusted_target_ledger.v1", "trusted-target ledger schema")
    require(ledger.get("status") == "frozen_reviewed_c_targets_no_model_calls", "trusted-target ledger status")
    require(int(ledger.get("target_count", 0)) == TRUSTED_TARGET_COUNT, "trusted-target ledger denominator")
    targets = ledger.get("targets", [])
    require(isinstance(targets, list) and len(targets) == TRUSTED_TARGET_COUNT, "trusted-target ledger targets")
    policy = ledger.get("matching_policy", {})
    require(
        policy.get("thresholds") == [50, 60, 80]
        and policy.get("joint_assignment_per_image") is True
        and policy.get("one_prediction_cannot_recover_two_targets") is True
        and policy.get("forced_credit") is False
        and policy.get("gt_independent") is True,
        "trusted-target matching policy",
    )
    sources = ledger.get("source_bindings", {})
    require(isinstance(sources, Mapping), "trusted-target source bindings")
    packet_training_binding = packet.get("training_input_identity_only", {}).get("inputs_v2")
    require(sources.get("training_input") == packet_training_binding, "trusted-target training input binding")
    for name in ("training_input", "acquisition_result", "final_review"):
        _verify_file_binding(sources.get(name, {}), f"trusted-target {name}")
    train_ids = {int(value) for value in packet["strata"]["train11"]["image_ids"]}
    require(len(train_ids) == ADMITTED_IMAGE_COUNT, "trusted-target train image denominator")
    seen_records: set[str] = set()
    seen_owners: set[str] = set()
    target_images: set[int] = set()
    for target in targets:
        require(isinstance(target, Mapping), "trusted-target package shape")
        record_id = str(target.get("record_id", ""))
        owner_id = str(target.get("owner_id", ""))
        image_id = int(target.get("image_id"))
        require(record_id and record_id not in seen_records, "trusted-target package identity")
        require(owner_id and owner_id not in seen_owners, "trusted-target owner identity")
        require(image_id in train_ids, "trusted-target outside train11")
        seen_records.add(record_id)
        seen_owners.add(owner_id)
        target_images.add(image_id)
        scoring_object = target.get("target", {}).get("scoring_object")
        require(isinstance(scoring_object, Mapping), "trusted-target scoring object")
        require(scoring_object.get("object_id") == owner_id, "trusted-target scoring owner")
        require(isinstance(scoring_object.get("description"), str) and scoring_object["description"], "trusted-target scoring class")
        bbox = scoring_object.get("bbox")
        require(isinstance(bbox, list) and len(bbox) == 4, "trusted-target scoring geometry")
        # The ledger is deliberately not a GT field.  This catches an accidental
        # fallback to the endpoint's ordinary annotation path at the boundary.
        require("gt" not in target and "gt" not in target.get("target", {}), "trusted-target GT contamination")
    require(len(seen_records) == TRUSTED_TARGET_COUNT and len(seen_owners) == TRUSTED_TARGET_COUNT, "trusted-target identity denominator")
    require(target_images == train_ids, "trusted-target image union")


def _write_blind_freeze(
    *, output: Path, selection_path: Path, fresh_records: Sequence[Mapping[str, Any]], artifact_tag: str = "v2"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the fixed32 visual identities and a concealed literal source map."""

    selection = read(selection_path)
    blind_ids = [int(value) for value in selection["blind_review_ids"]]
    require(len(blind_ids) == BLIND_SIZE and len(set(blind_ids)) == BLIND_SIZE, "blind32 identity")
    by_id = {int(record["image_id"]): record for record in fresh_records}
    require(set(blind_ids) <= set(by_id), "blind32 outside fresh panel")
    map_rows = [
        {
            "review_id": f"native-owner-scale-blind:{image_id}",
            "image_id": image_id,
            "example_id": str(by_id[image_id]["example_id"]),
            "literal_source_canvas_path": by_id[image_id]["case"]["image_path"],
            "image_width": by_id[image_id]["case"]["image_width"],
            "image_height": by_id[image_id]["case"]["image_height"],
        }
        for image_id in blind_ids
    ]
    source_map_path = output / f"candidate-blind-review-source-map-{artifact_tag}.json"
    publish(
        source_map_path,
        {
            "schema": "native_owner_scale_state.evaluation.blind_source_map.v2",
            "status": "hidden_source_map_frozen_before_candidate_outputs",
            "rows": map_rows,
            "boundary": "literal source paths only; no arm labels, prediction fields, GT labels or auto-negative labels",
        },
    )
    freeze = {
        "schema": "native_owner_scale_state.evaluation.blind_review_freeze.v2",
        "status": "frozen_before_candidate_outputs",
        "review_mode": "source_blind_mixed_union",
        "image_ids": blind_ids,
        "image_ids_sha256": digest(blind_ids),
        "selection": binding(selection_path),
        "source_map": binding(source_map_path),
        "scope": "fixed32 fresh identities; later Stable50/scaled-terminal proposals are merged with concealed source labels",
        "claim_boundary": "reviewed proposal coverage is not exhaustive scene recall or true recall; unmatched/unclear rows remain neutral and GT is untouched",
    }
    freeze_path = output / f"candidate-blind-review-freeze-{artifact_tag}.json"
    publish(freeze_path, freeze)
    return freeze, {"path": str(source_map_path), "binding": binding(source_map_path)}


def prepare_candidate_panel(
    *,
    output: str | Path = ROOT,
    current_packet_path: str | Path = CURRENT_EVALUATION_PACKET,
    legacy_packet_path: str | Path = LEGACY_EXPOSED_PACKET,
    baseline_packet_path: str | Path = BASELINE_CONTINUATION_PACKET,
    baseline_consumer_path: str | Path = BASELINE_CONTINUATION_CONSUMER,
    baseline_result_path: str | Path = BASELINE_CONTINUATION_RESULT,
    training_input_path: str | Path = TRAINING_INPUT_V2,
    snapshot_path: str | Path = EVALUATION_SNAPSHOT,
    artifact_tag: str = "v2",
) -> dict[str, Any]:
    """Prepare the 640-image candidate consumer contract without model calls."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current_packet_path = Path(current_packet_path)
    legacy_packet_path = Path(legacy_packet_path)
    baseline_packet_path = Path(baseline_packet_path)
    baseline_consumer_path = Path(baseline_consumer_path)
    baseline_result_path = Path(baseline_result_path)
    training_input_path = Path(training_input_path)
    snapshot_path = Path(snapshot_path)
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", artifact_tag) is not None, "candidate artifact tag")
    packet_path = output / f"candidate-panel-{artifact_tag}.json"
    projection_path = output / f"candidate-stable-projection-{artifact_tag}.json"
    rows_path = output / f"candidate-stable-rows-{artifact_tag}.jsonl"
    readiness_path = output / f"candidate-continuation-readiness-{artifact_tag}.json"
    freeze_path = output / f"candidate-blind-review-freeze-{artifact_tag}.json"
    source_map_path = output / f"candidate-blind-review-source-map-{artifact_tag}.json"
    ledger_path = output / f"candidate-trusted-target-ledger-{artifact_tag}.json"
    for path in (packet_path, projection_path, rows_path, readiness_path, freeze_path, source_map_path, ledger_path):
        require(not path.exists(), f"candidate preparation artifact already occupied: {path}")
    current_packet = read(current_packet_path)
    require(current_packet.get("status") == "cpu_prepared_candidate_pending", "current evaluation packet is not pending")
    require(len(current_packet.get("records", [])) == PANEL_SIZE, "fresh current packet denominator")
    selection_path = _verify_file_binding(current_packet["selection"], "fresh selection")
    selection = read(selection_path)
    validate_selection(selection, [int(row["image_id"]) for row in read_jsonl(SOURCE)])
    fresh_records = current_packet["records"]
    fresh_ids = [int(record["image_id"]) for record in fresh_records]
    require(fresh_ids == [int(value) for value in selection["image_ids"]], "fresh packet/selection order")
    legacy_packet = read(legacy_packet_path)
    legacy_records = legacy_packet.get("eval_records", [])
    require(len(legacy_records) == LEGACY_EXPOSED_SIZE, "legacy exposed packet denominator")
    legacy_ids = [int(record["image_id"]) for record in legacy_records]
    require(len(set(legacy_ids)) == LEGACY_EXPOSED_SIZE and not set(legacy_ids) & set(fresh_ids), "legacy/fresh panel overlap")
    training_input = read(training_input_path)
    admitted_ids, reference_ids = _training_strata_ids(training_input, legacy_records)
    raw_source_bindings: list[dict[str, Any]] = []
    for source in training_input.get("materialization_raw_sources", []):
        current_source = binding(source["path"])
        require(current_source["sha256"] == source["sha256"], "training raw materialization source changed")
        raw_source_bindings.append(current_source)
    require(raw_source_bindings, "training raw materialization provenance missing")
    trusted_target_ledger = _build_trusted_target_ledger(
        output_path=ledger_path,
        training_input_path=training_input_path,
    )
    records = [*legacy_records, *fresh_records]
    strata = _candidate_panel_strata(records, admitted_ids=admitted_ids, reference_ids=reference_ids, fresh_ids=fresh_ids)
    rows, descriptors, projection = _build_sealed_stable_projection(
        legacy_packet_path=legacy_packet_path,
        baseline_consumer_path=baseline_consumer_path,
        baseline_packet_path=baseline_packet_path,
        baseline_result_path=baseline_result_path,
        current_packet_path=current_packet_path,
    )
    _write_jsonl_exclusive(rows_path, rows)
    projection["output_rows"] = binding(rows_path)
    publish(projection_path, projection)
    freeze, source_map = _write_blind_freeze(output=output, selection_path=selection_path, fresh_records=fresh_records, artifact_tag=artifact_tag)
    descriptor_by_id = {int(item["image_id"]): item for item in descriptors}
    strata_by_id = {
        int(image_id): info["role"]
        for info in strata.values()
        for image_id in info["image_ids"]
        if info["role"] != "remaining_retention"
    }
    tagged_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_id = int(record["image_id"])
        tagged = copy.deepcopy(record)
        tagged["source_record_sha256"] = digest(record)
        tagged["evaluation_stratum"] = strata_by_id[image_id]
        tagged["source_panel"] = "legacy_exposed384" if index < LEGACY_EXPOSED_SIZE else "fresh256"
        tagged_records.append(tagged)
    old_adapter = legacy_packet["model"]["current_adapter"]
    bridged_adapter = _adapter_identity_bridge(old_adapter, legacy_packet["model"]["base_model_path"])
    baseline_packet = read(baseline_packet_path)
    baseline_result = read(baseline_result_path)
    previous_readiness = output / "candidate-continuation-readiness.json"
    # The old v1 readiness is kept immutable and is mentioned only as a
    # superseded record; the authoritative dependency below is inputs-v2.
    supersedes = binding(previous_readiness) if previous_readiness.exists() else None
    packet: dict[str, Any] = {
        "schema": "native_owner_scale_state.evaluation.candidate_panel.v3",
        "status": "candidate_adapter_pending",
        "records": tagged_records,
        "model": copy.deepcopy(current_packet["model"]),
        "config": copy.deepcopy(current_packet["config"]),
        "generation": copy.deepcopy(current_packet["generation"]),
        "selection": binding(selection_path),
        "strata": strata,
        "denominators": {
            "combined640": COMBINED_PANEL_SIZE,
            "train11": ADMITTED_IMAGE_COUNT,
            "reference54": REFERENCE_IMAGE_COUNT,
            "legacy_retention": LEGACY_EXPOSED_SIZE - ADMITTED_IMAGE_COUNT - REFERENCE_IMAGE_COUNT,
            "fresh256": PANEL_SIZE,
            "remaining_retention": COMBINED_PANEL_SIZE - ADMITTED_IMAGE_COUNT - REFERENCE_IMAGE_COUNT,
            "blind_review32": BLIND_SIZE,
            "trusted_target_packages": TRUSTED_TARGET_COUNT,
        },
        "stable50": {
            "endpoint_role": "sealed unchanged Stable50 baseline; legacy384 embedded traces plus fresh256 baseline consumer",
            "fresh_endpoint_adapter": copy.deepcopy(current_packet["stable50"]["adapter"]),
            "legacy_stored_adapter": copy.deepcopy(old_adapter),
            "legacy_rename_only_bridge": bridged_adapter,
            "legacy_receipt_boundary": "legacy384 traces predate live-object receipt serialization and are not relabeled as live receipts",
        },
        "scaled_terminal": {
            "status": "pending_A_terminal_adapter",
            "endpoint_role": "one unchanged natural scaled terminal adapter only after A supplies N>=16 and a cold receipt",
            "adapter_bound": False,
        },
        "stable_projection": {
            "rows": binding(rows_path),
            "sidecar": binding(projection_path),
            "source_rows": descriptors,
            "lossless_fields": list(_SEALED_STABLE_FIELDS),
        },
        "trusted_target_ledger": binding(ledger_path),
        "trusted_target_count": TRUSTED_TARGET_COUNT,
        "trusted_target_policy": trusted_target_ledger["matching_policy"],
        "blind_review": {
            "freeze": binding(freeze_path),
            "source_map": source_map["binding"],
            "review_mode": "source_blind_mixed_union",
            "no_exhaustive_recall": True,
            "no_auto_gt_negative": True,
        },
        "baseline": {
            "packet": binding(baseline_packet_path),
            "consumer": binding(baseline_consumer_path),
            "result": binding(baseline_result_path),
            "rows_reused_without_rerun": True,
            "metrics_not_copied_to_training": True,
        },
        "training_input_identity_only": {
            "inputs_v2": binding(training_input_path),
            "schema": training_input["schema"],
            "positive_records": len(training_input["positive_records"]),
            "admitted_image_ids": sorted(admitted_ids),
            "admitted_images": len(admitted_ids),
            "normal_reference_records": len(training_input["normal_keys"]),
            "materialization_raw_sources": raw_source_bindings,
            "do_not_use_baseline_metrics": True,
            "not_candidate_source_loader": True,
        },
        "candidate_contract": {
            "label": "scaled_terminal",
            "required_admitted_packages": 16,
            "natural_prefix_ids": [],
            "max_new_tokens": CAP,
            "dtype": "fp32",
            "attention": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "forced_credit": False,
            "no_candidate_hash_until_A_cold_receipt": True,
        },
        "consumer": {
            "module": "probes.native_owner_scale.evaluation",
            "bind_command": f"python -m probes.native_owner_scale.evaluation bind-candidate-panel --packet <{packet_path.name}> --candidate <A-terminal-receipt.json> --output-packet <candidate-panel-bound-{artifact_tag}.json>",
            "consume_command": f"python -m probes.native_owner_scale.evaluation consume-candidate-panel --packet <candidate-panel-bound-{artifact_tag}.json> --candidate-rows <scaled-terminal-640.jsonl> --output <candidate-panel-consumer-{artifact_tag}>",
            "metric_path": "accepted native_record -> candidate_opportunity.score -> margin_preserved_endpoint.burden/owner_changes",
            "visualization_path": "consumer output visualization-manifest.json with literal source-canvas image paths",
        },
        "producer": binding(Path(__file__).resolve()),
        "producer_snapshot": binding(snapshot_path),
        "source_packets": {
            "legacy_exposed384": binding(legacy_packet_path),
            "current_fresh256": binding(current_packet_path),
        },
        "provenance_boundary": "640 unique images are identity-bound to exposed384 plus the frozen fresh256 panel; no pretraining/SFT-disjointness claim. Stable50 baseline metrics are sealed consumer evidence and are not fed into A.",
        "claim_boundary": "No candidate output, transfer result, reviewed coverage, GT-negative label or hallucination claim exists until A supplies one bound cold endpoint and its natural rows; unclear review rows remain neutral.",
    }
    if supersedes is not None:
        packet["supersedes"] = supersedes
    previous_panels = sorted(
        path for path in output.glob("candidate-panel-*.json") if path != packet_path
    )
    if previous_panels:
        packet["supersedes_panel"] = binding(previous_panels[-1])
    _validate_candidate_panel_packet(packet)
    publish(packet_path, packet)
    readiness = {
        "schema": "native_owner_scale_state.evaluation.candidate_continuation_readiness.v3",
        "status": "candidate_adapter_pending",
        "panel_packet": binding(packet_path),
        "selection": binding(selection_path),
        "stable_projection": binding(projection_path),
        "stable_rows": binding(rows_path),
        "trusted_target_ledger": binding(ledger_path),
        "blind_freeze": binding(freeze_path),
        "denominators": packet["denominators"],
        "training_input": binding(training_input_path),
        "training_input_status": training_input["status"],
        "model_calls_performed_by_prepare": False,
        "sealed_baseline_reused_without_rerun": True,
        "exact_gap": "A terminal adapter cold receipt and future natural scaled-terminal rows are input-dependent; no candidate hash or comparative claim is fabricated while they are absent.",
        "stop_rule": "Stop CPU preparation here and resume only after A supplies one unmerged cold endpoint receipt; otherwise report adapter shortfall without a transfer claim.",
        "claim_boundary": packet["claim_boundary"],
    }
    publish(readiness_path, readiness)
    return packet


def bind_candidate_panel(
    *, packet_path: str | Path, candidate_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Bind A's one cold terminal adapter to the already-frozen 640 panel."""

    packet_path = Path(packet_path)
    candidate_path = Path(candidate_path)
    output_path = Path(output_path) if output_path is not None else packet_path.with_name("candidate-panel-bound-v2.json")
    packet = read(packet_path)
    _validate_candidate_panel_packet(packet)
    require(packet["status"] == "candidate_adapter_pending", "candidate panel already bound or not pending")
    candidate = read(candidate_path)
    validate_candidate_binding(candidate)
    if candidate.get("schema") == "native_owner_scale_state.evaluation.candidate_endpoint.v1":
        for name in ("training_receipt", "cold_check", "training_completion", "training_input"):
            _verify_file_binding(candidate[name], f"candidate {name}")
    bound = copy.deepcopy(packet)
    endpoint = copy.deepcopy(candidate)
    endpoint["receipt"] = binding(candidate_path)
    if "launch_contract" in endpoint:
        endpoint["launch_contract"] = _candidate_launch_contract(
            packet_path=output_path,
            output_root=endpoint["launch_contract"]["output_root"],
        )
        endpoint["resource_bounds"] = endpoint["launch_contract"]
    bound["scaled_terminal"] = endpoint
    bound["status"] = "ready_for_root_launch_gate"
    bound["base_packet"] = binding(packet_path)
    require(not output_path.exists(), "bound candidate panel already exists")
    bound["producer_parent"] = copy.deepcopy(packet.get("producer"))
    bound["producer"] = binding(Path(__file__).resolve())
    publish(output_path, bound)
    return bound


def _candidate_run_packet(packet: Mapping[str, Any]) -> None:
    """Check the bound candidate identity before any model load."""

    _validate_candidate_panel_packet(packet)
    require(packet.get("status") == "ready_for_root_launch_gate", "candidate panel is not bound")
    validate_candidate_binding(packet["scaled_terminal"])
    require(packet["scaled_terminal"].get("schema") == "native_owner_scale_state.evaluation.candidate_endpoint.v1", "candidate endpoint schema")
    require(packet["generation"].get("max_new_tokens") == CAP and packet["generation"].get("natural_prefix_ids") == [], "candidate generation identity")


def _candidate_materialized_case(case: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve only the ephemeral row against its frozen source, not the fresh JSONL."""

    path = Path(case["image_path"])
    require(path.is_absolute() and path.is_file(), "candidate bound image missing")
    require(file_hash(path) == case["image_plan"]["image_content_sha256"], "candidate bound image bytes changed")
    require(len(case["input_record"]["images"]) == 1, "candidate requires one frozen image")
    materialized = copy.deepcopy(case)
    # The raw-row contract requires a relative reference; derive it from the
    # bound absolute path rather than reinterpreting a legacy relative string.
    materialized["input_record"]["images"] = [os.path.relpath(path, Path(config["data"]["input_jsonl"]).parent)]
    return materialized


def candidate_worker(
    *, packet_path: str | Path, shard: int, physical_gpu: int, output: str | Path
) -> None:
    """Produce one bounded shard of natural scaled-terminal rows."""

    import torch

    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.eval.native_rows import native_detection_record as native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    packet_path = Path(packet_path)
    packet = read(packet_path)
    _candidate_run_packet(packet)
    require(shard in range(len(CANDIDATE_GPUS)), "candidate shard")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "candidate worker physical GPU")
    require(torch.cuda.device_count() == 1, "candidate worker requires one visible GPU")
    require(int(CANDIDATE_GPUS[shard]) == int(physical_gpu), "candidate shard/GPU identity")
    output = Path(output)
    require(not output.exists(), "candidate worker output occupied")
    output.mkdir(parents=True)
    records = list(packet["records"])
    shard_records = records[shard::len(CANDIDATE_GPUS)]
    packet_file_sha256 = file_hash(packet_path)
    packet_digest = digest(packet)
    adapter = packet["scaled_terminal"]["adapter"]
    terminal: dict[str, Any] = {
        "schema": "native_owner_scale_state.evaluation.candidate_terminal.v1",
        "status": "running",
        "arm": "scaled_terminal",
        "shard": shard,
        "physical_gpu": physical_gpu,
        "packet_sha256": packet_file_sha256,
        "packet_digest": packet_digest,
        "expected_continuations": len(shard_records),
        "continuations": 0,
        "new_tokens": 0,
        "model_forwards": 0,
        "image_forwards": 0,
        "model_loads": 0,
    }
    publish(output / "launch.json", terminal)
    handles: list[Any] = []
    started = time.monotonic()
    try:
        config = checkpoint_config(
            InferConfig.model_validate(packet["config"]), adapter["root"]
        )
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live_adapter = identity["model_identity"]["adapter"]
        require(
            live_adapter["adapter_path"] == adapter["root"]
            and live_adapter.get("merged_adapters", []) == [],
            "loaded candidate adapter identity changed",
        )
        require(
            identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
            and identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
            "live candidate FP32/SDPA identity changed",
        )
        publish(
            output / "model.json",
            {
                "schema": "native_owner_scale_state.evaluation.candidate_model_receipt.v1",
                "status": "live_model_loaded_before_inference",
                "packet": binding(packet_path),
                "shard": shard,
                "physical_gpu": physical_gpu,
                "identity": identity,
                "adapter": copy.deepcopy(adapter),
                "generation": copy.deepcopy(packet["generation"]),
            },
        )
        publish(
            output / "source.json",
            {
                "schema": "native_owner_scale_state.evaluation.candidate_source_receipt.v1",
                "status": "frozen_input_bound_before_inference",
                "packet": binding(packet_path),
                "selection": packet["selection"],
                "shard": shard,
                "image_ids": [int(record["image_id"]) for record in shard_records],
                "records": [
                    {
                        "image_id": int(record["image_id"]),
                        "example_id": record["example_id"],
                        "image_path": record["case"]["image_path"],
                        "prompt_token_ids_sha256": digest(record["prompt_token_ids"]),
                        "executed_media_sha256": record["case"]["image_plan"]["executed_media_sha256"],
                        "observed_image_grid_thw": record["case"]["image_plan"]["observed_image_grid_thw"],
                        "source_panel": record.get("source_panel"),
                        "evaluation_stratum": record.get("evaluation_stratum"),
                    }
                    for record in shard_records
                ],
                "native_source": packet.get("source_packets"),
            },
        )
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "candidate visual hook ambiguity")
        handles.append(visual[0].register_forward_pre_hook(lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        policy = NativeGenerationPolicy(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_model_defaults=False,
        )
        rows_path = output / "rows.jsonl"
        with rows_path.open("x") as stream:
            for frozen in shard_records:
                requests, _ = build_requests(qwen, packet["config"], [_candidate_materialized_case(frozen["case"], packet["config"])])
                batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0", record_media_identity=True)
                plan = frozen["case"]["image_plan"]
                require(
                    list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"]
                    and batch.media_sha256[0] == plan["executed_media_sha256"]
                    and list(batch.image_grids[0]) == plan["observed_image_grid_thw"],
                    "frozen candidate natural input identity changed",
                )
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        batch,
                        extensions=[[]],
                        budgets=[CAP],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                ids = list(generated.token_ids)
                stop = generated.stop_reason
                require(generated.request_id == frozen["example_id"], "candidate request identity")
                accepted._checked_action(ids, stop, CAP)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], stop)
                observed = {
                    "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                    "executed_media_sha256": batch.media_sha256[0],
                    "observed_image_grid_thw": list(batch.image_grids[0]),
                }
                row = {
                    "schema": "native_owner_scale_state.evaluation.candidate_natural.v1",
                    "arm": "scaled_terminal",
                    "shard": shard,
                    "example_id": frozen["example_id"],
                    "image_id": frozen["image_id"],
                    "split": frozen["split"],
                    "request_id": generated.request_id,
                    "action_ids": ids,
                    "prefix_ids": [],
                    "forced_ids": [],
                    "remaining_budget": CAP,
                    "text": text,
                    "stop_reason": stop,
                    "parsed": parsed,
                    "score": score(parsed, seed=None, length=len(ids), stop=stop),
                    "overlap_counts": overlap_counts(parsed),
                    "batch_identity_sha256": digest(observed),
                    "packet_sha256": packet_digest,
                    "candidate_panel_packet_sha256": packet_file_sha256,
                    "adapter_fingerprint": adapter["fingerprint"],
                    **observed,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(ids)
                del batch
        require(terminal["continuations"] == len(shard_records), "candidate shard denominator")
        require(terminal["new_tokens"] <= len(shard_records) * CAP, "candidate token cap")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        publish(output / "terminal.json", terminal)


def candidate_launch(*, packet_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Launch exactly eight one-GPU candidate workers; never retries."""

    packet_path = Path(packet_path)
    packet = read(packet_path)
    _candidate_run_packet(packet)
    output = Path(output)
    require(not output.exists(), "candidate launch output occupied; no automatic relaunch")
    output.mkdir(parents=True)
    contract = packet["scaled_terminal"].get("launch_contract", {})
    gpus = [int(value) for value in contract.get("physical_gpus", CANDIDATE_GPUS)]
    require(gpus == list(CANDIDATE_GPUS), "candidate GPU allocation changed")
    commands = []
    processes = []
    logs = []
    for shard, physical_gpu in enumerate(gpus):
        command = [
            sys.executable,
            "-m",
            "probes.native_owner_scale.evaluation",
            "candidate-worker",
            "--packet",
            str(packet_path),
            "--shard",
            str(shard),
            "--physical-gpu",
            str(physical_gpu),
            "--output",
            str(output / f"shard-{shard}"),
        ]
        commands.append(command)
        log = (output / f"shard-{shard}.log").open("x")
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        processes.append((shard, subprocess.Popen(command, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)))
    publish(
        output / "launch.json",
        {
            "schema": "native_owner_scale_state.evaluation.candidate_launch.v1",
            "status": "running",
            "packet": binding(packet_path),
            "image_count": COMBINED_PANEL_SIZE,
            "physical_gpus": gpus,
            "commands": commands,
            "no_retry": True,
            "resource_bounds": contract,
        },
    )
    exits = []
    for (shard, process), log in zip(processes, logs, strict=True):
        exits.append({"shard": shard, "exit_code": process.wait()})
        log.close()
    publish(output / "outer-exits.json", exits)
    require(all(value["exit_code"] == 0 for value in exits), "candidate producer failure; preserve partial outputs")
    return {"status": "completed", "images": COMBINED_PANEL_SIZE, "output": str(output)}


def candidate_merge(*, packet_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Merge candidate shards losslessly into packet order before cold consumption."""

    packet_path = Path(packet_path)
    packet = read(packet_path)
    _candidate_run_packet(packet)
    output = Path(output)
    exits = read(output / "outer-exits.json")
    require(len(exits) == len(CANDIDATE_GPUS) and all(value["exit_code"] == 0 for value in exits), "candidate outer exits")
    packet_digest = digest(packet)
    rows: list[dict[str, Any]] = []
    terminals = []
    for shard in range(len(CANDIDATE_GPUS)):
        shard_root = output / f"shard-{shard}"
        terminal = read(shard_root / "terminal.json")
        require(terminal.get("status") == "completed" and terminal.get("exit_code") == 0, "candidate shard terminal")
        expected = len(packet["records"][shard::len(CANDIDATE_GPUS)])
        require(terminal.get("continuations") == expected, "candidate shard continuation count")
        terminals.append(binding(shard_root / "terminal.json"))
        shard_rows = read_jsonl(shard_root / "rows.jsonl")
        require(len(shard_rows) == expected, "candidate shard row count")
        for row in shard_rows:
            require(row.get("arm") == "scaled_terminal" and row.get("packet_sha256") == packet_digest, "candidate row packet identity")
        rows.extend(shard_rows)
    expected_ids = [str(record["example_id"]) for record in packet["records"]]
    by_id = {str(row["example_id"]): row for row in rows}
    require(len(by_id) == len(rows) == COMBINED_PANEL_SIZE and set(by_id) == set(expected_ids), "candidate merged identity")
    ordered = [by_id[example_id] for example_id in expected_ids]
    rows_path = output / "rows.jsonl"
    require(not rows_path.exists(), "candidate merged rows occupied")
    with rows_path.open("x") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "schema": "native_owner_scale_state.evaluation.candidate_merge.v1",
        "status": "candidate_raw_merged_pending_consumer",
        "packet": binding(packet_path),
        "rows": binding(rows_path),
        "image_ids": [int(record["image_id"]) for record in packet["records"]],
        "terminals": terminals,
        "consumer_command": f"python -m probes.native_owner_scale.evaluation consume-candidate-panel --packet {packet_path} --candidate-rows {rows_path} --output {output.with_name(output.name + '-consumer')}",
        "claim_boundary": "Natural candidate rows only; no owner gain/loss or fit-to-recovery claim until the root-owned consumer passes.",
    }
    publish(output / "merge.json", result)
    return result


def consume_candidate_panel(
    *, packet_path: str | Path, candidate_rows_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Cold-consume one scaled endpoint against the sealed 640 Stable50 rows."""

    from transformers import AutoTokenizer

    packet_path = Path(packet_path)
    packet = read(packet_path)
    _validate_candidate_panel_packet(packet)
    require(
        packet.get("schema") == "native_owner_scale_state.evaluation.candidate_panel.v3",
        "candidate trusted-target ledger required",
    )
    require(packet["status"] == "ready_for_root_launch_gate", "candidate panel is not bound")
    validate_candidate_binding(packet["scaled_terminal"])
    projection_binding = packet["stable_projection"]["sidecar"]
    projection_path = _verify_file_binding(projection_binding, "candidate stable projection")
    stable = _validate_sealed_baseline_projection(projection_path, packet["stable_projection"]["rows"]["path"])
    trusted_target_path = _verify_file_binding(packet["trusted_target_ledger"], "candidate trusted-target ledger")
    trusted_target_ledger = read(trusted_target_path)
    _validate_trusted_target_ledger(trusted_target_ledger, packet)
    selection_path = _verify_file_binding(packet["selection"], "candidate selection")
    selection = read(selection_path)
    validate_selection(selection, [int(row["image_id"]) for row in read_jsonl(SOURCE)])
    freeze_path = _verify_file_binding(packet["blind_review"]["freeze"], "blind review freeze")
    frozen = read(freeze_path)
    require(frozen["image_ids"] == selection["blind_review_ids"], "blind freeze/selection identity")
    tokenizer = AutoTokenizer.from_pretrained(packet["model"]["base_model_path"], local_files_only=True)
    candidate_input = read_jsonl(candidate_rows_path)
    candidate = validate_endpoint_rows(
        candidate_input,
        packet,
        arm="scaled_terminal",
        tokenizer=tokenizer,
        expected_image_ids=[int(record["image_id"]) for record in packet["records"]],
    )
    stable_ids = [str(row["example_id"]) for row in stable]
    candidate_by_id = {int(row["image_id"]): row for row in candidate}
    candidate = [candidate_by_id[int(row["image_id"])] for row in stable]
    require([str(row["example_id"]) for row in candidate] == stable_ids, "candidate/stable paired identity")
    output = Path(output)
    require(not output.exists(), "candidate panel consumer output occupied")
    output.mkdir(parents=True)
    stable_by_id = {int(row["image_id"]): row for row in stable}
    candidate_by_id = {int(row["image_id"]): row for row in candidate}
    stratum_results: dict[str, Any] = {}
    for stratum_name in ("train11", "reference54", "remaining_retention", "legacy_retention", "fresh256"):
        image_ids = [int(value) for value in packet["strata"][stratum_name]["image_ids"]]
        left = [stable_by_id[image_id] for image_id in image_ids]
        right = [candidate_by_id[image_id] for image_id in image_ids]
        stratum_results[stratum_name] = {
            "images": len(image_ids),
            "Stable50": {"quality": aggregate_scores([row["score"] for row in left]), "burden": accepted.burden(left)},
            "scaled_terminal": {"quality": aggregate_scores([row["score"] for row in right]), "burden": accepted.burden(right)},
            "paired": paired_summary(left, right),
        }
    trusted_target_recovery = _trusted_target_panel_summary(stable, candidate, trusted_target_ledger)
    stratum_results["train11"]["trusted_target_recovery"] = trusted_target_recovery
    review = _proposal_queue({"Stable50": stable, "scaled_terminal": candidate}, packet, output)
    result = {
        "schema": "native_owner_scale_state.evaluation.candidate_panel_result.v3",
        "status": "cold_verified",
        "packet": binding(packet_path),
        "rows": {
            "Stable50": packet["stable_projection"]["rows"],
            "scaled_terminal": binding(candidate_rows_path),
        },
        "arms": {
            "Stable50": {"images": len(stable), "quality": aggregate_scores([row["score"] for row in stable]), "burden": accepted.burden(stable)},
            "scaled_terminal": {"images": len(candidate), "quality": aggregate_scores([row["score"] for row in candidate]), "burden": accepted.burden(candidate)},
        },
        "scaled_terminal_vs_Stable50": paired_summary(stable, candidate),
        "strata": stratum_results,
        "denominators": packet["denominators"],
        "trusted_target_ledger": binding(trusted_target_path),
        "trusted_target_recovery": trusted_target_recovery,
        "blind_review": review,
        "claim_boundary": "Natural paired owner gains/losses and burden across the fixed 640-image panel; train11 is exposure-adjacent retention, reference54 is exposed retention, remaining_retention includes legacy319 plus fresh256. Reviewed c target recovery is a separate 16-package GT-independent ledger scored with the existing class/native-pixel global matcher; it is not exhaustive scene recall and does not convert unmatched/unclear rows into negative labels. blind32 is source-blind reviewed proposal coverage, not exhaustive recall. Original GT is untouched and no hallucination label is inferred from unmatched predictions.",
    }
    publish(output / "stable-consumer.json", stable)
    publish(output / "scaled-terminal-consumer.json", candidate)
    publish(output / "result.json", result)
    return result


def _validate_natural_row(
    row: Mapping[str, Any], frozen: Mapping[str, Any], tokenizer: Any, arm: str, packet_sha256: str,
    adapter_fingerprint: str,
) -> dict[str, Any]:
    from src.eval.native_rows import native_detection_record as native_record

    require(row.get("arm") == arm, "endpoint arm mismatch")
    require(str(row.get("example_id")) == str(frozen["example_id"]), "endpoint example identity")
    require(int(row["image_id"]) == int(frozen["image_id"]), "endpoint image identity")
    require(row.get("prefix_ids") == row.get("forced_ids") == [], "natural endpoint contains supplied history")
    require(int(row.get("remaining_budget")) == CAP, "natural endpoint allowance changed")
    require(row.get("adapter_fingerprint") == adapter_fingerprint, "endpoint adapter identity")
    if "packet_sha256" in row:
        require(row["packet_sha256"] == packet_sha256, "endpoint packet identity")
    require(row.get("request_id", frozen["example_id"]) == frozen["example_id"], "request identity")
    ids = list(row["action_ids"])
    accepted._checked_action(ids, row["stop_reason"], CAP)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    require(text == row["text"], "endpoint token/text mismatch")
    parsed = native_record(text, frozen["case"], frozen["golden"], row["stop_reason"])
    require(parsed == row["parsed"], "endpoint cold parser mismatch")
    expected_score = score(parsed, seed=None, length=len(ids), stop=row["stop_reason"])
    require(expected_score == row["score"], "endpoint cold scorer mismatch")
    if "overlap_counts" in row:
        require(row["overlap_counts"] == overlap_counts(parsed), "endpoint overlap accounting mismatch")
    plan = frozen["case"]["image_plan"]
    require(row.get("prompt_token_ids_sha256") == digest(frozen["prompt_token_ids"]), "prompt identity")
    require(row.get("executed_media_sha256") == plan["executed_media_sha256"], "media identity")
    require(row.get("observed_image_grid_thw") == plan["observed_image_grid_thw"], "image grid identity")
    return dict(row, parsed=parsed, score=expected_score)


def validate_endpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    packet: Mapping[str, Any],
    *,
    arm: str,
    tokenizer: Any,
    expected_image_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    records = packet["records"]
    if expected_image_ids is not None:
        expected_set = {int(value) for value in expected_image_ids}
        require(len(expected_set) == len(expected_image_ids), "duplicate expected endpoint IDs")
        records = [record for record in records if int(record["image_id"]) in expected_set]
        require(len(records) == len(expected_set), "slice endpoint IDs not in frozen packet")
    expected = {str(record["example_id"]): record for record in records}
    expected_count = len(expected)
    require(expected_count == PANEL_SIZE or expected_image_ids is not None, "frozen endpoint denominator")
    require(len(rows) == expected_count, f"{arm} endpoint row count")
    actual_ids = [str(row["example_id"]) for row in rows]
    require(len(set(actual_ids)) == expected_count and set(actual_ids) == set(expected), f"{arm} endpoint population")
    endpoint = packet["stable50"] if arm == "Stable50" else packet["scaled_terminal"]
    adapter = endpoint["adapter"]
    packet_sha = digest(packet)
    return [
        _validate_natural_row(row, expected[str(row["example_id"])], tokenizer, arm, packet_sha, adapter["fingerprint"])
        for row in rows
    ]


def paired_summary(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Image-paired gains/losses and deterministic percentile bootstrap."""

    import numpy as np

    require([r["example_id"] for r in before] == [r["example_id"] for r in after], "unpaired image order")
    pairs = []
    for left, right in zip(before, after, strict=True):
        pairs.append(
            {
                "image_id": left["image_id"],
                "example_id": left["example_id"],
                "owner_changes": accepted.owner_changes(left["score"], right["score"]),
                "delta": {
                    threshold: {
                        key: right["score"][threshold][key] - left["score"][threshold][key]
                        for key in ("tp", "fp", "fn", "f1")
                    }
                    for threshold in ("50", "60", "80")
                },
            }
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(pairs), size=(10000, len(pairs)))
    intervals: dict[str, Any] = {}
    for threshold in ("50", "60", "80"):
        left = np.array([[row["score"][threshold][key] for key in ("tp", "fp", "fn")] for row in before], dtype=float)
        right = np.array([[row["score"][threshold][key] for key in ("tp", "fp", "fn")] for row in after], dtype=float)
        left_sum, right_sum = left[index].sum(1), right[index].sum(1)

        def f1(value: Any) -> Any:
            denominator = 2 * value[:, 0] + value[:, 1] + value[:, 2]
            return np.divide(2 * value[:, 0], denominator, out=np.zeros(len(value)), where=denominator > 0)

        delta = right[:, 0] - left[:, 0]
        intervals[threshold] = {
            "delta_tp95": np.quantile(right_sum[:, 0] - left_sum[:, 0], [0.025, 0.975]).tolist(),
            "delta_micro_f1_95": np.quantile(f1(right_sum) - f1(left_sum), [0.025, 0.975]).tolist(),
            "positive_images": int((delta > 0).sum()),
            "negative_images": int((delta < 0).sum()),
            "tied_images": int((delta == 0).sum()),
        }
    return {
        "owner_counts": accepted.owner_change_counts(
            [row["score"] for row in before], [row["score"] for row in after]
        ),
        "paired_image_bootstrap": {"replicates": 10000, "seed": BOOTSTRAP_SEED, "intervals": intervals},
        "images": pairs,
    }


def _proposal_queue(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], packet: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    """Write source-blind mixed prediction proposals and a sealed source map."""

    records = {int(record["image_id"]): record for record in packet["records"]}
    blind_ids = [int(value) for value in read(Path(packet["selection"]["path"]))["blind_review_ids"]]
    by_image = {
        arm: {int(row["image_id"]): row for row in rows}
        for arm, rows in rows_by_arm.items()
    }
    queue_path = output / "blind-review-queue.jsonl"
    source_map: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for image_id in blind_ids:
        frozen = records[image_id]
        candidates: list[tuple[str, int, dict[str, Any]]] = []
        for arm, rows in by_image.items():
            parsed = rows[image_id]["parsed"]
            for index, prediction in enumerate(parsed["pred"]):
                # Keep only literal geometry/class fields in the review queue;
                # parser spans and raw text can disclose arm or chronology.
                proposal = {
                    "description": prediction["description"],
                    "bbox": list(prediction["bbox"]),
                    "bbox_format": prediction.get("bbox_format", "xyxy"),
                }
                candidates.append((arm, index, proposal))
        candidates.sort(key=lambda item: digest({"image_id": image_id, "proposal": item[2], "arm_slot": item[0]}))
        proposals = []
        for ordinal, (arm, source_index, proposal) in enumerate(candidates):
            proposal_id = hashlib.sha256(
                f"{BLIND_SALT}{image_id}:{ordinal}:{digest(proposal)}".encode()
            ).hexdigest()
            proposals.append({"proposal_id": proposal_id, **proposal})
            source_map.append(
                {
                    "image_id": image_id,
                    "proposal_id": proposal_id,
                    "source_arm": arm,
                    "source_prediction_index": source_index,
                }
            )
        queue_rows.append(
            {
                "schema": "native_owner_scale_state.blind_review_item.v1",
                "review_id": f"native-owner-scale-blind:{image_id}",
                "image_id": image_id,
                "image_path": frozen["case"]["image_path"],
                "image_width": frozen["case"]["image_width"],
                "image_height": frozen["case"]["image_height"],
                "proposals": proposals,
                "source_blind": True,
                "scope": "mixed arm proposals; reviewed proposal coverage, not exhaustive scene recall/true recall",
            }
        )
    require(len(queue_rows) == BLIND_SIZE, "blind32 review denominator")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queue_rows))
    source_map_path = output / "blind-review-source-map.json"
    publish(source_map_path, {"schema": "native_owner_scale_state.blind_review_source_map.v1", "rows": source_map})
    visualization = {
        "schema": "native_owner_scale_state.visualization_manifest.v1",
        "status": "literal_source_paths_bound",
        "queue": binding(queue_path),
        "source_map": binding(source_map_path),
        "items": [
            {
                "review_id": row["review_id"],
                "image_id": row["image_id"],
                "literal_source_canvas_path": row["image_path"],
                "proposal_count": len(row["proposals"]),
            }
            for row in queue_rows
        ],
        "renderer_entry": "probes.native_owner_scale.evaluation; preserve source canvas and proposal boxes; no GT/arm labels in reviewer view",
        "boundary": "The queue is a source-blind mixed-prediction review aid and cannot establish exhaustive recall.",
    }
    publish(output / "visualization-manifest.json", visualization)
    return {
        "queue": binding(queue_path),
        "source_map": binding(source_map_path),
        "visualization": binding(output / "visualization-manifest.json"),
        "images": BLIND_SIZE,
        "proposals": len(source_map),
    }


def consume_slice(
    *,
    packet_path: str | Path,
    stable_rows_path: str | Path,
    candidate_rows_path: str | Path,
    output: str | Path,
    image_ids: Sequence[int],
) -> dict[str, Any]:
    """Consume the two-image persisted slice required by the root launch gate."""

    from transformers import AutoTokenizer

    packet_path = Path(packet_path)
    packet = read(packet_path)
    require(packet["status"] == "ready_for_root_launch_gate", "candidate endpoint is not bound")
    validate_candidate_binding(packet["scaled_terminal"])
    image_ids = [int(value) for value in image_ids]
    require(len(image_ids) == 2 and len(set(image_ids)) == 2, "root slice must contain two identities")
    frozen_ids = {int(record["image_id"]) for record in packet["records"]}
    require(set(image_ids) <= frozen_ids, "root slice image outside frozen panel")
    tokenizer = AutoTokenizer.from_pretrained(packet["model"]["base_model_path"], local_files_only=True)
    stable = validate_endpoint_rows(
        read_jsonl(stable_rows_path), packet, arm="Stable50", tokenizer=tokenizer, expected_image_ids=image_ids
    )
    candidate = validate_endpoint_rows(
        read_jsonl(candidate_rows_path),
        packet,
        arm="scaled_terminal",
        tokenizer=tokenizer,
        expected_image_ids=image_ids,
    )
    require([row["example_id"] for row in stable] == [row["example_id"] for row in candidate], "slice order")
    output = Path(output)
    require(not output.exists(), "root slice output occupied")
    output.mkdir(parents=True)
    result = {
        "schema": "native_owner_scale_state.evaluation.slice_result.v1",
        "status": "cold_verified",
        "packet": binding(packet_path),
        "rows": {"Stable50": binding(stable_rows_path), "scaled_terminal": binding(candidate_rows_path)},
        "image_ids": image_ids,
        "natural_policy": packet["generation"],
        "Stable50": {"quality": aggregate_scores([row["score"] for row in stable]), "burden": accepted.burden(stable)},
        "scaled_terminal": {"quality": aggregate_scores([row["score"] for row in candidate]), "burden": accepted.burden(candidate)},
        "paired": paired_summary(stable, candidate),
        "next_step": "Root may expand the unchanged contract to the remaining 254 fresh IDs only after this persisted consumer passes.",
    }
    publish(output / "result.json", result)
    publish(output / "stable-consumer.json", stable)
    publish(output / "scaled-terminal-consumer.json", candidate)
    return result


def consume(
    *, packet_path: str | Path, stable_rows_path: str | Path, candidate_rows_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Cold-validate exact endpoints and produce paired/visual review artifacts."""

    from transformers import AutoTokenizer

    packet_path = Path(packet_path)
    packet = read(packet_path)
    require(packet["status"] == "ready_for_root_launch_gate", "candidate endpoint is not bound")
    validate_candidate_binding(packet["scaled_terminal"])
    selection = read(packet["selection"]["path"])
    source_rows = read_jsonl(SOURCE)
    validate_selection(selection, [int(row["image_id"]) for row in source_rows])
    tokenizer = AutoTokenizer.from_pretrained(packet["model"]["base_model_path"], local_files_only=True)
    stable = validate_endpoint_rows(read_jsonl(stable_rows_path), packet, arm="Stable50", tokenizer=tokenizer)
    candidate = validate_endpoint_rows(read_jsonl(candidate_rows_path), packet, arm="scaled_terminal", tokenizer=tokenizer)
    require([row["example_id"] for row in stable] == [row["example_id"] for row in candidate], "paired endpoint order")
    output = Path(output)
    require(not output.exists(), "evaluation consumer output occupied")
    output.mkdir(parents=True)
    result = {
        "schema": "native_owner_scale_state.evaluation.result.v1",
        "status": "cold_verified",
        "packet": binding(packet_path),
        "selection": binding(packet["selection"]["path"]),
        "arms": {
            "Stable50": {"images": len(stable), "quality": aggregate_scores([row["score"] for row in stable]), "burden": accepted.burden(stable)},
            "scaled_terminal": {"images": len(candidate), "quality": aggregate_scores([row["score"] for row in candidate]), "burden": accepted.burden(candidate)},
        },
        "scaled_terminal_vs_Stable50": paired_summary(stable, candidate),
        "denominators": {
            "fresh256": PANEL_SIZE,
            "blind_review32": BLIND_SIZE,
            "blind_review_is_subset_of_fresh256": True,
            "all_attempted_rows_retained": True,
        },
        "claim_boundary": "Natural paired owner gains/losses and burden on the frozen fresh256 panel; blind32 is source-blind reviewed proposal coverage, not exhaustive recall. No GT edit and no hallucination inference from unmatched predictions.",
    }
    publish(output / "stable-consumer.json", stable)
    publish(output / "scaled-terminal-consumer.json", candidate)
    review = _proposal_queue({"Stable50": stable, "scaled_terminal": candidate}, packet, output)
    result["blind_review"] = review
    publish(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--output", default=str(ROOT))
    select.add_argument("--lane-manifest", action="append", default=[], metavar="ROLE=PATH")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", default=str(ROOT))
    candidate_prepare = sub.add_parser("prepare-candidate-panel")
    candidate_prepare.add_argument("--output", default=str(ROOT))
    candidate_prepare.add_argument("--current-packet", default=str(CURRENT_EVALUATION_PACKET))
    candidate_prepare.add_argument("--legacy-packet", default=str(LEGACY_EXPOSED_PACKET))
    candidate_prepare.add_argument("--baseline-packet", default=str(BASELINE_CONTINUATION_PACKET))
    candidate_prepare.add_argument("--baseline-consumer", default=str(BASELINE_CONTINUATION_CONSUMER))
    candidate_prepare.add_argument("--baseline-result", default=str(BASELINE_CONTINUATION_RESULT))
    candidate_prepare.add_argument("--training-input", default=str(TRAINING_INPUT_V2))
    candidate_prepare.add_argument("--snapshot", default=str(EVALUATION_SNAPSHOT))
    candidate_prepare.add_argument("--artifact-tag", default="v2")
    candidate_endpoint = sub.add_parser("prepare-candidate-endpoint")
    candidate_endpoint.add_argument("--panel", required=True)
    candidate_endpoint.add_argument("--completion", required=True)
    candidate_endpoint.add_argument("--output", required=True)
    candidate_endpoint.add_argument("--natural-output-root", default=None)
    baseline_prepare = sub.add_parser("prepare-baseline")
    baseline_prepare.add_argument("--packet", required=True)
    baseline_prepare.add_argument("--output-packet", default=str(ROOT / "baseline-packet.json"))
    baseline_continuation = sub.add_parser("prepare-baseline-continuation")
    baseline_continuation.add_argument("--packet", required=True)
    baseline_continuation.add_argument("--output-packet", default=str(ROOT / "baseline-continuation-packet.json"))
    baseline_launch_parser = sub.add_parser("baseline-launch")
    baseline_launch_parser.add_argument("--packet", required=True)
    baseline_launch_parser.add_argument("--phase", choices=BASELINE_PHASES, required=True)
    baseline_worker_parser = sub.add_parser("baseline-worker")
    baseline_worker_parser.add_argument("--packet", required=True)
    baseline_worker_parser.add_argument("--phase", choices=BASELINE_PHASES, required=True)
    baseline_worker_parser.add_argument("--shard", required=True, type=int)
    baseline_worker_parser.add_argument("--output", required=True)
    baseline_merge_parser = sub.add_parser("baseline-merge")
    baseline_merge_parser.add_argument("--packet", required=True)
    baseline_merge_parser.add_argument("--phase", choices=BASELINE_PHASES, required=True)
    baseline_receipts_parser = sub.add_parser("baseline-receipts")
    baseline_receipts_parser.add_argument("--packet", required=True)
    baseline_receipts_parser.add_argument("--phase", choices=BASELINE_PHASES, required=True)
    baseline_consume_parser = sub.add_parser("consume-baseline")
    baseline_consume_parser.add_argument("--packet", required=True)
    baseline_consume_parser.add_argument("--phase", choices=BASELINE_PHASES, required=True)
    baseline_consume_parser.add_argument("--rows", required=True)
    baseline_consume_parser.add_argument("--output", required=True)
    bind = sub.add_parser("bind-candidate")
    bind.add_argument("--packet", required=True)
    bind.add_argument("--candidate", required=True)
    bind.add_argument("--output-packet", default=None)
    bind_panel = sub.add_parser("bind-candidate-panel")
    bind_panel.add_argument("--packet", required=True)
    bind_panel.add_argument("--candidate", required=True)
    bind_panel.add_argument("--output-packet", default=None)
    candidate_worker_parser = sub.add_parser("candidate-worker")
    candidate_worker_parser.add_argument("--packet", required=True)
    candidate_worker_parser.add_argument("--shard", required=True, type=int)
    candidate_worker_parser.add_argument("--physical-gpu", required=True, type=int)
    candidate_worker_parser.add_argument("--output", required=True)
    candidate_launch_parser = sub.add_parser("candidate-launch")
    candidate_launch_parser.add_argument("--packet", required=True)
    candidate_launch_parser.add_argument("--output", required=True)
    candidate_merge_parser = sub.add_parser("candidate-merge")
    candidate_merge_parser.add_argument("--packet", required=True)
    candidate_merge_parser.add_argument("--output", required=True)
    slice_parser = sub.add_parser("slice")
    slice_parser.add_argument("--packet", required=True)
    slice_parser.add_argument("--stable-rows", required=True)
    slice_parser.add_argument("--candidate-rows", required=True)
    slice_parser.add_argument("--output", required=True)
    slice_parser.add_argument("--image-id", action="append", required=True, type=int)
    consume_parser = sub.add_parser("consume")
    consume_parser.add_argument("--packet", required=True)
    consume_parser.add_argument("--stable-rows", required=True)
    consume_parser.add_argument("--candidate-rows", required=True)
    consume_parser.add_argument("--output", required=True)
    consume_panel_parser = sub.add_parser("consume-candidate-panel")
    consume_panel_parser.add_argument("--packet", required=True)
    consume_panel_parser.add_argument("--candidate-rows", required=True)
    consume_panel_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "select":
        manifests = []
        for spec in args.lane_manifest:
            require("=" in spec, "lane manifest must be ROLE=PATH")
            manifests.append(tuple(spec.split("=", 1)))
        result = freeze_selection(output=args.output, lane_manifests=manifests)
        print(json.dumps({key: result[key] for key in ("status", "source_images", "excluded_source_images", "eligible_images", "image_ids_sha256", "blind_review_ids_sha256")}, sort_keys=True))
    elif args.command == "prepare":
        result = prepare_packet(output=args.output)
        print(json.dumps({"status": result["status"], "records": len(result["records"]), "packet_sha256": file_hash(Path(args.output) / "packet.json")}))
    elif args.command == "prepare-candidate-panel":
        result = prepare_candidate_panel(
            output=args.output,
            current_packet_path=args.current_packet,
            legacy_packet_path=args.legacy_packet,
            baseline_packet_path=args.baseline_packet,
            baseline_consumer_path=args.baseline_consumer,
            baseline_result_path=args.baseline_result,
            training_input_path=args.training_input,
            snapshot_path=args.snapshot,
            artifact_tag=args.artifact_tag,
        )
        print(json.dumps({"status": result["status"], "records": len(result["records"]), "packet_sha256": file_hash(Path(args.output) / f"candidate-panel-{args.artifact_tag}.json")}, sort_keys=True))
    elif args.command == "prepare-candidate-endpoint":
        result = prepare_candidate_endpoint(
            panel_path=args.panel,
            completion_path=args.completion,
            output_path=args.output,
            natural_output_root=args.natural_output_root,
        )
        print(json.dumps({"status": result["status"], "adapter_fingerprint": result["adapter"]["fingerprint"], "output": args.output}, sort_keys=True))
    elif args.command == "prepare-baseline":
        result = prepare_baseline(packet_path=args.packet, output_path=args.output_packet)
        print(json.dumps({"status": result["status"], "slice_ids": result["launch_gate"]["slice_ids"], "packet_sha256": file_hash(args.output_packet)}))
    elif args.command == "prepare-baseline-continuation":
        result = prepare_baseline_continuation(prior_packet_path=args.packet, output_path=args.output_packet)
        print(json.dumps({"status": result["status"], "remaining254": len(result["phases"]["full"]["image_ids"]), "packet_sha256": file_hash(args.output_packet)}))
    elif args.command == "baseline-launch":
        result = baseline_launch(packet_path=args.packet, phase=args.phase)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "baseline-worker":
        baseline_worker(packet_path=args.packet, phase=args.phase, shard=args.shard, output=args.output)
        print(json.dumps({"status": "completed", "phase": args.phase, "shard": args.shard}, sort_keys=True))
    elif args.command == "baseline-merge":
        result = baseline_merge(packet_path=args.packet, phase=args.phase)
        print(json.dumps({"status": result["status"], "phase": args.phase, "rows": len(result["image_ids"]), "rows_path": result["rows"]["path"]}, sort_keys=True))
    elif args.command == "baseline-receipts":
        result = baseline_receipts(packet_path=args.packet, phase=args.phase)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "consume-baseline":
        result = consume_baseline(packet_path=args.packet, phase=args.phase, rows_path=args.rows, output=args.output)
        print(json.dumps({"status": result["status"], "phase": args.phase, "images": result["images"]}, sort_keys=True))
    elif args.command == "bind-candidate":
        packet = bind_candidate(packet_path=args.packet, candidate_path=args.candidate, output_path=args.output_packet)
        print(json.dumps({"status": packet["status"], "candidate_fingerprint": packet["scaled_terminal"]["adapter"]["fingerprint"]}))
    elif args.command == "bind-candidate-panel":
        packet = bind_candidate_panel(packet_path=args.packet, candidate_path=args.candidate, output_path=args.output_packet)
        print(json.dumps({"status": packet["status"], "candidate_fingerprint": packet["scaled_terminal"]["adapter"]["fingerprint"]}, sort_keys=True))
    elif args.command == "candidate-worker":
        candidate_worker(packet_path=args.packet, shard=args.shard, physical_gpu=args.physical_gpu, output=args.output)
        print(json.dumps({"status": "completed", "shard": args.shard}, sort_keys=True))
    elif args.command == "candidate-launch":
        result = candidate_launch(packet_path=args.packet, output=args.output)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "candidate-merge":
        result = candidate_merge(packet_path=args.packet, output=args.output)
        print(json.dumps({"status": result["status"], "rows": len(result["image_ids"]), "rows_path": result["rows"]["path"]}, sort_keys=True))
    elif args.command == "slice":
        result = consume_slice(
            packet_path=args.packet,
            stable_rows_path=args.stable_rows,
            candidate_rows_path=args.candidate_rows,
            output=args.output,
            image_ids=args.image_id,
        )
        print(json.dumps({"status": result["status"], "images": len(result["image_ids"])}))
    else:
        if args.command == "consume-candidate-panel":
            result = consume_candidate_panel(packet_path=args.packet, candidate_rows_path=args.candidate_rows, output=args.output)
            print(json.dumps({"status": result["status"], "images": result["denominators"]["combined640"], "blind32": result["denominators"]["blind_review32"]}, sort_keys=True))
        else:
            result = consume(packet_path=args.packet, stable_rows_path=args.stable_rows, candidate_rows_path=args.candidate_rows, output=args.output)
            print(json.dumps({"status": result["status"], "images": result["denominators"]["fresh256"], "blind32": result["denominators"]["blind_review32"]}))


if __name__ == "__main__":
    main()
