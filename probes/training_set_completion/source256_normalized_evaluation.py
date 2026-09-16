"""Control reuse and CPU scoring for the Source256 CE-normalization successor.

The successor generates only its new B-normalized endpoints.  This module
admits the accepted Source/A/original-B saved outputs from the fixed-prefix
experiment byte-for-byte before it reuses their scores.  It deliberately does
not alter the predecessor producer or regenerate a matched control.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from probes.training_set_completion import source256_evaluation as predecessor_evaluation
from probes.training_set_completion import source256_readback as predecessor_readback
from probes.training_set_completion import training


SCHEMA = "training_set_completion.source256_completion_ce_normalization_evaluation.v1"
PRIOR_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion"
)
PRIOR_PREPARATION = PRIOR_ROOT / "preparation/source256-admitted-v1/preparation.json"
PRIOR_TRIAL = PRIOR_ROOT / "runtime/main-v1/trial.json"
PRIOR_PACKET = PRIOR_ROOT / "runtime/main-v1/evaluation/packet-reducer-fixed.json"
PRIOR_RESULT = PRIOR_ROOT / "runtime/main-v1/evaluation/result.json"
PRIOR_PREPARATION_SHA256 = "9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242"
PRIOR_TRIAL_SHA256 = "2cfd2a39d3aa3a682002586fe7849225b44a3d4dc80c643d39b33f5e989e409d"
PRIOR_PACKET_SHA256 = "095f9fd20ea37a0b0167c1003ee97eb433df01c9023420168fc6d65910173c2a"
PRIOR_RESULT_SHA256 = "1c45305af5c3b7689e75465092773437005a312632946ad2a83e706e48a8083d"
CONTROL_ENDPOINTS = (
    ("Source0", "Source", 0),
    ("A16", "A", 16),
    ("A64", "A", 64),
    ("B16", "B", 16),
    ("B64", "B", 64),
)
NORMALIZATION_CONTRACT = {
    "completion_denominator": "same_image_canonical_full_target_token_count_including_eos",
    "binary_ce_masks": True,
    "no_geometry_change": True,
    "geometry_weight": 0.01,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _fixed_binding(path: Path, sha256: str, name: str) -> dict[str, Any]:
    observed = training.binding(path)
    require(observed["path"] == str(path.resolve()), f"{name} fixed path")
    require(observed["sha256"] == sha256, f"{name} fixed digest")
    return observed


def _validate_prior_result(
    *,
    packet: Mapping[str, Any],
    checked_packet: Mapping[str, Any],
    result: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit controls only after rechecking their raw immutable shard payloads."""

    require(
        result.get("schema") == f"{predecessor_evaluation.SCHEMA}.result"
        and result.get("status") == "completed_saved_readback_evaluation",
        "accepted predecessor result schema/status",
    )
    require(result.get("packet") == bindings["packet"], "result repaired-packet binding")
    require(result.get("preparation") == bindings["preparation"], "result fixed-bank binding")
    require(
        result.get("matching_contract") == packet.get("evaluation_contract"),
        "result matching contract",
    )
    scores = result.get("scores")
    require(isinstance(scores, Mapping) and set(scores) == {item[0] for item in CONTROL_ENDPOINTS}, "control score endpoints")
    endpoint_by_label = {str(item["label"]): item for item in packet["endpoints"]}
    require(set(endpoint_by_label) == set(scores), "packet/result endpoint labels")

    admitted_shards: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, arm, step in CONTROL_ENDPOINTS:
        endpoint = endpoint_by_label[label]
        score = scores[label]
        require(
            score.get("schema") == f"{predecessor_evaluation.SCHEMA}.endpoint_score"
            and score.get("status") == "scored_saved_natural_readback"
            and score.get("endpoint") == {"label": label, "arm": arm, "step": step}
            and score.get("preparation") == bindings["preparation"],
            f"{label} accepted score identity",
        )
        split_bindings: dict[str, list[dict[str, Any]]] = {}
        for split in predecessor_readback.SPLIT_COUNTS:
            # This replays the packet's decoder, cohort, terminal, batch-size and
            # generated-row bindings.  It does not score or generate any tokens.
            _rows, observed = predecessor_evaluation._admit_rows(
                checked=checked_packet, endpoint=endpoint, split=split
            )
            require(
                score.get("readback_shards", {}).get(split) == observed,
                f"{label} {split} accepted raw-shard bindings",
            )
            split_score = score.get("splits", {}).get(split)
            require(
                isinstance(split_score, Mapping)
                and split_score.get("image_count") == predecessor_readback.SPLIT_COUNTS[split]
                and split_score.get("target_count", 0) > 0
                and split_score.get("confirmed_false_instance_count") is None
                and split_score.get("physical_debt_status") == "unresolved_no_endpoint_visual_review",
                f"{label} {split} fixed evidence semantics",
            )
            split_bindings[split] = observed
        admitted_shards[label] = split_bindings
    return {
        "scores": dict(scores),
        "admitted_shards": admitted_shards,
    }


def admit_control_reuse() -> dict[str, Any]:
    """Return the exact reusable Source/A/B score surface, or fail closed.

    The predecessor evaluator is deliberately invoked on the corrected packet;
    therefore an old summary cannot be reused after a raw-shard, tokenizer,
    decode, bank, or bound-producer drift.
    """

    bindings = {
        "preparation": _fixed_binding(
            PRIOR_PREPARATION, PRIOR_PREPARATION_SHA256, "predecessor preparation"
        ),
        "trial": _fixed_binding(PRIOR_TRIAL, PRIOR_TRIAL_SHA256, "predecessor trial"),
        "packet": _fixed_binding(PRIOR_PACKET, PRIOR_PACKET_SHA256, "repaired predecessor packet"),
        "result": _fixed_binding(PRIOR_RESULT, PRIOR_RESULT_SHA256, "accepted predecessor result"),
    }
    packet = read(PRIOR_PACKET)
    checked_packet = predecessor_evaluation._validate_packet(packet)
    require(
        packet.get("sources", {}).get("preparation") == bindings["preparation"]
        and packet.get("sources", {}).get("trial") == bindings["trial"],
        "predecessor packet scientific anchors",
    )
    reused = _validate_prior_result(
        packet=packet,
        checked_packet=checked_packet,
        result=read(PRIOR_RESULT),
        bindings=bindings,
    )
    return {
        "schema": f"{SCHEMA}.control_reuse",
        "status": "admitted_exact_prior_controls",
        "predecessor": bindings,
        "readback_plan": packet["sources"]["readback_plan"],
        "batch4_qualification": packet["sources"]["batch4_qualification"],
        "evaluation_contract": packet["evaluation_contract"],
        "scores": reused["scores"],
        "admitted_shards": reused["admitted_shards"],
        "scientific_identity": {
            "known_owner_bank": bindings["preparation"],
            "natural_decode": packet["sources"]["readback_plan"],
            "evaluation_producer": packet["sources"]["producer"],
            "repaired_packet": bindings["packet"],
            "same_control_endpoints": [
                {"label": label, "arm": arm, "step": step}
                for label, arm, step in CONTROL_ENDPOINTS
            ],
        },
    }


def publish_control_reuse(*, output: Path) -> dict[str, Any]:
    """Publish one CPU-only reusable-control receipt without model execution."""

    require(not output.exists() and not output.is_symlink(), "control reuse receipt collision")
    value = admit_control_reuse()
    value["producer"] = training.binding(Path(__file__))
    value["content_sha256"] = training.digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    training.publish(output, value)
    return value


def validate_control_reuse(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck a saved receipt against current immutable predecessor evidence."""

    require(value.get("schema") == f"{SCHEMA}.control_reuse", "control reuse schema")
    require(
        value.get("content_sha256")
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "control reuse content digest",
    )
    require(value.get("producer") == training.binding(Path(__file__)), "control reuse producer binding")
    current = admit_control_reuse()
    for key in (
        "status",
        "predecessor",
        "readback_plan",
        "batch4_qualification",
        "evaluation_contract",
        "scores",
        "admitted_shards",
        "scientific_identity",
    ):
        require(value.get(key) == current.get(key), f"control reuse {key} changed")
    return dict(value)


def validate_control_reuse_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Check a packet-bound reuse receipt without rereading all 80 old shards.

    The controller performs :func:`validate_control_reuse` before it starts a
    successor process.  Individual GPU workers repeat this cheaper immutable
    identity check so a worker cannot silently switch its bank, decode plan or
    accepted control result after that admission.
    """

    require(value.get("schema") == f"{SCHEMA}.control_reuse", "control reuse schema")
    require(value.get("status") == "admitted_exact_prior_controls", "control reuse status")
    require(
        value.get("content_sha256")
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "control reuse content digest",
    )
    require(value.get("producer") == training.binding(Path(__file__)), "control reuse producer binding")
    expected = {
        "preparation": _fixed_binding(
            PRIOR_PREPARATION, PRIOR_PREPARATION_SHA256, "predecessor preparation"
        ),
        "trial": _fixed_binding(PRIOR_TRIAL, PRIOR_TRIAL_SHA256, "predecessor trial"),
        "packet": _fixed_binding(PRIOR_PACKET, PRIOR_PACKET_SHA256, "repaired predecessor packet"),
        "result": _fixed_binding(PRIOR_RESULT, PRIOR_RESULT_SHA256, "accepted predecessor result"),
    }
    require(value.get("predecessor") == expected, "control reuse predecessor identity")
    require(
        isinstance(value.get("readback_plan"), Mapping)
        and isinstance(value.get("batch4_qualification"), Mapping)
        and isinstance(value.get("evaluation_contract"), Mapping),
        "control reuse readback/evaluation identities",
    )
    return dict(value)


def _normalized_runtime() -> Any:
    from probes.training_set_completion import source256_normalized_training

    return source256_normalized_training


def _normalized_trial() -> Any:
    from probes.training_set_completion import source256_normalized_trial

    return source256_normalized_trial


def _normalized_readback() -> Any:
    from probes.training_set_completion import source256_normalized_readback

    return source256_normalized_readback


def _verified_binding(value: Mapping[str, Any], name: str) -> Path:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    path = Path(str(value["path"])).resolve(strict=True)
    require(training.binding(path) == dict(value), f"{name} binding changed")
    return path


def _validate_actual_entry_qualification(
    *,
    qualification_path: Path,
    main_manifest_path: Path,
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate main64 on the training owner's two-update successor receipt."""

    runtime = _normalized_runtime()
    value = read(qualification_path)
    qualification_manifest_path = _verified_binding(
        value["qualification_manifest"], "qualification manifest"
    )
    qualification_terminal_path = _verified_binding(
        value["training_terminal"], "qualification terminal"
    )
    runtime.validate_qualification_receipt(
        value, manifest_path=qualification_manifest_path
    )
    qualification_manifest = runtime.validate_training_manifest(
        read(qualification_manifest_path)
    )
    main_manifest = runtime.validate_training_manifest(read(main_manifest_path))
    require(
        qualification_manifest.get("arm") == "B-normalized"
        and qualification_manifest.get("mode") == "qualification"
        and main_manifest.get("arm") == "B-normalized"
        and main_manifest.get("mode") == "main"
        and value.get("preparation") == dict(preparation)
        and value.get("normalization") == NORMALIZATION_CONTRACT,
        "accepted B-normalized actual-entry qualification",
    )
    reader = _normalized_readback()
    checked_terminal = reader.validate_training_terminal(
        training_manifest_path=qualification_manifest_path,
        terminal_path=qualification_terminal_path,
    )
    terminal_value = checked_terminal["terminal"]
    require(
        terminal_value.get("status") == "completed"
        and terminal_value.get("arm") == "B-normalized"
        and terminal_value.get("mode") == "qualification",
        "completed B-normalized qualification terminal",
    )
    return dict(value)


def _validate_successor_trial(
    *,
    trial_path: Path,
    main_manifest_path: Path,
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    value = read(trial_path)
    trial = _normalized_trial()
    checked = trial.validate_trial(value)
    require(
        checked.get("mode") == "main"
        and checked.get("preparation") == dict(preparation)
        and checked.get("arms") == {"B-normalized": training.binding(main_manifest_path)},
        "B-normalized main trial identity",
    )
    return checked


def _training_command(
    manifest_path: Path, output: Path, release_receipt: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--module",
        "probes.training_set_completion.source256_normalized_training",
        "run",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--release-receipt",
        str(release_receipt),
    ]


def prepare_packet(
    *,
    control_reuse_path: Path,
    trial_path: Path,
    main_manifest_path: Path,
    qualification_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Publish a held main64 packet; no model or GPU process is started."""

    require(not output.exists() and not output.is_symlink(), "normalized evaluation packet collision")
    control_reuse_path = control_reuse_path.resolve(strict=True)
    control = validate_control_reuse(read(control_reuse_path))
    main_manifest_path = main_manifest_path.resolve(strict=True)
    runtime = _normalized_runtime()
    manifest = runtime.validate_training_manifest(read(main_manifest_path))
    require(
        manifest.get("arm") == "B-normalized"
        and manifest.get("mode") == "main"
        and manifest.get("preparation") == control["predecessor"]["preparation"],
        "new main manifest preserves fixed scientific identity",
    )
    trial_path = trial_path.resolve(strict=True)
    _validate_successor_trial(
        trial_path=trial_path,
        main_manifest_path=main_manifest_path,
        preparation=control["predecessor"]["preparation"],
    )
    qualification_path = qualification_path.resolve(strict=True)
    _validate_actual_entry_qualification(
        qualification_path=qualification_path,
        main_manifest_path=main_manifest_path,
        preparation=control["predecessor"]["preparation"],
    )
    reader = _normalized_readback()
    runtime_root = output.parent.parent
    training_root = main_manifest_path.parent / "training"
    training_release_path = runtime_root / "main-training-release.json"
    endpoints = []
    for label, step in reader.ENDPOINTS:
        jobs = []
        for split in predecessor_readback.SPLIT_COUNTS:
            for shard in range(predecessor_readback.ENDPOINT_SHARDS):
                shard_output = runtime_root / "readback" / label / split / f"shard-{shard:02d}.json"
                jobs.append(
                    {
                        "split": split,
                        "shard": shard,
                        "visible_device": shard,
                        "output": str(shard_output),
                        "command": reader.endpoint_command(
                            control_reuse_path=control_reuse_path,
                            plan_path=Path(control["readback_plan"]["path"]),
                            qualification_path=Path(control["batch4_qualification"]["path"]),
                            training_manifest_path=main_manifest_path,
                            terminal_path=training_root / "terminal.json",
                            label=label,
                            step=step,
                            split=split,
                            shard=shard,
                            output=shard_output,
                            device="cuda:0",
                        ),
                    }
                )
        endpoints.append(
            {
                "label": label,
                "arm": "B-normalized",
                "step": step,
                "training_manifest": training.binding(main_manifest_path),
                "training_terminal": str(training_root / "terminal.json"),
                "jobs": jobs,
            }
        )
    value: dict[str, Any] = {
        "schema": f"{SCHEMA}.packet",
        "status": "held_for_main64_lead_release",
        "scope": "new B-normalized fresh64 training plus new-only Bnormalized16/64 batch4 readback; exact accepted Source/A/original-B reuse",
        "sources": {
            "control_reuse": training.binding(control_reuse_path),
            "successor_trial": training.binding(trial_path),
            "main_training_manifest": training.binding(main_manifest_path),
            "actual_entry_qualification": training.binding(qualification_path),
            "producer": training.binding(Path(__file__)),
            "readback_producer": training.binding(Path(reader.__file__)),
        },
        "normalization": NORMALIZATION_CONTRACT,
        "control_identity": {
            "predecessor": control["predecessor"],
            "readback_plan": control["readback_plan"],
            "batch4_qualification": control["batch4_qualification"],
        },
        "training_launch": {
            "visible_devices": [0, 1, 2, 3],
            "command": _training_command(
                main_manifest_path, training_root, training_release_path
            ),
            "output": str(training_root),
            "terminal": str(training_root / "terminal.json"),
            "release_receipt": str(training_release_path),
            "start_identity": "original Source step2444 plus paired embeddings and fresh AdamW",
        },
        "endpoints": endpoints,
        "evaluation_contract": {
            **control["evaluation_contract"],
            "successor_comparisons": "B-normalized versus Source0, original B and canonical A at saved16/64; primary global class-agnostic IoU50 FN/G/L, secondary class-consistent50/60/80",
            "unknown_semantics": "annotation-unmatched predictions remain unknown; no endpoint visual review or automatic false-instance count",
        },
        "bounds": {
            "training_updates": 64,
            "training_logical_forwards": 4096,
            "training_model_calls": 2048,
            "saved_checkpoints": [16, 32, 64],
            "new_endpoint_count": 2,
            "new_readback_shards": 32,
            "new_natural_image_requests": 768,
            "new_batch4_calls": 192,
        },
        "orchestration": {
            "tmux_session": "coordexp-source256-completion-ce-normalization-main-v1",
            "success_log_pattern": "SOURCE256_NORMALIZED_COMPLETED",
            "failure_log_pattern": "SOURCE256_NORMALIZED_FAILED",
            "recovery": "retain compatible completed training and shards; generate missing shards only; a reducer failure never retrains or regenerates valid model outputs",
        },
        "reducer": {
            "command": [
                sys.executable,
                "-m",
                "probes.training_set_completion.source256_normalized_evaluation",
                "reduce",
                "--packet",
                str(output),
                "--output",
                str(runtime_root / "evaluation/result.json"),
            ],
            "output": str(runtime_root / "evaluation/result.json"),
        },
        "launch": "held; packet creation makes no model calls",
        "content_sha256": None,
    }
    value["content_sha256"] = training.digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    training.publish(output, value)
    _validate_packet(value)
    return value


def _validate_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value.get("schema") == f"{SCHEMA}.packet", "normalized packet schema")
    require(
        value.get("content_sha256")
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "normalized packet digest",
    )
    sources = value.get("sources")
    require(isinstance(sources, Mapping) and set(sources) == {
        "control_reuse", "successor_trial", "main_training_manifest",
        "actual_entry_qualification", "producer", "readback_producer",
    }, "normalized packet source bindings")
    control_path = _verified_binding(sources["control_reuse"], "control reuse")
    control = validate_control_reuse_identity(read(control_path))
    trial_path = _verified_binding(sources["successor_trial"], "successor trial")
    manifest_path = _verified_binding(sources["main_training_manifest"], "main manifest")
    qualification_path = _verified_binding(sources["actual_entry_qualification"], "actual-entry qualification")
    require(sources["producer"] == training.binding(Path(__file__)), "normalized evaluator producer")
    reader = _normalized_readback()
    require(sources["readback_producer"] == training.binding(Path(reader.__file__)), "normalized readback producer")
    runtime = _normalized_runtime()
    manifest = runtime.validate_training_manifest(read(manifest_path))
    require(
        manifest.get("arm") == "B-normalized"
        and manifest.get("mode") == "main"
        and manifest.get("preparation") == control["predecessor"]["preparation"],
        "packet main manifest scientific identity",
    )
    _validate_successor_trial(
        trial_path=trial_path,
        main_manifest_path=manifest_path,
        preparation=control["predecessor"]["preparation"],
    )
    _validate_actual_entry_qualification(
        qualification_path=qualification_path,
        main_manifest_path=manifest_path,
        preparation=control["predecessor"]["preparation"],
    )
    require(value.get("normalization") == NORMALIZATION_CONTRACT, "only normalized CE delta")
    require(value.get("control_identity") == {
        "predecessor": control["predecessor"],
        "readback_plan": control["readback_plan"],
        "batch4_qualification": control["batch4_qualification"],
    }, "packet control identity")
    require(value.get("bounds") == {
        "training_updates": 64,
        "training_logical_forwards": 4096,
        "training_model_calls": 2048,
        "saved_checkpoints": [16, 32, 64],
        "new_endpoint_count": 2,
        "new_readback_shards": 32,
        "new_natural_image_requests": 768,
        "new_batch4_calls": 192,
    }, "fixed new-only runtime bounds")
    training_launch = value.get("training_launch")
    expected_training_root = manifest_path.parent / "training"
    expected_release = manifest_path.parent.parent / "main-training-release.json"
    require(
        isinstance(training_launch, Mapping)
        and training_launch.get("visible_devices") == [0, 1, 2, 3]
        and training_launch.get("output") == str(expected_training_root)
        and training_launch.get("terminal") == str(expected_training_root / "terminal.json")
        and training_launch.get("release_receipt") == str(expected_release)
        and training_launch.get("command")
        == _training_command(manifest_path, expected_training_root, expected_release)
        and training_launch.get("start_identity")
        == "original Source step2444 plus paired embeddings and fresh AdamW",
        "fixed four-rank fresh normalized training launch",
    )
    endpoints = value.get("endpoints")
    require(isinstance(endpoints, list) and [(item.get("label"), item.get("step")) for item in endpoints] == list(reader.ENDPOINTS), "new endpoint registry")
    for endpoint in endpoints:
        require(
            endpoint.get("arm") == "B-normalized"
            and endpoint.get("training_manifest") == sources["main_training_manifest"]
            and endpoint.get("training_terminal")
            == str(expected_training_root / "terminal.json")
            and len(endpoint.get("jobs", [])) == 16,
            "new endpoint training identity",
        )
        for split in predecessor_readback.SPLIT_COUNTS:
            selected = [job for job in endpoint["jobs"] if job.get("split") == split]
            require(
                len(selected) == 8
                and [job.get("shard") for job in selected] == list(range(8))
                and [job.get("visible_device") for job in selected] == list(range(8)),
                f"{endpoint['label']} {split} eight-shard plan",
            )
            for job in selected:
                shard = int(job["shard"])
                expected_output = (
                    expected_training_root.parent.parent
                    / "readback"
                    / str(endpoint["label"])
                    / split
                    / f"shard-{shard:02d}.json"
                )
                require(
                    job.get("output") == str(expected_output)
                    and job.get("command")
                    == reader.endpoint_command(
                        control_reuse_path=control_path,
                        plan_path=Path(control["readback_plan"]["path"]),
                        qualification_path=Path(control["batch4_qualification"]["path"]),
                        training_manifest_path=manifest_path,
                        terminal_path=expected_training_root / "terminal.json",
                        label=str(endpoint["label"]),
                        step=int(endpoint["step"]),
                        split=split,
                        shard=shard,
                        output=expected_output,
                        device="cuda:0",
                    ),
                    f"{endpoint['label']} {split} shard-{shard:02d} immutable command",
                )
    return {
        "packet": dict(value),
        "control": control,
        "paths": {
            "control_reuse": control_path,
            "trial": trial_path,
            "manifest": manifest_path,
            "qualification": qualification_path,
        },
    }


def _admit_new_rows(
    *, checked: Mapping[str, Any], endpoint: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reader = _normalized_readback()
    paths = checked["paths"]
    control = checked["control"]
    jobs = [job for job in endpoint["jobs"] if job["split"] == split]
    require(len(jobs) == 8, "new endpoint split shard count")
    admitted, bindings = [], []
    for job in jobs:
        path = Path(job["output"]).resolve(strict=True)
        value = read(path)
        reader.validate_endpoint_shard(
            value=value,
            control_reuse_path=paths["control_reuse"],
            plan_path=Path(control["readback_plan"]["path"]),
            qualification_path=Path(control["batch4_qualification"]["path"]),
            training_manifest_path=paths["manifest"],
            terminal_path=Path(endpoint["training_terminal"]),
            label=str(endpoint["label"]),
            step=int(endpoint["step"]),
            split=split,
            shard=int(job["shard"]),
        )
        admitted.extend(dict(row) for row in value["generation"]["rows"])
        bindings.append(training.binding(path))
    plan = predecessor_readback.validate_plan(read(control["readback_plan"]["path"]))
    order = plan["cohorts"][split]
    by_image = {int(row["image_id"]): row for row in admitted}
    require(len(by_image) == len(order) and set(by_image) == set(order), "new endpoint split cohort")
    return [by_image[image_id] for image_id in order], bindings


def _score_new_endpoints(
    *, checked: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Recompute the two new score payloads from their retained raw rows."""

    predecessor_checked = predecessor_evaluation._validate_packet(read(PRIOR_PACKET))
    from transformers import AutoTokenizer

    raw = predecessor_checked["prepared"]["preparation"]
    # The predecessor repair established that this field names tokenizer.json;
    # AutoTokenizer intentionally receives its directory, never the JSON file.
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(raw["identity"]["runtime_contract"]["tokenizer_path"]).parent),
        local_files_only=True,
    )
    targets = predecessor_evaluation._targets(predecessor_checked["prepared"])
    contexts = predecessor_evaluation._contexts(predecessor_checked["prepared"])
    new_scores: dict[str, dict[str, Any]] = {}
    for endpoint in checked["packet"]["endpoints"]:
        split_scores, shard_sources = {}, {}
        for split in predecessor_readback.SPLIT_COUNTS:
            rows, bindings = _admit_new_rows(checked=checked, endpoint=endpoint, split=split)
            split_scores[split] = predecessor_evaluation._score_split(
                split=split,
                rows=rows,
                targets=targets[split],
                contexts=contexts[split],
                tokenizer=tokenizer,
            )
            shard_sources[split] = bindings
        new_scores[str(endpoint["label"])] = {
            # The old schema is intentional: the matching and burden semantics
            # are identical, allowing the original owner-identity comparator.
            "schema": f"{predecessor_evaluation.SCHEMA}.endpoint_score",
            "status": "scored_saved_natural_readback",
            "endpoint": {key: endpoint[key] for key in ("label", "arm", "step")},
            "preparation": control["predecessor"]["preparation"],
            "readback_shards": shard_sources,
            "splits": split_scores,
        }
    return new_scores


def _report_surface(
    *, scores: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Project decision-bearing metrics without converting unknowns to false positives."""

    endpoint_diagnostics: dict[str, dict[str, Any]] = {}
    for label, score in scores.items():
        splits: dict[str, Any] = {}
        for split in predecessor_readback.SPLIT_COUNTS:
            split_score = score["splits"][split]
            primary = split_score["primary_class_agnostic_iou50"]
            burden = split_score["burden"]
            require(
                split_score.get("confirmed_false_instance_count") is None
                and split_score.get("physical_debt_status")
                == "unresolved_no_endpoint_visual_review",
                f"{label} {split} unknown/physical-debt semantics",
            )
            splits[split] = {
                "primary_class_agnostic_iou50": {
                    key: primary[key]
                    for key in ("target_count", "matched_count", "missing_count", "coverage")
                },
                "secondary_category_consistent_iou50_60_80": {
                    threshold: {
                        key: split_score["class_consistent"][threshold][key]
                        for key in ("target_count", "matched_count", "missing_count", "coverage")
                    }
                    for threshold in ("0.5", "0.6", "0.8")
                },
                "repeat_proxy": {
                    "strict_repeat_row_count": burden.get("strict_repeat_row_count", 0),
                    "semantics": "class-agnostic normalized-bin IoU strictly >0.95 against an earlier valid row; each later row counted once",
                },
                "malformed_row_count": burden.get("malformed_row_count", 0),
                "cap_eos_debt": {
                    "cap_debt": burden.get("cap_debt", 0),
                    "eos_debt": burden.get("eos_debt", 0),
                },
                "geometry_debt": {
                    "invalid_geometry_count": burden.get("invalid_geometry_count", 0),
                    "physical_debt_status": split_score["physical_debt_status"],
                },
                "annotation_unmatched": {
                    "annotation_unmatched_prediction_count": burden.get(
                        "annotation_unmatched_prediction_count", 0
                    ),
                    "confirmed_false_instance_count": None,
                    "semantics": "unknown relative to the fixed known-owner bank; not a confirmed false positive",
                },
            }
        endpoint_diagnostics[label] = splits
    return {
        "primary_estimand": "known-owner global class-agnostic IoU50 matched/missing count and paired retained/gained/lost identities",
        "secondary_estimand": "class-consistent IoU50/60/80 coverage diagnostics",
        "endpoint_diagnostics": endpoint_diagnostics,
        "dev_retention_vs_controls": {
            label: dict(comparison["splits"]["dev"]["primary_class_agnostic_iou50"])
            for label, comparison in comparisons.items()
        },
        "unknown_policy": "annotation-unmatched predictions remain unknown; no endpoint visual review or detector is used to claim physical false instances",
    }


def reduce(*, packet_path: Path, output: Path) -> dict[str, Any]:
    """Score new B-normalized saved rows and compare them with admitted controls."""

    require(not output.exists() and not output.is_symlink(), "normalized result collision")
    checked = _validate_packet(read(packet_path))
    # Re-run full control admission only at the CPU reducer boundary.  Workers
    # use a cheaper identity check; the final comparison cannot use stale raw
    # controls or a copied historical summary.
    control = validate_control_reuse(read(checked["paths"]["control_reuse"]))
    new_scores = _score_new_endpoints(checked=checked, control=control)
    all_scores = {**control["scores"], **new_scores}
    comparisons = {
        "Bnormalized16_vs_Source0": predecessor_evaluation.compare_scores(all_scores["Source0"], all_scores["Bnormalized16"], label="Bnormalized16_vs_Source0"),
        "Bnormalized64_vs_Source0": predecessor_evaluation.compare_scores(all_scores["Source0"], all_scores["Bnormalized64"], label="Bnormalized64_vs_Source0"),
        "Bnormalized16_vs_B16": predecessor_evaluation.compare_scores(all_scores["B16"], all_scores["Bnormalized16"], label="Bnormalized16_vs_B16"),
        "Bnormalized64_vs_B64": predecessor_evaluation.compare_scores(all_scores["B64"], all_scores["Bnormalized64"], label="Bnormalized64_vs_B64"),
        "Bnormalized16_vs_A16": predecessor_evaluation.compare_scores(all_scores["A16"], all_scores["Bnormalized16"], label="Bnormalized16_vs_A16"),
        "Bnormalized64_vs_A64": predecessor_evaluation.compare_scores(all_scores["A64"], all_scores["Bnormalized64"], label="Bnormalized64_vs_A64"),
    }
    value = {
        "schema": f"{SCHEMA}.result",
        "status": "completed_saved_readback_evaluation",
        "packet": training.binding(packet_path),
        "control_reuse": training.binding(checked["paths"]["control_reuse"]),
        "preparation": control["predecessor"]["preparation"],
        "matching_contract": checked["packet"]["evaluation_contract"],
        "new_scores": new_scores,
        "reused_controls": control["predecessor"],
        "comparisons": comparisons,
        "report": _report_surface(scores=all_scores, comparisons=comparisons),
        "disposition": "No scalar composite or automatic winner; unknown annotation-unmatched rows remain unknown and technical acceptance remains distinct from the scientific conclusion.",
    }
    training.publish(output, value)
    return value


def validate_result(value: Mapping[str, Any], *, packet_path: Path) -> dict[str, Any]:
    """Validate a completed reducer product without regenerating model outputs."""

    checked = _validate_packet(read(packet_path))
    control = validate_control_reuse(read(checked["paths"]["control_reuse"]))
    require(
        value.get("schema") == f"{SCHEMA}.result"
        and value.get("status") == "completed_saved_readback_evaluation"
        and value.get("packet") == training.binding(packet_path)
        and value.get("control_reuse") == training.binding(checked["paths"]["control_reuse"])
        and value.get("preparation") == control["predecessor"]["preparation"]
        and value.get("matching_contract") == checked["packet"]["evaluation_contract"]
        and value.get("reused_controls") == control["predecessor"],
        "normalized result identity",
    )
    scores = value.get("new_scores")
    require(isinstance(scores, Mapping) and set(scores) == {"Bnormalized16", "Bnormalized64"}, "new score endpoints")
    for label, arm, step in (("Bnormalized16", "B-normalized", 16), ("Bnormalized64", "B-normalized", 64)):
        score = scores[label]
        require(
            score.get("schema") == f"{predecessor_evaluation.SCHEMA}.endpoint_score"
            and score.get("status") == "scored_saved_natural_readback"
            and score.get("endpoint") == {"label": label, "arm": arm, "step": step}
            and score.get("preparation") == control["predecessor"]["preparation"],
            f"{label} result score identity",
        )
        endpoint = next(item for item in checked["packet"]["endpoints"] if item["label"] == label)
        for split in predecessor_readback.SPLIT_COUNTS:
            _rows, shard_bindings = _admit_new_rows(
                checked=checked, endpoint=endpoint, split=split
            )
            split_score = score.get("splits", {}).get(split)
            require(
                score.get("readback_shards", {}).get(split) == shard_bindings
                and isinstance(split_score, Mapping)
                and split_score.get("image_count") == predecessor_readback.SPLIT_COUNTS[split]
                and split_score.get("confirmed_false_instance_count") is None
                and split_score.get("physical_debt_status") == "unresolved_no_endpoint_visual_review",
                f"{label} {split} retained scored-shard identity",
            )
    require(set(value.get("comparisons", {})) == {
        "Bnormalized16_vs_Source0", "Bnormalized64_vs_Source0",
        "Bnormalized16_vs_B16", "Bnormalized64_vs_B64",
        "Bnormalized16_vs_A16", "Bnormalized64_vs_A64",
    }, "normalized result comparisons")
    expected_scores = _score_new_endpoints(checked=checked, control=control)
    require(scores == expected_scores, "new score payload must be recomputed from retained rows")
    all_scores = {**control["scores"], **expected_scores}
    expected_comparisons = {
        "Bnormalized16_vs_Source0": predecessor_evaluation.compare_scores(all_scores["Source0"], all_scores["Bnormalized16"], label="Bnormalized16_vs_Source0"),
        "Bnormalized64_vs_Source0": predecessor_evaluation.compare_scores(all_scores["Source0"], all_scores["Bnormalized64"], label="Bnormalized64_vs_Source0"),
        "Bnormalized16_vs_B16": predecessor_evaluation.compare_scores(all_scores["B16"], all_scores["Bnormalized16"], label="Bnormalized16_vs_B16"),
        "Bnormalized64_vs_B64": predecessor_evaluation.compare_scores(all_scores["B64"], all_scores["Bnormalized64"], label="Bnormalized64_vs_B64"),
        "Bnormalized16_vs_A16": predecessor_evaluation.compare_scores(all_scores["A16"], all_scores["Bnormalized16"], label="Bnormalized16_vs_A16"),
        "Bnormalized64_vs_A64": predecessor_evaluation.compare_scores(all_scores["A64"], all_scores["Bnormalized64"], label="Bnormalized64_vs_A64"),
    }
    require(value.get("comparisons") == expected_comparisons, "paired comparison payload recomputation")
    require(
        value.get("report")
        == _report_surface(scores=all_scores, comparisons=expected_comparisons),
        "reported burden and retention surface",
    )
    return dict(value)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("admit-controls")
    publish.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-controls")
    verify.add_argument("--receipt", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--control-reuse", type=Path, required=True)
    prepare.add_argument("--trial", type=Path, required=True)
    prepare.add_argument("--main-manifest", type=Path, required=True)
    prepare.add_argument("--qualification", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    reducer = sub.add_parser("reduce")
    reducer.add_argument("--packet", type=Path, required=True)
    reducer.add_argument("--output", type=Path, required=True)
    result_check = sub.add_parser("verify-result")
    result_check.add_argument("--packet", type=Path, required=True)
    result_check.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "admit-controls":
        result = publish_control_reuse(output=args.output)
    elif args.command == "verify-controls":
        result = validate_control_reuse(read(args.receipt))
    elif args.command == "prepare":
        result = prepare_packet(
            control_reuse_path=args.control_reuse,
            trial_path=args.trial,
            main_manifest_path=args.main_manifest,
            qualification_path=args.qualification,
            output=args.output,
        )
    elif args.command == "reduce":
        result = reduce(packet_path=args.packet, output=args.output)
    else:
        result = validate_result(read(args.result), packet_path=args.packet)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
