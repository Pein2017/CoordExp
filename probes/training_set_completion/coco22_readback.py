"""COCO22 native batched natural readback with immutable rows and missing-only recovery.

The tested COCO227 worker supplies native media preparation, generation, row
validation and raw-batch receipts. Its execution-local constants are scoped
here; the old producer and its hash-bound outputs are never edited.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from probes.training_set_completion import coco227_readback as native
from probes.training_set_completion import training
from src.runtime.process_completion import start_process_waiter

ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion")
SCHEMA = "training_set_completion.coco22_readback.v1"
IMAGE_COUNT = 22
STEPS = (0, 8, 16, 32, 64, 128, 256)
ARMS = ("S", "Source")
CAP = native.CAP
POLICY = dict(native.POLICY)
QUALIFICATION_TMUX = "coordexp-coco22-readback-qualification"
TRIAL_TMUX = "coordexp-coco22-cumulative-trial"
CONFIGS = {"serial": 1, "batch2": 2, "batch3": 3}
_PRIOR_CHECKPOINT_VALIDATOR = native._checkpoint_from_terminal
READBACK_AMENDMENT = ROOT / "execution-preparation-v1/predeclared-readback-amendment.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _routes(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = manifest.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT, "22 readback routes")
    require(len({route.get("image_id") for route in routes}) == IMAGE_COUNT, "unique readback image IDs")
    for route in routes:
        training.validate_route(
            route, eos_token_id=native.EOS,
            coordinate_token_ids=manifest["validity_hinge"]["coordinate_token_ids"],
        )
        require(len(route["continuation_token_ids"]) <= CAP, "teacher sequence exceeds natural cap")
    return [dict(route) for route in routes]


def _checkpoint_from_terminal(
    terminal: Mapping[str, Any], *, step: int, adapter: Mapping[str, Any]
) -> None:
    if step == 0:
        require(
            terminal.get("schema") == f"{SCHEMA}.source0_terminal"
            and terminal.get("status") == "cold_source_ready"
            and terminal.get("source_adapter") == dict(adapter),
            "cold source0 adapter receipt",
        )
    else:
        _PRIOR_CHECKPOINT_VALIDATOR(terminal, step=step, adapter=adapter)


@contextlib.contextmanager
def _native_scope():
    """Adapt only this process's call to the unchanged prior native worker."""
    names = {
        "SCHEMA": SCHEMA, "IMAGE_COUNT": IMAGE_COUNT, "ARMS": ARMS,
        "ENDPOINT_STEPS": STEPS, "_routes": _routes,
        "_checkpoint_from_terminal": _checkpoint_from_terminal,
    }
    prior = {name: getattr(native, name) for name in names}
    try:
        for name, value in names.items():
            setattr(native, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(native, name, value)


def source0_terminal(*, manifest_path: Path, adapter_path: Path, output: Path) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    manifest = read(manifest_path)
    _routes(manifest)
    adapter = inspect_dora_adapter_payload(
        adapter_path, manifest["model_config"]["model"]["base_model"]
    )
    require(adapter == manifest["source_adapter"], "cold source adapter differs from manifest")
    value = {
        "schema": f"{SCHEMA}.source0_terminal",
        "status": "cold_source_ready",
        "manifest": native.binding(manifest_path),
        "source_adapter": adapter,
        "optimizer_mode": "not_started",
        "updates": 0,
    }
    native.publish(output, value)
    return value


def worker(
    *, manifest_path: Path, terminal_path: Path, adapter_path: Path,
    row_root: Path, arm: str, step: int, source_kind: str, gpu: int,
    batch_size: int, attempt: str, trial_path: Path | None = None,
) -> None:
    require(arm in ARMS and step in STEPS, "readback arm/step")
    require(batch_size in CONFIGS.values(), "qualified native batch size")
    with _native_scope():
        native._run_worker(
            manifest_path=manifest_path, terminal_path=terminal_path,
            adapter_path=adapter_path, row_root=row_root, arm=arm, step=step,
            source_kind=source_kind, gpu=gpu, batch_size=batch_size,
            attempt=attempt, wall_seconds=0, trial_path=trial_path,
        )


def _row_root(output: Path, arm: str, step: int) -> Path:
    return output / arm / f"step-{step:05d}"


def collect_endpoint(
    *, manifest_path: Path, terminal_path: Path, adapter_path: Path,
    arm: str, step: int, output: Path, qualification_result: Path,
    teacher_bank_path: Path, trial_path: Path | None = None,
) -> dict[str, Any]:
    from probes.training_set_completion import coco22_evaluation

    require(arm in ARMS and step in STEPS, "collect arm/step")
    qualification = read(qualification_result)
    require(
        qualification.get("schema") == f"{SCHEMA}.qualification_result"
        and qualification.get("status") == "completed"
        and qualification.get("selected", {}).get("batch_size") in CONFIGS.values(),
        "readback qualification admission",
    )
    require(native.binding(qualification["manifest"]["path"]) == qualification["manifest"],
            "qualification manifest binding")
    qual_manifest = read(qualification["manifest"]["path"])
    manifest = read(manifest_path)
    require(
        qualification.get("arm") == arm
        and qual_manifest["routes"] == manifest["routes"]
        and qual_manifest["source_adapter"] == manifest["source_adapter"],
        "qualification and endpoint frozen arm/teacher/source",
    )
    batch_size = qualification["selected"]["batch_size"]
    routes = _routes(manifest)
    adapter = native._inspect_adapter(adapter_path, manifest)
    _checkpoint_from_terminal(read(terminal_path), step=step, adapter=adapter)
    tokenizer = native._tokenizer(manifest)
    root = _row_root(output, arm, step)
    manifest_binding = native.binding(manifest_path)
    terminal_binding = native.binding(terminal_path)
    trial_binding = native.binding(trial_path) if trial_path else None
    source_kind = "cold_source_step0" if step == 0 else "scientific_checkpoint_readback"
    rows = []
    with _native_scope():
        for route in routes:
            path = native._row_path(root, route["image_id"])
            row = read(path)
            native.validate_row(
                row, route=route, arm=arm, step=step, batch_size=batch_size,
                adapter=adapter, training_manifest=manifest_binding,
                training_terminal=terminal_binding, tokenizer=tokenizer,
                source_kind=source_kind, trial=trial_binding,
            )
            rows.append(row)
    value = {
        "schema": f"{SCHEMA}.endpoint", "status": "completed_unscored",
        "arm": arm, "step": step, "source_kind": source_kind,
        "policy": POLICY, "batch_size": batch_size,
        "training_manifest": manifest_binding, "training_terminal": terminal_binding,
        "trial": trial_binding, "checkpoint_adapter": adapter,
        "qualification_result": native.binding(qualification_result),
        "row_bindings": [native.binding(native._row_path(root, route["image_id"])) for route in routes],
        "rows": rows,
    }
    endpoint_path = root / "endpoint.json"
    if endpoint_path.is_file():
        require(read(endpoint_path) == value, "endpoint changed on recovery")
    else:
        native.publish(endpoint_path, value)
    admission = {
        "schema": coco22_evaluation.READBACK_ADMISSION_SCHEMA,
        "status": "admitted_natural_readback", "arm": arm, "step": step,
        "source_kind": source_kind, "rows": native.binding(endpoint_path),
        "teacher_bank": native.binding(teacher_bank_path),
        "training_manifest": manifest_binding, "training_terminal": terminal_binding,
        "adapter": adapter, "trial": trial_binding,
        "qualification_result": native.binding(qualification_result),
        "producer": native.binding(Path(__file__)),
    }
    admission_path = root / "admission.json"
    if admission_path.is_file():
        require(read(admission_path) == admission, "admission changed on recovery")
    else:
        native.publish(admission_path, admission)
    return {"endpoint": value, "admission": admission}


def _qualification_signature(
    row: Mapping[str, Any], route: Mapping[str, Any]
) -> dict[str, Any]:
    """Project every raw output burden and both frozen owner assignments."""
    from probes.training_set_completion import paired_evaluation as paired
    from probes.training_set_completion import readback_selectors as selectors
    from src.eval.native_rows import native_detection_record as native_record

    case = route["case"]
    parsed = native_record(
        row["raw_decode_text"], case,
        {
            "example_id": route["example_id"], "gt": [],
            "image_height": case["image_height"],
            "image_path": case["image_path"], "image_width": case["image_width"],
            "row_id": case["row_id"], "row_index": case["row_index"],
        },
        row["decode_stop_reason"],
    )
    valid, dropped = paired._matchable_rows_with_geometry_debt(parsed)
    raw = paired._raw_debt(dropped)
    targets = [
        paired._target(
            image_id=route["image_id"], owner_id=str(card["owner_id"]),
            bins=card["edited_fields"]["catalog_reference_coord_bins_1000"],
            description=card["edited_fields"]["selected_description"],
            class_status="verified_coco80",
        )
        for card in route["provenance"]["trace"]
    ]
    by_owner = {target["owner_id"]: target for target in targets}
    by_prediction = {prediction["prediction_id"]: prediction for prediction in valid}
    assignment = {}
    for threshold in (0.5, 0.8):
        ledger = paired._ledger_image(targets, valid, threshold=threshold)
        # These assignments enter a hash-bound JSON receipt and its cold replay.
        assignment[str(threshold)] = sorted(
            [
                match["reference_owner_id"],
                paired._class_correct(
                    target=by_owner[match["reference_owner_id"]],
                    prediction=by_prediction[match["prediction_id"]],
                ),
            ]
            for match in ledger["matches"]
        )
    outside = paired._outside_coco80(valid)
    return {
        "parse_status": parsed.get("parse_status"),
        "metric_bearing": parsed.get("metric_bearing"),
        "valid_count": len(valid),
        "raw": raw,
        "duplicate_pair_count": len(selectors.pairwise_iou95(valid)),
        "outside_coco80_count": len(outside),
        "stop": row["decode_stop_reason"],
        "cap_debt": selectors.termination_metrics(
            len(row["generated_token_ids"]), row["generated_token_ids"],
            row["decode_stop_reason"], cap=CAP,
        )["cap_debt"],
        "assignments": assignment,
        "valid": [
            {"description": prediction["description"],
             "coord_bins_1000": prediction["coord_bins_1000"]}
            for prediction in valid
        ],
    }


def strict_batch_consistency(
    reference_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    """Batch qualification is falsified by any hidden output/owner debt."""
    from src.eval.assignment import global_matches

    reference = _qualification_signature(reference_row, route)
    candidate = _qualification_signature(candidate_row, route)
    left = [("*", tuple(row["coord_bins_1000"])) for row in reference["valid"]]
    right = [("*", tuple(row["coord_bins_1000"])) for row in candidate["valid"]]
    matches = global_matches(left, right, 0.95)
    corresponding = (
        len(matches) == len(left) == len(right)
        and all(
            reference["valid"][li]["description"]
            == candidate["valid"][ri]["description"]
            for li, ri, _ in matches
        )
    )
    burdens = (
        reference["parse_status"] == candidate["parse_status"]
        and reference["metric_bearing"] == candidate["metric_bearing"]
        and reference["valid_count"] == candidate["valid_count"]
        and reference["raw"] == candidate["raw"]
        and reference["duplicate_pair_count"] == candidate["duplicate_pair_count"]
        and reference["outside_coco80_count"] == candidate["outside_coco80_count"]
        and reference["stop"] == candidate["stop"]
        and reference["cap_debt"] == candidate["cap_debt"]
    )
    owners = reference["assignments"] == candidate["assignments"]
    return {
        "image_id": int(route["image_id"]),
        "parity": corresponding and burdens and owners,
        "corresponding_prediction_count_iou_0_95": len(matches),
        "corresponding_same_class": corresponding,
        "equal_raw_burdens": burdens,
        "equal_frozen_owner_assignments_at_0_5_and_0_8": owners,
        "reference_valid_count": len(left),
        "candidate_valid_count": len(right),
        "reference_raw": reference["raw"],
        "candidate_raw": candidate["raw"],
        "reference_assignments": reference["assignments"],
        "candidate_assignments": candidate["assignments"],
        "exact_token_identity": (
            reference_row.get("generated_token_ids")
            == candidate_row.get("generated_token_ids")
        ),
    }


def qualification_result(
    *, manifest_path: Path, terminal_path: Path, adapter_path: Path,
    output: Path, attempt: str, arm: str = "S",
) -> dict[str, Any]:
    require(arm in ARMS, "qualification arm")
    manifest = read(manifest_path)
    amendment = read(READBACK_AMENDMENT)
    require(
        amendment.get("schema")
        == "training_set_completion.coco22_readback_qualification_amendment.v1"
        and amendment.get("status")
        == "predeclared_before_coco22_readback_measurement"
        and native.binding(amendment["predecessor"]["path"])
        == amendment["predecessor"]
        and amendment.get("request_bound") == 66,
        "frozen readback parity amendment",
    )
    routes = _routes(manifest)
    adapter = native._inspect_adapter(adapter_path, manifest)
    _checkpoint_from_terminal(read(terminal_path), step=0, adapter=adapter)
    tokenizer = native._tokenizer(manifest)
    rows: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    with _native_scope():
        for name, batch_size in CONFIGS.items():
            root = output / "configs" / name
            terminal = read(root / "attempts" / attempt / "terminal.json")
            require(terminal.get("status") == "completed" and
                    terminal.get("completed_request_count") == IMAGE_COUNT,
                    f"{name} qualification worker")
            timings[name] = float(terminal["generation_seconds"])
            rows[name] = []
            for route in routes:
                row = read(native._row_path(root, route["image_id"]))
                native.validate_row(
                    row, route=route, arm=arm, step=0, batch_size=batch_size,
                    adapter=adapter, training_manifest=native.binding(manifest_path),
                    training_terminal=native.binding(terminal_path),
                    tokenizer=tokenizer, source_kind="qualification_live_source0",
                    trial=None,
                )
                rows[name].append(row)
        comparisons = {}
        for name in ("batch2", "batch3"):
            comparison = [
                strict_batch_consistency(left, right, route)
                for left, right, route in zip(rows["serial"], rows[name], routes, strict=True)
            ]
            comparisons[name] = comparison
    eligible = ["serial"] + [
        name for name in ("batch2", "batch3")
        if all(row["parity"] for row in comparisons[name])
        and timings[name] < timings["serial"]
    ]
    selected = min(eligible, key=lambda name: timings[name])
    value = {
        "schema": f"{SCHEMA}.qualification_result", "status": "completed",
        "manifest": native.binding(manifest_path),
        "source0_terminal": native.binding(terminal_path),
        "predeclared_readback_amendment": native.binding(READBACK_AMENDMENT),
        "arm": arm,
        "source_adapter": adapter, "attempt": attempt,
        "requests": IMAGE_COUNT * len(CONFIGS),
        "generation_seconds": timings, "comparisons_to_live_serial": comparisons,
        "selected": {"name": selected, "batch_size": CONFIGS[selected],
                     "criterion": "fastest detection-parity native batch; serial fallback"},
    }
    result_path = output / "result.json"
    if result_path.is_file():
        require(read(result_path) == value, "qualification result changed")
    else:
        native.publish(result_path, value)
    return value


def _require_named_tmux(session: str) -> None:
    observed = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()
    require(observed == session, f"controller must run in named tmux {session}")


def qualification_controller(
    *, manifest_path: Path, terminal_path: Path, adapter_path: Path,
    output: Path, attempt: str, arm: str = "S",
) -> None:
    require(arm in ARMS, "qualification arm")
    _require_named_tmux(QUALIFICATION_TMUX)
    require(not (output / "controller-terminal.json").exists(), "qualification already terminal")
    jobs = []
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    launch = []
    started = time.monotonic()
    terminal: dict[str, Any] = {
        "schema": f"{SCHEMA}.qualification_controller_terminal",
        "status": "running", "attempt": attempt, "pid": os.getpid(),
        "tmux_session": QUALIFICATION_TMUX,
        "manifest": native.binding(manifest_path),
    }
    try:
        for gpu, (name, batch_size) in enumerate(CONFIGS.items()):
            command = [
                "python", "-m", "probes.training_set_completion.coco22_readback", "worker",
                "--training-manifest", str(manifest_path), "--training-terminal", str(terminal_path),
                "--adapter", str(adapter_path), "--arm", arm, "--step", "0",
                "--output", str(output / "configs" / name), "--gpu", str(gpu),
                "--batch-size", str(batch_size), "--attempt", attempt,
                "--source-kind", "qualification_live_source0",
            ]
            log = output / "logs" / f"{attempt}-{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("x")
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[2],
                stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                     "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"},
                start_new_session=True,
            )
            jobs.append((process, stream, name, log))
            launch.append({"name": name, "pid": process.pid, "gpu": gpu,
                           "batch_size": batch_size, "command": command, "log": str(log)})
            start_process_waiter(
                process,
                events,
                thread_name_prefix="coco22-readback-wait",
            )
        native.publish(output / "launch.json", {
            "schema": f"{SCHEMA}.qualification_launch", "status": "spawned",
            "tmux_session": QUALIFICATION_TMUX, "jobs": launch,
        })
        exits = [events.get() for _ in jobs]
        for _, stream, _, _ in jobs:
            stream.close()
        require(all(row["exit_code"] == 0 for row in exits), "qualification worker failed")
        result = qualification_result(
            manifest_path=manifest_path, terminal_path=terminal_path,
            adapter_path=adapter_path, output=output, attempt=attempt, arm=arm,
        )
        native.publish(output / "exits.json", {
            "schema": f"{SCHEMA}.qualification_exits", "exits": exits,
        })
        terminal.update(status="completed", result=native.binding(output / "result.json"))
        require(result == read(terminal["result"]["path"]), "qualification publication")
    except BaseException as exc:
        terminal.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc())
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        native.publish(output / "controller-terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "source0-terminal", "worker", "collect-endpoint",
        "qualification-controller", "collect-qualification"))
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-terminal", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="S")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--attempt", default="attempt-001")
    parser.add_argument("--source-kind", default="scientific_checkpoint_readback")
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--teacher-bank", type=Path)
    parser.add_argument("--trial", type=Path)
    args = parser.parse_args()
    if args.command == "source0-terminal":
        source0_terminal(manifest_path=args.training_manifest,
                         adapter_path=args.adapter, output=args.output)
    else:
        require(args.training_terminal is not None, "readback training terminal")
        if args.command == "worker":
            require(args.gpu is not None and args.batch_size is not None, "readback worker GPU/batch")
            worker(
                manifest_path=args.training_manifest, terminal_path=args.training_terminal,
                adapter_path=args.adapter, row_root=args.output, arm=args.arm,
                step=args.step, source_kind=args.source_kind, gpu=args.gpu,
                batch_size=args.batch_size, attempt=args.attempt, trial_path=args.trial,
            )
        elif args.command == "collect-endpoint":
            require(args.qualification_result and args.teacher_bank, "endpoint bindings")
            collect_endpoint(
                manifest_path=args.training_manifest, terminal_path=args.training_terminal,
                adapter_path=args.adapter, arm=args.arm, step=args.step,
                output=args.output, qualification_result=args.qualification_result,
                teacher_bank_path=args.teacher_bank, trial_path=args.trial,
            )
        elif args.command == "qualification-controller":
            qualification_controller(
                manifest_path=args.training_manifest, terminal_path=args.training_terminal,
                adapter_path=args.adapter, output=args.output, attempt=args.attempt,
                arm=args.arm,
            )
        else:
            qualification_result(
                manifest_path=args.training_manifest, terminal_path=args.training_terminal,
                adapter_path=args.adapter, output=args.output, attempt=args.attempt,
                arm=args.arm,
            )


if __name__ == "__main__":
    main()
