"""Freeze the lead-approved Lane A production admission (CPU-only).

This command reads already admitted cases and the accepted qualification launch.
It does not load a model, call a processor, resample data, or write source data.
The resulting JSON is an immutable execution contract for the parent driver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-address-readout-pilot"
)
ADMISSION = DEFAULT_ROOT / "selection-v3/selection/manifest.json"
LAUNCH = DEFAULT_ROOT / "qualification/launch-v2.json"
FEEDBACK_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-numerical-recurrence-feedback/panel.json"
)
MATURE_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-untied-highconfidence18-natural/panel.json"
)
DEV_SOURCE_IDS = (632, 885, 5586, 7511, 14038, 417044)
FEEDBACK_IDS = {632, 885, 5586}
MATURE_IDS = {7511, 14038, 417044}
# Names are the persisted fit-directory/checkpoint names used by production.py.
CONDITIONS = ["original", "aligned-1729", "permuted-1729", "aligned-2718", "permuted-2718"]
SEEDS = (1729, 2718)
UPDATES = 256
BATCH_SIZE = 8
EPOCHS = 16
NATIVE_CAP = 3084
CALIBRATION_CAP = 8
SCHEMA = "address_readout_pilot.production.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file(), f"expected file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def immutable_write(path: Path, value: Any) -> None:
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"immutable collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def selection_cases(admission: Mapping[str, Any], cohort: str) -> list[dict[str, Any]]:
    path = Path(admission["cohorts"][cohort]["cases_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == int(admission["cohorts"][cohort]["count"]), f"{cohort} count mismatch")
    return rows


def panel_case(panel: Mapping[str, Any], image_id: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
    hits: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for group in panel.get("groups", []):
        for case in group.get("cases", []):
            if int(case.get("input_record", {}).get("image_id", -1)) == image_id:
                hits.append((case, group))
    require(len(hits) == 1, f"development image {image_id} has {len(hits)} panel cases")
    return hits[0]


def dev_case(case: Mapping[str, Any], group: Mapping[str, Any], panel_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(case))
    image_id = int(result["input_record"]["image_id"])
    require(result["row_id"] == f"coco2017_{result['input_record']['metadata']['split']}_{image_id:012d}", f"row identity mismatch for {image_id}")
    result["cohort"] = "development_recurrence"
    result["config_key"] = "dev_feedback_tied" if image_id in FEEDBACK_IDS else "dev_mature_tied"
    result["source_provenance"] = {
        "panel": file_binding(panel_path),
        "panel_group": group["key"],
        "panel_group_input_jsonl": str(Path(group.get("input_jsonl", config["data"]["input_jsonl"])).resolve(strict=True)),
        "historical_pool": "six-image/11-source-trajectory recurrence pool; development evidence only",
        "executable_model": "tied step-2444",
        "annotation_semantics": "canonical existing positive boxes from retained source case; missing annotations are not negatives",
    }
    image_path = Path(result["image_path"]).resolve(strict=True)
    result["processed_image_path"] = str(image_path)
    result["processed_image_binding"] = file_binding(image_path)
    result["original_image_path"] = str(image_path)
    result["original_image_binding"] = file_binding(image_path)
    result["original_width"] = int(result["image_width"])
    result["original_height"] = int(result["image_height"])
    plan = result["image_plan"]
    require(len(plan.get("observed_image_grid_thw", [])) == 3, f"missing development grid {image_id}")
    require(int(plan["observed_image_grid_thw"][0]) == 1, f"unexpected temporal grid {image_id}")
    require(int(plan["observed_image_grid_thw"][1]) % 2 == 0 and int(plan["observed_image_grid_thw"][2]) % 2 == 0, f"non-merged development grid {image_id}")
    result["address_grid_key"] = f"{int(plan['observed_image_grid_thw'][1]) // 2}x{int(plan['observed_image_grid_thw'][2]) // 2}"
    return result


def paired_schedules(row_ids: list[str]) -> dict[str, list[list[str]]]:
    ordered = sorted(row_ids)
    require(len(ordered) == 128 and len(set(ordered)) == 128, "training IDs must be 128 unique sorted rows")
    schedules: dict[str, list[list[str]]] = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        batches: list[list[str]] = []
        for _ in range(EPOCHS):
            epoch = list(ordered)
            rng.shuffle(epoch)
            batches.extend(epoch[i : i + BATCH_SIZE] for i in range(0, len(epoch), BATCH_SIZE))
        require(len(batches) == UPDATES and all(len(batch) == BATCH_SIZE for batch in batches), f"schedule shape for seed {seed}")
        require(sorted(x for batch in batches for x in batch) == sorted(ordered * EPOCHS), f"schedule coverage for seed {seed}")
        schedules[str(seed)] = batches
    return schedules


def fresh_native_order(dev: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The ruling says sorted image IDs.  Row-ID lexical order would put the
    # train-split 417044 before the val-split development images.
    dev = sorted(dev, key=lambda x: int(x["input_record"]["image_id"]))
    fresh = sorted(fresh, key=lambda x: int(x["input_record"]["image_id"]))
    result: list[dict[str, Any]] = []
    di = fi = 0
    while di < len(dev) or fi < len(fresh):
        if di < len(dev):
            result.append(dev[di])
            di += 1
        for _ in range(6):
            if fi == len(fresh):
                break
            result.append(fresh[fi])
            fi += 1
    return result


def prepare(root: Path, admission_path: Path, launch_path: Path) -> dict[str, Any]:
    admission = load(admission_path)
    launch = load(launch_path)
    require(admission_path.resolve() == (root / "selection-v3/selection/manifest.json").resolve(), "production must use selection-v3")
    require(sha256_bytes(admission_path.read_bytes()) == "b9a4f088466c1874ffef15380ff4fab7931b752b4208cbb2abba0de9836834d7", "selection-v3 SHA changed")
    require(launch.get("schema") == "address_readout_pilot.qualification_launch.v1", "unexpected qualification launch schema")
    require(launch.get("status") == "frozen_for_qualification_only", "launch binding changed")
    require(admission.get("status") == "candidate_ready_for_lead_review", "admission status changed")

    train = selection_cases(admission, "train")
    calibration = selection_cases(admission, "calibration")
    fresh = selection_cases(admission, "evaluation")
    require(len(train) == 128 and len(calibration) == 32 and len(fresh) == 32, "selection-v3 denominator changed")
    train_ids = {x["row_id"] for x in train}
    calibration_ids = {x["row_id"] for x in calibration}
    fresh_ids = {x["row_id"] for x in fresh}
    require(train_ids.isdisjoint(calibration_ids | fresh_ids) and calibration_ids.isdisjoint(fresh_ids), "selection cohorts overlap")

    feedback = load(FEEDBACK_PANEL)
    mature = load(MATURE_PANEL)
    dev: list[dict[str, Any]] = []
    dev_configs: dict[str, dict[str, Any]] = {}
    for image_id in DEV_SOURCE_IDS:
        panel = feedback if image_id in FEEDBACK_IDS else mature
        panel_path = FEEDBACK_PANEL if image_id in FEEDBACK_IDS else MATURE_PANEL
        case, group = panel_case(panel, image_id)
        cfg = copy.deepcopy(panel["configs"]["tied"])
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["input_jsonl"] = str(Path(group["input_jsonl"]).resolve(strict=True))
        key = "dev_feedback_tied" if image_id in FEEDBACK_IDS else "dev_mature_tied"
        dev_configs[key] = cfg
        dev.append(dev_case(case, group, panel_path, cfg))
    require({int(x["input_record"]["image_id"]) for x in dev} == set(DEV_SOURCE_IDS), "development pool IDs changed")
    require(not ({int(x["input_record"]["image_id"]) for x in dev} & {int(x["input_record"]["image_id"]) for x in train + calibration + fresh}), "development overlaps pilot splits")

    configs = {
        "train": copy.deepcopy(launch["model_config"]),
        "calibration": copy.deepcopy(launch["calibration_config"]),
        "native_fresh": copy.deepcopy(launch["calibration_config"]),
        **dev_configs,
    }
    configs["native_fresh"]["generation"] = dict(configs["native_fresh"]["generation"])
    configs["native_fresh"]["generation"]["max_new_tokens"] = NATIVE_CAP
    for key in ("train", "calibration"):
        require(configs[key]["model"]["processor"]["do_resize"] is False, f"{key} processor resize changed")
    configs["native_fresh"]["data"] = dict(configs["native_fresh"]["data"])
    configs["native_fresh"]["data"]["input_jsonl"] = str(Path(admission["configs"]["evaluation"]["data"]["input_jsonl"]).resolve(strict=True))
    # Historical panels carry their old writer roots.  They are source
    # provenance only; every execution config points at this pilot's output.
    for config in configs.values():
        config["run"] = dict(config.get("run", {}))
        config["run"].update({"artifact_root": str((root / "production").resolve()), "name": "address-readout-pilot-production", "output_dir": None})

    training_cases = [dict(x, cohort="training", config_key="train", model_config=copy.deepcopy(configs["train"])) for x in sorted(train, key=lambda x: x["row_id"])]
    calibration_cases = [dict(x, cohort="calibration", config_key="calibration", model_config=copy.deepcopy(configs["calibration"])) for x in sorted(calibration, key=lambda x: x["row_id"])]
    fresh_cases = [dict(x, cohort="fresh_natural_evaluation", config_key="native_fresh", model_config=copy.deepcopy(configs["native_fresh"])) for x in fresh]
    for case in dev:
        case["model_config"] = copy.deepcopy(configs[case["config_key"]])
    for case in fresh_cases:
        require(case["row_id"] not in {x["row_id"] for x in dev}, "fresh evaluation overlaps development")
    native_cases = fresh_native_order(dev, fresh_cases)
    require(len(native_cases) == 38 and len({x["row_id"] for x in native_cases}) == 38, "native denominator/order changed")
    require([int(x["input_record"]["image_id"]) for x in native_cases[:7]] == [632, 11051, 11699, 45728, 57672, 129492, 137246], "native interleave order changed")

    referents = {str(x["row_id"]): str(x["referent_description"]) for x in admission["diagnostic_cases"]}
    require(set(referents) == calibration_ids, "calibration referent set changed")
    require(all("<|coord_" not in text for text in referents.values()), "coordinates leaked into referents")

    permutations = copy.deepcopy(launch["address_permutations"])
    grid_keys = sorted({x["address_grid_key"] for x in dev})
    grid_bindings = {}
    for key in grid_keys:
        require(key in permutations, f"qualification lacks roll-1 permutation for development grid {key}")
        perm = permutations[key]
        grid_bindings[key] = {
            "merged_grid": [int(key.split("x")[0]), int(key.split("x")[1])],
            "token_count": len(perm),
            "permutation": perm,
            "source": "qualification/launch-v2.json address_permutations; exact roll-1 mapping bound before outcomes",
        }

    source_paths = [admission_path, launch_path, FEEDBACK_PANEL, MATURE_PANEL,
                    Path(configs["dev_feedback_tied"]["data"]["input_jsonl"]),
                    Path(configs["dev_mature_tied"]["data"]["input_jsonl"]),
                    Path(configs["train"]["data"]["input_jsonl"]),
                    Path(configs["calibration"]["data"]["input_jsonl"]),
                    Path(configs["native_fresh"]["data"]["input_jsonl"]),
                    Path(__file__), Path(__file__).resolve().parents[3] / 'probes/model_profiles/mature_tied_untied.py',
                    Path("src/inference/bound_requests.py"), Path("src/inference/prompt.py"),
                    Path("src/data/examples.py"), Path(__file__).with_name("production.py"),
                    Path(__file__).with_name("runtime.py"), Path(__file__).with_name("training.py")]
    for entry in launch.get("input_bindings", []):
        source_paths.append(Path(entry["path"]))
    source_bindings = []
    seen = set()
    for path in source_paths:
        path = path.resolve(strict=True)
        if str(path) not in seen:
            source_bindings.append(file_binding(path))
            seen.add(str(path))
    for case in dev + fresh_cases:
        source_bindings.append(case["processed_image_binding"])

    manifest = {
        "schema": SCHEMA,
        "status": "candidate_ready_for_parent_driver",
        "model_calls": 0,
        "no_resampling": True,
        "admission": file_binding(admission_path),
        "qualification_launch": file_binding(launch_path),
        "admission_sha256_required": "b9a4f088466c1874ffef15380ff4fab7931b752b4208cbb2abba0de9836834d7",
        "model_config": copy.deepcopy(configs["train"]),
        "calibration_config": copy.deepcopy(configs["calibration"]),
        "configs": configs,
        "training_cases": training_cases,
        "calibration_cases": calibration_cases,
        "native_cases": native_cases,
        "schedules": paired_schedules([x["row_id"] for x in training_cases]),
        "conditions": CONDITIONS,
        "calibration_referents": referents,
        "training": {"seeds": list(SEEDS), "updates": UPDATES, "images_per_update": BATCH_SIZE, "epochs": EPOCHS, "optimizer": {"type": "AdamW", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}, "numeric_mode": "FP32/SDPA; TF32 disabled", "checkpoint_rule": "fixed final update 256; no best-panel selection", "paired_arms": "aligned and permuted share seed, schedule, initialization and optimizer state"},
        "calibration": {"conditions": CONDITIONS, "max_new_tokens": CALIBRATION_CAP, "referent_mode": "naturally unique class, description supplied without coordinates; freely generate all four coordinates", "recurrence_prefix": False},
        "native_evaluation": {"conditions": CONDITIONS, "max_new_tokens": NATIVE_CAP, "decode": {"empty_prefix": True, "greedy": True, "sampling": False, "repetition_penalty": 1.0, "temperature": 0.0, "top_p": 1.0}, "order": "one sorted development image followed by up to six sorted fresh images, continuing until both lists are exhausted", "development_ids": list(DEV_SOURCE_IDS), "fresh_count": len(fresh_cases), "count": len(native_cases)},
        "evaluation_assignments": {"order": "native_cases concatenated with calibration_cases", "workers": 8, "assignment": "global concatenated index modulo workers", "by_worker": {str(worker): [index for index in range(len(native_cases) + len(calibration_cases)) if index % 8 == worker] for worker in range(8)}},
        "budget": {"clock_origin_unix": 1790049699.020824, "model_wall_seconds_limit": 14400, "allocated_gpu_seconds_limit": 115200, "reserve_wall_seconds": 120, "reserve_gpu_seconds": 960, "devices": list(range(8))},
        "development_grid_bindings": grid_bindings,
        "address_permutations": permutations,
        "source_bindings": source_bindings,
        "provenance": {"development_pool": "six exact recurrence images with 11 tied/untied source trajectories from retained failure research; only tied executable", "refined5_guard": "417044 retained from mature refined-03 source case; no Refined5 replacement or expansion", "labels": "canonical existing positive boxes and geo_sorted_xy rows retained verbatim"},
        "checks": {"train_count": len(training_cases), "calibration_count": len(calibration_cases), "native_count": len(native_cases), "development_count": len(dev), "fresh_native_count": len(fresh_cases), "schedule_updates": {k: len(v) for k, v in paired_schedules([x["row_id"] for x in training_cases]).items()}, "paired_schedule_equal_shape": True, "address_roll1_grids_bound": grid_keys, "coordinates_absent_from_calibration_referents": True},
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--admission", type=Path, default=ADMISSION)
    parser.add_argument("--launch", type=Path, default=LAUNCH)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.output_root.resolve(), args.admission.resolve(), args.launch.resolve())
    target = (args.manifest_path or (args.output_root.resolve() / "production/manifest.json")).resolve()
    if not args.check_only:
        immutable_write(target, manifest)
    print(json.dumps({"manifest": str(target), "status": manifest["status"], "counts": {"train": len(manifest["training_cases"]), "calibration": len(manifest["calibration_cases"]), "native": len(manifest["native_cases"])}}, sort_keys=True))


if __name__ == "__main__":
    main()
