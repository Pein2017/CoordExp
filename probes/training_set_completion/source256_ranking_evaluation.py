"""Natural batch-four readback and saved-row reduction for the P/R repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import source256_evaluation as old_evaluation
from probes.training_set_completion import source256_normalized_readback as normalized_readback
from probes.training_set_completion import source256_ranking_data as ranking_data
from probes.training_set_completion import source256_readback as readback
from probes.training_set_completion import training


SCHEMA = f"{ranking_data.SCHEMA}.evaluation"
PLAN = ranking_data.OLD / "runtime/main-v1/readback-plan.json"
FIXED_RESULT = ranking_data.OLD / "runtime/main-v1/evaluation/result.json"
NORMALIZED_RESULT = ranking_data.NORMALIZED / "runtime/main-normalized-v1/evaluation/result.json"
FIXED_SHA256 = "1c45305af5c3b7689e75465092773437005a312632946ad2a83e706e48a8083d"
NORMALIZED_SHA256 = "d4674776e5ae509923462148028f428dda3b28c12af61789c82919827fb2cfa4"
QUALIFICATION_IMAGE_IDS = (548337, 536467, 158044, 203986)
DEBT_KEYS = (
    "malformed_row_count", "invalid_geometry_count", "annotation_unmatched_prediction_count",
    "strict_repeat_row_count", "cap_debt", "eos_debt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _bind(path: str | Path) -> dict[str, Any]:
    return training.binding(Path(path).resolve(strict=True))


def _manifest(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = path.resolve(strict=True)
    manifest = read(path)
    require(
        manifest.get("schema") == f"{ranking_data.SCHEMA}.manifest"
        and manifest.get("arm") in {"P", "R"}
        and manifest.get("mode") in {"main", "qualification"},
        "ranking manifest identity",
    )
    data = ranking_data.load_data(manifest)
    require(manifest["preparation"] == data["sources"]["preparation"], "manifest/data preparation")
    readback._model_config(manifest)
    return path, dict(manifest), data


def _plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    checked = readback.validate_plan(read(PLAN))
    require(manifest["preparation"] == checked["plan"]["preparation"], "old cohort binding")
    return checked


def _terminal(
    manifest_path: Path, manifest: Mapping[str, Any], terminal_path: Path
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    terminal_path = terminal_path.resolve(strict=True)
    terminal = read(terminal_path)
    step = int(manifest["runtime"]["updates"])
    require(
        terminal.get("status") == "completed"
        and terminal.get("manifest") == _bind(manifest_path)
        and terminal.get("arm") == manifest["arm"]
        and terminal.get("updates") == step,
        "completed terminal identity",
    )
    checkpoints = [item for item in terminal.get("checkpoints", []) if item.get("step") == step]
    require(len(checkpoints) == 1 and isinstance(checkpoints[0].get("adapter"), Mapping), "endpoint checkpoint")
    adapter = dict(checkpoints[0]["adapter"])
    adapter_path = Path(adapter.get("root", "")).resolve(strict=True)
    require(
        training.inspect_dora_adapter_payload(
            adapter_path, manifest["model_config"]["model"]["base_model"]
        ) == adapter,
        "checkpoint adapter bytes",
    )
    return adapter_path, adapter, terminal_path, dict(terminal)


def _items(
    plan: Mapping[str, Any], *, split: str | None, shard: int | None, qualification: bool
) -> tuple[str, int | None, list[dict[str, Any]]]:
    if qualification:
        require(split in (None, "train") and shard is None, "qualification requires train with no shard")
        items = readback._train_items(plan["prepared"], list(QUALIFICATION_IMAGE_IDS))
        require(len(items) == 4, "one frozen qualification batch")
        return "train", None, items
    require(split in readback.SPLIT_COUNTS and isinstance(shard, int), "endpoint split/shard")
    require(0 <= shard < readback.ENDPOINT_SHARDS, "endpoint shard range")
    ids = plan["plan"]["endpoint_shards"][split][shard]
    items = readback._train_items(plan["prepared"], ids) if split == "train" else readback._dev_items(plan["prepared"], ids)
    require(len(items) % 4 == 0, "full batch-four endpoint")
    return str(split), shard, items


def _validate_generation(*, generation: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> None:
    """Reuse the prior exact row-token/prompt/media consumer contract."""

    normalized_readback._validate_generation_identity(generation=generation, items=items)


def _shard_value(
    *, manifest_path: Path, manifest: Mapping[str, Any], terminal_path: Path,
    adapter: Mapping[str, Any], split: str, shard: int | None, qualification: bool,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}.shard", "status": "completed_unscored",
        "endpoint": {"arm": manifest["arm"], "step": manifest["runtime"]["updates"]},
        "qualification": qualification, "split": split, "shard": shard,
        "qualification_image_ids": list(QUALIFICATION_IMAGE_IDS) if qualification else None,
        "batch_size": 4, "policy": dict(readback.POLICY), "plan": _bind(PLAN),
        "training_manifest": _bind(manifest_path), "training_terminal": _bind(terminal_path),
        "checkpoint_adapter": dict(adapter), "generation": dict(generation),
    }


def _validate_shard(
    value: Mapping[str, Any], *, manifest_path: Path, manifest: Mapping[str, Any],
    terminal_path: Path, adapter: Mapping[str, Any], plan: Mapping[str, Any],
    split: str | None, shard: int | None, qualification: bool,
) -> dict[str, Any]:
    split, shard, items = _items(plan, split=split, shard=shard, qualification=qualification)
    expected = _shard_value(
        manifest_path=manifest_path, manifest=manifest, terminal_path=terminal_path,
        adapter=adapter, split=split, shard=shard, qualification=qualification,
        generation=value.get("generation", {}),
    )
    for key in expected:
        if key != "generation":
            require(value.get(key) == expected[key], f"shard {key} identity")
    _validate_generation(generation=value["generation"], items=items)
    return dict(value)


def validate_shard(
    *, value: Mapping[str, Any], manifest_path: Path, terminal_path: Path,
    split: str | None = None, shard: int | None = None, qualification: bool = False,
) -> dict[str, Any]:
    manifest_path, manifest, _ = _manifest(manifest_path)
    plan = _plan(manifest)
    _, adapter, terminal_path, _ = _terminal(manifest_path, manifest, terminal_path)
    return _validate_shard(
        value, manifest_path=manifest_path, manifest=manifest, terminal_path=terminal_path,
        adapter=adapter, plan=plan, split=split, shard=shard, qualification=qualification,
    )


def worker(
    *, manifest_path: Path, terminal_path: Path, output: Path, split: str | None = None,
    shard: int | None = None, qualification: bool = False, device: str = "cuda:0",
) -> dict[str, Any]:
    """Generate one recoverable P16/R16 shard or the frozen R step1 consumer."""

    require(not output.is_symlink(), "shard output symlink")
    output = output.resolve()
    if output.exists():
        return validate_shard(
            value=read(output), manifest_path=manifest_path, terminal_path=terminal_path,
            split=split, shard=shard, qualification=qualification,
        )
    manifest_path, manifest, _ = _manifest(manifest_path)
    if qualification:
        require(
            manifest["arm"] == "R" and manifest["mode"] == "qualification"
            and manifest["runtime"]["updates"] == 1,
            "only R qualification step1",
        )
    else:
        require(manifest["mode"] == "main" and manifest["runtime"]["updates"] == 16, "only P16/R16")
    plan = _plan(manifest)
    split, shard, items = _items(plan, split=split, shard=shard, qualification=qualification)
    adapter_path, adapter, terminal_path, _ = _terminal(manifest_path, manifest, terminal_path)
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA natural readback")
    config_data = readback._model_config(manifest)
    config = checkpoint_config(InferConfig.model_validate(config_data), str(adapter_path))
    torch.cuda.set_device(device)
    qwen, loaded = load_policy(config, device=torch.device(device))
    readback._loaded_identity(loaded, adapter_path=adapter_path)
    generation = readback._generate(qwen, config, config_data, items, batch_size=4, formal=True)
    value = _shard_value(
        manifest_path=manifest_path, manifest=manifest, terminal_path=terminal_path, adapter=adapter,
        split=split, shard=shard, qualification=qualification, generation=generation,
    )
    _validate_shard(
        value, manifest_path=manifest_path, manifest=manifest, terminal_path=terminal_path,
        adapter=adapter, plan=plan, split=split, shard=shard, qualification=qualification,
    )
    training.publish(output, value)
    return value


def _admit(
    *, manifest_path: Path, terminal_path: Path, root: Path, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path, manifest, data = _manifest(manifest_path)
    plan = _plan(manifest)
    _, adapter, terminal_path, _ = _terminal(manifest_path, manifest, terminal_path)
    expected = [image_id for group in plan["plan"]["endpoint_shards"][split] for image_id in group]
    rows = []
    for shard in range(8):
        path = (root / split / f"shard-{shard:02d}.json").resolve(strict=True)
        value = _validate_shard(
            read(path), manifest_path=manifest_path, manifest=manifest, terminal_path=terminal_path,
            adapter=adapter, plan=plan, split=split, shard=shard, qualification=False,
        )
        rows.extend(dict(row) for row in value["generation"]["rows"])
    by_image = {int(row["image_id"]): row for row in rows}
    require(len(by_image) == len(expected) and set(by_image) == set(expected), "complete saved endpoint cohort")
    return [by_image[image_id] for image_id in expected], manifest, data


def _control(path: Path, expected_sha256: str, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _bind(path)
    require(binding["sha256"] == expected_sha256, f"{name} SHA")
    value = read(path)
    require(value.get("status") == "completed_saved_readback_evaluation", f"{name} status")
    return value, binding


def _score(
    *, split: str, rows: Sequence[Mapping[str, Any]], targets: Mapping[int, Any],
    contexts: Mapping[int, Any], tokenizer: Any,
) -> dict[str, Any]:
    return old_evaluation._score_split(
        split=split, rows=rows, targets=targets, contexts=contexts, tokenizer=tokenizer
    )


def _subset(
    rows: Sequence[Mapping[str, Any]], image_ids: Sequence[int], *, targets: Mapping[int, Any],
    contexts: Mapping[int, Any], tokenizer: Any,
) -> dict[str, Any]:
    by_image = {int(row["image_id"]): row for row in rows}
    require(set(image_ids) <= set(by_image), "selected saved rows")
    return _score(
        split="train", rows=[by_image[image_id] for image_id in image_ids],
        targets=targets, contexts=contexts, tokenizer=tokenizer,
    )


def _selection(data: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    selected = sorted(map(int, data["pairs"]))
    all_ids = sorted(map(int, data["canonical_routes"]))
    require(len(selected) == 15 and len(set(selected)) == 15, "selected15")
    outside = sorted(set(all_ids) - set(selected))
    require(len(all_ids) == 256 and len(outside) == 241, "train256/outside241")
    return selected, outside


def _coverage(score: Mapping[str, Any], split: str) -> set[tuple[int, str]]:
    return {
        (int(row["image_id"]), str(owner)) for row in score["splits"][split]["per_image"]
        for owner in row["primary_class_agnostic_iou50"]["covered_owner_ids"]
    }


def _debt(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    by_split = {
        split: {key: int(candidate["splits"][split]["burden"].get(key, 0)) <= int(reference["splits"][split]["burden"].get(key, 0)) for key in DEBT_KEYS}
        for split in ("train", "dev")
    }
    return {"by_split": by_split, "passed": all(all(values.values()) for values in by_split.values())}


def _starting_source_new(
    *, baseline: Mapping[str, Any], endpoint: Mapping[str, Any], label: str
) -> dict[str, Any]:
    source, start = _coverage(baseline, "train"), _coverage(endpoint, "train") - _coverage(baseline, "train")
    require(len(start) == 91, "ACTUAL91 starting Source-new identities")
    return {"source_coverage": source, "starting": start, "label": label}


def _survival(
    *, start: set[tuple[int, str]], source: set[tuple[int, str]], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    final = _coverage(endpoint, "train")
    survived, lost, replacements = start & final, start - final, (final - source) - start
    rows = lambda values: [{"image_id": image_id, "owner_id": owner_id} for image_id, owner_id in sorted(values)]
    return {
        "starting_count": len(start), "survived_count": len(survived), "lost_count": len(lost),
        "replacement_gain_count": len(replacements), "final_gain_count_vs_Source0": len(final - source),
        "survived": rows(survived), "lost": rows(lost), "replacement_gains": rows(replacements),
    }


def _gates(
    *, p: Mapping[str, Any], r: Mapping[str, Any], source: Mapping[str, Any], a64: Mapping[str, Any],
    p_outside: Mapping[str, Any], r_outside: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons = {
        "P16_vs_Source0": old_evaluation.compare_scores(source, p, label="P16_vs_Source0"),
        "R16_vs_Source0": old_evaluation.compare_scores(source, r, label="R16_vs_Source0"),
    }
    r_source = comparisons["R16_vs_Source0"]["splits"]
    r_debt = _debt(r, p)
    repair = {
        "R_train_fn_lt_P": r["splits"]["train"]["primary_class_agnostic_iou50"]["missing_count"] < p["splits"]["train"]["primary_class_agnostic_iou50"]["missing_count"],
        "R_train_fn_lt_652": r["splits"]["train"]["primary_class_agnostic_iou50"]["missing_count"] < 652,
        "R_source_old_loss_lt_55": r_source["train"]["primary_class_agnostic_iou50"]["lost_count"] < 55,
        "R_debt_not_worse_than_P": r_debt["passed"],
        "R_dev_fn_lte_P": r["splits"]["dev"]["primary_class_agnostic_iou50"]["missing_count"] <= p["splits"]["dev"]["primary_class_agnostic_iou50"]["missing_count"],
        "R_dev_fn_lte_276": r["splits"]["dev"]["primary_class_agnostic_iou50"]["missing_count"] <= 276,
        "R_outside_reference_coverage_gte_P": r_outside["primary_class_agnostic_iou50"]["coverage"] >= p_outside["primary_class_agnostic_iou50"]["coverage"],
    }
    promotion = {}
    for label, score, comparison in (("P16", p, comparisons["P16_vs_Source0"]), ("R16", r, comparisons["R16_vs_Source0"])):
        debt = _debt(score, a64)
        old = comparison["splits"]
        checks = {
            "train_fn_lt_628": score["splits"]["train"]["primary_class_agnostic_iou50"]["missing_count"] < 628,
            "train_source_old_loss_lte_35": old["train"]["primary_class_agnostic_iou50"]["lost_count"] <= 35,
            "dev_fn_lte_271": score["splits"]["dev"]["primary_class_agnostic_iou50"]["missing_count"] <= 271,
            "dev_source_old_loss_lte_26": old["dev"]["primary_class_agnostic_iou50"]["lost_count"] <= 26,
            "debt_not_worse_than_A64": debt["passed"],
        }
        promotion[label] = {"checks": checks, "passed": all(checks.values()), "debt": debt}
    return {"repair_R_over_P": {"checks": repair, "passed": all(repair.values()), "debt": r_debt}, "promotion": promotion, "gain_count_diagnostic": {"R_final_gain_count": r_source["train"]["primary_class_agnostic_iou50"]["gained_count"], "R_original_gain_count_gte_91": r_source["train"]["primary_class_agnostic_iou50"]["gained_count"] >= 91, "identity_survival_is_separate": True}}, comparisons


def _likelihood(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"binding": _bind(path), "value": read(path)}


def reduce(
    *, p_manifest_path: Path, p_terminal_path: Path, p_readback_root: Path,
    r_manifest_path: Path, r_terminal_path: Path, r_readback_root: Path, output: Path,
    fixed_result_path: Path = FIXED_RESULT, normalized_result_path: Path = NORMALIZED_RESULT,
    p_likelihood_path: Path | None = None, r_likelihood_path: Path | None = None,
) -> dict[str, Any]:
    """Score P16/R16 saved rows; no generation, packet, or release action occurs."""

    fixed, fixed_binding = _control(fixed_result_path, FIXED_SHA256, "fixed control")
    normalized, normalized_binding = _control(normalized_result_path, NORMALIZED_SHA256, "normalized control")
    p_manifest_path, p_manifest, data = _manifest(p_manifest_path)
    r_manifest_path, r_manifest, r_data = _manifest(r_manifest_path)
    require(
        p_manifest["arm"] == "P" and r_manifest["arm"] == "R"
        and p_manifest["mode"] == r_manifest["mode"] == "main"
        and p_manifest["runtime"]["updates"] == r_manifest["runtime"]["updates"] == 16
        and p_manifest["data"] == r_manifest["data"],
        "paired P16/R16 manifests",
    )
    selected, outside = _selection(data)
    require(_selection(r_data) == (selected, outside), "paired selected15")
    plan = _plan(p_manifest)
    prepared = plan["prepared"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(prepared["preparation"]["identity"]["runtime_contract"]["tokenizer_path"]).parent),
        local_files_only=True,
    )
    targets, contexts = old_evaluation._targets(prepared), old_evaluation._contexts(prepared)
    endpoint_rows: dict[str, dict[str, list[dict[str, Any]]]] = {"P16": {}, "R16": {}}
    for split in ("train", "dev"):
        endpoint_rows["P16"][split], _, _ = _admit(manifest_path=p_manifest_path, terminal_path=p_terminal_path, root=p_readback_root, split=split)
        endpoint_rows["R16"][split], _, _ = _admit(manifest_path=r_manifest_path, terminal_path=r_terminal_path, root=r_readback_root, split=split)
    def endpoint(label: str, arm: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": f"{old_evaluation.SCHEMA}.endpoint_score", "status": "scored_saved_natural_readback",
            "endpoint": {"label": label, "arm": arm, "step": 16}, "preparation": manifest["preparation"],
            "splits": {split: _score(split=split, rows=endpoint_rows[label][split], targets=targets[split], contexts=contexts[split], tokenizer=tokenizer) for split in ("train", "dev")},
        }
    p, r = endpoint("P16", "P", p_manifest), endpoint("R16", "R", r_manifest)
    controls = {"Source0": fixed["scores"]["Source0"], "A64": fixed["scores"]["A64"], "B64": fixed["scores"]["B64"], "Bnormalized64": normalized["new_scores"]["Bnormalized64"]}
    require(all(score.get("preparation") == p_manifest["preparation"] for score in controls.values()), "control bank")
    slices = {
        label: {
            "selected15_train": _subset(rows["train"], selected, targets=targets["train"], contexts=contexts["train"], tokenizer=tokenizer),
            "selected14_excluding_548337_train": _subset(rows["train"], [image_id for image_id in selected if image_id != 548337], targets=targets["train"], contexts=contexts["train"], tokenizer=tokenizer),
            "outside_reference_train241": _subset(rows["train"], outside, targets=targets["train"], contexts=contexts["train"], tokenizer=tokenizer),
            "dev128": score["splits"]["dev"],
        }
        for label, rows, score in (("P16", endpoint_rows["P16"], p), ("R16", endpoint_rows["R16"], r))
    }
    gates, comparisons = _gates(p=p, r=r, source=controls["Source0"], a64=controls["A64"], p_outside=slices["P16"]["outside_reference_train241"], r_outside=slices["R16"]["outside_reference_train241"])
    for endpoint_label, score in (("P16", p), ("R16", r)):
        comparisons.update({f"{endpoint_label}_vs_{label}": old_evaluation.compare_scores(control, score, label=f"{endpoint_label}_vs_{label}") for label, control in controls.items()})
    start = _starting_source_new(baseline=controls["Source0"], endpoint=controls["Bnormalized64"], label="Bnormalized64_vs_Source0")
    recorded_start = {
        (int(row["image_id"]), str(row["owner_id"]))
        for row in normalized["comparisons"]["Bnormalized64_vs_Source0"]["splits"]["train"]["primary_class_agnostic_iou50"]["gained"]
    }
    require(recorded_start == start["starting"], "accepted ACTUAL91 identity set")
    value = {
        "schema": f"{SCHEMA}.result", "status": "completed_saved_readback_evaluation",
        "manifests": {"P": _bind(p_manifest_path), "R": _bind(r_manifest_path)}, "terminals": {"P": _bind(p_terminal_path), "R": _bind(r_terminal_path)},
        "controls": {"fixed_result": fixed_binding, "normalized_result": normalized_binding, "labels": list(controls)},
        "selection": {"image_ids": selected, "outside_reference_train_image_ids": outside, "long_loop_image_id": 548337, "descriptive_only": "selected14 excludes only image548337"},
        "scores": {**controls, "P16": p, "R16": r}, "slices": slices, "comparisons": comparisons, "gates": gates,
        "actual91_starting_source_new": {"baseline_label": start["label"], "identities": [{"image_id": image_id, "owner_id": owner_id} for image_id, owner_id in sorted(start["starting"])], "P16": _survival(start=start["starting"], source=start["source_coverage"], endpoint=p), "R16": _survival(start=start["starting"], source=start["source_coverage"], endpoint=r)},
        "likelihood_evidence": {"P": _likelihood(p_likelihood_path), "R": _likelihood(r_likelihood_path)},
        "disposition": "Technical saved-row evaluation only; R alone does not isolate negative credit.",
    }
    require(not output.is_symlink(), "result output symlink")
    output = output.resolve()
    training.publish(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--terminal", type=Path, required=True)
    worker_parser.add_argument("--split", choices=("train", "dev"))
    worker_parser.add_argument("--shard", type=int)
    worker_parser.add_argument("--qualification", action="store_true")
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--device", default="cuda:0")
    reducer = sub.add_parser("reduce")
    for arm in ("p", "r"):
        reducer.add_argument(f"--{arm}-manifest", type=Path, required=True)
        reducer.add_argument(f"--{arm}-terminal", type=Path, required=True)
        reducer.add_argument(f"--{arm}-readback-root", type=Path, required=True)
        reducer.add_argument(f"--{arm}-likelihood", type=Path)
    reducer.add_argument("--fixed-result", type=Path, default=FIXED_RESULT)
    reducer.add_argument("--normalized-result", type=Path, default=NORMALIZED_RESULT)
    reducer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        require((args.qualification and args.split is None and args.shard is None) or (not args.qualification and args.split is not None and args.shard is not None), "worker arguments")
        value = worker(manifest_path=args.manifest, terminal_path=args.terminal, split=args.split, shard=args.shard, qualification=args.qualification, output=args.output, device=args.device)
    else:
        value = reduce(p_manifest_path=args.p_manifest, p_terminal_path=args.p_terminal, p_readback_root=args.p_readback_root, r_manifest_path=args.r_manifest, r_terminal_path=args.r_terminal, r_readback_root=args.r_readback_root, fixed_result_path=args.fixed_result, normalized_result_path=args.normalized_result, p_likelihood_path=args.p_likelihood, r_likelihood_path=args.r_likelihood, output=args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
