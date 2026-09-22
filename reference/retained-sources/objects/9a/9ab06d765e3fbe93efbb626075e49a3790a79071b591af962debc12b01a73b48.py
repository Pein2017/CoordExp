#!/usr/bin/env python3
"""Paired all-image reducer for the COCO owner-focus successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    _gt_objects,
    _gt_signature,
    _pred_objects,
    _read_jsonl,
    _sha256,
    _strict_physical_owner_duplicate_candidates,
)
from scripts.research.reduce_coco_gt_correction import (
    _json_object,
    _monitor,
    _owner_ids,
    owner_cohorts_from_bank,
)
from src.data import load_raw_examples
from src.eval.detection_consumer import evaluate_scored_detection_artifacts
from src.inference.artifacts import MANIFEST_NAME, RAW_NAME, SUMMARY_NAME, validate_scored_artifact_set

EXPERIMENT_ID = "2026-09-07-coco-owner-focus-ablation"
ARMS = ("Source", "R", "M", "Rweak")
COMPARISONS = (("M", "Rweak"), ("M", "R"), ("M", "Source"))
IOU_THRESHOLDS = (0.50, 0.60, 0.80)
EXPECTED_GENERATION = {
    "batch_size": 4,
    "do_sample": False,
    "max_new_tokens": 3084,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_p": 1.0,
}
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_DRAWS = 10_000


def _validate_rows_against_input(rows: Mapping[str, dict[str, Any]], input_jsonl: Path) -> None:
    expected = {str(example.example_id): example for example in load_raw_examples(input_jsonl)}
    if set(rows) != set(expected):
        raise ValueError("raw output image IDs differ from frozen input")
    for row_id, example in expected.items():
        row = rows[row_id]
        try:
            observed = {
                "row_id": row.get("row_id"),
                "example_id": row.get("example_id"),
                "image_path": row.get("image_path"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "gt": [
                    {
                        "object_id": obj.get("object_id"),
                        "description": obj.get("description"),
                        "bbox": obj.get("bbox"),
                    }
                    for obj in row.get("gt", [])
                ],
            }
        except AttributeError as exc:
            raise ValueError(f"raw output identity is malformed for {row_id!r}") from exc
        wanted = {
            "row_id": row_id,
            "example_id": row_id,
            "image_path": str(example.image.path),
            "image_width": example.image.width,
            "image_height": example.image.height,
            "gt": [
                {
                    "object_id": obj.object_id,
                    "description": obj.description,
                    "bbox": list(obj.bbox),
                }
                for obj in example.objects
            ],
        }
        if observed != wanted:
            raise ValueError(f"raw output identity differs from frozen input for {row_id!r}")


def _validate_source_identity(identities: Mapping[str, Any], bank: Mapping[str, Any]) -> None:
    source = bank["source_identity"]
    adapter = identities.get("adapter", {})
    embedding = identities.get("embedding_delta", {})
    embedding_identity = embedding.get("identity", {})
    if (
        identities.get("base", {}).get("path") != source["base_model_path"]
        or adapter.get("adapter_path") != source["adapter_root"]
        or adapter.get("base_model_path") != source["base_model_path"]
        or adapter.get("status") != "validated"
        or embedding_identity.get("delta_path") != source["embedding_root"]
        or embedding_identity.get("base_model_path") != source["base_model_path"]
        or embedding_identity.get("status") != "validated"
        or embedding.get("status") != "loaded"
    ):
        raise ValueError("Source run composition differs from sealed bank")
    adapter_tensor = Path(source["adapter_root"]) / "adapter_model.safetensors"
    if not adapter_tensor.is_file() or _sha256(adapter_tensor) != source["adapter_tensor_sha256"]:
        raise ValueError("sealed Source adapter tensor differs from bank")


def _matched_by_image(rows: Mapping[str, dict[str, Any]], threshold: float) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for row_id, row in rows.items():
        gt = _gt_objects(row, row_id=row_id)
        pred, _ = _pred_objects(row)
        owner_ids = _owner_ids(row, row_id=row_id)
        result[row_id] = {owner_ids[index] for index, _, _ in _global_matches(gt, pred, threshold)}
    return result


def paired_image_bootstrap(
    candidate: Mapping[str, set[int]],
    reference: Mapping[str, set[int]],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("paired bootstrap requires the same nonempty image IDs")
    row_ids = sorted(candidate)
    differences = np.asarray(
        [len(candidate[row_id]) - len(reference[row_id]) for row_id in row_ids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "unit": "matched annotated owners per image",
        "image_count": len(row_ids),
        "draws": draws,
        "seed": seed,
        "observed_mean_difference": float(differences.mean()),
        "observed_total_difference": int(differences.sum()),
        "percentile_95_mean_interval": [float(low), float(high)],
        "percentile_95_total_equivalent_interval": [float(low * len(row_ids)), float(high * len(row_ids))],
        "diagnostic_only": True,
    }


def _debt_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float | int]:
    fields = (
        "generated_row_count",
        "invalid_prediction_count",
        "dropped_prediction_count",
        "strict_duplicate_candidate_count",
        "ambiguous_duplicate_attribution_count",
        "natural_eos_image_count",
        "length_stop_image_count",
        "cap_or_nontermination_image_count",
    )
    result: dict[str, float | int] = {field: int(candidate[field]) - int(reference[field]) for field in fields}
    if "lengths" in candidate and "lengths" in reference:
        result.update(
            {
                "total_generated_tokens": int(candidate["lengths"]["total_generated_tokens"])
                - int(reference["lengths"]["total_generated_tokens"]),
                "mean_generated_tokens": float(candidate["lengths"]["mean_generated_tokens"])
                - float(reference["lengths"]["mean_generated_tokens"]),
                "max_generated_tokens": int(candidate["lengths"]["max_generated_tokens"])
                - int(reference["lengths"]["max_generated_tokens"]),
            }
        )
    return result


def reduce_rows(
    arms: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    expected_rows: int,
    expected_owners: int,
    run_dirs: Mapping[str, Path] | None = None,
    owner_cohorts: Mapping[str, set[int]] | None = None,
) -> dict[str, Any]:
    if set(arms) != set(ARMS):
        raise ValueError(f"arms must be exactly {ARMS}")
    source = arms["Source"]
    if len(source) != expected_rows:
        raise ValueError(f"expected {expected_rows} images, found {len(source)}")
    owner_universe: set[int] = set()
    for row_id in sorted(source):
        signature = _gt_signature(source[row_id], row_id=row_id)
        ids = _owner_ids(source[row_id], row_id=row_id)
        if owner_universe.intersection(ids):
            raise ValueError(f"duplicate durable owner ID in row {row_id!r}")
        owner_universe.update(ids)
        for arm, rows in arms.items():
            if set(rows) != set(source):
                raise ValueError(f"{arm} image denominator differs from Source")
            if _gt_signature(rows[row_id], row_id=row_id) != signature or _owner_ids(rows[row_id], row_id=row_id) != ids:
                raise ValueError(f"{arm} GT identity differs for {row_id!r}")
    if len(owner_universe) != expected_owners:
        raise ValueError(f"expected {expected_owners} owners, found {len(owner_universe)}")
    for name, cohort in (owner_cohorts or {}).items():
        if not cohort.issubset(owner_universe):
            raise ValueError(f"{name} cohort contains foreign owner IDs")

    reports = {
        arm: {"monitors": _monitor(rows, run_dir=None if run_dirs is None else run_dirs[arm])}
        for arm, rows in arms.items()
    }
    paired: dict[str, Any] = {}
    for threshold in IOU_THRESHOLDS:
        key = f"iou_{threshold:.2f}"
        matched_images = {arm: _matched_by_image(rows, threshold) for arm, rows in arms.items()}
        matched = {arm: set().union(*by_image.values()) for arm, by_image in matched_images.items()}
        for arm in ARMS:
            unmatched = sum(
                len(_pred_objects(arms[arm][row_id])[0]) - len(matched_images[arm][row_id])
                for row_id in source
            )
            reports[arm][key] = {
                "matched_owner_count": len(matched[arm]),
                "annotated_owner_denominator": expected_owners,
                "annotated_owner_recall": len(matched[arm]) / expected_owners,
                "matched_owner_ids": sorted(matched[arm]),
                "unmatched_prediction_count": unmatched,
            }
            if owner_cohorts:
                reports[arm][key]["cohorts"] = {
                    name: {
                        "matched_owner_count": len(matched[arm] & cohort),
                        "annotated_owner_denominator": len(cohort),
                    }
                    for name, cohort in owner_cohorts.items()
                }
        paired[key] = {}
        for candidate, reference in COMPARISONS:
            gains = matched[candidate] - matched[reference]
            losses = matched[reference] - matched[candidate]
            name = f"{candidate}_minus_{reference}"
            paired[key][name] = {
                "gain_count": len(gains),
                "loss_count": len(losses),
                "net_count": len(gains) - len(losses),
                "gain_owner_ids": sorted(gains),
                "loss_owner_ids": sorted(losses),
                "image_level_paired_bootstrap": paired_image_bootstrap(
                    matched_images[candidate], matched_images[reference]
                ),
            }
            if owner_cohorts:
                paired[key][name]["cohorts"] = {
                    cohort_name: {
                        "gain_count": len(gains & cohort),
                        "loss_count": len(losses & cohort),
                        "net_count": len(gains & cohort) - len(losses & cohort),
                    }
                    for cohort_name, cohort in owner_cohorts.items()
                }
    debt = {
        f"{candidate}_minus_{reference}": _debt_delta(
            reports[candidate]["monitors"], reports[reference]["monitors"]
        )
        for candidate, reference in COMPARISONS
    }
    return {
        "image_denominator": expected_rows,
        "annotated_owner_denominator": expected_owners,
        "arms": reports,
        "paired": paired,
        "paired_debt_vector": debt,
    }


def _validate_topology(
    run_dir: Path,
    *,
    arm: str,
    input_jsonl: Path,
    expected_active_ranks: int,
    expected_completed_update: int,
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    validate_scored_artifact_set(run_dir)
    manifest = _json_object(run_dir / MANIFEST_NAME)
    summary = _json_object(run_dir / SUMMARY_NAME)
    if summary.get("terminal_status") != "completed":
        raise ValueError(f"{run_dir} is not completed")
    if manifest.get("backend") != "hf" or manifest.get("backend_mode") != "generate":
        raise ValueError(f"{run_dir} is not native HF generate")
    policy = manifest.get("generation_policy", {})
    if any(policy.get(key) != value for key, value in EXPECTED_GENERATION.items()):
        raise ValueError(f"{run_dir} generation policy differs")
    parallelism = manifest.get("parallelism", {})
    active = parallelism.get("active_ranks", parallelism.get("plan", {}).get("active_ranks"))
    if active != expected_active_ranks:
        raise ValueError(f"{run_dir} active-rank topology differs")
    if Path(str(manifest.get("dataset_identity", {}).get("input_jsonl", ""))).resolve() != input_jsonl:
        raise ValueError(f"{run_dir} dataset identity differs")
    settings = manifest.get("backend_session", {}).get("effective_settings", {})
    if settings.get("observed_attn_implementation") != "sdpa" or settings.get("observed_model_dtype", {}).get("parameter_dtype_names") != ["torch.float32"]:
        raise ValueError(f"{run_dir} is not frozen FP32 SDPA")
    identities = manifest.get("model_identity", {})
    cold = identities.get("coco_owner_focus")
    if arm == "Source":
        _validate_source_identity(identities, bank)
        if cold is not None or identities.get("coco_gt_correction") is not None:
            raise ValueError("Source run unexpectedly carries a correction payload")
    elif not isinstance(cold, dict) or any(
        cold.get(key) != value
        for key, value in {
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "objective_variant": arm,
            "surface": "dora",
            "completed_update": expected_completed_update,
            "bank_id": bank["bank_id"],
            "source_identity_sha256": bank["source_identity"]["sha256"],
            "seed": 20260908,
            "global_batch_size": 32,
        }.items()
    ):
        raise ValueError(f"{run_dir} has wrong or missing successor payload identity")
    return manifest


def reduce_runs(
    split_runs: Mapping[str, Mapping[str, Path]],
    *,
    train_input: Path,
    holdout_input: Path,
    holdout_manifest: Path,
    bank_manifest: Path,
    evaluation_root: Path,
    expected_active_ranks: int,
    expected_completed_update: int,
) -> dict[str, Any]:
    bank = _json_object(bank_manifest)
    holdout = _json_object(holdout_manifest)
    bank_train = bank.get("inputs", {}).get("train_jsonl", {})
    if Path(str(bank_train.get("path", ""))).resolve() != train_input or bank_train.get("sha256") != _sha256(train_input):
        raise ValueError("train input differs from sealed bank")
    frozen_holdout = holdout.get("holdout_jsonl", {})
    if (
        holdout.get("experiment_id") != EXPERIMENT_ID
        or holdout.get("status") != "frozen-outcome-blind"
        or Path(str(frozen_holdout.get("path", ""))).resolve() != holdout_input
        or frozen_holdout.get("sha256") != _sha256(holdout_input)
    ):
        raise ValueError("holdout input differs from frozen manifest")
    expected = {
        "train": (256, int(bank["annotated_owner_count"])),
        "holdout": (
            int(holdout["population"]["image_count"]),
            int(holdout["population"]["annotated_owner_count"]),
        ),
    }
    inputs = {"train": train_input, "holdout": holdout_input}
    reports: dict[str, Any] = {}
    run_receipts: dict[str, Any] = {}
    for split in ("train", "holdout"):
        runs = split_runs[split]
        if set(runs) != set(ARMS):
            raise ValueError(f"{split} runs must name exactly {ARMS}")
        rows = {arm: _read_jsonl(path / RAW_NAME) for arm, path in runs.items()}
        for arm_rows in rows.values():
            _validate_rows_against_input(arm_rows, inputs[split])
        run_receipts[split] = {}
        for arm, run_dir in runs.items():
            manifest = _validate_topology(
                run_dir,
                arm=arm,
                input_jsonl=inputs[split],
                expected_active_ranks=expected_active_ranks,
                expected_completed_update=expected_completed_update,
                bank=bank,
            )
            detection_dir = evaluation_root / "detection" / split / arm
            if detection_dir.exists():
                raise ValueError(f"refusing to overwrite detection evaluation {detection_dir}")
            detection = evaluate_scored_detection_artifacts(artifact_dir=run_dir, output_dir=detection_dir)
            run_receipts[split][arm] = {
                "run_dir": str(run_dir),
                "raw_sha256": _sha256(run_dir / RAW_NAME),
                "run_manifest_sha256": _sha256(run_dir / MANIFEST_NAME),
                "model_identity_fingerprint": manifest.get("model_identity_fingerprint"),
                "detection_metrics": detection.metrics,
            }
        reports[split] = reduce_rows(
            rows,
            expected_rows=expected[split][0],
            expected_owners=expected[split][1],
            run_dirs=runs,
            owner_cohorts=owner_cohorts_from_bank(Path(bank["records"]["path"])) if split == "train" else None,
        )
    return {
        "schema_version": "coco-owner-focus-reduction.v1",
        "status": "completed",
        "policy": {
            "experiment_id": EXPERIMENT_ID,
            "owner_matcher": "existing category-compatible cardinality-first maximum-IoU matcher",
            "iou_thresholds": list(IOU_THRESHOLDS),
            "comparisons": [f"{candidate}_minus_{reference}" for candidate, reference in COMPARISONS],
            "all_images_retained": True,
            "bootstrap": {"unit": "image", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "diagnostic_only": True},
            "native_eval_topology": {"backend": "hf", "dtype": "fp32", "attn_implementation": "sdpa", "active_ranks": expected_active_ranks, **EXPECTED_GENERATION},
        },
        "inputs": {split: {"path": str(path), "sha256": _sha256(path)} for split, path in inputs.items()},
        "bank": {"path": str(bank_manifest), "sha256": _sha256(bank_manifest), "bank_id": bank["bank_id"]},
        "holdout_manifest": {"path": str(holdout_manifest), "sha256": _sha256(holdout_manifest)},
        "runs": run_receipts,
        "splits": reports,
    }


def duplicate_concentration(scored_jsonl: Path) -> dict[str, Any]:
    rows = _read_jsonl(scored_jsonl)
    counts: list[tuple[int, str, int]] = []
    for row_id, row in rows.items():
        candidates, ambiguous = _strict_physical_owner_duplicate_candidates(
            row, row_id=row_id, annotation_iou_threshold=0.50, prediction_iou_threshold=0.95
        )
        counts.append((len(candidates), row_id, ambiguous))
    counts.sort(key=lambda item: (-item[0], item[1]))
    total = sum(item[0] for item in counts)
    return {
        "schema": "coco-owner-focus-duplicate-concentration.v1",
        "input": {"path": str(scored_jsonl), "sha256": _sha256(scored_jsonl)},
        "image_count": len(counts),
        "affected_image_count": sum(value > 0 for value, _, _ in counts),
        "strict_duplicate_candidate_count": total,
        "ambiguous_duplicate_attribution_count": sum(item[2] for item in counts),
        "top_images": [
            {"row_id": row_id, "duplicate_count": value, "share": 0.0 if total == 0 else value / total}
            for value, row_id, _ in counts[:10]
        ],
        "concentration": {
            f"top_{k}_share": 0.0 if total == 0 else sum(item[0] for item in counts[:k]) / total
            for k in (1, 5, 10)
        },
    }


def _named_runs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in ARMS or name in result:
            raise ValueError(f"run must be unique NAME=PATH for {ARMS}: {value!r}")
        result[name] = Path(path).expanduser().resolve(strict=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--train-run", action="append", required=True)
    reduce_parser.add_argument("--holdout-run", action="append", required=True)
    reduce_parser.add_argument("--train-input", type=Path, required=True)
    reduce_parser.add_argument("--holdout-input", type=Path, required=True)
    reduce_parser.add_argument("--holdout-manifest", type=Path, required=True)
    reduce_parser.add_argument("--bank-manifest", type=Path, required=True)
    reduce_parser.add_argument("--evaluation-root", type=Path, required=True)
    reduce_parser.add_argument("--out", type=Path, required=True)
    reduce_parser.add_argument("--expected-active-ranks", type=int, default=8)
    reduce_parser.add_argument("--expected-completed-update", type=int, default=64)
    duplicate_parser = subparsers.add_parser("duplicate-concentration")
    duplicate_parser.add_argument("--scored-jsonl", type=Path, required=True)
    duplicate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    if args.command == "duplicate-concentration":
        result = duplicate_concentration(args.scored_jsonl.expanduser().resolve(strict=True))
    else:
        result = reduce_runs(
            {"train": _named_runs(args.train_run), "holdout": _named_runs(args.holdout_run)},
            train_input=args.train_input.expanduser().resolve(strict=True),
            holdout_input=args.holdout_input.expanduser().resolve(strict=True),
            holdout_manifest=args.holdout_manifest.expanduser().resolve(strict=True),
            bank_manifest=args.bank_manifest.expanduser().resolve(strict=True),
            evaluation_root=args.evaluation_root.expanduser().resolve(),
            expected_active_ranks=args.expected_active_ranks,
            expected_completed_update=args.expected_completed_update,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
