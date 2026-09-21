#!/usr/bin/env python3
"""Identity-bound cold native-HF evaluation for the owner-focus successor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.coco_gt_correction_bank import file_sha256  # noqa: E402
from scripts.research.eval_coco_gt_correction import (  # noqa: E402
    EXPECTED_GENERATION as PORTFOLIO_GENERATION,
    SOURCE_RESOLVED_CONFIG,
    _json_object,
)
from scripts.research.train_coco_gt_correction import load_cold_checkpoint  # noqa: E402
from src.inference import pipeline, worker  # noqa: E402
from src.inference.data_parallel import resolve_visible_cuda_tokens  # noqa: E402
from src.inference.hf_backend import open_hf_backend_session  # noqa: E402

EXPERIMENT_ID = "2026-09-07-coco-owner-focus-ablation"
ARMS = ("Source", "R", "M", "Rweak")
EXPECTED_GENERATION = {
    "batch_size": 4,
    "max_new_tokens": 3084,
    "n": 1,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_p": 1.0,
}


def load_successor_checkpoint(
    checkpoint: Path,
    *,
    arm: str,
    bank_manifest: Path,
    expected_completed_update: int,
) -> dict[str, Any]:
    if arm not in ARMS[1:]:
        raise ValueError(f"successor checkpoint arm must be one of {ARMS[1:]}")
    bank = _json_object(bank_manifest)
    manifest = load_cold_checkpoint(
        checkpoint,
        expected_arm=arm,
        expected_surface="dora",
        expected_bank_id=str(bank["bank_id"]),
        expected_source_identity_sha256=str(bank["source_identity"]["sha256"]),
    )
    if int(manifest["completed_update"]) != expected_completed_update:
        raise ValueError("checkpoint completed update differs")
    if Path(manifest["bank"]["manifest_path"]).resolve() != bank_manifest:
        raise ValueError("checkpoint names another bank manifest")
    if file_sha256(bank_manifest) != manifest["bank"]["manifest_sha256"]:
        raise ValueError("checkpoint bank manifest digest differs")
    if manifest["source_identity"] != bank["source_identity"]:
        raise ValueError("checkpoint Source composition differs from bank")
    expected_recipe = {
        "experiment_id": EXPERIMENT_ID,
        "objective_variant": arm,
        "seed": 20260908,
        "global_batch_size": 32,
    }
    recipe = manifest.get("recipe", {})
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("checkpoint is not identity-bound to the owner-focus recipe")
    if manifest.get("surface") != "dora" or manifest.get("payload", {}).get("kind") != "dora_adapter":
        raise ValueError("owner-focus checkpoint is not a DoRA adapter")
    return manifest


def _cold_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_id": manifest["checkpoint_id"],
        "arm": manifest["arm"],
        "objective_variant": manifest["recipe"]["objective_variant"],
        "surface": manifest["surface"],
        "completed_update": manifest["completed_update"],
        "bank_id": manifest["bank"]["bank_id"],
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "seed": manifest["recipe"]["seed"],
        "global_batch_size": manifest["recipe"]["global_batch_size"],
        "payload": {"kind": manifest["payload"]["kind"], "files": manifest["payload"]["files"]},
    }


def write_eval_config(
    *,
    source_resolved_config: Path,
    bank_manifest: Path,
    manifest: Mapping[str, Any] | None,
    input_jsonl: Path,
    artifact_root: Path,
    run_name: str,
    source_gate_root: Path,
    batch_size: int = 4,
    config_root: Path | None = None,
) -> Path:
    if batch_size not in (1, 4, 8):
        raise ValueError("batch size must stay on the bounded 1/4/8 qualification ladder")
    bank = _json_object(bank_manifest)
    document = _json_object(source_resolved_config)
    config = deepcopy(document.get("config", document))
    source = bank["source_identity"]
    expected_source = {
        "base_model_path": config.get("model", {}).get("base_model"),
        "adapter_root": config.get("adapter", {}).get("path"),
        "embedding_root": config.get("embedding_delta", {}).get("path"),
        "dtype": config.get("model", {}).get("dtype"),
        "attention_implementation": config.get("backend", {}).get("hf", {}).get("attn_implementation"),
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValueError("Source resolved config differs from sealed bank composition")
    if config.get("backend", {}).get("type") != "hf" or config.get("generation") != PORTFOLIO_GENERATION:
        raise ValueError("Source resolved config differs from admitted native decode")
    if config.get("backend", {}).get("hf", {}).get("patch_embed_linearization") != "enabled":
        raise ValueError("Source resolved config changed patch embedding")
    config["generation"] = {**EXPECTED_GENERATION, "batch_size": batch_size}
    if batch_size == 1:
        config.setdefault("debug", {})["smoke"] = True
    config["data"]["input_jsonl"] = str(input_jsonl)
    config["run"].update(
        {"name": run_name, "artifact_root": str(artifact_root), "output_dir": None, "collision_policy": "fail"}
    )
    config["adapter"]["path"] = (
        source["adapter_root"] if manifest is None else manifest["payload"]["path"]
    )
    config["embedding_delta"]["path"] = source["embedding_root"]
    config["embedding_delta"]["source_gate_root"] = str(source_gate_root)
    tag = "source" if manifest is None else str(manifest["checkpoint_id"])[:12]
    target_root = artifact_root / "configs" if config_root is None else config_root
    target = target_root / f".coco-owner-focus-{run_name}-{tag}.yaml"
    if target.exists():
        raise ValueError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


class ColdSuccessorSession:
    def __init__(self, inner: Any, identity: Mapping[str, Any]) -> None:
        self._inner = inner
        self._identity = dict(identity)

    @property
    def receipt(self) -> Any:
        base = self._inner.receipt
        model_identity = dict(base.model_identity)
        model_identity["coco_owner_focus"] = dict(self._identity)
        return replace(base, model_identity=model_identity)

    def decode(self, requests: Sequence[Any]) -> Sequence[Any]:
        return self._inner.decode(requests)

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def session_opener(
    *, checkpoint: Path, arm: str, bank_manifest: Path, expected_completed_update: int
) -> Callable[[Any], ColdSuccessorSession]:
    def open_session(launch: Any) -> ColdSuccessorSession:
        manifest = load_successor_checkpoint(
            checkpoint,
            arm=arm,
            bank_manifest=bank_manifest,
            expected_completed_update=expected_completed_update,
        )
        inner = open_hf_backend_session(launch)
        return ColdSuccessorSession(inner, _cold_identity(manifest))

    return open_session


def _launch_worker(
    *, checkpoint: Path, arm: str, bank_manifest: Path, expected_completed_update: int, **kwargs: Any
) -> subprocess.Popen[Any]:
    if kwargs.get("execution_model_json") is not None or kwargs.get("execution_context_json") is not None:
        raise ValueError("cold HF workers do not accept execution-model/context transport")
    rank = int(kwargs["rank"])
    cache_root = Path(tempfile.mkdtemp(prefix=f"coco-owner-focus-eval-rank-{rank}-"))
    env = worker.build_worker_environment(
        base_env=None,
        parent_visible_device_token=str(kwargs["parent_visible_device_token"]),
        runtime_cache_root=cache_root,
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--checkpoint",
        str(checkpoint),
        "--arm",
        arm,
        "--bank-manifest",
        str(bank_manifest),
        "--expected-completed-update",
        str(expected_completed_update),
    ]
    for name in (
        "rank",
        "world_size",
        "parent_visible_device_token",
        "resolved_config_json",
        "shard_plan_json",
        "output_dir",
    ):
        command.extend((f"--{name.replace('_', '-')}", str(kwargs[name])))
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
    except BaseException:
        shutil.rmtree(cache_root, ignore_errors=True)
        raise
    setattr(process, "_coordexp_runtime_cache_root", str(cache_root))
    return process


def _run_worker(args: argparse.Namespace) -> int:
    resolved = worker.load_resolved_infer_config_artifact(args.resolved_config_json)
    plan = worker.load_data_parallel_plan_artifact(args.shard_plan_json)
    rank_plan = next((item for item in plan.ranks if item.rank == args.rank), None)
    if rank_plan is None or rank_plan.world_size != args.world_size:
        raise ValueError("worker rank/world size differs from shard plan")
    runtime = worker.build_worker_runtime_metadata(
        rank=args.rank,
        world_size=args.world_size,
        parent_visible_device_token=args.parent_visible_device_token,
    )
    pipeline.run_shard(
        resolved=resolved,
        output_dir=Path(args.output_dir),
        row_indices=rank_plan.row_indices,
        worker_metadata={
            "shard_plan_fingerprint": plan.fingerprint,
            "rank": runtime["rank"],
            "world_size": runtime["world_size"],
            "parent_visible_device_token": runtime["parent_visible_device_token"],
            "worker_cuda_visible_devices": runtime["worker_cuda_visible_devices"],
            "worker_logical_device": runtime["logical_device"],
            "cuda_device_count": runtime["cuda_device_count"],
            "cuda_current_device": runtime["cuda_current_device"],
            "model_first_parameter_device": runtime.get("model_first_parameter_device"),
            "per_device_batch_size": rank_plan.per_device_batch_size,
            "batch_ids": list(rank_plan.batch_ids),
        },
        rank_plan=rank_plan,
        session_opener=session_opener(
            checkpoint=args.checkpoint,
            arm=args.arm,
            bank_manifest=args.bank_manifest,
            expected_completed_update=args.expected_completed_update,
        ),
    )
    return 0


def run_eval(args: argparse.Namespace) -> int:
    manifest = None
    if args.arm != "Source":
        if args.checkpoint is None:
            raise ValueError("candidate evaluation requires --checkpoint")
        manifest = load_successor_checkpoint(
            args.checkpoint,
            arm=args.arm,
            bank_manifest=args.bank_manifest,
            expected_completed_update=args.expected_completed_update,
        )
    elif args.checkpoint is not None:
        raise ValueError("Source evaluation must not provide a checkpoint")
    visible = resolve_visible_cuda_tokens()
    if len(visible) != args.expected_active_ranks:
        raise ValueError(f"expected {args.expected_active_ranks} visible GPUs, found {len(visible)}")
    config = write_eval_config(
        source_resolved_config=args.source_resolved_config,
        bank_manifest=args.bank_manifest,
        manifest=manifest,
        input_jsonl=args.input_jsonl,
        artifact_root=args.artifact_root,
        run_name=args.run_name,
        source_gate_root=args.source_gate_root,
        batch_size=args.batch_size,
        config_root=REPO_ROOT / "configs" / "coordexp_swift" / "infer",
    )
    try:
        if args.arm == "Source":
            return pipeline.run(config_path=config)
        opener = session_opener(
            checkpoint=args.checkpoint,
            arm=args.arm,
            bank_manifest=args.bank_manifest,
            expected_completed_update=args.expected_completed_update,
        )
        launcher = lambda **kwargs: _launch_worker(  # noqa: E731
            checkpoint=args.checkpoint,
            arm=args.arm,
            bank_manifest=args.bank_manifest,
            expected_completed_update=args.expected_completed_update,
            **kwargs,
        )
        return pipeline.run(config_path=config, session_opener=opener, worker_launcher=launcher)
    finally:
        config.unlink(missing_ok=True)


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in result:
                raise ValueError(f"duplicate row ID in {path}: {row_id}")
            result[row_id] = row
    return result


def _token_ids(path: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("trace_type") == "generated_token" and row.get("is_pad") is not True:
                result.setdefault(str(row["row_id"]), []).append(int(row["token_id"]))
    return result


def compare_batching(baseline_run: Path, batch4_run: Path) -> dict[str, Any]:
    summaries = [_json_object(root / "summary.json") for root in (baseline_run, batch4_run)]
    policies = [summary.get("generation_policy", {}) for summary in summaries]
    if policies[0].get("batch_size") != 1 or policies[1].get("batch_size") != 4:
        raise ValueError("comparison requires batch-1 baseline and batch-4 candidate")
    for key, value in {**EXPECTED_GENERATION, "do_sample": False}.items():
        if key in ("batch_size", "n"):
            continue
        if any(policy.get(key) != value for policy in policies):
            raise ValueError(f"generation policy differs at {key}")
    rows = [_rows_by_id(root / "gt_vs_pred.jsonl") for root in (baseline_run, batch4_run)]
    if set(rows[0]) != set(rows[1]):
        raise ValueError("batching comparison row IDs differ")
    fields = ("raw_decode_text", "decode_stop_reason", "parse_status", "pred", "dropped_predictions")
    mismatches = {
        row_id: [field for field in fields if rows[0][row_id].get(field) != rows[1][row_id].get(field)]
        for row_id in rows[0]
    }
    mismatches = {row_id: fields for row_id, fields in mismatches.items() if fields}
    tokens = [_token_ids(root / "pred_token_trace.jsonl") for root in (baseline_run, batch4_run)]
    token_mismatch_ids = sorted(row_id for row_id in rows[0] if tokens[0].get(row_id, []) != tokens[1].get(row_id, []))
    stop_counts: dict[str, int] = {}
    for row in rows[0].values():
        reason = str(row.get("decode_stop_reason"))
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    if stop_counts.get("im_end", 0) < 1 or stop_counts.get("length", 0) < 1:
        raise ValueError("sensitivity panel must contain native EOS and long capped cases")
    performance = [summary["performance"] for summary in summaries]
    result = {
        "schema": "coco-owner-focus-batching-qualification.v1",
        "status": "qualified" if not mismatches and not token_mismatch_ids else "failed",
        "row_count": len(rows[0]),
        "stop_reason_counts": stop_counts,
        "prediction_field_mismatches": mismatches,
        "token_identity_mismatch_row_ids": token_mismatch_ids,
        "batch1": performance[0],
        "batch4": performance[1],
        "speedup": {
            "requests_per_second": performance[1]["requests_per_second"] / performance[0]["requests_per_second"],
            "generated_tokens_per_second": performance[1]["generated_tokens_per_second"] / performance[0]["generated_tokens_per_second"],
        },
    }
    if result["status"] != "qualified":
        raise RuntimeError(f"batch-4 prediction identity differs: {mismatches or token_mismatch_ids}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--checkpoint", type=Path)
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--bank-manifest", type=Path, required=True)
    run_parser.add_argument("--expected-completed-update", type=int, default=64)
    run_parser.add_argument("--source-resolved-config", type=Path, default=SOURCE_RESOLVED_CONFIG)
    run_parser.add_argument("--input-jsonl", type=Path, required=True)
    run_parser.add_argument("--artifact-root", type=Path, required=True)
    run_parser.add_argument("--run-name", required=True)
    run_parser.add_argument("--expected-active-ranks", type=int, required=True)
    run_parser.add_argument("--source-gate-root", type=Path, required=True)
    run_parser.add_argument("--batch-size", type=int, default=4)

    compare_parser = subparsers.add_parser("compare-batching")
    compare_parser.add_argument("--batch1-run", type=Path, required=True)
    compare_parser.add_argument("--batch4-run", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)

    worker_parser = subparsers.add_parser("_worker")
    worker_parser.add_argument("--checkpoint", type=Path, required=True)
    worker_parser.add_argument("--arm", choices=ARMS[1:], required=True)
    worker_parser.add_argument("--bank-manifest", type=Path, required=True)
    worker_parser.add_argument("--expected-completed-update", type=int, required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--world-size", type=int, required=True)
    worker_parser.add_argument("--parent-visible-device-token", required=True)
    worker_parser.add_argument("--resolved-config-json", type=Path, required=True)
    worker_parser.add_argument("--shard-plan-json", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "compare-batching":
        out = args.out.expanduser().resolve()
        if out.exists():
            raise ValueError(f"refusing to overwrite {out}")
        result = compare_batching(args.batch1_run.resolve(strict=True), args.batch4_run.resolve(strict=True))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    for name in ("checkpoint", "bank_manifest"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.expanduser().resolve(strict=True))
    if args.command == "run":
        args.source_resolved_config = args.source_resolved_config.expanduser().resolve(strict=True)
        args.input_jsonl = args.input_jsonl.expanduser().resolve(strict=True)
        args.artifact_root = args.artifact_root.expanduser().resolve()
        args.source_gate_root = args.source_gate_root.expanduser().resolve(strict=True)
        return run_eval(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
