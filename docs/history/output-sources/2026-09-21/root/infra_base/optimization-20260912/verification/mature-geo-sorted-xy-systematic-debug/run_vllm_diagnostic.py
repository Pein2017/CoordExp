"""Run one full current vLLM arm without mutating production admission.

This deliberately reuses the production frontend, execution-model resolver,
vLLM backend, parser, scorer, and artifact writer. Only the qualification
lookup is replaced with an explicit diagnostic receipt so that a failed
dynamic-HF composition gate does not prevent measuring the downstream engine
on the already authenticated dense snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

REPO = Path("/data/CoordExp/.worktrees/coordexp-infras")
sys.path.insert(0, str(REPO))

from src.config.fingerprint import sha256_json
from src.config.inference import load_infer_config
from src.config.writer import write_resolved_config_artifacts
from src.inference.artifacts import validate_scored_artifact_set
from src.inference.execution_model import resolve_execution_model
from src.inference.pipeline import run_shard
from src.inference.vllm_backend import open_vllm_backend_session


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "mature-vllm-bf16.yaml"
CACHE = (
    ROOT.parent
    / "mature-geo-sorted-xy-consistency"
    / "execution-models"
)
RUN_DIR = ROOT / "vllm-bf16"
RECEIPT = ROOT / "vllm-bf16-diagnostic-launch.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result: dict[str, Any] = {
        "status": "failed",
        "scope": "diagnostic_only_no_qualification_or_admission_write",
        "config": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        if RUN_DIR.exists():
            raise RuntimeError(f"output already exists: {RUN_DIR}")
        resolved = load_infer_config(CONFIG)
        config = resolved.config
        if config.backend.type != "vllm" or config.model.dtype != "bf16":
            raise RuntimeError("diagnostic requires a BF16 vLLM config")
        execution_model = resolve_execution_model(
            base_model_path=config.model.base_model,
            target_dtype="bf16",
            adapter_path=config.adapter.path if config.adapter is not None else None,
            adapter_name=config.adapter.name if config.adapter is not None else "default",
            embedding_delta_path=(
                config.embedding_delta.path
                if config.embedding_delta is not None
                else None
            ),
            cache_root=CACHE,
            _skip_existing_composition_fidelity=True,
        )
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        write_resolved_config_artifacts(resolved, RUN_DIR)
        (RUN_DIR / "execution_model.json").write_text(
            json.dumps(execution_model, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        def launch_qualifier(
            launch: Any,
            engine_kwargs: Mapping[str, object],
        ) -> Mapping[str, object]:
            return {
                "candidate_version": metadata.version("vllm"),
                "engine_settings": dict(engine_kwargs),
                "runtime_qualification": {
                    "status": "diagnostic_bypass_not_admitted",
                    "reason": "measure_engine_delta_after_failed_composition_gate",
                    "model_identity_sha256": sha256_json(execution_model),
                },
            }

        def raw_replay_qualifier(
            launch: Any,
            processor_identity: Mapping[str, object],
        ) -> Mapping[str, object]:
            return {
                "status": "diagnostic_bypass_not_admitted",
                "processor_source_sha256": processor_identity.get("source_sha256"),
            }

        def session_opener(launch: Any) -> Any:
            return open_vllm_backend_session(
                launch,
                launch_qualifier=launch_qualifier,
                raw_replay_qualifier=raw_replay_qualifier,
            )

        run_shard(
            resolved=resolved,
            output_dir=RUN_DIR,
            row_indices=tuple(range(32)),
            worker_metadata={
                "diagnostic_only": True,
                "qualification_bypassed": True,
                "reason": "isolate_vllm_engine_on_authenticated_dense_snapshot",
            },
            session_opener=session_opener,
            execution_model=execution_model,
        )
        validate_scored_artifact_set(RUN_DIR)
        summary = json.loads((RUN_DIR / "summary.json").read_text())
        manifest = json.loads((RUN_DIR / "run_manifest.json").read_text())
        result.update(
            {
                "status": "completed",
                "run_dir": str(RUN_DIR),
                "execution_model": {
                    key: execution_model[key]
                    for key in (
                        "composition_key",
                        "snapshot_fingerprint",
                        "receipt_fingerprint",
                        "model_path",
                        "target_dtype",
                    )
                },
                "backend_version": manifest["backend_session"]["backend_version"],
                "summary": {
                    key: summary[key]
                    for key in (
                        "terminal_status",
                        "row_count",
                        "decode_success_count",
                        "parser_failure_count",
                        "score_failure_count",
                        "scoreable_prediction_count",
                        "dropped_prediction_count",
                        "truncated_decode_count",
                        "decode_stop_reasons",
                    )
                },
            }
        )
    except BaseException as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc)[:1000],
        }
    RECEIPT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
