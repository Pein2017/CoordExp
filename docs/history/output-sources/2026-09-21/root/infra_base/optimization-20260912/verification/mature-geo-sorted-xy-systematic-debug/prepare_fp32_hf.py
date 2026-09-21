"""Materialize the mature composition in FP32 and emit a diagnostic HF config."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/data/CoordExp/.worktrees/coordexp-infras")

from src.config.inference import load_infer_config
from src.inference.execution_model import resolve_execution_model


ROOT = Path(__file__).resolve().parent
REFERENCE_CONFIG = (
    ROOT.parent
    / "mature-geo-sorted-xy-consistency"
    / "configs"
    / "dynamic-hf-bf16.yaml"
)
CACHE = ROOT / "fp32-execution-models"
OUTPUT_CONFIG = ROOT / "mature-materialized-hf-fp32.yaml"
RECEIPT = ROOT / "fp32-materialization-preparation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result: dict[str, object] = {
        "status": "failed",
        "reference_config": str(REFERENCE_CONFIG),
        "reference_config_sha256": _sha256(REFERENCE_CONFIG),
    }
    try:
        reference = load_infer_config(REFERENCE_CONFIG).config
        execution_model = resolve_execution_model(
            base_model_path=reference.model.base_model,
            target_dtype="fp32",
            adapter_path=(
                reference.adapter.path if reference.adapter is not None else None
            ),
            adapter_name=(
                reference.adapter.name if reference.adapter is not None else "default"
            ),
            embedding_delta_path=(
                reference.embedding_delta.path
                if reference.embedding_delta is not None
                else None
            ),
            cache_root=CACHE,
            _skip_existing_composition_fidelity=True,
        )
        payload = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
        payload["run"] = {
            "name": "mature-geo-sorted-xy-materialized-hf-fp32-diagnostic",
            "artifact_root": str(ROOT),
            "output_dir": "hf-fp32",
            "collision_policy": "fail",
        }
        payload["model"]["base_model"] = execution_model["model_path"]
        payload["model"]["dtype"] = "fp32"
        payload["backend"] = {
            "type": "hf",
            "hf": {
                "attn_implementation": "eager",
                "patch_embed_linearization": "enabled",
            },
        }
        payload.pop("adapter", None)
        payload.pop("embedding_delta", None)
        OUTPUT_CONFIG.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        emitted = load_infer_config(OUTPUT_CONFIG).config
        if emitted.model.base_model != execution_model["model_path"]:
            raise RuntimeError("emitted config does not resolve to FP32 snapshot")
        result.update(
            {
                "status": "prepared",
                "output_config": str(OUTPUT_CONFIG),
                "output_config_sha256": _sha256(OUTPUT_CONFIG),
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
                "factor": {
                    "weights_and_execution_dtype": "fp32",
                    "attention_implementation": "eager",
                    "reason": "FlashAttention2 does not support the FP32 diagnostic",
                },
            }
        )
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc)[:500],
        }
    RECEIPT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "prepared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
