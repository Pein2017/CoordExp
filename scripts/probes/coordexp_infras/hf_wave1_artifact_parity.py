#!/usr/bin/env python
"""Verify Wave 1 HF artifact parity under an explicit leaf-level policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXACT_FILES = (
    "gt_vs_pred.jsonl",
    "gt_vs_pred_scored.jsonl",
    "parse_diagnostics.jsonl",
)
PROJECTED_JSONL_FIELDS = {
    "image_plan.jsonl": {
        "backend_projection_evidence_kind",
        "executed_media_sha256",
        "image_content_sha256",
        "logical_transform_id",
    },
    "pred_token_trace.jsonl": {
        "raw_model_logprob",
        "raw_model_logprob_status",
    },
}
STRUCTURED_FILES = (
    "run_manifest.json",
    "summary.json",
    "gt_vs_pred_scored.jsonl.provenance.json",
)
LEAF = object()
QWEN_SCHEMA = {
    "architectures": (LEAF,),
    "config_class": LEAF,
    "config_dtype": LEAF,
    "model_type": LEAF,
    "text_hidden_size": LEAF,
    "text_vocab_size": LEAF,
    "tie_word_embeddings": LEAF,
}
PROCESSOR_SCHEMA = {
    "image_processor_class": LEAF,
    "merge_size": LEAF,
    "patch_size": LEAF,
    "processor_class": LEAF,
    "temporal_patch_size": LEAF,
    "tokenizer_class": LEAF,
}
TOKEN_SCHEMA = {
    "coord_token_count": LEAF,
    "coord_token_id_max": LEAF,
    "coord_token_id_min": LEAF,
    "coord_token_ids_contiguous": LEAF,
    "im_end_newline_split_verified": LEAF,
    "im_end_newline_text": LEAF,
    "im_end_newline_token_ids": (LEAF, LEAF),
    "im_end_token_ids": (LEAF,),
    "newline_token_ids": (LEAF,),
    "required_token_count": LEAF,
    "tokenizer_vocab_size": LEAF,
    "wrapper_token_ids": {
        "<|box_end|>": LEAF,
        "<|box_start|>": LEAF,
        "<|object_ref_end|>": LEAF,
        "<|object_ref_start|>": LEAF,
    },
}
LIKELIHOOD_SCHEMA = {
    "policy": LEAF,
    "raw": LEAF,
    "score_owned_channel": LEAF,
}
MODEL_IDENTITY_SCHEMA = {
    "adapter": LEAF,
    "base": {"path": LEAF},
    "embedding_delta": LEAF,
    "family": LEAF,
    "qwen": QWEN_SCHEMA,
}
BACKEND_SESSION_SCHEMA = {
    "backend": LEAF,
    "backend_mode": LEAF,
    "backend_version": LEAF,
    "effective_settings": {
        "backend_options": {
            "hf": {
                "attn_implementation": LEAF,
                "patch_embed_linearization": LEAF,
            }
        },
        "batch_size": LEAF,
        "device": LEAF,
        "output_scores": LEAF,
        "raw_output_logits": LEAF,
        "text_padding_side": LEAF,
    },
    "execution_model_identity": LEAF,
    "generation_config_fingerprint": LEAF,
    "likelihood_semantics": LIKELIHOOD_SCHEMA,
    "model_identity": MODEL_IDENTITY_SCHEMA,
    "processor_identity": PROCESSOR_SCHEMA,
    "response_family": LEAF,
    "tokenizer_identity": TOKEN_SCHEMA,
}
PATCH_RECEIPT_SCHEMA = {
    "applied": LEAF,
    "bias": LEAF,
    "dilation": LEAF,
    "embed_dim": LEAF,
    "equivalence_probe": LEAF,
    "groups": LEAF,
    "in_channels": LEAF,
    "kernel_size": LEAF,
    "name": LEAF,
    "original_class": LEAF,
    "original_forward_sha256": LEAF,
    "owner_class": LEAF,
    "owner_path": LEAF,
    "padding": LEAF,
    "patch_size": LEAF,
    "patched_class": LEAF,
    "policy": LEAF,
    "projection_class": LEAF,
    "reason": LEAF,
    "replacement_forward_sha256": LEAF,
    "stride": LEAF,
    "temporal_patch_size": LEAF,
    "weight_shape": LEAF,
}
FRONTEND_SCHEMA = {
    "attn_implementation": LEAF,
    "base_config_sha256": LEAF,
    "base_model_path": LEAF,
    "load_model": LEAF,
    "model": QWEN_SCHEMA,
    "package_versions": {
        "tokenizers": LEAF,
        "torch": LEAF,
        "transformers": LEAF,
    },
    "processor": PROCESSOR_SCHEMA,
    "runtime_patches": {
        "qwen3_vl_patch_embed_linearization": PATCH_RECEIPT_SCHEMA,
    },
    "tokenizer_sha256": LEAF,
    "tokens": TOKEN_SCHEMA,
}
MEDIA_SCHEMA = {
    "rows": [
        {
            "decoded_height": LEAF,
            "decoded_width": LEAF,
            "expected_image_grid_thw": (LEAF, LEAF, LEAF),
            "image_sha256": LEAF,
            "logical_transform_id": LEAF,
            "row_id": LEAF,
        }
    ]
}
COMMON_ADDITIVE_SCHEMAS = {
    "$.backend_session": BACKEND_SESSION_SCHEMA,
    "$.execution_model_identity": LEAF,
    "$.frontend_identity": FRONTEND_SCHEMA,
    "$.generation_policy.include_raw_model_logprob": LEAF,
    "$.likelihood_semantics": LIKELIHOOD_SCHEMA,
    "$.media_identity": MEDIA_SCHEMA,
    "$.model_identity.qwen": QWEN_SCHEMA,
    "$.raw_model_logprob_status": LEAF,
}
ADDITIVE_SCHEMAS = {
    "run_manifest.json": COMMON_ADDITIVE_SCHEMAS,
    "summary.json": {
        "$.generation_policy.include_raw_model_logprob": LEAF,
        "$.likelihood_semantics": LIKELIHOOD_SCHEMA,
        "$.raw_model_logprob_status": LEAF,
    },
    "gt_vs_pred_scored.jsonl.provenance.json": COMMON_ADDITIVE_SCHEMAS,
}
ALLOWED_CHANGED_PATHS = {
    "run_manifest.json": {
        "$.generation_config_fingerprint",
        "$.model_identity_fingerprint",
        "$.parallelism.direct_runtime.model_first_parameter_device",
        "$.parallelism.plan.fingerprint",
        "$.parallelism.plan.ranks[0].parent_visible_device_token",
        "$.parallelism.plan.visible_cuda_tokens[0]",
        "$.resolved_config_fingerprints.infer_config",
    },
    "summary.json": set(),
    "gt_vs_pred_scored.jsonl.provenance.json": {
        "$.decode_policy_fingerprint",
        "$.generation_config_fingerprint",
        "$.model_identity_fingerprint",
        "$.parallelism.direct_runtime.model_first_parameter_device",
        "$.parallelism.plan.fingerprint",
        "$.parallelism.plan.ranks[0].parent_visible_device_token",
        "$.parallelism.plan.visible_cuda_tokens[0]",
    },
}


def main() -> int:
    args = _parse_args()
    verifier_path = Path(__file__).resolve()
    try:
        verifier_display_path = str(verifier_path.relative_to(Path.cwd().resolve()))
    except ValueError:
        verifier_display_path = str(verifier_path)
    pairs = {
        "single_row": (Path(args.baseline_single), Path(args.candidate_single)),
        "heterogeneous_batch": (
            Path(args.baseline_batch),
            Path(args.candidate_batch),
        ),
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "policy": {
            "exact_files": list(EXACT_FILES),
            "projected_jsonl_fields": {
                name: sorted(fields)
                for name, fields in PROJECTED_JSONL_FIELDS.items()
            },
            "structured_files": list(STRUCTURED_FILES),
            "allowed_addition_schema_roots": {
                name: sorted(schemas)
                for name, schemas in ADDITIVE_SCHEMAS.items()
            },
            "allowed_changed_paths": {
                name: sorted(paths)
                for name, paths in ALLOWED_CHANGED_PATHS.items()
            },
        },
        "pairs": {
            name: _verify_pair(name, baseline, candidate)
            for name, (baseline, candidate) in pairs.items()
        },
        "verifier": {
            "path": verifier_display_path,
            "sha256": _sha256_file(verifier_path),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


def _verify_pair(name: str, baseline: Path, candidate: Path) -> dict[str, Any]:
    if not baseline.is_dir() or not candidate.is_dir():
        raise SystemExit(f"{name}: parity roots must be directories")
    exact = {}
    for filename in EXACT_FILES:
        baseline_bytes = (baseline / filename).read_bytes()
        candidate_bytes = (candidate / filename).read_bytes()
        if baseline_bytes != candidate_bytes:
            raise SystemExit(f"{name}: exact artifact changed: {filename}")
        exact[filename] = _sha256_bytes(candidate_bytes)

    projected = {}
    for filename, excluded_fields in PROJECTED_JSONL_FIELDS.items():
        baseline_rows = _read_jsonl(baseline / filename)
        candidate_rows = _read_jsonl(candidate / filename)
        baseline_projection = _project_rows(baseline_rows, excluded_fields)
        candidate_projection = _project_rows(candidate_rows, excluded_fields)
        if baseline_projection != candidate_projection:
            raise SystemExit(f"{name}: projected artifact changed: {filename}")
        projected[filename] = {
            "row_count": len(candidate_rows),
            "projected_sha256": _sha256_json(candidate_projection),
        }

    structured = {}
    row_count = len(_read_jsonl(candidate / "gt_vs_pred.jsonl"))
    for filename in STRUCTURED_FILES:
        structured[filename] = _verify_structured_file(
            name=name,
            filename=filename,
            baseline=_read_json(baseline / filename),
            candidate=_read_json(candidate / filename),
            expected_row_count=row_count,
        )
    return {
        "baseline_root": str(baseline),
        "candidate_root": str(candidate),
        "exact": exact,
        "projected": projected,
        "structured": structured,
    }


def _verify_structured_file(
    *,
    name: str,
    filename: str,
    baseline: Any,
    candidate: Any,
    expected_row_count: int,
) -> dict[str, Any]:
    _validate_additive_shapes(
        candidate,
        filename=filename,
        expected_row_count=expected_row_count,
    )
    baseline_leaves = _flatten(baseline)
    candidate_leaves = _flatten(candidate)
    removed = sorted(set(baseline_leaves) - set(candidate_leaves))
    added = sorted(set(candidate_leaves) - set(baseline_leaves))
    changed = sorted(
        path
        for path in set(baseline_leaves) & set(candidate_leaves)
        if baseline_leaves[path] != candidate_leaves[path]
    )
    unexpected_added = [
        path
        for path in added
        if not _matches_prefix(path, tuple(ADDITIVE_SCHEMAS[filename]))
    ]
    unexpected_changed = [
        path for path in changed if path not in ALLOWED_CHANGED_PATHS[filename]
    ]
    if removed or unexpected_added or unexpected_changed:
        raise SystemExit(
            f"{name}: structured parity failed for {filename}: "
            f"removed={removed}, unexpected_added={unexpected_added}, "
            f"unexpected_changed={unexpected_changed}"
        )
    observed_roots = sorted(
        prefix
        for prefix in ADDITIVE_SCHEMAS[filename]
        if any(_path_is_under(path, prefix) for path in added)
    )
    return {
        "removed_paths": removed,
        "observed_added_paths": added,
        "observed_addition_schema_roots": observed_roots,
        "observed_changed_paths": changed,
        "unexpected_addition_paths": unexpected_added,
        "unexpected_changed_paths": unexpected_changed,
    }


def _validate_additive_shapes(
    candidate: Any,
    *,
    filename: str,
    expected_row_count: int,
) -> None:
    for path, schema in ADDITIVE_SCHEMAS[filename].items():
        value = _resolve_path(candidate, path)
        _validate_shape(value, schema, path=path)
    media_identity = candidate.get("media_identity")
    if media_identity is not None and len(media_identity["rows"]) != expected_row_count:
        raise SystemExit(
            f"{filename}: media identity row count differs from artifact rows"
        )


def _validate_shape(value: Any, schema: Any, *, path: str) -> None:
    if schema is LEAF:
        if isinstance(value, (dict, list)):
            raise SystemExit(
                f"{path}: additive scalar leaf cannot contain nested data"
            )
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict) or set(value) != set(schema):
            observed = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise SystemExit(
                f"{path}: additive object keys differ: "
                f"expected={sorted(schema)}, observed={observed}"
            )
        for key, child_schema in schema.items():
            _validate_shape(value[key], child_schema, path=f"{path}.{key}")
        return
    if isinstance(schema, tuple):
        if not isinstance(value, list) or len(value) != len(schema):
            raise SystemExit(
                f"{path}: additive fixed sequence shape differs"
            )
        for index, child_schema in enumerate(schema):
            _validate_shape(value[index], child_schema, path=f"{path}[{index}]")
        return
    if isinstance(schema, list) and len(schema) == 1:
        if not isinstance(value, list):
            raise SystemExit(f"{path}: additive repeated sequence must be a list")
        for index, item in enumerate(value):
            _validate_shape(item, schema[0], path=f"{path}[{index}]")
        return
    raise AssertionError(f"unsupported additive schema at {path}")


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for field in path.removeprefix("$.").split("."):
        if not isinstance(current, dict) or field not in current:
            raise SystemExit(f"missing approved additive root: {path}")
        current = current[field]
    return current


def _flatten(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {path: value}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            flattened.update(_flatten(value[key], f"{path}.{key}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path: value}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{path}[{index}]"))
        return flattened
    return {path: value}


def _matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(_path_is_under(path, prefix) for prefix in prefixes)


def _path_is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(
        prefix + "["
    )


def _project_rows(
    rows: list[dict[str, Any]],
    excluded_fields: set[str],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in excluded_fields}
        for row in rows
    ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-single", required=True)
    parser.add_argument("--candidate-single", required=True)
    parser.add_argument("--baseline-batch", required=True)
    parser.add_argument("--candidate-batch", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
