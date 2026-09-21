"""CPU-only coordinate input/output geometry and init provenance audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


COORD_START = 151670
COORD_COUNT = 1000
BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
INIT_NOTE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-source-distribution/embedding-init-note.md"
)
WEIGHT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-18-untied-highconfidence18-natural/weights"
)
MODELS = {
    "tied": WEIGHT_ROOT / "tied-original/weights.pt",
    "untied": WEIGHT_ROOT / "untied-original/weights.pt",
}


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def tensor_binding(value: torch.Tensor, *, name: str) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "name": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }


def identity_bindings(weights_path: Path) -> dict[str, Any]:
    identity_path = weights_path.with_name("identity.json")
    identity = json.loads(identity_path.read_text())
    adapter = identity["adapter"]
    embedding = identity["embedding"]["identity"]
    adapter_evidence = adapter["adapter_payload_evidence"]
    adapter_paths = {
        "adapter_config": Path(adapter_evidence["config_path"]),
        "adapter_tensor": Path(adapter_evidence["tensor_path"]),
    }
    delta_root = Path(embedding["delta_path"])
    delta_paths = {
        "embedding_metadata": delta_root / "special_token_embeddings.json",
        "embedding_tensor": delta_root / "special_token_embeddings.safetensors",
    }
    return {
        "identity": binding(identity_path),
        "adapter": {name: binding(path) for name, path in adapter_paths.items()},
        "embedding_delta": {name: binding(path) for name, path in delta_paths.items()},
        "identity_fields": {
            "model": identity["model"],
            "base_model_path": adapter["base_model_path"],
            "embedding_tensor_key": embedding["metadata"]["tensor_key"],
            "tie_word_embeddings": embedding["metadata"]["tie_word_embeddings"],
            "delta_semantics": embedding["metadata"]["semantics"],
        },
    }


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def pair_geometry(rows: torch.Tensor, distances: list[int]) -> dict[str, Any]:
    rows = rows.double()
    norms = rows.norm(dim=1)
    result: dict[str, Any] = {
        "row_norm": summary(norms),
        "distances": {},
        "largest_adjacent": [],
    }
    for distance in distances:
        left = rows[:-distance]
        right = rows[distance:]
        delta = left - right
        euclidean = delta.norm(dim=1)
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
        result["distances"][str(distance)] = {
            "pairs": int(euclidean.numel()),
            "euclidean": summary(euclidean),
            "cosine": summary(cosine),
            "argmax_euclidean_index": int(torch.argmax(euclidean)),
            "argmin_cosine_index": int(torch.argmin(cosine)),
        }
        if distance == 1:
            top = torch.topk(euclidean, min(10, euclidean.numel()))
            result["largest_adjacent"] = [
                {
                    "left": int(index),
                    "right": int(index) + 1,
                    "euclidean": float(value),
                    "cosine": float(cosine[index]),
                }
                for value, index in zip(top.values, top.indices, strict=True)
            ]
    return result


def same_anchor_witnesses(rows: torch.Tensor) -> list[dict[str, Any]]:
    """Compare distances from one left anchor, not aggregate pair sets."""
    rows = rows.double()
    anchor = 0
    witnesses = []
    for short, long in ((4, 499), (4, 999), (1, 999)):
        if anchor + long >= rows.shape[0]:
            continue
        distances = {
            str(distance): float((rows[anchor] - rows[anchor + distance]).norm())
            for distance in (short, long)
        }
        witnesses.append(
            {
                "anchor": anchor,
                "distances": distances,
                "short_distance": short,
                "long_distance": long,
                "short_exceeds_long": distances[str(short)] > distances[str(long)],
            }
        )
    return witnesses


def positional_features(count: int, frequencies: int) -> torch.Tensor:
    denominator = max(count - 1, 1)
    positions = torch.arange(count, dtype=torch.float32) / denominator
    values = [positions]
    for index in range(frequencies):
        phase = 2 * math.pi * (2**index) * positions
        values.extend((torch.sin(phase), torch.cos(phase)))
    return torch.stack(values, dim=1)


def source_init_reconstruction(base_rows: torch.Tensor, tokenizer: Any) -> dict[str, Any]:
    """Rebuild the published natural-adjacent initialization on CPU."""
    with safe_open(
        str(BASE_MODEL / "model-00001-of-00002.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        embedding = handle.get_slice("model.language_model.embed_tokens.weight")
        digit_ids = [int(tokenizer.convert_tokens_to_ids(str(i))) for i in range(10)]
        digit_rows = embedding[digit_ids].float()
    digit_mean = digit_rows.mean(dim=0)
    features = positional_features(COORD_COUNT, frequencies=8)
    generator = torch.Generator(device="cpu").manual_seed(0)
    projection = torch.randn(
        features.shape[1], base_rows.shape[1], generator=generator, dtype=torch.float32
    ) / math.sqrt(features.shape[1])
    expected = digit_mean.unsqueeze(0) + 0.02 * features.matmul(projection)
    error = (base_rows.float() - expected).abs()
    return {
        "digit_token_ids": digit_ids,
        "feature_shape": list(features.shape),
        "projection_shape": list(projection.shape),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "rmse": float(error.square().mean().sqrt()),
        "base_row_norm_mean": float(base_rows.double().norm(dim=1).mean()),
        "expected_row_norm_mean": float(expected.double().norm(dim=1).mean()),
        "feature_geometry": pair_geometry(features, [1, 2, 4, 8, 16, 32, 64, 128, 256, 499, 999]),
        "source_claim": "natural_adjacent digit mean + seeded projected positional features",
        "comparison": "CPU reconstruction from coord_init.json and base checkpoint rows",
    }


def model_geometry(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = payload["selected_ids"].long()
    if ids[4:].tolist() != list(range(COORD_START, COORD_START + COORD_COUNT)):
        raise ValueError(f"unexpected selected IDs in {path}")
    input_rows = payload["input_rows"][4:].float()
    output_rows = payload["output_rows"][4:].float()
    base_rows = payload["base_rows"][4:].float()
    input_delta = payload["input_delta"][4:].float()
    output_delta = payload["output_delta"][4:].float()
    if float((input_rows - base_rows - input_delta).abs().max()) > 1e-5:
        raise ValueError(f"input base+delta mismatch: {path}")
    if float((output_rows - base_rows - output_delta).abs().max()) > 1e-5:
        raise ValueError(f"output base+delta mismatch: {path}")
    pair_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 499, 999]
    return {
        "weights": binding(path),
        "payload_bindings": identity_bindings(path),
        "tensor_shapes": {key: list(value.shape) for key, value in payload.items() if isinstance(value, torch.Tensor)},
        "input": pair_geometry(input_rows, pair_distances),
        "output": pair_geometry(output_rows, pair_distances),
        "base": pair_geometry(base_rows, pair_distances),
        "input_delta": pair_geometry(input_delta, pair_distances),
        "output_delta": pair_geometry(output_delta, pair_distances),
        "input_output_row_delta": pair_geometry(input_rows - output_rows, pair_distances),
        "input_output_row_delta_norm": summary((input_rows - output_rows).double().norm(dim=1)),
        "trained_surfaces": {
            "input_rows": tensor_binding(input_rows, name="input_rows"),
            "output_rows": tensor_binding(output_rows, name="output_rows"),
        },
        "same_anchor_witnesses": {
            "base": same_anchor_witnesses(base_rows),
            "input": same_anchor_witnesses(input_rows),
            "output": same_anchor_witnesses(output_rows),
        },
        "formula_errors": {
            "input_base_plus_delta_max_abs": float((input_rows - base_rows - input_delta).abs().max()),
            "output_base_plus_delta_max_abs": float((output_rows - base_rows - output_delta).abs().max()),
        },
        "input_output_max_abs": float((input_rows - output_rows).abs().max()),
        "input_output_mean_abs": float((input_rows - output_rows).abs().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coord-init", type=Path, default=BASE_MODEL / "coord_init.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    init = json.loads(args.coord_init.read_text())
    if init != {
        "schema_version": 1,
        "coord_init": "natural_adjacent",
        "num_numeric_coord_tokens": 1000,
        "num_frequencies": 8,
        "scale": 0.02,
        "seed": 0,
    }:
        raise ValueError("coord_init metadata differs from the frozen source contract")
    first = torch.load(MODELS["tied"], map_location="cpu", weights_only=False)
    source = source_init_reconstruction(first["base_rows"][4:].float(), tokenizer)
    result = {
        "schema": "coordinate_continuity.cpu_geometry.v1",
        "status": "cpu_complete",
        "coordinate_ids": {"start": COORD_START, "count": COORD_COUNT},
        "source": {
            "base_model": binding(BASE_MODEL / "config.json"),
            "base_embedding_shard": binding(
                BASE_MODEL / "model-00001-of-00002.safetensors"
            ),
            "embedding_init_note": binding(INIT_NOTE),
            "coord_init": binding(args.coord_init),
            "init": init,
            "source_code": binding(Path("scripts/tools/expand_coord_vocab.py").resolve()),
            "init_reconstruction": source,
        },
        "models": {name: model_geometry(path) for name, path in MODELS.items()},
        "interpretation": {
            "nearby_bins_are_not_assumed_to_be_near_vectors": True,
            "smooth_init_does_not_guarantee_slow_adjacent_change": True,
            "input_and_output_rows_are_separate_surfaces": True,
            "tied_and_untied_are_package_comparisons": True,
            "no_physical_identity_claim": True,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
