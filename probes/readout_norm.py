"""Explicit output-only median-row-norm rescaling; no model or data selection.

The median is torch.median (the lower middle value for an even number of rows).
The operation uses effective OUTPUT rows. Tied and untied input embeddings are
neither inferred nor modified. Finite-panel improvements are not recall claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

SOURCE_PATHS = ("probes/readout_norm.py", "src/artifacts/git_identity.py", "src/artifacts/json_values.py", "src/artifacts/__init__.py")


def median_norm_factors(effective_output_rows: torch.Tensor) -> torch.Tensor:
    if effective_output_rows.ndim != 2 or min(effective_output_rows.shape) == 0:
        raise ValueError("effective output rows must be a nonempty matrix")
    if not effective_output_rows.is_floating_point() or not torch.isfinite(effective_output_rows).all():
        raise ValueError("effective output rows must be finite floating point")
    norms = effective_output_rows.detach().cpu().double().norm(dim=1)
    if (norms <= 0).any():
        raise ValueError("zero-norm output row has no finite normalization")
    return norms.median() / norms


def rescale_selected_logits(
    scores: torch.Tensor, token_ids: torch.Tensor, factors: torch.Tensor,
) -> torch.Tensor:
    """Clone scores, modify only explicit output columns using FP64 multiplication."""
    if not scores.is_floating_point() or scores.ndim < 1:
        raise ValueError("scores must have a floating vocabulary dimension")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long or token_ids.numel() == 0:
        raise ValueError("selected IDs must be a nonempty long vector")
    if factors.ndim != 1 or factors.numel() != token_ids.numel() or not factors.is_floating_point():
        raise ValueError("one floating factor is required per selected ID")
    ids = token_ids.to(device=scores.device)
    if ids.unique().numel() != ids.numel() or (ids < 0).any() or (ids >= scores.shape[-1]).any():
        raise ValueError("selected IDs must be unique vocabulary columns")
    if not torch.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("norm factors must be positive and finite")
    result = scores.clone()
    result[..., ids] = (scores[..., ids].double() * factors.to(scores.device, torch.float64)).to(scores.dtype)
    return result


def qualify(input_path: Path, output: Path, *, source_receipt: Path | None = None) -> None:
    from src.artifacts import publish_json_exclusive
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity, SourceIdentityError

    identity = capture_source_identity(SOURCE_PATHS)
    raw = input_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if source_receipt is not None:
        previous = json.loads(source_receipt.read_text())
        if previous.get("schema") != "readout_norm.qualified_arrays.v1":
            raise SourceIdentityError("historical/unsupported for continuation: legacy norm receipt")
        verify_source_identity(previous.get("source_identity", {}), required_paths=SOURCE_PATHS)
        if previous.get("input_sha256") != digest:
            raise SourceIdentityError("historical/unsupported for continuation: input bytes changed")
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(raw)
    if set(data) != {"effective_output_rows", "scores", "token_ids"}:
        raise ValueError("input needs explicit output rows, scores and token_ids")
    if not isinstance(data["token_ids"], list) or any(type(i) is not int for i in data["token_ids"]):
        raise ValueError("token IDs must be integers, not booleans or floats")
    rows = torch.tensor(data["effective_output_rows"], dtype=torch.float64)
    scores = torch.tensor(data["scores"], dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("qualification scores must be finite")
    ids = torch.tensor(data["token_ids"], dtype=torch.long)
    factors = median_norm_factors(rows)
    changed = rescale_selected_logits(scores, ids, factors)
    if input_path.read_bytes() != raw:
        raise ValueError("input changed during qualification")
    verify_source_identity(identity, required_paths=SOURCE_PATHS)
    publish_json_exclusive(output, {
        "schema": "readout_norm.qualified_arrays.v1", "source_identity": identity,
        "input_sha256": digest, "input_path": str(input_path.resolve()),
        "token_ids": ids.tolist(), "factors": factors.tolist(), "scores": changed.tolist(),
        "scope": "explicit-array FP32 output-only rescaling; no model, rollout or owner claim",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    args = parser.parse_args()
    qualify(args.input, args.output, source_receipt=args.source_receipt)


if __name__ == "__main__":
    main()
