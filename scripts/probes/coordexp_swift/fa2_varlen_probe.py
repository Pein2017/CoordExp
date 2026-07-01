#!/usr/bin/env python3
"""Tiny FlashAttention 2 explicit-varlen branch probe.

This intentionally avoids loading Qwen. It exercises the installed
Transformers `_flash_attention_forward` branch with synthetic packed tensors and
monkeypatches only the lazy flash-attn function import so the receipt records
which branch was reached.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


RECEIPT_DIR = Path("outputs/probes/coordexp_swift/fa2_varlen")
RECEIPT_NAME = "fa2_varlen_probe_receipt.json"


@dataclass(frozen=True)
class PackedSegment:
    name: str
    start_position: int
    end_position: int

    @property
    def length(self) -> int:
        return self.end_position - self.start_position


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Transformers FA2 padding-free explicit-varlen branch."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Tensor device for the tiny synthetic call.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Tensor dtype; FA2 dispatch expects bf16 or fp16.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RECEIPT_DIR),
        help="Directory for the JSON receipt.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise AssertionError(f"Unsupported dtype: {name}")


def _tensor_to_list(value: torch.Tensor | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value.detach().cpu().tolist()]


def _build_segments() -> list[PackedSegment]:
    cursor = 0
    lengths = (2, 3, 1)
    segments: list[PackedSegment] = []
    for index, length in enumerate(lengths):
        segments.append(
            PackedSegment(
                name=f"segment_{index}",
                start_position=cursor,
                end_position=cursor + length,
            )
        )
        cursor += length
    return segments


def _cumulative_boundaries(segments: list[PackedSegment]) -> list[int]:
    boundaries = [0]
    for segment in segments:
        if segment.start_position != boundaries[-1]:
            raise AssertionError(
                "PackedSegment boundaries are not contiguous: "
                f"{segment.start_position} != {boundaries[-1]}"
            )
        boundaries.append(segment.end_position)
    return boundaries


def _validate_installed_transformers_branch(module: Any) -> dict[str, Any]:
    forward = module._flash_attention_forward
    signature = inspect.signature(forward)
    required_parameters = {
        "attention_mask",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    }
    missing_parameters = sorted(required_parameters - set(signature.parameters))
    if missing_parameters:
        raise RuntimeError(
            "Installed Transformers _flash_attention_forward signature is missing "
            f"expected explicit-varlen parameters: {missing_parameters}"
        )

    source_lines, start_line = inspect.getsourcelines(forward)
    source = "".join(source_lines)
    expected_fragments = (
        "is_fa_with_varlen_kwargs = all(",
        "kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)",
        "elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:",
        "out = flash_varlen_fn(",
        "out = flash_fn(query_states, key_states, value_states, **flash_kwargs)",
    )
    missing_fragments = [fragment for fragment in expected_fragments if fragment not in source]
    if missing_fragments:
        raise RuntimeError(
            "Installed Transformers _flash_attention_forward branch source does not "
            "match the expected explicit-varlen shape. Missing fragments: "
            f"{missing_fragments}. Refusing to produce a false proof."
        )

    return {
        "source_file": inspect.getsourcefile(forward),
        "source_start_line": start_line,
        "signature": str(signature),
        "explicit_varlen_branch_source_fragments_present": True,
    }


def _availability_receipt() -> dict[str, Any]:
    import flash_attn
    import transformers
    from transformers.utils import is_flash_attn_2_available

    return {
        "transformers_version": transformers.__version__,
        "transformers_file": getattr(transformers, "__file__", None),
        "flash_attn_version": getattr(flash_attn, "__version__", None),
        "flash_attn_file": getattr(flash_attn, "__file__", None),
        "is_flash_attn_2_available": bool(is_flash_attn_2_available()),
    }


def _run_probe(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    import transformers.modeling_flash_attention_utils as flash_utils

    branch_info = _validate_installed_transformers_branch(flash_utils)
    availability = _availability_receipt()
    if not availability["is_flash_attn_2_available"]:
        raise RuntimeError("transformers.utils.is_flash_attn_2_available() returned false")

    segments = _build_segments()
    boundaries = _cumulative_boundaries(segments)
    lengths = [segment.length for segment in segments]
    max_length = max(lengths)
    total_length = boundaries[-1]

    cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260630)
    query_states = torch.randn(
        (1, total_length, 2, 8), generator=generator, device=device, dtype=dtype
    )
    key_states = torch.randn_like(query_states)
    value_states = torch.randn_like(query_states)

    observed: dict[str, Any] = {
        "flash_fn_called": False,
        "flash_varlen_fn_called": False,
        "pad_fn_called": False,
        "unpad_fn_called": False,
        "varlen_call": None,
    }

    def fake_flash_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["flash_fn_called"] = True
        raise AssertionError("Unexpected ordinary flash attention path reached")

    def fake_flash_varlen_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        observed["flash_varlen_fn_called"] = True
        observed["varlen_call"] = {
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "q_device": str(q.device),
            "q_dtype": str(q.dtype),
            "cu_seqlens_q": _tensor_to_list(cu_seqlens_q),
            "cu_seqlens_k": _tensor_to_list(cu_seqlens_k),
            "max_seqlen_q": None if max_seqlen_q is None else int(max_seqlen_q),
            "max_seqlen_k": None if max_seqlen_k is None else int(max_seqlen_k),
            "flash_kwargs": {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in kwargs.items()
            },
        }
        return torch.zeros_like(q)

    def fake_pad_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["pad_fn_called"] = True
        raise AssertionError("Unexpected padding-mask pad path reached")

    def fake_unpad_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["unpad_fn_called"] = True
        raise AssertionError("Unexpected padding-mask unpad path reached")

    def fake_process_flash_kwargs_fn(**kwargs: Any) -> dict[str, Any]:
        return {
            "causal": bool(kwargs["is_causal"]),
            "dropout_p": float(kwargs["dropout"]),
            "softmax_scale": kwargs.get("softmax_scale"),
        }

    def fake_lazy_import_flash_attention(implementation: str | None = None) -> tuple[Any, Any]:
        if implementation != "flash_attention_2":
            raise AssertionError(f"Unexpected implementation requested: {implementation!r}")
        return (
            (fake_flash_fn, fake_flash_varlen_fn, fake_pad_fn, fake_unpad_fn),
            fake_process_flash_kwargs_fn,
        )

    original_lazy_import = flash_utils.lazy_import_flash_attention
    flash_utils.lazy_import_flash_attention = fake_lazy_import_flash_attention
    try:
        output = flash_utils._flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=None,
            query_length=total_length,
            is_causal=True,
            dropout=0.0,
            position_ids=None,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_length,
            max_length_k=max_length,
            target_dtype=dtype,
            implementation="flash_attention_2",
        )
    finally:
        flash_utils.lazy_import_flash_attention = original_lazy_import

    varlen_call = observed["varlen_call"]
    observed_branch = (
        "padding_free_varlen"
        if observed["flash_varlen_fn_called"]
        and not observed["flash_fn_called"]
        and not observed["pad_fn_called"]
        and not observed["unpad_fn_called"]
        else "unexpected"
    )

    receipt_assertions = {
        "attention_mask_is_none": True,
        "branch_is_padding_free_varlen": observed_branch == "padding_free_varlen",
        "cu_seq_lens_q_non_null": varlen_call is not None and varlen_call["cu_seqlens_q"] is not None,
        "cu_seq_lens_k_non_null": varlen_call is not None and varlen_call["cu_seqlens_k"] is not None,
        "max_length_q_non_null": varlen_call is not None and varlen_call["max_seqlen_q"] is not None,
        "max_length_k_non_null": varlen_call is not None and varlen_call["max_seqlen_k"] is not None,
        "cu_seq_lens_q_matches_packed_boundaries": varlen_call is not None
        and varlen_call["cu_seqlens_q"] == boundaries,
        "cu_seq_lens_k_matches_packed_boundaries": varlen_call is not None
        and varlen_call["cu_seqlens_k"] == boundaries,
        "max_length_q_matches_packed_segments": varlen_call is not None
        and varlen_call["max_seqlen_q"] == max_length,
        "max_length_k_matches_packed_segments": varlen_call is not None
        and varlen_call["max_seqlen_k"] == max_length,
        "output_shape_matches_input": list(output.shape) == list(query_states.shape),
    }
    failed_assertions = [
        name for name, passed in receipt_assertions.items() if not bool(passed)
    ]
    if failed_assertions:
        raise AssertionError(f"FA2 varlen probe failed assertions: {failed_assertions}")

    return {
        "probe": "coordexp_swift_fa2_varlen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "availability": availability,
        "installed_transformers_branch": branch_info,
        "resolved_attention_implementation": "flash_attention_2",
        "model_dtype": str(dtype),
        "device": str(device),
        "autocast": {
            "cuda_enabled": bool(torch.is_autocast_enabled("cuda")),
            "cpu_enabled": bool(torch.is_autocast_enabled("cpu")),
        },
        "segments": [asdict(segment) | {"length": segment.length} for segment in segments],
        "segment_count": len(segments),
        "segment_boundaries": boundaries,
        "cumulative_sequence_lengths": boundaries,
        "max_length_q": max_length,
        "max_length_k": max_length,
        "attention_mask": None,
        "position_ids": None,
        "branch_evidence_from_explicit_varlen_kwargs": True,
        "observed_branch": observed_branch,
        "observed_call": observed,
        "assertions": receipt_assertions,
    }


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    receipt = _run_probe(device=device, dtype=dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RECEIPT_NAME
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt_path": str(output_path), "status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
