#!/usr/bin/env python3
"""Record the installed stock-sampler gap and supported custom-generation seam.

This bounded source probe makes no model call and is not CUDA attestation. Its
artifact only establishes why the local request-scoped sampling seam is needed
and which exact installed/local source implementations were inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.generation.utils import GenerationMixin

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_source_attestation() -> dict[str, Any]:
    from src.inference.backend import (
        CUSTOM_SAMPLER_IDENTITY,
        custom_generate,
        custom_sampler_code_hash,
    )

    stock_signature = inspect.signature(GenerationMixin._sample)
    stock_source = inspect.getsource(GenerationMixin._sample)
    generate_signature = inspect.signature(GenerationMixin.generate)
    generate_source = inspect.getsource(GenerationMixin.generate)
    custom_signature = inspect.signature(custom_generate)
    custom_source = inspect.getsource(custom_generate)
    stock_has_generator_parameter = "generator" in stock_signature.parameters
    stock_uses_unbound_multinomial = (
        "torch.multinomial(probs, num_samples=1)" in stock_source
        and "generator=" not in stock_source
    )
    custom_has_request_generators_parameter = (
        "request_generators" in custom_signature.parameters
    )
    generate_has_custom_callable_parameter = (
        "custom_generate" in generate_signature.parameters
    )
    generate_has_use_model_defaults_parameter = (
        "use_model_defaults" in generate_signature.parameters
    )
    generate_dispatches_callable = (
        "isinstance(custom_generate, Callable)" in generate_source
        and "decoding_method = custom_generate" in generate_source
    )
    if stock_has_generator_parameter or not stock_uses_unbound_multinomial:
        raise RuntimeError(
            "installed stock GenerationMixin._sample no longer matches the recorded gap"
        )
    if not custom_has_request_generators_parameter:
        raise RuntimeError(
            "local custom_generate does not expose explicit request_generators"
        )
    if not (
        generate_has_custom_callable_parameter
        and generate_has_use_model_defaults_parameter
        and generate_dispatches_callable
    ):
        raise RuntimeError(
            "installed GenerationMixin.generate does not expose the supported "
            "custom callable and explicit model-default controls"
        )
    return {
        "schema_version": "request_scoped_sampling_source_attestation.v1",
        "scope": "source_only_no_model_call_not_cuda_attestation",
        "installed_runtime": {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        },
        "stock_sampling_path": {
            "owner": "transformers.generation.utils.GenerationMixin._sample",
            "signature": str(stock_signature),
            "source_sha256": _sha256_text(stock_source),
            "has_generator_parameter": stock_has_generator_parameter,
            "uses_unbound_multinomial": stock_uses_unbound_multinomial,
        },
        "supported_custom_generation_seam": {
            "owner": "transformers.generation.utils.GenerationMixin.generate",
            "signature": str(generate_signature),
            "source_sha256": _sha256_text(generate_source),
            "has_custom_generate_parameter": generate_has_custom_callable_parameter,
            "has_use_model_defaults_parameter": generate_has_use_model_defaults_parameter,
            "dispatches_callable_as_decoding_method": generate_dispatches_callable,
        },
        "custom_sampling_path": {
            "identity": CUSTOM_SAMPLER_IDENTITY,
            "owner": "src.inference.backend.custom_generate",
            "signature": str(custom_signature),
            "source_sha256": _sha256_text(custom_source),
            "implementation_source_sha256": custom_sampler_code_hash(),
            "has_request_generators_parameter": custom_has_request_generators_parameter,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Standard output is always emitted.",
    )
    args = parser.parse_args()
    payload = build_source_attestation()
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
