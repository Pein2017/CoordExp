"""Regression guard for the two backend helpers this script lost at 767e57f5e.

`src/inference/backend.py` deleted `canonical_float32_logprob` and
`_normalized_attested_model_identity_for_runtime_comparison`; both lazy imports in
`run_sampled_rescue_transition.py` kept targeting them, so the donor-identity path
raised ImportError and the first-free-token path remisdiagnosed that ImportError as
"first free token logprob is not finite float32". The helpers now live in the script
itself (single consumer) and this exercises both directly.
"""

from __future__ import annotations

import struct

import pytest

from src.common.errors import RuntimeContractError
from scripts.research.run_sampled_rescue_transition import (
    _canonical_float32_logprob,
    _first_free_token_evidence,
    _normalized_attested_model_identity_for_runtime_comparison,
)


def test_canonical_float32_logprob_rounds_to_binary32_and_rejects_non_finite() -> None:
    value = -0.1234567890123
    canonical = _canonical_float32_logprob(value)

    assert canonical == struct.unpack(">f", struct.pack(">f", value))[0]
    assert canonical != value

    with pytest.raises(RuntimeContractError) as exc_info:
        _canonical_float32_logprob(float("nan"), context={"step_index": 0})

    assert exc_info.value.code == "backend_receipt.non_finite_selected_logprob"


def _adapter_identity(root: str, *, rank: int = 16) -> dict[str, object]:
    return {
        "adapter": {
            "adapter_path": f"{root}/adapter",
            "adapter_payload_evidence": {
                "config_path": f"{root}/adapter/config.json",
                "tensor_path": f"{root}/adapter/model.safetensors",
            },
            "rank": rank,
        }
    }


def test_normalized_attested_model_identity_masks_only_payload_locations() -> None:
    relocated = _normalized_attested_model_identity_for_runtime_comparison(
        _adapter_identity("/mnt/one")
    )
    original = _normalized_attested_model_identity_for_runtime_comparison(
        _adapter_identity("/mnt/two")
    )

    assert relocated == original
    assert relocated["adapter"]["rank"] == 16
    assert relocated["adapter"]["adapter_path"].startswith("<relocatable")
    assert relocated["adapter"]["adapter_payload_evidence"]["tensor_path"].startswith(
        "<relocatable"
    )

    changed = _normalized_attested_model_identity_for_runtime_comparison(
        _adapter_identity("/mnt/two", rank=32)
    )
    assert changed != original

    with pytest.raises(RuntimeContractError) as exc_info:
        _normalized_attested_model_identity_for_runtime_comparison(
            _adapter_identity("relative")
        )

    assert exc_info.value.code == (
        "backend_sampling.attestation_model_payload_path_invalid"
    )


def test_first_free_token_evidence_no_longer_masks_helper_failures() -> None:
    class Result:
        def __init__(self, trace: object) -> None:
            self.token_trace = trace

    good = Result([{"step_index": 0, "token_id": 7, "is_pad": False, "logprob": -0.25}])
    evidence = _first_free_token_evidence(good, [7])

    assert evidence == {"token_id": 7, "log_probability_float32": -0.25}

    with pytest.raises(SystemExit) as exc_info:
        _first_free_token_evidence(
            Result([{"step_index": 0, "token_id": 7, "is_pad": False, "logprob": float("nan")}]),
            [7],
        )

    assert "not finite float32" in str(exc_info.value)
