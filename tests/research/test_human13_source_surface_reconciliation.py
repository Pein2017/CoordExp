from __future__ import annotations

from scripts.research.human13_hf_shared_surface import HFSharedSurfaceIdentity
from scripts.research.human13_rp_crossover_witness import SealedSourceDecode
from scripts.research.human13_source_surface_reconciliation import (
    SourceSurfaceReconciliationRequest,
    reconcile_source_surface,
)
from src.artifacts.json_values import json_sha256


def _identity() -> HFSharedSurfaceIdentity:
    return HFSharedSurfaceIdentity(
        checkpoint_payload_sha256="a" * 64,
        model_object_id=17,
        parameter_state_sha256="b" * 64,
        adapter_sha256="c" * 64,
        embedding_delta_sha256="d" * 64,
        dtype="bfloat16",
        attention_backend="flash_attention_2",
        model_mode="eval",
        tokenizer_sha256="e" * 64,
        prompt_sha256=json_sha256([10]),
        image_sha256="f" * 64,
        use_cache=False,
    )


def _request(*, check) -> SourceSurfaceReconciliationRequest:
    return SourceSurfaceReconciliationRequest(
        training_identity=_identity(),
        source_runtime_identities=(
            {
                "backend": "hf",
                "batch_size": 1,
                "observed_model_dtype_names": ["torch.float32"],
                "observed_attn_implementation": "sdpa",
            },
            {
                "backend": "hf",
                "batch_size": 1,
                "observed_model_dtype_names": ["torch.float32"],
                "observed_attn_implementation": "sdpa",
            },
        ),
        source_checkpoint_payload_sha256s=("a" * 64, "a" * 64),
        source_checkpoint_paths=("/source/checkpoint", "/source/checkpoint"),
        training_checkpoint_path="/source/checkpoint",
        source_adapter_sha256s=("c" * 64, "c" * 64),
        source_embedding_delta_sha256s=("d" * 64, "d" * 64),
        source_base_model_paths=("/source/base", "/source/base"),
        training_base_model_path="/source/base",
        source_manifest_sha256s=("1" * 64, "1" * 64),
        manifest_image_sha256="f" * 64,
        source_tokenizer_sha256s=("e" * 64, "e" * 64),
        source_prompt_sha256s=(json_sha256([10]), json_sha256([10])),
        source_image_sha256s=("f" * 64, "f" * 64),
        manifest_sha256="1" * 64,
        image_id=1584,
        source_audit_sha256s=(
            (1.0, json_sha256({"rp": 1.0})),
            (1.1, json_sha256({"rp": 1.1})),
        ),
        decodes=(
            SealedSourceDecode(1584, 1.0, (10,), (1, 2), ()),
            SealedSourceDecode(1584, 1.1, (10,), (1, 2), ()),
        ),
        check=check,
    )


def test_exact_surface_reconciliation_admits_and_binds_counts() -> None:
    calls: list[str] = []
    receipt = reconcile_source_surface(
        _request(check=lambda: calls.append("checked") or 0)
    )

    assert receipt.admitted is True
    assert receipt.checked_decode_count == 2
    assert receipt.checked_token_count == 4
    assert receipt.mismatch_count == 0
    assert receipt.failure_reason is None
    assert calls == ["checked"]
    assert receipt.content_sha256 == receipt.content_sha256
    assert receipt.to_dict()["source_surface"] == "gpu1:fp32/sdpa/batch1"
    assert receipt.to_dict()["training_surface"] == (
        "gpu0:bfloat16/flash_attention_2"
    )


def test_teacher_forced_token_mismatch_is_typed_non_admission() -> None:
    receipt = reconcile_source_surface(_request(check=lambda: 1))

    assert receipt.admitted is False
    assert receipt.mismatch_count == 1
    assert receipt.failure_reason == "teacher_forced_greedy_changed_tokens=1"


def test_surface_identity_drift_fails_before_checker() -> None:
    request = _request(check=lambda: (_ for _ in ()).throw(AssertionError("called")))
    request = SourceSurfaceReconciliationRequest(
        **{
            **request.__dict__,
            "source_runtime_identities": (
                {
                    **request.source_runtime_identities[0],
                    "observed_attn_implementation": "flash_attention_2",
                },
                request.source_runtime_identities[1],
            ),
        }
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is False
    assert receipt.mismatch_count == 1
    assert "source runtime identity" in (receipt.failure_reason or "")


def test_checker_exception_is_failure_receipt_without_admission() -> None:
    def check() -> int:
        raise RuntimeError("margin surface mismatch")

    receipt = reconcile_source_surface(_request(check=check))

    assert receipt.admitted is False
    assert receipt.mismatch_count == 1
    assert receipt.failure_reason == (
        "checker_error:RuntimeError: margin surface mismatch"
    )


def test_unserializable_runtime_identity_is_fail_closed() -> None:
    request = _request(check=lambda: (_ for _ in ()).throw(AssertionError("called")))
    request = SourceSurfaceReconciliationRequest(
        **{
            **request.__dict__,
            "source_runtime_identities": (
                {
                    **request.source_runtime_identities[0],
                    "model_identity": object(),
                },
                request.source_runtime_identities[1],
            ),
        }
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is False
    assert "not JSON-addressable" in (receipt.failure_reason or "")


def test_source_image_digest_must_bind_manifest_image() -> None:
    request = _request(check=lambda: (_ for _ in ()).throw(AssertionError("called")))
    request = SourceSurfaceReconciliationRequest(
        **{
            **request.__dict__,
            "source_image_sha256s": ("0" * 64, "f" * 64),
        }
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is False
    assert "image identity" in (receipt.failure_reason or "")
