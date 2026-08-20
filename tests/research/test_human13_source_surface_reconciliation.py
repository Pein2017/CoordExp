from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.research.human13_hf_shared_surface import HFSharedSurfaceIdentity
from scripts.research.human13_rp_crossover_witness import SealedSourceDecode
from scripts.research.human13_source_surface_reconciliation import (
    CanonicalSourceBaselineReceipt,
    CoordinateAliasReconciliation,
    SourceSurfaceReconciliationReceipt,
    SourceSurfaceReconciliationRequest,
    reconcile_coordinate_alias,
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


def _canonical_baselines() -> tuple[CanonicalSourceBaselineReceipt, ...]:
    return tuple(
        CanonicalSourceBaselineReceipt(
            surface=surface,
            repetition_penalty=rp,
            canonical_payload={
                "surface": surface,
                "repetition_penalty": rp,
                "generated_token_ids": [1, 2],
                "predictions": [],
            },
            owner_map={},
        )
        for surface, rp in (
            ("gpu1:fp32/sdpa/batch1", 1.0),
            ("gpu1:fp32/sdpa/batch1", 1.1),
            ("gpu0:bfloat16/flash_attention_2", 1.0),
            ("gpu0:bfloat16/flash_attention_2", 1.1),
        )
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
        canonical_baselines=_canonical_baselines(),
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


def test_training_image_digest_must_bind_source_manifest_image() -> None:
    request = _request(check=lambda: (_ for _ in ()).throw(AssertionError("called")))
    identity = HFSharedSurfaceIdentity(
        **{
            **request.training_identity.__dict__,
            "image_sha256": "0" * 64,
        }
    )
    request = SourceSurfaceReconciliationRequest(
        **{**request.__dict__, "training_identity": identity}
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is False
    assert "Source/training image_sha256 differs" in (receipt.failure_reason or "")


def _alias_kwargs(*, training_token: str = "<|coord_616|>") -> dict[str, Any]:
    return {
        "source_tokens": ("<|box|>", "<|coord_615|>", "<|end|>"),
        "training_tokens": ("<|box|>", training_token, "<|end|>"),
        "coordinate_roles": {1: ("gt:1584:2", "y1")},
        "source_boxes": {"gt:1584:2": (130.0, 630.0, 162.0, 722.0)},
        "training_boxes": {"gt:1584:2": (130.0, 631.0, 162.0, 722.0)},
        "gt_boxes": {"gt:1584:2": (130.0, 630.0, 162.0, 722.0)},
        "owner_match": {"gt:1584:2": "gt:1584:2"},
        "source_owner_rows": {"gt:1584:2": 0},
        "training_owner_rows": {"gt:1584:2": 0},
        "source_membership": {"gt:1584:2": "G"},
        "training_membership": {"gt:1584:2": "G"},
        "source_protected_g": ("gt:1584:2",),
        "training_protected_g": ("gt:1584:2",),
    }


def test_coordinate_alias_615_616_is_admitted_with_iou_evidence() -> None:
    receipt = reconcile_coordinate_alias(**_alias_kwargs())

    assert isinstance(receipt, CoordinateAliasReconciliation)
    assert receipt.admitted is True
    assert len(receipt.evidence) == 1
    evidence = receipt.evidence[0]
    assert evidence.delta_bin == 1
    assert evidence.owner_id == "gt:1584:2"
    assert evidence.source_iou > evidence.training_iou
    assert evidence.disposition == "metric_equivalent_coordinate_alias"


def test_coordinate_alias_inclusive_five_bins_passes() -> None:
    kwargs = _alias_kwargs(training_token="<|coord_620|>")
    kwargs["training_boxes"] = {"gt:1584:2": (130.0, 635.0, 162.0, 722.0)}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is True
    assert receipt.evidence[0].delta_bin == 5


def test_coordinate_alias_delta_six_fails_closed() -> None:
    kwargs = _alias_kwargs(training_token="<|coord_621|>")
    kwargs["training_boxes"] = {"gt:1584:2": (130.0, 636.0, 162.0, 722.0)}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "exceeds five" in (receipt.failure_reason or "")


def test_coordinate_alias_non_coordinate_difference_fails_closed() -> None:
    kwargs = _alias_kwargs()
    kwargs["training_tokens"] = ("<|other_box|>", "<|coord_616|>", "<|end|>")
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "non-coordinate" in (receipt.failure_reason or "")


def test_coordinate_alias_requires_canonical_token_wrapper() -> None:
    kwargs = _alias_kwargs()
    kwargs["source_tokens"] = ("<|coord_615", "<|end|>", "<|tail|>")
    kwargs["training_tokens"] = ("<|coord_616|>", "<|end|>", "<|tail|>")
    kwargs["coordinate_roles"] = {0: ("gt:1584:2", "y1")}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "non-coordinate" in (receipt.failure_reason or "")


def test_coordinate_alias_rejects_noncanonical_role_label() -> None:
    kwargs = _alias_kwargs()
    kwargs["coordinate_roles"] = {1: ("gt:1584:2", "category")}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert receipt.failure_reason == "coordinate role is not canonical"
    assert receipt.failure_evidence is not None


def test_coordinate_alias_rejects_noncanonical_leading_zero_token() -> None:
    kwargs = _alias_kwargs(training_token="<|coord_001|>")
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "non-coordinate" in (receipt.failure_reason or "")


def test_coordinate_alias_invalid_rectangle_fails_closed() -> None:
    kwargs = _alias_kwargs()
    kwargs["training_boxes"] = {"gt:1584:2": (130.0, 700.0, 162.0, 650.0)}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "legal rectangle" in (receipt.failure_reason or "")


def test_coordinate_alias_same_bins_but_owner_assignment_change_fails() -> None:
    kwargs = _alias_kwargs()
    kwargs["training_owner_rows"] = {"gt:1584:2": 1}
    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert "row assignment" in (receipt.failure_reason or "")


def test_coordinate_alias_two_row_owner_exchange_fails_even_with_small_deltas() -> None:
    receipt = reconcile_coordinate_alias(
        source_tokens=("<|coord_100|>", "<|coord_200|>"),
        training_tokens=("<|coord_105|>", "<|coord_195|>"),
        coordinate_roles={0: ("owner-a", "x1"), 1: ("owner-b", "x1")},
        source_boxes={
            "owner-a": (100.0, 100.0, 200.0, 200.0),
            "owner-b": (200.0, 100.0, 300.0, 200.0),
        },
        training_boxes={
            "owner-a": (105.0, 100.0, 200.0, 200.0),
            "owner-b": (195.0, 100.0, 300.0, 200.0),
        },
        gt_boxes={
            "owner-a": (100.0, 100.0, 200.0, 200.0),
            "owner-b": (200.0, 100.0, 300.0, 200.0),
        },
        owner_match={"owner-a": "owner-a", "owner-b": "owner-b"},
        source_owner_rows={"owner-a": 0, "owner-b": 1},
        training_owner_rows={"owner-a": 1, "owner-b": 0},
        source_membership={"owner-a": "G", "owner-b": "H"},
        training_membership={"owner-a": "G", "owner-b": "H"},
        source_protected_g=("owner-a",),
        training_protected_g=("owner-a",),
    )

    assert receipt.admitted is False
    assert "row assignment" in (receipt.failure_reason or "")


def test_owner_assignment_failure_publishes_compact_diagnostic_evidence() -> None:
    kwargs = _alias_kwargs()
    kwargs.update(
        {
            "source_token_ids": (101, 615, 102),
            "training_token_ids": (101, 616, 102),
            "repetition_penalty": 1.1,
            "training_boxes": {"gt:1584:4": (130.0, 630.0, 162.0, 722.0)},
            "training_owner_rows": {"gt:1584:4": 0},
            "training_membership": {"gt:1584:4": "H"},
            "training_protected_g": (),
        }
    )

    receipt = reconcile_coordinate_alias(**kwargs)

    assert receipt.admitted is False
    assert receipt.failure_reason == "canonical owner assignment differs"
    diagnostic = receipt.failure_evidence
    assert diagnostic is not None
    payload = cast(dict[str, Any], diagnostic.to_dict())
    assert payload["repetition_penalty"] == 1.1
    assert payload["source_owner_set"] == ["gt:1584:2"]
    assert payload["training_owner_set"] == ["gt:1584:4"]
    assert payload["symmetric_owner_set_difference"] == [
        "gt:1584:2",
        "gt:1584:4",
    ]
    assert payload["source_owner_rows"] == [["gt:1584:2", 0]]
    assert payload["training_owner_rows"] == [["gt:1584:4", 0]]
    mismatches = cast(list[dict[str, Any]], payload["token_mismatches"])
    assert mismatches[0]["source_token_id"] == 615
    assert mismatches[0]["training_token_id"] == 616
    affected_rows = cast(list[dict[str, Any]], payload["affected_rows"])
    affected = {
        row["owner_id"]: row for row in affected_rows
    }
    assert affected["gt:1584:2"]["source_bbox"] == [
        130.0,
        630.0,
        162.0,
        722.0,
    ]
    assert affected["gt:1584:4"]["training_bbox"] == [
        130.0,
        630.0,
        162.0,
        722.0,
    ]


def test_coordinate_alias_failure_receipt_reload_and_tamper_fail_closed() -> None:
    from scripts.research.human13_source_surface_reconciliation import (
        CoordinateAliasReconciliation,
    )

    kwargs = _alias_kwargs(training_token="<|coord_621|>")
    kwargs.update(
        {
            "source_token_ids": (101, 615, 102),
            "training_token_ids": (101, 621, 102),
            "repetition_penalty": 1.0,
            "training_boxes": {"gt:1584:2": (130.0, 636.0, 162.0, 722.0)},
        }
    )
    receipt = reconcile_coordinate_alias(**kwargs)
    restored = CoordinateAliasReconciliation.from_dict(receipt.to_dict())
    assert restored.to_dict() == receipt.to_dict()
    tampered = cast(dict[str, Any], receipt.to_dict())
    tampered["failure_evidence"]["token_mismatches"][0]["training_token_id"] = 620
    with __import__("pytest").raises((TypeError, ValueError), match="hash|schema"):
        CoordinateAliasReconciliation.from_dict(tampered)


def test_surface_receipt_publishes_coordinate_alias_evidence() -> None:
    alias = reconcile_coordinate_alias(**_alias_kwargs())
    request = _request(check=lambda: 0)
    request = SourceSurfaceReconciliationRequest(
        **{**request.__dict__, "coordinate_alias_check": lambda: alias}
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is True
    published = receipt.to_dict()["coordinate_alias"]
    assert isinstance(published, dict)
    assert published["evidence"][0]["delta_bin"] == 1


def test_source_surface_failure_receipt_reload_and_content_tamper_fail_closed() -> None:
    alias = reconcile_coordinate_alias(
        **_alias_kwargs(training_token="<|coord_621|>"),
        source_token_ids=(101, 615, 102),
        training_token_ids=(101, 621, 102),
        repetition_penalty=1.1,
    )
    request = _request(check=lambda: 0)
    request = SourceSurfaceReconciliationRequest(
        **{**request.__dict__, "coordinate_alias_check": lambda: alias}
    )

    receipt = reconcile_source_surface(request)
    assert receipt.admitted is True
    assert receipt.cross_surface_disposition == "diagnostic_only"
    restored = SourceSurfaceReconciliationReceipt.from_dict(receipt.to_dict())
    assert restored.to_dict() == receipt.to_dict()
    tampered = receipt.to_dict()
    tampered["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash"):
        SourceSurfaceReconciliationReceipt.from_dict(tampered)


def test_nested_rp1_divergence_retains_all_four_canonical_source_baselines() -> None:
    baselines = tuple(
        CanonicalSourceBaselineReceipt(
            surface=surface,
            repetition_penalty=rp,
            canonical_payload={
                "surface": surface,
                "repetition_penalty": rp,
                "generated_token_ids": [100, 200 + index],
                "predictions": [
                    {
                        "generated_order": 0,
                        "description": "person",
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "token_start": 0,
                        "token_end": 2,
                    }
                ],
            },
            owner_map={
                "gt:1584:2": {
                    "generated_order": 0,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            },
        )
        for index, (surface, rp) in enumerate(
            (
                ("gpu1:fp32/sdpa/batch1", 1.0),
                ("gpu1:fp32/sdpa/batch1", 1.1),
                ("gpu0:bfloat16/flash_attention_2", 1.0),
                ("gpu0:bfloat16/flash_attention_2", 1.1),
            )
        )
    )
    nested_rp1_divergence = CoordinateAliasReconciliation(
        admitted=False,
        mismatch_count=1,
        failure_reason="non-coordinate token differs at position 5",
    )
    request = _request(check=lambda: 0)
    request = SourceSurfaceReconciliationRequest(
        **{
            **request.__dict__,
            "coordinate_alias_check": lambda: nested_rp1_divergence,
            "canonical_baselines": baselines,
        }
    )

    receipt = reconcile_source_surface(request)

    assert receipt.admitted is True
    assert receipt.cross_surface_disposition == "diagnostic_only"
    published = cast(
        list[dict[str, Any]], receipt.to_dict()["canonical_baselines"]
    )
    assert [
        (item["surface"], item["repetition_penalty"]) for item in published
    ] == [
        ("gpu1:fp32/sdpa/batch1", 1.0),
        ("gpu1:fp32/sdpa/batch1", 1.1),
        ("gpu0:bfloat16/flash_attention_2", 1.0),
        ("gpu0:bfloat16/flash_attention_2", 1.1),
    ]
    for expected, observed in zip(baselines, published, strict=True):
        assert observed["canonical_payload"] == expected.canonical_payload
        assert observed["canonical_payload_sha256"] == json_sha256(
            expected.canonical_payload
        )
        assert observed["owner_map"] == expected.owner_map
        assert observed["owner_map_sha256"] == json_sha256(expected.owner_map)
    restored = SourceSurfaceReconciliationReceipt.from_dict(receipt.to_dict())
    assert restored.to_dict() == receipt.to_dict()
    incomplete = receipt.to_dict()
    incomplete["canonical_baselines"] = published[:-1]
    with pytest.raises(ValueError, match="bind fp32/BF16"):
        SourceSurfaceReconciliationReceipt.from_dict(incomplete)
    payload_tamper = receipt.to_dict()
    tampered_baselines = cast(
        list[dict[str, Any]], payload_tamper["canonical_baselines"]
    )
    tampered_payload = cast(
        dict[str, Any], tampered_baselines[3]["canonical_payload"]
    )
    tampered_payload["generated_token_ids"] = [999]
    with pytest.raises(ValueError, match="payload hash"):
        SourceSurfaceReconciliationReceipt.from_dict(payload_tamper)


def test_new_four_cell_receipt_uses_v3_without_masquerading_as_v2() -> None:
    receipt = reconcile_source_surface(_request(check=lambda: 0))

    assert receipt.to_dict()["schema_version"] == (
        "human13_source_surface_reconciliation.v3"
    )


def test_immutable_v6_legacy_v2_receipt_still_round_trips_strictly() -> None:
    phase_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/"
        "one-image-successor-20260820T-preflight-bf16-native-v6/receipts/"
        "007-source_surface_reconciliation.json"
    )
    phase = cast(dict[str, Any], json.loads(phase_path.read_text(encoding="utf-8")))
    evidence = cast(dict[str, Any], phase["evidence"])
    payload = cast(dict[str, Any], evidence["receipt"])

    restored = SourceSurfaceReconciliationReceipt.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.to_dict()["schema_version"] == (
        "human13_source_surface_reconciliation.v2"
    )
    assert "canonical_baselines" not in restored.to_dict()
    legacy_marked_v3 = dict(payload)
    legacy_marked_v3["schema_version"] = (
        "human13_source_surface_reconciliation.v3"
    )
    with pytest.raises(ValueError, match="fields differ"):
        SourceSurfaceReconciliationReceipt.from_dict(legacy_marked_v3)
    v2_with_v3_fields = dict(payload)
    v3 = reconcile_source_surface(_request(check=lambda: 0)).to_dict()
    v2_with_v3_fields["canonical_baselines"] = v3["canonical_baselines"]
    with pytest.raises(ValueError, match="fields differ"):
        SourceSurfaceReconciliationReceipt.from_dict(v2_with_v3_fields)
