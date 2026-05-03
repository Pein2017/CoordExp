from __future__ import annotations

from dataclasses import replace

import pytest

from src.detection.packing import (
    PackingFingerprintInput,
    PackingFingerprintMetadata,
    PackingProfile,
    StaticSftPackingFingerprintRequest,
    build_stage1_static_sft_packing_fingerprint,
    build_packing_fingerprint,
    build_static_sft_packing_fingerprint,
    require_static_sft_packing_eligibility,
)


def _base_input() -> PackingFingerprintInput:
    return PackingFingerprintInput(
        template_id="stage1_json_pretty",
        template_version=1,
        prompt_profile="stage1_detection+dense_detection+summary",
        tokenizer_id="Qwen3-VL-2B-Instruct-coordexp",
        object_ordering="sorted",
        objective_variant="sorted_sft",
        state_weighting_policy="none",
        normalization_policy="token_mean",
        loss_mask_version="template_spans_v1",
        preprocessing_version="detection_preprocess_v1",
        profile=PackingProfile(
            mode="static",
            packing_length=2048,
            runtime_flags={
                "drop_last": False,
                "nested": {
                    "bucket_size": 32,
                    "lossless_boundaries": True,
                },
            },
        ),
    )


def test_packing_fingerprint_is_deterministic_for_identical_inputs() -> None:
    fingerprint_a = build_packing_fingerprint(_base_input())
    fingerprint_b = build_packing_fingerprint(_base_input())

    assert fingerprint_a.sha256 == fingerprint_b.sha256
    assert fingerprint_a.canonical_json == fingerprint_b.canonical_json
    assert fingerprint_a.metadata.template_id == "stage1_json_pretty"
    assert fingerprint_a.metadata.packing_mode == "static"
    assert fingerprint_a.metadata.packing_length == 2048


def test_packing_fingerprint_is_stable_under_runtime_flag_key_order() -> None:
    profile_a = PackingProfile(
        mode="static",
        packing_length=2048,
        runtime_flags={
            "drop_last": False,
            "nested": {
                "bucket_size": 32,
                "lossless_boundaries": True,
            },
        },
    )
    profile_b = PackingProfile(
        mode="static",
        packing_length=2048,
        runtime_flags={
            "nested": {
                "lossless_boundaries": True,
                "bucket_size": 32,
            },
            "drop_last": False,
        },
    )

    fingerprint_a = build_packing_fingerprint(
        replace(_base_input(), profile=profile_a)
    )
    fingerprint_b = build_packing_fingerprint(
        replace(_base_input(), profile=profile_b)
    )

    assert fingerprint_a.sha256 == fingerprint_b.sha256
    assert fingerprint_a.canonical_json == fingerprint_b.canonical_json


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "template_id",
            lambda base: replace(base, template_id="compact_full"),
        ),
        (
            "template_version",
            lambda base: replace(base, template_version=2),
        ),
        (
            "tokenizer_id",
            lambda base: replace(
                base,
                tokenizer_id="Qwen3-VL-7B-Instruct-coordexp",
            ),
        ),
        (
            "prompt_profile",
            lambda base: replace(
                base,
                prompt_profile="stage1_detection+compact_detection+summary",
            ),
        ),
        (
            "object_ordering",
            lambda base: replace(base, object_ordering="random_permutation"),
        ),
        (
            "objective_variant",
            lambda base: replace(base, objective_variant="random_order_sft"),
        ),
        (
            "state_weighting_policy",
            lambda base: replace(
                base,
                state_weighting_policy="uniform_permutation",
            ),
        ),
        (
            "normalization_policy",
            lambda base: replace(
                base,
                normalization_policy="semantic_image_bucket_balanced",
            ),
        ),
        (
            "loss_mask_version",
            lambda base: replace(base, loss_mask_version="template_spans_v2"),
        ),
        (
            "preprocessing_version",
            lambda base: replace(
                base,
                preprocessing_version="detection_preprocess_v2",
            ),
        ),
        (
            "packing_mode",
            lambda base: replace(
                base,
                profile=replace(base.profile, mode="padding_free_packed"),
            ),
        ),
        (
            "packing_length",
            lambda base: replace(
                base,
                profile=replace(base.profile, packing_length=4096),
            ),
        ),
    ],
)
def test_packing_fingerprint_changes_when_behaviorally_meaningful_fields_change(
    label: str,
    mutate,
) -> None:
    baseline = build_packing_fingerprint(_base_input())
    changed = build_packing_fingerprint(mutate(_base_input()))

    assert changed.sha256 != baseline.sha256, label


def test_packing_profile_runtime_flags_are_deep_immutable() -> None:
    source_flags = {"nested": {"bucket_size": 32}, "items": [1, 2]}
    profile = PackingProfile(
        mode="static",
        packing_length=2048,
        runtime_flags=source_flags,
    )
    before = build_packing_fingerprint(replace(_base_input(), profile=profile))

    source_flags["nested"]["bucket_size"] = 64
    source_flags["items"].append(3)
    after = build_packing_fingerprint(replace(_base_input(), profile=profile))

    assert before.sha256 == after.sha256
    with pytest.raises(TypeError):
        profile.runtime_flags["nested"]["bucket_size"] = 64
    with pytest.raises(TypeError):
        before.metadata.runtime_flags["nested"]["bucket_size"] = 64


def test_packing_fingerprint_rejects_ambiguous_or_non_strict_json_runtime_flags() -> None:
    for bad_runtime_flags in ([("a", 1)], (("a", 1),)):
        with pytest.raises(TypeError, match="runtime_flags must be a mapping"):
            PackingProfile(
                mode="static",
                packing_length=2048,
                runtime_flags=bad_runtime_flags,
            )

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        PackingProfile(
            mode="static",
            packing_length=2048,
            runtime_flags={1: "int-key", "1": "string-key"},
        )

    with pytest.raises(ValueError, match="NaN or Infinity"):
        PackingProfile(
            mode="static",
            packing_length=2048,
            runtime_flags={"temperature": float("nan")},
        )

    with pytest.raises(ValueError, match="NaN or Infinity"):
        PackingProfile(
            mode="static",
            packing_length=2048,
            runtime_flags={"temperature": float("inf")},
        )


def test_packing_fingerprint_input_rejects_accidental_runtime_types() -> None:
    with pytest.raises(TypeError, match="template_id must be a string"):
        replace(_base_input(), template_id=123)

    with pytest.raises(TypeError, match="template_version must be an integer"):
        replace(_base_input(), template_version=True)

    with pytest.raises(TypeError, match="packing_length must be an integer"):
        replace(_base_input().profile, packing_length=True)

    with pytest.raises(TypeError, match="profile must be a PackingProfile"):
        replace(_base_input(), profile={"mode": "static"})


def test_packing_fingerprint_metadata_runtime_flags_are_deep_immutable() -> None:
    flags = {"nested": {"bucket_size": 32}, "items": [1, 2]}
    metadata = PackingFingerprintMetadata(
        template_id="stage1_json_pretty",
        template_version=1,
        prompt_profile="prompt",
        tokenizer_id="tokenizer",
        object_ordering="sorted",
        objective_variant="sorted_sft",
        state_weighting_policy="none",
        normalization_policy="token_mean",
        loss_mask_version="mask_v1",
        preprocessing_version="preprocess_v1",
        packing_mode="static",
        packing_length=2048,
        runtime_flags=flags,
    )

    flags["nested"]["bucket_size"] = 64
    assert metadata.runtime_flags["nested"]["bucket_size"] == 32
    with pytest.raises(TypeError):
        metadata.runtime_flags["nested"]["bucket_size"] = 128


def test_packing_fingerprint_metadata_rejects_non_mapping_runtime_flags() -> None:
    for bad_runtime_flags in ([("a", 1)], (("a", 1),)):
        with pytest.raises(TypeError, match="runtime_flags must be a mapping"):
            PackingFingerprintMetadata(
                template_id="stage1_json_pretty",
                template_version=1,
                prompt_profile="prompt",
                tokenizer_id="tokenizer",
                object_ordering="sorted",
                objective_variant="sorted_sft",
                state_weighting_policy="none",
                normalization_policy="token_mean",
                loss_mask_version="mask_v1",
                preprocessing_version="preprocess_v1",
                packing_mode="static",
                packing_length=2048,
                runtime_flags=bad_runtime_flags,
            )


def test_static_sft_fingerprint_wraps_legacy_fields_with_detection_contract() -> None:
    fingerprint = build_static_sft_packing_fingerprint(
        payload=_base_input(),
        runtime_fields={
            "dataset_split": "train",
            "dataset_seed": 17,
            "nested": {"b": 2, "a": [1, 2]},
        },
    )

    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["nested"] == {"a": [1, 2], "b": 2}
    contract = fingerprint["detection_packing_contract"]
    assert contract["schema_version"] == "detection_packing_fingerprint_v1"
    assert isinstance(contract["sha256"], str)
    assert contract["metadata"]["template_id"] == "stage1_json_pretty"
    assert contract["metadata"]["packing_mode"] == "static"


def test_stage1_static_sft_request_resolves_behavioral_contract_fields() -> None:
    fingerprint = build_stage1_static_sft_packing_fingerprint(
        StaticSftPackingFingerprintRequest(
            detection_sequence_format="compact_full",
            prompt_profile="compact-prompt",
            tokenizer_id="tokenizer-v1",
            object_ordering="random",
            profile=PackingProfile(mode="static", packing_length=2048),
            runtime_fields={"dataset_split": "train"},
        )
    )

    contract = fingerprint["detection_packing_contract"]
    assert contract["metadata"]["template_id"] == "compact_full"
    assert contract["metadata"]["objective_variant"] == "random_order_sft"
    assert contract["metadata"]["state_weighting_policy"] == "none"
    assert contract["metadata"]["normalization_policy"] == "token_mean"


def test_static_sft_eligibility_rejects_recursive_request_before_runtime_work() -> None:
    with pytest.raises(ValueError, match="trie target metadata preservation"):
        require_static_sft_packing_eligibility(
            detection_sequence_format="compact_full",
            object_ordering="random",
            objective_variant="random_permutation_et_rmp_ce",
            profile=PackingProfile(mode="static", packing_length=2048),
        )
