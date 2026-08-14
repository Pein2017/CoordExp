from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.research import human13_live_eval as live_eval
from scripts.research.build_human13_on_policy_frontier import CheckpointIdentity
from scripts.research.human13_live_eval import build_analyzer_output


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        binding=SimpleNamespace(
            unit_id="unit",
            purpose="overfit",
            artifact_root="/artifacts",
            panel=SimpleNamespace(panel_sha256="panel"),
            source=SimpleNamespace(
                checkpoint_path="/source",
                base_model_path="/base",
                adapter_sha256="a" * 64,
                special_embedding_sha256="b" * 64,
            ),
            surface=SimpleNamespace(
                prompt_policy_fingerprint="prompt",
                tokenizer_sha256="tokenizer",
                tokenizer_class="Tokenizer",
                wrapper="wrapper",
                parser="compact_object_box_closed_only",
            ),
        )
    )


def _image(image_id: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        image_id=image_id,
        panel_row_sha256=f"row-{image_id}",
        image_sha256=f"image-{image_id}",
        trajectories=(
            SimpleNamespace(
                trajectory_id=f"human13:{image_id}:source",
                request=SimpleNamespace(backend_version="4.57.1", max_new_tokens=3084),
            ),
        ),
    )


def _panel_manifest() -> SimpleNamespace:
    base = _manifest()
    return SimpleNamespace(
        binding=base.binding,
        images=tuple(_image(image_id) for image_id in range(1, 14)),
        full_panel=True,
    )


def _panel_outputs(
    manifest: SimpleNamespace,
    checkpoint: CheckpointIdentity,
) -> tuple[dict[str, object], ...]:
    return tuple(
        build_analyzer_output(
            manifest=manifest,
            manifest_sha256="c" * 64,
            image=image,
            arm_id="O-First-Safe",
            milestone=3,
            checkpoint_path=checkpoint.path,
            checkpoint_payload_sha256=checkpoint.payload_sha256,
            run_id="run",
            run_root="/run",
            resolved_arm_plan_sha256="e" * 64,
            resolved_config_sha256="f" * 64,
            trajectory_id=f"eval:O-First-Safe:3:{image.image_id}",
            generated_token_ids=(100 + image.image_id, 200 + image.image_id, 999),
            terminal_token_index=2,
            predictions=(
                {
                    "generated_order": 0,
                    "description": "person",
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "token_start": 0,
                    "token_end": 2,
                },
            ),
            parser="compact_object_box_closed_only",
            parser_status="accepted",
            stop_reason="im_end",
            malformed_row_count=0,
            runtime={"decode_seconds": 1.0},
        )
        for image in manifest.images
    )


def test_trajectory_analyzer_predictions_preserve_exact_token_spans() -> None:
    trajectory = SimpleNamespace(
        rows=(
            SimpleNamespace(
                row_index=0,
                category="person",
                bbox=(1.0, 2.0, 3.0, 4.0),
                token_start=7,
                token_end=13,
            ),
        )
    )

    predictions = live_eval.trajectory_analyzer_predictions(trajectory)

    assert predictions == (
        {
            "generated_order": 0,
            "description": "person",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "token_start": 7,
            "token_end": 13,
        },
    )


def test_build_analyzer_output_binds_clean_greedy_and_checkpoint_identity() -> None:
    result = build_analyzer_output(
        manifest=_manifest(),
        manifest_sha256="c" * 64,
        image=_image(),
        arm_id="A1",
        milestone=4,
        checkpoint_path="/run/checkpoints/step-4",
        checkpoint_payload_sha256="d" * 64,
        run_id="run-A1",
        run_root="/run",
        resolved_arm_plan_sha256="e" * 64,
        resolved_config_sha256="f" * 64,
        trajectory_id="eval:A1:4:7",
        generated_token_ids=(1, 2, 3),
        terminal_token_index=2,
        predictions=(
            {"generated_order": 0, "description": "person", "bbox": [1, 2, 3, 4]},
        ),
        parser="compact_object_box_closed_only",
        parser_status="accepted",
        stop_reason="im_end",
        malformed_row_count=2,
        runtime={"decode_seconds": 1.5},
    )

    assert result["decode_mode"] == "original_prompt_clean_greedy"
    assert result["repetition_penalty"] == 1.0
    assert result["generated_token_ids"] == [1, 2, 3]
    assert result["terminal_token_index"] == 2
    predictions = cast(list[dict[str, object]], result["predictions"])
    provenance = cast(dict[str, object], result["provenance"])
    assert predictions[0]["description"] == "person"
    assert result["parser"] == "compact_object_box_closed_only"
    assert result["parser_status"] == "accepted"
    assert provenance["manifest_sha256"] == "c" * 64
    assert provenance["checkpoint_path"] == "/run/checkpoints/step-4"
    assert provenance["physical_batch_size"] == 1
    assert provenance["do_sample"] is False


def test_build_analyzer_output_binds_explicit_rp110_policy() -> None:
    result = build_analyzer_output(
        manifest=_manifest(),
        manifest_sha256="c" * 64,
        image=_image(),
        arm_id="C",
        milestone=1,
        checkpoint_path="/run/checkpoints/step-1",
        checkpoint_payload_sha256="d" * 64,
        run_id="run-C",
        run_root="/run",
        resolved_arm_plan_sha256="e" * 64,
        resolved_config_sha256="f" * 64,
        trajectory_id="eval:C:1:7:rp1.10",
        generated_token_ids=(1, 2, 3),
        terminal_token_index=2,
        predictions=(),
        parser="compact_object_box_closed_only",
        parser_status="accepted",
        stop_reason="im_end",
        malformed_row_count=0,
        runtime={"decode_seconds": 1.0},
        repetition_penalty=1.10,
    )

    provenance = cast(dict[str, object], result["provenance"])
    assert result["repetition_penalty"] == 1.10
    assert provenance["repetition_penalty"] == 1.10


@pytest.mark.parametrize("repetition_penalty", (True, 1.05, float("nan")))
def test_evaluate_hf_checkpoint_rejects_unsealed_rp_before_live_boundary(
    repetition_penalty: object,
) -> None:
    with pytest.raises(ValueError, match="repetition_penalty"):
        live_eval.evaluate_hf_checkpoint(
            manifest=_panel_manifest(),
            manifest_sha256="c" * 64,
            checkpoint_path="/does/not/exist",
            arm_id="C",
            milestone=1,
            run_id="run-C",
            run_root="/run",
            resolved_arm_plan_sha256="e" * 64,
            resolved_config_sha256="f" * 64,
            source_config_path="/does/not/exist",
            repetition_penalty=cast(float, repetition_penalty),
        )


def test_hf_runtime_identity_is_bound_and_mixed_receipts_fail_closed() -> None:
    identity = {
        "backend": "hf",
        "backend_mode": "generate",
        "backend_version": "test",
        "batch_size": 1,
        "observed_model_dtype_names": ["torch.float32"],
        "observed_attn_implementation": "sdpa",
        "generation_config_fingerprint": "sealed",
        "model_identity": {"sha256": "model"},
        "tokenizer_identity": {"sha256": "tokenizer"},
        "processor_identity": {"sha256": "processor"},
    }
    outputs = tuple(
        {"image_id": image_id, "hf_runtime_identity": identity}
        for image_id in tuple(image.image_id for image in _panel_manifest().images)
    )

    assert live_eval.hf_runtime_identity_from_outputs(outputs) == identity
    with pytest.raises(ValueError, match="missing.*runtime identity"):
        live_eval.hf_runtime_identity_from_outputs(
            tuple({"image_id": item["image_id"]} for item in outputs)
        )
    mixed = list(outputs)
    mixed[-1] = {
        **mixed[-1],
        "hf_runtime_identity": {**identity, "observed_attn_implementation": "eager"},
    }
    with pytest.raises(ValueError, match="mixed.*runtime identity"):
        live_eval.hf_runtime_identity_from_outputs(tuple(mixed))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "vllm"),
        ("batch_size", 4),
        ("observed_model_dtype_names", ["torch.bfloat16"]),
        ("observed_attn_implementation", "flash_attention_2"),
    ),
)
def test_hf_output_runtime_identity_rejects_homogeneous_observed_drift(
    field: str, value: object
) -> None:
    identity = {
        "backend": "hf",
        "backend_mode": "generate",
        "backend_version": "test",
        "batch_size": 1,
        "observed_model_dtype_names": ["torch.float32"],
        "observed_attn_implementation": "sdpa",
        "generation_config_fingerprint": "sealed",
        "model_identity": {"sha256": "model"},
        "tokenizer_identity": {"sha256": "tokenizer"},
        "processor_identity": {"sha256": "processor"},
    }
    drifted = {**identity, field: value}
    outputs = tuple(
        {"image_id": image_id, "hf_runtime_identity": drifted}
        for image_id in tuple(image.image_id for image in _panel_manifest().images)
    )

    with pytest.raises(ValueError, match="observed runtime identity"):
        live_eval.hf_runtime_identity_from_outputs(outputs)


def test_build_analyzer_output_rejects_non_digest_or_wrong_image() -> None:
    with pytest.raises(ValueError, match="checkpoint_payload_sha256"):
        build_analyzer_output(
            manifest=_manifest(),
            manifest_sha256="c" * 64,
            image=_image(),
            arm_id="A1",
            milestone=1,
            checkpoint_path="/checkpoint",
            checkpoint_payload_sha256="not-a-digest",
            run_id="run",
            run_root="/run",
            resolved_arm_plan_sha256="e" * 64,
            resolved_config_sha256="f" * 64,
            trajectory_id="trajectory",
            generated_token_ids=(1,),
            terminal_token_index=0,
            predictions=(),
            parser="compact_object_box_closed_only",
            parser_status="accepted",
            stop_reason="im_end",
            malformed_row_count=0,
            runtime={},
        )


def test_current_decodes_from_outputs_preserves_same_decode_identity_and_order() -> (
    None
):
    manifest = _panel_manifest()
    checkpoint = CheckpointIdentity("/checkpoint", "d" * 64)
    outputs = _panel_outputs(manifest, checkpoint)

    decodes = live_eval.current_decodes_from_outputs(
        manifest=manifest,
        manifest_sha256="c" * 64,
        outputs=outputs,
        checkpoint=checkpoint,
    )

    assert tuple(decode.image_id for decode in decodes) == tuple(range(1, 14))
    assert decodes[6].trajectory_id == "eval:O-First-Safe:3:7"
    assert decodes[6].generated_token_ids == (107, 207, 999)
    assert decodes[6].terminal_token_index == 2
    assert decodes[6].malformed_row_count == 0
    assert decodes[6].predictions[0].token_start == 0
    assert decodes[6].predictions[0].token_end == 2
    assert decodes[6].parser == "compact_object_box_closed_only"
    assert decodes[6].parser_status == "accepted"
    assert decodes[6].checkpoint == checkpoint


def test_current_decodes_accepts_real_canonical_parser_status() -> None:
    manifest = _panel_manifest()
    checkpoint = CheckpointIdentity("/checkpoint", "d" * 64)
    outputs = list(_panel_outputs(manifest, checkpoint))
    for output in outputs:
        output["parser_status"] = "accepted"

    decodes = live_eval.current_decodes_from_outputs(
        manifest=manifest,
        manifest_sha256="c" * 64,
        outputs=outputs,
        checkpoint=checkpoint,
    )

    assert {decode.parser_status for decode in decodes} == {"accepted"}


def test_current_decodes_admits_the_sealed_rp110_source_surface() -> None:
    manifest = _panel_manifest()
    checkpoint = CheckpointIdentity("/checkpoint", "d" * 64)
    outputs = tuple(
        {**output, "repetition_penalty": 1.10}
        for output in _panel_outputs(manifest, checkpoint)
    )

    decodes = live_eval.current_decodes_from_outputs(
        manifest=manifest,
        manifest_sha256="c" * 64,
        outputs=outputs,
        checkpoint=checkpoint,
        repetition_penalty=1.10,
    )

    assert tuple(decode.image_id for decode in decodes) == tuple(range(1, 14))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda outputs: tuple(reversed(outputs)),
            "manifest image order",
        ),
        (
            lambda outputs: (
                {
                    **outputs[0],
                    "provenance": {
                        **cast(Mapping[str, object], outputs[0]["provenance"]),
                        "manifest_sha256": "0" * 64,
                    },
                },
                *outputs[1:],
            ),
            "manifest_sha256",
        ),
        (
            lambda outputs: (
                {
                    **outputs[0],
                    "provenance": {
                        **cast(Mapping[str, object], outputs[0]["provenance"]),
                        "checkpoint_payload_sha256": "0" * 64,
                    },
                },
                *outputs[1:],
            ),
            "checkpoint identity",
        ),
        (
            lambda outputs: (
                {
                    **outputs[0],
                    "provenance": {
                        **cast(Mapping[str, object], outputs[0]["provenance"]),
                        "image_sha256": "wrong",
                    },
                },
                *outputs[1:],
            ),
            "image_sha256",
        ),
        (
            lambda outputs: ({**outputs[0], "decode_mode": "sampled"}, *outputs[1:]),
            "clean-greedy surface",
        ),
        (
            lambda outputs: (
                {**outputs[0], "parser_status": "partial"},
                *outputs[1:],
            ),
            "parser status",
        ),
        (
            lambda outputs: (
                {**outputs[0], "terminal_token_index": 0},
                *outputs[1:],
            ),
            "terminal_token_index",
        ),
        (
            lambda outputs: (
                {**outputs[0], "malformed_row_count": -1},
                *outputs[1:],
            ),
            "malformed_row_count",
        ),
    ),
)
def test_current_decodes_from_outputs_rejects_order_or_sha_drift(
    mutation: Callable[[tuple[dict[str, object], ...]], tuple[dict[str, object], ...]],
    message: str,
) -> None:
    manifest = _panel_manifest()
    checkpoint = CheckpointIdentity("/checkpoint", "d" * 64)
    outputs = _panel_outputs(manifest, checkpoint)

    with pytest.raises(ValueError, match=message):
        live_eval.current_decodes_from_outputs(
            manifest=manifest,
            manifest_sha256="c" * 64,
            outputs=mutation(outputs),
            checkpoint=checkpoint,
        )
