from __future__ import annotations

from types import SimpleNamespace

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
                parser="parser",
            ),
        )
    )


def _image() -> SimpleNamespace:
    return SimpleNamespace(
        image_id=7,
        panel_row_sha256="row",
        image_sha256="image",
        trajectories=(
            SimpleNamespace(
                trajectory_id="human13:7:source",
                request=SimpleNamespace(backend_version="4.57.1", max_new_tokens=3084),
            ),
        ),
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
        predictions=(
            {"generated_order": 0, "description": "person", "bbox": [1, 2, 3, 4]},
        ),
        stop_reason="im_end",
        malformed_row_count=2,
        runtime={"decode_seconds": 1.5},
    )

    assert result["decode_mode"] == "original_prompt_clean_greedy"
    assert result["repetition_penalty"] == 1.0
    assert result["generated_token_ids"] == [1, 2, 3]
    assert result["predictions"][0]["description"] == "person"
    assert result["provenance"]["manifest_sha256"] == "c" * 64
    assert result["provenance"]["checkpoint_path"] == "/run/checkpoints/step-4"
    assert result["provenance"]["physical_batch_size"] == 1
    assert result["provenance"]["do_sample"] is False


def test_build_analyzer_output_rejects_non_digest_or_wrong_image() -> None:
    kwargs = dict(
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
        predictions=(),
        stop_reason="im_end",
        malformed_row_count=0,
        runtime={},
    )

    try:
        build_analyzer_output(**kwargs)
    except ValueError as exc:
        assert "checkpoint_payload_sha256" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("invalid digest was accepted")
