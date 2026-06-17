from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "gt_vs_pred_scored.jsonl"
    artifact.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    return artifact


def _write_raw_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "gt_vs_pred.jsonl"
    artifact.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    return artifact


def _write_named_artifact(tmp_path: Path, name: str) -> Path:
    artifact = tmp_path / name
    artifact.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    return artifact


def _fake_infer_owner() -> SimpleNamespace:
    gen_cfg = SimpleNamespace(
        temperature=0.0,
        top_p=None,
        max_new_tokens=128,
        repetition_penalty=None,
        seed=7,
        stop_pressure_mode="off",
        stop_pressure_min_new_tokens=0,
        stop_pressure_trigger_rule="none",
        stop_pressure_logit_bias=0.0,
        stop_pressure_active=False,
    )
    cfg = SimpleNamespace(
        checkpoint_mode="full_model",
        model_checkpoint="base-model",
        adapter_checkpoint="adapter",
        gt_jsonl="gt.jsonl",
        pred_coord_mode="norm1000",
        device="cpu",
        limit=1,
        distributed_enabled=False,
        backend={},
    )
    return SimpleNamespace(
        cfg=cfg,
        gen_cfg=gen_cfg,
        resolved_mode="coordjson",
        mode_reason="test",
        prompt_variant="default",
        bbox_format="xyxy",
        detection_sequence_format="coordjson",
        object_field_order=("desc", "bbox_2d"),
        object_ordering="sorted",
        prompt_template_hash="prompt-hash",
        requested_mode="manual",
        attn_implementation_requested=None,
        attn_implementation_selected=None,
    )


def test_top_level_canonical_string_provenance_is_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy_fingerprint": "score:1",
                "detection_template": {"id": "compact"},
                "artifacts": {"gt_vs_pred_scored_jsonl": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance"]["prompt_policy_fingerprint"] == "prompt:1"


def test_generated_style_inference_provenance_is_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy_fingerprint": "score:1",
                    "detection_template": {"id": "compact"},
                },
                "artifacts": {"gt_vs_pred_scored_jsonl": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert (
        loaded["provenance"]["inference_provenance"]["decode_policy_fingerprint"]
        == "decode:1"
    )


def test_unbound_run_level_provenance_is_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    other = tmp_path / "other_scored.jsonl"
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy_fingerprint": "score:1",
                "artifacts": {"gt_vs_pred_scored_jsonl": str(other)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not bound to artifact"):
        load_comparable_artifact(artifact)


def test_generated_summary_without_first_class_fingerprints_is_not_comparable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.infer.artifacts as artifacts

    monkeypatch.setattr(
        artifacts,
        "_qwen_chat_generation_meta",
        lambda owner: {
            "eos_token": "<|im_end|>",
            "eos_token_id": 1,
            "pad_token": "<|endoftext|>",
            "pad_token_id": 2,
            "stop_tokens": ["<|im_end|>"],
            "processor_do_resize": False,
        },
    )
    counters = SimpleNamespace(to_summary=lambda: {"total": 1})

    first = artifacts.build_infer_summary_payload(
        owner=_fake_infer_owner(),
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )
    second = artifacts.build_infer_summary_payload(
        owner=_fake_infer_owner(),
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )

    assert first["inference_provenance"] == second["inference_provenance"]
    assert first["inference_provenance"]["comparable"] is False
    assert set(first["inference_provenance"]["missing_provenance_fields"]) == {
        "prompt_policy_fingerprint",
        "decode_policy_fingerprint",
        "model_identity_fingerprint",
    }
    assert "prompt_policy_fingerprint" not in first["inference_provenance"]


def test_generated_summary_with_transitional_fingerprint_is_not_comparable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.infer.artifacts as artifacts

    owner = _fake_infer_owner()
    owner.prompt_policy_fingerprint = "transitional_prompt:1"
    owner.decode_policy_fingerprint = "decode:1"
    owner.model_identity_fingerprint = "model:1"
    monkeypatch.setattr(
        artifacts,
        "_qwen_chat_generation_meta",
        lambda owner: {
            "eos_token": "<|im_end|>",
            "eos_token_id": 1,
            "pad_token": "<|endoftext|>",
            "pad_token_id": 2,
            "stop_tokens": ["<|im_end|>"],
            "processor_do_resize": False,
        },
    )
    counters = SimpleNamespace(to_summary=lambda: {"total": 1})

    payload = artifacts.build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )

    assert payload["inference_provenance"]["comparable"] is False
    assert payload["inference_provenance"]["invalid_provenance_fields"] == (
        "prompt_policy_fingerprint",
    )
    assert "prompt_policy_fingerprint" not in payload["inference_provenance"]


def test_generated_summary_with_alias_fingerprints_is_not_comparable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.infer.artifacts as artifacts

    owner = _fake_infer_owner()
    owner.prompt_fingerprint = "prompt:legacy"
    owner.generation_policy_fingerprint = "decode:legacy"
    owner.model_fingerprint = "model:legacy"
    monkeypatch.setattr(
        artifacts,
        "_qwen_chat_generation_meta",
        lambda owner: {
            "eos_token": "<|im_end|>",
            "eos_token_id": 1,
            "pad_token": "<|endoftext|>",
            "pad_token_id": 2,
            "stop_tokens": ["<|im_end|>"],
            "processor_do_resize": False,
        },
    )
    counters = SimpleNamespace(to_summary=lambda: {"total": 1})

    payload = artifacts.build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )

    provenance = payload["inference_provenance"]
    assert provenance["comparable"] is False
    assert set(provenance["missing_provenance_fields"]) == {
        "prompt_policy_fingerprint",
        "decode_policy_fingerprint",
        "model_identity_fingerprint",
    }
    assert provenance["invalid_provenance_fields"] == ()
    assert "prompt_policy_fingerprint" not in provenance
    assert "decode_policy_fingerprint" not in provenance
    assert "model_identity_fingerprint" not in provenance


def test_generated_summary_with_first_class_fingerprints_loads_raw_comparable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.infer.artifacts as artifacts
    from src.infer.artifacts import load_comparable_artifact

    owner = _fake_infer_owner()
    owner.prompt_policy_fingerprint = "prompt:1"
    owner.decode_policy_fingerprint = "decode:1"
    owner.model_identity_fingerprint = "model:1"
    monkeypatch.setattr(
        artifacts,
        "_qwen_chat_generation_meta",
        lambda owner: {
            "eos_token": "<|im_end|>",
            "eos_token_id": 1,
            "pad_token": "<|endoftext|>",
            "pad_token_id": 2,
            "stop_tokens": ["<|im_end|>"],
            "processor_do_resize": False,
        },
    )
    counters = SimpleNamespace(to_summary=lambda: {"total": 1})
    artifact = _write_raw_artifact(tmp_path)
    payload = artifacts.build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )
    payload["artifacts"] = {"gt_vs_pred_jsonl": str(artifact)}
    (tmp_path / "summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance"]["inference_provenance"]["score_policy"] == "none"


def test_generated_summary_uses_cfg_level_first_class_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.infer.artifacts as artifacts
    from src.infer.artifacts import load_comparable_artifact

    owner = _fake_infer_owner()
    owner.cfg.prompt_policy_fingerprint = "prompt:cfg"
    owner.cfg.decode_policy_fingerprint = "decode:cfg"
    owner.cfg.model_identity_fingerprint = "model:cfg"
    monkeypatch.setattr(
        artifacts,
        "_qwen_chat_generation_meta",
        lambda owner: {
            "eos_token": "<|im_end|>",
            "eos_token_id": 1,
            "pad_token": "<|endoftext|>",
            "pad_token_id": 2,
            "stop_tokens": ["<|im_end|>"],
            "processor_do_resize": False,
        },
    )
    counters = SimpleNamespace(to_summary=lambda: {"total": 1})
    artifact = _write_raw_artifact(tmp_path)

    payload = artifacts.build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="test",
        batch_size=1,
    )
    payload["artifacts"] = {"gt_vs_pred_jsonl": str(artifact)}
    (tmp_path / "summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)
    provenance = loaded["provenance"]["inference_provenance"]

    assert provenance["comparable"] is True
    assert provenance["prompt_policy_fingerprint"] == "prompt:cfg"
    assert provenance["decode_policy_fingerprint"] == "decode:cfg"
    assert provenance["model_identity_fingerprint"] == "model:cfg"


def test_carrier_level_comparable_false_vetoes_nested_provenance(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"comparable":false,'
            '"inference_provenance":{'
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy_fingerprint":"score:1"'
            "}"
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_artifact_sidecar_can_override_run_level_raw_summary(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy":"none"'
            "}\n"
        ),
        encoding="utf-8",
    )
    artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy_fingerprint":"score:1",'
            '"detection_template":{"id":"compact"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance_path"].endswith(
        "gt_vs_pred_scored.jsonl.provenance.json"
    )
    assert loaded["provenance"]["score_policy_fingerprint"] == "score:1"


def test_transitional_fingerprints_are_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"transitional_prompt_policy_fingerprint:sha256:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy_fingerprint":"score:1"'
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_score_bearing_artifact_requires_score_fingerprint(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "detection_template": {"id": "compact"},
                "artifacts": {"gt_vs_pred_scored_jsonl": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score_policy_fingerprint"):
        load_comparable_artifact(artifact)


def test_raw_artifact_requires_score_policy_none(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_raw_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "detection_template": {"id": "compact"},
                "artifacts": {"gt_vs_pred_jsonl": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score_policy"):
        load_comparable_artifact(artifact)


def test_raw_artifact_with_score_policy_none_is_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_raw_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy": "none",
                "detection_template": {"id": "compact"},
                "artifacts": {"gt_vs_pred_jsonl": str(artifact)},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance"]["score_policy"] == "none"


def test_require_score_rejects_canonical_raw_artifact_name(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_raw_artifact(tmp_path)
    artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy_fingerprint": "score:1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw artifact family"):
        load_comparable_artifact(artifact, require_score=True)


def test_score_sidecar_must_be_bound_to_scored_artifact(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _write_artifact(first_dir)
    second = _write_artifact(second_dir)
    sidecar = first.with_suffix(first.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy_fingerprint": "score:1",
                "artifact_path": str(first),
            }
        ),
        encoding="utf-8",
    )
    second.with_suffix(second.suffix + ".provenance.json").write_text(
        sidecar.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not bound to artifact"):
        load_comparable_artifact(second, require_score=True)


def test_scored_substring_in_unknown_filename_can_be_explicit_raw(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_named_artifact(tmp_path, "gt_vs_pred_scored_backup.jsonl")
    artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy":"none",'
            '"detection_template":{"id":"compact"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance"]["score_policy"] == "none"


def test_custom_artifact_without_explicit_score_role_is_not_comparable(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_named_artifact(tmp_path, "custom_predictions.jsonl")
    artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1"'
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_custom_artifact_with_explicit_scored_role_is_comparable(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_named_artifact(tmp_path, "custom_predictions.jsonl")
    artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":"prompt:1",'
            '"decode_policy_fingerprint":"decode:1",'
            '"model_identity_fingerprint":"model:1",'
            '"score_policy_fingerprint":"score:1",'
            '"detection_template":{"id":"compact"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    loaded = load_comparable_artifact(artifact)

    assert loaded["provenance"]["score_policy_fingerprint"] == "score:1"


def test_moved_jsonl_without_provenance_is_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_existing_json_without_comparable_identity_is_not_comparable(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text('{"random":"json"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_alias_only_provenance_is_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"prompt":{"fingerprint":"prompt:alias"},'
            '"decode":{"fingerprint":"decode:alias"},'
            '"model":{"checkpoint":"model:alias"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_non_string_canonical_provenance_is_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"prompt_policy_fingerprint":true,'
            '"decode_policy_fingerprint":["decode"],'
            '"model_identity_fingerprint":{"value":"model"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)


def test_substring_false_positive_provenance_is_not_comparable(
    tmp_path: Path,
) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = _write_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        (
            "{"
            '"deprompted":{"id":"not-prompt"},'
            '"decode_error":{"id":"not-decode"},'
            '"model_parse_failure":{"id":"not-model"}'
            "}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)
