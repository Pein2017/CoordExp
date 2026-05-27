from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.infer.backend import (
    apply_hf_generation_config_from_decode_request,
    vllm_request_config_kwargs_from_decode_request,
)
from src.config.rollout_matching_schema import RolloutMatchingConfig
from src.infer.runtime import build_decode_request_from_rollout_matching_config
from src.trainers.stage2_rollout_runtime import Stage2RolloutRuntime


def _mk_uninit_trainer(cfg, *, include_decode_defaults: bool = True):
    t = Stage2RolloutRuntime.__new__(Stage2RolloutRuntime)
    if include_decode_defaults:
        merged = {
            "rollout_backend": "hf",
            "eval_rollout_backend": "hf",
            "rollout_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        }
        merged.update(dict(cfg))
        t.rollout_matching_cfg = merged
    else:
        t.rollout_matching_cfg = dict(cfg)
    return t


def test_validate_rollout_matching_cfg_rejects_legacy_keys():
    t = _mk_uninit_trainer({"temperature": 0.1})
    with pytest.raises(ValueError, match=r"Legacy rollout-matching keys have been removed"):
        t._validate_rollout_matching_cfg()


@pytest.mark.parametrize(
    "legacy_key",
    [
        "rollout_generate_batch_size",
        "rollout_infer_batch_size",
        "post_rollout_pack_scope",
    ],
)
def test_validate_rollout_matching_cfg_rejects_removed_rollout_keys(legacy_key: str):
    t = _mk_uninit_trainer({legacy_key: 1})
    with pytest.raises(ValueError, match=r"Legacy rollout-matching keys have been removed"):
        t._validate_rollout_matching_cfg()


def test_decode_batch_size_requires_explicit_context_keys():
    t = _mk_uninit_trainer({}, include_decode_defaults=False)
    with pytest.raises(
        ValueError,
        match=r"rollout_decode_batch_size must be provided explicitly",
    ):
        t._validate_rollout_matching_cfg()


def test_decode_batch_size_rejects_non_positive_values():
    t0 = _mk_uninit_trainer(
        {
            "rollout_decode_batch_size": 0,
            "eval_decode_batch_size": 1,
        }
    )
    with pytest.raises(
        ValueError,
        match=r"rollout_decode_batch_size must be > 0",
    ):
        t0._validate_rollout_matching_cfg()

    t1 = _mk_uninit_trainer(
        {
            "rollout_decode_batch_size": 1,
            "eval_decode_batch_size": -3,
        }
    )
    with pytest.raises(
        ValueError,
        match=r"eval_decode_batch_size must be > 0",
    ):
        t1._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_accepts_decoding_mapping():
    t = _mk_uninit_trainer({"decoding": {"temperature": 0.01, "top_p": 0.9, "top_k": -1}})
    t._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_rejects_invalid_ranges():
    t0 = _mk_uninit_trainer({"decoding": {"temperature": -1.0}})
    with pytest.raises(ValueError, match=r"decoding\.temperature must be >= 0"):
        t0._validate_rollout_matching_cfg()

    t1 = _mk_uninit_trainer({"decoding": {"top_p": 0.0}})
    with pytest.raises(ValueError, match=r"decoding\.top_p must be in \(0, 1\]"):
        t1._validate_rollout_matching_cfg()

    t2 = _mk_uninit_trainer({"decoding": {"top_k": 0}})
    with pytest.raises(ValueError, match=r"decoding\.top_k must be -1"):
        t2._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_accepts_eval_detection_block():
    t = _mk_uninit_trainer(
        {
            "eval_detection": {
                "enabled": True,
                "metrics": "coco",
                "score_mode": "constant",
                "constant_score": 1.0,
                "pred_score_source": "eval_rollout_constant",
                "pred_score_version": 2,
            }
        }
    )
    t._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_accepts_eval_detection_confidence_postop():
    t = _mk_uninit_trainer(
        {
            "eval_detection": {
                "score_mode": "confidence_postop",
                "pred_score_source": "eval_rollout_constant",
            }
        }
    )
    t._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_accepts_lvis_eval_detection_block():
    t = _mk_uninit_trainer(
        {
            "eval_detection": {
                "enabled": True,
                "metrics": "lvis",
                "lvis_max_dets": 300,
                "score_mode": "constant",
                "constant_score": 1.0,
                "pred_score_source": "eval_rollout_constant",
                "pred_score_version": 2,
            }
        }
    )
    t._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_rejects_eval_detection_bad_score_mode():
    t = _mk_uninit_trainer({"eval_detection": {"score_mode": "unsupported"}})
    with pytest.raises(
        ValueError,
        match=r"eval_detection\.score_mode must be one of \{'constant', 'confidence_postop'\}",
    ):
        t._validate_rollout_matching_cfg()


def test_validate_rollout_matching_cfg_rejects_unknown_eval_prompt_variant():
    t = _mk_uninit_trainer({"eval_prompt_variant": "not_a_variant"})
    with pytest.raises(ValueError, match=r"Unknown prompt variant"):
        t._validate_rollout_matching_cfg()


def test_rollout_matching_schema_accepts_training_prompt_variant():
    cfg = RolloutMatchingConfig(
        rollout_decode_batch_size=1,
        eval_decode_batch_size=1,
        prompt_variant="coco_80",
    )

    assert cfg.prompt_variant == "coco_80"


def test_rollout_matching_schema_rejects_unknown_training_prompt_variant():
    with pytest.raises(ValueError, match=r"Unknown prompt variant"):
        RolloutMatchingConfig(
            rollout_decode_batch_size=1,
            eval_decode_batch_size=1,
            prompt_variant="not_a_variant",
        )


def test_validate_rollout_matching_cfg_rejects_unknown_training_prompt_variant():
    t = _mk_uninit_trainer({"prompt_variant": "not_a_variant"})
    with pytest.raises(ValueError, match=r"Unknown prompt variant"):
        t._validate_rollout_matching_cfg()


def test_apply_hf_generation_config_from_decode_request_greedy_disables_sampling():
    request = build_decode_request_from_rollout_matching_config(
        {
            "max_new_tokens": 64,
            "decoding": {"temperature": 0.0, "top_p": 0.9, "top_k": 50},
        }
    )
    gen_cfg = SimpleNamespace()
    request = replace(request, repetition_penalty=1.05)
    apply_hf_generation_config_from_decode_request(gen_cfg=gen_cfg, request=request)
    assert gen_cfg.max_new_tokens == 64
    assert gen_cfg.do_sample is False
    assert gen_cfg.temperature == 1.0
    assert gen_cfg.top_p == 1.0
    assert gen_cfg.top_k == 0
    assert gen_cfg.repetition_penalty == 1.05
    assert gen_cfg.use_cache is True


def test_apply_hf_generation_config_from_decode_request_sampling_respects_top_p_and_top_k():
    request0 = build_decode_request_from_rollout_matching_config(
        {
            "max_new_tokens": 64,
            "decode_mode": "sampling",
            "repetition_penalty": 1.1,
            "decoding": {
                "temperature": 0.01,
                "top_p": 0.9,
                "top_k": -1,
            },
        }
    )
    gen_cfg0 = SimpleNamespace()
    apply_hf_generation_config_from_decode_request(gen_cfg=gen_cfg0, request=request0)
    assert gen_cfg0.do_sample is True
    assert gen_cfg0.temperature == 0.01
    assert gen_cfg0.top_p == 0.9
    assert gen_cfg0.top_k == 0
    assert gen_cfg0.use_cache is True

    request1 = build_decode_request_from_rollout_matching_config(
        {
            "max_new_tokens": 64,
            "decode_mode": "sampling",
            "repetition_penalty": 1.1,
            "decoding": {
                "temperature": 0.01,
                "top_p": 0.95,
                "top_k": 50,
            },
        }
    )
    gen_cfg1 = SimpleNamespace()
    apply_hf_generation_config_from_decode_request(gen_cfg=gen_cfg1, request=request1)
    assert gen_cfg1.do_sample is True
    assert gen_cfg1.temperature == 0.01
    assert gen_cfg1.top_p == 0.95
    assert gen_cfg1.top_k == 50
    assert gen_cfg1.use_cache is True


def test_rollout_vllm_request_config_kwargs_propagates_decoding_knobs():
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "max_new_tokens": 123,
            "decode_mode": "sampling",
            "decoding": {"temperature": 0.01, "top_p": 0.9, "top_k": 50},
            "repetition_penalty": 1.05,
        }
    )
    kwargs = vllm_request_config_kwargs_from_decode_request(request)

    assert kwargs["n"] == 1
    assert kwargs["max_tokens"] == 123
    assert kwargs["temperature"] == 0.01
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50
    assert kwargs["repetition_penalty"] == 1.05
    assert kwargs["stop"] == ["<|im_end|>"]
    assert kwargs["return_details"] is True


def test_merge_rollout_matching_batch_metrics_preserves_existing_keys():
    t = _mk_uninit_trainer({})
    batch = {"_rollout_matching_batch_metrics": {"rollout/max_new_tokens": 64.0}}
    t._merge_rollout_matching_batch_metrics(
        batch,
        {
            "rollout/max_new_tokens": 128.0,
            "packing/post_rollout_segments": 3.0,
        },
    )
    bm = batch.get("_rollout_matching_batch_metrics")
    assert isinstance(bm, dict)
    assert bm["rollout/max_new_tokens"] == 128.0
    assert bm["packing/post_rollout_segments"] == 3.0


def test_build_rollout_metrics_emits_only_canonical_decode_count_keys():
    t = _mk_uninit_trainer({})
    t._cfg = lambda _k, default=None: default

    meta = [
        {
            "decode_mode": "greedy",
            "gt_objects": 1,
            "matched_for_supervision": 1,
            "valid_pred_objects": 1,
        },
        {
            "decode_mode": "beam",
            "gt_objects": 1,
            "matched_for_supervision": 1,
            "valid_pred_objects": 1,
        },
    ]

    payload = t._build_rollout_metrics_from_meta(meta)
    assert payload["rollout/decode_non_beam_count"] == pytest.approx(1.0)
    assert payload["rollout/decode_beam_count"] == pytest.approx(1.0)
    assert "rollout/decode_greedy" not in payload
    assert "rollout/decode_beam" not in payload
    assert "rollout/match_rate" not in payload
