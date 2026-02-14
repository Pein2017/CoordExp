from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer, _IM_END


def _mk_uninit_trainer(cfg):
    t = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
    t.rollout_matching_cfg = cfg
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


def test_decode_batch_size_defaults_to_one_when_unset():
    t = _mk_uninit_trainer({})
    assert t._decode_batch_size() == 1


def test_decode_batch_size_rejects_non_positive_values():
    t0 = _mk_uninit_trainer({"decode_batch_size": 0})
    with pytest.raises(ValueError, match=r"decode_batch_size must be > 0"):
        t0._validate_rollout_matching_cfg()

    t1 = _mk_uninit_trainer({"decode_batch_size": -3})
    with pytest.raises(ValueError, match=r"decode_batch_size must be > 0"):
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


def test_apply_rollout_decoding_to_generation_config_greedy_disables_sampling():
    gen_cfg = SimpleNamespace()
    RolloutMatchingSFTTrainer._apply_rollout_decoding_to_generation_config(
        gen_cfg=gen_cfg,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
    )
    assert gen_cfg.do_sample is False
    assert gen_cfg.temperature == 1.0
    assert gen_cfg.top_p == 1.0
    assert gen_cfg.top_k == 0
    assert gen_cfg.repetition_penalty == 1.05


def test_apply_rollout_decoding_to_generation_config_sampling_respects_top_p_and_top_k():
    gen_cfg0 = SimpleNamespace()
    RolloutMatchingSFTTrainer._apply_rollout_decoding_to_generation_config(
        gen_cfg=gen_cfg0,
        temperature=0.01,
        top_p=0.9,
        top_k=-1,
        repetition_penalty=1.1,
    )
    assert gen_cfg0.do_sample is True
    assert gen_cfg0.temperature == 0.01
    assert gen_cfg0.top_p == 0.9
    assert gen_cfg0.top_k == 0

    gen_cfg1 = SimpleNamespace()
    RolloutMatchingSFTTrainer._apply_rollout_decoding_to_generation_config(
        gen_cfg=gen_cfg1,
        temperature=0.01,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
    )
    assert gen_cfg1.do_sample is True
    assert gen_cfg1.temperature == 0.01
    assert gen_cfg1.top_p == 0.95
    assert gen_cfg1.top_k == 50


def test_rollout_vllm_request_config_kwargs_propagates_decoding_knobs():
    kwargs = RolloutMatchingSFTTrainer._rollout_vllm_request_config_kwargs(
        max_tokens=123,
        temperature=0.01,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
    )
    assert kwargs["n"] == 1
    assert kwargs["max_tokens"] == 123
    assert kwargs["temperature"] == 0.01
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50
    assert kwargs["repetition_penalty"] == 1.05
    assert kwargs["stop"] == [_IM_END]
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
    t._decoding_params = lambda: (0.0, 1.0, -1)

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
