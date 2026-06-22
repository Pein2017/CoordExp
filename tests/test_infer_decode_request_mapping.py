from __future__ import annotations

import unittest

from src.infer.backend import (
    build_swift_request_config_from_decode_request,
    vllm_request_config_kwargs_from_decode_request,
)
from src.infer.runtime import build_decode_request_from_infer_config
from src.infer.runtime import DetectionDecodeRequest
from src.infer.runtime import build_decode_request_from_rollout_matching_config
from src.infer.runtime import build_model_identity_fingerprint
from src.infer.runtime import legacy_generation_kwargs_from_decode_request


def test_infer_generation_maps_to_shared_decode_request() -> None:
    request = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "decode_mode": "sampling",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 128,
                "repetition_penalty": 1.05,
                "seed": 11,
                "trace_logprobs": True,
            },
        }
    )

    assert request.backend == "hf"
    assert request.backend_mode == "local"
    assert request.decode_mode == "sampling"
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.top_k is None
    assert request.max_new_tokens == 128
    assert request.repetition_penalty == 1.05
    assert request.seed == 11
    assert request.trace_logprobs is True


def test_infer_generation_infers_greedy_decode_from_zero_temperature() -> None:
    request = build_decode_request_from_infer_config(
        {
            "backend": {"type": "vllm", "mode": "server"},
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 64,
            },
        }
    )

    assert request.backend == "vllm"
    assert request.backend_mode == "server"
    assert request.decode_mode == "greedy"
    assert request.temperature == 0.0


def test_infer_generation_preserves_existing_offline_defaults() -> None:
    request = build_decode_request_from_infer_config(
        {
            "generation": {
                "temperature": 0.01,
                "max_new_tokens": 64,
            },
        }
    )

    assert request.backend == "hf"
    assert request.backend_mode == "local"
    assert request.top_p == 0.95
    assert request.repetition_penalty == 1.05
    assert request.trace_prompt_logprobs is False


def test_infer_generation_normalizes_sample_alias() -> None:
    request = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "decode_mode": "sample",
                "temperature": 0.2,
                "max_new_tokens": 8,
            },
        }
    )

    assert request.decode_mode == "sampling"


def test_infer_generation_rejects_greedy_with_sampling_temperature() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "decode_mode=greedy requires temperature <= 0",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "decode_mode": "greedy",
                    "temperature": 0.2,
                    "max_new_tokens": 8,
                },
            }
        )


def test_infer_generation_rejects_sampling_with_zero_temperature() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "decode_mode=sampling requires temperature > 0",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "decode_mode": "sampling",
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                },
            }
        )


def test_infer_generation_requires_explicit_beam_mode_for_multiple_beams() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "decode_mode=beam",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "temperature": 0.0,
                    "num_beams": 4,
                    "max_new_tokens": 8,
                },
            }
        )


def test_infer_generation_rejects_beam_until_bridge_supports_it() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "decode_mode=beam is not supported",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "decode_mode": "beam",
                    "num_beams": 4,
                    "max_new_tokens": 8,
                },
            }
        )


def test_infer_generation_requires_multiple_beams_for_beam_mode() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "num_beams > 1",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "decode_mode": "beam",
                    "num_beams": 1,
                    "max_new_tokens": 8,
                },
            }
        )


def test_infer_decode_request_marks_unfinalized_stop_policy_and_fingerprint() -> None:
    request = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "trace_prompt_logprobs": True,
            },
        }
    )

    assert request.trace_prompt_logprobs is True
    assert request.stop_strings == ("<|im_end|>",)
    assert request.decode_policy_fingerprint.startswith("decode:")


def test_infer_decode_request_rejects_invalid_backend_type() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "infer.backend.type",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "bad"},
                "generation": {"max_new_tokens": 8},
            }
        )


def test_infer_decode_request_rejects_unsupported_backend_mode() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "infer.backend.mode",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "vllm", "mode": "servre"},
                "generation": {"max_new_tokens": 8},
            }
        )


def test_infer_decode_request_rejects_unsupported_top_k() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "top_k is not supported",
    ):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {"max_new_tokens": 8, "top_k": 32},
            }
        )


def test_infer_decode_request_projects_to_legacy_generation_kwargs() -> None:
    request = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "temperature": 0.4,
                "top_p": 0.8,
                "max_new_tokens": 33,
                "repetition_penalty": 1.2,
                "seed": 19,
            },
        }
    )

    kwargs = legacy_generation_kwargs_from_decode_request(
        request,
        batch_size=3,
        stop_pressure_mode="raw_text_object_boundary",
        stop_pressure_trigger_rule="raw_text_object_boundary",
        stop_pressure_logit_bias=2.5,
    )

    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_new_tokens"] == 33
    assert kwargs["repetition_penalty"] == 1.2
    assert kwargs["batch_size"] == 3
    assert kwargs["seed"] == 19
    assert kwargs["stop_pressure_mode"] == "raw_text_object_boundary"
    assert kwargs["stop_pressure_trigger_rule"] == "raw_text_object_boundary"
    assert kwargs["stop_pressure_logit_bias"] == 2.5
    assert "compact_grammar_enabled" not in kwargs
    assert "compact_grammar_format" not in kwargs


def test_rollout_matching_maps_to_shared_decode_request_with_sampling_overrides() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "decode_mode": "greedy",
            "max_new_tokens": 12,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        },
        decode_override={
            "decode_mode": "sampling",
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 24,
        },
    )

    assert request.backend == "hf"
    assert request.backend_mode == "local"
    assert request.decode_mode == "sampling"
    assert request.temperature == 0.7
    assert request.top_p == 0.92
    assert request.top_k == 24
    assert request.repetition_penalty == 1.1
    assert request.max_new_tokens == 12
    assert request.decode_policy_fingerprint.startswith("decode:")


def test_rollout_matching_infers_sampling_from_temperature_when_decode_mode_absent() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "max_new_tokens": 12,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "decoding": {"temperature": 0.7, "top_p": 0.92, "top_k": 24},
        },
    )

    assert request.decode_mode == "sampling"
    assert request.temperature == 0.7
    assert request.top_p == 0.92
    assert request.top_k == 24


def test_rollout_matching_rejects_greedy_with_sampling_temperature_override() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "decode_mode=greedy requires temperature <= 0",
    ):
        build_decode_request_from_rollout_matching_config(
            {
                "decode_mode": "greedy",
                "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
            },
            decode_override={"decode_mode": "greedy", "temperature": 0.7},
        )


def test_decode_policy_fingerprint_is_stable_and_excludes_operational_fields() -> None:
    first = build_decode_request_from_infer_config(
        {
            "backend": {"type": "vllm", "mode": "server", "base_url": "http://a"},
            "generation": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_new_tokens": 16,
                "batch_size": 1,
            },
        }
    )
    second = build_decode_request_from_infer_config(
        {
            "backend": {"type": "vllm", "mode": "server", "base_url": "http://b"},
            "generation": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_new_tokens": 16,
                "batch_size": 99,
            },
        }
    )

    assert first.decode_policy_fingerprint == second.decode_policy_fingerprint


def test_decode_policy_fingerprint_changes_with_backend_and_stop_pressure_constraints() -> None:
    unconstrained = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": 8,
            },
        }
    )
    vllm = build_decode_request_from_infer_config(
        {
            "backend": {"type": "vllm", "mode": "server"},
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": 8,
            },
        }
    )
    constrained = build_decode_request_from_infer_config(
        {
            "backend": {"type": "hf"},
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "stop_pressure": {
                    "mode": "min_new_tokens_after_object_open",
                    "min_new_tokens": 2,
                    "trigger_rule": "raw_text_object_open",
                },
            },
        }
    )

    assert unconstrained.decode_policy_fingerprint != vllm.decode_policy_fingerprint
    assert (
        unconstrained.decode_policy_fingerprint
        != constrained.decode_policy_fingerprint
    )
    assert constrained.generation_constraints == (
        (
            "stop_pressure",
            {
                "logit_bias": 0.0,
                "min_new_tokens": 2,
                "mode": "min_new_tokens_after_object_open",
                "trigger_rule": "raw_text_object_open",
            },
        ),
    )


def test_model_identity_fingerprint_is_stable_and_backend_sensitive() -> None:
    first = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
        backend="hf",
        backend_mode="local",
    )
    second = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
        backend="hf",
        backend_mode="local",
    )
    vllm = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
        backend="vllm",
        backend_mode="server",
    )

    assert first == second
    assert first.startswith("model:")
    assert first != vllm


def test_model_identity_fingerprint_tracks_backend_model_not_server_url() -> None:
    first = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="config-model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="config-model",
        resolved_adapter_checkpoint=None,
        backend="vllm",
        backend_mode="server",
        backend_model="served-a",
        backend_sync_identity={"base_url": "http://127.0.0.1:8000"},
    )
    moved_server = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="config-model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="config-model",
        resolved_adapter_checkpoint=None,
        backend="vllm",
        backend_mode="server",
        backend_model="served-a",
        backend_sync_identity={"base_url": "http://127.0.0.1:9000"},
    )
    other_model = build_model_identity_fingerprint(
        checkpoint_mode="full_model",
        requested_model_checkpoint="config-model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="config-model",
        resolved_adapter_checkpoint=None,
        backend="vllm",
        backend_mode="server",
        backend_model="served-b",
        backend_sync_identity={"base_url": "http://127.0.0.1:8000"},
    )

    assert first != other_model
    assert first == moved_server


def test_rollout_matching_maps_vllm_server_backend_context() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "vllm": {"mode": "server"},
            "decoding": {"temperature": 0.0},
        }
    )

    assert request.backend == "vllm"
    assert request.backend_mode == "server"


def test_rollout_matching_maps_vllm_colocate_backend_context() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "vllm": {"mode": "colocate"},
            "decoding": {"temperature": 0.0},
        }
    )

    assert request.backend == "vllm"
    assert request.backend_mode == "colocate"


def test_rollout_matching_preserves_default_decode_fields() -> None:
    request = build_decode_request_from_rollout_matching_config({})

    assert request.backend == "hf"
    assert request.backend_mode == "local"
    assert request.decode_mode == "greedy"
    assert request.max_new_tokens == 512
    assert request.num_beams == 1
    assert request.temperature == 0.0
    assert request.top_p == 1.0
    assert request.top_k == -1
    assert request.repetition_penalty == 1.0
    assert request.stop_strings == ("<|im_end|>",)


def test_vllm_request_config_kwargs_project_rollout_decode_request() -> None:
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "max_new_tokens": 77,
            "decode_mode": "sampling",
            "decoding": {"temperature": 0.3, "top_p": 0.91, "top_k": 17},
            "repetition_penalty": 1.07,
        }
    )

    kwargs = vllm_request_config_kwargs_from_decode_request(request)

    assert kwargs == {
        "n": 1,
        "max_tokens": 77,
        "temperature": 0.3,
        "top_p": 0.91,
        "top_k": 17,
        "repetition_penalty": 1.07,
        "stop": ["<|im_end|>"],
        "return_details": True,
    }


def test_vllm_request_config_kwargs_uses_canonical_stop_strings() -> None:
    request = DetectionDecodeRequest(
        backend="vllm",
        backend_mode="server",
        decode_mode="greedy",
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        stop_strings=("STOP_A", "STOP_B"),
    )

    kwargs = vllm_request_config_kwargs_from_decode_request(request)

    assert kwargs["stop"] == ["STOP_A", "STOP_B"]
    assert kwargs["return_details"] is True


def test_build_swift_request_config_from_decode_request_applies_overlays(
    monkeypatch,
) -> None:
    class _FakeRequestConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(
        "src.infer.backend.import_swift_request_config",
        lambda: _FakeRequestConfig,
    )
    request = build_decode_request_from_rollout_matching_config(
        {
            "rollout_backend": "vllm",
            "max_new_tokens": 77,
            "decode_mode": "sampling",
            "decoding": {"temperature": 0.3, "top_p": 0.91, "top_k": 17},
            "repetition_penalty": 1.07,
        }
    )

    cfg = build_swift_request_config_from_decode_request(
        request,
        seed=123,
        trace_logprobs=True,
    )

    assert cfg.kwargs == {
        "n": 1,
        "max_tokens": 77,
        "temperature": 0.3,
        "top_p": 0.91,
        "top_k": 17,
        "repetition_penalty": 1.07,
        "stop": ["<|im_end|>"],
        "return_details": True,
        "seed": 123,
        "logprobs": True,
    }


def test_rollout_matching_rejects_offline_sample_alias() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "rollout_matching.decode_mode",
    ):
        build_decode_request_from_rollout_matching_config(
            {"decode_mode": "sample", "decoding": {"temperature": 0.7}}
        )


def test_rollout_matching_rejects_invalid_decoding_values() -> None:
    bad_configs = [
        {"decoding": {"temperature": -0.1}},
        {"decoding": {"top_p": 0.0}},
        {"decoding": {"top_k": 0}},
        {"repetition_penalty": 0.0},
    ]
    for cfg in bad_configs:
        with unittest.TestCase().assertRaises((TypeError, ValueError)):
            build_decode_request_from_rollout_matching_config(cfg)


def test_rollout_matching_rejects_unknown_decode_override() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "Unknown rollout decode override keys",
    ):
        build_decode_request_from_rollout_matching_config(
            {"decoding": {"temperature": 0.0}},
            decode_override={"bad": 1},
        )


def test_infer_generation_rejects_missing_generation_section() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "infer.generation section is required",
    ):
        build_decode_request_from_infer_config({"backend": {"type": "hf"}})
