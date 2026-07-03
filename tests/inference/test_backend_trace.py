from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 151645
    im_end_id = 151645

    def __init__(self) -> None:
        self.id_to_token = {
            0: "<|pad|>",
            11: "p11",
            12: "p12",
            13: "p13",
            21: "A",
            22: "B",
            23: "C",
            24: "D",
            151645: "<|im_end|>",
        }
        self.decode_calls: list[dict[str, Any]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|im_end|>":
            return self.im_end_id
        raise KeyError(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False, **_: Any) -> str:
        self.decode_calls.append(
            {"token_ids": list(token_ids), "skip_special_tokens": skip_special_tokens}
        )
        pieces = [self.id_to_token[token_id] for token_id in token_ids]
        if skip_special_tokens:
            pieces = [piece for piece in pieces if not piece.startswith("<|")]
        return "".join(pieces)


class FakeHFModel:
    def __init__(self, sequences: list[list[int]], score_steps: list[torch.Tensor] | None) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.score_steps = None if score_steps is None else tuple(score_steps)
        self.generate_kwargs: dict[str, Any] | None = None
        self.transition_scores_seen_normalized: bool | None = None

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_kwargs = kwargs
        payload = {"sequences": self.sequences}
        if self.score_steps is not None:
            payload["scores"] = self.score_steps
        return SimpleNamespace(**payload)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        normalize_logits: bool,
    ) -> torch.Tensor:
        self.transition_scores_seen_normalized = normalize_logits
        generated = sequences[:, -len(scores) :]
        rows = []
        for step_index, step_logits in enumerate(scores):
            logprobs = torch.log_softmax(step_logits, dim=-1)
            rows.append(logprobs.gather(1, generated[:, step_index : step_index + 1]))
        return torch.cat(rows, dim=1)


class FakeHFModelWithoutTransitionScores(FakeHFModel):
    compute_transition_scores = None


class FakeHFModelTransitionRaises(FakeHFModel):
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        normalize_logits: bool,
    ) -> torch.Tensor:
        raise ValueError("transition boom")


def _logits_for_tokens(batch_tokens: list[int], *, vocab_size: int = 151646) -> torch.Tensor:
    logits = torch.full((len(batch_tokens), vocab_size), -20.0, dtype=torch.float32)
    for row_index, token_id in enumerate(batch_tokens):
        logits[row_index, token_id] = 20.0
    return logits


def test_decode_records_are_backend_neutral_and_complete() -> None:
    from src.inference.backend import DecodeRequest, DecodeResult, HFGenerateBackend, TokenTrace

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )
    request = DecodeRequest(
        request_id="row-1",
        prompt_token_ids=[11, 12],
        model_inputs={"input_ids": torch.tensor([11, 12])},
        max_new_tokens=2,
    )

    result = HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [request],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )[0]

    assert isinstance(result, DecodeResult)
    assert not hasattr(result, "hf_outputs")
    assert result.request_id == "row-1"
    assert result.backend == "hf"
    assert result.backend_mode == "generate"
    assert result.response_family == "hf"
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_token_ids == [21, tokenizer.eos_token_id]
    assert result.raw_generated_text == "A<|im_end|>"
    assert result.parser_text == "A"
    assert result.stop_reason == "im_end"
    assert result.model_identity == {"family": "base-only"}
    assert result.tokenizer_identity == {"sha256": "tok-sha"}
    assert result.generation_config_fingerprint == "gen-fp"
    assert all(isinstance(trace, TokenTrace) for trace in result.token_trace)
    result.validate_for_scored()


def test_hf_generate_requests_scored_deterministic_qwen_stop() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )

    HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={"input_ids": torch.tensor([11, 12])},
                max_new_tokens=2,
            )
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["return_dict_in_generate"] is True
    assert model.generate_kwargs["output_scores"] is True
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert model.generate_kwargs["pad_token_id"] == tokenizer.pad_token_id
    assert model.generate_kwargs["max_new_tokens"] == 2
    assert model.generate_kwargs["repetition_penalty"] == pytest.approx(1.10)
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs
    assert model.transition_scores_seen_normalized is True


def test_hf_backend_prefers_model_device_over_cpu_request_tensors() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    class FakeCudaModel(FakeHFModel):
        def parameters(self) -> Any:
            return iter([SimpleNamespace(device=torch.device("cuda:0"))])

    backend = HFGenerateBackend(
        model=FakeCudaModel(sequences=[[11, 12]], score_steps=[]),
        tokenizer=FakeTokenizer(),
    )
    device = backend._target_device(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={"pixel_values": torch.ones(1, 2)},
                max_new_tokens=2,
            )
        ]
    )

    assert str(device) == "cuda:0"


def test_hf_generate_rejects_mixed_repetition_penalty_in_batch() -> None:
    from src.common.errors import RuntimeContractError
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[
            [11, 12, 21, tokenizer.eos_token_id],
            [11, 12, 22, tokenizer.eos_token_id],
        ],
        score_steps=[
            _logits_for_tokens([21, 22]),
            _logits_for_tokens([tokenizer.eos_token_id, tokenizer.eos_token_id]),
        ],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=2,
                    repetition_penalty=1.10,
                ),
                DecodeRequest(
                    request_id="row-2",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=2,
                    repetition_penalty=1.0,
                ),
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.repetition_penalty_mismatch"


def test_missing_hf_scores_fail_with_contract_error() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=None,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=2,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.missing_scores"


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("prompt_token_ids", []),
        ("generated_token_ids", []),
        ("token_trace", []),
        ("stop_reason", ""),
        ("backend", ""),
        ("backend_mode", ""),
        ("response_family", ""),
        ("strip_policy", ""),
        ("model_identity", {}),
        ("tokenizer_identity", {}),
        ("generation_config_fingerprint", ""),
    ],
)
def test_missing_required_trace_fields_fail_before_scored_artifacts(
    field_name: str,
    bad_value: Any,
) -> None:
    from src.inference.backend import DecodeResult, TokenTrace

    payload = {
        "request_id": "row-1",
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
        "prompt_token_ids": [11, 12],
        "generated_token_ids": [21],
        "raw_generated_text": "A",
        "parser_text": "A",
        "strip_policy": "terminal_im_end",
        "stop_reason": "length",
        "model_identity": {"family": "base-only"},
        "tokenizer_identity": {"sha256": "tok-sha"},
        "generation_config_fingerprint": "gen-fp",
        "token_trace": [
            TokenTrace(
                step_index=0,
                token_id=21,
                token_text="A",
                logprob=-0.1,
                is_stop=False,
                is_pad=False,
                backend="hf",
                backend_mode="generate",
                response_family="hf",
            )
        ],
    }
    payload[field_name] = bad_value
    result = DecodeResult(**payload)

    with pytest.raises(RuntimeContractError) as exc_info:
        result.validate_for_scored()

    assert exc_info.value.code == "backend_trace.missing_field"
    assert exc_info.value.context["field"] == field_name


def test_invalid_strip_policy_fails_before_scored_artifacts() -> None:
    from src.inference.backend import DecodeResult, TokenTrace

    result = DecodeResult(
        request_id="row-1",
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=[11, 12],
        generated_token_ids=[21],
        raw_generated_text="A",
        parser_text="A",
        strip_policy="skip_special_tokens",
        stop_reason="length",
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
        token_trace=[
            TokenTrace(
                step_index=0,
                token_id=21,
                token_text="A",
                logprob=-0.1,
                is_stop=False,
                is_pad=False,
                backend="hf",
                backend_mode="generate",
                response_family="hf",
            )
        ],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        result.validate_for_scored()

    assert exc_info.value.code == "backend_trace.invalid_strip_policy"
    assert exc_info.value.context["strip_policy"] == "skip_special_tokens"


def test_special_token_raw_trace_preserves_im_end_without_skip_special_tokens() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )

    result = HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={"input_ids": torch.tensor([11, 12])},
                max_new_tokens=2,
            )
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )[0]

    assert result.token_trace[-1].token_text == "<|im_end|>"
    assert result.raw_generated_text.endswith("<|im_end|>")
    assert result.parser_text == "A"
    assert result.strip_policy == "terminal_im_end"
    assert {call["skip_special_tokens"] for call in tokenizer.decode_calls} == {False}


def test_batched_variable_prompt_width_alignment_and_post_stop_padding() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    prompt_width = 3
    model = FakeHFModel(
        sequences=[
            [tokenizer.pad_token_id, 11, 12, 21, tokenizer.eos_token_id, tokenizer.pad_token_id],
            [11, 12, 13, 22, 23, 24],
        ],
        score_steps=[
            _logits_for_tokens([21, 22]),
            _logits_for_tokens([tokenizer.eos_token_id, 23]),
            _logits_for_tokens([tokenizer.pad_token_id, 24]),
        ],
    )

    results = HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="short",
                prompt_token_ids=[11, 12],
                model_inputs={"input_ids": torch.tensor([11, 12])},
                max_new_tokens=3,
            ),
            DecodeRequest(
                request_id="long",
                prompt_token_ids=[11, 12, 13],
                model_inputs={"input_ids": torch.tensor([11, 12, 13])},
                max_new_tokens=3,
            ),
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )

    short, long = results
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["input_ids"].shape == (2, prompt_width)
    assert model.generate_kwargs["input_ids"].tolist() == [
        [tokenizer.pad_token_id, 11, 12],
        [11, 12, 13],
    ]
    assert model.generate_kwargs["attention_mask"].tolist() == [
        [0, 1, 1],
        [1, 1, 1],
    ]
    assert short.prompt_token_ids == [11, 12]
    assert short.generated_token_ids == [21, tokenizer.eos_token_id]
    assert [trace.token_id for trace in short.token_trace] == [21, tokenizer.eos_token_id, tokenizer.pad_token_id]
    assert [trace.step_index for trace in short.token_trace] == [0, 1, 2]
    assert [trace.is_stop for trace in short.token_trace] == [False, True, False]
    assert [trace.is_pad for trace in short.token_trace] == [False, False, True]
    assert short.token_trace[2].logprob is None
    assert short.raw_generated_text == "A<|im_end|>"
    assert short.parser_text == "A"

    assert long.prompt_token_ids == [11, 12, 13]
    assert long.generated_token_ids == [22, 23, 24]
    assert [trace.token_id for trace in long.token_trace] == [22, 23, 24]
    assert all(trace.logprob is not None for trace in long.token_trace)
    assert long.stop_reason == "length"


def test_prestop_generated_pad_token_fails_fast() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, tokenizer.pad_token_id, 21]],
        score_steps=[
            _logits_for_tokens([tokenizer.pad_token_id]),
            _logits_for_tokens([21]),
        ],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=2,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.unexpected_pad_token"
    assert exc_info.value.context["row_index"] == 0
    assert exc_info.value.context["step_index"] == 0
    assert exc_info.value.context["token_id"] == tokenizer.pad_token_id


def test_hf_generate_places_padded_text_inputs_on_request_tensor_device() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([11, 12], device=torch.device("cpu"))
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )

    HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={"input_ids": input_ids},
                max_new_tokens=2,
            )
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["input_ids"].device == input_ids.device
    assert model.generate_kwargs["attention_mask"].device == input_ids.device


def test_hf_generate_rejects_mixed_request_tensor_devices() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    meta_tensor = torch.empty((1,), device="meta")
    model = FakeHFModel(
        sequences=[[0, 11, 12, 21], [11, 12, 13, 22]],
        score_steps=[_logits_for_tokens([21, 22])],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="cpu",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=1,
                ),
                DecodeRequest(
                    request_id="meta",
                    prompt_token_ids=[11, 12, 13],
                    model_inputs={"input_ids": torch.tensor([11, 12, 13]), "pixel_values": meta_tensor},
                    max_new_tokens=1,
                ),
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.device_mismatch"


def test_hf_generate_rejects_mixed_max_new_tokens_in_batch() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[0, 11, 12, 21], [11, 12, 13, 22]],
        score_steps=[_logits_for_tokens([21, 22])],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="short",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=1,
                ),
                DecodeRequest(
                    request_id="long",
                    prompt_token_ids=[11, 12, 13],
                    model_inputs={"input_ids": torch.tensor([11, 12, 13])},
                    max_new_tokens=2,
                ),
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.max_new_tokens_mismatch"


def test_overlong_sequence_score_mismatch_fails_before_trace_alignment() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, 22, 99]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([22]),
        ],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=2,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.shape_mismatch"


def test_fallback_logprob_gather_uses_prompt_width_generated_suffix() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModelWithoutTransitionScores(
        sequences=[[11, 12, 21, 22]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([22]),
        ],
    )

    result = HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={"input_ids": torch.tensor([11, 12])},
                max_new_tokens=2,
            )
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )[0]

    assert result.generated_token_ids == [21, 22]
    assert [trace.token_id for trace in result.token_trace] == [21, 22]
    assert all(trace.logprob is not None and trace.logprob > -0.001 for trace in result.token_trace)


def test_hf_generate_forwards_non_text_model_inputs() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    pixel_values = torch.ones((2, 4))
    image_grid_thw = torch.tensor([[1, 1, 2]])
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )

    HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={
                    "input_ids": torch.tensor([11, 12]),
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                },
                max_new_tokens=2,
            )
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )

    assert model.generate_kwargs is not None
    assert torch.equal(model.generate_kwargs["pixel_values"], pixel_values)
    assert torch.equal(model.generate_kwargs["image_grid_thw"], image_grid_thw)


def test_hf_generate_collates_qwen_image_inputs_by_key() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    first_pixels = torch.ones((2, 4))
    second_pixels = torch.ones((3, 4))
    first_grid = torch.tensor([[1, 1, 2]])
    second_grid = torch.tensor([[1, 1, 3]])
    model = FakeHFModel(
        sequences=[
            [11, 12, 21, tokenizer.eos_token_id],
            [11, 12, 22, tokenizer.eos_token_id],
        ],
        score_steps=[
            _logits_for_tokens([21, 22]),
            _logits_for_tokens([tokenizer.eos_token_id, tokenizer.eos_token_id]),
        ],
    )

    HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
        [
            DecodeRequest(
                request_id="row-1",
                prompt_token_ids=[11, 12],
                model_inputs={
                    "input_ids": torch.tensor([11, 12]),
                    "pixel_values": first_pixels,
                    "image_grid_thw": first_grid,
                },
                max_new_tokens=2,
            ),
            DecodeRequest(
                request_id="row-2",
                prompt_token_ids=[11, 12],
                model_inputs={
                    "input_ids": torch.tensor([11, 12]),
                    "pixel_values": second_pixels,
                    "image_grid_thw": second_grid,
                },
                max_new_tokens=2,
            ),
        ],
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tok-sha"},
        generation_config_fingerprint="gen-fp",
    )

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["pixel_values"].shape == (5, 4)
    assert model.generate_kwargs["image_grid_thw"].shape == (2, 3)
    assert model.generate_kwargs["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


def test_hf_generate_rejects_bad_qwen_image_grid_shape() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={
                        "input_ids": torch.tensor([11, 12]),
                        "image_grid_thw": torch.tensor([1, 2]),
                    },
                    max_new_tokens=2,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.model_input_shape"
    assert exc_info.value.context["field"] == "image_grid_thw"


def test_hf_generate_rejects_wrong_score_batch_dimension() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21]],
        score_steps=[_logits_for_tokens([21, 22])],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=1,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.score_shape_mismatch"
    assert exc_info.value.context["step_index"] == 0


def test_hf_generate_rejects_rank_one_score_tensor() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModel(
        sequences=[[11, 12, 21]],
        score_steps=[torch.zeros((151646,), dtype=torch.float32)],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=1,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.score_shape_mismatch"
    assert exc_info.value.context["score_shape"] == (151646,)


def test_hf_transition_score_exception_is_wrapped_as_contract_error() -> None:
    from src.inference.backend import DecodeRequest, HFGenerateBackend

    tokenizer = FakeTokenizer()
    model = FakeHFModelTransitionRaises(
        sequences=[[11, 12, 21]],
        score_steps=[_logits_for_tokens([21])],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(model=model, tokenizer=tokenizer).generate_batch(
            [
                DecodeRequest(
                    request_id="row-1",
                    prompt_token_ids=[11, 12],
                    model_inputs={"input_ids": torch.tensor([11, 12])},
                    max_new_tokens=1,
                )
            ],
            model_identity={"family": "base-only"},
            tokenizer_identity={"sha256": "tok-sha"},
            generation_config_fingerprint="gen-fp",
        )

    assert exc_info.value.code == "backend_trace.transition_scores_failed"
    assert isinstance(exc_info.value.cause, ValueError)


def test_create_backend_rejects_vllm_execution_clearly() -> None:
    from src.inference.backend import create_backend

    with pytest.raises(RuntimeContractError) as exc_info:
        create_backend("vllm", model=object(), tokenizer=object())

    assert exc_info.value.code == "backend_trace.backend_not_implemented"
    assert exc_info.value.context["backend"] == "vllm"
