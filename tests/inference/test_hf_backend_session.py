from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

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
class FakeNativeProcessor:
    def __init__(self, prompt_rows: list[list[int]]) -> None:
        self.prompt_rows = prompt_rows
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = SimpleNamespace(padding_side="right")

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        width = max(len(row) for row in self.prompt_rows)
        input_ids = []
        attention_mask = []
        for row in self.prompt_rows:
            padding = [0] * (width - len(row))
            if self.tokenizer.padding_side == "left":
                input_ids.append(padding + row)
                attention_mask.append([0] * len(padding) + [1] * len(row))
            else:
                input_ids.append(row + padding)
                attention_mask.append([1] * len(row) + [0] * len(padding))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.ones((len(self.prompt_rows) * 2, 4)),
            "image_grid_thw": torch.tensor(
                [[1, 1, 2] for _ in self.prompt_rows],
                dtype=torch.long,
            ),
        }


class FakeRawHFModel(FakeHFModel):
    def __init__(
        self,
        *,
        sequences: list[list[int]],
        score_steps: list[torch.Tensor],
        raw_logit_steps: list[torch.Tensor] | None,
    ) -> None:
        super().__init__(sequences=sequences, score_steps=score_steps)
        self.raw_logit_steps = raw_logit_steps

    def parameters(self) -> Any:
        return iter(())

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        output = super().generate(**kwargs)
        if kwargs.get("output_logits") and self.raw_logit_steps is not None:
            output.logits = tuple(self.raw_logit_steps)
        return output


def _semantic_request(
    image_path: Any,
    *,
    request_id: str = "row-1",
    raw: bool = False,
    expected_prompt_token_ids: tuple[int, ...] = (11, 12),
    logical_transform_id: str = "identity",
    repetition_penalty: float = 1.0,
) -> Any:
    from src.inference.backend import DecodeRequest, GenerationPolicy

    image_bytes = image_path.read_bytes()
    return DecodeRequest(
        request_id=request_id,
        chat_text=f"chat-{request_id}",
        input_prompt_token_ids=(11,),
        expected_executed_prompt_token_ids=expected_prompt_token_ids,
        image_path=str(image_path),
        declared_image_width=2,
        declared_image_height=2,
        decoded_image_width=2,
        decoded_image_height=2,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        generation_policy=GenerationPolicy(
            max_new_tokens=2,
            repetition_penalty=repetition_penalty,
            include_raw_model_logprob=raw,
        ),
        expected_image_grid_thw=(1, 1, 2),
        logical_transform_id=logical_transform_id,
    )


def _backend_launch(*, batch_size: int = 2) -> Any:
    from src.inference.backend import BackendLaunch

    return BackendLaunch(
        backend="hf",
        model_path="/unused/test-model",
        model_dtype="fp32",
        batch_size=batch_size,
        generation_config_fingerprint="gen-fp",
        backend_options={
            "attn_implementation": "eager",
            "patch_embed_linearization": "enabled",
        },
    )


def _session_receipt(
    *,
    batch_size: int = 2,
    score_owned_channel: str = "policy_logprob",
) -> Any:
    from src.inference.backend import (
        POLICY_LIKELIHOOD_DEFINITION,
        RAW_LIKELIHOOD_DEFINITION,
        BackendSessionReceipt,
    )

    return BackendSessionReceipt(
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        backend_version="test-transformers",
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tokenizer"},
        processor_identity={"class": "FakeNativeProcessor"},
        generation_config_fingerprint="gen-fp",
        effective_settings={"batch_size": batch_size},
        likelihood_semantics={
            "policy": POLICY_LIKELIHOOD_DEFINITION,
            "raw": RAW_LIKELIHOOD_DEFINITION,
            "score_owned_channel": score_owned_channel,
        },
    )


def test_semantic_decode_request_contains_no_backend_native_payload(tmp_path: Any) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    request = _semantic_request(image_path)

    assert request.chat_text == "chat-row-1"
    assert request.input_prompt_token_ids == (11,)
    assert request.expected_executed_prompt_token_ids == (11, 12)
    assert request.prompt_token_ids == [11, 12]
    assert not hasattr(request, "model_inputs")


def test_backend_launch_rejects_native_tensor_objects() -> None:
    from src.inference.backend import BackendLaunch

    with pytest.raises(RuntimeContractError) as exc_info:
        BackendLaunch(
            backend="hf",
            model_path="/unused/test-model",
            model_dtype="fp32",
            batch_size=1,
            generation_config_fingerprint="gen-fp",
            backend_options={"pixel_values": torch.ones(1)},
        )

    assert exc_info.value.code == "backend_contract.native_object"


def test_injected_session_opener_validates_receipt_and_closes() -> None:
    from src.inference.backend import open_backend_session

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False
            self.receipt = _session_receipt()

        def decode(self, requests: Any) -> tuple[Any, ...]:
            return ()

        def close(self) -> None:
            self.closed = True

    session = FakeSession()
    with open_backend_session(_backend_launch(), opener=lambda _: session) as opened:
        assert opened is session
        assert session.closed is False
    assert session.closed is True


def test_session_receipt_rejects_non_policy_score_ownership() -> None:
    receipt = _session_receipt(score_owned_channel="raw_model_logprob")

    with pytest.raises(RuntimeContractError) as exc_info:
        receipt.validate_for_launch(_backend_launch())

    assert exc_info.value.code == "backend_contract.session_receipt"
    assert exc_info.value.context["field"] == (
        "likelihood_semantics.score_owned_channel"
    )


def test_hf_session_native_batch_records_policy_and_raw_fp32_channels(
    tmp_path: Any,
) -> None:
    from src.inference.hf_backend import HFBackendSession

    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    tokenizer = FakeTokenizer()
    policy_steps = [
        _logits_for_tokens([21, 22]),
        _logits_for_tokens([tokenizer.eos_token_id, tokenizer.eos_token_id]),
    ]
    raw_steps = [step.clone() for step in policy_steps]
    raw_steps[0][0, 21] = 0.0
    raw_steps[0][0, 22] = 20.0
    model = FakeRawHFModel(
        sequences=[
            [11, 12, 21, tokenizer.eos_token_id],
            [11, 12, 22, tokenizer.eos_token_id],
        ],
        score_steps=policy_steps,
        raw_logit_steps=raw_steps,
    )
    processor = FakeNativeProcessor([[11, 12], [11, 12]])
    session = HFBackendSession(
        launch=_backend_launch(batch_size=2),
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        receipt=_session_receipt(batch_size=2),
    )

    results = session.decode(
        [
            _semantic_request(image_paths[0], request_id="row-1", raw=True),
            _semantic_request(image_paths[1], request_id="row-2", raw=True),
        ]
    )

    assert len(processor.calls) == 1
    assert processor.calls[0]["do_resize"] is False
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["output_scores"] is True
    assert model.generate_kwargs["output_logits"] is True
    assert results[0].observed_image_grid_thw == (1, 1, 2)
    assert results[0].token_trace[0].policy_logprob is not None
    assert results[0].token_trace[0].raw_model_logprob is not None
    assert results[0].token_trace[0].logprob == results[0].token_trace[0].policy_logprob
    assert (
        results[0].token_trace[0].policy_logprob
        != results[0].token_trace[0].raw_model_logprob
    )
    assert results[0].generated_token_ids == (21, tokenizer.eos_token_id)
    assert results[0].parser_text == "A"
    performance = session.receipt.effective_settings["performance"]
    assert performance["request_count"] == 2
    assert performance["generated_token_count"] == 4
    assert performance["decode_elapsed_seconds"] > 0
    assert performance["requests_per_second"] > 0
    assert performance["generated_tokens_per_second"] > 0


def test_hf_session_forces_left_padding_for_heterogeneous_prompts(
    tmp_path: Any,
) -> None:
    from src.inference.hf_backend import HFBackendSession

    image_paths = [tmp_path / "short.png", tmp_path / "long.png"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    tokenizer = FakeTokenizer()
    processor = FakeNativeProcessor([[11], [11, 12]])
    model = FakeRawHFModel(
        sequences=[
            [tokenizer.pad_token_id, 11, 21, tokenizer.eos_token_id],
            [11, 12, 22, tokenizer.eos_token_id],
        ],
        score_steps=[
            _logits_for_tokens([21, 22]),
            _logits_for_tokens([tokenizer.eos_token_id, tokenizer.eos_token_id]),
        ],
        raw_logit_steps=None,
    )
    session = HFBackendSession(
        launch=_backend_launch(batch_size=2),
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        receipt=_session_receipt(batch_size=2),
    )

    session.decode(
        [
            _semantic_request(
                image_paths[0],
                request_id="short",
                expected_prompt_token_ids=(11,),
            ),
            _semantic_request(image_paths[1], request_id="long"),
        ]
    )

    assert tokenizer.padding_side == "left"
    assert processor.tokenizer.padding_side == "left"
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["input_ids"].tolist() == [[0, 11], [11, 12]]
    assert model.generate_kwargs["attention_mask"].tolist() == [[0, 1], [1, 1]]


def test_hf_repetition_penalty_groups_native_batches_by_prompt_width(
    tmp_path: Any,
) -> None:
    from src.inference.hf_backend import _native_batch_groups

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    requests = (
        _semantic_request(
            image_path,
            request_id="short-1",
            expected_prompt_token_ids=(11,),
            repetition_penalty=1.1,
        ),
        _semantic_request(
            image_path,
            request_id="long",
            expected_prompt_token_ids=(11, 12),
            repetition_penalty=1.1,
        ),
        _semantic_request(
            image_path,
            request_id="short-2",
            expected_prompt_token_ids=(13,),
            repetition_penalty=1.1,
        ),
    )

    groups = _native_batch_groups(requests, batch_size=3)

    assert [[item.request_id for item in group] for group in groups] == [
        ["short-1", "short-2"],
        ["long"],
    ]


def test_hf_raw_trace_fails_when_generate_returns_no_raw_logits(tmp_path: Any) -> None:
    from src.inference.hf_backend import HFBackendSession

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    tokenizer = FakeTokenizer()
    model = FakeRawHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
        raw_logit_steps=None,
    )
    session = HFBackendSession(
        launch=_backend_launch(batch_size=1),
        model=model,
        processor=FakeNativeProcessor([[11, 12]]),
        tokenizer=tokenizer,
        receipt=_session_receipt(batch_size=1),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        session.decode([_semantic_request(image_path, raw=True)])

    assert exc_info.value.code == "hf_backend.missing_logits"


def test_hf_session_rehashes_image_immediately_before_projection(tmp_path: Any) -> None:
    from src.inference.hf_backend import HFBackendSession

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    request = _semantic_request(image_path)
    Image.new("RGB", (2, 2), color="black").save(image_path)
    tokenizer = FakeTokenizer()
    model = FakeRawHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
        raw_logit_steps=None,
    )
    session = HFBackendSession(
        launch=_backend_launch(batch_size=1),
        model=model,
        processor=FakeNativeProcessor([[11, 12]]),
        tokenizer=tokenizer,
        receipt=_session_receipt(batch_size=1),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        session.decode([request])

    assert exc_info.value.code == "hf_backend.image_sha256_mismatch"


def test_hf_session_applies_semantic_logical_image_transform(tmp_path: Any) -> None:
    from src.inference.hf_backend import HFBackendSession
    from src.qwen.images import rgb_image_sha256

    image_path = tmp_path / "corners.png"
    image = Image.new("RGB", (2, 2), color="black")
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (255, 255, 0))
    image.save(image_path)
    tokenizer = FakeTokenizer()

    class CapturingProcessor(FakeNativeProcessor):
        def __init__(self) -> None:
            super().__init__([[11, 12]])
            self.executed_image: Image.Image | None = None

        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.executed_image = kwargs["images"][0].copy()
            return super().__call__(**kwargs)

    processor = CapturingProcessor()
    model = FakeRawHFModel(
        sequences=[[11, 12, 21, tokenizer.eos_token_id]],
        score_steps=[
            _logits_for_tokens([21]),
            _logits_for_tokens([tokenizer.eos_token_id]),
        ],
        raw_logit_steps=None,
    )
    session = HFBackendSession(
        launch=_backend_launch(batch_size=1),
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        receipt=_session_receipt(batch_size=1),
    )

    result = session.decode(
        [_semantic_request(image_path, logical_transform_id="hvflip")]
    )[0]

    executed = processor.executed_image
    assert executed is not None
    assert executed.getpixel((0, 0)) == (255, 255, 0)
    assert executed.getpixel((1, 1)) == (255, 0, 0)
    assert result.executed_media_sha256 == rgb_image_sha256(executed)


def test_raw_generation_logprob_matches_teacher_forced_fp32_reference() -> None:
    from src.inference.hf_backend import (
        compare_raw_generation_to_teacher_forced,
        teacher_forced_chosen_token_logprobs,
    )

    vocab_size = 32
    full_logits = torch.full((1, 4, vocab_size), -3.0, dtype=torch.float16)
    full_logits[0, 1, 21] = 2.0
    full_logits[0, 2, 22] = 1.0

    class TeacherForcedModel:
        def __call__(self, **_: Any) -> SimpleNamespace:
            return SimpleNamespace(logits=full_logits)

    reference = teacher_forced_chosen_token_logprobs(
        model=TeacherForcedModel(),
        native_prompt_inputs={
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        generated_token_ids=[21, 22],
    )
    generation_raw = torch.stack(
        [
            torch.log_softmax(full_logits[0, 1].float(), dim=-1)[21],
            torch.log_softmax(full_logits[0, 2].float(), dim=-1)[22],
        ]
    )

    comparison = compare_raw_generation_to_teacher_forced(
        generation_raw,
        reference,
    )

    assert comparison.compared_steps == 2
    assert comparison.max_absolute_difference == pytest.approx(0.0)
