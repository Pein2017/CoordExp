from __future__ import annotations

import gc
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from src.common.errors import RuntimeContractError
from src.inference.backend import (
    POLICY_LIKELIHOOD_DEFINITION,
    RAW_LIKELIHOOD_DEFINITION,
    BackendLaunch,
    BackendSessionReceipt,
    DecodeRequest,
    GenerationPolicy,
    token_ids_sha256,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 31
    vocab_size = 32

    def __init__(self) -> None:
        self.padding_side = "right"
        self.decode_calls: list[list[int]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if token != "<|im_end|>":
            raise KeyError(token)
        return self.eos_token_id

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        **_: Any,
    ) -> str:
        del skip_special_tokens
        self.decode_calls.append(list(token_ids))
        return "".join(f"<{token_id}>" for token_id in token_ids)


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "pixel_values": torch.ones((2, 4), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
        }


class FakeEvidenceModel:
    def __init__(
        self,
        *,
        observed_attention: str | None = "sdpa",
        parameter_specs: tuple[tuple[torch.dtype, int], ...] = (
            (torch.float32, 3),
        ),
    ) -> None:
        # The loaded execution head is authoritative when the base config
        # retains its pre-extension vocabulary size.
        self.config = SimpleNamespace(vocab_size=4)
        if observed_attention is not None:
            self.config._attn_implementation = observed_attention
        self.model = self
        self._parameters = tuple(
            torch.nn.Parameter(
                torch.zeros(size, dtype=dtype),
                requires_grad=False,
            )
            for dtype, size in parameter_specs
        )
        self.rope_calls: list[dict[str, Any]] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.to_calls: list[torch.device] = []
        self.eval_calls = 0

    def parameters(self) -> Any:
        return iter(self._parameters)

    def get_output_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(weight=torch.empty((32, 1)))

    def to(self, device: torch.device) -> FakeEvidenceModel:
        self.to_calls.append(torch.device(device))
        return self

    def eval(self) -> FakeEvidenceModel:
        self.eval_calls += 1
        return self

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        self.rope_calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "image_grid_thw": image_grid_thw.detach().clone(),
                "video_grid_thw": video_grid_thw,
            }
        )
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).view(1, 1, -1)
        return positions.expand(3, input_ids.shape[0], -1).clone(), None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 32),
            -1.0,
            dtype=torch.float16,
            device=input_ids.device,
        )
        logits[..., 0] = 3.0
        logits[..., 1] = 2.0
        logits[..., 2] = 2.0
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits)


def _launch() -> BackendLaunch:
    return BackendLaunch(
        backend="hf",
        model_path="/unused/test-model",
        model_dtype="fp32",
        batch_size=1,
        generation_config_fingerprint="gen-fp",
        backend_options={
            "attn_implementation": "eager",
            "patch_embed_linearization": "enabled",
        },
    )


def _receipt() -> BackendSessionReceipt:
    return BackendSessionReceipt(
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        backend_version="test-transformers",
        model_identity={"family": "base-only"},
        tokenizer_identity={"sha256": "tokenizer"},
        processor_identity={"class": "FakeProcessor"},
        generation_config_fingerprint="gen-fp",
        effective_settings={"batch_size": 1},
        likelihood_semantics={
            "policy": POLICY_LIKELIHOOD_DEFINITION,
            "raw": RAW_LIKELIHOOD_DEFINITION,
            "score_owned_channel": "policy_logprob",
        },
    )


def _request(tmp_path: Any, *, request_id: str = "exact-history") -> DecodeRequest:
    image_path = tmp_path / f"{request_id}.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    return DecodeRequest(
        request_id=request_id,
        chat_text=f"chat-{request_id}",
        input_prompt_token_ids=(11,),
        expected_executed_prompt_token_ids=(11, 12),
        image_path=str(image_path),
        declared_image_width=2,
        declared_image_height=2,
        decoded_image_width=2,
        decoded_image_height=2,
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        generation_policy=GenerationPolicy(max_new_tokens=1),
        expected_image_grid_thw=(1, 1, 2),
        logical_transform_id="identity",
    )


def _session(
    *,
    model: FakeEvidenceModel | None = None,
    processor: FakeProcessor | None = None,
    tokenizer: FakeTokenizer | None = None,
) -> Any:
    from src.inference.hf_backend import HFBackendSession

    return HFBackendSession(
        launch=_launch(),
        model=model or FakeEvidenceModel(),
        processor=processor or FakeProcessor(),
        tokenizer=tokenizer or FakeTokenizer(),
        receipt=_receipt(),
    )


def test_exact_history_prepare_and_literal_immutable_append(tmp_path: Any) -> None:
    processor = FakeProcessor()
    tokenizer = FakeTokenizer()
    session = _session(processor=processor, tokenizer=tokenizer)

    parent = session.prepare_exact_history(_request(tmp_path))
    decode_calls_before_append = list(tokenizer.decode_calls)
    child = session.extend_exact_history(parent, (13, 1, 3))

    assert parent.request_id == "exact-history"
    assert parent.conditioning_token_ids == (11, 12)
    assert parent.conditioning_token_ids_sha256 == token_ids_sha256((11, 12))
    assert child.conditioning_token_ids == (11, 12, 13, 1, 3)
    assert child.conditioning_token_ids_sha256 == token_ids_sha256(
        (11, 12, 13, 1, 3)
    )
    assert parent.conditioning_token_ids == (11, 12)
    assert child is not parent
    assert tokenizer.decode_calls == decode_calls_before_append
    assert session.special_token_ids == {"im_end": 31, "pad": 0}
    public = {
        name: value for name, value in vars(parent).items() if not name.startswith("_")
    }
    assert set(public) == {
        "request_id",
        "conditioning_token_ids",
        "conditioning_token_ids_sha256",
    }
    assert not hasattr(parent, "_session_binding")
    assert not any(isinstance(value, torch.Tensor) for value in public.values())


@pytest.mark.parametrize("bad_token_ids", [(-1,), (32,), (True,), (1.5,), ("1",)])
def test_exact_history_rejects_invalid_literal_token_ids_before_forward(
    tmp_path: Any,
    bad_token_ids: tuple[Any, ...],
) -> None:
    model = FakeEvidenceModel()
    session = _session(model=model)
    history = session.prepare_exact_history(_request(tmp_path))

    with pytest.raises(RuntimeContractError) as exc_info:
        session.extend_exact_history(history, bad_token_ids)

    assert exc_info.value.code == "hf_backend.invalid_token_id"
    assert model.forward_calls == []


def test_exact_history_rejects_forged_cross_session_and_closed_use(
    tmp_path: Any,
) -> None:
    from src.inference.hf_backend import HFExactHistory

    first_model = FakeEvidenceModel()
    second_model = FakeEvidenceModel()
    first = _session(model=first_model)
    second = _session(model=second_model)
    history = first.prepare_exact_history(_request(tmp_path, request_id="owned"))
    forged = HFExactHistory(
        request_id=history.request_id,
        conditioning_token_ids=history.conditioning_token_ids,
        conditioning_token_ids_sha256=history.conditioning_token_ids_sha256,
    )

    with pytest.raises(RuntimeContractError) as forged_exc:
        first.teacher_forced_evidence(forged, (1,))
    with pytest.raises(RuntimeContractError) as cross_exc:
        second.teacher_forced_evidence(history, (1,))
    first.close()
    with pytest.raises(RuntimeContractError) as closed_exc:
        first.teacher_forced_evidence(history, (1,))

    assert forged_exc.value.code == "hf_backend.exact_history_session"
    assert cross_exc.value.code == "hf_backend.exact_history_session"
    assert closed_exc.value.code == "hf_backend.session_closed"
    assert first_model.forward_calls == []
    assert second_model.forward_calls == []


def test_exact_history_rejects_unverifiable_vocabulary_before_forward(
    tmp_path: Any,
) -> None:
    model = FakeEvidenceModel()
    model.get_output_embeddings = lambda: None  # type: ignore[method-assign]
    model.config.vocab_size = None
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = None  # type: ignore[assignment]
    session = _session(model=model, tokenizer=tokenizer)

    with pytest.raises(RuntimeContractError) as exc_info:
        session.prepare_exact_history(_request(tmp_path))

    assert exc_info.value.code == "hf_backend.vocab_size_unavailable"
    assert model.forward_calls == []


def test_exact_history_registry_releases_dead_cases_and_clears_live_context(
    tmp_path: Any,
) -> None:
    session = _session()
    parent = session.prepare_exact_history(_request(tmp_path))
    child = session.extend_exact_history(parent, (13,))

    assert len(session._exact_history_states) == 2  # noqa: SLF001
    del parent
    del child
    gc.collect()
    assert len(session._exact_history_states) == 0  # noqa: SLF001

    live = session.prepare_exact_history(_request(tmp_path, request_id="live"))
    context = session._exact_history_states[live].context  # noqa: SLF001
    assert context.native_inputs is not None
    session.close()
    assert context.native_inputs is None


def test_teacher_forced_evidence_builds_positions_at_use_and_returns_only_evidence(
    tmp_path: Any,
) -> None:
    model = FakeEvidenceModel()
    session = _session(model=model)
    parent = session.prepare_exact_history(_request(tmp_path))
    history = session.extend_exact_history(parent, (13,))

    assert model.rope_calls == []
    one_token = session.teacher_forced_evidence(history, (1,))
    multi_token = session.teacher_forced_evidence(history, (1, 2, 3))

    assert [item.token_id for item in one_token] == [1]
    assert [item.token_id for item in multi_token] == [1, 2, 3]
    expected_logits = torch.full((32,), -1.0, dtype=torch.float32)
    expected_logits[0] = 3.0
    expected_logits[1] = 2.0
    expected_logits[2] = 2.0
    expected_logits[3] = 1.0
    expected_logprobs = torch.log_softmax(expected_logits, dim=-1)
    assert one_token[0].raw_model_logprob == pytest.approx(
        float(expected_logprobs[1])
    )
    assert [item.raw_model_logprob for item in multi_token] == pytest.approx(
        [
            float(expected_logprobs[1]),
            float(expected_logprobs[2]),
            float(expected_logprobs[3]),
        ]
    )
    assert [item.candidate_vocab_rank for item in multi_token] == [2, 2, 4]
    assert set(vars(multi_token[0])) == {
        "token_id",
        "raw_model_logprob",
        "candidate_vocab_rank",
    }
    assert not hasattr(multi_token[0], "logits")
    assert [call["input_ids"].tolist() for call in model.rope_calls] == [
        [[11, 12, 13, 1]],
        [[11, 12, 13, 1, 2, 3]],
    ]
    assert [call["attention_mask"].tolist() for call in model.rope_calls] == [
        [[1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1]],
    ]
    assert len(model.forward_calls) == 2
    assert all(call["use_cache"] is False for call in model.forward_calls)
    assert all(call["return_dict"] is True for call in model.forward_calls)
    assert all(call["logits_to_keep"] == 0 for call in model.forward_calls)
    assert parent.conditioning_token_ids == (11, 12)
    assert history.conditioning_token_ids == (11, 12, 13)


def test_teacher_forced_evidence_rejects_empty_or_invalid_before_forward(
    tmp_path: Any,
) -> None:
    model = FakeEvidenceModel()
    session = _session(model=model)
    history = session.prepare_exact_history(_request(tmp_path))

    with pytest.raises(RuntimeContractError) as empty_exc:
        session.teacher_forced_evidence(history, ())
    with pytest.raises(RuntimeContractError) as invalid_exc:
        session.teacher_forced_evidence(history, (32,))

    assert empty_exc.value.code == "hf_backend.teacher_forced_empty"
    assert invalid_exc.value.code == "hf_backend.invalid_token_id"
    assert model.rope_calls == []
    assert model.forward_calls == []


def test_open_hf_session_receipt_records_observed_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference.hf_backend import open_hf_backend_session

    monkeypatch.setattr("src.inference.hf_backend.metadata.version", lambda _: "test")
    monkeypatch.setattr("src.inference.hf_backend.torch.cuda.is_available", lambda: False)
    model = FakeEvidenceModel(
        observed_attention="sdpa",
        parameter_specs=((torch.float32, 3), (torch.float16, 2)),
    )
    loaded = SimpleNamespace(
        qwen=SimpleNamespace(
            model=model,
            processor=FakeProcessor(),
            tokenizer=FakeTokenizer(),
            token_identity=None,
            tokenizer_sha256="tokenizer",
            processor_identity=None,
            model_identity=None,
        ),
        adapter_receipt=None,
        embedding_delta_receipt=None,
    )

    session = open_hf_backend_session(_launch(), components_loader=lambda _: loaded)
    settings = session.receipt.effective_settings

    assert settings["observed_model_dtype"] == {
        "parameter_dtype_counts": {"torch.float16": 2, "torch.float32": 3},
        "parameter_dtype_names": ["torch.float16", "torch.float32"],
    }
    assert settings["observed_attn_implementation"] == "sdpa"
    backend_options = settings["backend_options"]
    assert isinstance(backend_options, dict)
    assert backend_options["attn_implementation"] == "eager"


def test_open_hf_session_receipt_uses_null_for_unavailable_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference.hf_backend import open_hf_backend_session

    monkeypatch.setattr("src.inference.hf_backend.metadata.version", lambda _: "test")
    monkeypatch.setattr("src.inference.hf_backend.torch.cuda.is_available", lambda: False)
    model = FakeEvidenceModel(observed_attention=None, parameter_specs=())
    loaded = SimpleNamespace(
        qwen=SimpleNamespace(
            model=model,
            processor=FakeProcessor(),
            tokenizer=FakeTokenizer(),
            token_identity=None,
            tokenizer_sha256="tokenizer",
            processor_identity=None,
            model_identity=None,
        ),
        adapter_receipt=None,
        embedding_delta_receipt=None,
    )

    session = open_hf_backend_session(_launch(), components_loader=lambda _: loaded)
    settings = session.receipt.effective_settings

    assert settings["observed_model_dtype"] is None
    assert settings["observed_attn_implementation"] is None
    backend_options = settings["backend_options"]
    assert isinstance(backend_options, dict)
    assert backend_options["attn_implementation"] == "eager"
