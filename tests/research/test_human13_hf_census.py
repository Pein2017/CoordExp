from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest
import torch

from src.inference.backend import (
    POLICY_LIKELIHOOD_DEFINITION,
    RAW_LIKELIHOOD_DEFINITION,
    BackendLaunch,
    BackendSessionReceipt,
    DecodeRequest,
    GenerationPolicy,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 31

    def __init__(self) -> None:
        self.padding_side = "right"
        self.decode_calls: list[list[int]] = []

    def __len__(self) -> int:
        return 32

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return self.eos_token_id

    def decode(self, token_ids: list[int], **_: Any) -> str:
        self.decode_calls.append(list(token_ids))
        return "decoded"


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.ones((2, 4), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
        }


class PositionModel:
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        logits_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = self
        self.config = SimpleNamespace(
            vocab_size=vocab_size, _attn_implementation="sdpa"
        )
        self._parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.vocab_size = vocab_size
        self.logits_dtype = logits_dtype
        self.forward_calls: list[dict[str, Any]] = []

    def parameters(self) -> Any:
        return iter((self._parameter,))

    def get_output_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(weight=torch.empty((self.vocab_size, 1)))

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        del image_grid_thw, video_grid_thw, attention_mask
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, 1, -1), None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        logits = torch.zeros(
            (1, input_ids.shape[1], self.vocab_size),
            dtype=self.logits_dtype,
            device=input_ids.device,
        )
        for position in range(input_ids.shape[1]):
            logits[0, position, 0] = position
        return SimpleNamespace(logits=logits)


def _launch() -> BackendLaunch:
    return BackendLaunch(
        backend="hf",
        model_path="/unused/source-model",
        model_dtype="fp32",
        batch_size=1,
        generation_config_fingerprint="source-generation",
        backend_options={
            "hf": {
                "attn_implementation": "sdpa",
                "patch_embed_linearization": "enabled",
            }
        },
    )


def _receipt(
    *,
    observed_dtypes: list[str] | None = None,
    observed_attention: str = "sdpa",
) -> BackendSessionReceipt:
    return BackendSessionReceipt(
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        backend_version="test",
        model_identity={"family": "source"},
        tokenizer_identity={"sha256": "tokenizer"},
        processor_identity={"class": "FakeProcessor"},
        generation_config_fingerprint="source-generation",
        effective_settings={
            "batch_size": 1,
            "observed_model_dtype": {
                "parameter_dtype_names": observed_dtypes or ["torch.float32"]
            },
            "observed_attn_implementation": observed_attention,
        },
        likelihood_semantics={
            "policy": POLICY_LIKELIHOOD_DEFINITION,
            "raw": RAW_LIKELIHOOD_DEFINITION,
            "score_owned_channel": "policy_logprob",
        },
    )


def _request(
    tmp_path: Any,
    *,
    image_id: int = 1,
    color: str | tuple[int, int, int] = "white",
) -> DecodeRequest:
    path = tmp_path / f"{image_id}.png"
    Image.new("RGB", (2, 2), color=color).save(path)
    return DecodeRequest(
        request_id=str(image_id),
        chat_text="canonical prompt",
        input_prompt_token_ids=(11,),
        expected_executed_prompt_token_ids=(11, 12),
        image_path=str(path),
        declared_image_width=2,
        declared_image_height=2,
        decoded_image_width=2,
        decoded_image_height=2,
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generation_policy=GenerationPolicy(max_new_tokens=1),
        expected_image_grid_thw=(1, 1, 2),
    )


def _session(
    *,
    model: PositionModel | None = None,
    processor: FakeProcessor | None = None,
    tokenizer: FakeTokenizer | None = None,
    receipt: BackendSessionReceipt | None = None,
) -> Any:
    from src.inference.hf_backend import HFBackendSession

    return HFBackendSession(
        launch=_launch(),
        model=model or PositionModel(),
        processor=processor or FakeProcessor(),
        tokenizer=tokenizer or FakeTokenizer(),
        receipt=receipt or _receipt(),
    )


@dataclass(frozen=True)
class Encoded:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_grid_thw: tuple[int, int, int] = (1, 1, 2)
    human13_image_id: int = 1


def test_scorer_selects_exact_causal_rows_from_literal_continuation(
    tmp_path: Any,
) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    model = PositionModel()
    processor = FakeProcessor()
    tokenizer = FakeTokenizer()
    scorer = Human13HFCensusScorer(
        session=_session(model=model, processor=processor, tokenizer=tokenizer),
        requests_by_image={1: _request(tmp_path)},
    )

    output = scorer.score_causal_logits(
        Encoded("a1:1", (11, 12, 13, 14, 15), 2),
        (1, 3),
    )

    assert output.logits.shape == (1, 2, 32)
    assert output.logits.dtype == torch.float32
    assert output.logits_position_ids == (1, 3)
    assert output.logits[0, :, 0].tolist() == [1.0, 3.0]
    assert model.forward_calls[0]["input_ids"].tolist() == [[11, 12, 13, 14, 15]]
    assert model.forward_calls[0]["logits_to_keep"] == 0
    assert len(processor.calls) == 1
    assert tokenizer.decode_calls == []


def test_scorer_reads_grid_from_canonical_encoded_image_encoding(tmp_path: Any) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    model = PositionModel()
    scorer = Human13HFCensusScorer(
        session=_session(model=model),
        requests_by_image={1: _request(tmp_path)},
    )
    encoded = SimpleNamespace(
        example_id="a1:1",
        input_ids=(11, 12, 13),
        prompt_token_count=2,
        human13_image_id=1,
        image_encoding=SimpleNamespace(image_grid_thw=(1, 1, 2)),
    )

    output = scorer.score_causal_logits(encoded, (1,))

    assert output.logits_position_ids == (1,)
    assert len(model.forward_calls) == 1


def test_scorer_routes_thirteen_shared_prompts_by_true_image_identity(
    tmp_path: Any,
) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    session = _session()
    prepared_request_ids: list[str] = []
    original_prepare = session.prepare_exact_history

    def record_prepare(request: DecodeRequest) -> Any:
        prepared_request_ids.append(request.request_id)
        return original_prepare(request)

    session.prepare_exact_history = record_prepare
    image_ids = tuple(range(100, 113))
    requests = {
        image_id: _request(
            tmp_path,
            image_id=image_id,
            color=(image_id - 100, 0, 0),
        )
        for image_id in image_ids
    }
    assert (
        len(
            {
                request.expected_executed_prompt_token_ids
                for request in requests.values()
            }
        )
        == 1
    )
    assert len({request.image_sha256 for request in requests.values()}) == 13

    scorer = Human13HFCensusScorer(
        session=session,
        requests_by_image=requests,
    )
    scorer.score_causal_logits(
        Encoded(
            "a1:107",
            (11, 12, 13),
            2,
            human13_image_id=107,
        ),
        (1,),
    )

    assert prepared_request_ids == ["107"]


@pytest.mark.parametrize(
    "encoded",
    (
        Encoded("a1:1", (11, 99, 13), 2),
        Encoded("a1:1", (11, 12, 13), 1),
        Encoded("a1:1", (11, 12, 13), 2, image_grid_thw=(1, 2, 2)),
    ),
)
def test_scorer_rejects_prompt_or_image_identity_drift_before_forward(
    tmp_path: Any,
    encoded: Encoded,
) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    model = PositionModel()
    scorer = Human13HFCensusScorer(
        session=_session(model=model),
        requests_by_image={1: _request(tmp_path)},
    )

    with pytest.raises(ValueError, match="prompt|image"):
        scorer.score_causal_logits(encoded, (1,))

    assert model.forward_calls == []


@pytest.mark.parametrize(
    ("model", "message"),
    (
        (PositionModel(vocab_size=31), "full vocabulary"),
        (PositionModel(logits_dtype=torch.float16), "fp32"),
    ),
)
def test_scorer_rejects_non_full_vocab_or_non_fp32_logits(
    tmp_path: Any,
    model: PositionModel,
    message: str,
) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    scorer = Human13HFCensusScorer(
        session=_session(model=model),
        requests_by_image={1: _request(tmp_path)},
    )

    with pytest.raises(ValueError, match=message):
        scorer.score_causal_logits(
            Encoded("a1:1", (11, 12, 13), 2),
            (1,),
        )


@pytest.mark.parametrize(
    "receipt",
    (
        _receipt(observed_dtypes=["torch.bfloat16", "torch.float32"]),
        _receipt(observed_attention="flash_attention_2"),
    ),
)
def test_scorer_rejects_non_exact_observed_runtime(
    tmp_path: Any,
    receipt: BackendSessionReceipt,
) -> None:
    from scripts.research.human13_hf_census import Human13HFCensusScorer

    with pytest.raises(ValueError, match="observed.*fp32|observed.*SDPA"):
        Human13HFCensusScorer(
            session=_session(receipt=receipt),
            requests_by_image={1: _request(tmp_path)},
        )


def test_source_context_opener_owns_session_cleanup(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from scripts.research import human13_hf_census as census

    session = _session()
    original_close = session.close
    close_calls = 0

    def close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    session.close = close

    @contextmanager
    def fake_session_context(_: Any) -> Any:
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(
        census,
        "_load_source_inputs",
        lambda _: (_launch(), {1: _request(tmp_path)}),
    )

    with census.open_source_hf_census_scorer(
        repo_root="/repo",
        session_context_factory=fake_session_context,
    ) as scorer:
        assert scorer.model_dtype == "torch.float32"
        assert scorer.attention_implementation == "sdpa"

    assert close_calls == 1


def test_checkpoint_context_opener_rebinds_only_adapter_payload(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from scripts.research import human13_hf_census as census

    checkpoint = tmp_path / "proposal"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "special_token_embeddings").mkdir()
    launch = _launch()
    launch_values = dict(vars(launch))
    launch_values.update(
        adapter={"path": "/source/adapter", "dtype": "fp32"},
        embedding_delta={"path": "/source/delta"},
    )
    launch = SimpleNamespace(**launch_values)
    session = _session()
    observed_launches: list[Any] = []

    @contextmanager
    def fake_session_context(bound_launch: Any) -> Any:
        observed_launches.append(bound_launch)
        yield session

    monkeypatch.setattr(
        census,
        "_load_source_inputs",
        lambda _: (launch, {1: _request(tmp_path)}),
    )

    with census.open_checkpoint_hf_census_scorer(
        repo_root="/repo",
        checkpoint_path=checkpoint,
        session_context_factory=fake_session_context,
    ) as scorer:
        assert scorer.model_dtype == "torch.float32"

    assert observed_launches[0].model_path == launch.model_path
    assert observed_launches[0].backend_options == launch.backend_options
    assert observed_launches[0].adapter["path"] == str(checkpoint / "adapter")
    assert observed_launches[0].embedding_delta["path"] == str(
        checkpoint / "special_token_embeddings"
    )


def test_checkpoint_context_rejects_missing_private_payload(tmp_path: Any) -> None:
    from scripts.research.human13_hf_census import open_checkpoint_hf_census_scorer

    with pytest.raises(ValueError, match="proposal checkpoint"):
        with open_checkpoint_hf_census_scorer(
            repo_root=tmp_path,
            checkpoint_path=tmp_path / "missing",
        ):
            pass
