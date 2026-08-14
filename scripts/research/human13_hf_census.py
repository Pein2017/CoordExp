"""Exact Source-HF scorer for the Human-13 no-update census.

The scorer deliberately reuses the inference backend's verified multimodal
materialization, then appends the already encoded continuation as literal
vocabulary IDs.  It never generates or re-tokenizes a candidate suffix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from scripts.research.human13_live_model import (
    HUMAN13_IMAGE_IDS,
    HUMAN13_PANEL_PATH,
    HUMAN13_SOURCE_INFER_CONFIG,
    SOURCE_ADAPTER_PATH,
    SOURCE_BASE_MODEL_PATH,
    SOURCE_SPECIAL_EMBEDDING_PATH,
)
from src.inference.backend import DecodeRequest, token_ids_sha256
from src.inference.hf_backend import (
    _derive_qwen_position_ids,
    _model_device,
    _tensor_to_device_or_none,
)


@dataclass(frozen=True)
class HFCausalLogits:
    """Full-vocabulary logits at explicitly named causal positions."""

    logits: torch.Tensor
    logits_position_ids: tuple[int, ...]


class Human13HFCensusScorer:
    """Session-bound exact fp32/SDPA scorer for processor-built segments."""

    model_dtype = "torch.float32"
    attention_implementation = "sdpa"

    def __init__(
        self,
        *,
        session: Any,
        requests_by_image: Mapping[int, DecodeRequest],
        launch: Any | None = None,
    ) -> None:
        self._session = session
        self._launch = (
            launch if launch is not None else getattr(session, "_launch", None)
        )
        self._validate_launch()
        self._model = getattr(session, "_model", None)
        self._tokenizer = getattr(session, "_tokenizer", None)
        if self._model is None or self._tokenizer is None:
            raise ValueError("HF census scorer requires a live HF model and tokenizer")
        if (
            not callable(getattr(session, "prepare_exact_history", None))
            or not callable(getattr(session, "extend_exact_history", None))
            or not callable(getattr(session, "_validated_exact_history", None))
        ):
            raise ValueError("HF census scorer requires the exact-history backend seam")

        normalized: dict[int, DecodeRequest] = {}
        for image_id, request in requests_by_image.items():
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id <= 0
                or not isinstance(request, DecodeRequest)
            ):
                raise ValueError(
                    "HF census requests must be keyed by positive image IDs"
                )
            normalized[image_id] = request
        if not normalized:
            raise ValueError("HF census scorer requires canonical panel requests")
        self._requests_by_image = normalized

    def _validate_launch(self) -> None:
        launch = self._launch
        nested = getattr(launch, "backend_options", {}).get("hf", {})
        if (
            getattr(launch, "backend", None) != "hf"
            or getattr(launch, "model_dtype", None) != "fp32"
            or nested.get("attn_implementation") != "sdpa"
        ):
            raise ValueError("HF census scorer requires the Source fp32/SDPA launch")
        receipt = getattr(self._session, "receipt", None)
        settings = getattr(receipt, "effective_settings", {})
        observed_dtype = settings.get("observed_model_dtype")
        observed_names = (
            observed_dtype.get("parameter_dtype_names")
            if isinstance(observed_dtype, Mapping)
            else None
        )
        if observed_names != ["torch.float32"]:
            raise ValueError("HF census observed runtime must be exclusively fp32")
        if settings.get("observed_attn_implementation") != "sdpa":
            raise ValueError("HF census observed runtime must use SDPA")

    def _resolve_request(
        self, encoded_example: Any
    ) -> tuple[int, DecodeRequest, tuple[int, ...]]:
        input_ids_value = getattr(encoded_example, "input_ids", None)
        prompt_count = getattr(encoded_example, "prompt_token_count", None)
        if (
            not isinstance(input_ids_value, tuple)
            or isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count <= 0
            or prompt_count >= len(input_ids_value)
        ):
            raise ValueError("HF census encoded example has an invalid prompt boundary")
        input_ids = tuple(int(value) for value in input_ids_value)
        prompt = input_ids[:prompt_count]
        image_id = getattr(encoded_example, "human13_image_id", None)
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id not in self._requests_by_image
        ):
            raise ValueError("HF census encoded image identity is not canonical")
        request = self._requests_by_image[image_id]
        expected_prompt = tuple(request.expected_executed_prompt_token_ids)
        if (
            len(expected_prompt) != prompt_count
            or prompt != expected_prompt
            or token_ids_sha256(prompt) != token_ids_sha256(expected_prompt)
        ):
            raise ValueError(
                "HF census encoded prompt identity differs from canonical requests"
            )
        observed_grid = getattr(encoded_example, "image_grid_thw", None)
        if observed_grid is None:
            observed_grid = getattr(
                getattr(encoded_example, "image_encoding", None),
                "image_grid_thw",
                None,
            )
        if request.expected_image_grid_thw is not None and (
            tuple(observed_grid) if observed_grid is not None else None
        ) != tuple(request.expected_image_grid_thw):
            raise ValueError(
                "HF census encoded image identity differs from request grid"
            )
        return image_id, request, input_ids

    def score_causal_logits(
        self,
        encoded_example: Any,
        causal_positions: tuple[int, ...],
    ) -> HFCausalLogits:
        """Score exact logits[position] for a literal native-token segment."""

        return self._score_causal_logits(
            encoded_example, causal_positions, retain_gradient=False
        )

    def score_causal_logits_with_grad(
        self,
        encoded_example: Any,
        causal_positions: tuple[int, ...],
    ) -> HFCausalLogits:
        """Use the same fp32/SDPA history seam while retaining autograd."""

        return self._score_causal_logits(
            encoded_example, causal_positions, retain_gradient=True
        )

    def _score_causal_logits(
        self,
        encoded_example: Any,
        causal_positions: tuple[int, ...],
        *,
        retain_gradient: bool,
    ) -> HFCausalLogits:
        _image_id, request, input_ids = self._resolve_request(encoded_example)
        if (
            not isinstance(causal_positions, tuple)
            or not causal_positions
            or any(
                isinstance(position, bool) or not isinstance(position, int)
                for position in causal_positions
            )
            or tuple(sorted(set(causal_positions))) != causal_positions
        ):
            raise ValueError("HF census causal positions must be sorted and unique")
        prompt_count = len(request.expected_executed_prompt_token_ids)
        if any(
            position < prompt_count - 1 or position >= len(input_ids) - 1
            for position in causal_positions
        ):
            raise ValueError("HF census causal position escapes the native segment")

        prompt_history = self._session.prepare_exact_history(request)
        if tuple(prompt_history.conditioning_token_ids) != tuple(
            request.expected_executed_prompt_token_ids
        ):
            raise ValueError("HF census executed prompt identity differs from request")
        continuation = input_ids[prompt_count:]
        history = self._session.extend_exact_history(prompt_history, continuation)
        if tuple(history.conditioning_token_ids) != input_ids:
            raise ValueError("HF census literal continuation identity differs")
        state = self._session._validated_exact_history(history)  # noqa: SLF001
        native_inputs = state.context.native_inputs
        if not isinstance(native_inputs, Mapping):
            raise ValueError("HF census exact history lost its multimodal inputs")

        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("HF census scorer was released before scoring")
        device = _model_device(model)
        native_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(native_input_ids, dtype=torch.long)
        image_grid_thw = native_inputs.get("image_grid_thw")
        if not isinstance(image_grid_thw, torch.Tensor):
            raise ValueError("HF census exact history lacks image_grid_thw")
        position_ids = _derive_qwen_position_ids(
            model=model,
            input_ids=native_input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw.to(device=device),
            video_grid_thw=_tensor_to_device_or_none(
                native_inputs.get("video_grid_thw"), device=device
            ),
        )
        forward_inputs = {
            key: value
            for key, value in native_inputs.items()
            if key
            not in {
                "input_ids",
                "attention_mask",
                "position_ids",
                "token_type_ids",
                "cache_position",
                "rope_deltas",
            }
        }
        forward_inputs.update(
            {
                "input_ids": native_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
                "return_dict": True,
                "logits_to_keep": torch.tensor(
                    causal_positions, dtype=torch.long, device=device
                ),
            }
        )
        context = torch.enable_grad() if retain_gradient else torch.inference_mode()
        with context:
            output = model(**forward_inputs)
        logits = getattr(output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or tuple(logits.shape[:2]) != (1, len(causal_positions))
        ):
            raise ValueError(
                "HF census model did not return position-selective causal logits"
            )
        try:
            expected_vocab_size = len(tokenizer)
        except (AttributeError, TypeError) as exc:
            raise ValueError("HF census tokenizer lacks its full vocabulary") from exc
        if int(logits.shape[2]) != expected_vocab_size:
            raise ValueError("HF census model did not return the full vocabulary")
        if logits.dtype != torch.float32:
            raise ValueError("HF census model logits must be fp32")
        return HFCausalLogits(
            logits=(logits if retain_gradient else logits.detach().cpu().contiguous()),
            logits_position_ids=causal_positions,
        )

    def exact_history_context(
        self, encoded_example: Any
    ) -> tuple[Any, Mapping[str, Any]]:
        """Return verified native multimodal inputs for a literal continuation."""

        _image_id, request, input_ids = self._resolve_request(encoded_example)
        history = self._session.prepare_exact_history(request)
        prompt_count = len(request.expected_executed_prompt_token_ids)
        history = self._session.extend_exact_history(history, input_ids[prompt_count:])
        state = self._session._validated_exact_history(history)  # noqa: SLF001
        native_inputs = state.context.native_inputs
        if not isinstance(native_inputs, Mapping):
            raise ValueError("HF exact history lost its multimodal inputs")
        return self._session, dict(native_inputs)


def _load_source_inputs(repo_root: str | Path) -> tuple[Any, dict[int, DecodeRequest]]:
    """Build the frozen processor-only frontend and canonical request map."""

    from scripts.research.run_current_seeded_sampled_rollouts import (
        _build_requests,
        physical_image_id,
    )
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend

    root = Path(repo_root).expanduser().resolve(strict=True)
    config = load_infer_config(root / HUMAN13_SOURCE_INFER_CONFIG).config
    if (
        config.backend.type != "hf"
        or config.model.dtype != "fp32"
        or config.backend.hf.attn_implementation != "sdpa"
        or str(Path(config.model.base_model).resolve()) != SOURCE_BASE_MODEL_PATH
        or str(Path(config.data.input_jsonl).resolve()) != HUMAN13_PANEL_PATH
        or config.adapter is None
        or str(Path(config.adapter.path).resolve()) != SOURCE_ADAPTER_PATH
        or config.embedding_delta is None
        or str(Path(config.embedding_delta.path).resolve())
        != SOURCE_SPECIAL_EMBEDDING_PATH
    ):
        raise ValueError("Human-13 Source HF census configuration drifted")
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    examples = tuple(load_raw_examples(config.data.input_jsonl))
    requests, _prompt_metadata = _build_requests(config, frontend, examples)
    image_ids = tuple(int(physical_image_id(example)) for example in examples)
    if image_ids != HUMAN13_IMAGE_IDS or len(requests) != len(image_ids):
        raise ValueError("Human-13 Source HF requests differ from the canonical panel")
    by_image = {
        image_id: request for image_id, request in zip(image_ids, requests, strict=True)
    }
    if len(by_image) != len(HUMAN13_IMAGE_IDS):
        raise ValueError("Human-13 Source HF request image identities are not unique")
    return frontend.launch, by_image


@contextmanager
def open_source_hf_census_scorer(
    *,
    repo_root: str | Path,
    session_context_factory: Callable[[Any], Any] | None = None,
) -> Iterator[Human13HFCensusScorer]:
    """Open the frozen Source HF session and guarantee backend cleanup."""

    from src.inference.backend import open_backend_session

    launch, requests_by_image = _load_source_inputs(repo_root)
    factory = session_context_factory or open_backend_session
    with factory(launch) as session:
        yield Human13HFCensusScorer(
            session=session,
            requests_by_image=requests_by_image,
            launch=launch,
        )


@contextmanager
def open_checkpoint_hf_census_scorer(
    *,
    repo_root: str | Path,
    checkpoint_path: str | Path,
    session_context_factory: Callable[[Any], Any] | None = None,
) -> Iterator[Human13HFCensusScorer]:
    """Open exact HF scoring on a private proposal or accepted checkpoint.

    The frozen Source frontend still owns base model, panel, prompt, tokenizer,
    fp32 dtype and SDPA. Only adapter and special-token payload paths are
    rebound to the caller's already materialized transaction checkpoint.
    """

    from src.inference.backend import open_backend_session

    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("proposal checkpoint must be a regular directory")
    checkpoint = checkpoint.resolve(strict=True)
    adapter_path = checkpoint / "adapter"
    delta_path = checkpoint / "special_token_embeddings"
    if (
        adapter_path.is_symlink()
        or delta_path.is_symlink()
        or not adapter_path.is_dir()
        or not delta_path.is_dir()
    ):
        raise ValueError("proposal checkpoint lacks adapter or special-token payload")
    launch, requests_by_image = _load_source_inputs(repo_root)
    adapter = dict(getattr(launch, "adapter", {}) or {})
    embedding_delta = dict(getattr(launch, "embedding_delta", {}) or {})
    adapter["path"] = str(adapter_path)
    embedding_delta["path"] = str(delta_path)
    bound_launch: Any
    if is_dataclass(launch) and not isinstance(launch, type):
        bound_launch = replace(
            cast(Any, launch),
            adapter=adapter,
            embedding_delta=embedding_delta,
        )
    else:
        bound_launch = SimpleNamespace(
            **{
                **vars(launch),
                "adapter": adapter,
                "embedding_delta": embedding_delta,
            }
        )
    factory: Callable[[Any], Any] = session_context_factory or open_backend_session
    with factory(bound_launch) as session:
        yield Human13HFCensusScorer(
            session=session,
            requests_by_image=requests_by_image,
            launch=bound_launch,
        )


__all__ = [
    "HFCausalLogits",
    "Human13HFCensusScorer",
    "open_checkpoint_hf_census_scorer",
    "open_source_hf_census_scorer",
]
