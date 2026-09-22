"""Native paired O/N producer for the frozen fresh128 readout-norm panel.

This is deliberately a thin native caller.  Panel construction and reduction
belong to their respective owners; this module only produces durable paired
greedy continuations and same-history N shadow evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probes.dora_owner_learning.runtime import load_policy
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import NativeBatch, prepare_native_inputs
from src.qwen.special_token_embeddings import (
    SelectedDeltaInputEmbedding,
    SelectedDeltaOutputHead,
)


EOS = 151645
MAX_NEW_TOKENS = 3084
MAX_CAPTURED_SAMPLES = 144


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _input_identity(batch: NativeBatch) -> dict[str, Any]:
    tensors = {
        name: _tensor_hash(value)
        for name, value in sorted(batch.inputs.items())
        if isinstance(value, torch.Tensor)
    }
    return {
        "request_ids": list(batch.request_ids),
        "prompt_token_ids": [list(row) for row in batch.prompt_token_ids],
        "media_sha256": None if batch.media_sha256 is None else list(batch.media_sha256),
        "image_grids": [None if grid is None else list(grid) for grid in batch.image_grids],
        "tensor_sha256": tensors,
    }


def _image_id(case: Mapping[str, Any]) -> int:
    value = case.get("image_id")
    if value is None:
        record = case.get("input_record")
        if isinstance(record, Mapping):
            value = record.get("image_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("every panel case needs a stable integer image_id")
    return value


def _saved_rows(rows: object) -> dict[int, Mapping[str, Any]]:
    if rows is None:
        return {}
    if not isinstance(rows, list):
        raise ValueError("saved rows must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("saved row must be an object")
        image_id = row.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("saved row needs image_id")
        if image_id in result:
            raise ValueError("saved rows have duplicate image_id")
        result[image_id] = row
    return result


def _saved_tokens(row: Mapping[str, Any]) -> list[int]:
    value = row.get("token_ids", row.get("generated_token_ids"))
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError("saved row has no integer token sequence")
    return value


def _saved_stop(row: Mapping[str, Any]) -> str:
    value = row.get("stop", row.get("decode_stop_reason"))
    if not isinstance(value, str):
        raise ValueError("saved row has no stop reason")
    return value


def _role(history: Sequence[int], chosen: int, coordinate_ids: set[int]) -> str:
    if chosen == EOS:
        return "eos"
    # These are the frozen object-box grammar token IDs from the predecessor.
    box_start = max((i for i, token in enumerate(history) if token == 151648), default=-1)
    box_end = max((i for i, token in enumerate(history) if token == 151649), default=-1)
    if chosen in coordinate_ids:
        if box_start > box_end and 0 <= len(history) - box_start - 1 < 4:
            return ("x1", "y1", "x2", "y2")[len(history) - box_start - 1]
        return "coordinate_other"
    if chosen == 151646:
        return "row_opener"
    if chosen in (151647, 151648, 151649):
        return "syntax_delimiter"
    return "category_or_other"


def _top_two_margin(scores: torch.Tensor) -> float:
    values = torch.topk(scores, 2).values
    return float((values[0] - values[1]).item())


class _ShadowLedger:
    """Record N decisions at the logits-processor seam without extra forwards."""

    def __init__(
        self,
        *,
        prompt_width: int,
        image_ids: Sequence[int],
        coordinate_ids: set[int],
        eos_token_id: int,
        capture_dir: Path,
        captured: set[int],
    ) -> None:
        self.prompt_width = prompt_width
        self.image_ids = tuple(image_ids)
        self.coordinate_ids = coordinate_ids
        self.eos_token_id = eos_token_id
        self.capture_dir = capture_dir
        self.captured = captured
        self.done = [False] * len(image_ids)
        self.tokens = [[] for _ in image_ids]
        self.trace: list[list[dict[str, Any]]] = [[] for _ in image_ids]
        self.captures: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        *,
        input_ids: torch.Tensor,
        before: torch.Tensor,
        after: torch.Tensor,
        head_input: torch.Tensor,
    ) -> None:
        offset = input_ids.shape[1] - self.prompt_width
        if head_input.ndim != 2 or head_input.shape[0] != len(self.image_ids):
            raise AssertionError("lm_head hook did not expose one final hidden row per sample")
        for index, image_id in enumerate(self.image_ids):
            if self.done[index]:
                continue  # HF may continue this row with padding after its EOS.
            history = input_ids[index, self.prompt_width :].tolist()
            if history != self.tokens[index]:
                raise AssertionError("N shadow history is not its literal generated prefix")
            original = int(torch.argmax(before[index]).item())
            transformed = int(torch.argmax(after[index]).item())
            entry = {
                "offset": offset,
                "history_sha256": _json_hash(history),
                "role": _role(history, transformed, self.coordinate_ids),
                "original_full_vocab_argmax": original,
                "transformed_argmax": transformed,
                "original_top2_margin": _top_two_margin(before[index]),
                "transformed_top2_margin": _top_two_margin(after[index]),
            }
            self.trace[index].append(entry)
            if original != transformed and image_id not in self.captured:
                if len(self.captured) >= MAX_CAPTURED_SAMPLES:
                    raise AssertionError("first-shadow capture limit exceeded")
                path = self.capture_dir / f"first-shadow-{image_id}.pt"
                torch.save(
                    {
                        "offset": offset,
                        "original_full_vocab_argmax": original,
                        "transformed_argmax": transformed,
                        "before_logits": before[index].detach().cpu(),
                        "after_logits": after[index].detach().cpu(),
                        "lm_head_input": head_input[index].detach().cpu(),
                    },
                    path,
                )
                self.captured.add(image_id)
                self.captures[str(image_id)] = _binding(path)
            self.tokens[index].append(transformed)
            if transformed == self.eos_token_id:
                self.done[index] = True

    def result(self) -> dict[str, Any]:
        return {
            "argmax_tie_policy": "torch.argmax returns the first maximal index",
            "eos_included": True,
            "post_eos_padding_excluded": True,
            "samples": {
                str(image_id): {"token_ids": tokens, "steps": trace}
                for image_id, tokens, trace in zip(
                    self.image_ids, self.tokens, self.trace, strict=True
                )
            },
            "first_shadow_disagreement_tensors": self.captures,
        }


def _check_sources(panel: Mapping[str, Any]) -> None:
    sources = panel.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("panel sources must be a list")
    for expected in sources:
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ValueError("panel source binding is malformed")
        path = Path(expected["path"])
        if _binding(path) != dict(expected):
            raise AssertionError(f"frozen source changed: {path}")


def _load_coefficients(
    *,
    panel: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    expected = panel.get("coefficient_binding", panel.get("coefficients"))
    if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
        raise ValueError("panel needs a coefficient binding")
    path = Path(expected["path"])
    if _binding(path) != dict(expected):
        raise AssertionError("fixed coefficient binding changed")
    coordinate_list = panel.get("coordinate_ids")
    if not isinstance(coordinate_list, list) or len(coordinate_list) != 1000:
        raise ValueError("panel must bind all 1000 coordinate IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate_list):
        raise ValueError("coordinate IDs must be integers")
    coordinate_ids = torch.tensor(coordinate_list, device=device, dtype=torch.long)
    expected_ids = [tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>") for index in range(1000)]
    if coordinate_ids.tolist() != expected_ids:
        raise AssertionError("loaded tokenizer coordinate IDs differ from panel")
    head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    if not isinstance(head, SelectedDeltaOutputHead) or not isinstance(
        embedding, SelectedDeltaInputEmbedding
    ):
        raise AssertionError("loaded model lacks the selected delta input/output seam")
    if head.bias is not None:
        raise AssertionError("readout bias must be absent")
    lookup = {int(token): index for index, token in enumerate(head.selected_token_ids.tolist())}
    if any(int(token) not in lookup for token in coordinate_ids.tolist()):
        raise AssertionError("coordinate IDs are not covered by shared output delta")
    selected = torch.tensor(
        [lookup[int(token)] for token in coordinate_ids.tolist()], device=device
    )
    effective = head.base.weight[coordinate_ids].detach() + head.shared_embed_delta[selected].detach()
    if not torch.equal(effective, embedding(coordinate_ids).detach()):
        raise AssertionError("effective output and input coordinate rows differ")
    coefficients = torch.load(path, map_location="cpu", weights_only=True)
    if _tensor_hash(effective) != coefficients["effective_rows_sha256"]:
        raise AssertionError("effective coordinate row hash differs from frozen coefficients")
    norms = effective.cpu().double().norm(dim=1)
    if not torch.equal(norms, coefficients["norms"]):
        raise AssertionError("effective coordinate row norms differ from frozen coefficients")
    if not torch.equal(norms.median() / norms, coefficients["factors"]):
        raise AssertionError("frozen norm factors differ from actual effective rows")
    factors = coefficients["factors"].to(device=device, dtype=torch.float64)
    receipt = {
        "coefficient_binding": _binding(path),
        "effective_rows_sha256": _tensor_hash(effective),
        "effective_row_norms_sha256": _tensor_hash(norms),
        "factor_sha256": _tensor_hash(factors),
        "effective_dtype": str(effective.dtype),
        "bias_exists": False,
        "base_weight_tied_to_input": head.base.weight.data_ptr() == embedding.base.weight.data_ptr(),
        "shared_delta_tied_to_input": head.shared_embed_delta.data_ptr()
        == embedding.shared_embed_delta.data_ptr(),
        "input_coordinate_sha256": _tensor_hash(embedding(coordinate_ids)),
        "base_coordinate_sha256": _tensor_hash(head.base.weight[coordinate_ids]),
        "shared_delta_sha256": _tensor_hash(head.shared_embed_delta),
    }
    return coordinate_ids, factors, receipt


def _run_policy(
    *,
    name: str,
    group: Mapping[str, Any],
    qwen: Any,
    config: Mapping[str, Any],
    coordinate_ids: torch.Tensor,
    factors: torch.Tensor,
    captured: set[int],
    output: Path,
    panel_binding: Mapping[str, Any],
    producer_binding: Mapping[str, Any],
    readout: Mapping[str, Any],
    initial_versions: Mapping[str, int],
) -> dict[str, Any]:
    cases = group.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("each group must contain exactly four cases")
    image_ids = [_image_id(case) for case in cases]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("group has duplicate image IDs")
    policy_dir = output / f"group-{group['key']}" / name
    policy_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "readout_norm_fresh128.native_group.v1",
        "status": "running",
        "group": group["key"],
        "policy": name,
        "batch_size": len(cases),
        "image_ids": image_ids,
        "panel": dict(panel_binding),
        "producer": dict(producer_binding),
        "readout": dict(readout),
        "raw_path": str(policy_dir / "raw.json"),
    }
    _write(policy_dir / "receipt.json", receipt)
    model = qwen.model
    tokenizer = qwen.tokenizer
    start = time.monotonic()
    handles: list[Any] = []
    model_forwards = 0
    last_head_input: torch.Tensor | None = None
    prefill_mrope: str | None = None
    try:
        request_config = dict(config)
        request_data = dict(config["data"])
        input_jsonl = group.get("input_jsonl", request_data["input_jsonl"])
        if not isinstance(input_jsonl, str):
            raise ValueError("group input_jsonl must be a path string")
        request_data["input_jsonl"] = input_jsonl
        request_config["data"] = request_data
        receipt["input_jsonl"] = input_jsonl
        requests, _ = build_bound_native_requests(qwen, request_config, cases)
        batch = prepare_native_inputs(
            qwen.processor, requests, device="cuda:0", record_media_identity=True
        )
        input_identity = _input_identity(batch)
        width = batch.inputs["input_ids"].shape[1]
        if width <= 0:
            raise AssertionError("prepared native batch has no prompt width")
        shadows = None
        coordinate_set = set(int(value) for value in coordinate_ids.tolist())
        if name == "N":
            shadows = _ShadowLedger(
                prompt_width=width,
                image_ids=image_ids,
                coordinate_ids=coordinate_set,
                eos_token_id=EOS,
                capture_dir=policy_dir,
                captured=captured,
            )

        def model_counter(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal model_forwards
            model_forwards += 1
            if model_forwards > MAX_NEW_TOKENS:
                raise AssertionError("generation exceeded frozen forward cap")

        def mrope_capture(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal prefill_mrope
            if model_forwards == 1:
                positions = kwargs.get("position_ids")
                if not isinstance(positions, torch.Tensor):
                    raise AssertionError("native prefill did not pass MRoPE position IDs")
                prefill_mrope = _tensor_hash(positions)

        def head_capture(module: Any, args: tuple[Any, ...]) -> None:
            nonlocal last_head_input
            if not args or not isinstance(args[0], torch.Tensor):
                raise AssertionError("lm_head forward input is not a tensor")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[0] != len(cases):
                raise AssertionError("lm_head input shape does not match the native batch")
            last_head_input = hidden[:, -1, :]

        handles.extend(
            [
                model.register_forward_pre_hook(model_counter, with_kwargs=True),
                model.model.language_model.register_forward_pre_hook(mrope_capture, with_kwargs=True),
                model.get_output_embeddings().register_forward_pre_hook(head_capture),
            ]
        )

        class NormProcessor(LogitsProcessor):
            def __call__(self, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal last_head_input
                transformed = scores.clone()
                transformed[:, coordinate_ids] = (
                    scores[:, coordinate_ids].double() * factors
                ).to(scores.dtype)
                # Assignment only targets coordinate columns; verify the actual seam,
                # including EOS, rather than relying on that implementation detail.
                changed = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
                changed[coordinate_ids] = True
                if not torch.equal(scores[:, ~changed], transformed[:, ~changed]):
                    raise AssertionError("non-coordinate logits changed at norm seam")
                if not torch.equal(scores[:, EOS], transformed[:, EOS]):
                    raise AssertionError("EOS logit changed at norm seam")
                if not torch.equal(
                    transformed[:, coordinate_ids],
                    (scores[:, coordinate_ids].double() * factors).to(scores.dtype),
                ):
                    raise AssertionError("coordinate norm formula changed at seam")
                if name == "N":
                    if last_head_input is None:
                        raise AssertionError("missing final lm_head input at logits seam")
                    assert shadows is not None
                    shadows.observe(
                        input_ids=ids,
                        before=scores,
                        after=transformed,
                        head_input=last_head_input,
                    )
                return transformed if name == "N" else scores

        original_generate = model.generate

        def generate_wrapper(**kwargs: Any) -> Any:
            if (
                kwargs.get("max_new_tokens") != MAX_NEW_TOKENS
                or kwargs.get("repetition_penalty") != 1
                or kwargs.get("do_sample")
                or "logits_processor" in kwargs
            ):
                raise AssertionError("native greedy generation settings changed")
            receipt["generate_settings"] = {
                key: kwargs.get(key)
                for key in (
                    "max_new_tokens",
                    "do_sample",
                    "repetition_penalty",
                    "eos_token_id",
                    "pad_token_id",
                    "use_model_defaults",
                    "return_dict_in_generate",
                    "output_scores",
                    "output_logits",
                )
            }
            processor = LogitsProcessorList([NormProcessor()]) if name == "N" else LogitsProcessorList()
            return original_generate(**kwargs, logits_processor=processor)

        model.generate = generate_wrapper
        try:
            with torch.no_grad():
                values = generate_continuations(
                    model,
                    batch,
                    extensions=[[] for _ in cases],
                    budgets=[MAX_NEW_TOKENS for _ in cases],
                    eos_token_id=EOS,
                    pad_token_id=tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(
                        temperature=0,
                        top_p=1,
                        top_k=0,
                        repetition_penalty=1,
                        use_model_defaults=False,
                    ),
                    trace="none",
                    seed=None,
                )
        finally:
            model.generate = original_generate
        if prefill_mrope is None:
            raise AssertionError("did not observe native prefill MRoPE")
        current_versions = {key: value._version for key, value in model.named_parameters()}
        if current_versions != dict(initial_versions):
            raise AssertionError("generation mutated model parameters")
        head = model.get_output_embeddings()
        embedding = model.get_input_embeddings()
        if not isinstance(head, SelectedDeltaOutputHead) or not isinstance(
            embedding, SelectedDeltaInputEmbedding
        ):
            raise AssertionError("readout seam changed during generation")
        if (
            _tensor_hash(embedding(coordinate_ids)) != readout["input_coordinate_sha256"]
            or _tensor_hash(head.base.weight[coordinate_ids]) != readout["base_coordinate_sha256"]
            or _tensor_hash(head.shared_embed_delta) != readout["shared_delta_sha256"]
        ):
            raise AssertionError("generation changed effective input/readout parameters")
        if _input_identity(batch) != input_identity:
            raise AssertionError("generation mutated native inputs")
        if len(values) != len(cases):
            raise AssertionError("generated batch cardinality changed")
        rows = []
        for image_id, request, value in zip(image_ids, requests, values, strict=True):
            tokens = list(value.token_ids)
            if len(tokens) > MAX_NEW_TOKENS:
                raise AssertionError("native output exceeds cap")
            rows.append(
                {
                    "image_id": image_id,
                    "request_id": request.request_id,
                    "token_ids": tokens,
                    "text": tokenizer.decode(
                        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    "stop": value.stop_reason,
                }
            )
        expected = _saved_rows(group.get("rows") if name == "O" else group.get("saved_norm_rows"))
        if expected:
            if set(expected) != set(image_ids):
                raise AssertionError("saved control rows do not match group images")
            for row in rows:
                prior = expected[row["image_id"]]
                if row["token_ids"] != _saved_tokens(prior) or row["stop"] != _saved_stop(prior):
                    raise AssertionError("saved policy control did not reproduce exactly")
                prior_text = prior.get("text", prior.get("raw_decode_text"))
                if prior_text is not None and row["text"] != prior_text:
                    raise AssertionError("saved policy control text did not reproduce exactly")
        if shadows is not None:
            for row, tokens in zip(rows, shadows.tokens, strict=True):
                if row["token_ids"] != tokens:
                    raise AssertionError("N shadow argmax IDs do not reproduce raw tokens")
            receipt["shadow"] = shadows.result()
        raw = {"group": group["key"], "policy": name, "empty_prefix": True, "rows": rows}
        _write(policy_dir / "raw.json", raw)
        torch.cuda.synchronize()
        receipt.update(
            status="candidate_complete",
            input_identity=input_identity,
            prefill_mrope_sha256=prefill_mrope,
            model_forwards=model_forwards,
            exact_saved_control=bool(expected),
            raw=_binding(policy_dir / "raw.json"),
            elapsed_seconds=time.monotonic() - start,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        _write(policy_dir / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - start)
        _write(policy_dir / "receipt.json", receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


def run(*, panel_path: Path, output: Path, groups: Sequence[str]) -> None:
    panel = _read(panel_path)
    _check_sources(panel)  # Bind all frozen sources once per shard before model load.
    panel_binding = _binding(panel_path)
    producer_binding = _binding(Path(__file__).resolve())
    configured_groups = panel.get("groups")
    if not isinstance(configured_groups, list):
        raise ValueError("panel groups must be a list")
    by_key = {group.get("key"): group for group in configured_groups if isinstance(group, Mapping)}
    if len(by_key) != len(configured_groups) or any(key not in by_key for key in groups):
        raise ValueError("requested groups are not unique panel groups")
    if len(set(groups)) != len(groups) or not groups:
        raise ValueError("--groups needs one or more unique comma-separated keys")
    config = panel.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("panel config is missing")
    infer = InferConfig.model_validate(config)
    if (
        infer.backend.type != "hf"
        or infer.model.dtype != "fp32"
        or infer.backend.hf.attn_implementation != "sdpa"
    ):
        raise AssertionError("panel must execute frozen FP32/SDPA native inference")
    qwen, loaded_identity = load_policy(infer, device=torch.device("cuda:0"))
    model = qwen.model
    model.eval()
    if loaded_identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] != ["torch.float32"]:
        raise AssertionError("loaded model is not FP32")
    if loaded_identity["effective_settings"]["observed_attn_implementation"] != "sdpa":
        raise AssertionError("loaded attention implementation is not SDPA")
    coordinate_ids, factors, readout = _load_coefficients(
        panel=panel, model=model, tokenizer=qwen.tokenizer, device=torch.device("cuda:0")
    )
    initial_versions = {key: value._version for key, value in model.named_parameters()}
    readout["loaded_model_identity"] = loaded_identity
    captured: set[int] = set()
    for key in groups:
        group = by_key[key]
        original = _run_policy(
            name="O", group=group, qwen=qwen, config=config,
            coordinate_ids=coordinate_ids, factors=factors, captured=captured,
            output=output, panel_binding=panel_binding, producer_binding=producer_binding,
            readout=readout, initial_versions=initial_versions,
        )
        treated = _run_policy(
            name="N", group=group, qwen=qwen, config=config,
            coordinate_ids=coordinate_ids, factors=factors, captured=captured,
            output=output, panel_binding=panel_binding, producer_binding=producer_binding,
            readout=readout, initial_versions=initial_versions,
        )
        if original["input_identity"] != treated["input_identity"]:
            raise AssertionError("O/N native input, token, media, or grid identity changed")
        if original["prefill_mrope_sha256"] != treated["prefill_mrope_sha256"]:
            raise AssertionError("O/N native prefill MRoPE identity changed")


def cpu_check(output: Path) -> None:
    """Falsify the two easy-to-miss seam properties without loading a model."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cpu-check.json"
    if path.exists():
        raise FileExistsError(path)
    scores = torch.tensor([[9.0, 3.0, 5.0, 1.0, 7.0, 8.0]], dtype=torch.float32)
    coordinates = torch.tensor([1, 4])
    factors = torch.tensor([2.0, 0.5], dtype=torch.float64)
    scaled = scores.clone()
    scaled[:, coordinates] = (scores[:, coordinates].double() * factors).to(scores.dtype)
    unchanged = torch.tensor([0, 2, 3, 5])
    assert torch.equal(scores[:, unchanged], scaled[:, unchanged])
    assert torch.equal(scores[:, 5], scaled[:, 5])
    assert torch.equal(scaled[:, coordinates], torch.tensor([[6.0, 3.5]]))
    # EOS is an active decision; a later padded processor call must not appear.
    ledger = _ShadowLedger(
        prompt_width=2, image_ids=[7], coordinate_ids={1, 4}, eos_token_id=5,
        capture_dir=output, captured=set(),
    )
    head = torch.zeros((1, 3))
    eos_before = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    eos_after = eos_before.clone()
    ledger.observe(input_ids=torch.tensor([[11, 12]]), before=eos_before, after=eos_after, head_input=head)
    ledger.observe(input_ids=torch.tensor([[11, 12, 5]]), before=eos_before, after=eos_after, head_input=head)
    assert ledger.tokens == [[5]] and len(ledger.trace[0]) == 1
    assert int(torch.argmax(torch.tensor([4.0, 4.0])).item()) == 0
    _write(path, {
        "status": "passed",
        "coordinate_scaling_and_noncoordinate_identity": True,
        "eos_retained_and_post_eos_padding_trimmed": True,
        "argmax_tie_policy": "first maximal index",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups")
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check(args.output)
        return
    if args.panel is None or args.groups is None:
        parser.error("--panel and --groups are required unless --cpu-check is used")
    run(panel_path=args.panel, output=args.output, groups=args.groups.split(","))


if __name__ == "__main__":
    main()
