"""Run one native complete-row owner-recurrence branch.

The panel owns the cohort, original native batch, row interval, and literal
replacement rows.  This module only replays that batch and records the
request-local logits seam.  An intervention changes one target row while the
other three requests and the target prefix/free suffix remain native greedy.
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


OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-17-owner-recurrence-row-branch"
)
EOS = 151645
MAX_NEW_TOKENS = 3084
ARMS = ("native", "same", "distinct", "covered")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _input_identity(batch: NativeBatch) -> dict[str, Any]:
    return {
        "request_ids": list(batch.request_ids),
        "prompt_token_ids": [list(row) for row in batch.prompt_token_ids],
        "media_sha256": None
        if batch.media_sha256 is None
        else list(batch.media_sha256),
        "image_grids": [
            None if grid is None else list(grid) for grid in batch.image_grids
        ],
        "tensor_sha256": {
            name: _tensor_hash(value)
            for name, value in sorted(batch.inputs.items())
            if isinstance(value, torch.Tensor)
        },
    }


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(f"{label} must be a list of integer token IDs")
    return list(value)


def _rows_from_raw(raw: Mapping[str, Any], label: str) -> dict[int, Mapping[str, Any]]:
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} has no rows list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a non-object row")
        image_id = row.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"{label} row has no integer image_id")
        if image_id in result:
            raise ValueError(f"{label} has duplicate image_id {image_id}")
        result[image_id] = row
    return result


def _row_tokens(row: Mapping[str, Any], label: str) -> list[int]:
    return _tokens(row.get("token_ids", row.get("generated_token_ids")), label)


def _row_stop(row: Mapping[str, Any], label: str) -> str:
    value = row.get("stop", row.get("decode_stop_reason"))
    if not isinstance(value, str):
        raise ValueError(f"{label} has no stop reason")
    return value


def _check_sources(panel: Mapping[str, Any]) -> None:
    sources = panel.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("panel sources must be a list")
    for expected in sources:
        if not isinstance(expected, Mapping) or not isinstance(
            expected.get("path"), str
        ):
            raise ValueError("panel source binding is malformed")
        path = Path(expected["path"])
        if _binding(path) != dict(expected):
            raise AssertionError(f"frozen source changed: {path}")


def _group(case: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("original_group", "group", "original"):
        value = case.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("cases"), list):
            return value
    # A caller may place the fresh128 group fields directly beside the case.
    if isinstance(case.get("cases"), list):
        return case
    raise ValueError("case must contain an original group with a cases list")


def _capture_offsets(
    panel: Mapping[str, Any], case: Mapping[str, Any], end_offset: int
) -> list[int]:
    value: object = case.get("capture_offsets")
    if value is None:
        value = case.get("diagnostic_offsets")
    if value is None:
        value = panel.get("capture_offsets", panel.get("diagnostic_offsets"))
    if isinstance(value, Mapping):
        later = value.get("later_offsets", value.get("later"))
        values: list[object] = []
        if value.get("row_end_plus_one") is not None:
            values.append(value["row_end_plus_one"])
        if isinstance(later, list):
            values.extend(later)
        value = values
    if not isinstance(value, list) or any(
        isinstance(offset, bool) or not isinstance(offset, int) for offset in value
    ):
        raise ValueError(
            "panel must declare capture_offsets (row end + 1 and two later offsets)"
        )
    offsets = list(dict.fromkeys(value))
    if end_offset not in offsets:
        raise ValueError("capture_offsets must include row end + 1")
    if len([offset for offset in offsets if offset > end_offset]) < 2:
        raise ValueError("capture_offsets must include two later row-related offsets")
    if any(offset < 0 or offset >= MAX_NEW_TOKENS for offset in offsets):
        raise ValueError("capture_offsets must be within the 3084-step budget")
    return offsets


def _literal_row(tokenizer: Any, ids: Sequence[int], expected_text: object, label: str) -> str:
    """Check the tokenizer's literal token mapping before model execution."""
    token_ids = _tokens(list(ids), label)
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    if not isinstance(pieces, Sequence) or len(pieces) != len(token_ids):
        raise AssertionError(f"{label} cannot be represented by tokenizer")
    roundtrip = tokenizer.convert_tokens_to_ids(pieces)
    if list(roundtrip) != token_ids:
        raise AssertionError(f"{label} token conversion is not literal")
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if expected_text is not None and (not isinstance(expected_text, str) or text != expected_text):
        raise AssertionError(f"{label} text does not match supplied token IDs")
    return text


def _config_gate(config: Mapping[str, Any]) -> InferConfig:
    infer = InferConfig.model_validate(config)
    if infer.backend.type != "hf":
        raise AssertionError("row branch requires the HF native backend")
    if infer.model.dtype != "fp32":
        raise AssertionError("row branch requires FP32")
    if infer.backend.hf.attn_implementation != "sdpa":
        raise AssertionError("row branch requires SDPA")
    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("panel generation config is missing")
    expected = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    for key, wanted in expected.items():
        if generation.get(key) != wanted:
            raise AssertionError(f"frozen generation {key} differs from {wanted}")
    return infer


def _prepare_case(
    panel: Mapping[str, Any], image_id: int, arm: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[int, Mapping[str, Any]], list[int], int, int, list[int], list[int], list[int]]:
    cases = panel.get("cases")
    if not isinstance(cases, list):
        raise ValueError("panel cases must be a list")
    selected = [
        item
        for item in cases
        if isinstance(item, Mapping) and item.get("image_id") == image_id
    ]
    if len(selected) != 1:
        raise ValueError(f"panel must contain exactly one case for image_id={image_id}")
    case = selected[0]
    group = _group(case)
    group_cases = group.get("cases")
    if not isinstance(group_cases, list) or len(group_cases) != 4:
        raise ValueError("original group must contain exactly four cases")
    ids = [_case_image_id(item) for item in group_cases]
    if image_id not in ids or len(set(ids)) != 4:
        raise ValueError("target image must occur once in the four-row native group")
    target_position = case.get("target_position", ids.index(image_id))
    if (
        isinstance(target_position, bool)
        or not isinstance(target_position, int)
        or target_position < 0
        or target_position >= 4
        or ids[target_position] != image_id
    ):
        raise ValueError("target_position does not identify image_id in the group")
    start = case.get("start_offset")
    end = case.get("end_offset")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
        raise ValueError("case row offsets must be integers")
    if start < 0 or end <= start or end > MAX_NEW_TOKENS:
        raise ValueError("case row offsets are outside the 3084-step budget")
    arms = case.get("arms")
    if not isinstance(arms, Mapping) or arm not in arms or not isinstance(arms[arm], Mapping):
        raise ValueError(f"case has no {arm} arm")
    supplied = _tokens(arms[arm].get("token_ids"), f"{arm} arm token_ids")
    if len(supplied) != end - start:
        raise ValueError("replacement row length differs from the declared interval")
    if EOS in supplied:
        raise ValueError("replacement complete row cannot contain EOS")
    saved_binding = case.get("saved_raw")
    if not isinstance(saved_binding, Mapping) or not isinstance(saved_binding.get("path"), str):
        raise ValueError("case needs a saved_raw binding")
    saved_path = Path(saved_binding["path"])
    if _binding(saved_path) != dict(saved_binding):
        raise AssertionError("saved O/raw.json binding changed")
    saved_receipt_binding = case.get("saved_receipt")
    if not isinstance(saved_receipt_binding, Mapping) or not isinstance(
        saved_receipt_binding.get("path"), str
    ):
        raise ValueError("case needs a saved_receipt binding")
    saved_receipt_path = Path(saved_receipt_binding["path"])
    if _binding(saved_receipt_path) != dict(saved_receipt_binding):
        raise AssertionError("saved native receipt binding changed")
    saved_receipt = _read(saved_receipt_path)
    if saved_receipt.get("status") != "candidate_complete":
        raise AssertionError("saved native receipt is not candidate_complete")
    saved = _rows_from_raw(_read(saved_path), "saved_raw")
    if set(saved) != set(ids):
        raise AssertionError("saved raw rows do not match the original four-row group")
    original = _row_tokens(saved[image_id], "saved target row")
    if end > len(original):
        raise ValueError("declared complete row extends beyond the saved original row")
    capture = _capture_offsets(panel, case, end)
    return case, group, saved, supplied, target_position, start, end, capture, ids


def _case_image_id(case: object) -> int:
    if not isinstance(case, Mapping):
        raise ValueError("group case must be an object")
    value = case.get("image_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("group case needs an integer image_id")
    return value


def _saved_text(row: Mapping[str, Any]) -> str | None:
    value = row.get("text", row.get("raw_decode_text"))
    return value if isinstance(value, str) else None


def _top_margin(scores: torch.Tensor) -> tuple[int, float]:
    top = torch.topk(scores, 2)
    return int(top.indices[0].item()), float((top.values[0] - top.values[1]).item())


def _run(
    *, panel_path: Path, image_id: int, arm: str, output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    panel = _read(panel_path)
    _check_sources(panel)
    config = panel.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("panel config is missing")
    infer = _config_gate(config)
    (
        case,
        group,
        saved,
        supplied,
        target_position,
        start,
        end,
        capture_offsets,
        image_ids,
    ) = _prepare_case(panel, image_id, arm)
    native_receipt: dict[str, Any] | None = None
    if arm != "native":
        native_dir = output_root / str(image_id) / "native"
        native_receipt_path = native_dir / "receipt.json"
        if not native_receipt_path.exists():
            raise AssertionError("intervention requires a successful native receipt")
        native_receipt = _read(native_receipt_path)
        if native_receipt.get("status") != "candidate_complete":
            raise AssertionError("intervention native gate is not candidate_complete")
    saved_receipt_binding = case["saved_receipt"]
    saved_receipt = _read(Path(saved_receipt_binding["path"]))
    run_dir = output_root / str(image_id) / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "owner_recurrence_row_branch.native.v1",
        "status": "running",
        "image_id": image_id,
        "arm": arm,
        "panel": _binding(panel_path),
        "producer": _binding(Path(__file__).resolve()),
        "group_key": group.get("key"),
        "image_ids": image_ids,
        "target_position": target_position,
        "start_offset": start,
        "end_offset": end,
        "capture_offsets": capture_offsets,
        "counts": {"model_forwards": 0, "vision_forwards": 0, "processor_calls": 0},
        "raw_path": str(run_dir / "raw.json"),
    }
    _write(run_dir / "receipt.json", receipt)
    began = time.monotonic()
    handles: list[Any] = []
    model_forwards = 0
    vision_forwards = 0
    processor_calls = 0
    head_input: torch.Tensor | None = None
    sparse_logits: dict[int, torch.Tensor] = {}
    sparse_head: dict[int, torch.Tensor] = {}
    row_slots: list[dict[str, Any]] = []
    histories: list[list[int]] = [[] for _ in image_ids]
    done = [False for _ in image_ids]
    try:
        request_data = dict(config["data"])
        input_jsonl = group.get("input_jsonl", request_data.get("input_jsonl"))
        if not isinstance(input_jsonl, str):
            raise ValueError("original group needs input_jsonl")
        request_data["input_jsonl"] = input_jsonl
        request_config = dict(config)
        request_config["data"] = request_data
        group_cases = group["cases"]
        qwen, loaded_identity = load_policy(infer, device=torch.device("cuda:0"))
        model = qwen.model
        tokenizer = qwen.tokenizer
        model.eval()
        if loaded_identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] != ["torch.float32"]:
            raise AssertionError("loaded model parameters are not FP32")
        if loaded_identity["effective_settings"]["observed_attn_implementation"] != "sdpa":
            raise AssertionError("loaded model attention is not SDPA")
        for arm_name, arm_value in case["arms"].items():
            if isinstance(arm_value, Mapping) and "token_ids" in arm_value:
                ids = _tokens(arm_value["token_ids"], f"{arm_name} arm token_ids")
                text = arm_value.get("text")
                _literal_row(tokenizer, ids, text, f"{arm_name} arm")
        requests, _ = build_bound_native_requests(qwen, request_config, group_cases)
        batch = prepare_native_inputs(
            qwen.processor, requests, device="cuda:0", record_media_identity=True
        )
        input_identity = _input_identity(batch)
        expected_input_identity = saved_receipt.get("input_identity")
        if not isinstance(expected_input_identity, Mapping):
            raise AssertionError("saved native receipt has no input identity")
        if input_identity != dict(expected_input_identity):
            raise AssertionError("native input identity differs from predecessor receipt")
        width = int(batch.inputs["input_ids"].shape[1])
        if len(requests) != 4 or width <= 0:
            raise AssertionError("native batch shape is not the frozen four-row batch")
        receipt.update(
            loaded_identity=loaded_identity,
            input_jsonl=input_jsonl,
            input_identity=input_identity,
            batch_shape={
                "size": 4,
                "padded_prompt_width": width,
                "image_ids": image_ids,
                "left_padding":[width - len(row) for row in batch.prompt_token_ids],
                "pad_token_id": tokenizer.pad_token_id,
            },
        )
        original_rows = {image: _row_tokens(saved[image], f"saved row {image}") for image in image_ids}
        target_original = original_rows[image_id]
        initial_versions = {key: value._version for key, value in model.named_parameters()}
        prefill_mrope: str | None = None

        def model_counter(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal model_forwards
            model_forwards += 1
            receipt["counts"]["model_forwards"] = model_forwards
            if model_forwards > MAX_NEW_TOKENS:
                raise AssertionError("generation exceeded frozen 3084 forward cap")

        def vision_counter(*_: Any) -> None:
            nonlocal vision_forwards
            vision_forwards += 1
            receipt["counts"]["vision_forwards"] = vision_forwards

        def mrope_capture(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal prefill_mrope
            if model_forwards == 1:
                positions = kwargs.get("position_ids")
                if not isinstance(positions, torch.Tensor):
                    raise AssertionError("native prefill did not pass position IDs")
                prefill_mrope = _tensor_hash(positions)

        def head_capture(module: Any, args: tuple[Any, ...]) -> None:
            nonlocal head_input
            if not args or not isinstance(args[0], torch.Tensor):
                raise AssertionError("lm_head hook did not receive a tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[0] != 4:
                raise AssertionError("lm_head input shape differs from native batch")
            head_input = value[:, -1, :].detach().clone()

        handles.extend(
            [
                model.register_forward_pre_hook(model_counter, with_kwargs=True),
                model.model.visual.register_forward_pre_hook(vision_counter),
                model.model.language_model.register_forward_pre_hook(
                    mrope_capture, with_kwargs=True
                ),
                model.get_output_embeddings().register_forward_pre_hook(head_capture),
            ]
        )

        fixed_capture = set(capture_offsets)
        first_difference: int | None = None

        class RowBranchProcessor(LogitsProcessor):
            def __call__(self, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal processor_calls, first_difference
                processor_calls += 1
                receipt["counts"]["processor_calls"] = processor_calls
                offset = int(ids.shape[1] - width)
                if offset < 0 or offset >= MAX_NEW_TOKENS:
                    raise AssertionError("native action offset is outside the frozen budget")
                if head_input is None:
                    raise AssertionError("missing final head input at native logits seam")
                edited_scores = scores
                target_edited = False
                for index, image in enumerate(image_ids):
                    if done[index]:
                        continue
                    history = ids[index, width:].tolist()
                    if history != histories[index]:
                        raise AssertionError(f"row {image} history is not its literal generated prefix")
                    before = scores[index]
                    winner, margin = _top_margin(before)
                    expected = original_rows[image]
                    target_native_prefix = index == target_position and (
                        arm == "native" or offset < start
                    )
                    if offset < len(expected) and (index != target_position or target_native_prefix) and winner != expected[offset]:
                        raise AssertionError(
                            f"row {image} native winner forked at offset {offset}: "
                            f"expected {expected[offset]}, got {winner}"
                        )
                    if index != target_position and offset < len(expected):
                        # A companion is never eligible for a processor edit.
                        if histories[index] != expected[:offset]:
                            raise AssertionError(f"companion {image} history forked")
                    if index == target_position and (arm == "native" or offset < end):
                        expected_history = expected[:offset]
                        if arm != "native" and offset >= start:
                            expected_history = expected[:start] + supplied[: offset - start]
                        if histories[index] != expected_history:
                            raise AssertionError("target history forked around row interval")
                    if index == target_position and offset < start:
                        if histories[index] != expected[:offset]:
                            raise AssertionError("target prehistory forked before row interval")
                    supplied_token: int | None = None
                    changed = False
                    if index == target_position and arm != "native" and start <= offset < end:
                        supplied_token = supplied[offset - start]
                        if not target_edited:
                            edited_scores = scores.clone()
                            target_edited = True
                        edited_scores[index, :] = -torch.inf
                        edited_scores[index, supplied_token] = 0
                        changed = True
                    transformed = edited_scores[index]
                    if not changed and not torch.equal(before, transformed):
                        raise AssertionError("baseline/free/companion logits changed")
                    if changed:
                        changed_rows = torch.zeros((scores.shape[0],), dtype=torch.bool, device=scores.device)
                        changed_rows[index] = True
                        if index != target_position or not torch.equal(
                            scores[~changed_rows], edited_scores[~changed_rows]
                        ):
                            raise AssertionError("only target row may be edited")
                    if index == target_position and start <= offset < end:
                        assert supplied_token is not None
                        lp = float(torch.log_softmax(before.float(), dim=-1)[supplied_token].item())
                        row_slots.append(
                            {
                                "action_offset": offset,
                                "supplied_token_id": supplied_token,
                                "native_winner_token_id": winner,
                                "native_winner_is_supplied": winner == supplied_token,
                                "native_full_vocab_margin": margin,
                                "native_supplied_token_logprob": lp,
                            }
                        )
                        if supplied_token != original_rows[image][offset]:
                            if first_difference is None:
                                first_difference = offset
                            if offset == first_difference:
                                sparse_logits[offset] = before.detach().cpu().clone()
                                sparse_head[offset] = head_input[index].detach().cpu().clone()
                    capture_here = offset in fixed_capture
                    if index == target_position and capture_here:
                        sparse_logits[offset] = before.detach().cpu().clone()
                        sparse_head[offset] = head_input[index].detach().cpu().clone()
                    chosen = int(torch.argmax(transformed).item())
                    if index == target_position and arm != "native" and start <= offset < end:
                        if chosen != supplied_token:
                            raise AssertionError("forced row slot did not select supplied token")
                    histories[index].append(chosen)
                    if chosen == EOS:
                        done[index] = True
                return edited_scores

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
                    "max_new_tokens", "do_sample", "repetition_penalty",
                    "eos_token_id", "pad_token_id", "use_model_defaults",
                    "return_dict_in_generate", "output_scores", "output_logits",
                )
            }
            return original_generate(
                **kwargs, logits_processor=LogitsProcessorList([RowBranchProcessor()])
            )

        model.generate = generate_wrapper
        try:
            values = generate_continuations(
                model,
                batch,
                extensions=[[] for _ in requests],
                budgets=[MAX_NEW_TOKENS for _ in requests],
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
        if len(values) != 4:
            raise AssertionError("generated batch cardinality changed")
        rows: list[dict[str, Any]] = []
        for index, (image, request, result) in enumerate(zip(image_ids, requests, values, strict=True)):
            ids = list(result.token_ids)
            row = {
                "image_id": image,
                "request_id": request.request_id,
                "token_ids": ids,
                "tokens_sha256": _digest(ids),
                "stop": result.stop_reason,
                "text": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            }
            expected = original_rows[image]
            if index != target_position or arm == "native":
                if ids != expected or _row_stop(saved[image], f"saved row {image}") != result.stop_reason:
                    raise AssertionError(f"saved native row mismatch for image {image}")
                prior_text = _saved_text(saved[image])
                if prior_text is not None and prior_text != row["text"]:
                    raise AssertionError(f"saved native text mismatch for image {image}")
            rows.append(row)
        target = rows[target_position]
        if target["token_ids"][:start] != target_original[:start]:
            raise AssertionError("target prehistory changed")
        if arm == "native":
            if target["token_ids"] != target_original:
                raise AssertionError("native target row is not exact")
        else:
            if target["token_ids"][start : start + len(supplied)] != supplied:
                raise AssertionError("target replacement row is not exact")
            if len(target["token_ids"]) > MAX_NEW_TOKENS:
                raise AssertionError("target output exceeds frozen budget")
        if len(row_slots) != end - start:
            raise AssertionError("complete-row native logprob capture is incomplete")
        values_lp = [slot["native_supplied_token_logprob"] for slot in row_slots]
        missing = sorted(fixed_capture.difference(sparse_logits))
        if missing:
            raise AssertionError(f"declared sparse capture offsets were not reached: {missing}")
        if arm != "native" and first_difference is None:
            raise AssertionError("intervention supplied row has no differing slot")
        if first_difference is not None and first_difference not in sparse_logits:
            raise AssertionError("first differing supplied slot was not captured")
        if head_input is None:
            raise AssertionError("missing final head input")
        torch.save(
            {
                "offsets": sorted(sparse_logits),
                "logits": {offset: sparse_logits[offset] for offset in sorted(sparse_logits)},
                "lm_head_input": {offset: sparse_head[offset] for offset in sorted(sparse_head)},
            },
            run_dir / "sparse-logits.pt",
        )
        current_versions = {key: value._version for key, value in model.named_parameters()}
        if current_versions != initial_versions:
            raise AssertionError("generation mutated model parameters")
        if _input_identity(batch) != input_identity:
            raise AssertionError("generation mutated native inputs")
        expected_prefill = saved_receipt.get("prefill_mrope_sha256")
        if not isinstance(expected_prefill, str) or prefill_mrope != expected_prefill:
            raise AssertionError("native prefill MRoPE differs from predecessor receipt")
        if arm != "native":
            assert native_receipt is not None
            if native_receipt.get("input_identity") != input_identity:
                raise AssertionError("intervention input identity differs from native arm")
            if native_receipt.get("prefill_mrope_sha256") != prefill_mrope:
                raise AssertionError("intervention prefill MRoPE differs from native arm")
        raw = {
            "schema": "owner_recurrence_row_branch.raw.v1",
            "image_id": image_id,
            "arm": arm,
            "group_key": group.get("key"),
            "target_position": target_position,
            "start_offset": start,
            "end_offset": end,
            "original_target_token_ids": target_original,
            "supplied_row_token_ids": supplied,
            "supplied_row_text": _literal_row(tokenizer, supplied, None, "supplied row"),
            "rows": rows,
            "input_identity": input_identity,
            "complete_row_native_logprobs": {
                "slots": row_slots,
                "sum": float(sum(values_lp)),
                "mean": float(sum(values_lp) / len(values_lp)),
            },
            "sparse_logits": _binding(run_dir / "sparse-logits.pt"),
            "policy": "native greedy from empty prefix; target complete row forced only for intervention interval",
        }
        _write(run_dir / "raw.json", raw)
        receipt.update(
            status="candidate_complete",
            model_forwards=model_forwards,
            vision_forwards=vision_forwards,
            processor_calls=processor_calls,
            loaded_identity=loaded_identity,
            prefill_mrope_sha256=prefill_mrope,
            raw=_binding(run_dir / "raw.json"),
            sparse_logits=_binding(run_dir / "sparse-logits.pt"),
            complete_row_native_logprob_sum=float(sum(values_lp)),
            complete_row_native_logprob_mean=float(sum(values_lp) / len(values_lp)),
            elapsed_seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        _write(run_dir / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(
            status="technical_invalid",
            error=repr(exc),
            elapsed_seconds=time.monotonic() - began,
        )
        _write(run_dir / "receipt.json", receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


def cpu_check(output: Path) -> None:
    """Exercise the processor's row isolation and half-open replacement rules."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cpu-check.json"
    if path.exists():
        raise FileExistsError(path)
    scores = torch.tensor(
        [[8.0, 1.0, 7.0, 2.0], [6.0, 5.0, 4.0, 3.0]], dtype=torch.float32
    )
    target = scores[1].clone()
    target[:] = -torch.inf
    target[2] = 0
    assert torch.equal(scores[0], scores[0])
    assert int(torch.argmax(target)) == 2
    assert torch.equal(scores[0], torch.tensor([8.0, 1.0, 7.0, 2.0]))
    assert len([4, 5, 6][1:]) == 2
    _write(
        path,
        {
            "status": "passed",
            "target_only_score_edit": True,
            "companion_score_identity": True,
            "half_open_row_interval": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check(args.output_root)
        return
    if args.panel is None or args.case is None or args.arm is None:
        parser.error("--panel, --case and --arm are required unless --cpu-check is used")
    _run(panel_path=args.panel, image_id=args.case, arm=args.arm, output_root=args.output_root)


if __name__ == "__main__":
    main()
