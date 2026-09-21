"""Small native runtime for fixed-prefix history and timed readout pulses.

Stage A is delegated to the frozen producer-v2 artifact.  Stage B keeps the
same native caller but adds one target-prefix force and a bounded coordinate
norm window.  The processor is deliberately local so the model, KV cache,
vision inputs, and decoder remain native.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probes.training_set_completion import readout_norm_fresh as fresh
from probes.training_set_completion import row_branch


EOS = fresh.EOS
MAX_NEW_TOKENS = fresh.MAX_NEW_TOKENS
BOX_END = 151649
PRODUCER_V2 = Path(row_branch.__file__)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _image_id(case: Mapping[str, Any]) -> int:
    value = case.get("image_id")
    if value is None and isinstance(case.get("input_record"), Mapping):
        value = case["input_record"].get("image_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("every native case needs an integer image_id")
    return value


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(f"{label} must be a list of integer token IDs")
    return list(value)


def _group(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    value = cell.get("group")
    if isinstance(value, Mapping) and isinstance(value.get("cases"), list):
        return value
    if isinstance(cell.get("cases"), list):
        return cell
    raise ValueError("cell must contain a native group with cases")


def _cell(panel: Mapping[str, Any], image_id: int) -> Mapping[str, Any]:
    cells = panel.get("cells", panel.get("cases"))
    if not isinstance(cells, list):
        raise ValueError("panel must contain cells or cases")
    found = [item for item in cells if isinstance(item, Mapping) and _image_id(item) == image_id]
    if len(found) != 1:
        raise ValueError(f"panel must contain exactly one cell for image_id={image_id}")
    return found[0]


def run_fixed_prefix(
    *, panel_path: Path, image_id: int, arm: str, output_root: Path
) -> dict[str, Any]:
    """Run the frozen complete-row force/free producer without copying it."""
    return row_branch._run(
        panel_path=panel_path, image_id=image_id, arm=arm, output_root=output_root
    )


def _saved_tokens(row: Mapping[str, Any]) -> list[int]:
    return _tokens(row.get("token_ids", row.get("generated_token_ids")), "saved row")


def _saved_stop(row: Mapping[str, Any]) -> str:
    value = row.get("stop", row.get("decode_stop_reason"))
    if not isinstance(value, str):
        raise ValueError("saved row has no stop reason")
    return value


def _rows(value: object) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("saved rows must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("saved row must be an object")
        image = _image_id(row)
        if image in result:
            raise ValueError(f"duplicate saved image_id={image}")
        result[image] = row
    return result


def _prefix(cell: Mapping[str, Any]) -> tuple[int, list[int]]:
    values = cell.get("target_prefix_token_ids")
    if values is None:
        values = cell.get("prefix_token_ids")
    tokens = [] if values is None else _tokens(values, "target_prefix_token_ids")
    start = cell.get("prefix_start_offset", 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("prefix_start_offset must be a nonnegative integer")
    if start + len(tokens) > MAX_NEW_TOKENS:
        raise ValueError("target prefix is outside the 3084-step budget")
    if EOS in tokens:
        raise ValueError("target prefix cannot contain EOS")
    return start, tokens


def _pulse(cell: Mapping[str, Any], mode: str) -> tuple[int, int | None]:
    start = cell.get("norm_start")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0 or start >= MAX_NEW_TOKENS:
        raise ValueError("norm_start must be within the 3084-step budget")
    if mode == "pulse1":
        limit: object = 1
    elif mode == "pulse4":
        limit = 4
    elif mode == "sustained":
        limit = "sustained"
    else:
        raise ValueError("mode must be prefix, pulse1, pulse4, or sustained")
    declared = cell.get("row_limit")
    if declared is not None and declared != limit:
        raise ValueError(f"cell row_limit {declared!r} does not match mode {mode}")
    return start, None if limit == "sustained" else int(limit)


def _force(scores: torch.Tensor, row: int, token: int) -> torch.Tensor:
    if scores.ndim != 2 or not 0 <= row < scores.shape[0]:
        raise ValueError("target row is outside score batch")
    if not 0 <= token < scores.shape[1]:
        raise ValueError("forced token is outside vocabulary")
    transformed = scores.clone()
    transformed[row].fill_(-torch.inf)
    transformed[row, token] = 0
    untouched = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    untouched[row] = False
    if not torch.equal(scores[untouched], transformed[untouched]):
        raise AssertionError("prefix force edited a native companion")
    return transformed


def _norm(
    scores: torch.Tensor,
    coordinate_ids: torch.Tensor,
    factors: torch.Tensor,
    target_position: int,
) -> torch.Tensor:
    if scores.ndim != 2 or not 0 <= target_position < scores.shape[0]:
        raise ValueError("target row is outside score batch")
    transformed = scores.clone()
    transformed[target_position, coordinate_ids] = (
        scores[target_position, coordinate_ids].double() * factors
    ).to(scores.dtype)
    changed = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    changed[coordinate_ids] = True
    if not torch.equal(scores[:, ~changed], transformed[:, ~changed]):
        raise AssertionError("norm changed a non-coordinate logit")
    companions = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    companions[target_position] = False
    if not torch.equal(scores[companions], transformed[companions]):
        raise AssertionError("norm changed a native companion")
    if EOS < scores.shape[1] and not torch.equal(scores[:, EOS], transformed[:, EOS]):
        raise AssertionError("norm changed EOS")
    return transformed


def _pulse_active(
    *, norm_started: bool, target_done: bool, completed_rows: int, row_limit: int | None
) -> bool:
    return norm_started and not target_done and (
        row_limit is None or completed_rows < row_limit
    )


def _capture_offsets(cell: Mapping[str, Any], prefix_end: int) -> set[int]:
    value = cell.get("capture_offsets")
    if value is None:
        return set(range(prefix_end, min(prefix_end + 5, MAX_NEW_TOKENS)))
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("capture_offsets must be integer offsets")
    offsets = set(value)
    if any(item < 0 or item >= MAX_NEW_TOKENS for item in offsets):
        raise ValueError("capture_offsets are outside the 3084-step budget")
    return offsets


def _saved_controls(
    cell: Mapping[str, Any], group: Mapping[str, Any], image_ids: Sequence[int]
) -> tuple[dict[int, Mapping[str, Any]] | None, dict[str, Any] | None]:
    binding = cell.get("saved_raw")
    receipt_binding = cell.get("saved_receipt")
    if binding is None and receipt_binding is None:
        return None, None
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise ValueError("saved_raw binding is malformed")
    raw_path = Path(binding["path"])
    if _binding(raw_path) != dict(binding):
        raise AssertionError("saved raw binding changed")
    if not isinstance(receipt_binding, Mapping) or not isinstance(receipt_binding.get("path"), str):
        raise ValueError("saved_receipt binding is malformed")
    receipt_path = Path(receipt_binding["path"])
    if _binding(receipt_path) != dict(receipt_binding):
        raise AssertionError("saved receipt binding changed")
    receipt = _read(receipt_path)
    if receipt.get("status") != "candidate_complete":
        raise AssertionError("saved native receipt is not candidate_complete")
    saved = _rows(_read(raw_path).get("rows"))
    if set(saved) != set(image_ids):
        raise AssertionError("saved raw rows do not match native group")
    return saved, receipt


def _expected_target_raw(cell: Mapping[str, Any], target: int) -> Mapping[str, Any] | None:
    binding = cell.get("expected_target_raw")
    if binding is None:
        return None
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise ValueError("expected_target_raw binding is malformed")
    path = Path(binding["path"])
    if _binding(path) != dict(binding):
        raise AssertionError("expected target raw binding changed")
    rows = _rows(_read(path).get("rows"))
    if target not in rows:
        raise AssertionError("expected target raw has no target row")
    return rows[target]


def _run_norm_cell(
    *,
    panel_path: Path,
    panel: Mapping[str, Any],
    cell: Mapping[str, Any],
    mode: str,
    output: Path,
    qwen: Any,
    infer: Any,
    loaded_identity: Mapping[str, Any],
    coordinate_ids: torch.Tensor,
    factors: torch.Tensor,
    readout: Mapping[str, Any],
) -> dict[str, Any]:
    group = _group(cell)
    cases = group.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("norm cell group must contain exactly four native cases")
    image_ids = [_image_id(case) for case in cases]
    if len(set(image_ids)) != 4:
        raise ValueError("norm cell has duplicate image IDs")
    target = _image_id(cell)
    if target not in image_ids:
        raise ValueError("cell target is not in its native group")
    target_position = int(cell.get("target_position", image_ids.index(target)))
    if not 0 <= target_position < 4 or image_ids[target_position] != target:
        raise ValueError("target_position does not identify the target")
    prefix_start, prefix_tokens = _prefix(cell)
    common_prefix = cell.get("common_native_prefix_length", prefix_start)
    if (
        isinstance(common_prefix, bool)
        or not isinstance(common_prefix, int)
        or common_prefix < 0
        or common_prefix > MAX_NEW_TOKENS
    ):
        raise ValueError("common_native_prefix_length must be within the action horizon")
    saved, saved_receipt = _saved_controls(cell, group, image_ids)
    if common_prefix and saved is None:
        raise ValueError("common_native_prefix_length requires saved native rows")
    expected_target_raw = _expected_target_raw(cell, target)
    capture_offsets = _capture_offsets(cell, prefix_start + len(prefix_tokens))
    norm_start, row_limit = _pulse(cell, mode) if mode != "prefix" else (MAX_NEW_TOKENS, None)
    if mode != "prefix":
        capture_offsets.add(norm_start)
    run_dir = output / str(target) / mode
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "repetition_history_runtime.norm_pulse.v1",
        "status": "running",
        "mode": mode,
        "image_id": target,
        "image_ids": image_ids,
        "target_position": target_position,
        "prefix_start_offset": prefix_start,
        "prefix_token_ids": prefix_tokens,
        "common_native_prefix_length": common_prefix,
        "capture_offsets": sorted(capture_offsets),
        "norm_start": None if mode == "prefix" else norm_start,
        "row_limit": row_limit if mode != "prefix" else 0,
        "panel": _binding(panel_path),
        "sources": list(panel.get("sources", [])),
        "producer": _binding(Path(__file__).resolve()),
        "pid": os.getpid(),
        "loaded_model_identity": dict(loaded_identity),
        "saved_raw": None if cell.get("saved_raw") is None else dict(cell["saved_raw"]),
        "saved_receipt": None if cell.get("saved_receipt") is None else dict(cell["saved_receipt"]),
        "expected_target_raw": None
        if cell.get("expected_target_raw") is None
        else dict(cell["expected_target_raw"]),
        "readout": dict(readout),
        "raw_path": str(run_dir / "raw.json"),
    }
    _write(run_dir / "receipt.json", receipt)
    model = qwen.model
    tokenizer = qwen.tokenizer
    handles: list[Any] = []
    began = time.monotonic()
    model_forwards = 0
    prefill_mrope: str | None = None
    head_input: torch.Tensor | None = None
    histories: list[list[int]] = [[] for _ in image_ids]
    done = [False] * 4
    forced_replay: list[dict[str, Any]] = []
    pulse_steps: list[dict[str, Any]] = []
    sparse: dict[int, dict[str, torch.Tensor]] = {}
    complete_rows = 0
    pulse_seen = False
    pulse_last_offset: int | None = None
    prefix_end = prefix_start + len(prefix_tokens)
    free_capture_start: int | None = prefix_end if mode == "prefix" else None
    free_capture_until = prefix_end + 4
    try:
        config = panel.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("panel config is missing")
        request_config = dict(config)
        request_data = dict(config["data"])
        input_jsonl = group.get("input_jsonl", request_data.get("input_jsonl"))
        if not isinstance(input_jsonl, str):
            raise ValueError("native group needs input_jsonl")
        request_data["input_jsonl"] = input_jsonl
        request_config["data"] = request_data
        requests, _ = fresh.build_bound_native_requests(qwen, request_config, cases)
        batch = fresh.prepare_native_inputs(
            qwen.processor, requests, device="cuda:0", record_media_identity=True
        )
        input_identity = fresh._input_identity(batch)
        if saved_receipt is not None:
            expected_identity = saved_receipt.get("input_identity")
            if not isinstance(expected_identity, Mapping):
                raise AssertionError("saved native receipt has no input identity")
            if input_identity != dict(expected_identity):
                raise AssertionError("native input identity differs from saved receipt")
        native_identity = cell.get("native_identity", False)
        if not isinstance(native_identity, bool):
            raise ValueError("native_identity must be boolean")
        if native_identity and saved is None:
            raise ValueError("native_identity requires saved_raw and saved_receipt")
        width = int(batch.inputs["input_ids"].shape[1])
        if width <= 0:
            raise AssertionError("prepared native batch has no prompt width")
        initial_versions = {key: value._version for key, value in model.named_parameters()}

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
            nonlocal head_input
            if not args or not isinstance(args[0], torch.Tensor):
                raise AssertionError("lm_head hook did not receive a tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[0] != 4:
                raise AssertionError("lm_head input shape differs from native batch")
            head_input = value[:, -1, :].detach().clone()

        handles.extend([
            model.register_forward_pre_hook(model_counter, with_kwargs=True),
            model.model.language_model.register_forward_pre_hook(mrope_capture, with_kwargs=True),
            model.get_output_embeddings().register_forward_pre_hook(head_capture),
        ])

        class Processor(LogitsProcessor):
            def __call__(self, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal complete_rows, pulse_seen, pulse_last_offset, free_capture_start, free_capture_until
                if head_input is None:
                    raise AssertionError("missing final lm_head input")
                offset = int(ids.shape[1] - width)
                if offset < 0 or offset >= MAX_NEW_TOKENS:
                    raise AssertionError("action offset is outside the frozen budget")
                transformed = scores
                target_done = done[target_position]
                active = _pulse_active(
                    norm_started=mode != "prefix" and offset >= norm_start,
                    target_done=target_done,
                    completed_rows=complete_rows,
                    row_limit=row_limit,
                )
                if mode != "prefix" and pulse_last_offset == offset - 1 and not active:
                    free_capture_start = offset
                    free_capture_until = offset + 4
                if active:
                    pulse_seen = True
                    pulse_last_offset = offset
                    transformed = _norm(
                        transformed, coordinate_ids, factors, target_position
                    )
                if not target_done and ids[target_position, width:].tolist() != histories[target_position]:
                    raise AssertionError("target history is not its literal generated prefix")
                forced = prefix_start <= offset < prefix_start + len(prefix_tokens)
                if forced:
                    token = prefix_tokens[offset - prefix_start]
                    transformed = _force(transformed, target_position, token)
                for index in range(4):
                    if done[index]:
                        continue
                    history = ids[index, width:].tolist()
                    if history != histories[index]:
                        raise AssertionError(f"native history forked for image {image_ids[index]}")
                    native = int(torch.argmax(scores[index]).item())
                    chosen = int(torch.argmax(transformed[index]).item())
                    if (
                        index == target_position
                        and saved is not None
                        and offset < common_prefix
                    ):
                        expected = _saved_tokens(saved[target])
                        if offset >= len(expected) or native != expected[offset]:
                            raise AssertionError("natural target prefix argmax differs from saved native row")
                    if index == target_position and forced:
                        expected = prefix_tokens[offset - prefix_start]
                        if chosen != expected:
                            raise AssertionError("forced token was not replayed exactly")
                        forced_replay.append({
                            "offset": offset,
                            "token_id": expected,
                            "native_winner_token_id": native,
                            "history_sha256": _digest(history),
                        })
                    if index == target_position and active:
                        pulse_steps.append({
                            "offset": offset,
                            "native_winner_token_id": native,
                            "transformed_winner_token_id": chosen,
                            "winner_changed": native != chosen,
                            "complete_rows_before": complete_rows,
                        })
                        if offset == norm_start or (
                            row_limit is not None
                            and complete_rows + 1 == row_limit
                            and chosen == BOX_END
                        ) or (
                            row_limit is None
                            and (chosen == EOS or offset == MAX_NEW_TOKENS - 1)
                        ) or (
                            row_limit is not None and chosen == EOS
                        ):
                            sparse[offset] = {
                                "before_logits": scores[index].detach().cpu().clone(),
                                "after_logits": transformed[index].detach().cpu().clone(),
                                "lm_head_input": head_input[index].detach().cpu().clone(),
                            }
                    if index == target_position and (
                        offset in capture_offsets
                        or (
                            free_capture_start is not None
                            and free_capture_start <= offset <= free_capture_until
                        )
                    ):
                        sparse[offset] = {
                            "before_logits": scores[index].detach().cpu().clone(),
                            "after_logits": transformed[index].detach().cpu().clone(),
                            "lm_head_input": head_input[index].detach().cpu().clone(),
                        }
                    histories[index].append(chosen)
                    if index == target_position and active and chosen == BOX_END:
                        complete_rows += 1
                    if chosen == EOS:
                        done[index] = True
                return transformed

        original_generate = model.generate

        def generate_wrapper(**kwargs: Any) -> Any:
            if (
                kwargs.get("max_new_tokens") != MAX_NEW_TOKENS
                or kwargs.get("repetition_penalty") != 1
                or kwargs.get("do_sample")
                or "logits_processor" in kwargs
            ):
                raise AssertionError("native greedy generation settings changed")
            return original_generate(
                **kwargs, logits_processor=LogitsProcessorList([Processor()])
            )

        model.generate = generate_wrapper
        try:
            with torch.no_grad():
                values = fresh.generate_continuations(
                    model,
                    batch,
                    extensions=[[] for _ in cases],
                    budgets=[MAX_NEW_TOKENS for _ in cases],
                    eos_token_id=EOS,
                    pad_token_id=tokenizer.pad_token_id,
                    policy=fresh.NativeGenerationPolicy(
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
        if len(values) != 4 or prefill_mrope is None:
            raise AssertionError("native generation identity is incomplete")
        current_versions = {key: value._version for key, value in model.named_parameters()}
        if current_versions != initial_versions:
            raise AssertionError("generation mutated model parameters")
        if fresh._input_identity(batch) != input_identity:
            raise AssertionError("generation mutated native inputs")
        head = model.get_output_embeddings()
        embedding = model.get_input_embeddings()
        if (
            _tensor_hash(embedding(coordinate_ids)) != readout["input_coordinate_sha256"]
            or _tensor_hash(head.base.weight[coordinate_ids]) != readout["base_coordinate_sha256"]
            or _tensor_hash(head.shared_embed_delta) != readout["shared_delta_sha256"]
        ):
            raise AssertionError("generation changed effective input/readout parameters")
        rows = []
        for image, request, value in zip(image_ids, requests, values, strict=True):
            token_ids = list(value.token_ids)
            rows.append({
                "image_id": image,
                "request_id": request.request_id,
                "token_ids": token_ids,
                "tokens_sha256": _digest(token_ids),
                "stop": value.stop_reason,
                "text": tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })
        if mode != "prefix" and not pulse_seen:
            raise AssertionError("norm pulse did not trigger at norm_start")
        if saved is not None:
            require_native_target = native_identity and (
                mode == "prefix" or expected_target_raw is None
            )
            for row in rows:
                image = row["image_id"]
                if image == target and not require_native_target:
                    continue
                prior = saved[image]
                if row["token_ids"] != _saved_tokens(prior) or row["stop"] != _saved_stop(prior):
                    raise AssertionError(f"saved native row changed for image {image}")
        if expected_target_raw is not None:
            target_row = rows[target_position]
            if (
                target_row["token_ids"] != _saved_tokens(expected_target_raw)
                or target_row["stop"] != _saved_stop(expected_target_raw)
            ):
                raise AssertionError("target output differs from expected raw control")
        for row in rows:
            if row["image_id"] == target:
                actual = row["token_ids"]
                if actual[prefix_start:prefix_end] != prefix_tokens:
                    raise AssertionError("target prefix was not retained in full output")
        capture_path = run_dir / "pulse-captures.pt"
        torch.save({"offsets": sorted(sparse), "captures": sparse}, capture_path)
        raw = {
            "schema": "repetition_history_runtime.raw.v1",
            "mode": mode,
            "image_id": target,
            "group_key": group.get("key"),
            "rows": rows,
            "input_identity": input_identity,
            "loaded_model_identity": dict(loaded_identity),
            "readout": dict(readout),
            "prefix": {"start_offset": prefix_start, "token_ids": prefix_tokens},
            "forced_replay": forced_replay,
            "norm_pulse": {
                "start_offset": None if mode == "prefix" else norm_start,
                "row_limit": row_limit if mode != "prefix" else 0,
                "complete_rows": complete_rows,
                "complete": mode == "prefix" or row_limit is None or complete_rows >= row_limit,
                "incomplete": mode != "prefix" and row_limit is not None and complete_rows < row_limit,
                "termination_reason": rows[target_position]["stop"],
                "last_active_offset": pulse_last_offset,
                "steps": pulse_steps,
                "captures": _binding(capture_path),
            },
            "policy": "native greedy with target literal prefix force; coordinate norm pulse then withdrawal",
        }
        _write(run_dir / "raw.json", raw)
        receipt.update(
            status="candidate_complete",
            input_identity=input_identity,
            prefill_mrope_sha256=prefill_mrope,
            model_forwards=model_forwards,
            forced_replay_slots=len(forced_replay),
            pulse_steps=len(pulse_steps),
            complete_rows=complete_rows,
            pulse_complete=mode == "prefix" or row_limit is None or complete_rows >= row_limit,
            pulse_incomplete=mode != "prefix" and row_limit is not None and complete_rows < row_limit,
            target_stop=rows[target_position]["stop"],
            last_active_offset=pulse_last_offset,
            raw=_binding(run_dir / "raw.json"),
            pulse_captures=_binding(capture_path),
            elapsed_seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        _write(run_dir / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - began)
        _write(run_dir / "receipt.json", receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


def run_norm(*, panel_path: Path, image_id: int, mode: str, output: Path) -> dict[str, Any]:
    panel = _read(panel_path)
    fresh._check_sources(panel)
    cell = _cell(panel, image_id)
    config = panel.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("panel config is missing")
    infer = fresh.InferConfig.model_validate(config)
    if infer.backend.type != "hf" or infer.model.dtype != "fp32" or infer.backend.hf.attn_implementation != "sdpa":
        raise AssertionError("norm pulse requires frozen FP32/SDPA HF inference")
    qwen, loaded_identity = fresh.load_policy(infer, device=torch.device("cuda:0"))
    qwen.model.eval()
    coordinate_ids, factors, readout = fresh._load_coefficients(
        panel=panel,
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        device=torch.device("cuda:0"),
    )
    readout["loaded_model_identity"] = loaded_identity
    return _run_norm_cell(
        panel_path=panel_path,
        panel=panel,
        cell=cell,
        mode=mode,
        output=output,
        qwen=qwen,
        infer=infer,
        loaded_identity=loaded_identity,
        coordinate_ids=coordinate_ids,
        factors=factors,
        readout=readout,
    )


def cpu_check(output: Path) -> None:
    """Check target-only force, coordinate-only norm, and pulse withdrawal."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cpu-check.json"
    if path.exists():
        raise FileExistsError(path)
    scores = torch.tensor([[8.0, 1.0, 7.0, 2.0], [6.0, 5.0, 4.0, 3.0]])
    forced = _force(scores, 1, 2)
    assert torch.equal(forced[0], scores[0])
    assert int(torch.argmax(forced[1]).item()) == 2
    scaled = _norm(
        scores,
        torch.tensor([1, 3]),
        torch.tensor([2.0, 0.5], dtype=torch.float64),
        1,
    )
    assert torch.equal(scaled[0], scores[0])
    assert torch.equal(scaled[:, [0, 2]], scores[:, [0, 2]])
    assert torch.equal(scaled[:, 1], torch.tensor([1.0, 10.0]))
    assert torch.equal(scaled[:, 3], torch.tensor([2.0, 1.5]))
    assert not _pulse_active(
        norm_started=False, target_done=False, completed_rows=0, row_limit=1
    )
    assert _pulse_active(
        norm_started=True, target_done=False, completed_rows=0, row_limit=1
    )
    assert not _pulse_active(
        norm_started=True, target_done=False, completed_rows=1, row_limit=1
    )
    assert _pulse_active(
        norm_started=True, target_done=False, completed_rows=3, row_limit=4
    )
    assert not _pulse_active(
        norm_started=True, target_done=False, completed_rows=4, row_limit=4
    )
    assert not _pulse_active(
        norm_started=True, target_done=True, completed_rows=0, row_limit=None
    )
    _write(path, {
        "status": "passed",
        "target_only_force": True,
        "coordinate_only_norm": True,
        "prefix_half_open_interval": True,
        "pulse_withdrawal_boundary": "after selected box_end",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--mode", choices=("prefix", "pulse1", "pulse4", "sustained"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/repetition-history"))
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check(args.output_root)
        return
    if args.panel is None or args.case is None:
        parser.error("--panel and --case are required unless --cpu-check is used")
    if args.arm is not None:
        if args.mode is not None:
            parser.error("--arm and --mode are separate stages")
        run_fixed_prefix(
            panel_path=args.panel,
            image_id=args.case,
            arm=args.arm,
            output_root=args.output_root,
        )
    elif args.mode is not None:
        run_norm(
            panel_path=args.panel,
            image_id=args.case,
            mode=args.mode,
            output=args.output_root,
        )
    else:
        parser.error("provide --arm for fixed-prefix or --mode for norm pulse")


if __name__ == "__main__":
    main()
