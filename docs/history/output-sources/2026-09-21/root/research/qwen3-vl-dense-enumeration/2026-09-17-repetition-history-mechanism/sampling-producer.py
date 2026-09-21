"""External categorical pulse on one native target row.

HF generation remains greedy.  A logits processor samples one target token
with a local torch.Generator, forces that token at the native seam, and then
withdraws after one syntactically complete row, 32 draws, EOS, or the cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import LogitsProcessor, LogitsProcessorList

ROOT = Path("/data/CoordExp/.worktrees/research-probes")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from probes.training_set_completion import repetition_history_runtime as runtime  # noqa: E402


EOS = runtime.EOS
MAX_NEW_TOKENS = runtime.MAX_NEW_TOKENS
ROW_OPEN = 151646
REF_END = 151647
BOX_START = 151648
BOX_END = 151649
MAX_SAMPLE_TOKENS = 32
ALLOWED_TEMPERATURES = (0.1, 0.3, 0.7)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(f"{label} must be an integer token list")
    return list(value)


def _rows(value: object) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("saved raw rows must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("saved raw row must be an object")
        image = runtime._image_id(row)
        if image in result:
            raise ValueError(f"duplicate saved image_id={image}")
        result[image] = row
    return result


def _saved_tokens(row: Mapping[str, Any]) -> list[int]:
    return _tokens(row.get("token_ids", row.get("generated_token_ids")), "saved row")


def _saved_stop(row: Mapping[str, Any]) -> str:
    value = row.get("stop", row.get("decode_stop_reason"))
    if not isinstance(value, str):
        raise ValueError("saved row has no stop reason")
    return value


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _complete_row(history: list[int], coordinate_ids: set[int]) -> bool:
    """Recognize one complete grammar row, retaining malformed BOX_END draws."""
    if len(history) < 9 or history[-1] != BOX_END:
        return False
    box_start = len(history) - 6
    ref_end = box_start - 1
    if history[ref_end] != REF_END or history[box_start] != BOX_START:
        return False
    if not all(token in coordinate_ids for token in history[-5:-1]):
        return False
    opener = max(
        (index for index, token in enumerate(history[:ref_end]) if token == ROW_OPEN),
        default=-1,
    )
    if opener < 0 or ref_end - opener < 2:
        return False
    category = history[opener + 1 : ref_end]
    return all(token not in {ROW_OPEN, REF_END, BOX_START, BOX_END, EOS} for token in category)


def _sample_token(
    scores: torch.Tensor, temperature: float, generator: torch.Generator
) -> tuple[int, dict[str, Any]]:
    logits = scores.float()
    raw_winner = int(torch.argmax(logits).item())
    scaled = logits / temperature
    scaled_winner = int(torch.argmax(scaled).item())
    if raw_winner != scaled_winner:
        raise AssertionError("positive-temperature scaling changed argmax")
    probabilities = torch.softmax(scaled, dim=-1)
    chosen = int(torch.multinomial(probabilities, 1, generator=generator).item())
    logprob = float(torch.log_softmax(scaled, dim=-1)[chosen].item())
    return chosen, {
        "raw_winner_token_id": raw_winner,
        "scaled_winner_token_id": scaled_winner,
        "raw_scaled_argmax_equal": True,
        "sampled_token_id": chosen,
        "sampled_logprob_at_temperature": logprob,
        "temperature": temperature,
    }


def _force(scores: torch.Tensor, row: int, token: int) -> torch.Tensor:
    transformed = scores.clone()
    transformed[row].fill_(-torch.inf)
    transformed[row, token] = 0
    companions = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    companions[row] = False
    if not torch.equal(scores[companions], transformed[companions]):
        raise AssertionError("target force edited a native companion")
    return transformed


def _cell_prefix(
    cell: Mapping[str, Any], saved: Mapping[int, Mapping[str, Any]], target: int
) -> tuple[int, list[int]]:
    values = cell.get("target_prefix_token_ids")
    if values is None:
        length = cell.get("target_prefix_length", 54)
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("target_prefix_length must be positive")
        values = _saved_tokens(saved[target])[:length]
    tokens = _tokens(values, "target_prefix_token_ids")
    start = cell.get("prefix_start_offset", 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("prefix_start_offset must be nonnegative")
    if start + len(tokens) > MAX_NEW_TOKENS or EOS in tokens:
        raise ValueError("target prefix is outside the frozen budget")
    return start, tokens


def run(*, panel_path: Path, image_id: int, output: Path) -> dict[str, Any]:
    if image_id != 309264:
        raise ValueError("StageC is frozen to bird image_id=309264")
    panel = _read(panel_path)
    runtime.fresh._check_sources(panel)
    cell = runtime._cell(panel, image_id)
    group = runtime._group(cell)
    cases = group.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("sampling group must contain four native cases")
    image_ids = [runtime._image_id(case) for case in cases]
    target_position = int(cell.get("target_position", image_ids.index(image_id)))
    if image_ids[target_position] != image_id:
        raise ValueError("target_position does not identify image_id")
    saved_binding = cell.get("saved_raw")
    receipt_binding = cell.get("saved_receipt")
    if not isinstance(saved_binding, Mapping) or not isinstance(saved_binding.get("path"), str):
        raise ValueError("sampling cell needs saved_raw binding")
    saved_path = Path(saved_binding["path"])
    if _binding(saved_path) != dict(saved_binding):
        raise AssertionError("saved raw binding changed")
    if not isinstance(receipt_binding, Mapping) or not isinstance(receipt_binding.get("path"), str):
        raise ValueError("sampling cell needs saved_receipt binding")
    receipt_path = Path(receipt_binding["path"])
    if _binding(receipt_path) != dict(receipt_binding):
        raise AssertionError("saved receipt binding changed")
    saved_receipt = _read(receipt_path)
    if saved_receipt.get("status") != "candidate_complete":
        raise AssertionError("saved native receipt is not candidate_complete")
    saved = _rows(_read(saved_path).get("rows"))
    if set(saved) != set(image_ids):
        raise AssertionError("saved rows do not match native group")
    prefix_start, prefix_tokens = _cell_prefix(cell, saved, image_id)
    prefix_end = prefix_start + len(prefix_tokens)
    if prefix_start != 0 or prefix_end != 54:
        raise ValueError("StageC requires the frozen original 54-token prefix")
    common_length = cell.get("common_native_prefix_length", prefix_end)
    if (
        isinstance(common_length, bool)
        or not isinstance(common_length, int)
        or common_length < 0
        or common_length > prefix_end
    ):
        raise ValueError("common_native_prefix_length is outside supplied prefix")
    temperature = cell.get("sampling_temperature")
    seed = cell.get("sampling_seed")
    max_sample_tokens = cell.get("max_sample_tokens", MAX_SAMPLE_TOKENS)
    if temperature not in ALLOWED_TEMPERATURES:
        raise ValueError("sampling_temperature must be one of 0.1, 0.3, 0.7")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 19 <= seed <= 26:
        raise ValueError("sampling_seed must be in 19..26")
    if max_sample_tokens != MAX_SAMPLE_TOKENS:
        raise ValueError("max_sample_tokens is frozen at 32")
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "repetition_history_runtime.external_sampling.v1",
        "status": "running",
        "image_id": image_id,
        "image_ids": image_ids,
        "target_position": target_position,
        "prefix_start_offset": prefix_start,
        "prefix_length": len(prefix_tokens),
        "common_native_prefix_length": common_length,
        "sampling_temperature": temperature,
        "sampling_seed": seed,
        "max_sample_tokens": max_sample_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling_pulse": "external categorical target-only; native greedy before/after",
        "panel": _binding(panel_path),
        "runtime": _binding(ROOT / "probes/training_set_completion/repetition_history_runtime.py"),
        "producer": _binding(Path(__file__).resolve()),
        "pid": os.getpid(),
        "saved_raw": dict(saved_binding),
        "saved_receipt": dict(receipt_binding),
        "norm_transform": "none; external categorical sampling only",
        "raw_path": str(output / "raw.json"),
    }
    _write(output / "receipt.json", receipt)
    began = time.monotonic()
    handles: list[Any] = []
    model_forwards = 0
    head_input: torch.Tensor | None = None
    prefill_mrope: str | None = None
    histories: list[list[int]] = [[] for _ in image_ids]
    done = [False] * 4
    forced_replay: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sparse: dict[int, dict[str, torch.Tensor]] = {}
    sampled_count = 0
    pulse_complete = False
    pulse_last_offset: int | None = None
    generator: torch.Generator | None = None
    rng_before: torch.Tensor | None = None
    try:
        config = panel.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("panel config is missing")
        infer = runtime.fresh.InferConfig.model_validate(config)
        if infer.backend.type != "hf" or infer.model.dtype != "fp32" or infer.backend.hf.attn_implementation != "sdpa":
            raise AssertionError("sampling pulse requires frozen FP32/SDPA HF inference")
        request_config = dict(config)
        request_data = dict(config["data"])
        input_jsonl = group.get("input_jsonl", request_data.get("input_jsonl"))
        if not isinstance(input_jsonl, str):
            raise ValueError("native group needs input_jsonl")
        request_data["input_jsonl"] = input_jsonl
        request_config["data"] = request_data
        qwen, loaded_identity = runtime.fresh.load_policy(infer, device=torch.device("cuda:0"))
        model = qwen.model.eval()
        requests, _ = runtime.fresh.build_bound_native_requests(qwen, request_config, cases)
        batch = runtime.fresh.prepare_native_inputs(
            qwen.processor, requests, device="cuda:0", record_media_identity=True
        )
        input_identity = runtime.fresh._input_identity(batch)
        expected_identity = saved_receipt.get("input_identity")
        if not isinstance(expected_identity, Mapping) or input_identity != dict(expected_identity):
            raise AssertionError("native input identity differs from saved receipt")
        width = int(batch.inputs["input_ids"].shape[1])
        if width <= 0:
            raise AssertionError("native batch has no prompt width")
        coordinate_ids = {
            qwen.tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>") for index in range(1000)
        }
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(seed)
        rng_before = generator.get_state().clone()
        initial_versions = {name: value._version for name, value in model.named_parameters()}

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
                    raise AssertionError("native prefill did not pass MRoPE positions")
                prefill_mrope = _tensor_hash(positions)

        def head_capture(module: Any, args: tuple[Any, ...]) -> None:
            nonlocal head_input
            if not args or not isinstance(args[0], torch.Tensor) or args[0].ndim != 3:
                raise AssertionError("lm_head input shape is invalid")
            if args[0].shape[0] != 4:
                raise AssertionError("lm_head input is not the native four-row batch")
            head_input = args[0][:, -1, :].detach().clone()

        handles.extend([
            model.register_forward_pre_hook(model_counter, with_kwargs=True),
            model.model.language_model.register_forward_pre_hook(mrope_capture, with_kwargs=True),
            model.get_output_embeddings().register_forward_pre_hook(head_capture),
        ])

        class SamplingProcessor(LogitsProcessor):
            def __call__(self, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal sampled_count, pulse_complete, pulse_last_offset
                if head_input is None or generator is None:
                    raise AssertionError("sampling seam lacks head input or generator")
                offset = int(ids.shape[1] - width)
                if offset < 0 or offset >= MAX_NEW_TOKENS:
                    raise AssertionError("action offset is outside the frozen budget")
                target_done = done[target_position]
                active = (
                    not target_done
                    and offset >= prefix_end
                    and not pulse_complete
                    and sampled_count < MAX_SAMPLE_TOKENS
                )
                transformed = scores
                if not target_done and offset < common_length:
                    expected = _saved_tokens(saved[image_id])
                    if offset >= len(expected) or int(torch.argmax(scores[target_position]).item()) != expected[offset]:
                        raise AssertionError("natural target prefix argmax differs from saved native row")
                forced = prefix_start <= offset < prefix_end
                if forced:
                    token = prefix_tokens[offset - prefix_start]
                    transformed = _force(scores, target_position, token)
                    forced_replay.append({
                        "offset": offset,
                        "token_id": token,
                        "native_winner_token_id": int(torch.argmax(scores[target_position]).item()),
                        "history_sha256": _json_hash(histories[target_position]),
                    })
                sampled_token: int | None = None
                sample_detail: dict[str, Any] | None = None
                if active:
                    draw_rng_before = generator.get_state().cpu().clone()
                    sampled_token, sample_detail = _sample_token(scores[target_position], temperature, generator)
                    transformed = _force(scores, target_position, sampled_token)
                    sampled_count += 1
                    pulse_last_offset = offset
                    sparse[offset] = {
                        "rng_before": draw_rng_before,
                        "rng_after": generator.get_state().cpu().clone(),
                        "raw_logits": scores[target_position].detach().cpu().clone(),
                        "after_scores": transformed[target_position].detach().cpu().clone(),
                        "lm_head_input": head_input[target_position].detach().cpu().clone(),
                    }
                for index, image in enumerate(image_ids):
                    if done[index]:
                        continue
                    history = ids[index, width:].tolist()
                    if history != histories[index]:
                        raise AssertionError(f"history forked for image {image}")
                    chosen = int(torch.argmax(transformed[index]).item())
                    if index == target_position and active:
                        assert sampled_token is not None and sample_detail is not None
                        if chosen != sampled_token:
                            raise AssertionError("external sampled token was not forced")
                        detail = {
                            "offset": offset,
                            "history_sha256": _json_hash(history),
                            **sample_detail,
                            "native_winner_token_id": sample_detail["raw_winner_token_id"],
                            "transformed_winner_token_id": sampled_token,
                            "winner_changed": sample_detail["raw_winner_token_id"] != sampled_token,
                            "complete_rows_before": int(pulse_complete),
                            "malformed_box_end": chosen == BOX_END and not _complete_row(history + [chosen], coordinate_ids),
                        }
                        samples.append(detail)
                    histories[index].append(chosen)
                    if index == target_position and active and _complete_row(histories[index], coordinate_ids):
                        pulse_complete = True
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
                raise AssertionError("generation must remain native greedy")
            return original_generate(
                **kwargs, logits_processor=LogitsProcessorList([SamplingProcessor()])
            )

        model.generate = generate_wrapper
        try:
            with torch.no_grad():
                values = runtime.fresh.generate_continuations(
                    model,
                    batch,
                    extensions=[[] for _ in cases],
                    budgets=[MAX_NEW_TOKENS for _ in cases],
                    eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id,
                    policy=runtime.fresh.NativeGenerationPolicy(
                        temperature=0, top_p=1, top_k=0, repetition_penalty=1,
                        use_model_defaults=False,
                    ),
                    trace="none",
                    seed=None,
                )
        finally:
            model.generate = original_generate
        if len(values) != 4 or prefill_mrope is None:
            raise AssertionError("native generation identity is incomplete")
        if saved_receipt.get("prefill_mrope_sha256") != prefill_mrope:
            raise AssertionError("native prefill MRoPE differs from saved receipt")
        if runtime.fresh._input_identity(batch) != input_identity:
            raise AssertionError("generation mutated native inputs")
        if {name: value._version for name, value in model.named_parameters()} != initial_versions:
            raise AssertionError("generation mutated model parameters")
        rows = []
        for image, request, value in zip(image_ids, requests, values, strict=True):
            token_ids = list(value.token_ids)
            rows.append({
                "image_id": image,
                "request_id": request.request_id,
                "token_ids": token_ids,
                "tokens_sha256": _json_hash(token_ids),
                "stop": value.stop_reason,
                "text": qwen.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })
        for row in rows:
            image = row["image_id"]
            if image != image_id:
                if row["token_ids"] != _saved_tokens(saved[image]) or row["stop"] != _saved_stop(saved[image]):
                    raise AssertionError(f"native companion changed for image {image}")
        target_tokens = rows[target_position]["token_ids"]
        if target_tokens[prefix_start:prefix_end] != prefix_tokens:
            raise AssertionError("target literal prefix was not retained")
        rng_after = generator.get_state().clone()
        rng_path = output / "rng-states.pt"
        torch.save({"before": rng_before.cpu(), "after": rng_after.cpu()}, rng_path)
        sparse_path = output / "sampled-logits.pt"
        torch.save({"offsets": sorted(sparse), "captures": sparse}, sparse_path)
        raw = {
            "schema": "repetition_history_runtime.external_sampling.raw.v1",
            "image_id": image_id,
            "image_ids": image_ids,
            "rows": rows,
            "input_identity": input_identity,
            "prefix": {"start_offset": prefix_start, "token_ids": prefix_tokens},
            "forced_replay": forced_replay,
            "sampling_pulse": {
                "start_offset": prefix_end,
                "row_limit": 1,
                "temperature": temperature,
                "seed": seed,
                "max_sample_tokens": MAX_SAMPLE_TOKENS,
                "steps": samples,
                "draw_count": sampled_count,
                "complete_row": pulse_complete,
                "incomplete": not pulse_complete,
                "last_active_offset": pulse_last_offset,
                "withdrawal": "after first syntactically complete row, 32 draws, target EOS, or cap",
                "rng_states": _binding(rng_path),
                "logits": _binding(sparse_path),
            },
            "policy": "native greedy seam with external target categorical pulse; no HF sampling and no norm transform",
        }
        _write(output / "raw.json", raw)
        torch.cuda.synchronize()
        receipt.update(
            status="candidate_complete",
            loaded_identity=loaded_identity,
            input_identity=input_identity,
            prefill_mrope_sha256=prefill_mrope,
            model_forwards=model_forwards,
            sampled_draws=sampled_count,
            pulse_complete=pulse_complete,
            pulse_incomplete=not pulse_complete,
            last_active_offset=pulse_last_offset,
            last_sample_offset=pulse_last_offset,
            rng_states=_binding(rng_path),
            logits=_binding(sparse_path),
            raw=_binding(output / "raw.json"),
            elapsed_seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        _write(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - began)
        _write(output / "receipt.json", receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


def cpu_check(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    coordinate_ids = {10, 11, 12, 13}
    complete = [ROW_OPEN, 99, REF_END, BOX_START, 10, 11, 12, 13, BOX_END]
    malformed = [ROW_OPEN, 99, REF_END, BOX_START, 10, 11, 12, BOX_END]
    assert _complete_row(complete, coordinate_ids)
    assert not _complete_row(malformed, coordinate_ids)
    scores = torch.tensor([[2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    forced = _force(scores, 1, 0)
    assert torch.equal(forced[0], scores[0]) and int(torch.argmax(forced[1])) == 0
    generator_a = torch.Generator(device="cpu").manual_seed(19)
    generator_b = torch.Generator(device="cpu").manual_seed(19)
    first, detail = _sample_token(scores[0], 0.3, generator_a)
    second, detail_b = _sample_token(scores[0], 0.3, generator_b)
    assert first == second and detail == detail_b
    assert detail["raw_scaled_argmax_equal"] is True
    assert _pulse_window(True, False, 0, 1)
    assert not _pulse_window(True, False, 1, 1)
    assert _pulse_window(True, False, 3, 4)
    assert not _pulse_window(True, False, 4, 4)
    assert not _pulse_window(True, True, 0, None)
    _write(path, {
        "status": "passed",
        "frozen_target_image_id": 309264,
        "frozen_prefix_length": 54,
        "seeded_generator_determinism": True,
        "positive_temperature_argmax_preserved": True,
        "target_only_force": True,
        "complete_row_requires_wrapper_category_box_and_four_coordinates": True,
        "malformed_box_end_keeps_sampling": True,
        "withdrawal_windows": ["first_complete_row", "32_draw_cap", "target_eos", "3084_cap"],
    })


def _pulse_window(started: bool, target_done: bool, completed_rows: int, row_limit: int | None) -> bool:
    return started and not target_done and (row_limit is None or completed_rows < row_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--image", type=int, default=309264)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("pulse1",), default="pulse1")
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check(args.out or Path("cpu-sampling-check.json"))
        return
    if args.panel is None or args.out is None:
        parser.error("--panel and --out are required unless --cpu-check is used")
    run(panel_path=args.panel, image_id=args.image, output=args.out)


if __name__ == "__main__":
    main()
