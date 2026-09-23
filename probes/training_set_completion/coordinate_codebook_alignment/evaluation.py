"""Native empty-prefix evaluation for the coordinate-codebook package.

The module exposes ``evaluate_cases`` for the parent queue and a small CLI for
one image or a task list.  Model loading is deliberately inside ``main``;
importing this module and running its fixture tests is CPU-only.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F

from probes.training_set_completion.artifacts import binding
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.inference.bound_requests import build_bound_native_requests
from src.inference.parsing import parse_compact_object_box_closed
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import prepare_native_inputs, prepare_replay
from src.templates.renderer import render_example
from src.config.models import TemplateConfig
from src.data.examples import raw_example_from_jsonl_row
from src.config.inference import InferConfig
from src.inference.inputs import plan_examples


CAP = 3084
EOS = 151645
POLICY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"dataset contains a non-object row: {path}")
    return rows


def _config(admission: Mapping[str, Any], dataset: Path) -> dict[str, Any]:
    config = copy.deepcopy(admission["source_config"])
    # The historical admission config records source provenance. Execution is
    # frozen here to the production dynamic-HF precision/attention pair.
    config["model"] = dict(config.get("model", {}), dtype="bf16")
    config["backend"] = dict(config.get("backend", {}), hf=dict(config.get("backend", {}).get("hf", {}), attn_implementation="flash_attention_2"))
    config["data"] = dict(config.get("data", {}), input_jsonl=str(dataset.resolve(strict=True)))
    config["generation"] = dict(config.get("generation", {}), max_new_tokens=CAP, repetition_penalty=1.0, temperature=0.0, top_p=1.0)
    return config


def _clean_input_record(row: Mapping[str, Any], dataset: Path) -> dict[str, Any]:
    """Project an admitted row into the strict relative-image JSONL schema."""
    source = row.get("input_record", row)
    clean = {key: value for key, value in source.items() if not key.startswith("_")}
    if "images" in clean:
        references = []
        for reference in clean["images"]:
            image = Path(str(reference))
            if not image.is_absolute():
                image = (dataset.parent / image).resolve(strict=True)
            else:
                image = image.resolve(strict=True)
            references.append(os.path.relpath(image, dataset.parent))
        clean["images"] = references
    return clean


def _target_ids(qwen: Any, row: Mapping[str, Any], config: Mapping[str, Any], dataset: Path) -> list[int]:
    # Normalized cases carry planning metadata beside the original JSONL
    # example. Re-render only the original input record.
    clean = _clean_input_record(row, dataset)
    raw = raw_example_from_jsonl_row(clean, jsonl_path=dataset, row_number=1, raw_line=json.dumps(clean))
    template = TemplateConfig(**{key: config["template"][key] for key in ("object_field_order", "object_ordering", "assistant_format", "prompt")})
    rendered = render_example(raw, template)
    return list(qwen.tokenizer.encode(rendered.supervised_response_text, add_special_tokens=False))


def _write_cell(path: Path, cell: Mapping[str, Any]) -> None:
    """Publish a cell atomically and refuse a conflicting duplicate."""
    payload = (json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"evaluation cell already exists with different content: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_case(qwen: Any, row: Mapping[str, Any], config: Mapping[str, Any], dataset: Path) -> dict[str, Any]:
    if "input_record" in row and "image_plan" in row:
        return dict(row)
    clean = _clean_input_record(row, dataset)
    raw = raw_example_from_jsonl_row(clean, jsonl_path=dataset, row_number=1, raw_line=json.dumps(clean))
    planned = plan_examples([raw], config=InferConfig.model_validate(config), components=qwen, row_indices=[0])[0]
    image = planned.image
    return {"row_id": str(row.get("_admission", {}).get("row_id", raw.example_id)), "row_index": 0, "input_record": clean, "image_path": image.image_path, "image_width": image.decoded_width, "image_height": image.decoded_height, "image_plan": {"backend_prompt_token_count": len(planned.prompt.expected_executed_prompt_token_ids), "image_content_sha256": image.image_content_sha256, "logical_transform_id": image.logical_transform_id, "merged_visual_tokens": image.merged_visual_tokens, "observed_image_grid_thw": image.expected_image_grid_thw}, "cohort": row.get("_admission", {}).get("cohort", row.get("cohort"))}


def _gt(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = []
    for index, obj in enumerate(row["input_record"]["objects"]):
        bins = [int(str(value)[len("<|coord_") : -2]) for value in obj["bbox_2d"]]
        values.append({"owner_id": str(obj["coco_ann_id"]), "description": str(obj["desc"]), "coord_bins": bins, "bbox": list(coord_bins_to_pixel_xyxy(bins, image_width=int(row["image_width"]), image_height=int(row["image_height"]), field=f"gt[{index}]"))})
    return values


def _hook_counters(model: Any) -> tuple[dict[str, int], list[Any]]:
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    counts = {"model_forwards": 0, "vision_forwards": 0}
    handles: list[Any] = []
    handles.append(model.register_forward_pre_hook(lambda *_args, **_kwargs: counts.__setitem__("model_forwards", counts["model_forwards"] + 1)))
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        handles.append(visual.register_forward_pre_hook(lambda *_args, **_kwargs: counts.__setitem__("vision_forwards", counts["vision_forwards"] + 1)))
    return counts, handles


def teacher_metrics(qwen: Any, batch: Any, row: Mapping[str, Any], config: Mapping[str, Any], dataset: Path) -> dict[str, Any]:
    targets = _target_ids(qwen, row, config, dataset)
    replay = prepare_replay(qwen.model, batch.inputs, prompt_token_ids=batch.prompt_token_ids[0], continuation_token_ids=targets, compact_logits=True)
    with torch.no_grad():
        logits = qwen.model(**replay.inputs).logits
    aligned = replay.aligned_logits(logits)
    target = torch.tensor(targets, device=aligned.device, dtype=torch.long)
    nll = F.cross_entropy(aligned.float(), target, reduction="none")
    target_logits = aligned.float().gather(1, target[:, None]).squeeze(1)
    competitor = aligned.float().clone()
    competitor.scatter_(1, target[:, None], float("-inf"))
    margins = target_logits - competitor.max(dim=1).values
    coordinate_ids = [qwen.tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>") for index in range(1000)]
    coordinate_bins = {token_id: index for index, token_id in enumerate(coordinate_ids)}
    coord_positions = [index for index, token in enumerate(targets) if token in coordinate_bins]
    coord_errors = []
    for index in coord_positions:
        predicted = int(aligned[index, coordinate_ids].argmax().item())
        expected = coordinate_bins[targets[index]]
        coord_errors.append(abs(predicted - expected) / 1000.0)
    return {"token_count": len(targets), "ce_sum": float(nll.sum().item()), "ce_mean": float(nll.mean().item()), "minimum_target_margin": float(margins.min().item()), "mean_target_margin": float(margins.mean().item()), "coordinate_token_count": len(coord_positions), "coordinate_mean_absolute_error": float(sum(coord_errors) / len(coord_errors)) if coord_errors else None}


def evaluate_cases(qwen: Any, *, admission: Mapping[str, Any], dataset: Path, cases: Sequence[Mapping[str, Any]], output_dir: Path, condition: str = "source", checkpoint: Mapping[str, Any] | None = None, include_teacher: bool = False, output_path: Path | None = None) -> list[Path]:
    """Evaluate bound cases using one loaded runtime; returns immutable cell paths."""
    config = _config(admission, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    for case_index, case in enumerate(cases):
        row = _normalize_case(qwen, case, config, dataset)
        request, _ = build_bound_native_requests(qwen, config, [row])
        batch = prepare_native_inputs(qwen.processor, request, device=next(qwen.model.parameters()).device, record_media_identity=True)
        counters, handles = _hook_counters(qwen.model)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=NativeGenerationPolicy(**POLICY, use_model_defaults=False), trace="raw_and_policy", seed=None)[0]
            text = qwen.tokenizer.decode(list(generated.token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            parsed = parse_compact_object_box_closed(text, row_id=str(row["row_id"]), row_index=int(row["row_index"]), image_width=int(row["image_width"]), image_height=int(row["image_height"])).to_artifact_dict()
            cell = {"schema": "coordinate_codebook_alignment.evaluation_cell.v1", "status": "complete", "condition": condition, "checkpoint": checkpoint, "case": {"row_id": row["row_id"], "image_id": row["input_record"]["image_id"], "cohort": row.get("_admission", {}).get("cohort", row.get("cohort")), "row_index": row["row_index"]}, "gt": _gt(row), "generation": {"token_ids": list(generated.token_ids), "text": text, "stop_reason": generated.stop_reason, "cap": CAP, "policy": POLICY, "raw_logprobs": list(generated.raw_logprobs or ()), "policy_logprobs": list(generated.policy_logprobs or ())}, "parser": parsed, "timing": {"wall_seconds": time.perf_counter() - started, **counters}, "media_sha256": list(batch.media_sha256), "prompt_token_ids": list(batch.prompt_token_ids[0])}
            cell["timing"]["decode_seconds"] = cell["timing"]["wall_seconds"]
            if include_teacher:
                teacher_started = time.perf_counter()
                cell["teacher"] = teacher_metrics(qwen, batch, row, config, dataset)
                cell["timing"]["teacher_seconds"] = time.perf_counter() - teacher_started
            cell["timing"].update(total_seconds=time.perf_counter() - started, **counters)
        except BaseException as exc:
            cell = {"schema": "coordinate_codebook_alignment.evaluation_cell.v1", "status": "technical_invalid", "condition": condition, "case": {"row_id": row["row_id"], "image_id": row["input_record"]["image_id"], "cohort": row.get("_admission", {}).get("cohort", row.get("cohort")), "row_index": row["row_index"]}, "error": repr(exc), "timing": {"wall_seconds": time.perf_counter() - started, **counters}}
        finally:
            for handle in handles:
                handle.remove()
        path = output_path if output_path is not None and len(cases) == 1 else output_dir / f"{row['row_id']}-{condition}.json"
        _write_cell(path, cell)
        results.append(path)
    return results


def _checkpoint_composition(checkpoint: str, admission: Mapping[str, Any]) -> tuple[Path, Path, Path | None, str]:
    source_config = admission["source_config"]
    if checkpoint == "source":
        adapter = Path(source_config["adapter"]["path"]).resolve(strict=True)
        embedding = Path(source_config["embedding_delta"]["path"]).resolve(strict=True)
        root = adapter.parent
    else:
        root = Path(checkpoint).expanduser().resolve(strict=True)
        if root.name == "adapter" and (root / "adapter_config.json").is_file():
            root = root.parent
        adapter = root / "adapter"
        embedding = root / "special_token_embeddings"
        if not adapter.is_dir() or not embedding.is_dir():
            raise ValueError(f"checkpoint root lacks adapter and special_token_embeddings: {root}")
    composition = root / "model_composition.json"
    if composition.exists():
        marker = json.loads(composition.read_text())
        expected = {"schema": 1, "coordinate_codebook": "coordinate_codebook", "untied_selected_rows": True}
        if marker != expected:
            raise ValueError(f"unsupported checkpoint model composition: {composition}")
    codebook = root / "coordinate_codebook"
    return adapter, embedding, (codebook if codebook.exists() else None), str(root)


def _load_runtime(checkpoint: str, device: str, admission: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load source or trained composition through the maintained dynamic HF backend."""
    from src.inference.backend import BackendLaunch
    from src.inference.hf_backend import _load_hf_components

    adapter, embedding, codebook, root = _checkpoint_composition(checkpoint, admission)
    model_path = str(Path(admission["source_config"]["model"]["base_model"]).resolve(strict=True))
    options = {"hf": {"attn_implementation": "flash_attention_2", "patch_embed_linearization": "enabled", "adapter_runtime": "live_promoted"}}
    if codebook is not None:
        options["hf"]["coordinate_codebook_path"] = str(codebook)
    launch = BackendLaunch(
        backend="hf", model_path=model_path, model_dtype="bf16", batch_size=1,
        generation_config_fingerprint=hashlib.sha256(json.dumps({"cap": CAP, "policy": POLICY}, sort_keys=True).encode()).hexdigest(),
        backend_options=options,
        adapter={"type": "dora", "name": "default", "path": str(adapter)},
        embedding_delta={"path": str(embedding), "source_gate_root": None},
    )
    loaded = _load_hf_components(launch)
    qwen = loaded.qwen
    qwen.model.to(torch.device(device)).eval()
    qwen.processor.image_processor.do_resize = False
    from src.adapters.dora import normalize_dora_state_key
    from probes.training_set_completion.coordinate_codebook_alignment.qualify import _tensor_hash
    parameter_hashes = {
        normalize_dora_state_key(name, adapter_name='default'): _tensor_hash(parameter)
        for name, parameter in qwen.model.named_parameters()
        if 'lora_' in name or name.endswith('shared_embed_delta') or name.endswith('coordinate_codebook.raw_gain') or name.endswith('coordinate_codebook.projection.weight')
    }
    payload_files = [*adapter.glob('*.safetensors'), adapter / 'adapter_config.json',
                     *embedding.glob('*.safetensors'), embedding / 'special_token_embeddings.json']
    if codebook is not None:
        payload_files += list(codebook.glob('*')) + [Path(root) / 'model_composition.json']
    return qwen, {"checkpoint_root": root, 'parameter_hashes': parameter_hashes,
                  'payload_bindings': [binding(p) for p in payload_files if p.is_file()],
                  "launch": {"backend": launch.backend, "model_path": launch.model_path, "model_dtype": launch.model_dtype, "backend_options": options, "adapter": launch.adapter, "embedding_delta": launch.embedding_delta}}


def _queue_specs(path: Path, common_dataset: Path | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        payload = {"specs": [{"image_id": item} if isinstance(item, int) else item for item in payload]}
    if not isinstance(payload, dict) or not isinstance(payload.get("specs"), list):
        raise ValueError("queue must be an object with a specs list")
    specs = []
    for index, item in enumerate(payload["specs"]):
        if not isinstance(item, dict) or "image_id" not in item or "condition" not in item:
            raise ValueError(f"queue spec {index} needs image_id and condition")
        spec = dict(item)
        spec["queue_index"] = index
        if "dataset" not in spec:
            if common_dataset is None:
                raise ValueError(f"queue spec {index} has no dataset")
            spec["dataset"] = str(common_dataset)
        spec.setdefault("cell_key", f"{spec['condition']}-{int(spec['image_id'])}")
        specs.append(spec)
    return specs


def _claim_queue(queue: Path, total: int) -> tuple[int, dict[str, Any]] | None:
    state_path = queue.with_name("queue-state.json")
    lock_path = queue.with_name("queue-state.json.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"next_index": 0, "claims": []}
        index = int(state.get("next_index", 0))
        if index >= total:
            return None
        state["next_index"] = index + 1
        state.setdefault("claims", []).append({"index": index, "pid": os.getpid(), "status": "claimed", "claimed_epoch": time.time()})
        temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(temporary, state_path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return index, state["claims"][-1]


def _finish_queue(queue: Path, index: int, status: str, output: str | None = None, error: str | None = None) -> None:
    state_path = queue.with_name("queue-state.json")
    lock_path = queue.with_name("queue-state.json.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        for claim in state.get("claims", []):
            if int(claim.get("index", -1)) == index:
                claim.update(status=status, finished_epoch=time.time())
                if output is not None:
                    claim["output"] = output
                if error is not None:
                    claim["error"] = error
                break
        temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(temporary, state_path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _cell_status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("status", "technical_invalid"))
    except Exception:
        return "technical_invalid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--condition", default="source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    if args.queue is None:
        if args.dataset is None:
            raise ValueError("--dataset is required without --queue")
        rows = _load_rows(args.dataset)
        cases = [row for row in rows if args.image_id is None or int(row["image_id"]) == args.image_id or int(row.get("input_record", {}).get("image_id", -1)) == args.image_id]
        if not cases:
            raise ValueError("no admitted case matches --image-id")
        qwen, identity = _load_runtime(args.checkpoint, args.device, admission)
        paths = evaluate_cases(qwen, admission=admission, dataset=args.dataset, cases=cases, output_dir=args.output.parent, condition=args.condition, checkpoint=identity, include_teacher=args.teacher, output_path=args.output)
        print(json.dumps({"status": "complete", "cells": [str(path) for path in paths], "model_calls": len(paths)}, sort_keys=True))
        return
    specs = _queue_specs(args.queue, args.dataset)
    qwen, identity = _load_runtime(args.checkpoint, args.device, admission)
    completed: list[str] = []
    while True:
        claim = _claim_queue(args.queue, len(specs))
        if claim is None:
            break
        index, _ = claim
        spec = specs[index]
        dataset = Path(str(spec["dataset"])).resolve(strict=True)
        try:
            rows = _load_rows(dataset)
            cases = [row for row in rows if int(row.get("image_id", row.get("input_record", {}).get("image_id", -1))) == int(spec["image_id"])]
            if len(cases) != 1:
                raise ValueError(f"queue image_id {spec['image_id']} resolved to {len(cases)} rows")
            output = args.output / "cells" / f"{spec['cell_key']}.json"
            paths = evaluate_cases(qwen, admission=admission, dataset=dataset, cases=cases, output_dir=output.parent, condition=str(spec["condition"]), checkpoint=identity, include_teacher=args.teacher, output_path=output)
            completed.extend(str(path) for path in paths)
            cell_status = "complete" if all(_cell_status(path) == "complete" for path in paths) else "technical_invalid"
            _finish_queue(args.queue, index, cell_status, output=str(output))
        except BaseException as exc:
            _finish_queue(args.queue, index, "technical_invalid", error=repr(exc))
            raise
    print(json.dumps({"status": "complete", "cells": completed, "queue": str(args.queue)}, sort_keys=True))


if __name__ == "__main__":
    main()
