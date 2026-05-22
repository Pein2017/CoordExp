#!/usr/bin/env python
"""Prepare offline Stage-2 residual-set rollout attempts.

This tool owns the v1 JSONL/provenance surface used by residual-set Channel-B
training. The current implementation provides a cheap preflight path and a
deterministic fixture producer; expensive model generation is intentionally not
started unless a future real backend is wired explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.config import ConfigLoader


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_model_checkpoint(config: Any) -> str:
    model_cfg = _as_mapping(getattr(config, "model", None))
    adapters = model_cfg.get("adapters")
    if isinstance(adapters, Sequence) and not isinstance(adapters, (str, bytes)):
        for adapter in adapters:
            if isinstance(adapter, str) and adapter.strip():
                return str(adapter).strip()
    model_path = model_cfg.get("model")
    if isinstance(model_path, str) and model_path.strip():
        return str(model_path).strip()
    raise ValueError("materialized config does not define model.model or model.adapters")


def _resolve_train_jsonl(config: Any) -> str:
    custom = getattr(config, "custom", None)
    train_jsonl = getattr(custom, "train_jsonl", None)
    if isinstance(train_jsonl, str) and train_jsonl.strip():
        return str(train_jsonl).strip()
    raise ValueError("materialized config does not define custom.train_jsonl")


def _residual_set_config(config: Any) -> Mapping[str, Any]:
    stage2_ab = getattr(config, "stage2_ab", None)
    pipeline = getattr(stage2_ab, "pipeline", None)
    objective = getattr(pipeline, "objective", ())
    for spec in objective or ():
        if getattr(spec, "name", None) != "residual_set_correction":
            continue
        if not bool(getattr(spec, "enabled", True)):
            continue
        channels = tuple(str(ch) for ch in (getattr(spec, "channels", ()) or ()))
        if "B" not in channels:
            continue
        cfg = getattr(spec, "config", None)
        if isinstance(cfg, Mapping):
            return cfg
    raise ValueError("config does not enable residual_set_correction on channel B")


def preflight_config(config_path: Path) -> dict[str, Any]:
    config = ConfigLoader.load_materialized_training_config(str(config_path))
    checkpoint = _resolve_model_checkpoint(config)
    train_jsonl = _resolve_train_jsonl(config)
    residual_cfg = dict(_residual_set_config(config))
    if "ckpt3664" in config_path.name and "checkpoint-3664" not in checkpoint:
        raise ValueError(
            "resolved checkpoint path must contain checkpoint-3664 for this smoke config: "
            f"{checkpoint}"
        )
    train_path = (ROOT / train_jsonl).resolve() if not Path(train_jsonl).is_absolute() else Path(train_jsonl)
    if not train_path.is_file():
        raise FileNotFoundError(f"resolved train JSONL does not exist: {train_path}")
    return {
        "config": str(config_path),
        "checkpoint": checkpoint,
        "train_jsonl": str(train_path),
        "prepared_rollout_jsonl": str(residual_cfg.get("prepared_rollout_jsonl", "")),
        "expected_num_rollouts": int(residual_cfg.get("expected_num_rollouts", 0) or 0),
    }


def _load_sample_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(records) >= int(limit):
                break
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if isinstance(record, dict):
                records.append(record)
    return records


def _infer_dataset_name_from_jsonl(path: Path) -> str:
    parts = {part.lower() for part in Path(str(path)).parts}
    for candidate in ("lvis", "coco", "refcoco", "refcoco+", "refcocog", "vg"):
        if candidate in parts:
            return candidate.replace("+", "plus")
    return Path(str(path)).stem


def _runtime_sample_id(dataset_name: str, base_idx: int) -> str:
    namespace = zlib.crc32(str(dataset_name).encode("utf-8")) & 0xFFFF
    return str((namespace << 32) | (int(base_idx) & 0xFFFFFFFF))


def _coord_token(value: Any) -> str:
    if isinstance(value, str) and value.startswith("<|coord_"):
        return value
    return f"<|coord_{int(value)}|>"


def _row_from_object(obj: Mapping[str, Any]) -> str:
    desc = str(obj.get("desc") or obj.get("category_name") or "object")
    bbox = obj.get("bbox_2d") or obj.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        bbox = [10, 20, 30, 40]
    coords = "".join(_coord_token(v) for v in bbox)
    return f"{OBJECT_REF_START_TOKEN}{desc}{BOX_START_TOKEN}{coords}"


def _raw_text_from_sample(sample: Mapping[str, Any], *, dirty_prefix: bool = False) -> str:
    objects = sample.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        objects = [{"desc": "object", "bbox_2d": [10, 20, 30, 40]}]
    rows = [_row_from_object(obj) for obj in objects[:2] if isinstance(obj, Mapping)]
    if not rows:
        rows = [_row_from_object({"desc": "object", "bbox_2d": [10, 20, 30, 40]})]
    raw_text = "\n".join(rows)
    if dirty_prefix:
        return "dirty-prefix " + raw_text
    return raw_text


def _fixture_variant_text(raw_text: str, *, sample_index: int, rollout_index: int) -> str:
    if int(rollout_index) <= 0:
        return raw_text
    x1 = 700 + int(sample_index) * 10 + int(rollout_index)
    extra = _row_from_object(
        {
            "desc": f"fixture_variant_{int(rollout_index)}",
            "bbox_2d": [x1, 20, x1 + 5, 40],
        }
    )
    return f"{raw_text}\n{extra}"


def _sample_id(
    sample: Mapping[str, Any],
    index: int,
    *,
    dataset_name: str | None = None,
) -> str:
    value = sample.get("sample_id")
    if value is not None:
        return str(value)
    if dataset_name:
        return _runtime_sample_id(str(dataset_name), int(index))
    image_id = sample.get("image_id")
    if image_id is not None:
        return str(image_id)
    return f"sample-{int(index)}"


def _image_id(sample: Mapping[str, Any], index: int) -> str:
    value = sample.get("image_id")
    return str(value) if value is not None else _sample_id(sample, int(index))


def _image_path(sample: Mapping[str, Any]) -> str:
    images = sample.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        return str(images[0])
    for key in ("image", "image_path", "file_name"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return ""


def generation_config_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_fixture_records(
    samples: Sequence[Mapping[str, Any]],
    *,
    encode_fn: Callable[[str], Sequence[int]],
    expected_num_rollouts: int,
    seed: int,
    greedy_rollouts: int,
    sampling_rollouts: int,
    dataset_name: str | None = None,
    include_debug_cases: Iterable[str] = (),
) -> list[dict[str, Any]]:
    gen_hash = generation_config_hash(
        {
            "expected_num_rollouts": int(expected_num_rollouts),
            "seed": int(seed),
            "greedy_rollouts": int(greedy_rollouts),
            "sampling_rollouts": int(sampling_rollouts),
            "producer": "fixture",
        }
    )
    debug_cases = {str(item).strip() for item in include_debug_cases if str(item).strip()}
    records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        sample_id = _sample_id(
            sample,
            int(sample_index),
            dataset_name=dataset_name,
        )
        source_sample_id = (
            str(sample.get("sample_id")) if sample.get("sample_id") is not None else None
        )
        rollout_specs = [
            ("greedy", ordinal)
            for ordinal in range(int(greedy_rollouts))
        ] + [
            ("sampling", ordinal)
            for ordinal in range(int(sampling_rollouts))
        ]
        for rollout_index, (decode_mode, decode_ordinal) in enumerate(rollout_specs):
            dirty = (
                "invalid_bbox_dirty_prefix" in debug_cases
                and int(sample_index) == 0
                and int(rollout_index) == len(rollout_specs) - 1
            )
            raw_text = _raw_text_from_sample(sample, dirty_prefix=dirty)
            if not dirty:
                raw_text = _fixture_variant_text(
                    raw_text,
                    sample_index=int(sample_index),
                    rollout_index=int(rollout_index),
                )
            record = {
                "sample_id": sample_id,
                "image_id": _image_id(sample, int(sample_index)),
                "image_path": _image_path(sample),
                "rollout_id": f"{sample_id}:r{int(rollout_index)}",
                "response_token_ids": [int(token_id) for token_id in encode_fn(raw_text)],
                "raw_text": raw_text,
                "decode_mode": str(decode_mode),
                "generation_config_hash": gen_hash,
                "sampling_seed": int(seed) + int(sample_index) * 1009 + int(rollout_index),
                "producer": "prepare_stage2_residual_rollouts.py",
                "producer_mode": "fixture",
                "decode_ordinal": int(decode_ordinal),
            }
            if dataset_name:
                record["dataset"] = str(dataset_name)
                record["base_idx"] = int(sample_index)
            if source_sample_id is not None and source_sample_id != sample_id:
                record["source_sample_id"] = source_sample_id
            if dirty:
                record["debug_case"] = "invalid_bbox_dirty_prefix"
            records.append(record)
        if "exact_duplicate_attempt" in debug_cases and int(sample_index) == 0 and rollout_specs:
            source_index = len(records) - len(rollout_specs)
            records[source_index]["debug_case"] = "exact_duplicate_attempt"
            duplicate = dict(records[source_index])
            duplicate["rollout_id"] = f"{duplicate['rollout_id']}:dup"
            duplicate["decode_mode"] = "sampling"
            duplicate["debug_case"] = "exact_duplicate_attempt"
            records.append(duplicate)
    return records


def _load_tokenizer_encode_fn(checkpoint: str) -> Callable[[str], Sequence[int]]:
    from transformers import AutoTokenizer

    tokenizer = None
    errors: list[str] = []
    candidates = _tokenizer_source_candidates(checkpoint)
    for source_index, source in enumerate(candidates):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            break
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            if not _should_try_next_tokenizer_source(
                source_index=source_index,
                candidates=candidates,
                source=source,
            ):
                raise
    if tokenizer is None:
        raise RuntimeError(
            "failed to load tokenizer for prepared residual rollout fixture; tried "
            + " | ".join(errors)
        )

    def _encode(text: str) -> Sequence[int]:
        return tokenizer.encode(str(text), add_special_tokens=False)

    return _encode


def _has_tokenizer_files(source: str) -> bool:
    path = Path(str(source))
    if not path.is_dir():
        return False
    return any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "merges.txt",
        )
    )


def _should_try_next_tokenizer_source(
    *,
    source_index: int,
    candidates: Sequence[str],
    source: str,
) -> bool:
    return (
        int(source_index) == 0
        and len(candidates) > 1
        and not _has_tokenizer_files(str(source))
        and (Path(str(source)) / "adapter_config.json").is_file()
    )


def _tokenizer_source_candidates(checkpoint: str) -> tuple[str, ...]:
    checkpoint_path = Path(str(checkpoint))
    candidates: list[str] = [str(checkpoint)]
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.is_file():
        try:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            adapter_config = {}
        base_model = adapter_config.get("base_model_name_or_path")
        if isinstance(base_model, str) and base_model.strip():
            candidates.append(base_model.strip())
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _parse_debug_cases(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-sample-limit", type=int, default=8)
    parser.add_argument("--expected-num-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--greedy-rollouts", type=int, default=1)
    parser.add_argument("--sampling-rollouts", type=int, default=3)
    parser.add_argument("--include-debug-cases", type=str, default="")
    parser.add_argument(
        "--mode",
        choices=("fixture", "preflight", "real"),
        default="fixture",
        help="fixture writes deterministic non-GPU records; real generation is not wired yet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --mode preflight; validates config/checkpoint/dataset without writing.",
    )
    args = parser.parse_args(argv)

    mode = "preflight" if bool(args.dry_run) else str(args.mode)
    if mode == "real":
        raise SystemExit(
            "real GPU generation is not implemented in this Task 2 producer; "
            "use --mode fixture for deterministic non-expensive JSONL output"
        )

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    info = preflight_config(config_path)
    if mode == "preflight":
        print(json.dumps({"status": "ok", **info}, sort_keys=True))
        return 0

    train_jsonl = Path(str(info["train_jsonl"]))
    records = _load_sample_records(train_jsonl, int(args.train_sample_limit))
    if not records:
        raise SystemExit(f"no training records found in {train_jsonl}")
    dataset_name = _infer_dataset_name_from_jsonl(train_jsonl)
    encode_fn = _load_tokenizer_encode_fn(str(info["checkpoint"]))
    out_records = build_fixture_records(
        records,
        encode_fn=encode_fn,
        expected_num_rollouts=int(args.expected_num_rollouts),
        seed=int(args.seed),
        greedy_rollouts=int(args.greedy_rollouts),
        sampling_rollouts=int(args.sampling_rollouts),
        dataset_name=dataset_name,
        include_debug_cases=_parse_debug_cases(args.include_debug_cases),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in out_records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "wrote",
                "out": str(args.out),
                "records": len(out_records),
                **info,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
