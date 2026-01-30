#!/usr/bin/env python
"""Dump raw generation ("rollout text") for a few real samples.

This is a lightweight debugging tool to sanity-check whether a checkpoint's
outputs are:
- valid JSON,
- contain coord tokens,
- end with <|im_end|>,
- truncated / malformed.

It uses the same inference engine utilities as `scripts/run_infer.py`, but dumps
RAW generation text for a small set of indices.

Example:
  CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/ms/bin/python scripts/tools/dump_rollout_text.py \
    --jsonl public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
    --ckpt output/1-26/checkpoint-1516-merged \
    --indices 2,6,8 \
    --out temp/rollout_text_samples.txt

Notes:
- This is intended for *short* runs (a handful of samples). For bulk evaluation,
  prefer `scripts/run_infer.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.parsing import extract_special_tokens, load_prediction_dict
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine


def _parse_indices(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    out: List[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _load_records(jsonl_path: Path, indices: Iterable[int]) -> List[tuple[int, Dict[str, Any]]]:
    wanted = set(int(i) for i in indices)
    if not wanted:
        return []

    found: List[tuple[int, Dict[str, Any]]] = []
    max_i = max(wanted)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > max_i:
                break
            if i not in wanted:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            found.append((i, rec))

    found.sort(key=lambda x: x[0])
    return found


def _safe_head(items: List[Any], n: int) -> List[Any]:
    return items[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to dataset JSONL")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint folder path")
    ap.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Comma-separated line indices to dump, e.g. 2,6,8",
    )
    ap.add_argument("--out", type=Path, default=Path("temp/rollout_text_samples.txt"))

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    jsonl_path: Path = args.jsonl
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    indices = _parse_indices(args.indices)
    if not indices:
        raise ValueError("--indices must be a non-empty comma-separated list")

    # Keep caller's ROOT_IMAGE_DIR if already set; otherwise default to JSONL parent.
    os.environ.setdefault("ROOT_IMAGE_DIR", str(jsonl_path.parent.resolve()))

    records = _load_records(jsonl_path, indices)
    if len(records) != len(set(indices)):
        found_idx = {i for i, _ in records}
        missing = [i for i in sorted(set(indices)) if i not in found_idx]
        raise IndexError(f"Some indices were not found in the first {max(indices)+1} lines: {missing}")

    cfg = InferenceConfig(
        gt_jsonl=str(jsonl_path),
        model_checkpoint=str(args.ckpt),
        mode="coord",
        device=str(args.device),
        limit=0,
        backend_type="hf",
    )

    gen_cfg = GenerationConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
    )

    engine = InferenceEngine(cfg, gen_cfg)
    engine.load_model()

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"CKPT: {args.ckpt}")
    lines.append(f"JSONL: {jsonl_path}")
    lines.append(f"ROOT_IMAGE_DIR={os.environ.get('ROOT_IMAGE_DIR')}")
    lines.append(
        "GEN: "
        f"temperature={gen_cfg.temperature} top_p={gen_cfg.top_p} "
        f"max_new_tokens={gen_cfg.max_new_tokens} repetition_penalty={gen_cfg.repetition_penalty} seed={gen_cfg.seed}"
    )
    lines.append("")

    for idx, rec in records:
        img_rel = (rec.get("images") or [None])[0]
        objs = rec.get("objects") or []
        gt_n = len(objs) if isinstance(objs, list) else 0
        w = rec.get("width")
        h = rec.get("height")

        img_path, image = engine._prepare_image(jsonl_path, rec)  # type: ignore[attr-defined]
        lines.append("=" * 80)
        lines.append(f"INDEX={idx} image={img_rel} size={w}x{h} gt_objects={gt_n}")
        if objs and isinstance(objs, list):
            descs = [o.get("desc") for o in objs[:8] if isinstance(o, dict)]
            lines.append(f"GT desc head: {descs}")

        if image is None:
            lines.append(f"IMAGE_LOAD_FAILED path={img_path}")
            lines.append("")
            continue

        raw = engine._generate(image)  # type: ignore[attr-defined]
        ends_im_end = raw.endswith("<|im_end|>")
        specials = extract_special_tokens(raw)

        lines.append(f"RAW ends_with_<|im_end|>: {ends_im_end}")
        lines.append(f"RAW special_tokens(head): {_safe_head(specials, 20)}")
        lines.append(f"RAW char_len: {len(raw)}")
        lines.append("--- RAW OUTPUT (truncated to 4000 chars for readability) ---")
        raw_show = raw if len(raw) <= 4000 else (raw[:4000] + "\n...<TRUNCATED>...")
        lines.append(raw_show)

        try:
            parsed = load_prediction_dict(raw)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"PARSE: FAILED {exc!r}")
            lines.append("")
            continue

        pred_n = len(parsed) if isinstance(parsed, dict) else -1
        keys_head = list(parsed.keys())[:5] if isinstance(parsed, dict) else []
        lines.append(f"PARSE: OK pred_objects={pred_n} keys_head={keys_head}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
