"""Report top-K records by object count and measure their GT assistant token length.

This is used after filtering to sanity-check whether the remaining densest
examples still fit within a reasonable generation/training budget.

We measure the token length of the *assistant JSON text only* (no system/user),
using the real tokenizer.

Example:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python \\
    scripts/report_topk_by_objects_tokens.py \\
    --jsonl public_data/lvis/rescale_32_768_poly_20/train.coord.filtered_u8_m200.jsonl \\
    --checkpoint model_cache/Qwen3-VL-8B-Instruct-coordexp \\
    --topk 5
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from src.coord_tokens.validator import annotate_coord_tokens
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import sort_objects_by_topleft


def _stream_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _norm_desc(desc: Any) -> str:
    if not isinstance(desc, str):
        return str(desc)
    return desc.strip().lower().split("/")[0]


def _effective_classes(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p + 1e-12)
    return float(math.exp(h)) if h > 0 else float(len(counts))


def _detect_coord_tokens_enabled(path: str) -> bool:
    name = os.path.basename(path).lower()
    if ".coord." in name or name.endswith(".coord.jsonl"):
        return True
    return False


def _load_selected_records(jsonl_path: str, line_nos: Sequence[int]) -> List[Tuple[int, Dict[str, Any]]]:
    wanted = set(int(x) for x in line_nos)
    out: List[Tuple[int, Dict[str, Any]]] = []
    for ln, rec in _stream_jsonl(jsonl_path):
        if ln in wanted:
            out.append((ln, rec))
            if len(out) >= len(wanted):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--coord_tokens_enabled",
        type=int,
        default=None,
        help="0/1 to override coord-token detection. Default: infer from filename.",
    )
    args = ap.parse_args()

    coord_tokens_enabled = (
        bool(int(args.coord_tokens_enabled))
        if args.coord_tokens_enabled is not None
        else _detect_coord_tokens_enabled(args.jsonl)
    )

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    builder = JSONLinesBuilder(
        user_prompt="Please describe all objects in the image.",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=coord_tokens_enabled,
    )

    # heap of (n_objects, line_no). Keep topk largest.
    heap: List[Tuple[int, int]] = []
    for ln, rec in _stream_jsonl(args.jsonl):
        objs = rec.get("objects") or []
        n = int(len(objs)) if isinstance(objs, list) else 0
        item = (n, int(ln))
        if len(heap) < int(args.topk):
            heapq.heappush(heap, item)
        else:
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
    line_nos = [ln for _, ln in selected]
    records = _load_selected_records(args.jsonl, line_nos)
    records_map = {ln: rec for ln, rec in records}

    print("Config")
    print(f"- jsonl: {args.jsonl}")
    print(f"- coord_tokens_enabled: {coord_tokens_enabled}")
    print(f"- checkpoint: {args.checkpoint}")
    print("")

    print(f"Top {len(selected)} by object count")
    for n_obj, ln in selected:
        rec = records_map.get(ln)
        if rec is None:
            print(f"- line={ln} objects={n_obj} (record not found?)")
            continue

        objs = rec.get("objects") or []
        if isinstance(objs, list) and objs:
            rec["objects"] = sort_objects_by_topleft(objs)
        if coord_tokens_enabled:
            annotate_coord_tokens(rec)

        counts: Counter[str] = Counter()
        polys = 0
        bboxes = 0
        max_pp = 0
        for o in rec.get("objects") or []:
            if not isinstance(o, dict):
                continue
            counts[_norm_desc(o.get("desc", ""))] += 1
            if "poly" in o:
                polys += 1
                pp = o.get("poly_points")
                if pp is not None:
                    try:
                        max_pp = max(max_pp, int(pp))
                    except Exception:
                        pass
            if "bbox_2d" in o:
                bboxes += 1

        n_unique = len([k for k in counts.keys() if k])
        top = counts.most_common(8)
        top1 = top[0][1] if top else 0
        top1_ratio = (float(top1) / float(n_obj)) if n_obj else 0.0
        eff = _effective_classes(counts)

        merged = builder.build_many([rec])
        assistant_text = merged["messages"][1]["content"][0]["text"]
        assistant_tokens = len(tok.encode(assistant_text, add_special_tokens=False))

        images = rec.get("images") or []
        img0 = images[0] if isinstance(images, list) and images else None
        print(
            "- line={} image={} objects={} unique={} top1_ratio={:.3f} eff={:.3f} polys={} bboxes={} max_pp={} assistant_tokens={} top8={}".format(
                ln,
                img0,
                n_obj,
                n_unique,
                top1_ratio,
                eff,
                polys,
                bboxes,
                max_pp,
                assistant_tokens,
                top,
            )
        )


if __name__ == "__main__":
    main()

