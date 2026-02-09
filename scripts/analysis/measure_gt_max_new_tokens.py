"""Measure GT (assistant) token-length distribution for dense-caption JSONLs.

This is primarily used to size `max_new_tokens` for rollout decoding:
`max_new_tokens` must be >= the number of tokens needed to emit the full GT
assistant JSON for the worst-case sample (or for a chosen percentile).

Example:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python \
    scripts/measure_gt_max_new_tokens.py \
    --config .worktrees/CoordExp-wt-2026-01-15-rollout-matching/configs/base_rollout_matching_sft.yaml \
    --checkpoint model_cache/Qwen3-VL-8B-Instruct-coordexp

Notes:
- We measure the token length of the *assistant JSON text only* (no system/user
  prompt tokens). In generation, you typically need +1 token margin for EOS.
- For coord-token JSONLs, use a coordexp-expanded tokenizer/checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from transformers import AutoTokenizer

from src.config.loader import ConfigLoader
from src.coord_tokens.validator import annotate_coord_tokens
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import sort_objects_by_topleft

_LVIS_DEFAULT_ROOT = "public_data/lvis/rescale_32_768_poly_20"


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    return len(v) >= 3 and v.startswith("<") and v.endswith(">")


def _pick_default_jsonl(*, split: str, coord_tokens_enabled: bool) -> str:
    suffix = ".coord.jsonl" if coord_tokens_enabled else ".jsonl"
    return os.path.join(_LVIS_DEFAULT_ROOT, f"{split}{suffix}")


def _stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse JSONL line {line_no} in {path!r}: {exc}"
                ) from exc


def _percentiles(values: np.ndarray, ps: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in ps:
        key = f"p{p:g}"
        out[key] = float(np.percentile(values, p))
    return out


@dataclass
class SplitStats:
    name: str
    n: int
    lengths: np.ndarray
    objects_per_sample: np.ndarray
    topk: List[Dict[str, Any]]

    def summarize(self) -> Dict[str, Any]:
        v = self.lengths
        o = self.objects_per_sample
        return {
            "name": self.name,
            "n": int(self.n),
            "assistant_tokens": {
                "min": int(v.min()) if self.n else None,
                "max": int(v.max()) if self.n else None,
                "mean": float(v.mean()) if self.n else None,
                "std": float(v.std(ddof=0)) if self.n else None,
                "percentiles": _percentiles(v, [50, 90, 95, 99, 99.5, 99.9])
                if self.n
                else {},
            },
            "objects_per_sample": {
                "min": int(o.min()) if self.n else None,
                "max": int(o.max()) if self.n else None,
                "mean": float(o.mean()) if self.n else None,
                "percentiles": _percentiles(o, [50, 90, 95, 99]) if self.n else {},
            },
            "topk_by_assistant_tokens": list(self.topk),
        }


def _measure_split(
    *,
    name: str,
    jsonl_path: str,
    builder: JSONLinesBuilder,
    tokenizer: Any,
    coord_tokens_enabled: bool,
    batch_size: int,
    limit: Optional[int],
    verbose_every: int,
    topk: int,
) -> SplitStats:
    lengths: List[int] = []
    objs_per: List[int] = []

    texts: List[str] = []
    pending_counts: List[int] = []
    pending_meta: List[Dict[str, Any]] = []
    # min-heap of (token_len, tiebreaker, meta) to avoid dict comparisons on ties
    heap: List[tuple[int, int, Dict[str, Any]]] = []

    def flush() -> None:
        nonlocal texts, pending_counts, pending_meta, heap
        if not texts:
            return
        enc = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        input_ids = enc["input_ids"]
        for ids, obj_n, meta in zip(input_ids, pending_counts, pending_meta):
            lengths.append(len(ids))
            objs_per.append(obj_n)
            if topk > 0:
                item = (len(ids), int(meta.get("line") or 0), meta)
                if len(heap) < topk:
                    import heapq

                    heapq.heappush(heap, item)
                else:
                    import heapq

                    if item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
        texts = []
        pending_counts = []
        pending_meta = []

    for idx, record in enumerate(_stream_jsonl(jsonl_path), start=1):
        if limit is not None and idx > limit:
            break

        objects = record.get("objects") or []
        if isinstance(objects, list) and objects:
            record["objects"] = sort_objects_by_topleft(objects)

        if coord_tokens_enabled:
            # Mirrors training: attach _coord_tokens/_coord_token_ints so the builder
            # always emits canonical "<|coord_k|>" strings.
            annotate_coord_tokens(record)  # validates width/height too

        merged = builder.build_many([record])
        try:
            assistant_text = merged["messages"][1]["content"][0]["text"]
        except Exception as exc:
            raise KeyError(
                f"Unexpected builder output structure for {name} sample {idx} from {jsonl_path}: {exc}"
            ) from exc

        texts.append(assistant_text)
        objects_for_meta = record.get("objects") or []
        pending_counts.append(len(objects_for_meta))

        # Minimal metadata for debugging extreme outliers.
        max_poly_points = 0
        polys = 0
        bboxes = 0
        if isinstance(objects_for_meta, list):
            for obj in objects_for_meta:
                if not isinstance(obj, dict):
                    continue
                if "poly" in obj:
                    polys += 1
                if "bbox_2d" in obj:
                    bboxes += 1
                pp = obj.get("poly_points")
                if pp is not None:
                    try:
                        max_poly_points = max(max_poly_points, int(pp))
                    except Exception:
                        pass
        images = record.get("images") or []
        img0 = images[0] if isinstance(images, list) and images else None
        pending_meta.append(
            {
                "line": idx,
                "image": img0,
                "objects": len(objects_for_meta)
                if isinstance(objects_for_meta, list)
                else None,
                "polys": polys,
                "bboxes": bboxes,
                "max_poly_points": max_poly_points,
            }
        )

        if len(texts) >= batch_size:
            flush()

        if verbose_every > 0 and idx % verbose_every == 0:
            # Lightweight progress print (avoid tqdm dependency).
            cur_max = max(lengths) if lengths else None
            print(f"[{name}] processed={idx} cur_max_assistant_tokens={cur_max}")

    flush()

    arr = np.asarray(lengths, dtype=np.int32)
    obj_arr = np.asarray(objs_per, dtype=np.int32)
    topk_sorted: List[Dict[str, Any]] = []
    if topk > 0 and heap:
        topk_sorted = [m for _, _, m in sorted(heap, key=lambda x: x[0], reverse=True)]
    return SplitStats(
        name=name,
        n=int(arr.shape[0]),
        lengths=arr,
        objects_per_sample=obj_arr,
        topk=topk_sorted,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path (used to infer coord_tokens + default train/val jsonl).",
    )
    ap.add_argument("--train_jsonl", type=str, default=None, help="Override train JSONL path.")
    ap.add_argument("--val_jsonl", type=str, default=None, help="Override val JSONL path.")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Tokenizer checkpoint/path (AutoTokenizer.from_pretrained).",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples per split (for quick smoke).",
    )
    ap.add_argument(
        "--verbose_every",
        type=int,
        default=0,
        help="Print progress every N samples (0 disables).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Also report the top-K longest GT assistant JSON samples (0 disables).",
    )
    ap.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary (stats + suggested max_new_tokens).",
    )
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            cfg = ConfigLoader.load_yaml_with_extends(args.config)
        except FileNotFoundError as exc:
            # Some templates in worktrees use an `extends:` path that is intended
            # for repo-root configs (e.g. `extends: configs/base.yaml`), which is
            # not resolvable relative to the worktree config folder.
            print(
                f"WARNING: failed to resolve config inheritance for {args.config!r}: {exc}\n"
                "         Falling back to direct YAML load (no extends)."
            )
            cfg = ConfigLoader.load_yaml(args.config) or {}

    custom = cfg.get("custom") if isinstance(cfg, dict) else None
    if not isinstance(custom, dict):
        custom = {}

    coord_tokens_cfg = custom.get("coord_tokens")
    coord_tokens_enabled = False
    if isinstance(coord_tokens_cfg, dict):
        coord_tokens_enabled = bool(coord_tokens_cfg.get("enabled", False))

    train_jsonl = args.train_jsonl or custom.get("train_jsonl")
    val_jsonl = args.val_jsonl or custom.get("val_jsonl")

    if not train_jsonl or _is_placeholder(train_jsonl):
        train_jsonl = _pick_default_jsonl(
            split="train", coord_tokens_enabled=coord_tokens_enabled
        )
    if not val_jsonl or _is_placeholder(val_jsonl):
        val_jsonl = _pick_default_jsonl(
            split="val", coord_tokens_enabled=coord_tokens_enabled
        )

    # If no YAML config is provided (or it doesn't specify coord_tokens),
    # infer coord-token mode from filename conventions.
    if not coord_tokens_enabled:
        for p in (train_jsonl, val_jsonl):
            if isinstance(p, str):
                name = p.lower()
                if ".coord." in name or name.endswith(".coord.jsonl"):
                    coord_tokens_enabled = True
                    break

    model_section = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model_section, dict):
        model_section = {}
    checkpoint = args.checkpoint or model_section.get("model")
    if not checkpoint or _is_placeholder(checkpoint):
        # Sensible default for coord-token runs in this repo.
        checkpoint = (
            "model_cache/Qwen3-VL-8B-Instruct-coordexp"
            if coord_tokens_enabled
            else "model_cache/models/Qwen/Qwen3-VL-8B-Instruct"
        )

    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"train_jsonl not found: {train_jsonl}")
    if not os.path.exists(val_jsonl):
        raise FileNotFoundError(f"val_jsonl not found: {val_jsonl}")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    print("Config")
    print(f"- coord_tokens_enabled: {coord_tokens_enabled}")
    print(f"- checkpoint: {checkpoint}")
    print(f"- train_jsonl: {train_jsonl}")
    print(f"- val_jsonl: {val_jsonl}")
    print(f"- batch_size: {args.batch_size}")
    if args.limit:
        print(f"- limit: {args.limit}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # user_prompt does not affect the assistant JSON; keep it stable anyway.
    builder = JSONLinesBuilder(
        user_prompt="Please describe all objects in the image.",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=coord_tokens_enabled,
    )

    train_stats = _measure_split(
        name="train",
        jsonl_path=train_jsonl,
        builder=builder,
        tokenizer=tokenizer,
        coord_tokens_enabled=coord_tokens_enabled,
        batch_size=int(args.batch_size),
        limit=args.limit,
        verbose_every=int(args.verbose_every),
        topk=int(args.topk),
    )
    val_stats = _measure_split(
        name="val",
        jsonl_path=val_jsonl,
        builder=builder,
        tokenizer=tokenizer,
        coord_tokens_enabled=coord_tokens_enabled,
        batch_size=int(args.batch_size),
        limit=args.limit,
        verbose_every=int(args.verbose_every),
        topk=int(args.topk),
    )

    summary = {
        "config": {
            "coord_tokens_enabled": coord_tokens_enabled,
            "checkpoint": checkpoint,
            "train_jsonl": train_jsonl,
            "val_jsonl": val_jsonl,
        },
        "train": train_stats.summarize(),
        "val": val_stats.summarize(),
    }

    # Practical suggestions for rollout decoding.
    worst = int(max(train_stats.lengths.max(), val_stats.lengths.max()))
    p999 = float(
        max(
            np.percentile(train_stats.lengths, 99.9),
            np.percentile(val_stats.lengths, 99.9),
        )
    )
    suggested = int(math.ceil(p999 + 1))
    summary["suggested_max_new_tokens"] = {
        "p99_9_plus_eos": suggested,
        "worst_case_plus_eos": int(worst + 1),
        "note": "Lengths measure assistant JSON only; add ~1 token for EOS and optional safety margin.",
    }

    print("Results")
    for split_name, stats in (("train", train_stats), ("val", val_stats)):
        s = stats.summarize()
        toks = s["assistant_tokens"]
        objs = s["objects_per_sample"]
        print(
            f"- {split_name}: n={s['n']} assistant_tokens[min={toks['min']}, p99={toks['percentiles'].get('p99')}, p99.9={toks['percentiles'].get('p99.9')}, max={toks['max']}] objects[min={objs['min']}, p99={objs['percentiles'].get('p99')}, max={objs['max']}]"
        )

    print("")
    print("Suggested max_new_tokens")
    print(f"- p99.9 + eos: {summary['suggested_max_new_tokens']['p99_9_plus_eos']}")
    print(f"- worst + eos: {summary['suggested_max_new_tokens']['worst_case_plus_eos']}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Wrote: {args.save_json}")


if __name__ == "__main__":
    main()
