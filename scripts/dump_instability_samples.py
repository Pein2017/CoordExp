#!/root/miniconda3/envs/ms/bin/python
"""Dump offending JSONL records referenced by instability monitor events.

Usage:
  /root/miniconda3/envs/ms/bin/python scripts/dump_instability_samples.py \
    --events output/.../instability_dumps/events.jsonl \
    --train_jsonl public_data/.../train.coord.jsonl \
    --val_jsonl public_data/.../val.coord.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl_records_by_index(jsonl_path: str, indices: list[int]) -> dict[int, Any]:
    wanted = sorted(set(int(i) for i in indices if int(i) >= 0))
    if not wanted:
        return {}
    out: dict[int, Any] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        current = 0
        target_pos = 0
        target = wanted[target_pos]
        for line in f:
            if current == target:
                raw = line.strip("\n")
                try:
                    out[current] = json.loads(raw)
                except Exception:
                    out[current] = {"_raw": raw, "_parse_error": True}
                target_pos += 1
                if target_pos >= len(wanted):
                    break
                target = wanted[target_pos]
            current += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="Path to instability_dumps/events.jsonl")
    ap.add_argument("--train_jsonl", default=None, help="Training JSONL path (for mode=train)")
    ap.add_argument("--val_jsonl", default=None, help="Validation JSONL path (for mode=eval)")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: events dir)")
    args = ap.parse_args()

    events_path = Path(args.events)
    out_dir = Path(args.out_dir) if args.out_dir else events_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl

    with events_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            mode = event.get("mode")
            step = event.get("global_step", idx)
            meta_json = event.get("meta_json")
            if not isinstance(meta_json, str):
                continue
            try:
                meta = json.loads(meta_json)
            except Exception:
                continue
            if not isinstance(meta, list):
                continue

            base_idxs: list[int] = []
            for pack in meta:
                if not isinstance(pack, dict):
                    continue
                samples = pack.get("samples")
                if not isinstance(samples, list):
                    continue
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    bi = s.get("base_idx")
                    try:
                        bi_i = int(bi)
                    except Exception:
                        continue
                    if bi_i >= 0:
                        base_idxs.append(bi_i)

            if not base_idxs:
                continue

            jsonl_path = None
            if mode == "train" and isinstance(train_jsonl, str) and train_jsonl:
                jsonl_path = train_jsonl
            if mode != "train" and isinstance(val_jsonl, str) and val_jsonl:
                jsonl_path = val_jsonl
            if jsonl_path is None:
                continue

            records = _load_jsonl_records_by_index(jsonl_path, base_idxs)
            out = {
                "event": event,
                "jsonl_path": jsonl_path,
                "base_idxs": sorted(set(base_idxs)),
                "records": records,
            }
            out_path = out_dir / f"dump_event{idx}_step{step}_{mode}.json"
            out_path.write_text(json.dumps(out, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

