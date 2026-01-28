#!/usr/bin/env python
"""
Inject coord_offset embeddings/logits directly into merged safetensor shards
without loading the full model.

Usage:
  python scripts/tools/inject_coord_offsets.py \
      --merged_dir output/debug/coord_merged \
      --adapter_dir output/debug/coord/v0-20251203-054636/epoch_30-dlora-lrs_4_2_8-sft_base/checkpoint-6
"""

import argparse
import glob
import json
import os
from typing import Dict, Tuple

import safetensors.torch as st
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_dir", required=True, help="LoRA-merged model directory (safetensors shards).")
    p.add_argument("--adapter_dir", required=True, help="Adapter checkpoint containing coord_offset.")
    return p.parse_args()


def load_offsets(adapter_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ck = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"adapter_model.safetensors not found at {ck}")
    d = st.load_file(ck)
    coord_ids = d["base_model.model.coord_offset_adapter.coord_ids"].long()
    embed_offset = d["base_model.model.coord_offset_adapter.embed_offset"]
    head_offset = d.get("base_model.model.coord_offset_adapter.head_offset")
    return coord_ids, embed_offset, head_offset


def is_tied_embeddings(merged_dir: str) -> bool:
    """Return True if the merged checkpoint ties input embeddings and lm_head.

    Some Qwen-family checkpoints do not store `lm_head.weight` as a standalone
    tensor shard because it is tied to `embed_tokens.weight`. In that case, to
    bake `coord_offset.head_offset` into the exported checkpoint, we must apply
    it to `embed_tokens.weight` rows as well.
    """

    cfg_path = os.path.join(merged_dir, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False

    if bool(cfg.get("tie_word_embeddings", False)):
        return True
    text_cfg = cfg.get("text_config") or {}
    if isinstance(text_cfg, dict) and bool(text_cfg.get("tie_word_embeddings", False)):
        return True
    return False


def find_embedding_keys(shard_path: str):
    keys = []
    with st.safe_open(shard_path, framework="pt") as f:
        keys = list(f.keys())
    embed_keys = [k for k in keys if k.endswith("embed_tokens.weight")]
    head_keys = [k for k in keys if k.endswith("lm_head.weight")]
    return embed_keys, head_keys


def load_shard(shard_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    tensors = {}
    with st.safe_open(shard_path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def main():
    args = parse_args()

    coord_ids, embed_offset, head_offset = load_offsets(args.adapter_dir)
    adapter_tie_head = head_offset is None
    if adapter_tie_head:
        # Single/shared lookup table: the same offset should be applied to both embedding lookup
        # and output projection (tie-head semantics).
        head_offset = embed_offset

    shard_paths = sorted(glob.glob(os.path.join(args.merged_dir, "model-*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No model-*.safetensors found in {args.merged_dir}")

    embed_key = head_key = None
    embed_shard = head_shard = None
    for sp in shard_paths:
        e_keys, h_keys = find_embedding_keys(sp)
        if e_keys and embed_key is None:
            embed_key, embed_shard = e_keys[0], sp
        if h_keys and head_key is None:
            head_key, head_shard = h_keys[0], sp
        if embed_key and head_key:
            break

    if embed_key is None:
        raise ValueError("Could not find embed_tokens.weight in merged shards.")

    # If we need distinct head offsets but the export omitted lm_head.weight (typical when
    # tie_word_embeddings=True), materialize lm_head.weight so we can preserve training semantics.
    materialize_head = False
    if head_key is None and not adapter_tie_head:
        head_key = "lm_head.weight"
        head_shard = embed_shard
        materialize_head = True
        print(
            f"lm_head.weight not found; will materialize {head_key} into {os.path.basename(embed_shard)}"
        )
    elif head_key is None and adapter_tie_head:
        print(
            "lm_head.weight not found; adapter uses tie_head=True, so only embed_tokens.weight will be patched."
        )

    # Patch embedding shard
    tensors_e, meta_e = load_shard(embed_shard)
    emb_base = tensors_e[embed_key]
    emb = emb_base.clone()
    with torch.no_grad():
        emb[coord_ids] = emb[coord_ids] + embed_offset.to(emb.dtype)
    tensors_e[embed_key] = emb

    if head_key is not None and head_shard == embed_shard:
        # Create or patch lm_head in the same shard.
        if materialize_head:
            head_base = emb_base
        else:
            head_base = tensors_e[head_key]
        head_t = head_base.clone()
        with torch.no_grad():
            head_t[coord_ids] = head_t[coord_ids] + head_offset.to(head_t.dtype)
        tensors_e[head_key] = head_t

        st.save_file(tensors_e, embed_shard, metadata=meta_e)
        print(f"Patched embed_tokens and lm_head in {embed_shard}")
    else:
        st.save_file(tensors_e, embed_shard, metadata=meta_e)
        print(f"Patched embedding in {embed_shard}")

        if head_key is not None:
            tensors_h, meta_h = load_shard(head_shard)
            if head_key not in tensors_h:
                raise ValueError(f"Could not find {head_key} in {head_shard}")
            head_t = tensors_h[head_key].clone()
            with torch.no_grad():
                head_t[coord_ids] = head_t[coord_ids] + head_offset.to(head_t.dtype)
            tensors_h[head_key] = head_t
            st.save_file(tensors_h, head_shard, metadata=meta_h)
            print(f"Patched lm_head in {head_shard}")
        else:
            print("Skipped lm_head patch (not present and not materialized).")

    # If we materialized lm_head.weight, update the safetensors index so HF loaders can find it.
    if materialize_head:
        index_path = os.path.join(args.merged_dir, "model.safetensors.index.json")
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map") or {}
            shard_name = os.path.basename(embed_shard)
            if head_key not in weight_map:
                weight_map[head_key] = shard_name
                index["weight_map"] = weight_map
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index, f, ensure_ascii=False, indent=2)
                print(f"Updated index: mapped {head_key} -> {shard_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to update safetensors index at {index_path}: {e}")

    # If embedding and head differ (separate offsets), ensure the exported config does not re-tie them.
    if not adapter_tie_head:
        cfg_path = os.path.join(args.merged_dir, "config.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            changed = False
            if cfg.get("tie_word_embeddings") is True:
                cfg["tie_word_embeddings"] = False
                changed = True
            text_cfg = cfg.get("text_config")
            if isinstance(text_cfg, dict) and text_cfg.get("tie_word_embeddings") is True:
                text_cfg["tie_word_embeddings"] = False
                cfg["text_config"] = text_cfg
                changed = True
            if changed:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                print("Set tie_word_embeddings=False in merged config.json")
        except Exception as e:
            print(f"WARNING: Failed to update tie_word_embeddings in {cfg_path}: {e}")

    print("Coord offsets injected into merged shards.")


if __name__ == "__main__":
    main()
