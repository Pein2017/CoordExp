#!/usr/bin/env python
"""
Inject coord_offset embeddings/logits directly into merged safetensor shards
without loading the full model.

Usage:
  python scripts/inject_coord_offsets.py \
      --merged_dir output/debug/coord_merged \
      --adapter_dir output/debug/coord/v0-20251203-054636/epoch_30-dlora-lrs_4_2_8-sft_base/checkpoint-6
"""

import argparse
import glob
import os
from typing import Dict, Tuple

import safetensors.torch as st
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_dir", required=True, help="LoRA-merged model directory (safetensors shards).")
    p.add_argument("--adapter_dir", required=True, help="Adapter checkpoint containing coord_offset.")
    return p.parse_args()


def load_offsets(adapter_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ck = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"adapter_model.safetensors not found at {ck}")
    d = st.load_file(ck)
    coord_ids = d["base_model.model.coord_offset_adapter.coord_ids"].long()
    embed_offset = d["base_model.model.coord_offset_adapter.embed_offset"]
    head_offset = d["base_model.model.coord_offset_adapter.head_offset"]
    return coord_ids, embed_offset, head_offset


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
    dtype = embed_offset.dtype

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
    if head_key is None:
        print("WARNING: lm_head.weight not found; only embedding will be patched.")

    # Patch embedding shard
    tensors_e, meta_e = load_shard(embed_shard)
    emb = tensors_e[embed_key].to(dtype)
    with torch.no_grad():
        emb[coord_ids] += embed_offset.to(emb.dtype)
    tensors_e[embed_key] = emb

    if head_key is not None and head_shard == embed_shard:
        # Patch head in same shard
        head_t = tensors_e[head_key].to(dtype)
        with torch.no_grad():
            head_t[coord_ids] += head_offset.to(head_t.dtype)
        tensors_e[head_key] = head_t
        st.save_file(tensors_e, embed_shard, metadata=meta_e)
        print(f"Patched embedding and lm_head in {embed_shard}")
    else:
        # Save embedding shard first
        st.save_file(tensors_e, embed_shard, metadata=meta_e)
        print(f"Patched embedding in {embed_shard}")

        if head_key is not None:
            tensors_h, meta_h = load_shard(head_shard)
            head_t = tensors_h[head_key].to(dtype)
            with torch.no_grad():
                head_t[coord_ids] += head_offset.to(head_t.dtype)
            tensors_h[head_key] = head_t
            st.save_file(tensors_h, head_shard, metadata=meta_h)
            print(f"Patched lm_head in {head_shard}")
        else:
            print("Skipped lm_head patch (not found).")

    print("Coord offsets injected into merged shards.")


if __name__ == "__main__":
    main()
