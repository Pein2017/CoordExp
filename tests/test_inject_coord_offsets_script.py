import json
import subprocess
import sys
from pathlib import Path

import safetensors.torch as st
import torch


def test_inject_coord_offsets_materializes_head_and_unties_config(tmp_path: Path) -> None:
    """Contract: an untied coord_offset adapter must materialize lm_head.weight when
    the merged export omitted it (typical when tie_word_embeddings=True).

    This ensures merged checkpoints preserve distinct embed vs head deltas.
    """

    repo_root = Path(__file__).resolve().parents[1]
    tool = repo_root / "scripts" / "tools" / "inject_coord_offsets.py"
    assert tool.is_file()

    merged_dir = tmp_path / "merged"
    adapter_dir = tmp_path / "adapter"
    merged_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Start from a "tied" export that omitted lm_head.weight.
    (merged_dir / "config.json").write_text(
        json.dumps({"tie_word_embeddings": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = merged_dir / shard_name

    vocab = 10
    hidden = 6
    base = torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden)
    st.save_file({"embed_tokens.weight": base}, str(shard_path))

    # Minimal index file so the injector can add lm_head.weight to weight_map.
    (merged_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed_tokens.weight": shard_name}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Untied adapter offsets (embed_offset + head_offset).
    coord_ids = torch.tensor([2, 5], dtype=torch.long)
    embed_offset = torch.full((2, hidden), 1.0, dtype=torch.float32)
    head_offset = torch.full((2, hidden), 2.0, dtype=torch.float32)
    st.save_file(
        {
            "base_model.model.coord_offset_adapter.coord_ids": coord_ids,
            "base_model.model.coord_offset_adapter.embed_offset": embed_offset,
            "base_model.model.coord_offset_adapter.head_offset": head_offset,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--merged_dir",
            str(merged_dir),
            "--adapter_dir",
            str(adapter_dir),
        ],
        check=True,
    )

    patched = st.load_file(str(shard_path))
    assert "embed_tokens.weight" in patched
    assert "lm_head.weight" in patched

    emb_patched = patched["embed_tokens.weight"]
    head_patched = patched["lm_head.weight"]

    assert torch.allclose(emb_patched[coord_ids], base[coord_ids] + embed_offset)
    assert torch.allclose(head_patched[coord_ids], base[coord_ids] + head_offset)

    cfg = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("tie_word_embeddings") is False

    index = json.loads(
        (merged_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index["weight_map"]["lm_head.weight"] == shard_name

