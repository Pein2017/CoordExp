import os
import subprocess
import sys
import json
import textwrap
from pathlib import Path

import pytest
import torch

from src.config.loader import ConfigLoader
from src.config.schema import CoordTokensConfig
from src.config.schema import TrainingConfig
from src.datasets.fusion import FusionConfig
from src.datasets.unified_fusion_dataset import FusionCaptionDataset
from src.datasets.wrappers.packed_caption import build_packed_dataset


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        # Used only in debug paths; keep minimal.
        return "<decoded>"


class _FakeTemplate:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.system = "SYSTEM"
        self.tokenizer = _FakeTokenizer()
        self.packing = False
        self.padding_free = False

    def encode(self, merged, return_length=False):
        # Minimal stub: return a deterministic length based on assistant text.
        messages = merged.get("messages", []) if isinstance(merged, dict) else []
        assistant_text = ""
        for turn in messages:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                for item in turn.get("content", []) or []:
                    if isinstance(item, dict) and item.get("type") == "text":
                        assistant_text = str(item.get("text") or "")
                        break
        length = max(4, min(64, len(assistant_text)))
        sample = {
            "input_ids": [0] * length,
            "labels": [0] * length,
            "length": length,
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }
        return sample


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def test_fusion_config_parses_targets_and_sources_and_preserves_order(tmp_path: Path):
    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: a
                train_jsonl: a.jsonl
                template: bbox_only
            sources:
              - dataset: jsonl
                name: b
                train_jsonl: b.jsonl
                template: bbox_only
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(cfg_path))
    assert [spec.name for spec in cfg.datasets] == ["a", "b"]


def test_fusion_extends_merges_by_dataset_id(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: vg
                train_jsonl: vg_train.jsonl
                val_jsonl: null
                template: bbox_only
                ratio: 0.2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        textwrap.dedent(
            f"""
            extends: {base_path.name}
            targets:
              - name: vg
                ratio: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(override_path))
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].name == "vg"
    assert cfg.datasets[0].ratio == 1.0


def test_fusion_extends_diamond_is_not_a_cycle(tmp_path: Path):
    # A extends B and C; both extend D. This is a DAG (diamond), not a cycle.
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "train.jsonl").write_text(
        json.dumps({"images": [], "width": 1, "height": 1, "objects": []}) + "\n",
        encoding="utf-8",
    )

    d_path = base_dir / "d.yaml"
    d_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: toy
                train_jsonl: ./train.jsonl
                template: bbox_only
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    b_path = base_dir / "b.yaml"
    b_path.write_text(
        textwrap.dedent(
            f"""
            extends: {d_path.name}
            targets:
              - name: toy
                ratio: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    c_path = base_dir / "c.yaml"
    c_path.write_text(
        textwrap.dedent(
            f"""
            extends: {d_path.name}
            targets:
              - name: toy
                ratio: 0.5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    a_path = base_dir / "a.yaml"
    a_path.write_text(
        textwrap.dedent(
            f"""
            extends: [{b_path.name}, {c_path.name}]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(a_path))
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].name == "toy"


def test_fusion_duplicate_dataset_ids_error(tmp_path: Path):
    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: dup
                train_jsonl: a.jsonl
                template: bbox_only
              - dataset: jsonl
                name: dup
                train_jsonl: b.jsonl
                template: bbox_only
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _ = FusionConfig.from_file(str(cfg_path))


def test_fusion_unknown_template_errors(tmp_path: Path):
    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: a
                train_jsonl: a.jsonl
                template: some_unknown_template
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _ = FusionConfig.from_file(str(cfg_path))


def test_fusion_resolves_dot_paths_relative_to_config_dir(tmp_path: Path):
    cfg_dir = tmp_path / "cfgs" / "nested"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    train_path = cfg_dir / "train.jsonl"
    train_path.write_text(
        json.dumps({"images": [], "width": 1, "height": 1, "objects": []}) + "\n",
        encoding="utf-8",
    )

    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: toy
                train_jsonl: ./train.jsonl
                template: bbox_only
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    override_cfg = tmp_path / "override.yaml"
    override_cfg.write_text(
        textwrap.dedent(
            f"""
            extends: {base_cfg.relative_to(tmp_path)}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(override_cfg))
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].train_jsonl == train_path.resolve()


def test_fusion_val_jsonl_null_skips_eval(tmp_path: Path):
    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: a
                train_jsonl: a.jsonl
                val_jsonl: null
                template: bbox_only
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(cfg_path))
    assert cfg.datasets[0].val_jsonl is None


def test_fusion_dataset_smoke_iterates_and_packs(tmp_path: Path):
    # Minimal valid dense-caption record (coords must be in [0, 999] when coord tokens are disabled).
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"\x00")

    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"

    record = {
        "images": [str(img_path)],
        "width": 1000,
        "height": 1000,
        "objects": [
            {"bbox_2d": [0, 0, 10, 10], "desc": "cat"},
            {"bbox_2d": [20, 20, 30, 30], "desc": "dog"},
        ],
    }
    _write_jsonl(train_jsonl, [record, record])
    _write_jsonl(val_jsonl, [record])

    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            targets:
              - dataset: jsonl
                name: toy
                train_jsonl: {train_jsonl}
                val_jsonl: {val_jsonl}
                template: bbox_only
                ratio: 1.0
            """.format(train_jsonl=str(train_jsonl), val_jsonl=str(val_jsonl))
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = FusionConfig.from_file(str(cfg_path))
    template = _FakeTemplate(max_length=64)

    ds_train = FusionCaptionDataset(
        fusion_config=cfg,
        base_template=template,
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense="SYSTEM",
        system_prompt_summary=None,
        coord_tokens=CoordTokensConfig(enabled=False),
        seed=123,
        shuffle=True,
        sample_limit=None,
        split="train",
        object_ordering="sorted",
    )

    assert len(ds_train) == 2
    sample = ds_train[0]
    assert "input_ids" in sample
    assert "messages" in sample
    assert sample.get("dataset") == "toy"

    # Eval split should include val_jsonl records.
    ds_eval = FusionCaptionDataset(
        fusion_config=cfg,
        base_template=template,
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense="SYSTEM",
        system_prompt_summary=None,
        coord_tokens=CoordTokensConfig(enabled=False),
        seed=123,
        shuffle=False,
        sample_limit=None,
        split="eval",
        object_ordering="sorted",
    )
    assert len(ds_eval) == 1

    # Packing wrapper should be able to iterate a few packs without fusion-specific errors.
    packed = build_packed_dataset(
        ds_train,
        template=template,
        packing_length=64,
        buffer_size=4,
        min_fill_ratio=0.1,
        drop_last=False,
        allow_single_long=True,
    )
    packs = list(packed)
    assert packs


def test_training_config_materializes_with_fusion_config_without_train_jsonl(
    tmp_path: Path,
):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            template:
              template: qwen3_vl
            custom:
              fusion_config: some/fusion.yaml
              emit_norm: none
              json_format: standard
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    raw = ConfigLoader.load_yaml_with_extends(str(cfg_path))
    prompts = ConfigLoader.resolve_prompts(raw)
    training = TrainingConfig.from_mapping(raw, prompts)
    assert training.custom.fusion_config == "some/fusion.yaml"


def test_offline_merge_is_reproducible_across_python_hash_seeds(tmp_path: Path):
    # This should not depend on PYTHONHASHSEED; we use a stable hash function.
    train_jsonl = tmp_path / "train.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            {"images": ["a.jpg"], "width": 1, "height": 1, "objects": []},
            {"images": ["b.jpg"], "width": 1, "height": 1, "objects": []},
        ],
    )

    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            targets:
              - dataset: jsonl
                name: toy
                train_jsonl: {train_jsonl}
                template: bbox_only
                ratio: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    script = textwrap.dedent(
        f"""
        import hashlib
        from pathlib import Path
        from src.datasets.fusion import FusionConfig, build_fused_jsonl

        cfg = FusionConfig.from_file({str(cfg_path)!r})
        out = Path({str(tmp_path / "out.jsonl")!r})
        build_fused_jsonl(cfg, str(out), seed=2025, shuffle=False)
        print(hashlib.md5(out.read_bytes()).hexdigest())
        """
    ).strip()

    env1 = dict(os.environ)
    env1["PYTHONHASHSEED"] = "1"
    env2 = dict(os.environ)
    env2["PYTHONHASHSEED"] = "2"

    digest1 = subprocess.check_output(
        [sys.executable, "-c", script],
        env=env1,
        cwd=str(Path(__file__).resolve().parents[1]),
    ).decode("utf-8").strip()
    digest2 = subprocess.check_output(
        [sys.executable, "-c", script],
        env=env2,
        cwd=str(Path(__file__).resolve().parents[1]),
    ).decode("utf-8").strip()

    assert digest1 == digest2
