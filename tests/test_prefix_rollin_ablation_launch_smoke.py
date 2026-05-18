from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from src.bootstrap.experiment_manifest import (
    build_experiment_manifest_payload,
    write_experiment_manifest_file,
)
from src.bootstrap.run_metadata import (
    build_run_metadata_payload,
    write_run_metadata_file_from_payload,
)
from src.config.loader import ConfigLoader
from src.config.schema import DetectionDataConfig, LatestDetectionTrainingConfig
from src.detection.runtime import (
    assert_latest_detection_runtime_supported,
    build_latest_detection_dataset,
    build_latest_detection_runtime_custom_shim,
    resolve_latest_detection_prompts,
    resolve_recursive_detection_ce_runtime_cfg,
)
from src.sft import (
    EncodedSampleCacheRuntimeConfig,
    PackingRuntimeConfig,
    _apply_checkpoint_mode,
    _build_effective_runtime_payload,
    _parse_packing_config,
)
from src.trainers.batch_extras import RECURSIVE_DETECTION_TARGETS_KEY
from src.trainers.metrics.mixins import RecursiveDetectionCEMixin
from src.utils.run_manifest import write_run_manifest_files


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    padding_side = "right"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|object_ref_start|>": 4,
            "<|box_start|>": 5,
        }
        self.eos_token_id = self._token_to_id[self.eos_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {"im_start": "<|im_start|>", "im_end": "<|im_end|>"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(
            self(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["input_ids"]
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{self._content_text(message['content'])}<|im_end|>\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        return list(
            self(
                rendered,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["input_ids"]
        )

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            match = _SPECIAL_TOKEN_RE.match(text, cursor)
            if match is not None:
                token_text = match.group(0)
                token_end = match.end()
            else:
                token_text = text[cursor]
                token_end = cursor + 1
            token_id = self._token_to_id.setdefault(
                token_text,
                len(self._token_to_id) + 1,
            )
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def _content_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        assert isinstance(content, list)
        parts: list[str] = []
        for item in content:
            assert isinstance(item, dict)
            if item.get("type") == "image":
                parts.append("<image>")
            elif item.get("type") == "text":
                parts.append(str(item.get("text")))
        return "".join(parts)


class FakeSwiftTemplate:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        assert return_length is True
        messages = [dict(message) for message in payload["messages"]]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        labels = [-100 for _ in input_ids]

        assistant_text = _assistant_text(messages)
        assistant_start = text.find(assistant_text)
        assert assistant_start >= 0
        assistant_end = assistant_start + len(assistant_text)
        stop_end = assistant_end
        if text.startswith("<|im_end|>", assistant_end):
            stop_end = assistant_end + len("<|im_end|>")

        for index, (start, end) in enumerate(offsets):
            if assistant_start <= start and end <= stop_end:
                labels[index] = input_ids[index]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1 for _ in input_ids],
            "length": len(input_ids),
        }


class _Metric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _DummyModel:
    training = True

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.forward_inputs: dict[str, object] | None = None

    def __call__(self, **inputs):
        self.forward_inputs = dict(inputs)
        return SimpleNamespace(logits=self.logits)


class _BaseTrainer:
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        raise AssertionError("recursive CE must own the model forward and loss")


class _Trainer(RecursiveDetectionCEMixin, _BaseTrainer):
    def __init__(self, cfg: object) -> None:
        self.recursive_detection_ce_cfg = cfg
        self.model = _DummyModel(torch.zeros((1, 1, 2), dtype=torch.float32))
        self.custom_metrics = {
            "train": defaultdict(_Metric),
            "eval": defaultdict(_Metric),
        }


def _assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    content = assistant_messages[0]["content"]
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    text_parts = [
        str(item["text"])
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    return "".join(text_parts)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ensure_image_root(tmp_path: Path) -> None:
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")


def _raw_row() -> dict[str, Any]:
    return {
        "images": ["images/train2017/example.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
            },
            {
                "bbox_2d": [
                    "<|coord_50|>",
                    "<|coord_60|>",
                    "<|coord_70|>",
                    "<|coord_80|>",
                ],
                "desc": "dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
            },
            {
                "bbox_2d": [
                    "<|coord_90|>",
                    "<|coord_100|>",
                    "<|coord_110|>",
                    "<|coord_120|>",
                ],
                "desc": "bus",
                "category_id": 6,
                "category_name": "bus",
                "coco_ann_id": 103,
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def _metric_values(trainer: _Trainer) -> dict[str, list[float]]:
    return {
        key: metric.values
        for key, metric in trainer.custom_metrics["train"].items()
        if metric.values
    }


def _build_logits_for_targets(
    *,
    input_ids: Sequence[int],
    target_payload: object,
) -> torch.Tensor:
    token_ids = {int(token_id) for token_id in input_ids}
    for target in target_payload.token_targets:
        token_ids.add(int(target.teacher_token_id))
        token_ids.update(int(token_id) for token_id in target.valid_token_ids)
        token_ids.update(int(token_id) for token_id in target.type_gate_token_ids)
        for branch_target in target.trie_branch_targets:
            token_ids.add(int(branch_target.token_id))
    vocab_size = max(token_ids) + 1
    logits = torch.zeros((1, len(input_ids), vocab_size), dtype=torch.float32)
    for target in target_payload.token_targets:
        logits[0, int(target.position) - 1, int(target.teacher_token_id)] = 3.0
        for token_id in target.valid_token_ids:
            logits[0, int(target.position) - 1, int(token_id)] = 2.5
        for branch_target in target.trie_branch_targets:
            logits[0, int(target.position) - 1, int(branch_target.token_id)] = 2.5
    return logits.requires_grad_()


def test_prefix_rollin_ablation_launch_smoke_covers_config_dataset_loss_and_manifests(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/ablation/"
        "compact_full_prefix_rollin_balance2.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))
    assert isinstance(cfg, LatestDetectionTrainingConfig)

    train_jsonl = tmp_path / "data" / "train.coord.jsonl"
    val_jsonl = tmp_path / "data" / "val.coord.jsonl"
    _write_jsonl(train_jsonl, [_raw_row()])
    _write_jsonl(val_jsonl, [{**_raw_row(), "metadata": {"source": "unit", "split": "val"}}])
    _ensure_image_root(tmp_path)
    cfg = replace(
        cfg,
        data=DetectionDataConfig(
            train_jsonl=str(train_jsonl),
            val_jsonl=str(val_jsonl),
            image_root=str(tmp_path / "image-root"),
            object_ordering=cfg.data.object_ordering,
        ),
        training={
            **dict(cfg.training),
            "output_root": str(tmp_path / "out"),
            "logging_root": str(tmp_path / "tb"),
            "artifact_subdir": "unit-prefix-rollin-smoke",
            "run_name": "unit-prefix-rollin-smoke",
            "num_train_epochs": 1,
            "max_steps": 1,
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_steps": 1,
        },
    )

    assert cfg.experiment is not None
    assert cfg.experiment.surface == "ablation"
    assert cfg.experiment.ablation_id == "E1"
    assert cfg.experiment.claim_scope == "none"
    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.objective.rollin.k_distribution.min_k == 0
    assert cfg.objective.rollin.k_distribution.max_k == "object_count"
    assert cfg.objective.target.support_weight == pytest.approx(1.0)
    assert cfg.objective.target.balance_weight == pytest.approx(2.0)
    assert cfg.training["packing"] is False
    assert cfg.training["eval_packing"] is False
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.packing.static_packing is False
    assert cfg.packing.padding_free_packed is False

    swift_template = FakeSwiftTemplate()
    encoded_cache_cfg = EncodedSampleCacheRuntimeConfig(enabled=False)
    assert_latest_detection_runtime_supported(
        cfg,
        encoded_sample_cache_cfg=encoded_cache_cfg,
        tokenizer=swift_template.tokenizer,
    )
    system_prompt, _user_prompt = resolve_latest_detection_prompts(cfg)
    custom_config = build_latest_detection_runtime_custom_shim(cfg)
    dataset = build_latest_detection_dataset(
        train_jsonl,
        swift_template=swift_template,
        training_config=cfg,
        custom_config=custom_config,
        system_prompt=system_prompt,
        seed=123,
        sample_limit=1,
        dataset_name="unit-train",
    )
    dataset.set_epoch(3)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["detection_metadata"]["template_id"] == "compact_full"
    assert sample["detection_metadata"]["mode"] == "prefix_rollin_et_rmp_ce"
    assert sample["detection_metadata"]["rollin_k"] == 1
    assert sample["detection_metadata"]["rollin_prefix_token_count"] > 0
    assert sample["detection_metadata"]["supervised_suffix_token_count"] > 0
    supervised_positions = tuple(
        index for index, label in enumerate(sample["labels"]) if label != -100
    )
    target_positions = tuple(
        target.position
        for target in sample["recursive_detection_targets"].token_targets
    )
    assert target_positions == supervised_positions
    im_end_id = swift_template.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_targets = [
        target
        for target in sample["recursive_detection_targets"].token_targets
        if target.teacher_token_id == im_end_id
    ]
    assert eos_targets
    assert eos_targets[-1].loss_weight == pytest.approx(1.0)

    recursive_cfg = resolve_recursive_detection_ce_runtime_cfg(cfg)
    assert recursive_cfg is not None
    assert recursive_cfg.enabled is True
    assert recursive_cfg.variant == "prefix_rollin_et_rmp_ce"
    assert recursive_cfg.trie_support_weight == pytest.approx(1.0)
    assert recursive_cfg.trie_balance_weight == pytest.approx(2.0)

    targets = sample["recursive_detection_targets"]
    logits = _build_logits_for_targets(input_ids=sample["input_ids"], target_payload=targets)
    model = _DummyModel(logits)
    trainer = _Trainer(recursive_cfg)
    loss, outputs = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([sample["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([sample["attention_mask"]], dtype=torch.long),
            "labels": torch.tensor([sample["labels"]], dtype=torch.long),
            RECURSIVE_DETECTION_TARGETS_KEY: (targets,),
            "detection_metadata": (sample["detection_metadata"],),
        },
        return_outputs=True,
    )

    assert torch.isfinite(loss)
    assert outputs.logits is logits
    assert model.forward_inputs is not None
    assert RECURSIVE_DETECTION_TARGETS_KEY not in model.forward_inputs
    assert "detection_metadata" not in model.forward_inputs
    assert "labels" not in model.forward_inputs
    metrics = _metric_values(trainer)
    assert metrics["loss/recursive_detection_ce"][-1] == pytest.approx(loss.item())
    assert metrics["recursive_detection_ce/batch_size"][-1] == pytest.approx(1.0)
    assert metrics["recursive_detection_ce/trie_support_weight"][-1] == pytest.approx(1.0)
    assert metrics["recursive_detection_ce/trie_balance_weight"][-1] == pytest.approx(2.0)
    assert "recursive_detection_ce/type_gate_loss" in metrics
    assert "detection_sequence/objective/recursive_detection_ce/loss_per_sample" in metrics
    assert "detection_sequence/objective/recursive_detection_ce/batch_size" in metrics
    assert "batch_loss" not in metrics
    assert "batch_size" not in metrics
    loss.backward()
    assert logits.grad is not None

    output_dir = tmp_path / "run-artifacts"
    train_args = SimpleNamespace(
        run_name=cfg.training["run_name"],
        output_dir=str(output_dir),
        logging_dir=str(tmp_path / "tb" / cfg.training["run_name"]),
        save_only_model=False,
        save_strategy="no",
        save_last_epoch=True,
        seed=123,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_train_epochs=1.0,
        dataloader_drop_last=False,
        deepspeed=None,
        resume_from_checkpoint=None,
    )
    _apply_checkpoint_mode(train_args, checkpoint_mode="artifact_only")
    packing_cfg = PackingRuntimeConfig(enabled=False, eval_packing=False)
    effective_runtime = _build_effective_runtime_payload(
        training_config=cfg,
        train_args=train_args,
        trainer_variant=None,
        dataset_seed=123,
        checkpoint_mode="artifact_only",
        packing_cfg=packing_cfg,
        encoded_sample_cache_cfg=encoded_cache_cfg,
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        pipeline_manifest=None,
    )
    written = write_run_manifest_files(
        output_dir=output_dir,
        training_config=cfg,
        config_path=str(cfg_path),
        base_config_path=None,
        dataset_seed=123,
        effective_runtime=effective_runtime,
        pipeline_manifest=None,
        train_data_provenance={"dataset_jsonl": str(train_jsonl), "sample_count": 1},
        eval_data_provenance={"dataset_jsonl": str(val_jsonl), "sample_count": 1},
    )
    run_metadata_payload = build_run_metadata_payload(
        output_dir=output_dir,
        config_path=str(cfg_path),
        base_config_path=None,
        run_name=str(cfg.training["run_name"]),
        dataset_seed=123,
        repo_root=repo_root,
        manifest_files=written,
        train_cache_info=None,
        eval_cache_info=None,
    )
    write_run_metadata_file_from_payload(
        output_dir=output_dir,
        payload=run_metadata_payload,
    )
    write_experiment_manifest_file(
        output_dir=output_dir,
        config_path=str(cfg_path),
        base_config_path=None,
        run_name=str(cfg.training["run_name"]),
        dataset_seed=123,
        experiment=cfg.to_mapping()["experiment"],
        effective_runtime=effective_runtime,
        pipeline_manifest=None,
        run_metadata=run_metadata_payload,
        manifest_files=written,
    )

    assert (output_dir / "resolved_config.json").is_file()
    assert (output_dir / "runtime_env.json").is_file()
    assert (output_dir / "effective_runtime.json").is_file()
    assert (output_dir / "train_data_provenance.json").is_file()
    assert (output_dir / "eval_data_provenance.json").is_file()
    assert (output_dir / "config_source.yaml").is_file()
    assert (output_dir / "run_metadata.json").is_file()
    assert (output_dir / "experiment_manifest.json").is_file()
    assert not (output_dir / "pipeline_manifest.json").exists()

    resolved = json.loads((output_dir / "resolved_config.json").read_text("utf-8"))
    assert resolved["resolved"]["objective"]["variant"] == "prefix_rollin_et_rmp_ce"
    assert resolved["resolved"]["objective"]["target"]["support_weight"] == pytest.approx(1.0)
    assert resolved["resolved"]["objective"]["target"]["balance_weight"] == pytest.approx(2.0)
    assert resolved["resolved"]["experiment"]["surface"] == "ablation"
    assert resolved["resolved"]["experiment"]["ablation_id"] == "E1"
    assert resolved["resolved"]["experiment"]["claim_scope"] == "none"

    effective = json.loads((output_dir / "effective_runtime.json").read_text("utf-8"))
    assert effective["runtime"]["checkpoint_mode"] == "artifact_only"
    assert effective["runtime"]["save_model_only"] is False
    assert effective["runtime"]["hf_save_only_model"] is True
    assert effective["runtime"]["max_steps"] == 1
    assert effective["runtime"]["effective_batch_size"] == 1
    assert effective["runtime"]["effective_batch_size_source"] == (
        "training.effective_batch_size"
    )
    assert effective["runtime"]["actual_global_effective_batch_size"] == 1
    assert effective["runtime"]["effective_batch_rounding"] == "exact"
    assert effective["runtime"]["world_size"] == 1
    assert effective["runtime"]["latest_detection_objective"] == {
        "id": "recursive_detection_ce",
        "variant": "prefix_rollin_et_rmp_ce",
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "template_id": "compact_full",
        "coordinate_surface": "coord_token",
        "bbox_format": "xyxy",
        "target_type": "entry_trie_support_balance",
        "support_weight": 1.0,
        "balance_weight": 2.0,
        "rollin_source": "ground_truth",
        "rollin_k_distribution": "uniform_inclusive",
        "type_gate_mode": "allowed_type_mass",
    }
    assert effective["runtime"]["model_source"]["raw_path"].endswith(
        "Qwen3-VL-2B-Instruct-coordexp"
    )
    assert effective["runtime"]["token_rows"]["expected_trainable_row_count"] == 1002
    assert effective["runtime"]["packing"]["enabled"] is False
    assert effective["runtime"]["encoded_sample_cache"]["enabled"] is False
    assert effective["runtime"]["dataset_source_train_jsonl"]["raw_path"] == str(
        train_jsonl
    )
    assert effective["runtime"]["dataset_source_val_jsonl"]["raw_path"] == str(
        val_jsonl
    )

    run_metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    assert run_metadata["run_manifest_files"] == written
    assert run_metadata["run_name"] == "unit-prefix-rollin-smoke"

    experiment_manifest = json.loads(
        (output_dir / "experiment_manifest.json").read_text("utf-8")
    )
    assert experiment_manifest["experiment"]["authored"]["surface"] == "ablation"
    assert (
        experiment_manifest["runtime_summary"]["latest_detection_objective"]["variant"]
        == "prefix_rollin_et_rmp_ce"
    )
    assert experiment_manifest["runtime_summary"]["effective_batch_size"] == 1
    assert (
        experiment_manifest["runtime_summary"]["actual_global_effective_batch_size"]
        == 1
    )
    assert experiment_manifest["artifacts"]["resolved_config"] == "resolved_config.json"
    assert experiment_manifest["artifacts"]["effective_runtime"] == "effective_runtime.json"
    assert "pipeline_manifest" not in experiment_manifest["artifacts"]


def test_prefix_rollin_bsz8_configs_record_eval_and_disabled_packing_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/ablation/"
        "compact_full_prefix_rollin_balance2_a3_bsz8_ebs128.yaml",
    )

    for cfg_path in config_paths:
        cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))
        assert isinstance(cfg, LatestDetectionTrainingConfig)
        assert cfg.training["per_device_train_batch_size"] == 8
        assert cfg.training["per_device_eval_batch_size"] == 8
        assert cfg.training["effective_batch_size"] == 128
        assert cfg.training["eval_steps"] == 600
        assert cfg.training["group_by_length"] is True
        assert cfg.training["length_column_name"] == "length"
        assert cfg.training["packing"] is False
        assert cfg.training["eval_packing"] is False
        assert cfg.packing.static_packing is False
        assert cfg.packing.padding_free_packed is False

        train_args = SimpleNamespace(
            run_name=cfg.training["run_name"],
            output_dir=str(repo_root / "temp" / "unit" / cfg.training["run_name"]),
            logging_dir=str(repo_root / "temp" / "unit" / "tb" / cfg.training["run_name"]),
            save_only_model=False,
            save_strategy="no",
            save_last_epoch=True,
            seed=123,
            per_device_train_batch_size=cfg.training["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg.training["per_device_eval_batch_size"],
            gradient_accumulation_steps=4,
            eval_strategy=cfg.training["eval_strategy"],
            eval_steps=cfg.training["eval_steps"],
            group_by_length=cfg.training["group_by_length"],
            length_column_name=cfg.training["length_column_name"],
            max_steps=cfg.training.get("max_steps", -1),
            num_train_epochs=cfg.training.get("num_train_epochs", 1.0),
            dataloader_drop_last=cfg.training.get("dataloader_drop_last", False),
            deepspeed=None,
            resume_from_checkpoint=None,
            max_model_len=cfg.template.get("max_length", 0),
        )
        _apply_checkpoint_mode(train_args, checkpoint_mode="artifact_only")
        packing_cfg = _parse_packing_config(
            cfg.training,
            template=SimpleNamespace(max_length=cfg.template.get("max_length", 0)),
            train_args=train_args,
        )
        effective_runtime = _build_effective_runtime_payload(
            training_config=cfg,
            train_args=train_args,
            trainer_variant=None,
            dataset_seed=123,
            checkpoint_mode="artifact_only",
            packing_cfg=packing_cfg,
            encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(enabled=False),
            train_jsonl=cfg.data.train_jsonl,
            val_jsonl=cfg.data.val_jsonl,
            pipeline_manifest=None,
        )

        assert effective_runtime["eval_strategy"] == str(
            getattr(train_args, "eval_strategy", "") or ""
        )
        assert effective_runtime["eval_steps"] == 600
        assert effective_runtime["per_device_eval_batch_size"] == 8
        assert effective_runtime["packing"]["enabled"] is False
        assert effective_runtime["packing"]["eval_packing"] is False
        assert effective_runtime["dataloader"]["group_by_length"] is True
        assert effective_runtime["dataloader"]["length_column_name"] == "length"
        length_bucketing = effective_runtime["dataloader"]["length_bucketing"]
        assert length_bucketing["enabled"] is True
        assert length_bucketing["mode"] == "row_atomic_length_bucketing"
        assert (
            length_bucketing["length_source"]
            == "DetectionTrainingDataset.encoded_length_for_row"
        )
        assert length_bucketing["cache_policy"] == "run_local_only"

        experiment_manifest = build_experiment_manifest_payload(
            output_dir=str(Path(train_args.output_dir)),
            config_path=str(cfg_path),
            base_config_path=None,
            run_name=str(train_args.run_name),
            dataset_seed=123,
            experiment=cfg.to_mapping()["experiment"],
            effective_runtime=effective_runtime,
            pipeline_manifest=None,
            run_metadata={},
            manifest_files={},
        )
        runtime_summary = experiment_manifest["runtime_summary"]
        assert runtime_summary["eval_strategy"] == effective_runtime["eval_strategy"]
        assert runtime_summary["eval_steps"] == 600
        assert runtime_summary["per_device_eval_batch_size"] == 8
        assert runtime_summary["packing"]["enabled"] is False
        assert runtime_summary["packing"]["eval_packing"] is False
        assert runtime_summary["dataloader"]["length_bucketing"]["enabled"] is True
