from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from probes.source_rweak_row_cross import prepare, run
from src.qwen.images import rgb_image_sha256


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    tokens = {
        1: "person",
        3: "car",
        151645: "<|im_end|>",
        151646: "<|object_ref_start|>",
        151647: "<|object_ref_end|>",
        151648: "<|box_start|>",
        151649: "<|box_end|>",
        **{151670 + n: f"<|coord_{n}|>" for n in (100, 200, 300, 400)},
    }

    def convert_tokens_to_ids(self, text):
        return 100 if text == "<|image_pad|>" else 151645

    def __call__(self, text, **kwargs):
        return {"input_ids": [100, 2]}

    def decode(self, ids, **kwargs):
        return "".join(self.tokens[i] for i in ids)


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return "chat"

    def __call__(self, *, text, images, **kwargs):
        n = len(text)
        return {
            "input_ids": torch.tensor([[100, 2]] * n),
            "attention_mask": torch.ones(n, 2, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * n),
            "pixel_values": torch.ones(n, 1),
        }


def tiny_model():
    from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutput

    class Tiny(PreTrainedModel, GenerationMixin):
        _supports_cache_class = False

        def __init__(self):
            super().__init__(
                PretrainedConfig(
                    vocab_size=152670,
                    eos_token_id=151645,
                    pad_token_id=0,
                    use_cache=False,
                )
            )
            self.bias = torch.nn.Parameter(torch.zeros(152670))
            with torch.no_grad():
                self.bias[151645] = 1
            self.visual = torch.nn.Identity()

        def forward(
            self,
            input_ids,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            **kwargs,
        ):
            if pixel_values is not None:
                self.visual(pixel_values)
            return CausalLMOutput(
                logits=self.bias.view(1, 1, -1).expand(input_ids.shape[0], 1, -1)
            )

    return Tiny().eval()


def binding(path):
    return {"path": str(path), "sha256": run.sha(path), "bytes": path.stat().st_size}


def fixture_manifest(tmp_path):
    tokenizer = Tokenizer()
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image_path)
    with Image.open(image_path) as image:
        media = rgb_image_sha256(image)

    def action(desc):
        return [151646, desc, 151647, 151648, 151770, 151870, 151970, 152070, 151649]

    config = {
        "model": {
            "base_model": "/unused/tiny",
            "dtype": "fp32",
            "processor": {"do_resize": False},
        },
        "backend": {
            "hf": {
                "attn_implementation": "sdpa",
                "patch_embed_linearization": "enabled",
            }
        },
        "generation": {
            "batch_size": 4,
            "max_new_tokens": run.CAP,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        },
        "adapter": {"path": "/unused/adapter", "name": "default"},
        "embedding_delta": {"path": "/unused/delta"},
        "data": {"input_jsonl": str(tmp_path / "input.jsonl")},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "geo_sorted_xy",
            "assistant_format": "object_box_closed",
            "prompt": {"system": "Detect.", "user": "Locate objects."},
        },
    }
    yaml_path = tmp_path / "resolved.yaml"
    yaml_path.write_text(json.dumps({"config": config}))
    cases = []
    for index in (1, 2):
        rid = f"coco2017_val_{index:012d}"
        prefix = [] if index == 1 else action(1)
        raw = {
            "row_id": rid,
            "row_index": index - 1,
            "example_id": rid,
            "image_path": str(image_path),
            "image_width": 32,
            "image_height": 32,
            "gt": [],
        }
        case = {
            "row_id": rid,
            "row_index": index - 1,
            "image_path": str(image_path),
            "image_width": 32,
            "image_height": 32,
            "input_record": {
                "file_name": "image.png",
                "image_id": index,
                "images": ["image.png"],
                "width": 32,
                "height": 32,
                "objects": [
                    {
                        "coco_ann_id": index,
                        "category_id": 1,
                        "category_name": "person",
                        "desc": "person",
                        "bbox_2d": [f"<|coord_{n}|>" for n in (100, 200, 300, 400)],
                    }
                ],
                "metadata": {"source": "coco2017", "split": "val"},
            },
            "common_prefix_token_ids": prefix,
            "image_plan": {
                "merged_visual_tokens": 1,
                "backend_prompt_token_count": 2,
                "observed_image_grid_thw": [1, 1, 1],
                "image_content_sha256": run.sha(image_path),
                "executed_media_sha256": media,
                "logical_transform_id": "identity",
            },
            "actions": {},
            "diagonals": {},
            "remaining_token_budgets": {},
            "continuation_token_budgets": {},
        }
        for arm, desc in [("source", 1), ("rweak", 3)]:
            ids = action(desc)
            case["actions"][arm] = {"kind": "row", "token_ids": ids}
            complete = prefix + ids + [run.EOS]
            case["diagonals"][arm] = {
                "generated_token_ids": complete,
                "raw_record": run.native_record(
                    tokenizer.decode(complete), case, raw, "im_end"
                ),
                "matched_gt_ids": {f"iou_{t:.2f}": [] for t in (0.5, 0.6, 0.8)},
            }
            case["remaining_token_budgets"][arm] = run.CAP - len(prefix) - len(ids)
            case["continuation_token_budgets"][arm] = case["remaining_token_budgets"][
                arm
            ]
        cases.append(case)
    manifest = {
        "schema": "row_cross_manifest_v1",
        "sources": {
            "source": {"config": config, "original_yaml": binding(yaml_path)},
            "code": [],
        },
        "policy": {
            "max_new_tokens": run.CAP,
            "repetition_penalty": 1.0,
            "stop_token_id": run.EOS,
        },
        "selection": {
            "population_images": 512,
            "population_gt": 3759,
            "exclusions": [{}] * 510,
        },
        "cases": cases,
        "qualification_case_ids": prepare.qualify(cases),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_actual_cli_plan_and_tiny_generation_consumer(tmp_path, monkeypatch):
    path, manifest = fixture_manifest(tmp_path)
    monkeypatch.setattr(run, "MANIFEST_SHA256", run.sha(path))
    calls = []

    def load(config, source, device):
        calls.append(str(device))
        processor = Processor()
        return SimpleNamespace(
            model=tiny_model(), tokenizer=processor.tokenizer, processor=processor
        ), {"kind": "cpu_tiny_model", "device": str(device)}

    monkeypatch.setattr(run, "load_components", load)
    common = [
        "--manifest",
        str(path),
        "--recipient",
        "source",
        "--mode",
        "cross",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--max-wall-seconds",
        "60",
    ]
    assert run.main(common + ["--output-dir", str(tmp_path / "plan")]) == 0
    assert calls == []
    assert (
        run.main(common + ["--output-dir", str(tmp_path / "executed"), "--execute"])
        == 0
    )
    assert calls == ["cpu"]
    receipt = json.loads((tmp_path / "executed/receipt.json").read_text())
    assert (
        receipt["status"] == "success"
        and receipt["native_execution"]["kind"] == "cpu_tiny_model"
    )
    assert "backend_session" not in receipt
    assert receipt["counters"] == {
        "model_forwards": 1,
        "image_forwards": 1,
        "image_instances": 2,
    }
    assert receipt["effective_generation"]["trace"] == "none"
    rows = [
        json.loads(line)
        for line in (tmp_path / "executed/rows.jsonl").read_text().splitlines()
    ]
    for row, case in zip(rows, manifest["cases"], strict=True):
        assert row["generated_token_ids"] == case["common_prefix_token_ids"] + case[
            "actions"
        ]["rweak"]["token_ids"] + [run.EOS]
    with pytest.raises(ValueError, match="overwrite"):
        run.main(common + ["--output-dir", str(tmp_path / "executed")])


def test_manifest_corruption_fails_before_model_entry(tmp_path, monkeypatch):
    path, manifest = fixture_manifest(tmp_path)
    manifest["cases"][0]["remaining_token_budgets"]["source"] += 1
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(run, "MANIFEST_SHA256", run.sha(path))
    monkeypatch.setattr(
        run,
        "load_components",
        lambda *args: pytest.fail("model load before manifest validation"),
    )
    with pytest.raises(ValueError, match="remaining budget"):
        run.main(
            [
                "--manifest",
                str(path),
                "--recipient",
                "source",
                "--mode",
                "cross",
                "--output-dir",
                str(tmp_path / "bad"),
                "--execute",
            ]
        )
    assert not (tmp_path / "bad").exists()


def test_record_identity_and_live_bindings(tmp_path):
    row = dict(
        manifest_sha256="frozen",
        generated_token_ids=[3, 4, 5],
        prefix_token_ids=[3],
        action_token_ids=[4],
        suffix_token_ids=[5],
        generated_token_count=3,
        remaining_budget=run.CAP - 2,
        decode_stop_reason="im_end",
    )
    run.validate_record(row, "frozen")
    for key, value in [
        ("generated_token_ids", [3, 9, 5]),
        ("manifest_sha256", "other"),
        ("remaining_budget", run.CAP),
        ("generated_token_count", 4),
        ("decode_stop_reason", "forced_eos"),
    ]:
        with pytest.raises(ValueError):
            run.validate_record(row | {key: value}, "frozen")
    path = tmp_path / "source"
    path.write_text("original")
    manifest = {
        "sources": {
            "code": [],
            "live": binding(path),
            "historical": {"path": str(path), "sha256": "old"},
        },
        "policy": {
            "max_new_tokens": run.CAP,
            "repetition_penalty": 1.0,
            "stop_token_id": run.EOS,
        },
    }
    run.verify_bindings(manifest)
    path.write_text("mutated")
    with pytest.raises(ValueError, match="digest"):
        run.verify_bindings(manifest)
