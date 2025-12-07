from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from src.coord_tokens.offset_adapter import install_coord_offset_adapter
from src.config.schema import CoordOffsetConfig

try:
    from src.optim.coord_offset_optimizer import create_multimodal_coord_offset_optimizer
except ImportError:
    pytest.skip(
        "swift.plugin.optimizer not installed; skipping coord_offset_optimizer tests",
        allow_module_level=True,
    )


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(20, 8)
        self.lm_head = nn.Linear(8, 20, bias=False)
        self.vision = nn.Linear(4, 4)
        self.aligner = nn.Linear(4, 4)
        self.llm = nn.Linear(4, 4)
        self.model_meta = SimpleNamespace(
            model_arch=SimpleNamespace(
                vision_tower=["vision"], aligner=["aligner"], language_model=["llm"]
            )
        )

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden)


def test_optimizer_groups_separate_coord_offsets():
    model = ToyModel()
    adapter = install_coord_offset_adapter(model, coord_ids=[3, 4], dtype="float32")

    coord_cfg = CoordOffsetConfig(
        enabled=True, ids=(3, 4), embed_lr=1e-3, head_lr=2e-3, weight_decay=0.0
    )

    # HF TrainingArguments with extra attrs expected by the optimizer
    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        weight_decay=0.01,
    )
    # Inject ms-swift style attrs
    args.vit_lr = 2e-4
    args.aligner_lr = 8e-4
    args.coord_offset_config = coord_cfg

    optimizer, _ = create_multimodal_coord_offset_optimizer(args, model, dataset=None)

    lr_by_param = {}
    wd_by_param = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        for p in group["params"]:
            lr_by_param[id(p)] = lr
            wd_by_param[id(p)] = wd

    assert lr_by_param[id(adapter.embed_offset)] == coord_cfg.embed_lr
    assert lr_by_param[id(adapter.head_offset)] == coord_cfg.head_lr
    assert wd_by_param[id(adapter.embed_offset)] == coord_cfg.weight_decay

    # Vision/aligner/llm params follow their respective LRs
    vision_weight = dict(model.named_parameters())["vision.weight"]
    aligner_weight = dict(model.named_parameters())["aligner.weight"]
    llm_weight = dict(model.named_parameters())["llm.weight"]

    assert lr_by_param[id(vision_weight)] == args.vit_lr
    assert lr_by_param[id(aligner_weight)] == args.aligner_lr
    assert lr_by_param[id(llm_weight)] == args.learning_rate
