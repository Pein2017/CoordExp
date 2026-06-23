from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from src.tokens.row_offsets import install_token_embeddings_adapter
from src.config.schema import TokenEmbeddingsAdapterConfig
from src.training.coverage_ledger.head import install_coverage_ledger_head

try:
    from src.optim.token_embeddings_adapter_optimizer import create_multimodal_token_embeddings_adapter_optimizer
except ImportError:
    pytest.skip(
        "swift.plugin.optimizer not installed; skipping token_embeddings_adapter optimizer tests",
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
        self.config = SimpleNamespace(
            hidden_size=4,
            vision_config=SimpleNamespace(hidden_size=4),
        )
        self.model_meta = SimpleNamespace(
            model_arch=SimpleNamespace(
                vision_tower=["vision"], aligner=["aligner"], language_model=["llm"]
            )
        )

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden)


class ToyModelWithoutMeta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(20, 8)
        self.lm_head = nn.Linear(8, 20, bias=False)
        self.body = nn.Linear(8, 8)

    def forward(self, input_ids):
        hidden = self.body(self.embed_tokens(input_ids))
        return self.lm_head(hidden)


def test_optimizer_groups_separate_token_embeddings_adapter_offsets():
    model = ToyModel()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[3, 4], tie_head=True, dtype="float32"
    )

    adapter_cfg = TokenEmbeddingsAdapterConfig(
        enabled=True,
        tie_head=True,
        embed_lr=1e-3,
        head_lr=2e-3,
        weight_decay=0.0,
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
    args.token_embeddings_adapter_config = adapter_cfg

    optimizer, _ = create_multimodal_token_embeddings_adapter_optimizer(
        args,
        model,
        dataset=None,
    )

    lr_by_param = {}
    wd_by_param = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        for p in group["params"]:
            lr_by_param[id(p)] = lr
            wd_by_param[id(p)] = wd

    # Tie-head: only a single shared offset table should exist.
    assert adapter.head_offset is None
    assert lr_by_param[id(adapter.embed_offset)] == adapter_cfg.embed_lr
    assert wd_by_param[id(adapter.embed_offset)] == adapter_cfg.weight_decay

    # Vision/aligner/llm params follow their respective LRs
    vision_weight = dict(model.named_parameters())["vision.weight"]
    aligner_weight = dict(model.named_parameters())["aligner.weight"]
    llm_weight = dict(model.named_parameters())["llm.weight"]

    assert lr_by_param[id(vision_weight)] == args.vit_lr
    assert lr_by_param[id(aligner_weight)] == args.aligner_lr
    assert lr_by_param[id(llm_weight)] == args.learning_rate


def test_optimizer_includes_coverage_ledger_head_with_main_lr_when_model_arch_splits_params():
    model = ToyModel()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[3, 4], tie_head=True, dtype="float32"
    )
    ledger_head = install_coverage_ledger_head(
        model,
        SimpleNamespace(enabled=True, ledger_projection_dim=4, normalize_eps=1.0e-6),
        visual_dim=4,
    )
    assert ledger_head is not None

    adapter_cfg = TokenEmbeddingsAdapterConfig(
        enabled=True,
        tie_head=True,
        embed_lr=1e-3,
        weight_decay=0.0,
    )
    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        weight_decay=0.01,
    )
    args.vit_lr = 2e-4
    args.aligner_lr = 8e-4
    args.token_embeddings_adapter_config = adapter_cfg

    optimizer, _ = create_multimodal_token_embeddings_adapter_optimizer(args, model, dataset=None)

    grouped_param_ids = [
        id(param) for group in optimizer.param_groups for param in group["params"]
    ]
    lr_by_param = {}
    wd_by_param = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            lr_by_param[id(param)] = group["lr"]
            wd_by_param[id(param)] = group.get("weight_decay", 0.0)

    for _name, param in ledger_head.named_parameters():
        assert grouped_param_ids.count(id(param)) == 1
        assert lr_by_param[id(param)] == args.learning_rate
        assert wd_by_param[id(param)] == args.weight_decay
    assert grouped_param_ids.count(id(adapter.embed_offset)) == 1


def test_optimizer_ledger_head_step_updates_param_when_arch_prefixes_would_miss_it():
    model = ToyModel()
    install_token_embeddings_adapter(
        model, token_ids=[3, 4], tie_head=True, dtype="float32"
    )
    ledger_head = install_coverage_ledger_head(
        model,
        SimpleNamespace(enabled=True, ledger_projection_dim=4, normalize_eps=1.0e-6),
        visual_dim=4,
    )
    assert ledger_head is not None
    adapter_cfg = TokenEmbeddingsAdapterConfig(enabled=True, tie_head=True)
    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=1,
        learning_rate=5e-2,
        weight_decay=0.0,
    )
    args.vit_lr = 2e-4
    args.aligner_lr = 8e-4
    args.token_embeddings_adapter_config = adapter_cfg
    optimizer, _ = create_multimodal_token_embeddings_adapter_optimizer(
        args,
        model,
        dataset=None,
    )

    before = {
        name: param.detach().clone() for name, param in ledger_head.named_parameters()
    }
    hidden = torch.ones((1, 4))
    visual = torch.ones((1, 4))
    loss = (
        ledger_head.state_projection(hidden).sum()
        + ledger_head.region_anchor_state_projection(visual).sum()
        + ledger_head.object_projection(hidden).sum()
    )
    loss.backward()
    optimizer.step()

    after = dict(ledger_head.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_optimizer_groups_untied_offsets_use_two_buckets():
    model = ToyModel()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[3, 4], tie_head=False, dtype="float32"
    )

    adapter_cfg = TokenEmbeddingsAdapterConfig(
        enabled=True,
        tie_head=False,
        embed_lr=1e-3,
        head_lr=2e-3,
        weight_decay=0.0,
    )

    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        weight_decay=0.01,
    )
    args.vit_lr = 2e-4
    args.aligner_lr = 8e-4
    args.token_embeddings_adapter_config = adapter_cfg

    optimizer, _ = create_multimodal_token_embeddings_adapter_optimizer(args, model, dataset=None)

    lr_by_param = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        for p in group["params"]:
            lr_by_param[id(p)] = lr

    assert adapter.head_offset is not None
    assert lr_by_param[id(adapter.embed_offset)] == adapter_cfg.embed_lr
    assert lr_by_param[id(adapter.head_offset)] == adapter_cfg.head_lr


def test_optimizer_fallback_without_model_meta_groups_remaining_params_once():
    model = ToyModelWithoutMeta()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[3, 4], tie_head=True, dtype="float32"
    )

    adapter_cfg = TokenEmbeddingsAdapterConfig(
        enabled=True,
        tie_head=True,
        embed_lr=1e-3,
        weight_decay=0.0,
    )
    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        weight_decay=0.01,
    )
    args.vit_lr = None
    args.aligner_lr = None
    args.token_embeddings_adapter_config = adapter_cfg

    optimizer, _ = create_multimodal_token_embeddings_adapter_optimizer(args, model, dataset=None)

    grouped_param_ids = [
        id(param) for group in optimizer.param_groups for param in group["params"]
    ]
    assert grouped_param_ids.count(id(adapter.embed_offset)) == 1
    assert grouped_param_ids.count(id(model.body.weight)) == 1
    assert grouped_param_ids.count(id(model.embed_tokens.weight)) == 0
    assert grouped_param_ids.count(id(model.lm_head.weight)) == 0
