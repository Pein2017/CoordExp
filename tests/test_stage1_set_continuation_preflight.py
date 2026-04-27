from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn

from src.data_collators.stage1_set_continuation_collator import (
    build_stage1_set_continuation_collator,
)
from src.config import ConfigLoader
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.trainers.stage1_set_continuation.branch_encoder import (
    encode_set_continuation_branch,
)
from src.trainers.stage1_set_continuation.branch_scorer import (
    TensorBranchScoreInput,
    score_tensor_batch_retained,
    score_tensor_retained,
)
from src.trainers.stage1_set_continuation.losses import (
    compute_candidate_full_entry_logprob,
    compute_mp_pem_losses,
)
from src.trainers.stage1_set_continuation.serialization import (
    build_candidate_entry_text,
    build_prefix_text,
    render_indexed_object_list,
)


OBJECT_A = {
    "desc": "alpha",
    "bbox_2d": [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ],
}
OBJECT_B = {
    "desc": "beta",
    "bbox_2d": [
        "<|coord_11|>",
        "<|coord_12|>",
        "<|coord_13|>",
        "<|coord_14|>",
    ],
}
OBJECT_C = {
    "desc": "gamma",
    "bbox_2d": [
        "<|coord_21|>",
        "<|coord_22|>",
        "<|coord_23|>",
        "<|coord_24|>",
    ],
}


class _FakeTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode_text(self, text: str) -> list[int]:
        tokens = re.findall(r"<\|coord_\d+\|>|[A-Za-z0-9_]+|[^\s]", text)
        ids: list[int] = []
        for token in tokens:
            if token not in self._token_to_id:
                next_id = len(self._token_to_id) + 1
                self._token_to_id[token] = next_id
                self._id_to_token[next_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[int(idx)] for idx in ids]


class _FakeTemplate:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.system = "original-system"

    def encode(
        self, rendered: Mapping[str, Any], return_length: bool = True
    ) -> dict[str, Any]:
        del return_length
        assistant = next(
            message
            for message in rendered["messages"]
            if message.get("role") == "assistant"
        )
        content = assistant["content"]
        if isinstance(content, str):
            text = content
        else:
            text = next(
                item["text"]
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
        ids = self.tokenizer.encode_text(text)
        return {
            "input_ids": ids,
            "labels": list(ids),
            "attention_mask": [1] * len(ids),
            "length": len(ids),
        }


class _TinyBranchModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        hidden = self.embed(input_ids)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        if kwargs:
            raise AssertionError(f"unexpected kwargs: {sorted(kwargs)}")
        if isinstance(logits_to_keep, int) and int(logits_to_keep) > 0:
            hidden = hidden[:, -int(logits_to_keep) :, :]
        return type("Output", (), {"logits": self.proj(hidden)})()


def _raw_sample() -> dict[str, Any]:
    return {
        "input_ids": [1, 2],
        "labels": [-100, 2],
        "assistant_payload": {"objects": [OBJECT_A, OBJECT_B, OBJECT_C]},
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/image.jpg"},
                    {"type": "text", "text": "detect"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
        "metadata": {"image_id": 1, "dataset": "toy"},
        "sample_id": "preflight-sample",
    }


def _meta() -> dict[str, Any]:
    return build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]


def _real_dataset_meta(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    config = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage1/set_continuation/production.yaml")
    )
    train_jsonl = repo_root / str(config.custom.train_jsonl)
    if not train_jsonl.is_file():
        pytest.skip(f"missing real train jsonl: {train_jsonl}")
    monkeypatch.setenv("ROOT_IMAGE_DIR", str(train_jsonl.parent))
    with train_jsonl.open("r", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    if len(record.get("objects") or []) < 3:
        pytest.skip("real preflight needs a sample with at least three objects")
    record["objects"] = list(record["objects"][:3])

    builder = JSONLinesBuilder(
        user_prompt=config.prompts.user,
        emit_norm="none",
        mode="dense",
        coord_tokens_enabled=bool(config.custom.coord_tokens.enabled),
        object_field_order=str(config.custom.object_field_order),
        bbox_format=str(config.custom.bbox_format),
    )
    rendered = builder.build(record)
    rendered["sample_id"] = "real-jsonl-preflight"
    return build_stage1_set_continuation_collator()([rendered])[
        "set_continuation_meta"
    ][0]


def _real_template(monkeypatch: pytest.MonkeyPatch) -> Any:
    from swift.llm import get_model_tokenizer, get_template

    repo_root = Path(__file__).resolve().parent.parent
    config = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage1/set_continuation/production.yaml")
    )
    model_dir = repo_root / str(config.model["model"])
    if not model_dir.is_dir():
        pytest.skip(f"missing production tokenizer/model dir: {model_dir}")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    _, processor = get_model_tokenizer(str(model_dir), load_model=False)
    template = get_template(
        str(config.template.get("template") or "qwen3_vl"),
        processor,
        max_length=8192,
        truncation_strategy="right",
        max_pixels=int(config.template.get("max_pixels") or 1048576),
        padding_free=False,
    )
    template.system = config.prompts.system
    template.set_mode("train")
    return template


def _label_tokens(template: _FakeTemplate, labels: torch.Tensor) -> list[str | None]:
    flattened = [int(value) for value in labels.reshape(-1).tolist()]
    tokens: list[str | None] = []
    for value in flattened:
        tokens.append(
            None
            if value == -100
            else template.tokenizer.convert_ids_to_tokens([value])[0]
        )
    return tokens


def _scored_tokens(
    template: _FakeTemplate,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> list[str]:
    tokens = _label_tokens(template, labels)
    return [
        str(token)
        for token, is_scored in zip(tokens, mask.reshape(-1).tolist(), strict=True)
        if is_scored
    ]


def test_preflight_serialization_truth_table_is_continuation_aware() -> None:
    rendered = render_indexed_object_list(
        [OBJECT_A, OBJECT_B, OBJECT_C],
        object_field_order="desc_first",
    )

    nonterminal = build_candidate_entry_text(
        rendered,
        prefix_indices=[0],
        candidate_index=1,
    )
    terminal = build_candidate_entry_text(
        rendered,
        prefix_indices=[0, 1],
        candidate_index=2,
    )

    assert nonterminal.continuation_mode == "continue"
    assert nonterminal.text.endswith(", ")
    assert nonterminal.post_candidate_span.end == len(nonterminal.text)
    assert (
        nonterminal.global_close_start_span.start
        == nonterminal.global_close_start_span.end
    )
    assert (
        nonterminal.global_full_close_span.start
        == nonterminal.global_full_close_span.end
    )

    assert terminal.continuation_mode == "close"
    assert terminal.text.endswith("]}")
    assert (
        terminal.text[
            terminal.global_close_start_span.start : terminal.global_close_start_span.end
        ]
        == "]"
    )
    assert (
        terminal.text[
            terminal.global_full_close_span.start : terminal.global_full_close_span.end
        ]
        == "]}"
    )


def test_preflight_token_masks_include_append_boundary_and_exclude_close() -> None:
    template = _FakeTemplate()
    branch = encode_set_continuation_branch(
        meta=_meta(),
        template=template,
        prefix_indices=[0],
        candidate_index=1,
        object_field_order="desc_first",
    )

    scored = _scored_tokens(
        template,
        branch.labels,
        branch.candidate_entry_label_mask,
    )

    assert branch.prefix_text + branch.continuation_text == branch.rendered_text
    assert branch.rendered_text.endswith(", ")
    assert scored[-1] == ","
    assert scored[-2:] != ["]", "}"]
    assert branch.structural_close_start_mask.sum().item() == 0
    assert branch.structural_close_sequence_mask.sum().item() == 0


def test_preflight_token_masks_include_terminal_close_boundary() -> None:
    template = _FakeTemplate()
    branch = encode_set_continuation_branch(
        meta=_meta(),
        template=template,
        prefix_indices=[0, 1],
        candidate_index=2,
        object_field_order="desc_first",
    )

    scored = _scored_tokens(
        template,
        branch.labels,
        branch.candidate_entry_label_mask,
    )
    close_sequence = _scored_tokens(
        template,
        branch.labels,
        branch.structural_close_sequence_mask,
    )

    assert branch.rendered_text.endswith("]}")
    assert scored[-2:] == ["]", "}"]
    assert close_sequence == ["]", "}"]
    assert branch.structural_close_start_mask.sum().item() == 1


def test_preflight_real_jsonl_and_template_score_continuation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _real_template(monkeypatch)
    meta = _real_dataset_meta(monkeypatch)

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0],
        candidate_index=1,
        object_field_order="desc_first",
    )
    scored = _scored_tokens(
        template,
        branch.labels,
        branch.candidate_entry_label_mask,
    )

    assert branch.rendered_text.endswith(", ")
    assert scored, "real tokenizer produced no scored candidate labels"
    assert "<|im_end|>" not in scored
    assert "".join(scored[-4:]).replace("Ġ", " ").endswith(", ")
    assert branch.structural_close_start_mask.sum().item() == 0
    assert branch.structural_close_sequence_mask.sum().item() == 0


def test_preflight_real_jsonl_and_template_empty_prefix_scores_schema_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _real_template(monkeypatch)
    meta = _real_dataset_meta(monkeypatch)

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[],
        candidate_index=0,
        object_field_order="desc_first",
    )
    objective = _scored_tokens(
        template,
        branch.labels,
        branch.objective_label_mask,
    )
    schema = _scored_tokens(
        template,
        branch.labels,
        branch.schema_open_label_mask,
    )
    candidate_object = _scored_tokens(
        template,
        branch.labels,
        branch.candidate_object_label_mask,
    )
    objective_text = "".join(objective).replace("Ġ", " ")
    schema_text = "".join(schema).replace("Ġ", " ")
    candidate_text = "".join(candidate_object).replace("Ġ", " ")

    assert branch.prefix_text == '{"objects": ['
    assert "<|im_end|>" not in objective
    assert objective_text.startswith('{"objects": [{"desc"')
    assert schema_text.startswith('{"objects": [')
    assert '[{"' in schema_text or schema_text.endswith("[")
    assert '{"desc"' in candidate_text[:16]
    assert objective_text.endswith(", ")


def test_preflight_real_jsonl_and_template_score_terminal_close_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _real_template(monkeypatch)
    meta = _real_dataset_meta(monkeypatch)

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0, 1],
        candidate_index=2,
        object_field_order="desc_first",
    )
    scored = _scored_tokens(
        template,
        branch.labels,
        branch.candidate_entry_label_mask,
    )
    close_sequence = _scored_tokens(
        template,
        branch.labels,
        branch.structural_close_sequence_mask,
    )

    assert branch.rendered_text.endswith("]}")
    assert scored, "real tokenizer produced no scored candidate labels"
    assert "<|im_end|>" not in scored
    assert "<|im_end|>" not in close_sequence
    assert "".join(scored[-8:]).endswith("]}")
    assert "".join(close_sequence).endswith("]}")
    assert branch.structural_close_start_mask.sum().item() >= 1


def test_preflight_real_jsonl_and_template_nonempty_prefix_does_not_rescore_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _real_template(monkeypatch)
    meta = _real_dataset_meta(monkeypatch)

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0],
        candidate_index=1,
        object_field_order="desc_first",
    )
    objective = _scored_tokens(
        template,
        branch.labels,
        branch.objective_label_mask,
    )
    schema = _scored_tokens(
        template,
        branch.labels,
        branch.schema_open_label_mask,
    )
    objective_text = "".join(objective).replace("Ġ", " ")

    assert schema == []
    assert not objective_text.startswith('{"objects"')
    assert objective_text.startswith('{"desc"') or objective_text.startswith(' {"desc"')
    assert objective_text.endswith(", ")


def test_preflight_json_structural_mask_scores_schema_keys_and_boundaries_only() -> (
    None
):
    template = _FakeTemplate()
    branch = encode_set_continuation_branch(
        meta=_meta(),
        template=template,
        prefix_indices=[],
        candidate_index=0,
        object_field_order="desc_first",
    )

    structural = _scored_tokens(
        template,
        branch.labels,
        branch.json_structural_label_mask,
    )

    assert structural[0] == "{"
    assert "desc" in structural
    assert "objects" in structural
    assert ":" in structural
    assert "[" in structural
    assert "bbox_2d" in structural
    assert structural[-1] == ","
    assert "alpha" not in structural
    assert not any(token.startswith("<|coord_") for token in structural)


def test_preflight_json_structural_mask_includes_terminal_close_but_not_payload() -> (
    None
):
    template = _FakeTemplate()
    branch = encode_set_continuation_branch(
        meta=_meta(),
        template=template,
        prefix_indices=[0, 1],
        candidate_index=2,
        object_field_order="desc_first",
    )

    structural = _scored_tokens(
        template,
        branch.labels,
        branch.json_structural_label_mask,
    )

    assert structural[-2:] == ["]", "}"]
    assert "gamma" not in structural
    assert not any(token.startswith("<|coord_") for token in structural)


def test_preflight_shifted_loss_scores_exact_next_token_positions() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    candidate_mask = torch.tensor([[False, False, True, True, True]])
    coord_mask = torch.zeros_like(candidate_mask)
    coord_token_ids = torch.tensor([8, 9], dtype=torch.long)

    aligned_logits = torch.full((1, 5, 10), -8.0)
    aligned_logits[0, 1, 2] = 8.0
    aligned_logits[0, 2, 3] = 8.0
    aligned_logits[0, 3, 4] = 8.0
    aligned = compute_candidate_full_entry_logprob(
        logits=aligned_logits,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )

    early_logits = torch.full((1, 5, 10), -8.0)
    early_logits[0, 0, 2] = 8.0
    early_logits[0, 1, 3] = 8.0
    early_logits[0, 2, 4] = 8.0
    early = compute_candidate_full_entry_logprob(
        logits=early_logits,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )

    no_boundary = compute_candidate_full_entry_logprob(
        logits=aligned_logits,
        labels=labels,
        candidate_entry_label_mask=torch.tensor([[False, False, True, True, False]]),
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )
    boundary_only = compute_candidate_full_entry_logprob(
        logits=aligned_logits,
        labels=labels,
        candidate_entry_label_mask=torch.tensor([[False, False, False, False, True]]),
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )

    assert aligned.score > early.score + 20.0
    assert torch.allclose(aligned.score, no_boundary.score + boundary_only.score)
    assert aligned.tokens == 3
    assert boundary_only.tokens == 1


def test_preflight_candidate_logprob_reports_json_structural_score_separately() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    candidate_mask = torch.tensor([[False, False, True, True, True]])
    structural_mask = torch.tensor([[False, False, True, False, True]])
    coord_mask = torch.zeros_like(candidate_mask)
    coord_token_ids = torch.tensor([8, 9], dtype=torch.long)
    logits = torch.full((1, 5, 10), -8.0)
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0
    logits[0, 3, 4] = 8.0

    result = compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
        json_structural_label_mask=structural_mask,
    )

    assert result.tokens == 3
    assert result.json_structural_tokens == 2
    assert result.json_structural_score < 0.0
    assert result.json_structural_score > result.score


def test_preflight_branch_runtime_equivalence_preserves_boundary_scores() -> None:
    torch.manual_seed(123)
    model_serial = _TinyBranchModel()
    model_batch = copy.deepcopy(model_serial)
    coord_token_ids = torch.tensor([20, 21, 22, 23], dtype=torch.long)
    comma_token = 7
    close_token = 8

    nonterminal = TensorBranchScoreInput(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4, comma_token]])},
        labels=torch.tensor([[1, 2, 3, 4, comma_token]]),
        candidate_entry_label_mask=torch.tensor([[False, False, True, True, True]]),
        coord_label_mask=torch.tensor([[False, False, False, False, False]]),
    )
    terminal = TensorBranchScoreInput(
        model_inputs={"input_ids": torch.tensor([[1, 2, 5, 6, close_token]])},
        labels=torch.tensor([[1, 2, 5, 6, close_token]]),
        candidate_entry_label_mask=torch.tensor([[False, False, True, True, True]]),
        coord_label_mask=torch.tensor([[False, False, False, False, False]]),
    )

    serial = [
        score_tensor_retained(
            model=model_serial,
            model_inputs=item.model_inputs,
            labels=item.labels,
            candidate_entry_label_mask=item.candidate_entry_label_mask,
            coord_label_mask=item.coord_label_mask,
            coord_token_ids=coord_token_ids,
            logits_mode="supervised_suffix",
        )
        for item in (nonterminal, terminal)
    ]
    batched = score_tensor_batch_retained(
        model=model_batch,
        items=[nonterminal, terminal],
        coord_token_ids=coord_token_ids,
        logits_mode="supervised_suffix",
    )

    for expected, actual in zip(serial, batched, strict=True):
        assert torch.allclose(expected.score.detach(), actual.score.detach(), atol=1e-6)
        assert expected.tokens == actual.tokens == 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_preflight_nonterminal_gradient_targets_append_not_close(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    labels = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device=device)
    candidate_mask = torch.tensor(
        [[False, False, True, True, True]],
        dtype=torch.bool,
        device=device,
    )
    coord_mask = torch.zeros_like(candidate_mask)
    coord_token_ids = torch.tensor([8, 9], dtype=torch.long, device=device)
    comma_token = 4
    close_token = 5
    logits = torch.zeros((1, 5, 10), device=device, requires_grad=True)

    result = compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )
    (-result.score).backward()
    assert logits.grad is not None

    append_grad = logits.grad[0, 3, comma_token].abs().item()
    append_target_grad = logits.grad[0, 3, comma_token].item()
    close_competitor_grad = logits.grad[0, 3, close_token].item()
    suffix_grad = logits.grad[0, 4].abs().sum().item()

    assert append_grad > 0.0
    assert append_target_grad < 0.0
    assert close_competitor_grad > 0.0
    assert suffix_grad == pytest.approx(0.0)


def test_preflight_legacy_objective_contrast_would_train_wrong_boundary() -> None:
    rendered = render_indexed_object_list(
        [OBJECT_A, OBJECT_B, OBJECT_C],
        object_field_order="desc_first",
    )
    prefix = build_prefix_text(rendered, [0])
    repaired = build_candidate_entry_text(
        rendered,
        prefix_indices=[0],
        candidate_index=1,
    )
    legacy_full_text = prefix.text + rendered.entry_texts_by_index[1] + "]}"

    assert repaired.continuation_mode == "continue"
    assert prefix.text + repaired.text != legacy_full_text
    assert (prefix.text + repaired.text).endswith(", ")
    assert legacy_full_text.endswith("]}")


def test_preflight_candidate_balanced_is_optimized_not_logz_mp() -> None:
    scores = torch.tensor([-3.0, -12.0])
    lengths = torch.tensor([3.0, 12.0])

    result = compute_mp_pem_losses(
        scores=scores,
        pem_mode="disabled",
        candidate_lengths=lengths,
    )

    expected_candidate_balanced = torch.tensor(1.0)
    logz_diagnostic = -torch.logsumexp(scores, dim=0)
    assert torch.allclose(result.total_objective, expected_candidate_balanced)
    assert torch.allclose(result.loss_candidate_balanced, expected_candidate_balanced)
    assert torch.allclose(result.loss_mp, logz_diagnostic)
    assert not torch.allclose(result.total_objective, result.loss_mp)
