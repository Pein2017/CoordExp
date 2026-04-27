from __future__ import annotations

import re
from typing import Any, Mapping

import pytest
import torch

from src.data_collators.stage1_set_continuation_collator import (
    build_stage1_set_continuation_collator,
)
from src.trainers.stage1_set_continuation.branch_encoder import (
    encode_set_continuation_branch,
)


OBJECT_A = {
    "desc": "cat",
    "bbox_2d": [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ],
}
OBJECT_B = {
    "desc": "dog",
    "bbox_2d": [
        "<|coord_11|>",
        "<|coord_12|>",
        "<|coord_13|>",
        "<|coord_14|>",
    ],
}
OBJECT_C = {
    "desc": "bird",
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
        self.system_values_seen: list[str] = []

    def encode(
        self, rendered: Mapping[str, Any], return_length: bool = True
    ) -> dict[str, Any]:
        self.system_values_seen.append(str(getattr(self, "system", "")))
        messages = list(rendered["messages"])
        assistant = next(
            message for message in messages if message.get("role") == "assistant"
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


class _CloseStartMergeTokenizer(_FakeTokenizer):
    def encode_text(self, text: str) -> list[int]:
        return super().encode_text(text.replace("]", ""))


class _CloseStartMergeTemplate(_FakeTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _CloseStartMergeTokenizer()


def _raw_sample() -> dict[str, Any]:
    return {
        "input_ids": [1, 2],
        "labels": [-100, 2],
        "assistant_payload": {"objects": [OBJECT_A, OBJECT_B]},
        "objects": [{"ref": "metadata-not-entry"}],
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
        "sample_id": "sample-1",
        "base_idx": 0,
    }


def test_collator_preserves_assistant_payload_objects_and_metadata_sidecar() -> None:
    batch = build_stage1_set_continuation_collator()([_raw_sample()])

    meta = batch["set_continuation_meta"][0]
    assert meta["assistant_payload"]["objects"][0]["desc"] == "cat"
    assert meta["objects"][0]["ref"] == "metadata-not-entry"
    assert meta["messages"][0]["role"] == "system"
    assert meta["metadata"]["image_id"] == 1
    assert meta["sample_id"] == "sample-1"
    assert meta["base_idx"] == 0
    assert meta["dataset_label"] == "toy"
    assert batch["dataset_labels"] == ["toy"]
    assert batch["pack_num_samples"].tolist() == [1]


def test_collator_keeps_raw_extras_under_meta_not_model_top_level() -> None:
    def _base_collator(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "labels": torch.tensor([[-100, 2]], dtype=torch.long),
        }

    batch = build_stage1_set_continuation_collator(_base_collator)([_raw_sample()])

    assert "assistant_payload" not in batch
    assert "messages" not in batch
    assert "objects" not in batch
    assert "set_continuation_meta" in batch
    assert batch["input_ids"].shape == (1, 2)


def test_branch_encoder_builds_candidate_masks_and_restores_template_system() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0],
        candidate_index=1,
        object_field_order="desc_first",
    )

    assert branch.branch_inputs["input_ids"].shape == branch.labels.shape
    assert branch.candidate_entry_label_mask.shape == branch.labels.shape
    assert branch.candidate_entry_label_mask.sum().item() > 0
    assert branch.coord_label_mask.sum().item() == 4
    assert branch.non_coord_label_mask.sum().item() > 0
    assert branch.structural_close_start_mask.sum().item() >= 1
    assert branch.structural_close_sequence_mask.sum().item() >= (
        branch.structural_close_start_mask.sum().item()
    )
    assert "<|coord_11|>" in branch.rendered_text
    assert branch.prefix_indices == (0,)
    assert branch.candidate_index == 1
    assert template.system == "original-system"
    assert "system prompt" in template.system_values_seen


def test_branch_encoder_scores_nonterminal_candidate_continue_boundary() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]
    meta["assistant_payload"]["objects"] = [OBJECT_A, OBJECT_B, OBJECT_C]

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0],
        candidate_index=1,
        object_field_order="desc_first",
    )
    label_tokens = template.tokenizer.convert_ids_to_tokens(
        [int(value) for value in branch.labels.reshape(-1).tolist()]
    )
    scored_tokens = [
        token
        for token, is_scored in zip(
            label_tokens,
            branch.candidate_entry_label_mask.reshape(-1).tolist(),
            strict=True,
        )
        if is_scored
    ]

    assert branch.rendered_text.endswith(", ")
    assert scored_tokens[-1] == ","
    assert branch.structural_close_start_mask.sum().item() == 0
    assert branch.structural_close_sequence_mask.sum().item() == 0


def test_branch_encoder_falls_back_when_close_start_token_merges() -> None:
    template = _CloseStartMergeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0, 1],
        candidate_index=None,
        object_field_order="desc_first",
    )

    assert branch.structural_close_sequence_mask.sum().item() > 0
    assert branch.structural_close_start_mask.sum().item() == 1


def test_branch_encoder_closes_partial_prefix_at_valid_object_boundary() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[0],
        candidate_index=None,
        object_field_order="desc_first",
    )

    assert '"desc": "cat"' in branch.rendered_text
    assert '"desc": "dog"' not in branch.rendered_text
    assert branch.rendered_text.endswith("]}")
    assert ", ]}" not in branch.rendered_text
    assert branch.prefix_text + branch.continuation_text == branch.rendered_text
    assert branch.candidate_entry_label_mask.sum().item() == 0
    assert branch.structural_close_start_mask.sum().item() >= 1


def test_branch_encoder_rejects_missing_messages_with_actionable_error() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]
    meta.pop("messages")

    with pytest.raises(ValueError, match="requires messages"):
        encode_set_continuation_branch(
            meta=meta,
            template=template,
            prefix_indices=[],
            candidate_index=0,
        )


def test_branch_encoder_rejects_malformed_object_entries_without_reindexing() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]
    meta["assistant_payload"]["objects"] = [OBJECT_A, "bad-object", OBJECT_B]

    with pytest.raises(ValueError, match="bad index=1"):
        encode_set_continuation_branch(
            meta=meta,
            template=template,
            prefix_indices=[],
            candidate_index=2,
        )


def test_branch_encoder_uses_assistant_payload_not_metadata_objects() -> None:
    template = _FakeTemplate()
    meta = build_stage1_set_continuation_collator()([_raw_sample()])[
        "set_continuation_meta"
    ][0]

    branch = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=[],
        candidate_index=0,
    )

    assert "metadata-not-entry" not in branch.rendered_text
    assert '"desc": "cat"' in branch.rendered_text
