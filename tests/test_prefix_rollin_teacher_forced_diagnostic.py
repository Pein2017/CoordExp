from __future__ import annotations

import json
import re

import pytest
import torch

from src.analysis.prefix_rollin_teacher_forced_diagnostic import (
    GeneratedPrefixCase,
    _generated_prefix_case_for_record,
    _normalize_prefix_probe_shard,
    _normalize_prefix_modes,
    _record_selected_for_prefix_probe,
    _resolve_k_values,
    _score_forced_prefix_boundary_logits,
    compute_prefix_rollin_position_delta,
    merge_prefix_rollin_shards,
    score_prefix_rollin_logits,
    summarize_prefix_rollin_probe_rows,
)
from src.detection.data import ObjectOrderingPlan, normalize_detection_row
from src.detection.objective import build_compact_prefix_rollin_example

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class FakeTokenizer:
    eos_token = "<|endoftext|>"
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

    def decode(self, token_ids: list[int]) -> str:
        reverse = {value: key for key, value in self._token_to_id.items()}
        return "".join(reverse.get(int(token_id), f"<token:{int(token_id)}>") for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}<|im_end|>\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        return self(rendered, return_offsets_mapping=True, add_special_tokens=False)[
            "input_ids"
        ]

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
            token_id = self._token_to_id.setdefault(token_text, len(self._token_to_id) + 1)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end
        return {"input_ids": input_ids, "offset_mapping": offsets}


def _raw_row() -> dict[str, object]:
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


def _example(*, k: int, object_count: int = 3):
    raw = _raw_row()
    raw["objects"] = raw["objects"][:object_count]
    sample = normalize_detection_row(
        raw=parse_raw_row_for_test(raw),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=123,
            seed_source="unit",
        ),
    )
    return build_compact_prefix_rollin_example(
        objects=sample.objects,
        rollin_order=sample.objects,
        k=k,
        tokenizer=FakeTokenizer(),
        normalized_sample=sample,
        system_prompt="You are a detector.",
        user_content="<image>\nDetect every object.",
    )


def parse_raw_row_for_test(raw):
    from src.detection.data import parse_raw_detection_row

    return parse_raw_detection_row(raw)


def _branch_row(rows):
    return next(row for row in rows if row["branch_kind"] == "valid_next_object")


def _eos_row(rows):
    return next(row for row in rows if row["branch_kind"] == "semantic_eos")


def test_branch_probe_uses_next_token_shift_and_valid_mass_margin() -> None:
    example = _example(k=0, object_count=3)
    branch_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
        and target.valid_token_ids
    )
    eos_id = example.stop_contract.im_end_token_id
    vocab_size = max((*branch_target.valid_token_ids, eos_id, *example.input_ids)) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    prediction_row = branch_target.position - 1
    for token_id in branch_target.valid_token_ids:
        logits[0, prediction_row, token_id] = 4.0
    logits[0, prediction_row, eos_id] = 1.5

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )
    row = _branch_row(rows)

    log_probs = torch.log_softmax(logits[0, prediction_row], dim=-1)
    expected_valid_mass = torch.logsumexp(
        log_probs[torch.tensor(branch_target.valid_token_ids)],
        dim=-1,
    )
    expected_continue_logsumexp = torch.logsumexp(
        logits[0, prediction_row, torch.tensor(branch_target.valid_token_ids)],
        dim=-1,
    )
    expected_margin = expected_valid_mass - log_probs[eos_id]
    expected_logit_margin = (
        logits[0, prediction_row, list(branch_target.valid_token_ids)].max()
        - logits[0, prediction_row, eos_id]
    )
    assert row["diagnostic"] == "forced_prefix_continue_vs_eos_v0"
    assert row["prefix_mode"] == "gt_prefix_entry_after_separator"
    assert row["prefix_k"] == 0
    assert row["gt_count"] == 3
    assert row["remaining_gt_count"] == 3
    assert row["target_position"] == branch_target.position
    assert row["processor_position"] == branch_target.position
    assert row["continue_logsumexp"] == pytest.approx(
        expected_continue_logsumexp.item()
    )
    assert row["valid_mass"] == pytest.approx(expected_valid_mass.exp().item())
    assert row["valid_logprob_mass"] == pytest.approx(expected_valid_mass.item())
    assert row["eos_logprob"] == pytest.approx(log_probs[eos_id].item())
    assert row["continue_minus_eos_margin"] == pytest.approx(expected_margin.item())
    assert row["margin_valid_mass_minus_eos_logprob"] == pytest.approx(
        expected_margin.item()
    )
    assert row["margin_valid_max_logit_minus_eos_logit"] == pytest.approx(
        expected_logit_margin.item()
    )


def test_singleton_remaining_object_is_still_reported_as_valid_next_branch() -> None:
    example = _example(k=2, object_count=3)
    remaining = {str(item) for item in example.rollin_state.remaining}
    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in remaining
    )
    assert target.kind == "hard_ce"
    assert len(target.valid_token_ids) == 1

    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *example.input_ids) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )
    row = _branch_row(rows)

    assert row["target_kind"] == "hard_ce"
    assert row["valid_next_object_branch_present"] is True
    assert row["valid_token_ids"] == list(target.valid_token_ids)


def test_gt_prefix_reports_separator_boundary_before_next_object() -> None:
    example = _example(k=1, object_count=3)
    eos_id = example.stop_contract.im_end_token_id
    vocab_size = max(eos_id, *example.input_ids) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )
    boundary_row = next(
        row
        for row in rows
        if row["prefix_mode"] == "gt_prefix_free_boundary"
    )

    assert boundary_row["boundary_kind"] == "separator_before_next_object"
    assert boundary_row["teacher_token_id"] == example.input_ids[
        boundary_row["target_position"]
    ]
    assert boundary_row["valid_token_ids"] == [boundary_row["teacher_token_id"]]


def test_full_prefix_state_reports_only_im_end() -> None:
    example = _example(k=3, object_count=3)
    eos_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.teacher_token_id == example.stop_contract.im_end_token_id
    )
    vocab_size = max(eos_target.teacher_token_id, *example.input_ids) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    logits[0, eos_target.position - 1, eos_target.teacher_token_id] = 5.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )

    assert not any(row["branch_kind"] == "valid_next_object" for row in rows)
    eos_row = _eos_row(rows)
    assert eos_row["valid_next_object_branch_present"] is False
    assert eos_row["teacher_token_id"] == example.stop_contract.im_end_token_id
    assert eos_row["prefix_k"] == 3
    assert eos_row["remaining_gt_count"] == 0
    assert eos_row["continue_minus_eos_margin"] is None


def test_processor_padding_rebase_preserves_target_identity() -> None:
    example = _example(k=0, object_count=3)
    pad_id = 0
    processor_ids = (pad_id, pad_id, *example.input_ids)
    delta = compute_prefix_rollin_position_delta(
        processor_input_ids=processor_ids,
        example=example,
    )
    assert delta == 2

    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
    )
    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *processor_ids) + 1
    logits = torch.zeros((1, len(processor_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position + delta - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=processor_ids,
    )
    assert _branch_row(rows)["processor_position"] == target.position + delta


def test_processor_vision_placeholder_rebase_can_align_by_assistant_suffix() -> None:
    example = _example(k=0, object_count=3)
    assistant_start = example.tokenized.assistant_token_span.start
    processor_ids = (999, 998, *example.input_ids[assistant_start:])
    delta = compute_prefix_rollin_position_delta(
        processor_input_ids=processor_ids,
        example=example,
    )
    assert delta == 2 - assistant_start

    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
    )
    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *processor_ids) + 1
    logits = torch.zeros((1, len(processor_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position + delta - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=processor_ids,
    )
    assert _branch_row(rows)["processor_position"] == target.position + delta


def test_summarize_prefix_rollin_probe_rows_reports_margin_health() -> None:
    rows = [
        {
            "branch_kind": "valid_next_object",
            "prefix_mode": "gt_prefix",
            "rollin_k": 0,
            "prefix_k": 0,
            "continue_minus_eos_margin": 2.0,
            "margin_valid_mass_minus_eos_logprob": 2.0,
            "valid_mass": 0.80,
        },
        {
            "branch_kind": "valid_next_object",
            "prefix_mode": "generated_prefix_free_boundary",
            "rollin_k": 1,
            "prefix_k": 1,
            "continue_minus_eos_margin": -1.0,
            "margin_valid_mass_minus_eos_logprob": -1.0,
            "valid_mass": 0.25,
        },
        {
            "branch_kind": "semantic_eos",
            "rollin_k": 2,
            "prefix_k": 2,
            "eos_logprob": -0.2,
        },
    ]

    summary = summarize_prefix_rollin_probe_rows(rows)

    assert summary["row_count"] == 3
    assert summary["valid_next_object_row_count"] == 2
    assert summary["semantic_eos_row_count"] == 1
    assert summary["valid_margin_mean"] == pytest.approx(0.5)
    assert summary["valid_margin_le_zero_rate"] == pytest.approx(0.5)
    assert summary["continue_margin_mean"] == pytest.approx(0.5)
    assert summary["continue_margin_le_zero_rate"] == pytest.approx(0.5)
    assert summary["valid_mass_mean"] == pytest.approx(0.525)
    assert summary["continue_margin_by_prefix_k"]["0"][
        "continue_margin_mean"
    ] == pytest.approx(2.0)
    assert summary["continue_margin_by_prefix_k"]["1"][
        "continue_margin_le_zero_rate"
    ] == pytest.approx(1.0)
    assert summary["continue_margin_by_prefix_mode"]["gt_prefix"][
        "continue_margin_mean"
    ] == pytest.approx(2.0)
    assert summary["continue_margin_by_prefix_mode"][
        "generated_prefix_free_boundary"
    ]["continue_margin_mean"] == pytest.approx(-1.0)
    assert summary["rollin_k_values"] == [0, 1, 2]
    assert summary["prefix_k_values"] == [0, 1, 2]


def test_resolve_k_values_every_scores_full_prefix_curve() -> None:
    assert _resolve_k_values(["every"], object_count=3) == [0, 1, 2, 3]
    assert _resolve_k_values(["0", "every", "all"], object_count=2) == [0, 1, 2]


def test_generated_prefix_case_prefers_exact_trace_tokens_without_im_end() -> None:
    decode_rows = {
        0: {
            "raw_ends_with_im_end": True,
            "raw_output_json": {
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [
                            "<|coord_1|>",
                            "<|coord_2|>",
                            "<|coord_3|>",
                            "<|coord_4|>",
                        ],
                    }
                ]
            },
        }
    }
    trace_rows = {
        0: {
            "generated_token_text": [
                "<|object_ref_start|>",
                "cat",
                "<|box_start|>",
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_3|>",
                "<|coord_4|>",
                "<|im_end|>",
                "ignored",
            ]
        }
    }

    case = _generated_prefix_case_for_record(
        record_idx=0,
        decode_rows=decode_rows,
        trace_rows=trace_rows,
    )

    assert case == GeneratedPrefixCase(
        prefix_text=(
            "<|object_ref_start|>cat<|box_start|>"
            "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        ),
        pred_count=1,
        prefix_text_source="pred_token_trace.generated_token_text",
        raw_ends_with_im_end=True,
    )


def test_generated_prefix_boundary_scores_object_ref_continue_vs_eos() -> None:
    tokenizer = FakeTokenizer()
    object_ref_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    processor_ids = [99, object_ref_id, 100]
    logits = torch.zeros((1, len(processor_ids), 101), dtype=torch.float32)
    logits[0, 0, object_ref_id] = 6.0
    logits[0, 0, eos_id] = 2.0

    row = _score_forced_prefix_boundary_logits(
        logits=logits,
        tokenizer=tokenizer,
        processor_input_ids=processor_ids,
        target_position=1,
        teacher_token_id=object_ref_id,
        valid_token_ids=(object_ref_id,),
        eos_token_id=eos_id,
        branch_kind="valid_next_object",
        top_k=3,
        metadata={
            "diagnostic": "forced_prefix_continue_vs_eos_v0",
            "probe_family": "generated_prefix_teacher_forced",
            "prefix_mode": "generated_prefix_count_depth",
            "prefix_k": 2,
            "gt_count": 4,
            "remaining_gt_count": 2,
        },
    )

    assert row["prefix_mode"] == "generated_prefix_count_depth"
    assert row["teacher_token_text"] == "<|object_ref_start|>"
    assert row["valid_token_ids"] == [object_ref_id]
    assert row["eos_token_id"] == eos_id
    assert row["continue_logsumexp"] == pytest.approx(6.0)
    assert row["continue_minus_eos_margin"] == pytest.approx(4.0)
    expected_valid_mass = torch.softmax(logits[0, 0], dim=-1)[object_ref_id]
    assert row["valid_mass"] == pytest.approx(expected_valid_mass.item())


def test_normalize_prefix_modes_accepts_both_alias() -> None:
    assert _normalize_prefix_modes(["both"]) == {"gt_prefix", "generated_prefix"}
    assert _normalize_prefix_modes(["self", "clean"]) == {
        "generated_prefix",
        "gt_prefix",
    }

def test_prefix_probe_shard_selection_keeps_full_record_curves() -> None:
    assert _record_selected_for_prefix_probe(
        0, limit=10, shard_index=0, num_shards=2
    )
    assert not _record_selected_for_prefix_probe(
        1, limit=10, shard_index=0, num_shards=2
    )
    assert _record_selected_for_prefix_probe(
        9, limit=10, shard_index=1, num_shards=2
    )
    assert not _record_selected_for_prefix_probe(
        10, limit=10, shard_index=0, num_shards=2
    )

def test_prefix_probe_shard_normalization_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        _normalize_prefix_probe_shard(shard_index=2, num_shards=2)
    with pytest.raises(ValueError):
        _normalize_prefix_probe_shard(shard_index=0, num_shards=0)

def _write_prefix_probe_shard(
    shards_dir,
    *,
    shard_index: int,
    num_shards: int,
    rows: list[dict[str, object]],
) -> None:
    label = f"shard_{shard_index:03d}-of-{num_shards:03d}"
    shard_dir = shards_dir / label
    shard_dir.mkdir(parents=True)
    per_case_path = shard_dir / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        **summarize_prefix_rollin_probe_rows(rows),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": label,
        "selected_record_count": len(
            {
                int(
                    row["source_line_idx"]
                    if "source_line_idx" in row
                    else row["record_idx"]
                )
                for row in rows
            }
        ),
        "per_case_jsonl": str(per_case_path),
    }
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

def _merge_fixture_row(record_idx: int, *, margin: float) -> dict[str, object]:
    return {
        "record_idx": record_idx,
        "source_line_idx": record_idx,
        "prefix_mode": "gt_prefix",
        "k": 0,
        "prefix_k": 0,
        "rollin_k": 0,
        "boundary_kind": "object_ref_after_forced_separator",
        "branch_kind": "valid_next_object",
        "position": 5,
        "target_position": 5,
        "object_instance_id": f"obj-{record_idx}",
        "continue_minus_eos_margin": margin,
        "margin_valid_mass_minus_eos_logprob": margin,
        "valid_mass": 0.5,
    }

def test_merge_prefix_rollin_shards_concatenates_rows_and_summaries(tmp_path) -> None:
    shards_dir = tmp_path / "shards"
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=2,
        rows=[_merge_fixture_row(0, margin=1.0)],
    )
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=1,
        num_shards=2,
        rows=[_merge_fixture_row(1, margin=-1.0)],
    )
    output_dir = tmp_path / "merged"

    per_case_path, summary_path = merge_prefix_rollin_shards(
        shards_dir=shards_dir,
        output_dir=output_dir,
        expected_shards=2,
    )

    assert per_case_path == output_dir / "per_case.jsonl"
    assert summary_path == output_dir / "summary.json"
    merged_rows = [
        json.loads(line)
        for line in per_case_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_idx"] for row in merged_rows] == [0, 1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["selected_record_count"] == 2
    assert summary["shard_index"] is None
    assert summary["num_shards"] == 2
    assert summary["shard_label"] is None
    merge_summary = json.loads(
        (output_dir / "merge_summary.json").read_text(encoding="utf-8")
    )
    assert merge_summary["shard_labels"] == [
        "shard_000-of-002",
        "shard_001-of-002",
    ]
    assert merge_summary["row_counts_by_shard"] == {
        "shard_000-of-002": 1,
        "shard_001-of-002": 1,
    }
    assert merge_summary["selected_record_count"] == 2
    assert len(merge_summary["source_summaries"]) == 2

def test_merge_prefix_rollin_shards_rejects_duplicate_case_keys(tmp_path) -> None:
    shards_dir = tmp_path / "shards"
    duplicate_row = _merge_fixture_row(0, margin=1.0)
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=2,
        rows=[duplicate_row],
    )
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=1,
        num_shards=2,
        rows=[{**duplicate_row, "shard_index": 1}],
    )

    with pytest.raises(ValueError, match="duplicate"):
        merge_prefix_rollin_shards(
            shards_dir=shards_dir,
            output_dir=tmp_path / "merged",
            expected_shards=2,
        )

def test_merge_prefix_rollin_shards_prefers_local_per_case_over_external_summary_path(
    tmp_path,
) -> None:
    shards_dir = tmp_path / "shards"
    local_row = _merge_fixture_row(0, margin=1.0)
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=1,
        rows=[local_row],
    )
    external_per_case = tmp_path / "stale_external_per_case.jsonl"
    external_per_case.write_text(
        json.dumps(_merge_fixture_row(99, margin=-9.0)) + "\n",
        encoding="utf-8",
    )
    summary_path = shards_dir / "shard_000-of-001" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["per_case_jsonl"] = str(external_per_case)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    per_case_path, _ = merge_prefix_rollin_shards(
        shards_dir=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shards=1,
    )

    merged_rows = [
        json.loads(line)
        for line in per_case_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["source_line_idx"] for row in merged_rows] == [0]

def test_merge_prefix_rollin_shards_rejects_external_per_case_without_local_file(
    tmp_path,
) -> None:
    shards_dir = tmp_path / "shards"
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=1,
        rows=[_merge_fixture_row(0, margin=1.0)],
    )
    external_per_case = tmp_path / "stale_external_per_case.jsonl"
    external_per_case.write_text(
        json.dumps(_merge_fixture_row(99, margin=-9.0)) + "\n",
        encoding="utf-8",
    )
    shard_dir = shards_dir / "shard_000-of-001"
    (shard_dir / "per_case.jsonl").unlink()
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["per_case_jsonl"] = str(external_per_case)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stale_prefix_rollin_shard_per_case"):
        merge_prefix_rollin_shards(
            shards_dir=shards_dir,
            output_dir=tmp_path / "merged",
            expected_shards=1,
        )

def test_merge_prefix_rollin_shards_accepts_source_line_idx_without_record_idx(
    tmp_path,
) -> None:
    shards_dir = tmp_path / "shards"
    row = _merge_fixture_row(7, margin=1.0)
    row.pop("record_idx")
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=1,
        rows=[row],
    )

    per_case_path, _ = merge_prefix_rollin_shards(
        shards_dir=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shards=1,
    )

    merged_rows = [
        json.loads(line)
        for line in per_case_path.read_text(encoding="utf-8").splitlines()
    ]
    assert merged_rows[0]["source_line_idx"] == 7
    assert "record_idx" not in merged_rows[0]


def test_merge_prefix_rollin_shards_accepts_semantic_eos_rows_without_boundary_kind(
    tmp_path,
) -> None:
    shards_dir = tmp_path / "shards"
    rows = []
    for prefix_k in range(2):
        row = _merge_fixture_row(0, margin=1.0)
        row.pop("boundary_kind")
        row["prefix_k"] = prefix_k
        row["rollin_k"] = prefix_k
        row["branch_kind"] = "semantic_eos"
        row["target_token_role"] = "terminal"
        row["target_semantic_role"] = "chat_stop"
        row["target_object_instance_id"] = None
        row["object_instance_id"] = None
        rows.append(row)
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=1,
        rows=rows,
    )

    per_case_path, _ = merge_prefix_rollin_shards(
        shards_dir=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shards=1,
    )

    merged_rows = [
        json.loads(line)
        for line in per_case_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["prefix_k"] for row in merged_rows] == [0, 1]


def test_merge_prefix_rollin_shards_rejects_missing_required_key_fields(
    tmp_path,
) -> None:
    shards_dir = tmp_path / "shards"
    row = _merge_fixture_row(0, margin=1.0)
    row.pop("prefix_mode")
    _write_prefix_probe_shard(
        shards_dir,
        shard_index=0,
        num_shards=1,
        rows=[row],
    )

    with pytest.raises(ValueError, match="missing_prefix_rollin_merge_key_fields"):
        merge_prefix_rollin_shards(
            shards_dir=shards_dir,
            output_dir=tmp_path / "merged",
            expected_shards=1,
        )
