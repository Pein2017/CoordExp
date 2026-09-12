from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from probes.dora_owner_learning.geometric_dedup import (
    duplicate_unlikelihood,
    preservation_kl,
    trajectory_layout,
)


OBJECT_START = "<|object_ref_start|>"
OBJECT_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
EOS = "<|im_end|>"


class PieceTokenizer:
    """A deterministic CPU tokenizer whose ids are already the history."""

    def __init__(self, pieces: list[str], *, eos_id: int | None = None) -> None:
        self._id_to_piece = dict(enumerate(pieces, start=100))
        self._piece_to_id = {piece: token_id for token_id, piece in self._id_to_piece.items()}
        self.eos_token_id = eos_id
        self.decode_calls: list[list[int]] = []

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        ids = [int(token_id) for token_id in ids]
        self.decode_calls.append(ids)
        return "".join(self._id_to_piece[token_id] for token_id in ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._piece_to_id.get(token)


def _row(description: str, bbox: tuple[int, int, int, int]) -> list[str]:
    return [
        OBJECT_START,
        description,
        OBJECT_END,
        BOX_START,
        *(f"<|coord_{value}|>" for value in bbox),
        BOX_END,
    ]


def _layout(
    rows: list[list[str]],
    *,
    width: int = 1000,
    height: int = 1000,
    malformed: list[str] | None = None,
    with_eos: bool = False,
) -> tuple[dict, PieceTokenizer, list[int]]:
    pieces: list[str] = []
    for row in rows:
        pieces.extend(row)
    if malformed:
        pieces.extend(malformed)
    if with_eos:
        pieces.append(EOS)
    tokenizer = PieceTokenizer(pieces)
    ids = list(tokenizer._id_to_piece)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(EOS) if with_eos else None
    return (
        trajectory_layout(
            ids,
            tokenizer,
            image_width=width,
            image_height=height,
            row_id="image-1",
        ),
        tokenizer,
        ids,
    )


def _layout_from_pieces(
    pieces: list[str], *, width: int = 1000, height: int = 1000,
) -> tuple[dict, PieceTokenizer, list[int]]:
    tokenizer = PieceTokenizer(pieces)
    ids = list(tokenizer._id_to_piece)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(EOS)
    return (
        trajectory_layout(
            ids,
            tokenizer,
            image_width=width,
            image_height=height,
            row_id="image-1",
        ),
        tokenizer,
        ids,
    )


def test_pixel_iou_is_strict_at_exact_point_95_and_flags_point_96() -> None:
    exact, _, _ = _layout(
        [
            _row("a", (0, 0, 39, 39)),
            _row("b", (1, 0, 40, 39)),
        ]
    )
    # (38 * 39) / (40 * 39) is exactly .95 in pixel space.
    assert exact["duplicate_row_indices"] == []

    above, _, _ = _layout(
        [
            _row("a", (0, 0, 49, 49)),
            _row("b", (1, 0, 50, 49)),
        ]
    )
    # (48 * 49) / (50 * 49) is .96; equality at .95 must not be accepted.
    assert above["duplicate_row_indices"] == [1]
    assert len(above["duplicate_coordinate_positions"]) == 1


def test_non_square_pixel_rounding_not_bin_space_iou() -> None:
    # At 37x101, both distinct normalized boxes round to the same one-pixel
    # box.  Bin-space IoU is 1/3, while the consumer pixel IoU is 1.0.
    layout, _, _ = _layout(
        [
            _row("first", (0, 0, 20, 10)),
            _row("second", (10, 0, 30, 10)),
        ],
        width=37,
        height=101,
    )
    assert layout["row_spans"][0]["bbox_pixel_xyxy"] == [0, 0, 1, 1]
    assert layout["row_spans"][1]["bbox_pixel_xyxy"] == [0, 0, 1, 1]
    assert layout["duplicate_row_indices"] == [1]


def test_any_earlier_valid_row_including_already_flagged_row_and_once_each() -> None:
    layout, _, _ = _layout(
        [_row("cat", (100, 100, 300, 300)), _row("dog", (100, 100, 300, 300)), _row("fox", (100, 100, 300, 300))]
    )
    assert layout["valid_row_count"] == 3
    assert layout["duplicate_row_indices"] == [1, 2]
    assert layout["duplicate_valid_row_indices"] == [1, 2]
    assert len(layout["duplicate_coordinate_positions"]) == 2
    assert layout["row_spans"][1]["duplicate_reference_row_indices"] == [0]
    assert layout["row_spans"][2]["duplicate_reference_row_indices"] == [0, 1]
    # Category is the only varying semantic field; no GT is accepted or used.
    assert [row["description"] for row in layout["row_spans"]] == ["cat", "dog", "fox"]


def test_malformed_and_degenerate_rows_are_drops_not_negatives() -> None:
    malformed = _row("broken", (1, 2, 3, 4))[:-2] + [BOX_END]
    pieces = malformed + _row("usable", (100, 100, 200, 200))
    layout, _, _ = _layout_from_pieces(pieces)
    assert layout["valid_row_count"] == 1
    assert layout["parser_drops"] == 1
    assert layout["parser_drop_rows"][0]["reason"] == "malformed_object_span"
    assert layout["invalid_geometry_rows"] == []
    usable = layout["row_spans"][0]
    assert usable["generated_order"] == 1
    assert layout["duplicate_row_indices"] == []
    assert set(layout["kl_positions"]) == set(usable["token_positions"])

    pieces = _row("flat", (300, 200, 300, 400)) + _row("usable", (100, 100, 200, 200))
    layout, _, _ = _layout_from_pieces(pieces)
    assert layout["valid_row_count"] == 1
    assert layout["parser_drops"] == 1
    assert layout["invalid_geometry_rows"] == [0]
    assert layout["row_spans"][0]["generated_order"] == 1


def test_kl_excludes_entire_duplicate_row_but_includes_terminal_eos() -> None:
    layout, _, ids = _layout(
        [_row("keep", (100, 100, 300, 300)), _row("duplicate", (100, 100, 300, 300))],
        with_eos=True,
    )
    keep = layout["row_spans"][0]
    duplicate = layout["row_spans"][1]
    assert layout["duplicate_row_indices"] == [1]
    assert set(layout["kl_positions"]) == set(keep["token_positions"]) | {len(ids) - 1}
    assert not set(duplicate["token_positions"]) & set(layout["kl_positions"])
    assert layout["terminal_im_end_position"] == len(ids) - 1


def test_real_tokenizers_json_resolves_im_end_by_exact_token_id() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    path = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/tokenizer.json")
    if not path.is_file():
        pytest.skip(f"real tokenizer fixture is unavailable: {path}")
    tokenizer = tokenizers.Tokenizer.from_file(str(path))
    pieces = [
        "<|object_ref_start|>",
        *tokenizer.encode("cat", add_special_tokens=False).tokens,
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
        "<|im_end|>",
    ]
    ids = [tokenizer.token_to_id(piece) for piece in pieces]
    assert all(isinstance(token_id, int) for token_id in ids)
    layout = trajectory_layout(
        ids,
        tokenizer,
        image_width=1000,
        image_height=500,
        row_id="real-tokenizer",
    )
    assert tokenizer.token_to_id("<|im_end|>") == 151645
    assert layout["terminal_im_end_position"] == len(ids) - 1
    assert layout["kl_positions"][-1] == len(ids) - 1


def test_layout_keeps_original_token_indices_and_is_json_serializable() -> None:
    layout, tokenizer, ids = _layout([_row("kept", (10, 10, 20, 20)), _row("dup", (10, 10, 20, 20))])
    assert layout["duplicate_coordinate_positions"][0] == layout["row_spans"][1]["coordinate_positions"]
    assert all(position < len(ids) for position in layout["duplicate_coordinate_positions"][0])
    # The tokenizer is only decoded; the helper never calls encode/tokenize.
    assert all(call == ids or len(call) == 1 for call in tokenizer.decode_calls)
    json.dumps(layout, allow_nan=False)


def test_duplicate_unlikelihood_uses_exact_coordinate_indices_and_all_valid_denominator() -> None:
    torch.manual_seed(4)
    logits = torch.randn(8, 11, requires_grad=True)
    targets = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
    layout = {
        "valid_row_count": 3,
        "duplicate_coordinate_positions": [[1, 3, 5, 7]],
    }
    result = duplicate_unlikelihood(logits, targets, layout)
    selected = logits[[1, 3, 5, 7]].log_softmax(-1).gather(
        1, targets[[1, 3, 5, 7], None]
    ).squeeze(1)
    q = selected.mean().exp()
    expected = -torch.log((1 - q + 1e-8) / (1 + 1e-8)) / 3
    torch.testing.assert_close(result, expected)
    result.backward()
    assert all(float(logits.grad[position].abs().sum()) > 0 for position in [1, 3, 5, 7])
    assert all(float(logits.grad[position].abs().sum()) == 0 for position in [0, 2, 4, 6])


def test_duplicate_unlikelihood_multirow_matches_scalar_reference_value_and_gradient() -> None:
    torch.manual_seed(19)
    base = torch.randn(13, 17)
    targets = torch.randint(0, 17, (13,), dtype=torch.long)
    layout = {
        "valid_row_count": 5,
        "duplicate_coordinate_positions": [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 12]],
    }
    vector_logits = base.clone().requires_grad_()
    scalar_logits = base.clone().requires_grad_()
    vector_value = duplicate_unlikelihood(vector_logits, targets, layout)

    scalar_rows = []
    for positions in layout["duplicate_coordinate_positions"]:
        selected = scalar_logits[positions].log_softmax(-1).gather(
            1, targets[positions, None]
        ).squeeze(1)
        q = selected.mean().exp()
        scalar_rows.append(-torch.log((1 - q + 1e-8) / (1 + 1e-8)))
    scalar_value = torch.stack(scalar_rows).sum() / layout["valid_row_count"]
    torch.testing.assert_close(vector_value, scalar_value)
    vector_value.backward()
    scalar_value.backward()
    torch.testing.assert_close(vector_logits.grad, scalar_logits.grad)


def test_zero_eligible_and_zero_valid_images_return_finite_differentiable_zero() -> None:
    logits = torch.zeros(4, 5, requires_grad=True)
    targets = torch.zeros(4, dtype=torch.long)
    for layout in (
        {"valid_row_count": 4, "duplicate_coordinate_positions": []},
        {"valid_row_count": 0, "duplicate_coordinate_positions": []},
    ):
        loss = duplicate_unlikelihood(logits, targets, layout)
        assert torch.isfinite(loss) and float(loss) == 0.0 and loss.requires_grad
        loss.backward(retain_graph=True)
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def test_unlikelihood_gradient_is_positive_on_sampled_coords_and_step_lowers_confidence() -> None:
    logits = torch.zeros(4, 3, requires_grad=True)
    targets = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        logits[:, 0] = 2.0
    before_q = logits.log_softmax(-1)[:, 0].mean().exp().item()
    loss = duplicate_unlikelihood(
        logits,
        targets,
        {"valid_row_count": 1, "duplicate_coordinate_positions": [[0, 1, 2, 3]]},
    )
    loss.backward()
    assert all(float(logits.grad[index, 0]) > 0 for index in range(4))
    with torch.no_grad():
        logits -= 0.1 * logits.grad
    after_q = logits.log_softmax(-1)[:, 0].mean().exp().item()
    assert after_q < before_q


def test_preservation_kl_is_full_vocab_forward_kl_and_reference_is_detached() -> None:
    logits = torch.randn(5, 7, requires_grad=True)
    reference = torch.log_softmax(torch.randn(5, 7), dim=-1).detach().requires_grad_()
    positions = [0, 3, 4]
    result = preservation_kl(logits, reference, positions)
    expected = (
        reference.detach()[positions].exp()
        * (reference.detach()[positions] - logits[positions].log_softmax(-1))
    ).sum(-1).mean()
    torch.testing.assert_close(result, expected)
    result.backward()
    assert reference.grad is None
    assert float(logits.grad[[0, 3, 4]].abs().sum()) > 0
    assert float(logits.grad[[1, 2]].abs().sum()) == 0


def test_preservation_kl_zero_selection_is_finite_differentiable_zero() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    reference = torch.log_softmax(torch.randn(0, 4), dim=-1)
    result = preservation_kl(logits, reference, [])
    assert torch.isfinite(result) and float(result) == 0.0 and result.requires_grad
    result.backward()
    assert float(logits.grad.abs().sum()) == 0.0


@pytest.mark.parametrize("bad_positions", ([0, 0], [[0, 1, 2]], [10]))
def test_loss_helpers_reject_ambiguous_or_out_of_range_positions(bad_positions) -> None:
    logits = torch.zeros(4, 3, requires_grad=True)
    targets = torch.zeros(4, dtype=torch.long)
    if isinstance(bad_positions, list) and bad_positions and isinstance(bad_positions[0], list):
        with pytest.raises(ValueError):
            duplicate_unlikelihood(
                logits,
                targets,
                {"valid_row_count": 1, "duplicate_coordinate_positions": bad_positions},
            )
    else:
        with pytest.raises(ValueError):
            preservation_kl(logits, torch.zeros(4, 3), bad_positions)
