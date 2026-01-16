from __future__ import annotations

import re

import pytest
import torch

from src.trainers.rollout_matching_sft import (
    GTObject,
    _build_labels_and_coord_targets_for_sample,
    _sinkhorn_barycentric_targets,
    hungarian_match_maskiou,
    parse_rollout_for_matching,
)


class _DummyTokenizerRM:
    """Minimal tokenizer stub for token-aligned rollout parsing tests.

    - coord tokens <|coord_k|> map to ids 0..999
    - all other characters map to ids >= 1000 (1 char = 1 token)
    - supports a special fused token id that decodes to '}}'
    """

    _coord_re = re.compile(r"<\|coord_(\d{1,4})\|>")

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._next = 1000
        # Fused '}}' token
        self.fused_brace_id = 2000
        self._id_to_piece[self.fused_brace_id] = "}}"

    def convert_tokens_to_ids(self, tokens):
        out = []
        for tok in tokens:
            m = self._coord_re.fullmatch(tok)
            if m:
                out.append(int(m.group(1)))
            else:
                out.append(-1)
        return out

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, **kwargs):
        pieces = []
        for tid in ids:
            tid = int(tid)
            if 0 <= tid <= 999:
                pieces.append(f"<|coord_{tid}|>")
            else:
                pieces.append(self._id_to_piece.get(tid, ""))
        return "".join(pieces)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        # Special-case token-internal cuts: allow encoding of single braces.
        if text == "}}":
            return [self.fused_brace_id]
        if text and text in self._char_to_id:
            return [self._char_to_id[text]]

        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<|coord_", i):
                j = text.find("|>", i)
                if j < 0:
                    raise ValueError("unterminated coord token in dummy encode")
                tok = text[i : j + 2]
                m = self._coord_re.fullmatch(tok)
                if not m:
                    raise ValueError(f"bad coord token: {tok}")
                ids.append(int(m.group(1)))
                i = j + 2
                continue
            ch = text[i]
            tid = self._char_to_id.get(ch)
            if tid is None:
                tid = self._next
                self._next += 1
                self._char_to_id[ch] = tid
                self._id_to_piece[tid] = ch
            ids.append(tid)
            i += 1
        return ids


def test_rollout_parse_preserves_appearance_order_and_prefix_cut():
    tok = _DummyTokenizerRM()
    text = (
        '{"object_10": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, '
        '"object_2": {"desc": "b", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"]}}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)

    assert [o.index for o in parsed.valid_objects] == [10, 2]
    assert parsed.max_object_index_in_prefix == 10
    # Prefix ends after last object value dict; top-level JSON is left open for append.
    assert parsed.prefix_text.endswith("}")
    assert not parsed.prefix_text.endswith("}}")


def test_rollout_parse_drops_invalid_objects_without_repair():
    tok = _DummyTokenizerRM()
    # object_2 has only 3 bbox coords -> invalid and dropped.
    text = (
        '{"object_1": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, '
        '"object_2": {"desc": "b", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>"]}, '
        '"object_3": {"desc": "c", "bbox_2d": ["<|coord_8|>", "<|coord_9|>", "<|coord_10|>", "<|coord_11|>"]}}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert [o.index for o in parsed.valid_objects] == [1, 3]
    assert parsed.dropped_invalid >= 1


def test_rollout_parse_poly_captures_coord_indices_through_nested_arrays():
    tok = _DummyTokenizerRM()
    text = (
        '{"object_1": {"desc": "p", "poly": [['
        '"<|coord_1|>", "<|coord_2|>"], ["<|coord_3|>", "<|coord_4|>"], '
        '["<|coord_5|>", "<|coord_6|>"], ["<|coord_7|>", "<|coord_8|>"]]}}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert len(parsed.valid_objects) == 1
    obj = parsed.valid_objects[0]
    assert obj.geom_type == "poly"
    assert len(obj.coord_token_indices) == 8


def test_rollout_parse_truncated_tail_is_suffix_trimmed():
    tok = _DummyTokenizerRM()
    # Truncated mid-object_2: only object_1 should remain.
    text = (
        '{"object_1": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, '
        '"object_2": {"desc": "b", "bbox_2d": ["<|coord_5|>"'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.truncated is True
    assert [o.index for o in parsed.valid_objects] == [1]
    assert parsed.prefix_text.endswith("}")


def test_rollout_parse_handles_fused_closing_braces_token_internal_cut():
    tok = _DummyTokenizerRM()
    base = '{"object_1": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}}'
    ids = tok.encode(base, add_special_tokens=False)
    # Replace the final two '}' character tokens with a single fused '}}' token.
    assert tok.decode(ids[-2:], clean_up_tokenization_spaces=False) == "}}"
    ids_fused = ids[:-2] + [tok.fused_brace_id]

    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids_fused)
    assert [o.index for o in parsed.valid_objects] == [1]
    # Prefix should not include the top-level closing brace.
    assert parsed.prefix_text.endswith("}")
    assert not parsed.prefix_text.endswith("}}")


def test_rollout_parse_strips_im_end_hard_stop():
    tok = _DummyTokenizerRM()
    base = (
        '{"object_1": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}}'
        "<|im_end|>GARBAGE"
    )
    ids = tok.encode(base, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert "<|im_end|>" not in parsed.response_text
    assert "<|im_end|>" not in parsed.prefix_text
    # Prefix ends after last object value dict; top-level JSON is left open for append.
    assert parsed.prefix_text.endswith("}")


def test_rollout_parse_fallback_when_no_open_brace():
    tok = _DummyTokenizerRM()
    text = "NOT_JSON<|im_end|>"
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.prefix_text == "{"
    assert parsed.valid_objects == []
    assert parsed.truncated is True


def test_hungarian_matching_with_gating_and_dummy_augmentation():
    preds = [
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[100, 100, 200, 200], desc=""),
        GTObject(index=2, geom_type="bbox_2d", points_norm1000=[500, 500, 600, 600], desc=""),
    ]
    gts = [
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[100, 100, 200, 200], desc="gt1"),
        GTObject(index=2, geom_type="bbox_2d", points_norm1000=[500, 500, 600, 600], desc="gt2"),
    ]
    match = hungarian_match_maskiou(
        preds=preds,
        gts=gts,
        top_k=2,
        gate_threshold=0.5,
        mask_resolution=64,
        fp_cost=1.0,
        fn_cost=1.0,
    )
    assert sorted(match.matched_pairs) == [(0, 0), (1, 1)]
    assert match.fn_gt_indices == []
    assert match.fp_pred_indices == []

    # With disjoint boxes and a non-trivial gate, nothing matches -> all FN and FP.
    preds_far = [
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[0, 0, 50, 50], desc=""),
        GTObject(index=2, geom_type="bbox_2d", points_norm1000=[900, 900, 950, 950], desc=""),
    ]
    match2 = hungarian_match_maskiou(
        preds=preds_far,
        gts=gts,
        top_k=2,
        gate_threshold=0.5,
        mask_resolution=64,
        fp_cost=1.0,
        fn_cost=1.0,
    )
    assert match2.matched_pairs == []
    assert match2.fn_gt_indices == [0, 1]
    assert match2.fp_pred_indices == [0, 1]


def test_sinkhorn_barycentric_targets_sanity_poly_involved():
    # Pred and GT are the same square -> barycentric targets should be close.
    pred = torch.tensor([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    gt = pred.clone()
    out = _sinkhorn_barycentric_targets(
        pred_points=pred.numpy(),
        gt_points=gt.numpy(),
        epsilon=1.0,
        iters=50,
        cost="l2",
    )
    assert out.shape == (4, 2)
    assert abs(out - gt.numpy()).max() < 2.0


def test_loss_masking_invariants_prefix_and_tail():
    # Build a synthetic sequence with:
    # - prompt_len=5
    # - prefix_len=3 (positions 5..7)
    # - train_len=6  (assistant content positions 5..10)
    # tail is positions [8..10)
    prompt_len = 5
    prefix_len = 3
    train_len = 6
    # coord tokens ids for this test:
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}

    # Sequence length 12, with a coord token in prefix and one in tail.
    input_ids = torch.tensor(
        [1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long
    )
    labels, coord_pos, coord_bins, coord_is_prefix = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        train_len=train_len,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],  # assistant-local index -> full position 6
        prefix_coord_target_bins=[123],
    )

    # Prefix tokens contribute no CE.
    assert labels[5].item() == -100
    assert labels[6].item() == -100  # coord token
    assert labels[7].item() == -100

    # Tail: non-coord tokens contribute to CE, coord tokens do not.
    assert labels[8].item() == 8
    assert labels[9].item() == -100  # coord token
    assert labels[10].item() == 9

    # Coord supervision includes both prefix slot and tail coord token.
    assert 6 in coord_pos
    assert 9 in coord_pos
    # Prefix uses provided target bin; tail uses bin derived from token id.
    prefix_idx = coord_pos.index(6)
    tail_idx = coord_pos.index(9)
    assert coord_bins[prefix_idx] == 123
    assert coord_bins[tail_idx] == 20
    assert coord_is_prefix[prefix_idx] is True
    assert coord_is_prefix[tail_idx] is False


def test_loss_masking_can_ignore_desc_like_tail_tokens():
    # Same setup as the invariant test, but ignore tail position 0 and 2
    # (simulating masking desc value tokens in the appended tail).
    prompt_len = 5
    prefix_len = 3
    train_len = 6
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long)
    labels, coord_pos, coord_bins, coord_is_prefix = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        train_len=train_len,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],
        prefix_coord_target_bins=[123],
        tail_ignore_pos=[0, 2],
    )
    # Tail spans positions [8..10). Ignore relative 0 -> p=8, and relative 2 -> p=10.
    assert labels[8].item() == -100
    assert labels[9].item() == -100  # coord token
    assert labels[10].item() == -100

    # Coord supervision still includes both prefix and tail coord token.
    assert 6 in coord_pos
    assert 9 in coord_pos
