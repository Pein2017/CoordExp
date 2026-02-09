from __future__ import annotations

import re
import threading

import pytest
import torch

from src.trainers.rollout_matching_sft import (
    GTObject,
    RolloutMatchingSFTTrainer,
    _build_labels_and_coord_targets_for_sample,
    _build_labels_and_coord_targets_for_batch,
    _serialize_append_fragment,
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


class _DummySession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyVLLMClient:
    def __init__(self) -> None:
        self.sessions = [_DummySession(), _DummySession()]
        self.close_calls = 0

    def close_communicator(self) -> None:
        self.close_calls += 1


def test_shutdown_vllm_server_client_closes_resources():
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer._vllm_server_client_lock = threading.Lock()
    client = _DummyVLLMClient()
    trainer._vllm_server_client = client
    trainer._vllm_server_comm_inited = True

    trainer._shutdown_vllm_server_client(
        close_communicator=True,
        close_sessions=True,
    )

    assert client.close_calls == 1
    assert all(s.closed for s in client.sessions)
    assert trainer._vllm_server_client is None
    assert trainer._vllm_server_comm_inited is False


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


def test_rollout_parse_max_index_includes_invalid_retained_keys():
    tok = _DummyTokenizerRM()
    # object_9 has empty desc -> dropped by strict validation, but key index must still
    # reserve FN numbering space for collision-safe append.
    text = (
        '{"object_2": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, '
        '"object_9": {"desc": " ", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"]}}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)

    assert [o.index for o in parsed.valid_objects] == [2]
    assert parsed.dropped_invalid >= 1
    assert parsed.max_object_index_in_prefix == 9


def test_rollout_parse_max_index_includes_non_dict_retained_keys():
    tok = _DummyTokenizerRM()
    # object_4 has a non-dict value; it is retained in prefix key space and must
    # reserve FN numbering to avoid duplicate object keys.
    text = (
        '{"object_4": 123, '
        '"object_3": {"desc": "a", "bbox_2d": ['
        '"<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)

    assert [o.index for o in parsed.valid_objects] == [3]
    assert parsed.max_object_index_in_prefix == 4

    fn_obj = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[5, 6, 7, 8],
        desc="fn",
    )
    append_text = _serialize_append_fragment(
        fn_objects=[fn_obj],
        start_index=(parsed.max_object_index_in_prefix or 0) + 1,
        prefix_text=parsed.prefix_text,
    )
    assert '"object_5"' in append_text
    assert '"object_4"' not in append_text


def test_serialize_append_fragment_comma_policy_is_prefix_entry_aware():
    fn_objs = [
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 3, 4],
            desc="fn",
        )
    ]

    frag_empty = _serialize_append_fragment(
        fn_objects=fn_objs,
        start_index=1,
        prefix_text="{",
    )
    assert frag_empty.startswith('"object_1"')

    prefix_with_entry = (
        '{"object_7": {"desc": "p", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", '
        '"<|coord_3|>", "<|coord_4|>"]}'
    )
    frag_non_empty = _serialize_append_fragment(
        fn_objects=fn_objs,
        start_index=8,
        prefix_text=prefix_with_entry,
    )
    assert frag_non_empty.startswith(", ")
    assert '"object_8"' in frag_non_empty


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


def test_rollout_parse_truncated_tail_keeps_all_previous_complete_objects():
    tok = _DummyTokenizerRM()
    # Truncated mid-object_3: object_1 and object_2 should remain.
    text = (
        '{"object_1": {"desc": "a", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, '
        '"object_2": {"desc": "b", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"]}, '
        '"object_3": {"desc": "c", "bbox_2d": ["<|coord_9|>"'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.truncated is True
    assert [o.index for o in parsed.valid_objects] == [1, 2]
    assert "object_3" not in parsed.prefix_text
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


def test_packed_label_mask_equivalence_two_segments():
    # Two synthetic segments; packed-mode labels should equal concatenated per-segment labels.
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}

    # Segment 1: reuse the invariant pattern (with one coord in prefix and one in tail).
    prompt_len1 = 5
    prefix_len1 = 3
    train_len1 = 6
    input_ids1 = torch.tensor([1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long)
    labels1, cpos1, cbins1, cis1 = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids1,
        prompt_len=prompt_len1,
        prefix_len=prefix_len1,
        train_len=train_len1,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],
        prefix_coord_target_bins=[123],
    )

    # Segment 2: different sizes to validate offset logic.
    prompt_len2 = 4
    prefix_len2 = 2
    train_len2 = 5
    # Place a coord token in the tail (assistant-local) and one prefix-supervised slot.
    input_ids2 = torch.tensor([11, 12, 13, 14, 15, 10, 20, 16, 17], dtype=torch.long)
    labels2, cpos2, cbins2, cis2 = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids2,
        prompt_len=prompt_len2,
        prefix_len=prefix_len2,
        train_len=train_len2,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],
        prefix_coord_target_bins=[999],
    )

    packed_ids = torch.cat([input_ids1, input_ids2], dim=0).unsqueeze(0)
    meta = [
        {
            "prompt_len": prompt_len1,
            "prompt_ids": input_ids1[:prompt_len1].tolist(),
            "prefix_len": prefix_len1,
            "train_len": train_len1,
            "encoded_len": int(input_ids1.numel()),
            "prefix_coord_pos": [1],
            "prefix_coord_target_bins": [123],
        },
        {
            "prompt_len": prompt_len2,
            "prompt_ids": input_ids2[:prompt_len2].tolist(),
            "prefix_len": prefix_len2,
            "train_len": train_len2,
            "encoded_len": int(input_ids2.numel()),
            "prefix_coord_pos": [1],
            "prefix_coord_target_bins": [999],
        },
    ]
    labels_packed, sb, spos, sbin, sis_prefix = _build_labels_and_coord_targets_for_batch(
        input_ids=packed_ids,
        meta=meta,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
    )

    expected = torch.cat([labels1, labels2], dim=0)
    assert labels_packed.shape == (1, expected.numel())
    assert torch.equal(labels_packed[0], expected)

    # Coord supervision positions for segment 2 must be offset by len(segment 1).
    off = int(input_ids1.numel())
    expected_pos = set(cpos1) | {off + int(p) for p in cpos2}
    got_pos = set(int(p) for p in spos)
    assert got_pos == expected_pos

    # Bins should match (order not guaranteed).
    expected_bins = [int(x) for x in cbins1] + [int(x) for x in cbins2]
    assert sorted(int(x) for x in sbin) == sorted(expected_bins)

    # Packed mode should report batch index 0 for all supervision targets.
    assert set(int(b) for b in sb) == {0}

    # Prefix flags should be preserved (order not guaranteed).
    expected_is_prefix = [bool(x) for x in cis1] + [bool(x) for x in cis2]
    assert sorted(bool(x) for x in sis_prefix) == sorted(expected_is_prefix)


def test_packed_prompt_prefix_sanity_check_rejects_mismatch():
    coord_id_set = {10}
    coord_id_to_bin = {10: 10}

    input_ids1 = torch.tensor([1, 2, 3, 4, 5, 10, 6], dtype=torch.long)
    input_ids2 = torch.tensor([7, 8, 9, 10, 11, 0], dtype=torch.long)
    packed_ids = torch.cat([input_ids1, input_ids2], dim=0).unsqueeze(0)

    # Segment 2 prompt_ids intentionally wrong to trigger the strict sanity check.
    meta = [
        {
            "prompt_len": 3,
            "prompt_ids": input_ids1[:3].tolist(),
            "prefix_len": 1,
            "train_len": 3,
            "encoded_len": int(input_ids1.numel()),
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
        },
        {
            "prompt_len": 2,
            "prompt_ids": [999, 999],  # mismatch
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": int(input_ids2.numel()),
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
        },
    ]

    with pytest.raises(ValueError, match="prompt tokenization mismatch"):
        _build_labels_and_coord_targets_for_batch(
            input_ids=packed_ids,
            meta=meta,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
        )


def test_adaptive_raw_microbatch_stacker_bumps_on_underfill() -> None:
    from src.trainers.rollout_matching_sft import _AdaptiveRawMicroBatchStacker

    class _DummyTrainer:
        def __init__(self) -> None:
            # Overestimated avg length would otherwise pick too few raw samples.
            self._rm_avg_segment_len = 1100.0
            # Last pack stats indicate underfill with 8 segments and an empty buffer.
            self._rm_last_pack_fill = 0.579
            self._rm_last_pack_segments = 8
            self._rm_last_pack_buffer_after = 0

        def _packing_length(self) -> int:
            return 12000

        def _packing_min_fill_ratio(self) -> float:
            return 0.7

        def _packing_buffer_cap(self) -> int:
            return 256

    trainer = _DummyTrainer()
    stacker = _AdaptiveRawMicroBatchStacker(dataloader=[], trainer=trainer)

    # Need >= ceil(8 * 0.7 / 0.579) == 10 to avoid repeating the same underfill.
    assert stacker._target_microbatch_size() >= 10
