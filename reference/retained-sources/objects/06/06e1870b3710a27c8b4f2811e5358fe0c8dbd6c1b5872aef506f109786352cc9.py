from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from probes.dora_owner_learning.repeat_recovery_train import (
    BOX_END,
    EOS,
    classify_event,
    combine_lifecycle_resource_observations,
    duplicate_indicator,
    event_loss,
    loaded_composition_evidence,
    positive_loss,
    reference_kl,
    sample_seed,
    truncate_first_action,
    validate_positive_case,
)


OBJECT_START = 151646
OBJECT_END = 151647
BOX_START = 151648


class Tokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces
        self.eos_token_id = EOS

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.pieces[int(token)] for token in ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        for token_id, piece in self.pieces.items():
            if piece == token:
                return token_id
        return None


def _row(description_id: int, coords: tuple[int, int, int, int]) -> list[int]:
    return [OBJECT_START, description_id, OBJECT_END, BOX_START, *coords, BOX_END]


def _tokenizer() -> Tokenizer:
    pieces = {
        OBJECT_START: "<|object_ref_start|>",
        OBJECT_END: "<|object_ref_end|>",
        BOX_START: "<|box_start|>",
        BOX_END: "<|box_end|>",
        EOS: "<|im_end|>",
        10: "first",
        11: "second",
        12: "third",
    }
    pieces.update({1000 + value: f"<|coord_{value}|>" for value in range(1001)})
    return Tokenizer(pieces)


def _coords(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(1000 + value for value in values)


def _positive_fixture(tmp_path: Path) -> dict:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    h = _row(10, _coords((10, 10, 20, 20)))
    c = _row(11, _coords((30, 30, 40, 40)))
    w = _row(12, _coords((50, 50, 60, 60)))
    h_end, c_end, w_end = len(h), len(h) + len(c), len(h) + len(c) + len(w)
    return {
        "candidate_id": "351017-c01",
        "prompt": {"token_ids": [7, 8], "length": 2},
        "h": {"token_ids": h, "length": len(h)},
        "c": {"token_ids": c, "length": len(c)},
        "w": {"token_ids": w, "length": len(w)},
        "continuation_offsets": {
            "h_span": [0, h_end],
            "c_target_span": [h_end, c_end],
            "w_only_kl_span": [c_end, w_end],
            "c_target_positions": list(range(h_end, c_end)),
            "w_only_kl_positions": list(range(c_end, w_end)),
            "eos_in_c_target": False,
            "eos_in_w_only_kl": False,
        },
        "image": {
            "image_id": 351017,
            "image_path": str(image),
            "image_sha256": hashlib.sha256(b"image").hexdigest(),
        },
    }


def test_complete_c_and_w_masks_reject_one_position_shift(tmp_path: Path) -> None:
    case = _positive_fixture(tmp_path)
    assert validate_positive_case(case)["continuation_offsets"]["c_target_span"] == [9, 18]

    for field, replacement in (
        ("c_target_positions", list(range(10, 19))),
        ("w_only_kl_positions", list(range(19, 28))),
    ):
        shifted = copy.deepcopy(case)
        shifted["continuation_offsets"][field] = replacement
        with pytest.raises(ValueError):
            validate_positive_case(shifted)


def test_positive_loss_sums_every_c_token_and_has_off_by_one_teeth() -> None:
    logits = torch.zeros(3, 7, requires_grad=True)
    targets = torch.tensor([1, 2, 3])
    loss = positive_loss(logits, targets, [1, 2, 3])
    assert torch.allclose(loss, torch.log(torch.tensor(7.0)))
    loss.backward()
    assert (logits.grad.abs().sum(-1) > 0).tolist() == [True, True, True]
    with pytest.raises(ValueError):
        positive_loss(logits[:2], targets[:2], [1, 2, 3])
    with pytest.raises(ValueError):
        positive_loss(logits, targets, [1, 2, 4])


def test_conditional_kl_uses_only_declared_w_positions() -> None:
    logits = torch.zeros(4, 5, requires_grad=True)
    reference = torch.log_softmax(torch.tensor([[3., 0., 0., 0., 0.], [0., 3., 0., 0., 0.]]), -1)
    kl = reference_kl(logits, reference, [1, 3])
    kl.backward()
    assert logits.grad[0].abs().sum() == 0
    assert logits.grad[2].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0
    assert logits.grad[3].abs().sum() > 0
    with pytest.raises(ValueError):
        reference_kl(logits, reference, [1])


def test_first_action_terminalizer_keeps_raw_suffix_and_real_eos_bookkeeping() -> None:
    action = truncate_first_action([4, BOX_END, 5, 6, EOS], stop_reason="im_end")
    assert action["terminal_class"] == "box_end"
    assert action["retained_ids"] == [4, BOX_END]
    assert action["discarded_tail_ids"] == [5, 6, EOS]
    assert action["native_eos_observed"] is True

    eos = truncate_first_action([4, EOS], stop_reason="im_end")
    assert eos["terminal_class"] == "eos" and eos["retained_ids"] == [4, EOS]
    censored = truncate_first_action([4] * 64, stop_reason="length")
    assert censored["terminal_class"] == "censored" and censored["retained_tokens"] == 64
    with pytest.raises(ValueError):
        truncate_first_action([4, EOS, 5], stop_reason="im_end")


def test_native_pixel_rounding_and_multiple_matches_count_one_event_row() -> None:
    tokenizer = _tokenizer()
    # At 37x101 these different normalized boxes both round to [0,0,1,1].
    h = _row(10, _coords((0, 0, 20, 10)))
    action_ids = _row(11, _coords((10, 0, 30, 10)))
    action = truncate_first_action(action_ids, stop_reason="length", budget=len(action_ids))
    event = classify_event(
        h_ids=h, action=action, tokenizer=tokenizer,
        image_width=37, image_height=101, row_id="rounding",
    )
    assert event["D"] == 1 and event["max_iou_to_h"] == 1.0
    assert event["counted_event_rows"] == 1

    result = duplicate_indicator([1, 1, 9, 9], [[1, 1, 9, 9], [1, 1, 9, 9]])
    assert result["matching_prior_row_count"] == 2
    assert result["counted_event_rows"] == 1 and result["D"] == 1


def test_ddp_normalizations_are_mean3_mean56_and_denominator24() -> None:
    # Positive/conditional rows are replicated, so each rank has the same mean3.
    replicated = sum(torch.tensor(value / 3) for value in (2.0, 4.0, 6.0))
    assert float(sum([replicated] * 8) / 8) == pytest.approx(4.0)

    # Seven normal references per rank, scaled 8/56 before DDP mean.
    values = list(range(1, 57))
    local = [sum(values[rank::8]) * (100 * 8 / 56) for rank in range(8)]
    assert sum(local) / 8 == pytest.approx(100 * sum(values) / 56)

    # Three locally divided actions on eight ranks become one global /24 sum.
    local_events = [sum((rank * 3 + index + 1) / 3 for index in range(3)) for rank in range(8)]
    assert sum(local_events) / 8 == pytest.approx(sum(range(1, 25)) / 24)


def test_zero_event_retains_differentiable_zero_gradient_path() -> None:
    chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss = event_loss(chosen, 0)
    assert loss.requires_grad and loss.grad_fn is not None and float(loss.detach()) == 0.0
    loss.backward()
    assert chosen.grad is not None and torch.equal(chosen.grad, torch.zeros_like(chosen))


def test_sampling_seed_is_unique_over_full_rank_case_lattice() -> None:
    seeds = {sample_seed(step, rank, case) for step in range(1, 33)
             for rank in range(8) for case in range(3)}
    assert len(seeds) == 32 * 8 * 3


def test_lifecycle_resources_keep_preparation_peak_and_terminal_export_cost() -> None:
    combined = combine_lifecycle_resource_observations([
        {
            "phase": "preparation", "peak_cuda_allocated_bytes": 25,
            "peak_cuda_reserved_bytes": 27, "peak_rss_bytes": 30,
            "elapsed_seconds": 4.0,
        },
        {
            # Simulates the old counterexample: a reset made training look lower.
            "phase": "training", "peak_cuda_allocated_bytes": 10,
            "peak_cuda_reserved_bytes": 12, "peak_rss_bytes": 28,
            "elapsed_seconds": 9.0,
        },
        {
            "phase": "post_export_terminal", "peak_cuda_allocated_bytes": 11,
            "peak_cuda_reserved_bytes": 13, "peak_rss_bytes": 35,
            "elapsed_seconds": 14.0,
        },
    ])
    assert combined["peak_cuda_allocated_bytes"] == 25
    assert combined["peak_cuda_reserved_bytes"] == 27
    assert combined["peak_rss_bytes"] == 35
    assert combined["elapsed_seconds"] == 14.0
    assert combined["phases"][-1] == "post_export_terminal"


def test_loaded_composition_uses_exact_cross_schema_embedding_identity() -> None:
    files = [{"relative_path": "delta.safetensors", "sha256": "abc", "size_bytes": 3}]
    semantics = {
        "base_model_path": "/model", "semantics": "additive_delta",
        "tensor_key": "delta", "tensor_shape": [4, 2], "tensor_dtype": "float32",
    }
    expected_embedding = {
        "root": "/embedding", "kind": "special_token_embedding_delta",
        "version": "coordexp-swift-special-token-embedding-delta-v1",
        "fingerprint": "swift-envelope", "file_count": 1,
        "files": files, "semantic_identity": semantics,
    }
    inspected_embedding = {
        **expected_embedding,
        "version": "coordexp-infras-special-token-embedding-delta-v1",
        "fingerprint": "infras-envelope",
    }
    loaded = {
        "model_identity": {
            "base": {"path": "/model"},
            "adapter": {
                "adapter_path": "/adapter", "merged_adapters": [],
                "adapter_state_evidence": {"state_checked": True},
            },
            "embedding_delta": {"status": "loaded", "load": {"loaded": True}},
        },
        "effective_settings": {
            "observed_attn_implementation": "sdpa",
            "observed_model_dtype": {"parameter_dtype_names": ["torch.float32"]},
        },
    }
    evidence = loaded_composition_evidence(
        loaded_identity=loaded, expected_base="/model", expected_adapter="/adapter",
        expected_embedding=expected_embedding, inspected_embedding=inspected_embedding,
    )
    assert evidence["passed"] is True
    assert evidence["actual"]["embedding"]["fingerprint"] == "infras-envelope"
    assert evidence["expected"]["embedding"]["fingerprint"] == "swift-envelope"

    wrong_files = copy.deepcopy(inspected_embedding)
    wrong_files["files"][0]["sha256"] = "changed"
    rejected = loaded_composition_evidence(
        loaded_identity=loaded, expected_base="/model", expected_adapter="/adapter",
        expected_embedding=expected_embedding, inspected_embedding=wrong_files,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["embedding_payload_files"] is False
