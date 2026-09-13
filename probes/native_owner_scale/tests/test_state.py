import pytest
import torch

from probes.native_owner_scale.state import ARMS, _key_vector, canonical_terminal_stop, k_only_branch_slices


def captured_fixture():
    native = [
        (
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 10 * layer,
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 20 * layer,
        )
        for layer in range(3)
    ]
    return {
        "native": native,
        "owner_a": [(keys + 2 * (layer + 1), values - 100) for layer, (keys, values) in enumerate(native)],
        "owner_b": [(keys + 0.25, values + 300) for keys, values in native],
    }


def test_k_only_self_and_a_preserve_every_value_byte():
    captured = captured_fixture()
    self_branch, self_stats = k_only_branch_slices(captured, "native_self")
    a_branch, a_stats = k_only_branch_slices(captured, "owner_a_k")
    assert self_stats["realized_k_norm"] == 0
    for layer in range(3):
        assert torch.equal(self_branch[layer][0], captured["native"][layer][0])
        assert torch.equal(self_branch[layer][1], captured["native"][layer][1])
        assert torch.equal(a_branch[layer][0], captured["owner_a"][layer][0])
        assert torch.equal(a_branch[layer][1], captured["native"][layer][1])
    assert a_stats["norm_rule"].endswith("4 coordinate positions")


def test_wrong_owner_k_is_globally_norm_matched_not_layer_matched():
    captured = captured_fixture()
    a_branch, _ = k_only_branch_slices(captured, "owner_a_k")
    b_branch, stats = k_only_branch_slices(captured, "owner_b_k_to_a")
    native = _key_vector(captured["native"])
    assert torch.linalg.vector_norm(_key_vector(a_branch) - native) == pytest.approx(
        torch.linalg.vector_norm(_key_vector(b_branch) - native), rel=2e-6
    )
    # B has a uniform direction; one global scale preserves that cross-layer ratio.
    delta = _key_vector(b_branch) - native
    assert delta[:24].mean() == pytest.approx(delta[48:72].mean())
    assert stats["scale"] > 1
    for layer in range(3):
        assert torch.equal(b_branch[layer][1], captured["native"][layer][1])


def test_unregistered_and_zero_norm_fail_closed():
    captured = captured_fixture()
    with pytest.raises(ValueError, match="unregistered"):
        k_only_branch_slices(captured, "adaptive_a")
    captured["owner_b"] = captured["native"]
    with pytest.raises(ValueError, match="zero-norm"):
        k_only_branch_slices(captured, "owner_b_k_to_a")
    assert ARMS == ("native_self", "owner_a_k", "owner_b_k_to_a")


def test_native_im_end_is_canonical_eos_but_early_length_fails_closed():
    assert canonical_terminal_stop("im_end", 90, 3074) == "eos"
    assert canonical_terminal_stop("eos", 390, 3064) == "eos"
    assert canonical_terminal_stop("length", 3064, 3064) == "length"
    with pytest.raises(ValueError, match="before EOS/full"):
        canonical_terminal_stop("length", 100, 3064)
    with pytest.raises(ValueError, match="unknown stop"):
        canonical_terminal_stop("cancelled", 100, 3064)
