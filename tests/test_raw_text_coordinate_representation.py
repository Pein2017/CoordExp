import torch

from src.analysis.raw_text_coordinate_representation import (
    pool_span_hidden_states,
    representation_rsa,
)


def test_pool_span_hidden_states_supports_last_digit_and_mean_digits() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        ]
    )
    pooled = pool_span_hidden_states(
        hidden_states=hidden,
        pooling=("last_digit", "mean_digits"),
    )

    assert torch.equal(pooled["last_digit"], torch.tensor([[3.0, 0.0]]))
    assert torch.equal(pooled["mean_digits"], torch.tensor([[2.0, 0.0]]))


def test_representation_rsa_is_one_for_perfectly_ordered_distances() -> None:
    states = torch.tensor([[0.0], [1.0], [3.0]])
    numeric_values = torch.tensor([100.0, 101.0, 103.0])

    assert representation_rsa(states=states, numeric_values=numeric_values) == 1.0
