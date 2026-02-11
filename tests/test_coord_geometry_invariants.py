from src.common.geometry.coord_utils import (
    coerce_point_list,
    ints_to_pixels_norm1000,
)
from src.eval.parsing import pair_points


def test_pair_points_preserves_input_order():
    pts = [10, 20, 30, 40, 50, 60]
    paired = pair_points(pts)
    assert paired == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


def test_coerce_point_list_preserves_token_sequence_order():
    raw = ["<|coord_5|>", "<|coord_7|>", "<|coord_9|>", "<|coord_11|>"]
    points, had_tokens = coerce_point_list(raw)
    assert had_tokens is True
    assert points == [5.0, 7.0, 9.0, 11.0]


def test_norm1000_conversion_is_deterministic():
    ints = [0, 0, 999, 999, 250, 750]
    a = ints_to_pixels_norm1000(ints, width=640, height=480)
    b = ints_to_pixels_norm1000(ints, width=640, height=480)
    assert a == b
