import torch

from src.coord_tokens.soft_rasterizer import soft_polygon_mask


def _square(center: float, size: float) -> torch.Tensor:
    half = size * 0.5
    left = center - half
    right = center + half
    top = center - half
    bottom = center + half
    return torch.tensor(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=torch.float32,
    )


def _mask(poly: torch.Tensor) -> torch.Tensor:
    return soft_polygon_mask(
        poly,
        mask_size=32,
        sigma_mask=1.5 / 32.0,
        tau_inside=0.08,
        beta_dist=100.0,
    )


def test_soft_polygon_mask_basic():
    poly = _square(0.5, 0.5)
    mask = _mask(poly)
    assert mask.shape == (32, 32)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    assert float(mask[16, 16]) > float(mask[0, 0])


def test_soft_polygon_mask_iou_shifted_lower():
    poly = _square(0.5, 0.5)
    shifted = _square(0.7, 0.5)
    mask_a = _mask(poly)
    mask_b = _mask(shifted)
    inter = (mask_a * mask_b).sum()
    union = (mask_a + mask_b - mask_a * mask_b).sum()
    iou = inter / union
    assert float(iou) < 0.5
