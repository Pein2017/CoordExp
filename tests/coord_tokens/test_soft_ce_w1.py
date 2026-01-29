import torch

from src.coord_tokens.soft_ce_w1 import gaussian_soft_targets, wasserstein_1_cdf


def test_gaussian_soft_targets_unimodal_and_normalized():
    target = torch.tensor([500], dtype=torch.long)
    probs = gaussian_soft_targets(target, num_bins=1000, sigma=2.0, truncate=16)
    assert probs.shape == (1, 1000)
    assert torch.all(probs >= 0).item()
    assert abs(float(probs.sum().item()) - 1.0) < 1e-6

    # Unimodal: peak at target bin, monotonic up to peak and down after peak (within tolerance).
    peak = int(probs.argmax(dim=-1).item())
    assert peak == 500

    left = probs[0, : peak + 1]
    right = probs[0, peak:]
    assert torch.all(torch.diff(left) >= -1e-10).item()
    assert torch.all(torch.diff(right) <= 1e-10).item()


def test_wasserstein_1_cdf_zero_for_identical_distributions():
    p = torch.zeros((1, 1000), dtype=torch.float32)
    p[0, 123] = 1.0
    q = p.clone()
    w1 = wasserstein_1_cdf(p, q, normalize=True)
    assert w1.shape == (1,)
    assert float(w1.item()) == 0.0


def test_wasserstein_1_cdf_matches_dirac_shift_distance():
    p = torch.zeros((1, 1000), dtype=torch.float32)
    q = torch.zeros((1, 1000), dtype=torch.float32)
    p[0, 10] = 1.0
    q[0, 12] = 1.0

    w1 = wasserstein_1_cdf(p, q, normalize=True)
    # With normalization by (K-1)=999, a 2-bin shift yields 2/999.
    assert abs(float(w1.item()) - (2.0 / 999.0)) < 1e-6

