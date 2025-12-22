from src.config.schema import CoordLossConfig


def test_coord_loss_poly_defaults_follow_mask_size():
    cfg = CoordLossConfig.from_mapping({"enabled": True, "poly_mask_size": 32})
    assert cfg.poly_mask_size == 32
    assert abs(cfg.poly_sigma_mask - (1.5 / 32.0)) < 1e-6


def test_coord_loss_poly_sigma_override():
    cfg = CoordLossConfig.from_mapping(
        {"enabled": True, "poly_mask_size": 32, "poly_sigma_mask": 0.25}
    )
    assert cfg.poly_sigma_mask == 0.25
