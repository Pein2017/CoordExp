from src.config.schema import TokenTypeMetricsConfig


def test_token_type_metrics_config_parses_optional_knobs() -> None:
    cfg = TokenTypeMetricsConfig.from_mapping(
        {
            "enabled": True,
            "include": ["lvis"],
            "log_top5": False,
            "coord_monitor_mass": False,
            "coord_monitor_mass_max_tokens": 123,
        }
    )
    assert cfg.enabled is True
    assert cfg.include == ("lvis",)
    assert cfg.log_top5 is False
    assert cfg.coord_monitor_mass is False
    assert cfg.coord_monitor_mass_max_tokens == 123

