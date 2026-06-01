from __future__ import annotations

from src.config.loader import ConfigLoader


def test_coord_token_mode_invariants_for_anchored_configs() -> None:
    stage1 = ConfigLoader.load_materialized_training_config(
        "configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first_1024_lvis_proxy.yaml"
    )
    stage2 = ConfigLoader.load_materialized_training_config(
        "configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml"
    )

    # Both anchored pipelines train on pre-quantized coord-token JSONLs, so runtime
    # normalization must be disabled to prevent double-normalization drift.
    assert stage1.custom.coord_tokens.enabled is True
    assert stage1.custom.coord_tokens.skip_bbox_norm is True
    assert stage2.custom.coord_tokens.enabled is True
    assert stage2.custom.coord_tokens.skip_bbox_norm is True

    # Stage-1 uses the coord-offset adapter to train only the 1000 coord-token IDs.
    assert stage1.custom.coord_offset.enabled is True
    ids = list(stage1.custom.coord_offset.ids)
    assert len(ids) == 1000
    assert ids == list(range(int(ids[0]), int(ids[0]) + 1000))

    # Stage-2 rollout correction should not enable the coord-offset adapter.
    assert stage2.custom.coord_offset.enabled is False
