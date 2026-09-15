import pytest

from probes.training_set_completion import recover_readback as recovery


def _manifest():
    return {"routes": [{"image_id": image_id} for image_id in range(11)]}


def test_partition_covers_33_jobs_with_one_grouped_checkpoint_reload():
    groups = recovery.partition(_manifest())
    assert len(groups) == 8
    assert sum(len(group) for group in groups) == 33
    assert all(len(group) <= 5 for group in groups)
    assert [job["step"] for group in groups for job in group].count(16) == 11
    assert [job["step"] for group in groups for job in group].count(32) == 11
    assert [job["step"] for group in groups for job in group].count(64) == 11
    mixed = [group for group in groups if len({job["step"] for job in group}) > 1]
    assert [[job["step"] for job in group] for group in mixed] == [[16, 16, 16, 64]]


def test_checkpoint_row_guard_rejects_mixed_worker_row_loaded_from_prior_adapter():
    manifest = {"checkpoint_adapters": {"16": {"fingerprint": "step16"}, "64": {"fingerprint": "step64"}}}
    # A one-adapter-per-worker implementation would attach step 16 to the
    # final GPU-2 step-64 image.  The consumer guard must reject it first.
    with pytest.raises(ValueError, match="row checkpoint adapter mismatch"):
        recovery.validate_row_checkpoint(
            {"checkpoint_step": 64, "image_id": 42, "checkpoint_adapter": {"fingerprint": "step16"}},
            {"step": 64, "image_id": 42},
            manifest,
        )
