import torch

from src.trainers.rollout_correction.coordination import build_trainable_parameter_zero_loss
from src.trainers.rollout_correction.pack_schedule import Stage2PackSchedule


def test_pack_schedule_appends_shadow_slots_after_real_packs() -> None:
    schedule = Stage2PackSchedule.from_rank_pack_counts(
        local_pack_count=2,
        rank_pack_counts=[3, 2],
    )

    assert [
        (slot.slot_index, slot.local_pack_index, slot.is_empty)
        for slot in schedule.slots
    ] == [
        (0, 0, False),
        (1, 1, False),
        (2, None, True),
    ]
    assert schedule.empty_slot_count == 1
    assert all(slot.sync_gradients for slot in schedule.slots)


def test_pack_schedule_keeps_full_rank_slots_real() -> None:
    schedule = Stage2PackSchedule.from_rank_pack_counts(
        local_pack_count=3,
        rank_pack_counts=[3, 2],
    )

    assert [
        (slot.slot_index, slot.local_pack_index, slot.is_empty)
        for slot in schedule.slots
    ] == [
        (0, 0, False),
        (1, 1, False),
        (2, 2, False),
    ]
    assert schedule.empty_slot_count == 0
    assert all(slot.sync_gradients for slot in schedule.slots)


def test_trainable_parameter_zero_loss_touches_all_trainable_params() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False),
        torch.nn.Linear(3, 1, bias=False),
    )
    model[1].weight.requires_grad_(False)

    loss = build_trainable_parameter_zero_loss(model)
    assert loss.item() == 0.0
    loss.backward()

    assert model[0].weight.grad is not None
    assert torch.count_nonzero(model[0].weight.grad).item() == 0
    assert model[1].weight.grad is None
