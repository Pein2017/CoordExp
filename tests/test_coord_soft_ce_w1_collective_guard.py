def test_coord_soft_ce_w1_allreduces_even_when_no_coord_tokens(monkeypatch):
    """Avoid DDP deadlocks when some ranks have no coord-token supervision.

    When `average_tokens_across_devices=True`, loss code may perform distributed
    collectives. Even if this batch has zero coord positions locally (and returns
    None), it must still participate in the denom collective so ranks never
    deadlock due to conditional all_reduce calls.
    """

    import torch
    import torch.distributed as dist

    from src.trainers.losses.coord_soft_ce_w1 import compute_coord_soft_ce_w1_loss

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    calls = {"n": 0}

    def _all_reduce(_tensor, *args, **kwargs):
        calls["n"] += 1

    monkeypatch.setattr(dist, "all_reduce", _all_reduce)

    logits = torch.zeros((1, 2, 16), dtype=torch.float32)
    labels = torch.tensor([[0, 5]], dtype=torch.long)
    masked_labels = labels.clone()

    # coord_token_ids must be within vocab_size; coord_id_map marks none of them as coord.
    coord_token_ids = [1, 2]
    coord_id_map = torch.full((16,), -1, dtype=torch.long)

    class Cfg:
        temperature = 1.0
        ce_weight = 0.0
        soft_ce_weight = 1.0
        w1_weight = 1.0
        gate_weight = 0.0
        target_sigma = 2.0
        target_truncate = None

    result = compute_coord_soft_ce_w1_loss(
        logits=logits,
        labels=labels,
        masked_labels=masked_labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        cfg=Cfg(),
        average_tokens_across_devices=True,
        model_accepts_loss_kwargs=True,
        accelerator_num_processes=2,
    )

    assert result is None
    assert calls["n"] == 1
