from types import SimpleNamespace

import torch


def test_position_ids_are_safe_in_differentiable_forward():
    # Qwen positions participate in differentiable rotary products. An inference
    # tensor cannot be saved by autograd even though positions require no grad.
    from src.qwen.native import derive_position_ids

    class Rope:
        def get_rope_index(self, ids, grid, video, *, attention_mask):
            return torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1), None

    ids = torch.tensor([[1, 2, 3]])
    positions = derive_position_ids(
        model=Rope(),
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        video_grid_thw=None,
    )
    weight = torch.nn.Parameter(torch.ones(3, 1, 3))
    (weight * positions).sum().backward()
    torch.testing.assert_close(weight.grad, positions.to(weight.dtype))


class TinyReplay(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(72, dtype=torch.float32).reshape(9, 8) / 71
        )

    def get_rope_index(self, ids, grid, video, *, attention_mask):
        return torch.arange(ids.shape[1]).view(1, 1, -1).expand(
            3, ids.shape[0], -1
        ), None

    def forward(self, input_ids, position_ids, logits_to_keep=0, **kwargs):
        # Position-dependent causal state: shifting any selected row changes gradients.
        hidden = torch.nn.functional.embedding(input_ids, self.weight)
        hidden = hidden + self.weight[0] * position_ids[0, :, :, None]
        logits = hidden @ self.weight.T
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def test_replay_compact_and_full_preserve_causal_scores_and_gradients():
    from src.qwen.native import prepare_replay

    model = TinyReplay()
    native = dict(
        input_ids=torch.tensor([[1, 2]]),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        pixel_values=torch.ones(1, 4),
        position_ids="stale",
        cache_position="stale",
        rope_deltas="stale",
        token_type_ids="stale",
        past_key_values="stale",
    )
    values = []
    for compact in (False, True):
        replay = prepare_replay(
            model,
            native,
            prompt_token_ids=[1, 2],
            continuation_token_ids=[3, 4, 5],
            compact_logits=compact,
        )
        assert replay.inputs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert replay.inputs["attention_mask"].tolist() == [[1] * 5]
        assert not any(
            key in replay.inputs
            for key in (
                "cache_position",
                "rope_deltas",
                "token_type_ids",
                "past_key_values",
            )
        )
        assert replay.inputs["pixel_values"] is native["pixel_values"]
        logits = model(**replay.inputs).logits
        aligned = replay.aligned_logits(logits)
        scores = (
            aligned.float()
            .log_softmax(-1)
            .gather(1, replay.target_ids[:, None])
            .squeeze(1)
        )
        grad = torch.autograd.grad(scores.sum(), model.weight)[0]
        values.append((scores, grad))
        wrong = (
            logits[0, -3:]
            .log_softmax(-1)
            .gather(1, replay.target_ids[:, None])
            .squeeze(1)
        )
        assert not torch.allclose(scores, wrong)  # off-by-one oracle has teeth
    torch.testing.assert_close(values[0][0], values[1][0])
    torch.testing.assert_close(values[0][1], values[1][1])
    assert native["position_ids"] == "stale"


def test_native_preparation_has_optional_identity_and_preserves_image_lifetime(
    monkeypatch,
):
    from PIL import Image
    import src.qwen.native as native

    image = Image.new("RGB", (2, 2))

    class Processor:
        tokenizer = SimpleNamespace(padding_side="right")

        def __call__(self, **kwargs):
            assert kwargs["do_resize"] is False
            assert kwargs["images"][0].getpixel((0, 0)) == (0, 0, 0)
            return dict(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
                pixel_values=torch.ones(1, 4),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
            )

    monkeypatch.setattr(
        native,
        "rgb_image_sha256",
        lambda _: (_ for _ in ()).throw(AssertionError("unrequested hash")),
    )
    batch = native.prepare_native_inputs(
        Processor(), [native.NativeRequest("r", "chat", image)]
    )
    assert batch.request_ids == ("r",)
    assert batch.prompt_token_ids == ((1, 2),)
    assert batch.image_grids == ((1, 1, 1),)
    assert batch.media_sha256 is None
    assert image.getpixel((0, 0)) == (0, 0, 0)


def test_literal_histories_fail_before_rope_on_invalid_ids():
    import pytest
    from src.qwen.native import exact_history_inputs

    model = TinyReplay()
    model.get_rope_index = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("must not derive positions")
    )
    for rows in ([[]], [[True]], [[1.2]], [[-1]]):
        with pytest.raises(ValueError):
            exact_history_inputs(model, {}, rows, pad_token_id=0)
