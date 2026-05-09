from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import torch

from src.infer.backends import generate_hf_batch


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 3
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|im_end|>": self.eos_token_id,
            "<|endoftext|>": self.pad_token_id,
        }.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.convert_tokens_to_ids(text)]

    def decode(self, token_ids, **_: object) -> str:
        return "".join(f"<tok:{int(token_id)}>" for token_id in token_ids)

    def batch_decode(self, token_ids, **_: object) -> list[str]:
        return [self.decode(ids) for ids in token_ids]


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.seen_do_resize: list[bool] = []

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return "prompt"

    def __call__(
        self,
        *,
        text,
        images,
        return_tensors: str,
        padding: bool,
        do_resize: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is True
        self.seen_do_resize.append(do_resize)
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


class _Model:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object):
        self.generate_kwargs = dict(kwargs)
        scores = (torch.zeros((1, 16), dtype=torch.float32),)
        return SimpleNamespace(
            sequences=torch.tensor([[11, 12, 2]], dtype=torch.long),
            scores=scores,
        )


def test_hf_batch_generation_uses_qwen_chat_eos_pad_and_disables_resize() -> None:
    processor = _Processor()
    model = _Model()
    owner = SimpleNamespace(
        model=model,
        processor=processor,
        cfg=SimpleNamespace(device="cpu"),
        gen_cfg=SimpleNamespace(
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=None,
            compact_grammar_enabled=False,
            build_hf_stop_pressure_logits_processor=lambda **_: None,
            apply_hf_stop_pressure=lambda kwargs: None,
        ),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _build_messages=lambda image: [{"role": "user", "content": "detect"}],
    )
    image = Image.new("RGB", (4, 4), color="white")

    outputs = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert processor.seen_do_resize == [False]
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["eos_token_id"] == processor.tokenizer.eos_token_id
    assert model.generate_kwargs["pad_token_id"] == processor.tokenizer.pad_token_id
    assert outputs[0].text == "<tok:2>"
