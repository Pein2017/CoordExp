import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from swift.llm import get_model_tokenizer, get_template


def test_stage2_ab_packing_masks_and_coord_grads_smoke():
    """Regression test mirroring temp/smoke_stage2_ab_packing.py.

    Validates:
    - ms-swift packing metadata matches concatenated encoded lengths
    - CE mask excludes coord tokens (no grad at positions predicting coord tokens)
    - Coord expectation loss touches only coord-vocab slice and nothing else

    This is intentionally tiny and skips if the local model/tokenizer path is absent.
    """

    torch.manual_seed(0)

    model_dir = os.environ.get(
        "COORDEXP_STAGE2_AB_MODEL", "output/1-26/checkpoint-1516-merged"
    )
    if not os.path.isdir(model_dir):
        pytest.skip(f"missing stage2-ab model dir: {model_dir}")

    _, processor = get_model_tokenizer(model_dir, load_model=False)
    template = get_template(
        "qwen3_vl",
        processor,
        max_length=256,
        truncation_strategy="right",
        max_pixels=1572864999,
        padding_free=False,
    )
    template.set_mode("train")
    tok = template.tokenizer

    im = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    def _make(desc: str, c: int):
        payload = {
            "object_1": {
                "desc": desc,
                "bbox_2d": [
                    f"<|coord_{int(c)}|>",
                    f"<|coord_{int(c)}|>",
                    "<|coord_999|>",
                    "<|coord_999|>",
                ],
            }
        }
        assistant_text = json.dumps(payload, ensure_ascii=True, separators=(", ", ": "))
        y_ids = tok.encode(assistant_text, add_special_tokens=False)
        return template.encode(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": im},
                            {"type": "text", "text": "describe"},
                        ],
                    },
                    {"role": "assistant", "content": y_ids},
                ]
            },
            return_length=True,
        )

    e1 = _make("cat", 0)
    e2 = _make("dog", 1)

    template.packing = True
    template.padding_free = True
    batch = template.data_collator([e1, e2])
    ids = batch["input_ids"]
    labels = batch["labels"]

    assert batch["cu_seq_lens_q"].tolist() == [
        0,
        int(e1["length"]),
        int(e1["length"]) + int(e2["length"]),
    ]

    coord_start = tok.convert_tokens_to_ids("<|coord_0|>")
    assert isinstance(coord_start, int) and coord_start >= 0
    coord_end = int(coord_start) + 1000

    coord_mask = (ids >= coord_start) & (ids < coord_end)
    pos = torch.where(coord_mask[0])[0].tolist()
    assert len(pos) == 8
    assert min(pos) > 0
    prv = torch.tensor([int(p) - 1 for p in pos], dtype=torch.long)

    # Remap to a tiny vocab: coord bins -> [0..999], everything else -> 1000.
    vocab_small = 1001
    ids_small = ids.clone()
    ids_small[coord_mask] = ids[coord_mask] - coord_start
    ids_small[~coord_mask] = 1000

    labels_small = labels.clone()
    lbl_keep = labels_small != -100
    labels_small[lbl_keep & coord_mask] = labels_small[lbl_keep & coord_mask] - coord_start
    labels_small[lbl_keep & ~coord_mask] = 1000

    # CE mask excludes coord tokens.
    logits = torch.randn((1, ids.shape[1], vocab_small), requires_grad=True)
    ce_keep = (labels_small != -100) & (ids_small == 1000)
    labels_masked = labels_small.clone()
    labels_masked[~ce_keep] = -100

    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_small),
        labels_masked[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    ce.backward()
    assert float(logits.grad[0, prv].abs().sum()) == 0.0
    assert float(logits.grad.abs().sum()) > 0.0

    # Coord expectation loss touches only the coord slice at the positions that predict coord tokens.
    logits.grad.zero_()
    coord_logits = logits[0, prv][:, :1000]
    probs = torch.softmax(coord_logits, dim=-1)
    exp = (
        probs
        * torch.arange(0, 1000, device=probs.device, dtype=torch.float32).unsqueeze(0)
    ).sum(dim=-1) / 999.0
    gt = (ids_small[0, pos].float() / 999.0).to(exp.device)
    (exp - gt).abs().mean().backward()

    grad_coord = logits.grad[0, prv][:, :1000].abs().sum()
    grad_total = logits.grad.abs().sum()
    assert float(grad_coord) > 0.0
    assert float(grad_total - grad_coord) == 0.0
