import torch

from src.data_collators.dataset_metrics import build_dataset_metrics_collator
from src.data_collators.token_types import TokenType, compute_token_types


class DummyTokenizer:
    """Character-level tokenizer stub with offset_mapping."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        offsets = [(i, i + 1) for i in range(len(text))]
        max_length = kwargs.get("max_length")
        if max_length is not None:
            offsets = offsets[: int(max_length)]
        return {"offset_mapping": offsets, "input_ids": [0] * len(offsets)}


class DummyTemplateMeta:
    suffix = []


class DummyTemplate:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.template_meta = DummyTemplateMeta()
        self.packing = False
        self.padding_free = False

    def data_collator(self, batch):
        # Padded collator: pad to max length across batch
        max_len = max(len(item["labels"]) for item in batch)
        input_ids = []
        labels = []
        attn = []
        for item in batch:
            pad = max_len - len(item["labels"])
            ids = item["input_ids"] + [0] * pad
            lab = item["labels"] + [-100] * pad
            mask = item.get("attention_mask", [1] * len(item["labels"])) + [0] * pad
            input_ids.append(ids)
            labels.append(lab)
            attn.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def _make_payload():
    return {
        "desc": "cat",
        "bbox_2d": [1.0, 2.0, 3.0, 4.0],
        "fmt": ["ok"],
    }


def test_compute_token_types_basic():
    tokenizer = DummyTokenizer()
    payload = _make_payload()
    # Use a fixed supervised length so the dummy tokenizer can trivially align ids.
    labels = torch.zeros(32, dtype=torch.long)
    types = compute_token_types(
        tokenizer=tokenizer,
        payload=payload,
        labels=labels,
        attention_mask=None,
        suffix_tokens=None,
    )
    assert types is not None
    assert types.shape[0] == labels.shape[0]
    assert (types != TokenType.IGNORE).any()


def test_collator_attaches_token_types_padded():
    template = DummyTemplate()
    collator = build_dataset_metrics_collator(
        template,
        token_type_cfg=type("Cfg", (), {"enabled": True, "include": ("default",), "exclude": ()}),
    )
    batch = [
        {
            "input_ids": [1, 2, 3],
            "labels": [0, 0, 0],
            "attention_mask": [1, 1, 1],
            "assistant_payload": _make_payload(),
        },
        {
            "input_ids": [4, 5],
            "labels": [0, 0],
            "attention_mask": [1, 1],
            "assistant_payload": _make_payload(),
        },
    ]
    out = collator(batch)
    assert "token_types" in out
    tt = out["token_types"]
    assert tt.shape == out["labels"].shape
    # Supervised tokens should be non-IGNORE at least somewhere
    assert (tt != TokenType.IGNORE).any()


def test_collator_attaches_token_types_packed_concat():
    template = DummyTemplate()

    def packing_collator(batch):
        # batch[0] is pack (list of samples)
        pack = batch[0]
        input_ids = []
        labels = []
        attn = []
        for sample in pack:
            input_ids.extend(sample["input_ids"])
            labels.extend(sample["labels"])
            attn.extend(sample.get("attention_mask", [1] * len(sample["labels"])))
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }

    collator = build_dataset_metrics_collator(
        template,
        base_collator=packing_collator,
        token_type_cfg=type("Cfg", (), {"enabled": True, "include": ("default",), "exclude": ()}),
    )

    pack = [
        {
            "input_ids": [1, 2],
            "labels": [0, 0],
            "attention_mask": [1, 1],
            "assistant_payload": _make_payload(),
        },
        {
            "input_ids": [3],
            "labels": [0],
            "attention_mask": [1],
            "assistant_payload": _make_payload(),
        },
    ]

    out = collator([pack])
    assert "token_types" in out
    tt = out["token_types"]
    labels = out["labels"]
    assert tt.shape == labels.shape
    # Packed single batch -> shape (1, total_len)
    assert tt.shape[0] == 1
    assert tt.shape[1] == labels.shape[1]
    assert (tt != TokenType.IGNORE).any()


def test_packed_include_exclude_filters_and_aligns():
    template = DummyTemplate()

    def packing_collator(batch):
        pack = batch[0]
        input_ids = []
        labels = []
        attn = []
        for sample in pack:
            input_ids.extend(sample["input_ids"])
            labels.extend(sample["labels"])
            attn.extend(sample.get("attention_mask", [1] * len(sample["labels"])))
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }

    cfg = type("Cfg", (), {"enabled": True, "include": ("lvis",), "exclude": ()})
    collator = build_dataset_metrics_collator(
        template,
        base_collator=packing_collator,
        token_type_cfg=cfg,
    )

    pack = [
        {
            "dataset": "lvis",
            "input_ids": [1, 2],
            "labels": [0, 0],
            "attention_mask": [1, 1],
            "assistant_payload": _make_payload(),
        },
        {
            "dataset": "coco",
            "input_ids": [3, 4],
            "labels": [0, 0],
            "attention_mask": [1, 1],
            "assistant_payload": _make_payload(),
        },
    ]

    out = collator([pack])
    tt = out["token_types"][0]
    # First two tokens (lvis) should be supervised types; last two (coco excluded) IGNORE
    assert (tt[:2] != TokenType.IGNORE).all()
    assert (tt[2:] == TokenType.IGNORE).all()


def test_packed_misalignment_falls_back_to_ignore():
    template = DummyTemplate()

    def bad_packing_collator(batch):
        # Intentionally truncate labels by one to force mismatch
        pack = batch[0]
        input_ids = []
        labels = []
        for sample in pack:
            input_ids.extend(sample["input_ids"])
            labels.extend(sample["labels"])
        # drop last label to create mismatch
        labels = labels[:-1]
        attn = [1] * len(labels)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }

    cfg = type("Cfg", (), {"enabled": True, "include": ("lvis", "coco"), "exclude": ()})
    collator = build_dataset_metrics_collator(
        template, base_collator=bad_packing_collator, token_type_cfg=cfg
    )

    pack = [
        {
            "dataset": "lvis",
            "input_ids": [1, 2],
            "labels": [10, 11],
            "assistant_payload": _make_payload(),
        },
        {
            "dataset": "coco",
            "input_ids": [3, 4],
            "labels": [12, 13],
            "assistant_payload": _make_payload(),
        },
    ]

    out = collator([pack])
    tt = out["token_types"][0]
    # On mismatch, collator should produce all IGNORE token types for the packed example
    assert (tt == TokenType.IGNORE).all()
