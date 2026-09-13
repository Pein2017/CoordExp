from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from probes.row_feedback.runtime import (
    BOX_END,
    EOS,
    FeedbackSourceOverride,
    _bound_endpoint_raw,
    _materialization_raw_sources,
    feedback_slot_embedding,
    generate_visible,
    mapped_teacher_kl,
    prepared_inputs_sha256,
    replay_visible,
    visible_nll_sum,
)


VOCAB = BOX_END + 64
HIDDEN = 4
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-row-feedback-pilot"
)


class FakeCache:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def get_seq_length(self) -> int:
        return self.values.shape[1]


class FakeLanguage(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(HIDDEN)
        self.layers = torch.nn.ModuleList([FakeLayer()])


class FakeSelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)


class FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeSelfAttention()


class FakeCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLanguage()


class FakeVisual(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels


class FakeModel(torch.nn.Module):
    def __init__(self, scripted_tokens: dict[int, int] | None = None) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.model = FakeCore()
        self.visual = FakeVisual()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.scripted_tokens = scripted_tokens or {}
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_grid_thw, video_grid_thw, attention_mask
        length = input_ids.shape[1]
        base = torch.arange(length, device=input_ids.device).view(1, 1, length)
        positions = torch.cat((base, base + 100, base + 200), dim=0)
        return positions, torch.zeros((1, 1), dtype=torch.long, device=input_ids.device)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: FakeCache | None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
        **kwargs: object,
    ) -> SimpleNamespace:
        del image_grid_thw, kwargs
        assert use_cache and return_dict
        if pixel_values is not None:
            self.visual(pixel_values)
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        language_layer = cast(FakeLayer, self.model.language_model.layers[0])
        keys = language_layer.self_attn.k_proj(x)
        prior = keys.new_zeros((1, 1, HIDDEN)) if past_key_values is None else past_key_values.values.sum(
            dim=1, keepdim=True
        )
        # Cache the K path: final post-norm output is not itself KV.
        cumulative = prior + keys.cumsum(dim=1)
        hidden = self.model.language_model.norm(cumulative)
        cache_values = keys if past_key_values is None else torch.cat((past_key_values.values, keys), dim=1)
        logits = self.lm_head(hidden)
        for row, position in enumerate(cache_position.tolist()):
            token = self.scripted_tokens.get(position)
            if token is not None:
                scripted = logits[row:row + 1].new_full((VOCAB,), -1000)
                scripted[token] = 1000
                logits[:, row] = scripted
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        self.calls.append({
            "input_kind": "ids" if input_ids is not None else "embeds",
            "cache_position": cache_position.tolist(),
            "position_ids": position_ids.detach().clone(),
            "attention_length": attention_mask.shape[1],
        })
        return SimpleNamespace(logits=logits, past_key_values=FakeCache(cache_values))


class FakeTokenizer:
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return "|".join(str(token) for token in ids)


def fixture(*, prompt: list[int] | None = None, scripted: dict[int, int] | None = None):
    prompt = prompt or [1, 2]
    model = FakeModel(scripted)
    qwen = SimpleNamespace(model=model, tokenizer=FakeTokenizer())
    inputs = {
        "input_ids": torch.tensor([prompt]),
        "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "pixel_values": torch.ones((1, 2)),
    }
    return qwen, inputs, prompt


def test_feedback_slot_uses_exact_frozen_rms_rule() -> None:
    embedding = torch.tensor([[[3.0, 4.0]]])
    source = torch.tensor([[[0.0, 2.0]]])
    slot = feedback_slot_embedding(embedding, source)
    expected = embedding + embedding.square().mean(-1, keepdim=True).sqrt() * (
        source / source.square().mean(-1, keepdim=True).sqrt()
    )
    assert torch.equal(slot, expected)
    with pytest.raises(ValueError, match="zero or nonfinite RMS"):
        feedback_slot_embedding(embedding, torch.zeros_like(source))


def test_replay_inserts_slots_only_after_visible_box_ends_and_aligns_targets() -> None:
    qwen, inputs, prompt = fixture(prompt=[BOX_END, 2])
    result = replay_visible(
        qwen,
        inputs,
        prompt_ids=prompt,
        history_ids=[3, BOX_END, 4],
        target_ids=[5, BOX_END, 6],
        arm="F",
        capture_feedback_sources=True,
    )
    assert result["logits"].shape == (3, VOCAB)
    assert result["target_ids"].tolist() == [5, BOX_END, 6]
    assert result["internal_slot_count"] == 2
    assert result["slot_work"] == {
        "prefill": 0, "history": 1, "target": 1, "generated": 0, "total": 2,
    }
    assert result["model_forwards"] == 7
    assert result["image_forwards"] == 1
    assert [row["physical_slot_position"] for row in result["feedback_boundaries"]] == [4, 8]
    assert [row["visible_boundary_index"] for row in result["feedback_boundaries"]] == [1, 4]
    assert set(result["feedback_sources"]) == {0, 1}
    assert [row["physical_position"] for row in result["physical_trace"]] == list(range(2, 10))
    assert all(row["mrope_position"] == [row["physical_position"],
                                          row["physical_position"] + 100,
                                          row["physical_position"] + 200]
               for row in result["physical_trace"])
    assert all(call["attention_length"] == call["cache_position"][-1] + 1 for call in qwen.model.calls)


def test_feedback_override_checks_both_occurrence_and_visible_boundary() -> None:
    qwen, inputs, prompt = fixture()
    source = torch.ones(HIDDEN)
    with pytest.raises(ValueError, match="visible boundary mismatch"):
        replay_visible(
            qwen, inputs, prompt_ids=prompt, history_ids=[3, BOX_END], target_ids=[4], arm="F",
            feedback_source_overrides={0: FeedbackSourceOverride(source, visible_boundary_index=0)},
        )
    qwen, inputs, prompt = fixture()
    with pytest.raises(ValueError, match="was not reached"):
        replay_visible(
            qwen, inputs, prompt_ids=prompt, history_ids=[3], target_ids=[4], arm="F",
            feedback_source_overrides={0: source},
        )
    qwen, inputs, prompt = fixture()
    with pytest.raises(ValueError, match="S arm rejects"):
        replay_visible(
            qwen, inputs, prompt_ids=prompt, history_ids=[3, BOX_END], target_ids=[4], arm="S",
            feedback_source_overrides={0: source},
        )


def test_detaching_only_feedback_source_removes_its_grad_but_keeps_native_history_grad() -> None:
    qwen, inputs, prompt = fixture()
    live = replay_visible(
        qwen, inputs, prompt_ids=prompt, history_ids=[3, BOX_END], target_ids=[4, 5],
        arm="F", capture_feedback_sources=True, capture_native_history_activation=True,
    )
    visible_nll_sum(live).backward()
    live_source = live["feedback_sources"][0]
    assert live_source.grad is not None
    assert float(live_source.grad.norm()) > 0
    assert live["native_history_activation"].grad is not None
    assert float(live["native_history_activation"].grad.norm()) > 0

    qwen, inputs, prompt = fixture()
    detached = replay_visible(
        qwen, inputs, prompt_ids=prompt, history_ids=[3, BOX_END], target_ids=[4, 5],
        arm="F", capture_feedback_sources=True, detach_feedback_source_boundaries=(0,),
        capture_native_history_activation=True,
    )
    visible_nll_sum(detached).backward()
    detached_source = detached["feedback_sources"][0]
    assert detached_source.grad is None or float(detached_source.grad.norm()) == 0
    # The same later-token loss still reaches an exact history K projection
    # through cached state after the final post-norm feedback source is cut.
    native_history = detached["native_history_activation"]
    assert native_history.grad is not None
    assert float(native_history.grad.norm()) > 0


def test_generation_keeps_internal_slots_out_of_visible_stream_and_budget() -> None:
    # Prompt logit -> BOX_END; slot logit -> 7; visible 7 logit -> EOS.
    qwen, inputs, prompt = fixture(scripted={1: BOX_END, 3: 7, 4: EOS})
    result = generate_visible(
        qwen, inputs, prompt_ids=prompt, arm="F", max_visible_tokens=5,
    )
    assert result["visible_token_ids"] == [BOX_END, 7, EOS]
    assert result["text"] == f"{BOX_END}|7|{EOS}"
    assert result["visible_generated_tokens"] == 3
    assert result["internal_slot_count"] == 1
    assert result["model_forwards"] == 5
    assert result["image_forwards"] == 1
    assert result["finish_reason"] == "eos"
    assert result["eos"] and not result["cap"]
    assert result["decode_contract"] == {
        "max_visible_tokens": 5,
        "eos_token_id": EOS,
        "do_sample": False,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
    }
    slots = [row for row in result["physical_trace"] if row["kind"] == "internal_slot"]
    assert len(slots) == 1
    assert all(row.get("token_id") != BOX_END or row["kind"] == "visible"
               for row in result["physical_trace"])


def test_teacher_kl_maps_visible_ordinals_without_input_parity_claim() -> None:
    teacher = torch.tensor([[2.0, 0.0], [0.5, 1.5]], dtype=torch.float32).log_softmax(-1)
    student = torch.tensor([[1.0, -1.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    value = mapped_teacher_kl(student, teacher, reduction="sum")
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    assert student.grad is not None and float(student.grad.norm()) > 0
    with pytest.raises(ValueError, match="identical"):
        mapped_teacher_kl(student[:1], teacher)


def test_full_bank_has_exact_bound_train_dev_materialization_coverage() -> None:
    from src.config.inference import load_research_infer_config
    from probes.dora_owner_learning.route_access import CONFIG

    bank = json.loads((ROOT / "data-v2/supervision-bank.json").read_text())
    config = load_research_infer_config(CONFIG).config
    reference, raw = _materialization_raw_sources(bank, config)
    expected = {record["example_id"] for record in bank["records"]}
    assert expected <= set(raw)
    assert len(expected) == 11
    assert reference == bank["sources"]["native_n16_training_input"]


def test_endpoint_embedded_native_row_is_recovered_from_exact_bound_ordinal() -> None:
    selection = json.loads((ROOT / "evaluation/selection-v2.json").read_text())
    record = selection["records"][0]
    raw = _bound_endpoint_raw(record, selection["native_source"])
    assert raw.example_id == record["example_id"]
    assert Path(raw.image.path).resolve() == Path(record["image_path"]).resolve()


def test_prepared_input_hash_covers_tensor_values_and_metadata() -> None:
    one = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    two = {**one, "input_ids": torch.tensor([[1, 3]])}
    assert prepared_inputs_sha256(one) == prepared_inputs_sha256(dict(reversed(list(one.items()))))
    assert prepared_inputs_sha256(one) != prepared_inputs_sha256(two)
