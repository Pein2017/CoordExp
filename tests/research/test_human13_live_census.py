from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research import human13_live_census as live
from scripts.research.human13_live_census import capture_human13_live_census
from scripts.research.materialize_human13_no_update_census import (
    A1SegmentPlan,
    CoherentSitePlan,
    Human13CensusPlan,
    NativeRowPlan,
    TrieSitePlan,
    run_census_with_exact_logits,
)
from src.inference.backend import token_ids_sha256


IMAGE_TOKEN_ID = 151655
PROMPT_IDS = (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID)


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int = 2


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merged_visual_tokens: int = 4
    plan: FakeImagePlan = field(default_factory=FakeImagePlan)
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.zeros((16, 2)))


@dataclass(frozen=True)
class Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merge_size: int = 2
    image_token_id: int = IMAGE_TOKEN_ID
    image_encoding: FakeImageEncoding = field(default_factory=FakeImageEncoding)


class FakeTokenizer:
    def __len__(self) -> int:
        return 512


def _plan() -> Human13CensusPlan:
    prompt_sha = token_ids_sha256(PROMPT_IDS)
    row_a = NativeRowPlan(
        1, "owner-a", "row-a", "k-a", 0, (100, 101), ("boundary", "row_terminator")
    )
    row_b = NativeRowPlan(
        1,
        "owner-b",
        "row-b",
        "k-b",
        1,
        (200, 201, 202),
        ("boundary", "coordinate", "row_terminator"),
    )
    coherent = (
        CoherentSitePlan(
            "a1:1:site:1",
            "a1:1",
            "a1_full_h",
            1,
            "owner-a",
            "row-a",
            0,
            0,
            1,
            (50,),
            100,
            "boundary",
            prompt_sha,
            "b" * 64,
        ),
        CoherentSitePlan(
            "a1:1:site:2",
            "a1:1",
            "a1_full_h",
            1,
            "owner-a",
            "row-a",
            0,
            1,
            2,
            (50, 100),
            101,
            "row_terminator",
            prompt_sha,
            "b" * 64,
        ),
        CoherentSitePlan(
            "a1:1:site:3",
            "a1:1",
            "a1_full_h",
            1,
            "owner-b",
            "row-b",
            1,
            0,
            3,
            (50, 100, 101),
            200,
            "boundary",
            prompt_sha,
            "b" * 64,
        ),
        CoherentSitePlan(
            "a1:1:site:4",
            "a1:1",
            "a1_full_h",
            1,
            "owner-b",
            "row-b",
            1,
            1,
            4,
            (50, 100, 101, 200),
            201,
            "coordinate",
            prompt_sha,
            "b" * 64,
        ),
        CoherentSitePlan(
            "a1:1:site:5",
            "a1:1",
            "a1_full_h",
            1,
            "owner-b",
            "row-b",
            1,
            2,
            5,
            (50, 100, 101, 200, 201),
            202,
            "row_terminator",
            prompt_sha,
            "b" * 64,
        ),
    )
    trie = (
        TrieSitePlan(
            "trie:1:root",
            "native-trie:1",
            "native_row_trie",
            1,
            (),
            (50,),
            (100, 200),
            ((100, "boundary"), (200, "row_terminator")),
            prompt_sha,
            "b" * 64,
        ),
        TrieSitePlan(
            "trie:1:b2",
            "native-trie:1",
            "native_row_trie",
            1,
            (200, 201),
            (50, 200, 201),
            (202,),
            ((202, "row_terminator"),),
            prompt_sha,
            "b" * 64,
        ),
        TrieSitePlan(
            "trie:1:a",
            "native-trie:1",
            "native_row_trie",
            1,
            (100,),
            (50, 100),
            (101,),
            ((101, "row_terminator"),),
            prompt_sha,
            "b" * 64,
        ),
        TrieSitePlan(
            "trie:1:b",
            "native-trie:1",
            "native_row_trie",
            1,
            (200,),
            (50, 200),
            (201,),
            ((201, "row_terminator"),),
            prompt_sha,
            "b" * 64,
        ),
    )
    return Human13CensusPlan(
        schema_version="human13_no_update_census_plan.v1",
        manifest_sha256="c" * 64,
        discovery_binding_sha256="d" * 64,
        selected_rows=(row_a, row_b),
        segments=(
            A1SegmentPlan(
                "a1:1",
                "a1_full_h",
                1,
                prompt_sha,
                "b" * 64,
                (50,),
                ("row-a", "row-b"),
                (50, 100, 101, 200, 201, 202),
            ),
        ),
        coherent_sites=coherent,
        trie_sites=trie,
        frozen_targets={"manifest_sha256": "c" * 64},
        actions={},
    )


def _vector(position: int, vocab_size: int = 512) -> torch.Tensor:
    result = torch.arange(vocab_size, dtype=torch.float32)
    result[0] = float(position)
    return result


class FakePackedForward:
    def __init__(self, *, omit_last: bool = False, vocab_size: int = 512) -> None:
        self.calls: list[tuple[Any, Any, Any, Any, tuple[int, ...]]] = []
        self.omit_last = omit_last
        self.vocab_size = vocab_size

    def __call__(
        self,
        model: Any,
        runtime: Any,
        tokenizer: Any,
        packed: Any,
        positions: tuple[int, ...],
    ) -> Any:
        self.calls.append((model, runtime, tokenizer, packed, positions))
        returned = tuple(reversed(positions))
        if self.omit_last:
            returned = returned[1:]
        return SimpleNamespace(
            logits=torch.stack(
                tuple(_vector(position, self.vocab_size) for position in returned)
            ).unsqueeze(0),
            logits_position_ids=returned,
        )


class FakeHFScorer:
    model_dtype = "torch.float32"
    attention_implementation = "sdpa"

    def __init__(self, *, omit_last: bool = False) -> None:
        self.calls: list[tuple[Any, tuple[int, ...]]] = []
        self.omit_last = omit_last

    def score_causal_logits(
        self,
        encoded_example: Any,
        causal_positions: tuple[int, ...],
    ) -> Any:
        self.calls.append((encoded_example, causal_positions))
        returned = tuple(reversed(causal_positions))
        if self.omit_last:
            returned = returned[1:]
        return SimpleNamespace(
            logits=torch.stack(
                tuple(_vector(position) for position in returned)
            ).unsqueeze(0),
            logits_position_ids=returned,
        )


def _by_key(result: Any) -> dict[tuple[str, str], Any]:
    return {(item.site_id, item.surface): item for item in result.evidence}


def test_capture_maps_target_to_predecessor_and_reuses_coherent_trie_contexts() -> None:
    packed = FakePackedForward()
    hf = FakeHFScorer()
    model = object()
    runtime = SimpleNamespace(accelerator=SimpleNamespace(device="cpu"))
    tokenizer = FakeTokenizer()

    result = capture_human13_live_census(
        plan=_plan(),
        prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
        packed_model=model,
        packed_runtime=runtime,
        tokenizer=tokenizer,
        hf_scorer=hf,
        packed_forward=packed,
    )

    evidence = _by_key(result)
    # Prompt width 5, P_clean width 1: target 100 is predicted at local position 5.
    assert evidence[("a1:1:site:1", "packed")].logits[0].item() == 5.0
    assert evidence[("a1:1:site:2", "packed")].logits[0].item() == 6.0
    assert evidence[("a1:1:site:3", "packed")].logits[0].item() == 7.0
    assert evidence[("a1:1:site:4", "packed")].logits[0].item() == 8.0
    assert evidence[("a1:1:site:5", "packed")].logits[0].item() == 9.0
    assert evidence[("a1:1:site:1", "hf")].logits[0].item() == 5.0
    assert evidence[("a1:1:site:2", "hf")].logits[0].item() == 6.0
    assert evidence[("a1:1:site:3", "hf")].logits[0].item() == 7.0
    assert evidence[("a1:1:site:4", "hf")].logits[0].item() == 8.0
    assert evidence[("a1:1:site:5", "hf")].logits[0].item() == 9.0

    # Root and row-a trie states are identical to two coherent chain states.
    assert torch.equal(
        evidence[("trie:1:root", "trie")].logits,
        evidence[("a1:1:site:1", "packed")].logits,
    )
    assert torch.equal(
        evidence[("trie:1:a", "trie")].logits,
        evidence[("a1:1:site:2", "packed")].logits,
    )
    # The row-b trie prefix diverges from A1 and owns the one isolated segment.
    assert evidence[("trie:1:b", "trie")].logits[0].item() == 17.0
    assert evidence[("trie:1:b2", "trie")].logits[0].item() == 18.0
    assert result.receipt["logical_segment_count"] == 2
    assert result.receipt["coherent_segment_count"] == 1
    assert result.receipt["isolated_trie_segment_count"] == 1
    assert result.receipt["packed_forward_count"] == 1
    assert len(hf.calls) == 1
    assert hf.calls[0][1] == (5, 6, 7, 8, 9)
    assert hf.calls[0][0].human13_image_id == 1
    assert packed.calls[0][:3] == (model, runtime, tokenizer)
    census = run_census_with_exact_logits(plan=_plan(), evidence=result.evidence)
    assert census["coherent_chain"]["site_count"] == 5


def test_empty_p_clean_uses_the_last_prompt_token_as_first_causal_site() -> None:
    plan = _plan()
    no_clean = replace(
        plan,
        segments=(
            replace(
                plan.segments[0],
                fixed_prefix_token_ids=(),
                token_ids=plan.segments[0].token_ids[1:],
            ),
        ),
        coherent_sites=tuple(
            replace(
                site,
                segment_token_offset=site.segment_token_offset - 1,
                segment_prefix_token_ids=site.segment_prefix_token_ids[1:],
            )
            for site in plan.coherent_sites
        ),
        trie_sites=tuple(
            replace(site, model_prefix_token_ids=site.model_prefix_token_ids[1:])
            for site in plan.trie_sites
        ),
    )

    result = capture_human13_live_census(
        plan=no_clean,
        prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
        packed_model=object(),
        packed_runtime=SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
        tokenizer=FakeTokenizer(),
        hf_scorer=FakeHFScorer(),
        packed_forward=FakePackedForward(),
    )

    evidence = _by_key(result)
    assert evidence[("a1:1:site:1", "packed")].logits[0].item() == 4.0
    assert evidence[("trie:1:root", "trie")].logits[0].item() == 4.0


def test_capture_fails_when_packed_or_hf_position_coverage_is_incomplete() -> None:
    kwargs = {
        "plan": _plan(),
        "prompt_skeletons": {1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
        "packed_model": object(),
        "packed_runtime": SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
        "tokenizer": FakeTokenizer(),
    }
    with pytest.raises(ValueError, match="packed logits position coverage"):
        capture_human13_live_census(
            **kwargs,
            hf_scorer=FakeHFScorer(),
            packed_forward=FakePackedForward(omit_last=True),
        )
    with pytest.raises(ValueError, match="HF logits position coverage"):
        capture_human13_live_census(
            **kwargs,
            hf_scorer=FakeHFScorer(omit_last=True),
            packed_forward=FakePackedForward(),
        )


def test_prepare_rejects_a_plan_missing_the_first_coherent_target_site() -> None:
    plan = _plan()
    missing_first = replace(plan, coherent_sites=plan.coherent_sites[1:])

    with pytest.raises(ValueError, match="coherent A1 target offsets"):
        capture_human13_live_census(
            plan=missing_first,
            prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
            packed_model=object(),
            packed_runtime=SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
            tokenizer=FakeTokenizer(),
            hf_scorer=FakeHFScorer(),
            packed_forward=FakePackedForward(),
        )


@pytest.mark.parametrize(
    ("dtype", "attention", "message"),
    (
        ("torch.bfloat16", "sdpa", "fp32"),
        ("torch.float32", "flash_attention_2", "SDPA"),
    ),
)
def test_hf_capture_requires_exact_fp32_sdpa_session(
    dtype: str,
    attention: str,
    message: str,
) -> None:
    hf = FakeHFScorer()
    hf.model_dtype = dtype
    hf.attention_implementation = attention

    with pytest.raises(ValueError, match=message):
        capture_human13_live_census(
            plan=_plan(),
            prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
            packed_model=object(),
            packed_runtime=SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
            tokenizer=FakeTokenizer(),
            hf_scorer=hf,
            packed_forward=FakePackedForward(),
        )


def test_prompt_skeleton_hash_and_full_vocab_are_fail_closed() -> None:
    skeleton = Skeleton("image:1", (*PROMPT_IDS[:-1], 99), len(PROMPT_IDS))
    with pytest.raises(ValueError, match="prompt token identity"):
        capture_human13_live_census(
            plan=_plan(),
            prompt_skeletons={1: skeleton},
            packed_model=object(),
            packed_runtime=SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
            tokenizer=FakeTokenizer(),
            hf_scorer=FakeHFScorer(),
            packed_forward=FakePackedForward(),
        )

    with pytest.raises(ValueError, match="full vocabulary"):
        capture_human13_live_census(
            plan=_plan(),
            prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
            packed_model=object(),
            packed_runtime=SimpleNamespace(accelerator=SimpleNamespace(device="cpu")),
            tokenizer=FakeTokenizer(),
            hf_scorer=FakeHFScorer(),
            packed_forward=FakePackedForward(vocab_size=511),
        )


def test_default_packed_capture_uses_current_compact_qwen_forward_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    runtime = SimpleNamespace(accelerator=SimpleNamespace(device="cpu"))
    tokenizer = FakeTokenizer()
    calls: dict[str, Any] = {}

    def fake_build(
        pack: Any,
        encoded_examples: Any,
        position_inputs: Any,
        **kwargs: Any,
    ) -> Any:
        calls["build"] = (pack, encoded_examples, position_inputs, kwargs)
        return SimpleNamespace()

    def fake_run(received_model: Any, inputs: Any, **kwargs: Any) -> Any:
        calls["run"] = (received_model, inputs, kwargs)
        positions = calls["build"][3]["logits_to_keep_positions"]
        return SimpleNamespace(
            logits=torch.stack(
                tuple(_vector(position) for position in positions)
            ).unsqueeze(0),
            logits_position_ids=positions,
        )

    monkeypatch.setattr(live, "build_qwen_forward_inputs", fake_build)
    monkeypatch.setattr(live, "run_qwen_forward", fake_run)

    result = capture_human13_live_census(
        plan=_plan(),
        prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
        packed_model=model,
        packed_runtime=runtime,
        tokenizer=tokenizer,
        hf_scorer=FakeHFScorer(),
    )

    assert calls["build"][3]["device"] == "cpu"
    assert calls["build"][3]["logits_to_keep_positions"] == (
        5,
        6,
        7,
        8,
        9,
        17,
        18,
    )
    assert calls["run"][0] is model
    assert calls["run"][2]["expected_vocab_size"] == 512
    assert calls["run"][2]["capture_fa2_branch"] is True
    assert calls["run"][2]["require_fa2_branch_proof"] is True
    assert result.receipt["evidence_count"] == 14


def test_default_packed_capture_rebuilds_fa2_metadata_on_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_device = torch.device("meta")
    calls: dict[str, Any] = {}

    def fake_build(
        pack: Any,
        encoded_examples: Any,
        position_inputs: Any,
        **kwargs: Any,
    ) -> Any:
        del pack, encoded_examples, position_inputs
        plan = kwargs["fa2_varlen_plan"]
        calls["q_device"] = plan.cu_seq_lens_q.device
        calls["k_device"] = plan.cu_seq_lens_k.device
        calls["positions"] = kwargs["logits_to_keep_positions"]
        return SimpleNamespace()

    def fake_run(model: Any, inputs: Any, **kwargs: Any) -> Any:
        del model, inputs, kwargs
        positions = calls["positions"]
        return SimpleNamespace(
            logits=torch.stack(
                tuple(_vector(position) for position in positions)
            ).unsqueeze(0),
            logits_position_ids=positions,
        )

    monkeypatch.setattr(live, "build_qwen_forward_inputs", fake_build)
    monkeypatch.setattr(live, "run_qwen_forward", fake_run)

    capture_human13_live_census(
        plan=_plan(),
        prompt_skeletons={1: Skeleton("image:1", PROMPT_IDS, len(PROMPT_IDS))},
        packed_model=object(),
        packed_runtime=SimpleNamespace(
            accelerator=SimpleNamespace(device=runtime_device)
        ),
        tokenizer=FakeTokenizer(),
        hf_scorer=FakeHFScorer(),
    )

    assert calls["q_device"] == runtime_device
    assert calls["k_device"] == runtime_device
