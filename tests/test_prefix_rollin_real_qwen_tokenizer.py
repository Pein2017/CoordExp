from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import CoordinateTokenBox, NormalizedDetectionObject
from src.detection.objective import build_compact_prefix_rollin_example
from src.detection.token_types import build_compact_token_type_groups

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COORD_EXP_ROOT = Path("/data/CoordExp")
_QWEN3_VL_2B_COORD_EXP_REMOTE_REL = Path(
    "model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
_QWEN3_VL_2B_COORD_EXP_LOCAL_REL = Path(
    "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
_QWEN3_VL_2B_COORD_EXP_CANDIDATES = (
    _REPO_ROOT / _QWEN3_VL_2B_COORD_EXP_REMOTE_REL,
    _COORD_EXP_ROOT / _QWEN3_VL_2B_COORD_EXP_REMOTE_REL,
    _REPO_ROOT / _QWEN3_VL_2B_COORD_EXP_LOCAL_REL,
    _COORD_EXP_ROOT / _QWEN3_VL_2B_COORD_EXP_LOCAL_REL,
)

_QWEN3_VL_2B_COORD_EXP_ADAPTER_REL = Path(
    "outputs/stage1_2b/recursive_detection_ce/"
    "compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/"
    "compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/"
    "v0-20260504-071356/checkpoint-3664"
)
_QWEN3_VL_2B_COORD_EXP_ADAPTER_CANDIDATES = (
    _REPO_ROOT / _QWEN3_VL_2B_COORD_EXP_ADAPTER_REL,
    _COORD_EXP_ROOT / _QWEN3_VL_2B_COORD_EXP_ADAPTER_REL,
)


def _qwen3_vl_coordexp_path() -> Path:
    for path in _QWEN3_VL_2B_COORD_EXP_CANDIDATES:
        if (
            (path / "config.json").is_file()
            and (path / "tokenizer_config.json").is_file()
            and (
                (path / "tokenizer.json").is_file()
                or (
                    (path / "vocab.json").is_file()
                    and (path / "merges.txt").is_file()
                )
            )
        ):
            return path
    pytest.skip(
        "local Qwen3-VL-2B-Instruct-coordexp tokenizer/config cache is unavailable"
    )


def _qwen3_vl_coordexp_adapter_path() -> Path:
    for path in _QWEN3_VL_2B_COORD_EXP_ADAPTER_CANDIDATES:
        if (path / "adapter_model.safetensors").exists():
            return path
    pytest.skip("local compact-full support2 adapter checkpoint is unavailable")


def _safetensor_shape(path: Path, tensor_name: str) -> tuple[int, ...]:
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(tensor_name).get_shape())


def _safetensor_int_vector(path: Path, tensor_name: str) -> list[int]:
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return list(handle.get_tensor(tensor_name).tolist())


def _safetensor_keys(path: Path) -> frozenset[str]:
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return frozenset(handle.keys())


def _model_safetensor_shape(tensor_name: str) -> tuple[int, ...]:
    import json

    model_path = _qwen3_vl_coordexp_path()
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_name = index["weight_map"].get(tensor_name)
        if shard_name is None:
            pytest.skip(f"tensor {tensor_name!r} is absent from the local model index")
        return _safetensor_shape(model_path / shard_name, tensor_name)

    safetensor_path = model_path / "model.safetensors"
    if safetensor_path.exists():
        return _safetensor_shape(safetensor_path, tensor_name)
    pytest.skip("local Qwen3-VL safetensor weights are unavailable")


@lru_cache(maxsize=1)
def _real_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        str(_qwen3_vl_coordexp_path()),
        trust_remote_code=True,
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def _real_model_config():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoConfig.from_pretrained(
        str(_qwen3_vl_coordexp_path()),
        trust_remote_code=True,
        local_files_only=True,
    )


def _objects() -> tuple[NormalizedDetectionObject, NormalizedDetectionObject]:
    return (
        NormalizedDetectionObject(
            normalized_object_index=0,
            source_object_index=0,
            object_instance_id="inst-a",
            desc="cat",
            bbox_2d=CoordinateTokenBox(
                "<|coord_10|>",
                "<|coord_20|>",
                "<|coord_30|>",
                "<|coord_40|>",
            ),
            category_id=1,
            category_name="cat",
            coco_ann_id=101,
        ),
        NormalizedDetectionObject(
            normalized_object_index=1,
            source_object_index=1,
            object_instance_id="inst-b",
            desc="dog",
            bbox_2d=CoordinateTokenBox(
                "<|coord_50|>",
                "<|coord_60|>",
                "<|coord_70|>",
                "<|coord_80|>",
            ),
            category_id=2,
            category_name="dog",
            coco_ann_id=102,
        ),
    )


def test_real_qwen_coordexp_tokenizer_uses_im_end_as_only_training_eos() -> None:
    tokenizer = _real_tokenizer()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.eos_token_id == im_end_id
    assert tokenizer.encode("<|im_end|>", add_special_tokens=False) == [im_end_id]
    for terminator in ("<|endoftext|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(terminator)
        encoded = tokenizer.encode(terminator, add_special_tokens=False)
        assert encoded != [im_end_id]
        if isinstance(token_id, int):
            assert token_id != im_end_id


def test_real_qwen_coordexp_model_config_covers_coord_and_struct_token_rows() -> None:
    tokenizer = _real_tokenizer()
    config = _real_model_config()
    vocab_size = int(config.text_config.vocab_size)
    token_ids = {
        "object_ref_start": tokenizer.convert_tokens_to_ids(OBJECT_REF_START_TOKEN),
        "box_start": tokenizer.convert_tokens_to_ids(BOX_START_TOKEN),
        "coord_0": tokenizer.convert_tokens_to_ids("<|coord_0|>"),
        "coord_500": tokenizer.convert_tokens_to_ids("<|coord_500|>"),
        "coord_999": tokenizer.convert_tokens_to_ids("<|coord_999|>"),
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    assert token_ids == {
        "object_ref_start": 151646,
        "box_start": 151648,
        "coord_0": 151670,
        "coord_500": 152170,
        "coord_999": 152669,
        "im_end": 151645,
    }
    for token_text, token_id in (
        (OBJECT_REF_START_TOKEN, token_ids["object_ref_start"]),
        (BOX_START_TOKEN, token_ids["box_start"]),
        ("<|im_end|>", token_ids["im_end"]),
    ):
        assert tokenizer.encode(token_text, add_special_tokens=False) == [token_id]
        assert token_id < vocab_size
    for coord in range(1000):
        token_text = f"<|coord_{coord}|>"
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        assert token_id == 151670 + coord
        assert tokenizer.encode(token_text, add_special_tokens=False) == [token_id]
        assert token_id < vocab_size
    assert len(tokenizer) == vocab_size


def test_real_qwen_coordexp_embedding_rows_match_tokenizer_vocab_without_loading_weights() -> None:
    config = _real_model_config()
    vocab_size = int(config.text_config.vocab_size)
    hidden_size = int(config.text_config.hidden_size)

    assert _model_safetensor_shape("model.language_model.embed_tokens.weight") == (
        vocab_size,
        hidden_size,
    )


def test_real_qwen_coordexp_adapter_token_embeddings_rows_match_compact_surface() -> None:
    tokenizer = _real_tokenizer()
    config = _real_model_config()
    adapter_path = _qwen3_vl_coordexp_adapter_path()
    adapter_safetensors = adapter_path / "adapter_model.safetensors"

    coord_surface_rows = 1000
    struct_surface_rows = 2
    expected_rows = coord_surface_rows + struct_surface_rows
    expected_token_ids = [
        tokenizer.convert_tokens_to_ids(OBJECT_REF_START_TOKEN),
        tokenizer.convert_tokens_to_ids(BOX_START_TOKEN),
        *[
            tokenizer.convert_tokens_to_ids(f"<|coord_{coord}|>")
            for coord in range(coord_surface_rows)
        ],
    ]

    assert _safetensor_shape(
        adapter_safetensors,
        "base_model.model.token_embeddings_adapter.token_ids",
    ) == (expected_rows,)
    assert _safetensor_shape(
        adapter_safetensors,
        "base_model.model.token_embeddings_adapter.embed_offset",
    ) == (expected_rows, int(config.text_config.hidden_size))
    assert _safetensor_int_vector(
        adapter_safetensors,
        "base_model.model.token_embeddings_adapter.token_ids",
    ) == expected_token_ids
    assert not any(
        key.endswith(".token_embeddings_adapter.head_offset")
        for key in _safetensor_keys(adapter_safetensors)
    )


def test_real_qwen_coordexp_token_type_groups_exclude_text_terminators_from_eos() -> None:
    tokenizer = _real_tokenizer()
    groups = build_compact_token_type_groups(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    assert groups.eos == frozenset({im_end_id})
    for terminator in ("<|endoftext|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(terminator)
        assert tokenizer.encode(terminator, add_special_tokens=False) != [im_end_id]
        if isinstance(token_id, int):
            assert token_id in groups.excluded_control
            assert token_id not in groups.eos
            assert token_id not in groups.desc
    assert tokenizer.convert_tokens_to_ids(OBJECT_REF_START_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids(BOX_START_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids("<|coord_0|>") in groups.coord
    assert tokenizer.convert_tokens_to_ids("<|coord_999|>") in groups.coord


@pytest.mark.parametrize("k", [0, 1, 2])
def test_real_qwen_coordexp_prefix_rollin_alignment_uses_template_im_end(k: int) -> None:
    tokenizer = _real_tokenizer()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    objects = _objects()

    example = build_compact_prefix_rollin_example(
        objects=objects,
        rollin_order=objects,
        k=k,
        tokenizer=tokenizer,
    )

    stop_span = example.tokenized.assistant_stop_token_span
    assert stop_span is not None
    assert stop_span.end - stop_span.start == 1
    assert example.input_ids[stop_span.start] == im_end_id
    assert "<|im_end|>" not in example.rendered_assistant.text
    assert example.tokenized.chat_text.endswith("<|im_end|>\n")
    for terminator in ("<|endoftext|>", "<|end_of_text|>"):
        assert terminator not in example.tokenized.chat_text

    target_positions = {
        target.position for target in example.recursive_detection_targets.token_targets
    }
    active_label_positions = {
        index for index, token_id in enumerate(example.labels) if token_id != -100
    }
    assert target_positions == active_label_positions
    for target in example.recursive_detection_targets.token_targets:
        assert example.input_ids[target.position] == target.teacher_token_id
        assert example.labels[target.position] == target.teacher_token_id

    for prefix_position in example.debug_spans["rollin_prefix"].token_positions:
        assert example.labels[prefix_position] == -100

    eos_targets = [
        target
        for target in example.recursive_detection_targets.token_targets
        if target.position == stop_span.start
    ]
    assert len(eos_targets) == 1
    assert eos_targets[0].teacher_token_id == im_end_id

    forbidden_singleton_ids = {
        token_id
        for token_id in (
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        )
        if isinstance(token_id, int)
    }
    teacher_ids = {
        target.teacher_token_id
        for target in example.recursive_detection_targets.token_targets
    }
    active_label_ids = {token_id for token_id in example.labels if token_id != -100}
    assert forbidden_singleton_ids.isdisjoint(teacher_ids)
    assert forbidden_singleton_ids.isdisjoint(active_label_ids)
