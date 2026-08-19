from __future__ import annotations

import copy
from dataclasses import dataclass
import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

from src.config.loader import load_train_config
from src.losses.vocab import TokenVocabularyGroups
from src.training import pack_cache
from src.training.supervised_trainer import SupervisedMicroStep


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")

TOKENIZER_ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
PROCESSOR_ASSET_NAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
MODEL_ASSET_NAMES = (
    "README.md",
    "config.json",
    "configuration.json",
    "coord_init.json",
    "coord_tokens.json",
    "generation_config.json",
    "model.safetensors.index.json",
)
NESTED_ASSET_NAMES = (
    "additional_chat_templates/default.jinja",
    "tokenizer.fast.json",
)
ALL_FRONTEND_ASSET_NAMES = tuple(
    sorted(
        (
            *TOKENIZER_ASSET_NAMES,
            *PROCESSOR_ASSET_NAMES,
            *MODEL_ASSET_NAMES,
            *NESTED_ASSET_NAMES,
        )
    )
)
EXPECTED_DETERMINANT_OWNERS = {
    "dataset_content": "src/data/examples.py",
    "template_config": "src/templates/renderer.py",
    "packing_config": "src/packing/planner.py",
    "processor_config": "src/qwen/runtime_loading.py",
    "ordering_config": "src/data/examples.py",
    "augmentation_config": "src/augmentation/geometry.py",
    "model_config_assets": "src/qwen/runtime_loading.py",
    "processor_assets": "src/qwen/runtime_loading.py",
    "tokenizer_assets": "src/qwen/runtime_loading.py",
    "token_identity": "src/qwen/tokens.py",
    "realized_vocab_groups": "src/losses/vocab.py",
    "encoding_runtime": "src/qwen/encoding.py",
    "augmentation_factory": "src/augmentation/factory.py",
    "augmentation_processor": "src/augmentation/processor.py",
    "coordinate_targets": "src/coordinate_targets.py",
    "dataset_geometry": "src/data/geometry.py",
    "dataset_image_resolver": "src/data/images.py",
    "dataset_jsonl_loader": "src/data/jsonl.py",
    "template_spans": "src/templates/spans.py",
    "renderer": "src/templates/renderer.py",
    "parser": "src/data/examples.py",
    "image_loader": "src/qwen/images.py",
    "pack_planner": "src/packing/planner.py",
    "supervision_mapper": "src/packing/supervision.py",
    "supervision_tokens": "src/supervision/tokens.py",
    "mrope_position_ids": "src/qwen/positions.py",
    "qwen_fa2_boundaries": "src/qwen/fa2.py",
    "qwen_forward_payload": "src/qwen/forward.py",
    "micro_step_runtime_config": "src/training/pipeline.py",
    "micro_step_schema": "src/training/supervised_trainer.py",
    "cache_serializer": "src/training/pack_cache.py",
}
UNIQUE_OWNER_PATHS = tuple(sorted(set(EXPECTED_DETERMINANT_OWNERS.values())))

RETIRED_CACHE_VERSIONS = (
    "coordexp-swift-pack-cache-v1",
    "coordexp-swift-pack-cache-v2",
)

DISABLED_AUGMENTATION = {
    "split": "train",
    "mode": "disabled",
    "policy": "geometry_flips",
    "enabled": False,
    "seed": 17,
    "input_example_count": 1,
    "output_example_count": 1,
    "presentation_count": 1,
    "object_ordering": "source_order",
}


@dataclass
class _Identity:
    payload: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


class _Tokenizer:
    chat_template = "unit-test-chat-template"

    def __init__(
        self,
        *,
        image_pad_token_id: int = 11,
        all_special_ids: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12),
        control_token_ids: dict[str, int] | None = None,
    ) -> None:
        self.image_pad_token_id = image_pad_token_id
        self.all_special_ids = all_special_ids
        self.control_token_ids = dict(control_token_ids or {"<think>": 12})

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|image_pad|>":
            return self.image_pad_token_id
        return self.control_token_ids.get(token)


class _Processor:
    chat_template = "unit-test-chat-template"


class _Components:
    def __init__(
        self,
        model_root: Path,
        *,
        token_identity: dict[str, Any] | None = None,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        self.base_model_path = model_root
        self.processor = _Processor()
        self.tokenizer = tokenizer or _Tokenizer()
        self.processor_identity = _Identity(
            {
                "processor_class": "UnitProcessor",
                "tokenizer_class": "UnitTokenizer",
                "image_processor_class": "UnitImageProcessor",
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
            }
        )
        self.token_identity = _Identity(
            copy.deepcopy(token_identity or _token_identity_payload())
        )
        self.package_versions = {
            "tokenizers": "unit-test",
            "transformers": "unit-test",
        }


@dataclass
class _CacheInputs:
    config: Any
    components: _Components
    dataset: Any
    vocab_groups: TokenVocabularyGroups


@pytest.fixture
def cache_inputs(tmp_path: Path) -> _CacheInputs:
    dataset_path = tmp_path / "train.coord.jsonl"
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image-revision-a")
    dataset_path.write_text(
        json.dumps(
            {
                "example_id": "cache-registry-test",
                "image": {"path": image_path.name, "width": 1, "height": 1},
                "objects": [
                    {
                        "object_id": "object-0",
                        "description": "object",
                        "bbox": [0, 0, 1, 1],
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    model_root = tmp_path / "model"
    model_root.mkdir()
    for asset_name in ALL_FRONTEND_ASSET_NAMES:
        asset_path = model_root / asset_name
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(
            '{"revision":"a"}\n',
            encoding="utf-8",
        )
    weight_shard = model_root / "model-00001-of-00001.safetensors"
    weight_shard.write_bytes(b"excluded-model-weight-a")
    (model_root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"revision": "a"},
                "weight_map": {"model.weight": weight_shard.name},
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"base_model": str(model_root)}),
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset_path)}
                    )
                }
            ),
        }
    )
    components = _Components(model_root)
    vocab_groups = _groups(blocked=(10, 11, 12))
    # Keep the realized value discoverable for implementations that choose to
    # carry it on the component bundle.  The helper below also passes the value
    # explicitly once the production builder accepts that dependency.
    components.vocab_groups = vocab_groups
    return _CacheInputs(
        config=config,
        components=components,
        dataset=config.data.train,
        vocab_groups=vocab_groups,
    )


@pytest.mark.parametrize("asset_name", ALL_FRONTEND_ASSET_NAMES)
def test_same_path_frontend_asset_edit_changes_fingerprint(
    cache_inputs: _CacheInputs,
    asset_name: str,
) -> None:
    asset_path = cache_inputs.components.base_model_path / asset_name
    baseline = _fingerprint(cache_inputs)

    _replace_same_path_content(asset_path)
    changed = _fingerprint(cache_inputs)

    assert changed != baseline, (
        f"same-path frontend asset mutation was absent from cache identity: {asset_name}"
    )


def test_new_unlisted_frontend_asset_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    baseline = _fingerprint(cache_inputs)

    new_asset = cache_inputs.components.base_model_path / "future_frontend.asset"
    new_asset.write_bytes(b"new-loader-input")

    assert _fingerprint(cache_inputs) != baseline


def test_resolved_local_model_root_path_is_a_determinant(
    cache_inputs: _CacheInputs,
    tmp_path: Path,
) -> None:
    baseline = _fingerprint(cache_inputs)
    copied_root = tmp_path / "copied-model"
    shutil.copytree(cache_inputs.components.base_model_path, copied_root)
    copied_components = _Components(copied_root)

    assert _fingerprint(cache_inputs, components=copied_components) != baseline


def test_unknown_binary_asset_is_hashed_not_silently_excluded(
    cache_inputs: _CacheInputs,
) -> None:
    asset = cache_inputs.components.base_model_path / "frontend_lookup.bin"
    asset.write_bytes(b"frontend-a")
    baseline = _fingerprint(cache_inputs)

    asset.write_bytes(b"frontend-b")

    assert _fingerprint(cache_inputs) != baseline


def test_declared_model_weight_payload_content_is_not_hashed(
    cache_inputs: _CacheInputs,
) -> None:
    shard = cache_inputs.components.base_model_path / "model-00001-of-00001.safetensors"
    baseline = _fingerprint(cache_inputs)

    shard.write_bytes(b"excluded-model-weight-b")

    assert _fingerprint(cache_inputs) == baseline


def test_conventional_standalone_model_safetensors_content_is_not_hashed(
    cache_inputs: _CacheInputs,
) -> None:
    standalone = cache_inputs.components.base_model_path / "model.safetensors"
    standalone.write_bytes(b"standalone-model-weight-a")
    baseline = _fingerprint(cache_inputs)

    standalone.write_bytes(b"standalone-model-weight-b")

    assert _fingerprint(cache_inputs) == baseline


def test_nested_conventional_weight_basename_is_hashed_as_frontend_asset(
    cache_inputs: _CacheInputs,
) -> None:
    nested = cache_inputs.components.base_model_path / "frontend" / "model.safetensors"
    nested.parent.mkdir()
    nested.write_bytes(b"nested-frontend-a")
    baseline = _fingerprint(cache_inputs)

    nested.write_bytes(b"nested-frontend-b")

    assert _fingerprint(cache_inputs) != baseline


def test_index_declared_nested_weight_shard_content_is_not_hashed(
    cache_inputs: _CacheInputs,
) -> None:
    nested = cache_inputs.components.base_model_path / "weights"
    nested.mkdir()
    shard = nested / "shard-00001.safetensors"
    shard.write_bytes(b"nested-weight-a")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": shard.name}}),
        encoding="utf-8",
    )
    baseline = _fingerprint(cache_inputs)

    shard.write_bytes(b"nested-weight-b")

    assert _fingerprint(cache_inputs) == baseline


def test_unrecognized_weight_like_payload_is_hashed(
    cache_inputs: _CacheInputs,
) -> None:
    asset = cache_inputs.components.base_model_path / "future_head.safetensors"
    asset.write_bytes(b"unrecognized-weight-like-a")
    baseline = _fingerprint(cache_inputs)

    asset.write_bytes(b"unrecognized-weight-like-b")

    assert _fingerprint(cache_inputs) != baseline


def test_frontend_inventory_nested_walk_error_fails_closed_with_bounded_error(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = cache_inputs.components.base_model_path / "nested" / ("x" * 2048)

    def injected_walk_error(
        _root: Path,
        *,
        followlinks: bool,
        onerror: Any,
    ) -> Any:
        assert followlinks is False
        assert callable(onerror)
        onerror(PermissionError(errno.EACCES, "injected scandir denial", nested))
        raise AssertionError("walk continued after an unreadable nested subtree")

    monkeypatch.setattr(pack_cache.os, "walk", injected_walk_error)

    with pytest.raises(ValueError, match="could not read subtree") as exc_info:
        _fingerprint(cache_inputs)

    assert len(str(exc_info.value)) < 768
    assert "PermissionError" in str(exc_info.value)


def test_model_root_symlink_escape_fails_closed(
    cache_inputs: _CacheInputs,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.asset"
    outside.write_bytes(b"outside")
    (cache_inputs.components.base_model_path / "escaped.asset").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes"):
        _fingerprint(cache_inputs)


def test_model_root_nonregular_entry_fails_closed(
    cache_inputs: _CacheInputs,
) -> None:
    fifo = cache_inputs.components.base_model_path / "frontend.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="only regular files"):
        _fingerprint(cache_inputs)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    (
        ("_MAX_FRONTEND_ASSET_FILES", 1, "asset count exceeds"),
        ("_MAX_FRONTEND_HASHED_BYTES", 1, "frontend bytes exceed"),
    ),
)
def test_frontend_inventory_declared_envelope_fails_closed(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    monkeypatch.setattr(pack_cache, limit_name, limit_value)

    with pytest.raises(ValueError, match=message):
        _fingerprint(cache_inputs)


def test_model_root_total_regular_file_limit_counts_excluded_weights(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pack_cache, "_MAX_MODEL_ROOT_REGULAR_FILES", 1)
    monkeypatch.setattr(pack_cache, "_MAX_FRONTEND_ASSET_FILES", 1000)

    with pytest.raises(ValueError, match="regular file count exceeds") as exc_info:
        _fingerprint(cache_inputs)

    assert len(str(exc_info.value)) < 256


def test_model_weight_index_declaration_limit_fails_closed_before_hashing(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = cache_inputs.components.base_model_path / "weights"
    nested.mkdir()
    shard = nested / "shard.safetensors"
    shard.write_bytes(b"weight")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": shard.name,
                    "layer.1": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pack_cache, "_MAX_MODEL_WEIGHT_INDEX_DECLARATIONS", 2)

    with pytest.raises(ValueError, match="declaration count exceeds") as exc_info:
        _fingerprint(cache_inputs)

    assert len(str(exc_info.value)) < 256


def test_same_path_dataset_image_edit_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    image_path = Path(cache_inputs.dataset.path).parent / "image.bin"
    baseline = _fingerprint(cache_inputs)

    _replace_same_path_bytes(image_path, b"image-revision-b")
    changed = _fingerprint(cache_inputs)

    assert changed != baseline, (
        "dataset-referenced image bytes must be cache determinants"
    )


def test_dataset_image_identity_is_bounded_streaming_aggregate(
    cache_inputs: _CacheInputs,
) -> None:
    image_identity = _determinants(cache_inputs)["dataset"]["image_content_identity"]

    assert image_identity == {
        "algorithm": "ordered-canonical-image-content-sha256-v1",
        "image_count": 1,
        "total_size_bytes": len(b"image-revision-a"),
        "sha256": image_identity["sha256"],
    }
    assert isinstance(image_identity["sha256"], str)
    assert len(image_identity["sha256"]) == 64
    assert not any(isinstance(value, list) for value in image_identity.values())


def test_pack_plan_policy_identity_is_bound_to_determinants_and_manifest(
    cache_inputs: _CacheInputs,
    tmp_path: Path,
) -> None:
    determinants = _determinants(cache_inputs)
    expected_packing_identity = {
        "schema": "coordexp-swift-pack-plan",
        "schema_version": 5,
        "global_max_length": 12_000,
        "policy_identity": {
            "policy": "source_order_next_fit",
            "algorithm_version": "coordexp-swift-source-order-next-fit-v2",
            "window_size": None,
            "lookahead": None,
            "tie_breaker": "source_ordinal_v1",
            "seed": 0,
            "requested_worker_count": 1,
            "worker_count_disposition": (
                "semantic_pending_upstream_materialization_equality"
            ),
            "worker_count_integration_requirement": (
                "canonical_concurrent_encoded_materialization_and_complete_plan_equality"
            ),
            "cursor_byte_budget": 65_536,
            "fragment_item_budget": 1_024,
            "fragment_byte_budget": 4_194_304,
        },
        "fragment_pack_budget": None,
    }

    assert determinants["packing"] == expected_packing_identity
    packing_entry = next(
        entry
        for entry in determinants["determinants"]
        if entry["name"] == "packing_config"
    )
    assert packing_entry["content_identity"] == expected_packing_identity

    fingerprint = _fingerprint(cache_inputs)
    cache_dir = pack_cache.cache_dir_for_fingerprint(tmp_path, fingerprint)
    manifest = pack_cache.write_micro_step_cache(
        cache_dir,
        (_micro_step(),),
        cache_root=tmp_path,
        fingerprint=fingerprint,
        determinants=determinants,
        chunk_size=1,
        materialization=pack_cache.build_packing_cache_materialization(workers=1),
        determinant_revalidator=lambda: determinants,
        augmentation=DISABLED_AUGMENTATION,
    )

    assert manifest["determinants"]["packing"] == expected_packing_identity


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    (
        ("policy", "source_order_next_fit-mutated"),
        ("algorithm_version", "algorithm-mutated"),
        ("window_size", 17),
        ("lookahead", 19),
        ("tie_breaker", "tie-breaker-mutated"),
        ("seed", 23),
        ("requested_worker_count", 8),
        ("worker_count_disposition", "disposition-mutated"),
        ("worker_count_integration_requirement", "requirement-mutated"),
        ("cursor_byte_budget", 65_537),
        ("fragment_item_budget", 1_025),
        ("fragment_byte_budget", 4_194_305),
    ),
)
def test_every_pack_plan_policy_identity_field_changes_cache_fingerprint(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutated_value: Any,
) -> None:
    baseline = _fingerprint(cache_inputs)
    original_builder = pack_cache.build_pack_plan_policy_identity

    def mutated_policy_identity(**kwargs: Any) -> dict[str, Any]:
        identity = original_builder(**kwargs)
        identity[field] = mutated_value
        return identity

    monkeypatch.setattr(
        pack_cache,
        "build_pack_plan_policy_identity",
        mutated_policy_identity,
    )

    assert _fingerprint(cache_inputs) != baseline, (
        f"PackPlan policy field was absent from cache identity: {field}"
    )


@pytest.mark.parametrize(
    ("config_field", "mutated_value"),
    (
        ("global_max_length", 12_001),
        ("worker_count", 8),
        ("cursor_byte_budget", 65_537),
        ("fragment_item_budget", 1_025),
        ("fragment_byte_budget", 4_194_305),
    ),
)
def test_pack_plan_config_determinant_mutations_change_cache_fingerprint(
    cache_inputs: _CacheInputs,
    config_field: str,
    mutated_value: int,
) -> None:
    changed_config = cache_inputs.config.model_copy(
        update={
            "packing": cache_inputs.config.packing.model_copy(
                update={config_field: mutated_value}
            )
        }
    )

    assert _fingerprint(cache_inputs, config=changed_config) != _fingerprint(
        cache_inputs
    )


def test_online_fragment_pack_budget_changes_cache_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    baseline_packing = cache_inputs.config.packing.model_copy(
        update={
            "policy": "online_window_binpack",
            "lookahead": 8,
            "max_packs_per_fragment": 16,
        }
    )
    changed_packing = baseline_packing.model_copy(update={"max_packs_per_fragment": 17})
    baseline_config = cache_inputs.config.model_copy(
        update={"packing": baseline_packing}
    )
    changed_config = cache_inputs.config.model_copy(update={"packing": changed_packing})

    assert _fingerprint(cache_inputs, config=changed_config) != _fingerprint(
        cache_inputs,
        config=baseline_config,
    )


def test_same_path_model_config_asset_edit_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    config_path = cache_inputs.components.base_model_path / "config.json"
    baseline = _fingerprint(cache_inputs)

    _replace_same_path_content(config_path)
    changed = _fingerprint(cache_inputs)

    assert changed != baseline, "model config asset bytes must be cache determinants"


def test_special_token_id_change_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    baseline = _fingerprint(cache_inputs)
    changed_components = _Components(
        cache_inputs.components.base_model_path,
        token_identity={
            **_token_identity_payload(),
            "wrapper_token_ids": {
                "<|object_ref_start|>": 13,
                "<|object_ref_end|>": 5,
                "<|box_start|>": 6,
                "<|box_end|>": 7,
            },
        },
    )
    changed_groups = _groups(schema=(13, 5, 6, 7), blocked=(10, 11, 12))

    changed = _fingerprint(
        cache_inputs,
        components=changed_components,
        vocab_groups=changed_groups,
    )

    assert changed != baseline, "special-token IDs must be cache determinants"


def test_control_token_id_change_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    baseline = _fingerprint(cache_inputs)
    changed_components = _Components(
        cache_inputs.components.base_model_path,
        tokenizer=_Tokenizer(
            all_special_ids=(4, 5, 6, 7, 8, 9, 10, 11, 13),
            control_token_ids={"<think>": 13},
        ),
    )
    changed_groups = _groups(blocked=(10, 11, 13))

    changed = _fingerprint(
        cache_inputs,
        components=changed_components,
        vocab_groups=changed_groups,
    )

    assert changed != baseline, "realized control-token IDs must be cache determinants"


def test_complete_realized_vocabulary_group_membership_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    baseline_groups = TokenVocabularyGroups(
        vocab_size=16,
        desc_text=(0, 1, 2, 3, 13, 14),
        schema=(4, 5, 6, 7),
        coordinate=(8,),
        eos=(9,),
        blocked=(10, 11, 12),
    )
    # Counts, vocabulary size, and the named non-text groups remain identical.
    # Only one desc_text member changes, catching summary-only identities.
    changed_groups = TokenVocabularyGroups(
        vocab_size=16,
        desc_text=(0, 1, 2, 3, 13, 15),
        schema=(4, 5, 6, 7),
        coordinate=(8,),
        eos=(9,),
        blocked=(10, 11, 12),
    )

    baseline = _fingerprint(cache_inputs, vocab_groups=baseline_groups)
    changed = _fingerprint(cache_inputs, vocab_groups=changed_groups)

    assert changed != baseline, (
        "cache identity must bind every realized vocabulary-group member, not only counts"
    )


def test_precision_serialized_by_micro_step_constructor_changes_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    determinants = _determinants(cache_inputs)
    entry = next(
        item
        for item in determinants["determinants"]
        if item["name"] == "micro_step_runtime_config"
    )
    assert determinants["micro_step_runtime_config"] == {
        "fa2_model_dtype": "bf16",
        "capture_fa2_branch": False,
        "require_fa2_branch_proof": False,
    }
    assert entry["content_identity"] == determinants["micro_step_runtime_config"]

    changed_config = cache_inputs.config.model_copy(
        update={
            "training": cache_inputs.config.training.model_copy(
                update={"precision": "fp16"}
            )
        }
    )

    assert _fingerprint(cache_inputs, config=changed_config) != _fingerprint(
        cache_inputs
    )


def test_fa2_proof_fields_serialized_by_micro_step_constructor_change_fingerprint(
    cache_inputs: _CacheInputs,
) -> None:
    changed_config = cache_inputs.config.model_copy(
        update={
            "model": cache_inputs.config.model.model_copy(
                update={"fa2_branch_proof": "every_forward"}
            )
        }
    )
    changed = _determinants(cache_inputs, config=changed_config)

    assert changed["micro_step_runtime_config"] == {
        "fa2_model_dtype": "bf16",
        "capture_fa2_branch": True,
        "require_fa2_branch_proof": True,
    }
    assert _fingerprint(cache_inputs, config=changed_config) != _fingerprint(
        cache_inputs
    )


def test_pipeline_micro_step_constructor_owner_is_explicit_and_source_bound(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (Path.cwd() / "src/training/pipeline.py").resolve()
    original_sha256 = pack_cache._file_sha256
    source_bytes = b"pipeline-owner-revision-a"

    def controlled_sha256(path: Path) -> str:
        if Path(path).resolve() == target:
            return hashlib.sha256(source_bytes).hexdigest()
        return original_sha256(Path(path))

    monkeypatch.setattr(pack_cache, "_file_sha256", controlled_sha256)
    baseline_determinants = _determinants(cache_inputs)
    entry = next(
        item
        for item in baseline_determinants["determinants"]
        if item["name"] == "micro_step_runtime_config"
    )
    assert entry["owner"] == "src/training/pipeline.py"
    assert entry["owner_source_identity"]["path"] == "src/training/pipeline.py"
    baseline = _fingerprint(cache_inputs)

    source_bytes = b"pipeline-owner-revision-b"

    assert _fingerprint(cache_inputs) != baseline


def test_supervised_micro_step_schema_owner_is_explicit_and_source_bound(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (Path.cwd() / "src/training/supervised_trainer.py").resolve()
    original_sha256 = pack_cache._file_sha256
    source_bytes = b"supervised-schema-revision-a"

    def controlled_sha256(path: Path) -> str:
        if Path(path).resolve() == target:
            return hashlib.sha256(source_bytes).hexdigest()
        return original_sha256(Path(path))

    monkeypatch.setattr(pack_cache, "_file_sha256", controlled_sha256)
    baseline_determinants = _determinants(cache_inputs)
    entry = next(
        item
        for item in baseline_determinants["determinants"]
        if item["name"] == "micro_step_schema"
    )
    assert entry["owner"] == "src/training/supervised_trainer.py"
    assert entry["owner_source_identity"]["path"] == (
        "src/training/supervised_trainer.py"
    )
    assert [field["name"] for field in entry["content_identity"]["fields"]] == [
        "pack",
        "encoded_examples",
        "position_inputs",
        "token_sequence",
        "vocab_groups",
        "metadata",
        "forward_device",
        "expected_vocab_size",
        "extra_model_kwargs",
        "fa2_branch_evidence",
        "fa2_model_dtype",
        "capture_fa2_branch",
        "require_fa2_branch_proof",
        "fa2_branch_proof_policy",
    ]
    baseline = _fingerprint(cache_inputs)

    source_bytes = b"supervised-schema-revision-b"

    assert _fingerprint(cache_inputs) != baseline


def test_realized_vocabulary_identity_is_bounded_for_production_size() -> None:
    groups = TokenVocabularyGroups(
        vocab_size=150_000,
        desc_text=tuple(range(0, 149_000)),
        schema=tuple(range(149_000, 149_100)),
        coordinate=tuple(range(149_100, 149_900)),
        eos=(149_900,),
        blocked=tuple(range(149_901, 150_000)),
    )

    identity = pack_cache._realized_vocab_group_identity(groups)
    encoded = json.dumps(identity, sort_keys=True)

    assert len(encoded) < 2_000
    assert identity["vocab_size"] == 150_000
    assert identity["groups"]["desc_text"]["count"] == 149_000
    assert all(
        set(group_identity) == {"count", "sha256", "min_id", "max_id"}
        for group_identity in identity["groups"].values()
    )


def test_determinant_registry_names_complete_declared_owner_inventory(
    cache_inputs: _CacheInputs,
) -> None:
    determinants = _determinants(cache_inputs)
    entries = determinants.get("determinants")

    assert determinants["registry_schema_version"] == 1
    assert determinants["aggregate_fingerprint"] == _fingerprint(cache_inputs)
    assert isinstance(entries, list), (
        "cache determinants must expose a versioned determinant list for manifest audit"
    )
    declared = {
        entry.get("name"): entry.get("owner")
        for entry in entries
        if isinstance(entry, dict)
    }
    assert pack_cache.PACKING_CACHE_DETERMINANT_OWNERS == EXPECTED_DETERMINANT_OWNERS
    assert declared == EXPECTED_DETERMINANT_OWNERS
    for entry in entries:
        assert isinstance(entry, dict)
        assert isinstance(entry.get("name"), str) and entry["name"]
        assert isinstance(entry.get("owner"), str) and entry["owner"]
        assert entry.get("content_identity") not in (None, "", {}, [])
        assert entry.get("owner_source_identity") == {
            "path": entry["owner"],
            "sha256": entry["owner_source_identity"]["sha256"],
        }
        assert len(entry["owner_source_identity"]["sha256"]) == 64
        assert isinstance(entry.get("reason"), str) and entry["reason"]
        assert isinstance(entry.get("schema_version"), int)
        assert not isinstance(entry["schema_version"], bool)
        assert entry["schema_version"] > 0


@pytest.mark.parametrize(
    "owner_path",
    UNIQUE_OWNER_PATHS,
    ids=lambda path: Path(path).stem,
)
def test_every_unique_declared_owner_source_edit_changes_fingerprint(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    owner_path: str,
) -> None:
    target = (Path.cwd() / owner_path).resolve()
    original_sha256 = pack_cache._file_sha256
    synthetic_content = {target: b"owner-revision-a"}

    def controlled_sha256(path: Path) -> str:
        resolved = Path(path).resolve()
        payload = synthetic_content.get(resolved)
        if payload is None:
            return original_sha256(Path(path))
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(pack_cache, "_file_sha256", controlled_sha256)
    baseline = _fingerprint(cache_inputs)

    synthetic_content[target] = b"owner-revision-b"
    changed = _fingerprint(cache_inputs)

    assert changed != baseline, (
        f"declared cached-payload owner source was absent from registry: {owner_path}"
    )


@pytest.mark.parametrize("mutation", ("missing", "mismatched-path", "mismatched-sha"))
def test_registry_rejects_missing_or_mismatched_owner_source_identity(
    cache_inputs: _CacheInputs,
    mutation: str,
) -> None:
    determinants = _determinants(cache_inputs)
    entry = determinants["determinants"][0]
    if mutation == "missing":
        entry.pop("owner_source_identity")
    elif mutation == "mismatched-path":
        entry["owner_source_identity"]["path"] = "src/unknown.py"
    else:
        entry["owner_source_identity"]["sha256"] = "f" * 64
    determinants["aggregate_fingerprint"] = pack_cache._registry_entries_fingerprint(
        determinants["determinants"]
    )
    if mutation != "missing":
        determinants["code_identity"] = pack_cache._registry_code_identity(
            determinants["determinants"]
        )

    with pytest.raises(ValueError, match="owner source identity mismatch"):
        pack_cache.packing_cache_fingerprint_from_determinants(determinants)


def test_manifest_rejects_undeclared_determinant_owner(
    cache_inputs: _CacheInputs,
    tmp_path: Path,
) -> None:
    determinants = _determinants(cache_inputs)
    mutated = _with_undeclared_owner(determinants)
    fingerprint = _canonical_fingerprint(mutated)
    cache_dir = pack_cache.cache_dir_for_fingerprint(tmp_path, fingerprint)

    with pytest.raises(
        (ValueError, pack_cache.PackingCacheInvalidError),
        match=r"(?i)(owner|determinant)",
    ):
        pack_cache.write_micro_step_cache(
            cache_dir,
            (_micro_step(),),
            cache_root=tmp_path,
            fingerprint=fingerprint,
            determinants=mutated,
            chunk_size=1,
            materialization=pack_cache.build_packing_cache_materialization(workers=1),
            determinant_revalidator=lambda: mutated,
            augmentation=DISABLED_AUGMENTATION,
        )
        pack_cache.load_cache_manifest(
            cache_dir,
            cache_root=tmp_path,
            expected_fingerprint=fingerprint,
            level="manifest",
        )


@pytest.mark.parametrize("retired_version", RETIRED_CACHE_VERSIONS)
def test_manifest_rejects_retired_cache_identity_version_with_current_version(
    cache_inputs: _CacheInputs,
    tmp_path: Path,
    retired_version: str,
) -> None:
    determinants = _determinants(cache_inputs)
    fingerprint = _fingerprint(cache_inputs)
    cache_dir = pack_cache.cache_dir_for_fingerprint(tmp_path, fingerprint)
    pack_cache.write_micro_step_cache(
        cache_dir,
        (_micro_step(),),
        cache_root=tmp_path,
        fingerprint=fingerprint,
        determinants=determinants,
        chunk_size=1,
        materialization=pack_cache.build_packing_cache_materialization(workers=1),
        determinant_revalidator=lambda: determinants,
        augmentation=DISABLED_AUGMENTATION,
    )
    manifest_path = cache_dir / pack_cache.PACKING_CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = retired_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(pack_cache.PackingCacheInvalidError) as exc_info:
        pack_cache.load_cache_manifest(
            cache_dir,
            cache_root=tmp_path,
            expected_fingerprint=fingerprint,
            level="manifest",
        )

    message = str(exc_info.value)
    assert retired_version in message
    assert pack_cache.PACKING_CACHE_VERSION in message


def _fingerprint(
    inputs: _CacheInputs,
    *,
    config: Any | None = None,
    components: _Components | None = None,
    vocab_groups: TokenVocabularyGroups | None = None,
) -> str:
    resolved_config = config or inputs.config
    resolved_components = components or inputs.components
    resolved_groups = vocab_groups or inputs.vocab_groups
    resolved_components.vocab_groups = resolved_groups
    kwargs: dict[str, Any] = {
        "dataset": inputs.dataset,
        "split": "train",
    }
    signature = inspect.signature(pack_cache.build_packing_cache_fingerprint)
    if "vocab_groups" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs["vocab_groups"] = resolved_groups
    return pack_cache.build_packing_cache_fingerprint(
        resolved_config,
        resolved_components,
        **kwargs,
    )


def _determinants(
    inputs: _CacheInputs,
    *,
    config: Any | None = None,
    components: _Components | None = None,
    vocab_groups: TokenVocabularyGroups | None = None,
) -> dict[str, Any]:
    resolved_config = config or inputs.config
    resolved_components = components or inputs.components
    resolved_groups = vocab_groups or inputs.vocab_groups
    resolved_components.vocab_groups = resolved_groups
    kwargs: dict[str, Any] = {
        "dataset": inputs.dataset,
        "split": "train",
    }
    signature = inspect.signature(pack_cache.build_packing_cache_determinants)
    if "vocab_groups" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs["vocab_groups"] = resolved_groups
    return pack_cache.build_packing_cache_determinants(
        resolved_config,
        resolved_components,
        **kwargs,
    )


def _replace_same_path_content(path: Path) -> None:
    before_stat = path.stat()
    before_size = before_stat.st_size
    before = path.read_bytes()
    after = before.replace(b'"a"', b'"b"', 1)
    assert after != before, f"fixture asset lacks same-size revision marker: {path}"
    path.write_bytes(after)
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    assert path.stat().st_size == before_size
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns


def _replace_same_path_bytes(path: Path, payload: bytes) -> None:
    before_stat = path.stat()
    path.write_bytes(payload)
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    assert path.stat().st_size == before_stat.st_size
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns


def _token_identity_payload() -> dict[str, Any]:
    return {
        "required_tokens": [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
        ],
        "wrapper_token_ids": {
            "<|object_ref_start|>": 4,
            "<|object_ref_end|>": 5,
            "<|box_start|>": 6,
            "<|box_end|>": 7,
        },
        "coordinate_token_ids": [8],
        "im_end_token_ids": [9],
        "newline_token_ids": [10],
        "im_end_newline_token_ids": [9, 10],
        "tokenizer_vocab_size": 16,
    }


def _groups(
    *,
    schema: tuple[int, ...] = (4, 5, 6, 7),
    blocked: tuple[int, ...],
) -> TokenVocabularyGroups:
    target_ids = set(schema) | {8, 9}
    blocked_ids = set(blocked)
    desc_text = tuple(
        token_id
        for token_id in range(16)
        if token_id not in target_ids and token_id not in blocked_ids
    )
    return TokenVocabularyGroups(
        vocab_size=16,
        desc_text=desc_text,
        schema=schema,
        coordinate=(8,),
        eos=(9,),
        blocked=blocked,
    )


def _with_undeclared_owner(determinants: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(determinants)
    unknown = {
        "name": "unit_test_unknown_owner",
        "owner": "src/undeclared/cache_payload_owner.py",
        "content_identity": {"sha256": "f" * 64},
        "reason": "negative control for fail-closed determinant admission",
        "schema_version": 1,
    }
    entries = mutated.get("determinants")
    if isinstance(entries, list):
        entries.append(unknown)
    else:
        # This fallback deliberately demonstrates that the current ad hoc v2
        # mapping accepts an undeclared owner-shaped determinant.
        mutated["unit_test_unknown_owner"] = unknown
    return mutated


def _canonical_fingerprint(determinants: dict[str, Any]) -> str:
    payload = json.dumps(
        determinants,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _micro_step() -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack="pack",
        encoded_examples=("example",),
        position_inputs="positions",
        token_sequence="tokens",
        vocab_groups="vocab",
        metadata={
            "pack_id": 0,
            "augmentation_receipt": dict(DISABLED_AUGMENTATION),
        },
    )


# ---------------------------------------------------------------------------
# Wave-0 pre-move characterization for
# `decompose-coordexp-swift-training-orchestration`.
#
# The determinant-owner registry is the surface Wave 3 narrows.  These additions
# freeze the exact current inventory, the two evidenced overbroad owners, and the
# four declared determinant sources whose identity is allowed to turn over.
# ---------------------------------------------------------------------------


WAVE0_DETERMINANT_OWNERS = {
    "augmentation_config": "src/augmentation/geometry.py",
    "augmentation_factory": "src/augmentation/factory.py",
    "augmentation_processor": "src/augmentation/processor.py",
    "cache_serializer": "src/training/pack_cache.py",
    "coordinate_targets": "src/coordinate_targets.py",
    "dataset_content": "src/data/examples.py",
    "dataset_geometry": "src/data/geometry.py",
    "dataset_image_resolver": "src/data/images.py",
    "dataset_jsonl_loader": "src/data/jsonl.py",
    "encoding_runtime": "src/qwen/encoding.py",
    "image_loader": "src/qwen/images.py",
    "micro_step_runtime_config": "src/training/pipeline.py",
    "micro_step_schema": "src/training/supervised_trainer.py",
    "model_config_assets": "src/qwen/runtime_loading.py",
    "mrope_position_ids": "src/qwen/positions.py",
    "ordering_config": "src/data/examples.py",
    "pack_planner": "src/packing/planner.py",
    "packing_config": "src/packing/planner.py",
    "parser": "src/data/examples.py",
    "processor_assets": "src/qwen/runtime_loading.py",
    "processor_config": "src/qwen/runtime_loading.py",
    "qwen_fa2_boundaries": "src/qwen/fa2.py",
    "qwen_forward_payload": "src/qwen/forward.py",
    "realized_vocab_groups": "src/losses/vocab.py",
    "renderer": "src/templates/renderer.py",
    "supervision_mapper": "src/packing/supervision.py",
    "supervision_tokens": "src/supervision/tokens.py",
    "template_config": "src/templates/renderer.py",
    "template_spans": "src/templates/spans.py",
    "token_identity": "src/qwen/tokens.py",
    "tokenizer_assets": "src/qwen/runtime_loading.py",
}

#: The four determinant sources the change declares may change identity.
WAVE0_DECLARED_DETERMINANT_SOURCE_CHANGES = (
    "cache_serializer",
    "micro_step_runtime_config",
    "micro_step_schema",
    "supervision_tokens",
)

#: The two evidenced overbroad owners Wave 3 rebinds to narrow owners.
WAVE0_OVERBROAD_DETERMINANT_OWNERS = {
    "micro_step_runtime_config": "src/training/pipeline.py",
    "micro_step_schema": "src/training/supervised_trainer.py",
}


def test_wave0_determinant_owner_registry_is_frozen() -> None:
    assert dict(pack_cache.PACKING_CACHE_DETERMINANT_OWNERS) == (
        WAVE0_DETERMINANT_OWNERS
    )
    assert pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION == 1


def test_wave0_overbroad_owners_are_exactly_the_two_declared_entries() -> None:
    observed = {
        name: owner
        for name, owner in pack_cache.PACKING_CACHE_DETERMINANT_OWNERS.items()
        if owner
        in {"src/training/pipeline.py", "src/training/supervised_trainer.py"}
    }

    assert observed == WAVE0_OVERBROAD_DETERMINANT_OWNERS


def test_wave0_declared_source_changes_are_current_registry_names() -> None:
    assert set(WAVE0_DECLARED_DETERMINANT_SOURCE_CHANGES) <= set(
        pack_cache.PACKING_CACHE_DETERMINANT_OWNERS
    )
    assert sorted(WAVE0_DECLARED_DETERMINANT_SOURCE_CHANGES) == list(
        WAVE0_DECLARED_DETERMINANT_SOURCE_CHANGES
    )


def test_wave0_determinant_owner_reasons_cover_every_owner_exactly() -> None:
    assert set(pack_cache._DETERMINANT_REASONS) == set(
        pack_cache.PACKING_CACHE_DETERMINANT_OWNERS
    )


def test_wave0_declared_owner_source_files_exist_at_the_baseline() -> None:
    missing = sorted(
        {
            owner
            for owner in pack_cache.PACKING_CACHE_DETERMINANT_OWNERS.values()
            if not Path(owner).is_file()
        }
    )

    assert missing == []


def test_wave0_supervised_micro_step_schema_identity_is_frozen() -> None:
    fixture = Path("tests/fixtures/training_orchestration/legacy_micro_step.json")
    frozen = json.loads(fixture.read_text(encoding="utf-8"))["schema_identity"]

    assert pack_cache._supervised_micro_step_schema_identity() == frozen
