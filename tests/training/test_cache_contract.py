"""Narrow cached micro-step determinant owners.

Wave 3 of ``decompose-coordexp-swift-training-orchestration`` gives the cached
micro-step runtime projection its own leaf owner and rebinds the two evidenced
overbroad determinant owners:

```text
micro_step_runtime_config -> src/training/cache_contract.py
micro_step_schema         -> src/training/micro_steps.py
```

The projection payload itself is protected: exactly three fields, exactly the
values the production micro-step constructor serializes.  Only the owner path
and its source hash move.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.config.loader import load_train_config
from src.losses.vocab import TokenVocabularyGroups
from src.training import cache_contract, pack_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")

#: The two determinant owners Wave 3 rebinds and nothing else.
REBOUND_DETERMINANT_OWNERS = {
    "micro_step_runtime_config": "src/training/cache_contract.py",
    "micro_step_schema": "src/training/micro_steps.py",
}

#: Owners this wave must never bind a determinant to again.
RETIRED_DETERMINANT_OWNER_PATHS = (
    "src/training/pipeline.py",
    "src/training/supervised_trainer.py",
)

#: Owners introduced by later waves; editing them must add no determinant.
NON_DETERMINANT_OWNER_PATHS = (
    "src/training/pipeline.py",
    "src/training/cache_workflow.py",
    "src/training/reporting.py",
    "src/training/session.py",
)

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


class _Identity:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


class _Tokenizer:
    chat_template = "unit-test-chat-template"

    def __init__(self) -> None:
        self.image_pad_token_id = 11
        self.all_special_ids = (4, 5, 6, 7, 8, 9, 10, 11, 12)
        self.control_token_ids = {"<think>": 12}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|image_pad|>":
            return self.image_pad_token_id
        return self.control_token_ids.get(token)


class _Processor:
    chat_template = "unit-test-chat-template"


class _Components:
    def __init__(self, model_root: Path) -> None:
        self.base_model_path = model_root
        self.processor = _Processor()
        self.tokenizer = _Tokenizer()
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
            {
                "schema_version": 1,
                "tokenizer_vocab_size": 20,
                "special_tokens": {
                    "obj_open": 4,
                    "obj_close": 5,
                    "desc_open": 6,
                    "desc_close": 7,
                    "box_open": 8,
                    "box_close": 9,
                    "sep": 10,
                },
                "image_pad_token_id": 11,
                "control_token_ids": {"<think>": 12},
                "all_special_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        )
        self.package_versions = {
            "tokenizers": "unit-test",
            "transformers": "unit-test",
        }


class _CacheInputs:
    def __init__(self, config: Any, components: _Components, vocab_groups: Any) -> None:
        self.config = config
        self.components = components
        self.dataset = config.data.train
        self.vocab_groups = vocab_groups


@pytest.fixture
def cache_inputs(tmp_path: Path) -> _CacheInputs:
    dataset_path = tmp_path / "train.coord.jsonl"
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image-revision-a")
    dataset_path.write_text(
        json.dumps(
            {
                "example_id": "cache-contract-test",
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
        asset_path.write_text('{"revision":"a"}\n', encoding="utf-8")
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
    vocab_groups = TokenVocabularyGroups(
        vocab_size=20,
        desc_text=(0, 1, 2, 3),
        schema=(4, 5, 6, 7, 8, 9),
        coordinate=(13, 14, 15),
        eos=(16,),
        blocked=(10, 11, 12),
    )
    return _CacheInputs(config, _Components(model_root), vocab_groups)


def _determinants(inputs: _CacheInputs, *, config: Any | None = None) -> dict[str, Any]:
    return pack_cache.build_packing_cache_determinants(
        inputs.config if config is None else config,
        inputs.components,
        dataset=(inputs.config if config is None else config).data.train,
        split="train",
        vocab_groups=inputs.vocab_groups,
    )


def _fingerprint(inputs: _CacheInputs, *, config: Any | None = None) -> str:
    return pack_cache.packing_cache_fingerprint_from_determinants(
        _determinants(inputs, config=config)
    )


# ---------------------------------------------------------------------------
# The three-field runtime projection
# ---------------------------------------------------------------------------


def test_runtime_config_identity_is_exactly_the_three_serialized_fields(
    cache_inputs: _CacheInputs,
) -> None:
    identity = cache_contract.micro_step_runtime_config_identity(cache_inputs.config)

    assert identity == {
        "fa2_model_dtype": "bf16",
        "capture_fa2_branch": False,
        "require_fa2_branch_proof": False,
    }
    assert list(identity) == [
        "fa2_model_dtype",
        "capture_fa2_branch",
        "require_fa2_branch_proof",
    ]


def test_runtime_config_identity_tracks_the_fa2_branch_proof_policy(
    cache_inputs: _CacheInputs,
) -> None:
    every_forward = cache_inputs.config.model_copy(
        update={
            "model": cache_inputs.config.model.model_copy(
                update={"fa2_branch_proof": "every_forward"}
            )
        }
    )
    first_micro_step = cache_inputs.config.model_copy(
        update={
            "model": cache_inputs.config.model.model_copy(
                update={"fa2_branch_proof": "first_micro_step"}
            )
        }
    )

    assert cache_contract.micro_step_runtime_config_identity(every_forward) == {
        "fa2_model_dtype": "bf16",
        "capture_fa2_branch": True,
        "require_fa2_branch_proof": True,
    }
    assert cache_contract.micro_step_runtime_config_identity(first_micro_step) == {
        "fa2_model_dtype": "bf16",
        "capture_fa2_branch": False,
        "require_fa2_branch_proof": False,
    }


def test_runtime_config_identity_tracks_the_training_precision(
    cache_inputs: _CacheInputs,
) -> None:
    changed = cache_inputs.config.model_copy(
        update={
            "training": cache_inputs.config.training.model_copy(
                update={"precision": "fp16"}
            )
        }
    )

    assert cache_contract.micro_step_runtime_config_identity(changed) == {
        "fa2_model_dtype": "fp16",
        "capture_fa2_branch": False,
        "require_fa2_branch_proof": False,
    }


def test_registry_projection_is_the_cache_contract_projection(
    cache_inputs: _CacheInputs,
) -> None:
    determinants = _determinants(cache_inputs)
    entry = next(
        item
        for item in determinants["determinants"]
        if item["name"] == "micro_step_runtime_config"
    )

    expected = cache_contract.micro_step_runtime_config_identity(cache_inputs.config)
    assert determinants["micro_step_runtime_config"] == expected
    assert entry["content_identity"] == expected


# ---------------------------------------------------------------------------
# Determinant owner mutation
# ---------------------------------------------------------------------------


def test_rebound_determinant_owners_are_the_two_narrow_owners() -> None:
    observed = {
        name: pack_cache.PACKING_CACHE_DETERMINANT_OWNERS[name]
        for name in REBOUND_DETERMINANT_OWNERS
    }

    assert observed == REBOUND_DETERMINANT_OWNERS
    assert pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION == 1
    for owner in REBOUND_DETERMINANT_OWNERS.values():
        assert (REPO_ROOT / owner).is_file()


def test_no_determinant_remains_bound_to_a_retired_overbroad_owner() -> None:
    observed = sorted(
        name
        for name, owner in pack_cache.PACKING_CACHE_DETERMINANT_OWNERS.items()
        if owner in RETIRED_DETERMINANT_OWNER_PATHS
    )

    assert observed == []


def _fingerprint_with_synthetic_owner_source(
    inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_path: str,
    payload: bytes,
) -> str:
    target = (REPO_ROOT / owner_path).resolve()
    original = pack_cache._file_sha256

    def controlled(path: Path) -> str:
        if Path(path).resolve() == target:
            return hashlib.sha256(payload).hexdigest()
        return original(Path(path))

    monkeypatch.setattr(pack_cache, "_file_sha256", controlled)
    return _fingerprint(inputs)


@pytest.mark.parametrize(
    "owner_path",
    sorted(set(REBOUND_DETERMINANT_OWNERS.values())),
    ids=lambda path: Path(path).stem,
)
def test_editing_a_narrow_owner_source_changes_the_aggregate(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    owner_path: str,
) -> None:
    baseline = _fingerprint_with_synthetic_owner_source(
        cache_inputs, monkeypatch, owner_path=owner_path, payload=b"owner-revision-a"
    )
    changed = _fingerprint_with_synthetic_owner_source(
        cache_inputs, monkeypatch, owner_path=owner_path, payload=b"owner-revision-b"
    )

    assert changed != baseline, (
        f"{owner_path} must bind the complete source of its determinant"
    )


@pytest.mark.parametrize("owner_path", NON_DETERMINANT_OWNER_PATHS)
def test_editing_a_non_determinant_owner_leaves_the_aggregate_equal(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
    owner_path: str,
) -> None:
    baseline = _fingerprint_with_synthetic_owner_source(
        cache_inputs, monkeypatch, owner_path=owner_path, payload=b"owner-revision-a"
    )
    changed = _fingerprint_with_synthetic_owner_source(
        cache_inputs, monkeypatch, owner_path=owner_path, payload=b"owner-revision-b"
    )

    assert changed == baseline, (
        f"{owner_path} owns no determinant and must add no cache identity churn"
    )


def test_owner_rebinding_changes_no_determinant_content_identity(
    cache_inputs: _CacheInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner source churn must never move a semantic content projection."""

    baseline = {
        entry["name"]: entry["content_identity"]
        for entry in _determinants(cache_inputs)["determinants"]
    }
    targets = {
        (REPO_ROOT / owner).resolve()
        for owner in REBOUND_DETERMINANT_OWNERS.values()
    }
    original = pack_cache._file_sha256

    def controlled(path: Path) -> str:
        if Path(path).resolve() in targets:
            return hashlib.sha256(b"rebound-owner-revision").hexdigest()
        return original(Path(path))

    monkeypatch.setattr(pack_cache, "_file_sha256", controlled)
    changed = _determinants(cache_inputs)
    observed = {
        entry["name"]: entry["content_identity"] for entry in changed["determinants"]
    }

    assert observed == baseline
    changed_owner_sources = sorted(
        entry["name"]
        for entry in changed["determinants"]
        if entry["owner"] in set(REBOUND_DETERMINANT_OWNERS.values())
    )
    assert changed_owner_sources == sorted(REBOUND_DETERMINANT_OWNERS)


def test_registry_entry_binds_the_narrow_owner_source_identity(
    cache_inputs: _CacheInputs,
) -> None:
    determinants = _determinants(cache_inputs)
    entries = {item["name"]: item for item in determinants["determinants"]}

    for name, owner in REBOUND_DETERMINANT_OWNERS.items():
        assert entries[name]["owner"] == owner
        assert entries[name]["owner_source_identity"]["path"] == owner
        assert len(entries[name]["owner_source_identity"]["sha256"]) == 64
    assert determinants["code_identity"]["micro_step_runtime_config"]["path"] == (
        "src/training/cache_contract.py"
    )
    assert determinants["code_identity"]["micro_step_schema"]["path"] == (
        "src/training/micro_steps.py"
    )


# ---------------------------------------------------------------------------
# Owner boundary
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_cache_contract_imports_no_orchestration_owner() -> None:
    imported = _imported_modules(REPO_ROOT / "src/training/cache_contract.py")

    assert "src.training.pipeline" not in imported
    assert "src.training.cache_workflow" not in imported
    assert "src.training.session" not in imported
    assert "src.training.pack_cache" not in imported
    assert "src.training.micro_step_assembler" not in imported


def test_micro_step_assembler_is_a_registered_determinant_owner() -> None:
    owner = pack_cache.PACKING_CACHE_DETERMINANT_OWNERS["micro_step_assembler"]

    assert owner == "src/training/micro_step_assembler.py"
    assert (REPO_ROOT / owner).is_file()
    assert pack_cache._DETERMINANT_REASONS["micro_step_assembler"]


def test_micro_step_assembler_imports_no_orchestration_owner() -> None:
    imported = _imported_modules(REPO_ROOT / "src/training/micro_step_assembler.py")

    assert "src.training.pipeline" not in imported
    assert "src.training.cache_workflow" not in imported
    assert "src.training.session" not in imported
    assert "src.training.pack_cache" not in imported
