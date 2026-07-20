from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from src.config.fingerprint import sha256_file, sha256_json
from src.config.models import ProcessorConfig
from src.inference.backend import token_ids_sha256
from src.qwen.runtime_loading import QwenProcessorIdentity
from src.rollout_calibration import CheckpointIdentity


IMAGE_TOKEN_ID = 151655
SYNTHETIC_SOURCE_CHECKPOINT = {
    "adapter_fingerprint": "1" * 64,
    "embedding_delta_fingerprint": "2" * 64,
    "base_config_sha256": "3" * 64,
    "tokenizer_sha256": "4" * 64,
    "token_identity_sha256": "5" * 64,
    "special_token_identity_sha256": "6" * 64,
    "processor_identity_sha256": "7" * 64,
}
SYNTHETIC_SOURCE_CHECKPOINT_ID = sha256_json(SYNTHETIC_SOURCE_CHECKPOINT)


@pytest.fixture
def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(**SYNTHETIC_SOURCE_CHECKPOINT)


@pytest.fixture
def prompt_identity_sha256() -> str:
    return "8" * 64


@pytest.fixture
def processor_config() -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=False,
        max_raw_pixels=1_000_000,
        max_merged_visual_tokens=4_096,
    )


@pytest.fixture
def fake_components() -> Any:
    return SimpleNamespace(
        tokenizer=TokenizerMustNotRun(),
        processor_identity=QwenProcessorIdentity(
            processor_class="SyntheticProcessor",
            tokenizer_class="TokenizerMustNotRun",
            image_processor_class="SyntheticImageProcessor",
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
        ),
        processor=SimpleNamespace(image_processor=SyntheticImageProcessor()),
    )


class TokenizerMustNotRun:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"exact-token replay must not access tokenizer.{name}")

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("exact-token replay must not tokenize stored ids")


class SyntheticImageProcessor:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, *, images: list[Any], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["do_resize"] is False
        self.batch_sizes.append(len(images))
        return {
            "pixel_values": torch.zeros((16 * len(images), 1536), dtype=torch.float32),
            "image_grid_thw": torch.tensor(
                [[1, 4, 4] for _ in images], dtype=torch.long
            ),
        }


def synthetic_inputs(
    tmp_path: Path,
    *,
    event_id: str = "synthetic-event-1",
    image_id: int | str = 42,
    split: str = "train",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    image_path = tmp_path / "synthetic.png"
    if not image_path.exists():
        Image.new("RGB", (64, 64), color=(12, 34, 56)).save(image_path)
    prompt_ids = [
        10,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        11,
    ]
    prefix_ids = [12, 13]
    positive_ids = [20, 21, 22]
    harmful_ids = [30]
    normalized_image_id = int(image_id)
    rollout = {
        "event_id": event_id,
        "image": {
            "image_id": image_id,
            "path": str(image_path),
            "width": 64,
            "height": 64,
            "content_sha256": sha256_file(image_path),
        },
        "split": split,
        "split_group_id": f"image:{normalized_image_id}",
        "executed_prompt_token_ids": prompt_ids,
        "executed_prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "image_pad_interval": [1, 5],
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
        "candidates": [
            {
                "candidate_id": "positive-a",
                "token_ids": positive_ids,
                "token_ids_sha256": token_ids_sha256(positive_ids),
                "generation_provenance": {
                    "mode": "sampled",
                    "seed": 11,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "repetition_penalty": 1.0,
                    "checkpoint_id": SYNTHETIC_SOURCE_CHECKPOINT_ID,
                    "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                    "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
                },
                "evidence_text": "optional evidence that is never retokenized",
            },
            {
                "candidate_id": "harmful-a",
                "token_ids": harmful_ids,
                "token_ids_sha256": token_ids_sha256(harmful_ids),
                "generation_provenance": {
                    "mode": "greedy",
                    "seed": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "checkpoint_id": SYNTHETIC_SOURCE_CHECKPOINT_ID,
                    "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                    "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
                },
                "evidence_text": None,
            },
        ],
    }
    review = {
        "event_id": event_id,
        "admission_status": "accepted",
        "rejection_reason": None,
        "physical_entities": [
            {
                "entity_id": "entity-a",
                "category": "person",
                "entity_trusted": True,
                "geometry_trusted": True,
                "reference_bbox": [100, 100, 400, 500],
                "review_source": "synthetic_unit_test",
                "reviewer": "synthetic",
                "review_confidence": "fixture_only",
                "comment": "not scientific evidence",
            },
            {
                "entity_id": "entity-covered",
                "category": "person",
                "entity_trusted": True,
                "geometry_trusted": True,
                "reference_bbox": [10, 10, 90, 90],
                "review_source": "synthetic_unit_test",
                "reviewer": "synthetic",
                "review_confidence": "fixture_only",
                "comment": "prefix owner only",
            },
        ],
        "prefix_covered_owner_proofs": [
            {
                "prefix_object_row_index": 0,
                "owner_id": "entity-covered",
                "review_provenance": {
                    "source": "synthetic_unit_test",
                    "reviewer": "synthetic",
                    "confidence": "fixture_only",
                    "comment": "not scientific evidence",
                },
            }
        ],
        "entity_transition_eligible": True,
        "coordinate_boundary_eligible": True,
        "candidates": [
            {
                "candidate_id": "positive-a",
                "role": "positive",
                "harmful_kind": None,
                "physical_owner_id": "entity-a",
                "coverage_status": "uncovered",
                "entity_review_status": "trusted",
                "geometry_review_status": "trusted",
                "entity_eligible": True,
                "geometry_eligible": True,
                "owner_resolution_interval": [0, 1],
                "coordinate_decision": {
                    "owner_id": "entity-a",
                    "observations": [
                        {
                            "coordinate": "x1",
                            "tolerance_axis": "horizontal",
                            "candidate_token_offset": 1,
                            "actual_coordinate_value": 500,
                            "acceptable_coordinate_values": [490, 491],
                            "review_provenance": {
                                "source": "synthetic_unit_test",
                                "reviewer": "synthetic",
                                "confidence": "fixture_only",
                                "comment": "not scientific evidence",
                            },
                        }
                    ],
                },
                "selected_sites": [
                    {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                    {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                ],
            },
            {
                "candidate_id": "harmful-a",
                "role": "harmful",
                "harmful_kind": "duplicate",
                "physical_owner_id": "entity-covered",
                "coverage_status": "covered",
                "entity_review_status": "trusted",
                "geometry_review_status": "unknown",
                "entity_eligible": True,
                "geometry_eligible": False,
                "owner_resolution_interval": [0, 1],
                "coordinate_decision": None,
                "selected_sites": [
                    {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
                ],
            },
        ],
        "review_provenance": {
            "review_set": "synthetic-only",
            "explicit_labels": True,
        },
    }
    return [copy.deepcopy(rollout)], [copy.deepcopy(review)], image_path


def source_artifacts() -> list[dict[str, str]]:
    return [
        {
            "artifact_id": "synthetic-rollout",
            "sha256": sha256_json({"kind": "rollout"}),
        },
        {"artifact_id": "synthetic-review", "sha256": sha256_json({"kind": "review"})},
    ]
