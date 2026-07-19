from __future__ import annotations

import pytest

from scripts.research.run_current_seeded_sampled_rollouts import (
    SCHEMA_VERSION,
    _trim_generated_ids,
    physical_image_id,
    parse_seed_list,
    select_examples,
    validate_artifact_payload,
)


def test_parse_seed_list_is_ordered_and_unique() -> None:
    assert parse_seed_list("11, 12,13") == (11, 12, 13)
    with pytest.raises(ValueError, match="unique"):
        parse_seed_list("11,11")
    with pytest.raises(ValueError, match="at least one"):
        parse_seed_list("")


def test_trim_generated_ids_records_stop_and_ignores_padding() -> None:
    assert _trim_generated_ids([4, 5, 99, 0, 0], stop_token_id=99, pad_token_id=0) == ([4, 5], "im_end")
    assert _trim_generated_ids([4, 5], stop_token_id=99, pad_token_id=0) == ([4, 5], "length")
    with pytest.raises(ValueError, match="pad before stop"):
        _trim_generated_ids([4, 0, 5], stop_token_id=99, pad_token_id=0)


def test_image_selector_accepts_physical_coco_id_and_preserves_example_id() -> None:
    class Example:
        def __init__(self, example_id: str, image_id: int) -> None:
            self.example_id = example_id
            self.metadata = {"source": {"image_id": image_id}}

    examples = [Example("coco2017_val_000000007574", 7574), Example("other", 8)]
    selected = select_examples(examples, {"7574"})
    assert len(selected) == 1
    assert physical_image_id(selected[0]) == 7574
    assert selected[0].example_id == "coco2017_val_000000007574"
    assert select_examples(examples, {"coco2017_val_000000007574"}) == selected


def test_validate_artifact_payload_requires_compact_rollout_evidence() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {"seeds": [11]},
        "rollouts": [{
            "image_id": "6471",
            "seed": 11,
            "generated_token_ids": [1],
            "generated_text": "text",
            "stop_reason": "length",
            "predictions": {"predictions": []},
        }],
    }
    validate_artifact_payload(payload)
    payload["rollouts"][0].pop("seed")
    with pytest.raises(ValueError, match="seed"):
        validate_artifact_payload(payload)
