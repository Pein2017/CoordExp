import json

import pytest

from probes.training_set_completion import source256_ranking_data as data
from probes.training_set_completion import source256_ranking_evaluation as evaluation
from probes.training_set_completion import training


def _item(index: int) -> dict:
    return {
        "split": "train",
        "row_index": index,
        "image_id": 1000 + index,
        "example_id": f"row-{index}",
        "expected_prompt_token_ids": [index],
        "expected_media_sha256": "a" * 64,
        "expected_grid": [1, 1, 1],
    }


def _generation(items: list[dict]) -> dict:
    rows = []
    for index, item in enumerate(items):
        prompt, ids = list(item["expected_prompt_token_ids"]), [evaluation.readback.EOS]
        rows.append(
            {
                "split": item["split"],
                "row_index": item["row_index"],
                "image_id": item["image_id"],
                "example_id": item["example_id"],
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": training.digest(prompt),
                "generated_token_ids": ids,
                "generated_token_ids_sha256": training.digest(ids),
                "decode_stop_reason": "im_end",
                "raw_decode_text": "",
                "executed_media_sha256": item["expected_media_sha256"],
                "observed_image_grid_thw": item["expected_grid"],
                "batch_index": index // 4,
                "actual_batch_size": 4,
            }
        )
    return {
        "status": "completed",
        "configured_batch_size": 4,
        "request_count": len(rows),
        "generated_tokens": len(rows),
        "batches": [
            {"batch_index": 0, "actual_batch_size": 4, "generated_tokens": len(rows)}
        ],
        "rows": rows,
    }


def _score(covered: set[tuple[int, str]]) -> dict:
    return {
        "splits": {
            "train": {
                "per_image": [
                    {
                        "image_id": image_id,
                        "primary_class_agnostic_iou50": {"covered_owner_ids": [owner_id]},
                    }
                    for image_id, owner_id in sorted(covered)
                ]
            }
        }
    }


def test_frozen_selection_is_exact_diagnosis_pair_set():
    manifest = json.loads((data.ROOT / "preparation/P-main.json").read_text())
    prepared = data.load_data(manifest)

    selected, outside = evaluation._selection(prepared)

    assert selected == [
        101636,
        133279,
        143132,
        158044,
        201145,
        203986,
        234328,
        270570,
        360573,
        422969,
        446835,
        527822,
        536467,
        548337,
        575627,
    ]
    assert len(outside) == 241
    assert (
        prepared["pairs"]["548337"]["preferred_tokens"],
        prepared["pairs"]["548337"]["rejected_tokens"],
    ) == (177, 3084)
    assert sum(
        pair["rejected_tokens"]
        for image_id, pair in prepared["pairs"].items()
        if image_id != "548337"
    ) == 2162


def test_generation_validation_binds_saved_train_prompt_and_media():
    items = [_item(index) for index in range(4)]
    generation = _generation(items)

    evaluation._validate_generation(generation=generation, items=items)
    generation["rows"][2]["prompt_token_ids"] = [999]

    with pytest.raises(ValueError, match="normalized generation row identity"):
        evaluation._validate_generation(generation=generation, items=items)


def test_source_new_survival_keeps_replacements_distinct():
    source = _score(set())
    starting = {(index, str(index)) for index in range(91)}
    normalized = _score(starting)
    baseline = evaluation._starting_source_new(
        baseline=source, endpoint=normalized, label="Bnormalized64_vs_Source0"
    )
    candidate = _score(
        {(index, str(index)) for index in range(89)} | {(1000, "a"), (1001, "b")}
    )

    result = evaluation._survival(
        start=baseline["starting"], source=baseline["source_coverage"], endpoint=candidate
    )

    assert result["starting_count"] == 91
    assert result["survived_count"] == 89
    assert result["lost_count"] == 2
    assert result["replacement_gain_count"] == 2
    assert result["final_gain_count_vs_Source0"] == 91


def test_qualification_saved_shard_accepts_resolved_train_split(tmp_path, monkeypatch):
    """The worker resolves None to train before checking its generated artifact."""
    items = [_item(index) for index in range(4)]
    monkeypatch.setattr(evaluation.readback, '_train_items', lambda prepared, ids: items)
    manifest_path=tmp_path/'manifest.json'; manifest_path.write_text('{}')
    terminal_path=tmp_path/'terminal.json'; terminal_path.write_text('{}')
    plan_path=tmp_path/'plan.json'; plan_path.write_text('{}')
    monkeypatch.setattr(evaluation, 'PLAN', plan_path)
    manifest={'arm':'R','runtime':{'updates':1}}
    value=evaluation._shard_value(manifest_path=manifest_path,manifest=manifest,terminal_path=terminal_path,adapter={},split='train',shard=None,qualification=True,generation=_generation(items))
    kwargs=dict(manifest_path=manifest_path,manifest=manifest,terminal_path=terminal_path,adapter={},plan={'prepared':{}},shard=None,qualification=True)
    assert evaluation._validate_shard(value,split='train',**kwargs)==value
    assert evaluation._validate_shard(value,split=None,**kwargs)==value
    with pytest.raises(ValueError,match='qualification'):
        evaluation._validate_shard(value,split='dev',**kwargs)
