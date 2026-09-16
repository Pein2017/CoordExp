from __future__ import annotations

from pathlib import Path

import pytest

from probes.training_set_completion import paired_evaluation
from probes.training_set_completion import source256_evaluation as evaluation
from probes.training_set_completion import source256_readback
from probes.training_set_completion import source256_training
from probes.training_set_completion import source256_trial
from probes.training_set_completion import training


PREPARATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion/preparation/"
    "source256-data-v3/preparation.json"
)
QUALIFICATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion/runtime/"
    "qualification-mechanics-v3/readback-batch4-qualification.json"
)


def _target(owner: str, description: str) -> dict:
    return {
        "image_id": 1,
        "owner_id": owner,
        "reference_coord_bins_1000": [0, 0, 100, 100],
        "description": description,
        "normalized_description": description,
    }


def _prediction(name: str, description: str) -> dict:
    return {
        "prediction_id": name,
        "generated_order": 0,
        "coord_bins_1000": [0, 0, 100, 100],
        "description": description,
    }


def test_primary_is_class_agnostic_while_registered_diagnostics_are_class_consistent() -> None:
    targets = [_target("owner-1", "person")]
    predictions = [_prediction("prediction-1", "car")]

    primary = paired_evaluation._ledger_image(targets, predictions, threshold=0.5)
    diagnostic = evaluation.class_consistent_matches(targets, predictions, 0.5)

    assert primary["covered_owner_ids"] == ["owner-1"]
    assert diagnostic["covered_owner_ids"] == []
    with pytest.raises(ValueError, match="registered"):
        evaluation.class_consistent_matches(targets, predictions, 0.7)


def test_repeat_counter_is_class_agnostic_strict_gt_095_and_counts_later_row_once() -> None:
    predictions = [
        {**_prediction("p0", "person"), "generated_order": 0, "coord_bins_1000": [0, 0, 100, 100]},
        {**_prediction("p1", "car"), "generated_order": 1, "coord_bins_1000": [0, 0, 95, 100]},
        {**_prediction("p2", "chair"), "generated_order": 2, "coord_bins_1000": [0, 0, 96, 100]},
    ]

    repeats = evaluation._strict_repeat_rows(predictions)

    assert [row["prediction_id"] for row in repeats] == ["p2"]
    assert repeats[0]["best_earlier_iou"] > 0.95


def test_split_reducer_replays_native_parser_and_keeps_burden_counters_separate() -> None:
    text = (
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_0|><|coord_0|><|coord_100|><|coord_100|><|box_end|><|im_end|>"
    )

    class Tokenizer:
        @staticmethod
        def decode(*_args, **_kwargs) -> str:
            return text

    rows = [
        {
            "image_id": 1,
            "generated_token_ids": [151645],
            "generated_token_ids_sha256": training.digest([151645]),
            "decode_stop_reason": "im_end",
            "raw_decode_text": text,
        }
    ]
    score = evaluation._score_split(
        split="train",
        rows=rows,
        targets={1: [_target("owner-1", "person")]},
        contexts={
            1: {"row_id": "row-1", "row_index": 0, "image_width": 1000, "image_height": 1000}
        },
        tokenizer=Tokenizer(),
    )

    assert score["primary_class_agnostic_iou50"]["matched_count"] == 1
    assert score["class_consistent"]["0.5"]["matched_count"] == 1
    assert score["burden"] == {
        "valid_prediction_count": 1,
        "malformed_row_count": 0,
        "invalid_geometry_count": 0,
        "annotation_unmatched_prediction_count": 0,
        "strict_repeat_row_count": 0,
        "cap_debt": 0,
        "eos_debt": 0,
    }


def _score(owner: str) -> dict:
    row = {
        "image_id": 1,
        "primary_class_agnostic_iou50": {"covered_owner_ids": [owner]},
        "class_consistent": {
            str(threshold): {"covered_owner_ids": [owner]}
            for threshold in evaluation.THRESHOLDS
        },
    }
    return {
        "schema": f"{evaluation.SCHEMA}.endpoint_score",
        "preparation": {"path": "bank", "sha256": "bank", "size_bytes": 1},
        "endpoint": {"label": owner},
        "splits": {
            "train": {"per_image": [row]},
            "dev": {"per_image": [row]},
        },
    }


def test_comparison_keeps_owner_identity_gains_and_losses_for_each_split() -> None:
    result = evaluation.compare_scores(_score("old"), _score("new"), label="new_vs_old")

    for split in ("train", "dev"):
        primary = result["splits"][split]["primary_class_agnostic_iou50"]
        assert primary["gained"] == [{"image_id": 1, "owner_id": "new"}]
        assert primary["lost"] == [{"image_id": 1, "owner_id": "old"}]
        assert primary["retained_count"] == 0


def test_v3_bank_and_dev_rows_materialize_as_complete_known_owner_ledgers() -> None:
    if not PREPARATION.is_file():
        pytest.skip("Source256 v3 preparation is unavailable")
    prepared = source256_training.validate_preparation(evaluation.read(PREPARATION))
    ledgers = evaluation._targets(prepared)

    assert len(ledgers["train"]) == 256
    assert len(ledgers["dev"]) == 128
    assert sum(map(len, ledgers["train"].values())) == 1955
    assert all(rows for rows in ledgers["train"].values())
    assert all(rows for rows in ledgers["dev"].values())


def test_main_packet_is_held_and_binds_fresh_source_training_plus_batch4_readback(tmp_path) -> None:
    if not (PREPARATION.is_file() and QUALIFICATION.is_file()):
        pytest.skip("Source256 mechanics artifacts are unavailable")
    trial_root = tmp_path / "main64"
    source256_trial.prepare(preparation_path=PREPARATION, output=trial_root, mode="main")
    plan_path = tmp_path / "readback-plan.json"
    source256_readback.build_plan(preparation_path=PREPARATION, output=plan_path)
    packet_path = tmp_path / "evaluation" / "packet.json"
    packet = evaluation.prepare_packet(
        preparation_path=PREPARATION,
        trial_path=trial_root / "trial.json",
        plan_path=plan_path,
        qualification_path=QUALIFICATION,
        output=packet_path,
    )

    assert packet["status"] == "held_for_main64_lead_release"
    assert [item["label"] for item in packet["endpoints"]] == [
        "Source0",
        "A16",
        "A64",
        "B16",
        "B64",
    ]
    assert packet["bounds"]["readback_worker_count"] == 80
    assert packet["bounds"]["natural_image_requests"] == 1920
    assert all(item["start_identity"].startswith("original Source") for item in packet["training_launch"])
    assert all(len(endpoint["jobs"]) == 16 for endpoint in packet["endpoints"])
    evaluation._validate_packet(packet)


def test_reduce_loads_directory_of_bound_tokenizer_file(tmp_path, monkeypatch):
    import pytest
    import transformers
    from probes.training_set_completion import source256_evaluation as evaluation

    tokenizer_file = tmp_path / 'model' / 'tokenizer.json'
    tokenizer_file.parent.mkdir()
    tokenizer_file.write_text('{}')
    packet = tmp_path / 'packet.json'
    packet.write_text('{}')
    checked = {'prepared': {'preparation': {'identity': {'runtime_contract': {
        'tokenizer_path': str(tokenizer_file)
    }}}}}
    monkeypatch.setattr(evaluation, '_validate_packet', lambda _: checked)

    def load(path, **kwargs):
        assert str(path) == str(tokenizer_file.parent)
        assert kwargs['local_files_only'] is True
        return object()

    class ReachedScoring(Exception):
        pass

    def stop_after_load(_):
        raise ReachedScoring

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', load)
    monkeypatch.setattr(evaluation, '_targets', stop_after_load)
    with pytest.raises(ReachedScoring):
        evaluation.reduce(packet_path=packet, output=tmp_path / 'result.json')
