from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from src.detection.data import (
    ObjectOrderingPlan,
    normalize_detection_row,
    parse_raw_detection_row,
)
from src.detection.dataset import (
    DetectionTrainingDataset,
    strip_non_model_detection_sidecars,
)
from src.detection.objective import (
    SemanticRole,
    normalize_recursive_detection_token_losses,
    prepare_detection_training_example,
)
from src.detection.template import get_detection_template
from test_detection_training_dataset import FakeSwiftTemplate


def _canonical_all_proxy_row() -> dict[str, Any]:
    return {
        "images": ["images/val2017/example.jpg"],
        "width": 640,
        "height": 480,
        "image_id": 139,
        "file_name": "000000000139.jpg",
        "metadata": {
            "source": "coco2017_lvis_proxy",
            "split": "val",
            "supervision": {
                "object_supervision": {
                    "lvis:ann:proxy-2": {
                        "source_role": "proxy_candidate",
                        "coordinate_weight": 0.0,
                        "regression_weight": 0.0,
                        "desc_ce_weight": 0.25,
                        "relation": "lvis_proxy_candidate",
                    },
                    "coco:ann:real-1": {
                        "source_role": "real",
                        "coordinate_weight": 1.0,
                        "regression_weight": 1.0,
                        "desc_ce_weight": 1.0,
                        "relation": "coco_ground_truth",
                    },
                },
                "support_objects": [
                    {
                        "object_id": "lvis:ann:support-3",
                        "bbox_2d": [10, 10, 20, 20],
                        "desc": "support umbrella",
                        "source_role": "support_object",
                    }
                ],
            },
        },
        "objects": [
            {
                "object_id": "coco:ann:real-1",
                "bbox_2d": [100, 100, 200, 220],
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": 111,
                "desc": "person",
            },
            {
                "object_id": "lvis:ann:proxy-2",
                "bbox_2d": [300, 120, 360, 180],
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": 222,
                "desc": "person",
            },
        ],
    }


def _normalized_all_proxy_sample():
    raw = parse_raw_detection_row(_canonical_all_proxy_row())
    return normalize_detection_row(raw, object_ordering=ObjectOrderingPlan.sorted())


def _span_sources_by_kind(
    sources: Sequence[Mapping[str, Any]], *, object_id: str, span_family: str
) -> list[Mapping[str, Any]]:
    matches = [
        source
        for source in sources
        if source.get("object_id") == object_id
        and source.get("span_family") == span_family
    ]
    assert matches
    return matches


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")


def _ensure_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image-root/images/val2017/example.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")


def test_metadata_supervision_joins_rendered_objects_by_stable_object_id() -> None:
    row = _canonical_all_proxy_row()

    raw = parse_raw_detection_row(row)
    sample = normalize_detection_row(raw, object_ordering=ObjectOrderingPlan.sorted())

    assert raw.metadata.supervision == row["metadata"]["supervision"]
    assert [obj.object_id for obj in sample.objects] == [
        "coco:ann:real-1",
        "lvis:ann:proxy-2",
    ]
    assert [
        obj.relation_snapshot["relation"] for obj in sample.objects if obj.relation_snapshot
    ] == ["coco_ground_truth", "lvis_proxy_candidate"]
    assert [obj.source_role for obj in sample.objects] == ["real", "proxy_candidate"]
    assert all(obj.object_id != obj.object_instance_id for obj in sample.objects)


def test_metadata_rejects_unknown_unrelated_keys_after_supervision_allowlist() -> None:
    row = _canonical_all_proxy_row()
    row["metadata"]["unexpected"] = "still strict"

    with pytest.raises(ValueError, match="metadata unsupported"):
        parse_raw_detection_row(row)


def test_compact_full_render_spans_carry_object_id_and_supervision_source() -> None:
    sample = _normalized_all_proxy_sample()

    rendered = get_detection_template("compact_full").render_assistant(sample)

    assert len(rendered.object_entries) == 2
    assert "support umbrella" not in rendered.text
    assert "lvis:ann:support-3" not in rendered.text
    assert [entry.object_id for entry in rendered.object_entries] == [
        "coco:ann:real-1",
        "lvis:ann:proxy-2",
    ]
    assert all(
        entry.object_id != entry.object_instance_id for entry in rendered.object_entries
    )

    object_events = [
        event
        for event in rendered.render_span_events
        if event.span_kind in {"description_text", "coordinate_slot"}
    ]
    assert object_events
    assert {event.object_id for event in object_events} == {
        "coco:ann:real-1",
        "lvis:ann:proxy-2",
    }
    assert {event.supervision_key for event in object_events} == {
        "coco:ann:real-1",
        "lvis:ann:proxy-2",
    }

    desc_events = [
        event for event in object_events if event.span_kind == "description_text"
    ]
    coord_events = [event for event in object_events if event.span_kind == "coordinate_slot"]
    assert all(event.span_family == "description" for event in desc_events)
    assert all(event.field_name == "desc" for event in desc_events)
    assert all(event.span_family == "geometry" for event in coord_events)
    assert all(event.field_name == "bbox_2d" for event in coord_events)
    assert all(event.relation_snapshot is not None for event in object_events)

    proxy_events = [
        event for event in object_events if event.object_id == "lvis:ann:proxy-2"
    ]
    assert proxy_events
    assert {event.source_role for event in proxy_events} == {"proxy_candidate"}
    assert all(event.coordinate_weight in {None, 0.0} for event in proxy_events)
    assert all(event.regression_weight in {None, 0.0} for event in proxy_events)
    assert all(event.hard_bbox_supervision is False for event in proxy_events)


def test_dataset_exposes_rendered_span_sources_without_model_input_leak(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "val.coord.jsonl"
    _write_jsonl(jsonl_path, [_canonical_all_proxy_row()])
    _ensure_image(tmp_path)
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="sorted_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        max_objects=60,
        seed=123,
        state_weighting="none",
        normalization="token_mean",
    )

    sample = dataset[0]

    assert sample["detection_metadata"]["object_count"] == 2
    sources = sample["rendered_span_sources"]
    assert sources
    real_desc = _span_sources_by_kind(
        sources,
        object_id="coco:ann:real-1",
        span_family="description",
    )[0]
    proxy_bbox_sources = _span_sources_by_kind(
        sources,
        object_id="lvis:ann:proxy-2",
        span_family="geometry",
    )
    assert real_desc["field_name"] == "desc"
    assert real_desc["source_role"] == "real"
    assert real_desc["supervision_key"] == "coco:ann:real-1"
    assert real_desc["relation_snapshot"]["relation"] == "coco_ground_truth"
    assert {source["field_name"] for source in proxy_bbox_sources} == {"bbox_2d"}
    assert {source["source_role"] for source in proxy_bbox_sources} == {
        "proxy_candidate"
    }
    assert all(source["coordinate_weight"] == 0.0 for source in proxy_bbox_sources)
    assert all(source["regression_weight"] == 0.0 for source in proxy_bbox_sources)
    assert all(source["hard_bbox_supervision"] is False for source in proxy_bbox_sources)

    strip_candidate = dict(sample)
    strip_candidate.pop("length", None)
    model_inputs = strip_non_model_detection_sidecars(strip_candidate)
    assert "rendered_span_sources" not in model_inputs


def test_proxy_candidate_bbox_coords_do_not_receive_recursive_bbox_supervision() -> None:
    raw = parse_raw_detection_row(_canonical_all_proxy_row())
    sample = normalize_detection_row(
        raw,
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=17,
            seed_source="unit-test",
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=get_detection_template("compact_full"),
        tokenizer=FakeSwiftTemplate().tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    assert prepared.recursive_detection_targets is not None
    targets_by_position = {
        target.position: target
        for target in prepared.recursive_detection_targets.token_targets
    }
    entries_by_object_id = {
        entry.object_id: entry for entry in prepared.tokenized.object_entries
    }
    real_entry = entries_by_object_id["coco:ann:real-1"]
    proxy_entry = entries_by_object_id["lvis:ann:proxy-2"]

    real_coord_targets = [
        targets_by_position[position]
        for coord_span in real_entry.coord_spans
        for position in coord_span.token_indices()
        if position in targets_by_position
    ]
    proxy_coord_positions = {
        position
        for coord_span in proxy_entry.coord_spans
        for position in coord_span.token_indices()
    }
    proxy_coord_targets = [
        targets_by_position[position]
        for position in sorted(proxy_coord_positions)
        if position in targets_by_position
    ]

    assert real_coord_targets
    assert not proxy_coord_targets
    proxy_coord_token_ids = {
        prepared.tokenized.input_ids[position]
        for position in proxy_coord_positions
    }
    assert all(
        target.loss_weight == pytest.approx(1.0) and target.coord_soft_targets
        for target in real_coord_targets
    )
    assert all(
        target.kind == "hard_ce"
        and target.semantic_role is SemanticRole.BBOX_COORD
        and target.valid_token_ids == (target.teacher_token_id,)
        and target.child_multiplicities == (1,)
        and target.child_probabilities == pytest.approx((1.0,))
        for target in real_coord_targets
    )
    assert all(
        proxy_coord_token_ids.isdisjoint(target.valid_token_ids)
        for target in real_coord_targets
    )
    assert all(
        proxy_coord_token_ids.isdisjoint(
            branch_target.token_id
            for branch_target in target.trie_branch_targets
        )
        for target in real_coord_targets
    )
    assert proxy_coord_positions.isdisjoint(targets_by_position)
    assert all(
        not (
            target.semantic_role is SemanticRole.OBJECT_CONTROL
            and target.position in proxy_coord_positions
        )
        for target in prepared.recursive_detection_targets.token_targets
    )

    uniform_losses = {
        target.position: 1.0
        for target in prepared.recursive_detection_targets.token_targets
    }
    normalized = normalize_recursive_detection_token_losses(
        prepared.recursive_detection_targets,
        uniform_losses,
    )

    assert normalized.normalized_loss == pytest.approx(1.0)
    assert normalized.diagnostics.component_losses["objects"] == pytest.approx(1.0)
    assert normalized.diagnostics.semantic_role_token_counts[
        SemanticRole.OBJECT_CONTROL
    ] == sum(
        1
        for target in prepared.recursive_detection_targets.token_targets
        if target.semantic_role is SemanticRole.OBJECT_CONTROL
        and target.position not in proxy_coord_positions
    )
    assert all(
        position not in proxy_coord_positions
        for atom in prepared.recursive_detection_targets.loss_atoms
        for position in atom.token_positions
    )
    assert normalized.diagnostics.state_weight_sum == pytest.approx(
        sum(
            target.state_weight
            for target in prepared.recursive_detection_targets.token_targets
            if target.position not in proxy_coord_positions
        )
    )
    assert all(
        target.loss_weight == pytest.approx(1.0)
        for target in prepared.recursive_detection_targets.token_targets
    )
