from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from src.common.detection_compact_rows import IM_END_TOKEN
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
from src.detection.token_types import build_compact_token_type_groups
from src.sft import (
    _require_root_image_dir_matches_detection,
    _resolve_root_image_dir_for_training,
)
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab
from test_detection_training_dataset import FakeSwiftTemplate


class RequiresNoResizeSwiftTemplate(FakeSwiftTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.do_resize_values: list[bool | None] = []

    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
        do_resize: bool | None = None,
    ) -> dict[str, Any]:
        self.do_resize_values.append(do_resize)
        if do_resize is not False:
            raise AssertionError("latest detection encode must pass do_resize=False")
        return super().encode(payload, return_length=return_length)


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


def _write_latest_compact_view(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path
    image_root = repo_root / "public_data/coco/images/res-1024"
    image_path = image_root / "images/val2017/000000000139.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")

    view_root = repo_root / "public_data/coco/views/coco80/len-12000"
    view_root.mkdir(parents=True, exist_ok=True)
    (view_root / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "annotation_view",
                "dataset": "coco",
                "view": "coco80/len-12000",
                "image_store": "public_data/coco/images/res-1024",
                "path_anchor": "repo_root",
                "image_path_semantics": "image_store_relative",
                "coordinate_space": "norm1000",
                "coordinate_storage": "integer",
                "coordinate_range": [0, 999],
                "coordinate_chart": "xyxy",
                "assistant_coordinate_rendering": "qwen_coord_tokens",
                "primary_jsonl": {"val": "val.jsonl"},
                "sample_policy": {
                    "type": "length_budget",
                    "max_total_tokens": 12000,
                },
                "length_budget_scope": {"rendered_families": ["assistant"]},
                "length_budget_template_id": "compact_full",
                "summary": {},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    row = _canonical_all_proxy_row()
    row["images"] = ["images/val2017/000000000139.jpg"]
    row["file_name"] = "000000000139.jpg"
    jsonl_path = view_root / "val.jsonl"
    _write_jsonl(jsonl_path, [row])

    return jsonl_path, image_root


def _write_symlinked_latest_compact_view(tmp_path: Path) -> tuple[Path, Path, str]:
    repo_root = tmp_path
    real_image_root = repo_root / "public_data/coco/shared/res-1024"
    image_path = real_image_root / "images/val2017/000000000139.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"unit-test-image-placeholder")

    metadata_image_store = repo_root / "public_data/coco/images/res-1024"
    metadata_image_store.parent.mkdir(parents=True, exist_ok=True)
    metadata_image_store.symlink_to(real_image_root, target_is_directory=True)

    explicit_image_store = repo_root / "public_data/coco/images-alias/res-1024"
    explicit_image_store.parent.mkdir(parents=True, exist_ok=True)
    explicit_image_store.symlink_to(real_image_root, target_is_directory=True)
    explicit_relative = "public_data/coco/images-alias/res-1024"

    view_root = repo_root / "public_data/coco/views/coco80/len-12000"
    view_root.mkdir(parents=True, exist_ok=True)
    (view_root / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "annotation_view",
                "dataset": "coco",
                "view": "coco80/len-12000",
                "image_store": "public_data/coco/images/res-1024",
                "path_anchor": "repo_root",
                "image_path_semantics": "image_store_relative",
                "coordinate_space": "norm1000",
                "coordinate_storage": "integer",
                "coordinate_range": [0, 999],
                "coordinate_chart": "xyxy",
                "assistant_coordinate_rendering": "qwen_coord_tokens",
                "primary_jsonl": {"val": "val.jsonl"},
                "sample_policy": {
                    "type": "length_budget",
                    "max_total_tokens": 12000,
                },
                "length_budget_scope": {"rendered_families": ["assistant"]},
                "length_budget_template_id": "compact_full",
                "summary": {},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    row = _canonical_all_proxy_row()
    row["images"] = ["images/val2017/000000000139.jpg"]
    row["file_name"] = "000000000139.jpg"
    jsonl_path = view_root / "val.jsonl"
    _write_jsonl(jsonl_path, [row])

    return jsonl_path, real_image_root, explicit_relative


def _load_latest_compact_view_dataset(
    jsonl_path: Path,
    *,
    image_root: str | Path | None = None,
) -> DetectionTrainingDataset:
    return DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=image_root,
        detection_template_id="compact_full",
        mode="sorted_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=123,
        state_weighting="none",
        normalization="token_mean",
    )


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


def test_teacher_forcing_hard_sft_dataset_emits_aligned_target_ir(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_canonical_all_proxy_row()])
    _ensure_image(tmp_path)
    swift_template = FakeSwiftTemplate()
    swift_template.tokenizer.bos_token_id = swift_template.tokenizer.convert_tokens_to_ids(
        "<|im_start|>"
    )
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=123,
        state_weighting="none",
        normalization="token_mean",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
    )

    sample = dataset[0]

    assert TEACHER_FORCING_TARGET_IR_KEY in sample
    assert "recursive_detection_targets" not in sample
    target_ir = sample[TEACHER_FORCING_TARGET_IR_KEY]
    assert target_ir.metadata["serialization_policy"] == "marker_delimited"
    assert target_ir.metadata["rollin_policy"] == "random_permutation"
    assert target_ir.metadata["stop_token_text"] == IM_END_TOKEN
    assert target_ir.metadata["pad_token_text"] == "<|endoftext|>"
    assert target_ir.metadata["parser_mode"] == "strict_expected"
    assert target_ir.metadata["compact_grammar_enabled"] is True
    assert all(
        atom.valid_token_ids == frozenset({atom.selected_token_id})
        for atom in target_ir.atoms
    )

    stop_token_id = swift_template.tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    assert target_ir.atoms[-1].selected_token_role is TokenRole.STOP
    assert target_ir.atoms[-1].selected_token_id == stop_token_id

    input_ids = sample["input_ids"]
    for atom in target_ir.atoms:
        assert atom.target_position == atom.logit_position + 1
        assert input_ids[atom.target_position] == atom.selected_token_id
        assert sample["labels"][atom.target_position] == atom.selected_token_id

    validate_target_ir(
        target_ir,
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        role_vocab=_role_vocab_for_fake_tokenizer(swift_template.tokenizer),
    )

    strip_candidate = dict(sample)
    strip_candidate.pop("length", None)
    model_inputs = strip_non_model_detection_sidecars(strip_candidate)
    assert TEACHER_FORCING_TARGET_IR_KEY not in model_inputs


def test_teacher_forcing_dataset_encode_passes_do_resize_false_when_supported(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_canonical_all_proxy_row()])
    _ensure_image(tmp_path)
    swift_template = RequiresNoResizeSwiftTemplate()
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=123,
        state_weighting="none",
        normalization="token_mean",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
    )

    sample = dataset[0]

    assert TEACHER_FORCING_TARGET_IR_KEY in sample
    assert swift_template.do_resize_values == [False]


def test_teacher_forcing_pure_valid_set_dataset_emits_ambiguous_atoms(
    tmp_path: Path,
) -> None:
    row = _canonical_all_proxy_row()
    row["objects"][0]["desc"] = "person"
    row["objects"][1]["desc"] = "person"
    row["objects"][0]["bbox_2d"] = [100, 100, 200, 220]
    row["objects"][1]["bbox_2d"] = [300, 120, 360, 180]
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [row])
    _ensure_image(tmp_path)
    swift_template = FakeSwiftTemplate()
    swift_template.tokenizer.bos_token_id = swift_template.tokenizer.convert_tokens_to_ids(
        "<|im_start|>"
    )
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=tmp_path / "image-root",
        detection_template_id="compact_full",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=123,
        state_weighting="none",
        normalization="token_mean",
        teacher_forcing_profile="pure_valid_set_marginal",
        teacher_forcing_rollin_base_seed=17,
    )

    target_ir = dataset[0][TEACHER_FORCING_TARGET_IR_KEY]

    ambiguous_x1_atoms = [
        atom
        for atom in target_ir.atoms
        if atom.coord_role == "x1" and len(atom.valid_token_ids) > 1
    ]
    assert ambiguous_x1_atoms
    assert any(
        atom.valid_token_ids != frozenset({atom.selected_token_id})
        for atom in target_ir.atoms
    )


def _role_vocab_for_fake_tokenizer(tokenizer: object) -> RoleVocab:
    groups = build_compact_token_type_groups(tokenizer)
    return RoleVocab(
        schema_token_ids=groups.struct,
        text_token_ids=groups.desc,
        coord_token_ids=groups.coord,
        stop_token_id=next(iter(groups.eos)),
    )


def test_dataset_loads_latest_compact_view_image_root_from_metadata(
    tmp_path: Path,
) -> None:
    jsonl_path, image_root = _write_latest_compact_view(tmp_path)

    dataset = _load_latest_compact_view_dataset(jsonl_path)
    sample = dataset[0]

    assert dataset._image_root == image_root.resolve()
    assert sample["messages"][1]["content"][0]["image"] == str(
        image_root / "images/val2017/000000000139.jpg"
    )
    assert sample["detection_metadata"]["object_count"] == 2


def test_dataset_accepts_explicit_image_root_matching_view_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl_path, image_root = _write_latest_compact_view(tmp_path)
    monkeypatch.chdir(jsonl_path.parent)

    dataset = _load_latest_compact_view_dataset(
        jsonl_path,
        image_root="public_data/coco/images/res-1024",
    )

    assert dataset._image_root == image_root.resolve()


def test_dataset_accepts_explicit_relative_image_root_matching_symlinked_metadata(
    tmp_path: Path,
) -> None:
    jsonl_path, real_image_root, explicit_relative = _write_symlinked_latest_compact_view(
        tmp_path
    )

    dataset = _load_latest_compact_view_dataset(
        jsonl_path,
        image_root=explicit_relative,
    )

    assert dataset._image_root == real_image_root.resolve()


def test_dataset_rejects_explicit_image_root_mismatching_view_metadata(
    tmp_path: Path,
) -> None:
    jsonl_path, _image_root = _write_latest_compact_view(tmp_path)

    with pytest.raises(ValueError, match="image_root.*does not match view metadata"):
        _load_latest_compact_view_dataset(
            jsonl_path,
            image_root=tmp_path / "public_data/coco/images/other-res",
        )


def test_sft_root_image_dir_uses_latest_compact_view_metadata_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl_path, image_root = _write_latest_compact_view(tmp_path)
    detection_config = type(
        "DetectionConfig",
        (),
        {"data": type("DataConfig", (), {"image_root": None})()},
    )()
    monkeypatch.chdir(tmp_path / "public_data/coco/views/coco80")

    resolved = _resolve_root_image_dir_for_training(
        detection_config=detection_config,
        train_jsonl=jsonl_path,
    )

    assert resolved == str(image_root.resolve())
    assert not resolved.endswith("/None")
    assert Path(resolved) != Path.cwd().resolve()


def test_sft_custom_root_image_dir_uses_view_metadata_not_jsonl_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl_path, image_root = _write_latest_compact_view(tmp_path)
    monkeypatch.chdir(tmp_path / "public_data/coco/views/coco80")

    resolved = _resolve_root_image_dir_for_training(
        detection_config=None,
        train_jsonl=jsonl_path,
    )

    assert resolved == str(image_root.resolve())
    assert Path(resolved) != jsonl_path.parent.resolve()


def test_sft_accepts_preexisting_root_image_dir_resolving_to_metadata_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jsonl_path, image_root, explicit_relative = _write_symlinked_latest_compact_view(
        tmp_path
    )
    detection_config = type(
        "DetectionConfig",
        (),
        {"data": type("DataConfig", (), {"image_root": None})()},
    )()
    resolved = _resolve_root_image_dir_for_training(
        detection_config=detection_config,
        train_jsonl=jsonl_path,
    )

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(tmp_path / explicit_relative))
    accepted = _require_root_image_dir_matches_detection(
        os.environ["ROOT_IMAGE_DIR"],
        resolved_image_root=resolved,
    )

    assert accepted == image_root.resolve()


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
