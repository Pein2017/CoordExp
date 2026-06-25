from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
from PIL import Image

import src.training.coverage_ledger.preflight as preflight_module
from src.config.loader import ConfigLoader
from src.common.model_paths import canonical_coordexp_repo_root
from src.detection.dataset import DetectionTrainingDataset
from src.training.coverage_ledger.artifacts import (
    CoverageLedgerOverlayCandidate,
    CoverageLedgerPreflightArtifactInputs,
    _norm1000_bbox_to_pixel_bbox,
    write_coverage_ledger_preflight_artifacts,
)
from src.training.coverage_ledger.preflight import (
    assert_smoke_config_diff_allowed,
    build_preflight_swift_template,
    _require_existing_file,
    _require_existing_path,
    _resolve_local_model_path,
    resolve_overlay_render_image_path,
    write_static_packed_materialization_preflight,
)
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.sidecars import SupervisionSidecars, TrainingSidecars
from src.training.coverage_ledger.visual_regions import VisualTokenRegion
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from test_detection_training_dataset import FakeSwiftTemplate, _raw_row, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/smoke"
BASELINE_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128_baseline.yaml"
LEDGER_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128.yaml"

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "source_jsonl_path",
    "source_jsonl_repo_path",
    "source_jsonl_resolved_path",
    "source_jsonl_sha256",
    "dataset_id",
    "split",
    "selected_row_indices",
    "selected_sample_ids",
    "selection_seed",
    "selection_algorithm",
    "template_id",
    "object_field_order",
    "tokenizer_id",
    "model_id",
    "processor_do_resize",
    "image_grid_metadata_version",
    "samples",
}

REQUIRED_SAMPLE_KEYS = {
    "row_index",
    "sample_id",
    "object_count",
    "image_identity",
    "processed_width",
    "processed_height",
    "image_grid_thw",
}


def _make_sidecar(index: int, image_path: Path) -> CoverageLedgerSidecar:
    entry = CoverageLedgerObjectEntry(
        object_instance_id=f"image-{index}:ann-{index}:src-0",
        source_object_index=0,
        emitted_order_index=0,
        image_index=0,
        bbox_norm1000_xyxy=(125, 150, 625, 700),
        object_ref_end_position=20 + index,
        box_start_position=21 + index,
        coord_label_positions=(22 + index, 23 + index, 24 + index, 25 + index),
        box_end_position=26 + index,
    )
    return CoverageLedgerSidecar(
        sample_id=f"sample-{index:03d}",
        prompt_end_position=19 + index,
        object_entries=(entry,),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        image_identity=str(image_path),
    )


def _make_target_ir(index: int) -> TeacherForcingTargetIR:
    return TeacherForcingTargetIR(
        schema_version=1,
        atoms=(
            SupervisionAtom(
                batch_index=0,
                logit_position=1,
                target_position=2,
                allowed_token_roles=frozenset({TokenRole.COORD}),
                selected_token_role=TokenRole.COORD,
                valid_token_ids=frozenset({100 + index}),
                selected_token_id=100 + index,
                latent_valid_token_ids=frozenset(),
                coverage_target_weights=None,
                loss_tags=frozenset({"coord"}),
                loss_weight=1.0,
                coord_role="x1",
                provenance={"coord_label_positions": (2, 3)},
            ),
        ),
        metadata={},
    )


class _TinyPackedPreflightDataset:
    object_ordering = "sorted"
    source_row_indices = (17, 23)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        index_i = int(index)
        image_path = Path(f"/tmp/unit-image-{index_i}.png")
        sidecar = _make_sidecar(index_i, image_path)
        token_base = 10 + index_i * 10
        return {
            "dataset": "unit",
            "base_idx": index_i,
            "sample_id": sidecar.sample_id,
            "input_ids": [token_base + offset for offset in range(5)],
            "labels": [token_base + offset for offset in range(5)],
            "attention_mask": [1, 1, 1, 1, 1],
            "length": 5,
            TEACHER_FORCING_TARGET_IR_KEY: _make_target_ir(index_i),
            "training_sidecars": TrainingSidecars(
                supervision=SupervisionSidecars(payloads=(sidecar,))
            ),
        }

    def _static_packing_precompute_info(self) -> dict[str, object]:
        return {"thread_safe": True}


class _TinyPackedPreflightTemplate:
    max_length = 10

    def __init__(self) -> None:
        self.packing = False
        self.padding_free = False

    def data_collator(self, batch: list[Any]) -> dict[str, Any]:
        assert self.packing is True
        assert self.padding_free is True
        assert len(batch) == 1
        pack = batch[0]
        input_ids: list[int] = []
        labels: list[int] = []
        attention_mask: list[int] = []
        offsets = [0]
        image_grid_rows: list[list[int]] = []
        for sample in pack:
            input_ids.extend(int(value) for value in sample["input_ids"])
            labels.extend(int(value) for value in sample["labels"])
            attention_mask.extend(int(value) for value in sample["attention_mask"])
            offsets.append(len(input_ids))
            image_grid_rows.append([1, 8, 8])
        text_position_ids = torch.empty((1, len(input_ids)), dtype=torch.long)
        for start, end in zip(offsets, offsets[1:]):
            text_position_ids[0, start:end] = torch.arange(end - start)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "position_ids": text_position_ids.unsqueeze(0).repeat(3, 1, 1),
            "text_position_ids": text_position_ids,
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(end - start for start, end in zip(offsets, offsets[1:])),
            "max_length_k": max(end - start for start, end in zip(offsets, offsets[1:])),
            "image_grid_thw": torch.tensor(image_grid_rows, dtype=torch.long),
        }


def _make_artifact_inputs(tmp_path: Path) -> CoverageLedgerPreflightArtifactInputs:
    source_jsonl = tmp_path / "train.coord.jsonl"
    source_jsonl.write_text(
        "".join(json.dumps({"row": index}) + "\n" for index in range(128)),
        encoding="utf-8",
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    sidecars: list[CoverageLedgerSidecar] = []
    regions_by_sample_id: dict[str, tuple[VisualTokenRegion, ...]] = {}
    overlay_candidates: list[CoverageLedgerOverlayCandidate] = []
    for index in range(128):
        image_path = image_dir / f"{index:03d}.png"
        Image.new(
            "RGB",
            (80, 80),
            color=(240 - index % 50, 245 - index % 40, 250 - index % 30),
        ).save(image_path)
        sidecar = _make_sidecar(index, image_path)
        sidecars.append(sidecar)
        region = VisualTokenRegion(
            row_start=1,
            row_end=4,
            col_start=1,
            col_end=4,
            flattened_indices=(5, 6, 7, 9, 10, 11, 13, 14, 15),
        )
        regions_by_sample_id[sidecar.sample_id] = (region,)
        if index < 16:
            overlay_candidates.append(
                CoverageLedgerOverlayCandidate(
                    sample_id=sidecar.sample_id,
                    row_index=index,
                    object_entry=sidecar.object_entries[0],
                    visual_region=region,
                    render_image_path=image_path,
                )
            )
    return CoverageLedgerPreflightArtifactInputs(
        output_root=tmp_path / "coverage_ledger_preflight_smoke",
        source_jsonl_path=source_jsonl,
        dataset_id="unit-detection",
        split="train",
        selected_row_indices=tuple(range(128)),
        selection_seed=20260623,
        selection_algorithm="seeded_random_without_replacement_v0",
        template_id="compact_object_box_closed",
        object_field_order="desc_first",
        tokenizer_id="unit-tokenizer",
        model_id="unit-model",
        processor_do_resize=False,
        image_grid_metadata_version="qwen3_vl_image_grid_thw_v0",
        sidecars=tuple(sidecars),
        visual_regions_by_sample_id=regions_by_sample_id,
        overlay_candidates=tuple(overlay_candidates),
        visual_grid_shape_by_sample_id={
            sidecar.sample_id: (4, 4) for sidecar in sidecars
        },
        expected_sample_count=128,
        expected_overlay_count=16,
    )


def test_artifact_writer_materializes_manifest_alignment_jsonl_and_overlay_index(
    tmp_path: Path,
) -> None:
    result = write_coverage_ledger_preflight_artifacts(_make_artifact_inputs(tmp_path))
    ledger_root = tmp_path / "coverage_ledger_preflight_smoke" / "ledger"

    assert result.ledger_root == ledger_root
    assert (ledger_root / "selected_samples.json").is_file()
    assert (ledger_root / "alignment_debug.jsonl").is_file()
    assert (ledger_root / "overlays").is_dir()
    assert (ledger_root / "overlays/index.json").is_file()

    manifest = json.loads((ledger_root / "selected_samples.json").read_text())
    assert REQUIRED_MANIFEST_KEYS <= set(manifest)
    assert manifest["source_jsonl_repo_path"] is None
    assert manifest["source_jsonl_resolved_path"] == str(
        (tmp_path / "train.coord.jsonl").resolve()
    )
    assert manifest["source_jsonl_sha256"]
    assert manifest["selected_row_indices"] == list(range(128))
    assert manifest["selected_sample_ids"] == [
        f"sample-{index:03d}" for index in range(128)
    ]
    assert manifest["selection_seed"] == 20260623
    assert manifest["selection_algorithm"] == "seeded_random_without_replacement_v0"
    assert manifest["template_id"] == "compact_object_box_closed"
    assert manifest["object_field_order"] == "desc_first"
    assert manifest["processor_do_resize"] is False
    assert len(manifest["samples"]) == 128
    assert REQUIRED_SAMPLE_KEYS <= set(manifest["samples"][0])

    alignment_lines = (ledger_root / "alignment_debug.jsonl").read_text().splitlines()
    assert len(alignment_lines) == 128
    first_alignment = json.loads(alignment_lines[0])
    assert first_alignment["failure_status"] == "ok"
    assert first_alignment["prompt_end_position"] == 19
    assert first_alignment["objects"][0]["object_ref_end_position"] == 20
    assert first_alignment["objects"][0]["box_start_position"] == 21
    assert first_alignment["objects"][0]["coord_label_positions"] == [22, 23, 24, 25]
    assert first_alignment["objects"][0]["box_end_position"] == 26
    assert first_alignment["objects"][0]["bbox_norm1000_xyxy"] == [125, 150, 625, 700]
    assert first_alignment["objects"][0]["mapped_visual_cells"] == {
        "row_start": 1,
        "row_end": 4,
        "col_start": 1,
        "col_end": 4,
        "flattened_indices": [5, 6, 7, 9, 10, 11, 13, 14, 15],
    }
    assert {json.loads(line)["failure_status"] for line in alignment_lines} == {"ok"}

    overlay_index = json.loads((ledger_root / "overlays/index.json").read_text())
    assert len(overlay_index["overlays"]) == 16
    overlay_paths = [ledger_root / item["path"] for item in overlay_index["overlays"]]
    assert all(path.is_file() for path in overlay_paths)
    assert all(path.suffix == ".png" for path in overlay_paths)
    assert overlay_index["overlays"][0]["sample_id"] == "sample-000"
    assert overlay_index["overlays"][0]["source_object_index"] == 0
    assert overlay_index["overlays"][0]["emitted_order_index"] == 0
    assert overlay_index["overlays"][0]["template_id"] == "compact_object_box_closed"


def test_static_packed_materialization_preflight_writes_packed_gate_json(
    tmp_path: Path,
) -> None:
    cfg = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))
    out_path = write_static_packed_materialization_preflight(
        cfg,
        dataset=_TinyPackedPreflightDataset(),
        swift_template=_TinyPackedPreflightTemplate(),
        output_root=tmp_path / "coverage_ledger_preflight_smoke",
    )

    payload = json.loads(out_path.read_text())
    assert out_path.name == "packed_materialization.json"
    assert payload["schema_version"] == "coverage_ledger_static_packed_materialization_v0"
    assert payload["packing_length"] == 10
    assert payload["segment_count"] == 2
    assert payload["pack_source_row_indices"] == [17, 23]
    assert payload["sample_ids"] == ["sample-000", "sample-001"]
    assert payload["segment_offsets"] == [
        {
            "sample_id": "sample-000",
            "packed_row_index": 0,
            "segment_index": 0,
            "token_start": 0,
            "token_end": 5,
        },
        {
            "sample_id": "sample-001",
            "packed_row_index": 0,
            "segment_index": 1,
            "token_start": 5,
            "token_end": 10,
        },
    ]
    assert payload["raw_position_ids_shape"] == [3, 1, 10]
    assert payload["forward_position_ids_shape"] == [4, 1, 10]
    assert payload["image_grid_row_count"] == 2
    assert payload["image_grid_thw"] == [[1, 8, 8], [1, 8, 8]]
    assert payload["teacher_forcing_target_ir_count"] == 2
    assert payload["coverage_ledger_sidecar_count"] == 2
    assert payload["shifted_sidecars"][1]["prompt_end_position"] == 25
    assert payload["shifted_sidecars"][1]["first_object_coord_label_positions"] == [
        28,
        29,
        30,
        31,
    ]


def test_artifact_writer_rejects_stale_overlay_files(tmp_path: Path) -> None:
    inputs = _make_artifact_inputs(tmp_path)
    stale_overlay = inputs.output_root / "ledger" / "overlays" / "stale.png"
    stale_overlay.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(stale_overlay)

    with pytest.raises(ValueError, match="stale|non-empty|overlays"):
        write_coverage_ledger_preflight_artifacts(inputs)


def test_overlay_bbox_conversion_uses_coordexp_norm999_contract() -> None:
    assert _norm1000_bbox_to_pixel_bbox(
        (0, 0, 999, 999),
        width=80,
        height=80,
    ) == [0, 0, 79, 79]

    with pytest.raises(ValueError, match="bbox_norm1000_xyxy"):
        _norm1000_bbox_to_pixel_bbox(
            (0, 0, 1000, 999),
            width=80,
            height=80,
        )


def test_preflight_enforces_smoke_config_diff_allowlist() -> None:
    baseline = ConfigLoader.load_materialized_training_config(str(BASELINE_CONFIG))
    ledger = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))

    assert_smoke_config_diff_allowed(baseline, ledger)


def test_preflight_swift_template_builder_uses_current_swift_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _Template:
        def __init__(self) -> None:
            self.modes: list[str] = []

        def set_mode(self, mode: str) -> None:
            self.modes.append(mode)

        def normalize_bbox(self, inputs: Any) -> None:
            raise AssertionError("coord adapter did not patch normalize_bbox")

    processor = object()
    template = _Template()

    def _fake_get_model_processor(*args: Any, **kwargs: Any) -> tuple[None, object]:
        calls["model_processor"] = (args, kwargs)
        return None, processor

    def _fake_get_template(*args: Any, **kwargs: Any) -> _Template:
        calls["template"] = (args, kwargs)
        assert args == ()
        assert kwargs["processor"] is processor
        return template

    monkeypatch.setattr(
        preflight_module,
        "get_model_processor",
        _fake_get_model_processor,
    )
    monkeypatch.setattr(preflight_module, "get_template", _fake_get_template)

    training_config = SimpleNamespace(
        model={
            "model": "local-model",
            "model_type": "qwen3_vl",
            "torch_dtype": "float32",
        },
        template={
            "template": "qwen3_vl",
            "max_length": 4096,
            "truncation_strategy": "raise",
            "max_pixels": 1048576,
        },
    )

    result = build_preflight_swift_template(
        training_config,  # type: ignore[arg-type]
        system_prompt="system",
    )

    assert result is template
    assert calls["model_processor"][0] == ("local-model",)
    assert calls["model_processor"][1]["model_type"] == "qwen3_vl"
    assert calls["model_processor"][1]["load_model"] is False
    assert calls["model_processor"][1]["download_model"] is False
    assert calls["template"][1]["template_type"] == "qwen3_vl"
    assert template.modes == ["train"]
    assert getattr(template, "_coord_tokens_skip_norm") is True


def test_preflight_packing_dummy_model_is_bound_to_qwen_template_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    dummy_model = object()

    class _Template:
        model = None
        dummy_model = None
        model_info = SimpleNamespace(model_type="qwen3_vl")

    def _fake_get_model_processor(*args: Any, **kwargs: Any) -> tuple[object, None]:
        calls["model_processor"] = (args, kwargs)
        return dummy_model, None

    monkeypatch.setattr(
        preflight_module,
        "get_model_processor",
        _fake_get_model_processor,
    )

    template = _Template()
    training_config = SimpleNamespace(
        model={
            "model": "local-model",
            "model_type": "qwen3_vl",
            "torch_dtype": "float32",
        }
    )

    preflight_module._ensure_preflight_packing_dummy_model(  # type: ignore[attr-defined]
        training_config,
        template,
    )

    assert calls["model_processor"][0] == ("local-model",)
    assert calls["model_processor"][1]["return_dummy_model"] is True
    assert calls["model_processor"][1]["model_type"] == "qwen3_vl"
    assert calls["model_processor"][1]["download_model"] is False
    assert template.dummy_model is dummy_model
    assert template.model is dummy_model


def test_preflight_prerequisite_guards_fail_before_swift_download(
    tmp_path: Path,
) -> None:
    missing_jsonl = tmp_path / "missing.coord.jsonl"
    with pytest.raises(FileNotFoundError, match="training JSONL"):
        _require_existing_file(missing_jsonl, "training JSONL")

    missing_model = tmp_path / "model-cache"
    with pytest.raises(FileNotFoundError, match="model cache"):
        _require_existing_path(missing_model, "model cache")

    assert _resolve_local_model_path("model_cache/models/local") == (
        canonical_coordexp_repo_root() / "model_cache/models/local"
    ).resolve(strict=False)
    assert _resolve_local_model_path("Qwen/Qwen3-VL-2B-Instruct") is None


class _ImageGridSwiftTemplate(FakeSwiftTemplate):
    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
        do_resize: bool | None = None,
    ) -> dict[str, Any]:
        encoded = super().encode(payload, return_length=return_length)
        encoded["image_grid_thw"] = (1, 8, 8)
        return encoded


def test_preflight_overlay_render_path_uses_resolved_dataset_image_path(
    tmp_path: Path,
) -> None:
    row = _raw_row()
    row["file_name"] = "images/train2017/example.jpg"
    row["images"] = ["images/train2017/example.jpg"]
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [row])
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(245, 245, 245)).save(image_path)

    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=_ImageGridSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=20260623,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order="desc_first",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
        coverage_ledger_enabled=True,
        dataset_name="unit",
    )

    sample = dataset[0]
    training_sidecars = sample["training_sidecars"]
    assert isinstance(training_sidecars, TrainingSidecars)
    sidecar = training_sidecars.supervision.payloads[0]
    assert isinstance(sidecar, CoverageLedgerSidecar)
    assert sidecar.image_identity == "images/train2017/example.jpg"

    assert resolve_overlay_render_image_path(sample) == image_path.resolve()
