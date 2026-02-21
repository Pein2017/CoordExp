from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from public_data.pipeline import PipelineConfig, PipelinePlanner
from public_data.pipeline.naming import apply_max_suffix, resolve_effective_preset
from public_data.pipeline.types import SplitArtifactPaths


def _write_image(path: Path, width: int = 128, height: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(120, 140, 160))
    img.save(path, format="JPEG")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_minimal_rows(split: str) -> list[dict]:
    return [
        {
            "images": [f"images/{split}2017/000000000001.jpg"],
            "objects": [
                {"bbox_2d": [10, 12, 80, 60], "desc": "person"},
                {"bbox_2d": [20, 16, 40, 50], "desc": "bag"},
            ],
            "width": 128,
            "height": 96,
        },
        {
            "images": [f"images/{split}2017/000000000002.jpg"],
            "objects": [{"bbox_2d": [15, 10, 70, 75], "desc": "dog"}],
            "width": 128,
            "height": 96,
        },
    ]


def _setup_dataset(dataset_dir: Path) -> Path:
    raw_dir = dataset_dir / "raw"
    train_rows = _make_minimal_rows("train")
    val_rows = _make_minimal_rows("val")

    _write_jsonl(raw_dir / "train.jsonl", train_rows)
    _write_jsonl(raw_dir / "val.jsonl", val_rows)

    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            rel = row["images"][0]
            _write_image(raw_dir / rel, width=row["width"], height=row["height"])

    return raw_dir


def test_adapter_registry_known_and_unknown() -> None:
    from public_data.pipeline.adapters import build_default_registry

    reg = build_default_registry()
    assert set(reg.ids()) >= {"coco", "lvis", "vg", "vg_ref"}
    assert reg.get("coco").dataset_id == "coco"
    assert reg.get("vg_ref").dataset_id == "vg_ref"

    with pytest.raises(KeyError, match="Unknown dataset id"):
        reg.get("unknown_dataset")


def test_adapter_registry_duplicate_registration_fails_fast() -> None:
    from public_data.pipeline.adapters.coco import CocoAdapter
    from public_data.pipeline.adapters.registry import AdapterRegistry

    reg = AdapterRegistry()
    reg.register(CocoAdapter())

    with pytest.raises(ValueError, match="already registered"):
        reg.register(CocoAdapter())


def test_adapter_ingestion_hooks_delegate_to_plugin_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.pipeline.adapters import build_default_registry
    from public_data.pipeline.adapters.base import DatasetAdapter

    calls: list[tuple[str, str, tuple[str, ...], Path]] = []

    def _fake_run_plugin_ingestion(
        self: DatasetAdapter,
        *,
        dataset_dir: Path,
        subcommand: str,
        passthrough_args=(),
    ) -> None:
        calls.append((self.dataset_id, subcommand, tuple(passthrough_args), dataset_dir))

    monkeypatch.setattr(DatasetAdapter, "_run_plugin_ingestion", _fake_run_plugin_ingestion)

    dataset_dir = tmp_path / "public_data" / "coco"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_default_registry().get("coco")
    adapter.download_raw_images(dataset_dir, passthrough_args=["--foo", "bar"])
    adapter.download_and_parse_annotations(dataset_dir, passthrough_args=["--split", "train"])

    assert calls == [
        ("coco", "download", ("--foo", "bar"), dataset_dir),
        ("coco", "convert", ("--split", "train"), dataset_dir),
    ]


@pytest.mark.parametrize(
    ("mode", "passthrough_args", "expected_subcommand"),
    [
        ("download", ["--foo", "bar"], "download"),
        ("convert", ["--split", "train"], "convert"),
    ],
)
def test_run_pipeline_factory_ingestion_dispatches_adapter_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    passthrough_args: list[str],
    expected_subcommand: str,
) -> None:
    from public_data.pipeline.adapters.base import DatasetAdapter
    from public_data.scripts import run_pipeline_factory

    class _StubAdapter(DatasetAdapter):
        dataset_id = "stub"

        def __init__(self) -> None:
            self.calls: list[tuple[str, Path, tuple[str, ...]]] = []

        def download_raw_images(self, dataset_dir: Path, *, passthrough_args=()) -> None:
            self.calls.append(("download", dataset_dir, tuple(passthrough_args)))

        def download_and_parse_annotations(self, dataset_dir: Path, *, passthrough_args=()) -> None:
            self.calls.append(("convert", dataset_dir, tuple(passthrough_args)))

    class _StubRegistry:
        def __init__(self, adapter: _StubAdapter) -> None:
            self._adapter = adapter

        def get(self, dataset_id: str) -> _StubAdapter:
            assert dataset_id == "stub"
            return self._adapter

    adapter = _StubAdapter()
    monkeypatch.setattr(run_pipeline_factory, "build_default_registry", lambda: _StubRegistry(adapter))

    dataset_dir = tmp_path / "public_data" / "stub"
    raw_dir = dataset_dir / "raw"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_factory.py",
            "--mode",
            mode,
            "--dataset-id",
            "stub",
            "--dataset-dir",
            str(dataset_dir),
            "--raw-dir",
            str(raw_dir),
            "--",
            *passthrough_args,
        ],
    )

    run_pipeline_factory.main()
    captured = capsys.readouterr()

    assert adapter.calls == [(expected_subcommand, dataset_dir, tuple(passthrough_args))]
    assert f"[pipeline] ingestion={mode}" in captured.out


def test_coco_adapter_prefers_fast_path_without_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.pipeline.adapters.base import DatasetAdapter
    from public_data.pipeline.adapters.coco import CocoAdapter

    dataset_dir = tmp_path / "public_data" / "coco"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    fast_path_calls: list[Path] = []
    plugin_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_fast_path(self: CocoAdapter, dir_path: Path) -> bool:
        fast_path_calls.append(dir_path)
        return True

    def _fake_plugin_ingestion(
        self: DatasetAdapter,
        *,
        dataset_dir: Path,
        subcommand: str,
        passthrough_args=(),
    ) -> None:
        plugin_calls.append((subcommand, tuple(passthrough_args)))

    monkeypatch.setattr(CocoAdapter, "_download_raw_images_aria2c", _fake_fast_path)
    monkeypatch.setattr(DatasetAdapter, "_run_plugin_ingestion", _fake_plugin_ingestion)

    CocoAdapter().download_raw_images(dataset_dir)

    assert fast_path_calls == [dataset_dir]
    assert plugin_calls == []


def test_coco_adapter_falls_back_to_plugin_when_fast_path_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.pipeline.adapters.base import DatasetAdapter
    from public_data.pipeline.adapters.coco import CocoAdapter

    dataset_dir = tmp_path / "public_data" / "coco"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    fast_path_calls: list[Path] = []
    plugin_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_fast_path(self: CocoAdapter, dir_path: Path) -> bool:
        fast_path_calls.append(dir_path)
        return False

    def _fake_plugin_ingestion(
        self: DatasetAdapter,
        *,
        dataset_dir: Path,
        subcommand: str,
        passthrough_args=(),
    ) -> None:
        plugin_calls.append((subcommand, tuple(passthrough_args)))

    monkeypatch.setattr(CocoAdapter, "_download_raw_images_aria2c", _fake_fast_path)
    monkeypatch.setattr(DatasetAdapter, "_run_plugin_ingestion", _fake_plugin_ingestion)

    CocoAdapter().download_raw_images(dataset_dir)

    assert fast_path_calls == [dataset_dir]
    assert plugin_calls == [("download", tuple())]


def test_coco_adapter_passthrough_skips_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.pipeline.adapters.base import DatasetAdapter
    from public_data.pipeline.adapters.coco import CocoAdapter

    dataset_dir = tmp_path / "public_data" / "coco"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    fast_path_calls: list[Path] = []
    plugin_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_fast_path(self: CocoAdapter, dir_path: Path) -> bool:
        fast_path_calls.append(dir_path)
        return True

    def _fake_plugin_ingestion(
        self: DatasetAdapter,
        *,
        dataset_dir: Path,
        subcommand: str,
        passthrough_args=(),
    ) -> None:
        plugin_calls.append((subcommand, tuple(passthrough_args)))

    monkeypatch.setattr(CocoAdapter, "_download_raw_images_aria2c", _fake_fast_path)
    monkeypatch.setattr(DatasetAdapter, "_run_plugin_ingestion", _fake_plugin_ingestion)

    CocoAdapter().download_raw_images(dataset_dir, passthrough_args=["--mirror", "cn"])

    assert fast_path_calls == []
    assert plugin_calls == [("download", ("--mirror", "cn"))]


def test_suffix_policy_and_legacy_equivalence(tmp_path: Path) -> None:
    _ = tmp_path

    assert apply_max_suffix("rescale_32_768_bbox", None) == "rescale_32_768_bbox"
    assert apply_max_suffix("rescale_32_768_bbox", 60) == "rescale_32_768_bbox_max60"
    assert apply_max_suffix("rescale_32_768_bbox_max60", 60) == "rescale_32_768_bbox_max60"
    assert resolve_effective_preset("rescale_32_768_bbox", 60) == "rescale_32_768_bbox_max60"

    with pytest.raises(ValueError, match="Legacy max-object suffix"):
        apply_max_suffix("rescale_32_768_bbox_max_60", 60)

    with pytest.raises(ValueError, match="Conflicting max_objects sources"):
        resolve_effective_preset("rescale_32_768_bbox_max60", 50)


def test_validate_raw_only_does_not_require_preset_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="",
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=cfg, mode="validate", validate_raw=True, validate_preset=False)

    assert "structural_preflight" in result.stage_stats
    assert "validation" in result.stage_stats
    assert not (dataset_dir / "pipeline_manifest.json").exists()
    assert not (dataset_dir / "train.jsonl").exists()


def test_structural_preflight_runs_without_optional_validation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=cfg, mode="rescale")

    train_paths = result.split_artifacts["train"]
    assert train_paths.raw.is_file()

    assert "validation" not in result.stage_stats
    assert "structural_preflight" in result.stage_stats
    assert result.stage_stats["structural_preflight"]["records"] == 4


def test_structural_preflight_fails_fast_before_downstream_stages(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = dataset_dir / "raw"
    bad_rows = [
        {
            "images": ["images/train2017/000000000001.jpg"],
            "objects": [
                {
                    "bbox_2d": [10, 12, 80, 60],
                    "poly": [10, 10, 20, 20, 20, 10],
                    "desc": "broken",
                }
            ],
            "width": 128,
            "height": 96,
        }
    ]
    _write_jsonl(raw_dir / "train.jsonl", bad_rows)
    _write_image(raw_dir / "images/train2017/000000000001.jpg")

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )

    with pytest.raises(ValueError, match="exactly one geometry key"):
        planner.run(config=cfg, mode="rescale")

    preset_dir = dataset_dir / "rescale_32_768_bbox_smoke"
    assert not (preset_dir / "train.jsonl").exists()


def test_max_objects_filter_drops_records_and_preserves_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    base_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )
    base_result = planner.run(config=base_cfg, mode="rescale")

    derived_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        max_objects=1,
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=derived_cfg, mode="coord")

    def read_rows(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    assert base_result.preset == "rescale_32_768_bbox_smoke"
    assert result.preset.endswith("_max1")

    for split in ("train", "val"):
        base_rows = read_rows(base_result.split_artifacts[split].raw)
        assert len(base_rows) == 2

        paths = result.split_artifacts[split]
        raw_rows = read_rows(paths.raw)
        norm_rows = read_rows(paths.norm)
        coord_rows = read_rows(paths.coord)

        assert len(raw_rows) == 1
        assert len(norm_rows) == 1
        assert len(coord_rows) == 1

        for rec in raw_rows:
            assert len(rec["objects"]) <= 1

        stats = json.loads(paths.filter_stats.read_text(encoding="utf-8"))
        assert stats["images_seen"] == 2
        assert stats["images_written"] == 1
        assert stats["images_dropped"] == 1
        assert stats["objects_seen"] == 3
        assert stats["objects_written"] == 1


def test_max_objects_requires_coord_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        max_objects=1,
        num_workers=1,
        run_validation_stage=False,
    )

    for mode in ("rescale", "full", "validate"):
        with pytest.raises(ValueError, match="only supported for mode 'coord'"):
            planner.run(config=cfg, mode=mode)


def test_normalize_stage_surfaces_object_drop_counters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """NormalizeStage may drop objects in rare safety-net cases.

    This test forces that path by monkeypatching bbox normalization to emit an
    invalid bbox once, and asserts that object-level counters are surfaced into
    the pipeline manifest stage_stats.
    """
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    import public_data.scripts.convert_to_coord_tokens as ctt

    orig = ctt._normalize_bbox_2d_to_norm1000
    calls = {"n": 0}

    def _patched_bbox(values, *, width: float, height: float, assume_normalized: bool):
        calls["n"] += 1
        if calls["n"] == 1:
            return [0, 0, 0, 0]
        return orig(values, width=width, height=height, assume_normalized=assume_normalized)

    monkeypatch.setattr(ctt, "_normalize_bbox_2d_to_norm1000", _patched_bbox)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=cfg, mode="full")

    normalize_stats = result.stage_stats["normalize_norm1000"]
    assert "objects" in normalize_stats

    train_obj = normalize_stats["objects"]["train"]
    assert train_obj["objects_seen"] == 3
    assert train_obj["objects_written"] == 2
    assert train_obj["objects_dropped_invalid_bbox"] == 1
    assert train_obj["objects_dropped_non_dict"] == 0

    val_obj = normalize_stats["objects"]["val"]
    assert val_obj["objects_seen"] == 3
    assert val_obj["objects_written"] == 3
    assert val_obj["objects_dropped_invalid_bbox"] == 0
    assert val_obj["objects_dropped_non_dict"] == 0


def test_full_pipeline_emits_jsonl_norm_coord(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=cfg, mode="full")
    assert "validation" not in result.stage_stats

    for split in ("train", "val"):
        paths = result.split_artifacts[split]
        assert paths.raw.is_file()
        assert paths.norm.is_file()
        assert paths.coord.is_file()

    manifest = result.preset_dir / "pipeline_manifest.json"
    assert manifest.is_file()


@pytest.mark.parametrize("mode", ["rescale", "coord", "full"])
def test_optional_validation_stage_wiring(tmp_path: Path, mode: str) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset=f"with_validation_{mode}",
        num_workers=1,
        run_validation_stage=True,
        skip_image_check=True,
    )

    if mode == "coord":
        prime_cfg = PipelineConfig(
            dataset_id="coco",
            dataset_dir=dataset_dir,
            raw_dir=raw_dir,
            preset=cfg.preset,
            num_workers=1,
            run_validation_stage=False,
            skip_image_check=True,
        )
        planner.run(config=prime_cfg, mode="rescale")

    result = planner.run(config=cfg, mode=mode)

    assert "validation" in result.stage_stats
    stats = result.stage_stats["validation"]
    assert stats["skip_image_check"] is True

    files = set(stats["files"])
    if mode == "rescale":
        expected = {
            str(result.preset_dir / "train.jsonl"),
            str(result.preset_dir / "val.jsonl"),
        }
    elif mode == "coord":
        expected = {
            str(result.preset_dir / "train.jsonl"),
            str(result.preset_dir / "val.jsonl"),
            str(result.preset_dir / "train.norm.jsonl"),
            str(result.preset_dir / "val.norm.jsonl"),
            str(result.preset_dir / "train.coord.jsonl"),
            str(result.preset_dir / "val.coord.jsonl"),
        }
    elif mode == "full":
        expected = {
            str(raw_dir / "train.jsonl"),
            str(raw_dir / "val.jsonl"),
            str(result.preset_dir / "train.jsonl"),
            str(result.preset_dir / "val.jsonl"),
            str(result.preset_dir / "train.norm.jsonl"),
            str(result.preset_dir / "val.norm.jsonl"),
            str(result.preset_dir / "train.coord.jsonl"),
            str(result.preset_dir / "val.coord.jsonl"),
        }
    else:
        raise AssertionError(mode)

    assert files == expected


def test_validation_stage_bounds_raw_image_open_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.scripts.validate_jsonl import JSONLValidator

    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    captured: list[tuple[str, str, int]] = []

    def _fake_validate_file(self, path: str) -> bool:
        captured.append((str(path), str(self.image_check_mode), int(self.image_check_n)))
        return True

    monkeypatch.setattr(JSONLValidator, "validate_file", _fake_validate_file)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=True,
        skip_image_check=False,
    )
    result = planner.run(config=cfg, mode="full")

    stats = result.stage_stats["validation"]
    assert int(stats["raw_image_open_check_n"]) == 64
    assert int(stats["preset_image_open_check_n"]) == 64

    raw_paths = {
        str(raw_dir / "train.jsonl"),
        str(raw_dir / "val.jsonl"),
    }
    raw_checks = [(mode, n) for path, mode, n in captured if path in raw_paths]
    assert raw_checks
    for mode, n in raw_checks:
        assert mode == "open"
        assert n == 64


def test_validation_stage_catches_preset_image_size_mismatch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    preset = "rescale_32_768_bbox_smoke"

    # First, build a valid preset.
    build_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset=preset,
        num_workers=1,
        run_validation_stage=False,
    )
    result = planner.run(config=build_cfg, mode="full")

    # Corrupt one preset image on disk (but keep JSONL meta unchanged).
    bad = result.preset_dir / "images/train2017/000000000001.jpg"
    _write_image(bad, width=64, height=64)

    validate_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset=preset,
        num_workers=1,
        run_validation_stage=False,
        skip_image_check=False,
    )

    with pytest.raises(RuntimeError, match="Validation stage failed"):
        planner.run(config=validate_cfg, mode="validate", validate_raw=False, validate_preset=True)


def test_rescale_stage_fails_fast_if_preset_images_path_exists(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    preset = "rescale_32_768_bbox_smoke"
    preset_dir = dataset_dir / preset
    preset_dir.mkdir(parents=True, exist_ok=True)

    os.symlink(raw_dir / "images", preset_dir / "images")

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset=preset,
        num_workers=1,
        run_validation_stage=False,
        skip_image_check=False,
    )

    with pytest.raises(RuntimeError, match="not fresh"):
        planner.run(config=cfg, mode="rescale")

    assert (preset_dir / "images").is_symlink()
    assert not (preset_dir / "train.jsonl").exists()


def test_rescale_stage_fails_fast_on_existing_preset_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )

    first = planner.run(config=cfg, mode="rescale")
    assert (first.preset_dir / "train.jsonl").is_file()
    assert (first.preset_dir / "images").is_dir()

    with pytest.raises(RuntimeError, match="not fresh"):
        planner.run(config=cfg, mode="rescale")


def test_materialize_derived_images_hardlinks_is_idempotent(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_preset"
    derived_dir = tmp_path / "derived_preset"

    src_img = base_dir / "images/train2017/000000000001.jpg"
    _write_image(src_img, width=128, height=96)

    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [{"bbox_2d": [10, 12, 80, 60], "desc": "person"}],
        "width": 128,
        "height": 96,
    }
    _write_jsonl(derived_dir / "train.jsonl", [row])

    split_paths = {
        "train": SplitArtifactPaths(
            split="train",
            raw=derived_dir / "train.jsonl",
            norm=derived_dir / "train.norm.jsonl",
            coord=derived_dir / "train.coord.jsonl",
            filter_stats=derived_dir / "train.filter_stats.json",
        )
    }

    PipelinePlanner._materialize_derived_images_hardlinks(
        base_preset_dir=base_dir,
        derived_preset_dir=derived_dir,
        split_paths=split_paths,
    )

    dst_img = derived_dir / "images/train2017/000000000001.jpg"
    src_stat = src_img.stat()
    dst_stat = dst_img.stat()
    assert src_stat.st_ino == dst_stat.st_ino
    assert src_stat.st_dev == dst_stat.st_dev

    # Idempotent: rerun should keep the same hardlink and succeed.
    PipelinePlanner._materialize_derived_images_hardlinks(
        base_preset_dir=base_dir,
        derived_preset_dir=derived_dir,
        split_paths=split_paths,
    )
    src_stat_after = src_img.stat()
    dst_stat_after = dst_img.stat()
    assert src_stat_after.st_ino == dst_stat_after.st_ino
    assert src_stat_after.st_dev == dst_stat_after.st_dev


def test_materialize_derived_images_hardlinks_fails_on_link_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base_preset"
    derived_dir = tmp_path / "derived_preset"

    src_img = base_dir / "images/train2017/000000000001.jpg"
    _write_image(src_img, width=128, height=96)

    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [{"bbox_2d": [10, 12, 80, 60], "desc": "person"}],
        "width": 128,
        "height": 96,
    }
    _write_jsonl(derived_dir / "train.jsonl", [row])

    split_paths = {
        "train": SplitArtifactPaths(
            split="train",
            raw=derived_dir / "train.jsonl",
            norm=derived_dir / "train.norm.jsonl",
            coord=derived_dir / "train.coord.jsonl",
            filter_stats=derived_dir / "train.filter_stats.json",
        )
    }

    def _fail_link(src: Path, dst: Path) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _fail_link)

    with pytest.raises(RuntimeError, match="no byte-copy fallback"):
        PipelinePlanner._materialize_derived_images_hardlinks(
            base_preset_dir=base_dir,
            derived_preset_dir=derived_dir,
            split_paths=split_paths,
        )


def test_coord_mode_materializes_derived_images_as_hardlinks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = _setup_dataset(dataset_dir)

    planner = PipelinePlanner()
    base_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        num_workers=1,
        run_validation_stage=False,
    )
    base_result = planner.run(config=base_cfg, mode="rescale")

    derived_cfg = PipelineConfig(
        dataset_id="coco",
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        preset="rescale_32_768_bbox_smoke",
        max_objects=1,
        num_workers=1,
        run_validation_stage=False,
    )
    derived_result = planner.run(config=derived_cfg, mode="coord")

    assert derived_result.preset.endswith("_max1")
    assert derived_result.preset_dir != base_result.preset_dir

    images_dir = derived_result.preset_dir / "images"
    assert images_dir.is_dir()
    assert not images_dir.is_symlink()

    with (derived_result.preset_dir / "train.jsonl").open("r", encoding="utf-8") as f:
        first_row = json.loads(f.readline())
    rel_img = Path(first_row["images"][0])

    base_img = base_result.preset_dir / rel_img
    derived_img = derived_result.preset_dir / rel_img
    assert base_img.exists()
    assert derived_img.exists()

    dropped_img = derived_result.preset_dir / "images/train2017/000000000001.jpg"
    assert not dropped_img.exists()

    base_stat = base_img.stat()
    derived_stat = derived_img.stat()
    assert base_stat.st_ino == derived_stat.st_ino
    assert base_stat.st_dev == derived_stat.st_dev


def test_run_pipeline_factory_cli_wires_run_validation_stage_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.scripts import run_pipeline_factory

    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = dataset_dir / "raw"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    captured_run_validation: list[bool] = []

    class _Result:
        def __init__(self, *, dataset_id: str, preset: str, preset_dir: Path) -> None:
            self.dataset_id = dataset_id
            self.preset = preset
            self.preset_dir = preset_dir
            self.split_artifacts: dict[str, object] = {}

    def _fake_run(self, *, config, mode, validate_raw=True, validate_preset=True):
        captured_run_validation.append(bool(config.run_validation_stage))
        return _Result(
            dataset_id=config.dataset_id,
            preset=config.preset,
            preset_dir=config.dataset_dir / config.preset,
        )

    monkeypatch.setattr(PipelinePlanner, "run", _fake_run)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_factory.py",
            "--mode",
            "full",
            "--dataset-id",
            "coco",
            "--dataset-dir",
            str(dataset_dir),
            "--raw-dir",
            str(raw_dir),
            "--preset",
            "rescale_32_768_bbox_smoke",
        ],
    )
    run_pipeline_factory.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_factory.py",
            "--mode",
            "full",
            "--dataset-id",
            "coco",
            "--dataset-dir",
            str(dataset_dir),
            "--raw-dir",
            str(raw_dir),
            "--preset",
            "rescale_32_768_bbox_smoke",
            "--no-run-validation-stage",
        ],
    )
    run_pipeline_factory.main()

    assert captured_run_validation == [True, False]


def test_run_pipeline_factory_cli_rejects_max_objects_outside_coord(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from public_data.scripts import run_pipeline_factory

    dataset_dir = tmp_path / "public_data" / "coco"
    raw_dir = dataset_dir / "raw"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_argv = [
        "run_pipeline_factory.py",
        "--dataset-id",
        "coco",
        "--dataset-dir",
        str(dataset_dir),
        "--raw-dir",
        str(raw_dir),
        "--preset",
        "rescale_32_768_bbox_smoke",
        "--max-objects",
        "1",
    ]

    for mode in ("rescale", "full", "validate"):
        argv = [base_argv[0], "--mode", mode, *base_argv[1:]]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc_info:
            run_pipeline_factory.parse_args()
        assert int(exc_info.value.code) == 2
