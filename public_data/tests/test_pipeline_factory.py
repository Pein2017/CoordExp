from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from public_data.pipeline import PipelineConfig, PipelinePlanner
from public_data.pipeline.naming import apply_max_suffix, resolve_effective_preset


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
    dataset_dir = tmp_path / "public_data" / "coco"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    assert apply_max_suffix("rescale_32_768_bbox", None) == "rescale_32_768_bbox"
    assert apply_max_suffix("rescale_32_768_bbox", 60) == "rescale_32_768_bbox_max_60"
    assert apply_max_suffix("rescale_32_768_bbox_max_60", 60) == "rescale_32_768_bbox_max_60"
    assert apply_max_suffix("rescale_32_768_bbox_max60", 60) == "rescale_32_768_bbox_max_60"

    legacy = dataset_dir / "rescale_32_768_bbox_max60"
    legacy.mkdir(parents=True, exist_ok=True)
    assert resolve_effective_preset(dataset_dir, "rescale_32_768_bbox", 60) == "rescale_32_768_bbox_max60"

    canonical = dataset_dir / "rescale_32_768_bbox_max_60"
    canonical.mkdir(parents=True, exist_ok=True)
    assert resolve_effective_preset(dataset_dir, "rescale_32_768_bbox", 60) == "rescale_32_768_bbox_max_60"


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
    assert not (dataset_dir / "train.raw.jsonl").exists()


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
    assert train_paths.legacy_raw_alias.is_file()

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
    assert not (preset_dir / "train.raw.jsonl").exists()


def test_max_objects_filter_drops_records_and_preserves_outputs(tmp_path: Path) -> None:
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
    result = planner.run(config=cfg, mode="full")

    def read_rows(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    for split in ("train", "val"):
        paths = result.split_artifacts[split]
        raw_rows = read_rows(paths.raw)
        norm_rows = read_rows(paths.norm)
        coord_rows = read_rows(paths.coord)

        assert len(raw_rows) == 1
        assert len(norm_rows) == 1
        assert len(coord_rows) == 1
        assert paths.legacy_raw_alias.is_file()

        for rec in raw_rows:
            assert len(rec["objects"]) <= 1

        stats = json.loads(paths.filter_stats.read_text(encoding="utf-8"))
        assert stats["images_seen"] == 2
        assert stats["images_written"] == 1
        assert stats["images_dropped"] == 1
        assert stats["objects_seen"] == 3
        assert stats["objects_written"] == 1


def test_full_pipeline_emits_raw_norm_coord_and_alias(tmp_path: Path) -> None:
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
        assert paths.legacy_raw_alias.is_file()

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
            str(result.preset_dir / "train.raw.jsonl"),
            str(result.preset_dir / "val.raw.jsonl"),
        }
    elif mode == "coord":
        expected = {
            str(result.preset_dir / "train.raw.jsonl"),
            str(result.preset_dir / "val.raw.jsonl"),
            str(result.preset_dir / "train.norm.jsonl"),
            str(result.preset_dir / "val.norm.jsonl"),
            str(result.preset_dir / "train.coord.jsonl"),
            str(result.preset_dir / "val.coord.jsonl"),
        }
    elif mode == "full":
        expected = {
            str(raw_dir / "train.jsonl"),
            str(raw_dir / "val.jsonl"),
            str(result.preset_dir / "train.raw.jsonl"),
            str(result.preset_dir / "val.raw.jsonl"),
            str(result.preset_dir / "train.norm.jsonl"),
            str(result.preset_dir / "val.norm.jsonl"),
            str(result.preset_dir / "train.coord.jsonl"),
            str(result.preset_dir / "val.coord.jsonl"),
        }
    else:
        raise AssertionError(mode)

    assert files == expected
