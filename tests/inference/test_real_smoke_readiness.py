from __future__ import annotations

import json
from pathlib import Path

import yaml


SMOKE_CONFIGS = {
    "single": Path("configs/coordexp_swift/infer/wave7_real_base_single_smoke.yaml"),
    "batched": Path("configs/coordexp_swift/infer/wave7_real_base_batched_smoke.yaml"),
    "adapter": Path("configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml"),
}
BENCHMARK_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml"
)
SINGLE_FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.single.jsonl")
TWO_ROW_FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl")
BENCHMARK_PACKET = Path(
    "docs/superpowers/plans/2026-07-02-coordexp-swift-wave7-benchmark-readiness.md"
)


def test_wave7_single_fixture_is_pinned_from_training_smoke_family() -> None:
    source_rows = _read_jsonl(TWO_ROW_FIXTURE)
    single_rows = _read_jsonl(SINGLE_FIXTURE)

    assert len(source_rows) == 2
    assert len(single_rows) == 1
    assert single_rows[0] == source_rows[0]
    image_path = SINGLE_FIXTURE.parent / single_rows[0]["image"]["path"]
    assert image_path.is_file()
    assert single_rows[0]["image"]["width"] == 1248
    assert single_rows[0]["image"]["height"] == 832
    assert [obj["object_id"] for obj in single_rows[0]["objects"]] == ["291613", "1155486"]


def test_wave7_smoke_configs_are_schema_valid_and_smoke_only() -> None:
    from src.config.inference import load_infer_config

    configs = {name: load_infer_config(path).config for name, path in SMOKE_CONFIGS.items()}

    assert configs["single"].data.input_jsonl.endswith("examples.single.jsonl")
    assert configs["single"].generation.batch_size == 1
    assert configs["batched"].data.input_jsonl.endswith("examples.jsonl")
    assert configs["batched"].generation.batch_size == 2
    assert configs["adapter"].adapter is not None
    assert configs["adapter"].embedding_delta is not None
    for config in configs.values():
        assert config.debug.smoke is True
        assert config.debug.dry_run is False
        assert config.run.collision_policy == "timestamp"
        assert config.generation.max_new_tokens <= 64
        assert Path(config.model.base_model) == Path(
            "/data/CoordExp/model_cache/models/Qwen/"
            "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
        ).resolve()


def test_wave7_benchmark_leaf_is_not_a_smoke_config() -> None:
    from src.config.inference import load_infer_config

    config = load_infer_config(BENCHMARK_CONFIG).config

    assert config.debug.smoke is False
    assert config.debug.dry_run is False
    assert config.generation.batch_size > 1
    assert config.generation.max_new_tokens >= 256
    assert config.data.input_jsonl == "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl"
    assert config.run.artifact_root.endswith("outputs/coordexp_swift/infer/benchmark")


def test_benchmark_packet_contains_required_handles_and_approval_stop() -> None:
    text = BENCHMARK_PACKET.read_text(encoding="utf-8")
    required = {
        "STATUS: BLOCKED_PENDING_FULL_BENCHMARK_APPROVAL",
        f"Production config: `{BENCHMARK_CONFIG}`",
        "Dataset: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`",
        "Base model: `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`",
        "Adapter checkpoint:",
        "Latest adapter smoke root:",
        "model_identity.embedding_delta.status=loaded",
        "Artifact root: `outputs/coordexp_swift/infer/benchmark`",
        "Evaluator command:",
        "full benchmark launch is still blocked pending explicit user approval.",
        "Tiny and sample-limited smokes are smoke/partial evidence only, not benchmark evidence.",
        "Base smokes did not produce non-empty selected-token scoring evidence.",
        "Base smokes did not naturally observe `<|im_end|>` stop or post-stop padding.",
        "Rollback path:",
    }

    for needle in required:
        assert needle in text


def test_new_inference_configs_do_not_use_legacy_infer_authority() -> None:
    assert Path("src/infer.py").is_file()
    assert not Path("src/infer").exists()
    for path in (*SMOKE_CONFIGS.values(), BENCHMARK_CONFIG):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload.get("extends") != "../../infer/base.yaml"
        assert "configs/infer" not in path.read_text(encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
