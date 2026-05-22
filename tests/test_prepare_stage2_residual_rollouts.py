from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from scripts.tools.prepare_stage2_residual_rollouts import (
    _infer_dataset_name_from_jsonl,
    _load_tokenizer_encode_fn,
    _runtime_sample_id,
    _should_try_next_tokenizer_source,
    _tokenizer_source_candidates,
    build_fixture_records,
)
from src.trainers.stage2_two_channel.rollout_views import (
    dedup_prepared_rollout_attempts,
    parse_prepared_rollout_attempt,
)


def test_prepare_stage2_residual_rollouts_fixture_records_have_v1_fields() -> None:
    records = build_fixture_records(
        [
            {
                "sample_id": "s0",
                "image_id": "image-0",
                "images": ["images/000000.jpg"],
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [10, 20, 30, 40],
                    }
                ],
            }
        ],
        encode_fn=lambda text: [ord(ch) for ch in text],
        expected_num_rollouts=2,
        seed=17,
        greedy_rollouts=1,
        sampling_rollouts=1,
        include_debug_cases=("exact_duplicate_attempt",),
    )

    assert len(records) == 3
    parsed = [
        parse_prepared_rollout_attempt(
            record,
            strict_prepared_rollout_tokens=True,
        )
        for record in records
    ]
    assert parsed[0].sample_id == "s0"
    assert parsed[0].image_id == "image-0"
    assert parsed[0].image_path == "images/000000.jpg"
    assert parsed[0].generation_config_hash.startswith("sha256:")
    assert parsed[0].response_token_ids == parsed[-1].response_token_ids
    kept, stats = dedup_prepared_rollout_attempts(
        parsed,
        legacy_reencode_fallback=False,
    )
    assert len(kept) == 2
    assert stats.K_after_dedup == 2
    assert stats.exact_duplicate_attempts == 1


def test_prepare_stage2_residual_rollouts_fixture_uses_runtime_sample_id_for_jsonl_records() -> None:
    records = build_fixture_records(
        [
            {
                "image_id": 9,
                "images": ["images/train2017/000000000009.jpg"],
                "objects": [
                    {
                        "desc": "orange",
                        "bbox_2d": [10, 20, 30, 40],
                    }
                ],
            }
        ],
        encode_fn=lambda text: [ord(ch) for ch in text],
        expected_num_rollouts=1,
        seed=17,
        greedy_rollouts=1,
        sampling_rollouts=0,
        dataset_name="coco",
    )

    assert records[0]["sample_id"] == _runtime_sample_id("coco", 0)
    assert records[0]["image_id"] == "9"
    assert records[0]["dataset"] == "coco"
    assert records[0]["base_idx"] == 0
    assert records[0]["rollout_id"] == f"{_runtime_sample_id('coco', 0)}:r0"


def test_prepare_stage2_residual_rollouts_infers_dataset_name_like_base_caption_dataset() -> None:
    assert (
        _infer_dataset_name_from_jsonl(
            Path("public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl")
        )
        == "coco"
    )


def test_prepare_stage2_residual_rollouts_fixture_keeps_clean_k_after_dedup() -> None:
    records = build_fixture_records(
        [
            {
                "sample_id": "s0",
                "image_id": "image-0",
                "images": ["images/000000.jpg"],
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [10, 20, 30, 40],
                    }
                ],
            }
        ],
        encode_fn=lambda text: [ord(ch) for ch in text],
        expected_num_rollouts=4,
        seed=17,
        greedy_rollouts=1,
        sampling_rollouts=3,
        include_debug_cases=(),
    )
    parsed = [
        parse_prepared_rollout_attempt(
            record,
            strict_prepared_rollout_tokens=True,
        )
        for record in records
    ]
    kept, stats = dedup_prepared_rollout_attempts(
        parsed,
        legacy_reencode_fallback=False,
    )

    assert len(records) == 4
    assert len({tuple(record["response_token_ids"]) for record in records}) == 4
    assert len(kept) == 4
    assert stats.K_after_dedup == 4


def test_prepare_stage2_residual_rollouts_tokenizer_falls_back_to_adapter_base(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/models/base-tokenizer"}),
        encoding="utf-8",
    )

    assert _tokenizer_source_candidates(str(adapter_dir)) == (
        str(adapter_dir),
        "/models/base-tokenizer",
    )
    assert _should_try_next_tokenizer_source(
        source_index=0,
        candidates=(str(adapter_dir), "/models/base-tokenizer"),
        source=str(adapter_dir),
    )


def test_prepare_stage2_residual_rollouts_tokenizer_loader_uses_adapter_fallback(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/models/base-tokenizer"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    class _FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(ch) for ch in text]

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source: str, *, trust_remote_code: bool) -> _FakeTokenizer:
            calls.append(str(source))
            if str(source) == str(adapter_dir):
                raise ValueError("missing tokenizer files")
            return _FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer),
    )

    encode = _load_tokenizer_encode_fn(str(adapter_dir))

    assert calls == [str(adapter_dir), "/models/base-tokenizer"]
    assert encode("ab") == [97, 98]


def test_prepare_stage2_residual_rollouts_tokenizer_loader_does_not_hide_bad_tokenizer(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/models/base-tokenizer"}),
        encoding="utf-8",
    )
    (adapter_dir / "tokenizer.model").write_text("broken", encoding="utf-8")
    assert not _should_try_next_tokenizer_source(
        source_index=0,
        candidates=(str(adapter_dir), "/models/base-tokenizer"),
        source=str(adapter_dir),
    )

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source: str, *, trust_remote_code: bool) -> object:
            raise ValueError(f"broken tokenizer at {source}")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer),
    )

    try:
        _load_tokenizer_encode_fn(str(adapter_dir))
    except ValueError as exc:
        assert "broken tokenizer" in str(exc)
    else:
        raise AssertionError("expected broken adapter-local tokenizer to surface")


def test_prepare_stage2_residual_rollouts_cli_preflight_checkpoint_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "tools" / "prepare_stage2_residual_rollouts.py"
    config = (
        root
        / "configs"
        / "stage2_two_channel"
        / "smoke"
        / "compact_full_residual_set_ckpt3664_hf_1step.yaml"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--out",
            str(root / "temp" / "prepared_rollouts_test.jsonl"),
            "--dry-run",
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert "checkpoint-3664" in payload["checkpoint"]
    assert payload["train_jsonl"].endswith("train.coord.jsonl")


def test_prepare_stage2_residual_rollouts_cli_real_mode_fails_before_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "tools" / "prepare_stage2_residual_rollouts.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(root / "does_not_exist.yaml"),
            "--out",
            str(root / "temp" / "prepared_rollouts_test.jsonl"),
            "--mode",
            "real",
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "real GPU generation is not implemented" in result.stderr
    assert "does_not_exist.yaml" not in result.stderr
