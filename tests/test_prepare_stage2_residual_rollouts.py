from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.tools.prepare_stage2_residual_rollouts import build_fixture_records
from src.trainers.stage2_two_channel.rollout_views import parse_prepared_rollout_attempt


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
