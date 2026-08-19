from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import src.prepare_train_cache as prepare_cli
from src.common.errors import RuntimeContractError
from src.training import cache_workflow


REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_receipt_hash(payload: dict[str, object]) -> None:
    observed = payload.pop("receipt_sha256")
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert observed == hashlib.sha256(encoded).hexdigest()


def test_prepare_cli_publishes_one_durable_completed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "model_loaded": False,
        "policy_identities": {"upstream_runtime_baseline": {"admitted": True}},
        "train": {"fingerprint": "train"},
        "eval": {"fingerprint": "eval"},
    }
    monkeypatch.setattr(
        prepare_cli,
        "prepare_training_pack_caches",
        lambda path, *, require_all_hit=False: result,
    )
    receipt_path = tmp_path / "receipt.json"

    assert (
        prepare_cli.main(
            ["--config", str(tmp_path / "config.yaml"), "--receipt", str(receipt_path)]
        )
        == 0
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    _assert_receipt_hash(payload)
    assert payload["schema"] == "coordexp-swift-pack-cache-preparation-receipt-v1"
    assert payload["terminal_status"] == "completed"
    assert payload["result"] == result
    assert payload["failure"] is None


def test_prepare_cli_publishes_bounded_failed_receipt_without_error_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(path: Path, *, require_all_hit: bool = False) -> dict[str, object]:
        raise RuntimeContractError(
            "secret-shaped diagnostic",
            code="training.prepare_failed",
        )

    monkeypatch.setattr(prepare_cli, "prepare_training_pack_caches", fail)
    receipt_path = tmp_path / "failed.json"

    with pytest.raises(RuntimeContractError):
        prepare_cli.main(
            ["--config", str(tmp_path / "config.yaml"), "--receipt", str(receipt_path)]
        )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    _assert_receipt_hash(payload)
    assert payload["terminal_status"] == "failed"
    assert payload["result"] is None
    assert payload["failure"] == {
        "error_type": "RuntimeContractError",
        "error_code": "training.prepare_failed",
    }
    assert "secret-shaped" not in receipt_path.read_text(encoding="utf-8")


def test_prepare_cli_rejects_existing_receipt_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        prepare_cli,
        "prepare_training_pack_caches",
        lambda path, *, require_all_hit=False: calls.append(path) or {},
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(Exception):
        prepare_cli.main(
            ["--config", str(tmp_path / "config.yaml"), "--receipt", str(receipt_path)]
        )

    assert calls == []
    assert receipt_path.read_text(encoding="utf-8") == "{}\n"


def test_production_cache_environment_crosses_real_strict_cpu_seam() -> None:
    child_environment = {
        **os.environ,
        **cache_workflow._cache_preparation_environment(),
    }
    child_environment.pop("CUDA_VISIBLE_DEVICES", None)
    code = """
import json
from types import SimpleNamespace

from src.training.cache_workflow import _establish_converged_runtime_determinism

receipt = _establish_converged_runtime_determinism(
    SimpleNamespace(
        seed=17,
        determinism=SimpleNamespace(mode="strict_cuda_replay_v1"),
    ),
    rank=0,
    world_size=1,
    rank_report_gatherer=None,
    phase="pack_cache_preparation",
)
print(json.dumps({
    "cuda_initialized": receipt["pre_apply_cuda_initialized"],
    "environment": receipt["environment"],
}, sort_keys=True))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "cuda_initialized": False,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        },
    }
