from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research import run_resumable_natural_boundary_support_shard as worker
from src.artifacts.json_values import load_canonical_json


class _FakeScorer:
    def __init__(self) -> None:
        self._session = object()
        self.open_count = 0
        self.close_count = 0

    def open(self) -> None:
        self.open_count += 1

    def close(self) -> None:
        self.close_count += 1

    def score(self, _owner: object, candidate: dict[str, object]) -> float:
        return float(candidate["score"])


class _FakeSupportProbe:
    @staticmethod
    def validate_live_runtime_identity(
        _session: object,
        *,
        expected_checkpoint: str,
        expected_config_fingerprint: str,
    ) -> dict[str, object]:
        return {
            "normalized_device": "cuda:0",
            "checkpoint": expected_checkpoint,
            "config_fingerprint": expected_config_fingerprint,
        }

    @staticmethod
    def support_features(
        scores: dict[str, float], _group: object, *, owner_id: str
    ) -> dict[str, object]:
        return {"assessed": True, "owner_id": owner_id, "peak_lift": max(scores.values())}


def _context() -> dict[str, object]:
    return {
        "context_id": "ctx:fixture",
        "stable_key": "fixture",
        "gt_owner_id": "gt:1:1",
        "image_id": 1,
        "category_name": "person",
        "candidate_ids": ["a", "b"],
    }


def test_live_observer_opens_once_writes_runtime_and_preserves_observation_shape(
    tmp_path: Path,
) -> None:
    scorer = _FakeScorer()
    runner = SimpleNamespace(
        CHECKPOINT="S",
        support_probe=_FakeSupportProbe,
        _finite=lambda value, _label: float(value),
    )
    bank = SimpleNamespace(
        h0_by_owner={"gt:1:1": {"owner": "fixture"}},
        by_id={"a": {"score": 0.1}, "b": {"score": 0.2}},
        groups={(1, "person"): ({"candidate_id": "a"}, {"candidate_id": "b"})},
    )
    output = tmp_path / "runtime.json"
    observer = worker._LiveContextObserver(
        runner=runner,
        plan={"h0_lineage": {"config_fingerprint": "fingerprint"}},
        bank=bank,
        scorer=scorer,
        device="cuda:0",
        runtime_receipt=output,
        runtime_binding={"fixture": True},
    )
    first = observer(_context())
    second = observer(_context())
    observer.close()

    assert first == second
    assert first["status"] == "measured"
    assert first["candidate_ids"] == ["a", "b"]
    assert first["candidate_score_count"] == 2
    assert first["failure_count"] == 0
    assert scorer.open_count == 1
    assert scorer.close_count == 1
    assert load_canonical_json(output)["kind"] == "live_worker_runtime_identity"


def test_live_observer_journals_failure_shaped_context_without_hidden_retry(
    tmp_path: Path,
) -> None:
    scorer = _FakeScorer()

    def fail_once(_owner: object, candidate: dict[str, object]) -> float:
        if candidate["score"] == 0.2:
            raise RuntimeError("fixture failure")
        return float(candidate["score"])

    scorer.score = fail_once  # type: ignore[method-assign]
    runner = SimpleNamespace(
        CHECKPOINT="S",
        support_probe=_FakeSupportProbe,
        _finite=lambda value, _label: float(value),
    )
    bank = SimpleNamespace(
        h0_by_owner={"gt:1:1": {}},
        by_id={"a": {"score": 0.1}, "b": {"score": 0.2}},
        groups={(1, "person"): ()},
    )
    observer = worker._LiveContextObserver(
        runner=runner,
        plan={"h0_lineage": {"config_fingerprint": "fingerprint"}},
        bank=bank,
        scorer=scorer,
        device="cuda:0",
        runtime_receipt=tmp_path / "runtime.json",
        runtime_binding={"fixture": True},
    )
    observation = observer(_context())
    observer.close()

    assert observation["status"] == "failed"
    assert observation["candidate_score_count"] == 1
    assert observation["failure_count"] == 1
    assert scorer.open_count == 1


def test_worker_refuses_operator_substitution_of_admitted_hashes() -> None:
    args = worker.build_arg_parser().parse_args(
        [
            "--consumer", "/missing/consumer.py",
            "--consumer-sha256", "0" * 64,
            "--plan", "/missing/plan.json",
            "--census", "/missing/census.json",
            "--infer-config", "/missing/config.yaml",
            "--adapter-tensor", "/missing/adapter.safetensors",
            "--embedding-delta", "/missing/delta.safetensors",
            "--journal-root", "/missing/journal",
            "--runtime-receipt", "/missing/runtime.json",
            "--execution-id", "fixture",
            "--context-id", "ctx:fixture",
        ]
    )
    with pytest.raises(worker.ResumableWorkerError, match="CLI identity differs"):
        worker.run(args)


def test_base_model_binding_requires_exact_regular_file_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_file = model_root / "weights.safetensors"
    model_file.write_bytes(b"fixture weights")
    expected = hashlib.sha256(model_file.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "EXPECTED_BASE_MODEL", model_root)
    monkeypatch.setattr(worker, "EXPECTED_BASE_MODEL_FILES", {model_file.name: expected})

    resolved, identity = worker._require_model_root(model_root)

    assert resolved == model_root.resolve()
    assert identity["file_count"] == 1
    assert identity["files"][model_file.name]["sha256"] == expected

    extra = model_root / "ignored-subdirectory"
    extra.mkdir()
    with pytest.raises(worker.ResumableWorkerError, match="denominator differs"):
        worker._require_model_root(model_root)


def test_base_model_binding_rejects_symlink_file_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    target = tmp_path / "target.safetensors"
    target.write_bytes(b"fixture weights")
    linked = model_root / "weights.safetensors"
    linked.symlink_to(target)
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "EXPECTED_BASE_MODEL", model_root)
    monkeypatch.setattr(worker, "EXPECTED_BASE_MODEL_FILES", {linked.name: expected})

    with pytest.raises(worker.ResumableWorkerError, match="non-regular or symlink"):
        worker._require_model_root(model_root)
