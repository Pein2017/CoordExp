from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from scripts.stamp_inference_provenance import stamp_run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path) -> None:
    path.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")


def test_stamp_refuses_to_invent_missing_fingerprints_by_default() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(run_dir / "summary.json", {"legacy": True})
        _write_json(run_dir / "resolved_config.json", {"infer": {}})
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        assert exit_code == 1
        assert "comparable=false" in stdout.getvalue()
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()


def test_stamp_writes_raw_sidecar_when_exact_provenance_exists() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                }
            },
        )
        _write_json(run_dir / "resolved_config.json", {})

        exit_code = stamp_run_dir(run_dir)

        sidecar = json.loads(
            (run_dir / "gt_vs_pred.jsonl.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert exit_code == 0
        assert sidecar["prompt_policy_fingerprint"] == "prompt:1"
        assert sidecar["decode_policy_fingerprint"] == "decode:1"
        assert sidecar["model_identity_fingerprint"] == "model:1"
        assert sidecar["score_policy"] == "none"
        assert sidecar["comparable"] is True


def test_stamp_requires_score_fingerprint_for_scored_artifacts() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred_scored.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                }
            },
        )
        _write_json(run_dir / "resolved_config.json", {})
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        assert exit_code == 1
        assert "score_policy_fingerprint" in stdout.getvalue()
        assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()


def test_stamp_can_write_inspection_only_sidecar_when_allowed() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(run_dir / "summary.json", {"legacy": True})
        _write_json(run_dir / "resolved_config.json", {})

        exit_code = stamp_run_dir(run_dir, allow_inspection_stamp=True)

        sidecar = json.loads(
            (run_dir / "gt_vs_pred.jsonl.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert exit_code == 0
        assert sidecar["comparable"] is False
        assert sidecar["metric_bearing"] is False
        assert sidecar["missing_provenance"] == [
            "prompt_policy_fingerprint",
            "decode_policy_fingerprint",
            "model_identity_fingerprint",
            "score_policy",
        ]


def test_stamp_respects_carrier_level_non_comparable_veto() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "comparable": False,
                "metric_bearing": False,
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        assert exit_code == 1
        assert "comparable=false" in stdout.getvalue()
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()


def test_stamp_run_level_veto_overrides_exact_resolved_config() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(run_dir / "summary.json", {"comparable": False})
        _write_json(
            run_dir / "resolved_config.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        assert exit_code == 1
        assert "comparable=false" in stdout.getvalue()
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()


def test_stamp_run_level_veto_writes_only_inspection_sidecar_when_allowed() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(run_dir / "summary.json", {"metric_bearing": False})
        _write_json(
            run_dir / "resolved_config.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )

        exit_code = stamp_run_dir(run_dir, allow_inspection_stamp=True)

        sidecar = json.loads(
            (run_dir / "gt_vs_pred.jsonl.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert exit_code == 0
        assert sidecar["comparable"] is False
        assert sidecar["metric_bearing"] is False
        assert sidecar["inspection_reason"] == "metric_bearing=false"


def test_stamp_respects_nested_non_metric_veto_when_inspection_allowed() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "metric_bearing": False,
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )

        exit_code = stamp_run_dir(run_dir, allow_inspection_stamp=True)

        sidecar = json.loads(
            (run_dir / "gt_vs_pred.jsonl.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert exit_code == 0
        assert sidecar["comparable"] is False
        assert sidecar["metric_bearing"] is False
        assert "metric_bearing=false" in sidecar["inspection_reason"]


def test_stamp_nested_veto_overrides_exact_resolved_config() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "comparable": False,
                    "prompt_policy_fingerprint": "prompt:diagnostic",
                    "decode_policy_fingerprint": "decode:diagnostic",
                    "model_identity_fingerprint": "model:diagnostic",
                    "score_policy": "none",
                },
            },
        )
        _write_json(
            run_dir / "resolved_config.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        assert exit_code == 1
        assert "comparable=false" in stdout.getvalue()
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()


def test_stamp_nested_veto_with_exact_resolved_config_stays_inspection_only() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "metric_bearing": False,
                    "prompt_policy_fingerprint": "prompt:diagnostic",
                    "decode_policy_fingerprint": "decode:diagnostic",
                    "model_identity_fingerprint": "model:diagnostic",
                    "score_policy": "none",
                },
            },
        )
        _write_json(
            run_dir / "resolved_config.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )

        exit_code = stamp_run_dir(run_dir, allow_inspection_stamp=True)

        sidecar = json.loads(
            (run_dir / "gt_vs_pred.jsonl.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert exit_code == 0
        assert sidecar["comparable"] is False
        assert sidecar["metric_bearing"] is False
        assert sidecar["inspection_reason"] == "metric_bearing=false"


def test_stamp_rejects_transitional_and_non_string_fingerprints() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "transitional_prompt:1",
                    "decode_policy_fingerprint": 7,
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )
        stdout = io.StringIO()

        exit_code = stamp_run_dir(run_dir, stdout=stdout)

        output = stdout.getvalue()
        assert exit_code == 1
        assert "prompt_policy_fingerprint" in output
        assert "decode_policy_fingerprint" in output
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()


def test_stamp_does_not_partially_write_mixed_raw_and_scored_failures() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = Path(raw_dir)
        _write_jsonl(run_dir / "gt_vs_pred.jsonl")
        _write_jsonl(run_dir / "gt_vs_pred_scored.jsonl")
        _write_json(
            run_dir / "summary.json",
            {
                "inference_provenance": {
                    "prompt_policy_fingerprint": "prompt:1",
                    "decode_policy_fingerprint": "decode:1",
                    "model_identity_fingerprint": "model:1",
                    "score_policy": "none",
                },
            },
        )

        exit_code = stamp_run_dir(run_dir)

        assert exit_code == 1
        assert not (run_dir / "gt_vs_pred.jsonl.provenance.json").exists()
        assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()
