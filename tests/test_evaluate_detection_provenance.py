from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "image": "demo.jpg",
                "width": 10,
                "height": 10,
                "gt": [],
                "pred": [{"bbox": [1, 1, 5, 5], "desc": "cat", "score": 1.0}],
                "pred_score_source": "test",
                "pred_score_version": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _argv(pred_jsonl: Path, out_dir: Path, *, metrics: str) -> list[str]:
    return [
        "evaluate_detection.py",
        "--pred_jsonl",
        str(pred_jsonl),
        "--out_dir",
        str(out_dir),
        "--metrics",
        metrics,
    ]


def test_evaluate_detection_cli_rejects_official_eval_without_score_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate_detection as eval_script

    pred_jsonl = tmp_path / "gt_vs_pred_scored.jsonl"
    _write_jsonl(pred_jsonl)

    def _forbidden_eval(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("evaluate_and_save should not run without provenance")

    monkeypatch.setattr(eval_script, "evaluate_and_save", _forbidden_eval)
    monkeypatch.setattr(
        "sys.argv",
        _argv(pred_jsonl, tmp_path / "eval", metrics="coco"),
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        eval_script.main()


def test_evaluate_detection_cli_allows_f1ish_inspection_without_score_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.evaluate_detection as eval_script

    pred_jsonl = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(pred_jsonl)

    captured: dict[str, Path | str] = {}

    def _fake_eval(pred_path, *, options):  # type: ignore[no-untyped-def]
        captured["pred_path"] = Path(pred_path)
        captured["metrics"] = str(options.metrics)
        return {"metrics": {"f1ish": 1.0}, "counters": {}}

    monkeypatch.setattr(eval_script, "evaluate_and_save", _fake_eval)
    monkeypatch.setattr(
        "sys.argv",
        _argv(pred_jsonl, tmp_path / "eval", metrics="f1ish"),
    )

    eval_script.main()

    assert captured == {"pred_path": pred_jsonl, "metrics": "f1ish"}
    out = capsys.readouterr().out
    assert '"f1ish": 1.0' in out
    assert '"evaluation_status": "inspection"' in out
    assert '"comparability": "non_comparable"' in out
    assert '"comparison_scope": "raw_f1ish"' in out


def test_evaluate_detection_cli_reports_f1ish_status_for_both_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.evaluate_detection as eval_script

    pred_jsonl = tmp_path / "gt_vs_pred_scored.jsonl"
    _write_jsonl(pred_jsonl)

    def _fake_eval(pred_path, *, options):  # type: ignore[no-untyped-def]
        return {
            "metrics": {"coco_map": 1.0, "f1ish": 0.5},
            "counters": {},
            "f1ish_evaluation_status": "inspection",
            "f1ish_comparability": "non_comparable",
            "f1ish_comparison_scope": "diagnostic_f1ish",
        }

    monkeypatch.setattr(eval_script, "load_comparable_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(eval_script, "evaluate_and_save", _fake_eval)
    monkeypatch.setattr(
        "sys.argv",
        _argv(pred_jsonl, tmp_path / "eval", metrics="both"),
    )

    eval_script.main()

    out = capsys.readouterr().out
    assert '"coco_map": 1.0' in out
    assert '"f1ish": 0.5' in out
    assert '"f1ish_evaluation_status": "inspection"' in out
    assert '"f1ish_comparability": "non_comparable"' in out
    assert '"f1ish_comparison_scope": "diagnostic_f1ish"' in out
