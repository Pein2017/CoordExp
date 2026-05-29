from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_submission_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pred_jsonl = tmp_path / "gt_vs_pred_scored.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    categories_json = tmp_path / "categories.json"
    out_json = tmp_path / "submission" / "coco_submission.json"
    _write_jsonl(
        pred_jsonl,
        [
            {
                "image": "demo.jpg",
                "width": 10,
                "height": 10,
                "gt": [],
                "pred": [{"bbox": [1, 1, 5, 5], "desc": "cat", "score": 1.0}],
                "pred_score_source": "test",
                "pred_score_version": 1,
            }
        ],
    )
    _write_jsonl(
        source_jsonl,
        [{"image": "demo.jpg", "image_id": 1, "width": 10, "height": 10}],
    )
    categories_json.write_text(
        json.dumps([{"id": 1, "name": "cat"}]),
        encoding="utf-8",
    )
    return pred_jsonl, source_jsonl, categories_json, out_json


def _argv(
    *,
    pred_jsonl: Path,
    source_jsonl: Path,
    categories_json: Path,
    out_json: Path,
) -> list[str]:
    return [
        "export_coco_submission.py",
        "--pred_jsonl",
        str(pred_jsonl),
        "--source_jsonl",
        str(source_jsonl),
        "--categories_json",
        str(categories_json),
        "--out_json",
        str(out_json),
    ]


def test_coco_submission_export_rejects_unprovenanced_scored_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.export_coco_submission as export_script

    pred_jsonl, source_jsonl, categories_json, out_json = _write_submission_inputs(
        tmp_path
    )

    def _forbidden_export(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("export_coco_submission should not run without provenance")

    monkeypatch.setattr(export_script, "export_coco_submission", _forbidden_export)
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            pred_jsonl=pred_jsonl,
            source_jsonl=source_jsonl,
            categories_json=categories_json,
            out_json=out_json,
        ),
    )

    with pytest.raises(ValueError, match="missing_provenance"):
        export_script.main()


def test_coco_submission_export_accepts_score_provenanced_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.export_coco_submission as export_script

    pred_jsonl, source_jsonl, categories_json, out_json = _write_submission_inputs(
        tmp_path
    )
    pred_jsonl.with_suffix(pred_jsonl.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:1",
                "decode_policy_fingerprint": "decode:1",
                "model_identity_fingerprint": "model:1",
                "score_policy_fingerprint": "score:1",
                "artifact_path": str(pred_jsonl),
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, Path] = {}

    def _fake_export(pred_path, *, source_jsonl, categories_json, out_json, options):
        captured["pred_path"] = Path(pred_path)
        captured["source_jsonl"] = Path(source_jsonl)
        captured["categories_json"] = Path(categories_json)
        captured["out_json"] = Path(out_json)
        return {"predictions_total": 1}

    monkeypatch.setattr(export_script, "export_coco_submission", _fake_export)
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            pred_jsonl=pred_jsonl,
            source_jsonl=source_jsonl,
            categories_json=categories_json,
            out_json=out_json,
        ),
    )

    export_script.main()

    assert captured == {
        "pred_path": pred_jsonl,
        "source_jsonl": source_jsonl,
        "categories_json": categories_json,
        "out_json": out_json,
    }
    assert "predictions_total=1" in capsys.readouterr().out
