from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.compact_full_parse_report import build_compact_full_parse_summary
from src.analysis.teacher_forcing_atom_probe import build_atom_probe_summary
from src.analysis.teacher_forcing_objective_report import build_objective_report_bundle


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_atom_probe_reports_coordinate_onset_and_text_ambiguity_separately(
    tmp_path: Path,
) -> None:
    atom_jsonl = tmp_path / "atoms.jsonl"
    _write_jsonl(
        atom_jsonl,
        [
            {
                "sample_id": "s1",
                "ambiguities": [
                    {"kind": "coordinate_onset", "roles": ["COORDINATE"]},
                    {"kind": "text", "roles": ["TEXT"]},
                    {"kind": "role_overlap", "roles": ["TEXT", "SCHEMA"]},
                ],
            }
        ],
    )

    summary = build_atom_probe_summary(atom_jsonl)

    assert summary["ambiguity"]["coordinate_onset_count"] == 1
    assert summary["ambiguity"]["text_count"] == 1
    assert summary["ambiguity"]["mixed_role_count"] == 1
    assert summary["ambiguity"]["mixed_role_pairs"]["SCHEMA|TEXT"] == 1
    assert summary["metrics"]["teacher_forcing/ambiguity/coordinate_onset_count"] == 1
    assert summary["metrics"]["teacher_forcing/ambiguity/mixed_role_count"] == 1


def test_objective_report_preserves_builder_rejection_codes_and_decode_absence(
    tmp_path: Path,
) -> None:
    metrics_jsonl = tmp_path / "metrics.jsonl"
    atoms_jsonl = tmp_path / "atoms.jsonl"
    builder_jsonl = tmp_path / "builder.jsonl"
    output_dir = tmp_path / "report"
    _write_jsonl(
        metrics_jsonl,
        [
            {
                "metrics": {
                    "teacher_forcing/loss/total": 2.0,
                    "teacher_forcing/loss/token_type_mass": 0.5,
                }
            }
        ],
    )
    _write_jsonl(
        atoms_jsonl,
        [
            {
                "ambiguities": [
                    {"kind": "coordinate_onset", "roles": ["COORDINATE"]},
                    {"kind": "role_overlap", "roles": ["TEXT", "SCHEMA"]},
                ]
            }
        ],
    )
    _write_jsonl(
        builder_jsonl,
        [
            {"status": "rejected", "reason": "missing_detection_list"},
            {"status": "rejected", "reason": "empty_detection_list"},
            {"status": "rejected", "reason": "overlength"},
            {"status": "rejected", "reason": "description_tokenization_failed"},
        ],
    )

    bundle = build_objective_report_bundle(
        metrics_jsonl=metrics_jsonl,
        output_dir=output_dir,
        atom_probe_jsonl=atoms_jsonl,
        builder_jsonl=builder_jsonl,
        decode_jsonl=None,
    )

    summary = json.loads(Path(bundle["summary_json"]).read_text(encoding="utf-8"))
    assert summary["builder"]["rejected_samples"] == 4
    assert summary["builder"]["rejection_reason_counts"] == {
        "description_tokenization_failed": 1,
        "empty_detection_list": 1,
        "missing_detection_list": 1,
        "overlength": 1,
    }
    assert summary["decode"]["artifact_status"] == "absent"
    assert "teacher_forcing/decode/object_coherence_rate" not in summary["metrics"]
    assert summary["metrics"]["teacher_forcing/loss/total/macro_avg"] == pytest.approx(2.0)
    assert "teacher_forcing/loss/total" not in summary["metrics"]
    assert "macro_avg suffix" in summary["metric_semantics"]["flat_input_metrics"]
    assert summary["metrics"]["teacher_forcing/ambiguity/coordinate_onset_count"] == 1
    assert summary["metrics"]["teacher_forcing/ambiguity/mixed_role_count"] == 1
    assert Path(bundle["report_md"]).read_text(encoding="utf-8").startswith(
        "# Teacher-Forcing Objective Diagnostics"
    )


def test_objective_report_emits_decode_rates_when_generation_artifact_is_present(
    tmp_path: Path,
) -> None:
    metrics_jsonl = tmp_path / "metrics.jsonl"
    decode_jsonl = tmp_path / "decode.jsonl"
    output_dir = tmp_path / "report"
    _write_jsonl(metrics_jsonl, [{"metrics": {"teacher_forcing/loss/total": 1.0}}])
    _write_jsonl(
        decode_jsonl,
        [
            {
                "object_coherent": True,
                "duplicate": False,
                "missed_object": True,
                "malformed_sequence": False,
            },
            {
                "object_coherent": False,
                "duplicate": True,
                "missed_object": False,
                "malformed_sequence": True,
            },
        ],
    )

    bundle = build_objective_report_bundle(
        metrics_jsonl=metrics_jsonl,
        output_dir=output_dir,
        decode_jsonl=decode_jsonl,
    )

    summary = json.loads(Path(bundle["summary_json"]).read_text(encoding="utf-8"))
    assert summary["decode"]["artifact_status"] == "present"
    assert summary["decode"]["sample_count"] == 2
    assert summary["metrics"]["teacher_forcing/decode/object_coherence_rate"] == pytest.approx(0.5)
    assert summary["metrics"]["teacher_forcing/decode/duplicate_rate"] == pytest.approx(0.5)
    assert summary["metrics"]["teacher_forcing/decode/missed_object_rate"] == pytest.approx(0.5)
    assert summary["metrics"]["teacher_forcing/decode/malformed_sequence_rate"] == pytest.approx(0.5)


def test_objective_report_marks_flat_metric_aggregation_as_macro_average(
    tmp_path: Path,
) -> None:
    metrics_jsonl = tmp_path / "metrics.jsonl"
    output_dir = tmp_path / "report"
    _write_jsonl(
        metrics_jsonl,
        [
            {"metrics": {"teacher_forcing/loss/total": 1.0}},
            {"metrics": {"teacher_forcing/loss/total": 3.0}},
        ],
    )

    bundle = build_objective_report_bundle(
        metrics_jsonl=metrics_jsonl,
        output_dir=output_dir,
    )

    summary = json.loads(Path(bundle["summary_json"]).read_text(encoding="utf-8"))
    assert summary["metrics"]["teacher_forcing/loss/total/macro_avg"] == pytest.approx(2.0)
    assert "teacher_forcing/loss/total" not in summary["metrics"]


def test_compact_full_parse_report_preserves_error_codes(tmp_path: Path) -> None:
    parse_jsonl = tmp_path / "parse.jsonl"
    _write_jsonl(
        parse_jsonl,
        [
            {"ok": False, "error_code": "missing_object_marker"},
            {"ok": False, "error_code": "missing_object_marker"},
            {"ok": False, "error_code": "bad_coordinate"},
            {"ok": True},
        ],
    )

    summary = build_compact_full_parse_summary(parse_jsonl)

    assert summary["rows"] == 4
    assert summary["ok_count"] == 1
    assert summary["error_counts"] == {
        "bad_coordinate": 1,
        "missing_object_marker": 2,
    }
    assert summary["metrics"]["infer/parse/compact_full/error/missing_object_marker"] == 2
    assert summary["metrics"]["infer/parse/compact_full/error/bad_coordinate"] == 1
