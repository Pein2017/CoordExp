from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_report import write_report_bundle


def test_write_report_bundle_materializes_required_outputs(tmp_path: Path) -> None:
    summary = {"q1": "inconclusive", "q2": "inconclusive"}
    write_report_bundle(
        output_dir=tmp_path,
        summary=summary,
        review_rows=[{"case_uid": "demo"}],
    )

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "review_queue.csv").exists()
