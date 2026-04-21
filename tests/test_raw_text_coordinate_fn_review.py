from pathlib import Path

from PIL import Image

from src.analysis.raw_text_coordinate_fn_review import (
    choose_best_oracle_run,
    materialize_fn_review_gallery,
    write_fn_review_gallery_html,
)


def test_choose_best_oracle_run_prefers_highest_full_hit_iou() -> None:
    chosen = choose_best_oracle_run(
        case_row={
            "oracle_runs": [
                {"label": "seed101", "full_hit": True, "loc_hit": True, "iou": 0.71},
                {"label": "seed102", "full_hit": True, "loc_hit": True, "iou": 0.83},
                {"label": "seed103", "full_hit": False, "loc_hit": True, "iou": 0.95},
            ]
        }
    )

    assert chosen["label"] == "seed102"


def test_write_fn_review_gallery_html_renders_dual_panels(tmp_path: Path) -> None:
    output_path = tmp_path / "review.html"
    write_fn_review_gallery_html(
        rows=[
            {
                "case_id": "fn:25:3:person",
                "gt_desc": "person",
                "gt_idx": 3,
                "image_id": 25,
                "recover_fraction_full": 1.0,
                "teacher_forced_support": 0.3,
                "competitor_margin": -0.1,
                "baseline_review_image": "cases/fn-25/baseline/rendered/vis_0000.png",
                "oracle_review_image": "cases/fn-25/oracle/rendered/vis_0000.png",
                "oracle_label": "raw_text_temp07_seed101",
                "oracle_iou": 0.86,
                "base_only_stop_pressure_signature": True,
                "base_plus_adapter_stop_pressure_signature": False,
                "base_only_continue_minus_eos_sum_logprob": -16.2,
                "base_only_continue_minus_eos_mean_logprob": 1.5,
                "base_plus_adapter_continue_minus_eos_sum_logprob": -24.1,
                "base_plus_adapter_continue_minus_eos_mean_logprob": -0.3,
                "mechanism_hint": "mixed_stop_pressure",
            }
        ],
        output_path=output_path,
        title="FN Review",
    )

    html_text = output_path.read_text(encoding="utf-8")

    assert "FN Review" in html_text
    assert "Baseline Miss" in html_text
    assert "Recovered Oracle" in html_text
    assert "cases/fn-25/baseline/rendered/vis_0000.png" in html_text
    assert "mixed_stop_pressure" in html_text


def test_materialize_fn_review_gallery_renders_case_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "demo.png"
    Image.new("RGB", (24, 24), color=(255, 255, 255)).save(image_path)

    baseline_jsonl = tmp_path / "baseline_gt_vs_pred.jsonl"
    baseline_jsonl.write_text(
        (
            '{"record_idx": 0, "image": "%s", "width": 24, "height": 24, '
            '"coord_mode": "pixel", '
            '"gt": [{"type": "bbox_2d", "points": [4, 4, 12, 12], "desc": "person"}], '
            '"pred": []}\n'
        )
        % image_path.as_posix(),
        encoding="utf-8",
    )

    oracle_jsonl = tmp_path / "oracle_gt_vs_pred.jsonl"
    oracle_jsonl.write_text(
        (
            '{"record_idx": 0, "image": "%s", "width": 24, "height": 24, '
            '"coord_mode": "pixel", '
            '"gt": [{"type": "bbox_2d", "points": [4, 4, 12, 12], "desc": "person"}], '
            '"pred": [{"type": "bbox_2d", "points": [4, 4, 12, 12], "desc": "person"}]}\n'
        )
        % image_path.as_posix(),
        encoding="utf-8",
    )

    rows = materialize_fn_review_gallery(
        selected_cases=[
            {
                "case_id": "fn:0:0:person",
                "record_idx": 0,
                "image_id": 0,
                "gt_idx": 0,
                "gt_desc": "person",
                "gt_bbox": [4, 4, 12, 12],
                "recover_fraction_full": 1.0,
                "teacher_forced_support": 0.5,
                "proposal_support": 0.0,
                "competitor_margin": -0.5,
                "oracle_runs": [
                    {
                        "label": "seed101",
                        "full_hit": True,
                        "loc_hit": True,
                        "iou": 0.9,
                        "pred_jsonl": oracle_jsonl.as_posix(),
                    }
                ],
            }
        ],
        margin_rows=[
            {
                "case_id": "fn:0:0:person",
                "model_alias": "base_only",
                "continue_minus_eos_sum_logprob": -10.0,
                "continue_minus_eos_mean_logprob": 1.0,
                "stop_pressure_signature": True,
            },
            {
                "case_id": "fn:0:0:person",
                "model_alias": "base_plus_adapter",
                "continue_minus_eos_sum_logprob": -4.0,
                "continue_minus_eos_mean_logprob": -0.2,
                "stop_pressure_signature": False,
            },
        ],
        baseline_gt_vs_pred_path=baseline_jsonl,
        output_dir=tmp_path / "gallery",
        title="FN Demo",
    )

    assert len(rows) == 1
    assert (tmp_path / "gallery" / "review.html").exists()
    assert (tmp_path / "gallery" / rows[0]["baseline_review_image"]).exists()
    assert (tmp_path / "gallery" / rows[0]["oracle_review_image"]).exists()
