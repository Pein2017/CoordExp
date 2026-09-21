import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.analysis.candidate_field_cardinality_tomography import unmatched_peak_review_gallery as gallery
from scripts.analysis.candidate_field_cardinality_tomography import phase_a2_dual_checkpoint_analysis as phase
from scripts.analysis import coordexp_infras_fa2_length_precision_probe as fa2
from scripts.probes.coordexp_infras import dora_roundtrip as dora
from src.artifacts.output_layout import scan_output_root


def test_gallery_main_preserves_reports_in_json(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "x1_candidate_field_rows.jsonl").write_text("")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["gallery", "--et-root", str(source),
        "--pure-root", str(source), "--output-root", str(output)])
    assert gallery.main() == 0
    assert scan_output_root(output)["passed"]
    summary = json.loads((output / "unmatched_peak_summary.json").read_text())
    assert summary["summary"] == gallery.build_report(summary, 0, output)
    index = json.loads((output / "gallery/index.json").read_text())
    assert index["summary"] == gallery.build_gallery_index([])
    assert (output / "gallery/gallery_rows.jsonl").read_text() == ""


def test_dora_main_packages_card_before_saved_adapter_check(tmp_path, monkeypatch):
    monkeypatch.setattr(dora, "parse_args", lambda: SimpleNamespace(
        model_path=tmp_path, output_dir=tmp_path, device="cpu", dtype="fp32", max_targets=1))
    monkeypatch.setattr(dora.torch.cuda, "is_available", lambda: False)
    for name in ("require_local_model", "load_base_model", "assert_trainable_dora_surface",
                 "tiny_batch", "eval_logits"):
        monkeypatch.setattr(dora, name, Mock())
    monkeypatch.setattr(dora, "discover_language_linear_targets", lambda *a: (["linear"], ["linear"]))
    model = Mock()
    card = "# Generated PEFT card\n模型\n"
    model.save_pretrained.side_effect = lambda path, **kw: (path / "README.md").write_text(card)
    monkeypatch.setattr(dora, "build_dora_model", lambda *a: (model, None))
    monkeypatch.setattr(dora, "forward_backward_check", lambda *a: (None, {}))

    class SavedAdapterChecked(Exception):
        pass

    def check_saved(path):
        assert scan_output_root(path)["passed"]
        assert json.loads((path / "model_card.json").read_text())["content_utf8"] == card
        raise SavedAdapterChecked

    monkeypatch.setattr(dora, "assert_saved_adapter", check_saved)
    with pytest.raises(SavedAdapterChecked):
        dora.main()


@pytest.mark.parametrize("kind", ["phase", "fa2"])
def test_analysis_main_keeps_report_text_in_summary_json(tmp_path, monkeypatch, kind):
    report = "# Generated report\n\nAll rendered text.\n"
    if kind == "phase":
        monkeypatch.setattr(sys, "argv", ["phase", "--output-root", str(tmp_path)])
        monkeypatch.setattr(phase, "load_run", lambda *a: {
            "rows": {}, "row_count": 0, "duplicate_case_id_count": 0})
        monkeypatch.setattr(phase, "sensitivity_summary", lambda *a, **kw: [])
        monkeypatch.setattr(phase, "make_plots", lambda *a: {})
        monkeypatch.setattr(phase, "build_report", lambda data: report)
        assert phase.main() == 0
        path = tmp_path / "phase_a2_summary.json"
    else:
        monkeypatch.setattr(fa2, "_parse_args", lambda: SimpleNamespace(
            seed=0, device="cpu", config="unused", output_dir=tmp_path, example_count=1,
            target_index=0, sweep_lengths=[], positions_per_target=1, max_abs_tolerance=0.1))
        monkeypatch.setattr(fa2.torch.cuda, "is_available", lambda: False)
        config = SimpleNamespace(training=SimpleNamespace(precision="fp32"),
                                 model=SimpleNamespace(attn_implementation="eager"))
        monkeypatch.setattr(fa2, "load_train_config", lambda *a: SimpleNamespace(config=config))
        monkeypatch.setattr(fa2, "load_qwen_components", lambda *a, **kw: SimpleNamespace(model=Mock()))
        monkeypatch.setattr(fa2, "_load_encoded_examples", lambda *a, **kw: [SimpleNamespace(example_id="one")])
        monkeypatch.setattr(fa2, "_build_sweep_variants", lambda *a, **kw: {})
        monkeypatch.setattr(fa2, "_build_comparisons", lambda *a: {})
        monkeypatch.setattr(fa2, "_verdict", lambda *a, **kw: "test-only")
        monkeypatch.setattr(fa2, "_render_markdown", lambda data: report)
        assert fa2.main() == 0
        path = next(tmp_path.glob("*/summary.json"))
    assert json.loads(path.read_text())["summary"] == report
    assert scan_output_root(tmp_path)["passed"]
