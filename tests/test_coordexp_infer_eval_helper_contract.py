from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_helper_module():
    script = (
        Path(__file__).resolve().parents[1]
        / ".codex"
        / "skills"
        / "coordexp-infer-eval-workflow"
        / "scripts"
        / "coordexp_infer_eval.py"
    )
    spec = importlib.util.spec_from_file_location("coordexp_infer_eval_helper", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _removed_generation_key(prefix: str, suffix: str) -> str:
    return prefix + suffix


def test_recursive_infer_eval_helper_emits_current_free_decode_surface(
    tmp_path: Path,
) -> None:
    helper = _load_helper_module()
    spec = helper.RecursiveInferEvalSpec(
        repo_root=tmp_path,
        checkpoint=tmp_path / "checkpoint-1",
        run_tag="surface-check",
        gpus="0",
        master_port=29501,
    )

    cfg = spec.build_config()

    assert cfg["detection_template"] == {"id": "compact_object_box_closed"}
    infer_cfg = cfg["infer"]
    generation_cfg = infer_cfg["generation"]
    assert "detection_sequence_format" not in infer_cfg
    assert _removed_generation_key("compact", "_grammar") not in generation_cfg
    assert _removed_generation_key("stop", "_pressure") not in generation_cfg
    assert _removed_generation_key("force", "_row_start") not in generation_cfg
    assert generation_cfg["top_p"] == 1.0
    assert infer_cfg["allow_diagnostic_gt_vs_pred"] is True
