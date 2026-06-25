from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.infer.runtime import build_decode_request_from_infer_config


def _removed_generation_key(prefix: str, suffix: str) -> str:
    return prefix + suffix


def _load_infer_eval_helper():
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


def test_infer_decode_constraint_modules_are_not_importable() -> None:
    module_name = ".".join(("src", "infer", "constraints"))
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    "key",
    [
        _removed_generation_key("compact", "_grammar"),
        _removed_generation_key("stop", "_pressure"),
    ],
)
def test_removed_infer_generation_knobs_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="use free decode"):
        build_decode_request_from_infer_config(
            {
                "backend": {"type": "hf"},
                "generation": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    key: {"enabled": True},
                },
            }
        )


def test_coordexp_infer_eval_helper_uses_current_free_decode_surface(
    tmp_path: Path,
) -> None:
    helper = _load_infer_eval_helper()
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
