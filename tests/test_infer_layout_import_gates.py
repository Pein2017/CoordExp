from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TEXT_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
    REPO_ROOT / "configs",
    REPO_ROOT / "docs",
    REPO_ROOT / "openspec" / "specs",
)
ACTIVE_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml"}


def _iter_active_text_files():
    for root in ACTIVE_TEXT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in ACTIVE_TEXT_SUFFIXES:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[:2] == ("docs", "superpowers"):
                continue
            yield path


def _find_spec_or_none(name: str):
    import importlib.util

    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


def test_infer_package_root_exports_only_shared_runtime_surface() -> None:
    import src.infer as infer

    assert "DetectionDecodeRequest" in infer.__all__
    assert "DetectionDecodeResult" in infer.__all__
    assert "DetectionPromptPolicy" in infer.__all__
    assert "build_prompt_bundle" in infer.__all__

    assert "GenerationConfig" not in infer.__all__
    assert "InferenceConfig" not in infer.__all__
    assert "InferenceEngine" not in infer.__all__
    assert not hasattr(infer, "GenerationConfig")
    assert not hasattr(infer, "InferenceConfig")
    assert not hasattr(infer, "InferenceEngine")


def test_infer_backends_module_is_removed() -> None:
    assert _find_spec_or_none("src.infer.backends") is None


def test_legacy_infer_engine_module_is_removed() -> None:
    assert _find_spec_or_none("src.infer.engine") is None


def test_trainer_vllm_compat_module_is_removed() -> None:
    assert _find_spec_or_none("src.trainers.rollout_runtime") is None
    assert _find_spec_or_none("src.trainers.rollout_runtime.vllm_compat") is None


def test_trainer_swift_infer_compat_module_is_removed() -> None:
    assert (
        _find_spec_or_none("src.trainers.rollout_runtime.swift_infer_compat")
        is None
    )


def test_backend_owns_swift_infer_compat_helpers() -> None:
    from src.infer import backend

    assert callable(backend.import_swift_request_config)
    assert callable(backend.import_swift_infer_request_and_config)
    assert callable(backend.import_swift_to_device)


def test_active_code_does_not_import_trainer_rollout_runtime() -> None:
    needles = (
        "src.trainers.rollout_runtime",
        "from .rollout_runtime",
        "from src.trainers.rollout_runtime",
        "rollout_runtime/",
    )
    offenders: list[str] = []
    for path in _iter_active_text_files():
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{needle}")
    assert offenders == []


def test_active_code_does_not_import_legacy_infer_engine_directly() -> None:
    needles = (
        "src.infer.engine",
        "from src.infer import engine",
        "from src.infer.engine",
        "import src.infer.engine",
    )
    allowed = {
        Path("src/infer/engine.py"),
        Path("src/infer/runtime.py"),
        Path("tests/test_infer_layout_import_gates.py"),
        Path("tests/test_inference_runtime_backend_facade.py"),
    }
    offenders: list[str] = []
    for path in _iter_active_text_files():
        rel = path.relative_to(REPO_ROOT)
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                offenders.append(f"{rel}:{needle}")
    assert offenders == []


def test_active_scripts_and_docs_do_not_import_stage2_rollout_runtime() -> None:
    needles = (
        "src.trainers.stage2_rollout_runtime",
        "from src.trainers.stage2_rollout_runtime",
    )
    roots = (
        REPO_ROOT / "scripts",
        REPO_ROOT / "configs",
        REPO_ROOT / "docs",
        REPO_ROOT / "openspec" / "specs",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in ACTIVE_TEXT_SUFFIXES:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[:2] == ("docs", "superpowers"):
                continue
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                if needle in text:
                    offenders.append(f"{rel}:{needle}")
    assert offenders == []


def test_active_docs_do_not_describe_stage2_runtime_as_shared_infer_runtime() -> None:
    forbidden_phrases = (
        "stage2_rollout_runtime.py` remains as an internal runtime base",
        "stage2_rollout_runtime.py` is an internal shared runtime base",
        "stage2_rollout_runtime.py` shares the refactored",
        "stage2_rollout_runtime.py` as the shared runtime",
    )
    offenders: list[str] = []
    for root in (REPO_ROOT / "docs", REPO_ROOT / "openspec" / "specs"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[:2] == ("docs", "superpowers"):
                continue
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden_phrases:
                if phrase in text:
                    offenders.append(f"{rel}:{phrase}")
    assert offenders == []


def test_infer_backend_modules_do_not_call_trainer_private_decode_resolver() -> None:
    roots = (REPO_ROOT / "src" / "infer",)
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "runtime.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "_resolve_rollout_decode_request" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_infer_modules_do_not_import_trainer_internals() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "src" / "infer").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from src.trainers" in text or "import src.trainers" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_infer_prompt_and_runtime_import_without_heavy_backend_dependencies() -> None:
    code = r'''
import builtins
blocked = {"torch", "transformers", "swift", "vllm"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked heavy dependency: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import src.infer.prompt
import src.infer.runtime
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_stage2_runtime_does_not_own_rollout_prompt_preparation_helpers() -> None:
    path = REPO_ROOT / "src" / "trainers" / "stage2_rollout_runtime.py"
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "build_dense_user_prompt",
        "build_dense_system_prompt",
        "strip_trailing_assistant_turns_for_rollout",
        "force_last_user_prompt_text",
        "ensure_system_prompt_message",
        "rollout_visual_metadata_from_sample",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []
