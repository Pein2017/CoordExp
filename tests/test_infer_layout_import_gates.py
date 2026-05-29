from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap


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
            if rel == Path("tests/test_infer_layout_import_gates.py"):
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
            if rel == Path("tests/test_infer_layout_import_gates.py"):
                continue
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden_phrases:
                if phrase in text:
                    offenders.append(f"{rel}:{phrase}")
    assert offenders == []


def test_stage2_runtime_path_mentions_are_qualified_outside_current_routing() -> None:
    needles = ("stage2_rollout_runtime", "src/trainers/stage2_rollout_runtime.py")
    allowed_context = (
        "trainer-owned",
        "facade",
        "retired",
        "historical",
        "removed",
        "fail fast",
        "variant",
        "test",
        "search",
    )
    skipped_files = {
        Path("openspec/specs/rollout-matching-sft/spec.md"),
    }
    offenders: list[str] = []
    for root in (REPO_ROOT / "docs", REPO_ROOT / "openspec" / "specs"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[:2] == ("docs", "superpowers") or rel in skipped_files:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                lowered = line.lower()
                if not any(needle in lowered for needle in needles):
                    continue
                if not any(context in lowered for context in allowed_context):
                    offenders.append(f"{rel}:{line_no}:{line.strip()}")
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


def test_infer_owner_coupling_is_confined_to_designated_adapters() -> None:
    adapter_files = {
        Path("src/infer/artifacts.py"),
        Path("src/infer/backend.py"),
        Path("src/infer/backend_vllm_config.py"),
        Path("src/infer/backend_vllm_engine.py"),
        Path("src/infer/backend_vllm_infer.py"),
        Path("src/infer/backend_vllm_server.py"),
        Path("src/infer/prompt.py"),
        Path("src/infer/rollout_dispatch.py"),
        Path("src/infer/runtime.py"),
    }
    owner_needles = (
        "owner: Any",
        "owner._",
        "getattr(owner",
        "rollout_matching_cfg = getattr(owner",
    )
    offenders: list[str] = []
    for path in (REPO_ROOT / "src" / "infer").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        if not any(needle in text for needle in owner_needles):
            continue
        if rel not in adapter_files:
            offenders.append(str(rel))

    assert offenders == []


def test_decode_and_artifact_cores_consume_resolved_facts() -> None:
    from src.infer.backend import (
        rollout_many_hf_traced_with_handles,
        rollout_many_hf_with_handles,
    )
    from src.infer.artifacts import (
        build_infer_resolved_meta_from_facts,
        build_infer_summary_payload_from_facts,
    )
    from src.infer.backend_vllm_infer import rollout_many_vllm_colocate_with_handles
    from src.infer.backend_vllm_server import (
        dispatch_vllm_server_rounds_with_handles,
        infer_on_vllm_server_slice_with_handles,
        prepare_vllm_server_rollout_with_handles,
        rollout_many_vllm_server_with_handles,
    )
    from src.infer.rollout_dispatch import (
        rollout_many_traced_with_handles,
        rollout_many_with_handles,
    )
    from src.infer.runtime import build_decode_request_from_rollout_facts

    def _uses_owner_or_getattr(fn) -> bool:  # type: ignore[no-untyped-def]
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "owner":
                return True
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "getattr":
                    first_arg = node.args[0] if node.args else None
                    if isinstance(first_arg, ast.Name) and first_arg.id == "owner":
                        return True
        return False

    offenders: list[str] = []
    for fn in (
        rollout_many_hf_with_handles,
        rollout_many_hf_traced_with_handles,
        rollout_many_vllm_colocate_with_handles,
        prepare_vllm_server_rollout_with_handles,
        infer_on_vllm_server_slice_with_handles,
        dispatch_vllm_server_rounds_with_handles,
        rollout_many_vllm_server_with_handles,
        rollout_many_with_handles,
        rollout_many_traced_with_handles,
        build_decode_request_from_rollout_facts,
        build_infer_resolved_meta_from_facts,
        build_infer_summary_payload_from_facts,
    ):
        if _uses_owner_or_getattr(fn):
            offenders.append(fn.__name__)
    assert offenders == []


def test_stage2_eval_score_provenance_uses_shared_sidecar_writer() -> None:
    evaluator = REPO_ROOT / "src" / "trainers" / "rollout_aligned_evaluator.py"
    text = evaluator.read_text(encoding="utf-8")

    assert "from src.infer.artifacts import write_score_provenance_sidecar" in text
    assert "write_score_provenance_sidecar(" in text
    assert "scored_path=scored_path" in text


def test_dead_rollout_matching_manifest_family_branch_is_absent() -> None:
    roots = (
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "docs",
        REPO_ROOT / "openspec" / "specs",
    )
    needles = (
        'manifest_family == "rollout_matching"',
        "manifest_family == 'rollout_matching'",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in ACTIVE_TEXT_SUFFIXES:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel == Path("tests/test_infer_layout_import_gates.py"):
                continue
            if rel.parts[:2] == ("docs", "superpowers"):
                continue
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                if needle in text:
                    offenders.append(f"{rel}:{needle}")
    assert offenders == []


def test_current_specs_do_not_have_archive_purpose_placeholders() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "openspec" / "specs").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if "TBD - created by archiving" in text or "Update Purpose after archive" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_retired_stage2_specs_are_not_current_routing_authority() -> None:
    retired_needles = (
        "stage2-ab-training",
        "rollout-matching-sft",
        "channel-b-lightweight-pseudopositive",
    )
    allowed_context = ("retired", "historical", "rejection")
    offenders: list[str] = []
    for root in (REPO_ROOT / "docs", REPO_ROOT / "openspec" / "specs"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[:2] == ("docs", "superpowers"):
                continue
            if rel == Path("openspec/specs/rollout-matching-sft/spec.md"):
                continue
            if rel == Path("openspec/specs/stage2-ab-training/spec.md"):
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                lowered = line.lower()
                if not any(needle in lowered for needle in retired_needles):
                    continue
                if not any(context in lowered for context in allowed_context):
                    offenders.append(f"{rel}:{line_no}:{line.strip()}")
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
