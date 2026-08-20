"""Live refusal proof for the historicized Attempt-6/8 packet executor.

The executor's frozen ``BASE_CONFIG_SHA256`` binds the pre-migration bytes of
the accelerate2_ebs2 smoke config (completed reconcile evidence, never
re-pinned). ``standardize-coordexp-swift-supervised-losses`` Wave 1 migrated
that supported config, so on the current tree the executor must fail closed —
``packet_executor.base_config_drift`` — before launching anything or
publishing a marker. The mechanism suite in
``test_reconcile_exact_resume_packet_executor.py`` is skipif-gated to trees
where the pinned bytes still exist; this module asserts the refusal that
replaces it on migrated trees.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ORIGINAL = Path(__file__).with_name("test_reconcile_exact_resume_packet_executor.py")


def _load_original() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reconcile_exact_resume_packet_executor_mechanism_suite", _ORIGINAL
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SUITE = _load_original()

pytestmark = pytest.mark.skipif(
    not _SUITE._LIVE_BASE_CONFIG_DRIFTED,
    reason=(
        "live base config still matches the frozen Attempt-6 pin; the full "
        "mechanism suite runs instead"
    ),
)


@pytest.fixture(scope="module")
def executor() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reconcile_exact_resume_packet_executor", _SUITE.SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_base_config_bytes_differ_from_the_frozen_attempt6_pin(
    executor: ModuleType,
) -> None:
    live = hashlib.sha256(_SUITE.BASE_CONFIG.read_bytes()).hexdigest()
    assert executor.BASE_CONFIG_SHA256 == _SUITE.BASE_CONFIG_SHA256
    assert live != executor.BASE_CONFIG_SHA256


@pytest.mark.parametrize("drift", ["intact_argv", "wrong_path", "missing_flag"])
def test_executor_refuses_migrated_live_config_before_launch_or_marker(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    case = _SUITE._fixture(tmp_path, monkeypatch)
    setup_argv = case["manifest"]["setup_command"]
    base_index = setup_argv.index("--base-config")
    if drift == "wrong_path":
        setup_argv[base_index + 1] = str(case["repo"] / "wrong-base.yaml")
    elif drift == "missing_flag":
        del setup_argv[base_index : base_index + 2]
    _SUITE._refresh_manifest_and_review(case)
    launched: list[list[str]] = []

    def launch(argv: list[str], **kwargs: Any) -> Any:
        launched.append(argv)
        return case["launch"](argv, **kwargs)

    with pytest.raises(executor.PacketExecutorError) as exc_info:
        _SUITE._execute(executor, case, launch=launch)

    # The byte gate precedes the setup-argv binding checks, so every variant —
    # including an intact argv — terminates at base_config_drift with zero
    # side effects.
    assert exc_info.value.code == "packet_executor.base_config_drift"
    assert launched == []
    assert not case["marker"].exists()
