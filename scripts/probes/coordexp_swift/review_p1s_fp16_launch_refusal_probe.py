"""P1-2 probe: declared fp16 without an active GradScaler is refused at launch.

Task 2.3 of OpenSpec change `close-coordexp-swift-review-p1s`.

What this proves that the unit tests cannot: the launch-time refusal fires on a
REAL `accelerate.Accelerator` (not the bounded stand-in the unit tests use) and
on a REAL `torch.amp.GradScaler` (not a fake), through the real validation
entry `src.runtime.train_runtime.validate_accelerator_runtime` and through the
real `TrainRuntime` constructor that calls it.

The fp16-declared-without-a-scaler shape is produced by accelerate itself:
`Accelerator(mixed_precision="fp16", cpu=True)` is a genuine Accelerator that
reports `mixed_precision == "fp16"` and `scaler is None`. No stand-in accelerator
is used anywhere in this probe.

Cases:
  a) REAL Accelerator, fp16 declared, `scaler is None`
       -> RuntimeContractError `runtime.fp16_scaler_missing`
       -> also refused through `TrainRuntime(...)`, BEFORE `accelerator.prepare`
  b) REAL Accelerator, fp16 declared, REAL `torch.amp.GradScaler` whose
     `is_enabled()` is False
       -> the same typed refusal (a disabled scaler is not an active scaler)
  c) CONTROL: REAL Accelerator, bf16 declared, no scaler
       -> no refusal from this check; `TrainRuntime` constructs and prepares
  d) OVER-REFUSAL CONTROL (beyond the task): REAL Accelerator declaring fp16
     with a REAL enabled `torch.amp.GradScaler`
       -> no refusal; the runtime resolves that exact scaler object

CPU-only: no model runs, no CUDA tensor is allocated. Case (d) constructs a
`GradScaler("cuda", ...)` object, which does not touch the device.

Sensitivity: `--simulate-pre-fix` neutralizes the new gate in-process (the
launch-time check becomes the no-op it was before P1-2, no source edit), so the
probe can be OBSERVED to fail for the right reason. In that mode the probe's
own PASS/FAIL is inverted: a PASS would mean the probe is blind.

Usage:
    PYTHONDONTWRITEBYTECODE=1 python \
        scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py
    PYTHONDONTWRITEBYTECODE=1 python \
        scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py \
        --simulate-pre-fix
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.state import AcceleratorState  # noqa: E402

from src.common.errors import RuntimeContractError  # noqa: E402
from src.config.models import RuntimeBatchResolution, RuntimeConfig  # noqa: E402
from src.runtime.train_runtime import (  # noqa: E402
    TrainRuntime,
    validate_accelerator_runtime,
)

EXPECTED_CODE = "runtime.fp16_scaler_missing"

_VERDICTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: Any = "") -> None:
    detail_text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    _VERDICTS.append((name, bool(passed), detail_text))
    print(f"  {'PASS' if passed else 'FAIL'} {name} :: {detail_text}", flush=True)


def _fresh_accelerator(*, mixed_precision: str) -> Accelerator:
    """A REAL accelerate Accelerator, CPU-pinned, with its state reset first.

    `AcceleratorState._reset_state` is private API; it is the only way to build
    more than one Accelerator in a single process, and the alternative (one
    subprocess per case) would add nothing to what is being probed.
    """

    AcceleratorState._reset_state(True)
    return Accelerator(mixed_precision=mixed_precision, cpu=True)


def _count_prepare(accelerator: Accelerator) -> dict[str, int]:
    """Count `prepare` calls on the REAL accelerator without replacing it."""

    counter = {"calls": 0}
    original = accelerator.prepare

    def _counting_prepare(*args: Any, **kwargs: Any) -> Any:
        counter["calls"] += 1
        return original(*args, **kwargs)

    accelerator.prepare = _counting_prepare  # type: ignore[method-assign]
    return counter


def _train_runtime(accelerator: Accelerator, *, expected_mixed_precision: str) -> Any:
    return TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=int(accelerator.num_processes),
            effective_batch_size=int(accelerator.num_processes),
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1, bias=False),
        optimizer=None,
        scheduler=None,
        expected_mixed_precision=expected_mixed_precision,
        max_grad_norm=None,
        accelerator=accelerator,
        rank_report_gatherer=None,
    )


def _describe(accelerator: Accelerator) -> dict[str, Any]:
    scaler = getattr(accelerator, "scaler", None)
    return {
        "accelerator_type": f"{type(accelerator).__module__}.{type(accelerator).__name__}",
        "mixed_precision": str(accelerator.mixed_precision),
        "distributed_type": str(accelerator.distributed_type),
        "device": str(accelerator.device),
        "num_processes": int(accelerator.num_processes),
        "scaler_type": None if scaler is None else type(scaler).__name__,
        "scaler_module": None if scaler is None else type(scaler).__module__,
        "scaler_is_enabled": (
            None if scaler is None else bool(scaler.is_enabled())
        ),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_a_fp16_scaler_absent() -> None:
    print("\n[case a] REAL Accelerator, fp16 declared, scaler absent", flush=True)
    accelerator = _fresh_accelerator(mixed_precision="fp16")
    print(f"  accelerator={json.dumps(_describe(accelerator), sort_keys=True)}", flush=True)
    check(
        "a.accelerator_is_real_accelerate",
        type(accelerator).__module__.startswith("accelerate"),
        type(accelerator).__module__,
    )
    check("a.declares_fp16", str(accelerator.mixed_precision) == "fp16")
    check("a.scaler_is_absent", getattr(accelerator, "scaler", None) is None)

    raised: RuntimeContractError | None = None
    try:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    except RuntimeContractError as error:
        raised = error
    check("a.validate_accelerator_runtime_refuses", raised is not None)
    if raised is not None:
        print(f"  error={raised}", flush=True)
        check("a.code_is_expected", raised.code == EXPECTED_CODE, raised.code)
        check(
            "a.context_reports_absent_scaler",
            raised.context.get("scaler_present") is False
            and raised.context.get("scaler_enabled") is None
            and raised.context.get("mixed_precision") == "fp16",
            raised.context,
        )

    prepare_counter = _count_prepare(accelerator)
    runtime_raised: RuntimeContractError | None = None
    try:
        _train_runtime(accelerator, expected_mixed_precision="fp16")
    except RuntimeContractError as error:
        runtime_raised = error
    check("a.train_runtime_construction_refuses", runtime_raised is not None)
    if runtime_raised is not None:
        check(
            "a.train_runtime_code_is_expected",
            runtime_raised.code == EXPECTED_CODE,
            runtime_raised.code,
        )
    check(
        "a.refusal_precedes_accelerator_prepare",
        prepare_counter["calls"] == 0,
        prepare_counter,
    )


def case_b_fp16_real_disabled_scaler() -> None:
    print(
        "\n[case b] REAL Accelerator, fp16 declared, REAL disabled GradScaler",
        flush=True,
    )
    accelerator = _fresh_accelerator(mixed_precision="fp16")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # A REAL torch GradScaler, not a mock. This host HAS a usable CUDA
        # build, so the scaler does not self-disable; `enabled=False` is the
        # supported way to construct the same real object in its disabled
        # state (identical class, `is_enabled() is False`).
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    accelerator.scaler = scaler
    print(f"  accelerator={json.dumps(_describe(accelerator), sort_keys=True)}", flush=True)
    check(
        "b.scaler_is_a_real_torch_gradscaler",
        isinstance(scaler, torch.amp.GradScaler)
        and type(scaler).__module__.startswith("torch"),
        f"{type(scaler).__module__}.{type(scaler).__name__}",
    )
    check("b.scaler_is_disabled", scaler.is_enabled() is False)

    raised: RuntimeContractError | None = None
    try:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    except RuntimeContractError as error:
        raised = error
    check("b.validate_accelerator_runtime_refuses", raised is not None)
    if raised is not None:
        print(f"  error={raised}", flush=True)
        check("b.code_is_expected", raised.code == EXPECTED_CODE, raised.code)
        check(
            "b.context_reports_present_but_disabled_scaler",
            raised.context.get("scaler_present") is True
            and raised.context.get("scaler_enabled") is False
            and raised.context.get("scaler_type") == "GradScaler",
            raised.context,
        )

    prepare_counter = _count_prepare(accelerator)
    runtime_raised: RuntimeContractError | None = None
    try:
        _train_runtime(accelerator, expected_mixed_precision="fp16")
    except RuntimeContractError as error:
        runtime_raised = error
    check("b.train_runtime_construction_refuses", runtime_raised is not None)
    if runtime_raised is not None:
        check(
            "b.train_runtime_code_is_expected",
            runtime_raised.code == EXPECTED_CODE,
            runtime_raised.code,
        )
    check(
        "b.refusal_precedes_accelerator_prepare",
        prepare_counter["calls"] == 0,
        prepare_counter,
    )


def case_c_bf16_control() -> None:
    print("\n[case c] CONTROL: REAL Accelerator, bf16 declared, no scaler", flush=True)
    accelerator = _fresh_accelerator(mixed_precision="bf16")
    print(f"  accelerator={json.dumps(_describe(accelerator), sort_keys=True)}", flush=True)
    check("c.declares_bf16", str(accelerator.mixed_precision) == "bf16")
    check("c.scaler_is_absent", getattr(accelerator, "scaler", None) is None)

    refusal: str | None = None
    try:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="bf16")
    except RuntimeContractError as error:  # pragma: no cover - probe reports it
        refusal = f"{error.code}: {error}"
    check("c.validate_accelerator_runtime_accepts", refusal is None, refusal or "no refusal")

    prepare_counter = _count_prepare(accelerator)
    runtime: Any = None
    construction_error: str | None = None
    try:
        runtime = _train_runtime(accelerator, expected_mixed_precision="bf16")
    except Exception as error:  # pragma: no cover - probe reports it
        construction_error = f"{type(error).__name__}: {error}"
    check("c.train_runtime_constructs", runtime is not None, construction_error or "ok")
    if runtime is not None:
        check("c.declared_fp16_is_false", runtime.declared_fp16 is False)
        check("c.no_active_fp16_scaler", runtime._active_fp16_scaler() is None)
        check(
            "c.accelerator_prepare_was_reached",
            prepare_counter["calls"] == 1,
            prepare_counter,
        )


def case_d_fp16_real_enabled_scaler() -> None:
    print(
        "\n[case d] OVER-REFUSAL CONTROL: fp16 declared, REAL enabled GradScaler",
        flush=True,
    )
    accelerator = _fresh_accelerator(mixed_precision="fp16")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    accelerator.scaler = scaler
    print(f"  accelerator={json.dumps(_describe(accelerator), sort_keys=True)}", flush=True)
    check("d.scaler_is_enabled", scaler.is_enabled() is True)

    refusal: str | None = None
    try:
        validate_accelerator_runtime(accelerator, expected_mixed_precision="fp16")
    except RuntimeContractError as error:  # pragma: no cover - probe reports it
        refusal = f"{error.code}: {error}"
    check("d.validate_accelerator_runtime_accepts", refusal is None, refusal or "no refusal")

    prepare_counter = _count_prepare(accelerator)
    runtime: Any = None
    construction_error: str | None = None
    try:
        runtime = _train_runtime(accelerator, expected_mixed_precision="fp16")
    except Exception as error:  # pragma: no cover - probe reports it
        construction_error = f"{type(error).__name__}: {error}"
    check("d.train_runtime_constructs", runtime is not None, construction_error or "ok")
    if runtime is not None:
        check("d.declared_fp16_is_true", runtime.declared_fp16 is True)
        check("d.resolves_that_exact_scaler", runtime._active_fp16_scaler() is scaler)
        check(
            "d.accelerator_prepare_was_reached",
            prepare_counter["calls"] == 1,
            prepare_counter,
        )


def _install_pre_fix_no_gate() -> None:
    """Remove the P1-2 launch gate in-process (the pre-fix behaviour)."""

    import src.runtime.train_runtime as train_runtime_module

    def _no_gate(accelerator: Any, *, mixed_precision: str) -> None:
        return None

    train_runtime_module._validate_fp16_scaler_contract = _no_gate


def main() -> int:
    started = time.monotonic()
    simulate_pre_fix = "--simulate-pre-fix" in sys.argv[1:]
    print(
        f"[probe] torch={torch.__version__} cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    import accelerate

    print(f"[probe] accelerate={accelerate.__version__}", flush=True)
    print(f"[probe] simulate_pre_fix={simulate_pre_fix}", flush=True)
    if simulate_pre_fix:
        print(
            "[probe] SENSITIVITY RUN: the P1-2 launch gate is neutralized "
            "in-process; a FAIL below is the expected outcome and this run's "
            "process exit code is inverted.",
            flush=True,
        )
        _install_pre_fix_no_gate()
    for case in (
        case_a_fp16_scaler_absent,
        case_b_fp16_real_disabled_scaler,
        case_c_bf16_control,
        case_d_fp16_real_enabled_scaler,
    ):
        try:
            case()
        except Exception:  # noqa: BLE001 - a crashing case is a probe failure
            traceback.print_exc()
            check(f"{case.__name__}.completed", False, "case raised")
    AcceleratorState._reset_state(True)

    overall = all(passed for _, passed, _ in _VERDICTS)
    failed = [name for name, passed, _ in _VERDICTS if not passed]
    print("\n[probe] --- summary ---", flush=True)
    print(f"[probe] assertions={len(_VERDICTS)} failed={failed}", flush=True)
    print(f"[probe] VERDICT: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"[probe] wall_seconds={time.monotonic() - started:.2f}", flush=True)
    if simulate_pre_fix:
        print(
            "[probe] SENSITIVITY VERDICT: "
            + (
                "FAIL - the probe did NOT notice the missing launch gate"
                if overall
                else "PASS - the probe fails without the launch gate"
            ),
            flush=True,
        )
        return 1 if overall else 0
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
