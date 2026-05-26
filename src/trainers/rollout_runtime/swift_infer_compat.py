from __future__ import annotations

from typing import Any, Tuple


def import_swift_request_config() -> Any:
    """Import ms-swift RequestConfig across pre/post infer_engine API layouts."""

    errors: list[BaseException] = []
    for module_name in (
        "swift.infer_engine",
        "swift.infer_engine.protocol",
        "swift.llm",
    ):
        try:
            module = __import__(module_name, fromlist=("RequestConfig",))
            return getattr(module, "RequestConfig")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift RequestConfig is required for vLLM rollouts; tried "
        "swift.infer_engine.RequestConfig, swift.infer_engine.protocol.RequestConfig, "
        "and swift.llm.RequestConfig"
    ) from errors[-1]


def import_swift_infer_request_and_config() -> Tuple[Any, Any]:
    """Import ms-swift InferRequest/RequestConfig across infer API layouts."""

    errors: list[BaseException] = []
    for module_name in (
        "swift.infer_engine",
        "swift.infer_engine.protocol",
        "swift.llm",
    ):
        try:
            module = __import__(
                module_name,
                fromlist=("InferRequest", "RequestConfig"),
            )
            return getattr(module, "InferRequest"), getattr(module, "RequestConfig")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift InferRequest and RequestConfig are required for vLLM rollouts; "
        "tried swift.infer_engine, swift.infer_engine.protocol, and swift.llm"
    ) from errors[-1]


def import_swift_to_device() -> Any:
    """Import ms-swift to_device across utility API layouts."""

    errors: list[BaseException] = []
    for module_name in ("swift.utils", "swift.utils.torch_utils", "swift.llm"):
        try:
            module = __import__(module_name, fromlist=("to_device",))
            return getattr(module, "to_device")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift to_device is required for Stage2 collation; tried "
        "swift.utils.to_device, swift.utils.torch_utils.to_device, and swift.llm.to_device"
    ) from errors[-1]
