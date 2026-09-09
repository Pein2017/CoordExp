"""Named Qwen text sites and bounded snapshots of actual module calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class QwenTextStack:
    language_model: torch.nn.Module
    layers: tuple[torch.nn.Module, ...]
    norm: torch.nn.Module
    head: torch.nn.Module


def resolve_text_stack(model: Any) -> QwenTextStack:
    """Resolve one distinct text stack through actual model wrapper aliases.

    Scientific layer counts and intervention locations remain caller assertions.
    No layer class is substituted when the loaded model has an unknown layout.
    """
    owners: dict[int, Any] = {}
    seen: set[int] = set()
    current = model
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        for candidate in (current, getattr(current, "language_model", None)):
            layers = getattr(candidate, "layers", None)
            norm = getattr(candidate, "norm", None)
            if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and isinstance(
                norm, torch.nn.Module
            ):
                owners[id(candidate)] = candidate
        current = getattr(current, "model", None)
    if len(owners) != 1:
        raise ValueError(f"expected one distinct Qwen text stack, found {len(owners)}")
    owner = next(iter(owners.values()))
    layers = tuple(owner.layers)
    if not layers or not all(isinstance(layer, torch.nn.Module) for layer in layers):
        raise ValueError("text stack has no executable decoder layers")
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if not isinstance(head, torch.nn.Module):
        raise ValueError("model has no output embedding module")
    return QwenTextStack(owner, layers, owner.norm, head)


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # A captured tensor may later be an input to functional_call/autograd.
        with torch.inference_mode(False):
            return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


class CaptureInputs:
    """Snapshot one actual call's positional inputs and selected keyword inputs."""

    def __init__(
        self, module: torch.nn.Module, *, keys: Sequence[str] | None = None
    ) -> None:
        self.module = module
        self.keys = None if keys is None else tuple(keys)
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}
        self.calls = 0
        self._handle: Any = None

    def _hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("single-call input capture fired more than once")
        if self.keys is not None:
            missing = set(self.keys) - kwargs.keys()
            if missing:
                raise ValueError(
                    f"captured call is missing requested inputs: {sorted(missing)}"
                )
            kwargs = {key: kwargs[key] for key in self.keys}
        self.args = _snapshot(args)
        self.kwargs = _snapshot(kwargs)

    def __enter__(self) -> CaptureInputs:
        if self._handle is not None:
            raise RuntimeError("input capture is already installed")
        self.calls = 0
        self.args, self.kwargs = (), {}
        self._handle = self.module.register_forward_pre_hook(
            self._hook, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc_type is None and self.calls != 1:
            raise RuntimeError("input capture did not observe one call")


class CaptureHiddenRows:
    """Clone selected rows at an explicit input or output module boundary.

    Output snapshots are taken before a caller can mutate the returned tensor,
    including Qwen's in-place DeepStack injection. Repeated calls are bounded by
    the explicit max_calls; they are not silently overwritten.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        positions: Sequence[int],
        *,
        boundary: Literal["input", "output"] = "input",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_calls: int = 1,
    ) -> None:
        if boundary not in ("input", "output"):
            raise ValueError("capture boundary must be input or output")
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls <= 0
        ):
            raise ValueError("max_calls must be a positive integer")
        self.positions = tuple(positions)
        if not self.positions or any(
            isinstance(p, bool) or not isinstance(p, int) or p < 0
            for p in self.positions
        ):
            raise ValueError("capture positions must be nonempty nonnegative integers")
        self.module, self.boundary = module, boundary
        self.device, self.dtype, self.max_calls = device, dtype, max_calls
        self.captures: list[torch.Tensor] = []
        self._handle: Any = None

    @property
    def calls(self) -> int:
        return len(self.captures)

    @property
    def hidden(self) -> torch.Tensor:
        if len(self.captures) != 1:
            raise RuntimeError("hidden requires exactly one captured call")
        return self.captures[0]

    def _capture(self, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[0] != 1
        ):
            raise ValueError("hidden capture requires [1, tokens, hidden] tensor")
        if max(self.positions) >= value.shape[1]:
            raise ValueError("capture positions are outside hidden states")
        if len(self.captures) >= self.max_calls:
            raise RuntimeError("hidden capture exceeded its declared call bound")
        with torch.inference_mode(False):
            self.captures.append(
                value[0, list(self.positions)]
                .detach()
                .to(device=self.device, dtype=self.dtype)
                .clone()
            )

    def _pre_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self._capture(args[0] if args else kwargs.get("hidden_states"))

    def _post_hook(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        self._capture(output)

    def __enter__(self) -> CaptureHiddenRows:
        if self._handle is not None:
            raise RuntimeError("hidden capture is already installed")
        self.captures.clear()
        if self.boundary == "input":
            self._handle = self.module.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
        else:
            self._handle = self.module.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc_type is None and not self.captures:
            raise RuntimeError("hidden capture did not observe a call")
