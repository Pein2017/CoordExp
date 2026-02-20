"""Strict, schema-derived parsing helpers for YAML-first configs.

This module provides a small strict parser that derives accepted keys from typed
`@dataclass` contracts.

Design goals (per OpenSpec change 2026-02-15):
- Fail fast on unknown keys.
- Report unknown keys with full dotted paths (including list indices).
- Support nested dataclass structures and list-of-dataclasses trees.

Non-goals:
- Rich value coercion (e.g., str->int) or semantic/range validation. Those should
  live in section-specific validators / `__post_init__`.
"""

from __future__ import annotations

import collections.abc
import types
from dataclasses import asdict, fields, is_dataclass
from functools import lru_cache
from typing import Any, Mapping, MutableMapping, TypeVar, Union, get_args, get_origin, get_type_hints


T = TypeVar("T")


def _join_path(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _index_path(parent: str, index: int) -> str:
    return f"{parent}[{int(index)}]"


def dataclass_asdict_no_none(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a dict, dropping keys whose value is None.

    This is useful when converting schema-parsed dataclasses back into the
    historical mapping form expected by downstream consumers.

    Notes:
    - Nested dataclasses are converted recursively.
    - Lists preserve element order (None elements are kept).
    """

    if not is_dataclass(obj):
        raise TypeError(f"dataclass_asdict_no_none expects a dataclass instance, got {type(obj)!r}")

    def _drop_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _drop_none(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_drop_none(v) for v in value]
        return value

    return _drop_none(asdict(obj))


@lru_cache(maxsize=None)
def _resolved_type_hints(schema_type: type) -> dict[str, Any]:
    try:
        return get_type_hints(schema_type, include_extras=True)
    except (NameError, TypeError, ValueError, AttributeError):
        # Fall back to dataclass field.type when annotations cannot be resolved.
        return {}


def _unwrap_annotated(expected_type: Any) -> Any:
    # typing.Annotated[T, ...] -> T
    origin = get_origin(expected_type)
    if origin is None:
        return expected_type
    if str(origin) == "typing.Annotated":
        args = get_args(expected_type)
        return args[0] if args else Any
    return expected_type


def _unwrap_optional(expected_type: Any) -> tuple[bool, Any]:
    """Return (is_optional, inner_type)."""

    expected_type = _unwrap_annotated(expected_type)
    origin = get_origin(expected_type)
    if origin is None:
        return False, expected_type

    # Optional[T] is Union[T, NoneType]
    if origin in {Union, types.UnionType}:
        args = get_args(expected_type)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) + 1 == len(args):
            inner = non_none[0] if len(non_none) == 1 else expected_type
            return True, inner

    return False, expected_type


def _parse_value(expected_type: Any, value: Any, *, path: str) -> Any:
    expected_type = _unwrap_annotated(expected_type)

    is_optional, inner = _unwrap_optional(expected_type)
    if value is None:
        if is_optional:
            return None
        # Let section-specific validators decide whether None is acceptable.
        return None

    expected_type = inner
    origin = get_origin(expected_type)

    if expected_type is Any or expected_type is object:
        return value

    # Nested dataclass.
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return parse_dataclass_strict(expected_type, value, path=path)

    # list[T]
    if origin in {list, tuple}:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a list")
        (item_type,) = get_args(expected_type)[:1] or (Any,)
        out: list[Any] = []
        for i, item in enumerate(list(value)):
            out.append(
                _parse_value(item_type, item, path=_index_path(path, int(i)))
            )
        return out

    # Mapping[K, V] / dict
    if origin is dict or (
        origin is not None
        and isinstance(origin, type)
        and issubclass(origin, collections.abc.Mapping)
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        return dict(value)

    if expected_type in {dict, Mapping, MutableMapping}:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        return dict(value)

    # Primitive / unknown typing surface: do not coerce or over-validate.
    return value


def parse_dataclass_strict(schema_type: type[T], payload: Any, *, path: str) -> T:
    """Parse `payload` into `schema_type`, rejecting unknown keys.

    Unknown key errors include full dotted paths. For list elements, nested paths
    use the format: `parent.list_field[<index>].<key>`.
    """

    if not isinstance(schema_type, type) or not is_dataclass(schema_type):
        raise TypeError(
            f"schema_type must be a dataclass type, got {schema_type!r}"
        )

    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must be a mapping")

    raw: dict[Any, Any] = dict(payload)

    allowed = {f.name for f in fields(schema_type)}
    unknown = [
        k for k in raw.keys() if (not isinstance(k, str)) or (k not in allowed)
    ]
    if unknown:
        rendered = [_join_path(path, str(k)) for k in sorted(unknown, key=lambda x: str(x))]
        raise ValueError(f"Unknown keys: {rendered}")

    hints = _resolved_type_hints(schema_type)

    kwargs: dict[str, Any] = {}
    for f in fields(schema_type):
        if f.name not in raw:
            continue
        expected = hints.get(f.name, f.type)
        kwargs[f.name] = _parse_value(expected, raw[f.name], path=_join_path(path, f.name))

    try:
        return schema_type(**kwargs)
    except TypeError as exc:
        raise TypeError(f"Failed to parse {path}: {exc}") from exc
