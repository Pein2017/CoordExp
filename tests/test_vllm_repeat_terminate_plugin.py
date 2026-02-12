from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vllm_repeat_terminate_plugin.py"
_SPEC = importlib.util.spec_from_file_location("_coordexp_vllm_repeat_plugin", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PLUGIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PLUGIN
_SPEC.loader.exec_module(_PLUGIN)


@dataclass
class _Inner:
    value: int


@dataclass
class _Outer:
    inner: _Inner


def test_as_plain_dict_supports_dataclass_response() -> None:
    payload = _Outer(inner=_Inner(value=7))
    out = _PLUGIN._as_plain_dict(payload)
    assert out == {"inner": {"value": 7}}
