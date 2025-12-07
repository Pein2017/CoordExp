import types
from dataclasses import dataclass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coord_tokens.template_adapter import apply_coord_template_adapter  # noqa: E402


class _DummyTemplate:
    def __init__(self):
        self.calls = 0

    def normalize_bbox(self, inputs):
        self.calls += 1
        inputs["touched"] = True


@dataclass
class _Cfg:
    enabled: bool = True
    skip_bbox_norm: bool = True


def test_template_adapter_skips_norm():
    tmpl = _DummyTemplate()
    cfg = _Cfg()
    apply_coord_template_adapter(tmpl, cfg)

    # After patch, normalize_bbox should be a no-op
    payload = {}
    tmpl.normalize_bbox(payload)
    assert getattr(tmpl, "_coord_tokens_skip_norm", False)
    assert payload == {}
    assert tmpl.calls == 0
