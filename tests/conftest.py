import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Put this repo first and drop any upstream Qwen3-VL path that also has a src/
root_str = str(ROOT)
sys.path = [root_str] + [p for p in sys.path if p != root_str and "Qwen3-VL" not in p]

# Ensure we use this repo's src package even if another is already imported
src_mod = sys.modules.get("src")
expected_prefix = str(ROOT / "src")
if src_mod is not None:
    mod_file = getattr(src_mod, "__file__", "") or ""
    if not mod_file.startswith(expected_prefix):
        sys.modules.pop("src", None)
