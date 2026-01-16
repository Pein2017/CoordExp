import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Put this repo first and drop any upstream Qwen3-VL path that also has a src/
root_str = str(ROOT)
sys.path = [root_str] + [p for p in sys.path if p != root_str and "Qwen3-VL" not in p]

# Ensure ms-swift is importable as `swift` in unit tests.
# In this workspace layout, CoordExp and ms-swift live under the same parent dir.
ms_swift_root = (ROOT.parent / "ms-swift").resolve()
if ms_swift_root.is_dir():
    ms_swift_str = str(ms_swift_root)
    # Force ms-swift to be early in sys.path even if it's already present via
    # easy-install.pth (editable install). This prevents accidental resolution
    # to a shadowing `swift` module that is not a package.
    sys.path = [p for p in sys.path if p != ms_swift_str]
    # Keep repo root at position 0; insert ms-swift immediately after.
    sys.path.insert(1, ms_swift_str)

def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


# Ensure `swift` resolves to ms-swift's package, not a shadowing non-package
# module that would break `import swift.trainers` during test collection.
_purge_modules("swift")

# Ensure we use this repo's src package even if another is already imported
_purge_modules("src")
