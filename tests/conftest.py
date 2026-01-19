import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Put this repo first and drop any upstream Qwen3-VL path that also has a src/
root_str = str(ROOT)
sys.path = [root_str] + [p for p in sys.path if p != root_str and "Qwen3-VL" not in p]

# Ensure ms-swift is importable as `swift` in unit tests.
# This repo may be checked out as a git worktree under `.worktrees/`, so we
# search upward for a sibling `ms-swift` directory instead of assuming a fixed
# relative layout.
ms_swift_root = None
for base in [ROOT.parent, *ROOT.parents]:
    candidate = (base / "ms-swift").resolve()
    if candidate.is_dir():
        ms_swift_root = candidate
        break

if ms_swift_root is not None:
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

# Proactively import ms-swift to avoid later tests accidentally shadowing the
# `swift` package with a non-package module.
try:
    import swift  # noqa: F401
except Exception as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "Unit tests require ms-swift to be importable as the `swift` package."
    ) from exc

if not getattr(sys.modules.get("swift"), "__path__", None):  # pragma: no cover - guard
    raise RuntimeError(
        "Resolved `swift` is not a package; ensure ms-swift is first on sys.path."
    )

# Ensure key submodules are loaded early so that other tests that use
# `sys.modules.setdefault(\"swift.llm\", ...)` for minimal stubs do not shadow the
# real ms-swift modules for the rest of the session.
try:
    import swift.llm  # noqa: F401
    import swift.llm.argument  # noqa: F401
except Exception:  # pragma: no cover - environment guard
    pass
