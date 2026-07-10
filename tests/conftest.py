import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep the canonical Swift test suite bound to this checkout. The former
# MS-Swift path injection belonged to the archived mainline test bootstrap and
# made the rebuilt suite depend on a sibling checkout.
root_str = str(ROOT)
sys.path = [root_str] + [
    p for p in sys.path if p != root_str and "Qwen3-VL" not in p
]


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


# Ensure this repository's src package and script entrypoints win over any
# unrelated editable packages already imported by the test runner.
_purge_modules("src")
_purge_modules("scripts")
