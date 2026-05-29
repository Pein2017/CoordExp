"""Dataset wrapper package.

The removed runtime-fusion registry no longer lives at package import time.
Import concrete wrappers from their modules, for example
``src.datasets.wrappers.packed_caption`` or
``src.datasets.wrappers.random_sample``.
"""

from __future__ import annotations

__all__: list[str] = []
