"""Compatibility surface for coord monitor helpers.

Historically these helpers lived under `src.trainers.metrics`.
The canonical import path is now `src.metrics.coord_monitors`.
"""

from src.trainers.metrics.coord_monitors import *  # noqa: F401,F403
