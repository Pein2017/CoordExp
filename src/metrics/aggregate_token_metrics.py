"""Compatibility surface for aggregate token metrics helpers.

Historically these helpers lived under `src.trainers.metrics`.
The canonical import path is now `src.metrics.aggregate_token_metrics`.
"""

from src.trainers.metrics.aggregate_token_metrics import *  # noqa: F401,F403
