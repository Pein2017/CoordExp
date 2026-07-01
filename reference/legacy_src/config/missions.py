#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Placeholder mission definitions for general/public detection.

Legacy BBU/RRU mission labels have been removed. Keep a minimal interface so
callers depending on mission validation do not break.
"""

from __future__ import annotations

from typing import Dict, List

# Single generic mission to keep downstream code functional
SUPPORTED_MISSIONS: List[str] = [
    "general_detection",
]

# Minimal focus hints (English, general-purpose)
STAGE_A_MISSION_FOCUS: Dict[str, str] = {
    "general_detection": "Detect and describe all visible objects with correct geometry.",
}

STAGE_B_MISSION_FOCUS: Dict[str, str] = {
    "general_detection": "Verify all required objects are detected with accurate geometry; no domain-specific rules.",
}


def validate_mission(mission: str) -> None:
    """Validate mission against supported values.

    Args:
        mission: Mission name to validate

    Raises:
        ValueError: If mission is not in SUPPORTED_MISSIONS
    """
    if mission not in SUPPORTED_MISSIONS:
        raise ValueError(
            f"Unsupported mission: {mission}. "
            f"Must be one of: {', '.join(SUPPORTED_MISSIONS)}"
        )


__all__ = [
    "SUPPORTED_MISSIONS",
    "STAGE_A_MISSION_FOCUS",
    "STAGE_B_MISSION_FOCUS",
    "validate_mission",
]
