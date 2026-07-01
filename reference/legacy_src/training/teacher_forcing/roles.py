from __future__ import annotations

from enum import Enum


class TokenRole(str, Enum):
    SCHEMA = "SCHEMA"
    TEXT = "TEXT"
    COORD = "COORD"
    STOP = "STOP"
