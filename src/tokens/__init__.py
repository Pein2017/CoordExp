"""Role-aware token utilities.

New token functionality should live under this package. The older
``src.coord_tokens`` namespace remains as a compatibility layer during the
compact-sequence pilot.
"""

from .roles import TokenRole, TokenRoleSets

__all__ = ["TokenRole", "TokenRoleSets"]
