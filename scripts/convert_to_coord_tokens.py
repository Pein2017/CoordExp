#!/usr/bin/env python3
"""
Shim to keep backward compatibility.
Delegates to public_data/scripts/convert_to_coord_tokens.py.
"""

from public_data.scripts.convert_to_coord_tokens import main

if __name__ == "__main__":
    main()
