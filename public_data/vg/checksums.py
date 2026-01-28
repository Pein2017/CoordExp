"""Expected checksums for Visual Genome (VG) raw artifacts.

This module is used by `public_data/scripts/prepare_visual_genome.py` to verify
downloads for reproducibility.

Notes:
- We key by the *basename* of the downloaded file (e.g., "images.zip").
- If a file is missing from this table, the downloader will warn and skip
  checksum verification for that file.
"""

from __future__ import annotations

EXPECTED_SHA256: dict[str, str] = {
    # Annotations (UW mirror; referenced by the HF dataset loader)
    "image_data.json.zip": "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a",
    "objects_v1_2_0.json.zip": "92200e9eac2ae159d8af0bba6bceb2979f240295d3d83ed9c9eedd3f5f534d5f",
    "region_descriptions.json.zip": "038382219029c9cb0d7b0308b1d45f7537cf5a47b98f493f6d0e30c8947fb15d",
    # Images (Stanford mirror)
    "images.zip": "51c682d2721f880150720bb416e0346a4c787e4c55d7f80dfd1bd3f73ba81646",
    "images2.zip": "99da1a0ddf87011319ff3b05cf9176ffee2731cc3c52951162d9ef0d68e3cfb5",
}

