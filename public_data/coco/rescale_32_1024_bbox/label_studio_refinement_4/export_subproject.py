#!/usr/bin/env python3
"""Publish the current five-image Label Studio annotations as GT."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
STATE_ROOT = (
    REPO_ROOT
    / "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000/label-studio/state"
)

os.environ["BASE_DATA_DIR"] = str(STATE_ROOT)
os.environ["LOCAL_FILES_DOCUMENT_ROOT"] = str(
    REPO_ROOT / "public_data/coco/rescale_32_1024_bbox/images"
)
os.environ["LOCAL_FILES_SERVING_ENABLED"] = "true"
os.environ["DJANGO_DB"] = "sqlite"
os.environ["DJANGO_SETTINGS_MODULE"] = "core.settings.label_studio"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "label-studio" / "label_studio"))

import django  # noqa: E402

django.setup()

from coordexp_refinement.gt_export import export_project_gt  # noqa: E402


if __name__ == "__main__":
    receipt = export_project_gt(3)
    if receipt is None:
        raise RuntimeError("Project 3 refinement manifest was not found")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
