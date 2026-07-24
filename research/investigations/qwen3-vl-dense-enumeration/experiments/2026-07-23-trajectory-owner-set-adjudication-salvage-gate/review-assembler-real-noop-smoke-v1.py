from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

from scripts.research.assemble_constant_dose_breadth_state_banks import (
    load_v2_b16_panel_adapter,
)
from scripts.research import analyze_trajectory_owner_set_admission_census as census
from scripts.research import assemble_trajectory_owner_set_adjudication_review as review


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


image_ids = ["2434", "831"]
adapter = load_v2_b16_panel_adapter(
    sampled_panel_root=census.FROZEN_SAMPLED_ROOT,
    source_b16_root=census.FROZEN_SOURCE_ROOT,
    candidate_pool=census.FROZEN_CANDIDATE_POOL_PATH,
    semantic_image_ids=image_ids,
)
frozen = review.load_frozen_census_records(
    census_path=review.FROZEN_CENSUS_PATH,
    expected_census_path=review.FROZEN_CENSUS_PATH,
    expected_census_sha256=review.FROZEN_CENSUS_SHA256,
    selected_image_ids=image_ids,
)
images: dict[str, object] = {}
for image_id in image_ids:
    candidates = census._image_candidates(image_id, adapter, reverse_input=False)
    reconstructed = census._analyze_image_candidates(image_id, candidates)
    frozen_record = frozen.records[image_id]
    reconstructed_candidates = reconstructed["candidates"]
    frozen_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in frozen_record["candidates"]
    }
    routes = []
    for candidate in reconstructed_candidates:
        route_id = str(candidate["candidate_id"])
        frozen_candidate = frozen_candidates[route_id]
        reconstructed_sha = sha256_bytes(
            review.canonical_json_text(candidate).encode("utf-8")
        )
        frozen_sha = sha256_bytes(
            review.canonical_json_text(frozen_candidate).encode("utf-8")
        )
        routes.append(
            {
                "route_id": route_id,
                "reconstructed_candidate_sha256": reconstructed_sha,
                "frozen_candidate_sha256": frozen_sha,
                "exact_reconstructed_row_equal": candidate == frozen_candidate,
            }
        )
    geometry_untrusted = [
        str(candidate["candidate_id"])
        for candidate in reconstructed_candidates
        if "geometry_untrusted" in candidate["exclusion_reasons"]
    ]
    reconstructed_sha = sha256_bytes(
        review.canonical_json_text(reconstructed).encode("utf-8")
    )
    frozen_sha = sha256_bytes(
        review.canonical_json_text(frozen_record).encode("utf-8")
    )
    images[image_id] = {
        "candidate_count": len(reconstructed_candidates),
        "ordered_route_ids": [
            str(candidate["candidate_id"])
            for candidate in reconstructed_candidates
        ],
        "routes": routes,
        "all_17_reconstructed_route_rows_equal": all(
            route["exact_reconstructed_row_equal"] for route in routes
        ),
        "reconstructed_census_record_sha256": reconstructed_sha,
        "frozen_census_record_sha256": frozen_sha,
        "exact_reconstructed_census_record_equal": reconstructed == frozen_record,
        "primary_natural_alias_admitted": reconstructed["admission"][
            "primary_natural_alias_admitted"
        ],
        "eligible_candidate_count": reconstructed["eligible_candidate_count"],
        "excluded_candidate_count": reconstructed["excluded_candidate_count"],
        "geometry_untrusted_candidate_count": len(geometry_untrusted),
        "geometry_untrusted_candidate_ids": geometry_untrusted,
        "geometry_untrusted_row_count": sum(
            int(candidate["geometry"]["untrusted_row_count"])
            for candidate in reconstructed_candidates
        ),
    }

expected_routes = list(review.EXPECTED_ROUTE_IDS)
assert all(
    image["candidate_count"] == 17
    and image["ordered_route_ids"] == expected_routes
    and image["all_17_reconstructed_route_rows_equal"] is True
    and image["exact_reconstructed_census_record_equal"] is True
    for image in images.values()
)
assert images["2434"]["primary_natural_alias_admitted"] is True
assert images["2434"]["geometry_untrusted_candidate_count"] == 3
assert images["831"]["primary_natural_alias_admitted"] is False
assert images["831"]["geometry_untrusted_candidate_count"] == 16

repository_root = Path(review.REPOSITORY_ROOT)
candidate_pool_binding = file_binding(census.FROZEN_CANDIDATE_POOL_PATH)
census_binding = file_binding(review.FROZEN_CENSUS_PATH)
receipt = {
    "schema_version": "trajectory_owner_set_review.real_noop_smoke.v1",
    "terminal_status": "passed",
    "mode": "read_only_no_op_adapter_census_reconstruction",
    "environment": {
        "conda_environment": "ms",
        "python_dont_write_bytecode": True,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pythonpath": str(repository_root),
    },
    "command": (
        "PYTHONDONTWRITEBYTECODE=1 "
        "PYTHONPATH=/data/CoordExp/.worktrees/research-probes "
        "conda run -n ms python /tmp/successor_review_real_smoke.py"
    ),
    "probe_script": file_binding(Path(__file__)),
    "code_bindings": {
        "review_assembler": file_binding(Path(review.__file__)),
        "review_assembler_tests": file_binding(
            repository_root
            / "tests/research/test_assemble_trajectory_owner_set_adjudication_review.py"
        ),
        "reviewer_instruction_packet": file_binding(
            repository_root
            / "research/investigations/qwen3-vl-dense-enumeration/experiments/"
            "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/"
            "reviewer-instruction-packet-v1.md"
        ),
        "frozen_selector": file_binding(review.FROZEN_SELECTOR_PATH),
        "frozen_selector_tests": file_binding(
            repository_root
            / "tests/research/"
            "test_select_trajectory_owner_set_adjudication_review_sample.py"
        ),
        "census_analyzer": file_binding(Path(census.__file__)),
        "panel_adapter": file_binding(
            repository_root
            / "scripts/research/assemble_constant_dose_breadth_state_banks.py"
        ),
        "global_matcher": file_binding(
            repository_root
            / "scripts/research/analyze_individual_trajectory_union_support.py"
        ),
    },
    "input_bindings": {
        "candidate_pool": candidate_pool_binding,
        "census": census_binding,
        "sampled_panel": {
            "path": str(census.FROZEN_SAMPLED_ROOT.resolve(strict=True)),
            "manifest_set_sha256": census.FROZEN_SAMPLED_MANIFEST_SET_SHA256,
        },
        "source_panel": {
            "path": str(census.FROZEN_SOURCE_ROOT.resolve(strict=True)),
            "manifest_set_sha256": census.FROZEN_SOURCE_MANIFEST_SET_SHA256,
        },
    },
    "ordered_image_ids": image_ids,
    "expected_route_ids": expected_routes,
    "images": images,
    "checks": {
        "candidate_pool_hash_matches_frozen": (
            candidate_pool_binding["sha256"]
            == review.FROZEN_CANDIDATE_POOL_SHA256
        ),
        "census_hash_matches_frozen": (
            census_binding["sha256"] == review.FROZEN_CENSUS_SHA256
        ),
        "both_images_have_exact_17_route_reconstruction": True,
        "both_images_have_exact_census_reconstruction": True,
        "image_2434_is_admitted": True,
        "image_831_retains_16_geometry_exclusions": True,
    },
}
assert all(receipt["checks"].values())
print(json.dumps(receipt, indent=2, sort_keys=True) + "\n", end="")
