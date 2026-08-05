"""Contract tests for the neutral-row control CPU plan builder.

The fixture is a complete synthetic predecessor tree -- geometry analysis,
crossing plan, sealed secondary capture and census plan registries -- wired with
real digests, so the whole fail-closed chain and the frozen 9/12/5 and 21/21/12
denominators are exercised end to end without touching a production artifact.

The synthetic owners deliberately carry the *frozen* owner and image ids: the
plan re-derives and enforces the named infeasibility ledger and clearly
separated subset, so a fixture with invented ids could not reach them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import analyze_sorted_crossing_owner_row_geometry as geometry
from scripts.research import (
    prepare_sorted_crossing_boundary_owner_release_realization as crossing_plan,
)
from scripts.research import prepare_sorted_crossing_neutral_row_control as sut
from scripts.research import score_sorted_crossing_boundary_owner_release as crossing_scorer

OBJ_START = crossing_scorer.OBJECT_REF_START
OBJ_END = crossing_scorer.OBJECT_REF_END
BOX_START = crossing_scorer.BOX_START
BOX_END = crossing_scorer.BOX_END
COORD_START = crossing_scorer.COORDINATE_TOKEN_ID_START

#: The frozen twelve-image panel and its 26 crossing owners.
CROSSING_OWNERS: tuple[tuple[str, str], ...] = (
    ("10707", "gt:10707:16"),
    ("13348", "gt:13348:7"),
    ("13923", "gt:13923:1"),
    ("13923", "gt:13923:11"),
    ("13923", "gt:13923:14"),
    ("14038", "gt:14038:10"),
    ("14038", "gt:14038:19"),
    ("14038", "gt:14038:23"),
    ("14439", "gt:14439:21"),
    ("14439", "gt:14439:3"),
    ("1584", "gt:1584:8"),
    ("16228", "gt:16228:11"),
    ("16228", "gt:16228:3"),
    ("16228", "gt:16228:38"),
    ("16228", "gt:16228:44"),
    ("16228", "gt:16228:47"),
    ("2685", "gt:2685:20"),
    ("4134", "gt:4134:22"),
    ("4134", "gt:4134:27"),
    ("4134", "gt:4134:29"),
    ("4134", "gt:4134:32"),
    ("4134", "gt:4134:34"),
    ("5001", "gt:5001:10"),
    ("5001", "gt:5001:17"),
    ("6040", "gt:6040:13"),
    ("7511", "gt:7511:1"),
)
IMAGE_IDS: tuple[str, ...] = tuple(dict.fromkeys(image for image, _owner in CROSSING_OWNERS))

MATERIAL_OWNERS: frozenset[str] = frozenset(
    {
        "gt:13348:7",
        "gt:13923:14",
        "gt:14038:19",
        "gt:1584:8",
        "gt:16228:11",
        "gt:16228:38",
        "gt:4134:27",
        "gt:4134:29",
        "gt:6040:13",
    }
)
INFEASIBLE_OWNERS: frozenset[str] = frozenset(sut.FROZEN_INFEASIBLE_OWNER_IDS)
SEPARATED_OWNERS: frozenset[str] = frozenset(sut.FROZEN_CLEARLY_SEPARATED_OWNER_IDS)

#: Candidate owners per image, as ``(suffix, category token count, box index)``.
#: One candidate is enough for image 6040 (exercises ``only_one_candidate``);
#: image 4134 gets three so the length-then-row-index order is exercised.
CANDIDATE_SPECS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "10707": (("n0", 2, 60), ("n1", 1, 61), ("overlap", 2, 20)),
    "13348": (("n0", 2, 60), ("n1", 3, 61)),
    "13923": (("n0", 2, 60), ("n1", 1, 61)),
    "14038": (("n0", 2, 60), ("n1", 5, 61)),
    "14439": (("n0", 2, 60), ("n1", 3, 61)),
    "1584": (("n0", 2, 60), ("n1", 1, 61)),
    "16228": (("n0", 2, 60), ("n1", 3, 61)),
    "2685": (("n0", 2, 60), ("n1", 1, 61), ("excluded", 2, 62)),
    "4134": (("n0", 2, 60), ("n1", 2, 61), ("n2", 4, 62)),
    "5001": (("n0", 2, 60), ("n1", 3, 61)),
    "6040": (("n0", 2, 60),),
    "7511": (("n0", 2, 60), ("n1", 3, 61)),
}
#: Candidate suffixes that must never be admitted, and why.
OVERLAPPING_CANDIDATE = "overlap"
EXCLUDED_CANDIDATE = "excluded"
#: ``gt:13348:7``'s downstream row strict-matches its own image's ``n1``, which
#: must therefore be refused by the "not the strict-matched owner of E" predicate.
E_OWNER_IS_CANDIDATE_TARGET = "gt:13348:7"


def _sha256_json(value: Any) -> str:
    return geometry.sha256_json(value)


def _box(index: int) -> tuple[int, int, int, int]:
    base = index * 12
    return (base, base, base + 8, base + 8)


def _coords(index: int) -> list[int]:
    return [COORD_START + value for value in _box(index)]


def _tokens(text: str, count: int) -> list[int]:
    seed = sum(ord(char) for char in text)
    return [20000 + (seed * (offset + 3)) % 900 for offset in range(count)]


def _row_tokens(description_tokens: list[int], coord_tokens: list[int]) -> list[int]:
    return [OBJ_START, *description_tokens, OBJ_END, BOX_START, *coord_tokens, BOX_END]


def _query_suffix(description_tokens: list[int]) -> list[int]:
    return [OBJ_START, *description_tokens, OBJ_END, BOX_START]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(geometry.canonical_json_bytes(row) + b"\n" for row in rows))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(geometry.canonical_json_bytes(payload) + b"\n")


def _seal(payload: dict[str, Any], key: str) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[key] = _sha256_json(sealed)
    return sealed


def _file_seal(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "byte_size": len(payload),
        "sha256": geometry.sha256_bytes(payload),
    }


# ---------------------------------------------------------------------------
# The synthetic sealed tree
# ---------------------------------------------------------------------------


class _Tree:
    """A complete synthetic predecessor tree with real digests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.census_dir = root / "census" / "plan"
        self.crossing_dir = root / "crossing" / "plan"
        self.merged_dir = root / "capture" / "secondary-merged"
        self.geometry_dir = root / "geometry" / "geometry-analysis"
        for directory in (
            self.census_dir,
            self.crossing_dir,
            self.merged_dir,
            self.geometry_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._build()

    # -- image layout ---------------------------------------------------
    def _owners_of(self, image_id: str) -> list[str]:
        return [owner for image, owner in CROSSING_OWNERS if image == image_id]

    def _boundary_of(self, image_id: str, gt_owner_id: str) -> int:
        return 5 + 2 * self._owners_of(image_id).index(gt_owner_id)

    def _c_description(self, gt_owner_id: str) -> str:
        if gt_owner_id in INFEASIBLE_OWNERS:
            return f"cand-{gt_owner_id.split(':')[1]}-n0"
        return f"targ-{gt_owner_id.replace(':', '-')}"

    def _e_description(self, image_id: str, gt_owner_id: str) -> str:
        if gt_owner_id in INFEASIBLE_OWNERS:
            return f"cand-{image_id}-n1"
        return "erow"

    def _e_owner(self, gt_owner_id: str) -> str | None:
        if gt_owner_id == E_OWNER_IS_CANDIDATE_TARGET:
            return "gt:13348:n1"
        return None

    def _candidates(self, image_id: str) -> tuple[tuple[str, int, int], ...]:
        return CANDIDATE_SPECS[image_id]

    def _build(self) -> None:
        self._build_census()
        self._build_crossing()
        self._build_capture()
        self._build_geometry()

    # -- census ---------------------------------------------------------
    def _build_census(self) -> None:
        owners: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        categories: dict[str, dict[str, Any]] = {}
        bank: list[dict[str, Any]] = []
        sidecars: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        self.rows_by_image: dict[str, list[list[int]]] = {}
        self.box_index: dict[str, int] = {}

        def _register_owner(
            gt_owner_id: str, image_id: str, description: str, box_index: int, *, excluded: bool
        ) -> None:
            self.box_index[gt_owner_id] = box_index
            owners.append(
                {
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "normalized_description": description,
                    "excluded_from_census": excluded,
                    "bbox_pixel_xyxy": list(_box(box_index)),
                    "owner_sort_key": [box_index, box_index],
                    "official_coco_category_id": 1,
                    "native_true_positive": False,
                    "row_kind": "census_owner",
                }
            )
            key = f"{image_id}:{description}"
            if key not in categories:
                tokens = _tokens(description, self.description_token_count[description])
                suffix = _query_suffix(tokens)
                categories[key] = {
                    "category_query_id": key,
                    "image_id": image_id,
                    "normalized_description": description,
                    "status": "admitted",
                    "category_token_ids": tokens,
                    "category_token_ids_sha256": _sha256_json(tokens),
                    "query_suffix_token_ids": suffix,
                    "query_suffix_token_ids_sha256": _sha256_json(suffix),
                    "owner_ids": [],
                }
            categories[key]["owner_ids"].append(gt_owner_id)
            coords = _coords(box_index)
            bank.append(
                {
                    "candidate_id": f"cand:{gt_owner_id}",
                    "image_id": image_id,
                    "normalized_description": description,
                    "coord_token_ids": coords,
                    "coord_token_ids_sha256": _sha256_json(coords),
                    "generators": [
                        {
                            "generator_gt_owner_id": gt_owner_id,
                            "logical_transform_role": "exact_gt_anchor",
                        }
                    ],
                }
            )

        # Description token counts drive the row-length predicate.
        self.description_token_count: dict[str, str | int] = {}
        for image_id in IMAGE_IDS:
            for suffix, token_count, _box_index in self._candidates(image_id):
                self.description_token_count[f"cand-{image_id}-{suffix}"] = token_count
            for gt_owner_id in self._owners_of(image_id):
                self.description_token_count[self._c_description(gt_owner_id)] = 2
            self.description_token_count["erow"] = 2
            self.description_token_count["tp"] = 2

        for image_index, image_id in enumerate(IMAGE_IDS):
            candidate_specs = self._candidates(image_id)
            for suffix, _count, box_index in candidate_specs:
                _register_owner(
                    f"gt:{image_id}:{suffix}",
                    image_id,
                    f"cand-{image_id}-{suffix}",
                    box_index,
                    excluded=suffix == EXCLUDED_CANDIDATE,
                )
            for owner_index, gt_owner_id in enumerate(self._owners_of(image_id)):
                _register_owner(
                    gt_owner_id,
                    image_id,
                    self._c_description(gt_owner_id),
                    20 + owner_index,
                    excluded=False,
                )
            _register_owner(f"gt:{image_id}:tp", image_id, "tp", 70, excluded=False)

            boundaries = [
                self._boundary_of(image_id, owner) for owner in self._owners_of(image_id)
            ]
            row_count = max(boundaries) + 3
            rows: list[list[int]] = []
            for row_index in range(row_count):
                owner_at_row = next(
                    (
                        owner
                        for owner in self._owners_of(image_id)
                        if self._boundary_of(image_id, owner) == row_index
                    ),
                    None,
                )
                if owner_at_row is not None:
                    description = self._e_description(image_id, owner_at_row)
                    box_index = 40 + self._owners_of(image_id).index(owner_at_row)
                else:
                    description = "erow"
                    box_index = 90 + row_index
                rows.append(
                    _row_tokens(
                        _tokens(description, self.description_token_count.get(description, 2)),
                        _coords(box_index),
                    )
                )
                strict_owner: str | None = None
                if row_index < len(candidate_specs):
                    strict_owner = f"gt:{image_id}:{candidate_specs[row_index][0]}"
                elif owner_at_row is not None:
                    strict_owner = self._e_owner(owner_at_row)
                sidecars.append(
                    {
                        "sidecar_id": f"sc:{image_id}:{row_index}",
                        "image_id": image_id,
                        "row_index": row_index,
                        "pred_row_id": f"pred:{image_id}:{row_index}",
                        "normalized_description": description,
                        "strict_match_status": (
                            crossing_plan.SIDECAR_MATCHED if strict_owner else "unmatched"
                        ),
                        "strict_match_gt_owner_id": strict_owner,
                        "coord_token_ids": _coords(box_index),
                        "coord_token_ids_sha256": _sha256_json(_coords(box_index)),
                        "raw_span_sha256": _sha256_json([image_id, row_index]),
                    }
                )
            self.rows_by_image[image_id] = rows
            prefix: list[int] = []
            for boundary_index in range(row_count + 1):
                context_id = crossing_plan.context_id_for(image_id, boundary_index)
                contexts.append(
                    {
                        "context_id": context_id,
                        "image_id": image_id,
                        "boundary_index": boundary_index,
                        "context_role": "root" if boundary_index == 0 else "row_boundary",
                        "generated_prefix_token_ids": list(prefix),
                        "generated_prefix_token_ids_sha256": _sha256_json(list(prefix)),
                    }
                )
                if boundary_index < row_count:
                    prefix = prefix + rows[boundary_index]
            images.append(
                {
                    "image_id": image_id,
                    "prompt_token_ids": [900 + image_index, 901 + image_index],
                    "coordinate_token_ids": {
                        "start": COORD_START,
                        "end_inclusive": COORD_START + 999,
                        "bin_count": 1000,
                    },
                }
            )

        files = {
            sut.CENSUS_OWNER_REGISTRY_NAME: owners,
            sut.CENSUS_IMAGE_REGISTRY_NAME: images,
            sut.CENSUS_CATEGORY_REGISTRY_NAME: list(categories.values()),
            sut.CENSUS_CANDIDATE_BANK_NAME: bank,
            sut.CENSUS_SIDECAR_REGISTRY_NAME: sidecars,
            sut.CENSUS_CONTEXT_REGISTRY_NAME: contexts,
        }
        for name, rows_payload in files.items():
            _write_jsonl(self.census_dir / name, rows_payload)
        receipt = _seal(
            {
                "unit_id": sut.CENSUS_UNIT_ID,
                "output_file_digests": {
                    name: {
                        "path": name,
                        "sha256": geometry.sha256_bytes((self.census_dir / name).read_bytes()),
                    }
                    for name in files
                },
            },
            "receipt_content_sha256",
        )
        _write_json(self.census_dir / sut.CENSUS_RECEIPT_NAME, receipt)
        self.census_receipt = receipt

    # -- crossing plan ---------------------------------------------------
    def _build_crossing(self) -> None:
        cohort_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        for image_id, gt_owner_id in CROSSING_OWNERS:
            boundary = self._boundary_of(image_id, gt_owner_id)
            e_tokens = self.rows_by_image[image_id][boundary]
            e_box_index = 40 + self._owners_of(image_id).index(gt_owner_id)
            c_coords = _coords(self.box_index[gt_owner_id])
            c_tokens = _row_tokens(
                _tokens(self._c_description(gt_owner_id), 2), c_coords
            )
            e_owner = self._e_owner(gt_owner_id)
            cohort_rows.append(
                {
                    "schema_version": crossing_plan.COHORT_SCHEMA_VERSION,
                    "unit_id": crossing_plan.UNIT_ID,
                    "cohort": crossing_plan.PRIMARY_COHORT,
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "normalized_description": self._c_description(gt_owner_id),
                    "crossing": {
                        "boundary_index_b": boundary,
                        "p_context_id": crossing_plan.context_id_for(image_id, boundary),
                    },
                    "e_row": {
                        "row_index": boundary,
                        "pred_row_id": f"pred:{image_id}:{boundary}",
                        "normalized_description": self._e_description(image_id, gt_owner_id),
                        "strict_match_status": (
                            crossing_plan.SIDECAR_MATCHED if e_owner else "unmatched"
                        ),
                        "strict_match_gt_owner_id": e_owner,
                        "stratum": "matched_e" if e_owner else "unmatched_e",
                        "full_row_token_ids": e_tokens,
                        "full_row_token_ids_sha256": _sha256_json(e_tokens),
                        "full_row_token_count": len(e_tokens),
                        "coord_token_ids": _coords(e_box_index),
                        "pre_row_context_id": crossing_plan.context_id_for(image_id, boundary),
                        "post_row_context_id": crossing_plan.context_id_for(
                            image_id, boundary + 1
                        ),
                    },
                    "inserted_clean_row_c": {
                        "token_ids": c_tokens,
                        "token_ids_sha256": _sha256_json(c_tokens),
                        "token_count": len(c_tokens),
                        "coord_token_ids": c_coords,
                    },
                }
            )
        for image_id in IMAGE_IDS:
            twin_coords = _coords(70)
            twin = _row_tokens(_tokens("tp", 2), twin_coords)
            following = self.rows_by_image[image_id][1]
            control_rows.append(
                {
                    "schema_version": crossing_plan.CONTROL_SCHEMA_VERSION,
                    "unit_id": crossing_plan.UNIT_ID,
                    "cohort": crossing_plan.TP_REPLAY_CONTROL_COHORT,
                    "gt_owner_id": f"gt:{image_id}:tp",
                    "image_id": image_id,
                    "normalized_description": "tp",
                    "due_context_id": crossing_plan.context_id_for(image_id, 0),
                    "row_index": 0,
                    "following_native_action": {
                        "kind": crossing_plan.NATIVE_ACTION_ROW,
                        "context_id": crossing_plan.context_id_for(image_id, 1),
                        "row_index": 1,
                        "token_ids": following,
                        "token_ids_sha256": _sha256_json(following),
                    },
                    "inserted_clean_row_c": {
                        "token_ids": twin,
                        "token_ids_sha256": _sha256_json(twin),
                        "token_count": len(twin),
                    },
                }
            )
        _write_jsonl(self.crossing_dir / sut.CROSSING_COHORT_REGISTRY_NAME, cohort_rows)
        _write_jsonl(self.crossing_dir / sut.CROSSING_CONTROL_REGISTRY_NAME, control_rows)
        self.cohort_rows = cohort_rows
        self.control_rows = control_rows
        manifest = _seal(
            {
                "unit_id": sut.CROSSING_UNIT_ID,
                "output_file_digests": {
                    name: {
                        "path": name,
                        "sha256": geometry.sha256_bytes(
                            (self.crossing_dir / name).read_bytes()
                        ),
                    }
                    for name in (
                        sut.CROSSING_COHORT_REGISTRY_NAME,
                        sut.CROSSING_CONTROL_REGISTRY_NAME,
                    )
                },
                "lineage": {
                    "census_run_root": str(self.census_dir.parent),
                    "census_plan_receipt_content_sha256": self.census_receipt[
                        "receipt_content_sha256"
                    ],
                    "census_input_files": {
                        f"plan/{name}": {
                            "path": f"plan/{name}",
                            "sha256": geometry.sha256_bytes(
                                (self.census_dir / name).read_bytes()
                            ),
                        }
                        for name in sut.CENSUS_PLAN_FILES
                    },
                },
            },
            "manifest_content_sha256",
        )
        _write_json(self.crossing_dir / sut.CROSSING_MANIFEST_NAME, manifest)
        self.crossing_manifest = manifest

    # -- sealed secondary capture ---------------------------------------
    def _relative_delta(self, gt_owner_id: str) -> float:
        return -2.5 if gt_owner_id in MATERIAL_OWNERS else 0.25

    def _paired_row(
        self, *, gt_owner_id: str, image_id: str, variant: str, tokens: list[int], delta: float
    ) -> dict[str, Any]:
        baseline_sum = -12.0
        return {
            "gt_owner_id": gt_owner_id,
            "image_id": image_id,
            "variant": variant,
            "request_id": f"req:{gt_owner_id}:{variant}",
            "scored_token_ids_sha256": _sha256_json(tokens),
            "scored_token_count": len(tokens),
            "deltas": {
                "coordinates": {
                    "delta": delta,
                    "sign": 1 if delta > 0 else (-1 if delta < 0 else 0),
                }
            },
            "roots": {
                "baseline_unmodified_native_root": {
                    "segment_sums": {"coordinates": {"sum": baseline_sum}},
                    "argmax_token_ids": list(tokens),
                    "argmax_reproduces_description_path": True,
                    "argmax_reproduces_complete_row": True,
                },
                "modified_inserted_clean_row_c_root": {
                    "segment_sums": {"coordinates": {"sum": baseline_sum + delta}},
                    "argmax_token_ids": list(tokens),
                },
            },
        }

    def _build_capture(self) -> None:
        rows: list[dict[str, Any]] = []
        for image_id, gt_owner_id in CROSSING_OWNERS:
            boundary = self._boundary_of(image_id, gt_owner_id)
            tokens = self.rows_by_image[image_id][boundary]
            # Raw C delta, chosen so the relative delta lands on the frozen label
            # once the benign reference (-0.5) is subtracted.
            delta = self._relative_delta(gt_owner_id) + self.benign_delta
            rows.append(
                self._paired_row(
                    gt_owner_id=gt_owner_id,
                    image_id=image_id,
                    variant=sut.ARM_CLEAN_REPLAY,
                    tokens=tokens,
                    delta=delta,
                )
            )
        for image_id in IMAGE_IDS:
            rows.append(
                self._paired_row(
                    gt_owner_id=f"gt:{image_id}:tp",
                    image_id=image_id,
                    variant=sut.ARM_BENIGN,
                    tokens=self.rows_by_image[image_id][1],
                    delta=self.benign_delta,
                )
            )
        _write_jsonl(self.merged_dir / sut.SECONDARY_MERGED_ROWS_NAME, rows)
        self.merged_rows = rows
        receipt = _seal(
            {
                "unit_id": sut.CROSSING_UNIT_ID,
                "output_file_digests": {
                    sut.SECONDARY_MERGED_ROWS_NAME: {
                        "path": sut.SECONDARY_MERGED_ROWS_NAME,
                        "sha256": geometry.sha256_bytes(
                            (self.merged_dir / sut.SECONDARY_MERGED_ROWS_NAME).read_bytes()
                        ),
                    }
                },
            },
            "receipt_content_sha256",
        )
        _write_json(self.merged_dir / sut.SECONDARY_MERGE_RECEIPT_NAME, receipt)
        self.merge_receipt = receipt

    benign_delta = -0.5

    # -- geometry analysis -----------------------------------------------
    def _build_geometry(self) -> None:
        rows: list[dict[str, Any]] = []
        for image_id, gt_owner_id in CROSSING_OWNERS:
            rows.append(
                {
                    "unit_id": sut.GEOMETRY_UNIT_ID,
                    "arm": sut.ARM_CLEAN_REPLAY,
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "enters_primary_decision": True,
                    "material_negative": gt_owner_id in MATERIAL_OWNERS,
                    "relative_coordinate_delta": self._relative_delta(gt_owner_id),
                    "geometry": {"clearly_separated": gt_owner_id in SEPARATED_OWNERS},
                }
            )
            rows.append(
                {
                    "unit_id": sut.GEOMETRY_UNIT_ID,
                    "arm": "p_plus_e_plus_c_then_f",
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "enters_primary_decision": True,
                    "material_negative": False,
                    "relative_coordinate_delta": 0.1,
                    "geometry": {"clearly_separated": True},
                }
            )
        _write_jsonl(self.geometry_dir / sut.GEOMETRY_OWNER_ROWS_NAME, rows)
        self.geometry_rows = rows
        receipt = _seal(
            {
                "unit_id": sut.GEOMETRY_UNIT_ID,
                "output_file_digests": {
                    sut.GEOMETRY_OWNER_ROWS_NAME: {
                        "path": sut.GEOMETRY_OWNER_ROWS_NAME,
                        "sha256": geometry.sha256_bytes(
                            (self.geometry_dir / sut.GEOMETRY_OWNER_ROWS_NAME).read_bytes()
                        ),
                    }
                },
                "binding": {
                    "plan_manifest_content_sha256": self.crossing_manifest[
                        "manifest_content_sha256"
                    ],
                    "secondary_merge_receipt_content_sha256": self.merge_receipt[
                        "receipt_content_sha256"
                    ],
                    "census_plan_receipt_content_sha256": self.census_receipt[
                        "receipt_content_sha256"
                    ],
                    "census_run_root": str(self.census_dir.parent),
                },
                "input_file_sha256": {
                    str(self.crossing_dir / name): geometry.sha256_bytes(
                        (self.crossing_dir / name).read_bytes()
                    )
                    for name in (
                        sut.CROSSING_MANIFEST_NAME,
                        sut.CROSSING_COHORT_REGISTRY_NAME,
                    )
                },
            },
            "receipt_content_sha256",
        )
        _write_json(self.geometry_dir / sut.GEOMETRY_RECEIPT_NAME, receipt)
        self.geometry_receipt = receipt

    # -- entry points -----------------------------------------------------
    def inputs(self) -> sut.SealedInputs:
        return sut.load_sealed_inputs(
            geometry_analysis_dir=self.geometry_dir,
            crossing_plan_dir=self.crossing_dir,
            secondary_merged_dir=self.merged_dir,
            census_plan_dir=self.census_dir,
        )

    def build(self, output_root: Path) -> dict[str, Any]:
        return sut.build_plan(
            geometry_analysis_dir=self.geometry_dir,
            crossing_plan_dir=self.crossing_dir,
            secondary_merged_dir=self.merged_dir,
            census_plan_dir=self.census_dir,
            output_root=output_root,
        )


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> _Tree:
    return _Tree(tmp_path_factory.mktemp("neutral-row-tree"))


@pytest.fixture(scope="module")
def built(tree: _Tree, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("neutral-row-plan")
    manifest = tree.build(root)
    plan_dir = root / sut.PLAN_DIR_NAME
    return {
        "manifest": manifest,
        "plan_dir": plan_dir,
        "selection": [
            json.loads(line)
            for line in (plan_dir / sut.SELECTION_REGISTRY_NAME).read_text().splitlines()
        ],
        "benign": [
            json.loads(line)
            for line in (plan_dir / sut.BENIGN_REGISTRY_NAME).read_text().splitlines()
        ],
        "requests": [
            json.loads(line)
            for line in (plan_dir / sut.REQUEST_PLAN_NAME).read_text().splitlines()
        ],
    }


# ---------------------------------------------------------------------------
# 1. Frozen constants
# ---------------------------------------------------------------------------


def test_frozen_cohort_request_and_gate_constants() -> None:
    assert (sut.VOTING_OWNER_COUNT, sut.SPECIFICITY_OWNER_COUNT, sut.INFEASIBLE_OWNER_COUNT) == (
        9,
        12,
        5,
    )
    assert sut.EXECUTED_OWNER_COUNT == 21
    assert dict(sut.EXPECTED_REQUEST_COUNT_BY_ARM) == {
        sut.ARM_NEUTRAL: 21,
        sut.ARM_CLEAN_REPLAY: 21,
        sut.ARM_BENIGN: 12,
    }
    assert sut.TOTAL_REQUEST_COUNT == 54
    assert sut.MATERIALITY_MAX_NATS == -1.0
    assert sut.MATERIALITY_MAX_NATS is geometry.MATERIAL_NEGATIVE_MAX_NATS
    assert sut.REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF == 1e-3
    assert sut.REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF == 0.05
    assert sut.MAX_QUARANTINED_OWNERS == 2
    assert sut.MAX_CLEAN_REPLAY_FAILURES == 2
    assert sut.SPECIFICITY_MATERIAL_MAX == 4
    assert sut.MAX_ROW_LENGTH_DELTA_TOKENS == 2
    assert sut.PROHIBITED_IMAGE_IDS == frozenset({"2299"})
    assert sut.SENTINEL_OWNER_IDS == ("gt:13923:14", "gt:4134:27")
    assert sut.POSTHOC_SUPPORT_UNCERTAIN_OWNER_IDS == ("gt:16228:11", "gt:4134:29")
    assert sut.SENSITIVITY_CUTOFFS_NATS == (-0.75, -1.25)


def test_frozen_route_thresholds_are_absolute_integers() -> None:
    assert (
        sut.ROUTE_C_MIN_NONMATERIAL,
        sut.ROUTE_C_MIN_IMAGES,
        sut.ROUTE_C_MIN_SEPARATED,
        sut.ROUTE_C_MIN_SEPARATED_IMAGES,
    ) == (5, 3, 2, 2)
    assert (
        sut.ROUTE_GENERIC_MIN_MATERIAL,
        sut.ROUTE_GENERIC_MIN_IMAGES,
        sut.ROUTE_GENERIC_MIN_SEPARATED,
        sut.ROUTE_GENERIC_MIN_SEPARATED_IMAGES,
    ) == (7, 3, 3, 2)
    assert sut.ROUTE_ORDER == (
        sut.ROUTE_C_CONTENT_SPECIFIC,
        sut.ROUTE_GENERIC_SUFFICIENT,
        sut.ROUTE_INCONCLUSIVE,
    )


def test_predicate_and_ordering_vocabulary_is_frozen() -> None:
    assert sut.PREDICATE_ORDER == (
        sut.PREDICATE_SAME_IMAGE_CENSUS_OWNER,
        sut.PREDICATE_NOT_C_AND_NOT_E_OWNER,
        sut.PREDICATE_COVERED_BEFORE_P,
        sut.PREDICATE_DIFFERENT_DESCRIPTION,
        sut.PREDICATE_GEOMETRICALLY_SEPARATED,
        sut.PREDICATE_ASSEMBLABLE,
        sut.PREDICATE_MATCHED_ROW_LENGTH,
    )
    # Assemblability is the structural prerequisite of the exact row-length
    # measurement, so it is declared -- and evaluated -- immediately before it.
    assert sut.STRUCTURAL_PREREQUISITE_PREDICATES == (sut.PREDICATE_ASSEMBLABLE,)
    assert sut.PREDICATE_ORDER.index(sut.PREDICATE_ASSEMBLABLE) + 1 == (
        sut.PREDICATE_ORDER.index(sut.PREDICATE_MATCHED_ROW_LENGTH)
    )
    assert sut.ORDERING_RULE == (
        "smallest_absolute_row_token_length_difference_from_c",
        "latest_strict_matched_native_row_index_before_p",
        "largest_minimum_normalized_center_distance_to_c_and_e",
        "ascending_physical_owner_id",
    )


def test_frozen_smoke_strata_are_the_geometry_axes() -> None:
    assert sut.E_STRATUM_AXIS is geometry.E_STRATUM_AXIS
    assert sut.DESCRIPTION_AXIS is geometry.DESCRIPTION_AXIS
    assert sut.REQUIRED_SMOKE_STRATA == (
        "matched_e",
        "unmatched_e",
        "same_description",
        "different_description",
    )


# ---------------------------------------------------------------------------
# 2. Exact counts
# ---------------------------------------------------------------------------


def test_plan_reproduces_the_frozen_9_12_5_split(built: dict[str, Any]) -> None:
    cohort = built["manifest"]["cohort"]
    assert cohort["voting_owner_count"] == 9
    assert cohort["specificity_owner_count"] == 12
    assert cohort["infeasible_owner_count"] == 5
    assert cohort["infeasible_owner_ids"] == sorted(sut.FROZEN_INFEASIBLE_OWNER_IDS)
    assert cohort["clearly_separated_voting_owner_ids"] == sorted(
        sut.FROZEN_CLEARLY_SEPARATED_OWNER_IDS
    )
    assert cohort["voting_owner_ids"] == sorted(MATERIAL_OWNERS)
    assert cohort["crossing_owner_count"] == 26
    assert cohort["image_count"] == 12
    assert "2299" not in cohort["image_ids"]


def test_plan_emits_exactly_21_21_and_12_requests(built: dict[str, Any]) -> None:
    assert built["manifest"]["cohort"]["request_count_total"] == 54
    assert built["manifest"]["cohort"]["request_count_by_arm"] == {
        sut.ARM_BENIGN: 12,
        sut.ARM_CLEAN_REPLAY: 21,
        sut.ARM_NEUTRAL: 21,
    }
    assert len(built["requests"]) == 54
    assert len(built["selection"]) == 26
    assert len(built["benign"]) == 12
    assert len({row["request_id"] for row in built["requests"]}) == 54


def test_infeasible_owners_are_ledgered_and_never_executed(built: dict[str, Any]) -> None:
    infeasible = [row for row in built["selection"] if not row["executed"]]
    assert sorted(row["gt_owner_id"] for row in infeasible) == sorted(
        sut.FROZEN_INFEASIBLE_OWNER_IDS
    )
    for row in infeasible:
        assert row["cohort_role"] == sut.COHORT_INFEASIBLE
        assert row["neutral_row_n"] is None
        assert row["selection"]["candidate_count"] == 0
        assert row["selection"]["evaluated_owner_count"] > 0
        assert row["request_ids"] == []
        assert row["selection"]["eliminated_by_predicate"]
    executed = {row["gt_owner_id"] for row in built["selection"] if row["executed"]}
    assert len(executed) == 21
    assert executed.isdisjoint(sut.FROZEN_INFEASIBLE_OWNER_IDS)


def test_every_executed_owner_carries_one_neutral_and_one_clean_request(
    built: dict[str, Any],
) -> None:
    by_owner: dict[str, set[str]] = {}
    for request in built["requests"]:
        by_owner.setdefault(request["gt_owner_id"], set()).add(request["arm"])
    for row in built["selection"]:
        arms = by_owner.get(row["gt_owner_id"], set())
        if row["executed"]:
            assert arms == {sut.ARM_NEUTRAL, sut.ARM_CLEAN_REPLAY}
        else:
            assert arms == set()
    for row in built["benign"]:
        assert by_owner[row["gt_owner_id"]] == {sut.ARM_BENIGN}


def test_image_7511_carries_only_its_benign_pair(built: dict[str, Any]) -> None:
    """An image whose only crossing owner is infeasible still needs its reference."""

    rows = [row for row in built["requests"] if row["image_id"] == "7511"]
    assert [row["arm"] for row in rows] == [sut.ARM_BENIGN]


# ---------------------------------------------------------------------------
# 3. Every predicate
# ---------------------------------------------------------------------------


def _target(tree: _Tree, gt_owner_id: str) -> tuple[sut.SealedInputs, sut.TargetBinding]:
    inputs = tree.inputs()
    return inputs, sut.bind_target(inputs, gt_owner_id)


@pytest.mark.parametrize(
    ("gt_owner_id", "candidate_owner_id", "expected"),
    [
        ("gt:10707:16", "gt:13348:n0", sut.PREDICATE_SAME_IMAGE_CENSUS_OWNER),
        ("gt:2685:20", "gt:2685:excluded", sut.PREDICATE_SAME_IMAGE_CENSUS_OWNER),
        ("gt:10707:16", "gt:10707:16", sut.PREDICATE_NOT_C_AND_NOT_E_OWNER),
        ("gt:13348:7", "gt:13348:n1", sut.PREDICATE_NOT_C_AND_NOT_E_OWNER),
        ("gt:10707:16", "gt:10707:tp", sut.PREDICATE_COVERED_BEFORE_P),
        ("gt:13923:1", "gt:13923:n0", sut.PREDICATE_DIFFERENT_DESCRIPTION),
        ("gt:13923:1", "gt:13923:n1", sut.PREDICATE_DIFFERENT_DESCRIPTION),
        ("gt:10707:16", "gt:10707:overlap", sut.PREDICATE_GEOMETRICALLY_SEPARATED),
        ("gt:14038:10", "gt:14038:n1", sut.PREDICATE_MATCHED_ROW_LENGTH),
    ],
)
def test_each_predicate_refuses_its_own_candidate(
    tree: _Tree, gt_owner_id: str, candidate_owner_id: str, expected: str
) -> None:
    inputs, target = _target(tree, gt_owner_id)
    candidate, failed = sut.evaluate_candidate(inputs, target, candidate_owner_id)
    assert candidate is None
    assert failed == expected


def test_an_admissible_candidate_satisfies_every_predicate(tree: _Tree) -> None:
    inputs, target = _target(tree, "gt:10707:16")
    candidate, failed = sut.evaluate_candidate(inputs, target, "gt:10707:n0")
    assert failed is None
    assert candidate is not None
    assert candidate.gt_owner_id == "gt:10707:n0"
    assert abs(candidate.row_length_delta_tokens) <= sut.MAX_ROW_LENGTH_DELTA_TOKENS
    assert candidate.matched_native_row_index < target.boundary_index_b
    assert candidate.rows_back_distance == (
        target.boundary_index_b - candidate.matched_native_row_index
    )
    assert candidate.relation_to_c["iou"] == 0.0
    assert candidate.relation_to_e["iou"] == 0.0
    assert candidate.relation_to_c["clearly_separated"]
    assert candidate.relation_to_e["clearly_separated"]
    assert candidate.min_center_distance == min(
        candidate.center_distance_to_c, candidate.center_distance_to_e
    )
    assert candidate.normalized_description not in {
        target.normalized_description,
        target.e_description,
    }


def _without_category(inputs: sut.SealedInputs, category_query_id: str) -> sut.SealedInputs:
    """The same sealed inputs with one category's query suffix unavailable."""

    stripped = copy.deepcopy(inputs.categories_by_query_id)
    stripped.pop(category_query_id)
    return sut.SealedInputs(**{**inputs.__dict__, "categories_by_query_id": stripped})


def test_an_unassemblable_candidate_is_refused_visibly(tree: _Tree) -> None:
    inputs, target = _target(tree, "gt:10707:16")
    patched = _without_category(inputs, "10707:cand-10707-n0")
    candidate, failed = sut.evaluate_candidate(patched, target, "gt:10707:n0")
    assert candidate is None
    assert failed == sut.PREDICATE_ASSEMBLABLE


def test_a_missing_suffix_is_reported_as_assemblability_not_as_row_length(
    tree: _Tree,
) -> None:
    """The declared refusal reason is the one that actually stopped the candidate.

    ``gt:14038:n1`` is the candidate whose assembled row is too long, so it
    normally fails the row-length predicate.  With its sealed query suffix gone
    there is no assembled length to measure, and the provenance must say so
    instead of attributing the refusal to a length it never computed.
    """

    inputs, target = _target(tree, "gt:14038:10")
    assert sut.evaluate_candidate(inputs, target, "gt:14038:n1")[1] == (
        sut.PREDICATE_MATCHED_ROW_LENGTH
    )
    patched = _without_category(inputs, "14038:cand-14038-n1")
    candidate, failed = sut.evaluate_candidate(patched, target, "gt:14038:n1")
    assert candidate is None
    assert failed == sut.PREDICATE_ASSEMBLABLE
    _, ledger = sut.select_neutral_row(patched, target)
    entry = next(item for item in ledger if item["gt_owner_id"] == "gt:14038:n1")
    assert entry["admitted"] is False
    assert entry["first_failed_predicate"] == sut.PREDICATE_ASSEMBLABLE


def test_a_candidate_covered_only_after_p_is_refused(tree: _Tree) -> None:
    """Coverage is strictly ``row_index < boundary b``, never at or after it."""

    inputs, target = _target(tree, "gt:10707:16")
    patched_rows = {
        owner_id: [
            {**row, "row_index": target.boundary_index_b}
            for row in rows
        ]
        for owner_id, rows in inputs.matched_rows_by_owner.items()
    }
    patched = sut.SealedInputs(
        **{**inputs.__dict__, "matched_rows_by_owner": patched_rows}
    )
    candidate, failed = sut.evaluate_candidate(patched, target, "gt:10707:n0")
    assert candidate is None
    assert failed == sut.PREDICATE_COVERED_BEFORE_P


# ---------------------------------------------------------------------------
# 4. Selection order
# ---------------------------------------------------------------------------


def _candidate(
    gt_owner_id: str,
    *,
    length_delta: int,
    row_index: int,
    min_distance: float,
) -> sut.Candidate:
    relation = {"center_distance_normalized": min_distance, "clearly_separated": True}
    return sut.Candidate(
        gt_owner_id=gt_owner_id,
        normalized_description="d",
        row_length_delta_tokens=length_delta,
        matched_native_row_index=row_index,
        rows_back_distance=1,
        center_distance_to_c=min_distance,
        center_distance_to_e=min_distance,
        min_center_distance=min_distance,
        relation_to_c=relation,
        relation_to_e=relation,
        assembled={"token_count": 10},
        coord_token_ids=(),
        box_norm1000_xyxy=(0, 0, 1, 1),
    )


def test_order_key_prefers_the_smallest_absolute_row_length_difference() -> None:
    near = _candidate("gt:z", length_delta=1, row_index=0, min_distance=0.0)
    far = _candidate("gt:a", length_delta=-2, row_index=9, min_distance=1.0)
    assert min([far, near], key=lambda item: item.order_key) is near


def test_order_key_then_prefers_the_latest_covered_native_row() -> None:
    early = _candidate("gt:a", length_delta=0, row_index=1, min_distance=1.0)
    late = _candidate("gt:z", length_delta=0, row_index=7, min_distance=0.0)
    assert min([early, late], key=lambda item: item.order_key) is late


def test_order_key_then_prefers_the_largest_minimum_center_distance() -> None:
    near = _candidate("gt:a", length_delta=0, row_index=3, min_distance=0.1)
    far = _candidate("gt:z", length_delta=0, row_index=3, min_distance=0.4)
    assert min([near, far], key=lambda item: item.order_key) is far


def test_order_key_breaks_the_final_tie_by_ascending_owner_id() -> None:
    first = _candidate("gt:a", length_delta=0, row_index=3, min_distance=0.2)
    second = _candidate("gt:b", length_delta=0, row_index=3, min_distance=0.2)
    assert min([second, first], key=lambda item: item.order_key) is first


def test_selection_takes_the_first_candidate_under_the_total_order(tree: _Tree) -> None:
    """Image 4134 ties two candidates on length; the later covered row wins."""

    inputs, target = _target(tree, "gt:4134:22")
    candidate, ledger = sut.select_neutral_row(inputs, target)
    assert candidate is not None
    admitted = [entry for entry in ledger if entry["admitted"]]
    assert {entry["gt_owner_id"] for entry in admitted} == {
        "gt:4134:n0",
        "gt:4134:n1",
        "gt:4134:n2",
    }
    assert candidate.gt_owner_id == "gt:4134:n1"
    assert candidate.row_length_delta_tokens == 0
    assert candidate.matched_native_row_index == 1


def test_a_single_candidate_image_reports_only_one_candidate(built: dict[str, Any]) -> None:
    row = next(item for item in built["selection"] if item["gt_owner_id"] == "gt:6040:13")
    assert row["selection"]["candidate_count"] == 1
    assert row["selection"]["only_one_candidate"] is True


def test_selection_records_every_disclosure_unit_md_requires(built: dict[str, Any]) -> None:
    row = next(item for item in built["selection"] if item["gt_owner_id"] == "gt:10707:16")
    neutral = row["neutral_row_n"]
    for field in (
        "row_length_delta_tokens",
        "matched_native_row_index",
        "rows_back_distance",
        "sorted_route_regression",
        "center_distance_to_c_normalized",
        "center_distance_to_e_normalized",
        "min_center_distance_normalized",
        "geometry_relation_to_c",
        "geometry_relation_to_e",
    ):
        assert field in neutral
    assert neutral["sorted_route_regression"]["is_regression"] is True
    assert row["selection"]["minimum_center_distance_floor"] is None
    assert row["selection"]["uses_scores"] is False


def test_the_manifest_declares_the_evaluation_order_and_its_prerequisite(
    built: dict[str, Any],
) -> None:
    rule = built["manifest"]["selection_rule"]
    assert rule["predicate_order"] == list(sut.PREDICATE_ORDER)
    assert rule["predicate_order_is_the_evaluation_order"] is True
    assert rule["structural_prerequisite_predicates"] == [sut.PREDICATE_ASSEMBLABLE]
    assert built["manifest"]["gates"]["representative_smoke_required_strata"] == list(
        sut.REQUIRED_SMOKE_STRATA
    )
    assert built["selection"][0]["selection"]["predicate_order"] == list(sut.PREDICATE_ORDER)


def test_every_selection_row_carries_strata_on_the_frozen_axes(
    built: dict[str, Any],
) -> None:
    strata = {
        row["gt_owner_id"]: sut.selection_row_strata(row) for row in built["selection"]
    }
    assert len(strata) == 26
    for gt_owner_id, values in strata.items():
        assert values["e_stratum"] in sut.E_STRATUM_AXIS, gt_owner_id
        assert values["description_relation"] in sut.DESCRIPTION_AXIS, gt_owner_id
    # The one synthetic owner whose E row strict-matches a physical owner.
    assert strata[E_OWNER_IS_CANDIDATE_TARGET]["e_stratum"] == sut.E_STRATUM_MATCHED
    assert strata["gt:10707:16"]["e_stratum"] == sut.E_STRATUM_UNMATCHED


def test_selection_row_strata_reads_the_description_axis_from_c_and_e() -> None:
    row = {
        "gt_owner_id": "gt:4134:29",
        "normalized_description": "person",
        "scored_e_row": {
            "strict_match_status": "unmatched",
            "stratum": sut.E_STRATUM_UNMATCHED,
            "normalized_description": "person",
        },
    }
    assert sut.selection_row_strata(row) == {
        "e_stratum": sut.E_STRATUM_UNMATCHED,
        "description_relation": sut.DESCRIPTION_SAME,
    }
    row["scored_e_row"]["normalized_description"] = "truck"
    assert sut.selection_row_strata(row)["description_relation"] == sut.DESCRIPTION_DIFFERENT


def test_selection_row_strata_fails_closed_on_a_drifted_sealed_pair() -> None:
    row = {
        "gt_owner_id": "gt:4134:27",
        "normalized_description": "person",
        "scored_e_row": {
            "strict_match_status": "matched",
            "stratum": sut.E_STRATUM_UNMATCHED,
            "normalized_description": "truck",
        },
    }
    with pytest.raises(sut.NeutralRowPlanContractError, match="sealed pair disagrees"):
        sut.selection_row_strata(row)


def test_selection_is_deterministic_across_repeated_builds(tree: _Tree, tmp_path: Path) -> None:
    first = tree.build(tmp_path / "a")
    second = tree.build(tmp_path / "b")
    assert first["manifest_content_sha256"] == second["manifest_content_sha256"]


# ---------------------------------------------------------------------------
# 5. Token identity and assembly
# ---------------------------------------------------------------------------


def test_neutral_rows_are_assembled_from_sealed_tokens_without_retokenization(
    tree: _Tree, built: dict[str, Any]
) -> None:
    inputs = tree.inputs()
    for row in built["selection"]:
        neutral = row["neutral_row_n"]
        if neutral is None:
            continue
        category = inputs.categories_by_query_id[neutral["category_query_id"]]
        suffix = list(category["query_suffix_token_ids"])
        anchor = inputs.exact_anchor_by_owner[neutral["gt_owner_id"]]
        assert neutral["token_ids"] == [
            *suffix,
            *anchor["coord_token_ids"],
            BOX_END,
        ]
        assert neutral["token_ids"][-1] == BOX_END
        assert len(neutral["coord_token_ids"]) == 4
        assert neutral["token_ids_sha256"] == _sha256_json(neutral["token_ids"])
        assert neutral["retokenized"] is False


def test_scored_e_tokens_are_identical_across_the_n_and_c_arms(built: dict[str, Any]) -> None:
    by_owner: dict[str, dict[str, dict[str, Any]]] = {}
    for request in built["requests"]:
        if request["arm"] == sut.ARM_BENIGN:
            continue
        by_owner.setdefault(request["gt_owner_id"], {})[request["arm"]] = request
    assert len(by_owner) == 21
    for gt_owner_id, arms in by_owner.items():
        neutral = arms[sut.ARM_NEUTRAL]["scored_target"]
        clean = arms[sut.ARM_CLEAN_REPLAY]["scored_target"]
        assert neutral["token_ids"] == clean["token_ids"], gt_owner_id
        assert neutral["token_ids_sha256"] == clean["token_ids_sha256"]
        assert _sha256_json(neutral["token_ids"]) == neutral["token_ids_sha256"]
        assert (
            arms[sut.ARM_NEUTRAL]["prefix"]["appended_token_ids"]
            != arms[sut.ARM_CLEAN_REPLAY]["prefix"]["appended_token_ids"]
        )


def test_token_identity_check_refuses_two_different_scored_rows() -> None:
    def _request(arm: str, tokens: list[int]) -> dict[str, Any]:
        return {
            "arm": arm,
            "gt_owner_id": "gt:x:1",
            "scored_target": {
                "token_ids": tokens,
                "token_ids_sha256": _sha256_json(tokens),
            },
        }

    good = [_request(sut.ARM_NEUTRAL, [1, 2]), _request(sut.ARM_CLEAN_REPLAY, [1, 2])]
    assert sut.assert_scored_token_identity_across_arms(good)["checked_owner_count"] == 1
    bad = [_request(sut.ARM_NEUTRAL, [1, 2]), _request(sut.ARM_CLEAN_REPLAY, [1, 3])]
    with pytest.raises(sut.NeutralRowPlanContractError, match="different E tokens"):
        sut.assert_scored_token_identity_across_arms(bad)


def test_a_missing_arm_fails_closed() -> None:
    lone = [
        {
            "arm": sut.ARM_NEUTRAL,
            "gt_owner_id": "gt:x:1",
            "scored_target": {"token_ids": [1], "token_ids_sha256": _sha256_json([1])},
        }
    ]
    with pytest.raises(sut.NeutralRowPlanContractError, match="both a neutral-row"):
        sut.assert_scored_token_identity_across_arms(lone)


def test_requests_pair_against_the_native_baseline_and_never_retokenize(
    built: dict[str, Any],
) -> None:
    for request in built["requests"]:
        assert request["prefix"]["retokenized"] is False
        assert request["inspects_new_model_logits"] is False
        assert request["score_blind_plan"] is True
        assert request["paired_roots"]["orientation"] == "modified_minus_baseline"
        assert request["scored_target"]["primary_segment"] == "coordinates"
        if request["arm"] == sut.ARM_BENIGN:
            assert (
                request["paired_roots"]["baseline_context_id"]
                != request["paired_roots"]["modified_context_id"]
            )
        else:
            assert (
                request["paired_roots"]["baseline_context_id"]
                == request["paired_roots"]["modified_context_id"]
            )


# ---------------------------------------------------------------------------
# 6. Sealed references and the fail-closed chain
# ---------------------------------------------------------------------------


def test_sealed_references_are_gate_references_only(built: dict[str, Any]) -> None:
    for row in built["selection"]:
        reference = row["sealed_clean_reference"]
        assert reference["role"] == "gate_reference_only_never_the_estimand"
        assert reference["materiality_cutoff_nats"] == -1.0
        assert reference["material_negative"] is (
            row["gt_owner_id"] in MATERIAL_OWNERS
        )
        assert "baseline_argmax_token_ids_sha256" in reference
    for row in built["benign"]:
        assert (
            row["sealed_benign_reference"]["role"]
            == "gate_reference_only_the_same_run_replay_owns_the_estimand"
        )
    assert (
        built["manifest"]["materiality"]["benign_reference_source"] == "same_run_replay_only"
    )


def test_manifest_self_seals_and_carries_complete_lineage(built: dict[str, Any]) -> None:
    manifest = dict(built["manifest"])
    declared = manifest.pop("manifest_content_sha256")
    assert _sha256_json(manifest) == declared
    lineage = built["manifest"]["lineage"]
    for seal in lineage["input_files"].values():
        assert set(seal) == {"path", "byte_size", "sha256"}
    for name, entry in built["manifest"]["output_file_digests"].items():
        payload = (built["plan_dir"] / name).read_bytes()
        assert entry["sha256"] == geometry.sha256_bytes(payload)
        assert entry["byte_size"] == len(payload)


def test_rerunning_the_build_is_create_or_identical(tree: _Tree, tmp_path: Path) -> None:
    root = tmp_path / "publish"
    first = tree.build(root)
    before = sorted(
        (path.name, path.read_bytes()) for path in (root / sut.PLAN_DIR_NAME).iterdir()
    )
    second = tree.build(root)
    after = sorted(
        (path.name, path.read_bytes()) for path in (root / sut.PLAN_DIR_NAME).iterdir()
    )
    assert first == second
    assert before == after


def test_a_drifted_geometry_receipt_fails_closed(tree: _Tree, tmp_path: Path) -> None:
    receipt = json.loads((tree.geometry_dir / sut.GEOMETRY_RECEIPT_NAME).read_text())
    receipt["binding"]["plan_manifest_content_sha256"] = "0" * 64
    edited = tmp_path / "geometry"
    edited.mkdir()
    for child in tree.geometry_dir.iterdir():
        (edited / child.name).write_bytes(child.read_bytes())
    _write_json(edited / sut.GEOMETRY_RECEIPT_NAME, receipt)
    with pytest.raises(sut.NeutralRowPlanContractError):
        sut.load_sealed_inputs(
            geometry_analysis_dir=edited,
            crossing_plan_dir=tree.crossing_dir,
            secondary_merged_dir=tree.merged_dir,
            census_plan_dir=tree.census_dir,
        )


def test_an_edited_geometry_row_file_fails_closed(tree: _Tree, tmp_path: Path) -> None:
    edited = tmp_path / "geometry-rows"
    edited.mkdir()
    for child in tree.geometry_dir.iterdir():
        (edited / child.name).write_bytes(child.read_bytes())
    rows = copy.deepcopy(tree.geometry_rows)
    rows[0]["material_negative"] = not rows[0]["material_negative"]
    _write_jsonl(edited / sut.GEOMETRY_OWNER_ROWS_NAME, rows)
    with pytest.raises(sut.NeutralRowPlanContractError):
        sut.load_sealed_inputs(
            geometry_analysis_dir=edited,
            crossing_plan_dir=tree.crossing_dir,
            secondary_merged_dir=tree.merged_dir,
            census_plan_dir=tree.census_dir,
        )


def test_a_prohibited_image_in_the_cohort_fails_closed(tree: _Tree) -> None:
    inputs = tree.inputs()
    with pytest.raises(sut.NeutralRowPlanContractError, match="prohibited image"):
        sut._assert_no_prohibited_image(["10707", "2299"], label="probe")
    assert "2299" not in inputs.images_by_id


def test_a_drifted_cohort_denominator_fails_closed(built: dict[str, Any]) -> None:
    selection = copy.deepcopy(built["selection"])
    victim = next(row for row in selection if row["cohort_role"] == sut.COHORT_SPECIFICITY)
    victim["cohort_role"] = sut.COHORT_INFEASIBLE
    victim["executed"] = False
    with pytest.raises(
        sut.NeutralRowPlanContractError, match="feasible specificity stratum holds 11"
    ):
        sut.derive_cohort_counts(selection, built["benign"], built["requests"])


def test_a_renamed_infeasible_owner_fails_the_frozen_ledger(built: dict[str, Any]) -> None:
    """The five ledgered ids are named by unit.md and are re-derived, not reported."""

    selection = copy.deepcopy(built["selection"])
    swap_out = next(row for row in selection if row["cohort_role"] == sut.COHORT_INFEASIBLE)
    swap_in = next(row for row in selection if row["cohort_role"] == sut.COHORT_SPECIFICITY)
    swap_out["cohort_role"] = sut.COHORT_SPECIFICITY
    swap_out["executed"] = True
    swap_in["cohort_role"] = sut.COHORT_INFEASIBLE
    swap_in["executed"] = False
    with pytest.raises(sut.NeutralRowPlanContractError, match="infeasibility ledger"):
        sut.derive_cohort_counts(selection, built["benign"], built["requests"])


def test_a_renamed_clearly_separated_subset_fails_closed(built: dict[str, Any]) -> None:
    selection = copy.deepcopy(built["selection"])
    victim = next(
        row for row in selection if row["votes"] and not row["clearly_separated"]
    )
    victim["clearly_separated"] = True
    with pytest.raises(
        sut.NeutralRowPlanContractError, match="clearly separated voting subset"
    ):
        sut.derive_cohort_counts(selection, built["benign"], built["requests"])


def test_a_material_owner_without_a_neutral_row_stops_for_review(tree: _Tree) -> None:
    """unit.md: preparation stops rather than relaxing a predicate."""

    inputs = tree.inputs()
    labels = sut.read_cohort_labels(inputs)
    patched = sut.SealedInputs(
        **{
            **inputs.__dict__,
            "matched_rows_by_owner": {
                owner_id: rows
                for owner_id, rows in inputs.matched_rows_by_owner.items()
                if not owner_id.startswith("gt:13348:")
            },
        }
    )
    with pytest.raises(sut.NeutralRowPlanContractError, match="stops preparation for review"):
        sut.build_selection_rows(patched, labels)


def test_the_frozen_material_label_is_the_only_likelihood_fact_selection_sees(
    tree: _Tree,
) -> None:
    labels = sut.read_cohort_labels(tree.inputs())
    assert len(labels) == 26
    assert sum(1 for label in labels.values() if label.material_negative) == 9
    assert set(sut.CohortLabel.__dataclass_fields__) == {
        "gt_owner_id",
        "image_id",
        "material_negative",
        "clearly_separated",
    }
