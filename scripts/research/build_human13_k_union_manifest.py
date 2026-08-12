#!/usr/bin/env python3
"""Build the hash-bound Human-13 K-union research manifest.

This module is deliberately experiment-local.  It validates and serializes the
frozen Human-13 identities without changing generic StateBank admission and
without importing a model runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    iou_xyxy,
)


SCHEMA_VERSION = "human13_k_union_manifest.v1"
UNIT_ID = "2026-08-12-human13-k-union-to-greedy-overfit-screen"
PURPOSE = "overfit_only"
PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"

EXPECTED_IMAGE_IDENTITIES = (
    (1584, "06b9d29a50b896f1bec14a267a57016723e54e205a1d1a40088237d95ce91206"),
    (2299, "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"),
    (2685, "84514a8aed88aba07163aa5b1be5c6e0ee75da4351496cad062d1e355a261409"),
    (4134, "60dfd1369e0efa83dfa6c7d4035f4d9d66ca6ba9a0ec6b760349d0a0e30d7b34"),
    (5001, "faaecb19a8b681495f02e18493b8ae01c96d022767f25ded19f5cbec9d95cf31"),
    (6040, "585c27309e9130400849315b4bf49d99717f700507239f2bd4dc6fece3cbe894"),
    (7511, "2843c07959515a93d2791183c998462d76b34100185e57a258d8949f112296e7"),
    (10707, "eec1e22dc3ed6ff70d35dd771e05a20024264fdecaf5a187fdad616e6d20e5d6"),
    (13348, "14f222b4cbf6d90e60eb90eb72aebc0f83cebaf150b4c5999f107bb11294fc36"),
    (13923, "c5c32b9999259b6041e92815693b975f8ee2291b29a6577d983685b6d486796f"),
    (14038, "055f28bbd181590b7a7c4844bf488d7f1be752d39916e07034c279f5f387acd2"),
    (14439, "d09e1ef4ec3bcbfbe3e16a2f9b3ff92a743e4dc061eaca4add29f59787567f73"),
    (16228, "5aade4c6e9dbdbf3bd64072e813bae02975a342b84c312efc91983ac63300d49"),
)
EXPECTED_K_SEEDS = tuple(range(21001, 21017))


@dataclass(frozen=True)
class ImageIdentityRecord:
    image_id: int
    image_sha256: str


@dataclass(frozen=True)
class PanelIdentity:
    panel_sha256: str
    ordering: str
    owner_count: int
    images: tuple[ImageIdentityRecord, ...]


@dataclass(frozen=True)
class SourceIdentity:
    checkpoint_path: str
    base_model_path: str
    adapter_sha256: str
    special_embedding_sha256: str


@dataclass(frozen=True)
class SurfaceIdentity:
    prompt_policy_fingerprint: str
    tokenizer_sha256: str
    tokenizer_class: str
    wrapper: str
    parser: str


@dataclass(frozen=True)
class MatcherIdentity:
    algorithm: str
    same_category: bool
    owner_iou_threshold: float
    duplicate_iou_threshold: float
    duplicate_comparison: str
    target_row_rule: str


@dataclass(frozen=True)
class BindingIdentity:
    unit_id: str
    purpose: str
    artifact_root: str
    panel: PanelIdentity
    source: SourceIdentity
    surface: SurfaceIdentity
    matcher: MatcherIdentity


@dataclass(frozen=True)
class RequestIdentity:
    backend: str
    backend_version: str
    mode: Literal["source_greedy", "k_sampled"]
    n: int
    seed: int | None
    physical_batch_index: int
    temperature: float
    top_p: float
    repetition_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class PredictionRowInput:
    row_id: str
    row_index: int
    category: str
    bbox: tuple[float, float, float, float]
    token_start: int
    token_end: int
    final_coordinate_token_index: int
    parser_status: str = "complete"
    geometry_valid: bool = True
    row_terminated: bool = True


@dataclass(frozen=True)
class TrajectoryInput:
    trajectory_id: str
    request: RequestIdentity
    token_ids: tuple[int, ...]
    terminal_token_index: int | None
    stop_reason: str
    parser_status: str
    rows: tuple[PredictionRowInput, ...]


@dataclass(frozen=True)
class OwnerInput:
    owner_id: str
    category: str
    bbox: tuple[float, float, float, float]
    source_object_index: int


@dataclass(frozen=True)
class ImageInput:
    image_id: int
    owners: tuple[OwnerInput, ...]
    source: TrajectoryInput
    sampled: tuple[TrajectoryInput, ...]


@dataclass(frozen=True)
class PrefixRecord:
    raw_token_ids: tuple[int, ...]
    clean_token_ids: tuple[int, ...]
    removed_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateEventRecord:
    event_id: str
    image_id: int
    trajectory_id: str
    duplicate_row_id: str
    retained_row_id: str
    decision_prefix_token_ids: tuple[int, ...]
    target_token_id: int
    target_token_index: int


@dataclass(frozen=True)
class SelectedRowRecord:
    owner_id: str
    row_id: str
    trajectory_id: str
    seed: int
    row_index: int
    owner_iou: float
    token_ids: tuple[int, ...]
    target_token_mask: tuple[bool, ...]


@dataclass(frozen=True)
class OwnerRecord:
    owner_id: str
    category: str
    bbox: tuple[float, float, float, float]
    source_object_index: int
    stratum: Literal["G", "H", "M"]
    source_row_ids: tuple[str, ...]
    sampled_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    request: RequestIdentity
    raw_token_ids: tuple[int, ...]
    terminal_token_index: int | None
    stop_reason: str
    parser_status: str
    prefix: PrefixRecord
    retained_row_ids: tuple[str, ...]
    duplicate_row_ids: tuple[str, ...]
    matched_row_ids: tuple[str, ...]
    replay_token_mask: tuple[bool, ...]
    duplicate_target_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    owners: tuple[OwnerRecord, ...]
    trajectories: tuple[TrajectoryRecord, ...]
    duplicate_events: tuple[DuplicateEventRecord, ...]
    selected_rows: tuple[SelectedRowRecord, ...]
    g_owner_ids: tuple[str, ...]
    h_owner_ids: tuple[str, ...]
    m_owner_ids: tuple[str, ...]
    replay_row_ids: tuple[str, ...]
    target_row_ids: tuple[str, ...]
    candidate_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArmIdentity:
    arm_id: str
    target_scope: str
    terminal_masked: bool


@dataclass(frozen=True)
class GlobalDenominatorIdentity:
    panel_image_count: int
    target_image_count: int
    target_owner_count: int
    replay_image_count: int
    replay_owner_count: int
    duplicate_image_count: int
    duplicate_event_count: int


@dataclass(frozen=True)
class Human13KUnionManifest:
    schema_version: str
    binding: BindingIdentity
    images: tuple[ImageRecord, ...]
    arms: tuple[ArmIdentity, ...]
    denominators: GlobalDenominatorIdentity
    full_panel: bool


def default_binding() -> BindingIdentity:
    """Return the only identity admitted by this experiment-local builder."""

    return BindingIdentity(
        unit_id=UNIT_ID,
        purpose=PURPOSE,
        artifact_root=(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-08-12-human13-k-union-to-greedy-overfit-screen"
        ),
        panel=PanelIdentity(
            panel_sha256=PANEL_SHA256,
            ordering="geo_sorted_xy",
            owner_count=392,
            images=tuple(
                ImageIdentityRecord(*item) for item in EXPECTED_IMAGE_IDENTITIES
            ),
        ),
        source=SourceIdentity(
            checkpoint_path=(
                "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
                "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
                "checkpoints/step-2444"
            ),
            base_model_path=(
                "/data/Qwen3-VL/model_cache/models/Qwen/"
                "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
            ),
            adapter_sha256="49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da",
            special_embedding_sha256="a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2",
        ),
        surface=SurfaceIdentity(
            prompt_policy_fingerprint="0b4fa411f289ccc29e3f6b59c65d32689ac04e6a014891cbe2dc2d976de634c1",
            tokenizer_sha256="ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8",
            tokenizer_class="Qwen2TokenizerFast",
            wrapper="object_box_closed",
            parser="compact_object_box_closed_only",
        ),
        matcher=MatcherIdentity(
            algorithm="cardinality_first_max_total_iou",
            same_category=True,
            owner_iou_threshold=0.50,
            duplicate_iou_threshold=0.95,
            duplicate_comparison="strictly_greater",
            target_row_rule="max_owner_iou_then_seed_then_row_index",
        ),
    )


def _validate_binding(binding: BindingIdentity) -> None:
    expected = default_binding()
    if binding.panel.panel_sha256 != expected.panel.panel_sha256:
        raise ValueError("panel SHA-256 does not match the frozen Human-13 panel")
    if binding != expected:
        raise ValueError("binding does not match the frozen Human-13 overfit identity")


def _arm_identities() -> tuple[ArmIdentity, ...]:
    return tuple(
        ArmIdentity(arm_id, target_scope, terminal_masked=True)
        for arm_id, target_scope in (
            ("frozen_source", "none"),
            ("full_gt_capacity", "GT"),
            ("A0", "none"),
            ("A1", "H"),
            ("A3", "H"),
            ("A4", "H"),
            ("A6", "H"),
            ("A7", "H"),
            ("A8-prime", "H"),
        )
    )


def _validate_request(request: RequestIdentity) -> None:
    if request.mode == "source_greedy":
        expected = ("hf", 1, None, 0, 0.0, 1.0, 1.0, 3084)
    else:
        if request.seed not in EXPECTED_K_SEEDS:
            raise ValueError("sampled request seed is outside 21001..21016")
        expected = (
            "vllm",
            1,
            request.seed,
            (request.seed - 21001) // 4,
            0.4,
            0.95,
            1.10,
            512,
        )
    observed = (
        request.backend,
        request.n,
        request.seed,
        request.physical_batch_index,
        request.temperature,
        request.top_p,
        request.repetition_penalty,
        request.max_new_tokens,
    )
    if observed != expected or not request.backend_version:
        raise ValueError(f"{request.mode} request does not match the frozen recipe")


def _validate_trajectory(trajectory: TrajectoryInput) -> tuple[PredictionRowInput, ...]:
    _validate_request(trajectory.request)
    token_count = len(trajectory.token_ids)
    terminal = trajectory.terminal_token_index
    if terminal is not None and not (0 <= terminal < token_count):
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has an invalid terminal index"
        )
    prefix_end = token_count if terminal is None else terminal
    ordered = tuple(
        sorted(trajectory.rows, key=lambda row: (row.row_index, row.row_id))
    )
    if len({row.row_id for row in ordered}) != len(ordered):
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has duplicate row IDs"
        )
    if len({row.row_index for row in ordered}) != len(ordered):
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has duplicate row indices"
        )
    prior_end = 0
    for row in ordered:
        if not (0 <= row.token_start < row.token_end <= prefix_end):
            raise ValueError(f"row {row.row_id!r} has an invalid token span")
        if row.token_start < prior_end:
            raise ValueError(f"row {row.row_id!r} overlaps an earlier row span")
        if not (row.token_start <= row.final_coordinate_token_index < row.token_end):
            raise ValueError(f"row {row.row_id!r} has an invalid final-coordinate site")
        prior_end = row.token_end
    return ordered


def _eligible_row(row: PredictionRowInput) -> bool:
    return row.parser_status == "complete" and row.geometry_valid and row.row_terminated


@dataclass(frozen=True)
class _TrajectoryBuild:
    record: TrajectoryRecord
    rows: tuple[PredictionRowInput, ...]
    matched_owner_by_row: Mapping[str, tuple[str, float]]
    duplicate_reference_by_row: Mapping[str, str]
    events: tuple[DuplicateEventRecord, ...]


def _build_trajectory(
    *,
    image_id: int,
    trajectory: TrajectoryInput,
    owners: tuple[OwnerInput, ...],
    source_replay: bool,
) -> _TrajectoryBuild:
    rows = _validate_trajectory(trajectory)
    retained: list[PredictionRowInput] = []
    duplicate_reference: dict[str, str] = {}
    for row in rows:
        if not _eligible_row(row):
            continue
        duplicate_of = next(
            (
                earlier.row_id
                for earlier in retained
                if iou_xyxy(earlier.bbox, row.bbox) > 0.95
            ),
            None,
        )
        if duplicate_of is None:
            retained.append(row)
        else:
            duplicate_reference[row.row_id] = duplicate_of

    gt = [(owner.category, owner.bbox) for owner in owners]
    pred = [(row.category, row.bbox) for row in retained]
    matches = _global_matches(gt, pred, 0.50)
    matched_owner_by_row = {
        retained[pred_index].row_id: (owners[owner_index].owner_id, overlap)
        for owner_index, pred_index, overlap in matches
    }

    terminal = trajectory.terminal_token_index
    prefix_end = len(trajectory.token_ids) if terminal is None else terminal
    raw_prefix = trajectory.token_ids[:prefix_end]
    removed_positions: set[int] = set()
    row_by_id = {row.row_id: row for row in rows}
    for row_id in duplicate_reference:
        row = row_by_id[row_id]
        removed_positions.update(range(row.token_start, row.token_end))
    clean_prefix = tuple(
        token
        for index, token in enumerate(trajectory.token_ids[:prefix_end])
        if index not in removed_positions
    )

    replay_mask = [False] * len(trajectory.token_ids)
    if source_replay:
        for row in retained:
            if row.row_id in matched_owner_by_row:
                replay_mask[row.token_start : row.token_end] = [True] * (
                    row.token_end - row.token_start
                )
    duplicate_mask = [False] * len(trajectory.token_ids)
    events: list[DuplicateEventRecord] = []
    for row in rows:
        retained_row_id = duplicate_reference.get(row.row_id)
        if retained_row_id is None:
            continue
        target_index = row.final_coordinate_token_index
        duplicate_mask[target_index] = True
        events.append(
            DuplicateEventRecord(
                event_id=f"dup:{image_id}:{trajectory.trajectory_id}:{row.row_id}",
                image_id=image_id,
                trajectory_id=trajectory.trajectory_id,
                duplicate_row_id=row.row_id,
                retained_row_id=retained_row_id,
                decision_prefix_token_ids=trajectory.token_ids[:target_index],
                target_token_id=trajectory.token_ids[target_index],
                target_token_index=target_index,
            )
        )

    record = TrajectoryRecord(
        trajectory_id=trajectory.trajectory_id,
        request=trajectory.request,
        raw_token_ids=trajectory.token_ids,
        terminal_token_index=terminal,
        stop_reason=trajectory.stop_reason,
        parser_status=trajectory.parser_status,
        prefix=PrefixRecord(
            raw_token_ids=raw_prefix,
            clean_token_ids=clean_prefix,
            removed_row_ids=tuple(
                row.row_id for row in rows if row.row_id in duplicate_reference
            ),
        ),
        retained_row_ids=tuple(row.row_id for row in retained),
        duplicate_row_ids=tuple(
            row.row_id for row in rows if row.row_id in duplicate_reference
        ),
        matched_row_ids=tuple(
            row.row_id for row in retained if row.row_id in matched_owner_by_row
        ),
        replay_token_mask=tuple(replay_mask),
        duplicate_target_mask=tuple(duplicate_mask),
    )
    return _TrajectoryBuild(
        record=record,
        rows=rows,
        matched_owner_by_row=matched_owner_by_row,
        duplicate_reference_by_row=duplicate_reference,
        events=tuple(events),
    )


def _build_image(image: ImageInput) -> ImageRecord:
    if image.image_id not in {item[0] for item in EXPECTED_IMAGE_IDENTITIES}:
        raise ValueError(f"image {image.image_id} is outside the frozen Human-13 panel")
    if len({owner.owner_id for owner in image.owners}) != len(image.owners):
        raise ValueError(f"image {image.image_id} has duplicate owner IDs")
    if image.source.request.mode != "source_greedy":
        raise ValueError(
            f"image {image.image_id} source trajectory is not source_greedy"
        )
    sampled_seeds = tuple(
        trajectory.request.seed
        for trajectory in sorted(
            image.sampled,
            key=lambda item: (
                -1 if item.request.seed is None else item.request.seed,
                item.trajectory_id,
            ),
        )
    )
    if sampled_seeds != EXPECTED_K_SEEDS:
        raise ValueError(f"image {image.image_id} requires exactly seeds 21001..21016")
    if any(item.request.mode != "k_sampled" for item in image.sampled):
        raise ValueError(f"image {image.image_id} contains a non-sampled K trajectory")

    trajectories = (image.source,) + tuple(
        sorted(image.sampled, key=lambda item: (item.request.seed, item.trajectory_id))
    )
    if len({item.trajectory_id for item in trajectories}) != len(trajectories):
        raise ValueError(f"image {image.image_id} has duplicate trajectory IDs")
    all_row_ids = [row.row_id for item in trajectories for row in item.rows]
    if len(set(all_row_ids)) != len(all_row_ids):
        raise ValueError(
            f"image {image.image_id} has row IDs reused across trajectories"
        )

    built = tuple(
        _build_trajectory(
            image_id=image.image_id,
            trajectory=trajectory,
            owners=image.owners,
            source_replay=index == 0,
        )
        for index, trajectory in enumerate(trajectories)
    )
    source_built = built[0]
    sampled_built = built[1:]
    source_owner_ids = {
        owner_id for owner_id, _ in source_built.matched_owner_by_row.values()
    }
    sampled_owner_ids = {
        owner_id
        for trajectory in sampled_built
        for owner_id, _ in trajectory.matched_owner_by_row.values()
    }
    h_owner_ids_set = sampled_owner_ids - source_owner_ids

    occurrence_by_owner: dict[
        str, list[tuple[float, int, int, str, str, tuple[int, ...]]]
    ] = {}
    candidate_occurrences: list[tuple[int, int, str, tuple[int, ...]]] = []
    for trajectory_input, trajectory_build in zip(trajectories[1:], sampled_built):
        seed = trajectory_input.request.seed
        assert seed is not None
        row_by_id = {row.row_id: row for row in trajectory_build.rows}
        for row_id, (
            owner_id,
            overlap,
        ) in trajectory_build.matched_owner_by_row.items():
            if owner_id not in h_owner_ids_set:
                continue
            row = row_by_id[row_id]
            tokens = trajectory_input.token_ids[row.token_start : row.token_end]
            occurrence_by_owner.setdefault(owner_id, []).append(
                (
                    overlap,
                    seed,
                    row.row_index,
                    row_id,
                    trajectory_input.trajectory_id,
                    tokens,
                )
            )
            candidate_occurrences.append((seed, row.row_index, row_id, tokens))

    selected_rows: list[SelectedRowRecord] = []
    for owner_id in sorted(h_owner_ids_set):
        occurrences = occurrence_by_owner[owner_id]
        overlap, seed, row_index, row_id, trajectory_id, tokens = min(
            occurrences,
            key=lambda item: (-item[0], item[1], item[2], item[3]),
        )
        selected_rows.append(
            SelectedRowRecord(
                owner_id=owner_id,
                row_id=row_id,
                trajectory_id=trajectory_id,
                seed=seed,
                row_index=row_index,
                owner_iou=overlap,
                token_ids=tokens,
                target_token_mask=(True,) * len(tokens),
            )
        )

    seen_candidate_tokens: set[tuple[int, ...]] = set()
    candidate_row_ids: list[str] = []
    for _, _, row_id, tokens in sorted(candidate_occurrences):
        if tokens in seen_candidate_tokens:
            continue
        seen_candidate_tokens.add(tokens)
        candidate_row_ids.append(row_id)

    owner_records: list[OwnerRecord] = []
    for owner in sorted(
        image.owners, key=lambda item: (item.source_object_index, item.owner_id)
    ):
        source_rows = tuple(
            row_id
            for row_id, (owner_id, _) in source_built.matched_owner_by_row.items()
            if owner_id == owner.owner_id
        )
        sampled_rows = tuple(
            row_id
            for trajectory in sampled_built
            for row_id, (owner_id, _) in trajectory.matched_owner_by_row.items()
            if owner_id == owner.owner_id
        )
        if owner.owner_id in source_owner_ids:
            stratum: Literal["G", "H", "M"] = "G"
        elif owner.owner_id in sampled_owner_ids:
            stratum = "H"
        else:
            stratum = "M"
        owner_records.append(
            OwnerRecord(
                owner_id=owner.owner_id,
                category=owner.category,
                bbox=owner.bbox,
                source_object_index=owner.source_object_index,
                stratum=stratum,
                source_row_ids=source_rows,
                sampled_row_ids=sampled_rows,
            )
        )

    replay_row_ids = source_built.record.matched_row_ids
    target_row_ids = tuple(item.row_id for item in selected_rows)
    duplicate_row_ids = {
        row_id for trajectory in built for row_id in trajectory.record.duplicate_row_ids
    }
    matched_row_ids = {
        row_id for trajectory in built for row_id in trajectory.record.matched_row_ids
    }
    for label, positive in (
        ("matched", matched_row_ids),
        ("replay", set(replay_row_ids)),
        ("target", set(target_row_ids)),
        ("candidate", set(candidate_row_ids)),
    ):
        overlap = duplicate_row_ids & positive
        if overlap:
            raise ValueError(
                f"duplicate rows intersect the {label} positive set: {sorted(overlap)}"
            )

    return ImageRecord(
        image_id=image.image_id,
        owners=tuple(owner_records),
        trajectories=tuple(item.record for item in built),
        duplicate_events=tuple(event for item in built for event in item.events),
        selected_rows=tuple(selected_rows),
        g_owner_ids=tuple(sorted(source_owner_ids)),
        h_owner_ids=tuple(sorted(h_owner_ids_set)),
        m_owner_ids=tuple(
            sorted(
                owner.owner_id
                for owner in image.owners
                if owner.owner_id not in source_owner_ids | sampled_owner_ids
            )
        ),
        replay_row_ids=replay_row_ids,
        target_row_ids=target_row_ids,
        candidate_row_ids=tuple(candidate_row_ids),
    )


def build_manifest(
    *,
    binding: BindingIdentity,
    images: Sequence[ImageInput],
    require_full_panel: bool = True,
) -> Human13KUnionManifest:
    """Build a manifest; partial inputs are mechanics-only and fail full admission."""

    _validate_binding(binding)
    if len({image.image_id for image in images}) != len(images):
        raise ValueError("manifest input contains duplicate image IDs")
    panel_order = {
        image_id: index for index, (image_id, _) in enumerate(EXPECTED_IMAGE_IDENTITIES)
    }
    image_records = tuple(
        _build_image(image)
        for image in sorted(
            images, key=lambda item: panel_order.get(item.image_id, 10**9)
        )
    )
    full_panel = (
        tuple(item.image_id for item in image_records)
        == tuple(item[0] for item in EXPECTED_IMAGE_IDENTITIES)
        and sum(len(item.owners) for item in image_records) == 392
    )
    if require_full_panel and not full_panel:
        raise ValueError(
            "full Human-13 manifest requires all thirteen images and 392 owners"
        )
    target_images = tuple(item for item in image_records if item.h_owner_ids)
    replay_images = tuple(item for item in image_records if item.g_owner_ids)
    duplicate_images = tuple(item for item in image_records if item.duplicate_events)
    return Human13KUnionManifest(
        schema_version=SCHEMA_VERSION,
        binding=binding,
        images=image_records,
        arms=_arm_identities(),
        denominators=GlobalDenominatorIdentity(
            panel_image_count=13,
            target_image_count=len(target_images),
            target_owner_count=sum(len(item.h_owner_ids) for item in image_records),
            replay_image_count=len(replay_images),
            replay_owner_count=sum(len(item.g_owner_ids) for item in image_records),
            duplicate_image_count=len(duplicate_images),
            duplicate_event_count=sum(
                len(item.duplicate_events) for item in image_records
            ),
        ),
        full_panel=full_panel,
    )


def _canonical_bytes(value: Human13KUnionManifest) -> bytes:
    return (
        json.dumps(
            asdict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_write(value: Human13KUnionManifest, path: str | Path) -> str:
    """Write one canonical manifest and its single adjacent SHA-256 digest."""

    target = Path(path)
    digest_path = Path(f"{target}.sha256")
    if target.exists() or digest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest receipt: {target}")
    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return digest


def _require_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} keys do not match the manifest schema")


def _binding_from_dict(value: Mapping[str, Any]) -> BindingIdentity:
    _require_keys(
        value,
        {
            "unit_id",
            "purpose",
            "artifact_root",
            "panel",
            "source",
            "surface",
            "matcher",
        },
        field="binding",
    )
    panel = value["panel"]
    source = value["source"]
    surface = value["surface"]
    matcher = value["matcher"]
    if not all(isinstance(item, Mapping) for item in (panel, source, surface, matcher)):
        raise ValueError("binding identities must be JSON objects")
    return BindingIdentity(
        unit_id=str(value["unit_id"]),
        purpose=str(value["purpose"]),
        artifact_root=str(value["artifact_root"]),
        panel=PanelIdentity(
            panel_sha256=str(panel["panel_sha256"]),
            ordering=str(panel["ordering"]),
            owner_count=int(panel["owner_count"]),
            images=tuple(
                ImageIdentityRecord(int(item["image_id"]), str(item["image_sha256"]))
                for item in panel["images"]
            ),
        ),
        source=SourceIdentity(
            **{key: str(source[key]) for key in SourceIdentity.__dataclass_fields__}
        ),
        surface=SurfaceIdentity(
            **{key: str(surface[key]) for key in SurfaceIdentity.__dataclass_fields__}
        ),
        matcher=MatcherIdentity(
            algorithm=str(matcher["algorithm"]),
            same_category=bool(matcher["same_category"]),
            owner_iou_threshold=float(matcher["owner_iou_threshold"]),
            duplicate_iou_threshold=float(matcher["duplicate_iou_threshold"]),
            duplicate_comparison=str(matcher["duplicate_comparison"]),
            target_row_rule=str(matcher["target_row_rule"]),
        ),
    )


def _request_from_dict(value: Mapping[str, Any]) -> RequestIdentity:
    _require_keys(value, set(RequestIdentity.__dataclass_fields__), field="request")
    request = RequestIdentity(
        backend=str(value["backend"]),
        backend_version=str(value["backend_version"]),
        mode=str(value["mode"]),  # type: ignore[arg-type]
        n=int(value["n"]),
        seed=None if value["seed"] is None else int(value["seed"]),
        physical_batch_index=int(value["physical_batch_index"]),
        temperature=float(value["temperature"]),
        top_p=float(value["top_p"]),
        repetition_penalty=float(value["repetition_penalty"]),
        max_new_tokens=int(value["max_new_tokens"]),
    )
    _validate_request(request)
    return request


def _prefix_from_dict(value: Mapping[str, Any]) -> PrefixRecord:
    _require_keys(value, set(PrefixRecord.__dataclass_fields__), field="prefix")
    return PrefixRecord(
        raw_token_ids=tuple(int(item) for item in value["raw_token_ids"]),
        clean_token_ids=tuple(int(item) for item in value["clean_token_ids"]),
        removed_row_ids=tuple(str(item) for item in value["removed_row_ids"]),
    )


def _trajectory_record_from_dict(value: Mapping[str, Any]) -> TrajectoryRecord:
    _require_keys(value, set(TrajectoryRecord.__dataclass_fields__), field="trajectory")
    terminal = value["terminal_token_index"]
    return TrajectoryRecord(
        trajectory_id=str(value["trajectory_id"]),
        request=_request_from_dict(value["request"]),
        raw_token_ids=tuple(int(item) for item in value["raw_token_ids"]),
        terminal_token_index=None if terminal is None else int(terminal),
        stop_reason=str(value["stop_reason"]),
        parser_status=str(value["parser_status"]),
        prefix=_prefix_from_dict(value["prefix"]),
        retained_row_ids=tuple(str(item) for item in value["retained_row_ids"]),
        duplicate_row_ids=tuple(str(item) for item in value["duplicate_row_ids"]),
        matched_row_ids=tuple(str(item) for item in value["matched_row_ids"]),
        replay_token_mask=tuple(bool(item) for item in value["replay_token_mask"]),
        duplicate_target_mask=tuple(
            bool(item) for item in value["duplicate_target_mask"]
        ),
    )


def _owner_record_from_dict(value: Mapping[str, Any]) -> OwnerRecord:
    _require_keys(value, set(OwnerRecord.__dataclass_fields__), field="owner")
    return OwnerRecord(
        owner_id=str(value["owner_id"]),
        category=str(value["category"]),
        bbox=tuple(float(item) for item in value["bbox"]),  # type: ignore[arg-type]
        source_object_index=int(value["source_object_index"]),
        stratum=str(value["stratum"]),  # type: ignore[arg-type]
        source_row_ids=tuple(str(item) for item in value["source_row_ids"]),
        sampled_row_ids=tuple(str(item) for item in value["sampled_row_ids"]),
    )


def _event_from_dict(value: Mapping[str, Any]) -> DuplicateEventRecord:
    _require_keys(
        value, set(DuplicateEventRecord.__dataclass_fields__), field="duplicate_event"
    )
    return DuplicateEventRecord(
        event_id=str(value["event_id"]),
        image_id=int(value["image_id"]),
        trajectory_id=str(value["trajectory_id"]),
        duplicate_row_id=str(value["duplicate_row_id"]),
        retained_row_id=str(value["retained_row_id"]),
        decision_prefix_token_ids=tuple(
            int(item) for item in value["decision_prefix_token_ids"]
        ),
        target_token_id=int(value["target_token_id"]),
        target_token_index=int(value["target_token_index"]),
    )


def _selected_row_from_dict(value: Mapping[str, Any]) -> SelectedRowRecord:
    _require_keys(
        value, set(SelectedRowRecord.__dataclass_fields__), field="selected_row"
    )
    return SelectedRowRecord(
        owner_id=str(value["owner_id"]),
        row_id=str(value["row_id"]),
        trajectory_id=str(value["trajectory_id"]),
        seed=int(value["seed"]),
        row_index=int(value["row_index"]),
        owner_iou=float(value["owner_iou"]),
        token_ids=tuple(int(item) for item in value["token_ids"]),
        target_token_mask=tuple(bool(item) for item in value["target_token_mask"]),
    )


def _image_record_from_dict(value: Mapping[str, Any]) -> ImageRecord:
    _require_keys(value, set(ImageRecord.__dataclass_fields__), field="image")
    return ImageRecord(
        image_id=int(value["image_id"]),
        owners=tuple(_owner_record_from_dict(item) for item in value["owners"]),
        trajectories=tuple(
            _trajectory_record_from_dict(item) for item in value["trajectories"]
        ),
        duplicate_events=tuple(
            _event_from_dict(item) for item in value["duplicate_events"]
        ),
        selected_rows=tuple(
            _selected_row_from_dict(item) for item in value["selected_rows"]
        ),
        g_owner_ids=tuple(str(item) for item in value["g_owner_ids"]),
        h_owner_ids=tuple(str(item) for item in value["h_owner_ids"]),
        m_owner_ids=tuple(str(item) for item in value["m_owner_ids"]),
        replay_row_ids=tuple(str(item) for item in value["replay_row_ids"]),
        target_row_ids=tuple(str(item) for item in value["target_row_ids"]),
        candidate_row_ids=tuple(str(item) for item in value["candidate_row_ids"]),
    )


def _validate_loaded_manifest(
    value: Human13KUnionManifest, *, require_full_panel: bool
) -> None:
    _validate_binding(value.binding)
    if value.arms != _arm_identities():
        raise ValueError("arm identities do not match the frozen Human-13 matrix")
    expected_ids = tuple(item[0] for item in EXPECTED_IMAGE_IDENTITIES)
    panel_order = {image_id: index for index, image_id in enumerate(expected_ids)}
    observed_ids = tuple(item.image_id for item in value.images)
    if len(set(observed_ids)) != len(observed_ids) or any(
        image_id not in panel_order for image_id in observed_ids
    ):
        raise ValueError("manifest image identities are not a Human-13 subset")
    if observed_ids != tuple(sorted(observed_ids, key=panel_order.__getitem__)):
        raise ValueError("manifest image identities are not in canonical panel order")
    for image in value.images:
        if (
            not image.trajectories
            or image.trajectories[0].request.mode != "source_greedy"
        ):
            raise ValueError(f"image {image.image_id} is missing its Source trajectory")
        sampled_seeds = tuple(
            trajectory.request.seed for trajectory in image.trajectories[1:]
        )
        if sampled_seeds != EXPECTED_K_SEEDS:
            raise ValueError(
                f"image {image.image_id} requires exactly seeds 21001..21016"
            )
        duplicate_rows = {
            row_id
            for trajectory in image.trajectories
            for row_id in trajectory.duplicate_row_ids
        }
        matched_rows = {
            row_id
            for trajectory in image.trajectories
            for row_id in trajectory.matched_row_ids
        }
        for positive in (
            matched_rows,
            set(image.replay_row_ids),
            set(image.target_row_ids),
            set(image.candidate_row_ids),
        ):
            if duplicate_rows & positive:
                raise ValueError("loaded duplicate row intersects a positive set")
        for trajectory in image.trajectories:
            if len(trajectory.replay_token_mask) != len(trajectory.raw_token_ids):
                raise ValueError("replay mask length does not match raw token IDs")
            if len(trajectory.duplicate_target_mask) != len(trajectory.raw_token_ids):
                raise ValueError("duplicate mask length does not match raw token IDs")
            terminal = trajectory.terminal_token_index
            if terminal is not None and (
                trajectory.replay_token_mask[terminal]
                or trajectory.duplicate_target_mask[terminal]
            ):
                raise ValueError("chat terminal is licensed as a target")
    expected_full = (
        observed_ids == expected_ids
        and sum(len(item.owners) for item in value.images) == 392
    )
    if value.full_panel != expected_full:
        raise ValueError("full_panel does not match the frozen panel contents")
    expected_denominators = GlobalDenominatorIdentity(
        panel_image_count=13,
        target_image_count=sum(bool(item.h_owner_ids) for item in value.images),
        target_owner_count=sum(len(item.h_owner_ids) for item in value.images),
        replay_image_count=sum(bool(item.g_owner_ids) for item in value.images),
        replay_owner_count=sum(len(item.g_owner_ids) for item in value.images),
        duplicate_image_count=sum(bool(item.duplicate_events) for item in value.images),
        duplicate_event_count=sum(len(item.duplicate_events) for item in value.images),
    )
    if value.denominators != expected_denominators:
        raise ValueError("global denominator counts do not match manifest records")
    if require_full_panel and not value.full_panel:
        raise ValueError("partial Human-13 manifest is mechanics-only")


def load_manifest(
    path: str | Path, *, require_full_panel: bool = True
) -> Human13KUnionManifest:
    """Load and strictly validate a canonical manifest written by this module."""

    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if not isinstance(document, Mapping):
        raise ValueError("manifest root must be an object")
    _require_keys(
        document,
        {"schema_version", "binding", "images", "arms", "denominators", "full_panel"},
        field="manifest",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version does not match")
    binding = _binding_from_dict(document["binding"])
    arms = tuple(ArmIdentity(**item) for item in document["arms"])
    denominators = GlobalDenominatorIdentity(**document["denominators"])
    result = Human13KUnionManifest(
        schema_version=SCHEMA_VERSION,
        binding=binding,
        images=tuple(_image_record_from_dict(item) for item in document["images"]),
        arms=arms,
        denominators=denominators,
        full_panel=bool(document["full_panel"]),
    )
    if raw != _canonical_bytes(result):
        raise ValueError("manifest is not canonically serialized")
    digest_path = Path(f"{path}.sha256")
    if not digest_path.is_file():
        raise ValueError("manifest digest is missing")
    digest = hashlib.sha256(raw).hexdigest()
    expected_digest_receipt = f"{digest}  {Path(path).name}\n"
    if digest_path.read_text(encoding="ascii") != expected_digest_receipt:
        raise ValueError("manifest digest does not match canonical bytes")
    _validate_loaded_manifest(result, require_full_panel=require_full_panel)
    return result


def dry_run_summary(value: Human13KUnionManifest) -> dict[str, Any]:
    """Return a plan-only summary with an explicit zero-action receipt."""

    _validate_binding(value.binding)
    return {
        "unit_id": value.binding.unit_id,
        "purpose": value.binding.purpose,
        "panel_sha256": value.binding.panel.panel_sha256,
        "declared_image_count": len(value.binding.panel.images),
        "materialized_image_count": len(value.images),
        "full_panel": value.full_panel,
        "actions": {
            "model_load": 0,
            "decode": 0,
            "forward": 0,
            "backward": 0,
            "optimizer_mutation": 0,
            "checkpoint_write": 0,
            "gpu_allocation": 0,
        },
    }


__all__ = [
    "BindingIdentity",
    "DuplicateEventRecord",
    "GlobalDenominatorIdentity",
    "Human13KUnionManifest",
    "ImageInput",
    "ImageRecord",
    "OwnerInput",
    "OwnerRecord",
    "PredictionRowInput",
    "PrefixRecord",
    "RequestIdentity",
    "SelectedRowRecord",
    "TrajectoryInput",
    "TrajectoryRecord",
    "build_manifest",
    "canonical_write",
    "default_binding",
    "dry_run_summary",
    "load_manifest",
]
