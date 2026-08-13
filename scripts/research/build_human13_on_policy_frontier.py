"""Materialize one immutable accepted-decode frontier for Human-13."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.build_human13_k_union_manifest import (
    Human13KUnionManifest,
    ImageRecord,
    load_manifest,
)
from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    iou_xyxy,
)
from src.eval.detection_categories import normalize_coco_category_name


SCHEMA_VERSION = "human13_on_policy_frontier.v1"
_DUPLICATE_IOU = 0.95
_OWNER_IOU = 0.5
_AUTHORIZED_PARSER = "compact_object_box_closed_only"


@dataclass(frozen=True)
class CheckpointIdentity:
    path: str
    payload_sha256: str


@dataclass(frozen=True)
class CurrentPrediction:
    generated_order: int
    category: str
    bbox: tuple[float, float, float, float]
    token_start: int
    token_end: int


@dataclass(frozen=True)
class CurrentDecode:
    image_id: int
    trajectory_id: str
    generated_token_ids: tuple[int, ...]
    predictions: tuple[CurrentPrediction, ...]
    parser: str
    parser_status: str
    stop_reason: str
    checkpoint: CheckpointIdentity


@dataclass(frozen=True)
class FrontierRow:
    generated_order: int
    category: str
    bbox: tuple[float, float, float, float]
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class FrontierCandidateAlias:
    owner_id: str
    row_id: str
    trajectory_id: str
    seed: int
    owner_iou: float
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class FrontierDuplicateEvent:
    duplicate_generated_order: int
    retained_generated_order: int
    prediction_to_prediction_iou: float


@dataclass(frozen=True)
class FrontierImage:
    image_id: int
    trajectory_id: str
    generated_token_ids: tuple[int, ...]
    parser: str
    parser_status: str
    stop_reason: str
    rows: tuple[FrontierRow, ...]
    canonical_owner_ids: tuple[str, ...]
    constrained_protected_owner_ids: tuple[str, ...]
    covered_h_owner_ids: tuple[str, ...]
    uncovered_h_owner_ids: tuple[str, ...]
    candidate_aliases: tuple[FrontierCandidateAlias, ...]
    duplicate_events: tuple[FrontierDuplicateEvent, ...]

    @property
    def candidate_owner_ids(self) -> tuple[str, ...]:
        return tuple(sorted({alias.owner_id for alias in self.candidate_aliases}))


@dataclass(frozen=True)
class Human13FrontierIteration:
    schema_version: str
    manifest_path: str
    manifest_sha256: str
    panel_sha256: str
    iteration: int
    checkpoint: CheckpointIdentity
    previous_frontier_path: str | None
    previous_frontier_sha256: str | None
    protected_owner_ids: tuple[str, ...]
    protected_owner_ages: tuple[tuple[str, int], ...]
    images: tuple[FrontierImage, ...]


def build_frontier_iteration(
    manifest: Human13KUnionManifest,
    *,
    manifest_path: str | Path,
    iteration: int,
    checkpoint: CheckpointIdentity,
    decodes: Sequence[CurrentDecode],
    previous: Human13FrontierIteration | None = None,
    previous_path: str | Path | None = None,
) -> Human13FrontierIteration:
    """Project one accepted natural panel decode into a sealed frontier."""

    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    _validate_checkpoint(checkpoint)
    target = Path(manifest_path).resolve(strict=True)
    manifest_sha256 = _verify_adjacent_digest(target, "manifest")
    bound_manifest = load_manifest(target, require_full_panel=manifest.full_panel)
    if bound_manifest != manifest:
        raise ValueError("caller manifest differs from the bound manifest path")
    if previous is not None:
        if iteration != previous.iteration + 1:
            raise ValueError("frontier iteration does not follow previous")
        if previous.manifest_sha256 != manifest_sha256:
            raise ValueError("previous frontier manifest differs")
        if previous_path is None:
            raise ValueError("a previous frontier path is required")
        previous_target = Path(previous_path).resolve(strict=True)
        loaded_previous = load_frontier_iteration(previous_target)
        if loaded_previous != previous:
            raise ValueError("previous frontier object differs from bound receipt")
        previous_sha256 = _verify_adjacent_digest(previous_target, "frontier")
    elif iteration != 0:
        raise ValueError("a nonzero frontier requires previous")
    elif previous_path is not None:
        raise ValueError("iteration zero cannot bind a previous frontier")
    else:
        previous_target = None
        previous_sha256 = None

    manifest_images = {image.image_id: image for image in bound_manifest.images}
    decode_images = {decode.image_id: decode for decode in decodes}
    if len(decode_images) != len(decodes) or set(decode_images) != set(manifest_images):
        raise ValueError("decodes must cover every manifest image exactly once")

    prior_ages = dict(previous.protected_owner_ages) if previous is not None else {}
    visible: set[str] = set()
    projected: list[FrontierImage] = []
    for image_id in sorted(manifest_images):
        image = manifest_images[image_id]
        decode = decode_images[image_id]
        if decode.checkpoint != checkpoint:
            raise ValueError("decode checkpoint identity does not match frontier")
        projected_image = _project_image(image, decode, protected=set(prior_ages))
        projected.append(projected_image)
        visible.update(projected_image.canonical_owner_ids)

    protected, ages = _derive_protection(
        bound_manifest, visible=visible, prior_ages=prior_ages
    )
    protected_set = set(protected)
    projected = [_with_constrained_protected(item, protected_set) for item in projected]
    result = Human13FrontierIteration(
        schema_version=SCHEMA_VERSION,
        manifest_path=str(target),
        manifest_sha256=manifest_sha256,
        panel_sha256=bound_manifest.binding.panel.panel_sha256,
        iteration=iteration,
        checkpoint=checkpoint,
        previous_frontier_path=(
            None if previous_target is None else str(previous_target)
        ),
        previous_frontier_sha256=previous_sha256,
        protected_owner_ids=protected,
        protected_owner_ages=tuple(sorted(ages.items())),
        images=tuple(projected),
    )
    validate_frontier_iteration(result, manifest=bound_manifest)
    return result


def _project_image(
    image: ImageRecord, decode: CurrentDecode, *, protected: set[str]
) -> FrontierImage:
    if decode.parser != _AUTHORIZED_PARSER or decode.parser_status != "complete":
        raise ValueError("current decode parser identity or status is not authorized")
    rows = _rows(decode)
    retained: list[FrontierRow] = []
    duplicates: list[FrontierDuplicateEvent] = []
    for row in rows:
        duplicate = next(
            (
                (earlier, overlap)
                for earlier in retained
                if (overlap := iou_xyxy(earlier.bbox, row.bbox)) > _DUPLICATE_IOU
            ),
            None,
        )
        if duplicate is not None:
            earlier, overlap = duplicate
            duplicates.append(
                FrontierDuplicateEvent(
                    duplicate_generated_order=row.generated_order,
                    retained_generated_order=earlier.generated_order,
                    prediction_to_prediction_iou=float(overlap),
                )
            )
        else:
            retained.append(row)

    owners = sorted(
        image.owners, key=lambda owner: (owner.source_object_index, owner.owner_id)
    )
    gt = [
        (normalize_coco_category_name(owner.category), owner.bbox) for owner in owners
    ]
    pred = [(normalize_coco_category_name(row.category), row.bbox) for row in retained]
    matches = _global_matches(gt, pred, _OWNER_IOU)
    canonical = tuple(sorted(owners[gt_index].owner_id for gt_index, _, _ in matches))
    canonical_set = set(canonical)
    covered_h = tuple(sorted(canonical_set & set(image.h_owner_ids)))
    uncovered_h = tuple(sorted(set(image.h_owner_ids) - canonical_set))
    aliases = tuple(
        sorted(
            (
                FrontierCandidateAlias(
                    owner_id=row.owner_id,
                    row_id=row.row_id,
                    trajectory_id=row.trajectory_id,
                    seed=row.seed,
                    owner_iou=row.owner_iou,
                    token_ids=row.token_ids,
                )
                for row in image.selected_rows
                if row.owner_id in uncovered_h
            ),
            key=lambda item: (item.owner_id, -item.owner_iou, item.seed, item.row_id),
        )
    )
    return FrontierImage(
        image_id=image.image_id,
        trajectory_id=decode.trajectory_id,
        generated_token_ids=decode.generated_token_ids,
        parser=decode.parser,
        parser_status=decode.parser_status,
        stop_reason=decode.stop_reason,
        rows=rows,
        canonical_owner_ids=canonical,
        constrained_protected_owner_ids=tuple(sorted(canonical_set & protected)),
        covered_h_owner_ids=covered_h,
        uncovered_h_owner_ids=uncovered_h,
        candidate_aliases=aliases,
        duplicate_events=tuple(duplicates),
    )


def _with_constrained_protected(
    image: FrontierImage, protected: set[str]
) -> FrontierImage:
    from dataclasses import replace

    return replace(
        image,
        constrained_protected_owner_ids=tuple(
            sorted(set(image.canonical_owner_ids) & protected)
        ),
    )


def _rows(decode: CurrentDecode) -> tuple[FrontierRow, ...]:
    rows: list[FrontierRow] = []
    prior_end = 0
    seen_orders: set[int] = set()
    for prediction in sorted(decode.predictions, key=lambda item: item.generated_order):
        if prediction.generated_order in seen_orders or prediction.generated_order < 0:
            raise ValueError("generated row order is invalid")
        seen_orders.add(prediction.generated_order)
        if (
            prediction.token_start < 0
            or prediction.token_end <= prediction.token_start
            or prediction.token_start < prior_end
            or prediction.token_end > len(decode.generated_token_ids) - 1
        ):
            raise ValueError("prediction token span is invalid")
        if (
            len(prediction.bbox) != 4
            or not prediction.category
            or any(not math.isfinite(float(value)) for value in prediction.bbox)
        ):
            raise ValueError("prediction bbox is not finite or row is invalid")
        prior_end = prediction.token_end
        rows.append(
            FrontierRow(
                generated_order=prediction.generated_order,
                category=prediction.category,
                bbox=tuple(float(value) for value in prediction.bbox),
                token_start=prediction.token_start,
                token_end=prediction.token_end,
                token_ids=decode.generated_token_ids[
                    prediction.token_start : prediction.token_end
                ],
            )
        )
    return tuple(rows)


def validate_frontier_iteration(
    frontier: Human13FrontierIteration, *, manifest: Human13KUnionManifest
) -> None:
    if frontier.schema_version != SCHEMA_VERSION:
        raise ValueError("frontier schema version does not match")
    _validate_checkpoint(frontier.checkpoint)
    if frontier.panel_sha256 != manifest.binding.panel.panel_sha256:
        raise ValueError("frontier panel identity does not match manifest")
    if tuple(image.image_id for image in frontier.images) != tuple(
        sorted(image.image_id for image in manifest.images)
    ):
        raise ValueError("frontier image order or coverage differs from manifest")
    manifest_images = {image.image_id: image for image in manifest.images}
    for image in frontier.images:
        reconstructed = CurrentDecode(
            image_id=image.image_id,
            trajectory_id=image.trajectory_id,
            generated_token_ids=image.generated_token_ids,
            predictions=tuple(
                CurrentPrediction(
                    generated_order=row.generated_order,
                    category=row.category,
                    bbox=row.bbox,
                    token_start=row.token_start,
                    token_end=row.token_end,
                )
                for row in image.rows
            ),
            parser=image.parser,
            parser_status=image.parser_status,
            stop_reason=image.stop_reason,
            checkpoint=frontier.checkpoint,
        )
        derived = _project_image(
            manifest_images[image.image_id],
            reconstructed,
            protected=set(frontier.protected_owner_ids),
        )
        if derived != image:
            raise ValueError("frontier image semantic fields do not rederive")
    m_ids = {owner_id for image in manifest.images for owner_id in image.m_owner_ids}
    if any(
        alias.owner_id in m_ids
        for image in frontier.images
        for alias in image.candidate_aliases
    ):
        raise ValueError("K-miss owner entered frontier candidate aliases")
    if set(frontier.protected_owner_ids) - set(dict(frontier.protected_owner_ages)):
        raise ValueError("protected owner lacks an age")
    if frontier.iteration == 0:
        if (
            frontier.previous_frontier_path is not None
            or frontier.previous_frontier_sha256 is not None
        ):
            raise ValueError("iteration zero has a previous frontier")
        prior_ages: dict[str, int] = {}
    else:
        if (
            frontier.previous_frontier_path is None
            or frontier.previous_frontier_sha256 is None
        ):
            raise ValueError("frontier previous receipt is missing")
        previous_target = Path(frontier.previous_frontier_path).resolve(strict=True)
        if (
            _verify_adjacent_digest(previous_target, "frontier")
            != frontier.previous_frontier_sha256
        ):
            raise ValueError("frontier previous receipt digest differs")
        previous = load_frontier_iteration(previous_target)
        if previous.iteration + 1 != frontier.iteration:
            raise ValueError("frontier previous iteration is not adjacent")
        prior_ages = dict(previous.protected_owner_ages)
    visible = {
        owner_id for image in frontier.images for owner_id in image.canonical_owner_ids
    }
    expected_protected, expected_ages = _derive_protection(
        manifest, visible=visible, prior_ages=prior_ages
    )
    if (
        frontier.protected_owner_ids != expected_protected
        or frontier.protected_owner_ages != tuple(sorted(expected_ages.items()))
    ):
        raise ValueError("frontier protection ages do not rederive")


def canonical_write(frontier: Human13FrontierIteration, path: str | Path) -> str:
    manifest = _load_bound_manifest(frontier)
    validate_frontier_iteration(frontier, manifest=manifest)
    target = Path(path)
    digest_path = Path(f"{target}.sha256")
    if target.exists() or digest_path.exists():
        raise FileExistsError(f"refusing to overwrite frontier receipt: {target}")
    payload = _canonical_bytes(frontier)
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return digest


def load_frontier_iteration(path: str | Path) -> Human13FrontierIteration:
    target = Path(path).resolve(strict=True)
    raw = target.read_bytes()
    digest_path = Path(f"{target}.sha256")
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not digest_path.is_file()
        or digest_path.read_text(encoding="ascii") != f"{digest}  {target.name}\n"
    ):
        raise ValueError("frontier digest does not match canonical bytes")
    document = json.loads(raw)
    result = _from_dict(document)
    if raw != _canonical_bytes(result):
        raise ValueError("frontier is not canonically serialized")
    manifest = _load_bound_manifest(result)
    if (
        _verify_adjacent_digest(Path(result.manifest_path), "manifest")
        != result.manifest_sha256
    ):
        raise ValueError("frontier manifest digest does not match live manifest")
    validate_frontier_iteration(result, manifest=manifest)
    return result


def _from_dict(document: Mapping[str, Any]) -> Human13FrontierIteration:
    def row(item: Mapping[str, Any]) -> FrontierRow:
        return FrontierRow(
            generated_order=int(item["generated_order"]),
            category=str(item["category"]),
            bbox=tuple(float(value) for value in item["bbox"]),  # type: ignore[arg-type]
            token_start=int(item["token_start"]),
            token_end=int(item["token_end"]),
            token_ids=tuple(int(value) for value in item["token_ids"]),
        )

    def image(item: Mapping[str, Any]) -> FrontierImage:
        return FrontierImage(
            image_id=int(item["image_id"]),
            trajectory_id=str(item["trajectory_id"]),
            generated_token_ids=tuple(
                int(value) for value in item["generated_token_ids"]
            ),
            parser=str(item["parser"]),
            parser_status=str(item["parser_status"]),
            stop_reason=str(item["stop_reason"]),
            rows=tuple(row(value) for value in item["rows"]),
            canonical_owner_ids=tuple(
                str(value) for value in item["canonical_owner_ids"]
            ),
            constrained_protected_owner_ids=tuple(
                str(value) for value in item["constrained_protected_owner_ids"]
            ),
            covered_h_owner_ids=tuple(
                str(value) for value in item["covered_h_owner_ids"]
            ),
            uncovered_h_owner_ids=tuple(
                str(value) for value in item["uncovered_h_owner_ids"]
            ),
            candidate_aliases=tuple(
                FrontierCandidateAlias(
                    owner_id=str(value["owner_id"]),
                    row_id=str(value["row_id"]),
                    trajectory_id=str(value["trajectory_id"]),
                    seed=int(value["seed"]),
                    owner_iou=float(value["owner_iou"]),
                    token_ids=tuple(int(token) for token in value["token_ids"]),
                )
                for value in item["candidate_aliases"]
            ),
            duplicate_events=tuple(
                FrontierDuplicateEvent(**value) for value in item["duplicate_events"]
            ),
        )

    checkpoint = CheckpointIdentity(**document["checkpoint"])
    return Human13FrontierIteration(
        schema_version=str(document["schema_version"]),
        manifest_path=str(document["manifest_path"]),
        manifest_sha256=str(document["manifest_sha256"]),
        panel_sha256=str(document["panel_sha256"]),
        iteration=int(document["iteration"]),
        checkpoint=checkpoint,
        previous_frontier_path=(
            None
            if document["previous_frontier_path"] is None
            else str(document["previous_frontier_path"])
        ),
        previous_frontier_sha256=(
            None
            if document["previous_frontier_sha256"] is None
            else str(document["previous_frontier_sha256"])
        ),
        protected_owner_ids=tuple(
            str(value) for value in document["protected_owner_ids"]
        ),
        protected_owner_ages=tuple(
            (str(value[0]), int(value[1])) for value in document["protected_owner_ages"]
        ),
        images=tuple(image(value) for value in document["images"]),
    )


def _load_bound_manifest(frontier: Human13FrontierIteration) -> Human13KUnionManifest:
    return load_manifest(frontier.manifest_path, require_full_panel=False)


def _derive_protection(
    manifest: Human13KUnionManifest,
    *,
    visible: set[str],
    prior_ages: Mapping[str, int],
) -> tuple[tuple[str, ...], dict[str, int]]:
    all_g = {owner_id for image in manifest.images for owner_id in image.g_owner_ids}
    all_h = {owner_id for image in manifest.images for owner_id in image.h_owner_ids}
    ages = {
        owner_id: int(prior_ages.get(owner_id, 0)) + 1
        for owner_id in sorted((all_g | all_h) & visible)
    }
    protected = tuple(
        sorted(
            owner_id for owner_id, age in ages.items() if owner_id in all_g or age >= 2
        )
    )
    return protected, ages


def _validate_checkpoint(checkpoint: CheckpointIdentity) -> None:
    if not checkpoint.path or len(checkpoint.payload_sha256) != 64:
        raise ValueError("checkpoint identity is invalid")
    try:
        int(checkpoint.payload_sha256, 16)
    except ValueError as exc:
        raise ValueError("checkpoint digest is invalid") from exc


def _verify_adjacent_digest(path: Path, label: str) -> str:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = Path(f"{path}.sha256")
    if (
        not receipt.is_file()
        or receipt.read_text(encoding="ascii") != f"{digest}  {path.name}\n"
    ):
        raise ValueError(f"{label} digest does not match canonical bytes")
    return digest


def _canonical_bytes(frontier: Human13FrontierIteration) -> bytes:
    return (
        json.dumps(
            asdict(frontier),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "CheckpointIdentity",
    "CurrentDecode",
    "CurrentPrediction",
    "FrontierCandidateAlias",
    "FrontierDuplicateEvent",
    "FrontierImage",
    "FrontierRow",
    "Human13FrontierIteration",
    "build_frontier_iteration",
    "canonical_write",
    "load_frontier_iteration",
    "validate_frontier_iteration",
]
