#!/usr/bin/env python3
"""Bind sealed Human-13 discovery artifacts to the no-update census.

This experiment-local adapter only materializes native rows and exact logit
sites.  It never loads a model, runs a forward, allocates a GPU, or invents a
logit vector.  Once a caller supplies complete content-bound observed logits,
the statistical calculation is delegated unchanged to
``census_human13_k_union_trie.run_no_update_census``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.build_human13_k_union_manifest import (  # noqa: E402
    PANEL_PATH,
    Human13KUnionManifest,
    build_manifest,
    default_binding,
    load_manifest,
)
from scripts.research.census_human13_k_union_trie import (  # noqa: E402
    CoherentChainSite,
    SelectedNativeRow,
    build_exact_token_trie,
    run_no_update_census,
)
from scripts.research.collect_human13_discovery import (  # noqa: E402
    RECORDS_NAME,
    artifacts_to_image_inputs,
)


SCHEMA_VERSION = "human13_no_update_census_plan.v1"
CAPTURE_KIND = "observed_forward"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COORD = re.compile(r"<\|coord_[0-9]+\|>")
_ACTIONS = {
    "model_imports": 0,
    "model_loads": 0,
    "forwards": 0,
    "gpu_allocations": 0,
    "artifact_writes": 0,
}


@dataclass(frozen=True)
class DiscoveryRecord:
    """Exact token evidence retained by a validated discovery record."""

    image_id: int
    trajectory_id: str
    prompt_token_ids_sha256: str
    chat_text_sha256: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]


@dataclass(frozen=True)
class NativeRowPlan:
    image_id: int
    owner_id: str
    row_id: str
    trajectory_id: str
    row_ordinal: int
    token_ids: tuple[int, ...]
    token_roles: tuple[str, ...]


@dataclass(frozen=True)
class A1SegmentPlan:
    segment_id: str
    segment_role: Literal["a1_full_h"]
    image_id: int
    prompt_token_ids_sha256: str
    chat_text_sha256: str
    fixed_prefix_token_ids: tuple[int, ...]
    row_ids: tuple[str, ...]
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CoherentSitePlan:
    site_id: str
    segment_id: str
    segment_role: Literal["a1_full_h"]
    image_id: int
    owner_id: str
    row_id: str
    row_ordinal: int
    token_offset: int
    segment_token_offset: int
    segment_prefix_token_ids: tuple[int, ...]
    target_token_id: int
    token_role: str
    prompt_token_ids_sha256: str
    chat_text_sha256: str


@dataclass(frozen=True)
class TrieSitePlan:
    site_id: str
    segment_id: str
    segment_role: Literal["native_row_trie"]
    image_id: int
    prefix_token_ids: tuple[int, ...]
    model_prefix_token_ids: tuple[int, ...]
    viable_child_token_ids: tuple[int, ...]
    token_roles_by_target: tuple[tuple[int, str], ...]
    prompt_token_ids_sha256: str
    chat_text_sha256: str


@dataclass(frozen=True)
class Human13CensusPlan:
    schema_version: str
    manifest_sha256: str
    discovery_binding_sha256: str
    selected_rows: tuple[NativeRowPlan, ...]
    segments: tuple[A1SegmentPlan, ...]
    coherent_sites: tuple[CoherentSitePlan, ...]
    trie_sites: tuple[TrieSitePlan, ...]
    frozen_targets: Mapping[str, Any]
    actions: Mapping[str, int]

    @property
    def plan_sha256(self) -> str:
        return _sha256(_canonical_bytes(asdict(self)))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["plan_sha256"] = self.plan_sha256
        return result


PlanSite = CoherentSitePlan | TrieSitePlan
LogitSurface = Literal["trie", "packed", "hf"]


@dataclass(frozen=True)
class ExactLogitEvidence:
    """One caller-captured vector, bound to one immutable plan site."""

    site_id: str
    surface: LogitSurface
    capture_kind: Literal["observed_forward"]
    plan_sha256: str
    site_binding_sha256: str
    logits_sha256: str
    logits: Any

    @classmethod
    def observed(
        cls,
        *,
        plan: Human13CensusPlan,
        site: PlanSite,
        surface: LogitSurface,
        logits: Any,
    ) -> ExactLogitEvidence:
        """Label an externally captured vector; this method does no inference."""

        return cls(
            site_id=site.site_id,
            surface=surface,
            capture_kind=CAPTURE_KIND,
            plan_sha256=plan.plan_sha256,
            site_binding_sha256=_site_sha256(site),
            logits_sha256=_logits_sha256(logits),
            logits=logits,
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"census plan is not canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")


def _site_sha256(site: PlanSite) -> str:
    return _sha256(_canonical_bytes(asdict(site)))


def _logits_sha256(logits: Any) -> str:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("exact logit evidence must contain one rank-one tensor")
    detached = logits.detach().cpu().contiguous()
    header = _canonical_bytes(
        {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
        }
    )
    return _sha256(header + b"\0" + detached.view(torch.uint8).numpy().tobytes())


def _row_roles(token_texts: Sequence[str]) -> tuple[str, ...]:
    """Classify one exact compact row without decoding or text searching."""

    texts = tuple(token_texts)
    valid = (
        len(texts) >= 9
        and texts[0] == "<|object_ref_start|>"
        and "<|object_ref_end|>" in texts
    )
    if not valid:
        raise ValueError("selected row does not have canonical compact row token shape")
    object_end = texts.index("<|object_ref_end|>")
    if (
        object_end < 2
        or texts.count("<|object_ref_end|>") != 1
        or object_end + 7 != len(texts)
        or texts[object_end + 1] != "<|box_start|>"
        or texts[-1] != "<|box_end|>"
        or any(_COORD.fullmatch(text) is None for text in texts[object_end + 2 : -1])
        or len(texts[object_end + 2 : -1]) != 4
    ):
        raise ValueError("selected row does not have canonical compact row token shape")
    description = texts[1:object_end]
    reserved = {
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
    }
    if any(text in reserved or _COORD.fullmatch(text) for text in description):
        raise ValueError("selected row does not have canonical compact row token shape")
    return (
        "boundary",
        *("description" for _ in description),
        "schema",
        "schema",
        "coordinate",
        "coordinate",
        "coordinate",
        "coordinate",
        "row_terminator",
    )


def _validate_discovery_records(
    *, manifest: Human13KUnionManifest | Any, records: Sequence[DiscoveryRecord]
) -> dict[str, DiscoveryRecord]:
    by_id: dict[str, DiscoveryRecord] = {}
    expected: dict[str, tuple[int, tuple[int, ...]]] = {}
    for image in manifest.images:
        for trajectory in image.trajectories:
            identity = trajectory.trajectory_id
            if identity in expected:
                raise ValueError("manifest trajectory identities are not unique")
            expected[identity] = (image.image_id, tuple(trajectory.raw_token_ids))
    for record in records:
        if record.trajectory_id in by_id:
            raise ValueError("discovery trajectory identities are not unique")
        _require_digest(
            record.prompt_token_ids_sha256,
            field="prompt token identity",
        )
        _require_digest(record.chat_text_sha256, field="chat text identity")
        if len(record.token_ids) != len(record.token_texts):
            raise ValueError("discovery token IDs/text coverage differs")
        by_id[record.trajectory_id] = record
    if set(by_id) != set(expected):
        raise ValueError("manifest and discovery trajectory coverage differs")
    for trajectory_id, (image_id, token_ids) in expected.items():
        record = by_id[trajectory_id]
        if record.image_id != image_id:
            raise ValueError("discovery trajectory image identity differs")
        if record.token_ids != token_ids:
            raise ValueError("manifest and discovery trajectory token IDs differ")
    return by_id


def materialize_census_plan(
    *,
    manifest: Human13KUnionManifest | Any,
    manifest_sha256: str,
    discovery_records: Sequence[DiscoveryRecord],
) -> Human13CensusPlan:
    """Materialize native rows, A1 chain sites, and exact trie sites."""

    if getattr(manifest, "full_panel", None) is not True:
        raise ValueError("census planning requires a sealed full-panel manifest")
    _require_digest(manifest_sha256, field="manifest identity")
    records = _validate_discovery_records(
        manifest=manifest,
        records=discovery_records,
    )
    selected_rows: list[NativeRowPlan] = []
    segments: list[A1SegmentPlan] = []
    coherent_sites: list[CoherentSitePlan] = []
    prompt_by_image: dict[int, tuple[str, str]] = {}
    clean_prefix_by_image: dict[int, tuple[int, ...]] = {}

    for image in manifest.images:
        source = image.trajectories[0]
        source_record = records[source.trajectory_id]
        prompt = (
            source_record.prompt_token_ids_sha256,
            source_record.chat_text_sha256,
        )
        prompt_by_image[image.image_id] = prompt
        for trajectory in image.trajectories[1:]:
            record = records[trajectory.trajectory_id]
            if (
                record.prompt_token_ids_sha256,
                record.chat_text_sha256,
            ) != prompt:
                raise ValueError(
                    f"image {image.image_id} discovery prompt identity differs"
                )
        if tuple(row.owner_id for row in image.selected_rows) != tuple(
            image.h_owner_ids
        ):
            raise ValueError("selected row order differs from frozen H owner order")

        clean_prefix = tuple(source.prefix.clean_token_ids)
        clean_prefix_by_image[image.image_id] = clean_prefix
        segment_tokens: list[int] = list(clean_prefix)
        segment_id = f"a1:{image.image_id}"
        image_rows: list[NativeRowPlan] = []
        for row_ordinal, selected in enumerate(image.selected_rows):
            trajectory_matches = tuple(
                trajectory
                for trajectory in image.trajectories
                if trajectory.trajectory_id == selected.trajectory_id
            )
            if len(trajectory_matches) != 1:
                raise ValueError("selected trajectory must resolve exactly once")
            trajectory = trajectory_matches[0]
            row_matches = tuple(
                row for row in trajectory.rows if row.row_id == selected.row_id
            )
            if len(row_matches) != 1:
                raise ValueError("selected native row must resolve exactly once")
            row = row_matches[0]
            token_ids = tuple(trajectory.raw_token_ids[row.token_start : row.token_end])
            if token_ids != tuple(selected.token_ids) or tuple(
                selected.target_token_mask
            ) != (True,) * len(token_ids):
                raise ValueError("selected native row differs from manifest token span")
            record = records[selected.trajectory_id]
            token_texts = record.token_texts[row.token_start : row.token_end]
            roles = _row_roles(token_texts)
            if len(roles) != len(token_ids):
                raise ValueError("selected row token-role coverage differs")
            native = NativeRowPlan(
                image_id=image.image_id,
                owner_id=selected.owner_id,
                row_id=selected.row_id,
                trajectory_id=selected.trajectory_id,
                row_ordinal=row_ordinal,
                token_ids=token_ids,
                token_roles=roles,
            )
            selected_rows.append(native)
            image_rows.append(native)
            for token_offset, (token_id, role) in enumerate(
                zip(token_ids, roles, strict=True)
            ):
                coherent_sites.append(
                    CoherentSitePlan(
                        site_id=f"{segment_id}:site:{len(segment_tokens)}",
                        segment_id=segment_id,
                        segment_role="a1_full_h",
                        image_id=image.image_id,
                        owner_id=selected.owner_id,
                        row_id=selected.row_id,
                        row_ordinal=row_ordinal,
                        token_offset=token_offset,
                        segment_token_offset=len(segment_tokens),
                        segment_prefix_token_ids=tuple(segment_tokens),
                        target_token_id=token_id,
                        token_role=role,
                        prompt_token_ids_sha256=prompt[0],
                        chat_text_sha256=prompt[1],
                    )
                )
                segment_tokens.append(token_id)
        if image_rows:
            segments.append(
                A1SegmentPlan(
                    segment_id=segment_id,
                    segment_role="a1_full_h",
                    image_id=image.image_id,
                    prompt_token_ids_sha256=prompt[0],
                    chat_text_sha256=prompt[1],
                    fixed_prefix_token_ids=clean_prefix,
                    row_ids=tuple(row.row_id for row in image_rows),
                    token_ids=tuple(segment_tokens),
                )
            )

    if not selected_rows:
        raise ValueError("sealed manifest has no selected native H rows")
    census_rows = tuple(
        SelectedNativeRow(str(row.image_id), row.owner_id, row.token_ids)
        for row in selected_rows
    )
    trie = build_exact_token_trie(census_rows)
    trie_sites: list[TrieSitePlan] = []
    for image_id, children in trie.children_by_image.items():
        prompt = prompt_by_image[int(image_id)]
        rows_for_image = tuple(
            row for row in selected_rows if str(row.image_id) == image_id
        )
        for prefix, viable in sorted(children.items()):
            roles_by_target: dict[int, str] = {}
            for row in rows_for_image:
                if row.token_ids[: len(prefix)] != prefix or len(row.token_ids) <= len(
                    prefix
                ):
                    continue
                target = row.token_ids[len(prefix)]
                role = row.token_roles[len(prefix)]
                prior = roles_by_target.setdefault(target, role)
                if prior != role:
                    raise ValueError("trie target has ambiguous token-role binding")
            if set(roles_by_target) != set(viable):
                raise ValueError("trie target-role coverage differs")
            prefix_digest = _sha256(_canonical_bytes(list(prefix)))[:16]
            trie_sites.append(
                TrieSitePlan(
                    site_id=f"trie:{image_id}:{prefix_digest}",
                    segment_id=f"native-trie:{image_id}",
                    segment_role="native_row_trie",
                    image_id=int(image_id),
                    prefix_token_ids=prefix,
                    model_prefix_token_ids=(
                        *clean_prefix_by_image[int(image_id)],
                        *prefix,
                    ),
                    viable_child_token_ids=viable,
                    token_roles_by_target=tuple(sorted(roles_by_target.items())),
                    prompt_token_ids_sha256=prompt[0],
                    chat_text_sha256=prompt[1],
                )
            )

    discovery_binding = tuple(
        {
            "image_id": record.image_id,
            "trajectory_id": record.trajectory_id,
            "prompt_token_ids_sha256": record.prompt_token_ids_sha256,
            "chat_text_sha256": record.chat_text_sha256,
            "token_ids": record.token_ids,
            "token_texts": record.token_texts,
        }
        for record in sorted(records.values(), key=lambda item: item.trajectory_id)
    )
    frozen_targets = {
        "manifest_sha256": manifest_sha256,
        "images": [
            {
                "image_id": image.image_id,
                "G": list(image.g_owner_ids),
                "H": list(image.h_owner_ids),
                "M": list(image.m_owner_ids),
                "selected_rows": [
                    {
                        "owner_id": row.owner_id,
                        "row_id": row.row_id,
                        "trajectory_id": row.trajectory_id,
                        "token_ids": list(row.token_ids),
                    }
                    for row in selected_rows
                    if row.image_id == image.image_id
                ],
            }
            for image in manifest.images
        ],
    }
    return Human13CensusPlan(
        schema_version=SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        discovery_binding_sha256=_sha256(_canonical_bytes(discovery_binding)),
        selected_rows=tuple(selected_rows),
        segments=tuple(segments),
        coherent_sites=tuple(coherent_sites),
        trie_sites=tuple(trie_sites),
        frozen_targets=frozen_targets,
        actions=dict(_ACTIONS),
    )


def run_census_with_exact_logits(
    *,
    plan: Human13CensusPlan,
    evidence: Sequence[ExactLogitEvidence],
) -> dict[str, Any]:
    """Validate full observed-logit coverage, then call the census authority."""

    site_by_id: dict[str, PlanSite] = {
        site.site_id: site for site in (*plan.trie_sites, *plan.coherent_sites)
    }
    if len(site_by_id) != len(plan.trie_sites) + len(plan.coherent_sites):
        raise ValueError("census plan site identities are not unique")
    expected = {
        *((site.site_id, "trie") for site in plan.trie_sites),
        *(
            (site.site_id, surface)
            for site in plan.coherent_sites
            for surface in ("packed", "hf")
        ),
    }
    provided: dict[tuple[str, str], ExactLogitEvidence] = {}
    for item in evidence:
        key = (item.site_id, item.surface)
        if key in provided:
            raise ValueError("exact logit evidence contains duplicate site/surface")
        provided[key] = item
    if set(provided) != expected:
        raise ValueError(
            "exact logit evidence coverage must include every trie, packed, and HF site"
        )
    for key, item in provided.items():
        site = site_by_id[item.site_id]
        if (
            item.capture_kind != CAPTURE_KIND
            or item.plan_sha256 != plan.plan_sha256
            or item.site_binding_sha256 != _site_sha256(site)
        ):
            raise ValueError(f"exact logit evidence content binding differs at {key}")
        if item.logits_sha256 != _logits_sha256(item.logits):
            raise ValueError(f"exact logit vector content differs at {key}")

    trie_logits: dict[str, dict[tuple[int, ...], Any]] = {}
    for site in plan.trie_sites:
        trie_logits.setdefault(str(site.image_id), {})[site.prefix_token_ids] = (
            provided[(site.site_id, "trie")].logits
        )
    coherent = tuple(
        CoherentChainSite(
            image_id=str(site.image_id),
            owner_id=site.owner_id,
            token_offset=site.token_offset,
            target_token_id=site.target_token_id,
            token_role=site.token_role,
            packed_logits=provided[(site.site_id, "packed")].logits,
            hf_logits=provided[(site.site_id, "hf")].logits,
        )
        for site in plan.coherent_sites
    )
    selected = tuple(
        SelectedNativeRow(str(row.image_id), row.owner_id, row.token_ids)
        for row in plan.selected_rows
    )
    return run_no_update_census(
        selected_rows=selected,
        trie_logits=trie_logits,
        coherent_sites=coherent,
        frozen_targets=plan.frozen_targets,
    )


def _read_discovery_records(root: str | Path) -> tuple[DiscoveryRecord, ...]:
    path = Path(root).expanduser().resolve(strict=True) / RECORDS_NAME
    records: list[DiscoveryRecord] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, Mapping):
            raise ValueError(f"discovery record line {line_number} is not an object")
        prompt = value.get("prompt_identity")
        trajectory = value.get("trajectory")
        trace = value.get("token_trace")
        if not isinstance(prompt, Mapping) or not isinstance(trajectory, Mapping):
            raise ValueError("discovery record identity payload is invalid")
        if not isinstance(trace, list):
            raise ValueError("discovery record token trace is invalid")
        non_pad = tuple(item for item in trace if not bool(item.get("is_pad")))
        records.append(
            DiscoveryRecord(
                image_id=int(value["image_id"]),
                trajectory_id=str(trajectory["trajectory_id"]),
                prompt_token_ids_sha256=str(prompt["prompt_token_ids_sha256"]),
                chat_text_sha256=str(prompt["chat_text_sha256"]),
                token_ids=tuple(int(item["token_id"]) for item in non_pad),
                token_texts=tuple(str(item["token_text"]) for item in non_pad),
            )
        )
    return tuple(records)


def load_canonical_census_plan(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    k_root: str | Path,
    panel_path: str | Path = PANEL_PATH,
) -> Human13CensusPlan:
    """Load and cross-bind the sealed manifest and both exact discovery roots."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_manifest(path, require_full_panel=True)
    images = artifacts_to_image_inputs(
        source_root=source_root,
        k_root=k_root,
        panel_path=panel_path,
    )
    rebuilt = build_manifest(
        binding=default_binding(),
        images=images,
        panel_path=panel_path,
        require_full_panel=True,
    )
    if rebuilt != manifest:
        raise ValueError("canonical manifest differs from exact discovery records")
    records = (
        *_read_discovery_records(source_root),
        *_read_discovery_records(k_root),
    )
    return materialize_census_plan(
        manifest=manifest,
        manifest_sha256=_sha256(path.read_bytes()),
        discovery_records=records,
    )


def _write_plan(plan: Human13CensusPlan, output: str | Path | None) -> None:
    payload = _canonical_bytes(plan.to_dict()) + b"\n"
    if output is None:
        print(payload.decode("utf-8"), end="")
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--k-root", required=True)
    parser.add_argument("--panel-path", default=PANEL_PATH)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    plan = load_canonical_census_plan(
        manifest_path=args.manifest,
        source_root=args.source_root,
        k_root=args.k_root,
        panel_path=args.panel_path,
    )
    _write_plan(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
