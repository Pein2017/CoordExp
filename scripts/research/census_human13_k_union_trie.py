#!/usr/bin/env python3
"""CPU-only no-update trie and coherent-chain census for Human-13.

The caller supplies frozen selected rows and already aligned compact logits.
This module does not load a model, select targets, alter owner sets, or build a
trainable candidate tree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import torch


TOKEN_ROLES = (
    "boundary",
    "schema",
    "description",
    "coordinate",
    "row_terminator",
)
A8_DRIFT_ALLOWANCE = 1.0e-4
A8_MAX_REQUIRED_MARGIN = 0.5


class CensusContractError(ValueError):
    """Raised when a no-update census input is incoherent."""


@dataclass(frozen=True)
class SelectedNativeRow:
    image_id: str
    owner_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CoherentChainSite:
    image_id: str
    owner_id: str
    token_offset: int
    target_token_id: int
    token_role: str
    packed_logits: torch.Tensor
    hf_logits: torch.Tensor
    competitor_status_by_token: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactTokenTrie:
    children_by_image: Mapping[str, Mapping[tuple[int, ...], tuple[int, ...]]]
    owners_by_image_leaf: Mapping[str, Mapping[tuple[int, ...], tuple[str, ...]]]
    original_row_count: int
    unique_row_count: int
    exact_duplicate_count: int


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CensusContractError(
            f"frozen targets are not canonical JSON: {exc}"
        ) from exc


def build_exact_token_trie(rows: Sequence[SelectedNativeRow]) -> ExactTokenTrie:
    """Build a diagnostic trie while retaining exact-row owner aliases."""

    children: dict[str, dict[tuple[int, ...], set[int]]] = {}
    owners_by_leaf: dict[str, dict[tuple[int, ...], list[str]]] = {}
    for row_index, row in enumerate(rows):
        _validate_row(row, row_index=row_index)
        image_owners = owners_by_leaf.setdefault(row.image_id, {})
        owners = image_owners.setdefault(row.token_ids, [])
        owners.append(row.owner_id)
        image_children = children.setdefault(row.image_id, {})
        for offset, token_id in enumerate(row.token_ids):
            image_children.setdefault(row.token_ids[:offset], set()).add(token_id)
    unique_count = sum(len(leaves) for leaves in owners_by_leaf.values())
    return ExactTokenTrie(
        children_by_image={
            image_id: {
                prefix: tuple(sorted(token_ids))
                for prefix, token_ids in sorted(image_children.items())
            }
            for image_id, image_children in sorted(children.items())
        },
        owners_by_image_leaf={
            image_id: {
                tokens: tuple(owner_ids)
                for tokens, owner_ids in sorted(image_owners.items())
            }
            for image_id, image_owners in sorted(owners_by_leaf.items())
        },
        original_row_count=len(rows),
        unique_row_count=unique_count,
        exact_duplicate_count=len(rows) - unique_count,
    )


def run_no_update_census(
    *,
    selected_rows: Sequence[SelectedNativeRow],
    trie_logits: Mapping[str, Mapping[tuple[int, ...], torch.Tensor]],
    coherent_sites: Sequence[CoherentChainSite],
    frozen_targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Return bounded trie, chain, drift, A8-prime, and immutability receipts."""

    frozen_before = canonical_json_bytes(frozen_targets)
    rows = tuple(selected_rows)
    sites = tuple(coherent_sites)
    trie = build_exact_token_trie(rows)
    _validate_full_coherent_traversal(rows, sites)

    trie_receipt = _score_trie(trie, trie_logits)
    chain_receipt, packed_margins, drifts, aligned_finite = _score_chain(sites)
    a8_receipt = _a8_prime_receipt(
        packed_margins=packed_margins,
        drifts=drifts,
        aligned_finite=aligned_finite,
    )

    frozen_after = canonical_json_bytes(frozen_targets)
    before_sha = hashlib.sha256(frozen_before).hexdigest()
    after_sha = hashlib.sha256(frozen_after).hexdigest()
    return {
        "schema_version": "human13_k_union_no_update_census.v1",
        "trie": trie_receipt,
        "coherent_chain": chain_receipt,
        "aligned_surface": {
            "all_finite": aligned_finite,
            "maximum_absolute_margin_drift": max(drifts) if drifts else None,
            "site_count": len(drifts),
        },
        "a8_prime": a8_receipt,
        "frozen_targets": {
            "sha256_before": before_sha,
            "sha256_after": after_sha,
            "byte_identical": frozen_before == frozen_after,
        },
    }


def _score_trie(
    trie: ExactTokenTrie,
    trie_logits: Mapping[str, Mapping[tuple[int, ...], torch.Tensor]],
) -> dict[str, Any]:
    images = [
        _score_image_trie(
            image_id=image_id,
            children=image_children,
            owners_by_leaf=trie.owners_by_image_leaf[image_id],
            trie_logits=trie_logits.get(image_id),
        )
        for image_id, image_children in trie.children_by_image.items()
    ]
    if set(trie_logits) != set(trie.children_by_image):
        raise CensusContractError(
            "trie logits must provide exactly one mapping per selected image"
        )
    return {
        "original_row_count": trie.original_row_count,
        "unique_row_count": trie.unique_row_count,
        "exact_duplicate_count": trie.exact_duplicate_count,
        "images": images,
    }


def _score_image_trie(
    *,
    image_id: str,
    children: Mapping[tuple[int, ...], tuple[int, ...]],
    owners_by_leaf: Mapping[tuple[int, ...], tuple[str, ...]],
    trie_logits: Mapping[tuple[int, ...], torch.Tensor] | None,
) -> dict[str, Any]:
    if trie_logits is None:
        raise CensusContractError(f"missing trie logits for image {image_id}")
    prefix: tuple[int, ...] = ()
    nodes: list[dict[str, Any]] = []
    reached_leaf = False
    while prefix in children:
        logits = _finite_vector(
            trie_logits.get(prefix),
            label=f"trie_logits[{image_id!r}]{prefix}",
        )
        viable = children[prefix]
        if viable and max(viable) >= int(logits.numel()):
            raise CensusContractError("trie child token is outside logits vocabulary")
        top1 = int(logits.argmax().item())
        viable_tensor = torch.tensor(viable, dtype=torch.long, device=logits.device)
        strongest_viable = int(viable_tensor[logits[viable_tensor].argmax()].item())
        nonviable_mask = torch.ones_like(logits, dtype=torch.bool)
        nonviable_mask[viable_tensor] = False
        strongest_nonviable_score = (
            float(logits[nonviable_mask].max().item())
            if bool(nonviable_mask.any().item())
            else float("-inf")
        )
        viable_margin = (
            float(logits[strongest_viable].item()) - strongest_nonviable_score
        )
        nodes.append(
            {
                "prefix_token_ids": list(prefix),
                "viable_child_token_ids": list(viable),
                "actual_top1_token_id": top1,
                "actual_top1_is_viable_child": top1 in viable,
                "strongest_viable_child_token_id": strongest_viable,
                "strongest_viable_child_margin": viable_margin,
                "top_tie_count": int((logits == logits.max()).sum().item()),
            }
        )
        if top1 not in viable:
            break
        prefix = (*prefix, top1)
        if prefix in owners_by_leaf:
            reached_leaf = True
            break
    return {
        "image_id": image_id,
        "nodes": nodes,
        "projected_token_ids": list(prefix),
        "projected_owner_ids": list(owners_by_leaf.get(prefix, ())),
        "reached_native_leaf": reached_leaf,
    }


def _score_chain(
    sites: Sequence[CoherentChainSite],
) -> tuple[dict[str, Any], list[float], list[float], bool]:
    receipts: list[dict[str, Any]] = []
    packed_margins: list[float] = []
    drifts: list[float] = []
    aligned_finite = True
    first_non_argmax: dict[str, Any] | None = None
    role_counts: Counter[str] = Counter()
    tie_site_count = 0

    for site_index, site in enumerate(sites):
        if site.token_role not in TOKEN_ROLES:
            raise CensusContractError(f"unknown token role: {site.token_role}")
        role_counts[site.token_role] += 1
        packed = _surface_score(site.packed_logits, site.target_token_id)
        hf = _surface_score(site.hf_logits, site.target_token_id)
        finite = packed is not None and hf is not None
        aligned_finite = aligned_finite and finite
        receipt: dict[str, Any] = {
            "site_index": site_index,
            "image_id": site.image_id,
            "owner_id": site.owner_id,
            "token_offset": site.token_offset,
            "target_token_id": site.target_token_id,
            "token_role": site.token_role,
            "aligned_finite": finite,
        }
        if finite:
            assert packed is not None and hf is not None
            drift = abs(packed["target_margin"] - hf["target_margin"])
            packed_margins.append(packed["target_margin"])
            drifts.append(drift)
            packed_ties = packed["top_tie_count"]
            tie_site_count += int(packed_ties > 1)
            receipt.update(
                {
                    "packed_competitor_token_id": packed["competitor_token_id"],
                    "hf_competitor_token_id": hf["competitor_token_id"],
                    "packed_target_margin": packed["target_margin"],
                    "hf_target_margin": hf["target_margin"],
                    "absolute_margin_drift": drift,
                    "packed_top_tie_count": packed_ties,
                    "hf_top_tie_count": hf["top_tie_count"],
                    "packed_competitor_status": site.competitor_status_by_token.get(
                        packed["competitor_token_id"], "unclassified"
                    ),
                }
            )
            if first_non_argmax is None and packed["target_margin"] <= 0.0:
                first_non_argmax = dict(receipt)
        receipts.append(receipt)

    minimum = min(packed_margins) if packed_margins else None
    return (
        {
            "site_count": len(sites),
            "sites": receipts,
            "first_non_argmax_site": first_non_argmax,
            "minimum_strict_margin": minimum,
            "tie_site_count": tie_site_count,
            "token_role_counts": {role: role_counts[role] for role in TOKEN_ROLES},
        },
        packed_margins,
        drifts,
        aligned_finite,
    )


def _a8_prime_receipt(
    *,
    packed_margins: Sequence[float],
    drifts: Sequence[float],
    aligned_finite: bool,
) -> dict[str, Any]:
    if not aligned_finite or not packed_margins or len(drifts) != len(packed_margins):
        return _blocked_a8("aligned_finite_scores_unavailable", required_margin=None)
    required_margin = max(drifts) + A8_DRIFT_ALLOWANCE
    if not math.isfinite(required_margin):
        return _blocked_a8("aligned_finite_scores_unavailable", required_margin=None)
    if required_margin > A8_MAX_REQUIRED_MARGIN:
        return _blocked_a8(
            "required_margin_exceeds_0_5",
            required_margin=required_margin,
        )
    violating = sum(margin < required_margin for margin in packed_margins)
    if violating == 0:
        return _blocked_a8(
            "no_margin_violations",
            required_margin=required_margin,
        )
    return {
        "applicable": True,
        "blocked": False,
        "block_reason": None,
        "required_margin": required_margin,
        "violating_site_count": violating,
    }


def _blocked_a8(reason: str, *, required_margin: float | None) -> dict[str, Any]:
    return {
        "applicable": False,
        "blocked": True,
        "block_reason": reason,
        "required_margin": required_margin,
        "violating_site_count": 0,
    }


def _surface_score(logits: torch.Tensor, target_token_id: int) -> dict[str, Any] | None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() < 2:
        raise CensusContractError(
            "aligned logits must be rank-one with vocabulary >= 2"
        )
    scores = logits.float()
    if target_token_id < 0 or target_token_id >= int(scores.numel()):
        raise CensusContractError("target token is outside aligned logits vocabulary")
    if not bool(torch.isfinite(scores).all().item()):
        return None
    non_target = scores.clone()
    non_target[target_token_id] = float("-inf")
    competitor_id = int(non_target.argmax().item())
    margin = float((scores[target_token_id] - scores[competitor_id]).item())
    return {
        "competitor_token_id": competitor_id,
        "target_margin": margin,
        "top_tie_count": int((scores == scores.max()).sum().item()),
    }


def _finite_vector(value: torch.Tensor | None, *, label: str) -> torch.Tensor:
    if value is None or not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise CensusContractError(f"{label} must be a rank-one tensor")
    checked = value.float()
    if not bool(torch.isfinite(checked).all().item()):
        raise CensusContractError(f"{label} must be finite")
    return checked


def _validate_row(row: SelectedNativeRow, *, row_index: int) -> None:
    if not isinstance(row, SelectedNativeRow):
        raise CensusContractError(f"selected row {row_index} has the wrong type")
    if not row.image_id or not row.owner_id or not row.token_ids:
        raise CensusContractError(f"selected row {row_index} is incomplete")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in row.token_ids
    ):
        raise CensusContractError(f"selected row {row_index} has an invalid token id")


def _validate_full_coherent_traversal(
    rows: Sequence[SelectedNativeRow],
    sites: Sequence[CoherentChainSite],
) -> None:
    expected = tuple(
        (row.image_id, row.owner_id, token_offset, token_id)
        for row in rows
        for token_offset, token_id in enumerate(row.token_ids)
    )
    observed = tuple(
        (site.image_id, site.owner_id, site.token_offset, site.target_token_id)
        for site in sites
    )
    if observed != expected:
        raise CensusContractError(
            "coherent sites must exhaust selected rows in frozen full-H order"
        )


__all__ = [
    "A8_DRIFT_ALLOWANCE",
    "A8_MAX_REQUIRED_MARGIN",
    "CensusContractError",
    "CoherentChainSite",
    "ExactTokenTrie",
    "SelectedNativeRow",
    "TOKEN_ROLES",
    "build_exact_token_trie",
    "canonical_json_bytes",
    "run_no_update_census",
]
