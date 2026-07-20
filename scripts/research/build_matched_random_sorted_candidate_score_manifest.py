#!/usr/bin/env python3
"""Build the bounded candidate-row scoring manifest for the 2026-07-20 screen.

The Stage-One selection receipt identifies the prefix pairs that changed the
greedy owner or covered-object recurrence.  This builder converts exactly
those pairs into the existing complete candidate-row scoring manifest format.
It deliberately uses canonical entity rows from the frozen ledger.  A parsed
unmatched generated row is never promoted to a positive candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "complete_candidate_row_scoring.manifest.v1"
EXPECTED_PROMOTED_PAIR_COUNT = 6

DEFAULT_EXPERIMENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-20-matched-random-sorted-prefix-order-screen"
)
DEFAULT_SELECTION_RECEIPT = (
    DEFAULT_EXPERIMENT_ROOT / "stage1/matched-random-sorted-prefix-order-selection.json"
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_EXPERIMENT_ROOT / "stage1/common-prefix-permutation"
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/.worktrees/research-probes/research/investigations/"
    "qwen3-vl-dense-enumeration/experiments/"
    "2026-07-20-matched-random-sorted-prefix-order-screen/"
    "candidate-row-scoring-manifest.json"
)


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case(document: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("source artifact lacks cases list")
    matches = [
        item for item in cases
        if isinstance(item, Mapping) and str(item.get("case_id")) == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one case_id={case_id!r}, found {len(matches)}")
    return matches[0]


def _arm(case: Mapping[str, Any], arm_name: str) -> Mapping[str, Any]:
    arms = case.get("arms")
    if not isinstance(arms, Mapping) or arm_name not in arms:
        raise ValueError(f"missing arm {arm_name!r}")
    arm = arms[arm_name]
    if not isinstance(arm, Mapping):
        raise ValueError(f"arm {arm_name!r} is not an object")
    return arm


def _greedy_runs(arm: Mapping[str, Any], comparison_id: str) -> list[Mapping[str, Any]]:
    runs = arm.get("runs")
    if not isinstance(runs, list):
        raise ValueError("arm lacks runs list")
    return [
        item for item in runs
        if isinstance(item, Mapping)
        and item.get("comparison_id") == comparison_id
        and item.get("mode") == "greedy"
        and item.get("seed") is None
    ]


def _comparison_id(case: Mapping[str, Any], left_name: str, right_name: str) -> str:
    left = _arm(case, left_name)
    right = _arm(case, right_name)
    left_ids = {str(run.get("comparison_id")) for run in left.get("runs", []) if isinstance(run, Mapping)}
    right_ids = {str(run.get("comparison_id")) for run in right.get("runs", []) if isinstance(run, Mapping)}
    common = sorted(left_ids & right_ids)
    if len(common) != 1:
        raise ValueError(
            f"expected one comparison between {left_name!r} and {right_name!r}, found {common}"
        )
    return common[0]


def _initial_prefix_selector(
    *, source_artifact: Path, case_id: str, arm_name: str, comparison_id: str
) -> dict[str, Any]:
    arm = _arm(_case(read_json(source_artifact), case_id), arm_name)
    runs = _greedy_runs(arm, comparison_id)
    if len(runs) != 1:
        raise ValueError(
            f"{source_artifact}: expected one greedy run for {arm_name}/{comparison_id}, found {len(runs)}"
        )
    run = runs[0]
    tokens = run.get("initial_prefix_token_ids")
    declared_hash = run.get("initial_prefix_token_ids_sha256")
    if not isinstance(tokens, list) or any(not isinstance(value, int) for value in tokens):
        raise ValueError(f"{arm_name}/{comparison_id} initial prefix is invalid")
    actual_hash = sha256_json(tokens)
    if declared_hash != actual_hash:
        raise ValueError(f"{arm_name}/{comparison_id} initial prefix hash mismatch")
    return {
        "selector": {
            "source_artifact": str(source_artifact.resolve()),
            "case_id": case_id,
            "arm_name": arm_name,
            "comparison_id": comparison_id,
            "mode": "greedy",
            "seed": None,
            "row_index": 0,
            "token_field": "initial_prefix_token_ids",
        },
        "token_ids_sha256": actual_hash,
    }


def _ledger_entity(case: Mapping[str, Any], entity_id: str) -> Mapping[str, Any]:
    ledger = case.get("entity_ledger")
    if not isinstance(ledger, list):
        raise ValueError("case lacks entity_ledger")
    matches = [
        item for item in ledger
        if isinstance(item, Mapping) and str(item.get("entity_id")) == entity_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one canonical ledger entity {entity_id!r}, found {len(matches)}")
    entity = matches[0]
    tokens = entity.get("row_token_ids")
    declared_hash = entity.get("row_token_ids_sha256")
    if not isinstance(tokens, list) or any(not isinstance(value, int) for value in tokens):
        raise ValueError(f"canonical entity {entity_id!r} has invalid row tokens")
    if declared_hash != sha256_json(tokens):
        raise ValueError(f"canonical entity {entity_id!r} row hash mismatch")
    return entity


def _canonical_row_reference(
    *, source_artifact: Path, case_id: str, arm_name: str, entity_id: str, role: str, covered: bool, notes: str | None = None
) -> dict[str, Any]:
    case = _case(read_json(source_artifact), case_id)
    entity = _ledger_entity(case, entity_id)
    row_hash = str(entity["row_token_ids_sha256"])
    candidate: dict[str, Any] = {
        "candidate_id": f"entity-{entity_id}",
        "owner": entity_id,
        "category": entity.get("description"),
        "role": role,
        "covered": bool(covered),
        "truth_status": "canonical_frozen_entity_row",
        "row": {
            "selector": {
                "source_artifact": str(source_artifact.resolve()),
                "case_id": case_id,
                "arm_name": arm_name,
                "entity_id": entity_id,
                "token_field": "entity_row_token_ids",
            },
            "token_ids_sha256": row_hash,
        },
    }
    if notes:
        candidate["notes"] = notes
    return candidate


def _strict_owner_ids(selection_pair: Mapping[str, Any], checkpoint: str) -> list[str]:
    result = selection_pair["checkpoint_results"][checkpoint]
    greedy = result.get("greedy")
    if not isinstance(greedy, Mapping):
        raise ValueError(f"{checkpoint} selection result lacks greedy summary")
    owners: list[str] = []
    for side in ("left", "right"):
        value = greedy.get(side, {}).get("strict_owner") if isinstance(greedy.get(side), Mapping) else None
        if isinstance(value, str) and value and value not in owners:
            owners.append(value)
    return owners


def _sampled_fallback_owner_ids(selection_pair: Mapping[str, Any], checkpoint: str, known: list[str]) -> list[str]:
    """Return canonical strict owners observed in paired samples, by frequency.

    This is only a fallback when greedy has an unmatched side.  It is kept
    explicitly marked in the manifest and is never inferred from an unmatched
    generated row.
    """

    result = selection_pair["checkpoint_results"][checkpoint]
    counts: dict[str, int] = {}
    for pair_run in result.get("paired_runs", []):
        if not isinstance(pair_run, Mapping) or pair_run.get("mode") != "sample":
            continue
        for side in ("left", "right"):
            side_result = pair_run.get(side)
            owner = side_result.get("strict_owner") if isinstance(side_result, Mapping) else None
            if isinstance(owner, str) and owner not in known:
                counts[owner] = counts.get(owner, 0) + 1
    return [owner for owner, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _covered_owner_id(case: Mapping[str, Any], left_name: str, right_name: str, excluded: set[str]) -> str | None:
    left = _arm(case, left_name)
    right = _arm(case, right_name)
    left_ids = {str(item) for item in left.get("covered_entity_ids", [])}
    right_ids = {str(item) for item in right.get("covered_entity_ids", [])}
    common = sorted((left_ids & right_ids) - excluded)
    if common:
        # The last frozen row is the most direct recurrence control while
        # remaining identical across both prefix arms.
        return common[-1]
    return None


def _promoted_pairs(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selection.get("pairs", [])):
        if not isinstance(pair, Mapping):
            raise ValueError(f"pairs[{pair_index}] must be an object")
        for checkpoint in ("sorted", "random"):
            result = pair.get("checkpoint_results", {}).get(checkpoint) if isinstance(pair.get("checkpoint_results"), Mapping) else None
            if isinstance(result, Mapping) and result.get("activated") is True:
                promoted.append({
                    "pair_index": pair_index,
                    "checkpoint": checkpoint,
                    "pair": pair,
                    "result": result,
                })
    if len(promoted) != EXPECTED_PROMOTED_PAIR_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_PROMOTED_PAIR_COUNT} activated pairs, found {len(promoted)}")
    return promoted


def build_manifest(
    *, selection: Mapping[str, Any], selection_path: Path, artifact_root: Path
) -> dict[str, Any]:
    promoted = _promoted_pairs(selection)
    images: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    ambiguities: list[dict[str, Any]] = []
    promoted_metadata: list[dict[str, Any]] = []

    for item in promoted:
        pair = item["pair"]
        checkpoint = item["checkpoint"]
        artifact_path = (artifact_root / checkpoint / str(pair["artifact_file"])).resolve(strict=True)
        document = read_json(artifact_path)
        case_id = str(pair["case_id"])
        case = _case(document, case_id)
        arms = item["result"].get("arms")
        if not isinstance(arms, list) or len(arms) != 2 or any(not isinstance(name, str) for name in arms):
            raise ValueError(f"promoted pair {item['pair_index']} must identify exactly two arms")
        left_name, right_name = arms
        comparison_id = _comparison_id(case, left_name, right_name)
        image_id = str(document.get("image", {}).get("image_id", ""))
        if not image_id:
            raise ValueError(f"{artifact_path} lacks image.image_id")

        if image_id not in images:
            base = document.get("base_prompt")
            if not isinstance(base, Mapping) or not isinstance(base.get("prompt_token_ids"), list):
                raise ValueError(f"{artifact_path} lacks base_prompt.prompt_token_ids")
            base_tokens = base["prompt_token_ids"]
            base_hash = base.get("prompt_token_ids_sha256")
            if base_hash != sha256_json(base_tokens):
                raise ValueError(f"{artifact_path} base prompt hash mismatch")
            images[image_id] = {
                "image_id": image_id,
                "base_prompt": {
                    "selector": {
                        "source_artifact": str(artifact_path),
                        "token_field": "base_prompt.prompt_token_ids",
                    },
                    "token_ids_sha256": base_hash,
                },
                "boundaries": [],
            }

        greedy_owners = _strict_owner_ids(pair, checkpoint)
        fallback_owners = _sampled_fallback_owner_ids(pair, checkpoint, greedy_owners)
        candidate_ids: list[str] = []
        candidate_rows: list[dict[str, Any]] = []
        for index, owner in enumerate(greedy_owners):
            # The selection receipt explicitly marks gt_0021 as a covered
            # recurrence in the depth-10 random checkpoint.  Do not label it
            # verified-uncovered merely because it is also the emitted owner.
            covered_recurrence = (
                item["result"].get("greedy", {}).get("left", {}).get("covered_recurrence") is True
                and owner == item["result"].get("greedy", {}).get("left", {}).get("strict_owner")
            )
            role = "greedy_left_owner" if index == 0 else "greedy_right_owner"
            notes = None
            if covered_recurrence:
                role = f"{role}_and_covered_recurrence"
                notes = "Selection receipt marks this canonical owner as a covered recurrence; do not interpret as a verified-uncovered positive."
            candidate_rows.append(_canonical_row_reference(
                source_artifact=artifact_path,
                case_id=case_id,
                arm_name=left_name,
                entity_id=owner,
                role=role,
                covered=covered_recurrence,
                notes=notes,
            ))
            candidate_ids.append(owner)

        if len(greedy_owners) < 2:
            if not fallback_owners:
                ambiguities.append({
                    "pair_index": item["pair_index"],
                    "checkpoint": checkpoint,
                    "case_id": case_id,
                    "comparison_id": comparison_id,
                    "issue": "one greedy arm has no strict canonical owner and no sampled strict-owner fallback exists",
                })
            else:
                # Preserve every canonical strict owner observed in the
                # paired samples.  Picking only the most frequent owner would
                # silently discard a plausible sampled alternative.
                for fallback in fallback_owners:
                    candidate_rows.append(_canonical_row_reference(
                        source_artifact=artifact_path,
                        case_id=case_id,
                        arm_name=left_name,
                        entity_id=fallback,
                        role="sampled_strict_owner_fallback",
                        covered=False,
                        notes="Greedy alternate was unmatched; this canonical row was observed as a strict owner in a paired sample and is included only as a flagged fallback.",
                    ))
                    candidate_ids.append(fallback)
                ambiguities.append({
                    "pair_index": item["pair_index"],
                    "checkpoint": checkpoint,
                    "case_id": case_id,
                    "comparison_id": comparison_id,
                    "issue": "one greedy arm was unmatched; sampled strict owner used as an explicit fallback",
                    "fallback_owners": fallback_owners,
                })

        covered_owner = _covered_owner_id(case, left_name, right_name, set(candidate_ids))
        if covered_owner is not None:
            candidate_rows.append(_canonical_row_reference(
                source_artifact=artifact_path,
                case_id=case_id,
                arm_name=left_name,
                entity_id=covered_owner,
                role="frozen_prefix_covered",
                covered=True,
                notes="Canonical row for an entity present in both frozen prefix arms; terminal is scored separately.",
            ))
        else:
            ambiguities.append({
                "pair_index": item["pair_index"],
                "checkpoint": checkpoint,
                "case_id": case_id,
                "comparison_id": comparison_id,
                "issue": "no common frozen-prefix covered canonical owner remained after excluding competing owners",
            })
        # The complete scorer consumes one prefix per boundary.  Emit both
        # initial prefixes as separate boundaries rather than hiding one as
        # metadata, so every candidate is scored at both order states.
        for side, arm_name in (("left", left_name), ("right", right_name)):
            boundary_id = f"{case_id}__{checkpoint}__{right_name}__{side}"
            if any(str(boundary.get("boundary_id")) == boundary_id for boundary in images[image_id]["boundaries"]):
                raise ValueError(f"duplicate generated boundary id {boundary_id}")
            images[image_id]["boundaries"].append({
                "boundary_id": boundary_id,
                "case_id": case_id,
                "checkpoint_arm": checkpoint,
                "comparison_id": comparison_id,
                "prefix_arm": arm_name,
                "paired_prefix_arm": right_name if side == "left" else left_name,
                # Stage-One stores the generated suffix after the common
                # processor prompt.  The scorer must prepend base_prompt
                # before materializing image-conditioned logits.
                "prefix_mode": "base_prompt_plus_generated",
                "prefix": _initial_prefix_selector(
                    source_artifact=artifact_path,
                    case_id=case_id,
                    arm_name=arm_name,
                    comparison_id=comparison_id,
                ),
                "candidates": json.loads(json.dumps(candidate_rows)),
            })
        promoted_metadata.append({
            "pair_index": item["pair_index"],
            "checkpoint": checkpoint,
            "case_id": case_id,
            "artifact_file": str(pair["artifact_file"]),
            "comparison_id": comparison_id,
            "arms": [left_name, right_name],
            "candidate_owner_ids": candidate_ids,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": "2026-07-20-matched-random-sorted-prefix-order-screen",
        "selection_receipt": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "activated_pair_count": len(promoted),
        },
        "builder": {
            "name": "build_matched_random_sorted_candidate_score_manifest.py",
            "candidate_policy": "canonical frozen entity rows only; unmatched generated rows are never positives",
            "terminal_policy": "terminal is scored separately by the existing scorer",
            "ambiguities": ambiguities,
        },
        "promoted_pairs": promoted_metadata,
        "images": list(images.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, default=DEFAULT_SELECTION_RECEIPT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selection_path = args.selection_receipt.expanduser().resolve(strict=True)
    manifest = build_manifest(
        selection=read_json(selection_path),
        selection_path=selection_path,
        artifact_root=args.artifact_root.expanduser().resolve(strict=True),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "images": len(manifest["images"]), "boundaries": sum(len(image["boundaries"]) for image in manifest["images"]), "ambiguities": len(manifest["builder"]["ambiguities"])}, sort_keys=True))


if __name__ == "__main__":
    main()
