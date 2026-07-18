#!/usr/bin/env python3
"""Score the same frozen rows under a small exact-prefix state factorial.

This file is intentionally local to the row-lag experiment.  It reuses the
lineage checks and direct model-forward helpers from
``run_next_row_likelihood_change.py``.  It does not introduce a decoder
constraint, a memory module, a repetition penalty, or a probability
normalization across rows.  A state is only a reconstructed token prefix; a
``forced_replay`` state is reported as such and is never promoted to a native
trajectory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_native_sibling_branch_replay import (  # noqa: E402
    _attention_implementation,
    _model_dtype_summary as _runtime_model_dtype_summary,
    complete_row_spans,
)
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    BOX_END,
    BOX_START,
    COORDINATE_TOKEN_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    _bundle_execution_value,
    _bundle_image_id,
    _canonical_row_phases,
    _forward_logits,
    _read_json,
    _token_ids,
    emit_candidate_score,
    score_token_logits,
    sha256_file,
    sha256_json,
    terminal_boundary_score,
    validate_active_source_lineage,
)


MANIFEST_SCHEMA_VERSION = "prefix_row_factorial_likelihood.manifest.v1"
RECEIPT_SCHEMA_VERSION = "prefix_row_factorial_likelihood.receipt.v1"
NATIVE_CALL_BUNDLE_SCHEMA_VERSION = "native_sibling_branch_replay.call_bundle.v1"
STATE_IDS = ("P0", "PA", "PB", "PAB")
STATE_KINDS = ("native_contiguous", "forced_replay")
PHASES = ("row_entry", "description", "geometry", "x1", "y1", "x2", "y2", "closure", "full_row")


def _resolve(path_value: str | Path, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    return (path if path.is_absolute() else manifest_path.parent / path).resolve(strict=True)


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{context} requires non-empty {key}")
    return result


def _span(value: Any, context: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(v, int) for v in value):
        raise ValueError(f"{context} requires integer source_token_span [start,end]")
    start, end = value
    if start < 0 or end <= start:
        raise ValueError(f"{context} source_token_span is invalid")
    return start, end


def _support_kind(value: Mapping[str, Any], context: str) -> str:
    support = value.get("natural_support")
    if not isinstance(support, Mapping):
        raise ValueError(f"{context} requires natural_support object")
    kind = support.get("state_kind", support.get("kind"))
    if kind not in STATE_KINDS:
        raise ValueError(f"{context} natural_support requires state_kind native_contiguous or forced_replay")
    return str(kind)


def _support_state_id(value: Mapping[str, Any], context: str) -> str:
    support = value.get("natural_support")
    if not isinstance(support, Mapping):
        raise ValueError(f"{context} requires natural_support object")
    state_id = support.get("support_state_id")
    if not isinstance(state_id, str) or state_id not in STATE_IDS:
        raise ValueError(f"{context} natural_support requires support_state_id P0, PA, PB, or PAB")
    return state_id


def _validate_metadata(item: Mapping[str, Any], context: str) -> None:
    for key in ("owner", "category", "description", "geometry"):
        if key not in item:
            raise ValueError(f"{context} requires {key} metadata")
    if not isinstance(item["geometry"], list) or len(item["geometry"]) != 4:
        raise ValueError(f"{context} geometry must contain four values")


def _same_image(parent_bundle: Mapping[str, Any], bundle: Mapping[str, Any], image_id: str, context: str) -> None:
    parent_image = _bundle_image_id(parent_bundle)
    image = _bundle_image_id(bundle)
    if image not in {None, image_id}:
        raise ValueError(f"{context} source bundle image does not match manifest")
    if parent_image not in {None, image_id}:
        raise ValueError("parent source bundle image does not match manifest")
    parent_digest = _bundle_execution_value(parent_bundle, "source_image_sha256")
    digest = _bundle_execution_value(bundle, "source_image_sha256")
    if parent_digest is not None and digest is not None and str(parent_digest) != str(digest):
        raise ValueError(f"{context} source image digest disagrees with parent lineage")


def _row_from_source(
    item: Mapping[str, Any], *, manifest_path: Path, parent_bundle: Mapping[str, Any], image_id: str, context: str
) -> dict[str, Any]:
    source = _resolve(_required_text(item, "source_bundle", context), manifest_path)
    bundle = _read_json(source)
    source_hash = sha256_file(source)
    if _required_text(item, "source_bundle_sha256", context) != source_hash:
        raise ValueError(f"{context} source_bundle_sha256 mismatch")
    _same_image(parent_bundle, bundle, image_id, context)
    source_prompt, source_generated = _token_ids(bundle)
    start, end = _span(item.get("source_token_span"), context)
    spans = complete_row_spans(source_generated)
    if (start, end) not in spans:
        raise ValueError(f"{context} source span is not one complete natural row")
    row = [int(value) for value in source_generated[start:end]]
    token_hash = sha256_json(row)
    if _required_text(item, "token_ids_sha256", context) != token_hash:
        raise ValueError(f"{context} token_ids_sha256 mismatch")
    _canonical_row_phases(row)
    _validate_metadata(item, context)
    return {
        **dict(item),
        "source_bundle": str(source),
        "source_bundle_sha256": source_hash,
        "source_prompt_token_ids": source_prompt,
        "source_prompt_ids_sha256": sha256_json(source_prompt),
        "source_generated_token_ids": source_generated,
        "source_token_span": [start, end],
        "token_ids": row,
        "token_ids_sha256": token_hash,
        "token_count": len(row),
        "natural_support_state_kind": _support_kind(item, context) if "natural_support" in item else None,
        "natural_support_state_id": _support_state_id(item, context) if "natural_support" in item else None,
    }


def _parent(manifest: Mapping[str, Any], *, manifest_path: Path, image_id: str) -> dict[str, Any]:
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("manifest requires parent object")
    source = _resolve(_required_text(parent, "source_bundle", "parent"), manifest_path)
    bundle = _read_json(source)
    source_hash = sha256_file(source)
    if _required_text(parent, "source_bundle_sha256", "parent") != source_hash:
        raise ValueError("parent source_bundle_sha256 mismatch")
    _same_image(bundle, bundle, image_id, "parent")
    base_prompt, generated = _token_ids(bundle)
    prefix_count = parent.get("prefix_token_count")
    if not isinstance(prefix_count, int) or not 0 <= prefix_count <= len(generated):
        raise ValueError("parent prefix_token_count is invalid")
    prefix = generated[:prefix_count]
    if prefix_count and (not complete_row_spans(prefix) or complete_row_spans(prefix)[-1][1] != prefix_count):
        raise ValueError("parent prefix must end at a complete native row")
    expected = [*base_prompt, *prefix]
    checks = {
        "prompt_token_ids_sha256": sha256_json(base_prompt),
        "prefix_token_ids_sha256": sha256_json(prefix),
        "reconstructed_prompt_token_ids_sha256": sha256_json(expected),
    }
    for key, actual in checks.items():
        declared = parent.get(key)
        if declared is not None and declared != actual:
            raise ValueError(f"parent {key} does not match source bundle")
    return {
        "source_bundle": str(source),
        "source_bundle_sha256": source_hash,
        "bundle": bundle,
        "base_prompt_token_ids": base_prompt,
        "prefix_token_ids": prefix,
        "prefix_token_count": prefix_count,
        "reconstructed_prompt_token_ids": expected,
        "prompt_token_ids_sha256": checks["prompt_token_ids_sha256"],
        "prefix_token_ids_sha256": checks["prefix_token_ids_sha256"],
        "reconstructed_prompt_token_ids_sha256": checks["reconstructed_prompt_token_ids_sha256"],
    }


def _state_kind(state: Mapping[str, Any], context: str) -> str:
    kind = state.get("state_kind")
    if kind not in STATE_KINDS:
        raise ValueError(f"{context} state_kind must be native_contiguous or forced_replay")
    return str(kind)


def validate_factorial_manifest(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    """Validate exact fixture, state, and candidate lineage without loading a model."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected {MANIFEST_SCHEMA_VERSION}")
    image_id = _required_text(manifest, "image_id", "manifest")
    parent = _parent(manifest, manifest_path=manifest_path, image_id=image_id)
    fixture_items = manifest.get("row_fixtures")
    state_items = manifest.get("states")
    candidates = manifest.get("candidate_variants")
    if not isinstance(fixture_items, list) or not fixture_items:
        raise ValueError("manifest requires non-empty row_fixtures list")
    if not isinstance(state_items, list):
        raise ValueError("manifest requires states list")
    state_ids = [str(item.get("state_id")) for item in state_items if isinstance(item, Mapping)]
    if len(state_ids) != len(set(state_ids)) or set(state_ids) != set(STATE_IDS):
        raise ValueError("states must define exactly P0, PA, PB, and PAB")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest requires non-empty candidate_variants list")

    fixtures: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(fixture_items):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row_fixtures[{index}] must be an object")
        fixture_id = _required_text(raw, "fixture_id", f"row_fixtures[{index}]")
        if fixture_id in fixtures:
            raise ValueError(f"duplicate fixture_id {fixture_id}")
        fixtures[fixture_id] = _row_from_source(
            raw, manifest_path=manifest_path, parent_bundle=parent["bundle"], image_id=image_id, context=fixture_id
        )

    state_map = {str(item["state_id"]): item for item in state_items}
    # This experiment is intentionally one very specific factorial.  Do not
    # let a more general state table silently change its meaning.
    if state_map["P0"].get("state_kind") != "native_contiguous" or state_map["P0"].get("row_fixture_ids") != []:
        raise ValueError("P0 must be native_contiguous with no row fixtures")
    if state_map["PA"].get("state_kind") != "native_contiguous":
        raise ValueError("PA must be native_contiguous")
    if state_map["PB"].get("state_kind") != "forced_replay":
        raise ValueError("PB must be forced_replay")
    if state_map["PAB"].get("state_kind") != "native_contiguous":
        raise ValueError("PAB must be native_contiguous")
    pa_fixture_ids = state_map["PA"].get("row_fixture_ids")
    pb_fixture_ids = state_map["PB"].get("row_fixture_ids")
    pab_fixture_ids = state_map["PAB"].get("row_fixture_ids")
    if not isinstance(pa_fixture_ids, list) or len(pa_fixture_ids) != 1:
        raise ValueError("PA must contain exactly one row fixture")
    if not isinstance(pb_fixture_ids, list) or len(pb_fixture_ids) != 1:
        raise ValueError("PB must contain exactly one row fixture")
    if not isinstance(pab_fixture_ids, list) or len(pab_fixture_ids) != 2:
        raise ValueError("PAB must contain exactly two row fixtures")
    if pab_fixture_ids != [pa_fixture_ids[0], pb_fixture_ids[0]]:
        raise ValueError("PAB must contain the same A fixture then B fixture as PA and PB")
    states: dict[str, dict[str, Any]] = {}
    base_prompt = parent["base_prompt_token_ids"]
    base_prefix = parent["prefix_token_ids"]
    for state_id in STATE_IDS:
        raw_state = state_map[state_id]
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"states[{state_id}] must be an object")
        kind = _state_kind(raw_state, state_id)
        fixture_ids = raw_state.get("row_fixture_ids")
        if not isinstance(fixture_ids, list) or any(not isinstance(v, str) for v in fixture_ids):
            raise ValueError(f"{state_id} requires ordered row_fixture_ids")
        if state_id == "P0" and fixture_ids:
            raise ValueError("P0 must have no row fixtures")
        if any(v not in fixtures for v in fixture_ids):
            raise ValueError(f"{state_id} references an unknown row fixture")
        prior_prompt = [*base_prompt, *base_prefix]
        rows: list[int] = []
        relation: list[dict[str, Any]] = []
        for fixture_index, fixture_id in enumerate(fixture_ids):
            fixture = fixtures[fixture_id]
            source_prompt = fixture["source_prompt_token_ids"]
            expected_start = parent["prefix_token_count"] if source_prompt == base_prompt and not rows else 0
            expected_prompt = prior_prompt
            is_native = kind == "native_contiguous"
            if is_native:
                if fixture.get("natural_support_state_kind") != "native_contiguous":
                    raise ValueError(f"{state_id} native state uses fixture {fixture_id} not declared native_contiguous")
                source_prompt_is_parent_base = not rows and source_prompt == base_prompt and parent["prefix_token_count"] > 0
                if source_prompt != expected_prompt and not source_prompt_is_parent_base:
                    raise ValueError(f"{state_id} fixture {fixture_id} source prompt is not exact prior state prompt")
                if fixture["source_token_span"][0] != expected_start:
                    raise ValueError(f"{state_id} fixture {fixture_id} is not contiguous with prior state")
            elif fixture.get("natural_support_state_kind") not in STATE_KINDS:
                raise ValueError(f"{state_id} forced fixture {fixture_id} lacks declared support state")
            rows.extend(fixture["token_ids"])
            prior_prompt = [*prior_prompt, *fixture["token_ids"]]
            relation.append({
                "fixture_id": fixture_id,
                "source_prompt_ids_sha256": fixture["source_prompt_ids_sha256"],
                "source_prompt_matches_expected": source_prompt == expected_prompt,
                "source_token_span": fixture["source_token_span"],
                "native_contiguous_proof": bool(is_native),
            })
        states[state_id] = {
            "state_id": state_id,
            "state_kind": kind,
            "row_fixture_ids": list(fixture_ids),
            "row_token_ids": rows,
            "row_token_ids_sha256": sha256_json(rows),
            "prefix_token_ids": [*base_prefix, *rows],
            "prefix_token_ids_sha256": sha256_json([*base_prefix, *rows]),
            "reconstructed_prompt_token_ids": prior_prompt,
            "reconstructed_prompt_token_ids_sha256": sha256_json(prior_prompt),
            "prompt_token_count": len(prior_prompt),
            "fixture_lineage": relation,
        }

    fixture_a = fixtures[pa_fixture_ids[0]]
    fixture_b = fixtures[pb_fixture_ids[0]]
    if fixture_a["natural_support_state_kind"] != "native_contiguous" or fixture_a["natural_support_state_id"] != "P0":
        raise ValueError("PA fixture A must declare native_contiguous natural support from P0")
    if fixture_b["natural_support_state_kind"] != "native_contiguous" or fixture_b["natural_support_state_id"] != "PA":
        raise ValueError("PB/PAB fixture B must declare native_contiguous natural support from PA")

    normalized_candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate_variants[{index}] must be an object")
        candidate_id = _required_text(raw, "candidate_id", f"candidate_variants[{index}]")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        candidate_ids.add(candidate_id)
        support_state_id = _required_text(raw, "support_state_id", candidate_id)
        if support_state_id not in states:
            raise ValueError(f"{candidate_id} support_state_id is unknown")
        candidate = _row_from_source(
            raw, manifest_path=manifest_path, parent_bundle=parent["bundle"], image_id=image_id, context=candidate_id
        )
        expected_prompt = states[support_state_id]["reconstructed_prompt_token_ids"]
        if candidate["source_prompt_token_ids"] != expected_prompt:
            raise ValueError(f"{candidate_id} source prompt does not equal support state {support_state_id}")
        expected_start = parent["prefix_token_count"] if support_state_id == "P0" else 0
        if candidate["source_token_span"][0] != expected_start:
            raise ValueError(f"{candidate_id} source span is not valid for support state {support_state_id}")
        normalized_candidates.append({
            **candidate,
            "candidate_id": candidate_id,
            "support_state_id": support_state_id,
            "support_state_kind": states[support_state_id]["state_kind"],
            "source_prompt_matches_base_prompt": candidate["source_prompt_token_ids"] == base_prompt,
            "source_prompt_matches_reconstructed_parent": True,
            "natural_support": candidate.get(
                "natural_support",
                {"state_kind": states[support_state_id]["state_kind"], "support_state_id": support_state_id, "source": "support_state", "count": 1, "policy": "declared_support"},
            ),
            "role": str(raw.get("role", "candidate")),
        })

    pairwise_raw = manifest.get("pairwise_contrasts", [])
    if not isinstance(pairwise_raw, list):
        raise ValueError("pairwise_contrasts must be a list")
    pairwise: list[dict[str, Any]] = []
    contrast_ids: set[str] = set()
    for index, raw in enumerate(pairwise_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"pairwise_contrasts[{index}] must be an object")
        contrast_id = _required_text(raw, "contrast_id", f"pairwise_contrasts[{index}]")
        if contrast_id in contrast_ids:
            raise ValueError(f"duplicate contrast_id {contrast_id}")
        contrast_ids.add(contrast_id)
        before = _required_text(raw, "before_state_id", contrast_id)
        after = _required_text(raw, "after_state_id", contrast_id)
        if before not in states or after not in states:
            raise ValueError(f"{contrast_id} references unknown state")
        ids = raw.get("candidate_ids", [item["candidate_id"] for item in normalized_candidates])
        if not isinstance(ids, list) or any(str(v) not in candidate_ids for v in ids):
            raise ValueError(f"{contrast_id} references unknown candidate")
        pairwise.append({"contrast_id": contrast_id, "before_state_id": before, "after_state_id": after, "candidate_ids": [str(v) for v in ids]})

    interaction_raw = manifest.get("two_by_two_interaction", manifest.get("interaction"))
    interaction: dict[str, Any] | None = None
    if interaction_raw is not None:
        if not isinstance(interaction_raw, Mapping):
            raise ValueError("two_by_two_interaction must be an object")
        ids = interaction_raw.get("candidate_ids", [item["candidate_id"] for item in normalized_candidates])
        if not isinstance(ids, list) or any(str(v) not in candidate_ids for v in ids):
            raise ValueError("two_by_two_interaction references unknown candidate")
        state_ids = interaction_raw.get("state_ids", {sid: sid for sid in STATE_IDS})
        if not isinstance(state_ids, Mapping) or any(str(state_ids.get(sid)) not in states for sid in STATE_IDS):
            raise ValueError("two_by_two_interaction state_ids must map P0, PA, PB, PAB to states")
        interaction = {
            "state_ids": {sid: str(state_ids[sid]) for sid in STATE_IDS},
            "candidate_ids": [str(v) for v in ids],
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "image_id": image_id,
        "parent": parent,
        "row_fixtures": fixtures,
        "states": states,
        "candidate_variants": normalized_candidates,
        "pairwise_contrasts": pairwise,
        "two_by_two_interaction": interaction,
    }


def paired_state_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Subtract the same frozen row score under two prefix states."""

    if before.get("token_ids_sha256") != after.get("token_ids_sha256"):
        raise ValueError("state scores refer to different candidate rows")
    if int(before.get("token_count", -1)) != int(after.get("token_count", -2)):
        raise ValueError("state scores have different candidate token counts")
    output: dict[str, Any] = {
        "token_ids_sha256": before["token_ids_sha256"],
        "token_count": int(before["token_count"]),
    }
    for phase in PHASES:
        left, right = before.get(phase), after.get(phase)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"missing phase {phase}")
        output[phase] = {
            "sum_delta": float(right["sum"]) - float(left["sum"]),
            "mean_delta": float(right["mean"]) - float(left["mean"]),
            "count": int(left["count"]),
        }
    return output


def factorial_interaction(
    scores_by_state: Mapping[str, Mapping[str, Any]], *, state_ids: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Compute (PAB-PA)-(PB-P0) for one frozen candidate."""

    mapping = {sid: sid for sid in STATE_IDS} if state_ids is None else {sid: str(state_ids[sid]) for sid in STATE_IDS}
    for sid in STATE_IDS:
        if mapping[sid] not in scores_by_state:
            raise ValueError(f"missing state {mapping[sid]} for factorial interaction")
    output: dict[str, Any] = {"formula": "(PAB - PA) - (PB - P0)", "state_ids": mapping}
    for phase in PHASES:
        p0 = scores_by_state[mapping["P0"]][phase]
        pa = scores_by_state[mapping["PA"]][phase]
        pb = scores_by_state[mapping["PB"]][phase]
        pab = scores_by_state[mapping["PAB"]][phase]
        output[phase] = {
            "sum_interaction": (float(pab["sum"]) - float(pa["sum"])) - (float(pb["sum"]) - float(p0["sum"])),
            "mean_interaction": (float(pab["mean"]) - float(pa["mean"])) - (float(pb["mean"]) - float(p0["mean"])),
            "count": int(p0["count"]),
        }
    return output


def emit_factorial_score(score: Mapping[str, Any], candidate: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Attach state and support provenance without overwriting phase mappings."""

    emitted = emit_candidate_score(score, candidate)
    emitted.update({
        "state_id": str(state["state_id"]),
        "state_kind": str(state["state_kind"]),
        "state_row_fixture_ids": list(state["row_fixture_ids"]),
        "state_prefix_token_ids_sha256": str(state["prefix_token_ids_sha256"]),
        "state_reconstructed_prompt_token_ids_sha256": str(state["reconstructed_prompt_token_ids_sha256"]),
        "candidate_support_state_id": str(candidate["support_state_id"]),
        "candidate_support_state_kind": str(candidate["support_state_kind"]),
    })
    return emitted


def _candidate_by_id(manifest: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for item in manifest["candidate_variants"]:
        if str(item["candidate_id"]) == str(candidate_id):
            return item
    raise ValueError(f"unknown candidate {candidate_id}")


def comparison_evidence(manifest: Mapping[str, Any], state_ids: Sequence[str]) -> dict[str, Any]:
    """Label whether a comparison includes the forced-replay PB state."""

    kinds = [str(manifest["states"][state_id]["state_kind"]) for state_id in state_ids]
    contains_forced = any(kind == "forced_replay" for kind in kinds)
    return {
        "evidence_scope": "descriptive_only" if contains_forced else "conclusion_eligible",
        "contains_forced_replay": contains_forced,
        "warning": (
            "This comparison includes forced_replay state PB; use it descriptively and do not call it native causal evidence."
            if contains_forced
            else None
        ),
    }


def build_pairwise_results(
    manifest: Mapping[str, Any], scores: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for contrast in manifest["pairwise_contrasts"]:
        before, after = contrast["before_state_id"], contrast["after_state_id"]
        values = {}
        for candidate_id in contrast["candidate_ids"]:
            values[candidate_id] = paired_state_delta(scores[before][candidate_id], scores[after][candidate_id])
        output.append({**contrast, **comparison_evidence(manifest, [before, after]), "candidate_deltas": values})
    return output


def build_interaction_results(
    manifest: Mapping[str, Any], scores: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any] | None:
    declaration = manifest.get("two_by_two_interaction")
    if declaration is None:
        return None
    state_ids = [declaration["state_ids"][sid] for sid in STATE_IDS]
    output: dict[str, Any] = {**declaration, **comparison_evidence(manifest, state_ids), "candidate_interactions": {}}
    for candidate_id in declaration["candidate_ids"]:
        output["candidate_interactions"][candidate_id] = factorial_interaction(
            {state_id: scores[state_id][candidate_id] for state_id in scores},
            state_ids=declaration["state_ids"],
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--runtime-dtype", choices=("config", "fp32"), default="config")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = validate_factorial_manifest(_read_json(manifest_path), manifest_path=manifest_path)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_jsonl = args.source_jsonl.expanduser().resolve(strict=True)

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime

    import contextlib
    import os

    @contextlib.contextmanager
    def temporary_cwd(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    with temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    if args.runtime_dtype == "fp32":
        qwen.model.to(dtype=torch.float32)
    qwen.model.eval()
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=qwen.model.config)
    raw_rows = load_raw_examples(source_jsonl)
    raw = next((row for row in raw_rows if str(row.metadata.get("source", {}).get("image_id")) == str(manifest["image_id"])), None)
    if raw is None:
        raise ValueError(f"image {manifest['image_id']} is absent from source JSONL")
    image_path = Path(raw.image.path).expanduser().resolve(strict=True)
    image_sha = sha256_file(image_path)
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    attention = _attention_implementation(qwen.model, resolved.config.model.attn_implementation)

    bundle_paths: dict[str, Path] = {"parent": Path(manifest["parent"]["source_bundle"])}
    bundle_paths.update({f"fixture:{key}": Path(value["source_bundle"]) for key, value in manifest["row_fixtures"].items()})
    bundle_paths.update({f"candidate:{value['candidate_id']}": Path(value["source_bundle"]) for value in manifest["candidate_variants"]})
    active_lineage: dict[str, Any] = {}
    seen: set[str] = set()
    for label, path in bundle_paths.items():
        resolved_path = path.expanduser().resolve(strict=True)
        if str(resolved_path) in seen:
            continue
        seen.add(str(resolved_path))
        active_lineage[str(resolved_path)] = validate_active_source_lineage(
            _read_json(resolved_path), image_id=str(manifest["image_id"]), image_sha256=image_sha,
            image_width=int(raw.image.width), image_height=int(raw.image.height), model_identity=model_identity,
            tokenizer_identity=tokenizer_identity, attention_implementation=attention,
            model_config_dtype=str(resolved.config.model.dtype),
        )

    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    if list(prompt_record.prompt_token_ids) != list(manifest["parent"]["base_prompt_token_ids"]):
        raise ValueError("active processor prompt does not equal frozen row-zero base prompt")
    image_plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
    model_inputs = image_plan.model_inputs_by_row_id[prompt_record.row_id]
    image_grid = model_inputs.get("image_grid_thw")
    if not isinstance(image_grid, torch.Tensor):
        raise ValueError("materialized image plan lacks image_grid_thw")

    row_entry_id = int(OBJECT_REF_START)
    eos_id = qwen.tokenizer.eos_token_id
    if eos_id is None or int(eos_id) < 0:
        raise ValueError("tokenizer does not expose a valid eos_token_id")
    terminal_scores: dict[str, Any] = {}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for state_id in STATE_IDS:
        state = manifest["states"][state_id]
        state_prompt = state["reconstructed_prompt_token_ids"]
        state_logits = _forward_logits(qwen.model, model_inputs, state_prompt, image_grid)
        terminal_scores[state_id] = terminal_boundary_score(state_logits, boundary_length=len(state_prompt), row_entry_token_id=row_entry_id, terminal_token_id=int(eos_id))
        per_candidate: dict[str, dict[str, Any]] = {}
        for candidate in manifest["candidate_variants"]:
            row_logits = _forward_logits(qwen.model, model_inputs, [*state_prompt, *candidate["token_ids"]], image_grid)
            raw_score = score_token_logits(row_logits, boundary_length=len(state_prompt), row_tokens=candidate["token_ids"])
            per_candidate[str(candidate["candidate_id"])] = emit_factorial_score(raw_score, candidate, state)
        scores[state_id] = per_candidate

    pairwise = build_pairwise_results(manifest, scores)
    interaction = build_interaction_results(manifest, scores)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": "2026-07-17-next-row-probability-transition-and-causal-source-trace",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "image_id": manifest["image_id"],
        "parent": {key: manifest["parent"][key] for key in ("source_bundle", "source_bundle_sha256", "prefix_token_count", "prefix_token_ids_sha256", "reconstructed_prompt_token_ids_sha256")},
        "states": {
            state_id: {key: state[key] for key in ("state_id", "state_kind", "row_fixture_ids", "row_token_ids_sha256", "prefix_token_ids_sha256", "reconstructed_prompt_token_ids_sha256", "prompt_token_count")}
            for state_id, state in manifest["states"].items()
        },
        "candidate_variants": [{key: candidate[key] for key in ("candidate_id", "owner", "category", "role", "support_state_id", "support_state_kind", "source_bundle", "source_bundle_sha256", "source_token_span", "token_ids_sha256", "token_count")} for candidate in manifest["candidate_variants"]],
        "raw_scores": scores,
        "terminal_boundaries": terminal_scores,
        "pairwise_contrasts": pairwise,
        "two_by_two_interaction": interaction,
        "runtime": {
            "physical_batch_size": 1,
            "score_accumulation_dtype": "torch.float32",
            "model_dtype": _runtime_model_dtype_summary(qwen.model),
            "runtime_dtype_mode": str(args.runtime_dtype),
            "config_path": str(config_path),
            "config_sha256": config_sha256_json(resolved.config.model_dump(mode="json")),
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "repetition_penalty_processing": False,
            "cache": False,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "attention_implementation": attention,
            "row_entry_token_id": row_entry_id,
            "eos_token_id": int(eos_id),
            "image_identity": {"path": str(image_path), "sha256": image_sha, "width": int(raw.image.width), "height": int(raw.image.height)},
        },
        "lineage_proof": {"active_source_bundles": active_lineage},
        "forbidden_comparisons": ["scores are raw teacher-forced log-likelihoods, not a probability distribution", "terminal boundary margins are not compared with full-row scores"],
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "receipt.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {output_path}")
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
