#!/usr/bin/env python3
"""Deterministically summarize sealed Sorted owner-basin score rows on CPU.

The runtime scorer is an untrusted producer.  This consumer therefore checks
the score artifact, every upstream digest named by its receipt, the frozen v2
rule identity, row cardinality/foreign keys, raw likelihood attestations, and
the fully identity-bound conditional-y1 surface before calling the pure core.
Unknown, unreviewed, or globally ambiguous owners are retained as raw-only
neutral records and never enter a decision-bearing basin measurement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import sorted_owner_basin_landscape as core  # noqa: E402


SUMMARY_SCHEMA_VERSION = "sorted_owner_basin_landscape_summary.v1"
RECEIPT_SCHEMA_VERSION = "sorted_owner_basin_landscape_summary_receipt.v1"
SCORE_SCHEMA_VERSION = "landscape_scores.v1"
SCORE_RECEIPT_SCHEMA_VERSION = "landscape_scores_receipt.v1"
SUMMARY_NAME = "landscape-summary.json"
RECEIPT_NAME = "landscape-summary-receipt.json"
NEUTRAL_STATUSES = frozenset({"unresolved", "unreviewed", "globally_ambiguous"})
REVIEWED_CANDIDATE_STATUS = "reviewed_candidate"
SEALED_CONTEXT_STATUS = "sealed_context"
LINEAGE_REVIEW_STATUSES = frozenset(
    {REVIEWED_CANDIDATE_STATUS, SEALED_CONTEXT_STATUS}
)
REVIEW_LINEAGE_FIELDS = frozenset(
    {
        "source_pred_row_id",
        "review_status",
        "registry_id",
        "foreign_keys",
        "source_binding",
        "lineage_sha256",
    }
)
COORDINATE_NAMES = ("x1", "y1", "x2", "y2")
LANDSCAPE_SURFACES = ("canonical_description_free", "restricted_gt_target")
CACHE_PARITY_ATOL = 1e-6
CACHE_PARITY_RTOL = 1e-5
KV_CACHE_SCORING_BACKEND = "kv_cache"
FULL_REFORWARD_SCORING_BACKEND = "full_reforward_uncached"
PARITY_FAILURE_FALLBACK_TRIGGER = "mandatory_cache_parity_failed"
DECISION_BEARING_SCORE_USE = "decision_bearing"
ACCEPTED_SCORER_EXECUTION_STATUSES = frozenset(
    {
        "live_model_scoring_completed_artifacts_sealed",
        "test_fixture_scorer_shaped_output",
    }
)
REQUIRED_CORE_CANDIDATE_FIELDS = frozenset(
    {
        "bank_name",
        "source_id",
        "extent_submode",
        "proposal_measure_id",
        "coordinate_bins",
        "role_id",
        "identity_kind",
        "identity_id",
        "geometry_identity",
    }
)
REQUIRED_ATTESTATION_FIELDS = frozenset(
    {
        "attestation_kind",
        "diagnostic_owner_id",
        "gt_owner_id",
        "image_identity",
        "gt_box",
        "canonical_description_text",
        "canonical_description_token_digest",
        "context_id",
        "context_token_digest",
        "tokenizer_identity",
        "model_identity",
        "runtime_identity",
        "geometry_identity_schema",
        "rule_digest",
        "declared_x1_bins",
        "per_x1_receipt_digests",
        "completeness_digest",
    }
)


class SummaryContractError(RuntimeError):
    """A conclusion-bearing aggregation precondition was not proven."""


def _fail(message: str) -> NoReturn:
    raise SummaryContractError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_rule_digest(rules: core.LandscapeRules) -> str:
    return rules.rule_digest


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{line_number} is not valid JSON: {exc}")
    return rows


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    except FileExistsError:
        _fail(f"refusing to overwrite immutable output: {path}")


def _validate_receipt_sources(receipt: Mapping[str, Any]) -> dict[str, str]:
    source_digests = _mapping(receipt.get("source_digests"), "score receipt.source_digests")
    result: dict[str, str] = {}
    for name, untyped in sorted(source_digests.items()):
        entry = _mapping(untyped, f"score receipt.source_digests.{name}")
        source_path = Path(_string(entry.get("path"), f"source {name}.path")).expanduser()
        expected = _sha256(entry.get("sha256"), f"source {name}.sha256")
        actual = sha256_file(source_path.resolve(strict=True))
        if actual != expected:
            _fail(f"score receipt source digest is stale for {name}: expected {expected}, observed {actual}")
        manifest_expected = entry.get("manifest_expected")
        if manifest_expected is not None and _sha256(manifest_expected, f"source {name}.manifest_expected") != actual:
            _fail(f"score receipt source {name} no longer matches its artifact-manifest digest")
        if entry.get("match") is False:
            _fail(f"score receipt records a failed source match for {name}")
        result[str(name)] = actual
    if not result:
        _fail("score receipt.source_digests must not be empty")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{context} must be a non-negative integer")
    return value


def _depth_counts(value: Any, context: str) -> dict[str, int]:
    source = _mapping(value, context)
    result: dict[str, int] = {}
    for depth, count in source.items():
        if not isinstance(depth, str) or not depth.isdigit():
            _fail(f"{context} depths must be decimal-string non-negative integers")
        result[depth] = _nonnegative_int(count, f"{context}[{depth}]")
    return result


def validate_scoring_backend_admission(
    receipt: Mapping[str, Any], *, row_count: int
) -> dict[str, Any]:
    """Admit exactly one parity-selected scoring backend with complete counts."""

    parity = _mapping(
        receipt.get("mandatory_cache_parity_gate"),
        "score receipt.mandatory_cache_parity_gate",
    )
    parity_status = parity.get("status")
    if parity_status not in {"passed", "failed"}:
        _fail("mandatory cache parity must be preserved as passed or failed")
    if parity.get("atol") != CACHE_PARITY_ATOL or parity.get("rtol") != CACHE_PARITY_RTOL:
        _fail("mandatory cache parity tolerance differs from the frozen exact tolerance")

    admission = dict(
        _mapping(
            receipt.get("scoring_backend_admission"),
            "score receipt.scoring_backend_admission",
        )
    )
    digest = admission.pop("sha256", None)
    if _sha256(digest, "scoring backend admission sha256") != sha256_json(admission):
        _fail("scoring backend admission digest does not reconstruct")
    if admission.get("status") != "passed":
        _fail("scoring backend admission did not pass")
    decision_use = admission.get("decision_use")
    if decision_use is None:
        legacy_policy = admission.get("cache_admission_policy")
        legacy_effective_mode = (
            legacy_policy.get("effective_mode")
            if isinstance(legacy_policy, Mapping)
            else None
        )
        if legacy_effective_mode == "relaxed_coordinate_behavior":
            _fail(
                "legacy relaxed-cache admission lacks a decision-use seal and "
                "cannot enter a decision-bearing summary"
            )
        decision_use = DECISION_BEARING_SCORE_USE
    if decision_use != DECISION_BEARING_SCORE_USE:
        _fail(
            "scoring backend admission is probe-only and cannot enter a "
            "decision-bearing summary"
        )
    if admission.get("parity_status") != parity_status:
        _fail("scoring backend admission relabels the mandatory parity result")
    if admission.get("atol") != CACHE_PARITY_ATOL or admission.get("rtol") != CACHE_PARITY_RTOL:
        _fail("scoring backend admission tolerance differs from the frozen exact tolerance")
    if admission.get("backend_mixing_detected") is not False:
        _fail("scoring backend admission does not prove an unmixed score path")
    if admission.get("score_row_count") != row_count:
        _fail("scoring backend admission row count differs from the score artifact")

    selected = admission.get("selected_backend")
    expected_by_parity = {
        "passed": {
            "selected_backend": KV_CACHE_SCORING_BACKEND,
            "cache_enabled": True,
            "use_cache": True,
            "fallback_trigger": None,
            "cache_score_row_count": row_count,
            "uncached_score_row_count": 0,
        },
        "failed": {
            "selected_backend": FULL_REFORWARD_SCORING_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            "fallback_trigger": PARITY_FAILURE_FALLBACK_TRIGGER,
            "cache_score_row_count": 0,
            "uncached_score_row_count": row_count,
        },
    }[str(parity_status)]
    for field, expected in expected_by_parity.items():
        if admission.get(field) != expected:
            _fail(
                f"scoring backend admission {field} is inconsistent with parity {parity_status}"
            )
    if admission.get("all_score_rows_backend") != selected:
        _fail("scoring backend admission does not bind every row to the selected backend")

    forward_accounting = _mapping(
        admission.get("forward_accounting"), "scoring backend forward_accounting"
    )
    entries = forward_accounting.get("per_context_group")
    if not isinstance(entries, list) or not entries:
        _fail("scoring backend admission requires non-empty per-context/group accounting")
    rebuilt_by_depth: dict[str, dict[str, int]] = {
        "logical_token_step_requests": {},
        "actual_forward_calls": {},
        "memo_hits": {},
        "memo_entries": {},
    }
    rebuilt: dict[str, Any] = {
        "context_group_count": len(entries),
        "root_calls": 0,
        "logical_token_step_requests": 0,
        "actual_forward_calls": 0,
        "memo_hits": 0,
        "memo_entries": 0,
    }
    seen_groups: set[tuple[str, str]] = set()
    depth_fields = {
        "logical_token_step_requests": "logical_token_step_requests_by_depth",
        "actual_forward_calls": "actual_forward_calls_by_depth",
        "memo_hits": "memo_hits_by_depth",
        "memo_entries": "memo_entries_by_depth",
    }
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"scoring backend accounting entry {index}")
        if entry.get("scoring_backend") != selected:
            _fail("per-context/group accounting mixes scoring backends")
        context_id = _string(entry.get("context_id"), "accounting context_id")
        group_id = _string(entry.get("group_id"), "accounting group_id")
        if (context_id, group_id) in seen_groups:
            _fail("duplicate scoring backend context/group accounting")
        seen_groups.add((context_id, group_id))
        _nonnegative_int(entry.get("root_prefix_length"), "accounting root_prefix_length")
        for scalar in ("root_calls", "logical_token_step_requests", "actual_forward_calls"):
            rebuilt[scalar] += _nonnegative_int(entry.get(scalar), f"accounting {scalar}")
        parsed_depths: dict[str, dict[str, int]] = {}
        for target, source in depth_fields.items():
            counts = _depth_counts(entry.get(source), f"accounting {source}")
            parsed_depths[target] = counts
            for depth, count in counts.items():
                rebuilt_by_depth[target][depth] = rebuilt_by_depth[target].get(depth, 0) + count
            if target in {"memo_hits", "memo_entries"}:
                rebuilt[target] += sum(counts.values())
        if sum(parsed_depths["logical_token_step_requests"].values()) != entry.get(
            "logical_token_step_requests"
        ):
            _fail("logical step depth accounting does not reconstruct")
        if sum(parsed_depths["actual_forward_calls"].values()) != entry.get(
            "actual_forward_calls"
        ):
            _fail("actual forward depth accounting does not reconstruct")
        if parsed_depths["actual_forward_calls"].get("0") != entry.get("root_calls"):
            _fail("root forward accounting does not reconstruct from depth zero")
        if selected == FULL_REFORWARD_SCORING_BACKEND:
            if set(parsed_depths["memo_hits"]) - {"1", "2"}:
                _fail("uncached fallback memo hits exist outside exact depths 1 and 2")
            if set(parsed_depths["memo_entries"]) != {"0", "1", "2"}:
                _fail("uncached fallback memo entries must record exact depths 0, 1, and 2")
            if entry.get("retained_relative_depths") != [0, 1, 2]:
                _fail("uncached fallback retained depths differ from exact root/depth-1/depth-2 memo")
            logical_depths = set(parsed_depths["logical_token_step_requests"])
            actual_depths = set(parsed_depths["actual_forward_calls"]) - {"0"}
            memo_hit_depths = set(parsed_depths["memo_hits"])
            if "0" in logical_depths or "0" in memo_hit_depths:
                _fail("uncached fallback step or memo-hit accounting contains depth zero")
            if any(
                count == 0
                for count in parsed_depths["logical_token_step_requests"].values()
            ) or any(
                count == 0
                for depth, count in parsed_depths["actual_forward_calls"].items()
                if depth != "0"
            ):
                _fail("uncached fallback step accounting contains an empty extra depth")
            step_depths = logical_depths | actual_depths | memo_hit_depths
            if not step_depths:
                _fail("uncached fallback requires positive-depth step accounting")
            for depth in sorted(step_depths, key=int):
                logical_requests = parsed_depths[
                    "logical_token_step_requests"
                ].get(depth, 0)
                actual_forwards = parsed_depths["actual_forward_calls"].get(
                    depth, 0
                )
                memo_hits = parsed_depths["memo_hits"].get(depth, 0)
                if logical_requests != actual_forwards + memo_hits:
                    _fail(
                        "uncached fallback forward accounting identity failed at "
                        + f"depth {depth}: logical requests must equal actual forwards plus memo hits"
                    )
        elif parsed_depths["memo_hits"] or parsed_depths["memo_entries"]:
            _fail("cache backend accounting unexpectedly claims uncached memo entries")
    rebuilt["by_depth"] = rebuilt_by_depth
    if forward_accounting.get("aggregate") != rebuilt:
        _fail("aggregate scoring backend accounting does not reconstruct from context/groups")
    return {**admission, "sha256": digest}


def _validate_score_receipt(
    receipt: Mapping[str, Any], *, scores_path: Path, rules_path: Path, rules: core.LandscapeRules, row_count: int
) -> tuple[dict[str, str], dict[str, Any]]:
    if receipt.get("schema_version") != SCORE_RECEIPT_SCHEMA_VERSION:
        _fail(f"score receipt.schema_version must be {SCORE_RECEIPT_SCHEMA_VERSION!r}")
    if receipt.get("runtime_execution_status") not in ACCEPTED_SCORER_EXECUTION_STATUSES:
        _fail(
            "score receipt runtime_execution_status does not attest sealed live scoring or an explicit test fixture"
        )
    if (
        rules.contract_mode == "production"
        and receipt.get("runtime_execution_status")
        != "live_model_scoring_completed_artifacts_sealed"
    ):
        _fail("production rules require sealed live scoring")
    provenance = _mapping(
        receipt.get("implementation_provenance"),
        "score receipt.implementation_provenance",
    )
    git_head = _string(provenance.get("git_head"), "implementation git_head")
    if len(git_head) != 40 or any(character not in "0123456789abcdef" for character in git_head):
        _fail("score receipt implementation git_head must be a full lowercase commit digest")
    _sha256(
        provenance.get("git_dirty_diff_sha256"),
        "implementation dirty-diff digest",
    )
    relevant_files = _mapping(
        provenance.get("relevant_file_digests"),
        "implementation relevant_file_digests",
    )
    if not relevant_files:
        _fail("score receipt must bind at least one relevant implementation file")
    for name, digest in relevant_files.items():
        _sha256(digest, f"implementation relevant file {name}")
    command_environment = _mapping(
        receipt.get("command_environment"), "score receipt.command_environment"
    )
    for field in (
        "cwd",
        "python_executable",
        "python_version",
        "torch_version",
        "transformers_version",
    ):
        _string(command_environment.get(field), f"score command environment {field}")
    if receipt.get("runtime_execution_status") == "live_model_scoring_completed_artifacts_sealed":
        runtime_admission = _mapping(
            receipt.get("runtime_identity_admission"),
            "score receipt.runtime_identity_admission",
        )
        config_admission = _mapping(
            runtime_admission.get("config"), "runtime config admission"
        )
        preload = _mapping(runtime_admission.get("preload"), "runtime preload admission")
        postload = _mapping(runtime_admission.get("postload"), "runtime postload admission")
        if (
            config_admission.get("status") != "passed"
            or config_admission.get("model_dtype") != "fp32"
            or preload.get("status") != "passed"
            or postload.get("status") != "passed"
        ):
            _fail(
                "live scorer receipt lacks passed config, pre-load, and post-load runtime admission"
            )
        _string(config_admission.get("source_jsonl"), "runtime config source_jsonl")
        _sha256(
            config_admission.get("binding_sha256"),
            "runtime config binding_sha256",
        )
        for field in (
            "identity_file_sha256",
            "identity_receipt_digest",
            "resolved_infer_fingerprint",
            "source_panel_sha256",
            "tokenizer_identity_sha256",
            "model_identity_sha256",
            "runtime_identity_sha256",
            "admission_sha256",
        ):
            _sha256(preload.get(field), f"runtime preload {field}")
        _sha256(postload.get("projection_sha256"), "runtime postload projection_sha256")
        runtime_identity_sha256 = _sha256(
            postload.get("runtime_identity_sha256"),
            "runtime postload runtime_identity_sha256",
        )
        if runtime_identity_sha256 != preload.get("runtime_identity_sha256"):
            _fail("post-load runtime identity does not match the pre-load frozen identity")
        if (
            _sha256(
                postload.get("observed_runtime_identity_sha256"),
                "runtime postload observed_runtime_identity_sha256",
            )
            != runtime_identity_sha256
        ):
            _fail("post-load observed runtime identity hash is not the frozen identity hash")
        if postload.get("expected_projection") != postload.get("observed_projection"):
            _fail("live scorer runtime projection is not an exact admitted match")
    output = _mapping(receipt.get("output_artifacts"), "score receipt.output_artifacts")
    score_entry = _mapping(output.get("landscape_scores"), "score receipt.output_artifacts.landscape_scores")
    expected_score_digest = _sha256(score_entry.get("sha256"), "score artifact sha256")
    if sha256_file(scores_path) != expected_score_digest:
        _fail("landscape-scores.jsonl digest does not match its sealed scorer receipt")
    if score_entry.get("row_count") != row_count:
        _fail("landscape-scores.jsonl cardinality does not match its sealed scorer receipt")
    decision = _mapping(receipt.get("decision_rules"), "score receipt.decision_rules")
    if _sha256(decision.get("file_sha256"), "score receipt decision-rules file_sha256") != sha256_file(rules_path):
        _fail("decision-rule file digest does not match the scorer receipt")
    if decision.get("core_rule_digest") != _core_rule_digest(rules):
        _fail("scorer receipt rule digest is incompatible with the landed pure-core v2 rules")
    scoring_backend_admission = validate_scoring_backend_admission(
        receipt, row_count=row_count
    )
    return _validate_receipt_sources(receipt), scoring_backend_admission


def _validate_phase_a_freeze_admission(
    receipt: Mapping[str, Any],
    *,
    rule_document: Mapping[str, Any],
    rules: core.LandscapeRules,
) -> None:
    structural_status = rule_document.get("structural_status")
    if structural_status not in {"draft_pre_smoke", "sealed_non_c_smoke"}:
        return
    admission = _mapping(
        receipt.get("phase_a_freeze_admission"),
        "score receipt.phase_a_freeze_admission",
    )
    if structural_status == "draft_pre_smoke":
        if (
            admission.get("status") != "not_applicable_non_sentinel"
            or admission.get("sentinel_dependency_consumed") is not False
        ):
            _fail("draft control score receipt carries a hidden sentinel dependency")
        return
    if (
        admission.get("status") != "passed_post_phase_a_freeze"
        or admission.get("sentinel_dependency_consumed") is not True
    ):
        _fail("sealed sentinel score receipt lacks passed post-freeze admission")
    declared_admission_sha256 = _sha256(
        admission.get("admission_sha256"), "Phase-A admission digest"
    )
    admission_payload = dict(admission)
    admission_payload.pop("admission_sha256")
    if sha256_json(admission_payload) != declared_admission_sha256:
        _fail("Phase-A admission digest is stale")
    rule_binding = _mapping(
        rule_document.get("non_c_smoke_freeze_receipt"),
        "sentinel rules non-C smoke freeze binding",
    )
    freeze = _mapping(admission.get("freeze_receipt"), "Phase-A freeze receipt")
    sentinel_rules = _mapping(
        admission.get("sentinel_rules"), "Phase-A sentinel rules"
    )
    candidate_receipt = _mapping(
        admission.get("candidate_receipt"), "Phase-A candidate receipt"
    )
    if freeze.get("c_outcomes_read") is not False:
        _fail("sealed sentinel score receipt did not consume a C-blind freeze")
    if (
        freeze.get("file_sha256") != rule_binding.get("sha256")
        or freeze.get("control_decision_rules_sha256")
        != rule_binding.get("control_decision_rules_sha256")
        or freeze.get("semantic_core_sha256") != rules.rule_digest
        or rule_binding.get("semantic_core_sha256") != rules.rule_digest
    ):
        _fail("sealed sentinel score receipt has stale Phase-A freeze lineage")
    if (
        sentinel_rules.get("file_sha256") != sha256_file(Path(_string(
            sentinel_rules.get("path"), "Phase-A sentinel rules path"
        )).resolve(strict=True))
        or sentinel_rules.get("semantic_core_sha256") != rules.rule_digest
        or sentinel_rules.get("bound_freeze_receipt_sha256")
        != freeze.get("file_sha256")
    ):
        _fail("Phase-A sentinel-rule provenance is stale")
    if (
        candidate_receipt.get("sentinel_rules_sha256")
        != sentinel_rules.get("file_sha256")
        or candidate_receipt.get("semantic_core_sha256") != rules.rule_digest
    ):
        _fail("Phase-A candidate receipt does not bind the sentinel rule lineage")


def _scoring_contract(
    rule_document: Mapping[str, Any], rules: core.LandscapeRules
) -> tuple[int, int, Mapping[str, Any]]:
    scoring = _mapping(rule_document.get("scoring_contract"), "rules.scoring_contract")
    start = _integer(
        scoring.get("coordinate_token_id_start"),
        "rules.scoring_contract.coordinate_token_id_start",
    )
    end = _integer(
        scoring.get("coordinate_token_id_end_exclusive"),
        "rules.scoring_contract.coordinate_token_id_end_exclusive",
    )
    if end - start != rules.coordinate_max - rules.coordinate_min + 1:
        _fail("rules.scoring_contract coordinate vocabulary does not match the core coordinate domain")
    foil_digests = _mapping(scoring.get("foil_set_digests"), "rules.scoring_contract.foil_set_digests")
    return start, end, foil_digests


def _validate_free_surface_execution(
    entry: Mapping[str, Any], surface_rows: Sequence[Mapping[str, Any]], *, production: bool
) -> str:
    declared = [
        _string(value, "free-surface declared request ID")
        for value in _sequence(
            entry.get("declared_request_ids"),
            "free-surface declared request IDs",
        )
    ]
    executed = _sequence(
        entry.get("execution_receipts"), "free-surface execution receipts"
    )
    if entry.get("execution_receipts_sha256") != sha256_json(
        [dict(_mapping(value, "free-surface execution receipt")) for value in executed]
    ):
        _fail("free-surface execution-receipt digest is stale")
    executed_ids: list[str] = []
    expected_counts: dict[str, int] = {}
    for value in executed:
        receipt = dict(_mapping(value, "free-surface execution receipt"))
        request_id = _string(receipt.get("request_id"), "free execution request ID")
        if (
            receipt.get("schema_version")
            != "sorted_owner_basin_free_tree_execution.v1"
            or receipt.get("surface") != "free_coordinate_tree"
            or receipt.get("status") != "executed"
        ):
            _fail("free-surface receipt does not attest an executed frozen tree")
        declared_digest = _sha256(
            receipt.pop("receipt_sha256", None), "free execution receipt digest"
        )
        if sha256_json(receipt) != declared_digest:
            _fail(f"free execution receipt digest is stale for {request_id}")
        counts = _mapping(receipt.get("counts"), "free execution counts")
        complete_box_count = _integer(
            counts.get("complete_box_count"), "free execution complete-box count"
        )
        if complete_box_count < 0:
            _fail("free execution complete-box count must be non-negative")
        executed_ids.append(request_id)
        expected_counts[request_id] = complete_box_count
    if len(declared) != len(set(declared)) or len(executed_ids) != len(
        set(executed_ids)
    ):
        _fail("free-surface request IDs must be unique")
    if sorted(declared) != sorted(executed_ids):
        _fail("every declared free-tree root must have exactly one execution receipt")
    if entry.get("executed_request_ids") != sorted(executed_ids):
        _fail("free-surface executed request IDs drift from execution receipts")
    observed_counts: dict[str, int] = {}
    for row in surface_rows:
        execution = _mapping(
            row.get("free_surface_execution"), "free score-row execution binding"
        )
        request_id = _string(
            execution.get("free_tree_request_id"), "free score-row request ID"
        )
        observed_counts[request_id] = observed_counts.get(request_id, 0) + 1
    if observed_counts != {
        request_id: count
        for request_id, count in expected_counts.items()
        if count > 0
    }:
        _fail("free score rows do not match their per-request execution counts")
    if production and not declared:
        _fail("production aggregation requires at least one declared free-tree root")
    if surface_rows:
        return "passed"
    if declared:
        return "executed_bounded_null_non_evidence"
    return "not_scored_test_fixture"


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, float, str]:
    penalty = _finite(row.get("native_repetition_penalty_stratum"), "row policy stratum")
    return (
        _string(row.get("diagnostic_owner_id"), "row.diagnostic_owner_id"),
        _string(row.get("image_id"), "row.image_id"),
        _string(row.get("context_id"), "row.context_id"),
        _string(row.get("landscape_surface"), "row.landscape_surface"),
        penalty,
        _string(row.get("foil_set_id"), "row.foil_set_id"),
    )


def _validate_common_row(
    row: Mapping[str, Any], rules: core.LandscapeRules, foil_digests: Mapping[str, Any]
) -> None:
    if row.get("schema_version") != SCORE_SCHEMA_VERSION:
        _fail(f"score row schema_version must be {SCORE_SCHEMA_VERSION!r}")
    if row.get("landscape_surface") not in LANDSCAPE_SURFACES:
        _fail(f"score row.landscape_surface must be one of {LANDSCAPE_SURFACES}")
    if row.get("rule_digest") != _core_rule_digest(rules):
        _fail(f"score row {_string(row.get('candidate_id'), 'row.candidate_id')} has a stale rule digest")
    penalty = _finite(row.get("native_repetition_penalty_stratum"), "row policy stratum")
    if penalty not in {1.0, 1.1}:
        _fail("score row native repetition-penalty stratum must be 1.0 or 1.10")
    foil_set_id = _string(row.get("foil_set_id"), "row.foil_set_id")
    expected_foil_digest = _sha256(
        foil_digests.get(foil_set_id), f"rules foil-set digest for {foil_set_id}"
    )
    if _sha256(row.get("foil_set_digest"), "row.foil_set_digest") != expected_foil_digest:
        _fail(f"score row foil-set digest is stale for {foil_set_id}")
    owner_status = row.get("owner_status")
    if owner_status not in {"gt", "aux", "unresolved"}:
        _fail("score row.owner_status must be gt, aux, or unresolved")
    ambiguity = _mapping(row.get("upstream_adjudication"), "row.upstream_adjudication")
    if ambiguity.get("global_ambiguity_status") not in {"clear", "globally_ambiguous"}:
        _fail("row.upstream_adjudication.global_ambiguity_status must be clear or globally_ambiguous")
    proposal = _mapping(row.get("proposal_verification"), "row.proposal_verification")
    if proposal.get("self_consistency") != "passed" or proposal.get("pure_core_recomputation") != "passed":
        _fail("every score row must pass scorer-side proposal self-consistency and pure-core recomputation")


def _validate_vocab_attestations(raw: Mapping[str, Any], *, rule_digest: str) -> None:
    attestations = _mapping(raw.get("vocab_attestation"), "raw_model_logprob.vocab_attestation")
    if set(attestations) != set(COORDINATE_NAMES):
        _fail("complete-box raw likelihood must attest all four coordinate vocabulary rows")
    signatures: set[tuple[Any, ...]] = set()
    for name in COORDINATE_NAMES:
        entry = _mapping(attestations[name], f"raw vocab attestation {name}")
        if entry.get("filtered") is not False or entry.get("rule_digest") != rule_digest:
            _fail("raw likelihood vocabulary attestation is filtered or bound to stale rules")
        signatures.add(
            (
                entry.get("vocab_size"),
                entry.get("domain_digest"),
                entry.get("tokenizer_identity_digest"),
                entry.get("model_identity_digest"),
                entry.get("runtime_receipt_id"),
            )
        )
    if len(signatures) != 1:
        _fail("the four raw coordinate likelihoods do not share one vocabulary/runtime identity")


def _is_reviewed(row: Mapping[str, Any]) -> bool:
    review_status = row.get("review_status")
    if review_status == "reviewed":
        return True
    if review_status not in LINEAGE_REVIEW_STATUSES:
        return False

    adjudication = _mapping(row.get("upstream_adjudication"), "row.upstream_adjudication")
    if adjudication.get("global_ambiguity_status") != "clear":
        return False
    lineage = _mapping(
        adjudication.get("source_review_foreign_key_lineage"),
        "row.upstream_adjudication.source_review_foreign_key_lineage",
    )
    if set(lineage) != REVIEW_LINEAGE_FIELDS:
        _fail(
            f"{review_status} lineage must contain exactly the frozen registry lineage fields"
        )
    if lineage.get("review_status") != review_status:
        _fail(f"{review_status} lineage relabels its review status")
    source_pred_row_id = lineage.get("source_pred_row_id")
    if review_status == REVIEWED_CANDIDATE_STATUS or source_pred_row_id is not None:
        _string(source_pred_row_id, f"{review_status} source_pred_row_id")
    _string(lineage.get("registry_id"), f"{review_status} registry_id")
    if not _mapping(lineage.get("foreign_keys"), f"{review_status} foreign_keys"):
        _fail(f"{review_status} foreign_keys must not be empty")
    if not _mapping(lineage.get("source_binding"), f"{review_status} source_binding"):
        _fail(f"{review_status} source_binding must not be empty")
    expected_digest = _sha256(
        lineage.get("lineage_sha256"), f"{review_status} lineage_sha256"
    )
    lineage_payload = dict(lineage)
    lineage_payload.pop("lineage_sha256")
    if expected_digest != sha256_json(lineage_payload):
        _fail(f"{review_status} registry lineage digest is stale")
    return True


def _neutral_reason(row: Mapping[str, Any]) -> str | None:
    if _mapping(row.get("upstream_adjudication"), "row.upstream_adjudication").get(
        "global_ambiguity_status"
    ) == "globally_ambiguous":
        return "globally_ambiguous"
    if row.get("owner_status") == "unresolved":
        return "unresolved"
    if not _is_reviewed(row):
        return "unreviewed"
    return None


def _full_attestation_by_key(
    receipt: Mapping[str, Any], rules: core.LandscapeRules
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    raw = receipt.get("conditional_y1_attestations")
    if raw is None:
        _fail(
            "score receipt is missing fully identity-bound conditional_y1_attestations with GT box, canonical text/token digest, "
            "exact context digest, tokenizer/model/runtime identities, declared x1 bins, and per-x1 raw receipts; "
            "the scorer-local conditional_y1_completeness digest is insufficient"
        )
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for index, untyped in enumerate(_sequence(raw, "score receipt.conditional_y1_attestations")):
        item = _mapping(untyped, f"conditional_y1_attestations[{index}]")
        missing = sorted(REQUIRED_ATTESTATION_FIELDS.difference(item))
        if missing:
            _fail(f"conditional-y1 attestation is missing identity-bound fields: {missing}")
        if item.get("rule_digest") != _core_rule_digest(rules):
            _fail("conditional-y1 attestation has a stale rule digest")
        key = (
            _string(item.get("diagnostic_owner_id"), "attestation diagnostic owner"),
            _string(item.get("image_identity"), "attestation image identity"),
            _string(item.get("context_id"), "attestation context"),
            _string(item.get("landscape_surface"), "attestation landscape surface"),
        )
        if key in result:
            _fail(f"duplicate conditional-y1 attestation for {key}")
        result[key] = item
    return result


def _image_identity_for_group(rows: Sequence[Mapping[str, Any]]) -> str:
    values = {_string(row.get("image_identity"), "score row.image_identity") for row in rows}
    if len(values) != 1:
        _fail("one owner/context group mixes image identities")
    return next(iter(values))


def _reconstruct_attestation(
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    rules: core.LandscapeRules,
    coordinate_start: int,
) -> core.ConditionalY1CompletenessAttestation:
    gt_box_value = metadata.get("gt_box")
    if isinstance(gt_box_value, Mapping):
        gt_box_raw = [gt_box_value.get(name) for name in COORDINATE_NAMES]
    else:
        gt_box_raw = list(_sequence(gt_box_value, "conditional-y1 gt_box"))
    if len(gt_box_raw) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in gt_box_raw):
        _fail("conditional-y1 gt_box must contain four integer coordinate bins")
    gt_box = core.CoordinateBox.from_values(
        *(
            _integer(value, f"conditional-y1 gt_box.{name}")
            for name, value in zip(COORDINATE_NAMES, gt_box_raw)
        )
    )
    scores_by_x1: dict[core.CoordinateBin, tuple[core.ConditionalY1ScoreReceipt, ...]] = {}
    dense_rows = [row for row in rows if row.get("request_kind") == "dense_scan" and row.get("scan_slot") == "y1"]
    for row in dense_rows:
        fixed = _sequence(row.get("fixed_coord_token_ids"), "y1 dense row.fixed_coord_token_ids")
        if len(fixed) != 1:
            _fail("y1 dense row must fix exactly one x1 coordinate token")
        x1 = core.CoordinateBin(int(fixed[0]) - coordinate_start)
        raw = _mapping(row.get("raw_bin_scan"), "y1 dense row.raw_bin_scan")
        bins = _sequence(raw.get("bin_logprobs"), "y1 dense raw_bin_scan.bin_logprobs")
        expected_entries = core.enumerate_complete_conditional_y1(x1, gt_box, rules)
        if len(bins) != len(expected_entries):
            _fail("y1 dense raw row does not cover the complete declared coordinate vocabulary")
        receipts = tuple(
            core.ConditionalY1ScoreReceipt(
                x1=entry.x1,
                y1=entry.y1,
                raw_selected_token_logprob=_finite(value, "conditional y1 raw logprob"),
                can_form_valid_box=entry.can_form_valid_box,
                invalid_box_reason=entry.invalid_box_reason,
            )
            for entry, value in zip(expected_entries, bins)
        )
        if x1 in scores_by_x1:
            _fail(f"duplicate complete conditional-y1 row for x1={x1.value}")
        scores_by_x1[x1] = receipts
    reconstructed = core.attest_complete_conditional_y1_scores(
        diagnostic_owner_id=_string(metadata.get("diagnostic_owner_id"), "attestation owner"),
        gt_owner_id=_string(metadata.get("gt_owner_id"), "attestation GT owner"),
        image_identity=_string(metadata.get("image_identity"), "attestation image identity"),
        context_id=_string(metadata.get("context_id"), "attestation context"),
        canonical_description_text=_string(
            metadata.get("canonical_description_text"), "attestation canonical description"
        ),
        canonical_description_token_digest=_sha256(
            metadata.get("canonical_description_token_digest"), "attestation description token digest"
        ),
        context_token_digest=_sha256(metadata.get("context_token_digest"), "attestation context token digest"),
        tokenizer_identity=_string(metadata.get("tokenizer_identity"), "attestation tokenizer identity"),
        model_identity=_string(metadata.get("model_identity"), "attestation model identity"),
        runtime_identity=_string(metadata.get("runtime_identity"), "attestation runtime identity"),
        gt_box=gt_box,
        scores_by_x1=scores_by_x1,
        rules=rules,
    )
    declared = tuple(int(value) for value in _sequence(metadata.get("declared_x1_bins"), "declared x1 bins"))
    per_x1 = tuple(
        (int(item[0]), _sha256(item[1], "per-x1 conditional-y1 digest"))
        for item in _sequence(metadata.get("per_x1_receipt_digests"), "per-x1 receipt digests")
    )
    if reconstructed.declared_x1_bins != declared or reconstructed.per_x1_receipt_digests != per_x1:
        _fail("conditional-y1 per-x1 receipts do not independently reconstruct the sealed attestation")
    if reconstructed.completeness_digest != metadata.get("completeness_digest"):
        _fail("conditional-y1 completeness digest does not independently reconstruct from raw rows")
    return reconstructed


def _complete_candidate(
    row: Mapping[str, Any], rules: core.LandscapeRules, *, coordinate_start: int
) -> tuple[core.CompleteBoxCandidate, Mapping[str, Any]]:
    metadata = _mapping(row.get("core_candidate"), "complete score row.core_candidate")
    missing = sorted(REQUIRED_CORE_CANDIDATE_FIELDS.difference(metadata))
    if missing:
        _fail(
            "score row cannot reconstruct the pure-core CompleteBoxCandidate; missing fields: "
            f"{missing}"
        )
    bins = _sequence(metadata.get("coordinate_bins"), "core_candidate.coordinate_bins")
    if len(bins) != 4:
        _fail("core_candidate.coordinate_bins must contain x1,y1,x2,y2")
    box = core.CoordinateBox.from_values(*(int(value) for value in bins))
    candidate = core.CompleteBoxCandidate(
        candidate_id=_string(row.get("candidate_id"), "row candidate ID"),
        bank_name=_string(metadata.get("bank_name"), "core candidate bank"),
        source_id=_string(metadata.get("source_id"), "core candidate source"),
        box=box,
        extent_submode=_string(metadata.get("extent_submode"), "core candidate extent submode"),
        proposal_measure_id=_string(metadata.get("proposal_measure_id"), "core candidate proposal measure"),
        anchor=None,
        physical_owner_hint=(
            _string(metadata.get("identity_id"), "core candidate reviewed owner identity")
            if metadata.get("identity_kind") == "reviewed_physical_owner"
            else None
        ),
    )
    expected_tokens = _sequence(row.get("coord_token_ids"), "score row.coord_token_ids")
    if len(expected_tokens) != 4:
        _fail("complete score row.coord_token_ids must contain four tokens")
    token_bins = tuple(int(token) - coordinate_start for token in expected_tokens)
    if token_bins != box.as_tuple():
        _fail("complete score row coordinate tokens do not reconstruct core_candidate.coordinate_bins")
    if _sha256(row.get("coord_token_ids_sha256"), "score row coordinate-token digest") != sha256_json(
        list(expected_tokens)
    ):
        _fail("complete score row coordinate-token digest is stale")
    geometry = _mapping(metadata.get("geometry_identity"), "core_candidate.geometry_identity")
    observed = core.canonical_geometry_identity(
        box,
        image_width=_integer(
            geometry.get("image_width"), "core candidate geometry image width"
        ),
        image_height=_integer(
            geometry.get("image_height"), "core candidate geometry image height"
        ),
        rules=rules,
    )
    if geometry.get("schema") != observed.schema:
        _fail("core candidate geometry identity uses a non-canonical schema")
    if tuple(_sequence(geometry.get("pixel_box_xyxy"), "geometry pixel box")) != observed.pixel_box_xyxy:
        _fail("core candidate pixel geometry does not reproduce canonical round(value*extent/1000)")
    if geometry.get("identity_digest") != observed.identity_digest:
        _fail("core candidate canonical geometry digest is stale")
    return candidate, metadata


def _candidate_score(row: Mapping[str, Any], candidate: core.CompleteBoxCandidate) -> core.CandidateScore:
    raw = _mapping(row.get("raw_model_logprob"), "complete score row.raw_model_logprob")
    _validate_vocab_attestations(raw, rule_digest=_string(row.get("rule_digest"), "row rule digest"))
    factors = [_finite(raw.get(f"{name}_logprob"), f"raw {name} logprob") for name in COORDINATE_NAMES]
    declared_total = _finite(raw.get("complete_box_logprob_sum"), "raw complete-box total")
    if not math.isclose(sum(factors), declared_total, abs_tol=1e-10, rel_tol=1e-10):
        _fail("complete-box raw total does not equal the four selected-token log probabilities")
    return core.CandidateScore(candidate, core.RawCoordinateLogprobs(*factors))


def _cluster_by_registered_identity(
    scored: Sequence[tuple[core.CandidateScore, Mapping[str, Any]]], rules: core.LandscapeRules
) -> tuple[core.PhysicalBasinCluster, ...]:
    partitions: dict[tuple[str, str], list[core.CandidateScore]] = defaultdict(list)
    for score, metadata in scored:
        identity_kind = metadata.get("identity_kind")
        if identity_kind not in {"reviewed_physical_owner", "registered_geometry"}:
            _fail("core_candidate.identity_kind must be reviewed_physical_owner or registered_geometry")
        identity_id = _string(metadata.get("identity_id"), "core candidate identity_id")
        partitions[(str(identity_kind), identity_id)].append(score)
    output: list[core.PhysicalBasinCluster] = []
    index = 1
    for _, members in sorted(partitions.items()):
        for cluster in core.cluster_physical_basins(members, rules):
            basin_id = f"physical-basin-{index:03d}"
            submodes = tuple(
                replace(submode, submode_id=submode.submode_id.replace(cluster.basin_id, basin_id, 1))
                for submode in cluster.extent_submodes
            )
            output.append(replace(cluster, basin_id=basin_id, extent_submodes=submodes))
            index += 1
    return tuple(output)


def _registrations(
    clusters: Sequence[core.PhysicalBasinCluster],
    scored: Sequence[tuple[core.CandidateScore, Mapping[str, Any]]],
    attestation: core.ConditionalY1CompletenessAttestation,
    rules: core.LandscapeRules,
) -> tuple[core.BasinRegistration, ...]:
    metadata_by_id = {score.candidate.candidate_id: metadata for score, metadata in scored}
    result: list[core.BasinRegistration] = []
    for cluster in clusters:
        members = [metadata_by_id[candidate_id] for candidate_id in cluster.candidate_ids]
        signatures = {
            (member.get("role_id"), member.get("identity_kind"), member.get("identity_id"))
            for member in members
        }
        if len(signatures) != 1:
            _fail(f"physical basin {cluster.basin_id} mixes registered role or identity bindings")
        role_id, identity_kind, identity_id = next(iter(signatures))
        role = rules.basin_role(_string(role_id, "core candidate role_id"))
        if role.identity_kind != identity_kind:
            _fail(f"physical basin {cluster.basin_id} identity kind conflicts with its frozen role")
        result.append(
            core.BasinRegistration(
                basin_id=cluster.basin_id,
                role_id=role.role_id,
                identity_kind=role.identity_kind,
                reviewed_physical_owner_id=(str(identity_id) if identity_kind == "reviewed_physical_owner" else None),
                registered_geometry_id=(str(identity_id) if identity_kind == "registered_geometry" else None),
                context_id=attestation.context_id,
                foil_set_id=role.foil_set_id,
                rule_digest=_core_rule_digest(rules),
                conditional_y1_completeness_digest=attestation.completeness_digest,
            )
        )
    return tuple(result)


def _mass_comparability(measurements: Sequence[core.BasinMeasurement]) -> dict[str, bool]:
    signatures: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for measurement in measurements:
        signatures[
            (
                measurement.proposal_measure_id,
                measurement.proposal_domain_candidate_ids,
                measurement.proposal_domain_log_weight_normalizer,
            )
        ].append(measurement.basin_id)
    return {
        basin_id: len(basin_ids) >= 2
        for basin_ids in signatures.values()
        for basin_id in basin_ids
    }


def _measurement_json(measurement: core.BasinMeasurement, *, mass_comparable: bool) -> dict[str, Any]:
    value = core.json_serializable_receipt(measurement)
    assert isinstance(value, dict)
    normalized_mass = {
        "status": "comparable" if mass_comparable else "not_comparable",
        "proposal_measure_id": measurement.proposal_measure_id,
        "proposal_comparability_group": measurement.proposal_comparability_group,
        "normalized_basin_log_mass": measurement.normalized_basin_log_mass if mass_comparable else None,
        "normalized_basin_mass": measurement.normalized_basin_mass if mass_comparable else None,
    }
    value.pop("normalized_basin_log_mass")
    value.pop("normalized_basin_mass")
    value["basin_mass"] = normalized_mass
    value["evidence_polarity"] = "neutral" if measurement.identity_kind == "registered_geometry" else "positive_or_comparator"
    return value


def _summarize_group(
    *,
    key: tuple[str, str, str, str, float, str],
    rows: Sequence[Mapping[str, Any]],
    attestation_metadata: Mapping[str, Any],
    rules: core.LandscapeRules,
    coordinate_start: int,
) -> dict[str, Any]:
    owner_id, image_id, context_id, surface, penalty, foil_set_id = key
    neutral_reasons = sorted({reason for row in rows if (reason := _neutral_reason(row)) is not None})
    raw_row_ids = sorted(_string(row.get("candidate_id"), "row candidate ID") for row in rows)
    gt_owner_ids = {row.get("gt_owner_id") for row in rows}
    if len(gt_owner_ids) != 1 or next(iter(gt_owner_ids)) is None:
        _fail(f"owner/context group {key} mixes or omits GT-owner foreign keys")
    base = {
        "diagnostic_owner_id": owner_id,
        "gt_owner_id": next(iter(gt_owner_ids)),
        "image_id": image_id,
        "image_identity": _image_identity_for_group(rows),
        "context_id": context_id,
        "landscape_surface": surface,
        "native_repetition_penalty_stratum": penalty,
        "foil_set_id": foil_set_id,
        "raw_row_ids": raw_row_ids,
        "raw_rows_sha256": sha256_json([dict(row) for row in sorted(rows, key=lambda item: str(item.get("candidate_id")))]),
    }
    if neutral_reasons:
        return {
            **base,
            "decision_status": "neutral_raw_only",
            "neutral_reasons": neutral_reasons,
            "basins": [],
            "peak_prominence": [],
            "scientific_conclusion": None,
        }
    attestation = _reconstruct_attestation(
        metadata=attestation_metadata,
        rows=rows,
        rules=rules,
        coordinate_start=coordinate_start,
    )
    scored_with_metadata = [
        (_candidate_score(row, candidate), metadata)
        for row in rows
        if row.get("request_kind") == "complete_box"
        for candidate, metadata in [_complete_candidate(row, rules, coordinate_start=coordinate_start)]
    ]
    if not scored_with_metadata:
        _fail(f"owner/context group {key} has no complete-box score rows")
    clusters = _cluster_by_registered_identity(scored_with_metadata, rules)
    registrations = _registrations(clusters, scored_with_metadata, attestation, rules)
    scores = tuple(score for score, _ in scored_with_metadata)
    measurements = core.compute_basin_measurements(scores, clusters, registrations, attestation, rules)
    mass_admission = _mass_comparability(measurements)
    targets = [measurement for measurement in measurements if measurement.role_kind == "target"]
    foils = [measurement for measurement in measurements if measurement.role_kind == "foil"]
    prominence = [
        core.json_serializable_receipt(core.compute_peak_prominence(target, foil, attestation, rules))
        for target in targets
        for foil in foils
    ]
    return {
        **base,
        "decision_status": "measured_no_conclusion",
        "neutral_reasons": [],
        "conditional_y1_completeness_digest": attestation.completeness_digest,
        "basins": [
            _measurement_json(measurement, mass_comparable=mass_admission[measurement.basin_id])
            for measurement in measurements
        ],
        "peak_prominence": prominence,
        "scientific_conclusion": None,
    }


def summarize(
    *, scores_path: Path, score_receipt_path: Path, rules_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    rule_document = _read_json(rules_path)
    try:
        rules = core.validate_rule_mapping(rule_document)
    except ValueError as exc:
        _fail(f"landscape-decision-rules.json is incompatible with pure-core v2: {exc}")
    if rules.contract_mode == "production" and "semantic_core" not in rule_document:
        _fail("production decision rules must carry the landed semantic_core digest")
    coordinate_start, _, foil_digests = _scoring_contract(rule_document, rules)
    rows = _read_jsonl(scores_path)
    receipt = _read_json(score_receipt_path)
    source_digests, scoring_backend_admission = _validate_score_receipt(
        receipt, scores_path=scores_path, rules_path=rules_path, rules=rules, row_count=len(rows)
    )
    _validate_phase_a_freeze_admission(
        receipt,
        rule_document=rule_document,
        rules=rules,
    )
    if not rows:
        _fail("landscape-scores.jsonl must not be empty")
    seen_ids: set[str] = set()
    groups: dict[tuple[str, str, str, str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _validate_common_row(row, rules, foil_digests)
        candidate_id = _string(row.get("candidate_id"), "score row candidate ID")
        if candidate_id in seen_ids:
            _fail(f"duplicate score-row candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        groups[_row_key(row)].append(row)
    surface_receipts = _mapping(receipt.get("landscape_surface_receipts"), "score receipt.landscape_surface_receipts")
    if set(surface_receipts) != set(LANDSCAPE_SURFACES):
        _fail(f"score receipt must bind exact free/restricted surface receipts: {LANDSCAPE_SURFACES}")
    for surface in LANDSCAPE_SURFACES:
        entry = _mapping(surface_receipts[surface], f"surface receipt {surface}")
        surface_rows = sorted(
            (row for row in rows if row.get("landscape_surface") == surface),
            key=lambda row: str(row.get("candidate_id")),
        )
        if entry.get("row_count") != len(surface_rows) or entry.get("score_rows_sha256") != sha256_json(
            [dict(row) for row in surface_rows]
        ):
            _fail(f"{surface} receipt cardinality/digest does not reconstruct from raw rows")
        expected_status = (
            _validate_free_surface_execution(
                entry,
                surface_rows,
                production=rules.contract_mode == "production",
            )
            if surface == "canonical_description_free"
            else "passed"
            if surface_rows
            else "not_scored_test_fixture"
        )
        if entry.get("status") != expected_status:
            _fail(f"{surface} receipt status does not match its raw-row presence")
        if (
            rules.contract_mode == "production"
            and surface == "restricted_gt_target"
            and not surface_rows
        ):
            _fail("production aggregation requires a passed, non-empty restricted surface")
    policy_keys: dict[tuple[str, str, str, str], set[float]] = defaultdict(set)
    for owner, image, context, surface, penalty, _ in groups:
        policy_keys[(owner, image, context, surface)].add(penalty)
    mixed = {key: sorted(values) for key, values in policy_keys.items() if len(values) > 1}
    if mixed:
        _fail(f"refusing to aggregate owner/context rows across decode-policy strata: {mixed}")
    attestations = _full_attestation_by_key(receipt, rules)
    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        owner, _, context, surface, _, _ = key
        image_identity = _image_identity_for_group(group_rows)
        attestation_metadata = attestations.get((owner, image_identity, context, surface))
        if attestation_metadata is None and not any(_neutral_reason(row) for row in group_rows):
            _fail(
                "missing fully identity-bound conditional-y1 attestation for "
                f"{(owner, image_identity, context, surface)}"
            )
        summaries.append(
            _summarize_group(
                key=key,
                rows=group_rows,
                attestation_metadata=attestation_metadata or {},
                rules=rules,
                coordinate_start=coordinate_start,
            )
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "contract_mode": rules.contract_mode,
        "rule_digest": _core_rule_digest(rules),
        "rules_file_sha256": sha256_file(rules_path),
        "score_artifact_sha256": sha256_file(scores_path),
        "score_receipt_sha256": sha256_file(score_receipt_path),
        "unknown_unresolved_semantics": "explicit_neutral_raw_only_never_negative",
        "background_geometry_semantics": "registered_owner_neutral_foil_never_physical_owner_negative",
        "scoring_backend_admission": scoring_backend_admission,
        "per_owner_context": summaries,
        "scientific_conclusion": None,
    }
    reconstruction_digest = sha256_json(summary)
    scoring_backend_gate = (
        "cache_parity_passed"
        if scoring_backend_admission["parity_status"] == "passed"
        else "cache_parity_failed_uncached_reference_used"
    )
    output_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_mode": rules.contract_mode,
        "rule_digest": _core_rule_digest(rules),
        "rules_file_sha256": sha256_file(rules_path),
        "score_artifact_sha256": sha256_file(scores_path),
        "score_receipt_sha256": sha256_file(score_receipt_path),
        "validated_source_digests": source_digests,
        "input_row_count": len(rows),
        "owner_context_count": len(summaries),
        "neutral_owner_context_count": sum(item["decision_status"] == "neutral_raw_only" for item in summaries),
        "independent_reconstruction": "passed",
        "mandatory_cache_parity_gate": scoring_backend_admission["parity_status"],
        "scoring_backend_gate": scoring_backend_gate,
        "scoring_backend_admission": scoring_backend_admission,
        "summary_payload_sha256": reconstruction_digest,
        "scientific_conclusion": None,
    }
    return summary, output_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-receipt", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, receipt = summarize(
        scores_path=args.scores.resolve(strict=True),
        score_receipt_path=args.score_receipt.resolve(strict=True),
        rules_path=args.decision_rules.resolve(strict=True),
    )
    output_dir = args.output_dir.expanduser().resolve()
    _write_once(output_dir / SUMMARY_NAME, summary)
    _write_once(output_dir / RECEIPT_NAME, receipt)
    print(json.dumps({"owner_contexts": receipt["owner_context_count"], "neutral": receipt["neutral_owner_context_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
