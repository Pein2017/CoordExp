from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.analysis.sampled_rescue_transition.artifacts import (
    CallRecord,
    _exact_maximum_flow_assignment,
    cluster_geometry_modes,
    load_case_table,
    load_call_records,
)
from src.analysis.sampled_rescue_transition.comparison import (
    compare_greedy_to_samples,
    common_prefix_length,
    trajectory_rows,
)
from scripts.research.run_sampled_rescue_transition import (
    _canonical_complete_row_factors,
    _compose_complete_row_from_receipts,
    _donor_boundary_grammar,
    _first_free_token_evidence,
    _first_action,
    _forced_context_evidence,
    _reconstruct_intervened_row,
    _request_id,
    _verify_donor_runtime_identity,
    _validate_forced_span_grammar,
    _write_bundle_once,
    build_parser,
    derive_sampling_seed,
)
from src.inference.parsing import parse_compact_object_box_closed


def _bundle(path: Path, *, request_id: str, seed: int, boxes: list[tuple[str, list[float]]], tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parse = [
        {
            "normalized_category_name": category,
            "parsed_bbox_xyxy": box,
            "generated_row_index": i,
            "score": 0.5,
        }
        for i, (category, box) in enumerate(boxes)
    ]
    payload = {
        "request_id": request_id,
        "stop_reason": "im_end",
        "execution_evidence": {"image_id": "1", "arm": {"arm_code": "FULL_BAG_K"}},
        "decode_result": {
            "prompt_token_ids": [11, 12],
            "generated_token_ids": tokens,
            "raw_generated_text": "",
            "execution_receipt": {
                "sampling_seed": seed,
                "prompt_token_identifiers_hash": "prompt-hash",
                "decode_generation_policy": {
                    "temperature": 0.4,
                    "repetition_penalty": 1.0,
                },
            },
        },
        "parse_score_receipts": parse,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_common_prefix_and_rows() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert trajectory_rows([151646, 10, 151648, 20, 21, 22, 23, 151649]) == ((0, 8),)


def test_loader_clusters_without_category_binding(tmp_path: Path) -> None:
    root = tmp_path / "calls"
    root.mkdir()
    _bundle(root / "a" / "terminal-output-bundle.json", request_id="a", seed=1, boxes=[("fork", [0, 0, 10, 10])], tokens=[151646, 1, 151648, 1, 2, 3, 4, 151649])
    _bundle(root / "b" / "terminal-output-bundle.json", request_id="b", seed=2, boxes=[("spoon", [1, 1, 9, 9])], tokens=[151646, 1, 151648, 1, 2, 3, 4, 151649])
    calls = load_call_records(root)
    assert len(calls) == 2
    modes = cluster_geometry_modes(calls)
    assert len(modes) == 1
    assert modes[0].hit_count == 2
    assert modes[0].prediction_count == 2
    assert modes[0].category_counts == {"fork": 1, "spoon": 1}


def test_compare_reports_legal_prompt_and_rescue(tmp_path: Path) -> None:
    root = tmp_path / "calls"
    root.mkdir()
    _bundle(root / "g" / "terminal-output-bundle.json", request_id="g", seed=0, boxes=[("person", [0, 0, 10, 10])], tokens=[151646, 1, 151648, 1, 2, 3, 4, 151649])
    _bundle(root / "s" / "terminal-output-bundle.json", request_id="s", seed=1, boxes=[("person", [0, 0, 10, 10]), ("book", [30, 30, 40, 40])], tokens=[151646, 1, 151648, 1, 2, 3, 4, 151649, 151646, 2, 151648, 5, 6, 7, 8, 151649])
    calls = load_call_records(root)
    result = compare_greedy_to_samples(calls[0], (calls[1],))
    assert len(result) == 1
    assert result[0].common_prompt
    assert result[0].common_generated_prefix_tokens == 8
    assert result[0].sampled_row_count == 2
    assert result[0].sampled_rescue_object_ids == ()
    assert result[0].sampled_geometry_only_mode_ids


def test_case_table_preserves_unmatched_ledger_state(tmp_path: Path) -> None:
    root = tmp_path / "calls"
    _bundle(root / "a" / "terminal-output-bundle.json", request_id="a", seed=1, boxes=[("fork", [0, 0, 10, 10])], tokens=[151646, 1, 151648, 1, 2, 3, 4, 151649])
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"image_id": 1, "object_identifier": "audit:1", "final_state": "accepted", "source_canvas_box_xyxy": [50, 50, 60, 60]}) + "\n", encoding="utf-8")
    table = load_case_table(root, image_ids=["1"], audit_ledger_path=ledger)
    assert table["cases"][0]["ledger_matches"][0]["match_status"] == "unmatched_or_uncertain"


def test_first_action_does_not_hide_malformed_first_span() -> None:
    parsed = parse_compact_object_box_closed(
        "<|object_ref_start|><|object_ref_end|><|box_start|>"
        "<|coord_1|><|coord_1|><|coord_2|><|coord_2|><|box_end|>"
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_1|><|coord_1|><|coord_2|><|coord_2|><|box_end|>",
        row_id="row",
        row_index=0,
        image_width=100,
        image_height=100,
    )
    first = _first_action(
        parsed,
        "",
        generated_token_ids=[151646, 1, 151648, 1, 2, 3, 4, 151649, 151646],
        stop_reason="im_end",
    )
    assert first["status"] == "malformed_row"
    assert first["generated_order"] == 0
    assert first["token_start"] == 0
    assert first["token_end"] == 8
    assert first["token_ids"]


def test_first_action_reports_immediate_terminal_token() -> None:
    parsed = parse_compact_object_box_closed(
        "",
        row_id="row",
        row_index=0,
        image_width=100,
        image_height=100,
    )
    first = _first_action(
        parsed,
        "",
        generated_token_ids=[999],
        stop_reason="im_end",
    )
    assert first["status"] == "immediate_terminal"
    assert first["token_start"] == 0
    assert first["token_end"] == 1
    assert first["token_ids"] == [999]


def test_donor_boundary_grammar_and_request_identity(tmp_path: Path) -> None:
    complete = _donor_boundary_grammar(
        [151646, 1, 151648, 1, 2, 3, 4, 151649],
        role="common_pre_row",
    )
    assert complete["boundary_grammar_verified"] is True
    phase = _donor_boundary_grammar([151646, 1, 151648, 1], role="phase_divergence")
    assert phase["boundary_grammar_verified"] is True
    assert _request_id(
        image_id="1",
        boundary_role="common_pre_row",
        prompt_token_hash="a" * 64,
        kind="sample",
        index=0,
        seed=1,
    ) != _request_id(
        image_id="1",
        boundary_role="common_pre_row",
        prompt_token_hash="a" * 64,
        kind="sample",
        index=1,
        seed=2,
    )
    assert _request_id(
        image_id="1",
        boundary_role="common_pre_row",
        prompt_token_hash="a" * 64,
        kind="sample",
        index=0,
        seed=1,
        sampling_temperature=0.2,
    ) != _request_id(
        image_id="1",
        boundary_role="common_pre_row",
        prompt_token_hash="a" * 64,
        kind="sample",
        index=0,
        seed=1,
        sampling_temperature=0.6,
    )
    path = tmp_path / "bundle.json"
    _write_bundle_once(path, {"value": 1})
    try:
        _write_bundle_once(path, {"value": 2})
    except SystemExit as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("existing bundle was overwritten")


def test_sampling_temperature_parser_default_and_validation() -> None:
    assert build_parser().parse_args([]).sampling_temperature == 0.4
    for invalid in ("0", "-0.2", "nan", "inf"):
        try:
            build_parser().parse_args(["--sampling-temperature", invalid])
        except SystemExit:
            continue
        raise AssertionError(f"invalid sampling temperature was accepted: {invalid}")


def test_retired_spatial_framework_helpers_preserve_frozen_semantics() -> None:
    assert derive_sampling_seed(
        root_seed=2026071301,
        role="baseline",
        image_id=7818,
        cell_or_call_label="call-0",
    ) == 65610539429397169
    assert _exact_maximum_flow_assignment(
        prediction_count=2,
        reference_count=2,
        candidate_rows=((0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.85)),
        benefits=(90, 80, 85),
    ) == frozenset({1, 2})


def test_leading_unmatched_drop_is_first_action() -> None:
    parsed = parse_compact_object_box_closed(
        "noise before object<|object_ref_start|>person<|object_ref_end|>"
        "<|box_start|><|coord_1|><|coord_1|><|coord_2|><|coord_2|><|box_end|>",
        row_id="row",
        row_index=0,
        image_width=100,
        image_height=100,
    )
    first = _first_action(
        parsed,
        "",
        generated_token_ids=[151646, 1, 151648, 1, 2, 3, 4, 151649],
        stop_reason="im_end",
    )
    assert first["status"] == "leading_unmatched"
    assert first["drop_reason"] == "unmatched_text"


def test_donor_runtime_identity_allows_relocation_but_rejects_payload_change() -> None:
    def identity(root: str, *, delta_sha: str = "c" * 64) -> dict[str, object]:
        return {
            "base_model": {"model_type": "qwen3_vl", "revision": "r1"},
            "adapter": {
                "adapter_path": f"{root}/adapter",
                "adapter_payload_evidence": {
                    "config_path": f"{root}/config.json",
                    "config_sha256": "a" * 64,
                    "tensor_path": f"{root}/adapter.safetensors",
                    "tensor_sha256": "b" * 64,
                },
            },
            "embedding_delta": {
                "identity": {
                    "delta_path": f"{root}/delta.safetensors",
                    "delta_sha256": delta_sha,
                    "metadata_path": f"{root}/metadata.json",
                    "metadata_sha256": "d" * 64,
                },
                "load": {
                    "metadata_path": f"{root}/metadata.json",
                    "metadata_sha256": "d" * 64,
                    "tensor_path": f"{root}/delta.safetensors",
                    "tensor_sha256": delta_sha,
                },
            },
        }

    donor = {
        "execution_evidence": {"image_id": "1"},
        "decode_result": {
            "model_identity": identity("/donor"),
            "tokenizer_identity": {"tokenizer": "same"},
            "generation_config_fingerprint": "gen",
        },
    }
    _verify_donor_runtime_identity(
        donor=donor,
        active_image_id="1",
        active_model_identity=identity("/active"),
        active_tokenizer_identity={"tokenizer": "same"},
        active_generation_config_fingerprint="gen",
    )
    donor["execution_evidence"]["image_id"] = "2"
    try:
        _verify_donor_runtime_identity(
            donor=donor,
            active_image_id="1",
            active_model_identity=identity("/active"),
            active_tokenizer_identity={"tokenizer": "same"},
            active_generation_config_fingerprint="gen",
        )
    except SystemExit as exc:
        assert "image_id" in str(exc)
    else:
        raise AssertionError("donor image mismatch was accepted")
    donor["execution_evidence"]["image_id"] = "1"
    donor["decode_result"]["model_identity"] = identity("/donor", delta_sha="f" * 64)
    try:
        _verify_donor_runtime_identity(
            donor=donor,
            active_image_id="1",
            active_model_identity=identity("/active"),
            active_tokenizer_identity={"tokenizer": "same"},
            active_generation_config_fingerprint="gen",
        )
    except SystemExit as exc:
        assert "model_identity" in str(exc)
    else:
        raise AssertionError("changed payload identity was accepted")
    donor["decode_result"]["model_identity"] = identity("/donor")
    donor["decode_result"].pop("generation_config_fingerprint")
    try:
        _verify_donor_runtime_identity(
            donor=donor,
            active_image_id="1",
            active_model_identity=identity("/active"),
            active_tokenizer_identity={"tokenizer": "same"},
            active_generation_config_fingerprint="gen",
        )
    except SystemExit as exc:
        assert "generation_config_fingerprint_missing" in str(exc)
    else:
        raise AssertionError("missing generation fingerprint was accepted")


def test_first_free_token_evidence_requires_matching_executed_float32_trace() -> None:
    class Result:
        def __init__(self, trace):
            self.token_trace = trace

    good = Result(
        [{"step_index": 0, "token_id": 7, "is_pad": False, "logprob": -0.25}]
    )
    evidence = _first_free_token_evidence(good, [7])
    assert evidence["token_id"] == 7
    assert isinstance(evidence["log_probability_float32"], float)
    for bad_trace in (
        None,
        [],
        [{"step_index": 1, "token_id": 7, "is_pad": False, "logprob": -0.25}],
        [{"step_index": 0, "token_id": 8, "is_pad": False, "logprob": -0.25}],
        [{"step_index": 0, "token_id": 7, "is_pad": True, "logprob": None}],
        [{"step_index": 0, "token_id": 7, "is_pad": False, "logprob": float("nan")}],
    ):
        try:
            _first_free_token_evidence(Result(bad_trace), [7])
        except SystemExit:
            continue
        raise AssertionError(f"invalid token trace was accepted: {bad_trace!r}")
    assert _first_free_token_evidence(Result([]), []) == {
        "token_id": None,
        "log_probability_float32": None,
    }


def test_forced_context_exact_and_syntax_only_policies() -> None:
    accepted = _forced_context_evidence(
        recipient_prompt_ids=[1],
        recipient_prefix_ids=[2],
        donor_prompt_ids=[1],
        donor_prefix_ids=[2],
        policy="exact",
    )
    assert accepted["exact_context"] is True
    mismatch = _forced_context_evidence(
        recipient_prompt_ids=[1],
        recipient_prefix_ids=[2],
        donor_prompt_ids=[9],
        donor_prefix_ids=[8],
        policy="syntax_only",
    )
    assert mismatch["context_mismatch"] is True
    try:
        _forced_context_evidence(
            recipient_prompt_ids=[1],
            recipient_prefix_ids=[2],
            donor_prompt_ids=[9],
            donor_prefix_ids=[8],
            policy="exact",
        )
    except SystemExit as exc:
        assert "context mismatch" in str(exc)
    else:
        raise AssertionError("exact context mismatch was silently downgraded")


def test_forced_span_grammar_and_partial_row_reconstruction() -> None:
    tokens = [151646, 42, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    cumulative = {
        "row_opener": (0, 1),
        "first_description_token": (0, 2),
        "complete_description": (0, 3),
        "first_coordinate": (0, 5),
        "complete_row": (0, 9),
    }
    for role, (start, end) in cumulative.items():
        assert _validate_forced_span_grammar(
            tokens, start=start, end=end, role=role
        )["verified"] is True
    assert _validate_forced_span_grammar(
        [151646, 42, 43, 151647], start=0, end=4, role="complete_description"
    )["verified"] is False
    assert _validate_forced_span_grammar(
        [151646, 42, 151647, 151648, 42], start=0, end=5, role="first_coordinate"
    )["verified"] is False
    assert _validate_forced_span_grammar(
        [151646, 42, 151647, 151648, 151670, 151671], start=0, end=6, role="first_coordinate"
    )["verified"] is False
    assert _validate_forced_span_grammar(
        [151646, 42, 151647, 151648, 151670, 151671, 151672, 151649],
        start=0,
        end=8,
        role="complete_row",
    )["verified"] is False
    assert _validate_forced_span_grammar(
        tokens, start=0, end=9, role="complete_row"
    )["verified"] is True
    assert _validate_forced_span_grammar(
        tokens, start=0, end=5, role="first_coordinate"
    )["verified"] is True

    class Tokenizer:
        def decode(self, values, **kwargs):
            return (
                "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
                "<|coord_7|><|coord_8|><|coord_9|><|coord_10|><|box_end|>"
            )

    result = _reconstruct_intervened_row(
        Tokenizer(),
        tokens,
        intervention_token_count=5,
        image_width=100,
        image_height=100,
        row_id="row",
    )
    assert result["status"] == "complete"
    assert result["parse"]["valid_prediction_count"] == 1


def test_compose_complete_row_crosses_description_and_geometry_receipts(
    tmp_path: Path,
) -> None:
    def write_receipt(path: Path, tokens: list[int], condition: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "condition_name": condition,
                    "first_rows": [
                        {
                            "reconstructed_intervened_row": {
                                "status": "complete",
                                "token_ids": tokens,
                                "text": condition,
                            }
                        }
                    ],
                    "call_bundles": [
                        {
                            "execution_evidence": {"image_id": "1"},
                            "decode_result": {
                                "model_identity": {},
                                "tokenizer_identity": {},
                                "generation_config_fingerprint": "gen",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    description = tmp_path / "description.json"
    geometry = tmp_path / "geometry.json"
    write_receipt(
        description,
        [151646, 42, 151647, 151648, 151670, 151671, 151672, 151673, 151649],
        "description-source",
    )
    write_receipt(
        geometry,
        [151646, 43, 151647, 151648, 151674, 151675, 151676, 151677, 151649],
        "geometry-source",
    )
    composed = _compose_complete_row_from_receipts(description, geometry)
    assert composed["token_ids"] == [
        151646,
        42,
        151647,
        151648,
        151674,
        151675,
        151676,
        151677,
        151649,
    ]
    assert composed["evidence"]["description_source_condition"] == "description-source"
    assert composed["evidence"]["geometry_source_condition"] == "geometry-source"


def test_canonical_factor_row_rejects_noncanonical_lengths_and_coordinates() -> None:
    valid = [151646, 42, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    assert _canonical_complete_row_factors(valid) == {
        "description_token_id": 42,
        "coordinate_token_ids": [151670, 151671, 151672, 151673],
    }
    for invalid in (
        valid[:-1],
        [151646, 42, 43, 151647, 151648, 151670, 151671, 151672, 151673, 151649],
        [151646, 42, 151647, 151648, 151670, 7, 151672, 151673, 151649],
    ):
        try:
            _canonical_complete_row_factors(invalid)
        except SystemExit:
            continue
        raise AssertionError(f"invalid factor row was accepted: {invalid!r}")


def test_composed_row_parser_surface() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "composed-row-smoke",
            "--description-row-receipt",
            "description.json",
            "--geometry-row-receipt",
            "geometry.json",
        ]
    )
    assert args.mode == "composed-row-smoke"
    assert args.description_row_receipt == Path("description.json")
    assert args.geometry_row_receipt == Path("geometry.json")
