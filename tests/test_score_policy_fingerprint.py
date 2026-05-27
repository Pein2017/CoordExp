from __future__ import annotations

import unittest

from src.infer.artifacts import build_score_policy_fingerprint


def test_score_policy_fingerprint_is_stable_and_order_independent() -> None:
    first = build_score_policy_fingerprint(
        policy_name="logprob_sequence_mean",
        score_source="generated_token_logprobs",
        aggregation_rule="mean",
        token_span_rule="generated_sequence",
        constant_score_value=None,
        source_raw_artifact_identity={
            "path": "gt_vs_pred.jsonl",
            "sha256": "abc",
        },
        parser_policy="strict",
        metric_bearing=True,
    )
    second = build_score_policy_fingerprint(
        policy_name="logprob_sequence_mean",
        score_source="generated_token_logprobs",
        aggregation_rule="mean",
        token_span_rule="generated_sequence",
        constant_score_value=None,
        source_raw_artifact_identity={
            "sha256": "abc",
            "path": "gt_vs_pred.jsonl",
        },
        parser_policy="strict",
        metric_bearing=True,
    )

    assert first == second
    assert first.startswith("score_policy:")


def test_score_policy_fingerprint_changes_with_metric_bearing_status() -> None:
    metric = build_score_policy_fingerprint(
        policy_name="constant",
        score_source="constant",
        aggregation_rule="constant",
        token_span_rule="none",
        constant_score_value=1.0,
        source_raw_artifact_identity="raw:abc",
        parser_policy="strict",
        metric_bearing=True,
    )
    diagnostic = build_score_policy_fingerprint(
        policy_name="constant",
        score_source="constant",
        aggregation_rule="constant",
        token_span_rule="none",
        constant_score_value=1.0,
        source_raw_artifact_identity="raw:abc",
        parser_policy="strict",
        metric_bearing=False,
    )

    assert metric != diagnostic


def test_score_policy_fingerprint_uses_raw_hash_not_raw_path() -> None:
    first = build_score_policy_fingerprint(
        policy_name="constant",
        score_source="constant",
        aggregation_rule="constant",
        token_span_rule="none",
        constant_score_value=1.0,
        source_raw_artifact_identity={
            "path": "/machine-a/run/gt_vs_pred.jsonl",
            "sha256": "same-bytes",
        },
        parser_policy="strict",
        metric_bearing=True,
    )
    second = build_score_policy_fingerprint(
        policy_name="constant",
        score_source="constant",
        aggregation_rule="constant",
        token_span_rule="none",
        constant_score_value=1.0,
        source_raw_artifact_identity={
            "path": "/machine-b/copied/gt_vs_pred.jsonl",
            "sha256": "same-bytes",
        },
        parser_policy="strict",
        metric_bearing=True,
    )
    changed = build_score_policy_fingerprint(
        policy_name="constant",
        score_source="constant",
        aggregation_rule="constant",
        token_span_rule="none",
        constant_score_value=1.0,
        source_raw_artifact_identity={
            "path": "/machine-b/copied/gt_vs_pred.jsonl",
            "sha256": "different-bytes",
        },
        parser_policy="strict",
        metric_bearing=True,
    )

    assert first == second
    assert first != changed


def test_score_policy_fingerprint_rejects_non_boolean_metric_status() -> None:
    with unittest.TestCase().assertRaisesRegex(
        TypeError,
        "metric_bearing must be a bool",
    ):
        build_score_policy_fingerprint(
            policy_name="constant",
            score_source="constant",
            aggregation_rule="constant",
            token_span_rule="none",
            constant_score_value=1.0,
            source_raw_artifact_identity="raw:abc",
            parser_policy="strict",
            metric_bearing="false",
        )
