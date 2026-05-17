from __future__ import annotations

import subprocess
import sys

import pytest

from src.training.stage2.rollout_codec import (
    Stage2RolloutDiagnosticSummary,
    Stage2RolloutParseResult,
    calculate_stage2_rollout_diagnostics,
    resolve_stage2_rollout_template_policy,
)


def test_compact_full_policy_is_explicit_and_unconstrained_by_default() -> None:
    policy = resolve_stage2_rollout_template_policy(
        "compact_full",
        custom_json_format="standard",
    )

    assert policy.template_family == "compact_full"
    assert policy.parser_id == "compact_full"
    assert policy.append_policy_id == "compact_full_fn_append"
    assert policy.decode_policy == "unconstrained"
    assert policy.invalid_rollout_policy == "fallback_gt_fn_append_only"
    assert policy.fallback_loss_weight == 1.0
    assert policy.diagnostics_metadata == {
        "resolved_rollout_template": "compact_full",
        "rollout_parser_id": "compact_full",
        "rollout_append_policy_id": "compact_full_fn_append",
        "rollout_decode_policy": "unconstrained",
        "invalid_rollout_policy": "fallback_gt_fn_append_only",
        "fallback_loss_weight": 1.0,
    }


def test_compact_full_policy_accepts_explicit_fallback_runtime_knobs() -> None:
    policy = resolve_stage2_rollout_template_policy(
        "compact_full",
        rollout_decode_policy="compact-grammar",
        invalid_rollout_policy="fallback_gt_fn_append_only",
        fallback_loss_weight=0.5,
    )

    assert policy.template_family == "compact_full"
    assert policy.decode_policy == "compact_grammar"
    assert policy.invalid_rollout_policy == "fallback_gt_fn_append_only"
    assert policy.fallback_loss_weight == pytest.approx(0.5)


@pytest.mark.parametrize("invalid_rollout_policy", ["abort", "dump_and_continue"])
def test_compact_full_policy_rejects_unimplemented_invalid_rollout_policies(
    invalid_rollout_policy: str,
) -> None:
    with pytest.raises(ValueError, match="invalid_rollout_policy for compact_full"):
        resolve_stage2_rollout_template_policy(
            "compact_full",
            invalid_rollout_policy=invalid_rollout_policy,
        )


def test_policy_resolver_does_not_infer_rollout_template_from_json_format() -> None:
    with pytest.raises(ValueError, match="rollout_template_family"):
        resolve_stage2_rollout_template_policy(custom_json_format="standard")


def test_coordjson_policy_is_explicit_legacy_surface() -> None:
    policy = resolve_stage2_rollout_template_policy("coordjson")

    assert policy.template_family == "coordjson"
    assert policy.parser_id == "coordjson_legacy"
    assert policy.append_policy_id == "coordjson_legacy_fn_append"
    assert policy.decode_policy == "legacy_coordjson"
    assert policy.invalid_rollout_policy == "abort"
    assert policy.fallback_loss_weight == 1.0


def test_coordjson_policy_rejects_compact_only_runtime_knobs() -> None:
    with pytest.raises(ValueError, match="rollout_decode_policy.*coordjson"):
        resolve_stage2_rollout_template_policy(
            "coordjson",
            rollout_decode_policy="unconstrained",
        )

    with pytest.raises(ValueError, match="fallback_gt_fn_append_only.*coordjson"):
        resolve_stage2_rollout_template_policy(
            "coordjson",
            invalid_rollout_policy="fallback_gt_fn_append_only",
        )


def test_rollout_diagnostics_report_fallback_and_mismatch_rates() -> None:
    results = (
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="malformed",
            valid_objects=(),
            invalid_rollout=True,
            empty_valid_object_set=False,
            truncated=False,
            fallback_reason="malformed_compact_full",
        ),
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="",
            valid_objects=(),
            invalid_rollout=False,
            empty_valid_object_set=True,
            truncated=False,
            fallback_reason="empty_valid_object_set",
        ),
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="truncated",
            valid_objects=(),
            invalid_rollout=True,
            empty_valid_object_set=False,
            truncated=True,
            fallback_reason="malformed_compact_full",
        ),
    )

    summary = calculate_stage2_rollout_diagnostics(
        results,
        parser_template_mismatch_count=1,
        fallback_dominance_threshold=0.4,
    )

    assert isinstance(summary, Stage2RolloutDiagnosticSummary)
    assert summary.invalid_fallback_gt_fn_count == 2
    assert summary.invalid_fallback_gt_fn_rate == pytest.approx(0.5)
    assert summary.empty_valid_object_rate == pytest.approx(0.25)
    assert summary.fallback_loss_share == pytest.approx(1.0)
    assert summary.fallback_dominance_warning is True
    assert summary.parse_truncated_rate == pytest.approx(0.25)
    assert summary.parser_template_mismatch_rate == pytest.approx(0.25)


def test_rollout_diagnostics_default_warns_above_approved_fallback_share() -> None:
    results = (
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="malformed",
            valid_objects=(),
            invalid_rollout=True,
            empty_valid_object_set=False,
            truncated=False,
            fallback_reason="malformed_compact_full",
        ),
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="",
            valid_objects=(),
            invalid_rollout=False,
            empty_valid_object_set=True,
            truncated=False,
            fallback_reason="empty_valid_object_set",
        ),
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="valid",
            valid_objects=(),
            invalid_rollout=False,
            empty_valid_object_set=False,
            truncated=False,
        ),
        Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id="compact_full",
            response_text="valid",
            valid_objects=(),
            invalid_rollout=False,
            empty_valid_object_set=False,
            truncated=False,
        ),
    )

    summary = calculate_stage2_rollout_diagnostics(results)

    assert summary.fallback_loss_share == pytest.approx(0.5)
    assert summary.fallback_dominance_warning is True


def test_rollout_diagnostics_default_threshold_boundary_cases() -> None:
    one_fallback = Stage2RolloutParseResult(
        template_family="compact_full",
        parser_id="compact_full",
        response_text="malformed",
        valid_objects=(),
        invalid_rollout=True,
        empty_valid_object_set=False,
        truncated=False,
        fallback_reason="malformed_compact_full",
    )
    valid = Stage2RolloutParseResult(
        template_family="compact_full",
        parser_id="compact_full",
        response_text="valid",
        valid_objects=(),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
    )

    assert (
        calculate_stage2_rollout_diagnostics((one_fallback, valid, valid))
        .fallback_dominance_warning
        is False
    )
    assert (
        calculate_stage2_rollout_diagnostics((one_fallback, valid))
        .fallback_dominance_warning
        is True
    )


def test_rollout_codec_import_does_not_load_training_framework_modules() -> None:
    probe = (
        "import sys\n"
        "import src.training.stage2.rollout_codec\n"
        "blocked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'torch' or name.startswith('torch.')\n"
        "    or name == 'transformers' or name.startswith('transformers.')\n"
        "    or name == 'swift' or name.startswith('swift.')\n"
        ")\n"
        "print(blocked)\n"
        "raise SystemExit(1 if blocked else 0)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
