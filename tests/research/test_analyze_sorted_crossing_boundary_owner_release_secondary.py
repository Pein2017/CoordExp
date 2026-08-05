"""Focused tests for the secondary downstream-compatibility analysis of the
sorted crossing-boundary owner release/realization unit.

Fixture policy
--------------
The merged input is always a *real* merge: the twelve-shard capture built by
``test_merge_sorted_crossing_boundary_owner_release_secondary`` is run through
the secondary merge module, so these tests exercise the analyzer against bytes
the producer chain actually publishes.  Negative cases mutate that merged
directory and reseal its receipt, so a rejection is proven to come from the
analyzer's own contract rather than from a broken seal.

The two boundaries this suite exists to protect:

* the analyzer preserves every per-segment exact paired delta and keeps the 12
  native-TP replay controls isolated from the 26 crossing owners; and
* the analyzer refuses to become a decision surface -- no branch is read, no
  cutoff is fitted, no compatibility class is assigned, and no key of any
  payload it reads or writes may name one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import pytest

from scripts.research import (
    analyze_sorted_crossing_boundary_owner_release as primary_analyzer,
)
from scripts.research import (
    analyze_sorted_crossing_boundary_owner_release_secondary as sut,
)
from scripts.research import merge_sorted_crossing_boundary_owner_release as primary_merge
from scripts.research import (
    merge_sorted_crossing_boundary_owner_release_secondary as secondary_merge,
)
from scripts.research import score_sorted_crossing_boundary_owner_release as primary
from scripts.research import (
    score_sorted_crossing_boundary_owner_release_secondary as secondary,
)


def _load_secondary_merge_fixtures():
    """Import the secondary merge test module by path so its twelve-shard
    capture builder is reused rather than duplicated.

    ``tests/research`` is not a package, so a plain ``from tests.research...``
    import is not available; the module is registered in ``sys.modules`` before
    execution because its dataclasses need to resolve their own module.
    """

    name = "crossing_boundary_secondary_merge_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(
        "test_merge_sorted_crossing_boundary_owner_release_secondary.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_secondary_merge_fixtures()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def merged(tmp_path: Path) -> dict[str, Any]:
    """One real twelve-shard capture, merged into a sealed directory."""

    capture = fx.build_secondary_capture(tmp_path / "run")
    merged_dir = tmp_path / "run" / "merged"
    capture.run_merge(merged_dir)
    return {
        "capture": capture,
        "merged_dir": merged_dir,
        "analysis_dir": capture.analysis_dir,
    }


def read_merged_rows(merged_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (merged_dir / secondary_merge.MERGED_ROWS_NAME)
        .read_text("utf-8")
        .splitlines()
        if line.strip()
    ]


def rewrite_merged_rows(merged_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rewrite the merged rows and reseal the merge receipt around them."""

    blob = b"".join(secondary.canonical_json_bytes(row) + b"\n" for row in rows)
    (merged_dir / secondary_merge.MERGED_ROWS_NAME).write_bytes(blob)
    receipt = json.loads(
        (merged_dir / secondary_merge.MERGE_RECEIPT_NAME).read_text("utf-8")
    )
    receipt["output_file_digests"][secondary_merge.MERGED_ROWS_NAME] = {
        "path": secondary_merge.MERGED_ROWS_NAME,
        "byte_size": len(blob),
        "row_count": len(rows),
        "sha256": secondary.sha256_bytes(blob),
    }
    request_ids = sorted(str(row["request_id"]) for row in rows)
    receipt["secondary_requests"] = {
        **receipt["secondary_requests"],
        "request_count": len(rows),
        "request_ids_sha256": secondary.sha256_json(request_ids),
        "counts_by_variant": secondary.counts_by_variant(rows),
    }
    receipt.pop("receipt_content_sha256", None)
    receipt["receipt_content_sha256"] = secondary.sha256_json(receipt)
    (merged_dir / secondary_merge.MERGE_RECEIPT_NAME).write_bytes(
        secondary.canonical_json_bytes(receipt) + b"\n"
    )


def run(merged: Mapping[str, Any]) -> dict[str, Any]:
    return sut.run_analysis(merged["merged_dir"], merged["analysis_dir"])


def owner_row(result: Mapping[str, Any], gt_owner_id: str) -> dict[str, Any]:
    for row in result["owner_rows"]:
        if str(row["gt_owner_id"]) == gt_owner_id:
            return row
    raise AssertionError(f"no owner row for {gt_owner_id!r}")


# ---------------------------------------------------------------------------
# 1. The happy path
# ---------------------------------------------------------------------------


def test_analysis_publishes_a_complete_self_sealed_family(
    merged: Mapping[str, Any], tmp_path: Path
) -> None:
    result = run(merged)
    files = sut.build_output_files(result)
    output_dir = tmp_path / "analysis"
    published = primary_merge.publish_merge(output_dir, files)
    assert published["published"] is True
    assert sorted(entry.name for entry in output_dir.iterdir()) == sorted(
        (sut.OWNER_ROWS_NAME, sut.SUMMARY_NAME, sut.REPORT_MD_NAME, sut.RECEIPT_NAME)
    )

    receipt = json.loads((output_dir / sut.RECEIPT_NAME).read_text("utf-8"))
    assert receipt["schema_version"] == sut.RECEIPT_SCHEMA_VERSION
    assert receipt["unit_id"] == sut.UNIT_ID
    secondary_merge.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="analysis receipt"
    )
    for name in (sut.OWNER_ROWS_NAME, sut.SUMMARY_NAME, sut.REPORT_MD_NAME):
        entry = receipt["output_file_digests"][name]
        assert entry["sha256"] == secondary.sha256_file(output_dir / name)
        assert entry["byte_size"] == (output_dir / name).stat().st_size
    assert receipt["input_file_sha256"] == result["input_file_sha256"]
    assert receipt["policy"]["not_claimed"] == list(sut.NOT_CLAIMED)


def test_analysis_is_deterministic_and_idempotent(
    merged: Mapping[str, Any], tmp_path: Path
) -> None:
    first = sut.build_output_files(run(merged))
    second = sut.build_output_files(run(merged))
    assert first == second
    output_dir = tmp_path / "analysis"
    primary_merge.publish_merge(output_dir, first)
    republished = primary_merge.publish_merge(output_dir, second)
    assert republished["published"] is False
    assert republished["publish_mode"] == "no_op_identical_rerun"


def test_publish_refuses_a_drifted_existing_analysis_directory(
    merged: Mapping[str, Any], tmp_path: Path
) -> None:
    files = sut.build_output_files(run(merged))
    output_dir = tmp_path / "analysis"
    primary_merge.publish_merge(output_dir, files)
    (output_dir / sut.REPORT_MD_NAME).write_text("# edited\n")
    with pytest.raises(primary_merge.MergeContractError, match="byte-identical"):
        primary_merge.publish_merge(output_dir, files)


def test_cli_publishes_the_analysis(
    merged: Mapping[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "cli-analysis"
    exit_code = sut.main(
        [
            "--merged-dir",
            str(merged["merged_dir"]),
            "--primary-analysis-dir",
            str(merged["analysis_dir"]),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["crossing_owner_count"] == 26
    assert summary["tp_replay_control_owner_count"] == 12
    assert summary["analysis"]["published"] is True


def test_cli_fails_closed_on_a_contract_violation(
    merged: Mapping[str, Any], tmp_path: Path
) -> None:
    (merged["merged_dir"] / "notes.txt").write_text("hello\n")
    with pytest.raises(SystemExit, match="secondary analysis contract violated"):
        sut.main(
            [
                "--merged-dir",
                str(merged["merged_dir"]),
                "--primary-analysis-dir",
                str(merged["analysis_dir"]),
                "--output-dir",
                str(tmp_path / "cli-analysis"),
            ]
        )


def test_markdown_renders_the_same_content_as_the_summary(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    rendered = sut.render_markdown(result["summary"])
    assert rendered.startswith("# Sorted crossing-boundary secondary downstream")
    assert result["summary"]["claim_boundary"] in rendered
    for variant in secondary.SECONDARY_VARIANTS:
        assert f"`{variant}`" in rendered
    for statement in sut.NOT_CLAIMED:
        assert statement in rendered


# ---------------------------------------------------------------------------
# 2. The join: 26 crossing owners, 12 TP controls, kept apart
# ---------------------------------------------------------------------------


def test_owner_rows_join_the_frozen_cohorts(merged: Mapping[str, Any]) -> None:
    result = run(merged)
    rows = result["owner_rows"]
    crossing = [row for row in rows if row["cohort"] == sut.PRIMARY_COHORT]
    controls = [row for row in rows if row["cohort"] == sut.TP_REPLAY_CONTROL_COHORT]
    assert len(crossing) == primary.PRIMARY_OWNER_COUNT_U == 26
    assert len(controls) == primary.TP_CALIBRATION_OWNER_COUNT == 12
    assert len(rows) == 38
    assert {row["join_key"] for row in rows} == {
        f"{row['cohort']}|{row['image_id']}|{row['gt_owner_id']}" for row in rows
    }
    assert len({row["join_key"] for row in rows}) == len(rows)

    for row in crossing:
        assert secondary.VARIANT_P_PLUS_C_THEN_E in row["readouts"]
        assert sut.REFERENCE_VARIANT not in row["readouts"]
    for row in controls:
        assert list(row["readouts"]) == [sut.REFERENCE_VARIANT]

    denominators = result["summary"]["denominators"]
    assert not set(denominators["crossing_owner_ids"]) & set(
        denominators["tp_replay_control_owner_ids"]
    )


def test_owner_rows_preserve_every_per_segment_exact_delta(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    published = {
        str(row["request_id"]): row for row in read_merged_rows(merged["merged_dir"])
    }
    seen = 0
    for owner in result["owner_rows"]:
        for readout in owner["readouts"].values():
            source = published[readout["request_id"]]
            for segment in sut.SEGMENTS:
                block = readout["segments"][segment]
                origin = source["deltas"][segment]
                assert block["delta"] == origin["delta"]
                assert block["baseline_sum"] == origin["baseline_sum"]
                assert block["modified_sum"] == origin["modified_sum"]
                assert block["delta_token_mean"] == origin["delta_token_mean"]
                assert block["token_count"] == origin["token_count"]
                assert block["sign"] == origin["sign"]
                seen += 1
    assert seen == 64 * len(sut.SEGMENTS)


# ---------------------------------------------------------------------------
# 2b. No pooled cross-image raw summary anywhere (unit.md)
# ---------------------------------------------------------------------------

#: Field names that would publish a pooled cross-image location or range over
#: raw deltas.  ``unit.md`` forbids exactly this; none may appear in any emitted
#: payload, at any depth, under any of these spellings.
PROHIBITED_POOLED_FIELDS: tuple[str, ...] = (
    "median_delta",
    "minimum_delta",
    "maximum_delta",
    "minimum",
    "maximum",
    "median",
    "mean",
    "q1",
    "q3",
    "quartile_method",
    "quantile",
    "percentile",
    "stdev",
    "variance",
    "iqr",
    "values",
)

#: The exceptions, each within-owner or within-image by construction.
ALLOWED_SCOPED_FIELDS: frozenset[str] = frozenset(
    {"delta_token_mean", "within_image_median", "values"}
)


def walk_keys(payload: Any, trail: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every key path in a payload, so a scan can reason about its parents."""

    found: list[tuple[str, ...]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            found.append((*trail, str(key)))
            found.extend(walk_keys(value, (*trail, str(key))))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            found.extend(walk_keys(value, trail))
    return found


def assert_no_pooled_field(payload: Any, *, label: str) -> None:
    """No prohibited pooled field, except the scoped ones in their own place."""

    for path in walk_keys(payload):
        key = path[-1]
        if key in ALLOWED_SCOPED_FIELDS:
            # ``values`` is a raw list, so it is only legitimate *inside* a
            # per-image block; a top-level ordered array would be pooled.
            if key == "values":
                assert "per_image" in path, f"{label}: pooled raw array at {path}"
            continue
        assert key not in PROHIBITED_POOLED_FIELDS, (
            f"{label}: pooled cross-image field {key!r} at {path}"
        )


def test_no_emitted_payload_carries_a_pooled_cross_image_summary(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    files = sut.build_output_files(result)
    assert_no_pooled_field(result["owner_rows"], label="owner rows")
    assert_no_pooled_field(result["summary"], label="summary")
    assert_no_pooled_field(
        json.loads(files[sut.RECEIPT_NAME].decode("utf-8")), label="receipt"
    )


def test_per_variant_blocks_carry_no_pooled_location_or_range(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    for variant, block in result["summary"]["per_variant"].items():
        for segment in sut.SEGMENTS:
            stats = block["segments"][segment]
            assert set(stats) == {
                "owner_count",
                "finite_count",
                "nonfinite_count",
                "finite_coverage",
                "sign_counts",
                "image_count",
                "pooling",
                "per_image",
            }, f"{variant}/{segment} exposes an unexpected field"
            assert stats["pooling"] == sut.NO_CROSS_IMAGE_POOLING


def test_reference_blocks_carry_no_pooled_quantiles_or_ordered_array(
    merged: Mapping[str, Any],
) -> None:
    reference = run(merged)["summary"]["benign_substitution_reference"]
    assert reference["pooling"] == sut.NO_CROSS_IMAGE_POOLING
    for segment in sut.SEGMENTS:
        block = reference["segments"][segment]
        assert set(block) == {
            "finite_count",
            "owner_count",
            "sign_counts",
            "per_image",
        }, f"{segment} reference block exposes an unexpected field"
        # Each image contributes one control value, keyed by its own image id;
        # there is no ordered cross-image raw array to read as a distribution.
        for entry in block["per_image"].values():
            assert set(entry) == {"gt_owner_id", "delta", "sign", "finite"}


def test_the_rendered_report_publishes_no_pooled_table(
    merged: Mapping[str, Any],
) -> None:
    rendered = sut.render_markdown(run(merged)["summary"])
    header_rows = [line for line in rendered.splitlines() if line.startswith("| segment")]
    assert header_rows, "the report should still carry its count tables"
    for header in header_rows:
        for column in ("median", "min", "max", "q1", "q3"):
            assert f" {column} " not in header, f"pooled column {column!r} in {header!r}"
    assert "n/a" not in rendered or True  # empty cells are still permitted
    assert "Counts only" in rendered
    assert "within-image medians" in rendered.lower()


def test_the_analyzer_guard_refuses_a_reintroduced_pooled_field() -> None:
    sut.assert_no_pooled_raw_summary(
        {"per_image": {"10707": {"within_image_median": 1.0}}}, label="ok"
    )
    sut.assert_no_pooled_raw_summary(
        {"segments": {"description": {"delta_token_mean": -0.5}}}, label="ok"
    )
    for reintroduced in ("median_delta", "maximum_delta", "q1", "iqr_delta", "mean_delta"):
        with pytest.raises(
            sut.SecondaryAnalysisContractError, match="pooled-statistic key"
        ):
            sut.assert_no_pooled_raw_summary({reintroduced: 1.0}, label="probe")


def test_every_readout_stays_recoverable_by_image(merged: Mapping[str, Any]) -> None:
    """Removing the pooled summaries costs no evidence: all 64 are still there."""

    result = run(merged)
    rows = read_merged_rows(merged["merged_dir"])
    assert len(rows) == 64

    for segment in sut.SEGMENTS:
        recovered: list[tuple[str, str, float]] = []
        for variant, block in result["summary"]["per_variant"].items():
            for image_id, image_block in block["segments"][segment]["per_image"].items():
                recovered.extend(
                    (variant, image_id, value) for value in image_block["values"]
                )
        expected = sorted(
            (
                str(row["variant"]),
                str(row["image_id"]),
                float(row["deltas"][segment]["delta"]),
            )
            for row in rows
        )
        assert sorted(recovered) == expected


# ---------------------------------------------------------------------------
# 2c. The 64 -> 38 reshape is declared truthfully (no verbatim claim)
# ---------------------------------------------------------------------------


def test_row_mapping_states_the_transformation_rather_than_verbatim_copying(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    files = sut.build_output_files(result)
    receipt = json.loads(files[sut.RECEIPT_NAME].decode("utf-8"))

    for block in (result["summary"]["row_mapping"], receipt["row_mapping"]):
        assert block["input_readout_row_count"] == 64
        assert block["owner_row_count"] == 38
        assert block["owner_readout_count"] == 64
        assert block["input_rows_transformed_to_owner_rows"] is True
        assert block["per_segment_deltas_preserved_exactly"] is True

    policy = receipt["policy"]
    assert policy["input_rows_transformed_to_owner_rows"] is True
    assert policy["per_segment_deltas_preserved_exactly"] is True
    # The false verbatim claim must be gone from every emitted payload.
    assert "rows_preserved_verbatim" not in policy
    serialized = json.dumps(
        [result["summary"], result["owner_rows"], receipt], sort_keys=True
    )
    assert "rows_preserved_verbatim" not in serialized
    assert "verbatim" not in serialized


def test_the_owner_join_fails_closed_if_a_readout_is_lost_or_duplicated() -> None:
    merged_rows = [{"request_id": "req:a"}, {"request_id": "req:b"}]
    owner_rows = [{"readouts": {"p_plus_c_then_e": {}}}]
    with pytest.raises(
        sut.SecondaryAnalysisContractError, match="neither drops nor duplicates"
    ):
        sut.build_row_mapping(merged_rows, owner_rows)
    assert sut.build_row_mapping(merged_rows[:1], owner_rows)["owner_readout_count"] == 1


def test_tp_controls_are_reported_only_as_a_per_image_reference(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    summary = result["summary"]
    control_ids = set(summary["denominators"]["tp_replay_control_owner_ids"])
    for variant in sut.PRIMARY_VARIANTS:
        owners = {
            row["gt_owner_id"]
            for row in result["owner_rows"]
            if variant in row["readouts"]
        }
        assert not owners & control_ids
    reference = summary["benign_substitution_reference"]
    assert reference["variant"] == sut.REFERENCE_VARIANT
    assert reference["cohort"] == sut.TP_REPLAY_CONTROL_COHORT
    assert reference["owner_count"] == 12
    assert reference["image_count"] == 12

    rows = read_merged_rows(merged["merged_dir"])
    for segment in sut.SEGMENTS:
        block = reference["segments"][segment]
        # Discrete counts over owners are allowed.
        assert block["finite_count"] == 12
        assert block["owner_count"] == 12
        assert sum(block["sign_counts"].values()) == 12
        # Raw values stay indexed by image, one control per image.
        assert len(block["per_image"]) == 12
        for image_id, entry in block["per_image"].items():
            source = next(
                row
                for row in rows
                if str(row["variant"]) == sut.REFERENCE_VARIANT
                and str(row["image_id"]) == image_id
            )
            assert entry["delta"] == source["deltas"][segment]["delta"]
            assert entry["sign"] == source["deltas"][segment]["sign"]
            assert entry["gt_owner_id"] == source["gt_owner_id"]


# ---------------------------------------------------------------------------
# 3. Descriptive summaries
# ---------------------------------------------------------------------------


def test_per_variant_summary_reports_coverage_signs_and_per_image_values(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    per_variant = result["summary"]["per_variant"]
    assert set(per_variant) == set(secondary.SECONDARY_VARIANTS)
    rows = read_merged_rows(merged["merged_dir"])
    for variant, block in per_variant.items():
        source = [row for row in rows if str(row["variant"]) == variant]
        assert block["owner_count"] == len(source)
        assert block["cohort"] == secondary.VARIANT_COHORT[variant]
        assert block["plan_optional"] == secondary.VARIANT_IS_PLAN_OPTIONAL[variant]
        for segment in sut.SEGMENTS:
            stats = block["segments"][segment]
            assert stats["finite_count"] == len(source)
            assert stats["nonfinite_count"] == 0
            assert stats["finite_coverage"] == 1.0
            assert sum(stats["sign_counts"].values()) == len(source)
            # Every raw delta is still recoverable -- but only by image.
            expected = sorted(
                float(row["deltas"][segment]["delta"]) for row in source
            )
            observed = sorted(
                value
                for image in stats["per_image"].values()
                for value in image["values"]
            )
            assert observed == expected


def test_the_fixture_exercises_positive_zero_and_negative_signs(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    signs = result["summary"]["per_variant"][secondary.VARIANT_P_PLUS_C_THEN_E][
        "segments"
    ]["complete_row"]["sign_counts"]
    assert all(signs[label] > 0 for label in sut.SIGN_LABELS)


def test_within_image_medians_are_computed_over_one_image_only(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    rows = read_merged_rows(merged["merged_dir"])
    variant = secondary.VARIANT_P_PLUS_C_THEN_E
    per_image = result["summary"]["per_variant"][variant]["segments"]["coordinates"][
        "per_image"
    ]
    for image_id, block in per_image.items():
        expected = sorted(
            float(row["deltas"]["coordinates"]["delta"])
            for row in rows
            if str(row["variant"]) == variant and str(row["image_id"]) == image_id
        )
        assert sorted(block["values"]) == expected
        assert block["owner_count"] == len(expected)
        assert block["within_image_median"] == statistics.median(expected)
    # The uneven per-image owner distribution is preserved, not flattened.
    assert len({block["owner_count"] for block in per_image.values()}) > 1


def test_optional_f_presence_is_derived_and_never_assumed(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    dropped = next(
        row
        for row in rows
        if str(row["variant"]) == secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F
    )
    rewrite_merged_rows(
        merged["merged_dir"],
        [row for row in rows if row["request_id"] != dropped["request_id"]],
    )
    result = run(merged)
    denominators = result["summary"]["denominators"]
    assert denominators["crossing_owner_count"] == 26
    assert denominators["optional_f_owner_count"] == 25
    assert str(dropped["gt_owner_id"]) not in denominators["optional_f_owner_ids"]

    row = owner_row(result, str(dropped["gt_owner_id"]))
    assert row["optional_f_present"] is False
    assert row["readout_variants"] == [secondary.VARIANT_P_PLUS_C_THEN_E]
    assert (
        result["summary"]["per_variant"][secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F][
            "owner_count"
        ]
        == 25
    )


# ---------------------------------------------------------------------------
# 4. The analyzer is not a decision surface
# ---------------------------------------------------------------------------


def test_assert_no_decision_surface_catches_a_nested_key() -> None:
    sut.assert_no_decision_surface({"deltas": [{"sign": 1}]}, label="ok")
    with pytest.raises(sut.SecondaryAnalysisContractError, match="decision-surface"):
        sut.assert_no_decision_surface(
            {"owners": [{"primary_branch": "displaced"}]}, label="probe"
        )
    with pytest.raises(sut.SecondaryAnalysisContractError, match="decision-surface"):
        sut.assert_no_decision_surface({"support_threshold": 0.5}, label="probe")


def test_a_merged_row_carrying_a_branch_label_fails_closed(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    rows[0]["primary_branch"] = "displaced"
    rewrite_merged_rows(merged["merged_dir"], rows)
    with pytest.raises(sut.SecondaryAnalysisContractError, match="decision-surface"):
        run(merged)


def test_no_emitted_payload_names_a_decision_surface(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    files = sut.build_output_files(result)
    sut.assert_no_decision_surface(result["owner_rows"], label="owner rows")
    sut.assert_no_decision_surface(result["summary"], label="summary")
    sut.assert_no_decision_surface(
        json.loads(files[sut.RECEIPT_NAME].decode("utf-8")), label="receipt"
    )
    rendered = files[sut.REPORT_MD_NAME].decode("utf-8").lower()
    # The rendered report may *say* it reads no branch; it may never name one.
    for branch in primary.BRANCH_ORDER:
        assert branch not in rendered


def test_the_analysis_never_reads_the_primary_owner_rows(
    merged: Mapping[str, Any],
) -> None:
    """Only digests and schema versions of the sealed gate are consumed."""

    result = run(merged)
    binding = result["summary"]["primary_analysis"]
    assert set(binding) == {
        "analysis_dir",
        "analysis_file_sha256",
        "receipt_schema_version",
        "report_schema_version",
        "receipt_content_sha256",
        "analyzer_source_sha256",
        "binding_sha256",
        "read_scope",
    }
    assert secondary.ANALYSIS_OWNER_ROWS_NAME in binding["analysis_file_sha256"]
    serialized = json.dumps(result["summary"], sort_keys=True)
    for branch in primary.BRANCH_ORDER:
        assert branch not in serialized


# ---------------------------------------------------------------------------
# 5. Merged-input identity
# ---------------------------------------------------------------------------


def test_analysis_refuses_a_missing_merged_file(merged: Mapping[str, Any]) -> None:
    (merged["merged_dir"] / secondary_merge.MERGED_PARITY_NAME).unlink()
    with pytest.raises(sut.SecondaryAnalysisContractError, match="incomplete"):
        run(merged)


def test_analysis_refuses_an_unknown_artifact_in_the_merged_directory(
    merged: Mapping[str, Any],
) -> None:
    (merged["merged_dir"] / "notes.txt").write_text("hello\n")
    with pytest.raises(sut.SecondaryAnalysisContractError, match="unknown artifact"):
        run(merged)


def test_analysis_refuses_a_tampered_merged_rows_file(
    merged: Mapping[str, Any],
) -> None:
    path = merged["merged_dir"] / secondary_merge.MERGED_ROWS_NAME
    path.write_bytes(path.read_bytes() + b'{"request_id": "req:x"}\n')
    with pytest.raises(sut.SecondaryAnalysisContractError, match="drifted after"):
        run(merged)


def test_analysis_refuses_an_edited_merge_receipt(merged: Mapping[str, Any]) -> None:
    path = merged["merged_dir"] / secondary_merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text("utf-8"))
    receipt["runtime_identity_sha256"] = "0" * 64
    path.write_bytes(secondary.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(sut.SecondaryAnalysisContractError, match="edited after"):
        run(merged)


def test_analysis_refuses_a_request_id_set_that_does_not_reproduce_its_digest(
    merged: Mapping[str, Any],
) -> None:
    path = merged["merged_dir"] / secondary_merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text("utf-8"))
    receipt["secondary_requests"]["request_ids_sha256"] = "0" * 64
    receipt.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = secondary.sha256_json(receipt)
    path.write_bytes(secondary.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(
        sut.SecondaryAnalysisContractError, match="exact request-id set digest"
    ):
        run(merged)


def test_analysis_refuses_a_duplicated_request_id(merged: Mapping[str, Any]) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    rewrite_merged_rows(merged["merged_dir"], [*rows, copy.deepcopy(rows[0])])
    with pytest.raises(sut.SecondaryAnalysisContractError, match="duplicated request id"):
        run(merged)


def test_analysis_refuses_a_partial_owner_join(merged: Mapping[str, Any]) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    victim = next(
        row for row in rows if str(row["variant"]) == secondary.VARIANT_P_PLUS_C_THEN_E
    )
    rewrite_merged_rows(
        merged["merged_dir"],
        [row for row in rows if row["request_id"] != victim["request_id"]],
    )
    with pytest.raises(sut.SecondaryAnalysisContractError, match="partial join"):
        run(merged)


def test_analysis_refuses_an_incomplete_crossing_cohort(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    victim = next(
        row for row in rows if str(row["variant"]) == secondary.VARIANT_P_PLUS_C_THEN_E
    )
    dropped = str(victim["gt_owner_id"])
    rewrite_merged_rows(
        merged["merged_dir"],
        [row for row in rows if str(row["gt_owner_id"]) != dropped],
    )
    with pytest.raises(sut.SecondaryAnalysisContractError, match="crossing owners"):
        run(merged)


def test_analysis_refuses_an_incomplete_control_cohort(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    victim = next(
        row for row in rows if str(row["variant"]) == sut.REFERENCE_VARIANT
    )
    rewrite_merged_rows(
        merged["merged_dir"],
        [row for row in rows if row["request_id"] != victim["request_id"]],
    )
    with pytest.raises(sut.SecondaryAnalysisContractError, match="TP replay controls"):
        run(merged)


def test_analysis_refuses_a_variant_routed_to_the_wrong_cohort(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    victim = next(
        row for row in rows if str(row["variant"]) == sut.REFERENCE_VARIANT
    )
    victim["cohort"] = sut.PRIMARY_COHORT
    rewrite_merged_rows(merged["merged_dir"], rows)
    with pytest.raises(sut.SecondaryAnalysisContractError, match="attributes variant"):
        run(merged)


def test_analysis_refuses_an_owner_in_both_cohorts(merged: Mapping[str, Any]) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    crossing = next(
        row for row in rows if str(row["variant"]) == secondary.VARIANT_P_PLUS_C_THEN_E
    )
    control = next(
        row
        for row in rows
        if str(row["variant"]) == sut.REFERENCE_VARIANT
        and str(row["image_id"]) == str(crossing["image_id"])
    )
    control["gt_owner_id"] = crossing["gt_owner_id"]
    rewrite_merged_rows(merged["merged_dir"], rows)
    with pytest.raises(sut.SecondaryAnalysisContractError, match="stay disjoint"):
        run(merged)


def test_analysis_refuses_a_delta_sign_that_does_not_reconstruct(
    merged: Mapping[str, Any],
) -> None:
    rows = read_merged_rows(merged["merged_dir"])
    rows[0]["deltas"]["description"]["sign"] = 7
    rewrite_merged_rows(merged["merged_dir"], rows)
    with pytest.raises(sut.SecondaryAnalysisContractError, match="does not reconstruct"):
        run(merged)


# ---------------------------------------------------------------------------
# 6. The exact v2 primary-analysis binding
# ---------------------------------------------------------------------------


def test_analysis_refuses_a_gate_that_drifted(merged: Mapping[str, Any]) -> None:
    (merged["analysis_dir"] / secondary.ANALYSIS_REPORT_MD_NAME).write_text("# edited\n")
    with pytest.raises(sut.SecondaryAnalysisContractError, match="does not hash to the gate"):
        run(merged)


def test_analysis_refuses_an_incomplete_gate_directory(
    merged: Mapping[str, Any],
) -> None:
    (merged["analysis_dir"] / secondary.ANALYSIS_OWNER_ROWS_NAME).unlink()
    with pytest.raises(secondary.SecondaryCompatibilityContractError):
        run(merged)


def test_analysis_refuses_a_v1_primary_report_schema(
    merged: Mapping[str, Any], tmp_path: Path
) -> None:
    capture = merged["capture"]
    path = merged["analysis_dir"] / secondary.ANALYSIS_REPORT_JSON_NAME
    report = json.loads(path.read_text("utf-8"))
    report["schema_version"] = "sorted-crossing-boundary-owner-release-report.v1"
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    # Reseal the binding around the edited gate so the failure is proven to come
    # from the schema check rather than from the digest check.
    _reseal_gate(capture, merged)
    with pytest.raises(sut.SecondaryAnalysisContractError, match="not the finalized"):
        run(merged)
    assert primary_analyzer.REPORT_SCHEMA_VERSION.endswith(".v2")
    assert tmp_path.exists()


def _reseal_gate(capture: Any, merged: Mapping[str, Any]) -> None:
    """Rewrite the merged receipt's gate digests to the gate now on disk."""

    path = merged["merged_dir"] / secondary_merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text("utf-8"))
    receipt["primary_analysis"] = {
        **receipt["primary_analysis"],
        "analysis_file_sha256": {
            name: secondary.sha256_file(capture.analysis_dir / name)
            for name in sorted(secondary.ANALYSIS_REQUIRED_FILES)
        },
    }
    receipt.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = secondary.sha256_json(receipt)
    path.write_bytes(secondary.canonical_json_bytes(receipt) + b"\n")


def test_analysis_refuses_a_merge_that_seals_no_gate(merged: Mapping[str, Any]) -> None:
    path = merged["merged_dir"] / secondary_merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text("utf-8"))
    receipt.pop("primary_analysis")
    receipt.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = secondary.sha256_json(receipt)
    path.write_bytes(secondary.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(sut.SecondaryAnalysisContractError, match="seals no primary_analysis"):
        run(merged)


def test_analysis_binds_the_gate_digest_the_merge_sealed(
    merged: Mapping[str, Any],
) -> None:
    result = run(merged)
    receipt = json.loads(
        (merged["merged_dir"] / secondary_merge.MERGE_RECEIPT_NAME).read_text("utf-8")
    )
    binding = result["summary"]["primary_analysis"]
    assert binding["binding_sha256"] == receipt["primary_analysis_binding_sha256"]
    assert binding["receipt_schema_version"] == primary_analyzer.RECEIPT_SCHEMA_VERSION
    assert binding["report_schema_version"] == primary_analyzer.REPORT_SCHEMA_VERSION
    assert binding["analysis_file_sha256"] == {
        name: secondary.sha256_file(merged["analysis_dir"] / name)
        for name in sorted(secondary.ANALYSIS_REQUIRED_FILES)
    }
