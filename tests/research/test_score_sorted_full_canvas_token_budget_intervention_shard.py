"""Focused tests for the full-canvas visual-token-budget treatment shard scorer.

CPU-only.  The sealed plan fixture and the deterministic fake backend are the
predecessor census test's, imported rather than re-stated, so these tests prove
the *wrapper's* behaviour against the real scorer rather than against a mock of
it.  The real-HF session wiring is covered by stub modules that stand in for
``src.inference`` and friends, which is enough to prove the treatment
grid/media/prompt self-consistency gate without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import (  # noqa: E402
    prepare_sorted_full_canvas_token_budget_intervention as prepare,
)
from scripts.research import score_sorted_owner_accessibility_census_shard as census  # noqa: E402
from scripts.research import (  # noqa: E402
    score_sorted_full_canvas_token_budget_intervention_shard as successor,
)
from test_score_sorted_owner_accessibility_census_shard import (  # noqa: E402
    IMAGE_ID,
    PROMPT_TOKEN_IDS,
    build_plan_rows,
    seal_plan,
)

TREATMENT_PROMPT = [*PROMPT_TOKEN_IDS, 15, 16, 17]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_dir(tmp_path: Path) -> Path:
    return seal_plan(tmp_path / "plan", build_plan_rows())


def make_overlay(
    plan_dir: Path,
    *,
    selected: list[str] | None = None,
    arm_id: str = prepare.ARM_ID,
    **treatment_changes: Any,
) -> dict[str, Any]:
    plan = census.load_plan(plan_dir)
    image = plan.images[IMAGE_ID]
    if selected is None:
        selected = sorted(
            str(row["query_group_id"])
            for row in plan.image_query_groups(IMAGE_ID)
            if row["status"] == "admitted"
        )
    treatment = {
        "arm_id": arm_id,
        "max_pixels": prepare.TREATMENT_MAX_PIXELS,
        "width": 1920,
        "height": 1056,
        "media_relative_path": "media/6040.jpg",
        "executed_media_sha256": "treatment-media-digest",
        "file_sha256": "treatment-file-digest",
        "image_grid_thw": [1, 66, 120],
        "merged_visual_tokens": 1980,
        "prompt_token_ids": list(TREATMENT_PROMPT),
        "prompt_token_ids_sha256": planner.sha256_json(TREATMENT_PROMPT),
        "prompt_token_count": len(TREATMENT_PROMPT),
        "merged_visual_token_ratio_vs_current": 1980 / 966,
    }
    treatment.update(treatment_changes)
    overlay: dict[str, Any] = {
        "schema_version": prepare.OVERLAY_SCHEMA_VERSION,
        "intervention_unit_id": prepare.INTERVENTION_UNIT_ID,
        "arm_id": arm_id,
        "baseline_arm_id": prepare.BASELINE_ARM_ID,
        "max_pixels": prepare.TREATMENT_MAX_PIXELS,
        "claim_scope": "denser_patch_token_sampling_over_the_same_raw_optical_information",
        "base": {
            "plan_receipt_content_sha256": plan.receipt_content_sha256,
            "capture_rules_sha256": plan.capture_rules_sha256,
            "predecessor_run_root": "/nowhere",
        },
        "treatment_panel": {"relative_path": prepare.TREATMENT_PANEL_NAME},
        "images": {
            IMAGE_ID: {
                "file_name": "6040.jpg",
                "current": {
                    "arm_id": prepare.BASELINE_ARM_ID,
                    "width": int(image["image_width"]),
                    "height": int(image["image_height"]),
                    "executed_media_sha256": str(image["executed_media_sha256"]),
                    "prompt_token_ids_sha256": str(image["prompt_token_ids_sha256"]),
                    "prompt_token_count": len(image["prompt_token_ids"]),
                    "merged_visual_tokens": 966,
                    "image_grid_thw": [1, 46, 84],
                },
                "treatment": treatment,
            }
        },
        "query_group_selection": {"query_group_ids_by_image": {IMAGE_ID: list(selected)}},
    }
    overlay["overlay_content_sha256"] = prepare.overlay_content_sha256(overlay)
    return overlay


def write_overlay(root: Path, overlay: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / prepare.OVERLAY_NAME
    path.write_bytes(planner.canonical_json_bytes(overlay) + b"\n")
    return path


@pytest.fixture
def bundle(tmp_path: Path, plan_dir: Path) -> successor.InterventionPlan:
    overlay_path = write_overlay(tmp_path / "overlay", make_overlay(plan_dir))
    return successor.load_intervention_plan(plan_dir, overlay_path)


# ---------------------------------------------------------------------------
# Overlay application through the real plan bundle
# ---------------------------------------------------------------------------


def test_loading_applies_the_treatment_prompt_and_reseals_admissions(
    plan_dir: Path, bundle: successor.InterventionPlan
):
    baseline = census.load_plan(plan_dir)
    image = bundle.plan.images[IMAGE_ID]
    assert image["prompt_token_ids"] == TREATMENT_PROMPT

    for group_id, group in bundle.plan.query_groups.items():
        original = baseline.query_groups[group_id]
        assert group["query_suffix_token_ids"] == original["query_suffix_token_ids"]
        assert group["candidate_ids"] == original["candidate_ids"]
        assert group["query_prefix_sha256"] != original["query_prefix_sha256"]

    # And the reseal is consistent enough for the real scorer to resolve every
    # work item: this is what proves the recompute matched the literal tokens.
    for group_id in bundle.plan.query_groups:
        item = census.resolve_work_item(bundle.plan, group_id)
        assert item.query_prefix_token_ids[: len(TREATMENT_PROMPT)] == TREATMENT_PROMPT


def test_a_foreign_plan_digest_is_refused(tmp_path: Path, plan_dir: Path):
    overlay = make_overlay(plan_dir)
    overlay["base"]["plan_receipt_content_sha256"] = "0" * 64
    overlay["overlay_content_sha256"] = prepare.overlay_content_sha256(overlay)
    path = write_overlay(tmp_path / "foreign", overlay)
    with pytest.raises(census.ShardContractError, match="different predecessor plan"):
        successor.load_intervention_plan(plan_dir, path)


def test_a_foreign_arm_is_refused(tmp_path: Path, plan_dir: Path):
    path = write_overlay(tmp_path / "arm", make_overlay(plan_dir, arm_id="other-arm"))
    with pytest.raises(census.ShardContractError, match="this unit executes"):
        successor.load_intervention_plan(plan_dir, path)


# ---------------------------------------------------------------------------
# Score-only capture
# ---------------------------------------------------------------------------


def test_successor_capture_is_score_only_and_fully_stamped(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    output = tmp_path / "shards" / IMAGE_ID
    result = successor.run_intervention_shard(
        bundle,
        image_id=IMAGE_ID,
        backend=census.FakeCensusBackend(),
        output_dir=output,
    )
    assert result.free_decodes == []
    assert result.scores and result.proposals and result.x1

    phases = result.receipt["phase_order"]
    assert phases["behavior_sidecars_captured"] is False
    assert phases["behavior_sidecars_intentionally_disabled"] is True
    assert phases["generation_phase_after_decision_scoring"] is False
    assert result.receipt["counts"]["free_decode_sidecar_rows"] == 0

    stamp = result.receipt[successor.ROW_STAMP_KEY]
    assert stamp["arm_id"] == prepare.ARM_ID
    assert stamp["overlay_content_sha256"] == bundle.overlay_content_sha256
    assert stamp["base_plan_receipt_content_sha256"] == (
        bundle.base_plan_receipt_content_sha256
    )
    assert stamp["max_pixels"] == prepare.TREATMENT_MAX_PIXELS
    assert stamp["image_grid_thw"] == [1, 66, 120]
    assert stamp["merged_visual_tokens"] == 1980
    assert stamp["baseline_merged_visual_tokens"] == 966
    assert stamp["executed_media_sha256"] == "treatment-media-digest"
    assert stamp["cross_arm_raw_logprob_comparison"] == "forbidden"
    assert result.receipt["intervention_capture_mode"]["score_only"] is True

    for rows in (result.scores, result.x1, result.proposals):
        assert rows
        for row in rows:
            assert row[successor.ROW_STAMP_KEY] == stamp

    assert (output / census.RECEIPT_NAME).is_file()
    published = json.loads((output / census.RECEIPT_NAME).read_text())
    assert published[successor.ROW_STAMP_KEY]["arm_id"] == prepare.ARM_ID


def test_score_only_capture_publishes_the_canonical_complete_file_set(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    """The score-only contract, stated as the published artifacts see it.

    The canonical writer publishes the schema-complete file set atomically, so
    the successor does *not* trim ``free-decode-sidecars.jsonl`` away.  The
    contract is therefore three separate claims:

    1. no free-decode row is generated;
    2. the schema-compatible sidecar artifact may be present, and when present
       it must be empty -- a zero-byte file is the expected shape of a
       score-only capture, never a truncated or failed one;
    3. the receipt fields, not the file's presence or size, are authoritative.
    """

    output = tmp_path / "shards" / IMAGE_ID
    result = successor.run_intervention_shard(
        bundle,
        image_id=IMAGE_ID,
        backend=census.FakeCensusBackend(),
        output_dir=output,
    )

    # (1) No row is generated, in memory or on disk.
    assert result.free_decodes == []

    # (2) The file set is the canonical one -- not trimmed -- and the sidecar
    #     artifact is present and exactly zero bytes.
    sidecar = output / census.FREE_DECODE_NAME
    published_names = {child.name for child in output.iterdir()}
    assert published_names == {
        census.SCORES_NAME,
        census.X1_NAME,
        census.PROPOSAL_NAME,
        census.FREE_DECODE_NAME,
        census.RECEIPT_NAME,
    }
    assert sidecar.is_file()
    assert sidecar.stat().st_size == 0
    assert sidecar.read_bytes() == b""
    # The other published files are non-empty, so "empty" is a statement about
    # this artifact and not about a failed publish.
    for name in (census.SCORES_NAME, census.PROPOSAL_NAME, census.RECEIPT_NAME):
        assert (output / name).stat().st_size > 0

    # (3) The receipt is the authority on the capture mode.
    receipt = json.loads((output / census.RECEIPT_NAME).read_text())
    assert receipt["phase_order"]["behavior_sidecars_intentionally_disabled"] is True
    assert receipt["counts"]["free_decode_sidecar_rows"] == 0
    assert receipt["intervention_capture_mode"] == {
        "score_only": True,
        "behavior_sidecars_captured": False,
        "behavior_sidecars_intentionally_disabled": True,
        "reason": "score_only_successor_no_free_greedy_box_or_row_sidecars",
    }


def test_the_module_contract_does_not_claim_the_sidecar_file_is_absent():
    """Guard the prose that the real-HF smoke contradicted.

    The published shard *does* carry a zero-byte ``free-decode-sidecars.jsonl``,
    so documentation promising that no such file is left behind is wrong and
    would make a correct capture look like a failed one to the next reader.
    """

    doc = successor.__doc__ or ""
    assert "free-decode-sidecars.jsonl" in doc
    assert "may be present" in doc
    assert "must" in doc and "empty" in doc
    assert "behavior_sidecars_intentionally_disabled" in doc
    assert "free_decode_sidecar_rows" in doc
    for retracted in (
        "rather than leaving an empty file",
        "no such file",
        "no sidecar file",
    ):
        assert retracted not in doc


def test_only_the_sealed_selection_is_executed(tmp_path: Path, plan_dir: Path):
    chosen = f"{IMAGE_ID}:boundary-000|person"
    overlay = make_overlay(plan_dir, selected=[chosen])
    bundle = successor.load_intervention_plan(
        plan_dir, write_overlay(tmp_path / "one", overlay)
    )
    result = successor.run_intervention_shard(
        bundle, image_id=IMAGE_ID, backend=census.FakeCensusBackend(), output_dir=None
    )
    assert {row["query_group_id"] for row in result.scores} == {chosen}
    assert result.receipt["intervention_selection"]["selected_query_group_ids"] == [chosen]
    assert result.receipt["intervention_selection"]["selection_uses_treatment_scores"] is False
    assert result.receipt["capture_completeness"] == "subset_smoke"


def test_full_run_is_complete_relative_to_the_overlay_despite_extra_base_groups(
    tmp_path: Path, plan_dir: Path
):
    """Base-plan subset != intervention subset.

    The base plan admits four groups for this image; the overlay freezes two.
    A full run therefore executes a *subset of the base plan* -- so the
    predecessor's base-relative ``capture_completeness`` correctly stays
    ``subset_smoke`` -- while being **complete** relative to the frozen overlay
    selection, which is the only completeness this unit's analysis reads.
    """

    plan = census.load_plan(plan_dir)
    all_admitted = sorted(
        str(row["query_group_id"])
        for row in plan.image_query_groups(IMAGE_ID)
        if row["status"] == "admitted"
    )
    assert len(all_admitted) == 4
    frozen = all_admitted[:2]
    bundle = successor.load_intervention_plan(
        plan_dir, write_overlay(tmp_path / "partial", make_overlay(plan_dir, selected=frozen))
    )
    result = successor.run_intervention_shard(
        bundle, image_id=IMAGE_ID, backend=census.FakeCensusBackend(), output_dir=None
    )

    # (1) The predecessor's base-plan fields are preserved verbatim, not relabelled.
    assert result.receipt["capture_completeness"] == "subset_smoke"
    assert result.receipt["subset_capture"]["is_subset"] is True
    assert result.receipt["subset_capture"]["planned_admitted_query_group_count"] == 4
    assert result.receipt["subset_capture"]["executed_query_group_count"] == 2
    assert result.receipt["subset_capture"]["usable_as_complete_shard_evidence"] is False

    # (2) The intervention-relative block is the unit's authority, and says complete.
    completeness = result.receipt[successor.COMPLETENESS_KEY]
    assert completeness["status"] == successor.COMPLETENESS_COMPLETE
    assert completeness["is_complete_frozen_overlay_selection"] is True
    assert completeness["relative_to"] == "overlay_frozen_selection_for_this_image"
    assert completeness["frozen_expected_query_group_count"] == 2
    assert completeness["executed_query_group_count"] == 2
    assert completeness["missing_query_group_ids"] == []
    assert completeness["extra_query_group_ids"] == []
    assert completeness["derived_from"] == "query_group_ids_carrying_score_rows"
    assert "provenance" in completeness["base_plan_fields_role"]
    assert {row["query_group_id"] for row in result.scores} == set(frozen)


def test_smoke_subset_publishes_subset_smoke_completeness(tmp_path: Path, plan_dir: Path):
    plan = census.load_plan(plan_dir)
    frozen = sorted(
        str(row["query_group_id"])
        for row in plan.image_query_groups(IMAGE_ID)
        if row["status"] == "admitted"
    )
    bundle = successor.load_intervention_plan(
        plan_dir, write_overlay(tmp_path / "smoke", make_overlay(plan_dir, selected=frozen))
    )
    result = successor.run_intervention_shard(
        bundle,
        image_id=IMAGE_ID,
        backend=census.FakeCensusBackend(),
        output_dir=None,
        max_query_groups=1,
    )
    completeness = result.receipt[successor.COMPLETENESS_KEY]
    assert completeness["status"] == successor.COMPLETENESS_SUBSET
    assert completeness["is_complete_frozen_overlay_selection"] is False
    assert completeness["frozen_expected_query_group_count"] == len(frozen)
    assert completeness["executed_query_group_count"] == 1
    assert completeness["missing_query_group_ids"] == sorted(frozen[1:])
    assert completeness["extra_query_group_ids"] == []


def test_completeness_is_derived_from_score_rows_not_from_the_request():
    """A group that produced no score row can never read as executed."""

    frozen = ["g:a", "g:b"]
    result = census.ShardResult(receipt={}, scores=[{"query_group_id": "g:a"}])
    block = successor.build_completeness_block(result, frozen_selection=frozen)
    assert block["status"] == successor.COMPLETENESS_SUBSET
    assert block["missing_query_group_ids"] == ["g:b"]
    assert block["executed_query_group_count"] == 1


def test_scoring_outside_the_frozen_selection_is_refused_in_either_mode():
    result = census.ShardResult(
        receipt={},
        scores=[{"query_group_id": "g:a"}, {"query_group_id": "g:rogue"}],
    )
    with pytest.raises(census.ShardContractError, match="outside the sealed overlay"):
        successor.build_completeness_block(result, frozen_selection=["g:a"])


def test_an_image_outside_the_selection_is_refused(bundle: successor.InterventionPlan):
    with pytest.raises(census.ShardContractError, match="selects no query group"):
        bundle.selected_query_group_ids("9999")


def test_contract_validation_is_cpu_only_and_reports_the_token_budget(
    bundle: successor.InterventionPlan,
):
    report = successor.validate_contract(bundle, IMAGE_ID)
    assert report["arm_id"] == prepare.ARM_ID
    assert report["treatment_merged_visual_tokens"] == 1980
    assert report["baseline_merged_visual_tokens"] == 966
    assert report["treatment_prompt_token_count"] == len(TREATMENT_PROMPT)
    assert report["selected_query_group_count"] == 4
    assert report["score_only"] is True


# ---------------------------------------------------------------------------
# Arm / overlay mixing
# ---------------------------------------------------------------------------


def _publish_foreign_receipt(root: Path, name: str, payload: dict[str, Any]) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / census.RECEIPT_NAME).write_text(json.dumps(payload), encoding="utf-8")


def test_arm_mixing_is_refused_at_the_shard_root(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    root = tmp_path / "shards"
    _publish_foreign_receipt(
        root,
        "9999",
        {
            successor.ROW_STAMP_KEY: {
                "arm_id": "another-arm",
                "overlay_content_sha256": bundle.overlay_content_sha256,
                "base_plan_receipt_content_sha256": bundle.base_plan_receipt_content_sha256,
            }
        },
    )
    with pytest.raises(census.ShardContractError, match="never pooled"):
        successor.assert_no_arm_mixing(root, bundle)


def test_overlay_mixing_is_refused_at_the_shard_root(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    root = tmp_path / "shards"
    _publish_foreign_receipt(
        root,
        "9999",
        {
            successor.ROW_STAMP_KEY: {
                "arm_id": prepare.ARM_ID,
                "overlay_content_sha256": "0" * 64,
                "base_plan_receipt_content_sha256": bundle.base_plan_receipt_content_sha256,
            }
        },
    )
    with pytest.raises(census.ShardContractError, match="mix overlays"):
        successor.assert_no_arm_mixing(root, bundle)


def test_an_unstamped_predecessor_capture_in_the_root_is_refused(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    root = tmp_path / "shards"
    _publish_foreign_receipt(root, "9999", {"unit_id": census.UNIT_ID})
    with pytest.raises(census.ShardContractError, match="never share a shard root"):
        successor.assert_no_arm_mixing(root, bundle)


def test_a_matching_arm_and_overlay_is_accepted(
    bundle: successor.InterventionPlan, tmp_path: Path
):
    root = tmp_path / "shards"
    _publish_foreign_receipt(
        root,
        "9999",
        {
            successor.ROW_STAMP_KEY: {
                "arm_id": prepare.ARM_ID,
                "overlay_content_sha256": bundle.overlay_content_sha256,
                "base_plan_receipt_content_sha256": bundle.base_plan_receipt_content_sha256,
            }
        },
    )
    report = successor.assert_no_arm_mixing(root, bundle)
    assert report["mixing_detected"] is False
    assert len(report["inspected_shard_receipts"]) == 1


def test_restamping_a_stamped_row_is_refused(bundle: successor.InterventionPlan):
    result = census.ShardResult(receipt={}, scores=[{successor.ROW_STAMP_KEY: {}}])
    with pytest.raises(census.ShardContractError, match="already carries"):
        successor.stamp_result(
            result,
            stamp={"arm_id": prepare.ARM_ID},
            selected=["g"],
            frozen_selection=["g"],
        )


def test_a_sidecar_bearing_result_is_refused_by_the_score_only_stamp():
    result = census.ShardResult(receipt={}, free_decodes=[{"row_kind": "x"}])
    with pytest.raises(census.ShardContractError, match="score only"):
        successor.stamp_result(
            result,
            stamp={"arm_id": prepare.ARM_ID},
            selected=["g"],
            frozen_selection=["g"],
        )


# ---------------------------------------------------------------------------
# Predecessor default compatibility
# ---------------------------------------------------------------------------


def test_predecessor_default_still_captures_behavior_sidecars(plan_dir: Path):
    plan = census.load_plan(plan_dir)
    default = census.run_shard(
        plan, image_id=IMAGE_ID, backend=census.FakeCensusBackend(), output_dir=None
    )
    assert default.free_decodes
    assert default.receipt["phase_order"]["generation_phase_after_decision_scoring"] is True
    assert default.receipt["phase_order"]["behavior_sidecars_captured"] is True
    assert default.receipt["phase_order"]["behavior_sidecars_intentionally_disabled"] is False

    disabled = census.run_shard(
        plan,
        image_id=IMAGE_ID,
        backend=census.FakeCensusBackend(),
        output_dir=None,
        capture_behavior_sidecars=False,
    )
    assert disabled.free_decodes == []
    assert disabled.receipt["phase_order"]["behavior_sidecars_intentionally_disabled"] is True
    assert disabled.receipt["phase_order"]["behavior_sidecar_disable_reason"]

    # Disabling the terminal behavior phase changes nothing about the decision
    # rows: the score-only successor measures the same quantity.
    assert [row["complete_box_logprob_sum"] for row in disabled.scores] == [
        row["complete_box_logprob_sum"] for row in default.scores
    ]
    assert disabled.receipt["counts"]["localization_score_rows"] == (
        default.receipt["counts"]["localization_score_rows"]
    )


# ---------------------------------------------------------------------------
# Treatment session self-consistency (stubbed production frontend)
# ---------------------------------------------------------------------------


class _ImagePlanRow:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _install_frontend_stubs(
    monkeypatch: pytest.MonkeyPatch, *, image_plan: _ImagePlanRow, prompt_token_ids: list[int]
) -> None:
    """Stand in for the production frontend without loading a processor."""

    def _module(name: str, **attributes: Any) -> None:
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    class _Generation:
        repetition_penalty = 1.0

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"repetition_penalty": 1.0}

    class _Config:
        backend = types.SimpleNamespace(type="hf")
        generation = _Generation()

    _module("src.config.fingerprint", sha256_json=lambda value: "config-digest")
    _module(
        "src.config.inference",
        load_infer_config=lambda path: types.SimpleNamespace(config=_Config()),
    )
    _module(
        "src.data",
        load_raw_examples=lambda path: (
            types.SimpleNamespace(metadata={"source": {"image_id": IMAGE_ID}}),
        ),
    )
    _module(
        "src.inference.backend",
        DecodeRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GenerationPolicy=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    _module(
        "src.inference.image_plan",
        plan_image_batch=lambda *args, **kwargs: types.SimpleNamespace(rows=[image_plan]),
    )
    _module(
        "src.inference.pipeline",
        _processor_config=lambda config: types.SimpleNamespace(do_resize=False),
        _template_config=lambda config: object(),
    )
    _module(
        "src.inference.prompt",
        build_prompt_record=lambda *args, **kwargs: types.SimpleNamespace(
            chat_text="chat",
            input_prompt_token_ids=tuple(prompt_token_ids),
            expected_executed_prompt_token_ids=tuple(prompt_token_ids),
        ),
    )
    _module(
        "src.inference.runtime",
        assemble_frontend=lambda config, **kwargs: types.SimpleNamespace(
            qwen=types.SimpleNamespace(processor=object()), launch=object()
        ),
    )


@pytest.fixture
def infer_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("backend:\n  type: hf\n", encoding="utf-8")
    return path


def _consistent_image_plan(**overrides: Any) -> _ImagePlanRow:
    base = {
        "expected_image_grid_thw": [1, 66, 120],
        "merged_visual_tokens": 1980,
        "declared_width": 1920,
        "declared_height": 1056,
        "decoded_width": 1920,
        "decoded_height": 1056,
        "image_content_sha256": "treatment-file-digest",
        "image_path": "/tmp/media/6040.jpg",
        "logical_transform_id": "identity",
    }
    base.update(overrides)
    return _ImagePlanRow(**base)


def test_treatment_session_spec_binds_the_overlay_identity(
    bundle: successor.InterventionPlan,
    monkeypatch: pytest.MonkeyPatch,
    infer_config: Path,
):
    _install_frontend_stubs(
        monkeypatch,
        image_plan=_consistent_image_plan(),
        prompt_token_ids=list(TREATMENT_PROMPT),
    )
    spec = successor.build_treatment_session_spec(
        bundle, IMAGE_ID, infer_config=infer_config
    )
    assert spec.image_grid_thw == (1, 66, 120)
    # The session gate is handed the *overlaid* plan values, so the real
    # materialized session is checked against the treatment identity.
    assert spec.planned_prompt_token_ids == TREATMENT_PROMPT
    assert spec.planned_executed_media_sha256 == "treatment-media-digest"
    assert spec.repetition_penalty_stratum == 1.0


@pytest.mark.parametrize(
    ("overrides", "prompt", "message"),
    [
        ({"expected_image_grid_thw": [1, 46, 84]}, None, "image_grid_thw"),
        ({"merged_visual_tokens": 966}, None, "merged visual tokens"),
        ({"decoded_width": 1024}, None, "canvas"),
        ({"image_content_sha256": "other"}, None, "media file digest"),
        ({}, [1, 2, 3], "prompt tokens differ"),
    ],
)
def test_treatment_session_spec_fails_closed_on_any_disagreement(
    bundle: successor.InterventionPlan,
    monkeypatch: pytest.MonkeyPatch,
    infer_config: Path,
    overrides: dict[str, Any],
    prompt: list[int] | None,
    message: str,
):
    _install_frontend_stubs(
        monkeypatch,
        image_plan=_consistent_image_plan(**overrides),
        prompt_token_ids=list(prompt or TREATMENT_PROMPT),
    )
    with pytest.raises(census.ShardContractError, match=message):
        successor.build_treatment_session_spec(
            bundle, IMAGE_ID, infer_config=infer_config
        )
