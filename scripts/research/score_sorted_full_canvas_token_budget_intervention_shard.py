#!/usr/bin/env python3
"""Treatment shard scorer for the sorted full-canvas visual-token-budget intervention.

This is a thin wrapper, not a second scorer.  Every scoring seam -- exact-prefix
admission, KV parity, candidate collapse, competition, proposal channels,
atomic publish -- is the predecessor census scorer
(``score_sorted_owner_accessibility_census_shard.py``) executed unchanged.  What
this module owns is:

1. loading the sealed predecessor plan **and** the sealed intervention overlay;
2. applying the overlay in memory (treatment prompt prefix, and the
   prompt-derived observed/query prefix digests and exact-prefix admission IDs);
3. building the real HF session from the *treatment* panel and media under the
   same sorted step-4887 fp32 rp1.0 model/config identity;
4. executing only the overlay's frozen query-group selection for one image;
5. stamping intervention/arm/overlay/base identity into the receipt and into
   every decision-bearing row before the atomic write;
6. refusing arm or overlay mixing at the shard root.

Score-only contract
-------------------
This successor is **score only**.  It passes ``capture_behavior_sidecars=False``
to the predecessor scorer, so **no free-decode row is generated**: no free greedy
box sidecar and no free next-row sidecar.

The published artifact set is *not* trimmed to match.  The canonical shard writer
publishes the schema-complete file set as one indivisible, atomically renamed
group, so ``free-decode-sidecars.jsonl`` **may be present and, when present, must
be empty (zero bytes)**.  That is the expected shape of a score-only capture, not
a truncated or failed one; keeping the file set uniform is what lets one reader
consume predecessor and successor shards without special-casing.

A zero-byte sidecar file is therefore never the authority on what was captured.
The authoritative receipt fields are:

* ``phase_order.behavior_sidecars_intentionally_disabled == true`` -- the
  terminal generation phase was never opened, so nothing was lost;
* ``counts.free_decode_sidecar_rows == 0`` -- and
  ``intervention_capture_mode.score_only == true``.

A capture that somehow produced behavior rows is refused by
:func:`stamp_result` before anything is written.

Completeness has two different meanings here
--------------------------------------------
This wrapper deliberately executes a *subset* of the predecessor base plan --
namely the overlay's frozen selection -- so ``census.run_shard`` correctly
stamps the base-plan-relative ``capture_completeness = "subset_smoke"`` even
when the intervention capture is complete.  Those base fields
(``capture_completeness``, ``subset_capture``) are preserved verbatim as
provenance and are never relabelled.

The authority for this unit is the separate ``intervention_completeness`` block,
which is relative to the overlay's frozen selection for that image:

* ``complete_frozen_overlay_selection`` -- every frozen group carries score
  rows; only this value may enter a conclusion-bearing analysis;
* ``subset_smoke`` -- a deliberate ``--smoke-query-groups`` subset.

It also records the frozen expected count, the executed count, and the missing
and extra ID lists (both empty for a full run).  Scoring outside the frozen
selection is refused in either mode.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import prepare_sorted_full_canvas_token_budget_intervention as prepare  # noqa: E402
from scripts.research import score_sorted_owner_accessibility_census_shard as census  # noqa: E402

INTERVENTION_UNIT_ID = prepare.INTERVENTION_UNIT_ID
ARM_ID = prepare.ARM_ID
RECEIPT_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-intervention-shard-receipt.v1"
ROW_STAMP_KEY = "intervention"

#: Receipt key holding this unit's *intervention-relative* completeness, as
#: distinct from the predecessor's base-plan-relative ``capture_completeness``.
COMPLETENESS_KEY = "intervention_completeness"

#: A shard that executed the overlay's complete frozen selection for its image.
#: Only this value may enter a conclusion-bearing analysis.
COMPLETENESS_COMPLETE = "complete_frozen_overlay_selection"

#: A deliberate ``--smoke-query-groups`` subset of that frozen selection.
COMPLETENESS_SUBSET = "subset_smoke"

ShardContractError = census.ShardContractError

EXECUTED_SOURCE_SHA256 = planner.sha256_file(Path(__file__).resolve())

DEFAULT_INFER_CONFIG = census.DEFAULT_INFER_CONFIG


# ---------------------------------------------------------------------------
# Plan + overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterventionPlan:
    """The predecessor plan with the treatment presentation applied in memory."""

    plan: census.PlanBundle
    overlay: Mapping[str, Any]
    overlay_path: Path
    application: Mapping[str, Any]

    @property
    def arm_id(self) -> str:
        return str(self.overlay["arm_id"])

    @property
    def overlay_content_sha256(self) -> str:
        return str(self.overlay["overlay_content_sha256"])

    @property
    def base_plan_receipt_content_sha256(self) -> str:
        return str(self.overlay["base"]["plan_receipt_content_sha256"])

    def selected_query_group_ids(self, image_id: str) -> list[str]:
        selection = self.overlay["query_group_selection"]["query_group_ids_by_image"]
        group_ids = selection.get(str(image_id))
        if not group_ids:
            raise ShardContractError(
                f"the overlay selects no query group for image {image_id!r}; this image is "
                "not part of the intervention capture"
            )
        return [str(value) for value in group_ids]

    def treatment_panel_path(self) -> Path:
        relative = str(self.overlay["treatment_panel"]["relative_path"])
        return (self.overlay_path.parent / relative).resolve()


def load_intervention_plan(base_plan_dir: Path, overlay_path: Path) -> InterventionPlan:
    """Load both sealed inputs and apply the overlay to an in-memory plan copy."""

    overlay_path = Path(overlay_path).expanduser().resolve(strict=True)
    try:
        overlay = prepare.load_overlay(overlay_path)
    except prepare.PrepareContractError as exc:
        raise ShardContractError(str(exc)) from exc

    plan = census.load_plan(Path(base_plan_dir))
    try:
        prepare.assert_overlay_binds_plan(
            overlay,
            plan_receipt_content_sha256=plan.receipt_content_sha256,
            capture_rules_sha256=plan.capture_rules_sha256,
        )
        application = prepare.apply_overlay(
            overlay,
            images=plan.images,
            contexts=plan.contexts,
            query_groups=plan.query_groups,
            categories=plan.categories,
            candidates=plan.candidates,
            owners=plan.owners,
        )
    except prepare.PrepareContractError as exc:
        raise ShardContractError(str(exc)) from exc

    return InterventionPlan(
        plan=plan,
        overlay=overlay,
        overlay_path=overlay_path,
        application=application,
    )


# ---------------------------------------------------------------------------
# Treatment session
# ---------------------------------------------------------------------------


def build_treatment_session_spec(
    bundle: InterventionPlan, image_id: str, *, infer_config: Path
) -> census.HFSessionSpec:
    """Resolve one image's *treatment* production request without opening a model.

    Same path as the predecessor's ``build_hf_session_spec`` -- resolved infer
    config -> ``assemble_frontend`` -> ``plan_image_batch`` ->
    ``build_prompt_record`` -- but reading the treatment panel instead of the
    config's frozen 1x panel, and cross-checking every resolved quantity against
    the sealed overlay before the model is ever opened.
    """

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    image_id = str(image_id)
    block = (bundle.overlay["images"] or {}).get(image_id)
    if block is None:
        raise ShardContractError(f"the overlay carries no image {image_id!r}")
    treatment = block["treatment"]

    resolved = load_infer_config(Path(infer_config).expanduser().resolve(strict=True))
    config = resolved.config
    if config.backend.type != "hf":
        raise ShardContractError("the intervention scorer requires backend.type: hf")
    census.assert_repetition_penalty_stratum(
        config.generation.repetition_penalty, label=f"infer config {infer_config}"
    )
    processor_config = _processor_config(config)
    if getattr(processor_config, "do_resize", False):
        raise ShardContractError(
            "processor do_resize must stay False; a processor-side resize would "
            "reinterpret the treatment canvas and dissolve the intervention"
        )

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    panel_path = bundle.treatment_panel_path()
    raw_rows = load_raw_examples(panel_path)
    raw = None
    row_index = 0
    for index, row in enumerate(raw_rows):
        source = row.metadata.get("source")
        if isinstance(source, Mapping) and str(source.get("image_id")) == image_id:
            raw = row
            row_index = index
            break
    if raw is None:
        raise ShardContractError(
            f"image {image_id!r} is absent from the treatment panel at {panel_path}"
        )

    image_plan = plan_image_batch(
        [raw],
        components=frontend.qwen,
        processor_config=processor_config,
        row_indices=[row_index],
    ).rows[0]
    prompt_record = build_prompt_record(
        raw,
        _template_config(config),
        processor=frontend.qwen.processor,
        row_index=row_index,
        merged_visual_tokens=image_plan.merged_visual_tokens,
    )
    grid = tuple(int(value) for value in image_plan.expected_image_grid_thw)

    if list(grid) != [int(value) for value in treatment["image_grid_thw"]]:
        raise ShardContractError(
            f"image {image_id!r}: resolved treatment image_grid_thw {list(grid)!r} differs "
            f"from the sealed overlay {list(treatment['image_grid_thw'])!r}"
        )
    if int(image_plan.merged_visual_tokens) != int(treatment["merged_visual_tokens"]):
        raise ShardContractError(
            f"image {image_id!r}: resolved merged visual tokens differ from the overlay"
        )
    if (int(image_plan.decoded_width), int(image_plan.decoded_height)) != (
        int(treatment["width"]),
        int(treatment["height"]),
    ):
        raise ShardContractError(
            f"image {image_id!r}: resolved treatment canvas differs from the overlay"
        )
    if str(image_plan.image_content_sha256) != str(treatment["file_sha256"]):
        raise ShardContractError(
            f"image {image_id!r}: treatment media file digest differs from the overlay"
        )
    executed_prompt = [int(value) for value in prompt_record.expected_executed_prompt_token_ids]
    if executed_prompt != [int(value) for value in treatment["prompt_token_ids"]]:
        raise ShardContractError(
            f"image {image_id!r}: resolved treatment prompt tokens differ from the overlay"
        )

    request = DecodeRequest(
        request_id=f"{INTERVENTION_UNIT_ID}:{ARM_ID}:{image_id}",
        chat_text=prompt_record.chat_text,
        input_prompt_token_ids=tuple(prompt_record.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(
            prompt_record.expected_executed_prompt_token_ids
        ),
        image_path=image_plan.image_path,
        declared_image_width=image_plan.declared_width,
        declared_image_height=image_plan.declared_height,
        decoded_image_width=image_plan.decoded_width,
        decoded_image_height=image_plan.decoded_height,
        image_sha256=image_plan.image_content_sha256,
        expected_image_grid_thw=(grid[0], grid[1], grid[2]),
        logical_transform_id=image_plan.logical_transform_id,
        generation_policy=GenerationPolicy(
            max_new_tokens=1,
            repetition_penalty=planner.NATIVE_REPETITION_PENALTY_STRATUM,
            temperature=0.0,
            top_p=1.0,
            include_raw_model_logprob=True,
        ),
    )
    return census.HFSessionSpec(
        image_id=image_id,
        infer_config=Path(infer_config),
        launch=frontend.launch,
        request=request,
        image_grid_thw=(grid[0], grid[1], grid[2]),
        # These are the *overlaid* plan values, so the predecessor session gate
        # verifies the materialized session against the treatment identity.
        planned_prompt_token_ids=[
            int(value) for value in bundle.plan.images[image_id]["prompt_token_ids"]
        ],
        planned_executed_media_sha256=str(
            bundle.plan.images[image_id]["executed_media_sha256"]
        ),
        repetition_penalty_stratum=float(planner.NATIVE_REPETITION_PENALTY_STRATUM),
    )


@contextlib.contextmanager
def open_treatment_backend(spec: census.HFSessionSpec, *, session_opener: Any = None):
    """Open the real treatment session through the predecessor's session gate."""

    with census.open_hf_backend(spec, session_opener=session_opener) as backend:
        yield backend


# ---------------------------------------------------------------------------
# Arm / overlay mixing
# ---------------------------------------------------------------------------


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def assert_no_arm_mixing(shard_root: Path, bundle: InterventionPlan) -> dict[str, Any]:
    """Refuse to add this capture to a root that already holds another arm.

    Pooling two presentation arms under one shard root is the failure this
    intervention is least able to detect downstream: the rows look identical
    apart from digests nobody eyeballs.  So it is refused here, at write time.
    """

    shard_root = Path(shard_root)
    inspected: list[str] = []
    if not shard_root.is_dir():
        return {"inspected_shard_receipts": inspected, "mixing_detected": False}
    for receipt_path in sorted(shard_root.glob(f"*/{census.RECEIPT_NAME}")):
        receipt = _read_json_if_present(receipt_path)
        if receipt is None:
            continue
        stamp = receipt.get(ROW_STAMP_KEY)
        if not isinstance(stamp, Mapping):
            raise ShardContractError(
                f"{receipt_path} carries no intervention stamp; a predecessor 1x capture and "
                "a treatment capture must never share a shard root"
            )
        inspected.append(str(receipt_path))
        if str(stamp.get("arm_id")) != bundle.arm_id:
            raise ShardContractError(
                f"{receipt_path} was captured under arm {stamp.get('arm_id')!r}; this "
                f"invocation is arm {bundle.arm_id!r}. Arms are never pooled in one root"
            )
        if str(stamp.get("overlay_content_sha256")) != bundle.overlay_content_sha256:
            raise ShardContractError(
                f"{receipt_path} was captured under a different sealed overlay; refusing to "
                "mix overlays in one shard root"
            )
        if str(stamp.get("base_plan_receipt_content_sha256")) != (
            bundle.base_plan_receipt_content_sha256
        ):
            raise ShardContractError(
                f"{receipt_path} was captured against a different predecessor plan"
            )
    return {"inspected_shard_receipts": inspected, "mixing_detected": False}


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------


def intervention_stamp(bundle: InterventionPlan, image_id: str) -> dict[str, Any]:
    """The identity every decision-bearing row and the receipt must carry."""

    block = (bundle.overlay["images"] or {})[str(image_id)]
    treatment = block["treatment"]
    current = block["current"]
    return {
        "intervention_unit_id": INTERVENTION_UNIT_ID,
        "arm_id": bundle.arm_id,
        "baseline_arm_id": str(bundle.overlay["baseline_arm_id"]),
        "overlay_content_sha256": bundle.overlay_content_sha256,
        "base_plan_receipt_content_sha256": bundle.base_plan_receipt_content_sha256,
        "max_pixels": int(bundle.overlay["max_pixels"]),
        "image_grid_thw": list(treatment["image_grid_thw"]),
        "merged_visual_tokens": int(treatment["merged_visual_tokens"]),
        "baseline_merged_visual_tokens": int(current["merged_visual_tokens"]),
        "executed_media_sha256": str(treatment["executed_media_sha256"]),
        "media_width": int(treatment["width"]),
        "media_height": int(treatment["height"]),
        "prompt_token_ids_sha256": str(treatment["prompt_token_ids_sha256"]),
        "prompt_token_count": int(treatment["prompt_token_count"]),
        "baseline_prompt_token_ids_sha256": str(current["prompt_token_ids_sha256"]),
        "cross_arm_raw_logprob_comparison": "forbidden",
    }


def build_completeness_block(
    result: census.ShardResult, *, frozen_selection: Sequence[str]
) -> dict[str, Any]:
    """The intervention-relative completeness authority for one shard.

    Why this exists.  The successor deliberately executes a *subset* of the
    predecessor base plan -- that subset is the overlay's frozen selection --
    so ``census.run_shard`` correctly stamps the base-plan-relative
    ``capture_completeness = "subset_smoke"`` even for a complete intervention
    capture.  Those base fields are preserved verbatim as provenance and are
    never relabelled; they answer "how much of the base plan ran?", which is not
    the question this unit's analysis asks.

    This block answers the unit's question instead: did this shard execute the
    *complete frozen overlay selection* for its image?  It is derived from the
    query group IDs that actually carry score rows, never from what was
    requested, so a group that silently produced nothing cannot read as
    complete.
    """

    frozen = {str(value) for value in frozen_selection}
    executed = {str(row["query_group_id"]) for row in result.scores}
    missing = sorted(frozen - executed)
    extra = sorted(executed - frozen)

    # Scoring outside the sealed selection is never legitimate, in either mode.
    if extra:
        raise ShardContractError(
            f"this shard scored {len(extra)} query groups outside the sealed overlay "
            f"selection (first: {extra[:3]!r}); the frozen selection is the only "
            "population this arm may execute"
        )

    status = COMPLETENESS_SUBSET if missing else COMPLETENESS_COMPLETE
    return {
        "relative_to": "overlay_frozen_selection_for_this_image",
        "status": status,
        "is_complete_frozen_overlay_selection": status == COMPLETENESS_COMPLETE,
        "frozen_expected_query_group_count": len(frozen),
        "executed_query_group_count": len(executed),
        "missing_query_group_ids": missing,
        "extra_query_group_ids": extra,
        "derived_from": "query_group_ids_carrying_score_rows",
        "base_plan_fields_role": (
            "capture_completeness and subset_capture describe coverage of the "
            "predecessor base plan and are preserved as provenance; this unit's "
            "analysis reads intervention_completeness instead"
        ),
    }


def stamp_result(
    result: census.ShardResult,
    *,
    stamp: Mapping[str, Any],
    selected: Sequence[str],
    frozen_selection: Sequence[str],
) -> census.ShardResult:
    """Stamp every decision-bearing row and the receipt, before any write.

    ``selected`` is what this invocation asked for; ``frozen_selection`` is the
    overlay's complete frozen selection for this image.  They differ only under
    ``--smoke-query-groups``.
    """

    payload = dict(stamp)
    for rows in (result.scores, result.x1, result.proposals):
        for row in rows:
            if ROW_STAMP_KEY in row:
                raise ShardContractError(
                    "a scored row already carries an intervention stamp; refusing to "
                    "restamp a row of unknown provenance"
                )
            row[ROW_STAMP_KEY] = dict(payload)
    # No free-decode row may exist at all.  The published
    # ``free-decode-sidecars.jsonl`` is still part of the canonical file set and
    # will be written empty; what is forbidden is a *row*.
    if result.free_decodes:
        raise ShardContractError(
            "this successor is score only, but the capture produced behavior sidecars"
        )

    receipt = result.receipt
    receipt[ROW_STAMP_KEY] = dict(payload)
    receipt["intervention_schema_version"] = RECEIPT_SCHEMA_VERSION
    receipt["intervention_source_sha256"] = EXECUTED_SOURCE_SHA256
    receipt["intervention_selection"] = {
        "selected_query_group_ids": sorted(str(value) for value in selected),
        "selected_query_group_count": len(set(str(value) for value in selected)),
        "selection_is_frozen_in_the_overlay": True,
        "selection_uses_treatment_scores": False,
    }
    receipt[COMPLETENESS_KEY] = build_completeness_block(
        result, frozen_selection=frozen_selection
    )
    # These fields, together with ``phase_order.behavior_sidecars_intentionally_
    # disabled`` and ``counts.free_decode_sidecar_rows``, are the authority on
    # what this capture contains.  The presence of a zero-byte
    # ``free-decode-sidecars.jsonl`` on disk is a property of the canonical
    # schema-complete file set and carries no capture meaning of its own.
    receipt["intervention_capture_mode"] = {
        "score_only": True,
        "behavior_sidecars_captured": False,
        "behavior_sidecars_intentionally_disabled": True,
        "reason": "score_only_successor_no_free_greedy_box_or_row_sidecars",
    }
    return result


# ---------------------------------------------------------------------------
# Shard execution
# ---------------------------------------------------------------------------


def run_intervention_shard(
    bundle: InterventionPlan,
    *,
    image_id: str,
    backend: census.CensusBackend,
    output_dir: Path | None,
    top_k: int = census.DEFAULT_TOP_K,
    candidate_batch_size: int = census.DEFAULT_CANDIDATE_BATCH_SIZE,
    max_query_groups: int | None = None,
) -> census.ShardResult:
    """Execute the overlay's selection for one image and publish it atomically.

    Publishing goes through the canonical :func:`census.write_shard`, which
    renames the *complete* schema file set into place in one step.  A score-only
    capture therefore lands a zero-byte ``free-decode-sidecars.jsonl`` alongside
    the score, x1 and proposal files; that is expected, and the receipt fields
    are what state the capture mode.
    """

    image_id = str(image_id)
    frozen_selection = bundle.selected_query_group_ids(image_id)
    selected = list(frozen_selection)
    if max_query_groups is not None:
        selected = selected[: int(max_query_groups)]
        if not selected:
            raise ShardContractError("--smoke-query-groups selected no group")

    started = time.time()
    result = census.run_shard(
        bundle.plan,
        image_id=image_id,
        backend=backend,
        output_dir=None,
        top_k=top_k,
        query_group_ids=selected,
        candidate_batch_size=candidate_batch_size,
        capture_behavior_sidecars=False,
    )
    stamp_result(
        result,
        stamp=intervention_stamp(bundle, image_id),
        selected=selected,
        frozen_selection=frozen_selection,
    )
    completeness = result.receipt[COMPLETENESS_KEY]
    if max_query_groups is None and completeness["status"] != COMPLETENESS_COMPLETE:
        raise ShardContractError(
            f"a full run of image {image_id!r} did not execute the complete frozen overlay "
            f"selection: {len(completeness['missing_query_group_ids'])} groups are missing "
            f"(first: {completeness['missing_query_group_ids'][:3]!r})"
        )
    result.receipt["intervention_elapsed_seconds"] = time.time() - started

    if output_dir is not None:
        output_dir = Path(output_dir)
        assert_no_arm_mixing(output_dir.parent, bundle)
        census.write_shard(output_dir, result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-plan-dir", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("hf", "fake"), default="fake")
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--top-k", type=int, default=census.DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-batch-size", type=int, default=census.DEFAULT_CANDIDATE_BATCH_SIZE
    )
    parser.add_argument(
        "--smoke-query-groups",
        type=int,
        default=None,
        help=(
            "execute only the first N selected groups of this shard, in sealed overlay "
            "order; a real-model smoke subset that never modifies the overlay.  The "
            f"receipt then publishes {COMPLETENESS_KEY}.status = {COMPLETENESS_SUBSET!r} "
            f"instead of {COMPLETENESS_COMPLETE!r}, and the analyzer refuses such a shard"
        ),
    )
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="CPU-only overlay/selection check; loads no model and writes nothing",
    )
    return parser.parse_args(argv)


def validate_contract(bundle: InterventionPlan, image_id: str) -> dict[str, Any]:
    """CPU-only proof that this image's selection resolves under the overlay."""

    image_id = str(image_id)
    selected = bundle.selected_query_group_ids(image_id)
    items = [census.resolve_work_item(bundle.plan, group_id) for group_id in selected]
    block = (bundle.overlay["images"] or {})[image_id]
    return {
        "schema_version": "sorted-full-canvas-visual-token-budget-intervention-contract-check.v1",
        "intervention_unit_id": INTERVENTION_UNIT_ID,
        "arm_id": bundle.arm_id,
        "overlay_content_sha256": bundle.overlay_content_sha256,
        "base_plan_receipt_content_sha256": bundle.base_plan_receipt_content_sha256,
        "image_id": image_id,
        "selected_query_group_count": len(selected),
        "distinct_context_count": len({item.context_id for item in items}),
        "distinct_category_count": len({item.category_query_id for item in items}),
        "candidate_score_row_count": sum(len(item.candidates) for item in items),
        "treatment_merged_visual_tokens": int(block["treatment"]["merged_visual_tokens"]),
        "baseline_merged_visual_tokens": int(block["current"]["merged_visual_tokens"]),
        "treatment_prompt_token_count": int(block["treatment"]["prompt_token_count"]),
        "baseline_prompt_token_count": int(block["current"]["prompt_token_count"]),
        "every_group_reconstructs_its_treatment_prefix": True,
        "score_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    image_id = str(args.image_id)
    try:
        bundle = load_intervention_plan(args.base_plan_dir, args.overlay)
    except ShardContractError as exc:
        print(f"plan/overlay error: {exc}", file=sys.stderr)
        return 2

    if args.validate_contract_only:
        try:
            report = validate_contract(bundle, image_id)
        except ShardContractError as exc:
            print(f"contract error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        print("--output-dir is required unless --validate-contract-only", file=sys.stderr)
        return 2

    kwargs = {
        "image_id": image_id,
        "output_dir": args.output_dir,
        "top_k": args.top_k,
        "candidate_batch_size": args.candidate_batch_size,
        "max_query_groups": args.smoke_query_groups,
    }
    try:
        if args.backend == "fake":
            result = run_intervention_shard(
                bundle, backend=census.FakeCensusBackend(), **kwargs
            )
        else:
            spec = build_treatment_session_spec(
                bundle, image_id, infer_config=args.infer_config
            )
            with open_treatment_backend(spec) as backend:
                result = run_intervention_shard(bundle, backend=backend, **kwargs)
    except Exception as exc:  # noqa: BLE001 - shard-local quarantine is the contract
        payload = census.write_quarantine(
            args.output_dir,
            image_id=image_id,
            reason=type(exc).__name__,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        print(f"intervention shard {image_id} quarantined", file=sys.stderr)
        return 3

    print(json.dumps(result.receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
