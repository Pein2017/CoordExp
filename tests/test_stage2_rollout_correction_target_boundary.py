from __future__ import annotations

import inspect
from types import SimpleNamespace

from src.detection.data import DetectionMetadata, ObjectOrderingPlan
from src.detection.scene import DetectionGeometry, DetectionObject, DetectionScene
from src.infer.backend import DetectionDecodeResult
from src.infer.runtime import DetectionDecodeRequest
from src.training.stage2.rollout_codec import (
    Stage2RolloutObject,
    Stage2RolloutParseResult,
    Stage2RolloutTemplatePolicy,
)
from src.trainers.rollout_correction.rollout_views import build_rollout_correction_view
from src.trainers.rollout_correction.target_builder import (
    RolloutCorrectionTargetContext,
    RolloutCorrectionTargetContextInput,
    construct_rollout_correction_target_context,
    construct_detection_scene_rollout_correction_target_context,
)
from src.trainers.rollout_correction.projections import (
    CorrectionEvent,
    annotate_correction_events_with_projection_provenance,
    assign_detection_scene_rollout_prediction,
    detection_decode_result_from_stage2_rollout,
    filter_rollout_prediction_duplicates,
    rollout_prediction_from_legacy_stage2_parse,
    rollout_prediction_from_shared_decode,
)
from src.trainers.rollout_matching.contracts import (
    GTObject,
    ParsedPredObject,
    RolloutParseResult,
)
from src.trainers.stage2_rollout_correction_impl import (
    _stage2_construct_detection_scene_target_state,
    _stage2_detection_scene_from_sample,
    _stage2_ul_rollout_evidence,
    _stage2_rollout_decode_provenance_from_request,
)


_REQUIRED_DECODE_PROVENANCE = {
    "model_identity": "unit-model",
    "model_identity_fingerprint": "model-fp",
    "checkpoint_identity": "ckpt-fp",
    "prompt_policy": "stage2-test-prompt",
    "prompt_policy_fingerprint": "prompt-fp",
    "decode_policy": "unconstrained",
    "decode_policy_fingerprint": "decode-fp",
    "metric_eligibility": True,
}


def _bbox_object(index: int, desc: str, box: list[int]) -> GTObject:
    return GTObject(
        index=int(index),
        geom_type="bbox_2d",
        points_norm1000=list(box),
        desc=str(desc),
    )


def _scene_object(index: int, desc: str, box: list[int]) -> DetectionObject:
    return DetectionObject(
        scene_object_index=int(index),
        source_object_index=int(index),
        object_instance_id=f"scene-object-{index}",
        label=str(desc),
        desc=str(desc),
        geometry=DetectionGeometry.from_bbox_2d(
            tuple(int(v) for v in box),
            coordinate_frame="image",
            coordinate_space="norm1000",
            bbox_chart="xyxy",
        ),
        category_id=int(index) + 1,
        category_name=str(desc),
        coco_ann_id=1000 + int(index),
    )


def _scene(*objects: DetectionObject) -> DetectionScene:
    return DetectionScene(
        image_id=77,
        image_reference="/tmp/stage2-scene.jpg",
        source_image_reference="stage2-scene.jpg",
        file_name="stage2-scene.jpg",
        width=1000,
        height=1000,
        coordinate_frame="image",
        coordinate_space="norm1000",
        bbox_chart="xyxy",
        objects=tuple(objects),
        object_ordering=ObjectOrderingPlan.sorted().with_realized(
            tuple(obj.source_object_index for obj in objects)
        ),
        metadata=DetectionMetadata(source="unit", split="train"),
    )


def _decoded(text: str) -> DetectionDecodeResult:
    return DetectionDecodeResult(
        text=str(text),
        generated_token_ids=[11, 12, 13],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=[-0.1, -0.2, -0.3],
        stop_reason="stop",
        backend="unit-runtime",
        backend_metadata=dict(_REQUIRED_DECODE_PROVENANCE),
        prompt_token_ids=[1, 2, 3],
    )


def _parse_result(
    text: str,
    *objects: Stage2RolloutObject,
    invalid: bool = False,
    dropped_invalid: int = 0,
) -> Stage2RolloutParseResult:
    return Stage2RolloutParseResult(
        template_family="compact_full",
        parser_id="compact_full",
        response_text=str(text),
        valid_objects=tuple(objects),
        invalid_rollout=bool(invalid),
        empty_valid_object_set=not objects and not invalid,
        truncated=False,
        fallback_reason="malformed_compact_full" if invalid else None,
        metadata={"rollout_parser_id": "compact_full", "parser_policy": "strict"},
        response_token_ids=(11, 12, 13),
        dropped_invalid=int(dropped_invalid),
        dropped_invalid_by_reason={"malformed_row": int(dropped_invalid)}
        if dropped_invalid
        else {},
    )


def _rollout_object(index: int, desc: str, box: list[int]) -> Stage2RolloutObject:
    return Stage2RolloutObject(
        object_id=f"pred-{index}",
        index=int(index),
        desc=str(desc),
        bbox_norm1000=tuple(int(v) for v in box),
        provenance="shared_decode_parse",
    )


class _PieceTokenizer:
    eos_token_id = 99

    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = dict(pieces)

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        _ = skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)


def _compact_view_text_and_tokenizer() -> tuple[str, list[int], _PieceTokenizer]:
    pieces = {
        1: "<|object_ref_start|>",
        2: "cat",
        3: "<|box_start|>",
        4: "<|coord_100|>",
        5: "<|coord_100|>",
        6: "<|coord_200|>",
        7: "<|coord_200|>",
    }
    token_ids = [1, 2, 3, 4, 5, 6, 7]
    return "".join(pieces[token_id] for token_id in token_ids), token_ids, _PieceTokenizer(pieces)


def _decode_request() -> DetectionDecodeRequest:
    return DetectionDecodeRequest(
        backend="hf",
        backend_mode="local",
        decode_mode="greedy",
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        num_beams=1,
        repetition_penalty=1.0,
        stop_strings=("<|im_end|>",),
        decode_policy_fingerprint="decode:unit-request-fp",
    )


def _prompt_sample() -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "detect"}]},
        ],
        "_coordexp_prompt_visual_metadata": {
            "image_count": 1,
            "do_resize": False,
        },
    }


def _owner_with_checkpoint(path: str = "/models/unit-checkpoint") -> SimpleNamespace:
    return SimpleNamespace(args=SimpleNamespace(model_name_or_path=path))


def test_rollout_correction_target_context_uses_parsed_rollout_facts_only() -> None:
    matched_gt = _bbox_object(0, "matched", [100, 100, 200, 200])
    unmatched_anchor = _bbox_object(1, "unlabeled", [500, 500, 600, 600])
    explorer_support = _bbox_object(0, "unlabeled", [502, 502, 602, 602])

    context = construct_rollout_correction_target_context(
        RolloutCorrectionTargetContextInput(
            sample_id="sample-target-boundary",
            gt_objects=[matched_gt],
            accepted_objects_clean=[matched_gt, unmatched_anchor],
            suppressed_duplicate_objects_by_boundary={},
            explorer_objects_raw_by_view=[[explorer_support]],
            anchor_match_by_pred={0: 0},
            explorer_match_by_pred_by_view=[{}],
            anchor_policy_statuses=[],
            unlabeled_consistent_iou_threshold=0.5,
            duplicate_iou_threshold=0.9,
            pseudo_positive_enabled=True,
            expected_peer_count=1,
        )
    )

    assert isinstance(context, RolloutCorrectionTargetContext)
    assert context.sample_id == "sample-target-boundary"
    assert context.metrics["gt_objects"] == 1.0
    assert context.metrics["accepted_objects"] == 2.0
    assert context.metrics["anchor_gt_backed"] == 1.0
    assert context.metrics["valid_explorer_count"] == 1.0
    assert context.triage.anchor_gt_backed_indices == [0]
    assert context.triage.anchor_support_counts[1] == 1
    assert context.triage.association_pairs_by_view == [[(1, 0)]]


def test_rollout_correction_target_context_boundary_has_no_lifecycle_inputs() -> None:
    signature = inspect.signature(construct_rollout_correction_target_context)

    assert list(signature.parameters) == ["request"]
    request_fields = set(RolloutCorrectionTargetContextInput.__dataclass_fields__)
    forbidden_fields = {
        "owner",
        "trainer",
        "model",
        "vllm",
        "ddp",
        "barrier",
        "compute_loss",
        "rollout_many",
        "training_step",
    }
    assert request_fields.isdisjoint(forbidden_fields)

    source = inspect.getsource(construct_rollout_correction_target_context)
    for forbidden in (
        "_rollout_many",
        "vllm",
        "barrier",
        "compute_loss",
        "training_step",
    ):
        assert forbidden not in source


def test_rollout_prediction_derives_from_shared_decode_and_strict_parse() -> None:
    text = "<|object_ref_start|> cat<|box_start|><|coord_100|><|coord_100|><|coord_200|><|coord_200|>"
    decoded = _decoded(text)
    parse = _parse_result(
        text,
        _rollout_object(0, "cat", [100, 100, 200, 200]),
        dropped_invalid=1,
    )

    prediction = rollout_prediction_from_shared_decode(
        decoded_result=decoded,
        parse_result=parse,
        metric_bearing=True,
        source_label="anchor",
    )

    assert prediction.decoded_result is decoded
    assert prediction.parse_result is parse
    assert prediction.metric_bearing is True
    assert prediction.valid_objects[0].points_norm1000 == [100, 100, 200, 200]
    assert prediction.invalid_drop_metadata == {
        "invalid_rollout": False,
        "empty_valid_object_set": False,
        "truncated": False,
        "fallback_reason": None,
        "dropped_invalid": 1,
        "dropped_ambiguous": 0,
        "dropped_invalid_by_reason": {"malformed_row": 1},
    }
    assert prediction.provenance["decode_result_type"] == "DetectionDecodeResult"
    assert prediction.provenance["backend"] == "unit-runtime"
    assert prediction.provenance["parser_id"] == "compact_full"
    assert prediction.provenance["metric_bearing"] is True
    assert (
        prediction.provenance["backend_metadata"]["model_identity_fingerprint"]
        == "model-fp"
    )


def test_detection_scene_assignment_duplicate_filter_and_context_projection() -> None:
    scene = _scene(_scene_object(0, "cat", [100, 100, 200, 200]))
    prediction = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("duplicate cat rollout"),
        parse_result=_parse_result(
            "duplicate cat rollout",
            _rollout_object(0, "cat", [100, 100, 200, 200]),
            _rollout_object(1, "cat", [102, 102, 202, 202]),
        ),
        metric_bearing=True,
        source_label="anchor",
    )
    explorer = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("support cat rollout"),
        parse_result=_parse_result(
            "support cat rollout",
            _rollout_object(0, "cat", [101, 101, 201, 201]),
        ),
        metric_bearing=True,
        source_label="peer",
    )

    duplicate_filter = filter_rollout_prediction_duplicates(
        prediction=prediction,
        explorer_predictions=[explorer],
        duplicate_iou_threshold=0.9,
        center_radius_scale=0.8,
        unlabeled_consistent_iou_threshold=0.5,
    )
    assignment = assign_detection_scene_rollout_prediction(
        scene=scene,
        prediction=duplicate_filter.prediction,
        min_iou=0.5,
    )
    context = construct_detection_scene_rollout_correction_target_context(
        scene=scene,
        rollout_prediction=prediction,
        explorer_predictions=[explorer],
        unlabeled_consistent_iou_threshold=0.5,
        duplicate_iou_threshold=0.9,
        center_radius_scale=0.8,
        pseudo_positive_enabled=True,
        expected_peer_count=1,
    )

    assert duplicate_filter.prediction.valid_objects[0].desc == "cat"
    assert (
        duplicate_filter.suppressed_duplicate_objects_by_boundary[1][0].desc
        == "cat"
    )
    assert assignment.matched_pairs == ((0, 0),)
    assert assignment.prediction_source == "shared_inference_runtime_decode"
    assert context.detection_scene is scene
    assert context.rollout_prediction is prediction
    assert context.assignment.matched_pairs == ((0, 0),)
    assert context.triage.anchor_gt_backed_indices == [0]


def test_correction_event_provenance_links_scene_prediction_and_assignment() -> None:
    scene = _scene(_scene_object(0, "cat", [100, 100, 200, 200]))
    prediction = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("cat rollout"),
        parse_result=_parse_result(
            "cat rollout",
            _rollout_object(0, "cat", [100, 100, 200, 200]),
        ),
        metric_bearing=True,
        source_label="anchor",
    )
    assignment = assign_detection_scene_rollout_prediction(
        scene=scene,
        prediction=prediction,
        min_iou=0.5,
    )
    event = CorrectionEvent(
        correction_kind="residual_continuation",
        sample_id="scene-77",
        atom_drafts=(),
        metadata={"correction_builder": "unit"},
    )

    (annotated,) = annotate_correction_events_with_projection_provenance(
        [event],
        scene=scene,
        rollout_prediction=prediction,
        assignment=assignment,
    )

    assert annotated.metadata["correction_builder"] == "unit"
    assert annotated.metadata["detection_scene"]["image_id"] == 77
    assert annotated.metadata["rollout_prediction"]["parser_id"] == "compact_full"
    assert annotated.metadata["rollout_prediction"]["metric_bearing"] is True
    assert annotated.metadata["rollout_prediction"]["prediction_id"]
    assert annotated.metadata["rollout_prediction"]["invalid_drop_metadata"] == dict(
        prediction.invalid_drop_metadata
    )
    assert (
        annotated.metadata["rollout_prediction"]["model_identity_fingerprint"]
        == "model-fp"
    )
    assert annotated.metadata["rollout_prediction"]["checkpoint_identity"] == "ckpt-fp"
    assert (
        annotated.metadata["rollout_prediction"]["prompt_policy_fingerprint"]
        == "prompt-fp"
    )
    assert (
        annotated.metadata["rollout_prediction"]["decode_policy_fingerprint"]
        == "decode-fp"
    )
    assert annotated.metadata["rollout_prediction"]["parser_policy"] == "strict"
    assert annotated.metadata["rollout_prediction"]["backend_metadata"][
        "metric_eligibility"
    ] is True
    assert annotated.metadata["rollout_prediction"]["parser_metadata"][
        "rollout_parser_id"
    ] == "compact_full"
    assert annotated.metadata["rollout_prediction"]["metric_eligibility"] is True
    assert annotated.metadata["detection_assignment"]["assignment_id"]
    assert annotated.metadata["detection_assignment"]["matched_pairs"] == [(0, 0)]


def test_metric_bearing_rollout_prediction_fails_closed_without_decode_provenance() -> None:
    text = "cat rollout"
    decoded = DetectionDecodeResult(
        text=text,
        generated_token_ids=[11],
        generated_tokens=None,
        generated_logprobs=None,
        stop_reason="stop",
        backend="unit-runtime",
        backend_metadata={"model_identity_fingerprint": "model-only"},
        prompt_token_ids=[1],
    )

    try:
        rollout_prediction_from_shared_decode(
            decoded_result=decoded,
            parse_result=_parse_result(
                text,
                _rollout_object(0, "cat", [100, 100, 200, 200]),
            ),
            metric_bearing=True,
            source_label="anchor",
        )
    except ValueError as exc:
        assert "requires shared decode provenance" in str(exc)
    else:  # pragma: no cover - defensive guard for direct invocation
        raise AssertionError("metric-bearing prediction accepted missing provenance")


def test_metric_bearing_rollout_prediction_fails_closed_without_parser_policy() -> None:
    text = "cat rollout"

    try:
        rollout_prediction_from_shared_decode(
            decoded_result=_decoded(text),
            parse_result=Stage2RolloutParseResult(
                template_family="compact_full",
                parser_id="compact_full",
                response_text=text,
                valid_objects=(
                    _rollout_object(0, "cat", [100, 100, 200, 200]),
                ),
                invalid_rollout=False,
                empty_valid_object_set=False,
                truncated=False,
                fallback_reason=None,
                metadata={"rollout_parser_id": "compact_full"},
                response_token_ids=(11, 12, 13),
            ),
            metric_bearing=True,
            source_label="anchor",
        )
    except ValueError as exc:
        assert "requires parser provenance" in str(exc)
        assert "parser_policy" in str(exc)
    else:  # pragma: no cover - defensive guard for direct invocation
        raise AssertionError("metric-bearing prediction accepted missing parser policy")


def test_metric_bearing_rollout_prediction_rejects_parser_migration_marker() -> None:
    text = "cat rollout"

    try:
        rollout_prediction_from_shared_decode(
            decoded_result=_decoded(text),
            parse_result=Stage2RolloutParseResult(
                template_family="compact_full",
                parser_id="compact_full",
                response_text=text,
                valid_objects=(
                    _rollout_object(0, "cat", [100, 100, 200, 200]),
                ),
                invalid_rollout=False,
                empty_valid_object_set=False,
                truncated=False,
                fallback_reason=None,
                metadata={
                    "rollout_parser_id": "compact_full",
                    "parser_policy": "strict",
                    "migration_only": True,
                },
                response_token_ids=(11, 12, 13),
            ),
            metric_bearing=True,
            source_label="anchor",
        )
    except ValueError as exc:
        assert "diagnostic/private parser metadata" in str(exc)
        assert "migration_only" in str(exc)
    else:  # pragma: no cover - defensive guard for direct invocation
        raise AssertionError("metric-bearing prediction accepted parser migration marker")


def test_metric_bearing_rollout_prediction_rejects_backend_private_parser_marker() -> None:
    text = "cat rollout"
    decoded = DetectionDecodeResult(
        text=text,
        generated_token_ids=[11],
        generated_tokens=None,
        generated_logprobs=None,
        stop_reason="stop",
        backend="unit-runtime",
        backend_metadata={
            **_REQUIRED_DECODE_PROVENANCE,
            "diagnostic_private_parser": True,
            "migration_source": "rollout_matching",
        },
        prompt_token_ids=[1],
    )

    try:
        rollout_prediction_from_shared_decode(
            decoded_result=decoded,
            parse_result=_parse_result(
                text,
                _rollout_object(0, "cat", [100, 100, 200, 200]),
            ),
            metric_bearing=True,
            source_label="anchor",
        )
    except ValueError as exc:
        assert "diagnostic/private parser decode provenance" in str(exc)
    else:  # pragma: no cover - defensive guard for direct invocation
        raise AssertionError("metric-bearing prediction accepted private parser marker")


def test_production_decode_provenance_requires_typed_request_and_checkpoint() -> None:
    policy = Stage2RolloutTemplatePolicy(
        template_family="compact_full",
        parser_id="compact_full",
        append_policy_id="compact_full_fn_append",
        decode_policy="unconstrained",
        invalid_rollout_policy="fallback_gt_fn_append_only",
    )
    provenance = _stage2_rollout_decode_provenance_from_request(
        owner=_owner_with_checkpoint(),
        request=_decode_request(),
        rollout_template_policy=policy,
        prompt_sample=_prompt_sample(),
    )

    assert provenance["decode_request_type"] == "DetectionDecodeRequest"
    assert provenance["checkpoint_identity"] == "/models/unit-checkpoint"
    assert provenance["model_identity"] == "/models/unit-checkpoint"
    assert str(provenance["model_identity_fingerprint"]).startswith("model:")
    assert provenance["decode_policy_fingerprint"] == "decode:unit-request-fp"
    assert str(provenance["prompt_policy_fingerprint"]).startswith("prompt_policy:v1:")
    assert provenance["metric_eligibility"] is True


def test_production_decode_provenance_rejects_local_fallback_identity() -> None:
    policy = Stage2RolloutTemplatePolicy(
        template_family="compact_full",
        parser_id="compact_full",
        append_policy_id="compact_full_fn_append",
        decode_policy="unconstrained",
        invalid_rollout_policy="fallback_gt_fn_append_only",
    )

    try:
        _stage2_rollout_decode_provenance_from_request(
            owner=SimpleNamespace(args=SimpleNamespace(model_name_or_path="unset")),
            request=_decode_request(),
            rollout_template_policy=policy,
            prompt_sample=_prompt_sample(),
        )
    except ValueError as exc:
        assert "requires resolved checkpoint_identity" in str(exc)
    else:  # pragma: no cover - defensive guard for direct invocation
        raise AssertionError("provenance accepted local fallback identity")


def test_legacy_coordjson_bridge_is_diagnostic_non_metric_even_when_requested() -> None:
    text = "legacy coordjson"
    decoded = detection_decode_result_from_stage2_rollout(
        response_text=text,
        response_token_ids=[11, 12],
        prompt_token_ids=[1, 2],
        decode_mode="legacy_coordjson",
        source_label="legacy",
        backend_metadata={
            **_REQUIRED_DECODE_PROVENANCE,
            "diagnostic_private_parser": True,
            "metric_eligibility": False,
        },
    )
    parse = RolloutParseResult(
        response_token_ids=[11, 12],
        response_text=text,
        prefix_token_ids=[11, 12],
        prefix_text=text,
        invalid_rollout=False,
        valid_objects=[
            ParsedPredObject(
                key="obj0",
                index=0,
                desc="cat",
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                value_span=(0, len(text)),
            )
        ],
        dropped_invalid=0,
    )

    prediction = rollout_prediction_from_legacy_stage2_parse(
        decoded_result=decoded,
        parse_result=parse,
        valid_objects=[_bbox_object(0, "cat", [100, 100, 200, 200])],
        metric_bearing=True,
        source_label="legacy",
    )

    assert prediction.metric_bearing is False
    assert prediction.provenance["parser_metadata"]["migration_only"] is True
    assert prediction.provenance["parser_metadata"]["diagnostic_private_parser"] is True


def test_compact_production_view_threads_required_metric_provenance() -> None:
    text, token_ids, tokenizer = _compact_view_text_and_tokenizer()
    policy = Stage2RolloutTemplatePolicy(
        template_family="compact_full",
        parser_id="compact_full",
        append_policy_id="compact_full_fn_append",
        decode_policy="unconstrained",
        invalid_rollout_policy="fallback_gt_fn_append_only",
    )
    decode_provenance = _stage2_rollout_decode_provenance_from_request(
        owner=_owner_with_checkpoint(),
        request=_decode_request(),
        rollout_template_policy=policy,
        prompt_sample=_prompt_sample(),
    )

    view = build_rollout_correction_view(
        tokenizer=tokenizer,
        object_field_order="desc_first",
        coord_id_to_bin={100: 100, 200: 200},
        duplicate_iou_threshold=0.9,
        center_radius_scale=0.8,
        max_new_tokens=32,
        rollout_result=(token_ids, text, "greedy", [101, 102]),
        source_label="anchor",
        parse_rollout_for_matching_fn=None,
        points_from_coord_tokens_fn=None,
        duplicate_diagnostics_fn=lambda _objects, **_kwargs: {},
        rollout_template_policy=policy,
        decode_provenance=decode_provenance,
    )

    prediction = view["rollout_prediction"]
    assert prediction.metric_bearing is True
    assert (
        prediction.provenance["backend_metadata"]["checkpoint_identity"]
        == "/models/unit-checkpoint"
    )
    assert prediction.provenance["parser_metadata"]["rollout_parser_id"] == "compact_full"
    assert view["decoded_result"].backend_metadata["metric_eligibility"] is True


def test_production_target_state_uses_scene_projection_not_raw_precontext_state() -> None:
    sample = {
        "image_id": 77,
        "images": ["/tmp/stage2-scene.jpg"],
        "assistant_payload": {
            "objects": [
                {
                    "desc": "cat",
                    "bbox_2d": [100, 100, 200, 200],
                }
            ]
        },
    }
    scene = _stage2_detection_scene_from_sample(sample)
    prediction = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("cat rollout"),
        parse_result=_parse_result(
            "cat rollout",
            _rollout_object(0, "cat", [100, 100, 200, 200]),
            _rollout_object(1, "cat", [102, 102, 202, 202]),
        ),
        metric_bearing=True,
        source_label="anchor",
    )
    explorer = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("support cat rollout"),
        parse_result=_parse_result(
            "support cat rollout",
            _rollout_object(0, "cat", [101, 101, 201, 201]),
        ),
        metric_bearing=True,
        source_label="peer",
    )

    target_state = _stage2_construct_detection_scene_target_state(
        sample=sample,
        detection_scene=scene,
        rollout_prediction=prediction,
        explorer_predictions=[explorer],
        unlabeled_consistent_iou_threshold=0.5,
        duplicate_iou_threshold=0.9,
        center_radius_scale=0.8,
        pseudo_positive_enabled=True,
        expected_peer_count=1,
        assignment_iou_threshold=0.5,
    )
    context = target_state.target_context

    assert context.detection_scene is scene
    assert context.rollout_prediction is prediction
    assert context.assignment is not None
    assert context.duplicate_filter is not None
    assert [obj.desc for obj in target_state.accepted_objects_clean] == ["cat"]
    assert target_state.match.matched_pairs == [(0, 0)]
    assert target_state.suppressed_duplicate_objects_by_boundary[1][0].desc == "cat"
    assert target_state.duplicate_survivor_anchor_indices == (0,)
    assert target_state.duplicate_exempt_anchor_indices == ()
    assert target_state.duplicate_suppressed_anchor_indices == (1,)
    assert target_state.explorer_objects_by_view[0][0].desc == "cat"
    assert target_state.explorer_match_by_pred_by_view[0] == {0: 0}

    helper_signature = inspect.signature(_stage2_construct_detection_scene_target_state)
    forbidden_raw_inputs = {
        "parsed_bbox_objects_raw",
        "accepted_objects_clean",
        "anchor_match_by_pred",
        "explorer_match_by_pred_by_view",
    }
    assert forbidden_raw_inputs.isdisjoint(helper_signature.parameters)


def test_ul_rollout_evidence_uses_rollout_prediction_not_raw_view_objects() -> None:
    prediction = rollout_prediction_from_shared_decode(
        decoded_result=_decoded("canonical cat rollout"),
        parse_result=_parse_result(
            "canonical cat rollout",
            _rollout_object(0, "canonical-cat", [100, 100, 200, 200]),
        ),
        metric_bearing=True,
        source_label="anchor",
    )
    raw_view_object = _bbox_object(99, "raw-view-dog", [700, 700, 800, 800])

    evidence = _stage2_ul_rollout_evidence(
        sample_id="ul-sample",
        view={
            "rollout_index": 3,
            "rollout_counts_as_valid_rollout": 1,
            "parsed_bbox_objects_raw": [raw_view_object],
        },
        rollout_prediction=prediction,
        gts=[],
        assignment_iou_threshold=0.5,
    )

    assert evidence.is_valid is True
    assert [member.desc_text for member in evidence.unmatched_members] == [
        "canonical-cat"
    ]
    assert [member.local_index for member in evidence.unmatched_members] == [0]
    assert all(
        member.desc_text != "raw-view-dog"
        for member in evidence.unmatched_members
    )


def test_ul_rollout_evidence_fails_closed_without_rollout_prediction() -> None:
    evidence = _stage2_ul_rollout_evidence(
        sample_id="ul-sample",
        view={
            "rollout_index": 4,
            "rollout_counts_as_valid_rollout": 1,
            "parsed_bbox_objects_raw": [
                _bbox_object(99, "raw-view-dog", [700, 700, 800, 800])
            ],
        },
        rollout_prediction=None,
        gts=[],
        assignment_iou_threshold=0.5,
    )

    assert evidence.is_valid is False
    assert evidence.unmatched_members == ()
