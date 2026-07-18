from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src.common.errors import ArtifactContractError, RuntimeContractError
from src.inference.backend import DecodeResult, LikelihoodPair, TokenTrace
from src.inference.parsing import parse_compact_object_box_closed


OBJECT_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)


def _trace(
    *,
    logprob: float = math.log(0.25),
    raw_model_logprob: float | None = None,
) -> tuple[TokenTrace, ...]:
    pieces = [
        "<|object_ref_start|>",
        "cat",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
    ]
    return tuple(
        TokenTrace(
            step_index=index,
            token_id=151646 + index,
            token_text=piece,
            likelihood=LikelihoodPair(
                policy_logprob=logprob,
                raw_model_logprob=raw_model_logprob,
            ),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(pieces)
    )


def _repeated_trace() -> tuple[TokenTrace, ...]:
    first = _trace(logprob=math.log(0.5))
    second = tuple(
        replace(
            item,
            step_index=item.step_index + len(first),
            likelihood=LikelihoodPair(
                policy_logprob=math.log(0.25),
                raw_model_logprob=None,
            ),
        )
        for item in _trace()
    )
    return first + second


def _decode_result(
    row_id: str,
    *,
    token_trace: tuple[TokenTrace, ...] | list[TokenTrace] | None = None,
) -> DecodeResult:
    trace = _trace() if token_trace is None else token_trace
    return DecodeResult(
        request_id=row_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        executed_prompt_token_ids=(11, 12),
        generated_token_ids=tuple(item.token_id for item in trace),
        raw_generated_text="".join(item.token_text for item in trace),
        parser_text="".join(item.token_text for item in trace),
        strip_policy="none",
        stop_reason="length",
        token_trace=tuple(trace),
        observed_image_grid_thw=(1, 4, 6),
        executed_media_sha256="a" * 64,
    )


def _raw_row(row_id: str, row_index: int, *, text: str = OBJECT_TEXT) -> dict:
    parse_row = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=row_index,
        image_width=1000,
        image_height=1000,
    )
    return {
        "row_id": row_id,
        "row_index": row_index,
        "example_id": row_id,
        "image_path": f"{row_id}.jpg",
        "image_width": 1000,
        "image_height": 1000,
        "gt": [{"description": "gt-cat", "bbox": [100, 100, 300, 300]}],
        "raw_decode_text": text,
        "parse": parse_row,
    }


def _image_plan_row(row_id: str, row_index: int) -> dict:
    return {"row_id": row_id, "row_index": row_index}


def _metadata() -> dict:
    return {
        "artifact_schema_version": 1,
        "detection_template_id": "compact-object-box-closed",
        "prompt_policy_fingerprint": "prompt-fp",
        "generation_config_fingerprint": "gen-fp",
        "model_identity_fingerprint": "model-fp",
        "processor_identity_fingerprint": "processor-fp",
        "template_identity": {"id": "template-v1"},
        "parser_policy": "compact_object_box_closed_only",
        "dataset_identity": {"name": "unit"},
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_artifact_writer_preserves_raw_and_scored_row_parity_without_diagnostic_rows(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[
            _raw_row("row-1", 0),
            _raw_row("row-2", 1, text="malformed"),
        ],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0), _image_plan_row("row-2", 1)],
        metadata=_metadata(),
    )

    raw_rows = _read_jsonl(paths.raw_jsonl)
    scored_rows = _read_jsonl(paths.scored_jsonl)
    diagnostics_rows = _read_jsonl(paths.parse_diagnostics_jsonl)

    assert [row["row_id"] for row in raw_rows] == ["row-1", "row-2"]
    assert [row["row_id"] for row in scored_rows] == ["row-1", "row-2"]
    assert len(diagnostics_rows) == 2
    assert all(row["row_id"] in {"row-1", "row-2"} for row in diagnostics_rows)


def test_scored_rows_keep_empty_pred_list_and_preserve_gt_and_image_identity(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[
            _raw_row("row-1", 0),
            _raw_row("row-2", 1, text="malformed"),
        ],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0), _image_plan_row("row-2", 1)],
        metadata=_metadata(),
    )

    scored_rows = _read_jsonl(paths.scored_jsonl)

    assert scored_rows[1]["pred"] == []
    for row in scored_rows:
        assert row["gt"] == [{"bbox": [100, 100, 300, 300], "description": "gt-cat"}]
        assert row["image_path"] == f"{row['row_id']}.jpg"
        assert row["image_width"] == 1000
        assert row["image_height"] == 1000


def test_artifact_writer_scores_two_identical_objects_using_absolute_span_offsets(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    text = OBJECT_TEXT + OBJECT_TEXT
    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0, text=text)],
        decode_results={"row-1": _decode_result("row-1", token_trace=_repeated_trace())},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )

    pred = _read_jsonl(paths.scored_jsonl)[0]["pred"]

    assert len(pred) == 2
    assert [item["score"] for item in pred] == [pytest.approx(0.5), pytest.approx(0.25)]


def test_every_scored_prediction_has_row_local_source_version_and_finite_score(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )

    scored = _read_jsonl(paths.scored_jsonl)[0]["pred"][0]

    assert 0.0 <= scored["score"] <= 1.0
    assert math.isfinite(scored["score"])
    assert scored["pred_score_source"]["row_id"] == "row-1"
    assert scored["pred_score_source"]["selected_count"] == 8
    assert isinstance(scored["pred_score_version"], int)


def test_provenance_sidecar_binds_raw_and_scored_sha_and_score_policy(tmp_path: Path) -> None:
    from src.inference.artifacts import sha256_file, write_inference_artifacts
    from src.inference.scoring import SCORE_POLICY_FINGERPRINT

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )

    provenance = json.loads(paths.provenance_json.read_text(encoding="utf-8"))

    assert provenance["raw_artifact"]["sha256"] == sha256_file(paths.raw_jsonl)
    assert provenance["scored_artifact"]["sha256"] == sha256_file(paths.scored_jsonl)
    assert provenance["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT
    assert provenance["raw_artifact"]["path"] == "gt_vs_pred.jsonl"
    assert provenance["scored_artifact"]["path"] == "gt_vs_pred_scored.jsonl"
    assert provenance["row_binding"]["row_count"] == 1


def test_provenance_and_manifest_record_tokenizer_and_embedding_delta_identity_when_available(
    tmp_path: Path,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    metadata = {
        **_metadata(),
        "tokenizer_identity": {"tokenizer_sha256": "tok-fp"},
        "embedding_delta_identity": {
            "status": "loaded",
            "fingerprint": "embed-delta-fp",
        },
    }
    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=metadata,
    )

    provenance = json.loads(paths.provenance_json.read_text(encoding="utf-8"))
    manifest = json.loads(paths.run_manifest_json.read_text(encoding="utf-8"))

    assert provenance["tokenizer_identity"] == {"tokenizer_sha256": "tok-fp"}
    assert provenance["embedding_delta_identity"] == {
        "status": "loaded",
        "fingerprint": "embed-delta-fp",
    }
    assert manifest["tokenizer_identity"] == {"tokenizer_sha256": "tok-fp"}
    assert manifest["embedding_delta_identity"] == {
        "status": "loaded",
        "fingerprint": "embed-delta-fp",
    }


def test_provenance_manifest_and_summary_publish_backend_session_likelihoods(
    tmp_path: Path,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    metadata = {
        **_metadata(),
        "backend_session": {
            "backend": "hf",
            "backend_version": "test-transformers",
            "effective_settings": {"text_padding_side": "left"},
        },
        "likelihood_semantics": {
            "policy": "processed",
            "raw": "unprocessed",
            "score_owned_channel": "policy_logprob",
        },
        "execution_model_identity": {"fingerprint": "model-snapshot"},
        "frontend_identity": {"processor": "fake-qwen"},
        "raw_model_logprob_enabled": False,
    }
    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=metadata,
    )

    provenance = json.loads(paths.provenance_json.read_text(encoding="utf-8"))
    manifest = json.loads(paths.run_manifest_json.read_text(encoding="utf-8"))
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    for artifact in (provenance, manifest):
        assert artifact["backend_session"] == metadata["backend_session"]
        assert artifact["likelihood_semantics"] == metadata["likelihood_semantics"]
        assert artifact["execution_model_identity"] == {
            "fingerprint": "model-snapshot"
        }
        assert artifact["frontend_identity"] == {"processor": "fake-qwen"}
        assert artifact["raw_model_logprob_status"] == "disabled"
    assert summary["likelihood_semantics"] == metadata["likelihood_semantics"]
    assert summary["raw_model_logprob_status"] == "disabled"


def test_raw_likelihood_is_additive_and_policy_score_remains_authoritative(
    tmp_path: Path,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    policy_logprob = math.log(0.25)
    raw_logprob = math.log(0.75)
    trace = _trace(
        logprob=policy_logprob,
        raw_model_logprob=raw_logprob,
    )
    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1", token_trace=trace)},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata={**_metadata(), "raw_model_logprob_enabled": True},
    )

    generated = [
        row
        for row in _read_jsonl(paths.token_trace_jsonl)
        if row["trace_type"] == "generated_token"
    ]
    assert {row["raw_model_logprob_status"] for row in generated} == {"available"}
    assert {row["raw_model_logprob"] for row in generated} == {raw_logprob}
    assert {row["logprob"] for row in generated} == {policy_logprob}
    scored = _read_jsonl(paths.scored_jsonl)[0]["pred"][0]
    assert scored["score"] == pytest.approx(0.25)


@pytest.mark.parametrize("invalid_raw", [None, 0.1, float("nan")])
def test_raw_enabled_artifacts_reject_incomplete_or_invalid_token_evidence(
    tmp_path: Path,
    invalid_raw: float | None,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    trace = list(_trace(raw_model_logprob=math.log(0.75)))
    trace[4] = replace(
        trace[4],
        likelihood=LikelihoodPair(
            policy_logprob=trace[4].policy_logprob,
            raw_model_logprob=invalid_raw,
        ),
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("row-1", token_trace=trace)},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata={**_metadata(), "raw_model_logprob_enabled": True},
        )

    assert exc_info.value.code == "artifacts.decode_result_invalid"
    assert not (tmp_path / "pred_token_trace.jsonl").exists()
    assert not (tmp_path / "gt_vs_pred_scored.jsonl").exists()


def test_decode_result_requires_executed_media_identity() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        replace(_decode_result("row-1"), executed_media_sha256=None)

    assert exc_info.value.code == "backend_trace.invalid_result"
    assert exc_info.value.context["field"] == "executed_media_sha256"


def test_explicit_adapter_and_delta_provenance_is_recorded(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    metadata = {
        **_metadata(),
        "adapter_identity": {
            "status": "validated",
            "adapter_path": "checkpoints/step-5/adapter",
        },
        "embedding_delta_identity": {
            "status": "loaded",
            "identity": {"metadata_path": "checkpoints/step-5/special_token_embeddings"},
        },
    }

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=metadata,
    )

    provenance = json.loads(paths.provenance_json.read_text(encoding="utf-8"))
    manifest = json.loads(paths.run_manifest_json.read_text(encoding="utf-8"))
    assert provenance["adapter_identity"] == metadata["adapter_identity"]
    assert manifest["embedding_delta_identity"] == metadata["embedding_delta_identity"]


def test_trace_artifact_recomputes_stored_scores(tmp_path: Path) -> None:
    from src.inference.artifacts import recompute_scores_from_artifacts, write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )

    recomputed = recompute_scores_from_artifacts(
        scored_jsonl=paths.scored_jsonl,
        token_trace_jsonl=paths.token_trace_jsonl,
    )
    scored = _read_jsonl(paths.scored_jsonl)[0]["pred"][0]

    assert recomputed[("row-1", "row-1:span-0")] == pytest.approx(scored["score"])


def test_manifest_records_artifact_paths_without_claiming_wave5_benchmark_eligibility(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts
    from src.inference.scoring import SCORE_POLICY_FINGERPRINT

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )

    manifest = json.loads(paths.run_manifest_json.read_text(encoding="utf-8"))
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))

    assert manifest["trace_scoring_status"] == "scored"
    assert manifest["scored_artifact_materialized"] is True
    assert manifest["benchmark_eligible"] is False
    assert manifest["evaluator_consumer_status"] == "available_not_run"
    assert manifest["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT
    assert manifest["artifacts"]["gt_vs_pred_scored"] == "gt_vs_pred_scored.jsonl"
    assert summary["scored_artifact_materialized"] is True
    assert summary["benchmark_eligible"] is False
    assert summary["row_count"] == 1
    assert summary["scoreable_prediction_count"] == 1


def test_manifest_preserves_explicit_benchmark_eligibility(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    metadata = _metadata()
    metadata["benchmark_eligible"] = True
    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=metadata,
    )

    manifest = json.loads(paths.run_manifest_json.read_text(encoding="utf-8"))
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))

    assert manifest["benchmark_eligible"] is True
    assert summary["benchmark_eligible"] is True


def test_terminal_status_artifacts_do_not_publish_partial_summary_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.inference.artifacts as artifacts
    from src.inference.artifacts import write_terminal_status_artifacts

    real_replace = artifacts.os.replace
    replaced_final_names: list[str] = []

    def fail_before_manifest_replace(src: Path, dst: Path) -> None:
        dst_path = Path(dst)
        if dst_path.parent == tmp_path and dst_path.name == "summary.json":
            real_replace(src, dst)
            replaced_final_names.append(dst_path.name)
            return
        if replaced_final_names:
            raise OSError("forced terminal manifest publish failure")
        real_replace(src, dst)

    monkeypatch.setattr(artifacts.os, "replace", fail_before_manifest_replace)

    with pytest.raises(ArtifactContractError) as exc_info:
        write_terminal_status_artifacts(
            output_dir=tmp_path,
            metadata=_metadata(),
            summary={"terminal_status": "failed", "failure_class": "unit_failure"},
        )

    assert exc_info.value.code == "artifacts.terminal_publish_failed"
    assert replaced_final_names == ["summary.json"]
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "run_manifest.json").exists()


def test_artifact_writer_refuses_empty_image_plan_rows_before_status_claims(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("row-1")},
            image_plan_rows=[],
            metadata=_metadata(),
        )

    assert exc_info.value.code == "artifacts.image_plan_row_mismatch"


def test_artifact_writer_rejects_non_finite_generated_token_logprob_without_jsonl_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    bad_trace = list(_trace())
    bad_trace[4] = replace(
        bad_trace[4],
        likelihood=LikelihoodPair(
            policy_logprob=float("nan"),
            raw_model_logprob=None,
        ),
    )
    monkeypatch.setattr(
        DecodeResult,
        "validate_for_scored",
        lambda self, **kwargs: None,
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("row-1", token_trace=bad_trace)},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata=_metadata(),
        )

    assert exc_info.value.code == "artifacts.non_finite_trace_logprob"
    assert not (tmp_path / "pred_token_trace.jsonl").exists()
    assert not (tmp_path / "gt_vs_pred_scored.jsonl").exists()


def test_artifact_writer_preserves_prior_final_artifacts_when_rerun_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference.artifacts import write_inference_artifacts

    valid_paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )
    prior_scored = valid_paths.scored_jsonl.read_text(encoding="utf-8")
    prior_manifest = valid_paths.run_manifest_json.read_text(encoding="utf-8")
    bad_trace = list(_trace())
    bad_trace[4] = replace(
        bad_trace[4],
        likelihood=LikelihoodPair(
            policy_logprob=float("nan"),
            raw_model_logprob=None,
        ),
    )
    monkeypatch.setattr(
        DecodeResult,
        "validate_for_scored",
        lambda self, **kwargs: None,
    )

    with pytest.raises(ArtifactContractError):
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("row-1", token_trace=bad_trace)},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata=_metadata(),
        )

    assert valid_paths.scored_jsonl.read_text(encoding="utf-8") == prior_scored
    assert valid_paths.run_manifest_json.read_text(encoding="utf-8") == prior_manifest


def test_artifact_writer_rolls_back_all_final_artifacts_when_publish_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.inference.artifacts as artifacts
    from src.inference.artifacts import write_inference_artifacts

    valid_paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )
    final_paths = [
        valid_paths.raw_jsonl,
        valid_paths.scored_jsonl,
        valid_paths.provenance_json,
        valid_paths.token_trace_jsonl,
        valid_paths.parse_diagnostics_jsonl,
        valid_paths.image_plan_jsonl,
        valid_paths.summary_json,
        valid_paths.run_manifest_json,
    ]
    prior = {path: path.read_bytes() for path in final_paths}
    real_replace = artifacts.os.replace
    replaced_final_names: list[str] = []

    def fail_after_scored_replace(src: Path, dst: Path) -> None:
        dst_path = Path(dst)
        if dst_path.parent == tmp_path and dst_path.name == "gt_vs_pred_scored.jsonl":
            real_replace(src, dst)
            replaced_final_names.append(dst_path.name)
            return
        if replaced_final_names:
            raise OSError("forced replace failure after scored artifact")
        real_replace(src, dst)

    monkeypatch.setattr(artifacts.os, "replace", fail_after_scored_replace)

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("row-1")},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata=_metadata(),
        )

    assert exc_info.value.code == "artifacts.publish_failed"
    assert replaced_final_names == ["gt_vs_pred_scored.jsonl"]
    assert {path: path.read_bytes() for path in final_paths} == prior


def test_artifact_writer_rejects_decode_result_request_id_mismatch(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={"row-1": _decode_result("other-row")},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata=_metadata(),
        )

    assert exc_info.value.code == "artifacts.decode_result_row_mismatch"


def test_scored_artifact_production_refuses_missing_trace(tmp_path: Path) -> None:
    from src.inference.artifacts import write_inference_artifacts

    with pytest.raises(ArtifactContractError) as exc_info:
        write_inference_artifacts(
            output_dir=tmp_path,
            rows=[_raw_row("row-1", 0)],
            decode_results={},
            image_plan_rows=[_image_plan_row("row-1", 0)],
            metadata=_metadata(),
        )

    assert exc_info.value.code == "artifacts.missing_trace"


def test_scored_artifact_validation_refuses_missing_provenance(tmp_path: Path) -> None:
    from src.inference.artifacts import validate_scored_artifact_set, write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )
    paths.provenance_json.unlink()

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_scored_artifact_set(tmp_path)

    assert exc_info.value.code == "artifacts.missing_provenance"


def test_trace_recomputation_fails_when_selected_generated_token_row_is_missing(tmp_path: Path) -> None:
    from src.inference.artifacts import recompute_scores_from_artifacts, write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )
    rows = _read_jsonl(paths.token_trace_jsonl)
    rows = [
        row
        for row in rows
        if not (
            row.get("trace_type") == "generated_token"
            and row["row_id"] == "row-1"
            and row["generated_step_index"] == 4
        )
    ]
    _write_jsonl(paths.token_trace_jsonl, rows)

    with pytest.raises(ArtifactContractError) as exc_info:
        recompute_scores_from_artifacts(
            scored_jsonl=paths.scored_jsonl,
            token_trace_jsonl=paths.token_trace_jsonl,
        )

    assert exc_info.value.code == "artifacts.generated_trace_missing"


def test_trace_recomputation_fails_when_selected_generated_token_row_mismatches(tmp_path: Path) -> None:
    from src.inference.artifacts import recompute_scores_from_artifacts, write_inference_artifacts

    paths = write_inference_artifacts(
        output_dir=tmp_path,
        rows=[_raw_row("row-1", 0)],
        decode_results={"row-1": _decode_result("row-1")},
        image_plan_rows=[_image_plan_row("row-1", 0)],
        metadata=_metadata(),
    )
    rows = _read_jsonl(paths.token_trace_jsonl)
    for row in rows:
        if (
            row.get("trace_type") == "generated_token"
            and row["row_id"] == "row-1"
            and row["generated_step_index"] == 4
        ):
            row["token_text"] = "<|coord_999|>"
            break
    _write_jsonl(paths.token_trace_jsonl, rows)

    with pytest.raises(ArtifactContractError) as exc_info:
        recompute_scores_from_artifacts(
            scored_jsonl=paths.scored_jsonl,
            token_trace_jsonl=paths.token_trace_jsonl,
        )

    assert exc_info.value.code == "artifacts.generated_trace_mismatch"
