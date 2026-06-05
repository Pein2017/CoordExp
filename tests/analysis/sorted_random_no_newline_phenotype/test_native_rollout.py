from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.analysis.sorted_random_no_newline_phenotype import (
    PHASE_ID,
    POLICY_OBJECTIVE_RUN_IDS,
    PROJECT_ID,
    RUN_ID,
    SCHEMA_VERSION,
)
from src.analysis.sorted_random_no_newline_phenotype.config import (
    A32Config,
    CheckpointConfig,
    FNProbeConfig,
    PeakConfig,
    RolloutConfig,
    SamplingConfig,
    TemplateContractConfig,
)
from src.analysis.sorted_random_no_newline_phenotype.native_rollout import (
    decorate_native_rollout_artifacts,
    materialize_role_infer_config,
)
from src.analysis.sorted_random_no_newline_phenotype.status import (
    CONSTRAINT_POLICY,
    DECODE_POLICY,
    REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
)


def test_decorate_native_rollout_artifacts_adds_status_runtime_markers(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "rollout" / "role"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "summary.json").write_text(
        json.dumps(
            {
                "inference_provenance": {
                    "model_identity_fingerprint": "model:abc",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "gt_vs_pred.jsonl",
        [{"image_id": 1, "gt": [], "pred": []}],
    )
    _write_jsonl(
        rollout_dir / "pred_token_trace.jsonl",
        [
            {
                "line_idx": 0,
                "generated_token_text": ["<|im_end|>"],
                "token_logprobs": [-0.01],
            }
        ],
    )

    decorate_native_rollout_artifacts(
        rollout_dir,
        checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
        gpu_id="3",
        native_prompt_ordering="sorted",
        template_contract={
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
        },
    )

    summary = json.loads((rollout_dir / "summary.json").read_text(encoding="utf-8"))
    gt_row = json.loads((rollout_dir / "gt_vs_pred.jsonl").read_text(encoding="utf-8"))
    trace_row = json.loads(
        (rollout_dir / "pred_token_trace.jsonl").read_text(encoding="utf-8")
    )
    for row in (summary, gt_row, trace_row):
        assert row["runtime_kind"] == REAL_NATIVE_ROLLOUT_RUNTIME_KIND
        assert row["checkpoint_role"] == "fullobj_sorted_pure_ce_ckpt3668"
        assert row["decode_policy"] == DECODE_POLICY
        assert row["constraint_policy"] == CONSTRAINT_POLICY
        assert row["gpu_id"] == "3"
        assert row["checkpoint_fingerprint"] == "model:abc"
    assert gt_row["source_line_idx"] == 0
    assert trace_row["source_line_idx"] == 0
    assert trace_row["token_trace_sha256"]
    assert trace_row["raw_output_sha256"]
    assert summary["template_contract"]["row_separator"] == "none"


def test_materialized_native_rollout_config_enables_diagnostic_free_text_capture(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    config = A32Config(
        project_id=PROJECT_ID,
        phase_id=PHASE_ID,
        schema_version=SCHEMA_VERSION,
        run_id=RUN_ID,
        artifact_root=tmp_path / "artifacts",
        train_jsonl=tmp_path / "train.coord.jsonl",
        val_jsonl=tmp_path / "val.coord.jsonl",
        image_root=tmp_path / "images",
        checkpoints={
            "fullobj_random_pure_ce_ckpt3668": CheckpointConfig(
                checkpoint_path=checkpoint_path,
                training_ordering="random_permutation",
                readout_prompt_ordering="sorted",
            )
        },
        template_contract=TemplateContractConfig(
            detection_sequence_format="compact_full",
            coordinate_surface="coord_token",
            bbox_format="xyxy",
            row_separator="none",
        ),
        sampling=SamplingConfig(max_prefix_states=8, num_shards=8, seed=3668),
        rollout=RolloutConfig(limit_images=32),
        fn_probe=FNProbeConfig(),
        peak=PeakConfig(),
    )

    path = materialize_role_infer_config(
        config,
        checkpoint_role="fullobj_random_pure_ce_ckpt3668",
        gpu_id="0",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["infer"]["allow_diagnostic_gt_vs_pred"] is True
    assert raw["infer"]["row_separator"] == "none"
    assert raw["infer"]["generation"]["decode_mode"] == "greedy"
    assert raw["infer"]["generation"]["do_sample"] is False
    assert raw["infer"]["generation"]["num_beams"] == 1
    assert "compact_grammar" not in raw["infer"]["generation"]


def test_materialized_native_rollout_config_supports_et_rmp_role_by_ordering_template(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    config = A32Config(
        project_id=PROJECT_ID,
        phase_id=PHASE_ID,
        schema_version=SCHEMA_VERSION,
        run_id=POLICY_OBJECTIVE_RUN_IDS[0],
        artifact_root=tmp_path / "artifacts",
        train_jsonl=tmp_path / "train.coord.jsonl",
        val_jsonl=tmp_path / "val.coord.jsonl",
        image_root=tmp_path / "images",
        checkpoints={
            "fullobj_random_et_rmp_ce_ckpt3668": CheckpointConfig(
                checkpoint_path=checkpoint_path,
                training_ordering="random_permutation",
                readout_prompt_ordering="sorted",
                objective_policy="et_rmp_ce",
                template_contract_id="compact_full_no_newline_native_v1",
                comparison_group="fullobj_2x2_20260601",
            )
        },
        template_contract=TemplateContractConfig(
            detection_sequence_format="compact_full",
            coordinate_surface="coord_token",
            bbox_format="xyxy",
            row_separator="none",
        ),
        sampling=SamplingConfig(max_prefix_states=8, num_shards=8, seed=3668),
        rollout=RolloutConfig(limit_images=32),
        fn_probe=FNProbeConfig(),
        peak=PeakConfig(),
    )

    path = materialize_role_infer_config(
        config,
        checkpoint_role="fullobj_random_et_rmp_ce_ckpt3668",
        gpu_id="2",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["metadata"]["run_id"] == POLICY_OBJECTIVE_RUN_IDS[0]
    assert raw["run"]["name"] == (
        f"{POLICY_OBJECTIVE_RUN_IDS[0]}_fullobj_random_et_rmp_ce_ckpt3668"
    )
    assert raw["infer"]["model_checkpoint"] == str(checkpoint_path)
    assert raw["infer"]["object_ordering"] == "random"
    assert raw["infer"]["row_separator"] == "none"
    assert raw["infer"]["generation"]["decode_mode"] == "greedy"
    assert raw["a3_2_launch_spec"]["checkpoint_role"] == (
        "fullobj_random_et_rmp_ce_ckpt3668"
    )
    assert raw["a3_2_launch_spec"]["template_contract"]["row_separator"] == "none"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
