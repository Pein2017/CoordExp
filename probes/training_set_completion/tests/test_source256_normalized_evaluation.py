from __future__ import annotations

import json
import sys
import types

import pytest

from probes.training_set_completion import source256_normalized_evaluation as evaluation
from probes.training_set_completion import training


def test_control_reuse_replays_all_prior_raw_shard_bindings() -> None:
    value = evaluation.admit_control_reuse()

    assert value["status"] == "admitted_exact_prior_controls"
    assert set(value["scores"]) == {"Source0", "A16", "A64", "B16", "B64"}
    assert sum(
        len(shards)
        for endpoint in value["admitted_shards"].values()
        for shards in endpoint.values()
    ) == 80
    assert value["predecessor"]["preparation"]["sha256"] == evaluation.PRIOR_PREPARATION_SHA256
    for score in value["scores"].values():
        for split in ("train", "dev"):
            observed = score["splits"][split]
            assert observed["confirmed_false_instance_count"] is None
            assert observed["physical_debt_status"] == "unresolved_no_endpoint_visual_review"


def test_control_reuse_rejects_the_registered_result_when_its_digest_is_not_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "PRIOR_RESULT_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="accepted predecessor result fixed digest"):
        evaluation.admit_control_reuse()


def test_actual_entry_qualification_uses_the_training_owner_receipt_shape(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_manifest = tmp_path / "qualification-manifest.json"
    qualification_manifest.write_text('{"kind": "qualification"}')
    qualification_terminal = tmp_path / "qualification-terminal.json"
    qualification_terminal.write_text("{}")
    main_manifest = tmp_path / "main-manifest.json"
    main_manifest.write_text('{"kind": "main"}')
    preparation = {"path": "/fixed/preparation.json", "sha256": "a" * 64, "size_bytes": 1}
    receipt = {
        "schema": "normalized.training.qualification",
        "status": "accepted_actual_entry",
        "unit_id": "unit",
        "arm": "B-normalized",
        "mode": "qualification",
        "qualification_manifest": training.binding(qualification_manifest),
        "training_terminal": training.binding(qualification_terminal),
        "preparation": preparation,
        "normalization": evaluation.NORMALIZATION_CONTRACT,
    }
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(json.dumps(receipt))

    class Runtime:
        SCHEMA = "normalized.training"

        @staticmethod
        def validate_training_manifest(value):
            if value == {"kind": "qualification"}:
                return {"mode": "qualification", "arm": "B-normalized"}
            if value == {"kind": "main"}:
                return {"mode": "main", "arm": "B-normalized"}
            raise AssertionError(value)

        @staticmethod
        def validate_qualification_receipt(value, *, manifest_path):
            assert manifest_path == qualification_manifest
            assert value == receipt
            return dict(value)

    class Readback:
        @staticmethod
        def validate_training_terminal(*, training_manifest_path, terminal_path):
            assert training_manifest_path == qualification_manifest
            assert terminal_path == qualification_terminal
            return {
                "manifest": {"mode": "qualification", "arm": "B-normalized"},
                "terminal": {"status": "completed", "mode": "qualification", "arm": "B-normalized"},
            }

    monkeypatch.setattr(evaluation, "_normalized_runtime", lambda: Runtime)
    monkeypatch.setattr(evaluation, "_normalized_readback", lambda: Readback)

    observed = evaluation._validate_actual_entry_qualification(
        qualification_path=qualification_path,
        main_manifest_path=main_manifest,
        preparation=preparation,
    )

    assert observed == receipt


def test_main_training_command_uses_runtime_run_and_bound_release_receipt(tmp_path) -> None:
    command = evaluation._training_command(
        tmp_path / "main-manifest.json",
        tmp_path / "training",
        tmp_path / "training-release.json",
    )

    assert command[command.index("--module") + 1] == "probes.training_set_completion.source256_normalized_training"
    assert command[command.index("--module") + 2] == "run"
    assert command[command.index("--release-receipt") + 1].endswith("training-release.json")


def test_report_surface_exposes_burden_unknowns_and_dev_retention() -> None:
    split = {
        "primary_class_agnostic_iou50": {
            "target_count": 10,
            "matched_count": 7,
            "missing_count": 3,
            "coverage": 0.7,
        },
        "class_consistent": {
            threshold: {
                "target_count": 10,
                "matched_count": 6,
                "missing_count": 4,
                "coverage": 0.6,
            }
            for threshold in ("0.5", "0.6", "0.8")
        },
        "burden": {
            "strict_repeat_row_count": 2,
            "malformed_row_count": 3,
            "cap_debt": 1,
            "eos_debt": 0,
            "invalid_geometry_count": 4,
            "annotation_unmatched_prediction_count": 5,
        },
        "confirmed_false_instance_count": None,
        "physical_debt_status": "unresolved_no_endpoint_visual_review",
    }
    scores = {
        "Bnormalized64": {
            "endpoint": {"label": "Bnormalized64", "arm": "B-normalized", "step": 64},
            "splits": {"train": split, "dev": split},
        }
    }
    changes = {
        "retained_count": 5,
        "gained_count": 2,
        "lost_count": 1,
    }
    report = evaluation._report_surface(
        scores=scores,
        comparisons={"Bnormalized64_vs_B64": {"splits": {"dev": {"primary_class_agnostic_iou50": changes}}}},
    )

    observed = report["endpoint_diagnostics"]["Bnormalized64"]["dev"]
    assert observed["repeat_proxy"]["strict_repeat_row_count"] == 2
    assert observed["malformed_row_count"] == 3
    assert observed["cap_eos_debt"] == {"cap_debt": 1, "eos_debt": 0}
    assert observed["geometry_debt"]["invalid_geometry_count"] == 4
    assert observed["annotation_unmatched"]["confirmed_false_instance_count"] is None
    assert report["dev_retention_vs_controls"]["Bnormalized64_vs_B64"] == changes


def test_offline_score_consumer_loads_tokenizer_from_the_bound_parent_directory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    calls = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            calls.append((path, local_files_only))
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(
        evaluation.predecessor_evaluation,
        "_validate_packet",
        lambda value: {
            "prepared": {
                "preparation": {
                    "identity": {"runtime_contract": {"tokenizer_path": str(tokenizer_path)}}
                }
            }
        },
    )
    monkeypatch.setattr(
        evaluation.predecessor_evaluation,
        "_targets",
        lambda prepared: {"train": {}, "dev": {}},
    )
    monkeypatch.setattr(
        evaluation.predecessor_evaluation,
        "_contexts",
        lambda prepared: {"train": {}, "dev": {}},
    )
    monkeypatch.setattr(
        evaluation,
        "_admit_new_rows",
        lambda *, checked, endpoint, split: ([{"image_id": 1}], [{"split": split}]),
    )
    monkeypatch.setattr(
        evaluation.predecessor_evaluation,
        "_score_split",
        lambda **kwargs: {"split": kwargs["split"], "image_count": 1},
    )

    observed = evaluation._score_new_endpoints(
        checked={
            "packet": {
                "endpoints": [
                    {"label": "Bnormalized16", "arm": "B-normalized", "step": 16}
                ]
            }
        },
        control={"predecessor": {"preparation": {"path": "/fixed/preparation.json"}}},
    )

    assert calls == [(str(tmp_path), True)]
    assert observed["Bnormalized16"]["endpoint"] == {
        "label": "Bnormalized16", "arm": "B-normalized", "step": 16
    }
    assert set(observed["Bnormalized16"]["splits"]) == {"train", "dev"}
