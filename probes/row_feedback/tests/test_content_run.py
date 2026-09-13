from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from probes.row_feedback import content
from probes.row_feedback.content_run import run_diagnostic


PACKET = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-row-feedback-pilot/content/packet-v2.json"
)


@dataclass(frozen=True)
class _Override:
    source: torch.Tensor
    visible_boundary_index: int


def _boundary(index: int, visible: int, source: torch.Tensor, *, override: bool) -> dict:
    receipt = content.tensor_receipt(source)
    return {
        "boundary_index": index,
        "visible_boundary_index": visible,
        "physical_slot_position": visible + index + 1,
        "source_shape": list(source.shape[-1:]),
        "source_dtype": str(source.dtype),
        "source_sha256": receipt["sha256"],
        "source_rms": receipt["rms"],
        "native_source_sha256": receipt["sha256"],
        "override_applied": override,
        "detach_applied": False,
    }


class _Runtime:
    FeedbackSourceOverride = _Override

    def __init__(self):
        self.replay_calls = 0
        self.generate_calls = 0
        self.bank_materialize_calls = 0

    def load_feedback_policy(self, *, adapter_path, device):
        return object(), object(), object(), {"adapter": str(adapter_path), "device": str(device)}

    def materialize_record(self, *_args, **_kwargs):
        raise AssertionError("content consumer must not use configured-source single-record materialization")

    def materialize_bank_records(self, _qwen, _frontend, _config, bank):
        self.bank_materialize_calls += 1
        return {
            record["record_id"]: {
                "inputs": {},
                "prompt_ids": list(record["prompt_token_ids"]),
                "prepared_inputs_sha256": "0" * 64,
                "materialization": {
                    "example_id": record["example_id"],
                    "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                    "image_sha256": record["image"]["image_sha256"],
                    "executed_media_sha256": record["image"]["executed_media_sha256"],
                    "raw_source_packet": dict(bank["sources"]["native_n16_training_input"]),
                },
            }
            for record in bank["records"]
        }

    def replay_visible(self, _qwen, _inputs, *, history_ids, target_ids, **_kwargs):
        assert torch.is_inference_mode_enabled()
        self.replay_calls += 1
        index = list(history_ids).count(content.BOX_END) + list(target_ids).count(content.BOX_END) - 1
        visible = len(history_ids) + len(target_ids) - 1
        source = torch.tensor([[[-1.0, 1.0]]])
        return {
            "feedback_sources": {index: source},
            "feedback_boundaries": [_boundary(index, visible, source, override=False)],
        }

    def generate_visible(self, _qwen, _inputs, *, history_ids, feedback_source_overrides=None, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.generate_calls += 1
        index = list(history_ids).count(content.BOX_END) - 1
        visible = len(history_ids) - 1
        native = torch.tensor([[[1.0, -1.0]]])
        overrides = feedback_source_overrides or {}
        used = overrides.get(index)
        ids = [1, 3] if used is not None and torch.equal(used.source, torch.tensor([[[-1.0, 1.0]]])) else [1, 2]
        return {
            "arm": "F",
            "visible_token_ids": ids,
            "text": str(ids),
            "finish_reason": "eos",
            "eos": True,
            "cap": False,
            "visible_generated_tokens": len(ids),
            "internal_slot_count": 1,
            "physical_token_count": len(history_ids) + 1,
            "model_forwards": 1,
            "image_forwards": 1,
            "slot_work": {"prefill": 0, "history": 1, "generated": 0, "total": 1},
            "feedback_boundaries": [_boundary(index, visible, native, override=used is not None)],
            "feedback_sources": {index: native} if kwargs.get("capture_feedback_sources") else {},
            "timing": {"wall_seconds": 0.0},
            "decode_contract": {"max_visible_tokens": kwargs["max_visible_tokens"]},
        }


def test_content_run_persists_fixed_three_receipts_without_training_graphs(tmp_path) -> None:
    output = tmp_path / "run"
    runtime = _Runtime()
    result = run_diagnostic(
        packet_path=PACKET,
        adapter_path=tmp_path / "trained-f-adapter",
        output=output,
        device=torch.device("cpu"),
        runtime_api=runtime,
    )

    assert result["status"] == "completed_non_gating_content_diagnostic"
    assert result["case_count"] == 3
    assert not result["technical_invalid_cases"]
    assert runtime.bank_materialize_calls == 1
    assert runtime.replay_calls == 3
    assert runtime.generate_calls == 9
    for binding in result["case_receipts"]:
        case_path = output / binding["path"]
        case = json.loads(case_path.read_text())
        assert len(binding["sha256"]) == 64
        assert case["non_gating"] is True
        assert case["materialized_input"]["prepared_inputs_sha256"] == "0" * 64
        assert case["materialized_input"]["materialization"]["example_id"].startswith("coco2017_train_")
        assert case["arms"]["exact_self_replay"]["exact_visible_equality_to_correct"] is True
        assert case["arms"]["exact_self_replay"]["override"]["count"] == 1
        assert case["arms"]["wrong_owner"]["override"]["count"] == 1
        assert case["mechanical_classification"]["status"] == "valid_visible_dependence"
        assert "not a causal deployable" in case["future_completion_caveat"]


def test_bound_cases_require_the_training_train_dev_union() -> None:
    from probes.dora_owner_learning.route_access import CONFIG
    from probes.parallel_owner_research.training import _load_materialization_raw
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples

    packet = json.loads(PACKET.read_text())
    bank_path = next(
        Path(source["path"])
        for source in packet["sources"]
        if source["role"] == "provenance_authority_and_literal_h_c_w_crosscheck"
    )
    bank = json.loads(bank_path.read_text())
    config = load_research_infer_config(CONFIG).config
    selected = {str(case["image"]["row_id"]) for case in packet["cases"]}
    configured = {str(raw.example_id) for raw in load_raw_examples(config.data.input_jsonl)}
    source = bank["sources"]["native_n16_training_input"]
    union = _load_materialization_raw(json.loads(Path(source["path"]).read_text()), config_input=config.data.input_jsonl)

    assert selected & configured == {"coco2017_train_000000417044"}
    assert selected <= set(union)
