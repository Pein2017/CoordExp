from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.infer.artifacts import (
    build_infer_resolved_meta,
    build_infer_summary_payload,
)


class _Counters:
    def to_summary(self) -> dict[str, int]:
        return {"count": 0}


def _owner() -> SimpleNamespace:
    cfg = SimpleNamespace(
        model_checkpoint="model",
        adapter_checkpoint=None,
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
        gt_jsonl="gt.jsonl",
        pred_coord_mode="auto",
        device="cpu",
        limit=200,
        distributed_enabled=False,
    )
    gen_cfg = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
        stop_pressure_mode=None,
        stop_pressure_min_new_tokens=0,
        stop_pressure_trigger_rule=None,
        stop_pressure_logit_bias=0.0,
        stop_pressure_active=False,
        compact_grammar_enabled=True,
        compact_grammar_format="compact_full",
        compact_grammar_force_row_start=True,
    )
    return SimpleNamespace(
        cfg=cfg,
        gen_cfg=gen_cfg,
        resolved_mode="text",
        requested_mode="text",
        mode_reason="requested",
        prompt_variant="coco_80",
        bbox_format="xyxy",
        detection_sequence_format="compact_full",
        object_field_order="compact_full_row",
        object_ordering="original",
        prompt_template_hash="0" * 64,
        attn_implementation_requested=None,
        attn_implementation_selected=None,
    )


def test_infer_artifacts_record_compact_grammar_decode_provenance() -> None:
    owner = _owner()

    resolved = build_infer_resolved_meta(
        owner=owner,
        backend="hf",
        batch_size=1,
        out_path=Path("gt_vs_pred.jsonl"),
        summary_path=Path("summary.json"),
        trace_path=Path("pred_token_trace.jsonl"),
    )
    summary = build_infer_summary_payload(
        owner=owner,
        counters=_Counters(),
        backend="hf",
        determinism="deterministic",
        batch_size=1,
    )

    assert resolved["detection_sequence_format"] == "compact_full"
    assert resolved["generation"]["compact_grammar"] == {
        "enabled": True,
        "format": "compact_full",
        "force_row_start": True,
        "active": True,
    }
    assert summary["infer"]["detection_sequence_format"] == "compact_full"
    assert summary["generation"]["compact_grammar"] == {
        "enabled": True,
        "format": "compact_full",
        "force_row_start": True,
        "active": True,
    }
